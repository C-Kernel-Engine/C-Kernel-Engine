#!/usr/bin/env python3
"""
build_ir_train_v7.py

Build IR1 (train-forward) for v7:
- classify inputs as weight vs activation from manifest truth
- select training forward kernels (fp32-first)
- derive save_for_backward from grad rules
- emit tensor registry with explicit kinds and producers
- keep IR as source-of-truth: avoid hard-coded model-specific shape logic in codegen

This is intentionally standalone from build_ir_v7.py to avoid inference regressions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import build_ir_v7


SCRIPT_DIR = Path(__file__).resolve().parent
V7_ROOT = SCRIPT_DIR.parent
KERNEL_REGISTRY_PATH = V7_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"
KERNEL_BINDINGS_PATH = V7_ROOT / "kernel_maps" / "kernel_bindings.json"
DEFAULT_GRAD_RULES_PATH = SCRIPT_DIR / "grad_rules_v7.json"
TEMPLATES_DIR = V7_ROOT / "templates"


# NOTE: some forward kernels below are in-place at C API level (currently qk_norm_forward
# and rope_forward_qk). IR1 intentionally remains out-of-place (explicit input/output tensors)
# for deterministic graph semantics. Codegen must stage/copy before invoking those kernels.
# Training maps activation ops to exact forward kernels so the train runtime uses the
# same stable math as the backward path. Fast approximations remain available to inference.
FORWARD_KERNEL_BY_OP = {
    "dense_embedding_lookup": "dense_embedding_lookup",
    "rmsnorm": "rmsnorm_forward",
    "q_proj": "gemm_blocked_serial",
    "q_gate_proj": "gemm_blocked_serial",
    "k_proj": "gemm_blocked_serial",
    "v_proj": "gemm_blocked_serial",
    "recurrent_qkv_proj": "gemm_blocked_serial",
    "recurrent_gate_proj": "gemm_blocked_serial",
    "recurrent_alpha_proj": "gemm_blocked_serial",
    "recurrent_beta_proj": "gemm_blocked_serial",
    "recurrent_out_proj": "gemm_blocked_serial",
    "qk_norm": "qk_norm_forward",
    "rope_qk": "rope_forward_qk",
    "split_q_gate": "split_q_gate_forward",
    "recurrent_split_qkv": "recurrent_split_qkv_forward",
    "recurrent_dt_gate": "recurrent_dt_gate_forward",
    "recurrent_conv_state_update": "recurrent_conv_state_update_forward",
    "recurrent_ssm_conv": "ssm_conv1d_forward",
    "recurrent_silu": "recurrent_silu_forward",
    "recurrent_split_conv_qkv": "recurrent_split_conv_qkv_forward",
    "recurrent_qk_l2_norm": "recurrent_qk_l2_norm_forward",
    "recurrent_core": "gated_deltanet_autoregressive_forward",
    "recurrent_norm_gate": "recurrent_norm_gate_forward",
    "attn": "attention_forward_causal_head_major_gqa_flash_strided",
    "attn_sliding": "attention_forward_causal_head_major_gqa_flash_strided_sliding",
    "attn_gate_sigmoid_mul": "attn_gate_sigmoid_mul_forward",
    "out_proj": "gemm_blocked_serial",
    "residual_add": "ck_residual_add_token_major",
    "mlp_gate_up": "gemm_blocked_serial",
    "silu_mul": "swiglu_forward_exact",
    "geglu": "geglu_forward_exact",
    "mlp_down": "gemm_blocked_serial",
    "logits": "gemm_blocked_serial",
}

FORWARD_BRIDGE_PLAN_SPECS: Dict[str, Dict[str, Any]] = {
    "qk_norm": {
        "kernel_layout": "head_major",
        "tensor_layout": "token_major",
        "pre_roles": (
            {"role": "q", "input_key": "q", "tmp_role": "q_pre", "head_group": "num_heads"},
            {"role": "k", "input_key": "k", "tmp_role": "k_pre", "head_group": "num_kv_heads"},
        ),
        "post_roles": (
            {"role": "q", "output_key": "q", "head_group": "num_heads", "source": "pre:q"},
            {"role": "k", "output_key": "k", "head_group": "num_kv_heads", "source": "pre:k"},
        ),
    },
    "rope_qk": {
        "kernel_layout": "head_major",
        "tensor_layout": "token_major",
        "pre_roles": (
            {"role": "q", "input_key": "q", "tmp_role": "q_pre", "head_group": "num_heads"},
            {"role": "k", "input_key": "k", "tmp_role": "k_pre", "head_group": "num_kv_heads"},
        ),
        "post_roles": (
            {"role": "q", "output_key": "q", "head_group": "num_heads", "source": "pre:q"},
            {"role": "k", "output_key": "k", "head_group": "num_kv_heads", "source": "pre:k"},
        ),
    },
    "attn": {
        "kernel_layout": "head_major",
        "tensor_layout": "token_major",
        "pre_roles": (
            {"role": "q", "input_key": "q", "tmp_role": "q_pre", "head_group": "num_heads"},
            {"role": "k", "input_key": "k", "tmp_role": "k_pre", "head_group": "num_kv_heads"},
            {"role": "v", "input_key": "v", "tmp_role": "v_pre", "head_group": "num_kv_heads"},
        ),
        "kernel_output_roles": (
            {"name": "out", "tmp_role": "out_post", "head_group": "num_heads"},
        ),
        "post_roles": (
            {"role": "out", "output_key": "out", "head_group": "num_heads", "source": "kernel:out"},
        ),
    },
    "attn_sliding": {
        "kernel_layout": "head_major",
        "tensor_layout": "token_major",
        "pre_roles": (
            {"role": "q", "input_key": "q", "tmp_role": "q_pre", "head_group": "num_heads"},
            {"role": "k", "input_key": "k", "tmp_role": "k_pre", "head_group": "num_kv_heads"},
            {"role": "v", "input_key": "v", "tmp_role": "v_pre", "head_group": "num_kv_heads"},
        ),
        "kernel_output_roles": (
            {"name": "out", "tmp_role": "out_post", "head_group": "num_heads"},
        ),
        "post_roles": (
            {"role": "out", "output_key": "out", "head_group": "num_heads", "source": "kernel:out"},
        ),
    },
}


WEIGHT_PATTERNS = dict(build_ir_v7.WEIGHT_PATTERNS)


WEIGHTS_BY_LOGICAL_OP = {
    "dense_embedding_lookup": [("weight", "token_emb", True)],
    "rmsnorm": [("gamma", "ln1_gamma", True)],  # remapped to ln2/final per context
    "q_proj": [("W", "wq", True), ("bias", "bq", False)],
    "q_gate_proj": [("W", "wq", True), ("bias", "bq", False)],
    "k_proj": [("W", "wk", True), ("bias", "bk", False)],
    "v_proj": [("W", "wv", True), ("bias", "bv", False)],
    "recurrent_qkv_proj": [("W", "attn_qkv", True)],
    "recurrent_gate_proj": [("W", "attn_gate", True)],
    "recurrent_alpha_proj": [("W", "ssm_alpha", True)],
    "recurrent_beta_proj": [("W", "ssm_beta", True)],
    "recurrent_dt_gate": [("dt_bias", "ssm_dt_bias", True), ("A", "ssm_a", True)],
    "recurrent_ssm_conv": [("kernel", "ssm_conv1d", True)],
    "recurrent_norm_gate": [("weight", "ssm_norm", True)],
    "recurrent_out_proj": [("W", "ssm_out", True)],
    "qk_norm": [("q_gamma", "q_norm", False), ("k_gamma", "k_norm", False)],
    "out_proj": [("W", "wo", True), ("bias", "bo", False)],
    "mlp_gate_up": [("W", "w1", True), ("bias", "b1", False), ("W_aux", "w3", False)],
    "mlp_down": [("W", "w2", True), ("bias", "b2", False)],
    "logits": [("W", "output_weight", False), ("W_tied", "token_emb", False)],
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _template_sections(template: Dict[str, Any]) -> Tuple[List[str], Any, List[str]]:
    sequence = template.get("sequence", [])
    if not sequence:
        raise RuntimeError("Template missing `sequence`")
    block_name = sequence[0]
    block = template.get("block_types", {}).get(block_name, {})
    header = list(block.get("header", []))
    body_def = block.get("body", {})
    body = body_def if isinstance(body_def, dict) else list(body_def or [])
    footer = list(block.get("footer", []))
    return header, body, footer


def _manifest_weight_index(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    # Manifest entries are the source of truth for persistent weight tensors.
    # IR1 should not infer trainable params from template names alone.
    out = {}
    for entry in manifest.get("entries", []):
        name = entry.get("name")
        if isinstance(name, str) and name:
            out[name] = entry
    return out


def _resolve_weight_name(weight_index: Dict[str, Dict[str, Any]], key: str, layer: Optional[int]) -> Optional[str]:
    patterns = WEIGHT_PATTERNS.get(key, [key])
    for pattern in patterns:
        candidate = pattern
        if layer is not None:
            candidate = candidate.replace("{L}", str(layer))
        if candidate in weight_index:
            return candidate
    return None


def _is_trainable_dtype(dtype: str) -> bool:
    return str(dtype).lower() in ("fp32", "bf16")


def _shape_numel(shape: Any) -> Optional[int]:
    if not isinstance(shape, list) or not shape:
        return None
    n = 1
    for d in shape:
        if not isinstance(d, int) or d <= 0:
            return None
        n *= d
    return int(n)


def _entry_numel(entry: Dict[str, Any]) -> Optional[int]:
    n = _shape_numel(entry.get("shape"))
    if isinstance(n, int) and n > 0:
        return n
    size = entry.get("size")
    dtype = str(entry.get("dtype", "")).lower()
    if not isinstance(size, int) or size <= 0:
        return None
    if dtype in ("fp32", "f32", "int32", "i32") and (size % 4 == 0):
        return int(size // 4)
    if dtype in ("bf16", "bfloat16") and (size % 2 == 0):
        return int(size // 2)
    return None


def _cfg_int(config: Dict[str, Any], keys: List[str], default: int) -> int:
    for k in keys:
        if k in config:
            try:
                v = int(config.get(k))
                if v > 0:
                    return v
            except Exception:
                pass
    return int(default)


def _train_dims(config: Dict[str, Any]) -> Dict[str, int]:
    d_model = _cfg_int(config, ["embed_dim", "hidden_size", "d_model"], 128)
    hidden = _cfg_int(config, ["intermediate_size", "hidden_dim"], max(2 * d_model, d_model))
    vocab = _cfg_int(config, ["vocab_size"], 256)
    num_heads = _cfg_int(config, ["num_heads"], 1)
    num_kv_heads = _cfg_int(config, ["num_kv_heads"], num_heads)
    head_dim = _cfg_int(config, ["head_dim"], max(1, d_model // max(1, num_heads)))
    aligned_head_dim = _cfg_int(config, ["aligned_head_dim"], head_dim)
    attn_out_dim = _cfg_int(config, ["attn_out_dim"], max(1, num_heads * head_dim))
    q_gate_proj_dim = _cfg_int(config, ["q_gate_proj_dim", "attn_q_gate_proj_dim"], max(1, 2 * attn_out_dim))
    attn_gate_dim = _cfg_int(config, ["attn_gate_dim"], max(1, q_gate_proj_dim - attn_out_dim))
    q_dim = _cfg_int(config, ["q_dim"], max(1, num_heads * head_dim))
    k_dim = _cfg_int(config, ["k_dim"], q_dim)
    v_dim = _cfg_int(config, ["v_dim"], hidden)
    gate_dim = _cfg_int(config, ["gate_dim", "ssm_time_step_rank"], max(1, num_heads))
    recurrent_head_dim = _cfg_int(config, ["recurrent_head_dim"], max(1, v_dim // max(1, gate_dim)))
    ssm_conv_kernel = _cfg_int(config, ["ssm_conv_kernel"], 4)
    ssm_conv_history = _cfg_int(config, ["ssm_conv_history"], max(0, ssm_conv_kernel - 1))
    ssm_conv_channels = _cfg_int(config, ["ssm_conv_channels"], max(1, q_dim + k_dim + v_dim))
    # Runtime token count is compile-time today; default 1 for legacy parity.
    token_count = _cfg_int(config, ["train_tokens", "tokens", "seq_len"], 1)
    kv_dim = max(1, num_kv_heads * head_dim)
    gate_up_dim = max(1, 2 * hidden)
    return {
        "tokens": token_count,
        "d_model": d_model,
        "hidden": hidden,
        "vocab": vocab,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "aligned_head_dim": aligned_head_dim,
        "attn_out_dim": attn_out_dim,
        "q_gate_proj_dim": q_gate_proj_dim,
        "attn_gate_dim": attn_gate_dim,
        "q_dim": q_dim,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "gate_dim": gate_dim,
        "recurrent_head_dim": recurrent_head_dim,
        "ssm_conv_kernel": ssm_conv_kernel,
        "ssm_conv_history": ssm_conv_history,
        "ssm_conv_channels": ssm_conv_channels,
        "kv_dim": kv_dim,
        "gate_up_dim": gate_up_dim,
    }


def _infer_output_shape_numel(logical_op: str, out_name: str, config: Dict[str, Any]) -> Tuple[List[int], int]:
    dims = _train_dims(config)
    t = dims["tokens"]
    d_model = dims["d_model"]
    hidden = dims["hidden"]
    vocab = dims["vocab"]
    attn_out_dim = dims["attn_out_dim"]
    q_gate_proj_dim = dims["q_gate_proj_dim"]
    attn_gate_dim = dims["attn_gate_dim"]
    q_dim = dims["q_dim"]
    k_dim = dims["k_dim"]
    v_dim = dims["v_dim"]
    gate_dim = dims["gate_dim"]
    recurrent_head_dim = dims["recurrent_head_dim"]
    ssm_conv_history = dims["ssm_conv_history"]
    ssm_conv_channels = dims["ssm_conv_channels"]
    kv_dim = dims["kv_dim"]
    gate_up_dim = dims["gate_up_dim"]

    if out_name == "rstd_cache":
        return [t], t

    if logical_op == "dense_embedding_lookup":
        return [t, d_model], t * d_model
    if logical_op in ("rmsnorm", "out_proj", "residual_add", "mlp_down"):
        return [t, d_model], t * d_model
    if logical_op == "q_proj":
        return [t, q_dim], t * q_dim
    if logical_op in ("k_proj", "v_proj"):
        return [t, kv_dim], t * kv_dim
    if logical_op in ("qk_norm", "rope_qk"):
        dim = kv_dim if out_name == "k" else q_dim
        return [t, dim], t * dim
    if logical_op == "q_gate_proj":
        return [t, q_gate_proj_dim], t * q_gate_proj_dim
    if logical_op == "split_q_gate":
        dim = attn_gate_dim if out_name == "gate" else attn_out_dim
        return [t, dim], t * dim
    if logical_op in ("attn", "attn_sliding"):
        return [t, attn_out_dim], t * attn_out_dim
    if logical_op == "attn_gate_sigmoid_mul":
        return [t, attn_out_dim], t * attn_out_dim
    if logical_op == "recurrent_qkv_proj":
        packed = q_dim + k_dim + v_dim
        return [t, packed], t * packed
    if logical_op == "recurrent_gate_proj":
        return [t, v_dim], t * v_dim
    if logical_op in ("recurrent_alpha_proj", "recurrent_beta_proj", "recurrent_dt_gate"):
        return [t, gate_dim], t * gate_dim
    if logical_op == "recurrent_split_qkv":
        dim = q_dim if out_name == "q" else (k_dim if out_name == "k" else v_dim)
        return [t, dim], t * dim
    if logical_op == "recurrent_conv_state_update":
        if out_name == "conv_x":
            numel = ssm_conv_channels * max(1, t + ssm_conv_history)
            return [1, ssm_conv_channels, max(1, t + ssm_conv_history)], numel
        numel = ssm_conv_channels * max(1, ssm_conv_history)
        return [max(1, ssm_conv_history), ssm_conv_channels], numel
    if logical_op in ("recurrent_ssm_conv", "recurrent_silu"):
        return [t, ssm_conv_channels], t * ssm_conv_channels
    if logical_op == "recurrent_split_conv_qkv":
        dim = q_dim if out_name == "q" else (k_dim if out_name == "k" else v_dim)
        return [t, dim], t * dim
    if logical_op == "recurrent_qk_l2_norm":
        dim = q_dim if out_name == "q" else k_dim
        return [t, dim], t * dim
    if logical_op == "recurrent_core":
        if out_name == "state_out":
            state_shape = list(build_ir_v7._recurrent_state_shape(config))
            return state_shape, int(state_shape[0] * state_shape[1] * state_shape[2])
        return [t, v_dim], t * v_dim
    if logical_op == "recurrent_norm_gate":
        return [t, v_dim], t * v_dim
    if logical_op == "mlp_gate_up":
        return [t, gate_up_dim], t * gate_up_dim
    if logical_op in ("silu_mul", "geglu"):
        return [t, hidden], t * hidden
    if logical_op == "logits":
        return [t, vocab], t * vocab
    return [t, d_model], t * d_model


def _infer_saved_shape_numel(saved_key: str, config: Dict[str, Any]) -> Tuple[List[int], int]:
    dims = _train_dims(config)
    t = dims["tokens"]
    key = str(saved_key).lower()
    if key in ("rstd", "rrms"):
        return [t], t
    if key in ("lse",):
        n = max(1, dims["num_heads"] * t)
        return [n], n
    if key in ("attn_weights",):
        n = max(1, dims["num_heads"] * t * t)
        return [dims["num_heads"], t, t], n
    return [t, dims["d_model"]], max(1, t * dims["d_model"])


def _infer_head_major_bridge_shape_numel(config: Dict[str, Any], head_group: str) -> Tuple[List[int], int]:
    dims = _train_dims(config)
    tokens = max(1, int(dims["tokens"]))
    aligned_head_dim = max(1, int(dims.get("aligned_head_dim", dims["head_dim"])))
    group = str(head_group or "num_heads").strip().lower()
    if group in ("num_kv_heads", "kv", "k", "v"):
        heads = max(1, int(dims["num_kv_heads"]))
    else:
        heads = max(1, int(dims["num_heads"]))
    numel = max(1, tokens * heads * aligned_head_dim)
    return [heads, tokens, aligned_head_dim], numel


def _infer_external_input_shape_numel(
    tensor_id: Optional[str],
    external_from: Optional[str],
    config: Dict[str, Any],
) -> Tuple[Optional[List[int]], Optional[int]]:
    ref = str(external_from or tensor_id or "").strip().lower()
    dims = _train_dims(config)
    if ref.endswith("recurrent_conv_state"):
        shape = [max(1, dims["ssm_conv_history"]), dims["ssm_conv_channels"]]
        return shape, int(shape[0] * shape[1])
    if ref.endswith("recurrent_ssm_state"):
        shape = list(build_ir_v7._recurrent_state_shape(config))
        return shape, int(shape[0] * shape[1] * shape[2])
    return None, None


def _kernel_ids(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for kernel in registry.get("kernels", []):
        kid = kernel.get("id")
        if isinstance(kid, str):
            out[kid] = kernel
    return out


def _binding_ids(bindings_doc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(bindings_doc.get("bindings", {}))


def _op_family_grad_rule(grad_rules: Dict[str, Any], op_name: str) -> Optional[str]:
    return grad_rules.get("template_op_to_rule", {}).get(op_name)


def _op_base_name(op_name: str, layer: int, instance: int, section: str) -> str:
    if layer < 0:
        return "S%s.%s.%d" % (section, op_name, instance)
    return "L%d.%s.%d" % (layer, op_name, instance)


def _weight_key_override(op_name: str, alias: str, layer: int, rmsnorm_idx: int, section: str, norm_variant: Optional[str] = None) -> str:
    # Route rmsnorm aliases by position.
    if op_name != "rmsnorm":
        return alias
    variant = str(norm_variant or "rmsnorm")
    if section == "footer":
        if alias in ("gamma", "ln1_gamma", "ln2_gamma"):
            return "final_ln_weight"
        return alias
    if alias in ("gamma", "ln1_gamma", "ln2_gamma"):
        if variant == "attn_norm":
            return "ln1_gamma"
        if variant == "ffn_norm":
            return "ln2_gamma"
        if variant == "post_attention_norm":
            return "post_attention_norm"
        if variant == "post_ffn_norm":
            return "post_ffn_norm"
        return "ln1_gamma" if rmsnorm_idx == 0 else "ln2_gamma"
    return alias


def _choose_template_with_source(
    manifest: Dict[str, Any],
    explicit_template: Optional[Path],
) -> Tuple[Dict[str, Any], str]:
    if isinstance(manifest.get("template"), dict) and manifest.get("template"):
        return manifest["template"], "manifest.template"
    if explicit_template is not None:
        return _load_json(explicit_template), "repo.template"
    model = str((manifest.get("config") or {}).get("model", "qwen3")).lower()
    candidate = TEMPLATES_DIR / ("%s.json" % model)
    if candidate.exists():
        return _load_json(candidate), "repo.template"
    raise RuntimeError("No template found in manifest and no template file for model=%s" % model)


def _choose_template(manifest: Dict[str, Any], explicit_template: Optional[Path]) -> Dict[str, Any]:
    template, _template_source = _choose_template_with_source(manifest, explicit_template)
    return template


def _make_saved_tensor_id(op_id: int, key: str) -> str:
    return "saved.op%d.%s" % (op_id, key)


def _resolve_forward_kernels(config: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, str]:
    resolved = dict(FORWARD_KERNEL_BY_OP)
    template_kernels = dict(template.get("kernels") or {})
    resolved["rope_qk"] = build_ir_v7._resolve_rope_qk_kernel(config, template_kernels)
    return resolved


def _normalize_train_attention_runtime_contract(raw_train_contract: Any) -> Dict[str, Any]:
    out = dict(raw_train_contract) if isinstance(raw_train_contract, dict) else {}
    saved_overrides = out.get("saved_tensor_kernel_overrides")
    if isinstance(saved_overrides, dict):
        saved_overrides = {
            str(key).strip(): str(value).strip()
            for key, value in saved_overrides.items()
            if str(key).strip() and str(value).strip()
        }
    else:
        saved_overrides = {}
    out["saved_tensor_kernel_overrides"] = saved_overrides
    zero_saved = out.get("requires_zero_sliding_window_for_saved_tensors")
    if isinstance(zero_saved, list):
        zero_saved = [str(item).strip() for item in zero_saved if str(item).strip()]
    else:
        zero_saved = []
    out["requires_zero_sliding_window_for_saved_tensors"] = zero_saved
    return out


def _default_train_attention_runtime_contract(attn_variant: str) -> Dict[str, Any]:
    out = {
        "saved_tensor_kernel_overrides": {
            "attn_weights": "attention_forward_causal_head_major_gqa_exact"
        },
        "requires_zero_sliding_window_for_saved_tensors": [],
    }
    if attn_variant == "sliding_window":
        out["requires_zero_sliding_window_for_saved_tensors"] = ["attn_weights"]
    return out


def _template_attention_runtime_contract(
    template: Dict[str, Any],
    *,
    template_source: str,
) -> Tuple[Dict[str, Any], Optional[str]]:
    contract = ((template.get("contract") or {}).get("attention_contract") or {})
    if not isinstance(contract, dict):
        contract = {}
    template_name = str(template.get("name") or "unknown")
    attn_variant = str(contract.get("attn_variant", "") or "").strip().lower()
    out = _normalize_train_attention_runtime_contract(contract.get("train_runtime_contract"))
    saved_overrides = out["saved_tensor_kernel_overrides"]
    zero_saved = out["requires_zero_sliding_window_for_saved_tensors"]
    missing_fields: List[str] = []
    if not str(saved_overrides.get("attn_weights") or "").strip():
        missing_fields.append("saved_tensor_kernel_overrides.attn_weights")
    if attn_variant == "sliding_window" and "attn_weights" not in set(zero_saved):
        missing_fields.append("requires_zero_sliding_window_for_saved_tensors.attn_weights")
    if not missing_fields:
        return out, None

    if template_source == "manifest.template":
        fallback = _default_train_attention_runtime_contract(attn_variant)
        merged_saved = dict(fallback["saved_tensor_kernel_overrides"])
        merged_saved.update(saved_overrides)
        merged_zero = list(zero_saved)
        for item in fallback["requires_zero_sliding_window_for_saved_tensors"]:
            if item not in merged_zero:
                merged_zero.append(item)
        out["saved_tensor_kernel_overrides"] = merged_saved
        out["requires_zero_sliding_window_for_saved_tensors"] = merged_zero
        warning = (
            "Embedded manifest template `%s` is missing training attention runtime contract "
            "fields (%s); applied compatibility defaults. Regenerate the run manifest so "
            "the template carries explicit train_runtime_contract truth."
        ) % (template_name, ", ".join(missing_fields))
        return out, warning

    raise RuntimeError(
        "Repo template `%s` must define contract.attention_contract.train_runtime_contract "
        "fields: %s"
        % (template_name, ", ".join(missing_fields))
    )


def _resolve_train_config(manifest_config: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
    config = build_ir_v7._normalize_manifest_config(dict(manifest_config or {}))
    rope_layout = build_ir_v7._normalize_rope_layout_value(config.get("rope_layout"))
    if rope_layout:
        config["rope_layout"] = rope_layout
        return config

    attention_contract = ((template.get("contract") or {}).get("attention_contract") or {})
    template_rope_layout = build_ir_v7._normalize_rope_layout_value(attention_contract.get("rope_layout"))
    if template_rope_layout:
        config["rope_layout"] = template_rope_layout
    return config


def build_ir1_train(
    manifest: Dict[str, Any],
    registry: Dict[str, Any],
    bindings_doc: Dict[str, Any],
    grad_rules: Dict[str, Any],
    max_layers: Optional[int],
    strict: bool,
    bridge_lowering: str = "legacy",
) -> Dict[str, Any]:
    template, template_source = _choose_template_with_source(manifest, explicit_template=None)
    config = _resolve_train_config(manifest.get("config", {}), template)
    bridge_lowering = str(bridge_lowering or "legacy").strip().lower()
    if bridge_lowering not in {"legacy", "explicit"}:
        raise RuntimeError("Unsupported bridge_lowering=%r (expected legacy|explicit)" % bridge_lowering)
    num_layers = int(config.get("num_layers", 0) or 0)
    if max_layers is not None:
        num_layers = min(num_layers, int(max_layers))
    if num_layers <= 0:
        raise RuntimeError("Invalid num_layers in manifest/config")

    token_count = int(_train_dims(config).get("tokens", 1) or 1)

    header_ops, body_def, footer_ops = _template_sections(template)
    forward_kernels = _resolve_forward_kernels(config, template)
    weight_index = _manifest_weight_index(manifest)
    kernels = _kernel_ids(registry)
    binding_ids = _binding_ids(bindings_doc)

    ops: List[Dict[str, Any]] = []
    tensors: Dict[str, Dict[str, Any]] = {}
    issues: List[str] = []
    warnings: List[str] = []
    bridge_enabled = bridge_lowering == "explicit"
    train_attention_runtime_contract, train_attention_contract_warning = _template_attention_runtime_contract(
        template,
        template_source=template_source,
    )
    if train_attention_contract_warning:
        warnings.append(train_attention_contract_warning)

    op_id = 0
    instance_counter: Dict[str, int] = {}

    # External inputs.
    tensors["input.token_ids"] = {
        "dtype": "int32",
        "kind": "input",
        "requires_grad": False,
        "persistent": False,
        "producer": None,
        "shape": [token_count],
        "numel": token_count,
    }
    tensors["input.targets"] = {
        "dtype": "int32",
        "kind": "input",
        "requires_grad": False,
        "persistent": False,
        "producer": None,
        "shape": [token_count],
        "numel": token_count,
    }

    def next_instance(op_name: str, layer: int, section: str) -> int:
        key = "%s:%d:%s" % (section, layer, op_name)
        n = instance_counter.get(key, 0)
        instance_counter[key] = n + 1
        return n

    def ensure_tensor(
        tensor_id: str,
        dtype: str,
        kind: str,
        requires_grad: bool,
        persistent: bool,
        producer: Optional[Dict[str, Any]],
        shape: Optional[List[int]] = None,
        numel: Optional[int] = None,
    ) -> None:
        if tensor_id in tensors:
            cur = tensors[tensor_id]
            cur_shape = cur.get("shape")
            if shape is not None and (not isinstance(cur_shape, list) or not cur_shape):
                cur["shape"] = list(shape)
            if isinstance(numel, int) and numel > 0:
                cur_numel = cur.get("numel")
                if not isinstance(cur_numel, int) or cur_numel <= 0:
                    cur["numel"] = int(numel)
            return
        tensors[tensor_id] = {
            "dtype": dtype,
            "kind": kind,
            "requires_grad": bool(requires_grad),
            "persistent": bool(persistent),
            "producer": producer,
            "shape": list(shape) if isinstance(shape, list) else None,
            "numel": int(numel) if isinstance(numel, int) and numel > 0 else None,
        }

    def resolve_weights_for_op(
        logical_op: str,
        layer: int,
        section: str,
        rmsnorm_idx: int,
        norm_variant: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        # Resolve template aliases -> concrete manifest tensors.
        # This keeps IR1 data-driven and avoids model-family conditionals later.
        resolved = {}
        specs = WEIGHTS_BY_LOGICAL_OP.get(logical_op, [])
        for kernel_alias, logical_weight_key, required in specs:
            weight_key = _weight_key_override(logical_op, logical_weight_key, layer, rmsnorm_idx, section, norm_variant)
            weight_name = _resolve_weight_name(weight_index, weight_key, None if weight_key in ("token_emb", "output_weight", "final_ln_weight", "final_ln_bias") else layer)
            # logits prefers output_weight when present; otherwise tied token_emb.
            if logical_op == "logits" and logical_weight_key == "output_weight" and weight_name is None:
                continue
            if logical_op == "logits" and logical_weight_key == "token_emb":
                # only use tied emb when lm_head not present
                lm_head_name = _resolve_weight_name(weight_index, "output_weight", layer)
                if lm_head_name is not None:
                    continue
            if weight_name is None:
                if required:
                    issues.append("Missing required weight `%s` for op=%s layer=%d" % (weight_key, logical_op, layer))
                continue
            entry = weight_index[weight_name]
            dtype = str(entry.get("dtype", "fp32")).lower()
            shape = entry.get("shape")
            numel = _entry_numel(entry)
            tensor_id = "weight.%s" % weight_name
            ensure_tensor(
                tensor_id=tensor_id,
                dtype=dtype,
                kind="weight",
                requires_grad=_is_trainable_dtype(dtype),
                persistent=True,
                producer=None,
                shape=shape if isinstance(shape, list) else None,
                numel=numel,
            )
            resolved[kernel_alias] = {
                "name": weight_name,
                "tensor": tensor_id,
                "dtype": dtype,
                "shape": shape if isinstance(shape, list) else None,
                "numel": int(numel) if isinstance(numel, int) and numel > 0 else None,
                "kind": "weight",
                "requires_grad": _is_trainable_dtype(dtype),
                "persistent": True,
                "from_manifest": True
            }
        # Training rule convenience: when logits are tied-only, expose W alias.
        if logical_op == "logits" and "W" not in resolved and "W_tied" in resolved:
            alias = dict(resolved["W_tied"])
            alias["alias_of"] = "W_tied"
            resolved["W"] = alias
        return resolved

    def _bridge_output_spec(
        *,
        tensor_id: str,
        shape: List[int],
        numel: int,
        kind: str = "tmp",
        requires_grad: bool = False,
        dtype: str = "fp32",
        persistent: bool = False,
    ) -> Dict[str, Any]:
        return {
            "tensor": tensor_id,
            "dtype": dtype,
            "kind": kind,
            "requires_grad": bool(requires_grad),
            "persistent": bool(persistent),
            "shape": list(shape),
            "numel": int(numel),
        }

    def _make_bridge_tmp_spec(owner_op: str, role: str, layer: int, section: str, head_group: str) -> Dict[str, Any]:
        shape, numel = _infer_head_major_bridge_shape_numel(config, head_group)
        layer_tag = "L%d" % int(layer) if int(layer) >= 0 else str(section)
        tid = "tmp.bridge.%s.%s.%s.%d" % (layer_tag, owner_op, role, op_id)
        return _bridge_output_spec(
            tensor_id=tid,
            shape=shape,
            numel=numel,
            kind="tmp",
            requires_grad=False,
            dtype="fp32",
            persistent=False,
        )

    def add_bridge_op(
        bridge_kind: str,
        section: str,
        layer: int,
        *,
        input_ref: Dict[str, Any],
        output_spec: Dict[str, Any],
        owner_op: str,
        role: str,
        head_group: str,
        layout_in: str,
        layout_out: str,
    ) -> Dict[str, Any]:
        nonlocal op_id
        instance = next_instance(bridge_kind, layer, section)
        ensure_tensor(
            tensor_id=str(output_spec.get("tensor")),
            dtype=str(output_spec.get("dtype", "fp32") or "fp32"),
            kind=str(output_spec.get("kind", "tmp") or "tmp"),
            requires_grad=bool(output_spec.get("requires_grad", False)),
            persistent=bool(output_spec.get("persistent", False)),
            producer={"op_id": op_id, "output_name": "dst"},
            shape=output_spec.get("shape") if isinstance(output_spec.get("shape"), list) else None,
            numel=output_spec.get("numel") if isinstance(output_spec.get("numel"), int) else None,
        )
        in_ref = {
            "tensor": input_ref.get("tensor"),
            "dtype": input_ref.get("dtype", "fp32"),
            "kind": input_ref.get("kind", "activation"),
            "requires_grad": bool(input_ref.get("requires_grad", False)),
            "shape": input_ref.get("shape") if isinstance(input_ref.get("shape"), list) else None,
            "numel": input_ref.get("numel") if isinstance(input_ref.get("numel"), int) else None,
        }
        if "from_op" in input_ref:
            in_ref["from_op"] = input_ref["from_op"]
            in_ref["from_output"] = input_ref["from_output"]
        else:
            in_ref["from"] = input_ref.get("from", "bridge")
        out_ref = {
            "tensor": output_spec["tensor"],
            "dtype": output_spec.get("dtype", "fp32"),
            "kind": output_spec.get("kind", "tmp"),
            "requires_grad": bool(output_spec.get("requires_grad", False)),
            "shape": output_spec.get("shape"),
            "numel": output_spec.get("numel"),
        }
        op = {
            "op_id": op_id,
            "op": bridge_kind,
            "kernel_id": None,
            "section": section,
            "layer": layer,
            "instance": instance,
            "phase": "bridge",
            "dataflow": {
                "inputs": {"src": in_ref},
                "outputs": {"dst": out_ref},
            },
            "weights": {},
            "grad_rule": None,
            "requires_grad": False,
            "save_for_backward": {},
            "attrs": {
                "owner_op": owner_op,
                "role": role,
                "head_group": head_group,
                "layout_in": layout_in,
                "layout_out": layout_out,
            },
        }
        ops.append(op)
        created_op_id = op_id
        op_id += 1
        return {
            "op_id": created_op_id,
            "op": bridge_kind,
            "role": role,
            "head_group": head_group,
            "input_tensor": in_ref.get("tensor"),
            "output_tensor": out_ref.get("tensor"),
            "layout_in": layout_in,
            "layout_out": layout_out,
            "output_ref": {
                "tensor": out_ref["tensor"],
                "dtype": out_ref["dtype"],
                "kind": out_ref["kind"],
                "requires_grad": out_ref["requires_grad"],
                "shape": out_ref.get("shape"),
                "numel": out_ref.get("numel"),
                "from_op": created_op_id,
                "from_output": "dst",
            },
        }

    def derive_save_for_backward(op: Dict[str, Any]) -> None:
        # save_for_backward is derived from grad rules, not handwritten per op.
        # This keeps forward IR and backward synthesis coupled by one contract.
        grad_rule_name = op.get("grad_rule")
        if not grad_rule_name:
            op["save_for_backward"] = {}
            return
        rule = (grad_rules.get("rules", {}) or {}).get(grad_rule_name)
        if not isinstance(rule, dict):
            op["save_for_backward"] = {}
            warnings.append("No grad rule found for `%s` (op_id=%d)" % (grad_rule_name, op["op_id"]))
            return

        requires_saved = list(rule.get("requires_saved", []) or [])
        extra_saved = list(rule.get("extra_saved", []) or [])
        saved = {}
        unresolved = []

        for key in requires_saved:
            if key in op.get("weights", {}):
                winfo = op["weights"][key]
                saved[key] = {
                    "tensor": winfo["tensor"],
                    "kind": "weight",
                    "shape": winfo.get("shape"),
                    "numel": winfo.get("numel"),
                }
                continue
            if key in op.get("dataflow", {}).get("inputs", {}):
                iref = op["dataflow"]["inputs"][key]
                item = {
                    "tensor": iref.get("tensor"),
                    "kind": iref.get("kind", "activation"),
                    "shape": iref.get("shape"),
                    "numel": iref.get("numel"),
                }
                if "from_op" in iref:
                    item["from_op"] = iref["from_op"]
                    item["from_output"] = iref["from_output"]
                saved[key] = item
                continue
            if key in op.get("dataflow", {}).get("outputs", {}):
                oref = op["dataflow"]["outputs"][key]
                saved[key] = {
                    "tensor": oref["tensor"],
                    "kind": oref.get("kind", "activation"),
                    "shape": oref.get("shape"),
                    "numel": oref.get("numel"),
                }
                continue
            unresolved.append(key)

        for key in extra_saved:
            saved_shape, saved_numel = _infer_saved_shape_numel(key, config)
            saved_tensor = _make_saved_tensor_id(op["op_id"], key)
            ensure_tensor(
                tensor_id=saved_tensor,
                dtype="fp32",
                kind="saved_activation",
                requires_grad=False,
                persistent=True,
                producer={"op_id": op["op_id"], "output_name": key},
                shape=saved_shape,
                numel=saved_numel,
            )
            saved[key] = {
                "tensor": saved_tensor,
                "kind": "saved_activation",
                "computed_by_kernel": True,
                "shape": saved_shape,
                "numel": saved_numel,
            }

        op["save_for_backward"] = saved
        if unresolved:
            op["save_for_backward_unresolved"] = unresolved
            msg = "Unresolved save_for_backward keys for op_id=%d (%s): %s" % (
                op["op_id"], op["op"], ",".join(unresolved)
            )
            if strict:
                issues.append(msg)
            else:
                warnings.append(msg)

    def _resolve_attention_runtime_contract(
        logical_op: str,
        kernel_id: str,
        saved: Dict[str, Any],
    ) -> Dict[str, Any]:
        saved_overrides = train_attention_runtime_contract.get("saved_tensor_kernel_overrides")
        if not isinstance(saved_overrides, dict):
            saved_overrides = {}
        zero_saved = train_attention_runtime_contract.get("requires_zero_sliding_window_for_saved_tensors")
        zero_saved_keys = {
            str(item).strip()
            for item in (zero_saved if isinstance(zero_saved, list) else [])
            if str(item).strip()
        }

        materialized_saved: List[str] = []
        matched_saved_key: Optional[str] = None
        runtime_kernel_id = kernel_id
        for key, value in saved.items():
            if not isinstance(value, dict):
                continue
            tensor_id = str(value.get("tensor", "") or "").strip()
            if not tensor_id:
                continue
            materialized_saved.append(str(key))
            override = str(saved_overrides.get(key, "") or "").strip()
            if override and matched_saved_key is None:
                runtime_kernel_id = override
                matched_saved_key = str(key)

        runtime_contract = {
            "version": 1,
            "semantic_op": logical_op,
            "kernel_id": runtime_kernel_id,
            "base_kernel_id": kernel_id,
            "materialize_saved_attn_weights": "attn_weights" in materialized_saved,
            "materialized_saved_tensors": materialized_saved,
            "contract_source": "template.contract.attention_contract.train_runtime_contract",
        }
        if matched_saved_key is not None:
            runtime_contract["saved_tensor_kernel_key"] = matched_saved_key
        if matched_saved_key in zero_saved_keys:
            runtime_contract["requires_zero_sliding_window"] = True
        return runtime_contract

    def attach_runtime_contract(op: Dict[str, Any]) -> None:
        logical_op = str(op.get("op", "") or "")
        kernel_id = str(op.get("kernel_id", "") or "")
        if logical_op not in ("attn", "attn_sliding") or not kernel_id:
            return
        saved = op.get("save_for_backward") if isinstance(op.get("save_for_backward"), dict) else {}
        runtime_contract = _resolve_attention_runtime_contract(logical_op, kernel_id, saved if isinstance(saved, dict) else {})
        runtime_kernel_id = str(runtime_contract.get("kernel_id", "") or kernel_id)
        op["runtime_kernel_id"] = runtime_kernel_id
        op["runtime_contract"] = runtime_contract

    def prepare_forward_bridge_plan(
        logical_op: str,
        *,
        section: str,
        layer: int,
        inputs: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not bridge_enabled:
            return None
        spec = FORWARD_BRIDGE_PLAN_SPECS.get(logical_op)
        if not isinstance(spec, dict):
            return None

        pre_entries: List[Dict[str, Any]] = []
        pre_refs: Dict[str, Dict[str, Any]] = {}
        for row in spec.get("pre_roles", ()):
            input_key = str(row.get("input_key", "") or "")
            input_ref = inputs.get(input_key)
            if not isinstance(input_ref, dict):
                return None
            bridge = add_bridge_op(
                "bridge_token_to_head_major",
                section=section,
                layer=layer,
                input_ref=input_ref,
                output_spec=_make_bridge_tmp_spec(
                    logical_op,
                    str(row.get("tmp_role", row.get("role", input_key)) or input_key),
                    layer,
                    section,
                    str(row.get("head_group", "num_heads") or "num_heads"),
                ),
                owner_op=logical_op,
                role=str(row.get("role", input_key) or input_key),
                head_group=str(row.get("head_group", "num_heads") or "num_heads"),
                layout_in="token_major",
                layout_out="head_major",
            )
            pre_entries.append(bridge)
            pre_refs[str(row.get("role", input_key) or input_key)] = bridge["output_ref"]

        kernel_outputs: Dict[str, Dict[str, Any]] = {}
        for row in spec.get("kernel_output_roles", ()):
            head_group = str(row.get("head_group", "num_heads") or "num_heads")
            tmp_spec = _make_bridge_tmp_spec(
                logical_op,
                str(row.get("tmp_role", row.get("name", "tmp")) or "tmp"),
                layer,
                section,
                head_group,
            )
            ensure_tensor(
                tensor_id=str(tmp_spec.get("tensor")),
                dtype=str(tmp_spec.get("dtype", "fp32") or "fp32"),
                kind=str(tmp_spec.get("kind", "tmp") or "tmp"),
                requires_grad=bool(tmp_spec.get("requires_grad", False)),
                persistent=bool(tmp_spec.get("persistent", False)),
                producer=None,
                shape=tmp_spec.get("shape") if isinstance(tmp_spec.get("shape"), list) else None,
                numel=tmp_spec.get("numel") if isinstance(tmp_spec.get("numel"), int) else None,
            )
            kernel_outputs[str(row.get("name", "") or "")] = tmp_spec

        return {
            "logical_op": logical_op,
            "section": section,
            "layer": layer,
            "spec": spec,
            "pre_entries": pre_entries,
            "pre_refs": pre_refs,
            "kernel_outputs": kernel_outputs,
        }

    def finalize_forward_bridge_plan(
        prepared: Optional[Dict[str, Any]],
        *,
        outputs: Dict[str, Dict[str, Any]],
        runtime_contract: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(prepared, dict):
            return None
        spec = prepared.get("spec")
        if not isinstance(spec, dict):
            return None
        logical_op = str(prepared.get("logical_op", "") or "")
        section = str(prepared.get("section", "") or "")
        layer = int(prepared.get("layer", 0) or 0)
        pre_entries = list(prepared.get("pre_entries") or [])
        pre_refs = prepared.get("pre_refs") if isinstance(prepared.get("pre_refs"), dict) else {}
        kernel_outputs = prepared.get("kernel_outputs") if isinstance(prepared.get("kernel_outputs"), dict) else {}

        post_entries: List[Dict[str, Any]] = []
        for row in spec.get("post_roles", ()):
            output_key = str(row.get("output_key", "") or "")
            output_ref = outputs.get(output_key)
            if not isinstance(output_ref, dict):
                return None
            source = str(row.get("source", "") or "")
            source_kind, _, source_name = source.partition(":")
            if not source_kind or not source_name:
                return None
            if source_kind == "pre":
                input_ref = pre_refs.get(source_name)
            elif source_kind == "kernel":
                input_ref = kernel_outputs.get(source_name)
            else:
                return None
            if not isinstance(input_ref, dict):
                return None
            bridge = add_bridge_op(
                "bridge_head_to_token_major",
                section=section,
                layer=layer,
                input_ref=input_ref,
                output_spec=dict(output_ref),
                owner_op=logical_op,
                role=str(row.get("role", output_key) or output_key),
                head_group=str(row.get("head_group", "num_heads") or "num_heads"),
                layout_in="head_major",
                layout_out="token_major",
            )
            post_entries.append(bridge)

        plan: Dict[str, Any] = {
            "version": 1,
            "mode": "explicit",
            "semantic_op": logical_op,
            "kernel_layout": str(spec.get("kernel_layout", "head_major") or "head_major"),
            "tensor_layout": str(spec.get("tensor_layout", "token_major") or "token_major"),
            "pre": pre_entries,
            "post": post_entries,
        }
        if isinstance(runtime_contract, dict) and runtime_contract:
            plan["runtime_contract"] = dict(runtime_contract)
        return plan

    def add_op(
        logical_op: str,
        kernel_id: Optional[str],
        section: str,
        layer: int,
        inputs: Dict[str, Dict[str, Any]],
        output_specs: Dict[str, str],
        rmsnorm_idx: int = 0,
        norm_variant: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        nonlocal op_id
        instance = next_instance(logical_op, layer, section)

        if kernel_id:
            if kernel_id not in kernels:
                issues.append("Kernel `%s` not in KERNEL_REGISTRY for op=%s" % (kernel_id, logical_op))
            if kernel_id not in binding_ids:
                issues.append("Kernel `%s` missing bindings entry for op=%s" % (kernel_id, logical_op))

        weights = resolve_weights_for_op(logical_op, layer, section, rmsnorm_idx, norm_variant)
        grad_rule = _op_family_grad_rule(grad_rules, logical_op)

        op_base = _op_base_name(logical_op, layer, instance, section)
        outputs = {}
        for out_name, out_dtype in output_specs.items():
            out_shape, out_numel = _infer_output_shape_numel(logical_op, out_name, config)
            tensor_id = "act.%s.%s" % (op_base, out_name)
            ensure_tensor(
                tensor_id=tensor_id,
                dtype=out_dtype,
                kind="activation",
                requires_grad=True,
                persistent=False,
                producer={"op_id": op_id, "output_name": out_name},
                shape=out_shape,
                numel=out_numel,
            )
            outputs[out_name] = {
                "tensor": tensor_id,
                "dtype": out_dtype,
                "kind": "activation",
                "requires_grad": True,
                "shape": out_shape,
                "numel": out_numel,
            }

        # Classify inputs with manifest/dataflow truth.
        in_data = {}
        for in_name, src in inputs.items():
            src_shape = src.get("shape")
            src_numel = src.get("numel")
            if not isinstance(src_shape, list) or _shape_numel(src_shape) is None:
                src_shape = None
            if not isinstance(src_numel, int) or src_numel <= 0:
                src_numel = None
            if src_shape is None or src_numel is None:
                inferred_shape, inferred_numel = _infer_external_input_shape_numel(
                    src.get("tensor"),
                    src.get("from"),
                    config,
                )
                if src_shape is None:
                    src_shape = inferred_shape
                if src_numel is None:
                    src_numel = inferred_numel

            tensor_id = src.get("tensor")
            if isinstance(tensor_id, str) and tensor_id:
                ensure_tensor(
                    tensor_id=tensor_id,
                    dtype=str(src.get("dtype", "fp32") or "fp32"),
                    kind=str(src.get("kind", "activation") or "activation"),
                    requires_grad=bool(src.get("requires_grad", True)),
                    persistent=bool(src.get("persistent", False)),
                    producer=None,
                    shape=src_shape,
                    numel=src_numel,
                )

            item = {
                "tensor": tensor_id,
                "dtype": src.get("dtype", "fp32"),
                "kind": src.get("kind", "activation"),
                "requires_grad": bool(src.get("requires_grad", True)),
                "shape": src_shape,
                "numel": src_numel,
            }
            if "from_op" in src:
                item["from_op"] = src["from_op"]
                item["from_output"] = src["from_output"]
            else:
                item["from"] = src.get("from", "external")
            in_data[in_name] = item

        op = {
            "op_id": op_id,
            "op": logical_op,
            "kernel_id": kernel_id,
            "section": section,
            "layer": layer,
            "instance": instance,
            "phase": "forward",
            "dataflow": {
                "inputs": in_data,
                "outputs": outputs
            },
            "weights": weights,
            "grad_rule": grad_rule,
            "requires_grad": True
        }

        derive_save_for_backward(op)
        attach_runtime_contract(op)
        ops.append(op)
        op_id += 1

        # Return output refs for chaining.
        out_refs = {}
        for out_name, out_obj in outputs.items():
            out_refs[out_name] = {
                "tensor": out_obj["tensor"],
                "dtype": out_obj["dtype"],
                "kind": "activation",
                "requires_grad": out_obj["requires_grad"],
                "shape": out_obj.get("shape"),
                "numel": out_obj.get("numel"),
                "from_op": op["op_id"],
                "from_output": out_name
            }
        return out_refs

    # Header: tokenize/embedding stream setup (single-token training-step contract).
    current_main = None
    for raw_op in header_ops:
        if raw_op in ("bpe_tokenizer", "wordpiece_tokenizer", "tokenizer"):
            # Keep metadata-only op for traceability.
            op = {
                "op_id": op_id,
                "op": raw_op,
                "kernel_id": None,
                "section": "header",
                "layer": -1,
                "instance": next_instance(raw_op, -1, "header"),
                "phase": "forward",
                "dataflow": {"inputs": {}, "outputs": {}},
                "weights": {},
                "grad_rule": None,
                "requires_grad": False,
                "save_for_backward": {}
            }
            ops.append(op)
            op_id += 1
            continue
        if raw_op == "dense_embedding_lookup":
            out = add_op(
                logical_op="dense_embedding_lookup",
                kernel_id=forward_kernels["dense_embedding_lookup"],
                section="header",
                layer=-1,
                inputs={
                    "token_ids": {
                        "tensor": "input.token_ids",
                        "dtype": "int32",
                        "kind": "input",
                        "requires_grad": False,
                        "from": "external:token_ids"
                    }
                },
                output_specs={"out": "fp32"}
            )
            current_main = out["out"]

    if current_main is None:
        raise RuntimeError("Header did not produce main activation stream")

    # Body per layer: keep forward order stable so IR2 can reverse-traverse cleanly.
    for layer in range(num_layers):
        rmsnorm_count = 0
        residual_slot = None
        q_ref = None
        k_ref = None
        v_ref = None
        recurrent_proj_input = None
        layer_body_ops = (
            build_ir_v7._resolve_body_ops_for_layer(body_def, config, layer)
            if isinstance(body_def, dict)
            else list(body_def)
        )
        recurrent_gate_ref = None
        recurrent_alpha_ref = None
        recurrent_beta_ref = None
        recurrent_q_preconv_ref = None
        recurrent_k_preconv_ref = None
        recurrent_v_preconv_ref = None
        attn_gate_ref = None
        for body_idx, raw_op in enumerate(layer_body_ops):
            prev_raw_op = layer_body_ops[body_idx - 1] if body_idx > 0 else None
            if raw_op in ("rmsnorm", "attn_norm", "ffn_norm", "post_attention_norm", "post_ffn_norm"):
                should_snapshot_residual = raw_op in ("rmsnorm", "attn_norm", "ffn_norm")
                if raw_op in ("post_attention_norm", "post_ffn_norm") and prev_raw_op == "residual_add":
                    should_snapshot_residual = True
                if should_snapshot_residual:
                    # Snapshot the branch input before norms that start an attention/MLP branch.
                    # post_attention_norm/post_ffn_norm should only do this when they
                    # follow an actual residual merge. Gemma-family templates place
                    # those markers before the merge, and resnapshotting there would
                    # incorrectly turn residual_add(a, b) into residual_add(a, a).
                    residual_slot = dict(current_main)
                if raw_op in ("post_attention_norm", "post_ffn_norm"):
                    opt_key = raw_op
                    if _resolve_weight_name(weight_index, opt_key, layer) is None:
                        continue
                out = add_op(
                    logical_op="rmsnorm",
                    kernel_id=forward_kernels["rmsnorm"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"output": "fp32", "rstd_cache": "fp32"},
                    rmsnorm_idx=rmsnorm_count,
                    norm_variant=raw_op,
                )
                current_main = out["output"]
                if raw_op == "rmsnorm":
                    rmsnorm_count += 1
            elif raw_op == "qkv_proj":
                # Keep q/k/v as explicit ops so backward can map per-projection dW paths.
                q_out = add_op(
                    logical_op="q_proj",
                    kernel_id=forward_kernels["q_proj"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"y": "fp32"}
                )
                k_out = add_op(
                    logical_op="k_proj",
                    kernel_id=forward_kernels["k_proj"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"y": "fp32"}
                )
                v_out = add_op(
                    logical_op="v_proj",
                    kernel_id=forward_kernels["v_proj"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"y": "fp32"}
                )
                q_ref = q_out["y"]
                k_ref = k_out["y"]
                v_ref = v_out["y"]
            elif raw_op == "q_gate_proj":
                qg_out = add_op(
                    logical_op="q_gate_proj",
                    kernel_id=forward_kernels["q_gate_proj"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"y": "fp32"}
                )
                current_main = qg_out["y"]
            elif raw_op == "split_q_gate":
                split_out = add_op(
                    logical_op="split_q_gate",
                    kernel_id=forward_kernels["split_q_gate"],
                    section="body",
                    layer=layer,
                    inputs={"packed_qg": current_main},
                    output_specs={"q": "fp32", "gate": "fp32"}
                )
                q_ref = split_out["q"]
                attn_gate_ref = split_out["gate"]
            elif raw_op == "recurrent_qkv_proj":
                recurrent_proj_input = current_main
                rec_out = add_op(
                    logical_op="recurrent_qkv_proj",
                    kernel_id=forward_kernels["recurrent_qkv_proj"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"y": "fp32"}
                )
                current_main = rec_out["y"]
            elif raw_op == "recurrent_gate_proj":
                rec_input = recurrent_proj_input or current_main
                rec_gate_out = add_op(
                    logical_op="recurrent_gate_proj",
                    kernel_id=forward_kernels["recurrent_gate_proj"],
                    section="body",
                    layer=layer,
                    inputs={"input": rec_input},
                    output_specs={"y": "fp32"}
                )
                recurrent_gate_ref = rec_gate_out["y"]
            elif raw_op == "recurrent_alpha_proj":
                rec_input = recurrent_proj_input or current_main
                rec_alpha_out = add_op(
                    logical_op="recurrent_alpha_proj",
                    kernel_id=forward_kernels["recurrent_alpha_proj"],
                    section="body",
                    layer=layer,
                    inputs={"input": rec_input},
                    output_specs={"y": "fp32"}
                )
                recurrent_alpha_ref = rec_alpha_out["y"]
            elif raw_op == "recurrent_beta_proj":
                rec_input = recurrent_proj_input or current_main
                rec_beta_out = add_op(
                    logical_op="recurrent_beta_proj",
                    kernel_id=forward_kernels["recurrent_beta_proj"],
                    section="body",
                    layer=layer,
                    inputs={"input": rec_input},
                    output_specs={"y": "fp32"}
                )
                recurrent_beta_ref = rec_beta_out["y"]
            elif raw_op == "recurrent_split_qkv":
                rec_split = add_op(
                    logical_op="recurrent_split_qkv",
                    kernel_id=forward_kernels["recurrent_split_qkv"],
                    section="body",
                    layer=layer,
                    inputs={"packed_qkv": current_main},
                    output_specs={"q": "fp32", "k": "fp32", "v": "fp32"}
                )
                recurrent_q_preconv_ref = rec_split["q"]
                recurrent_k_preconv_ref = rec_split["k"]
                recurrent_v_preconv_ref = rec_split["v"]
            elif raw_op == "recurrent_dt_gate":
                if recurrent_alpha_ref is None:
                    issues.append("recurrent_dt_gate missing alpha input at layer=%d" % layer)
                    continue
                rec_gate = add_op(
                    logical_op="recurrent_dt_gate",
                    kernel_id=forward_kernels["recurrent_dt_gate"],
                    section="body",
                    layer=layer,
                    inputs={"alpha": recurrent_alpha_ref},
                    output_specs={"gate": "fp32"}
                )
                recurrent_alpha_ref = rec_gate["gate"]
            elif raw_op == "recurrent_conv_state_update":
                if recurrent_q_preconv_ref is None or recurrent_k_preconv_ref is None or recurrent_v_preconv_ref is None:
                    issues.append("recurrent_conv_state_update missing q/k/v inputs at layer=%d" % layer)
                    continue
                rec_conv = add_op(
                    logical_op="recurrent_conv_state_update",
                    kernel_id=forward_kernels["recurrent_conv_state_update"],
                    section="body",
                    layer=layer,
                    inputs={
                        "state_in": {"tensor": "input.recurrent_conv_state", "dtype": "fp32", "kind": "input", "requires_grad": False, "from": "external:recurrent_conv_state"},
                        "q": recurrent_q_preconv_ref,
                        "k": recurrent_k_preconv_ref,
                        "v": recurrent_v_preconv_ref,
                    },
                    output_specs={"conv_x": "fp32", "state_out": "fp32"}
                )
                current_main = rec_conv["conv_x"]
            elif raw_op == "recurrent_ssm_conv":
                rec_conv_out = add_op(
                    logical_op="recurrent_ssm_conv",
                    kernel_id=forward_kernels["recurrent_ssm_conv"],
                    section="body",
                    layer=layer,
                    inputs={"conv_x": current_main},
                    output_specs={"out": "fp32"}
                )
                current_main = rec_conv_out["out"]
            elif raw_op == "recurrent_silu":
                rec_silu = add_op(
                    logical_op="recurrent_silu",
                    kernel_id=forward_kernels["recurrent_silu"],
                    section="body",
                    layer=layer,
                    inputs={"x": current_main},
                    output_specs={"out": "fp32"}
                )
                current_main = rec_silu["out"]
            elif raw_op == "recurrent_split_conv_qkv":
                rec_split = add_op(
                    logical_op="recurrent_split_conv_qkv",
                    kernel_id=forward_kernels["recurrent_split_conv_qkv"],
                    section="body",
                    layer=layer,
                    inputs={"packed_qkv": current_main},
                    output_specs={"q": "fp32", "k": "fp32", "v": "fp32"}
                )
                q_ref = rec_split["q"]
                k_ref = rec_split["k"]
                v_ref = rec_split["v"]
            elif raw_op == "recurrent_qk_l2_norm":
                if q_ref is None or k_ref is None:
                    issues.append("recurrent_qk_l2_norm missing q/k inputs at layer=%d" % layer)
                    continue
                rec_norm = add_op(
                    logical_op="recurrent_qk_l2_norm",
                    kernel_id=forward_kernels["recurrent_qk_l2_norm"],
                    section="body",
                    layer=layer,
                    inputs={"q": q_ref, "k": k_ref},
                    output_specs={"q": "fp32", "k": "fp32"}
                )
                q_ref = rec_norm["q"]
                k_ref = rec_norm["k"]
            elif raw_op == "recurrent_core":
                if q_ref is None or k_ref is None or v_ref is None or recurrent_alpha_ref is None or recurrent_beta_ref is None:
                    issues.append("recurrent_core missing q/k/v/g/beta inputs at layer=%d" % layer)
                    continue
                rec_core = add_op(
                    logical_op="recurrent_core",
                    kernel_id=forward_kernels["recurrent_core"],
                    section="body",
                    layer=layer,
                    inputs={
                        "q": q_ref,
                        "k": k_ref,
                        "v": v_ref,
                        "g": recurrent_alpha_ref,
                        "beta": recurrent_beta_ref,
                        "state_in": {"tensor": "input.recurrent_ssm_state", "dtype": "fp32", "kind": "input", "requires_grad": False, "from": "external:recurrent_ssm_state"},
                    },
                    output_specs={"out": "fp32", "state_out": "fp32"}
                )
                current_main = rec_core["out"]
            elif raw_op == "recurrent_norm_gate":
                if recurrent_gate_ref is None:
                    issues.append("recurrent_norm_gate missing gate input at layer=%d" % layer)
                    continue
                rec_norm = add_op(
                    logical_op="recurrent_norm_gate",
                    kernel_id=forward_kernels["recurrent_norm_gate"],
                    section="body",
                    layer=layer,
                    inputs={"x": current_main, "gate": recurrent_gate_ref},
                    output_specs={"out": "fp32"}
                )
                current_main = rec_norm["out"]
            elif raw_op == "qk_norm":
                if q_ref is None or k_ref is None:
                    issues.append("qk_norm missing q/k inputs at layer=%d" % layer)
                    continue
                prepared_plan = prepare_forward_bridge_plan(
                    "qk_norm",
                    section="body",
                    layer=layer,
                    inputs={"q": q_ref, "k": k_ref},
                )
                qk_out = add_op(
                    logical_op="qk_norm",
                    kernel_id=forward_kernels["qk_norm"],
                    section="body",
                    layer=layer,
                    inputs={"q": q_ref, "k": k_ref},
                    output_specs={"q": "fp32", "k": "fp32"}
                )
                qk_op = ops[-1]
                plan = finalize_forward_bridge_plan(
                    prepared_plan,
                    outputs={"q": qk_out["q"], "k": qk_out["k"]},
                )
                if isinstance(plan, dict):
                    qk_op["bridge_plan"] = plan
                q_ref = qk_out["q"]
                k_ref = qk_out["k"]
            elif raw_op == "rope_qk":
                if q_ref is None or k_ref is None:
                    issues.append("rope_qk missing q/k inputs at layer=%d" % layer)
                    continue
                prepared_plan = prepare_forward_bridge_plan(
                    "rope_qk",
                    section="body",
                    layer=layer,
                    inputs={"q": q_ref, "k": k_ref},
                )
                rope_out = add_op(
                    logical_op="rope_qk",
                    kernel_id=forward_kernels["rope_qk"],
                    section="body",
                    layer=layer,
                    inputs={"q": q_ref, "k": k_ref},
                    output_specs={"q": "fp32", "k": "fp32"}
                )
                rope_op = ops[-1]
                plan = finalize_forward_bridge_plan(
                    prepared_plan,
                    outputs={"q": rope_out["q"], "k": rope_out["k"]},
                )
                if isinstance(plan, dict):
                    rope_op["bridge_plan"] = plan
                q_ref = rope_out["q"]
                k_ref = rope_out["k"]
            elif raw_op in ("attn", "attn_sliding"):
                if q_ref is None or k_ref is None or v_ref is None:
                    issues.append("attention missing q/k/v inputs at layer=%d" % layer)
                    continue
                op_name = "attn_sliding" if raw_op == "attn_sliding" else "attn"
                kid = forward_kernels[op_name]
                prepared_plan = prepare_forward_bridge_plan(
                    op_name,
                    section="body",
                    layer=layer,
                    inputs={"q": q_ref, "k": k_ref, "v": v_ref},
                )
                attn_out = add_op(
                    logical_op=op_name,
                    kernel_id=kid,
                    section="body",
                    layer=layer,
                    inputs={"q": q_ref, "k": k_ref, "v": v_ref},
                    output_specs={"out": "fp32"}
                )
                attn_op = ops[-1]
                runtime_contract = attn_op.get("runtime_contract") if isinstance(attn_op.get("runtime_contract"), dict) else {}
                plan = finalize_forward_bridge_plan(
                    prepared_plan,
                    outputs={"out": attn_out["out"]},
                    runtime_contract=runtime_contract,
                )
                if isinstance(plan, dict):
                    attn_op["bridge_plan"] = plan
                current_main = attn_out["out"]
            elif raw_op == "attn_gate_sigmoid_mul":
                if attn_gate_ref is None:
                    issues.append("attn_gate_sigmoid_mul missing gate input at layer=%d" % layer)
                    continue
                gate_out = add_op(
                    logical_op="attn_gate_sigmoid_mul",
                    kernel_id=forward_kernels["attn_gate_sigmoid_mul"],
                    section="body",
                    layer=layer,
                    inputs={"x": current_main, "gate": attn_gate_ref},
                    output_specs={"out": "fp32"}
                )
                current_main = gate_out["out"]
            elif raw_op == "out_proj":
                op_out = add_op(
                    logical_op="out_proj",
                    kernel_id=forward_kernels["out_proj"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"y": "fp32"}
                )
                current_main = op_out["y"]
            elif raw_op == "recurrent_out_proj":
                op_out = add_op(
                    logical_op="recurrent_out_proj",
                    kernel_id=forward_kernels["recurrent_out_proj"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"y": "fp32"}
                )
                current_main = op_out["y"]
            elif raw_op == "residual_add":
                if residual_slot is None:
                    issues.append("residual_add has no saved residual at layer=%d" % layer)
                    continue
                res_out = add_op(
                    logical_op="residual_add",
                    kernel_id=forward_kernels["residual_add"],
                    section="body",
                    layer=layer,
                    inputs={"a": current_main, "b": residual_slot},
                    output_specs={"out": "fp32"}
                )
                current_main = res_out["out"]
            elif raw_op == "mlp_gate_up":
                mlp_up = add_op(
                    logical_op="mlp_gate_up",
                    kernel_id=forward_kernels["mlp_gate_up"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"y": "fp32"}
                )
                current_main = mlp_up["y"]
            elif raw_op in ("silu_mul", "geglu"):
                logical = "silu_mul" if raw_op == "silu_mul" else "geglu"
                act_out = add_op(
                    logical_op=logical,
                    kernel_id=forward_kernels[logical],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"out": "fp32"}
                )
                current_main = act_out["out"]
            elif raw_op == "mlp_down":
                down_out = add_op(
                    logical_op="mlp_down",
                    kernel_id=forward_kernels["mlp_down"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"y": "fp32"}
                )
                current_main = down_out["y"]
            else:
                warnings.append("Unsupported body op `%s` ignored for train IR" % raw_op)

    # Footer: final norm + logits projection used to seed CE backward in IR2.
    for raw_op in footer_ops:
        if raw_op in ("lm_head", "weight_tying"):
            op = {
                "op_id": op_id,
                "op": raw_op,
                "kernel_id": None,
                "section": "footer",
                "layer": -1,
                "instance": next_instance(raw_op, -1, "footer"),
                "phase": "forward",
                "dataflow": {"inputs": {}, "outputs": {}},
                "weights": {},
                "grad_rule": None,
                "requires_grad": False,
                "save_for_backward": {}
            }
            ops.append(op)
            op_id += 1
            continue
        if raw_op in ("rmsnorm", "final_rmsnorm"):
            out = add_op(
                logical_op="rmsnorm",
                kernel_id=forward_kernels["rmsnorm"],
                section="footer",
                layer=-1,
                inputs={"input": current_main},
                output_specs={"output": "fp32", "rstd_cache": "fp32"},
                rmsnorm_idx=2
            )
            current_main = out["output"]
        elif raw_op == "logits":
            logits_out = add_op(
                logical_op="logits",
                kernel_id=forward_kernels["logits"],
                section="footer",
                layer=-1,
                inputs={"input": current_main},
                output_specs={"y": "fp32"}
            )
            current_main = logits_out["y"]

    if strict and issues:
        raise RuntimeError("IR1 train build failed:\n- " + "\n- ".join(issues))

    return {
        "format": "ir1-train-v7",
        "version": 1,
        "bridge_lowering": bridge_lowering,
        "config": config,
        "template_name": template.get("name", "unknown"),
        "template_source": template_source,
        "num_layers": num_layers,
        "ops": ops,
        "tensors": tensors,
        "stats": {
            "forward_ops": len([o for o in ops if o.get("phase") == "forward"]),
            "bridge_ops": len([o for o in ops if o.get("phase") == "bridge"]),
            "metadata_ops": len([o for o in ops if o.get("kernel_id") is None]),
            "tensors": len(tensors),
            "weights": len([t for t in tensors.values() if t.get("kind") == "weight"]),
            "saved_tensors": len([t for t in tensors.values() if t.get("kind") == "saved_activation"]),
            "issues": len(issues),
            "warnings": len(warnings)
        },
        "issues": issues,
        "warnings": warnings
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build IR1 train-forward for v7.")
    ap.add_argument("--manifest", required=True, help="weights_manifest.json path")
    ap.add_argument("--output", required=True, help="Output ir1_train_forward.json")
    ap.add_argument("--grad-rules", default=str(DEFAULT_GRAD_RULES_PATH), help="grad_rules_v7.json path")
    ap.add_argument("--max-layers", type=int, default=None, help="Optional cap for fast smoke runs")
    ap.add_argument("--tokens", type=int, default=1, help="Compile-time token count for train IR/runtime (default: 1)")
    ap.add_argument("--strict", action="store_true", help="Fail on unresolved weights/save-for-backward")
    ap.add_argument("--bridge-lowering", choices=("legacy", "explicit"), default="legacy", help="How explicitly to model train layout bridges in IR1")
    ap.add_argument("--report-out", default=None, help="Optional report JSON path")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    grad_rules_path = Path(args.grad_rules)

    manifest = _load_json(manifest_path)
    if not isinstance(manifest.get("config"), dict):
        manifest["config"] = {}
    manifest["config"]["train_tokens"] = max(1, int(args.tokens or 1))
    registry = _load_json(KERNEL_REGISTRY_PATH)
    bindings_doc = _load_json(KERNEL_BINDINGS_PATH)
    grad_rules = _load_json(grad_rules_path)

    ir1 = build_ir1_train(
        manifest=manifest,
        registry=registry,
        bindings_doc=bindings_doc,
        grad_rules=grad_rules,
        max_layers=args.max_layers,
        strict=args.strict,
        bridge_lowering=str(args.bridge_lowering),
    )
    _save_json(output_path, ir1)
    print("Wrote IR1 train-forward: %s (ops=%d tensors=%d)" % (output_path, len(ir1["ops"]), len(ir1["tensors"])))

    if args.report_out:
        report = {
            "format": ir1.get("format"),
            "ops": len(ir1.get("ops", [])),
            "tensors": len(ir1.get("tensors", {})),
            "issues": ir1.get("issues", []),
            "warnings": ir1.get("warnings", []),
            "stats": ir1.get("stats", {})
        }
        _save_json(Path(args.report_out), report)
        print("Wrote report: %s" % args.report_out)

    if ir1.get("issues"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
