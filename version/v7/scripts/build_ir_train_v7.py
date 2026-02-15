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


SCRIPT_DIR = Path(__file__).resolve().parent
V7_ROOT = SCRIPT_DIR.parent
KERNEL_REGISTRY_PATH = V7_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"
KERNEL_BINDINGS_PATH = V7_ROOT / "kernel_maps" / "kernel_bindings.json"
DEFAULT_GRAD_RULES_PATH = SCRIPT_DIR / "grad_rules_v7.json"
TEMPLATES_DIR = V7_ROOT / "templates"


# NOTE: some forward kernels below are in-place at C API level (currently qk_norm_forward
# and rope_forward_qk). IR1 intentionally remains out-of-place (explicit input/output tensors)
# for deterministic graph semantics. Codegen must stage/copy before invoking those kernels.
FORWARD_KERNEL_BY_OP = {
    "dense_embedding_lookup": "dense_embedding_lookup",
    "rmsnorm": "rmsnorm_forward",
    "q_proj": "gemm_blocked_serial",
    "k_proj": "gemm_blocked_serial",
    "v_proj": "gemm_blocked_serial",
    "qk_norm": "qk_norm_forward",
    "rope_qk": "rope_forward_qk",
    "attn": "attention_forward_causal_head_major_gqa_flash_strided",
    "attn_sliding": "attention_forward_causal_head_major_gqa_flash_strided_sliding",
    "out_proj": "gemm_blocked_serial",
    "residual_add": "ck_residual_add_token_major",
    "mlp_gate_up": "gemm_blocked_serial",
    "silu_mul": "swiglu_forward",
    "geglu": "geglu_forward_fp32",
    "mlp_down": "gemm_blocked_serial",
    "logits": "gemm_blocked_serial",
}


WEIGHT_PATTERNS = {
    "token_emb": ["token_emb", "token_embd.weight", "embed_tokens.weight"],
    "output_weight": ["output.weight", "lm_head.weight"],
    "final_ln_weight": ["final_ln_weight", "norm.weight"],
    "final_ln_bias": ["final_ln_bias", "norm.bias"],
    "ln1_gamma": ["layer.{L}.ln1_gamma", "layers.{L}.attention_norm.weight"],
    "ln2_gamma": ["layer.{L}.ln2_gamma", "layers.{L}.ffn_norm.weight"],
    "wq": ["layer.{L}.wq", "layers.{L}.attention.wq"],
    "wk": ["layer.{L}.wk", "layers.{L}.attention.wk"],
    "wv": ["layer.{L}.wv", "layers.{L}.attention.wv"],
    "bq": ["layer.{L}.bq", "layers.{L}.attention.bq"],
    "bk": ["layer.{L}.bk", "layers.{L}.attention.bk"],
    "bv": ["layer.{L}.bv", "layers.{L}.attention.bv"],
    "q_norm": ["layer.{L}.q_norm", "layers.{L}.attention.q_norm"],
    "k_norm": ["layer.{L}.k_norm", "layers.{L}.attention.k_norm"],
    "wo": ["layer.{L}.wo", "layers.{L}.attention.wo"],
    "bo": ["layer.{L}.bo", "layers.{L}.attention.bo"],
    "w1": ["layer.{L}.w1", "layers.{L}.feed_forward.w1"],
    "w2": ["layer.{L}.w2", "layers.{L}.feed_forward.w2"],
    "w3": ["layer.{L}.w3", "layers.{L}.feed_forward.w3"],
    "b1": ["layer.{L}.b1", "layers.{L}.feed_forward.b1"],
    "b2": ["layer.{L}.b2", "layers.{L}.feed_forward.b2"]
}


WEIGHTS_BY_LOGICAL_OP = {
    "dense_embedding_lookup": [("weight", "token_emb", True)],
    "rmsnorm": [("gamma", "ln1_gamma", True)],  # remapped to ln2/final per context
    "q_proj": [("W", "wq", True), ("bias", "bq", False)],
    "k_proj": [("W", "wk", True), ("bias", "bk", False)],
    "v_proj": [("W", "wv", True), ("bias", "bv", False)],
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


def _template_sections(template: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    sequence = template.get("sequence", [])
    if not sequence:
        raise RuntimeError("Template missing `sequence`")
    block_name = sequence[0]
    block = template.get("block_types", {}).get(block_name, {})
    header = list(block.get("header", []))
    body_def = block.get("body", {})
    if isinstance(body_def, dict):
        body = list(body_def.get("ops", []))
    else:
        body = list(body_def or [])
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
    # Runtime currently consumes one token per ck_train_step call.
    token_count = 1
    q_dim = max(1, num_heads * head_dim)
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
        "q_dim": q_dim,
        "kv_dim": kv_dim,
        "gate_up_dim": gate_up_dim,
    }


def _infer_output_shape_numel(logical_op: str, out_name: str, config: Dict[str, Any]) -> Tuple[List[int], int]:
    dims = _train_dims(config)
    t = dims["tokens"]
    d_model = dims["d_model"]
    hidden = dims["hidden"]
    vocab = dims["vocab"]
    q_dim = dims["q_dim"]
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
    if logical_op in ("attn", "attn_sliding"):
        return [t, d_model], t * d_model
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


def _weight_key_override(op_name: str, alias: str, layer: int, rmsnorm_idx: int, section: str) -> str:
    # Route rmsnorm aliases by position.
    if op_name != "rmsnorm":
        return alias
    if section == "footer":
        if alias in ("gamma", "ln1_gamma", "ln2_gamma"):
            return "final_ln_weight"
        return alias
    if alias in ("gamma", "ln1_gamma", "ln2_gamma"):
        return "ln1_gamma" if rmsnorm_idx == 0 else "ln2_gamma"
    return alias


def _choose_template(manifest: Dict[str, Any], explicit_template: Optional[Path]) -> Dict[str, Any]:
    if isinstance(manifest.get("template"), dict) and manifest.get("template"):
        return manifest["template"]
    if explicit_template is not None:
        return _load_json(explicit_template)
    model = str((manifest.get("config") or {}).get("model", "qwen3")).lower()
    candidate = TEMPLATES_DIR / ("%s.json" % model)
    if candidate.exists():
        return _load_json(candidate)
    raise RuntimeError("No template found in manifest and no template file for model=%s" % model)


def _make_saved_tensor_id(op_id: int, key: str) -> str:
    return "saved.op%d.%s" % (op_id, key)


def build_ir1_train(
    manifest: Dict[str, Any],
    registry: Dict[str, Any],
    bindings_doc: Dict[str, Any],
    grad_rules: Dict[str, Any],
    max_layers: Optional[int],
    strict: bool
) -> Dict[str, Any]:
    config = dict(manifest.get("config", {}))
    num_layers = int(config.get("num_layers", 0) or 0)
    if max_layers is not None:
        num_layers = min(num_layers, int(max_layers))
    if num_layers <= 0:
        raise RuntimeError("Invalid num_layers in manifest/config")

    template = _choose_template(manifest, explicit_template=None)
    header_ops, body_ops, footer_ops = _template_sections(template)
    weight_index = _manifest_weight_index(manifest)
    kernels = _kernel_ids(registry)
    binding_ids = _binding_ids(bindings_doc)

    ops: List[Dict[str, Any]] = []
    tensors: Dict[str, Dict[str, Any]] = {}
    issues: List[str] = []
    warnings: List[str] = []

    op_id = 0
    instance_counter: Dict[str, int] = {}

    # External inputs.
    tensors["input.token_ids"] = {
        "dtype": "int32",
        "kind": "input",
        "requires_grad": False,
        "persistent": False,
        "producer": None,
        "shape": [1],
        "numel": 1,
    }
    tensors["input.targets"] = {
        "dtype": "int32",
        "kind": "input",
        "requires_grad": False,
        "persistent": False,
        "producer": None,
        "shape": [1],
        "numel": 1,
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
            if shape is not None and "shape" not in cur:
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
        rmsnorm_idx: int
    ) -> Dict[str, Dict[str, Any]]:
        # Resolve template aliases -> concrete manifest tensors.
        # This keeps IR1 data-driven and avoids model-family conditionals later.
        resolved = {}
        specs = WEIGHTS_BY_LOGICAL_OP.get(logical_op, [])
        for kernel_alias, logical_weight_key, required in specs:
            weight_key = _weight_key_override(logical_op, logical_weight_key, layer, rmsnorm_idx, section)
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

    def add_op(
        logical_op: str,
        kernel_id: Optional[str],
        section: str,
        layer: int,
        inputs: Dict[str, Dict[str, Any]],
        output_specs: Dict[str, str],
        rmsnorm_idx: int = 0
    ) -> Dict[str, Dict[str, Any]]:
        nonlocal op_id
        instance = next_instance(logical_op, layer, section)

        if kernel_id:
            if kernel_id not in kernels:
                issues.append("Kernel `%s` not in KERNEL_REGISTRY for op=%s" % (kernel_id, logical_op))
            if kernel_id not in binding_ids:
                issues.append("Kernel `%s` missing bindings entry for op=%s" % (kernel_id, logical_op))

        weights = resolve_weights_for_op(logical_op, layer, section, rmsnorm_idx)
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
            item = {
                "tensor": src.get("tensor"),
                "dtype": src.get("dtype", "fp32"),
                "kind": src.get("kind", "activation"),
                "requires_grad": bool(src.get("requires_grad", True)),
                "shape": src.get("shape"),
                "numel": src.get("numel"),
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
                kernel_id=FORWARD_KERNEL_BY_OP["dense_embedding_lookup"],
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
        for raw_op in body_ops:
            if raw_op == "rmsnorm":
                if rmsnorm_count in (0, 1):
                    # Snapshot pre-block activation for both residual additions in layer body.
                    residual_slot = dict(current_main)
                out = add_op(
                    logical_op="rmsnorm",
                    kernel_id=FORWARD_KERNEL_BY_OP["rmsnorm"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"output": "fp32", "rstd_cache": "fp32"},
                    rmsnorm_idx=rmsnorm_count
                )
                current_main = out["output"]
                rmsnorm_count += 1
            elif raw_op == "qkv_proj":
                # Keep q/k/v as explicit ops so backward can map per-projection dW paths.
                q_out = add_op(
                    logical_op="q_proj",
                    kernel_id=FORWARD_KERNEL_BY_OP["q_proj"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"y": "fp32"}
                )
                k_out = add_op(
                    logical_op="k_proj",
                    kernel_id=FORWARD_KERNEL_BY_OP["k_proj"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"y": "fp32"}
                )
                v_out = add_op(
                    logical_op="v_proj",
                    kernel_id=FORWARD_KERNEL_BY_OP["v_proj"],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"y": "fp32"}
                )
                q_ref = q_out["y"]
                k_ref = k_out["y"]
                v_ref = v_out["y"]
            elif raw_op == "qk_norm":
                if q_ref is None or k_ref is None:
                    issues.append("qk_norm missing q/k inputs at layer=%d" % layer)
                    continue
                qk_out = add_op(
                    logical_op="qk_norm",
                    kernel_id=FORWARD_KERNEL_BY_OP["qk_norm"],
                    section="body",
                    layer=layer,
                    inputs={"q": q_ref, "k": k_ref},
                    output_specs={"q": "fp32", "k": "fp32"}
                )
                q_ref = qk_out["q"]
                k_ref = qk_out["k"]
            elif raw_op == "rope_qk":
                if q_ref is None or k_ref is None:
                    issues.append("rope_qk missing q/k inputs at layer=%d" % layer)
                    continue
                rope_out = add_op(
                    logical_op="rope_qk",
                    kernel_id=FORWARD_KERNEL_BY_OP["rope_qk"],
                    section="body",
                    layer=layer,
                    inputs={"q": q_ref, "k": k_ref},
                    output_specs={"q": "fp32", "k": "fp32"}
                )
                q_ref = rope_out["q"]
                k_ref = rope_out["k"]
            elif raw_op in ("attn", "attn_sliding"):
                if q_ref is None or k_ref is None or v_ref is None:
                    issues.append("attention missing q/k/v inputs at layer=%d" % layer)
                    continue
                op_name = "attn_sliding" if raw_op == "attn_sliding" else "attn"
                kid = FORWARD_KERNEL_BY_OP[op_name]
                attn_out = add_op(
                    logical_op=op_name,
                    kernel_id=kid,
                    section="body",
                    layer=layer,
                    inputs={"q": q_ref, "k": k_ref, "v": v_ref},
                    output_specs={"out": "fp32"}
                )
                current_main = attn_out["out"]
            elif raw_op == "out_proj":
                op_out = add_op(
                    logical_op="out_proj",
                    kernel_id=FORWARD_KERNEL_BY_OP["out_proj"],
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
                    kernel_id=FORWARD_KERNEL_BY_OP["residual_add"],
                    section="body",
                    layer=layer,
                    inputs={"a": current_main, "b": residual_slot},
                    output_specs={"out": "fp32"}
                )
                current_main = res_out["out"]
            elif raw_op == "mlp_gate_up":
                mlp_up = add_op(
                    logical_op="mlp_gate_up",
                    kernel_id=FORWARD_KERNEL_BY_OP["mlp_gate_up"],
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
                    kernel_id=FORWARD_KERNEL_BY_OP[logical],
                    section="body",
                    layer=layer,
                    inputs={"input": current_main},
                    output_specs={"out": "fp32"}
                )
                current_main = act_out["out"]
            elif raw_op == "mlp_down":
                down_out = add_op(
                    logical_op="mlp_down",
                    kernel_id=FORWARD_KERNEL_BY_OP["mlp_down"],
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
                kernel_id=FORWARD_KERNEL_BY_OP["rmsnorm"],
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
                kernel_id=FORWARD_KERNEL_BY_OP["logits"],
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
        "config": config,
        "template_name": template.get("name", "unknown"),
        "num_layers": num_layers,
        "ops": ops,
        "tensors": tensors,
        "stats": {
            "forward_ops": len([o for o in ops if o.get("phase") == "forward"]),
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
    ap.add_argument("--strict", action="store_true", help="Fail on unresolved weights/save-for-backward")
    ap.add_argument("--report-out", default=None, help="Optional report JSON path")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    grad_rules_path = Path(args.grad_rules)

    manifest = _load_json(manifest_path)
    registry = _load_json(KERNEL_REGISTRY_PATH)
    bindings_doc = _load_json(KERNEL_BINDINGS_PATH)
    grad_rules = _load_json(grad_rules_path)

    ir1 = build_ir1_train(
        manifest=manifest,
        registry=registry,
        bindings_doc=bindings_doc,
        grad_rules=grad_rules,
        max_layers=args.max_layers,
        strict=args.strict
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
