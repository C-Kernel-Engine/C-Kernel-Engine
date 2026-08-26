#!/usr/bin/env python3
from __future__ import annotations

"""Inspect a HF-style model config and summarize the CK v8 bring-up contract.

This is intentionally config-first.  It does not download weights and it does
not pretend unsupported architectures are compatible.  The output is a small
JSON report that says which layer families and kernels CK can already lower,
and which kernels/templates must be added before safetensors->BUMP conversion
should be attempted.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

SUPPORTED_DENSE_ARCHES = {"llama", "qwen2", "qwen3", "gemma3"}
SUPPORTED_HYBRID_ARCHES = {
    "qwen35", "gemma4", "nemotron_h", "kimi_vl", "instella_moe",
}


def _load_json_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").replace("Infinity", "1e100")
    return json.loads(text)


def _load_config(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "config.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return _load_json_file(path)


def _model_root(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.name == "config.json":
        return path.parent
    return path.parent


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config") if isinstance(config.get("text_config"), dict) else config


def _infer_arch(config: dict[str, Any]) -> str:
    text = _text_config(config)
    model_type = str(text.get("model_type") or config.get("model_type") or "").lower()
    architectures = " ".join(str(x).lower() for x in config.get("architectures", []))
    # Instella deliberately publishes model_type=deepseek_v3 because it reuses
    # Transformers' DeepSeek-V3 blocks.  Its architecture class is authoritative:
    # gated MLA and FarSkip two-stream residuals are not stock DeepSeek-V3.
    if "instellamoe" in architectures or "instella_moe" in architectures:
        return "instella_moe"
    if "qwen3_5" in model_type or "qwen3.5" in model_type or "qwen35" in model_type:
        return "qwen35"
    if "nemotron_h" in model_type or "nemotronh" in architectures:
        return "nemotron_h"
    if "kimi_vl" in model_type or "kimivl" in architectures or "kimi" in architectures:
        return "kimi_vl"
    if "cohere" in model_type or "cohere" in architectures or "command" in model_type:
        return "cohere"
    if "gemma" in model_type and "4" in model_type:
        return "gemma4"
    if "gemma" in model_type and "3" in model_type:
        return "gemma3"
    if "qwen3" in model_type:
        return "qwen3"
    if "qwen2" in model_type or model_type == "qwen":
        return "qwen2"
    if "llama" in model_type:
        return "llama"
    return model_type or "unknown"


def _parse_nemotron_h_pattern(pattern: str, num_layers: int) -> list[str]:
    """Parse Nemotron-H per-layer pattern characters into CK layer labels.

    NVIDIA's config code defines the pattern as one character per layer:
    M = Mamba2, * = attention, - = dense MLP, E = MoE.  The hyphen is a layer
    kind, not a separator.
    """
    mapping = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}
    chars = [ch for ch in str(pattern).strip() if not ch.isspace()]
    kinds = [mapping.get(ch, "unknown") for ch in chars]
    if num_layers > 0 and len(kinds) != num_layers:
        kinds.extend(["unspecified"] * max(0, num_layers - len(kinds)))
        kinds = kinds[:num_layers]
    return kinds


def _generic_dense_layers(config: dict[str, Any]) -> list[str]:
    text = _text_config(config)
    n = int(text.get("num_hidden_layers") or 0)
    return ["attention"] * n


def _qwen35_layers(config: dict[str, Any]) -> list[str]:
    text = _text_config(config)
    layer_types = [str(x) for x in text.get("layer_types", [])]
    if layer_types:
        return ["attention" if x == "full_attention" else "deltanet" for x in layer_types]
    n = int(text.get("num_hidden_layers") or 0)
    interval = int(text.get("full_attention_interval") or 4)
    return ["attention" if (i + 1) % interval == 0 else "deltanet" for i in range(n)]


def _gemma4_layers(config: dict[str, Any]) -> list[str]:
    text = _text_config(config)
    layer_kinds = text.get("layer_kinds") or config.get("layer_kinds")
    if isinstance(layer_kinds, list):
        return [str(x) for x in layer_kinds]
    n = int(text.get("num_hidden_layers") or 0)
    return ["attention_or_sliding_attention"] * n


def _kimi_vl_layers(config: dict[str, Any]) -> list[str]:
    text = _text_config(config)
    n = int(text.get("num_hidden_layers") or 0)
    first_dense = int(text.get("first_k_dense_replace") or 0)
    moe_freq = max(1, int(text.get("moe_layer_freq") or 1))
    kinds: list[str] = []
    for layer in range(n):
        if layer < first_dense or (layer - first_dense) % moe_freq != 0:
            kinds.append("mla_dense_mlp")
        else:
            kinds.append("mla_moe")
    return kinds


def _instella_moe_layers(config: dict[str, Any]) -> list[str]:
    text = _text_config(config)
    n = int(text.get("num_hidden_layers") or 0)
    first_dense = int(text.get("first_k_dense_replace") or 0)
    moe_freq = max(1, int(text.get("moe_layer_freq") or 1))
    kinds: list[str] = []
    for layer in range(n):
        if layer < first_dense or (layer - first_dense) % moe_freq != 0:
            kinds.append("mla_gated_dense_mlp")
        elif bool(text.get("farskip", False)):
            kinds.append("mla_gated_farskip_moe")
        else:
            kinds.append("mla_gated_moe")
    return kinds


def _layer_kinds(arch: str, config: dict[str, Any]) -> list[str]:
    text = _text_config(config)
    if arch == "nemotron_h":
        return _parse_nemotron_h_pattern(str(text.get("hybrid_override_pattern") or ""), int(text.get("num_hidden_layers") or 0))
    if arch == "qwen35":
        return _qwen35_layers(config)
    if arch == "gemma4":
        return _gemma4_layers(config)
    if arch == "kimi_vl":
        return _kimi_vl_layers(config)
    if arch == "instella_moe":
        return _instella_moe_layers(config)
    return _generic_dense_layers(config)


def _required_ops(arch: str, config: dict[str, Any], layer_kinds: list[str]) -> list[str]:
    ops = {"embedding", "rmsnorm", "matmul", "residual_add", "logits"}
    if any(kind in {"attention", "full_attention", "sliding_attention", "attention_or_sliding_attention"} for kind in layer_kinds):
        ops.update({"attention", "rope"})
    if any(kind.startswith("mla_") for kind in layer_kinds):
        ops.update({
            "mla_attention",
            "rope",
            "kv_lora_decompress",
            "partial_rope_concat",
        })
    if any(kind == "deltanet" for kind in layer_kinds):
        ops.update({
            "recurrent_dt_gate",
            "recurrent_conv_state_update",
            "ssm_conv1d",
            "recurrent_qk_l2_norm",
            "gated_deltanet",
            "recurrent_norm_gate",
        })
    if any(kind == "mamba" for kind in layer_kinds):
        ops.update({
            "mamba_in_proj_split",
            "mamba_conv1d_state_update",
            "mamba_dt_softplus",
            "mamba_selective_scan",
            "mamba_rmsnorm_gate",
            "mamba_out_proj",
        })
    if any(kind == "moe" for kind in layer_kinds):
        ops.update({
            "group_limited_topk_router",
            "moe_relu2_expert_mlp",
            "shared_expert_mlp",
        })
    if any(kind in {"mla_moe", "mla_gated_moe", "mla_gated_farskip_moe"} for kind in layer_kinds):
        ops.update({
            "group_limited_topk_router",
            "sigmoid_router_scores",
            "moe_swiglu_expert_mlp",
            "shared_swiglu_expert_mlp",
        })
    if any(kind.startswith("mla_gated_") for kind in layer_kinds):
        ops.update({"attention_gate_projection", "attention_gate_sigmoid_mul"})
    if any(kind == "mla_gated_farskip_moe" for kind in layer_kinds):
        # The composite provider emits both residual streams.  Their persistent
        # connectivity is a circuit edge/lifetime contract, not a second math op.
        ops.add("farskip_routed_shared_combine")
    act = str(_text_config(config).get("mlp_hidden_act") or _text_config(config).get("hidden_act") or "").lower()
    if act == "relu2":
        ops.add("relu2_mlp")
    else:
        ops.add("swiglu_or_geglu")
    if arch == "gemma4":
        ops.update({"gemma4_per_layer_embed", "gemma4_final_logit_softcap"})
    if arch == "kimi_vl":
        ops.update({"moonvit_encoder", "moonvit_projector", "media_placeholder_merge", "tiktoken_tokenizer"})
    return sorted(ops)


def _missing_ops(arch: str, required_ops: list[str]) -> list[str]:
    missing = []
    mamba_decode_covered = {
        "mamba_in_proj_split",
        "mamba_conv1d_state_update",
        "mamba_dt_softplus",
        "mamba_rmsnorm_gate",
        "mamba_out_proj",
        "mamba_selective_scan",
    }
    for op in required_ops:
        if op.startswith("mamba_") and op not in mamba_decode_covered:
            missing.append(op)
    if arch == "cohere":
        missing.append("cohere_tensor_name_mapping_audit")
    if arch == "kimi_vl":
        missing.append("moonvit_bridge_contract")
    if arch not in SUPPORTED_DENSE_ARCHES | SUPPORTED_HYBRID_ARCHES:
        missing.append("v8_template_contract")
        missing.append("safetensors_to_bump_mapping")
    return sorted(set(missing))


def _load_registry_ops(registry_path: Path | None = None) -> set[str]:
    path = registry_path or (REPO_ROOT / "version" / "v8" / "kernel_maps" / "KERNEL_REGISTRY.json")
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    return {str(k.get("op")) for k in registry.get("kernels", []) if k.get("op")}


def _registry_requirements_for_op(op: str) -> list[list[str]]:
    """Return OR-groups of registry ops that satisfy a conceptual model op."""
    mapping: dict[str, list[list[str]]] = {
        "embedding": [["embedding"]],
        "rmsnorm": [["rmsnorm"]],
        "matmul": [["gemv"], ["gemm"]],
        "residual_add": [["residual_add"]],
        "logits": [["gemv"], ["gemm"]],
        "attention": [["attention"]],
        "mla_attention": [["attention"]],
        "rope": [["rope"]],
        "kv_lora_decompress": [["kv_lora_decompress"]],
        "partial_rope_concat": [["partial_rope_concat"]],
        "recurrent_dt_gate": [["recurrent_dt_gate"]],
        "recurrent_conv_state_update": [["recurrent_conv_state_update"]],
        "ssm_conv1d": [["ssm_conv1d"]],
        "recurrent_qk_l2_norm": [["recurrent_qk_l2_norm"]],
        "gated_deltanet": [["gated_deltanet"]],
        "recurrent_norm_gate": [["recurrent_norm_gate"]],
        "mamba_in_proj_split": [["mamba_in_proj_split"]],
        "mamba_conv1d_state_update": [["mamba_conv1d_state_update"]],
        "mamba_dt_softplus": [["mamba_dt_softplus"]],
        "mamba_selective_scan": [["mamba_selective_scan"]],
        "mamba_rmsnorm_gate": [["mamba_rmsnorm_gate"]],
        "mamba_out_proj": [["gemv"], ["gemm"]],
        "group_limited_topk_router": [["group_limited_topk_router"]],
        "sigmoid_router_scores": [["attn_gate_sigmoid_mul"]],
        "moe_relu2_expert_mlp": [["moe_relu2_expert_mlp"]],
        "shared_expert_mlp": [["shared_relu2_expert_mlp"], ["shared_swiglu_expert_mlp"]],
        "shared_relu2_expert_mlp": [["shared_relu2_expert_mlp"]],
        "moe_swiglu_expert_mlp": [["moe_swiglu_expert_mlp"]],
        "shared_swiglu_expert_mlp": [["shared_swiglu_expert_mlp"]],
        "attention_gate_projection": [["gemv"], ["gemm"]],
        "attention_gate_sigmoid_mul": [["attn_gate_sigmoid_mul"]],
        "farskip_two_stream_residual": [["farskip_two_stream_residual"]],
        "farskip_routed_shared_combine": [["farskip_routed_shared_combine"]],
        "relu2_mlp": [["relu2"]],
        "swiglu_or_geglu": [["swiglu"], ["geglu"], ["gelu"]],
        "gemma4_per_layer_embed": [["gemma4_per_layer_embed"]],
        "gemma4_final_logit_softcap": [["final_logit_softcap"]],
    }
    return mapping.get(op, [[op]])


def _kernel_registry_report(required_ops: list[str]) -> dict[str, Any]:
    registry_ops = _load_registry_ops()
    supported: list[str] = []
    missing: list[dict[str, Any]] = []
    for op in required_ops:
        groups = _registry_requirements_for_op(op)
        unsatisfied = [group for group in groups if not any(candidate in registry_ops for candidate in group)]
        if not unsatisfied:
            supported.append(op)
        else:
            missing.append(
                {
                    "op": op,
                    "registry_candidates": groups,
                    "missing_candidate_groups": unsatisfied,
                }
            )
    return {
        "supported_ops": supported,
        "missing_kernel_ops": missing,
        "registry_loaded": bool(registry_ops),
    }


def _template_body_ops_for_kind(kind: str, arch: str) -> list[str]:
    if kind in {"attention", "full_attention"}:
        return ["attn_norm", "q_proj", "k_proj", "v_proj", "qk_norm", "rope_qk", "attn", "out_proj", "residual_add", "ffn_norm", "mlp_gate_up", "silu_mul", "mlp_down", "residual_add"]
    if kind == "sliding_attention":
        return ["attn_norm", "q_proj", "k_proj", "v_proj", "qk_norm", "rope_qk", "attn_sliding", "out_proj", "residual_add", "ffn_norm", "mlp_gate_up", "silu_mul", "mlp_down", "residual_add"]
    if kind == "attention_or_sliding_attention":
        return ["attn_norm", "q_proj", "k_proj", "v_proj", "qk_norm", "rope_qk", "attn_or_attn_sliding", "out_proj", "residual_add", "ffn_norm", "mlp_gate_up", "silu_mul", "mlp_down", "residual_add"]
    if kind == "deltanet":
        return ["attn_norm", "recurrent_qkv_proj", "recurrent_dt_gate", "recurrent_conv_state_update", "recurrent_qk_l2_norm", "recurrent_core", "recurrent_norm_gate", "recurrent_out_proj", "residual_add", "ffn_norm", "mlp_gate_up", "silu_mul", "mlp_down", "residual_add"]
    if kind == "mamba":
        return ["attn_norm", "mamba_in_proj", "mamba_in_proj_split", "mamba_conv1d_silu", "mamba_dt_softplus", "mamba_selective_scan", "mamba_rmsnorm_gate", "mamba_out_proj", "residual_add"]
    if kind == "moe":
        return ["attn_norm", "q_proj", "k_proj", "v_proj", "rope_qk", "attn", "out_proj", "residual_add", "ffn_norm", "moe_router", "group_limited_topk_router", "moe_relu2_expert_mlp", "shared_relu2_expert_mlp", "residual_add"]
    if kind == "mla_dense_mlp":
        return ["attn_norm", "q_proj", "kv_a_proj", "kv_a_layernorm", "kv_lora_decompress", "partial_rope_concat", "mla_attention", "out_proj", "residual_add", "ffn_norm", "mlp_gate_up", "silu_mul", "mlp_down", "residual_add"]
    if kind == "mla_moe":
        return ["attn_norm", "q_proj", "kv_a_proj", "kv_a_layernorm", "kv_lora_decompress", "partial_rope_concat", "mla_attention", "out_proj", "residual_add", "ffn_norm", "moe_router", "group_limited_topk_router", "moe_swiglu_expert_mlp", "shared_swiglu_expert_mlp", "residual_add"]
    if kind == "mla_gated_dense_mlp":
        return ["attn_norm", "q_proj", "kv_a_proj", "kv_a_layernorm", "kv_lora_decompress", "partial_rope_concat_interleaved_yarn", "mla_attention", "attention_gate_projection", "attention_gate_sigmoid_mul", "out_proj", "residual_add", "ffn_norm", "mlp_gate_up", "silu_mul", "mlp_down", "residual_add"]
    if kind == "mla_gated_moe":
        return ["attn_norm", "q_proj", "kv_a_proj", "kv_a_layernorm", "kv_lora_decompress", "partial_rope_concat_interleaved_yarn", "mla_attention", "attention_gate_projection", "attention_gate_sigmoid_mul", "out_proj", "residual_add", "ffn_norm", "moe_router", "group_limited_topk_router", "moe_swiglu_expert_mlp", "shared_swiglu_expert_mlp", "residual_add"]
    if kind == "mla_gated_farskip_moe":
        return ["farskip_select_attention_stream", "attn_norm", "q_proj", "kv_a_proj", "kv_a_layernorm", "kv_lora_decompress", "partial_rope_concat_interleaved_yarn", "mla_attention", "attention_gate_projection", "attention_gate_sigmoid_mul", "out_proj", "farskip_attention_residual", "farskip_select_mlp_stream", "ffn_norm", "moe_router", "group_limited_topk_router", "moe_swiglu_expert_mlp", "farskip_routed_shared_combine"]
    if arch == "gemma4" and kind in {"sliding_attention_shared_kv", "full_attention_shared_kv"}:
        attn = "attn_sliding_shared_kv" if kind.startswith("sliding") else "attn_shared_kv"
        return ["attn_norm", "q_proj", "q_norm", "rope_q", attn, "out_proj", "post_attention_norm", "residual_add", "ffn_norm", "mlp_gate_up", "gelu", "mlp_down", "post_ffn_norm", "residual_add", "gemma4_per_layer_embed"]
    return ["attn_norm", "q_proj", "k_proj", "v_proj", "rope_qk", "attn", "out_proj", "residual_add", "ffn_norm", "mlp_gate_up", "silu_mul", "mlp_down", "residual_add"]


def _template_candidate_name(arch: str) -> str | None:
    candidates = {
        "llama": "llama.json",
        "qwen2": "qwen2.json",
        "qwen3": "qwen3.json",
        "qwen35": "qwen35.json",
        "gemma3": "gemma3.json",
        "gemma4": "gemma4.json",
        "nemotron_h": "nemotron_h.json",
        "kimi_vl": "kimi_vl.json",
        "instella_moe": "instella_moe.json",
    }
    name = candidates.get(arch)
    if name and (REPO_ROOT / "version" / "v8" / "circuits" / name).exists():
        return name
    return name


def suggest_template(config: dict[str, Any]) -> dict[str, Any]:
    arch = _infer_arch(config)
    layer_kinds = _layer_kinds(arch, config)
    unique_kinds = sorted(set(layer_kinds))
    text = _text_config(config)
    template_name = _template_candidate_name(arch)
    existing = bool(template_name and (REPO_ROOT / "version" / "v8" / "circuits" / template_name).exists())
    header = ["bpe_tokenizer", "dense_embedding_lookup"]
    if arch == "gemma4":
        header.append("gemma4_per_layer_prepare")
    if arch == "kimi_vl":
        header = ["tiktoken_tokenizer", "dense_embedding_lookup"]
    body_by_kind = {kind: _template_body_ops_for_kind(kind, arch) for kind in unique_kinds}
    footer = ["final_rmsnorm", "weight_tying", "logits"]
    if arch == "gemma4":
        footer = ["final_rmsnorm", "final_logit_softcap", "weight_tying", "logits"]
    modalities = ["text"]
    if isinstance(config.get("vision_config"), dict):
        modalities.append("vision")
    if isinstance(config.get("audio_config"), dict):
        modalities.append("audio")
    return {
        "candidate_template": template_name,
        "candidate_template_exists": existing,
        "confidence": "existing_template" if existing else "sketch",
        "modalities": modalities,
        "sequence": ["decoder"],
        "block_sketch": {
            "header": header,
            "body": {
                "kind_config_key": "layer_kinds" if unique_kinds else None,
                "ops_by_kind": body_by_kind,
            },
            "footer": footer,
        },
        "shape_hints": {
            "hidden_size": int(text.get("hidden_size") or 0),
            "intermediate_size": int(text.get("intermediate_size") or 0),
            "num_layers": int(text.get("num_hidden_layers") or len(layer_kinds) or 0),
            "num_attention_heads": int(text.get("num_attention_heads") or 0),
            "num_key_value_heads": int(text.get("num_key_value_heads") or 0),
        },
    }


def _find_safetensors_index(root: Path) -> Path | None:
    candidates = [
        root / "model.safetensors.index.json",
        root / "model.safetensors.index.json".replace("model.", ""),
    ]
    candidates.extend(sorted(root.glob("*.safetensors.index.json")))
    for path in candidates:
        if path.exists():
            return path
    return None


def _weights_audit(root: Path, arch: str) -> dict[str, Any]:
    index = _find_safetensors_index(root)
    if index is None:
        return {
            "status": "skipped",
            "reason": "no model.safetensors.index.json found",
        }
    import audit_safetensors_index_v8  # type: ignore

    report = audit_safetensors_index_v8.audit_index(index, arch=arch)
    return {
        "status": "pass" if report.get("status", "pass") != "fail" else "fail",
        "index": str(index),
        "families": report.get("families", {}),
        "layer_count": report.get("layer_count", 0),
        "tensor_count": report.get("tensor_count", 0),
        "shard_count": report.get("shard_count", 0),
        "model_map_status": report.get("model_map_status"),
        "required_tensor_patterns": report.get("required_tensor_patterns"),
    }


def inspect_config(config: dict[str, Any], *, model_root: Path | None = None, include_weights: bool = False) -> dict[str, Any]:
    text = _text_config(config)
    arch = _infer_arch(config)
    layer_kinds = _layer_kinds(arch, config)
    required_ops = _required_ops(arch, config, layer_kinds)
    missing_ops = _missing_ops(arch, required_ops)
    kernel_report = _kernel_registry_report(required_ops)
    template_suggestion = suggest_template(config)
    status = "supported" if not missing_ops else "bringup_required"
    report = {
        "arch": arch,
        "model_type": text.get("model_type") or config.get("model_type"),
        "architectures": config.get("architectures", []),
        "status": status,
        "num_layers": int(text.get("num_hidden_layers") or len(layer_kinds) or 0),
        "hidden_size": int(text.get("hidden_size") or 0),
        "intermediate_size": int(text.get("intermediate_size") or 0),
        "num_attention_heads": int(text.get("num_attention_heads") or 0),
        "num_key_value_heads": int(text.get("num_key_value_heads") or 0),
        "layer_kind_counts": dict(sorted(Counter(layer_kinds).items())),
        "required_ops": required_ops,
        "missing_ops": missing_ops,
        "kernel_registry": kernel_report,
        "template_suggestion": template_suggestion,
        "notes": _notes(arch, config, layer_kinds, missing_ops),
    }
    if include_weights:
        report["weights_audit"] = _weights_audit(model_root or Path("."), arch)
    return report


def _notes(arch: str, config: dict[str, Any], layer_kinds: list[str], missing_ops: list[str]) -> list[str]:
    text = _text_config(config)
    notes: list[str] = []
    if arch == "nemotron_h":
        pattern_chars = [ch for ch in str(text.get("hybrid_override_pattern") or "").strip() if not ch.isspace()]
        if int(text.get("num_hidden_layers") or 0) and len(pattern_chars) != int(text.get("num_hidden_layers") or 0):
            notes.append(
                f"hybrid_override_pattern describes {len(pattern_chars)} layers but num_hidden_layers={text.get('num_hidden_layers')}; "
                "confirm whether the pattern repeats or whether checkpoint code expands it."
            )
        notes.append("Nemotron-H is a hybrid Mamba/attention decoder; CK DeltaNet kernels are not a substitute for Mamba selective scan.")
        notes.append(f"hybrid_override_pattern={text.get('hybrid_override_pattern')}")
        notes.append(f"mamba heads={text.get('mamba_num_heads')} head_dim={text.get('mamba_head_dim')} state={text.get('ssm_state_size')} conv_kernel={text.get('conv_kernel')}")
        notes.append("Mamba2 decode split/conv/dt/state-update/gated-norm reference kernels exist; full selective scan has a scalar reference; chunk-optimized scan remains future performance work.")
        notes.append(f"MLP activation={text.get('mlp_hidden_act')}; ReLU2 primitive exists, dense/shared MLP lowering uses matmul -> relu2 -> matmul.")
    if arch == "cohere":
        notes.append("Cohere Command configs are gated in this environment; require config/weight access before mapping tensor names.")
        notes.append("Likely first target is dense decoder mapping/audit before any new math kernels.")
    if arch == "kimi_vl":
        notes.append("Kimi-VL text uses DeepSeek-V3-style MLA: q_proj plus compressed KV, kv_a_layernorm, kv_b_proj, qk_nope/qk_rope split, and RoPE only on the rope slice.")
        notes.append(
            f"MoE policy: first_k_dense_replace={text.get('first_k_dense_replace')} "
            f"moe_layer_freq={text.get('moe_layer_freq')} routed_experts={text.get('n_routed_experts')} "
            f"top_k={text.get('num_experts_per_tok')} shared_experts={text.get('n_shared_experts')} "
            f"router={text.get('scoring_func')}/{text.get('topk_method')} scale={text.get('routed_scaling_factor')}."
        )
        notes.append("The text decoder has safetensors-to-BUMP mapping, an explicit TikToken chat contract, MLA lowering, and routed/shared SwiGLU execution. Real-weight text smoke is coherent and repeatable; MoonViT encoder/projector lowering and the multimodal bridge remain open.")
        if isinstance(config.get("vision_config"), dict):
            vision = config["vision_config"]
            notes.append(
                f"Vision path is MoonViT: layers={vision.get('num_hidden_layers')} hidden={vision.get('hidden_size')} "
                f"patch={vision.get('patch_size')} merge={vision.get('merge_kernel_size')}; defer until text MLA+MoE parity is stamped."
            )
    if arch == "instella_moe":
        notes.append("Instella-MoE reuses DeepSeek-V3 MLA primitives but changes the graph with a learned sigmoid attention gate and FarSkip two-stream residual connectivity; it must not resolve to a stock DeepSeek or Kimi circuit.")
        notes.append(
            f"MoE policy: first_k_dense_replace={text.get('first_k_dense_replace')} "
            f"moe_layer_freq={text.get('moe_layer_freq')} routed_experts={text.get('n_routed_experts')} "
            f"top_k={text.get('num_experts_per_tok')} shared_experts={text.get('n_shared_experts')} "
            f"router={text.get('scoring_func')}/{text.get('topk_method')} scale={text.get('routed_scaling_factor')}."
        )
        notes.append("The executable circuit owns gated-MLA wiring, FarSkip main/routed-free lifetimes, and contiguous-position YaRN cache initialization. Real-weight PyTorch X-ray parity remains the promotion gate beyond config-level support.")
    if not missing_ops:
        notes.append("Config-level contract has no known missing op, but weight-name audit and hidden parity are still required.")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect a model config and report CK v8 compatibility contract")
    ap.add_argument("config", type=Path, help="config.json or model directory")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--weights-audit", action="store_true", help="audit safetensors index tensor families if present")
    ap.add_argument("--suggest-template", action="store_true", help="include candidate template/circuit sketch (currently included by default)")
    args = ap.parse_args()

    report = inspect_config(_load_config(args.config), model_root=_model_root(args.config), include_weights=args.weights_audit)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "supported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
