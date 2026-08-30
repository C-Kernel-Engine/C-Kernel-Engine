#!/usr/bin/env python3
from __future__ import annotations

"""Convert HF safetensors language weights into v8 BUMPWGT5 artifacts.

This is a BUMP-first importer.  GGUF and safetensors are source formats; CK
runtime consumes weights.bump + sidecars.  Supported targets map source tensor
names into CK-internal manifest names so existing v8 IR/codegen can consume
safetensors and GGUF through the same BUMP runtime contract.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_gguf_to_bump_v8 import (  # type: ignore
    BUMP_META_FOOTER_MAGIC,
    BUMP_VERSION_V5,
    CACHE_ALIGN,
    CK_DT_BF16,
    CK_DT_FP16,
    CK_DT_FP32,
    DATA_START,
    EXT_METADATA_SIZE,
    HEADER_SIZE,
    _canonical_json_bytes,
    _inject_runtime_config_defaults,
    apply_model_contract_overrides,
    build_bumpv5_metadata,
    build_qwen35_execution_plan,
    calculate_manifest_hash,
    calculate_metadata_hash,
    calculate_template_hash,
    inspect_tokenizer_json,
    load_template_for_arch,
    load_tokenizer_json,
    _apply_tokenizer_contract_overrides,
    write_bumpv5_footer,
)


def _effective_attention_scale(
    head_dim: int,
    rope_scaling: dict[str, Any] | None = None,
) -> float:
    if head_dim <= 0:
        raise ValueError("attention head_dim must be positive")
    scale = 1.0 / math.sqrt(float(head_dim))
    rope = rope_scaling if isinstance(rope_scaling, dict) else {}
    rope_type = str(rope.get("rope_type") or rope.get("type") or "default").lower()
    mscale_all_dim = float(rope.get("mscale_all_dim") or 0.0)
    factor = float(rope.get("factor") or 1.0)
    if rope_type not in {"default", "none"} and mscale_all_dim and factor > 1.0:
        mscale = 0.1 * mscale_all_dim * math.log(factor) + 1.0
        scale *= mscale * mscale
    return scale

DTYPE_TO_CK = {
    "F32": ("fp32", CK_DT_FP32, 4),
    "BF16": ("bf16", CK_DT_BF16, 2),
    "F16": ("fp16", CK_DT_FP16, 2),
    "float32": ("fp32", CK_DT_FP32, 4),
    "bfloat16": ("bf16", CK_DT_BF16, 2),
    "float16": ("fp16", CK_DT_FP16, 2),
}

SAFETENSORS_CK_MAP_PATH = SCRIPT_DIR.parent / "model_maps" / "safetensors_ck_map.json"
_SAFETENSORS_CK_MAP_CACHE: dict[str, Any] | None = None


def align_up(n: int, a: int = CACHE_ALIGN) -> int:
    return ((int(n) + int(a) - 1) // int(a)) * int(a)


@dataclass(frozen=True)
class TensorRef:
    ck_name: str
    source_names: tuple[str, ...]
    dtype: str | None = None
    role: str | None = None
    synth: str | None = None
    shape: tuple[int, ...] | None = None
    transform: str | None = None


@dataclass
class HeaderTensor:
    name: str
    dtype: str
    shape: list[int]
    shard: Path


class HashingWriter:
    def __init__(self, f):
        self.f = f
        self.h = hashlib.sha256()
        self.bytes_written = 0

    def write(self, data: bytes) -> None:
        self.f.write(data)
        self.h.update(data)
        self.bytes_written += len(data)

    def digest(self) -> bytes:
        return self.h.digest()


def _load_safetensors_headers(model_dir: Path) -> dict[str, HeaderTensor]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("safetensors package is required") from exc

    index = model_dir / "model.safetensors.index.json"
    out: dict[str, HeaderTensor] = {}
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        shards = sorted({str(v) for v in weight_map.values()})
        for shard_name in shards:
            shard = model_dir / shard_name
            with safe_open(shard, framework="pt") as sf:
                for key in sf.keys():
                    sl = sf.get_slice(key)
                    out[key] = HeaderTensor(key, str(sl.get_dtype()), [int(x) for x in sl.get_shape()], shard)
        return out

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"No safetensors files found in {model_dir}")
    for shard in files:
        with safe_open(shard, framework="pt") as sf:
            for key in sf.keys():
                sl = sf.get_slice(key)
                out[key] = HeaderTensor(key, str(sl.get_dtype()), [int(x) for x in sl.get_shape()], shard)
    return out


def _load_tensor(model_dir: Path, headers: dict[str, HeaderTensor], name: str):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("safetensors package is required") from exc
    h = headers[name]
    with safe_open(h.shard, framework="pt") as sf:
        return sf.get_tensor(name).detach().cpu().contiguous()


def _torch_to_bytes(t, dtype_policy: str) -> tuple[bytes, str, list[int]]:
    import torch

    shape = [int(x) for x in t.shape]
    if dtype_policy == "fp32":
        arr = t.to(dtype=torch.float32).numpy().astype(np.float32, copy=False)
        return arr.tobytes(order="C"), "fp32", shape
    if dtype_policy == "bf16":
        u16 = t.to(dtype=torch.bfloat16).view(torch.uint16).numpy()
        return u16.tobytes(order="C"), "bf16", shape
    if dtype_policy == "fp16":
        u16 = t.to(dtype=torch.float16).view(torch.uint16).numpy()
        return u16.tobytes(order="C"), "fp16", shape
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes(order="C"), "bf16", shape
    if t.dtype == torch.float16:
        return t.numpy().view(np.uint16).tobytes(order="C"), "fp16", shape
    arr = t.to(dtype=torch.float32).numpy().astype(np.float32, copy=False)
    return arr.tobytes(order="C"), "fp32", shape


def _dtype_code(dtype_name: str) -> int:
    if dtype_name == "fp32":
        return CK_DT_FP32
    if dtype_name == "bf16":
        return CK_DT_BF16
    if dtype_name == "fp16":
        return CK_DT_FP16
    raise ValueError(f"unsupported dtype {dtype_name}")


def _header_dtype_to_ck(dtype: str) -> tuple[str, int]:
    row = DTYPE_TO_CK.get(str(dtype))
    if row:
        return row[0], row[2]
    return "fp32", 4


def _synth_bytes(kind: str, shape: tuple[int, ...], dtype_policy: str) -> tuple[bytes, str, list[int]]:
    n = int(np.prod(shape, dtype=np.int64))
    if kind == "zeros_fp32":
        return np.zeros(n, dtype=np.float32).tobytes(), "fp32", list(shape)
    if kind == "ones_fp32":
        return np.ones(n, dtype=np.float32).tobytes(), "fp32", list(shape)
    raise ValueError(f"unknown synth tensor kind {kind}")


def _hf_config(model_dir: Path) -> dict[str, Any]:
    p = model_dir / "config.json"
    if not p.exists():
        raise SystemExit(f"Missing config.json in {model_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_runtime_config_template(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "config" in data and isinstance(data["config"], dict):
        return dict(data["config"])
    return dict(data)



def _load_safetensors_ck_map() -> dict[str, Any]:
    global _SAFETENSORS_CK_MAP_CACHE
    if _SAFETENSORS_CK_MAP_CACHE is not None:
        return _SAFETENSORS_CK_MAP_CACHE
    if not SAFETENSORS_CK_MAP_PATH.exists():
        _SAFETENSORS_CK_MAP_CACHE = {"version": 0, "architectures": {}}
        return _SAFETENSORS_CK_MAP_CACHE
    data = json.loads(SAFETENSORS_CK_MAP_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{SAFETENSORS_CK_MAP_PATH}: expected JSON object")
    archs = data.get("architectures")
    if archs is None:
        data["architectures"] = {}
    elif not isinstance(archs, dict):
        raise SystemExit(f"{SAFETENSORS_CK_MAP_PATH}: architectures must be an object")
    _SAFETENSORS_CK_MAP_CACHE = data
    return data


def _safetensors_arch_contract(arch: str) -> dict[str, Any]:
    contracts = _load_safetensors_ck_map().get("architectures") or {}
    row = contracts.get(str(arch or "").lower())
    return row if isinstance(row, dict) else {}


def _apply_linear_weight_runtime_config(
    config: dict[str, Any], contract: dict[str, Any], linear_weight_dtype: str
) -> None:
    """Apply precision-sensitive runtime policy declared by the model map."""
    policies = contract.get("runtime_config_by_linear_weight_dtype")
    if policies is None:
        return
    if not isinstance(policies, dict):
        raise SystemExit(
            f"{SAFETENSORS_CK_MAP_PATH}: runtime_config_by_linear_weight_dtype "
            "must be an object"
        )
    selected = policies.get(str(linear_weight_dtype))
    if selected is None:
        selected = policies.get("default")
    if not isinstance(selected, dict):
        raise SystemExit(
            f"{SAFETENSORS_CK_MAP_PATH}: no runtime configuration for "
            f"linear weight dtype {linear_weight_dtype!r}"
        )
    for key, value in selected.items():
        if not isinstance(key, str) or not key:
            raise SystemExit(
                f"{SAFETENSORS_CK_MAP_PATH}: runtime configuration keys "
                "must be non-empty strings"
            )
        config[key] = value


def _apply_vision_numerics_profile(
    config: dict[str, Any], arch: str, profile: str
) -> None:
    if arch not in {"qwen3_vl_vision", "cohere_compass_vision"}:
        if profile != "pytorch_exact":
            raise SystemExit(
                f"--vision-numerics={profile} is only valid for a vision encoder"
            )
        return
    if profile == "pytorch_exact":
        config["vision_patch_frontend"] = "integrated_temporal2"
        config["vision_patch_projection_reduction_policy"] = (
            "pytorch_onednn_conv3d_exact"
        )
        config["vision_mrope_reduction_policy"] = "pytorch_mkl_exact"
        config["vision_layernorm_reduction_policy"] = "pytorch_welford_exact"
        config["vision_projection_reduction_policy"] = (
            "pytorch_onednn_brgemm_exact"
        )
        config["vision_attention_reduction_policy"] = "pytorch_amx_exact"
        config["vision_projector_activation_reduction_policy"] = (
            "pytorch_sleef_exact"
        )
        return
    config["vision_patch_frontend"] = "integrated_temporal2"
    config["vision_patch_projection_reduction_policy"] = "native_pair_dot"
    config["vision_mrope_reduction_policy"] = "portable_fp32_reference"
    config["vision_layernorm_reduction_policy"] = "pytorch_welford_exact"
    config["vision_projection_reduction_policy"] = "native_pair_dot"
    config["vision_attention_reduction_policy"] = "portable_tiled_sdpa"
    config["vision_projector_activation_reduction_policy"] = "portable_libm_erf"


def _contract_ignored_source_tensor(
    contract: dict[str, Any], name: str
) -> str | None:
    """Resolve artifact ownership exclusions from a safetensors model map."""
    rules = contract.get("ignored_source_tensors") or []
    if not isinstance(rules, list):
        raise SystemExit(
            f"{SAFETENSORS_CK_MAP_PATH}: ignored_source_tensors must be a list"
        )
    for rule in rules:
        if not isinstance(rule, dict):
            raise SystemExit(
                f"{SAFETENSORS_CK_MAP_PATH}: ignored source rules must be objects"
            )
        reason = rule.get("reason")
        exact = rule.get("exact")
        prefix = rule.get("prefix")
        if not isinstance(reason, str) or not reason:
            raise SystemExit(
                f"{SAFETENSORS_CK_MAP_PATH}: ignored source rules require a reason"
            )
        selectors = int(exact is not None) + int(prefix is not None)
        if selectors != 1:
            raise SystemExit(
                f"{SAFETENSORS_CK_MAP_PATH}: ignored source rules require "
                "exactly one of exact or prefix"
            )
        if exact is not None and name == str(exact):
            return reason
        if prefix is not None and name.startswith(str(prefix)):
            return reason
    return None


def _shape_symbol_value(name: str, config: dict[str, Any]) -> int:
    key = str(name)
    if key == "embed_dim":
        return int(config.get("embed_dim") or config.get("hidden_size") or 0)
    if key == "intermediate_size":
        return int(config.get("intermediate_size") or 0)
    if key == "q_dim":
        return int(config.get("num_heads") or config.get("num_attention_heads") or 0) * int(config.get("head_dim") or 0)
    if key == "kv_dim":
        return int(config.get("num_kv_heads") or config.get("num_key_value_heads") or config.get("num_heads") or 0) * int(config.get("head_dim") or 0)
    if key.isdigit():
        return int(key)
    raise SystemExit(f"Unsupported safetensors map shape symbol: {key}")


def _shape_from_spec(spec: Any, config: dict[str, Any]) -> tuple[int, ...]:
    if spec is None:
        return ()
    if not isinstance(spec, list):
        raise SystemExit(f"safetensors map shape must be a list, got {type(spec).__name__}")
    dims: list[int] = []
    for item in spec:
        if isinstance(item, int):
            dims.append(int(item))
            continue
        text = str(item).strip()
        if "*" in text:
            total = 1
            for part in text.split("*"):
                total *= _shape_symbol_value(part.strip(), config)
            dims.append(total)
        else:
            dims.append(_shape_symbol_value(text, config))
    return tuple(dims)


def _first_existing_from_patterns(headers: dict[str, HeaderTensor], patterns: Iterable[str], layer: int | None = None) -> str | None:
    for pattern in patterns:
        name = str(pattern)
        if layer is not None:
            name = name.replace("{L}", str(layer))
        if name in headers:
            return name
    return None


def _refs_from_safetensors_contract(arch: str, config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef] | None:
    contract = _safetensors_arch_contract(arch)
    refs_spec = contract.get("tensor_refs") if isinstance(contract, dict) else None
    if not isinstance(refs_spec, list):
        return None
    num_layers = int(config.get("num_layers") or config.get("num_hidden_layers") or 0)
    refs: list[TensorRef] = []
    for spec in refs_spec:
        if not isinstance(spec, dict):
            raise SystemExit(f"{SAFETENSORS_CK_MAP_PATH}: tensor_refs entries must be objects")
        target = str(spec.get("target") or "")
        if not target:
            raise SystemExit(f"{SAFETENSORS_CK_MAP_PATH}: tensor_refs entry missing target")
        layers: list[int | None]
        if "{L}" in target:
            layers = list(range(num_layers))
        else:
            layers = [None]
        for layer in layers:
            ck_name = target.replace("{L}", str(layer)) if layer is not None else target
            dtype = spec.get("dtype")
            dtype = str(dtype) if dtype is not None else None
            role = spec.get("role")
            role = str(role) if role is not None else None
            transform = spec.get("transform")
            transform = str(transform) if transform is not None else None
            synth = spec.get("synth")
            synth = str(synth) if synth is not None else None
            shape = _shape_from_spec(spec.get("shape"), config) if (synth or spec.get("fallback_synth")) else None
            sources_raw = spec.get("sources") or []
            if isinstance(sources_raw, str):
                sources_raw = [sources_raw]
            if not isinstance(sources_raw, list):
                raise SystemExit(f"{SAFETENSORS_CK_MAP_PATH}: sources for {ck_name} must be a list")
            if synth:
                refs.append(TensorRef(ck_name, (), dtype=dtype, role=role, synth=synth, shape=shape, transform=transform))
                continue
            combine_mode = str(spec.get("combine") or "").strip().lower()
            if combine_mode in {"concat", "concat_or_single"}:
                named_sources = [str(src).replace("{L}", str(layer)) if layer is not None else str(src) for src in sources_raw]
                if combine_mode == "concat_or_single" and named_sources and named_sources[0] in headers:
                    refs.append(TensorRef(ck_name, (named_sources[0],), dtype=dtype, role=role, transform=transform))
                    continue
                concat_sources = named_sources[1:] if combine_mode == "concat_or_single" else named_sources
                missing = [name for name in concat_sources if name not in headers]
                if missing or not concat_sources:
                    raise SystemExit(f"Missing required safetensors tensor for {ck_name}: tried {named_sources}")
                refs.append(TensorRef(ck_name, tuple(concat_sources), dtype=dtype, role=role, transform=transform))
                continue
            found = _first_existing_from_patterns(headers, (str(x) for x in sources_raw), layer)
            if found is not None:
                refs.append(TensorRef(ck_name, (found,), dtype=dtype, role=role, transform=transform))
                continue
            fallback = spec.get("fallback_synth")
            if fallback:
                refs.append(TensorRef(ck_name, (), dtype=dtype, role=role, synth=str(fallback), shape=shape, transform=transform))
                continue
            if bool(spec.get("optional", False)):
                continue
            raise SystemExit(f"Missing required safetensors tensor for {ck_name}: tried {sources_raw}")
    return refs

def _gemma4_text_refs(config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef]:
    num_layers = int(config.get("num_layers") or config.get("num_hidden_layers") or 0)
    embed_dim = int(config.get("embed_dim") or config.get("hidden_size") or 0)
    intermediate = int(config.get("intermediate_size") or 0)
    per_layer_dim = int(config.get("per_layer_dim") or 0)
    if num_layers <= 0 or embed_dim <= 0 or intermediate <= 0:
        raise SystemExit("Gemma4 config missing num_layers/embed_dim/intermediate_size")
    if per_layer_dim <= 0:
        # HF tensor shape is [vocab, layers, per_layer_dim] or equivalent in current checkpoints.
        h = headers.get("model.language_model.per_layer_projection_norm.weight")
        if h and h.shape:
            per_layer_dim = int(h.shape[-1])
        else:
            raise SystemExit("Gemma4 config missing per_layer_dim and cannot infer it")
        config["per_layer_dim"] = per_layer_dim

    refs: list[TensorRef] = [
        TensorRef("token_emb", ("model.language_model.embed_tokens.weight",)),
        TensorRef("per_layer_token_emb", ("model.language_model.embed_tokens_per_layer.weight",)),
        TensorRef("per_layer_model_proj", ("model.language_model.per_layer_model_projection.weight",)),
        TensorRef("per_layer_proj_norm", ("model.language_model.per_layer_projection_norm.weight",), dtype="fp32"),
        TensorRef("rope_freqs", (), dtype="fp32", synth="ones_fp32", shape=(int(config.get("max_rotary_dim") or config.get("rotary_dim") or 256),)),
    ]
    for layer in range(num_layers):
        p = f"model.language_model.layers.{layer}"
        refs.extend([
            TensorRef(f"layer.{layer}.ln1_gamma", (f"{p}.input_layernorm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.ln2_gamma", (f"{p}.pre_feedforward_layernorm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.post_attention_norm", (f"{p}.post_attention_layernorm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.post_ffn_norm", (f"{p}.post_feedforward_layernorm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.wq", (f"{p}.self_attn.q_proj.weight",)),
            TensorRef(f"layer.{layer}.bq", (), dtype="fp32", synth="zeros_fp32", shape=(int(config.get("layer_q_dim", [embed_dim])[layer]) if isinstance(config.get("layer_q_dim"), list) else embed_dim,)),
            TensorRef(f"layer.{layer}.q_norm", (f"{p}.self_attn.q_norm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.wk", (f"{p}.self_attn.k_proj.weight",)),
            TensorRef(f"layer.{layer}.bk", (), dtype="fp32", synth="zeros_fp32", shape=(int(config.get("layer_kv_dim", [embed_dim])[layer]) if isinstance(config.get("layer_kv_dim"), list) else int(config.get("num_key_value_heads", config.get("num_kv_heads", 2))) * int(config.get("head_dim", 256)),)),
            TensorRef(f"layer.{layer}.k_norm", (f"{p}.self_attn.k_norm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.wv", (f"{p}.self_attn.v_proj.weight",)),
            TensorRef(f"layer.{layer}.bv", (), dtype="fp32", synth="zeros_fp32", shape=(int(config.get("layer_kv_dim", [embed_dim])[layer]) if isinstance(config.get("layer_kv_dim"), list) else int(config.get("num_key_value_heads", config.get("num_kv_heads", 2))) * int(config.get("head_dim", 256)),)),
            TensorRef(f"layer.{layer}.wo", (f"{p}.self_attn.o_proj.weight",)),
            TensorRef(f"layer.{layer}.bo", (), dtype="fp32", synth="zeros_fp32", shape=(embed_dim,)),
            TensorRef(f"layer.{layer}.w1", (f"{p}.mlp.gate_proj.weight", f"{p}.mlp.up_proj.weight")),
            TensorRef(f"layer.{layer}.b1", (), dtype="fp32", synth="zeros_fp32", shape=(2 * intermediate,)),
            TensorRef(f"layer.{layer}.w2", (f"{p}.mlp.down_proj.weight",)),
            TensorRef(f"layer.{layer}.b2", (), dtype="fp32", synth="zeros_fp32", shape=(embed_dim,)),
            TensorRef(f"layer.{layer}.per_layer_inp_gate", (f"{p}.per_layer_input_gate.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.per_layer_proj", (f"{p}.per_layer_projection.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.per_layer_post_norm", (f"{p}.post_per_layer_input_norm.weight",), dtype="fp32"),
            TensorRef(f"layer.{layer}.layer_output_scale", (f"{p}.layer_scalar",), dtype="fp32"),
        ])
    refs.extend([
        TensorRef("final_ln_weight", ("model.language_model.norm.weight",), dtype="fp32"),
        TensorRef("final_ln_bias", (), dtype="fp32", synth="zeros_fp32", shape=(embed_dim,)),
    ])
    if "model.language_model.lm_head.weight" in headers:
        refs.append(TensorRef("output.weight", ("model.language_model.lm_head.weight",)))
    elif "lm_head.weight" in headers:
        refs.append(TensorRef("output.weight", ("lm_head.weight",)))
    return refs


def _qwen35_text_refs(config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef]:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    num_layers = int(config.get("num_layers") or config.get("num_hidden_layers") or text.get("num_hidden_layers") or 0)
    embed_dim = int(config.get("embed_dim") or config.get("hidden_size") or text.get("hidden_size") or 0)
    intermediate = int(config.get("intermediate_size") or text.get("intermediate_size") or 0)
    num_heads = int(config.get("num_heads") or text.get("num_attention_heads") or 0)
    num_kv_heads = int(config.get("num_kv_heads") or text.get("num_key_value_heads") or 0)
    head_dim = int(config.get("head_dim") or text.get("head_dim") or (embed_dim // max(1, num_heads)))
    if num_layers <= 0 or embed_dim <= 0 or intermediate <= 0 or num_heads <= 0:
        raise SystemExit("Qwen3.5 config missing num_layers/embed_dim/intermediate_size/num_heads")

    prefix = "model.language_model."
    layer_types = list(config.get("layer_types") or text.get("layer_types") or [])
    if layer_types and len(layer_types) != num_layers:
        raise SystemExit(f"Qwen3.5 layer_types length {len(layer_types)} != num_layers {num_layers}")

    refs: list[TensorRef] = [
        TensorRef("token_emb", (_require_existing(headers, (f"{prefix}embed_tokens.weight",), "token_emb"),)),
    ]

    for layer in range(num_layers):
        layer_prefix = f"{prefix}layers.{layer}"
        layer_type = str(layer_types[layer] if layer_types else ("full_attention" if (layer + 1) % 4 == 0 else "linear_attention"))
        refs.extend([
            TensorRef(f"layer.{layer}.attn_norm", (_require_existing(headers, (f"{layer_prefix}.input_layernorm.weight",), f"layer {layer} input norm"),), dtype="fp32"),
            TensorRef(f"layer.{layer}.post_attention_norm", (_require_existing(headers, (f"{layer_prefix}.post_attention_layernorm.weight",), f"layer {layer} post attention norm"),), dtype="fp32"),
            TensorRef(f"layer.{layer}.ffn_gate", (_require_existing(headers, (f"{layer_prefix}.mlp.gate_proj.weight",), f"layer {layer} gate_proj"),)),
            TensorRef(f"layer.{layer}.ffn_up", (_require_existing(headers, (f"{layer_prefix}.mlp.up_proj.weight",), f"layer {layer} up_proj"),)),
            TensorRef(f"layer.{layer}.ffn_down", (_require_existing(headers, (f"{layer_prefix}.mlp.down_proj.weight",), f"layer {layer} down_proj"),)),
        ])
        if layer_type in {"linear_attention", "recurrent"}:
            refs.extend([
                TensorRef(f"layer.{layer}.attn_qkv", (_require_existing(headers, (f"{layer_prefix}.linear_attn.in_proj_qkv.weight",), f"layer {layer} recurrent qkv"),)),
                TensorRef(f"layer.{layer}.attn_gate", (_require_existing(headers, (f"{layer_prefix}.linear_attn.in_proj_z.weight",), f"layer {layer} recurrent gate"),)),
                TensorRef(f"layer.{layer}.ssm_alpha", (_require_existing(headers, (f"{layer_prefix}.linear_attn.in_proj_a.weight",), f"layer {layer} recurrent alpha"),)),
                TensorRef(f"layer.{layer}.ssm_beta", (_require_existing(headers, (f"{layer_prefix}.linear_attn.in_proj_b.weight",), f"layer {layer} recurrent beta"),)),
                TensorRef(f"layer.{layer}.ssm_conv1d", (_require_existing(headers, (f"{layer_prefix}.linear_attn.conv1d.weight",), f"layer {layer} recurrent conv1d"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.ssm_dt_bias", (_require_existing(headers, (f"{layer_prefix}.linear_attn.dt_bias",), f"layer {layer} recurrent dt bias"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.ssm_a", (_require_existing(headers, (f"{layer_prefix}.linear_attn.A_log",), f"layer {layer} recurrent A_log"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.ssm_norm", (_require_existing(headers, (f"{layer_prefix}.linear_attn.norm.weight",), f"layer {layer} recurrent norm"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.ssm_out", (_require_existing(headers, (f"{layer_prefix}.linear_attn.out_proj.weight",), f"layer {layer} recurrent out"),)),
            ])
        elif layer_type == "full_attention":
            refs.extend([
                TensorRef(f"layer.{layer}.attn_q_gate", (_require_existing(headers, (f"{layer_prefix}.self_attn.q_proj.weight",), f"layer {layer} q_proj"),)),
                TensorRef(f"layer.{layer}.attn_k", (_require_existing(headers, (f"{layer_prefix}.self_attn.k_proj.weight",), f"layer {layer} k_proj"),)),
                TensorRef(f"layer.{layer}.attn_v", (_require_existing(headers, (f"{layer_prefix}.self_attn.v_proj.weight",), f"layer {layer} v_proj"),)),
                TensorRef(f"layer.{layer}.attn_output", (_require_existing(headers, (f"{layer_prefix}.self_attn.o_proj.weight",), f"layer {layer} o_proj"),)),
                TensorRef(f"layer.{layer}.attn_q_norm", (_require_existing(headers, (f"{layer_prefix}.self_attn.q_norm.weight",), f"layer {layer} q_norm"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.attn_k_norm", (_require_existing(headers, (f"{layer_prefix}.self_attn.k_norm.weight",), f"layer {layer} k_norm"),), dtype="fp32"),
            ])
        else:
            raise SystemExit(f"Unsupported Qwen3.5 layer_type at layer {layer}: {layer_type}")

    refs.append(TensorRef("final_ln_weight", (_require_existing(headers, (f"{prefix}norm.weight",), "final norm"),), dtype="fp32"))
    lm_head = _first_existing(headers, ("lm_head.weight", f"{prefix}lm_head.weight"))
    if lm_head is not None:
        refs.append(TensorRef("output.weight", (lm_head,)))
    return refs


def _qwen35_architecture_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and preserve execution metadata shared by Qwen3.5-family checkpoints."""
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    vision = config.get("vision_config") if isinstance(config.get("vision_config"), dict) else {}
    num_layers = int(text.get("num_hidden_layers") or 0)
    hidden_size = int(text.get("hidden_size") or 0)
    intermediate_size = int(text.get("intermediate_size") or 0)
    num_heads = int(text.get("num_attention_heads") or 0)
    num_kv_heads = int(text.get("num_key_value_heads") or 0)
    head_dim = int(text.get("head_dim") or 0)
    if min(num_layers, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim) <= 0:
        raise SystemExit("Qwen3.5 config missing required decoder dimensions")

    interval = int(text.get("full_attention_interval") or 4)
    if interval <= 0:
        raise SystemExit("Qwen3.5 full_attention_interval must be positive")
    layer_types = [str(value) for value in (text.get("layer_types") or [])]
    if not layer_types:
        layer_types = [
            "full_attention" if (layer + 1) % interval == 0 else "linear_attention"
            for layer in range(num_layers)
        ]
    if len(layer_types) != num_layers:
        raise SystemExit(
            f"Qwen3.5 layer_types length {len(layer_types)} != num_layers {num_layers}"
        )
    unsupported = sorted(set(layer_types) - {"linear_attention", "recurrent", "full_attention"})
    if unsupported:
        raise SystemExit(f"Unsupported Qwen3.5 layer types: {unsupported}")

    rope = text.get("rope_parameters") if isinstance(text.get("rope_parameters"), dict) else {}
    partial_rotary_factor = float(
        rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0))
    )
    rotary_width = int(head_dim * partial_rotary_factor)
    if rotary_width <= 0 or rotary_width > head_dim or rotary_width % 2:
        raise SystemExit(
            "Qwen3.5 partial rotary width must be positive, even, and no larger than head_dim"
        )
    sections = [int(value) for value in (rope.get("mrope_section") or [])]
    interleaved = bool(rope.get("mrope_interleaved", False))
    if sections and interleaved and 2 * sum(sections) != rotary_width:
        raise SystemExit(
            "Qwen3.5 interleaved M-RoPE sections must describe the configured rotary width"
        )

    mtp_layers = int(text.get("mtp_num_hidden_layers") or 0)
    has_vision = bool(vision) and not bool(config.get("language_model_only", False))
    return {
        "layer_types": layer_types,
        "layer_kinds": [
            "full_attention" if kind == "full_attention" else "recurrent"
            for kind in layer_types
        ],
        "full_attention_interval": interval,
        "rotary_dim": rotary_width,
        "mrope_sections": sections,
        "mrope_n_dims": rotary_width,
        "mrope_interleaved": interleaved,
        "has_vision_encoder": has_vision,
        "vision_arch": "qwen3_vl_vision" if has_vision else "",
        "vision_depth": int(vision.get("depth") or vision.get("num_hidden_layers") or 0),
        "vision_hidden_size": int(vision.get("hidden_size") or 0),
        "vision_output_size": int(vision.get("out_hidden_size") or hidden_size),
        "image_token_id": int(config.get("image_token_id") or 0),
        "video_token_id": int(config.get("video_token_id") or 0),
        "vision_start_token_id": int(config.get("vision_start_token_id") or 0),
        "vision_end_token_id": int(config.get("vision_end_token_id") or 0),
        "has_mtp_assistant": mtp_layers > 0,
        "mtp_num_layers": mtp_layers,
        "mtp_use_dedicated_embeddings": bool(text.get("mtp_use_dedicated_embeddings", False)),
    }


def _parse_nemotron_h_pattern(pattern: str, num_layers: int) -> list[str]:
    mapping = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}
    chars = [ch for ch in str(pattern or "").strip() if not ch.isspace()]
    if not chars:
        return ["mamba"] * num_layers
    if len(chars) != num_layers:
        raise SystemExit(f"Nemotron-H hybrid_override_pattern length {len(chars)} != num_layers {num_layers}")
    try:
        return [mapping[ch] for ch in chars]
    except KeyError as exc:
        raise SystemExit(f"Unsupported Nemotron-H layer pattern character: {exc.args[0]!r}") from exc





def _kimi_vl_text_refs(
    config: dict[str, Any],
    headers: dict[str, HeaderTensor],
    *,
    source_root: str = "language_model.model",
    architecture_label: str = "Kimi-VL",
    include_attention_gate: bool = False,
) -> list[TensorRef]:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    num_layers = int(text.get("num_hidden_layers") or config.get("num_layers") or 0)
    hidden = int(text.get("hidden_size") or config.get("hidden_size") or 0)
    intermediate = int(text.get("intermediate_size") or config.get("intermediate_size") or 0)
    moe_intermediate = int(text.get("moe_intermediate_size") or config.get("moe_intermediate_size") or 0)
    n_experts = int(text.get("n_routed_experts") or config.get("n_routed_experts") or 0)
    n_shared = int(text.get("n_shared_experts") or config.get("n_shared_experts") or 0)
    layer_kinds = list(config.get("layer_kinds") or [])
    if not layer_kinds:
        first_dense = int(text.get("first_k_dense_replace") or 0)
        moe_freq = int(text.get("moe_layer_freq") or 1)
        layer_kinds = ["mla_dense_mlp" if layer < first_dense or (moe_freq > 0 and layer % moe_freq != 0) else "mla_moe" for layer in range(num_layers)]
    if num_layers <= 0 or hidden <= 0 or intermediate <= 0:
        raise SystemExit(f"{architecture_label} config missing num_hidden_layers/hidden_size/intermediate_size")
    if len(layer_kinds) != num_layers:
        raise SystemExit(f"{architecture_label} layer_kinds length {len(layer_kinds)} != num_layers {num_layers}")

    refs: list[TensorRef] = [
        TensorRef("token_emb", (_require_existing(headers, (f"{source_root}.embed_tokens.weight", "model.embed_tokens.weight"), "token_emb"),)),
    ]

    for layer, kind in enumerate(layer_kinds):
        pfx = f"{source_root}.layers.{layer}"
        refs.extend([
            TensorRef(f"layer.{layer}.block_norm", (_require_existing(headers, (f"{pfx}.input_layernorm.weight",), f"layer {layer} input norm"),), dtype="fp32"),
            TensorRef(f"layer.{layer}.post_attention_norm", (_require_existing(headers, (f"{pfx}.post_attention_layernorm.weight",), f"layer {layer} post attention norm"),), dtype="fp32"),
            TensorRef(f"layer.{layer}.mla_q_proj", (_require_existing(headers, (f"{pfx}.self_attn.q_proj.weight",), f"layer {layer} MLA q_proj"),)),
            TensorRef(f"layer.{layer}.mla_kv_a_proj", (_require_existing(headers, (f"{pfx}.self_attn.kv_a_proj_with_mqa.weight",), f"layer {layer} MLA kv_a_proj_with_mqa"),)),
            TensorRef(f"layer.{layer}.mla_kv_a_norm", (_require_existing(headers, (f"{pfx}.self_attn.kv_a_layernorm.weight",), f"layer {layer} MLA kv_a_layernorm"),), dtype="fp32"),
            TensorRef(f"layer.{layer}.mla_kv_b_proj", (_require_existing(headers, (f"{pfx}.self_attn.kv_b_proj.weight",), f"layer {layer} MLA kv_b_proj"),)),
            TensorRef(f"layer.{layer}.mla_out_proj", (_require_existing(headers, (f"{pfx}.self_attn.o_proj.weight",), f"layer {layer} MLA o_proj"),)),
        ])
        if include_attention_gate:
            refs.append(TensorRef(
                f"layer.{layer}.mla_gate_proj",
                (_require_existing(headers, (f"{pfx}.self_attn.gate_proj.weight",), f"layer {layer} MLA gate_proj"),),
            ))
        mlp = f"{pfx}.mlp"
        if kind == "mla_dense_mlp":
            refs.extend([
                TensorRef(f"layer.{layer}.mlp_gate", (_require_existing(headers, (f"{mlp}.gate_proj.weight",), f"layer {layer} dense gate_proj"),)),
                TensorRef(f"layer.{layer}.mlp_up", (_require_existing(headers, (f"{mlp}.up_proj.weight",), f"layer {layer} dense up_proj"),)),
                TensorRef(f"layer.{layer}.mlp_down", (_require_existing(headers, (f"{mlp}.down_proj.weight",), f"layer {layer} dense down_proj"),)),
            ])
        elif kind == "mla_moe":
            if n_experts <= 0:
                raise SystemExit(f"{architecture_label} MoE layer requires n_routed_experts")
            refs.append(TensorRef(f"layer.{layer}.moe_router", (_require_existing(headers, (f"{mlp}.gate.weight",), f"layer {layer} moe router"),), dtype="fp32"))
            correction = _maybe_tensor_ref(headers, f"layer.{layer}.moe_router_bias", (f"{mlp}.gate.e_score_correction_bias",), dtype="fp32")
            if correction is not None:
                refs.append(correction)
            expert_gate_sources: list[str] = []
            expert_up_sources: list[str] = []
            expert_down_sources: list[str] = []
            expert_gate_shape: tuple[int, ...] | None = None
            expert_up_shape: tuple[int, ...] | None = None
            expert_down_shape: tuple[int, ...] | None = None
            for expert in range(n_experts):
                ep = f"{mlp}.experts.{expert}"
                gate = _require_existing(headers, (f"{ep}.gate_proj.weight",), f"layer {layer} expert {expert} gate_proj")
                up = _require_existing(headers, (f"{ep}.up_proj.weight",), f"layer {layer} expert {expert} up_proj")
                down = _require_existing(headers, (f"{ep}.down_proj.weight",), f"layer {layer} expert {expert} down_proj")
                expert_gate_sources.append(gate)
                expert_up_sources.append(up)
                expert_down_sources.append(down)
                if expert_gate_shape is None:
                    expert_gate_shape = tuple([n_experts] + list(headers[gate].shape))
                if expert_up_shape is None:
                    expert_up_shape = tuple([n_experts] + list(headers[up].shape))
                if expert_down_shape is None:
                    expert_down_shape = tuple([n_experts] + list(headers[down].shape))
            refs.extend([
                TensorRef(f"layer.{layer}.moe_expert_gate", tuple(expert_gate_sources), shape=expert_gate_shape),
                TensorRef(f"layer.{layer}.moe_expert_up", tuple(expert_up_sources), shape=expert_up_shape),
                TensorRef(f"layer.{layer}.moe_expert_down", tuple(expert_down_sources), shape=expert_down_shape),
            ])
            sp = f"{mlp}.shared_experts"
            refs.extend([
                TensorRef(f"layer.{layer}.moe_shared_gate", (_require_existing(headers, (f"{sp}.gate_proj.weight",), f"layer {layer} shared gate_proj"),)),
                TensorRef(f"layer.{layer}.moe_shared_up", (_require_existing(headers, (f"{sp}.up_proj.weight",), f"layer {layer} shared up_proj"),)),
                TensorRef(f"layer.{layer}.moe_shared_down", (_require_existing(headers, (f"{sp}.down_proj.weight",), f"layer {layer} shared down_proj"),)),
            ])
        else:
            raise SystemExit(f"Unsupported {architecture_label} layer kind at {layer}: {kind}")

    refs.append(TensorRef("final_ln_weight", (_require_existing(headers, (f"{source_root}.norm.weight", "model.norm.weight"), "final norm"),), dtype="fp32"))
    refs.append(TensorRef("final_ln_bias", (), dtype="fp32", synth="zeros_fp32", shape=(hidden,)))
    lm_head = _first_existing(headers, ("language_model.lm_head.weight", "lm_head.weight", "model.lm_head.weight"))
    if lm_head is not None:
        refs.append(TensorRef("output.weight", (lm_head,)))
    return refs


def _instella_moe_refs(config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef]:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    adapted = dict(config)
    num_layers = int(text.get("num_hidden_layers") or 0)
    first_dense = int(text.get("first_k_dense_replace") or 0)
    moe_freq = max(1, int(text.get("moe_layer_freq") or 1))
    declared_kinds = list(config.get("layer_kinds") or [])
    if declared_kinds:
        # The runtime circuit uses richer topology names (first FarSkip versus
        # continuation), while tensor mapping only needs dense versus MoE.
        adapted["layer_kinds"] = [
            "mla_dense_mlp" if "dense_mlp" in str(kind) else "mla_moe"
            for kind in declared_kinds
        ]
    else:
        adapted["layer_kinds"] = [
            "mla_dense_mlp"
            if layer < first_dense or (layer - first_dense) % moe_freq != 0
            else "mla_moe"
            for layer in range(num_layers)
        ]
    return _kimi_vl_text_refs(
        adapted,
        headers,
        source_root="model",
        architecture_label="Instella-MoE",
        include_attention_gate=True,
    )


def _nemotron_dt_limit(config: dict[str, Any]) -> tuple[float, float]:
    """Return CK runtime dt clamp bounds for Nemotron-H.

    Nemotron-H uses time_step_min/max for initialization only. Runtime forward
    clamps softplus(dt + bias) with time_step_limit; the common (0, inf) limit
    is represented as (0, 0) so the CK kernel disables clamping.
    """
    limit = config.get("time_step_limit")
    if isinstance(limit, (list, tuple)) and len(limit) >= 2:
        lo = float(limit[0] or 0.0)
        try:
            hi = float(limit[1])
        except Exception:
            hi = float("inf")
        if hi == float("inf"):
            return 0.0, 0.0
        return lo, hi
    return 0.0, 0.0

def _nemotron_h_text_refs(config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef]:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    num_layers = int(text.get("num_hidden_layers") or config.get("num_layers") or 0)
    hidden = int(text.get("hidden_size") or config.get("hidden_size") or 0)
    intermediate = int(text.get("intermediate_size") or config.get("intermediate_size") or 0)
    moe_intermediate = int(text.get("moe_intermediate_size") or intermediate)
    shared_intermediate = int(text.get("moe_shared_expert_intermediate_size") or moe_intermediate)
    n_experts = int(text.get("n_routed_experts") or 0)
    layer_kinds = list(config.get("layer_kinds") or _parse_nemotron_h_pattern(str(text.get("hybrid_override_pattern") or ""), num_layers))
    if num_layers <= 0 or hidden <= 0 or intermediate <= 0:
        raise SystemExit("Nemotron-H config missing num_hidden_layers/hidden_size/intermediate_size")
    if len(layer_kinds) != num_layers:
        raise SystemExit(f"Nemotron-H layer_kinds length {len(layer_kinds)} != num_layers {num_layers}")

    refs: list[TensorRef] = [
        TensorRef("token_emb", (_require_existing(headers, ("backbone.embeddings.weight", "model.embed_tokens.weight"), "token_emb"),)),
    ]

    for layer, kind in enumerate(layer_kinds):
        pfx = f"backbone.layers.{layer}"
        refs.append(TensorRef(f"layer.{layer}.block_norm", (_require_existing(headers, (f"{pfx}.norm.weight",), f"layer {layer} block norm"),), dtype="fp32"))
        mp = f"{pfx}.mixer"
        if kind == "mamba":
            refs.extend([
                TensorRef(f"layer.{layer}.mamba_in_proj", (_require_existing(headers, (f"{mp}.in_proj.weight",), f"layer {layer} mamba in_proj"),)),
                TensorRef(f"layer.{layer}.mamba_conv1d", (_require_existing(headers, (f"{mp}.conv1d.weight",), f"layer {layer} mamba conv1d"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.mamba_conv1d_bias", (_require_existing(headers, (f"{mp}.conv1d.bias",), f"layer {layer} mamba conv1d bias"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.mamba_dt_bias", (_require_existing(headers, (f"{mp}.dt_bias",), f"layer {layer} mamba dt_bias"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.mamba_a", (_require_existing(headers, (f"{mp}.A_log",), f"layer {layer} mamba A_log"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.mamba_d", (_require_existing(headers, (f"{mp}.D",), f"layer {layer} mamba D"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.mamba_norm", (_require_existing(headers, (f"{mp}.norm.weight",), f"layer {layer} mamba norm"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.mamba_out_proj", (_require_existing(headers, (f"{mp}.out_proj.weight",), f"layer {layer} mamba out_proj"),)),
            ])
        elif kind == "attention":
            refs.extend([
                TensorRef(f"layer.{layer}.attn_q", (_require_existing(headers, (f"{mp}.q_proj.weight",), f"layer {layer} q_proj"),)),
                TensorRef(f"layer.{layer}.attn_k", (_require_existing(headers, (f"{mp}.k_proj.weight",), f"layer {layer} k_proj"),)),
                TensorRef(f"layer.{layer}.attn_v", (_require_existing(headers, (f"{mp}.v_proj.weight",), f"layer {layer} v_proj"),)),
                TensorRef(f"layer.{layer}.attn_o", (_require_existing(headers, (f"{mp}.o_proj.weight",), f"layer {layer} o_proj"),)),
            ])
        elif kind == "mlp":
            refs.extend([
                TensorRef(f"layer.{layer}.mlp_up", (_require_existing(headers, (f"{mp}.up_proj.weight",), f"layer {layer} mlp up_proj"),)),
                TensorRef(f"layer.{layer}.mlp_down", (_require_existing(headers, (f"{mp}.down_proj.weight",), f"layer {layer} mlp down_proj"),)),
            ])
        elif kind == "moe":
            if n_experts <= 0:
                raise SystemExit("Nemotron-H MoE layer requires n_routed_experts")
            refs.extend([
                TensorRef(f"layer.{layer}.moe_router", (_require_existing(headers, (f"{mp}.gate.weight",), f"layer {layer} moe router"),), dtype="fp32"),
                TensorRef(f"layer.{layer}.moe_router_bias", (_require_existing(headers, (f"{mp}.gate.e_score_correction_bias",), f"layer {layer} moe router correction bias"),), dtype="fp32"),
            ])
            for expert in range(n_experts):
                refs.extend([
                    TensorRef(f"layer.{layer}.moe_expert.{expert}.up", (_require_existing(headers, (f"{mp}.experts.{expert}.up_proj.weight",), f"layer {layer} expert {expert} up_proj"),)),
                    TensorRef(f"layer.{layer}.moe_expert.{expert}.down", (_require_existing(headers, (f"{mp}.experts.{expert}.down_proj.weight",), f"layer {layer} expert {expert} down_proj"),)),
                ])
            refs.extend([
                TensorRef(f"layer.{layer}.moe_shared_up", (_require_existing(headers, (f"{mp}.shared_experts.up_proj.weight",), f"layer {layer} shared expert up_proj"),)),
                TensorRef(f"layer.{layer}.moe_shared_down", (_require_existing(headers, (f"{mp}.shared_experts.down_proj.weight",), f"layer {layer} shared expert down_proj"),)),
            ])
        else:
            raise SystemExit(f"Unsupported Nemotron-H layer kind at {layer}: {kind}")

    refs.append(TensorRef("final_ln_weight", (_require_existing(headers, ("backbone.norm_f.weight", "model.norm.weight"), "final norm"),), dtype="fp32"))
    refs.append(TensorRef("final_ln_bias", (), dtype="fp32", synth="zeros_fp32", shape=(hidden,)))
    lm_head = _first_existing(headers, ("lm_head.weight", "backbone.lm_head.weight"))
    if lm_head is not None:
        refs.append(TensorRef("output.weight", (lm_head,)))
    return refs


def _first_existing(headers: dict[str, HeaderTensor], names: Iterable[str]) -> str | None:
    for name in names:
        if name in headers:
            return name
    return None


def _require_existing(headers: dict[str, HeaderTensor], names: Iterable[str], desc: str) -> str:
    found = _first_existing(headers, names)
    if found is None:
        raise SystemExit(f"Missing required safetensors tensor for {desc}: tried {list(names)}")
    return found


def _language_prefix(headers: dict[str, HeaderTensor]) -> str:
    if "model.embed_tokens.weight" in headers:
        return "model."
    if "model.language_model.embed_tokens.weight" in headers:
        return "model.language_model."
    return "model."


def _maybe_tensor_ref(
    headers: dict[str, HeaderTensor],
    ck_name: str,
    source_names: tuple[str, ...],
    *,
    dtype: str | None = None,
) -> TensorRef | None:
    found = _first_existing(headers, source_names)
    if found is None:
        return None
    return TensorRef(ck_name, (found,), dtype=dtype)


def _llama_family_text_refs(config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef]:
    num_layers = int(config.get("num_layers") or config.get("num_hidden_layers") or 0)
    embed_dim = int(config.get("embed_dim") or config.get("hidden_size") or 0)
    intermediate = int(config.get("intermediate_size") or 0)
    num_heads = int(config.get("num_heads") or config.get("num_attention_heads") or 0)
    num_kv_heads = int(config.get("num_kv_heads") or config.get("num_key_value_heads") or num_heads or 0)
    head_dim = int(config.get("head_dim") or (embed_dim // max(1, num_heads)))
    if num_layers <= 0 or embed_dim <= 0 or intermediate <= 0 or num_heads <= 0:
        raise SystemExit("Llama-family config missing num_layers/embed_dim/intermediate_size/num_heads")

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    prefix = _language_prefix(headers)
    refs: list[TensorRef] = [
        TensorRef("token_emb", (_require_existing(headers, (f"{prefix}embed_tokens.weight", "model.embed_tokens.weight"), "token_emb"),)),
    ]

    for layer in range(num_layers):
        layer_prefix = f"{prefix}layers.{layer}"
        refs.extend([
            TensorRef(f"layer.{layer}.ln1_gamma", (_require_existing(headers, (f"{layer_prefix}.input_layernorm.weight",), f"layer {layer} input norm"),), dtype="fp32"),
            TensorRef(f"layer.{layer}.ln2_gamma", (_require_existing(headers, (f"{layer_prefix}.post_attention_layernorm.weight", f"{layer_prefix}.pre_feedforward_layernorm.weight"), f"layer {layer} ffn norm"),), dtype="fp32"),
            TensorRef(f"layer.{layer}.wq", (_require_existing(headers, (f"{layer_prefix}.self_attn.q_proj.weight",), f"layer {layer} q_proj"),)),
            _maybe_tensor_ref(headers, f"layer.{layer}.bq", (f"{layer_prefix}.self_attn.q_proj.bias",), dtype="fp32")
            or TensorRef(f"layer.{layer}.bq", (), dtype="fp32", synth="zeros_fp32", shape=(q_dim,)),
        ])
        q_norm = _maybe_tensor_ref(headers, f"layer.{layer}.q_norm", (f"{layer_prefix}.self_attn.q_norm.weight",), dtype="fp32")
        if q_norm is not None:
            refs.append(q_norm)
        refs.extend([
            TensorRef(f"layer.{layer}.wk", (_require_existing(headers, (f"{layer_prefix}.self_attn.k_proj.weight",), f"layer {layer} k_proj"),)),
            _maybe_tensor_ref(headers, f"layer.{layer}.bk", (f"{layer_prefix}.self_attn.k_proj.bias",), dtype="fp32")
            or TensorRef(f"layer.{layer}.bk", (), dtype="fp32", synth="zeros_fp32", shape=(kv_dim,)),
        ])
        k_norm = _maybe_tensor_ref(headers, f"layer.{layer}.k_norm", (f"{layer_prefix}.self_attn.k_norm.weight",), dtype="fp32")
        if k_norm is not None:
            refs.append(k_norm)
        refs.extend([
            TensorRef(f"layer.{layer}.wv", (_require_existing(headers, (f"{layer_prefix}.self_attn.v_proj.weight",), f"layer {layer} v_proj"),)),
            _maybe_tensor_ref(headers, f"layer.{layer}.bv", (f"{layer_prefix}.self_attn.v_proj.bias",), dtype="fp32")
            or TensorRef(f"layer.{layer}.bv", (), dtype="fp32", synth="zeros_fp32", shape=(kv_dim,)),
            TensorRef(f"layer.{layer}.wo", (_require_existing(headers, (f"{layer_prefix}.self_attn.o_proj.weight",), f"layer {layer} o_proj"),)),
            _maybe_tensor_ref(headers, f"layer.{layer}.bo", (f"{layer_prefix}.self_attn.o_proj.bias",), dtype="fp32")
            or TensorRef(f"layer.{layer}.bo", (), dtype="fp32", synth="zeros_fp32", shape=(embed_dim,)),
            TensorRef(
                f"layer.{layer}.w1",
                (
                    _require_existing(headers, (f"{layer_prefix}.mlp.gate_proj.weight",), f"layer {layer} gate_proj"),
                    _require_existing(headers, (f"{layer_prefix}.mlp.up_proj.weight",), f"layer {layer} up_proj"),
                ),
            ),
            TensorRef(f"layer.{layer}.b1", (), dtype="fp32", synth="zeros_fp32", shape=(2 * intermediate,)),
            TensorRef(f"layer.{layer}.w2", (_require_existing(headers, (f"{layer_prefix}.mlp.down_proj.weight",), f"layer {layer} down_proj"),)),
            TensorRef(f"layer.{layer}.b2", (), dtype="fp32", synth="zeros_fp32", shape=(embed_dim,)),
        ])

    refs.extend([
        TensorRef("final_ln_weight", (_require_existing(headers, (f"{prefix}norm.weight", "model.norm.weight"), "final norm"),), dtype="fp32"),
        TensorRef("final_ln_bias", (), dtype="fp32", synth="zeros_fp32", shape=(embed_dim,)),
    ])
    lm_head = _first_existing(headers, ("lm_head.weight", f"{prefix}lm_head.weight", "model.lm_head.weight"))
    if lm_head is not None:
        refs.append(TensorRef("output.weight", (lm_head,)))
    return refs


def _qwen3vl_vision_refs(config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef]:
    num_layers = int(config.get("num_layers") or 0)
    hidden = int(config.get("embed_dim") or config.get("hidden_size") or 0)
    intermediate = int(config.get("intermediate_size") or config.get("intermediate_dim") or 0)
    patch_dim = int(config.get("patch_dim") or 0)
    if num_layers <= 0 or hidden <= 0 or intermediate <= 0 or patch_dim <= 0:
        raise SystemExit("Qwen3-VL vision config missing num_layers/embed_dim/intermediate_size/patch_dim")

    patch_src = _require_existing(
        headers,
        ("model.visual.patch_embed.proj.weight", "visual.patch_embed.proj.weight"),
        "Qwen3-VL temporal patch projection",
    )
    refs: list[TensorRef] = [
        TensorRef("v.patch_embd.weight", (patch_src,), shape=(hidden, patch_dim), transform="qwen3vl_patch_temporal_0"),
        TensorRef("v.patch_embd.weight.1", (patch_src,), shape=(hidden, patch_dim), transform="qwen3vl_patch_temporal_1"),
        TensorRef(
            "v.patch_embd.bias",
            (_require_existing(headers, ("model.visual.patch_embed.proj.bias", "visual.patch_embed.proj.bias"), "Qwen3-VL patch bias"),),
            dtype="fp32",
        ),
        TensorRef(
            "v.position_embd.weight",
            (_require_existing(headers, ("model.visual.pos_embed.weight", "visual.pos_embed.weight"), "Qwen3-VL position embeddings"),),
            dtype="fp32",
        ),
    ]

    for layer in range(num_layers):
        pfx = f"model.visual.blocks.{layer}"
        refs.extend([
            TensorRef(f"v.blk.{layer}.ln1.weight", (_require_existing(headers, (f"{pfx}.norm1.weight",), f"vision layer {layer} norm1 weight"),), dtype="fp32"),
            TensorRef(f"v.blk.{layer}.ln1.bias", (_require_existing(headers, (f"{pfx}.norm1.bias",), f"vision layer {layer} norm1 bias"),), dtype="fp32"),
            TensorRef(f"v.blk.{layer}.ln2.weight", (_require_existing(headers, (f"{pfx}.norm2.weight",), f"vision layer {layer} norm2 weight"),), dtype="fp32"),
            TensorRef(f"v.blk.{layer}.ln2.bias", (_require_existing(headers, (f"{pfx}.norm2.bias",), f"vision layer {layer} norm2 bias"),), dtype="fp32"),
            TensorRef(f"v.blk.{layer}.attn_qkv.weight", (_require_existing(headers, (f"{pfx}.attn.qkv.weight",), f"vision layer {layer} qkv weight"),)),
            TensorRef(f"v.blk.{layer}.attn_qkv.bias", (_require_existing(headers, (f"{pfx}.attn.qkv.bias",), f"vision layer {layer} qkv bias"),), dtype="fp32"),
            TensorRef(f"v.blk.{layer}.attn_out.weight", (_require_existing(headers, (f"{pfx}.attn.proj.weight",), f"vision layer {layer} attention output weight"),)),
            TensorRef(f"v.blk.{layer}.attn_out.bias", (_require_existing(headers, (f"{pfx}.attn.proj.bias",), f"vision layer {layer} attention output bias"),), dtype="fp32"),
            TensorRef(f"v.blk.{layer}.ffn_up.weight", (_require_existing(headers, (f"{pfx}.mlp.linear_fc1.weight",), f"vision layer {layer} mlp fc1 weight"),)),
            TensorRef(f"v.blk.{layer}.ffn_up.bias", (_require_existing(headers, (f"{pfx}.mlp.linear_fc1.bias",), f"vision layer {layer} mlp fc1 bias"),), dtype="fp32"),
            TensorRef(f"v.blk.{layer}.ffn_down.weight", (_require_existing(headers, (f"{pfx}.mlp.linear_fc2.weight",), f"vision layer {layer} mlp fc2 weight"),)),
            TensorRef(f"v.blk.{layer}.ffn_down.bias", (_require_existing(headers, (f"{pfx}.mlp.linear_fc2.bias",), f"vision layer {layer} mlp fc2 bias"),), dtype="fp32"),
        ])

    refs.extend([
        TensorRef("v.post_ln.weight", (_require_existing(headers, ("model.visual.merger.norm.weight",), "Qwen3-VL merger norm weight"),), dtype="fp32"),
        TensorRef("v.post_ln.bias", (_require_existing(headers, ("model.visual.merger.norm.bias",), "Qwen3-VL merger norm bias"),), dtype="fp32"),
        TensorRef("mm.0.weight", (_require_existing(headers, ("model.visual.merger.linear_fc1.weight",), "Qwen3-VL merger fc1 weight"),)),
        TensorRef("mm.0.bias", (_require_existing(headers, ("model.visual.merger.linear_fc1.bias",), "Qwen3-VL merger fc1 bias"),), dtype="fp32"),
        TensorRef("mm.2.weight", (_require_existing(headers, ("model.visual.merger.linear_fc2.weight",), "Qwen3-VL merger fc2 weight"),)),
        TensorRef("mm.2.bias", (_require_existing(headers, ("model.visual.merger.linear_fc2.bias",), "Qwen3-VL merger fc2 bias"),), dtype="fp32"),
    ])

    deepstack_layer_indices = [int(x) for x in (config.get("deepstack_layer_indices") or [])]
    for compact_idx, layer in enumerate(deepstack_layer_indices):
        pfx = f"model.visual.deepstack_merger_list.{compact_idx}"
        refs.extend([
            TensorRef(f"v.deepstack.{layer}.norm.weight", (_require_existing(headers, (f"{pfx}.norm.weight",), f"deepstack {compact_idx} norm weight"),), dtype="fp32"),
            TensorRef(f"v.deepstack.{layer}.norm.bias", (_require_existing(headers, (f"{pfx}.norm.bias",), f"deepstack {compact_idx} norm bias"),), dtype="fp32"),
            TensorRef(f"v.deepstack.{layer}.fc1.weight", (_require_existing(headers, (f"{pfx}.linear_fc1.weight",), f"deepstack {compact_idx} fc1 weight"),)),
            TensorRef(f"v.deepstack.{layer}.fc1.bias", (_require_existing(headers, (f"{pfx}.linear_fc1.bias",), f"deepstack {compact_idx} fc1 bias"),), dtype="fp32"),
            TensorRef(f"v.deepstack.{layer}.fc2.weight", (_require_existing(headers, (f"{pfx}.linear_fc2.weight",), f"deepstack {compact_idx} fc2 weight"),)),
            TensorRef(f"v.deepstack.{layer}.fc2.bias", (_require_existing(headers, (f"{pfx}.linear_fc2.bias",), f"deepstack {compact_idx} fc2 bias"),), dtype="fp32"),
        ])
    return refs


def _infer_arch(hf: dict[str, Any]) -> str:
    text = hf.get("text_config") if isinstance(hf.get("text_config"), dict) else hf
    root_mt = str(hf.get("model_type") or "").lower()
    if root_mt == "gemma4_assistant":
        return "gemma4_assistant"
    mt = str(text.get("model_type") or hf.get("model_type") or "").lower()
    arch_names = [str(x).lower() for x in (text.get("architectures") or hf.get("architectures") or []) if isinstance(x, str)]
    contracts = (_load_safetensors_ck_map().get("architectures") or {})
    for ck_arch, contract in contracts.items():
        aliases = [str(ck_arch).lower()]
        if isinstance(contract, dict):
            aliases.extend(str(x).lower() for x in contract.get("aliases", []) if isinstance(x, str))
        if mt in aliases or any(name in aliases for name in arch_names):
            return str(ck_arch).lower()
    if "gemma" in mt and "4" in mt:
        return "gemma4"
    if "gemma" in mt and "3" in mt:
        return "gemma3"
    if "qwen3_5" in mt or "qwen3.5" in mt or "qwen35" in mt:
        return "qwen35"
    if "qwen3_vl" in mt or any("qwen3vl" in name or "qwen3_vl" in name for name in arch_names):
        return "qwen3vl"
    if "qwen3" in mt:
        return "qwen3"
    if "qwen2" in mt or mt == "qwen":
        return "qwen2"
    if "glm" in mt:
        return "glm4"
    if "llama" in mt:
        return "llama"
    if "nemotron_h" in mt or "nemotron-h" in mt:
        return "nemotron_h"
    return mt


def _refs_for_arch(arch: str, config: dict[str, Any], headers: dict[str, HeaderTensor]) -> list[TensorRef]:
    mapped_refs = _refs_from_safetensors_contract(arch, config, headers)
    if mapped_refs is not None:
        return mapped_refs
    tensor_mapper = str(_safetensors_arch_contract(arch).get("tensor_mapper") or "").strip().lower()
    if tensor_mapper == "qwen3_vl_vision":
        return _qwen3vl_vision_refs(config, headers)
    if arch == "gemma4_assistant":
        return _refs_from_safetensors_contract("gemma4_assistant", config, headers) or _llama_family_text_refs(config, headers)
    if arch == "gemma4":
        return _gemma4_text_refs(config, headers)
    if arch == "qwen35":
        return _qwen35_text_refs(config, headers)
    if arch == "qwen3_vl_vision":
        return _qwen3vl_vision_refs(config, headers)
    if arch == "nemotron_h":
        return _nemotron_h_text_refs(config, headers)
    if arch == "kimi_vl":
        return _kimi_vl_text_refs(config, headers)
    if arch == "instella_moe":
        return _instella_moe_refs(config, headers)
    if arch in {"llama", "qwen2", "qwen3", "qwen3vl", "gemma3"}:
        return _llama_family_text_refs(config, headers)
    raise SystemExit(f"Unsupported safetensors arch for v8 importer: {arch}")


def _load_safetensors_template_for_arch(arch: str) -> dict[str, Any]:
    contract = _safetensors_arch_contract(arch)
    template_name = str(contract.get("template") or arch).strip().lower()
    return load_template_for_arch(template_name)


def _build_gemma4_attention_plan_from_hf(text: dict[str, Any], headers: dict[str, HeaderTensor]) -> dict[str, Any]:
    num_layers = int(text.get("num_hidden_layers") or 0)
    if num_layers <= 0:
        raise SystemExit("Gemma4 config missing num_hidden_layers")

    layer_types = [str(x) for x in (text.get("layer_types") or [])]
    if not layer_types:
        interval = int(text.get("full_attention_interval") or 5)
        layer_types = ["full_attention" if (layer + 1) % interval == 0 else "sliding_attention" for layer in range(num_layers)]
    if len(layer_types) != num_layers:
        raise SystemExit(f"Gemma4 layer_types length {len(layer_types)} != num_layers {num_layers}")

    shared_kv_layers = int(text.get("num_kv_shared_layers") or 0)
    if shared_kv_layers < 0 or shared_kv_layers > num_layers:
        raise SystemExit(
            f"Gemma4 num_kv_shared_layers must be between 0 and num_layers "
            f"(got {shared_kv_layers}, num_layers={num_layers})"
        )
    first_shared_kv_layer = num_layers - shared_kv_layers
    full_kv_producer = first_shared_kv_layer - 1
    sliding_kv_producer = first_shared_kv_layer - 2

    sliding_window = int(text.get("sliding_window") or 0)
    num_heads = int(text.get("num_attention_heads") or 0)
    num_kv_heads = int(text.get("num_key_value_heads") or 0)
    default_head_dim = int(text.get("head_dim") or 0)

    rope_params = text.get("rope_parameters") if isinstance(text.get("rope_parameters"), dict) else {}
    full_rope = rope_params.get("full_attention") if isinstance(rope_params.get("full_attention"), dict) else {}
    sliding_rope = rope_params.get("sliding_attention") if isinstance(rope_params.get("sliding_attention"), dict) else {}
    full_partial = float(full_rope.get("partial_rotary_factor", 1.0) or 1.0)
    sliding_partial = float(sliding_rope.get("partial_rotary_factor", 1.0) or 1.0)

    layer_kinds: list[str] = []
    layer_kv_policy: list[str] = []
    layer_kv_source: list[int] = []
    layer_sliding_window: list[int] = []
    layer_rope_kind: list[str] = []
    layer_q_head_dim: list[int] = []
    layer_k_head_dim: list[int] = []
    layer_v_head_dim: list[int] = []
    layer_rotary_dim: list[int] = []
    layer_q_dim: list[int] = []
    layer_kv_dim: list[int] = []
    layer_attention_plan: list[dict[str, Any]] = []

    for layer, raw_kind in enumerate(layer_types):
        attention_kind = "full" if raw_kind == "full_attention" else "sliding"
        owns_kv = layer < first_shared_kv_layer
        if owns_kv:
            kv_source = layer
        elif attention_kind == "sliding":
            kv_source = sliding_kv_producer
        else:
            kv_source = full_kv_producer
        if kv_source < 0:
            raise SystemExit("Gemma4 shared-KV plan has no producer layer to reuse")

        prefix = f"model.language_model.layers.{layer}.self_attn"
        q = headers.get(f"{prefix}.q_proj.weight")
        k = headers.get(f"{prefix}.k_proj.weight")
        v = headers.get(f"{prefix}.v_proj.weight")
        q_dim = int(q.shape[0]) if q and q.shape else num_heads * default_head_dim
        k_dim = int(k.shape[0]) if k and k.shape else num_kv_heads * default_head_dim
        v_dim = int(v.shape[0]) if v and v.shape else k_dim
        q_head_dim = int(q_dim // max(1, num_heads)) if q_dim else default_head_dim
        k_head_dim = int(k_dim // max(1, num_kv_heads)) if k_dim else default_head_dim
        v_head_dim = int(v_dim // max(1, num_kv_heads)) if v_dim else k_head_dim
        partial = full_partial if attention_kind == "full" else sliding_partial
        rotary_dim = int(q_head_dim * partial)

        kind = f"{attention_kind}_attention_kv" if owns_kv else f"{attention_kind}_attention_shared_kv"
        layer_kinds.append(kind)
        layer_kv_policy.append("produce" if owns_kv else "reuse")
        layer_kv_source.append(kv_source)
        layer_sliding_window.append(sliding_window if attention_kind == "sliding" else 0)
        layer_rope_kind.append("full" if attention_kind == "full" else "swa")
        layer_q_head_dim.append(q_head_dim)
        layer_k_head_dim.append(k_head_dim)
        layer_v_head_dim.append(v_head_dim)
        layer_rotary_dim.append(rotary_dim)
        layer_q_dim.append(q_dim)
        layer_kv_dim.append(k_dim)
        layer_attention_plan.append({
            "layer": layer,
            "kind": kind,
            "attention_kind": attention_kind,
            "kv_policy": "produce" if owns_kv else "reuse",
            "kv_source_layer": kv_source,
            "sliding_window": sliding_window if attention_kind == "sliding" else 0,
            "rope_kind": "full" if attention_kind == "full" else "swa",
            "q_head_dim": q_head_dim,
            "k_head_dim": k_head_dim,
            "v_head_dim": v_head_dim,
            "rotary_dim": rotary_dim,
            "q_dim": q_dim,
            "kv_dim": k_dim,
        })

    layer_k_cache_offset: list[int] = []
    layer_v_cache_offset: list[int] = []
    kv_cache_token_stride_total = 0
    for k_head, v_head in zip(layer_k_head_dim, layer_v_head_dim):
        k_elems = num_kv_heads * int(k_head)
        v_elems = num_kv_heads * int(v_head)
        layer_k_cache_offset.append(kv_cache_token_stride_total)
        kv_cache_token_stride_total += k_elems
        layer_v_cache_offset.append(kv_cache_token_stride_total)
        kv_cache_token_stride_total += v_elems

    return {
        "layer_types": layer_types,
        "layer_kinds": layer_kinds,
        "hybrid_block_pattern": layer_kinds[:],
        "layer_attention_plan": layer_attention_plan,
        "layer_kv_policy": layer_kv_policy,
        "layer_kv_source": layer_kv_source,
        "layer_sliding_window": layer_sliding_window,
        "layer_rope_kind": layer_rope_kind,
        "layer_q_head_dim": layer_q_head_dim,
        "layer_k_head_dim": layer_k_head_dim,
        "layer_v_head_dim": layer_v_head_dim,
        "layer_rotary_dim": layer_rotary_dim,
        "layer_q_dim": layer_q_dim,
        "layer_kv_dim": layer_kv_dim,
        "layer_k_cache_offset": layer_k_cache_offset,
        "layer_v_cache_offset": layer_v_cache_offset,
        "kv_cache_layer_stride_variable": True,
        "kv_cache_token_stride_total": kv_cache_token_stride_total,
        "max_q_head_dim": max(layer_q_head_dim) if layer_q_head_dim else 0,
        "max_k_head_dim": max(layer_k_head_dim) if layer_k_head_dim else 0,
        "max_v_head_dim": max(layer_v_head_dim) if layer_v_head_dim else 0,
        "kv_cache_head_dim": max(
            max(layer_k_head_dim) if layer_k_head_dim else 0,
            max(layer_v_head_dim) if layer_v_head_dim else 0,
        ),
        "max_rotary_dim": max(layer_rotary_dim) if layer_rotary_dim else 0,
        "shared_kv_layers": shared_kv_layers,
        "first_shared_kv_layer": first_shared_kv_layer,
    }


def _build_config(model_dir: Path, arch: str, config_template: Path | None) -> dict[str, Any]:
    hf = _hf_config(model_dir)
    base = _load_runtime_config_template(config_template)
    text = hf.get("text_config") if isinstance(hf.get("text_config"), dict) else hf
    cfg = dict(base)
    arch_contract = _safetensors_arch_contract(arch)
    contract_config = (
        arch_contract.get("config")
        if isinstance(arch_contract.get("config"), dict)
        else {}
    )
    for key, value in contract_config.items():
        cfg.setdefault(str(key), value)
    config_builder = str(arch_contract.get("config_builder") or "").strip().lower()
    cfg.setdefault("model", arch)
    cfg.setdefault("model_type", arch)
    cfg.setdefault("num_layers", text.get("num_hidden_layers"))
    cfg.setdefault("num_hidden_layers", text.get("num_hidden_layers"))
    cfg.setdefault("embed_dim", text.get("hidden_size"))
    cfg.setdefault("hidden_size", text.get("hidden_size"))
    cfg.setdefault("intermediate_size", text.get("intermediate_size"))
    cfg.setdefault("num_heads", text.get("num_attention_heads"))
    cfg.setdefault("num_attention_heads", text.get("num_attention_heads"))
    cfg.setdefault("num_kv_heads", text.get("num_key_value_heads"))
    cfg.setdefault("num_key_value_heads", text.get("num_key_value_heads"))
    cfg.setdefault("head_dim", text.get("head_dim") or (int(text.get("hidden_size", 0)) // max(1, int(text.get("num_attention_heads", 1)))))
    cfg.setdefault("vocab_size", text.get("vocab_size"))
    cfg.setdefault("context_length", text.get("max_position_embeddings") or text.get("sliding_window"))
    for token_key in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        token_value = text.get(token_key)
        if token_value is None:
            token_value = hf.get(token_key)
        if isinstance(token_value, (list, tuple)):
            token_value = token_value[0] if token_value else None
        if token_value is not None:
            cfg.setdefault(token_key, int(token_value))
    rope_params = text.get("rope_parameters") if isinstance(text.get("rope_parameters"), dict) else {}
    full_rope = rope_params.get("full_attention") if isinstance(rope_params.get("full_attention"), dict) else {}
    sliding_rope = rope_params.get("sliding_attention") if isinstance(rope_params.get("sliding_attention"), dict) else {}
    rope_parameters = text.get("rope_parameters") if isinstance(text.get("rope_parameters"), dict) else {}
    cfg.setdefault("rope_theta", full_rope.get("rope_theta", rope_parameters.get("rope_theta", text.get("rope_theta", 1000000.0))))
    cfg.setdefault("rope_theta_swa", sliding_rope.get("rope_theta", 10000.0))
    mrope_sections_raw = rope_parameters.get("mrope_section") or []
    rope_interleaved = bool(rope_parameters.get("mrope_interleaved", False))
    if mrope_sections_raw:
        cfg.setdefault("rope_layout", "multi_section_1d")
    else:
        cfg.setdefault("rope_layout", "pairwise" if rope_interleaved else "split")
    cfg.setdefault("rope_param_mode", "per_layer_direct")
    cfg.setdefault("rms_eps", text.get("rms_norm_eps", text.get("rms_norm_epsilon", 1e-6)))
    cfg.setdefault("rms_norm_eps", cfg.get("rms_eps"))
    cfg.setdefault(
        "tie_word_embeddings",
        bool(text.get("tie_word_embeddings", hf.get("tie_word_embeddings", True))),
    )
    if arch in {"whisper_encoder", "whisper_decoder"}:
        embed_dim = int(hf.get("d_model") or 0)
        is_encoder = arch == "whisper_encoder"
        num_heads = int(
            hf.get("encoder_attention_heads" if is_encoder else "decoder_attention_heads")
            or 0
        )
        context_length = int(
            hf.get("max_source_positions" if is_encoder else "max_target_positions")
            or 0
        )
        feature_channels = int(hf.get("num_mel_bins") or 0)
        num_layers = int(
            hf.get("encoder_layers" if is_encoder else "decoder_layers")
            or hf.get("num_hidden_layers")
            or 0
        )
        intermediate_size = int(
            hf.get("encoder_ffn_dim" if is_encoder else "decoder_ffn_dim")
            or 0
        )
        if min(
            embed_dim,
            num_heads,
            context_length,
            feature_channels,
            num_layers,
            intermediate_size,
        ) <= 0:
            raise SystemExit(f"{arch} config is missing required dimensions")
        if embed_dim % num_heads != 0:
            raise SystemExit(
                f"{arch} d_model={embed_dim} is not divisible by heads={num_heads}"
            )
        contract = _safetensors_arch_contract(arch)
        contract_cfg = (
            contract.get("config") if isinstance(contract.get("config"), dict) else {}
        )
        cfg.update(contract_cfg)
        cfg.update(
            {
                "num_layers": num_layers,
                "num_hidden_layers": num_layers,
                "embed_dim": embed_dim,
                "hidden_size": embed_dim,
                "intermediate_size": intermediate_size,
                "num_heads": num_heads,
                "num_attention_heads": num_heads,
                "num_kv_heads": num_heads,
                "num_key_value_heads": num_heads,
                "head_dim": embed_dim // num_heads,
                "context_length": context_length,
                "attention_scale": 1.0 / math.sqrt(float(embed_dim // num_heads)),
                "rms_eps": 1.0e-5,
                "prefer_q8_activation": False,
                "numerical_contract_mode": "production",
                "tie_word_embeddings": bool(hf.get("tie_word_embeddings", True)),
                "vocab_size": 1 if is_encoder else int(hf.get("vocab_size") or 0),
            }
        )
        if is_encoder:
            cfg.update(
                {
                    "audio_sample_rate": 16000,
                    "audio_sample_extent": context_length * 2 * 160,
                    "audio_max_source_frames": context_length * 2 * 160 * 3,
                    "audio_resample_radius": 16,
                    "audio_n_fft": 400,
                    "audio_hop_length": 160,
                    "audio_power_bins": 201,
                    "audio_feature_channels": feature_channels,
                    "audio_feature_frames": context_length * 2,
                    "audio_conv1_output_channels": embed_dim,
                    "audio_conv2_output_channels": embed_dim,
                    "audio_conv1_kernel_size": 3,
                    "audio_conv1_stride": 1,
                    "audio_conv1_padding": 1,
                    "audio_conv1_output_frames": context_length * 2,
                    "audio_conv1_elements": embed_dim * context_length * 2,
                    "audio_conv2_kernel_size": 3,
                    "audio_conv2_stride": 2,
                    "audio_conv2_padding": 1,
                    "audio_conv2_output_frames": context_length,
                    "audio_conv2_elements": embed_dim * context_length,
                }
            )
        else:
            cfg.update(
                {
                    "encoder_memory_length": int(hf.get("max_source_positions") or 0),
                    "decoder_start_token_id": int(hf.get("decoder_start_token_id") or 0),
                    "eos_token_id": int(hf.get("eos_token_id") or 0),
                    "pad_token_id": int(hf.get("pad_token_id") or 0),
                    "uses_cross_attention": True,
                }
            )
    if arch == "nemotron_h":
        layer_kinds = _parse_nemotron_h_pattern(str(text.get("hybrid_override_pattern") or ""), int(cfg.get("num_layers") or 0))
        mamba_num_heads = int(text.get("mamba_num_heads") or 0)
        mamba_head_dim = int(text.get("mamba_head_dim") or 0)
        ssm_state_size = int(text.get("ssm_state_size") or 0)
        ssm_group_count = int(text.get("n_groups") or 0)
        ssm_conv_kernel = int(text.get("conv_kernel") or 0)
        ssm_inner_size = mamba_num_heads * mamba_head_dim
        mamba_conv_dim = ssm_inner_size + 2 * ssm_group_count * ssm_state_size
        mamba_projection_size = ssm_inner_size + mamba_conv_dim + mamba_num_heads
        cfg.update({
            "model": "nemotron_h",
            "arch": "nemotron_h",
            "model_type": "nemotron_h",
            "layer_kinds": layer_kinds,
            "hybrid_block_pattern": layer_kinds[:],
            "layer_state_policy": ["mamba2" if k == "mamba" else "none" for k in layer_kinds],
            "layer_attention_policy": ["full_attention" if k == "attention" else "none" for k in layer_kinds],
            "layer_recurrent_policy": ["mamba2" if k == "mamba" else "none" for k in layer_kinds],
            "layer_moe_policy": ["routed_relu2" if k == "moe" else "none" for k in layer_kinds],
            "layer_mlp_policy": ["relu2" if k == "mlp" else "none" for k in layer_kinds],
            "layer_kv_policy": ["attention_kv_cache" if k == "attention" else "none" for k in layer_kinds],
            "intermediate_dim": int(text.get("intermediate_size") or 0),
            "mamba_num_heads": mamba_num_heads,
            "mamba_head_dim": mamba_head_dim,
            "ssm_state_size": ssm_state_size,
            "ssm_conv_kernel": ssm_conv_kernel,
            "ssm_projection_layout": "mamba2_v_qk_dt",
            "ssm_conv_history_mode": "kernel_width",
            "recurrent_state_layout": "heads_head_dim_state",
            "ssm_group_count": ssm_group_count,
            "mamba_norm_group_size": int(ssm_inner_size // ssm_group_count) if ssm_group_count else int(mamba_head_dim),
            "chunk_size": int(text.get("chunk_size") or 0),
            "mamba_dt_min": _nemotron_dt_limit(text)[0],
            "mamba_dt_max": _nemotron_dt_limit(text)[1],
            "ssm_inner_size": ssm_inner_size,
            "ssm_conv_channels": mamba_conv_dim,
            "ssm_conv_history": max(ssm_conv_kernel, 0),
            "recurrent_num_heads": mamba_num_heads,
            "recurrent_head_dim": mamba_head_dim,
            "recurrent_state_heads": mamba_num_heads,
            "recurrent_state_rows": mamba_head_dim,
            "recurrent_state_cols": ssm_state_size,
            "mamba_conv_dim": mamba_conv_dim,
            "mamba_projection_size": mamba_projection_size,
            "moe_intermediate_size": int(text.get("moe_intermediate_size") or 0),
            "moe_shared_expert_intermediate_size": int(text.get("moe_shared_expert_intermediate_size") or 0),
            "n_routed_experts": int(text.get("n_routed_experts") or 0),
            "num_experts_per_tok": int(text.get("num_experts_per_tok") or 0),
            "n_group": int(text.get("n_group") or text.get("n_groups") or 0),
            "topk_group": int(text.get("topk_group") or 0),
            "norm_topk_prob": bool(text.get("norm_topk_prob", True)),
            "routed_scaling_factor": float(text.get("routed_scaling_factor") or 1.0),
            "rope_layout": "split",
            "rope_theta": float(text.get("rope_theta") or 10000.0),
            "rope_freq_base": float(text.get("rope_theta") or 10000.0),
            "use_rope_freq_factors": 0,
            "has_attention_biases": bool(text.get("attention_bias", False)),
            "has_qk_norm": False,
            "prefill_policy": "batched",
        })

    if arch in {"kimi_vl", "instella_moe"}:
        rope_scaling = text.get("rope_scaling") if isinstance(text.get("rope_scaling"), dict) else {}
        first_dense = int(text.get("first_k_dense_replace") or 0)
        moe_freq = int(text.get("moe_layer_freq") or 1)
        num_layers = int(cfg.get("num_layers") or 0)
        layer_kinds = []
        for layer in range(num_layers):
            if layer < first_dense or (moe_freq > 0 and layer % moe_freq != 0):
                layer_kinds.append("mla_dense_mlp")
            else:
                layer_kinds.append("mla_moe")
        if arch == "instella_moe":
            farskip_start = int(text.get("farskip_start_idx") or 0)
            farskip_end = min(
                int(text.get("farskip_end_idx") or num_layers - 1),
                num_layers - 1,
            )
            first_farskip = True
            for layer, kind in enumerate(layer_kinds):
                if not bool(text.get("farskip", False)):
                    continue
                if not farskip_start <= layer <= farskip_end:
                    continue
                if kind == "mla_dense_mlp":
                    layer_kinds[layer] = "mla_gated_farskip_dense_mlp"
                elif kind == "mla_moe":
                    layer_kinds[layer] = (
                        "mla_gated_farskip_moe_first"
                        if first_farskip
                        else "mla_gated_farskip_moe"
                    )
                    first_farskip = False
            layer_kinds = [
                "mla_gated_dense_mlp" if kind == "mla_dense_mlp" else kind
                for kind in layer_kinds
            ]
        qk_nope = int(text.get("qk_nope_head_dim") or 0)
        qk_rope = int(text.get("qk_rope_head_dim") or 0)
        v_head = int(text.get("v_head_dim") or 0)
        cfg.update({
            "model": arch,
            "arch": arch,
            "model_type": arch,
            "layer_kinds": layer_kinds,
            "hybrid_block_pattern": layer_kinds[:],
            "layer_attention_policy": ["mla" for _ in layer_kinds],
            "layer_moe_policy": ["routed_swiglu" if "moe" in k else "none" for k in layer_kinds],
            "layer_mlp_policy": ["swiglu" if "dense_mlp" in k else "none" for k in layer_kinds],
            "layer_kv_policy": ["compressed_mla_kv" for _ in layer_kinds],
            "intermediate_dim": int(text.get("intermediate_size") or cfg.get("intermediate_size") or 0),
            "moe_intermediate_size": int(text.get("moe_intermediate_size") or 0),
            "n_shared_experts": int(text.get("n_shared_experts") or 0),
            "n_routed_experts": int(text.get("n_routed_experts") or 0),
            "num_experts_per_tok": int(text.get("num_experts_per_tok") or 0),
            "n_group": int(text.get("n_group") or 0),
            "topk_group": int(text.get("topk_group") or 0),
            "norm_topk_prob": bool(text.get("norm_topk_prob", True)),
            "router_num_groups": int(text.get("n_group") or 0),
            "router_topk_group": int(text.get("topk_group") or 0),
            "router_norm_topk_prob": 1 if bool(text.get("norm_topk_prob", True)) else 0,
            "routed_scaling_factor": float(text.get("routed_scaling_factor") or 1.0),
            "scoring_func": str(text.get("scoring_func") or "sigmoid"),
            "topk_method": str(text.get("topk_method") or "noaux_tc"),
            "kv_lora_rank": int(text.get("kv_lora_rank") or 0),
            "q_lora_rank": text.get("q_lora_rank"),
            "qk_nope_head_dim": qk_nope,
            "qk_rope_head_dim": qk_rope,
            "v_head_dim": v_head,
            "head_dim": qk_nope + qk_rope,
            "attention_scale": _effective_attention_scale(
                qk_nope + qk_rope, rope_scaling
            ),
            "rotary_dim": qk_rope,
            "mla_q_head_dim": qk_nope + qk_rope,
            "mla_k_head_dim": qk_nope + qk_rope,
            "mla_v_head_dim": v_head,
            "rope_layout": (
                "partial_interleaved_yarn"
                if arch == "instella_moe" and bool(text.get("rope_interleave", False))
                else "partial_pairwise_concat"
            ),
            "rope_theta": float(text.get("rope_theta") or 10000.0),
            "rope_freq_base": float(text.get("rope_theta") or 10000.0),
            "rope_scaling_type": str(rope_scaling.get("type") or "none"),
            "rope_scaling_factor": float(rope_scaling.get("factor") or 1.0),
            "rope_original_context_length": int(
                rope_scaling.get("original_max_position_embeddings")
                or text.get("max_position_embeddings")
                or cfg.get("context_length")
                or 0
            ),
            "rope_beta_fast": float(rope_scaling.get("beta_fast") or 0.0),
            "rope_beta_slow": float(rope_scaling.get("beta_slow") or 0.0),
            "rope_mscale": float(rope_scaling.get("mscale") or 1.0),
            "rope_mscale_all_dim": float(rope_scaling.get("mscale_all_dim") or 1.0),
            "has_attention_biases": False,
            "has_qk_norm": bool(text.get("qk_layernorm", False)) if arch == "instella_moe" else False,
            "gated_attention": bool(text.get("gated_attention", False)),
            "farskip": bool(text.get("farskip", False)),
            "farskip_start_idx": int(text.get("farskip_start_idx") or 0),
            "farskip_end_idx": int(text.get("farskip_end_idx") or num_layers - 1),
            "attn_only_farskip": bool(text.get("attn_only_farskip", False)),
            "mlp_only_farskip": bool(text.get("mlp_only_farskip", False)),
            "rope_interleave": bool(text.get("rope_interleave", False)),
            "rope_scaling": rope_scaling or None,
            "moe_shared_intermediate_size": int(text.get("moe_intermediate_size") or 0) * int(text.get("n_shared_experts") or 0),
            "prefill_policy": "batched",
        })

    if arch == "gemma4_assistant":
        headers = _load_safetensors_headers(model_dir)
        num_layers = int(cfg.get("num_layers") or 0)
        layer_types = [str(x) for x in (text.get("layer_types") or [])]
        if not layer_types and num_layers > 0:
            layer_types = ["full_attention" if layer == num_layers - 1 else "sliding_attention" for layer in range(num_layers)]
        layer_kinds = [
            "full_attention_q_only_k_eq_v" if kind == "full_attention" else "sliding_attention_q_only_k_eq_v"
            for kind in layer_types
        ]
        layer_q_dim: list[int] = []
        layer_o_input_dim: list[int] = []
        layer_q_norm_dim: list[int] = []
        layer_q_head_dim: list[int] = []
        layer_rotary_dim: list[int] = []
        for layer in range(num_layers):
            q = headers.get(f"model.layers.{layer}.self_attn.q_proj.weight")
            o = headers.get(f"model.layers.{layer}.self_attn.o_proj.weight")
            qn = headers.get(f"model.layers.{layer}.self_attn.q_norm.weight")
            layer_q_dim.append(int(q.shape[0]) if q and q.shape else int(cfg.get("num_heads") or 0) * int(cfg.get("head_dim") or 0))
            layer_o_input_dim.append(int(o.shape[1]) if o and len(o.shape) >= 2 else layer_q_dim[-1])
            layer_q_norm_dim.append(int(qn.shape[0]) if qn and qn.shape else int(cfg.get("head_dim") or 0))
            q_head = layer_q_norm_dim[-1]
            layer_q_head_dim.append(q_head)
            layer_rotary_dim.append(q_head)
        rope_params = text.get("rope_parameters") if isinstance(text.get("rope_parameters"), dict) else {}
        full_rope = rope_params.get("full_attention") if isinstance(rope_params.get("full_attention"), dict) else {}
        sliding_rope = rope_params.get("sliding_attention") if isinstance(rope_params.get("sliding_attention"), dict) else {}
        cfg.update({
            "model": "gemma4_assistant",
            "arch": "gemma4_assistant",
            "model_type": "gemma4_assistant",
            "assistant_role": "mtp_drafter",
            "backbone_hidden_size": int(hf.get("backbone_hidden_size") or 0),
            "attention_k_eq_v": bool(text.get("attention_k_eq_v", True)),
            "num_kv_heads": int(cfg.get("num_heads") or text.get("num_attention_heads") or 0),
            "num_key_value_heads": int(cfg.get("num_heads") or text.get("num_attention_heads") or 0),
            "hidden_activation": str(text.get("hidden_activation") or "gelu_pytorch_tanh"),
            "intermediate_dim": int(cfg.get("intermediate_size") or 0),
            "max_seq_len": int(cfg.get("context_length") or 0),
            "global_head_dim": int(text.get("global_head_dim") or cfg.get("head_dim") or 0),
            "num_global_key_value_heads": int(text.get("num_global_key_value_heads") or 0),
            "num_kv_shared_layers": int(text.get("num_kv_shared_layers") or num_layers),
            "layer_types": layer_types,
            "layer_kinds": layer_kinds,
            "hybrid_block_pattern": layer_kinds[:],
            "layer_attention_policy": [
                "q_only_full_attention" if kind.startswith("full") else "q_only_sliding_attention"
                for kind in layer_kinds
            ],
            "layer_kv_policy": ["q_equals_k_equals_v" for _ in layer_kinds],
            "layer_recurrent_policy": ["none" for _ in layer_kinds],
            "layer_state_policy": ["none" for _ in layer_kinds],
            "layer_moe_policy": ["none" for _ in layer_kinds],
            "layer_mlp_policy": ["gelu_pytorch_tanh" for _ in layer_kinds],
            "layer_rope_kind": ["full" if kind.startswith("full") else "sliding" for kind in layer_kinds],
            "layer_sliding_window": [int(text.get("sliding_window") or 0) if not kind.startswith("full") else 0 for kind in layer_kinds],
            "layer_q_dim": layer_q_dim,
            "layer_o_input_dim": layer_o_input_dim,
            "layer_q_norm_dim": layer_q_norm_dim,
            "layer_q_head_dim": layer_q_head_dim,
            "layer_k_head_dim": layer_q_head_dim[:],
            "layer_v_head_dim": layer_q_head_dim[:],
            "layer_rotary_dim": layer_rotary_dim,
            "rope_layout": "split",
            "rope_param_mode": "per_layer_direct",
            "rope_theta": float(full_rope.get("rope_theta", cfg.get("rope_theta", 1000000.0)) or 1000000.0),
            "rope_theta_swa": float(sliding_rope.get("rope_theta", cfg.get("rope_theta_swa", 10000.0)) or 10000.0),
            "rope_partial_rotary_factor": float(full_rope.get("partial_rotary_factor", 1.0) or 1.0),
            "has_attention_biases": bool(text.get("attention_bias", False)),
            "has_qk_norm": True,
            "has_k_norm": False,
            "prefill_policy": "batched",
            "tie_word_embeddings": bool(hf.get("tie_word_embeddings", text.get("tie_word_embeddings", True))),
            "use_ordered_embeddings": bool(hf.get("use_ordered_embeddings", False)),
            "assistant_pre_projection": True,
            "assistant_post_projection": True,
            "assistant_projection_mode": "mtp_bridge",
            "assistant_layer_scalar_mode": "layer_output_scale",
            "standalone_text_inference_supported": False,
        })

    if arch == "gemma4":
        headers = _load_safetensors_headers(model_dir)
        attention_plan = _build_gemma4_attention_plan_from_hf(text, headers)
        cfg.update({
            "model": "gemma4",
            "arch": "gemma4",
            "model_type": "gemma4",
            "intermediate_dim": int(cfg.get("intermediate_size") or 0),
            "max_seq_len": int(cfg.get("context_length") or 0),
            "has_attention_biases": bool(text.get("attention_bias", False)),
            "has_qk_norm": True,
            "prefill_policy": "batched",
            "gemma4_per_layer_embedding": True,
            "rope_layout": "split",
            "rope_param_mode": "per_layer_direct",
            "num_kv_shared_layers": int(text.get("num_kv_shared_layers") or 0),
            "sampler_defaults": {"repeat_penalty": 1.05, "repeat_last_n": 64},
        })
        cfg.update(attention_plan)

    if arch == "glm4":
        partial_key = "partial_rotary_factor"
        contract = _safetensors_arch_contract("glm4")
        contract_cfg = contract.get("config") if isinstance(contract.get("config"), dict) else {}
        if isinstance(contract_cfg.get("partial_rotary_factor_key"), str):
            partial_key = str(contract_cfg["partial_rotary_factor_key"])
        partial = float(text.get(partial_key, text.get("partial_rotary_factor", 1.0)) or 1.0)
        head_dim_value = int(cfg.get("head_dim") or 0)
        rotary_dim = int(head_dim_value * partial) if head_dim_value > 0 else int(cfg.get("rotary_dim") or 0)
        cfg.update({k: v for k, v in contract_cfg.items() if k != "partial_rotary_factor_key"})
        cfg.update({
            "model": "glm4",
            "arch": "glm4",
            "model_type": "glm4",
            "intermediate_dim": int(cfg.get("intermediate_size") or 0),
            "attn_out_dim": int(cfg.get("num_heads") or 0) * head_dim_value,
            "rotary_dim": rotary_dim,
            "partial_rotary_factor": partial,
            "max_seq_len": int(cfg.get("context_length") or 0),
            "rms_eps": float(text.get("rms_norm_eps", cfg.get("rms_eps", 1e-5)) or 1e-5),
            "rms_norm_eps": float(text.get("rms_norm_eps", cfg.get("rms_eps", 1e-5)) or 1e-5),
            "rope_theta": float(text.get("rope_theta", cfg.get("rope_theta", 10000.0)) or 10000.0),
            "rope_freq_base": float(text.get("rope_theta", cfg.get("rope_theta", 10000.0)) or 10000.0),
            "has_attention_biases": bool(text.get("attention_bias", True)),
            "has_qk_norm": False,
            "prefill_policy": "batched",
        })


    if config_builder == "cohere2_text":
        sliding_rope = (
            rope_parameters.get("sliding_attention")
            if isinstance(rope_parameters.get("sliding_attention"), dict)
            else {}
        )
        layer_kinds = [str(kind) for kind in (text.get("layer_types") or [])]
        num_layers = int(cfg.get("num_layers") or 0)
        if not layer_kinds and num_layers > 0:
            layer_kinds = [
                "full_attention" if (layer + 1) % 4 == 0 else "sliding_attention"
                for layer in range(num_layers)
            ]
        if len(layer_kinds) != num_layers:
            raise SystemExit(
                f"{arch} layer_types has {len(layer_kinds)} entries for {num_layers} layers"
            )
        mrope = [int(value) for value in (sliding_rope.get("mrope_section") or [])]
        head_dim = int(cfg.get("head_dim") or 0)
        cfg.update({
            "model": arch,
            "arch": arch,
            "model_type": arch,
            "layer_kinds": layer_kinds,
            "hybrid_block_pattern": layer_kinds[:],
            "layer_attention_policy": layer_kinds[:],
            "layer_mlp_policy": ["parallel_swiglu" for _ in layer_kinds],
            "layer_kv_policy": ["attention_kv_cache" for _ in layer_kinds],
            "intermediate_dim": int(cfg.get("intermediate_size") or 0),
            "attn_out_dim": int(cfg.get("num_heads") or 0) * head_dim,
            "rotary_dim": head_dim,
            "max_seq_len": int(cfg.get("context_length") or 0),
            "rms_eps": float(text.get("layer_norm_eps", cfg.get("rms_eps", 1e-5)) or 1e-5),
            "rms_norm_eps": float(text.get("layer_norm_eps", cfg.get("rms_eps", 1e-5)) or 1e-5),
            "sliding_window": int(text.get("sliding_window") or 0),
            "logit_scale": float(text.get("logit_scale") or 1.0),
            "rope_layout": "multi_section_1d",
            "rope_param_mode": "per_layer_direct",
            "rope_theta": float(sliding_rope.get("rope_theta") or 10000.0),
            "rope_freq_base": float(sliding_rope.get("rope_theta") or 10000.0),
            "has_attention_biases": bool(text.get("attention_bias", False)),
            "has_qk_norm": False,
            "prefill_policy": "batched",
            "image_token_id": int(hf.get("image_token_id") or 0),
            "video_token_id": int(hf.get("video_token_id") or 0),
            "num_deepstack_layers": len((hf.get("vision_config") or {}).get("deepstack_visual_indexes") or []),
        })
        if mrope:
            cfg["mrope_sections"] = mrope + ([0] if len(mrope) == 3 else [])
            cfg["mrope_n_dims"] = head_dim or sum(mrope)
            cfg["mrope_interleaved"] = bool(sliding_rope.get("mrope_interleaved", False))

    if arch == "qwen3vl":
        rope_scaling = text.get("rope_scaling") if isinstance(text.get("rope_scaling"), dict) else {}
        rope_params_effective = rope_scaling or rope_parameters
        mrope = list(rope_params_effective.get("mrope_section") or [])
        cfg.update({
            "model": "qwen3vl",
            "arch": "qwen3vl",
            "model_type": "qwen3vl",
            "intermediate_dim": int(cfg.get("intermediate_size") or 0),
            "attn_out_dim": int(cfg.get("num_heads") or 0) * int(cfg.get("head_dim") or 0),
            "rotary_dim": int(cfg.get("head_dim") or 0),
            "max_seq_len": int(cfg.get("context_length") or 0),
            "rope_layout": "multi_section_1d" if mrope else "pairwise",
            "rope_param_mode": "per_layer_direct",
            "rope_theta": float(text.get("rope_theta", cfg.get("rope_theta", 5000000.0)) or 5000000.0),
            "rope_freq_base": float(text.get("rope_theta", cfg.get("rope_theta", 5000000.0)) or 5000000.0),
            "has_attention_biases": bool(text.get("attention_bias", False)),
            "has_qk_norm": True,
            "prefill_policy": "batched",
            "image_token_id": int(hf.get("image_token_id") or 0),
            "video_token_id": int(hf.get("video_token_id") or 0),
            "vision_start_token_id": int(hf.get("vision_start_token_id") or 0),
            "vision_end_token_id": int(hf.get("vision_end_token_id") or 0),
            "num_deepstack_layers": len((hf.get("vision_config") or {}).get("deepstack_visual_indexes") or []),
            "decoder_norm_storage_boundary": "bf16",
            "decoder_norm_reduction_policy": "pytorch_avx2_cascade_exact",
            "decoder_qk_norm_reduction_policy": "pytorch_avx2_cascade_exact",
            "decoder_projection_reduction_policy": "pytorch_onednn_brgemm_exact",
            "decoder_residual_storage_boundary": "bf16",
            "decoder_mrope_storage_boundary": "pytorch_bf16_exact",
            "decoder_swiglu_storage_boundary": "pytorch_bf16_exact",
            "decode_kv_cache_dtype": "bf16",
        })
        if mrope:
            cfg["mrope_sections"] = [int(v) for v in mrope] + ([0] if len(mrope) == 3 else [])
            cfg["mrope_n_dims"] = int(cfg.get("head_dim") or sum(int(v) for v in mrope))
            cfg["mrope_interleaved"] = bool(rope_params_effective.get("mrope_interleaved", False))

    if arch == "qwen3_vl_vision" or config_builder == "qwen3_vl_vision":
        vision = hf.get("vision_config") if isinstance(hf.get("vision_config"), dict) else {}
        headers = _load_safetensors_headers(model_dir)
        hidden = int(vision.get("hidden_size") or 0)
        num_heads = int(vision.get("num_heads") or vision.get("num_attention_heads") or 0)
        head_dim = int(hidden // max(1, num_heads)) if hidden else 0
        depth = int(vision.get("depth") or vision.get("num_hidden_layers") or 0)
        patch_size = int(vision.get("patch_size") or 16)
        temporal_patch_size = int(vision.get("temporal_patch_size") or 2)
        channels = int(vision.get("num_channels") or vision.get("in_channels") or 3)
        pos_rows = int(vision.get("num_position_embeddings") or 0)
        pos_header = headers.get("model.visual.pos_embed.weight")
        if pos_header and pos_header.shape:
            pos_rows = int(pos_header.shape[0])
        grid = int(round(pos_rows ** 0.5)) if pos_rows > 0 else 0
        if grid * grid != pos_rows:
            grid = pos_rows
        merge = int(vision.get("spatial_merge_size") or hf.get("spatial_merge_size") or 2)
        merge_factor = merge * merge
        projector_in = hidden * merge_factor
        merger_fc1 = headers.get("model.visual.merger.linear_fc1.weight")
        merger_fc2 = headers.get("model.visual.merger.linear_fc2.weight")
        projector_hidden = int(merger_fc1.shape[0]) if merger_fc1 and merger_fc1.shape else projector_in
        projector_out = int(vision.get("out_hidden_size") or (merger_fc2.shape[0] if merger_fc2 and merger_fc2.shape else text.get("hidden_size") or 0))
        deepstack_layers = [int(x) for x in (vision.get("deepstack_visual_indexes") or vision.get("deepstack_layer_indices") or [])]
        preproc = {}
        preproc_path = model_dir / "preprocessor_config.json"
        if preproc_path.exists():
            try:
                preproc = json.loads(preproc_path.read_text(encoding="utf-8"))
            except Exception:
                preproc = {}
        if head_dim > 0:
            axis_pairs = max(1, head_dim // 4)
            # Vision M-RoPE has two spatial axes. Keep the four-slot ABI, but
            # do not populate decoder-only time/extra sections.
            vision_mrope_sections = [axis_pairs, axis_pairs, 0, 0]
        else:
            vision_mrope_sections = [1, 1, 0, 0]
        cfg.update({
            "model": arch,
            "arch": arch,
            "model_type": arch,
            "num_layers": depth,
            "num_hidden_layers": depth,
            "embed_dim": hidden,
            "hidden_size": hidden,
            "intermediate_size": int(vision.get("intermediate_size") or 0),
            "intermediate_dim": int(vision.get("intermediate_size") or 0),
            "num_heads": num_heads,
            "num_attention_heads": num_heads,
            "num_kv_heads": num_heads,
            "num_key_value_heads": num_heads,
            "head_dim": head_dim,
            "attn_out_dim": hidden,
            "q_dim": hidden,
            "k_dim": hidden,
            "v_dim": hidden,
            "context_length": pos_rows,
            "max_seq_len": pos_rows,
            "vocab_size": int(text.get("vocab_size") or 0),
            "n_vocab": int(text.get("vocab_size") or 0),
            "image_size": int(grid * patch_size) if grid > 0 else int(preproc.get("size", {}).get("shortest_edge", 0) or 0),
            "image_height": int(grid * patch_size) if grid > 0 else 0,
            "image_width": int(grid * patch_size) if grid > 0 else 0,
            "patch_size": patch_size,
            "temporal_patch_size": temporal_patch_size,
            "vision_channels": channels,
            "patch_dim": channels * patch_size * patch_size,
            "vision_grid_h": grid,
            "vision_grid_w": grid,
            "position_grid_size": grid,
            "position_interpolation_policy": "align_corners_bilinear",
            "vision_position_storage_boundary": "bf16",
            "vision_num_patches": pos_rows,
            "spatial_merge_size": merge,
            "spatial_merge_factor": merge_factor,
            "vision_merged_tokens": int(pos_rows // max(1, merge_factor)),
            "projector_in_dim": projector_in,
            "projector_hidden_dim": projector_hidden,
            "projector_out_dim": projector_out,
            "projector_total_out_dim": int(projector_out * (1 + len(deepstack_layers))),
            "projection_dim": projector_out,
            "deepstack_layer_indices": deepstack_layers,
            "num_deepstack_layers": len(deepstack_layers),
            "image_mean": [float(v) for v in preproc.get("image_mean", [0.48145466, 0.4578275, 0.40821073])[:3]],
            "image_std": [float(v) for v in preproc.get("image_std", [0.26862954, 0.26130258, 0.27577711])[:3]],
            "image_min_pixels": int(preproc.get("min_pixels") or 0),
            "image_max_pixels": int(preproc.get("max_pixels") or 0),
            "preproc_image_size": int(preproc.get("size", {}).get("shortest_edge", 0) or 0) if isinstance(preproc.get("size"), dict) else 0,
            "rope_layout": "multi_section_2d",
            "vision_mrope_sections": vision_mrope_sections,
            "vision_mrope_n_dims": max(1, int(head_dim)),
            "vision_patch_projection_reduction_policy": "pytorch_onednn_conv3d_exact",
            "vision_patch_frontend": "integrated_temporal2",
            "vision_mrope_storage_boundary": "bf16",
            "vision_mrope_reduction_policy": "pytorch_mkl_exact",
            "vision_layernorm_storage_boundary": "bf16",
            "vision_layernorm_reduction_policy": "pytorch_welford_exact",
            "vision_projection_storage_boundary": "bf16",
            "vision_projection_reduction_policy": "pytorch_onednn_brgemm_exact",
            "vision_attention_storage_boundary": "bf16",
            "vision_residual_storage_boundary": "bf16",
            "vision_activation_storage_boundary": "bf16",
            "vision_mrope_freq_base": 10000.0,
            "vision_mrope_freq_scale": 1.0,
            "vision_mrope_ext_factor": 0.0,
            "vision_mrope_attn_factor": 1.0,
            "vision_mrope_beta_fast": 32.0,
            "vision_mrope_beta_slow": 1.0,
            "vision_mrope_original_context_length": 32768,
            "prefer_q8_activation": False,
            "has_vision_encoder": True,
            "dtype": "bf16",
        })

    if arch == "qwen35":
        architecture = _qwen35_architecture_metadata(hf)
        headers = _load_safetensors_headers(model_dir)
        q_gate_proj_dim = 0
        attn_out_dim = 0
        recurrent_q_dim = int(text.get("linear_num_key_heads", 0) or 0) * int(text.get("linear_key_head_dim", 0) or 0)
        recurrent_k_dim = recurrent_q_dim
        recurrent_v_dim = int(text.get("linear_num_value_heads", 0) or 0) * int(text.get("linear_value_head_dim", 0) or 0)
        recurrent_qkv_weight_dtypes: set[str] = set()
        ssm_time_step_rank = 0
        for name, h in headers.items():
            if name.endswith(".self_attn.q_proj.weight") and len(h.shape) >= 2:
                q_gate_proj_dim = int(h.shape[0])
                if q_gate_proj_dim % 2 == 0:
                    attn_out_dim = q_gate_proj_dim // 2
            elif name.endswith(".linear_attn.in_proj_qkv.weight") and len(h.shape) >= 2:
                recurrent_qkv_weight_dtypes.add(_header_dtype_to_ck(h.dtype)[0])
                total = int(h.shape[0])
                if recurrent_q_dim <= 0 or recurrent_k_dim <= 0 or recurrent_v_dim <= 0:
                    recurrent_q_dim = total // 3
                    recurrent_k_dim = total // 3
                    recurrent_v_dim = total - recurrent_q_dim - recurrent_k_dim
            elif name.endswith(".linear_attn.in_proj_a.weight") and len(h.shape) >= 2:
                ssm_time_step_rank = max(ssm_time_step_rank, int(h.shape[0]))
        if attn_out_dim <= 0:
            attn_out_dim = int(text.get("num_attention_heads", cfg.get("num_heads", 0))) * int(text.get("head_dim", cfg.get("head_dim", 0)))
        if q_gate_proj_dim <= 0:
            q_gate_proj_dim = attn_out_dim * 2
        if ssm_time_step_rank <= 0:
            ssm_time_step_rank = int(text.get("linear_num_key_heads") or text.get("linear_num_value_heads") or 0)
        if len(recurrent_qkv_weight_dtypes) > 1:
            raise SystemExit(
                "Qwen3.5 recurrent QKV projection weights must use one storage dtype; "
                f"found {sorted(recurrent_qkv_weight_dtypes)}"
            )
        recurrent_qkv_weight_dtype = (
            next(iter(recurrent_qkv_weight_dtypes))
            if recurrent_qkv_weight_dtypes
            else ""
        )

        layer_types = list(architecture["layer_types"])
        layer_kinds = list(architecture["layer_kinds"])
        execution_plan = build_qwen35_execution_plan(layer_kinds)
        cfg.update({
            "model": "qwen35",
            "arch": "qwen35",
            "model_type": "qwen35",
            "attn_out_dim": int(attn_out_dim),
            "attn_q_gate_proj_dim": int(q_gate_proj_dim),
            "intermediate_dim": int(cfg.get("intermediate_size") or 0),
            "max_seq_len": int(cfg.get("context_length") or 0),
            "rotary_dim": int(architecture["rotary_dim"]),
            "mrope_sections": list(architecture["mrope_sections"]),
            "mrope_n_dims": int(architecture["mrope_n_dims"]),
            "mrope_interleaved": bool(architecture["mrope_interleaved"]),
            "has_qk_norm": any(kind == "full_attention" for kind in layer_kinds),
            "has_attention_biases": False,
            "layer_types": layer_types,
            "layer_kinds": layer_kinds,
            "hybrid_block_pattern": layer_kinds[:],
            "layer_execution_plan": execution_plan["layer_execution_plan"],
            "layer_state_policy": execution_plan["layer_state_policy"],
            "layer_attention_policy": execution_plan["layer_attention_policy"],
            "layer_recurrent_policy": execution_plan["layer_recurrent_policy"],
            "layer_kv_policy": execution_plan["layer_kv_policy"],
            "prefill_policy": "batched",
            "recurrent_state_physical_layout": (
                "head_key_value_contiguous"
                if recurrent_qkv_weight_dtype == "bf16"
                else "head_value_key_contiguous"
            ),
            "full_attention_interval": int(architecture["full_attention_interval"]),
            "ssm_conv_kernel": int(text.get("linear_conv_kernel_dim") or 4),
            "ssm_state_size": int(text.get("linear_key_head_dim") or max(1, recurrent_q_dim)),
            "ssm_group_count": int(text.get("linear_num_key_heads") or max(1, recurrent_q_dim // max(1, int(text.get("linear_key_head_dim") or recurrent_q_dim)))),
            "ssm_time_step_rank": int(ssm_time_step_rank),
            "ssm_inner_size": int(recurrent_v_dim),
            "q_dim": int(recurrent_q_dim),
            "k_dim": int(recurrent_k_dim),
            "v_dim": int(recurrent_v_dim),
            "gate_dim": int(ssm_time_step_rank),
            "ssm_conv_channels": int(recurrent_q_dim + recurrent_k_dim + recurrent_v_dim),
            "recurrent_num_heads": int(ssm_time_step_rank),
            "recurrent_head_dim": int(recurrent_v_dim // max(1, ssm_time_step_rank)) if ssm_time_step_rank else int(text.get("linear_key_head_dim") or 0),
            "sampler_defaults": {"repeat_penalty": 1.12, "repeat_last_n": 96, "no_repeat_ngram_size": 4},
        })
        if recurrent_qkv_weight_dtype:
            cfg["recurrent_qkv_weight_dtype"] = recurrent_qkv_weight_dtype
        if recurrent_qkv_weight_dtype == "bf16":
            cfg.update({
                "decoder_norm_storage_boundary": "bf16",
                "decoder_norm_reduction_policy": "pytorch_avx2_cascade_exact",
                "decoder_qk_norm_reduction_policy": "pytorch_avx2_cascade_exact",
                "decoder_projection_reduction_policy": "pytorch_onednn_brgemm_exact",
                "decoder_residual_storage_boundary": "bf16",
                "decoder_swiglu_storage_boundary": "pytorch_bf16_exact",
                "decode_kv_cache_dtype": "bf16",
            })
        cfg.update({
            key: architecture[key]
            for key in (
                "has_vision_encoder",
                "vision_arch",
                "vision_depth",
                "vision_hidden_size",
                "vision_output_size",
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
                "has_mtp_assistant",
                "mtp_num_layers",
                "mtp_use_dedicated_embeddings",
            )
        })
        mrope = list(rope_parameters.get("mrope_section") or [])
        if mrope:
            cfg["mrope_sections"] = [int(v) for v in mrope] + ([0] if len(mrope) == 3 else [])
            # Qwen3.5 may rotate only a fraction of each attention head. Keep
            # the validated architecture width instead of expanding M-RoPE
            # back to the full head dimension.
            cfg["mrope_n_dims"] = int(
                cfg.get("rotary_dim")
                or cfg.get("head_dim")
                or cfg.get("hidden_size", 0)
                // max(1, int(cfg.get("num_attention_heads", 1)))
                or 2 * sum(int(v) for v in mrope)
            )
    cfg = _inject_runtime_config_defaults(cfg, arch)
    if arch == "gemma4" and "layer_kinds" not in cfg:
        raise SystemExit("Gemma4 safetensors conversion currently requires --config-template with explicit layer_kinds/shared KV policy")
    return cfg


def _effective_ref_dtype_policy(
    ref: TensorRef,
    dtype_policy: str,
    linear_weight_dtype: str,
) -> str:
    if ref.role == "linear_weight" and linear_weight_dtype != "preserve":
        return linear_weight_dtype
    return ref.dtype or dtype_policy


def _entry_size_from_header(
    ref: TensorRef,
    headers: dict[str, HeaderTensor],
    dtype_policy: str,
    linear_weight_dtype: str = "preserve",
) -> tuple[str, int, list[int]]:
    if ref.synth:
        data, dt, shape = _synth_bytes(ref.synth, ref.shape or (), dtype_policy)
        return dt, len(data), shape
    if ref.transform and ref.shape is not None and ref.source_names:
        h = headers[ref.source_names[0]]
        effective_policy = _effective_ref_dtype_policy(
            ref, dtype_policy, linear_weight_dtype
        )
        if effective_policy != "preserve":
            out_dtype = effective_policy
            elem = {"fp32": 4, "bf16": 2, "fp16": 2}[out_dtype]
        else:
            out_dtype, elem = _header_dtype_to_ck(h.dtype)
        shape = [int(x) for x in ref.shape]
        size = int(np.prod(shape, dtype=np.int64)) * elem
        return out_dtype, size, shape
    total = 0
    effective_policy = _effective_ref_dtype_policy(
        ref, dtype_policy, linear_weight_dtype
    )
    out_dtype: str | None = None if effective_policy == "preserve" else effective_policy
    out_shape: list[int] = []
    for src in ref.source_names:
        if src not in headers:
            raise KeyError(src)
        h = headers[src]
        shape = h.shape
        if effective_policy != "preserve":
            dtype = effective_policy
            elem = {"fp32": 4, "bf16": 2, "fp16": 2}[dtype]
        else:
            dtype, elem = _header_dtype_to_ck(h.dtype)
        total += int(np.prod(shape, dtype=np.int64)) * elem
        out_dtype = out_dtype or dtype
        out_shape = shape if not out_shape else out_shape
    if ref.shape is not None:
        out_shape = [int(x) for x in ref.shape]
    return out_dtype or "fp32", total, out_shape


def _is_qwen35_shifted_norm_ref(ref: TensorRef) -> bool:
    if not ref.source_names:
        return False
    src = ref.source_names[0]
    if not src.endswith("norm.weight"):
        return False
    # llama.cpp shifts Qwen3.5 norm weights by +1, except the recurrent
    # linear_attn.norm.weight used inside the DeltaNet norm-gate.
    if src.endswith("linear_attn.norm.weight"):
        return False
    return ref.ck_name == "final_ln_weight" or ref.ck_name.endswith((
        ".attn_norm",
        ".post_attention_norm",
        ".attn_q_norm",
        ".attn_k_norm",
    ))


def _ref_transform(arch: str, ref: TensorRef) -> str | None:
    if ref.transform:
        return ref.transform
    if ref.ck_name.endswith(".ssm_a") or ref.ck_name.endswith(".mamba_a"):
        return "neg_exp_a_log"
    if arch == "qwen35" and _is_qwen35_shifted_norm_ref(ref):
        return "qwen35_norm_plus_one"
    return None


def _ignored_source_tensor(arch: str, name: str) -> str | None:
    if name.startswith("mtp.") or name.startswith("model.mtp."):
        return "mtp_decoder_block_not_in_main_pass"
    if arch == "gemma4_assistant" and name.startswith("masked_embedding."):
        return "ordered_embedding_sidecar_not_in_current_runtime"
    if arch == "qwen35" and (name.startswith("model.visual.") or name.startswith("visual.")):
        return "vision_tower_not_in_decoder_pass"
    if arch == "qwen35" and name.startswith("model.vision_model."):
        return "vision_tower_not_in_decoder_pass"
    if arch == "qwen3vl" and (name.startswith("model.visual.") or name.startswith("visual.")):
        return "vision_tower_not_in_decoder_pass"
    if arch == "qwen3_vl_vision" and (name.startswith("model.language_model.") or name.startswith("model.model.") or name == "lm_head.weight"):
        return "language_model_not_in_vision_pass"
    contract_reason = _contract_ignored_source_tensor(
        _safetensors_arch_contract(arch), name
    )
    if contract_reason is not None:
        return contract_reason
    if arch == "kimi_vl" and (
        name.startswith("vision_tower.")
        or name.startswith("multi_modal_projector.")
    ):
        return "vision_tower_not_in_decoder_pass"
    if (
        arch == "kimi_vl"
        and name.startswith("language_model.model.layers.")
        and name.endswith(".self_attn.rotary_emb.inv_freq")
    ):
        return "rope_frequencies_derived_from_runtime_contract"
    return None


def _build_source_audit(arch: str, headers: dict[str, HeaderTensor], refs: list[TensorRef]) -> dict[str, Any]:
    consumed: dict[str, list[str]] = {}
    synthetic: list[str] = []
    transforms: list[dict[str, str]] = []
    for ref in refs:
        transform = _ref_transform(arch, ref)
        if ref.source_names:
            for src in ref.source_names:
                consumed.setdefault(src, []).append(ref.ck_name)
                if transform:
                    transforms.append({"source": src, "target": ref.ck_name, "transform": transform})
        else:
            synthetic.append(ref.ck_name)
    ignored: list[dict[str, str]] = []
    unmapped: list[str] = []
    for name in sorted(headers):
        if name in consumed:
            continue
        reason = _ignored_source_tensor(arch, name)
        if reason:
            ignored.append({"source": name, "reason": reason})
        else:
            unmapped.append(name)
    return {
        "arch": arch,
        "source_tensor_count": len(headers),
        "consumed_source_count": len(consumed),
        "consumed_source_tensors": sorted(consumed),
        "source_to_targets": {name: sorted(targets) for name, targets in sorted(consumed.items())},
        "entry_count": len(refs),
        "synthetic_entries": synthetic,
        "ignored_source_tensors": ignored,
        "unmapped_source_tensors": unmapped,
        "transforms": transforms,
        "verdict": "fail" if unmapped else "pass",
    }


def _write_ref(
    w: HashingWriter,
    model_dir: Path,
    headers: dict[str, HeaderTensor],
    ref: TensorRef,
    dtype_policy: str,
    arch: str,
    linear_weight_dtype: str = "preserve",
) -> tuple[str, int, list[int]]:
    if ref.synth:
        data, dt, shape = _synth_bytes(ref.synth, ref.shape or (), dtype_policy)
        w.write(data)
        return dt, len(data), shape
    total = 0
    effective_policy = _effective_ref_dtype_policy(
        ref, dtype_policy, linear_weight_dtype
    )
    out_dtype: str | None = None if effective_policy == "preserve" else effective_policy
    out_shape: list[int] = []
    for src in ref.source_names:
        t = _load_tensor(model_dir, headers, src)
        transform = _ref_transform(arch, ref)
        if transform == "neg_exp_a_log":
            import torch
            t = -torch.exp(t.to(dtype=torch.float32))
        elif transform == "qwen35_norm_plus_one":
            import torch
            t = t.to(dtype=torch.float32) + 1.0
        elif transform in {"qwen3vl_patch_temporal_0", "qwen3vl_patch_temporal_1"}:
            idx = 0 if transform.endswith("_0") else 1
            if len(t.shape) != 5:
                raise SystemExit(f"{src}: expected Qwen3-VL patch tensor [out,in,t,h,w], got {tuple(t.shape)}")
            if int(t.shape[2]) <= idx:
                raise SystemExit(f"{src}: temporal dimension {int(t.shape[2])} too small for {transform}")
            t = t[:, :, idx, :, :].contiguous().reshape(int(t.shape[0]), -1)
        data, dt, shape = _torch_to_bytes(t, effective_policy)
        w.write(data)
        total += len(data)
        out_dtype = out_dtype or dt
        out_shape = shape if not out_shape else out_shape
    if ref.shape is not None:
        out_shape = [int(x) for x in ref.shape]
    return out_dtype or "fp32", total, out_shape


def _tokenizer_payloads_from_json(model_dir: Path, vocab_size: int) -> tuple[list[tuple[str, str, bytes, list[int], str]], dict[str, Any] | None, dict[str, Any]]:
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.exists() or vocab_size <= 0:
        tiktoken_path = model_dir / "tiktoken.model"
        if not tiktoken_path.exists() or vocab_size <= 0:
            return [], None, {}
        return (
            [],
            {
                "tokenizer_type": "tiktoken",
                "source": "tiktoken_model_sidecar",
                "path": str(tiktoken_path),
            },
            _special_tokens_from_tokenizer_config(model_dir, "tiktoken"),
        )
    info = inspect_tokenizer_json(str(tok_path))
    tok_type = str(info.get("model_type") or "").strip().lower()
    if tok_type not in {"bpe", "wordpiece"}:
        return [], None, {}
    offsets, strings_blob, merges, _scores, _types = load_tokenizer_json(str(tok_path), int(vocab_size))
    offsets_bytes = struct.pack(f"<{len(offsets)}i", *offsets)
    merges_bytes = struct.pack(f"<{len(merges)}i", *merges) if merges else b""
    payloads = [
        ("vocab_offsets", "i32", offsets_bytes, [len(offsets)], "tokenizer_json"),
        ("vocab_strings", "u8", strings_blob, [len(strings_blob)], "tokenizer_json"),
        ("vocab_merges", "i32", merges_bytes, [len(merges)], "tokenizer_json"),
    ]
    contract: dict[str, Any] = {
        "tokenizer_type": tok_type,
        "source": "tokenizer_json",
        "path": str(tok_path),
    }
    special = _special_tokens_from_tokenizer_config(model_dir, tok_type)
    return payloads, contract, special


def _special_tokens_from_tokenizer_config(model_dir: Path, tokenizer_type: str) -> dict[str, Any]:
    out: dict[str, Any] = {"tokenizer_model": tokenizer_type}
    cfg_path = model_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        return out
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return out

    def _content(value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("content"), str):
            return str(value.get("content"))
        return None

    added = cfg.get("added_tokens_decoder") if isinstance(cfg.get("added_tokens_decoder"), dict) else {}
    token_to_id: dict[str, int] = {}
    for key, row in added.items():
        if not isinstance(row, dict):
            continue
        content = row.get("content")
        if not isinstance(content, str):
            continue
        try:
            token_to_id[content] = int(key)
        except Exception:
            continue

    for name, field in (("bos_token", "bos_token"), ("eos_token", "eos_token"), ("unk_token", "unk_token"), ("pad_token", "pad_token")):
        tok = _content(cfg.get(field))
        if tok is not None:
            out[name] = tok
            if tok in token_to_id:
                out[f"{name}_id"] = token_to_id[tok]
    for key, out_key in (("add_bos_token", "add_bos_token"), ("add_eos_token", "add_eos_token"), ("add_prefix_space", "add_space_prefix")):
        if key in cfg:
            out[out_key] = bool(cfg.get(key))
    return out


def _copy_tokenizer_sidecars(model_dir: Path, out_dir: Path) -> None:
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "tiktoken.model",
        "tokenizer.model",
    ):
        src = model_dir / name
        if src.exists():
            dst = out_dir / name
            if src.resolve() != dst.resolve():
                dst.write_bytes(src.read_bytes())
    for pattern in ("tokenization_*.py", "configuration_*.py"):
        for src in model_dir.glob(pattern):
            dst = out_dir / src.name
            if src.resolve() != dst.resolve():
                dst.write_bytes(src.read_bytes())

    if (model_dir / "tiktoken.model").exists():
        source_dir = out_dir / "tokenizer_source"
        source_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "tiktoken.model",
            "tokenizer.model",
        ):
            src = model_dir / name
            if src.exists():
                (source_dir / name).write_bytes(src.read_bytes())
        for pattern in ("tokenization_*.py", "configuration_*.py"):
            for src in model_dir.glob(pattern):
                (source_dir / src.name).write_bytes(src.read_bytes())


def _resolve_output_path(output: Path | None, *, ram_output: bool, ram_dir: Path, checkpoint: Path) -> Path:
    if not ram_output:
        if output is None:
            raise SystemExit("--output is required unless --ram-output is used")
        return output

    ram_root = ram_dir.resolve()
    if output is not None and output.is_absolute() and str(output.resolve()).startswith(str(ram_root)):
        return output

    name = checkpoint.resolve().name or "model"
    base = ram_root / "ck-engine-v8" / name
    if output is None:
        return base / "weights.bump"
    return base / output.name




def _is_proc_fd_path(path: Path) -> bool:
    text = str(path)
    return text.startswith("/proc/self/fd/") or (text.startswith("/proc/") and "/fd/" in text)

def _check_output_capacity(path: Path, estimated_bytes: int, *, ram_output: bool) -> None:
    if _is_proc_fd_path(path):
        return
    probe = path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    reserve = max(256 * 1024 * 1024, int(estimated_bytes * 0.02))
    required = int(estimated_bytes) + reserve
    if usage.free < required:
        kind = "RAM/tmpfs" if ram_output else "filesystem"
        raise SystemExit(
            f"Not enough free space in {kind} for BUMP output: need about "
            f"{required / 1024 / 1024 / 1024:.2f} GiB including reserve, "
            f"free {usage.free / 1024 / 1024 / 1024:.2f} GiB at {path.parent}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert HF safetensors language weights to v8 BUMPWGT5")
    ap.add_argument("--checkpoint", required=True, type=Path, help="HF safetensors model directory")
    ap.add_argument("--output", type=Path, help="output weights.bump")
    ap.add_argument("--ram-output", action="store_true", help="write weights.bump under a RAM-backed tmpfs path instead of the checkpoint/output directory")
    ap.add_argument("--ram-dir", type=Path, default=Path("/dev/shm"), help="tmpfs directory for --ram-output; default: /dev/shm")
    ap.add_argument("--config-out", required=True, type=Path)
    ap.add_argument("--manifest-out", required=True, type=Path)
    ap.add_argument("--arch", default="auto", choices=["auto", "gemma4", "gemma4_assistant", "gemma3", "llama", "qwen2", "qwen3", "qwen3vl", "qwen3_vl_vision", "cohere_compass_text", "cohere_compass_vision", "qwen35", "nemotron_h", "glm4", "kimi_vl", "instella_moe", "whisper_encoder", "whisper_decoder"])
    ap.add_argument("--config-template", type=Path, help="existing v8 config/manifest to reuse explicit runtime policy")
    ap.add_argument("--dtype", default="preserve", choices=["preserve", "bf16", "fp32"])
    ap.add_argument(
        "--vision-numerics",
        default="pytorch_exact",
        choices=["pytorch_exact", "native"],
        help=(
            "vision reduction profile: PyTorch oneDNN/MKL/AMX exact, or "
            "portable CKE native providers"
        ),
    )
    ap.add_argument(
        "--linear-weight-dtype",
        default="preserve",
        choices=["preserve", "fp16", "bf16", "fp32"],
        help=(
            "override only tensors declared with role=linear_weight in the "
            "safetensors model map"
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="validate mapping and write JSON reports only; do not write BUMP")
    ap.add_argument("--audit-out", type=Path, help="write safetensors source coverage audit JSON; default is manifest directory/conversion_audit.json")
    ap.add_argument("--allow-unmapped", action="store_true", help="allow non-ignored source tensors to remain unmapped")
    args = ap.parse_args()

    model_dir = args.checkpoint.resolve()
    args.output = _resolve_output_path(args.output, ram_output=bool(args.ram_output), ram_dir=args.ram_dir, checkpoint=model_dir)
    headers = _load_safetensors_headers(model_dir)
    hf = _hf_config(model_dir)
    arch = args.arch
    if arch == "auto":
        arch = _infer_arch(hf)

    config = _build_config(model_dir, arch, args.config_template)
    _apply_linear_weight_runtime_config(
        config, _safetensors_arch_contract(arch), args.linear_weight_dtype
    )
    _apply_vision_numerics_profile(config, arch, args.vision_numerics)
    refs = _refs_for_arch(arch, config, headers)
    if str(config.get("artifact_scope") or "") == "encoder_only":
        tokenizer_payloads, tokenizer_contract, special_tokens = [], None, {}
    else:
        tokenizer_payloads, tokenizer_contract, special_tokens = _tokenizer_payloads_from_json(
            model_dir, int(config.get("vocab_size") or 0)
        )
    missing: list[str] = []
    entries_preview: list[dict[str, Any]] = []
    dtype_table: list[int] = []
    offset = align_up(DATA_START + 4 + len(refs) + len(tokenizer_payloads))
    for ref in refs:
        for src in ref.source_names:
            if src not in headers:
                missing.append(src)
        if missing:
            continue
        offset = align_up(offset)
        dt, size, shape = _entry_size_from_header(
            ref, headers, args.dtype, args.linear_weight_dtype
        )
        dtype_table.append(_dtype_code(dt))
        preview_entry = {
            "name": ref.ck_name,
            "dtype": dt,
            "file_offset": offset,
            "size": size,
            "source_name": "+".join(ref.source_names) if ref.source_names else f"synthetic:{ref.synth}",
            "shape": shape,
        }
        if ref.role:
            preview_entry["role"] = ref.role
        transform = _ref_transform(arch, ref)
        if transform:
            preview_entry["transform"] = transform
        entries_preview.append(preview_entry)
        offset += size
    if missing:
        uniq = sorted(set(missing))
        raise SystemExit("Missing required safetensors tensors:\n  " + "\n  ".join(uniq[:80]))

    audit = _build_source_audit(arch, headers, refs)
    audit_out = args.audit_out or (args.manifest_out.parent / "conversion_audit.json")
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["unmapped_source_tensors"] and not args.allow_unmapped:
        sample = "\n  ".join(audit["unmapped_source_tensors"][:80])
        raise SystemExit(f"Unmapped safetensors source tensors; see {audit_out}:\n  {sample}")

    decoder_projection_suffixes = (".wq", ".wk", ".wv", ".wo", ".w1", ".w2", ".w3")
    decoder_projection_entries = [
        entry for entry in entries_preview
        if str(entry.get("name") or "").startswith("layer.")
        and str(entry.get("name") or "").endswith(decoder_projection_suffixes)
    ]
    if (
        arch == "qwen3vl"
        and decoder_projection_entries
        and all(entry.get("dtype") == "bf16" for entry in decoder_projection_entries)
    ):
        # The AMX projection provider is validated for batched prefill. Decode
        # remains on the generic BF16 provider because its reduction is a
        # separate numerical contract and can alter long-generation ranking.
        config.setdefault("decoder_prefill_projection_storage_boundary", "bf16")

    template = apply_model_contract_overrides(
        _load_safetensors_template_for_arch(arch),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", True)),
        has_untied_output_weight=any(e["name"] == "output.weight" for e in entries_preview),
    )
    if tokenizer_contract:
        template = _apply_tokenizer_contract_overrides(
            template,
            str(tokenizer_contract.get("tokenizer_type") or "").strip().lower(),
        )
        config["tokenizer_contract"] = tokenizer_contract
        config["special_tokens"] = special_tokens
    quant_summary: dict[str, Any] = {
        "source": "safetensors",
        "dtype_policy": args.dtype,
        "linear_weight_dtype_policy": args.linear_weight_dtype,
    }
    for name, dt, payload, shape, source in tokenizer_payloads:
        offset = align_up(offset)
        dtype_table.append(CK_DT_FP32)
        entries_preview.append({
            "name": name,
            "dtype": dt,
            "file_offset": offset,
            "size": len(payload),
            "source_name": source,
            "shape": shape,
        })
        offset += len(payload)

    for e in entries_preview:
        if e["name"].startswith("layer."):
            layer_key = ".".join(e["name"].split(".")[:2])
            quant_summary.setdefault(layer_key, {})[e["name"].split(".")[-1]] = e["dtype"]
        else:
            quant_summary[e["name"]] = e["dtype"]

    def _int_or_zero(value: Any) -> int:
        return int(value or 0)

    manifest = {
        "version": 5,
        "model": arch,
        "source_format": "safetensors",
        "bump_layout": {
            "header_size": HEADER_SIZE,
            "ext_metadata_size": EXT_METADATA_SIZE,
            "data_start": DATA_START,
        },
        "config": config,
        "template": template,
        "quant_summary": quant_summary,
        "num_layers": _int_or_zero(config.get("num_layers")),
        "embed_dim": _int_or_zero(config.get("embed_dim")),
        "num_heads": _int_or_zero(config.get("num_heads")),
        "num_kv_heads": _int_or_zero(config.get("num_kv_heads")),
        "head_dim": _int_or_zero(config.get("head_dim")),
        "intermediate_size": _int_or_zero(config.get("intermediate_size")),
        "vocab_size": _int_or_zero(config.get("vocab_size")),
        "context_length": _int_or_zero(config.get("context_length")),
        "has_attention_biases": bool(config.get("has_attention_biases", False)),
        "has_qk_norm": bool(config.get("has_qk_norm", False)),
        "source_audit": audit,
        "special_tokens": special_tokens,
        "tokenizer_contract": tokenizer_contract if tokenizer_contract else None,
        "entries": entries_preview,
    }

    print(f"[safetensors->bump] arch={arch} tensors={len(refs)} entries={len(entries_preview)} dry_run={args.dry_run}")
    estimated_bytes = int(offset - DATA_START)
    print(f"[safetensors->bump] output={args.output}")
    if args.ram_output:
        print(f"[safetensors->bump] ram_output=True ram_dir={args.ram_dir}")
    print(f"[safetensors->bump] estimated weights payload={(offset - (DATA_START + 4 + len(refs))) / 1024 / 1024:.1f} MiB")

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        return 0

    _check_output_capacity(args.output, estimated_bytes, ram_output=bool(args.ram_output))
    if not _is_proc_fd_path(args.output):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _copy_tokenizer_sidecars(model_dir, args.output.parent)
    with args.output.open("w+b") as f:
        f.write(b"\x00" * HEADER_SIZE)
        f.write(b"\x00" * EXT_METADATA_SIZE)
        w = HashingWriter(f)
        w.write(struct.pack("<I", len(dtype_table)))
        w.write(bytes(dtype_table))
        entries: list[dict[str, Any]] = []
        current_offset = DATA_START + 4 + len(dtype_table)
        for ref in refs:
            aligned_offset = align_up(current_offset)
            if aligned_offset > current_offset:
                w.write(b"\x00" * (aligned_offset - current_offset))
                current_offset = aligned_offset
            start = current_offset
            dt, size, shape = _write_ref(
                w,
                model_dir,
                headers,
                ref,
                args.dtype,
                arch,
                args.linear_weight_dtype,
            )
            current_offset += size
            entry = {
                "name": ref.ck_name,
                "dtype": dt,
                "file_offset": start,
                "size": size,
                "source_name": "+".join(ref.source_names) if ref.source_names else f"synthetic:{ref.synth}",
                "shape": shape,
            }
            if ref.role:
                entry["role"] = ref.role
            transform = _ref_transform(arch, ref)
            if transform:
                entry["transform"] = transform
            entries.append(entry)
        for name, dt, payload, shape, source in tokenizer_payloads:
            aligned_offset = align_up(current_offset)
            if aligned_offset > current_offset:
                w.write(b"\x00" * (aligned_offset - current_offset))
                current_offset = aligned_offset
            start = current_offset
            w.write(payload)
            current_offset += len(payload)
            entries.append({
                "name": name,
                "dtype": dt,
                "file_offset": start,
                "size": len(payload),
                "source_name": source,
                "shape": shape,
            })
        checksum = w.digest()
        manifest["entries"] = entries
        args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_hash = calculate_manifest_hash(manifest)
        metadata = build_bumpv5_metadata(template, config, quant_summary, manifest_hash, "convert_safetensors_to_bump_v8.py")
        metadata["template_hash"] = calculate_template_hash(template)
        metadata_bytes = _canonical_json_bytes(metadata)
        meta_hash = calculate_metadata_hash(metadata)

        f.flush()
        f.seek(0)
        f.write(b"BUMPWGT5")
        f.write(struct.pack("<I", BUMP_VERSION_V5))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", _int_or_zero(config.get("num_layers"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("vocab_size"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("embed_dim"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("intermediate_size"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("context_length"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("num_heads"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("num_kv_heads"))))
        f.write(struct.pack("<I", _int_or_zero(config.get("head_dim"))))
        for value in (
            _int_or_zero(config.get("embed_dim")),
            _int_or_zero(config.get("head_dim")),
            _int_or_zero(config.get("intermediate_size")),
            _int_or_zero(config.get("context_length")),
        ):
            f.write(struct.pack("<Q", value))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(checksum)
        f.seek(0, os.SEEK_END)
        f.write(metadata_bytes)
        write_bumpv5_footer(f, len(metadata_bytes), meta_hash)
    print(f"[safetensors->bump] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
