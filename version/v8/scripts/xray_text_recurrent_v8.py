#!/usr/bin/env python3
from __future__ import annotations

"""Schedule-preserving X-ray attribution for recurrent text decoders."""

import argparse
import hashlib
import json
import multiprocessing
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from compare_first_token_logits_v8 import load_ck_logits_segmented
from decoder_first_token_parity_v8 import _run_llama_capture


V8_ROOT = Path(__file__).resolve().parents[1]
KERNEL_MAPS_DIR = V8_ROOT / "kernel_maps"
RUNTIME_STAMP_NAME = ".ck_runtime_bundle.json"
RUNTIME_BUNDLE_SCHEMA = "ck-v8-runtime-bundle-v2"
RUNTIME_OUTPUT_NAMES = (
    "libmodel.so",
    "libckernel_engine.so",
    "libckernel_tokenizer.so",
)
SUPPORTED_RECURRENT_STATE_LAYOUTS = {
    "head_key_value_contiguous",
    "head_value_key_contiguous",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime_provenance(model_dir: Path) -> dict[str, Any]:
    """Fail closed when X-Ray's runtime binaries are not one stamped bundle."""
    model_dir = model_dir.resolve()
    stamp_path = model_dir / RUNTIME_STAMP_NAME
    if not stamp_path.is_file():
        raise RuntimeError(
            "X-Ray runtime provenance check failed: missing "
            f"{stamp_path}; rebuild through ck_run_v8.py before capture"
        )
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"X-Ray runtime provenance check failed: unreadable {stamp_path}: {exc}"
        ) from exc

    inputs = stamp.get("inputs")
    outputs = stamp.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise RuntimeError(
            "X-Ray runtime provenance check failed: runtime stamp must contain "
            f"object-valued inputs and outputs: {stamp_path}"
        )
    schema = inputs.get("schema")
    if schema != RUNTIME_BUNDLE_SCHEMA:
        raise RuntimeError(
            "X-Ray runtime provenance check failed: unsupported runtime stamp "
            f"schema {schema!r}; expected {RUNTIME_BUNDLE_SCHEMA!r}: {stamp_path}"
        )

    accepted: dict[str, dict[str, Any]] = {}
    for name in RUNTIME_OUTPUT_NAMES:
        expected = outputs.get(name)
        if not isinstance(expected, dict):
            raise RuntimeError(
                "X-Ray runtime provenance check failed: stamp has no identity for "
                f"{name}: {stamp_path}"
            )
        path = model_dir / name
        if not path.is_file():
            raise RuntimeError(
                f"X-Ray runtime provenance check failed: missing runtime binary {path}"
            )
        actual_size = path.stat().st_size
        actual_sha256 = _sha256_file(path)
        expected_size = expected.get("size")
        expected_sha256 = expected.get("sha256")
        if actual_size != expected_size or actual_sha256 != expected_sha256:
            raise RuntimeError(
                "X-Ray runtime provenance check failed: runtime binary does not "
                f"match its stamp: {path} "
                f"(stamped size={expected_size}, sha256={expected_sha256}; "
                f"actual size={actual_size}, sha256={actual_sha256}); rebuild "
                "through ck_run_v8.py before capture"
            )
        accepted[name] = {
            "path": str(path),
            "size": actual_size,
            "sha256": actual_sha256,
        }

    linked = inputs.get("linked_libraries")
    if not isinstance(linked, dict):
        raise RuntimeError(
            "X-Ray runtime provenance check failed: stamp has no linked_libraries "
            f"contract: {stamp_path}"
        )
    for link_name, output_name in (
        ("engine", "libckernel_engine.so"),
        ("tokenizer", "libckernel_tokenizer.so"),
    ):
        linked_identity = linked.get(link_name)
        linked_sha256 = (
            linked_identity.get("sha256")
            if isinstance(linked_identity, dict)
            else None
        )
        runtime_sha256 = accepted[output_name]["sha256"]
        if linked_sha256 != runtime_sha256:
            raise RuntimeError(
                "X-Ray runtime provenance check failed: generated model was linked "
                f"against {link_name} sha256={linked_sha256}, but the runtime "
                f"contains {output_name} sha256={runtime_sha256}: {stamp_path}"
            )

    return {
        "status": "verified",
        "schema": schema,
        "stamp": str(stamp_path),
        "outputs": accepted,
    }


BOUNDARIES = (
    "attn_norm",
    "attn_hyper_norm",
    "attn_hyper_gate",
    "attn_mixed_input",
    "linear_attn_qkv_mixed",
    "z",
    "conv_output_raw",
    "conv_output_silu",
    "q_conv_predelta",
    "k_conv_predelta",
    "alpha",
    "gate",
    "beta",
    "new_state",
    "attn_output",
    "final_output",
    "linear_attn_out",
    "q_proj",
    "k_proj",
    "v_proj",
    "qk_norm_q",
    "qk_norm_k",
    "rope_q",
    "rope_k",
    "attn_gate",
    "attn_pregate",
    "attn_out",
    "out_proj",
    "after_attn",
    "post_attn_norm",
    "mlp_gate",
    "mlp_up",
    "mlp_swiglu",
    "mlp_down",
    "after_attn_hyper",
    "mlp_hyper_norm",
    "mlp_hyper_gate",
    "mlp_mixed_input",
    "moe_router_logits",
    "moe_routing_weights",
    "moe_routed_output",
    "moe_combined_output",
    "layer_out_hyper",
    "layer_out",
)

RECURRENT_BOUNDARIES = (
    "attn_norm",
    "linear_attn_qkv_mixed",
    "z",
    "conv_output_raw",
    "conv_output_silu",
    "q_conv_predelta",
    "k_conv_predelta",
    "alpha",
    "gate",
    "beta",
    "new_state",
    "attn_output",
    "final_output",
    "linear_attn_out",
    "after_attn",
    "post_attn_norm",
    "mlp_gate",
    "mlp_up",
    "mlp_swiglu",
    "mlp_down",
    "layer_out",
)

HYPER_MOE_RECURRENT_BOUNDARIES = (
    "attn_hyper_norm",
    "attn_hyper_gate",
    "attn_mixed_input",
    "linear_attn_qkv_mixed",
    "z",
    "conv_output_raw",
    "conv_output_silu",
    "q_conv_predelta",
    "k_conv_predelta",
    "alpha",
    "gate",
    "beta",
    "new_state",
    "attn_output",
    "final_output",
    "linear_attn_out",
    "after_attn_hyper",
    "mlp_hyper_norm",
    "mlp_hyper_gate",
    "mlp_mixed_input",
    "moe_router_logits",
    "moe_routing_weights",
    "moe_routed_output",
    "moe_combined_output",
    "layer_out_hyper",
)

PLE_HYPER_MOE_RECURRENT_BOUNDARIES = (
    "ple_key_projected",
    "ple_value_projected",
    "ple_key_normed",
    "ple_query_normed",
    "ple_gated_value",
    "ple_conv_normed",
    "ple_conv_out",
    "ple_layer_out",
    *HYPER_MOE_RECURRENT_BOUNDARIES,
)

FULL_ATTENTION_BOUNDARIES = (
    "attn_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "qk_norm_q",
    "qk_norm_k",
    "rope_q",
    "rope_k",
    "attn_gate",
    "attn_pregate",
    "attn_out",
    "out_proj",
    "after_attn",
    "post_attn_norm",
    "mlp_gate",
    "mlp_up",
    "mlp_swiglu",
    "mlp_down",
    "layer_out",
)

HYPER_MOE_ATTENTION_BOUNDARIES = (
    "attn_hyper_norm",
    "attn_hyper_gate",
    "attn_mixed_input",
    "q_proj",
    "k_proj",
    "v_proj",
    "qk_norm_q",
    "qk_norm_k",
    "rope_q",
    "rope_k",
    "attn_gate",
    "attn_pregate",
    "attn_out",
    "out_proj",
    "after_attn_hyper",
    "mlp_hyper_norm",
    "mlp_hyper_gate",
    "mlp_mixed_input",
    "moe_router_logits",
    "moe_routing_weights",
    "moe_routed_output",
    "moe_combined_output",
    "layer_out_hyper",
)

# CKE checkpoint labels describe circuit edges; llama.cpp labels describe graph
# nodes. Keep that vocabulary translation explicit instead of teaching either
# backend to guess the other backend's names.
ORACLE_BOUNDARY_NAMES = {
    "attn_hyper_norm": "hc_norm",
    "attn_hyper_gate": "hc_gate",
    "attn_mixed_input": "hc_mixed",
    "after_attn_hyper": "hc_combine",
    "mlp_hyper_norm": "hc_norm",
    "mlp_hyper_gate": "hc_gate",
    "mlp_mixed_input": "hc_mixed",
    "moe_router_logits": "ffn_moe_logits",
    "moe_routing_weights": "ffn_moe_weights_norm",
    "moe_routed_output": "ffn_moe_out",
    "moe_combined_output": "ffn_out",
    "layer_out_hyper": "l_last",
    "q_proj": "Qcur_full",
    "k_proj": "Kcur",
    "v_proj": "Vcur",
    "qk_norm_q": "Qcur_normed",
    "qk_norm_k": "Kcur_normed",
    "rope_q": "Qcur",
    "rope_k": "Kcur",
    "attn_gate": "gate_reshaped",
    "attn_pregate": "attn_pregate",
    "attn_out": "attn_gated",
    "out_proj": "attn_output",
    "after_attn": "attn_residual",
    "post_attn_norm": "attn_post_norm",
    "mlp_gate": "ffn_gate",
    "mlp_up": "ffn_up",
    "mlp_swiglu": "ffn_swiglu",
    "mlp_down": "ffn_out",
    "layer_out": "l_out",
}

ORACLE_BOUNDARY_OCCURRENCES = {
    # Qwen3.5 emits Kcur once after projection and again after RoPE.
    "rope_k": 1,
    "mlp_hyper_norm": 1,
    "mlp_hyper_gate": 1,
    "mlp_mixed_input": 1,
}

CK_BOUNDARY_NAMES = {
    "layer_out_hyper": "layer_out",
}


def ck_capture_names(boundaries: Sequence[str]) -> tuple[str, ...]:
    names = [CK_BOUNDARY_NAMES.get(name, name) for name in boundaries]
    if "mlp_gate" in names or "mlp_up" in names:
        names.append("mlp_gate_up")
    return tuple(dict.fromkeys(names))


def boundaries_for_layer(config: dict[str, Any], layer: int) -> tuple[str, ...]:
    layer_kinds = config.get("layer_kinds")
    if not isinstance(layer_kinds, list) or not layer_kinds:
        raise ValueError("model config must declare non-empty layer_kinds")
    if layer < 0 or layer >= len(layer_kinds):
        raise ValueError(
            f"X-ray layer {layer} is outside configured layer_kinds extent "
            f"{len(layer_kinds)}"
        )
    kind = str(layer_kinds[layer])
    if kind == "recurrent":
        if int(config.get("hc_count", 1)) > 1 and int(
            config.get("num_experts", 0)
        ) > 0:
            ple_owner_layers = config.get("ple_owner_layers", [])
            if isinstance(ple_owner_layers, list) and layer in {
                int(owner) for owner in ple_owner_layers
            }:
                return PLE_HYPER_MOE_RECURRENT_BOUNDARIES
            return HYPER_MOE_RECURRENT_BOUNDARIES
        return RECURRENT_BOUNDARIES
    if kind == "full_attention":
        return FULL_ATTENTION_BOUNDARIES
    if kind == "sparse_attention":
        if int(config.get("hc_count", 1)) > 1 and int(
            config.get("num_experts", 0)
        ) > 0:
            return HYPER_MOE_ATTENTION_BOUNDARIES
        return FULL_ATTENTION_BOUNDARIES
    raise ValueError(f"unsupported X-ray layer kind at layer {layer}: {kind}")


def recurrent_state_layout_from_selected_provider(
    model_dir: Path,
    config: dict[str, Any],
    layer: int,
    *,
    kernel_maps_dir: Path = KERNEL_MAPS_DIR,
) -> tuple[str, dict[str, str]]:
    call_path = model_dir / "lowered_decode_call.json"
    if not call_path.is_file():
        layout = str(
            config.get(
                "recurrent_state_physical_layout",
                "head_key_value_contiguous",
            )
        )
        if layout not in SUPPORTED_RECURRENT_STATE_LAYOUTS:
            raise ValueError(
                "unsupported recurrent_state_physical_layout: "
                f"{layout!r}"
            )
        return layout, {"source": "config"}

    lowered_call = json.loads(call_path.read_text(encoding="utf-8"))
    operations = lowered_call.get("operations")
    if not isinstance(operations, list):
        raise ValueError(f"call-ready IR has no operations list: {call_path}")
    candidates = [
        operation
        for operation in operations
        if operation.get("layer") == layer
        and operation.get("op") == "recurrent_core"
    ]
    if len(candidates) != 1:
        raise ValueError(
            "recurrent X-Ray requires exactly one selected recurrent_core "
            f"provider for layer {layer}, found {len(candidates)}"
        )

    selected = candidates[0]
    call_abi = selected.get("call_abi")
    if not isinstance(call_abi, dict) or call_abi.get("owner") != "kernel_map":
        raise ValueError(
            "selected recurrent_core provider must have a kernel-map-owned call ABI"
        )
    kernel_id = str(call_abi.get("kernel_id", ""))
    source_file = str(call_abi.get("source_file", ""))
    if not kernel_id or not source_file or Path(source_file).name != source_file:
        raise ValueError("selected recurrent_core provider has invalid kernel-map identity")
    map_path = kernel_maps_dir / source_file
    if not map_path.is_file():
        raise ValueError(f"selected recurrent_core kernel map is missing: {map_path}")

    kernel_map = json.loads(map_path.read_text(encoding="utf-8"))
    if kernel_map.get("id") != kernel_id:
        raise ValueError(
            "selected recurrent_core kernel-map identity mismatch: "
            f"call ABI {kernel_id!r}, map {kernel_map.get('id')!r}"
        )
    if kernel_map.get("op") != "gated_deltanet":
        raise ValueError(
            "selected recurrent_core provider has incompatible operation class: "
            f"{kernel_map.get('op')!r}"
        )
    function = str(selected.get("function", ""))
    if kernel_map.get("impl", {}).get("function") != function:
        raise ValueError(
            "selected recurrent_core implementation mismatch: "
            f"call IR {function!r}, map {kernel_map.get('impl', {}).get('function')!r}"
        )

    state_in = next(
        (item for item in kernel_map.get("inputs", []) if item.get("name") == "state_in"),
        None,
    )
    state_out = next(
        (item for item in kernel_map.get("outputs", []) if item.get("name") == "state_out"),
        None,
    )
    input_layout = state_in.get("layout") if isinstance(state_in, dict) else None
    output_layout = state_out.get("layout") if isinstance(state_out, dict) else None
    if input_layout != output_layout or output_layout not in SUPPORTED_RECURRENT_STATE_LAYOUTS:
        raise ValueError(
            "selected recurrent_core provider must declare matching supported "
            f"state layouts, got input={input_layout!r}, output={output_layout!r}"
        )
    return str(output_layout), {
        "source": "selected_kernel_map",
        "kernel_id": kernel_id,
        "function": function,
        "source_file": source_file,
    }


@contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def canonicalize_named_axes(
    name: str,
    ck: np.ndarray,
    oracle: np.ndarray,
    state_size: int,
    recurrent_state_physical_layout: str = "head_key_value_contiguous",
) -> tuple[np.ndarray, np.ndarray, str]:
    if name == "new_state":
        state_elements = state_size * state_size
        if state_size <= 0 or ck.size != oracle.size or ck.size % state_elements != 0:
            raise ValueError("new_state requires matching [head, key, value] extents")
        if recurrent_state_physical_layout == "head_value_key_contiguous":
            return ck.reshape(-1), oracle.reshape(-1), "identity:[head,value,key]"
        if recurrent_state_physical_layout != "head_key_value_contiguous":
            raise ValueError(
                "unsupported recurrent_state_physical_layout: "
                f"{recurrent_state_physical_layout!r}"
            )
        heads = ck.size // state_elements
        ck = ck.reshape(heads, state_size, state_size).transpose(0, 2, 1).reshape(-1)
        return ck, oracle.reshape(-1), "ck:[head,key,value]->[head,value,key]"
    return ck.reshape(-1), oracle.reshape(-1), "identity"


def compare_arrays(
    name: str,
    ck: np.ndarray,
    oracle: np.ndarray,
    state_size: int = 128,
    recurrent_state_physical_layout: str = "head_key_value_contiguous",
) -> dict[str, Any]:
    ck, oracle, transform = canonicalize_named_axes(
        name, ck, oracle, state_size, recurrent_state_physical_layout
    )
    if ck.shape != oracle.shape:
        return {
            "status": "shape_mismatch",
            "ck_shape": list(ck.shape),
            "oracle_shape": list(oracle.shape),
            "axis_transform": transform,
        }
    delta = ck.astype(np.float64) - oracle.astype(np.float64)
    abs_delta = np.abs(delta)
    return {
        "status": "exact" if np.array_equal(ck, oracle) else "different",
        "elements": int(ck.size),
        "different_elements": int(np.count_nonzero(abs_delta)),
        "max_abs_diff": float(abs_delta.max(initial=0.0)),
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "axis_transform": transform,
    }


def classify(
    rows: list[dict[str, Any]],
    schedules: dict[str, str],
    *,
    material_abs_floor: float = 1e-5,
) -> dict[str, Any] | None:
    previous_comparable: dict[tuple[int, int], dict[str, Any]] = {}
    previous_exact: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row.get("logical_token", -1)), int(row.get("layer", -1)))
        status = row.get("status")
        if status not in {"exact", "different"}:
            continue
        previous = previous_comparable.get(key)
        previous_comparable[key] = row
        last_exact = previous_exact.get(key)
        if status == "exact":
            previous_exact[key] = row
        if row.get("status") != "different" or float(row.get("max_abs_diff", 0.0)) < material_abs_floor:
            continue
        classification = "VALUE_MISMATCH"
        if (
            row.get("boundary") == "linear_attn_qkv_mixed"
            and previous
            and previous.get("boundary") in {"attn_norm", "attn_mixed_input"}
            and previous.get("status") == "exact"
        ):
            classification = "PROJECTION_PROVIDER_MISMATCH"
        elif row.get("boundary") in {"alpha", "beta", "new_state"}:
            normalization = next((
                candidate for candidate in rows
                if candidate.get("logical_token") == row.get("logical_token")
                and candidate.get("layer") == row.get("layer")
                and candidate.get("boundary") == "attn_norm"
                and candidate.get("status") == "different"
                and float(candidate.get("max_abs_diff", 0.0)) < material_abs_floor
            ), None)
            if normalization is not None:
                classification = "NORMALIZATION_TO_QUANTIZATION_AMPLIFICATION"
        elif row.get("boundary") == "new_state":
            classification = "RECURRENT_STATE_REDUCTION_MISMATCH"
        return {
            "classification": classification,
            "logical_token": int(row["logical_token"]),
            "layer": int(row["layer"]),
            "boundary": str(row["boundary"]),
            "previous_exact_boundary": str(last_exact["boundary"]) if last_exact else None,
            "previous_comparable_boundary": str(previous["boundary"]) if previous else None,
            "amplification_source": "attn_norm" if classification == "NORMALIZATION_TO_QUANTIZATION_AMPLIFICATION" else None,
            "ck_schedule": schedules["ck"],
            "oracle_prefix_schedule": schedules["oracle_prefix"],
            "oracle_decode_schedule": schedules["oracle_decode"],
        }
    return None


def _load_oracle_row(
    root: Path,
    name: str,
    layer: int,
    logical_token: int,
    prompt_tokens: int,
    expected_count: int,
) -> np.ndarray:
    boundary = name
    name = ORACLE_BOUNDARY_NAMES.get(boundary, boundary)
    occurrence = ORACLE_BOUNDARY_OCCURRENCES.get(boundary, 0)
    physical_token = prompt_tokens - 1 if logical_token < prompt_tokens else logical_token
    path = root / f"{name}-{layer}-token-{physical_token:06d}-occ-{occurrence:03d}.bin"
    data = np.fromfile(path, dtype=np.float32)
    if logical_token < prompt_tokens and name != "new_state":
        if data.size != prompt_tokens * expected_count:
            raise ValueError(f"batched oracle extent mismatch for {name}: {data.size} != {prompt_tokens}*{expected_count}")
        return data.reshape(prompt_tokens, expected_count)[logical_token]
    if data.size != expected_count:
        raise ValueError(f"oracle extent mismatch for {name}: {data.size} != {expected_count}")
    return data


def _load_ck_row(
    root: Path,
    name: str,
    layer: int,
    logical_token: int,
    prompt_tokens: int,
    expected_count: int,
    ck_prefill_mode: str,
    attention_heads: int = 0,
    attention_kv_heads: int = 0,
) -> np.ndarray:
    physical_name = CK_BOUNDARY_NAMES.get(name, name)
    if ck_prefill_mode == "hybrid" and logical_token < prompt_tokens:
        token_path = root / (
            f"tok_{logical_token:04d}_layer_{layer:03d}_{physical_name}.f32"
        )
        if token_path.is_file():
            token_data = np.fromfile(token_path, dtype=np.float32)
            if name == "new_state":
                if logical_token != prompt_tokens - 1 or token_data.size != expected_count:
                    raise ValueError(
                        "batched CK recurrent state is available only after the final prompt row"
                    )
                return token_data
            if token_data.size == expected_count:
                return token_data
            if logical_token != 0:
                raise ValueError(
                    f"token-scoped batched CK extent mismatch for {name}: "
                    f"{token_data.size} != {expected_count}"
                )

        path = root / f"tok_{0:04d}_layer_{layer:03d}_{physical_name}.f32"
        if not path.is_file() and name in {"mlp_gate", "mlp_up"}:
            token_combined_path = root / (
                f"tok_{logical_token:04d}_layer_{layer:03d}_mlp_gate_up.f32"
            )
            combined_path = (
                token_combined_path
                if token_combined_path.is_file()
                else root / f"tok_{0:04d}_layer_{layer:03d}_mlp_gate_up.f32"
            )
            combined = np.fromfile(combined_path, dtype=np.float32)
            combined_width = 2 * expected_count
            if combined.size == combined_width:
                offset = 0 if name == "mlp_gate" else expected_count
                return combined[offset : offset + expected_count]
            if combined.size != prompt_tokens * combined_width:
                raise ValueError(
                    "batched CK fused gate/up extent mismatch: "
                    f"{combined.size} != {prompt_tokens}*{combined_width}"
                )
            rows = combined.reshape(prompt_tokens, combined_width)
            offset = 0 if name == "mlp_gate" else expected_count
            return rows[logical_token, offset : offset + expected_count]
        data = np.fromfile(path, dtype=np.float32)
        if name == "new_state":
            if logical_token != prompt_tokens - 1 or data.size != expected_count:
                raise ValueError("batched CK recurrent state is available only after the final prompt row")
            return data
        if data.size != prompt_tokens * expected_count:
            raise ValueError(
                f"batched CK extent mismatch for {name}: "
                f"{data.size} != {prompt_tokens}*{expected_count}"
            )
        head_major_heads = 0
        if name in {"qk_norm_q", "rope_q", "attn_pregate"}:
            head_major_heads = attention_heads
        elif name in {"qk_norm_k", "rope_k"}:
            head_major_heads = attention_kv_heads
        if head_major_heads > 0:
            if expected_count % head_major_heads != 0:
                raise ValueError("attention row width is not divisible by the declared head count")
            head_dim = expected_count // head_major_heads
            return data.reshape(head_major_heads, prompt_tokens, head_dim).transpose(1, 0, 2)[logical_token].reshape(-1)
        return data.reshape(prompt_tokens, expected_count)[logical_token]
    path = root / (
        f"tok_{logical_token:04d}_layer_{layer:03d}_{physical_name}.f32"
    )
    data = np.fromfile(path, dtype=np.float32)
    if data.size != expected_count:
        raise ValueError(f"CK extent mismatch for {name}: {data.size} != {expected_count}")
    return data


def _infer_oracle_row_count(
    root: Path,
    name: str,
    layer: int,
    logical_token: int,
    prompt_tokens: int,
) -> int:
    boundary = name
    name = ORACLE_BOUNDARY_NAMES.get(boundary, boundary)
    occurrence = ORACLE_BOUNDARY_OCCURRENCES.get(boundary, 0)
    physical_token = prompt_tokens - 1 if logical_token < prompt_tokens else logical_token
    path = root / f"{name}-{layer}-token-{physical_token:06d}-occ-{occurrence:03d}.bin"
    elements = path.stat().st_size // np.dtype(np.float32).itemsize
    if logical_token < prompt_tokens and name != "new_state":
        if elements % prompt_tokens != 0:
            raise ValueError(f"cannot infer oracle row width for {name}: {elements} is not divisible by {prompt_tokens}")
        return int(elements // prompt_tokens)
    return int(elements)


def analyze_capture(
    ck_root: Path,
    oracle_root: Path,
    prompt_tokens: int,
    total_tokens: int,
    layer: int,
    state_size: int = 128,
    ck_prefill_mode: str = "sequential",
    attention_heads: int = 0,
    attention_kv_heads: int = 0,
    boundaries: Sequence[str] = BOUNDARIES,
    recurrent_state_physical_layout: str = "head_key_value_contiguous",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for logical_token in range(total_tokens):
        for boundary in boundaries:
            if boundary == "new_state" and logical_token < prompt_tokens - 1:
                continue
            try:
                expected_count = _infer_oracle_row_count(
                    oracle_root, boundary, layer, logical_token, prompt_tokens
                )
                ck = _load_ck_row(
                    ck_root,
                    boundary,
                    layer,
                    logical_token,
                    prompt_tokens,
                    expected_count,
                    ck_prefill_mode,
                    attention_heads,
                    attention_kv_heads,
                )
                oracle = _load_oracle_row(
                    oracle_root, boundary, layer, logical_token, prompt_tokens, expected_count
                )
            except (FileNotFoundError, ValueError) as exc:
                rows.append({
                    "logical_token": logical_token,
                    "layer": layer,
                    "boundary": boundary,
                    "status": "missing_or_incompatible",
                    "error": str(exc),
                })
                continue
            row = {"logical_token": logical_token, "layer": layer, "boundary": boundary}
            row.update(
                compare_arrays(
                    boundary,
                    ck,
                    oracle,
                    state_size,
                    recurrent_state_physical_layout,
                )
            )
            rows.append(row)
    schedules = {
        "ck": "batched_then_sequential" if ck_prefill_mode == "hybrid" else "sequential_decode",
        "ck_prefix": "batched" if ck_prefill_mode == "hybrid" else "sequential",
        "ck_decode": "sequential",
        "oracle_prefix": "batched",
        "oracle_decode": "sequential",
    }
    first_value_divergence = next((
        {
            "logical_token": int(row["logical_token"]),
            "layer": int(row["layer"]),
            "boundary": str(row["boundary"]),
            "max_abs_diff": float(row.get("max_abs_diff", 0.0)),
        }
        for row in rows if row.get("status") == "different"
    ), None)
    material = classify(rows, schedules)
    return {
        "schema": "cke.xray.text-recurrent.v1",
        "schedules": schedules,
        "diagnostic_material_abs_floor": 1e-5,
        "acceptance_policy": "all value differences are reported; the material floor only prioritizes attribution",
        "rows": rows,
        "first_value_divergence": first_value_divergence,
        "first_material_divergence": material,
        "first_divergence": material or first_value_divergence,
    }


def _capture_ck_worker(
    model_dir: Path,
    prompt: list[int],
    generated: list[int],
    ck_prefill_mode: str,
    environment: dict[str, str],
) -> None:
    with _temporary_environment(environment):
        load_ck_logits_segmented(
            model_dir,
            prompt,
            generated,
            ck_prefill_mode=ck_prefill_mode,
        )


def _run_isolated_ck_capture(
    model_dir: Path,
    prompt: list[int],
    generated: list[int],
    ck_prefill_mode: str,
    environment: dict[str, str],
) -> None:
    # A loaded ctypes model and its mapped weights are not reliably unloaded
    # before the llama.cpp capture starts. Isolating CK keeps real 27B runs
    # below the pod memory ceiling and makes backend ownership explicit.
    process = multiprocessing.get_context("fork").Process(
        target=_capture_ck_worker,
        args=(model_dir, prompt, generated, ck_prefill_mode, environment),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"isolated CK X-ray capture failed with exit code {process.exitcode}"
        )


def _validate_reused_oracle_capture(
    oracle_root: Path,
    boundaries: Sequence[str],
    layer: int,
    physical_token: int,
) -> Path:
    for boundary in boundaries:
        name = ORACLE_BOUNDARY_NAMES.get(boundary, boundary)
        occurrence = ORACLE_BOUNDARY_OCCURRENCES.get(boundary, 0)
        checkpoint = oracle_root / (
            f"{name}-{layer}-token-{physical_token:06d}-occ-{occurrence:03d}.bin"
        )
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            return checkpoint
    raise ValueError(
        "--reuse-oracle-capture requested but no requested checkpoint exists "
        f"for layer {layer}, token {physical_token}: {oracle_root}"
    )


def _validate_reused_ck_capture(
    ck_root: Path,
    boundaries: Sequence[str],
    layer: int,
    logical_token: int,
) -> Path:
    for boundary in ck_capture_names(boundaries):
        checkpoint = ck_root / (
            f"tok_{logical_token:04d}_layer_{layer:03d}_{boundary}.f32"
        )
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            return checkpoint
    raise ValueError(
        "--reuse-ck-capture requested but no requested checkpoint exists "
        f"for layer {layer}, token {logical_token}: {ck_root}"
    )


def _reject_existing_ck_capture(
    ck_root: Path,
    boundaries: Sequence[str],
    layer: int,
) -> None:
    for boundary in ck_capture_names(boundaries):
        matches = tuple(
            ck_root.glob(f"tok_*_layer_{layer:03d}_{boundary}.f32")
        )
        if matches:
            raise ValueError(
                "fresh CK capture requested but a checkpoint already exists; "
                f"use a new capture root or --reuse-ck-capture: {matches[0]}"
            )


def capture_and_analyze(
    model_dir: Path,
    gguf: Path,
    parity_report: Path,
    capture_root: Path,
    layer: int,
    ctx_len: int,
    threads: int,
    ck_prefill_mode: str = "sequential",
    reuse_ck_capture: bool = False,
    reuse_oracle_capture: bool = False,
) -> dict[str, Any]:
    runtime_provenance = validate_runtime_provenance(model_dir)
    source = json.loads(parity_report.read_text(encoding="utf-8"))
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    state_size = int(config.get("ssm_state_size", 0))
    if state_size <= 0:
        raise ValueError("model config must declare a positive ssm_state_size")
    attention_heads = int(config.get("num_attention_heads", config.get("num_heads", 0)))
    if attention_heads <= 0:
        raise ValueError("model config must declare a positive attention head count")
    attention_kv_heads = int(config.get("num_key_value_heads", config.get("num_kv_heads", 0)))
    if attention_kv_heads <= 0:
        raise ValueError("model config must declare a positive key/value head count")
    boundaries = boundaries_for_layer(config, layer)
    physical_ck_boundaries = ck_capture_names(boundaries)
    if "new_state" in boundaries:
        (
            recurrent_state_physical_layout,
            recurrent_state_layout_provenance,
        ) = recurrent_state_layout_from_selected_provider(
            model_dir,
            config,
            layer,
        )
    else:
        recurrent_state_physical_layout = "head_key_value_contiguous"
        recurrent_state_layout_provenance = {
            "source": "not_applicable",
            "reason": "selected layer has no recurrent-state boundary",
        }
    prompt = [int(token) for token in source["initial_tokens"]]
    full_prefix = [int(token) for token in source["final_prefix"]]
    if full_prefix[: len(prompt)] != prompt:
        raise ValueError("parity report final_prefix does not begin with initial_tokens")
    generated = full_prefix[len(prompt) :]
    ck_root = capture_root / "ck"
    oracle_root = capture_root / "llama"
    ck_root.mkdir(parents=True, exist_ok=True)
    oracle_names = ",".join(
        f"{ORACLE_BOUNDARY_NAMES.get(name, name)}-{layer}" for name in boundaries
    )

    if reuse_ck_capture:
        _validate_reused_ck_capture(ck_root, boundaries, layer, 0)
    else:
        _reject_existing_ck_capture(ck_root, boundaries, layer)
        _run_isolated_ck_capture(
            model_dir,
            prompt,
            generated,
            ck_prefill_mode,
            {
                "CK_DEBUG_EXPORT_HIDDEN": str(ck_root),
                "CK_DEBUG_EXPORT_HIDDEN_LAYER": str(layer),
                "CK_DEBUG_EXPORT_HIDDEN_NAMES": ",".join(physical_ck_boundaries),
            },
        )

    if reuse_oracle_capture:
        _validate_reused_oracle_capture(
            oracle_root,
            boundaries,
            layer,
            len(prompt) - 1,
        )
        llama_meta = {
            "token_count_before": len(prompt),
            "token_count_after": len(full_prefix),
            "decode_mode": "sequential",
        }
    else:
        llama = _run_llama_capture(
            gguf,
            generated,
            ctx_len,
            20,
            threads,
            tokens_before=prompt,
            prefix_decode_mode="batched",
            decode_mode="sequential",
            dump_dir=oracle_root,
            dump_names=oracle_names,
        )
        llama_meta = llama["meta"]
    report = analyze_capture(
        ck_root, oracle_root, len(prompt), len(full_prefix), layer, state_size,
        ck_prefill_mode=ck_prefill_mode,
        attention_heads=attention_heads,
        attention_kv_heads=attention_kv_heads,
        boundaries=boundaries,
        recurrent_state_physical_layout=recurrent_state_physical_layout,
    )
    report["layer_kind"] = str(config["layer_kinds"][layer])
    report["requested_boundaries"] = list(boundaries)
    report["ck_capture_boundaries"] = list(physical_ck_boundaries)
    report["recurrent_state_physical_layout"] = recurrent_state_physical_layout
    report["recurrent_state_layout_provenance"] = recurrent_state_layout_provenance
    report["source_parity_report"] = str(parity_report)
    report["capture_root"] = str(capture_root)
    report["runtime_provenance"] = runtime_provenance
    report["llama_capture"] = {
        "token_count_before": int(llama_meta.get("token_count_before", -1)),
        "token_count_after": int(llama_meta.get("token_count_after", -1)),
        "decode_mode": str(llama_meta.get("decode_mode", "")),
        "reused": bool(reuse_oracle_capture),
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--parity-report", required=True, type=Path)
    ap.add_argument("--capture-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--ctx-len", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ck-prefill-mode", choices=("sequential", "hybrid"), default="sequential")
    ap.add_argument(
        "--reuse-ck-capture",
        action="store_true",
        help="Reuse an existing CK capture and run only the llama oracle plus analysis.",
    )
    ap.add_argument(
        "--reuse-oracle-capture",
        action="store_true",
        help="Reuse an existing llama capture and run only CK plus analysis.",
    )
    args = ap.parse_args()
    report = capture_and_analyze(
        args.model_dir.resolve(), args.gguf.resolve(), args.parity_report.resolve(),
        args.capture_root.resolve(), int(args.layer), int(args.ctx_len), int(args.threads),
        str(args.ck_prefill_mode),
        bool(args.reuse_ck_capture),
        bool(args.reuse_oracle_capture),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report.get("first_divergence"), sort_keys=True))
    return 0 if report.get("first_divergence") is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
