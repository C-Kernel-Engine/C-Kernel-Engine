#!/usr/bin/env python3
"""Compare an executed llama.cpp non-flash attention capture with a verified CKE packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .replay_attention_from_dumps_v8 import _metrics
    from .xray_operation_bundle_v8 import BundleError, load_bundle
else:
    from replay_attention_from_dumps_v8 import _metrics
    from xray_operation_bundle_v8 import BundleError, load_bundle


_DTYPES = {0: np.dtype("<f4"), 1: np.dtype("<f2")}
_ROLES = ("q", "k", "v", "mask")


def read_tensor(base: Path, role: str | None) -> tuple[np.ndarray, dict]:
    meta_path = Path(str(base) + ".json")
    raw_path = Path(str(base) + ".bin")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dtype = _DTYPES.get(meta.get("type"))
    ne, nb, nbytes = meta.get("ne"), meta.get("nb"), meta.get("nbytes")
    if ((role is not None and meta.get("role") != role) or dtype is None or not isinstance(ne, list) or
            not isinstance(nb, list) or len(ne) != 4 or len(nb) != 4 or
            any(type(value) is not int or value < 1 for value in ne + nb) or
            type(nbytes) is not int or nbytes < 1 or nbytes > 64 * 1024 * 1024):
        raise BundleError(f"invalid {role} capture metadata")
    extent = dtype.itemsize + sum((size - 1) * stride for size, stride in zip(ne, nb))
    if raw_path.stat().st_size != nbytes:
        raise BundleError(f"incomplete or contradictory {role} capture bytes")
    raw = raw_path.read_bytes()
    if extent != nbytes or len(raw) != nbytes or nb[0] != dtype.itemsize:
        raise BundleError(f"incomplete or contradictory {role} capture bytes")
    values = np.ndarray(tuple(reversed(ne)), dtype=dtype, buffer=raw,
                        strides=tuple(reversed(nb))).copy()
    meta["sha256"] = hashlib.sha256(raw).hexdigest()
    return values, meta


def read_capture(directory: Path, token: int, layer: int = 0) -> tuple[dict[str, np.ndarray], dict]:
    stem = directory / f"kqv_out-{layer}-token-{token:06d}-occ-000"
    tensors = {}
    metadata = {}
    for role in _ROLES:
        tensors[role], metadata[role] = read_tensor(Path(str(stem) + f".nonflash_{role}"), role)
    if metadata["q"].get("capture_stage") != "score_matmul":
        raise BundleError("non-flash query must be captured at its score matmul")
    for role, name in (("scores", "kq"), ("softmax", "kq_soft_max")):
        stage = directory / f"{name}-{layer}-token-{token:06d}-occ-000"
        tensors[role], metadata[role] = read_tensor(stage, None)
        if metadata[role].get("token_id") != token or metadata[role].get("base_name") != f"{name}-{layer}":
            raise BundleError(f"{role} was not captured at the selected execution step")
    output_meta = json.loads(Path(str(stem) + ".json").read_text(encoding="utf-8"))
    output_path = Path(str(stem) + ".bin")
    if type(output_meta.get("nbytes")) is not int or output_meta["nbytes"] > 64 * 1024 * 1024 or (
        output_path.stat().st_size != output_meta["nbytes"]
    ):
        raise BundleError("oracle attention output has invalid byte extent")
    output = np.fromfile(output_path, dtype="<f4")
    if (output_meta.get("type") != 0 or output_meta.get("nbytes") != output.nbytes or
            output_meta.get("token_id") != token):
        raise BundleError("oracle attention output metadata is inconsistent")
    tensors["output"] = output
    output_meta["sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata["output"] = output_meta
    return tensors, metadata


def compare(bundle: Path, directory: Path, token: int) -> dict:
    manifest, cke = load_bundle(bundle)
    context = manifest.get("context") or {}
    if (manifest.get("operation") != "attention_sliding" or
            context.get("oracle_mode") not in {"disabled", "disabled_for_internal_dump"}):
        raise BundleError("packet is not a non-flash attention comparison")
    segments = context.get("segments")
    if not isinstance(segments, list) or any(type(n) is not int or n < 1 for n in segments):
        raise BundleError("invalid CKE segment plan")
    tokens = sum(segments)
    expected_axes = {
        "Qcur_rope": ["token", "head", "channel"],
        "Kcache_post_store": ["position", "kv_head", "channel"],
        "Vcache_post_store": ["position", "kv_head", "channel"],
        "kqv_out": ["token", "head", "channel"],
    }
    for role, axes in expected_axes.items():
        if (role not in cke or manifest["tensors"][role].get("axes") != axes):
            raise BundleError(f"{role} has missing or incompatible semantic axes")
    if token < 0 or token >= tokens or context.get("valid_cache_positions") != [0, tokens]:
        raise BundleError("selected query or CKE cache range is inconsistent")
    oracle, metadata = read_capture(directory, token, context["layer"])
    expected_types = {"q": 0, "k": 1, "v": 1, "mask": 0, "scores": 0, "softmax": 0}
    if any(metadata[role].get("type") != dtype for role, dtype in expected_types.items()):
        raise BundleError("oracle non-flash storage types differ from the declared FP16-cache contract")
    q = oracle["q"]
    k = oracle["k"]
    v = oracle["v"]
    mask = oracle["mask"]
    scores = oracle["scores"]
    probabilities = oracle["softmax"]
    heads, kv_heads, dim = cke["Qcur_rope"].shape[1], cke["Kcache_post_store"].shape[1], cke["Qcur_rope"].shape[2]
    query_rows = q.shape[2]
    query_start = token + 1 - query_rows
    segment_start = 0
    for length in segments:
        if segment_start <= query_start and token < segment_start + length:
            break
        segment_start += length
    else:
        raise BundleError("oracle query rows cross a declared CKE segment boundary")
    physical_keys = mask.shape[-1]
    if (query_rows < 1 or query_start < 0 or physical_keys < token + 1 or
            q.shape != (1, heads, query_rows, dim) or k.shape != (1, kv_heads, physical_keys, dim) or
            v.shape != (1, kv_heads, dim, mask.shape[-1]) or
            mask.shape != (1, 1, query_rows, physical_keys) or
            scores.shape != (1, heads, query_rows, physical_keys) or probabilities.shape != scores.shape or
            metadata["output"].get("ne") != [heads * dim, query_rows, 1, 1] or
            oracle["output"].shape != (query_rows * heads * dim,) or heads % kv_heads):
        raise BundleError("oracle graph geometry does not match the CKE packet")
    query = q[0].transpose(1, 0, 2).astype(np.float32)
    keys = k[0, :, :token + 1, :].astype(np.float32)
    values = v[0, :, :, :token + 1].transpose(0, 2, 1).astype(np.float32)
    expected_q = cke["Qcur_rope"][query_start:token + 1]
    expected_k = cke["Kcache_post_store"][:token + 1].transpose(1, 0, 2).astype(np.float32)
    expected_v = cke["Vcache_post_store"][:token + 1].transpose(1, 0, 2).astype(np.float32)
    if keys.shape != expected_k.shape or values.shape != expected_v.shape:
        raise BundleError("oracle cache extent differs from the CKE valid cache")
    exact_output = np.empty((query_rows, heads, dim), dtype=np.float64)
    f16_connected_output = np.empty_like(exact_output)
    f16_value_output = np.empty_like(exact_output)
    score_pairs = []
    probability_pairs = []
    f16_scores_by_row = []
    f16_probabilities_by_row = []
    masked_probability_max = 0.0
    for row in range(query_rows):
        position = query_start + row
        row_mask = mask[0, 0, row]
        if (not np.array_equal(row_mask[:position + 1], np.zeros(position + 1, dtype=np.float32)) or
                not np.isneginf(row_mask[position + 1:]).all()):
            raise BundleError("oracle mask does not describe the packet's causal valid range")
        masked_probability_max = max(
            masked_probability_max,
            float(np.max(np.abs(probabilities[0, :, row, position + 1:])))
            if position + 1 < physical_keys else 0.0,
        )
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            exact_scores = keys[kv_head, :position + 1].astype(np.float64) @ query[row, head].astype(np.float64)
            weights = np.exp(exact_scores - np.max(exact_scores))
            weights /= np.sum(weights)
            with np.errstate(over="ignore", invalid="ignore"):
                rounded_query = query[row, head].astype(np.float16).astype(np.float64)
            if not np.isfinite(rounded_query).all():
                raise BundleError("oracle query exceeds the FP16 matrix-multiply input contract")
            f16_scores = keys[kv_head, :position + 1].astype(np.float64) @ rounded_query
            f16_weights = np.exp(f16_scores - np.max(f16_scores))
            f16_weights /= np.sum(f16_weights)
            captured_weights = probabilities[0, head, row, :position + 1]
            score_pairs.append((scores[0, head, row, :position + 1], exact_scores))
            probability_pairs.append((captured_weights, weights))
            f16_scores_by_row.append(f16_scores)
            f16_probabilities_by_row.append(f16_weights)
            f16_value_output[row, head] = (
                captured_weights.astype(np.float16).astype(np.float64) @
                values[kv_head, :position + 1].astype(np.float64)
            )
            exact_output[row, head] = weights @ values[kv_head, :position + 1].astype(np.float64)
            f16_connected_output[row, head] = (
                f16_weights.astype(np.float16).astype(np.float64) @
                values[kv_head, :position + 1].astype(np.float64)
            )
    captured_scores = np.concatenate([pair[0] for pair in score_pairs])
    exact_scores = np.concatenate([pair[1] for pair in score_pairs])
    captured_probabilities = np.concatenate([pair[0] for pair in probability_pairs])
    exact_probabilities = np.concatenate([pair[1] for pair in probability_pairs])
    f16_scores = np.concatenate(f16_scores_by_row)
    f16_probabilities = np.concatenate(f16_probabilities_by_row)
    oracle_output = oracle["output"].reshape(query_rows, heads, dim)
    cke_output = cke["kqv_out"][query_start:token + 1]
    if not all(np.isfinite(item).all() for item in (
        query, keys, values, captured_scores, captured_probabilities, oracle_output,
    )):
        raise BundleError("nonfinite valid attention operand or result")
    row_error = np.max(np.abs(cke_output - oracle_output), axis=(1, 2))
    return {
        "status": "diagnostic_only",
        "interpretation": "Consumed operands and producer-stage captures are aligned; input drift and local arithmetic must be judged separately",
        "token": token,
        "query_start": query_start,
        "query_rows": query_rows,
        "segments": segments,
        "physical_keys": physical_keys,
        "valid_keys": token + 1,
        "oracle_capture_sha256": {role: metadata[role]["sha256"] for role in (*_ROLES, "scores", "softmax", "output")},
        "operand_metrics": {
            "q": _metrics(query, expected_q),
            "k_post_store": _metrics(keys, expected_k),
            "v_post_store": _metrics(values, expected_v),
        },
        "operand_bitwise_equal": {
            "q": bool(np.array_equal(query.view(np.uint32), expected_q.view(np.uint32))),
            "k_post_store": bool(np.array_equal(keys.astype(np.float16).view(np.uint16),
                                                   expected_k.astype(np.float16).view(np.uint16))),
            "v_post_store": bool(np.array_equal(values.astype(np.float16).view(np.uint16),
                                                   expected_v.astype(np.float16).view(np.uint16))),
        },
        "oracle_vs_fp64": {
            "scores": _metrics(captured_scores, exact_scores),
            "softmax": _metrics(captured_probabilities, exact_probabilities),
            "output": _metrics(oracle_output, exact_output),
        },
        "oracle_vs_f16_gemm": {
            "contract": "FP16-rounded Q for score GEMM; FP16-rounded softmax for value GEMM",
            "scores": _metrics(captured_scores, f16_scores),
            "softmax": _metrics(captured_probabilities, f16_probabilities),
            "value_output_from_captured_softmax": _metrics(oracle_output, f16_value_output),
            "connected_output": _metrics(oracle_output, f16_connected_output),
        },
        "cke_vs_fp64_output": _metrics(cke_output, exact_output),
        "cke_vs_oracle_output": _metrics(cke_output, oracle_output),
        "worst_cke_oracle_query_position": query_start + int(np.argmax(row_error)),
        "worst_cke_oracle_query_max_abs": float(np.max(row_error)),
        "oracle_masked_probability_max_abs": masked_probability_max,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--token", required=True, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.bundle, args.capture_dir, args.token)
    except (BundleError, OSError, ValueError, KeyError, TypeError) as exc:
        result = {"status": "incomplete", "error": str(exc)}
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
