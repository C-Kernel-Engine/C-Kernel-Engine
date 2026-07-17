#!/usr/bin/env python3
"""Compare execution policy, cache state, and attention arithmetic in diagnostic order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator


SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "xray_execution_trace.schema.json"

ROLE_STAGE = {
    "new_key": (2, "CACHE_APPEND_SOURCE_DIVERGENCE"),
    "new_value": (2, "CACHE_APPEND_SOURCE_DIVERGENCE"),
    "stored_key": (3, "CACHE_APPEND_ROUNDTRIP_DIVERGENCE"),
    "stored_value": (3, "CACHE_APPEND_ROUNDTRIP_DIVERGENCE"),
    "previous_key_before": (3, "CACHE_PREVIOUS_ROW_DIVERGENCE"),
    "previous_key_after": (3, "CACHE_PREVIOUS_ROW_DIVERGENCE"),
    "previous_value_before": (3, "CACHE_PREVIOUS_ROW_DIVERGENCE"),
    "previous_value_after": (3, "CACHE_PREVIOUS_ROW_DIVERGENCE"),
    "valid_key_cache": (4, "CACHE_CONTENT_DIVERGENCE"),
    "valid_value_cache": (4, "CACHE_CONTENT_DIVERGENCE"),
    "query": (5, "ATTENTION_INPUT_DIVERGENCE"),
    "attention_output": (6, "ATTENTION_ARITHMETIC_DIVERGENCE"),
    "logits": (7, "RANKING_DIVERGENCE"),
}

REMEDIATIONS = {
    "EXECUTION_POLICY_MISMATCH": "Align prefill/decode segmentation and cache transitions in the circuit/runtime contract before tensor bisection.",
    "KERNEL_CONTRACT_MISMATCH": "Align the exact resolved provider, declared numerical contract, and shape-selected effective contract before comparing tensors.",
    "KERNEL_BATCH_SHAPE_MISMATCH": "Make both backends invoke the same semantic kernel over equivalent M/N/K batches before comparing arithmetic.",
    "POSITION_STATE_MISMATCH": "Align text/M-RoPE position policy and runtime position values before attention diagnostics.",
    "CACHE_STATE_METADATA_MISMATCH": "Fix cache token count, append index, or physical strides before comparing cache bytes.",
    "MISSING_STATE_ARTIFACT": "Export the requested bounded state artifact on both backends; do not substitute a downstream tensor.",
    "CACHE_APPEND_SOURCE_DIVERGENCE": "Compare the current token post-RoPE K/V producer before cache storage.",
    "CACHE_APPEND_ROUNDTRIP_DIVERGENCE": "Fix cache cast, append offset, or write/read layout; the newly stored row must round-trip exactly.",
    "CACHE_PREVIOUS_ROW_DIVERGENCE": "Fix cache overwrite or stride arithmetic; appending a token must not modify the previous row.",
    "CACHE_CONTENT_DIVERGENCE": "Find the first differing valid cache row and fix state history before attention arithmetic.",
    "ATTENTION_INPUT_DIVERGENCE": "Align query and valid cache inputs before comparing attention providers.",
    "ATTENTION_ARITHMETIC_DIVERGENCE": "With identical inputs, compare split threshold, worker partition, probability/value rounding, and reduction/merge order.",
    "RANKING_DIVERGENCE": "Teacher-force the same token sequence and attribute the first material logit ranking flip.",
    "DIAGNOSTIC_TRACE_ERROR": "Fix trace bounds, dtype, shape, checksum, or semantic role before diagnosing model math.",
}


class TraceError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TraceError(f"expected JSON object: {path}")
    return value


def _validate(trace: dict[str, Any], context: str) -> None:
    schema = _load_json(SCHEMA)
    errors = sorted(Draft202012Validator(schema).iter_errors(trace), key=lambda e: tuple(e.absolute_path))
    if errors:
        error = errors[0]
        where = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise TraceError(f"{context} violates execution trace schema at {where}: {error.message}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dtype(name: str) -> np.dtype:
    return {
        "fp32": np.dtype(np.float32), "fp16": np.dtype(np.float16),
        "bf16": np.dtype(np.uint16), "u8": np.dtype(np.uint8),
        "i32": np.dtype(np.int32), "i64": np.dtype(np.int64),
    }[name]


def _load_artifact(entry: dict[str, Any]) -> tuple[np.ndarray, bytes]:
    path = Path(entry["tensor_path"])
    if not path.is_file():
        raise TraceError(f"missing state artifact: {path}")
    raw = path.read_bytes()
    if _sha256(path) != entry["sha256"]:
        raise TraceError(f"state artifact checksum changed: {path}")
    values = np.frombuffer(raw, dtype=_dtype(entry["dtype"]))
    shape = tuple(int(value) for value in entry["shape"])
    if values.size != math.prod(shape):
        raise TraceError(f"{entry['checkpoint_id']}: {values.size} values != shape {shape}")
    values = values.reshape(shape)
    if entry["dtype"] == "bf16":
        values = (values.astype(np.uint32) << 16).view(np.float32)
    elif entry["dtype"] in {"fp16", "fp32"}:
        values = values.astype(np.float32)
    return values, raw


def _round_for_storage(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "fp16":
        return values.astype(np.float16).astype(np.float32)
    if dtype == "bf16":
        fp32 = values.astype(np.float32)
        bits = fp32.view(np.uint32)
        rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
        return ((rounded >> 16) << 16).view(np.float32)
    if dtype == "fp32":
        return values.astype(np.float32)
    return values.astype(_dtype(dtype))


def _row_evidence(left: np.ndarray, right: np.ndarray, row_axis: int) -> dict[str, Any]:
    lhs = np.moveaxis(left, row_axis, 0).reshape(left.shape[row_axis], -1)
    rhs = np.moveaxis(right, row_axis, 0).reshape(right.shape[row_axis], -1)
    differing = np.flatnonzero(np.any(lhs != rhs, axis=1))
    first = int(differing[0]) if differing.size else None
    worst = None
    if lhs.size:
        row_max = np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64)), axis=1)
        worst = int(np.argmax(row_max))
    hashes = []
    for index in sorted({0, max(0, lhs.shape[0] - 1), *(filter(lambda x: x is not None, [first, worst]))}):
        hashes.append({
            "row": int(index),
            "subject_sha256": hashlib.sha256(np.ascontiguousarray(lhs[index]).tobytes()).hexdigest(),
            "oracle_sha256": hashlib.sha256(np.ascontiguousarray(rhs[index]).tobytes()).hexdigest(),
        })
    return {"first_differing_row": first, "worst_row": worst, "row_hashes": hashes}


def _tensor_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    lhs = left.astype(np.float64, copy=False)
    rhs = right.astype(np.float64, copy=False)
    diff = lhs - rhs
    abs_diff = np.abs(diff)
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    flat = int(np.argmax(abs_diff)) if abs_diff.size else 0
    return {
        "cosine": float(np.dot(lhs.reshape(-1), rhs.reshape(-1)) / denom) if denom else 1.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "max_abs": float(abs_diff.reshape(-1)[flat]) if diff.size else 0.0,
        "mean_abs": float(np.mean(abs_diff)) if diff.size else 0.0,
        "worst_flat_index": flat,
        "finite": bool(np.isfinite(lhs).all() and np.isfinite(rhs).all()),
    }


def _fail(classification: str, **evidence: Any) -> dict[str, Any]:
    return {
        "status": "fail", "classification": classification,
        "recommended_action": REMEDIATIONS[classification], **evidence,
    }


def _canonical_calls(trace: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for call in trace["execution"]["calls"]:
        calls.append({key: call[key] for key in ("kind", "start", "count", "position_start", "cache_action")})
    return calls


def _kernel_batches(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [batch for call in trace["execution"]["calls"] for batch in call.get("kernel_batches", [])]


def _kernel_contracts(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("checkpoint_id", "kernel_id", "numerical_contract_id", "effective_contract_id")
    return [{field: batch[field] for field in fields} for batch in batches]


def _kernel_shapes(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("checkpoint_id", "m", "n", "k")
    return [{field: batch[field] for field in fields} for batch in batches]


def _artifact_index(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for entry in trace["artifacts"]:
        role = entry["role"]
        if role in indexed:
            raise TraceError(f"duplicate state artifact role: {role}")
        indexed[role] = entry
    return indexed


def _intra_trace_state_checks(trace: dict[str, Any], artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for stem in ("key", "value"):
        source = artifacts.get(f"new_{stem}")
        stored = artifacts.get(f"stored_{stem}")
        if (source is None) != (stored is None):
            return [_fail(
                "MISSING_STATE_ARTIFACT", stage=f"{stem}_append_roundtrip", backend=trace["backend"],
                source_present=source is not None, stored_present=stored is not None,
            )]
        if source is not None and stored is not None:
            try:
                source_values, _ = _load_artifact(source)
                stored_values, _ = _load_artifact(stored)
            except (TraceError, ValueError) as exc:
                return [_fail("DIAGNOSTIC_TRACE_ERROR", stage=f"{stem}_append_roundtrip", detail=str(exc))]
            expected = _round_for_storage(source_values, stored["dtype"])
            if expected.shape != stored_values.shape:
                return [_fail("DIAGNOSTIC_TRACE_ERROR", stage=f"{stem}_append_roundtrip",
                              detail=f"source shape {expected.shape} != stored shape {stored_values.shape}")]
            exact = bool(np.array_equal(expected, stored_values))
            row_axis = int(stored["row_axis"])
            evidence = {
                "stage": f"{stem}_append_roundtrip", "backend": trace["backend"],
                "status": "pass" if exact else "fail", "exact_after_storage_rounding": exact,
                "source_dtype": source["dtype"], "storage_dtype": stored["dtype"],
                "metrics": _tensor_metrics(expected, stored_values),
                **_row_evidence(expected, stored_values, row_axis),
            }
            if not exact:
                return [{**_fail("CACHE_APPEND_ROUNDTRIP_DIVERGENCE"), **evidence}]
            checks.append(evidence)

        before = artifacts.get(f"previous_{stem}_before")
        after = artifacts.get(f"previous_{stem}_after")
        if (before is None) != (after is None):
            return [_fail(
                "MISSING_STATE_ARTIFACT", stage=f"previous_{stem}_preservation", backend=trace["backend"],
                before_present=before is not None, after_present=after is not None,
            )]
        if before is not None and after is not None:
            try:
                before_values, before_raw = _load_artifact(before)
                after_values, after_raw = _load_artifact(after)
            except (TraceError, ValueError) as exc:
                return [_fail("DIAGNOSTIC_TRACE_ERROR", stage=f"previous_{stem}_preservation", detail=str(exc))]
            exact = before["dtype"] == after["dtype"] and before_values.shape == after_values.shape and before_raw == after_raw
            evidence = {
                "stage": f"previous_{stem}_preservation", "backend": trace["backend"],
                "status": "pass" if exact else "fail", "exact_bytes": exact,
            }
            if not exact:
                if before_values.shape == after_values.shape:
                    evidence["metrics"] = _tensor_metrics(before_values, after_values)
                    evidence.update(_row_evidence(before_values, after_values, int(before["row_axis"])))
                return [{**_fail("CACHE_PREVIOUS_ROW_DIVERGENCE"), **evidence}]
            checks.append(evidence)
    return checks


def compare_traces(subject: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    _validate(subject, "subject")
    _validate(oracle, "oracle")
    checks: list[dict[str, Any]] = []

    subject_calls = _canonical_calls(subject)
    oracle_calls = _canonical_calls(oracle)
    if subject["execution"]["policy_id"] != oracle["execution"]["policy_id"] or subject_calls != oracle_calls:
        checks.append(_fail(
            "EXECUTION_POLICY_MISMATCH", stage="execution_contract",
            subject_policy=subject["execution"]["policy_id"], oracle_policy=oracle["execution"]["policy_id"],
            subject_calls=subject_calls, oracle_calls=oracle_calls,
        ))
        return _report(subject, oracle, checks)
    checks.append({"stage": "execution_contract", "status": "pass"})

    subject_batches = _kernel_batches(subject)
    oracle_batches = _kernel_batches(oracle)
    subject_contracts = _kernel_contracts(subject_batches)
    oracle_contracts = _kernel_contracts(oracle_batches)
    if subject_contracts != oracle_contracts:
        checks.append(_fail(
            "KERNEL_CONTRACT_MISMATCH", stage="kernel_contracts",
            subject=subject_contracts, oracle=oracle_contracts,
        ))
        return _report(subject, oracle, checks)
    checks.append({"stage": "kernel_contracts", "status": "pass"})

    subject_shapes = _kernel_shapes(subject_batches)
    oracle_shapes = _kernel_shapes(oracle_batches)
    if subject_shapes != oracle_shapes:
        checks.append(_fail("KERNEL_BATCH_SHAPE_MISMATCH", stage="kernel_batches", subject=subject_shapes, oracle=oracle_shapes))
        return _report(subject, oracle, checks)
    checks.append({"stage": "kernel_batches", "status": "pass"})

    left_state, right_state = subject["state"], oracle["state"]
    position_fields = ("position_policy_id", "position")
    mismatch = next((field for field in position_fields if left_state[field] != right_state[field]), None)
    if mismatch:
        checks.append(_fail("POSITION_STATE_MISMATCH", stage="position_state", field=mismatch,
                            subject=left_state[mismatch], oracle=right_state[mismatch]))
        return _report(subject, oracle, checks)
    checks.append({"stage": "position_state", "status": "pass"})

    layout_left = {key: value for key, value in left_state["cache_layout"].items() if key != "base_address"}
    layout_right = {key: value for key, value in right_state["cache_layout"].items() if key != "base_address"}
    state_pairs = {
        "cache_token_count": (left_state["cache_token_count"], right_state["cache_token_count"]),
        "append_index": (left_state["append_index"], right_state["append_index"]),
        "cache_layout": (layout_left, layout_right),
    }
    mismatch = next((field for field, values in state_pairs.items() if values[0] != values[1]), None)
    if mismatch:
        checks.append(_fail("CACHE_STATE_METADATA_MISMATCH", stage="cache_metadata", field=mismatch,
                            subject=state_pairs[mismatch][0], oracle=state_pairs[mismatch][1],
                            subject_base=left_state["cache_layout"].get("base_address"),
                            oracle_base=right_state["cache_layout"].get("base_address")))
        return _report(subject, oracle, checks)
    checks.append({"stage": "cache_metadata", "status": "pass",
                   "subject_base": left_state["cache_layout"].get("base_address"),
                   "oracle_base": right_state["cache_layout"].get("base_address")})

    try:
        left_artifacts = _artifact_index(subject)
        right_artifacts = _artifact_index(oracle)
    except TraceError as exc:
        checks.append(_fail("DIAGNOSTIC_TRACE_ERROR", stage="artifact_index", detail=str(exc)))
        return _report(subject, oracle, checks)
    for trace, artifacts in ((subject, left_artifacts), (oracle, right_artifacts)):
        intra_checks = _intra_trace_state_checks(trace, artifacts)
        checks.extend(intra_checks)
        if intra_checks and intra_checks[-1]["status"] == "fail":
            return _report(subject, oracle, checks)
    roles = sorted(set(left_artifacts) | set(right_artifacts), key=lambda role: (ROLE_STAGE[role][0], role))
    for role in roles:
        left, right = left_artifacts.get(role), right_artifacts.get(role)
        if left is None or right is None:
            checks.append(_fail("MISSING_STATE_ARTIFACT", stage=role,
                                subject_present=left is not None, oracle_present=right is not None))
            return _report(subject, oracle, checks)
        try:
            left_values, left_raw = _load_artifact(left)
            right_values, right_raw = _load_artifact(right)
        except (TraceError, ValueError) as exc:
            checks.append(_fail("DIAGNOSTIC_TRACE_ERROR", stage=role, detail=str(exc)))
            return _report(subject, oracle, checks)
        if left["dtype"] != right["dtype"] or left_values.shape != right_values.shape or left["row_axis"] != right["row_axis"]:
            checks.append(_fail("DIAGNOSTIC_TRACE_ERROR", stage=role, detail="dtype, shape, or row axis differs",
                                subject={"dtype": left["dtype"], "shape": list(left_values.shape), "row_axis": left["row_axis"]},
                                oracle={"dtype": right["dtype"], "shape": list(right_values.shape), "row_axis": right["row_axis"]}))
            return _report(subject, oracle, checks)
        exact = left_raw == right_raw
        evidence = {
            "stage": role, "status": "pass" if exact else "fail", "exact_bytes": exact,
            "shape": list(left_values.shape), "dtype": left["dtype"],
            "metrics": _tensor_metrics(left_values, right_values),
            **_row_evidence(left_values, right_values, int(left["row_axis"])),
        }
        if not exact:
            classification = ROLE_STAGE[role][1]
            checks.append({**_fail(classification), **evidence})
            return _report(subject, oracle, checks)
        checks.append(evidence)
    return _report(subject, oracle, checks)


def _report(subject: dict[str, Any], oracle: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    first = next((check for check in checks if check["status"] == "fail"), None)
    return {
        "schema": "cke.xray_execution_state_report", "schema_version": 1,
        "subject_backend": subject["backend"], "oracle_backend": oracle["backend"],
        "status": "fail" if first else "pass", "checks": checks, "first_divergence": first,
        "diagnostic_order": [
            "execution_contract", "kernel_contracts", "kernel_batches", "position_state", "cache_metadata",
            "cache_append_source", "cache_append_roundtrip", "all_valid_cache_rows",
            "attention_inputs", "attention_arithmetic", "ranking",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-trace", type=Path, required=True)
    parser.add_argument("--oracle-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_traces(_load_json(args.subject_trace), _load_json(args.oracle_trace))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    first = result.get("first_divergence") or {}
    print(f"status={result['status']}")
    if first:
        print(f"stage={first.get('stage')}")
        print(f"class={first.get('classification')}")
        print(f"action={first.get('recommended_action')}")
    return 1 if first else 0


if __name__ == "__main__":
    raise SystemExit(main())
