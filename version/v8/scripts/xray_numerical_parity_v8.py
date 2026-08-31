#!/usr/bin/env python3
"""Canonical checkpoint comparison and bounded numerical divergence diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from jsonschema import Draft202012Validator


V8_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA = V8_ROOT / "schemas" / "checkpoint_manifest.schema.json"
PROFILE_SCHEMA = V8_ROOT / "schemas" / "parity_profile.schema.json"
RANKING_SCHEMA = V8_ROOT / "schemas" / "xray_ranking_report.schema.json"
PLANNER_PATH = Path(__file__).resolve().parent / "plan_parity_bisection_v8.py"


class XRayError(RuntimeError):
    pass


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise XRayError(f"expected JSON object: {path}")
    return value


def validate(value: Dict[str, Any], schema_path: Path, context: str) -> None:
    errors = sorted(
        Draft202012Validator(load_json(schema_path)).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise XRayError(f"{context} violates {schema_path.name} at {location}: {error.message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tensor(entry: Dict[str, Any]) -> np.ndarray:
    path = Path(entry["tensor_path"])
    dtype = entry["exported_dtype"]
    if dtype == "fp32":
        values = np.fromfile(path, dtype=np.float32)
    elif dtype == "fp16":
        values = np.fromfile(path, dtype=np.float16).astype(np.float32)
    elif dtype == "bf16":
        raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32)
        values = (raw << 16).view(np.float32)
    else:
        raise XRayError(f"unsupported exported dtype {dtype!r}")
    physical_shape = tuple(int(value) for value in entry["physical_shape"])
    if math.prod(physical_shape) != values.size:
        raise XRayError(
            f"{entry['checkpoint_id']}: file has {values.size} values, "
            f"physical_shape requires {math.prod(physical_shape)}"
        )
    tensor = values.reshape(physical_shape)
    physical_axes = list(entry["physical_axis_names"])
    logical_axes = list(entry["axis_names"])
    if set(physical_axes) != set(logical_axes) or len(physical_axes) != len(logical_axes):
        raise XRayError(
            f"{entry['checkpoint_id']}: physical axes {physical_axes} cannot canonicalize to {logical_axes}"
        )
    permutation = [physical_axes.index(axis) for axis in logical_axes]
    tensor = np.transpose(tensor, axes=permutation) if permutation != list(range(len(permutation))) else tensor
    logical_shape = tuple(int(value) for value in entry["logical_shape"])
    if tensor.shape != logical_shape:
        raise XRayError(
            f"{entry['checkpoint_id']}: canonical shape {tensor.shape} != declared {logical_shape}"
        )
    return np.ascontiguousarray(tensor, dtype=np.float32)


def _index_manifest(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    validate(manifest, MANIFEST_SCHEMA, f"{manifest.get('backend', 'backend')} checkpoint manifest")
    indexed: Dict[str, Dict[str, Any]] = {}
    for entry in manifest["checkpoints"]:
        checkpoint_id = entry["checkpoint_id"]
        if checkpoint_id in indexed:
            raise XRayError(f"duplicate checkpoint ID in manifest: {checkpoint_id}")
        path = Path(entry["tensor_path"])
        if not path.is_file():
            raise XRayError(f"checkpoint tensor does not exist: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise XRayError(f"checkpoint tensor checksum changed: {path}")
        indexed[checkpoint_id] = entry
    return indexed


def _metadata_fault(subject: Dict[str, Any], oracle: Dict[str, Any], required: Iterable[str]) -> tuple[str, str] | None:
    field_map = {"checkpoint_id": "checkpoint_id", "producer": "producer", "logical_layout": "logical_layout", "axis_names": "axis_names", "resolved_contract_id": "resolved_contract_id", "kernel_id": "kernel_id", "function": "function"}
    for requested in required:
        field = field_map.get(requested, requested)
        if subject.get(field) == oracle.get(field):
            continue
        if field == "producer":
            return "CIRCUIT_PRODUCER_MISMATCH", field
        if field in {"logical_layout", "axis_names", "logical_shape"}:
            return "LAYOUT_MISMATCH", field
        if field == "resolved_contract_id":
            checkpoint_id = str(subject.get("checkpoint_id", ""))
            if "rope" in checkpoint_id or "position" in checkpoint_id:
                return "POSITION_CONTRACT_MISMATCH", field
            if ".attention." in checkpoint_id:
                return "REDUCTION_CONTRACT_MISMATCH", field
            return "NUMERICAL_CONTRACT_MISMATCH", field
        if field in {"kernel_id", "function"}:
            return "KERNEL_BINDING_MISMATCH", field
        return "CHECKPOINT_ABI_MISMATCH", field
    if subject["storage_dtype"] != oracle["storage_dtype"]:
        return "STORAGE_CONTRACT_MISMATCH", "storage_dtype"
    return None


REMEDIATIONS = {
    "MISSING_CHECKPOINT": "Add the semantic checkpoint exporter or backend mapping; do not compare a substitute tensor.",
    "CIRCUIT_PRODUCER_MISMATCH": "Fix the circuit producer/consumer edge or stale backend mapping.",
    "LAYOUT_MISMATCH": "Correct named-axis canonicalization or the declared logical tensor layout.",
    "STORAGE_CONTRACT_MISMATCH": "Declare and implement the backend-matched storage/rounding boundary through the circuit and kernel map.",
    "REDUCTION_CONTRACT_MISMATCH": "Select or implement a kernel with the required accumulator, reduction, merge, and threading order.",
    "POSITION_CONTRACT_MISMATCH": "Align RoPE/M-RoPE pairing, width, axes, frequency precision, and rounding contract.",
    "NUMERICAL_CONTRACT_MISMATCH": "Register the measured semantic contract and resolve exactly one compatible kernel.",
    "KERNEL_BINDING_MISMATCH": "Fix resolver/lowering propagation; generated IR must retain the exact kernel ID and function.",
    "DIAGNOSTIC_EXPORT_MAPPING": "Fix exporter extent, dtype, shape, or axis metadata before blaming model math.",
    "KERNEL_IMPLEMENTATION_DIVERGENCE": "Reproduce the checkpoint in the isolated kernel parity test and fix its arithmetic.",
    "NONFINITE_OUTPUT": "Stop at this edge and fix the first NaN/Inf-producing kernel or input contract.",
    "RANKING_DIVERGENCE": "Attribute logits at this teacher-forced position; do not follow independently generated tokens.",
    "STATE_CACHE_DIVERGENCE": "Compare persistent decode with full replay and fix cache/state commit semantics.",
    "MISSING_TOLERANCE_PROFILE": "Add a backend/dtype threshold to the parity profile, not the circuit.",
}

FIX_OWNERS = {
    "MISSING_CHECKPOINT": "reference_adapter",
    "CIRCUIT_PRODUCER_MISMATCH": "circuit_or_reference_adapter",
    "LAYOUT_MISMATCH": "circuit_or_reference_adapter",
    "STORAGE_CONTRACT_MISMATCH": "circuit_and_kernel_map",
    "REDUCTION_CONTRACT_MISMATCH": "kernel_map_or_kernel",
    "POSITION_CONTRACT_MISMATCH": "circuit_kernel_map_or_kernel",
    "NUMERICAL_CONTRACT_MISMATCH": "circuit_and_kernel_map",
    "KERNEL_BINDING_MISMATCH": "generic_compiler_hardening",
    "DIAGNOSTIC_EXPORT_MAPPING": "reference_adapter",
    "KERNEL_IMPLEMENTATION_DIVERGENCE": "kernel",
    "NONFINITE_OUTPUT": "kernel_or_input_contract",
    "RANKING_DIVERGENCE": "first_divergent_edge",
    "STATE_CACHE_DIVERGENCE": "state_cache_kernel_or_contract",
    "MISSING_TOLERANCE_PROFILE": "parity_profile",
}

ARCHITECTURE_POLICY = {
    "dsl_role": "deterministically stitch and emit the circuit and resolved kernel-map decisions",
    "allowed_dsl_changes": [
        "generic validation",
        "fail-closed resolution",
        "metadata propagation",
        "deterministic code generation",
        "removal of implicit assumptions",
    ],
    "forbidden_fix": "Do not add model-name or checkpoint-specific kernel-selection branches to DSL/codegen.",
    "required_kernel_validation": "Kernel arithmetic fixes require isolated scalar/reference parity plus the applicable llama.cpp or PyTorch parity gate.",
}


XRAY_FIX_PROGRESSION = {
    "policy": "advance_only_after_numerical_evidence",
    "steps": [
        "Stop at the first divergent semantic edge; do not debug later outputs.",
        "Classify storage, compute, reduction, position, layout, circuit, binding, or implementation semantics.",
        "Check the kernel map for an exact compatible numerical capability; never choose the nearest variant.",
        "If absent, add one additive kernel variant. If present but wrong, fix that implementation without changing unrelated variants.",
        "Extend the kernel-family unit matrix for the complete input-storage, compute, accumulator/reduction, rounding, output-storage, and threading contract.",
        "Validate against an independent scalar formula and the requested PyTorch or llama.cpp backend oracle.",
        "Register the exact function and validated contract in the kernel map; unsupported and ambiguous resolutions must hard-fail.",
        "When a small stored delta precedes a later failure, run same-backend forward/VJP sensitivity ablations before blaming a backward provider; Q, K, and V perturbations must be isolated.",
        "If the final encoder prefix is byte-exact but teacher-forced ranking fails, close the encoder interval and switch X-ray to decoder mixed-prefill/decode checkpoints at the first failing position.",
        "When sparse tensors pass but ranking fails, expand only the measured interval with the largest positive relative-RMSE growth; never reopen an all-exact phase.",
        "Run isolated kernel tests, numerical-contract resolution, stitched checkpoint parity, mixed-prefill logits, and teacher-forced parity in that order.",
        "Rerun X-ray from the last passing checkpoint and confirm the first failure progresses to a later semantic edge.",
        "Preserve the new evidence in nightly and the HTML test-report capability accordion so the bug cannot silently recur."
    ]
}


def _max_bf16_ulp_distance(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    abs_floor: float,
) -> int | None:
    ref32 = np.asarray(reference, dtype=np.float32)
    got32 = np.asarray(actual, dtype=np.float32)
    if not np.isfinite(ref32).all() or not np.isfinite(got32).all():
        return None
    material = np.abs(got32 - ref32) >= float(abs_floor)
    if not np.any(material):
        return 0
    ref_bits32 = ref32.view(np.uint32)
    got_bits32 = got32.view(np.uint32)
    if (
        np.any(ref_bits32[material] & np.uint32(0xFFFF))
        or np.any(got_bits32[material] & np.uint32(0xFFFF))
    ):
        return None
    ref_bits = (ref_bits32[material] >> np.uint32(16)).astype(np.uint16)
    got_bits = (got_bits32[material] >> np.uint32(16)).astype(np.uint16)

    def ordered(bits: np.ndarray) -> np.ndarray:
        magnitude = (bits & np.uint16(0x7FFF)).astype(np.int32)
        return np.where(bits & np.uint16(0x8000), 0x8000 - magnitude, 0x8000 + magnitude)

    return int(np.max(np.abs(ordered(ref_bits) - ordered(got_bits))))


def _metrics(
    reference: np.ndarray,
    actual: np.ndarray,
    axes: list[str],
    *,
    bf16_abs_floor: float | None = None,
) -> Dict[str, Any]:
    if reference.shape != actual.shape:
        raise XRayError(f"canonical tensor shape mismatch: {reference.shape} != {actual.shape}")
    ref64 = reference.astype(np.float64, copy=False)
    got64 = actual.astype(np.float64, copy=False)
    diff = got64 - ref64
    abs_diff = np.abs(diff)
    flat_index = int(np.argmax(abs_diff)) if diff.size else 0
    coordinate = np.unravel_index(flat_index, diff.shape) if diff.size else tuple(0 for _ in diff.shape)
    denom = float(np.linalg.norm(ref64) * np.linalg.norm(got64))
    rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    ref_rms = float(np.sqrt(np.mean(ref64 * ref64))) if diff.size else 0.0
    exact_elements = int(np.count_nonzero(reference == actual))
    total_elements = int(reference.size)
    return {
        "cosine": float(np.dot(ref64.reshape(-1), got64.reshape(-1)) / denom) if denom else 1.0,
        "rmse": rmse,
        "relative_rmse": rmse / ref_rms if ref_rms else (0.0 if rmse == 0.0 else float("inf")),
        "mean_abs": float(np.mean(abs_diff)) if diff.size else 0.0,
        "max_abs": float(abs_diff.reshape(-1)[flat_index]) if diff.size else 0.0,
        "worst_coordinate": {axis: int(value) for axis, value in zip(axes, coordinate)},
        "finite": bool(np.isfinite(reference).all() and np.isfinite(actual).all()),
        "exact_elements": exact_elements,
        "total_elements": total_elements,
        "exact_ratio": exact_elements / total_elements if total_elements else 1.0,
        "byte_exact": exact_elements == total_elements,
        "bf16_abs_floor": bf16_abs_floor,
        "max_bf16_ulp_over_abs_floor": (
            _max_bf16_ulp_distance(reference, actual, abs_floor=bf16_abs_floor)
            if bf16_abs_floor is not None else None
        ),
    }


def _metric_status(metrics: Dict[str, Any], threshold: Dict[str, Any]) -> tuple[str, list[str]]:
    failures = []
    if threshold["finite_required"] and not metrics["finite"]:
        failures.append("nonfinite")
    if metrics["cosine"] < threshold["cosine_min"]:
        failures.append("cosine")
    if metrics["rmse"] > threshold["rmse_max"]:
        failures.append("rmse")
    if metrics["relative_rmse"] > threshold["relative_rmse_max"]:
        failures.append("relative_rmse")
    max_bf16_ulp = threshold.get("max_bf16_ulp_max")
    if max_bf16_ulp is None:
        if metrics["max_abs"] > threshold["max_abs_max"]:
            failures.append("max_abs")
    else:
        safety_max = float(threshold.get("max_abs_safety_max", threshold["max_abs_max"]))
        if metrics["max_abs"] > safety_max:
            failures.append("max_abs_safety")
        if metrics.get("max_bf16_ulp_over_abs_floor") is None:
            failures.append("bf16_ulp_unavailable")
        elif metrics["max_bf16_ulp_over_abs_floor"] > max_bf16_ulp:
            failures.append("max_bf16_ulp")
    return ("fail" if failures else "pass"), failures


def _resolved_execution(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the exact numerical provider attached to every measured edge."""
    return {
        "producer": entry.get("producer"),
        "phase": entry.get("phase"),
        "layer": entry.get("layer"),
        "storage_dtype": entry.get("storage_dtype"),
        "exported_dtype": entry.get("exported_dtype"),
        "resolved_contract_id": entry.get("resolved_contract_id"),
        "kernel_id": entry.get("kernel_id"),
        "function": entry.get("function"),
    }


def _drift_progression(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Describe error accumulation without inventing another pass threshold.

    Relative RMSE is scale-normalized, so its delta is useful across adjacent
    semantic checkpoints. Ratios are reported only after a non-zero baseline;
    the first rounding difference is intentionally not called infinite
    amplification.
    """
    checkpoints = []
    previous = None
    first_non_exact = None
    largest_boundary = None
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        current = float(metrics["relative_rmse"])
        item = {
            "checkpoint_id": row["checkpoint_id"],
            "status": row["status"],
            "relative_rmse": current,
            "rmse": float(metrics["rmse"]),
            "max_abs": float(metrics["max_abs"]),
            "byte_exact": bool(metrics["byte_exact"]),
            "resolved_execution": row.get("resolved_execution"),
            "previous_checkpoint_id": previous["checkpoint_id"] if previous else None,
            "relative_rmse_delta": current - previous["relative_rmse"] if previous else None,
            "relative_rmse_ratio": (
                current / previous["relative_rmse"]
                if previous and previous["relative_rmse"] > 0.0
                else None
            ),
        }
        checkpoints.append(item)
        if first_non_exact is None and not item["byte_exact"]:
            first_non_exact = item
        if previous is not None:
            boundary = {
                "from_checkpoint_id": previous["checkpoint_id"],
                "to_checkpoint_id": item["checkpoint_id"],
                "relative_rmse_before": previous["relative_rmse"],
                "relative_rmse_after": current,
                "relative_rmse_delta": item["relative_rmse_delta"],
                "relative_rmse_ratio": item["relative_rmse_ratio"],
                "resolved_execution": item["resolved_execution"],
            }
            if largest_boundary is None or boundary["relative_rmse_delta"] > largest_boundary["relative_rmse_delta"]:
                largest_boundary = boundary
        previous = item
    return {
        "metric": "relative_rmse",
        "policy": "observational_no_additional_tolerance",
        "first_non_exact": first_non_exact,
        "largest_amplification_boundary": largest_boundary,
        "checkpoints": checkpoints,
    }


def _load_planner():
    spec = importlib.util.spec_from_file_location("plan_parity_bisection_v8", PLANNER_PATH)
    if spec is None or spec.loader is None:
        raise XRayError(f"cannot load parity planner: {PLANNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_manifests(
    subject: Dict[str, Any],
    oracle: Dict[str, Any],
    profile: Dict[str, Any],
    ranking_report: Dict[str, Any] | None = None,
    checkpoint_order: list[str] | None = None,
) -> Dict[str, Any]:
    validate(profile, PROFILE_SCHEMA, "parity profile")
    subject_index = _index_manifest(subject)
    oracle_index = _index_manifest(oracle)
    rows = []
    first_classification = None
    first_non_exact = None
    last_passing_checkpoint = None
    unresolved_contracts = []
    active_order = checkpoint_order or profile["checkpoint_order"]
    for checkpoint_id in active_order:
        left = subject_index.get(checkpoint_id)
        right = oracle_index.get(checkpoint_id)
        if left is None or right is None:
            row = {"checkpoint_id": checkpoint_id, "status": "fail", "classification": "MISSING_CHECKPOINT", "subject_present": left is not None, "oracle_present": right is not None}
            rows.append(row)
            first_classification = row
            break
        if left.get("resolved_contract_id") == "unresolved":
            unresolved_contracts.append(checkpoint_id)
        mismatch = _metadata_fault(left, right, profile["required_match_fields"])
        if mismatch is not None:
            classification, field = mismatch
            row = {"checkpoint_id": checkpoint_id, "status": "fail", "classification": classification, "field": field, "subject": left.get(field), "oracle": right.get(field)}
        else:
            try:
                ref = _load_tensor(right)
                got = _load_tensor(left)
            except (XRayError, ValueError) as exc:
                row = {"checkpoint_id": checkpoint_id, "status": "fail", "classification": "DIAGNOSTIC_EXPORT_MAPPING", "detail": str(exc)}
                rows.append(row)
                first_classification = row
                break
            threshold_key = left["storage_dtype"] if left["storage_dtype"] in profile["dtype_thresholds"] else left["exported_dtype"]
            threshold = profile["dtype_thresholds"].get(threshold_key)
            if threshold is None:
                row = {"checkpoint_id": checkpoint_id, "status": "fail", "classification": "MISSING_TOLERANCE_PROFILE", "dtype": threshold_key}
            else:
                metrics = _metrics(
                    ref,
                    got,
                    left["axis_names"],
                    bf16_abs_floor=(
                        float(threshold["max_abs_max"])
                        if threshold.get("max_bf16_ulp_max") is not None else None
                    ),
                )
                status, failed_metrics = _metric_status(metrics, threshold)
                classification = "NONFINITE_OUTPUT" if "nonfinite" in failed_metrics else ("KERNEL_IMPLEMENTATION_DIVERGENCE" if status == "fail" else "MATCH")
                row = {
                    "checkpoint_id": checkpoint_id,
                    "status": status,
                    "classification": classification,
                    "metrics": metrics,
                    "failed_metrics": failed_metrics,
                    "threshold": threshold,
                    "resolved_execution": _resolved_execution(left),
                }
        rows.append(row)
        if (
            first_non_exact is None
            and row.get("metrics") is not None
            and not bool(row["metrics"].get("byte_exact", False))
        ):
            first_non_exact = {
                "checkpoint_id": checkpoint_id,
                "status": row["status"],
                "classification": "NON_BYTE_EXACT",
                "metrics": row["metrics"],
            }
        if row["status"] == "pass":
            last_passing_checkpoint = checkpoint_id
        if row["status"] == "fail" and first_classification is None:
            first_classification = row
            break

    ranking = None
    if first_classification is None and ranking_report is not None:
        validate(ranking_report, RANKING_SCHEMA, "X-ray ranking report")
        checks = ranking_report.get("checks", [])
        first = next((item for item in checks if item.get("status") == "fail"), None)
        if first is not None:
            kind = str(first.get("kind", "ranking"))
            classification = "STATE_CACHE_DIVERGENCE" if kind == "persistent_vs_replay" else "RANKING_DIVERGENCE"
            ranking = {**first, "classification": classification}
            first_classification = ranking

    if first_classification is not None:
        classification = str(first_classification.get("classification", ""))
        first_classification["fix_owner"] = FIX_OWNERS.get(classification, "first_divergent_edge")
        first_classification["recommended_action"] = REMEDIATIONS.get(
            classification, "Inspect the first failing semantic edge and its resolved contract metadata."
        )

    planner = _load_planner()
    plan = planner.plan(
        profile,
        {"comparisons": rows, "ranking_divergence": ranking},
        checkpoint_order=active_order,
    )
    return {
        "schema": "cke.xray_numerical_report",
        "schema_version": 1,
        "subject_backend": subject["backend"],
        "oracle_backend": oracle["backend"],
        "status": "fail" if first_classification is not None else "pass",
        "comparisons": rows,
        "first_divergence": first_classification,
        "first_non_exact_checkpoint": first_non_exact,
        "drift_progression": _drift_progression(rows),
        "last_passing_checkpoint": last_passing_checkpoint,
        "unresolved_contract_checkpoints": unresolved_contracts,
        "ranking_divergence": ranking,
        "next_plan": plan,
        "architecture_policy": ARCHITECTURE_POLICY,
        "fix_progression": XRAY_FIX_PROGRESSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-manifest", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--ranking-report", type=Path)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_manifests(
        load_json(args.subject_manifest), load_json(args.oracle_manifest), load_json(args.profile),
        load_json(args.ranking_report) if args.ranking_report else None,
        args.checkpoint or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    divergence = result.get("first_divergence") or {}
    print(f"status={result['status']}")
    if divergence:
        print(f"fail_at={divergence.get('checkpoint_id', divergence.get('position', 'ranking'))}")
        print(f"class={divergence.get('classification')}")
        print(f"fix_owner={divergence.get('fix_owner')}")
        print(f"action={divergence.get('recommended_action')}")
    amplification = (result.get("drift_progression") or {}).get("largest_amplification_boundary")
    if amplification:
        print(
            "largest_amplification="
            f"{amplification['from_checkpoint_id']}->{amplification['to_checkpoint_id']} "
            f"relative_rmse_delta={amplification['relative_rmse_delta']:.9g}"
        )
    print(f"next={','.join(result['next_plan'].get('next_checkpoints', []))}")
    return 1 if result["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
