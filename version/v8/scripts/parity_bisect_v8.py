#!/usr/bin/env python3
from __future__ import annotations

"""Compare canonical tensor checkpoints and identify the first divergent edge.

"Bisection" means progressively narrowing an ordered graph interval. The tool
does not infer model semantics or modify execution: parity profiles provide the
semantic names and thresholds, while backend manifests provide bounded tensor
exports. Missing evidence remains unresolved.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator


V8_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = V8_ROOT / "schemas"
MANIFEST_SCHEMA = SCHEMA_ROOT / "parity_tensor_manifest.schema.json"
PROFILE_SCHEMA = SCHEMA_ROOT / "parity_bisection_profile.schema.json"


class ParityContractError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParityContractError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ParityContractError(f"expected JSON object: {path}")
    return value


def _validate_schema(document: dict[str, Any], schema_path: Path, label: str) -> None:
    schema = _load_json(schema_path)
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda e: list(e.path))
    if not errors:
        return
    details = []
    for error in errors[:12]:
        location = ".".join(str(part) for part in error.path) or "<root>"
        details.append(f"{location}: {error.message}")
    raise ParityContractError(f"HARD PARITY CONTRACT FAULT in {label}: " + "; ".join(details))


def validate_manifest(document: dict[str, Any], label: str = "tensor manifest") -> None:
    _validate_schema(document, MANIFEST_SCHEMA, label)
    seen: set[str] = set()
    for tensor in document["tensors"]:
        checkpoint = tensor["checkpoint"]
        if checkpoint in seen:
            raise ParityContractError(f"HARD PARITY CONTRACT FAULT: duplicate checkpoint {checkpoint!r} in {label}")
        seen.add(checkpoint)
        shape = tensor["shape"]
        axes = tensor["axes"]
        canonical = tensor["canonical_axes"]
        if len(shape) != len(axes):
            raise ParityContractError(
                f"HARD PARITY CONTRACT FAULT: {checkpoint!r} shape rank {len(shape)} "
                f"does not match axes rank {len(axes)}"
            )
        if set(axes) != set(canonical) or len(axes) != len(canonical):
            raise ParityContractError(
                f"HARD PARITY CONTRACT FAULT: {checkpoint!r} axes {axes} and "
                f"canonical_axes {canonical} are not a permutation"
            )


def validate_profile(document: dict[str, Any]) -> None:
    _validate_schema(document, PROFILE_SCHEMA, "bisection profile")
    checkpoints = document["checkpoints"]
    groups = document["bisection"]["groups"]
    root = document["bisection"]["root_group"]
    if root not in groups:
        raise ParityContractError(f"HARD PARITY CONTRACT FAULT: unknown root group {root!r}")
    for group_id, group in groups.items():
        for checkpoint in group["checkpoints"]:
            if checkpoint not in checkpoints:
                raise ParityContractError(
                    f"HARD PARITY CONTRACT FAULT: group {group_id!r} names unknown checkpoint {checkpoint!r}"
                )
        for checkpoint, child in group["expand_on_failure"].items():
            if checkpoint not in group["checkpoints"]:
                raise ParityContractError(
                    f"HARD PARITY CONTRACT FAULT: group {group_id!r} expands checkpoint {checkpoint!r} "
                    "that it does not evaluate"
                )
            if child not in groups:
                raise ParityContractError(
                    f"HARD PARITY CONTRACT FAULT: group {group_id!r} names unknown child group {child!r}"
                )


def _dtype(dtype: str) -> np.dtype[Any]:
    return {
        "fp32": np.dtype("<f4"),
        "fp16": np.dtype("<f2"),
        "bf16": np.dtype("<u2"),
        "int32": np.dtype("<i4"),
        "int64": np.dtype("<i8"),
    }[dtype]


def _to_fp32(array: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        words = np.asarray(array, dtype=np.uint16)
        return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return array.astype(np.float32, copy=False)


def load_canonical_tensor(manifest_path: Path, tensor: dict[str, Any]) -> np.ndarray:
    data_path = (manifest_path.parent / tensor["path"]).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if tensor["format"] == "npy":
        raw = np.load(data_path, allow_pickle=False)
        expected_dtype = _dtype(tensor["dtype"])
        if raw.dtype != expected_dtype:
            raise ParityContractError(
                f"HARD PARITY CONTRACT FAULT: {tensor['checkpoint']!r} declares "
                f"{tensor['dtype']} but NPY stores {raw.dtype}"
            )
    else:
        raw = np.fromfile(data_path, dtype=_dtype(tensor["dtype"]))
    expected = math.prod(int(value) for value in tensor["shape"])
    if int(raw.size) != expected:
        raise ParityContractError(
            f"HARD PARITY CONTRACT FAULT: {tensor['checkpoint']!r} contains {raw.size} elements, "
            f"expected {expected} from shape {tensor['shape']}"
        )
    physical = _to_fp32(raw.reshape(tuple(tensor["shape"])), tensor["dtype"])
    permutation = [tensor["axes"].index(axis) for axis in tensor["canonical_axes"]]
    return np.ascontiguousarray(np.transpose(physical, axes=permutation), dtype=np.float32)


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ParityContractError(
            f"HARD PARITY CONTRACT FAULT: canonical shape mismatch: "
            f"reference={reference.shape}, candidate={candidate.shape}"
        )
    ref = reference.astype(np.float64, copy=False)
    got = candidate.astype(np.float64, copy=False)
    finite = bool(np.all(np.isfinite(ref)) and np.all(np.isfinite(got)))
    if ref.size == 0:
        raise ParityContractError("HARD PARITY CONTRACT FAULT: empty canonical tensor")
    diff = got - ref
    abs_diff = np.abs(diff)
    max_flat = int(np.argmax(abs_diff))
    max_coord = list(np.unravel_index(max_flat, reference.shape))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ref_rms = float(np.sqrt(np.mean(ref * ref)))
    denom = float(np.linalg.norm(ref.reshape(-1)) * np.linalg.norm(got.reshape(-1)))
    cosine = float(np.dot(ref.reshape(-1), got.reshape(-1)) / denom) if denom else (1.0 if np.array_equal(ref, got) else 0.0)
    return {
        "finite": finite,
        "cosine": cosine,
        "rmse": rmse,
        "ref_rms": ref_rms,
        "relative_rmse": rmse / ref_rms if ref_rms else (0.0 if rmse == 0.0 else float("inf")),
        "mean_abs": float(np.mean(abs_diff)),
        "max_abs": float(abs_diff.reshape(-1)[max_flat]),
        "worst_coordinate": max_coord,
        "worst_reference": float(ref.reshape(-1)[max_flat]),
        "worst_candidate": float(got.reshape(-1)[max_flat]),
    }


def _thresholds(profile: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    return profile["checkpoints"][checkpoint].get("thresholds", profile["defaults"])


def _evaluate(
    checkpoint: str,
    profile: dict[str, Any],
    reference_path: Path,
    reference_index: dict[str, dict[str, Any]],
    candidate_path: Path,
    candidate_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base = {
        "checkpoint": checkpoint,
        "producer": profile["checkpoints"][checkpoint]["producer"],
        "contract": profile["checkpoints"][checkpoint].get("contract"),
        "thresholds": _thresholds(profile, checkpoint),
    }
    missing = []
    if checkpoint not in reference_index:
        missing.append("reference")
    if checkpoint not in candidate_index:
        missing.append("candidate")
    if missing:
        return {**base, "status": "unresolved", "missing": missing}
    expected_producer = base["producer"]
    expected_contract = base["contract"]
    for backend, tensor in (("reference", reference_index[checkpoint]), ("candidate", candidate_index[checkpoint])):
        if tensor["producer"] != expected_producer:
            raise ParityContractError(
                f"HARD PARITY CONTRACT FAULT: {backend} checkpoint {checkpoint!r} producer "
                f"{tensor['producer']!r} does not match profile {expected_producer!r}"
            )
        if expected_contract is not None and tensor.get("contract") != expected_contract:
            raise ParityContractError(
                f"HARD PARITY CONTRACT FAULT: {backend} checkpoint {checkpoint!r} contract "
                f"{tensor.get('contract')!r} does not match profile {expected_contract!r}"
            )
    try:
        reference = load_canonical_tensor(reference_path, reference_index[checkpoint])
        candidate = load_canonical_tensor(candidate_path, candidate_index[checkpoint])
        metrics = _metrics(reference, candidate)
    except FileNotFoundError as exc:
        return {**base, "status": "unresolved", "missing_path": str(exc)}
    thresholds = base["thresholds"]
    failures = []
    if thresholds["require_finite"] and not metrics["finite"]:
        failures.append("non_finite")
    if metrics["cosine"] < thresholds["min_cosine"]:
        failures.append("cosine")
    if metrics["rmse"] > thresholds["max_rmse"]:
        failures.append("rmse")
    if metrics["relative_rmse"] > thresholds["max_relative_rmse"]:
        failures.append("relative_rmse")
    if metrics["max_abs"] > thresholds["max_abs"]:
        failures.append("max_abs")
    return {**base, "status": "fail" if failures else "pass", "failures": failures, "metrics": metrics}


def run_bisection(profile: dict[str, Any], reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = _load_json(reference_path)
    candidate = _load_json(candidate_path)
    validate_manifest(reference, "reference tensor manifest")
    validate_manifest(candidate, "candidate tensor manifest")
    validate_profile(profile)
    if reference["backend"] != profile["reference_backend"]:
        raise ParityContractError(
            f"reference backend {reference['backend']!r} does not match profile {profile['reference_backend']!r}"
        )
    if candidate["backend"] != profile["candidate_backend"]:
        raise ParityContractError(
            f"candidate backend {candidate['backend']!r} does not match profile {profile['candidate_backend']!r}"
        )
    ref_index = {row["checkpoint"]: row for row in reference["tensors"]}
    got_index = {row["checkpoint"]: row for row in candidate["tensors"]}
    groups = profile["bisection"]["groups"]
    group_id = profile["bisection"]["root_group"]
    visited: set[str] = set()
    group_reports = []
    first_observed_divergence = None
    first_divergence = None
    next_request = None
    while group_id:
        if group_id in visited:
            raise ParityContractError(f"HARD PARITY CONTRACT FAULT: bisection cycle at group {group_id!r}")
        visited.add(group_id)
        group = groups[group_id]
        rows = [
            _evaluate(checkpoint, profile, reference_path, ref_index, candidate_path, got_index)
            for checkpoint in group["checkpoints"]
        ]
        issue = next((row for row in rows if row["status"] != "pass"), None)
        group_reports.append({"group": group_id, "comparisons": rows, "first_issue": issue})
        if issue is None:
            break
        if issue["status"] == "fail":
            if first_observed_divergence is None:
                first_observed_divergence = issue
            first_divergence = issue
        child = group["expand_on_failure"].get(issue["checkpoint"])
        if not child:
            break
        child_checkpoints = groups[child]["checkpoints"]
        if any(checkpoint not in ref_index or checkpoint not in got_index for checkpoint in child_checkpoints):
            next_request = {"group": child, "checkpoints": child_checkpoints, "reason": issue["checkpoint"]}
            break
        group_id = child
    unresolved = any(
        row["status"] == "unresolved"
        for group in group_reports
        for row in group["comparisons"]
    )
    status = "unresolved" if unresolved and first_divergence is None else ("fail" if first_divergence else "pass")
    return {
        "schema": "cke.v8.parity_bisection_report",
        "schema_version": 1,
        "profile": profile["id"],
        "reference": {"backend": reference["backend"], "run": reference["run"]},
        "candidate": {"backend": candidate["backend"], "run": candidate["run"]},
        "status": status,
        "first_observed_divergence": first_observed_divergence,
        "first_divergence": first_divergence,
        "next_request": next_request,
        "groups": group_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_bisection(_load_json(args.profile), args.reference_manifest, args.candidate_manifest)
    except ParityContractError as exc:
        print(str(exc))
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary:
        print(json.dumps({
            "status": report["status"],
            "first_divergence": report["first_divergence"],
            "next_request": report["next_request"],
            "report": str(args.json_out),
        }, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
