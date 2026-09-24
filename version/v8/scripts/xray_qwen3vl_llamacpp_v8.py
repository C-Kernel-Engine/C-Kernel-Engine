#!/usr/bin/env python3
"""Qwen3-VL GGUF adapter for the unified vision X-ray surface."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import activation_parity_qwen3vl_mmproj_v8 as capture_adapter
import xray_numerical_parity_v8 as xray


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE = SCRIPT_DIR.parent / "parity_profiles" / "qwen3vl_llamacpp_q8_v1.json"

# This producer is specifically a vision-encoder prefill capture: the phase is
# a property of the executed capture path, never a user declaration. If future
# execution modes are added, phase must be derived from the mode actually run.
CAPTURE_PHASE = "prefill"


def _format_layer(value: str, layer: int) -> str:
    return value.replace("{layer}", str(layer))


def _active_checkpoints(profile: dict[str, Any], layer: int) -> list[tuple[str, dict[str, Any]]]:
    active: list[tuple[str, dict[str, Any]]] = []
    mappings = profile["backend_mappings"]
    for template_id in profile["checkpoint_order"]:
        mapping = mappings.get(template_id)
        if not isinstance(mapping, dict):
            raise RuntimeError(f"llama.cpp profile has no mapping for {template_id}")
        active.append((_format_layer(template_id, layer), mapping))
    return active


def _legacy_result_index(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for row in report.get("results", []):
        key = (int(row.get("layer", -1)), str(row.get("op", "")))
        if key in indexed:
            raise RuntimeError(f"ambiguous legacy parity result for layer={key[0]} op={key[1]}")
        indexed[key] = row
    return indexed


def _observed_dtype(profile: dict[str, Any], checkpoint_id: str) -> str | None:
    """Profile-declared observed storage dtype (not a resolved kernel dtype)."""
    observed = profile.get("observed_storage")
    if not isinstance(observed, dict):
        return None
    per_checkpoint = observed.get("checkpoints")
    if isinstance(per_checkpoint, dict):
        value = per_checkpoint.get(checkpoint_id)
        if isinstance(value, str) and value:
            return value
    default = observed.get("default")
    return str(default) if isinstance(default, str) and default else None


def _resolved_execution(
    mapping: dict[str, Any],
    layer: int,
    profile: dict[str, Any],
    checkpoint_id: str,
    phase: str,
) -> dict[str, Any]:
    """Canonical execution identity for circuit join.

    Only semantic fields known from the parity profile are populated
    (producer, phase, layer). op_idx, function, kernel_id, and
    resolved_contract_id must be resolved from call IR / kernel maps by a
    resolver with access to that metadata; they stay null here rather than
    being fabricated from the producer name or a GGUF filename.
    """
    return {
        "producer": str(mapping.get("producer", "unknown")),
        "phase": phase,
        "layer": int(mapping.get("result_layer", layer)),
        "observed_dtype": _observed_dtype(profile, checkpoint_id),
        "op_idx": None,
        "function": None,
        "kernel_id": None,
        "resolved_contract_id": None,
        "storage_dtype": None,
        "exported_dtype": None,
    }


def _canonical_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """Map legacy parity row metrics to the canonical checkpoint contract.

    Metrics are preserved under their mathematically correct names. The
    capture adapter reports mean absolute error (mean_abs_diff) and mean
    relative error (mean_rel_err); it does not compute squared-error
    statistics, so rmse / relative_rmse stay null unless the capture row
    actually provides them. MAE is never substituted for RMSE.
    """
    max_abs = row.get("max_abs_diff")
    mean_abs = row.get("mean_abs_diff")
    rmse = row.get("rmse")
    relative_rmse = row.get("relative_rmse")
    byte_exact = bool(row.get("byte_exact", False))
    exact_ratio = row.get("exact_ratio")
    if not byte_exact and max_abs is not None and max_abs == 0.0:
        byte_exact = True
    if exact_ratio is None and byte_exact:
        exact_ratio = 1.0
    metrics: dict[str, Any] = {
        "max_abs": float(max_abs) if max_abs is not None else None,
        "mean_abs": float(mean_abs) if mean_abs is not None else None,
        "rmse": float(rmse) if rmse is not None else None,
        "relative_rmse": float(relative_rmse) if relative_rmse is not None else None,
        "byte_exact": byte_exact,
        "exact_ratio": float(exact_ratio) if exact_ratio is not None else None,
    }
    # Preserve legacy fields for diagnostics under their source names.
    for key in ("max_rel_err", "mean_rel_err", "diverge_idx"):
        if key in row:
            metrics[key] = row[key]
    return metrics


def normalize_capture_report(
    report: dict[str, Any],
    profile: dict[str, Any],
    layer: int,
    execution_mode: str = "strict",
    phase: str = "prefill",
) -> dict[str, Any]:
    indexed = _legacy_result_index(report)
    comparisons: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    first_incomplete_capture: dict[str, Any] | None = None
    incomplete_before_divergence = False
    first_non_exact: dict[str, Any] | None = None
    last_passing: str | None = None
    consumed_keys: set[tuple[int, str]] = set()
    # Normalize every profile checkpoint in profile order. Normalization does
    # not stop at the first failure: the capture already paid for every dump,
    # and the drift chart needs the full observed progression. Only the first
    # failure is attributed (first_divergence); later failures are reported as
    # OBSERVED_DIVERGENCE — they must not be labelled propagated without an
    # exact-input replay that proves the local kernel passes.
    for checkpoint_id, mapping in _active_checkpoints(profile, layer):
        result_layer = int(mapping.get("result_layer", layer))
        result_name = str(mapping["result_tensor"])
        row = indexed.get((result_layer, result_name))
        if row is None:
            comparison = {
                "checkpoint_id": checkpoint_id,
                "status": "fail",
                "classification": "MISSING_CHECKPOINT",
                "subject_present": False,
                "oracle_present": False,
                "resolved_execution": _resolved_execution(
                    mapping, layer, profile, checkpoint_id, phase
                ),
            }
        else:
            consumed_keys.add((result_layer, result_name))
            legacy_status = str(row.get("status", "ERROR")).upper()
            incomplete = bool(row.get("size_mismatch"))
            status = "incomplete" if incomplete else ("pass" if legacy_status == "PASS" else "fail")
            if incomplete:
                classification = "INCOMPLETE_CAPTURE"
            elif status == "pass":
                classification = "MATCH"
            elif bool(row.get("has_nan")) or bool(row.get("has_inf")):
                classification = "NONFINITE_OUTPUT"
            elif first_divergence is None and first_incomplete_capture is None:
                classification = "KERNEL_IMPLEMENTATION_DIVERGENCE"
            else:
                classification = "OBSERVED_DIVERGENCE"
            comparison = {
                "checkpoint_id": checkpoint_id,
                "status": status,
                "classification": classification,
                "legacy_tensor": result_name,
                "metrics": _canonical_metrics(row),
                "capture_extents": {
                    "subject": row.get("test_shape"), "oracle": row.get("ref_shape")
                } if incomplete else None,
                "resolved_execution": _resolved_execution(
                    mapping, layer, profile, checkpoint_id, phase
                ),
            }
        comparisons.append(comparison)
        if comparison["status"] == "incomplete":
            if first_incomplete_capture is None:
                first_incomplete_capture = comparison
            continue
        if comparison["status"] == "pass":
            metrics = comparison.get("metrics") or {}
            if first_non_exact is None and float(metrics.get("max_abs", 0.0) or 0.0) != 0.0:
                first_non_exact = {
                    "checkpoint_id": checkpoint_id,
                    "legacy_tensor": result_name,
                    "metrics": metrics,
                }
            last_passing = checkpoint_id
            continue
        if first_divergence is None:
            first_divergence = comparison
            incomplete_before_divergence = first_incomplete_capture is not None

    if first_divergence is not None:
        if incomplete_before_divergence:
            first_divergence["classification"] = "OBSERVED_DIVERGENCE"
            first_divergence["attribution_status"] = "observed_after_incomplete_capture"
            first_divergence["fix_owner"] = "capture_and_exact_input_control"
            first_divergence["recommended_action"] = (
                "Repair the earlier incomplete capture, then replay this aligned numerical "
                "failure with exact inputs before attributing its origin."
            )
        elif first_non_exact is not None:
            first_divergence["classification"] = "DOWNSTREAM_OR_PROPAGATED_DIVERGENCE"
            first_divergence["causal_origin_candidate"] = first_non_exact
            first_divergence["fix_owner"] = "exact_input_control"
            first_divergence["recommended_action"] = (
                "Replay this operation with the oracle tensor from the first non-exact "
                "upstream checkpoint. Fix the downstream kernel only if that exact-input "
                "control still diverges."
            )
        else:
            classification = str(first_divergence["classification"])
            first_divergence["fix_owner"] = xray.FIX_OWNERS.get(
                classification, "first_divergent_edge"
            )
            first_divergence["recommended_action"] = xray.REMEDIATIONS.get(
                classification, "Inspect the first failing semantic edge."
            )

    # Capture rows the profile never references (e.g. adapter sweeps at the
    # frontend scope). They are disclosed, never silently discarded.
    unmatched_capture_rows = [
        {"layer": int(row.get("layer", -1)), "op": str(row.get("op", "")),
         "status": str(row.get("status", "ERROR"))}
        for row in report.get("results", [])
        if (int(row.get("layer", -1)), str(row.get("op", ""))) not in consumed_keys
    ]

    return {
        "schema": "cke.xray_numerical_report",
        "schema_version": 1,
        "subject_backend": "ck",
        "oracle_backend": "llamacpp",
        "execution_mode": execution_mode,
        "circuit_scope": "vision_encoder",
        "status": "fail" if first_divergence is not None else ("incomplete" if first_incomplete_capture else "pass"),
        "coverage_status": "incomplete" if first_incomplete_capture else "complete",
        "comparisons": comparisons,
        "first_divergence": first_divergence,
        "first_incomplete_capture": first_incomplete_capture,
        "last_passing_checkpoint": last_passing,
        "first_non_exact_checkpoint": first_non_exact,
        "unmatched_capture_rows": unmatched_capture_rows,
        "unresolved_contract_checkpoints": [],
        "ranking_divergence": None,
        "next_plan": {
            "status": (
                "first_observed_comparable_divergence"
                if first_divergence and incomplete_before_divergence
                else "first_divergence_attributed" if first_divergence
                else "incomplete_capture" if first_incomplete_capture
                else "complete"
            ),
            "first_failure": first_divergence["checkpoint_id"] if first_divergence else None,
            "passing_lower_bound": last_passing,
            "next_checkpoints": [],
        },
        "architecture_policy": xray.ARCHITECTURE_POLICY,
        "fix_progression": xray.XRAY_FIX_PROGRESSION,
    }


def _capture_args(args: argparse.Namespace, profile: dict[str, Any], report_path: Path) -> list[str]:
    active = _active_checkpoints(profile, args.layer)
    names = list(dict.fromkeys(str(mapping["capture_tensor"]) for _, mapping in active))
    command = [
        "--gguf", str(args.gguf),
        "--output-dir", str(args.output_dir / "capture"),
        "--threads", str(args.threads),
        "--ck-threads", str(args.ck_threads or args.threads),
        "--llama-dump-names", ",".join(names),
        "--llama-dump-layer", str(args.layer),
        "--ck-dump-layer", str(args.layer),
        "--ck-stop-layer", str(args.layer),
        "--quiet",
        "--report", str(report_path),
        "--llama-flash-attn", "disabled" if args.execution_mode == "strict" else "enabled",
    ]
    if args.execution_mode == "strict":
        command.append("--strict-parity")
    if args.image is not None:
        command.extend(["--image-path", str(args.image)])
    else:
        command.extend(["--image-mode", args.image_mode])
    if args.image_min_tokens is not None:
        command.extend(["--image-min-tokens", str(args.image_min_tokens)])
    if args.image_max_tokens is not None:
        command.extend(["--image-max-tokens", str(args.image_max_tokens)])
    return command


def _validate_oracle_execution(args: argparse.Namespace) -> dict[str, Any]:
    oracle_threads = int(args.threads)
    deterministic = oracle_threads == 1
    if not deterministic and not bool(args.allow_nondeterministic_oracle):
        raise RuntimeError(
            "exact llama.cpp X-ray capture requires --threads 1 because multi-threaded "
            "GGML reductions can change dump bytes between runs; pass "
            "--allow-nondeterministic-oracle only for explicitly non-exact diagnostics"
        )
    return {
        "threads": oracle_threads,
        "deterministic": deterministic,
        "nondeterministic_opt_in": bool(args.allow_nondeterministic_oracle),
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_provenance(
    args: argparse.Namespace,
    capture_report: dict[str, Any],
    run_id: str,
    phase: str,
) -> dict[str, Any]:
    """Run provenance following the repo convention (commits + *_sha256).

    Reuses the capture adapter's binary_provenance (engine / generated model /
    llama shim sha256). Any value that is not actually known stays null.
    """
    binary = capture_report.get("binary_provenance") or {}
    engine = binary.get("engine") or {}
    generated = binary.get("generated_model") or {}
    shim = binary.get("llama_shim") or {}
    llama_oracle = binary.get("llama_oracle") or {}
    flash = capture_report.get("llama_flash_attn")
    oracle_mode = llama_oracle.get("mode")
    if oracle_mode is None and isinstance(flash, str) and flash:
        oracle_mode = "flash-disabled" if flash == "disabled" else f"flash-{flash}"
    return {
        "run_id": run_id,
        "phase": phase,
        "model_sha256": _sha256_file(args.gguf) if args.gguf.is_file() else None,
        "subject": {
            "backend": "ck",
            "runtime_sha256": engine.get("sha256"),
            "generated_model_sha256": generated.get("sha256"),
            "compiler": None,
            "isa": None,
        },
        "oracle": {
            "backend": "llamacpp",
            "runtime_sha256": shim.get("sha256"),
            "commit": llama_oracle.get("commit"),
            "mode": oracle_mode,
            "fingerprint_sha256": llama_oracle.get("fingerprint_sha256"),
            "components": llama_oracle.get("components") or [],
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    oracle_execution = _validate_oracle_execution(args)
    profile = xray.load_json(args.profile)
    xray.validate(profile, xray.PROFILE_SCHEMA, "llama.cpp parity profile")
    if profile["backend"] != "llamacpp":
        raise RuntimeError(f"expected a llama.cpp profile, got {profile['backend']!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase = CAPTURE_PHASE
    run_id = str(args.run_id) if args.run_id else (
        "xray-qwen3vl-llamacpp-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    capture_report_path = args.output_dir / "capture_report.json"
    capture_rc = capture_adapter.main(_capture_args(args, profile, capture_report_path))
    if not capture_report_path.is_file():
        raise RuntimeError(
            f"llama.cpp capture adapter returned rc={capture_rc} without {capture_report_path}"
        )
    capture_report = xray.load_json(capture_report_path)
    report = normalize_capture_report(
        capture_report, profile, args.layer, args.execution_mode, phase
    )
    result = {
        "schema": "cke.xray_orchestration_report",
        "schema_version": 1,
        "backend": "llamacpp",
        "execution_mode": args.execution_mode,
        "circuit_scope": "vision_encoder",
        "run_id": run_id,
        "phase": phase,
        "provenance": _build_provenance(args, capture_report, run_id, phase),
        "oracle_execution": oracle_execution,
        "subject_execution": {"threads": int(args.ck_threads)},
        "status": report["status"],
        "rounds": [{
            "round": 0,
            "layer": args.layer,
            "capture_report": str(capture_report_path),
            "status": report["status"],
        }],
        "preflight": None,
        "final_report": report,
    }
    (args.output_dir / "xray_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--image-mode", choices=("gradient", "gray", "checker"), default="gradient")
    parser.add_argument("--image-min-tokens", type=int)
    parser.add_argument("--image-max-tokens", type=int)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="llama.cpp oracle threads; exact X-ray capture requires 1",
    )
    parser.add_argument("--ck-threads", type=int, default=20)
    parser.add_argument(
        "--allow-nondeterministic-oracle",
        action="store_true",
        help="permit a multi-threaded llama.cpp oracle for non-exact diagnostics",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("strict", "production"),
        default="strict",
        help="Run CK with strict reference semantics or the optimized production path.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier recorded in the report provenance; defaults to a UTC timestamped id.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("build/xray/qwen3vl_llamacpp"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run(args)
    divergence = (result.get("final_report") or {}).get("first_divergence") or {}
    print(f"status={result['status']} backend=llamacpp")
    if divergence:
        print(f"fail_at={divergence.get('checkpoint_id')}")
        print(f"class={divergence.get('classification')}")
    print(f"report={args.output_dir / 'xray_summary.json'}")
    return 1 if result["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
