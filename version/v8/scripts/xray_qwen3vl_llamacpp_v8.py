#!/usr/bin/env python3
"""Qwen3-VL GGUF adapter for the unified vision X-ray surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import activation_parity_qwen3vl_mmproj_v8 as capture_adapter
import xray_numerical_parity_v8 as xray


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE = SCRIPT_DIR.parent / "parity_profiles" / "qwen3vl_llamacpp_q8_v1.json"


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


def normalize_capture_report(
    report: dict[str, Any], profile: dict[str, Any], layer: int,
    execution_mode: str = "strict",
) -> dict[str, Any]:
    indexed = _legacy_result_index(report)
    comparisons: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    first_non_exact: dict[str, Any] | None = None
    last_passing: str | None = None
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
            }
        else:
            legacy_status = str(row.get("status", "ERROR")).upper()
            status = "pass" if legacy_status == "PASS" else "fail"
            if status == "pass":
                classification = "MATCH"
            elif bool(row.get("has_nan")) or bool(row.get("has_inf")):
                classification = "NONFINITE_OUTPUT"
            else:
                classification = "KERNEL_IMPLEMENTATION_DIVERGENCE"
            comparison = {
                "checkpoint_id": checkpoint_id,
                "status": status,
                "classification": classification,
                "legacy_tensor": result_name,
                "metrics": {
                    key: row[key]
                    for key in (
                        "max_abs_diff",
                        "mean_abs_diff",
                        "max_rel_err",
                        "mean_rel_err",
                        "diverge_idx",
                    )
                    if key in row
                },
            }
        comparisons.append(comparison)
        if comparison["status"] == "pass":
            metrics = comparison.get("metrics") or {}
            if first_non_exact is None and float(metrics.get("max_abs_diff", 0.0) or 0.0) != 0.0:
                first_non_exact = {
                    "checkpoint_id": checkpoint_id,
                    "legacy_tensor": result_name,
                    "metrics": metrics,
                }
            last_passing = checkpoint_id
            continue
        first_divergence = comparison
        break

    if first_divergence is not None:
        if first_non_exact is not None:
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
    return {
        "schema": "cke.xray_numerical_report",
        "schema_version": 1,
        "subject_backend": "ck",
        "oracle_backend": "llamacpp",
        "execution_mode": execution_mode,
        "status": "fail" if first_divergence is not None else "pass",
        "comparisons": comparisons,
        "first_divergence": first_divergence,
        "last_passing_checkpoint": last_passing,
        "first_non_exact_checkpoint": first_non_exact,
        "unresolved_contract_checkpoints": [],
        "ranking_divergence": None,
        "next_plan": {
            "status": "first_divergence_attributed" if first_divergence else "complete",
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    oracle_execution = _validate_oracle_execution(args)
    profile = xray.load_json(args.profile)
    xray.validate(profile, xray.PROFILE_SCHEMA, "llama.cpp parity profile")
    if profile["backend"] != "llamacpp":
        raise RuntimeError(f"expected a llama.cpp profile, got {profile['backend']!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture_report_path = args.output_dir / "capture_report.json"
    capture_rc = capture_adapter.main(_capture_args(args, profile, capture_report_path))
    if not capture_report_path.is_file():
        raise RuntimeError(
            f"llama.cpp capture adapter returned rc={capture_rc} without {capture_report_path}"
        )
    report = normalize_capture_report(
        xray.load_json(capture_report_path), profile, args.layer, args.execution_mode
    )
    result = {
        "schema": "cke.xray_orchestration_report",
        "schema_version": 1,
        "backend": "llamacpp",
        "execution_mode": args.execution_mode,
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
