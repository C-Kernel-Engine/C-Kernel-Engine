#!/usr/bin/env python3
"""Normalize backend layer-output dumps into a canonical X-Ray drift report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _final_row(path: Path, token_count: int, hidden_size: int) -> tuple[np.ndarray, str]:
    values = np.fromfile(path, dtype=np.float32)
    full_size = token_count * hidden_size
    if values.size == full_size:
        return values.reshape(token_count, hidden_size)[-1], "full_sequence_final_row"
    if values.size == hidden_size:
        return values, "final_row_only"
    raise ValueError(
        f"{path}: expected {hidden_size} or {full_size} fp32 values, got {values.size}"
    )


def _metrics(subject: np.ndarray, oracle: np.ndarray) -> dict[str, Any]:
    if subject.shape != oracle.shape:
        raise ValueError(f"shape mismatch: {subject.shape} != {oracle.shape}")
    subject64 = subject.astype(np.float64, copy=False)
    oracle64 = oracle.astype(np.float64, copy=False)
    delta = subject64 - oracle64
    abs_delta = np.abs(delta)
    total = int(delta.size)
    exact = int(np.count_nonzero(subject == oracle))
    denominator = float(np.linalg.norm(subject64) * np.linalg.norm(oracle64))
    rmse = float(np.sqrt(np.mean(delta * delta))) if total else 0.0
    oracle_rms = float(np.sqrt(np.mean(oracle64 * oracle64))) if total else 0.0
    return {
        "cosine": float(np.dot(subject64, oracle64) / denominator) if denominator else 1.0,
        "rmse": rmse,
        "relative_rmse": rmse / oracle_rms if oracle_rms else (0.0 if rmse == 0.0 else None),
        "mean_abs": float(np.mean(abs_delta)) if total else 0.0,
        "max_abs": float(np.max(abs_delta)) if total else 0.0,
        "exact_elements": exact,
        "total_elements": total,
        "exact_ratio": exact / total if total else 1.0,
        "byte_exact": bool(np.array_equal(subject, oracle)),
        "finite": bool(np.all(np.isfinite(subject)) and np.all(np.isfinite(oracle))),
    }


def build_report(
    *,
    subject_dir: Path,
    oracle_dir: Path,
    parity_report: dict[str, Any],
    production_report: dict[str, Any] | None,
    model: str,
    layers: int,
    token_count: int,
    hidden_size: int,
    logical_token: int,
    subject_pattern: str,
    oracle_pattern: str,
    phase: str = "prefill",
    comparison_mode: str = "full_replay",
    ordered_capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in {"prefill", "decode"}:
        raise ValueError(f"unsupported phase: {phase}")
    if comparison_mode not in {"full_replay", "production_decode"}:
        raise ValueError(f"unsupported comparison mode: {comparison_mode}")
    comparison_label = (
        "CK persistent decode vs llama.cpp production decode"
        if comparison_mode == "production_decode"
        else "CK full replay vs llama.cpp diagnostic full replay"
    )
    comparisons: list[dict[str, Any]] = []
    selected_layers: list[int] = []
    source_artifacts: list[dict[str, Any]] = []
    checkpoints = None
    if ordered_capture is not None:
        checkpoints = ordered_capture["checkpoints"]
        recorded_phase = ordered_capture["phase"]
        if (ordered_capture["position"] != logical_token or len(checkpoints) != layers
                or recorded_phase not in {"prefill", "append"}
                or (phase == "decode" and recorded_phase != "append")
                or sorted(check["state_slot"] for check in checkpoints) != list(range(layers))
                or [check["sequence_index"] for check in checkpoints] != list(range(layers))
                or len({check["id"] for check in checkpoints}) != layers):
            raise ValueError("ordered layer capture does not match the requested sweep")

    for layer in range(layers):
        check = checkpoints[layer] if checkpoints is not None else None
        oracle_slot = check["state_slot"] if check else layer
        subject_filename = (check["filename"] if check else
                            subject_pattern.format(layer=layer, token=logical_token))
        if check and Path(subject_filename).name != subject_filename:
            raise ValueError("ordered capture filename must name a local tensor")
        subject_path = subject_dir / subject_filename
        oracle_path = oracle_dir / oracle_pattern.format(
            layer=oracle_slot, token=logical_token
        )
        if not subject_path.is_file() or not oracle_path.is_file():
            missing = [
                str(path)
                for path in (subject_path, oracle_path)
                if not path.is_file()
            ]
            raise FileNotFoundError("missing layer-output capture: " + ", ".join(missing))
        subject, subject_extent = _final_row(subject_path, token_count, hidden_size)
        oracle, oracle_extent = _final_row(oracle_path, token_count, hidden_size)
        metrics = _metrics(subject, oracle)
        checkpoint_id = check["id"] if check else f"decoder.layer.{layer}.output.final_row"
        comparisons.append({
            "sequence_index": layer,
            "checkpoint_id": checkpoint_id,
            "op_idx": None,
            "status": "exact" if metrics["byte_exact"] else "different",
            "classification": (
                "EXACT"
                if metrics["byte_exact"]
                else "OBSERVED_LAYER_OUTPUT_DIVERGENCE"
            ),
            "metrics": metrics,
            "resolved_execution": {
                "phase": phase,
                "layer": check["weight_layer"] if check else layer,
                **({"state_slot": oracle_slot, "loop_pass": check["loop_pass"]} if check else {}),
                "function": None,
                "kernel_id": None,
                "resolved_contract_id": None,
                "storage_dtype": "fp32",
            },
            "capture_extent": {
                "subject": subject_extent,
                "oracle": oracle_extent,
                "selected_row": token_count - 1,
            },
            "source": {
                "subject_path": str(subject_path),
                "subject_sha256": _sha256(subject_path),
                "oracle_path": str(oracle_path),
                "oracle_sha256": _sha256(oracle_path),
            },
        })
        selected_layers.append(layer)

    first_non_exact = next(
        (row for row in comparisons if not row["metrics"]["byte_exact"]),
        None,
    )
    parity_divergence = parity_report.get("first_divergence")
    production_divergence = (
        production_report.get("first_divergence") if production_report else None
    )
    runtime = parity_report.get("ck_runtime") or {}
    gguf_path = Path(str(parity_report.get("gguf_path", ""))).expanduser()
    model_sha256 = _sha256(gguf_path) if gguf_path.is_file() else None

    for path in (subject_dir / "index.json", oracle_dir / "index.json"):
        if path.is_file():
            source_artifacts.append({
                "path": str(path),
                "sha256": _sha256(path),
            })

    status = "pass" if first_non_exact is None and not parity_divergence else "fail"
    return {
        "schema": "cke.xray_numerical_report",
        "schema_version": 1,
        "backend": "llamacpp",
        "status": status,
        "circuit_scope": "decoder",
        "run_id": f"xray-{model}-layer-output-final-row-{logical_token}",
        "phase": phase,
        "run": {
            "id": f"xray-{model}-layer-output-final-row-{logical_token}",
            "phase": phase,
            "logical_token": logical_token,
            "token_count": token_count,
            "comparison_mode": comparison_mode,
        },
        "model": model,
        "capture_scope": {
            "phase": phase,
            "checkpoint_granularity": "layer_output_final_causal_row",
            "covered_layer_count": len(selected_layers),
            "model_layer_count": layers,
            "selected_layers": selected_layers,
            "comparison": comparison_label,
            "logical_token": logical_token,
            "token_count": token_count,
            "hidden_size": hidden_size,
        },
        "provenance": {
            "threads": parity_report.get("threads"),
            "model_sha256": model_sha256,
            "subject": {
                "backend": "cke",
                "runtime_path": runtime.get("path"),
                "runtime_sha256": runtime.get("sha256"),
            },
            "oracle": {
                "backend": "llama.cpp",
                "mode": (
                    "production_graph_tensor_dump"
                    if comparison_mode == "production_decode"
                    else "diagnostic_tensor_dump"
                ),
                "flash_inputs": (parity_report.get("llama_capture") or {}).get("flash_inputs"),
                "attention_mode": (parity_report.get("llama_capture") or {}).get("attention_mode"),
            },
            "sources": source_artifacts,
        },
        "production_baseline": {
            "source_report": (
                str(production_report.get("_source_path"))
                if production_report and production_report.get("_source_path")
                else None
            ),
            "first_divergence": production_divergence,
            "note": (
                "Production parity was established without tensor dumps. "
                "The layer sweep is diagnostic evidence and does not replace that baseline."
            ),
        },
        "diagnostic_ranking": {
            "first_divergence": parity_divergence,
            "note": (
                "Production layer outputs are compared at the recorded persistent "
                "decode boundary; this sweep localizes drift but does not by itself "
                "identify the first offending primitive."
                if comparison_mode == "production_decode"
                else
                "This sweep compares replayed outputs. Identical-input operation replay "
                "is needed to attribute a difference to the producer."
            ),
        },
        "first_non_exact_stop": (
            first_non_exact["sequence_index"] if first_non_exact else None
        ),
        "first_non_exact_checkpoint": (
            {
                "checkpoint_id": first_non_exact["checkpoint_id"],
                "classification": first_non_exact["classification"],
            }
            if first_non_exact
            else None
        ),
        "first_divergence": (
            {
                "checkpoint_id": first_non_exact["checkpoint_id"],
                "classification": "OBSERVED_LAYER_OUTPUT_DIVERGENCE",
                "note": (
                    "This is the first observed layer-output checkpoint, not proof that "
                    "the layer-output producer contains the first offending primitive."
                ),
            }
            if first_non_exact
            else None
        ),
        "comparisons": comparisons,
        "limitations": [
            "Only the final causal row is compared consistently across all layers.",
            "No call-IR operation identity is inferred from filenames.",
            "Layer-output drift localizes accumulation but does not identify a leaf kernel.",
            (
                "Production decode capture preserves the recorded llama.cpp graph mode, "
                "but callback observation is still disclosed in source provenance."
                if comparison_mode == "production_decode"
                else
                "Diagnostic llama.cpp tensor capture may use a different execution mode than production."
            ),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--ordered-capture", action="store_true",
                        help="use the subject index to align repeated layer occurrences")
    parser.add_argument("--production-report", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--token-count", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--logical-token", type=int, required=True)
    parser.add_argument(
        "--phase",
        choices=("prefill", "decode"),
        default="prefill",
        help="Execution phase represented by both captures.",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=("full_replay", "production_decode"),
        default="full_replay",
        help="Controls truthful comparison labels and oracle provenance.",
    )
    parser.add_argument(
        "--subject-pattern",
        default="tok_0000_layer_{layer:03d}_layer_out.f32",
    )
    parser.add_argument(
        "--oracle-pattern",
        default="l_out-{layer}-token-{token:06d}-occ-000.bin",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    parity = _load_json(args.parity_report.resolve())
    production = None
    if args.production_report:
        production = _load_json(args.production_report.resolve())
        production["_source_path"] = str(args.production_report.resolve())
    report = build_report(
        subject_dir=args.subject_dir.resolve(),
        oracle_dir=args.oracle_dir.resolve(),
        parity_report=parity,
        production_report=production,
        model=args.model,
        layers=args.layers,
        token_count=args.token_count,
        hidden_size=args.hidden_size,
        logical_token=args.logical_token,
        subject_pattern=args.subject_pattern,
        oracle_pattern=args.oracle_pattern,
        phase=str(args.phase),
        comparison_mode=str(args.comparison_mode),
        ordered_capture=_load_json(args.subject_dir / "index.json") if args.ordered_capture else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "xray_summary.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report={output}")
    print(
        f"coverage={report['capture_scope']['covered_layer_count']}/"
        f"{report['capture_scope']['model_layer_count']} "
        f"first_non_exact={report['first_non_exact_stop']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
