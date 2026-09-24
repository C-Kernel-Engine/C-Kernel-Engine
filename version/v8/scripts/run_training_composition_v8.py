#!/usr/bin/env python3
"""Certify two authored FP32 circuits through the v8 generated training loop."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "version/v7"))
sys.path.insert(0, str(ROOT / "version/v8/tools"))

import ckernel_engine as cke  # noqa: E402
from open_ir_visualizer_v8 import validate_training_experiment_manifest  # noqa: E402


PROFILES: dict[str, dict[str, int]] = {
    "2l_dense": {"layers": 2, "dim": 32, "hidden": 64, "heads": 4, "kv_heads": 4, "seq_len": 17},
    "2l_gqa_tail": {"layers": 2, "dim": 40, "hidden": 72, "heads": 4, "kv_heads": 2, "seq_len": 17},
}
CORPUS = ROOT / "version/v8/training/english_byte_v1.json"


def build_experiment(case: str, run_dir: Path, *, timeout: int = 1200):
    profile = PROFILES[case]
    model = cke.models.qwen3_tiny(
        vocab=256, dim=profile["dim"], layers=profile["layers"], hidden=profile["hidden"],
        heads=profile["heads"], kv_heads=profile["kv_heads"], context_len=profile["seq_len"],
        rope_theta=10_000.0, init="normal_0p02", dtype="float32", name=f"composition_{case}",
    )

    def run_command(command, cwd: Path) -> None:
        subprocess.run([str(part) for part in command], cwd=cwd, timeout=timeout, check=True)

    return cke.v8.compile(
        model, run_name=f"v8-composition-{case}", run_dir=run_dir,
        dataset=cke.v8.DatasetConfig(corpus=CORPUS, max_train_tokens=320, max_validation_tokens=64),
        tokenizer=cke.v8.TokenizerConfig(kind="byte", vocab_size=256),
        training=cke.v8.TrainingConfig(
            epochs=2, grad_accum=4,
            parameter_tolerance=1e-3, moment_tolerance=1e-4,
            gradient_tolerance=1e-3, loss_tolerance=1e-4, logits_tolerance=2e-3,
        ), command_runner=run_command,
    )


def validate_case(case: str, experiment, report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    config = report.get("configuration", {})
    for key, value in {
        "layers": 2, "d_model": PROFILES[case]["dim"], "hidden": PROFILES[case]["hidden"],
        "heads": PROFILES[case]["heads"], "kv_heads": PROFILES[case]["kv_heads"],
        "seq_len": 17, "epochs": 2, "grad_accum": 4, "dtype": "fp32",
    }.items():
        if config.get(key) != value:
            errors.append(f"configuration.{key}")
    checks = report.get("checks", {})
    for key, value in {"parameter": 1e-3, "moment": 1e-4, "gradient": 1e-3,
                       "loss": 1e-4, "logits": 2e-3}.items():
        if checks.get("pytorch_trajectory", {}).get("tolerances", {}).get(key) != value:
            errors.append(f"tolerances.{key}")
    for name in ("authored_semantic_graph", "pytorch_trajectory", "final_partial_update",
                 "fresh_process_resume", "inference_export", "negative_control_detection",
                 "training_ir_visualizer", "runtime_provenance"):
        if checks.get(name, {}).get("passed") is not True:
            errors.append(f"checks.{name}")
    trajectory = checks.get("pytorch_trajectory", {}).get("trajectory", [])
    if len(trajectory) != 10:
        errors.append("trajectory.update_count")
    for index, row in enumerate(trajectory, 1):
        if row.get("step") != index or row.get("passed") is not True:
            errors.append(f"trajectory.update_{index}")
        for kind, count in (("forward_logits", 1), ("gradients", 23),
                            ("weights", 23), ("optimizer_moments", 46)):
            comparison = row.get(kind, {})
            if comparison.get("passed") is not True or comparison.get("tensor_count") != count:
                errors.append(f"trajectory.update_{index}.{kind}")
    if checks.get("final_partial_update", {}).get("present") is not True:
        errors.append("final_partial_update.missing")
    if checks.get("negative_control_detection", {}).get("gradient_routing", {}).get("passed") is not True:
        errors.append("gradient_routing_control")
    manifest_path = experiment.manifest_path
    if not manifest_path.is_file():
        errors.append("manifest.missing")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = validate_training_experiment_manifest(manifest, manifest_path)
        if validation["status"] != "MATCHED" or manifest.get("verdict", {}).get("passed") is not True:
            errors.append("manifest.not_matched")
    return errors


def _write_report(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=tuple(PROFILES))
    parser.add_argument("--run-root", type=Path, default=ROOT / "version/v8/.cache/training_composition")
    parser.add_argument("--json-out", type=Path, default=ROOT / "version/v8/.cache/reports/training_composition_latest.json")
    parser.add_argument("--case-timeout", type=int, default=1200)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.case_timeout <= 0:
        parser.error("--case-timeout must be positive")
    run_id = str(uuid.uuid4())
    cases = args.case or list(PROFILES)
    result: dict[str, Any] = {"schema": "cke.v8.training_composition_matrix.v1", "status": "FAIL",
                              "run_id": run_id, "cases": [], "preflight_only": args.preflight_only}
    for case in cases:
        run_dir = args.run_root.resolve() / run_id / case
        row: dict[str, Any] = {"case": case, "profile": PROFILES[case], "run_dir": str(run_dir), "status": "FAIL"}
        try:
            experiment = build_experiment(case, run_dir, timeout=args.case_timeout)
            preflight = experiment.preflight()
            row["preflight"] = str(experiment.preflight_path)
            if preflight.get("can_launch_generated_workflow") is not True:
                raise RuntimeError("candidate backward capability inventory is incomplete")
            if args.preflight_only:
                row["status"] = "PREFLIGHT_ONLY"
            else:
                report = experiment.run()
                row["workflow_report"] = str(experiment.report_path)
                row["manifest"] = str(experiment.manifest_path)
                row["errors"] = validate_case(case, experiment, report)
                row["status"] = "PASS" if report.get("status") == "PASS" and not row["errors"] else "FAIL"
                row["update_count"] = len(report.get("checks", {}).get("pytorch_trajectory", {}).get("trajectory", []))
                row["first_failed_update"] = report.get("checks", {}).get("pytorch_trajectory", {}).get("first_failed_update")
        except Exception as exc:
            row["errors"] = [f"{type(exc).__name__}: {exc}"]
        result["cases"].append(row)
        _write_report(args.json_out.resolve(), result)
    result["status"] = "PREFLIGHT_ONLY" if args.preflight_only and all(row["status"] == "PREFLIGHT_ONLY" for row in result["cases"]) else "PASS" if all(row["status"] == "PASS" for row in result["cases"]) else "FAIL"
    _write_report(args.json_out.resolve(), result)
    print(json.dumps({"status": result["status"], "report": str(args.json_out.resolve()),
                      "cases": [{"case": row["case"], "status": row["status"]} for row in result["cases"]]}))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
