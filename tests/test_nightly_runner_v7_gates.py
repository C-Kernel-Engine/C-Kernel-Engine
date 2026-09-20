#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "nightly_runner.py"


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nightly = _load_module("nightly_runner_v7_gate_test", SCRIPT)


class NightlyRunnerV7GateTests(unittest.TestCase):
    def test_v7_make_targets_include_kernel_map_and_training_matrix(self) -> None:
        self.assertIn("v7_kernel_map_contracts", nightly.MAKE_TARGETS)
        self.assertEqual(
            nightly.MAKE_TARGETS["v7_kernel_map_contracts"]["target"],
            "v7-kernel-map-contracts",
        )
        self.assertIn("v7_training_family_regression_full", nightly.MAKE_TARGETS)
        self.assertEqual(
            nightly.MAKE_TARGETS["v7_training_family_regression_full"]["target"],
            "regression-training-full",
        )
        self.assertEqual(
            nightly.MAKE_TARGETS["v8_training_certification_fp32"]["target"],
            "v8-training-certify-fp32",
        )
        self.assertEqual(
            nightly.MAKE_TARGETS["v8_training_workflow_fp32"]["target"],
            "v8-training-workflow-fp32",
        )
        self.assertEqual(
            nightly.MAKE_TARGETS["v8_training_matrix_gqa"]["target"],
            "v8-training-matrix-nightly",
        )
        self.assertEqual(
            nightly.MAKE_TARGET_FAILURE_ARTIFACTS["v8-training-matrix-nightly"],
            ROOT / "version" / "v8" / ".cache" / "reports" / "training_matrix_latest.json",
        )

    def test_kernel_map_failure_artifact_summary_uses_validator_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "kernel_map_validation_latest.json"
            report.write_text(
                json.dumps(
                    {
                        "summary": {
                            "status": "pass",
                            "kernel_maps": 112,
                            "passed": 112,
                            "failed": 0,
                            "warnings": 33,
                        },
                        "warnings": ["warning one", "warning two"],
                    }
                ),
                encoding="utf-8",
            )
            original = nightly.MAKE_TARGET_FAILURE_ARTIFACTS["v7-kernel-map-contracts"]
            nightly.MAKE_TARGET_FAILURE_ARTIFACTS["v7-kernel-map-contracts"] = report
            try:
                summary = nightly._summarize_make_failure_artifact(
                    "v7-kernel-map-contracts",
                    start_ts=0.0,
                )
            finally:
                nightly.MAKE_TARGET_FAILURE_ARTIFACTS["v7-kernel-map-contracts"] = original
            self.assertIn("passed:112/112", summary)
            self.assertIn("warnings:33", summary)
            self.assertIn("warning one", summary)

    def test_backprop_failure_artifact_summary_uses_family_failures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "v7_backprop_family_matrix_latest.json"
            report.write_text(
                json.dumps(
                    {
                        "summary": {
                            "passed": False,
                            "passed_families": 3,
                            "total_families": 4,
                        },
                        "results": [
                            {
                                "family": "qwen3",
                                "passed": False,
                                "rc": 1,
                                "summary": {"failed_stage_ids": ["B2", "C2"]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            original = nightly.MAKE_TARGET_FAILURE_ARTIFACTS["regression-training-full"]
            nightly.MAKE_TARGET_FAILURE_ARTIFACTS["regression-training-full"] = report
            try:
                summary = nightly._summarize_make_failure_artifact(
                    "regression-training-full",
                    start_ts=0.0,
                )
            finally:
                nightly.MAKE_TARGET_FAILURE_ARTIFACTS["regression-training-full"] = original
            self.assertIn("families:3/4", summary)
            self.assertIn("qwen3:B2,C2", summary)

    def test_v8_training_certification_failure_summary_names_failed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "training_certification_latest.json"
            report.write_text(
                json.dumps(
                    {
                        "status": "FAIL",
                        "checks": {"logits": {"passed": False}},
                        "negative_controls": {"stale_library": {"passed": True}},
                        "failures": ["logits"],
                    }
                ),
                encoding="utf-8",
            )
            original = nightly.MAKE_TARGET_FAILURE_ARTIFACTS["v8-training-certify-fp32"]
            nightly.MAKE_TARGET_FAILURE_ARTIFACTS["v8-training-certify-fp32"] = report
            try:
                summary = nightly._summarize_make_failure_artifact(
                    "v8-training-certify-fp32",
                    start_ts=0.0,
                )
            finally:
                nightly.MAKE_TARGET_FAILURE_ARTIFACTS["v8-training-certify-fp32"] = original
            self.assertIn("status=FAIL", summary)
            self.assertIn("failed_checks=logits", summary)

    def test_v8_training_matrix_failure_summary_names_case_and_reason(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "training_matrix_latest.json"
            report.write_text(json.dumps({"status": "FAIL", "cases": [
                {"profile": "6l_gqa", "passed": False, "failure_kind": "stale_report",
                 "identity_errors": ["matrix_identity.run_id"]}
            ]}), encoding="utf-8")
            original = nightly.MAKE_TARGET_FAILURE_ARTIFACTS["v8-training-matrix-nightly"]
            nightly.MAKE_TARGET_FAILURE_ARTIFACTS["v8-training-matrix-nightly"] = report
            try:
                summary = nightly._summarize_make_failure_artifact("v8-training-matrix-nightly", start_ts=0.0)
            finally:
                nightly.MAKE_TARGET_FAILURE_ARTIFACTS["v8-training-matrix-nightly"] = original
            self.assertIn("6l_gqa:stale_report:matrix_identity.run_id", summary)


if __name__ == "__main__":
    unittest.main()
