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
    def test_v7_make_targets_include_kernel_map_and_backprop_matrix(self) -> None:
        self.assertIn("v7_kernel_map_contracts", nightly.MAKE_TARGETS)
        self.assertEqual(
            nightly.MAKE_TARGETS["v7_kernel_map_contracts"]["target"],
            "v7-kernel-map-contracts",
        )
        self.assertIn("v7_backprop_family_parity_fast", nightly.MAKE_TARGETS)
        self.assertEqual(
            nightly.MAKE_TARGETS["v7_backprop_family_parity_fast"]["target"],
            "v7-regression-backprop-fast",
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
            original = nightly.MAKE_TARGET_FAILURE_ARTIFACTS["v7-regression-backprop-fast"]
            nightly.MAKE_TARGET_FAILURE_ARTIFACTS["v7-regression-backprop-fast"] = report
            try:
                summary = nightly._summarize_make_failure_artifact(
                    "v7-regression-backprop-fast",
                    start_ts=0.0,
                )
            finally:
                nightly.MAKE_TARGET_FAILURE_ARTIFACTS["v7-regression-backprop-fast"] = original
            self.assertIn("families:3/4", summary)
            self.assertIn("qwen3:B2,C2", summary)


if __name__ == "__main__":
    unittest.main()
