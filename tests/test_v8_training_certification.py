#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "run_training_certification_v8.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_training_certification_v8_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cert = _load_module()


class V8TrainingCertificationTests(unittest.TestCase):
    def test_tensor_comparison_rejects_corruption_and_nonfinite_values(self) -> None:
        reference = np.asarray([0.25, -0.5], dtype=np.float32)
        self.assertTrue(cert._compare_tensor(reference.copy(), reference, 1e-6, 1e-6)["passed"])
        self.assertFalse(cert._compare_tensor(reference + 1.0, reference, 1e-6, 1e-6)["passed"])
        nonfinite = reference.copy()
        nonfinite[0] = np.nan
        result = cert._compare_tensor(nonfinite, reference, 1e-6, 1e-6)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "non_finite")

    def test_parameter_inventory_requires_exact_named_shape_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            serialized = np.arange(6, dtype=np.float32)
            (run_dir / "weights.bump").write_bytes(serialized.tobytes())
            (run_dir / "weights_manifest.json").write_text(
                json.dumps(
                    {"entries": [{"name": "weight.a", "shape": [2, 3], "dtype": "fp32", "offset": 0}]}
                ),
                encoding="utf-8",
            )
            summary = {
                "parameter_gradient_order": ["weight.a"],
                "parameter_gradient_numel": [6],
                "tensor_slots": [
                    {"name": "weight.weight.a", "section": "weights", "offset": 0, "numel": 6},
                    {"name": "grad.weight.weight.a"},
                ],
            }
            inventory = cert._validate_parameter_inventory(summary, run_dir)
            self.assertEqual(inventory[0]["shape"], [2, 3])
            np.testing.assert_array_equal(cert._serialized_weight_snapshot(run_dir, summary), serialized)

            duplicate = dict(summary)
            duplicate["parameter_gradient_order"] = ["weight.a", "weight.a"]
            duplicate["parameter_gradient_numel"] = [6, 6]
            with self.assertRaisesRegex(ValueError, "duplicate"):
                cert._validate_parameter_inventory(duplicate, run_dir)

            missing = dict(summary)
            missing["parameter_gradient_order"] = ["weight.missing"]
            with self.assertRaisesRegex(ValueError, "mismatch"):
                cert._validate_parameter_inventory(missing, run_dir)

    def test_initialization_failure_still_publishes_complete_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            args = argparse.Namespace(
                run_dir=base / "run",
                report=base / "report.json",
                vocab=32,
                d_model=16,
                hidden=32,
                seq_len=4,
                seed=7,
            )
            failed = subprocess.CompletedProcess(args=[], returncode=2, stdout="injected init failure")
            with mock.patch.object(cert, "_run", return_value=failed):
                report = cert.run(args)
            published = json.loads(args.report.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(published["status"], "FAIL")
            self.assertFalse(published["passed"])
            self.assertIn("tiny model initialization failed", published["failures"])
            self.assertIn("completed_at", published)


if __name__ == "__main__":
    unittest.main()
