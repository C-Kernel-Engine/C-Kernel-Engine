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
    def test_nightly_requires_current_run_attempt_and_commit_for_pass(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "nightly.yml").read_text(encoding="utf-8")
        self.assertIn('github_identity.get("run_id")', workflow)
        self.assertIn('github_identity.get("run_attempt")', workflow)
        self.assertIn('execution.get("git_commit")', workflow)
        self.assertIn('"historical_evidence": report_payload', workflow)
        self.assertIn('"current_result": training_cert_result', workflow)
        self.assertIn('"status": "fail"', workflow)
        self.assertIn('training_cert_payload["passed"] = False', workflow)

    def test_alternate_engine_control_has_distinct_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            engine = base / "libckernel_engine.so"
            engine.write_bytes(b"ELF fixture")
            alternate = cert._alternate_engine_control(engine, base / "run")
            self.assertEqual(alternate.name, engine.name)
            self.assertNotEqual(cert._sha256(alternate), cert._sha256(engine))
            self.assertTrue(alternate.read_bytes().startswith(engine.read_bytes()))

    def test_tensor_comparison_rejects_corruption_and_nonfinite_values(self) -> None:
        reference = np.asarray([0.25, -0.5], dtype=np.float32)
        self.assertTrue(cert._compare_tensor(reference.copy(), reference, 1e-6, 1e-6)["passed"])
        self.assertFalse(cert._compare_tensor(reference + 1.0, reference, 1e-6, 1e-6)["passed"])
        nonfinite = reference.copy()
        nonfinite[0] = np.nan
        result = cert._compare_tensor(nonfinite, reference, 1e-6, 1e-6)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "non_finite")

        mixed_reference = np.asarray([1000.0, 0.0], dtype=np.float32)
        mixed_actual = np.asarray([1000.0, 1.0], dtype=np.float32)
        mixed = cert._compare_tensor(mixed_actual, mixed_reference, 3e-4, 3e-3)
        self.assertFalse(mixed["passed"])
        self.assertEqual(mixed["first_violating_index"], 1)
        self.assertEqual(mixed["worst_index"], 1)

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
            (run_dir / "ir1_train_forward.json").write_text(
                json.dumps(
                    {
                        "tensors": {
                            "weight.weight.a": {
                                "kind": "weight",
                                "requires_grad": True,
                            }
                        }
                    }
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
            expected = [{"name": "weight.a", "manifest_name": "weight.a", "shape": [2, 3], "numel": 6}]
            inventory = cert._validate_parameter_inventory(summary, run_dir, expected)
            self.assertEqual(inventory[0]["shape"], [2, 3])
            np.testing.assert_array_equal(cert._serialized_weight_snapshot(run_dir, summary), serialized)

            duplicate = dict(summary)
            duplicate["parameter_gradient_order"] = ["weight.a", "weight.a"]
            duplicate["parameter_gradient_numel"] = [6, 6]
            with self.assertRaisesRegex(ValueError, "duplicate"):
                cert._validate_parameter_inventory(duplicate, run_dir, expected)

            missing = dict(summary)
            missing["parameter_gradient_order"] = ["weight.missing"]
            with self.assertRaisesRegex(ValueError, "mismatch"):
                cert._validate_parameter_inventory(missing, run_dir, expected)

            omitted_everywhere = {
                "parameter_gradient_order": ["weight.a"],
                "parameter_gradient_numel": [6],
                "tensor_slots": [{"name": "grad.weight.weight.a"}],
            }
            independently_expected = [
                *expected,
                {"name": "weight.b", "manifest_name": "weight.b", "shape": [1], "numel": 1},
            ]
            (run_dir / "ir1_train_forward.json").write_text(
                json.dumps(
                    {
                        "tensors": {
                            "weight.weight.a": {"kind": "weight", "requires_grad": True},
                            "weight.weight.b": {"kind": "weight", "requires_grad": True},
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing_expected=.*weight.b"):
                cert._validate_parameter_inventory(omitted_everywhere, run_dir, independently_expected)

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
            self.assertEqual(published["coverage"]["certified"], [])
            self.assertTrue(published["coverage"]["requested"])
            self.assertIn("tiny model initialization failed", published["failures"])
            self.assertIn("completed_at", published)


if __name__ == "__main__":
    unittest.main()
