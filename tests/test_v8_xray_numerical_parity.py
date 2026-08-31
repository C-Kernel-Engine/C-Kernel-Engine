#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


xray = load_module("xray_numerical_parity_v8", SCRIPTS / "xray_numerical_parity_v8.py")
builder = load_module("build_xray_checkpoint_manifest_v8", SCRIPTS / "build_xray_checkpoint_manifest_v8.py")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class XRayNumericalParityTests(unittest.TestCase):
    def test_bf16_ulp_threshold_tracks_representable_steps(self):
        threshold = {
            "cosine_min": 0.99,
            "rmse_max": 2.0,
            "relative_rmse_max": 1.0,
            "max_abs_max": 0.25,
            "max_bf16_ulp_max": 1,
            "max_abs_safety_max": 16.0,
            "finite_required": True,
        }
        one_ulp = xray._metrics(
            np.array([-71.0], dtype=np.float32),
            np.array([-70.5], dtype=np.float32),
            ["channel"],
            bf16_abs_floor=0.25,
        )
        two_ulp = xray._metrics(
            np.array([-71.0], dtype=np.float32),
            np.array([-70.0], dtype=np.float32),
            ["channel"],
            bf16_abs_floor=0.25,
        )
        self.assertEqual(one_ulp["max_bf16_ulp_over_abs_floor"], 1)
        self.assertEqual(xray._metric_status(one_ulp, threshold), ("pass", []))
        self.assertEqual(two_ulp["max_bf16_ulp_over_abs_floor"], 2)
        self.assertEqual(xray._metric_status(two_ulp, threshold), ("fail", ["max_bf16_ulp"]))

    def test_bf16_ulp_threshold_ignores_subfloor_sign_crossing(self):
        metrics = xray._metrics(
            np.array([-0.00123], dtype=np.float32),
            np.array([0.00123], dtype=np.float32),
            ["channel"],
            bf16_abs_floor=0.25,
        )
        self.assertLess(metrics["max_abs"], 0.25)
        self.assertEqual(metrics["max_bf16_ulp_over_abs_floor"], 0)

    def test_bf16_ulp_threshold_includes_error_equal_to_floor(self):
        threshold = {
            "cosine_min": -1.0,
            "rmse_max": 1.0,
            "relative_rmse_max": 1.0,
            "max_abs_max": 0.25,
            "max_bf16_ulp_max": 2,
            "max_abs_safety_max": 16.0,
            "finite_required": True,
        }
        metrics = xray._metrics(
            np.array([-0.2578125], dtype=np.float32),
            np.array([-0.5078125], dtype=np.float32),
            ["channel"],
            bf16_abs_floor=0.25,
        )
        self.assertEqual(metrics["max_abs"], 0.25)
        self.assertGreater(metrics["max_bf16_ulp_over_abs_floor"], 2)
        self.assertEqual(
            xray._metric_status(metrics, threshold),
            ("fail", ["max_bf16_ulp"]),
        )

    def test_bf16_ulp_threshold_retains_absolute_safety_ceiling(self):
        threshold = {
            "cosine_min": -1.0,
            "rmse_max": 100.0,
            "relative_rmse_max": 100.0,
            "max_abs_max": 0.25,
            "max_bf16_ulp_max": 128,
            "max_abs_safety_max": 16.0,
            "finite_required": True,
        }
        metrics = xray._metrics(
            np.array([256.0], dtype=np.float32),
            np.array([274.0], dtype=np.float32),
            ["channel"],
            bf16_abs_floor=0.25,
        )
        self.assertEqual(
            xray._metric_status(metrics, threshold),
            ("fail", ["max_abs_safety"]),
        )

    def test_nightly_xray_gate_runs_capture_neutrality_contracts(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        target = makefile.split("test-bf16-xray:", 1)[1].split(
            "\nxray-vision-parity:", 1
        )[0]

        self.assertIn(
            "version/v8/scripts/compare_multitoken_logits_v8.py",
            target,
        )
        self.assertIn("tests/test_compare_multitoken_logits_v8.py", target)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="xray_v8_")
        self.root = Path(self.temp.name)
        self.profile = {
            "schema": "cke.parity_profile", "schema_version": 1, "name": "test", "backend": "pytorch",
            "contract_schema_version": 1,
            "required_match_fields": ["checkpoint_id", "producer", "logical_layout", "axis_names", "resolved_contract_id", "kernel_id", "function"],
            "observed_storage": {"default": "fp32", "checkpoints": {}},
            "dtype_thresholds": {
                "fp32": {"cosine_min": 0.99999, "rmse_max": 1e-4, "relative_rmse_max": 1e-4, "max_abs_max": 1e-3, "finite_required": True},
                "bf16": {"cosine_min": 0.999, "rmse_max": 0.02, "relative_rmse_max": 0.02, "max_abs_max": 0.25, "finite_required": True},
            },
            "checkpoint_order": ["vision.layer.0.output", "vision.layer.8.output"],
            "interval_expansions": {}, "backend_mappings": {},
        }

    def tearDown(self):
        self.temp.cleanup()

    def entry(self, checkpoint: str, path: Path, *, storage="fp32", producer="block", physical_axes=None):
        return {
            "checkpoint_id": checkpoint, "producer": producer, "phase": "prefill", "layer": 0,
            "tensor_path": str(path), "storage_dtype": storage, "exported_dtype": "fp32",
            "logical_shape": [2, 3], "physical_shape": [2, 3], "logical_layout": "token_major",
            "axis_names": ["token", "channel"], "physical_axis_names": physical_axes or ["token", "channel"],
            "resolved_contract_id": "contract.test", "kernel_id": "kernel.test", "function": "kernel_test",
            "sha256": digest(path),
        }

    def manifest(self, backend: str, entries):
        return {"schema": "cke.checkpoint_manifest", "schema_version": 1, "backend": backend,
                "run": {"model": "fixture", "phase": "prefill", "source": "unit"}, "checkpoints": entries}

    def test_matching_tensors_pass_and_report_worst_coordinate(self):
        a = self.root / "a.f32"; b = self.root / "b.f32"
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        values.tofile(a); values.tofile(b)
        left = self.manifest("ck", [self.entry("vision.layer.0.output", a)])
        right = self.manifest("pytorch", [self.entry("vision.layer.0.output", b)])
        result = xray.compare_manifests(left, right, self.profile, checkpoint_order=["vision.layer.0.output"])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["comparisons"][0]["classification"], "MATCH")
        self.assertTrue(result["comparisons"][0]["metrics"]["byte_exact"])
        self.assertIsNone(result["first_non_exact_checkpoint"])

    def test_reports_first_non_exact_checkpoint_without_mislabeling_material_failure(self):
        a = self.root / "a.f32"; b = self.root / "b.f32"
        got = np.ones((2, 3), np.float32)
        ref = got.copy()
        got[0, 1] += np.float32(1.0e-5)
        got.tofile(a); ref.tofile(b)
        left = self.manifest("ck", [self.entry("vision.layer.0.output", a)])
        right = self.manifest("pytorch", [self.entry("vision.layer.0.output", b)])

        result = xray.compare_manifests(
            left, right, self.profile, checkpoint_order=["vision.layer.0.output"]
        )

        self.assertEqual(result["status"], "pass")
        self.assertIsNone(result["first_divergence"])
        non_exact = result["first_non_exact_checkpoint"]
        self.assertEqual(non_exact["checkpoint_id"], "vision.layer.0.output")
        self.assertEqual(non_exact["classification"], "NON_BYTE_EXACT")
        self.assertFalse(non_exact["metrics"]["byte_exact"])
        self.assertEqual(non_exact["metrics"]["exact_elements"], 5)

    def test_reports_accumulated_drift_and_resolved_provider_boundary(self):
        paths = []
        reference = np.ones((2, 3), np.float32)
        entries_left = []
        entries_right = []
        checkpoints = ["vision.layer.0.output", "vision.layer.1.output", "vision.layer.2.output"]
        deltas = [1.0e-5, 4.0e-5, 1.0e-2]
        for index, (checkpoint, delta) in enumerate(zip(checkpoints, deltas)):
            got_path = self.root / f"got-{index}.f32"
            ref_path = self.root / f"ref-{index}.f32"
            got = reference.copy()
            got[0, 0] += np.float32(delta)
            got.tofile(got_path)
            reference.tofile(ref_path)
            left = self.entry(checkpoint, got_path)
            left["kernel_id"] = f"kernel.{index}"
            left["function"] = f"kernel_{index}"
            right = self.entry(checkpoint, ref_path)
            right["kernel_id"] = left["kernel_id"]
            right["function"] = left["function"]
            entries_left.append(left)
            entries_right.append(right)
            paths.extend([got_path, ref_path])

        result = xray.compare_manifests(
            self.manifest("ck", entries_left),
            self.manifest("pytorch", entries_right),
            self.profile,
            checkpoint_order=checkpoints,
        )

        progression = result["drift_progression"]
        self.assertEqual(progression["policy"], "observational_no_additional_tolerance")
        self.assertEqual(progression["first_non_exact"]["checkpoint_id"], checkpoints[0])
        boundary = progression["largest_amplification_boundary"]
        self.assertEqual(boundary["from_checkpoint_id"], checkpoints[1])
        self.assertEqual(boundary["to_checkpoint_id"], checkpoints[2])
        self.assertGreater(boundary["relative_rmse_ratio"], 100.0)
        self.assertEqual(boundary["resolved_execution"]["kernel_id"], "kernel.2")
        self.assertEqual(boundary["resolved_execution"]["function"], "kernel_2")

    def test_named_axes_are_canonicalized_before_comparison(self):
        logical = np.arange(6, dtype=np.float32).reshape(2, 3)
        a = self.root / "a.f32"; b = self.root / "b.f32"
        logical.T.tofile(a); logical.tofile(b)
        left_entry = self.entry("vision.layer.0.output", a, physical_axes=["channel", "token"])
        left_entry["physical_shape"] = [3, 2]
        left = self.manifest("ck", [left_entry])
        right = self.manifest("pytorch", [self.entry("vision.layer.0.output", b)])
        self.assertEqual(xray.compare_manifests(left, right, self.profile, checkpoint_order=["vision.layer.0.output"])["status"], "pass")

    def test_storage_mismatch_is_classified_before_value_comparison(self):
        a = self.root / "a.f32"; b = self.root / "b.f32"
        np.zeros(6, np.float32).tofile(a); np.zeros(6, np.float32).tofile(b)
        left = self.manifest("ck", [self.entry("vision.layer.0.output", a, storage="fp32")])
        right = self.manifest("pytorch", [self.entry("vision.layer.0.output", b, storage="bf16")])
        result = xray.compare_manifests(left, right, self.profile, checkpoint_order=["vision.layer.0.output"])
        self.assertEqual(result["first_divergence"]["classification"], "STORAGE_CONTRACT_MISMATCH")
        self.assertEqual(result["first_divergence"]["fix_owner"], "circuit_and_kernel_map")
        self.assertIn("Do not add model-name", result["architecture_policy"]["forbidden_fix"])
        self.assertEqual(result["fix_progression"]["policy"], "advance_only_after_numerical_evidence")
        self.assertTrue(any("kernel-family unit matrix" in step for step in result["fix_progression"]["steps"]))
        self.assertTrue(any("forward/VJP sensitivity" in step for step in result["fix_progression"]["steps"]))
        self.assertTrue(any("first failure progresses" in step for step in result["fix_progression"]["steps"]))

    def test_producer_mismatch_is_not_mislabeled_as_kernel_math(self):
        a = self.root / "a.f32"; b = self.root / "b.f32"
        np.zeros(6, np.float32).tofile(a); np.ones(6, np.float32).tofile(b)
        left = self.manifest("ck", [self.entry("vision.layer.0.output", a, producer="wrong")])
        right = self.manifest("pytorch", [self.entry("vision.layer.0.output", b)])
        result = xray.compare_manifests(left, right, self.profile, checkpoint_order=["vision.layer.0.output"])
        self.assertEqual(result["first_divergence"]["classification"], "CIRCUIT_PRODUCER_MISMATCH")
        self.assertNotIn("metrics", result["first_divergence"])

    def test_value_failure_reports_logical_coordinate(self):
        a = self.root / "a.f32"; b = self.root / "b.f32"
        got = np.zeros((2, 3), np.float32); ref = np.zeros((2, 3), np.float32)
        got[1, 2] = 1.0
        got.tofile(a); ref.tofile(b)
        result = xray.compare_manifests(
            self.manifest("ck", [self.entry("vision.layer.0.output", a)]),
            self.manifest("pytorch", [self.entry("vision.layer.0.output", b)]), self.profile,
            checkpoint_order=["vision.layer.0.output"],
        )
        divergence = result["first_divergence"]
        self.assertEqual(divergence["classification"], "KERNEL_IMPLEMENTATION_DIVERGENCE")
        self.assertEqual(divergence["metrics"]["worst_coordinate"], {"token": 1, "channel": 2})

    def test_ranking_failure_is_reported_after_tensor_passes(self):
        a = self.root / "a.f32"; b = self.root / "b.f32"
        np.arange(6, dtype=np.float32).tofile(a); np.arange(6, dtype=np.float32).tofile(b)
        ranking = {"schema": "cke.xray_ranking_report", "schema_version": 1,
                   "checks": [{"kind": "teacher_forced", "position": 12, "status": "fail", "ck_top1": 4, "oracle_top1": 5}]}
        result = xray.compare_manifests(
            self.manifest("ck", [self.entry("vision.layer.0.output", a)]),
            self.manifest("pytorch", [self.entry("vision.layer.0.output", b)]), self.profile, ranking,
            checkpoint_order=["vision.layer.0.output"],
        )
        self.assertEqual(result["first_divergence"]["classification"], "RANKING_DIVERGENCE")

    def test_builder_uses_call_ir_as_checkpoint_authority(self):
        tensor = self.root / "tensor.f32"; np.arange(6, dtype=np.float32).tofile(tensor)
        checkpoint = {
            "id": "vision.layer.0.output", "producer": "mlp_residual", "tensor": "layer_out",
            "logical_layout": "token_major", "axis_names": ["token", "channel"], "storage_dtype": "fp32",
            "phase": "prefill", "layer": 0, "kernel_id": "residual", "function": "residual_forward",
            "resolved_contract_id": "contract.residual",
        }
        call_ir = {"operations": [{"semantic_checkpoints": [checkpoint]}]}
        report = {"torch": {"tensors": {"layer_out@0": {"path": str(tensor), "shape": [2, 3]}}},
                  "comparisons": {"layer_out@0": {"ck_path": str(tensor), "shape": [6]}}}
        manifest = builder.build_manifest(
            backend="pytorch", call_ir=call_ir, tensor_report=report, model="fixture", source="unit",
            phase="prefill", storage_dtype_override="bf16",
        )
        self.assertEqual(manifest["checkpoints"][0]["producer"], "mlp_residual")
        self.assertEqual(manifest["checkpoints"][0]["kernel_id"], "residual")
        self.assertEqual(manifest["checkpoints"][0]["storage_dtype"], "bf16")

    def test_requested_missing_checkpoint_is_a_diagnostic_failure(self):
        a = self.root / "a.f32"; np.zeros(6, np.float32).tofile(a)
        result = xray.compare_manifests(
            self.manifest("ck", [self.entry("vision.layer.0.output", a)]),
            self.manifest("pytorch", [self.entry("vision.layer.0.output", a)]), self.profile,
        )
        self.assertEqual(result["first_divergence"]["classification"], "MISSING_CHECKPOINT")
        self.assertIn("exporter", result["first_divergence"]["recommended_action"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
