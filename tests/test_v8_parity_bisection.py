from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "version" / "v8" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import parity_bisect_v8 as parity  # noqa: E402


THRESHOLDS = {
    "min_cosine": 0.9999,
    "max_rmse": 1.0e-4,
    "max_relative_rmse": 1.0e-4,
    "max_abs": 1.0e-3,
    "require_finite": True,
}


def _profile() -> dict:
    checkpoints = {
        name: {"producer": name.rsplit(".", 1)[0]}
        for name in (
            "vision.layer.0.output",
            "vision.layer.8.output",
            "vision.layer.1.output",
            "vision.layer.4.output",
        )
    }
    return {
        "schema": "cke.v8.parity_bisection_profile",
        "schema_version": 1,
        "id": "synthetic-vision",
        "circuit": "synthetic",
        "reference_backend": "pytorch",
        "candidate_backend": "cke",
        "defaults": dict(THRESHOLDS),
        "checkpoints": checkpoints,
        "bisection": {
            "root_group": "sparse",
            "groups": {
                "sparse": {
                    "checkpoints": ["vision.layer.0.output", "vision.layer.8.output"],
                    "expand_on_failure": {"vision.layer.8.output": "layers_1_8"},
                },
                "layers_1_8": {
                    "checkpoints": ["vision.layer.1.output", "vision.layer.4.output"],
                    "expand_on_failure": {},
                },
            },
        },
    }


def _tensor(checkpoint: str, path: str, shape: list[int], axes: list[str], *, dtype: str = "fp32") -> dict:
    return {
        "checkpoint": checkpoint,
        "path": path,
        "format": "raw",
        "dtype": dtype,
        "shape": shape,
        "axes": axes,
        "canonical_axes": ["token", "head", "channel"],
        "producer": checkpoint.rsplit(".", 1)[0],
    }


def _write_manifest(root: Path, name: str, backend: str, tensors: list[dict]) -> Path:
    path = root / f"{name}.json"
    path.write_text(json.dumps({
        "schema": "cke.v8.parity_tensor_manifest",
        "schema_version": 1,
        "backend": backend,
        "run": {"id": name},
        "tensors": tensors,
    }), encoding="utf-8")
    return path


class ParityBisectionTests(unittest.TestCase):
    def test_canonical_axes_remove_storage_layout_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_s:
            root = Path(tmp_s)
            logical = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            head_major = logical.transpose(1, 0, 2).copy()
            logical.tofile(root / "token_major.f32")
            head_major.tofile(root / "head_major.f32")
            ref_path = _write_manifest(root, "ref", "pytorch", [
                _tensor("vision.layer.0.output", "token_major.f32", [2, 3, 4], ["token", "head", "channel"]),
            ])
            got_path = _write_manifest(root, "got", "cke", [
                _tensor("vision.layer.0.output", "head_major.f32", [3, 2, 4], ["head", "token", "channel"]),
            ])
            ref_doc = parity._load_json(ref_path)
            got_doc = parity._load_json(got_path)
            parity.validate_manifest(ref_doc)
            parity.validate_manifest(got_doc)
            ref = parity.load_canonical_tensor(ref_path, ref_doc["tensors"][0])
            got = parity.load_canonical_tensor(got_path, got_doc["tensors"][0])
            np.testing.assert_array_equal(ref, got)

    def test_bisection_reports_deepest_first_divergent_edge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_s:
            root = Path(tmp_s)
            checkpoints = list(_profile()["checkpoints"])
            ref_tensors = []
            got_tensors = []
            for index, checkpoint in enumerate(checkpoints):
                ref = np.full((2, 1, 4), index + 1, dtype=np.float32)
                got = ref.copy()
                if checkpoint in {"vision.layer.8.output", "vision.layer.4.output"}:
                    got[1, 0, 2] += np.float32(0.1)
                ref_name = f"ref_{index}.f32"
                got_name = f"got_{index}.f32"
                ref.tofile(root / ref_name)
                got.tofile(root / got_name)
                ref_tensors.append(_tensor(checkpoint, ref_name, [2, 1, 4], ["token", "head", "channel"]))
                got_tensors.append(_tensor(checkpoint, got_name, [2, 1, 4], ["token", "head", "channel"]))
            ref_path = _write_manifest(root, "ref", "pytorch", ref_tensors)
            got_path = _write_manifest(root, "got", "cke", got_tensors)
            report = parity.run_bisection(_profile(), ref_path, got_path)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["first_observed_divergence"]["checkpoint"], "vision.layer.8.output")
            self.assertEqual(report["first_divergence"]["checkpoint"], "vision.layer.4.output")
            self.assertEqual([row["group"] for row in report["groups"]], ["sparse", "layers_1_8"])

    def test_missing_granular_tensors_produces_bounded_next_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_s:
            root = Path(tmp_s)
            ref = np.ones((2, 1, 4), dtype=np.float32)
            got = ref.copy()
            got[0, 0, 0] += np.float32(0.1)
            ref.tofile(root / "ref.f32")
            got.tofile(root / "got.f32")
            checkpoint = "vision.layer.8.output"
            ref_path = _write_manifest(root, "ref", "pytorch", [
                _tensor(checkpoint, "ref.f32", [2, 1, 4], ["token", "head", "channel"]),
            ])
            got_path = _write_manifest(root, "got", "cke", [
                _tensor(checkpoint, "got.f32", [2, 1, 4], ["token", "head", "channel"]),
            ])
            profile = _profile()
            profile["bisection"]["groups"]["sparse"]["checkpoints"] = [checkpoint]
            report = parity.run_bisection(profile, ref_path, got_path)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["next_request"]["group"], "layers_1_8")
            self.assertEqual(report["next_request"]["reason"], checkpoint)

    def test_bf16_raw_export_is_compared_as_fp32(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_s:
            root = Path(tmp_s)
            expected = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32).reshape(1, 1, 4)
            words = (expected.reshape(-1).view(np.uint32) >> np.uint32(16)).astype(np.uint16)
            words.tofile(root / "tensor.bf16")
            manifest_path = _write_manifest(root, "bf16", "pytorch", [
                _tensor("vision.layer.0.output", "tensor.bf16", [1, 1, 4], ["token", "head", "channel"], dtype="bf16"),
            ])
            document = parity._load_json(manifest_path)
            got = parity.load_canonical_tensor(manifest_path, document["tensors"][0])
            np.testing.assert_array_equal(got, expected)

    def test_unknown_profile_field_is_a_hard_fault(self) -> None:
        profile = _profile()
        profile["fallback"] = "accept"
        with self.assertRaisesRegex(parity.ParityContractError, "HARD PARITY CONTRACT FAULT"):
            parity.validate_profile(profile)

    def test_duplicate_manifest_checkpoint_is_a_hard_fault(self) -> None:
        tensor = _tensor("vision.layer.0.output", "unused.f32", [1, 1, 1], ["token", "head", "channel"])
        document = {
            "schema": "cke.v8.parity_tensor_manifest",
            "schema_version": 1,
            "backend": "cke",
            "run": {"id": "duplicate"},
            "tensors": [tensor, dict(tensor)],
        }
        with self.assertRaisesRegex(parity.ParityContractError, "duplicate checkpoint"):
            parity.validate_manifest(document)

    def test_profile_contract_must_match_both_backend_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_s:
            root = Path(tmp_s)
            value = np.ones((1, 1, 1), dtype=np.float32)
            value.tofile(root / "value.f32")
            checkpoint = "vision.layer.0.output"
            ref_tensor = _tensor(checkpoint, "value.f32", [1, 1, 1], ["token", "head", "channel"])
            got_tensor = dict(ref_tensor)
            ref_tensor["contract"] = "mrope_full_width"
            got_tensor["contract"] = "mrope_wrong_width"
            ref_path = _write_manifest(root, "ref", "pytorch", [ref_tensor])
            got_path = _write_manifest(root, "got", "cke", [got_tensor])
            profile = _profile()
            profile["checkpoints"][checkpoint]["contract"] = "mrope_full_width"
            profile["bisection"]["groups"]["sparse"]["checkpoints"] = [checkpoint]
            profile["bisection"]["groups"]["sparse"]["expand_on_failure"] = {}
            with self.assertRaisesRegex(parity.ParityContractError, "contract.*does not match"):
                parity.run_bisection(profile, ref_path, got_path)


if __name__ == "__main__":
    unittest.main()
