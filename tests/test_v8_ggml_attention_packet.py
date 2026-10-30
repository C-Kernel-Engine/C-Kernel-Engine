import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from version.v8.scripts.replay_ggml_attention_packet_v8 import (
    FLASH_SOURCE, OBSERVED_SOURCE, _compile_ggml_source, _observed_capture,
    fp64_full_flash_reference_rows,
    fp64_reference, pinned_tree_identity, run_ggml, run_ggml_flash_rows,
    replay_observed_oracle, validate_oracle_contract,
)
from version.v8.scripts.xray_operation_bundle_v8 import BundleError


ROOT = Path(__file__).resolve().parents[1]


class GgmlAttentionPacketTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"context": {"oracle_mode": "disabled", "valid_cache_positions": [0, 3]}}
        self.call = {"sliding_window": 8, "head_dim": 4, "aligned_head_dim": 4,
                     "selector": "CK_ATTN_REDUCTION_F16_REGULAR_NOSCALE"}

    def test_rejects_incompatible_contracts(self):
        validate_oracle_contract(self.manifest, self.call, [1, 2])
        cases = (
            ({"oracle_mode": "enabled"}, {}, "non-flash"),
            ({"valid_cache_positions": [0, 2]}, {}, "position range"),
            ({}, {"sliding_window": 2}, "wrapped"),
            ({}, {"aligned_head_dim": 8}, "padded"),
            ({}, {"selector": "CK_ATTN_REDUCTION_SCALED"}, "scaling"),
        )
        for context_patch, call_patch, message in cases:
            with self.subTest(message=message):
                manifest = {"context": {**self.manifest["context"], **context_patch}}
                call = {**self.call, **call_patch}
                with self.assertRaisesRegex(BundleError, message):
                    validate_oracle_contract(manifest, call, [1, 2])

    def _synthetic_observed_capture(self, root):
        name = "kqv_out-0-token-000001-occ-000"
        q = np.array([
            [[0.5, -0.25, 0.75, 0.125], [-0.5, 0.4, 0.2, 0.3]],
            [[0.1, 0.2, -0.7, 0.9], [0.25, -0.3, 0.6, 0.4]],
        ], dtype="<f4")
        k = np.zeros((32, 1, 4), dtype="<f2")
        k[0, 0] = [0.5, 0.25, -0.25, 0.75]
        k[1, 0] = [-0.5, 0.1, 0.8, -0.3]
        v = np.zeros((32, 1, 4), dtype="<f2")
        v[0, 0] = [0.1, 0.2, 0.3, 0.4]
        v[1, 0] = [-0.2, 0.4, 0.8, 0.1]
        mask = np.full((2, 32), -np.inf, dtype="<f4")
        mask[0, 0] = 0
        mask[1, :2] = 0
        entries = {
            "q": (q.tobytes(), 0, [4, 2, 2, 1], [4, 32, 16, 64]),
            "k": (k.tobytes(), 1, [4, 32, 1, 1], [2, 8, 8, 256]),
            "v": (v[:, 0].T.copy().tobytes(), 1, [32, 4, 1, 1], [2, 64, 256, 256]),
            "mask": (mask.tobytes(), 0, [32, 2, 1, 1], [4, 128, 256, 256]),
        }
        for role, (raw, dtype, ne, nb) in entries.items():
            base = root / f"{name}.nonflash_{role}"
            Path(str(base) + ".bin").write_bytes(raw)
            Path(str(base) + ".json").write_text(json.dumps({
                "role": role, "type": dtype, "ne": ne, "nb": nb, "nbytes": len(raw),
            }), encoding="ascii")
        (root / f"{name}.bin").write_bytes(np.zeros(16, dtype="<f4").tobytes())
        (root / f"{name}.json").write_text(json.dumps({
            "type": 0, "ne": [8, 2, 1, 1], "nb": [4, 32, 64, 64],
        }), encoding="ascii")
        return name, v, mask

    def test_observed_nonflash_capture_checks_physical_layout_and_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name, _, _ = self._synthetic_observed_capture(root)
            specs, geometry = _observed_capture(root, name)
            self.assertEqual((geometry["queries"], geometry["keys"]), (2, 32))
            self.assertEqual(specs["v"]["nb"][1], 64)
            mask_path = root / f"{name}.nonflash_mask.bin"
            original = mask_path.read_bytes()
            mask_path.write_bytes(original[:-4])
            with self.assertRaisesRegex(BundleError, "storage contract"):
                _observed_capture(root, name)
            mask_path.write_bytes(original)
            report_path = root / "parity.json"
            report_path.write_text(json.dumps({"dump_compare": {
                "llama_flash_attention_mode": "disabled",
                "llama_dump_dir": str(root / "other"),
                "llama_nonflash_input_captures": 1,
            }}), encoding="ascii")
            with self.assertRaisesRegex(BundleError, "does not bind"):
                replay_observed_oracle(
                    Path("unused"), root, name, scale=1, alibi=0, threads=1,
                    parity_report=report_path,
                )

    def test_observed_nonflash_ggml_replay_detects_wrong_mask(self):
        tree = Path(os.environ.get("CK_PINNED_LLAMA_TREE", str(ROOT / "llama.cpp")))
        try:
            pinned_tree_identity(tree)
        except (BundleError, OSError, subprocess.CalledProcessError):
            self.skipTest("pinned GGML libraries unavailable; asset-backed replay runs locally")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name, v, mask = self._synthetic_observed_capture(root)
            with tempfile.TemporaryDirectory(prefix="cke-observed-synthetic-") as build:
                executable = Path(build) / "replay"
                output = Path(build) / "out.f32"
                _compile_ggml_source(tree, OBSERVED_SOURCE, executable)
                args = [str(executable)]
                args.extend(str(root / f"{name}.nonflash_{role}.bin")
                            for role in ("q", "k", "v", "mask"))
                args.extend((str(output), "2", "32", "2", "1", "4",
                             "32", "16", "8", "8", "64", "256", "128",
                             "1", "0", "1"))
                subprocess.run(args, check=True, capture_output=True, text=True)
                actual = np.fromfile(output, dtype="<f4").reshape(2, 2, 4)
                self.assertLess(float(np.max(np.abs(actual[0] - v[0, 0]))), 1e-6)
                self.assertGreater(float(np.max(np.abs(actual[1] - actual[0]))), 0.05)
                actual.tofile(root / f"{name}.bin")
            report = replay_observed_oracle(tree, root, name, scale=1, alibi=0, threads=1)
            self.assertEqual(report["oracle_replay_fidelity"], "pass")
            altered = mask.copy()
            altered[1, 0] = -np.inf
            mask_path = root / f"{name}.nonflash_mask.bin"
            mask_path.write_bytes(altered.tobytes())
            changed = replay_observed_oracle(tree, root, name, scale=1, alibi=0, threads=1)
            self.assertEqual(changed["oracle_replay_fidelity"], "fail")
            self.assertGreater(changed["comparison"]["max_abs"], 0.05)

    def test_pinned_ggml_graph_bounded_against_small_fp64_case(self):
        tree = Path(os.environ.get("CK_PINNED_LLAMA_TREE", str(ROOT / "llama.cpp")))
        try:
            pinned_tree_identity(tree)
        except (BundleError, OSError, subprocess.CalledProcessError):
            self.skipTest("pinned GGML libraries unavailable; asset-backed replay runs locally")
        q = np.array([[[0.25, -0.5, 0.75, 1.0], [0.4, -0.2, 0.8, -0.1]],
                      [[1.0, 0.5, -0.75, 0.25], [-0.3, 0.2, 0.7, 0.8]],
                      [[0.5, 0.4, -0.6, 0.2], [0.6, -0.7, 0.1, 0.3]]], dtype=np.float32)
        k = np.array([[[0.5, 0.25, -0.25, 0.75]],
                      [[-0.5, 0.1, 0.8, -0.3]],
                      [[0.7, -0.4, 0.2, 0.6]]], dtype=np.float16)
        v = np.array([[[0.1, 0.2, 0.3, 0.4]],
                      [[-0.2, 0.4, 0.8, 0.1]],
                      [[0.9, -0.1, 0.2, 0.7]]], dtype=np.float16)
        actual, identity = run_ggml(tree, q, k, v)
        expected = fp64_reference(q, k, v)
        self.assertEqual(identity["source_revision"], subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=tree, text=True).strip())
        self.assertLess(float(np.max(np.abs(actual - expected))), 2e-4)
        self.assertGreater(float(np.max(np.abs(actual - expected))), 1e-5)
        self.assertGreater(float(np.max(np.abs(actual[0] - actual[-1]))), 0.1)

    def test_full_flash_geometry_and_operand_contract(self):
        q = np.zeros((3, 2, 4), dtype=np.float32)
        k = np.zeros((3, 1, 4), dtype=np.float16)
        v = np.zeros_like(k)
        tree = Path("unused-for-rejected-inputs")
        with self.assertRaisesRegex(BundleError, "full-query geometry"):
            run_ggml_flash_rows(tree, q[:1], k, v, scale=1.0)
        with self.assertRaisesRegex(BundleError, "finite operands"):
            run_ggml_flash_rows(tree, q, k, v, scale=float("nan"))
        with self.assertRaisesRegex(BundleError, "matching K/V"):
            run_ggml_flash_rows(tree, q, k, v[:2], scale=1.0)
        with self.assertRaisesRegex(BundleError, "observed FP32 Q"):
            run_ggml_flash_rows(tree, q, k.astype(np.float32), v, scale=1.0)
        with self.assertRaisesRegex(BundleError, "finite operands"):
            bad_q = q.copy()
            bad_q[0, 0, 0] = np.inf
            run_ggml_flash_rows(tree, bad_q, k, v, scale=1.0)

    def test_pinned_full_flash_against_noncausal_fp64(self):
        tree = Path(os.environ.get("CK_PINNED_LLAMA_TREE", str(ROOT / "llama.cpp")))
        try:
            pinned_tree_identity(tree)
        except (BundleError, OSError, subprocess.CalledProcessError):
            self.skipTest("pinned GGML libraries unavailable; asset-backed replay runs locally")
        q = np.array([[[0.25, -0.5, 0.75, 1.0], [0.4, -0.2, 0.8, -0.1]],
                      [[1.0, 0.5, -0.75, 0.25], [-0.3, 0.2, 0.7, 0.8]],
                      [[0.5, 0.4, -0.6, 0.2], [0.6, -0.7, 0.1, 0.3]]], dtype=np.float32)
        k = np.array([[[0.5, 0.25, -0.25, 0.75]],
                      [[-0.5, 0.1, 0.8, -0.3]],
                      [[0.7, -0.4, 0.2, 0.6]]], dtype=np.float16)
        v = np.array([[[0.1, 0.2, 0.3, 0.4]],
                      [[-0.2, 0.4, 0.8, 0.1]],
                      [[0.9, -0.1, 0.2, 0.7]]], dtype=np.float16)
        actual, identity = run_ggml_flash_rows(tree, q, k, v, scale=1.0)
        expected = np.empty_like(q)
        for token in range(q.shape[0]):
            for head in range(q.shape[1]):
                scores = k[:, 0].astype(np.float64) @ q[token, head].astype(np.float64)
                weights = np.exp(scores - scores.max())
                expected[token, head] = (weights / weights.sum()) @ v[:, 0].astype(np.float64)
        self.assertEqual(identity["contract"]["query_extent"], "complete")
        self.assertEqual(identity["contract"]["mode"], "full_noncausal_flash")
        self.assertLess(float(np.max(np.abs(actual - expected))), 2e-4)
        np.testing.assert_allclose(
            fp64_full_flash_reference_rows(q, k, v, [0, 2], scale=1.0),
            expected[[0, 2]], rtol=0, atol=1e-7,
        )
        with self.assertRaisesRegex(BundleError, "distinct in-range"):
            fp64_full_flash_reference_rows(q, k, v, [0, 0], scale=1.0)
        changed = v.copy()
        changed[-1, 0, 0] += np.float16(1.0)
        altered, _ = run_ggml_flash_rows(tree, q, k, changed, scale=1.0)
        self.assertGreater(float(abs(altered[0, 0, 0] - actual[0, 0, 0])), 0.05)
        subset, evidence = run_ggml_flash_rows(
            tree, q[:1], k, v, scale=1.0, allow_subset_queries=True,
        )
        self.assertEqual(subset.shape, (1, 2, 4))
        self.assertEqual(evidence["contract"]["query_extent"], "subset_not_connected_fidelity")

    def test_flash_helper_rejects_truncated_operand(self):
        tree = Path(os.environ.get("CK_PINNED_LLAMA_TREE", str(ROOT / "llama.cpp")))
        try:
            pinned_tree_identity(tree)
        except (BundleError, OSError, subprocess.CalledProcessError):
            self.skipTest("pinned GGML libraries unavailable; asset-backed replay runs locally")
        with tempfile.TemporaryDirectory(prefix="cke-flash-truncated-") as temp:
            directory = Path(temp)
            executable = directory / "replay"
            _compile_ggml_source(tree, FLASH_SOURCE, executable)
            q, k, v, output = (directory / name for name in ("q.f32", "k.f16", "v.f16", "out.f32"))
            q.write_bytes(np.zeros((3, 2, 4), dtype="<f4").tobytes())
            k.write_bytes(np.zeros((1,), dtype="<f2").tobytes())
            v.write_bytes(np.zeros((3, 1, 4), dtype="<f2").tobytes())
            env = dict(os.environ, LD_LIBRARY_PATH=str(tree / "build" / "bin"))
            result = subprocess.run(
                [str(executable), str(q), str(k), str(v), str(output),
                 "3", "3", "2", "1", "4", "1", "1"],
                env=env, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("incorrectly sized", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
