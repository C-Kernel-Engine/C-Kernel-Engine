import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from version.v8.scripts.compare_nonflash_attention_operands_v8 import compare
from version.v8.scripts.xray_operation_bundle_v8 import BundleError, write_bundle


class NonflashAttentionOperandsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cke-nonflash-operands-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.capture = self.root / "captures"
        self.capture.mkdir()
        self.runtime = self.root / "runtime.so"
        self.runtime.write_bytes(b"fixture-runtime")
        self.q = np.array([
            [[0.2, -0.1], [0.3, 0.4]],
            [[-0.2, 0.5], [0.6, -0.3]],
            [[0.7, 0.1], [-0.4, 0.2]],
        ], dtype=np.float32)
        self.k = np.array([[[0.5, 0.2]], [[-0.1, 0.7]], [[0.3, -0.2]]], dtype=np.float16)
        self.v = np.array([[[0.1, -0.3]], [[0.8, 0.4]], [[-0.2, 0.6]]], dtype=np.float16)
        scores = np.array([
            self.k[:, 0].astype(np.float64) @ self.q[-1, head].astype(np.float16).astype(np.float64)
            for head in range(2)
        ])
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        output = weights.astype(np.float16).astype(np.float64) @ self.v[:, 0].astype(np.float64)
        self.bundle = write_bundle(
            self.root / "bundle", operation="attention_sliding", provider="fixture",
            contract="CK_ATTN_REDUCTION_F16_REGULAR_NOSCALE", runtime=self.runtime,
            context={"layer": 0, "oracle_mode": "disabled", "segments": [1, 2],
                     "valid_cache_positions": [0, 3]},
            tensors={
                "Qcur_rope": (self.q, "fp32", ("token", "head", "channel")),
                "Kcache_post_store": (self.k, "fp16", ("position", "kv_head", "channel")),
                "Vcache_post_store": (self.v, "fp16", ("position", "kv_head", "channel")),
                "kqv_out": (np.zeros_like(self.q), "fp32", ("token", "head", "channel")),
            },
        )
        stem = "kqv_out-0-token-000002-occ-000"
        q = self.q[-1].reshape(1, 2, 1, 2)
        k = np.zeros((1, 1, 8, 2), dtype=np.float16)
        k[0, 0, :3] = self.k[:, 0]
        v = np.zeros((1, 1, 2, 8), dtype=np.float16)
        v[0, 0, :, :3] = self.v[:, 0].T
        mask = np.full((1, 1, 1, 8), -np.inf, dtype=np.float32)
        mask[..., :3] = 0
        score_tensor = np.full((1, 2, 1, 8), -np.inf, dtype=np.float32)
        score_tensor[0, :, 0, :3] = scores
        probability_tensor = np.zeros((1, 2, 1, 8), dtype=np.float32)
        probability_tensor[0, :, 0, :3] = weights
        for role, values in (("q", q), ("k", k), ("v", v), ("mask", mask)):
            self._write_capture(f"{stem}.nonflash_{role}", values, role)
        self._write_capture("kq-0-token-000002-occ-000", score_tensor)
        self._write_capture("kq_soft_max-0-token-000002-occ-000", probability_tensor)
        output_base = self.capture / stem
        output.astype("<f4").tofile(str(output_base) + ".bin")
        Path(str(output_base) + ".json").write_text(json.dumps({
            "type": 0, "nbytes": output.size * 4, "token_id": 2,
            "ne": [4, 1, 1, 1],
        }), encoding="utf-8")

    def _write_capture(self, name, values, role=None):
        array = np.ascontiguousarray(values)
        base = self.capture / name
        array.tofile(str(base) + ".bin")
        strides = list(reversed(array.strides))
        metadata = {"type": 1 if array.dtype == np.float16 else 0,
                    "ne": list(reversed(array.shape)), "nb": strides,
                    "nbytes": array.nbytes, "token_id": 2}
        if role is not None:
            metadata["role"] = role
            metadata["capture_stage"] = "score_matmul" if role == "q" else "attention_output"
        else:
            metadata["base_name"] = name.split("-token-", 1)[0]
        Path(str(base) + ".json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_query_captured_after_score_matmul_is_rejected(self):
        path = self.capture / "kqv_out-0-token-000002-occ-000.nonflash_q.json"
        metadata = json.loads(path.read_text())
        metadata["capture_stage"] = "attention_output"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(BundleError, "query must be captured at its score matmul"):
            compare(self.bundle, self.capture, 2)

    def test_strided_v_and_mask_produce_aligned_comparison(self):
        result = compare(self.bundle, self.capture, 2)
        self.assertEqual(result["status"], "diagnostic_only")
        for role in ("q", "k_post_store", "v_post_store"):
            self.assertEqual(result["operand_metrics"][role]["max_abs"], 0.0)
            self.assertTrue(result["operand_bitwise_equal"][role])
        self.assertEqual(result["physical_keys"], 8)
        self.assertEqual(result["valid_keys"], 3)
        for role in ("scores", "softmax", "value_output_from_captured_softmax", "connected_output"):
            self.assertLess(result["oracle_vs_f16_gemm"][role]["max_abs"], 1e-7)
        self.assertGreater(result["oracle_vs_fp64"]["output"]["max_abs"], 1e-6)

    def test_batched_query_rows_use_each_absolute_mask_position(self):
        manifest = json.loads(self.bundle.read_text(encoding="utf-8"))
        manifest["context"]["segments"] = [3]
        self.bundle.write_text(json.dumps(manifest), encoding="utf-8")
        q = self.q.transpose(1, 0, 2).reshape(1, 2, 3, 2)
        mask = np.full((1, 1, 3, 8), -np.inf, dtype=np.float32)
        scores = np.full((1, 2, 3, 8), -np.inf, dtype=np.float32)
        probabilities = np.zeros((1, 2, 3, 8), dtype=np.float32)
        output = np.empty((3, 2, 2), dtype=np.float32)
        for row in range(3):
            mask[0, 0, row, :row + 1] = 0
            for head in range(2):
                logits = (self.k[:row + 1, 0].astype(np.float64) @
                          self.q[row, head].astype(np.float16).astype(np.float64))
                weights = np.exp(logits - logits.max())
                weights /= weights.sum()
                scores[0, head, row, :row + 1] = logits
                probabilities[0, head, row, :row + 1] = weights
                output[row, head] = (weights.astype(np.float16).astype(np.float64) @
                                     self.v[:row + 1, 0].astype(np.float64))
        stem = "kqv_out-0-token-000002-occ-000"
        self._write_capture(f"{stem}.nonflash_q", q, "q")
        self._write_capture(f"{stem}.nonflash_mask", mask, "mask")
        self._write_capture("kq-0-token-000002-occ-000", scores)
        self._write_capture("kq_soft_max-0-token-000002-occ-000", probabilities)
        output_base = self.capture / stem
        output.tofile(str(output_base) + ".bin")
        Path(str(output_base) + ".json").write_text(json.dumps({
            "type": 0, "nbytes": output.nbytes, "token_id": 2,
            "ne": [4, 3, 1, 1],
        }), encoding="utf-8")
        result = compare(self.bundle, self.capture, 2)
        self.assertEqual(result["query_start"], 0)
        self.assertEqual(result["query_rows"], 3)
        self.assertTrue(all(result["operand_bitwise_equal"].values()))
        self.assertEqual(result["oracle_masked_probability_max_abs"], 0.0)
        self.assertLess(result["oracle_vs_f16_gemm"]["connected_output"]["max_abs"], 1e-7)
        manifest["context"]["segments"] = [1, 2]
        self.bundle.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(BundleError, "segment boundary"):
            compare(self.bundle, self.capture, 2)

    def test_rejects_wrong_v_layout(self):
        path = self.capture / "kqv_out-0-token-000002-occ-000.nonflash_v.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["ne"] = [2, 8, 1, 1]
        metadata["nb"] = [2, 4, 32, 32]
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(BundleError, "geometry"):
            compare(self.bundle, self.capture, 2)

    def test_accepts_executed_nonflash_mode_but_rejects_flash_mode(self):
        manifest = json.loads(self.bundle.read_text(encoding="utf-8"))
        manifest["context"]["oracle_mode"] = "disabled_for_internal_dump"
        self.bundle.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(compare(self.bundle, self.capture, 2)["status"], "diagnostic_only")
        manifest["context"]["oracle_mode"] = "enabled"
        self.bundle.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(BundleError, "not a non-flash"):
            compare(self.bundle, self.capture, 2)

    def test_rejects_wrong_mask(self):
        path = self.capture / "kqv_out-0-token-000002-occ-000.nonflash_mask.bin"
        mask = np.fromfile(path, dtype="<f4")
        mask[1] = -np.inf
        mask.tofile(path)
        with self.assertRaisesRegex(BundleError, "mask"):
            compare(self.bundle, self.capture, 2)

    def test_rejects_truncated_operand(self):
        path = self.capture / "kqv_out-0-token-000002-occ-000.nonflash_k.bin"
        path.write_bytes(path.read_bytes()[:-2])
        with self.assertRaisesRegex(BundleError, "capture bytes"):
            compare(self.bundle, self.capture, 2)

    def test_fp16_gemm_stage_metrics_detect_changed_scores_and_probabilities(self):
        baseline = compare(self.bundle, self.capture, 2)["oracle_vs_f16_gemm"]
        score_path = self.capture / "kq-0-token-000002-occ-000.bin"
        scores = np.fromfile(score_path, dtype="<f4")
        scores[0] += 0.01
        scores.tofile(score_path)
        changed_scores = compare(self.bundle, self.capture, 2)["oracle_vs_f16_gemm"]
        self.assertGreater(changed_scores["scores"]["max_abs"], 0.009)
        self.assertLess(baseline["scores"]["max_abs"], 1e-7)

        probability_path = self.capture / "kq_soft_max-0-token-000002-occ-000.bin"
        probabilities = np.fromfile(probability_path, dtype="<f4")
        probabilities[0] += 0.01
        probabilities.tofile(probability_path)
        changed_probabilities = compare(self.bundle, self.capture, 2)["oracle_vs_f16_gemm"]
        self.assertGreater(changed_probabilities["softmax"]["max_abs"], 0.009)
        self.assertGreater(changed_probabilities["value_output_from_captured_softmax"]["max_abs"], 1e-4)

    def test_rejects_missing_producer_stage(self):
        (self.capture / "kq-0-token-000002-occ-000.bin").unlink()
        with self.assertRaises(FileNotFoundError):
            compare(self.bundle, self.capture, 2)


if __name__ == "__main__":
    unittest.main()
