#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "xray_execution_state_v8.py"
SPEC = importlib.util.spec_from_file_location("xray_execution_state_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
xray = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(xray)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class XRayExecutionStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="xray_state_v8_")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def artifact(self, role: str, values: np.ndarray, name: str | None = None):
        path = self.root / f"{name or role}.f16"
        np.asarray(values, dtype=np.float16).tofile(path)
        return {
            "checkpoint_id": f"decoder.layer.0.{role}", "role": role,
            "tensor_path": str(path), "dtype": "fp16", "shape": list(values.shape),
            "row_axis": 0, "sha256": digest(path),
        }

    def trace(self, backend: str, calls, artifacts=None):
        return {
            "schema": "cke.xray_execution_trace", "schema_version": 1, "backend": backend,
            "run": {"model": "fixture", "phase": "mixed_prefill", "source": "unit"},
            "execution": {"policy_id": "segmented_mixed_prefill", "calls": calls},
            "state": {
                "position_policy_id": "mrope", "position": [41, 41, 41],
                "cache_token_count": 1307, "append_index": 1306,
                "cache_layout": {
                    "layout_id": "layer_head_token_channel", "base_address": "0x1234",
                    "layer_stride_bytes": 65536, "head_stride_bytes": 8192,
                    "token_stride_bytes": 256, "channel_stride_bytes": 2,
                },
            },
            "artifacts": artifacts or [],
        }

    @staticmethod
    def call(call_id: str, kind: str, start: int, count: int, action: str):
        return {
            "call_id": call_id, "kind": kind, "start": start, "count": count,
            "position_start": start, "cache_action": action,
            "kernel_batches": [{
                "checkpoint_id": "decoder.layer.0.q_proj", "kernel_id": "gemm_q4k_q8k",
                "numerical_contract_id": "q4k_q8k_independent_rows",
                "effective_contract_id": "q4k_q8k_independent_rows",
                "m": count, "n": 4096, "k": 4096,
            }],
        }

    def segmented_calls(self):
        return [
            self.call("text_before", "text", 0, 33, "reset"),
            self.call("visual", "visual", 33, 1008, "append"),
            self.call("text_after", "text", 1041, 266, "append"),
        ]

    def test_identical_execution_and_cache_trace_passes(self):
        values = np.arange(24, dtype=np.float16).reshape(3, 8)
        left_artifact = self.artifact("valid_key_cache", values, "left")
        right_artifact = self.artifact("valid_key_cache", values, "right")
        result = xray.compare_traces(
            self.trace("ck", self.segmented_calls(), [left_artifact]),
            self.trace("llamacpp", self.segmented_calls(), [right_artifact]),
        )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["checks"][-1]["exact_bytes"])

    def test_combined_vs_segmented_prefill_fails_before_tensor_loading(self):
        combined = [self.call("combined", "mixed", 0, 1307, "reset")]
        left = self.trace("ck", combined)
        left["execution"]["policy_id"] = "combined_mixed_prefill"
        result = xray.compare_traces(left, self.trace("llamacpp", self.segmented_calls()))
        self.assertEqual(result["first_divergence"]["classification"], "EXECUTION_POLICY_MISMATCH")
        self.assertEqual(result["first_divergence"]["subject_calls"][0]["count"], 1307)
        self.assertEqual([call["count"] for call in result["first_divergence"]["oracle_calls"]], [33, 1008, 266])
        self.assertEqual(len(result["checks"]), 1)

    def test_cache_append_index_mismatch_precedes_cache_bytes(self):
        left = self.trace("ck", self.segmented_calls())
        right = self.trace("llamacpp", self.segmented_calls())
        left["state"]["append_index"] = 1305
        result = xray.compare_traces(left, right)
        self.assertEqual(result["first_divergence"]["classification"], "CACHE_STATE_METADATA_MISMATCH")
        self.assertEqual(result["first_divergence"]["field"], "append_index")

    def test_effective_shape_dispatch_contract_mismatch_precedes_tensors(self):
        left = self.trace("ck", self.segmented_calls())
        right = self.trace("llamacpp", self.segmented_calls())
        visual_left = left["execution"]["calls"][1]["kernel_batches"][0]
        visual_right = right["execution"]["calls"][1]["kernel_batches"][0]
        for batch in (visual_left, visual_right):
            batch["kernel_id"] = "attention_prefill_f16cache_flash_auto"
            batch["numerical_contract_id"] = "f16_flash_auto_qtile64"
        visual_left["effective_contract_id"] = "f16_online_single_range"
        visual_right["effective_contract_id"] = "f16_kv_tiled64_fp32_softmax_fp64_sum_update"

        result = xray.compare_traces(left, right)
        failure = result["first_divergence"]
        self.assertEqual(failure["classification"], "KERNEL_CONTRACT_MISMATCH")
        self.assertEqual(failure["stage"], "kernel_contracts")
        self.assertEqual(
            failure["subject"][1]["effective_contract_id"],
            "f16_online_single_range",
        )

    def test_first_changed_cache_row_is_reported(self):
        expected = np.arange(32, dtype=np.float16).reshape(4, 8)
        actual = expected.copy()
        actual[2, 5] += np.float16(1.0)
        result = xray.compare_traces(
            self.trace("ck", self.segmented_calls(), [self.artifact("valid_key_cache", actual, "actual")]),
            self.trace("llamacpp", self.segmented_calls(), [self.artifact("valid_key_cache", expected, "expected")]),
        )
        failure = result["first_divergence"]
        self.assertEqual(failure["classification"], "CACHE_CONTENT_DIVERGENCE")
        self.assertEqual(failure["first_differing_row"], 2)
        self.assertGreater(failure["metrics"]["max_abs"], 0.0)

    def test_cache_append_roundtrip_is_checked_within_subject(self):
        source = np.arange(16, dtype=np.float32).reshape(2, 8) / 7.0
        stored = source.astype(np.float16)
        stored[1, 4] += np.float16(0.25)
        source_path = self.root / "new_key.f32"
        source.tofile(source_path)
        source_artifact = {
            "checkpoint_id": "decoder.layer.0.new_key", "role": "new_key",
            "tensor_path": str(source_path), "dtype": "fp32", "shape": [2, 8],
            "row_axis": 0, "sha256": digest(source_path),
        }
        subject = self.trace("ck", self.segmented_calls(), [source_artifact, self.artifact("stored_key", stored)])
        oracle = self.trace("llamacpp", self.segmented_calls(), [source_artifact, self.artifact("stored_key", source.astype(np.float16), "oracle_stored")])
        result = xray.compare_traces(subject, oracle)
        failure = result["first_divergence"]
        self.assertEqual(failure["classification"], "CACHE_APPEND_ROUNDTRIP_DIVERGENCE")
        self.assertEqual(failure["backend"], "ck")
        self.assertEqual(failure["first_differing_row"], 1)

    def test_identical_inputs_then_different_output_is_arithmetic(self):
        query = np.arange(16, dtype=np.float16).reshape(2, 8)
        expected = np.ones((2, 8), dtype=np.float16)
        actual = expected.copy(); actual[1, 3] += np.float16(0.125)
        result = xray.compare_traces(
            self.trace("ck", self.segmented_calls(), [
                self.artifact("query", query, "left_query"),
                self.artifact("attention_output", actual, "left_output"),
            ]),
            self.trace("llamacpp", self.segmented_calls(), [
                self.artifact("query", query, "right_query"),
                self.artifact("attention_output", expected, "right_output"),
            ]),
        )
        failure = result["first_divergence"]
        self.assertEqual(failure["classification"], "ATTENTION_ARITHMETIC_DIVERGENCE")
        self.assertEqual(failure["stage"], "attention_output")


if __name__ == "__main__":
    unittest.main(verbosity=2)
