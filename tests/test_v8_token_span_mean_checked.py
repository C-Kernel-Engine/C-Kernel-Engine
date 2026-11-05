"""Reusable checked token-span pooling, including ordinary generated execution."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/kernels/token_span_mean_checked.c"
FP = ctypes.POINTER(ctypes.c_float)
IP = ctypes.POINTER(ctypes.c_int32)


def sequential_fp64_span_mean(values: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """Independent implementation of the map's ascending-token reduction."""
    result = np.empty((len(spans), values.shape[1]), dtype=np.float32)
    for row, (start, end) in enumerate(spans):
        for channel in range(values.shape[1]):
            total = 0.0
            for token in range(int(start), int(end)):
                total += float(values[token, channel])
            result[row, channel] = np.float32(total / (int(end) - int(start)))
    return result


def _circuit() -> dict:
    return {
        "version": 3, "name": "token_span_pool_synthetic",
        "family": "bounded_graph", "checked_native_entry": True,
        "contract": {"runtime_invariants": {"inference_only": True}},
        "activation_buffers": {
            "input": {"shape": [5, 3]},
            "spans": {"shape": [2, 2]},
            "output": {"shape": [2, 3]},
        },
        "activation_bindings": {"input": "input", "spans": "spans",
                                "output": "output"},
        "native_entry": {"function": "ck_token_span_pool", "params": [
            {"c_type": "uint8_t *", "name": "arena"},
            {"c_type": "size_t", "name": "arena_bytes"}],
            "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
        "sequence": ["pool"],
        "block_types": {"pool": {"sequence": ["header", "body", "footer"],
            "header": [], "body": {"type": "dense", "ops": [{
                "id": "pool", "op": "token_span_mean_checked",
                "kernel": "token_span_mean_checked_f32", "returns_status": True,
                "graph_slots": {"inputs": {"input": "external:input",
                                           "spans": "external:spans"},
                                "outputs": {"output": "external:output"}},
                "params": {"T": 5, "S": 2, "C": 3, "call_constants": {
                    "input_elements": 15, "input_stride": 3,
                    "token_capacity": 5, "valid_tokens": 4, "channels": 3,
                    "span_elements": 4, "span_count": 2,
                    "output_elements": 6, "output_stride": 3}},
            }]}, "footer": []}},
        "runtime_constants": {}, "runtime_lengths": {},
    }


class TokenSpanMeanCheckedTest(unittest.TestCase):
    def test_pinned_clef_flash_head_identity(self):
        fixture = ROOT / "tests/fixtures/decision/clef_flash"
        identity = json.loads((fixture / "identity.json").read_text())
        config_bytes = (fixture / "config.json").read_bytes()
        head_bytes = (fixture / "joint_head_config.json").read_bytes()
        self.assertEqual(hashlib.sha256(config_bytes).hexdigest(),
                         identity["config_sha256"])
        self.assertEqual(hashlib.sha256(head_bytes).hexdigest(),
                         identity["joint_head_config_sha256"])
        config = json.loads(config_bytes)
        head = json.loads(head_bytes)
        self.assertEqual(config["architectures"],
                         ["Qwen3_5ForConditionalGeneration"])
        self.assertEqual(config["text_config"]["hidden_size"],
                         head["hidden_size"])
        self.assertEqual(head, {"hidden_size": 4096, "width": 1024,
                                "routing_layers": 2, "layers": 4,
                                "heads": 16, "feedforward": 4096})
        self.assertEqual(config["text_config"]["num_hidden_layers"], 32)

    def _direct(self, folder: Path):
        library = folder / "pool.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                        "-I", str(ROOT / "include"), str(SOURCE), "-lm",
                        "-o", str(library)], check=True, capture_output=True)
        fn = ctypes.CDLL(str(library)).ck_token_span_mean_checked_f32
        fn.argtypes = [FP, ctypes.c_size_t, ctypes.c_size_t,
                       ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                       IP, ctypes.c_size_t, ctypes.c_size_t,
                       FP, ctypes.c_size_t, ctypes.c_size_t]
        fn.restype = ctypes.c_int
        return fn

    @staticmethod
    def _call(fn, values, spans, output, *, valid=4, out_elements=None):
        return fn(values.ctypes.data_as(FP), values.size, values.shape[1],
                  values.shape[0], valid, 3, spans.ctypes.data_as(IP),
                  spans.size, spans.shape[0], output.ctypes.data_as(FP),
                  output.size if out_elements is None else out_elements, 4)

    def test_oracle_rejection_padding_and_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            fn = self._direct(Path(temp))
            values = np.arange(15, dtype=np.float32).reshape(5, 3) / 7
            spans = np.array([[0, 1], [1, 4]], dtype=np.int32)
            output = np.full((2, 4), -901, dtype=np.float32)
            self.assertEqual(self._call(fn, values, spans, output), 0)
            expected = sequential_fp64_span_mean(values, spans)
            np.testing.assert_array_equal(output[:, :3], expected)
            np.testing.assert_array_equal(output[:, 3], [-901, -901])
            for bad_spans, bad_valid, short_output in (
                (np.array([[0, 1], [3, 5]], np.int32), 4, None),
                (np.array([[0, 0], [1, 4]], np.int32), 4, None),
                (np.array([[-1, 1], [1, 4]], np.int32), 4, None),
                (spans, 6, None), (spans, 4, 6),
            ):
                with self.subTest(spans=bad_spans.tolist(), valid=bad_valid):
                    before = output.copy()
                    self.assertNotEqual(self._call(fn, values, bad_spans, output,
                                                  valid=bad_valid,
                                                  out_elements=short_output), 0)
                    np.testing.assert_array_equal(output, before)
            values[2, 1] = np.nan
            before = output.copy()
            self.assertNotEqual(self._call(fn, values, spans, output), 0)
            np.testing.assert_array_equal(output, before)
            values[2, 1] = 1.0
            self.assertEqual(self._call(fn, values, spans, output), 0)
            # Large finite values must not overflow an FP32 partial sum.
            values[:] = np.finfo(np.float32).max
            self.assertEqual(self._call(fn, values, spans, output), 0)
            np.testing.assert_array_equal(
                output[:, :3], np.finfo(np.float32).max)
            before = values.copy()
            self.assertNotEqual(fn(values.ctypes.data_as(FP), values.size, 3,
                                   5, 4, 3, spans.ctypes.data_as(IP),
                                   spans.size, 2, values.ctypes.data_as(FP),
                                   values.size, 3), 0)
            np.testing.assert_array_equal(values, before)
            output[:] = -901
            self.assertNotEqual(fn(None, values.size, 3, 5, 4, 3,
                                   spans.ctypes.data_as(IP), spans.size, 2,
                                   output.ctypes.data_as(FP), output.size, 4), 0)
            np.testing.assert_array_equal(output, -901)

    def test_fp64_order_cancellation_and_head_width(self):
        with tempfile.TemporaryDirectory() as temp:
            fn = self._direct(Path(temp))
            channels = 1024
            rows = 257
            rng = np.random.default_rng(12051)
            values = rng.standard_normal((rows, channels)).astype(np.float32)
            values[:3, 0] = [1e8, 1, -1e8]
            values[:, 1] = 0
            spans = np.array([[0, 1], [0, 3], [0, rows],
                              [1, rows], [rows - 1, rows]], dtype=np.int32)
            output = np.full((len(spans), channels + 3), -901, dtype=np.float32)
            rc = fn(values.ctypes.data_as(FP), values.size, channels,
                    rows, rows, channels, spans.ctypes.data_as(IP),
                    spans.size, len(spans), output.ctypes.data_as(FP),
                    output.size, channels + 3)
            self.assertEqual(rc, 0)
            np.testing.assert_array_equal(output[:, :channels],
                                          sequential_fp64_span_mean(values, spans))
            np.testing.assert_array_equal(output[:, channels:], -901)
            self.assertEqual(output[1, 0], np.float32(1 / 3))
            self.assertEqual(output[1, 1], 0)

    def test_live_pytorch_mean_is_separate_diagnostic(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable; the FP64 contract oracle still runs")
        with tempfile.TemporaryDirectory() as temp:
            fn = self._direct(Path(temp))
            values = np.array([[0.2, -0.6, 1.1],
                               [2.0, 0.3, -4.0],
                               [-1.0, 1.2, 3.5],
                               [0.7, -0.8, 0.4]], dtype=np.float32)
            spans = np.array([[0, 4], [1, 3]], dtype=np.int32)
            output = np.full((2, 4), -901, dtype=np.float32)
            self.assertEqual(self._call(fn, values, spans, output), 0)
            oracle = torch.stack([
                torch.from_numpy(values[a:b]).mean(dim=0)
                for a, b in spans]).numpy()
            np.testing.assert_allclose(output[:, :3], oracle, atol=2e-7, rtol=0)

    def test_direct_abi_capacity_stride_alignment_and_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            fn = self._direct(Path(temp))
            values = np.arange(25, dtype=np.float32).reshape(5, 5)
            values[:, 3:] = np.nan  # Physical padding must never be read.
            spans = np.array([[0, 1], [1, 4]], dtype=np.int32)
            output = np.full((2, 4), -901, dtype=np.float32)
            in_ptr, span_ptr, out_ptr = (
                values.ctypes.data_as(FP), spans.ctypes.data_as(IP),
                output.ctypes.data_as(FP))

            def call(*, input_ptr=in_ptr, input_elements=25, input_stride=5,
                     token_capacity=5, valid_tokens=4, channels=3,
                     spans_ptr=span_ptr, span_elements=4, span_count=2,
                     output_ptr=out_ptr, output_elements=8, output_stride=4):
                return fn(input_ptr, input_elements, input_stride,
                          token_capacity, valid_tokens, channels,
                          spans_ptr, span_elements, span_count,
                          output_ptr, output_elements, output_stride)

            self.assertEqual(call(), 0)
            np.testing.assert_array_equal(output[:, :3],
                                          sequential_fp64_span_mean(values[:, :3], spans))
            np.testing.assert_array_equal(output[:, 3], -901)
            for override in (
                {"input_elements": 22}, {"span_elements": 3},
                {"output_elements": 6}, {"input_stride": 0},
                {"input_stride": 2}, {"output_stride": 0},
                {"output_stride": 2}, {"spans_ptr": None},
                {"input_ptr": None}, {"output_ptr": None},
                {"token_capacity": ctypes.c_size_t(-1).value},
                {"span_count": ctypes.c_size_t(-1).value},
                {"output_stride": ctypes.c_size_t(-1).value},
            ):
                with self.subTest(override=override):
                    before = output.copy()
                    before_values = values.copy()
                    before_spans = spans.copy()
                    self.assertNotEqual(call(**override), 0)
                    np.testing.assert_array_equal(output, before)
                    np.testing.assert_array_equal(values, before_values)
                    np.testing.assert_array_equal(spans, before_spans)

            shared = np.array([0, 1, 1, 4, -901, -901, -901, -901],
                              dtype=np.int32)
            before_shared = shared.copy()
            self.assertNotEqual(call(spans_ptr=shared.ctypes.data_as(IP),
                                     output_ptr=shared.ctypes.data_as(FP)), 0)
            np.testing.assert_array_equal(shared, before_shared)

            unaligned_input = (ctypes.c_uint8 * (values.nbytes + 1))()
            unaligned_spans = (ctypes.c_uint8 * (spans.nbytes + 1))()
            unaligned_output = (ctypes.c_uint8 * (output.nbytes + 1))()
            for override in (
                {"input_ptr": ctypes.cast(ctypes.addressof(unaligned_input) + 1, FP)},
                {"spans_ptr": ctypes.cast(ctypes.addressof(unaligned_spans) + 1, IP)},
                {"output_ptr": ctypes.cast(ctypes.addressof(unaligned_output) + 1, FP)},
            ):
                with self.subTest(unaligned=tuple(override)):
                    before = output.copy()
                    unaligned_before = bytes(unaligned_output)
                    self.assertNotEqual(call(**override), 0)
                    np.testing.assert_array_equal(output, before)
                    self.assertEqual(bytes(unaligned_output), unaligned_before)
            output[:] = -901
            self.assertEqual(call(), 0)  # Reject-then-success reuse.
            np.testing.assert_array_equal(output[:, :3],
                                          sequential_fp64_span_mean(values[:, :3], spans))

    def test_generated_call_and_checked_arena(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            graph = _circuit()
            source = {"config": {
                "model": graph["name"], "arch": graph["name"],
                "num_layers": 1, "embed_dim": 3, "num_heads": 1,
                "num_kv_heads": 1, "head_dim": 3,
                "intermediate_size": 3, "context_length": 5,
                "max_seq_len": 5, "vocab_size": 16,
                "activation_buffer_dtypes": {"spans": "i32"}},
                "entries": [], "quant_summary": {}, "template": graph}
            layout, calls, _, _, execute = compile_native_graph(
                folder, source, graph)
            self.assertIn("ck_token_span_mean_checked_f32",
                          [op["function"] for op in calls["operations"]])
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            acts = {item["name"]: item for item in
                    layout["memory"]["activations"]["buffers"]}

            def view(name, dtype, shape):
                return np.ndarray(shape, dtype=dtype, buffer=arena,
                                  offset=acts[name]["offset"])

            values = view("input", np.float32, (5, 3))
            spans = view("spans", np.int32, (2, 2))
            output = view("output", np.float32, (2, 3))
            values[:] = np.arange(15, dtype=np.float32).reshape(5, 3)
            spans[:] = [[0, 1], [1, 4]]
            output[:] = -901
            self.assertEqual(execute(arena, size), 0)
            np.testing.assert_array_equal(output,
                                          sequential_fp64_span_mean(values, spans))
            output[:] = -901
            self.assertNotEqual(execute(arena, size - 1), 0)
            np.testing.assert_array_equal(output, -901)
            spans[1] = [4, 5]
            self.assertNotEqual(execute(arena, size), 0)
            np.testing.assert_array_equal(output, -901)
            spans[1] = [1, 4]
            self.assertEqual(execute(arena, size), 0)


if __name__ == "__main__":
    unittest.main()
