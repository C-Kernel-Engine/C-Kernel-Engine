#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int32)
I64P = ctypes.POINTER(ctypes.c_int64)
U16P = ctypes.POINTER(ctypes.c_uint16)

LIB.qwen4_ple_ngram_embed_q5_0.argtypes = [
    I32P, ctypes.c_void_p, I64P, I64P, I64P, F32P, F32P, F32P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int,
]
for name in (
    "qwen4_ple_gate_conv_inject_bf16",
    "qwen4_ple_gate_conv_inject_fp16",
    "qwen4_ple_gate_conv_inject_llama_fp16",
):
    getattr(LIB, name).argtypes = [
        F32P, F32P, F32P, F32P, F32P, F32P, U16P, F32P,
        F32P, F32P, F32P, F32P, F32P, F32P,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
    ]
LIB.recurrent_silu_forward_ggml.argtypes = [
    F32P, F32P, ctypes.c_int, ctypes.c_int,
]


def ptr(value: np.ndarray, kind: object) -> object:
    return value.ctypes.data_as(kind)


def bf16_bits(value: torch.Tensor) -> np.ndarray:
    return value.to(torch.bfloat16).view(torch.uint16).numpy().copy()


class TestQwen4ExpPleQuantized(unittest.TestCase):
    def test_q5_0_embedding_reads_selected_rows_with_quantized_stride(self) -> None:
        rows, head_dim = 12, 32
        # A Q5_0 block with qh=0 and qs=0 decodes every lane as -16 * scale.
        table = np.zeros((rows, 22), dtype=np.uint8)
        scales = np.asarray([(index + 1) / 32 for index in range(rows)], dtype=np.float16)
        table[:, :2] = scales.view(np.uint8).reshape(rows, 2)

        tokens = np.asarray([3], dtype=np.int32)
        multipliers = np.asarray([1, 1], dtype=np.int64)
        offsets = np.asarray([0], dtype=np.int64)
        vocab_sizes = np.asarray([rows], dtype=np.int64)
        state = np.asarray([2], dtype=np.float32)
        state_out = np.zeros_like(state)
        output = np.zeros((1, head_dim), dtype=np.float32)
        LIB.qwen4_ple_ngram_embed_q5_0(
            ptr(tokens, I32P), table.ctypes.data_as(ctypes.c_void_p),
            ptr(multipliers, I64P), ptr(offsets, I64P), ptr(vocab_sizes, I64P),
            ptr(output, F32P), ptr(state, F32P), ptr(state_out, F32P),
            1, 2, 1, head_dim, 0, 1,
        )
        selected_row = int((np.uint64(3) ^ np.uint64(2)) % np.uint64(rows))
        expected = np.full(head_dim, -16.0 * float(scales[selected_row]), dtype=np.float32)
        np.testing.assert_array_equal(output[0], expected)

    def test_fp16_conv_provider_does_not_interpret_weights_as_bf16(self) -> None:
        rng = np.random.default_rng(17)
        rows, streams, hidden, kernel, dilation = 2, 2, 8, 2, 2
        channels = streams * hidden
        history = (kernel - 1) * dilation
        hyper_input = rng.standard_normal((rows, channels), dtype=np.float32)
        key = rng.standard_normal((rows, channels), dtype=np.float32)
        value = rng.standard_normal((rows, hidden), dtype=np.float32)
        norms = [np.ones(channels, dtype=np.float32) for _ in range(3)]
        weights = np.full((channels, kernel), 0.125, dtype=np.float32)

        def run(function: object, weight_bits: np.ndarray) -> np.ndarray:
            output = np.zeros((rows, channels), dtype=np.float32)
            scratch = [np.zeros_like(output) for _ in range(4)]
            state_in = np.zeros((history, channels), dtype=np.float32)
            state_out = np.zeros_like(state_in)
            function(
                ptr(hyper_input, F32P), ptr(key, F32P), ptr(value, F32P),
                *(ptr(item, F32P) for item in norms), ptr(weight_bits, U16P),
                ptr(output, F32P), *(ptr(item, F32P) for item in scratch),
                ptr(state_in, F32P), ptr(state_out, F32P), rows, streams,
                hidden, kernel, dilation, ctypes.c_float(1.0e-6),
            )
            return output

        bf16_weight = bf16_bits(torch.from_numpy(weights))
        fp16_weight = weights.astype(np.float16).view(np.uint16)
        expected = run(LIB.qwen4_ple_gate_conv_inject_bf16, bf16_weight)
        actual = run(LIB.qwen4_ple_gate_conv_inject_fp16, fp16_weight)
        np.testing.assert_array_equal(actual, expected)

    def test_llama_provider_uses_grouped_llama_rmsnorm(self) -> None:
        rng = np.random.default_rng(29)
        rows, streams, hidden, kernel, dilation = 2, 4, 32, 3, 2
        channels = streams * hidden
        history = (kernel - 1) * dilation
        hyper_input = rng.standard_normal((rows, channels), dtype=np.float32)
        key = rng.standard_normal((rows, channels), dtype=np.float32)
        value = rng.standard_normal((rows, hidden), dtype=np.float32)
        norms = [rng.standard_normal(channels, dtype=np.float32) for _ in range(3)]
        weights = rng.standard_normal((channels, kernel), dtype=np.float32)
        fp16_weight = weights.astype(np.float16).view(np.uint16)
        output = np.zeros((rows, channels), dtype=np.float32)
        scratch = [np.zeros_like(output) for _ in range(4)]
        state_in = np.zeros((history, channels), dtype=np.float32)
        state_out = np.zeros_like(state_in)

        LIB.qwen4_ple_gate_conv_inject_llama_fp16(
            ptr(hyper_input, F32P), ptr(key, F32P), ptr(value, F32P),
            *(ptr(item, F32P) for item in norms), ptr(fp16_weight, U16P),
            ptr(output, F32P), *(ptr(item, F32P) for item in scratch),
            ptr(state_in, F32P), ptr(state_out, F32P), rows, streams,
            hidden, kernel, dilation, ctypes.c_float(1.0e-6),
        )

        expected_key = np.zeros_like(key)
        expected_query = np.zeros_like(hyper_input)
        LIB.rmsnorm_forward_llama_production.argtypes = [
            F32P, F32P, F32P, F32P, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_float,
        ]
        for row in range(rows):
            for stream in range(streams):
                span = slice(stream * hidden, (stream + 1) * hidden)
                LIB.rmsnorm_forward_llama_production(
                    ptr(key[row, span], F32P), ptr(norms[0][span], F32P),
                    ptr(expected_key[row, span], F32P), None, 1, hidden,
                    hidden, ctypes.c_float(1.0e-6),
                )
                LIB.rmsnorm_forward_llama_production(
                    ptr(hyper_input[row, span], F32P), ptr(norms[1][span], F32P),
                    ptr(expected_query[row, span], F32P), None, 1, hidden,
                    hidden, ctypes.c_float(1.0e-6),
                )

        np.testing.assert_array_equal(scratch[0], expected_key)
        np.testing.assert_array_equal(scratch[1], expected_query)
        self.assertTrue(np.isfinite(output).all())

    def test_llama_provider_keeps_convolution_multiply_and_add_separate(self) -> None:
        rng = np.random.default_rng(41)
        rows, streams, hidden, kernel, dilation = 1, 1, 32, 4, 3
        channels = streams * hidden
        history = (kernel - 1) * dilation
        hyper_input = rng.standard_normal((rows, channels), dtype=np.float32)
        key = rng.standard_normal((rows, channels), dtype=np.float32)
        value = rng.standard_normal((rows, hidden), dtype=np.float32)
        norms = [rng.standard_normal(channels, dtype=np.float32) for _ in range(3)]
        weights = rng.standard_normal((channels, kernel), dtype=np.float32).astype(np.float16)
        weight_bits = weights.view(np.uint16)
        output = np.zeros((rows, channels), dtype=np.float32)
        scratch = [np.zeros_like(output) for _ in range(4)]
        state_in = np.zeros((history, channels), dtype=np.float32)
        state_in[history - dilation] = rng.standard_normal(channels, dtype=np.float32)
        state_out = np.zeros_like(state_in)

        LIB.qwen4_ple_gate_conv_inject_llama_fp16(
            ptr(hyper_input, F32P), ptr(key, F32P), ptr(value, F32P),
            *(ptr(item, F32P) for item in norms), ptr(weight_bits, U16P),
            ptr(output, F32P), *(ptr(item, F32P) for item in scratch),
            ptr(state_in, F32P), ptr(state_out, F32P), rows, streams,
            hidden, kernel, dilation, ctypes.c_float(1.0e-6),
        )

        old_term = np.multiply(
            state_in[history - dilation], weights[:, kernel - 2], dtype=np.float32,
        )
        current_term = np.multiply(
            scratch[3][0], weights[:, kernel - 1], dtype=np.float32,
        )
        conv_raw = np.add(old_term, current_term, dtype=np.float32)
        conv_silu = np.empty_like(conv_raw)
        LIB.recurrent_silu_forward_ggml(
            ptr(conv_raw, F32P), ptr(conv_silu, F32P), 1, channels,
        )
        expected = np.add(
            hyper_input[0], np.add(scratch[2][0], conv_silu, dtype=np.float32),
            dtype=np.float32,
        )
        np.testing.assert_array_equal(output[0], expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
