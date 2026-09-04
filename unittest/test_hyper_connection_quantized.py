#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import struct
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(str(ROOT / "build/libckernel_engine.so"))
F32P = ctypes.POINTER(ctypes.c_float)
QK_K = 256
Q4_K_BYTES = 144
Q5_0_BYTES = 22
Q6_K_BYTES = 210


def _f32(value: np.ndarray) -> F32P:
    return value.ctypes.data_as(F32P)


def _packed_blocks(
    count: int,
    size: int,
    seed: int,
    scale_count: int,
    scale_offset: int = 0,
) -> ctypes.Array:
    rng = np.random.default_rng(seed)
    raw = bytearray(rng.integers(0, 256, size=count * size, dtype=np.uint8).tobytes())
    for block in range(count):
        offset = block * size
        for scale in range(scale_count):
            struct.pack_into(
                "<H", raw, offset + scale_offset + scale * 2, 0x211F + scale
            )
    return ctypes.create_string_buffer(bytes(raw), len(raw))


class QuantizedHyperConnectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        LIB.hyper_stream_expand_f32.argtypes = [
            F32P, F32P, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        composite_args = [
            F32P, F32P, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            F32P, F32P, F32P, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_int,
        ]
        LIB.hyper_connection_mix_q4k_q5_0_q4k.argtypes = composite_args
        LIB.hyper_connection_mix_q6k_q5_0_q4k.argtypes = composite_args
        LIB.gemv_q4_k.argtypes = [F32P, ctypes.c_void_p, F32P, ctypes.c_int, ctypes.c_int]
        LIB.gemv_q5_0.argtypes = [F32P, ctypes.c_void_p, F32P, ctypes.c_int, ctypes.c_int]
        LIB.quantize_row_q8_k.argtypes = [F32P, ctypes.c_void_p, ctypes.c_int]
        LIB.quantize_row_q8_0.argtypes = [F32P, ctypes.c_void_p, ctypes.c_int]
        LIB.recurrent_silu_forward_ggml.argtypes = [
            F32P, F32P, ctypes.c_int, ctypes.c_int,
        ]
        LIB.recurrent_sigmoid_forward_ggml.argtypes = [
            F32P, F32P, ctypes.c_int, ctypes.c_int,
        ]
        LIB.gemv_q4_k_q8_k.argtypes = [F32P, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        LIB.gemv_q4_k_q8_k_avx2.argtypes = [F32P, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        LIB.gemv_q5_0_q8_0.argtypes = [F32P, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        LIB.gemv_q4_k_q8_k_repacked_parallel_dispatch.argtypes = [
            F32P, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ]
        LIB.gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

    def test_f32_stream_expand_preserves_quantized_embedding_values(self) -> None:
        rows, streams, hidden = 2, 4, 256
        rng = np.random.default_rng(0x4839)
        source = np.ascontiguousarray(
            rng.normal(0, 0.2, (rows, hidden)), dtype=np.float32
        )
        expanded = np.empty((rows, streams * hidden), dtype=np.float32)
        LIB.hyper_stream_expand_f32(
            _f32(source), _f32(expanded), rows, streams, hidden
        )
        np.testing.assert_array_equal(
            expanded.reshape(rows, streams, hidden),
            np.repeat(source[:, None, :], streams, axis=1),
        )

    def test_q4k_q5_0_q4k_matches_declared_composition(self) -> None:
        rows, streams, hidden, dynamic = 2, 1, QK_K, 64
        hyper_dim = streams * hidden
        rng = np.random.default_rng(0x4838)
        hyper = np.ascontiguousarray(rng.normal(0, 0.2, (rows, hyper_dim)), dtype=np.float32)
        norm_weight = np.ascontiguousarray(rng.normal(1, 0.03, hyper_dim), dtype=np.float32)
        down = _packed_blocks(dynamic, Q4_K_BYTES, 1, 2)
        up = _packed_blocks(hyper_dim * (dynamic // 32), Q5_0_BYTES, 2, 1)
        inject = _packed_blocks(streams, Q4_K_BYTES, 3, 2)

        mixed = np.empty((rows, hidden), dtype=np.float32)
        injection = np.empty((rows, streams), dtype=np.float32)
        normalized = np.empty((rows, hyper_dim), dtype=np.float32)
        dynamic_scratch = np.empty((rows, dynamic), dtype=np.float32)
        mix_scratch = np.empty((rows, hyper_dim), dtype=np.float32)
        LIB.hyper_connection_mix_q4k_q5_0_q4k(
            _f32(hyper), _f32(norm_weight), down, up, inject,
            _f32(mixed), _f32(injection), _f32(normalized),
            _f32(dynamic_scratch), _f32(mix_scratch),
            rows, streams, hidden, dynamic, ctypes.c_float(1e-6), 1,
        )

        expected_norm = np.empty_like(normalized)
        expected_mixed = np.empty_like(mixed)
        expected_injection = np.empty_like(injection)
        for row in range(rows):
            grouped = hyper[row].reshape(streams, hidden)
            rstd = 1.0 / np.sqrt(np.mean(grouped * grouped, axis=1, keepdims=True) + 1e-6)
            norm = (grouped * rstd * norm_weight.reshape(streams, hidden)).reshape(-1).astype(np.float32)
            expected_norm[row] = norm
            lo = np.empty(dynamic, dtype=np.float32)
            norm_q8 = ctypes.create_string_buffer((hyper_dim // 256) * 292)
            LIB.quantize_row_q8_k(_f32(normalized[row]), norm_q8, hyper_dim)
            LIB.gemv_q4_k_q8_k_repacked_parallel_dispatch(
                _f32(lo), down, norm_q8, dynamic, hyper_dim
            )
            lo /= streams
            lo = np.ascontiguousarray(lo, dtype=np.float32)
            LIB.recurrent_silu_forward_ggml(_f32(lo), _f32(lo), 1, dynamic)
            gate = np.empty(hyper_dim, dtype=np.float32)
            lo_q8 = ctypes.create_string_buffer((dynamic // 32) * 34)
            LIB.quantize_row_q8_0(_f32(lo), lo_q8, dynamic)
            LIB.gemv_q5_0_q8_0(_f32(gate), up, lo_q8, hyper_dim, dynamic)
            LIB.recurrent_sigmoid_forward_ggml(
                _f32(gate), _f32(gate), 1, hyper_dim
            )
            expected_mixed[row] = (
                normalized[row].reshape(streams, hidden)
                * gate.reshape(streams, hidden)
            ).mean(axis=0)
            raw_inject = np.empty(streams, dtype=np.float32)
            LIB.gemv_q4_k_q8_k_avx2(
                _f32(raw_inject), inject, norm_q8, streams, hyper_dim
            )
            raw_inject /= streams
            LIB.recurrent_sigmoid_forward_ggml(
                _f32(raw_inject), _f32(raw_inject), 1, streams
            )
            expected_injection[row] = raw_inject * 2.0

            np.testing.assert_array_equal(mix_scratch[row], gate)

        np.testing.assert_allclose(normalized, expected_norm, atol=3e-6, rtol=2e-6)
        np.testing.assert_allclose(mixed, expected_mixed, atol=3e-6, rtol=2e-6)
        np.testing.assert_array_equal(injection, expected_injection)

    def test_q6k_down_uses_declared_q4k_injection_provider(self) -> None:
        rows, streams, hidden, dynamic = 1, 4, QK_K, 64
        hyper_dim = streams * hidden
        rng = np.random.default_rng(0x483A)
        hyper = np.ascontiguousarray(
            rng.normal(0, 0.2, (rows, hyper_dim)), dtype=np.float32
        )
        norm_weight = np.ascontiguousarray(
            rng.normal(1, 0.03, hyper_dim), dtype=np.float32
        )
        down = ctypes.create_string_buffer(
            dynamic * (hyper_dim // QK_K) * Q6_K_BYTES
        )
        up = ctypes.create_string_buffer(
            hyper_dim * (dynamic // 32) * Q5_0_BYTES
        )
        inject = _packed_blocks(streams * (hyper_dim // QK_K), Q4_K_BYTES, 6, 2)

        mixed = np.empty((rows, hidden), dtype=np.float32)
        injection = np.empty((rows, streams), dtype=np.float32)
        normalized = np.empty((rows, hyper_dim), dtype=np.float32)
        dynamic_scratch = np.empty((rows, dynamic), dtype=np.float32)
        mix_scratch = np.empty((rows, hyper_dim), dtype=np.float32)
        LIB.hyper_connection_mix_q6k_q5_0_q4k(
            _f32(hyper), _f32(norm_weight), down, up, inject,
            _f32(mixed), _f32(injection), _f32(normalized),
            _f32(dynamic_scratch), _f32(mix_scratch),
            rows, streams, hidden, dynamic, ctypes.c_float(1e-6), 1,
        )

        norm_q8 = ctypes.create_string_buffer((hyper_dim // QK_K) * 292)
        LIB.quantize_row_q8_k(_f32(normalized[0]), norm_q8, hyper_dim)
        raw_injection = np.empty(streams, dtype=np.float32)
        LIB.gemv_q4_k_q8_k_avx2(
            _f32(raw_injection), inject, norm_q8, streams, hyper_dim
        )
        expected = 2.0 / (1.0 + np.exp(-(raw_injection / streams)))

        self.assertTrue(np.isfinite(mixed).all())
        self.assertTrue(np.isfinite(injection).all())
        np.testing.assert_allclose(injection[0], expected, atol=3e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
