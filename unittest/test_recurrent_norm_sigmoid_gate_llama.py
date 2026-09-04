#!/usr/bin/env python3
"""Numerical contract for llama-compatible sigmoid-gated recurrent RMSNorm."""

from __future__ import annotations

import ctypes
import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "build" / "libckernel_engine.so"
if not LIB_PATH.exists():  # pragma: no cover
    print("[SKIP] libckernel_engine.so not found")
    sys.exit(0)
LIB = ctypes.CDLL(str(LIB_PATH))
LIB.recurrent_norm_sigmoid_gate_llama_avx2_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]


def _as_ptr(values: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class RecurrentNormSigmoidGateLlamaTests(unittest.TestCase):
    def test_fp32_sigmoid_gate_contract(self) -> None:
        rows, heads, dim = 3, 4, 128
        rng = np.random.default_rng(47)
        x = (0.3 * rng.standard_normal((rows, heads, dim))).astype(np.float32)
        gate = (0.5 * rng.standard_normal((rows, heads, dim))).astype(np.float32)
        weight = (0.2 * rng.standard_normal(dim)).astype(np.float32)
        actual = np.empty_like(x)
        eps = np.float32(1e-6)

        LIB.recurrent_norm_sigmoid_gate_llama_avx2_forward(
            _as_ptr(x),
            _as_ptr(gate),
            _as_ptr(weight),
            _as_ptr(actual),
            rows,
            heads,
            dim,
            ctypes.c_float(eps),
        )

        expected = np.empty_like(x)
        for row in range(rows):
            for head in range(heads):
                source = x[row, head]
                sum_sq = 0.0
                for value in source:
                    square = np.float32(value * value)
                    sum_sq += float(square)
                mean = np.float32(sum_sq / dim)
                rstd = np.float32(1.0 / math.sqrt(float(np.float32(mean + eps))))
                for col in range(dim):
                    normalized = np.float32(
                        np.float32(source[col] * rstd) * weight[col]
                    )
                    sigmoid = np.float32(
                        1.0 / (1.0 + math.exp(float(-gate[row, head, col])))
                    )
                    expected[row, head, col] = np.float32(normalized * sigmoid)

        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
