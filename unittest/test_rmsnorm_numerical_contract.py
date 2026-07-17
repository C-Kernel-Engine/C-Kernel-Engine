#!/usr/bin/env python3
"""Concrete oracle for the FP32-square/FP64-sum RMSNorm contract."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def reference(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    result = np.empty_like(x)
    eps32 = np.float32(eps)
    for row_index, row in enumerate(x):
        total = 0.0
        for value in row:
            square = np.float32(value * value)
            total += float(square)
        mean = np.float32(total / row.size)
        rstd = np.float32(np.float32(1.0) / np.sqrt(np.float32(mean + eps32)))
        for column, value in enumerate(row):
            normalized = np.float32(value * rstd)
            result[row_index, column] = np.float32(normalized * gamma[column])
    return result


def main() -> int:
    lib = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
    fn = lib.rmsnorm_forward_fp64_sum
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    qk_fn = lib.qk_norm_forward_fp64_sum
    qk_fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]

    rng = np.random.default_rng(20260714)
    for rows, width in ((3, 128), (2, 4096)):
        x = rng.normal(0.0, 0.35, size=(rows, width)).astype(np.float32)
        gamma = rng.normal(1.0, 0.1, size=width).astype(np.float32)
        expected = reference(x, gamma, 1.0e-6)
        actual = np.empty_like(x)
        fn(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            None,
            rows,
            width,
            width,
            ctypes.c_float(1.0e-6),
        )
        if not np.array_equal(actual, expected):
            diff = float(np.max(np.abs(actual - expected)))
            raise AssertionError(f"RMSNorm contract mismatch rows={rows} width={width} max_diff={diff}")
        print(f"rmsnorm_fp64_sum_{rows}x{width} max_diff=0 tol=0 [PASS]")

    q = rng.normal(0.0, 0.25, size=(4, 128)).astype(np.float32)
    k = rng.normal(0.0, 0.25, size=(2, 128)).astype(np.float32)
    q_gamma = rng.normal(1.0, 0.1, size=128).astype(np.float32)
    k_gamma = rng.normal(1.0, 0.1, size=128).astype(np.float32)
    expected_q = reference(q.copy(), q_gamma, 1.0e-6)
    expected_k = reference(k.copy(), k_gamma, 1.0e-6)
    qk_fn(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q_gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k_gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        4,
        2,
        1,
        128,
        ctypes.c_float(1.0e-6),
    )
    if not np.array_equal(q, expected_q) or not np.array_equal(k, expected_k):
        raise AssertionError("Q/K RMSNorm contract wrapper changed arithmetic")
    print("qk_norm_fp64_sum_decode max_diff=0 tol=0 [PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
