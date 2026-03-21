#!/usr/bin/env python3
"""PyTorch parity test for recurrent_split_conv_qkv."""

from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"[SKIP] torch not available: {exc}")
    sys.exit(0)


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so")) if (ROOT / "build" / "libckernel_engine.so").exists() else None
if LIB is None:  # pragma: no cover
    print("[SKIP] libckernel_engine.so not found")
    sys.exit(0)

LIB.recurrent_split_conv_qkv_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
LIB.recurrent_split_conv_qkv_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


def _as_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class TestRecurrentSplitConvQKV(unittest.TestCase):
    def _run_case(self, rows: int, q_dim: int, k_dim: int, v_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        packed = (0.25 * rng.standard_normal((rows, q_dim + k_dim + v_dim))).astype(np.float32)
        d_q = (0.20 * rng.standard_normal((rows, q_dim))).astype(np.float32)
        d_k = (0.20 * rng.standard_normal((rows, k_dim))).astype(np.float32)
        d_v = (0.20 * rng.standard_normal((rows, v_dim))).astype(np.float32)
        ck_q = np.zeros((rows, q_dim), dtype=np.float32)
        ck_k = np.zeros((rows, k_dim), dtype=np.float32)
        ck_v = np.zeros((rows, v_dim), dtype=np.float32)
        ck_d_packed = np.zeros_like(packed)

        LIB.recurrent_split_conv_qkv_forward(_as_ptr(packed), _as_ptr(ck_q), _as_ptr(ck_k), _as_ptr(ck_v), rows, q_dim, k_dim, v_dim)
        LIB.recurrent_split_conv_qkv_backward(_as_ptr(d_q), _as_ptr(d_k), _as_ptr(d_v), _as_ptr(ck_d_packed), rows, q_dim, k_dim, v_dim)

        t_packed = torch.tensor(packed, dtype=torch.float32)
        tq, tk, tv = torch.split(t_packed, [q_dim, k_dim, v_dim], dim=1)
        torch_d_packed = torch.cat([torch.tensor(d_q), torch.tensor(d_k), torch.tensor(d_v)], dim=1).numpy()

        np.testing.assert_allclose(ck_q, tq.numpy(), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(ck_k, tk.numpy(), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(ck_v, tv.numpy(), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(ck_d_packed, torch_d_packed, atol=0.0, rtol=0.0)

    def test_small_case(self) -> None:
        self._run_case(5, 64, 64, 128, 3)

    def test_medium_case(self) -> None:
        self._run_case(9, 128, 128, 256, 11)


if __name__ == "__main__":
    unittest.main(verbosity=2)
