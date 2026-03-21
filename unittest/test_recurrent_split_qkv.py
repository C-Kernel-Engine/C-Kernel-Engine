#!/usr/bin/env python3
"""
PyTorch parity test for recurrent_split_qkv.

This covers:
- forward parity: q, k, v
- backward parity: d_packed_qkv

Run directly:
    python3 unittest/test_recurrent_split_qkv.py
"""

from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - dependency skip
    print(f"[SKIP] torch not available: {exc}")
    sys.exit(0)


ROOT = Path(__file__).resolve().parents[1]
LIB_CANDIDATES = [
    ROOT / "build" / "libckernel_engine.so",
    ROOT / "libckernel_engine.so",
]


def _load_lib() -> ctypes.CDLL | None:
    for path in LIB_CANDIDATES:
        if path.exists():
            lib = ctypes.CDLL(str(path))
            fwd = lib.recurrent_split_qkv_forward
            fwd.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            fwd.restype = None

            bwd = lib.recurrent_split_qkv_backward
            bwd.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            bwd.restype = None
            return lib
    return None


LIB = _load_lib()
if LIB is None:  # pragma: no cover - dependency skip
    print("[SKIP] libckernel_engine.so not found")
    sys.exit(0)


def _as_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class TestRecurrentSplitQKVParity(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.atol = 0.0

    def _run_case(self, rows: int, q_dim: int, k_dim: int, v_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        packed = (0.25 * rng.standard_normal((rows, q_dim + k_dim + v_dim))).astype(np.float32)
        d_q = (0.20 * rng.standard_normal((rows, q_dim))).astype(np.float32)
        d_k = (0.20 * rng.standard_normal((rows, k_dim))).astype(np.float32)
        d_v = (0.20 * rng.standard_normal((rows, v_dim))).astype(np.float32)

        ck_q = np.zeros((rows, q_dim), dtype=np.float32)
        ck_k = np.zeros((rows, k_dim), dtype=np.float32)
        ck_v = np.zeros((rows, v_dim), dtype=np.float32)
        LIB.recurrent_split_qkv_forward(
            _as_ptr(packed),
            _as_ptr(ck_q),
            _as_ptr(ck_k),
            _as_ptr(ck_v),
            rows,
            q_dim,
            k_dim,
            v_dim,
        )

        ck_d_packed = np.zeros_like(packed)
        LIB.recurrent_split_qkv_backward(
            _as_ptr(d_q),
            _as_ptr(d_k),
            _as_ptr(d_v),
            _as_ptr(ck_d_packed),
            rows,
            q_dim,
            k_dim,
            v_dim,
        )

        t_packed = torch.tensor(packed, dtype=torch.float32, requires_grad=True)
        t_q, t_k, t_v = torch.split(t_packed, [q_dim, k_dim, v_dim], dim=1)

        torch_q = t_q.detach().numpy()
        torch_k = t_k.detach().numpy()
        torch_v = t_v.detach().numpy()

        t_q.backward(torch.tensor(d_q), retain_graph=True)
        t_k.backward(torch.tensor(d_k), retain_graph=True)
        t_v.backward(torch.tensor(d_v))
        torch_d_packed = t_packed.grad.detach().numpy()

        np.testing.assert_allclose(ck_q, torch_q, atol=self.atol, rtol=0.0)
        np.testing.assert_allclose(ck_k, torch_k, atol=self.atol, rtol=0.0)
        np.testing.assert_allclose(ck_v, torch_v, atol=self.atol, rtol=0.0)
        np.testing.assert_allclose(ck_d_packed, torch_d_packed, atol=self.atol, rtol=0.0)

    def test_small_case(self) -> None:
        self._run_case(rows=3, q_dim=8, k_dim=8, v_dim=16, seed=7)

    def test_medium_case(self) -> None:
        self._run_case(rows=9, q_dim=32, k_dim=32, v_dim=64, seed=13)

    def test_qwen35_like_case(self) -> None:
        self._run_case(rows=5, q_dim=2048, k_dim=2048, v_dim=2048, seed=23)


if __name__ == "__main__":
    unittest.main(verbosity=2)
