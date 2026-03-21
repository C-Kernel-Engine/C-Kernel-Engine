#!/usr/bin/env python3
"""
PyTorch parity test for recurrent_dt_gate.

This covers:
- forward parity: gate
- backward parity: d_alpha, d_dt_bias, d_a

Run directly:
    python3 unittest/test_recurrent_dt_gate.py
"""

from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
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
            fwd = lib.recurrent_dt_gate_forward
            fwd.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
            ]
            fwd.restype = None

            bwd = lib.recurrent_dt_gate_backward
            bwd.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
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


class TestRecurrentDTGateParity(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.atol_forward = 2e-6
        self.atol_backward = 5e-5

    def _run_case(self, rows: int, dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        alpha = (0.30 * rng.standard_normal((rows, dim))).astype(np.float32)
        dt_bias = (0.20 * rng.standard_normal(dim)).astype(np.float32)
        a = (0.25 * rng.standard_normal(dim)).astype(np.float32)
        d_gate = (0.35 * rng.standard_normal((rows, dim))).astype(np.float32)

        ck_gate = np.zeros((rows, dim), dtype=np.float32)
        LIB.recurrent_dt_gate_forward(
            _as_ptr(alpha),
            _as_ptr(dt_bias),
            _as_ptr(a),
            _as_ptr(ck_gate),
            rows,
            dim,
        )

        ck_d_alpha = np.zeros_like(alpha)
        ck_d_dt_bias = np.zeros_like(dt_bias)
        ck_d_a = np.zeros_like(a)
        LIB.recurrent_dt_gate_backward(
            _as_ptr(d_gate),
            _as_ptr(alpha),
            _as_ptr(dt_bias),
            _as_ptr(a),
            _as_ptr(ck_d_alpha),
            _as_ptr(ck_d_dt_bias),
            _as_ptr(ck_d_a),
            rows,
            dim,
        )

        t_alpha = torch.tensor(alpha, dtype=torch.float32, requires_grad=True)
        t_dt_bias = torch.tensor(dt_bias, dtype=torch.float32, requires_grad=True)
        t_a = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        t_d_gate = torch.tensor(d_gate, dtype=torch.float32)

        t_gate = F.softplus(t_alpha + t_dt_bias) * t_a
        t_gate.backward(t_d_gate)

        torch_gate = t_gate.detach().numpy()
        torch_d_alpha = t_alpha.grad.detach().numpy()
        torch_d_dt_bias = t_dt_bias.grad.detach().numpy()
        torch_d_a = t_a.grad.detach().numpy()

        np.testing.assert_allclose(ck_gate, torch_gate, atol=self.atol_forward, rtol=0.0)
        np.testing.assert_allclose(ck_d_alpha, torch_d_alpha, atol=self.atol_backward, rtol=0.0)
        np.testing.assert_allclose(ck_d_dt_bias, torch_d_dt_bias, atol=self.atol_backward, rtol=0.0)
        np.testing.assert_allclose(ck_d_a, torch_d_a, atol=self.atol_backward, rtol=0.0)

    def test_small_case(self) -> None:
        self._run_case(rows=4, dim=8, seed=5)

    def test_medium_case(self) -> None:
        self._run_case(rows=9, dim=32, seed=17)

    def test_qwen35_like_case(self) -> None:
        self._run_case(rows=7, dim=16, seed=29)


if __name__ == "__main__":
    unittest.main(verbosity=2)
