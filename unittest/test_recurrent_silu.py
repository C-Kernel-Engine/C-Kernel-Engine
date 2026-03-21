#!/usr/bin/env python3
"""PyTorch parity test for recurrent_silu."""

from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    print(f"[SKIP] torch not available: {exc}")
    sys.exit(0)


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so")) if (ROOT / "build" / "libckernel_engine.so").exists() else None
if LIB is None:  # pragma: no cover
    print("[SKIP] libckernel_engine.so not found")
    sys.exit(0)

LIB.recurrent_silu_forward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
LIB.recurrent_silu_backward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]


def _as_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class TestRecurrentSilu(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def _run_case(self, rows: int, dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = (0.25 * rng.standard_normal((rows, dim))).astype(np.float32)
        d_out = (0.20 * rng.standard_normal((rows, dim))).astype(np.float32)
        ck_out = np.zeros_like(x)
        ck_dx = np.zeros_like(x)
        LIB.recurrent_silu_forward(_as_ptr(x), _as_ptr(ck_out), rows, dim)
        LIB.recurrent_silu_backward(_as_ptr(d_out), _as_ptr(x), _as_ptr(ck_dx), rows, dim)

        tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        tout = F.silu(tx)
        tout.backward(torch.tensor(d_out))

        np.testing.assert_allclose(ck_out, tout.detach().numpy(), atol=5e-6, rtol=0.0)
        np.testing.assert_allclose(ck_dx, tx.grad.detach().numpy(), atol=5e-5, rtol=0.0)

    def test_small_case(self) -> None:
        self._run_case(7, 64, 5)

    def test_medium_case(self) -> None:
        self._run_case(11, 128, 17)


if __name__ == "__main__":
    unittest.main(verbosity=2)
