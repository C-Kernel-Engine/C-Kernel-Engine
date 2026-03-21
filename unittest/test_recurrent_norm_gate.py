#!/usr/bin/env python3
"""PyTorch parity test for recurrent_norm_gate."""

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

LIB.recurrent_norm_gate_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
LIB.recurrent_norm_gate_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]


def _as_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _torch_ref(x, gate, weight, num_heads, head_dim, eps):
    rows = x.shape[0]
    xh = x.view(rows, num_heads, head_dim)
    gh = gate.view(rows, num_heads, head_dim)
    rms = torch.rsqrt(xh.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = xh * rms * weight.view(1, 1, head_dim)
    return (normed * F.silu(gh)).reshape(rows, num_heads * head_dim)


class TestRecurrentNormGate(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def _run_case(self, rows: int, num_heads: int, head_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        dim = num_heads * head_dim
        x = (0.25 * rng.standard_normal((rows, dim))).astype(np.float32)
        gate = (0.25 * rng.standard_normal((rows, dim))).astype(np.float32)
        weight = (0.20 * rng.standard_normal(head_dim)).astype(np.float32)
        d_out = (0.20 * rng.standard_normal((rows, dim))).astype(np.float32)
        eps = 1e-6

        ck_out = np.zeros_like(x)
        ck_dx = np.zeros_like(x)
        ck_dg = np.zeros_like(gate)
        ck_dw = np.zeros_like(weight)
        LIB.recurrent_norm_gate_forward(_as_ptr(x), _as_ptr(gate), _as_ptr(weight), _as_ptr(ck_out), rows, num_heads, head_dim, ctypes.c_float(eps))
        LIB.recurrent_norm_gate_backward(_as_ptr(d_out), _as_ptr(x), _as_ptr(gate), _as_ptr(weight), _as_ptr(ck_dx), _as_ptr(ck_dg), _as_ptr(ck_dw), rows, num_heads, head_dim, ctypes.c_float(eps))

        tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        tg = torch.tensor(gate, dtype=torch.float32, requires_grad=True)
        tw = torch.tensor(weight, dtype=torch.float32, requires_grad=True)
        tout = _torch_ref(tx, tg, tw, num_heads, head_dim, eps)
        tout.backward(torch.tensor(d_out))

        np.testing.assert_allclose(ck_out, tout.detach().numpy(), atol=5e-5, rtol=0.0)
        np.testing.assert_allclose(ck_dx, tx.grad.detach().numpy(), atol=8e-5, rtol=0.0)
        np.testing.assert_allclose(ck_dg, tg.grad.detach().numpy(), atol=8e-5, rtol=0.0)
        np.testing.assert_allclose(ck_dw, tw.grad.detach().numpy(), atol=8e-5, rtol=0.0)

    def test_small_case(self) -> None:
        self._run_case(4, 8, 16, 19)

    def test_qwen35_like_case(self) -> None:
        self._run_case(5, 16, 128, 31)


if __name__ == "__main__":
    unittest.main(verbosity=2)
