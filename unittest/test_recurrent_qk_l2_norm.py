#!/usr/bin/env python3
"""
PyTorch parity test for recurrent_qk_l2_norm.
"""

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
LIB_CANDIDATES = [
    ROOT / "build" / "libckernel_engine.so",
    ROOT / "libckernel_engine.so",
]


def _load_lib() -> ctypes.CDLL | None:
    for path in LIB_CANDIDATES:
        if path.exists():
            lib = ctypes.CDLL(str(path))
            fwd = lib.recurrent_qk_l2_norm_forward
            fwd.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]
            fwd.restype = None

            bwd = lib.recurrent_qk_l2_norm_backward
            bwd.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
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
            bwd.restype = None
            return lib
    return None


LIB = _load_lib()
if LIB is None:  # pragma: no cover
    print("[SKIP] libckernel_engine.so not found")
    sys.exit(0)


def _as_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _torch_ref(x: torch.Tensor, dim: int, head_dim: int, eps: float) -> torch.Tensor:
    num_heads = dim // head_dim
    shaped = x.view(x.shape[0], num_heads, head_dim)
    denom = torch.sqrt((shaped * shaped).sum(dim=-1, keepdim=True) + eps)
    return (shaped / denom).view(x.shape[0], dim)


class TestRecurrentQKL2Norm(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.atol = 5e-5

    def _run_case(self, rows: int, q_dim: int, k_dim: int, head_dim: int, eps: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        q = (0.20 * rng.standard_normal((rows, q_dim))).astype(np.float32)
        k = (0.20 * rng.standard_normal((rows, k_dim))).astype(np.float32)

        ck_q = q.copy()
        ck_k = k.copy()
        LIB.recurrent_qk_l2_norm_forward(_as_ptr(ck_q), _as_ptr(ck_k), rows, q_dim, k_dim, head_dim, eps)

        d_q_out = (0.25 * rng.standard_normal((rows, q_dim))).astype(np.float32)
        d_k_out = (0.25 * rng.standard_normal((rows, k_dim))).astype(np.float32)
        ck_d_q = np.zeros_like(q)
        ck_d_k = np.zeros_like(k)
        LIB.recurrent_qk_l2_norm_backward(
            _as_ptr(d_q_out),
            _as_ptr(d_k_out),
            _as_ptr(q),
            _as_ptr(k),
            _as_ptr(ck_d_q),
            _as_ptr(ck_d_k),
            rows,
            q_dim,
            k_dim,
            head_dim,
            eps,
        )

        t_q = torch.tensor(q, dtype=torch.float32, requires_grad=True)
        t_k = torch.tensor(k, dtype=torch.float32, requires_grad=True)
        t_q_norm = _torch_ref(t_q, q_dim, head_dim, eps)
        t_k_norm = _torch_ref(t_k, k_dim, head_dim, eps)
        t_q_norm.backward(torch.tensor(d_q_out), retain_graph=True)
        torch_d_q = t_q.grad.detach().clone().numpy()
        t_q.grad.zero_()
        t_k_norm.backward(torch.tensor(d_k_out))
        torch_d_k = t_k.grad.detach().clone().numpy()

        np.testing.assert_allclose(ck_q, t_q_norm.detach().numpy(), atol=self.atol, rtol=0.0)
        np.testing.assert_allclose(ck_k, t_k_norm.detach().numpy(), atol=self.atol, rtol=0.0)
        np.testing.assert_allclose(ck_d_q, torch_d_q, atol=self.atol, rtol=0.0)
        np.testing.assert_allclose(ck_d_k, torch_d_k, atol=self.atol, rtol=0.0)

    def test_small_case(self) -> None:
        self._run_case(rows=4, q_dim=32, k_dim=32, head_dim=8, eps=1e-5, seed=7)

    def test_qwen35_like_case(self) -> None:
        self._run_case(rows=3, q_dim=2048, k_dim=2048, head_dim=128, eps=1e-5, seed=11)


if __name__ == "__main__":
    unittest.main(verbosity=2)
