#!/usr/bin/env python3
"""
PyTorch parity test for the Gated DeltaNet recurrent kernel.

This covers both:
- forward parity: state_out + out
- backward parity: d_q, d_k, d_v, d_g, d_beta, d_state_in

Run directly:
    python3 tests/test_deltanet.py
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
            fwd = lib.gated_deltanet_autoregressive_forward
            fwd.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # q
                ctypes.POINTER(ctypes.c_float),  # k
                ctypes.POINTER(ctypes.c_float),  # v
                ctypes.POINTER(ctypes.c_float),  # g
                ctypes.POINTER(ctypes.c_float),  # beta
                ctypes.POINTER(ctypes.c_float),  # state_in
                ctypes.POINTER(ctypes.c_float),  # state_out
                ctypes.POINTER(ctypes.c_float),  # out
                ctypes.c_int,                    # num_heads
                ctypes.c_int,                    # state_dim
                ctypes.c_float,                  # norm_eps
            ]
            fwd.restype = None

            bwd = lib.gated_deltanet_autoregressive_backward
            bwd.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # d_out
                ctypes.POINTER(ctypes.c_float),  # d_state_out
                ctypes.POINTER(ctypes.c_float),  # q
                ctypes.POINTER(ctypes.c_float),  # k
                ctypes.POINTER(ctypes.c_float),  # v
                ctypes.POINTER(ctypes.c_float),  # g
                ctypes.POINTER(ctypes.c_float),  # beta
                ctypes.POINTER(ctypes.c_float),  # state_in
                ctypes.POINTER(ctypes.c_float),  # state_out
                ctypes.POINTER(ctypes.c_float),  # d_q
                ctypes.POINTER(ctypes.c_float),  # d_k
                ctypes.POINTER(ctypes.c_float),  # d_v
                ctypes.POINTER(ctypes.c_float),  # d_g
                ctypes.POINTER(ctypes.c_float),  # d_beta
                ctypes.POINTER(ctypes.c_float),  # d_state_in
                ctypes.c_int,                    # num_heads
                ctypes.c_int,                    # state_dim
                ctypes.c_float,                  # norm_eps
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


def torch_deltanet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_in: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_dim = q.shape[-1]
    q_scale = 1.0 / (state_dim ** 0.5)

    # q/k are already normalized by the explicit recurrent_qk_l2_norm graph op.
    q_hat = q * q_scale
    k_hat = k
    beta_s = torch.sigmoid(beta).unsqueeze(-1)
    gate = torch.exp(g).unsqueeze(-1).unsqueeze(-1)

    gated_state = state_in * gate
    kv_mem = torch.matmul(gated_state.transpose(-1, -2), k_hat.unsqueeze(-1)).squeeze(-1)
    delta = (v - kv_mem) * beta_s
    state_out = gated_state + torch.matmul(k_hat.unsqueeze(-1), delta.unsqueeze(-2))
    out = torch.matmul(state_out.transpose(-1, -2), q_hat.unsqueeze(-1)).squeeze(-1)
    return state_out, out


class TestDeltaNetParity(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.norm_eps = 1e-6
        self.atol_forward = 2e-5
        self.atol_backward = 5e-4

    def _run_case(self, num_heads: int, state_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        q = (0.25 * rng.standard_normal((num_heads, state_dim))).astype(np.float32)
        k = (0.25 * rng.standard_normal((num_heads, state_dim))).astype(np.float32)
        v = (0.25 * rng.standard_normal((num_heads, state_dim))).astype(np.float32)
        g = (0.10 * rng.standard_normal((num_heads,))).astype(np.float32)
        beta = (0.50 * rng.standard_normal((num_heads,))).astype(np.float32)
        state_in = (0.20 * rng.standard_normal((num_heads, state_dim, state_dim))).astype(np.float32)
        d_out = (0.25 * rng.standard_normal((num_heads, state_dim))).astype(np.float32)
        d_state_out = (0.15 * rng.standard_normal((num_heads, state_dim, state_dim))).astype(np.float32)

        q /= np.linalg.norm(q, axis=-1, keepdims=True) + self.norm_eps
        k /= np.linalg.norm(k, axis=-1, keepdims=True) + self.norm_eps

        ck_state_out = np.zeros_like(state_in)
        ck_out = np.zeros_like(q)
        LIB.gated_deltanet_autoregressive_forward(
            _as_ptr(q),
            _as_ptr(k),
            _as_ptr(v),
            _as_ptr(g),
            _as_ptr(beta),
            _as_ptr(state_in),
            _as_ptr(ck_state_out),
            _as_ptr(ck_out),
            num_heads,
            state_dim,
            ctypes.c_float(self.norm_eps),
        )

        ck_d_q = np.zeros_like(q)
        ck_d_k = np.zeros_like(k)
        ck_d_v = np.zeros_like(v)
        ck_d_g = np.zeros_like(g)
        ck_d_beta = np.zeros_like(beta)
        ck_d_state_in = np.zeros_like(state_in)
        LIB.gated_deltanet_autoregressive_backward(
            _as_ptr(d_out),
            _as_ptr(d_state_out),
            _as_ptr(q),
            _as_ptr(k),
            _as_ptr(v),
            _as_ptr(g),
            _as_ptr(beta),
            _as_ptr(state_in),
            _as_ptr(ck_state_out),
            _as_ptr(ck_d_q),
            _as_ptr(ck_d_k),
            _as_ptr(ck_d_v),
            _as_ptr(ck_d_g),
            _as_ptr(ck_d_beta),
            _as_ptr(ck_d_state_in),
            num_heads,
            state_dim,
            ctypes.c_float(self.norm_eps),
        )

        tq = torch.tensor(q, dtype=torch.float32, requires_grad=True)
        tk = torch.tensor(k, dtype=torch.float32, requires_grad=True)
        tv = torch.tensor(v, dtype=torch.float32, requires_grad=True)
        tg = torch.tensor(g, dtype=torch.float32, requires_grad=True)
        tb = torch.tensor(beta, dtype=torch.float32, requires_grad=True)
        ts = torch.tensor(state_in, dtype=torch.float32, requires_grad=True)
        td_out = torch.tensor(d_out, dtype=torch.float32)
        td_state_out = torch.tensor(d_state_out, dtype=torch.float32)

        t_state_out, t_out = torch_deltanet(tq, tk, tv, tg, tb, ts, self.norm_eps)
        torch.autograd.backward((t_out, t_state_out), (td_out, td_state_out))

        np.testing.assert_allclose(ck_state_out, t_state_out.detach().numpy(), atol=self.atol_forward, rtol=0.0)
        np.testing.assert_allclose(ck_out, t_out.detach().numpy(), atol=self.atol_forward, rtol=0.0)
        np.testing.assert_allclose(ck_d_q, tq.grad.detach().numpy(), atol=self.atol_backward, rtol=0.0)
        np.testing.assert_allclose(ck_d_k, tk.grad.detach().numpy(), atol=self.atol_backward, rtol=0.0)
        np.testing.assert_allclose(ck_d_v, tv.grad.detach().numpy(), atol=self.atol_backward, rtol=0.0)
        np.testing.assert_allclose(ck_d_g, tg.grad.detach().numpy(), atol=self.atol_backward, rtol=0.0)
        np.testing.assert_allclose(ck_d_beta, tb.grad.detach().numpy(), atol=self.atol_backward, rtol=0.0)
        np.testing.assert_allclose(ck_d_state_in, ts.grad.detach().numpy(), atol=self.atol_backward, rtol=0.0)

    def test_small_case(self) -> None:
        self._run_case(num_heads=2, state_dim=8, seed=7)

    def test_medium_case(self) -> None:
        self._run_case(num_heads=4, state_dim=16, seed=11)

    def test_wider_case(self) -> None:
        self._run_case(num_heads=3, state_dim=24, seed=19)


if __name__ == "__main__":
    unittest.main(verbosity=2)
