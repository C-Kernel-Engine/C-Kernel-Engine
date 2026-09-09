#!/usr/bin/env python3
"""
PyTorch parity test for recurrent_conv_state_update.
"""

from __future__ import annotations

import ctypes
import subprocess
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
            fwd = lib.recurrent_conv_state_update_forward
            fwd.argtypes = [
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
                ctypes.c_int,
                ctypes.c_int,
            ]
            fwd.restype = None

            bwd = lib.recurrent_conv_state_update_backward
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
                ctypes.c_int,
                ctypes.c_int,
            ]
            bwd.restype = None
            bwd_workspace = lib.recurrent_conv_state_update_backward_workspace
            bwd_workspace.argtypes = bwd.argtypes[:6] + [
                ctypes.POINTER(ctypes.c_float)
            ] + bwd.argtypes[6:]
            bwd_workspace.restype = None
            lib.ck_set_num_threads.argtypes = [ctypes.c_int]
            lib.ck_set_num_threads.restype = None
            lib.ck_threadpool_global_destroy.argtypes = []
            lib.ck_threadpool_global_destroy.restype = None
            return lib
    return None


LIB = _load_lib()
if LIB is None:  # pragma: no cover
    print("[SKIP] libckernel_engine.so not found")
    sys.exit(0)


def _as_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _torch_ref(state_in, q, k, v, history_len, num_seqs, num_tokens, q_dim, k_dim, v_dim):
    channels = q_dim + k_dim + v_dim
    qkv = torch.cat([q, k, v], dim=1).view(num_seqs, num_tokens, channels).transpose(1, 2)
    conv_x = torch.cat([state_in, qkv], dim=2)
    state_out = conv_x[:, :, num_tokens:]
    return conv_x, state_out


class TestRecurrentConvStateUpdate(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.atol = 5e-5

    def _run_case(self, history_len: int, num_seqs: int, num_tokens: int, q_dim: int, k_dim: int, v_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        channels = q_dim + k_dim + v_dim
        state_in = (0.20 * rng.standard_normal((num_seqs, channels, history_len))).astype(np.float32)
        q = (0.25 * rng.standard_normal((num_seqs * num_tokens, q_dim))).astype(np.float32)
        k = (0.25 * rng.standard_normal((num_seqs * num_tokens, k_dim))).astype(np.float32)
        v = (0.25 * rng.standard_normal((num_seqs * num_tokens, v_dim))).astype(np.float32)

        ck_conv_x = np.zeros((num_seqs, channels, history_len + num_tokens), dtype=np.float32)
        ck_state_out = np.zeros((num_seqs, channels, history_len), dtype=np.float32)
        LIB.recurrent_conv_state_update_forward(
            _as_ptr(state_in), _as_ptr(q), _as_ptr(k), _as_ptr(v),
            _as_ptr(ck_conv_x), _as_ptr(ck_state_out),
            history_len, num_seqs, num_tokens, q_dim, k_dim, v_dim,
        )

        d_conv_x = (0.30 * rng.standard_normal(ck_conv_x.shape)).astype(np.float32)
        d_state_out = (0.20 * rng.standard_normal(ck_state_out.shape)).astype(np.float32)
        ck_d_state_in = np.zeros_like(state_in)
        ck_d_q = np.zeros_like(q)
        ck_d_k = np.zeros_like(k)
        ck_d_v = np.zeros_like(v)
        workspace = np.empty_like(ck_conv_x)
        LIB.recurrent_conv_state_update_backward_workspace(
            _as_ptr(d_conv_x), _as_ptr(d_state_out),
            _as_ptr(ck_d_state_in), _as_ptr(ck_d_q), _as_ptr(ck_d_k), _as_ptr(ck_d_v),
            _as_ptr(workspace),
            history_len, num_seqs, num_tokens, q_dim, k_dim, v_dim,
        )

        t_state_in = torch.tensor(state_in, dtype=torch.float32, requires_grad=True)
        t_q = torch.tensor(q, dtype=torch.float32, requires_grad=True)
        t_k = torch.tensor(k, dtype=torch.float32, requires_grad=True)
        t_v = torch.tensor(v, dtype=torch.float32, requires_grad=True)
        t_conv_x, t_state_out = _torch_ref(t_state_in, t_q, t_k, t_v, history_len, num_seqs, num_tokens, q_dim, k_dim, v_dim)
        t_conv_x.backward(torch.tensor(d_conv_x), retain_graph=True)
        grads_from_conv = (t_state_in.grad.detach().clone(), t_q.grad.detach().clone(), t_k.grad.detach().clone(), t_v.grad.detach().clone())
        t_state_in.grad.zero_()
        t_q.grad.zero_()
        t_k.grad.zero_()
        t_v.grad.zero_()
        t_state_out.backward(torch.tensor(d_state_out))
        torch_d_state_in = (grads_from_conv[0] + t_state_in.grad).numpy()
        torch_d_q = (grads_from_conv[1] + t_q.grad).numpy()
        torch_d_k = (grads_from_conv[2] + t_k.grad).numpy()
        torch_d_v = (grads_from_conv[3] + t_v.grad).numpy()

        self.assertTrue(np.array_equal(ck_conv_x, t_conv_x.detach().numpy()))
        self.assertTrue(np.array_equal(ck_state_out, t_state_out.detach().numpy()))
        np.testing.assert_allclose(ck_d_state_in, torch_d_state_in, atol=self.atol, rtol=0.0)
        np.testing.assert_allclose(ck_d_q, torch_d_q, atol=self.atol, rtol=0.0)
        np.testing.assert_allclose(ck_d_k, torch_d_k, atol=self.atol, rtol=0.0)
        np.testing.assert_allclose(ck_d_v, torch_d_v, atol=self.atol, rtol=0.0)

    def test_small_case(self) -> None:
        self._run_case(3, 2, 4, 16, 16, 24, 7)

    def test_qwen35_like_case(self) -> None:
        self._run_case(3, 1, 5, 128 * 16, 128 * 16, 2048, 13)

    def test_multi_token_forward_matches_repeated_single_token_updates(self) -> None:
        history_len, num_seqs, num_tokens = 3, 2, 5
        q_dim, k_dim, v_dim = 7, 5, 6
        channels = q_dim + k_dim + v_dim
        rng = np.random.default_rng(31)
        state_in = (0.20 * rng.standard_normal((num_seqs, channels, history_len))).astype(np.float32)
        q = (0.25 * rng.standard_normal((num_seqs * num_tokens, q_dim))).astype(np.float32)
        k = (0.25 * rng.standard_normal((num_seqs * num_tokens, k_dim))).astype(np.float32)
        v = (0.25 * rng.standard_normal((num_seqs * num_tokens, v_dim))).astype(np.float32)

        batch_conv = np.zeros((num_seqs, channels, history_len + num_tokens), dtype=np.float32)
        batch_state = np.zeros((num_seqs, channels, history_len), dtype=np.float32)
        LIB.recurrent_conv_state_update_forward(
            _as_ptr(state_in), _as_ptr(q), _as_ptr(k), _as_ptr(v),
            _as_ptr(batch_conv), _as_ptr(batch_state),
            history_len, num_seqs, num_tokens, q_dim, k_dim, v_dim,
        )

        step_state = state_in.copy()
        step_rows = []
        for tok in range(num_tokens):
            q_step = np.ascontiguousarray(q[tok::num_tokens])
            k_step = np.ascontiguousarray(k[tok::num_tokens])
            v_step = np.ascontiguousarray(v[tok::num_tokens])
            step_conv = np.zeros((num_seqs, channels, history_len + 1), dtype=np.float32)
            next_state = np.zeros_like(step_state)
            LIB.recurrent_conv_state_update_forward(
                _as_ptr(step_state), _as_ptr(q_step), _as_ptr(k_step), _as_ptr(v_step),
                _as_ptr(step_conv), _as_ptr(next_state),
                history_len, num_seqs, 1, q_dim, k_dim, v_dim,
            )
            step_rows.append(step_conv[:, :, -1:])
            step_state = next_state

        step_tail = np.concatenate(step_rows, axis=2)
        self.assertTrue(np.array_equal(step_tail, batch_conv[:, :, history_len:]))
        self.assertTrue(np.array_equal(step_state, batch_state))

    def test_parallel_forward_is_bit_exact_with_in_place_state(self) -> None:
        history_len, num_seqs, num_tokens = 3, 2, 257
        q_dim, k_dim, v_dim = 97, 83, 122
        channels = q_dim + k_dim + v_dim
        rng = np.random.default_rng(47)
        state = rng.standard_normal((num_seqs, channels, history_len)).astype(np.float32)
        q = rng.standard_normal((num_seqs * num_tokens, q_dim)).astype(np.float32)
        k = rng.standard_normal((num_seqs * num_tokens, k_dim)).astype(np.float32)
        v = rng.standard_normal((num_seqs * num_tokens, v_dim)).astype(np.float32)

        def run(threads: int) -> tuple[np.ndarray, np.ndarray]:
            LIB.ck_threadpool_global_destroy()
            LIB.ck_set_num_threads(threads)
            state_io = state.copy()
            conv_x = np.empty(
                (num_seqs, channels, history_len + num_tokens), dtype=np.float32
            )
            LIB.recurrent_conv_state_update_forward(
                _as_ptr(state_io), _as_ptr(q), _as_ptr(k), _as_ptr(v),
                _as_ptr(conv_x), _as_ptr(state_io),
                history_len, num_seqs, num_tokens, q_dim, k_dim, v_dim,
            )
            return conv_x, state_io

        try:
            serial_conv, serial_state = run(1)
            for _ in range(8):
                parallel_conv, parallel_state = run(4)
                self.assertTrue(np.array_equal(serial_conv, parallel_conv))
                self.assertTrue(np.array_equal(serial_state, parallel_state))
        finally:
            LIB.ck_threadpool_global_destroy()
            LIB.ck_set_num_threads(0)

    def test_rejects_sequence_token_product_overflow(self) -> None:
        code = f"""
import ctypes
lib = ctypes.CDLL({LIB._name!r})
fn = lib.recurrent_conv_state_update_forward
ptr = ctypes.POINTER(ctypes.c_float)
fn.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr] + [ctypes.c_int] * 6
value = (ctypes.c_float * 1)(123.0)
fn(value, value, value, value, value, value,
   1, 1073741824, 2, 1, 0, 0)
assert value[0] == 123.0
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
