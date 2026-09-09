#!/usr/bin/env python3
import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))

LIB.attn_gate_sigmoid_mul_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
LIB.attn_gate_sigmoid_mul_forward.restype = None

LIB.attn_gate_sigmoid_mul_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
LIB.attn_gate_sigmoid_mul_backward.restype = None
LIB.ck_set_num_threads.argtypes = [ctypes.c_int]
LIB.ck_set_num_threads.restype = None
LIB.ck_threadpool_global_destroy.argtypes = []
LIB.ck_threadpool_global_destroy.restype = None


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def main() -> None:
    torch.manual_seed(0)
    rows, num_heads, state_dim = 9, 4, 32
    dim = num_heads * state_dim
    x = torch.randn(rows, dim, dtype=torch.float32)
    gate = torch.randn(rows, dim, dtype=torch.float32)
    ref_out = x * torch.sigmoid(gate)

    x_np = x.numpy().copy()
    gate_np = gate.numpy().copy()
    out_np = np.zeros((rows, dim), dtype=np.float32)
    LIB.attn_gate_sigmoid_mul_forward(_ptr(x_np), _ptr(gate_np), _ptr(out_np), rows, num_heads, state_dim)
    np.testing.assert_allclose(out_np, ref_out.numpy(), rtol=1e-6, atol=1e-6)

    d_out = torch.randn(rows, dim, dtype=torch.float32)
    sig = torch.sigmoid(gate)
    ref_d_x = d_out * sig
    ref_d_gate = d_out * x * sig * (1.0 - sig)

    d_x_np = np.zeros((rows, dim), dtype=np.float32)
    d_gate_np = np.zeros((rows, dim), dtype=np.float32)
    LIB.attn_gate_sigmoid_mul_backward(
        _ptr(d_out.numpy().copy()),
        _ptr(x_np),
        _ptr(gate_np),
        _ptr(d_x_np),
        _ptr(d_gate_np),
        rows,
        num_heads,
        state_dim,
    )
    np.testing.assert_allclose(d_x_np, ref_d_x.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(d_gate_np, ref_d_gate.numpy(), rtol=1e-6, atol=1e-6)

    rows, num_heads, state_dim = 257, 16, 128
    dim = num_heads * state_dim
    rng = np.random.default_rng(37)
    large_x = rng.standard_normal((rows, dim)).astype(np.float32)
    large_gate = rng.standard_normal((rows, dim)).astype(np.float32)

    def run(threads: int, in_place: bool) -> np.ndarray:
        LIB.ck_threadpool_global_destroy()
        LIB.ck_set_num_threads(threads)
        x_arg = large_x.copy()
        out_arg = x_arg if in_place else np.empty_like(x_arg)
        LIB.attn_gate_sigmoid_mul_forward(
            _ptr(x_arg), _ptr(large_gate), _ptr(out_arg),
            rows, num_heads, state_dim,
        )
        return out_arg

    try:
        serial = run(1, False)
        np.testing.assert_array_equal(run(1, True), serial)
        for _ in range(8):
            np.testing.assert_array_equal(run(4, False), serial)
            np.testing.assert_array_equal(run(4, True), serial)
    finally:
        LIB.ck_threadpool_global_destroy()
        LIB.ck_set_num_threads(0)

    print("attn_gate_sigmoid_mul forward/backward parity: PASS")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
