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
]
LIB.attn_gate_sigmoid_mul_backward.restype = None


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def main() -> None:
    torch.manual_seed(0)
    rows, dim = 9, 128
    x = torch.randn(rows, dim, dtype=torch.float32)
    gate = torch.randn(rows, dim, dtype=torch.float32)
    ref_out = x * torch.sigmoid(gate)

    x_np = x.numpy().copy()
    gate_np = gate.numpy().copy()
    out_np = np.zeros((rows, dim), dtype=np.float32)
    LIB.attn_gate_sigmoid_mul_forward(_ptr(x_np), _ptr(gate_np), _ptr(out_np), rows, dim)
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
        dim,
    )
    np.testing.assert_allclose(d_x_np, ref_d_x.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(d_gate_np, ref_d_gate.numpy(), rtol=1e-6, atol=1e-6)

    print("attn_gate_sigmoid_mul forward/backward parity: PASS")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
