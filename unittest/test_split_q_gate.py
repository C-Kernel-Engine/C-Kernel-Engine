#!/usr/bin/env python3
import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))

LIB.split_q_gate_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
LIB.split_q_gate_forward.restype = None

LIB.split_q_gate_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
LIB.split_q_gate_backward.restype = None


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _make_interleaved(q: torch.Tensor, gate: torch.Tensor, group_dim: int) -> torch.Tensor:
    rows, q_dim = q.shape
    gate_dim = gate.shape[1]
    num_groups = q_dim // group_dim
    gate_group_dim = gate_dim // max(1, num_groups)
    parts = []
    for group in range(num_groups):
        q0 = group * group_dim
        g0 = group * gate_group_dim
        parts.append(q[:, q0:q0 + group_dim])
        parts.append(gate[:, g0:g0 + gate_group_dim])
    return torch.cat(parts, dim=1)


def _run_case(rows: int, q_dim: int, gate_dim: int, group_dim: int) -> None:
    torch.manual_seed(rows + q_dim + gate_dim + group_dim)
    q = torch.randn(rows, q_dim, dtype=torch.float32)
    gate = torch.randn(rows, gate_dim, dtype=torch.float32)
    packed = _make_interleaved(q, gate, group_dim)

    packed_np = packed.numpy().copy()
    q_np = np.zeros((rows, q_dim), dtype=np.float32)
    gate_np = np.zeros((rows, gate_dim), dtype=np.float32)

    LIB.split_q_gate_forward(_ptr(packed_np), _ptr(q_np), _ptr(gate_np), rows, q_dim, gate_dim, group_dim)

    np.testing.assert_allclose(q_np, q.numpy(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(gate_np, gate.numpy(), rtol=0.0, atol=0.0)

    d_q = torch.randn(rows, q_dim, dtype=torch.float32)
    d_gate = torch.randn(rows, gate_dim, dtype=torch.float32)
    ref_d_packed = _make_interleaved(d_q, d_gate, group_dim)

    d_packed_np = np.zeros((rows, q_dim + gate_dim), dtype=np.float32)
    LIB.split_q_gate_backward(
        _ptr(d_q.numpy().copy()),
        _ptr(d_gate.numpy().copy()),
        _ptr(d_packed_np),
        rows,
        q_dim,
        gate_dim,
        group_dim,
    )
    np.testing.assert_allclose(d_packed_np, ref_d_packed.numpy(), rtol=0.0, atol=0.0)


def main() -> None:
    _run_case(rows=7, q_dim=64, gate_dim=64, group_dim=64)
    _run_case(rows=3, q_dim=2048, gate_dim=2048, group_dim=256)
    print("split_q_gate forward/backward parity: PASS")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
