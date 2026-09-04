#!/usr/bin/env python3
"""PyTorch oracle for decode GEMV with a BF16 output-storage boundary."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
F32P = ctypes.POINTER(ctypes.c_float)
KERNEL = LIB.gemv_bf16_bf16_storage_parallel_dispatch
KERNEL.argtypes = [F32P, ctypes.c_void_p, F32P, ctypes.c_int, ctypes.c_int]
KERNEL.restype = None


def bf16_bits(values: torch.Tensor) -> np.ndarray:
    return values.to(torch.bfloat16).view(torch.uint16).cpu().numpy().copy()


def main() -> int:
    torch.set_num_threads(1)
    torch.manual_seed(73)
    for rows, cols in ((8, 16), (80, 128), (384, 128)):
        x = torch.randn(cols, dtype=torch.bfloat16)
        weight = torch.randn(rows, cols, dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(x, weight).float().numpy()
        actual = np.empty(rows, dtype=np.float32)
        weight_bits = bf16_bits(weight)
        x_values = x.float().numpy().copy()
        KERNEL(
            actual.ctypes.data_as(F32P),
            ctypes.c_void_p(weight_bits.ctypes.data),
            x_values.ctypes.data_as(F32P),
            rows,
            cols,
        )
        np.testing.assert_array_equal(actual, expected)
        if not np.isfinite(actual).all():
            raise AssertionError("decode GEMV returned non-finite values")
    print("BF16 GEMV output storage parity: 3/3 exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
