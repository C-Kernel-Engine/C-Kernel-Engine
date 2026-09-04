#!/usr/bin/env python3
"""Exact Qwen4/PyTorch BF16 attention-gate contract test."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = Path(
    os.environ.get("CK_ENGINE_SO")
    or os.environ.get("CK_ENGINE_LIB")
    or ROOT / "build" / "libckernel_engine.so"
)
FLOAT_P = ctypes.POINTER(ctypes.c_float)


def main() -> int:
    if torch.backends.cpu.get_cpu_capability() != "AVX512":
        print("Qwen4 PyTorch BF16 attention gate [SKIP: AVX-512 unavailable]")
        return 0
    if not os.environ.get("CK_SLEEF_LIBRARY"):
        print("Qwen4 PyTorch BF16 attention gate [SKIP: CK_SLEEF_LIBRARY unset]")
        return 0

    lib = ctypes.CDLL(str(LIB_PATH))
    kernel = lib.attn_gate_sigmoid_mul_pytorch_bf16_storage
    kernel.argtypes = [
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    kernel.restype = None

    torch.manual_seed(73)
    rows, heads, head_dim = 3, 8, 256
    x = torch.randn(rows, heads * head_dim, dtype=torch.bfloat16)
    gate = torch.randn(rows, heads * head_dim, dtype=torch.bfloat16)
    expected = (x * torch.sigmoid(gate)).float().numpy()
    x_np = x.float().numpy()
    gate_np = gate.float().numpy()
    actual = np.empty_like(x_np)
    kernel(
        x_np.ctypes.data_as(FLOAT_P),
        gate_np.ctypes.data_as(FLOAT_P),
        actual.ctypes.data_as(FLOAT_P),
        rows,
        heads,
        head_dim,
    )
    np.testing.assert_array_equal(actual, expected)
    print(f"Qwen4 PyTorch BF16 attention gate exact={actual.size}/{actual.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
