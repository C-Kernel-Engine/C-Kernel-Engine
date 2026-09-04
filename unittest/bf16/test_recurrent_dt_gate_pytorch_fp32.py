#!/usr/bin/env python3
"""Bit-exact PyTorch AVX-512 oracle for the recurrent decay gate."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
FLOAT_P = ctypes.POINTER(ctypes.c_float)


def main() -> int:
    if torch.backends.cpu.get_cpu_capability() != "AVX512":
        print("PyTorch recurrent dt gate [SKIP: AVX-512 unavailable]")
        return 0
    if not os.environ.get("CK_SLEEF_LIBRARY"):
        candidate = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cpu.so"
        if not candidate.is_file():
            raise RuntimeError("CK_SLEEF_LIBRARY must export PyTorch SLEEF exp/log1p")
        os.environ["CK_SLEEF_LIBRARY"] = str(candidate)

    library = Path(os.environ.get("CK_ENGINE_SO", ROOT / "build" / "libckernel_engine.so"))
    lib = ctypes.CDLL(str(library))
    kernel = lib.recurrent_dt_gate_forward_pytorch_fp32
    kernel.argtypes = [FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    kernel.restype = None

    torch.set_num_threads(1)
    torch.manual_seed(193)
    rows, heads = 7, 48
    alpha = torch.randn(rows, heads, dtype=torch.float32) * 0.7
    dt_bias = torch.randn(heads, dtype=torch.float32) * 0.4
    a_log = torch.randn(heads, dtype=torch.float32) * 0.3
    a = -torch.exp(a_log)
    expected = (F.softplus(alpha + dt_bias) * a).numpy()

    alpha_np = alpha.numpy()
    dt_bias_np = dt_bias.numpy()
    a_np = a.numpy()
    actual = np.empty_like(alpha_np)
    kernel(
        alpha_np.ctypes.data_as(FLOAT_P),
        dt_bias_np.ctypes.data_as(FLOAT_P),
        a_np.ctypes.data_as(FLOAT_P),
        actual.ctypes.data_as(FLOAT_P),
        rows,
        heads,
        1,
    )
    np.testing.assert_array_equal(actual, expected)
    print(f"PyTorch recurrent dt gate exact={actual.size}/{actual.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
