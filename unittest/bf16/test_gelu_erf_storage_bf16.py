#!/usr/bin/env python3
"""Numerical contract for portable exact-form GELU over BF16 storage."""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(
    os.environ.get("CK_ENGINE_SO", str(ROOT / "build" / "libckernel_engine.so"))
)
KERNEL = LIB.gelu_erf_bf16_storage
KERNEL.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]


def bf16_values(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((source >> 16) & np.uint32(1))
    bits = ((source + rounding) >> 16).astype(np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def main() -> int:
    rng = np.random.default_rng(20260829)
    source = bf16_values(
        np.concatenate(
            [
                np.linspace(-12.0, 12.0, 4097, dtype=np.float32),
                rng.standard_normal(3 * 4608, dtype=np.float32) * 3.0,
            ]
        )
    )
    actual = source.copy()
    KERNEL(actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), actual.size)
    expected_f32 = np.asarray(
        [
            0.5 * float(value) * (1.0 + math.erf(float(value) / math.sqrt(2.0)))
            for value in source
        ],
        dtype=np.float32,
    )
    expected = bf16_values(expected_f32)
    mismatch = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
    if mismatch.size:
        index = int(mismatch[0])
        raise AssertionError(
            f"portable BF16 erf GELU differs at {mismatch.size}/{actual.size}; "
            f"first={index} input={source[index]!r} actual={actual[index]!r} "
            f"expected={expected[index]!r}"
        )
    print(f"portable BF16 erf GELU: {actual.size}/{actual.size} exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
