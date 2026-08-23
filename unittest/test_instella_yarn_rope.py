#!/usr/bin/env python3
"""AMD Instella-MoE YaRN cache parity at production configuration boundaries."""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(os.environ.get("CK_ENGINE_SO", str(ROOT / "build/libckernel_engine.so")))
FPTR = ctypes.POINTER(ctypes.c_float)
I32PTR = ctypes.POINTER(ctypes.c_int32)
U16PTR = ctypes.POINTER(ctypes.c_uint16)

COMMON = [I32PTR, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
          ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
LIB.yarn_rope_cache_explicit_positions_f32.argtypes = [FPTR, FPTR] + COMMON
LIB.yarn_rope_cache_explicit_positions_bf16.argtypes = [U16PTR, U16PTR] + COMMON
LIB.yarn_rope_cache_contiguous_positions_f32.argtypes = [
    FPTR, FPTR, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
]


def reference(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dim, base, factor, original = 32, 8_000_000.0, 40.0, 4096
    beta_fast, beta_slow = 32.0, 1.0

    def correction(rotations: float) -> float:
        return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction(beta_fast)), 0)
    high = min(math.ceil(correction(beta_slow)), dim - 1)
    ramp = torch.clamp(
        (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1
    )
    pos_freq = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv = (1.0 / (factor * pos_freq)) * ramp + (1.0 / pos_freq) * (1.0 - ramp)
    angles = positions.float()[:, None] * inv[None, :]
    # Instella has mscale == mscale_all_dim == 1, hence attention scaling is 1.
    return angles.cos(), angles.sin()


def test_instella_yarn_cache_fp32_and_bf16() -> None:
    positions = np.ascontiguousarray(
        np.array([0, 1, 31, 4095, 4096, 32767], dtype=np.int32)
    )
    shape = (len(positions), 16)
    cos = np.empty(shape, dtype=np.float32)
    sin = np.empty(shape, dtype=np.float32)
    cos_b = np.empty(shape, dtype=np.uint16)
    sin_b = np.empty(shape, dtype=np.uint16)
    args = (
        positions.ctypes.data_as(I32PTR), len(positions), 32,
        8_000_000.0, 40.0, 4096, 32.0, 1.0, 1.0, 1.0,
    )
    LIB.yarn_rope_cache_explicit_positions_f32(
        cos.ctypes.data_as(FPTR), sin.ctypes.data_as(FPTR), *args
    )
    LIB.yarn_rope_cache_explicit_positions_bf16(
        cos_b.ctypes.data_as(U16PTR), sin_b.ctypes.data_as(U16PTR), *args
    )

    expected_cos, expected_sin = reference(torch.from_numpy(positions))
    np.testing.assert_allclose(cos, expected_cos.numpy(), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(sin, expected_sin.numpy(), rtol=2e-6, atol=2e-6)
    expected_cos_b = expected_cos.to(torch.bfloat16).view(torch.uint16).numpy()
    expected_sin_b = expected_sin.to(torch.bfloat16).view(torch.uint16).numpy()
    np.testing.assert_array_equal(cos_b, expected_cos_b)
    np.testing.assert_array_equal(sin_b, expected_sin_b)


def test_contiguous_cache_matches_explicit_positions_bit_exactly() -> None:
    positions = np.arange(257, dtype=np.int32)
    shape = (len(positions), 16)
    explicit_cos = np.empty(shape, dtype=np.float32)
    explicit_sin = np.empty(shape, dtype=np.float32)
    contiguous_cos = np.empty(shape, dtype=np.float32)
    contiguous_sin = np.empty(shape, dtype=np.float32)
    params = (len(positions), 32, 8_000_000.0, 40.0, 4096, 32.0, 1.0, 1.0, 1.0)

    LIB.yarn_rope_cache_explicit_positions_f32(
        explicit_cos.ctypes.data_as(FPTR),
        explicit_sin.ctypes.data_as(FPTR),
        positions.ctypes.data_as(I32PTR),
        *params,
    )
    LIB.yarn_rope_cache_contiguous_positions_f32(
        contiguous_cos.ctypes.data_as(FPTR),
        contiguous_sin.ctypes.data_as(FPTR),
        *params,
    )

    np.testing.assert_array_equal(contiguous_cos, explicit_cos)
    np.testing.assert_array_equal(contiguous_sin, explicit_sin)


if __name__ == "__main__":
    test_instella_yarn_cache_fp32_and_bf16()
    test_contiguous_cache_matches_explicit_positions_bit_exactly()
    print("Instella YaRN cache: explicit and contiguous position parity passed")
