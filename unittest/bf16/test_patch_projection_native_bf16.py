#!/usr/bin/env python3
"""Numerical contract for the portable BF16 image patch projection."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(
    os.environ.get("CK_ENGINE_SO", str(ROOT / "build" / "libckernel_engine.so"))
)
KERNEL = LIB.patch_projection_image_bf16_native_storage
FLOAT_P = ctypes.POINTER(ctypes.c_float)
U16_P = ctypes.POINTER(ctypes.c_uint16)
KERNEL.argtypes = [
    FLOAT_P,
    U16_P,
    U16_P,
    FLOAT_P,
    FLOAT_P,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
KERNEL.restype = None


def bf16_bits(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((source >> 16) & np.uint32(1))
    return ((source + rounding) >> 16).astype(np.uint16)


def bf16_values(values: np.ndarray) -> np.ndarray:
    return (bf16_bits(values).astype(np.uint32) << 16).view(np.float32)


def run_scalar_case() -> None:
    channels, patch, merge, grid_h, grid_w, outputs = 2, 2, 2, 4, 6, 7
    image_h, image_w = grid_h * patch, grid_w * patch
    rng = np.random.default_rng(20260829)
    image = rng.standard_normal((channels, image_h, image_w), dtype=np.float32)
    weight0_f = rng.standard_normal(
        (outputs, channels, patch, patch), dtype=np.float32
    )
    weight1_f = rng.standard_normal(
        (outputs, channels, patch, patch), dtype=np.float32
    )
    weight0 = bf16_bits(weight0_f)
    weight1 = bf16_bits(weight1_f)
    bias = rng.standard_normal(outputs, dtype=np.float32)
    actual = np.empty((grid_h * grid_w, outputs), dtype=np.float32)

    KERNEL(
        image.ctypes.data_as(FLOAT_P),
        weight0.ctypes.data_as(U16_P),
        weight1.ctypes.data_as(U16_P),
        bias.ctypes.data_as(FLOAT_P),
        actual.ctypes.data_as(FLOAT_P),
        channels,
        image_h,
        image_w,
        patch,
        outputs,
        merge,
    )

    image_bf16 = bf16_values(image)
    weights = (bf16_values(weight0_f), bf16_values(weight1_f))
    expected = np.empty_like(actual)
    tile_area = merge * merge
    tiles_per_row = grid_w // merge
    for token in range(grid_h * grid_w):
        tile, within = divmod(token, tile_area)
        patch_y = (tile // tiles_per_row) * merge + within // merge
        patch_x = (tile % tiles_per_row) * merge + within % merge
        for output in range(outputs):
            total = np.float32(bf16_values(bias)[output])
            for channel in range(channels):
                for weight in weights:
                    for py in range(patch):
                        for px in range(patch):
                            value = image_bf16[
                                channel, patch_y * patch + py, patch_x * patch + px
                            ]
                            product = np.float32(
                                value * weight[output, channel, py, px]
                            )
                            total = np.float32(total + product)
            expected[token, output] = bf16_values(
                np.asarray([total], dtype=np.float32)
            )[0]

    if not np.array_equal(actual, expected):
        mismatch = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
        raise AssertionError(
            f"native patch projection differs at {mismatch.size}/{actual.size} values; "
            f"max_abs={float(np.max(np.abs(actual - expected)))}"
        )


def run_patch16_case() -> None:
    """Exercise the production patch shape, including AVX-512 BF16 when compiled."""
    channels, patch, merge, grid_h, grid_w, outputs = 1, 16, 2, 2, 2, 5
    image_h, image_w = grid_h * patch, grid_w * patch
    rng = np.random.default_rng(20260830)
    image = rng.standard_normal((channels, image_h, image_w), dtype=np.float32)
    weight0_f = rng.standard_normal(
        (outputs, channels, patch, patch), dtype=np.float32
    )
    weight1_f = rng.standard_normal(
        (outputs, channels, patch, patch), dtype=np.float32
    )
    weight0 = bf16_bits(weight0_f)
    weight1 = bf16_bits(weight1_f)
    bias = rng.standard_normal(outputs, dtype=np.float32)
    actual = np.empty((grid_h * grid_w, outputs), dtype=np.float32)

    KERNEL(
        image.ctypes.data_as(FLOAT_P),
        weight0.ctypes.data_as(U16_P),
        weight1.ctypes.data_as(U16_P),
        bias.ctypes.data_as(FLOAT_P),
        actual.ctypes.data_as(FLOAT_P),
        channels,
        image_h,
        image_w,
        patch,
        outputs,
        merge,
    )

    image_bf16 = bf16_values(image)
    weights = (bf16_values(weight0_f), bf16_values(weight1_f))
    expected = np.empty_like(actual)
    for token in range(grid_h * grid_w):
        patch_y, patch_x = divmod(token, grid_w)
        for output in range(outputs):
            total = np.float32(bf16_values(bias)[output])
            for weight in weights:
                for py in range(patch):
                    for px in range(patch):
                        value = image_bf16[
                            0, patch_y * patch + py, patch_x * patch + px
                        ]
                        total = np.float32(
                            total
                            + np.float32(value * weight[output, 0, py, px])
                        )
            expected[token, output] = bf16_values(
                np.asarray([total], dtype=np.float32)
            )[0]

    if not np.all(np.isfinite(actual)):
        raise AssertionError("native patch-16 projection produced non-finite output")
    max_abs = float(np.max(np.abs(actual - expected)))
    if max_abs > 0.5:
        raise AssertionError(
            "native patch-16 projection exceeds the BF16 reduction envelope; "
            f"max_abs={max_abs}"
        )


def main() -> int:
    run_scalar_case()
    run_patch16_case()
    print("native BF16 image patch projection: scalar exact; patch-16 within envelope")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
