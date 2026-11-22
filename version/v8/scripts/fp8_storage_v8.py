"""Expand checked E4M3FN checkpoint storage for reference execution."""

import numpy as np


def decode_e4m3fn(codes):
    """Decode the Safetensors F8_E4M3 byte representation."""
    if codes.dtype != np.uint8:
        raise ValueError("E4M3FN requires byte codes")
    magnitude = codes & 127
    if np.any(magnitude == 127):
        raise ValueError("invalid E4M3FN NaN code")
    exponent = magnitude >> 3
    fraction = (magnitude & 7).astype(np.float32) * np.float32(0.125)
    normal = np.ldexp(1 + fraction, exponent.astype(np.int32) - 7)
    value = np.where(exponent == 0, np.ldexp(fraction, -6), normal)
    return np.where(codes & 128, -value, value)


def expand_block_scaled_e4m3fn(weights, scales, *, row_block, column_block):
    """Expand a two-axis FP32 multiplier grid over packed E4M3FN weights."""
    if (weights.ndim != 2 or not weights.size or weights.dtype != np.uint8 or
            scales.ndim != 2 or scales.dtype != np.float32):
        raise ValueError("block FP8 requires a 2D byte matrix and 2D FP32 multipliers")
    if type(row_block) is not int or type(column_block) is not int or min(row_block, column_block) < 1:
        raise ValueError("invalid block FP8 geometry")
    rows, columns = weights.shape
    expected = ((rows + row_block - 1) // row_block,
                (columns + column_block - 1) // column_block)
    if scales.shape != expected:
        raise ValueError("block FP8 multiplier grid disagrees with the weight matrix")
    if not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("block FP8 multipliers must be finite and nonnegative")
    values = decode_e4m3fn(weights)
    expanded = scales.repeat(row_block, axis=0).repeat(column_block, axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        values *= expanded[:rows, :columns]
    if not np.isfinite(values).all():
        raise ValueError("expanded block FP8 weights are nonfinite")
    return values
