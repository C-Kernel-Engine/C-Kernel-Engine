"""Decode bounded publisher expert row tiles for the FP32 reference path."""
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


def decode_expert_rows(weights, scales, *, format_name, amax=None, denominator=None):
    """Expand one expert's consecutive rows using its actual serialized scale grid."""
    if (weights.ndim != 2 or not all(weights.shape) or weights.dtype != np.uint8
            or scales.dtype != np.uint8):
        raise ValueError("expert weights and scales must use nonempty byte rows")
    rows, stored_columns = weights.shape
    if format_name == "nvfp4":
        width, block = stored_columns * 2, 16
    elif format_name == "mxfp8":
        width, block = stored_columns, 32
    else:
        raise ValueError(f"unsupported expert format: {format_name}")
    if width % block or scales.shape != (rows, width // block):
        raise ValueError("expert scale grid disagrees with the weight rows")

    with np.errstate(over="ignore", invalid="ignore"):
        if format_name == "nvfp4":
            if (amax is None or denominator is None or not np.isfinite(amax)
                    or amax < 0 or not np.isfinite(denominator) or denominator <= 0):
                raise ValueError("invalid NVFP4 outer scale")
            magnitudes = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
            nibbles = np.empty((rows, width), dtype=np.uint8)
            nibbles[:, ::2] = weights & 15
            nibbles[:, 1::2] = weights >> 4
            values = magnitudes[nibbles & 7]
            values[nibbles >= 8] *= -1
            block_scales = decode_e4m3fn(scales)
            values.reshape(rows, -1, block)[:] *= block_scales[..., None]
            values *= np.float32(amax) / np.float32(denominator)
        else:
            if amax is not None or denominator is not None or np.any(scales == 255):
                raise ValueError("invalid MXFP8 scale")
            values = decode_e4m3fn(weights)
            block_scales = np.ldexp(np.ones(scales.shape, dtype=np.float32), scales.astype(np.int32) - 127)
            values.reshape(rows, -1, block)[:] *= block_scales[..., None]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite expanded expert weights")
    return values


def split_gate_up_rows(rows):
    """Split a tile starting at an even checkpoint row into gate and up rows."""
    if rows.ndim != 2 or not rows.shape[0] or rows.shape[0] % 2:
        raise ValueError("expected complete interleaved gate/up row pairs")
    return rows[::2].copy(), rows[1::2].copy()
