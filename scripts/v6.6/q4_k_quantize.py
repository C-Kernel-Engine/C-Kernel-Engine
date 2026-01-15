"""
Q4_K (K-quants) encoder
======================

This module implements an *offline* float32 -> Q4_K packer that produces
`block_q4_K`-compatible bytes consumed by the engine's `dequant_q4_k_*` and
`gemm_nt_q4_k` kernels.

Notes / constraints
-------------------
  - Q4_K is a block format: 256 values per block, with 8 sub-blocks of 32.
  - The runtime dequantizes with:
        w = (d * sc[sub]) * q + (dmin * m[sub])
    where q is a signed 4-bit value in [-8, 7] (stored as unsigned nibble
    0..15 and shifted by -8 during dequant).
  - This encoder is *not* intended to be bit-identical to any external tool's
    quantizer. The goal is correctness and reasonable error for generating
    weights compatible with the engine kernels.
"""

from __future__ import annotations

import struct
from typing import Iterable, Iterator, Sequence

import numpy as np

QK_K = 256
SUB_BLOCK = 32
SUB_BLOCKS = 8
SCALES_BYTES = 12
BLOCK_BYTES = 144


def _fp16_round_to_f32(x: float) -> float:
    return float(np.float16(x).astype(np.float32))


def _fp16_bits(x: float) -> int:
    return int(np.float16(x).view(np.uint16).item())


def pack_q4_k_scales(sc: Sequence[int], m: Sequence[int]) -> bytes:
    if len(sc) != 8 or len(m) != 8:
        raise ValueError("Q4_K expects 8 scale and 8 min values")
    sc_u = [int(v) & 0x3F for v in sc]
    m_u = [int(v) & 0x3F for v in m]

    out = bytes(
        [
            (sc_u[0] & 0x3F) | ((sc_u[1] & 0x03) << 6),
            ((sc_u[1] >> 2) & 0x0F) | ((sc_u[2] & 0x0F) << 4),
            ((sc_u[2] >> 4) & 0x03) | ((sc_u[3] & 0x3F) << 2),
            (sc_u[4] & 0x3F) | ((sc_u[5] & 0x03) << 6),
            ((sc_u[5] >> 2) & 0x0F) | ((sc_u[6] & 0x0F) << 4),
            ((sc_u[6] >> 4) & 0x03) | ((sc_u[7] & 0x3F) << 2),
            (m_u[0] & 0x3F) | ((m_u[1] & 0x03) << 6),
            ((m_u[1] >> 2) & 0x0F) | ((m_u[2] & 0x0F) << 4),
            ((m_u[2] >> 4) & 0x03) | ((m_u[3] & 0x3F) << 2),
            (m_u[4] & 0x3F) | ((m_u[5] & 0x03) << 6),
            ((m_u[5] >> 2) & 0x0F) | ((m_u[6] & 0x0F) << 4),
            ((m_u[6] >> 4) & 0x03) | ((m_u[7] & 0x3F) << 2),
        ]
    )
    if len(out) != SCALES_BYTES:
        raise AssertionError("internal error: packed scales must be 12 bytes")
    return out


def quantize_q4_k_block(values_256: np.ndarray) -> bytes:
    """
    Quantize 256 float32 values into one `block_q4_K` (144 bytes).
    """
    v = np.asarray(values_256, dtype=np.float32).reshape(-1)
    if v.size != QK_K:
        raise ValueError("Q4_K block must have 256 values")

    sub_scale = np.empty(SUB_BLOCKS, dtype=np.float32)

    for sub in range(SUB_BLOCKS):
        seg = v[sub * SUB_BLOCK : (sub + 1) * SUB_BLOCK]
        vmin = float(seg.min())
        vmax = float(seg.max())
        # Symmetric quantization around 0. We deliberately keep the per-sub-block
        # offset term disabled (dmin=0, m=0) so offsets never have mixed signs.
        # This makes the encoder robust while still producing a valid Q4_K block
        # that the runtime can dequantize.
        if vmax == 0.0 and vmin == 0.0:
            sub_scale[sub] = 0.0
        else:
            sub_scale[sub] = max(0.0, vmax / 7.0, (-vmin) / 8.0)

    max_scale = float(sub_scale.max())
    d = 0.0 if max_scale <= 0.0 else (max_scale / 63.0)
    d_f = _fp16_round_to_f32(d)

    if d_f == 0.0:
        sc_u8 = np.zeros(SUB_BLOCKS, dtype=np.uint8)
    else:
        sc_u8 = np.rint(sub_scale / d_f).astype(np.int32)
        sc_u8 = np.clip(sc_u8, 0, 63)
        sc_u8 = np.where((sc_u8 == 0) & (sub_scale > 0), 1, sc_u8).astype(np.uint8)

    scale_q = d_f * sc_u8.astype(np.float32)

    # Offset disabled (symmetric): dmin=0, m[sub]=0 => min_q=0 for all subs.
    dmin_f = 0.0
    m_u8 = np.zeros(SUB_BLOCKS, dtype=np.uint8)
    min_q = np.zeros(SUB_BLOCKS, dtype=np.float32)

    qs = np.empty(128, dtype=np.uint8)
    for sub in range(SUB_BLOCKS):
        seg = v[sub * SUB_BLOCK : (sub + 1) * SUB_BLOCK]
        scale = float(scale_q[sub])
        off = 0.0
        if scale == 0.0:
            q = np.zeros(SUB_BLOCK, dtype=np.int32)
        else:
            q = np.rint((seg - off) / scale).astype(np.int32)
            q = np.clip(q, -8, 7)
        q_u = (q + 8).astype(np.uint8)
        base = sub * 16
        for i in range(16):
            qs[base + i] = (q_u[2 * i] & 0x0F) | ((q_u[2 * i + 1] & 0x0F) << 4)

    header = struct.pack("<H", _fp16_bits(d_f)) + struct.pack("<H", _fp16_bits(dmin_f))
    scales = pack_q4_k_scales(sc_u8.tolist(), m_u8.tolist())
    out = header + scales + qs.tobytes()
    if len(out) != BLOCK_BYTES:
        raise AssertionError("internal error: Q4_K block must be 144 bytes")
    return out


def iter_q4_k_row_bytes(values: np.ndarray) -> Iterator[bytes]:
    """
    Yield `block_q4_K` bytes for a row whose length is a multiple of 256.
    """
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    if v.size % QK_K != 0:
        raise ValueError("Q4_K rows must be a multiple of 256 elements")
    for b in range(v.size // QK_K):
        yield quantize_q4_k_block(v[b * QK_K : (b + 1) * QK_K])


def quantize_q4_k_row(values: np.ndarray) -> bytes:
    return b"".join(iter_q4_k_row_bytes(values))


def quantize_q4_k_rows(rows: Iterable[np.ndarray]) -> Iterator[bytes]:
    for row in rows:
        yield quantize_q4_k_row(row)
