#!/usr/bin/env python3
"""Exact mixed-precision position interpolation parity with PyTorch."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(os.environ.get("CK_ENGINE_SO", str(ROOT / "build" / "libckernel_engine.so")))
FUNCTION = LIB.position_embeddings_add_tiled_2d_align_corners_fp32_interp_bf16
FUNCTION.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
FUNCTION.restype = None


def _tile_order_indices(grid_h: int, grid_w: int, merge_size: int) -> torch.Tensor:
    return torch.tensor(
        [
            (tile_y + dy) * grid_w + tile_x + dx
            for tile_y in range(0, grid_h, merge_size)
            for tile_x in range(0, grid_w, merge_size)
            for dy in range(merge_size)
            for dx in range(merge_size)
        ],
        dtype=torch.long,
    )


def _reference(
    x: torch.Tensor,
    table: torch.Tensor,
    grid_h: int,
    grid_w: int,
    merge_size: int,
) -> torch.Tensor:
    source = int(round(table.shape[0] ** 0.5))
    row_major = _tile_order_indices(grid_h, grid_w, merge_size)
    rows = (row_major // grid_w).to(torch.float32)
    cols = (row_major % grid_w).to(torch.float32)
    src_y = rows * (source - 1) / max(1, grid_h - 1)
    src_x = cols * (source - 1) / max(1, grid_w - 1)
    y0 = torch.floor(src_y).long()
    x0 = torch.floor(src_x).long()
    y1 = (y0 + 1).clamp(max=source - 1)
    x1 = (x0 + 1).clamp(max=source - 1)
    y_floor = torch.floor(src_y)
    x_floor = torch.floor(src_x)
    y_weights = torch.stack(
        [(1 - (src_y - y_floor).abs()).clamp(min=0),
         (1 - (src_y - y_floor - 1).abs()).clamp(min=0)],
        dim=1,
    )
    x_weights = torch.stack(
        [(1 - (src_x - x_floor).abs()).clamp(min=0),
         (1 - (src_x - x_floor - 1).abs()).clamp(min=0)],
        dim=1,
    )
    indices = torch.stack(
        [y0 * source + x0, y0 * source + x1, y1 * source + x0, y1 * source + x1],
        dim=1,
    )
    weights = (y_weights[:, :, None] * x_weights[:, None, :]).reshape(-1, 4)
    position = (table.to(torch.bfloat16)[indices] * weights[:, :, None]).sum(1)
    return (x.to(torch.bfloat16) + position.to(torch.bfloat16)).to(torch.float32)


def _run_case(grid_h: int, grid_w: int, source: int, dim: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(grid_h * grid_w, dim, generator=generator, dtype=torch.float32)
    table = torch.randn(source * source, dim, generator=generator, dtype=torch.float32).to(torch.bfloat16).float()
    expected = _reference(x, table, grid_h, grid_w, 2).numpy()
    actual = x.numpy().copy()
    table_np = table.numpy()
    FUNCTION(
        actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        table_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        grid_h, grid_w, dim, 2, source,
    )
    if not np.array_equal(actual, expected):
        diff = np.abs(actual - expected)
        index = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise AssertionError(
            f"mixed position mismatch shape={grid_h}x{grid_w} dim={dim} "
            f"max_abs={float(diff[index])} index={index} got={actual[index]} ref={expected[index]}"
        )


if __name__ == "__main__":
    _run_case(6, 10, 4, 16, 5301)
    _run_case(106, 138, 48, 32, 5302)
    print("Mixed BF16/FP32 tiled position parity: 2/2 exact")
