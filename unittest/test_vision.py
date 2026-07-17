from __future__ import annotations
import ctypes
import os
import argparse
import time
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib


lib = load_lib("libckernel_vision.so", "libckernel_engine.so")

# void im2patch(const float *image, float *patches, int C, int H, int W, int P)
lib.im2patch.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, # C
    ctypes.c_int, # H
    ctypes.c_int, # W
    ctypes.c_int  # P
]
lib.im2patch.restype = None

# void patch2im(const float *d_patches, float *d_image, int C, int H, int W, int P)
lib.patch2im.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, # C
    ctypes.c_int, # H
    ctypes.c_int, # W
    ctypes.c_int  # P
]
lib.patch2im.restype = None

lib.position_embeddings_add.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.position_embeddings_add.restype = None

lib.position_embeddings_add_tiled_2d.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.position_embeddings_add_tiled_2d.restype = None

lib.position_embeddings_add_tiled_2d_align_corners.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.position_embeddings_add_tiled_2d_align_corners.restype = None

lib.vision_position_ids_2d_merge.argtypes = [
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.vision_position_ids_2d_merge.restype = None

lib.rowwise_bias_add.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
]
lib.rowwise_bias_add.restype = None

lib.spatial_merge_2x2.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.spatial_merge_2x2.restype = None

lib.spatial_merge_contiguous_tiled.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.spatial_merge_contiguous_tiled.restype = None

lib.feature_concat_2way.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.feature_concat_2way.restype = None

lib.feature_slice_copy.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.feature_slice_copy.restype = None

lib.feature_concat.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.feature_concat.restype = None

lib.add_stream_reorder_2d.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.add_stream_reorder_2d.restype = None

lib.mrope_qk_vision.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
]
lib.mrope_qk_vision.restype = None

for _mrope_storage_fn in ("mrope_qk_vision_bf16_storage", "mrope_qk_vision_fp16_storage"):
    _fn = getattr(lib, _mrope_storage_fn)
    _fn.argtypes = lib.mrope_qk_vision.argtypes
    _fn.restype = None


def tensor_to_ptr(t: torch.Tensor):
    if not t.is_contiguous():
        raise ValueError("ctypes kernel test inputs must be contiguous")
    return t.view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def tensor_to_ptr_i32(t: torch.Tensor):
    if not t.is_contiguous():
        raise ValueError("ctypes kernel test inputs must be contiguous")
    return t.view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def run_c_im2patch(image: torch.Tensor, P: int) -> torch.Tensor:
    C, H, W = image.shape
    num_patches = (H // P) * (W // P)
    patch_dim = C * P * P
    
    patches_out = torch.empty((num_patches, patch_dim), dtype=torch.float32)
    
    lib.im2patch(
        tensor_to_ptr(image),
        tensor_to_ptr(patches_out),
        C, H, W, P
    )
    return patches_out


def run_c_patch2im(d_patches: torch.Tensor, C, H, W, P) -> torch.Tensor:
    d_image_out = torch.empty((C, H, W), dtype=torch.float32)
    
    lib.patch2im(
        tensor_to_ptr(d_patches),
        tensor_to_ptr(d_image_out),
        C, H, W, P
    )
    return d_image_out


def run_c_position_embeddings_add(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    num_tokens, embed_dim = out.shape
    num_positions = pos.shape[0]
    lib.position_embeddings_add(
        tensor_to_ptr(out),
        tensor_to_ptr(pos),
        num_tokens,
        embed_dim,
        num_positions,
    )
    return out


def run_c_position_embeddings_add_tiled_2d(
    x: torch.Tensor,
    pos: torch.Tensor,
    grid_h: int,
    grid_w: int,
    merge_size: int,
    source_grid_size: int | None = None,
) -> torch.Tensor:
    out = x.clone()
    _, embed_dim = out.shape
    if source_grid_size is None:
        source_grid_size = grid_h
    lib.position_embeddings_add_tiled_2d(
        tensor_to_ptr(out),
        tensor_to_ptr(pos),
        grid_h,
        grid_w,
        embed_dim,
        merge_size,
        source_grid_size,
    )
    return out


def run_c_position_embeddings_add_tiled_2d_align_corners(
    x: torch.Tensor,
    pos: torch.Tensor,
    grid_h: int,
    grid_w: int,
    merge_size: int,
    source_grid_size: int,
) -> torch.Tensor:
    out = x.clone()
    _, embed_dim = out.shape
    lib.position_embeddings_add_tiled_2d_align_corners(
        tensor_to_ptr(out), tensor_to_ptr(pos), grid_h, grid_w, embed_dim, merge_size, source_grid_size
    )
    return out


def run_c_vision_position_ids(grid_h: int, grid_w: int, merge_size: int) -> torch.Tensor:
    positions = torch.empty(4 * grid_h * grid_w, dtype=torch.int32)
    lib.vision_position_ids_2d_merge(
        tensor_to_ptr_i32(positions),
        grid_h,
        grid_w,
        merge_size,
    )
    return positions.view(4, grid_h * grid_w)


def run_c_rowwise_bias_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    rows, dim = out.shape
    lib.rowwise_bias_add(
        tensor_to_ptr(out),
        tensor_to_ptr(bias),
        rows,
        dim,
    )
    return out


def run_c_spatial_merge_2x2(x: torch.Tensor, grid_h: int, grid_w: int, embed_dim: int) -> torch.Tensor:
    merged = torch.empty(((grid_h // 2) * (grid_w // 2), embed_dim * 4), dtype=torch.float32)
    lib.spatial_merge_2x2(
        tensor_to_ptr(x),
        tensor_to_ptr(merged),
        grid_h,
        grid_w,
        embed_dim,
    )
    return merged


def run_c_feature_concat_2way(main_input: torch.Tensor, branch_input: torch.Tensor) -> torch.Tensor:
    rows, main_dim = main_input.shape
    branch_dim = branch_input.shape[1]
    out = torch.empty((rows, main_dim + branch_dim), dtype=torch.float32)
    lib.feature_concat_2way(
        tensor_to_ptr(main_input),
        tensor_to_ptr(branch_input),
        tensor_to_ptr(out),
        rows,
        main_dim,
        branch_dim,
        1,
    )
    return out


def run_c_add_stream_reorder_2d(
    main_input: torch.Tensor,
    aux_input: torch.Tensor,
    grid_h: int,
    grid_w: int,
    merge_size: int,
) -> torch.Tensor:
    main_out = main_input.clone()
    aux_scratch = aux_input.clone()
    _, embed_dim = main_out.shape
    lib.add_stream_reorder_2d(
        tensor_to_ptr(main_out),
        tensor_to_ptr(aux_scratch),
        grid_h,
        grid_w,
        embed_dim,
        merge_size,
    )
    return main_out


def run_c_feature_slice_copy(src: torch.Tensor, dst_dim: int, dst_feature_offset: int) -> torch.Tensor:
    rows, src_dim = src.shape
    dst = torch.full((rows, dst_dim), -123.0, dtype=torch.float32)
    lib.feature_slice_copy(
        tensor_to_ptr(src),
        tensor_to_ptr(dst),
        rows,
        src_dim,
        dst_dim,
        dst_feature_offset,
    )
    return dst


def run_c_feature_concat(main_input: torch.Tensor, branch_slices: list[torch.Tensor]) -> torch.Tensor:
    rows, main_dim = main_input.shape
    branch_slice_dim = branch_slices[0].shape[1] if branch_slices else 0
    num_branch_slices = len(branch_slices)
    branch_input = (
        torch.stack(branch_slices, dim=0).contiguous()
        if branch_slices
        else torch.empty((0, rows, 0), dtype=torch.float32)
    )
    output = torch.empty((rows, main_dim + branch_slice_dim * num_branch_slices), dtype=torch.float32)
    lib.feature_concat(
        tensor_to_ptr(main_input),
        tensor_to_ptr(branch_input) if branch_slices else ctypes.POINTER(ctypes.c_float)(),
        tensor_to_ptr(output),
        rows,
        main_dim,
        branch_slice_dim,
        num_branch_slices,
    )
    return output


def run_c_spatial_merge_contiguous_tiled(
    x: torch.Tensor,
    grid_h: int,
    grid_w: int,
    merge_size: int,
) -> torch.Tensor:
    tokens, embed_dim = x.shape
    merge_factor = merge_size * merge_size
    merged = torch.empty((tokens // merge_factor, embed_dim * merge_factor), dtype=torch.float32)
    lib.spatial_merge_contiguous_tiled(
        tensor_to_ptr(x),
        tensor_to_ptr(merged),
        grid_h,
        grid_w,
        embed_dim,
        merge_size,
    )
    return merged


def run_c_mrope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    n_dims: int,
    sections: list[int],
    n_ctx_orig: int = 32768,
    freq_base: float = 10000.0,
    freq_scale: float = 1.0,
    ext_factor: float = 0.0,
    attn_factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    function: str = "mrope_qk_vision",
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = q.clone()
    k_out = k.clone()
    num_heads, num_tokens, head_dim = q_out.shape
    num_kv_heads = k_out.shape[0]
    q_np = q_out.view(-1).numpy()
    k_np = k_out.view(-1).numpy()
    pos_np = positions.contiguous().view(-1).numpy()
    getattr(lib, function)(
        q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pos_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        num_heads,
        num_kv_heads,
        num_tokens,
        head_dim,
        head_dim,
        n_dims,
        int(sections[0]),
        int(sections[1]),
        int(sections[2]),
        int(sections[3]),
        n_ctx_orig,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast,
        beta_slow,
    )
    return q_out, k_out


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _tile_order_indices(grid_h: int, grid_w: int, merge_size: int) -> torch.Tensor:
    order = []
    for y in range(0, grid_h, merge_size):
        for x in range(0, grid_w, merge_size):
            for dy in range(merge_size):
                for dx in range(merge_size):
                    yy = y + dy
                    xx = x + dx
                    if yy >= grid_h or xx >= grid_w:
                        continue
                    order.append(yy * grid_w + xx)
    return torch.tensor(order, dtype=torch.long)


def _ref_position_embeddings_add_tiled_2d(
    x: torch.Tensor,
    pos: torch.Tensor,
    grid_h: int,
    grid_w: int,
    merge_size: int,
    source_grid_size: int | None = None,
) -> torch.Tensor:
    if source_grid_size is None:
        source_grid_size = grid_h
    if source_grid_size == grid_h and source_grid_size == grid_w:
        order = _tile_order_indices(grid_h, grid_w, merge_size)
        return x + pos.index_select(0, order)

    def f32(v) -> float:
        return np.float32(v).item()

    x_np = x.detach().cpu().numpy().astype(np.float32, copy=True)
    pos_np = pos.detach().cpu().numpy().astype(np.float32, copy=False)
    tokens, embed_dim = x_np.shape
    if tokens != grid_h * grid_w:
        raise ValueError(f"expected {grid_h * grid_w} tokens, got {tokens}")
    if pos_np.shape[0] != source_grid_size * source_grid_size:
        raise ValueError(
            f"expected {source_grid_size * source_grid_size} source positions, got {pos_np.shape[0]}"
        )

    sf_x = f32(float(grid_w) / float(source_grid_size))
    sf_y = f32(float(grid_h) / float(source_grid_size))
    pixel_offset = np.float32(0.5).item()
    support_x = f32(max(1.0, f32(1.0 / sf_x)))
    support_y = f32(max(1.0, f32(1.0 / sf_y)))
    invscale_x = f32(1.0 / support_x)
    invscale_y = f32(1.0 / support_y)

    order = _tile_order_indices(grid_h, grid_w, merge_size).tolist()
    for tok, src_tok in enumerate(order):
        dst_y = src_tok // grid_w
        dst_x = src_tok - dst_y * grid_w
        x_src = f32(f32(float(dst_x) + pixel_offset) / sf_x)
        y_src = f32(f32(float(dst_y) + pixel_offset) / sf_y)
        x_min = int(f32(f32(x_src - support_x) + pixel_offset))
        x_max = int(f32(f32(x_src + support_x) + pixel_offset))
        y_min = int(f32(f32(y_src - support_y) + pixel_offset))
        y_max = int(f32(f32(y_src + support_y) + pixel_offset))
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(source_grid_size, x_max)
        y_max = min(source_grid_size, y_max)

        # Channels are independent and use the same interpolation weights.
        # Vectorize that dimension while retaining the scalar oracle's
        # FP32-FMA rounding after every source sample. float64 represents the
        # exact product and sum of two float32 values before the explicit cast.
        val = np.zeros(embed_dim, dtype=np.float32)
        total_weight = f32(0.0)
        for sy in range(y_min, y_max):
            wy_arg = f32(f32(f32(float(sy) - y_src) + pixel_offset) * invscale_y)
            wy = f32(max(f32(1.0 - abs(wy_arg)), 0.0))
            if wy <= 0.0:
                continue
            for sx in range(x_min, x_max):
                wx_arg = f32(f32(f32(float(sx) - x_src) + pixel_offset) * invscale_x)
                wx = f32(max(f32(1.0 - abs(wx_arg)), 0.0))
                weight = f32(wx * wy)
                if weight <= 0.0:
                    continue
                sample = pos_np[sy * source_grid_size + sx]
                val = (
                    sample.astype(np.float64) * float(weight)
                    + val.astype(np.float64)
                ).astype(np.float32)
                total_weight = f32(total_weight + weight)
        if total_weight > 0.0:
            interpolated = (val / np.float32(total_weight)).astype(np.float32)
            x_np[tok] = (x_np[tok] + interpolated).astype(np.float32)
    return torch.from_numpy(x_np)


def _ref_position_embeddings_add_tiled_2d_align_corners(
    x: torch.Tensor,
    pos: torch.Tensor,
    grid_h: int,
    grid_w: int,
    merge_size: int,
    source_grid_size: int,
) -> torch.Tensor:
    ys = torch.linspace(0, source_grid_size - 1, grid_h, dtype=torch.float32)
    xs = torch.linspace(0, source_grid_size - 1, grid_w, dtype=torch.float32)
    y0 = ys.to(torch.long)
    x0 = xs.to(torch.long)
    y1 = (y0 + 1).clamp(max=source_grid_size - 1)
    x1 = (x0 + 1).clamp(max=source_grid_size - 1)
    dy = ys - y0
    dx = xs - x0
    table = pos.view(source_grid_size, source_grid_size, -1)
    interp = (
        table[y0[:, None], x0[None, :]] * ((1 - dy)[:, None] * (1 - dx)[None, :])[..., None]
        + table[y0[:, None], x1[None, :]] * ((1 - dy)[:, None] * dx[None, :])[..., None]
        + table[y1[:, None], x0[None, :]] * (dy[:, None] * (1 - dx)[None, :])[..., None]
        + table[y1[:, None], x1[None, :]] * (dy[:, None] * dx[None, :])[..., None]
    ).reshape(grid_h * grid_w, -1)
    return x + interp.index_select(0, _tile_order_indices(grid_h, grid_w, merge_size))


def _ref_add_stream_reorder_2d(
    main_input: torch.Tensor,
    aux_input: torch.Tensor,
    grid_h: int,
    grid_w: int,
    merge_size: int,
) -> torch.Tensor:
    order = _tile_order_indices(grid_h, grid_w, merge_size)
    summed = main_input + aux_input
    return summed.index_select(0, order)


def _ref_spatial_merge_contiguous_tiled(x: torch.Tensor, merge_size: int) -> torch.Tensor:
    merge_factor = merge_size * merge_size
    rows, embed_dim = x.shape
    return x.view(rows // merge_factor, merge_factor, embed_dim).reshape(rows // merge_factor, merge_factor * embed_dim)

def test_im2patch(C=3, H=224, W=224, P=16):
    print(f"\n--- Testing im2patch (Image {H}x{W}, Patch {P}) ---")
    torch.manual_seed(42)
    image = torch.randn(C, H, W, dtype=torch.float32)

    # PyTorch reference using unfold
    # unfold(dimension, size, step)
    # To get non-overlapping patches:
    # 1. Unfold H: [C, H/P, W, P]
    # 2. Unfold W: [C, H/P, W/P, P, P]
    # 3. Permute and reshape
    t0 = time.time()
    unfold = torch.nn.Unfold(kernel_size=(P, P), stride=(P, P))
    # input must be [B, C, H, W]
    ref = unfold(image.unsqueeze(0)) # [1, C*P*P, num_patches]
    ref = ref.squeeze(0).transpose(0, 1) # [num_patches, C*P*P]
    t_torch = time.time() - t0

    t0 = time.time()
    out_c = run_c_im2patch(image, P)
    t_c = time.time() - t0

    diff = max_diff(out_c, ref)
    print(f"PyTorch time: {t_torch*1000:.3f} ms")
    print(f"C Kernel time: {t_c*1000:.3f} ms")
    print(f"Max diff: {diff:.2e}")

    if diff > 1e-7:
        raise AssertionError("im2patch forward mismatch!")


def test_patch2im(C=3, H=224, W=224, P=16):
    print(f"\n--- Testing patch2im (Image {H}x{W}, Patch {P}) ---")
    torch.manual_seed(43)
    num_patches = (H // P) * (W // P)
    patch_dim = C * P * P
    d_patches = torch.randn(num_patches, patch_dim, dtype=torch.float32)

    # PyTorch reference using fold
    # fold(output_size, kernel_size, stride)
    t0 = time.time()
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=(P, P), stride=(P, P))
    # input must be [B, C*P*P, num_patches]
    ref = fold(d_patches.transpose(0, 1).unsqueeze(0)) # [1, C, H, W]
    ref = ref.squeeze(0)
    t_torch = time.time() - t0

    t0 = time.time()
    out_c = run_c_patch2im(d_patches, C, H, W, P)
    t_c = time.time() - t0

    diff = max_diff(out_c, ref)
    print(f"PyTorch time: {t_torch*1000:.3f} ms")
    print(f"C Kernel time: {t_c*1000:.3f} ms")
    print(f"Max diff: {diff:.2e}")

    if diff > 1e-7:
        raise AssertionError("patch2im backward mismatch!")


def test_position_embeddings_add(tokens=2304, embed_dim=1152):
    print(f"\n--- Testing position_embeddings_add ({tokens} tokens, dim {embed_dim}) ---")
    torch.manual_seed(44)
    x = torch.randn(tokens, embed_dim, dtype=torch.float32)
    pos = torch.randn(tokens, embed_dim, dtype=torch.float32)

    t0 = time.time()
    ref = x + pos
    t_torch = time.time() - t0

    t0 = time.time()
    out_c = run_c_position_embeddings_add(x, pos)
    t_c = time.time() - t0

    diff = max_diff(out_c, ref)
    print(f"PyTorch time: {t_torch*1000:.3f} ms")
    print(f"C Kernel time: {t_c*1000:.3f} ms")
    print(f"Max diff: {diff:.2e}")

    if diff > 1e-7:
        raise AssertionError("position_embeddings_add mismatch!")


def test_position_embeddings_add_tiled_2d(grid_h=6, grid_w=6, embed_dim=8, merge_size=3):
    print(
        f"\n--- Testing position_embeddings_add_tiled_2d "
        f"({grid_h}x{grid_w}, dim {embed_dim}, merge {merge_size}) ---"
    )
    tokens = grid_h * grid_w
    x = torch.arange(tokens * embed_dim, dtype=torch.float32).view(tokens, embed_dim)
    pos = (1000.0 + torch.arange(tokens * embed_dim, dtype=torch.float32)).view(tokens, embed_dim)

    ref = _ref_position_embeddings_add_tiled_2d(x, pos, grid_h, grid_w, merge_size)

    t0 = time.time()
    out_c = run_c_position_embeddings_add_tiled_2d(x, pos, grid_h, grid_w, merge_size)
    t_c = time.time() - t0

    diff = max_diff(out_c, ref)
    print(f"C Kernel time: {t_c*1000:.3f} ms")
    print(f"Max diff: {diff:.2e}")

    if diff > 1e-7:
        raise AssertionError("position_embeddings_add_tiled_2d mismatch!")


def test_position_embeddings_add_tiled_2d_qwen3vl_resize_order(
    grid_h=56,
    grid_w=72,
    source_grid_size=48,
    embed_dim=1152,
    merge_size=2,
):
    print(
        f"\n--- Testing position_embeddings_add_tiled_2d Qwen3-VL resize "
        f"({grid_h}x{grid_w} from {source_grid_size}x{source_grid_size}, dim {embed_dim}, merge {merge_size}) ---"
    )
    tokens = grid_h * grid_w
    source_tokens = source_grid_size * source_grid_size
    g = torch.Generator().manual_seed(4408)
    x = torch.randn(tokens, embed_dim, generator=g, dtype=torch.float32) * 0.25
    pos = torch.randn(source_tokens, embed_dim, generator=g, dtype=torch.float32) * 0.5

    ref = _ref_position_embeddings_add_tiled_2d(
        x,
        pos,
        grid_h,
        grid_w,
        merge_size,
        source_grid_size,
    )
    out_c = run_c_position_embeddings_add_tiled_2d(
        x,
        pos,
        grid_h,
        grid_w,
        merge_size,
        source_grid_size,
    )

    diff = max_diff(out_c, ref)
    print(f"Max diff: {diff:.2e}")

    # This is the production Qwen3-VL geometry. The kernel uses an explicit
    # ordered FMA contract, so compiler-dependent reassociation is a failure.
    if diff != 0.0:
        raise AssertionError("position_embeddings_add_tiled_2d Qwen3-VL resize order mismatch!")


def test_position_embeddings_add_tiled_2d_ragged_edges():
    grid_h, grid_w, source_grid_size, embed_dim, merge_size = 5, 7, 4, 11, 3
    print(
        f"\n--- Testing position_embeddings_add_tiled_2d ragged edges "
        f"({grid_h}x{grid_w}, merge {merge_size}) ---"
    )
    g = torch.Generator().manual_seed(4409)
    x = torch.randn(grid_h * grid_w, embed_dim, generator=g, dtype=torch.float32)
    pos = torch.randn(source_grid_size**2, embed_dim, generator=g, dtype=torch.float32)
    ref = _ref_position_embeddings_add_tiled_2d(
        x, pos, grid_h, grid_w, merge_size, source_grid_size
    )
    out_c = run_c_position_embeddings_add_tiled_2d(
        x, pos, grid_h, grid_w, merge_size, source_grid_size
    )
    diff = max_diff(out_c, ref)
    print(f"Max diff: {diff:.2e}")
    if diff > 2.5e-5:
        raise AssertionError("position_embeddings_add_tiled_2d ragged edge mismatch!")


def test_position_embeddings_add_tiled_2d_align_corners():
    grid_h, grid_w, source_grid_size, embed_dim, merge_size = 6, 10, 4, 8, 2
    g = torch.Generator().manual_seed(4410)
    x = torch.randn(grid_h * grid_w, embed_dim, generator=g)
    pos = torch.randn(source_grid_size * source_grid_size, embed_dim, generator=g)
    ref = _ref_position_embeddings_add_tiled_2d_align_corners(
        x, pos, grid_h, grid_w, merge_size, source_grid_size
    )
    out_c = run_c_position_embeddings_add_tiled_2d_align_corners(
        x, pos, grid_h, grid_w, merge_size, source_grid_size
    )
    diff = max_diff(out_c, ref)
    print(f"\n--- Testing align-corners tiled position embeddings (max diff {diff:.2e}) ---")
    if diff > 2e-6:
        raise AssertionError("align-corners tiled position embeddings mismatch")


def test_rowwise_bias_add(tokens=2304, embed_dim=1152):
    print(f"\n--- Testing rowwise_bias_add ({tokens} tokens, dim {embed_dim}) ---")
    torch.manual_seed(46)
    x = torch.randn(tokens, embed_dim, dtype=torch.float32)
    bias = torch.randn(embed_dim, dtype=torch.float32)

    t0 = time.time()
    ref = x + bias.unsqueeze(0)
    t_torch = time.time() - t0

    t0 = time.time()
    out_c = run_c_rowwise_bias_add(x, bias)
    t_c = time.time() - t0

    diff = max_diff(out_c, ref)
    print(f"PyTorch time: {t_torch*1000:.3f} ms")
    print(f"C Kernel time: {t_c*1000:.3f} ms")
    print(f"Max diff: {diff:.2e}")

    if diff > 1e-7:
        raise AssertionError("rowwise_bias_add mismatch!")


def test_add_stream_reorder_2d(grid_h=6, grid_w=6, embed_dim=8, merge_size=3):
    print(
        f"\n--- Testing add_stream_reorder_2d "
        f"({grid_h}x{grid_w}, dim {embed_dim}, merge {merge_size}) ---"
    )
    tokens = grid_h * grid_w
    main_input = torch.arange(tokens * embed_dim, dtype=torch.float32).view(tokens, embed_dim)
    aux_input = (5000.0 + torch.arange(tokens * embed_dim, dtype=torch.float32)).view(tokens, embed_dim)

    ref = _ref_add_stream_reorder_2d(main_input, aux_input, grid_h, grid_w, merge_size)

    t0 = time.time()
    out_c = run_c_add_stream_reorder_2d(main_input, aux_input, grid_h, grid_w, merge_size)
    t_c = time.time() - t0

    diff = max_diff(out_c, ref)
    print(f"C Kernel time: {t_c*1000:.3f} ms")
    print(f"Max diff: {diff:.2e}")

    if diff > 1e-7:
        raise AssertionError("add_stream_reorder_2d mismatch!")


def test_spatial_merge_2x2(grid_h=48, grid_w=48, embed_dim=64):
    print(f"\n--- Testing spatial_merge_2x2 ({grid_h}x{grid_w}, dim {embed_dim}) ---")
    torch.manual_seed(45)
    tokens = grid_h * grid_w
    x = torch.randn(tokens, embed_dim, dtype=torch.float32)

    t0 = time.time()
    ref = x.view(grid_h, grid_w, embed_dim)
    ref = ref.view(grid_h // 2, 2, grid_w // 2, 2, embed_dim)
    ref = ref.permute(0, 2, 1, 3, 4).contiguous().view((grid_h // 2) * (grid_w // 2), embed_dim * 4)
    t_torch = time.time() - t0

    t0 = time.time()
    out_c = run_c_spatial_merge_2x2(x, grid_h, grid_w, embed_dim)
    t_c = time.time() - t0

    diff = max_diff(out_c, ref)
    print(f"PyTorch time: {t_torch*1000:.3f} ms")
    print(f"C Kernel time: {t_c*1000:.3f} ms")
    print(f"Max diff: {diff:.2e}")

    if diff > 1e-7:
        raise AssertionError("spatial_merge_2x2 mismatch!")


def test_spatial_merge_contiguous_tiled(grid_h=6, grid_w=6, embed_dim=5, merge_size=3):
    print(
        f"\n--- Testing spatial_merge_contiguous_tiled "
        f"({grid_h}x{grid_w}, dim {embed_dim}, merge {merge_size}) ---"
    )
    tokens = grid_h * grid_w
    x = torch.arange(tokens * embed_dim, dtype=torch.float32).view(tokens, embed_dim)

    ref = _ref_spatial_merge_contiguous_tiled(x, merge_size)

    t0 = time.time()
    out_c = run_c_spatial_merge_contiguous_tiled(x, grid_h, grid_w, merge_size)
    t_c = time.time() - t0

    diff = max_diff(out_c, ref)
    print(f"C Kernel time: {t_c*1000:.3f} ms")
    print(f"Max diff: {diff:.2e}")

    if diff > 1e-7:
        raise AssertionError("spatial_merge_contiguous_tiled mismatch!")


def test_feature_concat_2way(rows=576, main_dim=4096, branch_dim=12288):
    print(f"\n--- Testing feature_concat_2way ({rows} rows, main {main_dim}, branch {branch_dim}) ---")
    torch.manual_seed(47)
    main_input = torch.randn(rows, main_dim, dtype=torch.float32)
    branch_input = torch.randn(rows, branch_dim, dtype=torch.float32)

    t0 = time.time()
    ref = torch.cat([main_input, branch_input], dim=1)
    t_torch = time.time() - t0

    t0 = time.time()
    out_c = run_c_feature_concat_2way(main_input, branch_input)
    t_c = time.time() - t0

    diff = max_diff(out_c, ref)
    print(f"PyTorch time: {t_torch*1000:.3f} ms")
    print(f"C Kernel time: {t_c*1000:.3f} ms")
    print(f"Max diff: {diff:.2e}")

    if diff > 1e-7:
        raise AssertionError("feature_concat_2way mismatch!")


def test_feature_slice_copy(rows=6, src_dim=4, dst_dim=11, dst_feature_offset=3):
    print(f"\n--- Testing feature_slice_copy ({rows} rows, src {src_dim}, dst {dst_dim}, off {dst_feature_offset}) ---")
    torch.manual_seed(48)
    src = torch.randn(rows, src_dim, dtype=torch.float32)
    ref = torch.full((rows, dst_dim), -123.0, dtype=torch.float32)
    ref[:, dst_feature_offset:dst_feature_offset + src_dim] = src

    t0 = time.time()
    out_c = run_c_feature_slice_copy(src, dst_dim, dst_feature_offset)
    t_c = time.time() - t0

    diff = max_diff(out_c, ref)
    print(f"C Kernel time: {t_c*1000:.3f} ms")
    print(f"Max diff: {diff:.2e}")

    if diff > 1e-7:
        raise AssertionError("feature_slice_copy mismatch!")


def test_feature_concat(rows=5, main_dim=3, branch_slice_dim=2, num_branch_slices=3):
    print(
        f"\n--- Testing feature_concat ({rows} rows, main {main_dim}, "
        f"branch {branch_slice_dim} x {num_branch_slices}) ---"
    )
    torch.manual_seed(49)
    main_input = torch.randn(rows, main_dim, dtype=torch.float32)
    branch_slices = [
        torch.randn(rows, branch_slice_dim, dtype=torch.float32)
        for _ in range(num_branch_slices)
    ]
    ref = torch.cat([main_input] + branch_slices, dim=1)

    t0 = time.time()
    out_c = run_c_feature_concat(main_input, branch_slices)
    t_c = time.time() - t0

    diff = max_diff(out_c, ref)
    print(f"C Kernel time: {t_c*1000:.3f} ms")
    print(f"Max diff: {diff:.2e}")

    if diff > 1e-7:
        raise AssertionError("feature_concat mismatch!")


def test_feature_concat_inplace_expand(rows=5, main_dim=3, branch_slice_dim=2, num_branch_slices=3):
    print(
        f"\n--- Testing feature_concat in-place expand ({rows} rows, main {main_dim}, "
        f"branch {branch_slice_dim} x {num_branch_slices}) ---"
    )
    torch.manual_seed(50)
    main_input = torch.randn(rows, main_dim, dtype=torch.float32)
    branch_slices = [
        torch.randn(rows, branch_slice_dim, dtype=torch.float32)
        for _ in range(num_branch_slices)
    ]
    ref = torch.cat([main_input] + branch_slices, dim=1)
    branch_input = torch.stack(branch_slices, dim=0).contiguous()
    output = torch.empty_like(ref)
    output.view(-1)[: main_input.numel()] = main_input.reshape(-1)

    lib.feature_concat(
        tensor_to_ptr(output),
        tensor_to_ptr(branch_input),
        tensor_to_ptr(output),
        rows,
        main_dim,
        branch_slice_dim,
        num_branch_slices,
    )

    diff = max_diff(output, ref)
    print(f"Max diff: {diff:.2e}")
    if diff > 1e-7:
        raise AssertionError("feature_concat in-place expand mismatch!")


def _ref_vision_position_ids(grid_h: int, grid_w: int, merge_size: int) -> torch.Tensor:
    num_tokens = grid_h * grid_w
    out = torch.zeros((4, num_tokens), dtype=torch.int32)
    ptr = 0
    for y in range(0, grid_h, merge_size):
        for x in range(0, grid_w, merge_size):
            for dy in range(merge_size):
                for dx in range(merge_size):
                    yy = y + dy
                    xx = x + dx
                    if yy >= grid_h or xx >= grid_w:
                        continue
                    out[0, ptr] = yy
                    out[1, ptr] = xx
                    out[2, ptr] = yy
                    out[3, ptr] = xx
                    ptr += 1
    return out


def _ref_mrope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    n_dims: int,
    sections: list[int],
    freq_base: float = 10000.0,
    freq_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    def apply(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        theta_scale = freq_base ** (-2.0 / n_dims)
        sec_w = sections[0] + sections[1]
        sec_e = sec_w + sections[2]
        sect_dims = sum(sections)
        num_tokens = out.shape[1]
        for h in range(out.shape[0]):
            for tok in range(num_tokens):
                theta_t = float(positions[0, tok].item())
                theta_h = float(positions[1, tok].item())
                theta_w = float(positions[2, tok].item())
                theta_e = float(positions[3, tok].item())
                for chan in range(n_dims):
                    sector = chan % sect_dims if sect_dims > 0 else chan
                    if sector == 0:
                        theta_t = float(positions[0, tok].item())
                    elif sector == sections[0]:
                        theta_h = float(positions[1, tok].item())
                    elif sector == sec_w:
                        theta_w = float(positions[2, tok].item())
                    elif sector == sec_e:
                        theta_e = float(positions[3, tok].item())
                    theta = theta_t
                    if sections[0] <= sector < sec_w:
                        theta = theta_h
                    elif sec_w <= sector < sec_e:
                        theta = theta_w
                    elif sector >= sec_e:
                        theta = theta_e
                    angle = theta * freq_scale
                    c = math.cos(angle)
                    s = math.sin(angle)
                    x0 = float(out[h, tok, chan].item())
                    x1 = float(out[h, tok, chan + n_dims].item())
                    out[h, tok, chan] = x0 * c - x1 * s
                    out[h, tok, chan + n_dims] = x0 * s + x1 * c
                    theta_t *= theta_scale
                    theta_h *= theta_scale
                    theta_w *= theta_scale
                    theta_e *= theta_scale
        return out

    return apply(q), apply(k)


def _ref_qwen3vl_vision_mrope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    n_dims: int,
    freq_base: float = 10000.0,
    freq_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    def apply(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        head_dim = out.shape[-1]
        rotary_width = min(int(n_dims), int(head_dim))
        rope_pairs = rotary_width // 2
        axis_pairs = rope_pairs // 2
        if axis_pairs <= 0 or 2 * rope_pairs > head_dim:
            return out
        theta_scale = freq_base ** (-2.0 / float(rope_pairs))
        num_tokens = out.shape[1]
        for h in range(out.shape[0]):
            for tok in range(num_tokens):
                theta_y = float(positions[0, tok].item())
                theta_x = float(positions[1, tok].item())
                for pair in range(rope_pairs):
                    is_x_axis = pair >= axis_pairs
                    base_theta = theta_x if is_x_axis else theta_y
                    theta = base_theta * (theta_scale ** (pair % axis_pairs)) * freq_scale
                    c = math.cos(theta)
                    s = math.sin(theta)
                    x0 = float(out[h, tok, pair].item())
                    x1 = float(out[h, tok, pair + rope_pairs].item())
                    out[h, tok, pair] = x0 * c - x1 * s
                    out[h, tok, pair + rope_pairs] = x0 * s + x1 * c
        return out

    return apply(q), apply(k)


def _ref_ggml_vision_mrope_exact(
    x: torch.Tensor,
    positions: torch.Tensor,
    n_dims: int,
    sections: list[int],
    freq_base: float = 10000.0,
) -> torch.Tensor:
    """Reference ggml's FP32/libm/FMA vision M-RoPE contract."""
    libm = ctypes.CDLL("libm.so.6")
    cosf = libm.cosf
    sinf = libm.sinf
    fmaf = libm.fmaf
    cosf.argtypes = [ctypes.c_float]
    sinf.argtypes = [ctypes.c_float]
    fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    cosf.restype = ctypes.c_float
    sinf.restype = ctypes.c_float
    fmaf.restype = ctypes.c_float

    out = x.numpy().copy()
    pos = positions.numpy()
    rope_pairs = n_dims // 2
    powf = libm.powf
    powf.argtypes = [ctypes.c_float, ctypes.c_float]
    powf.restype = ctypes.c_float
    theta_scale = np.float32(powf(
        ctypes.c_float(freq_base),
        ctypes.c_float(np.float32(-2.0) / np.float32(rope_pairs)),
    ))
    for head in range(out.shape[0]):
        for token in range(out.shape[1]):
            theta_y = np.float32(pos[0, token])
            theta_x = np.float32(pos[1, token])
            for pair in range(rope_pairs):
                if pair == sections[0]:
                    theta_x = np.float32(pos[1, token])
                theta = theta_x if pair >= sections[0] else theta_y
                cosine = np.float32(cosf(ctypes.c_float(theta)))
                sine = np.float32(sinf(ctypes.c_float(theta)))
                x0 = np.float32(out[head, token, pair])
                x1 = np.float32(out[head, token, pair + rope_pairs])
                x1_sin = np.float32(x1 * sine)
                x1_cos = np.float32(x1 * cosine)
                out[head, token, pair] = np.float32(fmaf(
                    ctypes.c_float(x0), ctypes.c_float(cosine), ctypes.c_float(-x1_sin)
                ))
                out[head, token, pair + rope_pairs] = np.float32(fmaf(
                    ctypes.c_float(x0), ctypes.c_float(sine), ctypes.c_float(x1_cos)
                ))
                theta_y = np.float32(theta_y * theta_scale)
                theta_x = np.float32(theta_x * theta_scale)
    return torch.from_numpy(out)


def test_vision_position_ids(grid_h=4, grid_w=4, merge_size=2):
    print(f"\n--- Testing vision_position_ids_2d_merge ({grid_h}x{grid_w}, merge {merge_size}) ---")
    ref = _ref_vision_position_ids(grid_h, grid_w, merge_size)
    out_c = run_c_vision_position_ids(grid_h, grid_w, merge_size)
    diff = (out_c - ref).abs().max().item()
    print(f"Max diff: {diff}")

    if diff != 0:
        raise AssertionError("vision_position_ids_2d_merge mismatch!")


def test_mrope_qk_vision(num_heads=2, num_kv_heads=2, num_tokens=4, head_dim=8):
    print(f"\n--- Testing mrope_qk_vision ({num_heads} heads, {num_tokens} tokens, dim {head_dim}) ---")
    torch.manual_seed(50)
    q = torch.randn(num_heads, num_tokens, head_dim, dtype=torch.float32)
    k = torch.randn(num_kv_heads, num_tokens, head_dim, dtype=torch.float32)
    positions = _ref_vision_position_ids(2, 2, 2)
    axis_pairs = max(1, head_dim // 4)
    sections = [axis_pairs, axis_pairs, 0, 0]
    n_dims = head_dim

    ref_q, ref_k = _ref_qwen3vl_vision_mrope_qk(q, k, positions, n_dims)
    out_q, out_k = run_c_mrope_qk(q, k, positions, n_dims, sections)

    q_diff = max_diff(out_q, ref_q)
    k_diff = max_diff(out_k, ref_k)
    print(f"Q max diff: {q_diff:.2e}")
    print(f"K max diff: {k_diff:.2e}")

    if q_diff > 1e-6 or k_diff > 1e-6:
        raise AssertionError("mrope_qk_vision mismatch!")


def test_mrope_qk_vision_storage_matrix():
    rows = []
    for head_dim in (8, 72):
        torch.manual_seed(5000 + head_dim)
        q = torch.randn(2, 4, head_dim, dtype=torch.float32)
        k = torch.randn(2, 4, head_dim, dtype=torch.float32)
        positions = _ref_vision_position_ids(2, 2, 2)
        axis_pairs = head_dim // 4
        sections = [axis_pairs, axis_pairs, 0, 0]
        ref_q, ref_k = _ref_qwen3vl_vision_mrope_qk(q, k, positions, head_dim)
        variants = (
            ("fp32_fp32_fp32", "mrope_qk_vision", ref_q, ref_k, 1e-6),
            ("fp32_fp32_bf16", "mrope_qk_vision_bf16_storage", ref_q.to(torch.bfloat16).float(), ref_k.to(torch.bfloat16).float(), 0.0),
            ("fp32_fp32_fp16", "mrope_qk_vision_fp16_storage", ref_q.to(torch.float16).float(), ref_k.to(torch.float16).float(), 0.0),
        )
        for capability, function, oracle_q, oracle_k, tolerance in variants:
            out_q, out_k = run_c_mrope_qk(q, k, positions, head_dim, sections, function=function)
            max_abs = max(max_diff(out_q, oracle_q), max_diff(out_k, oracle_k))
            if max_abs > tolerance:
                raise AssertionError(f"{capability} head_dim={head_dim} max_abs={max_abs} > {tolerance}")
            rows.append({
                "kernel_family": "vision_mrope",
                "capability": capability,
                "input_storage": "fp32",
                "compute": "fp32",
                "reduction": "none",
                "output_storage": capability.rsplit("_", 1)[-1],
                "oracle": "pytorch_formula_and_dtype_cast",
                "head_dim": head_dim,
                "threads": "serial_independent_heads",
                "max_abs": max_abs,
                "status": "pass",
            })
    report = {"schema": "cke.v8.numerical_capability_evidence", "schema_version": 1, "status": "pass", "rows": rows}
    report_path = Path(os.environ.get("CK_NUMERICAL_CAPABILITY_REPORT", "version/v8/.cache/reports/mrope_capabilities_latest.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def test_mrope_qk_vision_qwen3vl_full_head():
    test_mrope_qk_vision(num_heads=2, num_kv_heads=2, num_tokens=4, head_dim=72)


def test_mrope_qk_vision_ggml_libm_exact():
    """Catch ICX libimf trigonometric drift at production rotary width."""
    torch.manual_seed(713)
    head_dim = 72
    sections = [18, 18, 0, 0]
    q = torch.randn(1, 72, head_dim, dtype=torch.float32)
    k = torch.randn(1, 72, head_dim, dtype=torch.float32)
    positions = _ref_vision_position_ids(1, 72, 1)
    ref_q = _ref_ggml_vision_mrope_exact(q, positions, head_dim, sections)
    ref_k = _ref_ggml_vision_mrope_exact(k, positions, head_dim, sections)
    out_q, out_k = run_c_mrope_qk(q, k, positions, head_dim, sections)
    q_differing = int(torch.count_nonzero(out_q.view(torch.int32) != ref_q.view(torch.int32)))
    k_differing = int(torch.count_nonzero(out_k.view(torch.int32) != ref_k.view(torch.int32)))
    print(
        "GGML/libm-exact vision M-RoPE: "
        f"Q different={q_differing}/{out_q.numel()}, "
        f"K different={k_differing}/{out_k.numel()}"
    )
    if q_differing or k_differing:
        raise AssertionError("vision M-RoPE changed the ggml system-libm contract")


if __name__ == "__main__":
    test_im2patch()
    test_patch2im()
    test_position_embeddings_add()
    test_position_embeddings_add_tiled_2d()
    test_position_embeddings_add_tiled_2d_qwen3vl_resize_order()
    test_position_embeddings_add_tiled_2d_ragged_edges()
    test_position_embeddings_add_tiled_2d_align_corners()
    test_vision_position_ids()
    test_vision_position_ids(grid_h=6, grid_w=6, merge_size=3)
    test_rowwise_bias_add()
    test_add_stream_reorder_2d()
    test_spatial_merge_2x2()
    test_spatial_merge_contiguous_tiled()
    test_feature_slice_copy()
    test_feature_concat()
    test_feature_concat_inplace_expand()
    test_mrope_qk_vision()
    test_mrope_qk_vision_qwen3vl_full_head()
    test_mrope_qk_vision_ggml_libm_exact()
    test_mrope_qk_vision_storage_matrix()
    
    # Test a non-multiple size just in case (though ViT usually uses multiples)
    test_im2patch(C=3, H=32, W=32, P=8)
    test_patch2im(C=3, H=32, W=32, P=8)
