#!/usr/bin/env python3
"""
Unit test for mega_fused_outproj_mlp_prefill (out-proj + RMSNorm2 + MLP)
========================================================================

WHAT IT DOES:
    - Tests mega_fused_outproj_mlp_prefill kernel correctness
    - Compares against baseline of separate operations:
      * head-major out-proj (Q5_0 weights x Q8_0 activations)
      * residual add
      * RMSNorm2
      * fused_mlp_swiglu_prefill_w1w2_quant
      * residual add
    - Verifies the full post-attention fusion pipeline

WHEN TO RUN:
    - After modifying mega_fused_outproj_mlp_prefill.c
    - When changing MLP fusion or residual handling
    - As unit test for OutProj+MLP fusion

TRIGGERED BY:
    - Makefile PY_TESTS list (make test-full)
    - Direct execution for development testing

DEPENDENCIES:
    - build/libckernel_engine.so
    - unittest/lib_loader.py, unittest/test_utils.py

STATUS: ACTIVE - Unit test for OutProj+MLP prefill fusion
"""
import ctypes
import os
import sys

import numpy as np

# Add unittest/ to path (parent of fusion/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib_loader import load_lib
from test_utils import TestReport, TestResult

CK_DT_Q4_K = 7
CK_DT_Q6_K = 8
CK_DT_Q8_0 = 9
CK_DT_Q8_K = 10
CK_DT_Q5_0 = 11


def row_bytes(dt: int, n_elements: int) -> int:
    if dt == CK_DT_Q5_0:
        block_size, block_bytes = 32, 22
    elif dt == CK_DT_Q8_0:
        block_size, block_bytes = 32, 34
    elif dt == CK_DT_Q4_K:
        block_size, block_bytes = 256, 144
    elif dt == CK_DT_Q6_K:
        block_size, block_bytes = 256, 210
    elif dt == CK_DT_Q8_K:
        block_size, block_bytes = 256, 292
    else:
        raise ValueError(f"unsupported dt={dt}")
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * block_bytes


def ptr_f32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def ptr_void(arr: np.ndarray):
    return ctypes.c_void_p(arr.ctypes.data)


def make_q5_0_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes(CK_DT_Q5_0, cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + 31) // 32
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * 22
            data[off:off + 2] = scale_bytes
    return data


def make_q4_k_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes(CK_DT_Q4_K, cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    dmin_bytes = np.frombuffer(np.float16(0.0).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + 255) // 256
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * 144
            data[off:off + 2] = scale_bytes
            data[off + 2:off + 4] = dmin_bytes
    return data


def make_q6_k_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes(CK_DT_Q6_K, cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + 255) // 256
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * 210
            data[off + 208:off + 210] = scale_bytes
    return data


def run_case(w2_dt: int, name: str) -> TestResult:
    lib = load_lib("libckernel_engine.so")

    lib.quantize_row_q8_0.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.quantize_row_q8_0.restype = None

    lib.vec_dot_q5_0_q8_0.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.vec_dot_q5_0_q8_0.restype = None

    lib.rmsnorm_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib.rmsnorm_forward.restype = None

    lib.fused_mlp_swiglu_prefill_w1w2_quant.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.fused_mlp_swiglu_prefill_w1w2_quant.restype = None

    lib.fused_mlp_swiglu_prefill_w1w2_quant_scratch_size.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.fused_mlp_swiglu_prefill_w1w2_quant_scratch_size.restype = ctypes.c_size_t

    lib.mega_fused_outproj_mlp_prefill.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.POINTER(ctypes.c_float),  # attn_out (head-major)
        ctypes.POINTER(ctypes.c_float),  # residual
        ctypes.POINTER(ctypes.c_float),  # ln2_gamma
        ctypes.c_void_p,                 # wo
        ctypes.POINTER(ctypes.c_float),  # bo
        ctypes.c_int,                    # wo_dt
        ctypes.c_void_p,                 # w1
        ctypes.POINTER(ctypes.c_float),  # b1
        ctypes.c_int,                    # w1_dt
        ctypes.c_void_p,                 # w2
        ctypes.POINTER(ctypes.c_float),  # b2
        ctypes.c_int,                    # w2_dt
        ctypes.c_int,                    # tokens
        ctypes.c_int,                    # embed_dim
        ctypes.c_int,                    # aligned_embed_dim
        ctypes.c_int,                    # num_heads
        ctypes.c_int,                    # aligned_head_dim
        ctypes.c_int,                    # intermediate_dim
        ctypes.c_int,                    # aligned_intermediate_dim
        ctypes.c_float,                  # eps
        ctypes.c_void_p,                 # scratch
    ]
    lib.mega_fused_outproj_mlp_prefill.restype = None

    lib.mega_fused_outproj_mlp_prefill_scratch_size.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.mega_fused_outproj_mlp_prefill_scratch_size.restype = ctypes.c_size_t

    tokens = 4
    embed_dim = 64
    aligned_embed_dim = 64
    num_heads = 2
    head_dim = 32
    aligned_head_dim = 32
    intermediate = 256
    aligned_intermediate = 256
    eps = 1e-6

    np.random.seed(123)
    attn_out = np.random.randn(num_heads, tokens, aligned_head_dim).astype(np.float32)
    residual = np.random.randn(tokens, aligned_embed_dim).astype(np.float32)
    ln2_gamma = np.random.randn(aligned_embed_dim).astype(np.float32)

    wo = make_q5_0_weights(aligned_embed_dim, aligned_embed_dim)
    w1 = make_q5_0_weights(2 * aligned_intermediate, aligned_embed_dim)
    if w2_dt == CK_DT_Q4_K:
        w2 = make_q4_k_weights(aligned_embed_dim, aligned_intermediate)
    else:
        w2 = make_q6_k_weights(aligned_embed_dim, aligned_intermediate)

    bo = np.random.randn(aligned_embed_dim).astype(np.float32)
    b1 = np.random.randn(2 * aligned_intermediate).astype(np.float32)
    b2 = np.random.randn(aligned_embed_dim).astype(np.float32)

    # Baseline out-proj (Q5_0 weights x Q8_0 activations)
    q8_row_bytes = row_bytes(CK_DT_Q8_0, aligned_head_dim)
    attn_q8 = np.zeros(num_heads * tokens * q8_row_bytes, dtype=np.uint8)
    for h in range(num_heads):
        for t in range(tokens):
            row = attn_out[h, t]
            off = (h * tokens + t) * q8_row_bytes
            lib.quantize_row_q8_0(ptr_f32(row),
                                  ctypes.c_void_p(attn_q8.ctypes.data + off),
                                  ctypes.c_int(aligned_head_dim))

    blocks_per_head = aligned_head_dim // 32
    blocks_per_row = aligned_embed_dim // 32
    block_bytes = 22

    proj = np.zeros((tokens, aligned_embed_dim), dtype=np.float32)
    for t in range(tokens):
        for n in range(aligned_embed_dim):
            total = bo[n]
            w_row_off = n * blocks_per_row * block_bytes
            for h in range(num_heads):
                w_head_off = w_row_off + h * blocks_per_head * block_bytes
                a_row_off = (h * tokens + t) * q8_row_bytes
                partial = ctypes.c_float(0.0)
                lib.vec_dot_q5_0_q8_0(
                    ctypes.c_int(aligned_head_dim),
                    ctypes.byref(partial),
                    ctypes.c_void_p(wo.ctypes.data + w_head_off),
                    ctypes.c_void_p(attn_q8.ctypes.data + a_row_off),
                )
                total += partial.value
            proj[t, n] = total

    h1 = proj + residual

    ln2_out = np.zeros_like(h1)
    rstd_cache = np.zeros(tokens, dtype=np.float32)
    lib.rmsnorm_forward(ptr_f32(h1),
                        ptr_f32(ln2_gamma),
                        ptr_f32(ln2_out),
                        ptr_f32(rstd_cache),
                        ctypes.c_int(tokens),
                        ctypes.c_int(embed_dim),
                        ctypes.c_int(aligned_embed_dim),
                        ctypes.c_float(eps))

    mlp_scratch_bytes = lib.fused_mlp_swiglu_prefill_w1w2_quant_scratch_size(
        ctypes.c_int(aligned_embed_dim),
        ctypes.c_int(aligned_intermediate),
    )
    mlp_scratch = np.zeros(mlp_scratch_bytes, dtype=np.uint8)
    mlp_out = np.zeros_like(h1)

    lib.fused_mlp_swiglu_prefill_w1w2_quant(ptr_f32(ln2_out),
                                           ptr_void(w1),
                                           ptr_f32(b1),
                                           ctypes.c_int(CK_DT_Q5_0),
                                           ptr_void(w2),
                                           ptr_f32(b2),
                                           ctypes.c_int(w2_dt),
                                           ptr_f32(mlp_out),
                                           ctypes.c_int(tokens),
                                           ctypes.c_int(embed_dim),
                                           ctypes.c_int(aligned_embed_dim),
                                           ctypes.c_int(intermediate),
                                           ctypes.c_int(aligned_intermediate),
                                           ptr_void(mlp_scratch))

    out_ref = mlp_out + h1

    scratch_bytes = lib.mega_fused_outproj_mlp_prefill_scratch_size(
        ctypes.c_int(tokens),
        ctypes.c_int(aligned_embed_dim),
        ctypes.c_int(num_heads),
        ctypes.c_int(aligned_head_dim),
        ctypes.c_int(aligned_intermediate),
    )
    scratch = np.zeros(scratch_bytes, dtype=np.uint8)
    out_fused = np.zeros_like(out_ref)

    lib.mega_fused_outproj_mlp_prefill(ptr_f32(out_fused),
                                       ptr_f32(attn_out),
                                       ptr_f32(residual),
                                       ptr_f32(ln2_gamma),
                                       ptr_void(wo),
                                       ptr_f32(bo),
                                       ctypes.c_int(CK_DT_Q5_0),
                                       ptr_void(w1),
                                       ptr_f32(b1),
                                       ctypes.c_int(CK_DT_Q5_0),
                                       ptr_void(w2),
                                       ptr_f32(b2),
                                       ctypes.c_int(w2_dt),
                                       ctypes.c_int(tokens),
                                       ctypes.c_int(embed_dim),
                                       ctypes.c_int(aligned_embed_dim),
                                       ctypes.c_int(num_heads),
                                       ctypes.c_int(aligned_head_dim),
                                       ctypes.c_int(intermediate),
                                       ctypes.c_int(aligned_intermediate),
                                       ctypes.c_float(eps),
                                       ptr_void(scratch))

    diff = float(np.max(np.abs(out_ref - out_fused)))
    tol = 1e-4
    return TestResult(name, diff <= tol, diff, tol)


if __name__ == "__main__":
    report = TestReport("mega_fused_outproj_mlp_prefill")
    report.add_result(run_case(CK_DT_Q4_K, "outproj_mlp_q4_k"))
    report.add_result(run_case(CK_DT_Q6_K, "outproj_mlp_q6_k"))
    report.print_report()
    sys.exit(0 if report.all_passed() else 1)
