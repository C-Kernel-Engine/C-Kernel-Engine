#!/usr/bin/env python3
"""
Unit test for mega_fused_attention_prefill (Q5_0 weights)
=========================================================

WHAT IT DOES:
    - Tests mega_fused_attention_prefill with Q5_0 quantized weights
    - Compares against baseline: fused_rmsnorm_qkv_prefill_head_major_quant
      + flash attention + quantized out-proj + residual add
    - Verifies numerical correctness of the fused kernel

WHEN TO RUN:
    - After modifying mega_fused_attention_prefill.c
    - When changing Q5_0 weight handling in prefill path
    - As unit test for prefill attention fusion

TRIGGERED BY:
    - Makefile PY_TESTS list (make test-full)
    - Direct execution for development testing

DEPENDENCIES:
    - build/libckernel_engine.so
    - unittest/lib_loader.py, unittest/test_utils.py

STATUS: ACTIVE - Unit test for Q5_0 prefill attention
"""
import ctypes
import os
import sys

import numpy as np
import torch

# Add unittest/ to path (parent of fusion/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib_loader import load_lib
from test_utils import TestReport, TestResult, max_diff

CK_DT_Q5_0 = 11


def row_bytes_q5_0(n_elements: int) -> int:
    block_size, block_bytes = 32, 22
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * block_bytes


def ptr_f32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def ptr_void(arr: np.ndarray):
    return ctypes.c_void_p(arr.ctypes.data)

def make_q5_0_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes_q5_0(cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + 31) // 32
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * 22
            data[off:off + 2] = scale_bytes
    return data


def flatten_head_major(attn_out: np.ndarray, tokens: int, num_heads: int, aligned_head_dim: int) -> np.ndarray:
    aligned_embed_dim = num_heads * aligned_head_dim
    flat = np.zeros((tokens, aligned_embed_dim), dtype=np.float32)
    for t in range(tokens):
        for h in range(num_heads):
            flat[t, h * aligned_head_dim:(h + 1) * aligned_head_dim] = attn_out[h, t]
    return flat


def run_case() -> TestResult:
    lib = load_lib("libckernel_engine.so")

    lib.fused_rmsnorm_qkv_prefill_head_major_quant.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    lib.fused_rmsnorm_qkv_prefill_head_major_quant.restype = None

    lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size.argtypes = [ctypes.c_int]
    lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size.restype = ctypes.c_size_t

    lib.attention_forward_causal_head_major_gqa_flash_strided.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.attention_forward_causal_head_major_gqa_flash_strided.restype = None

    lib.ck_gemm_nt_quant.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.ck_gemm_nt_quant.restype = None

    lib.ck_residual_add_token_major.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.ck_residual_add_token_major.restype = None

    try:
        lib.mega_fused_attention_prefill.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # residual
            ctypes.POINTER(ctypes.c_float),  # ln1_gamma
            ctypes.c_void_p,                 # wq
            ctypes.POINTER(ctypes.c_float),  # bq
            ctypes.c_int,                    # wq_dt
            ctypes.c_void_p,                 # wk
            ctypes.POINTER(ctypes.c_float),  # bk
            ctypes.c_int,                    # wk_dt
            ctypes.c_void_p,                 # wv
            ctypes.POINTER(ctypes.c_float),  # bv
            ctypes.c_int,                    # wv_dt
            ctypes.c_void_p,                 # wo
            ctypes.POINTER(ctypes.c_float),  # bo
            ctypes.c_int,                    # wo_dt
            ctypes.POINTER(ctypes.c_float),  # kv_cache_k
            ctypes.POINTER(ctypes.c_float),  # kv_cache_v
            ctypes.c_void_p,                 # rope_cos
            ctypes.c_void_p,                 # rope_sin
            ctypes.c_int,                    # start_pos
            ctypes.c_int,                    # tokens
            ctypes.c_int,                    # cache_capacity
            ctypes.c_int,                    # embed_dim
            ctypes.c_int,                    # aligned_embed_dim
            ctypes.c_int,                    # num_heads
            ctypes.c_int,                    # num_kv_heads
            ctypes.c_int,                    # head_dim
            ctypes.c_int,                    # aligned_head_dim
            ctypes.c_float,                  # eps
            ctypes.c_void_p,                 # scratch
        ]
        lib.mega_fused_attention_prefill.restype = None

        lib.mega_fused_attention_prefill_scratch_size.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.mega_fused_attention_prefill_scratch_size.restype = ctypes.c_size_t
    except AttributeError:
        return TestResult(
            name="mega_fused_attention_prefill",
            passed=False,
            max_diff=0.0,
            tolerance=0.0,
        )

    tokens = 4
    embed_dim = 128
    aligned_embed_dim = 128
    num_heads = 4
    num_kv_heads = 2
    head_dim = 32
    aligned_head_dim = 32
    cache_capacity = tokens
    eps = 1e-6

    np.random.seed(42)
    x = np.random.randn(tokens, aligned_embed_dim).astype(np.float32)
    residual = np.random.randn(tokens, aligned_embed_dim).astype(np.float32)
    gamma = np.random.randn(aligned_embed_dim).astype(np.float32)

    wq = make_q5_0_weights(num_heads * aligned_head_dim, aligned_embed_dim)
    wk = make_q5_0_weights(num_kv_heads * aligned_head_dim, aligned_embed_dim)
    wv = make_q5_0_weights(num_kv_heads * aligned_head_dim, aligned_embed_dim)
    wo = make_q5_0_weights(aligned_embed_dim, aligned_embed_dim)

    bq = np.random.randn(num_heads * aligned_head_dim).astype(np.float32)
    bk = np.random.randn(num_kv_heads * aligned_head_dim).astype(np.float32)
    bv = np.random.randn(num_kv_heads * aligned_head_dim).astype(np.float32)

    kv_cache_k = np.zeros((num_kv_heads, cache_capacity, aligned_head_dim), dtype=np.float32)
    kv_cache_v = np.zeros_like(kv_cache_k)

    q = np.zeros((num_heads, tokens, aligned_head_dim), dtype=np.float32)

    # Baseline
    qkv_scratch_size = lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size(ctypes.c_int(aligned_embed_dim))
    qkv_scratch = np.zeros(qkv_scratch_size, dtype=np.uint8)

    lib.fused_rmsnorm_qkv_prefill_head_major_quant(
        ptr_f32(x),
        ptr_f32(gamma),
        ptr_void(wq), ptr_f32(bq), ctypes.c_int(CK_DT_Q5_0),
        ptr_void(wk), ptr_f32(bk), ctypes.c_int(CK_DT_Q5_0),
        ptr_void(wv), ptr_f32(bv), ctypes.c_int(CK_DT_Q5_0),
        ptr_f32(q),
        ptr_f32(kv_cache_k),
        ptr_f32(kv_cache_v),
        ctypes.c_int(tokens),
        ctypes.c_int(embed_dim),
        ctypes.c_int(aligned_embed_dim),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(head_dim),
        ctypes.c_int(aligned_head_dim),
        ctypes.c_int(cache_capacity),
        ctypes.c_float(eps),
        ptr_void(qkv_scratch),
    )

    attn_out = np.zeros_like(q)
    lib.attention_forward_causal_head_major_gqa_flash_strided(
        ptr_f32(q),
        ptr_f32(kv_cache_k),
        ptr_f32(kv_cache_v),
        ptr_f32(attn_out),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(tokens),
        ctypes.c_int(head_dim),
        ctypes.c_int(aligned_head_dim),
        ctypes.c_int(cache_capacity),
    )

    proj_in = flatten_head_major(attn_out, tokens, num_heads, aligned_head_dim)
    out_ref = np.zeros_like(x)
    lib.ck_gemm_nt_quant(ptr_f32(proj_in), ptr_void(wo), None, ptr_f32(out_ref),
                         ctypes.c_int(tokens), ctypes.c_int(aligned_embed_dim),
                         ctypes.c_int(aligned_embed_dim), ctypes.c_int(CK_DT_Q5_0))
    lib.ck_residual_add_token_major(ptr_f32(residual), ptr_f32(out_ref), ptr_f32(out_ref),
                                    ctypes.c_int(tokens), ctypes.c_int(aligned_embed_dim))

    # Fused
    out_fused = np.zeros_like(out_ref)
    scratch_size = lib.mega_fused_attention_prefill_scratch_size(
        ctypes.c_int(tokens), ctypes.c_int(aligned_embed_dim),
        ctypes.c_int(num_heads), ctypes.c_int(aligned_head_dim)
    )
    scratch = np.zeros(scratch_size, dtype=np.uint8)

    lib.mega_fused_attention_prefill(
        ptr_f32(out_fused),
        ptr_f32(x),
        ptr_f32(residual),
        ptr_f32(gamma),
        ptr_void(wq), ptr_f32(bq), ctypes.c_int(CK_DT_Q5_0),
        ptr_void(wk), ptr_f32(bk), ctypes.c_int(CK_DT_Q5_0),
        ptr_void(wv), ptr_f32(bv), ctypes.c_int(CK_DT_Q5_0),
        ptr_void(wo), None, ctypes.c_int(CK_DT_Q5_0),
        ptr_f32(kv_cache_k),
        ptr_f32(kv_cache_v),
        None,
        None,
        ctypes.c_int(0),
        ctypes.c_int(tokens),
        ctypes.c_int(cache_capacity),
        ctypes.c_int(embed_dim),
        ctypes.c_int(aligned_embed_dim),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(head_dim),
        ctypes.c_int(aligned_head_dim),
        ctypes.c_float(eps),
        ptr_void(scratch),
    )

    max_d = max_diff(torch.from_numpy(out_ref), torch.from_numpy(out_fused))
    tol = 1e-4
    return TestResult(
        name="mega_fused_attention_prefill",
        passed=max_d <= tol,
        max_diff=max_d,
        tolerance=tol,
    )


def main() -> int:
    report = TestReport("mega_fused_attention_prefill")
    report.add_result(run_case())
    report.print_report()
    return 0 if report.all_passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
