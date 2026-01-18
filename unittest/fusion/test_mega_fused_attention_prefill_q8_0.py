#!/usr/bin/env python3
"""
Unit test for mega_fused_attention_prefill_q8_0 (Q8_0 out-proj)
===============================================================

WHAT IT DOES:
    - Tests mega_fused_attention_prefill with Q8_0 output projection weights
    - Compares against baseline: fused_rmsnorm_qkv_prefill_head_major_quant
      + flash attention + Q8_0 out-proj (quantized attn_out + vec_dot_q8_0_q8_0)
    - Verifies numerical correctness with higher precision Q8_0 path

WHEN TO RUN:
    - After modifying mega_fused_attention_prefill_q8_0.c
    - When changing Q8_0 weight handling
    - As unit test for Q8_0 prefill attention variant

TRIGGERED BY:
    - Makefile PY_TESTS list (make test-full)
    - Direct execution for development testing

DEPENDENCIES:
    - build/libckernel_engine.so
    - unittest/lib_loader.py, unittest/test_utils.py

STATUS: ACTIVE - Unit test for Q8_0 prefill attention
"""
import ctypes
import os
import sys

import numpy as np

# Add unittest/ to path (parent of fusion/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib_loader import load_lib
from test_utils import TestReport, TestResult

CK_DT_Q8_0 = 9


def row_bytes_q8_0(n_elements: int) -> int:
    block_size, block_bytes = 32, 34
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * block_bytes


def ptr_f32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def ptr_void(arr: np.ndarray):
    return ctypes.c_void_p(arr.ctypes.data)


def make_q8_0_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes_q8_0(cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + 31) // 32
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * 34
            data[off:off + 2] = scale_bytes
    return data


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

    lib.quantize_row_q8_0.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.quantize_row_q8_0.restype = None

    lib.vec_dot_q8_0_q8_0.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.vec_dot_q8_0_q8_0.restype = None

    lib.mega_fused_attention_prefill_q8_0.argtypes = [
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
    lib.mega_fused_attention_prefill_q8_0.restype = None

    lib.mega_fused_attention_prefill_q8_0_scratch_size.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.mega_fused_attention_prefill_q8_0_scratch_size.restype = ctypes.c_size_t

    tokens = 4
    embed_dim = 64
    num_heads = 2
    num_kv_heads = 1
    head_dim = 32
    aligned_embed_dim = embed_dim
    aligned_head_dim = head_dim
    cache_capacity = tokens
    eps = 1e-6

    x = np.random.randn(tokens, aligned_embed_dim).astype(np.float32)
    gamma = np.random.randn(aligned_embed_dim).astype(np.float32)

    wq = make_q8_0_weights(num_heads * aligned_head_dim, aligned_embed_dim)
    wk = make_q8_0_weights(num_kv_heads * aligned_head_dim, aligned_embed_dim)
    wv = make_q8_0_weights(num_kv_heads * aligned_head_dim, aligned_embed_dim)
    wo = make_q8_0_weights(aligned_embed_dim, aligned_embed_dim)
    bq = np.random.randn(num_heads * aligned_head_dim).astype(np.float32)
    bk = np.random.randn(num_kv_heads * aligned_head_dim).astype(np.float32)
    bv = np.random.randn(num_kv_heads * aligned_head_dim).astype(np.float32)

    q = np.zeros((num_heads, tokens, aligned_head_dim), dtype=np.float32)
    kv_cache_k = np.zeros((num_kv_heads, cache_capacity, aligned_head_dim), dtype=np.float32)
    kv_cache_v = np.zeros_like(kv_cache_k)
    attn_out = np.zeros_like(q)

    qkv_scratch_bytes = lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size(
        ctypes.c_int(aligned_embed_dim)
    )
    qkv_scratch = np.zeros(qkv_scratch_bytes, dtype=np.uint8)

    lib.fused_rmsnorm_qkv_prefill_head_major_quant(
        ptr_f32(x),
        ptr_f32(gamma),
        ptr_void(wq), ptr_f32(bq), ctypes.c_int(CK_DT_Q8_0),
        ptr_void(wk), ptr_f32(bk), ctypes.c_int(CK_DT_Q8_0),
        ptr_void(wv), ptr_f32(bv), ctypes.c_int(CK_DT_Q8_0),
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

    q8_row_bytes = row_bytes_q8_0(aligned_head_dim)
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
    block_bytes = 34

    out_ref = np.zeros((tokens, aligned_embed_dim), dtype=np.float32)
    for t in range(tokens):
        for n in range(aligned_embed_dim):
            total = 0.0
            w_row_off = n * blocks_per_row * block_bytes
            for h in range(num_heads):
                w_head_off = w_row_off + h * blocks_per_head * block_bytes
                a_row_off = (h * tokens + t) * q8_row_bytes
                partial = ctypes.c_float(0.0)
                lib.vec_dot_q8_0_q8_0(
                    ctypes.c_int(aligned_head_dim),
                    ctypes.byref(partial),
                    ctypes.c_void_p(wo.ctypes.data + w_head_off),
                    ctypes.c_void_p(attn_q8.ctypes.data + a_row_off),
                )
                total += partial.value
            out_ref[t, n] = total

    scratch_bytes = lib.mega_fused_attention_prefill_q8_0_scratch_size(
        ctypes.c_int(tokens),
        ctypes.c_int(aligned_embed_dim),
        ctypes.c_int(num_heads),
        ctypes.c_int(aligned_head_dim),
    )
    scratch = np.zeros(scratch_bytes, dtype=np.uint8)
    out_fused = np.zeros_like(out_ref)

    rope_null = ctypes.c_void_p(0)
    lib.mega_fused_attention_prefill_q8_0(
        ptr_f32(out_fused),
        ptr_f32(x),
        None,
        ptr_f32(gamma),
        ptr_void(wq), ptr_f32(bq), ctypes.c_int(CK_DT_Q8_0),
        ptr_void(wk), ptr_f32(bk), ctypes.c_int(CK_DT_Q8_0),
        ptr_void(wv), ptr_f32(bv), ctypes.c_int(CK_DT_Q8_0),
        ptr_void(wo), None, ctypes.c_int(CK_DT_Q8_0),
        ptr_f32(kv_cache_k),
        ptr_f32(kv_cache_v),
        rope_null,
        rope_null,
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

    diff = float(np.max(np.abs(out_ref - out_fused)))
    tol = 1e-4
    return TestResult("mega_fused_attention_prefill_q8_0", diff <= tol, diff, tol)


if __name__ == "__main__":
    report = TestReport("mega_fused_attention_prefill_q8_0")
    report.add_result(run_case())
    report.print_report()
    sys.exit(0 if report.all_passed() else 1)
