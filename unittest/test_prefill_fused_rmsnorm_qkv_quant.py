#!/usr/bin/env python3
"""
Fused RMSNorm + QKV prefill kernel unit tests (Q8 activations).

Tests fused kernel against SEPARATE kernels with Q5_0/Q8_0 weights.

The fusion benefit comes from keeping normed[] and q8_tile[] in L1/L2 cache:
- Separate: rmsnorm() writes normed[] -> quantize -> q8[] -> 3x gemv reads (may evict)
- Fused:    normed[] + q8_tile[] stay hot in L2 for all QKV projections

Key insight: We test fused vs separate with SAME quantized weights to measure
actual fusion benefit, not FP32 vs quantized difference.
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np
import torch

# Add unittest dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info, max_diff, numpy_to_ptr,
    time_function, print_system_info, TimingResult
)

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'

# Data type enum (must match ckernel_engine.h)
CK_DT_Q5_0 = 6
CK_DT_Q8_0 = 9


def ptr_f32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def ptr_void(arr: np.ndarray):
    return ctypes.c_void_p(arr.ctypes.data)


# ═══════════════════════════════════════════════════════════════════════════════
# Load Library and Setup
# ═══════════════════════════════════════════════════════════════════════════════

lib = load_lib("libckernel_engine.so")

# RMSNorm
lib.rmsnorm_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
lib.rmsnorm_forward.restype = None

# Quantize row Q8_0
lib.quantize_row_q8_0.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
]
lib.quantize_row_q8_0.restype = None

# GEMM Q8_0 x Q8_0
lib.gemm_nt_q8_0_q8_0.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.gemm_nt_q8_0_q8_0.restype = None

# Check for fused kernel
HAS_FUSED = False
try:
    lib.fused_rmsnorm_qkv_prefill_head_major_quant.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # gamma
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # Wq, Bq, wq_dt
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # Wk, Bk, wk_dt
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # Wv, Bv, wv_dt
        ctypes.POINTER(ctypes.c_float),  # Q
        ctypes.POINTER(ctypes.c_float),  # K
        ctypes.POINTER(ctypes.c_float),  # V
        ctypes.c_int,  # seq_len
        ctypes.c_int,  # embed_dim
        ctypes.c_int,  # aligned_embed_dim
        ctypes.c_int,  # num_heads
        ctypes.c_int,  # num_kv_heads
        ctypes.c_int,  # head_dim
        ctypes.c_int,  # aligned_head_dim
        ctypes.c_int,  # kv_stride_tokens
        ctypes.c_float,  # eps
        ctypes.c_void_p,  # scratch
    ]
    lib.fused_rmsnorm_qkv_prefill_head_major_quant.restype = None

    lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size.argtypes = [ctypes.c_int]
    lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size.restype = ctypes.c_size_t
    HAS_FUSED = True
except AttributeError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_matrix_q8_0(w_f32: np.ndarray, row_bytes: int) -> np.ndarray:
    """Quantize FP32 matrix to Q8_0 format."""
    rows = w_f32.shape[0]
    q8 = np.zeros(rows * row_bytes, dtype=np.uint8)
    for r in range(rows):
        row_ptr = ptr_f32(w_f32[r])
        dst_ptr = ctypes.c_void_p(q8.ctypes.data + r * row_bytes)
        lib.quantize_row_q8_0(row_ptr, dst_ptr, ctypes.c_int(w_f32.shape[1]))
    return q8


def run_baseline(x, gamma, wq_q8, wk_q8, wv_q8, bq, bk, bv,
                 tokens, embed_dim, aligned_embed_dim,
                 num_heads, num_kv_heads, head_dim, aligned_head_dim,
                 row_bytes, eps):
    """Run separate RMSNorm -> Quantize -> GEMM (baseline)."""
    # Step 1: RMSNorm
    normed = np.zeros_like(x)
    lib.rmsnorm_forward(
        ptr_f32(x), ptr_f32(gamma), ptr_f32(normed), None,
        ctypes.c_int(tokens), ctypes.c_int(embed_dim),
        ctypes.c_int(aligned_embed_dim), ctypes.c_float(eps)
    )

    # Step 2: Quantize each row
    q8_rows = np.zeros(tokens * row_bytes, dtype=np.uint8)
    for t in range(tokens):
        row_ptr = ptr_f32(normed[t])
        dst_ptr = ctypes.c_void_p(q8_rows.ctypes.data + t * row_bytes)
        lib.quantize_row_q8_0(row_ptr, dst_ptr, ctypes.c_int(aligned_embed_dim))

    # Step 3: GEMM for Q, K, V
    q_out = np.zeros((num_heads, tokens, aligned_head_dim), dtype=np.float32)
    k_out = np.zeros((num_kv_heads, tokens, aligned_head_dim), dtype=np.float32)
    v_out = np.zeros((num_kv_heads, tokens, aligned_head_dim), dtype=np.float32)

    for h in range(num_heads):
        wq_head_ptr = ctypes.c_void_p(wq_q8.ctypes.data + h * aligned_head_dim * row_bytes)
        bq_head_ptr = ptr_f32(bq[h * aligned_head_dim:(h + 1) * aligned_head_dim])
        lib.gemm_nt_q8_0_q8_0(
            ptr_void(q8_rows), wq_head_ptr, bq_head_ptr, ptr_f32(q_out[h]),
            ctypes.c_int(tokens), ctypes.c_int(aligned_head_dim), ctypes.c_int(aligned_embed_dim)
        )

    for h in range(num_kv_heads):
        wk_head_ptr = ctypes.c_void_p(wk_q8.ctypes.data + h * aligned_head_dim * row_bytes)
        wv_head_ptr = ctypes.c_void_p(wv_q8.ctypes.data + h * aligned_head_dim * row_bytes)
        bk_head_ptr = ptr_f32(bk[h * aligned_head_dim:(h + 1) * aligned_head_dim])
        bv_head_ptr = ptr_f32(bv[h * aligned_head_dim:(h + 1) * aligned_head_dim])

        lib.gemm_nt_q8_0_q8_0(
            ptr_void(q8_rows), wk_head_ptr, bk_head_ptr, ptr_f32(k_out[h]),
            ctypes.c_int(tokens), ctypes.c_int(aligned_head_dim), ctypes.c_int(aligned_embed_dim)
        )
        lib.gemm_nt_q8_0_q8_0(
            ptr_void(q8_rows), wv_head_ptr, bv_head_ptr, ptr_f32(v_out[h]),
            ctypes.c_int(tokens), ctypes.c_int(aligned_head_dim), ctypes.c_int(aligned_embed_dim)
        )

    return q_out, k_out, v_out


def run_fused(x, gamma, wq_q8, wk_q8, wv_q8, bq, bk, bv,
              tokens, embed_dim, aligned_embed_dim,
              num_heads, num_kv_heads, head_dim, aligned_head_dim,
              row_bytes, eps, scratch):
    """Run fused RMSNorm + QKV kernel."""
    q_out = np.zeros((num_heads, tokens, aligned_head_dim), dtype=np.float32)
    k_out = np.zeros((num_kv_heads, tokens, aligned_head_dim), dtype=np.float32)
    v_out = np.zeros((num_kv_heads, tokens, aligned_head_dim), dtype=np.float32)

    lib.fused_rmsnorm_qkv_prefill_head_major_quant(
        ptr_f32(x), ptr_f32(gamma),
        ptr_void(wq_q8), ptr_f32(bq), ctypes.c_int(CK_DT_Q8_0),
        ptr_void(wk_q8), ptr_f32(bk), ctypes.c_int(CK_DT_Q8_0),
        ptr_void(wv_q8), ptr_f32(bv), ctypes.c_int(CK_DT_Q8_0),
        ptr_f32(q_out), ptr_f32(k_out), ptr_f32(v_out),
        ctypes.c_int(tokens),
        ctypes.c_int(embed_dim),
        ctypes.c_int(aligned_embed_dim),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(head_dim),
        ctypes.c_int(aligned_head_dim),
        ctypes.c_int(tokens),  # kv_stride_tokens
        ctypes.c_float(eps),
        ptr_void(scratch),
    )

    return q_out, k_out, v_out


# ═══════════════════════════════════════════════════════════════════════════════
# Test Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_accuracy_test(tokens: int, embed_dim: int = 896,
                      num_heads: int = 14, num_kv_heads: int = 2,
                      head_dim: int = 64) -> TestResult:
    """Run accuracy test for a specific token count."""
    aligned_embed_dim = embed_dim
    aligned_head_dim = head_dim
    eps = 1e-6

    np.random.seed(42 + tokens)
    x = np.random.randn(tokens, aligned_embed_dim).astype(np.float32)
    gamma = np.random.randn(aligned_embed_dim).astype(np.float32)

    # Create FP32 weights and quantize
    wq_f = np.random.randn(num_heads * aligned_head_dim, aligned_embed_dim).astype(np.float32)
    wk_f = np.random.randn(num_kv_heads * aligned_head_dim, aligned_embed_dim).astype(np.float32)
    wv_f = np.random.randn(num_kv_heads * aligned_head_dim, aligned_embed_dim).astype(np.float32)

    bq = np.random.randn(num_heads * aligned_head_dim).astype(np.float32)
    bk = np.random.randn(num_kv_heads * aligned_head_dim).astype(np.float32)
    bv = np.random.randn(num_kv_heads * aligned_head_dim).astype(np.float32)

    row_bytes = (aligned_embed_dim // 32) * 34  # Q8_0 block size
    wq_q8 = quantize_matrix_q8_0(wq_f, row_bytes)
    wk_q8 = quantize_matrix_q8_0(wk_f, row_bytes)
    wv_q8 = quantize_matrix_q8_0(wv_f, row_bytes)

    # Get scratch buffer
    scratch_size = lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size(
        ctypes.c_int(aligned_embed_dim)
    )
    scratch = np.zeros(scratch_size, dtype=np.uint8)

    # Run baseline
    q_ref, k_ref, v_ref = run_baseline(
        x, gamma, wq_q8, wk_q8, wv_q8, bq, bk, bv,
        tokens, embed_dim, aligned_embed_dim,
        num_heads, num_kv_heads, head_dim, aligned_head_dim,
        row_bytes, eps
    )

    # Run fused
    q_fused, k_fused, v_fused = run_fused(
        x, gamma, wq_q8, wk_q8, wv_q8, bq, bk, bv,
        tokens, embed_dim, aligned_embed_dim,
        num_heads, num_kv_heads, head_dim, aligned_head_dim,
        row_bytes, eps, scratch
    )

    # Compare
    max_q = max_diff(torch.from_numpy(q_ref), torch.from_numpy(q_fused))
    max_k = max_diff(torch.from_numpy(k_ref), torch.from_numpy(k_fused))
    max_v = max_diff(torch.from_numpy(v_ref), torch.from_numpy(v_fused))
    max_all = max(max_q, max_k, max_v)

    tol = 1e-4
    passed = max_all <= tol

    return TestResult(
        name=f"tokens={tokens}",
        passed=passed,
        max_diff=max_all,
        tolerance=tol,
    )


def run_perf_test(tokens: int, embed_dim: int = 896,
                  num_heads: int = 14, num_kv_heads: int = 2,
                  head_dim: int = 64,
                  warmup: int = 3, iterations: int = 10) -> TestResult:
    """Run performance test comparing fused vs baseline."""
    aligned_embed_dim = embed_dim
    aligned_head_dim = head_dim
    eps = 1e-6

    np.random.seed(42 + tokens)
    x = np.random.randn(tokens, aligned_embed_dim).astype(np.float32)
    gamma = np.random.randn(aligned_embed_dim).astype(np.float32)

    wq_f = np.random.randn(num_heads * aligned_head_dim, aligned_embed_dim).astype(np.float32)
    wk_f = np.random.randn(num_kv_heads * aligned_head_dim, aligned_embed_dim).astype(np.float32)
    wv_f = np.random.randn(num_kv_heads * aligned_head_dim, aligned_embed_dim).astype(np.float32)

    bq = np.random.randn(num_heads * aligned_head_dim).astype(np.float32)
    bk = np.random.randn(num_kv_heads * aligned_head_dim).astype(np.float32)
    bv = np.random.randn(num_kv_heads * aligned_head_dim).astype(np.float32)

    row_bytes = (aligned_embed_dim // 32) * 34
    wq_q8 = quantize_matrix_q8_0(wq_f, row_bytes)
    wk_q8 = quantize_matrix_q8_0(wk_f, row_bytes)
    wv_q8 = quantize_matrix_q8_0(wv_f, row_bytes)

    scratch_size = lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size(
        ctypes.c_int(aligned_embed_dim)
    )
    scratch = np.zeros(scratch_size, dtype=np.uint8)

    # Time baseline
    baseline_time = time_function(
        lambda: run_baseline(
            x, gamma, wq_q8, wk_q8, wv_q8, bq, bk, bv,
            tokens, embed_dim, aligned_embed_dim,
            num_heads, num_kv_heads, head_dim, aligned_head_dim,
            row_bytes, eps
        ),
        warmup=warmup,
        iterations=iterations,
        name="baseline"
    )

    # Time fused
    fused_time = time_function(
        lambda: run_fused(
            x, gamma, wq_q8, wk_q8, wv_q8, bq, bk, bv,
            tokens, embed_dim, aligned_embed_dim,
            num_heads, num_kv_heads, head_dim, aligned_head_dim,
            row_bytes, eps, scratch
        ),
        warmup=warmup,
        iterations=iterations,
        name="fused"
    )

    # Get accuracy for this run
    q_ref, k_ref, v_ref = run_baseline(
        x, gamma, wq_q8, wk_q8, wv_q8, bq, bk, bv,
        tokens, embed_dim, aligned_embed_dim,
        num_heads, num_kv_heads, head_dim, aligned_head_dim,
        row_bytes, eps
    )
    q_fused, k_fused, v_fused = run_fused(
        x, gamma, wq_q8, wk_q8, wv_q8, bq, bk, bv,
        tokens, embed_dim, aligned_embed_dim,
        num_heads, num_kv_heads, head_dim, aligned_head_dim,
        row_bytes, eps, scratch
    )
    max_q = max_diff(torch.from_numpy(q_ref), torch.from_numpy(q_fused))
    max_k = max_diff(torch.from_numpy(k_ref), torch.from_numpy(k_fused))
    max_v = max_diff(torch.from_numpy(v_ref), torch.from_numpy(v_fused))
    max_all = max(max_q, max_k, max_v)

    tol = 1e-4
    passed = max_all <= tol

    return TestResult(
        name=f"tokens={tokens}",
        passed=passed,
        max_diff=max_all,
        tolerance=tol,
        pytorch_time=baseline_time,  # Using "pytorch" slot for baseline
        kernel_time=fused_time,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fused RMSNorm+QKV Prefill Test (Q8 activations)")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer iterations)")
    parser.add_argument("--perf", action="store_true", help="Run performance tests")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iter", type=int, default=10, help="Benchmark iterations")
    args = parser.parse_args()

    cpu = get_cpu_info()

    print()
    print("=" * 80)
    print(f"  TEST: fused_rmsnorm_qkv_prefill_head_major_quant")
    print("=" * 80)
    print()
    print("  SYSTEM INFO")
    print("  " + "-" * 40)
    print(f"  CPU:        {cpu.model_name}")
    print(f"  Cores:      {cpu.num_cores}")
    print(f"  SIMD:       {cpu.best_simd}")

    if not HAS_FUSED:
        print()
        print(f"  {RED}ERROR: Fused kernel not found in library{RESET}")
        print("=" * 80)
        return 1

    # Qwen2-0.5B dimensions
    embed_dim = 896
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64

    print(f"  Dtype:      Q8_0 activations, Q8_0 weights")
    print(f"  Shape:      embed={embed_dim}, heads={num_heads}/{num_kv_heads}, head_dim={head_dim}")

    token_counts = [32, 64, 128, 256] if not args.quick else [32, 64]

    # ─────────────────────────────────────────────────────────────────────────────
    # Accuracy Tests
    # ─────────────────────────────────────────────────────────────────────────────
    print()
    print("  ACCURACY (fused vs baseline)")
    print("  " + "-" * 40)

    all_passed = True
    for tokens in token_counts:
        result = run_accuracy_test(
            tokens=tokens,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        status = f"{GREEN}PASS{RESET}" if result.passed else f"{RED}FAIL{RESET}"
        print(f"  tokens={tokens:<4}  max_diff={result.max_diff:.2e}  tol={result.tolerance:.0e}  [{status}]")
        if not result.passed:
            all_passed = False

    # ─────────────────────────────────────────────────────────────────────────────
    # Performance Tests
    # ─────────────────────────────────────────────────────────────────────────────
    if args.perf or not args.quick:
        print()
        print("  PERFORMANCE (fused vs baseline)")
        print("  " + "-" * 40)
        print(f"  {'tokens':<8}  {'baseline (us)':<15}  {'fused (us)':<15}  {'speedup':<10}")
        print("  " + "-" * 55)

        for tokens in token_counts:
            result = run_perf_test(
                tokens=tokens,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                warmup=args.warmup,
                iterations=args.iter,
            )
            speedup = result.speedup if result.speedup else 1.0
            if speedup >= 1.1:
                sp_color = GREEN
            elif speedup >= 0.95:
                sp_color = YELLOW
            else:
                sp_color = RED

            baseline_us = result.pytorch_time.mean_us if result.pytorch_time else 0
            fused_us = result.kernel_time.mean_us if result.kernel_time else 0

            print(f"  {tokens:<8}  {baseline_us:<15.1f}  {fused_us:<15.1f}  {sp_color}{speedup:.2f}x{RESET}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────────
    print()
    print("  " + "-" * 40)
    if all_passed:
        print(f"  {GREEN}ALL ACCURACY TESTS PASSED{RESET}")
    else:
        print(f"  {RED}SOME TESTS FAILED{RESET}")
    print("=" * 80)
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
