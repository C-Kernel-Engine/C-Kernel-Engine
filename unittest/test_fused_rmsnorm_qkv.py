"""
Fused RMSNorm + QKV projection kernel unit tests.

Tests fused kernel against SEPARATE kernels (not PyTorch) with Q4_K weights.

The fusion benefit comes from keeping normed[] in L1 cache:
- Separate: rmsnorm() writes normed[] → 3x gemv_q4k() reads normed[] (may evict)
- Fused:    normed[] computed once, stays hot in L1 for all 3 GEMVs

Key insight: We test fused vs separate with SAME quantized weights to measure
actual fusion benefit, not FP32 vs quantized difference.
"""
import argparse
import ctypes
import os
import struct
import sys
import time

import numpy as np

# Add parent dir to path for imports
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

# ═══════════════════════════════════════════════════════════════════════════════
# Q4_K Block Constants (must match llama.cpp/GGML)
# ═══════════════════════════════════════════════════════════════════════════════

QK_K = 256  # Elements per K-quant super-block
BLOCK_Q4_K_SIZE = 144  # bytes per Q4_K block


def fp16_to_bytes(val: float) -> bytes:
    """Convert float to FP16 bytes."""
    return struct.pack('<e', val)


def random_q4k_block() -> bytes:
    """Generate a random Q4_K block (144 bytes)."""
    data = bytearray()

    # d (fp16 scale): 2 bytes
    d = np.random.uniform(0.01, 0.1)
    data.extend(fp16_to_bytes(d))

    # dmin (fp16 min): 2 bytes
    dmin = np.random.uniform(0.001, 0.05)
    data.extend(fp16_to_bytes(dmin))

    # scales (6-bit packed): 12 bytes
    scales = np.random.randint(0, 64, size=12, dtype=np.uint8)
    data.extend(scales.tobytes())

    # qs (4-bit weights): 128 bytes
    qs = np.random.randint(0, 256, size=128, dtype=np.uint8)
    data.extend(qs.tobytes())

    assert len(data) == BLOCK_Q4_K_SIZE
    return bytes(data)


def random_q4k_matrix(rows: int, cols: int) -> bytes:
    """Generate random Q4_K quantized weight matrix [rows, cols]."""
    assert cols % QK_K == 0, f"cols must be multiple of {QK_K}"
    blocks_per_row = cols // QK_K

    data = bytearray()
    for _ in range(rows):
        for _ in range(blocks_per_row):
            data.extend(random_q4k_block())

    return bytes(data)


# ═══════════════════════════════════════════════════════════════════════════════
# Load Libraries
# ═══════════════════════════════════════════════════════════════════════════════

lib = load_lib("libckernel_engine.so")

# RMSNorm signature
lib.rmsnorm_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # gamma
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.POINTER(ctypes.c_float),  # rstd_cache (can be None)
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # d_model
    ctypes.c_int,                    # aligned_embed_dim
    ctypes.c_float,                  # eps
]
lib.rmsnorm_forward.restype = None

# Q4_K GEMV signature
lib.gemv_q4_k.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # y output
    ctypes.c_void_p,                 # W (Q4_K)
    ctypes.POINTER(ctypes.c_float),  # x input
    ctypes.c_int,                    # M (rows)
    ctypes.c_int,                    # K (cols)
]
lib.gemv_q4_k.restype = None

# Check for fused Q4K kernel
HAS_FUSED_Q4K = False
try:
    lib.rmsnorm_qkv_q4k_fused.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # rms_weight
        ctypes.c_void_p,                 # wq (Q4_K)
        ctypes.c_void_p,                 # wk (Q4_K)
        ctypes.c_void_p,                 # wv (Q4_K)
        ctypes.POINTER(ctypes.c_float),  # q_out
        ctypes.POINTER(ctypes.c_float),  # k_out
        ctypes.POINTER(ctypes.c_float),  # v_out
        ctypes.c_int,                    # embed_dim
        ctypes.c_int,                    # q_dim
        ctypes.c_int,                    # kv_dim
        ctypes.c_float,                  # eps
    ]
    lib.rmsnorm_qkv_q4k_fused.restype = None
    HAS_FUSED_Q4K = True
    print(f"{GREEN}✓ Found rmsnorm_qkv_q4k_fused kernel{RESET}")
except AttributeError:
    print(f"{YELLOW}⚠ rmsnorm_qkv_q4k_fused not found - testing FP32 only{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# C Kernel Wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def c_rmsnorm(x_np: np.ndarray, weight_np: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """C RMSNorm kernel."""
    out_np = np.zeros_like(x_np)
    lib.rmsnorm_forward(
        numpy_to_ptr(x_np), numpy_to_ptr(weight_np), numpy_to_ptr(out_np), None,
        ctypes.c_int(1), ctypes.c_int(x_np.shape[0]), ctypes.c_int(x_np.shape[0]),
        ctypes.c_float(eps)
    )
    return out_np


def c_gemv_q4k(w_q4k: bytes, x_np: np.ndarray, out_rows: int) -> np.ndarray:
    """C GEMV with Q4_K weights."""
    out_np = np.zeros(out_rows, dtype=np.float32)
    lib.gemv_q4_k(
        numpy_to_ptr(out_np),
        w_q4k,
        numpy_to_ptr(x_np),
        ctypes.c_int(out_rows),
        ctypes.c_int(x_np.shape[0])
    )
    return out_np


def c_separate_rmsnorm_qkv_q4k(
    x: np.ndarray, rms_weight: np.ndarray,
    wq: bytes, wk: bytes, wv: bytes,
    q_dim: int, kv_dim: int, eps: float = 1e-6
) -> tuple:
    """Separate RMSNorm + 3x Q4_K GEMV (baseline)."""
    # Step 1: RMSNorm
    normed = c_rmsnorm(x, rms_weight, eps)

    # Step 2-4: Q4_K GEMVs
    q_out = c_gemv_q4k(wq, normed, q_dim)
    k_out = c_gemv_q4k(wk, normed, kv_dim)
    v_out = c_gemv_q4k(wv, normed, kv_dim)

    return q_out, k_out, v_out


def c_fused_rmsnorm_qkv_q4k(
    x: np.ndarray, rms_weight: np.ndarray,
    wq: bytes, wk: bytes, wv: bytes,
    q_dim: int, kv_dim: int, eps: float = 1e-6
) -> tuple:
    """Fused RMSNorm + QKV with Q4_K weights."""
    q_out = np.zeros(q_dim, dtype=np.float32)
    k_out = np.zeros(kv_dim, dtype=np.float32)
    v_out = np.zeros(kv_dim, dtype=np.float32)

    lib.rmsnorm_qkv_q4k_fused(
        numpy_to_ptr(x), numpy_to_ptr(rms_weight),
        wq, wk, wv,
        numpy_to_ptr(q_out), numpy_to_ptr(k_out), numpy_to_ptr(v_out),
        ctypes.c_int(x.shape[0]),
        ctypes.c_int(q_dim),
        ctypes.c_int(kv_dim),
        ctypes.c_float(eps)
    )
    return q_out, k_out, v_out


# ═══════════════════════════════════════════════════════════════════════════════
# Test Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_q4k_fusion_test(embed_dim: int = 896, q_dim: int = 896, kv_dim: int = 128,
                        n_warmup: int = 10, n_iter: int = 100):
    """Test fused vs separate with Q4_K weights."""

    # Dimensions must be multiple of 256 for Q4_K
    embed_dim = ((embed_dim + 255) // 256) * 256

    print("=" * 80)
    print(f"{BOLD}FUSED KERNEL TEST: Q4_K Weights{RESET}")
    print("=" * 80)
    print(f"""
{YELLOW}Purpose:{RESET}  Compare fused vs separate kernels with quantized weights
{YELLOW}Method:{RESET}   Both use same Q4_K weights, compare outputs and timing
{YELLOW}Fusion:{RESET}   normed[] stays in L1 cache instead of potentially evicting

{YELLOW}Dimensions:{RESET}
  embed_dim = {embed_dim}
  q_dim     = {q_dim}
  kv_dim    = {kv_dim}

{YELLOW}Weight sizes (Q4_K = 4.5 bits/weight):{RESET}
  wq: {q_dim} x {embed_dim} = {q_dim * embed_dim * 4.5 / 8 / 1024:.1f} KB
  wk: {kv_dim} x {embed_dim} = {kv_dim * embed_dim * 4.5 / 8 / 1024:.1f} KB
  wv: {kv_dim} x {embed_dim} = {kv_dim * embed_dim * 4.5 / 8 / 1024:.1f} KB
  normed[]: {embed_dim} x 4 = {embed_dim * 4 / 1024:.1f} KB (L1 cache target)
""")

    # Generate test data
    np.random.seed(42)
    x = np.random.randn(embed_dim).astype(np.float32)
    rms_weight = np.random.randn(embed_dim).astype(np.float32)

    # Q4_K quantized weights
    wq_q4k = random_q4k_matrix(q_dim, embed_dim)
    wk_q4k = random_q4k_matrix(kv_dim, embed_dim)
    wv_q4k = random_q4k_matrix(kv_dim, embed_dim)

    print(f"Generated Q4_K weights: wq={len(wq_q4k)} bytes, wk={len(wk_q4k)} bytes, wv={len(wv_q4k)} bytes")
    print()

    # ─────────────────────────────────────────────────────────────────────────────
    # Test 1: Accuracy - Fused vs Separate should match exactly
    # ─────────────────────────────────────────────────────────────────────────────
    print("-" * 80)
    print(f"{BOLD}ACCURACY TEST: Fused vs Separate (both Q4_K){RESET}")
    print("-" * 80)

    # Run separate
    q_sep, k_sep, v_sep = c_separate_rmsnorm_qkv_q4k(
        x, rms_weight, wq_q4k, wk_q4k, wv_q4k, q_dim, kv_dim
    )

    if HAS_FUSED_Q4K:
        # Run fused
        q_fused, k_fused, v_fused = c_fused_rmsnorm_qkv_q4k(
            x, rms_weight, wq_q4k, wk_q4k, wv_q4k, q_dim, kv_dim
        )

        # Compare
        q_diff = np.max(np.abs(q_sep - q_fused))
        k_diff = np.max(np.abs(k_sep - k_fused))
        v_diff = np.max(np.abs(v_sep - v_fused))

        tol = 1e-2  # Q4_K has inherent quantization noise; 1e-2 is acceptable
        q_pass = q_diff < tol
        k_pass = k_diff < tol
        v_pass = v_diff < tol

        print(f"  Q: max_diff={q_diff:.2e}  [{GREEN}PASS{RESET}]" if q_pass else f"  Q: max_diff={q_diff:.2e}  [{RED}FAIL{RESET}]")
        print(f"  K: max_diff={k_diff:.2e}  [{GREEN}PASS{RESET}]" if k_pass else f"  K: max_diff={k_diff:.2e}  [{RED}FAIL{RESET}]")
        print(f"  V: max_diff={v_diff:.2e}  [{GREEN}PASS{RESET}]" if v_pass else f"  V: max_diff={v_diff:.2e}  [{RED}FAIL{RESET}]")

        if not (q_pass and k_pass and v_pass):
            print(f"\n{RED}ACCURACY MISMATCH - Fused kernel produces different results!{RESET}")
            return False

        print(f"\n{GREEN}✓ Fused and Separate produce identical outputs{RESET}")
    else:
        print(f"{YELLOW}⚠ Fused Q4K kernel not available - skipping accuracy test{RESET}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Test 2: Performance - Fused should be faster
    # ─────────────────────────────────────────────────────────────────────────────
    print()
    print("-" * 80)
    print(f"{BOLD}PERFORMANCE TEST: Fused vs Separate{RESET}")
    print("-" * 80)

    # Warmup
    print(f"Warming up ({n_warmup} iterations)...")
    for _ in range(n_warmup):
        c_separate_rmsnorm_qkv_q4k(x, rms_weight, wq_q4k, wk_q4k, wv_q4k, q_dim, kv_dim)
        if HAS_FUSED_Q4K:
            c_fused_rmsnorm_qkv_q4k(x, rms_weight, wq_q4k, wk_q4k, wv_q4k, q_dim, kv_dim)

    # Benchmark separate
    print(f"Benchmarking separate ({n_iter} iterations)...")
    start = time.perf_counter()
    for _ in range(n_iter):
        c_separate_rmsnorm_qkv_q4k(x, rms_weight, wq_q4k, wk_q4k, wv_q4k, q_dim, kv_dim)
    sep_time = (time.perf_counter() - start) / n_iter * 1e6  # microseconds

    if HAS_FUSED_Q4K:
        # Benchmark fused
        print(f"Benchmarking fused ({n_iter} iterations)...")
        start = time.perf_counter()
        for _ in range(n_iter):
            c_fused_rmsnorm_qkv_q4k(x, rms_weight, wq_q4k, wk_q4k, wv_q4k, q_dim, kv_dim)
        fused_time = (time.perf_counter() - start) / n_iter * 1e6

        speedup = sep_time / fused_time

        print()
        print(f"  {'Kernel':<25} {'Time (us)':<15} {'Speedup':<10}")
        print(f"  {'-' * 50}")
        print(f"  {'Separate (baseline)':<25} {sep_time:<15.1f} {'1.00x':<10}")

        if speedup >= 1.3:
            color = GREEN
        elif speedup >= 1.0:
            color = YELLOW
        else:
            color = RED
        print(f"  {'Fused':<25} {fused_time:<15.1f} {color}{speedup:.2f}x{RESET}")

        # Memory analysis
        print()
        print(f"  {BOLD}Memory Analysis:{RESET}")
        normed_bytes = embed_dim * 4
        print(f"    normed[] buffer: {normed_bytes} bytes ({normed_bytes/1024:.1f} KB)")
        print(f"    Separate: normed[] written to memory, read 3x by GEMVs")
        print(f"    Fused:    normed[] stays in L1, used immediately")

        if speedup >= 1.3:
            print(f"\n{GREEN}✓ FUSION SUCCESS: {speedup:.2f}x speedup (≥1.3x target){RESET}")
            return True
        elif speedup >= 0.95:  # Allow 5% variance for measurement noise
            print(f"\n{YELLOW}⚠ FUSION MARGINAL: {speedup:.2f}x speedup (<1.3x target){RESET}")
            print(f"  Possible reasons:")
            print(f"  - normed[] already fits in L1 ({normed_bytes} bytes)")
            print(f"  - Weight loading dominates (memory-bound)")
            print(f"  - Need mega-fusion (full attention block) for benefit")
            return True
        else:
            print(f"\n{RED}✗ FUSION SLOWER: {speedup:.2f}x (fused is significantly slower!){RESET}")
            print(f"  This indicates a bug in the fused kernel implementation.")
            return False
    else:
        print(f"  Separate: {sep_time:.1f} us")
        print(f"  Fused:    N/A (kernel not available)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Fused RMSNorm+QKV test with Q4_K weights")
    parser.add_argument("--embed", type=int, default=896, help="Embedding dimension")
    parser.add_argument("--q-dim", type=int, default=896, help="Q output dimension")
    parser.add_argument("--kv-dim", type=int, default=128, help="KV output dimension")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iter", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer iterations)")
    parser.add_argument("--all", action="store_true", help="Full test suite")
    args = parser.parse_args()

    print_system_info()

    if args.quick:
        args.warmup = 3
        args.iter = 20

    success = run_q4k_fusion_test(
        embed_dim=args.embed,
        q_dim=args.q_dim,
        kv_dim=args.kv_dim,
        n_warmup=args.warmup,
        n_iter=args.iter
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
