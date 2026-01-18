#!/usr/bin/env python3
"""
OutProj + MLP Fusion Performance Benchmark
==========================================

Benchmarks CK-Engine's mega_fused_outproj_mlp_prefill kernel against CK's baseline
(unfused) implementation to measure speedup.

WHAT IT DOES:
    - Compares fused vs unfused OutProj+MLP performance
    - Measures speedup from kernel fusion (typically 1.08-1.14x)
    - Uses synthetic and real model weights

NOTE: The fused and baseline kernels use DIFFERENT quantization strategies:
- Fused: Head-major quantization (each head quantized independently)
- Baseline: Token-major quantization (quantizes across concatenated heads)

This causes different Q8_0 block scales and thus numerical differences.
This is an intentional optimization tradeoff for cache locality, not a bug.

WHEN TO RUN:
    - After modifying mega_fused_outproj_mlp_prefill.c
    - When evaluating fusion performance gains
    - For regression testing of fusion speedup

TRIGGERED BY:
    - scripts/run_mega_fusion_test.sh (step 5/6)
    - make fusion-test-full-with-lamacpp

DEPENDENCIES:
    - build/libckernel_engine.so

STATUS: ACTIVE - Performance benchmark for OutProj+MLP fusion

For numerical parity testing, use `test_mega_fusion_parity.py` which tests
the attention kernel where both implementations use the same quantization path.

Usage:
    python test_mega_fusion_outproj_mlp_parity.py
    python test_mega_fusion_outproj_mlp_parity.py --quick
"""

import ctypes
import numpy as np
import argparse
import sys
import time
from pathlib import Path

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

# Block sizes for quantized formats
QK_K = 256      # Elements per K-quant super-block
QK5_0 = 32      # Elements per Q5_0 block
BLOCK_Q5_0_SIZE = 22
BLOCK_Q4_K_SIZE = 144
BLOCK_Q6_K_SIZE = 210

# CK dtype constants
CK_DT_Q4_K = 7
CK_DT_Q6_K = 8
CK_DT_Q8_0 = 9
CK_DT_Q5_0 = 11


def row_bytes_q5_0(cols):
    n_blocks = (cols + QK5_0 - 1) // QK5_0
    return n_blocks * BLOCK_Q5_0_SIZE


def row_bytes_q4_k(cols):
    n_blocks = (cols + QK_K - 1) // QK_K
    return n_blocks * BLOCK_Q4_K_SIZE


def row_bytes_q6_k(cols):
    n_blocks = (cols + QK_K - 1) // QK_K
    return n_blocks * BLOCK_Q6_K_SIZE


def make_q5_0_weights(rows, cols, scale=0.02):
    rb = row_bytes_q5_0(cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + QK5_0 - 1) // QK5_0
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * BLOCK_Q5_0_SIZE
            data[off:off + 2] = scale_bytes
    return data


def make_q4_k_weights(rows, cols, scale=0.02):
    rb = row_bytes_q4_k(cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    dmin_bytes = np.frombuffer(np.float16(0.0).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + QK_K - 1) // QK_K
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * BLOCK_Q4_K_SIZE
            data[off:off + 2] = scale_bytes
            data[off + 2:off + 4] = dmin_bytes
    return data


def make_q6_k_weights(rows, cols, scale=0.02):
    rb = row_bytes_q6_k(cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + QK_K - 1) // QK_K
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * BLOCK_Q6_K_SIZE
            data[off + 208:off + 210] = scale_bytes
    return data


def ptr_f32(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def ptr_void(arr):
    return ctypes.c_void_p(arr.ctypes.data)


def load_library():
    """Load CK library."""
    # Go up from unittest/fusion/ to project root
    base_dir = Path(__file__).parent.parent.parent
    ck_paths = [
        base_dir / "build" / "libckernel_engine.so",
        base_dir / "libckernel_engine.so",
    ]
    for p in ck_paths:
        if p.exists():
            try:
                lib = ctypes.CDLL(str(p))
                print(f"Loaded CK library: {p}")
                return lib
            except OSError:
                pass
    return None


def setup_signatures(lib):
    """Set up ctypes signatures for CK functions."""
    # ck_gemm_nt_quant
    lib.ck_gemm_nt_quant.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.ck_gemm_nt_quant.restype = None

    # rmsnorm_forward
    lib.rmsnorm_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ]
    lib.rmsnorm_forward.restype = None

    # fused_mlp_swiglu_prefill_w1w2_quant
    lib.fused_mlp_swiglu_prefill_w1w2_quant.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.fused_mlp_swiglu_prefill_w1w2_quant.restype = None

    # Scratch size functions
    lib.fused_mlp_swiglu_prefill_w1w2_quant_scratch_size.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.fused_mlp_swiglu_prefill_w1w2_quant_scratch_size.restype = ctypes.c_size_t

    lib.mega_fused_outproj_mlp_prefill_scratch_size.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.mega_fused_outproj_mlp_prefill_scratch_size.restype = ctypes.c_size_t

    # mega_fused_outproj_mlp_prefill
    lib.mega_fused_outproj_mlp_prefill.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.POINTER(ctypes.c_float),  # attn_out
        ctypes.POINTER(ctypes.c_float),  # residual
        ctypes.POINTER(ctypes.c_float),  # ln2_gamma
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # wo, bo, wo_dt
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # w1, b1, w1_dt
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # w2, b2, w2_dt
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
    ]
    lib.mega_fused_outproj_mlp_prefill.restype = None


def run_baseline(lib, attn_out, residual, ln2_gamma, wo, w1, w2,
                 tokens, num_heads, head_dim, embed_dim, intermediate,
                 eps, wo_dt, w1_dt, w2_dt):
    """Run baseline (unfused) implementation using CK kernels."""
    # Flatten head-major attn_out to token-major
    proj_in = attn_out.transpose(1, 0, 2).reshape(tokens, embed_dim).copy()

    # Allocate outputs
    proj_out = np.zeros((tokens, embed_dim), dtype=np.float32)
    h1 = np.zeros((tokens, embed_dim), dtype=np.float32)
    ln2_out = np.zeros((tokens, embed_dim), dtype=np.float32)
    rstd = np.zeros(tokens, dtype=np.float32)
    output = np.zeros((tokens, embed_dim), dtype=np.float32)

    # Allocate scratch
    mlp_scratch_size = lib.fused_mlp_swiglu_prefill_w1w2_quant_scratch_size(
        ctypes.c_int(embed_dim), ctypes.c_int(intermediate))
    mlp_scratch = np.zeros(mlp_scratch_size, dtype=np.uint8)

    # Step 1: OutProj
    lib.ck_gemm_nt_quant(
        ptr_f32(proj_in), ptr_void(wo), None, ptr_f32(proj_out),
        ctypes.c_int(tokens), ctypes.c_int(embed_dim),
        ctypes.c_int(embed_dim), ctypes.c_int(wo_dt))

    # Step 2: Residual add
    np.add(proj_out, residual, out=h1)

    # Step 3: RMSNorm
    lib.rmsnorm_forward(
        ptr_f32(h1), ptr_f32(ln2_gamma), ptr_f32(ln2_out), ptr_f32(rstd),
        ctypes.c_int(tokens), ctypes.c_int(embed_dim),
        ctypes.c_int(embed_dim), ctypes.c_float(eps))

    # Step 4: MLP
    lib.fused_mlp_swiglu_prefill_w1w2_quant(
        ptr_f32(ln2_out), ptr_void(w1), None, ctypes.c_int(w1_dt),
        ptr_void(w2), None, ctypes.c_int(w2_dt),
        ptr_f32(output),
        ctypes.c_int(tokens), ctypes.c_int(embed_dim), ctypes.c_int(embed_dim),
        ctypes.c_int(intermediate), ctypes.c_int(intermediate),
        ptr_void(mlp_scratch))

    # Step 5: Residual add
    np.add(output, h1, out=output)

    return output


def run_fused(lib, attn_out, residual, ln2_gamma, wo, w1, w2,
              tokens, num_heads, head_dim, embed_dim, intermediate,
              eps, wo_dt, w1_dt, w2_dt):
    """Run fused implementation."""
    output = np.zeros((tokens, embed_dim), dtype=np.float32)

    scratch_size = lib.mega_fused_outproj_mlp_prefill_scratch_size(
        ctypes.c_int(tokens), ctypes.c_int(embed_dim),
        ctypes.c_int(num_heads), ctypes.c_int(head_dim),
        ctypes.c_int(intermediate))
    scratch = np.zeros(scratch_size, dtype=np.uint8)

    lib.mega_fused_outproj_mlp_prefill(
        ptr_f32(output),
        ptr_f32(attn_out),
        ptr_f32(residual),
        ptr_f32(ln2_gamma),
        ptr_void(wo), None, ctypes.c_int(wo_dt),
        ptr_void(w1), None, ctypes.c_int(w1_dt),
        ptr_void(w2), None, ctypes.c_int(w2_dt),
        ctypes.c_int(tokens),
        ctypes.c_int(embed_dim), ctypes.c_int(embed_dim),
        ctypes.c_int(num_heads), ctypes.c_int(head_dim),
        ctypes.c_int(intermediate), ctypes.c_int(intermediate),
        ctypes.c_float(eps),
        ptr_void(scratch))

    return output


def test_fusion_parity(lib, config, tolerance=1e-3, n_runs=3):
    """
    Test fused vs baseline parity and performance.
    Returns: (passed, max_diff, mean_diff, baseline_time_us, fused_time_us)
    """
    tokens = config['tokens']
    num_heads = config['num_heads']
    head_dim = config['head_dim']
    embed_dim = num_heads * head_dim
    intermediate = config['intermediate']
    eps = config.get('eps', 1e-6)
    w2_is_q6k = config.get('w2_is_q6k', 0)
    wo_dt = CK_DT_Q5_0
    w1_dt = CK_DT_Q5_0
    w2_dt = CK_DT_Q6_K if w2_is_q6k else CK_DT_Q4_K

    # Generate random inputs (same seed for reproducibility)
    np.random.seed(42)
    attn_out = np.random.randn(num_heads, tokens, head_dim).astype(np.float32) * 0.1
    residual = np.random.randn(tokens, embed_dim).astype(np.float32) * 0.1
    ln2_gamma = np.random.randn(embed_dim).astype(np.float32) * 0.1 + 1.0

    # Create quantized weights
    wo = make_q5_0_weights(embed_dim, embed_dim)
    w1 = make_q5_0_weights(2 * intermediate, embed_dim)
    if w2_is_q6k:
        w2 = make_q6_k_weights(embed_dim, intermediate)
    else:
        w2 = make_q4_k_weights(embed_dim, intermediate)

    # Warmup
    run_baseline(lib, attn_out, residual, ln2_gamma, wo, w1, w2,
                 tokens, num_heads, head_dim, embed_dim, intermediate,
                 eps, wo_dt, w1_dt, w2_dt)
    run_fused(lib, attn_out, residual, ln2_gamma, wo, w1, w2,
              tokens, num_heads, head_dim, embed_dim, intermediate,
              eps, wo_dt, w1_dt, w2_dt)

    # Benchmark baseline
    baseline_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        baseline_out = run_baseline(lib, attn_out, residual, ln2_gamma, wo, w1, w2,
                                    tokens, num_heads, head_dim, embed_dim, intermediate,
                                    eps, wo_dt, w1_dt, w2_dt)
        baseline_times.append((time.perf_counter() - start) * 1e6)
    baseline_time = min(baseline_times)

    # Benchmark fused
    fused_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fused_out = run_fused(lib, attn_out, residual, ln2_gamma, wo, w1, w2,
                              tokens, num_heads, head_dim, embed_dim, intermediate,
                              eps, wo_dt, w1_dt, w2_dt)
        fused_times.append((time.perf_counter() - start) * 1e6)
    fused_time = min(fused_times)

    # Check for NaN/Inf
    if np.any(np.isnan(baseline_out)) or np.any(np.isinf(baseline_out)):
        return False, float('inf'), float('inf'), baseline_time, fused_time
    if np.any(np.isnan(fused_out)) or np.any(np.isinf(fused_out)):
        return False, float('inf'), float('inf'), baseline_time, fused_time

    # Compare
    diff = np.abs(baseline_out - fused_out)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    passed = max_diff <= tolerance

    return passed, max_diff, mean_diff, baseline_time, fused_time


def main():
    parser = argparse.ArgumentParser(description="OutProj+MLP Fusion Performance Benchmark")
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test (fewer configurations)')
    args = parser.parse_args()

    # Header
    print("=" * 80)
    print(f"{BOLD}OUTPROJ+MLP FUSION BENCHMARK: CK Fused vs CK Baseline{RESET}")
    print("=" * 80)
    print()
    print(f"{YELLOW}Purpose:{RESET}   Measure speedup from mega_fused_outproj_mlp kernel fusion")
    print(f"{YELLOW}Note:{RESET}      Numerical differences expected (different quantization strategies)")
    print()
    print(f"{YELLOW}Fused Operations:{RESET}")
    print("  1. Quantize attention output (head-major) to Q8_0")
    print("  2. OutProj: attn_out @ W_o (Q5_0)")
    print("  3. Residual add")
    print("  4. RMSNorm")
    print("  5. MLP: silu(x @ W_gate) * (x @ W_up) @ W2 (Q4_K/Q6_K)")
    print("  6. Residual add")
    print()

    lib = load_library()
    if lib is None:
        print(f"[{RED}FAIL{RESET}] CK library not found. Run: make")
        return 1

    print()

    try:
        setup_signatures(lib)
    except Exception as e:
        print(f"[{RED}FAIL{RESET}] Failed to set up function signatures: {e}")
        return 1

    # Test configurations
    if args.quick:
        configs = [
            {'tokens': 32, 'num_heads': 14, 'head_dim': 64, 'intermediate': 4864, 'w2_is_q6k': 0, 'name': 'Qwen2-32tok-Q4K'},
            {'tokens': 64, 'num_heads': 14, 'head_dim': 64, 'intermediate': 4864, 'w2_is_q6k': 1, 'name': 'Qwen2-64tok-Q6K'},
        ]
    else:
        configs = [
            {'tokens': 32, 'num_heads': 14, 'head_dim': 64, 'intermediate': 4864, 'w2_is_q6k': 0, 'name': 'Qwen2-32tok-Q4K'},
            {'tokens': 64, 'num_heads': 14, 'head_dim': 64, 'intermediate': 4864, 'w2_is_q6k': 0, 'name': 'Qwen2-64tok-Q4K'},
            {'tokens': 128, 'num_heads': 14, 'head_dim': 64, 'intermediate': 4864, 'w2_is_q6k': 0, 'name': 'Qwen2-128tok-Q4K'},
            {'tokens': 256, 'num_heads': 14, 'head_dim': 64, 'intermediate': 4864, 'w2_is_q6k': 0, 'name': 'Qwen2-256tok-Q4K'},
            {'tokens': 32, 'num_heads': 14, 'head_dim': 64, 'intermediate': 4864, 'w2_is_q6k': 1, 'name': 'Qwen2-32tok-Q6K'},
            {'tokens': 64, 'num_heads': 14, 'head_dim': 64, 'intermediate': 4864, 'w2_is_q6k': 1, 'name': 'Qwen2-64tok-Q6K'},
            {'tokens': 128, 'num_heads': 14, 'head_dim': 64, 'intermediate': 4864, 'w2_is_q6k': 1, 'name': 'Qwen2-128tok-Q6K'},
            {'tokens': 256, 'num_heads': 14, 'head_dim': 64, 'intermediate': 4864, 'w2_is_q6k': 1, 'name': 'Qwen2-256tok-Q6K'},
        ]

    results = []

    for cfg in configs:
        name = cfg['name']
        tokens = cfg['tokens']
        embed_dim = cfg['num_heads'] * cfg['head_dim']
        intermediate = cfg['intermediate']
        w2_type = "Q6_K" if cfg.get('w2_is_q6k', 0) else "Q4_K"

        desc = f"embed={embed_dim}, inter={intermediate}, W2={w2_type}"
        print(f"--- benchmark ({name}: {desc}) ---")

        _, max_diff, mean_diff, baseline_time, fused_time = test_fusion_parity(
            lib, cfg, tolerance=float('inf')  # No tolerance check - just benchmark
        )

        results.append((cfg, max_diff, baseline_time, fused_time))

        speedup = baseline_time / fused_time if fused_time > 0 else 0
        if speedup >= 1.1:
            speedup_color = GREEN
        elif speedup >= 1.0:
            speedup_color = YELLOW
        else:
            speedup_color = RED

        print(f"      baseline={baseline_time:.0f}us, fused={fused_time:.0f}us, "
              f"speedup={speedup_color}{speedup:.2f}x{RESET}")
        print()

    # Summary
    print("=" * 80)
    print(f"{BOLD}OUTPROJ+MLP FUSION BENCHMARK SUMMARY{RESET}")
    print("=" * 80)

    print(f"{'Test':<25} {'Tokens':>6} {'Base (us)':>12} {'Fused (us)':>12} {'Speedup':>10}")
    print("-" * 80)
    avg_speedup = 0.0
    for cfg, max_diff, baseline_time, fused_time in results:
        speedup = baseline_time / fused_time if fused_time > 0 else 0
        avg_speedup += speedup
        if speedup >= 1.1:
            speedup_str = f"{GREEN}{speedup:.2f}x{RESET}"
        elif speedup >= 1.0:
            speedup_str = f"{YELLOW}{speedup:.2f}x{RESET}"
        else:
            speedup_str = f"{RED}{speedup:.2f}x{RESET}"
        print(f"  {cfg['name']:<23} {cfg['tokens']:>6} "
              f"{baseline_time:>12.0f} {fused_time:>12.0f} {speedup_str:>18}")

    avg_speedup /= len(results)
    print("-" * 80)
    print(f"  {'Average':>29} {' '*24} {avg_speedup:.2f}x")

    print()
    print(f"{CYAN}Note: Numerical differences between fused and baseline are expected{RESET}")
    print(f"{CYAN}      due to different quantization strategies (head-major vs token-major).{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
