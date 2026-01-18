#!/usr/bin/env python3
"""
Attention Kernel Parity Test
============================

Tests CK-Engine's attention kernel against llama.cpp reference implementation.
Format matches `make llamacpp-parity-full` for consistency.

WHAT IT DOES:
    - Compares CK's ck_test_attention_causal against llama.cpp's test_attention_flash
    - Tests numerical parity with configurable tolerance (default 1e-3)
    - Reports max/mean differences and pass/fail status

WHEN TO RUN:
    - After modifying attention kernels (flash_attention_*.c)
    - After changing quantization or precision handling
    - As part of CI to ensure llama.cpp compatibility

TRIGGERED BY:
    - scripts/run_mega_fusion_test.sh (steps 4/6)
    - make fusion-test-full-with-lamacpp
    - make fusion-test-quick

DEPENDENCIES:
    - build/libck_parity.so (CK parity library)
    - llama.cpp/libggml_kernel_test.so (llama.cpp test library)

STATUS: ACTIVE - Core parity test, do not remove

Usage:
    python test_mega_fusion_parity.py
    python test_mega_fusion_parity.py --quick
    python test_mega_fusion_parity.py --tol 1e-4
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


def load_libraries():
    """Load both CK and llama.cpp test libraries."""
    # Go up from unittest/fusion/ to project root
    base_dir = Path(__file__).parent.parent.parent

    ck_paths = [
        base_dir / "build" / "libck_parity.so",
        base_dir / "libck_parity.so",
    ]
    ck_lib = None
    for p in ck_paths:
        if p.exists():
            try:
                ck_lib = ctypes.CDLL(str(p))
                print(f"Loaded CK library: {p}")
                break
            except OSError:
                pass

    llama_paths = [
        base_dir / "llama.cpp" / "libggml_kernel_test.so",
        base_dir / "llama.cpp" / "build" / "libggml_kernel_test.so",
    ]
    llama_lib = None
    for p in llama_paths:
        if p.exists():
            try:
                llama_lib = ctypes.CDLL(str(p))
                print(f"Loaded llama.cpp library: {p}")
                break
            except OSError:
                pass

    return ck_lib, llama_lib


def setup_ck_signatures(lib):
    """Set up ctypes signatures for CK functions."""
    lib.ck_test_attention_causal.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.ck_test_attention_causal.restype = None


def setup_llama_signatures(lib):
    """Set up ctypes signatures for llama.cpp functions."""
    lib.test_init.argtypes = []
    lib.test_init.restype = None
    lib.test_init()

    lib.test_attention_causal_multihead.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.test_attention_causal_multihead.restype = None


def test_attention_kernel(ck_lib, llama_lib, config, tolerance=1e-3, n_runs=3):
    """
    Test attention kernel parity and performance.
    Returns: (passed, max_diff, mean_diff, ck_time_us, llama_time_us)
    """
    tokens = config['tokens']
    num_heads = config['num_heads']
    num_kv_heads = config['num_kv_heads']
    head_dim = config['head_dim']
    seq_len = tokens

    # Generate random Q, K, V (head-major layout)
    np.random.seed(42)
    q = np.random.randn(num_heads, tokens, head_dim).astype(np.float32) * 0.1
    k = np.random.randn(num_kv_heads, seq_len, head_dim).astype(np.float32) * 0.1
    v = np.random.randn(num_kv_heads, seq_len, head_dim).astype(np.float32) * 0.1

    ck_out = np.zeros((num_heads, tokens, head_dim), dtype=np.float32)
    llama_out = np.zeros((num_heads, tokens, head_dim), dtype=np.float32)

    q_ptr = q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    k_ptr = k.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    v_ptr = v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ck_out_ptr = ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    llama_out_ptr = llama_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Warmup
    ck_lib.ck_test_attention_causal(q_ptr, k_ptr, v_ptr, ck_out_ptr,
                                     num_heads, num_kv_heads, tokens, seq_len, head_dim)
    llama_lib.test_attention_causal_multihead(q_ptr, k_ptr, v_ptr, llama_out_ptr,
                                               num_heads, num_kv_heads, tokens, seq_len, head_dim)

    # Benchmark CK (multiple runs, take min)
    ck_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        ck_lib.ck_test_attention_causal(q_ptr, k_ptr, v_ptr, ck_out_ptr,
                                         num_heads, num_kv_heads, tokens, seq_len, head_dim)
        ck_times.append((time.perf_counter() - start) * 1e6)
    ck_time = min(ck_times)

    # Benchmark llama.cpp (multiple runs, take min)
    llama_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        llama_lib.test_attention_causal_multihead(q_ptr, k_ptr, v_ptr, llama_out_ptr,
                                                   num_heads, num_kv_heads, tokens, seq_len, head_dim)
        llama_times.append((time.perf_counter() - start) * 1e6)
    llama_time = min(llama_times)

    # Check for NaN/Inf
    if np.any(np.isnan(ck_out)) or np.any(np.isinf(ck_out)):
        return False, float('inf'), float('inf'), ck_time, llama_time
    if np.any(np.isnan(llama_out)) or np.any(np.isinf(llama_out)):
        return False, float('inf'), float('inf'), ck_time, llama_time

    # Compare
    diff = np.abs(ck_out - llama_out)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    passed = max_diff <= tolerance

    return passed, max_diff, mean_diff, ck_time, llama_time


def main():
    parser = argparse.ArgumentParser(description="Attention Parity Test (CK vs llama.cpp)")
    parser.add_argument('--tol', '--tolerance', type=float, default=1e-3,
                        help='Tolerance for numerical parity (default: 1e-3)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed statistics')
    parser.add_argument('--perf', action='store_true',
                        help='Always enabled - kept for compatibility')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test (fewer configurations)')
    args = parser.parse_args()

    # Header
    print("=" * 80)
    print(f"{BOLD}ATTENTION KERNEL PARITY TESTS: C-Kernel-Engine vs llama.cpp{RESET}")
    print("=" * 80)
    print()
    print(f"{YELLOW}Purpose:{RESET}   Verify CK attention kernel produces identical outputs to reference")
    print(f"{YELLOW}Method:{RESET}    Run both implementations with same Q/K/V inputs, compare outputs")
    print(f"{YELLOW}Tolerance:{RESET} max_diff < {args.tol:.0e} for all elements")
    print()
    print(f"{YELLOW}Kernels Tested:{RESET}")
    print("  - Causal Multi-Head Attention (MHA): num_heads == num_kv_heads")
    print("  - Causal Grouped-Query Attention (GQA): num_heads > num_kv_heads")
    print()

    # Load libraries
    ck_lib, llama_lib = load_libraries()

    if ck_lib is None:
        print(f"[{RED}FAIL{RESET}] CK library not found. Run: make libck_parity.so")
        return 1

    if llama_lib is None:
        print(f"[{RED}FAIL{RESET}] llama.cpp library not found. Run: make llamacpp-parity-full")
        return 1

    print()

    # Set up function signatures
    try:
        setup_ck_signatures(ck_lib)
        setup_llama_signatures(llama_lib)
    except Exception as e:
        print(f"[{RED}FAIL{RESET}] Failed to set up function signatures: {e}")
        return 1

    # Test configurations
    if args.quick:
        configs = [
            {'tokens': 32, 'num_heads': 14, 'num_kv_heads': 2, 'head_dim': 64, 'name': 'GQA-Qwen2-32'},
            {'tokens': 128, 'num_heads': 32, 'num_kv_heads': 8, 'head_dim': 128, 'name': 'GQA-Llama7B-128'},
        ]
    else:
        configs = [
            # MHA configurations
            {'tokens': 16, 'num_heads': 8, 'num_kv_heads': 8, 'head_dim': 64, 'name': 'MHA-16tok'},
            {'tokens': 64, 'num_heads': 12, 'num_kv_heads': 12, 'head_dim': 64, 'name': 'MHA-64tok'},
            # GQA Qwen2-0.5B style (14 heads, 2 kv_heads, head_dim=64)
            {'tokens': 32, 'num_heads': 14, 'num_kv_heads': 2, 'head_dim': 64, 'name': 'GQA-Qwen2-32tok'},
            {'tokens': 128, 'num_heads': 14, 'num_kv_heads': 2, 'head_dim': 64, 'name': 'GQA-Qwen2-128tok'},
            {'tokens': 512, 'num_heads': 14, 'num_kv_heads': 2, 'head_dim': 64, 'name': 'GQA-Qwen2-512tok'},
            {'tokens': 1024, 'num_heads': 14, 'num_kv_heads': 2, 'head_dim': 64, 'name': 'GQA-Qwen2-1024tok'},
            # GQA Llama-7B style (32 heads, 8 kv_heads, head_dim=128)
            {'tokens': 32, 'num_heads': 32, 'num_kv_heads': 8, 'head_dim': 128, 'name': 'GQA-Llama7B-32tok'},
            {'tokens': 128, 'num_heads': 32, 'num_kv_heads': 8, 'head_dim': 128, 'name': 'GQA-Llama7B-128tok'},
            {'tokens': 512, 'num_heads': 32, 'num_kv_heads': 8, 'head_dim': 128, 'name': 'GQA-Llama7B-512tok'},
            {'tokens': 1024, 'num_heads': 32, 'num_kv_heads': 8, 'head_dim': 128, 'name': 'GQA-Llama7B-1024tok'},
        ]

    # Run tests
    results = []
    passed_count = 0

    for cfg in configs:
        name = cfg['name']
        tokens = cfg['tokens']
        num_heads = cfg['num_heads']
        num_kv_heads = cfg['num_kv_heads']
        head_dim = cfg['head_dim']

        desc = f"heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}"
        print(f"--- test_attention ({name}: {desc}) ---")

        passed, max_diff, mean_diff, ck_time, llama_time = test_attention_kernel(
            ck_lib, llama_lib, cfg, args.tol
        )

        results.append((cfg, passed, max_diff, mean_diff, ck_time, llama_time))

        if passed:
            passed_count += 1
            status = f"[{GREEN}PASS{RESET}]"
        else:
            status = f"[{RED}FAIL{RESET}]"

        speedup = llama_time / ck_time if ck_time > 0 else 0
        if speedup >= 1.2:
            speedup_color = GREEN
        elif speedup >= 1.0:
            speedup_color = YELLOW
        else:
            speedup_color = RED

        print(f"{status} {name}: max_diff={max_diff:.2e}, mean={mean_diff:.2e} "
              f"(tol={args.tol:.0e})")
        print(f"      Performance: CK={ck_time:.0f}us, ref={llama_time:.0f}us, "
              f"speedup={speedup_color}{speedup:.2f}x{RESET}")
        print()

    # Summary
    print("=" * 80)
    print(f"{BOLD}ATTENTION KERNEL TEST SUMMARY{RESET}")
    print("=" * 80)
    print(f"Passed: {passed_count}/{len(configs)}")
    print()

    # Results table
    print(f"{'Test':<25} {'Tokens':>6} {'Max Diff':>12} {'CK (us)':>10} {'Ref (us)':>10} {'Speedup':>10}")
    print("-" * 80)
    for cfg, passed, max_diff, mean_diff, ck_time, llama_time in results:
        speedup = llama_time / ck_time if ck_time > 0 else 0
        status = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
        if speedup >= 1.2:
            speedup_str = f"{GREEN}{speedup:.2f}x{RESET}"
        elif speedup >= 1.0:
            speedup_str = f"{YELLOW}{speedup:.2f}x{RESET}"
        else:
            speedup_str = f"{RED}{speedup:.2f}x{RESET}"
        print(f"{status} {cfg['name']:<23} {cfg['tokens']:>6} {max_diff:>12.2e} "
              f"{ck_time:>10.0f} {llama_time:>10.0f} {speedup_str:>18}")

    print()
    if passed_count == len(configs):
        print(f"{GREEN}All attention kernels match llama.cpp reference!{RESET}")
        return 0
    else:
        print(f"{RED}Some attention kernel tests FAILED!{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
