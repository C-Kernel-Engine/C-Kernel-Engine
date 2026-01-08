#!/usr/bin/env python3
"""
Unit test for GEMM microkernel with 8x8 register blocking.

Tests:
1. Accuracy vs PyTorch matmul (B not transposed)
2. Accuracy vs PyTorch matmul (B transposed)
3. Performance comparison vs naive GEMM (skipped in --quick mode)
4. Performance comparison vs PyTorch (skipped in --quick mode)

Usage:
  python test_gemm_microkernel.py           # Full test with benchmarks
  python test_gemm_microkernel.py --quick   # Fast accuracy tests only
"""

import argparse
import ctypes
import numpy as np
import time
import os
import sys

# Parse arguments early
parser = argparse.ArgumentParser(description='GEMM microkernel unit test')
parser.add_argument('--quick', action='store_true',
                    help='Quick mode: accuracy tests only, skip benchmarks')
args = parser.parse_args()

# Try to import torch for reference
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found, using NumPy for reference")

def load_library():
    """Load the C-Kernel-Engine library."""
    lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'libckernel_engine.so')
    if not os.path.exists(lib_path):
        print(f"Library not found at {lib_path}")
        print("Run 'make' first to build the library")
        sys.exit(1)
    return ctypes.CDLL(lib_path)

def setup_functions(lib):
    """Set up function signatures for GEMM kernels."""
    # gemm_microkernel(A, B, C, M, N, K, B_transposed)
    lib.gemm_microkernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.POINTER(ctypes.c_float),  # C
        ctypes.c_int,                     # M
        ctypes.c_int,                     # N
        ctypes.c_int,                     # K
        ctypes.c_int,                     # B_transposed
    ]
    lib.gemm_microkernel.restype = None

    # gemm_naive_parallel(A, B, bias, C, M, N, K)
    lib.gemm_naive_parallel.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.POINTER(ctypes.c_float),  # bias
        ctypes.POINTER(ctypes.c_float),  # C
        ctypes.c_int,                     # M
        ctypes.c_int,                     # N
        ctypes.c_int,                     # K
    ]
    lib.gemm_naive_parallel.restype = None

    return lib

def test_accuracy_b_not_transposed(lib, M, N, K, tol=1e-4):
    """Test microkernel accuracy when B is not transposed."""
    np.random.seed(42)

    # Create random matrices
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)  # B is [K, N]
    C_micro = np.zeros((M, N), dtype=np.float32)

    # Get C pointers
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C_ptr = C_micro.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call microkernel (B_transposed=0)
    lib.gemm_microkernel(A_ptr, B_ptr, C_ptr, M, N, K, 0)

    # Reference computation
    if HAS_TORCH:
        A_t = torch.from_numpy(A)
        B_t = torch.from_numpy(B)
        C_ref = (A_t @ B_t).numpy()
    else:
        C_ref = A @ B

    # Check accuracy
    max_diff = np.max(np.abs(C_micro - C_ref))
    mean_diff = np.mean(np.abs(C_micro - C_ref))

    # Relative error for better comparison
    rel_error = max_diff / (np.max(np.abs(C_ref)) + 1e-10)

    passed = max_diff < tol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] B not transposed ({M}x{K} @ {K}x{N}): max_diff={max_diff:.2e}, rel_err={rel_error:.2e}")

    return passed, max_diff, rel_error

def test_accuracy_b_transposed(lib, M, N, K, tol=1e-4):
    """Test microkernel accuracy when B is transposed (common in NN)."""
    np.random.seed(42)

    # Create random matrices
    # A is [M, K], B is stored as [N, K] (transposed)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(N, K).astype(np.float32)  # B is [N, K] (transposed layout)
    C_micro = np.zeros((M, N), dtype=np.float32)

    # Get C pointers
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C_ptr = C_micro.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call microkernel (B_transposed=1)
    lib.gemm_microkernel(A_ptr, B_ptr, C_ptr, M, N, K, 1)

    # Reference computation: C = A @ B.T
    if HAS_TORCH:
        A_t = torch.from_numpy(A)
        B_t = torch.from_numpy(B)
        C_ref = (A_t @ B_t.T).numpy()
    else:
        C_ref = A @ B.T

    # Check accuracy
    max_diff = np.max(np.abs(C_micro - C_ref))
    mean_diff = np.mean(np.abs(C_micro - C_ref))
    rel_error = max_diff / (np.max(np.abs(C_ref)) + 1e-10)

    passed = max_diff < tol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] B transposed ({M}x{K} @ {N}x{K}.T): max_diff={max_diff:.2e}, rel_err={rel_error:.2e}")

    return passed, max_diff, rel_error

def benchmark_microkernel_vs_naive(lib, M, N, K, iters=10):
    """Benchmark microkernel vs naive GEMM."""
    np.random.seed(42)

    # Create random matrices
    A = np.random.randn(M, K).astype(np.float32)
    B_micro = np.random.randn(K, N).astype(np.float32)  # For microkernel (B not transposed)
    C = np.zeros((M, N), dtype=np.float32)

    # For naive GEMM, B should be [N, K] (transposed)
    B_naive = B_micro.T.copy()

    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_naive_ptr = B_naive.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_micro_ptr = B_micro.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Warmup
    lib.gemm_naive_parallel(A_ptr, B_naive_ptr, None, C_ptr, M, N, K)
    lib.gemm_microkernel(A_ptr, B_micro_ptr, C_ptr, M, N, K, 0)

    # Benchmark naive
    start = time.perf_counter()
    for _ in range(iters):
        lib.gemm_naive_parallel(A_ptr, B_naive_ptr, None, C_ptr, M, N, K)
    naive_time = (time.perf_counter() - start) / iters

    # Benchmark microkernel
    start = time.perf_counter()
    for _ in range(iters):
        lib.gemm_microkernel(A_ptr, B_micro_ptr, C_ptr, M, N, K, 0)
    micro_time = (time.perf_counter() - start) / iters

    # Calculate GFLOPS
    flops = 2.0 * M * N * K
    naive_gflops = flops / naive_time / 1e9
    micro_gflops = flops / micro_time / 1e9
    speedup = naive_time / micro_time

    return naive_time, micro_time, naive_gflops, micro_gflops, speedup

def benchmark_vs_pytorch(lib, M, N, K, iters=20):
    """Benchmark microkernel vs PyTorch matmul (MKL/OpenBLAS backend).

    NOTE: We compare MULTI-THREADED performance for both.
    torch.set_num_threads(1) also affects OpenMP globally, which would
    unfairly throttle our kernel. So we let both use all available threads.
    """
    if not HAS_TORCH:
        return None, None, None, None, None

    np.random.seed(42)

    # Create matrices
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)
    C_np = np.zeros((M, N), dtype=np.float32)

    # PyTorch tensors
    A_t = torch.from_numpy(A_np.copy())
    B_t = torch.from_numpy(B_np.copy())

    # C kernel pointers
    A_ptr = A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C_ptr = C_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Warmup (both use all threads)
    _ = A_t @ B_t
    lib.gemm_microkernel(A_ptr, B_ptr, C_ptr, M, N, K, 0)

    # Benchmark PyTorch (multi-threaded)
    start = time.perf_counter()
    for _ in range(iters):
        C_t = A_t @ B_t
    torch_time = (time.perf_counter() - start) / iters

    # Benchmark microkernel
    start = time.perf_counter()
    for _ in range(iters):
        lib.gemm_microkernel(A_ptr, B_ptr, C_ptr, M, N, K, 0)
    micro_time = (time.perf_counter() - start) / iters

    # Calculate GFLOPS
    flops = 2.0 * M * N * K
    torch_gflops = flops / torch_time / 1e9
    micro_gflops = flops / micro_time / 1e9
    ratio = micro_time / torch_time  # <1 means microkernel is faster

    return torch_time, micro_time, torch_gflops, micro_gflops, ratio

def main():
    print("=" * 80)
    print("GEMM Microkernel Unit Test (8x8 register blocking)")
    print("=" * 80)

    if HAS_TORCH:
        print(f"PyTorch version: {torch.__version__}")
        # Try to get BLAS info
        try:
            blas_info = torch.__config__.show()
            if "MKL" in blas_info:
                print("PyTorch BLAS backend: MKL")
            elif "OpenBLAS" in blas_info:
                print("PyTorch BLAS backend: OpenBLAS")
            else:
                print("PyTorch BLAS backend: Unknown")
        except:
            pass

    lib = load_library()
    lib = setup_functions(lib)

    all_passed = True
    accuracy_results = []

    # ==========================================================================
    # Test 1: Accuracy (B not transposed)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("1. ACCURACY TEST vs PyTorch (B not transposed: C = A @ B)")
    print("=" * 80)

    test_sizes = [
        (8, 8, 8),       # Exactly one microkernel
        (16, 16, 16),    # 2x2 microkernels
        (32, 32, 32),    # 4x4 microkernels
        (64, 64, 64),    # Full cache block
        (128, 256, 128), # Larger, tests cache blocking
        (256, 256, 256), # Medium size
        (512, 512, 512), # Large size
        (13, 17, 11),    # Non-aligned (edge cases)
        (7, 9, 5),       # Smaller than microkernel
    ]

    for M, N, K in test_sizes:
        passed, max_diff, rel_err = test_accuracy_b_not_transposed(lib, M, N, K)
        all_passed = all_passed and passed
        accuracy_results.append(("B_normal", M, N, K, max_diff, rel_err, passed))

    # ==========================================================================
    # Test 2: Accuracy (B transposed)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("2. ACCURACY TEST vs PyTorch (B transposed: C = A @ B.T)")
    print("=" * 80)

    for M, N, K in test_sizes:
        passed, max_diff, rel_err = test_accuracy_b_transposed(lib, M, N, K, tol=5e-4)
        all_passed = all_passed and passed
        accuracy_results.append(("B_transposed", M, N, K, max_diff, rel_err, passed))

    # ==========================================================================
    # Test 3: Performance vs Naive GEMM (skip in --quick mode)
    # ==========================================================================
    avg_naive_speedup = 0.0
    avg_ratio = 0.0
    pytorch_ratios = []

    if args.quick:
        print("\n[SKIPPED] Benchmark tests (--quick mode)")
    else:
        print("\n" + "=" * 80)
        print("3. PERFORMANCE: Microkernel vs Naive GEMM")
        print("=" * 80)
        print(f"{'Size':>15} | {'Naive':>12} | {'Micro':>12} | {'Speedup':>8}")
        print("-" * 55)

        bench_sizes = [
            (32, 32, 32),
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
        ]

        naive_speedups = []
        for M, N, K in bench_sizes:
            naive_t, micro_t, naive_gf, micro_gf, speedup = benchmark_microkernel_vs_naive(lib, M, N, K)
            naive_speedups.append(speedup)
            print(f"{M}x{N}x{K:>4} | {naive_gf:>8.2f} GF/s | {micro_gf:>8.2f} GF/s | {speedup:>6.2f}x")

        avg_naive_speedup = sum(naive_speedups) / len(naive_speedups)
        print("-" * 55)
        print(f"{'Average speedup vs naive:':<40} {avg_naive_speedup:.2f}x")

        # ==========================================================================
        # Test 4: Performance vs PyTorch
        # ==========================================================================
        if HAS_TORCH:
            print("\n" + "=" * 80)
            print("4. PERFORMANCE: Microkernel vs PyTorch (multi-threaded)")
            print("=" * 80)
            print(f"{'Size':>15} | {'PyTorch':>12} | {'Micro':>12} | {'Ratio':>10}")
            print("-" * 60)

            for M, N, K in bench_sizes:
                torch_t, micro_t, torch_gf, micro_gf, ratio = benchmark_vs_pytorch(lib, M, N, K)
                if ratio is not None:
                    pytorch_ratios.append(ratio)
                    status = "FASTER" if ratio < 1.0 else "slower"
                    print(f"{M}x{N}x{K:>4} | {torch_gf:>8.2f} GF/s | {micro_gf:>8.2f} GF/s | {ratio:>5.2f}x ({status})")

            if pytorch_ratios:
                avg_ratio = sum(pytorch_ratios) / len(pytorch_ratios)
                print("-" * 60)
                if avg_ratio < 1.0:
                    print(f"{'Average:':<40} {avg_ratio:.2f}x (microkernel is {1/avg_ratio:.2f}x FASTER)")
                else:
                    print(f"{'Average:':<40} {avg_ratio:.2f}x (PyTorch is {avg_ratio:.2f}x faster)")
        else:
            print("\n[SKIPPED] PyTorch comparison - torch not installed")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Accuracy summary
    max_errors = [r[4] for r in accuracy_results]
    print(f"  Accuracy:     max_diff range = [{min(max_errors):.2e}, {max(max_errors):.2e}]")

    # Performance summary (only if benchmarks were run)
    if not args.quick:
        print(f"  vs Naive:     {avg_naive_speedup:.2f}x average speedup")
        if HAS_TORCH and pytorch_ratios:
            if avg_ratio < 1.0:
                print(f"  vs PyTorch:   {1/avg_ratio:.2f}x FASTER than PyTorch/MKL")
            else:
                print(f"  vs PyTorch:   {avg_ratio:.2f}x slower than PyTorch/MKL")
                print(f"                (PyTorch uses highly optimized MKL/OpenBLAS)")
    else:
        print("  Benchmarks:   [SKIPPED] (--quick mode)")

    print()
    if all_passed:
        print("  STATUS: All accuracy tests PASSED!")
    else:
        print("  STATUS: Some accuracy tests FAILED!")
        sys.exit(1)
    print("=" * 80)

if __name__ == "__main__":
    main()
