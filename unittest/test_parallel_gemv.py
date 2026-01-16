#!/usr/bin/env python3
"""
Test parallel GEMV Q4_K_Q8_K kernel.

Tests:
1. Correctness: parallel version matches single-threaded
2. Performance: benchmark with different thread counts
3. Compare against llama.cpp (if available)
"""

import ctypes
import sys
from pathlib import Path
import time
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np

from lib_loader import load_lib
from test_utils import get_cpu_info, print_system_info

# Load CK library
lib = load_lib("libckernel_engine.so")

# Q4_K block structure: 144 bytes per block, 256 elements per block
QK_K = 256
BLOCK_Q4_K_SIZE = 144

# Q8_K block structure: 4 + 256 + 32 = 292 bytes per block
# d (float32, 4 bytes) + qs (256 int8) + bsums (16 int16 = 32 bytes)
BLOCK_Q8_K_SIZE = 4 + QK_K + 32  # = 292


def ptr(arr):
    """Get ctypes pointer to numpy array."""
    return arr.ctypes.data_as(ctypes.c_void_p)


def ptr_float(arr):
    """Get ctypes pointer to float array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def create_random_q4k_weights(M, K):
    """Create random Q4_K weights."""
    assert K % QK_K == 0
    n_blocks = (M * K) // QK_K
    # Allocate raw bytes for Q4_K blocks
    weights = np.zeros(n_blocks * BLOCK_Q4_K_SIZE, dtype=np.uint8)
    # Fill with random data (not perfectly valid Q4_K, but good for testing)
    np.random.seed(42)
    weights[:] = np.random.randint(0, 256, size=len(weights), dtype=np.uint8)
    # Set d and dmin to reasonable values (first 4 bytes of each block)
    for b in range(n_blocks):
        offset = b * BLOCK_Q4_K_SIZE
        # d and dmin as FP16 - set to small positive values
        weights[offset:offset+2] = np.array([0x00, 0x3C], dtype=np.uint8)  # ~1.0 in FP16
        weights[offset+2:offset+4] = np.array([0x00, 0x38], dtype=np.uint8)  # ~0.5 in FP16
    return weights


def create_random_q8k_activations(K):
    """Create random Q8_K activations."""
    assert K % QK_K == 0
    n_blocks = K // QK_K
    # Allocate raw bytes for Q8_K blocks
    activations = np.zeros(n_blocks * BLOCK_Q8_K_SIZE, dtype=np.uint8)
    np.random.seed(123)
    # Fill qs with random int8 values
    for b in range(n_blocks):
        offset = b * BLOCK_Q8_K_SIZE
        # d as FP32 (first 4 bytes)
        d_val = np.array([1.0], dtype=np.float32)
        activations[offset:offset+4] = d_val.view(np.uint8)
        # qs: 256 int8 values (store as uint8 bytes)
        qs = np.random.randint(-128, 127, size=QK_K, dtype=np.int8)
        activations[offset+4:offset+4+QK_K] = qs.view(np.uint8)
        # bsums: 16 int16 values (32 bytes)
        # Just set to zeros for testing
    return activations


def setup_functions():
    """Set up ctypes function signatures."""
    # Single-threaded version
    lib.gemv_q4_k_q8_k.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # y
        ctypes.c_void_p,                  # W
        ctypes.c_void_p,                  # x_q8
        ctypes.c_int,                     # M
        ctypes.c_int,                     # K
    ]
    lib.gemv_q4_k_q8_k.restype = None

    # Parallel version
    lib.gemv_q4_k_q8_k_parallel.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # y
        ctypes.c_void_p,                  # W
        ctypes.c_void_p,                  # x_q8
        ctypes.c_int,                     # M
        ctypes.c_int,                     # K
        ctypes.c_int,                     # ith
        ctypes.c_int,                     # nth
    ]
    lib.gemv_q4_k_q8_k_parallel.restype = None


def test_correctness(M=896, K=1024):
    """Test that parallel version produces same results as single-threaded."""
    # Ensure K is multiple of QK_K (256)
    K = ((K + QK_K - 1) // QK_K) * QK_K

    print(f"\n{'='*70}")
    print(f"Correctness Test: M={M}, K={K}")
    print(f"{'='*70}")

    # Create test data
    weights = create_random_q4k_weights(M, K)
    activations = create_random_q8k_activations(K)

    # Output buffers
    y_ref = np.zeros(M, dtype=np.float32)
    y_parallel = np.zeros(M, dtype=np.float32)

    # Run single-threaded reference
    lib.gemv_q4_k_q8_k(
        ptr_float(y_ref),
        ptr(weights),
        ptr(activations),
        ctypes.c_int(M),
        ctypes.c_int(K)
    )

    # Test with different thread counts
    for nth in [1, 2, 4, 8]:
        y_parallel.fill(0)

        # Simulate parallel execution (call kernel for each thread)
        for ith in range(nth):
            lib.gemv_q4_k_q8_k_parallel(
                ptr_float(y_parallel),
                ptr(weights),
                ptr(activations),
                ctypes.c_int(M),
                ctypes.c_int(K),
                ctypes.c_int(ith),
                ctypes.c_int(nth)
            )

        # Compare
        max_diff = np.max(np.abs(y_ref - y_parallel))
        passed = max_diff < 1e-5

        print(f"  nth={nth}: max_diff={max_diff:.2e} {'PASS' if passed else 'FAIL'}")

        if not passed:
            # Show first few differences
            diffs = np.abs(y_ref - y_parallel)
            bad_idx = np.where(diffs > 1e-5)[0][:5]
            for idx in bad_idx:
                print(f"    y_ref[{idx}]={y_ref[idx]:.6f}, y_parallel[{idx}]={y_parallel[idx]:.6f}")
            return False

    print("  All thread counts PASSED!")
    return True


def benchmark_parallel(M=896, K=1024, iterations=100):
    """Benchmark parallel vs single-threaded."""
    # Ensure K is multiple of QK_K (256)
    K = ((K + QK_K - 1) // QK_K) * QK_K

    print(f"\n{'='*70}")
    print(f"Performance Benchmark: M={M}, K={K}")
    print(f"{'='*70}")

    # Create test data
    weights = create_random_q4k_weights(M, K)
    activations = create_random_q8k_activations(K)
    y = np.zeros(M, dtype=np.float32)

    # Warmup
    for _ in range(10):
        lib.gemv_q4_k_q8_k(ptr_float(y), ptr(weights), ptr(activations),
                          ctypes.c_int(M), ctypes.c_int(K))

    # Benchmark single-threaded
    start = time.perf_counter()
    for _ in range(iterations):
        lib.gemv_q4_k_q8_k(ptr_float(y), ptr(weights), ptr(activations),
                          ctypes.c_int(M), ctypes.c_int(K))
    t_single = (time.perf_counter() - start) / iterations * 1e6  # us

    print(f"  Single-threaded: {t_single:.1f} us")

    # Benchmark parallel with different thread counts
    # Note: This simulates what orchestration would do with OpenMP
    for nth in [2, 4, 8]:
        y.fill(0)

        # Warmup
        for _ in range(10):
            for ith in range(nth):
                lib.gemv_q4_k_q8_k_parallel(
                    ptr_float(y), ptr(weights), ptr(activations),
                    ctypes.c_int(M), ctypes.c_int(K),
                    ctypes.c_int(ith), ctypes.c_int(nth))

        # Benchmark (simulating sequential calls - NOT true parallel)
        start = time.perf_counter()
        for _ in range(iterations):
            for ith in range(nth):
                lib.gemv_q4_k_q8_k_parallel(
                    ptr_float(y), ptr(weights), ptr(activations),
                    ctypes.c_int(M), ctypes.c_int(K),
                    ctypes.c_int(ith), ctypes.c_int(nth))
        t_parallel = (time.perf_counter() - start) / iterations * 1e6  # us

        # Note: This is sequential simulation, real parallel would be faster
        print(f"  Parallel nth={nth} (sequential sim): {t_parallel:.1f} us")

    print("\n  NOTE: True parallel performance requires OpenMP at orchestration level")
    print("        The 'parallel' times above are sequential simulation")


def test_qwen_dimensions():
    """Test with Qwen2-0.5B typical dimensions."""
    print(f"\n{'='*70}")
    print("Testing Qwen2-0.5B dimensions")
    print(f"{'='*70}")

    configs = [
        # (M, K, name)
        (896, 896, "Q projection (q_dim x embed_dim)"),
        (128, 896, "K projection (kv_dim x embed_dim)"),
        (128, 896, "V projection (kv_dim x embed_dim)"),
        (896, 896, "O projection (embed_dim x q_dim)"),
        (4864, 896, "Gate projection (inter_dim x embed_dim)"),
        (4864, 896, "Up projection (inter_dim x embed_dim)"),
        (896, 4864, "Down projection (embed_dim x inter_dim)"),
    ]

    all_passed = True
    for M, K, name in configs:
        # Ensure K is multiple of QK_K
        K_aligned = ((K + QK_K - 1) // QK_K) * QK_K

        print(f"\n  {name}: M={M}, K={K_aligned}")

        weights = create_random_q4k_weights(M, K_aligned)
        activations = create_random_q8k_activations(K_aligned)

        y_ref = np.zeros(M, dtype=np.float32)
        y_parallel = np.zeros(M, dtype=np.float32)

        # Reference
        lib.gemv_q4_k_q8_k(
            ptr_float(y_ref), ptr(weights), ptr(activations),
            ctypes.c_int(M), ctypes.c_int(K_aligned))

        # Parallel with 4 threads
        for ith in range(4):
            lib.gemv_q4_k_q8_k_parallel(
                ptr_float(y_parallel), ptr(weights), ptr(activations),
                ctypes.c_int(M), ctypes.c_int(K_aligned),
                ctypes.c_int(ith), ctypes.c_int(4))

        max_diff = np.max(np.abs(y_ref - y_parallel))
        passed = max_diff < 1e-5
        print(f"    max_diff={max_diff:.2e} {'PASS' if passed else 'FAIL'}")

        if not passed:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print_system_info()
    setup_functions()

    # Test correctness
    correct = test_correctness()

    # Test Qwen dimensions
    qwen_ok = test_qwen_dimensions()

    # Benchmark
    benchmark_parallel()

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  Correctness: {'PASS' if correct else 'FAIL'}")
    print(f"  Qwen dims:   {'PASS' if qwen_ok else 'FAIL'}")

    if not (correct and qwen_ok):
        sys.exit(1)
