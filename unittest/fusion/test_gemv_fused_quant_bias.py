#!/usr/bin/env python3
"""
Fused GEMV Kernel Parity Test
=============================

Tests CK-Engine's fused GEMV kernels (gemv_fused_q5_0_bias, gemv_fused_q8_0_bias)
against:
  1. CK unfused sequence (quantize_row_q8_0 + gemv_q5/8_0_q8_0 + add_inplace_f32)
  2. llama.cpp reference (test_gemv_q5_0 / test_gemv_q8_0 from libggml_kernel_test.so)

WHAT IT DOES:
    - Compares fused kernel output against CK unfused sequence
    - Compares fused kernel output directly against llama.cpp reference
    - Tests numerical parity with configurable tolerance (default 1e-4)
    - Benchmarks both versions for speedup measurement
    - Reports max/mean differences and pass/fail status

WHEN TO RUN:
    - After modifying fused GEMV kernels (gemv_fused_quant_bias.c)
    - After changing quantization precision
    - As part of CI to ensure fusion correctness

TRIGGERED BY:
    - make fusion-test-gemv
    - make test-fusion-all

DEPENDENCIES:
    - build/libckernel_engine.so (CK engine library)
    - llama.cpp/libggml_kernel_test.so (llama.cpp reference, optional)

STATUS: ACTIVE - Core fusion parity test

Usage:
    python test_gemv_fused_quant_bias.py
    python test_gemv_fused_quant_bias.py --quick
    python test_gemv_fused_quant_bias.py --tol 1e-5
"""

import ctypes
import numpy as np
import argparse
import sys
import time
import struct
from pathlib import Path

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

# Quantization constants
QK5_0 = 32  # Block size for Q5_0
QK8_0 = 32  # Block size for Q8_0
BLOCK_Q5_0_SIZE = 22  # bytes: 2 (FP16 d) + 4 (qh) + 16 (qs)
BLOCK_Q8_0_SIZE = 34  # bytes: 2 (FP16 d) + 32 (qs)


def float_to_fp16(f):
    """Convert float32 to IEEE FP16 (uint16)."""
    packed = struct.pack('e', f)
    return struct.unpack('H', packed)[0]


def fp16_to_float(h):
    """Convert IEEE FP16 (uint16) to float32."""
    packed = struct.pack('H', h)
    return struct.unpack('e', packed)[0]


def create_q5_0_weights(M, K, seed=42):
    """
    Create random Q5_0 quantized weights.

    Q5_0 block (22 bytes per 32 weights):
    - d: FP16 scale
    - qh[4]: high bits (32 bits packed)
    - qs[16]: low 4-bit nibbles (2 per byte)

    Returns raw bytes buffer.
    """
    np.random.seed(seed)
    blocks_per_row = K // QK5_0
    num_blocks = M * blocks_per_row

    # Allocate buffer
    buffer = bytearray(num_blocks * BLOCK_Q5_0_SIZE)

    for block_idx in range(num_blocks):
        offset = block_idx * BLOCK_Q5_0_SIZE

        # Random scale (positive, reasonable magnitude)
        scale = np.random.uniform(0.001, 0.1)
        d_fp16 = float_to_fp16(scale)
        struct.pack_into('H', buffer, offset, d_fp16)

        # Random high bits (4 bytes = 32 bits)
        qh = np.random.randint(0, 256, 4, dtype=np.uint8)
        buffer[offset + 2:offset + 6] = qh.tobytes()

        # Random low nibbles (16 bytes = 32 x 4-bit values)
        qs = np.random.randint(0, 256, 16, dtype=np.uint8)
        buffer[offset + 6:offset + 22] = qs.tobytes()

    return bytes(buffer)


def create_q8_0_weights(M, K, seed=42):
    """
    Create random Q8_0 quantized weights.

    Q8_0 block (34 bytes per 32 weights):
    - d: FP16 scale
    - qs[32]: int8 quantized values

    Returns raw bytes buffer.
    """
    np.random.seed(seed)
    blocks_per_row = K // QK8_0
    num_blocks = M * blocks_per_row

    # Allocate buffer
    buffer = bytearray(num_blocks * BLOCK_Q8_0_SIZE)

    for block_idx in range(num_blocks):
        offset = block_idx * BLOCK_Q8_0_SIZE

        # Random scale (positive, reasonable magnitude)
        scale = np.random.uniform(0.001, 0.1)
        d_fp16 = float_to_fp16(scale)
        struct.pack_into('H', buffer, offset, d_fp16)

        # Random int8 quantized values
        qs = np.random.randint(-127, 128, 32, dtype=np.int8)
        buffer[offset + 2:offset + 34] = qs.tobytes()

    return bytes(buffer)


def load_library():
    """Load CK library."""
    base_dir = Path(__file__).parent.parent.parent

    lib_paths = [
        base_dir / "build" / "libckernel_engine.so",
        base_dir / "libckernel_engine.so",
    ]

    for p in lib_paths:
        if p.exists():
            try:
                lib = ctypes.CDLL(str(p))
                print(f"Loaded CK library: {p}")
                return lib
            except OSError as e:
                print(f"Failed to load {p}: {e}")

    return None


def load_llamacpp_library():
    """Load llama.cpp kernel test library (optional)."""
    base_dir = Path(__file__).parent.parent.parent

    lib_paths = [
        base_dir / "llama.cpp" / "libggml_kernel_test.so",
        base_dir / "llama.cpp" / "build" / "libggml_kernel_test.so",
        base_dir / "llama.cpp" / "build" / "bin" / "libggml_kernel_test.so",
    ]

    for p in lib_paths:
        if p.exists():
            try:
                lib = ctypes.CDLL(str(p))
                lib.test_init.argtypes = []
                lib.test_init.restype = None
                lib.test_init()
                print(f"Loaded llama.cpp library: {p}")
                return lib
            except OSError as e:
                print(f"Failed to load {p}: {e}")

    return None


def setup_llamacpp_signatures(lib):
    """Set up ctypes signatures for llama.cpp test functions."""
    # test_gemv_q5_0(const void *W, const float *x, float *output, int cols)
    # Single-row: quantizes x to Q8_0, then vec_dot_q5_0_q8_0
    lib.test_gemv_q5_0.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.test_gemv_q5_0.restype = None

    # test_gemv_q8_0(const void *W, const float *x, float *output, int cols)
    # Single-row: quantizes x to Q8_0, then vec_dot_q8_0_q8_0
    lib.test_gemv_q8_0.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.test_gemv_q8_0.restype = None


def setup_signatures(lib):
    """Set up ctypes signatures for kernel functions."""
    # quantize_row_q8_0(const float *x, void *vy, int k)
    lib.quantize_row_q8_0.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.quantize_row_q8_0.restype = None

    # gemv_q5_0_q8_0(float *y, const void *W, const void *x_q8, int M, int K)
    lib.gemv_q5_0_q8_0.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemv_q5_0_q8_0.restype = None

    # gemv_q8_0_q8_0(float *y, const void *W, const void *x_q8, int M, int K)
    lib.gemv_q8_0_q8_0.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemv_q8_0_q8_0.restype = None

    # add_inplace_f32(float *a, const float *b, size_t n)
    lib.add_inplace_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.add_inplace_f32.restype = None

    # gemv_fused_q5_0_bias(float *y, const void *W, const float *x, const float *bias, int M, int K)
    lib.gemv_fused_q5_0_bias.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemv_fused_q5_0_bias.restype = None

    # gemv_fused_q8_0_bias(float *y, const void *W, const float *x, const float *bias, int M, int K)
    lib.gemv_fused_q8_0_bias.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemv_fused_q8_0_bias.restype = None

    # Dispatch functions (select AVX/scalar at compile time)
    fused_sig = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemv_fused_q5_0_bias_dispatch.argtypes = fused_sig
    lib.gemv_fused_q5_0_bias_dispatch.restype = None
    lib.gemv_fused_q8_0_bias_dispatch.argtypes = fused_sig
    lib.gemv_fused_q8_0_bias_dispatch.restype = None


def test_gemv_fused_q5_0(lib, M, K, tolerance=1e-4, n_runs=5, with_bias=True, use_dispatch=False):
    """
    Test gemv_fused_q5_0_bias against unfused sequence.

    Unfused: quantize_row_q8_0 + gemv_q5_0_q8_0 + add_inplace_f32
    Fused: gemv_fused_q5_0_bias (or dispatch variant)

    Returns: (passed, max_diff, mean_diff, unfused_time_us, fused_time_us)
    """
    np.random.seed(42)

    # Create test data
    x = np.random.randn(K).astype(np.float32) * 0.1
    bias = np.random.randn(M).astype(np.float32) * 0.01 if with_bias else None

    # Create Q5_0 weights
    W_bytes = create_q5_0_weights(M, K)
    W = (ctypes.c_uint8 * len(W_bytes)).from_buffer_copy(W_bytes)

    # Allocate Q8_0 buffer for quantized input
    blocks_per_row = K // QK8_0
    x_q8_size = blocks_per_row * BLOCK_Q8_0_SIZE
    x_q8 = (ctypes.c_uint8 * x_q8_size)()

    # Output buffers
    y_unfused = np.zeros(M, dtype=np.float32)
    y_fused = np.zeros(M, dtype=np.float32)

    # Get ctypes pointers
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias_ptr = bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias is not None else None
    y_unfused_ptr = y_unfused.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    y_fused_ptr = y_fused.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # --- Unfused sequence ---
    def run_unfused():
        # Step 1: Quantize input
        lib.quantize_row_q8_0(x_ptr, ctypes.cast(x_q8, ctypes.c_void_p), K)
        # Step 2: GEMV
        lib.gemv_q5_0_q8_0(y_unfused_ptr, ctypes.cast(W, ctypes.c_void_p),
                           ctypes.cast(x_q8, ctypes.c_void_p), M, K)
        # Step 3: Bias add
        if bias is not None:
            lib.add_inplace_f32(y_unfused_ptr, bias_ptr, M)

    # --- Fused kernel ---
    fused_fn = lib.gemv_fused_q5_0_bias_dispatch if use_dispatch else lib.gemv_fused_q5_0_bias
    def run_fused():
        fused_fn(y_fused_ptr, ctypes.cast(W, ctypes.c_void_p),
                 x_ptr, bias_ptr, M, K)

    # Warmup
    for _ in range(3):
        run_unfused()
        run_fused()

    # Reset outputs
    y_unfused.fill(0)
    y_fused.fill(0)

    # Run once for correctness
    run_unfused()
    run_fused()

    # Check for NaN/Inf
    if np.any(np.isnan(y_unfused)) or np.any(np.isinf(y_unfused)):
        return False, float('inf'), float('inf'), 0, 0
    if np.any(np.isnan(y_fused)) or np.any(np.isinf(y_fused)):
        return False, float('inf'), float('inf'), 0, 0

    # Compare
    diff = np.abs(y_fused - y_unfused)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    passed = max_diff <= tolerance

    # Benchmark unfused
    unfused_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_unfused()
        unfused_times.append((time.perf_counter() - start) * 1e6)
    unfused_time = min(unfused_times)

    # Benchmark fused
    fused_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_fused()
        fused_times.append((time.perf_counter() - start) * 1e6)
    fused_time = min(fused_times)

    return passed, max_diff, mean_diff, unfused_time, fused_time


def test_gemv_fused_q8_0(lib, M, K, tolerance=1e-4, n_runs=5, with_bias=True, use_dispatch=False):
    """
    Test gemv_fused_q8_0_bias against unfused sequence.

    Unfused: quantize_row_q8_0 + gemv_q8_0_q8_0 + add_inplace_f32
    Fused: gemv_fused_q8_0_bias (or dispatch variant)

    Returns: (passed, max_diff, mean_diff, unfused_time_us, fused_time_us)
    """
    np.random.seed(42)

    # Create test data
    x = np.random.randn(K).astype(np.float32) * 0.1
    bias = np.random.randn(M).astype(np.float32) * 0.01 if with_bias else None

    # Create Q8_0 weights
    W_bytes = create_q8_0_weights(M, K)
    W = (ctypes.c_uint8 * len(W_bytes)).from_buffer_copy(W_bytes)

    # Allocate Q8_0 buffer for quantized input
    blocks_per_row = K // QK8_0
    x_q8_size = blocks_per_row * BLOCK_Q8_0_SIZE
    x_q8 = (ctypes.c_uint8 * x_q8_size)()

    # Output buffers
    y_unfused = np.zeros(M, dtype=np.float32)
    y_fused = np.zeros(M, dtype=np.float32)

    # Get ctypes pointers
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias_ptr = bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias is not None else None
    y_unfused_ptr = y_unfused.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    y_fused_ptr = y_fused.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # --- Unfused sequence ---
    def run_unfused():
        # Step 1: Quantize input
        lib.quantize_row_q8_0(x_ptr, ctypes.cast(x_q8, ctypes.c_void_p), K)
        # Step 2: GEMV
        lib.gemv_q8_0_q8_0(y_unfused_ptr, ctypes.cast(W, ctypes.c_void_p),
                           ctypes.cast(x_q8, ctypes.c_void_p), M, K)
        # Step 3: Bias add
        if bias is not None:
            lib.add_inplace_f32(y_unfused_ptr, bias_ptr, M)

    # --- Fused kernel ---
    fused_fn = lib.gemv_fused_q8_0_bias_dispatch if use_dispatch else lib.gemv_fused_q8_0_bias
    def run_fused():
        fused_fn(y_fused_ptr, ctypes.cast(W, ctypes.c_void_p),
                 x_ptr, bias_ptr, M, K)

    # Warmup
    for _ in range(3):
        run_unfused()
        run_fused()

    # Reset outputs
    y_unfused.fill(0)
    y_fused.fill(0)

    # Run once for correctness
    run_unfused()
    run_fused()

    # Check for NaN/Inf
    if np.any(np.isnan(y_unfused)) or np.any(np.isinf(y_unfused)):
        return False, float('inf'), float('inf'), 0, 0
    if np.any(np.isnan(y_fused)) or np.any(np.isinf(y_fused)):
        return False, float('inf'), float('inf'), 0, 0

    # Compare
    diff = np.abs(y_fused - y_unfused)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    passed = max_diff <= tolerance

    # Benchmark unfused
    unfused_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_unfused()
        unfused_times.append((time.perf_counter() - start) * 1e6)
    unfused_time = min(unfused_times)

    # Benchmark fused
    fused_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_fused()
        fused_times.append((time.perf_counter() - start) * 1e6)
    fused_time = min(fused_times)

    return passed, max_diff, mean_diff, unfused_time, fused_time


def test_fused_vs_llamacpp_q5_0(ck_lib, llama_lib, M, K, tolerance=1e-4, n_runs=5, with_bias=True):
    """
    Test CK fused dispatch against llama.cpp reference directly.

    llama.cpp: test_gemv_q5_0 (per-row: quantize_row_q8_0_ref + ggml_vec_dot_q5_0_q8_0)
    CK fused:  gemv_fused_q5_0_bias_dispatch (all rows in one call)

    Returns: (passed, max_diff, mean_diff, llama_time_us, fused_time_us)
    """
    np.random.seed(42)

    x = np.random.randn(K).astype(np.float32) * 0.1
    bias = np.random.randn(M).astype(np.float32) * 0.01 if with_bias else None

    W_bytes = create_q5_0_weights(M, K)
    W = (ctypes.c_uint8 * len(W_bytes)).from_buffer_copy(W_bytes)

    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    row_bytes = (K // QK5_0) * BLOCK_Q5_0_SIZE
    bias_ptr = bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias is not None else None

    # --- llama.cpp reference (row by row) ---
    def run_llama():
        y = np.zeros(M, dtype=np.float32)
        for row in range(M):
            out_val = ctypes.c_float(0.0)
            w_row_ptr = ctypes.cast(ctypes.addressof(W) + row * row_bytes, ctypes.c_void_p)
            llama_lib.test_gemv_q5_0(w_row_ptr, x_ptr, ctypes.byref(out_val), K)
            y[row] = out_val.value
        if bias is not None:
            y += bias
        return y

    # --- CK fused dispatch ---
    y_fused = np.zeros(M, dtype=np.float32)
    y_fused_ptr = y_fused.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def run_fused():
        y_fused.fill(0)
        ck_lib.gemv_fused_q5_0_bias_dispatch(
            y_fused_ptr, ctypes.cast(W, ctypes.c_void_p), x_ptr, bias_ptr, M, K)

    # Warmup
    for _ in range(3):
        run_llama()
        run_fused()

    # Correctness
    y_llama = run_llama()
    run_fused()

    if np.any(np.isnan(y_llama)) or np.any(np.isnan(y_fused)):
        return False, float('inf'), float('inf'), 0, 0

    diff = np.abs(y_fused - y_llama)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    passed = max_diff <= tolerance

    # Benchmark llama.cpp
    llama_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_llama()
        llama_times.append((time.perf_counter() - start) * 1e6)
    llama_time = min(llama_times)

    # Benchmark CK fused
    fused_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_fused()
        fused_times.append((time.perf_counter() - start) * 1e6)
    fused_time = min(fused_times)

    return passed, max_diff, mean_diff, llama_time, fused_time


def test_fused_vs_llamacpp_q8_0(ck_lib, llama_lib, M, K, tolerance=1e-4, n_runs=5, with_bias=True):
    """
    Test CK fused dispatch against llama.cpp reference directly.

    llama.cpp: test_gemv_q8_0 (per-row: quantize_row_q8_0_ref + ggml_vec_dot_q8_0_q8_0)
    CK fused:  gemv_fused_q8_0_bias_dispatch (all rows in one call)

    Returns: (passed, max_diff, mean_diff, llama_time_us, fused_time_us)
    """
    np.random.seed(42)

    x = np.random.randn(K).astype(np.float32) * 0.1
    bias = np.random.randn(M).astype(np.float32) * 0.01 if with_bias else None

    W_bytes = create_q8_0_weights(M, K)
    W = (ctypes.c_uint8 * len(W_bytes)).from_buffer_copy(W_bytes)

    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    row_bytes = (K // QK8_0) * BLOCK_Q8_0_SIZE
    bias_ptr = bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias is not None else None

    # --- llama.cpp reference (row by row) ---
    def run_llama():
        y = np.zeros(M, dtype=np.float32)
        for row in range(M):
            out_val = ctypes.c_float(0.0)
            w_row_ptr = ctypes.cast(ctypes.addressof(W) + row * row_bytes, ctypes.c_void_p)
            llama_lib.test_gemv_q8_0(w_row_ptr, x_ptr, ctypes.byref(out_val), K)
            y[row] = out_val.value
        if bias is not None:
            y += bias
        return y

    # --- CK fused dispatch ---
    y_fused = np.zeros(M, dtype=np.float32)
    y_fused_ptr = y_fused.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def run_fused():
        y_fused.fill(0)
        ck_lib.gemv_fused_q8_0_bias_dispatch(
            y_fused_ptr, ctypes.cast(W, ctypes.c_void_p), x_ptr, bias_ptr, M, K)

    # Warmup
    for _ in range(3):
        run_llama()
        run_fused()

    # Correctness
    y_llama = run_llama()
    run_fused()

    if np.any(np.isnan(y_llama)) or np.any(np.isnan(y_fused)):
        return False, float('inf'), float('inf'), 0, 0

    diff = np.abs(y_fused - y_llama)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    passed = max_diff <= tolerance

    # Benchmark llama.cpp
    llama_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_llama()
        llama_times.append((time.perf_counter() - start) * 1e6)
    llama_time = min(llama_times)

    # Benchmark CK fused
    fused_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_fused()
        fused_times.append((time.perf_counter() - start) * 1e6)
    fused_time = min(fused_times)

    return passed, max_diff, mean_diff, llama_time, fused_time


def main():
    parser = argparse.ArgumentParser(description="Fused GEMV Kernel Parity Test")
    parser.add_argument('--tol', '--tolerance', type=float, default=1e-4,
                        help='Tolerance for CK internal parity (default: 1e-4)')
    parser.add_argument('--llama-tol-q5', type=float, default=1e-2,
                        help='Tolerance for Q5_0 vs llama.cpp (default: 1e-2)')
    parser.add_argument('--llama-tol-q8', type=float, default=5e-2,
                        help='Tolerance for Q8_0 vs llama.cpp (default: 5e-2)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed statistics')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test (fewer configurations)')
    args = parser.parse_args()

    # Header
    print("=" * 80)
    print(f"{BOLD}FUSED GEMV KERNEL PARITY TESTS: Fused vs Unfused + llama.cpp{RESET}")
    print("=" * 80)
    print()
    print(f"{YELLOW}Purpose:{RESET}   Verify fused GEMV kernels match both CK unfused AND llama.cpp")
    print(f"{YELLOW}Method:{RESET}    Compare gemv_fused_q*_bias vs (quantize + gemv + bias_add)")
    print(f"{YELLOW}         {RESET}  Compare gemv_fused_q*_bias vs llama.cpp test_gemv_q*_0")
    print(f"{YELLOW}Tolerance:{RESET} max_diff < {args.tol:.0e} for all elements")
    print()
    print(f"{YELLOW}Kernels Tested:{RESET}")
    print("  - gemv_fused_q5_0_bias: FP32 input → online Q8 → Q5_0 weights → FP32 + bias")
    print("  - gemv_fused_q8_0_bias: FP32 input → online Q8 → Q8_0 weights → FP32 + bias")
    print()

    # Load libraries
    lib = load_library()
    if lib is None:
        print(f"[{RED}FAIL{RESET}] CK library not found. Run: make")
        return 1

    llama_lib = load_llamacpp_library()
    if llama_lib is None:
        print(f"{YELLOW}[WARN]{RESET} llama.cpp library not found - skipping direct llama.cpp comparison")
    print()

    # Set up function signatures
    try:
        setup_signatures(lib)
    except Exception as e:
        print(f"[{RED}FAIL{RESET}] Failed to set up function signatures: {e}")
        print(f"       Make sure fused kernels are compiled into the library.")
        return 1

    if llama_lib:
        try:
            setup_llamacpp_signatures(llama_lib)
        except Exception as e:
            print(f"{YELLOW}[WARN]{RESET} Failed to set up llama.cpp signatures: {e}")
            llama_lib = None

    # Test configurations
    if args.quick:
        configs = [
            # (M, K, name)
            (896, 896, "Qwen2-embed"),
            (4864, 896, "Qwen2-MLP-up"),
        ]
    else:
        configs = [
            # Qwen2-0.5B dimensions
            (896, 896, "Qwen2-QKV"),
            (128, 896, "Qwen2-KV"),
            (4864, 896, "Qwen2-MLP-up"),
            (896, 4864, "Qwen2-MLP-down"),
            # General sizes
            (512, 512, "Square-512"),
            (1024, 512, "Rect-1024x512"),
            (2048, 896, "Large-2048"),
        ]

    # Run Q5_0 tests
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{BOLD}Q5_0 FUSED GEMV TESTS{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")

    q5_results = []
    q5_passed = 0

    for M, K, name in configs:
        print(f"--- test_gemv_fused_q5_0 ({name}: M={M}, K={K}) ---")

        passed, max_diff, mean_diff, unfused_time, fused_time = test_gemv_fused_q5_0(
            lib, M, K, args.tol
        )

        q5_results.append((name, M, K, passed, max_diff, mean_diff, unfused_time, fused_time))

        if passed:
            q5_passed += 1
            status = f"[{GREEN}PASS{RESET}]"
        else:
            status = f"[{RED}FAIL{RESET}]"

        speedup = unfused_time / fused_time if fused_time > 0 else 0
        if speedup >= 1.2:
            speedup_color = GREEN
        elif speedup >= 1.0:
            speedup_color = YELLOW
        else:
            speedup_color = RED

        print(f"{status} {name}: max_diff={max_diff:.2e}, mean={mean_diff:.2e} (tol={args.tol:.0e})")
        print(f"      Performance: unfused={unfused_time:.0f}us, fused={fused_time:.0f}us, "
              f"speedup={speedup_color}{speedup:.2f}x{RESET}")
        print()

    # Run Q8_0 tests
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{BOLD}Q8_0 FUSED GEMV TESTS{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")

    q8_results = []
    q8_passed = 0

    for M, K, name in configs:
        print(f"--- test_gemv_fused_q8_0 ({name}: M={M}, K={K}) ---")

        passed, max_diff, mean_diff, unfused_time, fused_time = test_gemv_fused_q8_0(
            lib, M, K, args.tol
        )

        q8_results.append((name, M, K, passed, max_diff, mean_diff, unfused_time, fused_time))

        if passed:
            q8_passed += 1
            status = f"[{GREEN}PASS{RESET}]"
        else:
            status = f"[{RED}FAIL{RESET}]"

        speedup = unfused_time / fused_time if fused_time > 0 else 0
        if speedup >= 1.2:
            speedup_color = GREEN
        elif speedup >= 1.0:
            speedup_color = YELLOW
        else:
            speedup_color = RED

        print(f"{status} {name}: max_diff={max_diff:.2e}, mean={mean_diff:.2e} (tol={args.tol:.0e})")
        print(f"      Performance: unfused={unfused_time:.0f}us, fused={fused_time:.0f}us, "
              f"speedup={speedup_color}{speedup:.2f}x{RESET}")
        print()

    # Run Q5_0 DISPATCH tests (SIMD path)
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{BOLD}Q5_0 FUSED GEMV DISPATCH (SIMD) TESTS{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")

    q5d_results = []
    q5d_passed = 0

    for M, K, name in configs:
        print(f"--- test_gemv_fused_q5_0_dispatch ({name}: M={M}, K={K}) ---")

        passed, max_diff, mean_diff, unfused_time, fused_time = test_gemv_fused_q5_0(
            lib, M, K, args.tol, use_dispatch=True
        )

        q5d_results.append((name, M, K, passed, max_diff, mean_diff, unfused_time, fused_time))

        if passed:
            q5d_passed += 1
            status = f"[{GREEN}PASS{RESET}]"
        else:
            status = f"[{RED}FAIL{RESET}]"

        speedup = unfused_time / fused_time if fused_time > 0 else 0
        if speedup >= 1.2:
            speedup_color = GREEN
        elif speedup >= 1.0:
            speedup_color = YELLOW
        else:
            speedup_color = RED

        print(f"{status} {name}: max_diff={max_diff:.2e}, mean={mean_diff:.2e} (tol={args.tol:.0e})")
        print(f"      Performance: unfused={unfused_time:.0f}us, fused={fused_time:.0f}us, "
              f"speedup={speedup_color}{speedup:.2f}x{RESET}")
        print()

    # Run Q8_0 DISPATCH tests (SIMD path)
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{BOLD}Q8_0 FUSED GEMV DISPATCH (SIMD) TESTS{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")

    q8d_results = []
    q8d_passed = 0

    for M, K, name in configs:
        print(f"--- test_gemv_fused_q8_0_dispatch ({name}: M={M}, K={K}) ---")

        passed, max_diff, mean_diff, unfused_time, fused_time = test_gemv_fused_q8_0(
            lib, M, K, args.tol, use_dispatch=True
        )

        q8d_results.append((name, M, K, passed, max_diff, mean_diff, unfused_time, fused_time))

        if passed:
            q8d_passed += 1
            status = f"[{GREEN}PASS{RESET}]"
        else:
            status = f"[{RED}FAIL{RESET}]"

        speedup = unfused_time / fused_time if fused_time > 0 else 0
        if speedup >= 1.2:
            speedup_color = GREEN
        elif speedup >= 1.0:
            speedup_color = YELLOW
        else:
            speedup_color = RED

        print(f"{status} {name}: max_diff={max_diff:.2e}, mean={mean_diff:.2e} (tol={args.tol:.0e})")
        print(f"      Performance: unfused={unfused_time:.0f}us, fused={fused_time:.0f}us, "
              f"speedup={speedup_color}{speedup:.2f}x{RESET}")
        print()

    # Run llama.cpp direct comparison tests
    llama_q5_results = []
    llama_q5_passed = 0
    llama_q8_results = []
    llama_q8_passed = 0

    if llama_lib:
        print(f"\n{CYAN}{'='*80}{RESET}")
        print(f"{BOLD}Q5_0 CK FUSED vs LLAMA.CPP DIRECT{RESET}")
        print(f"{CYAN}{'='*80}{RESET}")
        print(f"  Note: CK and llama.cpp use different SIMD paths, so FP rounding differs.")
        print(f"  Tolerance: {args.llama_tol_q5:.0e} (matches existing parity tests)\n")

        for M, K, name in configs:
            print(f"--- CK fused_dispatch vs llama.cpp test_gemv_q5_0 ({name}: M={M}, K={K}) ---")

            passed, max_diff, mean_diff, llama_time, fused_time = test_fused_vs_llamacpp_q5_0(
                lib, llama_lib, M, K, args.llama_tol_q5
            )

            llama_q5_results.append((name, M, K, passed, max_diff, mean_diff, llama_time, fused_time))

            if passed:
                llama_q5_passed += 1
                status = f"[{GREEN}PASS{RESET}]"
            else:
                status = f"[{RED}FAIL{RESET}]"

            speedup = llama_time / fused_time if fused_time > 0 else 0
            if speedup >= 1.2:
                speedup_color = GREEN
            elif speedup >= 1.0:
                speedup_color = YELLOW
            else:
                speedup_color = RED

            print(f"{status} {name}: max_diff={max_diff:.2e}, mean={mean_diff:.2e} (tol={args.llama_tol_q5:.0e})")
            print(f"      Performance: llama.cpp={llama_time:.0f}us, CK fused={fused_time:.0f}us, "
                  f"speedup={speedup_color}{speedup:.2f}x{RESET}")
            print()

        print(f"\n{CYAN}{'='*80}{RESET}")
        print(f"{BOLD}Q8_0 CK FUSED vs LLAMA.CPP DIRECT{RESET}")
        print(f"{CYAN}{'='*80}{RESET}")
        print(f"  Note: CK and llama.cpp use different SIMD paths, so FP rounding differs.")
        print(f"  Tolerance: {args.llama_tol_q8:.0e} (matches existing parity tests)\n")

        for M, K, name in configs:
            print(f"--- CK fused_dispatch vs llama.cpp test_gemv_q8_0 ({name}: M={M}, K={K}) ---")

            passed, max_diff, mean_diff, llama_time, fused_time = test_fused_vs_llamacpp_q8_0(
                lib, llama_lib, M, K, args.llama_tol_q8
            )

            llama_q8_results.append((name, M, K, passed, max_diff, mean_diff, llama_time, fused_time))

            if passed:
                llama_q8_passed += 1
                status = f"[{GREEN}PASS{RESET}]"
            else:
                status = f"[{RED}FAIL{RESET}]"

            speedup = llama_time / fused_time if fused_time > 0 else 0
            if speedup >= 1.2:
                speedup_color = GREEN
            elif speedup >= 1.0:
                speedup_color = YELLOW
            else:
                speedup_color = RED

            print(f"{status} {name}: max_diff={max_diff:.2e}, mean={mean_diff:.2e} (tol={args.llama_tol_q8:.0e})")
            print(f"      Performance: llama.cpp={llama_time:.0f}us, CK fused={fused_time:.0f}us, "
                  f"speedup={speedup_color}{speedup:.2f}x{RESET}")
            print()

    # Summary
    total_tests = len(configs) * 4  # scalar Q5, scalar Q8, dispatch Q5, dispatch Q8
    total_passed = q5_passed + q8_passed + q5d_passed + q8d_passed

    if llama_lib:
        total_tests += len(configs) * 2  # llama Q5, llama Q8
        total_passed += llama_q5_passed + llama_q8_passed

    print("=" * 80)
    print(f"{BOLD}FUSED GEMV KERNEL TEST SUMMARY{RESET}")
    print("=" * 80)
    print(f"Q5_0 Scalar:   {q5_passed}/{len(configs)}")
    print(f"Q8_0 Scalar:   {q8_passed}/{len(configs)}")
    print(f"Q5_0 Dispatch: {q5d_passed}/{len(configs)}")
    print(f"Q8_0 Dispatch: {q8d_passed}/{len(configs)}")
    if llama_lib:
        print(f"Q5_0 vs llama: {llama_q5_passed}/{len(configs)}")
        print(f"Q8_0 vs llama: {llama_q8_passed}/{len(configs)}")
    else:
        print(f"Q5_0 vs llama: {YELLOW}SKIPPED{RESET} (llama.cpp library not found)")
        print(f"Q8_0 vs llama: {YELLOW}SKIPPED{RESET} (llama.cpp library not found)")
    print(f"Total Passed:  {total_passed}/{total_tests}")
    print()

    # Performance results table
    def print_results_table(label, results):
        print(f"\n{BOLD}{label}:{RESET}")
        for name, M, K, passed, max_diff, mean_diff, unfused_time, fused_time in results:
            speedup = unfused_time / fused_time if fused_time > 0 else 0
            status = f"{GREEN}\u2713{RESET}" if passed else f"{RED}\u2717{RESET}"
            if speedup >= 1.2:
                speedup_str = f"{GREEN}{speedup:.2f}x{RESET}"
            elif speedup >= 1.0:
                speedup_str = f"{YELLOW}{speedup:.2f}x{RESET}"
            else:
                speedup_str = f"{RED}{speedup:.2f}x{RESET}"
            print(f"{status} {name:<18} {M:>6} {K:>6} {max_diff:>12.2e} {unfused_time:>10.0f} "
                  f"{fused_time:>10.0f} {speedup_str:>18}")

    def print_parity_table(label, results):
        print(f"\n{BOLD}{label}:{RESET}")
        for name, M, K, passed, max_diff, mean_diff in results:
            status = f"{GREEN}\u2713{RESET}" if passed else f"{RED}\u2717{RESET}"
            print(f"{status} {name:<18} {M:>6} {K:>6} {max_diff:>12.2e} {mean_diff:>12.2e}")

    print(f"{'Test':<20} {'M':>6} {'K':>6} {'Max Diff':>12} {'Unfused':>10} {'Fused':>10} {'Speedup':>10}")
    print("-" * 80)

    print_results_table("Q5_0 Scalar", q5_results)
    print_results_table("Q8_0 Scalar", q8_results)
    print_results_table("Q5_0 Dispatch (SIMD)", q5d_results)
    print_results_table("Q8_0 Dispatch (SIMD)", q8d_results)

    if llama_lib:
        def print_llama_table(label, results):
            print(f"\n{BOLD}{label}:{RESET}")
            for name, M, K, passed, max_diff, mean_diff, llama_time, fused_time in results:
                speedup = llama_time / fused_time if fused_time > 0 else 0
                status = f"{GREEN}\u2713{RESET}" if passed else f"{RED}\u2717{RESET}"
                if speedup >= 1.2:
                    speedup_str = f"{GREEN}{speedup:.2f}x{RESET}"
                elif speedup >= 1.0:
                    speedup_str = f"{YELLOW}{speedup:.2f}x{RESET}"
                else:
                    speedup_str = f"{RED}{speedup:.2f}x{RESET}"
                print(f"{status} {name:<18} {M:>6} {K:>6} {max_diff:>12.2e} {llama_time:>10.0f} "
                      f"{fused_time:>10.0f} {speedup_str:>18}")

        print(f"\n{'Test':<20} {'M':>6} {'K':>6} {'Max Diff':>12} {'llama.cpp':>10} {'CK fused':>10} {'Speedup':>10}")
        print("-" * 80)
        print_llama_table("Q5_0 CK Fused vs llama.cpp", llama_q5_results)
        print_llama_table("Q8_0 CK Fused vs llama.cpp", llama_q8_results)

    print()
    if total_passed == total_tests:
        print(f"{GREEN}All fused GEMV kernels match references!{RESET}")
        return 0
    else:
        print(f"{RED}Some fused GEMV kernel tests FAILED!{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
