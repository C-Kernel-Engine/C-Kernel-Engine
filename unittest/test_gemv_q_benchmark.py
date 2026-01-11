#!/usr/bin/env python3
"""
Comprehensive benchmark for quantized GEMV kernels.

Tests:
1. Numerical accuracy vs reference implementation
2. Performance (throughput in GFLOPS, latency in ms)
3. Comparison with llama.cpp (if available)

Usage:
    python unittest/test_gemv_q_benchmark.py
    python unittest/test_gemv_q_benchmark.py --quick    # Fast test
    python unittest/test_gemv_q_benchmark.py --verbose  # Detailed output
"""

import ctypes
import numpy as np
import time
import os
import sys
import struct
import argparse
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Configuration
# ============================================================================

# Block sizes for different quant formats
QK5_0 = 32  # Q5_0: 32 weights per block
QK8_0 = 32  # Q8_0: 32 weights per block
QK_K = 256  # K-quants: 256 weights per block

# Test configurations
TEST_CONFIGS = [
    # (M, K, name) - M=output rows, K=input dimension
    (896, 896, "small_square"),      # Small attention projection
    (896, 4864, "qkv_proj"),         # QKV projection (Qwen 0.5B)
    (4864, 896, "mlp_up"),           # MLP up projection
    (896, 4864, "mlp_down"),         # MLP down projection
    (1536, 1536, "medium_square"),   # Medium
    (4096, 4096, "large_square"),    # Large (7B model size)
]

# ============================================================================
# FP16 conversion helpers
# ============================================================================

def fp32_to_fp16(f):
    """Convert float32 to float16 (IEEE 754)"""
    return np.float16(f).view(np.uint16)

def fp16_to_fp32(h):
    """Convert float16 to float32"""
    return np.array([h], dtype=np.uint16).view(np.float16).astype(np.float32)[0]

# ============================================================================
# Q5_0 quantization (reference)
# ============================================================================

def quantize_q5_0_block(weights):
    """Quantize 32 floats to Q5_0 block format (llama.cpp compatible)"""
    assert len(weights) == QK5_0

    # Find scale
    amax = np.max(np.abs(weights))
    d = amax / 15.0 if amax > 0 else 1.0
    id_ = 1.0 / d if d != 0 else 0.0

    # Quantize to 5-bit (0-31, subtract 16 for signed)
    q5 = np.clip(np.round(weights * id_) + 16, 0, 31).astype(np.uint8)

    # Pack into Q5_0 format:
    # - d: FP16 scale (2 bytes)
    # - qh: high bits packed (4 bytes for 32 weights)
    # - qs: low 4 bits packed (16 bytes for 32 weights)

    d_fp16 = fp32_to_fp16(d)

    # Extract low 4 bits and high 1 bit
    low4 = q5 & 0x0F
    high1 = (q5 >> 4) & 0x01

    # Pack low 4 bits: 2 weights per byte
    # First 16 weights in low nibbles, next 16 in high nibbles
    qs = np.zeros(16, dtype=np.uint8)
    for j in range(16):
        qs[j] = low4[j] | (low4[j + 16] << 4)

    # Pack high bits: 32 bits into 4 bytes
    # Bit j is for weight j (first 16), bit j+12 is for weight j+16 (second 16)
    qh = 0
    for j in range(16):
        if high1[j]:
            qh |= (1 << j)
        if high1[j + 16]:
            qh |= (1 << (j + 12))
    qh_bytes = struct.pack('<I', qh)

    # Pack block: d (2) + qh (4) + qs (16) = 22 bytes
    block = struct.pack('<H', d_fp16) + qh_bytes + bytes(qs)
    return block

def dequantize_q5_0_block(block_bytes):
    """Dequantize Q5_0 block to 32 floats"""
    d_fp16 = struct.unpack('<H', block_bytes[0:2])[0]
    d = fp16_to_fp32(d_fp16)

    qh = struct.unpack('<I', block_bytes[2:6])[0]
    qs = np.frombuffer(block_bytes[6:22], dtype=np.uint8)

    weights = np.zeros(32, dtype=np.float32)
    for j in range(16):
        lo = qs[j] & 0x0F
        hi = qs[j] >> 4
        xh_0 = ((qh >> j) << 4) & 0x10
        xh_1 = (qh >> (j + 12)) & 0x10

        q0 = (lo | xh_0) - 16
        q1 = (hi | xh_1) - 16

        weights[j] = d * q0
        weights[j + 16] = d * q1

    return weights

def quantize_matrix_q5_0(weights):
    """Quantize MxK matrix to Q5_0 format"""
    M, K = weights.shape
    assert K % QK5_0 == 0
    blocks_per_row = K // QK5_0

    blocks = []
    for row in range(M):
        for b in range(blocks_per_row):
            block_weights = weights[row, b*QK5_0:(b+1)*QK5_0]
            blocks.append(quantize_q5_0_block(block_weights))

    return b''.join(blocks)

# ============================================================================
# Q4_K quantization (reference)
# ============================================================================

def quantize_q4_k_block(weights):
    """Quantize 256 floats to Q4_K block format"""
    assert len(weights) == QK_K

    # Q4_K has nested scales: super-block scale + 8 sub-block scales
    # Each sub-block is 32 weights

    # Find super-block scale
    amax = np.max(np.abs(weights))
    d = amax / 127.0 if amax > 0 else 1.0

    # Compute sub-block scales and mins
    scales = []
    mins = []
    for i in range(8):
        sub = weights[i*32:(i+1)*32]
        sub_max = np.max(sub)
        sub_min = np.min(sub)

        # Scale and min for 4-bit quantization (0-15)
        if sub_max - sub_min > 0:
            sc = (sub_max - sub_min) / 15.0
            m = sub_min
        else:
            sc = 1.0
            m = 0.0

        scales.append(sc / d if d != 0 else 0)
        mins.append(m / d if d != 0 else 0)

    # Quantize weights
    qs = np.zeros(128, dtype=np.uint8)  # 256 weights, 4 bits each

    for i in range(8):
        sub = weights[i*32:(i+1)*32]
        sc = scales[i] * d
        m = mins[i] * d

        for j in range(32):
            if sc > 0:
                q = int(np.clip(np.round((sub[j] - m) / sc), 0, 15))
            else:
                q = 0

            # Pack 4-bit values
            byte_idx = i * 16 + j // 2
            if j % 2 == 0:
                qs[byte_idx] = q
            else:
                qs[byte_idx] |= (q << 4)

    # Pack block (simplified - actual format is more complex)
    d_fp16 = fp32_to_fp16(d)
    dmin_fp16 = fp32_to_fp16(0.0)  # Simplified

    # Pack scales (simplified)
    scale_bytes = bytes(int(np.clip(s * 63, 0, 63)) for s in scales[:8])

    block = struct.pack('<HH', d_fp16, dmin_fp16) + scale_bytes + bytes([0]*4) + bytes(qs)
    return block

# ============================================================================
# Load C library
# ============================================================================

def load_library():
    """Load the C kernel library"""
    lib_path = Path(__file__).parent.parent / "build" / "libckernel_engine.so"
    if not lib_path.exists():
        print(f"Library not found at {lib_path}")
        print("Run 'make' first to build the library")
        return None

    lib = ctypes.CDLL(str(lib_path))

    # Define function signatures
    # gemv_q5_0(float *y, const void *W, const float *x, int M, int K)
    lib.gemv_q5_0.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    lib.gemv_q5_0.restype = None

    # gemv_q4_k
    lib.gemv_q4_k.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    lib.gemv_q4_k.restype = None

    # gemv_q8_0
    lib.gemv_q8_0.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    lib.gemv_q8_0.restype = None

    # gemv_q6_k
    lib.gemv_q6_k.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    lib.gemv_q6_k.restype = None

    return lib

# ============================================================================
# Reference implementation (pure Python)
# ============================================================================

def gemv_q5_0_ref(W_bytes, x, M, K):
    """Reference Q5_0 GEMV in pure Python"""
    blocks_per_row = K // QK5_0
    block_size = 22  # bytes per Q5_0 block

    y = np.zeros(M, dtype=np.float32)

    for row in range(M):
        sum_val = 0.0
        for b in range(blocks_per_row):
            block_offset = (row * blocks_per_row + b) * block_size
            block_bytes = W_bytes[block_offset:block_offset + block_size]

            # Dequantize
            weights = dequantize_q5_0_block(block_bytes)

            # Dot product
            x_slice = x[b * QK5_0:(b + 1) * QK5_0]
            sum_val += np.dot(weights, x_slice)

        y[row] = sum_val

    return y

# ============================================================================
# Benchmark functions
# ============================================================================

def benchmark_kernel(func, args, warmup=5, iterations=100):
    """Benchmark a kernel function"""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
    }

def run_accuracy_test(lib, M, K, quant_type='q5_0', verbose=False):
    """Test accuracy of C kernel vs Python reference"""
    np.random.seed(42)

    # Generate random weights and input
    W_float = np.random.randn(M, K).astype(np.float32) * 0.1
    x = np.random.randn(K).astype(np.float32)

    if quant_type == 'q5_0':
        # Quantize weights
        W_bytes = quantize_matrix_q5_0(W_float)
        block_size = 22

        # Python reference
        y_ref = gemv_q5_0_ref(W_bytes, x, M, K)

        # C kernel
        y_c = np.zeros(M, dtype=np.float32)
        W_ptr = (ctypes.c_ubyte * len(W_bytes)).from_buffer_copy(W_bytes)
        x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        lib.gemv_q5_0(y_ptr, ctypes.cast(W_ptr, ctypes.c_void_p), x_ptr, M, K)
    else:
        raise ValueError(f"Unknown quant type: {quant_type}")

    # Compare
    max_diff = np.max(np.abs(y_ref - y_c))
    mean_diff = np.mean(np.abs(y_ref - y_c))
    rel_diff = max_diff / (np.max(np.abs(y_ref)) + 1e-10)

    passed = max_diff < 1e-4

    if verbose:
        print(f"  Accuracy test M={M}, K={K}:")
        print(f"    max_diff:  {max_diff:.6e}")
        print(f"    mean_diff: {mean_diff:.6e}")
        print(f"    rel_diff:  {rel_diff:.6e}")
        print(f"    Status: {'PASS' if passed else 'FAIL'}")

    return {
        'passed': passed,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'rel_diff': rel_diff,
    }

def run_performance_test(lib, M, K, quant_type='q5_0', iterations=50, verbose=False):
    """Benchmark C kernel performance"""
    np.random.seed(42)

    # Generate random weights and input
    W_float = np.random.randn(M, K).astype(np.float32) * 0.1
    x = np.random.randn(K).astype(np.float32)

    if quant_type == 'q5_0':
        W_bytes = quantize_matrix_q5_0(W_float)
        block_size = 22

        y_c = np.zeros(M, dtype=np.float32)
        W_ptr = (ctypes.c_ubyte * len(W_bytes)).from_buffer_copy(W_bytes)
        x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        def kernel_call():
            lib.gemv_q5_0(y_ptr, ctypes.cast(W_ptr, ctypes.c_void_p), x_ptr, M, K)
    else:
        raise ValueError(f"Unknown quant type: {quant_type}")

    # Benchmark
    results = benchmark_kernel(kernel_call, (), warmup=5, iterations=iterations)

    # Calculate throughput
    # GEMV: M*K multiply-adds = 2*M*K FLOPs
    flops = 2 * M * K
    gflops = (flops / 1e9) / (results['mean_ms'] / 1000)

    # Memory bandwidth (approximate)
    # Read: W_bytes + K floats, Write: M floats
    bytes_accessed = len(W_bytes) + K * 4 + M * 4
    gbps = (bytes_accessed / 1e9) / (results['mean_ms'] / 1000)

    results['gflops'] = gflops
    results['gbps'] = gbps
    results['flops'] = flops
    results['bytes'] = bytes_accessed

    if verbose:
        print(f"  Performance test M={M}, K={K}:")
        print(f"    Time:       {results['mean_ms']:.3f} ms (+/- {results['std_ms']:.3f})")
        print(f"    Throughput: {gflops:.2f} GFLOPS")
        print(f"    Bandwidth:  {gbps:.2f} GB/s")

    return results

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark quantized GEMV kernels')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer iterations')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quant', choices=['q5_0', 'all'], default='q5_0', help='Quantization type')
    args = parser.parse_args()

    iterations = 10 if args.quick else 50

    print("=" * 80)
    print("  QUANTIZED GEMV KERNEL BENCHMARK")
    print("=" * 80)

    # Load library
    lib = load_library()
    if lib is None:
        return 1

    print(f"\nLibrary loaded successfully")
    print(f"Iterations per test: {iterations}")

    # Get CPU info
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if 'model name' in line:
                    print(f"CPU: {line.split(':')[1].strip()}")
                    break
    except:
        pass

    print("\n" + "=" * 80)
    print("  ACCURACY TESTS")
    print("=" * 80)

    all_passed = True
    accuracy_results = []

    for M, K, name in TEST_CONFIGS:
        if K % QK5_0 != 0:
            continue

        result = run_accuracy_test(lib, M, K, 'q5_0', args.verbose)
        status = "\033[92mPASS\033[0m" if result['passed'] else "\033[91mFAIL\033[0m"
        print(f"  {name:20s} [{M:4d}x{K:4d}]: max_diff={result['max_diff']:.2e}  [{status}]")

        accuracy_results.append((name, M, K, result))
        if not result['passed']:
            all_passed = False

    print("\n" + "=" * 80)
    print("  PERFORMANCE TESTS")
    print("=" * 80)

    print(f"\n  {'Config':<20s} {'Shape':>12s} {'Time (ms)':>12s} {'GFLOPS':>10s} {'GB/s':>10s}")
    print("  " + "-" * 70)

    perf_results = []
    for M, K, name in TEST_CONFIGS:
        if K % QK5_0 != 0:
            continue

        result = run_performance_test(lib, M, K, 'q5_0', iterations, args.verbose)
        print(f"  {name:<20s} [{M:4d}x{K:4d}] {result['mean_ms']:>10.3f}ms {result['gflops']:>10.2f} {result['gbps']:>10.2f}")

        perf_results.append((name, M, K, result))

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    if all_passed:
        print("\n  \033[92mAll accuracy tests PASSED\033[0m")
    else:
        print("\n  \033[91mSome accuracy tests FAILED\033[0m")

    # Calculate average throughput
    avg_gflops = np.mean([r[3]['gflops'] for r in perf_results])
    avg_gbps = np.mean([r[3]['gbps'] for r in perf_results])

    print(f"\n  Average throughput: {avg_gflops:.2f} GFLOPS")
    print(f"  Average bandwidth:  {avg_gbps:.2f} GB/s")

    # Estimate tokens/s for Qwen 0.5B
    # Decode: ~324 Q5_0 GEMVs per token
    # Average shape: ~(896, 4864) + (4864, 896) + ...
    # Rough estimate: 50ms per token -> 20 tok/s
    total_gemv_time = sum(r[3]['mean_ms'] for r in perf_results)
    estimated_tok_s = 1000 / (total_gemv_time * 10)  # Rough multiplier for full model

    print(f"\n  Estimated decode: ~{estimated_tok_s:.1f} tok/s (rough estimate)")
    print(f"  llama.cpp target: ~35 tok/s")
    print(f"  Gap: ~{35/estimated_tok_s:.1f}x")

    print("\n" + "=" * 80)

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
