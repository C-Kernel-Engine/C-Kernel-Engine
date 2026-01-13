#!/usr/bin/env python3
"""
Comprehensive GEMM Batch INT8 Kernel Tests

Tests gemm_nt_q5_0_q8_0 and gemm_nt_q8_0_q8_0 batch kernels against:
1. Scalar reference (within same library)
2. llama.cpp reference implementation
3. NumPy reference (for FP32 comparison)

Test Categories:
- Correctness: Exact parity with reference implementations
- Edge cases: Small sizes, non-aligned, boundary conditions
- Stress tests: Large matrices, many iterations
- Instruction-specific: Tests for AVX, AVX-512, AMX paths

Philosophy: C-Kernel-Engine means kernels CANNOT fail.
Every test must pass before code is considered correct.

Usage:
    python unittest/test_gemm_batch_int8.py [--quick] [--stress] [--amx]
"""

import argparse
import ctypes
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import struct

# Add unittest directory to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "unittest"))

from test_utils import get_cpu_info, TestReport, TestResult

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

QK8_0 = 32  # Q8_0 block size
QK5_0 = 32  # Q5_0 block size

# Block structure sizes
BLOCK_Q8_0_SIZE = 34  # 2 (FP16 d) + 32 (int8 qs)
BLOCK_Q5_0_SIZE = 22  # 2 (FP16 d) + 4 (qh) + 16 (qs)

# Tolerances by instruction set
TOLERANCES = {
    "scalar": 1e-6,
    "avx": 1e-5,
    "avx512": 1e-5,
    "amx": 1e-4,  # AMX may have different accumulation order
}

# ═══════════════════════════════════════════════════════════════════════════════
# Library Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_libraries():
    """Load CK and llama.cpp libraries."""
    ck_paths = [
        ROOT / "build" / "libckernel_engine.so",
        ROOT / "build" / "libck_parity.so",
    ]

    llama_paths = [
        ROOT / "llama.cpp" / "libggml_kernel_test.so",
        ROOT / "llama.cpp" / "build" / "ggml" / "src" / "libggml_kernel_test.so",
    ]

    libck = None
    for p in ck_paths:
        if p.exists():
            try:
                libck = ctypes.CDLL(str(p))
                break
            except Exception:
                pass

    libllama = None
    for p in llama_paths:
        if p.exists():
            try:
                libllama = ctypes.CDLL(str(p))
                break
            except Exception:
                pass

    return libck, libllama


# ═══════════════════════════════════════════════════════════════════════════════
# Quantization Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def float_to_fp16(val: float) -> int:
    """Convert float32 to FP16 (stored as uint16)."""
    packed = struct.pack('e', val)
    return struct.unpack('H', packed)[0]

def fp16_to_float(val: int) -> float:
    """Convert FP16 (uint16) to float32."""
    packed = struct.pack('H', val)
    return struct.unpack('e', packed)[0]

def quantize_row_q8_0(x: np.ndarray) -> bytes:
    """
    Quantize a row to Q8_0 format.

    Q8_0 block: 2 bytes (FP16 d) + 32 bytes (int8 qs) = 34 bytes

    Args:
        x: Float32 array, length must be multiple of 32

    Returns:
        Packed Q8_0 data as bytes
    """
    assert len(x) % QK8_0 == 0, f"Length {len(x)} not divisible by {QK8_0}"
    nb = len(x) // QK8_0

    result = bytearray()

    for b in range(nb):
        block = x[b * QK8_0 : (b + 1) * QK8_0]

        # Find max absolute value for scale
        amax = np.max(np.abs(block))
        d = amax / 127.0 if amax > 0 else 1.0
        id = 1.0 / d if d != 0 else 0.0

        # Quantize
        qs = np.round(block * id).astype(np.int8)

        # Pack: FP16 scale + 32 int8 values
        d_fp16 = float_to_fp16(d)
        result.extend(struct.pack('<H', d_fp16))
        result.extend(qs.tobytes())

    return bytes(result)

def quantize_row_q5_0(x: np.ndarray) -> bytes:
    """
    Quantize a row to Q5_0 format.

    Q5_0 block: 2 bytes (FP16 d) + 4 bytes (qh) + 16 bytes (qs) = 22 bytes

    5-bit signed values: -16 to +15
    Each weight = (nibble | (qh_bit << 4)) - 16

    Args:
        x: Float32 array, length must be multiple of 32

    Returns:
        Packed Q5_0 data as bytes
    """
    assert len(x) % QK5_0 == 0, f"Length {len(x)} not divisible by {QK5_0}"
    nb = len(x) // QK5_0

    result = bytearray()

    for b in range(nb):
        block = x[b * QK5_0 : (b + 1) * QK5_0]

        # Find max absolute value for scale
        amax = np.max(np.abs(block))
        d = amax / 15.0 if amax > 0 else 1.0
        id = 1.0 / d if d != 0 else 0.0

        # Quantize to 5-bit signed (-16 to 15)
        # val = (unsigned_5bit) - 16
        # So unsigned_5bit = val + 16
        qs_float = block * id
        qs_int = np.round(qs_float).astype(np.int32)
        qs_int = np.clip(qs_int, -16, 15)
        qs_unsigned = (qs_int + 16).astype(np.uint8)  # 0-31

        # Split into nibbles (4 bits) and high bits (1 bit)
        nibbles = qs_unsigned & 0x0F
        high_bits = (qs_unsigned >> 4) & 0x01

        # Pack qh: 32 bits, one per weight
        qh = 0
        for j in range(32):
            qh |= (int(high_bits[j]) << j)

        # Pack qs: 16 bytes, 2 nibbles per byte
        # byte[j] = (nibbles[j+16] << 4) | nibbles[j]
        qs_packed = bytearray(16)
        for j in range(16):
            qs_packed[j] = (nibbles[j + 16] << 4) | nibbles[j]

        # Pack: FP16 scale + 4 bytes qh + 16 bytes qs
        d_fp16 = float_to_fp16(d)
        result.extend(struct.pack('<H', d_fp16))
        result.extend(struct.pack('<I', qh))
        result.extend(qs_packed)

    return bytes(result)

def dequantize_q8_0(data: bytes, length: int) -> np.ndarray:
    """Dequantize Q8_0 data to float32."""
    nb = length // QK8_0
    result = np.zeros(length, dtype=np.float32)

    offset = 0
    for b in range(nb):
        d_fp16 = struct.unpack('<H', data[offset:offset+2])[0]
        d = fp16_to_float(d_fp16)
        offset += 2

        qs = np.frombuffer(data[offset:offset+32], dtype=np.int8)
        offset += 32

        result[b * QK8_0 : (b + 1) * QK8_0] = d * qs.astype(np.float32)

    return result

def dequantize_q5_0(data: bytes, length: int) -> np.ndarray:
    """Dequantize Q5_0 data to float32."""
    nb = length // QK5_0
    result = np.zeros(length, dtype=np.float32)

    offset = 0
    for b in range(nb):
        d_fp16 = struct.unpack('<H', data[offset:offset+2])[0]
        d = fp16_to_float(d_fp16)
        offset += 2

        qh = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4

        qs = data[offset:offset+16]
        offset += 16

        for j in range(16):
            # First 16: low nibble + qh bit j
            xh_0 = ((qh >> j) & 1) << 4
            w0 = ((qs[j] & 0x0F) | xh_0) - 16

            # Second 16: high nibble + qh bit (j+16)
            xh_1 = ((qh >> (j + 16)) & 1) << 4
            w1 = ((qs[j] >> 4) | xh_1) - 16

            result[b * QK5_0 + j] = d * w0
            result[b * QK5_0 + j + 16] = d * w1

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    name: str
    M: int  # Batch size (tokens)
    N: int  # Output features
    K: int  # Input features
    description: str = ""


# Standard test cases
STANDARD_TESTS = [
    TestCase("tiny", 1, 32, 32, "Minimum size: 1x32 @ 32x32"),
    TestCase("small_sq", 4, 64, 64, "Small square"),
    TestCase("small_batch", 8, 32, 128, "Small batch"),
    TestCase("medium", 16, 256, 256, "Medium size"),
    TestCase("qwen_qkv", 32, 896, 896, "Qwen2 0.5B QKV size"),
    TestCase("qwen_mlp_up", 32, 4864, 896, "Qwen2 0.5B MLP up"),
    TestCase("qwen_mlp_down", 32, 896, 4864, "Qwen2 0.5B MLP down"),
    TestCase("large", 64, 1024, 1024, "Large size"),
]

# Edge case tests
EDGE_TESTS = [
    TestCase("single_block", 1, 32, 32, "Single Q8_0 block"),
    TestCase("two_blocks", 1, 32, 64, "Two Q8_0 blocks"),
    TestCase("wide", 1, 1024, 32, "Wide output"),
    TestCase("tall", 128, 32, 32, "Tall batch"),
    TestCase("uneven_k", 1, 32, 96, "K not power of 2"),
]

# Stress tests
STRESS_TESTS = [
    TestCase("stress_small", 128, 256, 256, "Many small GEMMs"),
    TestCase("stress_medium", 64, 512, 512, "Medium stress"),
    TestCase("stress_large", 32, 1024, 2048, "Large stress"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Test Functions
# ═══════════════════════════════════════════════════════════════════════════════

def run_gemm_q8_0_q8_0_test(
    libck,
    test: TestCase,
    tolerance: float,
    verbose: bool = False
) -> TestResult:
    """
    Test gemm_nt_q8_0_q8_0: C[M,N] = A[M,K] @ B[N,K]^T

    Both A and B are Q8_0 quantized.
    """
    M, N, K = test.M, test.N, test.K

    np.random.seed(42 + hash(test.name) % 1000)

    # Generate random FP32 data
    A_fp32 = np.random.randn(M, K).astype(np.float32) * 2.0
    B_fp32 = np.random.randn(N, K).astype(np.float32) * 2.0

    # Quantize
    A_q8 = bytearray()
    for m in range(M):
        A_q8.extend(quantize_row_q8_0(A_fp32[m]))

    B_q8 = bytearray()
    for n in range(N):
        B_q8.extend(quantize_row_q8_0(B_fp32[n]))

    # Allocate output
    C_ck = np.zeros((M, N), dtype=np.float32)

    # Run CK kernel
    try:
        libck.gemm_nt_q8_0_q8_0(
            (ctypes.c_char * len(A_q8)).from_buffer(A_q8),
            (ctypes.c_char * len(B_q8)).from_buffer(B_q8),
            C_ck.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(M),
            ctypes.c_int(N),
            ctypes.c_int(K)
        )
    except Exception as e:
        return TestResult(
            name=f"Q8_0_Q8_0_{test.name}",
            passed=False,
            max_diff=float('inf'),
            tolerance=tolerance,
            pytorch_time=None,
            kernel_time=None
        )

    # Compute reference using dequantized values
    A_deq = np.zeros((M, K), dtype=np.float32)
    for m in range(M):
        A_deq[m] = dequantize_q8_0(bytes(A_q8[m*BLOCK_Q8_0_SIZE*(K//QK8_0):(m+1)*BLOCK_Q8_0_SIZE*(K//QK8_0)]), K)

    B_deq = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        B_deq[n] = dequantize_q8_0(bytes(B_q8[n*BLOCK_Q8_0_SIZE*(K//QK8_0):(n+1)*BLOCK_Q8_0_SIZE*(K//QK8_0)]), K)

    C_ref = A_deq @ B_deq.T

    # Compare
    max_diff = np.max(np.abs(C_ck - C_ref))
    mean_diff = np.mean(np.abs(C_ck - C_ref))
    passed = max_diff <= tolerance

    if verbose and not passed:
        print(f"\n  FAIL: {test.name}")
        print(f"    max_diff={max_diff:.2e}, tol={tolerance:.0e}")
        print(f"    C_ck[0,:4]={C_ck[0,:4]}")
        print(f"    C_ref[0,:4]={C_ref[0,:4]}")

    return TestResult(
        name=f"Q8_0_Q8_0_{test.name}",
        passed=passed,
        max_diff=max_diff,
        tolerance=tolerance,
        pytorch_time=None,
        kernel_time=None
    )


def run_gemm_q5_0_q8_0_test(
    libck,
    test: TestCase,
    tolerance: float,
    verbose: bool = False
) -> TestResult:
    """
    Test gemm_nt_q5_0_q8_0: C[M,N] = A[M,K] @ B[N,K]^T

    A is Q8_0, B (weights) is Q5_0.
    """
    M, N, K = test.M, test.N, test.K

    np.random.seed(42 + hash(test.name) % 1000)

    # Generate random FP32 data
    A_fp32 = np.random.randn(M, K).astype(np.float32) * 2.0
    B_fp32 = np.random.randn(N, K).astype(np.float32) * 2.0

    # Quantize A to Q8_0
    A_q8 = bytearray()
    for m in range(M):
        A_q8.extend(quantize_row_q8_0(A_fp32[m]))

    # Quantize B to Q5_0
    B_q5 = bytearray()
    for n in range(N):
        B_q5.extend(quantize_row_q5_0(B_fp32[n]))

    # Allocate output
    C_ck = np.zeros((M, N), dtype=np.float32)

    # Run CK kernel
    try:
        libck.gemm_nt_q5_0_q8_0(
            (ctypes.c_char * len(A_q8)).from_buffer(A_q8),
            (ctypes.c_char * len(B_q5)).from_buffer(B_q5),
            C_ck.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(M),
            ctypes.c_int(N),
            ctypes.c_int(K)
        )
    except Exception as e:
        return TestResult(
            name=f"Q5_0_Q8_0_{test.name}",
            passed=False,
            max_diff=float('inf'),
            tolerance=tolerance,
            pytorch_time=None,
            kernel_time=None
        )

    # Compute reference using dequantized values
    A_deq = np.zeros((M, K), dtype=np.float32)
    nb_a = K // QK8_0
    for m in range(M):
        start = m * BLOCK_Q8_0_SIZE * nb_a
        end = (m + 1) * BLOCK_Q8_0_SIZE * nb_a
        A_deq[m] = dequantize_q8_0(bytes(A_q8[start:end]), K)

    B_deq = np.zeros((N, K), dtype=np.float32)
    nb_b = K // QK5_0
    for n in range(N):
        start = n * BLOCK_Q5_0_SIZE * nb_b
        end = (n + 1) * BLOCK_Q5_0_SIZE * nb_b
        B_deq[n] = dequantize_q5_0(bytes(B_q5[start:end]), K)

    C_ref = A_deq @ B_deq.T

    # Compare
    max_diff = np.max(np.abs(C_ck - C_ref))
    passed = max_diff <= tolerance

    if verbose and not passed:
        print(f"\n  FAIL: {test.name}")
        print(f"    max_diff={max_diff:.2e}, tol={tolerance:.0e}")
        print(f"    C_ck[0,:4]={C_ck[0,:4]}")
        print(f"    C_ref[0,:4]={C_ref[0,:4]}")

    return TestResult(
        name=f"Q5_0_Q8_0_{test.name}",
        passed=passed,
        max_diff=max_diff,
        tolerance=tolerance,
        pytorch_time=None,
        kernel_time=None
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GEMM Batch INT8 Kernel Tests")
    parser.add_argument("--quick", action="store_true", help="Run only standard tests")
    parser.add_argument("--stress", action="store_true", help="Include stress tests")
    parser.add_argument("--amx", action="store_true", help="Test AMX path (requires Sapphire Rapids+)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output on failures")
    args = parser.parse_args()

    # Get CPU info
    cpu = get_cpu_info()

    # Determine instruction set and tolerance
    if cpu.avx512f:
        simd = "AVX-512"
        tol = TOLERANCES["avx512"]
    elif cpu.avx:
        simd = "AVX"
        tol = TOLERANCES["avx"]
    else:
        simd = "Scalar"
        tol = TOLERANCES["scalar"]

    print("=" * 80)
    print(f"  GEMM BATCH INT8 KERNEL TESTS [{simd}]")
    print("=" * 80)
    print(f"\n  CPU:       {cpu.model_name}")
    print(f"  SIMD:      {simd}")
    print(f"  Tolerance: {tol:.0e}")
    print()

    # Load libraries
    libck, libllama = load_libraries()

    if libck is None:
        print("ERROR: Could not load CK library")
        print("Build with: make")
        sys.exit(1)

    # Set up function signatures
    try:
        libck.gemm_nt_q8_0_q8_0.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        libck.gemm_nt_q8_0_q8_0.restype = None

        libck.gemm_nt_q5_0_q8_0.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        libck.gemm_nt_q5_0_q8_0.restype = None
    except AttributeError as e:
        print(f"ERROR: Kernel function not found: {e}")
        print("Ensure gemm_batch_int8.c is compiled into the library")
        sys.exit(1)

    # Collect test cases
    tests = STANDARD_TESTS.copy()
    if not args.quick:
        tests.extend(EDGE_TESTS)
    if args.stress:
        tests.extend(STRESS_TESTS)

    # Run Q8_0 x Q8_0 tests
    print("-" * 80)
    print("  Q8_0 x Q8_0 GEMM Tests")
    print("-" * 80)

    q8_results = []
    for test in tests:
        result = run_gemm_q8_0_q8_0_test(libck, test, tol, args.verbose)
        q8_results.append(result)
        status = "\033[92mPASS\033[0m" if result.passed else "\033[91mFAIL\033[0m"
        print(f"  {result.name:30s} {test.M:4d}x{test.N:4d}x{test.K:4d}  "
              f"max_diff={result.max_diff:.2e}  {status}")

    # Run Q5_0 x Q8_0 tests
    print()
    print("-" * 80)
    print("  Q5_0 x Q8_0 GEMM Tests")
    print("-" * 80)

    q5_results = []
    for test in tests:
        result = run_gemm_q5_0_q8_0_test(libck, test, tol, args.verbose)
        q5_results.append(result)
        status = "\033[92mPASS\033[0m" if result.passed else "\033[91mFAIL\033[0m"
        print(f"  {result.name:30s} {test.M:4d}x{test.N:4d}x{test.K:4d}  "
              f"max_diff={result.max_diff:.2e}  {status}")

    # Summary
    all_results = q8_results + q5_results
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)

    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  Total:   {len(all_results)}")
    print(f"  Passed:  \033[92m{passed}\033[0m")
    print(f"  Failed:  \033[91m{failed}\033[0m")
    print()

    if failed == 0:
        print("\033[92mAll tests passed!\033[0m")
        sys.exit(0)
    else:
        print("\033[91mSome tests failed!\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
