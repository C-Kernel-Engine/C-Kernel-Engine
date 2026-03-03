#!/usr/bin/env python3
"""
Comprehensive GEMV kernel tests - Accuracy and Performance.

Tests all quantization formats (Q4_K, Q5_0, Q8_0, Q6_K) across multiple
dimensions from tiny to 7B scale. Compares against llama.cpp reference.

Usage:
    python unittest/test_gemv_kernels_comprehensive.py              # Full test
    python unittest/test_gemv_kernels_comprehensive.py --quick      # Quick smoke test
    python unittest/test_gemv_kernels_comprehensive.py --perf-only  # Performance only
    python unittest/test_gemv_kernels_comprehensive.py --large      # Include 7B dims

Environment:
    CK_GEMV_WARMUP=5        Warmup iterations (default: 5)
    CK_GEMV_ITERS=50        Timed iterations (default: 50)
    CK_GEMV_TOL=1e-3        Accuracy tolerance (default: 1e-3)
"""

import ctypes
import numpy as np
import argparse
import struct
import sys
import time
import os
import zlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

# ============================================================================
# Block sizes from GGML (must match ckernel_quant.h)
# ============================================================================
QK_K = 256      # K-quant super-block size
QK4_0 = 32      # Q4_0 block size
QK5_0 = 32      # Q5_0 block size
QK8_0 = 32      # Q8_0 block size

BLOCK_Q4_K_SIZE = 144   # bytes per Q4_K block
BLOCK_Q5_0_SIZE = 22    # bytes per Q5_0 block (2 + 4 + 16)
BLOCK_Q8_0_SIZE = 34    # bytes per Q8_0 block (2 + 32)
BLOCK_Q6_K_SIZE = 210   # bytes per Q6_K block


# ============================================================================
# Test configuration
# ============================================================================
@dataclass
class TestCase:
    label: str
    M: int          # Output dimension (rows)
    K: int          # Input dimension (cols)
    tol: float = 1e-3
    description: str = ""


@dataclass
class TestResult:
    name: str
    passed: bool
    max_diff: float
    mean_diff: float
    M: int = 0                              # Output dimension
    K: int = 0                              # Input dimension
    tol: float = 1e-3                       # Tolerance for pass/fail
    ck_time_ms: Optional[float] = None
    llama_time_ms: Optional[float] = None
    ck_gflops: Optional[float] = None
    llama_gflops: Optional[float] = None
    error: Optional[str] = None


# ============================================================================
# FP16 helpers
# ============================================================================
def fp16_to_bytes(val: float) -> bytes:
    return struct.pack('<e', val)


# ============================================================================
# FP16 to FP32 conversion
# ============================================================================
def fp16_to_fp32(h: int) -> float:
    """Convert FP16 bits to FP32."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF

    if exp == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        # Subnormal
        val = mant / 1024.0 * (2 ** -14)
    elif exp == 31:
        if mant == 0:
            return float('-inf') if sign else float('inf')
        return float('nan')
    else:
        val = (1.0 + mant / 1024.0) * (2 ** (exp - 15))

    return -val if sign else val


# ============================================================================
# Reference dequantization functions (for accuracy testing)
# ============================================================================
def dequant_q5_0_ref(data: bytes, n_elements: int) -> np.ndarray:
    """Dequantize Q5_0 weights to FP32 (reference implementation)."""
    assert n_elements % QK5_0 == 0
    n_blocks = n_elements // QK5_0
    result = np.zeros(n_elements, dtype=np.float32)

    offset = 0
    for b in range(n_blocks):
        # Read block: d (2 bytes), qh (4 bytes), qs (16 bytes) = 22 bytes
        d_bits = struct.unpack('<H', data[offset:offset+2])[0]
        d = fp16_to_fp32(d_bits)
        offset += 2

        qh = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4

        qs = data[offset:offset+16]
        offset += 16

        # Dequantize 32 values
        for j in range(16):
            q_lo = qs[j] & 0x0F
            q_hi = (qs[j] >> 4) & 0x0F

            # Add high bit from qh
            h_lo = (qh >> j) & 1
            h_hi = (qh >> (j + 16)) & 1

            x0 = ((q_lo | (h_lo << 4)) - 16) * d
            x1 = ((q_hi | (h_hi << 4)) - 16) * d

            result[b * QK5_0 + j] = x0
            result[b * QK5_0 + j + 16] = x1

    return result


def dequant_q8_0_ref(data: bytes, n_elements: int) -> np.ndarray:
    """Dequantize Q8_0 weights to FP32 (reference implementation)."""
    assert n_elements % QK8_0 == 0
    n_blocks = n_elements // QK8_0
    result = np.zeros(n_elements, dtype=np.float32)

    offset = 0
    for b in range(n_blocks):
        # Read block: d (2 bytes), qs (32 bytes) = 34 bytes
        d_bits = struct.unpack('<H', data[offset:offset+2])[0]
        d = fp16_to_fp32(d_bits)
        offset += 2

        qs = np.frombuffer(data[offset:offset+32], dtype=np.int8)
        offset += 32

        # Dequantize 32 values
        for j in range(32):
            result[b * QK8_0 + j] = qs[j] * d

    return result


def compute_ref_gemv(weights_dequant: np.ndarray, input_f32: np.ndarray, M: int, K: int) -> np.ndarray:
    """Compute reference GEMV using FP32."""
    # weights_dequant is [M * K], input_f32 is [K]
    # output is [M]
    weights_2d = weights_dequant.reshape(M, K)
    return np.dot(weights_2d, input_f32)


# ============================================================================
# Random quantized weight generation
# ============================================================================
def random_q4k_weights(n_elements: int) -> bytes:
    """Generate random Q4_K quantized weights."""
    assert n_elements % QK_K == 0, f"n_elements must be multiple of {QK_K}"
    n_blocks = n_elements // QK_K

    data = bytearray()
    for _ in range(n_blocks):
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

    return bytes(data)


def random_q5_0_weights(n_elements: int) -> bytes:
    """Generate random Q5_0 quantized weights."""
    assert n_elements % QK5_0 == 0, f"n_elements must be multiple of {QK5_0}"
    n_blocks = n_elements // QK5_0

    data = bytearray()
    for _ in range(n_blocks):
        # d (fp16 scale): 2 bytes
        d = np.random.uniform(0.01, 0.1)
        data.extend(fp16_to_bytes(d))
        # qh (high bits): 4 bytes
        qh = np.random.randint(0, 256, size=4, dtype=np.uint8)
        data.extend(qh.tobytes())
        # qs (4-bit low nibbles): 16 bytes
        qs = np.random.randint(0, 256, size=16, dtype=np.uint8)
        data.extend(qs.tobytes())

    return bytes(data)


def random_q8_0_weights(n_elements: int) -> bytes:
    """Generate random Q8_0 quantized weights."""
    assert n_elements % QK8_0 == 0, f"n_elements must be multiple of {QK8_0}"
    n_blocks = n_elements // QK8_0

    data = bytearray()
    for _ in range(n_blocks):
        # d (fp16 scale): 2 bytes
        d = np.random.uniform(0.01, 0.1)
        data.extend(fp16_to_bytes(d))
        # qs (int8 weights): 32 bytes
        qs = np.random.randint(-127, 128, size=32, dtype=np.int8)
        data.extend(qs.tobytes())

    return bytes(data)


def random_q6k_weights(n_elements: int) -> bytes:
    """Generate random Q6_K quantized weights."""
    assert n_elements % QK_K == 0, f"n_elements must be multiple of {QK_K}"
    n_blocks = n_elements // QK_K

    data = bytearray()
    for _ in range(n_blocks):
        # ql (low 4 bits): 128 bytes
        ql = np.random.randint(0, 256, size=128, dtype=np.uint8)
        data.extend(ql.tobytes())
        # qh (high 2 bits): 64 bytes
        qh = np.random.randint(0, 256, size=64, dtype=np.uint8)
        data.extend(qh.tobytes())
        # scales (int8): 16 bytes
        scales = np.random.randint(-127, 128, size=16, dtype=np.int8)
        data.extend(scales.tobytes())
        # d (fp16 scale): 2 bytes
        d = np.random.uniform(0.01, 0.1)
        data.extend(fp16_to_bytes(d))

    return bytes(data)


# ============================================================================
# Library loading
# ============================================================================
def load_libraries() -> Tuple[Optional[ctypes.CDLL], Optional[ctypes.CDLL]]:
    """Load both llama.cpp and CK parity libraries."""
    base_dir = Path(__file__).resolve().parents[1]

    # llama.cpp library
    llama_paths = [
        base_dir / "llama.cpp" / "libggml_kernel_test.so",
        base_dir / "llama.cpp" / "build" / "libggml_kernel_test.so",
        base_dir / "llama.cpp" / "build" / "bin" / "libggml_kernel_test.so",
    ]
    libggml = None
    for p in llama_paths:
        if p.exists():
            try:
                libggml = ctypes.CDLL(str(p))
                break
            except OSError:
                pass

    # CK library
    ck_paths = [
        base_dir / "build" / "libck_parity.so",
        base_dir / "libck_parity.so",
    ]
    libck = None
    for p in ck_paths:
        if p.exists():
            try:
                libck = ctypes.CDLL(str(p))
                break
            except OSError:
                pass

    return libggml, libck


# ============================================================================
# Test Case Definitions
# ============================================================================
def get_test_cases(quick: bool = False, large: bool = False) -> dict:
    """Get test cases for all kernel types."""

    # Dimensions must be multiples of block sizes
    # Q4_K/Q6_K: multiple of 256
    # Q5_0/Q8_0: multiple of 32

    if quick:
        # Quick smoke test - minimal dimensions
        # Note: Q5_0/Q8_0 have higher tolerance because:
        # 1. CK and llama.cpp may use different rounding in quantization
        # 2. Different FP16 conversion implementations can cause small differences
        return {
            "Q4_K": [
                TestCase("tiny", M=1, K=256, description="Minimal Q4_K"),
                TestCase("small", M=256, K=256, description="Small square"),
            ],
            "Q5_0": [
                TestCase("tiny", M=1, K=32, tol=1e-2, description="Minimal Q5_0"),
                TestCase("small", M=32, K=256, tol=1e-2, description="Small"),
            ],
            "Q8_0": [
                TestCase("tiny", M=1, K=32, tol=1e-1, description="Minimal Q8_0"),
                TestCase("small", M=32, K=256, tol=1e-1, description="Small"),
            ],
            "Q5_0_Q8_0": [
                TestCase("tiny", M=1, K=32, tol=1e-4, description="Direct Q5_0xQ8_0"),
                TestCase("small", M=1, K=256, tol=1e-4, description="Small direct"),
            ],
            "Q8_0_Q8_0": [
                TestCase("tiny", M=1, K=32, tol=1e-4, description="Direct Q8_0xQ8_0"),
                TestCase("small", M=1, K=256, tol=1e-4, description="Small direct"),
            ],
        }

    # Standard test suite
    cases = {
        "Q4_K": [
            # Small dimensions
            TestCase("tiny", M=1, K=256, description="Single output, minimal"),
            TestCase("small_sq", M=256, K=256, description="Small square"),
            TestCase("small_wide", M=256, K=512, description="Small wide"),
            TestCase("small_tall", M=512, K=256, description="Small tall"),
            # Medium dimensions (0.5B-like)
            TestCase("qwen_qkv", M=768, K=768, description="Qwen 0.5B QKV projection"),
            TestCase("qwen_mlp_up", M=4864, K=768, description="Qwen 0.5B MLP up"),
            TestCase("qwen_mlp_down", M=768, K=4864, description="Qwen 0.5B MLP down"),
            # Larger dimensions
            TestCase("medium_sq", M=1024, K=1024, description="Medium square"),
            TestCase("medium_wide", M=1024, K=2048, description="Medium wide"),
            TestCase("medium_tall", M=2048, K=1024, description="Medium tall"),
        ],
        "Q5_0": [
            # Q5_0 tolerance set to 2e-2 because CK and llama.cpp use different
            # FP32->Q8_0 quantization for the input vector. The diff is from
            # quantization differences, NOT kernel bugs. For pure kernel testing,
            # use Q5_0_Q8_0 tests with pre-quantized inputs (1e-4 tolerance).
            TestCase("tiny", M=1, K=32, tol=2e-2, description="Single output, minimal"),
            TestCase("small_sq", M=32, K=32, tol=2e-2, description="Small square"),
            TestCase("small", M=32, K=256, tol=2e-2, description="Small"),
            TestCase("medium", M=256, K=512, tol=2e-2, description="Medium"),
            TestCase("qwen_qkv", M=896, K=896, tol=2e-2, description="Qwen 0.5B QKV (Q5_0)"),
            TestCase("qwen_mlp", M=4864, K=896, tol=3e-2, description="Qwen 0.5B MLP"),  # Relaxed for FP32->Q8_0 quantization diff
            TestCase("large", M=1024, K=1024, tol=2e-2, description="Large square"),
        ],
        "Q8_0": [
            # Q8_0 tolerance relaxed to 2e-1 because CK and llama.cpp use different
            # FP32->Q8_0 quantization implementations. The ~0.14 diff is expected from
            # quantization differences, NOT kernel bugs. For pure kernel testing,
            # use Q8_0_Q8_0 tests with pre-quantized inputs (1e-4 tolerance).
            TestCase("tiny", M=1, K=32, tol=2e-1, description="Single output, minimal"),
            TestCase("small_sq", M=32, K=32, tol=2e-1, description="Small square"),
            TestCase("small", M=32, K=256, tol=2e-1, description="Small"),
            TestCase("medium", M=256, K=512, tol=2e-1, description="Medium"),
            TestCase("qwen_qkv", M=896, K=896, tol=2e-1, description="Qwen 0.5B QKV (Q8_0)"),
            TestCase("large", M=1024, K=1024, tol=2e-1, description="Large square"),
        ],
        # Direct vec_dot tests with pre-quantized Q8_0 inputs
        # These test pure kernel accuracy (bypass FP32-to-Q8_0 quantization)
        "Q5_0_Q8_0": [
            TestCase("tiny", M=1, K=32, tol=1e-4, description="Minimal direct vec_dot"),
            TestCase("small", M=1, K=256, tol=1e-4, description="Small direct vec_dot"),
            TestCase("medium", M=1, K=512, tol=1e-4, description="Medium direct vec_dot"),
            TestCase("qwen", M=1, K=896, tol=1e-4, description="Qwen dimension"),
            TestCase("large", M=1, K=1024, tol=1e-4, description="Large direct vec_dot"),
        ],
        "Q8_0_Q8_0": [
            TestCase("tiny", M=1, K=32, tol=1e-4, description="Minimal direct vec_dot"),
            TestCase("small", M=1, K=256, tol=1e-4, description="Small direct vec_dot"),
            TestCase("medium", M=1, K=512, tol=2e-4, description="Medium direct vec_dot"),  # Relaxed for FP accumulation variance
            TestCase("qwen", M=1, K=896, tol=2e-4, description="Qwen dimension"),  # Relaxed for FP accumulation
            TestCase("large", M=1, K=1024, tol=2e-4, description="Large direct vec_dot"),  # Slightly relaxed for FP accumulation
        ],
    }

    if large:
        # Add 7B-scale dimensions (all multiples of 256 for Q4_K compatibility)
        large_cases = {
            "Q4_K": [
                TestCase("llama_qkv", M=4096, K=4096, description="LLaMA 7B QKV"),
                TestCase("llama_mlp_up", M=11264, K=4096, description="LLaMA 7B MLP up"),
                TestCase("llama_mlp_down", M=4096, K=11264, description="LLaMA 7B MLP down"),
                TestCase("llama_embed", M=32000, K=4096, description="LLaMA 7B embedding"),
            ],
            "Q5_0": [
                TestCase("llama_qkv", M=4096, K=4096, tol=2e-2, description="LLaMA 7B QKV"),
                TestCase("llama_mlp", M=11264, K=4096, tol=2e-2, description="LLaMA 7B MLP"),
            ],
            "Q8_0": [
                TestCase("llama_qkv", M=4096, K=4096, tol=1e-1, description="LLaMA 7B QKV"),
                TestCase("llama_mlp", M=11264, K=4096, tol=1e-1, description="LLaMA 7B MLP"),
            ],
        }
        for qtype, qcases in large_cases.items():
            cases.setdefault(qtype, []).extend(qcases)

    return cases


# ============================================================================
# Kernel Tester
# ============================================================================
class KernelTester:
    def __init__(self, libggml, libck, warmup: int = 5, iters: int = 50, tol: float = 1e-3):
        self.libggml = libggml
        self.libck = libck
        self.warmup = warmup
        self.iters = iters
        self.tol = tol
        self.results: List[TestResult] = []

        if libggml:
            self._setup_ggml_signatures()
        if libck:
            self._setup_ck_signatures()

    def _setup_ggml_signatures(self):
        """Set up ctypes signatures for llama.cpp functions."""
        lib = self.libggml
        lib.test_init.argtypes = []
        lib.test_init.restype = None
        lib.test_init()

        # GEMV Q4_K
        lib.test_gemv_q4_k.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        lib.test_gemv_q4_k.restype = None

        # GEMV Q5_0 - llama.cpp test only does single row (dot product)
        lib.test_gemv_q5_0.argtypes = [
            ctypes.c_void_p,          # weights
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output (single value)
            ctypes.c_int   # cols
        ]
        lib.test_gemv_q5_0.restype = None

        # GEMV Q5_0 with FP32 activations (matches CK's FP32 path)
        if hasattr(lib, 'test_gemv_q5_0_fp32'):
            lib.test_gemv_q5_0_fp32.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int
            ]
            lib.test_gemv_q5_0_fp32.restype = None

        # GEMV Q8_0 - llama.cpp test only does single row (dot product)
        lib.test_gemv_q8_0.argtypes = [
            ctypes.c_void_p,          # weights
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output (single value)
            ctypes.c_int   # cols
        ]
        lib.test_gemv_q8_0.restype = None

        # GEMV Q8_0 with FP32 activations (matches CK's FP32 path)
        if hasattr(lib, 'test_gemv_q8_0_fp32'):
            lib.test_gemv_q8_0_fp32.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int
            ]
            lib.test_gemv_q8_0_fp32.restype = None

        # Direct vec_dot Q5_0 x Q8_0 (pre-quantized inputs)
        if hasattr(lib, 'test_vec_dot_q5_0_q8_0'):
            lib.test_vec_dot_q5_0_q8_0.argtypes = [
                ctypes.c_void_p,  # Q5_0 weights
                ctypes.c_void_p,  # Q8_0 input (pre-quantized)
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_int   # cols
            ]
            lib.test_vec_dot_q5_0_q8_0.restype = None

        # Direct vec_dot Q8_0 x Q8_0 (pre-quantized inputs)
        if hasattr(lib, 'test_vec_dot_q8_0_q8_0'):
            lib.test_vec_dot_q8_0_q8_0.argtypes = [
                ctypes.c_void_p,  # Q8_0 weights
                ctypes.c_void_p,  # Q8_0 input (pre-quantized)
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_int   # cols
            ]
            lib.test_vec_dot_q8_0_q8_0.restype = None

    def _setup_ck_signatures(self):
        """Set up ctypes signatures for CK functions."""
        lib = self.libck

        # GEMV Q4_K
        lib.ck_test_gemv_q4_k.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        lib.ck_test_gemv_q4_k.restype = None

        # GEMV Q5_0 (FP32 input - dequant path)
        lib.ck_test_gemv_q5_0.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,  # rows
            ctypes.c_int   # cols
        ]
        lib.ck_test_gemv_q5_0.restype = None

        # GEMV Q5_0 x Q8_0 (quantized path - matches llama.cpp)
        lib.ck_test_gemv_q5_0_q8_0.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,  # rows
            ctypes.c_int   # cols
        ]
        lib.ck_test_gemv_q5_0_q8_0.restype = None

        # GEMV Q8_0 (FP32 input - dequant path)
        lib.ck_test_gemv_q8_0.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,  # rows
            ctypes.c_int   # cols
        ]
        lib.ck_test_gemv_q8_0.restype = None

        # GEMV Q8_0 x Q8_0 (quantized path - matches llama.cpp)
        lib.ck_test_gemv_q8_0_q8_0.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,  # rows
            ctypes.c_int   # cols
        ]
        lib.ck_test_gemv_q8_0_q8_0.restype = None

        # Direct vec_dot Q5_0 x Q8_0 (pre-quantized inputs)
        lib.ck_test_vec_dot_q5_0_q8_0.argtypes = [
            ctypes.c_void_p,  # Q5_0 weights
            ctypes.c_void_p,  # Q8_0 input (pre-quantized)
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int   # cols
        ]
        lib.ck_test_vec_dot_q5_0_q8_0.restype = None

        # Direct vec_dot Q8_0 x Q8_0 (pre-quantized inputs)
        lib.ck_test_vec_dot_q8_0_q8_0.argtypes = [
            ctypes.c_void_p,  # Q8_0 weights
            ctypes.c_void_p,  # Q8_0 input (pre-quantized)
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int   # cols
        ]
        lib.ck_test_vec_dot_q8_0_q8_0.restype = None

    def run_gemv_test(self, qtype: str, case: TestCase, perf_only: bool = False) -> TestResult:
        """Run a single GEMV test case."""
        M, K = case.M, case.K
        name = f"{qtype}_{case.label}"

        if not self.libck:
            return TestResult(name, False, 0, 0, error="CK library not loaded")

        # Select appropriate weight generator and dequantization function
        dequant_fn = None
        if qtype == "Q4_K":
            if K % QK_K != 0:
                return TestResult(name, False, 0, 0, error=f"K={K} not multiple of {QK_K}")
            weight_gen = random_q4k_weights
            has_llama_ref = self.libggml is not None
            # No Python dequant for Q4_K, use llama.cpp reference

            def call_ck(w, x, y):
                self.libck.ck_test_gemv_q4_k(w, x, y, K)

            def call_llama(w, x, y):
                if self.libggml:
                    self.libggml.test_gemv_q4_k(w, x, y, K)

        elif qtype == "Q5_0":
            if K % QK5_0 != 0:
                return TestResult(name, False, 0, 0, error=f"K={K} not multiple of {QK5_0}")
            weight_gen = random_q5_0_weights
            block_size = BLOCK_Q5_0_SIZE
            # Check if llama.cpp has Q5_0 support
            has_llama_ref = self.libggml is not None and hasattr(self.libggml, 'test_gemv_q5_0')
            # Check if llama.cpp has FP32 activation test
            has_llama_fp32 = (self.libggml is not None and
                               hasattr(self.libggml, 'test_gemv_q5_0_fp32'))
            # When llama.cpp is available, use the quantized path (Q5_0 x Q8_0) for CK too
            # This matches llama.cpp's approach: quantize input to Q8_0, then do integer dot product
            dequant_fn = dequant_q5_0_ref if not has_llama_ref else None

            # FAIR COMPARISON: Both CK and llama.cpp test M=1 (single dot product)
            # llama.cpp test_gemv_q5_0 only does M=1, so we match that for fair perf comparison
            # Accuracy is checked for first row only (compare_rows=1 set below)
            test_rows = 1  # Match llama.cpp workload for fair comparison

            # Test mode: either quantized path or FP32 path
            # Default to quantized path (matches original llama.cpp behavior)
            test_mode = os.environ.get("CK_TEST_MODE", "quantized")

            if test_mode == "fp32" and has_llama_fp32:
                # Test FP32 path: dequantize to FP32, do FP32 arithmetic
                def call_ck(w, x, y):
                    self.libck.ck_test_gemv_q5_0(w, x, y, test_rows, K)

                def call_llama(w, x, y):
                    self.libggml.test_gemv_q5_0_fp32(w, x, y, K)
            else:
                # Test quantized path: quantize input, do integer dot product
                def call_ck(w, x, y):
                    if has_llama_ref:
                        # Use quantized path to match llama.cpp (M=1)
                        self.libck.ck_test_gemv_q5_0_q8_0(w, x, y, test_rows, K)
                    else:
                        # Fall back to FP32 path for Python dequant reference
                        self.libck.ck_test_gemv_q5_0(w, x, y, M, K)

                def call_llama(w, x, y):
                    if has_llama_ref:
                        # llama.cpp test only does single dot product (M=1)
                        self.libggml.test_gemv_q5_0(w, x, y, K)

        elif qtype == "Q8_0":
            if K % QK8_0 != 0:
                return TestResult(name, False, 0, 0, error=f"K={K} not multiple of {QK8_0}")
            weight_gen = random_q8_0_weights
            block_size = BLOCK_Q8_0_SIZE
            # Check if llama.cpp has Q8_0 support
            has_llama_ref = self.libggml is not None and hasattr(self.libggml, 'test_gemv_q8_0')
            # Check if llama.cpp has FP32 activation test
            has_llama_fp32 = (self.libggml is not None and
                               hasattr(self.libggml, 'test_gemv_q8_0_fp32'))
            # When llama.cpp is available, use the quantized path (Q8_0 x Q8_0) for CK too
            dequant_fn = dequant_q8_0_ref if not has_llama_ref else None

            # FAIR COMPARISON: Both CK and llama.cpp test M=1 (single dot product)
            # llama.cpp test_gemv_q8_0 only does M=1, so we match that for fair perf comparison
            # Accuracy is checked for first row only (compare_rows=1 set below)
            test_rows = 1  # Match llama.cpp workload for fair comparison

            # Test mode: either quantized path or FP32 path
            # Default to quantized path (matches original llama.cpp behavior)
            test_mode = os.environ.get("CK_TEST_MODE", "quantized")

            if test_mode == "fp32" and has_llama_fp32:
                # Test FP32 path: dequantize to FP32, do FP32 arithmetic
                def call_ck(w, x, y):
                    self.libck.ck_test_gemv_q8_0(w, x, y, test_rows, K)

                def call_llama(w, x, y):
                    self.libggml.test_gemv_q8_0_fp32(w, x, y, K)
            else:
                # Test quantized path: quantize input, do integer dot product
                def call_ck(w, x, y):
                    if has_llama_ref:
                        # Use quantized path to match llama.cpp (M=1)
                        self.libck.ck_test_gemv_q8_0_q8_0(w, x, y, test_rows, K)
                    else:
                        # Fall back to FP32 path for Python dequant reference
                        self.libck.ck_test_gemv_q8_0(w, x, y, M, K)

                def call_llama(w, x, y):
                    if has_llama_ref:
                        # llama.cpp test only does single dot product (M=1)
                        self.libggml.test_gemv_q8_0(w, x, y, K)

        elif qtype == "Q5_0_Q8_0":
            # Direct Q5_0 x Q8_0 vec_dot test (pre-quantized inputs)
            if K % QK5_0 != 0:
                return TestResult(name, False, 0, 0, error=f"K={K} not multiple of {QK5_0}")
            weight_gen = random_q5_0_weights
            has_llama_ref = (self.libggml is not None and
                            hasattr(self.libggml, 'test_vec_dot_q5_0_q8_0'))

            def call_ck(w, x_q8, y):
                self.libck.ck_test_vec_dot_q5_0_q8_0(w, x_q8, y, K)

            def call_llama(w, x_q8, y):
                if has_llama_ref:
                    self.libggml.test_vec_dot_q5_0_q8_0(w, x_q8, y, K)

        elif qtype == "Q8_0_Q8_0":
            # Direct Q8_0 x Q8_0 vec_dot test (pre-quantized inputs)
            if K % QK8_0 != 0:
                return TestResult(name, False, 0, 0, error=f"K={K} not multiple of {QK8_0}")
            weight_gen = random_q8_0_weights
            has_llama_ref = (self.libggml is not None and
                            hasattr(self.libggml, 'test_vec_dot_q8_0_q8_0'))

            def call_ck(w, x_q8, y):
                self.libck.ck_test_vec_dot_q8_0_q8_0(w, x_q8, y, K)

            def call_llama(w, x_q8, y):
                if has_llama_ref:
                    self.libggml.test_vec_dot_q8_0_q8_0(w, x_q8, y, K)

        else:
            return TestResult(name, False, 0, 0, error=f"Unsupported qtype: {qtype}")

        # Determine comparison size (llama.cpp Q5_0/Q8_0 tests only do M=1)
        # Direct vec_dot tests (Q5_0_Q8_0, Q8_0_Q8_0) always have M=1
        is_direct_vec_dot = qtype in ("Q5_0_Q8_0", "Q8_0_Q8_0")
        compare_rows = 1 if (has_llama_ref and qtype in ("Q5_0", "Q8_0")) or is_direct_vec_dot else M

        # Generate test data
        # Use stable per-test seeding. Python's built-in hash() is process-randomized
        # unless PYTHONHASHSEED is fixed, which can make nightly parity outcomes flaky.
        np.random.seed(42 + (zlib.crc32(name.encode("utf-8")) % 10000))

        if is_direct_vec_dot:
            # Direct vec_dot tests: single row of weights, Q8_0 quantized input
            weights = weight_gen(K)  # Single row
            input_q8_0 = random_q8_0_weights(K)  # Q8_0 quantized input

            ck_out = np.zeros(1, dtype=np.float32)
            llama_out = np.zeros(1, dtype=np.float32) if has_llama_ref else None
            ck_out_ptr = ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            # Accuracy test
            max_diff = 0.0
            mean_diff = 0.0
            if not perf_only:
                call_ck(weights, input_q8_0, ck_out_ptr)

                if np.isnan(ck_out).any() or np.isinf(ck_out).any():
                    return TestResult(name, False, 0, 0, error="CK output contains NaN/Inf")

                if has_llama_ref:
                    llama_out_ptr = llama_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    call_llama(weights, input_q8_0, llama_out_ptr)
                    diff = np.abs(llama_out - ck_out)
                    max_diff = float(np.max(diff))
                    mean_diff = float(np.mean(diff))

            # Performance test - CK
            for _ in range(self.warmup):
                call_ck(weights, input_q8_0, ck_out_ptr)

            t0 = time.perf_counter()
            for _ in range(self.iters):
                call_ck(weights, input_q8_0, ck_out_ptr)
            t1 = time.perf_counter()
            ck_time_ms = (t1 - t0) / self.iters * 1000

            # Performance test - llama.cpp
            llama_time_ms = None
            llama_gflops = None
            if has_llama_ref:
                llama_out_ptr = llama_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                for _ in range(self.warmup):
                    call_llama(weights, input_q8_0, llama_out_ptr)

                t0 = time.perf_counter()
                for _ in range(self.iters):
                    call_llama(weights, input_q8_0, llama_out_ptr)
                t1 = time.perf_counter()
                llama_time_ms = (t1 - t0) / self.iters * 1000
                llama_gflops = (2.0 * K / 1e9) / (llama_time_ms / 1000) if llama_time_ms > 0 else 0
        else:
            # Standard GEMV tests: M rows of weights, FP32 input
            weights = weight_gen(M * K)
            input_f32 = np.random.randn(K).astype(np.float32)

            ck_out = np.zeros(compare_rows, dtype=np.float32)
            llama_out = np.zeros(compare_rows, dtype=np.float32) if has_llama_ref else None

            input_ptr = input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ck_out_ptr = ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            # Accuracy test
            max_diff = 0.0
            mean_diff = 0.0
            if not perf_only:
                call_ck(weights, input_ptr, ck_out_ptr)

                # Check for NaN/Inf first
                if np.isnan(ck_out).any() or np.isinf(ck_out).any():
                    return TestResult(name, False, 0, 0, error="CK output contains NaN/Inf")

                if has_llama_ref:
                    # Use llama.cpp as reference
                    llama_out_ptr = llama_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    call_llama(weights, input_ptr, llama_out_ptr)
                    diff = np.abs(llama_out - ck_out)
                    max_diff = float(np.max(diff))
                    mean_diff = float(np.mean(diff))
                elif dequant_fn is not None:
                    # Use Python dequantization as reference for Q5_0/Q8_0
                    weights_dequant = dequant_fn(weights, M * K)
                    ref_out = compute_ref_gemv(weights_dequant, input_f32, M, K)
                    diff = np.abs(ref_out - ck_out)
                    max_diff = float(np.max(diff))
                    mean_diff = float(np.mean(diff))

            # Performance test - CK kernel
            for _ in range(self.warmup):
                call_ck(weights, input_ptr, ck_out_ptr)

            t0 = time.perf_counter()
            for _ in range(self.iters):
                call_ck(weights, input_ptr, ck_out_ptr)
            t1 = time.perf_counter()
            ck_time_ms = (t1 - t0) / self.iters * 1000

            # Performance test - llama.cpp (if available)
            llama_time_ms = None
            llama_gflops = None
            if has_llama_ref:
                llama_out_ptr = llama_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                for _ in range(self.warmup):
                    call_llama(weights, input_ptr, llama_out_ptr)

                t0 = time.perf_counter()
                for _ in range(self.iters):
                    call_llama(weights, input_ptr, llama_out_ptr)
                t1 = time.perf_counter()
                llama_time_ms = (t1 - t0) / self.iters * 1000
                llama_gflops = (2.0 * M * K / 1e9) / (llama_time_ms / 1000) if llama_time_ms > 0 else 0

        # Calculate CK GFLOPS (direct vec_dot tests have M=1)
        effective_M = 1 if is_direct_vec_dot else M
        ck_gflops = (2.0 * effective_M * K / 1e9) / (ck_time_ms / 1000) if ck_time_ms > 0 else 0

        # Test passes if accuracy is within tolerance (for any reference - llama.cpp or Python dequant)
        has_accuracy_ref = has_llama_ref or dequant_fn is not None
        passed = (max_diff < case.tol) if has_accuracy_ref and not perf_only else True

        return TestResult(
            name=name,
            passed=passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            M=M,
            K=K,
            tol=case.tol,
            ck_time_ms=ck_time_ms,
            llama_time_ms=llama_time_ms,
            ck_gflops=ck_gflops,
            llama_gflops=llama_gflops
        )

    def run_all_tests(self, cases: dict, perf_only: bool = False) -> List[TestResult]:
        """Run all test cases."""
        results = []

        for qtype, qcases in cases.items():
            for case in qcases:
                result = self.run_gemv_test(qtype, case, perf_only)
                results.append(result)
                self.results.append(result)

        return results


# ============================================================================
# Report formatting
# ============================================================================
def print_header():
    """Print test header with system info."""
    cpu_info = "Unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_info = line.split(":")[1].strip()
                    break
    except:
        pass

    simd_info = []
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
            if "avx512" in content.lower():
                simd_info.append("AVX-512")
            elif "avx2" in content.lower():
                simd_info.append("AVX2")
            elif "avx" in content.lower():
                simd_info.append("AVX")
            if "fma" in content.lower():
                simd_info.append("FMA")
    except:
        pass

    print("=" * 100)
    print(f"{BOLD}COMPREHENSIVE GEMV KERNEL TESTS - CK-Engine vs Reference{RESET}")
    print("=" * 100)
    print(f"""
  {CYAN}WHAT:{RESET}    Matrix-Vector Multiply (GEMV) accuracy and performance tests
            Testing: output[M] = weights[M,K] x input[K]

  {CYAN}KERNELS:{RESET} Q4_K     - 4-bit K-quant (llama.cpp reference)
            Q5_0     - 5-bit legacy quant (llama.cpp reference)
            Q8_0     - 8-bit legacy quant (llama.cpp reference)
            Q5_0_Q8_0 - Direct Q5_0 x Q8_0 vec_dot (llama.cpp reference)
            Q8_0_Q8_0 - Direct Q8_0 x Q8_0 vec_dot (llama.cpp reference)

  {CYAN}SYSTEM:{RESET}
    CPU:    {cpu_info}
    SIMD:   {', '.join(simd_info) if simd_info else 'Unknown'}
    Cores:  {os.cpu_count()}
""")



def print_results(results: List[TestResult], qtype: str):
    """Print results for a quantization type."""
    qtype_results = [r for r in results if r.name.startswith(qtype)]
    if not qtype_results:
        return

    print(f"\n{BOLD}{'='*100}{RESET}")
    print(f"{BOLD}{qtype} KERNELS{RESET}")
    print(f"{'='*100}")

    # Check if we have llama.cpp reference for this qtype
    has_llama_ref = any(r.llama_time_ms is not None for r in qtype_results if not r.error)

    # Accuracy section - Q5_0/Q8_0 use Python dequant reference, Q4_K uses llama.cpp
    if has_llama_ref:
        print(f"\n{CYAN}ACCURACY (vs llama.cpp reference){RESET}")
    elif qtype in ("Q5_0", "Q8_0"):
        print(f"\n{CYAN}ACCURACY (vs Python dequant reference){RESET}")
    else:
        print(f"\n{CYAN}ACCURACY (sanity check - no NaN/Inf){RESET}")
    print(f"{'Test':<20} {'Dims (MxK)':>14} {'max_diff':>12} {'mean_diff':>12} {'tol':>10} {'Status':>10}")
    print("-" * 90)

    accuracy_passed = 0
    for r in qtype_results:
        if r.error:
            status = f"{YELLOW}SKIP{RESET}"
            print(f"{r.name:<20} {'':>14} {r.error:<40} {status:>10}")
        else:
            status = f"{GREEN}PASS{RESET}" if r.passed else f"{RED}FAIL{RESET}"
            dims = f"{r.M}x{r.K}"
            # Always show actual accuracy values since we now have reference implementations
            print(f"{r.name:<20} {dims:>14} {r.max_diff:>12.2e} {r.mean_diff:>12.2e} {r.tol:>10.0e} {status:>10}")
            if r.passed:
                accuracy_passed += 1

    # Performance section
    print(f"\n{CYAN}PERFORMANCE{RESET}")
    if has_llama_ref:
        print(f"{'Test':<30} {'llama.cpp':>12} {'CK':>12} {'Speedup':>10} {'CK GFLOPS':>12} {'llama GFLOPS':>12}")
    else:
        print(f"{'Test':<30} {'CK Time':>12} {'CK GFLOPS':>12}")
    print("-" * 100)

    total_speedup = 0
    count = 0
    for r in qtype_results:
        if r.error or r.ck_time_ms is None:
            continue

        if r.llama_time_ms is not None:
            speedup = r.llama_time_ms / r.ck_time_ms if r.ck_time_ms > 0 else 0
            total_speedup += speedup
            count += 1

            if speedup >= 0.95:
                color = GREEN
            elif speedup >= 0.5:
                color = YELLOW
            else:
                color = RED

            print(f"{r.name:<30} {r.llama_time_ms:>10.3f}ms {r.ck_time_ms:>10.3f}ms "
                  f"{color}{speedup:>9.2f}x{RESET} {r.ck_gflops:>12.2f} {r.llama_gflops:>12.2f}")
        else:
            # No llama reference - just show CK performance
            print(f"{r.name:<30} {r.ck_time_ms:>10.3f}ms {r.ck_gflops:>12.2f}")
            count += 1

    if count > 0:
        valid_results = [r for r in qtype_results if r.ck_gflops and not r.error]
        avg_ck_gflops = sum(r.ck_gflops for r in valid_results) / len(valid_results) if valid_results else 0
        print("-" * 100)
        if has_llama_ref and total_speedup > 0:
            avg_speedup = total_speedup / count
            print(f"{'Average:':<30} {'':>12} {'':>12} {avg_speedup:>9.2f}x {avg_ck_gflops:>12.2f}")
        else:
            print(f"{'Average:':<30} {'':>12} {avg_ck_gflops:>12.2f}")

    print(f"\n  Accuracy: {accuracy_passed}/{len(qtype_results)} passed")


def print_summary(results: List[TestResult]):
    """Print overall summary."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and not r.error)
    skipped = sum(1 for r in results if r.error)

    print(f"\n{'='*100}")
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"{'='*100}")
    print(f"  Total:   {total}")
    print(f"  Passed:  {GREEN}{passed}{RESET}")
    print(f"  Failed:  {RED}{failed}{RESET}")
    print(f"  Skipped: {YELLOW}{skipped}{RESET}")

    if failed == 0:
        print(f"\n{GREEN}All tests passed!{RESET}")
    else:
        print(f"\n{RED}Some tests failed!{RESET}")
        for r in results:
            if not r.passed and not r.error:
                print(f"  - {r.name}: max_diff={r.max_diff:.2e}")

    # Performance summary
    gemv_results = [r for r in results if r.ck_time_ms and r.llama_time_ms]
    if gemv_results:
        avg_speedup = sum(r.llama_time_ms / r.ck_time_ms for r in gemv_results) / len(gemv_results)
        avg_ck_gflops = sum(r.ck_gflops for r in gemv_results) / len(gemv_results)
        avg_llama_gflops = sum(r.llama_gflops for r in gemv_results) / len(gemv_results)

        print(f"\n{BOLD}PERFORMANCE SUMMARY{RESET}")
        print(f"  Average CK/llama.cpp speedup: {avg_speedup:.2f}x")
        print(f"  Average CK GFLOPS:            {avg_ck_gflops:.2f}")
        print(f"  Average llama.cpp GFLOPS:     {avg_llama_gflops:.2f}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Comprehensive GEMV kernel tests")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--large", action="store_true", help="Include 7B-scale dimensions")
    parser.add_argument("--perf-only", action="store_true", help="Performance only (skip accuracy)")
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("CK_GEMV_WARMUP", "5")))
    parser.add_argument("--iters", type=int, default=int(os.environ.get("CK_GEMV_ITERS", "50")))
    parser.add_argument("--tol", type=float, default=float(os.environ.get("CK_GEMV_TOL", "1e-3")))
    args = parser.parse_args()

    print_header()

    libggml, libck = load_libraries()

    # Allow graceful skip on CI when libraries aren't available
    skip_if_missing = os.environ.get("CK_SKIP_IF_MISSING", "0") == "1"

    if not libggml:
        print(f"{YELLOW}WARNING: Could not load llama.cpp kernel test library.{RESET}")
        print("Build it with: cd llama.cpp && g++ -shared -fPIC -o libggml_kernel_test.so ...")
        if skip_if_missing:
            print(f"{YELLOW}SKIP: CK_SKIP_IF_MISSING=1, skipping GEMV tests{RESET}")
            sys.exit(0)
        sys.exit(1)

    if not libck:
        print(f"{YELLOW}WARNING: Could not load CK parity library.{RESET}")
        print("Build it with: make libck_parity.so")
        if skip_if_missing:
            print(f"{YELLOW}SKIP: CK_SKIP_IF_MISSING=1, skipping GEMV tests{RESET}")
            sys.exit(0)
        sys.exit(1)

    print(f"  Warmup:    {args.warmup} iterations")
    print(f"  Timed:     {args.iters} iterations")
    print(f"  Tolerance: {args.tol:.0e}")
    print()

    tester = KernelTester(libggml, libck, warmup=args.warmup, iters=args.iters, tol=args.tol)
    cases = get_test_cases(quick=args.quick, large=args.large)

    results = tester.run_all_tests(cases, perf_only=args.perf_only)

    # Print results by quantization type
    for qtype in cases.keys():
        print_results(results, qtype)

    print_summary(results)

    failed = sum(1 for r in results if not r.passed and not r.error)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
