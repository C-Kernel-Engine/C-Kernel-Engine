"""
Attention kernel unit tests with performance metrics.

Tests causal attention forward pass against PyTorch reference.
Reports accuracy, timing, and system information.

Tests:
- Flash attention (SIMD optimized, O(N) memory)
- Score-matrix attention (for backward compatibility)
- Exact (scalar) version for accuracy reference
"""
import ctypes
import math

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


# Prefer the full engine build because it already carries the attention kernels
# and avoids standalone shared-library dependency drift.
lib = load_lib("libckernel_engine.so", "libckernel_attention.so")

# ===============================================================================
# Function signatures
# ===============================================================================

# Score-matrix version (scalar Q·K^T and W·V, SIMD softmax)
# Note: This version materializes the full score matrix for backward pass compatibility
lib.attention_forward_causal_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # scores
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
    ctypes.c_int,                    # aligned_context_window
]
lib.attention_forward_causal_head_major.restype = None

# Exact version (uses standard library expf)
lib.attention_forward_causal_head_major_exact.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # scores
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
    ctypes.c_int,                    # aligned_context_window
]
lib.attention_forward_causal_head_major_exact.restype = None

# Flash attention (SIMD optimized, online softmax, O(N) memory)
lib.attention_forward_causal_head_major_gqa_flash.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_kv_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
]
lib.attention_forward_causal_head_major_gqa_flash.restype = None


# ===============================================================================
# Reference implementation
# ===============================================================================

def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for causal attention (head-major layout)."""
    H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)

    # Compute scores: [H, T, T]
    scores = torch.bmm(q, k.transpose(-2, -1)) * scale

    # Apply causal mask
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))

    # Softmax
    weights = F.softmax(scores, dim=-1)

    # Output: [H, T, D]
    return torch.bmm(weights, v)


def attention_reference_loop(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Loop-based reference for exact numerical comparison."""
    H, T, D = q.shape
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(D)

    for h in range(H):
        for i in range(T):
            qi = q[h, i]
            kj = k[h, : i + 1, :]
            scores = torch.mv(kj, qi) * scale
            weights = torch.softmax(scores, dim=-1)
            vj = v[h, : i + 1, :]
            out[h, i, :] = torch.matmul(weights, vj)

    return out


# ===============================================================================
# Tests
# ===============================================================================

def run_forward_tests(H=8, T=64, D=64, warmup=10, iterations=500):
    """Run forward pass tests with accuracy and timing.

    Tests:
    - Flash attention (SIMD optimized, O(N) memory) - main performance benchmark
    - Score-matrix (for backward compatibility) - not SIMD optimized
    - Exact (scalar) - for accuracy reference
    """
    np.random.seed(0)

    # Pre-allocate numpy arrays
    q_np = np.random.randn(H, T, D).astype(np.float32)
    k_np = np.random.randn(H, T, D).astype(np.float32)
    v_np = np.random.randn(H, T, D).astype(np.float32)
    scores_np = np.zeros((H, T, T), dtype=np.float32)
    out_flash_np = np.zeros((H, T, D), dtype=np.float32)
    out_score_np = np.zeros((H, T, D), dtype=np.float32)
    out_exact_np = np.zeros((H, T, D), dtype=np.float32)

    # Get pointers
    q_ptr = numpy_to_ptr(q_np)
    k_ptr = numpy_to_ptr(k_np)
    v_ptr = numpy_to_ptr(v_np)
    scores_ptr = numpy_to_ptr(scores_np)
    out_flash_ptr = numpy_to_ptr(out_flash_np)
    out_score_ptr = numpy_to_ptr(out_score_np)
    out_exact_ptr = numpy_to_ptr(out_exact_np)

    # Torch tensors
    q = torch.from_numpy(q_np.copy())
    k = torch.from_numpy(k_np.copy())
    v = torch.from_numpy(v_np.copy())

    report = TestReport(
        test_name="Attention Forward (Causal)",
        dtype="fp32",
        shape=f"H={H}, T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference (vectorized)
    ref = attention_reference(q, k, v)

    # Time PyTorch
    pytorch_time = time_function(
        lambda: attention_reference(q, k, v),
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # === Flash attention (SIMD optimized, O(N) memory) ===
    def c_attention_flash():
        lib.attention_forward_causal_head_major_gqa_flash(
            q_ptr, k_ptr, v_ptr, out_flash_ptr,
            ctypes.c_int(H), ctypes.c_int(H),  # num_heads, num_kv_heads (same for non-GQA)
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D)
        )

    c_attention_flash()
    out_flash = torch.from_numpy(out_flash_np.copy())
    diff_flash = max_diff(out_flash, ref)
    kernel_time_flash = time_function(c_attention_flash, warmup=warmup, iterations=iterations, name="C Flash")

    # Flash version: SIMD optimized with online softmax
    report.add_result(TestResult(
        name="Flash (SIMD)",
        passed=diff_flash <= 1e-4,  # Tight tolerance - flash uses standard expf
        max_diff=diff_flash,
        tolerance=1e-4,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time_flash
    ))

    # === Score-matrix version (for backward pass compatibility) ===
    def c_attention_score():
        lib.attention_forward_causal_head_major(
            q_ptr, k_ptr, v_ptr, scores_ptr, out_score_ptr,
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(T)
        )

    c_attention_score()
    out_score = torch.from_numpy(out_score_np.copy())
    diff_score = max_diff(out_score, ref)
    kernel_time_score = time_function(c_attention_score, warmup=warmup, iterations=iterations, name="C Score-matrix")

    # Score-matrix version: materializes full score matrix, uses SIMD exp approximation
    report.add_result(TestResult(
        name="Score-matrix (exp approx)",
        passed=diff_score <= 0.1,  # 10% tolerance for exp approximation
        max_diff=diff_score,
        tolerance=0.1,
        pytorch_time=None,
        kernel_time=kernel_time_score
    ))

    # === Exact version (scalar using standard library expf) ===
    def c_attention_exact():
        lib.attention_forward_causal_head_major_exact(
            q_ptr, k_ptr, v_ptr, scores_ptr, out_exact_ptr,
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(T)
        )

    c_attention_exact()
    out_exact = torch.from_numpy(out_exact_np.copy())
    diff_exact = max_diff(out_exact, ref)
    kernel_time_exact = time_function(c_attention_exact, warmup=warmup, iterations=iterations, name="C Exact")

    # Exact version: full precision using standard library expf
    report.add_result(TestResult(
        name="Exact (scalar)",
        passed=diff_exact <= 1e-5,
        max_diff=diff_exact,
        tolerance=1e-5,
        pytorch_time=None,
        kernel_time=kernel_time_exact
    ))

    return report


def run_accuracy_tests():
    """Run small tests for numerical accuracy verification.

    Tests flash, score-matrix, and exact versions at various sizes.
    """
    np.random.seed(42)

    report = TestReport(
        test_name="Attention Accuracy (Small)",
        dtype="fp32",
        shape="Various small sizes",
        cpu_info=get_cpu_info()
    )

    test_configs = [
        (2, 8, 16, "Tiny"),
        (4, 16, 32, "Small"),
        (8, 32, 64, "Medium"),
    ]

    for H, T, D, name in test_configs:
        q_np = np.random.randn(H, T, D).astype(np.float32)
        k_np = np.random.randn(H, T, D).astype(np.float32)
        v_np = np.random.randn(H, T, D).astype(np.float32)
        scores_np = np.zeros((H, T, T), dtype=np.float32)
        out_flash_np = np.zeros((H, T, D), dtype=np.float32)
        out_score_np = np.zeros((H, T, D), dtype=np.float32)
        out_exact_np = np.zeros((H, T, D), dtype=np.float32)

        q_ptr = numpy_to_ptr(q_np)
        k_ptr = numpy_to_ptr(k_np)
        v_ptr = numpy_to_ptr(v_np)
        scores_ptr = numpy_to_ptr(scores_np)
        out_flash_ptr = numpy_to_ptr(out_flash_np)
        out_score_ptr = numpy_to_ptr(out_score_np)
        out_exact_ptr = numpy_to_ptr(out_exact_np)

        q = torch.from_numpy(q_np.copy())
        k = torch.from_numpy(k_np.copy())
        v = torch.from_numpy(v_np.copy())

        # Use loop reference for exact comparison
        ref = attention_reference_loop(q, k, v)

        # Flash version (SIMD optimized)
        lib.attention_forward_causal_head_major_gqa_flash(
            q_ptr, k_ptr, v_ptr, out_flash_ptr,
            ctypes.c_int(H), ctypes.c_int(H),  # num_heads, num_kv_heads
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D)
        )
        out_flash = torch.from_numpy(out_flash_np.copy())
        diff_flash = max_diff(out_flash, ref)

        report.add_result(TestResult(
            name=f"Flash {name} (H={H},T={T},D={D})",
            passed=diff_flash <= 1e-4,
            max_diff=diff_flash,
            tolerance=1e-4,
            pytorch_time=None,
            kernel_time=None
        ))

        # Score-matrix version (uses exp approximation)
        lib.attention_forward_causal_head_major(
            q_ptr, k_ptr, v_ptr, scores_ptr, out_score_ptr,
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(T)
        )
        out_score = torch.from_numpy(out_score_np.copy())
        diff_score = max_diff(out_score, ref)

        report.add_result(TestResult(
            name=f"Score-matrix {name} (H={H},T={T},D={D})",
            passed=diff_score <= 0.1,  # 10% tolerance for exp approximation
            max_diff=diff_score,
            tolerance=0.1,
            pytorch_time=None,
            kernel_time=None
        ))

        # Exact version (scalar)
        lib.attention_forward_causal_head_major_exact(
            q_ptr, k_ptr, v_ptr, scores_ptr, out_exact_ptr,
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(T)
        )
        out_exact = torch.from_numpy(out_exact_np.copy())
        diff_exact = max_diff(out_exact, ref)

        report.add_result(TestResult(
            name=f"Exact {name} (H={H},T={T},D={D})",
            passed=diff_exact <= 1e-5,
            max_diff=diff_exact,
            tolerance=1e-5,
            pytorch_time=None,
            kernel_time=None
        ))

    return report


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":
    print_system_info()

    # Accuracy tests (small sizes)
    acc_report = run_accuracy_tests()
    acc_report.print_report()

    # Performance tests (larger size)
    perf_report = run_forward_tests(H=8, T=64, D=64, warmup=10, iterations=500)
    perf_report.print_report()

    # Exit with error if any tests failed
    if not acc_report.all_passed() or not perf_report.all_passed():
        exit(1)
