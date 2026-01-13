"""
Causal Softmax kernel unit tests with performance metrics.

Tests forward and backward passes against PyTorch reference.
Reports accuracy, timing, and system information.
"""
import ctypes

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


# Load the library
lib = load_lib("libckernel_engine.so")

# ═══════════════════════════════════════════════════════════════════════════════
# Function signatures
# ═══════════════════════════════════════════════════════════════════════════════

lib.causal_softmax_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,  # num_heads
    ctypes.c_int,  # num_tokens
    ctypes.c_int,  # aligned_context_window
]
lib.causal_softmax_head_major.restype = None

# Exact version using standard library expf (slower but accurate)
lib.causal_softmax_head_major_exact.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,  # num_heads
    ctypes.c_int,  # num_tokens
    ctypes.c_int,  # aligned_context_window
]
lib.causal_softmax_head_major_exact.restype = None

lib.backward_causal_softmax_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_scores (in/out)
    ctypes.POINTER(ctypes.c_float),  # weights
    ctypes.c_int,  # num_heads
    ctypes.c_int,  # num_tokens
    ctypes.c_int,  # aligned_context_window
]
lib.backward_causal_softmax_head_major.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Reference implementations
# ═══════════════════════════════════════════════════════════════════════════════

def softmax_causal_reference(scores: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for causal softmax."""
    H, T, _ = scores.shape
    ref = scores.clone()
    for h in range(H):
        for i in range(T):
            row = ref[h, i, : i + 1]
            ref[h, i, : i + 1] = F.softmax(row, dim=-1)
            ref[h, i, i + 1 :] = 0.0
    return ref


def backward_causal_softmax_ref(dY: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for causal softmax backward."""
    H, T, _ = Y.shape
    dX = torch.zeros_like(Y)
    for h in range(H):
        for i in range(T):
            y_row = Y[h, i, : i + 1]
            dy_row = dY[h, i, : i + 1]
            dot = (y_row * dy_row).sum()
            dX[h, i, : i + 1] = y_row * (dy_row - dot)
            dX[h, i, i + 1 :] = 0.0
    return dX


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_forward_tests(H=8, T=64, warmup=10, iterations=1000):
    """Run forward pass tests with accuracy and timing."""
    np.random.seed(0)

    # Pre-allocate numpy arrays
    scores_np = np.random.randn(H, T, T).astype(np.float32)
    out_fast_np = scores_np.copy()
    out_exact_np = scores_np.copy()

    # Get pointers
    out_fast_ptr = numpy_to_ptr(out_fast_np)
    out_exact_ptr = numpy_to_ptr(out_exact_np)

    # Torch tensor
    scores = torch.from_numpy(scores_np.copy())

    report = TestReport(
        test_name="Causal Softmax Forward",
        dtype="fp32",
        shape=f"heads={H}, tokens={T}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    ref = softmax_causal_reference(scores)

    # Time PyTorch (using vectorized mask approach for fair comparison)
    def pytorch_causal_softmax():
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        s = scores.clone()
        s.masked_fill_(mask, float('-inf'))
        return F.softmax(s, dim=-1)

    pytorch_time = time_function(
        pytorch_causal_softmax,
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # === Fast version (SIMD, uses exp512_approx) ===
    def c_softmax_fast():
        np.copyto(out_fast_np, scores_np)
        lib.causal_softmax_head_major(out_fast_ptr, H, T, T)

    c_softmax_fast()
    out_fast = torch.from_numpy(out_fast_np.copy())
    diff_fast = max_diff(out_fast, ref)
    kernel_time_fast = time_function(c_softmax_fast, warmup=warmup, iterations=iterations, name="C Fast")

    # Fast version: trades accuracy for speedup on AVX-512
    report.add_result(TestResult(
        name="Fast (SIMD approx)",
        passed=diff_fast <= 2e-2,
        max_diff=diff_fast,
        tolerance=3e-2,  # Relaxed for SIMD exp approximation
        pytorch_time=pytorch_time,
        kernel_time=kernel_time_fast
    ))

    # === Exact version (scalar, uses standard library expf) ===
    def c_softmax_exact():
        np.copyto(out_exact_np, scores_np)
        lib.causal_softmax_head_major_exact(out_exact_ptr, H, T, T)

    c_softmax_exact()
    out_exact = torch.from_numpy(out_exact_np.copy())
    diff_exact = max_diff(out_exact, ref)
    kernel_time_exact = time_function(c_softmax_exact, warmup=warmup, iterations=iterations, name="C Exact")

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


def run_backward_tests(H=8, T=64, warmup=10, iterations=1000):
    """Run backward pass tests with accuracy and timing."""
    np.random.seed(1)

    # Pre-allocate numpy arrays
    scores_np = np.random.randn(H, T, T).astype(np.float32)
    weights_fast_np = np.zeros((H, T, T), dtype=np.float32)
    weights_exact_np = np.zeros((H, T, T), dtype=np.float32)
    dY_np = np.random.randn(H, T, T).astype(np.float32)
    dX_fast_np = dY_np.copy()
    dX_exact_np = dY_np.copy()

    # Get pointers
    weights_fast_ptr = numpy_to_ptr(weights_fast_np)
    weights_exact_ptr = numpy_to_ptr(weights_exact_np)
    dX_fast_ptr = numpy_to_ptr(dX_fast_np)
    dX_exact_ptr = numpy_to_ptr(dX_exact_np)

    # Torch tensors
    scores = torch.from_numpy(scores_np.copy())
    dY = torch.from_numpy(dY_np.copy())

    report = TestReport(
        test_name="Causal Softmax Backward",
        dtype="fp32",
        shape=f"heads={H}, tokens={T}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference (exact)
    ref_Y = softmax_causal_reference(scores)
    dX_ref = backward_causal_softmax_ref(dY, ref_Y)

    # PyTorch forward only
    def pytorch_forward():
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        s = scores.clone()
        s.masked_fill_(mask, float('-inf'))
        return F.softmax(s, dim=-1)

    # PyTorch forward+backward
    def pytorch_fwd_bwd():
        s = scores.clone().requires_grad_(True)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        s_masked = s.masked_fill(mask, float('-inf'))
        y = F.softmax(s_masked, dim=-1)
        y.backward(dY)
        return s.grad

    # === Fast version (SIMD forward + backward) ===
    np.copyto(weights_fast_np, scores_np)
    lib.causal_softmax_head_major(weights_fast_ptr, H, T, T)

    def c_fwd_bwd_fast():
        np.copyto(weights_fast_np, scores_np)
        lib.causal_softmax_head_major(weights_fast_ptr, H, T, T)
        np.copyto(dX_fast_np, dY_np)
        lib.backward_causal_softmax_head_major(dX_fast_ptr, weights_fast_ptr, H, T, T)

    # Run once for accuracy
    np.copyto(dX_fast_np, dY_np)
    lib.backward_causal_softmax_head_major(dX_fast_ptr, weights_fast_ptr, H, T, T)
    dX_fast = torch.from_numpy(dX_fast_np.copy())
    diff_fast = max_diff(dX_fast, dX_ref)

    # === Exact version (scalar forward + backward) ===
    np.copyto(weights_exact_np, scores_np)
    lib.causal_softmax_head_major_exact(weights_exact_ptr, H, T, T)

    def c_fwd_bwd_exact():
        np.copyto(weights_exact_np, scores_np)
        lib.causal_softmax_head_major_exact(weights_exact_ptr, H, T, T)
        np.copyto(dX_exact_np, dY_np)
        lib.backward_causal_softmax_head_major(dX_exact_ptr, weights_exact_ptr, H, T, T)

    # Run once for accuracy
    np.copyto(dX_exact_np, dY_np)
    lib.backward_causal_softmax_head_major(dX_exact_ptr, weights_exact_ptr, H, T, T)
    dX_exact = torch.from_numpy(dX_exact_np.copy())
    diff_exact = max_diff(dX_exact, dX_ref)

    # Timing
    pt_fwd_time = time_function(pytorch_forward, warmup=warmup, iterations=iterations, name="PyTorch Fwd")
    pt_fwd_bwd_time = time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch Fwd+Bwd")
    c_fwd_bwd_fast_time = time_function(c_fwd_bwd_fast, warmup=warmup, iterations=iterations, name="C Fast")
    c_fwd_bwd_exact_time = time_function(c_fwd_bwd_exact, warmup=warmup, iterations=iterations, name="C Exact")

    pt_bwd_est = pt_fwd_bwd_time.mean_us - pt_fwd_time.mean_us

    # Fast version: uses exp512_approx in forward, relaxed tolerance
    report.add_result(TestResult(
        name="Fast (SIMD fwd+bwd)",
        passed=diff_fast <= 5e-2,
        max_diff=diff_fast,
        tolerance=5e-2,
        pytorch_time=pt_fwd_bwd_time,
        kernel_time=c_fwd_bwd_fast_time
    ))

    # Exact version: uses expf in forward, strict tolerance
    report.add_result(TestResult(
        name="Exact (scalar fwd+bwd)",
        passed=diff_exact <= 1e-5,
        max_diff=diff_exact,
        tolerance=1e-5,
        pytorch_time=None,
        kernel_time=c_fwd_bwd_exact_time
    ))

    # Store timing data
    report.timing_breakdown = {
        'pt_fwd': pt_fwd_time.mean_us,
        'pt_bwd_est': pt_bwd_est,
        'pt_fwd_bwd': pt_fwd_bwd_time.mean_us,
        'c_fast': c_fwd_bwd_fast_time.mean_us,
        'c_exact': c_fwd_bwd_exact_time.mean_us,
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_system_info()

    # Forward tests
    fwd_report = run_forward_tests(H=8, T=64, warmup=10, iterations=1000)
    fwd_report.print_report()

    # Backward tests
    bwd_report = run_backward_tests(H=8, T=64, warmup=10, iterations=1000)
    bwd_report.print_report()

    # Print detailed timing breakdown
    if hasattr(bwd_report, 'timing_breakdown'):
        t = bwd_report.timing_breakdown
        print("  DETAILED TIMING BREAKDOWN (Forward+Backward variants)")
        print("  " + "-" * 68)
        print(f"  {'Operation':<25} {'PyTorch (us)':<15} {'C Kernel (us)':<15} {'Speedup':<10}")
        print("  " + "-" * 68)
        print(f"  {'PyTorch Fwd+Bwd':<25} {t['pt_fwd_bwd']:<15.1f} {'-':<15} {'-':<10}")
        print(f"  {'C Fast (SIMD)':<25} {t['pt_fwd_bwd']:<15.1f} {t['c_fast']:<15.1f} {t['pt_fwd_bwd']/t['c_fast']:.2f}x")
        print(f"  {'C Exact (scalar)':<25} {t['pt_fwd_bwd']:<15.1f} {t['c_exact']:<15.1f} {t['pt_fwd_bwd']/t['c_exact']:.2f}x")
        print("  " + "-" * 68)
        print()

    # Exit with error if any tests failed
    if not fwd_report.all_passed() or not bwd_report.all_passed():
        exit(1)
