"""
Cross-Entropy Loss kernel unit tests with performance metrics.

Tests combined forward+backward (fused softmax + cross-entropy) against PyTorch reference.
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

lib.softmax_cross_entropy_loss.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # logits
    ctypes.POINTER(ctypes.c_int32),  # targets
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # vocab_size
    ctypes.POINTER(ctypes.c_float),  # d_logits
    ctypes.POINTER(ctypes.c_float),  # loss_out
]
lib.softmax_cross_entropy_loss.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def ptr_int32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def run_c_cross_entropy(logits_np: np.ndarray, targets_np: np.ndarray):
    """Run CK CE kernel and return (loss, d_logits)."""
    t, v = logits_np.shape
    dlogits_np = np.zeros_like(logits_np, dtype=np.float32)
    loss_c = ctypes.c_float(0.0)
    lib.softmax_cross_entropy_loss(
        numpy_to_ptr(logits_np),
        ptr_int32(targets_np),
        ctypes.c_int(t), ctypes.c_int(v),
        numpy_to_ptr(dlogits_np),
        ctypes.byref(loss_c),
    )
    return float(loss_c.value), dlogits_np


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests(T=128, V=1000, warmup=10, iterations=1000):
    """Run cross-entropy tests with accuracy and timing."""
    np.random.seed(0)

    # Pre-allocate numpy arrays
    logits_np = np.random.randn(T, V).astype(np.float32)
    targets_np = np.random.randint(0, V, (T,), dtype=np.int32)
    # Torch tensors
    logits = torch.from_numpy(logits_np.copy())
    targets = torch.from_numpy(targets_np).long()

    report = TestReport(
        test_name="Cross-Entropy Loss",
        dtype="fp32",
        shape=f"T={T}, V={V}",
        cpu_info=get_cpu_info()
    )

    # PyTorch forward only
    def pytorch_forward():
        return F.cross_entropy(logits, targets, reduction="mean")

    # PyTorch forward+backward
    def pytorch_fwd_bwd():
        log = logits.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(log, targets, reduction="mean")
        loss.backward()
        return loss, log.grad

    # Get reference
    loss_ref, dlogits_ref = pytorch_fwd_bwd()

    # C kernel (fused forward + backward)
    def c_cross_entropy():
        run_c_cross_entropy(logits_np, targets_np)

    # Run once for accuracy
    loss_c_val, dlogits_np = run_c_cross_entropy(logits_np, targets_np)
    loss_diff = abs(loss_c_val - float(loss_ref.detach()))
    dlogits_c = torch.from_numpy(dlogits_np.copy())
    grad_diff = max_diff(dlogits_c, dlogits_ref)

    # Timing
    pt_fwd_time = time_function(pytorch_forward, warmup=warmup, iterations=iterations, name="PyTorch Fwd")
    pt_fwd_bwd_time = time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch Fwd+Bwd")
    c_time = time_function(c_cross_entropy, warmup=warmup, iterations=iterations, name="C Fused")

    pt_bwd_est = pt_fwd_bwd_time.mean_us - pt_fwd_time.mean_us

    report.add_result(TestResult(
        name="Loss value",
        passed=loss_diff <= 1e-5,
        max_diff=loss_diff,
        tolerance=1e-5,
        pytorch_time=pt_fwd_bwd_time,
        kernel_time=c_time
    ))

    report.add_result(TestResult(
        name="d_logits",
        passed=grad_diff <= 1e-5,
        max_diff=grad_diff,
        tolerance=1e-5,
        pytorch_time=None,
        kernel_time=None
    ))

    # Store timing data
    report.timing_breakdown = {
        'pt_fwd': pt_fwd_time.mean_us,
        'pt_bwd_est': pt_bwd_est,
        'pt_fwd_bwd': pt_fwd_bwd_time.mean_us,
        'c_fused': c_time.mean_us,
    }

    return report


def run_ignore_index_semantics_tests(T=96, V=503, warmup=10, iterations=500):
    """Validate PyTorch ignore_index mean-reduction semantics."""
    np.random.seed(1234)

    logits_np = np.random.randn(T, V).astype(np.float32)
    targets_np = np.random.randint(0, V, (T,), dtype=np.int32)
    ignore_mask = np.random.rand(T) < 0.2
    targets_np[ignore_mask] = -100

    logits = torch.from_numpy(logits_np.copy()).requires_grad_(True)
    targets = torch.from_numpy(targets_np.astype(np.int64))
    loss_ref = F.cross_entropy(logits, targets, reduction="mean", ignore_index=-100)
    loss_ref.backward()
    dlogits_ref = logits.grad

    loss_c_val, dlogits_c_np = run_c_cross_entropy(logits_np, targets_np)
    dlogits_c = torch.from_numpy(dlogits_c_np.copy())

    report = TestReport(
        test_name="Cross-Entropy ignore_index semantics",
        dtype="fp32",
        shape=f"T={T}, V={V}",
        cpu_info=get_cpu_info(),
    )

    loss_diff = abs(loss_c_val - float(loss_ref.detach()))
    grad_diff = max_diff(dlogits_c, dlogits_ref)
    report.add_result(TestResult(
        name="Loss (ignore_index=-100)",
        passed=loss_diff <= 1e-5,
        max_diff=loss_diff,
        tolerance=1e-5,
        pytorch_time=time_function(
            lambda: F.cross_entropy(logits.detach(), targets, reduction="mean", ignore_index=-100),
            warmup=warmup,
            iterations=iterations,
            name="PyTorch Fwd",
        ),
        kernel_time=time_function(
            lambda: run_c_cross_entropy(logits_np, targets_np),
            warmup=warmup,
            iterations=iterations,
            name="C Kernel",
        ),
    ))
    report.add_result(TestResult(
        name="d_logits (ignore_index=-100)",
        passed=grad_diff <= 1e-5,
        max_diff=grad_diff,
        tolerance=1e-5,
        pytorch_time=None,
        kernel_time=None,
    ))

    # All targets ignored: PyTorch mean reduction returns NaN and zero gradients.
    targets_all_ignored_np = np.full((T,), -100, dtype=np.int32)
    loss_c_all, dlogits_c_all_np = run_c_cross_entropy(logits_np, targets_all_ignored_np)

    logits_all = torch.from_numpy(logits_np.copy()).requires_grad_(True)
    targets_all = torch.from_numpy(targets_all_ignored_np.astype(np.int64))
    loss_ref_all = F.cross_entropy(logits_all, targets_all, reduction="mean", ignore_index=-100)
    loss_ref_all.backward()

    both_nan = bool(np.isnan(loss_c_all) and torch.isnan(loss_ref_all).item())
    all_grad_diff = max_diff(torch.from_numpy(dlogits_c_all_np.copy()), logits_all.grad)
    report.add_result(TestResult(
        name="Loss (all ignored => NaN)",
        passed=both_nan,
        max_diff=0.0 if both_nan else float("inf"),
        tolerance=0.0,
        pytorch_time=None,
        kernel_time=None,
    ))
    report.add_result(TestResult(
        name="d_logits (all ignored => zeros)",
        passed=all_grad_diff <= 1e-7,
        max_diff=all_grad_diff,
        tolerance=1e-7,
        pytorch_time=None,
        kernel_time=None,
    ))

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_system_info()

    # Run tests with different sizes
    small_report = run_tests(T=64, V=500, warmup=10, iterations=1000)
    small_report.print_report()

    large_report = run_tests(T=128, V=1000, warmup=10, iterations=500)
    large_report.print_report()

    ignore_report = run_ignore_index_semantics_tests(T=96, V=503, warmup=10, iterations=300)
    ignore_report.print_report()

    # Print detailed timing breakdown for the larger test
    if hasattr(large_report, 'timing_breakdown'):
        t = large_report.timing_breakdown
        print("  DETAILED TIMING BREAKDOWN (T=128, V=1000)")
        print("  " + "-" * 60)
        print(f"  {'Operation':<20} {'PyTorch (us)':<15} {'C Kernel (us)':<15} {'Speedup':<10}")
        print("  " + "-" * 60)
        print(f"  {'Forward only':<20} {t['pt_fwd']:<15.1f} {'-':<15} {'-':<10}")
        print(f"  {'Backward (est)':<20} {t['pt_bwd_est']:<15.1f} {'-':<15} {'-':<10}")
        print(f"  {'Fused Fwd+Bwd':<20} {t['pt_fwd_bwd']:<15.1f} {t['c_fused']:<15.1f} {t['pt_fwd_bwd']/t['c_fused']:.2f}x")
        print("  " + "-" * 60)
        print()

    # Exit with error if any tests failed
    if not small_report.all_passed() or not large_report.all_passed() or not ignore_report.all_passed():
        exit(1)
