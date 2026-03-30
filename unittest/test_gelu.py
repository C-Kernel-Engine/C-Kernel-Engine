"""
GELU kernel unit tests with performance metrics.

Tests forward and backward passes against PyTorch reference.
Reports accuracy, timing, and system information.
"""
import ctypes
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


# Load the library
lib = load_lib("libckernel_gelu.so", "libckernel_engine.so")

# ═══════════════════════════════════════════════════════════════════════════════
# Function signatures
# ═══════════════════════════════════════════════════════════════════════════════

lib.gelu_fast_inplace.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.gelu_fast_inplace.restype = None

# Exact version using standard library tanhf (slower but accurate)
lib.gelu_exact_inplace.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.gelu_exact_inplace.restype = None

lib.gelu_ggml_inplace.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.gelu_ggml_inplace.restype = None

lib.gelu_backward_exact.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.c_size_t,                 # n
]
lib.gelu_backward_exact.restype = None

lib.gelu_backward_fast.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.c_size_t,                 # n
]
lib.gelu_backward_fast.restype = None

# Scalar exact backward using standard library tanhf (slower but accurate)
lib.gelu_backward_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.c_size_t,                 # n
]
lib.gelu_backward_scalar.restype = None


def ggml_gelu_ref(x_arr):
    """Reference ggml fp16-table GELU using the local ggml table when available."""
    ggml_lib_path = Path(__file__).resolve().parents[1] / "llama.cpp" / "build" / "bin" / "libggml-cpu.so"
    try:
        ggml = ctypes.CDLL(str(ggml_lib_path))
        ggml.ggml_cpu_init()
        ggml.ggml_fp32_to_fp16.argtypes = [ctypes.c_float]
        ggml.ggml_fp32_to_fp16.restype = ctypes.c_uint16
        ggml.ggml_fp16_to_fp32.argtypes = [ctypes.c_uint16]
        ggml.ggml_fp16_to_fp32.restype = ctypes.c_float
        table_type = ctypes.c_uint16 * (1 << 16)
        table = table_type.in_dll(ggml, "ggml_table_gelu_f16")

        x = x_arr.astype(np.float32, copy=False)
        out = np.empty_like(x)
        for i, xv in enumerate(x):
            if xv <= -10.0:
                out[i] = 0.0
            elif xv >= 10.0:
                out[i] = xv
            else:
                bits = ggml.ggml_fp32_to_fp16(ctypes.c_float(float(xv)))
                out[i] = ggml.ggml_fp16_to_fp32(table[bits])
        return out
    except Exception:
        x = x_arr.astype(np.float32, copy=True)
        out = np.empty_like(x)
        lo = x <= -10.0
        hi = x >= 10.0
        mid = ~(lo | hi)
        out[lo] = 0.0
        out[hi] = x[hi]
        mid_x = x[mid].astype(np.float16).astype(np.float32)
        mid_y = F.gelu(torch.from_numpy(mid_x), approximate="tanh").numpy()
        out[mid] = mid_y.astype(np.float16).astype(np.float32)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_forward_tests(N=4096, warmup=10, iterations=1000):
    """Run forward pass tests with accuracy and timing."""
    np.random.seed(0)

    # Pre-allocate numpy arrays
    x_np = np.random.randn(N).astype(np.float32)
    out_fast_np = x_np.copy()
    out_exact_np = x_np.copy()
    out_ggml_np = x_np.copy()

    # Get pointers
    out_fast_ptr = numpy_to_ptr(out_fast_np)
    out_exact_ptr = numpy_to_ptr(out_exact_np)
    out_ggml_ptr = numpy_to_ptr(out_ggml_np)

    # Torch tensor for PyTorch comparison
    x = torch.from_numpy(x_np.copy())

    report = TestReport(
        test_name="GELU Forward",
        dtype="fp32",
        shape=f"N={N}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    ref = F.gelu(x, approximate="tanh")

    # Time PyTorch
    pytorch_time = time_function(
        lambda: F.gelu(x, approximate="tanh"),
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # === Fast version (vectorized, uses tanh512_fast approximation) ===
    def c_gelu_fast():
        np.copyto(out_fast_np, x_np)
        lib.gelu_fast_inplace(out_fast_ptr, ctypes.c_size_t(N))

    c_gelu_fast()
    out_fast = torch.from_numpy(out_fast_np.copy())
    diff_fast = max_diff(out_fast, ref)
    kernel_time_fast = time_function(c_gelu_fast, warmup=warmup, iterations=iterations, name="C GELU Fast")

    # Fast version: trades accuracy for ~2.8x speedup on AVX-512
    report.add_result(TestResult(
        name="Fast (SIMD approx)",
        passed=diff_fast <= 2e-2,
        max_diff=diff_fast,
        tolerance=2e-2,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time_fast
    ))

    # === Exact version (scalar, uses standard library tanhf) ===
    def c_gelu_exact():
        np.copyto(out_exact_np, x_np)
        lib.gelu_exact_inplace(out_exact_ptr, ctypes.c_size_t(N))

    c_gelu_exact()
    out_exact = torch.from_numpy(out_exact_np.copy())
    diff_exact = max_diff(out_exact, ref)
    kernel_time_exact = time_function(c_gelu_exact, warmup=warmup, iterations=iterations, name="C GELU Exact")

    # Exact version: full precision using standard library tanhf
    report.add_result(TestResult(
        name="Exact (scalar)",
        passed=diff_exact <= 1e-6,
        max_diff=diff_exact,
        tolerance=1e-6,
        pytorch_time=None,
        kernel_time=kernel_time_exact
    ))

    ggml_ref = torch.from_numpy(ggml_gelu_ref(x_np))

    def c_gelu_ggml():
        np.copyto(out_ggml_np, x_np)
        lib.gelu_ggml_inplace(out_ggml_ptr, ctypes.c_size_t(N))

    c_gelu_ggml()
    out_ggml = torch.from_numpy(out_ggml_np.copy())
    diff_ggml = max_diff(out_ggml, ggml_ref)
    kernel_time_ggml = time_function(c_gelu_ggml, warmup=warmup, iterations=iterations, name="C GELU GGML")

    report.add_result(TestResult(
        name="GGML (fp16 table)",
        passed=diff_ggml <= 1e-6,
        max_diff=diff_ggml,
        tolerance=1e-6,
        pytorch_time=None,
        kernel_time=kernel_time_ggml
    ))

    return report


def run_backward_tests(N=4096, warmup=10, iterations=1000):
    """Run backward pass tests with accuracy and timing."""
    np.random.seed(1)

    # Pre-allocate numpy arrays
    x_np = np.random.randn(N).astype(np.float32)
    upstream_np = np.random.randn(N).astype(np.float32)
    dx_simd_np = np.zeros(N, dtype=np.float32)
    dx_scalar_np = np.zeros(N, dtype=np.float32)
    dx_fast_np = np.zeros(N, dtype=np.float32)

    # Get pointers
    x_ptr = numpy_to_ptr(x_np)
    up_ptr = numpy_to_ptr(upstream_np)
    dx_simd_ptr = numpy_to_ptr(dx_simd_np)
    dx_scalar_ptr = numpy_to_ptr(dx_scalar_np)
    dx_fast_ptr = numpy_to_ptr(dx_fast_np)

    # Torch tensors
    x = torch.from_numpy(x_np.copy())
    upstream = torch.from_numpy(upstream_np)

    report = TestReport(
        test_name="GELU Backward",
        dtype="fp32",
        shape=f"N={N}",
        cpu_info=get_cpu_info()
    )

    # PyTorch forward only
    def pytorch_forward():
        return F.gelu(x, approximate="tanh")

    # PyTorch forward+backward
    def pytorch_fwd_bwd():
        x_ref = x.clone().detach().requires_grad_(True)
        y = F.gelu(x_ref, approximate="tanh")
        y.backward(upstream)
        return x_ref.grad

    # Get reference grad
    dx_ref = pytorch_fwd_bwd()

    # === SIMD version (vectorized, uses tanh512_fast approximation) ===
    def c_backward_simd():
        lib.gelu_backward_exact(x_ptr, up_ptr, dx_simd_ptr, ctypes.c_size_t(N))

    c_backward_simd()
    dx_simd = torch.from_numpy(dx_simd_np.copy())
    diff_simd = max_diff(dx_simd, dx_ref)

    # === Scalar exact version (uses standard library tanhf) ===
    def c_backward_scalar():
        lib.gelu_backward_scalar(x_ptr, up_ptr, dx_scalar_ptr, ctypes.c_size_t(N))

    c_backward_scalar()
    dx_scalar = torch.from_numpy(dx_scalar_np.copy())
    diff_scalar = max_diff(dx_scalar, dx_ref)

    # === Fast version (sigmoid approximation) ===
    def c_backward_fast():
        lib.gelu_backward_fast(x_ptr, up_ptr, dx_fast_ptr, ctypes.c_size_t(N))

    c_backward_fast()
    dx_fast = torch.from_numpy(dx_fast_np.copy())
    diff_fast = max_diff(dx_fast, dx_ref)

    # Timing
    pt_fwd_time = time_function(pytorch_forward, warmup=warmup, iterations=iterations, name="PyTorch Fwd")
    pt_fwd_bwd_time = time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch Fwd+Bwd")
    c_simd_time = time_function(c_backward_simd, warmup=warmup, iterations=iterations, name="C SIMD")
    c_scalar_time = time_function(c_backward_scalar, warmup=warmup, iterations=iterations, name="C Scalar")
    c_fast_time = time_function(c_backward_fast, warmup=warmup, iterations=iterations, name="C Fast")

    pt_bwd_est = pt_fwd_bwd_time.mean_us - pt_fwd_time.mean_us

    # SIMD version: uses tanh512_fast for performance on AVX-512
    report.add_result(TestResult(
        name="SIMD (tanh approx)",
        passed=diff_simd <= 1e-1,
        max_diff=diff_simd,
        tolerance=1e-1,
        pytorch_time=pt_fwd_bwd_time,
        kernel_time=c_simd_time
    ))

    # Scalar exact version: full precision using standard library tanhf
    report.add_result(TestResult(
        name="Exact (scalar)",
        passed=diff_scalar <= 1e-6,
        max_diff=diff_scalar,
        tolerance=1e-6,
        pytorch_time=None,
        kernel_time=c_scalar_time
    ))

    # Fast version: sigmoid approximation, fastest but least accurate
    report.add_result(TestResult(
        name="Fast (sigmoid)",
        passed=diff_fast <= 0.15,
        max_diff=diff_fast,
        tolerance=0.15,
        pytorch_time=None,
        kernel_time=c_fast_time
    ))

    # Store timing data
    report.timing_breakdown = {
        'pt_fwd': pt_fwd_time.mean_us,
        'pt_bwd_est': pt_bwd_est,
        'pt_fwd_bwd': pt_fwd_bwd_time.mean_us,
        'c_simd': c_simd_time.mean_us,
        'c_scalar': c_scalar_time.mean_us,
        'c_fast': c_fast_time.mean_us,
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_system_info()

    # Forward tests
    fwd_report = run_forward_tests(N=4096, warmup=10, iterations=1000)
    fwd_report.print_report()

    # Backward tests
    bwd_report = run_backward_tests(N=4096, warmup=10, iterations=1000)
    bwd_report.print_report()

    # Print detailed timing breakdown
    if hasattr(bwd_report, 'timing_breakdown'):
        t = bwd_report.timing_breakdown
        print("  DETAILED TIMING BREAKDOWN (Backward variants)")
        print("  " + "-" * 68)
        print(f"  {'Operation':<25} {'PyTorch (us)':<15} {'C Kernel (us)':<15} {'Speedup':<10}")
        print("  " + "-" * 68)
        print(f"  {'Forward':<25} {t['pt_fwd']:<15.1f} {'(inplace)':<15} {'-':<10}")
        print(f"  {'Backward SIMD':<25} {t['pt_bwd_est']:<15.1f} {t['c_simd']:<15.1f} {t['pt_bwd_est']/t['c_simd']:.2f}x")
        print(f"  {'Backward Exact':<25} {t['pt_bwd_est']:<15.1f} {t['c_scalar']:<15.1f} {t['pt_bwd_est']/t['c_scalar']:.2f}x")
        print(f"  {'Backward Fast':<25} {t['pt_bwd_est']:<15.1f} {t['c_fast']:<15.1f} {t['pt_bwd_est']/t['c_fast']:.2f}x")
        print("  " + "-" * 68)
        print()

    # Exit with error if any tests failed
    if not fwd_report.all_passed() or not bwd_report.all_passed():
        exit(1)
