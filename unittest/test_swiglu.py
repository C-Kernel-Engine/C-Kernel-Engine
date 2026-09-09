"""
SwiGLU kernel unit tests with performance metrics.

Tests forward and backward passes against PyTorch reference.
Reports accuracy, timing, and system information.

Tests both fast (SIMD approximation) and exact (scalar) versions.
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
lib = load_lib("libckernel_swiglu.so", "libckernel_engine.so")

# ═══════════════════════════════════════════════════════════════════════════════
# Function signatures
# ═══════════════════════════════════════════════════════════════════════════════

# Fast version (SIMD with exp approximation)
lib.swiglu_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input [T × 2D]
    ctypes.POINTER(ctypes.c_float),  # output [T × D]
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # dim
]
lib.swiglu_forward.restype = None

lib.swiglu_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input [T × 2D]
    ctypes.POINTER(ctypes.c_float),  # d_output [T × D]
    ctypes.POINTER(ctypes.c_float),  # d_input [T × 2D]
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # dim
]
lib.swiglu_backward.restype = None

# Exact version (scalar using standard library expf)
lib.swiglu_forward_exact.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input [T × 2D]
    ctypes.POINTER(ctypes.c_float),  # output [T × D]
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # dim
]
lib.swiglu_forward_exact.restype = None

lib.swiglu_forward_ggml.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
]
lib.swiglu_forward_ggml.restype = None

lib.ck_set_num_threads.argtypes = [ctypes.c_int]
lib.ck_set_num_threads.restype = None
lib.ck_threadpool_global_destroy.argtypes = []
lib.ck_threadpool_global_destroy.restype = None

lib.swiglu_backward_exact.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input [T × 2D]
    ctypes.POINTER(ctypes.c_float),  # d_output [T × D]
    ctypes.POINTER(ctypes.c_float),  # d_input [T × 2D]
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # dim
]
lib.swiglu_backward_exact.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Reference implementation
# ═══════════════════════════════════════════════════════════════════════════════

def swiglu_torch(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: x: [T, 2D] => gate, value: [T, D]"""
    T, twoD = x.shape
    D = twoD // 2
    gate, value = x[:, :D], x[:, D:]
    return F.silu(gate) * value


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_forward_tests(T=64, D=128, warmup=10, iterations=1000):
    """Run forward pass tests with accuracy and timing.

    Tests both fast (SIMD) and exact (scalar) versions.
    """
    np.random.seed(0)

    # Pre-allocate numpy arrays
    x_np = np.random.randn(T, 2 * D).astype(np.float32)
    out_fast_np = np.zeros((T, D), dtype=np.float32)
    out_exact_np = np.zeros((T, D), dtype=np.float32)

    # Get pointers
    x_ptr = numpy_to_ptr(x_np)
    out_fast_ptr = numpy_to_ptr(out_fast_np)
    out_exact_ptr = numpy_to_ptr(out_exact_np)

    # Torch tensor
    x = torch.from_numpy(x_np.copy())

    report = TestReport(
        test_name="SwiGLU Forward",
        dtype="fp32",
        shape=f"T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    ref = swiglu_torch(x)

    # Time PyTorch
    pytorch_time = time_function(
        lambda: swiglu_torch(x),
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # === Fast version (SIMD with exp approximation) ===
    def c_swiglu_fast():
        lib.swiglu_forward(x_ptr, out_fast_ptr, ctypes.c_int(T), ctypes.c_int(D))

    c_swiglu_fast()
    out_fast = torch.from_numpy(out_fast_np.copy())
    diff_fast = max_diff(out_fast, ref)
    kernel_time_fast = time_function(c_swiglu_fast, warmup=warmup, iterations=iterations, name="C Fast")

    # Fast version: uses exp approximation, relaxed tolerance
    report.add_result(TestResult(
        name="Fast (SIMD approx)",
        passed=diff_fast <= 5e-5,
        max_diff=diff_fast,
        tolerance=5e-5,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time_fast
    ))

    # === Exact version (scalar using standard library expf) ===
    def c_swiglu_exact():
        lib.swiglu_forward_exact(x_ptr, out_exact_ptr, ctypes.c_int(T), ctypes.c_int(D))

    c_swiglu_exact()
    out_exact = torch.from_numpy(out_exact_np.copy())
    diff_exact = max_diff(out_exact, ref)
    kernel_time_exact = time_function(c_swiglu_exact, warmup=warmup, iterations=iterations, name="C Exact")

    # Exact version: full precision using standard library expf
    report.add_result(TestResult(
        name="Exact (scalar)",
        passed=diff_exact <= 1e-6,
        max_diff=diff_exact,
        tolerance=1e-6,
        pytorch_time=None,
        kernel_time=kernel_time_exact
    ))

    return report


def run_backward_tests(T=64, D=128, warmup=10, iterations=1000):
    """Run backward pass tests with accuracy and timing.

    Tests both fast (SIMD) and exact (scalar) versions.
    """
    np.random.seed(1)

    # Pre-allocate numpy arrays
    x_np = np.random.randn(T, 2 * D).astype(np.float32)
    upstream_np = np.random.randn(T, D).astype(np.float32)
    dx_fast_np = np.zeros((T, 2 * D), dtype=np.float32)
    dx_exact_np = np.zeros((T, 2 * D), dtype=np.float32)

    # Get pointers
    x_ptr = numpy_to_ptr(x_np)
    upstream_ptr = numpy_to_ptr(upstream_np)
    dx_fast_ptr = numpy_to_ptr(dx_fast_np)
    dx_exact_ptr = numpy_to_ptr(dx_exact_np)

    # Torch tensors
    x = torch.from_numpy(x_np.copy())
    upstream = torch.from_numpy(upstream_np)

    report = TestReport(
        test_name="SwiGLU Backward",
        dtype="fp32",
        shape=f"T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch forward only
    def pytorch_forward():
        return swiglu_torch(x)

    # PyTorch forward+backward
    def pytorch_fwd_bwd():
        x_ref = x.clone().detach().requires_grad_(True)
        y = swiglu_torch(x_ref)
        y.backward(upstream)
        return x_ref.grad

    # Get reference grad
    dx_ref = pytorch_fwd_bwd()

    # Timing for PyTorch
    pt_fwd_time = time_function(pytorch_forward, warmup=warmup, iterations=iterations, name="PyTorch Fwd")
    pt_fwd_bwd_time = time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch Fwd+Bwd")
    pt_bwd_est = pt_fwd_bwd_time.mean_us - pt_fwd_time.mean_us

    # === Fast version (SIMD with exp approximation) ===
    def c_backward_fast():
        lib.swiglu_backward(x_ptr, upstream_ptr, dx_fast_ptr, ctypes.c_int(T), ctypes.c_int(D))

    c_backward_fast()
    dx_fast = torch.from_numpy(dx_fast_np.copy())
    diff_fast = max_diff(dx_fast, dx_ref)
    c_bwd_fast_time = time_function(c_backward_fast, warmup=warmup, iterations=iterations, name="C Bwd Fast")

    # Fast version: uses exp approximation, relaxed tolerance
    report.add_result(TestResult(
        name="Fast d_input",
        passed=diff_fast <= 1e-4,
        max_diff=diff_fast,
        tolerance=1e-4,
        pytorch_time=pt_fwd_bwd_time,
        kernel_time=c_bwd_fast_time
    ))

    # === Exact version (scalar using standard library expf) ===
    def c_backward_exact():
        lib.swiglu_backward_exact(x_ptr, upstream_ptr, dx_exact_ptr, ctypes.c_int(T), ctypes.c_int(D))

    c_backward_exact()
    dx_exact = torch.from_numpy(dx_exact_np.copy())
    diff_exact = max_diff(dx_exact, dx_ref)
    c_bwd_exact_time = time_function(c_backward_exact, warmup=warmup, iterations=iterations, name="C Bwd Exact")

    # Exact version: full precision using standard library expf
    report.add_result(TestResult(
        name="Exact d_input",
        passed=diff_exact <= 1e-6,
        max_diff=diff_exact,
        tolerance=1e-6,
        pytorch_time=None,
        kernel_time=c_bwd_exact_time
    ))

    # Store timing data
    report.timing_breakdown = {
        'pt_fwd': pt_fwd_time.mean_us,
        'pt_bwd_est': pt_bwd_est,
        'pt_fwd_bwd': pt_fwd_bwd_time.mean_us,
        'c_bwd_fast': c_bwd_fast_time.mean_us,
        'c_bwd_exact': c_bwd_exact_time.mean_us,
    }

    return report


def run_parallel_consistency_tests():
    rng = np.random.default_rng(20260908)
    cases = ((1, 127), (2, 16384), (17, 4097), (128, 4096))
    functions = (lib.swiglu_forward_exact, lib.swiglu_forward_ggml)

    try:
        for tokens, dim in cases:
            source = rng.normal(0.0, 1.0, (tokens, 2 * dim)).astype(np.float32)
            for function in functions:
                serial = np.empty((tokens, dim), dtype=np.float32)
                parallel = np.empty_like(serial)

                lib.ck_threadpool_global_destroy()
                lib.ck_set_num_threads(1)
                function(
                    numpy_to_ptr(source), numpy_to_ptr(serial), tokens, dim
                )

                lib.ck_threadpool_global_destroy()
                lib.ck_set_num_threads(4)
                function(
                    numpy_to_ptr(source), numpy_to_ptr(parallel), tokens, dim
                )
                assert np.array_equal(serial, parallel), (
                    function.__name__, tokens, dim,
                    float(np.max(np.abs(serial - parallel))),
                )

                serial_inplace = source.copy()
                lib.ck_threadpool_global_destroy()
                lib.ck_set_num_threads(1)
                function(
                    numpy_to_ptr(serial_inplace),
                    numpy_to_ptr(serial_inplace), tokens, dim,
                )

                lib.ck_threadpool_global_destroy()
                lib.ck_set_num_threads(4)
                for repetition in range(8):
                    parallel_inplace = source.copy()
                    function(
                        numpy_to_ptr(parallel_inplace),
                        numpy_to_ptr(parallel_inplace), tokens, dim,
                    )
                    assert np.array_equal(
                        serial_inplace.ravel()[:tokens * dim],
                        parallel_inplace.ravel()[:tokens * dim],
                    ), (
                        function.__name__, "inplace", tokens, dim, repetition
                    )
    finally:
        lib.ck_threadpool_global_destroy()
        lib.ck_set_num_threads(0)

    print("SwiGLU serial/threaded exactness [PASS]")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_system_info()

    run_parallel_consistency_tests()

    # Forward tests
    fwd_report = run_forward_tests(T=64, D=128, warmup=10, iterations=1000)
    fwd_report.print_report()

    # Backward tests
    bwd_report = run_backward_tests(T=64, D=128, warmup=10, iterations=1000)
    bwd_report.print_report()

    # Print detailed timing breakdown
    if hasattr(bwd_report, 'timing_breakdown'):
        t = bwd_report.timing_breakdown
        print("  DETAILED TIMING BREAKDOWN (Forward vs Backward)")
        print("  " + "-" * 68)
        print(f"  {'Operation':<25} {'PyTorch (us)':<15} {'C Kernel (us)':<15} {'Speedup':<10}")
        print("  " + "-" * 68)
        print(f"  {'PyTorch Fwd+Bwd':<25} {t['pt_fwd_bwd']:<15.1f} {'-':<15} {'-':<10}")
        print(f"  {'C Fast (SIMD)':<25} {t['pt_fwd_bwd']:<15.1f} {t['c_bwd_fast']:<15.1f} {t['pt_fwd_bwd']/t['c_bwd_fast']:.2f}x")
        print(f"  {'C Exact (scalar)':<25} {t['pt_fwd_bwd']:<15.1f} {t['c_bwd_exact']:<15.1f} {t['pt_fwd_bwd']/t['c_bwd_exact']:.2f}x")
        print("  " + "-" * 68)
        print()

    # Exit with error if any tests failed
    if not fwd_report.all_passed() or not bwd_report.all_passed():
        exit(1)
