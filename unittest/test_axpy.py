"""
AXPY kernel unit tests with performance metrics.

Tests y += alpha * x operations against PyTorch reference.
Reports accuracy, timing, and system information.
"""
import ctypes

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


# Load the library
lib = load_lib("libckernel_engine.so")

# =============================================================================
# Function signatures
# =============================================================================

lib.axpy_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # y (in/out)
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.c_float,                   # alpha
    ctypes.c_int,                     # n
]
lib.axpy_f32.restype = None

lib.scal_copy_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # y (out)
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.c_float,                   # alpha
    ctypes.c_int,                     # n
]
lib.scal_copy_f32.restype = None

lib.axpy_zero_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # y (out)
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.c_float,                   # alpha
    ctypes.c_int,                     # n
]
lib.axpy_zero_f32.restype = None

lib.axpy_2d_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # Y
    ctypes.POINTER(ctypes.c_float),  # X
    ctypes.c_float,                   # alpha
    ctypes.c_int,                     # num_tokens
    ctypes.c_int,                     # dim
    ctypes.c_int,                     # y_stride
    ctypes.c_int,                     # x_stride
]
lib.axpy_2d_f32.restype = None

lib.moe_accumulate_expert_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.POINTER(ctypes.c_float),  # expert_output
    ctypes.c_float,                   # routing_weight
    ctypes.c_int,                     # hidden_dim
]
lib.moe_accumulate_expert_f32.restype = None


# =============================================================================
# Tests
# =============================================================================

def run_axpy_tests(n=4096, warmup=10, iterations=1000):
    """Run basic AXPY tests with accuracy and timing."""
    np.random.seed(42)

    # Generate random vectors
    y_np = np.random.randn(n).astype(np.float32)
    x_np = np.random.randn(n).astype(np.float32)
    alpha = 0.75

    # Make copies for C kernel
    y_c = y_np.copy()
    y_ptr = numpy_to_ptr(y_c)
    x_ptr = numpy_to_ptr(x_np)

    # PyTorch reference
    y_torch = torch.from_numpy(y_np.copy())
    x_torch = torch.from_numpy(x_np.copy())

    report = TestReport(
        test_name="AXPY (y += alpha * x)",
        dtype="fp32",
        shape=f"n={n}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    def pytorch_axpy():
        y_t = y_torch.clone()
        y_t.add_(x_torch, alpha=alpha)
        return y_t

    def c_axpy():
        np.copyto(y_c, y_np)
        lib.axpy_f32(y_ptr, x_ptr, alpha, n)

    # Run once for accuracy
    c_axpy()
    pt_result = pytorch_axpy()
    c_result = torch.from_numpy(y_c.copy())

    diff = max_diff(c_result, pt_result)

    # Timing
    pt_time = time_function(pytorch_axpy, warmup=warmup, iterations=iterations, name="PyTorch")
    c_time = time_function(c_axpy, warmup=warmup, iterations=iterations, name="C axpy_f32")

    report.add_result(TestResult(
        name="axpy_f32",
        passed=diff < 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=pt_time,
        kernel_time=c_time
    ))

    return report


def run_scal_copy_tests(n=4096, warmup=10, iterations=1000):
    """Run scaled copy tests: y = alpha * x."""
    np.random.seed(43)

    x_np = np.random.randn(n).astype(np.float32)
    y_np = np.zeros(n, dtype=np.float32)
    alpha = 2.5

    y_ptr = numpy_to_ptr(y_np)
    x_ptr = numpy_to_ptr(x_np)

    x_torch = torch.from_numpy(x_np.copy())

    report = TestReport(
        test_name="Scaled Copy (y = alpha * x)",
        dtype="fp32",
        shape=f"n={n}",
        cpu_info=get_cpu_info()
    )

    def pytorch_scal_copy():
        return x_torch * alpha

    def c_scal_copy():
        lib.scal_copy_f32(y_ptr, x_ptr, alpha, n)

    # Run once for accuracy
    c_scal_copy()
    pt_result = pytorch_scal_copy()
    c_result = torch.from_numpy(y_np.copy())

    diff = max_diff(c_result, pt_result)

    # Timing
    pt_time = time_function(pytorch_scal_copy, warmup=warmup, iterations=iterations, name="PyTorch")
    c_time = time_function(c_scal_copy, warmup=warmup, iterations=iterations, name="C scal_copy")

    report.add_result(TestResult(
        name="scal_copy_f32",
        passed=diff < 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=pt_time,
        kernel_time=c_time
    ))

    return report


def run_moe_accumulate_tests(hidden_dim=2048, num_experts=4, warmup=10, iterations=1000):
    """Run MoE-style expert accumulation test."""
    np.random.seed(44)

    # Simulate MoE gathering: output = sum_i(weight_i * expert_output_i)
    expert_outputs = [np.random.randn(hidden_dim).astype(np.float32) for _ in range(num_experts)]
    weights = np.random.rand(num_experts).astype(np.float32)
    weights /= weights.sum()  # Normalize like softmax

    output_np = np.zeros(hidden_dim, dtype=np.float32)
    output_ptr = numpy_to_ptr(output_np)

    expert_ptrs = [numpy_to_ptr(e) for e in expert_outputs]
    expert_tensors = [torch.from_numpy(e.copy()) for e in expert_outputs]
    weights_torch = torch.from_numpy(weights.copy())

    report = TestReport(
        test_name="MoE Expert Accumulation",
        dtype="fp32",
        shape=f"hidden={hidden_dim}, experts={num_experts}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    def pytorch_moe_gather():
        result = torch.zeros(hidden_dim)
        for i in range(num_experts):
            result.add_(expert_tensors[i], alpha=float(weights[i]))
        return result

    def c_moe_gather():
        output_np.fill(0)
        for i in range(num_experts):
            lib.moe_accumulate_expert_f32(output_ptr, expert_ptrs[i], weights[i], hidden_dim)

    # Run once for accuracy
    c_moe_gather()
    pt_result = pytorch_moe_gather()
    c_result = torch.from_numpy(output_np.copy())

    diff = max_diff(c_result, pt_result)

    # Timing
    pt_time = time_function(pytorch_moe_gather, warmup=warmup, iterations=iterations, name="PyTorch")
    c_time = time_function(c_moe_gather, warmup=warmup, iterations=iterations, name="C moe_accum")

    report.add_result(TestResult(
        name="moe_accumulate_expert_f32",
        passed=diff < 1e-4,
        max_diff=diff,
        tolerance=1e-4,
        pytorch_time=pt_time,
        kernel_time=c_time
    ))

    return report


def run_axpy_2d_tests(num_tokens=64, dim=2048, warmup=10, iterations=1000):
    """Run 2D batched AXPY tests."""
    np.random.seed(45)

    Y_np = np.random.randn(num_tokens, dim).astype(np.float32)
    X_np = np.random.randn(num_tokens, dim).astype(np.float32)
    alpha = 0.5

    Y_c = Y_np.copy()
    Y_ptr = numpy_to_ptr(Y_c)
    X_ptr = numpy_to_ptr(X_np)

    Y_torch = torch.from_numpy(Y_np.copy())
    X_torch = torch.from_numpy(X_np.copy())

    report = TestReport(
        test_name="AXPY 2D Batched",
        dtype="fp32",
        shape=f"tokens={num_tokens}, dim={dim}",
        cpu_info=get_cpu_info()
    )

    def pytorch_axpy_2d():
        Y_t = Y_torch.clone()
        Y_t.add_(X_torch, alpha=alpha)
        return Y_t

    def c_axpy_2d():
        np.copyto(Y_c, Y_np)
        lib.axpy_2d_f32(Y_ptr, X_ptr, alpha, num_tokens, dim, dim, dim)

    # Run once for accuracy
    c_axpy_2d()
    pt_result = pytorch_axpy_2d()
    c_result = torch.from_numpy(Y_c.copy())

    diff = max_diff(c_result, pt_result)

    # Timing
    pt_time = time_function(pytorch_axpy_2d, warmup=warmup, iterations=iterations, name="PyTorch")
    c_time = time_function(c_axpy_2d, warmup=warmup, iterations=iterations, name="C axpy_2d")

    report.add_result(TestResult(
        name="axpy_2d_f32",
        passed=diff < 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=pt_time,
        kernel_time=c_time
    ))

    return report


def run_large_vector_tests(warmup=5, iterations=100):
    """Test performance on large vectors (model-scale)."""
    np.random.seed(46)

    # Typical LLM hidden dimensions
    sizes = [2048, 4096, 8192, 14336]  # Common sizes

    report = TestReport(
        test_name="AXPY Large Vector Performance",
        dtype="fp32",
        shape="various",
        cpu_info=get_cpu_info()
    )

    for n in sizes:
        y_np = np.random.randn(n).astype(np.float32)
        x_np = np.random.randn(n).astype(np.float32)
        alpha = 0.5

        y_c = y_np.copy()
        y_ptr = numpy_to_ptr(y_c)
        x_ptr = numpy_to_ptr(x_np)

        y_torch = torch.from_numpy(y_np.copy())
        x_torch = torch.from_numpy(x_np.copy())

        def pytorch_axpy():
            y_t = y_torch.clone()
            y_t.add_(x_torch, alpha=alpha)
            return y_t

        def c_axpy():
            np.copyto(y_c, y_np)
            lib.axpy_f32(y_ptr, x_ptr, alpha, n)

        c_axpy()
        pt_result = pytorch_axpy()
        c_result = torch.from_numpy(y_c.copy())
        diff = max_diff(c_result, pt_result)

        pt_time = time_function(pytorch_axpy, warmup=warmup, iterations=iterations, name=f"PyTorch n={n}")
        c_time = time_function(c_axpy, warmup=warmup, iterations=iterations, name=f"C n={n}")

        # Calculate bandwidth (2 reads + 1 write = 3 * n * 4 bytes)
        bytes_moved = 3 * n * 4
        bandwidth_gbs = bytes_moved / (c_time.mean_us * 1e-6) / 1e9

        report.add_result(TestResult(
            name=f"n={n} ({bandwidth_gbs:.1f} GB/s)",
            passed=diff < 1e-5,
            max_diff=diff,
            tolerance=1e-5,
            pytorch_time=pt_time,
            kernel_time=c_time
        ))

    return report


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print_system_info()

    # Basic AXPY
    axpy_report = run_axpy_tests(n=4096, warmup=10, iterations=1000)
    axpy_report.print_report()

    # Scaled copy
    scal_report = run_scal_copy_tests(n=4096, warmup=10, iterations=1000)
    scal_report.print_report()

    # MoE accumulation pattern
    moe_report = run_moe_accumulate_tests(hidden_dim=2048, num_experts=4, warmup=10, iterations=1000)
    moe_report.print_report()

    # 2D batched
    axpy_2d_report = run_axpy_2d_tests(num_tokens=64, dim=2048, warmup=10, iterations=1000)
    axpy_2d_report.print_report()

    # Large vector performance
    large_report = run_large_vector_tests(warmup=5, iterations=100)
    large_report.print_report()

    # Exit with error if any tests failed
    all_passed = (
        axpy_report.all_passed() and
        scal_report.all_passed() and
        moe_report.all_passed() and
        axpy_2d_report.all_passed() and
        large_report.all_passed()
    )
    if not all_passed:
        exit(1)
