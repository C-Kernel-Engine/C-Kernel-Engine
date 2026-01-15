"""
Sigmoid BF16 kernel unit tests with accuracy metrics.

Runs the forward/backward BF16 wrappers and compares them against PyTorch float32
references. The BF16 helpers in bf16_utils.py are used to convert between float32
arrays and the uint16 bit patterns expected by the kernels.
"""
import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "unittest"
for path in (ROOT, UNITS):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, time_function, print_system_info
)
from bf16_utils import (
    float32_to_bf16, bf16_to_float32, numpy_to_uint16_ptr
)

cpu = get_cpu_info()
if not cpu.avx512bf16:
    print("BF16 kernels require AVX-512 BF16; skipping this test on the current CPU.")
    sys.exit(0)


# Load the BF16-enabled shared library
lib = load_lib("libckernel_sigmoid.so", "libckernel_engine.so")

# Function signatures - updated with scratch buffer params (no internal malloc)
lib.sigmoid_forward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # input
    ctypes.POINTER(ctypes.c_uint16),  # output
    ctypes.c_size_t,                   # n
    ctypes.POINTER(ctypes.c_float),   # scratch_input [n]
    ctypes.POINTER(ctypes.c_float),   # scratch_output [n]
]
lib.sigmoid_forward_bf16.restype = None

lib.sigmoid_backward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # input
    ctypes.POINTER(ctypes.c_uint16),  # d_output
    ctypes.POINTER(ctypes.c_uint16),  # d_input
    ctypes.c_size_t,                   # n
    ctypes.POINTER(ctypes.c_float),   # scratch_input [n]
    ctypes.POINTER(ctypes.c_float),   # scratch_d_output [n]
    ctypes.POINTER(ctypes.c_float),   # scratch_d_input [n]
]
lib.sigmoid_backward_bf16.restype = None


def run_forward_tests(N=4096, warmup=10, iterations=1000):
    """Run the BF16 forward pass and compare against PyTorch."""
    np.random.seed(0)
    x_np = np.random.randn(N).astype(np.float32)
    out_bf16 = np.zeros(N, dtype=np.uint16)

    x_bf16 = float32_to_bf16(x_np)
    x_ptr = numpy_to_uint16_ptr(x_bf16)
    out_ptr = numpy_to_uint16_ptr(out_bf16)

    # Allocate scratch buffers (caller-provided, no malloc in kernel)
    scratch_input = np.zeros(N, dtype=np.float32)
    scratch_output = np.zeros(N, dtype=np.float32)
    scratch_input_ptr = scratch_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    scratch_output_ptr = scratch_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    x = torch.from_numpy(x_np.copy()).to(dtype=torch.bfloat16)
    report = TestReport(
        test_name="Sigmoid Forward",
        dtype="bf16",
        shape=f"N={N}",
        cpu_info=get_cpu_info()
    )

    def pytorch_sigmoid():
        return torch.sigmoid(x)

    pytorch_ref = pytorch_sigmoid().to(dtype=torch.bfloat16)

    def c_sigmoid():
        lib.sigmoid_forward_bf16(x_ptr, out_ptr, ctypes.c_size_t(N),
                                 scratch_input_ptr, scratch_output_ptr)

    c_sigmoid()
    out_fp32 = torch.from_numpy(bf16_to_float32(out_bf16.copy()))
    diff = max_diff(out_fp32, pytorch_ref.to(dtype=torch.float32))

    report.add_result(TestResult(
        name="Sigmoid",
        passed=diff <= 1e-2,
        max_diff=diff,
        tolerance=1e-2,
        pytorch_time=time_function(pytorch_sigmoid, warmup=warmup, iterations=iterations, name="PyTorch"),
        kernel_time=time_function(c_sigmoid, warmup=warmup, iterations=iterations, name="C Sigmoid")
    ))

    return report


def run_backward_tests(N=4096, warmup=10, iterations=1000):
    """Run the BF16 backward pass and compare against PyTorch gradients."""
    np.random.seed(1)
    x_np = np.random.randn(N).astype(np.float32)
    upstream_np = np.random.randn(N).astype(np.float32)
    dx_bf16 = np.zeros(N, dtype=np.uint16)

    x_bf16 = float32_to_bf16(x_np)
    upstream_bf16 = float32_to_bf16(upstream_np)

    x_ptr = numpy_to_uint16_ptr(x_bf16)
    upstream_ptr = numpy_to_uint16_ptr(upstream_bf16)
    dx_ptr = numpy_to_uint16_ptr(dx_bf16)

    # Allocate scratch buffers (caller-provided, no malloc in kernel)
    scratch_input = np.zeros(N, dtype=np.float32)
    scratch_d_output = np.zeros(N, dtype=np.float32)
    scratch_d_input = np.zeros(N, dtype=np.float32)
    scratch_input_ptr = scratch_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    scratch_d_output_ptr = scratch_d_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    scratch_d_input_ptr = scratch_d_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    x = torch.from_numpy(x_np.copy()).to(dtype=torch.bfloat16)
    upstream = torch.from_numpy(upstream_np.copy()).to(dtype=torch.bfloat16)

    report = TestReport(
        test_name="Sigmoid Backward",
        dtype="bf16",
        shape=f"N={N}",
        cpu_info=get_cpu_info()
    )

    def pytorch_forward():
        return torch.sigmoid(x)

    def pytorch_fwd_bwd():
        x_ref = x.clone().detach().requires_grad_(True)
        out = torch.sigmoid(x_ref)
        out.backward(upstream)
        return x_ref.grad

    dx_ref = pytorch_fwd_bwd()

    def c_backward():
        lib.sigmoid_backward_bf16(x_ptr, upstream_ptr, dx_ptr, ctypes.c_size_t(N),
                                  scratch_input_ptr, scratch_d_output_ptr, scratch_d_input_ptr)

    c_backward()
    dx_c = torch.from_numpy(bf16_to_float32(dx_bf16.copy()))
    diff = max_diff(dx_c, dx_ref)

    report.add_result(TestResult(
        name="d_input",
        passed=diff <= 1e-2,
        max_diff=diff,
        tolerance=1e-2,
        pytorch_time=time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch Fwd+Bwd"),
        kernel_time=time_function(c_backward, warmup=warmup, iterations=iterations, name="C Sigmoid Bwd")
    ))

    return report


if __name__ == "__main__":
    print_system_info()

    fwd_report = run_forward_tests()
    fwd_report.print_report()

    bwd_report = run_backward_tests()
    bwd_report.print_report()

    if not fwd_report.all_passed() or not bwd_report.all_passed():
        exit(1)
