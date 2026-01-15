"""
GELU BF16 kernel unit tests.

Verifies the in-place forward pass and backward gradient path exposed in
`gelu_kernels_bf16.c` against PyTorch's BF16 implementation.

Note: our BF16 GELU kernels treat BF16 as a storage format and do the math in
FP32 using the tanh GELU approximation (PyTorch `approximate=\"tanh\"`), then
round back to BF16. The reference below matches that contract.
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
    TestReport, TestResult, get_cpu_info, max_diff, time_function,
    print_system_info
)
from bf16_utils import (
    float32_to_bf16, bf16_to_float32, numpy_to_uint16_ptr
)

cpu = get_cpu_info()
if not cpu.avx512bf16:
    print("BF16 kernels require AVX-512 BF16; skipping this test on the current CPU.")
    sys.exit(0)

# Load the library that contains the GELU BF16 wrappers.
lib = load_lib("libckernel_gelu.so", "libckernel_engine.so")

# Function signatures - updated with scratch buffer params (no internal malloc)
lib.gelu_fast_inplace_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # data
    ctypes.c_size_t,                   # n
    ctypes.POINTER(ctypes.c_float),   # scratch [n]
]
lib.gelu_fast_inplace_bf16.restype = None

lib.gelu_backward_exact_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # input
    ctypes.POINTER(ctypes.c_uint16),  # d_output
    ctypes.POINTER(ctypes.c_uint16),  # d_input
    ctypes.c_size_t,                   # n
    ctypes.POINTER(ctypes.c_float),   # scratch_input [n]
    ctypes.POINTER(ctypes.c_float),   # scratch_d_output [n]
    ctypes.POINTER(ctypes.c_float),   # scratch_d_input [n]
]
lib.gelu_backward_exact_bf16.restype = None


def run_forward_tests(N=4096, warmup=10, iterations=1000):
    np.random.seed(0)
    x_np = np.random.randn(N).astype(np.float32)
    data = float32_to_bf16(x_np.copy())
    data_ptr = numpy_to_uint16_ptr(data)

    # Allocate scratch buffer (caller-provided, no malloc in kernel)
    scratch = np.zeros(N, dtype=np.float32)
    scratch_ptr = scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    report = TestReport(
        test_name="GELU Forward",
        dtype="bf16",
        shape=f"N={N}",
        cpu_info=get_cpu_info()
    )

    def pytorch_gelu():
        x = torch.from_numpy(x_np.copy()).to(dtype=torch.bfloat16)
        # BF16 storage -> FP32 compute -> BF16 output (tanh approx).
        return torch.nn.functional.gelu(x.to(dtype=torch.float32), approximate="tanh").to(dtype=torch.bfloat16)

    def c_gelu():
        lib.gelu_fast_inplace_bf16(data_ptr, ctypes.c_size_t(N), scratch_ptr)

    c_gelu()
    out_fp32 = torch.from_numpy(bf16_to_float32(data.copy()))
    diff = max_diff(out_fp32, pytorch_gelu().to(dtype=torch.float32))

    report.add_result(TestResult(
        name="GELU",
        passed=diff <= 1e-2,
        max_diff=diff,
        tolerance=1e-2,
        pytorch_time=time_function(pytorch_gelu, warmup=warmup, iterations=iterations, name="PyTorch"),
        kernel_time=time_function(c_gelu, warmup=warmup, iterations=iterations, name="C GELU")
    ))

    return report


def run_backward_tests(N=4096, warmup=10, iterations=1000):
    np.random.seed(1)
    x_np = np.random.randn(N).astype(np.float32)
    upstream_np = np.random.randn(N).astype(np.float32)
    x_bf16 = float32_to_bf16(x_np)
    upstream_bf16 = float32_to_bf16(upstream_np)
    dx_bf16 = np.zeros(N, dtype=np.uint16)

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

    x_q = torch.from_numpy(bf16_to_float32(x_bf16.copy())).to(dtype=torch.float32)
    upstream_q = torch.from_numpy(bf16_to_float32(upstream_bf16.copy())).to(dtype=torch.float32)

    report = TestReport(
        test_name="GELU Backward",
        dtype="bf16",
        shape=f"N={N}",
        cpu_info=get_cpu_info()
    )

    def pytorch_forward():
        x_ref = x_q.clone().detach().requires_grad_(True)
        return torch.nn.functional.gelu(x_ref, approximate="tanh")

    def pytorch_fwd_bwd():
        x_ref = x_q.clone().detach().requires_grad_(True)
        out = torch.nn.functional.gelu(x_ref, approximate="tanh")
        out.backward(upstream_q)
        # Kernel returns BF16 gradients (quantized), so quantize the reference too.
        return x_ref.grad.to(dtype=torch.bfloat16).to(dtype=torch.float32)

    dx_ref = pytorch_fwd_bwd()

    def c_backward():
        lib.gelu_backward_exact_bf16(x_ptr, upstream_ptr, dx_ptr, ctypes.c_size_t(N),
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
        kernel_time=time_function(c_backward, warmup=warmup, iterations=iterations, name="C GELU Bwd")
    ))

    return report


if __name__ == "__main__":
    print_system_info()

    fwd_report = run_forward_tests()
    fwd_report.print_report()

    bwd_report = run_backward_tests()
    bwd_report.print_report()

    if not fwd_report.all_passed() or not bwd_report.all_passed():
        sys.exit(1)
