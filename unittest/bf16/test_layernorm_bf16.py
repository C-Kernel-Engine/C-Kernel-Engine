"""
LayerNorm BF16 kernel unit tests.

Exercises the `layernorm_forward_unrolled_slice_bf16` and
`layernorm_backward_kernel_bf16` helpers against PyTorch's LayerNorm using
BF16 inputs + float32 scale/bias.
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

lib = load_lib("libckernel_layernorm.so", "libckernel_engine.so")

# Function signatures - updated with scratch buffer params (no internal malloc)
lib.layernorm_forward_unrolled_slice_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # input
    ctypes.POINTER(ctypes.c_float),   # gamma
    ctypes.POINTER(ctypes.c_float),   # beta
    ctypes.POINTER(ctypes.c_uint16),  # output
    ctypes.POINTER(ctypes.c_float),   # mean cache
    ctypes.POINTER(ctypes.c_float),   # rstd cache
    ctypes.c_int,                      # tokens
    ctypes.c_int,                      # d_model
    ctypes.c_float,                    # eps
    ctypes.POINTER(ctypes.c_float),   # scratch_input [tokens * d_model]
    ctypes.POINTER(ctypes.c_float),   # scratch_output [tokens * d_model]
]
lib.layernorm_forward_unrolled_slice_bf16.restype = None

lib.layernorm_backward_kernel_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # d_output
    ctypes.POINTER(ctypes.c_uint16),  # input
    ctypes.POINTER(ctypes.c_float),   # gamma
    ctypes.POINTER(ctypes.c_float),   # mean
    ctypes.POINTER(ctypes.c_float),   # rstd
    ctypes.POINTER(ctypes.c_uint16),  # d_input
    ctypes.POINTER(ctypes.c_float),   # d_gamma
    ctypes.POINTER(ctypes.c_float),   # d_beta
    ctypes.c_int,                      # tokens
    ctypes.c_int,                      # d_model
    ctypes.c_int,                      # aligned_embed_dim
    ctypes.POINTER(ctypes.c_float),   # scratch_d_output [tokens * aligned_embed_dim]
    ctypes.POINTER(ctypes.c_float),   # scratch_input [tokens * aligned_embed_dim]
    ctypes.POINTER(ctypes.c_float),   # scratch_d_input [tokens * aligned_embed_dim]
]
lib.layernorm_backward_kernel_bf16.restype = None


def run_forward_tests(tokens=4, dim=16, eps=1e-5):
    np.random.seed(0)
    x_np = np.random.randn(tokens, dim).astype(np.float32)
    gamma_np = np.random.randn(dim).astype(np.float32)
    beta_np = np.random.randn(dim).astype(np.float32)

    # Allocate scratch buffers (caller-provided, no malloc in kernel)
    scratch_input = np.zeros(tokens * dim, dtype=np.float32)
    scratch_output = np.zeros(tokens * dim, dtype=np.float32)
    scratch_input_ptr = scratch_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    scratch_output_ptr = scratch_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def c_layernorm_forward():
        data = float32_to_bf16(x_np.copy().reshape(-1))
        out_bf16 = np.zeros_like(data)
        mean_cache = np.zeros(tokens, dtype=np.float32)
        rstd_cache = np.zeros(tokens, dtype=np.float32)
        lib.layernorm_forward_unrolled_slice_bf16(
            numpy_to_uint16_ptr(data),
            gamma_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            beta_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            numpy_to_uint16_ptr(out_bf16),
            mean_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rstd_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(tokens),
            ctypes.c_int(dim),
            ctypes.c_float(eps),
            scratch_input_ptr,
            scratch_output_ptr,
        )
        return out_bf16, mean_cache, rstd_cache

    out_bf16, mean_cache, rstd_cache = c_layernorm_forward()
    out_fp32 = torch.from_numpy(bf16_to_float32(out_bf16.reshape(tokens, dim)))

    def pytorch_layernorm():
        x = torch.from_numpy(x_np.copy()).to(dtype=torch.bfloat16)
        return torch.nn.functional.layer_norm(
            x, (dim,), weight=torch.from_numpy(gamma_np.copy()),
            bias=torch.from_numpy(beta_np.copy()), eps=eps
        )

    diff = max_diff(out_fp32, pytorch_layernorm().to(dtype=torch.float32))
    pytorch_time = time_function(pytorch_layernorm, warmup=10, iterations=100, name="PyTorch")
    kernel_time = time_function(c_layernorm_forward, warmup=5, iterations=100, name="C LayerNorm")

    report = TestReport(
        test_name="LayerNorm Forward",
        dtype="bf16",
        shape=f"T={tokens},D={dim}",
        cpu_info=get_cpu_info()
    )

    report.add_result(TestResult(
        name="LayerNorm",
        passed=diff <= 1e-2,
        max_diff=diff,
        tolerance=1e-2,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time,
    ))

    return report, mean_cache, rstd_cache, x_np, gamma_np, beta_np


def run_backward_tests(tokens=4, dim=16, eps=1e-5):
    fwd_report, mean_cache, rstd_cache, x_np, gamma_np, beta_np = run_forward_tests(tokens, dim, eps)

    d_output_np = np.random.randn(tokens, dim).astype(np.float32)
    d_output_bf16 = float32_to_bf16(d_output_np.copy().reshape(-1))
    dx_bf16 = np.zeros_like(d_output_bf16)
    d_gamma = np.zeros(dim, dtype=np.float32)
    d_beta = np.zeros(dim, dtype=np.float32)

    # Allocate scratch buffers (caller-provided, no malloc in kernel)
    scratch_d_output = np.zeros(tokens * dim, dtype=np.float32)
    scratch_input = np.zeros(tokens * dim, dtype=np.float32)
    scratch_d_input = np.zeros(tokens * dim, dtype=np.float32)
    scratch_d_output_ptr = scratch_d_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    scratch_input_ptr = scratch_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    scratch_d_input_ptr = scratch_d_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.layernorm_backward_kernel_bf16(
        numpy_to_uint16_ptr(d_output_bf16),
        numpy_to_uint16_ptr(float32_to_bf16(x_np.copy().reshape(-1))),
        gamma_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        mean_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rstd_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        numpy_to_uint16_ptr(dx_bf16),
        d_gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        d_beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(tokens),
        ctypes.c_int(dim),
        ctypes.c_int(dim),
        scratch_d_output_ptr,
        scratch_input_ptr,
        scratch_d_input_ptr,
    )

    report = TestReport(
        test_name="LayerNorm Backward",
        dtype="bf16",
        shape=f"T={tokens},D={dim}",
        cpu_info=get_cpu_info()
    )

    x = torch.from_numpy(x_np.copy()).to(dtype=torch.bfloat16).requires_grad_(True)
    gamma = torch.from_numpy(gamma_np.copy()).to(dtype=torch.float32).requires_grad_(True)
    beta = torch.from_numpy(beta_np.copy()).to(dtype=torch.float32).requires_grad_(True)
    out = torch.nn.functional.layer_norm(
        x, (dim,), weight=gamma, bias=beta, eps=eps
    )
    upstream = torch.from_numpy(d_output_np.copy()).to(dtype=torch.bfloat16)
    out.backward(upstream)

    dx_ref = x.grad.to(dtype=torch.float32)
    d_gamma_ref = gamma.grad
    d_beta_ref = beta.grad

    dx_c = torch.from_numpy(bf16_to_float32(dx_bf16.reshape(tokens, dim)))
    diff_input = max_diff(dx_c, dx_ref)
    diff_gamma = max_diff(torch.from_numpy(d_gamma), d_gamma_ref)
    diff_beta = max_diff(torch.from_numpy(d_beta), d_beta_ref)

    report.add_result(TestResult(
        name="d_input",
        passed=diff_input <= 1e-2,
        max_diff=diff_input,
        tolerance=1e-2,
        pytorch_time=0.0,
        kernel_time=0.0,
    ))
    report.add_result(TestResult(
        name="d_gamma",
        passed=diff_gamma <= 1e-2,
        max_diff=diff_gamma,
        tolerance=1e-2,
    ))
    report.add_result(TestResult(
        name="d_beta",
        passed=diff_beta <= 1e-2,
        max_diff=diff_beta,
        tolerance=1e-2,
    ))

    return report


if __name__ == "__main__":
    print_system_info()

    fwd_report, *_ = run_forward_tests()
    fwd_report.print_report()

    bwd_report = run_backward_tests()
    bwd_report.print_report()

    if not fwd_report.all_passed() or not bwd_report.all_passed():
        sys.exit(1)
