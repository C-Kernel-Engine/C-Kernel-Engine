"""
RoPE BF16 kernel unit tests.

Validates the BF16 wrappers (`rope_forward_bf16` / `rope_backward_bf16`)
against the floating-point RoPE kernels in `rope_kernels.c`.
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
from test_utils import TestReport, TestResult, get_cpu_info, max_diff, print_system_info
from bf16_utils import float32_to_bf16, bf16_to_float32, numpy_to_uint16_ptr

cpu = get_cpu_info()
if not cpu.avx512bf16:
    print("BF16 kernels require AVX-512 BF16; skipping this test on the current CPU.")
    sys.exit(0)

lib = load_lib("libckernel_rope.so", "libckernel_engine.so")

lib.rope_precompute_cache.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_float,
]
lib.rope_precompute_cache.restype = None

lib.rope_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.rope_forward.restype = None

# Function signatures - updated with scratch buffer params (no internal malloc)
lib.rope_forward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),  # scratch [num_heads * num_tokens * aligned_head_dim]
]
lib.rope_forward_bf16.restype = None

lib.rope_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.rope_backward.restype = None

lib.rope_backward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),  # scratch_d_out [num_heads * num_tokens * aligned_head_dim]
    ctypes.POINTER(ctypes.c_float),  # scratch_d_x [num_heads * num_tokens * aligned_head_dim]
]
lib.rope_backward_bf16.restype = None


def build_cache(tokens, head_dim, base=10000.0):
    half_dim = head_dim // 2
    max_seq = tokens
    cos_cache = np.zeros((max_seq, half_dim), dtype=np.float32)
    sin_cache = np.zeros_like(cos_cache)
    lib.rope_precompute_cache(
        cos_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sin_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(max_seq),
        ctypes.c_int(head_dim),
        ctypes.c_float(base),
        ctypes.c_int(head_dim),
        ctypes.c_char_p(b"none"),
        ctypes.c_float(1.0),
    )
    return cos_cache, sin_cache


def run_forward_tests(num_heads=2, tokens=4, head_dim=16, pos_offset=0):
    cos_cache, sin_cache = build_cache(tokens + pos_offset, head_dim)
    aligned = head_dim

    x_np = np.random.randn(num_heads, tokens, aligned).astype(np.float32)
    # Match BF16 wrapper contract: BF16 is storage; math is FP32.
    # Quantize inputs to BF16, run FP32 reference, then quantize outputs to BF16.
    x_bf16 = float32_to_bf16(x_np.reshape(-1))
    q_ref = bf16_to_float32(x_bf16.reshape(num_heads, tokens, aligned)).copy()
    lib.rope_forward(
        q_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cos_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sin_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(num_heads),
        ctypes.c_int(tokens),
        ctypes.c_int(head_dim),
        ctypes.c_int(aligned),
        ctypes.c_int(pos_offset),
    )

    # Allocate scratch buffer (caller-provided, no malloc in kernel)
    scratch = np.zeros(num_heads * tokens * aligned, dtype=np.float32)
    scratch_ptr = scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    q_ref_bf16 = float32_to_bf16(q_ref.reshape(-1))
    lib.rope_forward_bf16(
        numpy_to_uint16_ptr(x_bf16),
        cos_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sin_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(num_heads),
        ctypes.c_int(tokens),
        ctypes.c_int(head_dim),
        ctypes.c_int(aligned),
        ctypes.c_int(pos_offset),
        scratch_ptr,
    )

    out_bf16 = bf16_to_float32(x_bf16.reshape(num_heads, tokens, aligned))
    q_ref_out = bf16_to_float32(q_ref_bf16.reshape(num_heads, tokens, aligned))
    diff = max_diff(torch.from_numpy(out_bf16), torch.from_numpy(q_ref_out))

    report = TestReport(
        test_name="RoPE Forward",
        dtype="bf16",
        shape=f"H={num_heads},T={tokens},D={head_dim}",
        cpu_info=get_cpu_info()
    )
    report.add_result(TestResult(
        name="RoPE",
        passed=diff <= 1e-3,
        max_diff=diff,
        tolerance=1e-3,
    ))
    return report


def run_backward_tests(num_heads=2, tokens=4, head_dim=16, pos_offset=0):
    cos_cache, sin_cache = build_cache(tokens + pos_offset, head_dim)
    aligned = head_dim

    d_out_np = np.random.randn(num_heads, tokens, aligned).astype(np.float32)
    # Match BF16 wrapper contract: quantize input to BF16, run FP32 reference,
    # then quantize output to BF16.
    d_out_bf16 = float32_to_bf16(d_out_np.reshape(-1))
    d_out_ref = bf16_to_float32(d_out_bf16.reshape(num_heads, tokens, aligned)).copy()
    dx_ref = np.zeros_like(d_out_ref)
    lib.rope_backward(
        d_out_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dx_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cos_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sin_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(num_heads),
        ctypes.c_int(tokens),
        ctypes.c_int(head_dim),
        ctypes.c_int(aligned),
        ctypes.c_int(pos_offset),
    )

    # Allocate scratch buffers (caller-provided, no malloc in kernel)
    scratch_d_out = np.zeros(num_heads * tokens * aligned, dtype=np.float32)
    scratch_d_x = np.zeros(num_heads * tokens * aligned, dtype=np.float32)
    scratch_d_out_ptr = scratch_d_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    scratch_d_x_ptr = scratch_d_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    dx_ref_bf16 = float32_to_bf16(dx_ref.reshape(-1))
    dx_bf16 = np.zeros_like(d_out_bf16)

    lib.rope_backward_bf16(
        numpy_to_uint16_ptr(d_out_bf16),
        numpy_to_uint16_ptr(dx_bf16),
        cos_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sin_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(num_heads),
        ctypes.c_int(tokens),
        ctypes.c_int(head_dim),
        ctypes.c_int(aligned),
        ctypes.c_int(pos_offset),
        scratch_d_out_ptr,
        scratch_d_x_ptr,
    )

    dx_c = bf16_to_float32(dx_bf16.reshape(num_heads, tokens, aligned))
    dx_ref_out = bf16_to_float32(dx_ref_bf16.reshape(num_heads, tokens, aligned))
    diff = max_diff(torch.from_numpy(dx_c), torch.from_numpy(dx_ref_out))

    report = TestReport(
        test_name="RoPE Backward",
        dtype="bf16",
        shape=f"H={num_heads},T={tokens},D={head_dim}",
        cpu_info=get_cpu_info()
    )
    report.add_result(TestResult(
        name="d_input",
        passed=diff <= 1e-3,
        max_diff=diff,
        tolerance=1e-3,
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
