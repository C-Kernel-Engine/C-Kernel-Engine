"""
KV-cache + flash-style attention unit tests.

Validates:
  - `attention_forward_causal_head_major_gqa_flash` matches the reference
    score-matrix attention for prefill.
  - `kv_cache_write_head_major` + `attention_forward_decode_head_major_gqa_flash`
    matches full causal attention token-by-token (decode).
"""

import ctypes
import math

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


lib = load_lib("libckernel_engine.so")

lib.attention_forward_causal_head_major_gqa.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # scores
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_kv_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
    ctypes.c_int,                    # aligned_context_window
]
lib.attention_forward_causal_head_major_gqa.restype = None

# Exact version using standard library expf (for accurate reference)
lib.attention_forward_causal_head_major_gqa_exact.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # scores
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_kv_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
    ctypes.c_int,                    # aligned_context_window
]
lib.attention_forward_causal_head_major_gqa_exact.restype = None

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

lib.attention_forward_causal_head_major_gqa_flash_strided.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_kv_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
    ctypes.c_int,                    # kv_stride_tokens
]
lib.attention_forward_causal_head_major_gqa_flash_strided.restype = None

lib.attention_forward_decode_head_major_gqa_flash.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q_token
    ctypes.POINTER(ctypes.c_float),  # k_cache
    ctypes.POINTER(ctypes.c_float),  # v_cache
    ctypes.POINTER(ctypes.c_float),  # out_token
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_kv_heads
    ctypes.c_int,                    # kv_tokens
    ctypes.c_int,                    # cache_capacity
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
]
lib.attention_forward_decode_head_major_gqa_flash.restype = None

lib.kv_cache_write_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # k_token
    ctypes.POINTER(ctypes.c_float),  # v_token
    ctypes.POINTER(ctypes.c_float),  # k_cache
    ctypes.POINTER(ctypes.c_float),  # v_cache
    ctypes.c_int,                    # num_kv_heads
    ctypes.c_int,                    # token_index
    ctypes.c_int,                    # cache_capacity
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
]
lib.kv_cache_write_head_major.restype = None


def run_flash_prefill_tests(H=8, H_kv=4, T=64, D=64, aligned_head_dim=64, warmup=10, iterations=200):
    np.random.seed(0)
    q_np = np.random.randn(H, T, aligned_head_dim).astype(np.float32)
    k_np = np.random.randn(H_kv, T, aligned_head_dim).astype(np.float32)
    v_np = np.random.randn(H_kv, T, aligned_head_dim).astype(np.float32)
    if D < aligned_head_dim:
        q_np[:, :, D:] = 0.0
        k_np[:, :, D:] = 0.0
        v_np[:, :, D:] = 0.0

    scores_np = np.zeros((H, T, T), dtype=np.float32)
    out_ref_np = np.zeros((H, T, aligned_head_dim), dtype=np.float32)
    out_flash_np = np.zeros_like(out_ref_np)

    report = TestReport(
        test_name="Flash Attention Forward (GQA)",
        dtype="fp32",
        shape=f"H={H},H_kv={H_kv},T={T},D={D},aligned={aligned_head_dim}",
        cpu_info=get_cpu_info()
    )

    def c_ref():
        # Use exact reference (standard expf) for accurate comparison
        lib.attention_forward_causal_head_major_gqa_exact(
            numpy_to_ptr(q_np), numpy_to_ptr(k_np), numpy_to_ptr(v_np),
            numpy_to_ptr(scores_np), numpy_to_ptr(out_ref_np),
            ctypes.c_int(H), ctypes.c_int(H_kv), ctypes.c_int(T),
            ctypes.c_int(D), ctypes.c_int(aligned_head_dim), ctypes.c_int(T)
        )

    def c_flash():
        lib.attention_forward_causal_head_major_gqa_flash(
            numpy_to_ptr(q_np), numpy_to_ptr(k_np), numpy_to_ptr(v_np),
            numpy_to_ptr(out_flash_np),
            ctypes.c_int(H), ctypes.c_int(H_kv), ctypes.c_int(T),
            ctypes.c_int(D), ctypes.c_int(aligned_head_dim)
        )

    c_ref()
    c_flash()

    diff = max_diff(torch.from_numpy(out_flash_np), torch.from_numpy(out_ref_np))

    report.add_result(TestResult(
        name="flash_vs_ref",
        passed=diff <= 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=None,
        kernel_time=time_function(c_flash, warmup=warmup, iterations=iterations, name="C Flash"),
    ))

    return report


def run_flash_prefill_strided_tests(H=8, H_kv=4, T=64, D=64, aligned_head_dim=64,
                                    stride_tokens=128, warmup=10, iterations=200):
    np.random.seed(2)
    q_np = np.random.randn(H, T, aligned_head_dim).astype(np.float32)
    k_np = np.random.randn(H_kv, T, aligned_head_dim).astype(np.float32)
    v_np = np.random.randn(H_kv, T, aligned_head_dim).astype(np.float32)
    if D < aligned_head_dim:
        q_np[:, :, D:] = 0.0
        k_np[:, :, D:] = 0.0
        v_np[:, :, D:] = 0.0

    stride_tokens = max(stride_tokens, T)
    k_strided = np.zeros((H_kv, stride_tokens, aligned_head_dim), dtype=np.float32)
    v_strided = np.zeros_like(k_strided)
    k_strided[:, :T, :] = k_np
    v_strided[:, :T, :] = v_np

    scores_np = np.zeros((H, T, T), dtype=np.float32)
    out_ref_np = np.zeros((H, T, aligned_head_dim), dtype=np.float32)
    out_flash_np = np.zeros_like(out_ref_np)

    report = TestReport(
        test_name="Flash Attention Forward (GQA, Strided KV)",
        dtype="fp32",
        shape=f"H={H},H_kv={H_kv},T={T},D={D},aligned={aligned_head_dim},stride={stride_tokens}",
        cpu_info=get_cpu_info()
    )

    def c_ref():
        lib.attention_forward_causal_head_major_gqa_exact(
            numpy_to_ptr(q_np), numpy_to_ptr(k_np), numpy_to_ptr(v_np),
            numpy_to_ptr(scores_np), numpy_to_ptr(out_ref_np),
            ctypes.c_int(H), ctypes.c_int(H_kv), ctypes.c_int(T),
            ctypes.c_int(D), ctypes.c_int(aligned_head_dim), ctypes.c_int(T)
        )

    def c_flash():
        lib.attention_forward_causal_head_major_gqa_flash_strided(
            numpy_to_ptr(q_np), numpy_to_ptr(k_strided), numpy_to_ptr(v_strided),
            numpy_to_ptr(out_flash_np),
            ctypes.c_int(H), ctypes.c_int(H_kv), ctypes.c_int(T),
            ctypes.c_int(D), ctypes.c_int(aligned_head_dim), ctypes.c_int(stride_tokens)
        )

    c_ref()
    c_flash()

    diff = max_diff(torch.from_numpy(out_flash_np), torch.from_numpy(out_ref_np))

    report.add_result(TestResult(
        name="flash_strided_vs_ref",
        passed=diff <= 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=None,
        kernel_time=time_function(c_flash, warmup=warmup, iterations=iterations, name="C Flash Strided"),
    ))

    return report


def run_kv_cache_decode_tests(H=8, H_kv=4, T=64, D=64, aligned_head_dim=64, warmup=10, iterations=200):
    np.random.seed(1)
    q_np = np.random.randn(H, T, aligned_head_dim).astype(np.float32)
    k_np = np.random.randn(H_kv, T, aligned_head_dim).astype(np.float32)
    v_np = np.random.randn(H_kv, T, aligned_head_dim).astype(np.float32)
    if D < aligned_head_dim:
        q_np[:, :, D:] = 0.0
        k_np[:, :, D:] = 0.0
        v_np[:, :, D:] = 0.0

    scores_np = np.zeros((H, T, T), dtype=np.float32)
    out_ref_np = np.zeros((H, T, aligned_head_dim), dtype=np.float32)
    out_decode_np = np.zeros_like(out_ref_np)

    k_cache = np.zeros((H_kv, T, aligned_head_dim), dtype=np.float32)
    v_cache = np.zeros_like(k_cache)

    report = TestReport(
        test_name="KV Cache Decode Attention (GQA)",
        dtype="fp32",
        shape=f"H={H},H_kv={H_kv},T={T},D={D},aligned={aligned_head_dim}",
        cpu_info=get_cpu_info()
    )

    def c_full_ref():
        # Use exact reference (standard expf) for accurate comparison
        lib.attention_forward_causal_head_major_gqa_exact(
            numpy_to_ptr(q_np), numpy_to_ptr(k_np), numpy_to_ptr(v_np),
            numpy_to_ptr(scores_np), numpy_to_ptr(out_ref_np),
            ctypes.c_int(H), ctypes.c_int(H_kv), ctypes.c_int(T),
            ctypes.c_int(D), ctypes.c_int(aligned_head_dim), ctypes.c_int(T)
        )

    c_full_ref()

    # Fill KV cache token-by-token.
    for t in range(T):
        k_tok = np.ascontiguousarray(k_np[:, t, :], dtype=np.float32)
        v_tok = np.ascontiguousarray(v_np[:, t, :], dtype=np.float32)
        lib.kv_cache_write_head_major(
            numpy_to_ptr(k_tok),
            numpy_to_ptr(v_tok),
            numpy_to_ptr(k_cache),
            numpy_to_ptr(v_cache),
            ctypes.c_int(H_kv),
            ctypes.c_int(t),
            ctypes.c_int(T),
            ctypes.c_int(D),
            ctypes.c_int(aligned_head_dim),
        )

        q_tok = np.ascontiguousarray(q_np[:, t, :], dtype=np.float32)
        out_tok = np.zeros((H, aligned_head_dim), dtype=np.float32)
        lib.attention_forward_decode_head_major_gqa_flash(
            numpy_to_ptr(q_tok),
            numpy_to_ptr(k_cache),
            numpy_to_ptr(v_cache),
            numpy_to_ptr(out_tok),
            ctypes.c_int(H),
            ctypes.c_int(H_kv),
            ctypes.c_int(t + 1),
            ctypes.c_int(T),
            ctypes.c_int(D),
            ctypes.c_int(aligned_head_dim),
        )
        out_decode_np[:, t, :] = out_tok

    diff = max_diff(torch.from_numpy(out_decode_np), torch.from_numpy(out_ref_np))

    # Decode timing for the last token (largest kv_tokens).
    q_tok = np.ascontiguousarray(q_np[:, T - 1, :], dtype=np.float32)
    out_tok = np.zeros((H, aligned_head_dim), dtype=np.float32)

    def c_decode_last():
        lib.attention_forward_decode_head_major_gqa_flash(
            numpy_to_ptr(q_tok),
            numpy_to_ptr(k_cache),
            numpy_to_ptr(v_cache),
            numpy_to_ptr(out_tok),
            ctypes.c_int(H),
            ctypes.c_int(H_kv),
            ctypes.c_int(T),
            ctypes.c_int(T),
            ctypes.c_int(D),
            ctypes.c_int(aligned_head_dim),
        )

    report.add_result(TestResult(
        name="decode_vs_full",
        passed=diff <= 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=None,
        kernel_time=time_function(c_decode_last, warmup=warmup, iterations=iterations, name="C Decode"),
    ))

    return report


if __name__ == "__main__":
    print_system_info()

    flash_report = run_flash_prefill_tests()
    flash_report.print_report()

    decode_report = run_kv_cache_decode_tests()
    decode_report.print_report()

    flash_strided_report = run_flash_prefill_strided_tests()
    flash_strided_report.print_report()

    flash_report_padded = run_flash_prefill_tests(T=32, D=40, aligned_head_dim=64)
    flash_report_padded.print_report()

    decode_report_padded = run_kv_cache_decode_tests(T=32, D=40, aligned_head_dim=64)
    decode_report_padded.print_report()

    if (not flash_report.all_passed()
            or not decode_report.all_passed()
            or not flash_strided_report.all_passed()
            or not flash_report_padded.all_passed()
            or not decode_report_padded.all_passed()):
        raise SystemExit(1)
