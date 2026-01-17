#!/usr/bin/env python3
"""
Unit test for fused_mlp_swiglu_prefill_w1w2_quant.

Validates the fused kernel against a baseline composed of:
- quantize_row_q8_0
- gemm_nt_q5_0_q8_0 (W1 gate/up)
- SwiGLU
- quantize_row_q8_k
- gemm_nt_q4_k_q8_k or gemm_nt_q6_k_q8_k (W2 down)
"""
import ctypes
import os
import sys

import numpy as np
import torch

# Add unittest dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib_loader import load_lib
from test_utils import TestReport, TestResult, max_diff

CK_DT_Q4_K = 7
CK_DT_Q6_K = 8
CK_DT_Q8_0 = 9
CK_DT_Q8_K = 10
CK_DT_Q5_0 = 11


def row_bytes(dt: int, n_elements: int) -> int:
    if dt == CK_DT_Q5_0:
        block_size, block_bytes = 32, 22
    elif dt == CK_DT_Q8_0:
        block_size, block_bytes = 32, 34
    elif dt == CK_DT_Q4_K:
        block_size, block_bytes = 256, 144
    elif dt == CK_DT_Q6_K:
        block_size, block_bytes = 256, 210
    elif dt == CK_DT_Q8_K:
        block_size, block_bytes = 256, 292
    else:
        raise ValueError(f"unsupported dt={dt}")
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * block_bytes


def ptr_f32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def ptr_void(arr: np.ndarray):
    return ctypes.c_void_p(arr.ctypes.data)

def make_q5_0_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes(CK_DT_Q5_0, cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + 31) // 32
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * 22
            data[off:off + 2] = scale_bytes
    return data


def make_q4_k_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes(CK_DT_Q4_K, cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    dmin_bytes = np.frombuffer(np.float16(0.0).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + 255) // 256
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * 144
            data[off:off + 2] = scale_bytes
            data[off + 2:off + 4] = dmin_bytes
    return data


def make_q6_k_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes(CK_DT_Q6_K, cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    blocks_per_row = (cols + 255) // 256
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * 210
            data[off + 208:off + 210] = scale_bytes
    return data


def run_case(w2_dt: int, name: str) -> TestResult:
    lib = load_lib("libckernel_engine.so")

    lib.quantize_row_q8_0.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.quantize_row_q8_0.restype = None

    lib.quantize_row_q8_k.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.quantize_row_q8_k.restype = None

    lib.gemm_nt_q5_0_q8_0.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemm_nt_q5_0_q8_0.restype = None

    lib.gemm_nt_q4_k_q8_k.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemm_nt_q4_k_q8_k.restype = None

    lib.gemm_nt_q6_k_q8_k.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemm_nt_q6_k_q8_k.restype = None

    try:
        lib.fused_mlp_swiglu_prefill_w1w2_quant.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.fused_mlp_swiglu_prefill_w1w2_quant.restype = None

        lib.fused_mlp_swiglu_prefill_w1w2_quant_scratch_size.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.fused_mlp_swiglu_prefill_w1w2_quant_scratch_size.restype = ctypes.c_size_t
    except AttributeError:
        return TestResult(
            name=name,
            passed=False,
            max_diff=0.0,
            tolerance=0.0,
        )

    tokens = 4
    embed_dim = 128
    aligned_embed_dim = 128
    intermediate = 512
    aligned_intermediate = 512

    np.random.seed(123)
    x = np.random.randn(tokens, aligned_embed_dim).astype(np.float32)

    w1_bytes = make_q5_0_weights(2 * aligned_intermediate, aligned_embed_dim)
    if w2_dt == CK_DT_Q4_K:
        w2_bytes = make_q4_k_weights(aligned_embed_dim, aligned_intermediate)
    else:
        w2_bytes = make_q6_k_weights(aligned_embed_dim, aligned_intermediate)

    # Baseline
    q8_row_bytes = row_bytes(CK_DT_Q8_0, aligned_embed_dim)
    q8_rows = np.zeros(tokens * q8_row_bytes, dtype=np.uint8)
    for t in range(tokens):
        lib.quantize_row_q8_0(ptr_f32(x[t]),
                              ctypes.c_void_p(q8_rows.ctypes.data + t * q8_row_bytes),
                              ctypes.c_int(aligned_embed_dim))

    gate = np.zeros((tokens, aligned_intermediate), dtype=np.float32)
    up = np.zeros((tokens, aligned_intermediate), dtype=np.float32)

    w1_row_bytes = row_bytes(CK_DT_Q5_0, aligned_embed_dim)
    w1_gate_ptr = ctypes.c_void_p(w1_bytes.ctypes.data)
    w1_up_ptr = ctypes.c_void_p(w1_bytes.ctypes.data + aligned_intermediate * w1_row_bytes)

    lib.gemm_nt_q5_0_q8_0(ptr_void(q8_rows), w1_gate_ptr, None,
                          ptr_f32(gate),
                          ctypes.c_int(tokens), ctypes.c_int(aligned_intermediate), ctypes.c_int(aligned_embed_dim))
    lib.gemm_nt_q5_0_q8_0(ptr_void(q8_rows), w1_up_ptr, None,
                          ptr_f32(up),
                          ctypes.c_int(tokens), ctypes.c_int(aligned_intermediate), ctypes.c_int(aligned_embed_dim))

    hidden = gate / (1.0 + np.exp(-gate)) * up

    q8k_row_bytes = row_bytes(CK_DT_Q8_K, aligned_intermediate)
    q8k_rows = np.zeros(tokens * q8k_row_bytes, dtype=np.uint8)
    for t in range(tokens):
        lib.quantize_row_q8_k(ptr_f32(hidden[t]),
                              ctypes.c_void_p(q8k_rows.ctypes.data + t * q8k_row_bytes),
                              ctypes.c_int(aligned_intermediate))

    out_ref = np.zeros((tokens, aligned_embed_dim), dtype=np.float32)
    if w2_dt == CK_DT_Q4_K:
        lib.gemm_nt_q4_k_q8_k(ptr_void(q8k_rows), ptr_void(w2_bytes), None,
                              ptr_f32(out_ref),
                              ctypes.c_int(tokens), ctypes.c_int(aligned_embed_dim), ctypes.c_int(aligned_intermediate))
    else:
        lib.gemm_nt_q6_k_q8_k(ptr_void(q8k_rows), ptr_void(w2_bytes), None,
                              ptr_f32(out_ref),
                              ctypes.c_int(tokens), ctypes.c_int(aligned_embed_dim), ctypes.c_int(aligned_intermediate))

    # Fused
    out_fused = np.zeros_like(out_ref)
    scratch_size = lib.fused_mlp_swiglu_prefill_w1w2_quant_scratch_size(
        ctypes.c_int(aligned_embed_dim), ctypes.c_int(aligned_intermediate)
    )
    scratch = np.zeros(scratch_size, dtype=np.uint8)

    lib.fused_mlp_swiglu_prefill_w1w2_quant(
        ptr_f32(x),
        ptr_void(w1_bytes),
        None,
        ctypes.c_int(CK_DT_Q5_0),
        ptr_void(w2_bytes),
        None,
        ctypes.c_int(w2_dt),
        ptr_f32(out_fused),
        ctypes.c_int(tokens),
        ctypes.c_int(embed_dim),
        ctypes.c_int(aligned_embed_dim),
        ctypes.c_int(intermediate),
        ctypes.c_int(aligned_intermediate),
        ptr_void(scratch),
    )

    max_d = max_diff(torch.from_numpy(out_ref), torch.from_numpy(out_fused))
    tol = 1e-3

    return TestResult(
        name=name,
        passed=max_d <= tol,
        max_diff=max_d,
        tolerance=tol,
    )


def main() -> int:
    report = TestReport("prefill_fused_mlp_quant")
    report.add_result(run_case(CK_DT_Q4_K, "fused_mlp_w2_q4_k"))
    report.add_result(run_case(CK_DT_Q6_K, "fused_mlp_w2_q6_k"))
    report.print_report()
    return 0 if report.all_passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
