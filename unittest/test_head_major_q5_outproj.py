#!/usr/bin/env python3
"""
Direct parity and benchmark coverage for ck_gemm_nt_head_major_q5_0.

This kernel is a CK-specific layout-aware building block. The right oracle is:
  1. flatten_head_major(attn_out) + ck_gemm_nt_quant() for numeric parity
  2. the same flattened baseline for end-to-end performance comparison
"""

import argparse
import ctypes
import struct
import time

import numpy as np

from lib_loader import load_lib


CK_DT_Q5_0 = 11
QK5_0 = 32
BLOCK_Q5_0_SIZE = 22


def ptr_f32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def ptr_void(arr: np.ndarray):
    return ctypes.c_void_p(arr.ctypes.data)


def row_bytes_q5_0(n_elements: int) -> int:
    assert n_elements % QK5_0 == 0
    return (n_elements // QK5_0) * BLOCK_Q5_0_SIZE


def make_q5_0_weights(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
    rb = row_bytes_q5_0(cols)
    data = np.random.randint(0, 256, size=rows * rb, dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(scale).tobytes(), dtype=np.uint8)
    blocks_per_row = cols // QK5_0
    for r in range(rows):
        base = r * rb
        for b in range(blocks_per_row):
            off = base + b * BLOCK_Q5_0_SIZE
            data[off:off + 2] = scale_bytes
    return data


def flatten_head_major(attn_out: np.ndarray, tokens: int, num_heads: int, head_dim: int) -> np.ndarray:
    return np.transpose(attn_out, (1, 0, 2)).reshape(tokens, num_heads * head_dim).copy()


def dequant_q5_0_matrix(weights_q5: np.ndarray, rows: int, cols: int) -> np.ndarray:
    blocks_per_row = cols // QK5_0
    rb = row_bytes_q5_0(cols)
    buf = memoryview(weights_q5)
    out = np.zeros((rows, cols), dtype=np.float32)

    for row in range(rows):
        row_base = row * rb
        for b in range(blocks_per_row):
            off = row_base + b * BLOCK_Q5_0_SIZE
            d = float(np.frombuffer(buf[off:off + 2], dtype=np.float16)[0])
            qh = struct.unpack_from("<I", buf, off + 2)[0]
            qs_off = off + 6
            for j in range(QK5_0 // 2):
                packed = int(weights_q5[qs_off + j])
                lo = packed & 0x0F
                hi = packed >> 4
                xh0 = ((qh >> (j + 0)) << 4) & 0x10
                xh1 = ((qh >> (j + 12))) & 0x10
                out[row, b * QK5_0 + j] = d * float((lo | xh0) - 16)
                out[row, b * QK5_0 + j + 16] = d * float((hi | xh1) - 16)
    return out


def time_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000.0 / float(iterations)


def run_case(lib, label: str, tokens: int, num_heads: int, head_dim: int,
             tol: float, check_numpy: bool, warmup: int, iterations: int,
             max_slowdown: float | None) -> bool:
    embed_dim = num_heads * head_dim
    assert head_dim % QK5_0 == 0
    assert embed_dim % QK5_0 == 0

    attn_out = np.random.randn(num_heads, tokens, head_dim).astype(np.float32)
    bias = np.random.randn(embed_dim).astype(np.float32)
    weights_q5 = make_q5_0_weights(embed_dim, embed_dim)

    out_direct = np.empty((tokens, embed_dim), dtype=np.float32)
    out_flat = np.empty((tokens, embed_dim), dtype=np.float32)

    flat = flatten_head_major(attn_out, tokens, num_heads, head_dim)

    def run_direct():
        lib.ck_gemm_nt_head_major_q5_0(
            ptr_f32(attn_out),
            ptr_void(weights_q5),
            ptr_f32(bias),
            ptr_f32(out_direct),
            ctypes.c_int(tokens),
            ctypes.c_int(embed_dim),
            ctypes.c_int(num_heads),
            ctypes.c_int(head_dim),
        )

    def run_flatten_baseline():
        flat_local = flatten_head_major(attn_out, tokens, num_heads, head_dim)
        lib.ck_gemm_nt_quant(
            ptr_f32(flat_local),
            ptr_void(weights_q5),
            ptr_f32(bias),
            ptr_f32(out_flat),
            ctypes.c_int(tokens),
            ctypes.c_int(embed_dim),
            ctypes.c_int(embed_dim),
            ctypes.c_int(CK_DT_Q5_0),
        )

    run_direct()
    run_flatten_baseline()

    diff_flat = float(np.max(np.abs(out_direct - out_flat)))
    print(
        f"\n--- Testing ck_gemm_nt_head_major_q5_0 ({label}: tokens={tokens}, "
        f"heads={num_heads}, dim={head_dim}) ---"
    )
    print(f"Flatten+GEMM max diff: {diff_flat:.2e}")

    passed = diff_flat <= tol

    if check_numpy:
        weights_f32 = dequant_q5_0_matrix(weights_q5, embed_dim, embed_dim)
        ref = flat @ weights_f32.T + bias
        diff_numpy = float(np.max(np.abs(out_direct - ref)))
        print(f"NumPy oracle max diff: {diff_numpy:.2e}")
        passed = passed and diff_numpy <= tol

    direct_ms = time_ms(run_direct, warmup=warmup, iterations=iterations)
    baseline_ms = time_ms(run_flatten_baseline, warmup=warmup, iterations=iterations)
    speedup = baseline_ms / direct_ms if direct_ms > 0.0 else 0.0
    print(f"Head-major kernel: {direct_ms:.3f} ms")
    print(f"Flatten+GEMM:      {baseline_ms:.3f} ms")
    print(f"Speedup:           {speedup:.2f}x")

    if max_slowdown is not None and direct_ms > baseline_ms * max_slowdown:
        print(
            f"Performance regression: direct kernel is "
            f"{direct_ms / baseline_ms:.2f}x slower than flatten+GEMM "
            f"(limit {max_slowdown:.2f}x)"
        )
        passed = False

    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a smaller, faster coverage set")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-slowdown", type=float, default=1.35)
    args = parser.parse_args()

    np.random.seed(42)

    lib = load_lib("libckernel_engine.so")
    lib.ck_gemm_nt_head_major_q5_0.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.ck_gemm_nt_head_major_q5_0.restype = None

    lib.ck_gemm_nt_quant.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.ck_gemm_nt_quant.restype = None

    cases = [
        ("small", 4, 4, 32, 1e-4, True),
        ("prefill", 16, 8, 64, 1e-4, False),
    ]
    if not args.quick:
        cases.append(("larger", 32, 16, 64, 2e-4, False))

    ok = True
    for label, tokens, num_heads, head_dim, tol, check_numpy in cases:
        ok = run_case(
            lib,
            label,
            tokens,
            num_heads,
            head_dim,
            tol,
            check_numpy,
            warmup=args.warmup,
            iterations=args.iters,
            max_slowdown=args.max_slowdown,
        ) and ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
