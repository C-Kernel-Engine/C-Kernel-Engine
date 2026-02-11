"""
Sliding-window attention contract test (no pytest dependency).

This validates both exposed sliding kernels:
  - prefill: attention_forward_causal_head_major_gqa_flash_strided_sliding
  - decode:  attention_forward_decode_head_major_gqa_flash_sliding

The test compares C kernel outputs against a NumPy reference implementation.
"""

import ctypes
import math
import sys

import numpy as np

from lib_loader import load_lib


TOL = 1e-4


def _ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _ref_prefill(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    num_heads: int,
    num_kv_heads: int,
    num_tokens: int,
    head_dim: int,
    sliding_window: int,
) -> np.ndarray:
    out = np.zeros((num_heads, num_tokens, head_dim), dtype=np.float32)
    scale = 1.0 / math.sqrt(head_dim)

    for h in range(num_heads):
        kv_h = (h * num_kv_heads) // num_heads
        for i in range(num_tokens):
            start = 0
            if sliding_window > 0:
                start = max(0, i - sliding_window + 1)
            logits = np.empty(i - start + 1, dtype=np.float32)
            for idx, j in enumerate(range(start, i + 1)):
                logits[idx] = float(np.dot(q[h, i, :head_dim], k[kv_h, j, :head_dim])) * scale
            logits -= np.max(logits)
            probs = np.exp(logits)
            probs /= np.sum(probs)
            acc = np.zeros(head_dim, dtype=np.float32)
            for idx, j in enumerate(range(start, i + 1)):
                acc += probs[idx] * v[kv_h, j, :head_dim]
            out[h, i, :] = acc

    return out


def _ref_decode(
    q_token: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    num_heads: int,
    num_kv_heads: int,
    kv_tokens: int,
    head_dim: int,
    sliding_window: int,
) -> np.ndarray:
    out = np.zeros((num_heads, head_dim), dtype=np.float32)
    scale = 1.0 / math.sqrt(head_dim)
    pos = kv_tokens - 1

    for h in range(num_heads):
        kv_h = (h * num_kv_heads) // num_heads
        start = 0
        if sliding_window > 0:
            start = max(0, pos - sliding_window + 1)
        logits = np.empty(pos - start + 1, dtype=np.float32)
        for idx, j in enumerate(range(start, pos + 1)):
            logits[idx] = float(np.dot(q_token[h, :head_dim], k_cache[kv_h, j, :head_dim])) * scale
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)
        acc = np.zeros(head_dim, dtype=np.float32)
        for idx, j in enumerate(range(start, pos + 1)):
            acc += probs[idx] * v_cache[kv_h, j, :head_dim]
        out[h, :] = acc

    return out


def _print_result(name: str, diff: float, tol: float, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"{name}  max_diff={diff:.3e}  tol={tol:.1e}  [{status}]")


def main() -> int:
    np.random.seed(123)

    lib = load_lib("libckernel_engine.so")
    lib.attention_forward_causal_head_major_gqa_flash_strided_sliding.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # q
        ctypes.POINTER(ctypes.c_float),  # k
        ctypes.POINTER(ctypes.c_float),  # v
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,  # num_heads
        ctypes.c_int,  # num_kv_heads
        ctypes.c_int,  # num_tokens
        ctypes.c_int,  # head_dim
        ctypes.c_int,  # aligned_head_dim
        ctypes.c_int,  # kv_stride_tokens
        ctypes.c_int,  # sliding_window
    ]
    lib.attention_forward_causal_head_major_gqa_flash_strided_sliding.restype = None

    lib.attention_forward_decode_head_major_gqa_flash_sliding.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # q_token
        ctypes.POINTER(ctypes.c_float),  # k_cache
        ctypes.POINTER(ctypes.c_float),  # v_cache
        ctypes.POINTER(ctypes.c_float),  # out_token
        ctypes.c_int,  # num_heads
        ctypes.c_int,  # num_kv_heads
        ctypes.c_int,  # kv_tokens
        ctypes.c_int,  # cache_capacity
        ctypes.c_int,  # head_dim
        ctypes.c_int,  # aligned_head_dim
        ctypes.c_int,  # sliding_window
    ]
    lib.attention_forward_decode_head_major_gqa_flash_sliding.restype = None

    num_heads = 8
    num_kv_heads = 2
    num_tokens = 12
    head_dim = 32
    aligned_head_dim = 32
    sliding_window = 4

    q = (np.random.randn(num_heads, num_tokens, aligned_head_dim) * 0.1).astype(np.float32)
    k = (np.random.randn(num_kv_heads, num_tokens, aligned_head_dim) * 0.1).astype(np.float32)
    v = (np.random.randn(num_kv_heads, num_tokens, aligned_head_dim) * 0.1).astype(np.float32)
    out = np.zeros((num_heads, num_tokens, aligned_head_dim), dtype=np.float32)

    lib.attention_forward_causal_head_major_gqa_flash_strided_sliding(
        _ptr(q),
        _ptr(k),
        _ptr(v),
        _ptr(out),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(num_tokens),
        ctypes.c_int(head_dim),
        ctypes.c_int(aligned_head_dim),
        ctypes.c_int(num_tokens),
        ctypes.c_int(sliding_window),
    )

    ref_out = _ref_prefill(
        q, k, v, num_heads, num_kv_heads, num_tokens, head_dim, sliding_window
    )
    prefill_diff = float(np.max(np.abs(out[:, :, :head_dim] - ref_out)))
    prefill_ok = prefill_diff <= TOL
    _print_result("prefill_sliding", prefill_diff, TOL, prefill_ok)

    out_causal = np.zeros_like(out)
    lib.attention_forward_causal_head_major_gqa_flash_strided_sliding(
        _ptr(q),
        _ptr(k),
        _ptr(v),
        _ptr(out_causal),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(num_tokens),
        ctypes.c_int(head_dim),
        ctypes.c_int(aligned_head_dim),
        ctypes.c_int(num_tokens),
        ctypes.c_int(-1),
    )
    ref_causal = _ref_prefill(
        q, k, v, num_heads, num_kv_heads, num_tokens, head_dim, -1
    )
    causal_diff = float(np.max(np.abs(out_causal[:, :, :head_dim] - ref_causal)))
    causal_ok = causal_diff <= TOL
    _print_result("prefill_no_window", causal_diff, TOL, causal_ok)

    cache_capacity = 16
    kv_tokens = 12
    q_decode = (np.random.randn(num_heads, aligned_head_dim) * 0.1).astype(np.float32)
    k_cache = (np.random.randn(num_kv_heads, cache_capacity, aligned_head_dim) * 0.1).astype(
        np.float32
    )
    v_cache = (np.random.randn(num_kv_heads, cache_capacity, aligned_head_dim) * 0.1).astype(
        np.float32
    )
    out_decode = np.zeros((num_heads, aligned_head_dim), dtype=np.float32)

    lib.attention_forward_decode_head_major_gqa_flash_sliding(
        _ptr(q_decode),
        _ptr(k_cache),
        _ptr(v_cache),
        _ptr(out_decode),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(kv_tokens),
        ctypes.c_int(cache_capacity),
        ctypes.c_int(head_dim),
        ctypes.c_int(aligned_head_dim),
        ctypes.c_int(sliding_window),
    )

    ref_decode = _ref_decode(
        q_decode,
        k_cache,
        v_cache,
        num_heads,
        num_kv_heads,
        kv_tokens,
        head_dim,
        sliding_window,
    )
    decode_diff = float(np.max(np.abs(out_decode[:, :head_dim] - ref_decode)))
    decode_ok = decode_diff <= TOL
    _print_result("decode_sliding", decode_diff, TOL, decode_ok)

    all_ok = prefill_ok and causal_ok and decode_ok
    if all_ok:
        print("Sliding window kernel contract: PASS")
        return 0
    print("Sliding window kernel contract: FAIL")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
