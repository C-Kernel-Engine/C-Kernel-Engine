"""Contracts for llama.cpp-compatible regular Gemma attention."""

import ctypes
import math

import numpy as np

from lib_loader import load_lib


FLOAT_P = ctypes.POINTER(ctypes.c_float)


def _ptr(values: np.ndarray) -> FLOAT_P:
    return values.ctypes.data_as(FLOAT_P)


def _reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    sliding_window: int,
) -> np.ndarray:
    heads, tokens, dim = q.shape
    kv_heads = k.shape[0]
    output = np.zeros_like(q)
    scale = np.float32(1.0 / math.sqrt(dim))
    q_f16 = q.astype(np.float16).astype(np.float32)
    k_f16 = k.astype(np.float16).astype(np.float32)
    v_f16 = v.astype(np.float16).astype(np.float32)
    for head in range(heads):
        kv_head = head * kv_heads // heads
        for query in range(tokens):
            first = max(0, query - sliding_window + 1)
            scores = np.asarray(
                [
                    np.dot(q_f16[head, query], k_f16[kv_head, key])
                    for key in range(first, query + 1)
                ],
                dtype=np.float32,
            )
            scores *= scale
            probabilities = np.exp(scores - np.max(scores), dtype=np.float32)
            probabilities /= np.sum(probabilities, dtype=np.float32)
            probabilities_f16 = probabilities.astype(np.float16).astype(np.float32)
            output[head, query] = probabilities_f16 @ v_f16[kv_head, first : query + 1]
    return output


def test_prefill_uses_declared_scratch_capacity_with_compact_kv_stride() -> None:
    lib = load_lib("libckernel_engine.so")
    kernel = lib.attention_forward_causal_head_major_gqa_llama_regular_strided_sliding_workspace
    kernel.argtypes = [
        FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        FLOAT_P, ctypes.c_size_t, FLOAT_P, ctypes.c_size_t,
        FLOAT_P, ctypes.c_size_t,
    ]
    kernel.restype = None

    rng = np.random.default_rng(20260905)
    heads, kv_heads, tokens, dim = 4, 2, 7, 8
    padded_tokens = 256
    sliding_window = 4
    q = rng.standard_normal((heads, tokens, dim), dtype=np.float32)
    k = rng.standard_normal((kv_heads, tokens, dim), dtype=np.float32)
    v = rng.standard_normal((kv_heads, tokens, dim), dtype=np.float32)
    # Keep the KV heads observably different so an incorrect head stride
    # cannot pass by aliasing equivalent data.
    k[1] += np.float32(3.0)
    v[1] -= np.float32(5.0)
    output = np.zeros_like(q)
    scores = np.full(padded_tokens, np.nan, dtype=np.float32)
    scaled_scores = np.full(padded_tokens, np.nan, dtype=np.float32)
    value_columns = np.full((dim, padded_tokens), np.nan, dtype=np.float32)

    kernel(
        _ptr(q), _ptr(k), _ptr(v), _ptr(output),
        heads, kv_heads, tokens, dim, dim, tokens, sliding_window,
        _ptr(scores), scores.nbytes,
        _ptr(value_columns), value_columns.nbytes,
        _ptr(scaled_scores), scaled_scores.nbytes,
    )

    zeros = np.zeros(padded_tokens - tokens, dtype=np.float32)
    assert np.array_equal(scores[tokens:].view(np.uint32), zeros.view(np.uint32))
    assert np.all(np.isneginf(scaled_scores[tokens:]))
    assert np.all(value_columns[:, tokens:] == 0.0)
    np.testing.assert_allclose(
        output, _reference(q, k, v, sliding_window), rtol=2e-5, atol=2e-5
    )


def test_global_attention_does_not_inherit_512_token_window() -> None:
    lib = load_lib("libckernel_engine.so")
    kernel = lib.attention_forward_causal_head_major_gqa_llama_regular_strided_sliding_workspace
    kernel.argtypes = [
        FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        FLOAT_P, ctypes.c_size_t, FLOAT_P, ctypes.c_size_t,
        FLOAT_P, ctypes.c_size_t,
    ]
    kernel.restype = None

    heads = kv_heads = 1
    tokens, dim, padded_tokens = 520, 8, 768
    q = np.zeros((heads, tokens, dim), dtype=np.float32)
    k = np.zeros((kv_heads, tokens, dim), dtype=np.float32)
    v = np.zeros((kv_heads, tokens, dim), dtype=np.float32)
    v[0, 0, 0] = np.float32(1.0)
    output = np.zeros_like(q)
    scores = np.empty(padded_tokens, dtype=np.float32)
    scaled_scores = np.empty(padded_tokens, dtype=np.float32)
    value_columns = np.empty((dim, padded_tokens), dtype=np.float32)

    kernel(
        _ptr(q), _ptr(k), _ptr(v), _ptr(output),
        heads, kv_heads, tokens, dim, dim, tokens, 0,
        _ptr(scores), scores.nbytes,
        _ptr(value_columns), value_columns.nbytes,
        _ptr(scaled_scores), scaled_scores.nbytes,
    )

    # All causal scores are equal, so the first value contributes 1 / 520.
    # A stale 512-token mask would exclude token zero and produce zero.
    expected = np.float32(np.float16(1.0 / tokens))
    np.testing.assert_allclose(output[0, -1, 0], expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    test_prefill_uses_declared_scratch_capacity_with_compact_kv_stride()
    test_global_attention_does_not_inherit_512_token_window()
    print("llama regular attention contracts: PASS")
