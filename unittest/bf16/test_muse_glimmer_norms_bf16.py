#!/usr/bin/env python3
"""Exact Muse-Glimmer BF16 normalization, RoPE, and logits contracts."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = Path(
    os.environ.get("CK_ENGINE_SO")
    or os.environ.get("CK_ENGINE_LIB")
    or ROOT / "build" / "libckernel_engine.so"
)
LIB = ctypes.CDLL(str(LIB_PATH))
FLOAT_P = ctypes.POINTER(ctypes.c_float)
U16_P = ctypes.POINTER(ctypes.c_uint16)


def bf16_values(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values).to(torch.bfloat16).float().numpy()


def configure_kernels() -> tuple[ctypes._CFuncPtr, ...]:
    weighted_args = [
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    unweighted_args = [
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    qk_args = [
        FLOAT_P,
        FLOAT_P,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
    ]
    centered = LIB.rmsnorm_forward_muse_centered_pytorch_bf16_storage
    weighted = LIB.rmsnorm_forward_muse_weighted_pytorch_bf16_storage
    unweighted = LIB.rmsnorm_forward_muse_unweighted_pytorch_bf16_storage
    qk = LIB.qk_norm_forward_muse_unweighted_scaled_pytorch_bf16_storage
    centered.argtypes = weighted_args
    weighted.argtypes = weighted_args
    unweighted.argtypes = unweighted_args
    qk.argtypes = qk_args
    for kernel in (centered, weighted, unweighted, qk):
        kernel.restype = None
    return centered, weighted, unweighted, qk


CENTERED, WEIGHTED, UNWEIGHTED, QK = configure_kernels()

ROPE = LIB.rope_forward_qk_split_direct_muse_pytorch_bf16_storage
ROPE.argtypes = [
    FLOAT_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
ROPE.restype = None
LOGIT_SCALE = LIB.final_logit_scale_muse_pytorch_bf16_storage
LOGIT_SOFTCAP = LIB.final_logit_softcap_muse_pytorch_bf16_storage
for kernel in (LOGIT_SCALE, LOGIT_SOFTCAP):
    kernel.argtypes = [FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    kernel.restype = None

ATTENTION = LIB.attention_forward_causal_head_major_gqa_muse_eager_bf16_storage_sliding
ATTENTION.argtypes = [
    FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    FLOAT_P, ctypes.c_size_t,
    U16_P, ctypes.c_size_t,
    U16_P, ctypes.c_size_t,
    U16_P, ctypes.c_size_t,
    U16_P, ctypes.c_size_t,
]
ATTENTION.restype = None
DECODE_ATTENTION = LIB.attention_forward_decode_head_major_gqa_muse_eager_bf16_storage_sliding
DECODE_ATTENTION.argtypes = [
    FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    FLOAT_P, ctypes.c_size_t,
    U16_P, ctypes.c_size_t,
    U16_P, ctypes.c_size_t,
    U16_P, ctypes.c_size_t,
    U16_P, ctypes.c_size_t,
]
DECODE_ATTENTION.restype = None


def run_rmsnorm_case(tokens: int, dim: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    values = bf16_values(
        rng.standard_normal((tokens, dim), dtype=np.float32) * 0.1
    )
    weight = bf16_values(rng.standard_normal(dim, dtype=np.float32) * 0.1)
    x = torch.from_numpy(values).to(torch.bfloat16)
    w = torch.from_numpy(weight).to(torch.bfloat16)
    mean_squared = x.float().pow(2).mean(-1, keepdim=True) + 1.0e-5
    expected = {
        "centered": (
            (x.float() * torch.rsqrt(mean_squared)) * (1.0 + w.float())
        ).to(x.dtype).float().numpy(),
        "weighted": (
            (x.float() * torch.pow(mean_squared, -0.5)) * w.float()
        ).to(x.dtype).float().numpy(),
        "unweighted": (
            x.float() * torch.pow(mean_squared, -0.5)
        ).to(x.dtype).float().numpy(),
    }

    for name, kernel in (
        ("centered", CENTERED),
        ("weighted", WEIGHTED),
        ("unweighted", UNWEIGHTED),
    ):
        actual = np.empty_like(values)
        rstd = np.empty(tokens, dtype=np.float32)
        if name == "unweighted":
            kernel(
                values.ctypes.data_as(FLOAT_P),
                actual.ctypes.data_as(FLOAT_P),
                rstd.ctypes.data_as(FLOAT_P),
                tokens,
                dim,
                dim,
                ctypes.c_float(1.0e-5),
            )
        else:
            kernel(
                values.ctypes.data_as(FLOAT_P),
                weight.ctypes.data_as(FLOAT_P),
                actual.ctypes.data_as(FLOAT_P),
                rstd.ctypes.data_as(FLOAT_P),
                tokens,
                dim,
                dim,
                ctypes.c_float(1.0e-5),
            )
        np.testing.assert_array_equal(actual, expected[name])


def run_qk_case(tokens: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    num_heads, num_kv_heads, head_dim = 32, 2, 128
    q = bf16_values(
        rng.standard_normal((num_heads, tokens, head_dim), dtype=np.float32) * 0.1
    )
    k = bf16_values(
        rng.standard_normal((num_kv_heads, tokens, head_dim), dtype=np.float32) * 0.1
    )
    expected_q_input = q.copy()
    expected_k_input = k.copy()

    QK(
        q.ctypes.data_as(FLOAT_P),
        k.ctypes.data_as(FLOAT_P),
        num_heads,
        num_kv_heads,
        tokens,
        head_dim,
        ctypes.c_float(1.0e-5),
        ctypes.c_float(3.87),
    )

    def reference(values: np.ndarray, scale: float) -> np.ndarray:
        tensor = torch.from_numpy(values).to(torch.bfloat16)
        normalized = (
            tensor.float()
            * torch.pow(
                tensor.float().pow(2).mean(-1, keepdim=True) + 1.0e-5,
                -0.5,
            )
        ).to(tensor.dtype)
        return (normalized * scale).float().numpy()

    np.testing.assert_array_equal(q, reference(expected_q_input, 3.87))
    np.testing.assert_array_equal(k, reference(expected_k_input, 1.0))


def run_rope_case(pos_offset: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    heads, kv_heads, tokens, head_dim = 32, 2, 3, 128
    q = bf16_values(rng.standard_normal((heads, tokens, head_dim), dtype=np.float32))
    k = bf16_values(rng.standard_normal((kv_heads, tokens, head_dim), dtype=np.float32))
    q_input, k_input = q.copy(), k.copy()
    ROPE(
        q.ctypes.data_as(FLOAT_P), k.ctypes.data_as(FLOAT_P), None,
        0, heads, kv_heads, tokens, head_dim, head_dim, pos_offset,
        head_dim, ctypes.c_float(500000.0),
    )

    positions = torch.arange(pos_offset, pos_offset + tokens, dtype=torch.float32)
    inv_freq = 1.0 / (
        500000.0
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cosine = emb.cos().to(torch.bfloat16)[None, :, :]
    sine = emb.sin().to(torch.bfloat16)[None, :, :]

    def reference(values: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(values).to(torch.bfloat16)
        first, second = tensor.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return (tensor * cosine + rotated * sine).float().numpy()

    np.testing.assert_array_equal(q, reference(q_input))
    np.testing.assert_array_equal(k, reference(k_input))


def run_logits_case(seed: int) -> None:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((2, 4096), dtype=np.float32) * 12.0
    actual = values.copy()
    LOGIT_SCALE(actual.ctypes.data_as(FLOAT_P), 2, 4096, ctypes.c_float(0.19611613513818404))
    LOGIT_SOFTCAP(actual.ctypes.data_as(FLOAT_P), 2, 4096, ctypes.c_float(20.0))

    expected = torch.from_numpy(values).to(torch.bfloat16)
    expected = expected * 0.19611613513818404
    expected = torch.tanh(expected / 20.0) * 20.0
    np.testing.assert_array_equal(actual, expected.float().numpy())


def run_attention_case(tokens: int, sliding_window: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    heads, kv_heads, head_dim = 4, 2, 128
    q = bf16_values(
        rng.standard_normal((heads, tokens, head_dim), dtype=np.float32)
    )
    k = bf16_values(
        rng.standard_normal((kv_heads, tokens, head_dim), dtype=np.float32)
    )
    v = bf16_values(
        rng.standard_normal((kv_heads, tokens, head_dim), dtype=np.float32)
    )
    actual = np.empty_like(q)
    scores = np.empty((tokens, tokens), dtype=np.float32)
    key_rows = np.empty((tokens, head_dim), dtype=np.uint16)
    value_columns = np.empty((head_dim, tokens), dtype=np.uint16)
    gemm_rows = np.empty((tokens, head_dim), dtype=np.uint16)
    gemm_scores = np.empty((tokens, tokens), dtype=np.uint16)
    ATTENTION(
        q.ctypes.data_as(FLOAT_P),
        k.ctypes.data_as(FLOAT_P),
        v.ctypes.data_as(FLOAT_P),
        actual.ctypes.data_as(FLOAT_P),
        heads,
        kv_heads,
        tokens,
        head_dim,
        head_dim,
        tokens,
        sliding_window,
        scores.ctypes.data_as(FLOAT_P), scores.nbytes,
        key_rows.ctypes.data_as(U16_P), key_rows.nbytes,
        value_columns.ctypes.data_as(U16_P), value_columns.nbytes,
        gemm_rows.ctypes.data_as(U16_P), gemm_rows.nbytes,
        gemm_scores.ctypes.data_as(U16_P), gemm_scores.nbytes,
    )

    query = torch.from_numpy(q).to(torch.bfloat16)[None]
    key = torch.from_numpy(k).to(torch.bfloat16).repeat_interleave(
        heads // kv_heads, dim=0
    )[None]
    value = torch.from_numpy(v).to(torch.bfloat16).repeat_interleave(
        heads // kv_heads, dim=0
    )[None]
    scores = torch.matmul(query, key.transpose(2, 3)) * (head_dim ** -0.5)
    mask = torch.full((tokens, tokens), float("-inf"), dtype=torch.bfloat16)
    for row in range(tokens):
        begin = max(0, row + 1 - sliding_window) if sliding_window > 0 else 0
        mask[row, begin : row + 1] = 0
    probabilities = torch.softmax(
        scores + mask[None, None], dim=-1, dtype=torch.float32
    ).to(torch.bfloat16)
    expected = torch.matmul(probabilities, value)[0].float().numpy()
    np.testing.assert_array_equal(actual, expected)


def run_undersized_attention_probe() -> None:
    tokens, heads, kv_heads, head_dim = 2, 4, 2, 128
    q = np.zeros((heads, tokens, head_dim), dtype=np.float32)
    k = np.zeros((kv_heads, tokens, head_dim), dtype=np.float32)
    v = np.zeros_like(k)
    out = np.empty_like(q)
    scores = np.empty((tokens, tokens), dtype=np.float32)
    key_rows = np.empty((tokens, head_dim), dtype=np.uint16)
    value_columns = np.empty((head_dim, tokens), dtype=np.uint16)
    gemm_rows = np.empty((tokens, head_dim), dtype=np.uint16)
    gemm_scores = np.empty((tokens, tokens), dtype=np.uint16)
    ATTENTION(
        q.ctypes.data_as(FLOAT_P), k.ctypes.data_as(FLOAT_P),
        v.ctypes.data_as(FLOAT_P), out.ctypes.data_as(FLOAT_P),
        heads, kv_heads, tokens, head_dim, head_dim, tokens, 0,
        scores.ctypes.data_as(FLOAT_P), scores.nbytes - 1,
        key_rows.ctypes.data_as(U16_P), key_rows.nbytes,
        value_columns.ctypes.data_as(U16_P), value_columns.nbytes,
        gemm_rows.ctypes.data_as(U16_P), gemm_rows.nbytes,
        gemm_scores.ctypes.data_as(U16_P), gemm_scores.nbytes,
    )
    raise AssertionError("undersized Muse workspace was accepted")


def run_decode_stride_case(seed: int) -> None:
    rng = np.random.default_rng(seed)
    heads, kv_heads, live_tokens, capacity, head_dim = 4, 2, 3, 7, 128
    q = bf16_values(rng.standard_normal((heads, head_dim), dtype=np.float32))
    k = bf16_values(
        rng.standard_normal((kv_heads, capacity, head_dim), dtype=np.float32)
    )
    v = bf16_values(
        rng.standard_normal((kv_heads, capacity, head_dim), dtype=np.float32)
    )
    actual = np.empty_like(q)
    scores = np.empty(live_tokens, dtype=np.float32)
    key_rows = np.empty((live_tokens, head_dim), dtype=np.uint16)
    value_columns = np.empty((head_dim, live_tokens), dtype=np.uint16)
    gemm_rows = np.empty(head_dim, dtype=np.uint16)
    gemm_scores = np.empty(live_tokens, dtype=np.uint16)
    DECODE_ATTENTION(
        q.ctypes.data_as(FLOAT_P), k.ctypes.data_as(FLOAT_P),
        v.ctypes.data_as(FLOAT_P), actual.ctypes.data_as(FLOAT_P),
        heads, kv_heads, live_tokens, capacity, head_dim, head_dim, 2048,
        scores.ctypes.data_as(FLOAT_P), scores.nbytes,
        key_rows.ctypes.data_as(U16_P), key_rows.nbytes,
        value_columns.ctypes.data_as(U16_P), value_columns.nbytes,
        gemm_rows.ctypes.data_as(U16_P), gemm_rows.nbytes,
        gemm_scores.ctypes.data_as(U16_P), gemm_scores.nbytes,
    )
    query = torch.from_numpy(q).to(torch.bfloat16)[None, :, None, :]
    key = torch.from_numpy(k[:, :live_tokens]).to(torch.bfloat16)
    value = torch.from_numpy(v[:, :live_tokens]).to(torch.bfloat16)
    key = key.repeat_interleave(heads // kv_heads, dim=0)[None]
    value = value.repeat_interleave(heads // kv_heads, dim=0)[None]
    probabilities = torch.softmax(
        torch.matmul(query, key.transpose(2, 3)) * (head_dim ** -0.5),
        dim=-1,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    expected = torch.matmul(probabilities, value)[0, :, 0].float().numpy()
    np.testing.assert_array_equal(actual, expected)


def main() -> int:
    if os.environ.get("CK_MUSE_UNDERSIZED_PROBE") == "1":
        run_undersized_attention_probe()
        return 1
    if torch.backends.cpu.get_cpu_capability() not in {"AVX2", "AVX512"}:
        print("Muse-Glimmer BF16 normalization contracts [SKIP: AVX2 unavailable]")
        return 0
    run_rmsnorm_case(3, 6656, 91)
    run_qk_case(2, 92)
    run_qk_case(69, 98)
    run_rope_case(0, 93)
    run_rope_case(2047, 94)
    run_logits_case(95)
    run_attention_case(4, 3, 96)
    run_decode_stride_case(97)
    probe_env = dict(os.environ)
    probe_env["CK_MUSE_UNDERSIZED_PROBE"] = "1"
    probe = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        env=probe_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert probe.returncode != 0
    assert "undersized planner-owned workspace" in probe.stderr
    print("Muse-Glimmer BF16 numerical contracts: exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
