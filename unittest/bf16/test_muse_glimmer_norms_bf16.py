#!/usr/bin/env python3
"""Exact Muse-Glimmer BF16 normalization, RoPE, and logits contracts."""

from __future__ import annotations

import ctypes
import os
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


def main() -> int:
    if torch.backends.cpu.get_cpu_capability() not in {"AVX2", "AVX512"}:
        print("Muse-Glimmer BF16 normalization contracts [SKIP: AVX2 unavailable]")
        return 0
    run_rmsnorm_case(3, 6656, 91)
    run_qk_case(2, 92)
    run_rope_case(0, 93)
    run_rope_case(2047, 94)
    run_logits_case(95)
    print("Muse-Glimmer BF16 numerical contracts: exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
