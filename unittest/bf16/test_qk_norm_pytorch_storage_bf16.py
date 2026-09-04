#!/usr/bin/env python3
"""Exact PyTorch BF16 per-head Q/K RMSNorm contract tests."""

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
KERNEL = LIB.qk_norm_forward_pytorch_bf16_storage
QWEN4_KERNEL = LIB.qk_norm_forward_qwen4_pytorch_bf16_storage
FLOAT_P = ctypes.POINTER(ctypes.c_float)
KERNEL_ARGS = [
    FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
KERNEL.argtypes = KERNEL_ARGS
KERNEL.restype = None
QWEN4_KERNEL.argtypes = KERNEL_ARGS
QWEN4_KERNEL.restype = None


def bf16_values(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values).to(torch.bfloat16).float().numpy()


def pytorch_qk_norm(values: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(values).to(torch.bfloat16)
    weight = torch.from_numpy(gamma).to(torch.bfloat16)
    normalized = (
        x.float()
        * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1.0e-6)
    ).to(torch.bfloat16)
    return (weight * normalized).float().numpy()


def pytorch_qwen4_qk_norm(values: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(values).to(torch.bfloat16)
    weight = torch.from_numpy(gamma).float()
    normalized = x.float() * torch.rsqrt(
        x.float().pow(2).mean(-1, keepdim=True) + 1.0e-6
    )
    return (normalized * weight).to(torch.bfloat16).float().numpy()


def run_case(num_heads: int, num_kv_heads: int, tokens: int, seed: int) -> None:
    head_dim = 128
    rng = np.random.default_rng(seed)
    q = bf16_values(
        rng.standard_normal((num_heads, tokens, head_dim), dtype=np.float32) * 0.03
    )
    k = bf16_values(
        rng.standard_normal((num_kv_heads, tokens, head_dim), dtype=np.float32) * 0.03
    )
    q_gamma = bf16_values(rng.standard_normal(head_dim, dtype=np.float32) * 0.1 + 1.0)
    k_gamma = bf16_values(rng.standard_normal(head_dim, dtype=np.float32) * 0.1 + 1.0)
    expected_q = pytorch_qk_norm(q, q_gamma)
    expected_k = pytorch_qk_norm(k, k_gamma)

    KERNEL(
        q.ctypes.data_as(FLOAT_P),
        k.ctypes.data_as(FLOAT_P),
        q_gamma.ctypes.data_as(FLOAT_P),
        k_gamma.ctypes.data_as(FLOAT_P),
        num_heads,
        num_kv_heads,
        tokens,
        head_dim,
        ctypes.c_float(1.0e-6),
    )

    for name, actual, expected in (("q", q, expected_q), ("k", k, expected_k)):
        differing = int(np.count_nonzero(actual != expected))
        if differing:
            diff = np.abs(actual - expected)
            worst = np.unravel_index(int(np.argmax(diff)), diff.shape)
            raise AssertionError(
                f"PyTorch BF16 Q/K RMSNorm mismatch {name}: "
                f"H={num_heads} KV={num_kv_heads} T={tokens} D={head_dim} "
                f"differing={differing} max_abs={float(diff[worst]):.9g} "
                f"worst={worst}"
            )
    print(
        f"H={num_heads} KV={num_kv_heads} T={tokens} D={head_dim} "
        f"exact={q.size + k.size}/{q.size + k.size}"
    )


def run_qwen4_case(tokens: int, seed: int) -> None:
    num_heads, num_kv_heads, head_dim = 8, 2, 256
    rng = np.random.default_rng(seed)
    q = bf16_values(
        rng.standard_normal((num_heads, tokens, head_dim), dtype=np.float32) * 0.03
    )
    k = bf16_values(
        rng.standard_normal((num_kv_heads, tokens, head_dim), dtype=np.float32) * 0.03
    )
    q_gamma = 1.0 + bf16_values(
        rng.standard_normal(head_dim, dtype=np.float32) * 0.1
    )
    k_gamma = 1.0 + bf16_values(
        rng.standard_normal(head_dim, dtype=np.float32) * 0.1
    )
    expected_q = pytorch_qwen4_qk_norm(q, q_gamma)
    expected_k = pytorch_qwen4_qk_norm(k, k_gamma)

    QWEN4_KERNEL(
        q.ctypes.data_as(FLOAT_P),
        k.ctypes.data_as(FLOAT_P),
        q_gamma.ctypes.data_as(FLOAT_P),
        k_gamma.ctypes.data_as(FLOAT_P),
        num_heads,
        num_kv_heads,
        tokens,
        head_dim,
        ctypes.c_float(1.0e-6),
    )

    np.testing.assert_array_equal(q, expected_q)
    np.testing.assert_array_equal(k, expected_k)
    print(
        f"Qwen4 H={num_heads} KV={num_kv_heads} T={tokens} D={head_dim} "
        f"exact={q.size + k.size}/{q.size + k.size}"
    )


def main() -> int:
    if torch.backends.cpu.get_cpu_capability() not in {"AVX2", "AVX512"}:
        print("PyTorch BF16 Q/K RMSNorm contract [SKIP: AVX2 unavailable]")
        return 0
    run_case(4, 2, 3, 41)
    run_case(32, 8, 177, 42)
    run_qwen4_case(1, 43)
    run_qwen4_case(64, 44)
    print("PyTorch BF16 Q/K RMSNorm contracts: 4/4 exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
