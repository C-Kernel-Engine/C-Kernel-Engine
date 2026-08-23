#!/usr/bin/env python3
"""Exact PyTorch BF16 full-width RMSNorm storage-contract tests."""

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
PYTORCH_CAPABILITY = torch.backends.cpu.get_cpu_capability()
KERNEL_NAME = "rmsnorm_forward_pytorch_bf16_storage"
KERNEL = getattr(LIB, KERNEL_NAME)
FLOAT_P = ctypes.POINTER(ctypes.c_float)
KERNEL.argtypes = [
    FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
KERNEL.restype = None
STRIDED_KERNEL_NAME = "rmsnorm_forward_strided_pytorch_bf16_storage"
STRIDED_KERNEL = getattr(LIB, STRIDED_KERNEL_NAME)
STRIDED_KERNEL.argtypes = [
    FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
STRIDED_KERNEL.restype = None


def bf16_values(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values).to(torch.bfloat16).float().numpy()


def run_case(tokens: int, dim: int, seed: int, scale: float = 0.03) -> None:
    rng = np.random.default_rng(seed)
    inputs = bf16_values(rng.standard_normal((tokens, dim), dtype=np.float32) * scale)
    gamma = bf16_values(rng.standard_normal(dim, dtype=np.float32) * 0.1 + 1.0)
    actual = np.empty_like(inputs)
    rstd = np.empty(tokens, dtype=np.float32)
    KERNEL(
        inputs.ctypes.data_as(FLOAT_P), gamma.ctypes.data_as(FLOAT_P),
        actual.ctypes.data_as(FLOAT_P), rstd.ctypes.data_as(FLOAT_P),
        tokens, dim, dim, ctypes.c_float(1.0e-6),
    )
    x = torch.from_numpy(inputs).to(torch.bfloat16)
    weight = torch.from_numpy(gamma).to(torch.bfloat16)
    normalized = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1.0e-6)).to(torch.bfloat16)
    expected = (weight * normalized).float().numpy()
    differing = int(np.count_nonzero(actual != expected))
    if differing:
        diff = np.abs(actual - expected)
        raise AssertionError(
            f"PyTorch BF16 RMSNorm mismatch T={tokens} D={dim}: "
            f"differing={differing} max_abs={float(diff.max()):.9g}"
        )
    print(f"T={tokens} D={dim} exact={actual.size}/{actual.size}")


def run_strided_case(
    tokens: int,
    dim: int,
    input_stride: int,
    output_stride: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    # Leave the FP32 carrier deliberately unrounded. Instella's preceding BF16
    # projection writes FP32 accumulators, while PyTorch materializes a BF16
    # linear-output boundary before this RMSNorm.
    inputs = rng.standard_normal((tokens, input_stride), dtype=np.float32)
    gamma = bf16_values(rng.standard_normal(dim, dtype=np.float32) * 0.1 + 1.0)
    actual = np.full((tokens, output_stride), 123.0, dtype=np.float32)
    rstd = np.empty(tokens, dtype=np.float32)
    STRIDED_KERNEL(
        inputs.ctypes.data_as(FLOAT_P), gamma.ctypes.data_as(FLOAT_P),
        actual.ctypes.data_as(FLOAT_P), rstd.ctypes.data_as(FLOAT_P),
        tokens, dim, input_stride, output_stride, ctypes.c_float(1.0e-6),
    )
    x = torch.from_numpy(inputs[:, :dim]).to(torch.bfloat16)
    weight = torch.from_numpy(gamma).to(torch.bfloat16)
    normalized = (
        x.float()
        * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1.0e-6)
    ).to(torch.bfloat16)
    expected = (weight * normalized).float().numpy()
    differing = int(np.count_nonzero(actual[:, :dim] != expected))
    if differing:
        diff = np.abs(actual[:, :dim] - expected)
        raise AssertionError(
            f"strided PyTorch BF16 RMSNorm mismatch T={tokens} D={dim}: "
            f"differing={differing} max_abs={float(diff.max()):.9g}"
        )
    if output_stride > dim and np.count_nonzero(actual[:, dim:]):
        raise AssertionError("strided RMSNorm did not zero output-row padding")
    print(
        f"T={tokens} D={dim} input_stride={input_stride} "
        f"output_stride={output_stride} exact={expected.size}/{expected.size}"
    )


def main() -> int:
    if torch.backends.cpu.get_cpu_capability() not in {"AVX2", "AVX512"}:
        print("PyTorch AVX2-cascade BF16 RMSNorm storage contract [SKIP: AVX2 unavailable]")
        return 0
    cases = (
        (1, 128, 11, 0.03),
        (3, 128, 12, 0.03),
        (1, 4096, 13, 0.03),
        (4, 4096, 14, 0.03),
        # Public practical-width boundary fixture. The former two-accumulator
        # reduction differs from PyTorch in 38 outputs for this row.
        (1, 4096, 30, 0.15),
    )
    for case in cases:
        run_case(*case)
    run_strided_case(1, 512, 544, 512, 31)
    run_strided_case(16, 512, 544, 520, 32)
    print(
        f"PyTorch {PYTORCH_CAPABILITY} full-width BF16 RMSNorm storage contract "
        f"via {KERNEL_NAME} and {STRIDED_KERNEL_NAME}: "
        f"{len(cases) + 2}/{len(cases) + 2} exact"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
