#!/usr/bin/env python3
"""PyTorch oracle for BF16 GEMM with BF16 output storage."""

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
KERNEL = LIB.gemm_nt_bf16_bf16_storage
STORAGE_RANGE_KERNEL = LIB.gemm_nt_bf16_bf16_storage_row_range
STORAGE_PARALLEL_KERNEL = LIB.gemm_nt_bf16_bf16_storage_parallel_dispatch
EXACT_KERNEL = LIB.gemm_nt_bf16
EXACT_RANGE_KERNEL = LIB.gemm_nt_bf16_row_range
NATIVE_KERNEL = LIB.gemm_nt_bf16_native_bf16_storage
AMX_KERNEL = LIB.gemm_nt_bf16_amx_bf16_storage
AMX_WORKSPACE_KERNEL = LIB.gemm_nt_bf16_amx_bf16_storage_workspace
SHAPE_SAFE_KERNEL = LIB.gemm_nt_bf16_prefill_shape_safe_bf16_storage
FLOAT_P = ctypes.POINTER(ctypes.c_float)
UINT16_P = ctypes.POINTER(ctypes.c_uint16)
KERNEL.argtypes = [
    FLOAT_P, ctypes.c_void_p, FLOAT_P, FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
KERNEL.restype = None
STORAGE_RANGE_KERNEL.argtypes = KERNEL.argtypes + [ctypes.c_int, ctypes.c_int]
STORAGE_RANGE_KERNEL.restype = None
STORAGE_PARALLEL_KERNEL.argtypes = KERNEL.argtypes
STORAGE_PARALLEL_KERNEL.restype = None
EXACT_KERNEL.argtypes = KERNEL.argtypes
EXACT_KERNEL.restype = None
EXACT_RANGE_KERNEL.argtypes = KERNEL.argtypes + [ctypes.c_int, ctypes.c_int]
EXACT_RANGE_KERNEL.restype = None
NATIVE_KERNEL.argtypes = KERNEL.argtypes
NATIVE_KERNEL.restype = None
AMX_KERNEL.argtypes = KERNEL.argtypes
AMX_KERNEL.restype = None
AMX_WORKSPACE_KERNEL.argtypes = KERNEL.argtypes + [UINT16_P, ctypes.c_size_t]
AMX_WORKSPACE_KERNEL.restype = None
SHAPE_SAFE_KERNEL.argtypes = KERNEL.argtypes
SHAPE_SAFE_KERNEL.restype = None
AMX_AVAILABLE = LIB.ck_gemm_bf16_amx_available
AMX_AVAILABLE.argtypes = []
AMX_AVAILABLE.restype = ctypes.c_int


def amx_bf16_supported() -> bool:
    return bool(AMX_AVAILABLE())


def bf16_bits(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values).to(torch.bfloat16).view(torch.uint16).numpy()


def bf16_values(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values).to(torch.bfloat16).float().numpy()


def run_case_detailed(m: int, n: int, k: int, seed: int, *, kernel=KERNEL) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = bf16_values(rng.standard_normal((m, k), dtype=np.float32))
    b_values = bf16_values(rng.standard_normal((n, k), dtype=np.float32))
    b = bf16_bits(b_values)
    bias = bf16_values(rng.standard_normal(n, dtype=np.float32))
    actual = np.empty((m, n), dtype=np.float32)
    kernel(
        a.ctypes.data_as(FLOAT_P),
        ctypes.c_void_p(b.ctypes.data),
        bias.ctypes.data_as(FLOAT_P),
        actual.ctypes.data_as(FLOAT_P),
        m, n, k,
    )
    expected = torch.nn.functional.linear(
        torch.from_numpy(a).to(torch.bfloat16),
        torch.from_numpy(b_values).to(torch.bfloat16),
        torch.from_numpy(bias).to(torch.bfloat16),
    ).float().numpy()
    diff = np.abs(actual - expected)
    return {
        "max_abs": float(diff.max(initial=0.0)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "exact_ratio": float(np.count_nonzero(diff == 0.0) / diff.size),
        "different_outputs": int(np.count_nonzero(diff)),
        "output_count": int(diff.size),
    }


def run_amx_workspace_case(m: int, n: int, k: int, seed: int) -> dict[str, float]:
    workspace = np.empty(m * k, dtype=np.uint16)

    def invoke(a, b, bias, output, rows, cols, reduction):
        AMX_WORKSPACE_KERNEL(
            a, b, bias, output, rows, cols, reduction,
            workspace.ctypes.data_as(UINT16_P), ctypes.c_size_t(workspace.nbytes),
        )

    return run_case_detailed(m, n, k, seed, kernel=invoke)


def run_case(m: int, n: int, k: int, seed: int) -> tuple[float, float]:
    metrics = run_case_detailed(m, n, k, seed)
    return metrics["max_abs"], metrics["rmse"]


def main() -> int:
    rng = np.random.default_rng(19)
    m, n, k = 7, 13, 24
    a = np.ascontiguousarray(rng.standard_normal((m, k), dtype=np.float32))
    b = bf16_bits(rng.standard_normal((n, k), dtype=np.float32))
    bias = np.ascontiguousarray(rng.standard_normal(n, dtype=np.float32))
    exact = np.empty((m, n), dtype=np.float32)
    split = np.empty_like(exact)
    args = (
        a.ctypes.data_as(FLOAT_P), ctypes.c_void_p(b.ctypes.data),
        bias.ctypes.data_as(FLOAT_P), exact.ctypes.data_as(FLOAT_P), m, n, k,
    )
    EXACT_KERNEL(*args)
    for begin, end in ((0, 2), (2, 5), (5, m)):
        EXACT_RANGE_KERNEL(
            args[0], args[1], args[2], split.ctypes.data_as(FLOAT_P),
            m, n, k, begin, end,
        )
    if exact.tobytes() != split.tobytes():
        raise AssertionError("BF16 exact GEMM row ranges changed output bytes")

    storage_exact = np.empty_like(exact)
    storage_split = np.empty_like(exact)
    storage_parallel = np.empty_like(exact)
    STORAGE_RANGE_KERNEL(
        args[0], args[1], args[2], storage_exact.ctypes.data_as(FLOAT_P),
        m, n, k, 0, m,
    )
    for begin, end in ((0, 2), (2, 5), (5, m)):
        STORAGE_RANGE_KERNEL(
            args[0], args[1], args[2], storage_split.ctypes.data_as(FLOAT_P),
            m, n, k, begin, end,
        )
    STORAGE_PARALLEL_KERNEL(
        args[0], args[1], args[2], storage_parallel.ctypes.data_as(FLOAT_P),
        m, n, k,
    )
    if storage_exact.tobytes() != storage_split.tobytes():
        raise AssertionError("BF16 storage row ranges changed output bytes")
    if storage_exact.tobytes() != storage_parallel.tobytes():
        raise AssertionError("BF16 storage parallel dispatch changed output bytes")

    cases = [(2, 24, 16, 1), (3, 144, 72, 2), (1, 3456, 1152, 3)]
    for m, n, k, seed in cases:
        max_abs, rmse = run_case(m, n, k, seed)
        if max_abs > 0.5 or rmse > 0.03:
            raise AssertionError(
                f"BF16 GEMM storage mismatch M={m} N={n} K={k}: "
                f"max_abs={max_abs:.9g} rmse={rmse:.9g}"
            )
        print(f"M={m} N={n} K={k} max_abs={max_abs:.9g} rmse={rmse:.9g}")
    tested = len(cases)
    native_cases = [(4, 24, 16, 11), (5, 24, 16, 12), (7, 36, 32, 13)]
    for m, n, k, seed in native_cases:
        metrics = run_case_detailed(m, n, k, seed, kernel=NATIVE_KERNEL)
        if metrics["max_abs"] > 0.5 or metrics["rmse"] > 0.03:
            raise AssertionError(
                f"native BF16 row-tile mismatch M={m} N={n} K={k}: {metrics}"
            )
        tested += 1
        print(
            f"native M={m} N={n} K={k} max_abs={metrics['max_abs']:.9g} "
            f"rmse={metrics['rmse']:.9g}"
        )
    segmented = run_case_detailed(4, 32, 32, 14, kernel=SHAPE_SAFE_KERNEL)
    if segmented["max_abs"] > 0.5 or segmented["rmse"] > 0.03:
        raise AssertionError(f"shape-safe short-segment mismatch: {segmented}")
    tested += 1
    print(
        "shape-safe short segment M=4 N=32 K=32 "
        f"max_abs={segmented['max_abs']:.9g} rmse={segmented['rmse']:.9g}"
    )
    if amx_bf16_supported():
        amx = run_amx_workspace_case(16, 32, 32, 4)
        if amx["max_abs"] != 0.0 or amx["rmse"] != 0.0:
            raise AssertionError(f"AMX BF16 storage mismatch: {amx}")
        tested += 1
        print("AMX M=16 N=32 K=32 exact")
        shape_aligned = run_case_detailed(16, 32, 32, 4, kernel=SHAPE_SAFE_KERNEL)
        if shape_aligned != amx:
            raise AssertionError(
                f"shape-safe aligned provider did not preserve AMX metrics: "
                f"shape_safe={shape_aligned} amx={amx}"
            )
        tested += 1
        print("shape-safe aligned M=16 N=32 K=32 exact")
    else:
        print("AMX M=16 N=32 K=32 SKIP (AMX BF16 unavailable)")
    print(f"BF16 GEMM output storage parity: {tested}/{tested}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
