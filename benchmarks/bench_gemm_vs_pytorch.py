#!/usr/bin/env python3
"""
GEMM benchmark: CK native/MKL shared libs vs PyTorch.

Used by `make bench_gemm` in nightly CI.
"""
from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path

import numpy as np
import torch


GEMM_SIG = [
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.POINTER(ctypes.c_float),  # bias
    ctypes.POINTER(ctypes.c_float),  # C
    ctypes.c_int,                    # M
    ctypes.c_int,                    # N
    ctypes.c_int,                    # K
]


def _as_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _load_ck_lib(path: str) -> ctypes.CDLL:
    lib = ctypes.cdll.LoadLibrary(path)
    # Use blocked serial kernel for stable ABI across variants.
    lib.gemm_blocked_serial.argtypes = GEMM_SIG
    lib.gemm_blocked_serial.restype = None
    return lib


def _bench_ck(lib: ctypes.CDLL, a: np.ndarray, b: np.ndarray, bias: np.ndarray, reps: int, warmup: int) -> tuple[np.ndarray, float]:
    m, k = a.shape
    n = b.shape[0]
    c = np.zeros((m, n), dtype=np.float32)
    ap, bp, biasp, cp = _as_ptr(a), _as_ptr(b), _as_ptr(bias), _as_ptr(c)

    for _ in range(warmup):
        lib.gemm_blocked_serial(ap, bp, biasp, cp, m, n, k)

    t0 = time.perf_counter()
    for _ in range(reps):
        lib.gemm_blocked_serial(ap, bp, biasp, cp, m, n, k)
    dt = (time.perf_counter() - t0) / float(reps)
    return c, dt


def _bench_torch(a_t: torch.Tensor, b_t: torch.Tensor, bias_t: torch.Tensor, reps: int, warmup: int) -> tuple[torch.Tensor, float]:
    for _ in range(warmup):
        _ = a_t @ b_t.t() + bias_t

    t0 = time.perf_counter()
    out = None
    for _ in range(reps):
        out = a_t @ b_t.t() + bias_t
    dt = (time.perf_counter() - t0) / float(reps)
    assert out is not None
    return out, dt


def _gflops(m: int, n: int, k: int, seconds: float) -> float:
    # GEMM MACs: M*N*K, FLOPs ~= 2*MAC
    return (2.0 * m * n * k) / (seconds * 1e9)


def _resolve_lib(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    return p if p.exists() else None


def main() -> int:
    np.random.seed(1234)
    torch.manual_seed(1234)

    m, n, k = 256, 1024, 512
    reps = int(os.environ.get("CK_BENCH_REPS", "40"))
    warmup = int(os.environ.get("CK_BENCH_WARMUP", "8"))
    tol = float(os.environ.get("CK_BENCH_TOL", "2e-4"))

    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(n, k).astype(np.float32)   # N x K
    bias = np.random.randn(n).astype(np.float32)

    a_t = torch.from_numpy(a)
    b_t = torch.from_numpy(b)
    bias_t = torch.from_numpy(bias)
    torch_out, torch_dt = _bench_torch(a_t, b_t, bias_t, reps=reps, warmup=warmup)
    torch_gf = _gflops(m, n, k, torch_dt)
    torch_np = torch_out.detach().cpu().numpy()

    native_path = _resolve_lib("CK_NATIVE_LIB") or _resolve_lib("CK_LIB_PATH")
    mkl_path = _resolve_lib("CK_MKL_LIB")
    if native_path is None:
        print("ERROR: CK_NATIVE_LIB (or CK_LIB_PATH) not found")
        return 2

    print("=== GEMM Benchmark (CK vs PyTorch) ===")
    print(f"Shape: M={m}, N={n}, K={k} | reps={reps} warmup={warmup}")
    print(f"PyTorch: {torch_gf:.2f} GFLOPS")

    native_lib = _load_ck_lib(str(native_path))
    native_out, native_dt = _bench_ck(native_lib, a, b, bias, reps=reps, warmup=warmup)
    native_gf = _gflops(m, n, k, native_dt)
    native_diff = float(np.max(np.abs(native_out - torch_np)))
    print(f"CK Native: {native_gf:.2f} GFLOPS (max_diff={native_diff:.3e})")

    if not np.isfinite(native_diff) or native_diff > tol:
        print(f"ERROR: CK Native diff too large (tol={tol:.3e})")
        return 3

    best_ck = native_gf
    if mkl_path is not None:
        mkl_lib = _load_ck_lib(str(mkl_path))
        mkl_out, mkl_dt = _bench_ck(mkl_lib, a, b, bias, reps=reps, warmup=warmup)
        mkl_gf = _gflops(m, n, k, mkl_dt)
        mkl_diff = float(np.max(np.abs(mkl_out - torch_np)))
        print(f"CK MKL: {mkl_gf:.2f} GFLOPS (max_diff={mkl_diff:.3e})")
        if not np.isfinite(mkl_diff) or mkl_diff > tol:
            print(f"ERROR: CK MKL diff too large (tol={tol:.3e})")
            return 4
        best_ck = max(best_ck, mkl_gf)
    else:
        print("CK MKL: SKIP (CK_MKL_LIB missing)")

    print(f"Best CK: {best_ck:.2f} GFLOPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

