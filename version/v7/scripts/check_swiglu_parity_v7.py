#!/usr/bin/env python3
"""Quick CK vs PyTorch SwiGLU parity probe (fast vs strict kernel math)."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np


try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"torch is required (activate .venv): {exc}")


def _load_lib(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path.resolve()))
    lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
    lib.ck_set_strict_parity.restype = None
    lib.swiglu_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.swiglu_forward.restype = None
    lib.swiglu_backward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.swiglu_backward.restype = None
    return lib


def _torch_swiglu(inp: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(inp.astype(np.float32))
    dim = x.shape[1] // 2
    return (F.silu(x[:, :dim]) * x[:, dim:]).cpu().numpy()


def _torch_swiglu_backward(inp: np.ndarray, dy: np.ndarray) -> np.ndarray:
    x = torch.tensor(inp.astype(np.float32), requires_grad=True)
    dy_t = torch.tensor(dy.astype(np.float32))
    dim = x.shape[1] // 2
    y = F.silu(x[:, :dim]) * x[:, dim:]
    y.backward(dy_t)
    return x.grad.detach().cpu().numpy()


def _run_case(lib: ctypes.CDLL, tokens: int, dim: int, scale: float, strict: bool, rng: np.random.Generator) -> dict:
    inp = (rng.standard_normal((tokens, 2 * dim), dtype=np.float32) * scale).astype(np.float32)
    dy = (rng.standard_normal((tokens, dim), dtype=np.float32) * scale).astype(np.float32)

    out = np.empty((tokens, dim), dtype=np.float32)
    dx = np.empty((tokens, 2 * dim), dtype=np.float32)

    lib.ck_set_strict_parity(1 if strict else 0)
    lib.swiglu_forward(
        inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tokens,
        dim,
    )
    lib.swiglu_backward(
        inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tokens,
        dim,
    )

    y_ref = _torch_swiglu(inp)
    dx_ref = _torch_swiglu_backward(inp, dy)

    f_abs = np.abs(out - y_ref)
    b_abs = np.abs(dx - dx_ref)

    return {
        "f_max": float(np.max(f_abs)),
        "f_mean": float(np.mean(f_abs)),
        "f_p999": float(np.quantile(f_abs, 0.999)),
        "b_max": float(np.max(b_abs)),
        "b_mean": float(np.mean(b_abs)),
        "b_p999": float(np.quantile(b_abs, 0.999)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Check CK SwiGLU parity vs PyTorch")
    ap.add_argument("--lib", default="build/libckernel_engine.so", help="Path to libckernel_engine.so")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lib = _load_lib(Path(args.lib))
    rng = np.random.default_rng(args.seed)

    fast = _run_case(lib, args.tokens, args.dim, args.scale, strict=False, rng=rng)
    strict = _run_case(lib, args.tokens, args.dim, args.scale, strict=True, rng=rng)

    print(f"case: T={args.tokens} D={args.dim} scale={args.scale}")
    print(
        "forward max/mean/p99.9: "
        f"fast={fast['f_max']:.3e}/{fast['f_mean']:.3e}/{fast['f_p999']:.3e} "
        f"strict={strict['f_max']:.3e}/{strict['f_mean']:.3e}/{strict['f_p999']:.3e}"
    )
    print(
        "backward max/mean/p99.9: "
        f"fast={fast['b_max']:.3e}/{fast['b_mean']:.3e}/{fast['b_p999']:.3e} "
        f"strict={strict['b_max']:.3e}/{strict['b_mean']:.3e}/{strict['b_p999']:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
