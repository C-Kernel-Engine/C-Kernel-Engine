#!/usr/bin/env python3
"""
check_rms_swiglu_backward_parity_v7.py

Why this script exists:
- Focused kernel-level parity sweep for training-critical backward math.
- Provides fast, isolated proof that RMSNorm/SwiGLU backward kernels align with
  PyTorch references before debugging full harness/runtime behavior.

Covers:
- rmsnorm_backward (default and strict scalar parity paths)
- swiglu_backward_exact (strict reference path)
- swiglu_backward (fast approximation path with looser tolerance)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - env dependent
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc


ROOT = Path(__file__).resolve().parents[3]
UNITTEST_DIR = ROOT / "unittest"


def _ptr_f32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _load_lib():
    if str(UNITTEST_DIR) not in sys.path:
        sys.path.insert(0, str(UNITTEST_DIR))
    from lib_loader import load_lib  # noqa: E402

    lib = load_lib("libckernel_engine.so")

    lib.rmsnorm_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib.rmsnorm_forward.restype = None

    lib.rmsnorm_backward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.rmsnorm_backward.restype = None

    lib.swiglu_backward_exact.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.swiglu_backward_exact.restype = None

    lib.swiglu_backward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.swiglu_backward.restype = None

    try:
        lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
        lib.ck_set_strict_parity.restype = None
        has_strict = True
    except Exception:
        has_strict = False
    lib._has_ck_set_strict = has_strict  # type: ignore[attr-defined]
    return lib


def _rms_torch_backward(x_np: np.ndarray, gamma_np: np.ndarray, d_out_np: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    gamma = torch.tensor(gamma_np, dtype=torch.float32, requires_grad=True)
    d_out = torch.tensor(d_out_np, dtype=torch.float32)

    var = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = (var + float(eps)).rsqrt()
    y = x * rstd * gamma
    loss = (y * d_out).sum()
    loss.backward()
    return x.grad.detach().cpu().numpy(), gamma.grad.detach().cpu().numpy()


def _swiglu_torch_backward(inp_np: np.ndarray, d_out_np: np.ndarray) -> np.ndarray:
    x = torch.tensor(inp_np, dtype=torch.float32, requires_grad=True)
    d_out = torch.tensor(d_out_np, dtype=torch.float32)
    half = x.shape[1] // 2
    gate = x[:, :half]
    up = x[:, half:]
    y = F.silu(gate) * up
    loss = (y * d_out).sum()
    loss.backward()
    return x.grad.detach().cpu().numpy()


def _run_rms_cases(lib, rng: np.random.Generator, tol_default: float, tol_strict: float) -> Dict[str, Any]:
    cases = [
        (1, 64, 64, 1e-6),
        (4, 128, 128, 1e-6),
        (8, 255, 256, 1e-6),
        (8, 256, 256, 1e-6),
        (3, 384, 384, 1e-5),
    ]
    rows: List[Dict[str, Any]] = []
    passed = True
    has_strict = bool(getattr(lib, "_has_ck_set_strict", False))

    modes = [("default", 0, float(tol_default))]
    if has_strict:
        modes.append(("strict", 1, float(tol_strict)))

    for mode_name, strict_val, tol in modes:
        if has_strict:
            lib.ck_set_strict_parity(ctypes.c_int(strict_val))
        for (t, d_model, aligned, eps) in cases:
            x = (rng.standard_normal((t, aligned)).astype(np.float32) * 0.5).astype(np.float32)
            x[:, d_model:] = 0.0
            gamma = (rng.standard_normal((d_model,)).astype(np.float32) * 0.2 + 1.0).astype(np.float32)
            d_out = (rng.standard_normal((t, aligned)).astype(np.float32) * 0.5).astype(np.float32)
            d_out[:, d_model:] = 0.0

            out = np.empty_like(x, dtype=np.float32)
            rstd = np.empty((t,), dtype=np.float32)
            d_x = np.empty_like(x, dtype=np.float32)
            d_gamma = np.empty_like(gamma, dtype=np.float32)

            lib.rmsnorm_forward(
                _ptr_f32(x.reshape(-1)),
                _ptr_f32(gamma),
                _ptr_f32(out.reshape(-1)),
                _ptr_f32(rstd),
                ctypes.c_int(t),
                ctypes.c_int(d_model),
                ctypes.c_int(aligned),
                ctypes.c_float(float(eps)),
            )
            lib.rmsnorm_backward(
                _ptr_f32(d_out.reshape(-1)),
                _ptr_f32(x.reshape(-1)),
                _ptr_f32(gamma),
                _ptr_f32(rstd),
                _ptr_f32(d_x.reshape(-1)),
                _ptr_f32(d_gamma),
                ctypes.c_int(t),
                ctypes.c_int(d_model),
                ctypes.c_int(aligned),
            )

            ref_dx, ref_dgamma = _rms_torch_backward(
                x[:, :d_model].copy(),
                gamma.copy(),
                d_out[:, :d_model].copy(),
                float(eps),
            )

            dx_max = float(np.max(np.abs(d_x[:, :d_model] - ref_dx)))
            dg_max = float(np.max(np.abs(d_gamma - ref_dgamma)))
            md = max(dx_max, dg_max)
            row_ok = bool(md <= tol)
            passed = passed and row_ok
            rows.append(
                {
                    "kernel": "rmsnorm_backward",
                    "mode": mode_name,
                    "shape": {"tokens": t, "d_model": d_model, "aligned": aligned},
                    "eps": float(eps),
                    "tol": float(tol),
                    "passed": row_ok,
                    "max_diff": md,
                    "max_diff_breakdown": {"d_input": dx_max, "d_gamma": dg_max},
                }
            )

    if has_strict:
        lib.ck_set_strict_parity(ctypes.c_int(0))
    return {"passed": bool(passed), "rows": rows, "strict_supported": has_strict}


def _run_swiglu_cases(lib, rng: np.random.Generator, tol_exact: float, tol_fast: float, require_fast: bool) -> Dict[str, Any]:
    cases = [
        (1, 64),
        (4, 128),
        (8, 256),
        (3, 511),
        (2, 1024),
    ]
    rows: List[Dict[str, Any]] = []
    exact_pass = True
    fast_pass = True

    for (tokens, dim) in cases:
        inp = (rng.standard_normal((tokens, 2 * dim)).astype(np.float32) * 0.6).astype(np.float32)
        d_out = (rng.standard_normal((tokens, dim)).astype(np.float32) * 0.6).astype(np.float32)
        ref_dx = _swiglu_torch_backward(inp.copy(), d_out.copy())

        d_in_exact = np.empty_like(inp, dtype=np.float32)
        lib.swiglu_backward_exact(
            _ptr_f32(inp.reshape(-1)),
            _ptr_f32(d_out.reshape(-1)),
            _ptr_f32(d_in_exact.reshape(-1)),
            ctypes.c_int(tokens),
            ctypes.c_int(dim),
        )
        ex_max = float(np.max(np.abs(d_in_exact - ref_dx)))
        ex_ok = bool(ex_max <= tol_exact)
        exact_pass = exact_pass and ex_ok
        rows.append(
            {
                "kernel": "swiglu_backward_exact",
                "shape": {"tokens": tokens, "dim": dim},
                "tol": float(tol_exact),
                "passed": ex_ok,
                "max_diff": ex_max,
            }
        )

        d_in_fast = np.empty_like(inp, dtype=np.float32)
        lib.swiglu_backward(
            _ptr_f32(inp.reshape(-1)),
            _ptr_f32(d_out.reshape(-1)),
            _ptr_f32(d_in_fast.reshape(-1)),
            ctypes.c_int(tokens),
            ctypes.c_int(dim),
        )
        fast_max = float(np.max(np.abs(d_in_fast - ref_dx)))
        fast_ok = bool(fast_max <= tol_fast)
        fast_pass = fast_pass and fast_ok
        rows.append(
            {
                "kernel": "swiglu_backward",
                "shape": {"tokens": tokens, "dim": dim},
                "tol": float(tol_fast),
                "passed": fast_ok,
                "max_diff": fast_max,
                "note": "fast path uses SIMD sigmoid approximation; tolerance is intentionally looser.",
            }
        )

    passed = bool(exact_pass and (fast_pass if require_fast else True))
    return {
        "passed": passed,
        "rows": rows,
        "require_fast": bool(require_fast),
        "exact_passed": bool(exact_pass),
        "fast_passed": bool(fast_pass),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Parity sweep for v7 RMSNorm/SwiGLU backward kernels.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rms-tol-default", type=float, default=3e-5)
    ap.add_argument("--rms-tol-strict", type=float, default=8e-6)
    ap.add_argument("--swiglu-exact-tol", type=float, default=5e-6)
    ap.add_argument("--swiglu-fast-tol", type=float, default=3e-4)
    ap.add_argument("--require-fast", action="store_true", help="Fail if fast SwiGLU path exceeds tolerance.")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if torch is None:
        print("ERROR: PyTorch not available for parity checks.", file=sys.stderr)
        print(f"DETAIL: {_TORCH_IMPORT_ERROR}", file=sys.stderr)
        print("HINT: activate venv with torch installed.", file=sys.stderr)
        return 2

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    lib = _load_lib()
    rms = _run_rms_cases(
        lib=lib,
        rng=rng,
        tol_default=float(args.rms_tol_default),
        tol_strict=float(args.rms_tol_strict),
    )
    swiglu = _run_swiglu_cases(
        lib=lib,
        rng=rng,
        tol_exact=float(args.swiglu_exact_tol),
        tol_fast=float(args.swiglu_fast_tol),
        require_fast=bool(args.require_fast),
    )

    passed = bool(rms["passed"] and swiglu["passed"])
    payload: Dict[str, Any] = {
        "passed": passed,
        "seed": int(args.seed),
        "rmsnorm_backward": rms,
        "swiglu_backward": swiglu,
    }

    print(
        "rms+swiglu backward parity: %s (rms=%s swiglu=%s fast_required=%s)"
        % (
            "PASS" if passed else "FAIL",
            "PASS" if rms["passed"] else "FAIL",
            "PASS" if swiglu["passed"] else "FAIL",
            "yes" if bool(args.require_fast) else "no",
        )
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON: {args.json_out}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
