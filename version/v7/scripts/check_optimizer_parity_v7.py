#!/usr/bin/env python3
"""
check_optimizer_parity_v7.py

Quick parity checks for training optimizer kernels used by v7:
  - adamw_update_f32
  - gradient_clip_norm_f32
  - adamw_clip_update_multi_f32 (global grad-norm clip semantics)
  - gradient_accumulate_f32
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
UNITTEST_DIR = ROOT / "unittest"


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _load_lib():
    if str(UNITTEST_DIR) not in sys.path:
        sys.path.insert(0, str(UNITTEST_DIR))
    from lib_loader import load_lib  # noqa: E402

    lib = load_lib("libckernel_engine.so")

    lib.adamw_update_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # grad
        ctypes.POINTER(ctypes.c_float),  # weight
        ctypes.POINTER(ctypes.c_float),  # m
        ctypes.POINTER(ctypes.c_float),  # v
        ctypes.c_size_t,                 # numel
        ctypes.c_float,                  # lr
        ctypes.c_float,                  # beta1
        ctypes.c_float,                  # beta2
        ctypes.c_float,                  # eps
        ctypes.c_float,                  # weight_decay
        ctypes.c_int,                    # step
    ]
    lib.adamw_update_f32.restype = None

    lib.gradient_clip_norm_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # grad
        ctypes.c_size_t,                 # numel
        ctypes.c_float,                  # max_norm
    ]
    lib.gradient_clip_norm_f32.restype = ctypes.c_float

    lib.gradient_accumulate_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # dst
        ctypes.POINTER(ctypes.c_float),  # src
        ctypes.c_size_t,                 # numel
    ]
    lib.gradient_accumulate_f32.restype = None

    lib.adamw_clip_update_multi_f32.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # grads
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # weights
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # m_states
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # v_states
        ctypes.POINTER(ctypes.c_size_t),                 # numels
        ctypes.c_int,                                    # tensor_count
        ctypes.c_float,                                  # lr
        ctypes.c_float,                                  # beta1
        ctypes.c_float,                                  # beta2
        ctypes.c_float,                                  # eps
        ctypes.c_float,                                  # weight_decay
        ctypes.c_float,                                  # max_grad_norm
        ctypes.c_int,                                    # step
    ]
    lib.adamw_clip_update_multi_f32.restype = None

    return lib


def _adamw_ref(
    grad: np.ndarray,
    weight: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> Dict[str, np.ndarray]:
    # Reference math matching optimizer_kernels.c behavior.
    grad32 = grad.astype(np.float32, copy=False)
    w = weight.astype(np.float32, copy=True)
    m_out = m.astype(np.float32, copy=True)
    v_out = v.astype(np.float32, copy=True)

    m_out = beta1 * m_out + (1.0 - beta1) * grad32
    v_out = beta2 * v_out + (1.0 - beta2) * (grad32 * grad32)

    b1 = 1.0 - (beta1 ** step)
    b2 = 1.0 - (beta2 ** step)
    m_hat = m_out / b1
    v_hat = v_out / b2
    update = m_hat / (np.sqrt(v_hat) + eps) + weight_decay * w
    w = w - lr * update

    return {"weight": w.astype(np.float32), "m": m_out.astype(np.float32), "v": v_out.astype(np.float32)}


def _adamw_clip_multi_ref(
    grads: List[np.ndarray],
    weights: List[np.ndarray],
    m_states: List[np.ndarray],
    v_states: List[np.ndarray],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    max_grad_norm: float,
    step: int,
) -> Dict[str, object]:
    total_sum_sq = 0.0
    for g in grads:
        if g.size == 0:
            continue
        total_sum_sq += float(np.sum(g.astype(np.float64) ** 2))
    global_norm = float(np.sqrt(total_sum_sq))
    scale = 1.0
    if max_grad_norm > 0.0 and global_norm > max_grad_norm and global_norm > 0.0:
        scale = float(max_grad_norm / global_norm)

    g_out: List[np.ndarray] = []
    w_out: List[np.ndarray] = []
    m_out: List[np.ndarray] = []
    v_out: List[np.ndarray] = []
    for g, w, m, v in zip(grads, weights, m_states, v_states):
        gs = (g.astype(np.float32, copy=True) * np.float32(scale)).astype(np.float32, copy=False)
        ref = _adamw_ref(gs, w, m, v, lr, beta1, beta2, eps, weight_decay, step)
        g_out.append(gs)
        w_out.append(ref["weight"])
        m_out.append(ref["m"])
        v_out.append(ref["v"])

    return {
        "global_norm": global_norm,
        "scale": float(scale),
        "grads": g_out,
        "weights": w_out,
        "m_states": m_out,
        "v_states": v_out,
    }


def run(seed: int, tol: float) -> Dict[str, object]:
    np.random.seed(seed)
    lib = _load_lib()

    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.01
    step = 7

    adamw_shapes = [1, 7, 64, 513, 4096]
    clip_shapes = [1, 17, 257, 4096]
    accum_shapes = [1, 31, 1024, 4096]

    adamw_rows: List[Dict[str, object]] = []
    clip_rows: List[Dict[str, object]] = []
    clip_multi_rows: List[Dict[str, object]] = []
    accum_rows: List[Dict[str, object]] = []
    passed = True

    for n in adamw_shapes:
        grad = (np.random.randn(n).astype(np.float32) * 0.1).astype(np.float32)
        weight = np.random.randn(n).astype(np.float32)
        m = np.random.randn(n).astype(np.float32) * 0.01
        v = np.abs(np.random.randn(n).astype(np.float32)) * 0.01

        weight_c = weight.copy()
        m_c = m.copy()
        v_c = v.copy()

        lib.adamw_update_f32(
            _ptr(grad),
            _ptr(weight_c),
            _ptr(m_c),
            _ptr(v_c),
            ctypes.c_size_t(n),
            ctypes.c_float(lr),
            ctypes.c_float(beta1),
            ctypes.c_float(beta2),
            ctypes.c_float(eps),
            ctypes.c_float(weight_decay),
            ctypes.c_int(step),
        )

        ref = _adamw_ref(grad, weight, m, v, lr, beta1, beta2, eps, weight_decay, step)
        dw = float(np.max(np.abs(weight_c - ref["weight"])))
        dm = float(np.max(np.abs(m_c - ref["m"])))
        dv = float(np.max(np.abs(v_c - ref["v"])))
        md = max(dw, dm, dv)
        row_ok = bool(md <= tol)
        passed = passed and row_ok
        adamw_rows.append(
            {
                "numel": n,
                "passed": row_ok,
                "max_diff": md,
                "max_diff_breakdown": {"weight": dw, "m": dm, "v": dv},
                "tol": tol,
            }
        )

    max_norm = 1.0
    for n in clip_shapes:
        grad = (np.random.randn(n).astype(np.float32) * 2.0).astype(np.float32)
        grad_c = grad.copy()

        ref_norm = float(np.linalg.norm(grad.astype(np.float64)))
        ref_out = grad.copy()
        if ref_norm > max_norm and ref_norm > 0.0:
            ref_out *= np.float32(max_norm / ref_norm)

        got_norm = float(lib.gradient_clip_norm_f32(_ptr(grad_c), ctypes.c_size_t(n), ctypes.c_float(max_norm)))
        dnorm = abs(got_norm - ref_norm)
        dgrad = float(np.max(np.abs(grad_c - ref_out)))
        md = max(dnorm, dgrad)
        clip_tol = max(tol, 5e-5)
        row_ok = bool(md <= clip_tol)
        passed = passed and row_ok
        clip_rows.append(
            {
                "numel": n,
                "passed": row_ok,
                "max_diff": md,
                "returned_norm_diff": dnorm,
                "grad_diff": dgrad,
                "tol": clip_tol,
            }
        )

    clip_multi_cases = [
        {"sizes": [7, 64, 513], "max_grad_norm": 1.0},
        {"sizes": [31, 257, 1024], "max_grad_norm": 0.25},
        {"sizes": [17, 4096], "max_grad_norm": 0.0},  # no clipping path
    ]
    for case in clip_multi_cases:
        sizes = [int(x) for x in case["sizes"]]
        max_grad_norm_case = float(case["max_grad_norm"])

        grads = [(np.random.randn(n).astype(np.float32) * 2.0).astype(np.float32) for n in sizes]
        weights = [np.random.randn(n).astype(np.float32) for n in sizes]
        m_states = [(np.random.randn(n).astype(np.float32) * 0.01).astype(np.float32) for n in sizes]
        v_states = [(np.abs(np.random.randn(n).astype(np.float32)) * 0.01).astype(np.float32) for n in sizes]

        grads_c = [g.copy() for g in grads]
        weights_c = [w.copy() for w in weights]
        m_c = [m.copy() for m in m_states]
        v_c = [v.copy() for v in v_states]

        tc = len(sizes)
        grad_ptrs = (ctypes.POINTER(ctypes.c_float) * tc)(*[_ptr(g) for g in grads_c])
        weight_ptrs = (ctypes.POINTER(ctypes.c_float) * tc)(*[_ptr(w) for w in weights_c])
        m_ptrs = (ctypes.POINTER(ctypes.c_float) * tc)(*[_ptr(m) for m in m_c])
        v_ptrs = (ctypes.POINTER(ctypes.c_float) * tc)(*[_ptr(v) for v in v_c])
        numels = (ctypes.c_size_t * tc)(*[ctypes.c_size_t(int(n)) for n in sizes])

        lib.adamw_clip_update_multi_f32(
            grad_ptrs,
            weight_ptrs,
            m_ptrs,
            v_ptrs,
            numels,
            ctypes.c_int(tc),
            ctypes.c_float(lr),
            ctypes.c_float(beta1),
            ctypes.c_float(beta2),
            ctypes.c_float(eps),
            ctypes.c_float(weight_decay),
            ctypes.c_float(max_grad_norm_case),
            ctypes.c_int(step),
        )

        ref = _adamw_clip_multi_ref(
            grads=grads,
            weights=weights,
            m_states=m_states,
            v_states=v_states,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm_case,
            step=step,
        )

        max_w = 0.0
        max_m = 0.0
        max_v = 0.0
        max_g = 0.0
        for i in range(tc):
            max_w = max(max_w, float(np.max(np.abs(weights_c[i] - ref["weights"][i]))))
            max_m = max(max_m, float(np.max(np.abs(m_c[i] - ref["m_states"][i]))))
            max_v = max(max_v, float(np.max(np.abs(v_c[i] - ref["v_states"][i]))))
            max_g = max(max_g, float(np.max(np.abs(grads_c[i] - ref["grads"][i]))))

        md = max(max_w, max_m, max_v, max_g)
        multi_tol = max(tol, 5e-5)
        row_ok = bool(md <= multi_tol)
        passed = passed and row_ok
        clip_multi_rows.append(
            {
                "sizes": sizes,
                "max_grad_norm": max_grad_norm_case,
                "passed": row_ok,
                "max_diff": md,
                "max_diff_breakdown": {
                    "weight": max_w,
                    "m": max_m,
                    "v": max_v,
                    "grad_scaled": max_g,
                },
                "global_norm_ref": float(ref["global_norm"]),
                "global_scale_ref": float(ref["scale"]),
                "tol": multi_tol,
            }
        )

    for n in accum_shapes:
        dst = np.random.randn(n).astype(np.float32)
        src = np.random.randn(n).astype(np.float32)
        dst_c = dst.copy()

        lib.gradient_accumulate_f32(_ptr(dst_c), _ptr(src), ctypes.c_size_t(n))
        ref = dst + src
        d = float(np.max(np.abs(dst_c - ref)))
        row_ok = bool(d <= 1e-6)
        passed = passed and row_ok
        accum_rows.append({"numel": n, "passed": row_ok, "max_diff": d, "tol": 1e-6})

    return {
        "passed": bool(passed),
        "seed": seed,
        "tol": tol,
        "adamw": adamw_rows,
        "clip_norm": clip_rows,
        "clip_update_multi": clip_multi_rows,
        "accumulate": accum_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick v7 optimizer kernel parity checks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=2e-5)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    result = run(seed=int(args.seed), tol=float(args.tol))
    print(
        "optimizer parity: %s (adamw=%d clip=%d clip_multi=%d accum=%d)"
        % (
            "PASS" if result["passed"] else "FAIL",
            len(result["adamw"]),
            len(result["clip_norm"]),
            len(result["clip_update_multi"]),
            len(result["accumulate"]),
        )
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"JSON: {args.json_out}")
    return 0 if bool(result["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
