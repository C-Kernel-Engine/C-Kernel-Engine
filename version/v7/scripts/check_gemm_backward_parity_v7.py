#!/usr/bin/env python3
"""
check_gemm_backward_parity_v7.py

Shape-sweep parity for the v7 training gemm backward kernel-id path.
In kernel bindings, `gemm_backward_f32` maps to `fc2_backward_kernel`, so this test
calls `fc2_backward_kernel` directly across diverse (T, aligned_in, aligned_out)
triples and compares against a NumPy reference.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
UNITTEST_DIR = ROOT / "unittest"


DEFAULT_SHAPES: List[Tuple[int, int, int]] = [
    # (T, aligned_in, aligned_out)
    (1, 64, 64),
    (1, 64, 96),
    (2, 96, 64),
    (4, 128, 192),
    (8, 64, 128),
    (3, 127, 191),
    (5, 80, 56),
    (16, 192, 64),
]


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _load_lib():
    if str(UNITTEST_DIR) not in sys.path:
        sys.path.insert(0, str(UNITTEST_DIR))
    from lib_loader import load_lib  # noqa: E402

    lib = load_lib("libckernel_engine.so")
    lib.fc2_backward_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # d_output [T, O]
        ctypes.POINTER(ctypes.c_float),  # fc2_input [T, I]
        ctypes.POINTER(ctypes.c_float),  # W_fc2 [O, I]
        ctypes.POINTER(ctypes.c_float),  # d_input [T, I]
        ctypes.POINTER(ctypes.c_float),  # d_W_fc2 [O, I]
        ctypes.POINTER(ctypes.c_float),  # d_b_fc2 [O]
        ctypes.c_int,                    # T
        ctypes.c_int,                    # aligned_in
        ctypes.c_int,                    # aligned_out
        ctypes.c_int,                    # num_threads
    ]
    lib.fc2_backward_kernel.restype = None
    return lib


def _parse_shapes(spec: str) -> List[Tuple[int, int, int]]:
    shapes: List[Tuple[int, int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.lower().split("x")
        if len(parts) != 3:
            raise ValueError(f"invalid shape token `{token}`; expected T×I×O")
        t, i, o = (int(parts[0]), int(parts[1]), int(parts[2]))
        if t <= 0 or i <= 0 or o <= 0:
            raise ValueError(f"shape values must be positive: `{token}`")
        shapes.append((t, i, o))
    if not shapes:
        raise ValueError("no valid shapes parsed")
    return shapes


def run(seed: int, tol: float, shapes: List[Tuple[int, int, int]], num_threads: int) -> Dict[str, object]:
    np.random.seed(seed)
    lib = _load_lib()

    rows: List[Dict[str, object]] = []
    passed = True

    for (t, i, o) in shapes:
        d_output = np.random.randn(t, o).astype(np.float32)
        fc2_input = np.random.randn(t, i).astype(np.float32)
        w_fc2 = np.random.randn(o, i).astype(np.float32)

        d_input = np.zeros((t, i), dtype=np.float32)
        d_w = np.zeros((o, i), dtype=np.float32)

        db_init = (np.random.randn(o).astype(np.float32) * 0.01).astype(np.float32)
        d_b = db_init.copy()

        lib.fc2_backward_kernel(
            _ptr(d_output.reshape(-1)),
            _ptr(fc2_input.reshape(-1)),
            _ptr(w_fc2.reshape(-1)),
            _ptr(d_input.reshape(-1)),
            _ptr(d_w.reshape(-1)),
            _ptr(d_b),
            ctypes.c_int(t),
            ctypes.c_int(i),
            ctypes.c_int(o),
            ctypes.c_int(num_threads),
        )

        # Reference
        d_input_ref = (d_output @ w_fc2).astype(np.float32)
        d_w_ref = (d_output.T @ fc2_input).astype(np.float32)
        d_b_ref = (db_init + d_output.sum(axis=0)).astype(np.float32)

        di = float(np.max(np.abs(d_input - d_input_ref)))
        dw = float(np.max(np.abs(d_w - d_w_ref)))
        db = float(np.max(np.abs(d_b - d_b_ref)))
        md = max(di, dw, db)

        row_ok = bool(md <= tol)
        passed = passed and row_ok

        rows.append(
            {
                "shape": {"T": t, "aligned_in": i, "aligned_out": o},
                "passed": row_ok,
                "max_diff": md,
                "max_diff_breakdown": {
                    "d_input": di,
                    "d_weight": dw,
                    "d_bias": db,
                },
                "tol": tol,
            }
        )

    return {
        "passed": bool(passed),
        "seed": seed,
        "tol": tol,
        "num_threads": num_threads,
        "shapes": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Shape-sweep parity for v7 gemm backward kernel path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=2e-4)
    parser.add_argument(
        "--shapes",
        type=str,
        default="",
        help="Custom shape list T×I×O comma-separated (example: 1x64x64,4x128x192)",
    )
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    shape_list = DEFAULT_SHAPES if not args.shapes else _parse_shapes(str(args.shapes))
    result = run(
        seed=int(args.seed),
        tol=float(args.tol),
        shapes=shape_list,
        num_threads=int(args.num_threads),
    )
    print(
        "gemm_backward shape sweep: %s (%d shapes)"
        % ("PASS" if result["passed"] else "FAIL", len(result["shapes"]))
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"JSON: {args.json_out}")
    return 0 if bool(result["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
