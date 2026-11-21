#!/usr/bin/env python3
"""Replay a bias-free, non-overlapping patch projection on captured pixels."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

from convert_safetensors_to_bump_v8 import _load_safetensors_headers, _load_tensor
from xray_operation_bundle_v8 import sha256


def patch_rows(pixels: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
    if pixels.ndim != 4 or pixels.dtype != np.float32 or not all(pixels.shape):
        raise ValueError("patch input must be nonempty FP32 NCHW")
    batch, channels, height, width = pixels.shape
    if patch_h <= 0 or patch_w <= 0 or height % patch_h or width % patch_w:
        raise ValueError("patch geometry must exactly tile the image")
    gh, gw = height // patch_h, width // patch_w
    # NCHW -> batch, patch row/column, channel, kernel row/column.
    tiled = pixels.reshape(batch, channels, gh, patch_h, gw, patch_w)
    return np.ascontiguousarray(tiled.transpose(0, 2, 4, 1, 3, 5)).reshape(
        batch * gh * gw, channels * patch_h * patch_w)


def replay(pixels: np.ndarray, weights: np.ndarray, engine: Path) -> np.ndarray:
    if weights.ndim != 4 or weights.dtype != np.float16 or not all(weights.shape):
        raise ValueError("patch weights must be nonempty FP16 OIHW")
    outputs, channels, patch_h, patch_w = weights.shape
    rows = patch_rows(pixels, patch_h, patch_w)
    if channels != pixels.shape[1] or not np.isfinite(rows).all() or not np.isfinite(weights).all():
        raise ValueError("patch operands have mismatched channels or nonfinite values")
    if max(rows.size, weights.size, len(rows) * outputs) > (1 << 31) - 1:
        raise ValueError("patch projection exceeds the kernel's integer indexing range")
    weights = np.ascontiguousarray(weights)
    result = np.empty((len(rows), outputs), dtype=np.float32)
    fn = ctypes.CDLL(str(engine.resolve())).gemm_f16
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    fn.restype = None
    fn(result.ctypes.data, weights.ctypes.data, rows.ctypes.data, outputs, len(rows), rows.shape[1])
    batch, _, height, width = pixels.shape
    return result.reshape(batch, height // patch_h, width // patch_w, outputs).transpose(0, 3, 1, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--weight", required=True, help="Exact SafeTensors OIHW key")
    ap.add_argument("--capture", type=Path, required=True, help="NPZ with patch_input and patch_output")
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    headers = _load_safetensors_headers(args.checkpoint)
    if args.weight not in headers or headers[args.weight].dtype != "F16":
        ap.error("the requested FP16 weight is missing from the checkpoint")
    weights = _load_tensor(args.checkpoint, headers, args.weight).numpy()
    with np.load(args.capture, allow_pickle=False) as capture:
        pixels = capture["patch_input"]
        reference = capture["patch_output"]
    actual = replay(pixels, weights, args.engine)
    if reference.dtype != np.float32 or reference.shape != actual.shape:
        ap.error("captured output must match the FP32 NCHW replay shape")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        ap.error("projection output contains nonfinite values")
    delta = actual.astype(np.float64) - reference
    worst = np.unravel_index(np.abs(delta).argmax(), delta.shape)
    report = {
        "scope": "identical-input operation diagnostic; not connected encoder parity",
        "kernel": "gemm_f16",
        "engine_sha256": sha256(args.engine),
        "capture_sha256": sha256(args.capture),
        "weight_key": args.weight,
        "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "input_shape": list(pixels.shape),
        "output_shape": list(actual.shape),
        "max_abs": float(abs(delta[worst])),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "worst_index_nchw": [int(index) for index in worst],
        "worst_reference": float(reference[worst]),
        "worst_actual": float(actual[worst]),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
