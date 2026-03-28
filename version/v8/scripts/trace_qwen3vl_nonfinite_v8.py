#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numeric_parity_qwen3vl_mmproj_v8 as npv8  # type: ignore  # noqa: E402


def _parse_stops(text: str, max_ops: int) -> list[int]:
    text = text.strip()
    if not text:
        return list(range(max_ops))
    if ":" in text:
        start_s, end_s = text.split(":", 1)
        start = int(start_s)
        end = int(end_s)
        if end < start:
            raise ValueError(f"invalid stop range: {text}")
        return list(range(start, min(end, max_ops - 1) + 1))
    stops = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        stops.append(int(part))
    return [idx for idx in stops if 0 <= idx < max_ops]


def _first_output_arg(op: dict[str, Any]) -> dict[str, Any] | None:
    for arg in op.get("args", []):
        if str(arg.get("source", "")).startswith("output:"):
            return arg
    return None


def _buffer_stats(values: ctypes.Array[Any]) -> dict[str, Any]:
    finite = 0
    nan = 0
    inf = 0
    min_v = math.inf
    max_v = -math.inf
    sample: list[float] = []

    for idx, value in enumerate(values):
        x = float(value)
        if idx < 8:
            sample.append(x)
        if math.isnan(x):
            nan += 1
            continue
        if math.isinf(x):
            inf += 1
            continue
        finite += 1
        min_v = min(min_v, x)
        max_v = max(max_v, x)

    return {
        "len": len(values),
        "finite": finite,
        "nan": nan,
        "inf": inf,
        "min": min_v if finite else None,
        "max": max_v if finite else None,
        "sample": sample,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Trace the first non-finite stop op in the v8 Qwen3-VL vision runtime")
    ap.add_argument("--output-dir", type=Path, required=True, help="Generated artifact directory")
    ap.add_argument("--gguf", type=Path, default=None, help="Optional GGUF path to materialize artifacts if missing")
    ap.add_argument("--stops", type=str, default="", help="Stop indices: empty=all, a:b range, or comma list")
    ap.add_argument("--image-mode", choices=("gradient", "gray", "checker"), default="gradient")
    ap.add_argument("--threads", type=int, default=1, help="CK runtime thread count")
    args = ap.parse_args(argv)

    if args.gguf is not None:
        npv8._ensure_runtime_artifacts(args.gguf, args.output_dir)

    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    model_so = npv8._compile_generated_model(args.output_dir)
    layout = npv8._load_layout(args.output_dir / "layout.json")
    offsets = npv8._load_activation_offsets(args.output_dir / "layout.json")
    call = json.loads((args.output_dir / "call.json").read_text(encoding="utf-8"))
    ops = call["operations"]
    stops = _parse_stops(args.stops, len(ops))

    if not stops:
        raise RuntimeError("no valid stops selected")

    config = report["config"]
    _, planar = npv8._build_test_image(int(config["image_size"]), int(config["image_size"]), args.image_mode)
    weights_bump = Path(report["weights_bump"])
    manifest_map = args.output_dir / "weights_manifest.map"
    image_buf = offsets["image_input"]

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    first_bad = None

    for stop in stops:
        os.environ["CK_STOP_OP"] = str(stop)
        op = ops[stop]
        out_arg = _first_output_arg(op)
        summary: dict[str, Any] = {
            "stop": stop,
            "op": op.get("op"),
            "function": op.get("function"),
            "layer": op.get("layer"),
            "section": op.get("section"),
            "buffer": out_arg.get("buffer_ref") if out_arg else None,
        }

        lib = npv8._load_generated_lib(model_so)
        rc = lib.ck_model_init_with_manifest(str(weights_bump).encode(), str(manifest_map).encode())
        if rc != 0:
            summary["error"] = f"init rc={rc}"
            print(json.dumps(summary), flush=True)
            lib.ck_model_free()
            return rc

        try:
            base_ptr = int(lib.ck_model_get_base_ptr())
            if base_ptr == 0:
                raise RuntimeError("ck_model_get_base_ptr returned null")

            image_len = npv8._buffer_nbytes(image_buf) // ctypes.sizeof(ctypes.c_float)
            image_arr = (ctypes.c_float * image_len).from_address(
                base_ptr + npv8._activation_runtime_offset(layout, image_buf)
            )
            image_arr[:] = planar

            rc = lib.ck_model_decode(0, None)
            if rc != 0:
                summary["error"] = f"decode rc={rc}"
                print(json.dumps(summary), flush=True)
                return rc

            if out_arg is None:
                summary["warning"] = "stop op has no output buffer"
                print(json.dumps(summary), flush=True)
                continue

            buf = offsets[out_arg["buffer_ref"]]
            n = npv8._buffer_nbytes(buf) // ctypes.sizeof(ctypes.c_float)
            values = (ctypes.c_float * n).from_address(
                base_ptr + npv8._activation_runtime_offset(layout, buf)
            )
            summary["stats"] = _buffer_stats(values)
            print(json.dumps(summary), flush=True)

            stats = summary["stats"]
            if first_bad is None and (stats["nan"] > 0 or stats["inf"] > 0):
                first_bad = stop
                break
        finally:
            lib.ck_model_free()

    if "CK_STOP_OP" in os.environ:
        del os.environ["CK_STOP_OP"]

    result = {
        "first_bad_stop": first_bad,
        "num_stops_checked": len(stops),
    }
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
