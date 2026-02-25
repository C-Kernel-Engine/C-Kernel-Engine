#!/usr/bin/env python3
"""
Regression check for kv_cache_batch_copy call-IR stitching.

Usage:
  python3 version/v7/scripts/test_kv_cache_batch_copy_call_ir_v7.py \
    --run-dir ~/.cache/ck-engine-v7/models/mradermacher--Nanbeige4.1-3B-GGUF
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _resolve_call_ir_path(run_dir: Path, mode: str) -> Path | None:
    candidates = [
        run_dir / f"lowered_{mode}_call.json",
        run_dir / ".ck_build" / f"lowered_{mode}_call.json",
        run_dir / ".ck_build" / f"lowered_{mode}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _is_numeric_expr(expr: str) -> bool:
    return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", expr.strip()))


def _main() -> int:
    ap = argparse.ArgumentParser(description="Validate kv_cache_batch_copy call-IR wiring.")
    ap.add_argument("--run-dir", required=True, help="Run/model directory containing lowered_*_call.json")
    ap.add_argument("--mode", default="prefill", choices=["prefill", "decode"], help="Mode to validate")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    path = _resolve_call_ir_path(run_dir, args.mode)
    if path is None:
        print(f"[FAIL] missing lowered call-IR for mode={args.mode} under: {run_dir}")
        return 1

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[FAIL] could not parse JSON: {path} ({e})")
        return 1

    top_errors = doc.get("errors") if isinstance(doc.get("errors"), list) else []
    ops = doc.get("operations") if isinstance(doc.get("operations"), list) else []
    if not ops:
        print(f"[FAIL] no operations in: {path}")
        return 1

    failures: list[str] = []

    for err in top_errors:
        text = json.dumps(err, ensure_ascii=False) if isinstance(err, (dict, list)) else str(err)
        if "_kv_copy_bytes" in text:
            failures.append(f"top-level error contains _kv_copy_bytes: {text}")

    kv_ops = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        if op.get("op") == "kv_cache_batch_copy" or op.get("function") == "kv_cache_batch_copy":
            kv_ops.append(op)

    if not kv_ops:
        failures.append("no kv_cache_batch_copy op found in call-IR")

    for i, op in enumerate(kv_ops):
        op_errs = op.get("errors") if isinstance(op.get("errors"), list) else []
        if op_errs:
            failures.append(f"kv op #{i} has op-level errors: {op_errs}")

        args_list = op.get("args") if isinstance(op.get("args"), list) else []
        if not args_list:
            failures.append(f"kv op #{i} has no args")
            continue

        size_arg = None
        for a in args_list:
            if isinstance(a, dict) and a.get("name") == "size":
                size_arg = a
                break
        if size_arg is None:
            failures.append(f"kv op #{i} missing 'size' arg")
            continue

        expr = str(size_arg.get("expr", "")).strip()
        if expr == "" or expr == "0":
            failures.append(f"kv op #{i} size expr is empty/zero: '{expr}'")
        elif _is_numeric_expr(expr):
            try:
                value = float(expr)
            except Exception:
                value = 0.0
            if value <= 0:
                failures.append(f"kv op #{i} numeric size expr is non-positive: {expr}")

    if failures:
        print(f"[FAIL] kv_cache_batch_copy call-IR check failed ({len(failures)} issues)")
        for item in failures[:16]:
            print(f"  - {item}")
        return 1

    print(f"[PASS] kv_cache_batch_copy call-IR is valid: {path}")
    print(f"       kv_ops={len(kv_ops)} top_errors={len(top_errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

