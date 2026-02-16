#!/usr/bin/env python3
"""Validate CK-runtime long-horizon oracle report for Make gates.

This checker is intentionally strict:
- requires CK backend report,
- requires oracle enabled/available/strict snapshot mode,
- requires zero oracle failures,
- requires parity pass,
- requires minimum optimizer-step horizon.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _err(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate CK long-horizon oracle parity JSON")
    ap.add_argument("--json", dest="json_path", required=True, help="Path to train report JSON")
    ap.add_argument("--min-steps", type=int, default=1000, help="Minimum optimizer steps required")
    ap.add_argument("--max-loss-abs-diff", type=float, default=None, help="Optional strict cap for max_loss_abs_diff")
    ap.add_argument("--max-param-abs-diff", type=float, default=None, help="Optional strict cap for final_param_max_abs_diff")
    args = ap.parse_args()

    p = Path(args.json_path)
    if not p.exists():
        return _err(f"missing report: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return _err(f"failed to parse JSON {p}: {e}")

    backend = str(data.get("backend", "")).strip().lower()
    if backend != "ck":
        return _err(f"backend is not ck (got: {backend or '<empty>'})")

    steps = int(_num(data.get("steps"), 0))
    if steps < args.min_steps:
        return _err(f"optimizer steps too low: {steps} < {args.min_steps}")

    pass_parity = bool(data.get("pass_parity", False))
    if not pass_parity:
        return _err("pass_parity=false")

    oracle = data.get("oracle") if isinstance(data.get("oracle"), dict) else {}
    if not bool(oracle.get("enabled", False)):
        return _err("oracle.enabled=false")
    if not bool(oracle.get("available", False)):
        return _err("oracle.available=false")
    if not bool(oracle.get("strict", False)):
        return _err("oracle.strict=false")
    if not bool(oracle.get("snapshot_torch_enabled", False)):
        return _err("oracle.snapshot_torch_enabled=false")

    failures = oracle.get("failures") if isinstance(oracle.get("failures"), list) else []
    if failures:
        first = failures[0] if isinstance(failures[0], dict) else {}
        step = first.get("step")
        reason = first.get("reason") or first.get("signature") or "unknown"
        return _err(f"oracle failures present (count={len(failures)}), first step={step}, reason={reason}")

    checks = oracle.get("checks") if isinstance(oracle.get("checks"), list) else []
    if checks:
        try:
            max_checked = max(int(x) for x in checks)
        except Exception:
            max_checked = 0
        if max_checked < args.min_steps:
            return _err(f"oracle check coverage too low: max(checks)={max_checked} < {args.min_steps}")

    max_loss = _num(data.get("max_loss_abs_diff"), 0.0)
    max_param = _num(data.get("final_param_max_abs_diff"), 0.0)

    if args.max_loss_abs_diff is not None and max_loss > float(args.max_loss_abs_diff):
        return _err(f"max_loss_abs_diff too high: {max_loss:.6e} > {args.max_loss_abs_diff:.6e}")

    if args.max_param_abs_diff is not None and max_param > float(args.max_param_abs_diff):
        return _err(f"final_param_max_abs_diff too high: {max_param:.6e} > {args.max_param_abs_diff:.6e}")

    print("CK long-horizon oracle: PASS")
    print(f"  report: {p}")
    print(f"  steps: {steps}")
    print(f"  max_loss_abs_diff: {max_loss:.6e}")
    print(f"  final_param_max_abs_diff: {max_param:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
