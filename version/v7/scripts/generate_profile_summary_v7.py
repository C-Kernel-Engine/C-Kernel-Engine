#!/usr/bin/env python3
"""
Generate profile_summary.json from profile_decode.csv for v7.

This is used when profiling through ck-cli-v7 (pure C runtime), where
ck_run_v7.py is not the execution path and therefore does not auto-emit
summary JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def build_summary(entries: List[Dict[str, str]]) -> Dict[str, object]:
    # Legacy token-0 view (kept for backward compatibility in viewer/scripts)
    by_op: Dict[str, float] = {}
    by_layer: Dict[int, Dict[str, float]] = {}
    total_us = 0.0

    for e in entries:
        if e.get("token_id", "0") != "0":
            continue
        op = e.get("op", "unknown")
        layer = int(e.get("layer", -1))
        us = float(e.get("time_us", 0))
        total_us += us
        by_op[op] = by_op.get(op, 0.0) + us
        if layer >= 0:
            if layer not in by_layer:
                by_layer[layer] = {}
            by_layer[layer][op] = by_layer[layer].get(op, 0.0) + us

    # Full by-mode view (decode/prefill split)
    by_mode: Dict[str, Dict[str, object]] = {}
    for e in entries:
        mode = e.get("mode", "unknown")
        op = e.get("op", "unknown")
        us = float(e.get("time_us", 0))
        bucket = by_mode.setdefault(mode, {"total_us": 0.0, "by_op": {}})
        bucket["total_us"] = float(bucket["total_us"]) + us
        op_map = bucket["by_op"]  # type: ignore[index]
        op_map[op] = op_map.get(op, 0.0) + us

    return {
        "total_us": total_us,
        "total_ms": total_us / 1000.0,
        "by_op": by_op,
        "by_layer": {str(k): v for k, v in sorted(by_layer.items())},
        "by_mode": by_mode,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate v7 profile summary JSON from CSV")
    parser.add_argument("--work-dir", required=True, type=Path, help="Model directory containing profile_decode.csv")
    parser.add_argument("--csv", type=Path, default=None, help="Optional explicit CSV path")
    parser.add_argument("--out", type=Path, default=None, help="Optional explicit output JSON path")
    args = parser.parse_args()

    work_dir = args.work_dir
    csv_path = args.csv or (work_dir / "profile_decode.csv")
    out_path = args.out or (work_dir / "profile_summary.json")

    if not csv_path.exists():
        raise FileNotFoundError(f"profile CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        entries = [row for row in reader]

    summary = build_summary(entries)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    total_ms = float(summary["total_ms"])
    print(f"Wrote {out_path}")
    print(f"Total (token_id=0 view): {total_ms:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

