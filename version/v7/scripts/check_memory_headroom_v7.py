#!/usr/bin/env python3
"""
check_memory_headroom_v7.py

Live system memory headroom gate for v7 parity/training commands.

Why this exists:
- Prevent obvious parity/train launches when the host is already under memory pressure.
- Emit one normalized JSON artifact operators can inspect alongside parity reports.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_GIB = 1024.0 ** 3


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_meminfo() -> dict[str, int]:
    path = Path("/proc/meminfo")
    if not path.exists():
        raise FileNotFoundError(f"/proc/meminfo not found: {path}")
    out: dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw:
            continue
        key, rest = raw.split(":", 1)
        fields = rest.strip().split()
        if not fields:
            continue
        try:
            value_kb = int(fields[0])
        except ValueError:
            continue
        out[key.strip()] = value_kb * 1024
    return out


def _read_memory_pressure() -> dict[str, Any]:
    path = Path("/proc/pressure/memory")
    if not path.exists():
        return {"available": False}
    payload: dict[str, Any] = {"available": True}
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.strip().split()
        if not fields:
            continue
        class_name = str(fields[0]).strip().lower()
        stats: dict[str, float] = {}
        for field in fields[1:]:
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            try:
                stats[str(key)] = float(value)
            except ValueError:
                continue
        payload[class_name] = stats
    return payload


def _top_rss_processes(limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,rss=,comm=,args=", "--sort=-rss"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []

    rows: list[dict[str, Any]] = []
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            rss_kb = int(parts[1])
        except ValueError:
            continue
        rows.append(
            {
                "pid": pid,
                "rss_bytes": rss_kb * 1024,
                "rss_gb": float((rss_kb * 1024) / _GIB),
                "comm": str(parts[2]),
                "cmd": str(parts[3]) if len(parts) > 3 else str(parts[2]),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _gib(value_bytes: int) -> float:
    return float(value_bytes) / _GIB


def main() -> int:
    ap = argparse.ArgumentParser(description="Check live system memory headroom for v7 parity/training commands.")
    ap.add_argument("--label", default="memory_headroom", help="Short label for reporting")
    ap.add_argument("--min-available-gb", type=float, default=6.0,
                    help="Minimum MemAvailable floor in GiB before the check passes")
    ap.add_argument("--min-available-ratio", type=float, default=0.20,
                    help="Minimum MemAvailable / MemTotal ratio used in the effective floor")
    ap.add_argument("--warn-swap-used-gb", type=float, default=0.5,
                    help="Warn when swap usage exceeds this GiB threshold")
    ap.add_argument("--top-procs", type=int, default=5,
                    help="Include the top-N RSS processes in the report (0 disables)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if float(args.min_available_gb) < 0.0:
        print("ERROR: --min-available-gb must be >= 0", file=sys.stderr)
        return 2
    if not 0.0 <= float(args.min_available_ratio) <= 1.0:
        print("ERROR: --min-available-ratio must be in [0, 1]", file=sys.stderr)
        return 2
    if float(args.warn_swap_used_gb) < 0.0:
        print("ERROR: --warn-swap-used-gb must be >= 0", file=sys.stderr)
        return 2
    if int(args.top_procs) < 0:
        print("ERROR: --top-procs must be >= 0", file=sys.stderr)
        return 2

    meminfo = _read_meminfo()
    mem_total = int(meminfo.get("MemTotal", 0) or 0)
    mem_available = int(meminfo.get("MemAvailable", meminfo.get("MemFree", 0)) or 0)
    mem_free = int(meminfo.get("MemFree", 0) or 0)
    swap_total = int(meminfo.get("SwapTotal", 0) or 0)
    swap_free = int(meminfo.get("SwapFree", 0) or 0)
    swap_used = max(0, swap_total - swap_free)

    available_ratio = (float(mem_available) / float(mem_total)) if mem_total > 0 else 0.0
    effective_min_available_bytes = max(
        int(float(args.min_available_gb) * _GIB),
        int(float(mem_total) * float(args.min_available_ratio)),
    )

    reasons: list[str] = []
    warnings: list[str] = []
    passed = True

    if mem_total <= 0:
        passed = False
        reasons.append("MemTotal is missing or zero in /proc/meminfo")
    if mem_available < effective_min_available_bytes:
        passed = False
        reasons.append(
            "MemAvailable below effective floor: "
            f"available={_gib(mem_available):.2f} GiB "
            f"required={_gib(effective_min_available_bytes):.2f} GiB"
        )
    if swap_total > 0 and _gib(swap_used) > float(args.warn_swap_used_gb):
        warnings.append(
            f"swap usage is elevated: used={_gib(swap_used):.2f} GiB free={_gib(swap_free):.2f} GiB"
        )

    pressure = _read_memory_pressure()
    top_procs = _top_rss_processes(int(args.top_procs))

    payload = {
        "generated_at": _utc_now_iso(),
        "label": str(args.label),
        "passed": bool(passed),
        "thresholds": {
            "min_available_gb": float(args.min_available_gb),
            "min_available_ratio": float(args.min_available_ratio),
            "effective_min_available_gb": float(_gib(effective_min_available_bytes)),
            "warn_swap_used_gb": float(args.warn_swap_used_gb),
        },
        "memory": {
            "mem_total_bytes": int(mem_total),
            "mem_total_gb": float(_gib(mem_total)),
            "mem_available_bytes": int(mem_available),
            "mem_available_gb": float(_gib(mem_available)),
            "mem_free_bytes": int(mem_free),
            "mem_free_gb": float(_gib(mem_free)),
            "available_ratio": float(available_ratio),
            "swap_total_bytes": int(swap_total),
            "swap_total_gb": float(_gib(swap_total)),
            "swap_free_bytes": int(swap_free),
            "swap_free_gb": float(_gib(swap_free)),
            "swap_used_bytes": int(swap_used),
            "swap_used_gb": float(_gib(swap_used)),
        },
        "pressure": pressure,
        "warnings": warnings,
        "reasons": reasons,
        "top_processes": top_procs,
    }

    print("=" * 88)
    print("v7 MEMORY HEADROOM CHECK")
    print("=" * 88)
    print(f"- label: {payload['label']}")
    print(
        "- available: %.2f GiB / %.2f GiB total (ratio=%.3f)"
        % (
            payload["memory"]["mem_available_gb"],
            payload["memory"]["mem_total_gb"],
            payload["memory"]["available_ratio"],
        )
    )
    print(
        "- effective_min_available_gb: %.2f (base=%.2f ratio=%.2f)"
        % (
            payload["thresholds"]["effective_min_available_gb"],
            payload["thresholds"]["min_available_gb"],
            payload["thresholds"]["min_available_ratio"],
        )
    )
    print(
        "- swap: used=%.2f GiB free=%.2f GiB total=%.2f GiB"
        % (
            payload["memory"]["swap_used_gb"],
            payload["memory"]["swap_free_gb"],
            payload["memory"]["swap_total_gb"],
        )
    )
    if warnings:
        for warning in warnings:
            print(f"- warning: {warning}")
    if reasons:
        for reason in reasons:
            print(f"- fail: {reason}")
    print("MEMORY_HEADROOM:", "PASS" if passed else "FAIL")
    print("=" * 88)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("JSON:", args.json_out)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
