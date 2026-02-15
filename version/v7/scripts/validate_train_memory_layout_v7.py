#!/usr/bin/env python3
"""
validate_train_memory_layout_v7.py

Static validation for layout_train.json:
- alignment
- region/tensor bounds
- overlap detection
- IR2 tensor coverage
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _shape_numel(shape: Any) -> Optional[int]:
    if not isinstance(shape, list) or not shape:
        return None
    n = 1
    for d in shape:
        if not isinstance(d, int) or d <= 0:
            return None
        n *= d
    return int(n)


def _tensor_numel(meta: Dict[str, Any]) -> Optional[int]:
    n = meta.get("numel")
    if isinstance(n, int) and n > 0:
        return int(n)
    s = _shape_numel(meta.get("shape"))
    if isinstance(s, int) and s > 0:
        return int(s)
    return None


def _collect_intervals(tensors: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
    rows: List[Tuple[int, int, str]] = []
    for t in tensors:
        tid = str(t.get("id", ""))
        off = int(t.get("offset", -1) or -1)
        end = int(t.get("end", -1) or -1)
        rows.append((off, end, tid))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return rows


def _validate_layout(layout: Dict[str, Any], ir2: Optional[Dict[str, Any]], strict: bool) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    failures: List[str] = []
    warnings: List[str] = []

    fmt = str(layout.get("format", ""))
    align_bytes = int(layout.get("align_bytes", 1) or 1)
    total_bytes = int(layout.get("total_bytes", 0) or 0)
    regions = layout.get("regions") if isinstance(layout.get("regions"), list) else []
    tensors = layout.get("tensors") if isinstance(layout.get("tensors"), list) else []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            failures.append(f"{name}: {detail}")

    add_check("format", fmt == "layout-train-v7", f"format={fmt}")
    add_check("non_empty_tensors", len(tensors) > 0, f"tensor_count={len(tensors)}")
    add_check("positive_total_bytes", total_bytes > 0, f"total_bytes={total_bytes}")

    # Tensor-level checks
    bad_align = 0
    bad_bounds = 0
    bad_span = 0
    for t in tensors:
        off_raw = t.get("offset", None)
        end_raw = t.get("end", None)
        bytes_raw = t.get("bytes", None)
        off = int(off_raw) if isinstance(off_raw, (int, float)) else -1
        end = int(end_raw) if isinstance(end_raw, (int, float)) else -1
        b = int(bytes_raw) if isinstance(bytes_raw, (int, float)) else 0
        if off < 0 or end < 0 or b <= 0:
            bad_span += 1
            continue
        if end - off != b:
            bad_span += 1
        if align_bytes > 1 and (off % align_bytes) != 0:
            bad_align += 1
        if end > total_bytes:
            bad_bounds += 1

    add_check("tensor_span_consistency", bad_span == 0, f"bad_span={bad_span}")
    add_check("tensor_alignment", bad_align == 0, f"misaligned={bad_align} align={align_bytes}")
    add_check("tensor_bounds", bad_bounds == 0, f"out_of_bounds={bad_bounds}")

    # Overlap checks are the primary memory-safety gate for generated offsets.
    overlap_count = 0
    intervals = _collect_intervals(tensors)
    prev_off, prev_end, prev_id = -1, -1, ""
    for off, end, tid in intervals:
        if prev_end > off:
            overlap_count += 1
            if overlap_count <= 10:
                warnings.append(f"overlap: {prev_id} [{prev_off},{prev_end}) vs {tid} [{off},{end})")
        if end > prev_end:
            prev_off, prev_end, prev_id = off, end, tid
    add_check("tensor_non_overlap", overlap_count == 0, f"overlaps={overlap_count}")

    # Region checks
    bad_region_bounds = 0
    bad_region_counts = 0
    for r in regions:
        roff_raw = r.get("offset", None)
        rbytes_raw = r.get("bytes", None)
        rcnt_raw = r.get("count", None)
        roff = int(roff_raw) if isinstance(roff_raw, (int, float)) else -1
        rbytes = int(rbytes_raw) if isinstance(rbytes_raw, (int, float)) else -1
        rcnt = int(rcnt_raw) if isinstance(rcnt_raw, (int, float)) else -1
        if roff < 0 or rbytes < 0 or (roff + rbytes) > total_bytes:
            bad_region_bounds += 1
        if rcnt < 0:
            bad_region_counts += 1
    add_check("region_bounds", bad_region_bounds == 0, f"bad_regions={bad_region_bounds}")
    add_check("region_counts", bad_region_counts == 0, f"bad_counts={bad_region_counts}")

    # IR2 coverage check
    if isinstance(ir2, dict):
        ir2_tensors = ir2.get("tensors") if isinstance(ir2.get("tensors"), dict) else {}
        expected = set()
        for tid, meta in ir2_tensors.items():
            if not isinstance(tid, str):
                continue
            if _tensor_numel(meta if isinstance(meta, dict) else {}) is None:
                continue
            expected.add(tid)

        actual = {str(t.get("id", "")) for t in tensors if str(t.get("id", ""))}
        missing = sorted(expected - actual)
        add_check("ir2_tensor_coverage", len(missing) == 0, f"missing={len(missing)}")
        if missing:
            warnings.append("missing tensors: " + ", ".join(missing[:16]))

    passed = len(failures) == 0
    if strict and warnings:
        # keep warnings non-fatal in strict mode; strict enforces hard failures above.
        pass

    return {
        "generated_at": _utc_now_iso(),
        "format": "layout-train-audit-v7",
        "passed": passed,
        "strict": bool(strict),
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Validate training memory layout for overlap/bounds/coverage.")
    p.add_argument("--layout", type=Path, required=True, help="layout_train.json path")
    p.add_argument("--ir2", type=Path, default=None, help="Optional ir2_train_backward.json for coverage checks")
    p.add_argument("--output", type=Path, default=None, help="Output audit json path")
    p.add_argument("--strict", action="store_true", help="Enable strict mode (report only; checks are already hard)")
    args = p.parse_args()

    layout = _load_json(args.layout)
    ir2 = _load_json(args.ir2) if args.ir2 and args.ir2.exists() else None
    report = _validate_layout(layout=layout, ir2=ir2, strict=bool(args.strict))

    if args.output:
        _save_json(args.output, report)
        print(f"Wrote train memory audit: {args.output}")

    summary = "PASS" if report.get("passed") else "FAIL"
    print(f"v7 train memory audit: {summary} (checks={len(report.get('checks', []))} failures={len(report.get('failures', []))})")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
