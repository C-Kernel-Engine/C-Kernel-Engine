#!/usr/bin/env python3
"""Classify bundle probe misses by primary failure mode."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"\[([a-zA-Z0-9_]+):([^\]]+)\]")

STYLE_KEYS = ("theme", "tone", "density", "background")
TOPOLOGY_KEYS = ("segments", "brackets", "cards", "stages", "arrows", "links", "footer", "terminal")


def _rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "rows", "cases", "items"):
        value = doc.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    raise RuntimeError("probe report missing results list")


def _bundle_fields(text: str | None) -> dict[str, str]:
    body = str(text or "")
    if not body:
        return {}
    start = body.find("[bundle]")
    end = body.rfind("[/bundle]")
    if start >= 0 and end > start:
        body = body[start:end]
    out: dict[str, str] = {}
    for key, value in TOKEN_RE.findall(body):
        out[str(key)] = str(value)
    return out


def classify_case(row: dict[str, Any]) -> dict[str, Any]:
    expected = _bundle_fields(row.get("expected_output"))
    actual = _bundle_fields(row.get("parsed_output") or row.get("parsed_output_raw") or row.get("response_text"))
    exact = bool(row.get("exact_match"))
    renderable = bool(row.get("renderable"))

    if exact:
        primary = "exact"
    elif not renderable:
        primary = "syntax"
    elif expected.get("family") and actual.get("family") and expected.get("family") != actual.get("family"):
        primary = "family"
    elif expected.get("form") and actual.get("form") and expected.get("form") != actual.get("form"):
        primary = "form"
    elif any(expected.get(key) and actual.get(key) and expected.get(key) != actual.get(key) for key in STYLE_KEYS):
        primary = "style"
    elif any(expected.get(key) and actual.get(key) and expected.get(key) != actual.get(key) for key in TOPOLOGY_KEYS):
        primary = "topology"
    elif not actual:
        primary = "syntax"
    else:
        primary = "unknown"

    mismatched_fields = [
        key
        for key in ("family", "form", *STYLE_KEYS, *TOPOLOGY_KEYS)
        if expected.get(key) and actual.get(key) and expected.get(key) != actual.get(key)
    ]
    return {
        "id": row.get("id"),
        "split": row.get("split"),
        "label": row.get("label"),
        "prompt": row.get("prompt"),
        "primary_failure": primary,
        "exact_match": exact,
        "renderable": renderable,
        "repair_applied": bool(row.get("repair_applied")),
        "missing_stop_marker": bool(row.get("missing_stop_marker")),
        "truncated_at_budget": bool(row.get("truncated_at_budget")),
        "hit_decode_ceiling": bool(row.get("hit_decode_ceiling")),
        "render_error": row.get("render_error"),
        "repair_note": row.get("repair_note"),
        "expected_fields": expected,
        "actual_fields": actual,
        "mismatched_fields": mismatched_fields,
    }


def build_autopsy(probe_report: Path) -> dict[str, Any]:
    doc = json.loads(probe_report.read_text(encoding="utf-8"))
    rows = _rows(doc)
    classified = [classify_case(row) for row in rows]
    misses = [row for row in classified if row["primary_failure"] != "exact"]

    overall = Counter(row["primary_failure"] for row in classified)
    split_breakdown: dict[str, dict[str, int]] = {}
    for row in classified:
        split = str(row.get("split") or "unknown")
        split_breakdown.setdefault(split, {})
        split_breakdown[split][row["primary_failure"]] = split_breakdown[split].get(row["primary_failure"], 0) + 1

    return {
        "schema": "ck.bundle_probe_autopsy.v1",
        "probe_report": str(probe_report),
        "total_cases": len(classified),
        "exact_cases": overall.get("exact", 0),
        "miss_cases": len(misses),
        "overall_failure_counts": dict(sorted(overall.items())),
        "split_failure_counts": {split: dict(sorted(counts.items())) for split, counts in sorted(split_breakdown.items())},
        "misses": misses,
    }


def _write_md(path: Path, autopsy: dict[str, Any]) -> None:
    lines = [
        "# Bundle Probe Autopsy",
        "",
        f"- probe_report: `{autopsy['probe_report']}`",
        f"- total_cases: `{autopsy['total_cases']}`",
        f"- exact_cases: `{autopsy['exact_cases']}`",
        f"- miss_cases: `{autopsy['miss_cases']}`",
        "",
        "## Overall Failure Counts",
        "",
    ]
    for key, value in (autopsy.get("overall_failure_counts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Split Failure Counts", ""])
    for split, counts in (autopsy.get("split_failure_counts") or {}).items():
        items = ", ".join(f"{key}={value}" for key, value in counts.items())
        lines.append(f"- `{split}`: {items}")
    lines.extend(["", "## Misses", ""])
    for row in autopsy.get("misses") or []:
        mismatch = ", ".join(row.get("mismatched_fields") or []) or "n/a"
        lines.append(f"- `{row.get('split')}` `{row.get('primary_failure')}` `{mismatch}` :: {row.get('prompt')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify bundle probe misses by primary failure mode")
    ap.add_argument("--probe-report", required=True, type=Path)
    ap.add_argument("--json-out", required=True, type=Path)
    ap.add_argument("--md-out", required=True, type=Path)
    args = ap.parse_args()

    autopsy = build_autopsy(args.probe_report.expanduser().resolve())
    args.json_out.expanduser().resolve().write_text(json.dumps(autopsy, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_md(args.md_out.expanduser().resolve(), autopsy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
