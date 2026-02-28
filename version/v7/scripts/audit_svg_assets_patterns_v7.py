#!/usr/bin/env python3
"""
audit_svg_assets_patterns_v7.py

Quick structural audit for SVG asset corpora.
Emits counts for element/attribute coverage and top typography/color patterns
so spec catalogs can be aligned with real repo assets.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


FEATURE_PATTERNS: dict[str, str] = {
    "rect": r"<rect\b",
    "text": r"<text\b",
    "line": r"<line\b",
    "circle": r"<circle\b",
    "ellipse": r"<ellipse\b",
    "polygon": r"<polygon\b",
    "polyline": r"<polyline\b",
    "path": r"<path\b",
    "linearGradient": r"<linearGradient\b",
    "radialGradient": r"<radialGradient\b",
    "marker": r"<marker\b",
    "filter": r"<filter\b",
    "clipPath": r"<clipPath\b",
    "mask": r"<mask\b",
    "stroke_dasharray": r"stroke-dasharray\s*=",
    "stroke_linecap": r"stroke-linecap\s*=",
    "stroke_linejoin": r"stroke-linejoin\s*=",
    "font_family": r"font-family\s*=",
    "font_weight": r"font-weight\s*=",
    "font_size": r"font-size\s*=",
    "transform": r"transform\s*=",
    "opacity": r"\bopacity\s*=",
    "viewBox": r"viewBox\s*=",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compile_feature_patterns() -> dict[str, re.Pattern[str]]:
    return {k: re.compile(v, re.IGNORECASE) for k, v in FEATURE_PATTERNS.items()}


def _audit(paths: list[str], top_n: int) -> dict:
    pats = _compile_feature_patterns()
    file_hits: Counter[str] = Counter()
    occ_counts: Counter[str] = Counter()
    font_families: Counter[str] = Counter()
    font_weights: Counter[str] = Counter()
    font_sizes: Counter[str] = Counter()
    dash_values: Counter[str] = Counter()
    hex_colors: Counter[str] = Counter()
    parse_errors: list[dict[str, str]] = []
    hex_re = re.compile(r"#[0-9a-fA-F]{3,8}\b")

    for p in paths:
        path = Path(p)
        txt = path.read_text(encoding="utf-8", errors="ignore")
        for name, pat in pats.items():
            matches = pat.findall(txt)
            if matches:
                file_hits[name] += 1
                occ_counts[name] += len(matches)
        font_families.update([x.strip() for x in re.findall(r'font-family\s*=\s*"([^"]+)"', txt, re.IGNORECASE)])
        font_weights.update([x.strip() for x in re.findall(r'font-weight\s*=\s*"([^"]+)"', txt, re.IGNORECASE)])
        font_sizes.update([x.strip() for x in re.findall(r'font-size\s*=\s*"([^"]+)"', txt, re.IGNORECASE)])
        dash_values.update([x.strip() for x in re.findall(r'stroke-dasharray\s*=\s*"([^"]+)"', txt, re.IGNORECASE)])
        hex_colors.update([h.lower() for h in hex_re.findall(txt)])

        try:
            ET.fromstring(txt)
        except Exception as exc:
            parse_errors.append({"path": str(path), "error": str(exc)})

    features = {
        name: {"files": int(file_hits.get(name, 0)), "occurrences": int(occ_counts.get(name, 0))}
        for name in sorted(FEATURE_PATTERNS.keys())
    }
    return {
        "format": "v7-svg-assets-audit.v1",
        "generated_at": _now_iso(),
        "files_total": int(len(paths)),
        "files_parse_error": int(len(parse_errors)),
        "parse_errors": parse_errors,
        "features": features,
        "top_font_family": [{"value": k, "count": int(v)} for k, v in font_families.most_common(top_n)],
        "top_font_weight": [{"value": k, "count": int(v)} for k, v in font_weights.most_common(top_n)],
        "top_font_size": [{"value": k, "count": int(v)} for k, v in font_sizes.most_common(top_n)],
        "top_dasharray": [{"value": k, "count": int(v)} for k, v in dash_values.most_common(top_n)],
        "top_hex_colors": [{"value": k, "count": int(v)} for k, v in hex_colors.most_common(top_n)],
    }


def _recommendations(payload: dict) -> list[str]:
    f = payload.get("features") or {}
    rec: list[str] = []
    if int((f.get("text") or {}).get("files", 0)) > 0:
        rec.append("Keep typography-rich specs (label density + hierarchy), text appears in most assets.")
    if int((f.get("linearGradient") or {}).get("files", 0)) > 0:
        rec.append("Retain gradient specs as first-class (linearGradient usage is common).")
    if int((f.get("stroke_dasharray") or {}).get("files", 0)) > 0:
        rec.append("Add dedicated dashed/dotted stroke specs; dash patterns are present in repo assets.")
    if int((f.get("marker") or {}).get("files", 0)) > 0:
        rec.append("Include arrow/marker specs for flow lines and graph connectors.")
    if int((f.get("filter") or {}).get("files", 0)) > 0:
        rec.append("Keep filter/effect bridge specs so model sees drop-shadow style patterns.")
    if int((f.get("mask") or {}).get("files", 0)) == 0:
        rec.append("Mask usage is absent; keep mask-related specs as optional/stub until assets require them.")
    if int(payload.get("files_parse_error", 0)) > 0:
        rec.append("Fix invalid SVG assets before using full corpus in strict XML-based gates.")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit SVG asset pattern coverage.")
    ap.add_argument(
        "--assets-glob",
        default="docs/site/assets/*.svg",
        help="Glob for SVG assets.",
    )
    ap.add_argument(
        "--out",
        default="version/v7/data/svg_assets_pattern_audit_v1.json",
        help="Output JSON path.",
    )
    ap.add_argument("--top-n", type=int, default=20, help="Top-N values for colors/fonts/dash.")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(args.assets_glob)))
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = _audit(paths, max(1, int(args.top_n)))
    payload["assets_glob"] = str(args.assets_glob)
    payload["recommendations"] = _recommendations(payload)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[OK] wrote: {out}")
    print(f"[OK] files: {payload['files_total']} parse_errors: {payload['files_parse_error']}")
    print("[OK] key features:")
    for key in ("text", "rect", "line", "path", "linearGradient", "marker", "filter", "stroke_dasharray"):
        row = (payload.get("features") or {}).get(key) or {}
        print(f"  - {key}: files={row.get('files',0)} occ={row.get('occurrences',0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
