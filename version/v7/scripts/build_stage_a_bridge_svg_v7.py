#!/usr/bin/env python3
"""
build_stage_a_bridge_svg_v7.py

Create a small Stage-A bridge corpus by selecting Stage-B rows that contain
SVG syntax/features missing from Stage-A baseline data.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


FEATURE_REGEX: "OrderedDict[str, re.Pattern[str]]" = OrderedDict(
    [
        ("g_tag", re.compile(r"<g\b", re.IGNORECASE)),
        ("defs", re.compile(r"<defs\b", re.IGNORECASE)),
        ("use", re.compile(r"<use\b", re.IGNORECASE)),
        ("linearGradient", re.compile(r"<linearGradient\b", re.IGNORECASE)),
        ("radialGradient", re.compile(r"<radialGradient\b", re.IGNORECASE)),
        ("marker", re.compile(r"<marker\b", re.IGNORECASE)),
        ("clipPath", re.compile(r"<clipPath\b", re.IGNORECASE)),
        ("mask", re.compile(r"<mask\b", re.IGNORECASE)),
        ("filter", re.compile(r"<filter\b|<fe[A-Za-z]+\b", re.IGNORECASE)),
        ("transform_attr", re.compile(r"\btransform\s*=\s*\"", re.IGNORECASE)),
        ("stroke_dasharray", re.compile(r"\bstroke-dasharray\s*=\s*\"", re.IGNORECASE)),
        ("viewBox", re.compile(r"\bviewBox\s*=\s*\"", re.IGNORECASE)),
    ]
)


@dataclass(frozen=True)
class Coverage:
    rows: int
    counts: Dict[str, int]


def _load_rows(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    rows: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        row = raw.strip()
        if row:
            rows.append(row)
    return rows


def _coverage(rows: Iterable[str], feature_keys: List[str]) -> Coverage:
    items = list(rows)
    counts: Dict[str, int] = {}
    for key in feature_keys:
        rx = FEATURE_REGEX[key]
        counts[key] = sum(1 for row in items if rx.search(row))
    return Coverage(rows=len(items), counts=counts)


def _missing_features(stage_a: Coverage, stage_b: Coverage, feature_keys: List[str]) -> List[str]:
    out: List[str] = []
    for key in feature_keys:
        if stage_a.counts.get(key, 0) == 0 and stage_b.counts.get(key, 0) > 0:
            out.append(key)
    return out


def _row_features(row: str, feature_keys: List[str]) -> List[str]:
    return [key for key in feature_keys if FEATURE_REGEX[key].search(row)]


def _dedupe_keep_order(rows: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for row in rows:
        if row in seen:
            continue
        seen.add(row)
        out.append(row)
    return out


def _sample_bridge_rows(
    stage_b_rows: List[str],
    missing_features: List[str],
    per_feature_cap: int,
    max_total: int,
    seed: int,
) -> List[str]:
    rng = random.Random(seed)
    feature_to_rows: Dict[str, List[str]] = {k: [] for k in missing_features}
    for row in stage_b_rows:
        present = _row_features(row, missing_features)
        for key in present:
            feature_to_rows[key].append(row)

    selected: List[str] = []
    for key in missing_features:
        pool = feature_to_rows.get(key, [])
        rng.shuffle(pool)
        picked = pool[: max(0, per_feature_cap)]
        selected.extend(picked)

    selected = _dedupe_keep_order(selected)
    if len(selected) >= max_total:
        return selected[:max_total]

    selected_set = set(selected)
    residual = [row for row in stage_b_rows if row not in selected_set and _row_features(row, missing_features)]
    rng.shuffle(residual)
    selected.extend(residual[: max_total - len(selected)])
    return selected


def _parse_features(raw: str) -> List[str]:
    value = raw.strip().lower()
    if value == "auto":
        return list(FEATURE_REGEX.keys())
    out: List[str] = []
    for piece in raw.split(","):
        key = piece.strip()
        if not key:
            continue
        if key not in FEATURE_REGEX:
            raise SystemExit(f"unknown feature '{key}'. use --list-features")
        out.append(key)
    if not out:
        raise SystemExit("no features selected")
    return out


def _print_coverage(label: str, cov: Coverage, feature_keys: List[str]) -> None:
    print(f"[{label}] rows={cov.rows}")
    for key in feature_keys:
        c = cov.counts.get(key, 0)
        pct = (100.0 * c / cov.rows) if cov.rows else 0.0
        print(f"  {key:16s} {c:6d} ({pct:6.2f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Stage-A bridge SVG corpus from Stage-B syntax gaps.")
    ap.add_argument("--stage-a", required=True, help="Stage-A dataset path (baseline).")
    ap.add_argument("--stage-b", required=True, help="Stage-B dataset path (richer corpus).")
    ap.add_argument("--out", required=True, help="Output bridge corpus (line-per-svg).")
    ap.add_argument("--manifest", default="", help="Optional JSON manifest path.")
    ap.add_argument(
        "--features",
        default="auto",
        help="Comma-separated feature keys or 'auto' (default). use --list-features.",
    )
    ap.add_argument("--list-features", action="store_true", help="Print available feature keys and exit.")
    ap.add_argument("--per-feature-cap", type=int, default=8, help="Max sampled rows per missing feature (default: 8).")
    ap.add_argument("--max-total", type=int, default=96, help="Max total bridge rows (default: 96).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = ap.parse_args()

    if args.list_features:
        print("\n".join(FEATURE_REGEX.keys()))
        return 0

    stage_a_path = Path(args.stage_a).resolve()
    stage_b_path = Path(args.stage_b).resolve()
    out_path = Path(args.out).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else out_path.with_suffix(".manifest.json")
    feature_keys = _parse_features(args.features)

    stage_a_rows = _load_rows(stage_a_path)
    stage_b_rows = _load_rows(stage_b_path)
    cov_a = _coverage(stage_a_rows, feature_keys)
    cov_b = _coverage(stage_b_rows, feature_keys)
    missing = _missing_features(cov_a, cov_b, feature_keys)

    selected = _sample_bridge_rows(
        stage_b_rows=stage_b_rows,
        missing_features=missing,
        per_feature_cap=max(1, int(args.per_feature_cap)),
        max_total=max(1, int(args.max_total)),
        seed=int(args.seed),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(selected)
    if payload:
        payload += "\n"
    out_path.write_text(payload, encoding="utf-8")

    bridge_cov = _coverage(selected, feature_keys)
    combined_cov = _coverage(stage_a_rows + selected, feature_keys)
    missing_after_bridge = [k for k in feature_keys if combined_cov.counts.get(k, 0) == 0 and cov_b.counts.get(k, 0) > 0]
    selected_feature_rows: Dict[str, int] = {}
    for key in feature_keys:
        rx = FEATURE_REGEX[key]
        selected_feature_rows[key] = sum(1 for row in selected if rx.search(row))

    manifest = {
        "format": "v7-stage-a-bridge-corpus",
        "stage_a_path": str(stage_a_path),
        "stage_b_path": str(stage_b_path),
        "out_path": str(out_path),
        "seed": int(args.seed),
        "per_feature_cap": int(args.per_feature_cap),
        "max_total": int(args.max_total),
        "features": feature_keys,
        "missing_features_from_stage_a": missing,
        "missing_features_after_bridge_only": missing_after_bridge,
        "stage_a_rows": cov_a.rows,
        "stage_b_rows": cov_b.rows,
        "bridge_rows": len(selected),
        "stage_a_feature_counts": cov_a.counts,
        "stage_b_feature_counts": cov_b.counts,
        "bridge_feature_counts": selected_feature_rows,
        "combined_stage_a_plus_bridge_feature_counts": combined_cov.counts,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _print_coverage("stage_a", cov_a, feature_keys)
    _print_coverage("stage_b", cov_b, feature_keys)
    _print_coverage("bridge", bridge_cov, feature_keys)
    _print_coverage("stage_a_plus_bridge", combined_cov, feature_keys)
    print(f"[gap] missing_from_stage_a={','.join(missing) if missing else '-'}")
    print(f"[gap] missing_after_bridge_only={','.join(missing_after_bridge) if missing_after_bridge else '-'}")
    print(f"[ok] out={out_path}")
    print(f"[ok] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
