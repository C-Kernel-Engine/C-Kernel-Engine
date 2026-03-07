#!/usr/bin/env python3
"""
Classify normalized spec SVG assets into stage-oriented candidate buckets.

This does not build final corpora. It produces a deterministic manifest that
answers:
  - which assets can be used as small full-document pretrain rows
  - which assets should be treated as panel/group/defs-rich structural sources
  - which assets are plausible SFT seed candidates
  - which assets are better reserved for holdout/canary use
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _family_from_name(name: str) -> str:
    n = name.lower()
    if any(tok in n for tok in ("bar", "line-chart", "scatter", "pie", "chart", "graph", "plot", "roofline")):
        return "chart"
    if any(tok in n for tok in ("flow", "pipeline", "timeline")):
        return "flow"
    if any(tok in n for tok in ("architecture", "topology", "network", "deployment", "infra", "stack")):
        return "architecture"
    if any(tok in n for tok in ("infographic", "comparison", "breakdown", "overview", "summary", "matrix")):
        return "infographic"
    if any(tok in n for tok in ("memory", "quant", "attention", "kernel", "activation", "weight")):
        return "technical"
    return "other"


def _size_band(chars: int) -> str:
    if chars <= 1500:
        return "tiny"
    if chars <= 6000:
        return "small"
    if chars <= 15000:
        return "medium"
    if chars <= 30000:
        return "large"
    return "xlarge"


def _count(tags: dict[str, Any], name: str) -> int:
    try:
        return int(tags.get(name, 0) or 0)
    except Exception:
        return 0


def _classify_entry(entry: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(entry.get("normalized_path") or ""))
    name = path.name
    chars = int(entry.get("chars") or 0)
    tags = entry.get("tag_counts") if isinstance(entry.get("tag_counts"), dict) else {}
    placeholders = entry.get("placeholders") if isinstance(entry.get("placeholders"), dict) else {}

    g = _count(tags, "g")
    defs = _count(tags, "defs")
    grad = _count(tags, "linearGradient")
    filt = _count(tags, "filter")
    marker = _count(tags, "marker")
    text = _count(tags, "text")
    tspan = _count(tags, "tspan")
    path_n = _count(tags, "path")
    rect = _count(tags, "rect")
    line = _count(tags, "line")
    circle = _count(tags, "circle")
    polygon = _count(tags, "polygon")

    family = _family_from_name(name)
    size_band = _size_band(chars)

    defs_heavy = (defs + grad + filt + marker) >= 6
    group_heavy = g >= 8
    text_heavy = (text + tspan) >= 80
    path_heavy = path_n >= 8
    panel_like = rect >= 6 and text >= 8 and (g >= 2 or line >= 2 or grad >= 1)
    chart_like = family == "chart" or (line >= 4 and rect >= 2 and text >= 6)

    role_flags: list[str] = []
    if chars <= 12000 and not text_heavy:
        role_flags.append("small_full")
    if panel_like and chars <= 26000:
        role_flags.append("panel_like")
    if group_heavy:
        role_flags.append("group_heavy")
    if defs_heavy:
        role_flags.append("defs_heavy")
    if path_heavy:
        role_flags.append("path_heavy")
    if text_heavy:
        role_flags.append("text_heavy")

    # SFT candidates should be structurally rich enough to be useful, but not so
    # large/text-dense that they become whole-document memorization problems.
    sft_seed = (
        chars <= 14000
        and family in {"chart", "infographic", "technical", "flow"}
        and not text_heavy
        and (rect + line + circle + polygon + path_n) >= 4
    )
    if sft_seed:
        role_flags.append("sft_seed_candidate")

    # Holdout candidates: medium-sized, readable, family-diverse artifacts.
    holdout_candidate = (
        chars <= 18000
        and family in {"chart", "infographic", "architecture", "technical", "flow"}
        and (text >= 4)
        and not (size_band == "tiny")
    )
    if holdout_candidate:
        role_flags.append("holdout_candidate")

    return {
        "source_name": entry.get("source_name"),
        "source_path": entry.get("source_path"),
        "normalized_path": str(path),
        "normalized_sha256": entry.get("normalized_sha256"),
        "chars": chars,
        "size_band": size_band,
        "family": family,
        "features": {
            "defs_heavy": defs_heavy,
            "group_heavy": group_heavy,
            "text_heavy": text_heavy,
            "path_heavy": path_heavy,
            "panel_like": panel_like,
            "chart_like": chart_like,
        },
        "tag_counts": {
            "g": g,
            "defs": defs,
            "linearGradient": grad,
            "filter": filt,
            "marker": marker,
            "text": text,
            "tspan": tspan,
            "path": path_n,
            "rect": rect,
            "line": line,
            "circle": circle,
            "polygon": polygon,
        },
        "placeholders": placeholders,
        "roles": role_flags,
    }


def _select_holdout(rows: list[dict[str, Any]], per_family: int = 3) -> list[str]:
    fam_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "holdout_candidate" not in row["roles"]:
            continue
        fam_map[str(row["family"])].append(row)
    selected: list[str] = []
    for fam, fam_rows in sorted(fam_map.items()):
        fam_rows = sorted(fam_rows, key=lambda r: (abs(int(r["chars"]) - 9000), str(r["normalized_path"])))
        selected.extend(str(r["normalized_path"]) for r in fam_rows[:per_family])
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify normalized SVG assets into spec03 candidate buckets")
    ap.add_argument("--workspace", required=True, help="Spec workspace root, e.g. version/v7/data/spec03")
    ap.add_argument("--normalized-manifest", default=None, help="Optional normalized manifest path")
    ap.add_argument("--output", default=None, help="Optional output classification manifest path")
    args = ap.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    normalized_manifest = (
        Path(args.normalized_manifest).expanduser().resolve()
        if args.normalized_manifest
        else workspace / "manifests" / "normalized_assets_manifest.json"
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else workspace / "manifests" / "asset_classification_manifest.json"
    )

    doc = json.loads(normalized_manifest.read_text(encoding="utf-8"))
    entries = doc.get("entries") if isinstance(doc.get("entries"), list) else []
    if not entries:
        raise SystemExit(f"ERROR: no entries in {normalized_manifest}")

    classified = [_classify_entry(e) for e in entries]
    counts = Counter()
    family_counts = Counter()
    for row in classified:
        family_counts[str(row["family"])] += 1
        for role in row["roles"]:
            counts[str(role)] += 1

    holdout = _select_holdout(classified, per_family=3)
    holdout_set = set(holdout)
    pretrain_small_full = [str(r["normalized_path"]) for r in classified if "small_full" in r["roles"] and str(r["normalized_path"]) not in holdout_set]
    pretrain_structural = [
        str(r["normalized_path"])
        for r in classified
        if any(role in r["roles"] for role in ("panel_like", "group_heavy", "defs_heavy", "path_heavy"))
        and str(r["normalized_path"]) not in holdout_set
    ]
    midtrain_transform = [
        str(r["normalized_path"])
        for r in classified
        if any(role in r["roles"] for role in ("panel_like", "group_heavy"))
        and str(r["normalized_path"]) not in holdout_set
    ]
    sft_seed = [str(r["normalized_path"]) for r in classified if "sft_seed_candidate" in r["roles"] and str(r["normalized_path"]) not in holdout_set]

    manifest = {
        "schema": "ck.svg_asset_classification_manifest.v1",
        "workspace": str(workspace),
        "normalized_manifest": str(normalized_manifest),
        "counts": dict(counts),
        "family_counts": dict(family_counts),
        "suggested_splits": {
            "pretrain_small_full": pretrain_small_full,
            "pretrain_structural": pretrain_structural,
            "midtrain_transform_candidates": midtrain_transform,
            "sft_seed_candidates": sft_seed,
            "holdout_candidates": holdout,
        },
        "entries": classified,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] classified={len(classified)} families={dict(family_counts)}")
    print(
        "[OK] splits "
        f"small_full={len(pretrain_small_full)} "
        f"structural={len(pretrain_structural)} "
        f"midtrain={len(midtrain_transform)} "
        f"sft_seed={len(sft_seed)} "
        f"holdout={len(holdout)}"
    )
    print(f"[OK] manifest={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
