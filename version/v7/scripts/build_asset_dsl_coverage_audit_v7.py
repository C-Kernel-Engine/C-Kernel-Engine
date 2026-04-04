#!/usr/bin/env python3
"""Audit how much of docs/site/assets is covered by the current DSL/compiler lines."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ASSETS_ROOT = ROOT / "docs" / "site" / "assets"
SCRIPTS_ROOT = ROOT / "version" / "v7" / "scripts"

GENERATOR_GLOBS = (
    "generate_svg_structured_spec14a_v7.py",
    "generate_svg_structured_spec14b_v7.py",
    "generate_svg_structured_spec15a_v7.py",
    "generate_svg_structured_spec15b_v7.py",
    "generate_svg_structured_spec16_v7.py",
    "generate_svg_structured_spec17_v7.py",
    "generate_svg_structured_spec18_v7.py",
    "generate_svg_structured_spec19_v7.py",
)

ASSET_RE = re.compile(r'source_asset\s*=\s*"([^"]+\.svg)"')


def _bucket_for_asset(stem: str) -> str:
    name = stem.lower().replace("_", "-")
    if any(key in name for key in ("timeline", "evolution", "gate-ladder", "runtime-modes")):
        return "timeline"
    if any(key in name for key in ("memory", "allocator", "weight", "grad-accum", "backprop", "canary")):
        return "memory-map-or-training"
    if any(key in name for key in ("pipeline", "flow", "architecture", "topology", "registry", "dataflow", "qwen", "operator", "rdma")):
        return "system-diagram-or-flow"
    if any(
        key in name
        for key in (
            "comparison",
            "performance",
            "quantization",
            "format",
            "grouping",
            "spectrum",
            "economics",
            "chasm",
            "analysis",
            "constraints",
            "map",
            "plan",
            "edge-case",
            "cheatsheet",
        )
    ):
        return "board-chart-or-comparison"
    if any(key in name for key in ("concept", "overview", "infographic", "compass", "philosophy", "principles", "softmax")):
        return "editorial-or-concept"
    return "uncategorized"


def _collect_assets(root: Path) -> list[Path]:
    return sorted(root.rglob("*.svg"))


def _collect_generator_asset_map(scripts_root: Path) -> dict[str, list[str]]:
    asset_map: dict[str, list[str]] = defaultdict(list)
    for name in GENERATOR_GLOBS:
        path = scripts_root / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for asset in sorted(set(ASSET_RE.findall(text))):
            asset_map[asset].append(name)
    return dict(sorted(asset_map.items()))


def _summarize(asset_paths: list[Path], asset_map: dict[str, list[str]]) -> dict[str, Any]:
    assets = [p.name for p in asset_paths]
    covered = sorted([name for name in assets if name in asset_map])
    missing = sorted([name for name in assets if name not in asset_map])

    covered_bucket_counts: Counter[str] = Counter()
    missing_bucket_counts: Counter[str] = Counter()
    missing_by_bucket: dict[str, list[str]] = defaultdict(list)

    for name in covered:
        covered_bucket_counts[_bucket_for_asset(Path(name).stem)] += 1
    for name in missing:
        bucket = _bucket_for_asset(Path(name).stem)
        missing_bucket_counts[bucket] += 1
        missing_by_bucket[bucket].append(name)

    return {
        "asset_count": len(assets),
        "covered_count": len(covered),
        "missing_count": len(missing),
        "coverage_rate": (len(covered) / len(assets)) if assets else 0.0,
        "covered_assets": covered,
        "missing_assets": missing,
        "covered_bucket_counts": dict(sorted(covered_bucket_counts.items())),
        "missing_bucket_counts": dict(sorted(missing_bucket_counts.items())),
        "missing_by_bucket": {k: sorted(v) for k, v in sorted(missing_by_bucket.items())},
        "asset_to_generators": asset_map,
        "supported_family_summary": {
            "comparison_board_spec14a": [
                "tokenizer-performance-comparison.svg",
                "sentencepiece-vs-bpe-wordpiece.svg",
                "rope-layouts-compared.svg",
                "compute-bandwidth-chasm.svg",
                "quantization-formats.svg",
            ],
            "timeline_spec14b": [
                "ir-v66-evolution-timeline.svg",
                "ir-timeline-why.svg",
            ],
            "memory_map_spec15a": [
                "memory-layout-map.svg",
                "bump_allocator_quant.svg",
                "v7-train-memory-canary.svg",
            ],
            "system_diagram_spec15b": [
                "pipeline-overview.svg",
                "ir-pipeline-flow.svg",
                "kernel-registry-flow.svg",
            ],
        },
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Asset DSL Coverage Audit")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Asset count: `{summary['asset_count']}`")
    lines.append(f"- Covered by current gold DSL generators: `{summary['covered_count']}`")
    lines.append(f"- Missing from current gold DSL generators: `{summary['missing_count']}`")
    lines.append(f"- Coverage rate: `{summary['coverage_rate']:.4f}`")
    lines.append("")
    lines.append("## Covered Families")
    lines.append("")
    for family, assets in (summary.get("supported_family_summary") or {}).items():
        lines.append(f"- `{family}`: `{len(assets)}` assets")
    lines.append("")
    lines.append("## Missing By Bucket")
    lines.append("")
    for bucket, count in (summary.get("missing_bucket_counts") or {}).items():
        lines.append(f"- `{bucket}`: `{count}`")
    lines.append("")
    lines.append("## Covered Assets")
    lines.append("")
    for name in summary.get("covered_assets") or []:
        generators = ", ".join(summary["asset_to_generators"].get(name) or [])
        lines.append(f"- `{name}` via `{generators}`")
    lines.append("")
    lines.append("## Missing Assets")
    lines.append("")
    for bucket, names in (summary.get("missing_by_bucket") or {}).items():
        lines.append(f"- `{bucket}`")
        for name in names:
            lines.append(f"  - `{name}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Current training has been operating on a narrow compiler language relative to the site asset library.")
    lines.append("- Broader training should start by expanding DSL/compiler coverage inside the strongest missing buckets, not by adding more local rung patches.")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit docs/site/assets coverage by current structured SVG DSL generators")
    ap.add_argument("--assets-root", type=Path, default=ASSETS_ROOT, help="Root containing target SVG assets")
    ap.add_argument("--scripts-root", type=Path, default=SCRIPTS_ROOT, help="Root containing generator scripts")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    ap.add_argument("--md-out", type=Path, default=None, help="Optional Markdown output path")
    args = ap.parse_args()

    asset_paths = _collect_assets(args.assets_root.expanduser().resolve())
    asset_map = _collect_generator_asset_map(args.scripts_root.expanduser().resolve())
    summary = _summarize(asset_paths, asset_map)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_render_markdown(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
