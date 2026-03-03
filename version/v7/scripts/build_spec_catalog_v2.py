#!/usr/bin/env python3
"""
Build spec_catalog_v2.json for balanced SFT control training.

The catalog is intentionally focused on SFT control coverage:
- Cross shape/palette combinations
- Prompt mode variation (concise/detailed/constraints)
- Mixed labeled/unlabeled tags (to avoid hard lock on [labeled])
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT_MODES = ["concise", "detailed", "constraints"]
PALETTES = ["neutral", "bold", "warm", "cool", "pastel", "dark"]

SHAPES = [
    {
        "shape_tag": "circle",
        "generator": "circle",
        "style": "minimal",
    },
    {
        "shape_tag": "rect",
        "generator": "rect",
        "style": "minimal",
    },
    {
        "shape_tag": "bar-chart",
        "generator": "bar_chart",
        "style": "filled",
    },
    {
        "shape_tag": "line-chart",
        "generator": "polyline",
        "style": "minimal",
    },
    {
        "shape_tag": "scatter",
        "generator": "scatter",
        "style": "minimal",
    },
]


def _build_specs(samples_train: int, samples_holdout: int) -> list[dict]:
    specs: list[dict] = []
    spec_index = 0
    for prompt_idx, prompt in enumerate(PROMPT_MODES):
        for shape_idx, shape in enumerate(SHAPES):
            for pal_idx, palette in enumerate(PALETTES):
                spec_index += 1
                labeled = ((prompt_idx + shape_idx + pal_idx) % 2) == 0
                tags = [
                    f"[prompt:{prompt}]",
                    f"[{shape['shape_tag']}]",
                    f"[palette:{palette}]",
                    f"[style:{shape['style']}]",
                ]
                if labeled:
                    tags.append("[labeled]")

                spec_id = (
                    f"sft.v2.{prompt}.{shape['shape_tag'].replace('-', '_')}"
                    f".{palette}.{'labeled' if labeled else 'plain'}.v1"
                )
                specs.append(
                    {
                        "id": spec_id,
                        "stage": "sft",
                        "group": "instruction_control_v2",
                        "tags": tags,
                        "generator": shape["generator"],
                        "palette_family": palette,
                        "style": shape["style"],
                        "constraints": {
                            "in_bounds": True,
                            "ascii_only": True,
                        },
                        "samples_train": int(samples_train),
                        "samples_holdout": int(samples_holdout),
                        "weight": 1.0,
                        "status": "ready",
                    }
                )
    return specs


def main() -> int:
    ap = argparse.ArgumentParser(description="Build balanced SFT spec catalog v2")
    ap.add_argument(
        "--out",
        default="version/v7/data/spec_catalog_v2.json",
        help="Output catalog path",
    )
    ap.add_argument(
        "--samples-train",
        type=int,
        default=120,
        help="Per-spec train rows",
    )
    ap.add_argument(
        "--samples-holdout",
        type=int,
        default=12,
        help="Per-spec holdout rows",
    )
    args = ap.parse_args()

    specs = _build_specs(args.samples_train, args.samples_holdout)
    payload = {
        "format": "spec-catalog-v2",
        "version": "2.0.0",
        "coverage_rules": {
            "min_train_per_spec": int(args.samples_train),
            "min_holdout_per_spec": int(args.samples_holdout),
            "min_pair_count": 30,
        },
        "notes": [
            "SFT-focused catalog for control robustness.",
            "Crosses prompt mode x shape x palette and alternates labeled/plain.",
        ],
        "specs": specs,
    }
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    total_train = sum(int(s["samples_train"]) for s in specs)
    total_holdout = sum(int(s["samples_holdout"]) for s in specs)
    print(
        json.dumps(
            {
                "out": str(out_path),
                "spec_count": len(specs),
                "total_train_rows": total_train,
                "total_holdout_rows": total_holdout,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

