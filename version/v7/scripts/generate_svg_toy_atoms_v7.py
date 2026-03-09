#!/usr/bin/env python3
"""
Generate a tiny closed DSL -> SVG dataset for intuition building.

Example prompt:
  [shape:circle][color:red][size:big]

Output rows:
  [shape:circle][color:red][size:big]<svg ...>...</svg><eos>

The dataset uses a fixed holdout split so the user can test whether the model
can compose unseen combinations instead of only memorizing seen rows.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path


SHAPES = ("circle", "rect", "triangle")
COLORS = ("red", "blue", "green")
SIZES = ("small", "big")

RESERVED_TOKENS = [
    "[shape:circle]",
    "[shape:rect]",
    "[shape:triangle]",
    "[color:red]",
    "[color:blue]",
    "[color:green]",
    "[size:small]",
    "[size:big]",
]

HOLDOUT_COMBOS = {
    ("circle", "red", "big"),
    ("circle", "blue", "small"),
    ("rect", "green", "big"),
    ("rect", "red", "small"),
    ("triangle", "blue", "big"),
    ("triangle", "green", "small"),
}


@dataclass(frozen=True)
class Combo:
    shape: str
    color: str
    size: str

    @property
    def prompt(self) -> str:
        return f"[shape:{self.shape}][color:{self.color}][size:{self.size}]"


def _svg_for(combo: Combo) -> str:
    width = 128
    height = 128
    stroke = "black"
    stroke_width = 2
    color = combo.color

    if combo.shape == "circle":
        r = 18 if combo.size == "small" else 36
        body = (
            f'<circle cx="64" cy="64" r="{r}" '
            f'fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
    elif combo.shape == "rect":
        w = 44 if combo.size == "small" else 76
        h = 32 if combo.size == "small" else 58
        x = (width - w) // 2
        y = (height - h) // 2
        body = (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
            f'fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
    elif combo.shape == "triangle":
        if combo.size == "small":
            points = "64,34 36,86 92,86"
        else:
            points = "64,20 22,98 106,98"
        body = (
            f'<polygon points="{points}" '
            f'fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
    else:
        raise RuntimeError(f"unsupported shape: {combo.shape}")

    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'


def _row(combo: Combo) -> str:
    return f"{combo.prompt}{_svg_for(combo)}<eos>"


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate tiny DSL->SVG atom dataset")
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="toy_svg_atoms", help="Output file prefix")
    ap.add_argument("--train-repeats", type=int, default=12, help="How many times to repeat each seen combo in train")
    ap.add_argument("--holdout-repeats", type=int, default=1, help="How many times to repeat each holdout combo")
    args = ap.parse_args()

    if args.train_repeats < 1:
        raise SystemExit("--train-repeats must be >= 1")
    if args.holdout_repeats < 1:
        raise SystemExit("--holdout-repeats must be >= 1")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = [Combo(shape=s, color=c, size=z) for s, c, z in itertools.product(SHAPES, COLORS, SIZES)]
    holdout = [combo for combo in combos if (combo.shape, combo.color, combo.size) in HOLDOUT_COMBOS]
    train = [combo for combo in combos if (combo.shape, combo.color, combo.size) not in HOLDOUT_COMBOS]

    train_rows = [_row(combo) for combo in train for _ in range(int(args.train_repeats))]
    holdout_rows = [_row(combo) for combo in holdout for _ in range(int(args.holdout_repeats))]
    all_rows = [*train_rows, *holdout_rows]
    svg_train = [_svg_for(combo) for combo in train for _ in range(int(args.train_repeats))]
    svg_holdout = [_svg_for(combo) for combo in holdout for _ in range(int(args.holdout_repeats))]
    tokenizer_corpus = [*all_rows, *(combo.prompt for combo in combos)]
    canary_prompts = [combo.prompt for combo in combos]
    seen_prompts = [combo.prompt for combo in train]
    holdout_prompts = [combo.prompt for combo in holdout]

    _write_lines(out_dir / f"{args.prefix}_train.txt", train_rows)
    _write_lines(out_dir / f"{args.prefix}_holdout.txt", holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_all.txt", all_rows)
    _write_lines(out_dir / f"{args.prefix}_svg_train.txt", svg_train)
    _write_lines(out_dir / f"{args.prefix}_svg_holdout.txt", svg_holdout)
    _write_lines(out_dir / f"{args.prefix}_tokenizer_corpus.txt", tokenizer_corpus)
    _write_lines(out_dir / f"{args.prefix}_reserved_control_tokens.txt", RESERVED_TOKENS)
    _write_lines(out_dir / f"{args.prefix}_canary_prompts.txt", canary_prompts)
    _write_lines(out_dir / f"{args.prefix}_seen_prompts.txt", seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_holdout_prompts.txt", holdout_prompts)

    manifest = {
        "format": "toy-svg-atoms-manifest.v1",
        "prefix": args.prefix,
        "out_dir": str(out_dir),
        "train_repeats": int(args.train_repeats),
        "holdout_repeats": int(args.holdout_repeats),
        "shapes": list(SHAPES),
        "colors": list(COLORS),
        "sizes": list(SIZES),
        "train_unique_combos": [
            {"shape": combo.shape, "color": combo.color, "size": combo.size}
            for combo in train
        ],
        "holdout_unique_combos": [
            {"shape": combo.shape, "color": combo.color, "size": combo.size}
            for combo in holdout
        ],
        "counts": {
            "unique_total": len(combos),
            "unique_train": len(train),
            "unique_holdout": len(holdout),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "tokenizer_rows": len(tokenizer_corpus),
        },
        "artifacts": {
            "train": str(out_dir / f"{args.prefix}_train.txt"),
            "holdout": str(out_dir / f"{args.prefix}_holdout.txt"),
            "all": str(out_dir / f"{args.prefix}_all.txt"),
            "svg_train": str(out_dir / f"{args.prefix}_svg_train.txt"),
            "svg_holdout": str(out_dir / f"{args.prefix}_svg_holdout.txt"),
            "tokenizer_corpus": str(out_dir / f"{args.prefix}_tokenizer_corpus.txt"),
            "reserved_control_tokens": str(out_dir / f"{args.prefix}_reserved_control_tokens.txt"),
            "canary_prompts": str(out_dir / f"{args.prefix}_canary_prompts.txt"),
            "seen_prompts": str(out_dir / f"{args.prefix}_seen_prompts.txt"),
            "holdout_prompts": str(out_dir / f"{args.prefix}_holdout_prompts.txt"),
        },
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "prefix": args.prefix,
                "unique_train": len(train),
                "unique_holdout": len(holdout),
                "train_rows": len(train_rows),
                "holdout_rows": len(holdout_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
