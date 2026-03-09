#!/usr/bin/env python3
"""
Generate a semantic SVG toy dataset with:
- discrete size buckets
- palette buckets
- a fixed no-merge tokenizer
- explicit stage datasets for shapes, cards, charts, and curves

Stage 1 focuses on shape composition.
Later stages reuse the same tokenizer and add richer structured targets.
"""

from __future__ import annotations

import argparse
import itertools
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from render_svg_semantic_ir_v7 import render_ir


SYSTEM_TOKENS = ["<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>"]

SHAPE_PROMPT_TOKENS = [
    "[task:svg]",
    "[shape:circle]",
    "[shape:rect]",
    "[shape:triangle]",
    "[palette:warm]",
    "[palette:cool]",
    "[palette:mono]",
    "[palette:signal]",
    "[size:xs]",
    "[size:sm]",
    "[size:md]",
    "[size:lg]",
    "[size:xl]",
    "[OUT]",
]

SHAPE_OUTPUT_TOKENS = [
    "[svg]",
    "[/svg]",
    "[layout:center]",
    "[circle:xs]",
    "[circle:sm]",
    "[circle:md]",
    "[circle:lg]",
    "[circle:xl]",
    "[rect:xs]",
    "[rect:sm]",
    "[rect:md]",
    "[rect:lg]",
    "[rect:xl]",
    "[triangle:xs]",
    "[triangle:sm]",
    "[triangle:md]",
    "[triangle:lg]",
    "[triangle:xl]",
    "[fill:warm]",
    "[fill:cool]",
    "[fill:mono]",
    "[fill:signal]",
    "[stroke:default]",
]

CARD_PROMPT_TOKENS = [
    "[task:card]",
    "[theme:light]",
    "[theme:dark]",
    "[accent:warm]",
    "[accent:cool]",
    "[accent:mono]",
    "[text:title]",
    "[text:bullet2]",
    "[text:bullet3]",
]

CARD_OUTPUT_TOKENS = [
    "[card:sm]",
    "[card:md]",
    "[card:lg]",
    "[bg:light]",
    "[bg:dark]",
    "[fg:light]",
    "[fg:dark]",
    "[accent:warm]",
    "[accent:cool]",
    "[accent:mono]",
    "[title_slot]",
    "[bullet_slot:2]",
    "[bullet_slot:3]",
]

CHART_PROMPT_TOKENS = [
    "[task:chart]",
    "[chart:bar]",
    "[chart:line]",
    "[bars:3]",
    "[trend:up]",
    "[trend:down]",
    "[trend:flat]",
]

CHART_OUTPUT_TOKENS = [
    "[chart:bar]",
    "[chart:line]",
    "[bars3:up]",
    "[bars3:down]",
    "[bars3:flat]",
]

CURVE_PROMPT_TOKENS = [
    "[task:plot]",
    "[curve:linear-up]",
    "[curve:quad-up]",
    "[curve:quad-down]",
    "[curve:s-curve]",
]

CURVE_OUTPUT_TOKENS = [
    "[plot:curve]",
    "[curve:linear-up]",
    "[curve:quad-up]",
    "[curve:quad-down]",
    "[curve:s-curve]",
]

ASCII_BASE = ["\t", "\n", "\r"] + [chr(i) for i in range(32, 127)]

SHAPES = ("circle", "rect", "triangle")
PALETTES = ("warm", "cool", "mono", "signal")
SIZES = ("xs", "sm", "md", "lg", "xl")
CARD_THEMES = ("light", "dark")
CARD_ACCENTS = ("warm", "cool", "mono")
CARD_BULLETS = (2, 3)
CHART_TYPES = ("bar", "line")
CHART_TRENDS = ("up", "down", "flat")
CURVE_FAMILIES = ("linear-up", "quad-up", "quad-down", "s-curve")

HOLDOUT_COMBOS = {
    ("circle", "warm", "xl"),
    ("circle", "mono", "md"),
    ("rect", "cool", "xs"),
    ("rect", "signal", "lg"),
    ("triangle", "warm", "sm"),
    ("triangle", "cool", "xl"),
    ("triangle", "mono", "lg"),
    ("circle", "signal", "xs"),
}

CARD_HOLDOUT_COMBOS = {
    ("dark", "cool", 3),
    ("light", "warm", 2),
}

CHART_HOLDOUT_COMBOS = {
    ("bar", "flat"),
    ("line", "down"),
}

CURVE_HOLDOUT_COMBOS = {
    "quad-down",
}


@dataclass(frozen=True)
class Combo:
    shape: str
    palette: str
    size: str

    @property
    def prompt_tokens(self) -> list[str]:
        return [
            "[task:svg]",
            f"[shape:{self.shape}]",
            f"[palette:{self.palette}]",
            f"[size:{self.size}]",
            "[OUT]",
        ]

    @property
    def prompt_text(self) -> str:
        return " ".join(self.prompt_tokens)


@dataclass(frozen=True)
class CardCombo:
    theme: str
    accent: str
    bullets: int

    @property
    def prompt_tokens(self) -> list[str]:
        return [
            "[task:card]",
            f"[theme:{self.theme}]",
            f"[accent:{self.accent}]",
            "[text:title]",
            f"[text:bullet{self.bullets}]",
            "[OUT]",
        ]

    @property
    def prompt_text(self) -> str:
        return " ".join(self.prompt_tokens)


@dataclass(frozen=True)
class ChartCombo:
    chart: str
    trend: str

    @property
    def prompt_tokens(self) -> list[str]:
        return [
            "[task:chart]",
            f"[chart:{self.chart}]",
            "[bars:3]",
            f"[trend:{self.trend}]",
            "[OUT]",
        ]

    @property
    def prompt_text(self) -> str:
        return " ".join(self.prompt_tokens)


@dataclass(frozen=True)
class CurveCombo:
    curve: str

    @property
    def prompt_tokens(self) -> list[str]:
        return [
            "[task:plot]",
            f"[curve:{self.curve}]",
            "[OUT]",
        ]

    @property
    def prompt_text(self) -> str:
        return " ".join(self.prompt_tokens)


def _resolved_shape_token(combo: Combo) -> str:
    return f"[{combo.shape}:{combo.size}]"


def _row(combo: Combo) -> str:
    out = [
        "[svg]",
        "[layout:center]",
        _resolved_shape_token(combo),
        f"[fill:{combo.palette}]",
        "[stroke:default]",
        "[/svg]",
    ]
    return " ".join(combo.prompt_tokens + out)


def _card_row(combo: CardCombo) -> str:
    card_size = "[card:md]" if combo.bullets == 2 else "[card:lg]"
    bg = "[bg:dark]" if combo.theme == "dark" else "[bg:light]"
    fg = "[fg:light]" if combo.theme == "dark" else "[fg:dark]"
    out = [
        "[svg]",
        card_size,
        bg,
        fg,
        f"[accent:{combo.accent}]",
        "[title_slot]",
        f"[bullet_slot:{combo.bullets}]",
        "[/svg]",
    ]
    return " ".join(combo.prompt_tokens + out)


def _chart_row(combo: ChartCombo) -> str:
    out = [
        "[svg]",
        f"[chart:{combo.chart}]",
        f"[bars3:{combo.trend}]",
        "[/svg]",
    ]
    return " ".join(combo.prompt_tokens + out)


def _curve_row(combo: CurveCombo) -> str:
    out = [
        "[svg]",
        "[plot:curve]",
        f"[curve:{combo.curve}]",
        "[/svg]",
    ]
    return " ".join(combo.prompt_tokens + out)


def _palette_fill(name: str) -> str:
    return {
        "warm": "#ef4444",
        "cool": "#2563eb",
        "mono": "#475569",
        "signal": "#f59e0b",
    }[name]


def _shape_geometry(shape: str, size: str) -> tuple[str, dict[str, str]]:
    circle_r = {"xs": "10", "sm": "18", "md": "26", "lg": "36", "xl": "46"}
    rect_dims = {
        "xs": ("26", "18"),
        "sm": ("40", "28"),
        "md": ("56", "38"),
        "lg": ("76", "58"),
        "xl": ("92", "70"),
    }
    tri_points = {
        "xs": "64,48 48,78 80,78",
        "sm": "64,36 40,82 88,82",
        "md": "64,28 34,88 94,88",
        "lg": "64,20 22,98 106,98",
        "xl": "64,14 16,106 112,106",
    }
    if shape == "circle":
        return "circle", {"cx": "64", "cy": "64", "r": circle_r[size]}
    if shape == "rect":
        w, h = rect_dims[size]
        x = str((128 - int(w)) // 2)
        y = str((128 - int(h)) // 2)
        return "rect", {"x": x, "y": y, "width": w, "height": h, "rx": "8"}
    if shape == "triangle":
        return "polygon", {"points": tri_points[size]}
    raise RuntimeError(f"unsupported shape: {shape}")


def render_svg(combo: Combo) -> str:
    fill = _palette_fill(combo.palette)
    stroke = "#0f172a"
    tag, attrs = _shape_geometry(combo.shape, combo.size)
    attr_text = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    body = f'<{tag} {attr_text} fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    return f'<svg width="128" height="128" xmlns="http://www.w3.org/2000/svg">{body}</svg>'


def _render_catalog_row(prompt_tokens: list[str], output_tokens: list[str], split: str, stage: str) -> dict:
    output_ir = " ".join(output_tokens)
    return {
        "stage": stage,
        "prompt": " ".join(prompt_tokens),
        "output_ir": output_ir,
        "svg_xml": render_ir(output_ir),
        "split": split,
    }


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _build_tokenizer_payload(domain_tokens: list[str]) -> tuple[dict, dict]:
    vocab_list: list[str] = []
    seen: set[str] = set()

    def _push(token: str) -> None:
        if token in seen:
            return
        seen.add(token)
        vocab_list.append(token)

    for token in SYSTEM_TOKENS:
        _push(token)
    for token in domain_tokens:
        _push(token)
    for token in ASCII_BASE:
        _push(token)

    vocab = {token: idx for idx, token in enumerate(vocab_list)}
    added_tokens = [
        {
            "id": vocab[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for token in vocab_list
        if token in seen and token not in ASCII_BASE
    ]
    tokenizer = {
        "version": "1.0",
        "ck_mode": "ascii_bpe",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<|unk|>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": True,
            "vocab": vocab,
            "merges": [],
        },
    }
    meta = {
        "schema": "ck.bpe.binary.v1",
        "mode": "ascii_bpe",
        "vocab_size": len(vocab_list),
        "num_merges": 0,
        "num_reserved_special_tokens": len(domain_tokens),
        "max_piece_bytes": 0,
    }
    return tokenizer, meta


def _write_tokenizer_artifacts(tokenizer: dict, meta: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json = out_dir / "tokenizer.json"
    tokenizer_json.write_text(json.dumps(tokenizer, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    vocab = tokenizer["model"]["vocab"]
    sorted_vocab = sorted(vocab.items(), key=lambda item: int(item[1]))
    offsets: list[int] = []
    strings_blob = b""
    current_offset = 0
    for token, _ in sorted_vocab:
        token_bytes = token.encode("utf-8")
        offsets.append(current_offset)
        strings_blob += token_bytes + b"\0"
        current_offset += len(token_bytes) + 1

    bin_dir = out_dir / "tokenizer_bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "vocab_offsets.bin").write_bytes(struct.pack("<" + ("i" * len(offsets)), *offsets))
    (bin_dir / "vocab_strings.bin").write_bytes(strings_blob)
    (bin_dir / "vocab_merges.bin").write_bytes(b"")

    meta_doc = dict(meta)
    meta_doc["offsets"] = str((bin_dir / "vocab_offsets.bin").resolve())
    meta_doc["strings"] = str((bin_dir / "vocab_strings.bin").resolve())
    meta_doc["merges"] = str((bin_dir / "vocab_merges.bin").resolve())
    (bin_dir / "tokenizer_meta.json").write_text(
        json.dumps(meta_doc, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return tokenizer_json, bin_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate semantic SVG shapes dataset with fixed vocab tokenizer")
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="toy_svg_semantic_shapes", help="Output file prefix")
    ap.add_argument("--train-repeats", type=int, default=8, help="Repeats for seen combos")
    ap.add_argument("--holdout-repeats", type=int, default=1, help="Repeats for holdout combos")
    args = ap.parse_args()

    if args.train_repeats < 1:
        raise SystemExit("--train-repeats must be >= 1")
    if args.holdout_repeats < 1:
        raise SystemExit("--holdout-repeats must be >= 1")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shape_combos = [Combo(shape=s, palette=p, size=z) for s, p, z in itertools.product(SHAPES, PALETTES, SIZES)]
    shape_holdout = [combo for combo in shape_combos if (combo.shape, combo.palette, combo.size) in HOLDOUT_COMBOS]
    shape_train = [combo for combo in shape_combos if (combo.shape, combo.palette, combo.size) not in HOLDOUT_COMBOS]

    card_combos = [CardCombo(theme=t, accent=a, bullets=b) for t, a, b in itertools.product(CARD_THEMES, CARD_ACCENTS, CARD_BULLETS)]
    card_holdout = [combo for combo in card_combos if (combo.theme, combo.accent, combo.bullets) in CARD_HOLDOUT_COMBOS]
    card_train = [combo for combo in card_combos if (combo.theme, combo.accent, combo.bullets) not in CARD_HOLDOUT_COMBOS]

    chart_combos = [ChartCombo(chart=c, trend=t) for c, t in itertools.product(CHART_TYPES, CHART_TRENDS)]
    chart_holdout = [combo for combo in chart_combos if (combo.chart, combo.trend) in CHART_HOLDOUT_COMBOS]
    chart_train = [combo for combo in chart_combos if (combo.chart, combo.trend) not in CHART_HOLDOUT_COMBOS]

    curve_combos = [CurveCombo(curve=c) for c in CURVE_FAMILIES]
    curve_holdout = [combo for combo in curve_combos if combo.curve in CURVE_HOLDOUT_COMBOS]
    curve_train = [combo for combo in curve_combos if combo.curve not in CURVE_HOLDOUT_COMBOS]

    stage1_train_rows = [_row(combo) for combo in shape_train for _ in range(int(args.train_repeats))]
    stage1_holdout_rows = [_row(combo) for combo in shape_holdout for _ in range(int(args.holdout_repeats))]
    stage2_train_rows = [_card_row(combo) for combo in card_train for _ in range(int(args.train_repeats))]
    stage2_holdout_rows = [_card_row(combo) for combo in card_holdout for _ in range(int(args.holdout_repeats))]
    stage3_train_rows = [_chart_row(combo) for combo in chart_train for _ in range(int(args.train_repeats))]
    stage3_holdout_rows = [_chart_row(combo) for combo in chart_holdout for _ in range(int(args.holdout_repeats))]
    stage4_train_rows = [_curve_row(combo) for combo in curve_train for _ in range(int(args.train_repeats))]
    stage4_holdout_rows = [_curve_row(combo) for combo in curve_holdout for _ in range(int(args.holdout_repeats))]

    train_rows = stage1_train_rows
    holdout_rows = stage1_holdout_rows
    all_train_rows = stage1_train_rows + stage2_train_rows + stage3_train_rows + stage4_train_rows
    all_holdout_rows = stage1_holdout_rows + stage2_holdout_rows + stage3_holdout_rows + stage4_holdout_rows

    tokenizer_corpus = (
        [_row(combo) for combo in shape_combos]
        + [_card_row(combo) for combo in card_combos]
        + [_chart_row(combo) for combo in chart_combos]
        + [_curve_row(combo) for combo in curve_combos]
        + [combo.prompt_text for combo in shape_combos]
        + [combo.prompt_text for combo in card_combos]
        + [combo.prompt_text for combo in chart_combos]
        + [combo.prompt_text for combo in curve_combos]
    )
    seen_prompts = [combo.prompt_text for combo in shape_train]
    holdout_prompts = [combo.prompt_text for combo in shape_holdout]
    card_seen_prompts = [combo.prompt_text for combo in card_train]
    card_holdout_prompts = [combo.prompt_text for combo in card_holdout]
    chart_seen_prompts = [combo.prompt_text for combo in chart_train]
    chart_holdout_prompts = [combo.prompt_text for combo in chart_holdout]
    curve_seen_prompts = [combo.prompt_text for combo in curve_train]
    curve_holdout_prompts = [combo.prompt_text for combo in curve_holdout]

    domain_tokens = (
        SHAPE_PROMPT_TOKENS
        + SHAPE_OUTPUT_TOKENS
        + CARD_PROMPT_TOKENS
        + CARD_OUTPUT_TOKENS
        + CHART_PROMPT_TOKENS
        + CHART_OUTPUT_TOKENS
        + CURVE_PROMPT_TOKENS
        + CURVE_OUTPUT_TOKENS
    )
    tokenizer, tokenizer_meta = _build_tokenizer_payload(domain_tokens)
    tokenizer_json, tokenizer_bin = _write_tokenizer_artifacts(
        tokenizer,
        tokenizer_meta,
        out_dir / f"{args.prefix}_tokenizer",
    )

    render_catalog = [
        _render_catalog_row(
            combo.prompt_tokens,
            ["[svg]", "[layout:center]", _resolved_shape_token(combo), f"[fill:{combo.palette}]", "[stroke:default]", "[/svg]"],
            "holdout" if combo in shape_holdout else "train",
            "stage1_shapes",
        )
        for combo in shape_combos
    ] + [
        _render_catalog_row(
            combo.prompt_tokens,
            [
                "[svg]",
                "[card:md]" if combo.bullets == 2 else "[card:lg]",
                "[bg:dark]" if combo.theme == "dark" else "[bg:light]",
                "[fg:light]" if combo.theme == "dark" else "[fg:dark]",
                f"[accent:{combo.accent}]",
                "[title_slot]",
                f"[bullet_slot:{combo.bullets}]",
                "[/svg]",
            ],
            "holdout" if combo in card_holdout else "train",
            "stage2_cards",
        )
        for combo in card_combos
    ] + [
        _render_catalog_row(
            combo.prompt_tokens,
            ["[svg]", f"[chart:{combo.chart}]", f"[bars3:{combo.trend}]", "[/svg]"],
            "holdout" if combo in chart_holdout else "train",
            "stage3_charts",
        )
        for combo in chart_combos
    ] + [
        _render_catalog_row(
            combo.prompt_tokens,
            ["[svg]", "[plot:curve]", f"[curve:{combo.curve}]", "[/svg]"],
            "holdout" if combo in curve_holdout else "train",
            "stage4_curves",
        )
        for combo in curve_combos
    ]

    future_vocab_groups = {
        "shape_prompt_tokens": SHAPE_PROMPT_TOKENS,
        "shape_output_tokens": SHAPE_OUTPUT_TOKENS,
        "card_prompt_tokens": CARD_PROMPT_TOKENS,
        "card_output_tokens": CARD_OUTPUT_TOKENS,
        "chart_prompt_tokens": CHART_PROMPT_TOKENS,
        "chart_output_tokens": CHART_OUTPUT_TOKENS,
        "curve_prompt_tokens": CURVE_PROMPT_TOKENS,
        "curve_output_tokens": CURVE_OUTPUT_TOKENS,
    }

    _write_lines(out_dir / f"{args.prefix}_train.txt", train_rows)
    _write_lines(out_dir / f"{args.prefix}_holdout.txt", holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_stage1_shapes_train.txt", stage1_train_rows)
    _write_lines(out_dir / f"{args.prefix}_stage1_shapes_holdout.txt", stage1_holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_stage2_cards_train.txt", stage2_train_rows)
    _write_lines(out_dir / f"{args.prefix}_stage2_cards_holdout.txt", stage2_holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_stage3_charts_train.txt", stage3_train_rows)
    _write_lines(out_dir / f"{args.prefix}_stage3_charts_holdout.txt", stage3_holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_stage4_curves_train.txt", stage4_train_rows)
    _write_lines(out_dir / f"{args.prefix}_stage4_curves_holdout.txt", stage4_holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_all_train.txt", all_train_rows)
    _write_lines(out_dir / f"{args.prefix}_all_holdout.txt", all_holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_tokenizer_corpus.txt", tokenizer_corpus)
    _write_lines(out_dir / f"{args.prefix}_seen_prompts.txt", seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_holdout_prompts.txt", holdout_prompts)
    _write_lines(out_dir / f"{args.prefix}_stage2_cards_seen_prompts.txt", card_seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_stage2_cards_holdout_prompts.txt", card_holdout_prompts)
    _write_lines(out_dir / f"{args.prefix}_stage3_charts_seen_prompts.txt", chart_seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_stage3_charts_holdout_prompts.txt", chart_holdout_prompts)
    _write_lines(out_dir / f"{args.prefix}_stage4_curves_seen_prompts.txt", curve_seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_stage4_curves_holdout_prompts.txt", curve_holdout_prompts)
    _write_lines(out_dir / f"{args.prefix}_reserved_control_tokens.txt", domain_tokens)

    (out_dir / f"{args.prefix}_render_catalog.json").write_text(
        json.dumps(render_catalog, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"{args.prefix}_future_vocab_groups.json").write_text(
        json.dumps(future_vocab_groups, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    vocab_spec = {
        "format": "svg-semantic-fixed-vocab.v1",
        "prefix": args.prefix,
        "system_tokens": SYSTEM_TOKENS,
        "future_vocab_groups": future_vocab_groups,
        "ascii_base_count": len(ASCII_BASE),
        "tokenizer_json": str(tokenizer_json),
        "tokenizer_bin": str(tokenizer_bin),
        "vocab_size": int(tokenizer_meta["vocab_size"]),
        "num_merges": int(tokenizer_meta["num_merges"]),
    }
    (out_dir / f"{args.prefix}_vocab.json").write_text(
        json.dumps(vocab_spec, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "format": "toy-svg-semantic-shapes-manifest.v1",
        "prefix": args.prefix,
        "out_dir": str(out_dir),
        "train_repeats": int(args.train_repeats),
        "holdout_repeats": int(args.holdout_repeats),
        "counts": {
            "unique_total": len(shape_combos) + len(card_combos) + len(chart_combos) + len(curve_combos),
            "unique_train": len(shape_train),
            "unique_holdout": len(shape_holdout),
            "shape_unique_total": len(shape_combos),
            "shape_unique_train": len(shape_train),
            "shape_unique_holdout": len(shape_holdout),
            "card_unique_total": len(card_combos),
            "card_unique_train": len(card_train),
            "card_unique_holdout": len(card_holdout),
            "chart_unique_total": len(chart_combos),
            "chart_unique_train": len(chart_train),
            "chart_unique_holdout": len(chart_holdout),
            "curve_unique_total": len(curve_combos),
            "curve_unique_train": len(curve_train),
            "curve_unique_holdout": len(curve_holdout),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "all_train_rows": len(all_train_rows),
            "all_holdout_rows": len(all_holdout_rows),
            "tokenizer_rows": len(tokenizer_corpus),
            "vocab_size": int(tokenizer_meta["vocab_size"]),
            "domain_token_count": len(domain_tokens),
        },
        "artifacts": {
            "train": str(out_dir / f"{args.prefix}_train.txt"),
            "holdout": str(out_dir / f"{args.prefix}_holdout.txt"),
            "stage1_shapes_train": str(out_dir / f"{args.prefix}_stage1_shapes_train.txt"),
            "stage1_shapes_holdout": str(out_dir / f"{args.prefix}_stage1_shapes_holdout.txt"),
            "stage2_cards_train": str(out_dir / f"{args.prefix}_stage2_cards_train.txt"),
            "stage2_cards_holdout": str(out_dir / f"{args.prefix}_stage2_cards_holdout.txt"),
            "stage3_charts_train": str(out_dir / f"{args.prefix}_stage3_charts_train.txt"),
            "stage3_charts_holdout": str(out_dir / f"{args.prefix}_stage3_charts_holdout.txt"),
            "stage4_curves_train": str(out_dir / f"{args.prefix}_stage4_curves_train.txt"),
            "stage4_curves_holdout": str(out_dir / f"{args.prefix}_stage4_curves_holdout.txt"),
            "all_train": str(out_dir / f"{args.prefix}_all_train.txt"),
            "all_holdout": str(out_dir / f"{args.prefix}_all_holdout.txt"),
            "tokenizer_corpus": str(out_dir / f"{args.prefix}_tokenizer_corpus.txt"),
            "seen_prompts": str(out_dir / f"{args.prefix}_seen_prompts.txt"),
            "holdout_prompts": str(out_dir / f"{args.prefix}_holdout_prompts.txt"),
            "stage2_cards_seen_prompts": str(out_dir / f"{args.prefix}_stage2_cards_seen_prompts.txt"),
            "stage2_cards_holdout_prompts": str(out_dir / f"{args.prefix}_stage2_cards_holdout_prompts.txt"),
            "stage3_charts_seen_prompts": str(out_dir / f"{args.prefix}_stage3_charts_seen_prompts.txt"),
            "stage3_charts_holdout_prompts": str(out_dir / f"{args.prefix}_stage3_charts_holdout_prompts.txt"),
            "stage4_curves_seen_prompts": str(out_dir / f"{args.prefix}_stage4_curves_seen_prompts.txt"),
            "stage4_curves_holdout_prompts": str(out_dir / f"{args.prefix}_stage4_curves_holdout_prompts.txt"),
            "reserved_control_tokens": str(out_dir / f"{args.prefix}_reserved_control_tokens.txt"),
            "future_vocab_groups": str(out_dir / f"{args.prefix}_future_vocab_groups.json"),
            "vocab": str(out_dir / f"{args.prefix}_vocab.json"),
            "render_catalog": str(out_dir / f"{args.prefix}_render_catalog.json"),
            "tokenizer_json": str(tokenizer_json),
            "tokenizer_bin": str(tokenizer_bin),
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
                "shape_unique_train": len(shape_train),
                "shape_unique_holdout": len(shape_holdout),
                "card_unique_train": len(card_train),
                "card_unique_holdout": len(card_holdout),
                "chart_unique_train": len(chart_train),
                "chart_unique_holdout": len(chart_holdout),
                "curve_unique_train": len(curve_train),
                "curve_unique_holdout": len(curve_holdout),
                "train_rows": len(train_rows),
                "holdout_rows": len(holdout_rows),
                "all_train_rows": len(all_train_rows),
                "all_holdout_rows": len(all_holdout_rows),
                "vocab_size": int(tokenizer_meta["vocab_size"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
