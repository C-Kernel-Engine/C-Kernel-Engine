#!/usr/bin/env python3
"""
Generate a richer structured SVG atoms dataset plus a fixed no-merge tokenizer.

The representation stays symbolic and renderer-backed:
- prompt side uses closed control tags
- output side is a flat SVG DSL made of stable atoms
- the DSL can express single shapes, paired compositions, label cards, and badges
"""

from __future__ import annotations

import argparse
import itertools
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from render_svg_structured_atoms_v7 import render_structured_svg_atoms


SYSTEM_TOKENS = ["<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>"]
ASCII_BASE = ["\t", "\n", "\r"] + [chr(i) for i in range(32, 127)]

SHAPES = ("circle", "rect", "triangle")
COLORS = ("red", "blue", "green", "orange", "purple")
BACKGROUNDS = ("none", "paper", "mint", "slate")
SIZES = ("small", "big")
LAYOUTS = ("single", "pair-h", "pair-v", "label-card", "badge")
LABELS = {
    "ok": "OK",
    "go": "GO",
    "data": "DATA",
    "note": "NOTE",
    "flow": "FLOW",
    "ai": "AI",
}


@dataclass(frozen=True)
class Scene:
    kind: str
    layout: str
    shape: str = "rect"
    shape2: str = "none"
    color: str = "blue"
    color2: str = "red"
    size: str = "small"
    bg: str = "none"
    label: str = "ok"

    @property
    def prompt_tokens(self) -> list[str]:
        tokens = ["[task:svg]", f"[layout:{self.layout}]"]
        if self.kind in {"single", "pair", "badge"}:
            tokens.append(f"[shape:{self.shape}]")
        if self.kind == "pair":
            tokens.append(f"[shape2:{self.shape2}]")
        if self.kind in {"single", "pair", "label-card", "badge"}:
            tokens.append(f"[color:{self.color}]")
        if self.kind == "pair":
            tokens.append(f"[color2:{self.color2}]")
        if self.kind in {"single", "pair"}:
            tokens.append(f"[size:{self.size}]")
        tokens.append(f"[bg:{self.bg}]")
        if self.kind in {"label-card", "badge"}:
            tokens.append(f"[label:{self.label}]")
        tokens.append("[OUT]")
        return tokens

    @property
    def prompt_text(self) -> str:
        return " ".join(self.prompt_tokens)


def _circle(cx: int, cy: int, r: int, fill: str, stroke: str = "black", sw: int = 2) -> list[str]:
    return [
        "[circle]",
        f"[cx:{cx}]",
        f"[cy:{cy}]",
        f"[r:{r}]",
        f"[fill:{fill}]",
        f"[stroke:{stroke}]",
        f"[sw:{sw}]",
    ]


def _rect(x: int, y: int, width: int, height: int, rx: int, fill: str, stroke: str = "black", sw: int = 2) -> list[str]:
    return [
        "[rect]",
        f"[x:{x}]",
        f"[y:{y}]",
        f"[width:{width}]",
        f"[height:{height}]",
        f"[rx:{rx}]",
        f"[fill:{fill}]",
        f"[stroke:{stroke}]",
        f"[sw:{sw}]",
    ]


def _triangle(points: str, fill: str, stroke: str = "black", sw: int = 2) -> list[str]:
    return [
        "[polygon]",
        f"[points:{points}]",
        f"[fill:{fill}]",
        f"[stroke:{stroke}]",
        f"[sw:{sw}]",
    ]


def _text(tx: int, ty: int, font: int, anchor: str, fill: str, content: str) -> list[str]:
    return [
        "[text]",
        f"[tx:{tx}]",
        f"[ty:{ty}]",
        f"[font:{font}]",
        f"[anchor:{anchor}]",
        f"[fill:{fill}]",
        content,
        "[/text]",
    ]


def _shape_tokens(shape: str, slot: str, size: str, color: str, stroke: str) -> list[str]:
    if slot == "single":
        if shape == "circle":
            return _circle(64, 64, 18 if size == "small" else 30, color, stroke)
        if shape == "rect":
            if size == "small":
                return _rect(38, 48, 52, 32, 8, color, stroke)
            return _rect(24, 38, 80, 52, 10, color, stroke)
        if size == "small":
            return _triangle("64,30|38,82|90,82", color, stroke)
        return _triangle("64,18|26,98|102,98", color, stroke)

    if slot == "pair-left":
        if shape == "circle":
            return _circle(36, 64, 16 if size == "small" else 22, color, stroke)
        if shape == "rect":
            if size == "small":
                return _rect(18, 52, 32, 24, 6, color, stroke)
            return _rect(12, 47, 44, 34, 7, color, stroke)
        if size == "small":
            return _triangle("36,36|18,78|54,78", color, stroke)
        return _triangle("36,28|12,84|60,84", color, stroke)

    if slot == "pair-right":
        if shape == "circle":
            return _circle(92, 64, 16 if size == "small" else 22, color, stroke)
        if shape == "rect":
            if size == "small":
                return _rect(74, 52, 32, 24, 6, color, stroke)
            return _rect(70, 47, 44, 34, 7, color, stroke)
        if size == "small":
            return _triangle("92,36|74,78|110,78", color, stroke)
        return _triangle("92,28|68,84|116,84", color, stroke)

    if slot == "pair-top":
        if shape == "circle":
            return _circle(64, 38, 16 if size == "small" else 22, color, stroke)
        if shape == "rect":
            if size == "small":
                return _rect(48, 24, 32, 24, 6, color, stroke)
            return _rect(42, 18, 44, 34, 7, color, stroke)
        if size == "small":
            return _triangle("64,18|46,54|82,54", color, stroke)
        return _triangle("64,10|40,62|88,62", color, stroke)

    if slot == "pair-bottom":
        if shape == "circle":
            return _circle(64, 90, 16 if size == "small" else 22, color, stroke)
        if shape == "rect":
            if size == "small":
                return _rect(48, 76, 32, 24, 6, color, stroke)
            return _rect(42, 70, 44, 34, 7, color, stroke)
        if size == "small":
            return _triangle("64,70|46,106|82,106", color, stroke)
        return _triangle("64,62|40,114|88,114", color, stroke)

    if slot == "badge-icon":
        if shape == "circle":
            return _circle(30, 57, 12, color, stroke)
        if shape == "rect":
            return _rect(18, 45, 24, 24, 5, color, stroke)
        return _triangle("30,43|18,67|42,67", color, stroke)

    raise RuntimeError(f"unsupported slot: {slot}")


def _scene_output_tokens(scene: Scene) -> list[str]:
    tokens = ["[svg]", "[w:128]", "[h:128]", f"[bg:{scene.bg}]", f"[layout:{scene.layout}]"]
    icon_stroke = "white" if scene.bg == "slate" else "black"
    label_text = LABELS[scene.label]

    if scene.kind == "single":
        tokens.extend(_shape_tokens(scene.shape, "single", scene.size, scene.color, icon_stroke))
    elif scene.kind == "pair":
        first_slot = "pair-left" if scene.layout == "pair-h" else "pair-top"
        second_slot = "pair-right" if scene.layout == "pair-h" else "pair-bottom"
        tokens.extend(_shape_tokens(scene.shape, first_slot, scene.size, scene.color, icon_stroke))
        tokens.extend(_shape_tokens(scene.shape2, second_slot, scene.size, scene.color2, icon_stroke))
    elif scene.kind == "label-card":
        card_stroke = "white" if scene.bg == "slate" else "black"
        text_fill = "white"
        tokens.extend(_rect(18, 38, 92, 44, 8, scene.color, card_stroke))
        tokens.extend(_text(64, 64, 14, "middle", text_fill, label_text))
    elif scene.kind == "badge":
        tokens.extend(_rect(14, 40, 100, 34, 8, "white", icon_stroke))
        tokens.extend(_shape_tokens(scene.shape, "badge-icon", "small", scene.color, icon_stroke))
        tokens.extend(_text(52, 61, 12, "start", "black", label_text))
    else:
        raise RuntimeError(f"unsupported scene kind: {scene.kind}")

    tokens.append("[/svg]")
    return tokens


def _row(scene: Scene) -> str:
    return " ".join(scene.prompt_tokens + _scene_output_tokens(scene))


def _render_svg(scene: Scene) -> str:
    return render_structured_svg_atoms(" ".join(_scene_output_tokens(scene)))


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
        for token in (SYSTEM_TOKENS + domain_tokens)
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


def _scene_sort_key(scene: Scene) -> tuple[str, str, str, str, str, str, str, str]:
    return (scene.kind, scene.layout, scene.shape, scene.shape2, scene.color, scene.color2, scene.size, scene.label)


def _is_holdout(scene: Scene) -> bool:
    if scene.kind == "single":
        return scene.color in {"orange", "purple"} and scene.size == "big"
    if scene.kind == "pair":
        return scene.layout == "pair-v" and scene.shape == "triangle" and scene.color in {"orange", "purple"}
    if scene.kind == "label-card":
        return scene.label in {"flow", "ai"} and scene.bg == "slate"
    if scene.kind == "badge":
        return scene.shape == "triangle" and scene.label in {"data", "note"}
    return False


def _build_scenes() -> list[Scene]:
    scenes: list[Scene] = []

    for shape, color, size, bg in itertools.product(SHAPES, COLORS, SIZES, ("none", "paper")):
        scenes.append(Scene(kind="single", layout="single", shape=shape, color=color, size=size, bg=bg))

    pair_bg = ("paper", "mint")
    for layout, shape, shape2, color, color2, size, bg in itertools.product(
        ("pair-h", "pair-v"),
        SHAPES,
        SHAPES,
        COLORS,
        COLORS,
        SIZES,
        pair_bg,
    ):
        if shape == shape2 or color == color2:
            continue
        scenes.append(
            Scene(
                kind="pair",
                layout=layout,
                shape=shape,
                shape2=shape2,
                color=color,
                color2=color2,
                size=size,
                bg=bg,
            )
        )

    for color, label, bg in itertools.product(COLORS, LABELS, ("paper", "slate")):
        scenes.append(Scene(kind="label-card", layout="label-card", color=color, bg=bg, label=label))

    for shape, color, label, bg in itertools.product(SHAPES, COLORS, LABELS, ("paper", "mint")):
        scenes.append(Scene(kind="badge", layout="badge", shape=shape, color=color, bg=bg, label=label))

    return sorted(scenes, key=_scene_sort_key)


def _ordered_domain_tokens(rows: list[str]) -> list[str]:
    domain_tokens: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for token in row.split():
            if not (token.startswith("[") and token.endswith("]")):
                continue
            if token in seen:
                continue
            seen.add(token)
            domain_tokens.append(token)
    return domain_tokens


def _write_probe_report_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_probe_report_contract.json"
    payload = {
        "schema": "ck.probe_report_contract.v1",
        "title": "Structured SVG Scene Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/svg]"],
            "renderer": "structured_svg_atoms.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 80,
            "temperature": 0.0,
            "stop_on_text": "<|eos|>",
        },
        "catalog": {
            "format": "json_rows",
            "path": f"{prefix}_render_catalog.json",
            "prompt_key": "prompt",
            "output_key": "output_tokens",
            "rendered_key": "svg_xml",
            "rendered_mime": "image/svg+xml",
            "split_key": "split",
        },
        "splits": [
            {
                "name": "train",
                "label": "Train prompts",
                "format": "lines",
                "path": f"{prefix}_seen_prompts.txt",
                "limit": 10,
            },
            {
                "name": "test",
                "label": "Holdout prompts",
                "format": "lines",
                "path": f"{prefix}_holdout_prompts.txt",
                "limit": 10,
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _probe_record(scene: Scene, probe_type: str = "svg_gen") -> dict[str, str]:
    expect_shape = scene.shape if scene.kind in {"single", "pair", "badge"} else "rect"
    return {
        "id": (
            f"{scene.kind}_{scene.layout}_{scene.shape}_{scene.shape2}_"
            f"{scene.color}_{scene.color2}_{scene.size}_{scene.label}_{scene.bg}"
        ).replace(":", "_"),
        "description": f"Structured SVG {scene.kind} probe for {scene.prompt_text}",
        "prompt": scene.prompt_text,
        "type": probe_type,
        "expect_shape": expect_shape,
        "expect_color": scene.color,
    }


def _write_eval_contract(out_dir: Path, prefix: str, holdout_scenes: list[Scene]) -> Path:
    path = out_dir / f"{prefix}_eval_contract.json"
    selected: list[Scene] = []
    for kind in ("single", "pair", "label-card", "badge"):
        matches = [scene for scene in holdout_scenes if scene.kind == kind][:2]
        selected.extend(matches)
    probes = [_probe_record(scene) for scene in selected]
    probes.extend(
        [
            {
                "id": "ood_missing_color",
                "description": "OOD probe: same DSL family but missing color tag.",
                "prompt": "[task:svg] [layout:single] [shape:circle] [size:small] [bg:paper] [OUT]",
                "type": "ood",
            },
            {
                "id": "ood_badge_missing_label",
                "description": "OOD probe: badge layout without label tag.",
                "prompt": "[task:svg] [layout:badge] [shape:triangle] [color:purple] [bg:mint] [OUT]",
                "type": "ood",
            },
        ]
    )
    payload = {
        "schema": "ck.eval_contract.v1",
        "dataset_type": "svg",
        "goal": "Render structured SVG DSL outputs that materialize into valid SVG and respect the scene control tags.",
        "notes": [
            "This line emits structured SVG atoms, not raw XML.",
            "Use the output_adapter to materialize the DSL into SVG before computing valid_svg_rate.",
            "expect_color is a spec04 extension used for literal color control instead of palette buckets.",
        ],
        "scorer": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/svg]"],
            "renderer": "structured_svg_atoms.v1",
            "preview_mime": "image/svg+xml",
        },
        "stage_metrics": [
            {"key": "valid_svg_rate", "label": "Valid SVG", "description": "Fraction of probe outputs that materialize into valid SVG.", "source": "valid_svg", "probe_type": "svg_gen", "good": 0.75, "warn": 0.35, "format": "pct", "higher_is_better": True, "headline": True},
            {"key": "closure_success_rate", "label": "Closure", "description": "Fraction of outputs that reach the closing [/svg] and render to a closed SVG.", "source": "closure", "probe_type": "svg_gen", "good": 0.70, "warn": 0.30, "format": "pct", "higher_is_better": True},
            {"key": "prefix_integrity", "label": "Prefix", "description": "How often output begins cleanly at the expected DSL boundary.", "source": "prefix_integrity", "probe_type": "all", "good": 0.80, "warn": 0.40, "format": "pct", "higher_is_better": True},
            {"key": "ood_robustness", "label": "OOD", "description": "Materialized SVG validity under prompt omissions inside the same DSL family.", "source": "valid_svg", "probe_type": "ood", "good": 0.50, "warn": 0.20, "format": "pct", "higher_is_better": True, "headline": True},
            {"key": "adherence", "label": "Adherence", "description": "Instruction adherence score based on expected shape and literal color control.", "source": "adherence", "probe_type": "svg_gen", "good": 0.70, "warn": 0.30, "format": "pct", "higher_is_better": True, "regression_watch": True, "headline": True},
            {"key": "repetition_loop_score", "label": "Loop Score", "description": "Repetition risk on the emitted DSL token stream (lower is better).", "source": "repetition", "probe_type": "all", "good": 0.10, "warn": 0.30, "format": "pct", "higher_is_better": False},
            {"key": "tag_adherence", "label": "Tag Adh", "description": "Match score for explicit control tags such as shape/color atoms echoed in the DSL.", "source": "tag_adherence", "probe_type": "svg_gen", "good": 0.70, "warn": 0.30, "format": "pct", "higher_is_better": True},
        ],
        "probe_metrics": [
            {"key": "valid_svg", "label": "Valid", "description": "This probe materialized into valid SVG.", "good": 0.75, "warn": 0.35, "format": "pct", "higher_is_better": True},
            {"key": "closure", "label": "Closure", "description": "This probe reaches a closed rendered SVG.", "good": 0.70, "warn": 0.30, "format": "pct", "higher_is_better": True},
            {"key": "prefix_integrity", "label": "Prefix", "description": "Output starts cleanly at the DSL boundary.", "good": 0.80, "warn": 0.40, "format": "pct", "higher_is_better": True},
            {"key": "repetition", "label": "Loop", "description": "Repetition score on the emitted DSL (lower is better).", "good": 0.10, "warn": 0.30, "format": "pct", "higher_is_better": False},
            {"key": "adherence", "label": "Adh", "description": "Instruction adherence for this specific structured scene prompt.", "good": 0.70, "warn": 0.30, "format": "pct", "higher_is_better": True},
            {"key": "tag_adherence", "label": "Tag Adh", "description": "Tag-level adherence for this structured scene prompt.", "good": 0.70, "warn": 0.30, "format": "pct", "higher_is_better": True},
        ],
        "headline_metrics": ["valid_svg_rate", "ood_robustness", "adherence"],
        "probes": probes,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate structured SVG scene dataset with fixed vocab tokenizer")
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="toy_svg_structured_atoms", help="Output file prefix")
    ap.add_argument("--train-repeats", type=int, default=12, help="Repeats for seen combos")
    ap.add_argument("--holdout-repeats", type=int, default=1, help="Repeats for holdout combos")
    args = ap.parse_args()

    if args.train_repeats < 1:
        raise SystemExit("--train-repeats must be >= 1")
    if args.holdout_repeats < 1:
        raise SystemExit("--holdout-repeats must be >= 1")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = _build_scenes()
    holdout = [scene for scene in scenes if _is_holdout(scene)]
    train = [scene for scene in scenes if not _is_holdout(scene)]

    train_rows = [_row(scene) for scene in train for _ in range(int(args.train_repeats))]
    holdout_rows = [_row(scene) for scene in holdout for _ in range(int(args.holdout_repeats))]
    tokenizer_corpus = [_row(scene) for scene in scenes] + [scene.prompt_text for scene in scenes]
    seen_prompts = [scene.prompt_text for scene in train]
    holdout_prompts = [scene.prompt_text for scene in holdout]
    render_rows = [
        {
            "prompt": scene.prompt_text,
            "output_tokens": " ".join(_scene_output_tokens(scene)),
            "svg_xml": _render_svg(scene),
            "split": "holdout" if scene in holdout else "train",
            "kind": scene.kind,
            "layout": scene.layout,
        }
        for scene in scenes
    ]

    domain_tokens = _ordered_domain_tokens(tokenizer_corpus)
    tokenizer, tokenizer_meta = _build_tokenizer_payload(domain_tokens)
    tokenizer_json, tokenizer_bin = _write_tokenizer_artifacts(
        tokenizer,
        tokenizer_meta,
        out_dir / f"{args.prefix}_tokenizer",
    )

    _write_lines(out_dir / f"{args.prefix}_train.txt", train_rows)
    _write_lines(out_dir / f"{args.prefix}_holdout.txt", holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_tokenizer_corpus.txt", tokenizer_corpus)
    _write_lines(out_dir / f"{args.prefix}_seen_prompts.txt", seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_holdout_prompts.txt", holdout_prompts)
    _write_lines(out_dir / f"{args.prefix}_reserved_control_tokens.txt", domain_tokens)

    vocab_spec = {
        "format": "svg-structured-fixed-vocab.v2",
        "prefix": args.prefix,
        "system_tokens": SYSTEM_TOKENS,
        "domain_tokens": domain_tokens,
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
    (out_dir / f"{args.prefix}_render_catalog.json").write_text(
        json.dumps(render_rows, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    probe_contract_path = _write_probe_report_contract(out_dir, args.prefix)
    eval_contract_path = _write_eval_contract(out_dir, args.prefix, holdout)

    kind_counts: dict[str, dict[str, int]] = {}
    for kind in ("single", "pair", "label-card", "badge"):
        kind_counts[kind] = {
            "train": sum(1 for scene in train if scene.kind == kind),
            "holdout": sum(1 for scene in holdout if scene.kind == kind),
        }

    manifest = {
        "format": "structured-svg-atoms-manifest.v2",
        "prefix": args.prefix,
        "out_dir": str(out_dir),
        "train_repeats": int(args.train_repeats),
        "holdout_repeats": int(args.holdout_repeats),
        "counts": {
            "unique_total": len(scenes),
            "unique_train": len(train),
            "unique_holdout": len(holdout),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "tokenizer_rows": len(tokenizer_corpus),
            "vocab_size": int(tokenizer_meta["vocab_size"]),
            "domain_token_count": len(domain_tokens),
            "scene_kind_counts": kind_counts,
        },
        "artifacts": {
            "train": str(out_dir / f"{args.prefix}_train.txt"),
            "holdout": str(out_dir / f"{args.prefix}_holdout.txt"),
            "tokenizer_corpus": str(out_dir / f"{args.prefix}_tokenizer_corpus.txt"),
            "seen_prompts": str(out_dir / f"{args.prefix}_seen_prompts.txt"),
            "holdout_prompts": str(out_dir / f"{args.prefix}_holdout_prompts.txt"),
            "reserved_control_tokens": str(out_dir / f"{args.prefix}_reserved_control_tokens.txt"),
            "vocab": str(out_dir / f"{args.prefix}_vocab.json"),
            "render_catalog": str(out_dir / f"{args.prefix}_render_catalog.json"),
            "probe_report_contract": str(probe_contract_path),
            "eval_contract": str(eval_contract_path),
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
                "unique_train": len(train),
                "unique_holdout": len(holdout),
                "train_rows": len(train_rows),
                "holdout_rows": len(holdout_rows),
                "scene_kind_counts": kind_counts,
                "vocab_size": int(tokenizer_meta["vocab_size"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
