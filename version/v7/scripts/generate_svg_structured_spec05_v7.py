#!/usr/bin/env python3
"""
Generate a richer structured SVG atoms dataset for spec05.

Spec05 keeps the renderer-backed SVG atom DSL from spec04, but expands the
control space with simple composition controls:
- density: compact vs airy spacing
- frame: plain vs card framing

The goal is not open-ended art. It is a denser structured generation benchmark
that spans simple single-object scenes through multi-element compositions while
remaining deterministic and automatically scorable.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from render_svg_structured_atoms_v7 import render_structured_svg_atoms


SYSTEM_TOKENS = ["<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>"]
ASCII_BASE = ["\t", "\n", "\r"] + [chr(i) for i in range(32, 127)]

SHAPES = ("circle", "rect", "triangle")
COLORS = ("red", "blue", "green", "orange", "purple", "teal", "gold", "gray")
BACKGROUNDS = ("none", "paper", "mint", "slate")
SIZES = ("small", "big")
DENSITIES = ("compact", "airy")
FRAMES = ("plain", "card")
LAYOUTS = ("single", "pair-h", "pair-v", "label-card", "badge")
LABELS = {
    "ok": "OK",
    "go": "GO",
    "data": "DATA",
    "note": "NOTE",
    "flow": "FLOW",
    "ai": "AI",
    "ops": "OPS",
    "sync": "SYNC",
    "map": "MAP",
    "plan": "PLAN",
}

PAIR_COLOR_ALT = {
    "red": "blue",
    "blue": "green",
    "green": "orange",
    "orange": "purple",
    "purple": "teal",
    "teal": "gold",
    "gold": "gray",
    "gray": "red",
}

PAIR_COLOR_ALT_V = {
    "red": "teal",
    "blue": "orange",
    "green": "purple",
    "orange": "gray",
    "purple": "gold",
    "teal": "red",
    "gold": "blue",
    "gray": "green",
}

PAIR_SHAPE_ALT = {
    "circle": "rect",
    "rect": "triangle",
    "triangle": "circle",
}

PAIR_SHAPE_ALT_V = {
    "circle": "triangle",
    "rect": "circle",
    "triangle": "rect",
}


def _pair_color_options(primary: str, layout: str, density: str) -> tuple[str, ...]:
    if layout == "pair-h":
        return (PAIR_COLOR_ALT[primary],)
    options = [PAIR_COLOR_ALT_V[primary]]
    # Add one train-only alternate secondary color so pair-v must bind color2 explicitly.
    if not (density == "airy" and primary in {"teal", "gold"}):
        extra = PAIR_COLOR_ALT[primary]
        if extra != primary and extra not in options:
            options.append(extra)
    return tuple(options)


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
    density: str = "compact"
    frame: str = "plain"

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
        tokens.append(f"[frame:{self.frame}]")
        tokens.append(f"[density:{self.density}]")
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


def _stroke_for_bg(bg: str) -> str:
    return "white" if bg == "slate" else "black"


def _frame_fill(bg: str) -> str:
    if bg == "slate":
        return "gray"
    if bg == "mint":
        return "white"
    if bg == "paper":
        return "white"
    return "paper"


def _single_geom(size: str, density: str) -> tuple[int, int, int, int, int]:
    if size == "big":
        if density == "compact":
            return 64, 64, 32, 84, 56
        return 64, 64, 26, 72, 48
    if density == "compact":
        return 64, 64, 22, 62, 42
    return 64, 64, 18, 52, 34


def _pair_centers(layout: str, density: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if layout == "pair-h":
        if density == "compact":
            return (40, 64), (88, 64)
        return (34, 64), (94, 64)
    if density == "compact":
        return (64, 42), (64, 86)
    return (64, 34), (64, 94)


def _pair_extent(size: str, density: str) -> tuple[int, int, int]:
    if size == "big":
        return (24, 18, 28) if density == "compact" else (20, 16, 24)
    return (18, 14, 22) if density == "compact" else (16, 12, 18)


def _shape_tokens(
    shape: str,
    slot: str,
    size: str,
    color: str,
    stroke: str,
    density: str,
) -> list[str]:
    if slot == "single":
        cx, cy, circle_r, rect_w, rect_h = _single_geom(size, density)
        rect_x = cx - (rect_w // 2)
        rect_y = cy - (rect_h // 2)
        tri_top = rect_y - 8
        tri_left = rect_x
        tri_right = rect_x + rect_w
        tri_bottom = rect_y + rect_h
        if shape == "circle":
            return _circle(cx, cy, circle_r, color, stroke)
        if shape == "rect":
            return _rect(rect_x, rect_y, rect_w, rect_h, 10 if size == "big" else 8, color, stroke)
        return _triangle(
            f"{cx},{tri_top}|{tri_left},{tri_bottom}|{tri_right},{tri_bottom}",
            color,
            stroke,
        )

    if slot in {"pair-left", "pair-right", "pair-top", "pair-bottom"}:
        layout = "pair-h" if slot in {"pair-left", "pair-right"} else "pair-v"
        first, second = _pair_centers(layout, density)
        cx, cy = first if slot in {"pair-left", "pair-top"} else second
        rect_w, circle_r, tri_h = _pair_extent(size, density)
        rect_h = rect_w - 6 if size == "small" else rect_w - 4
        rect_x = cx - (rect_w // 2)
        rect_y = cy - (rect_h // 2)
        tri_top = cy - tri_h
        tri_bottom = cy + tri_h
        tri_left = cx - rect_w // 2
        tri_right = cx + rect_w // 2
        if shape == "circle":
            return _circle(cx, cy, circle_r, color, stroke)
        if shape == "rect":
            return _rect(rect_x, rect_y, rect_w, rect_h, 7 if size == "big" else 6, color, stroke)
        return _triangle(
            f"{cx},{tri_top}|{tri_left},{tri_bottom}|{tri_right},{tri_bottom}",
            color,
            stroke,
        )

    if slot == "badge-icon":
        if density == "compact":
            cx, cy, circle_r = 28, 58, 13
            rect_x, rect_y, rect_w, rect_h = 16, 46, 26, 24
            tri = "28,44|16,70|40,70"
        else:
            cx, cy, circle_r = 32, 57, 11
            rect_x, rect_y, rect_w, rect_h = 20, 46, 22, 22
            tri = "31,45|19,67|43,67"
        if shape == "circle":
            return _circle(cx, cy, circle_r, color, stroke)
        if shape == "rect":
            return _rect(rect_x, rect_y, rect_w, rect_h, 5, color, stroke)
        return _triangle(tri, color, stroke)

    raise RuntimeError(f"unsupported slot: {slot}")


def _frame_tokens(scene: Scene, stroke: str) -> list[str]:
    if scene.frame != "card":
        return []
    fill = _frame_fill(scene.bg)
    if scene.kind == "single":
        return _rect(14, 14, 100, 100, 18, fill, stroke)
    if scene.kind == "pair":
        return _rect(10, 20, 108, 88, 18, fill, stroke)
    if scene.kind == "label-card":
        return _rect(10, 26, 108, 76, 18, fill, stroke)
    if scene.kind == "badge":
        return _rect(10, 32, 108, 50, 16, fill, stroke)
    return []


def _scene_output_tokens(scene: Scene) -> list[str]:
    tokens = [
        "[svg]",
        "[w:128]",
        "[h:128]",
        f"[bg:{scene.bg}]",
        f"[layout:{scene.layout}]",
        f"[frame:{scene.frame}]",
        f"[density:{scene.density}]",
    ]
    stroke = _stroke_for_bg(scene.bg)
    label_text = LABELS[scene.label]
    text_fill = "white" if scene.bg == "slate" else "black"

    tokens.extend(_frame_tokens(scene, stroke))

    if scene.kind == "single":
        tokens.extend(_shape_tokens(scene.shape, "single", scene.size, scene.color, stroke, scene.density))
    elif scene.kind == "pair":
        first_slot = "pair-left" if scene.layout == "pair-h" else "pair-top"
        second_slot = "pair-right" if scene.layout == "pair-h" else "pair-bottom"
        tokens.extend(_shape_tokens(scene.shape, first_slot, scene.size, scene.color, stroke, scene.density))
        tokens.extend(_shape_tokens(scene.shape2, second_slot, scene.size, scene.color2, stroke, scene.density))
    elif scene.kind == "label-card":
        inner_x = 18 if scene.frame == "plain" else 20
        inner_y = 38 if scene.frame == "plain" else 42
        inner_w = 92 if scene.frame == "plain" else 88
        inner_h = 44 if scene.frame == "plain" else 36
        radius = 10 if scene.frame == "card" else 8
        font = 14 if scene.density == "compact" else 13
        text_y = 65 if scene.density == "compact" else 64
        card_stroke = "white" if scene.bg == "slate" else "black"
        fill = scene.color
        if scene.frame == "card" and scene.bg == "slate":
            fill = scene.color
            text_fill = "white"
        tokens.extend(_rect(inner_x, inner_y, inner_w, inner_h, radius, fill, card_stroke))
        tokens.extend(_text(64, text_y, font, "middle", text_fill, label_text))
    elif scene.kind == "badge":
        if scene.frame == "card":
            badge_fill = _frame_fill(scene.bg)
            badge_stroke = stroke
            inner = _rect(18, 42, 92, 30, 10, badge_fill, badge_stroke)
        else:
            inner = _rect(14, 40, 100, 34, 8, "white", stroke)
        text_x = 48 if scene.density == "compact" else 52
        text_y = 61 if scene.density == "compact" else 60
        font = 12 if scene.density == "compact" else 11
        tokens.extend(inner)
        tokens.extend(_shape_tokens(scene.shape, "badge-icon", "small", scene.color, stroke, scene.density))
        tokens.extend(_text(text_x, text_y, font, "start", "black", label_text))
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


def _scene_sort_key(scene: Scene) -> tuple[str, str, str, str, str, str, str, str, str, str]:
    return (
        scene.kind,
        scene.layout,
        scene.shape,
        scene.shape2,
        scene.color,
        scene.color2,
        scene.size,
        scene.label,
        scene.frame,
        scene.density,
    )


def _is_holdout(scene: Scene) -> bool:
    if scene.kind == "single":
        return scene.color in {"teal", "gold"} and scene.bg in {"mint", "slate"} and scene.frame == "card"
    if scene.kind == "pair":
        return scene.layout == "pair-v" and scene.color in {"teal", "gold"} and scene.density == "airy"
    if scene.kind == "label-card":
        return scene.label in {"flow", "ops", "plan"} and scene.bg == "slate" and scene.frame == "card"
    if scene.kind == "badge":
        return scene.shape == "triangle" and scene.label in {"note", "sync", "map"} and scene.frame == "card"
    return False


def _build_scenes() -> list[Scene]:
    scenes: list[Scene] = []

    for shape in SHAPES:
        for color in COLORS:
            for size in SIZES:
                for bg in BACKGROUNDS:
                    for density in DENSITIES:
                        for frame in FRAMES:
                            scenes.append(
                                Scene(
                                    kind="single",
                                    layout="single",
                                    shape=shape,
                                    color=color,
                                    size=size,
                                    bg=bg,
                                    density=density,
                                    frame=frame,
                                )
                            )

    for layout in ("pair-h", "pair-v"):
        for shape in SHAPES:
            shape2 = PAIR_SHAPE_ALT[shape] if layout == "pair-h" else PAIR_SHAPE_ALT_V[shape]
            for color in COLORS:
                for size in SIZES:
                    for bg in ("paper", "mint", "slate"):
                        for density in DENSITIES:
                            for frame in FRAMES:
                                for color2 in _pair_color_options(color, layout, density):
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
                                            density=density,
                                            frame=frame,
                                        )
                                    )

    for color in COLORS:
        for label in LABELS:
            for bg in ("paper", "slate"):
                for density in DENSITIES:
                    for frame in FRAMES:
                        scenes.append(
                            Scene(
                                kind="label-card",
                                layout="label-card",
                                color=color,
                                bg=bg,
                                label=label,
                                density=density,
                                frame=frame,
                            )
                        )

    for shape in SHAPES:
        for color in COLORS:
            for label in LABELS:
                for bg in ("paper", "mint"):
                    for density in DENSITIES:
                        for frame in FRAMES:
                            scenes.append(
                                Scene(
                                    kind="badge",
                                    layout="badge",
                                    shape=shape,
                                    color=color,
                                    bg=bg,
                                    label=label,
                                    density=density,
                                    frame=frame,
                                )
                            )

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
        "title": "Spec05 Structured Scenes Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/svg]"],
            "renderer": "structured_svg_atoms.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 128,
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
            f"{scene.color}_{scene.color2}_{scene.size}_{scene.label}_{scene.bg}_"
            f"{scene.frame}_{scene.density}"
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
                "prompt": "[task:svg] [layout:single] [shape:circle] [size:small] [bg:paper] [frame:plain] [density:compact] [OUT]",
                "type": "ood",
            },
            {
                "id": "ood_badge_missing_label",
                "description": "OOD probe: badge layout without label tag.",
                "prompt": "[task:svg] [layout:badge] [shape:triangle] [color:teal] [bg:mint] [frame:card] [density:airy] [OUT]",
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
            "Spec05 adds frame and density composition controls while keeping literal shape/color control.",
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
    ap = argparse.ArgumentParser(description="Generate spec05 structured SVG scene dataset with fixed vocab tokenizer")
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec05_structured_svg_atoms", help="Output file prefix")
    ap.add_argument("--train-repeats", type=int, default=4, help="Repeats for seen combos")
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
        "format": "svg-structured-fixed-vocab.v3",
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
        "format": "structured-svg-atoms-manifest.v3",
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
