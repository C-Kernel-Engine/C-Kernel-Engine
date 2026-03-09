#!/usr/bin/env python3
"""
Generate a composition-first SVG DSL curriculum for v7.

This path intentionally sits between spec02 and spec03:
- closed control tags like spec02
- richer SVG composition than single-shape toy rows
- no direct dependency on full infographic asset corpora

Outputs:
- per-stage instruction train/holdout/all text files
- per-stage svg-only train/holdout/all text files
- combined curriculum instruction/svg train/holdout/all files
- reserved control tokens file
- spec catalog json
- coverage manifests
- summary manifest
"""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PALETTES: dict[str, tuple[str, str, str]] = {
    "mono": ("white", "black", "gray"),
    "cool": ("#E0F2FE", "#0284C7", "#0C4A6E"),
    "warm": ("#FFEDD5", "#EA580C", "#7C2D12"),
    "bold": ("#111827", "#22C55E", "#F9FAFB"),
}

TITLE_WORDS = [
    "ALPHA",
    "BETA",
    "DELTA",
    "FOCUS",
    "STACK",
    "FRAME",
    "SHAPE",
    "CHART",
]

BULLET_WORDS = [
    "ALIGN",
    "GROUP",
    "ORDER",
    "FLOW",
    "SCALE",
    "BALANCE",
    "LABEL",
    "VALUE",
]

TAG_GROUP_ORDER = [
    "component",
    "shape",
    "combo",
    "composition",
    "palette",
    "style",
    "layout",
    "text",
    "count",
]


@dataclass(frozen=True)
class Spec:
    id: str
    stage: str
    tags: tuple[str, ...]
    generator: str
    samples_train: int
    samples_holdout: int
    description: str
    group: str = "svg_basics"
    shape: str = ""
    combo: str = ""
    palette_family: str = "mono"
    style: str = "outline"
    layout: str = "center"
    text_mode: str = "none"
    count: int = 1


@dataclass
class Sample:
    spec_id: str
    stage: str
    sample_type: str
    tags: tuple[str, ...]
    svg: str

    def instruction_line(self) -> str:
        return f"{''.join(self.tags)}{self.svg}<eos>"


def _assert_ascii(text: str, what: str) -> None:
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"non-ascii content in {what}: {exc}") from exc


def _validate_svg(svg: str) -> None:
    root = ET.fromstring(svg)
    tag = root.tag.split("}", 1)[-1] if "}" in root.tag else root.tag
    if tag != "svg":
        raise RuntimeError(f"invalid root tag: {tag}")


def _svg_wrap(width: int, height: int, body: str) -> str:
    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'


def _choice(rng: random.Random, values: list[str]) -> str:
    return values[rng.randrange(len(values))]


def _layout_center(layout: str, width: int, height: int) -> tuple[int, int]:
    if layout == "left":
        return int(width * 0.28), height // 2
    if layout == "right":
        return int(width * 0.72), height // 2
    if layout == "top":
        return width // 2, int(height * 0.28)
    if layout == "bottom":
        return width // 2, int(height * 0.72)
    return width // 2, height // 2


def _shape_body(shape: str, cx: int, cy: int, fill: str, stroke: str, *, filled: bool, rng: random.Random) -> str:
    sw = 2
    if shape == "circle":
        r = rng.randint(18, 28)
        return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill if filled else "none"}" stroke="{stroke}" stroke-width="{sw}"/>'
    if shape == "rect":
        rw = rng.randint(54, 78)
        rh = rng.randint(32, 48)
        x = cx - rw // 2
        y = cy - rh // 2
        rx = rng.randint(6, 12)
        return (
            f'<rect x="{x}" y="{y}" width="{rw}" height="{rh}" rx="{rx}" '
            f'fill="{fill if filled else "none"}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
    if shape == "ellipse":
        rx = rng.randint(24, 34)
        ry = rng.randint(16, 24)
        return (
            f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" '
            f'fill="{fill if filled else "none"}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
    if shape == "triangle":
        size = rng.randint(28, 38)
        p1 = f"{cx},{cy - size}"
        p2 = f"{cx - size},{cy + size}"
        p3 = f"{cx + size},{cy + size}"
        return (
            f'<polygon points="{p1} {p2} {p3}" '
            f'fill="{fill if filled else "none"}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
    if shape == "line":
        dx = rng.randint(34, 48)
        dy = rng.randint(18, 28)
        x1 = cx - dx
        y1 = cy + dy
        x2 = cx + dx
        y2 = cy - dy
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw + 1}"/>'
    raise RuntimeError(f"unsupported shape: {shape}")


def _build_single_shape(spec: Spec, rng: random.Random) -> Sample:
    width = 180
    height = 120
    bg, accent, text = PALETTES[spec.palette_family]
    filled = spec.style == "filled"
    cx, cy = _layout_center(spec.layout, width, height)
    shape = spec.shape or "rect"
    body = ""
    if spec.palette_family != "mono":
        body += f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}" opacity="0.24"/>'
    body += _shape_body(shape, cx, cy, accent, text if spec.palette_family == "bold" else accent, filled=filled, rng=rng)
    svg = _svg_wrap(width, height, body)
    return Sample(spec.id, spec.stage, shape, spec.tags, svg)


def _build_pair(spec: Spec, rng: random.Random) -> Sample:
    width = 220
    height = 120
    stroke = "black"
    fill = "none"
    combo = spec.combo
    if combo == "rect-circle":
        left = _shape_body("rect", 62, 60, fill, stroke, filled=False, rng=rng)
        right = _shape_body("circle", 158, 60, fill, stroke, filled=False, rng=rng)
    elif combo == "circle-line":
        left = _shape_body("circle", 62, 60, fill, stroke, filled=False, rng=rng)
        right = _shape_body("line", 158, 60, fill, stroke, filled=False, rng=rng)
    elif combo == "rect-triangle":
        left = _shape_body("rect", 62, 60, fill, stroke, filled=False, rng=rng)
        right = _shape_body("triangle", 158, 60, fill, stroke, filled=False, rng=rng)
    elif combo == "ellipse-line":
        left = _shape_body("ellipse", 62, 60, fill, stroke, filled=False, rng=rng)
        right = _shape_body("line", 158, 60, fill, stroke, filled=False, rng=rng)
    else:
        raise RuntimeError(f"unsupported combo: {combo}")
    body = left + right
    svg = _svg_wrap(width, height, body)
    return Sample(spec.id, spec.stage, "pair", spec.tags, svg)


def _card_x(layout: str, width: int, card_w: int) -> int:
    if layout == "left":
        return 18
    if layout == "right":
        return width - card_w - 18
    return (width - card_w) // 2


def _build_card(spec: Spec, rng: random.Random) -> Sample:
    width = 320
    height = 180
    card_w = 208
    card_h = 124
    x = _card_x(spec.layout, width, card_w)
    y = 28
    bg, accent, text = PALETTES[spec.palette_family]
    title = f"{_choice(rng, TITLE_WORDS)} {_choice(rng, TITLE_WORDS)}"
    bullets = [f"- {_choice(rng, BULLET_WORDS)} {_choice(rng, TITLE_WORDS)}" for _ in range(max(3, spec.count))]
    body = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="12" fill="{bg}" stroke="{accent}" stroke-width="2"/>',
        f'<rect x="{x + 14}" y="{y + 14}" width="42" height="6" rx="3" fill="{accent}"/>',
        f'<text x="{x + 14}" y="{y + 40}" font-size="16" fill="{text}">{title}</text>',
    ]
    for idx, line in enumerate(bullets[:3]):
        ty = y + 66 + idx * 18
        body.append(f'<text x="{x + 14}" y="{ty}" font-size="13" fill="{text}">{line}</text>')
    svg = _svg_wrap(width, height, "".join(body))
    return Sample(spec.id, spec.stage, "card", spec.tags, svg)


def _build_chart(spec: Spec, rng: random.Random) -> Sample:
    width = 240
    height = 150
    bg, accent, text = PALETTES[spec.palette_family]
    title = f"{_choice(rng, TITLE_WORDS)} {_choice(rng, TITLE_WORDS)}"
    body = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']
    if spec.shape == "bar-chart":
        values = [rng.randint(18, 64), rng.randint(26, 82), rng.randint(20, 74)]
        colors = [accent, text, bg]
        body.append(f'<text x="18" y="22" font-size="14" fill="{text}">{title}</text>')
        body.append(f'<line x1="18" y1="118" x2="220" y2="118" stroke="{text}" stroke-width="1"/>')
        body.append(f'<line x1="18" y1="32" x2="18" y2="118" stroke="{text}" stroke-width="1"/>')
        for idx, value in enumerate(values):
            bar_w = 34
            x = 42 + idx * 52
            y = 118 - value
            color = colors[idx % len(colors)]
            body.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{value}" fill="{color}" stroke="{text}" stroke-width="1"/>')
            body.append(f'<text x="{x + 10}" y="134" font-size="10" fill="{text}">{chr(65 + idx)}</text>')
    else:
        points: list[str] = []
        body.append(f'<text x="18" y="22" font-size="14" fill="{text}">{title}</text>')
        body.append(f'<line x1="18" y1="118" x2="220" y2="118" stroke="{text}" stroke-width="1"/>')
        body.append(f'<line x1="18" y1="32" x2="18" y2="118" stroke="{text}" stroke-width="1"/>')
        for idx in range(5):
            x = 28 + idx * 42
            y = rng.randint(44, 102)
            points.append(f"{x},{y}")
            body.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{accent}" stroke="{text}" stroke-width="1"/>')
        body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{accent}" stroke-width="3"/>')
    svg = _svg_wrap(width, height, "".join(body))
    return Sample(spec.id, spec.stage, spec.shape.replace("-", "_"), spec.tags, svg)


def _build_path(spec: Spec, rng: random.Random) -> Sample:
    width = 220
    height = 120
    bg, accent, text = PALETTES[spec.palette_family]
    x_shift = {"left": -18, "center": 0, "right": 18}.get(spec.layout, 0)
    x0 = 24 + x_shift
    y0 = 88
    cx = 108 + x_shift
    cy = rng.randint(18, 44)
    x1 = 194 + x_shift
    y1 = 72
    x0 = max(12, min(width - 48, x0))
    cx = max(64, min(width - 64, cx))
    x1 = max(96, min(width - 12, x1))
    body = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<path d="M{x0} {y0} Q{cx} {cy} {x1} {y1}" fill="none" stroke="{accent}" stroke-width="3"/>',
        f'<circle cx="{x0}" cy="{y0}" r="4" fill="{text}" stroke="{text}" stroke-width="1"/>',
        f'<circle cx="{x1}" cy="{y1}" r="4" fill="{text}" stroke="{text}" stroke-width="1"/>',
    ]
    if spec.palette_family != "mono":
        body.insert(1, f'<rect x="14" y="16" width="{width - 28}" height="{height - 32}" rx="10" fill="{bg}" opacity="0.18"/>')
    svg = _svg_wrap(width, height, "".join(body))
    return Sample(spec.id, spec.stage, "path", spec.tags, svg)


def _ordered_tags(*, component: str = "", shape: str = "", combo: str = "", composition: str = "", palette: str = "",
                  style: str = "", layout: str = "", text: str = "", count: int = 0) -> tuple[str, ...]:
    groups: dict[str, list[str]] = {key: [] for key in TAG_GROUP_ORDER}
    if component:
        groups["component"].append(f"[{component}]")
    if shape:
        groups["shape"].append(f"[{shape}]")
    if combo:
        groups["combo"].append(f"[combo:{combo}]")
    if composition:
        groups["composition"].append(f"[composition:{composition}]")
    if palette:
        groups["palette"].append(f"[palette:{palette}]")
    if style:
        groups["style"].append(f"[style:{style}]")
    if layout:
        groups["layout"].append(f"[layout:{layout}]")
    if text:
        groups["text"].append(f"[text:{text}]")
    if count > 0:
        groups["count"].append(f"[count:{count}]")
    ordered: list[str] = []
    for key in TAG_GROUP_ORDER:
        ordered.extend(groups[key])
    return tuple(ordered)


def _build_specs(samples_train: int, samples_holdout: int) -> list[Spec]:
    specs: list[Spec] = []

    def add(**kwargs: Any) -> None:
        specs.append(
            Spec(
                samples_train=int(samples_train),
                samples_holdout=int(samples_holdout),
                **kwargs,
            )
        )

    for shape in ("circle", "rect", "ellipse", "triangle", "line"):
        add(
            id=f"basics.stage1.{shape}.v1",
            stage="stage1_shapes",
            generator="single_shape",
            tags=_ordered_tags(shape=shape, composition="single", style="outline", text="none"),
            description=f"single {shape} primitive",
            shape=shape,
            style="outline",
            text_mode="none",
        )

    for combo in ("rect-circle", "circle-line", "rect-triangle", "ellipse-line"):
        primary = combo.split("-", 1)[0]
        add(
            id=f"basics.stage2.{combo.replace('-', '_')}.v1",
            stage="stage2_pairs",
            generator="pair",
            tags=_ordered_tags(shape=primary, combo=combo, composition="pair", layout="row", text="none", count=2),
            description=f"pair composition {combo}",
            shape=primary,
            combo=combo,
            layout="row",
            text_mode="none",
            count=2,
        )

    for shape in ("circle", "rect", "triangle"):
        for palette in ("mono", "cool", "warm", "bold"):
            add(
                id=f"basics.stage3.{shape}.{palette}.v1",
                stage="stage3_color",
                generator="single_shape",
                tags=_ordered_tags(shape=shape, composition="single", palette=palette, style="filled", text="none"),
                description=f"{shape} with {palette} palette",
                shape=shape,
                palette_family=palette,
                style="filled",
                text_mode="none",
            )

    for shape in ("circle", "rect"):
        for palette in ("cool", "warm"):
            for layout in ("left", "center", "right", "top", "bottom"):
                add(
                    id=f"basics.stage4.{shape}.{palette}.{layout}.v1",
                    stage="stage4_layout",
                    generator="single_shape",
                    tags=_ordered_tags(shape=shape, composition="single", palette=palette, style="filled", layout=layout, text="none"),
                    description=f"{shape} with {palette} palette at {layout}",
                    shape=shape,
                    palette_family=palette,
                    style="filled",
                    layout=layout,
                    text_mode="none",
                )

    for palette in ("mono", "cool", "warm", "bold"):
        for layout in ("left", "center", "right"):
            add(
                id=f"basics.stage5.card.{palette}.{layout}.v1",
                stage="stage5_cards",
                generator="card",
                tags=_ordered_tags(component="card", shape="rect", composition="card", palette=palette, style="filled", layout=layout, text="bullets", count=3),
                description=f"card with three bullet lines in {palette} palette at {layout}",
                shape="rect",
                palette_family=palette,
                style="filled",
                layout=layout,
                text_mode="bullets",
                count=3,
            )

    for shape in ("bar-chart", "line-chart"):
        for palette in ("cool", "warm", "bold"):
            add(
                id=f"basics.stage6.{shape.replace('-', '_')}.{palette}.v1",
                stage="stage6_charts",
                generator="chart",
                tags=_ordered_tags(shape=shape, composition="chart", palette=palette, style="filled", layout="center", text="title", count=3),
                description=f"{shape} with {palette} palette",
                shape=shape,
                palette_family=palette,
                style="filled",
                layout="center",
                text_mode="title",
                count=3,
            )

    for palette in ("mono", "cool", "warm", "bold"):
        for layout in ("left", "center", "right"):
            add(
                id=f"basics.stage7.path.{palette}.{layout}.v1",
                stage="stage7_paths",
                generator="path",
                tags=_ordered_tags(shape="path", composition="path", palette=palette, style="outline", layout=layout, text="none"),
                description=f"path composition in {palette} palette at {layout}",
                shape="path",
                palette_family=palette,
                style="outline",
                layout=layout,
                text_mode="none",
            )

    return specs


def _build_sample(spec: Spec, rng: random.Random) -> Sample:
    if spec.generator == "single_shape":
        return _build_single_shape(spec, rng)
    if spec.generator == "pair":
        return _build_pair(spec, rng)
    if spec.generator == "card":
        return _build_card(spec, rng)
    if spec.generator == "chart":
        return _build_chart(spec, rng)
    if spec.generator == "path":
        return _build_path(spec, rng)
    raise RuntimeError(f"unknown generator: {spec.generator}")


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _coverage_payload(prefix: str, stage: str, specs: list[Spec], train_rows: list[Sample], holdout_rows: list[Sample]) -> dict[str, Any]:
    spec_map = {spec.id: spec for spec in specs}
    train_counts: Counter[str] = Counter(sample.spec_id for sample in train_rows)
    holdout_counts: Counter[str] = Counter(sample.spec_id for sample in holdout_rows)
    failures: list[str] = []
    for spec in specs:
        got_train = int(train_counts.get(spec.id, 0))
        got_hold = int(holdout_counts.get(spec.id, 0))
        if got_train != int(spec.samples_train):
            failures.append(f"{spec.id}: train {got_train}/{spec.samples_train}")
        if got_hold != int(spec.samples_holdout):
            failures.append(f"{spec.id}: holdout {got_hold}/{spec.samples_holdout}")
    return {
        "format": "v7-svg-basics-coverage.v1",
        "dataset_prefix": prefix,
        "stage": stage,
        "spec_count": len(specs),
        "spec_counts_train": dict(sorted(train_counts.items())),
        "spec_counts_holdout": dict(sorted(holdout_counts.items())),
        "sample_type_counts_train": dict(sorted(Counter(sample.sample_type for sample in train_rows).items())),
        "sample_type_counts_holdout": dict(sorted(Counter(sample.sample_type for sample in holdout_rows).items())),
        "gate": {
            "passed": len(failures) == 0,
            "failures": failures,
        },
        "specs": [
            {
                "id": spec.id,
                "stage": spec.stage,
                "generator": spec.generator,
                "tags": list(spec.tags),
                "samples_train": int(spec.samples_train),
                "samples_holdout": int(spec.samples_holdout),
                "description": spec.description,
            }
            for spec in specs
            if spec.id in spec_map
        ],
    }


def _rows_for_stages(stage_names: set[str], stage_rows: dict[str, list[Sample]]) -> list[Sample]:
    out: list[Sample] = []
    for stage, rows in stage_rows.items():
        if stage in stage_names:
            out.extend(rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a composition-first SVG DSL curriculum")
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="svg_basics_curriculum", help="Output file prefix")
    ap.add_argument("--samples-train", type=int, default=96, help="Per-spec train rows")
    ap.add_argument("--samples-holdout", type=int, default=12, help="Per-spec holdout rows")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    if int(args.samples_train) < 1:
        raise SystemExit("--samples-train must be >= 1")
    if int(args.samples_holdout) < 1:
        raise SystemExit("--samples-holdout must be >= 1")

    rng = random.Random(int(args.seed))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_specs(int(args.samples_train), int(args.samples_holdout))
    stages = list(dict.fromkeys(spec.stage for spec in specs))
    generator_registry = {spec.generator for spec in specs}

    stage_train: dict[str, list[Sample]] = {stage: [] for stage in stages}
    stage_holdout: dict[str, list[Sample]] = {stage: [] for stage in stages}

    for spec in specs:
        for _ in range(int(spec.samples_train)):
            sample = _build_sample(spec, rng)
            _assert_ascii(sample.instruction_line(), "instruction line")
            _assert_ascii(sample.svg, "svg")
            _validate_svg(sample.svg)
            stage_train[spec.stage].append(sample)
        for _ in range(int(spec.samples_holdout)):
            sample = _build_sample(spec, rng)
            _assert_ascii(sample.instruction_line(), "instruction holdout line")
            _assert_ascii(sample.svg, "holdout svg")
            _validate_svg(sample.svg)
            stage_holdout[spec.stage].append(sample)

    reserved_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for spec in specs:
        for token in spec.tags:
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            reserved_tokens.append(token)

    all_train: list[Sample] = []
    all_holdout: list[Sample] = []
    for stage in stages:
        rng.shuffle(stage_train[stage])
        rng.shuffle(stage_holdout[stage])
        all_train.extend(stage_train[stage])
        all_holdout.extend(stage_holdout[stage])

        stage_prefix = f"{args.prefix}_{stage}"
        instruction_train = [sample.instruction_line() for sample in stage_train[stage]]
        instruction_holdout = [sample.instruction_line() for sample in stage_holdout[stage]]
        instruction_all = [*instruction_train, *instruction_holdout]
        svg_train = [sample.svg for sample in stage_train[stage]]
        svg_holdout = [sample.svg for sample in stage_holdout[stage]]
        svg_all = [*svg_train, *svg_holdout]

        _write_lines(out_dir / f"{stage_prefix}_instruction_train.txt", instruction_train)
        _write_lines(out_dir / f"{stage_prefix}_instruction_holdout.txt", instruction_holdout)
        _write_lines(out_dir / f"{stage_prefix}_instruction_all.txt", instruction_all)
        _write_lines(out_dir / f"{stage_prefix}_svg_train.txt", svg_train)
        _write_lines(out_dir / f"{stage_prefix}_svg_holdout.txt", svg_holdout)
        _write_lines(out_dir / f"{stage_prefix}_svg_all.txt", svg_all)

        coverage = _coverage_payload(
            stage_prefix,
            stage,
            [spec for spec in specs if spec.stage == stage],
            stage_train[stage],
            stage_holdout[stage],
        )
        (out_dir / f"{stage_prefix}_coverage_manifest.json").write_text(
            json.dumps(coverage, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    combined_instruction_train = [sample.instruction_line() for sample in all_train]
    combined_instruction_holdout = [sample.instruction_line() for sample in all_holdout]
    combined_instruction_all = [*combined_instruction_train, *combined_instruction_holdout]
    combined_svg_train = [sample.svg for sample in all_train]
    combined_svg_holdout = [sample.svg for sample in all_holdout]
    combined_svg_all = [*combined_svg_train, *combined_svg_holdout]

    _write_lines(out_dir / f"{args.prefix}_instruction_train.txt", combined_instruction_train)
    _write_lines(out_dir / f"{args.prefix}_instruction_holdout.txt", combined_instruction_holdout)
    _write_lines(out_dir / f"{args.prefix}_instruction_all.txt", combined_instruction_all)
    _write_lines(out_dir / f"{args.prefix}_svg_train.txt", combined_svg_train)
    _write_lines(out_dir / f"{args.prefix}_svg_holdout.txt", combined_svg_holdout)
    _write_lines(out_dir / f"{args.prefix}_svg_all.txt", combined_svg_all)
    _write_lines(out_dir / f"{args.prefix}_reserved_control_tokens.txt", reserved_tokens)

    canary_prompts = [
        "[circle][composition:single][style:outline][text:none]<svg",
        "[rect][composition:single][palette:cool][style:filled][layout:left][text:none]<svg",
        "[card][rect][composition:card][palette:warm][style:filled][layout:center][text:bullets][count:3]<svg",
        "[bar-chart][composition:chart][palette:bold][style:filled][layout:center][text:title][count:3]<svg",
        "[path][composition:path][palette:cool][style:outline][layout:right][text:none]<svg",
    ]
    _write_lines(out_dir / f"{args.prefix}_canary_prompts.txt", canary_prompts)

    midtrain_stage_names = {"stage1_shapes", "stage2_pairs", "stage3_color", "stage4_layout"}
    sft_stage_names = {"stage5_cards", "stage6_charts", "stage7_paths"}
    midtrain_rows = _rows_for_stages(midtrain_stage_names, stage_train)
    sft_rows = _rows_for_stages(sft_stage_names, stage_train)
    midtrain_holdout_rows = _rows_for_stages(midtrain_stage_names, stage_holdout)
    sft_holdout_rows = _rows_for_stages(sft_stage_names, stage_holdout)

    midtrain_instruction_train = [sample.instruction_line() for sample in midtrain_rows]
    midtrain_instruction_holdout = [sample.instruction_line() for sample in midtrain_holdout_rows]
    sft_instruction_train = [sample.instruction_line() for sample in sft_rows]
    sft_instruction_holdout = [sample.instruction_line() for sample in sft_holdout_rows]
    tokenizer_corpus_rows = [
        *combined_svg_train,
        *combined_instruction_train,
        *canary_prompts,
    ]

    _write_lines(out_dir / f"{args.prefix}_midtrain_instruction_train.txt", midtrain_instruction_train)
    _write_lines(out_dir / f"{args.prefix}_midtrain_instruction_holdout.txt", midtrain_instruction_holdout)
    _write_lines(out_dir / f"{args.prefix}_sft_instruction_train.txt", sft_instruction_train)
    _write_lines(out_dir / f"{args.prefix}_sft_instruction_holdout.txt", sft_instruction_holdout)
    _write_lines(out_dir / f"{args.prefix}_tokenizer_corpus.txt", tokenizer_corpus_rows)

    spec_catalog = {
        "format": "svg-basics-spec-catalog.v1",
        "version": "1.0.0",
        "notes": [
            "Composition-first SVG DSL curriculum.",
            "Designed as a middle ground between spec02 synthetic control and spec03 asset-heavy structure.",
        ],
        "coverage_rules": {
            "min_train_per_spec": int(args.samples_train),
            "min_holdout_per_spec": int(args.samples_holdout),
        },
        "specs": [
            {
                "id": spec.id,
                "stage": spec.stage,
                "group": spec.group,
                "tags": list(spec.tags),
                "generator": spec.generator,
                "shape": spec.shape,
                "combo": spec.combo,
                "palette_family": spec.palette_family,
                "style": spec.style,
                "layout": spec.layout,
                "text_mode": spec.text_mode,
                "count": int(spec.count),
                "samples_train": int(spec.samples_train),
                "samples_holdout": int(spec.samples_holdout),
                "description": spec.description,
                "status": "ready",
            }
            for spec in specs
        ],
    }
    spec_catalog_path = out_dir / f"{args.prefix}_spec_catalog.json"
    spec_catalog_path.write_text(json.dumps(spec_catalog, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    combined_coverage = _coverage_payload(args.prefix, "all", specs, all_train, all_holdout)
    combined_coverage["artifacts"] = {
        "instruction_train": str(out_dir / f"{args.prefix}_instruction_train.txt"),
        "instruction_holdout": str(out_dir / f"{args.prefix}_instruction_holdout.txt"),
        "svg_train": str(out_dir / f"{args.prefix}_svg_train.txt"),
        "svg_holdout": str(out_dir / f"{args.prefix}_svg_holdout.txt"),
        "midtrain_instruction_train": str(out_dir / f"{args.prefix}_midtrain_instruction_train.txt"),
        "midtrain_instruction_holdout": str(out_dir / f"{args.prefix}_midtrain_instruction_holdout.txt"),
        "sft_instruction_train": str(out_dir / f"{args.prefix}_sft_instruction_train.txt"),
        "sft_instruction_holdout": str(out_dir / f"{args.prefix}_sft_instruction_holdout.txt"),
        "tokenizer_corpus": str(out_dir / f"{args.prefix}_tokenizer_corpus.txt"),
        "reserved_control_tokens": str(out_dir / f"{args.prefix}_reserved_control_tokens.txt"),
        "spec_catalog": str(spec_catalog_path),
    }
    (out_dir / f"{args.prefix}_coverage_manifest.json").write_text(
        json.dumps(combined_coverage, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "format": "svg-basics-curriculum-manifest.v1",
        "prefix": args.prefix,
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "samples_train_per_spec": int(args.samples_train),
        "samples_holdout_per_spec": int(args.samples_holdout),
        "stage_order": stages,
        "stage_counts": {
            stage: {
                "specs": len([spec for spec in specs if spec.stage == stage]),
                "train_rows": len(stage_train[stage]),
                "holdout_rows": len(stage_holdout[stage]),
            }
            for stage in stages
        },
        "total_specs": len(specs),
        "total_train_rows": len(all_train),
        "total_holdout_rows": len(all_holdout),
        "generator_kinds": sorted(generator_registry),
        "reserved_control_tokens": len(reserved_tokens),
        "artifacts": {
            "instruction_train": str(out_dir / f"{args.prefix}_instruction_train.txt"),
            "instruction_holdout": str(out_dir / f"{args.prefix}_instruction_holdout.txt"),
            "svg_train": str(out_dir / f"{args.prefix}_svg_train.txt"),
            "svg_holdout": str(out_dir / f"{args.prefix}_svg_holdout.txt"),
            "midtrain_instruction_train": str(out_dir / f"{args.prefix}_midtrain_instruction_train.txt"),
            "midtrain_instruction_holdout": str(out_dir / f"{args.prefix}_midtrain_instruction_holdout.txt"),
            "sft_instruction_train": str(out_dir / f"{args.prefix}_sft_instruction_train.txt"),
            "sft_instruction_holdout": str(out_dir / f"{args.prefix}_sft_instruction_holdout.txt"),
            "tokenizer_corpus": str(out_dir / f"{args.prefix}_tokenizer_corpus.txt"),
            "reserved_control_tokens": str(out_dir / f"{args.prefix}_reserved_control_tokens.txt"),
            "spec_catalog": str(spec_catalog_path),
            "coverage_manifest": str(out_dir / f"{args.prefix}_coverage_manifest.json"),
            "canary_prompts": str(out_dir / f"{args.prefix}_canary_prompts.txt"),
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
                "total_specs": len(specs),
                "total_train_rows": len(all_train),
                "total_holdout_rows": len(all_holdout),
                "stages": stages,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
