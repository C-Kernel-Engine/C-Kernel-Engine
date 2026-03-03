#!/usr/bin/env python3
"""
Generate synthetic ASCII-only instruction->SVG datasets for v7 training.

Outputs:
  - instruction dataset: <task>...</task><svg ...>...</svg><eos>
  - svg-only dataset:    <svg ...>...</svg>
  - train/holdout splits
  - optional JSONL
  - manifest with type counts and paths

Includes primitive geometry, charts/tables, and style-heavy templates
(e.g., gradient cards and infographic layouts).
"""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


PALETTE = [
    "red",
    "orange",
    "gold",
    "green",
    "teal",
    "blue",
    "navy",
    "purple",
    "black",
    "gray",
    "brown",
    "pink",
]

TEXT_WORDS = [
    "SVG",
    "DEMO",
    "DATA",
    "CHART",
    "TABLE",
    "ALPHA",
    "BETA",
    "GAMMA",
    "DELTA",
    "KERNEL",
]

LABELS = ["A", "B", "C", "D", "E", "F", "G"]
THEME_PAIRS = [
    ("#0B1020", "#1E3A8A", "white"),
    ("#0F172A", "#2563EB", "white"),
    ("#111827", "#0EA5E9", "white"),
    ("#1F2937", "#0F766E", "white"),
    ("#172554", "#1D4ED8", "white"),
]

CURRENT_FILL_MODE = "mixed"
_SPEC_CTX: dict = {}

PALETTE_FAMILIES: dict[str, list[tuple[str, str, str]]] = {
    "neutral": [
        ("#374151", "#6B7280", "#E5E7EB"),
        ("#4B5563", "#9CA3AF", "#F3F4F6"),
        ("#334155", "#94A3B8", "#E2E8F0"),
        ("#52525B", "#A1A1AA", "#F4F4F5"),
        ("#3F3F46", "#A8A29E", "#E7E5E4"),
    ],
    "bold": [
        ("#1A1A2E", "#E91E63", "#FFFFFF"),
        ("#0F172A", "#FF5722", "#FFFFFF"),
        ("#111827", "#2563EB", "#FFFFFF"),
        ("#0B1020", "#22C55E", "#FFFFFF"),
        ("#1F2937", "#F59E0B", "#FFFFFF"),
    ],
    "warm": [
        ("#7C2D12", "#EA580C", "#FDE68A"),
        ("#9A3412", "#F97316", "#FED7AA"),
        ("#78350F", "#D97706", "#FDE68A"),
        ("#92400E", "#F59E0B", "#FDE68A"),
        ("#7F1D1D", "#EF4444", "#FECACA"),
    ],
    "cool": [
        ("#0C4A6E", "#0284C7", "#E0F2FE"),
        ("#164E63", "#06B6D4", "#CFFAFE"),
        ("#1E3A8A", "#3B82F6", "#DBEAFE"),
        ("#1D4ED8", "#60A5FA", "#EFF6FF"),
        ("#155E75", "#14B8A6", "#CCFBF1"),
    ],
    "pastel": [
        ("#9D174D", "#F9A8D4", "#FCE7F3"),
        ("#6D28D9", "#C4B5FD", "#EDE9FE"),
        ("#0369A1", "#93C5FD", "#DBEAFE"),
        ("#166534", "#86EFAC", "#DCFCE7"),
        ("#9A3412", "#FDBA74", "#FFEDD5"),
    ],
    "dark": [
        ("#0B1020", "#1E293B", "#F8FAFC"),
        ("#111827", "#1F2937", "#F9FAFB"),
        ("#0F172A", "#334155", "#E2E8F0"),
        ("#0B132B", "#1C2541", "#E0E1DD"),
        ("#030712", "#1F2937", "#F3F4F6"),
    ],
}


@dataclass
class Sample:
    sample_type: str
    task: str
    svg: str
    tags: list[str] = field(default_factory=list)

    def instruction_line(self) -> str:
        return f"<task>{self.task}</task>{self.svg}<eos>"

    def tag_line(self) -> str:
        prefix = "".join(self.tags) if self.tags else f"[{self.sample_type}]"
        return f"{prefix}{self.svg}<eos>"


class CoverageTracker:
    def __init__(self) -> None:
        self.spec_counts_total: dict[str, int] = {}
        self.spec_counts_train: dict[str, int] = {}
        self.spec_counts_holdout: dict[str, int] = {}
        self.pair_counts: Counter[tuple[str, str]] = Counter()

    def record(self, spec_id: str, tags: list[str], *, split: str) -> None:
        self.spec_counts_total[spec_id] = int(self.spec_counts_total.get(spec_id, 0) + 1)
        if split == "train":
            self.spec_counts_train[spec_id] = int(self.spec_counts_train.get(spec_id, 0) + 1)
        elif split == "holdout":
            self.spec_counts_holdout[spec_id] = int(self.spec_counts_holdout.get(spec_id, 0) + 1)
        uniq = list(dict.fromkeys(tags))
        for i, t1 in enumerate(uniq):
            for t2 in uniq[i + 1:]:
                key = tuple(sorted((str(t1), str(t2))))
                self.pair_counts[key] += 1

    def gate(self, specs: list[dict], min_pair_count: int = 30) -> dict:
        failures: list[str] = []
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            if str(spec.get("status", "")).lower() == "stub":
                continue
            spec_id = str(spec.get("id", "")).strip()
            if not spec_id:
                continue
            req_train = int(spec.get("samples_train", 0) or 0)
            req_holdout = int(spec.get("samples_holdout", 0) or 0)
            got_train = int(self.spec_counts_train.get(spec_id, 0))
            got_holdout = int(self.spec_counts_holdout.get(spec_id, 0))
            if got_train < req_train:
                failures.append(
                    f"spec {spec_id}: train {got_train}/{req_train}"
                )
            if got_holdout < req_holdout:
                failures.append(
                    f"spec {spec_id}: holdout {got_holdout}/{req_holdout}"
                )
        low_pairs = [(k, v) for k, v in self.pair_counts.items() if int(v) < int(min_pair_count)]
        if low_pairs:
            failures.append(
                f"tag_pair_floor: {len(low_pairs)} pair(s) below {int(min_pair_count)}"
            )
        return {
            "passed": len(failures) == 0,
            "failures": failures,
            "min_pair_count": int(min_pair_count),
        }


def _assert_ascii(text: str, what: str) -> None:
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"non-ASCII content in {what}: {exc}") from exc


def _validate_svg(svg: str) -> None:
    root = ET.fromstring(svg)
    tag = root.tag.split("}", 1)[-1] if "}" in root.tag else root.tag
    if tag != "svg":
        raise RuntimeError(f"invalid root tag: {tag}")


def _svg_wrap(width: int, height: int, body: str) -> str:
    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'


def _choice(rng: random.Random, values: list[str]) -> str:
    return values[rng.randrange(len(values))]


def _resolve_fill(rng: random.Random, fill_color: str) -> str:
    mode = str(CURRENT_FILL_MODE).strip().lower()
    if mode == "filled":
        return fill_color
    if mode == "outline":
        return "none"
    # mixed
    return fill_color if rng.random() >= 0.35 else "none"


def _spec_palette_variant(rng: random.Random) -> tuple[str, str, str] | None:
    family = str(_SPEC_CTX.get("palette_family", "") or "").strip().lower()
    if family and family in PALETTE_FAMILIES:
        return rng.choice(PALETTE_FAMILIES[family])
    return None


def _spec_colors(rng: random.Random) -> tuple[str, str, str]:
    variant = _spec_palette_variant(rng)
    if variant is None:
        fill_color = _choice(rng, PALETTE)
        stroke = _choice(rng, [c for c in PALETTE if c != fill_color])
        fill = _resolve_fill(rng, fill_color)
        return fill_color, stroke, fill

    dark, mid, light = variant
    style = str(_SPEC_CTX.get("style", "mixed") or "mixed").strip().lower()
    if style in {"outlined", "outline"}:
        fill = "none"
    elif style == "filled":
        fill = mid
    elif style == "gradient":
        # Keep generic generators self-contained; dedicated gradient generators
        # build explicit defs/ids.
        fill = mid
    else:
        fill = _resolve_fill(rng, mid)
    return mid, dark, fill


def _line_sample(rng: random.Random) -> Sample:
    w = rng.choice([120, 140, 160, 180, 200])
    h = rng.choice([80, 100, 120, 140])
    x1 = rng.randint(6, w - 30)
    y1 = rng.randint(6, h - 30)
    x2 = rng.randint(20, w - 6)
    y2 = rng.randint(20, h - 6)
    variant = _spec_palette_variant(rng)
    color = variant[1] if variant else _choice(rng, PALETTE)
    sw = rng.randint(1, 5)
    body = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"/>'
    task_templates = [
        "draw a {color} line from ({x1},{y1}) to ({x2},{y2})",
        "create an svg line in {color} with endpoints {x1},{y1} and {x2},{y2}",
        "make a single straight {color} segment from {x1},{y1} to {x2},{y2}",
    ]
    task = _choice(rng, task_templates).format(color=color, x1=x1, y1=y1, x2=x2, y2=y2)
    return Sample("line", task, _svg_wrap(w, h, body))


def _triangle_sample(rng: random.Random) -> Sample:
    w = rng.choice([140, 160, 180, 200])
    h = rng.choice([100, 120, 140])
    p1 = (rng.randint(10, w // 2), rng.randint(10, h - 10))
    p2 = (rng.randint(w // 2, w - 10), rng.randint(10, h - 10))
    p3 = (rng.randint(10, w - 10), rng.randint(10, h - 10))
    fill_color, stroke, fill = _spec_colors(rng)
    body = (
        f'<polygon points="{p1[0]},{p1[1]} {p2[0]},{p2[1]} {p3[0]},{p3[1]}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )
    task_templates = [
        "draw a filled triangle with points ({x1},{y1}) ({x2},{y2}) ({x3},{y3}) using {fill}",
        "create a {fill} triangle outlined in {stroke}",
        "make one triangle polygon with vertices {x1},{y1} {x2},{y2} {x3},{y3}",
    ]
    task = _choice(rng, task_templates).format(
        x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1], x3=p3[0], y3=p3[1], fill=fill_color, stroke=stroke
    )
    return Sample("triangle", task, _svg_wrap(w, h, body))


def _ellipse_sample(rng: random.Random) -> Sample:
    w = rng.choice([140, 160, 180, 220])
    h = rng.choice([90, 110, 130, 150])
    cx = rng.randint(26, w - 26)
    cy = rng.randint(24, h - 24)
    rx = rng.randint(16, max(18, w // 4))
    ry = rng.randint(10, max(12, h // 4))
    fill_color, stroke, fill = _spec_colors(rng)
    body = (
        f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )
    task_templates = [
        "draw an ellipse centered at ({cx},{cy}) with radii {rx},{ry}",
        "create an svg ellipse with stroke {stroke}",
        "make one ellipse shape with rx={rx} and ry={ry}",
    ]
    task = _choice(rng, task_templates).format(cx=cx, cy=cy, rx=rx, ry=ry, stroke=stroke)
    return Sample("ellipse", task, _svg_wrap(w, h, body))


def _circle_sample(rng: random.Random) -> Sample:
    w = rng.choice([120, 140, 160, 180, 200])
    h = rng.choice([120, 140, 160, 180, 200])
    r = rng.randint(16, min(w, h) // 3)
    cx = rng.randint(r + 8, w - r - 8)
    cy = rng.randint(r + 8, h - r - 8)
    fill_color, stroke, fill = _spec_colors(rng)
    body = (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )
    task_templates = [
        "draw a circle centered at ({cx},{cy}) with radius {r}",
        "create one circular shape with a clean stroke",
        "make an svg circle icon with balanced padding",
    ]
    task = _choice(rng, task_templates).format(cx=cx, cy=cy, r=r)
    return Sample("circle", task, _svg_wrap(w, h, body))


def _rect_sample(rng: random.Random) -> Sample:
    w = rng.choice([140, 160, 180, 200, 220])
    h = rng.choice([90, 110, 130, 150])
    x = rng.randint(8, max(10, w // 4))
    y = rng.randint(8, max(10, h // 4))
    rw = rng.randint(w // 3, w - x - 8)
    rh = rng.randint(h // 3, h - y - 8)
    rx = rng.randint(0, 12)
    fill_color, stroke, fill = _spec_colors(rng)
    body = (
        f'<rect x="{x}" y="{y}" width="{rw}" height="{rh}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )
    task_templates = [
        "draw a rectangle at ({x},{y}) sized {rw}x{rh}",
        "create a rounded rectangle card with simple styling",
        "make one rectangular panel with clean stroke and fill",
    ]
    task = _choice(rng, task_templates).format(x=x, y=y, rw=rw, rh=rh)
    return Sample("rect", task, _svg_wrap(w, h, body))


def _polygon_sample(rng: random.Random) -> Sample:
    w = rng.choice([150, 180, 210, 240])
    h = rng.choice([100, 120, 140, 160])
    n = rng.randint(5, 8)
    pts: list[tuple[int, int]] = []
    for i in range(n):
        x = rng.randint(12, w - 12)
        y = rng.randint(12, h - 12)
        pts.append((x, y))
    fill_color, stroke, fill = _spec_colors(rng)
    point_s = " ".join(f"{x},{y}" for x, y in pts)
    body = f'<polygon points="{point_s}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    task_templates = [
        "draw a polygon with {n} points",
        "create an svg polygon shape with {n} vertices",
        "make a multi-point polygon outlined in {stroke}",
    ]
    task = _choice(rng, task_templates).format(n=n, stroke=stroke)
    return Sample("polygon", task, _svg_wrap(w, h, body))


def _arrow_sample(rng: random.Random) -> Sample:
    w = rng.choice([160, 180, 200, 220])
    h = rng.choice([90, 110, 130])
    x1 = rng.randint(10, 30)
    y1 = rng.randint(20, h - 20)
    x2 = rng.randint(w - 50, w - 12)
    y2 = rng.randint(20, h - 20)
    variant = _spec_palette_variant(rng)
    stroke = variant[1] if variant else _choice(rng, PALETTE)
    sw = rng.randint(2, 4)
    hx = x2
    hy = y2
    head = f"{hx},{hy} {hx-12},{hy-7} {hx-12},{hy+7}"
    body = (
        f'<line x1="{x1}" y1="{y1}" x2="{x2-10}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"/>'
        f'<polygon points="{head}" fill="{stroke}" stroke="{stroke}"/>'
    )
    task_templates = [
        "draw a right-pointing arrow",
        "create a single directional arrow from left to right",
        "make an svg arrow line with one arrowhead",
    ]
    task = _choice(rng, task_templates)
    return Sample("arrow", task, _svg_wrap(w, h, body))


def _double_arrow_sample(rng: random.Random) -> Sample:
    w = rng.choice([160, 190, 220])
    h = rng.choice([90, 110, 130])
    x1 = rng.randint(18, 30)
    x2 = rng.randint(w - 30, w - 18)
    y = rng.randint(20, h - 20)
    variant = _spec_palette_variant(rng)
    stroke = variant[1] if variant else _choice(rng, PALETTE)
    sw = rng.randint(2, 4)
    head_l = f"{x1},{y} {x1+11},{y-7} {x1+11},{y+7}"
    head_r = f"{x2},{y} {x2-11},{y-7} {x2-11},{y+7}"
    body = (
        f'<line x1="{x1+8}" y1="{y}" x2="{x2-8}" y2="{y}" stroke="{stroke}" stroke-width="{sw}"/>'
        f'<polygon points="{head_l}" fill="{stroke}" stroke="{stroke}"/>'
        f'<polygon points="{head_r}" fill="{stroke}" stroke="{stroke}"/>'
    )
    task_templates = [
        "draw a double-headed arrow",
        "create a bidirectional arrow",
        "make an svg line with arrowheads on both ends",
    ]
    task = _choice(rng, task_templates)
    return Sample("double_arrow", task, _svg_wrap(w, h, body))


def _rounded_triangle_sample(rng: random.Random) -> Sample:
    w = rng.choice([150, 170, 190, 210])
    h = rng.choice([100, 120, 140])
    x1, y1 = rng.randint(18, 40), rng.randint(h // 2, h - 14)
    x2, y2 = rng.randint(w // 2 - 10, w // 2 + 10), rng.randint(12, 28)
    x3, y3 = rng.randint(w - 40, w - 18), rng.randint(h // 2, h - 14)
    fill_color, stroke, fill = _spec_colors(rng)
    d = (
        f"M {x1} {y1} "
        f"Q {x2} {y2} {x3} {y3} "
        f"Q {x2} {y2 + 18} {x1} {y1} Z"
    )
    body = f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    task_templates = [
        "draw a rounded triangle using a path",
        "create a soft-corner triangle shape",
        "make a rounded triangular path with stroke {stroke}",
    ]
    task = _choice(rng, task_templates).format(stroke=stroke)
    return Sample("rounded_triangle", task, _svg_wrap(w, h, body))


def _path_sample(rng: random.Random) -> Sample:
    w = rng.choice([160, 180, 200, 240])
    h = rng.choice([100, 120, 140])
    x0 = rng.randint(10, 24)
    y0 = rng.randint(h // 2, h - 20)
    cx = rng.randint(w // 3, w // 2)
    cy = rng.randint(12, h // 2)
    x1 = rng.randint(w - 30, w - 12)
    y1 = rng.randint(h // 2, h - 14)
    fill_color, stroke, fill = _spec_colors(rng)
    d = f"M{x0} {y0} Q{cx} {cy} {x1} {y1}"
    body = f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'
    task_templates = [
        "draw a curved svg path from left to right",
        "create a bezier-like curve path",
        "make one path curve with stroke {stroke}",
    ]
    task = _choice(rng, task_templates).format(stroke=stroke)
    return Sample("path", task, _svg_wrap(w, h, body))


def _text_sample(rng: random.Random) -> Sample:
    w = rng.choice([140, 160, 180, 220])
    h = rng.choice([80, 100, 120])
    txt = f"{_choice(rng, TEXT_WORDS)} {_choice(rng, TEXT_WORDS)}"
    x = rng.randint(8, 24)
    y = rng.randint(24, h - 10)
    fs = rng.choice([12, 14, 16, 18, 20, 22])
    variant = _spec_palette_variant(rng)
    fill = variant[1] if variant else _choice(rng, PALETTE)
    body = f'<text x="{x}" y="{y}" font-size="{fs}" fill="{fill}">{txt}</text>'
    task_templates = [
        'draw text "{txt}" in {fill}',
        'create an svg label "{txt}" at x={x}, y={y}',
        'render the phrase "{txt}" with font size {fs}',
    ]
    task = _choice(rng, task_templates).format(txt=txt, fill=fill, x=x, y=y, fs=fs)
    return Sample("text", task, _svg_wrap(w, h, body))


def _rect_circle_scene_sample(rng: random.Random) -> Sample:
    w = rng.choice([160, 180, 200, 240])
    h = rng.choice([100, 120, 140, 160])
    rx = rng.randint(8, w // 3)
    ry = rng.randint(8, h // 3)
    rw = rng.randint(w // 3, w - rx - 8)
    rh = rng.randint(h // 3, h - ry - 8)
    cx = rng.randint(20, w - 20)
    cy = rng.randint(20, h - 20)
    r = rng.randint(10, min(w, h) // 5)
    rc, stroke_rc, fill_rc = _spec_colors(rng)
    cc, stroke_cc, fill_cc = _spec_colors(rng)
    body = (
        f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" fill="{fill_rc}" stroke="{stroke_rc}"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill_cc}" stroke="{stroke_cc}"/>'
    )
    task_templates = [
        "draw a rectangle and a circle in one svg",
        "create a scene with one filled rectangle and one filled circle",
        "make a simple two-shape icon: rectangle plus circle",
    ]
    task = _choice(rng, task_templates)
    return Sample("rect_circle", task, _svg_wrap(w, h, body))


def _bar_chart_sample(rng: random.Random) -> Sample:
    w = 220
    h = 140
    values = [rng.randint(10, 100), rng.randint(10, 100), rng.randint(10, 100)]
    labels = rng.sample(LABELS, 3)
    variant = _spec_palette_variant(rng)
    if variant:
        dark, mid, light = variant
        colors = [mid, light, dark]
    else:
        colors = rng.sample(PALETTE, 3)
    max_v = max(values)
    chart_top = 16
    chart_bottom = 110
    chart_h = chart_bottom - chart_top
    bar_w = 34
    x0 = 24
    gap = 26
    body_parts = [
        f'<line x1="12" y1="{chart_bottom}" x2="{w - 10}" y2="{chart_bottom}" stroke="black"/>',
        f'<line x1="12" y1="{chart_top}" x2="12" y2="{chart_bottom}" stroke="black"/>',
    ]
    for i, (val, lab, col) in enumerate(zip(values, labels, colors)):
        bh = max(4, int(chart_h * (val / max_v)))
        x = x0 + i * (bar_w + gap)
        y = chart_bottom - bh
        bar_fill = _resolve_fill(rng, col)
        body_parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" fill="{bar_fill}" stroke="black"/>')
        body_parts.append(f'<text x="{x + 8}" y="{chart_bottom + 14}" font-size="10" fill="black">{lab}</text>')
        body_parts.append(f'<text x="{x + 6}" y="{y - 4}" font-size="10" fill="black">{val}</text>')
    task_templates = [
        "draw a bar chart comparing {a}={va}, {b}={vb}, {c}={vc}",
        "create an svg chart with three bars for {a}, {b}, and {c}",
        "make a comparison bar graph from values {va}, {vb}, and {vc}",
    ]
    task = _choice(rng, task_templates).format(
        a=labels[0], b=labels[1], c=labels[2], va=values[0], vb=values[1], vc=values[2]
    )
    return Sample("bar_chart", task, _svg_wrap(w, h, "".join(body_parts)))


def _comparison_table_sample(rng: random.Random) -> Sample:
    w = 260
    h = 140
    headers = ["ITEM", "A", "B"]
    rows = [rng.sample(["CPU", "GPU", "TPU", "RAM", "CACHE", "SSD"], 1)[0] for _ in range(3)]
    vals_a = [rng.randint(1, 99) for _ in range(3)]
    vals_b = [rng.randint(1, 99) for _ in range(3)]

    x_cols = [10, 110, 180, 250]
    y_rows = [12, 36, 60, 84, 108]
    body_parts = []
    for x in x_cols:
        body_parts.append(f'<line x1="{x}" y1="{y_rows[0]}" x2="{x}" y2="{y_rows[-1]}" stroke="black"/>')
    for y in y_rows:
        body_parts.append(f'<line x1="{x_cols[0]}" y1="{y}" x2="{x_cols[-1]}" y2="{y}" stroke="black"/>')
    body_parts.append(f'<text x="20" y="28" font-size="10" fill="black">{headers[0]}</text>')
    body_parts.append(f'<text x="132" y="28" font-size="10" fill="black">{headers[1]}</text>')
    body_parts.append(f'<text x="202" y="28" font-size="10" fill="black">{headers[2]}</text>')
    for i in range(3):
        y = 52 + i * 24
        body_parts.append(f'<text x="20" y="{y}" font-size="10" fill="black">{rows[i]}</text>')
        body_parts.append(f'<text x="132" y="{y}" font-size="10" fill="black">{vals_a[i]}</text>')
        body_parts.append(f'<text x="202" y="{y}" font-size="10" fill="black">{vals_b[i]}</text>')

    task_templates = [
        "draw a comparison table with ITEM, A, B columns and three rows",
        "create an svg table that compares three items across columns A and B",
        "make a small grid table showing three row comparisons",
    ]
    task = _choice(rng, task_templates)
    return Sample("comparison_table", task, _svg_wrap(w, h, "".join(body_parts)))


def _polyline_sample(rng: random.Random) -> Sample:
    w = rng.choice([140, 160, 180, 200])
    h = rng.choice([90, 100, 120])
    n = rng.randint(4, 6)
    points = []
    for i in range(n):
        x = int(10 + i * (w - 20) / (n - 1))
        y = rng.randint(12, h - 12)
        points.append(f"{x},{y}")
    variant = _spec_palette_variant(rng)
    color = variant[1] if variant else _choice(rng, PALETTE)
    sw = rng.randint(2, 4)
    body = f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="{sw}"/>'
    task_templates = [
        "draw a polyline trend with {n} points in {color}",
        "create a zigzag line chart style polyline",
        "make an svg polyline path with multiple connected points",
    ]
    task = _choice(rng, task_templates).format(n=n, color=color)
    return Sample("polyline", task, _svg_wrap(w, h, body))


def _scatter_sample(rng: random.Random) -> Sample:
    w = rng.choice([200, 220, 240, 260])
    h = rng.choice([120, 140, 160])
    n = rng.randint(6, 10)
    variant = _spec_palette_variant(rng)
    if variant:
        dark, mid, light = variant
    else:
        dark = _choice(rng, PALETTE)
        mid = _choice(rng, [c for c in PALETTE if c != dark])
        light = "none"

    points = []
    for _ in range(n):
        x = rng.randint(20, w - 20)
        y = rng.randint(16, h - 24)
        r = rng.randint(2, 4)
        points.append((x, y, r))

    show_axes = rng.random() < 0.7
    body_parts: list[str] = []
    if light != "none" and rng.random() < 0.5:
        body_parts.append(
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="{light}" opacity="0.08"/>'
        )
    if show_axes:
        body_parts.append(
            f'<line x1="16" y1="{h - 18}" x2="{w - 14}" y2="{h - 18}" stroke="{dark}" stroke-width="1"/>'
        )
        body_parts.append(
            f'<line x1="16" y1="10" x2="16" y2="{h - 18}" stroke="{dark}" stroke-width="1"/>'
        )
    for x, y, r in points:
        body_parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{mid}" stroke="{dark}" stroke-width="1"/>'
        )

    task_templates = [
        "draw a scatter chart with {n} points",
        "create an svg scatter plot with clear point markers",
        "make a scatter diagram with point clusters and simple axes",
    ]
    task = _choice(rng, task_templates).format(n=n)
    return Sample("scatter", task, _svg_wrap(w, h, "".join(body_parts)))


def _gradient_square_sample(rng: random.Random) -> Sample:
    w = rng.choice([160, 180, 200, 220])
    h = rng.choice([90, 110, 130])
    x = rng.randint(10, 24)
    y = rng.randint(10, 24)
    side = rng.randint(52, min(w - x - 10, h - y - 10))
    variant = _spec_palette_variant(rng)
    if variant:
        c0, c1, txt = variant
    else:
        c0, c1, txt = rng.choice(THEME_PAIRS)
    gid = f"g{rng.randint(1000, 9999)}"
    title = _choice(rng, ["BLUE", "FLOW", "STACK", "MAP"])
    body = (
        f'<defs><linearGradient id="{gid}" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="{c0}"/>'
        f'<stop offset="100%" stop-color="{c1}"/>'
        f"</linearGradient></defs>"
        f'<rect x="{x}" y="{y}" width="{side}" height="{side}" rx="8" fill="url(#{gid})" stroke="white" stroke-width="1"/>'
        f'<text x="{x + 10}" y="{y + side // 2 + 4}" font-size="12" fill="{txt}">{title}</text>'
    )
    task_templates = [
        "draw a blue gradient square with white label text",
        "create a square card with a diagonal blue gradient background",
        "make an infographic square tile using a dark-to-blue gradient",
    ]
    task = _choice(rng, task_templates)
    return Sample("gradient_square", task, _svg_wrap(w, h, body))


def _infographic_layout_sample(rng: random.Random) -> Sample:
    w = 320
    h = 180
    variant = _spec_palette_variant(rng)
    if variant:
        c0, c1, txt = variant
    else:
        c0, c1, txt = rng.choice(THEME_PAIRS)
    gid = f"bg{rng.randint(1000, 9999)}"
    cards = []
    labels = rng.sample(["CPU", "MEM", "LAT", "TOK", "PAR", "LOSS"], 3)
    vals = [str(rng.randint(10, 99)) for _ in range(3)]
    for i in range(3):
        cx = 20 + i * 98
        cards.append(
            f'<rect x="{cx}" y="56" width="84" height="84" rx="10" fill="rgba(255,255,255,0.12)" stroke="white" stroke-width="1"/>'
            f'<text x="{cx + 10}" y="84" font-size="11" fill="{txt}">{labels[i]}</text>'
            f'<text x="{cx + 10}" y="114" font-size="22" fill="{txt}">{vals[i]}</text>'
        )
    body = (
        f'<defs><linearGradient id="{gid}" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%" stop-color="{c0}"/>'
        f'<stop offset="100%" stop-color="{c1}"/>'
        f"</linearGradient></defs>"
        f'<rect x="0" y="0" width="{w}" height="{h}" rx="12" fill="url(#{gid})"/>'
        f'<text x="20" y="30" font-size="16" fill="{txt}">SVG INFOGRAPHIC</text>'
        + "".join(cards)
    )
    task_templates = [
        "create an infographic with three metric cards and a blue gradient background",
        "draw a dashboard row of cards with white text on a dark blue gradient",
        "make a small svg infographic panel with three labeled value boxes",
    ]
    task = _choice(rng, task_templates)
    return Sample("infographic_layout", task, _svg_wrap(w, h, body))


def _generator_registry() -> dict[str, Callable[[random.Random], Sample]]:
    return {
        "circle": _circle_sample,
        "rect": _rect_sample,
        "line": _line_sample,
        "ellipse": _ellipse_sample,
        "triangle": _triangle_sample,
        "rounded_triangle": _rounded_triangle_sample,
        "polygon": _polygon_sample,
        "path": _path_sample,
        "arrow": _arrow_sample,
        "double_arrow": _double_arrow_sample,
        "text": _text_sample,
        "rect_circle": _rect_circle_scene_sample,
        "bar_chart": _bar_chart_sample,
        "scatter": _scatter_sample,
        "comparison_table": _comparison_table_sample,
        "polyline": _polyline_sample,
        "gradient_square": _gradient_square_sample,
        "infographic_layout": _infographic_layout_sample,
    }


def _pick_generator(rng: random.Random, enabled_types: list[str]) -> Callable[[random.Random], Sample]:
    registry = _generator_registry()
    key = enabled_types[rng.randrange(len(enabled_types))]
    return registry[key]


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_spec_catalog(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid spec catalog (expected object): {path}")
    specs = payload.get("specs")
    if not isinstance(specs, list) or not specs:
        raise SystemExit(f"invalid spec catalog (missing non-empty specs): {path}")
    return payload


def _normalize_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for raw in tags:
        t = str(raw).strip()
        if not t:
            continue
        if not (t.startswith("[") and t.endswith("]")):
            t = f"[{t}]"
        out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic SVG instruction dataset for v7")
    ap.add_argument("--out-dir", default="version/v7/data", help="Output directory")
    ap.add_argument("--prefix", default="svg_instruction", help="Filename prefix")
    ap.add_argument("--num-samples", type=int, default=10000, help="Total samples to generate")
    ap.add_argument("--holdout-ratio", type=float, default=0.10, help="Holdout split ratio [0,1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--types",
        default="all",
        help="Comma-separated shape families to generate (default: all). Use --list-types to inspect options.",
    )
    ap.add_argument(
        "--fill-mode",
        choices=["mixed", "filled", "outline"],
        default="mixed",
        help="Fill policy for closed shapes: mixed (default), filled, or outline-only (fill=none).",
    )
    ap.add_argument("--list-types", action="store_true", help="Print available generator type names and exit")
    ap.add_argument("--jsonl", action="store_true", help="Also emit JSONL files")
    ap.add_argument(
        "--spec-catalog",
        default=None,
        help="Optional spec catalog JSON. If set, generation is spec-driven with closed tags.",
    )
    ap.add_argument(
        "--spec-stage",
        default="all",
        help="When --spec-catalog is set, filter to one stage (or comma list), e.g. pretrain_a,pretrain_b,sft",
    )
    ap.add_argument(
        "--min-pair-count",
        type=int,
        default=30,
        help="Coverage gate floor for co-occurring tag pairs (spec mode).",
    )
    ap.add_argument(
        "--strict-coverage",
        action="store_true",
        help="Exit non-zero if coverage gate fails in spec mode.",
    )
    args = ap.parse_args()

    registry = _generator_registry()
    available_types = sorted(registry.keys())
    if args.list_types:
        print("\n".join(available_types))
        return 0

    raw_types = str(args.types).strip()
    if not raw_types or raw_types.lower() == "all":
        enabled_types = list(available_types)
    else:
        enabled_types = []
        seen: set[str] = set()
        for part in raw_types.split(","):
            t = part.strip().lower()
            if not t:
                continue
            if t not in registry:
                raise SystemExit(f"unknown type '{t}'. Use --list-types.")
            if t in seen:
                continue
            seen.add(t)
            enabled_types.append(t)
        if not enabled_types:
            raise SystemExit("no valid --types selected. Use --list-types.")

    spec_mode = bool(args.spec_catalog)
    if not spec_mode and args.num_samples < 1:
        raise SystemExit("--num-samples must be >= 1")
    if not spec_mode and not (0.0 <= args.holdout_ratio < 1.0):
        raise SystemExit("--holdout-ratio must be in [0,1)")
    if int(args.min_pair_count) < 0:
        raise SystemExit("--min-pair-count must be >= 0")

    global CURRENT_FILL_MODE, _SPEC_CTX
    CURRENT_FILL_MODE = str(args.fill_mode)
    _SPEC_CTX = {}

    out_dir = Path(args.out_dir).expanduser().resolve()
    rng = random.Random(int(args.seed))

    samples: list[Sample] = []
    train: list[Sample] = []
    holdout: list[Sample] = []
    type_counts: dict[str, int] = {}
    coverage_payload: dict[str, Any] | None = None

    if spec_mode:
        spec_catalog_path = Path(str(args.spec_catalog)).expanduser().resolve()
        if not spec_catalog_path.exists():
            raise SystemExit(f"--spec-catalog not found: {spec_catalog_path}")
        catalog = _load_spec_catalog(spec_catalog_path)
        raw_specs = catalog.get("specs") if isinstance(catalog, dict) else None
        assert isinstance(raw_specs, list)
        stage_filter_raw = str(args.spec_stage or "all").strip().lower()
        stage_filters = {
            s.strip().lower()
            for s in stage_filter_raw.split(",")
            if s.strip()
        } if stage_filter_raw and stage_filter_raw != "all" else set()

        selected_specs: list[dict] = []
        for spec in raw_specs:
            if not isinstance(spec, dict):
                continue
            if str(spec.get("status", "")).lower() == "stub":
                selected_specs.append(spec)
                continue
            stage_name = str(spec.get("stage", "unknown")).strip().lower()
            if stage_filters and stage_name not in stage_filters:
                continue
            selected_specs.append(spec)
        if not selected_specs:
            raise SystemExit(
                "no specs selected from catalog. Check --spec-stage filter."
            )

        tracker = CoverageTracker()
        for spec in selected_specs:
            if str(spec.get("status", "")).lower() == "stub":
                continue
            spec_id = str(spec.get("id", "")).strip()
            gen_name = str(spec.get("generator", "")).strip().lower()
            if not spec_id:
                raise SystemExit("spec entry missing id")
            if gen_name not in registry:
                raise SystemExit(f"spec {spec_id} references unknown generator: {gen_name}")
            spec_tags = _normalize_tags(list(spec.get("tags") or []))
            if not spec_tags:
                raise SystemExit(f"spec {spec_id} has empty tags")
            stage_name = str(spec.get("stage", "unknown")).strip().lower()
            _SPEC_CTX = {
                "spec_id": spec_id,
                "stage": stage_name,
                "palette_family": str(spec.get("palette_family", "")).strip().lower(),
                "style": str(spec.get("style", "mixed")).strip().lower(),
            }
            gen = registry[gen_name]
            n_train = int(spec.get("samples_train", 0) or 0)
            n_hold = int(spec.get("samples_holdout", 0) or 0)
            if n_train < 0 or n_hold < 0:
                raise SystemExit(f"spec {spec_id} has negative sample counts")
            for _ in range(n_train):
                s = gen(rng)
                s.tags = list(spec_tags)
                _assert_ascii(s.svg, "svg")
                _validate_svg(s.svg)
                _assert_ascii(s.tag_line(), "tag line")
                train.append(s)
                tracker.record(spec_id, s.tags, split="train")
                type_counts[s.sample_type] = int(type_counts.get(s.sample_type, 0) + 1)
            for _ in range(n_hold):
                s = gen(rng)
                s.tags = list(spec_tags)
                _assert_ascii(s.svg, "svg")
                _validate_svg(s.svg)
                _assert_ascii(s.tag_line(), "tag line")
                holdout.append(s)
                tracker.record(spec_id, s.tags, split="holdout")
                type_counts[s.sample_type] = int(type_counts.get(s.sample_type, 0) + 1)

        rng.shuffle(train)
        rng.shuffle(holdout)
        samples = [*train, *holdout]
        gate = tracker.gate(selected_specs, min_pair_count=int(args.min_pair_count))
        coverage_payload = {
            "format": "v7-svg-coverage-manifest",
            "spec_catalog_path": str(spec_catalog_path),
            "spec_stage_filter": sorted(list(stage_filters)) if stage_filters else ["all"],
            "spec_counts_total": dict(sorted(tracker.spec_counts_total.items())),
            "spec_counts_train": dict(sorted(tracker.spec_counts_train.items())),
            "spec_counts_holdout": dict(sorted(tracker.spec_counts_holdout.items())),
            "tag_pair_counts": [
                {"pair": [a, b], "count": int(c)}
                for (a, b), c in sorted(tracker.pair_counts.items(), key=lambda kv: (kv[0][0], kv[0][1]))
            ],
            "gate": gate,
        }
        if bool(args.strict_coverage) and not bool(gate.get("passed")):
            raise SystemExit(
                "spec coverage gate failed:\n  - "
                + "\n  - ".join([str(x) for x in gate.get("failures", [])])
            )
    else:
        for _ in range(int(args.num_samples)):
            gen = _pick_generator(rng, enabled_types)
            s = gen(rng)
            _assert_ascii(s.task, "task")
            _assert_ascii(s.svg, "svg")
            _validate_svg(s.svg)
            line = s.instruction_line()
            _assert_ascii(line, "instruction line")
            samples.append(s)
            type_counts[s.sample_type] = int(type_counts.get(s.sample_type, 0) + 1)

        rng.shuffle(samples)
        holdout_n = int(round(len(samples) * float(args.holdout_ratio)))
        holdout_n = min(max(0, holdout_n), len(samples))
        train_n = len(samples) - holdout_n
        train = samples[:train_n]
        holdout = samples[train_n:]

    if spec_mode:
        instruction_all = [s.tag_line() for s in samples]
        instruction_train = [s.tag_line() for s in train]
        instruction_holdout = [s.tag_line() for s in holdout]
        instruction_format = "[tag... ]<svg ...>...</svg><eos>"
    else:
        instruction_all = [s.instruction_line() for s in samples]
        instruction_train = [s.instruction_line() for s in train]
        instruction_holdout = [s.instruction_line() for s in holdout]
        instruction_format = "<task>...</task><svg ...>...</svg><eos>"

    svg_all = [s.svg for s in samples]
    svg_train = [s.svg for s in train]
    svg_holdout = [s.svg for s in holdout]

    paths = {
        "instruction_all_txt": out_dir / f"{args.prefix}_instruction_all.txt",
        "instruction_train_txt": out_dir / f"{args.prefix}_instruction_train.txt",
        "instruction_holdout_txt": out_dir / f"{args.prefix}_instruction_holdout.txt",
        "svg_all_txt": out_dir / f"{args.prefix}_svg_all.txt",
        "svg_train_txt": out_dir / f"{args.prefix}_svg_train.txt",
        "svg_holdout_txt": out_dir / f"{args.prefix}_svg_holdout.txt",
        "manifest_json": out_dir / f"{args.prefix}_manifest.json",
        "coverage_manifest_json": out_dir / f"{args.prefix}_coverage_manifest.json",
    }

    _write_lines(paths["instruction_all_txt"], instruction_all)
    _write_lines(paths["instruction_train_txt"], instruction_train)
    _write_lines(paths["instruction_holdout_txt"], instruction_holdout)
    _write_lines(paths["svg_all_txt"], svg_all)
    _write_lines(paths["svg_train_txt"], svg_train)
    _write_lines(paths["svg_holdout_txt"], svg_holdout)

    manifest = {
        "format": "v7-svg-instruction-dataset",
        "num_samples": int(len(samples)),
        "num_train": int(len(train)),
        "num_holdout": int(len(holdout)),
        "holdout_ratio": float(args.holdout_ratio),
        "seed": int(args.seed),
        "enabled_types": enabled_types,
        "fill_mode": str(args.fill_mode),
        "type_counts": type_counts,
        "paths": {k: str(v) for k, v in paths.items()},
        "instruction_format": instruction_format,
        "spec_mode": bool(spec_mode),
        "spec_catalog": str(args.spec_catalog) if spec_mode else None,
        "spec_stage": str(args.spec_stage) if spec_mode else None,
        "notes": [
            "ASCII-only content",
            "Single-line records",
            "Use instruction_* files for instruction-following training",
            "Use svg_* files for strict <svg-prefix completion training",
        ],
    }
    if coverage_payload is not None:
        manifest["coverage_manifest_path"] = str(paths["coverage_manifest_json"])
        manifest["coverage_gate"] = dict(coverage_payload.get("gate", {}))
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if coverage_payload is not None:
        paths["coverage_manifest_json"].write_text(json.dumps(coverage_payload, indent=2), encoding="utf-8")

    if args.jsonl:
        jsonl_all = [
            {
                "text": (s.tag_line() if spec_mode else s.instruction_line()),
                "task": s.task,
                "svg": s.svg,
                "type": s.sample_type,
                "tags": list(s.tags),
            }
            for s in samples
        ]
        jsonl_train = [
            {
                "text": (s.tag_line() if spec_mode else s.instruction_line()),
                "task": s.task,
                "svg": s.svg,
                "type": s.sample_type,
                "tags": list(s.tags),
            }
            for s in train
        ]
        jsonl_holdout = [
            {
                "text": (s.tag_line() if spec_mode else s.instruction_line()),
                "task": s.task,
                "svg": s.svg,
                "type": s.sample_type,
                "tags": list(s.tags),
            }
            for s in holdout
        ]
        _write_jsonl(out_dir / f"{args.prefix}_instruction_all.jsonl", jsonl_all)
        _write_jsonl(out_dir / f"{args.prefix}_instruction_train.jsonl", jsonl_train)
        _write_jsonl(out_dir / f"{args.prefix}_instruction_holdout.jsonl", jsonl_holdout)

    print(f"[OK] Generated {len(samples)} samples")
    print(f"[OK] train={len(train)} holdout={len(holdout)}")
    print(f"[OK] instruction train txt: {paths['instruction_train_txt']}")
    print(f"[OK] svg train txt:         {paths['svg_train_txt']}")
    print(f"[OK] manifest:              {paths['manifest_json']}")
    if coverage_payload is not None:
        print(f"[OK] coverage manifest:     {paths['coverage_manifest_json']}")
        print(f"[OK] coverage gate:         {coverage_payload.get('gate', {})}")
    print(f"[OK] type counts:           {type_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
