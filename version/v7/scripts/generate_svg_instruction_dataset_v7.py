#!/usr/bin/env python3
"""
Generate synthetic ASCII-only instruction->SVG datasets for v7 training.

Outputs:
  - instruction dataset: <task>...</task><svg ...>...</svg><eos>
  - svg-only dataset:    <svg ...>...</svg>
  - train/holdout splits
  - optional JSONL
  - manifest with type counts and paths
"""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


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

CURRENT_FILL_MODE = "mixed"


@dataclass
class Sample:
    sample_type: str
    task: str
    svg: str

    def instruction_line(self) -> str:
        return f"<task>{self.task}</task>{self.svg}<eos>"


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


def _line_sample(rng: random.Random) -> Sample:
    w = rng.choice([120, 140, 160, 180, 200])
    h = rng.choice([80, 100, 120, 140])
    x1 = rng.randint(6, w - 30)
    y1 = rng.randint(6, h - 30)
    x2 = rng.randint(20, w - 6)
    y2 = rng.randint(20, h - 6)
    color = _choice(rng, PALETTE)
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
    fill_color = _choice(rng, PALETTE)
    stroke = _choice(rng, [c for c in PALETTE if c != fill_color])
    fill = _resolve_fill(rng, fill_color)
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
    fill_color = _choice(rng, PALETTE)
    stroke = _choice(rng, [c for c in PALETTE if c != fill_color])
    fill = _resolve_fill(rng, fill_color)
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


def _polygon_sample(rng: random.Random) -> Sample:
    w = rng.choice([150, 180, 210, 240])
    h = rng.choice([100, 120, 140, 160])
    n = rng.randint(5, 8)
    pts: list[tuple[int, int]] = []
    for i in range(n):
        x = rng.randint(12, w - 12)
        y = rng.randint(12, h - 12)
        pts.append((x, y))
    fill_color = _choice(rng, PALETTE)
    stroke = _choice(rng, [c for c in PALETTE if c != fill_color])
    fill = _resolve_fill(rng, fill_color)
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
    stroke = _choice(rng, PALETTE)
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
    stroke = _choice(rng, PALETTE)
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
    fill_color = _choice(rng, PALETTE)
    stroke = _choice(rng, [c for c in PALETTE if c != fill_color])
    fill = _resolve_fill(rng, fill_color)
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
    stroke = _choice(rng, PALETTE)
    fill_color = _choice(rng, [c for c in PALETTE if c != stroke])
    fill = _resolve_fill(rng, fill_color)
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
    fill = _choice(rng, PALETTE)
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
    rc = _choice(rng, PALETTE)
    cc = _choice(rng, [c for c in PALETTE if c != rc])
    fill_rc = _resolve_fill(rng, rc)
    fill_cc = _resolve_fill(rng, cc)
    body = (
        f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" fill="{fill_rc}" stroke="black"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill_cc}" stroke="black"/>'
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
    color = _choice(rng, PALETTE)
    sw = rng.randint(2, 4)
    body = f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="{sw}"/>'
    task_templates = [
        "draw a polyline trend with {n} points in {color}",
        "create a zigzag line chart style polyline",
        "make an svg polyline path with multiple connected points",
    ]
    task = _choice(rng, task_templates).format(n=n, color=color)
    return Sample("polyline", task, _svg_wrap(w, h, body))


def _generator_registry() -> dict[str, Callable[[random.Random], Sample]]:
    return {
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
        "comparison_table": _comparison_table_sample,
        "polyline": _polyline_sample,
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

    if args.num_samples < 1:
        raise SystemExit("--num-samples must be >= 1")
    if not (0.0 <= args.holdout_ratio < 1.0):
        raise SystemExit("--holdout-ratio must be in [0,1)")

    global CURRENT_FILL_MODE
    CURRENT_FILL_MODE = str(args.fill_mode)

    out_dir = Path(args.out_dir).expanduser().resolve()
    rng = random.Random(int(args.seed))

    samples: list[Sample] = []
    type_counts: dict[str, int] = {}
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

    instruction_all = [s.instruction_line() for s in samples]
    instruction_train = [s.instruction_line() for s in train]
    instruction_holdout = [s.instruction_line() for s in holdout]
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
        "instruction_format": "<task>...</task><svg ...>...</svg><eos>",
        "notes": [
            "ASCII-only content",
            "Single-line records",
            "Use instruction_* files for instruction-following training",
            "Use svg_* files for strict <svg-prefix completion training",
        ],
    }
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.jsonl:
        jsonl_all = [
            {"text": s.instruction_line(), "task": s.task, "svg": s.svg, "type": s.sample_type}
            for s in samples
        ]
        jsonl_train = [
            {"text": s.instruction_line(), "task": s.task, "svg": s.svg, "type": s.sample_type}
            for s in train
        ]
        jsonl_holdout = [
            {"text": s.instruction_line(), "task": s.task, "svg": s.svg, "type": s.sample_type}
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
    print(f"[OK] type counts:           {type_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
