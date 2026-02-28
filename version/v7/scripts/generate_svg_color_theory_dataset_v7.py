#!/usr/bin/env python3
"""
Generate SVG color-theory datasets for v7 training.

Output rows are ASCII and one-sample-per-line:
  <task>...</task><svg ...>...</svg><eos>

The generator encodes:
- harmony families (complementary / analogous / triadic / split-complementary / monochromatic)
- background gradients
- readable text colors selected with WCAG contrast checks
- font-family variation
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


SCHEMES = (
    "complementary",
    "analogous",
    "triadic",
    "split_complementary",
    "monochromatic",
)

TEMPLATES = ("card", "dashboard", "chart")

DEFAULT_FONTS = (
    "Arial",
    "Verdana",
    "Tahoma",
    "Trebuchet MS",
    "Georgia",
)

TEXT_CANDIDATES_LIGHT = ("#0B1020", "#111827", "#1F2937", "#374151")
TEXT_CANDIDATES_DARK = ("#FFFFFF", "#F8FAFC", "#E5E7EB", "#D1D5DB")
HEX_RE = re.compile(r"#[0-9A-Fa-f]{6}")


@dataclass
class Sample:
    scheme: str
    template: str
    task: str
    svg: str
    palette: dict[str, str]
    contrast: dict[str, float]
    theme: str
    font_family: str

    def instruction_line(self) -> str:
        return f"<task>{self.task}</task>{self.svg}<eos>"


def _assert_ascii(text: str, where: str) -> None:
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"non-ASCII content in {where}: {exc}") from exc


def _validate_svg(svg: str) -> None:
    root = ET.fromstring(svg)
    tag = root.tag.split("}", 1)[-1] if "}" in root.tag else root.tag
    if tag != "svg":
        raise RuntimeError(f"invalid SVG root: {tag}")


def _norm_h(h: float) -> float:
    out = float(h) % 360.0
    return out if out >= 0.0 else out + 360.0


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    h = _norm_h(h) / 360.0
    s = min(max(float(s), 0.0), 1.0)
    l = min(max(float(l), 0.0), 1.0)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(round(r * 255)):02X}{int(round(g * 255)):02X}{int(round(b * 255)):02X}"


def _hex_to_rgb01(color: str) -> tuple[float, float, float]:
    c = color.strip()
    if not c.startswith("#") or len(c) != 7:
        raise ValueError(f"expected #RRGGBB, got: {color}")
    r = int(c[1:3], 16) / 255.0
    g = int(c[3:5], 16) / 255.0
    b = int(c[5:7], 16) / 255.0
    return r, g, b


def _linearize(v: float) -> float:
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _relative_luminance(color: str) -> float:
    r, g, b = _hex_to_rgb01(color)
    rl = _linearize(r)
    gl = _linearize(g)
    bl = _linearize(b)
    return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl


def _contrast_ratio(a: str, b: str) -> float:
    la = _relative_luminance(a)
    lb = _relative_luminance(b)
    hi = max(la, lb)
    lo = min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _scheme_hues(base_h: float, scheme: str, rng: random.Random) -> tuple[float, float, float]:
    if scheme == "complementary":
        return base_h, _norm_h(base_h + 180.0), _norm_h(base_h + 180.0)
    if scheme == "analogous":
        d = rng.choice((20.0, 24.0, 30.0))
        return _norm_h(base_h - d), base_h, _norm_h(base_h + d)
    if scheme == "triadic":
        return base_h, _norm_h(base_h + 120.0), _norm_h(base_h + 240.0)
    if scheme == "split_complementary":
        return base_h, _norm_h(base_h + 150.0), _norm_h(base_h + 210.0)
    if scheme == "monochromatic":
        return base_h, base_h, base_h
    raise ValueError(f"unknown scheme: {scheme}")


def _pick_text_color(candidates: tuple[str, ...], backgrounds: tuple[str, ...]) -> tuple[str, float]:
    best_color = candidates[0]
    best_min = -1.0
    for c in candidates:
        m = min(_contrast_ratio(c, bg) for bg in backgrounds)
        if m > best_min:
            best_min = m
            best_color = c
    return best_color, best_min


def _mix_hex(a: str, b: str, t: float) -> str:
    t = min(max(float(t), 0.0), 1.0)
    ar, ag, ab = _hex_to_rgb01(a)
    br, bg, bb = _hex_to_rgb01(b)
    r = ar * (1.0 - t) + br * t
    g = ag * (1.0 - t) + bg * t
    b2 = ab * (1.0 - t) + bb * t
    return f"#{int(round(r * 255)):02X}{int(round(g * 255)):02X}{int(round(b2 * 255)):02X}"


def _load_external_palettes(paths: list[str]) -> list[list[str]]:
    out: list[list[str]] = []
    for raw_path in paths:
        p = Path(raw_path).expanduser().resolve()
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            colors = [c.upper() for c in HEX_RE.findall(line)]
            if len(colors) < 3:
                continue
            dedup = list(dict.fromkeys(colors))
            if len(dedup) < 3:
                continue
            out.append(dedup[:5])
    return out


def _build_palette_from_external(colors: list[str], min_contrast: float) -> tuple[dict[str, str], dict[str, float], str]:
    bg_a = colors[0]
    bg_b = colors[1]
    accent = colors[2]
    card = colors[3] if len(colors) >= 4 else _mix_hex(bg_a, bg_b, 0.50)
    border = colors[4] if len(colors) >= 5 else _mix_hex(accent, card, 0.35)

    avg_l = (_relative_luminance(bg_a) + _relative_luminance(bg_b)) / 2.0
    dark_theme = avg_l < 0.42
    text_candidates = (TEXT_CANDIDATES_DARK + TEXT_CANDIDATES_LIGHT) if dark_theme else (TEXT_CANDIDATES_LIGHT + TEXT_CANDIDATES_DARK)
    text_main, min_seen = _pick_text_color(text_candidates, (bg_a, bg_b, card))
    if min_seen < min_contrast:
        bw, bw_min = _pick_text_color(("#FFFFFF", "#000000"), (bg_a, bg_b, card))
        if bw_min > min_seen:
            text_main = bw
    text_sub, _ = _pick_text_color(text_candidates, (card,))

    palette = {
        "bg_a": bg_a,
        "bg_b": bg_b,
        "card": card,
        "accent": accent,
        "border": border,
        "text_main": text_main,
        "text_sub": text_sub,
    }
    contrast = {
        "text_main_vs_bg_a": _contrast_ratio(text_main, bg_a),
        "text_main_vs_bg_b": _contrast_ratio(text_main, bg_b),
        "text_main_vs_card": _contrast_ratio(text_main, card),
        "text_sub_vs_card": _contrast_ratio(text_sub, card),
        "min_text_contrast": min(
            _contrast_ratio(text_main, bg_a),
            _contrast_ratio(text_main, bg_b),
            _contrast_ratio(text_main, card),
        ),
    }
    return palette, contrast, ("dark" if dark_theme else "light")


def _build_palette(rng: random.Random, scheme: str, min_contrast: float) -> tuple[dict[str, str], dict[str, float], str]:
    base_h = float(rng.randint(0, 359))
    h0, h1, h2 = _scheme_hues(base_h, scheme, rng)
    dark_theme = rng.random() < 0.62
    theme = "dark" if dark_theme else "light"

    if dark_theme:
        s_bg = rng.uniform(0.45, 0.85)
        bg_l1 = rng.uniform(0.08, 0.18)
        bg_l2 = rng.uniform(0.20, 0.32)
        card_l = rng.uniform(0.16, 0.28)
        accent_l = rng.uniform(0.52, 0.66)
        text_candidates = TEXT_CANDIDATES_DARK + TEXT_CANDIDATES_LIGHT
    else:
        s_bg = rng.uniform(0.35, 0.70)
        bg_l1 = rng.uniform(0.84, 0.94)
        bg_l2 = rng.uniform(0.70, 0.84)
        card_l = rng.uniform(0.92, 0.98)
        accent_l = rng.uniform(0.36, 0.52)
        text_candidates = TEXT_CANDIDATES_LIGHT + TEXT_CANDIDATES_DARK

    bg_a = _hsl_to_hex(h0, s_bg, bg_l1)
    bg_b = _hsl_to_hex(h1, min(1.0, s_bg + 0.08), bg_l2)
    card = _hsl_to_hex(h0, min(0.45, s_bg * 0.55), card_l)
    accent = _hsl_to_hex(h2, rng.uniform(0.62, 0.98), accent_l)
    border = _hsl_to_hex(h2, rng.uniform(0.25, 0.55), 0.72 if dark_theme else 0.30)

    text_main, min_seen = _pick_text_color(text_candidates, (bg_a, bg_b, card))
    if min_seen < min_contrast:
        # Force black/white fallback if needed.
        bw, bw_min = _pick_text_color(("#FFFFFF", "#000000"), (bg_a, bg_b, card))
        if bw_min > min_seen:
            text_main = bw
            min_seen = bw_min
    text_sub, _ = _pick_text_color(text_candidates, (card,))

    palette = {
        "bg_a": bg_a,
        "bg_b": bg_b,
        "card": card,
        "accent": accent,
        "border": border,
        "text_main": text_main,
        "text_sub": text_sub,
    }
    contrast = {
        "text_main_vs_bg_a": _contrast_ratio(text_main, bg_a),
        "text_main_vs_bg_b": _contrast_ratio(text_main, bg_b),
        "text_main_vs_card": _contrast_ratio(text_main, card),
        "text_sub_vs_card": _contrast_ratio(text_sub, card),
        "min_text_contrast": min(
            _contrast_ratio(text_main, bg_a),
            _contrast_ratio(text_main, bg_b),
            _contrast_ratio(text_main, card),
        ),
    }
    return palette, contrast, theme


def _svg_wrap(width: int, height: int, body: str) -> str:
    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'


def _template_card(rng: random.Random, palette: dict[str, str], font: str) -> str:
    gid = f"g{rng.randint(1000, 9999)}"
    body = (
        f'<defs><linearGradient id="{gid}" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="{palette["bg_a"]}"/>'
        f'<stop offset="100%" stop-color="{palette["bg_b"]}"/>'
        f"</linearGradient></defs>"
        f'<rect x="0" y="0" width="320" height="180" rx="14" fill="url(#{gid})"/>'
        f'<rect x="18" y="48" width="284" height="108" rx="12" fill="{palette["card"]}" stroke="{palette["border"]}" stroke-width="1"/>'
        f'<text x="24" y="30" font-family="{font}, sans-serif" font-size="15" fill="{palette["text_main"]}">COLOR SPECTRUM</text>'
        f'<text x="30" y="84" font-family="{font}, sans-serif" font-size="12" fill="{palette["text_sub"]}">Readable contrast + harmony</text>'
        f'<circle cx="40" cy="122" r="7" fill="{palette["accent"]}"/>'
        f'<circle cx="62" cy="122" r="7" fill="{palette["border"]}"/>'
        f'<rect x="84" y="114" width="120" height="14" rx="7" fill="{palette["accent"]}"/>'
    )
    return _svg_wrap(320, 180, body)


def _template_dashboard(rng: random.Random, palette: dict[str, str], font: str) -> str:
    gid = f"bg{rng.randint(1000, 9999)}"
    vals = [rng.randint(10, 99) for _ in range(3)]
    labels = rng.sample(["CPU", "MEM", "LAT", "TOK", "PAR", "LOSS"], 3)
    cards = []
    for i in range(3):
        x = 18 + i * 100
        cards.append(
            f'<rect x="{x}" y="58" width="88" height="92" rx="10" fill="{palette["card"]}" stroke="{palette["border"]}" stroke-width="1"/>'
            f'<text x="{x + 10}" y="86" font-family="{font}, sans-serif" font-size="11" fill="{palette["text_sub"]}">{labels[i]}</text>'
            f'<text x="{x + 10}" y="118" font-family="{font}, sans-serif" font-size="22" fill="{palette["text_main"]}">{vals[i]}</text>'
        )
    body = (
        f'<defs><linearGradient id="{gid}" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%" stop-color="{palette["bg_a"]}"/>'
        f'<stop offset="100%" stop-color="{palette["bg_b"]}"/>'
        f"</linearGradient></defs>"
        f'<rect x="0" y="0" width="340" height="190" rx="14" fill="url(#{gid})"/>'
        f'<text x="20" y="34" font-family="{font}, sans-serif" font-size="16" fill="{palette["text_main"]}">INFOGRAPHIC DASHBOARD</text>'
        + "".join(cards)
    )
    return _svg_wrap(340, 190, body)


def _template_chart(rng: random.Random, palette: dict[str, str], font: str) -> str:
    gid = f"c{rng.randint(1000, 9999)}"
    bars = [rng.randint(26, 86) for _ in range(5)]
    body_parts = [
        f'<defs><linearGradient id="{gid}" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="{palette["bg_a"]}"/>'
        f'<stop offset="100%" stop-color="{palette["bg_b"]}"/>'
        f"</linearGradient></defs>",
        f'<rect x="0" y="0" width="360" height="200" rx="14" fill="url(#{gid})"/>',
        f'<rect x="16" y="44" width="328" height="138" rx="10" fill="{palette["card"]}" stroke="{palette["border"]}" stroke-width="1"/>',
        f'<line x1="30" y1="168" x2="334" y2="168" stroke="{palette["text_sub"]}" stroke-width="1"/>',
    ]
    for i, v in enumerate(bars):
        x = 42 + i * 58
        y = 168 - v
        body_parts.append(f'<rect x="{x}" y="{y}" width="30" height="{v}" fill="{palette["accent"]}" rx="4"/>')
    body_parts.append(
        f'<text x="24" y="30" font-family="{font}, sans-serif" font-size="15" fill="{palette["text_main"]}">COLOR-THEORY CHART CARD</text>'
    )
    return _svg_wrap(360, 200, "".join(body_parts))


def _sample_task(scheme: str, palette: dict[str, str], contrast: dict[str, float], font: str, template: str) -> str:
    scheme_words = scheme.replace("_", " ")
    return (
        f"create a {scheme_words} {template} infographic using gradient {palette['bg_a']} to {palette['bg_b']}, "
        f"accent {palette['accent']}, font {font}, and readable text color {palette['text_main']} "
        f"(wcag min contrast {contrast['min_text_contrast']:.2f})"
    )


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _parse_list(raw: str) -> list[str]:
    items = [p.strip() for p in raw.split(",") if p.strip()]
    dedup: list[str] = []
    seen: set[str] = set()
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        dedup.append(it)
    return dedup


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SVG color-theory dataset with contrast-safe palettes")
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="svg_color_theory", help="Output prefix")
    ap.add_argument("--num-samples", type=int, default=20000, help="Total samples")
    ap.add_argument("--holdout-ratio", type=float, default=0.10, help="Holdout ratio [0,1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--min-contrast", type=float, default=4.5, help="Target minimum text contrast ratio")
    ap.add_argument("--fonts", default="Arial,Verdana,Tahoma,Trebuchet MS,Georgia", help="Comma-separated font families")
    ap.add_argument("--schemes", default="all", help="Comma-separated harmony schemes or 'all'")
    ap.add_argument("--templates", default="all", help="Comma-separated templates or 'all'")
    ap.add_argument(
        "--palette-file",
        action="append",
        default=[],
        help="Optional external palette source (text/csv/jsonl with #RRGGBB values per line). Repeatable.",
    )
    ap.add_argument(
        "--external-palette-prob",
        type=float,
        default=0.0,
        help="Probability [0,1] to draw a sample from --palette-file palettes instead of generated harmony.",
    )
    ap.add_argument("--jsonl", action="store_true", help="Also emit JSONL rows with metadata")
    ap.add_argument("--catalog-csv", action="store_true", help="Also emit CSV palette catalog")
    ap.add_argument("--list-schemes", action="store_true", help="Print available scheme names and exit")
    ap.add_argument("--list-templates", action="store_true", help="Print available template names and exit")
    args = ap.parse_args()

    if args.list_schemes:
        print("\n".join(SCHEMES))
        return 0
    if args.list_templates:
        print("\n".join(TEMPLATES))
        return 0

    if args.num_samples < 1:
        raise SystemExit("--num-samples must be >= 1")
    if not (0.0 <= args.holdout_ratio < 1.0):
        raise SystemExit("--holdout-ratio must be in [0,1)")
    if args.min_contrast < 1.0 or args.min_contrast > 21.0:
        raise SystemExit("--min-contrast must be in [1,21]")
    if not (0.0 <= float(args.external_palette_prob) <= 1.0):
        raise SystemExit("--external-palette-prob must be in [0,1]")

    fonts = _parse_list(args.fonts)
    if not fonts:
        fonts = list(DEFAULT_FONTS)

    if str(args.schemes).strip().lower() == "all":
        schemes = list(SCHEMES)
    else:
        schemes = _parse_list(str(args.schemes).lower())
        bad = [s for s in schemes if s not in SCHEMES]
        if bad:
            raise SystemExit(f"unknown schemes: {bad}; use --list-schemes")
    if str(args.templates).strip().lower() == "all":
        templates = list(TEMPLATES)
    else:
        templates = _parse_list(str(args.templates).lower())
        bad = [t for t in templates if t not in TEMPLATES]
        if bad:
            raise SystemExit(f"unknown templates: {bad}; use --list-templates")

    renderers = {
        "card": _template_card,
        "dashboard": _template_dashboard,
        "chart": _template_chart,
    }

    rng = random.Random(int(args.seed))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    external_palettes = _load_external_palettes(list(args.palette_file))

    rows: list[Sample] = []
    scheme_counts: dict[str, int] = {s: 0 for s in schemes}
    if external_palettes:
        scheme_counts["external_palette"] = 0
    template_counts: dict[str, int] = {t: 0 for t in templates}

    target_min_contrast = float(args.min_contrast)
    for _ in range(int(args.num_samples)):
        prefer_external = bool(external_palettes) and (rng.random() < float(args.external_palette_prob))
        template = templates[rng.randrange(len(templates))]
        font = fonts[rng.randrange(len(fonts))]
        chosen: Sample | None = None
        best: Sample | None = None
        best_score = -1.0
        for attempt in range(12):
            if prefer_external and attempt < 4 and external_palettes:
                scheme = "external_palette"
                palette_seed = external_palettes[rng.randrange(len(external_palettes))]
                palette, contrast, theme = _build_palette_from_external(palette_seed, target_min_contrast)
            else:
                scheme = schemes[rng.randrange(len(schemes))]
                palette, contrast, theme = _build_palette(rng, scheme, target_min_contrast)

            svg = renderers[template](rng, palette, font)
            task = _sample_task(scheme, palette, contrast, font, template)
            candidate = Sample(
                scheme=scheme,
                template=template,
                task=task,
                svg=svg,
                palette=palette,
                contrast=contrast,
                theme=theme,
                font_family=font,
            )
            score = float(candidate.contrast["min_text_contrast"])
            if score > best_score:
                best_score = score
                best = candidate
            if score >= target_min_contrast:
                chosen = candidate
                break

        sample = chosen or best
        if sample is None:
            raise RuntimeError("failed to generate any sample")
        line = sample.instruction_line()
        _assert_ascii(sample.task, "task")
        _assert_ascii(sample.svg, "svg")
        _assert_ascii(line, "instruction_line")
        _validate_svg(sample.svg)
        rows.append(sample)
        scheme_counts[scheme] = int(scheme_counts.get(scheme, 0) + 1)
        template_counts[template] = int(template_counts.get(template, 0) + 1)

    rng.shuffle(rows)
    holdout_n = int(round(len(rows) * float(args.holdout_ratio)))
    holdout_n = min(max(0, holdout_n), len(rows))
    train_n = len(rows) - holdout_n
    train_rows = rows[:train_n]
    holdout_rows = rows[train_n:]

    instruction_all = [r.instruction_line() for r in rows]
    instruction_train = [r.instruction_line() for r in train_rows]
    instruction_holdout = [r.instruction_line() for r in holdout_rows]
    svg_all = [r.svg for r in rows]
    svg_train = [r.svg for r in train_rows]
    svg_holdout = [r.svg for r in holdout_rows]

    min_contrasts = [r.contrast["min_text_contrast"] for r in rows]
    contrast_stats = {
        "min": float(min(min_contrasts)) if min_contrasts else math.nan,
        "max": float(max(min_contrasts)) if min_contrasts else math.nan,
        "avg": float(sum(min_contrasts) / len(min_contrasts)) if min_contrasts else math.nan,
        "below_target": int(sum(1 for v in min_contrasts if v < float(args.min_contrast))),
        "target": float(args.min_contrast),
    }

    paths = {
        "instruction_all_txt": out_dir / f"{args.prefix}_instruction_all.txt",
        "instruction_train_txt": out_dir / f"{args.prefix}_instruction_train.txt",
        "instruction_holdout_txt": out_dir / f"{args.prefix}_instruction_holdout.txt",
        "svg_all_txt": out_dir / f"{args.prefix}_svg_all.txt",
        "svg_train_txt": out_dir / f"{args.prefix}_svg_train.txt",
        "svg_holdout_txt": out_dir / f"{args.prefix}_svg_holdout.txt",
        "manifest_json": out_dir / f"{args.prefix}_manifest.json",
        "catalog_csv": out_dir / f"{args.prefix}_palette_catalog.csv",
        "all_jsonl": out_dir / f"{args.prefix}_all.jsonl",
        "train_jsonl": out_dir / f"{args.prefix}_train.jsonl",
        "holdout_jsonl": out_dir / f"{args.prefix}_holdout.jsonl",
    }

    _write_lines(paths["instruction_all_txt"], instruction_all)
    _write_lines(paths["instruction_train_txt"], instruction_train)
    _write_lines(paths["instruction_holdout_txt"], instruction_holdout)
    _write_lines(paths["svg_all_txt"], svg_all)
    _write_lines(paths["svg_train_txt"], svg_train)
    _write_lines(paths["svg_holdout_txt"], svg_holdout)

    if args.jsonl:
        def _to_jsonl_obj(s: Sample) -> dict:
            return {
                "text": s.instruction_line(),
                "task": s.task,
                "svg": s.svg,
                "scheme": s.scheme,
                "template": s.template,
                "theme": s.theme,
                "font_family": s.font_family,
                "palette": s.palette,
                "contrast": s.contrast,
            }
        _write_jsonl(paths["all_jsonl"], [_to_jsonl_obj(s) for s in rows])
        _write_jsonl(paths["train_jsonl"], [_to_jsonl_obj(s) for s in train_rows])
        _write_jsonl(paths["holdout_jsonl"], [_to_jsonl_obj(s) for s in holdout_rows])

    if args.catalog_csv:
        csv_rows = []
        for s in rows:
            csv_rows.append(
                {
                    "scheme": s.scheme,
                    "template": s.template,
                    "theme": s.theme,
                    "font_family": s.font_family,
                    "bg_a": s.palette["bg_a"],
                    "bg_b": s.palette["bg_b"],
                    "card": s.palette["card"],
                    "accent": s.palette["accent"],
                    "text_main": s.palette["text_main"],
                    "text_sub": s.palette["text_sub"],
                    "min_text_contrast": f"{s.contrast['min_text_contrast']:.4f}",
                }
            )
        _write_csv(paths["catalog_csv"], csv_rows)

    manifest = {
        "format": "v7-svg-color-theory-dataset",
        "num_samples": int(len(rows)),
        "num_train": int(len(train_rows)),
        "num_holdout": int(len(holdout_rows)),
        "holdout_ratio": float(args.holdout_ratio),
        "seed": int(args.seed),
        "schemes": schemes,
        "external_palette_files": [str(Path(p).expanduser().resolve()) for p in args.palette_file],
        "external_palette_count": int(len(external_palettes)),
        "external_palette_prob": float(args.external_palette_prob),
        "templates": templates,
        "fonts": fonts,
        "scheme_counts": scheme_counts,
        "template_counts": template_counts,
        "contrast_stats": contrast_stats,
        "paths": {k: str(v) for k, v in paths.items()},
        "instruction_format": "<task>...</task><svg ...>...</svg><eos>",
        "notes": [
            "Palette generation uses harmony families + WCAG contrast target checks.",
            "Rows are ASCII-only and one-line samples.",
            "Use instruction_* for semantic+SVG training; use svg_* for strict SVG completion.",
        ],
    }
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] generated: {len(rows)} rows (train={len(train_rows)} holdout={len(holdout_rows)})")
    print(f"[OK] instruction train: {paths['instruction_train_txt']}")
    print(f"[OK] svg train:         {paths['svg_train_txt']}")
    print(f"[OK] manifest:          {paths['manifest_json']}")
    print(f"[OK] contrast:          min={contrast_stats['min']:.3f} avg={contrast_stats['avg']:.3f} target={contrast_stats['target']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
