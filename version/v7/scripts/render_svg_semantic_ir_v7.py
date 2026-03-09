#!/usr/bin/env python3
"""Render semantic SVG IR tokens to concrete SVG XML."""

from __future__ import annotations

import argparse
from pathlib import Path


PALETTE_FILL = {
    "[fill:warm]": "#ef4444",
    "[fill:cool]": "#2563eb",
    "[fill:mono]": "#475569",
    "[fill:signal]": "#f59e0b",
}

THEME_BG = {
    "[bg:light]": "#f8fafc",
    "[bg:dark]": "#0f172a",
}

THEME_FG = {
    "[fg:light]": "#f8fafc",
    "[fg:dark]": "#0f172a",
}


def _split_tokens(text: str) -> list[str]:
    return [tok.strip() for tok in text.split() if tok.strip()]


def _render_shape(tokens: list[str]) -> str | None:
    fill = next((PALETTE_FILL[t] for t in tokens if t in PALETTE_FILL), "#2563eb")
    stroke = "#0f172a"
    shape_map = {
        "[circle:xs]": '<circle cx="64" cy="64" r="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[circle:sm]": '<circle cx="64" cy="64" r="18" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[circle:md]": '<circle cx="64" cy="64" r="26" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[circle:lg]": '<circle cx="64" cy="64" r="36" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[circle:xl]": '<circle cx="64" cy="64" r="46" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[rect:xs]": '<rect x="51" y="55" width="26" height="18" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[rect:sm]": '<rect x="44" y="50" width="40" height="28" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[rect:md]": '<rect x="36" y="45" width="56" height="38" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[rect:lg]": '<rect x="26" y="35" width="76" height="58" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[rect:xl]": '<rect x="18" y="29" width="92" height="70" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[triangle:xs]": '<polygon points="64,48 48,78 80,78" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[triangle:sm]": '<polygon points="64,36 40,82 88,82" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[triangle:md]": '<polygon points="64,28 34,88 94,88" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[triangle:lg]": '<polygon points="64,20 22,98 106,98" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        "[triangle:xl]": '<polygon points="64,14 16,106 112,106" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
    }
    shape_token = next((tok for tok in tokens if tok in shape_map), None)
    if shape_token is None:
        return None
    body = shape_map[shape_token].format(fill=fill, stroke=stroke)
    return f'<svg width="128" height="128" xmlns="http://www.w3.org/2000/svg">{body}</svg>'


def _render_card(tokens: list[str]) -> str | None:
    card_token = next((tok for tok in tokens if tok in {"[card:sm]", "[card:md]", "[card:lg]"}), None)
    if card_token is None:
        return None
    bg = THEME_BG.get(next((tok for tok in tokens if tok in THEME_BG), "[bg:light]"))
    fg = THEME_FG.get(next((tok for tok in tokens if tok in THEME_FG), "[fg:dark]"))
    accent = {
        "[accent:warm]": "#ef4444",
        "[accent:cool]": "#2563eb",
        "[accent:mono]": "#475569",
    }.get(next((tok for tok in tokens if tok.startswith("[accent:")), "[accent:cool]"), "#2563eb")
    dims = {
        "[card:sm]": ("20", "26", "88", "76"),
        "[card:md]": ("14", "20", "100", "88"),
        "[card:lg]": ("10", "14", "108", "100"),
    }[card_token]
    x, y, w, h = dims
    bullets = 3 if "[bullet_slot:3]" in tokens else 2 if "[bullet_slot:2]" in tokens else 0
    title = '[title_slot]' in tokens
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{bg}" stroke="{accent}" stroke-width="2"/>',
        f'<rect x="{x}" y="{y}" width="{w}" height="8" rx="12" fill="{accent}"/>',
    ]
    if title:
        parts.append(f'<rect x="{int(x)+12}" y="{int(y)+18}" width="{int(w)-24}" height="8" rx="4" fill="{fg}" opacity="0.9"/>')
    for i in range(bullets):
        yy = int(y) + 34 + i * 14
        parts.append(f'<circle cx="{int(x)+14}" cy="{yy+3}" r="3" fill="{fg}"/>')
        parts.append(f'<rect x="{int(x)+24}" y="{yy}" width="{int(w)-36}" height="6" rx="3" fill="{fg}" opacity="0.85"/>')
    return f'<svg width="128" height="128" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def _render_chart(tokens: list[str]) -> str | None:
    chart_token = next((tok for tok in tokens if tok in {"[chart:bar]", "[chart:line]"}), None)
    if chart_token is None:
        return None
    bg = "#f8fafc"
    accent = "#2563eb"
    parts = [f'<rect x="8" y="8" width="112" height="112" rx="10" fill="{bg}" stroke="#cbd5e1" stroke-width="2"/>']
    if chart_token == "[chart:bar]":
        heights = {
            "[bars3:up]": (22, 42, 68),
            "[bars3:down]": (68, 42, 22),
            "[bars3:flat]": (40, 40, 40),
        }
        bars_token = next((tok for tok in tokens if tok in heights), "[bars3:up]")
        hs = heights[bars_token]
        xs = (24, 52, 80)
        for x, h in zip(xs, hs):
            y = 104 - h
            parts.append(f'<rect x="{x}" y="{y}" width="16" height="{h}" rx="4" fill="{accent}"/>')
    else:
        curve = {
            "[bars3:up]": "20,84 52,62 92,34",
            "[bars3:down]": "20,34 52,62 92,84",
            "[bars3:flat]": "20,58 52,58 92,58",
        }
        bars_token = next((tok for tok in tokens if tok in curve), "[bars3:up]")
        parts.append(f'<polyline points="{curve[bars_token]}" fill="none" stroke="{accent}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
    return f'<svg width="128" height="128" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def _render_curve(tokens: list[str]) -> str | None:
    if "[plot:curve]" not in tokens:
        return None
    stroke = "#7c3aed"
    d = {
        "[curve:linear-up]": "M18 96 L110 24",
        "[curve:quad-up]": "M18 96 Q64 18 110 36",
        "[curve:quad-down]": "M18 30 Q64 108 110 34",
        "[curve:s-curve]": "M18 86 C34 12 92 110 110 38",
    }
    curve_token = next((tok for tok in tokens if tok in d), "[curve:linear-up]")
    parts = [
        '<rect x="8" y="8" width="112" height="112" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>',
        '<path d="M18 106 L18 18 M18 106 L110 106" fill="none" stroke="#94a3b8" stroke-width="2"/>',
        f'<path d="{d[curve_token]}" fill="none" stroke="{stroke}" stroke-width="4" stroke-linecap="round"/>',
    ]
    return f'<svg width="128" height="128" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def render_ir(text: str) -> str:
    tokens = _split_tokens(text)
    for fn in (_render_shape, _render_card, _render_chart, _render_curve):
        svg = fn(tokens)
        if svg is not None:
            return svg
    raise SystemExit("unsupported IR tokens")


def main() -> int:
    ap = argparse.ArgumentParser(description="Render semantic SVG IR to SVG XML")
    ap.add_argument("--ir", default=None, help="Inline IR token text")
    ap.add_argument("--input", default=None, help="File containing IR token text")
    ap.add_argument("--output", default=None, help="Optional SVG output file")
    args = ap.parse_args()

    if bool(args.ir) == bool(args.input):
        raise SystemExit("provide exactly one of --ir or --input")

    if args.ir:
        text = str(args.ir)
    else:
        text = Path(args.input).expanduser().resolve().read_text(encoding="utf-8")

    svg = render_ir(text)
    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(svg + "\n", encoding="utf-8")
    else:
        print(svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
