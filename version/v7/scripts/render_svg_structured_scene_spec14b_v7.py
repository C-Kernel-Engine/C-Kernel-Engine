#!/usr/bin/env python3
"""Render strict spec14b timeline scenes into SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec09_v7 import (
    _GAP_PX,
    _INSET_PX,
    _background_motif,
    _color_token,
    _defs,
    _palette,
    _panel_shell,
    _svg_text_block,
)
from render_svg_structured_scene_spec12_v7 import _payload_obj
from spec14b_scene_canonicalizer_v7 import canonicalize_scene_text


def _canvas_size(scene: dict[str, Any]) -> tuple[int, int]:
    _ = scene
    return (1600, 980)


def _component_entries(scene: dict[str, Any], name: str) -> list[dict[str, str]]:
    return list(scene["components_by_name"].get(name, []))


def _header_from_ref(scene: dict[str, Any], palette: dict[str, str], width: int, inset: float) -> tuple[str, float]:
    entries = _component_entries(scene, "header_band")
    ref = str(entries[0].get("ref") or "header") if entries else "header"
    header = _payload_obj(scene.get("_content"), ref)
    if not isinstance(header, dict):
        header = {}
    kicker = str(header.get("kicker") or "")
    title = str(header.get("headline") or scene.get("topic") or "Timeline")
    subtitle = str(header.get("subtitle") or "")
    accent = _color_token(palette, header.get("accent"), fallback=palette["accent"])
    body: list[str] = []
    body.append(
        f'<rect x="{inset:.1f}" y="{inset:.1f}" width="{width - 2 * inset:.1f}" height="118" rx="26" '
        f'fill="{palette["surface"]}" fill-opacity="0.16" stroke="{palette["border"]}" stroke-width="1.2"/>'
    )
    body.append(
        f'<rect x="{inset + 18:.1f}" y="{inset + 16:.1f}" width="250" height="9" rx="4.5" fill="{accent}" filter="url(#accentGlow)"/>'
    )
    if kicker:
        body.append(_svg_text_block(inset + 24, inset + 42, kicker, font_size=13, fill=accent, weight=700, max_chars=36))
    body.append(_svg_text_block(inset + 24, inset + 76, title, font_size=31, fill=palette["ink"], weight=700, max_chars=44))
    if subtitle:
        body.append(_svg_text_block(inset + 24, inset + 106, subtitle, font_size=15, fill=palette["muted"], max_chars=92))
    return "".join(body), inset + 150


def _lane_y(lane: str, *, body_top: float, spine_y: float, card_h: float) -> float:
    if lane == "top":
        return body_top
    if lane == "bottom":
        return spine_y + 54
    return spine_y - card_h / 2


def _stage_card(
    scene: dict[str, Any],
    palette: dict[str, str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    payload: dict[str, Any],
    accent: str,
) -> str:
    badge = str(payload.get("badge") or "")
    title = str(payload.get("title") or "")
    lines = payload.get("lines") if isinstance(payload.get("lines"), list) else []
    note = str(payload.get("note") or "")
    body: list[str] = []
    body.append(_panel_shell(x, y, width, height, scene, palette))
    body.append(f'<rect x="{x + 18:.1f}" y="{y + 18:.1f}" width="{width - 36:.1f}" height="11" rx="5.5" fill="{accent}" opacity="0.95"/>')
    if badge:
        pill_w = min(max(116.0, len(badge) * 7.8 + 24.0), width - 56.0)
        body.append(f'<rect x="{x + 22:.1f}" y="{y + 38:.1f}" width="{pill_w:.1f}" height="30" rx="15" fill="{accent}" fill-opacity="0.16" stroke="{accent}" stroke-width="1"/>')
        body.append(_svg_text_block(x + 22 + pill_w / 2, y + 58, badge, font_size=12, fill=accent, weight=700, max_chars=18, anchor="middle"))
    body.append(_svg_text_block(x + 24, y + 100, title, font_size=22, fill=palette["ink"], weight=700, max_chars=24))
    line_y = y + 136
    for line in lines[:4]:
        body.append(_svg_text_block(x + 24, line_y, str(line), font_size=14, fill=palette["muted"], max_chars=28))
        line_y += 24
    if note:
        body.append(_svg_text_block(x + 24, y + height - 20, note, font_size=11, fill=accent, weight=700, max_chars=28))
    return "".join(body)


def _footer_from_ref(scene: dict[str, Any], palette: dict[str, str], width: int, height: int, inset: float) -> str:
    entries = _component_entries(scene, "footer_note")
    if not entries:
        return ""
    ref = str(entries[0].get("ref") or "").strip()
    if not ref:
        return ""
    footer = _payload_obj(scene.get("_content"), ref)
    if isinstance(footer, dict):
        title = str(footer.get("title") or "")
        body = str(footer.get("body") or "")
    else:
        title = ""
        body = str(footer or "")
    if not title and not body:
        return ""
    box_y = height - inset - 76
    parts = [
        f'<rect x="{inset:.1f}" y="{box_y:.1f}" width="{width - 2 * inset:.1f}" height="64" rx="18" fill="{palette["surface"]}" fill-opacity="0.12" stroke="{palette["border"]}" stroke-width="1"/>'
    ]
    if title:
        parts.append(_svg_text_block(inset + 22, box_y + 24, title, font_size=13, fill=palette["accent"], weight=700, max_chars=34))
    if body:
        parts.append(_svg_text_block(inset + 22, box_y + 46, body, font_size=11, fill=palette["muted"], max_chars=128))
    return "".join(parts)


def _timeline(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX.get(str(scene["inset"]), 56)
    gap = _GAP_PX.get(str(scene["gap"]), 26)
    body: list[str] = []
    header, y = _header_from_ref(scene, palette, width, inset)
    body.append(header)

    stages = _component_entries(scene, "timeline_stage")
    arrows = _component_entries(scene, "timeline_arrow")
    count = len(stages)
    if count < 2:
        raise ValueError("spec14b timeline renderer requires at least two stages")

    card_w = min(272.0, max(212.0, (width - 2 * inset - gap * (count - 1)) / count))
    card_h = 228.0 if str(scene["density"]) != "compact" else 206.0
    body_top = y + 24
    spine_y = min(height - inset - 188.0, body_top + 264.0)
    center_start = inset + card_w / 2
    center_end = width - inset - card_w / 2
    step = (center_end - center_start) / max(1, count - 1)
    centers: dict[str, tuple[float, float]] = {}

    body.append(f'<line x1="{center_start:.1f}" y1="{spine_y:.1f}" x2="{center_end:.1f}" y2="{spine_y:.1f}" stroke="{palette["accent"]}" stroke-width="5" stroke-linecap="round" opacity="0.58"/>')

    for idx, entry in enumerate(stages):
        stage_id = str(entry.get("stage_id") or f"stage_{idx + 1}")
        ref = str(entry.get("ref") or "").strip()
        lane = str(entry.get("lane") or "center").strip()
        payload = _payload_obj(scene.get("_content"), ref)
        if not isinstance(payload, dict):
            payload = {}
        accent = _color_token(palette, payload.get("accent"), fallback=palette["accent"])
        cx = center_start + idx * step
        card_x = cx - card_w / 2
        card_y = _lane_y(lane, body_top=body_top, spine_y=spine_y, card_h=card_h)
        if lane == "top":
            body.append(f'<line x1="{cx:.1f}" y1="{card_y + card_h:.1f}" x2="{cx:.1f}" y2="{spine_y:.1f}" stroke="{palette["accent_2"]}" stroke-width="3"/>')
        elif lane == "bottom":
            body.append(f'<line x1="{cx:.1f}" y1="{spine_y:.1f}" x2="{cx:.1f}" y2="{card_y:.1f}" stroke="{palette["accent_2"]}" stroke-width="3"/>')
        else:
            body.append(f'<line x1="{cx:.1f}" y1="{card_y + card_h:.1f}" x2="{cx:.1f}" y2="{spine_y - 18:.1f}" stroke="{palette["accent_2"]}" stroke-width="3" opacity="0.75"/>')
        body.append(_stage_card(scene, palette, x=card_x, y=card_y, width=card_w, height=card_h, payload=payload, accent=accent))
        body.append(f'<circle cx="{cx:.1f}" cy="{spine_y:.1f}" r="12" fill="{accent}" stroke="{palette["bg"]}" stroke-width="3"/>')
        centers[stage_id] = (cx, spine_y)

    for arrow in arrows:
        src = str(arrow.get("from_stage") or "").strip()
        dst = str(arrow.get("to_stage") or "").strip()
        if src not in centers or dst not in centers:
            continue
        x1, y1 = centers[src]
        x2, y2 = centers[dst]
        body.append(
            f'<line x1="{x1 + 18:.1f}" y1="{y1:.1f}" x2="{x2 - 22:.1f}" y2="{y2:.1f}" '
            f'stroke="{palette["accent"]}" stroke-width="3" marker-end="url(#arrowHead)"/>'
        )

    footer = _footer_from_ref(scene, palette, width, height, inset)
    if footer:
        body.append(footer)
    return "".join(body)


def render_structured_scene_spec14b_svg(text: str, content: dict[str, Any] | None = None) -> str:
    scene_doc = canonicalize_scene_text(text)
    scene = scene_doc.to_runtime()
    scene["_content"] = content or {}
    width, height = _canvas_size(scene)
    palette = _palette(scene)
    body = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#bgGrad)"/>',
        _background_motif(scene, palette, width, height),
        _timeline(scene, palette, width, height),
    ]
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f"{_defs(scene, palette)}"
        f'{"".join(body)}'
        "</svg>"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Render strict spec14b timeline scene DSL to SVG.")
    ap.add_argument("--scene", default=None, help="Inline scene document.")
    ap.add_argument("--scene-file", default=None, help="Path to a compact scene document file.")
    ap.add_argument("--content-json", default=None, help="Optional content JSON payload path.")
    ap.add_argument("--out", default=None, help="Optional output SVG path.")
    args = ap.parse_args()

    if bool(args.scene) == bool(args.scene_file):
        raise SystemExit("ERROR: pass exactly one of --scene or --scene-file")
    text = args.scene if args.scene is not None else Path(args.scene_file).read_text(encoding="utf-8")
    content = None
    if args.content_json:
        content = json.loads(Path(args.content_json).read_text(encoding="utf-8"))
    svg = render_structured_scene_spec14b_svg(text, content=content)
    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(svg, encoding="utf-8")
        print(f"[OK] wrote: {out}")
    else:
        print(svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
