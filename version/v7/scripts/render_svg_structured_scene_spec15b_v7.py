#!/usr/bin/env python3
"""Render strict spec15b system_diagram scenes into SVG."""

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
from spec15b_scene_canonicalizer_v7 import canonicalize_scene_text


def _canvas_size(scene: dict[str, Any]) -> tuple[int, int]:
    density = str(scene.get("density") or "balanced")
    if density == "compact":
        return (1760, 860)
    if density == "airy":
        return (1820, 940)
    return (1780, 900)


def _component_entries(scene: dict[str, Any], name: str) -> list[dict[str, str]]:
    return list(scene["components_by_name"].get(name, []))


def _header_from_ref(scene: dict[str, Any], palette: dict[str, str], width: int, inset: float) -> tuple[str, float]:
    entries = _component_entries(scene, "header_band")
    ref = str(entries[0].get("ref") or "header") if entries else "header"
    header = _payload_obj(scene.get("_content"), ref)
    if not isinstance(header, dict):
        header = {}
    kicker = str(header.get("kicker") or "")
    title = str(header.get("headline") or scene.get("topic") or "System Diagram")
    subtitle = str(header.get("subtitle") or "")
    accent = _color_token(palette, header.get("accent"), fallback=palette["accent"])
    body: list[str] = []
    body.append(
        f'<rect x="{inset:.1f}" y="{inset:.1f}" width="{width - 2 * inset:.1f}" height="124" rx="26" '
        f'fill="{palette["surface"]}" fill-opacity="0.16" stroke="{palette["border"]}" stroke-width="1.2"/>'
    )
    body.append(
        f'<rect x="{inset + 18:.1f}" y="{inset + 18:.1f}" width="280" height="10" rx="5" fill="{accent}" filter="url(#accentGlow)"/>'
    )
    if kicker:
        body.append(_svg_text_block(inset + 24, inset + 46, kicker, font_size=13, fill=accent, weight=700, max_chars=40))
    body.append(_svg_text_block(inset + 24, inset + 82, title, font_size=31, fill=palette["ink"], weight=700, max_chars=46))
    if subtitle:
        body.append(_svg_text_block(inset + 24, inset + 112, subtitle, font_size=15, fill=palette["muted"], max_chars=96))
    return "".join(body), inset + 158


def _detail_list(payload: dict[str, Any], *, limit: int) -> list[str]:
    raw = payload.get("detail")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw[:limit] if str(item).strip()]


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
    body_text = str(payload.get("body") or "")
    details = _detail_list(payload, limit=2)
    body: list[str] = []
    body.append(_panel_shell(x, y, width, height, scene, palette))
    body.append(f'<rect x="{x + 18:.1f}" y="{y + 16:.1f}" width="{width - 36:.1f}" height="9" rx="4.5" fill="{accent}" opacity="0.95"/>')
    if badge:
        pill_w = min(max(112.0, len(badge) * 7.8 + 20.0), width - 56.0)
        body.append(f'<rect x="{x + 22:.1f}" y="{y + 34:.1f}" width="{pill_w:.1f}" height="28" rx="14" fill="{accent}" fill-opacity="0.14" stroke="{accent}" stroke-width="1"/>')
        body.append(_svg_text_block(x + 22 + pill_w / 2, y + 53, badge, font_size=11, fill=accent, weight=700, max_chars=18, anchor="middle"))
    body.append(_svg_text_block(x + 22, y + 88, title, font_size=22, fill=palette["ink"], weight=700, max_chars=22))
    body.append(_svg_text_block(x + 22, y + 116, body_text, font_size=12, fill=palette["muted"], max_chars=34))
    cy = y + 148
    for detail in details:
        body.append(_svg_text_block(x + 26, cy, detail, font_size=11, fill=accent, weight=700, max_chars=22))
        cy += 18
    return "".join(body)


def _terminal_card(
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
    body_text = str(payload.get("body") or "")
    body: list[str] = []
    body.append(_panel_shell(x, y, width, height, scene, palette))
    body.append(f'<rect x="{x + 16:.1f}" y="{y + 16:.1f}" width="{width - 32:.1f}" height="12" rx="6" fill="{accent}" opacity="0.95"/>')
    if badge:
        body.append(f'<rect x="{x + 20:.1f}" y="{y + 40:.1f}" width="112" height="30" rx="15" fill="{accent}" fill-opacity="0.14" stroke="{accent}" stroke-width="1"/>')
        body.append(_svg_text_block(x + 76, y + 60, badge, font_size=12, fill=accent, weight=700, max_chars=16, anchor="middle"))
    body.append(_svg_text_block(x + 22, y + 100, title, font_size=24, fill=palette["ink"], weight=700, max_chars=22))
    body.append(_svg_text_block(x + 22, y + 132, body_text, font_size=12, fill=palette["muted"], max_chars=32))
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
        body = str(footer.get("body") or footer.get("note") or "")
    else:
        title = ""
        body = str(footer or "")
    if not title and not body:
        return ""
    box_y = height - inset - 74
    parts = [
        f'<rect x="{inset:.1f}" y="{box_y:.1f}" width="{width - 2 * inset:.1f}" height="60" rx="18" fill="{palette["surface"]}" fill-opacity="0.12" stroke="{palette["border"]}" stroke-width="1"/>'
    ]
    if title:
        parts.append(_svg_text_block(inset + 22, box_y + 23, title, font_size=13, fill=palette["accent"], weight=700, max_chars=34))
    if body:
        parts.append(_svg_text_block(inset + 22, box_y + 44, body, font_size=11, fill=palette["muted"], max_chars=134))
    return "".join(parts)


def _link_label(scene: dict[str, Any], ref: str) -> str:
    payload = _payload_obj(scene.get("_content"), ref)
    if isinstance(payload, dict):
        return str(payload.get("label") or payload.get("title") or "").strip()
    return str(payload or "").strip()


def _system_diagram(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX.get(str(scene["inset"]), 56)
    gap = max(40.0, float(_GAP_PX.get(str(scene["gap"]), 26)) * 2.2)
    body: list[str] = []
    header, y = _header_from_ref(scene, palette, width, inset)
    body.append(header)

    stages = _component_entries(scene, "system_stage")
    links = _component_entries(scene, "system_link")
    terminals = _component_entries(scene, "terminal_panel")
    if not stages or not terminals:
        raise ValueError("spec15b renderer requires stages and one terminal panel")

    stage_count = len(stages)
    stage_w = 248.0
    stage_h = 184.0 if str(scene["density"]) != "compact" else 170.0
    terminal_w = 232.0
    terminal_h = 216.0 if str(scene["density"]) != "compact" else 198.0
    available_w = width - 2 * inset
    total_w = stage_count * stage_w + terminal_w + stage_count * gap
    gap_w = gap if total_w <= available_w else max(28.0, (available_w - stage_count * stage_w - terminal_w) / max(1, stage_count))
    start_x = inset + max(0.0, (available_w - (stage_count * stage_w + terminal_w + stage_count * gap_w)) / 2)
    stage_y = y + 118
    term_y = stage_y - 12
    line_y = stage_y + stage_h / 2

    centers: dict[str, tuple[float, float, float]] = {}
    stage_ids = {str(entry.get("stage_id") or "").strip() for entry in stages}

    for idx, entry in enumerate(stages):
        stage_id = str(entry.get("stage_id") or f"stage_{idx + 1}")
        ref = str(entry.get("ref") or "").strip()
        payload = _payload_obj(scene.get("_content"), ref)
        if not isinstance(payload, dict):
            payload = {}
        accent = _color_token(palette, payload.get("accent"), fallback=palette["accent"])
        x = start_x + idx * (stage_w + gap_w)
        body.append(_stage_card(scene, palette, x=x, y=stage_y, width=stage_w, height=stage_h, payload=payload, accent=accent))
        centers[stage_id] = (x + stage_w, x + stage_w / 2, line_y)
        body.append(f'<circle cx="{x + stage_w / 2:.1f}" cy="{line_y:.1f}" r="10" fill="{accent}" stroke="{palette["bg"]}" stroke-width="3"/>')

    terminal_entry = terminals[0]
    terminal_id = str(terminal_entry.get("panel_id") or "terminal").strip()
    terminal_ref = str(terminal_entry.get("ref") or "").strip()
    terminal_payload = _payload_obj(scene.get("_content"), terminal_ref)
    if not isinstance(terminal_payload, dict):
        terminal_payload = {}
    terminal_accent = _color_token(palette, terminal_payload.get("accent"), fallback=palette["accent_2"])
    term_x = start_x + stage_count * (stage_w + gap_w)
    body.append(_terminal_card(scene, palette, x=term_x, y=term_y, width=terminal_w, height=terminal_h, payload=terminal_payload, accent=terminal_accent))
    centers[terminal_id] = (term_x, term_x + terminal_w / 2, term_y + terminal_h / 2)
    body.append(f'<circle cx="{term_x + terminal_w / 2:.1f}" cy="{term_y + terminal_h / 2:.1f}" r="10" fill="{terminal_accent}" stroke="{palette["bg"]}" stroke-width="3"/>')

    connector = str(scene.get("connector") or "arrow")
    marker = ' marker-end="url(#arrowHead)"' if connector == "arrow" else ""
    for link in links:
        src = str(link.get("from_stage") or "").strip()
        dst = str(link.get("to_stage") or "").strip()
        ref = str(link.get("ref") or "").strip()
        if src not in stage_ids or dst not in centers:
            continue
        src_right, _, src_y = centers[src]
        dst_left, _, dst_y = centers[dst]
        mid_x = (src_right + dst_left) / 2
        body.append(
            f'<path d="M {src_right + 8:.1f} {src_y:.1f} C {mid_x:.1f} {src_y:.1f}, {mid_x:.1f} {dst_y:.1f}, {dst_left - 14:.1f} {dst_y:.1f}" '
            f'stroke="{palette["accent"]}" stroke-width="3" fill="none"{marker}/>'
        )
        label = _link_label(scene, ref)
        if label:
            body.append(_svg_text_block(mid_x, min(src_y, dst_y) - 12, label, font_size=11, fill=palette["accent"], weight=700, max_chars=18, anchor="middle"))

    footer = _footer_from_ref(scene, palette, width, height, inset)
    if footer:
        body.append(footer)
    return "".join(body)


def render_structured_scene_spec15b_svg(text: str, content: dict[str, Any] | None = None) -> str:
    scene_doc = canonicalize_scene_text(text)
    scene = scene_doc.to_runtime()
    scene["_content"] = content or {}
    width, height = _canvas_size(scene)
    palette = _palette(scene)
    body = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#bgGrad)"/>',
        _background_motif(scene, palette, width, height),
        _system_diagram(scene, palette, width, height),
    ]
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f"{_defs(scene, palette)}"
        f'{"".join(body)}'
        "</svg>"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Render strict spec15b system_diagram scene DSL to SVG.")
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
    svg = render_structured_scene_spec15b_svg(text, content=content)
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

