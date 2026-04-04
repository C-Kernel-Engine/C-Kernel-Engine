#!/usr/bin/env python3
"""Render spec14a scene DSL with the comparison-board successor family."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec09_v7 import (
    _GAP_PX,
    _INSET_PX,
    _background_motif,
    _color_token,
    _defs as _spec09_defs,
    _palette,
    _panel_shell,
    _svg_text_block,
)
from render_svg_structured_scene_spec12_v7 import (
    _header_from_ref,
    _memory_map,
    _payload_obj,
    _table_matrix,
)
from render_svg_structured_scene_spec13b_v7 import (
    _decision_tree_generalized,
    _flow_graph_generalized,
)
from spec14a_scene_canonicalizer_v7 import canonicalize_scene_text


def _canvas_size(scene: dict[str, Any]) -> tuple[int, int]:
    layout = str(scene.get("layout") or "")
    if layout == "comparison_board":
        if scene.get("canvas") == "tall":
            return (1480, 1660)
        if scene.get("canvas") == "square":
            return (1480, 1480)
        return (1720, 1320)
    if layout == "decision_tree":
        return (1780, 1320)
    if layout == "flow_graph":
        return (1760, 920)
    if layout == "table_matrix":
        return (1400, 1880) if scene["canvas"] == "tall" else (1400, 1560)
    if layout == "memory_map":
        return (1280, 980)
    raise ValueError(f"unsupported spec14a scene layout: {layout}")


def _defs(scene: dict[str, Any], palette: dict[str, str]) -> str:
    proxy = dict(scene)
    proxy["connector"] = str(scene.get("connector") or "arrow")
    return _spec09_defs(proxy, palette)


def _component_entries(scene: dict[str, Any], name: str) -> list[dict[str, str]]:
    return list(scene["components_by_name"].get(name, []))


def _chip_width(text: str) -> float:
    return max(92.0, min(186.0, 26.0 + len(str(text or "")) * 7.2))


def _comparison_column_height(column: dict[str, Any]) -> float:
    chips = column.get("chips") if isinstance(column.get("chips"), list) else []
    bullets = column.get("bullets") if isinstance(column.get("bullets"), list) else []
    note = str(column.get("note") or "")
    subtitle = str(column.get("subtitle") or "")
    base = 222.0
    chip_rows = max(0, math.ceil(min(6, len(chips)) / 2))
    bullet_rows = min(5, len(bullets))
    extra = chip_rows * 38.0 + bullet_rows * 24.0
    if subtitle:
        extra += 24.0
    if note:
        extra += 28.0
    return max(292.0, base + extra)


def _comparison_metric_height(metric: dict[str, Any]) -> float:
    detail = str(metric.get("detail") or "")
    note = str(metric.get("note") or "")
    extra = (18.0 if detail else 0.0) + (18.0 if note else 0.0)
    return 142.0 + extra


def _comparison_callout_height(callout: dict[str, Any]) -> float:
    lines = callout.get("lines") if isinstance(callout.get("lines"), list) else []
    body = str(callout.get("body") or "")
    extra = min(4, len(lines)) * 22.0 + (22.0 if body else 0.0)
    return max(188.0, 152.0 + extra)


def _draw_comparison_column(
    x: float,
    y: float,
    width: float,
    height: float,
    column: dict[str, Any],
    scene: dict[str, Any],
    palette: dict[str, str],
) -> str:
    accent = _color_token(palette, str(column.get("accent") or ""), fallback=palette["accent"])
    title = str(column.get("title") or "Column")
    subtitle = str(column.get("subtitle") or "")
    kicker = str(column.get("kicker") or "")
    stat = str(column.get("stat") or column.get("score") or "")
    chips = column.get("chips") if isinstance(column.get("chips"), list) else []
    bullets = column.get("bullets") if isinstance(column.get("bullets"), list) else []
    note = str(column.get("note") or "")

    parts = ['<g class="comparison-column">']
    parts.append(_panel_shell(x, y, width, height, scene, palette))
    parts.append(
        f'<rect x="{x + 18:.1f}" y="{y + 18:.1f}" width="{width - 36:.1f}" height="8" rx="4" '
        f'fill="{accent}" fill-opacity="0.92"/>'
    )
    if kicker:
        pill_w = max(104.0, min(width * 0.44, 42.0 + len(kicker) * 7.6))
        parts.append(
            f'<rect x="{x + 22:.1f}" y="{y + 34:.1f}" width="{pill_w:.1f}" height="28" rx="14" '
            f'fill="{accent}" fill-opacity="0.12" stroke="{accent}" stroke-width="1"/>'
        )
        parts.append(_svg_text_block(x + 22 + pill_w / 2, y + 52, kicker, font_size=11, fill=accent, weight=700, max_chars=22, anchor="middle"))
    if stat:
        stat_w = max(118.0, min(width * 0.34, 42.0 + len(stat) * 7.6))
        stat_x = x + width - 22 - stat_w
        parts.append(
            f'<rect x="{stat_x:.1f}" y="{y + 34:.1f}" width="{stat_w:.1f}" height="28" rx="14" '
            f'fill="{accent}" fill-opacity="0.10" stroke="{accent}" stroke-width="1"/>'
        )
        parts.append(_svg_text_block(stat_x + stat_w / 2, y + 52, stat, font_size=11, fill=accent, weight=700, max_chars=22, anchor="middle"))
    parts.append(_svg_text_block(x + 24, y + 92, title, font_size=22, fill=palette["ink"], weight=700, max_chars=28))
    if subtitle:
        parts.append(_svg_text_block(x + 24, y + 118, subtitle, font_size=12, fill=palette["muted"], max_chars=52))

    chip_y = y + 148
    chip_gap = 10.0
    chip_x = x + 24
    row_height = 32.0
    for idx, chip in enumerate(chips[:6]):
        label = str(chip)
        chip_w = _chip_width(label)
        if chip_x + chip_w > x + width - 24:
            chip_x = x + 24
            chip_y += row_height + 8.0
        parts.append(
            f'<rect x="{chip_x:.1f}" y="{chip_y:.1f}" width="{chip_w:.1f}" height="{row_height:.1f}" rx="16" '
            f'fill="{accent}" fill-opacity="0.10" stroke="{accent}" stroke-width="1"/>'
        )
        parts.append(_svg_text_block(chip_x + chip_w / 2, chip_y + 20, label, font_size=11, fill=accent, weight=700, max_chars=24, anchor="middle"))
        chip_x += chip_w + chip_gap

    bullet_y = chip_y + (row_height + 18.0 if chips else 8.0)
    for bullet in bullets[:5]:
        parts.append(f'<circle cx="{x + 30:.1f}" cy="{bullet_y + 5:.1f}" r="4" fill="{accent}" opacity="0.9"/>')
        parts.append(_svg_text_block(x + 44, bullet_y + 10, str(bullet), font_size=12, fill=palette["ink"], max_chars=48))
        bullet_y += 24.0

    if note:
        parts.append(f'<line x1="{x + 24:.1f}" y1="{height + y - 48:.1f}" x2="{x + width - 24:.1f}" y2="{height + y - 48:.1f}" stroke="{palette["border"]}" stroke-width="1"/>')
        parts.append(_svg_text_block(x + 24, y + height - 22, note, font_size=11, fill=palette["muted"], max_chars=56))
    parts.append("</g>")
    return "".join(parts)


def _draw_metric_card(
    x: float,
    y: float,
    width: float,
    height: float,
    metric: dict[str, Any],
    scene: dict[str, Any],
    palette: dict[str, str],
) -> str:
    state = str(metric.get("state") or metric.get("accent") or "")
    accent = _color_token(palette, state, fallback=palette["accent"])
    label = str(metric.get("label") or "Metric")
    value = str(metric.get("value") or "")
    detail = str(metric.get("detail") or "")
    note = str(metric.get("note") or "")
    parts = ['<g class="comparison-metric">']
    parts.append(_panel_shell(x, y, width, height, scene, palette))
    parts.append(
        f'<rect x="{x + 18:.1f}" y="{y + 18:.1f}" width="112" height="24" rx="12" '
        f'fill="{accent}" fill-opacity="0.10" stroke="{accent}" stroke-width="1"/>'
    )
    parts.append(_svg_text_block(x + 74, y + 34, label, font_size=10, fill=accent, weight=700, max_chars=18, anchor="middle"))
    parts.append(_svg_text_block(x + 22, y + 74, value or "n/a", font_size=26, fill=palette["ink"], weight=700, max_chars=20))
    if detail:
        parts.append(_svg_text_block(x + 22, y + 102, detail, font_size=12, fill=palette["muted"], max_chars=32))
    if note:
        parts.append(_svg_text_block(x + 22, y + height - 20, note, font_size=11, fill=palette["muted"], max_chars=34))
    parts.append("</g>")
    return "".join(parts)


def _draw_callout_card(
    x: float,
    y: float,
    width: float,
    height: float,
    callout: dict[str, Any],
    scene: dict[str, Any],
    palette: dict[str, str],
) -> str:
    state = str(callout.get("state") or callout.get("accent") or "")
    accent = _color_token(palette, state, fallback=palette["accent_2"])
    badge = str(callout.get("badge") or "Callout")
    title = str(callout.get("title") or "")
    body = str(callout.get("body") or "")
    lines = callout.get("lines") if isinstance(callout.get("lines"), list) else []

    parts = ['<g class="comparison-callout">']
    parts.append(_panel_shell(x, y, width, height, scene, palette))
    badge_w = max(104.0, min(width * 0.42, 42.0 + len(badge) * 7.5))
    parts.append(
        f'<rect x="{x + 20:.1f}" y="{y + 18:.1f}" width="{badge_w:.1f}" height="28" rx="14" '
        f'fill="{accent}" fill-opacity="0.10" stroke="{accent}" stroke-width="1"/>'
    )
    parts.append(_svg_text_block(x + 20 + badge_w / 2, y + 36, badge, font_size=11, fill=accent, weight=700, max_chars=20, anchor="middle"))
    parts.append(_svg_text_block(x + 22, y + 82, title, font_size=20, fill=palette["ink"], weight=700, max_chars=26))
    if body:
        parts.append(_svg_text_block(x + 22, y + 108, body, font_size=12, fill=palette["muted"], max_chars=48))
    line_y = y + 138
    for line in lines[:4]:
        parts.append(f'<rect x="{x + 22:.1f}" y="{line_y - 14:.1f}" width="{width - 44:.1f}" height="26" rx="13" fill="{accent}" fill-opacity="0.06" stroke="{accent}" stroke-width="1"/>')
        parts.append(_svg_text_block(x + 36, line_y + 4, str(line), font_size=11, fill=palette["ink"], weight=600, max_chars=42))
        line_y += 30.0
    parts.append("</g>")
    return "".join(parts)


def _comparison_board(scene: dict[str, Any], palette: dict[str, str]) -> tuple[str, int, int]:
    inset = _INSET_PX.get(str(scene["inset"]), 56)
    gap = float(_GAP_PX.get(str(scene["gap"]), 26))
    width, _ = _canvas_size(scene)
    content = scene.get("_content") or {}
    body: list[str] = []

    header, y = _header_from_ref(scene, palette, width, inset)
    body.append(header)

    legend_entries = _component_entries(scene, "legend_block")
    if legend_entries:
        legend_ref = str(legend_entries[0].get("ref") or "")
        legend = _payload_obj(content, legend_ref)
        if isinstance(legend, dict):
            legend_h = 94.0
            legend_parts: list[str] = []
            legend_parts.append(_panel_shell(inset, y, width - 2 * inset, legend_h, scene, palette))
            legend_parts.append(_svg_text_block(inset + 24, y + 28, str(legend.get("title") or "Legend"), font_size=15, fill=palette["ink"], weight=700, max_chars=24))
            chip_x = inset + 24
            chip_y = y + 42
            for item in (legend.get("items") if isinstance(legend.get("items"), list) else [])[:6]:
                label = str(item)
                chip_w = _chip_width(label)
                if chip_x + chip_w > width - inset - 24:
                    chip_x = inset + 24
                    chip_y += 30.0
                legend_parts.append(
                    f'<rect x="{chip_x:.1f}" y="{chip_y:.1f}" width="{chip_w:.1f}" height="24" rx="12" '
                    f'fill="{palette["accent"]}" fill-opacity="0.10" stroke="{palette["accent"]}" stroke-width="1"/>'
                )
                legend_parts.append(_svg_text_block(chip_x + chip_w / 2, chip_y + 16, label, font_size=10, fill=palette["accent"], weight=700, max_chars=22, anchor="middle"))
                chip_x += chip_w + 8.0
            note = str(legend.get("note") or "")
            if note:
                legend_parts.append(_svg_text_block(inset + 24, y + legend_h - 16, note, font_size=11, fill=palette["muted"], max_chars=120))
            body.append(f'<g class="comparison-legend">{"".join(legend_parts)}</g>')
            y += legend_h + gap

    column_entries = _component_entries(scene, "comparison_column")
    metric_entries = _component_entries(scene, "comparison_metric")
    callout_entries = _component_entries(scene, "comparison_callout")
    requested_columns = max(1, min(3, int(str(scene.get("columns") or len(column_entries) or "3"))))
    column_count = min(requested_columns, max(1, len(column_entries))) if column_entries else requested_columns
    available_w = width - 2 * inset
    column_w = (available_w - gap * max(0, column_count - 1)) / max(1, column_count)

    column_payloads: list[dict[str, Any]] = []
    for entry in column_entries[:column_count]:
        payload = _payload_obj(content, str(entry.get("ref") or ""))
        if isinstance(payload, dict):
            column_payloads.append(payload)
    if column_payloads:
        column_h = max(_comparison_column_height(payload) for payload in column_payloads)
        for idx, payload in enumerate(column_payloads):
            x = inset + idx * (column_w + gap)
            body.append(_draw_comparison_column(x, y, column_w, column_h, payload, scene, palette))
        y += column_h + gap

    metric_payloads: list[dict[str, Any]] = []
    for entry in metric_entries:
        payload = _payload_obj(content, str(entry.get("ref") or ""))
        if isinstance(payload, dict):
            metric_payloads.append(payload)
    if metric_payloads:
        metric_cols = 4 if len(metric_payloads) >= 4 else 3 if len(metric_payloads) == 3 else 2
        metric_w = (available_w - gap * max(0, metric_cols - 1)) / metric_cols
        row_y = y
        row_h = 0.0
        for idx, payload in enumerate(metric_payloads):
            row = idx // metric_cols
            col = idx % metric_cols
            if col == 0 and idx:
                row_y += row_h + gap
                row_h = 0.0
            card_h = _comparison_metric_height(payload)
            row_h = max(row_h, card_h)
            x = inset + col * (metric_w + gap)
            body.append(_draw_metric_card(x, row_y, metric_w, card_h, payload, scene, palette))
        y = row_y + row_h + gap

    callout_payloads: list[dict[str, Any]] = []
    for entry in callout_entries:
        payload = _payload_obj(content, str(entry.get("ref") or ""))
        if isinstance(payload, dict):
            callout_payloads.append(payload)
    if callout_payloads:
        callout_cols = 3 if len(callout_payloads) >= 3 else 2
        callout_w = (available_w - gap * max(0, callout_cols - 1)) / callout_cols
        row_y = y
        row_h = 0.0
        for idx, payload in enumerate(callout_payloads):
            col = idx % callout_cols
            if col == 0 and idx:
                row_y += row_h + gap
                row_h = 0.0
            card_h = _comparison_callout_height(payload)
            row_h = max(row_h, card_h)
            x = inset + col * (callout_w + gap)
            body.append(_draw_callout_card(x, row_y, callout_w, card_h, payload, scene, palette))
        y = row_y + row_h + gap

    footer_entries = _component_entries(scene, "footer_note")
    footer_ref = str(footer_entries[0].get("ref") or "footer") if footer_entries else "footer"
    footer = _payload_obj(content, footer_ref)
    note = str(footer.get("note") if isinstance(footer, dict) else footer or "")
    if note:
        footer_h = 56.0
        body.append(
            f'<rect x="{inset:.1f}" y="{y:.1f}" width="{width - 2 * inset:.1f}" height="{footer_h:.1f}" rx="16" '
            f'fill="{palette["surface_2"]}" fill-opacity="0.92" stroke="{palette["border"]}" stroke-width="1"/>'
        )
        body.append(_svg_text_block(inset + 24, y + 34, note, font_size=12, fill=palette["muted"], max_chars=152))
        y += footer_h + gap

    height = max(_canvas_size(scene)[1], int(y + inset))
    return "".join(body), width, height


def render_structured_scene_spec14a_svg(text: str, content: dict[str, Any] | None = None) -> str:
    scene = canonicalize_scene_text(text).to_runtime()
    scene["_content"] = content or {}
    palette = _palette(scene)
    layout = str(scene["layout"])
    if layout == "comparison_board":
        body, width, height = _comparison_board(scene, palette)
    elif layout == "decision_tree":
        body, width, height = _decision_tree_generalized(scene, palette)
    elif layout == "flow_graph":
        body, width, height = _flow_graph_generalized(scene, palette)
    else:
        width, height = _canvas_size(scene)
        if layout == "table_matrix":
            body = _table_matrix(scene, palette, width, height)
        elif layout == "memory_map":
            body = _memory_map(scene, palette, width, height)
        else:
            raise ValueError(f"unsupported spec14a scene layout: {layout}")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f"{_defs(scene, palette)}"
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#bgGrad)"/>'
        f"{_background_motif(scene, palette, width, height)}"
        f"{body}"
        "</svg>"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Render spec14a scene DSL to SVG.")
    ap.add_argument("--scene", default=None, help="Inline scene document.")
    ap.add_argument("--scene-file", default=None, help="Path to a scene document file.")
    ap.add_argument("--content-json", default=None, help="Optional content JSON payload path.")
    ap.add_argument("--out", default=None, help="Optional output SVG path.")
    args = ap.parse_args()

    if bool(args.scene) == bool(args.scene_file):
        raise SystemExit("ERROR: pass exactly one of --scene or --scene-file")
    text = args.scene if args.scene is not None else Path(args.scene_file).read_text(encoding="utf-8")
    content = None
    if args.content_json:
        content = json.loads(Path(args.content_json).read_text(encoding="utf-8"))
    svg = render_structured_scene_spec14a_svg(text, content=content)
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
