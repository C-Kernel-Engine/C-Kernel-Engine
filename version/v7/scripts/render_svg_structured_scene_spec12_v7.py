#!/usr/bin/env python3
"""Render compact spec12 scene DSL families into SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec09_v7 import (
    _GAP_PX,
    _INSET_PX,
    _background_motif,
    _canvas_size as _spec09_canvas_size,
    _color_token,
    _defs as _spec09_defs,
    _escape,
    _lookup_content_path,
    _palette,
    _panel_shell,
    _svg_text_block,
)
from spec12_scene_canonicalizer_v7 import (
    BACKGROUNDS,
    CANVAS,
    DENSITIES,
    FRAMES,
    LAYOUTS,
    THEMES,
    TONES,
    canonicalize_scene_text,
)


def _payload_obj(content: dict[str, Any] | None, ref: str) -> Any:
    return _lookup_content_path(content, str(ref or "").strip())


def _payload_text(content: dict[str, Any] | None, ref: str) -> str:
    value = _payload_obj(content, ref)
    if value is None:
        return f"missing:{ref}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _canvas_size(scene: dict[str, Any]) -> tuple[int, int]:
    layout = str(scene["layout"])
    if layout == "table_matrix":
        return (1400, 1880) if scene["canvas"] == "tall" else (1400, 1560)
    if layout == "decision_tree":
        return (1780, 1220)
    if layout == "memory_map":
        return (1280, 980)
    return _spec09_canvas_size(str(scene["canvas"]))


def _defs(scene: dict[str, Any], palette: dict[str, str]) -> str:
    if scene["layout"] == "decision_tree":
        proxy = dict(scene)
        proxy["connector"] = "arrow"
        return _spec09_defs(proxy, palette)
    return _spec09_defs(scene, palette)


def _component_entries(scene: dict[str, Any], name: str) -> list[dict[str, str]]:
    return list(scene["components_by_name"].get(name, []))


def _header_from_ref(scene: dict[str, Any], palette: dict[str, str], width: int, inset: float) -> tuple[str, float]:
    entries = _component_entries(scene, "header_band")
    ref = str(entries[0].get("ref") or "header") if entries else "header"
    header = _payload_obj(scene.get("_content"), ref)
    if not isinstance(header, dict):
        header = {}
    kicker = str(header.get("kicker") or "")
    title = str(header.get("headline") or scene.get("topic") or "Untitled")
    subtitle = str(header.get("subtitle") or "")
    body: list[str] = []
    body.append(
        f'<rect x="{inset:.1f}" y="{inset:.1f}" width="{width - 2 * inset:.1f}" height="112" rx="22" '
        f'fill="{palette["surface"]}" fill-opacity="0.18" stroke="{palette["border"]}" stroke-width="1.2"/>'
    )
    body.append(
        f'<rect x="{inset + 18:.1f}" y="{inset + 16:.1f}" width="220" height="8" rx="4" fill="url(#heroGrad)" filter="url(#accentGlow)"/>'
    )
    if kicker:
        body.append(_svg_text_block(inset + 24, inset + 42, kicker, font_size=13, fill=palette["accent"], weight=700, max_chars=32))
    body.append(_svg_text_block(inset + 24, inset + 74, title, font_size=30, fill=palette["ink"], weight=700, max_chars=38))
    if subtitle:
        body.append(_svg_text_block(inset + 24, inset + 104, subtitle, font_size=15, fill=palette["muted"], max_chars=86))
    return "".join(body), inset + 144


def _table_matrix(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX.get(str(scene["inset"]), 56)
    gap = _GAP_PX.get(str(scene["gap"]), 26)
    body: list[str] = []
    header, y = _header_from_ref(scene, palette, width, inset)
    body.append(header)

    legend_entries = _component_entries(scene, "legend_block")
    legend_ref = str(legend_entries[0].get("ref") or "legend") if legend_entries else "legend"
    legend = _payload_obj(scene.get("_content"), legend_ref)
    if not isinstance(legend, dict):
        legend = {}
    legend_y = y
    legend_h = 88.0
    legend_parts: list[str] = []
    legend_parts.append(_panel_shell(inset, legend_y, width - 2 * inset, legend_h, scene, palette))
    legend_parts.append(f'<rect x="{inset + 18:.1f}" y="{legend_y + 18:.1f}" width="{width - 2 * inset - 36:.1f}" height="22" rx="11" fill="{palette["accent"]}" fill-opacity="0.08"/>')
    legend_parts.append(_svg_text_block(inset + 24, legend_y + 28, str(legend.get("title") or "Legend"), font_size=16, fill=palette["ink"], weight=700, max_chars=28))
    items = legend.get("items") if isinstance(legend.get("items"), list) else []
    item_x = inset + 24
    for idx, item in enumerate(items[:5]):
        color = ["#ffb400", palette["success"], "#56aefc", "#b576ff", palette["accent_2"]][idx % 5]
        w = 176
        x = item_x + idx * (w + 10)
        legend_parts.append(f'<rect x="{x:.1f}" y="{legend_y + 42:.1f}" width="{w:.1f}" height="24" rx="12" fill="{color}" fill-opacity="0.14" stroke="{color}" stroke-width="1"/>')
        legend_parts.append(_svg_text_block(x + 12, legend_y + 58, str(item), font_size=11, fill=color, weight=700, max_chars=20))
    if legend.get("note"):
        legend_parts.append(_svg_text_block(inset + 24, legend_y + 78, str(legend.get("note")), font_size=11, fill=palette["muted"], max_chars=120))
    warning = str(legend.get("warning") or "")
    if warning:
        legend_parts.append(_svg_text_block(width - inset - 280, legend_y + 78, warning, font_size=11, fill=_color_token(palette, "warning", fallback=palette["accent_2"]), weight=700, max_chars=44))
    body.append(f'<g class="legend-block">{"".join(legend_parts)}</g>')

    current_y = legend_y + legend_h + gap
    blocks = [str(entry.get("ref") or "") for entry in _component_entries(scene, "table_block")]
    panel_w = width - 2 * inset
    for raw in blocks:
        group = _payload_obj(scene.get("_content"), raw)
        if not isinstance(group, dict):
            continue
        rows = group.get("rows") if isinstance(group.get("rows"), list) else []
        block_h = 118 + 58 * max(1, len(rows))
        block_parts: list[str] = []
        block_parts.append(_panel_shell(inset, current_y, panel_w, block_h, scene, palette))
        block_parts.append(f'<rect x="{inset + 18:.1f}" y="{current_y + 18:.1f}" width="{panel_w - 36:.1f}" height="30" rx="10" fill="{palette["accent_2"]}" fill-opacity="0.09" stroke="{palette["accent_2"]}" stroke-width="1"/>')
        block_parts.append(_svg_text_block(inset + 28, current_y + 38, str(group.get("title") or ""), font_size=21, fill=palette["ink"], weight=700, max_chars=48))
        if group.get("caption"):
            block_parts.append(_svg_text_block(inset + 28, current_y + 62, str(group.get("caption")), font_size=12, fill=palette["muted"], max_chars=96))
        header_cells = group.get("header") if isinstance(group.get("header"), list) else []
        col_x = [inset + 36, inset + panel_w * 0.42, inset + panel_w * 0.67, inset + panel_w - 120]
        hy = current_y + 76
        block_parts.append(f'<rect x="{inset + 22:.1f}" y="{hy:.1f}" width="{panel_w - 44:.1f}" height="34" rx="10" fill="{palette["accent"]}" fill-opacity="0.10" stroke="{palette["accent"]}" stroke-width="1"/>')
        for idx, label in enumerate(header_cells[:4]):
            block_parts.append(_svg_text_block(col_x[idx], hy + 22, str(label), font_size=11, fill=palette["accent"], weight=700, max_chars=20))
        divider_x = [inset + panel_w * 0.37, inset + panel_w * 0.62, inset + panel_w * 0.83]
        for x in divider_x:
            block_parts.append(f'<line x1="{x:.1f}" y1="{hy + 4:.1f}" x2="{x:.1f}" y2="{current_y + block_h - 22:.1f}" stroke="{palette["border"]}" stroke-width="1"/>')
        row_y = hy + 42
        for row in rows:
            cells = row.get("cells") if isinstance(row, dict) and isinstance(row.get("cells"), list) else []
            state = str(row.get("state") or "normal") if isinstance(row, dict) else "normal"
            stripe = _color_token(palette, state, fallback=palette["border"])
            fill_op = "0.12" if state in {"highlight", "warning", "success", "danger"} else "0.86"
            fill = stripe if fill_op != "0.86" else palette["surface_2"]
            block_parts.append(f'<rect x="{inset + 22:.1f}" y="{row_y:.1f}" width="{panel_w - 44:.1f}" height="44" rx="12" fill="{fill}" fill-opacity="{fill_op}" stroke="{stripe}" stroke-width="1"/>')
            if fill_op != "0.86":
                block_parts.append(f'<rect x="{inset + 22:.1f}" y="{row_y:.1f}" width="7" height="44" rx="3.5" fill="{stripe}"/>')
            for idx, cell in enumerate(cells[:4]):
                color = stripe if idx > 0 and fill_op != "0.86" else (palette["ink"] if idx == 0 else palette["muted"])
                weight = 700 if idx == 0 else 600 if idx == 3 else 500
                block_parts.append(_svg_text_block(col_x[idx], row_y + 27, str(cell), font_size=12, fill=color, weight=weight, max_chars=24))
            block_parts.append(f'<line x1="{inset + 24:.1f}" y1="{row_y + 44:.1f}" x2="{inset + panel_w - 24:.1f}" y2="{row_y + 44:.1f}" stroke="{palette["border"]}" stroke-width="1"/>')
            row_y += 52
        body.append(f'<g class="table-block">{"".join(block_parts)}</g>')
        current_y += block_h + gap

    footer_entries = _component_entries(scene, "note_band")
    footer_ref = str(footer_entries[0].get("ref") or "footer") if footer_entries else "footer"
    footer = _payload_obj(scene.get("_content"), footer_ref)
    if isinstance(footer, dict):
        body.append('<g class="note-band">')
        body.append(f'<rect x="{inset:.1f}" y="{current_y:.1f}" width="{panel_w:.1f}" height="64" rx="18" fill="{palette["accent"]}" fill-opacity="0.08" stroke="{palette["accent"]}" stroke-width="1"/>')
        body.append(f'<line x1="{inset + 20:.1f}" y1="{current_y + 20:.1f}" x2="{inset + panel_w - 20:.1f}" y2="{current_y + 20:.1f}" stroke="{palette["accent"]}" stroke-width="1"/>')
        body.append(_svg_text_block(inset + 26, current_y + 25, str(footer.get("title") or ""), font_size=15, fill=palette["accent"], weight=700, max_chars=40))
        body.append(_svg_text_block(inset + 26, current_y + 47, str(footer.get("body") or ""), font_size=11, fill=palette["muted"], max_chars=128))
        body.append("</g>")
    return "".join(body)


def _decision_tree(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX.get(str(scene["inset"]), 56)
    body: list[str] = []
    header, y = _header_from_ref(scene, palette, width, inset)
    body.append(header)

    entry_entries = _component_entries(scene, "entry_badge")
    entry_ref = str(entry_entries[0].get("ref") or "entry") if entry_entries else "entry"
    entry = _payload_obj(scene.get("_content"), entry_ref)
    entry_label = str(entry.get("label") if isinstance(entry, dict) else "ENTRY")
    body.append(f'<rect x="{width/2 - 88:.1f}" y="{y + 4:.1f}" width="176" height="30" rx="15" fill="#ffe0a8" stroke="none"/>')
    body.append(_svg_text_block(width / 2, y + 24, entry_label, font_size=12, fill="#132537", weight=700, max_chars=18, anchor="middle"))

    nodes_y = y + 50
    node_w = 460.0
    node_h = 180.0
    xs = [inset, (width - node_w) / 2, width - inset - node_w]
    top_pos = {"l0_l1": (xs[0], nodes_y + 170), "l2": (xs[1], nodes_y + 170), "l3_l4": (xs[2], nodes_y + 170)}
    start_pos = {"start": ((width - node_w) / 2, nodes_y)}
    outcome_y = nodes_y + 420
    outcome_h = 190.0
    outcome_pos = {
        "contract_drift": (xs[0], outcome_y),
        "model_row": (xs[1], outcome_y),
        "parity_branch": (xs[2], outcome_y),
        "finish": ((width - node_w) / 2, outcome_y + 250),
    }

    def draw_box(x: float, y0: float, title: str, body_text: str, details: list[str], edge: str) -> str:
        box_h = node_h if y0 < outcome_y else outcome_h
        parts = ['<g class="decision-box">']
        parts.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{node_w:.1f}" height="{box_h:.1f}" rx="14" fill="{palette["surface"]}" fill-opacity="0.92" stroke="#436486" stroke-width="1.4"/>')
        parts.append(_svg_text_block(x + 26, y0 + 44, title, font_size=20, fill=palette["ink"], weight=700, max_chars=28))
        parts.append(_svg_text_block(x + 26, y0 + 70, body_text, font_size=13, fill=palette["muted"], max_chars=56))
        cy = y0 + 102
        for detail in details[:3]:
            parts.append(_svg_text_block(x + 30, cy, detail, font_size=12, fill=edge, weight=700, max_chars=58, family="IBM Plex Mono, SFMono-Regular, monospace"))
            cy += 20
        parts.append("</g>")
        return "".join(parts)

    node_refs = {
        str(entry.get("node_id") or ""): str(entry.get("ref") or "")
        for entry in _component_entries(scene, "decision_node")
        if entry.get("node_id") and entry.get("ref")
    }
    positions = {}
    for node_id, pos in list(start_pos.items()) + list(top_pos.items()):
        ref = node_refs.get(node_id)
        node = _payload_obj(scene.get("_content"), ref) if ref else None
        if not isinstance(node, dict):
            continue
        positions[node_id] = (pos[0] + node_w / 2, pos[1] + node_h / 2)
        body.append(draw_box(pos[0], pos[1], str(node.get("title") or ""), str(node.get("body") or ""), list(node.get("detail") or []), "#8fdfff"))

    outcome_refs = {
        str(entry.get("panel_id") or ""): str(entry.get("ref") or "")
        for entry in _component_entries(scene, "outcome_panel")
        if entry.get("panel_id") and entry.get("ref")
    }
    for panel_id, pos in outcome_pos.items():
        ref = outcome_refs.get(panel_id)
        node = _payload_obj(scene.get("_content"), ref) if ref else None
        if not isinstance(node, dict):
            continue
        positions[panel_id] = (pos[0] + node_w / 2, pos[1] + outcome_h / 2)
        body.append(draw_box(pos[0], pos[1], str(node.get("title") or ""), str(node.get("body") or ""), list(node.get("detail") or []), "#8fdfff"))

    line_parts: list[str] = ['<g class="tree-connectors">']
    label_parts: list[str] = []
    for entry in _component_entries(scene, "decision_edge"):
        src = str(entry.get("from_ref") or "")
        dst = str(entry.get("to_ref") or "")
        label_ref = str(entry.get("label_ref") or "")
        if not src or not dst or not label_ref:
            continue
        label = _payload_text(scene.get("_content"), label_ref)
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 - 12
        if src == "start":
            line_parts.append(f'<line x1="{x1:.1f}" y1="{y1 + 34:.1f}" x2="{x2:.1f}" y2="{y2 - 92:.1f}" stroke="#8dc0e8" stroke-width="3" marker-end="url(#arrowHead)"/>')
            label_parts.append(_svg_text_block(mid_x, mid_y, label, font_size=11, fill="#c5d9ed", weight=700, max_chars=18, anchor="middle"))
        elif src == "l2" and dst == "finish":
            line_parts.append(f'<line x1="{x1:.1f}" y1="{y1 + 88:.1f}" x2="{x2:.1f}" y2="{y2 - 108:.1f}" stroke="#8dc0e8" stroke-width="3" marker-end="url(#arrowHead)"/>')
            label_parts.append(_svg_text_block(mid_x, mid_y + 10, label, font_size=11, fill="#c5d9ed", weight=700, max_chars=18, anchor="middle"))
    top_to_outcome = [("l0_l1", "contract_drift"), ("l2", "model_row"), ("l3_l4", "parity_branch")]
    for src, dst in top_to_outcome:
        if src in positions and dst in positions:
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            line_parts.append(f'<line x1="{x1:.1f}" y1="{y1 + 88:.1f}" x2="{x2:.1f}" y2="{y2 - 108:.1f}" stroke="#8dc0e8" stroke-width="3" marker-end="url(#arrowHead)"/>')
    line_parts.extend(label_parts)
    line_parts.append("</g>")
    body.append("".join(line_parts))

    footer_entries = _component_entries(scene, "footer_note")
    footer_ref = str(footer_entries[0].get("ref") or "footer") if footer_entries else "footer"
    footer = _payload_obj(scene.get("_content"), footer_ref)
    note = str(footer.get("note") if isinstance(footer, dict) else footer or "")
    if note:
        body.append(f'<rect x="{inset:.1f}" y="{height - 86:.1f}" width="{width - 2 * inset:.1f}" height="52" rx="14" fill="{palette["surface_2"]}" fill-opacity="0.92" stroke="{palette["border"]}" stroke-width="1"/>')
        body.append(_svg_text_block(inset + 24, height - 54, note, font_size=12, fill=palette["muted"], max_chars=140))
    return "".join(body)


def _memory_map(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX.get(str(scene["inset"]), 56)
    gap = _GAP_PX.get(str(scene["gap"]), 26)
    body: list[str] = []
    header, y = _header_from_ref(scene, palette, width, inset)
    body.append(header)

    left_x = inset
    left_w = width * 0.56 - inset
    right_x = left_x + left_w + gap
    right_w = width - right_x - inset
    tower_y = y + 8
    tower_h = height - tower_y - inset
    body.append('<g class="memory-tower">')
    body.append(_panel_shell(left_x, tower_y, left_w, tower_h, scene, palette))
    body.append(f'<line x1="{left_x + 10:.1f}" y1="{tower_y + 24:.1f}" x2="{left_x + 10:.1f}" y2="{tower_y + tower_h - 24:.1f}" stroke="{palette["border"]}" stroke-width="1.2"/>')

    offset_entries = _component_entries(scene, "address_strip")
    offsets_ref = str(offset_entries[0].get("ref") or "offsets") if offset_entries else "offsets"
    offsets = _payload_obj(scene.get("_content"), offsets_ref)
    if not isinstance(offsets, list):
        offsets = []
    for idx, offset in enumerate(offsets[:7]):
        oy = tower_y + 42 + idx * 84
        body.append(_svg_text_block(left_x - 12, oy, str(offset), font_size=11, fill=palette["muted"], max_chars=18, anchor="end", family="IBM Plex Mono, SFMono-Regular, monospace"))
        body.append(f'<circle cx="{left_x + 10:.1f}" cy="{oy - 4:.1f}" r="4" fill="{palette["accent_2"]}" opacity="0.9"/>')

    seg_order = []
    for entry in _component_entries(scene, "memory_segment"):
        seg_id = str(entry.get("segment_id") or "")
        ref = str(entry.get("ref") or "")
        if seg_id and ref:
            seg_order.append((seg_id, ref))
    seg_heights = {
        "embeddings": 78.0,
        "layer_0": 84.0,
        "layer_1": 84.0,
        "layer_23": 84.0,
        "lm_head": 68.0,
        "runtime": 112.0,
    }
    current_y = tower_y + 18
    bracket_anchor = None
    for seg_id, ref in seg_order:
        seg = _payload_obj(scene.get("_content"), ref)
        if not isinstance(seg, dict):
            continue
        h = seg_heights.get(seg_id, 76.0)
        color = palette["accent"] if seg_id in {"embeddings", "lm_head"} else palette["surface_2"]
        stroke = palette["accent"] if seg_id in {"embeddings", "lm_head"} else palette["border"]
        body.append('<g class="memory-segment">')
        body.append(f'<rect x="{left_x + 18:.1f}" y="{current_y:.1f}" width="{left_w - 36:.1f}" height="{h:.1f}" rx="12" fill="{color}" fill-opacity="0.16" stroke="{stroke}" stroke-width="1.4"/>')
        body.append(_svg_text_block(left_x + 34, current_y + 28, str(seg.get("title") or ""), font_size=16, fill=palette["ink"], weight=700, max_chars=24))
        if seg.get("size"):
            body.append(_svg_text_block(left_x + left_w - 34, current_y + 28, str(seg.get("size") or ""), font_size=12, fill=palette["accent"], weight=700, max_chars=18, anchor="end", family="IBM Plex Mono, SFMono-Regular, monospace"))
        if seg.get("caption"):
            body.append(_svg_text_block(left_x + 34, current_y + 50, str(seg.get("caption") or ""), font_size=11, fill=palette["muted"], max_chars=60))
        parts = seg.get("parts") if isinstance(seg.get("parts"), list) else []
        if parts:
            px = left_x + 34
            py = current_y + h - 30
            for part in parts[:5]:
                w = 58 if len(str(part)) <= 3 else 84
                body.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{w:.1f}" height="22" rx="8" fill="{palette["surface"]}" fill-opacity="0.90" stroke="{palette["border"]}" stroke-width="1"/>')
                body.append(_svg_text_block(px + w/2, py + 15, str(part), font_size=10, fill=palette["ink"], weight=700, max_chars=10, anchor="middle", family="IBM Plex Mono, SFMono-Regular, monospace"))
                px += w + 8
        body.append("</g>")
        if seg_id == "layer_1":
            bracket_anchor = current_y + h
        current_y += h + 12
        if seg_id == "layer_1":
            current_y += 60
        if seg_id == "lm_head":
            current_y += 8

    bracket_entries = _component_entries(scene, "region_bracket")
    bracket_ref = str(bracket_entries[0].get("ref") or "segments.mid_layers") if bracket_entries else "segments.mid_layers"
    bracket = _payload_obj(scene.get("_content"), bracket_ref)
    if isinstance(bracket, dict) and bracket_anchor is not None:
        bx = left_x + left_w - 26
        by = bracket_anchor + 6
        body.append(f'<path d="M {bx} {by} q 18 0 18 18 v 42 q 0 18 18 18" fill="none" stroke="{palette["accent_2"]}" stroke-width="3"/>')
        body.append(_svg_text_block(bx - 8, by + 34, str(bracket.get("title") or ""), font_size=12, fill=palette["accent"], weight=700, max_chars=16, anchor="end"))
        body.append(_svg_text_block(bx - 8, by + 54, str(bracket.get("caption") or ""), font_size=10, fill=palette["muted"], max_chars=30, anchor="end"))

    card_y = tower_y
    for entry in _component_entries(scene, "info_card"):
        ref = str(entry.get("ref") or "")
        card_id = str(entry.get("card_id") or "")
        if not ref:
            continue
        card = _payload_obj(scene.get("_content"), ref)
        if not isinstance(card, dict):
            continue
        card_h = 190.0 if "items" in card else 156.0 if "lines" in card and len(card.get("lines", [])) > 2 else 120.0
        body.append('<g class="info-card">')
        card_stroke = "#ffb400" if "benefit" in card_id else "#4299e1" if "direct" in card_id else palette["border"]
        body.append(f'<rect x="{right_x:.1f}" y="{card_y:.1f}" width="{right_w:.1f}" height="{card_h:.1f}" rx="14" fill="{palette["surface"]}" fill-opacity="0.92" stroke="{card_stroke}" stroke-width="1.4"/>')
        body.append(_svg_text_block(right_x + 24, card_y + 32, str(card.get("title") or ""), font_size=16, fill=card_stroke, weight=700, max_chars=28))
        if isinstance(card.get("items"), list):
            iy = card_y + 62
            for item in card["items"][:5]:
                body.append(f'<circle cx="{right_x + 20:.1f}" cy="{iy - 5:.1f}" r="6" fill="{palette["success"]}"/>')
                body.append(_svg_text_block(right_x + 36, iy, str(item), font_size=12, fill=palette["ink"], max_chars=28))
                iy += 24
        elif isinstance(card.get("lines"), list):
            iy = card_y + 62
            for line in card["lines"][:5]:
                text_fill = "#a78bfa" if '"' in str(line) else "#f5f5f5"
                if "0x" in str(line):
                    text_fill = "#f59e0b"
                if str(card.get("title") or "").lower() == "direct access" and "rdma_read" in str(line):
                    text_fill = "#4299e1"
                body.append(_svg_text_block(right_x + 24, iy, str(line), font_size=11, fill=text_fill, max_chars=30, family="IBM Plex Mono, SFMono-Regular, monospace"))
                iy += 20
        body.append("</g>")
        card_y += card_h + gap * 0.8
    body.append("</g>")
    return "".join(body)


def render_structured_scene_spec12_svg(text: str, content: dict[str, Any] | None = None) -> str:
    scene = canonicalize_scene_text(text).to_runtime()
    scene["_content"] = content or {}
    width, height = _canvas_size(scene)
    palette = _palette(scene)
    body = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#bgGrad)"/>',
        _background_motif(scene, palette, width, height),
    ]
    layout = str(scene["layout"])
    if layout == "table_matrix":
        body.append(_table_matrix(scene, palette, width, height))
    elif layout == "decision_tree":
        body.append(_decision_tree(scene, palette, width, height))
    elif layout == "memory_map":
        body.append(_memory_map(scene, palette, width, height))
    else:
        raise ValueError(f"unsupported spec12 scene layout: {layout}")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f"{_defs(scene, palette)}"
        f'{"".join(body)}'
        "</svg>"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Render compact spec12 scene DSL to SVG.")
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
    svg = render_structured_scene_spec12_svg(text, content=content)
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
