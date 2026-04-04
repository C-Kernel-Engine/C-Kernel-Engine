#!/usr/bin/env python3
"""Render spec13b scene DSL with a generalized decision-tree layout path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec09_v7 import (
    _GAP_PX,
    _INSET_PX,
    _background_motif,
    _defs as _spec09_defs,
    _palette,
    _svg_text_block,
)
from render_svg_structured_scene_spec12_v7 import (
    _header_from_ref,
    _memory_map,
    _payload_obj,
    _table_matrix,
)
from spec12_scene_canonicalizer_v7 import canonicalize_scene_text
from spec13b_decision_graph_v7 import grouped_layers, lower_legacy_decision_scene


def _canvas_size(scene: dict[str, Any], *, layer_count: int = 0) -> tuple[int, int]:
    layout = str(scene.get("layout") or "")
    if layout == "decision_tree":
        width = 1780
        height = max(1220, 280 + max(1, layer_count) * 270)
        return width, height
    if layout == "table_matrix":
        return (1400, 1880) if scene["canvas"] == "tall" else (1400, 1560)
    if layout == "memory_map":
        return (1280, 980)
    raise ValueError(f"unsupported spec13b scene layout: {layout}")


def _defs(scene: dict[str, Any], palette: dict[str, str]) -> str:
    proxy = dict(scene)
    proxy["connector"] = str(scene.get("connector") or "arrow")
    return _spec09_defs(proxy, palette)


def _connector_style_attrs(connector: str, *, stroke: str) -> str:
    mode = str(connector or "arrow").strip() or "arrow"
    marker = ' marker-end="url(#arrowHead)"' if mode in {"arrow", "curve"} else ""
    dash = ""
    linecap = ""
    if mode == "dashed":
        dash = ' stroke-dasharray="10 8"'
    elif mode == "dotted":
        dash = ' stroke-dasharray="2 10"'
        linecap = ' stroke-linecap="round"'
    return f'stroke="{stroke}" stroke-width="3" fill="none"{dash}{linecap}{marker}'


def _box_height(details: tuple[str, ...], *, kind: str) -> float:
    base = 126.0 if kind == "decision" else 142.0
    extra = min(4, len(details)) * 20.0
    return min(244.0, base + extra)


def _decision_tree_generalized(scene: dict[str, Any], palette: dict[str, str]) -> str:
    graph = lower_legacy_decision_scene(scene)
    layers = grouped_layers(graph)
    inset = _INSET_PX.get(str(scene["inset"]), 56)
    gap_y = max(64.0, float(_GAP_PX.get(str(scene["gap"]), 26)) * 2.4)
    width, height = _canvas_size(scene, layer_count=len(layers))

    body: list[str] = []
    header, y = _header_from_ref(scene, palette, width, inset)
    body.append(header)

    entry_y = y + 4
    body.append(f'<rect x="{width/2 - 88:.1f}" y="{entry_y:.1f}" width="176" height="30" rx="15" fill="#ffe0a8" stroke="none"/>')
    body.append(_svg_text_block(width / 2, entry_y + 20, graph.entry_label or "ENTRY", font_size=12, fill="#132537", weight=700, max_chars=18, anchor="middle"))

    available_w = width - 2 * inset
    max_per_layer = max((len(layer) for layer in layers), default=1)
    box_w = min(420.0, max(280.0, (available_w - (max_per_layer - 1) * 28.0) / max_per_layer))

    positions: dict[str, tuple[float, float, float, float]] = {}
    current_y = entry_y + 46
    for layer in layers:
        heights = [_box_height(node.details, kind=node.kind) for node in layer]
        layer_h = max(heights, default=160.0)
        count = len(layer)
        row_w = count * box_w + max(0, count - 1) * 28.0
        start_x = inset + (available_w - row_w) / 2
        for idx, node in enumerate(layer):
            x = start_x + idx * (box_w + 28.0)
            h = heights[idx]
            y0 = current_y + (layer_h - h) / 2
            positions[node.node_id] = (x, y0, box_w, h)
        current_y += layer_h + gap_y

    # Recompute canvas height to include footer breathing room.
    height = max(height, int(current_y + 120))

    def draw_box(node_id: str, title: str, body_text: str, details: tuple[str, ...], *, kind: str) -> str:
        x, y0, w, h = positions[node_id]
        stroke = "#436486" if kind == "decision" else palette["accent_2"]
        accent = "#8fdfff" if kind == "decision" else palette["accent"]
        parts = ['<g class="decision-box">']
        parts.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" rx="14" fill="{palette["surface"]}" fill-opacity="0.92" stroke="{stroke}" stroke-width="1.4"/>')
        parts.append(_svg_text_block(x + 24, y0 + 40, title, font_size=20, fill=palette["ink"], weight=700, max_chars=30))
        parts.append(_svg_text_block(x + 24, y0 + 66, body_text, font_size=13, fill=palette["muted"], max_chars=56))
        cy = y0 + 98
        for detail in details[:4]:
            parts.append(_svg_text_block(x + 28, cy, detail, font_size=12, fill=accent, weight=700, max_chars=60, family="IBM Plex Mono, SFMono-Regular, monospace"))
            cy += 20
        parts.append("</g>")
        return "".join(parts)

    for node in graph.nodes:
        body.append(draw_box(node.node_id, node.title, node.body, node.details, kind=node.kind))

    connector_mode = str(graph.connector or "arrow")
    connector_stroke = palette["accent"]
    connector_attrs = _connector_style_attrs(connector_mode, stroke=connector_stroke)
    line_parts: list[str] = ['<g class="tree-connectors">']
    for edge in graph.edges:
        src = positions.get(edge.source_id)
        dst = positions.get(edge.target_id)
        if src is None or dst is None:
            continue
        sx, sy, sw, sh = src
        dx, dy, dw, _ = dst
        x1 = sx + sw / 2
        y1 = sy + sh
        x2 = dx + dw / 2
        y2 = dy
        mid_y = (y1 + y2) / 2
        line_parts.append(
            f'<path d="M {x1:.1f} {y1:.1f} C {x1:.1f} {mid_y:.1f}, {x2:.1f} {mid_y:.1f}, {x2:.1f} {y2 - 8:.1f}" '
            f'{connector_attrs}/>'
        )
        if edge.label:
            line_parts.append(
                _svg_text_block((x1 + x2) / 2, mid_y - 8, edge.label, font_size=11, fill=connector_stroke, weight=700, max_chars=22, anchor="middle")
            )
    line_parts.append("</g>")
    body.append("".join(line_parts))

    if graph.footer_note:
        footer_y = height - 86
        body.append(f'<rect x="{inset:.1f}" y="{footer_y:.1f}" width="{width - 2 * inset:.1f}" height="52" rx="14" fill="{palette["surface_2"]}" fill-opacity="0.92" stroke="{palette["border"]}" stroke-width="1"/>')
        body.append(_svg_text_block(inset + 24, footer_y + 32, graph.footer_note, font_size=12, fill=palette["muted"], max_chars=140))

    return "".join(body), width, height


def _flow_graph_generalized(scene: dict[str, Any], palette: dict[str, str]) -> str:
    graph = lower_legacy_decision_scene(scene)
    layers = grouped_layers(graph)
    inset = _INSET_PX.get(str(scene["inset"]), 56)
    width = 1760

    body: list[str] = []
    header, y = _header_from_ref(scene, palette, width, inset)
    body.append(header)

    available_w = width - 2 * inset
    total_layers = max(1, len(layers))
    gap_x = 72.0
    gap_y = 36.0
    box_w = min(360.0, max(280.0, (available_w - gap_x * (total_layers - 1)) / total_layers))
    box_h = 136.0
    col_w = (available_w - gap_x * (total_layers - 1)) / total_layers
    top_y = y + 28
    height = max(820, int(top_y + max(len(layer) for layer in layers) * (box_h + gap_y) + 160))

    positions: dict[str, tuple[float, float, float, float]] = {}
    for layer_idx, layer in enumerate(layers):
        x = inset + layer_idx * (col_w + gap_x) + max(0.0, (col_w - box_w) / 2)
        layer_total_h = len(layer) * box_h + max(0, len(layer) - 1) * gap_y
        start_y = top_y + max(0.0, ((height - 140) - top_y - layer_total_h) / 2)
        for row_idx, node in enumerate(layer):
            y0 = start_y + row_idx * (box_h + gap_y)
            positions[node.node_id] = (x, y0, box_w, box_h)

    for node in graph.nodes:
        x, y0, w, h = positions[node.node_id]
        accent = palette["accent"] if node.kind != "outcome" else palette["accent_2"]
        badge = "OUTCOME" if node.kind == "outcome" else "STAGE"
        body.append('<g class="flow-node">')
        body.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" rx="16" fill="{palette["surface"]}" fill-opacity="0.94" stroke="{accent}" stroke-width="1.6"/>')
        body.append(f'<rect x="{x + 18:.1f}" y="{y0 + 16:.1f}" width="96" height="24" rx="12" fill="{accent}" fill-opacity="0.14" stroke="{accent}" stroke-width="1"/>')
        body.append(_svg_text_block(x + 66, y0 + 32, badge, font_size=10, fill=accent, weight=700, max_chars=14, anchor="middle"))
        body.append(_svg_text_block(x + 18, y0 + 58, node.title, font_size=19, fill=palette["ink"], weight=700, max_chars=26))
        body.append(_svg_text_block(x + 18, y0 + 84, node.body, font_size=12, fill=palette["muted"], max_chars=42))
        cy = y0 + 112
        for detail in node.details[:2]:
            body.append(_svg_text_block(x + 22, cy, detail, font_size=11, fill=accent, weight=700, max_chars=36))
            cy += 18
        body.append("</g>")

    connector_mode = str(graph.connector or "arrow")
    connector_stroke = palette["accent"]
    connector_attrs = _connector_style_attrs(connector_mode, stroke=connector_stroke)
    line_parts: list[str] = ['<g class="flow-connectors">']
    for edge in graph.edges:
        src = positions.get(edge.source_id)
        dst = positions.get(edge.target_id)
        if src is None or dst is None:
            continue
        sx, sy, sw, sh = src
        dx, dy, _, dh = dst
        x1 = sx + sw
        y1 = sy + sh / 2
        x2 = dx
        y2 = dy + dh / 2
        mid_x = (x1 + x2) / 2
        line_parts.append(
            f'<path d="M {x1:.1f} {y1:.1f} C {mid_x:.1f} {y1:.1f}, {mid_x:.1f} {y2:.1f}, {x2 - 8:.1f} {y2:.1f}" '
            f'{connector_attrs}/>'
        )
        if edge.label:
            line_parts.append(_svg_text_block(mid_x, min(y1, y2) - 8, edge.label, font_size=11, fill=connector_stroke, weight=700, max_chars=18, anchor="middle"))
    line_parts.append("</g>")
    body.append("".join(line_parts))

    if graph.footer_note:
        footer_y = height - 72
        body.append(f'<rect x="{inset:.1f}" y="{footer_y:.1f}" width="{width - 2 * inset:.1f}" height="44" rx="14" fill="{palette["surface_2"]}" fill-opacity="0.92" stroke="{palette["border"]}" stroke-width="1"/>')
        body.append(_svg_text_block(inset + 20, footer_y + 28, graph.footer_note, font_size=12, fill=palette["muted"], max_chars=140))

    return "".join(body), width, height


def render_structured_scene_spec13b_svg(text: str, content: dict[str, Any] | None = None) -> str:
    scene = canonicalize_scene_text(text).to_runtime()
    scene["_content"] = content or {}
    palette = _palette(scene)
    layout = str(scene["layout"])
    if layout == "decision_tree":
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
            raise ValueError(f"unsupported spec13b scene layout: {layout}")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f"{_defs(scene, palette)}"
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#bgGrad)"/>'
        f"{_background_motif(scene, palette, width, height)}"
        f"{body}"
        "</svg>"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Render spec13b scene DSL to SVG.")
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
    svg = render_structured_scene_spec13b_svg(text, content=content)
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
