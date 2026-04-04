#!/usr/bin/env python3
"""Render a generalized spec13b flow-graph IR into SVG."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec09_v7 import (
    _background_motif,
    _defs as _spec09_defs,
    _palette,
    _svg_text_block,
)


def _graph_layers(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    node_ids = [str(node.get("id") or "") for node in nodes if str(node.get("id") or "").strip()]
    nodes_by_id = {str(node.get("id")): node for node in nodes if str(node.get("id") or "").strip()}
    indegree = {node_id: 0 for node_id in node_ids}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        src = str(edge.get("source") or "").strip()
        dst = str(edge.get("target") or "").strip()
        if src in nodes_by_id and dst in nodes_by_id:
            outgoing[src].append(dst)
            indegree[dst] = indegree.get(dst, 0) + 1

    roots = [node_id for node_id in node_ids if indegree.get(node_id, 0) == 0]
    if not roots and node_ids:
        roots = [node_ids[0]]

    layers: dict[str, int] = {}
    queue = deque((node_id, 0) for node_id in roots)
    while queue:
        node_id, depth = queue.popleft()
        if node_id in layers and layers[node_id] <= depth:
            continue
        layers[node_id] = depth
        for child in outgoing.get(node_id, ()):
            queue.append((child, depth + 1))

    max_depth = max(layers.values(), default=0)
    for node_id in node_ids:
        if node_id not in layers:
            max_depth += 1
            layers[node_id] = max_depth

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        grouped[layers[str(node["id"])]].append(node)
    return [grouped[idx] for idx in sorted(grouped)]


def render_flow_graph_svg(doc: dict[str, Any]) -> str:
    scene = {
        "layout": "decision_tree",
        "theme": str(doc.get("theme") or "infra_dark"),
        "tone": str(doc.get("tone") or "blue"),
        "density": str(doc.get("density") or "balanced"),
        "background": str(doc.get("background") or "none"),
        "connector": "arrow",
    }
    palette = _palette(scene)
    proxy = dict(scene)
    proxy["connector"] = "arrow"

    nodes = [node for node in (doc.get("nodes") or []) if isinstance(node, dict) and str(node.get("id") or "").strip()]
    edges = [edge for edge in (doc.get("edges") or []) if isinstance(edge, dict)]
    layers = _graph_layers(nodes, edges)

    width = 1760
    inset = 56.0
    gap_x = 80.0
    gap_y = 44.0
    header_h = 150.0
    footer_h = 70.0 if doc.get("footer") else 0.0
    box_w = 360.0
    box_h = 140.0
    total_layers = max(1, len(layers))
    height = int(header_h + 120 + max(len(layer) for layer in layers) * (box_h + gap_y) + footer_h + 120)

    parts: list[str] = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        _spec09_defs(proxy, palette),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#bgGrad)"/>',
        _background_motif(proxy, palette, width, height),
    ]

    header = doc.get("header") if isinstance(doc.get("header"), dict) else {}
    parts.append(f'<rect x="{inset:.1f}" y="{inset:.1f}" width="{width - 2*inset:.1f}" height="112" rx="22" fill="{palette["surface"]}" fill-opacity="0.18" stroke="{palette["border"]}" stroke-width="1.2"/>')
    parts.append(f'<rect x="{inset + 18:.1f}" y="{inset + 16:.1f}" width="240" height="8" rx="4" fill="url(#heroGrad)" filter="url(#accentGlow)"/>')
    parts.append(_svg_text_block(inset + 24, inset + 74, str(header.get("headline") or "Flow Graph"), font_size=30, fill=palette["ink"], weight=700, max_chars=44))
    if header.get("subtitle"):
        parts.append(_svg_text_block(inset + 24, inset + 104, str(header.get("subtitle")), font_size=15, fill=palette["muted"], max_chars=100))

    positions: dict[str, tuple[float, float, float, float]] = {}
    top_y = header_h + 36.0
    col_w = (width - 2 * inset - (total_layers - 1) * gap_x) / total_layers
    for layer_idx, layer in enumerate(layers):
        x = inset + layer_idx * (col_w + gap_x) + max(0.0, (col_w - box_w) / 2)
        layer_total_h = len(layer) * box_h + max(0, len(layer) - 1) * gap_y
        start_y = top_y + max(0.0, ((height - footer_h - 120) - top_y - layer_total_h) / 2)
        for row_idx, node in enumerate(layer):
            y = start_y + row_idx * (box_h + gap_y)
            positions[str(node["id"])] = (x, y, box_w, box_h)

    # connectors first so boxes sit on top
    parts.append('<g class="flow-connectors">')
    for edge in edges:
        src = positions.get(str(edge.get("source") or ""))
        dst = positions.get(str(edge.get("target") or ""))
        if src is None or dst is None:
            continue
        sx, sy, sw, sh = src
        dx, dy, _, _ = dst
        x1 = sx + sw
        y1 = sy + sh / 2
        x2 = dx
        y2 = dy + box_h / 2
        mid_x = (x1 + x2) / 2
        parts.append(
            f'<path d="M {x1:.1f} {y1:.1f} C {mid_x:.1f} {y1:.1f}, {mid_x:.1f} {y2:.1f}, {x2 - 8:.1f} {y2:.1f}" '
            f'stroke="#8dc0e8" stroke-width="3" fill="none" marker-end="url(#arrowHead)"/>'
        )
        label = str(edge.get("label") or "").strip()
        if label:
            parts.append(_svg_text_block(mid_x, min(y1, y2) - 8, label, font_size=11, fill="#c5d9ed", weight=700, max_chars=20, anchor="middle"))
    parts.append("</g>")

    for node in nodes:
        x, y, w, h = positions[str(node["id"])]
        kind = str(node.get("kind") or "stage")
        stroke = palette["accent"] if kind in {"stage", "entry"} else palette["accent_2"] if kind == "decision" else palette["border"]
        parts.append('<g class="flow-node">')
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="16" fill="{palette["surface"]}" fill-opacity="0.94" stroke="{stroke}" stroke-width="1.6"/>')
        if node.get("badge"):
            parts.append(f'<rect x="{x+20:.1f}" y="{y+18:.1f}" width="92" height="24" rx="12" fill="{stroke}" fill-opacity="0.14" stroke="{stroke}" stroke-width="1"/>')
            parts.append(_svg_text_block(x + 66, y + 34, str(node.get("badge")), font_size=10, fill=stroke, weight=700, max_chars=14, anchor="middle"))
        parts.append(_svg_text_block(x + 20, y + 58, str(node.get("title") or node["id"]), font_size=19, fill=palette["ink"], weight=700, max_chars=24))
        if node.get("body"):
            parts.append(_svg_text_block(x + 20, y + 84, str(node.get("body")), font_size=12, fill=palette["muted"], max_chars=42))
        parts.append("</g>")

    footer = doc.get("footer") if isinstance(doc.get("footer"), dict) else {}
    note = str(footer.get("note") or "").strip()
    if note:
        fy = height - 72.0
        parts.append(f'<rect x="{inset:.1f}" y="{fy:.1f}" width="{width - 2*inset:.1f}" height="44" rx="14" fill="{palette["surface_2"]}" fill-opacity="0.92" stroke="{palette["border"]}" stroke-width="1"/>')
        parts.append(_svg_text_block(inset + 20, fy + 28, note, font_size=12, fill=palette["muted"], max_chars=140))

    parts.append("</svg>")
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render generalized spec13b flow-graph JSON to SVG.")
    ap.add_argument("--graph-json", required=True, help="Path to a flow-graph JSON payload.")
    ap.add_argument("--out", required=True, help="Output SVG path.")
    args = ap.parse_args()

    payload = json.loads(Path(args.graph_json).read_text(encoding="utf-8"))
    svg = render_flow_graph_svg(payload)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"[OK] wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
