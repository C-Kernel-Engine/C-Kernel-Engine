#!/usr/bin/env python3
"""
Build a production-facing SVG asset library for spec09 planning.

This scans shipped SVG assets and extracts:
  - canvas classes and scene-family candidates
  - reusable component tokens inferred from geometry and structure
  - compiler-owned style/effect tokens
  - aggregate vocabulary recommendations with example assets

It is intentionally heuristic. The goal is not perfect reverse compilation from
arbitrary SVG back into the scene DSL. The goal is to build an asset-grounded
library so spec09/spec10 can target the visual language we actually ship.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


SCENE_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "poster_stack": "Tall stacked infographic with multiple vertically arranged sections/cards.",
    "pipeline_lane": "Wide staged process diagram with cards, connectors, and phase dividers.",
    "comparison_span_chart": "Comparison graphic that emphasizes relative span, gap, or performance distance.",
    "table_analysis": "Structured table or matrix analysis with explicit rows and header hierarchy.",
    "dual_panel_compare": "Two-sided comparison with paired cards or mirrored panels.",
    "dashboard_cards": "Wide card dashboard with multiple metric or summary panels.",
    "timeline_flow": "Sequential flow with directional progression across steps or phases.",
    "technical_diagram": "Dense technical explainer with mixed shapes, labels, and grouped structure.",
    "architecture_map": "System or topology map with structural grouping and connections.",
}

COMPONENT_DESCRIPTIONS: dict[str, str] = {
    "header_band": "Full-width or near-full-width header treatment near the top edge.",
    "section_card": "Large rounded container card used to hold a major section.",
    "side_rail": "Thin vertical accent strip attached to a card edge.",
    "metric_bar": "Wide horizontal bar used for quantity or span comparison.",
    "table_row": "Repeated wide row blocks for tabular analysis.",
    "stage_card": "Repeated medium-sized stage boxes used in pipelines or flows.",
    "phase_divider": "Long divider line separating phases or regions.",
    "flow_arrow": "Directional connector with clear source-to-target motion.",
    "curved_connector": "Connector that bends or arcs, usually via path geometry.",
    "span_bracket": "Bracket or span marker calling out distance or total gap.",
    "floor_band": "Wide thin band that anchors a baseline, floor, or shared constraint.",
    "badge_pill": "Small rounded pill badge for status or emphasis.",
    "thesis_box": "Centered or focal box containing the key claim or thesis.",
    "conclusion_strip": "Bottom or near-bottom strip carrying a concluding message.",
    "footer_note": "Small footer annotation or attribution text near the bottom edge.",
}

STYLE_TOKEN_DESCRIPTIONS: dict[str, str] = {
    "paint:gradient_linear": "Uses linear gradients for fills or headers.",
    "paint:gradient_radial": "Uses radial gradients for spot effects.",
    "effect:drop_shadow": "Uses drop shadow filtering.",
    "effect:glow": "Uses blur/glow-style filtering.",
    "connector:arrow_marker": "Uses SVG marker-based arrowheads.",
    "connector:dashed": "Uses dashed strokes for guides, dividers, or brackets.",
    "surface:rounded_card": "Uses rounded rectangular surfaces heavily.",
    "surface:grouped_sections": "Uses grouping containers to organize sections.",
    "text:wrapped_tspan": "Uses tspans or multi-line text blocks.",
    "background:grid_pattern": "Uses a grid/pattern background.",
    "background:dark_canvas": "Uses a dark background canvas.",
    "background:light_canvas": "Uses a light or paper-like background canvas.",
    "accent:amber": "Prominent amber/yellow accent family.",
    "accent:green": "Prominent green accent family.",
    "accent:blue": "Prominent blue/cyan accent family.",
    "accent:purple": "Prominent purple/pink accent family.",
    "accent:red": "Prominent red accent family.",
    "accent:mixed": "Multiple non-neutral accent families are used together.",
}

COMPONENT_ORDER: tuple[str, ...] = (
    "header_band",
    "section_card",
    "side_rail",
    "metric_bar",
    "table_row",
    "stage_card",
    "phase_divider",
    "flow_arrow",
    "curved_connector",
    "span_bracket",
    "floor_band",
    "badge_pill",
    "thesis_box",
    "conclusion_strip",
    "footer_note",
)

STYLE_ORDER: tuple[str, ...] = (
    "paint:gradient_linear",
    "paint:gradient_radial",
    "effect:drop_shadow",
    "effect:glow",
    "connector:arrow_marker",
    "connector:dashed",
    "surface:rounded_card",
    "surface:grouped_sections",
    "text:wrapped_tspan",
    "background:grid_pattern",
    "background:dark_canvas",
    "background:light_canvas",
    "accent:amber",
    "accent:green",
    "accent:blue",
    "accent:purple",
    "accent:red",
    "accent:mixed",
)


@dataclass(frozen=True)
class RectGeom:
    x: float
    y: float
    w: float
    h: float
    rx: float
    fill: str | None
    stroke: str | None


@dataclass(frozen=True)
class LineGeom:
    x1: float
    y1: float
    x2: float
    y2: float
    dashed: bool


@dataclass(frozen=True)
class PathGeom:
    d: str
    dashed: bool
    curved: bool
    has_marker: bool


@dataclass(frozen=True)
class TextGeom:
    x: float
    y: float
    size: float
    anchor: str
    value: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_name(tag: str) -> str:
    if "}" in str(tag):
        return str(tag).split("}", 1)[1]
    return str(tag)


def _parse_style_attr(raw: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not raw:
        return out
    for chunk in str(raw).split(";"):
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            out[key] = value
    return out


def _style_map(elem: ET.Element) -> dict[str, str]:
    out = _parse_style_attr(elem.attrib.get("style"))
    for key in (
        "fill",
        "stroke",
        "stroke-width",
        "stroke-dasharray",
        "font-size",
        "font-weight",
        "font-family",
        "text-anchor",
        "marker-end",
        "marker-start",
        "filter",
    ):
        value = elem.attrib.get(key)
        if value is not None:
            out[key] = value.strip()
    return out


def _float_attr(elem: ET.Element, key: str, default: float = 0.0) -> float:
    raw = str(elem.attrib.get(key, "")).strip()
    if not raw:
        return default
    raw = raw.replace("px", "").strip()
    if raw.endswith("%"):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_viewbox(root: ET.Element) -> tuple[float, float] | None:
    raw = str(root.attrib.get("viewBox", "")).strip()
    if not raw:
        return None
    parts = re.split(r"[\s,]+", raw)
    if len(parts) != 4:
        return None
    try:
        return float(parts[2]), float(parts[3])
    except ValueError:
        return None


def _canvas_size(root: ET.Element) -> tuple[float, float]:
    from_viewbox = _parse_viewbox(root)
    if from_viewbox:
        return from_viewbox
    width = _float_attr(root, "width", 0.0)
    height = _float_attr(root, "height", 0.0)
    if width > 0 and height > 0:
        return width, height
    return 1200.0, 800.0


def _canvas_class(width: float, height: float) -> str:
    if width <= 0 or height <= 0:
        return "wide"
    ratio = width / height
    if ratio >= 1.2:
        return "wide"
    if ratio <= 0.85:
        return "tall"
    return "square"


def _hex_to_rgb(value: str) -> tuple[int, int, int] | None:
    raw = str(value).strip().lower()
    if not raw.startswith("#"):
        return None
    raw = raw[1:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) not in {6, 8}:
        return None
    try:
        return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    except ValueError:
        return None


def _rgb_to_hsv_bucket(value: tuple[int, int, int]) -> str:
    r, g, b = [c / 255.0 for c in value]
    mx = max(r, g, b)
    mn = min(r, g, b)
    delta = mx - mn
    if mx <= 0.0 or delta < 0.08:
        return "neutral"
    sat = delta / mx
    if sat < 0.18:
        return "neutral"
    if mx == r:
        hue = (60 * ((g - b) / delta) + 360) % 360
    elif mx == g:
        hue = 60 * ((b - r) / delta + 2)
    else:
        hue = 60 * ((r - g) / delta + 4)
    if hue < 25 or hue >= 345:
        return "red"
    if hue < 70:
        return "amber"
    if hue < 170:
        return "green"
    if hue < 255:
        return "blue"
    return "purple"


def _luminance(rgb: tuple[int, int, int]) -> float:
    def _channel(x: int) -> float:
        v = x / 255.0
        if v <= 0.03928:
            return v / 12.92
        return ((v + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def _extract_hex_colors(raw: str) -> list[str]:
    return [m.group(0).lower() for m in re.finditer(r"#[0-9a-fA-F]{3,8}\b", raw)]


def _sanitize_xml_text_nodes(raw: str) -> str:
    out: list[str] = []
    in_tag = False
    quote: str | None = None
    i = 0
    n = len(raw)
    entity_re = re.compile(r"^&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]+);")

    while i < n:
        ch = raw[i]
        if in_tag:
            out.append(ch)
            if quote is not None:
                if ch == quote:
                    quote = None
            else:
                if ch in {'"', "'"}:
                    quote = ch
                elif ch == ">":
                    in_tag = False
            i += 1
            continue

        if ch == "<":
            nxt = raw[i + 1] if i + 1 < n else ""
            if nxt in "/!?_" or nxt.isalpha():
                in_tag = True
                out.append(ch)
            else:
                out.append("&lt;")
            i += 1
            continue
        if ch == "&":
            if entity_re.match(raw[i:]):
                out.append(ch)
            else:
                out.append("&amp;")
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _parse_svg_root(raw: str) -> ET.Element:
    try:
        return ET.fromstring(raw)
    except ET.ParseError:
        return ET.fromstring(_sanitize_xml_text_nodes(raw))


def _dominant_accent(colors: list[str]) -> tuple[str, list[str]]:
    buckets: Counter[str] = Counter()
    for color in colors:
        rgb = _hex_to_rgb(color)
        if rgb is None:
            continue
        bucket = _rgb_to_hsv_bucket(rgb)
        if bucket == "neutral":
            continue
        buckets[bucket] += 1
    if not buckets:
        return "neutral", []
    ordered = [bucket for bucket, _ in buckets.most_common()]
    top = ordered[0]
    if len(ordered) >= 2 and buckets[ordered[1]] >= max(2, math.ceil(buckets[top] * 0.5)):
        return "mixed", ordered
    return top, ordered


def _score_scene_families(
    asset_name: str,
    canvas_class: str,
    component_hits: dict[str, bool],
    counts: dict[str, int],
) -> list[tuple[str, int]]:
    name = asset_name.lower()
    scores: Counter[str] = Counter()

    if canvas_class == "tall":
        scores["poster_stack"] += 3
    if component_hits["section_card"]:
        scores["poster_stack"] += 2
        scores["dashboard_cards"] += 1
    if counts["section_cards"] >= 3 and canvas_class == "wide":
        scores["dashboard_cards"] += 3
    if component_hits["table_row"]:
        scores["table_analysis"] += 4
        scores["poster_stack"] += 1
    if component_hits["stage_card"]:
        scores["pipeline_lane"] += 3
        scores["timeline_flow"] += 2
        scores["dashboard_cards"] += 1
    if component_hits["phase_divider"]:
        scores["pipeline_lane"] += 3
    if component_hits["flow_arrow"]:
        scores["pipeline_lane"] += 2
        scores["timeline_flow"] += 2
    if component_hits["curved_connector"]:
        scores["timeline_flow"] += 1
        scores["technical_diagram"] += 1
    if component_hits["metric_bar"]:
        scores["comparison_span_chart"] += 2
        scores["dual_panel_compare"] += 1
    if component_hits["span_bracket"]:
        scores["comparison_span_chart"] += 4
    if component_hits["thesis_box"]:
        scores["comparison_span_chart"] += 2
        scores["technical_diagram"] += 1
    if component_hits["conclusion_strip"]:
        scores["comparison_span_chart"] += 1
        scores["poster_stack"] += 1
    if counts["large_side_by_side_cards"] >= 2:
        scores["dual_panel_compare"] += 4
    if counts["groups"] >= 8:
        scores["technical_diagram"] += 2
        scores["architecture_map"] += 2
    if counts["lines"] + counts["paths"] >= 8 and counts["texts"] >= 12:
        scores["technical_diagram"] += 2
    if "pipeline" in name:
        scores["pipeline_lane"] += 3
    if "flow" in name or "timeline" in name:
        scores["timeline_flow"] += 3
    if "overview" in name and component_hits["stage_card"]:
        scores["pipeline_lane"] += 1
    if "comparison" in name or "balance" in name or "-vs-" in name or " vs " in name:
        scores["comparison_span_chart"] += 3
        scores["dual_panel_compare"] += 1
    if "memory" in name or "infographic" in name:
        scores["poster_stack"] += 2
    if "architecture" in name or "topology" in name or "network" in name:
        scores["architecture_map"] += 3
    if not scores:
        scores["technical_diagram"] = 1

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:3]


def _collect_asset_entry(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    root = _parse_svg_root(raw)
    width, height = _canvas_size(root)
    canvas_class = _canvas_class(width, height)

    tags: Counter[str] = Counter()
    rects: list[RectGeom] = []
    lines: list[LineGeom] = []
    paths: list[PathGeom] = []
    texts: list[TextGeom] = []
    gradients = 0
    radial_gradients = 0
    markers = 0
    filters = 0
    patterns = 0
    groups = 0
    tspans = 0
    rounded_rects = 0
    rx_rects = 0

    for elem in root.iter():
        tag = _local_name(elem.tag)
        tags[tag] += 1
        style = _style_map(elem)
        if tag == "rect":
            rx = max(_float_attr(elem, "rx", 0.0), _float_attr(elem, "ry", 0.0))
            rect = RectGeom(
                x=_float_attr(elem, "x", 0.0),
                y=_float_attr(elem, "y", 0.0),
                w=_float_attr(elem, "width", 0.0),
                h=_float_attr(elem, "height", 0.0),
                rx=rx,
                fill=style.get("fill"),
                stroke=style.get("stroke"),
            )
            rects.append(rect)
            if rx > 0:
                rounded_rects += 1
                rx_rects += 1
        elif tag == "line":
            lines.append(
                LineGeom(
                    x1=_float_attr(elem, "x1", 0.0),
                    y1=_float_attr(elem, "y1", 0.0),
                    x2=_float_attr(elem, "x2", 0.0),
                    y2=_float_attr(elem, "y2", 0.0),
                    dashed=bool(style.get("stroke-dasharray")),
                )
            )
        elif tag == "path":
            d = elem.attrib.get("d", "")
            has_marker = bool(style.get("marker-end") or style.get("marker-start"))
            curved = bool(re.search(r"[CQSTA]", d))
            paths.append(
                PathGeom(
                    d=d,
                    dashed=bool(style.get("stroke-dasharray")),
                    curved=curved,
                    has_marker=has_marker,
                )
            )
        elif tag == "text":
            texts.append(
                TextGeom(
                    x=_float_attr(elem, "x", 0.0),
                    y=_float_attr(elem, "y", 0.0),
                    size=_float_attr(elem, "font-size", 0.0) or _float_attr(elem, "fontSize", 0.0),
                    anchor=str(style.get("text-anchor", "start")),
                    value=" ".join("".join(elem.itertext()).split()),
                )
            )
        elif tag == "linearGradient":
            gradients += 1
        elif tag == "radialGradient":
            radial_gradients += 1
        elif tag == "marker":
            markers += 1
        elif tag == "filter":
            filters += 1
        elif tag == "pattern":
            patterns += 1
        elif tag == "g":
            groups += 1
        elif tag == "tspan":
            tspans += 1

    background_rects = [
        r
        for r in rects
        if r.w >= width * 0.85 and r.h >= height * 0.75 and r.x <= width * 0.05 and r.y <= height * 0.08
    ]
    bg_fill = next((r.fill for r in sorted(background_rects, key=lambda x: x.w * x.h, reverse=True) if r.fill), None)
    bg_rgb = _hex_to_rgb(bg_fill) if bg_fill else None
    bg_luma = _luminance(bg_rgb) if bg_rgb else None

    large_cards = [
        r for r in rects if r.w >= width * 0.35 and r.h >= height * 0.08 and r.w < width * 0.98 and r.h < height * 0.75 and r.rx > 0
    ]
    medium_cards = [
        r
        for r in rects
        if width * 0.08 <= r.w <= width * 0.22 and height * 0.14 <= r.h <= height * 0.42 and r.rx > 0
    ]
    wide_rows = [
        r for r in rects if r.w >= width * 0.60 and height * 0.02 <= r.h <= height * 0.09 and r.h < width * 0.10
    ]
    wide_bands = [
        r for r in rects if r.w >= width * 0.72 and height * 0.02 <= r.h <= height * 0.14 and r.rx > 0
    ]
    narrow_rails = [
        r for r in rects if r.w <= width * 0.04 and r.h >= height * 0.12 and r.rx > 0 and r.h > r.w * 4
    ]
    metric_bars = [
        r
        for r in rects
        if r.w >= width * 0.18 and height * 0.03 <= r.h <= height * 0.14 and r.w > r.h * 3 and r.w < width * 0.98
    ]
    pills = [
        r
        for r in rects
        if width * 0.05 <= r.w <= width * 0.45 and height * 0.02 <= r.h <= height * 0.10 and r.rx >= r.h * 0.35
    ]
    thesis_boxes = [
        r
        for r in rects
        if width * 0.22 <= r.w <= width * 0.55
        and height * 0.10 <= r.h <= height * 0.36
        and abs((r.x + r.w / 2.0) - width / 2.0) <= width * 0.18
    ]
    side_by_side_cards = []
    for i, left in enumerate(large_cards):
        for right in large_cards[i + 1 :]:
            similar_w = abs(left.w - right.w) <= width * 0.08
            similar_h = abs(left.h - right.h) <= height * 0.08
            same_row = abs(left.y - right.y) <= height * 0.08
            separated = abs((left.x + left.w) - right.x) <= width * 0.12 or abs((right.x + right.w) - left.x) <= width * 0.12
            if similar_w and similar_h and same_row and separated:
                side_by_side_cards.append((left, right))

    big_numeric_text = any(
        txt.size >= 18 and bool(re.search(r"\d", txt.value)) and any(tok in txt.value.lower() for tok in ("x", "$", "gb", "tb", "k"))
        for txt in texts
    )
    footer_text = any(txt.y >= height * 0.92 and 0 < txt.size <= 16 for txt in texts)
    phase_divider = any(
        (
            (abs(line.x1 - line.x2) <= 1.0 and abs(line.y2 - line.y1) >= height * 0.45)
            or (abs(line.y1 - line.y2) <= 1.0 and abs(line.x2 - line.x1) >= width * 0.45)
        )
        for line in lines
    )
    dashed_long_lines = [
        line
        for line in lines
        if line.dashed
        and (
            (abs(line.x1 - line.x2) <= 1.0 and abs(line.y2 - line.y1) >= height * 0.25)
            or (abs(line.y1 - line.y2) <= 1.0 and abs(line.x2 - line.x1) >= width * 0.25)
        )
    ]
    curved_with_marker = any(path.curved and path.has_marker for path in paths)
    flow_arrow = markers > 0 or "marker-end=" in raw or curved_with_marker

    component_hits = {
        "header_band": any(r.y <= height * 0.12 and r.w >= width * 0.75 and height * 0.03 <= r.h <= height * 0.18 for r in rects),
        "section_card": len(large_cards) >= 2,
        "side_rail": len(narrow_rails) >= 1,
        "metric_bar": len(metric_bars) >= 2,
        "table_row": len(wide_rows) >= 4,
        "stage_card": len(medium_cards) >= 3,
        "phase_divider": phase_divider,
        "flow_arrow": flow_arrow,
        "curved_connector": any(path.curved for path in paths),
        "span_bracket": len(dashed_long_lines) >= 2 and big_numeric_text,
        "floor_band": any(r.w >= width * 0.75 and height * 0.015 <= r.h <= height * 0.07 for r in rects),
        "badge_pill": len(pills) >= 1,
        "thesis_box": len(thesis_boxes) >= 1 and len(texts) >= 6,
        "conclusion_strip": any(r.y >= height * 0.78 and r.w >= width * 0.75 and height * 0.05 <= r.h <= height * 0.16 for r in rects),
        "footer_note": footer_text,
    }

    colors = _extract_hex_colors(raw)
    accent, accent_order = _dominant_accent(colors)
    style_hits = {
        "paint:gradient_linear": gradients > 0,
        "paint:gradient_radial": radial_gradients > 0,
        "effect:drop_shadow": "feDropShadow" in raw or filters > 0,
        "effect:glow": "feGaussianBlur" in raw or "glow" in raw.lower(),
        "connector:arrow_marker": markers > 0 or "marker-end=" in raw,
        "connector:dashed": "stroke-dasharray" in raw,
        "surface:rounded_card": rx_rects >= 4,
        "surface:grouped_sections": groups >= 4,
        "text:wrapped_tspan": tspans > 0,
        "background:grid_pattern": patterns > 0 or 'id="grid"' in raw.lower(),
        "background:dark_canvas": bg_luma is not None and bg_luma < 0.25,
        "background:light_canvas": bg_luma is not None and bg_luma > 0.70,
        "accent:amber": accent == "amber",
        "accent:green": accent == "green",
        "accent:blue": accent == "blue",
        "accent:purple": accent == "purple",
        "accent:red": accent == "red",
        "accent:mixed": accent == "mixed",
    }

    counts = {
        "rects": len(rects),
        "lines": len(lines),
        "paths": len(paths),
        "texts": len(texts),
        "groups": groups,
        "gradients": gradients,
        "markers": markers,
        "filters": filters,
        "patterns": patterns,
        "tspans": tspans,
        "section_cards": len(large_cards),
        "stage_cards": len(medium_cards),
        "large_side_by_side_cards": len(side_by_side_cards),
    }
    scene_candidates = _score_scene_families(path.name, canvas_class, component_hits, counts)
    primary_family = scene_candidates[0][0] if scene_candidates else "technical_diagram"

    return {
        "path": str(path),
        "name": path.name,
        "canvas": {
            "width": width,
            "height": height,
            "class": canvas_class,
        },
        "counts": counts,
        "component_tokens": [token for token in COMPONENT_ORDER if component_hits.get(token)],
        "components": component_hits,
        "style_tokens": [token for token in STYLE_ORDER if style_hits.get(token)],
        "styles": style_hits,
        "palette": {
            "background_fill": bg_fill,
            "accent_bucket": accent,
            "accent_bucket_order": accent_order,
            "hex_colors_top": Counter(colors).most_common(8),
        },
        "scene_family": primary_family,
        "scene_family_candidates": [{"token": token, "score": score} for token, score in scene_candidates],
        "tag_counts": dict(sorted(tags.items())),
    }


def _top_examples(bucket: dict[str, list[dict[str, Any]]], token: str, *, limit: int = 3) -> list[str]:
    rows = bucket.get(token, [])
    rows = sorted(rows, key=lambda row: (row["name"], row["path"]))
    return [row["name"] for row in rows[:limit]]


def _aggregate_library(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    canvas_counts: Counter[str] = Counter()

    for entry in entries:
        by_family[str(entry["scene_family"])].append(entry)
        canvas_counts[str(entry["canvas"]["class"])] += 1
        for token in entry["component_tokens"]:
            by_component[str(token)].append(entry)
        for token in entry["style_tokens"]:
            by_style[str(token)].append(entry)

    scene_vocab = []
    for token in sorted(by_family, key=lambda key: (-len(by_family[key]), key)):
        scene_vocab.append(
            {
                "token": token,
                "description": SCENE_FAMILY_DESCRIPTIONS.get(token, ""),
                "files": len(by_family[token]),
                "examples": _top_examples(by_family, token),
            }
        )

    component_vocab = []
    for token in COMPONENT_ORDER:
        if token not in by_component:
            continue
        component_vocab.append(
            {
                "token": token,
                "description": COMPONENT_DESCRIPTIONS[token],
                "files": len(by_component[token]),
                "examples": _top_examples(by_component, token),
            }
        )

    style_vocab = []
    for token in STYLE_ORDER:
        if token not in by_style:
            continue
        style_vocab.append(
            {
                "token": token,
                "description": STYLE_TOKEN_DESCRIPTIONS[token],
                "files": len(by_style[token]),
                "examples": _top_examples(by_style, token),
            }
        )

    spec09_seed = {
        "scene_families": [
            token["token"]
            for token in scene_vocab
            if token["token"] in {
                "poster_stack",
                "pipeline_lane",
                "comparison_span_chart",
                "table_analysis",
                "dual_panel_compare",
                "dashboard_cards",
                "timeline_flow",
            }
        ],
        "components": [
            token["token"]
            for token in component_vocab
            if token["token"]
            in {
                "header_band",
                "section_card",
                "side_rail",
                "metric_bar",
                "table_row",
                "stage_card",
                "phase_divider",
                "flow_arrow",
                "curved_connector",
                "span_bracket",
                "floor_band",
                "badge_pill",
                "thesis_box",
                "conclusion_strip",
                "footer_note",
            }
        ],
        "styles": [
            token["token"]
            for token in style_vocab
            if token["token"]
            in {
                "paint:gradient_linear",
                "effect:drop_shadow",
                "effect:glow",
                "connector:arrow_marker",
                "connector:dashed",
                "surface:rounded_card",
                "background:grid_pattern",
                "background:dark_canvas",
                "background:light_canvas",
                "accent:amber",
                "accent:green",
                "accent:blue",
                "accent:purple",
                "accent:red",
                "accent:mixed",
            }
        ],
    }

    return {
        "canvas_counts": dict(canvas_counts),
        "scene_family_vocab": scene_vocab,
        "component_vocab": component_vocab,
        "style_vocab": style_vocab,
        "spec09_seed_vocabulary": spec09_seed,
    }


def _build_markdown(doc: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Spec09 Asset Library")
    lines.append("")
    lines.append(f"- Generated at: `{doc['generated_at']}`")
    lines.append(f"- Files scanned: `{doc['files_total']}`")
    lines.append(f"- Asset glob: `{doc['assets_glob']}`")
    lines.append("")
    canvas_counts = doc["library"]["canvas_counts"]
    lines.append("## Canvas Mix")
    lines.append("")
    for key in ("wide", "tall", "square"):
        lines.append(f"- `{key}`: `{canvas_counts.get(key, 0)}`")
    lines.append("")
    lines.append("## Scene Families")
    lines.append("")
    for row in doc["library"]["scene_family_vocab"]:
        examples = ", ".join(f"`{name}`" for name in row["examples"])
        lines.append(f"- `{row['token']}`: `{row['files']}` files. {row['description']} Examples: {examples}")
    lines.append("")
    lines.append("## Components")
    lines.append("")
    for row in doc["library"]["component_vocab"]:
        examples = ", ".join(f"`{name}`" for name in row["examples"])
        lines.append(f"- `{row['token']}`: `{row['files']}` files. {row['description']} Examples: {examples}")
    lines.append("")
    lines.append("## Style Tokens")
    lines.append("")
    for row in doc["library"]["style_vocab"]:
        examples = ", ".join(f"`{name}`" for name in row["examples"])
        lines.append(f"- `{row['token']}`: `{row['files']}` files. {row['description']} Examples: {examples}")
    lines.append("")
    seed = doc["library"]["spec09_seed_vocabulary"]
    lines.append("## Spec09 Seed Vocabulary")
    lines.append("")
    lines.append(f"- Scene families: `{', '.join(seed['scene_families'])}`")
    lines.append(f"- Components: `{', '.join(seed['components'])}`")
    lines.append(f"- Styles: `{', '.join(seed['styles'])}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This library is heuristic and asset-grounded, not a reverse compiler.")
    lines.append("- Use it to hand-map 3 to 5 gold assets into the next scene DSL before building spec10+ training corpora.")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a production-oriented SVG asset library for spec09 planning.")
    ap.add_argument(
        "--assets-glob",
        action="append",
        default=["docs/site/assets/*.svg"],
        help="Glob for source SVG assets. Repeatable.",
    )
    ap.add_argument("--out-json", required=True, help="Output JSON manifest path.")
    ap.add_argument("--out-md", default=None, help="Optional markdown summary path.")
    args = ap.parse_args()

    files: list[Path] = []
    seen: set[str] = set()
    for pattern in args.assets_glob:
        for match in sorted(glob.glob(str(pattern))):
            path = Path(match)
            if path.suffix.lower() != ".svg":
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    if not files:
        raise SystemExit("ERROR: no SVG assets matched")

    entries = [_collect_asset_entry(path) for path in files]
    doc = {
        "schema": "ck.svg_asset_scene_library.v1",
        "generated_at": _now_iso(),
        "assets_glob": args.assets_glob,
        "files_total": len(entries),
        "library": _aggregate_library(entries),
        "assets": entries,
    }

    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    if args.out_md:
        out_md = Path(args.out_md).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_build_markdown(doc), encoding="utf-8")

    print(f"[OK] wrote JSON: {out_json}")
    if args.out_md:
        print(f"[OK] wrote Markdown: {Path(args.out_md).expanduser().resolve()}")
    print(f"[OK] files: {len(entries)}")
    scene_head = doc["library"]["scene_family_vocab"][:5]
    for row in scene_head:
        print(f"  - {row['token']}: files={row['files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
