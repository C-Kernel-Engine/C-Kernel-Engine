#!/usr/bin/env python3
"""Render the spec09 scene DSL into richer SVG."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LAYOUTS = {
    "comparison_span_chart",
    "dashboard_cards",
    "pipeline_lane",
    "poster_stack",
    "dual_panel_compare",
    "timeline_flow",
    "table_analysis",
}
THEMES = {"paper_editorial", "infra_dark", "signal_glow"}
TONES = {"amber", "green", "blue", "purple", "mixed"}
FRAMES = {"none", "card", "panel"}
DENSITIES = {"compact", "balanced", "airy"}
INSETS = {"sm", "md", "lg"}
GAPS = {"sm", "md", "lg"}
HERO_MODES = {"left", "center", "split"}
COLUMNS = {"1", "2", "3", "4"}
EMPHASIS = {"top", "left", "center"}
RAILS = {"accent", "muted", "none"}
BACKGROUNDS = {"grid", "mesh", "rings", "none"}
CONNECTORS = {"line", "arrow", "bracket", "curve"}
CANVAS = {"wide", "tall", "square"}
ROW_STATES = {"normal", "highlight", "success", "warning", "danger"}
PANEL_VARIANTS = {"default", "hero", "metric", "note", "warning", "success"}

_ATTRS = {
    "canvas": CANVAS,
    "layout": LAYOUTS,
    "theme": THEMES,
    "tone": TONES,
    "frame": FRAMES,
    "density": DENSITIES,
    "inset": INSETS,
    "gap": GAPS,
    "hero": HERO_MODES,
    "columns": COLUMNS,
    "emphasis": EMPHASIS,
    "rail": RAILS,
    "background": BACKGROUNDS,
    "connector": CONNECTORS,
}

_TONE_ACCENTS = {
    "amber": {"accent": "#ffb400", "accent_2": "#ffd66b", "soft": "#ffefbf"},
    "green": {"accent": "#47b475", "accent_2": "#79d79d", "soft": "#c7f4d6"},
    "blue": {"accent": "#56aefc", "accent_2": "#8bcaff", "soft": "#d8eeff"},
    "purple": {"accent": "#b576ff", "accent_2": "#e8a7ff", "soft": "#f3d8ff"},
    "mixed": {"accent": "#ffb400", "accent_2": "#56aefc", "soft": "#ffe7a8"},
}

_THEME_BASE = {
    "paper_editorial": {
        "bg": "#f3ede3",
        "bg_2": "#fbf7f0",
        "surface": "#fffaf2",
        "surface_2": "#f8efe0",
        "ink": "#17212b",
        "muted": "#64748b",
        "border": "#d8c7ad",
        "shadow": "rgba(23,33,43,0.14)",
        "grid": "rgba(114,95,72,0.10)",
    },
    "infra_dark": {
        "bg": "#0b1018",
        "bg_2": "#141d2b",
        "surface": "#121b29",
        "surface_2": "#192638",
        "ink": "#f3f7fc",
        "muted": "#92a4b8",
        "border": "#334155",
        "shadow": "rgba(2,6,23,0.46)",
        "grid": "rgba(148,163,184,0.10)",
    },
    "signal_glow": {
        "bg": "#07131f",
        "bg_2": "#11253a",
        "surface": "#0d2033",
        "surface_2": "#16314d",
        "ink": "#f8fafc",
        "muted": "#9dd5ff",
        "border": "#28577d",
        "shadow": "rgba(8,22,38,0.55)",
        "grid": "rgba(86,174,255,0.11)",
    },
}

_INSET_PX = {"sm": 36, "md": 56, "lg": 74}
_GAP_PX = {"sm": 16, "md": 26, "lg": 38}
_CARD_RADIUS = {"compact": 18, "balanced": 24, "airy": 28}
_TEXT_MAX = {"compact": 26, "balanced": 32, "airy": 38}

_WORD_MAP = {
    "api": "API",
    "amx": "AMX",
    "avx": "AVX",
    "bpe": "BPE",
    "ck": "CK",
    "cpu": "CPU",
    "ctx": "Context",
    "dsl": "DSL",
    "fp16": "FP16",
    "gqa": "GQA",
    "gpu": "GPU",
    "hbm": "HBM",
    "io": "I/O",
    "ir": "IR",
    "json": "JSON",
    "kv": "KV",
    "llm": "LLM",
    "mlp": "MLP",
    "qkv": "QKV",
    "qwen": "Qwen",
    "ram": "RAM",
    "rope": "RoPE",
    "svg": "SVG",
    "tb": "TB",
    "tpu": "TPU",
    "vram": "VRAM",
}


def _token_value(token: str, prefix: str) -> str | None:
    if token.startswith(prefix) and token.endswith("]"):
        return token[len(prefix) : -1]
    return None


def _escape(text: Any) -> str:
    return html.escape(str(text or ""), quote=False)


def _humanize(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if "->" in text:
        left, right = text.split("->", 1)
        return f"{_humanize(left)} -> {_humanize(right)}"
    if re.search(r"[A-Z]{2,}", text) and "_" not in text and "-" not in text:
        return text
    text = text.replace("-", " ").replace("_", " ")
    out: list[str] = []
    for part in text.split():
        low = part.lower()
        if low in _WORD_MAP:
            out.append(_WORD_MAP[low])
        elif re.fullmatch(r"\d+[a-zA-Z]*", part):
            out.append(part.upper())
        else:
            out.append(part.capitalize())
    return " ".join(out)


def _lookup_content_path(content: dict[str, Any] | None, path: str) -> Any | None:
    if not isinstance(content, dict) or not path:
        return None

    def _walk(root: Any, dotted: str) -> Any | None:
        cur = root
        for part in dotted.split("."):
            if isinstance(cur, dict):
                if part not in cur:
                    return None
                cur = cur[part]
            elif isinstance(cur, list):
                if not part.isdigit():
                    return None
                idx = int(part)
                if idx < 0 or idx >= len(cur):
                    return None
                cur = cur[idx]
            else:
                return None
        return cur

    found = _walk(content, path)
    if found is not None:
        return found
    slots = content.get("slots")
    if isinstance(slots, dict):
        return _walk(slots, path)
    return None


def _resolve_text_value(raw: str, content: dict[str, Any] | None = None) -> str:
    token = str(raw or "").strip()
    if not token:
        return ""
    if token.startswith("@"):
        looked_up = _lookup_content_path(content, token[1:])
        if looked_up is None:
            return f"missing:{token[1:]}"
        if isinstance(looked_up, (dict, list)):
            return json.dumps(looked_up, ensure_ascii=False)
        return str(looked_up)
    return _humanize(token)


def _payload_items(raw: str, content: dict[str, Any] | None = None) -> list[str]:
    return [_resolve_text_value(part, content) for part in str(raw or "").split("|") if part]


def _payload_parts_meta(
    raw: str,
    meta_keys: set[str] | None = None,
    *,
    content: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, str]]:
    parts: list[str] = []
    meta: dict[str, str] = {}
    for item in str(raw or "").split("|"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if meta_keys is None or key in meta_keys:
                meta[key] = str(_resolve_text_value(value, content))
                continue
        parts.append(_resolve_text_value(item, content))
    return parts, meta


def _numeric_hint(text: str) -> float:
    m = re.search(r"\d+(?:\.\d+)?", str(text))
    if not m:
        return 1.0
    try:
        return float(m.group(0))
    except ValueError:
        return 1.0


def _color_token(palette: dict[str, str], token: str | None, *, fallback: str | None = None) -> str:
    raw = str(token or "").strip().lower()
    if raw in {"amber", "warning"}:
        return "#ffb400"
    if raw in {"green", "success"}:
        return palette["success"]
    if raw == "blue":
        return "#56aefc"
    if raw == "purple":
        return "#b576ff"
    if raw in {"red", "danger"}:
        return palette["danger"]
    if raw == "muted":
        return palette["muted"]
    if raw in {"mixed", "accent"}:
        return palette["accent"]
    return fallback or palette["accent"]


def _row_style(palette: dict[str, str], state: str, accent: str | None = None) -> tuple[str, str, str, str]:
    line = _color_token(palette, accent, fallback=palette["border"])
    if state == "success":
        return palette["success"], "0.10", palette["success"], palette["success"]
    if state == "warning":
        warn = _color_token(palette, "amber")
        return warn, "0.12", warn, warn
    if state == "danger":
        return palette["danger"], "0.12", palette["danger"], palette["danger"]
    if state == "highlight":
        hi = _color_token(palette, accent, fallback=palette["accent"])
        return hi, "0.14", hi, hi
    return palette["surface_2"], "0.88", line, _color_token(palette, accent, fallback=palette["accent"])


def _variant_colors(palette: dict[str, str], variant: str, accent: str | None = None) -> tuple[str, str]:
    if variant == "warning":
        color = _color_token(palette, accent or "amber")
        return color, color
    if variant == "success":
        color = _color_token(palette, accent or "green")
        return color, color
    if variant == "note":
        color = _color_token(palette, accent or "blue")
        return color, color
    if variant == "hero":
        return palette["accent_2"], palette["accent"]
    if variant == "metric":
        return _color_token(palette, accent, fallback=palette["accent"]), _color_token(palette, accent, fallback=palette["accent"])
    return _color_token(palette, accent, fallback=palette["accent"]), palette["accent"]


def _legend_items(raw: str, content: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for chunk in str(raw or "").split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            tone, label = chunk.split("=", 1)
            items.append((tone.strip().lower(), _resolve_text_value(label, content)))
        else:
            items.append(("", _resolve_text_value(chunk, content)))
    return items


def _scene_defaults() -> dict[str, Any]:
    return {
        "canvas": "wide",
        "layout": "",
        "theme": "infra_dark",
        "tone": "blue",
        "frame": "card",
        "density": "balanced",
        "inset": "md",
        "gap": "md",
        "hero": "left",
        "columns": "1",
        "emphasis": "top",
        "rail": "none",
        "background": "none",
        "connector": "line",
        "topic": "",
        "components": [],
        "components_by_name": {},
        "_content": {},
    }


def _validate_layout_components(scene: dict[str, Any]) -> None:
    by_name = scene["components_by_name"]
    layout = str(scene["layout"])

    def count(name: str) -> int:
        return len(by_name.get(name, []))

    missing: list[str] = []
    if count("header_band") < 1:
        missing.append("header_band<1")
    if layout == "comparison_span_chart":
        if count("compare_bar") + count("metric_bar") < 2:
            missing.append("compare_bar|metric_bar<2")
        if count("thesis_box") < 1:
            missing.append("thesis_box<1")
    elif layout == "dashboard_cards":
        if count("section_card") + count("callout_card") < 2:
            missing.append("section_card|callout_card<2")
    elif layout == "pipeline_lane":
        if count("stage_card") < 3:
            missing.append("stage_card<3")
        if count("flow_arrow") < 2:
            missing.append("flow_arrow<2")
    elif layout == "poster_stack":
        poster_blocks = count("section_card") + count("compare_bar") + count("metric_bar") + count("table_row") + count("compare_panel")
        if poster_blocks < 4:
            missing.append("poster_content<4")
    elif layout == "dual_panel_compare":
        if count("compare_panel") + count("section_card") < 2:
            missing.append("compare_panel|section_card<2")
    elif layout == "timeline_flow":
        if count("stage_card") < 3:
            missing.append("stage_card<3")
        if count("flow_arrow") < 2:
            missing.append("flow_arrow<2")
    elif layout == "table_analysis":
        if count("table_row") < 4:
            missing.append("table_row<4")
    if missing:
        raise ValueError(f"scene missing required components for {layout}: {', '.join(missing)}")


def _parse_scene_document(text: str) -> dict[str, Any]:
    tokens = [tok.strip() for tok in str(text or "").split() if tok.strip()]
    if "[scene]" in tokens:
        tokens = tokens[tokens.index("[scene]") :]
    if not tokens or tokens[0] != "[scene]":
        raise ValueError("spec09 scene DSL must start with [scene]")
    if "[/scene]" not in tokens:
        raise ValueError("spec09 scene DSL must end with [/scene]")
    tokens = tokens[: tokens.index("[/scene]") + 1]

    scene = _scene_defaults()
    components: list[tuple[str, str]] = []
    for token in tokens[1:]:
        if token == "[/scene]":
            break
        handled = False
        for key, allowed in _ATTRS.items():
            value = _token_value(token, f"[{key}:")
            if value is None:
                continue
            if value not in allowed:
                raise ValueError(f"unsupported {key}: {value}")
            scene[key] = value
            handled = True
            break
        if handled:
            continue
        value = _token_value(token, "[topic:")
        if value is not None:
            scene["topic"] = value
            continue
        if token.startswith("[") and token.endswith("]") and ":" in token:
            comp, raw = token[1:-1].split(":", 1)
            components.append((comp, raw))

    layout = str(scene["layout"])
    if layout not in LAYOUTS:
        raise ValueError(f"unsupported scene layout: {layout or '<empty>'}")
    scene["components"] = components
    by_name: dict[str, list[str]] = defaultdict(list)
    for comp, raw in components:
        by_name[comp].append(raw)
    scene["components_by_name"] = dict(by_name)
    _validate_layout_components(scene)
    return scene


def _palette(scene: dict[str, Any]) -> dict[str, str]:
    theme = _THEME_BASE[str(scene["theme"])]
    tone = _TONE_ACCENTS[str(scene["tone"])]
    return {
        **theme,
        **tone,
        "surface_edge": tone["accent_2"] if scene["theme"] != "paper_editorial" else theme["border"],
        "hero_grad_a": tone["accent"],
        "hero_grad_b": tone["accent_2"],
        "panel_glow": tone["soft"],
        "success": "#47b475",
        "danger": "#f87171",
    }


def _canvas_size(canvas: str) -> tuple[int, int]:
    return {
        "wide": (1200, 780),
        "tall": (960, 1420),
        "square": (1024, 1024),
    }.get(str(canvas or "wide"), (1200, 780))


def _wrap_lines(text: str, *, max_chars: int) -> list[str]:
    words = str(text or "").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _svg_text_block(
    x: float,
    y: float,
    text: str,
    *,
    font_size: int,
    fill: str,
    weight: int = 500,
    max_chars: int = 30,
    line_height: float = 1.25,
    anchor: str = "start",
    family: str = "IBM Plex Sans, Segoe UI, sans-serif",
) -> str:
    lines = _wrap_lines(text, max_chars=max_chars)
    body = [
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{font_size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" font-family="{family}">'
    ]
    for idx, line in enumerate(lines):
        dy = "0" if idx == 0 else f"{font_size * line_height:.1f}"
        body.append(f'<tspan x="{x:.1f}" dy="{dy}">{_escape(line)}</tspan>')
    body.append("</text>")
    return "".join(body)


def _defs(scene: dict[str, Any], palette: dict[str, str]) -> str:
    arrow = ""
    if scene["connector"] in {"arrow", "curve"}:
        arrow = (
            '<marker id="arrowHead" markerWidth="12" markerHeight="8" refX="10" refY="4" orient="auto">'
            f'<path d="M0,0 L12,4 L0,8 z" fill="{palette["accent"]}"/></marker>'
        )
    return (
        "<defs>"
        f'<linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="{palette["bg"]}"/>'
        f'<stop offset="100%" stop-color="{palette["bg_2"]}"/>'
        "</linearGradient>"
        f'<linearGradient id="heroGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%" stop-color="{palette["hero_grad_a"]}"/>'
        f'<stop offset="100%" stop-color="{palette["hero_grad_b"]}"/>'
        "</linearGradient>"
        f'<linearGradient id="panelGrad" x1="0%" y1="0%" x2="0%" y2="100%">'
        f'<stop offset="0%" stop-color="{palette["surface"]}"/>'
        f'<stop offset="100%" stop-color="{palette["surface_2"]}"/>'
        "</linearGradient>"
        f'<filter id="softShadow" x="-20%" y="-20%" width="140%" height="160%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{palette["shadow"]}"/>'
        "</filter>"
        f'<filter id="accentGlow" x="-30%" y="-30%" width="160%" height="180%">'
        f'<feDropShadow dx="0" dy="0" stdDeviation="8" flood-color="{palette["accent"]}"/>'
        "</filter>"
        f"{arrow}"
        "</defs>"
    )


def _background_motif(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    style = str(scene["background"])
    if style == "none":
        return ""
    if style == "grid":
        lines: list[str] = []
        for x in range(0, width + 1, 64):
            lines.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" stroke="{palette["grid"]}" stroke-width="1"/>')
        for y in range(0, height + 1, 64):
            lines.append(f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" stroke="{palette["grid"]}" stroke-width="1"/>')
        return '<g opacity="0.9">' + "".join(lines) + "</g>"
    if style == "mesh":
        return (
            '<g opacity="0.24">'
            f'<circle cx="{width * 0.18:.1f}" cy="{height * 0.24:.1f}" r="180" fill="{palette["panel_glow"]}"/>'
            f'<circle cx="{width * 0.88:.1f}" cy="{height * 0.18:.1f}" r="140" fill="{palette["soft"]}"/>'
            f'<circle cx="{width * 0.68:.1f}" cy="{height * 0.84:.1f}" r="220" fill="{palette["accent_2"]}" opacity="0.20"/>'
            "</g>"
        )
    return (
        '<g opacity="0.20">'
        f'<circle cx="{width * 0.82:.1f}" cy="{height * 0.20:.1f}" r="180" fill="none" stroke="{palette["grid"]}" stroke-width="2"/>'
        f'<circle cx="{width * 0.82:.1f}" cy="{height * 0.20:.1f}" r="250" fill="none" stroke="{palette["grid"]}" stroke-width="1.5"/>'
        f'<circle cx="{width * 0.14:.1f}" cy="{height * 0.82:.1f}" r="120" fill="none" stroke="{palette["accent"]}" stroke-width="2"/>'
        "</g>"
    )


def _panel_shell(x: float, y: float, width: float, height: float, scene: dict[str, Any], palette: dict[str, str]) -> str:
    radius = _CARD_RADIUS[str(scene["density"])]
    stroke = palette["surface_edge"] if scene["frame"] != "none" else "none"
    sw = "1.5" if scene["frame"] != "none" else "0"
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="{radius}" '
        f'fill="url(#panelGrad)" stroke="{stroke}" stroke-width="{sw}" filter="url(#softShadow)"/>'
    )


def _accent_rail(scene: dict[str, Any], palette: dict[str, str], x: float, y: float, height: float) -> str:
    if scene["rail"] == "none":
        return ""
    color = palette["accent"] if scene["rail"] == "accent" else palette["muted"]
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="8" height="{height:.1f}" rx="4" '
        f'fill="{color}" filter="url(#accentGlow)"/>'
    )


def _component_values(scene: dict[str, Any], name: str) -> list[str]:
    return list(scene["components_by_name"].get(name, []))


def _header_band(scene: dict[str, Any], palette: dict[str, str], width: int, inset: float) -> tuple[str, float]:
    raw = _component_values(scene, "header_band")[0]
    parts = _payload_items(raw, scene.get("_content"))
    kicker = parts[0] if len(parts) > 2 else ""
    title = parts[-2] if len(parts) > 2 else parts[0] if parts else _humanize(scene.get("topic", "Untitled"))
    subtitle = parts[-1] if len(parts) > 1 else _humanize(scene.get("topic", ""))
    anchor = "middle" if scene["hero"] == "center" else "start"
    text_x = width / 2 if anchor == "middle" else inset + 12
    body: list[str] = []
    if scene["frame"] != "none":
        body.append(
            f'<rect x="{inset:.1f}" y="{inset:.1f}" width="{width - 2 * inset:.1f}" height="108" rx="22" '
            f'fill="{palette["surface"]}" fill-opacity="0.18" stroke="{palette["border"]}" stroke-width="1.2"/>'
        )
    accent_x = text_x - 110 if anchor == "middle" else inset + 12
    body.append(
        f'<rect x="{accent_x:.1f}" y="{inset + 12:.1f}" width="220" height="8" rx="4" fill="url(#heroGrad)" filter="url(#accentGlow)"/>'
    )
    if kicker:
        body.append(_svg_text_block(text_x, inset + 40, kicker, font_size=13, fill=palette["accent"], weight=700, max_chars=32, anchor=anchor))
    body.append(_svg_text_block(text_x, inset + 72, title, font_size=30, fill=palette["ink"], weight=700, max_chars=28, anchor=anchor))
    if subtitle:
        body.append(_svg_text_block(text_x, inset + 104, subtitle, font_size=15, fill=palette["muted"], weight=500, max_chars=64, anchor=anchor))
    return "".join(body), inset + 138


def _footer_note(scene: dict[str, Any], palette: dict[str, str], width: int, y: float) -> str:
    values = _component_values(scene, "footer_note")
    if not values:
        return ""
    items = _payload_items(values[0], scene.get("_content"))
    text = items[0] if items else _resolve_text_value(values[0], scene.get("_content"))
    return _svg_text_block(width / 2, y, text, font_size=13, fill=palette["muted"], max_chars=88, anchor="middle")


def _badge_pill(text: str, palette: dict[str, str], x: float, y: float, width: float = 220.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="46" rx="23" fill="url(#heroGrad)" filter="url(#accentGlow)"/>'
        + _svg_text_block(x + width / 2, y + 30, text, font_size=15, fill=_THEME_BASE["paper_editorial"]["ink"], weight=700, max_chars=22, anchor="middle")
    )


def _render_section_card(
    scene: dict[str, Any],
    palette: dict[str, str],
    x: float,
    y: float,
    width: float,
    height: float,
    raw: str,
    *,
    rail: bool = False,
) -> str:
    items, meta = _payload_parts_meta(raw, {"variant", "accent"}, content=scene.get("_content"))
    title = items[0] if items else ""
    lead = items[1] if len(items) > 1 else ""
    tail = items[2] if len(items) > 2 else ""
    variant = str(meta.get("variant", "default")).lower()
    accent = meta.get("accent")
    edge, accent_fill = _variant_colors(palette, variant, accent)
    body = [_panel_shell(x, y, width, height, scene, palette)]
    if rail:
        body.append(_accent_rail(scene, palette, x + 20, y + 20, height - 40))
    body.append(f'<rect x="{x + 18:.1f}" y="{y + 18:.1f}" width="{width - 36:.1f}" height="10" rx="5" fill="{accent_fill}" opacity="0.9"/>')
    if variant in {"warning", "success", "note"}:
        body.append(f'<rect x="{x + width - 74:.1f}" y="{y + 18:.1f}" width="56" height="24" rx="12" fill="{edge}" fill-opacity="0.14" stroke="{edge}" stroke-width="1"/>')
        body.append(_svg_text_block(x + width - 46, y + 35, variant.upper(), font_size=9, fill=edge, weight=700, max_chars=8, anchor="middle"))
    body.append(_svg_text_block(x + 44, y + 58, title, font_size=24, fill=palette["ink"], weight=700, max_chars=20))
    if lead:
        size = 18 if len(lead) > 18 else 28
        body.append(_svg_text_block(x + 44, y + 112, lead, font_size=size, fill=edge, weight=700, max_chars=28))
    if tail:
        body.append(_svg_text_block(x + 44, y + height - 54, tail, font_size=15, fill=palette["muted"], max_chars=44))
    return "".join(body)


def _render_poster_stack(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX[str(scene["inset"])]
    gap = _GAP_PX[str(scene["gap"])]
    body: list[str] = []
    header, y = _header_band(scene, palette, width, inset)
    body.append(header)
    cards = _component_values(scene, "section_card")
    compare_bars = _component_values(scene, "compare_bar")
    compare_bars = compare_bars if compare_bars else _component_values(scene, "metric_bar")
    table_header = _component_values(scene, "table_header")
    table_rows = _component_values(scene, "table_row")
    compare_panels = _component_values(scene, "compare_panel")
    callouts = _component_values(scene, "callout_card")
    badge = _component_values(scene, "badge_pill")
    footer = _component_values(scene, "footer_note")
    badge_space = 66 if badge or callouts else 0
    footer_space = 28 if footer else 0
    current_y = y

    if cards:
        hero_h = 188.0
        body.append(_render_section_card(scene, palette, inset, current_y, width - 2 * inset, hero_h, cards[0], rail=(scene["rail"] != "none")))
        current_y += hero_h + gap

    if compare_bars:
        panel_h = 228.0
        body.append(_panel_shell(inset, current_y, width - 2 * inset, panel_h, scene, palette))
        if len(cards) > 1:
            title_items, _ = _payload_parts_meta(cards[1], {"variant", "accent"}, content=scene.get("_content"))
            if title_items:
                body.append(_svg_text_block(inset + 34, current_y + 42, title_items[0], font_size=24, fill=palette["ink"], weight=700, max_chars=24))
            if len(title_items) > 1:
                body.append(_svg_text_block(inset + 34, current_y + 66, title_items[1], font_size=13, fill=palette["muted"], max_chars=48))
        inner_x = inset + 36
        inner_w = width - 2 * inset - 72
        base_y = current_y + 94
        bar_h = 46.0
        bar_gap = 34.0
        parsed_bars = [
            _payload_parts_meta(raw, {"accent", "note"}, content=scene.get("_content"))
            for raw in compare_bars[:2]
        ]
        numeric_values = [_numeric_hint(parts[1] if len(parts) > 1 else "1") for parts, _ in parsed_bars]
        max_value = max(numeric_values + [1.0])
        for idx, (parts, meta) in enumerate(parsed_bars):
            by = base_y + idx * (bar_h + bar_gap)
            color = _color_token(palette, meta.get("accent"), fallback=palette["accent"] if idx == 0 else palette["success"])
            body.append(f'<rect x="{inner_x:.1f}" y="{by:.1f}" width="{inner_w:.1f}" height="{bar_h:.1f}" rx="12" fill="{palette["surface_2"]}" fill-opacity="0.80" stroke="{palette["border"]}" stroke-width="1"/>')
            value = numeric_values[idx]
            fill_w = max(110.0, inner_w * (value / max_value))
            body.append(f'<rect x="{inner_x:.1f}" y="{by:.1f}" width="{fill_w:.1f}" height="{bar_h:.1f}" rx="12" fill="{color}" fill-opacity="0.20" stroke="{color}" stroke-width="1.2"/>')
            label = parts[0] if parts else f"Bar {idx + 1}"
            amount = parts[1] if len(parts) > 1 else ""
            caption = parts[2] if len(parts) > 2 else ""
            body.append(_svg_text_block(inner_x + 16, by + 20, label, font_size=14, fill=palette["ink"], weight=700, max_chars=26))
            if amount:
                body.append(_svg_text_block(inner_x + 16, by + 38, amount, font_size=18, fill=color, weight=700, max_chars=22, family="IBM Plex Mono, SFMono-Regular, monospace"))
            if caption:
                body.append(_svg_text_block(inner_x + inner_w - 14, by + 30, caption, font_size=13, fill=palette["muted"], max_chars=24, anchor="end"))
        if len(parsed_bars) == 2:
            ratio = max(numeric_values) / max(min(numeric_values), 1.0)
            ratio_label = f"{ratio:.1f}x More Capacity" if ratio < 10 else f"{ratio:.0f}x More Capacity"
            body.append(_svg_text_block(width / 2, current_y + panel_h - 24, ratio_label, font_size=18, fill=palette["success"], weight=700, max_chars=28, anchor="middle"))
        current_y += panel_h + gap
    elif len(cards) > 1:
        mid_h = 176.0
        body.append(_render_section_card(scene, palette, inset, current_y, width - 2 * inset, mid_h, cards[1], rail=False))
        current_y += mid_h + gap

    if table_rows:
        row_h = 48.0
        title_space = 54.0 if len(cards) > 2 else 18.0
        header_space = 54.0 if table_header else 0.0
        panel_h = title_space + header_space + row_h * len(table_rows) + gap * 0.35 * max(0, len(table_rows) - 1) + 28.0
        body.append(_panel_shell(inset, current_y, width - 2 * inset, panel_h, scene, palette))
        cursor_y = current_y + 24
        if len(cards) > 2:
            title_items, _ = _payload_parts_meta(cards[2], {"variant", "accent"}, content=scene.get("_content"))
            if title_items:
                body.append(_svg_text_block(inset + 34, cursor_y + 18, title_items[0], font_size=22, fill=palette["ink"], weight=700, max_chars=26))
            if len(title_items) > 1:
                body.append(_svg_text_block(inset + 34, cursor_y + 40, title_items[1], font_size=12, fill=palette["muted"], max_chars=82))
            cursor_y += 54
        if table_header:
            labels = _payload_items(table_header[0], scene.get("_content"))
            hx = inset + 26
            hy = cursor_y
            body.append(f'<rect x="{hx:.1f}" y="{hy:.1f}" width="{width - 2 * inset - 52:.1f}" height="38" rx="12" fill="{palette["surface_2"]}" fill-opacity="0.95" stroke="{palette["border"]}" stroke-width="1"/>')
            col_x = [inset + 94, width * 0.58, width - inset - 92]
            anchors = ["middle", "middle", "middle"]
            for idx, label in enumerate(labels[:3]):
                body.append(_svg_text_block(col_x[idx], hy + 24, label, font_size=12, fill=palette["ink"], weight=700, max_chars=18, anchor=anchors[idx]))
            cursor_y += 54
        for idx, raw in enumerate(table_rows):
            items, meta = _payload_parts_meta(raw, {"state", "accent"}, content=scene.get("_content"))
            state = str(meta.get("state", "normal")).lower()
            if state not in ROW_STATES:
                state = "normal"
            fill, fill_op, stroke, accent = _row_style(palette, state, meta.get("accent"))
            ry = cursor_y + idx * (row_h + gap * 0.35)
            body.append(f'<rect x="{inset + 26:.1f}" y="{ry:.1f}" width="{width - 2 * inset - 52:.1f}" height="{row_h:.1f}" rx="12" fill="{fill}" fill-opacity="{fill_op}" stroke="{stroke}" stroke-width="1"/>')
            if state in {"highlight", "success", "warning", "danger"}:
                body.append(f'<rect x="{inset + 26:.1f}" y="{ry:.1f}" width="7" height="{row_h:.1f}" rx="3.5" fill="{accent}"/>')
            if items:
                body.append(_svg_text_block(inset + 82, ry + 30, items[0], font_size=15, fill=palette["ink"], weight=700, max_chars=18, anchor="middle"))
            if len(items) > 1:
                body.append(_svg_text_block(width * 0.58, ry + 30, items[1], font_size=15, fill=accent, weight=700, max_chars=18, anchor="middle", family="IBM Plex Mono, SFMono-Regular, monospace"))
            if len(items) > 2:
                body.append(_svg_text_block(width - inset - 92, ry + 30, items[2], font_size=15, fill=accent if state != "normal" else palette["success"], weight=700, max_chars=18, anchor="middle"))
            if len(items) > 3:
                body.append(_svg_text_block(inset + 42, ry + row_h - 10, items[3], font_size=11, fill=palette["muted"], max_chars=80))
        current_y += panel_h + gap
    elif len(cards) > 2:
        mid_h = 176.0
        body.append(_render_section_card(scene, palette, inset, current_y, width - 2 * inset, mid_h, cards[2], rail=False))
        current_y += mid_h + gap

    if compare_panels:
        panel_w = (width - 2 * inset - gap) / 2
        panel_h = 192.0
        for idx, raw in enumerate(compare_panels[:2]):
            x = inset + idx * (panel_w + gap)
            items, meta = _payload_parts_meta(raw, {"variant", "accent"}, content=scene.get("_content"))
            title = items[0] if items else ""
            value = items[1] if len(items) > 1 else ""
            caption = items[2] if len(items) > 2 else ""
            variant = str(meta.get("variant", "metric")).lower()
            edge, accent_fill = _variant_colors(palette, variant, meta.get("accent"))
            tint = edge if variant in {"warning", "success"} else palette["surface_2"]
            body.append(f'<rect x="{x:.1f}" y="{current_y:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" rx="22" fill="{tint}" fill-opacity="0.14" stroke="{edge}" stroke-width="1.5" filter="url(#softShadow)"/>')
            body.append(f'<rect x="{x:.1f}" y="{current_y:.1f}" width="8" height="{panel_h:.1f}" rx="4" fill="{accent_fill}"/>')
            body.append(_svg_text_block(x + 34, current_y + 44, title, font_size=20, fill=edge, weight=700, max_chars=20))
            if value:
                body.append(_svg_text_block(x + 34, current_y + 108, value, font_size=34, fill=palette["ink"], weight=700, max_chars=14, family="IBM Plex Mono, SFMono-Regular, monospace"))
            if caption:
                body.append(_svg_text_block(x + 34, current_y + panel_h - 28, caption, font_size=14, fill=palette["muted"], max_chars=30))
        current_y += panel_h + gap
    elif len(cards) > 3:
        mid_h = 176.0
        body.append(_render_section_card(scene, palette, inset, current_y, width - 2 * inset, mid_h, cards[3], rail=False))
        current_y += mid_h + gap

    if callouts:
        items, meta = _payload_parts_meta(callouts[0], {"accent"}, content=scene.get("_content"))
        label = items[0] if items else ""
        note = items[1] if len(items) > 1 else ""
        color = _color_token(palette, meta.get("accent"), fallback=palette["accent"])
        body.append(f'<rect x="{width / 2 - 250:.1f}" y="{current_y + 4:.1f}" width="500" height="64" rx="32" fill="{color}" filter="url(#softShadow)"/>')
        body.append(_svg_text_block(width / 2, current_y + 31, label, font_size=20, fill=_THEME_BASE["paper_editorial"]["ink"], weight=700, max_chars=28, anchor="middle"))
        if note:
            body.append(_svg_text_block(width / 2, current_y + 51, note, font_size=11, fill=_THEME_BASE["paper_editorial"]["ink"], max_chars=46, anchor="middle"))
        current_y += 78
    elif badge:
        badge_items = _payload_items(badge[0], scene.get("_content"))
        text = badge_items[0] if badge_items else _resolve_text_value(badge[0], scene.get("_content"))
        body.append(_badge_pill(text, palette, width / 2 - 200, current_y + 8, 400))
    if footer:
        body.append(_footer_note(scene, palette, width, height - 24))
    return "".join(body)


def _stage_visual_markup(title: str, x: float, y: float, width: float, stroke: str) -> str:
    low = str(title or "").lower()
    cx = x + width / 2
    if "config" in low:
        return (
            f'<rect x="{cx - 24:.1f}" y="{y + 8:.1f}" width="48" height="56" rx="6" fill="rgba(0,0,0,0.12)" stroke="{stroke}" stroke-width="1.2"/>'
            f'<rect x="{cx - 24:.1f}" y="{y + 8:.1f}" width="48" height="14" rx="4" fill="{stroke}" fill-opacity="0.20"/>'
            f'<line x1="{cx - 14:.1f}" y1="{y + 34:.1f}" x2="{cx + 14:.1f}" y2="{y + 34:.1f}" stroke="{stroke}" stroke-width="1"/>'
            f'<line x1="{cx - 14:.1f}" y1="{y + 44:.1f}" x2="{cx + 8:.1f}" y2="{y + 44:.1f}" stroke="{stroke}" stroke-width="1"/>'
        )
    if "parse" in low:
        return (
            f'<circle cx="{cx:.1f}" cy="{y + 40:.1f}" r="22" fill="none" stroke="{stroke}" stroke-width="2"/>'
            f'<circle cx="{cx:.1f}" cy="{y + 40:.1f}" r="11" fill="{stroke}" fill-opacity="0.22"/>'
            + _svg_text_block(cx, y + 44, "IR", font_size=10, fill=stroke, weight=700, anchor="middle")
        )
    if "generate" in low:
        return (
            f'<rect x="{cx - 28:.1f}" y="{y + 8:.1f}" width="56" height="58" rx="6" fill="rgba(0,0,0,0.12)" stroke="{stroke}" stroke-width="1.2"/>'
            f'<line x1="{cx - 18:.1f}" y1="{y + 24:.1f}" x2="{cx + 12:.1f}" y2="{y + 24:.1f}" stroke="{stroke}" stroke-width="1.2"/>'
            f'<line x1="{cx - 18:.1f}" y1="{y + 36:.1f}" x2="{cx + 16:.1f}" y2="{y + 36:.1f}" stroke="{stroke}" stroke-width="1.2"/>'
            f'<line x1="{cx - 18:.1f}" y1="{y + 48:.1f}" x2="{cx + 8:.1f}" y2="{y + 48:.1f}" stroke="{stroke}" stroke-width="1.2"/>'
        )
    if "weight" in low:
        return "".join(
            f'<rect x="{cx - 28 + idx * 4:.1f}" y="{y + 10 + idx * 12:.1f}" width="{56 - idx * 8:.1f}" height="10" rx="5" fill="{stroke}" fill-opacity="{0.24 + idx * 0.10:.2f}" stroke="{stroke}" stroke-width="1"/>'
            for idx in range(3)
        )
    if "compile" in low:
        return (
            f'<rect x="{cx - 32:.1f}" y="{y + 10:.1f}" width="64" height="48" rx="7" fill="rgba(0,0,0,0.18)" stroke="{stroke}" stroke-width="1.2"/>'
            + _svg_text_block(cx - 18, y + 31, "$", font_size=11, fill=stroke, weight=700)
            + _svg_text_block(cx - 4, y + 31, "gcc -O3", font_size=10, fill=stroke, max_chars=10)
            + _svg_text_block(cx - 18, y + 45, "-mavx512", font_size=9, fill=stroke, max_chars=10)
        )
    if "run" in low:
        return f'<polygon points="{cx - 12:.1f},{y + 14:.1f} {cx - 12:.1f},{y + 62:.1f} {cx + 20:.1f},{y + 38:.1f}" fill="{stroke}" fill-opacity="0.72"/>'
    return f'<circle cx="{cx:.1f}" cy="{y + 38:.1f}" r="18" fill="{stroke}" fill-opacity="0.22" stroke="{stroke}" stroke-width="1.5"/>'


def _render_comparison_span_chart(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX[str(scene["inset"])]
    body: list[str] = []
    header, y = _header_band(scene, palette, width, inset)
    body.append(header)
    left_x = inset + 20
    right_x = width - inset - 250
    top_y = y + 12
    col_w = 210.0
    compare_bars = _component_values(scene, "compare_bar")
    m_bars = compare_bars if compare_bars else _component_values(scene, "metric_bar")
    left, left_meta = _payload_parts_meta(m_bars[0], {"accent", "note"}, content=scene.get("_content")) if len(m_bars) > 0 else ([], {})
    right, right_meta = _payload_parts_meta(m_bars[1], {"accent", "note"}, content=scene.get("_content")) if len(m_bars) > 1 else ([], {})
    left_h = 326.0
    right_h = 216.0
    body.append(_panel_shell(left_x, top_y, col_w, left_h, scene, palette))
    body.append(_panel_shell(right_x, top_y + 120, col_w, right_h, scene, palette))
    left_color = _color_token(palette, left_meta.get("accent"), fallback=palette["accent"])
    right_color = _color_token(palette, right_meta.get("accent") or "green", fallback=palette["success"])
    lv = _numeric_hint(left[1] if len(left) > 1 else "1")
    rv = _numeric_hint(right[1] if len(right) > 1 else "1")
    mx = max(lv, rv, 1.0)
    left_bar_h = max(80.0, left_h * 0.18 + left_h * 0.52 * (lv / mx))
    right_bar_h = max(60.0, right_h * 0.18 + right_h * 0.48 * (rv / mx))
    body.append(f'<rect x="{left_x:.1f}" y="{top_y + left_h - left_bar_h:.1f}" width="{col_w:.1f}" height="{left_bar_h:.1f}" rx="18" fill="{left_color}" fill-opacity="0.16" stroke="{left_color}" stroke-width="1.5"/>')
    body.append(f'<rect x="{right_x:.1f}" y="{top_y + 120 + right_h - right_bar_h:.1f}" width="{col_w:.1f}" height="{right_bar_h:.1f}" rx="18" fill="{right_color}" fill-opacity="0.16" stroke="{right_color}" stroke-width="1.5"/>')
    if left:
        body.append(_svg_text_block(left_x + col_w / 2, top_y + 42, left[0], font_size=18, fill=left_color, weight=700, max_chars=18, anchor="middle"))
        body.append(_svg_text_block(left_x + col_w / 2, top_y + 174, left[1] if len(left) > 1 else "Span", font_size=24, fill=palette["ink"], weight=700, max_chars=14, anchor="middle", family="IBM Plex Mono, SFMono-Regular, monospace"))
        if len(left) > 2:
            body.append(_svg_text_block(left_x + col_w / 2, top_y + left_h - 26, left[2], font_size=13, fill=palette["muted"], max_chars=18, anchor="middle"))
        if left_meta.get("note"):
            body.append(_svg_text_block(left_x + col_w / 2, top_y + left_h - 50, left_meta["note"], font_size=11, fill=palette["muted"], max_chars=18, anchor="middle"))
    if right:
        body.append(_svg_text_block(right_x + col_w / 2, top_y + 158, right[0], font_size=18, fill=right_color, weight=700, max_chars=18, anchor="middle"))
        body.append(_svg_text_block(right_x + col_w / 2, top_y + 252, right[1] if len(right) > 1 else "Span", font_size=24, fill=palette["ink"], weight=700, max_chars=14, anchor="middle", family="IBM Plex Mono, SFMono-Regular, monospace"))
        if len(right) > 2:
            body.append(_svg_text_block(right_x + col_w / 2, top_y + 308, right[2], font_size=13, fill=palette["muted"], max_chars=18, anchor="middle"))
        if right_meta.get("note"):
            body.append(_svg_text_block(right_x + col_w / 2, top_y + 284, right_meta["note"], font_size=11, fill=palette["muted"], max_chars=18, anchor="middle"))
    axis = _component_values(scene, "axis")
    if axis:
        axis_items = _payload_items(axis[0], scene.get("_content"))
        axis_label = axis_items[0] if axis_items else "Axis"
        axis_note = axis_items[1] if len(axis_items) > 1 else ""
        ax_x = width / 2
        body.append(f'<line x1="{ax_x:.1f}" y1="{top_y + 20:.1f}" x2="{ax_x:.1f}" y2="{height - 152:.1f}" stroke="{palette["grid"]}" stroke-width="2"/>')
        body.append(_svg_text_block(ax_x, height - 118, axis_label, font_size=11, fill=palette["muted"], weight=700, max_chars=18, anchor="middle"))
        if axis_note:
            body.append(_svg_text_block(ax_x, height - 100, axis_note, font_size=10, fill=palette["muted"], max_chars=18, anchor="middle"))
    thesis = _component_values(scene, "thesis_box")
    thesis_items = _payload_items(thesis[0], scene.get("_content")) if thesis else []
    thesis_x = width / 2 - 210
    thesis_y = y + 108
    thesis_w = 420.0
    thesis_h = 216.0
    body.append(_panel_shell(thesis_x, thesis_y, thesis_w, thesis_h, scene, palette))
    if thesis_items:
        body.append(_svg_text_block(width / 2, thesis_y + 44, thesis_items[0], font_size=18, fill=palette["accent"], weight=700, max_chars=34, anchor="middle"))
        for idx, line in enumerate(thesis_items[1:], start=1):
            body.append(_svg_text_block(width / 2, thesis_y + 44 + idx * 48, line, font_size=15, fill=palette["muted"], max_chars=52, anchor="middle"))
    brackets = _component_values(scene, "span_bracket")
    for idx, raw in enumerate(brackets[:2]):
        items = _payload_items(raw, scene.get("_content"))
        bx = left_x + col_w + 28 if idx == 0 else right_x - 44
        top = top_y if idx == 0 else top_y + 120
        bottom = top_y + left_h if idx == 0 else top_y + 120 + right_h
        stroke = palette["accent"] if idx == 0 else palette["success"]
        body.append(
            f'<path d="M {bx:.1f} {top:.1f} L {bx:.1f} {bottom:.1f} M {bx - 10:.1f} {top:.1f} L {bx + 10:.1f} {top:.1f} M {bx - 10:.1f} {bottom:.1f} L {bx + 10:.1f} {bottom:.1f}" '
            f'fill="none" stroke="{stroke}" stroke-width="2.2" stroke-dasharray="6 5"/>'
        )
        if items:
            body.append(_svg_text_block(bx + (52 if idx == 0 else -52), (top + bottom) / 2 - 12, items[1] if len(items) > 1 else items[0], font_size=24, fill=stroke, weight=700, max_chars=12, anchor="middle"))
            body.append(_svg_text_block(bx + (52 if idx == 0 else -52), (top + bottom) / 2 + 16, items[0], font_size=12, fill=palette["muted"], max_chars=14, anchor="middle"))
    floor = _component_values(scene, "floor_band")
    if floor:
        fy = height - 138
        body.append(
            f'<rect x="{inset:.1f}" y="{fy:.1f}" width="{width - 2 * inset:.1f}" height="28" rx="8" fill="{palette["surface"]}" fill-opacity="0.06" stroke="{palette["border"]}" stroke-width="1"/>'
        )
        floor_items = _payload_items(floor[0], scene.get("_content"))
        label = floor_items[0] if floor_items else _resolve_text_value(floor[0], scene.get("_content"))
        body.append(_svg_text_block(width / 2, fy + 18, label, font_size=11, fill=palette["muted"], max_chars=72, anchor="middle"))
    legend = _component_values(scene, "legend_row")
    if legend:
        items = _legend_items(legend[0], scene.get("_content"))
        lx = width / 2 - (len(items) * 84) / 2
        ly = top_y + 12
        for idx, (tone, label) in enumerate(items):
            cx = lx + idx * 112
            color = _color_token(palette, tone, fallback=palette["accent"])
            body.append(f'<circle cx="{cx:.1f}" cy="{ly:.1f}" r="7" fill="{color}"/>')
            body.append(_svg_text_block(cx + 14, ly + 4, label, font_size=12, fill=palette["muted"], max_chars=20))
    annotations = _component_values(scene, "annotation")
    for idx, raw in enumerate(annotations[:2]):
        items, meta = _payload_parts_meta(raw, {"accent"}, content=scene.get("_content"))
        label = items[0] if items else ""
        note = items[1] if len(items) > 1 else ""
        color = _color_token(palette, meta.get("accent"), fallback=palette["accent"])
        ax = width / 2 - 140 + idx * 280
        ay = top_y + 18
        body.append(f'<rect x="{ax:.1f}" y="{ay:.1f}" width="180" height="38" rx="14" fill="{color}" fill-opacity="0.10" stroke="{color}" stroke-width="1"/>')
        body.append(_svg_text_block(ax + 14, ay + 18, label, font_size=11, fill=color, weight=700, max_chars=24))
        if note:
            body.append(_svg_text_block(ax + 14, ay + 32, note, font_size=10, fill=palette["muted"], max_chars=24))
    dividers = _component_values(scene, "divider")
    if dividers:
        style = str(dividers[0]).strip().lower()
        dash = "7 6" if "dash" in style else ""
        attrs = f' stroke-dasharray="{dash}"' if dash else ""
        body.append(f'<line x1="{inset + 12:.1f}" y1="{y + 92:.1f}" x2="{width - inset - 12:.1f}" y2="{y + 92:.1f}" stroke="{palette["grid"]}" stroke-width="1.5"{attrs}/>')
    conclusion = _component_values(scene, "conclusion_strip")
    if conclusion:
        cy = height - 92
        body.append(
            f'<rect x="{inset + 12:.1f}" y="{cy:.1f}" width="{width - 2 * inset - 24:.1f}" height="56" rx="18" fill="{palette["surface"]}" fill-opacity="0.1" stroke="{palette["accent"]}" stroke-width="1"/>'
        )
        concl_items = _payload_items(conclusion[0], scene.get("_content"))
        label = concl_items[0] if concl_items else _resolve_text_value(conclusion[0], scene.get("_content"))
        body.append(_svg_text_block(width / 2, cy + 35, label, font_size=16, fill=palette["ink"], weight=700, max_chars=74, anchor="middle"))
    footer = _component_values(scene, "footer_note")
    if footer:
        body.append(_footer_note(scene, palette, width, height - 16))
    return "".join(body)


def _layout_stage_positions(width: int, height: int, inset: float, gap: float, count: int) -> list[tuple[float, float, float, float]]:
    if count <= 5:
        card_w = (width - 2 * inset - gap * max(0, count - 1)) / max(1, count)
        y = inset + 160
        return [(inset + idx * (card_w + gap), y, card_w, 146.0) for idx in range(count)]
    top = min(5, count - 1)
    card_w = (width - 2 * inset - gap * max(0, top - 1)) / top
    top_y = inset + 160
    pos = [(inset + idx * (card_w + gap), top_y, card_w, 146.0) for idx in range(top)]
    bottom_w = card_w
    pos.append((width - inset - bottom_w, top_y + 174, bottom_w, 130.0))
    return pos


def _render_pipeline_lane(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX[str(scene["inset"])]
    gap = _GAP_PX[str(scene["gap"])]
    body: list[str] = []
    header, _ = _header_band(scene, palette, width, inset)
    body.append(header)
    phase = _component_values(scene, "phase_divider")
    if phase:
        items = _payload_items(phase[0], scene.get("_content"))
        divider_x = width / 2
        body.append(f'<line x1="{divider_x:.1f}" y1="{inset + 120:.1f}" x2="{divider_x:.1f}" y2="{height - 110:.1f}" stroke="{palette["grid"]}" stroke-width="2" stroke-dasharray="7 6"/>')
        if items:
            body.append(_svg_text_block(divider_x - 140, inset + 132, items[0], font_size=12, fill=palette["accent"], weight=700, max_chars=20, anchor="middle"))
            if len(items) > 1:
                body.append(_svg_text_block(divider_x + 140, inset + 132, items[1], font_size=12, fill=palette["success"], weight=700, max_chars=20, anchor="middle"))
    stages = _component_values(scene, "stage_card")
    stage_pos = _layout_stage_positions(width, height, inset, gap, len(stages))
    centers: list[tuple[float, float]] = []
    for idx, raw in enumerate(stages):
        x, y, w, h = stage_pos[idx]
        items = _payload_items(raw, scene.get("_content"))
        stage_color = palette["accent"] if x + w / 2 < width / 2 else palette["success"]
        body.append(_panel_shell(x, y, w, h, scene, palette))
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="32" rx="18" fill="{stage_color}" fill-opacity="0.90"/>')
        body.append(_svg_text_block(x + w / 2, y + 22, f"{idx + 1}. {items[0] if items else 'Stage'}", font_size=11, fill=_THEME_BASE["paper_editorial"]["ink"], weight=700, max_chars=18, anchor="middle"))
        body.append(_stage_visual_markup(items[0] if items else "", x, y + 36, w, stage_color))
        if len(items) > 1:
            body.append(_svg_text_block(x + w / 2, y + 114, items[1], font_size=13, fill=palette["muted"], max_chars=18, anchor="middle"))
        else:
            body.append(_svg_text_block(x + w / 2, y + 114, items[0] if items else "Process Step", font_size=13, fill=palette["muted"], max_chars=18, anchor="middle"))
        centers.append((x + w / 2, y + h / 2))
    arrows = _component_values(scene, "flow_arrow")
    for idx, _ in enumerate(arrows):
        if idx + 1 >= len(stage_pos):
            break
        x1, y1 = centers[idx]
        x2, y2 = centers[idx + 1]
        if abs(y1 - y2) < 40:
            body.append(
                f'<line x1="{x1 + 74:.1f}" y1="{y1:.1f}" x2="{x2 - 74:.1f}" y2="{y2:.1f}" stroke="{palette["accent"]}" stroke-width="3" marker-end="url(#arrowHead)"/>'
            )
    curves = _component_values(scene, "curved_connector")
    if curves and len(stage_pos) >= 6:
        x1, y1 = centers[4]
        x2, y2 = centers[5]
        body.append(
            f'<path d="M {x1 + 58:.1f} {y1:.1f} Q {x1 + 140:.1f} {y1:.1f} {x1 + 140:.1f} {y1 + 90:.1f} Q {x1 + 140:.1f} {y2:.1f} {x2:.1f} {y2:.1f}" '
            f'fill="none" stroke="{palette["success"]}" stroke-width="3" marker-end="url(#arrowHead)"/>'
        )
    badge = _component_values(scene, "badge_pill")
    info_cards = _component_values(scene, "section_card")
    if info_cards:
        info_y = height - inset - 166
        card_h = 112.0
        count = min(3, len(info_cards))
        info_w = (width - 2 * inset - gap * max(0, count - 1)) / max(1, count)
        for idx, raw in enumerate(info_cards[:count]):
            x = inset + idx * (info_w + gap)
            items, meta = _payload_parts_meta(raw, {"variant", "accent"}, content=scene.get("_content"))
            title = items[0] if items else f"Info {idx + 1}"
            line_1 = items[1] if len(items) > 1 else ""
            line_2 = items[2] if len(items) > 2 else ""
            accent_fill = _color_token(palette, meta.get("accent"), fallback=palette["accent"] if idx < 2 else palette["success"])
            body.append(_panel_shell(x, info_y, info_w, card_h, scene, palette))
            body.append(f'<rect x="{x + 16:.1f}" y="{info_y + 16:.1f}" width="{info_w - 32:.1f}" height="9" rx="4.5" fill="{accent_fill}" fill-opacity="0.88"/>')
            body.append(_svg_text_block(x + info_w / 2, info_y + 42, title, font_size=12, fill=accent_fill, weight=700, max_chars=24, anchor="middle"))
            if line_1:
                body.append(_svg_text_block(x + info_w / 2, info_y + 72, line_1, font_size=11, fill=palette["muted"], max_chars=28, anchor="middle"))
            if line_2:
                body.append(_svg_text_block(x + info_w / 2, info_y + 90, line_2, font_size=10, fill=palette["muted"], max_chars=28, anchor="middle"))
    if badge:
        badge_items = _payload_items(badge[0], scene.get("_content"))
        text = badge_items[0] if badge_items else _resolve_text_value(badge[0], scene.get("_content"))
        body.append(_badge_pill(text, palette, width - inset - 188, height - 72, 188))
    footer = _component_values(scene, "footer_note")
    if footer:
        body.append(_footer_note(scene, palette, width, height - 18))
    return "".join(body)


def _render_dashboard_cards(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX[str(scene["inset"])]
    gap = _GAP_PX[str(scene["gap"])]
    cols = max(2, min(4, int(scene["columns"])))
    body: list[str] = []
    header, y = _header_band(scene, palette, width, inset)
    body.append(header)
    cards = _component_values(scene, "section_card")
    rows = max(1, (len(cards) + cols - 1) // cols)
    card_w = (width - 2 * inset - gap * (cols - 1)) / cols
    card_h = max(170.0, (height - y - inset - gap * (rows - 1) - 34) / rows)
    for idx, raw in enumerate(cards):
        row = idx // cols
        col = idx % cols
        x = inset + col * (card_w + gap)
        cy = y + row * (card_h + gap)
        body.append(_render_section_card(scene, palette, x, cy, card_w, card_h, raw, rail=(col == 0 and row == 0 and scene["rail"] != "none")))
    footer = _component_values(scene, "footer_note")
    if footer:
        body.append(_footer_note(scene, palette, width, height - 18))
    return "".join(body)


def _render_dual_panel_compare(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX[str(scene["inset"])]
    gap = _GAP_PX[str(scene["gap"])]
    body: list[str] = []
    header, y = _header_band(scene, palette, width, inset)
    body.append(header)
    panels = _component_values(scene, "compare_panel")[:2]
    legacy = False
    if not panels:
        panels = _component_values(scene, "section_card")[:2]
        legacy = True
    panel_w = (width - 2 * inset - gap) / 2
    panel_h = height - y - inset - 60
    for idx, raw in enumerate(panels):
        x = inset + idx * (panel_w + gap)
        if legacy:
            body.append(_render_section_card(scene, palette, x, y, panel_w, panel_h, raw, rail=(idx == 0 and scene["rail"] != "none")))
        else:
            items, meta = _payload_parts_meta(raw, {"variant", "accent"}, content=scene.get("_content"))
            title = items[0] if items else ""
            value = items[1] if len(items) > 1 else ""
            caption = items[2] if len(items) > 2 else ""
            variant = str(meta.get("variant", "metric")).lower()
            edge, accent_fill = _variant_colors(palette, variant, meta.get("accent"))
            body.append(_panel_shell(x, y, panel_w, panel_h, scene, palette))
            if idx == 0 and scene["rail"] != "none":
                body.append(_accent_rail(scene, palette, x + 18, y + 22, panel_h - 44))
            body.append(f'<rect x="{x + 22:.1f}" y="{y + 20:.1f}" width="{panel_w - 44:.1f}" height="12" rx="6" fill="{accent_fill}" opacity="0.9"/>')
            body.append(_svg_text_block(x + 38, y + 64, title, font_size=24, fill=edge, weight=700, max_chars=18))
            if value:
                body.append(_svg_text_block(x + 38, y + 146, value, font_size=36, fill=palette["ink"], weight=700, max_chars=14, family="IBM Plex Mono, SFMono-Regular, monospace"))
            if caption:
                body.append(_svg_text_block(x + 38, y + panel_h - 58, caption, font_size=15, fill=palette["muted"], max_chars=34))
    callouts = _component_values(scene, "callout_card")
    if callouts:
        items, meta = _payload_parts_meta(callouts[0], {"accent"}, content=scene.get("_content"))
        label = items[0] if items else ""
        note = items[1] if len(items) > 1 else ""
        color = _color_token(palette, meta.get("accent"), fallback=palette["accent"])
        cy = height - inset - 72
        body.append(f'<rect x="{width/2 - 230:.1f}" y="{cy:.1f}" width="460" height="54" rx="18" fill="{color}" fill-opacity="0.10" stroke="{color}" stroke-width="1"/>')
        body.append(_svg_text_block(width / 2, cy + 24, label, font_size=15, fill=color, weight=700, max_chars=40, anchor="middle"))
        if note:
            body.append(_svg_text_block(width / 2, cy + 41, note, font_size=11, fill=palette["muted"], max_chars=46, anchor="middle"))
    footer = _component_values(scene, "footer_note")
    if footer:
        body.append(_footer_note(scene, palette, width, height - 18))
    return "".join(body)


def _render_timeline_flow(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX[str(scene["inset"])]
    gap = _GAP_PX[str(scene["gap"])]
    body: list[str] = []
    header, y = _header_band(scene, palette, width, inset)
    body.append(header)
    stages = _component_values(scene, "stage_card")
    count = len(stages)
    card_w = (width - 2 * inset - gap * max(0, count - 1)) / max(1, count)
    card_h = 190.0
    centers: list[tuple[float, float]] = []
    for idx, raw in enumerate(stages):
        x = inset + idx * (card_w + gap)
        items = _payload_items(raw, scene.get("_content"))
        body.append(_panel_shell(x, y + 46, card_w, card_h, scene, palette))
        body.append(f'<circle cx="{x + card_w / 2:.1f}" cy="{y + 34:.1f}" r="18" fill="{palette["accent"]}" filter="url(#accentGlow)"/>')
        body.append(_svg_text_block(x + card_w / 2, y + 40, str(idx + 1), font_size=14, fill=_THEME_BASE["paper_editorial"]["ink"], weight=700, anchor="middle"))
        body.append(_svg_text_block(x + card_w / 2, y + 102, items[0] if items else "Step", font_size=18, fill=palette["ink"], weight=700, max_chars=16, anchor="middle"))
        if len(items) > 1:
            body.append(_svg_text_block(x + card_w / 2, y + 142, items[1], font_size=14, fill=palette["muted"], max_chars=18, anchor="middle"))
        centers.append((x + card_w / 2, y + 140))
    arrows = _component_values(scene, "flow_arrow")
    for idx, _ in enumerate(arrows):
        if idx + 1 >= len(centers):
            break
        x1, y1 = centers[idx]
        x2, y2 = centers[idx + 1]
        body.append(
            f'<line x1="{x1 + card_w / 2 - 14:.1f}" y1="{y1:.1f}" x2="{x2 - card_w / 2 + 14:.1f}" y2="{y2:.1f}" stroke="{palette["accent"]}" stroke-width="3" marker-end="url(#arrowHead)"/>'
        )
    footer = _component_values(scene, "footer_note")
    if footer:
        body.append(_footer_note(scene, palette, width, height - 18))
    return "".join(body)


def _render_table_analysis(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX[str(scene["inset"])]
    gap = _GAP_PX[str(scene["gap"])]
    body: list[str] = []
    header, y = _header_band(scene, palette, width, inset)
    body.append(header)
    panel_y = y + 18
    panel_h = height - panel_y - inset - 24
    panel_w = width - 2 * inset
    body.append(_panel_shell(inset, panel_y, panel_w, panel_h, scene, palette))
    headers = _component_values(scene, "table_header")
    col_labels = _payload_items(headers[0], scene.get("_content")) if headers else []
    if col_labels:
        hy = panel_y + 18
        body.append(
            f'<rect x="{inset + 20:.1f}" y="{hy:.1f}" width="{panel_w - 40:.1f}" height="38" rx="12" fill="{palette["accent"]}" fill-opacity="0.12" stroke="{palette["accent"]}" stroke-width="1"/>'
        )
        col_x = [inset + 42, inset + panel_w * 0.50, inset + panel_w - 42]
        anchors = ["start", "middle", "end"]
        for idx, label in enumerate(col_labels[:3]):
            body.append(_svg_text_block(col_x[idx], hy + 24, label, font_size=12, fill=palette["accent"], weight=700, max_chars=18, anchor=anchors[idx]))
    rows = _component_values(scene, "table_row")
    top_pad = 76 if col_labels else 28
    row_h = min(74.0, (panel_h - top_pad - 28 - gap * max(0, len(rows) - 1)) / max(1, len(rows)))
    for idx, raw in enumerate(rows):
        items, meta = _payload_parts_meta(raw, {"state", "accent"}, content=scene.get("_content"))
        state = str(meta.get("state", "normal")).lower()
        if state not in ROW_STATES:
            state = "normal"
        fill, fill_op, stroke, accent = _row_style(palette, state, meta.get("accent"))
        ry = panel_y + top_pad + idx * (row_h + gap * 0.5)
        body.append(
            f'<rect x="{inset + 20:.1f}" y="{ry:.1f}" width="{panel_w - 40:.1f}" height="{row_h:.1f}" rx="14" fill="{fill}" fill-opacity="{fill_op}" stroke="{stroke}" stroke-width="1"/>'
        )
        if state in {"highlight", "success", "warning", "danger"}:
            body.append(f'<rect x="{inset + 20:.1f}" y="{ry:.1f}" width="8" height="{row_h:.1f}" rx="4" fill="{accent}"/>')
        if items:
            body.append(_svg_text_block(inset + 42, ry + 30, items[0], font_size=16, fill=palette["ink"], weight=700, max_chars=28))
        if len(items) > 1:
            if len(col_labels) >= 3 and len(items) > 2:
                body.append(_svg_text_block(inset + panel_w * 0.50, ry + 30, items[1], font_size=16, fill=accent, weight=700, max_chars=18, anchor="middle", family="IBM Plex Mono, SFMono-Regular, monospace"))
                body.append(_svg_text_block(inset + panel_w - 42, ry + 30, items[2], font_size=16, fill=accent, weight=700, max_chars=18, anchor="end", family="IBM Plex Mono, SFMono-Regular, monospace"))
            else:
                body.append(_svg_text_block(inset + panel_w - 42, ry + 30, items[1], font_size=16, fill=accent, weight=700, max_chars=18, anchor="end", family="IBM Plex Mono, SFMono-Regular, monospace"))
        if len(items) > 2:
            note_idx = 3 if len(col_labels) >= 3 and len(items) > 3 else 2
            if note_idx < len(items):
                body.append(_svg_text_block(inset + 42, ry + row_h - 14, items[note_idx], font_size=12, fill=palette["muted"], max_chars=60))
    callouts = _component_values(scene, "callout_card")
    if callouts:
        items, meta = _payload_parts_meta(callouts[0], {"accent"}, content=scene.get("_content"))
        label = items[0] if items else ""
        note = items[1] if len(items) > 1 else ""
        color = _color_token(palette, meta.get("accent"), fallback=palette["accent"])
        cy = height - inset - 84
        body.append(f'<rect x="{inset + 20:.1f}" y="{cy:.1f}" width="{panel_w - 40:.1f}" height="52" rx="16" fill="{color}" fill-opacity="0.08" stroke="{color}" stroke-width="1"/>')
        body.append(_svg_text_block(inset + 38, cy + 22, label, font_size=14, fill=color, weight=700, max_chars=42))
        if note:
            body.append(_svg_text_block(inset + 38, cy + 39, note, font_size=11, fill=palette["muted"], max_chars=64))
    footer = _component_values(scene, "footer_note")
    if footer:
        body.append(_footer_note(scene, palette, width, height - 18))
    return "".join(body)


def render_structured_scene_spec09_svg(text: str, content: dict[str, Any] | None = None) -> str:
    scene = _parse_scene_document(text)
    scene["_content"] = content or {}
    width, height = _canvas_size(str(scene["canvas"]))
    palette = _palette(scene)
    body = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#bgGrad)"/>',
        _background_motif(scene, palette, width, height),
    ]
    layout = str(scene["layout"])
    if layout == "poster_stack":
        body.append(_render_poster_stack(scene, palette, width, height))
    elif layout == "comparison_span_chart":
        body.append(_render_comparison_span_chart(scene, palette, width, height))
    elif layout == "pipeline_lane":
        body.append(_render_pipeline_lane(scene, palette, width, height))
    elif layout == "dashboard_cards":
        body.append(_render_dashboard_cards(scene, palette, width, height))
    elif layout == "dual_panel_compare":
        body.append(_render_dual_panel_compare(scene, palette, width, height))
    elif layout == "timeline_flow":
        body.append(_render_timeline_flow(scene, palette, width, height))
    elif layout == "table_analysis":
        body.append(_render_table_analysis(scene, palette, width, height))
    else:
        raise ValueError(f"unsupported scene layout: {layout}")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f"{_defs(scene, palette)}"
        f'{"".join(body)}'
        "</svg>"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Render a spec09 scene DSL document to SVG.")
    ap.add_argument("--scene", default=None, help="Inline scene document.")
    ap.add_argument("--scene-file", default=None, help="Path to a scene document file.")
    ap.add_argument("--content-json", default=None, help="Optional JSON content payload used to resolve @paths in the scene.")
    ap.add_argument("--out", default=None, help="Optional output SVG path.")
    args = ap.parse_args()

    if bool(args.scene) == bool(args.scene_file):
        raise SystemExit("ERROR: pass exactly one of --scene or --scene-file")
    text = args.scene if args.scene is not None else Path(args.scene_file).read_text(encoding="utf-8")
    content = None
    if args.content_json:
        content = json.loads(Path(args.content_json).read_text(encoding="utf-8"))
    svg = render_structured_scene_spec09_svg(text, content=content)
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
