#!/usr/bin/env python3
"""Render a richer structured scene DSL directly into SVG."""

from __future__ import annotations

import html
from typing import Any

from spec06_infographic_content_v7 import TOPIC_LIBRARY, resolve_text_slot


LAYOUTS = {"bullet-panel", "compare-panels", "stat-cards", "spectrum-band", "flow-steps"}
THEMES = {"paper_editorial", "infra_dark", "signal_glow"}
TONES = {"orange", "green", "blue", "purple", "gray"}
FRAMES = {"none", "card", "panel"}
DENSITIES = {"compact", "balanced", "airy"}
INSETS = {"sm", "md", "lg"}
GAPS = {"sm", "md", "lg"}
HERO_MODES = {"left", "center", "split"}
COLUMNS = {"1", "2", "3"}
EMPHASIS = {"top", "left", "center"}
RAILS = {"accent", "muted", "none"}
BACKGROUNDS = {"grid", "mesh", "rings", "none"}
CONNECTORS = {"line", "arrow", "bracket"}
CANVAS = {"wide", "tall", "square"}

_REQUIRED_COMPONENTS = {
    "bullet-panel": {"hero", "bullet_list", "callout"},
    "compare-panels": {"hero", "compare_block", "footer"},
    "stat-cards": {"hero", "stats", "callout"},
    "spectrum-band": {"hero", "band", "footer"},
    "flow-steps": {"hero", "step_flow", "badge"},
}

_TONE_ACCENTS = {
    "orange": {"accent": "#f97316", "accent_2": "#fb923c", "soft": "#fed7aa"},
    "green": {"accent": "#22c55e", "accent_2": "#34d399", "soft": "#bbf7d0"},
    "blue": {"accent": "#38bdf8", "accent_2": "#6366f1", "soft": "#bfdbfe"},
    "purple": {"accent": "#a855f7", "accent_2": "#ec4899", "soft": "#e9d5ff"},
    "gray": {"accent": "#94a3b8", "accent_2": "#cbd5e1", "soft": "#e2e8f0"},
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
        "shadow": "rgba(23,33,43,0.12)",
        "grid": "rgba(114,95,72,0.10)",
    },
    "infra_dark": {
        "bg": "#09111f",
        "bg_2": "#172033",
        "surface": "#0f172a",
        "surface_2": "#162235",
        "ink": "#f8fafc",
        "muted": "#94a3b8",
        "border": "#334155",
        "shadow": "rgba(2,6,23,0.45)",
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

_INSET_PX = {"sm": 40, "md": 56, "lg": 72}
_GAP_PX = {"sm": 16, "md": 24, "lg": 34}
_CARD_RADIUS = {"compact": 18, "balanced": 22, "airy": 26}
_TEXT_MAX = {"compact": 28, "balanced": 32, "airy": 36}


def _token_value(token: str, prefix: str) -> str | None:
    if token.startswith(prefix) and token.endswith("]"):
        return token[len(prefix) : -1]
    return None


def _escape(text: Any) -> str:
    return html.escape(str(text or ""), quote=False)


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
            continue
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
    max_chars: int = 32,
    line_height: float = 1.28,
    anchor: str = "start",
) -> str:
    lines = _wrap_lines(text, max_chars=max_chars)
    body: list[str] = [
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{font_size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        'font-family="IBM Plex Sans, Segoe UI, sans-serif">'
    ]
    for idx, line in enumerate(lines):
        dy = "0" if idx == 0 else f"{font_size * line_height:.1f}"
        body.append(f'<tspan x="{x:.1f}" dy="{dy}">{_escape(line)}</tspan>')
    body.append("</text>")
    return "".join(body)


def _slot_list(raw: str, *, topic: str) -> list[str]:
    return [resolve_text_slot(item, topic=topic) for item in str(raw or "").split("|") if item]


def _scene_defaults() -> dict[str, Any]:
    return {
        "canvas": "wide",
        "layout": "",
        "theme": "paper_editorial",
        "tone": "blue",
        "frame": "card",
        "density": "balanced",
        "inset": "md",
        "gap": "md",
        "hero_align": "left",
        "columns": "1",
        "emphasis": "top",
        "rail": "accent",
        "background": "grid",
        "connector": "line",
        "topic": "",
        "components": {},
    }


def _parse_scene_document(text: str) -> dict[str, Any]:
    tokens = [tok.strip() for tok in str(text or "").split() if tok.strip()]
    if "[scene]" in tokens:
        tokens = tokens[tokens.index("[scene]") :]
    if not tokens or tokens[0] != "[scene]":
        raise ValueError("structured scene DSL must start with [scene]")
    if "[/scene]" not in tokens:
        raise ValueError("structured scene DSL must end with [/scene]")
    tokens = tokens[: tokens.index("[/scene]") + 1]

    scene = _scene_defaults()
    components: dict[str, str] = {}
    for token in tokens[1:]:
        if token == "[/scene]":
            break
        for key, allowed in (
            ("canvas", CANVAS),
            ("layout", LAYOUTS),
            ("theme", THEMES),
            ("tone", TONES),
            ("frame", FRAMES),
            ("density", DENSITIES),
            ("inset", INSETS),
            ("gap", GAPS),
            ("hero_align", HERO_MODES),
            ("columns", COLUMNS),
            ("emphasis", EMPHASIS),
            ("rail", RAILS),
            ("background", BACKGROUNDS),
            ("connector", CONNECTORS),
            ("topic", set(TOPIC_LIBRARY)),
        ):
            value = _token_value(token, f"[{key}:")
            if value is not None:
                if key != "topic" and value not in allowed:
                    raise ValueError(f"unsupported {key}: {value}")
                if key == "topic" and value not in TOPIC_LIBRARY:
                    raise ValueError(f"unsupported topic: {value}")
                scene[key] = value
                break
        else:
            if token.startswith("[") and token.endswith("]") and ":" in token:
                comp, raw = token[1:-1].split(":", 1)
                components[comp] = raw

    layout = str(scene["layout"] or "")
    topic = str(scene["topic"] or "")
    if layout not in LAYOUTS:
        raise ValueError(f"unsupported scene layout: {layout or '<empty>'}")
    if topic not in TOPIC_LIBRARY:
        raise ValueError(f"unsupported scene topic: {topic or '<empty>'}")
    missing = sorted(_REQUIRED_COMPONENTS.get(layout, set()) - set(components))
    if missing:
        raise ValueError(f"scene missing required components for {layout}: {', '.join(missing)}")
    scene["components"] = components
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
    }


def _canvas_size(canvas: str) -> tuple[int, int]:
    return {
        "wide": (1200, 780),
        "tall": (960, 1280),
        "square": (1024, 1024),
    }.get(str(canvas or "wide"), (1200, 780))


def _defs(scene: dict[str, Any], palette: dict[str, str]) -> str:
    marker = ""
    if scene["connector"] == "arrow":
        marker = (
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
        f'<feDropShadow dx="0" dy="14" stdDeviation="18" flood-color="{palette["shadow"]}"/>'
        "</filter>"
        f'<filter id="accentGlow" x="-30%" y="-30%" width="160%" height="180%">'
        f'<feDropShadow dx="0" dy="0" stdDeviation="10" flood-color="{palette["accent"]}"/>'
        "</filter>"
        f"{marker}"
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
            '<g opacity="0.26">'
            f'<circle cx="{width * 0.18:.1f}" cy="{height * 0.24:.1f}" r="180" fill="{palette["panel_glow"]}"/>'
            f'<circle cx="{width * 0.88:.1f}" cy="{height * 0.18:.1f}" r="140" fill="{palette["soft"]}"/>'
            f'<circle cx="{width * 0.68:.1f}" cy="{height * 0.84:.1f}" r="220" fill="{palette["accent_2"]}" opacity="0.28"/>'
            "</g>"
        )
    return (
        '<g opacity="0.22">'
        f'<circle cx="{width * 0.82:.1f}" cy="{height * 0.20:.1f}" r="180" fill="none" stroke="{palette["grid"]}" stroke-width="2"/>'
        f'<circle cx="{width * 0.82:.1f}" cy="{height * 0.20:.1f}" r="250" fill="none" stroke="{palette["grid"]}" stroke-width="1.5"/>'
        f'<circle cx="{width * 0.14:.1f}" cy="{height * 0.82:.1f}" r="120" fill="none" stroke="{palette["accent"]}" stroke-width="2"/>'
        "</g>"
    )


def _hero_block(scene: dict[str, Any], palette: dict[str, str], x: float, y: float, width: float) -> str:
    hero_slots = _slot_list(scene["components"]["hero"], topic=str(scene["topic"]))
    title = hero_slots[0] if hero_slots else resolve_text_slot("title", topic=str(scene["topic"]))
    subtitle = hero_slots[1] if len(hero_slots) > 1 else resolve_text_slot("subtitle", topic=str(scene["topic"]))
    anchor = "middle" if scene["hero_align"] == "center" else "start"
    text_x = x + width / 2 if anchor == "middle" else x
    title_block = _svg_text_block(
        text_x,
        y,
        title,
        font_size=34 if scene["density"] != "compact" else 30,
        fill=palette["ink"],
        weight=700,
        max_chars=22,
        anchor=anchor,
    )
    subtitle_block = _svg_text_block(
        text_x,
        y + 48,
        subtitle,
        font_size=15,
        fill=palette["muted"],
        weight=500,
        max_chars=48,
        anchor=anchor,
    )
    accent_x = x if anchor == "start" else (x + width / 2 - 100)
    accent = (
        f'<rect x="{accent_x:.1f}" y="{y - 22:.1f}" width="200" height="8" rx="4" fill="url(#heroGrad)" '
        'filter="url(#accentGlow)"/>'
    )
    return accent + title_block + subtitle_block


def _panel_shell(x: float, y: float, width: float, height: float, scene: dict[str, Any], palette: dict[str, str]) -> str:
    radius = _CARD_RADIUS[str(scene["density"])]
    stroke = palette["surface_edge"] if scene["frame"] != "none" else "none"
    sw = "1.5" if scene["frame"] != "none" else "0"
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="{radius}" '
        f'fill="url(#panelGrad)" stroke="{stroke}" stroke-width="{sw}" filter="url(#softShadow)"/>'
    )


def _rail(scene: dict[str, Any], palette: dict[str, str], x: float, y: float, height: float) -> str:
    if scene["rail"] == "none":
        return ""
    color = palette["accent"] if scene["rail"] == "accent" else palette["muted"]
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="8" height="{height:.1f}" rx="4" '
        f'fill="{color}" filter="url(#accentGlow)"/>'
    )


def _render_bullet_panel(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX[str(scene["inset"])]
    gap = _GAP_PX[str(scene["gap"])]
    hero_y = inset + 48
    panel_x = inset
    panel_y = inset + 120
    panel_w = width - 2 * inset
    panel_h = height - panel_y - inset
    bullets = _slot_list(scene["components"]["bullet_list"], topic=str(scene["topic"]))
    callout = resolve_text_slot(scene["components"]["callout"], topic=str(scene["topic"]))
    card = _panel_shell(panel_x, panel_y, panel_w, panel_h, scene, palette)
    rail = _rail(scene, palette, panel_x + 28, panel_y + 30, panel_h - 60)
    body: list[str] = [_hero_block(scene, palette, inset, hero_y, panel_w), card, rail]
    start_x = panel_x + 56
    start_y = panel_y + 76
    for idx, bullet in enumerate(bullets, start=1):
        y = start_y + (idx - 1) * (30 + gap * 0.4)
        body.append(f'<circle cx="{start_x:.1f}" cy="{y - 7:.1f}" r="7" fill="{palette["accent"]}"/>')
        body.append(_svg_text_block(start_x + 18, y, bullet, font_size=16, fill=palette["ink"], max_chars=42))
    callout_y = panel_y + panel_h - 74
    body.append(
        f'<rect x="{panel_x + 36:.1f}" y="{callout_y:.1f}" width="{panel_w - 72:.1f}" height="44" rx="18" '
        f'fill="{palette["surface_2"]}" stroke="{palette["border"]}" stroke-width="1"/>'
    )
    body.append(_svg_text_block(panel_x + 58, callout_y + 28, callout, font_size=15, fill=palette["ink"], max_chars=58))
    return "".join(body)


def _render_compare_panels(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX[str(scene["inset"])]
    gap = _GAP_PX[str(scene["gap"])]
    hero_h = 132
    panel_y = inset + hero_h
    panel_h = height - panel_y - inset - 76
    panel_w = (width - 2 * inset - gap) / 2
    slots = _slot_list(scene["components"]["compare_block"], topic=str(scene["topic"]))
    left = slots[:3]
    right = slots[3:6]
    footer = resolve_text_slot(scene["components"]["footer"], topic=str(scene["topic"]))
    body: list[str] = [_hero_block(scene, palette, inset, inset + 48, width - 2 * inset)]
    for idx, block in enumerate((left, right)):
        x = inset + idx * (panel_w + gap)
        body.append(_panel_shell(x, panel_y, panel_w, panel_h, scene, palette))
        if idx == 0 and scene["rail"] != "none":
            body.append(_rail(scene, palette, x + 22, panel_y + 24, panel_h - 48))
        title = block[0] if block else ""
        lines = block[1:] if len(block) > 1 else []
        body.append(_svg_text_block(x + 42, panel_y + 58, title, font_size=24, fill=palette["ink"], weight=700, max_chars=18))
        for line_idx, line in enumerate(lines, start=1):
            body.append(_svg_text_block(x + 42, panel_y + 58 + line_idx * 40, line, font_size=16, fill=palette["muted"], max_chars=24))
    connector_y = panel_y + panel_h / 2
    if scene["connector"] == "arrow":
        body.append(
            f'<line x1="{inset + panel_w:.1f}" y1="{connector_y:.1f}" x2="{inset + panel_w + gap:.1f}" y2="{connector_y:.1f}" '
            f'stroke="{palette["accent"]}" stroke-width="3" marker-end="url(#arrowHead)"/>'
        )
    else:
        body.append(
            f'<line x1="{inset + panel_w:.1f}" y1="{connector_y:.1f}" x2="{inset + panel_w + gap:.1f}" y2="{connector_y:.1f}" '
            f'stroke="{palette["accent"]}" stroke-width="3" stroke-dasharray="8 8"/>'
        )
    footer_y = height - inset - 22
    body.append(_svg_text_block(width / 2, footer_y, footer, font_size=15, fill=palette["muted"], max_chars=72, anchor="middle"))
    return "".join(body)


def _render_stat_cards(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX[str(scene["inset"])]
    gap = _GAP_PX[str(scene["gap"])]
    hero_h = 140
    cards_y = inset + hero_h
    card_h = 250
    card_w = (width - 2 * inset - 2 * gap) / 3
    slots = _slot_list(scene["components"]["stats"], topic=str(scene["topic"]))
    pairs = list(zip(slots[::2], slots[1::2]))
    callout = resolve_text_slot(scene["components"]["callout"], topic=str(scene["topic"]))
    body: list[str] = [_hero_block(scene, palette, inset, inset + 50, width - 2 * inset)]
    for idx, pair in enumerate(pairs[:3]):
        x = inset + idx * (card_w + gap)
        body.append(_panel_shell(x, cards_y, card_w, card_h, scene, palette))
        body.append(
            f'<rect x="{x + 18:.1f}" y="{cards_y + 18:.1f}" width="{card_w - 36:.1f}" height="10" rx="5" fill="url(#heroGrad)"/>'
        )
        body.append(_svg_text_block(x + 28, cards_y + 112, pair[0], font_size=54, fill=palette["ink"], weight=700, max_chars=8))
        body.append(_svg_text_block(x + 28, cards_y + 160, pair[1], font_size=16, fill=palette["muted"], max_chars=18))
    footer_y = cards_y + card_h + 48
    body.append(
        f'<rect x="{inset:.1f}" y="{footer_y:.1f}" width="{width - 2 * inset:.1f}" height="56" rx="24" '
        f'fill="{palette["surface_2"]}" stroke="{palette["border"]}" stroke-width="1"/>'
    )
    body.append(_svg_text_block(inset + 28, footer_y + 36, callout, font_size=16, fill=palette["ink"], max_chars=72))
    return "".join(body)


def _render_spectrum_band(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX[str(scene["inset"])]
    hero_h = 138
    band_y = inset + hero_h + 80
    band_h = 110
    band_w = width - 2 * inset
    slots = _slot_list(scene["components"]["band"], topic=str(scene["topic"]))
    footer = resolve_text_slot(scene["components"]["footer"], topic=str(scene["topic"]))
    segment_w = band_w / max(1, len(slots[:3]))
    body: list[str] = [_hero_block(scene, palette, inset, inset + 48, width - 2 * inset)]
    body.append(_panel_shell(inset, band_y - 40, band_w, 210, scene, palette))
    for idx, label in enumerate(slots[:3]):
        x = inset + idx * segment_w
        fill = palette["accent"] if idx == 1 else palette["accent_2"] if idx == 2 else palette["soft"]
        text_fill = palette["ink"] if scene["theme"] == "paper_editorial" or idx == 0 else palette["ink"]
        body.append(
            f'<rect x="{x + 14:.1f}" y="{band_y:.1f}" width="{segment_w - 28:.1f}" height="{band_h}" rx="26" fill="{fill}" opacity="0.88"/>'
        )
        body.append(_svg_text_block(x + segment_w / 2, band_y + 62, label, font_size=18, fill=text_fill, max_chars=16, anchor="middle"))
    if scene["connector"] == "bracket":
        left = inset + 24
        right = inset + band_w - 24
        by = band_y + band_h + 32
        body.append(
            f'<path d="M {left:.1f} {by:.1f} L {left:.1f} {by + 24:.1f} L {right:.1f} {by + 24:.1f} L {right:.1f} {by:.1f}" '
            f'fill="none" stroke="{palette["accent"]}" stroke-width="4"/>'
        )
    else:
        mid_x = inset + band_w / 2
        body.append(
            f'<line x1="{mid_x:.1f}" y1="{band_y - 26:.1f}" x2="{mid_x:.1f}" y2="{band_y - 2:.1f}" stroke="{palette["accent"]}" stroke-width="4" marker-end="url(#arrowHead)"/>'
        )
    body.append(_svg_text_block(width / 2, band_y + band_h + 86, footer, font_size=16, fill=palette["muted"], max_chars=78, anchor="middle"))
    return "".join(body)


def _render_flow_steps(scene: dict[str, Any], palette: dict[str, str], width: int, height: int) -> str:
    inset = _INSET_PX[str(scene["inset"])]
    gap = _GAP_PX[str(scene["gap"])]
    hero_h = 138
    steps_y = inset + hero_h + 28
    step_h = 270
    step_w = (width - 2 * inset - 2 * gap) / 3
    slots = _slot_list(scene["components"]["step_flow"], topic=str(scene["topic"]))
    pairs = list(zip(slots[::2], slots[1::2]))
    badge = resolve_text_slot(scene["components"]["badge"], topic=str(scene["topic"]))
    body: list[str] = [_hero_block(scene, palette, inset, inset + 48, width - 2 * inset)]
    for idx, pair in enumerate(pairs[:3]):
        x = inset + idx * (step_w + gap)
        body.append(_panel_shell(x, steps_y, step_w, step_h, scene, palette))
        body.append(
            f'<circle cx="{x + 46:.1f}" cy="{steps_y + 44:.1f}" r="18" fill="{palette["accent"]}" filter="url(#accentGlow)"/>'
        )
        body.append(_svg_text_block(x + 42, steps_y + 50, str(idx + 1), font_size=15, fill=palette["surface"], weight=700))
        body.append(_svg_text_block(x + 34, steps_y + 100, pair[0], font_size=24, fill=palette["ink"], weight=700, max_chars=16))
        body.append(_svg_text_block(x + 34, steps_y + 146, pair[1], font_size=16, fill=palette["muted"], max_chars=20))
        if idx < 2:
            x1 = x + step_w
            x2 = x + step_w + gap
            y = steps_y + step_h / 2
            attrs = 'marker-end="url(#arrowHead)"' if scene["connector"] == "arrow" else 'stroke-dasharray="8 8"'
            body.append(
                f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{palette["accent"]}" stroke-width="3" {attrs}/>'
            )
    badge_w = 180
    badge_x = width / 2 - badge_w / 2
    badge_y = steps_y + step_h + 44
    body.append(
        f'<rect x="{badge_x:.1f}" y="{badge_y:.1f}" width="{badge_w}" height="48" rx="24" fill="url(#heroGrad)" filter="url(#accentGlow)"/>'
    )
    body.append(_svg_text_block(width / 2, badge_y + 31, badge, font_size=16, fill=palette["surface"], weight=700, max_chars=22, anchor="middle"))
    return "".join(body)


def render_structured_scene_rich_svg(text: str) -> str:
    scene = _parse_scene_document(text)
    width, height = _canvas_size(str(scene["canvas"]))
    palette = _palette(scene)
    body = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#bgGrad)"/>',
        _background_motif(scene, palette, width, height),
    ]
    layout = str(scene["layout"])
    if layout == "bullet-panel":
        body.append(_render_bullet_panel(scene, palette, width, height))
    elif layout == "compare-panels":
        body.append(_render_compare_panels(scene, palette, width, height))
    elif layout == "stat-cards":
        body.append(_render_stat_cards(scene, palette, width, height))
    elif layout == "spectrum-band":
        body.append(_render_spectrum_band(scene, palette, width, height))
    elif layout == "flow-steps":
        body.append(_render_flow_steps(scene, palette, width, height))
    else:  # pragma: no cover
        raise ValueError(f"unsupported scene layout: {layout}")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'{_defs(scene, palette)}'
        f'{"".join(body)}'
        "</svg>"
    )
