#!/usr/bin/env python3
"""Render the structured SVG atoms DSL into concrete SVG XML."""

from __future__ import annotations

from typing import Any

try:
    from spec06_infographic_content_v7 import build_text_slot_map, resolve_text_slot
except ImportError:
    build_text_slot_map = None  # type: ignore[assignment]
    resolve_text_slot = None  # type: ignore[assignment]


_COLOR_MAP = {
    "red": "#ef4444",
    "blue": "#3b82f6",
    "green": "#22c55e",
    "orange": "#f97316",
    "purple": "#a855f7",
    "paper": "#f8fafc",
    "mint": "#ecfdf5",
    "slate": "#1f2937",
    "black": "#111827",
    "white": "#f8fafc",
    "gray": "#94a3b8",
}

_ELEMENT_STARTS = {"[circle]", "[rect]", "[polygon]", "[text]"}
_TEXT_SLOT_MAP = build_text_slot_map() if callable(build_text_slot_map) else {}


def _render_color(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "black"
    return _COLOR_MAP.get(raw, raw)


def _token_value(token: str, prefix: str) -> str | None:
    if token.startswith(prefix) and token.endswith("]"):
        return token[len(prefix) : -1]
    return None


def _parse_shape(tokens: list[str], start: int, kind: str) -> tuple[int, dict[str, Any]]:
    defaults: dict[str, Any]
    if kind == "circle":
        defaults = {"cx": "64", "cy": "64", "r": "18", "fill": "red", "stroke": "black", "sw": "2"}
    elif kind == "rect":
        defaults = {"x": "42", "y": "48", "width": "44", "height": "32", "rx": "6", "fill": "red", "stroke": "black", "sw": "2"}
    else:
        defaults = {"points": "64,34 36,86 92,86", "fill": "red", "stroke": "black", "sw": "2"}
    i = start + 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "[/svg]" or tok in _ELEMENT_STARTS:
            break
        for key in tuple(defaults):
            value = _token_value(tok, f"[{key}:")
            if value is not None:
                defaults[key] = value.replace("|", " ") if key == "points" else value
                break
        i += 1
    defaults["kind"] = kind
    return i, defaults


def _parse_text(tokens: list[str], start: int, *, current_topic: str = "") -> tuple[int, dict[str, Any]]:
    elem = {
        "kind": "text",
        "tx": "64",
        "ty": "64",
        "font": "14",
        "anchor": "middle",
        "fill": "black",
        "text": "",
    }
    text_bits: list[str] = []
    i = start + 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "[/text]":
            i += 1
            break
        if tok == "[/svg]" or tok in _ELEMENT_STARTS:
            break
        matched = False
        for key in ("tx", "ty", "font", "anchor", "fill"):
            value = _token_value(tok, f"[{key}:")
            if value is not None:
                elem[key] = value
                matched = True
                break
        if not matched:
            slot = _token_value(tok, "[slot:")
            if slot is not None:
                if callable(resolve_text_slot):
                    text_bits.append(resolve_text_slot(slot, topic=current_topic))
                else:
                    text_bits.append(_TEXT_SLOT_MAP.get(slot, slot.replace("__", " ").replace("_", " ")))
            else:
                text_bits.append(tok)
        i += 1
    elem["text"] = " ".join(text_bits).strip()
    return i, elem


def _render_element(elem: dict[str, Any]) -> str:
    kind = str(elem.get("kind") or "").strip().lower()
    if kind == "circle":
        return (
            f'<circle cx="{elem["cx"]}" cy="{elem["cy"]}" r="{elem["r"]}" '
            f'fill="{_render_color(elem["fill"])}" stroke="{_render_color(elem["stroke"])}" '
            f'stroke-width="{elem["sw"]}"/>'
        )
    if kind == "rect":
        return (
            f'<rect x="{elem["x"]}" y="{elem["y"]}" width="{elem["width"]}" '
            f'height="{elem["height"]}" rx="{elem["rx"]}" fill="{_render_color(elem["fill"])}" '
            f'stroke="{_render_color(elem["stroke"])}" stroke-width="{elem["sw"]}"/>'
        )
    if kind == "polygon":
        return (
            f'<polygon points="{elem["points"]}" fill="{_render_color(elem["fill"])}" '
            f'stroke="{_render_color(elem["stroke"])}" stroke-width="{elem["sw"]}"/>'
        )
    if kind == "text":
        return (
            f'<text x="{elem["tx"]}" y="{elem["ty"]}" font-size="{elem["font"]}" '
            f'text-anchor="{elem["anchor"]}" fill="{_render_color(elem["fill"])}" '
            'font-family="monospace">'
            f'{elem["text"]}</text>'
        )
    raise ValueError(f"unsupported structured SVG element: {kind}")


def render_structured_svg_atoms(text: str) -> str:
    tokens = [tok.strip() for tok in str(text or "").split() if tok.strip()]
    if "[svg]" in tokens:
        tokens = tokens[tokens.index("[svg]") :]
    if not tokens or tokens[0] != "[svg]":
        raise ValueError("structured SVG DSL must start with [svg]")
    if "[/svg]" in tokens:
        tokens = tokens[: tokens.index("[/svg]") + 1]

    width = "128"
    height = "128"
    background = "none"
    current_topic = ""
    elements: list[dict[str, Any]] = []
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "[/svg]":
            break
        value = _token_value(tok, "[w:")
        if value is not None:
            width = value
            i += 1
            continue
        value = _token_value(tok, "[h:")
        if value is not None:
            height = value
            i += 1
            continue
        value = _token_value(tok, "[bg:")
        if value is not None:
            background = value
            i += 1
            continue
        value = _token_value(tok, "[topic:")
        if value is not None:
            current_topic = value
            i += 1
            continue
        if tok == "[circle]":
            i, elem = _parse_shape(tokens, i, "circle")
            elements.append(elem)
            continue
        if tok == "[rect]":
            i, elem = _parse_shape(tokens, i, "rect")
            elements.append(elem)
            continue
        if tok == "[polygon]":
            i, elem = _parse_shape(tokens, i, "polygon")
            elements.append(elem)
            continue
        if tok == "[text]":
            i, elem = _parse_text(tokens, i, current_topic=current_topic)
            elements.append(elem)
            continue
        i += 1

    body: list[str] = []
    if background != "none":
        body.append(
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="{_render_color(background)}"/>'
        )
    body.extend(_render_element(elem) for elem in elements)
    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">{"".join(body)}</svg>'
