#!/usr/bin/env python3
"""Shared adapters and rendering helpers for generic probe reports."""

from __future__ import annotations

import re
from typing import Any, Callable

from render_svg_structured_scene_spec09_v7 import render_structured_scene_spec09_svg as _render_structured_scene_spec09_svg_v7
from render_svg_structured_scene_rich_v7 import render_structured_scene_rich_svg as _render_structured_scene_rich_svg_v7
from render_svg_structured_scene_v7 import render_structured_scene_svg as _render_structured_scene_svg_v7
from render_svg_structured_atoms_v7 import render_structured_svg_atoms as _render_structured_svg_atoms_v7


_SVG_OPEN_RE = re.compile(r"<svg\b", re.IGNORECASE)
_SVG_CLOSE_RE = re.compile(r"</svg>", re.IGNORECASE)
_RESPONSE_MARKERS = ("\nResponse: ", "\nResponse:", "Response: ", "Response:")


def extract_response_text(raw: str, prompt: str = "") -> str:
    for marker in _RESPONSE_MARKERS:
        idx = raw.find(marker)
        if idx != -1:
            return raw[idx + len(marker) :].strip()
    idx = raw.lower().find("<svg")
    if idx != -1:
        return raw[idx:].strip()
    if prompt and raw.startswith(prompt):
        return raw[len(prompt) :].strip()
    return raw.strip()


def normalize_whitespace(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def truncate_at_markers(text: str, markers: list[str] | tuple[str, ...]) -> tuple[str, str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return "", ""
    cut = None
    marker_text = ""
    for marker in markers:
        token = str(marker or "").strip()
        if not token:
            continue
        idx = cleaned.find(token)
        if idx == -1:
            continue
        end = idx + len(token)
        if cut is None or end < cut:
            cut = end
            marker_text = token
    if cut is None:
        return cleaned, ""
    parsed = cleaned[:cut].strip()
    tail = cleaned[cut:].strip()
    if tail.startswith(marker_text):
        tail = tail[len(marker_text) :].strip()
    return parsed, tail


def extract_svg_root(text: str) -> tuple[str | None, str, str]:
    cleaned = str(text or "")
    start = cleaned.lower().find("<svg")
    if start < 0:
        return None, cleaned.strip(), ""
    end = cleaned.lower().find("</svg>", start)
    if end < 0:
        return None, cleaned[:start].strip(), cleaned[start:].strip()
    end += len("</svg>")
    return cleaned[start:end].strip(), cleaned[:start].strip(), cleaned[end:].strip()


def normalize_svg(svg_text: Any) -> str:
    return re.sub(r"\s+", " ", str(svg_text or "").strip())


def is_valid_svg(svg_text: str | None) -> bool:
    if not svg_text:
        return False
    return bool(_SVG_OPEN_RE.search(svg_text) and _SVG_CLOSE_RE.search(svg_text))


def render_structured_svg_atoms(text: str) -> str:
    return _render_structured_svg_atoms_v7(text)


def render_structured_scene_svg(text: str) -> str:
    return _render_structured_scene_svg_v7(text)


def render_structured_scene_rich_svg(text: str) -> str:
    return _render_structured_scene_rich_svg_v7(text)


def render_structured_scene_spec09_svg(text: str) -> str:
    return _render_structured_scene_spec09_svg_v7(text)


_SVG_DSL_RENDERERS: dict[str, Callable[[str], str]] = {
    "structured_svg_atoms.v1": render_structured_svg_atoms,
    "structured_svg_scene.v1": render_structured_scene_svg,
    "structured_svg_scene_rich.v1": render_structured_scene_rich_svg,
    "structured_svg_scene_spec09.v1": render_structured_scene_spec09_svg,
}


_RENDERER_MIME_TYPES: dict[str, str] = {
    "structured_svg_atoms.v1": "image/svg+xml",
    "structured_svg_scene.v1": "image/svg+xml",
    "structured_svg_scene_rich.v1": "image/svg+xml",
    "structured_svg_scene_spec09.v1": "image/svg+xml",
}


def render_registered_output(text: str, renderer_name: str) -> str:
    renderer = _SVG_DSL_RENDERERS.get(str(renderer_name or "").strip())
    if renderer is None:
        raise ValueError(f"unknown registered renderer: {renderer_name}")
    return renderer(text)


def guess_renderer_mime(renderer_name: str) -> str | None:
    mime = _RENDERER_MIME_TYPES.get(str(renderer_name or "").strip())
    return mime or None


def _adapt_svg_xml(text: str, config: dict[str, Any]) -> dict[str, Any]:
    svg_root, prefix_text, tail_text = extract_svg_root(text)
    parsed_output = svg_root if svg_root else str(text or "").strip()
    valid_svg = is_valid_svg(svg_root)
    return {
        "parsed_output": parsed_output,
        "materialized_output": svg_root if valid_svg else None,
        "materialized_mime": "image/svg+xml" if valid_svg else None,
        "renderable": valid_svg,
        "valid_materialized_output": valid_svg,
        "prefix_text": prefix_text,
        "tail_text": tail_text,
        "render_error": None if valid_svg else "No extractable SVG root",
    }


def _adapt_text_renderer(text: str, config: dict[str, Any]) -> dict[str, Any]:
    markers = [
        str(marker).strip()
        for marker in (config.get("stop_markers") or [])
        if str(marker).strip()
    ]
    parsed_output, tail_text = truncate_at_markers(text, markers) if markers else (str(text or "").strip(), "")
    renderer_name = str(config.get("renderer") or "").strip()
    materialized_output = None
    materialized_mime = None
    render_error = None
    renderable = False
    valid_materialized_output = False
    if renderer_name:
        try:
            materialized_output = render_registered_output(parsed_output, renderer_name)
            materialized_mime = str(config.get("preview_mime") or guess_renderer_mime(renderer_name) or "").strip() or None
            renderable = bool(materialized_output)
            valid_materialized_output = bool(materialized_output)
        except Exception as exc:  # pragma: no cover - surfaced in report output
            render_error = str(exc)
    return {
        "parsed_output": parsed_output,
        "materialized_output": materialized_output,
        "materialized_mime": materialized_mime,
        "renderable": renderable,
        "valid_materialized_output": valid_materialized_output,
        "prefix_text": "",
        "tail_text": tail_text,
        "render_error": render_error,
    }


def _adapt_plain_text(text: str, config: dict[str, Any]) -> dict[str, Any]:
    parsed_output = str(text or "").strip()
    return {
        "parsed_output": parsed_output,
        "materialized_output": None,
        "materialized_mime": None,
        "renderable": False,
        "valid_materialized_output": False,
        "prefix_text": "",
        "tail_text": "",
        "render_error": None,
    }


_OUTPUT_ADAPTERS: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]] = {
    "plain_text": _adapt_plain_text,
    "svg_xml": _adapt_svg_xml,
    "text_renderer": _adapt_text_renderer,
}


def apply_output_adapter(adapter_name: str, text: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    adapter = _OUTPUT_ADAPTERS.get(str(adapter_name or "").strip())
    if adapter is None:
        raise ValueError(f"unknown output adapter: {adapter_name}")
    return adapter(text, config if isinstance(config, dict) else {})
