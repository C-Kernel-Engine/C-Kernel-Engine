#!/usr/bin/env python3
"""Lower the structured scene DSL into spec06 atoms and render SVG."""

from __future__ import annotations

from typing import Any

from generate_svg_structured_spec06_v7 import (
    ACCENTS,
    BACKGROUNDS,
    DENSITIES,
    FRAMES,
    LAYOUTS,
    TOPIC_LIBRARY,
    Scene as AtomScene,
    _scene_output_tokens as _atom_scene_output_tokens,
)
from render_svg_structured_atoms_v7 import render_structured_svg_atoms


_COMPONENT_KEYS = {
    "hero",
    "rail",
    "bullet_list",
    "callout",
    "compare_pair",
    "footer",
    "stats",
    "band",
    "marker",
    "steps",
    "badge",
}

_REQUIRED_COMPONENTS = {
    "bullet-panel": {"hero", "rail", "bullet_list", "callout"},
    "compare-panels": {"hero", "compare_pair", "footer"},
    "stat-cards": {"hero", "stats", "callout"},
    "spectrum-band": {"hero", "band", "marker", "footer"},
    "flow-steps": {"hero", "steps", "badge"},
}


def _token_value(token: str, prefix: str) -> str | None:
    if token.startswith(prefix) and token.endswith("]"):
        return token[len(prefix) : -1]
    return None


def _parse_scene_document(text: str) -> dict[str, Any]:
    tokens = [tok.strip() for tok in str(text or "").split() if tok.strip()]
    if "[scene]" in tokens:
        tokens = tokens[tokens.index("[scene]") :]
    if not tokens or tokens[0] != "[scene]":
        raise ValueError("structured scene DSL must start with [scene]")
    if "[/scene]" not in tokens:
        raise ValueError("structured scene DSL must end with [/scene]")
    tokens = tokens[: tokens.index("[/scene]") + 1]

    scene_doc: dict[str, Any] = {
        "canvas": "wide",
        "theme": "paper",
        "layout": "",
        "topic": "",
        "accent": "blue",
        "frame": "plain",
        "density": "compact",
    }
    components: set[str] = set()
    for tok in tokens[1:]:
        if tok == "[/scene]":
            break
        value = _token_value(tok, "[canvas:")
        if value is not None:
            scene_doc["canvas"] = value
            continue
        value = _token_value(tok, "[theme:")
        if value is not None:
            scene_doc["theme"] = value
            continue
        value = _token_value(tok, "[bg:")
        if value is not None:
            scene_doc["theme"] = value
            continue
        value = _token_value(tok, "[layout:")
        if value is not None:
            scene_doc["layout"] = value
            continue
        value = _token_value(tok, "[topic:")
        if value is not None:
            scene_doc["topic"] = value
            continue
        value = _token_value(tok, "[accent:")
        if value is not None:
            scene_doc["accent"] = value
            continue
        value = _token_value(tok, "[frame:")
        if value is not None:
            scene_doc["frame"] = value
            continue
        value = _token_value(tok, "[density:")
        if value is not None:
            scene_doc["density"] = value
            continue
        if tok.startswith("[") and tok.endswith("]") and ":" in tok:
            components.add(tok[1:-1].split(":", 1)[0])

    layout = str(scene_doc["layout"] or "").strip()
    topic = str(scene_doc["topic"] or "").strip()
    theme = str(scene_doc["theme"] or "").strip()
    accent = str(scene_doc["accent"] or "").strip()
    frame = str(scene_doc["frame"] or "").strip()
    density = str(scene_doc["density"] or "").strip()

    if layout not in LAYOUTS:
        raise ValueError(f"unsupported scene layout: {layout or '<empty>'}")
    if topic not in TOPIC_LIBRARY:
        raise ValueError(f"unsupported scene topic: {topic or '<empty>'}")
    if theme not in BACKGROUNDS:
        raise ValueError(f"unsupported scene theme: {theme or '<empty>'}")
    if accent not in ACCENTS:
        raise ValueError(f"unsupported scene accent: {accent or '<empty>'}")
    if frame not in FRAMES:
        raise ValueError(f"unsupported scene frame: {frame or '<empty>'}")
    if density not in DENSITIES:
        raise ValueError(f"unsupported scene density: {density or '<empty>'}")

    missing = sorted(_REQUIRED_COMPONENTS.get(layout, set()) - components)
    if missing:
        raise ValueError(f"scene missing required components for {layout}: {', '.join(missing)}")

    return scene_doc


def compile_structured_scene_to_atoms(text: str) -> str:
    scene_doc = _parse_scene_document(text)
    scene = AtomScene(
        layout=str(scene_doc["layout"]),
        topic=str(scene_doc["topic"]),
        accent=str(scene_doc["accent"]),
        bg=str(scene_doc["theme"]),
        frame=str(scene_doc["frame"]),
        density=str(scene_doc["density"]),
    )
    return " ".join(_atom_scene_output_tokens(scene))


def render_structured_scene_svg(text: str) -> str:
    return render_structured_svg_atoms(compile_structured_scene_to_atoms(text))

