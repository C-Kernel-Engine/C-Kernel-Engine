#!/usr/bin/env python3
"""Parser + canonicalizer for the spec14a successor scene DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


LAYOUTS = {"table_matrix", "decision_tree", "memory_map", "flow_graph", "comparison_board"}
THEMES = {"paper_editorial", "infra_dark", "signal_glow"}
TONES = {"amber", "green", "blue", "purple", "mixed"}
DENSITIES = {"compact", "balanced", "airy"}
CANVAS = {"wide", "tall", "square"}
FRAMES = {"none", "card", "panel"}
BACKGROUNDS = {"grid", "mesh", "rings", "none"}
BLOCK_COMPONENTS = {
    "header_band",
    "legend_block",
    "table_block",
    "note_band",
    "entry_badge",
    "decision_node",
    "decision_edge",
    "outcome_panel",
    "footer_note",
    "address_strip",
    "memory_segment",
    "region_bracket",
    "info_card",
    "comparison_column",
    "comparison_metric",
    "comparison_callout",
}
COMPONENT_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "header_band": ("ref",),
    "legend_block": ("ref",),
    "table_block": ("ref",),
    "note_band": ("ref",),
    "entry_badge": ("ref",),
    "decision_node": ("node_id", "ref"),
    "decision_edge": ("from_ref", "to_ref", "label_ref"),
    "outcome_panel": ("panel_id", "ref"),
    "footer_note": ("ref",),
    "address_strip": ("ref",),
    "memory_segment": ("segment_id", "ref"),
    "region_bracket": ("ref",),
    "info_card": ("card_id", "ref"),
    "comparison_column": ("column_id", "ref"),
    "comparison_metric": ("metric_id", "ref"),
    "comparison_callout": ("callout_id", "ref"),
}


def _token_value(token: str, prefix: str) -> str | None:
    if token.startswith(prefix) and token.endswith("]"):
        return token[len(prefix) : -1]
    return None


def _split_payload(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split("|") if part.strip()]


def _normalize_scene_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    while "][" in cleaned:
        cleaned = cleaned.replace("][", "] [")
    return " ".join(cleaned.split())


@dataclass(frozen=True)
class SceneComponent:
    name: str
    fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneDocument:
    canvas: str = "wide"
    layout: str = ""
    theme: str = "infra_dark"
    tone: str = "blue"
    frame: str = "panel"
    density: str = "balanced"
    inset: str = "md"
    gap: str = "md"
    hero: str = "left"
    columns: str = "1"
    emphasis: str = "top"
    rail: str = "none"
    background: str = "none"
    connector: str = "line"
    topic: str = ""
    components: tuple[SceneComponent, ...] = ()

    def to_runtime(self) -> dict[str, Any]:
        by_name: dict[str, list[dict[str, str]]] = {}
        component_rows: list[tuple[str, dict[str, str]]] = []
        for component in self.components:
            entry = dict(component.fields)
            by_name.setdefault(component.name, []).append(entry)
            component_rows.append((component.name, entry))
        return {
            "canvas": self.canvas,
            "layout": self.layout,
            "theme": self.theme,
            "tone": self.tone,
            "frame": self.frame,
            "density": self.density,
            "inset": self.inset,
            "gap": self.gap,
            "hero": self.hero,
            "columns": self.columns,
            "emphasis": self.emphasis,
            "rail": self.rail,
            "background": self.background,
            "connector": self.connector,
            "topic": self.topic,
            "components": component_rows,
            "components_by_name": by_name,
            "_content": {},
        }


def _append_legacy_component(name: str, raw: str) -> SceneComponent:
    parts = _split_payload(raw)
    if name in {"header_band", "legend_block", "table_block", "note_band", "entry_badge", "footer_note", "address_strip"}:
        fields = {"ref": parts[0]} if parts else {}
    elif name == "decision_node":
        fields = {"node_id": parts[0], "ref": parts[1]} if len(parts) >= 2 else {}
    elif name == "decision_edge":
        if len(parts) >= 2 and "->" in parts[0]:
            src, dst = parts[0].split("->", 1)
            fields = {"from_ref": src, "to_ref": dst, "label_ref": parts[1]}
        else:
            fields = {}
    elif name == "outcome_panel":
        fields = {"panel_id": parts[0], "ref": parts[1]} if len(parts) >= 2 else {}
    elif name == "memory_segment":
        fields = {"segment_id": parts[0], "ref": parts[1]} if len(parts) >= 2 else {}
    elif name == "region_bracket":
        fields = {"bracket_id": parts[0], "ref": parts[1]} if len(parts) >= 2 else {"ref": parts[0]} if parts else {}
    elif name == "info_card":
        fields = {"card_id": parts[0], "ref": parts[1]} if len(parts) >= 2 else {}
    elif name == "comparison_column":
        fields = {"column_id": parts[0], "ref": parts[1]} if len(parts) >= 2 else {}
    elif name == "comparison_metric":
        fields = {"metric_id": parts[0], "ref": parts[1]} if len(parts) >= 2 else {}
    elif name == "comparison_callout":
        fields = {"callout_id": parts[0], "ref": parts[1]} if len(parts) >= 2 else {}
    else:
        fields = {"raw": raw}
    return SceneComponent(name=name, fields=fields)


def parse_scene_document(text: str) -> SceneDocument:
    tokens = [tok.strip() for tok in _normalize_scene_text(text).split() if tok.strip()]
    if "[scene]" in tokens:
        tokens = tokens[tokens.index("[scene]") :]
    if not tokens or tokens[0] != "[scene]":
        raise ValueError("spec14a scene DSL must start with [scene]")
    if "[/scene]" not in tokens:
        raise ValueError("spec14a scene DSL must end with [/scene]")
    tokens = tokens[: tokens.index("[/scene]") + 1]

    scene: dict[str, Any] = {
        "canvas": "wide",
        "layout": "",
        "theme": "infra_dark",
        "tone": "blue",
        "frame": "panel",
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
    }
    components: list[SceneComponent] = []
    active_name: str | None = None
    active_fields: dict[str, str] | None = None

    for token in tokens[1:-1]:
        if active_name is not None:
            if token == f"[/{active_name}]":
                components.append(SceneComponent(name=active_name, fields=dict(active_fields or {})))
                active_name = None
                active_fields = None
                continue
            if token.startswith("[") and token.endswith("]") and ":" in token:
                key, value = token[1:-1].split(":", 1)
                if active_fields is None:
                    active_fields = {}
                active_fields[key.strip()] = value.strip()
                continue
            raise ValueError(f"unsupported token inside {active_name}: {token}")

        handled = False
        for key, allowed in {
            "canvas": CANVAS,
            "layout": LAYOUTS,
            "theme": THEMES,
            "tone": TONES,
            "frame": FRAMES,
            "density": DENSITIES,
            "background": BACKGROUNDS,
        }.items():
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

        for key in ("inset", "gap", "hero", "columns", "emphasis", "rail", "connector", "topic"):
            value = _token_value(token, f"[{key}:")
            if value is not None:
                scene[key] = value
                handled = True
                break
        if handled:
            continue

        if token.startswith("[/") and token.endswith("]"):
            raise ValueError(f"unexpected closing token outside component block: {token}")
        if token.startswith("[") and token.endswith("]") and ":" not in token:
            name = token[1:-1].strip()
            if name in BLOCK_COMPONENTS:
                active_name = name
                active_fields = {}
                continue
        if token.startswith("[") and token.endswith("]") and ":" in token:
            name, raw = token[1:-1].split(":", 1)
            components.append(_append_legacy_component(name.strip(), raw.strip()))
            continue
        raise ValueError(f"unsupported token in spec14a scene: {token}")

    if active_name is not None:
        raise ValueError(f"unclosed component block: {active_name}")

    return SceneDocument(components=tuple(components), **scene)


def canonicalize_scene(scene: SceneDocument) -> SceneDocument:
    if scene.layout not in LAYOUTS:
        raise ValueError(f"unsupported spec14a layout: {scene.layout or '<empty>'}")
    if scene.theme not in THEMES:
        raise ValueError(f"unsupported spec14a theme: {scene.theme}")
    if scene.tone not in TONES:
        raise ValueError(f"unsupported spec14a tone: {scene.tone}")
    if scene.density not in DENSITIES:
        raise ValueError(f"unsupported spec14a density: {scene.density}")
    if scene.canvas not in CANVAS:
        raise ValueError(f"unsupported spec14a canvas: {scene.canvas}")
    if scene.frame not in FRAMES:
        raise ValueError(f"unsupported spec14a frame: {scene.frame}")
    if scene.background not in BACKGROUNDS:
        raise ValueError(f"unsupported spec14a background: {scene.background}")

    validated: list[SceneComponent] = []
    for component in scene.components:
        if component.name not in BLOCK_COMPONENTS:
            raise ValueError(f"unsupported spec14a component: {component.name}")
        required = COMPONENT_REQUIRED_FIELDS.get(component.name, ())
        missing = [field for field in required if not str(component.fields.get(field) or "").strip()]
        if missing:
            raise ValueError(f"{component.name} missing required fields: {', '.join(missing)}")
        validated.append(SceneComponent(name=component.name, fields=dict(component.fields)))

    return SceneDocument(
        canvas=scene.canvas,
        layout=scene.layout,
        theme=scene.theme,
        tone=scene.tone,
        frame=scene.frame,
        density=scene.density,
        inset=scene.inset,
        gap=scene.gap,
        hero=scene.hero,
        columns=scene.columns,
        emphasis=scene.emphasis,
        rail=scene.rail,
        background=scene.background,
        connector=scene.connector,
        topic=scene.topic,
        components=tuple(validated),
    )


def canonicalize_scene_text(text: str) -> SceneDocument:
    return canonicalize_scene(parse_scene_document(text))
