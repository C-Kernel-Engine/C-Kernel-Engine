#!/usr/bin/env python3
"""Parser + canonicalizer for the spec15b system_diagram scene DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from scene_dsl_pre_repair_v7 import repair_compact_scene_text
except ModuleNotFoundError:  # pragma: no cover - import path fallback for module-style execution
    from version.v7.scripts.scene_dsl_pre_repair_v7 import repair_compact_scene_text


GENERIC_TOPIC = "system_diagram_generic"
LAYOUTS = {"system_diagram"}
THEMES = {"paper_editorial", "infra_dark", "signal_glow"}
TONES = {"amber", "green", "blue", "purple", "mixed"}
DENSITIES = {"compact", "balanced", "airy"}
CANVAS = {"wide"}
FRAMES = {"none", "card", "panel"}
BACKGROUNDS = {"grid", "mesh", "rings", "none"}
BLOCK_COMPONENTS = {
    "header_band",
    "system_stage",
    "system_link",
    "terminal_panel",
    "footer_note",
}
COMPONENT_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "header_band": ("ref",),
    "system_stage": ("stage_id", "ref"),
    "system_link": ("from_stage", "to_stage", "ref"),
    "terminal_panel": ("panel_id", "ref"),
    "footer_note": ("ref",),
}


def _token_value(token: str, prefix: str) -> str | None:
    if token.startswith(prefix) and token.endswith("]"):
        return token[len(prefix) : -1]
    return None


def _split_payload(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split("|") if part.strip()]


def _normalize_scene_text(text: str) -> str:
    repair = repair_compact_scene_text(
        text,
        singleton_names={
            "scene",
            "canvas",
            "layout",
            "theme",
            "tone",
            "frame",
            "density",
            "inset",
            "gap",
            "background",
            "connector",
            "topic",
            "header_band",
            "terminal_panel",
            "footer_note",
        },
        prompt_only_names={"task", "form", "stages", "links", "terminal", "footer", "out"},
    )
    return " ".join(str(repair.get("repaired_text") or "").split())


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
    background: str = "none"
    connector: str = "arrow"
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
            "background": self.background,
            "connector": self.connector,
            "topic": self.topic,
            "components": component_rows,
            "components_by_name": by_name,
            "_content": {},
        }


def _append_legacy_component(name: str, raw: str) -> SceneComponent:
    parts = _split_payload(raw)
    if name in {"header_band", "footer_note"}:
        fields = {"ref": parts[0]} if parts else {}
    elif name == "system_stage":
        fields = {}
        if len(parts) >= 2:
            fields["stage_id"] = parts[0]
            fields["ref"] = parts[1]
    elif name == "terminal_panel":
        fields = {}
        if len(parts) >= 2:
            fields["panel_id"] = parts[0]
            fields["ref"] = parts[1]
    elif name == "system_link":
        fields = {}
        if len(parts) >= 2:
            link = parts[0]
            if "->" in link:
                src, dst = link.split("->", 1)
                fields["from_stage"] = src.strip()
                fields["to_stage"] = dst.strip()
            fields["ref"] = parts[1]
    else:
        fields = {"raw": raw}
    return SceneComponent(name=name, fields=fields)


def parse_scene_document(text: str) -> SceneDocument:
    tokens = [tok.strip() for tok in _normalize_scene_text(text).split() if tok.strip()]
    if "[scene]" in tokens:
        tokens = tokens[tokens.index("[scene]") :]
    if not tokens or tokens[0] != "[scene]":
        raise ValueError("spec15b scene DSL must start with [scene]")
    if "[/scene]" not in tokens:
        raise ValueError("spec15b scene DSL must end with [/scene]")
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
        "background": "none",
        "connector": "arrow",
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

        for key in ("inset", "gap", "connector", "topic"):
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
        raise ValueError(f"unsupported token in spec15b scene: {token}")

    if active_name is not None:
        raise ValueError(f"unclosed component block: {active_name}")

    return SceneDocument(components=tuple(components), **scene)


def canonicalize_scene(scene: SceneDocument) -> SceneDocument:
    if scene.layout not in LAYOUTS:
        raise ValueError(f"unsupported spec15b layout: {scene.layout or '<empty>'}")
    if scene.theme not in THEMES:
        raise ValueError(f"unsupported spec15b theme: {scene.theme}")
    if scene.tone not in TONES:
        raise ValueError(f"unsupported spec15b tone: {scene.tone}")
    if scene.density not in DENSITIES:
        raise ValueError(f"unsupported spec15b density: {scene.density}")
    if scene.canvas not in CANVAS:
        raise ValueError(f"unsupported spec15b canvas: {scene.canvas}")
    if scene.frame not in FRAMES:
        raise ValueError(f"unsupported spec15b frame: {scene.frame}")
    if scene.background not in BACKGROUNDS:
        raise ValueError(f"unsupported spec15b background: {scene.background}")

    validated: list[SceneComponent] = []
    stage_ids: list[str] = []
    panel_ids: list[str] = []
    link_pairs: list[tuple[str, str]] = []
    for component in scene.components:
        if component.name not in BLOCK_COMPONENTS:
            raise ValueError(f"unsupported spec15b component: {component.name}")
        required = COMPONENT_REQUIRED_FIELDS.get(component.name, ())
        missing = [field for field in required if not str(component.fields.get(field) or "").strip()]
        if missing:
            raise ValueError(f"{component.name} missing required fields: {', '.join(missing)}")
        fields = dict(component.fields)
        if component.name == "system_stage":
            stage_id = str(fields.get("stage_id") or "").strip()
            if stage_id in stage_ids:
                raise ValueError(f"duplicate system_stage stage_id: {stage_id}")
            stage_ids.append(stage_id)
        elif component.name == "terminal_panel":
            panel_id = str(fields.get("panel_id") or "").strip()
            if panel_id in panel_ids:
                raise ValueError(f"duplicate terminal_panel panel_id: {panel_id}")
            panel_ids.append(panel_id)
        elif component.name == "system_link":
            pair = (str(fields.get("from_stage") or "").strip(), str(fields.get("to_stage") or "").strip())
            if pair in link_pairs:
                raise ValueError(f"duplicate system_link pair: {pair[0]}->{pair[1]}")
            link_pairs.append(pair)
        validated.append(SceneComponent(name=component.name, fields=fields))

    if len(stage_ids) < 2:
        raise ValueError("spec15b system_diagram requires at least 2 system_stage components")
    if len(panel_ids) != 1:
        raise ValueError("spec15b system_diagram requires exactly one terminal_panel component")
    if len(link_pairs) < len(stage_ids):
        raise ValueError("spec15b system_diagram requires at least stage_count system_link components")
    valid_targets = set(stage_ids) | set(panel_ids)
    for src, dst in link_pairs:
        if src not in stage_ids:
            raise ValueError(f"system_link references unknown source stage id: {src}")
        if dst not in valid_targets:
            raise ValueError(f"system_link references unknown target id: {dst}")

    return SceneDocument(
        canvas=scene.canvas,
        layout=scene.layout,
        theme=scene.theme,
        tone=scene.tone,
        frame=scene.frame,
        density=scene.density,
        inset=scene.inset,
        gap=scene.gap,
        background=scene.background,
        connector=scene.connector,
        topic=GENERIC_TOPIC,
        components=tuple(validated),
    )


def canonicalize_scene_text(text: str) -> SceneDocument:
    return canonicalize_scene(parse_scene_document(text))
