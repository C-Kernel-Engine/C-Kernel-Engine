#!/usr/bin/env python3
"""Parser + canonicalizer for the spec14b timeline scene DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from scene_dsl_pre_repair_v7 import repair_compact_scene_text
except ModuleNotFoundError:  # pragma: no cover - import path fallback for module-style execution
    from version.v7.scripts.scene_dsl_pre_repair_v7 import repair_compact_scene_text


GENERIC_TOPIC = "timeline_generic"
LAYOUTS = {"timeline"}
THEMES = {"paper_editorial", "infra_dark", "signal_glow"}
TONES = {"amber", "green", "blue", "purple", "mixed"}
DENSITIES = {"compact", "balanced", "airy"}
CANVAS = {"wide"}
FRAMES = {"none", "card", "panel"}
BACKGROUNDS = {"grid", "mesh", "rings", "none"}
STAGE_LANES = {"top", "bottom", "center"}
BLOCK_COMPONENTS = {
    "header_band",
    "timeline_stage",
    "timeline_arrow",
    "footer_note",
}
COMPONENT_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "header_band": ("ref",),
    "timeline_stage": ("stage_id", "ref"),
    "timeline_arrow": ("from_stage", "to_stage"),
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
            "hero",
            "columns",
            "emphasis",
            "rail",
            "background",
            "connector",
            "topic",
            "header_band",
            "footer_note",
        },
        prompt_only_names={"task", "form", "stages", "arrows", "footer", "out"},
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
    hero: str = "left"
    columns: str = "1"
    emphasis: str = "top"
    rail: str = "none"
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
    if name in {"header_band", "footer_note"}:
        fields = {"ref": parts[0]} if parts else {}
    elif name == "timeline_stage":
        fields: dict[str, str] = {}
        if len(parts) >= 2:
            fields["stage_id"] = parts[0]
            fields["ref"] = parts[1]
        for part in parts[2:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
    elif name == "timeline_arrow":
        fields = {}
        if parts:
            arc = parts[0]
            if "->" in arc:
                src, dst = arc.split("->", 1)
                fields["from_stage"] = src.strip()
                fields["to_stage"] = dst.strip()
    else:
        fields = {"raw": raw}
    return SceneComponent(name=name, fields=fields)


def parse_scene_document(text: str) -> SceneDocument:
    tokens = [tok.strip() for tok in _normalize_scene_text(text).split() if tok.strip()]
    if "[scene]" in tokens:
        tokens = tokens[tokens.index("[scene]") :]
    if not tokens or tokens[0] != "[scene]":
        raise ValueError("spec14b scene DSL must start with [scene]")
    if "[/scene]" not in tokens:
        raise ValueError("spec14b scene DSL must end with [/scene]")
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
        raise ValueError(f"unsupported token in spec14b scene: {token}")

    if active_name is not None:
        raise ValueError(f"unclosed component block: {active_name}")

    return SceneDocument(components=tuple(components), **scene)


def canonicalize_scene(scene: SceneDocument) -> SceneDocument:
    if scene.layout not in LAYOUTS:
        raise ValueError(f"unsupported spec14b layout: {scene.layout or '<empty>'}")
    if scene.theme not in THEMES:
        raise ValueError(f"unsupported spec14b theme: {scene.theme}")
    if scene.tone not in TONES:
        raise ValueError(f"unsupported spec14b tone: {scene.tone}")
    if scene.density not in DENSITIES:
        raise ValueError(f"unsupported spec14b density: {scene.density}")
    if scene.canvas not in CANVAS:
        raise ValueError(f"unsupported spec14b canvas: {scene.canvas}")
    if scene.frame not in FRAMES:
        raise ValueError(f"unsupported spec14b frame: {scene.frame}")
    if scene.background not in BACKGROUNDS:
        raise ValueError(f"unsupported spec14b background: {scene.background}")

    validated: list[SceneComponent] = []
    stage_ids: list[str] = []
    arrow_pairs: list[tuple[str, str]] = []
    for component in scene.components:
        if component.name not in BLOCK_COMPONENTS:
            raise ValueError(f"unsupported spec14b component: {component.name}")
        required = COMPONENT_REQUIRED_FIELDS.get(component.name, ())
        missing = [field for field in required if not str(component.fields.get(field) or "").strip()]
        if missing:
            raise ValueError(f"{component.name} missing required fields: {', '.join(missing)}")
        fields = dict(component.fields)
        if component.name == "timeline_stage":
            lane = str(fields.get("lane") or "center").strip()
            if lane not in STAGE_LANES:
                raise ValueError(f"unsupported timeline_stage lane: {lane}")
            fields["lane"] = lane
            stage_id = str(fields.get("stage_id") or "").strip()
            if stage_id in stage_ids:
                raise ValueError(f"duplicate timeline_stage stage_id: {stage_id}")
            stage_ids.append(stage_id)
        if component.name == "timeline_arrow":
            pair = (str(fields.get("from_stage") or "").strip(), str(fields.get("to_stage") or "").strip())
            arrow_pairs.append(pair)
        validated.append(SceneComponent(name=component.name, fields=fields))

    if len(stage_ids) < 2:
        raise ValueError("spec14b timeline requires at least 2 timeline_stage components")
    if len(arrow_pairs) < max(1, len(stage_ids) - 1):
        raise ValueError("spec14b timeline requires at least stage_count-1 timeline_arrow components")
    for src, dst in arrow_pairs:
        if src not in stage_ids or dst not in stage_ids:
            raise ValueError(f"timeline_arrow references unknown stage ids: {src}->{dst}")

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
        topic=GENERIC_TOPIC,
        components=tuple(validated),
    )


def canonicalize_scene_text(text: str) -> SceneDocument:
    return canonicalize_scene(parse_scene_document(text))
