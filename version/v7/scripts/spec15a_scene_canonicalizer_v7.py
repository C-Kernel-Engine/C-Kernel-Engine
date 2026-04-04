#!/usr/bin/env python3
"""Strict family-generic canonicalizer for spec15a memory_map scene DSL."""

from __future__ import annotations

try:
    from scene_dsl_pre_repair_v7 import repair_compact_scene_text
except ModuleNotFoundError:  # pragma: no cover - import path fallback for module-style execution
    from version.v7.scripts.scene_dsl_pre_repair_v7 import repair_compact_scene_text
try:
    from spec12_scene_canonicalizer_v7 import (
        SceneComponent,
        SceneDocument,
        canonicalize_scene as _canonicalize_spec12_scene,
        parse_scene_document as _parse_spec12_scene_document,
    )
except ModuleNotFoundError:  # pragma: no cover - import path fallback for module-style execution
    from version.v7.scripts.spec12_scene_canonicalizer_v7 import (
        SceneComponent,
        SceneDocument,
        canonicalize_scene as _canonicalize_spec12_scene,
        parse_scene_document as _parse_spec12_scene_document,
    )


GENERIC_TOPIC = "memory_map_generic"
ALLOWED_LAYOUT = "memory_map"
ALLOWED_COMPONENTS = {
    "header_band",
    "address_strip",
    "memory_segment",
    "region_bracket",
    "info_card",
}


def _ref_prefix(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parts = raw.split(".")
    if len(parts) <= 1:
        return raw
    return ".".join(parts[:-1])


def _normalize_component(component: SceneComponent) -> SceneComponent:
    fields = dict(component.fields)
    name = component.name
    if name == "header_band" and not fields.get("ref"):
        fields["ref"] = _ref_prefix(fields.get("headline_ref") or fields.get("subtitle_ref") or "header")
    elif name == "address_strip" and not fields.get("ref"):
        fields["ref"] = _ref_prefix(fields.get("offset_ref") or "offsets.0")
    elif name == "memory_segment" and not fields.get("ref"):
        fields["ref"] = _ref_prefix(
            fields.get("title_ref")
            or fields.get("size_ref")
            or fields.get("caption_ref")
            or fields.get("subregion_ref")
            or ""
        )
    elif name == "region_bracket" and not fields.get("ref"):
        fields["ref"] = _ref_prefix(fields.get("title_ref") or fields.get("caption_ref") or "")
    elif name == "info_card" and not fields.get("ref"):
        fields["ref"] = _ref_prefix(
            fields.get("title_ref") or fields.get("bullet_ref") or fields.get("code_ref") or ""
        )
    if name == "info_card" and not fields.get("card_id"):
        ref = str(fields.get("ref") or "").strip()
        fields["card_id"] = ref.split(".")[-1] if ref else "card"
    return SceneComponent(name=name, fields=fields)


def canonicalize_scene_text(text: str) -> SceneDocument:
    """Canonicalize a spec15a scene by restricting spec12 memory_map to a generic family contract."""
    repaired = repair_compact_scene_text(
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
            "address_strip",
        },
        prompt_only_names={"task", "form", "segments", "brackets", "cards", "out"},
    )
    parsed_scene = _parse_spec12_scene_document(str(repaired.get("repaired_text") or ""))
    normalized_spec12_scene = SceneDocument(
        canvas=parsed_scene.canvas,
        layout=parsed_scene.layout,
        theme=parsed_scene.theme,
        tone=parsed_scene.tone,
        frame=parsed_scene.frame,
        density=parsed_scene.density,
        inset=parsed_scene.inset,
        gap=parsed_scene.gap,
        hero=parsed_scene.hero,
        columns=parsed_scene.columns,
        emphasis=parsed_scene.emphasis,
        rail=parsed_scene.rail,
        background=parsed_scene.background,
        connector=parsed_scene.connector,
        topic=parsed_scene.topic,
        components=tuple(_normalize_component(component) for component in parsed_scene.components),
    )
    scene = _canonicalize_spec12_scene(normalized_spec12_scene)
    if scene.layout != ALLOWED_LAYOUT:
        raise ValueError(f"spec15a only supports [{ALLOWED_LAYOUT}] scenes")

    normalized_components = []
    for component in scene.components:
        if component.name not in ALLOWED_COMPONENTS:
            raise ValueError(
                f"spec15a memory_map does not allow component {component.name!r}; "
                f"allowed={sorted(ALLOWED_COMPONENTS)}"
            )
        normalized_components.append(_normalize_component(component))

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
        components=tuple(normalized_components),
    )
