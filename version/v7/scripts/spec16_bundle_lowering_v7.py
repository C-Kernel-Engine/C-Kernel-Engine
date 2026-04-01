#!/usr/bin/env python3
"""Lower spec16 shared scene bundles into family-specific scene DSL."""

from __future__ import annotations

from typing import Any

try:
    from spec14b_decode_validate_rerank_v7 import _component_token_from_fields as _spec14b_component_token
    from spec14b_decode_validate_rerank_v7 import _emit_scene_text as _emit_spec14b_scene_text
    from spec14b_decode_validate_rerank_v7 import _form_templates as _spec14b_form_templates
    from spec14b_scene_canonicalizer_v7 import canonicalize_scene_text as _canonicalize_spec14b_scene_text
    from spec15a_decode_validate_rerank_v7 import _component_token_from_fields as _spec15a_component_token
    from spec15a_decode_validate_rerank_v7 import _form_templates as _spec15a_form_templates
    from spec15a_scene_canonicalizer_v7 import canonicalize_scene_text as _canonicalize_spec15a_scene_text
    from spec15b_decode_validate_rerank_v7 import _component_token_from_fields as _spec15b_component_token
    from spec15b_decode_validate_rerank_v7 import _emit_scene_text as _emit_spec15b_scene_text
    from spec15b_decode_validate_rerank_v7 import _form_templates as _spec15b_form_templates
    from spec15b_scene_canonicalizer_v7 import canonicalize_scene_text as _canonicalize_spec15b_scene_text
except ModuleNotFoundError:  # pragma: no cover
    from version.v7.scripts.spec14b_decode_validate_rerank_v7 import _component_token_from_fields as _spec14b_component_token
    from version.v7.scripts.spec14b_decode_validate_rerank_v7 import _emit_scene_text as _emit_spec14b_scene_text
    from version.v7.scripts.spec14b_decode_validate_rerank_v7 import _form_templates as _spec14b_form_templates
    from version.v7.scripts.spec14b_scene_canonicalizer_v7 import canonicalize_scene_text as _canonicalize_spec14b_scene_text
    from version.v7.scripts.spec15a_decode_validate_rerank_v7 import _component_token_from_fields as _spec15a_component_token
    from version.v7.scripts.spec15a_decode_validate_rerank_v7 import _form_templates as _spec15a_form_templates
    from version.v7.scripts.spec15a_scene_canonicalizer_v7 import canonicalize_scene_text as _canonicalize_spec15a_scene_text
    from version.v7.scripts.spec15b_decode_validate_rerank_v7 import _component_token_from_fields as _spec15b_component_token
    from version.v7.scripts.spec15b_decode_validate_rerank_v7 import _emit_scene_text as _emit_spec15b_scene_text
    from version.v7.scripts.spec15b_decode_validate_rerank_v7 import _form_templates as _spec15b_form_templates
    from version.v7.scripts.spec15b_scene_canonicalizer_v7 import canonicalize_scene_text as _canonicalize_spec15b_scene_text

try:
    from spec16_scene_bundle_v7 import SceneBundle, canonicalize_scene_bundle
except ModuleNotFoundError:  # pragma: no cover
    from version.v7.scripts.spec16_scene_bundle_v7 import SceneBundle, canonicalize_scene_bundle


def _bundle_prompt_map(bundle: SceneBundle) -> dict[str, str]:
    prompt_bundle = {
        "layout": bundle.family,
        "form": bundle.form,
        "theme": bundle.theme,
        "tone": bundle.tone,
        "density": bundle.density,
        "background": bundle.background,
    }
    for key, value in bundle.topology.items():
        prompt_bundle[key] = str(value)
    return prompt_bundle


def lower_scene_bundle_to_scene_dsl(bundle_doc: dict[str, Any] | SceneBundle) -> str:
    bundle = bundle_doc if isinstance(bundle_doc, SceneBundle) else canonicalize_scene_bundle(bundle_doc)
    prompt_bundle = _bundle_prompt_map(bundle)

    if bundle.family == "memory_map":
        template = _spec15a_form_templates()[bundle.form]
        scene_text = _emit_spec15a_memory_map(prompt_bundle, template)
        canonical = _canonicalize_spec15a_scene_text(scene_text)
        return _serialize_spec15a_scene(canonical)

    if bundle.family == "timeline":
        template = _spec14b_form_templates()[bundle.form]
        scene_text = _emit_spec14b_scene_text(prompt_bundle, template)
        canonical = _canonicalize_spec14b_scene_text(scene_text)
        return _serialize_spec14b_scene(canonical)

    if bundle.family == "system_diagram":
        template = _spec15b_form_templates()[bundle.form]
        scene_text = _emit_spec15b_scene_text(prompt_bundle, template)
        canonical = _canonicalize_spec15b_scene_text(scene_text)
        return _serialize_spec15b_scene(canonical)

    raise ValueError(f"unsupported spec16 family: {bundle.family!r}")


def _emit_spec15a_memory_map(prompt_bundle: dict[str, str], template: dict[str, Any]) -> str:
    scene = template["scene"]
    background_value = str(prompt_bundle.get("background") or scene.background).strip() or scene.background
    scene_tokens = [
        "[scene]",
        f"[layout:{str(prompt_bundle.get('layout') or 'memory_map').strip() or 'memory_map'}]",
        f"[theme:{str(prompt_bundle.get('theme') or scene.theme).strip() or scene.theme}]",
        f"[tone:{str(prompt_bundle.get('tone') or scene.tone).strip() or scene.tone}]",
        f"[density:{str(prompt_bundle.get('density') or scene.density).strip() or scene.density}]",
        f"[topic:{scene.topic}]",
        *list(template["component_tokens"]),
        "[/scene]",
    ]
    if background_value and background_value != "none":
        scene_tokens.insert(5, f"[background:{background_value}]")
    return " ".join(scene_tokens)


def _serialize_spec14b_scene(scene: Any) -> str:
    tokens = [
        "[scene]",
        f"[canvas:{scene.canvas}]",
        f"[layout:{scene.layout}]",
        f"[theme:{scene.theme}]",
        f"[tone:{scene.tone}]",
        f"[frame:{scene.frame}]",
        f"[density:{scene.density}]",
        f"[inset:{scene.inset}]",
        f"[gap:{scene.gap}]",
        f"[hero:{scene.hero}]",
        f"[columns:{scene.columns}]",
        f"[emphasis:{scene.emphasis}]",
        f"[rail:{scene.rail}]",
        f"[background:{scene.background}]",
        f"[connector:{scene.connector}]",
        f"[topic:{scene.topic}]",
        *[_spec14b_component_token(component.name, component.fields) for component in scene.components],
        "[/scene]",
    ]
    return " ".join(tokens)


def _serialize_spec15a_scene(scene: Any) -> str:
    tokens = [
        "[scene]",
        f"[layout:{scene.layout}]",
        f"[theme:{scene.theme}]",
        f"[tone:{scene.tone}]",
        f"[density:{scene.density}]",
    ]
    if str(scene.background).strip() and str(scene.background).strip() != "none":
        tokens.append(f"[background:{scene.background}]")
    tokens.extend(
        [
            f"[topic:{scene.topic}]",
            *[_spec15a_component_token(component.name, component.fields) for component in scene.components],
            "[/scene]",
        ]
    )
    return " ".join(tokens)


def _serialize_spec15b_scene(scene: Any) -> str:
    tokens = [
        "[scene]",
        f"[canvas:{scene.canvas}]",
        f"[layout:{scene.layout}]",
        f"[theme:{scene.theme}]",
        f"[tone:{scene.tone}]",
        f"[frame:{scene.frame}]",
        f"[density:{scene.density}]",
        f"[inset:{scene.inset}]",
        f"[gap:{scene.gap}]",
        f"[background:{scene.background}]",
        f"[connector:{scene.connector}]",
        f"[topic:{scene.topic}]",
        *[_spec15b_component_token(component.name, component.fields) for component in scene.components],
        "[/scene]",
    ]
    return " ".join(tokens)


__all__ = [
    "lower_scene_bundle_to_scene_dsl",
]
