#!/usr/bin/env python3
"""Spec15a decode-time validation and near-miss scene repair helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from spec15a_scene_canonicalizer_v7 import SceneDocument, canonicalize_scene_text
except ModuleNotFoundError:  # pragma: no cover
    from version.v7.scripts.spec15a_scene_canonicalizer_v7 import SceneDocument, canonicalize_scene_text


ROOT = Path(__file__).resolve().parents[3]
GOLD = ROOT / "version" / "v7" / "reports" / "spec15a_gold_mappings"
GENERIC_TOPIC = "memory_map_generic"

_PROMPT_TAG_RE = re.compile(r"\[([a-z_]+):([^\]]+)\]")
_BRACKET_TOKEN_RE = re.compile(r"\[[^\[\]\n]+\]")
_SCENE_TOKEN_NAMES = {
    "scene",
    "/scene",
    "layout",
    "theme",
    "tone",
    "density",
    "background",
    "topic",
    "header_band",
    "address_strip",
    "memory_segment",
    "region_bracket",
    "info_card",
}
_PROMPT_ONLY_TOKEN_NAMES = {
    "task",
    "form",
    "segments",
    "brackets",
    "cards",
    "out",
}
_FORM_TEMPLATE_FILES = {
    "layer_stack": GOLD / "memory-layout-map.scene.compact.dsl",
    "typed_regions": GOLD / "bump_allocator_quant.scene.compact.dsl",
    "arena_sections": GOLD / "v7-train-memory-canary.scene.compact.dsl",
}


def _payload_lookup(content: dict[str, Any] | None, ref: str) -> Any:
    current: Any = content or {}
    for part in str(ref or "").strip().split("."):
        if not part:
            return None
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def _component_ref(name: str, fields: dict[str, str]) -> str:
    if name in {"header_band", "address_strip"}:
        return str(fields.get("ref") or "").strip()
    if name == "memory_segment":
        return str(fields.get("ref") or "").strip()
    if name == "region_bracket":
        return str(fields.get("ref") or "").strip()
    if name == "info_card":
        return str(fields.get("ref") or "").strip()
    return ""


def _parse_prompt_bundle(prompt: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in _PROMPT_TAG_RE.findall(str(prompt or "")):
        out[str(key).strip().lower()] = str(value).strip()
    return out


def _split_token(token: str) -> tuple[str, str]:
    body = str(token or "").strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    if ":" not in body:
        return body.strip().lower(), ""
    name, payload = body.split(":", 1)
    return name.strip().lower(), payload.strip()


def _component_token_from_fields(name: str, fields: dict[str, str]) -> str:
    if name in {"header_band", "address_strip"}:
        payload = str(fields.get("ref") or "").strip()
    elif name == "memory_segment":
        payload = f"{str(fields.get('segment_id') or '').strip()}|{str(fields.get('ref') or '').strip()}"
    elif name == "region_bracket":
        payload = f"{str(fields.get('bracket_id') or '').strip()}|{str(fields.get('ref') or '').strip()}"
    elif name == "info_card":
        payload = f"{str(fields.get('card_id') or '').strip()}|{str(fields.get('ref') or '').strip()}"
    else:
        raise ValueError(f"unsupported spec15a component: {name}")
    return f"[{name}:{payload}]"


@lru_cache(maxsize=8)
def _form_templates() -> dict[str, dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    for form_token, path in _FORM_TEMPLATE_FILES.items():
        scene = canonicalize_scene_text(path.read_text(encoding="utf-8"))
        component_tokens = []
        for component in scene.components:
            component_tokens.append(_component_token_from_fields(component.name, component.fields))
        templates[form_token] = {
            "scene": scene,
            "component_tokens": tuple(component_tokens),
        }
    return templates


def _normalize_component_token(token: str, content_json: dict[str, Any] | None) -> str | None:
    name, payload = _split_token(token)
    if name == "header_band":
        ref = str(payload or "header").strip() or "header"
        if _payload_lookup(content_json, ref) is None:
            return None
        return f"[header_band:{ref}]"
    if name == "address_strip":
        ref = str(payload or "offsets").strip() or "offsets"
        if _payload_lookup(content_json, ref) is None:
            return None
        return f"[address_strip:{ref}]"
    if name == "memory_segment":
        parts = [part.strip() for part in payload.split("|") if part.strip()]
        if len(parts) < 2:
            return None
        ref = str(parts[-1]).strip()
        if not ref.startswith("segments.") or _payload_lookup(content_json, ref) is None:
            return None
        segment_id = ref.split(".")[-1]
        return f"[memory_segment:{segment_id}|{ref}]"
    if name == "region_bracket":
        parts = [part.strip() for part in payload.split("|") if part.strip()]
        if len(parts) < 2:
            return None
        ref = str(parts[-1]).strip()
        if not ref.startswith("segments.") or _payload_lookup(content_json, ref) is None:
            return None
        bracket_id = ref.split(".")[-1]
        return f"[region_bracket:{bracket_id}|{ref}]"
    if name == "info_card":
        parts = [part.strip() for part in payload.split("|") if part.strip()]
        if len(parts) < 2:
            return None
        ref = str(parts[-1]).strip()
        if not ref.startswith("cards.") or _payload_lookup(content_json, ref) is None:
            return None
        card_id = ref.split(".")[-1]
        return f"[info_card:{card_id}|{ref}]"
    return None


def _extract_scene_tokens(text: str) -> list[str]:
    out: list[str] = []
    for token in _BRACKET_TOKEN_RE.findall(str(text or "")):
        name, _ = _split_token(token)
        if name in _SCENE_TOKEN_NAMES or name in _PROMPT_ONLY_TOKEN_NAMES:
            out.append(token)
    return out


def _scene_matches_prompt_contract(
    scene: SceneDocument,
    *,
    prompt_bundle: dict[str, str],
    content_json: dict[str, Any] | None,
    template_tokens: list[str],
) -> bool:
    if str(prompt_bundle.get("layout") or "memory_map").strip() != str(scene.layout).strip():
        return False
    if prompt_bundle.get("theme") and str(prompt_bundle.get("theme")).strip() != str(scene.theme).strip():
        return False
    if prompt_bundle.get("tone") and str(prompt_bundle.get("tone")).strip() != str(scene.tone).strip():
        return False
    if prompt_bundle.get("density") and str(prompt_bundle.get("density")).strip() != str(scene.density).strip():
        return False
    background = str(prompt_bundle.get("background") or "").strip()
    if background and background != str(scene.background).strip():
        return False
    runtime = scene.to_runtime()
    component_counts = runtime.get("components_by_name", {})
    if prompt_bundle.get("segments") and len(component_counts.get("memory_segment", [])) != int(prompt_bundle["segments"]):
        return False
    if prompt_bundle.get("brackets") and len(component_counts.get("region_bracket", [])) != int(prompt_bundle["brackets"]):
        return False
    if prompt_bundle.get("cards") and len(component_counts.get("info_card", [])) != int(prompt_bundle["cards"]):
        return False
    for component in scene.components:
        ref = _component_ref(component.name, component.fields)
        if ref and _payload_lookup(content_json, ref) is None:
            return False
    actual_tokens = [_component_token_from_fields(component.name, component.fields) for component in scene.components]
    return actual_tokens == template_tokens


def repair_spec15a_scene_response(
    *,
    raw_text: str,
    parsed_output: str,
    prompt: str,
    content_json: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Repair near-miss spec15a outputs using prompt bundle + valid content refs.

    This is intentionally narrow. It only repairs outputs that already show
    strong evidence of the correct form by matching at least half of the gold
    component template after invalid refs and prompt-only control tags are
    removed.
    """

    prompt_bundle = _parse_prompt_bundle(prompt)
    form_token = str(prompt_bundle.get("form") or "").strip()
    template = _form_templates().get(form_token)
    if template is None:
        return None
    try:
        parsed_scene = canonicalize_scene_text(parsed_output)
    except Exception:
        parsed_scene = None
    if parsed_scene is not None and _scene_matches_prompt_contract(
        parsed_scene,
        prompt_bundle=prompt_bundle,
        content_json=content_json,
        template_tokens=list(template["component_tokens"]),
    ):
        return None

    extracted_tokens = _extract_scene_tokens(raw_text)
    normalized_components: list[str] = []
    seen_component_keys: set[tuple[str, str]] = set()
    for token in extracted_tokens:
        name, _ = _split_token(token)
        if name in {"header_band", "address_strip", "memory_segment", "region_bracket", "info_card"}:
            normalized = _normalize_component_token(token, content_json)
            if not normalized:
                continue
            comp_name, payload = _split_token(normalized)
            comp_key = (comp_name, payload.split("|")[-1])
            if comp_key in seen_component_keys:
                continue
            seen_component_keys.add(comp_key)
            normalized_components.append(normalized)

    template_tokens = list(template["component_tokens"])
    template_matches = [token for token in template_tokens if token in normalized_components]
    minimum_matches = max(3, len(template_tokens) // 2)
    if len(template_matches) < minimum_matches:
        return None

    repaired_components = list(template_tokens)
    filled_tokens = [token for token in repaired_components if token not in normalized_components]
    background_value = str(prompt_bundle.get("background") or template["scene"].background).strip() or template["scene"].background
    scene_tokens = [
        "[scene]",
        f"[layout:{str(prompt_bundle.get('layout') or 'memory_map').strip() or 'memory_map'}]",
        f"[theme:{str(prompt_bundle.get('theme') or template['scene'].theme).strip() or template['scene'].theme}]",
        f"[tone:{str(prompt_bundle.get('tone') or template['scene'].tone).strip() or template['scene'].tone}]",
        f"[density:{str(prompt_bundle.get('density') or template['scene'].density).strip() or template['scene'].density}]",
        f"[topic:{GENERIC_TOPIC}]",
        *repaired_components,
        "[/scene]",
    ]
    if background_value and background_value != "none":
        scene_tokens.insert(5, f"[background:{background_value}]")
    repaired_text = " ".join(scene_tokens)
    canonicalized = canonicalize_scene_text(repaired_text)
    canonical_tokens = [
        "[scene]",
        f"[layout:{canonicalized.layout}]",
        f"[theme:{canonicalized.theme}]",
        f"[tone:{canonicalized.tone}]",
        f"[density:{canonicalized.density}]",
        f"[topic:{canonicalized.topic}]",
        *[_component_token_from_fields(component.name, component.fields) for component in canonicalized.components],
        "[/scene]",
    ]
    if str(canonicalized.background).strip() and str(canonicalized.background).strip() != "none":
        canonical_tokens.insert(5, f"[background:{canonicalized.background}]")
    canonical_text = " ".join(canonical_tokens)
    return {
        "parsed_output": canonical_text,
        "repair_applied": True,
        "repairer": "spec15a_scene_bundle.v1",
        "repair_note": (
            f"repaired from {len(template_matches)}/{len(template_tokens)} matched template components; "
            f"filled {len(filled_tokens)} missing components from the {form_token} template"
        ),
        "repair_diag": {
            "form_token": form_token,
            "template_component_count": len(template_tokens),
            "matched_component_count": len(template_matches),
            "filled_component_count": len(filled_tokens),
            "extracted_component_count": len(normalized_components),
        },
    }
