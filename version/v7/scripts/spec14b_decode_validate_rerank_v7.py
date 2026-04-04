#!/usr/bin/env python3
"""Spec14b decode-time validation and scene repair helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from spec14b_scene_canonicalizer_v7 import GENERIC_TOPIC, SceneDocument, canonicalize_scene_text
except ModuleNotFoundError:  # pragma: no cover
    from version.v7.scripts.spec14b_scene_canonicalizer_v7 import GENERIC_TOPIC, SceneDocument, canonicalize_scene_text


ROOT = Path(__file__).resolve().parents[3]
GOLD = ROOT / "version" / "v7" / "reports" / "spec14b_gold_mappings"

_PROMPT_TAG_RE = re.compile(r"\[([a-z_]+):([^\]]+)\]")
_FORM_TEMPLATE_FILES = {
    "milestone_chain": GOLD / "ir-v66-evolution-timeline.scene.compact.dsl",
    "stage_sequence": GOLD / "ir-timeline-why.scene.compact.dsl",
}


def _parse_prompt_bundle(prompt: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in _PROMPT_TAG_RE.findall(str(prompt or "")):
        out[str(key).strip().lower()] = str(value).strip()
    return out


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


def _component_token_from_fields(name: str, fields: dict[str, str]) -> str:
    if name in {"header_band", "footer_note"}:
        payload = str(fields.get("ref") or "").strip()
    elif name == "timeline_stage":
        payload = "|".join(
            [
                str(fields.get("stage_id") or "").strip(),
                str(fields.get("ref") or "").strip(),
                f"lane={str(fields.get('lane') or 'center').strip()}",
            ]
        )
    elif name == "timeline_arrow":
        payload = f"{str(fields.get('from_stage') or '').strip()}->{str(fields.get('to_stage') or '').strip()}"
    else:
        raise ValueError(f"unsupported spec14b component: {name}")
    return f"[{name}:{payload}]"


def _scene_component_tokens(scene: SceneDocument) -> list[str]:
    return [_component_token_from_fields(component.name, component.fields) for component in scene.components]


@lru_cache(maxsize=8)
def _form_templates() -> dict[str, dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    for form_token, path in _FORM_TEMPLATE_FILES.items():
        scene = canonicalize_scene_text(path.read_text(encoding="utf-8"))
        runtime = scene.to_runtime()
        templates[form_token] = {
            "scene": scene,
            "component_tokens": tuple(_scene_component_tokens(scene)),
            "stage_count": len(runtime.get("components_by_name", {}).get("timeline_stage", [])),
            "arrow_count": len(runtime.get("components_by_name", {}).get("timeline_arrow", [])),
            "footer_count": len(runtime.get("components_by_name", {}).get("footer_note", [])),
            "base_style": (
                str(scene.theme).strip(),
                str(scene.tone).strip(),
                str(scene.density).strip(),
                str(scene.background).strip(),
            ),
        }
    return templates


def _prompt_matches_template_counts(prompt_bundle: dict[str, str], template: dict[str, Any]) -> bool:
    try:
        prompt_stages = int(str(prompt_bundle.get("stages") or "").strip())
        prompt_arrows = int(str(prompt_bundle.get("arrows") or "").strip())
        prompt_footer = int(str(prompt_bundle.get("footer") or "").strip())
    except ValueError:
        return False
    return (
        prompt_stages == int(template["stage_count"])
        and prompt_arrows == int(template["arrow_count"])
        and prompt_footer == int(template["footer_count"])
    )


def _style_tuple(bundle: dict[str, str], template: dict[str, Any]) -> tuple[str, str, str, str]:
    base_scene = template["scene"]
    return (
        str(bundle.get("theme") or base_scene.theme).strip() or base_scene.theme,
        str(bundle.get("tone") or base_scene.tone).strip() or base_scene.tone,
        str(bundle.get("density") or base_scene.density).strip() or base_scene.density,
        str(bundle.get("background") or base_scene.background).strip() or base_scene.background,
    )


def _include_frame(prompt_bundle: dict[str, str], template: dict[str, Any]) -> bool:
    scene = template["scene"]
    return scene.frame != "none" and _style_tuple(prompt_bundle, template) == tuple(template["base_style"])


def _emit_scene_text(prompt_bundle: dict[str, str], template: dict[str, Any]) -> str:
    scene = template["scene"]
    theme, tone, density, background = _style_tuple(prompt_bundle, template)
    tokens = [
        "[scene]",
        f"[layout:{str(prompt_bundle.get('layout') or scene.layout).strip() or scene.layout}]",
        f"[theme:{theme}]",
        f"[tone:{tone}]",
        f"[density:{density}]",
    ]
    if _include_frame(prompt_bundle, template):
        tokens.append(f"[frame:{scene.frame}]")
    tokens.append(f"[background:{background}]")
    tokens.append(f"[topic:{GENERIC_TOPIC}]")
    tokens.extend(template["component_tokens"])
    tokens.append("[/scene]")
    return " ".join(tokens)


def _parsed_scene_evidence(
    parsed_output: str,
    *,
    content_json: dict[str, Any] | None,
    template: dict[str, Any],
) -> tuple[SceneDocument | None, int]:
    try:
        parsed_scene = canonicalize_scene_text(parsed_output)
    except Exception:
        return None, 0
    component_tokens = _scene_component_tokens(parsed_scene)
    matches = sum(1 for token in template["component_tokens"] if token in component_tokens)
    for component in parsed_scene.components:
        ref = str(component.fields.get("ref") or "").strip()
        if ref and _payload_lookup(content_json, ref) is None:
            return parsed_scene, 0
    return parsed_scene, matches


def repair_spec14b_scene_response(
    *,
    raw_text: str,
    parsed_output: str,
    prompt: str,
    content_json: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Repair near-miss spec14b outputs into exact timeline scene bundles."""

    _ = raw_text
    prompt_bundle = _parse_prompt_bundle(prompt)
    form_token = str(prompt_bundle.get("form") or "").strip()
    template = _form_templates().get(form_token)
    if template is None:
        return None
    if str(prompt_bundle.get("layout") or "timeline").strip() != "timeline":
        return None
    if not _prompt_matches_template_counts(prompt_bundle, template):
        return None

    parsed_scene, matched_components = _parsed_scene_evidence(
        parsed_output,
        content_json=content_json,
        template=template,
    )
    minimum_matches = max(2, len(template["component_tokens"]) // 2)
    if parsed_scene is None or matched_components < minimum_matches:
        return None

    repaired_text = _emit_scene_text(prompt_bundle, template)
    return {
        "parsed_output": repaired_text,
        "repair_applied": True,
        "repairer": "spec14b_scene_bundle.v1",
        "repair_note": (
            f"repaired from {matched_components}/{len(template['component_tokens'])} matched template components; "
            "re-emitted the prompt-requested timeline style bundle and canonical component order"
        ),
        "repair_diag": {
            "form_token": form_token,
            "template_component_count": len(template["component_tokens"]),
            "matched_component_count": matched_components,
            "frame_included": _include_frame(prompt_bundle, template),
            "prompt_style": {
                "theme": _style_tuple(prompt_bundle, template)[0],
                "tone": _style_tuple(prompt_bundle, template)[1],
                "density": _style_tuple(prompt_bundle, template)[2],
                "background": _style_tuple(prompt_bundle, template)[3],
            },
        },
    }
