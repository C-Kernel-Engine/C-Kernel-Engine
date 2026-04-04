#!/usr/bin/env python3
"""Spec15b decode-time validation and near-miss scene repair helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from spec15b_scene_canonicalizer_v7 import GENERIC_TOPIC, SceneDocument, canonicalize_scene_text
except ModuleNotFoundError:  # pragma: no cover
    from version.v7.scripts.spec15b_scene_canonicalizer_v7 import GENERIC_TOPIC, SceneDocument, canonicalize_scene_text


ROOT = Path(__file__).resolve().parents[3]
GOLD = ROOT / "version" / "v7" / "reports" / "spec15b_gold_mappings"

_PROMPT_TAG_RE = re.compile(r"\[([a-z_]+):([^\]]+)\]")
_BRACKET_TOKEN_RE = re.compile(r"\[[^\[\]\n]+\]")
_FORM_TEMPLATE_FILES = {
    "linear_pipeline": GOLD / "pipeline-overview-system.scene.compact.dsl",
    "build_path": GOLD / "ir-pipeline-flow-system.scene.compact.dsl",
    "selection_path": GOLD / "kernel-registry-flow-system.scene.compact.dsl",
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


def _split_token(token: str) -> tuple[str, str]:
    body = str(token or "").strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    if ":" not in body:
        return body.strip().lower(), ""
    name, payload = body.split(":", 1)
    return name.strip().lower(), payload.strip()


def _component_token_from_fields(name: str, fields: dict[str, str]) -> str:
    if name in {"header_band", "footer_note"}:
        payload = str(fields.get("ref") or "").strip()
    elif name == "system_stage":
        payload = "|".join(
            [
                str(fields.get("stage_id") or "").strip(),
                str(fields.get("ref") or "").strip(),
            ]
        )
    elif name == "system_link":
        payload = "|".join(
            [
                f"{str(fields.get('from_stage') or '').strip()}->{str(fields.get('to_stage') or '').strip()}",
                str(fields.get("ref") or "").strip(),
            ]
        )
    elif name == "terminal_panel":
        payload = "|".join(
            [
                str(fields.get("panel_id") or "").strip(),
                str(fields.get("ref") or "").strip(),
            ]
        )
    else:
        raise ValueError(f"unsupported spec15b component: {name}")
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
            "stage_count": len(runtime.get("components_by_name", {}).get("system_stage", [])),
            "link_count": len(runtime.get("components_by_name", {}).get("system_link", [])),
            "terminal_count": len(runtime.get("components_by_name", {}).get("terminal_panel", [])),
            "footer_count": len(runtime.get("components_by_name", {}).get("footer_note", [])),
        }
    return templates


def _prompt_matches_template_counts(prompt_bundle: dict[str, str], template: dict[str, Any]) -> bool:
    try:
        prompt_stages = int(str(prompt_bundle.get("stages") or "").strip())
        prompt_links = int(str(prompt_bundle.get("links") or "").strip())
        prompt_terminal = int(str(prompt_bundle.get("terminal") or "").strip())
        prompt_footer = int(str(prompt_bundle.get("footer") or "").strip())
    except ValueError:
        return False
    return (
        prompt_stages == int(template["stage_count"])
        and prompt_links == int(template["link_count"])
        and prompt_terminal == int(template["terminal_count"])
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


def _emit_scene_text(prompt_bundle: dict[str, str], template: dict[str, Any]) -> str:
    scene = template["scene"]
    theme, tone, density, background = _style_tuple(prompt_bundle, template)
    tokens = [
        "[scene]",
        f"[canvas:{scene.canvas}]",
        f"[layout:{str(prompt_bundle.get('layout') or scene.layout).strip() or scene.layout}]",
        f"[theme:{theme}]",
        f"[tone:{tone}]",
        f"[frame:{scene.frame}]",
        f"[density:{density}]",
        f"[inset:{scene.inset}]",
        f"[gap:{scene.gap}]",
        f"[background:{background}]",
        f"[connector:{scene.connector}]",
        f"[topic:{GENERIC_TOPIC}]",
    ]
    tokens.extend(template["component_tokens"])
    tokens.append("[/scene]")
    return " ".join(tokens)


def _extract_scene_tokens(text: str) -> list[str]:
    return [tok.strip() for tok in _BRACKET_TOKEN_RE.findall(str(text or "")) if tok.strip()]


def _normalize_component_token(token: str, content_json: dict[str, Any] | None) -> str | None:
    name, payload = _split_token(token)
    parts = [part.strip() for part in payload.split("|") if part.strip()]
    if name in {"header_band", "footer_note"}:
        ref = str(parts[0] if parts else "").strip()
        if not ref or _payload_lookup(content_json, ref) is None:
            return None
        return f"[{name}:{ref}]"
    if name == "system_stage":
        if len(parts) < 2:
            return None
        ref = str(parts[-1]).strip()
        if not ref.startswith("stages.") or _payload_lookup(content_json, ref) is None:
            return None
        stage_id = ref.split(".")[-1]
        return f"[system_stage:{stage_id}|{ref}]"
    if name == "system_link":
        if len(parts) < 2:
            return None
        ref = str(parts[-1]).strip()
        if not ref.startswith("links.") or _payload_lookup(content_json, ref) is None:
            return None
        link_key = ref.split(".")[-1]
        if "_" not in link_key:
            return None
        raw_pair = link_key.replace("_terminal", "->terminal")
        if "->" not in raw_pair:
            first, second = raw_pair.split("_", 1)
            raw_pair = f"{first}->{second}"
        else:
            raw_pair = raw_pair.replace("_", "->", 1)
        if "->" not in raw_pair:
            return None
        if raw_pair.count("->") > 1:
            src, dst = raw_pair.split("->", 1)
            dst = dst.replace("_", "->")
            raw_pair = f"{src}->{dst}"
        return f"[system_link:{raw_pair}|{ref}]"
    if name == "terminal_panel":
        if len(parts) < 2:
            return None
        ref = str(parts[-1]).strip()
        if str(ref) != "terminal" or _payload_lookup(content_json, ref) is None:
            return None
        return "[terminal_panel:terminal|terminal]"
    return None


def _raw_scene_evidence(raw_text: str, *, content_json: dict[str, Any] | None, template: dict[str, Any]) -> int:
    normalized_components: list[str] = []
    seen_component_keys: set[tuple[str, str]] = set()
    for token in _extract_scene_tokens(raw_text):
        normalized = _normalize_component_token(token, content_json)
        if not normalized:
            continue
        comp_name, payload = _split_token(normalized)
        comp_key = (comp_name, payload.split("|")[-1])
        if comp_key in seen_component_keys:
            continue
        seen_component_keys.add(comp_key)
        normalized_components.append(normalized)
    return sum(1 for token in template["component_tokens"] if token in normalized_components)


def repair_spec15b_scene_response(
    *,
    raw_text: str,
    parsed_output: str,
    prompt: str,
    content_json: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Repair near-miss spec15b outputs into exact system_diagram scene bundles."""

    prompt_bundle = _parse_prompt_bundle(prompt)
    form_token = str(prompt_bundle.get("form") or "").strip()
    template = _form_templates().get(form_token)
    if template is None:
        return None
    if str(prompt_bundle.get("layout") or "system_diagram").strip() != "system_diagram":
        return None
    if not _prompt_matches_template_counts(prompt_bundle, template):
        return None

    try:
        parsed_scene = canonicalize_scene_text(parsed_output)
    except Exception:
        parsed_scene = None
    if parsed_scene is not None:
        if _scene_component_tokens(parsed_scene) == list(template["component_tokens"]):
            return None

    matched_components = _raw_scene_evidence(raw_text, content_json=content_json, template=template)
    minimum_matches = max(3, len(template["component_tokens"]) // 2)
    if matched_components < minimum_matches:
        return None

    repaired_text = _emit_scene_text(prompt_bundle, template)
    return {
        "parsed_output": repaired_text,
        "repair_applied": True,
        "repairer": "spec15b_scene_bundle.v1",
        "repair_note": (
            f"repaired from {matched_components}/{len(template['component_tokens'])} matched template components; "
            "re-emitted the prompt-requested system_diagram style bundle and canonical component order"
        ),
        "repair_diag": {
            "form_token": form_token,
            "template_component_count": len(template["component_tokens"]),
            "matched_component_count": matched_components,
            "prompt_style": {
                "theme": _style_tuple(prompt_bundle, template)[0],
                "tone": _style_tuple(prompt_bundle, template)[1],
                "density": _style_tuple(prompt_bundle, template)[2],
                "background": _style_tuple(prompt_bundle, template)[3],
            },
        },
    }

