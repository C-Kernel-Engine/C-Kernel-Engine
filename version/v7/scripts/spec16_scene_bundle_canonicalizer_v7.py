#!/usr/bin/env python3
"""Canonicalize compact spec16 scene-bundle text."""

from __future__ import annotations

import re
from typing import Any

try:
    from scene_dsl_pre_repair_v7 import repair_compact_scene_text
    from spec16_scene_bundle_v7 import SceneBundle, canonicalize_scene_bundle, family_specs
except ModuleNotFoundError:  # pragma: no cover
    from version.v7.scripts.scene_dsl_pre_repair_v7 import repair_compact_scene_text
    from version.v7.scripts.spec16_scene_bundle_v7 import SceneBundle, canonicalize_scene_bundle, family_specs


_BRACKET_TOKEN_RE = re.compile(r"\[[^\[\]\n]+\]")
_PROMPT_TAG_RE = re.compile(r"\[([a-z_]+):([^\]]+)\]")
_PROMPT_ONLY_NAMES = {"task", "out"}
_ALL_COUNT_FIELDS = {
    str(field)
    for spec in family_specs().values()
    for field in tuple(spec.get("count_fields") or ())
}
_SINGLETON_NAMES = {
    "family",
    "layout",
    "form",
    "theme",
    "tone",
    "density",
    "background",
    *_ALL_COUNT_FIELDS,
}


def _extract_tokens(text: str) -> list[str]:
    return [token.strip() for token in _BRACKET_TOKEN_RE.findall(str(text or "")) if token.strip()]


def _split_token(token: str) -> tuple[str, str]:
    body = str(token or "").strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    if ":" not in body:
        return body.strip().lower(), ""
    name, payload = body.split(":", 1)
    return name.strip().lower(), payload.strip()


def _coerce_bundle_markers(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    tokens = _extract_tokens(raw)
    if tokens and "[bundle]" not in raw and "[/bundle]" not in raw:
        return "[bundle] " + " ".join(tokens) + " [/bundle]"
    return raw.replace("[bundle]", "[scene]").replace("[/bundle]", "[/scene]")


def _prompt_bundle_fields(prompt: str) -> dict[str, Any]:
    prompt_fields: dict[str, Any] = {}
    topology: dict[str, Any] = {}
    for key, value in _PROMPT_TAG_RE.findall(str(prompt or "")):
        name = str(key or "").strip().lower()
        payload = str(value or "").strip()
        if name == "layout":
            prompt_fields["family"] = payload
            continue
        if name in {"family", "form", "theme", "tone", "density", "background"}:
            prompt_fields[name] = payload
            continue
        if name in _ALL_COUNT_FIELDS:
            topology[name] = payload
    if topology:
        prompt_fields["topology"] = topology
    return prompt_fields


def repair_scene_bundle_text(text: Any) -> dict[str, Any]:
    """Repair syntax-only issues in compact bundle text."""

    repaired = repair_compact_scene_text(
        _coerce_bundle_markers(text),
        singleton_names=_SINGLETON_NAMES,
        prompt_only_names=_PROMPT_ONLY_NAMES,
    )
    repaired_text = str(repaired.get("repaired_text") or "")
    if repaired_text:
        repaired_text = repaired_text.replace("[scene]", "[bundle]").replace("[/scene]", "[/bundle]")
    return {
        "repaired_text": repaired_text,
        "changed": bool(repaired.get("changed")),
        "diag": repaired.get("diag") if isinstance(repaired.get("diag"), dict) else {},
    }


def canonicalize_scene_bundle_text(text: Any) -> SceneBundle:
    repaired = repair_scene_bundle_text(text)
    repaired_text = str(repaired.get("repaired_text") or "").strip()
    tokens = _extract_tokens(repaired_text)
    if not tokens or tokens[0] != "[bundle]":
        raise ValueError("spec16 bundle must start with [bundle]")
    if tokens[-1] != "[/bundle]":
        raise ValueError("spec16 bundle must end with [/bundle]")

    fields: dict[str, Any] = {}
    topology: dict[str, Any] = {}
    for token in tokens[1:-1]:
        name, payload = _split_token(token)
        if not name:
            continue
        if name in _PROMPT_ONLY_NAMES:
            continue
        if name in {"family", "layout", "form", "theme", "tone", "density", "background"}:
            fields[name] = payload
            continue
        if name in _ALL_COUNT_FIELDS:
            topology[name] = payload
            continue
        raise ValueError(f"unexpected bundle token: {token}")

    if topology:
        fields["topology"] = topology
    return canonicalize_scene_bundle(fields)


def _repair_bundle_from_prompt(candidate: str, prompt: str) -> str | None:
    prompt_fields = _prompt_bundle_fields(prompt)
    prompt_family = str(prompt_fields.get("family") or "").strip()
    prompt_form = str(prompt_fields.get("form") or "").strip()
    if not prompt_family or not prompt_form:
        return None

    repaired = repair_scene_bundle_text(candidate)
    tokens = _extract_tokens(str(repaired.get("repaired_text") or "").strip())
    if not tokens or tokens[0] != "[bundle]":
        return None

    merged: dict[str, Any] = {}
    topology: dict[str, Any] = {}
    saw_any_bundle_field = False
    for token in tokens[1:-1]:
        name, payload = _split_token(token)
        if not name or name in _PROMPT_ONLY_NAMES:
            continue
        if name == "layout":
            name = "family"
        if name in {"family", "form", "theme", "tone", "density", "background"}:
            saw_any_bundle_field = True
            if name in merged:
                continue
            merged[name] = payload
            continue
        if name in _ALL_COUNT_FIELDS:
            saw_any_bundle_field = True
            if name in topology:
                continue
            topology[name] = payload
            continue
    if not saw_any_bundle_field:
        return None

    family_value = str(merged.get("family") or prompt_family).strip()
    form_value = str(merged.get("form") or prompt_form).strip()
    if family_value and family_value != prompt_family:
        return None
    if form_value and form_value != prompt_form:
        return None

    merged["family"] = prompt_family
    merged["form"] = prompt_form
    for key in ("theme", "tone", "density", "background"):
        if key not in merged and prompt_fields.get(key):
            merged[key] = prompt_fields.get(key)

    prompt_topology = prompt_fields.get("topology")
    if isinstance(prompt_topology, dict):
        for key, value in prompt_topology.items():
            topology.setdefault(str(key), value)
    if topology:
        merged["topology"] = topology

    try:
        bundle = canonicalize_scene_bundle(merged)
    except Exception:
        return None
    return serialize_scene_bundle(bundle)


def serialize_scene_bundle(bundle_doc: SceneBundle | dict[str, Any]) -> str:
    bundle = bundle_doc if isinstance(bundle_doc, SceneBundle) else canonicalize_scene_bundle(bundle_doc)
    tokens = [
        "[bundle]",
        f"[family:{bundle.family}]",
        f"[form:{bundle.form}]",
        f"[theme:{bundle.theme}]",
        f"[tone:{bundle.tone}]",
        f"[density:{bundle.density}]",
        f"[background:{bundle.background}]",
    ]
    for key in bundle.family_spec()["count_fields"]:
        tokens.append(f"[{key}:{bundle.topology[key]}]")
    tokens.append("[/bundle]")
    return " ".join(tokens)


def repair_spec16_bundle_response(
    *,
    raw_text: str,
    parsed_output: str,
    prompt: str,
    content_json: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Canonicalize recoverable spec16 bundle syntax without inventing semantics."""

    _ = prompt
    _ = content_json
    for source_name, candidate in (("parsed_output", parsed_output), ("raw_text", raw_text)):
        try:
            repair = repair_scene_bundle_text(candidate)
            bundle = canonicalize_scene_bundle_text(candidate)
        except Exception:
            fallback_text = _repair_bundle_from_prompt(candidate, prompt)
            if not fallback_text:
                continue
            return {
                "parsed_output": fallback_text,
                "repair_applied": True,
                "repair_note": "filled missing bundle singleton fields from prompt controls",
                "repair_diag": {
                    "source": source_name,
                    "syntax_diag": repair_scene_bundle_text(candidate).get("diag"),
                },
            }
        canonical_text = serialize_scene_bundle(bundle)
        original_norm = " ".join(str(candidate or "").strip().split())
        changed = canonical_text != original_norm or bool(repair.get("changed"))
        return {
            "parsed_output": canonical_text,
            "repair_applied": changed,
            "repair_note": "canonicalized spec16 bundle syntax" if changed else None,
            "repair_diag": {
                "source": source_name,
                "bundle": bundle.to_dict(),
                "syntax_diag": repair.get("diag") if isinstance(repair.get("diag"), dict) else {},
            },
        }
    return None


__all__ = [
    "canonicalize_scene_bundle_text",
    "repair_scene_bundle_text",
    "repair_spec16_bundle_response",
    "serialize_scene_bundle",
]
