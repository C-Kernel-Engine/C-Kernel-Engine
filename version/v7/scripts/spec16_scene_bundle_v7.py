#!/usr/bin/env python3
"""Shared scene-bundle schema for the first generalized visual-DSL line."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SCHEMA_NAME = "ck.visual_scene_bundle.v1"

_FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "memory_map": {
        "forms": ("typed_regions", "arena_sections", "layer_stack"),
        "count_fields": ("segments", "brackets", "cards"),
    },
    "timeline": {
        "forms": ("milestone_chain", "stage_sequence"),
        "count_fields": ("stages", "arrows", "footer"),
    },
    "system_diagram": {
        "forms": ("linear_pipeline", "build_path", "selection_path"),
        "count_fields": ("stages", "links", "terminal", "footer"),
    },
}

_COMMON_FIELDS = ("family", "form", "theme", "tone", "density", "background")


def _clean_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"missing {field}")
    return text


def _clean_count(value: Any, *, field: str) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer for {field}: {value!r}") from exc
    if count < 0:
        raise ValueError(f"negative count for {field}: {count}")
    return count


@dataclass(frozen=True)
class SceneBundle:
    family: str
    form: str
    theme: str
    tone: str
    density: str
    background: str
    topology: dict[str, int]

    def family_spec(self) -> dict[str, Any]:
        return _FAMILY_SPECS[self.family]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_NAME,
            "family": self.family,
            "layout": self.family,
            "form": self.form,
            "theme": self.theme,
            "tone": self.tone,
            "density": self.density,
            "background": self.background,
            "topology": dict(self.topology),
        }

    def to_prompt_tags(self) -> str:
        tokens = [
            "[task:svg]",
            f"[layout:{self.family}]",
            f"[form:{self.form}]",
            f"[theme:{self.theme}]",
            f"[tone:{self.tone}]",
            f"[density:{self.density}]",
            f"[background:{self.background}]",
        ]
        for key in self.family_spec()["count_fields"]:
            tokens.append(f"[{key}:{self.topology[key]}]")
        tokens.append("[OUT]")
        return " ".join(tokens)


def canonicalize_scene_bundle(data: dict[str, Any]) -> SceneBundle:
    if not isinstance(data, dict):
        raise ValueError("scene bundle must be a JSON object")
    family = _clean_text(data.get("family") or data.get("layout"), field="family").lower()
    family_spec = _FAMILY_SPECS.get(family)
    if family_spec is None:
        raise ValueError(f"unsupported family: {family!r}")

    form = _clean_text(data.get("form"), field="form").lower()
    if form not in family_spec["forms"]:
        raise ValueError(f"unsupported form {form!r} for family {family!r}")

    theme = _clean_text(data.get("theme"), field="theme")
    tone = _clean_text(data.get("tone"), field="tone")
    density = _clean_text(data.get("density"), field="density")
    background = _clean_text(data.get("background"), field="background")

    topology_doc = data.get("topology")
    if topology_doc is None:
        topology_doc = {key: data.get(key) for key in family_spec["count_fields"]}
    if not isinstance(topology_doc, dict):
        raise ValueError("topology must be a JSON object")

    topology: dict[str, int] = {}
    for key in family_spec["count_fields"]:
        topology[key] = _clean_count(topology_doc.get(key), field=key)
    extra_keys = sorted(str(key) for key in topology_doc.keys() if str(key) not in family_spec["count_fields"])
    if extra_keys:
        raise ValueError(f"unexpected topology keys for {family}: {', '.join(extra_keys)}")

    return SceneBundle(
        family=family,
        form=form,
        theme=theme,
        tone=tone,
        density=density,
        background=background,
        topology=topology,
    )


def family_specs() -> dict[str, dict[str, Any]]:
    return {name: {"forms": tuple(spec["forms"]), "count_fields": tuple(spec["count_fields"])} for name, spec in _FAMILY_SPECS.items()}


__all__ = [
    "SCHEMA_NAME",
    "SceneBundle",
    "canonicalize_scene_bundle",
    "family_specs",
]
