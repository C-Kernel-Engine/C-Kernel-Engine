#!/usr/bin/env python3
"""Render compositional keyed scene DSL by lowering into the spec09 renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec09_v7 import (
    LAYOUTS,
    _ATTRS,
    _parse_scene_document as _parse_scene_document_spec09,
    render_structured_scene_spec09_svg,
)


PASS_THROUGH_COMPONENTS = {"flow_arrow", "curved_connector", "divider"}
BLOCK_COMPONENTS = {
    "annotation",
    "axis",
    "badge_pill",
    "callout_card",
    "compare_bar",
    "compare_panel",
    "conclusion_strip",
    "floor_band",
    "footer_note",
    "header_band",
    "legend_row",
    "phase_divider",
    "section_card",
    "span_bracket",
    "stage_card",
    "table_header",
    "table_row",
    "thesis_box",
}
COMPONENT_FIELD_ORDER: dict[str, tuple[str, ...]] = {
    "annotation": ("label", "note"),
    "axis": ("label", "note"),
    "badge_pill": ("text",),
    "callout_card": ("title", "note"),
    "compare_bar": ("label", "value", "caption", "note"),
    "compare_panel": ("title", "value", "caption"),
    "conclusion_strip": ("text",),
    "floor_band": ("text",),
    "footer_note": ("text",),
    "header_band": ("kicker", "headline", "subtitle"),
    "legend_row": ("amber", "green", "blue", "purple", "mixed", "item_1", "item_2", "item_3", "item_4"),
    "phase_divider": ("left", "right"),
    "section_card": ("title", "value", "caption"),
    "span_bracket": ("label", "value"),
    "stage_card": ("title", "caption"),
    "table_header": ("column_1", "column_2", "column_3", "column_4"),
    "table_row": ("column_1", "column_2", "column_3", "note"),
    "thesis_box": ("headline", "support_1", "support_2", "support_3"),
}
COMPONENT_META_ORDER: dict[str, tuple[str, ...]] = {
    "callout_card": ("accent",),
    "compare_bar": ("accent",),
    "compare_panel": ("variant", "accent"),
    "section_card": ("variant", "accent"),
    "table_row": ("state", "accent"),
}


def _field_name(token: str) -> str | None:
    if token.startswith("[field:") and token.endswith("]"):
        return token[len("[field:") : -1].strip()
    return None


def _component_open(token: str) -> str | None:
    if not (token.startswith("[") and token.endswith("]")):
        return None
    body = token[1:-1].strip()
    if not body or ":" in body or body.startswith("/"):
        return None
    return body


def _component_close(token: str) -> str | None:
    if token.startswith("[/") and token.endswith("]"):
        return token[2:-1].strip()
    return None


def _payload_for_component(name: str, fields: list[tuple[str, str]], meta: dict[str, str]) -> str:
    ordered_field_names = COMPONENT_FIELD_ORDER.get(name, ())
    payload: list[str] = []
    seen_fields: set[str] = set()
    if name == "legend_row":
        for field_name, value in fields:
            payload.append(f"{field_name}={value}")
            seen_fields.add(field_name)
    else:
        for field_name in ordered_field_names:
            for current_name, value in fields:
                if current_name == field_name and current_name not in seen_fields:
                    if name == "compare_bar" and field_name == "note":
                        break
                    payload.append(value)
                    seen_fields.add(current_name)
                    break
        for current_name, value in fields:
            if current_name in seen_fields:
                continue
            if name == "compare_bar" and current_name == "note":
                continue
            payload.append(value)
            seen_fields.add(current_name)

    if name == "compare_bar":
        note_value = next((value for field_name, value in fields if field_name == "note"), None)
        if note_value:
            meta = dict(meta)
            meta["note"] = note_value

    for key in COMPONENT_META_ORDER.get(name, ()):
        value = meta.get(key)
        if value:
            payload.append(f"{key}={value}")
    for key, value in meta.items():
        if key in COMPONENT_META_ORDER.get(name, ()):
            continue
        payload.append(f"{key}={value}")
    return "|".join(payload)


def lower_structured_scene_spec11(text: str) -> str:
    tokens = [tok.strip() for tok in str(text or "").split() if tok.strip()]
    if "[scene]" in tokens:
        tokens = tokens[tokens.index("[scene]") :]
    if not tokens or tokens[0] != "[scene]":
        raise ValueError("spec11 scene DSL must start with [scene]")
    if "[/scene]" not in tokens:
        raise ValueError("spec11 scene DSL must end with [/scene]")
    end = tokens.index("[/scene]")
    scene_tokens = tokens[: end + 1]

    lowered: list[str] = ["[scene]"]
    i = 1
    while i < len(scene_tokens) - 1:
        token = scene_tokens[i]
        handled = False
        for key in _ATTRS:
            if token.startswith(f"[{key}:") and token.endswith("]"):
                lowered.append(token)
                handled = True
                break
        if handled:
            i += 1
            continue
        if token.startswith("[topic:") and token.endswith("]"):
            lowered.append(token)
            i += 1
            continue
        if (
            token.startswith("[")
            and token.endswith("]")
            and ":" in token
            and _component_open(token) is None
        ):
            name = token[1:-1].split(":", 1)[0].strip()
            if name in PASS_THROUGH_COMPONENTS:
                lowered.append(token)
                i += 1
                continue
        comp_name = _component_open(token)
        if comp_name:
            if comp_name not in BLOCK_COMPONENTS:
                raise ValueError(f"unsupported spec11 component: {comp_name}")
            i += 1
            fields: list[tuple[str, str]] = []
            meta: dict[str, str] = {}
            while i < len(scene_tokens) - 1:
                current = scene_tokens[i]
                close_name = _component_close(current)
                if close_name:
                    if close_name != comp_name:
                        raise ValueError(f"mismatched component close: expected [/{comp_name}] got [/{close_name}]")
                    break
                field_name = _field_name(current)
                if field_name is not None:
                    if i + 1 >= len(scene_tokens) - 1:
                        raise ValueError(f"missing field payload after {current}")
                    value = scene_tokens[i + 1]
                    fields.append((field_name, value))
                    i += 2
                    continue
                if current.startswith("[") and current.endswith("]") and ":" in current:
                    key, value = current[1:-1].split(":", 1)
                    meta[key.strip()] = value.strip()
                    i += 1
                    continue
                raise ValueError(f"unsupported token inside {comp_name}: {current}")
            if i >= len(scene_tokens) - 1 or _component_close(scene_tokens[i]) != comp_name:
                raise ValueError(f"component {comp_name} missing closing token")
            lowered.append(f"[{comp_name}:{_payload_for_component(comp_name, fields, meta)}]")
            i += 1
            continue
        raise ValueError(f"unsupported token in spec11 scene: {token}")
    lowered.append("[/scene]")
    return " ".join(lowered)


def _parse_scene_document(text: str) -> dict[str, Any]:
    lowered = lower_structured_scene_spec11(text)
    return _parse_scene_document_spec09(lowered)


def render_structured_scene_spec11_svg(text: str, content: dict[str, Any] | None = None) -> str:
    lowered = lower_structured_scene_spec11(text)
    return render_structured_scene_spec09_svg(lowered, content=content)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render compositional spec11 scene DSL into SVG")
    ap.add_argument("--scene", required=True, type=Path, help="Input scene DSL text file")
    ap.add_argument("--content-json", default=None, type=Path, help="Optional content JSON file")
    ap.add_argument("--output", required=True, type=Path, help="Output SVG path")
    args = ap.parse_args()

    scene_text = args.scene.read_text(encoding="utf-8")
    content = None
    if args.content_json:
        content = json.loads(args.content_json.read_text(encoding="utf-8"))
    svg = render_structured_scene_spec11_svg(scene_text, content=content)
    args.output.write_text(svg, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
