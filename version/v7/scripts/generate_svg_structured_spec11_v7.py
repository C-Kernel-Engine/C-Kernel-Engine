#!/usr/bin/env python3
"""
Generate an asset-grounded keyed scene-DSL dataset for spec11.

Spec11 separates structure from content:
- the model target is the scene DSL only
- visible text is carried in external content_json
- the compiler blends scene DSL + content_json into final SVG
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from build_spec09_asset_alignment_report_v7 import GOLD_CASES
from generate_svg_structured_spec06_v7 import (
    _build_tokenizer_payload,
    _write_lines,
    _write_tokenizer_artifacts,
)
from render_svg_structured_scene_spec09_v7 import (
    _resolve_text_value,
)
from render_svg_structured_scene_spec11_v7 import render_structured_scene_spec11_svg


LAYOUTS = (
    "poster_stack",
    "comparison_span_chart",
    "pipeline_lane",
    "dual_panel_compare",
    "dashboard_cards",
)
THEMES = ("infra_dark", "paper_editorial", "signal_glow")
TONES = ("amber", "green", "blue", "purple", "mixed")
DENSITIES = ("compact", "balanced", "airy")

SCENE_ATTR_KEYS = {
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
}

PASS_THROUGH_COMPONENTS = {"flow_arrow", "curved_connector", "divider"}
COMPONENT_META_KEYS: dict[str, set[str]] = {
    "section_card": {"variant", "accent"},
    "compare_bar": {"accent", "note"},
    "table_row": {"state", "accent"},
    "compare_panel": {"variant", "accent"},
    "callout_card": {"accent"},
    "annotation": {"accent"},
}
COMPONENT_CONTENT_FIELDS: dict[str, tuple[str, ...]] = {
    "header_band": ("kicker", "headline", "subtitle"),
    "section_card": ("title", "value", "caption"),
    "compare_bar": ("label", "value", "caption"),
    "table_header": ("column_1", "column_2", "column_3", "column_4"),
    "table_row": ("column_1", "column_2", "column_3", "note"),
    "compare_panel": ("title", "value", "caption"),
    "callout_card": ("title", "note"),
    "footer_note": ("text",),
    "axis": ("label", "note"),
    "annotation": ("label", "note"),
    "span_bracket": ("label", "value"),
    "floor_band": ("text",),
    "thesis_box": ("headline", "support_1", "support_2", "support_3"),
    "conclusion_strip": ("text",),
    "phase_divider": ("left", "right"),
    "stage_card": ("title", "caption"),
    "badge_pill": ("text",),
}
META_CONTENT_KEYS: dict[str, set[str]] = {"compare_bar": {"note"}}


def _display_text(raw: str) -> str:
    return str(_resolve_text_value(raw, None))


def _parse_case_scene(case_dsl: str) -> dict[str, Any]:
    attrs: dict[str, str] = {}
    components: list[str] = []
    for raw in str(case_dsl or "").splitlines():
        token = raw.strip()
        if not token or token in {"[scene]", "[/scene]"}:
            continue
        if token.startswith("[") and token.endswith("]") and ":" in token:
            key, value = token[1:-1].split(":", 1)
            if key in SCENE_ATTR_KEYS:
                attrs[key] = value
                continue
        components.append(token)
    return {"attrs": attrs, "components": components}


def _split_component(token: str) -> tuple[str, str]:
    text = str(token or "").strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"invalid component token: {token}")
    body = text[1:-1]
    if ":" not in body:
        raise ValueError(f"invalid component payload: {token}")
    name, payload = body.split(":", 1)
    return name.strip(), payload.strip()


def _split_payload(name: str, payload: str) -> tuple[list[str], dict[str, str]]:
    meta_keys = COMPONENT_META_KEYS.get(name, set())
    parts: list[str] = []
    meta: dict[str, str] = {}
    for item in str(payload or "").split("|"):
        chunk = item.strip()
        if not chunk:
            continue
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in meta_keys:
                meta[key] = value
                continue
        parts.append(chunk)
    return parts, meta


def _legend_entry(idx: int, payload: str, content: dict[str, Any]) -> list[str]:
    entries = content.setdefault("legend_row", [])
    entry: dict[str, str] = {}
    out_tokens: list[str] = ["[legend_row]"]
    for chunk in str(payload or "").split("|"):
        item = chunk.strip()
        if not item:
            continue
        if "=" in item:
            tone, label = item.split("=", 1)
            tone = tone.strip().lower()
            field = tone if tone and tone not in entry else f"item_{len(entry) + 1}"
            entry[field] = _display_text(label)
            out_tokens.extend([f"[field:{field}]", f"@legend_row.{idx}.{field}"])
        else:
            field = f"item_{len(entry) + 1}"
            entry[field] = _display_text(item)
            out_tokens.extend([f"[field:{field}]", f"@legend_row.{idx}.{field}"])
    entries.append(entry)
    out_tokens.append("[/legend_row]")
    return out_tokens


def _rewrite_component(token: str, counters: dict[str, int], content: dict[str, Any]) -> list[str]:
    name, payload = _split_component(token)
    idx = int(counters.get(name, 0))
    counters[name] = idx + 1

    if name in PASS_THROUGH_COMPONENTS:
        return [token]
    if name == "legend_row":
        return _legend_entry(idx, payload, content)

    fields = COMPONENT_CONTENT_FIELDS.get(name)
    if not fields:
        return [token]

    parts, meta = _split_payload(name, payload)
    entry: dict[str, str] = {}
    out_tokens: list[str] = [f"[{name}]"]
    for part_idx, part in enumerate(parts):
        field = fields[part_idx] if part_idx < len(fields) else f"field_{part_idx + 1}"
        entry[field] = _display_text(part)
        out_tokens.extend([f"[field:{field}]", f"@{name}.{idx}.{field}"])

    for key, value in meta.items():
        if key in META_CONTENT_KEYS.get(name, set()):
            entry[key] = _display_text(value)
            out_tokens.extend([f"[field:{key}]", f"@{name}.{idx}.{key}"])
        else:
            out_tokens.append(f"[{key}:{value}]")

    content.setdefault(name, []).append(entry)
    out_tokens.append(f"[/{name}]")
    return out_tokens


def _rewrite_case(case_dsl: str) -> dict[str, Any]:
    parsed = _parse_case_scene(case_dsl)
    content: dict[str, Any] = {}
    counters: dict[str, int] = defaultdict(int)
    rewritten_components: list[str] = []
    for token in parsed["components"]:
        rewritten_components.extend(_rewrite_component(token, counters, content))
    return {
        "attrs": dict(parsed["attrs"]),
        "components": rewritten_components,
        "content_json": content,
    }


CASE_BY_TOPIC: dict[str, dict[str, Any]] = {}
for _case in GOLD_CASES:
    rewritten = _rewrite_case(str(_case.get("dsl") or ""))
    topic = str(rewritten["attrs"].get("topic") or "")
    if topic:
        CASE_BY_TOPIC[topic] = {
            "asset": str(_case.get("asset") or ""),
            "title": str(_case.get("title") or ""),
            "attrs": dict(rewritten["attrs"]),
            "components": list(rewritten["components"]),
            "content_json": dict(rewritten["content_json"]),
        }

TOPICS = tuple(sorted(CASE_BY_TOPIC))


@dataclass(frozen=True)
class Scene:
    layout: str
    topic: str
    theme: str = "infra_dark"
    tone: str = "blue"
    density: str = "compact"

    @property
    def prompt_tokens(self) -> list[str]:
        return [
            "[task:svg]",
            f"[layout:{self.layout}]",
            f"[topic:{self.topic}]",
            f"[theme:{self.theme}]",
            f"[tone:{self.tone}]",
            f"[density:{self.density}]",
            "[OUT]",
        ]

    @property
    def prompt_text(self) -> str:
        return " ".join(self.prompt_tokens)


def _scene_sort_key(scene: Scene) -> tuple[str, str, str, str, str]:
    return (scene.layout, scene.topic, scene.theme, scene.tone, scene.density)


def _is_holdout(scene: Scene) -> bool:
    score = (
        LAYOUTS.index(scene.layout)
        + TOPICS.index(scene.topic)
        + THEMES.index(scene.theme)
        + TONES.index(scene.tone)
        + DENSITIES.index(scene.density)
    )
    return (score % 6) == 0


def _style_density(scene: Scene) -> str:
    return scene.density if scene.density in DENSITIES else "balanced"


def _frame(scene: Scene, base_attrs: dict[str, str]) -> str:
    frame = str(base_attrs.get("frame") or "").strip()
    if frame:
        return frame
    if scene.layout in {"pipeline_lane", "dual_panel_compare", "dashboard_cards"}:
        return "panel"
    if scene.layout in {"poster_stack", "comparison_span_chart"}:
        return "card"
    return "none"


def _inset(scene: Scene, base_attrs: dict[str, str]) -> str:
    default = str(base_attrs.get("inset") or "md")
    return {
        "compact": default if default in {"sm", "md"} else "md",
        "balanced": "md",
        "airy": "lg",
    }.get(_style_density(scene), default)


def _gap(scene: Scene, base_attrs: dict[str, str]) -> str:
    default = str(base_attrs.get("gap") or "md")
    return {
        "compact": "sm",
        "balanced": default if default in {"sm", "md"} else "md",
        "airy": "lg",
    }.get(_style_density(scene), default)


def _rail(scene: Scene, base_attrs: dict[str, str]) -> str:
    if scene.layout == "poster_stack":
        return "muted" if scene.tone == "mixed" else "accent"
    return str(base_attrs.get("rail") or "none")


def _background(scene: Scene) -> str:
    return {
        "paper_editorial": "grid",
        "signal_glow": "mesh",
        "infra_dark": "rings",
    }.get(scene.theme, "rings")


def _valid_style_combo(theme: str, tone: str) -> bool:
    allowed = {
        "infra_dark": {"amber", "green", "blue", "purple", "mixed"},
        "paper_editorial": {"amber", "green", "blue", "mixed"},
        "signal_glow": {"green", "blue", "purple"},
    }
    return tone in allowed.get(theme, set())


def _scene_output_tokens(scene: Scene) -> list[str]:
    case = CASE_BY_TOPIC[scene.topic]
    base_attrs = dict(case["attrs"])
    return [
        "[scene]",
        f"[canvas:{base_attrs.get('canvas', 'wide')}]",
        f"[layout:{scene.layout}]",
        f"[theme:{scene.theme}]",
        f"[tone:{scene.tone}]",
        f"[frame:{_frame(scene, base_attrs)}]",
        f"[density:{_style_density(scene)}]",
        f"[inset:{_inset(scene, base_attrs)}]",
        f"[gap:{_gap(scene, base_attrs)}]",
        f"[hero:{base_attrs.get('hero', 'left')}]",
        f"[columns:{base_attrs.get('columns', '1')}]",
        f"[emphasis:{base_attrs.get('emphasis', 'top')}]",
        f"[rail:{_rail(scene, base_attrs)}]",
        f"[background:{_background(scene)}]",
        f"[connector:{base_attrs.get('connector', 'line')}]",
        f"[topic:{scene.topic}]",
        *list(case["components"]),
        "[/scene]",
    ]


def _row(scene: Scene) -> str:
    return " ".join(scene.prompt_tokens + _scene_output_tokens(scene))


def _render_svg(scene: Scene) -> str:
    case = CASE_BY_TOPIC[scene.topic]
    return render_structured_scene_spec11_svg(
        " ".join(_scene_output_tokens(scene)),
        content=dict(case["content_json"]),
    )


def _ordered_domain_tokens(rows: list[str]) -> list[str]:
    domain_tokens: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for token in row.split():
            if not token:
                continue
            if token.startswith("[") and token.endswith("]"):
                if token not in seen:
                    seen.add(token)
                    domain_tokens.append(token)
                continue
            if token.startswith("@"):
                if token not in seen:
                    seen.add(token)
                    domain_tokens.append(token)
    return domain_tokens


def _build_scenes() -> list[Scene]:
    scenes: list[Scene] = []
    for topic, case in CASE_BY_TOPIC.items():
        layout = str(case["attrs"].get("layout") or "")
        if layout not in LAYOUTS:
            continue
        for theme in THEMES:
            for tone in TONES:
                if not _valid_style_combo(theme, tone):
                    continue
                for density in DENSITIES:
                    scenes.append(
                        Scene(
                            layout=layout,
                            topic=topic,
                            theme=theme,
                            tone=tone,
                            density=density,
                        )
                    )
    return sorted(scenes, key=_scene_sort_key)


def _write_probe_report_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_probe_report_contract.json"
    payload = {
        "schema": "ck.probe_report_contract.v1",
        "title": "Spec11 Keyed Scene DSL Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec11.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 256,
            "temperature": 0.0,
        },
        "catalog": {
            "format": "json_rows",
            "path": f"{prefix}_render_catalog.json",
            "prompt_key": "prompt",
            "output_key": "output_tokens",
            "rendered_key": "svg_xml",
            "rendered_mime": "image/svg+xml",
            "split_key": "split",
        },
        "splits": [
            {"name": "train", "label": "Train prompts", "format": "lines", "path": f"{prefix}_seen_prompts.txt", "limit": 12},
            {"name": "test", "label": "Holdout prompts", "format": "lines", "path": f"{prefix}_holdout_prompts.txt", "limit": 12},
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _write_eval_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_eval_contract.json"
    payload = {
        "schema": "ck.eval_contract.v1",
        "dataset_type": "svg",
        "goal": "Emit a keyed asset-grounded scene DSL that compiles with external content_json into production-style infographic SVG.",
        "notes": [
            "Spec11 separates structure from content; the model emits keyed scene structure and the compiler resolves external text/data.",
            "The compiler owns gradients, markers, wrapped text, panel composition, and exact SVG geometry.",
            "Probe scoring remains exact-match on the scene DSL plus exact-match on the compiled SVG under the provided content_json.",
        ],
        "scorer": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec11.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 256,
            "temperature": 0.0,
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate asset-grounded spec11 keyed scene DSL dataset with compiler-backed SVG renders")
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec11_keyed_scene_dsl", help="Output file prefix")
    ap.add_argument("--train-repeats", type=int, default=3, help="Repeats for seen combos")
    ap.add_argument("--holdout-repeats", type=int, default=1, help="Repeats for holdout combos")
    args = ap.parse_args()

    if args.train_repeats < 1:
        raise SystemExit("--train-repeats must be >= 1")
    if args.holdout_repeats < 1:
        raise SystemExit("--holdout-repeats must be >= 1")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = _build_scenes()
    holdout = [scene for scene in scenes if _is_holdout(scene)]
    train = [scene for scene in scenes if not _is_holdout(scene)]

    train_rows = [_row(scene) for scene in train for _ in range(int(args.train_repeats))]
    holdout_rows = [_row(scene) for scene in holdout for _ in range(int(args.holdout_repeats))]
    tokenizer_corpus = [_row(scene) for scene in scenes] + [scene.prompt_text for scene in scenes]
    seen_prompts = [scene.prompt_text for scene in train]
    holdout_prompts = [scene.prompt_text for scene in holdout]
    render_rows = [
        {
            "prompt": scene.prompt_text,
            "output_tokens": " ".join(_scene_output_tokens(scene)),
            "content_json": dict(CASE_BY_TOPIC[scene.topic]["content_json"]),
            "svg_xml": _render_svg(scene),
            "split": "holdout" if scene in holdout else "train",
            "layout": scene.layout,
            "topic": scene.topic,
            "source_asset": CASE_BY_TOPIC[scene.topic]["asset"],
        }
        for scene in scenes
    ]

    domain_tokens = _ordered_domain_tokens(tokenizer_corpus)
    tokenizer, tokenizer_meta = _build_tokenizer_payload(domain_tokens)
    tokenizer_json, tokenizer_bin = _write_tokenizer_artifacts(
        tokenizer,
        tokenizer_meta,
        out_dir / f"{args.prefix}_tokenizer",
    )

    _write_lines(out_dir / f"{args.prefix}_train.txt", train_rows)
    _write_lines(out_dir / f"{args.prefix}_holdout.txt", holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_tokenizer_corpus.txt", tokenizer_corpus)
    _write_lines(out_dir / f"{args.prefix}_seen_prompts.txt", seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_holdout_prompts.txt", holdout_prompts)
    _write_lines(out_dir / f"{args.prefix}_reserved_control_tokens.txt", domain_tokens)

    vocab_spec = {
        "format": "svg-structured-fixed-vocab.v7",
        "prefix": args.prefix,
        "system_tokens": tokenizer.get("added_tokens", []),
        "tokenizer_json": str(tokenizer_json),
        "tokenizer_bin": str(tokenizer_bin),
        "vocab_size": int(tokenizer_meta["vocab_size"]),
        "num_merges": int(tokenizer_meta["num_merges"]),
    }
    (out_dir / f"{args.prefix}_vocab.json").write_text(
        json.dumps(vocab_spec, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"{args.prefix}_render_catalog.json").write_text(
        json.dumps(render_rows, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    probe_contract_path = _write_probe_report_contract(out_dir, args.prefix)
    eval_contract_path = _write_eval_contract(out_dir, args.prefix)

    manifest = {
        "schema": "ck.generated_dataset.v1",
        "line_name": "spec11_keyed_scene_dsl",
        "prefix": args.prefix,
        "out_dir": str(out_dir),
        "topics": list(TOPICS),
        "layouts": list(LAYOUTS),
        "source_assets": {topic: CASE_BY_TOPIC[topic]["asset"] for topic in TOPICS},
        "artifacts": {
            "train": str(out_dir / f"{args.prefix}_train.txt"),
            "holdout": str(out_dir / f"{args.prefix}_holdout.txt"),
            "seen_prompts": str(out_dir / f"{args.prefix}_seen_prompts.txt"),
            "holdout_prompts": str(out_dir / f"{args.prefix}_holdout_prompts.txt"),
            "render_catalog": str(out_dir / f"{args.prefix}_render_catalog.json"),
            "tokenizer_corpus": str(out_dir / f"{args.prefix}_tokenizer_corpus.txt"),
            "reserved_control_tokens": str(out_dir / f"{args.prefix}_reserved_control_tokens.txt"),
            "vocab": str(out_dir / f"{args.prefix}_vocab.json"),
            "probe_report_contract": str(probe_contract_path),
            "eval_contract": str(eval_contract_path),
        },
        "counts": {
            "all": len(scenes),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "holdout_prompts": len(holdout_prompts),
            "tokenizer_rows": len(tokenizer_corpus),
        },
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
