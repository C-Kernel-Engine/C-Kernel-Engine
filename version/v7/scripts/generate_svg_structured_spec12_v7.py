#!/usr/bin/env python3
"""
Generate a spec12 asset-grounded scene-DSL dataset from compact gold mappings.

Spec12 keeps structure and content separate:
- model target: scene DSL only
- visible text/data: external content_json
- compiler: render scene DSL + content_json -> SVG

Unlike spec10/spec11 literal row packing, spec12 uses smaller component blocks
so the model learns compositional structure rather than monolithic payload rows.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generate_svg_structured_spec06_v7 import (
    _build_tokenizer_payload,
    _write_lines,
    _write_tokenizer_artifacts,
)
from render_svg_structured_scene_spec12_v7 import render_structured_scene_spec12_svg


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "version" / "v7" / "reports" / "spec12_gold_mappings" / "spec12_gold_mappings_compact_20260318.json"
FREEZE_VOCAB_ALIAS = os.environ.get("CK_SPEC12_FREEZE_VOCAB_ALIAS", "").strip().lower() in {"1", "true", "yes", "on"}

LAYOUTS = ("table_matrix", "decision_tree", "memory_map")
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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_style_combo(theme: str, tone: str) -> bool:
    allowed = {
        "infra_dark": {"amber", "green", "blue", "purple", "mixed"},
        "paper_editorial": {"amber", "green", "blue", "mixed"},
        "signal_glow": {"green", "blue", "purple"},
    }
    return tone in allowed.get(theme, set())


def _parse_scene_document(text: str) -> dict[str, Any]:
    attrs: dict[str, str] = {}
    components: list[str] = []
    for raw in str(text or "").splitlines():
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
        raise ValueError(f"invalid compact component token: {token}")
    body = text[1:-1]
    if ":" not in body:
        raise ValueError(f"invalid compact component payload: {token}")
    name, payload = body.split(":", 1)
    return name.strip(), payload.strip()


def _split_payload(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split("|") if part.strip()]


def _decompose_component(token: str) -> list[str]:
    name, payload = _split_component(token)
    parts = _split_payload(payload)

    if name in {"header_band", "legend_block", "table_block", "note_band", "entry_badge", "footer_note", "address_strip"}:
        ref = parts[0] if parts else ""
        return [f"[{name}]", f"[ref:{ref}]", f"[/{name}]"]

    if name == "decision_node":
        if len(parts) < 2:
            raise ValueError(f"decision_node requires node id + ref: {token}")
        return [f"[{name}]", f"[node_id:{parts[0]}]", f"[ref:{parts[1]}]", f"[/{name}]"]

    if name == "decision_edge":
        if len(parts) < 2 or "->" not in parts[0]:
            raise ValueError(f"decision_edge requires src->dst + label ref: {token}")
        src, dst = parts[0].split("->", 1)
        return [f"[{name}]", f"[from_ref:{src}]", f"[to_ref:{dst}]", f"[label_ref:{parts[1]}]", f"[/{name}]"]

    if name == "outcome_panel":
        if len(parts) < 2:
            raise ValueError(f"outcome_panel requires panel id + ref: {token}")
        return [f"[{name}]", f"[panel_id:{parts[0]}]", f"[ref:{parts[1]}]", f"[/{name}]"]

    if name == "memory_segment":
        if len(parts) < 2:
            raise ValueError(f"memory_segment requires segment id + ref: {token}")
        return [f"[{name}]", f"[segment_id:{parts[0]}]", f"[ref:{parts[1]}]", f"[/{name}]"]

    if name == "region_bracket":
        if len(parts) >= 2:
            return [f"[{name}]", f"[bracket_id:{parts[0]}]", f"[ref:{parts[1]}]", f"[/{name}]"]
        if len(parts) == 1:
            return [f"[{name}]", f"[ref:{parts[0]}]", f"[/{name}]"]
        raise ValueError(f"region_bracket requires ref payload: {token}")

    if name == "info_card":
        if len(parts) < 2:
            raise ValueError(f"info_card requires card id + ref: {token}")
        return [f"[{name}]", f"[card_id:{parts[0]}]", f"[ref:{parts[1]}]", f"[/{name}]"]

    raise ValueError(f"unsupported spec12 component: {name}")


def _decompose_components(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        out.extend(_decompose_component(token))
    return out


def _ordered_domain_tokens(rows: list[str]) -> list[str]:
    domain_tokens: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for token in row.split():
            if not (token.startswith("[") and token.endswith("]")):
                continue
            if token in seen:
                continue
            seen.add(token)
            domain_tokens.append(token)
    return domain_tokens


def _scene_sort_key(scene: "Scene") -> tuple[str, str, str, str, str]:
    return (scene.layout, scene.case_id, scene.theme, scene.tone, scene.density)


def _is_holdout(scene: "Scene") -> bool:
    score = (
        LAYOUTS.index(scene.layout)
        + CASE_IDS.index(scene.case_id)
        + THEMES.index(scene.theme)
        + TONES.index(scene.tone)
        + DENSITIES.index(scene.density)
    )
    return (score % 5) == 0


@dataclass(frozen=True)
class Scene:
    case_id: str
    layout: str
    topic_token: str
    prompt_topic: str
    theme: str = "infra_dark"
    tone: str = "blue"
    density: str = "compact"

    @property
    def prompt_tokens(self) -> list[str]:
        return [
            "[task:svg]",
            f"[layout:{self.layout}]",
            f"[topic:{self.topic_token}]",
            f"[theme:{self.theme}]",
            f"[tone:{self.tone}]",
            f"[density:{self.density}]",
            "[OUT]",
        ]

    @property
    def prompt_text(self) -> str:
        return " ".join(self.prompt_tokens)


def _topic_phrase(topic: str) -> str:
    return str(topic or "").replace("_", " ")


def _layout_phrase(layout: str) -> str:
    return {
        "table_matrix": "table matrix infographic",
        "decision_tree": "decision tree infographic",
        "memory_map": "memory map infographic",
    }.get(str(layout or ""), "scene infographic")


def _style_phrase(theme: str, tone: str, density: str) -> str:
    theme_phrase = {
        "infra_dark": "an infra dark",
        "paper_editorial": "a paper editorial",
        "signal_glow": "a signal glow",
    }.get(str(theme or ""), "a structured")
    density_phrase = {
        "compact": "compact",
        "balanced": "balanced",
        "airy": "airy",
    }.get(str(density or ""), "balanced")
    tone_phrase = str(tone or "blue").replace("_", " ")
    return f"{theme_phrase} style with {tone_phrase} tone and {density_phrase} spacing"


def _dedupe_preserve(rows: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        text = str(row or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _bridge_prompt_variants(scene: "Scene") -> list[str]:
    layout = _layout_phrase(scene.layout)
    topic = _topic_phrase(scene.prompt_topic)
    style = _style_phrase(scene.theme, scene.tone, scene.density)
    tag_tail = scene.prompt_text
    return _dedupe_preserve(
        [
            f"Create a {layout} about {topic} in {style}. Return scene DSL only. {tag_tail}",
            f"Plan a {layout} for {topic} using {style}. Emit [scene] ... [/scene] only. {tag_tail}",
            f"Draft the scene DSL for a {layout} covering {topic}. Keep the compiler-facing tags exact and nothing else. {tag_tail}",
        ]
    )


def _hidden_prompt_variants(scene: "Scene") -> list[str]:
    layout = _layout_phrase(scene.layout)
    topic = _topic_phrase(scene.prompt_topic)
    style = _style_phrase(scene.theme, scene.tone, scene.density)
    tag_tail = scene.prompt_text
    return _dedupe_preserve(
        [
            f"Compose a complete {layout} on {topic} with {style}. Output only the structured scene program. {tag_tail}",
            f"Draft the scene DSL for a {layout} covering {topic}. Keep the compiler-facing tags exact and nothing else. {tag_tail}",
            f"Produce the scene DSL for a {layout} about {topic} in {style}. Return only compiler-safe scene tags. {tag_tail}",
            f"Prepare a complete {layout} for {topic} using {style}. Output [scene] ... [/scene] only. {tag_tail}",
        ]
    )


@dataclass(frozen=True)
class SceneCase:
    case_id: str
    topic_token: str
    prompt_topic: str
    layout: str
    asset: str
    attrs: dict[str, str]
    components: tuple[str, ...]
    content_json: dict[str, Any]


RAW_CASE_BY_TOPIC: dict[str, dict[str, Any]] = {}
MANIFEST_DOC = _load_json(DEFAULT_MANIFEST)
for row in MANIFEST_DOC.get("mappings") or []:
    if not isinstance(row, dict):
        continue
    scene_path = ROOT / str(row.get("scene_dsl") or "")
    content_path = ROOT / str(row.get("content_json") or "")
    asset = str(row.get("asset") or "")
    if not scene_path.exists() or not content_path.exists():
        continue
    parsed = _parse_scene_document(scene_path.read_text(encoding="utf-8"))
    topic = str(parsed["attrs"].get("topic") or "")
    layout = str(parsed["attrs"].get("layout") or "")
    if not topic or layout not in LAYOUTS:
        continue
    RAW_CASE_BY_TOPIC[topic] = {
        "asset": asset,
        "layout": layout,
        "attrs": dict(parsed["attrs"]),
        "components": _decompose_components(list(parsed["components"])),
        "content_json": _load_json(content_path),
    }


def _card_as_lines(card: Any, *, fallback_title: str) -> dict[str, Any]:
    if not isinstance(card, dict):
        return {"title": fallback_title, "lines": []}
    if isinstance(card.get("lines"), list):
        return {
            "title": str(card.get("title") or fallback_title),
            "lines": [str(line) for line in card.get("lines") or []],
        }
    if isinstance(card.get("items"), list):
        return {
            "title": str(card.get("title") or fallback_title),
            "lines": [str(line) for line in card.get("items") or []],
        }
    return {"title": str(card.get("title") or fallback_title), "lines": []}


def _adapt_training_memory_canary_content(content: dict[str, Any]) -> dict[str, Any]:
    segments = content.get("segments") if isinstance(content.get("segments"), dict) else {}
    cards = content.get("cards") if isinstance(content.get("cards"), dict) else {}
    grad_weights = segments.get("grad_weights") if isinstance(segments.get("grad_weights"), dict) else {}
    return {
        "header": dict(content.get("header") or {}),
        "offsets": list(content.get("offsets") or []),
        "segments": {
            "embeddings": dict(segments.get("weights") or {}),
            "layer_0": dict(segments.get("activations") or {}),
            "layer_1": dict(segments.get("grad_act") or {}),
            "mid_layers": {
                "title": str(grad_weights.get("title") or "grad_weights"),
                "caption": str(grad_weights.get("caption") or ""),
            },
            "layer_23": dict(segments.get("optimizer") or {}),
            "lm_head": dict(segments.get("temp_aux") or {}),
            "runtime": {
                "title": "phases + checks",
                "size": "training guards",
                "caption": "Phase routing, audit checks, and guard metadata stay visible during long runs.",
                "parts": ["phase", "check", "guard"],
            },
        },
        "cards": {
            "benefits": dict(cards.get("canaries") or {}),
            "layout_json": _card_as_lines(cards.get("phases"), fallback_title="Phases"),
            "direct_access": _card_as_lines(cards.get("checks"), fallback_title="Checks"),
        },
    }


def _build_cases(*, freeze_vocab_alias: bool) -> dict[str, SceneCase]:
    cases: dict[str, SceneCase] = {}
    if not freeze_vocab_alias:
        for topic, case in RAW_CASE_BY_TOPIC.items():
            cases[topic] = SceneCase(
                case_id=topic,
                topic_token=topic,
                prompt_topic=topic,
                layout=str(case["layout"]),
                asset=str(case["asset"]),
                attrs=dict(case["attrs"]),
                components=tuple(case["components"]),
                content_json=dict(case["content_json"]),
            )
        return cases

    for topic in ("failure_triage", "model_memory_layout", "quantization_formats"):
        case = RAW_CASE_BY_TOPIC[topic]
        cases[topic] = SceneCase(
            case_id=topic,
            topic_token=topic,
            prompt_topic=topic,
            layout=str(case["layout"]),
            asset=str(case["asset"]),
            attrs=dict(case["attrs"]),
            components=tuple(case["components"]),
            content_json=dict(case["content_json"]),
        )

    edge_case = RAW_CASE_BY_TOPIC["edge_case_matrix"]
    cases["edge_case_matrix"] = SceneCase(
        case_id="edge_case_matrix",
        topic_token="quantization_formats",
        prompt_topic="edge_case_matrix",
        layout=str(edge_case["layout"]),
        asset=str(edge_case["asset"]),
        attrs=dict(edge_case["attrs"]),
        components=tuple(edge_case["components"]),
        content_json=dict(edge_case["content_json"]),
    )

    template_memory = RAW_CASE_BY_TOPIC["model_memory_layout"]
    canary_case = RAW_CASE_BY_TOPIC["training_memory_canary"]
    cases["training_memory_canary"] = SceneCase(
        case_id="training_memory_canary",
        topic_token="model_memory_layout",
        prompt_topic="training_memory_canary",
        layout=str(template_memory["layout"]),
        asset=str(canary_case["asset"]),
        attrs=dict(template_memory["attrs"]),
        components=tuple(template_memory["components"]),
        content_json=_adapt_training_memory_canary_content(dict(canary_case["content_json"])),
    )
    return cases


CASES_BY_ID = _build_cases(freeze_vocab_alias=FREEZE_VOCAB_ALIAS)
CASE_IDS = tuple(sorted(CASES_BY_ID))
TOPICS = tuple(sorted({case.topic_token for case in CASES_BY_ID.values()}))


def _scene_output_tokens(scene: Scene) -> list[str]:
    case = CASES_BY_ID[scene.case_id]
    attrs = dict(case.attrs)
    tokens = [
        "[scene]",
        f"[layout:{scene.layout}]",
        f"[theme:{scene.theme}]",
        f"[tone:{scene.tone}]",
        f"[density:{scene.density}]",
        f"[topic:{scene.topic_token}]",
    ]
    for optional in ("canvas", "frame", "inset", "gap", "background", "connector"):
        value = str(attrs.get(optional) or "").strip()
        if value:
            tokens.append(f"[{optional}:{value}]")
    tokens.extend(list(case.components))
    tokens.append("[/scene]")
    return tokens


def _row(scene: Scene) -> str:
    return " ".join(scene.prompt_tokens + _scene_output_tokens(scene))


def _render_svg(scene: Scene) -> str:
    case = CASES_BY_ID[scene.case_id]
    return render_structured_scene_spec12_svg(" ".join(_scene_output_tokens(scene)), content=dict(case.content_json))


def _build_scenes() -> list[Scene]:
    scenes: list[Scene] = []
    for case_id, case in CASES_BY_ID.items():
        layout = str(case.layout)
        for theme in THEMES:
            for tone in TONES:
                if not _valid_style_combo(theme, tone):
                    continue
                for density in DENSITIES:
                    scenes.append(
                        Scene(
                            case_id=case_id,
                            layout=layout,
                            topic_token=case.topic_token,
                            prompt_topic=case.prompt_topic,
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
        "title": "Spec12 Scene DSL Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec12.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {"max_tokens": 384, "temperature": 0.0},
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
        "goal": "Emit a compositional spec12 scene DSL that compiles with external content_json into richer infographic SVG.",
        "notes": [
            "Spec12 uses smaller component blocks than spec11 so the model learns composition, ids, refs, and topology without monolithic payload tokens.",
            "The compiler still owns exact geometry, gradients, markers, grouping, and typography.",
            "Probe scoring is exact-match on the scene DSL plus exact-match on the compiled SVG under content_json.",
        ],
        "scorer": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec12.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {"max_tokens": 384, "temperature": 0.0},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _scene_catalog_rows(
    scene: Scene,
    *,
    output_tokens: str,
    svg_xml: str,
    split: str,
    train_prompt_variants: int,
    hidden_prompt_variants: int,
) -> list[dict[str, Any]]:
    tags = {
        "layout": scene.layout,
        "topic": scene.topic_token,
        "theme": scene.theme,
        "tone": scene.tone,
        "density": scene.density,
    }
    case = CASES_BY_ID[scene.case_id]
    rows: list[dict[str, Any]] = []

    def add(prompt: str, prompt_surface: str, prompt_split: str, *, training_prompt: bool) -> None:
        rows.append(
            {
                "prompt": prompt,
                "canonical_prompt": scene.prompt_text,
                "output_tokens": output_tokens,
                "content_json": dict(case.content_json),
                "svg_xml": svg_xml,
                "split": prompt_split,
                "layout": scene.layout,
                "topic": scene.topic_token,
                "case_id": scene.case_id,
                "prompt_topic": scene.prompt_topic,
                "theme": scene.theme,
                "tone": scene.tone,
                "density": scene.density,
                "source_asset": case.asset,
                "prompt_surface": prompt_surface,
                "training_prompt": bool(training_prompt),
                "tags": dict(tags),
            }
        )

    add(scene.prompt_text, "tag_canonical", split, training_prompt=True)

    for idx, prompt in enumerate(_bridge_prompt_variants(scene)[: max(0, int(train_prompt_variants))], start=1):
        add(prompt, f"bridge_{idx}", split, training_prompt=True)

    hidden_split = "probe_hidden_train" if split == "train" else "probe_hidden_holdout"
    for idx, prompt in enumerate(_hidden_prompt_variants(scene)[: max(0, int(hidden_prompt_variants))], start=1):
        add(prompt, f"hidden_{idx}", hidden_split, training_prompt=False)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a compositional spec12 asset-grounded scene DSL dataset")
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec12_scene_dsl", help="Output file prefix")
    ap.add_argument("--train-repeats", type=int, default=4, help="Repeats for seen combos")
    ap.add_argument("--holdout-repeats", type=int, default=1, help="Repeats for holdout combos")
    ap.add_argument("--train-prompt-variants", type=int, default=3, help="How many hybrid paraphrase prompts to add per scene to the train/holdout corpora")
    ap.add_argument("--hidden-prompt-variants", type=int, default=4, help="How many withheld paraphrase prompts to add per scene for hidden eval only")
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

    render_rows: list[dict[str, Any]] = []
    train_rows: list[str] = []
    holdout_rows: list[str] = []
    tokenizer_corpus: list[str] = []
    seen_prompts: list[str] = []
    holdout_prompts: list[str] = []
    hidden_seen_prompts: list[str] = []
    hidden_holdout_prompts: list[str] = []

    for scene in scenes:
        split = "holdout" if scene in holdout else "train"
        output_tokens = " ".join(_scene_output_tokens(scene))
        svg_xml = _render_svg(scene)
        scene_rows = _scene_catalog_rows(
            scene,
            output_tokens=output_tokens,
            svg_xml=svg_xml,
            split=split,
            train_prompt_variants=int(args.train_prompt_variants),
            hidden_prompt_variants=int(args.hidden_prompt_variants),
        )
        render_rows.extend(scene_rows)
        for row in scene_rows:
            prompt = str(row.get("prompt") or "").strip()
            row_text = f"{prompt} {output_tokens}".strip()
            tokenizer_corpus.append(row_text)
            tokenizer_corpus.append(prompt)
            row_split = str(row.get("split") or "")
            if row_split == "train":
                seen_prompts.append(prompt)
                repeats = int(args.train_repeats) if str(row.get("prompt_surface") or "") == "tag_canonical" else 1
                for _ in range(max(1, repeats)):
                    train_rows.append(row_text)
            elif row_split == "holdout":
                holdout_prompts.append(prompt)
                repeats = int(args.holdout_repeats) if str(row.get("prompt_surface") or "") == "tag_canonical" else 1
                for _ in range(max(1, repeats)):
                    holdout_rows.append(row_text)
            elif row_split == "probe_hidden_train":
                hidden_seen_prompts.append(prompt)
            elif row_split == "probe_hidden_holdout":
                hidden_holdout_prompts.append(prompt)

    seen_prompts = _dedupe_preserve(seen_prompts)
    holdout_prompts = _dedupe_preserve(holdout_prompts)
    hidden_seen_prompts = _dedupe_preserve(hidden_seen_prompts)
    hidden_holdout_prompts = _dedupe_preserve(hidden_holdout_prompts)
    tokenizer_corpus = _dedupe_preserve(tokenizer_corpus)

    domain_tokens = _ordered_domain_tokens(tokenizer_corpus)
    tokenizer, tokenizer_meta = _build_tokenizer_payload(domain_tokens)
    tokenizer_json, tokenizer_bin = _write_tokenizer_artifacts(tokenizer, tokenizer_meta, out_dir / f"{args.prefix}_tokenizer")

    _write_lines(out_dir / f"{args.prefix}_train.txt", train_rows)
    _write_lines(out_dir / f"{args.prefix}_holdout.txt", holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_tokenizer_corpus.txt", tokenizer_corpus)
    _write_lines(out_dir / f"{args.prefix}_seen_prompts.txt", seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_holdout_prompts.txt", holdout_prompts)
    _write_lines(out_dir / f"{args.prefix}_hidden_seen_prompts.txt", hidden_seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_hidden_holdout_prompts.txt", hidden_holdout_prompts)
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
    (out_dir / f"{args.prefix}_vocab.json").write_text(json.dumps(vocab_spec, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (out_dir / f"{args.prefix}_render_catalog.json").write_text(json.dumps(render_rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    probe_contract_path = _write_probe_report_contract(out_dir, args.prefix)
    eval_contract_path = _write_eval_contract(out_dir, args.prefix)

    manifest = {
        "schema": "ck.generated_dataset.v1",
        "line_name": "spec12_scene_dsl",
        "prefix": args.prefix,
        "out_dir": str(out_dir),
        "topics": list(TOPICS),
        "case_ids": list(CASE_IDS),
        "layouts": list(LAYOUTS),
        "source_assets": {case_id: CASES_BY_ID[case_id].asset for case_id in CASE_IDS},
        "artifacts": {
            "train": str(out_dir / f"{args.prefix}_train.txt"),
            "holdout": str(out_dir / f"{args.prefix}_holdout.txt"),
            "seen_prompts": str(out_dir / f"{args.prefix}_seen_prompts.txt"),
            "holdout_prompts": str(out_dir / f"{args.prefix}_holdout_prompts.txt"),
            "hidden_seen_prompts": str(out_dir / f"{args.prefix}_hidden_seen_prompts.txt"),
            "hidden_holdout_prompts": str(out_dir / f"{args.prefix}_hidden_holdout_prompts.txt"),
            "render_catalog": str(out_dir / f"{args.prefix}_render_catalog.json"),
            "tokenizer_corpus": str(out_dir / f"{args.prefix}_tokenizer_corpus.txt"),
            "reserved_control_tokens": str(out_dir / f"{args.prefix}_reserved_control_tokens.txt"),
            "vocab": str(out_dir / f"{args.prefix}_vocab.json"),
            "probe_report_contract": str(probe_contract_path),
            "eval_contract": str(eval_contract_path),
        },
        "counts": {
            "all": len(scenes),
            "cases": len(CASE_IDS),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "train_prompts": len(seen_prompts),
            "holdout_prompts": len(holdout_prompts),
            "hidden_seen_prompts": len(hidden_seen_prompts),
            "hidden_holdout_prompts": len(hidden_holdout_prompts),
            "tokenizer_rows": len(tokenizer_corpus),
        },
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
