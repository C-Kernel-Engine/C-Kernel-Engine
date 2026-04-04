#!/usr/bin/env python3
"""Generate a minimal comparison-board spec14a dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generate_svg_structured_spec06_v7 import (
    _build_tokenizer_payload,
    _ordered_domain_tokens,
    _write_lines,
    _write_tokenizer_artifacts,
)
from render_svg_structured_scene_spec14a_v7 import render_structured_scene_spec14a_svg
from spec14a_scene_canonicalizer_v7 import SceneComponent, SceneDocument, canonicalize_scene_text


ROOT = Path(__file__).resolve().parents[3]
GOLD = ROOT / "version" / "v7" / "reports" / "spec14a_gold_mappings"


@dataclass(frozen=True)
class BoardCase:
    case_id: str
    prompt_topic: str
    source_asset: str
    scene_path: Path
    content_path: Path
    split: str = "train"
    scene_text_override: str | None = None
    content_json_override: dict[str, Any] | None = None

    @property
    def scene_text(self) -> str:
        if self.scene_text_override is not None:
            return str(self.scene_text_override).strip()
        return self.scene_path.read_text(encoding="utf-8").strip()

    @property
    def content_json(self) -> dict[str, Any]:
        if self.content_json_override is not None:
            return json.loads(json.dumps(self.content_json_override))
        return json.loads(self.content_path.read_text(encoding="utf-8"))


def _cases() -> list[BoardCase]:
    return [
        BoardCase(
            case_id="tokenizer_performance_comparison",
            prompt_topic="tokenizer throughput comparison",
            source_asset="tokenizer-performance-comparison.svg",
            scene_path=GOLD / "tokenizer-performance-comparison.scene.compact.dsl",
            content_path=GOLD / "tokenizer-performance-comparison.content.json",
        ),
        BoardCase(
            case_id="tokenizer_algorithms_comparison",
            prompt_topic="tokenizer algorithm comparison",
            source_asset="sentencepiece-vs-bpe-wordpiece.svg",
            scene_path=GOLD / "sentencepiece-vs-bpe-wordpiece.scene.compact.dsl",
            content_path=GOLD / "sentencepiece-vs-bpe-wordpiece.content.json",
        ),
        BoardCase(
            case_id="rope_layouts_comparison",
            prompt_topic="rope layout comparison",
            source_asset="rope-layouts-compared.svg",
            scene_path=GOLD / "rope-layouts-compared.scene.compact.dsl",
            content_path=GOLD / "rope-layouts-compared.content.json",
        ),
        BoardCase(
            case_id="compute_bandwidth_chasm",
            prompt_topic="compute to bandwidth comparison",
            source_asset="compute-bandwidth-chasm.svg",
            scene_path=GOLD / "compute-bandwidth-chasm.scene.compact.dsl",
            content_path=GOLD / "compute-bandwidth-chasm.content.json",
        ),
        BoardCase(
            case_id="quantization_formats_comparison",
            prompt_topic="quantization format comparison",
            source_asset="quantization-formats.svg",
            scene_path=GOLD / "quantization-formats.scene.compact.dsl",
            content_path=GOLD / "quantization-formats.content.json",
            split="holdout",
        ),
    ]


def _component_token(component: SceneComponent) -> str:
    fields = component.fields
    if component.name in {"header_band", "legend_block", "footer_note"}:
        payload = str(fields.get("ref") or "").strip()
    elif component.name == "comparison_column":
        payload = "|".join(
            [
                str(fields.get("column_id") or "").strip(),
                str(fields.get("ref") or "").strip(),
            ]
        )
    elif component.name == "comparison_metric":
        payload = "|".join(
            [
                str(fields.get("metric_id") or "").strip(),
                str(fields.get("ref") or "").strip(),
            ]
        )
    elif component.name == "comparison_callout":
        payload = "|".join(
            [
                str(fields.get("callout_id") or "").strip(),
                str(fields.get("ref") or "").strip(),
            ]
        )
    else:
        raise ValueError(f"unsupported component for spec14a scene synthesis: {component.name}")
    return f"[{component.name}:{payload}]"


def _scene_text_from_document(scene: SceneDocument) -> str:
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
        f"[columns:{scene.columns}]",
        f"[background:{scene.background}]",
        f"[topic:{scene.topic}]",
    ]
    tokens.extend(_component_token(component) for component in scene.components)
    tokens.append("[/scene]")
    return " ".join(tokens)


def _remix_scene_document(
    scene: SceneDocument,
    *,
    theme: str,
    tone: str,
    background: str,
    metric_count: int,
    callout_count: int,
) -> SceneDocument:
    metric_ids = {f"metric_{idx}" for idx in range(1, int(metric_count) + 1)}
    callout_ids = {f"callout_{idx}" for idx in range(1, int(callout_count) + 1)}
    remixed_components: list[SceneComponent] = []
    for component in scene.components:
        if component.name == "comparison_metric":
            metric_id = str(component.fields.get("metric_id") or "").strip()
            if metric_id not in metric_ids:
                continue
        if component.name == "comparison_callout":
            callout_id = str(component.fields.get("callout_id") or "").strip()
            if callout_id not in callout_ids:
                continue
        remixed_components.append(component)
    return SceneDocument(
        canvas=scene.canvas,
        layout=scene.layout,
        theme=theme,
        tone=tone,
        frame=scene.frame,
        density=scene.density,
        inset=scene.inset,
        gap=scene.gap,
        hero=scene.hero,
        columns=scene.columns,
        emphasis=scene.emphasis,
        rail=scene.rail,
        background=background,
        connector=scene.connector,
        topic=scene.topic,
        components=tuple(remixed_components),
    )


def _composed_training_cases(cases: list[BoardCase]) -> list[BoardCase]:
    by_id = {case.case_id: case for case in cases}
    compose_specs = {
        "tokenizer_performance_comparison": [
            ("paper_editorial", "blue", "mesh"),
            ("infra_dark", "blue", "mesh"),
            ("infra_dark", "blue", "grid"),
        ],
        "tokenizer_algorithms_comparison": [
            ("infra_dark", "green", "grid"),
            ("infra_dark", "blue", "mesh"),
            ("paper_editorial", "green", "mesh"),
        ],
    }
    composed: list[BoardCase] = []
    for base_case_id, combos in compose_specs.items():
        base = by_id.get(base_case_id)
        if base is None:
            continue
        base_scene = canonicalize_scene_text(base.scene_text)
        base_runtime = base_scene.to_runtime()
        metric_count = len(base_runtime.get("components_by_name", {}).get("comparison_metric", []))
        callout_count = len(base_runtime.get("components_by_name", {}).get("comparison_callout", []))
        if metric_count != 4 or callout_count != 2:
            continue
        for theme, tone, background in combos:
            if (
                base_runtime["theme"] == theme
                and base_runtime["tone"] == tone
                and base_runtime["background"] == background
            ):
                continue
            scene_text = _scene_text_from_document(
                _remix_scene_document(
                    base_scene,
                    theme=theme,
                    tone=tone,
                    background=background,
                    metric_count=metric_count,
                    callout_count=callout_count,
                )
            )
            composed.append(
                BoardCase(
                    case_id=f"{base.case_id}_compose_{theme}_{tone}_{background}",
                    prompt_topic=base.prompt_topic,
                    source_asset=f"{base.source_asset}#compose:{theme}:{tone}:{background}",
                    scene_path=base.scene_path,
                    content_path=base.content_path,
                    split="train_aug",
                    scene_text_override=scene_text,
                    content_json_override=base.content_json,
                )
            )
    return composed


def _canonical_prompt(scene_runtime: dict[str, Any]) -> str:
    metric_count = len(scene_runtime.get("components_by_name", {}).get("comparison_metric", []))
    callout_count = len(scene_runtime.get("components_by_name", {}).get("comparison_callout", []))
    return " ".join(
        [
            "[task:svg]",
            f"[layout:{scene_runtime['layout']}]",
            f"[topic:{scene_runtime['topic']}]",
            f"[theme:{scene_runtime['theme']}]",
            f"[tone:{scene_runtime['tone']}]",
            f"[background:{scene_runtime['background']}]",
            f"[density:{scene_runtime['density']}]",
            f"[columns:{scene_runtime['columns']}]",
            f"[metrics:{metric_count}]",
            f"[callouts:{callout_count}]",
            "[OUT]",
        ]
    )


def _is_default_board(scene_runtime: dict[str, Any]) -> bool:
    return (
        scene_runtime["theme"] == "infra_dark"
        and scene_runtime["tone"] == "amber"
        and scene_runtime.get("background") == "rings"
        and len(scene_runtime.get("components_by_name", {}).get("comparison_metric", [])) == 3
        and len(scene_runtime.get("components_by_name", {}).get("comparison_callout", [])) == 2
    )


def _bridge_prompt_variants(
    case: BoardCase,
    prompt_text: str,
    scene_runtime: dict[str, Any],
) -> list[tuple[str, str]]:
    theme = str(scene_runtime["theme"]).replace("_", " ")
    tone = str(scene_runtime["tone"])
    density = str(scene_runtime["density"])
    columns = str(scene_runtime["columns"])
    background = str(scene_runtime.get("background") or "grid")
    metric_count = len(scene_runtime.get("components_by_name", {}).get("comparison_metric", []))
    callout_count = len(scene_runtime.get("components_by_name", {}).get("comparison_callout", []))
    metric_guard = (
        "Include [comparison_metric:metric_4|metrics.metric_4] exactly once and do not duplicate any metric tag."
        if metric_count == 4
        else "Keep exactly three comparison_metric tags and do not emit [comparison_metric:metric_4|metrics.metric_4]."
    )
    callout_guard = (
        "Include [comparison_callout:callout_3|callouts.callout_3] exactly once and do not duplicate any callout tag."
        if callout_count == 3
        else "Keep exactly two comparison_callout tags and do not emit [comparison_callout:callout_3|callouts.callout_3]."
    )
    prompts: list[tuple[str, str]] = [
        (
            "bridge_create",
            f"Create a comparison board infographic about {case.prompt_topic} using {columns} columns, "
            f"{theme} styling, {tone} tone, {background} background, {metric_count} metric cards, "
            f"{callout_count} callouts, and {density} spacing. Return scene DSL only. {prompt_text}"
        ),
        (
            "bridge_stop",
            f"Plan a comparison board for {case.prompt_topic}. Keep the requested background and component counts, "
            f"output one [scene]...[/scene] block only, and stop after [/scene]. {prompt_text}"
        ),
        (
            "bridge_layout",
            f"Draft the structured board layout for {case.prompt_topic}. Use the requested {background} background, "
            f"{metric_count} metrics, and {callout_count} callouts. Emit only the scene program with the requested tags. {prompt_text}"
        ),
        (
            "repair_compiler_ready",
            f"Compose a compiler-ready comparison board scene for {case.prompt_topic}. "
            f"Keep theme {theme}, tone {tone}, background {background}, {metric_count} metrics, "
            f"and {callout_count} callouts. Return one [scene]...[/scene] block only. {prompt_text}"
        ),
        (
            "repair_scene_only",
            f"Return only the comparison board scene for {case.prompt_topic}. No commentary before or after the scene. "
            f"End exactly at [/scene]. Keep theme {theme}, tone {tone}, background {background}, "
            f"{metric_count} metrics, and {callout_count} callouts. {prompt_text}"
        ),
        (
            "repair_count_guard",
            f"Emit the scene program for {case.prompt_topic}. Keep exactly {metric_count} comparison metrics "
            f"and {callout_count} comparison callouts. Do not add extra metrics or extra callouts. "
            f"Stop at [/scene]. {prompt_text}"
        ),
        (
            "repair_singleton_tags",
            f"Emit the scene program for {case.prompt_topic}. Use each singleton layout tag exactly once: "
            f"[frame:panel], [density:{scene_runtime['density']}], [inset:md], [gap:md], "
            f"[header_band:header], [legend_block:legend], and [footer_note:footer]. "
            f"Do not duplicate any structural tag. Stop at [/scene]. {prompt_text}"
        ),
        (
            "repair_terminal_exact",
            f"Emit the scene program for {case.prompt_topic}. Use the tag [inset:md] exactly once. "
            f"Use the tag [gap:md] exactly once. Use the tag [footer_note:footer] exactly once and place it "
            f"immediately before the closing scene tag [/scene]. Do not duplicate those tags and do not attach "
            f"stray letters to end-of-scene control tags. {prompt_text}"
        ),
        (
            "repair_metric_guard",
            f"Emit the scene program for {case.prompt_topic}. {metric_guard} "
            f"Keep theme {theme}, tone {tone}, and background {background}. Stop at [/scene]. {prompt_text}"
        ),
        (
            "repair_callout_guard",
            f"Emit the scene program for {case.prompt_topic}. {callout_guard} "
            f"Keep theme {theme}, tone {tone}, and background {background}. Stop at [/scene]. {prompt_text}"
        ),
        (
            "repair_tag_exact",
            f"Given only these control tags, emit the exact matching comparison board scene. "
            f"Do not switch theme, tone, background, metric count, or callout count, and stop at [/scene]. {prompt_text}"
        ),
        (
            "repair_hidden_compose_like",
            f"Compose the final scene program for {case.prompt_topic}. Keep compiler-facing DSL only, "
            f"preserve the requested control tags, and return one exact [scene]...[/scene] block. {prompt_text}"
        ),
        (
            "repair_hidden_compose_anchor",
            f"Compose the scene program for a comparison board about {case.prompt_topic}. Keep only compiler-facing DSL. "
            f"Open with [scene] [canvas:wide] [layout:comparison_board] then continue the requested board and finish with [/scene]. "
            f"Preserve theme {theme}, tone {tone}, background {background}, {metric_count} metrics, and {callout_count} callouts exactly. {prompt_text}"
        ),
        (
            "repair_hidden_stop_like",
            f"Return the scene program for {case.prompt_topic}. Stop immediately after [/scene], "
            f"do not emit another board, and preserve the requested control tags exactly. {prompt_text}"
        ),
    ]
    if _is_default_board(scene_runtime):
        prompts.append(
            (
                "contrast_keep_default",
                f"Emit the default amber rings comparison board for {case.prompt_topic}. "
                f"Keep theme infra dark, tone amber, background rings, 3 metrics, and 2 callouts. "
                f"Do not switch theme or background, and do not add metric 4 or callout 3. "
                f"Stop at [/scene]. {prompt_text}"
            )
        )
    else:
        prompts.append(
            (
                "bridge_anti_default",
                f"Plan a comparison board for {case.prompt_topic}. Do not fall back to the default amber rings board. "
                f"Keep theme {theme}, tone {tone}, background {background}, {metric_count} metrics, "
                f"and {callout_count} callouts. Stop after [/scene]. {prompt_text}"
            )
        )
        prompts.append(
            (
                "contrast_from_default",
                f"Start from the default amber rings comparison board for {case.prompt_topic}, "
                f"but apply the requested control deltas only: theme {theme}, tone {tone}, background {background}, "
                f"{metric_count} metrics, and {callout_count} callouts. "
                f"Return only the final [scene]...[/scene] program. {prompt_text}"
            )
        )
    return prompts


def _hidden_prompt_variants(case: BoardCase, prompt_text: str) -> list[tuple[str, str]]:
    return [
        (
            "hidden_compose",
            f"Compose the scene program for a comparison board about {case.prompt_topic}. Keep only compiler-facing DSL. {prompt_text}",
        ),
        (
            "hidden_stop",
            f"Return the board scene for {case.prompt_topic} and stop exactly at [/scene]. {prompt_text}",
        ),
    ]


def _compose_prompt_variants(
    case: BoardCase,
    prompt_text: str,
    scene_runtime: dict[str, Any],
) -> list[tuple[str, str]]:
    theme = str(scene_runtime["theme"]).replace("_", " ")
    tone = str(scene_runtime["tone"])
    background = str(scene_runtime.get("background") or "grid")
    metric_count = len(scene_runtime.get("components_by_name", {}).get("comparison_metric", []))
    callout_count = len(scene_runtime.get("components_by_name", {}).get("comparison_callout", []))
    return [
        (
            "compose_control_blend",
            f"Compose a comparison board for {case.prompt_topic} using theme {theme}, tone {tone}, "
            f"background {background}, {metric_count} metrics, and {callout_count} callouts. "
            f"Return scene DSL only. {prompt_text}"
        ),
        (
            "compose_scene_only",
            f"Return only the final [scene]...[/scene] program for {case.prompt_topic}. "
            f"Keep theme {theme}, tone {tone}, background {background}, "
            f"{metric_count} metrics, and {callout_count} callouts. Stop exactly at [/scene]. {prompt_text}"
        ),
        (
            "compose_anti_anchor",
            f"Build the board for {case.prompt_topic} with the requested control bundle only. "
            f"Do not fall back to the green grid board or the paper blue mesh board. "
            f"Keep theme {theme}, tone {tone}, background {background}, "
            f"{metric_count} metrics, and {callout_count} callouts. Stop at [/scene]. {prompt_text}"
        ),
        (
            "compose_hidden_compose_like",
            f"Compose the final scene program for {case.prompt_topic}. Keep compiler-facing DSL only, "
            f"preserve theme {theme}, tone {tone}, background {background}, "
            f"{metric_count} metrics, and {callout_count} callouts, and return one exact [scene]...[/scene] block. {prompt_text}"
        ),
        (
            "compose_hidden_stop_like",
            f"Return the scene program for {case.prompt_topic}. Stop immediately after [/scene], "
            f"do not emit another board, and preserve theme {theme}, tone {tone}, background {background}, "
            f"{metric_count} metrics, and {callout_count} callouts exactly. {prompt_text}"
        ),
        (
            "compose_terminal_exact",
            f"Emit the scene program for {case.prompt_topic}. Use the tag [inset:md] exactly once. "
            f"Use the tag [gap:md] exactly once. Use the tag [footer_note:footer] exactly once and place it "
            f"immediately before the closing scene tag [/scene]. Do not duplicate those tags and do not attach "
            f"stray letters to end-of-scene control tags. {prompt_text}"
        ),
    ]


def _is_holdout(case: BoardCase) -> bool:
    return str(case.split or "train") == "holdout"


def _write_probe_report_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_probe_report_contract.json"
    payload = {
        "schema": "ck.probe_report_contract.v1",
        "title": "Spec14a Comparison-Board Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec14a.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 512,
            "temperature": 0.0,
            "stop_on_text": [
                "Create a comparison board",
                "Plan a comparison board",
                "Draft the structured board",
                "Compose the scene program",
                "Return the board scene",
            ],
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
            {"name": "train", "label": "Train prompts", "format": "lines", "path": f"{prefix}_seen_prompts.txt", "limit": 16},
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
        "goal": "Learn a generalized comparison-board scene DSL with deterministic compiler-backed rendering.",
        "notes": [
            "This line introduces one new family only: comparison_board.",
            "The output contract stays symbolic and compiler-backed.",
            "Current spec14a keeps asset identity in prompt text, case_id, and content_json; the scene DSL stays family-generic.",
            "Spec15+ should remove domain-bearing prompt text from the model-facing input as well, leaving only family-generic design/control wording.",
            "The board family should preserve carry-forward compatibility with the older compiler-backed layouts later in the line.",
        ],
        "scorer": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec14a.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 512,
            "temperature": 0.0,
            "stop_on_text": [
                "Create a comparison board",
                "Plan a comparison board",
                "Draft the structured board",
                "Compose the scene program",
                "Return the board scene",
            ],
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec14a_scene_dsl", help="Output prefix")
    ap.add_argument("--train-repeats", type=int, default=8)
    ap.add_argument("--holdout-repeats", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _cases()
    training_cases = cases + _composed_training_cases(cases)
    render_rows: list[dict[str, Any]] = []
    train_rows: list[str] = []
    holdout_rows: list[str] = []
    tokenizer_corpus: list[str] = []
    seen_prompts: list[str] = []
    holdout_prompts: list[str] = []
    hidden_seen_prompts: list[str] = []
    hidden_holdout_prompts: list[str] = []

    for case in training_cases:
        scene_runtime = canonicalize_scene_text(case.scene_text).to_runtime()
        prompt_text = _canonical_prompt(scene_runtime)
        split = "holdout" if _is_holdout(case) else str(case.split or "train")
        output_tokens = case.scene_text
        content_json = case.content_json
        svg_xml = render_structured_scene_spec14a_svg(output_tokens, content=content_json)
        metric_count = len(scene_runtime.get("components_by_name", {}).get("comparison_metric", []))
        callout_count = len(scene_runtime.get("components_by_name", {}).get("comparison_callout", []))

        rows: list[tuple[str, str, str, bool]] = [(prompt_text, "tag_canonical", split, True)]
        if split == "train_aug":
            rows.extend((prompt, prompt_surface, split, True) for prompt_surface, prompt in _compose_prompt_variants(case, prompt_text, scene_runtime))
        else:
            rows.extend((prompt, prompt_surface, split, True) for prompt_surface, prompt in _bridge_prompt_variants(case, prompt_text, scene_runtime))
            hidden_split = "probe_hidden_train" if split == "train" else "probe_hidden_holdout"
            rows.extend((prompt, prompt_surface, hidden_split, False) for prompt_surface, prompt in _hidden_prompt_variants(case, prompt_text))

        for prompt, prompt_surface, row_split, training_prompt in rows:
            render_rows.append(
                {
                    "prompt": prompt,
                    "canonical_prompt": prompt_text,
                    "output_tokens": output_tokens,
                    "content_json": dict(content_json),
                    "svg_xml": svg_xml,
                    "split": row_split,
                    "layout": scene_runtime["layout"],
                    "topic": scene_runtime["topic"],
                    "case_id": case.case_id,
                    "router_case_id": case.case_id,
                    "prompt_topic": case.prompt_topic,
                    "theme": scene_runtime["theme"],
                    "tone": scene_runtime["tone"],
                    "background": scene_runtime["background"],
                    "metric_count": metric_count,
                    "callout_count": callout_count,
                    "density": scene_runtime["density"],
                    "source_asset": case.source_asset,
                    "prompt_surface": prompt_surface,
                    "training_prompt": bool(training_prompt),
                }
            )
            row_text = f"{prompt} {output_tokens}".strip()
            tokenizer_corpus.extend([row_text, prompt])
            if row_split in {"train", "train_aug"}:
                if row_split == "train":
                    seen_prompts.append(prompt)
                repeats = int(args.train_repeats) if prompt_surface == "tag_canonical" else 1
                for _ in range(max(1, repeats)):
                    train_rows.append(row_text)
            elif row_split == "holdout":
                holdout_prompts.append(prompt)
                repeats = int(args.holdout_repeats) if prompt_surface == "tag_canonical" else 1
                for _ in range(max(1, repeats)):
                    holdout_rows.append(row_text)
            elif row_split == "probe_hidden_train":
                hidden_seen_prompts.append(prompt)
            elif row_split == "probe_hidden_holdout":
                hidden_holdout_prompts.append(prompt)

    def dedupe(rows: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for row in rows:
            text = str(row or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    seen_prompts = dedupe(seen_prompts)
    holdout_prompts = dedupe(holdout_prompts)
    hidden_seen_prompts = dedupe(hidden_seen_prompts)
    hidden_holdout_prompts = dedupe(hidden_holdout_prompts)
    tokenizer_corpus = dedupe(tokenizer_corpus)

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
        "line_name": "spec14a_scene_dsl",
        "prefix": args.prefix,
        "family_mode": "comparison_board",
        "out_dir": str(out_dir),
        "layouts": ["comparison_board"],
        "case_ids": [case.case_id for case in cases],
        "source_assets": {case.case_id: case.source_asset for case in cases},
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
            "eval_contract": str(eval_contract_path)
        },
        "counts": {
            "cases": len(cases),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "train_prompts": len(seen_prompts),
            "holdout_prompts": len(holdout_prompts),
            "hidden_seen_prompts": len(hidden_seen_prompts),
            "hidden_holdout_prompts": len(hidden_holdout_prompts)
        }
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
