#!/usr/bin/env python3
"""Generate a strict domain-agnostic timeline spec14b dataset."""

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
from render_svg_structured_scene_spec14b_v7 import render_structured_scene_spec14b_svg
from spec14b_scene_canonicalizer_v7 import SceneComponent, SceneDocument, canonicalize_scene_text


ROOT = Path(__file__).resolve().parents[3]
GOLD = ROOT / "version" / "v7" / "reports" / "spec14b_gold_mappings"


@dataclass(frozen=True)
class TimelineCase:
    case_id: str
    form_token: str
    form_phrase: str
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


def _cases() -> list[TimelineCase]:
    return [
        TimelineCase(
            case_id="ir_v66_evolution_timeline",
            form_token="milestone_chain",
            form_phrase="a dated milestone chain",
            source_asset="ir-v66-evolution-timeline.svg",
            scene_path=GOLD / "ir-v66-evolution-timeline.scene.compact.dsl",
            content_path=GOLD / "ir-v66-evolution-timeline.content.json",
        ),
        TimelineCase(
            case_id="ir_timeline_why",
            form_token="stage_sequence",
            form_phrase="a compact stage sequence",
            source_asset="ir-timeline-why.svg",
            scene_path=GOLD / "ir-timeline-why.scene.compact.dsl",
            content_path=GOLD / "ir-timeline-why.content.json",
        ),
    ]


def _component_token(component: SceneComponent) -> str:
    fields = component.fields
    if component.name in {"header_band", "footer_note"}:
        payload = str(fields.get("ref") or "").strip()
    elif component.name == "timeline_stage":
        payload = "|".join(
            [
                str(fields.get("stage_id") or "").strip(),
                str(fields.get("ref") or "").strip(),
                f"lane={str(fields.get('lane') or 'center').strip()}",
            ]
        )
    elif component.name == "timeline_arrow":
        payload = f"{str(fields.get('from_stage') or '').strip()}->{str(fields.get('to_stage') or '').strip()}"
    else:
        raise ValueError(f"unsupported component for spec14b scene synthesis: {component.name}")
    return f"[{component.name}:{payload}]"


def _scene_text_from_document(scene: SceneDocument) -> str:
    tokens = [
        "[scene]",
        f"[layout:{scene.layout}]",
        f"[theme:{scene.theme}]",
        f"[tone:{scene.tone}]",
        f"[density:{scene.density}]",
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
    density: str,
    background: str,
) -> SceneDocument:
    return SceneDocument(
        canvas=scene.canvas,
        layout=scene.layout,
        theme=theme,
        tone=tone,
        frame=scene.frame,
        density=density,
        inset=scene.inset,
        gap=scene.gap,
        hero=scene.hero,
        columns=scene.columns,
        emphasis=scene.emphasis,
        rail=scene.rail,
        background=background,
        connector=scene.connector,
        topic=scene.topic,
        components=scene.components,
    )


def _variant_case(
    base: TimelineCase,
    *,
    theme: str,
    tone: str,
    density: str,
    background: str,
    split: str,
) -> TimelineCase:
    base_scene = canonicalize_scene_text(base.scene_text)
    scene_text = _scene_text_from_document(
        _remix_scene_document(
            base_scene,
            theme=theme,
            tone=tone,
            density=density,
            background=background,
        )
    )
    return TimelineCase(
        case_id=f"{base.case_id}_{split}_{theme}_{tone}_{density}_{background}",
        form_token=base.form_token,
        form_phrase=base.form_phrase,
        source_asset=f"{base.source_asset}#{split}:{theme}:{tone}:{density}:{background}",
        scene_path=base.scene_path,
        content_path=base.content_path,
        split=split,
        scene_text_override=scene_text,
        content_json_override=base.content_json,
    )


def _train_aug_cases(cases: list[TimelineCase]) -> list[TimelineCase]:
    by_id = {case.case_id: case for case in cases}
    specs = [
        ("ir_v66_evolution_timeline", "paper_editorial", "amber", "balanced", "grid"),
        ("ir_timeline_why", "signal_glow", "blue", "compact", "rings"),
        # Full style swaps across forms to break form->style coupling.
        ("ir_v66_evolution_timeline", "infra_dark", "amber", "compact", "none"),
        ("ir_timeline_why", "signal_glow", "blue", "balanced", "mesh"),
        # Single-axis rehearsals for milestone_chain.
        ("ir_v66_evolution_timeline", "signal_glow", "green", "balanced", "mesh"),
        ("ir_v66_evolution_timeline", "signal_glow", "blue", "compact", "mesh"),
        ("ir_v66_evolution_timeline", "signal_glow", "blue", "balanced", "rings"),
        # Single-axis rehearsals for stage_sequence.
        ("ir_timeline_why", "paper_editorial", "amber", "compact", "none"),
        ("ir_timeline_why", "infra_dark", "blue", "compact", "none"),
        ("ir_timeline_why", "infra_dark", "amber", "airy", "none"),
        ("ir_timeline_why", "infra_dark", "amber", "compact", "grid"),
    ]
    out: list[TimelineCase] = []
    for case_id, theme, tone, density, background in specs:
        base = by_id.get(case_id)
        if base is None:
            continue
        out.append(_variant_case(base, theme=theme, tone=tone, density=density, background=background, split="train_aug"))
    return out


def _holdout_cases(cases: list[TimelineCase]) -> list[TimelineCase]:
    by_id = {case.case_id: case for case in cases}
    specs = {
        "ir_v66_evolution_timeline": ("signal_glow", "green", "compact", "rings"),
        "ir_timeline_why": ("paper_editorial", "blue", "airy", "grid"),
    }
    out: list[TimelineCase] = []
    for case_id, (theme, tone, density, background) in specs.items():
        base = by_id.get(case_id)
        if base is None:
            continue
        out.append(_variant_case(base, theme=theme, tone=tone, density=density, background=background, split="holdout"))
    return out


def _structure_counts(scene_runtime: dict[str, Any]) -> tuple[int, int, int]:
    stages = len(scene_runtime.get("components_by_name", {}).get("timeline_stage", []))
    arrows = len(scene_runtime.get("components_by_name", {}).get("timeline_arrow", []))
    footer = len(scene_runtime.get("components_by_name", {}).get("footer_note", []))
    return stages, arrows, footer


def _canonical_prompt(case: TimelineCase, scene_runtime: dict[str, Any]) -> str:
    stages, arrows, footer = _structure_counts(scene_runtime)
    return " ".join(
        [
            "[task:svg]",
            "[layout:timeline]",
            f"[form:{case.form_token}]",
            f"[theme:{scene_runtime['theme']}]",
            f"[tone:{scene_runtime['tone']}]",
            f"[density:{scene_runtime['density']}]",
            f"[background:{scene_runtime['background']}]",
            f"[stages:{stages}]",
            f"[arrows:{arrows}]",
            f"[footer:{footer}]",
            "[OUT]",
        ]
    )


def _bridge_prompt_variants(case: TimelineCase, prompt_text: str, scene_runtime: dict[str, Any]) -> list[tuple[str, str]]:
    theme = str(scene_runtime["theme"]).replace("_", " ")
    tone = str(scene_runtime["tone"]).replace("_", " ")
    density = str(scene_runtime["density"]).replace("_", " ")
    background = str(scene_runtime["background"]).replace("_", " ")
    stages, arrows, footer = _structure_counts(scene_runtime)
    footer_phrase = "with a footer note" if footer else "with no footer note"
    return [
        (
            "bridge_create",
            f"Create a timeline infographic with {case.form_phrase}, {stages} ordered stages, {arrows} arrows, and {footer_phrase}. Use {theme} styling, {tone} tone, {density} spacing, and {background} background. Return scene DSL only. {prompt_text}",
        ),
        (
            "bridge_scene_only",
            f"Return only one [scene]...[/scene] timeline with {case.form_phrase}, {stages} stages, {arrows} arrows, and {footer_phrase}. Keep {theme} styling, {tone} tone, {density} spacing, and {background} background. {prompt_text}",
        ),
        (
            "bridge_count_guard",
            f"Emit the timeline scene program with exactly {stages} timeline_stage tags, exactly {arrows} timeline_arrow tags, and footer count {footer}. Do not add extra structural tags and stop at [/scene]. {prompt_text}",
        ),
        (
            "bridge_form_focus",
            f"Plan the scene as a timeline using the {case.form_phrase} form. Preserve the requested theme, tone, density, background, and structural counts. Emit compiler-facing scene DSL only. {prompt_text}",
        ),
        (
            "bridge_terminal_exact",
            f"Emit the timeline scene program. Open with [scene], preserve the requested control tags exactly, and end immediately at [/scene] with no extra text. {prompt_text}",
        ),
        (
            "repair_style_bundle",
            f"Preserve the requested theme, tone, density, and background exactly. Do not switch to a nearby seen style bundle. Return scene DSL only. {prompt_text}",
        ),
        (
            "repair_background_lock",
            f"Keep the requested background token exactly. [background:mesh] must stay mesh, [background:rings] must stay rings, [background:grid] must stay grid, and [background:none] must stay none. Return one clean scene only. {prompt_text}",
        ),
        (
            "repair_density_lock",
            f"Keep the requested density token exactly. [density:balanced], [density:compact], and [density:airy] are not interchangeable. Return one clean scene only. {prompt_text}",
        ),
        (
            "repair_stage_uniqueness",
            f"Emit each timeline_stage exactly once in order and do not repeat any stage id. Keep the requested arrow count and stop at [/scene]. {prompt_text}",
        ),
        (
            "repair_stop_boundary",
            f"Emit exactly one timeline scene and end immediately after [/scene]. Do not continue into another prompt, another [scene], or any extra text. {prompt_text}",
        ),
        (
            "repair_no_prompt_echo",
            f"After the closing [/scene], emit nothing else. Do not echo the prompt controls back into the answer. Return one clean scene only. {prompt_text}",
        ),
    ]


def _hidden_prompt_variants(case: TimelineCase, prompt_text: str, scene_runtime: dict[str, Any]) -> list[tuple[str, str]]:
    stages, arrows, footer = _structure_counts(scene_runtime)
    footer_phrase = "with the requested footer setting" if footer else "without adding a footer"
    return [
        (
            "hidden_compose",
            f"Compose the scene program for a timeline with {case.form_phrase}, {stages} ordered stages, {arrows} directional links, and {footer_phrase}. Keep compiler-facing DSL only. {prompt_text}",
        ),
        (
            "hidden_stop",
            f"Return the timeline scene for {case.form_phrase} and stop exactly at [/scene]. Preserve the requested structure and style controls. {prompt_text}",
        ),
    ]


def _train_hidden_prompt_variants(case: TimelineCase, prompt_text: str, scene_runtime: dict[str, Any]) -> list[tuple[str, str]]:
    variants = [
        (f"train_{name}", prompt)
        for name, prompt in _hidden_prompt_variants(case, prompt_text, scene_runtime)
    ]
    variants.append(
        (
            "train_hidden_style_bundle",
            f"Compose one timeline scene for {case.form_phrase}. Preserve the requested theme, tone, density, and background exactly, keep the requested stage and arrow counts, and stop at [/scene]. {prompt_text}",
        )
    )
    return variants


def _write_probe_report_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_probe_report_contract.json"
    payload = {
        "schema": "ck.probe_report_contract.v1",
        "title": "Spec14b Timeline Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec14b.v1",
            "repairer": "spec14b_scene_bundle.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 384,
            "temperature": 0.0,
            "stop_on_text": [
                "Create a timeline infographic",
                "Return only one [scene]",
                "Emit the timeline scene program",
                "Compose the scene program for a timeline",
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
            {"name": "train", "label": "Train prompts", "format": "lines", "path": f"{prefix}_seen_prompts.txt", "limit": 8},
            {"name": "test", "label": "Holdout prompts", "format": "lines", "path": f"{prefix}_holdout_prompts.txt", "limit": 8},
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _write_eval_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_eval_contract.json"
    payload = {
        "schema": "ck.eval_contract.v1",
        "dataset_type": "svg",
        "goal": "Learn a strict domain-agnostic timeline scene DSL with compiler-backed rendering.",
        "notes": [
            "Model-facing prompts stay family-generic and contain only design/control language.",
            "Visible text and dated milestone copy remain external in content.json.",
            "The output contract is compact scene DSL only.",
        ],
        "scorer": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec14b.v1",
            "repairer": "spec14b_scene_bundle.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 384,
            "temperature": 0.0,
            "stop_on_text": [
                "Create a timeline infographic",
                "Return only one [scene]",
                "Emit the timeline scene program",
                "Compose the scene program for a timeline",
            ],
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec14b_scene_dsl", help="Output prefix")
    ap.add_argument("--train-repeats", type=int, default=6)
    ap.add_argument("--holdout-repeats", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cases = _cases()
    all_cases = base_cases + _train_aug_cases(base_cases) + _holdout_cases(base_cases)
    render_rows: list[dict[str, Any]] = []
    train_rows: list[str] = []
    holdout_rows: list[str] = []
    tokenizer_corpus: list[str] = []
    seen_prompts: list[str] = []
    holdout_prompts: list[str] = []
    hidden_seen_prompts: list[str] = []
    hidden_holdout_prompts: list[str] = []

    for case in all_cases:
        scene_runtime = canonicalize_scene_text(case.scene_text).to_runtime()
        prompt_text = _canonical_prompt(case, scene_runtime)
        split = str(case.split or "train")
        output_tokens = case.scene_text
        content_json = case.content_json
        svg_xml = render_structured_scene_spec14b_svg(output_tokens, content=content_json)
        stages, arrows, footer = _structure_counts(scene_runtime)

        rows: list[tuple[str, str, str, bool]] = [(prompt_text, "tag_canonical", split, True)]
        rows.extend((prompt, prompt_surface, split, True) for prompt_surface, prompt in _bridge_prompt_variants(case, prompt_text, scene_runtime))
        rows.extend((prompt, prompt_surface, split, True) for prompt_surface, prompt in _train_hidden_prompt_variants(case, prompt_text, scene_runtime))
        hidden_split = "probe_hidden_train" if split in {"train", "train_aug"} else "probe_hidden_holdout"
        rows.extend((prompt, prompt_surface, hidden_split, False) for prompt_surface, prompt in _hidden_prompt_variants(case, prompt_text, scene_runtime))

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
                    "form_token": case.form_token,
                    "theme": scene_runtime["theme"],
                    "tone": scene_runtime["tone"],
                    "density": scene_runtime["density"],
                    "background": scene_runtime["background"],
                    "stage_count": stages,
                    "arrow_count": arrows,
                    "footer_count": footer,
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

    (out_dir / f"{args.prefix}_render_catalog.json").write_text(json.dumps(render_rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (out_dir / f"{args.prefix}_vocab.json").write_text(json.dumps(tokenizer, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_probe_report_contract(out_dir, args.prefix)
    _write_eval_contract(out_dir, args.prefix)

    manifest = {
        "schema": "ck.spec14b.generated_dataset.v1",
        "prefix": args.prefix,
        "family": "timeline",
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "seen_prompts": len(seen_prompts),
        "holdout_prompts": len(holdout_prompts),
        "hidden_seen_prompts": len(hidden_seen_prompts),
        "hidden_holdout_prompts": len(hidden_holdout_prompts),
        "tokenizer_json": str(tokenizer_json),
        "tokenizer_bin": str(tokenizer_bin),
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
