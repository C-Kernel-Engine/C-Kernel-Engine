#!/usr/bin/env python3
"""Generate a strict domain-agnostic memory-map spec15a dataset."""

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
from render_svg_structured_scene_spec15a_v7 import render_structured_scene_spec15a_svg
from spec15a_scene_canonicalizer_v7 import SceneComponent, SceneDocument, canonicalize_scene_text


ROOT = Path(__file__).resolve().parents[3]
GOLD = ROOT / "version" / "v7" / "reports" / "spec15a_gold_mappings"


@dataclass(frozen=True)
class MemoryCase:
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


def _cases() -> list[MemoryCase]:
    return [
        MemoryCase(
            case_id="memory_layout_map",
            form_token="layer_stack",
            form_phrase="a stacked layer tower",
            source_asset="memory-layout-map.svg",
            scene_path=GOLD / "memory-layout-map.scene.compact.dsl",
            content_path=GOLD / "memory-layout-map.content.json",
        ),
        MemoryCase(
            case_id="bump_allocator_quant",
            form_token="typed_regions",
            form_phrase="typed allocator regions",
            source_asset="bump_allocator_quant.svg",
            scene_path=GOLD / "bump_allocator_quant.scene.compact.dsl",
            content_path=GOLD / "bump_allocator_quant.content.json",
        ),
        MemoryCase(
            case_id="v7_train_memory_canary",
            form_token="arena_sections",
            form_phrase="an operator-facing arena with guarded sections",
            source_asset="v7-train-memory-canary.svg",
            scene_path=GOLD / "v7-train-memory-canary.scene.compact.dsl",
            content_path=GOLD / "v7-train-memory-canary.content.json",
        ),
    ]


def _component_token(component: SceneComponent) -> str:
    fields = component.fields
    if component.name in {"header_band", "address_strip"}:
        payload = str(fields.get("ref") or "").strip()
    elif component.name == "memory_segment":
        payload = "|".join(
            [
                str(fields.get("segment_id") or "").strip(),
                str(fields.get("ref") or "").strip(),
            ]
        )
    elif component.name == "region_bracket":
        bracket_id = str(fields.get("bracket_id") or "").strip()
        ref = str(fields.get("ref") or "").strip()
        if not bracket_id and ref:
            bracket_id = ref.split(".")[-1]
        payload = "|".join([part for part in (bracket_id, ref) if part])
    elif component.name == "info_card":
        payload = "|".join(
            [
                str(fields.get("card_id") or "").strip(),
                str(fields.get("ref") or "").strip(),
            ]
        )
    else:
        raise ValueError(f"unsupported component for spec15a scene synthesis: {component.name}")
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
    base: MemoryCase,
    *,
    theme: str,
    tone: str,
    density: str,
    background: str,
    split: str,
) -> MemoryCase:
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
    return MemoryCase(
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


def _train_aug_cases(cases: list[MemoryCase]) -> list[MemoryCase]:
    by_id = {case.case_id: case for case in cases}
    specs = [
        # Cross-form bundle anchors for the held-out style bundles.
        # Teach the exact bundle semantics on the *other* forms so the model
        # must compose style + form instead of memorizing one held-out pair.
        # layer_stack holdout bundle: signal_glow / green / compact / mesh
        ("bump_allocator_quant", "signal_glow", "green", "compact", "mesh"),
        ("v7_train_memory_canary", "signal_glow", "green", "compact", "mesh"),
        # typed_regions holdout bundle: paper_editorial / amber / balanced / grid
        ("memory_layout_map", "paper_editorial", "amber", "balanced", "grid"),
        ("v7_train_memory_canary", "paper_editorial", "amber", "balanced", "grid"),
        # arena_sections holdout bundle: signal_glow / blue / compact / rings
        ("memory_layout_map", "signal_glow", "blue", "compact", "rings"),
        ("bump_allocator_quant", "signal_glow", "blue", "compact", "rings"),
    ]
    out: list[MemoryCase] = []
    for case_id, theme, tone, density, background in specs:
        base = by_id.get(case_id)
        if base is None:
            continue
        out.append(_variant_case(base, theme=theme, tone=tone, density=density, background=background, split="train_aug"))
    return out


def _holdout_cases(cases: list[MemoryCase]) -> list[MemoryCase]:
    by_id = {case.case_id: case for case in cases}
    specs = {
        "memory_layout_map": ("signal_glow", "green", "compact", "mesh"),
        "bump_allocator_quant": ("paper_editorial", "amber", "balanced", "grid"),
        "v7_train_memory_canary": ("signal_glow", "blue", "compact", "rings"),
    }
    out: list[MemoryCase] = []
    for case_id, (theme, tone, density, background) in specs.items():
        base = by_id.get(case_id)
        if base is None:
            continue
        out.append(_variant_case(base, theme=theme, tone=tone, density=density, background=background, split="holdout"))
    return out


def _structure_counts(scene_runtime: dict[str, Any]) -> tuple[int, int, int]:
    segments = len(scene_runtime.get("components_by_name", {}).get("memory_segment", []))
    brackets = len(scene_runtime.get("components_by_name", {}).get("region_bracket", []))
    cards = len(scene_runtime.get("components_by_name", {}).get("info_card", []))
    return segments, brackets, cards


def _canonical_prompt(case: MemoryCase, scene_runtime: dict[str, Any]) -> str:
    segments, brackets, cards = _structure_counts(scene_runtime)
    return " ".join(
        [
            "[task:svg]",
            "[layout:memory_map]",
            f"[form:{case.form_token}]",
            f"[theme:{scene_runtime['theme']}]",
            f"[tone:{scene_runtime['tone']}]",
            f"[density:{scene_runtime['density']}]",
            f"[background:{scene_runtime['background']}]",
            f"[segments:{segments}]",
            f"[brackets:{brackets}]",
            f"[cards:{cards}]",
            "[OUT]",
        ]
    )


def _bridge_prompt_variants(case: MemoryCase, prompt_text: str, scene_runtime: dict[str, Any]) -> list[tuple[str, str]]:
    theme = str(scene_runtime["theme"]).replace("_", " ")
    tone = str(scene_runtime["tone"]).replace("_", " ")
    density = str(scene_runtime["density"]).replace("_", " ")
    background = str(scene_runtime["background"]).replace("_", " ")
    segments, brackets, cards = _structure_counts(scene_runtime)
    bracket_phrase = "no grouped bracket" if brackets == 0 else f"{brackets} grouped bracket"
    return [
        (
            "bridge_create",
            f"Create a memory map infographic with {case.form_phrase}, {segments} ordered regions, {bracket_phrase}, and {cards} side cards. Use {theme} styling, {tone} tone, {density} spacing, and {background} background. Return scene DSL only. {prompt_text}",
        ),
        (
            "bridge_scene_only",
            f"Return only one [scene]...[/scene] memory map with {case.form_phrase}, {segments} ordered regions, {bracket_phrase}, and {cards} side cards. Keep {theme} styling, {tone} tone, {density} spacing, and {background} background. {prompt_text}",
        ),
        (
            "bridge_count_guard",
            f"Emit the memory map scene program with exactly {segments} memory segments, {brackets} grouped bracket tags, and {cards} info cards. Do not add extra structural tags and stop at [/scene]. {prompt_text}",
        ),
        (
            "bridge_form_focus",
            f"Plan the scene as a memory map using the {case.form_phrase} form. Preserve the requested theme, tone, density, background, and structural counts. Emit compiler-facing scene DSL only. {prompt_text}",
        ),
        (
            "bridge_terminal_exact",
            f"Emit the memory map scene program. Open with [scene], preserve the requested control tags exactly, and end immediately at [/scene] with no extra text. {prompt_text}",
        ),
        (
            "repair_strip_prompt_tags",
            f"Treat the form, count, task, and output control markers as input-only guidance. Do not copy prompt-only control tags into the answer. Emit scene DSL tags only and stop at [/scene]. {prompt_text}",
        ),
        (
            "repair_singletons",
            f"Emit exactly one [layout:memory_map], one [header_band:header], and one [address_strip:offsets]. Do not duplicate singleton tags. Return one clean [scene]...[/scene]. {prompt_text}",
        ),
        (
            "repair_style_bundle",
            f"Preserve the requested theme, tone, density, and background exactly. Do not switch to a nearby seen style bundle. Return scene DSL only. {prompt_text}",
        ),
        (
            "repair_no_prompt_echo",
            f"After the closing [/scene], emit nothing else. Do not echo the prompt controls back into the answer. Return one clean scene only. {prompt_text}",
        ),
    ]


def _hidden_prompt_variants(case: MemoryCase, prompt_text: str, scene_runtime: dict[str, Any]) -> list[tuple[str, str]]:
    segments, brackets, cards = _structure_counts(scene_runtime)
    return [
        (
            "hidden_compose",
            f"Compose the scene program for a memory map with {case.form_phrase}, {segments} ordered regions, {brackets} grouped brackets, and {cards} side cards. Keep compiler-facing DSL only. {prompt_text}",
        ),
        (
            "hidden_stop",
            f"Return the memory map scene for {case.form_phrase} and stop exactly at [/scene]. Preserve the requested structure and style controls. {prompt_text}",
        ),
    ]


def _train_hidden_prompt_variants(case: MemoryCase, prompt_text: str, scene_runtime: dict[str, Any]) -> list[tuple[str, str]]:
    variants = [
        (f"train_{name}", prompt)
        for name, prompt in _hidden_prompt_variants(case, prompt_text, scene_runtime)
    ]
    variants.append(
        (
            "train_hidden_style_bundle",
            f"Compose one memory map scene for {case.form_phrase}. Preserve the requested theme, tone, density, and background exactly, keep the requested structure, and stop at [/scene]. {prompt_text}",
        )
    )
    if case.form_token == "layer_stack":
        variants.append(
            (
                "train_hidden_component_binding",
                "Preserve the exact layer_stack component ids and ordering. "
                "After [region_bracket:mid_layers|segments.mid_layers], keep "
                "[memory_segment:layer_23|segments.layer_23] and do not replace it "
                f"with scratch or any other segment. Return scene DSL only. {prompt_text}",
            )
        )
    return variants


def _write_probe_report_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_probe_report_contract.json"
    payload = {
        "schema": "ck.probe_report_contract.v1",
        "title": "Spec15a Memory-Map Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec15a.v1",
            "repairer": "spec15a_scene_bundle.v1",
            "preview_mime": "image/svg+xml"
        },
        "decode": {
            "max_tokens": 384,
            "temperature": 0.0,
            "stop_on_text": [
                "Create a memory map infographic",
                "Return only one [scene]",
                "Emit the memory map scene program",
                "Compose the scene program for a memory map"
            ]
        },
        "catalog": {
            "format": "json_rows",
            "path": f"{prefix}_render_catalog.json",
            "prompt_key": "prompt",
            "output_key": "output_tokens",
            "rendered_key": "svg_xml",
            "rendered_mime": "image/svg+xml",
            "split_key": "split"
        },
        "splits": [
            {"name": "train", "label": "Train prompts", "format": "lines", "path": f"{prefix}_seen_prompts.txt", "limit": 12},
            {"name": "test", "label": "Holdout prompts", "format": "lines", "path": f"{prefix}_holdout_prompts.txt", "limit": 12}
        ]
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _write_eval_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_eval_contract.json"
    payload = {
        "schema": "ck.eval_contract.v1",
        "dataset_type": "svg",
        "goal": "Learn a strict domain-agnostic memory_map scene DSL with compiler-backed rendering.",
        "notes": [
            "Model-facing prompts stay family-generic and contain only design/control language.",
            "Visible text and measurements remain external in content.json.",
            "The output contract is compact scene DSL only."
        ],
        "scorer": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec15a.v1",
            "repairer": "spec15a_scene_bundle.v1",
            "preview_mime": "image/svg+xml"
        },
        "decode": {
            "max_tokens": 384,
            "temperature": 0.0,
            "stop_on_text": [
                "Create a memory map infographic",
                "Return only one [scene]",
                "Emit the memory map scene program",
                "Compose the scene program for a memory map"
            ]
        }
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec15a_scene_dsl", help="Output prefix")
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
        svg_xml = render_structured_scene_spec15a_svg(output_tokens, content=content_json)
        segments, brackets, cards = _structure_counts(scene_runtime)

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
                    "segment_count": segments,
                    "bracket_count": brackets,
                    "card_count": cards,
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
        "line_name": "spec15a_scene_dsl",
        "prefix": args.prefix,
        "family_mode": "memory_map",
        "out_dir": str(out_dir),
        "layouts": ["memory_map"],
        "case_ids": [case.case_id for case in base_cases],
        "source_assets": {case.case_id: case.source_asset for case in base_cases},
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
            "cases": len(base_cases),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "train_prompts": len(seen_prompts),
            "holdout_prompts": len(holdout_prompts),
            "hidden_seen_prompts": len(hidden_seen_prompts),
            "hidden_holdout_prompts": len(hidden_holdout_prompts),
        },
        "notes": [
            "Prompts are family-generic and contain only design/control language.",
            "Payload facts remain external in content_json.",
            "Compact scene DSL is the only model-side output contract.",
        ],
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
