#!/usr/bin/env python3
"""Generate the first generalized spec16 visual-scene-bundle dataset."""

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
from render_svg_structured_scene_spec16_v7 import render_structured_scene_spec16_svg
from spec14b_scene_canonicalizer_v7 import canonicalize_scene_text as canonicalize_timeline_scene_text
from spec15a_scene_canonicalizer_v7 import canonicalize_scene_text as canonicalize_memory_scene_text
from spec15b_scene_canonicalizer_v7 import canonicalize_scene_text as canonicalize_system_scene_text
from spec16_scene_bundle_canonicalizer_v7 import serialize_scene_bundle
from spec16_scene_bundle_v7 import SceneBundle, canonicalize_scene_bundle, family_specs


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "version" / "v7" / "reports"
GOLD_15A = REPORTS / "spec15a_gold_mappings"
GOLD_14B = REPORTS / "spec14b_gold_mappings"
GOLD_15B = REPORTS / "spec15b_gold_mappings"


@dataclass(frozen=True)
class BundleCase:
    case_id: str
    family: str
    form_token: str
    form_phrase: str
    source_asset: str
    scene_path: Path
    content_path: Path
    split: str = "train"
    bundle_override: dict[str, Any] | None = None
    content_json_override: dict[str, Any] | None = None

    @property
    def scene_text(self) -> str:
        return self.scene_path.read_text(encoding="utf-8").strip()

    @property
    def content_json(self) -> dict[str, Any]:
        if self.content_json_override is not None:
            return json.loads(json.dumps(self.content_json_override))
        return json.loads(self.content_path.read_text(encoding="utf-8"))

    @property
    def bundle(self) -> SceneBundle:
        if self.bundle_override is not None:
            return canonicalize_scene_bundle(self.bundle_override)
        return _bundle_from_scene(self.family, self.form_token, self.scene_text)


def _cases() -> list[BundleCase]:
    return [
        BundleCase(
            case_id="memory_layout_map",
            family="memory_map",
            form_token="layer_stack",
            form_phrase="a stacked layer tower",
            source_asset="memory-layout-map.svg",
            scene_path=GOLD_15A / "memory-layout-map.scene.compact.dsl",
            content_path=GOLD_15A / "memory-layout-map.content.json",
        ),
        BundleCase(
            case_id="bump_allocator_quant",
            family="memory_map",
            form_token="typed_regions",
            form_phrase="typed allocator regions",
            source_asset="bump_allocator_quant.svg",
            scene_path=GOLD_15A / "bump_allocator_quant.scene.compact.dsl",
            content_path=GOLD_15A / "bump_allocator_quant.content.json",
        ),
        BundleCase(
            case_id="v7_train_memory_canary",
            family="memory_map",
            form_token="arena_sections",
            form_phrase="an operator-facing arena with guarded sections",
            source_asset="v7-train-memory-canary.svg",
            scene_path=GOLD_15A / "v7-train-memory-canary.scene.compact.dsl",
            content_path=GOLD_15A / "v7-train-memory-canary.content.json",
        ),
        BundleCase(
            case_id="ir_v66_evolution_timeline",
            family="timeline",
            form_token="milestone_chain",
            form_phrase="a dated milestone chain",
            source_asset="ir-v66-evolution-timeline.svg",
            scene_path=GOLD_14B / "ir-v66-evolution-timeline.scene.compact.dsl",
            content_path=GOLD_14B / "ir-v66-evolution-timeline.content.json",
        ),
        BundleCase(
            case_id="ir_timeline_why",
            family="timeline",
            form_token="stage_sequence",
            form_phrase="a compact stage sequence",
            source_asset="ir-timeline-why.svg",
            scene_path=GOLD_14B / "ir-timeline-why.scene.compact.dsl",
            content_path=GOLD_14B / "ir-timeline-why.content.json",
        ),
        BundleCase(
            case_id="pipeline_overview_system",
            family="system_diagram",
            form_token="linear_pipeline",
            form_phrase="a linear stage pipeline",
            source_asset="pipeline-overview.svg",
            scene_path=GOLD_15B / "pipeline-overview-system.scene.compact.dsl",
            content_path=GOLD_15B / "pipeline-overview-system.content.json",
        ),
        BundleCase(
            case_id="ir_pipeline_flow_system",
            family="system_diagram",
            form_token="build_path",
            form_phrase="a staged build path",
            source_asset="ir-pipeline-flow.svg",
            scene_path=GOLD_15B / "ir-pipeline-flow-system.scene.compact.dsl",
            content_path=GOLD_15B / "ir-pipeline-flow-system.content.json",
        ),
        BundleCase(
            case_id="kernel_registry_flow_system",
            family="system_diagram",
            form_token="selection_path",
            form_phrase="a staged selection path",
            source_asset="kernel-registry-flow.svg",
            scene_path=GOLD_15B / "kernel-registry-flow-system.scene.compact.dsl",
            content_path=GOLD_15B / "kernel-registry-flow-system.content.json",
        ),
    ]


def _bundle_from_scene(family: str, form_token: str, scene_text: str) -> SceneBundle:
    family = str(family or "").strip()
    if family == "memory_map":
        runtime = canonicalize_memory_scene_text(scene_text).to_runtime()
        topology = {
            "segments": len(runtime.get("components_by_name", {}).get("memory_segment", [])),
            "brackets": len(runtime.get("components_by_name", {}).get("region_bracket", [])),
            "cards": len(runtime.get("components_by_name", {}).get("info_card", [])),
        }
    elif family == "timeline":
        runtime = canonicalize_timeline_scene_text(scene_text).to_runtime()
        topology = {
            "stages": len(runtime.get("components_by_name", {}).get("timeline_stage", [])),
            "arrows": len(runtime.get("components_by_name", {}).get("timeline_arrow", [])),
            "footer": len(runtime.get("components_by_name", {}).get("footer_note", [])),
        }
    elif family == "system_diagram":
        runtime = canonicalize_system_scene_text(scene_text).to_runtime()
        topology = {
            "stages": len(runtime.get("components_by_name", {}).get("system_stage", [])),
            "links": len(runtime.get("components_by_name", {}).get("system_link", [])),
            "terminal": len(runtime.get("components_by_name", {}).get("terminal_panel", [])),
            "footer": len(runtime.get("components_by_name", {}).get("footer_note", [])),
        }
    else:
        raise ValueError(f"unsupported spec16 family: {family!r}")
    return canonicalize_scene_bundle(
        {
            "family": family,
            "form": form_token,
            "theme": runtime["theme"],
            "tone": runtime["tone"],
            "density": runtime["density"],
            "background": runtime["background"],
            "topology": topology,
        }
    )


def _variant_case(
    base: BundleCase,
    *,
    theme: str,
    tone: str,
    density: str,
    background: str,
    split: str,
) -> BundleCase:
    bundle_doc = base.bundle.to_dict()
    bundle_doc["theme"] = theme
    bundle_doc["tone"] = tone
    bundle_doc["density"] = density
    bundle_doc["background"] = background
    return BundleCase(
        case_id=f"{base.case_id}_{split}_{theme}_{tone}_{density}_{background}",
        family=base.family,
        form_token=base.form_token,
        form_phrase=base.form_phrase,
        source_asset=f"{base.source_asset}#{split}:{theme}:{tone}:{density}:{background}",
        scene_path=base.scene_path,
        content_path=base.content_path,
        split=split,
        bundle_override=bundle_doc,
        content_json_override=base.content_json,
    )


def _form_phrase_lookup(cases: list[BundleCase]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for case in cases:
        key = (case.family, case.form_token)
        lookup.setdefault(key, case.form_phrase)
    return lookup


def _bundle_variant_case(
    base: BundleCase,
    *,
    split: str,
    variant_key: str,
    form_token: str | None = None,
    form_phrase: str | None = None,
    theme: str | None = None,
    tone: str | None = None,
    density: str | None = None,
    background: str | None = None,
) -> BundleCase:
    bundle_doc = base.bundle.to_dict()
    if form_token is not None:
        bundle_doc["form"] = form_token
    if theme is not None:
        bundle_doc["theme"] = theme
    if tone is not None:
        bundle_doc["tone"] = tone
    if density is not None:
        bundle_doc["density"] = density
    if background is not None:
        bundle_doc["background"] = background
    return BundleCase(
        case_id=f"{base.case_id}_{split}_{variant_key}",
        family=base.family,
        form_token=form_token or base.form_token,
        form_phrase=form_phrase or base.form_phrase,
        source_asset=f"{base.source_asset}#{split}:{variant_key}",
        scene_path=base.scene_path,
        content_path=base.content_path,
        split=split,
        bundle_override=bundle_doc,
        content_json_override=base.content_json,
    )


def _train_aug_cases(cases: list[BundleCase]) -> list[BundleCase]:
    by_id = {case.case_id: case for case in cases}
    specs = [
        ("memory_layout_map", "signal_glow", "green", "compact", "mesh"),
        ("bump_allocator_quant", "infra_dark", "blue", "balanced", "none"),
        ("v7_train_memory_canary", "paper_editorial", "blue", "airy", "grid"),
        ("ir_v66_evolution_timeline", "signal_glow", "amber", "balanced", "mesh"),
        ("ir_timeline_why", "infra_dark", "green", "compact", "rings"),
        ("pipeline_overview_system", "signal_glow", "blue", "balanced", "mesh"),
        ("ir_pipeline_flow_system", "infra_dark", "amber", "compact", "rings"),
        ("kernel_registry_flow_system", "paper_editorial", "amber", "balanced", "grid"),
    ]
    out: list[BundleCase] = []
    for case_id, theme, tone, density, background in specs:
        base = by_id.get(case_id)
        if base is None:
            continue
        out.append(_variant_case(base, theme=theme, tone=tone, density=density, background=background, split="train_aug"))
    return out


def _style_axis_aug_cases(cases: list[BundleCase]) -> list[BundleCase]:
    by_id = {case.case_id: case for case in cases}
    specs = [
        ("memory_layout_map", "infra_dark", "purple", "compact", "none"),
        ("memory_layout_map", "signal_glow", "green", "balanced", "rings"),
        ("bump_allocator_quant", "signal_glow", "blue", "balanced", "mesh"),
        ("bump_allocator_quant", "infra_dark", "mixed", "compact", "none"),
        ("v7_train_memory_canary", "paper_editorial", "blue", "balanced", "grid"),
        ("ir_v66_evolution_timeline", "signal_glow", "blue", "balanced", "mesh"),
        ("ir_v66_evolution_timeline", "paper_editorial", "blue", "airy", "grid"),
        ("ir_timeline_why", "infra_dark", "amber", "balanced", "none"),
        ("ir_timeline_why", "signal_glow", "green", "compact", "mesh"),
        ("pipeline_overview_system", "paper_editorial", "amber", "balanced", "grid"),
        ("pipeline_overview_system", "signal_glow", "blue", "balanced", "mesh"),
        ("ir_pipeline_flow_system", "infra_dark", "amber", "compact", "rings"),
        ("ir_pipeline_flow_system", "signal_glow", "green", "balanced", "mesh"),
        ("kernel_registry_flow_system", "paper_editorial", "amber", "balanced", "grid"),
        ("kernel_registry_flow_system", "infra_dark", "blue", "balanced", "none"),
    ]
    out: list[BundleCase] = []
    for case_id, theme, tone, density, background in specs:
        base = by_id.get(case_id)
        if base is None:
            continue
        out.append(_variant_case(base, theme=theme, tone=tone, density=density, background=background, split="style_aug"))
    return out


def _form_contrast_cases(cases: list[BundleCase]) -> list[BundleCase]:
    phrase_lookup = _form_phrase_lookup(cases)
    specs = family_specs()
    out: list[BundleCase] = []
    for base in cases:
        for form_token in specs[base.family]["forms"]:
            if form_token == base.form_token:
                continue
            out.append(
                _bundle_variant_case(
                    base,
                    split="contrast_aug",
                    variant_key=f"form_{form_token}",
                    form_token=form_token,
                    form_phrase=phrase_lookup[(base.family, form_token)],
                )
            )
    return out


def _memory_style_contrast_cases(cases: list[BundleCase]) -> list[BundleCase]:
    target_styles = [
        ("infra_dark", "green", "balanced", "none"),
        ("infra_dark", "amber", "compact", "rings"),
        ("signal_glow", "blue", "compact", "mesh"),
        ("signal_glow", "amber", "balanced", "rings"),
        ("paper_editorial", "blue", "airy", "grid"),
    ]
    out: list[BundleCase] = []
    for base in cases:
        if base.family != "memory_map":
            continue
        base_bundle = base.bundle
        for theme, tone, density, background in target_styles:
            if (
                base_bundle.theme == theme
                and base_bundle.tone == tone
                and base_bundle.density == density
                and base_bundle.background == background
            ):
                continue
            variant_key = f"style_{theme}_{tone}_{density}_{background}"
            out.append(
                _bundle_variant_case(
                    base,
                    split="contrast_aug",
                    variant_key=variant_key,
                    theme=theme,
                    tone=tone,
                    density=density,
                    background=background,
                )
            )
    return out


def _memory_single_axis_style_cases(cases: list[BundleCase]) -> list[BundleCase]:
    themes = ["infra_dark", "signal_glow", "paper_editorial"]
    tones = ["green", "blue", "amber", "mixed", "purple"]
    densities = ["compact", "balanced", "airy"]
    backgrounds = ["none", "mesh", "rings", "grid"]
    out: list[BundleCase] = []
    for base in cases:
        if base.family != "memory_map":
            continue
        bundle = base.bundle
        for theme in themes:
            if theme == bundle.theme:
                continue
            out.append(
                _bundle_variant_case(
                    base,
                    split="contrast_aug",
                    variant_key=f"theme_only_{theme}",
                    theme=theme,
                )
            )
        for tone in tones:
            if tone == bundle.tone:
                continue
            out.append(
                _bundle_variant_case(
                    base,
                    split="contrast_aug",
                    variant_key=f"tone_only_{tone}",
                    tone=tone,
                )
            )
        for density in densities:
            if density == bundle.density:
                continue
            out.append(
                _bundle_variant_case(
                    base,
                    split="contrast_aug",
                    variant_key=f"density_only_{density}",
                    density=density,
                )
            )
        for background in backgrounds:
            if background == bundle.background:
                continue
            out.append(
                _bundle_variant_case(
                    base,
                    split="contrast_aug",
                    variant_key=f"background_only_{background}",
                    background=background,
                )
            )
    return out


def _memory_cross_form_frontier_cases(cases: list[BundleCase]) -> list[BundleCase]:
    phrase_lookup = _form_phrase_lookup(cases)
    frontier_specs = [
        ("typed_regions", "signal_glow", "blue", "compact", "mesh"),
        ("typed_regions", "infra_dark", "amber", "compact", "rings"),
        ("arena_sections", "signal_glow", "amber", "balanced", "rings"),
        ("arena_sections", "signal_glow", "blue", "compact", "mesh"),
        ("layer_stack", "paper_editorial", "blue", "airy", "grid"),
        ("layer_stack", "signal_glow", "amber", "balanced", "rings"),
    ]
    out: list[BundleCase] = []
    for base in cases:
        if base.family != "memory_map":
            continue
        for form_token, theme, tone, density, background in frontier_specs:
            if form_token == base.form_token:
                continue
            variant_key = f"frontier_{form_token}_{theme}_{tone}_{density}_{background}"
            out.append(
                _bundle_variant_case(
                    base,
                    split="cross_form",
                    variant_key=variant_key,
                    form_token=form_token,
                    form_phrase=phrase_lookup[(base.family, form_token)],
                    theme=theme,
                    tone=tone,
                    density=density,
                    background=background,
                )
            )
    return out


def _system_cross_form_repair_cases(cases: list[BundleCase]) -> list[BundleCase]:
    phrase_lookup = _form_phrase_lookup(cases)
    specs = [
        ("pipeline_overview_system", "build_path", "signal_glow", "green", "compact", "mesh"),
        ("pipeline_overview_system", "selection_path", "infra_dark", "blue", "balanced", "none"),
        ("ir_pipeline_flow_system", "linear_pipeline", "paper_editorial", "amber", "balanced", "grid"),
        ("ir_pipeline_flow_system", "selection_path", "infra_dark", "blue", "balanced", "none"),
        ("kernel_registry_flow_system", "linear_pipeline", "paper_editorial", "amber", "balanced", "grid"),
        ("kernel_registry_flow_system", "build_path", "signal_glow", "green", "compact", "mesh"),
    ]
    by_id = {case.case_id: case for case in cases}
    out: list[BundleCase] = []
    for case_id, form_token, theme, tone, density, background in specs:
        base = by_id.get(case_id)
        if base is None or base.family != "system_diagram":
            continue
        out.append(
            _bundle_variant_case(
                base,
                split="cross_form",
                variant_key=f"system_frontier_{form_token}_{theme}_{tone}_{density}_{background}",
                form_token=form_token,
                form_phrase=phrase_lookup[(base.family, form_token)],
                theme=theme,
                tone=tone,
                density=density,
                background=background,
            )
        )
    return out


def _cross_form_transfer_cases(cases: list[BundleCase]) -> list[BundleCase]:
    by_id = {case.case_id: case for case in cases}
    specs = [
        # Memory-map: teach held-out style bundles on sibling forms so style is not tied to one form.
        ("memory_layout_map", "signal_glow", "blue", "compact", "mesh"),
        ("v7_train_memory_canary", "signal_glow", "blue", "compact", "mesh"),
        ("bump_allocator_quant", "infra_dark", "amber", "compact", "rings"),
        ("memory_layout_map", "paper_editorial", "blue", "airy", "grid"),
        # Timeline: transfer the paper/grid and signal/mesh looks across both forms.
        ("ir_timeline_why", "paper_editorial", "blue", "airy", "grid"),
        ("ir_v66_evolution_timeline", "signal_glow", "green", "compact", "mesh"),
        # System diagrams: keep the same style family independent of form.
        ("pipeline_overview_system", "infra_dark", "blue", "balanced", "none"),
        ("kernel_registry_flow_system", "paper_editorial", "amber", "balanced", "grid"),
        ("ir_pipeline_flow_system", "signal_glow", "green", "compact", "mesh"),
    ]
    out: list[BundleCase] = []
    for case_id, theme, tone, density, background in specs:
        base = by_id.get(case_id)
        if base is None:
            continue
        out.append(
            _variant_case(
                base,
                theme=theme,
                tone=tone,
                density=density,
                background=background,
                split="cross_form",
            )
        )
    return out


def _holdout_cases(cases: list[BundleCase]) -> list[BundleCase]:
    by_id = {case.case_id: case for case in cases}
    specs = [
        ("memory_layout_map", "infra_dark", "amber", "compact", "rings"),
        ("bump_allocator_quant", "signal_glow", "blue", "compact", "mesh"),
        ("v7_train_memory_canary", "signal_glow", "amber", "balanced", "rings"),
        ("ir_v66_evolution_timeline", "paper_editorial", "blue", "airy", "grid"),
        ("ir_timeline_why", "signal_glow", "green", "compact", "mesh"),
        ("pipeline_overview_system", "paper_editorial", "amber", "balanced", "grid"),
        ("ir_pipeline_flow_system", "signal_glow", "green", "compact", "mesh"),
        ("kernel_registry_flow_system", "infra_dark", "blue", "balanced", "none"),
    ]
    out: list[BundleCase] = []
    for case_id, theme, tone, density, background in specs:
        base = by_id.get(case_id)
        if base is None:
            continue
        out.append(_variant_case(base, theme=theme, tone=tone, density=density, background=background, split="holdout"))
    return out


def _family_label(family: str) -> str:
    return {
        "memory_map": "memory map",
        "timeline": "timeline",
        "system_diagram": "system diagram",
    }.get(str(family or ""), str(family or "visual scene").replace("_", " "))


def _count_phrase(bundle: SceneBundle) -> str:
    if bundle.family == "memory_map":
        return (
            f"{bundle.topology['segments']} ordered memory segments, "
            f"{bundle.topology['brackets']} grouped brackets, and "
            f"{bundle.topology['cards']} side cards"
        )
    if bundle.family == "timeline":
        return (
            f"{bundle.topology['stages']} timeline stages, "
            f"{bundle.topology['arrows']} connector arrows, and "
            f"footer count {bundle.topology['footer']}"
        )
    return (
        f"{bundle.topology['stages']} system stages, "
        f"{bundle.topology['links']} links, "
        f"terminal count {bundle.topology['terminal']}, and "
        f"footer count {bundle.topology['footer']}"
    )


def _canonical_prompt(bundle: SceneBundle) -> str:
    return bundle.to_prompt_tags()


def _bridge_prompt_variants(case: BundleCase, prompt_text: str, bundle: SceneBundle) -> list[tuple[str, str]]:
    family_label = _family_label(bundle.family)
    family_forms = ", ".join(str(form).replace("_", " ") for form in family_specs()[bundle.family]["forms"])
    count_phrase = _count_phrase(bundle)
    theme = str(bundle.theme).replace("_", " ")
    tone = str(bundle.tone).replace("_", " ")
    density = str(bundle.density).replace("_", " ")
    background = str(bundle.background).replace("_", " ")
    return [
        (
            "bridge_create",
            f"Create a generalized visual bundle for a {family_label} using {case.form_phrase}, {count_phrase}, {theme} theme, {tone} tone, {density} spacing, and {background} background. Return [bundle] DSL only. {prompt_text}",
        ),
        (
            "bridge_count_guard",
            f"Emit the generalized visual bundle with exact topology counts: {count_phrase}. Keep family, form, and style aligned with the prompt and stop at [/bundle]. {prompt_text}",
        ),
        (
            "bridge_bundle_only",
            f"Return only one generalized visual scene bundle from [bundle] to [/bundle]. Preserve the requested family, form, style, and topology exactly. {prompt_text}",
        ),
        (
            "bridge_style_lock",
            f"Preserve family, form, theme, tone, density, and background exactly. Do not drift to a nearby seen style bundle. Return bundle DSL only. {prompt_text}",
        ),
        (
            "repair_family_form_lock",
            f"Keep the requested family and form exact. Do not drift to a nearby sibling family or sibling form. Return one clean [bundle] only. {prompt_text}",
        ),
        (
            "repair_bundle_singletons",
            f"Emit exactly one tag for each singleton field: family, form, theme, tone, density, background, and each topology counter. Return one clean [bundle] only. {prompt_text}",
        ),
        (
            "repair_topology_lock",
            f"Preserve the requested topology counters exactly and keep them attached to the right family and form. Do not drift to a nearby seen count pattern. {prompt_text}",
        ),
        (
            "repair_clean_stop",
            f"Return exactly one clean bundle block from [bundle] to [/bundle] and stop. Do not emit prompt text, a second bundle, or any continuation after [/bundle]. {prompt_text}",
        ),
        (
            "repair_control_stop",
            f"Return one clean [bundle] only. Keep family, form, style, and topology exact. Do not copy instruction wrappers or control markers from the prompt, and stop immediately after [/bundle]. {prompt_text}",
        ),
    ]


def _hidden_prompt_variants(case: BundleCase, prompt_text: str, bundle: SceneBundle) -> list[tuple[str, str]]:
    family_label = _family_label(bundle.family)
    count_phrase = _count_phrase(bundle)
    return [
        (
            "hidden_compose",
            f"Compose the generalized visual scene bundle for a {family_label} using {case.form_phrase} with {count_phrase}. Keep compiler-facing bundle tags only. {prompt_text}",
        ),
        (
            "hidden_stop",
            f"Return one generalized visual scene bundle for a {family_label} and stop exactly at [/bundle]. Preserve the requested family, form, style, and topology. {prompt_text}",
        ),
    ]


def _train_hidden_prompt_variants(case: BundleCase, prompt_text: str, bundle: SceneBundle) -> list[tuple[str, str]]:
    family_label = _family_label(bundle.family)
    count_phrase = _count_phrase(bundle)
    return [
        (
            "train_hidden_compose",
            f"Compose one generalized visual scene bundle for a {family_label} using {case.form_phrase} with {count_phrase}. Keep compiler-facing bundle tags only. {prompt_text}",
        ),
        (
            "train_hidden_stop",
            f"Return one generalized visual scene bundle for a {family_label} and stop exactly at [/bundle]. Preserve the requested family, form, style, and topology. {prompt_text}",
        ),
        (
            "train_hidden_style_bundle",
            f"Compose one generalized visual scene bundle. Preserve the requested family, form, theme, tone, density, background, and topology exactly, and stop at [/bundle]. {prompt_text}",
        ),
        (
            "train_hidden_clean_stop",
            f"Return exactly one clean generalized visual scene bundle from [bundle] to [/bundle] and stop. Do not emit a second bundle, prompt text, or any continuation after [/bundle]. {prompt_text}",
        ),
    ]


def _write_probe_report_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_probe_report_contract.json"
    payload = {
        "schema": "ck.probe_report_contract.v1",
        "title": "Spec16 Generalized Visual Scene Bundle Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/bundle]"],
            "renderer": "structured_svg_scene_spec16.v1",
            "repairer": "spec16_scene_bundle.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 256,
            "temperature": 0.0,
            "stop_on_text": [
                "Create a generalized visual bundle",
                "Return only one [bundle]",
                "Emit the generalized visual bundle",
                "Compose the generalized visual scene bundle",
                "Compose one generalized visual scene bundle",
                "Emit one generalized visual bundle",
                "Preserve the requested topology counters",
                "Keep the requested family and form exact",
                "Emit exactly one tag for each singleton field",
                "Return exactly one clean [bundle]",
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
        "goal": "Learn a strict generalized visual scene bundle that lowers into multiple solved infographic families.",
        "notes": [
            "Model-facing prompts stay family-generic and contain only design/control language.",
            "Visible text and factual payload stay external in content.json.",
            "The output contract is the shared [bundle] generalized visual DSL only.",
        ],
        "scorer": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/bundle]"],
            "renderer": "structured_svg_scene_spec16.v1",
            "repairer": "spec16_scene_bundle.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 256,
            "temperature": 0.0,
            "stop_on_text": [
                "Create a generalized visual bundle",
                "Return only one [bundle]",
                "Compose the generalized visual scene bundle",
                "Compose one generalized visual scene bundle",
                "Emit one generalized visual bundle",
                "Preserve the requested topology counters",
                "Keep the requested family and form exact",
                "Emit exactly one tag for each singleton field",
                "Return exactly one clean [bundle]",
            ],
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec16_scene_bundle", help="Output prefix")
    ap.add_argument("--train-repeats", type=int, default=6)
    ap.add_argument("--holdout-repeats", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cases = _cases()
    all_cases = (
        base_cases
        + _train_aug_cases(base_cases)
        + _style_axis_aug_cases(base_cases)
        + _form_contrast_cases(base_cases)
        + _memory_style_contrast_cases(base_cases)
        + _memory_single_axis_style_cases(base_cases)
        + _memory_cross_form_frontier_cases(base_cases)
        + _system_cross_form_repair_cases(base_cases)
        + _cross_form_transfer_cases(base_cases)
        + _holdout_cases(base_cases)
    )
    render_rows: list[dict[str, Any]] = []
    train_rows: list[str] = []
    holdout_rows: list[str] = []
    tokenizer_corpus: list[str] = []
    seen_prompts: list[str] = []
    holdout_prompts: list[str] = []
    hidden_seen_prompts: list[str] = []
    hidden_holdout_prompts: list[str] = []

    for case in all_cases:
        bundle = case.bundle
        prompt_text = _canonical_prompt(bundle)
        output_tokens = serialize_scene_bundle(bundle)
        split = str(case.split or "train")
        content_json = case.content_json
        svg_xml = render_structured_scene_spec16_svg(output_tokens, content=content_json)

        rows: list[tuple[str, str, str, bool]] = [(prompt_text, "tag_canonical", split, True)]
        bridge_rows = _bridge_prompt_variants(case, prompt_text, bundle)
        hidden_train_rows = _train_hidden_prompt_variants(case, prompt_text, bundle)

        if split == "train":
            keep_bridge = {
                "bridge_create",
                "bridge_count_guard",
                "bridge_bundle_only",
                "bridge_style_lock",
                "repair_family_form_lock",
                "repair_bundle_singletons",
                "repair_topology_lock",
                "repair_clean_stop",
                "repair_control_stop",
            }
            keep_hidden = {"train_hidden_compose", "train_hidden_stop", "train_hidden_style_bundle", "train_hidden_clean_stop"}
        elif split in {"train_aug", "style_aug", "cross_form", "contrast_aug"}:
            keep_bridge = {
                "bridge_create",
                "bridge_count_guard",
                "bridge_bundle_only",
                "bridge_style_lock",
                "repair_family_form_lock",
                "repair_bundle_singletons",
                "repair_topology_lock",
                "repair_clean_stop",
                "repair_control_stop",
            }
            keep_hidden = {"train_hidden_compose", "train_hidden_stop", "train_hidden_style_bundle", "train_hidden_clean_stop"}
        else:
            keep_bridge = {"bridge_create", "bridge_bundle_only", "repair_family_form_lock"}
            keep_hidden = set()

        rows.extend(
            (prompt, prompt_surface, split, True)
            for prompt_surface, prompt in bridge_rows
            if prompt_surface in keep_bridge
        )
        rows.extend(
            (prompt, prompt_surface, split, True)
            for prompt_surface, prompt in hidden_train_rows
            if prompt_surface in keep_hidden
        )
        hidden_split = "probe_hidden_train" if split in {"train", "train_aug"} else "probe_hidden_holdout"
        rows.extend((prompt, prompt_surface, hidden_split, False) for prompt_surface, prompt in _hidden_prompt_variants(case, prompt_text, bundle))

        for prompt, prompt_surface, row_split, training_prompt in rows:
            render_rows.append(
                {
                    "prompt": prompt,
                    "canonical_prompt": prompt_text,
                    "output_tokens": output_tokens,
                    "content_json": dict(content_json),
                    "svg_xml": svg_xml,
                    "split": row_split,
                    "layout": bundle.family,
                    "family": bundle.family,
                    "case_id": case.case_id,
                    "form_token": case.form_token,
                    "theme": bundle.theme,
                    "tone": bundle.tone,
                    "density": bundle.density,
                    "background": bundle.background,
                    "topology": dict(bundle.topology),
                    "source_asset": case.source_asset,
                    "prompt_surface": prompt_surface,
                    "training_prompt": bool(training_prompt),
                    **{key: int(value) for key, value in bundle.topology.items()},
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
        "schema": "ck.spec16.generated_dataset.v1",
        "prefix": args.prefix,
        "line": "spec16",
        "target": "generalized_visual_scene_bundle",
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
