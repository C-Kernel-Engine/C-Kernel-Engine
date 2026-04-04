#!/usr/bin/env python3
"""Generate the first spec17 bounded-intent dataset over the frozen spec16 bundle contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generate_svg_structured_spec16_v7 import BundleCase, _cases
from render_svg_structured_scene_spec16_v7 import render_structured_scene_spec16_svg
from spec16_scene_bundle_canonicalizer_v7 import serialize_scene_bundle
from spec16_scene_bundle_v7 import SceneBundle, canonicalize_scene_bundle


ROOT = Path(__file__).resolve().parents[3]
BLUEPRINT_PATH = ROOT / "version" / "v7" / "reports" / "spec17_curriculum_blueprint.json"

PROFILE_TO_BASE_CASE = {
    "memory_layout_map": "memory_layout_map",
    "allocator_regions": "bump_allocator_quant",
    "arena_guard_flow": "v7_train_memory_canary",
    "ir_evolution_timeline": "ir_v66_evolution_timeline",
    "why_pipeline_timeline": "ir_timeline_why",
    "operator_changeover_timeline": "ir_timeline_why",
    "pipeline_overview_flow": "pipeline_overview_system",
    "build_pipeline_flow": "ir_pipeline_flow_system",
    "registry_selection_flow": "kernel_registry_flow_system",
}

TRAIN_SURFACES = (
    "explicit_bundle_anchor",
    "clean_stop_anchor",
    "explicit_permuted_anchor",
    "layout_topic_bridge",
    "layout_form_bridge",
    "layout_hint_bridge",
    "intent_family_bridge",
    "form_disambiguation_contrast",
    "style_bundle_inference",
    "topology_budget_bridge",
    "topic_goal_recombination",
    "paraphrase_bridge",
    "family_confusion_contrast",
)

FORM_CONTRAST_PARTNER_IDS = {
    "memory_layout_map": "allocator_regions",
    "allocator_regions": "arena_guard_flow",
    "arena_guard_flow": "memory_layout_map",
    "ir_evolution_timeline": "why_pipeline_timeline",
    "why_pipeline_timeline": "ir_evolution_timeline",
    "operator_changeover_timeline": "ir_evolution_timeline",
    "pipeline_overview_flow": "build_pipeline_flow",
    "build_pipeline_flow": "registry_selection_flow",
    "registry_selection_flow": "pipeline_overview_flow",
}

FAMILY_CONTRAST_PARTNER_IDS = {
    "memory_layout_map": "build_pipeline_flow",
    "allocator_regions": "ir_evolution_timeline",
    "arena_guard_flow": "registry_selection_flow",
    "ir_evolution_timeline": "pipeline_overview_flow",
    "why_pipeline_timeline": "arena_guard_flow",
    "operator_changeover_timeline": "registry_selection_flow",
    "pipeline_overview_flow": "why_pipeline_timeline",
    "build_pipeline_flow": "memory_layout_map",
    "registry_selection_flow": "arena_guard_flow",
}

FORM_DECISION_HINTS = {
    ("memory_map", "layer_stack"): "stack_layers",
    ("memory_map", "typed_regions"): "compare_regions",
    ("memory_map", "arena_sections"): "guarded_zones",
    ("timeline", "milestone_chain"): "dated_milestones",
    ("timeline", "stage_sequence"): "ordered_steps",
    ("system_diagram", "linear_pipeline"): "steady_pipeline",
    ("system_diagram", "build_path"): "build_route",
    ("system_diagram", "selection_path"): "decision_route",
}

FAMILY_DECISION_HINTS = {
    "memory_map": "spatial_structure",
    "timeline": "ordered_sequence",
    "system_diagram": "process_route",
}


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(row).strip() for row in rows if str(row).strip())
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


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


def _load_blueprint() -> dict[str, Any]:
    return json.loads(BLUEPRINT_PATH.read_text(encoding="utf-8"))


def _format_tag_prompt(fields: list[tuple[str, str]]) -> str:
    tokens = ["[task:svg]"]
    for key, value in fields:
        text = str(value or "").strip()
        if not text:
            continue
        tokens.append(f"[{key}:{text}]")
    tokens.append("[OUT]")
    return " ".join(tokens)


def _constraint_for_bundle(bundle: SceneBundle) -> str:
    density = str(bundle.density or "").strip().lower()
    if density in {"compact", "dense"}:
        return density
    return "balanced"


def _content_pack_for_bundle(bundle: SceneBundle) -> str:
    density = str(bundle.density or "").strip().lower()
    if density in {"compact", "airy"}:
        return "brief"
    if density == "dense":
        return "dense"
    return "default"


def _plain_paraphrase(case: "IntentBundleCase") -> str:
    return (
        f"task svg choose one shared visual bundle for topic {case.prompt_topic} "
        f"goal {case.goal} audience {case.audience} emphasis {case.emphasis} "
        f"return bundle only [OUT]"
    )


def _explicit_permuted_prompt(case: "IntentBundleCase") -> str:
    bundle = case.bundle
    permuted_fields: list[tuple[str, str]] = [
        ("theme", bundle.theme),
        ("layout", bundle.family),
        ("tone", bundle.tone),
        ("form", bundle.form),
        ("density", bundle.density),
        ("background", bundle.background),
    ]
    permuted_fields.extend((key, str(value)) for key, value in bundle.topology.items())
    return _format_tag_prompt(permuted_fields)


def _decision_boundary(target_hint: str, distractor_hint: str) -> str:
    target = str(target_hint or "").strip()
    distractor = str(distractor_hint or "").strip()
    if not target:
        return distractor
    if not distractor or distractor == target:
        return target
    return f"{target}_over_{distractor}"


@dataclass(frozen=True)
class IntentBundleCase:
    profile_id: str
    prompt_topic: str
    goal: str
    audience: str
    emphasis: str
    base_case: BundleCase
    bundle_override: dict[str, Any]

    @property
    def case_id(self) -> str:
        return self.profile_id

    @property
    def bundle(self) -> SceneBundle:
        return canonicalize_scene_bundle(self.bundle_override)

    @property
    def source_asset(self) -> str:
        return f"{self.base_case.source_asset}#spec17:{self.profile_id}"

    @property
    def content_json(self) -> dict[str, Any]:
        return self.base_case.content_json


def _build_profile_cases() -> list[IntentBundleCase]:
    blueprint = _load_blueprint()
    profiles = {
        str(row.get("id") or "").strip(): row
        for row in (blueprint.get("intent_profiles") or [])
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    base_cases = {case.case_id: case for case in _cases()}
    out: list[IntentBundleCase] = []
    for profile_id, base_case_id in PROFILE_TO_BASE_CASE.items():
        profile = profiles[profile_id]
        base_case = base_cases[base_case_id]
        bundle_doc = base_case.bundle.to_dict()
        bundle_doc["form"] = str(profile.get("form") or bundle_doc["form"])
        style = profile.get("canonical_style") if isinstance(profile.get("canonical_style"), dict) else {}
        for key in ("theme", "tone", "density", "background"):
            value = str(style.get(key) or "").strip()
            if value:
                bundle_doc[key] = value
        out.append(
            IntentBundleCase(
                profile_id=profile_id,
                prompt_topic=str(profile.get("topic") or profile_id),
                goal=str(profile.get("goal") or "show_flow"),
                audience=str(profile.get("audience") or "technical"),
                emphasis=str(profile.get("emphasis") or "flow"),
                base_case=base_case,
                bundle_override=bundle_doc,
            )
        )
    return out


def _profile_lookup(cases: list[IntentBundleCase] | None = None) -> dict[str, IntentBundleCase]:
    profile_cases = cases if cases is not None else _build_profile_cases()
    return {case.profile_id: case for case in profile_cases}


def _form_contrast_case(case: IntentBundleCase, lookup: dict[str, IntentBundleCase]) -> IntentBundleCase:
    partner_id = FORM_CONTRAST_PARTNER_IDS[case.profile_id]
    partner = lookup[partner_id]
    if partner.bundle.family != case.bundle.family:
        raise ValueError(f"form contrast partner for {case.profile_id} must stay within family")
    if partner.bundle.form == case.bundle.form:
        raise ValueError(f"form contrast partner for {case.profile_id} must use a sibling form")
    return partner


def _family_contrast_case(case: IntentBundleCase, lookup: dict[str, IntentBundleCase]) -> IntentBundleCase:
    partner_id = FAMILY_CONTRAST_PARTNER_IDS[case.profile_id]
    partner = lookup[partner_id]
    if partner.bundle.family == case.bundle.family:
        raise ValueError(f"family contrast partner for {case.profile_id} must use a different family")
    return partner


def _prompt_rows(
    case: IntentBundleCase,
    profile_lookup: dict[str, IntentBundleCase] | None = None,
) -> list[tuple[str, str, str, bool]]:
    bundle = case.bundle
    lookup = profile_lookup or _profile_lookup()
    form_partner = _form_contrast_case(case, lookup)
    family_partner = _family_contrast_case(case, lookup)
    explicit_prompt = bundle.to_prompt_tags()
    explicit_permuted_prompt = _explicit_permuted_prompt(case)
    constraint = _constraint_for_bundle(bundle)
    content_pack = _content_pack_for_bundle(bundle)
    form_hint = FORM_DECISION_HINTS[(bundle.family, bundle.form)]
    form_partner_hint = FORM_DECISION_HINTS[(form_partner.bundle.family, form_partner.bundle.form)]
    family_hint = FAMILY_DECISION_HINTS[bundle.family]
    family_partner_hint = FAMILY_DECISION_HINTS[family_partner.bundle.family]
    base_fields = [
        ("topic", case.prompt_topic),
        ("goal", case.goal),
        ("audience", case.audience),
    ]
    return [
        (explicit_prompt, "explicit_bundle_anchor", "train", True),
        (
            f"Plan exactly one compiler-facing shared visual bundle and stop at [/bundle]. {explicit_prompt}",
            "clean_stop_anchor",
            "train",
            True,
        ),
        (explicit_permuted_prompt, "explicit_permuted_anchor", "train", True),
        (
            _format_tag_prompt([("layout", bundle.family)] + base_fields),
            "layout_topic_bridge",
            "train",
            True,
        ),
        (
            _format_tag_prompt([("layout", bundle.family), ("form", bundle.form)] + base_fields),
            "layout_form_bridge",
            "train",
            True,
        ),
        (
            _format_tag_prompt(
                [
                    ("layout", bundle.family),
                    ("form", bundle.form),
                    ("topic", case.prompt_topic),
                    ("goal", case.goal),
                    ("audience", case.audience),
                    ("emphasis", case.emphasis),
                    ("constraint", constraint),
                    ("content_pack", content_pack),
                ]
            ),
            "layout_hint_bridge",
            "train",
            True,
        ),
        (_format_tag_prompt(base_fields), "intent_family_bridge", "train", True),
        (
            _format_tag_prompt(
                base_fields
                + [
                    ("emphasis", case.emphasis),
                    ("contrast_topic", form_partner.prompt_topic),
                    ("contrast_form", form_partner.bundle.form),
                    ("contrast_emphasis", form_partner.emphasis),
                    ("decision_hint", _decision_boundary(form_hint, form_partner_hint)),
                ]
            ),
            "form_disambiguation_contrast",
            "train",
            True,
        ),
        (
            _format_tag_prompt(base_fields + [("emphasis", case.emphasis), ("content_pack", content_pack)]),
            "style_bundle_inference",
            "train",
            True,
        ),
        (
            _format_tag_prompt(base_fields + [("constraint", constraint), ("content_pack", content_pack)]),
            "topology_budget_bridge",
            "train",
            True,
        ),
        (
            _format_tag_prompt([("goal", case.goal), ("topic", case.prompt_topic), ("audience", case.audience), ("emphasis", case.emphasis)]),
            "topic_goal_recombination",
            "train",
            True,
        ),
        (_plain_paraphrase(case), "paraphrase_bridge", "train", True),
        (
            _format_tag_prompt(
                [
                    ("audience", case.audience),
                    ("topic", case.prompt_topic),
                    ("goal", case.goal),
                    ("emphasis", case.emphasis),
                    ("contrast_topic", family_partner.prompt_topic),
                    ("contrast_goal", family_partner.goal),
                    ("contrast_family", family_partner.bundle.family),
                    ("decision_hint", _decision_boundary(family_hint, family_partner_hint)),
                    ("constraint", constraint),
                ]
            ),
            "family_confusion_contrast",
            "train",
            True,
        ),
        (
            _format_tag_prompt([("audience", case.audience), ("goal", case.goal), ("topic", case.prompt_topic)]),
            "holdout_intent_bridge",
            "holdout",
            False,
        ),
        (
            _format_tag_prompt([("topic", case.prompt_topic), ("audience", case.audience), ("goal", case.goal), ("content_pack", content_pack)]),
            "holdout_style_bundle",
            "holdout",
            False,
        ),
        (
            f"choose one shared bundle for topic {case.prompt_topic} with goal {case.goal} for audience {case.audience}; return bundle only [OUT]",
            "hidden_paraphrase",
            "probe_hidden_train",
            False,
        ),
        (
            _format_tag_prompt([("audience", case.audience), ("topic", case.prompt_topic), ("goal", case.goal), ("emphasis", case.emphasis), ("constraint", constraint)]),
            "hidden_recombination",
            "probe_hidden_holdout",
            False,
        ),
    ]


def _write_eval_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_eval_contract.json"
    payload = {
        "schema": "ck.eval_contract.v1",
        "dataset_type": "svg",
        "goal": "Learn a bounded intent bridge over the frozen spec16 shared scene-bundle contract.",
        "notes": [
            "Prompts use topic, goal, audience, and bounded hints instead of explicit family and style controls.",
            "The output contract remains the shared [bundle] DSL only.",
            "Visible copy and facts remain external in content.json.",
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
                "task svg choose one shared visual bundle",
                "choose one shared bundle",
                "Plan exactly one compiler-facing shared visual bundle",
            ],
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec17_scene_bundle", help="Output prefix")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_cases = _build_profile_cases()
    profile_lookup = _profile_lookup(profile_cases)

    render_rows: list[dict[str, Any]] = []
    seen_prompts: list[str] = []
    holdout_prompts: list[str] = []
    hidden_seen_prompts: list[str] = []
    hidden_holdout_prompts: list[str] = []

    for case in profile_cases:
        bundle = case.bundle
        output_tokens = serialize_scene_bundle(bundle)
        svg_xml = render_structured_scene_spec16_svg(output_tokens, content=case.content_json)
        for prompt, prompt_surface, split, training_prompt in _prompt_rows(case, profile_lookup):
            row = {
                "prompt": prompt,
                "output_tokens": output_tokens,
                "content_json": dict(case.content_json),
                "svg_xml": svg_xml,
                "split": split,
                "layout": bundle.family,
                "family": bundle.family,
                "case_id": case.case_id,
                "profile_id": case.profile_id,
                "form_token": bundle.form,
                "theme": bundle.theme,
                "tone": bundle.tone,
                "density": bundle.density,
                "background": bundle.background,
                "topology": dict(bundle.topology),
                "source_asset": case.source_asset,
                "prompt_surface": prompt_surface,
                "training_prompt": bool(training_prompt),
                "topic": case.prompt_topic,
                "goal": case.goal,
                "audience": case.audience,
                "emphasis": case.emphasis,
                "constraint": _constraint_for_bundle(bundle),
                "content_pack": _content_pack_for_bundle(bundle),
                **{key: int(value) for key, value in bundle.topology.items()},
            }
            render_rows.append(row)
            if split == "train":
                seen_prompts.append(prompt)
            elif split == "holdout":
                holdout_prompts.append(prompt)
            elif split == "probe_hidden_train":
                hidden_seen_prompts.append(prompt)
            elif split == "probe_hidden_holdout":
                hidden_holdout_prompts.append(prompt)

    render_rows = sorted(
        render_rows,
        key=lambda row: (
            str(row.get("family") or ""),
            str(row.get("profile_id") or ""),
            str(row.get("split") or ""),
            str(row.get("prompt_surface") or ""),
            str(row.get("prompt") or ""),
        ),
    )

    (out_dir / f"{args.prefix}_render_catalog.json").write_text(
        json.dumps(render_rows, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_lines(out_dir / f"{args.prefix}_seen_prompts.txt", _dedupe_preserve(seen_prompts))
    _write_lines(out_dir / f"{args.prefix}_holdout_prompts.txt", _dedupe_preserve(holdout_prompts))
    _write_lines(out_dir / f"{args.prefix}_hidden_seen_prompts.txt", _dedupe_preserve(hidden_seen_prompts))
    _write_lines(out_dir / f"{args.prefix}_hidden_holdout_prompts.txt", _dedupe_preserve(hidden_holdout_prompts))
    _write_eval_contract(out_dir, args.prefix)

    summary = {
        "profile_cases": len(profile_cases),
        "families": sorted({case.bundle.family for case in profile_cases}),
        "prompt_surfaces": sorted({str(row.get("prompt_surface") or "") for row in render_rows}),
        "train_rows": sum(1 for row in render_rows if str(row.get("split") or "") == "train"),
        "holdout_rows": sum(1 for row in render_rows if str(row.get("split") or "") == "holdout"),
        "hidden_train_rows": sum(1 for row in render_rows if str(row.get("split") or "") == "probe_hidden_train"),
        "hidden_holdout_rows": sum(1 for row in render_rows if str(row.get("split") or "") == "probe_hidden_holdout"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
