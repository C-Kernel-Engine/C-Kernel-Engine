#!/usr/bin/env python3
"""Generate the spec19 textbook-routing dataset over the frozen spec16 bundle contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generate_svg_structured_spec17_v7 import (
    FAMILY_CONTRAST_PARTNER_IDS,
    FAMILY_DECISION_HINTS,
    FORM_CONTRAST_PARTNER_IDS,
    FORM_DECISION_HINTS,
    IntentBundleCase,
    PROFILE_TO_BASE_CASE,
    _constraint_for_bundle,
    _content_pack_for_bundle,
    _decision_boundary,
    _dedupe_preserve,
    _explicit_permuted_prompt,
    _format_tag_prompt,
    _write_lines,
)
from generate_svg_structured_spec16_v7 import BundleCase, _cases
from render_svg_structured_scene_spec16_v7 import render_structured_scene_spec16_svg
from spec16_scene_bundle_canonicalizer_v7 import serialize_scene_bundle


ROOT = Path(__file__).resolve().parents[3]
BLUEPRINT_PATH = ROOT / "version" / "v7" / "reports" / "spec19_curriculum_blueprint.json"

TRAIN_SURFACES = (
    "explicit_bundle_anchor",
    "explicit_permuted_anchor",
    "clean_stop_anchor",
    "routebook_direct",
    "routebook_direct_hint",
    "form_minimal_pair",
    "family_minimal_pair",
    "routebook_paraphrase",
    "style_topology_bridge",
    "recombination_bridge",
)


def _load_blueprint() -> dict[str, Any]:
    return json.loads(BLUEPRINT_PATH.read_text(encoding="utf-8"))


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


def _plain_paraphrase(case: IntentBundleCase) -> str:
    return (
        f"task svg pick one shared bundle for topic {case.prompt_topic} "
        f"goal {case.goal} audience {case.audience} emphasis {case.emphasis} "
        f"return bundle only [OUT]"
    )


def _hidden_recombination_prompt(case: IntentBundleCase, constraint: str) -> str:
    return _format_tag_prompt(
        [
            ("audience", case.audience),
            ("goal", case.goal),
            ("topic", case.prompt_topic),
            ("emphasis", case.emphasis),
            ("constraint", constraint),
        ]
    )


def _surface_rows(prompts: list[str], surface: str, split: str, training_prompt: bool) -> list[tuple[str, str, str, bool]]:
    return [
        (prompt, surface, split, training_prompt)
        for prompt in _dedupe_preserve(prompts)
    ]


def _routebook_direct_prompts(case: IntentBundleCase) -> list[str]:
    return [
        _format_tag_prompt([("topic", case.prompt_topic), ("goal", case.goal), ("audience", case.audience)]),
        _format_tag_prompt([("goal", case.goal), ("topic", case.prompt_topic), ("audience", case.audience)]),
        _format_tag_prompt([("audience", case.audience), ("topic", case.prompt_topic), ("goal", case.goal)]),
        _format_tag_prompt([("topic", case.prompt_topic), ("audience", case.audience), ("goal", case.goal)]),
    ]


def _routebook_direct_hint_prompts(
    case: IntentBundleCase,
    *,
    constraint: str,
    content_pack: str,
) -> list[str]:
    return [
        _format_tag_prompt(
            [
                ("topic", case.prompt_topic),
                ("goal", case.goal),
                ("audience", case.audience),
                ("emphasis", case.emphasis),
                ("constraint", constraint),
                ("content_pack", content_pack),
            ]
        ),
        _format_tag_prompt(
            [
                ("goal", case.goal),
                ("topic", case.prompt_topic),
                ("audience", case.audience),
                ("content_pack", content_pack),
                ("emphasis", case.emphasis),
                ("constraint", constraint),
            ]
        ),
        _format_tag_prompt(
            [
                ("audience", case.audience),
                ("topic", case.prompt_topic),
                ("goal", case.goal),
                ("constraint", constraint),
                ("emphasis", case.emphasis),
                ("content_pack", content_pack),
            ]
        ),
        _format_tag_prompt(
            [
                ("topic", case.prompt_topic),
                ("audience", case.audience),
                ("goal", case.goal),
                ("content_pack", content_pack),
                ("constraint", constraint),
                ("emphasis", case.emphasis),
            ]
        ),
        _format_tag_prompt(
            [
                ("goal", case.goal),
                ("audience", case.audience),
                ("topic", case.prompt_topic),
                ("emphasis", case.emphasis),
                ("content_pack", content_pack),
                ("constraint", constraint),
            ]
        ),
    ]


def _form_minimal_pair_prompts(
    case: IntentBundleCase,
    *,
    form_partner: IntentBundleCase,
    decision_hint: str,
) -> list[str]:
    return [
        _format_tag_prompt(
            [
                ("topic", case.prompt_topic),
                ("goal", case.goal),
                ("audience", case.audience),
                ("emphasis", case.emphasis),
                ("contrast_topic", form_partner.prompt_topic),
                ("contrast_form", form_partner.bundle.form),
                ("contrast_emphasis", form_partner.emphasis),
                ("decision_hint", decision_hint),
            ]
        ),
        _format_tag_prompt(
            [
                ("goal", case.goal),
                ("topic", case.prompt_topic),
                ("audience", case.audience),
                ("decision_hint", decision_hint),
                ("contrast_topic", form_partner.prompt_topic),
                ("contrast_form", form_partner.bundle.form),
                ("emphasis", case.emphasis),
                ("contrast_emphasis", form_partner.emphasis),
            ]
        ),
        _format_tag_prompt(
            [
                ("audience", case.audience),
                ("topic", case.prompt_topic),
                ("goal", case.goal),
                ("contrast_topic", form_partner.prompt_topic),
                ("contrast_emphasis", form_partner.emphasis),
                ("decision_hint", decision_hint),
                ("contrast_form", form_partner.bundle.form),
                ("emphasis", case.emphasis),
            ]
        ),
        _format_tag_prompt(
            [
                ("topic", case.prompt_topic),
                ("audience", case.audience),
                ("goal", case.goal),
                ("contrast_form", form_partner.bundle.form),
                ("decision_hint", decision_hint),
                ("contrast_topic", form_partner.prompt_topic),
                ("emphasis", case.emphasis),
                ("contrast_emphasis", form_partner.emphasis),
            ]
        ),
        _format_tag_prompt(
            [
                ("goal", case.goal),
                ("audience", case.audience),
                ("topic", case.prompt_topic),
                ("contrast_topic", form_partner.prompt_topic),
                ("decision_hint", decision_hint),
                ("contrast_form", form_partner.bundle.form),
                ("contrast_emphasis", form_partner.emphasis),
                ("emphasis", case.emphasis),
            ]
        ),
    ]


def _family_minimal_pair_prompts(
    case: IntentBundleCase,
    *,
    family_partner: IntentBundleCase,
    decision_hint: str,
) -> list[str]:
    return [
        _format_tag_prompt(
            [
                ("audience", case.audience),
                ("topic", case.prompt_topic),
                ("goal", case.goal),
                ("emphasis", case.emphasis),
                ("contrast_topic", family_partner.prompt_topic),
                ("contrast_goal", family_partner.goal),
                ("contrast_family", family_partner.bundle.family),
                ("decision_hint", decision_hint),
            ]
        ),
        _format_tag_prompt(
            [
                ("topic", case.prompt_topic),
                ("audience", case.audience),
                ("goal", case.goal),
                ("contrast_family", family_partner.bundle.family),
                ("contrast_topic", family_partner.prompt_topic),
                ("decision_hint", decision_hint),
                ("contrast_goal", family_partner.goal),
                ("emphasis", case.emphasis),
            ]
        ),
        _format_tag_prompt(
            [
                ("goal", case.goal),
                ("audience", case.audience),
                ("topic", case.prompt_topic),
                ("contrast_topic", family_partner.prompt_topic),
                ("contrast_goal", family_partner.goal),
                ("decision_hint", decision_hint),
                ("contrast_family", family_partner.bundle.family),
                ("emphasis", case.emphasis),
            ]
        ),
        _format_tag_prompt(
            [
                ("audience", case.audience),
                ("goal", case.goal),
                ("topic", case.prompt_topic),
                ("decision_hint", decision_hint),
                ("contrast_family", family_partner.bundle.family),
                ("contrast_topic", family_partner.prompt_topic),
                ("contrast_goal", family_partner.goal),
                ("emphasis", case.emphasis),
            ]
        ),
        _format_tag_prompt(
            [
                ("topic", case.prompt_topic),
                ("goal", case.goal),
                ("audience", case.audience),
                ("contrast_goal", family_partner.goal),
                ("contrast_topic", family_partner.prompt_topic),
                ("emphasis", case.emphasis),
                ("decision_hint", decision_hint),
                ("contrast_family", family_partner.bundle.family),
            ]
        ),
    ]


def _routebook_paraphrase_prompts(case: IntentBundleCase) -> list[str]:
    return [
        _plain_paraphrase(case),
        (
            f"task svg choose one compiler facing bundle for topic {case.prompt_topic} "
            f"goal {case.goal} audience {case.audience} emphasis {case.emphasis} "
            f"and output bundle only [OUT]"
        ),
        (
            f"choose one shared bundle for topic {case.prompt_topic} "
            f"goal {case.goal} audience {case.audience} emphasis {case.emphasis} [OUT]"
        ),
        (
            f"task svg route topic {case.prompt_topic} to one shared bundle "
            f"for audience {case.audience} with goal {case.goal} and emphasis {case.emphasis} [OUT]"
        ),
        (
            f"pick one visual bundle only for audience {case.audience} "
            f"topic {case.prompt_topic} goal {case.goal} emphasis {case.emphasis} [OUT]"
        ),
        (
            f"return one shared bundle only: topic {case.prompt_topic}, "
            f"goal {case.goal}, audience {case.audience}, emphasis {case.emphasis} [OUT]"
        ),
    ]


def _style_topology_bridge_prompts(
    case: IntentBundleCase,
    *,
    constraint: str,
    content_pack: str,
) -> list[str]:
    return [
        _format_tag_prompt(
            [
                ("topic", case.prompt_topic),
                ("goal", case.goal),
                ("audience", case.audience),
                ("emphasis", case.emphasis),
                ("constraint", constraint),
                ("content_pack", content_pack),
            ]
        ),
        _format_tag_prompt(
            [
                ("goal", case.goal),
                ("topic", case.prompt_topic),
                ("audience", case.audience),
                ("constraint", constraint),
                ("content_pack", content_pack),
                ("emphasis", case.emphasis),
            ]
        ),
        _format_tag_prompt(
            [
                ("audience", case.audience),
                ("goal", case.goal),
                ("topic", case.prompt_topic),
                ("content_pack", content_pack),
                ("emphasis", case.emphasis),
                ("constraint", constraint),
            ]
        ),
        _format_tag_prompt(
            [
                ("topic", case.prompt_topic),
                ("audience", case.audience),
                ("goal", case.goal),
                ("emphasis", case.emphasis),
                ("content_pack", content_pack),
                ("constraint", constraint),
            ]
        ),
        _format_tag_prompt(
            [
                ("goal", case.goal),
                ("audience", case.audience),
                ("topic", case.prompt_topic),
                ("constraint", constraint),
                ("emphasis", case.emphasis),
                ("content_pack", content_pack),
            ]
        ),
    ]


def _recombination_bridge_prompts(case: IntentBundleCase, *, constraint: str) -> list[str]:
    return [
        _hidden_recombination_prompt(case, constraint),
        _format_tag_prompt(
            [
                ("topic", case.prompt_topic),
                ("audience", case.audience),
                ("goal", case.goal),
                ("constraint", constraint),
                ("emphasis", case.emphasis),
            ]
        ),
        _format_tag_prompt(
            [
                ("goal", case.goal),
                ("audience", case.audience),
                ("topic", case.prompt_topic),
                ("emphasis", case.emphasis),
                ("constraint", constraint),
            ]
        ),
        _format_tag_prompt(
            [
                ("audience", case.audience),
                ("constraint", constraint),
                ("topic", case.prompt_topic),
                ("goal", case.goal),
                ("emphasis", case.emphasis),
            ]
        ),
    ]


def _holdout_routebook_prompts(case: IntentBundleCase) -> list[str]:
    return [
        _format_tag_prompt([("goal", case.goal), ("topic", case.prompt_topic), ("audience", case.audience)]),
        _format_tag_prompt([("audience", case.audience), ("goal", case.goal), ("topic", case.prompt_topic)]),
    ]


def _holdout_style_topology_prompts(
    case: IntentBundleCase,
    *,
    constraint: str,
    content_pack: str,
) -> list[str]:
    return [
        _format_tag_prompt(
            [
                ("topic", case.prompt_topic),
                ("audience", case.audience),
                ("goal", case.goal),
                ("emphasis", case.emphasis),
                ("constraint", constraint),
                ("content_pack", content_pack),
            ]
        ),
        _format_tag_prompt(
            [
                ("goal", case.goal),
                ("topic", case.prompt_topic),
                ("audience", case.audience),
                ("content_pack", content_pack),
                ("constraint", constraint),
                ("emphasis", case.emphasis),
            ]
        ),
    ]


def _hidden_routebook_paraphrase_prompts(case: IntentBundleCase) -> list[str]:
    return [
        (
            f"choose one shared bundle for topic {case.prompt_topic} "
            f"with goal {case.goal} for audience {case.audience}; return bundle only [OUT]"
        ),
        (
            f"pick one bundle only for audience {case.audience} "
            f"topic {case.prompt_topic} goal {case.goal} [OUT]"
        ),
    ]


def _hidden_recombination_prompts(case: IntentBundleCase, *, constraint: str) -> list[str]:
    return [
        _hidden_recombination_prompt(case, constraint),
        _format_tag_prompt(
            [
                ("goal", case.goal),
                ("topic", case.prompt_topic),
                ("audience", case.audience),
                ("constraint", constraint),
                ("emphasis", case.emphasis),
            ]
        ),
    ]


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
    form_decision_hint = _decision_boundary(form_hint, form_partner_hint)
    family_decision_hint = _decision_boundary(family_hint, family_partner_hint)
    rows: list[tuple[str, str, str, bool]] = []
    rows.extend(_surface_rows([explicit_prompt], "explicit_bundle_anchor", "train", True))
    rows.extend(_surface_rows([explicit_permuted_prompt], "explicit_permuted_anchor", "train", True))
    rows.extend(
        _surface_rows(
            [
                f"Plan exactly one compiler-facing shared visual bundle and stop at [/bundle]. {explicit_prompt}",
                f"Return exactly one [bundle] and stop after [/bundle]. {explicit_prompt}",
            ],
            "clean_stop_anchor",
            "train",
            True,
        )
    )
    rows.extend(_surface_rows(_routebook_direct_prompts(case), "routebook_direct", "train", True))
    rows.extend(
        _surface_rows(
            _routebook_direct_hint_prompts(case, constraint=constraint, content_pack=content_pack),
            "routebook_direct_hint",
            "train",
            True,
        )
    )
    rows.extend(
        _surface_rows(
            _form_minimal_pair_prompts(case, form_partner=form_partner, decision_hint=form_decision_hint),
            "form_minimal_pair",
            "train",
            True,
        )
    )
    rows.extend(
        _surface_rows(
            _family_minimal_pair_prompts(case, family_partner=family_partner, decision_hint=family_decision_hint),
            "family_minimal_pair",
            "train",
            True,
        )
    )
    rows.extend(_surface_rows(_routebook_paraphrase_prompts(case), "routebook_paraphrase", "train", True))
    rows.extend(
        _surface_rows(
            _style_topology_bridge_prompts(case, constraint=constraint, content_pack=content_pack),
            "style_topology_bridge",
            "train",
            True,
        )
    )
    rows.extend(
        _surface_rows(
            _recombination_bridge_prompts(case, constraint=constraint),
            "recombination_bridge",
            "train",
            True,
        )
    )
    rows.extend(_surface_rows(_holdout_routebook_prompts(case), "holdout_routebook_direct", "holdout", False))
    rows.extend(
        _surface_rows(
            _holdout_style_topology_prompts(case, constraint=constraint, content_pack=content_pack),
            "holdout_style_topology_bridge",
            "holdout",
            False,
        )
    )
    rows.extend(
        _surface_rows(
            _hidden_routebook_paraphrase_prompts(case),
            "hidden_routebook_paraphrase",
            "probe_hidden_train",
            False,
        )
    )
    rows.extend(
        _surface_rows(
            _hidden_recombination_prompts(case, constraint=constraint),
            "hidden_recombination",
            "probe_hidden_holdout",
            False,
        )
    )
    return rows


def _write_eval_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_eval_contract.json"
    payload = {
        "schema": "ck.eval_contract.v1",
        "dataset_type": "svg",
        "goal": "Learn textbook-style routing and minimal-pair boundaries over the frozen spec16 shared scene-bundle contract.",
        "notes": [
            "Prompts front-load direct routebook rows and minimal pairs before widening to style and topology inference.",
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
                "task svg pick one shared bundle",
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
    ap.add_argument("--prefix", default="spec19_scene_bundle", help="Output prefix")
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
                "constraint": constraint if (constraint := _constraint_for_bundle(bundle)) else "balanced",
                "content_pack": content_pack if (content_pack := _content_pack_for_bundle(bundle)) else "default",
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
