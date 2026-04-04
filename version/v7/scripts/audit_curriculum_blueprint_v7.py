#!/usr/bin/env python3
"""Audit a structured curriculum blueprint before launching a new spec."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from spec16_scene_bundle_v7 import family_specs as spec16_family_specs
except ModuleNotFoundError:  # pragma: no cover
    from version.v7.scripts.spec16_scene_bundle_v7 import family_specs as spec16_family_specs


SCHEMA = "ck.curriculum_blueprint_audit.v1"
BLUEPRINT_SCHEMA = "ck.curriculum_blueprint.v1"
BANNED_SURFACE_TYPES = {"repair_prose", "warning_language", "negative_only"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _finding(level: str, message: str) -> dict[str, str]:
    return {"level": level, "message": message}


def _generator_surface_counts(render_catalog_doc: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(render_catalog_doc, list):
        return counts
    for row in render_catalog_doc:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("training_prompt")):
            continue
        surface = str(row.get("prompt_surface") or "").strip()
        if not surface:
            continue
        counts[surface] += 1
    return counts


def audit_blueprint_doc(
    doc: dict[str, Any],
    *,
    blueprint_path: str = "",
    render_catalog_doc: Any = None,
    render_catalog_path: str = "",
) -> dict[str, Any]:
    findings: list[dict[str, str]] = []

    if str(doc.get("schema") or "") != BLUEPRINT_SCHEMA:
        findings.append(_finding("fail", f"blueprint schema must be {BLUEPRINT_SCHEMA}"))

    family_docs = [row for row in (doc.get("families") or []) if isinstance(row, dict)]
    profile_docs = [row for row in (doc.get("intent_profiles") or []) if isinstance(row, dict)]
    surface_docs = [row for row in (doc.get("surfaces") or []) if isinstance(row, dict)]
    competency_docs = [row for row in (doc.get("competencies") or []) if isinstance(row, dict)]
    frontier_docs = [row for row in (doc.get("predicted_failure_frontiers") or []) if isinstance(row, dict)]
    stage_docs = [row for row in (doc.get("curriculum_stages") or []) if isinstance(row, dict)]
    holdouts = doc.get("holdouts") if isinstance(doc.get("holdouts"), dict) else {}

    family_map = {str(row.get("name") or "").strip(): row for row in family_docs if str(row.get("name") or "").strip()}
    profile_map = {str(row.get("id") or "").strip(): row for row in profile_docs if str(row.get("id") or "").strip()}
    surface_map = {str(row.get("id") or "").strip(): row for row in surface_docs if str(row.get("id") or "").strip()}
    competency_map = {str(row.get("id") or "").strip(): row for row in competency_docs if str(row.get("id") or "").strip()}
    frontier_map = {str(row.get("id") or "").strip(): row for row in frontier_docs if str(row.get("id") or "").strip()}
    stage_map = {str(row.get("id") or "").strip(): row for row in stage_docs if str(row.get("id") or "").strip()}

    if len(family_map) != len(family_docs):
        findings.append(_finding("fail", "family names must be unique and non-empty"))
    if len(profile_map) != len(profile_docs):
        findings.append(_finding("fail", "intent profile ids must be unique and non-empty"))
    if len(surface_map) != len(surface_docs):
        findings.append(_finding("fail", "surface ids must be unique and non-empty"))
    if len(competency_map) != len(competency_docs):
        findings.append(_finding("fail", "competency ids must be unique and non-empty"))
    if len(frontier_map) != len(frontier_docs):
        findings.append(_finding("fail", "failure-frontier ids must be unique and non-empty"))
    if len(stage_map) != len(stage_docs):
        findings.append(_finding("fail", "stage ids must be unique and non-empty"))

    spec16_specs = spec16_family_specs()
    family_profile_counts: Counter[str] = Counter()
    family_form_coverage: dict[str, set[str]] = defaultdict(set)
    family_surface_counts: Counter[str] = Counter()
    stage_surface_ids: dict[str, list[str]] = {}
    competency_stage_coverage: dict[str, set[str]] = defaultdict(set)
    frontier_stage_coverage: dict[str, set[str]] = defaultdict(set)

    for family_name, family_doc in family_map.items():
        forms = tuple(_unique_strings(list(family_doc.get("forms") or [])))
        count_fields = tuple(_unique_strings(list(family_doc.get("count_fields") or [])))
        if not forms:
            findings.append(_finding("fail", f"family {family_name} must declare at least one form"))
        if not count_fields:
            findings.append(_finding("fail", f"family {family_name} must declare at least one topology field"))
        reference = spec16_specs.get(family_name)
        if reference is not None:
            if forms != tuple(reference.get("forms") or ()):
                findings.append(_finding("fail", f"family {family_name} forms do not match the frozen spec16 bundle contract"))
            if count_fields != tuple(reference.get("count_fields") or ()):
                findings.append(_finding("fail", f"family {family_name} count fields do not match the frozen spec16 bundle contract"))
        intent_profiles = _unique_strings(list(family_doc.get("intent_profiles") or []))
        for profile_id in intent_profiles:
            if profile_id not in profile_map:
                findings.append(_finding("fail", f"family {family_name} references missing intent profile {profile_id}"))

    for profile_id, profile_doc in profile_map.items():
        family = str(profile_doc.get("family") or "").strip()
        form = str(profile_doc.get("form") or "").strip()
        if family not in family_map:
            findings.append(_finding("fail", f"profile {profile_id} references unknown family {family or '<empty>'}"))
            continue
        allowed_forms = set(_unique_strings(list(family_map[family].get("forms") or [])))
        if form not in allowed_forms:
            findings.append(_finding("fail", f"profile {profile_id} uses form {form or '<empty>'} outside family {family}"))
        family_profile_counts[family] += 1
        family_form_coverage[family].add(form)

    for family_name, family_doc in family_map.items():
        expected_forms = set(_unique_strings(list(family_doc.get("forms") or [])))
        missing_forms = sorted(expected_forms - family_form_coverage.get(family_name, set()))
        if missing_forms:
            findings.append(_finding("fail", f"family {family_name} is missing intent-profile coverage for forms: {', '.join(missing_forms)}"))

    if family_profile_counts:
        min_profiles = min(family_profile_counts.values())
        max_profiles = max(family_profile_counts.values())
        if max_profiles - min_profiles > 1:
            findings.append(_finding("warn", "intent-profile counts are materially imbalanced across families"))

    for surface_id, surface_doc in surface_map.items():
        surface_type = str(surface_doc.get("surface_type") or "").strip()
        if surface_type in BANNED_SURFACE_TYPES:
            findings.append(_finding("fail", f"surface {surface_id} uses banned surface_type {surface_type}"))
        families = _unique_strings(list(surface_doc.get("families") or []))
        if not families:
            findings.append(_finding("fail", f"surface {surface_id} must declare family coverage"))
        for family in families:
            if family not in family_map:
                findings.append(_finding("fail", f"surface {surface_id} references unknown family {family}"))
            else:
                family_surface_counts[family] += 1
        for competency_id in _unique_strings(list(surface_doc.get("competencies") or [])):
            if competency_id not in competency_map:
                findings.append(_finding("fail", f"surface {surface_id} references unknown competency {competency_id}"))
        for frontier_id in _unique_strings(list(surface_doc.get("frontiers") or [])):
            if frontier_id not in frontier_map:
                findings.append(_finding("fail", f"surface {surface_id} references unknown failure frontier {frontier_id}"))

    for competency_id, competency_doc in competency_map.items():
        touched = False
        for surface_doc in surface_docs:
            if competency_id in _unique_strings(list(surface_doc.get("competencies") or [])):
                touched = True
                break
        if not touched:
            findings.append(_finding("fail", f"competency {competency_id} is never taught by any surface"))
        for stage_id in _unique_strings(list(competency_doc.get("must_appear_in_stages") or [])):
            if stage_id not in stage_map:
                findings.append(_finding("fail", f"competency {competency_id} requires unknown stage {stage_id}"))

    for frontier_id, frontier_doc in frontier_map.items():
        covered_by = _unique_strings(list(frontier_doc.get("covered_by_surfaces") or []))
        if not covered_by:
            findings.append(_finding("fail", f"failure frontier {frontier_id} must list at least one covering surface"))
        for surface_id in covered_by:
            if surface_id not in surface_map:
                findings.append(_finding("fail", f"failure frontier {frontier_id} references unknown covering surface {surface_id}"))

    for stage_id, stage_doc in stage_map.items():
        mix = [row for row in (stage_doc.get("surface_mix") or []) if isinstance(row, dict)]
        total_weight = 0
        seen_surfaces: list[str] = []
        for row in mix:
            surface_id = str(row.get("surface") or "").strip()
            weight = int(row.get("weight") or 0)
            if surface_id not in surface_map:
                findings.append(_finding("fail", f"stage {stage_id} references unknown surface {surface_id or '<empty>'}"))
                continue
            if weight <= 0:
                findings.append(_finding("fail", f"stage {stage_id} surface {surface_id} must have positive weight"))
                continue
            total_weight += weight
            seen_surfaces.append(surface_id)
            surface_doc = surface_map[surface_id]
            for competency_id in _unique_strings(list(surface_doc.get("competencies") or [])):
                competency_stage_coverage[competency_id].add(stage_id)
            for frontier_id in _unique_strings(list(surface_doc.get("frontiers") or [])):
                frontier_stage_coverage[frontier_id].add(stage_id)
        stage_surface_ids[stage_id] = seen_surfaces
        if total_weight != 100:
            findings.append(_finding("fail", f"stage {stage_id} surface weights must sum to 100, found {total_weight}"))
        non_anchor_count = sum(
            1
            for surface_id in seen_surfaces
            if str(surface_map[surface_id].get("surface_type") or "").strip() != "anchor"
        )
        if stage_id == "stage_b" and non_anchor_count < 4:
            findings.append(_finding("warn", "stage_b should carry several non-anchor surfaces so planning pressure is real"))

    for competency_id, competency_doc in competency_map.items():
        for stage_id in _unique_strings(list(competency_doc.get("must_appear_in_stages") or [])):
            if stage_id not in competency_stage_coverage.get(competency_id, set()):
                findings.append(_finding("fail", f"competency {competency_id} is required in {stage_id} but no stage surface teaches it"))

    for frontier_id in frontier_map:
        if frontier_id not in frontier_stage_coverage:
            findings.append(_finding("fail", f"failure frontier {frontier_id} is not covered by any stage"))

    for family_name in family_map:
        if family_surface_counts.get(family_name, 0) < 3:
            findings.append(_finding("warn", f"family {family_name} appears in too few curriculum surfaces"))

    for split_name in ("visible", "hidden"):
        split_doc = holdouts.get(split_name) if isinstance(holdouts.get(split_name), dict) else {}
        per_family = int(split_doc.get("per_family") or 0)
        if per_family <= 0:
            findings.append(_finding("fail", f"holdout split {split_name} must set per_family > 0"))
        for surface_id in _unique_strings(list(split_doc.get("required_surfaces") or [])):
            if surface_id not in surface_map:
                findings.append(_finding("fail", f"holdout split {split_name} references unknown surface {surface_id}"))

    for surface_id in _unique_strings(list(holdouts.get("non_regression") or [])):
        if surface_id not in surface_map:
            findings.append(_finding("fail", f"non_regression surface {surface_id} is missing"))
        else:
            for stage_id, seen_surfaces in stage_surface_ids.items():
                if surface_id not in seen_surfaces:
                    findings.append(_finding("fail", f"non_regression surface {surface_id} must appear in stage {stage_id}"))

    actual_surface_counts = _generator_surface_counts(render_catalog_doc)
    actual_surface_ids = set(actual_surface_counts)
    expected_surface_ids = set(surface_map)
    generator_missing_surfaces = sorted(expected_surface_ids - actual_surface_ids)
    generator_extra_surfaces = sorted(actual_surface_ids - expected_surface_ids)
    if render_catalog_doc is not None:
        if generator_missing_surfaces:
            findings.append(
                _finding(
                    "fail",
                    "render catalog is missing declared training surfaces: "
                    + ", ".join(generator_missing_surfaces),
                )
            )
        if generator_extra_surfaces:
            findings.append(
                _finding(
                    "warn",
                    "render catalog contains undeclared training surfaces: "
                    + ", ".join(generator_extra_surfaces),
                )
            )

    verdict = "pass"
    if any(row["level"] == "fail" for row in findings):
        verdict = "fail"
    elif any(row["level"] == "warn" for row in findings):
        verdict = "warn"

    taxonomy = doc.get("intent_taxonomy") if isinstance(doc.get("intent_taxonomy"), dict) else {}
    summary = {
        "family_count": len(family_map),
        "surface_count": len(surface_map),
        "stage_count": len(stage_map),
        "competency_count": len(competency_map),
        "failure_frontier_count": len(frontier_map),
        "intent_profile_count": len(profile_map),
        "topic_count": len(_unique_strings(list(taxonomy.get("topics") or []))),
        "goal_count": len(_unique_strings(list(taxonomy.get("goals") or []))),
        "audience_count": len(_unique_strings(list(taxonomy.get("audiences") or []))),
        "family_profile_counts": {key: int(family_profile_counts.get(key, 0)) for key in sorted(family_map)},
        "family_surface_counts": {key: int(family_surface_counts.get(key, 0)) for key in sorted(family_map)},
        "stage_surface_counts": {key: len(value) for key, value in sorted(stage_surface_ids.items())},
        "competency_stage_coverage": {key: sorted(value) for key, value in sorted(competency_stage_coverage.items())},
        "frontier_stage_coverage": {key: sorted(value) for key, value in sorted(frontier_stage_coverage.items())},
        "generator_surface_count": len(actual_surface_counts),
        "generator_surface_counts": dict(sorted(actual_surface_counts.items())),
        "generator_missing_surfaces": generator_missing_surfaces,
        "generator_extra_surfaces": generator_extra_surfaces,
    }

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blueprint_path": blueprint_path,
        "render_catalog_path": render_catalog_path,
        "spec": doc.get("spec"),
        "title": doc.get("title"),
        "verdict": verdict,
        "summary": summary,
        "findings": findings,
    }


def audit_blueprint(path: Path, *, render_catalog_path: Path | None = None) -> dict[str, Any]:
    render_catalog_doc = _load_json(render_catalog_path) if render_catalog_path else None
    return audit_blueprint_doc(
        _load_json(path),
        blueprint_path=str(path),
        render_catalog_doc=render_catalog_doc,
        render_catalog_path="" if render_catalog_path is None else str(render_catalog_path),
    )


def _build_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    findings = payload.get("findings") or []
    grouped: dict[str, list[str]] = {"fail": [], "warn": [], "note": []}
    for row in findings:
        level = str(row.get("level") or "note").strip().lower()
        grouped.setdefault(level, []).append(str(row.get("message") or "").strip())

    lines = [
        "# Curriculum Blueprint Audit",
        "",
        f"- spec: `{payload.get('spec')}`",
        f"- title: `{payload.get('title')}`",
        f"- verdict: `{payload.get('verdict')}`",
        f"- blueprint: `{payload.get('blueprint_path')}`",
        f"- render catalog: `{payload.get('render_catalog_path') or 'not provided'}`",
        f"- generated at: `{payload.get('generated_at')}`",
        "",
        "## Summary",
        "",
        f"- families: `{summary.get('family_count')}`",
        f"- intent profiles: `{summary.get('intent_profile_count')}`",
        f"- topics: `{summary.get('topic_count')}`",
        f"- goals: `{summary.get('goal_count')}`",
        f"- audiences: `{summary.get('audience_count')}`",
        f"- surfaces: `{summary.get('surface_count')}`",
        f"- competencies: `{summary.get('competency_count')}`",
        f"- failure frontiers: `{summary.get('failure_frontier_count')}`",
        "",
        "## Generator Coverage",
        "",
        f"- actual training surfaces: `{summary.get('generator_surface_count')}`",
        f"- missing declared surfaces: `{', '.join(summary.get('generator_missing_surfaces') or []) or 'none'}`",
        f"- undeclared extra surfaces: `{', '.join(summary.get('generator_extra_surfaces') or []) or 'none'}`",
        "",
        "## Family Coverage",
        "",
    ]
    family_profile_counts = summary.get("family_profile_counts") or {}
    family_surface_counts = summary.get("family_surface_counts") or {}
    for family in sorted(family_profile_counts):
        lines.append(
            f"- `{family}`: profiles=`{family_profile_counts.get(family)}`, surfaces=`{family_surface_counts.get(family, 0)}`"
        )
    lines.extend(["", "## Findings", ""])
    for level in ("fail", "warn", "note"):
        lines.append(f"### {level.upper()}")
        items = grouped.get(level) or []
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit a curriculum blueprint JSON file.")
    ap.add_argument("--blueprint", required=True, help="Path to the curriculum blueprint JSON.")
    ap.add_argument("--render-catalog", default=None, help="Optional generated render catalog to validate against the blueprint.")
    ap.add_argument("--out-json", default=None, help="Optional output JSON path.")
    ap.add_argument("--out-md", default=None, help="Optional output markdown path.")
    ap.add_argument("--strict", action="store_true", help="Exit nonzero when the audit verdict is fail.")
    args = ap.parse_args()

    blueprint_path = Path(args.blueprint).expanduser().resolve()
    render_catalog_path = Path(args.render_catalog).expanduser().resolve() if args.render_catalog else None
    payload = audit_blueprint(blueprint_path, render_catalog_path=render_catalog_path)

    out_json = (
        Path(args.out_json).expanduser().resolve()
        if args.out_json
        else blueprint_path.with_name(blueprint_path.stem.replace("_blueprint", "_audit") + ".json")
    )
    out_md = (
        Path(args.out_md).expanduser().resolve()
        if args.out_md
        else blueprint_path.with_name("SPEC17_CURRICULUM_AUDIT_2026-04-01.md")
    )

    _write_json(out_json, payload)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_build_markdown(payload), encoding="utf-8")

    print(f"[OK] verdict={payload['verdict']} families={payload['summary']['family_count']} surfaces={payload['summary']['surface_count']}")
    print(f"[OK] wrote: {out_json}")
    print(f"[OK] wrote: {out_md}")
    if args.strict and str(payload.get("verdict") or "") == "fail":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
