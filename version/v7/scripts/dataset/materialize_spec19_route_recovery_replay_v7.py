#!/usr/bin/env python3
"""Materialize a replay-heavy spec19 workspace with family/form/syntax recovery coverage."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
R3D_MATERIALIZER = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec19_balanced_coverage_replay_v7.py"
SPEC19_MATERIALIZER = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec19_scene_bundle_v7.py"
SPEC19_GENERATOR = ROOT / "version" / "v7" / "scripts" / "generate_svg_structured_spec19_v7.py"
LINE_NAME = "spec19_route_recovery_replay"
CURRICULUM_BALANCE = ROOT / "version" / "v7" / "scripts" / "dataset" / "spec19_curriculum_balance_v7.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _humanize(value: str) -> str:
    return str(value or "").replace("_", " ").strip()


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


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


def _extract_prompt(row: str) -> str:
    text = str(row or "").strip()
    if not text:
        return ""
    for marker in (" [bundle]", "[bundle]"):
        idx = text.find(marker)
        if idx >= 0:
            return text[:idx].strip()
    return ""


def _filter_prompts(prompts: list[str], forbidden: set[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        text = str(prompt or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        if text in forbidden:
            removed.append(text)
        else:
            kept.append(text)
    return kept, removed


def _filter_catalog_rows(rows: list[dict[str, Any]], forbidden: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    kept: list[dict[str, Any]] = []
    removed: list[str] = []
    seen_removed: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        if prompt and prompt in forbidden:
            if prompt not in seen_removed:
                seen_removed.add(prompt)
                removed.append(prompt)
            continue
        kept.append(row)
    return kept, removed


def _build_recovery_rows(*, gen19: Any, spec19: Any, base: Any) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    r3d = _load_module(R3D_MATERIALIZER, "materialize_spec19_balanced_coverage_replay_v7")
    cases = gen19._build_profile_cases()
    lookup = gen19._profile_lookup(cases)
    pretrain_rows: list[str] = []
    midtrain_rows: list[str] = []
    pretrain_meta: list[dict[str, Any]] = []
    midtrain_meta: list[dict[str, Any]] = []

    for case in cases:
        bundle = case.bundle
        output_tokens = gen19.serialize_scene_bundle(bundle)
        explicit_prompt = bundle.to_prompt_tags()
        explicit_permuted_prompt = gen19._explicit_permuted_prompt(case)
        constraint = gen19._constraint_for_bundle(bundle)
        content_pack = gen19._content_pack_for_bundle(bundle)
        form_partner = gen19._form_contrast_case(case, lookup)
        family_partner = gen19._family_contrast_case(case, lookup)
        form_hint = gen19.FORM_DECISION_HINTS[(bundle.family, bundle.form)]
        form_partner_hint = gen19.FORM_DECISION_HINTS[(form_partner.bundle.family, form_partner.bundle.form)]
        family_hint = gen19.FAMILY_DECISION_HINTS[bundle.family]
        family_partner_hint = gen19.FAMILY_DECISION_HINTS[family_partner.bundle.family]

        pretrain_prompts = [
            (
                "clean_stop_anchor",
                f"Return exactly one [bundle] that keeps family {bundle.family} and form {bundle.form} exactly; stop after [/bundle]. {explicit_prompt}",
            ),
            (
                "clean_stop_anchor",
                f"Use only the fields named by these explicit tags; keep style and topology exact; no extra text after [/bundle]. {explicit_prompt}",
            ),
            (
                "clean_stop_anchor",
                f"Do not switch to {_humanize(form_partner_hint)} or {_humanize(family_partner_hint)}; echo one exact compiler-facing bundle only. {explicit_prompt}",
            ),
            (
                "clean_stop_anchor",
                f"Return exactly one [bundle] that matches these permuted explicit tags and stop after [/bundle]. {explicit_permuted_prompt}",
            ),
            (
                "clean_stop_anchor",
                f"Use only the fields named by these permuted explicit tags; keep style and topology exact; no extra text after [/bundle]. {explicit_permuted_prompt}",
            ),
            (
                "form_minimal_pair",
                f"choose one shared bundle for topic {case.prompt_topic} goal {case.goal} audience {case.audience}; "
                f"prefer {_humanize(form_hint)} inside {_humanize(family_hint)} and not {_humanize(form_partner_hint)}; return bundle only [OUT]",
            ),
            (
                "family_minimal_pair",
                f"choose one shared bundle for topic {case.prompt_topic} goal {case.goal} audience {case.audience}; "
                f"prefer {_humanize(family_hint)} and not {_humanize(family_partner_hint)}; return bundle only [OUT]",
            ),
            (
                "routebook_direct_hint",
                gen19._format_tag_prompt(
                    [
                        ("audience", case.audience),
                        ("goal", case.goal),
                        ("topic", case.prompt_topic),
                        ("emphasis", case.emphasis),
                        ("constraint", constraint),
                        ("content_pack", content_pack),
                        ("decision_hint", f"{family_hint}_{form_hint}"),
                    ]
                ),
            ),
            (
                "routebook_paraphrase",
                f"pick one bundle only for audience {case.audience} topic {case.prompt_topic} goal {case.goal}; "
                f"keep canonical {_humanize(family_hint)} and {_humanize(form_hint)} route; return bundle only [OUT]",
            ),
            (
                "style_topology_bridge",
                f"choose one shared bundle for topic {case.prompt_topic} goal {case.goal} audience {case.audience}; "
                f"use canonical default presentation and family-appropriate fields only; keep it {constraint}; return bundle only [OUT]",
            ),
        ]

        midtrain_prompts = list(pretrain_prompts)
        midtrain_prompts.extend(
            [
                (
                    "routebook_paraphrase",
                    f"choose one shared bundle for topic {case.prompt_topic} with goal {case.goal} for audience {case.audience}; "
                    f"return one bundle only and stop after [/bundle] [OUT]",
                ),
                (
                    "family_minimal_pair",
                    f"choose one bundle for topic {case.prompt_topic} goal {case.goal} audience {case.audience}; "
                    f"keep {_humanize(family_hint)} rather than {_humanize(family_partner_hint)} and keep family-appropriate fields only [OUT]",
                ),
                (
                    "form_minimal_pair",
                    f"choose one bundle for topic {case.prompt_topic} goal {case.goal} audience {case.audience}; "
                    f"keep {_humanize(form_hint)} rather than {_humanize(form_partner_hint)} and keep family-appropriate fields only [OUT]",
                ),
                (
                    "routebook_direct_hint",
                    gen19._format_tag_prompt(
                        [
                            ("topic", case.prompt_topic),
                            ("goal", case.goal),
                            ("audience", case.audience),
                            ("constraint", constraint),
                            ("content_pack", content_pack),
                            ("decision_hint", f"{family_hint}_over_{family_partner_hint}"),
                            ("emphasis", case.emphasis),
                        ]
                    ),
                ),
                (
                    "clean_stop_anchor",
                    f"Return one compiler-facing bundle only; do not replay the prompt; use only family-appropriate fields. {explicit_prompt}",
                ),
            ]
        )

        def _row_and_meta(surface: str, prompt: str) -> tuple[str, dict[str, Any]]:
            text = spec19._row_from_catalog(prompt, output_tokens, base=base)
            meta = {
                "prompt": prompt,
                "prompt_surface": surface,
                "family": bundle.family,
                "layout": bundle.family,
                "form_token": bundle.form,
                "theme": bundle.theme,
                "tone": bundle.tone,
                "density": bundle.density,
                "background": bundle.background,
                "segments": int(bundle.topology.get("segments", 0) or 0),
                "brackets": int(bundle.topology.get("brackets", 0) or 0),
                "cards": int(bundle.topology.get("cards", 0) or 0),
                "stages": int(bundle.topology.get("stages", 0) or 0),
                "arrows": int(bundle.topology.get("arrows", 0) or 0),
                "links": int(bundle.topology.get("links", 0) or 0),
                "footer": int(bundle.topology.get("footer", 0) or 0),
                "terminal": int(bundle.topology.get("terminal", 0) or 0),
                "profile_id": case.profile_id,
                "case_id": case.case_id,
            }
            return text, meta

        for surface, prompt in pretrain_prompts:
            text, meta = _row_and_meta(surface, prompt)
            pretrain_rows.append(text)
            pretrain_meta.append(meta)
        for surface, prompt in midtrain_prompts:
            text, meta = _row_and_meta(surface, prompt)
            midtrain_rows.append(text)
            midtrain_meta.append(meta)

    return (
        _dedupe_preserve(pretrain_rows),
        _dedupe_preserve(midtrain_rows),
        pretrain_meta,
        midtrain_meta,
    )


def materialize_workspace(
    workspace: Path,
    *,
    seed_workspace: Path,
    prefix: str,
    freeze_tokenizer_run: Path,
    source_runs: list[Path],
    weight_quantum: int,
    python_exec: str,
    force: bool,
) -> dict[str, Any]:
    r3d = _load_module(R3D_MATERIALIZER, "materialize_spec19_balanced_coverage_replay_v7")
    spec19 = _load_module(SPEC19_MATERIALIZER, "materialize_spec19_scene_bundle_v7")
    gen19 = _load_module(SPEC19_GENERATOR, "generate_svg_structured_spec19_v7")
    balance = _load_module(CURRICULUM_BALANCE, "spec19_curriculum_balance_v7")
    base = spec19._load_base_module()

    workspace = workspace.expanduser().resolve()
    summary = r3d.materialize_workspace(
        workspace,
        seed_workspace=seed_workspace,
        prefix=prefix,
        freeze_tokenizer_run=freeze_tokenizer_run,
        source_runs=source_runs,
        weight_quantum=weight_quantum,
        python_exec=python_exec,
        force=force,
    )

    render_catalog_path = workspace / "manifests" / "generated" / "structured_atoms" / f"{prefix}_render_catalog.json"
    render_catalog_rows = json.loads(render_catalog_path.read_text(encoding="utf-8"))
    if not isinstance(render_catalog_rows, list):
        raise RuntimeError("spec19 render catalog must be a JSON list")
    catalog_prompt_meta = {
        str(row.get("prompt") or "").strip(): row
        for row in render_catalog_rows
        if isinstance(row, dict) and str(row.get("prompt") or "").strip()
    }

    pretrain_aug_rows, midtrain_aug_rows, pretrain_aug_meta, midtrain_aug_meta = _build_recovery_rows(
        gen19=gen19,
        spec19=spec19,
        base=base,
    )

    pretrain_train_path = workspace / "pretrain" / "train" / f"{prefix}_pretrain_train.txt"
    midtrain_train_path = workspace / "midtrain" / "train" / f"{prefix}_midtrain_train.txt"
    workspace_manifest_path = workspace / "manifests" / f"{prefix}_workspace_manifest.json"
    workspace_manifest = json.loads(workspace_manifest_path.read_text(encoding="utf-8"))
    pretrain_base_rows = _read_lines(pretrain_train_path)
    midtrain_base_rows = _read_lines(midtrain_train_path)
    pretrain_base_counts = dict(
        (workspace_manifest.get("stages") or {}).get("pretrain", {}).get("summary", {}).get("prompt_surface_counts") or {}
    )
    midtrain_base_counts = dict(
        (workspace_manifest.get("stages") or {}).get("midtrain", {}).get("summary", {}).get("prompt_surface_counts") or {}
    )
    pretrain_aug_rows, pretrain_aug_meta, pretrain_balance_report = balance.balance_delta_rows(
        stage="pretrain",
        base_train_rows=len(pretrain_base_rows),
        base_surface_counts=pretrain_base_counts,
        proposed_rows=pretrain_aug_rows,
        proposed_meta=pretrain_aug_meta,
    )
    midtrain_aug_rows, midtrain_aug_meta, midtrain_balance_report = balance.balance_delta_rows(
        stage="midtrain",
        base_train_rows=len(midtrain_base_rows),
        base_surface_counts=midtrain_base_counts,
        proposed_rows=midtrain_aug_rows,
        proposed_meta=midtrain_aug_meta,
    )
    pretrain_train_rows = _dedupe_preserve(pretrain_base_rows + pretrain_aug_rows)
    midtrain_train_rows = _dedupe_preserve(midtrain_base_rows + midtrain_aug_rows)
    _write_lines(pretrain_train_path, pretrain_train_rows)
    _write_lines(midtrain_train_path, midtrain_train_rows)

    train_prompt_set: set[str] = set()
    for row in pretrain_train_rows + midtrain_train_rows:
        prompt = _extract_prompt(row)
        if prompt:
            train_prompt_set.add(prompt)

    holdout_dev_catalog_rows, holdout_test_catalog_rows = spec19._split_holdout_catalog_rows(render_catalog_rows)
    filtered_holdout_dev_catalog_rows, removed_dev_prompts = _filter_catalog_rows(holdout_dev_catalog_rows, train_prompt_set)
    filtered_holdout_test_catalog_rows, removed_test_prompts = _filter_catalog_rows(holdout_test_catalog_rows, train_prompt_set)

    generated_dir = workspace / "manifests" / "generated" / "structured_atoms"
    hidden_seen_prompts, removed_hidden_seen_prompts = _filter_prompts(
        _read_lines(generated_dir / f"{prefix}_hidden_seen_prompts.txt"),
        train_prompt_set,
    )
    hidden_holdout_prompts, removed_hidden_holdout_prompts = _filter_prompts(
        _read_lines(generated_dir / f"{prefix}_hidden_holdout_prompts.txt"),
        train_prompt_set,
    )
    seen_prompts = _dedupe_preserve(_read_lines(generated_dir / f"{prefix}_seen_prompts.txt"))
    holdout_prompt_dev = spec19._prompts_from_catalog(filtered_holdout_dev_catalog_rows)
    holdout_prompt_test = spec19._prompts_from_catalog(filtered_holdout_test_catalog_rows)
    holdout_prompt_rows = _dedupe_preserve(holdout_prompt_dev + holdout_prompt_test)

    dev_rows = spec19._rows_from_catalog(filtered_holdout_dev_catalog_rows, base=base)
    test_rows = spec19._rows_from_catalog(filtered_holdout_test_catalog_rows, base=base)
    _write_lines(workspace / "pretrain" / "dev" / f"{prefix}_pretrain_dev.txt", dev_rows)
    _write_lines(workspace / "pretrain" / "test" / f"{prefix}_pretrain_test.txt", test_rows)
    _write_lines(workspace / "midtrain" / "dev" / f"{prefix}_midtrain_dev.txt", dev_rows)
    _write_lines(workspace / "midtrain" / "test" / f"{prefix}_midtrain_test.txt", test_rows)
    _write_lines(workspace / "holdout" / f"{prefix}_holdout_prompts.txt", holdout_prompt_rows)
    _write_lines(workspace / "holdout" / f"{prefix}_holdout_prompts_dev.txt", holdout_prompt_dev)
    _write_lines(workspace / "holdout" / f"{prefix}_holdout_prompts_test.txt", holdout_prompt_test)
    _write_lines(workspace / "holdout" / f"{prefix}_seen_prompts.txt", seen_prompts)
    _write_lines(workspace / "holdout" / f"{prefix}_hidden_seen_prompts.txt", hidden_seen_prompts)
    _write_lines(workspace / "holdout" / f"{prefix}_hidden_holdout_prompts.txt", hidden_holdout_prompts)

    tokenizer_dir = workspace / "tokenizer"
    tokenizer_corpus_rows = _dedupe_preserve(_read_lines(tokenizer_dir / f"{prefix}_tokenizer_corpus.txt") + pretrain_aug_rows + midtrain_aug_rows)
    _write_lines(tokenizer_dir / f"{prefix}_tokenizer_corpus.txt", tokenizer_corpus_rows)
    _write_lines(tokenizer_dir / f"{prefix}_seen_prompts.txt", seen_prompts)
    _write_lines(tokenizer_dir / f"{prefix}_holdout_prompts.txt", holdout_prompt_rows)
    _write_lines(tokenizer_dir / f"{prefix}_hidden_seen_prompts.txt", hidden_seen_prompts)
    _write_lines(tokenizer_dir / f"{prefix}_hidden_holdout_prompts.txt", hidden_holdout_prompts)

    contracts_dir = workspace / "contracts"
    probe_contract_dst = contracts_dir / f"{prefix}_probe_report_contract.json"
    probe_cmd = [
        python_exec,
        str(ROOT / "version" / "v7" / "scripts" / "build_spec19_probe_contract_v7.py"),
        "--run",
        str(workspace),
        "--prefix",
        prefix,
        "--output",
        str(probe_contract_dst),
        "--per-split",
        "12",
        "--hidden-per-split",
        "6",
    ]
    probe_proc = base.subprocess.run(probe_cmd, cwd=ROOT, stdout=base.subprocess.PIPE, stderr=base.subprocess.STDOUT, text=True)
    if probe_proc.returncode != 0:
        raise RuntimeError(f"spec19 probe contract build failed (rc={probe_proc.returncode}):\n{probe_proc.stdout.strip()}")
    base.shutil.copy2(probe_contract_dst, tokenizer_dir / f"{prefix}_probe_report_contract.json")

    pretrain_aug_meta_map = {str(item["prompt"]): item for item in pretrain_aug_meta if str(item.get("prompt") or "")}
    midtrain_aug_meta_map = {str(item["prompt"]): item for item in midtrain_aug_meta if str(item.get("prompt") or "")}
    pretrain_prompt_meta = dict(catalog_prompt_meta)
    pretrain_prompt_meta.update(pretrain_aug_meta_map)
    midtrain_prompt_meta = dict(catalog_prompt_meta)
    midtrain_prompt_meta.update(midtrain_aug_meta_map)

    stage_pretrain_summary = r3d.r3c._summarize_stage(pretrain_train_rows, pretrain_prompt_meta, spec19=spec19)
    stage_midtrain_summary = r3d.r3c._summarize_stage(midtrain_train_rows, midtrain_prompt_meta, spec19=spec19)

    manifests_dir = workspace / "manifests"
    mixture_manifest_path = manifests_dir / f"{prefix}_mixture_manifest.json"
    coherent_manifest_path = manifests_dir / f"{prefix}_coherent_replay_manifest.json"
    workspace_manifest = json.loads(workspace_manifest_path.read_text(encoding="utf-8"))
    mixture_manifest = json.loads(mixture_manifest_path.read_text(encoding="utf-8"))
    coherent_manifest = json.loads(coherent_manifest_path.read_text(encoding="utf-8"))

    recovery_manifest = {
        "format": "ck.spec19_route_recovery.v1",
        "line": LINE_NAME,
        "workspace": str(workspace),
        "source_runs": [str(path.expanduser().resolve()) for path in source_runs],
        "stages": {
            "pretrain": {
                "added_rows": len(pretrain_aug_rows),
                "prompt_surface_counts": dict(sorted(Counter(item["prompt_surface"] for item in pretrain_aug_meta).items())),
                "balance_policy": pretrain_balance_report,
            },
            "midtrain": {
                "added_rows": len(midtrain_aug_rows),
                "prompt_surface_counts": dict(sorted(Counter(item["prompt_surface"] for item in midtrain_aug_meta).items())),
                "balance_policy": midtrain_balance_report,
            },
        },
        "eval_collision_filter": {
            "removed_dev_prompts": removed_dev_prompts,
            "removed_test_prompts": removed_test_prompts,
            "removed_hidden_seen_prompts": removed_hidden_seen_prompts,
            "removed_hidden_holdout_prompts": removed_hidden_holdout_prompts,
            "removed_total": len(removed_dev_prompts) + len(removed_test_prompts) + len(removed_hidden_seen_prompts) + len(removed_hidden_holdout_prompts),
        },
    }
    base._write_json(manifests_dir / f"{prefix}_route_recovery_manifest.json", recovery_manifest)

    coherent_manifest["line"] = LINE_NAME
    coherent_manifest["train_prompt_count"] = len(train_prompt_set)
    coherent_manifest["eval_collision_filter"] = recovery_manifest["eval_collision_filter"]
    coherent_manifest["stages"]["pretrain"]["rows"] = len(pretrain_train_rows)
    coherent_manifest["stages"]["pretrain"]["unique_rows"] = len(set(pretrain_train_rows))
    coherent_manifest["stages"]["pretrain"]["summary"] = stage_pretrain_summary
    coherent_manifest["stages"]["midtrain"]["rows"] = len(midtrain_train_rows)
    coherent_manifest["stages"]["midtrain"]["unique_rows"] = len(set(midtrain_train_rows))
    coherent_manifest["stages"]["midtrain"]["summary"] = stage_midtrain_summary
    base._write_json(coherent_manifest_path, coherent_manifest)

    workspace_manifest["line"] = LINE_NAME
    workspace_manifest["route_recovery_manifest"] = f"manifests/{prefix}_route_recovery_manifest.json"
    workspace_manifest["stages"]["pretrain"]["counts"]["train_rows"] = len(pretrain_train_rows)
    workspace_manifest["stages"]["pretrain"]["counts"]["dev_rows"] = len(dev_rows)
    workspace_manifest["stages"]["pretrain"]["counts"]["test_rows"] = len(test_rows)
    workspace_manifest["stages"]["pretrain"]["summary"] = stage_pretrain_summary
    workspace_manifest["stages"]["midtrain"]["counts"]["train_rows"] = len(midtrain_train_rows)
    workspace_manifest["stages"]["midtrain"]["counts"]["dev_rows"] = len(dev_rows)
    workspace_manifest["stages"]["midtrain"]["counts"]["test_rows"] = len(test_rows)
    workspace_manifest["stages"]["midtrain"]["summary"] = stage_midtrain_summary
    workspace_manifest["stages"]["midtrain"]["notes"] = (
        "R3e keeps replay-heavy cumulative coverage from prior winner-line corpora and adds broader family, form, and syntax recovery "
        "rows around the surviving r3d miss classes without teaching exact held-out prompts."
    )
    base._write_json(workspace_manifest_path, workspace_manifest)

    mixture_manifest["line"] = LINE_NAME
    mixture_manifest["stages"]["pretrain"]["route_recovery_rows"] = len(pretrain_aug_rows)
    mixture_manifest["stages"]["pretrain"]["materialized_prompt_surface_counts"] = stage_pretrain_summary["prompt_surface_counts"]
    mixture_manifest["stages"]["pretrain"]["train_rows"] = len(pretrain_train_rows)
    mixture_manifest["stages"]["pretrain"]["balance_policy"] = pretrain_balance_report
    mixture_manifest["stages"]["midtrain"]["route_recovery_rows"] = len(midtrain_aug_rows)
    mixture_manifest["stages"]["midtrain"]["materialized_prompt_surface_counts"] = stage_midtrain_summary["prompt_surface_counts"]
    mixture_manifest["stages"]["midtrain"]["train_rows"] = len(midtrain_train_rows)
    mixture_manifest["stages"]["midtrain"]["balance_policy"] = midtrain_balance_report
    base._write_json(mixture_manifest_path, mixture_manifest)

    return {
        "pretrain_train_rows": len(pretrain_train_rows),
        "midtrain_train_rows": len(midtrain_train_rows),
        "recovery_pretrain_rows": len(pretrain_aug_rows),
        "recovery_midtrain_rows": len(midtrain_aug_rows),
        "holdout_dev_rows": len(dev_rows),
        "holdout_test_rows": len(test_rows),
        "hidden_seen_prompts": len(hidden_seen_prompts),
        "hidden_holdout_prompts": len(hidden_holdout_prompts),
        "train_prompt_count": len(train_prompt_set),
        "base_summary": summary,
    }


def main() -> int:
    spec19 = _load_module(SPEC19_MATERIALIZER, "materialize_spec19_scene_bundle_v7")
    base = spec19._load_base_module()
    import argparse

    ap = argparse.ArgumentParser(description="Materialize a replay-heavy spec19 workspace with family/form/syntax recovery coverage")
    ap.add_argument("--workspace", required=True, type=Path, help="Destination workspace")
    ap.add_argument("--seed-workspace", default=str(base.DEFAULT_SEED_WORKSPACE), type=Path, help="Seed spec workspace template to copy")
    ap.add_argument("--prefix", default="spec19_scene_bundle", help="Dataset prefix")
    ap.add_argument("--freeze-tokenizer-run", required=True, type=Path, help="Run directory whose tokenizer should be copied unchanged")
    ap.add_argument("--source-run", action="append", dest="source_runs", type=Path, required=True, help="Completed run whose stage train rows should seed the cumulative curriculum")
    ap.add_argument("--weight-quantum", type=int, default=5, help="Accepted for shared launcher compatibility; recorded in manifests")
    ap.add_argument("--python-exec", default=sys.executable, help="Python executable for generators")
    ap.add_argument("--force", action="store_true", help="Replace workspace if it exists")
    args = ap.parse_args()

    summary = materialize_workspace(
        args.workspace,
        seed_workspace=args.seed_workspace,
        prefix=args.prefix,
        freeze_tokenizer_run=args.freeze_tokenizer_run,
        source_runs=list(args.source_runs or []),
        weight_quantum=int(args.weight_quantum),
        python_exec=str(args.python_exec),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
