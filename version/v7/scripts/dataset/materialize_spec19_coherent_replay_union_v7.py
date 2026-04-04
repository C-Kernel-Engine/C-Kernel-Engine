#!/usr/bin/env python3
"""Materialize a replay-heavy coherent spec19 workspace from prior train corpora."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SPEC19_MATERIALIZER = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec19_scene_bundle_v7.py"
SPEC19_BLUEPRINT = ROOT / "version" / "v7" / "reports" / "spec19_curriculum_blueprint.json"
LINE_NAME = "spec19_coherent_replay_union"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_dataset_root(source_run: Path) -> Path:
    dataset_root = source_run / "dataset"
    return dataset_root if dataset_root.exists() else source_run


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def _load_stage_rows(source_dataset: Path, prefix: str, stage: str) -> list[str]:
    return _read_lines(source_dataset / stage / "train" / f"{prefix}_{stage}_train.txt")


def _merge_stage_rows(
    *,
    source_runs: list[Path],
    prefix: str,
    stage: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    merged: list[str] = []
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_run in source_runs:
        dataset_root = _source_dataset_root(source_run)
        rows = _load_stage_rows(dataset_root, prefix, stage)
        unique_rows = len(set(rows))
        new_unique = 0
        for row in rows:
            text = str(row or "").strip()
            if not text:
                continue
            if text not in seen:
                seen.add(text)
                merged.append(text)
                new_unique += 1
        summaries.append(
            {
                "source_run": str(source_run),
                "dataset_root": str(dataset_root),
                "rows": len(rows),
                "unique_rows": unique_rows,
                "new_unique_rows_added": new_unique,
            }
        )
    return merged, summaries


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
    spec19 = _load_module(SPEC19_MATERIALIZER, "materialize_spec19_scene_bundle_v7")
    base = spec19._load_base_module()

    workspace = workspace.expanduser().resolve()
    seed_workspace = seed_workspace.expanduser().resolve()
    source_runs = [path.expanduser().resolve() for path in source_runs]
    if not seed_workspace.exists():
        raise FileNotFoundError(f"seed workspace not found: {seed_workspace}")
    if not source_runs:
        raise RuntimeError("at least one --source-run is required")

    base._copy_seed_workspace(seed_workspace, workspace, force=force)
    base._ensure_split_dirs(workspace)

    generated_dir = workspace / "manifests" / "generated" / "structured_atoms"
    generated_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exec,
        str(ROOT / "version" / "v7" / "scripts" / "generate_svg_structured_spec19_v7.py"),
        "--out-dir",
        str(generated_dir),
        "--prefix",
        prefix,
    ]
    proc = base.subprocess.run(cmd, cwd=ROOT, stdout=base.subprocess.PIPE, stderr=base.subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"spec19 generator failed (rc={proc.returncode}):\n{proc.stdout.strip()}")

    render_catalog_rows = json.loads((generated_dir / f"{prefix}_render_catalog.json").read_text(encoding="utf-8"))
    if not isinstance(render_catalog_rows, list):
        raise RuntimeError("spec19 render catalog must be a JSON list")
    catalog_by_prompt = {
        str(row.get("prompt") or "").strip(): row
        for row in render_catalog_rows
        if isinstance(row, dict) and str(row.get("prompt") or "").strip()
    }

    original_seen_prompts = _read_lines(generated_dir / f"{prefix}_seen_prompts.txt")
    original_hidden_seen_prompts = _read_lines(generated_dir / f"{prefix}_hidden_seen_prompts.txt")
    original_hidden_holdout_prompts = _read_lines(generated_dir / f"{prefix}_hidden_holdout_prompts.txt")
    holdout_dev_catalog_rows, holdout_test_catalog_rows = spec19._split_holdout_catalog_rows(render_catalog_rows)
    original_holdout_dev_prompts = spec19._prompts_from_catalog(holdout_dev_catalog_rows)
    original_holdout_test_prompts = spec19._prompts_from_catalog(holdout_test_catalog_rows)

    pretrain_train_rows, pretrain_sources = _merge_stage_rows(source_runs=source_runs, prefix=prefix, stage="pretrain")
    midtrain_train_rows, midtrain_sources = _merge_stage_rows(source_runs=source_runs, prefix=prefix, stage="midtrain")

    train_prompt_set: set[str] = set()
    for row in pretrain_train_rows + midtrain_train_rows:
        prompt = _extract_prompt(row)
        if prompt:
            train_prompt_set.add(prompt)

    filtered_holdout_dev_catalog_rows, removed_dev_prompts = _filter_catalog_rows(holdout_dev_catalog_rows, train_prompt_set)
    filtered_holdout_test_catalog_rows, removed_test_prompts = _filter_catalog_rows(holdout_test_catalog_rows, train_prompt_set)
    hidden_seen_prompts, removed_hidden_seen_prompts = _filter_prompts(original_hidden_seen_prompts, train_prompt_set)
    hidden_holdout_prompts, removed_hidden_holdout_prompts = _filter_prompts(original_hidden_holdout_prompts, train_prompt_set)

    seen_prompts = _dedupe_preserve(original_seen_prompts)
    holdout_prompt_dev = spec19._prompts_from_catalog(filtered_holdout_dev_catalog_rows)
    holdout_prompt_test = spec19._prompts_from_catalog(filtered_holdout_test_catalog_rows)
    holdout_prompt_rows = _dedupe_preserve(holdout_prompt_dev + holdout_prompt_test)

    dev_rows = spec19._rows_from_catalog(filtered_holdout_dev_catalog_rows, base=base)
    test_rows = spec19._rows_from_catalog(filtered_holdout_test_catalog_rows, base=base)

    pretrain_train = workspace / "pretrain" / "train" / f"{prefix}_pretrain_train.txt"
    pretrain_dev = workspace / "pretrain" / "dev" / f"{prefix}_pretrain_dev.txt"
    pretrain_test = workspace / "pretrain" / "test" / f"{prefix}_pretrain_test.txt"
    base._write_lines(pretrain_train, pretrain_train_rows)
    base._write_lines(pretrain_dev, dev_rows)
    base._write_lines(pretrain_test, test_rows)

    midtrain_train = workspace / "midtrain" / "train" / f"{prefix}_midtrain_train.txt"
    midtrain_dev = workspace / "midtrain" / "dev" / f"{prefix}_midtrain_dev.txt"
    midtrain_test = workspace / "midtrain" / "test" / f"{prefix}_midtrain_test.txt"
    base._write_lines(midtrain_train, midtrain_train_rows)
    base._write_lines(midtrain_dev, dev_rows)
    base._write_lines(midtrain_test, test_rows)

    for split in ("train", "dev", "test"):
        base._write_lines(workspace / "sft" / split / f"{prefix}_sft_{split}.txt", [])

    holdout_dir = workspace / "holdout"
    base._write_lines(holdout_dir / f"{prefix}_holdout_prompts.txt", holdout_prompt_rows)
    base._write_lines(holdout_dir / f"{prefix}_holdout_prompts_dev.txt", holdout_prompt_dev)
    base._write_lines(holdout_dir / f"{prefix}_holdout_prompts_test.txt", holdout_prompt_test)
    base._write_lines(holdout_dir / f"{prefix}_seen_prompts.txt", seen_prompts)
    base._write_lines(holdout_dir / f"{prefix}_hidden_seen_prompts.txt", hidden_seen_prompts)
    base._write_lines(holdout_dir / f"{prefix}_hidden_holdout_prompts.txt", hidden_holdout_prompts)

    tokenizer_dir = workspace / "tokenizer"
    frozen = spec19._copy_frozen_tokenizer(freeze_tokenizer_run, tokenizer_dir, prefix, base=base)
    tokenizer_corpus_rows = spec19._tokenizer_corpus_rows(render_catalog_rows, base=base)
    base.shutil.copy2(generated_dir / f"{prefix}_render_catalog.json", tokenizer_dir / f"{prefix}_render_catalog.json")
    base._write_lines(tokenizer_dir / f"{prefix}_tokenizer_corpus.txt", tokenizer_corpus_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_seen_prompts.txt", seen_prompts)
    base._write_lines(tokenizer_dir / f"{prefix}_holdout_prompts.txt", holdout_prompt_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_hidden_seen_prompts.txt", hidden_seen_prompts)
    base._write_lines(tokenizer_dir / f"{prefix}_hidden_holdout_prompts.txt", hidden_holdout_prompts)

    contracts_dir = workspace / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    base.shutil.copy2(generated_dir / f"{prefix}_eval_contract.json", contracts_dir / "eval_contract.svg.v1.json")
    base.shutil.copy2(SPEC19_BLUEPRINT, contracts_dir / "spec19_curriculum_blueprint.json")

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

    tokenizer_manifest = {
        "format": "ck.structured_svg_atoms.tokenizer_manifest.v1",
        "line": LINE_NAME,
        "workspace": str(workspace),
        "frozen_from_run": str(frozen["frozen_from_run"]),
        "artifacts": {
            "tokenizer_json": base._rel(Path(frozen["tokenizer_json"]), workspace),
            "tokenizer_bin": base._rel(Path(frozen["tokenizer_bin"]), workspace),
            "render_catalog": base._rel(tokenizer_dir / f"{prefix}_render_catalog.json", workspace),
            "reserved_control_tokens": base._rel(Path(frozen["reserved_control_tokens"]), workspace),
            "tokenizer_corpus": base._rel(tokenizer_dir / f"{prefix}_tokenizer_corpus.txt", workspace),
            "seen_prompts": base._rel(tokenizer_dir / f"{prefix}_seen_prompts.txt", workspace),
            "holdout_prompts": base._rel(tokenizer_dir / f"{prefix}_holdout_prompts.txt", workspace),
            "hidden_seen_prompts": base._rel(tokenizer_dir / f"{prefix}_hidden_seen_prompts.txt", workspace),
            "hidden_holdout_prompts": base._rel(tokenizer_dir / f"{prefix}_hidden_holdout_prompts.txt", workspace),
        },
    }
    base._write_json(tokenizer_dir / f"{prefix}_tokenizer_manifest.json", tokenizer_manifest)

    def _stage_summary(rows: list[str]) -> dict[str, Any]:
        materialized_rows: list[dict[str, Any]] = []
        missing_prompt_rows = 0
        for text in rows:
            prompt = _extract_prompt(text)
            catalog_row = catalog_by_prompt.get(prompt)
            row_doc = dict(catalog_row) if isinstance(catalog_row, dict) else {}
            row_doc["_stage_row_text"] = text
            if not row_doc:
                missing_prompt_rows += 1
            materialized_rows.append(row_doc)
        summary = spec19._summarize_stage_rows(materialized_rows)
        summary["missing_prompt_rows"] = int(missing_prompt_rows)
        return summary

    coherent_manifest = {
        "format": "ck.spec19_coherent_replay_union.v1",
        "line": LINE_NAME,
        "workspace": str(workspace),
        "prefix": prefix,
        "frozen_tokenizer_run": str(freeze_tokenizer_run.expanduser().resolve()),
        "source_runs": [str(path) for path in source_runs],
        "train_prompt_count": len(train_prompt_set),
        "eval_collision_filter": {
            "removed_dev_prompts": removed_dev_prompts,
            "removed_test_prompts": removed_test_prompts,
            "removed_hidden_seen_prompts": removed_hidden_seen_prompts,
            "removed_hidden_holdout_prompts": removed_hidden_holdout_prompts,
            "removed_total": len(removed_dev_prompts) + len(removed_test_prompts) + len(removed_hidden_seen_prompts) + len(removed_hidden_holdout_prompts),
        },
        "stages": {
            "pretrain": {
                "sources": pretrain_sources,
                "rows": len(pretrain_train_rows),
                "unique_rows": len(set(pretrain_train_rows)),
                "summary": _stage_summary(pretrain_train_rows),
            },
            "midtrain": {
                "sources": midtrain_sources,
                "rows": len(midtrain_train_rows),
                "unique_rows": len(set(midtrain_train_rows)),
                "summary": _stage_summary(midtrain_train_rows),
            },
        },
    }

    workspace_manifest = {
        "format": "ck.structured_svg_atoms.workspace_materialization.v1",
        "source_seed_workspace": str(seed_workspace),
        "workspace": str(workspace),
        "line": LINE_NAME,
        "generator": str(ROOT / "version" / "v7" / "scripts" / "generate_svg_structured_spec19_v7.py"),
        "prefix": prefix,
        "frozen_tokenizer_run": str(freeze_tokenizer_run.expanduser().resolve()),
        "weight_quantum": int(weight_quantum),
        "train_summary": spec19._summarize_catalog(render_catalog_rows, split="train"),
        "holdout_summary": spec19._summarize_catalog(render_catalog_rows, split="holdout"),
        "hidden_train_summary": spec19._summarize_catalog(render_catalog_rows, split="probe_hidden_train"),
        "hidden_holdout_summary": spec19._summarize_catalog(render_catalog_rows, split="probe_hidden_holdout"),
        "coherent_manifest": f"manifests/{prefix}_coherent_replay_manifest.json",
        "stages": {
            "pretrain": {
                "train": base._rel(pretrain_train, workspace),
                "dev": base._rel(pretrain_dev, workspace),
                "test": base._rel(pretrain_test, workspace),
                "counts": {
                    "train_rows": len(pretrain_train_rows),
                    "dev_rows": len(dev_rows),
                    "test_rows": len(test_rows),
                },
                "summary": coherent_manifest["stages"]["pretrain"]["summary"],
            },
            "midtrain": {
                "train": base._rel(midtrain_train, workspace),
                "dev": base._rel(midtrain_dev, workspace),
                "test": base._rel(midtrain_test, workspace),
                "counts": {
                    "train_rows": len(midtrain_train_rows),
                    "dev_rows": len(dev_rows),
                    "test_rows": len(test_rows),
                },
                "summary": coherent_manifest["stages"]["midtrain"]["summary"],
                "notes": "Chronological spec19 train corpora are unioned and deduped to preserve replay breadth while keeping train-eval prompt collisions out of dev/test/hidden probe pools.",
            },
        },
    }

    manifests_dir = workspace / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    base._write_json(manifests_dir / f"{prefix}_workspace_manifest.json", workspace_manifest)
    base._write_json(manifests_dir / f"{prefix}_coherent_replay_manifest.json", coherent_manifest)
    mixture_manifest = {
        "format": "ck.named_mixture_manifest.v1",
        "line": LINE_NAME,
        "workspace": str(workspace),
        "weight_quantum": int(weight_quantum),
        "stages": {
            "pretrain": {
                "source_rows": pretrain_sources,
                "materialized_prompt_surface_counts": coherent_manifest["stages"]["pretrain"]["summary"]["prompt_surface_counts"],
                "train_rows": len(pretrain_train_rows),
            },
            "midtrain": {
                "source_rows": midtrain_sources,
                "materialized_prompt_surface_counts": coherent_manifest["stages"]["midtrain"]["summary"]["prompt_surface_counts"],
                "train_rows": len(midtrain_train_rows),
            },
        },
    }
    base._write_json(manifests_dir / f"{prefix}_mixture_manifest.json", mixture_manifest)

    return {
        "pretrain_train_rows": len(pretrain_train_rows),
        "midtrain_train_rows": len(midtrain_train_rows),
        "holdout_dev_rows": len(dev_rows),
        "holdout_test_rows": len(test_rows),
        "hidden_seen_prompts": len(hidden_seen_prompts),
        "hidden_holdout_prompts": len(hidden_holdout_prompts),
        "train_prompt_count": len(train_prompt_set),
    }


def main() -> int:
    spec19 = _load_module(SPEC19_MATERIALIZER, "materialize_spec19_scene_bundle_v7")
    base = spec19._load_base_module()
    import argparse

    ap = argparse.ArgumentParser(description="Materialize a replay-heavy coherent spec19 workspace from prior train corpora")
    ap.add_argument("--workspace", required=True, type=Path, help="Destination workspace")
    ap.add_argument("--seed-workspace", default=str(base.DEFAULT_SEED_WORKSPACE), type=Path, help="Seed spec workspace template to copy")
    ap.add_argument("--prefix", default="spec19_scene_bundle", help="Dataset prefix")
    ap.add_argument("--freeze-tokenizer-run", required=True, type=Path, help="Run directory whose tokenizer should be copied unchanged")
    ap.add_argument("--source-run", action="append", dest="source_runs", type=Path, required=True, help="Completed run whose stage train rows should be unioned into the coherent curriculum")
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
