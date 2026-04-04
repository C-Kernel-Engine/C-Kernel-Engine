#!/usr/bin/env python3
"""Materialize a replay-anchored spec19 delta workspace from probe misses."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SPEC19_MATERIALIZER = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec19_scene_bundle_v7.py"
SPEC19_BLUEPRINT = ROOT / "version" / "v7" / "reports" / "spec19_curriculum_blueprint.json"


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


def _sample_evenly(rows: list[str], count: int) -> list[str]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return list(rows)
    step = float(len(rows)) / float(count)
    out: list[str] = []
    seen: set[int] = set()
    for idx in range(count):
        pos = min(len(rows) - 1, int(idx * step))
        while pos in seen and pos + 1 < len(rows):
            pos += 1
        if pos in seen:
            pos = max(0, pos - 1)
            while pos in seen and pos > 0:
                pos -= 1
        seen.add(pos)
        out.append(rows[pos])
    return out


def _interleave(replay_rows: list[str], delta_rows: list[str], replay_to_delta_ratio: int) -> list[str]:
    if not delta_rows:
        return list(replay_rows)
    replay_ratio = max(1, int(replay_to_delta_ratio))
    out: list[str] = []
    replay_idx = 0
    for delta in delta_rows:
        out.append(delta)
        for _ in range(replay_ratio):
            if replay_idx >= len(replay_rows):
                break
            out.append(replay_rows[replay_idx])
            replay_idx += 1
    out.extend(replay_rows[replay_idx:])
    return out


def _find_catalog(dataset_root: Path, prefix: str) -> Path:
    path = dataset_root / "manifests" / "generated" / "structured_atoms" / f"{prefix}_render_catalog.json"
    if not path.exists():
        raise FileNotFoundError(f"render catalog missing: {path}")
    return path


def _find_probe_report(source_run: Path, override: Path | None) -> Path:
    if override is not None:
        path = override.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"probe report missing: {path}")
        return path
    default = source_run / "spec19_probe_report.json"
    if not default.exists():
        raise FileNotFoundError(f"probe report missing: {default}")
    return default


def _build_delta_rows(
    *,
    probe_report_path: Path,
    render_catalog_rows: list[dict[str, Any]],
    row_from_catalog,
) -> tuple[list[str], list[dict[str, Any]]]:
    doc = json.loads(probe_report_path.read_text(encoding="utf-8"))
    results = doc.get("results")
    if not isinstance(results, list):
        raise RuntimeError("probe report missing results list")
    catalog_by_prompt = {
        str(row.get("prompt") or "").strip(): row
        for row in render_catalog_rows
        if isinstance(row, dict) and str(row.get("prompt") or "").strip()
    }
    delta_rows: list[str] = []
    delta_manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in results:
        if not isinstance(row, dict) or bool(row.get("exact_match")):
            continue
        prompt = str(row.get("prompt") or "").strip()
        expected_output = str(row.get("expected_output") or "").strip()
        if not prompt or not expected_output:
            continue
        text = row_from_catalog(prompt, expected_output)
        if not text or text in seen:
            continue
        seen.add(text)
        catalog_row = catalog_by_prompt.get(prompt) or {}
        delta_rows.append(text)
        delta_manifest.append(
            {
                "id": str(row.get("id") or ""),
                "split": str(row.get("split") or ""),
                "label": str(row.get("label") or ""),
                "prompt": prompt,
                "expected_output": expected_output,
                "prompt_surface": str(catalog_row.get("prompt_surface") or ""),
                "family": str(catalog_row.get("family") or catalog_row.get("layout") or ""),
                "profile_id": str(catalog_row.get("profile_id") or catalog_row.get("case_id") or ""),
            }
        )
    return delta_rows, delta_manifest


def materialize_workspace(
    workspace: Path,
    *,
    seed_workspace: Path,
    prefix: str,
    freeze_tokenizer_run: Path,
    source_run: Path,
    probe_report: Path | None,
    replay_to_delta_ratio: int,
    weight_quantum: int,
    python_exec: str,
    force: bool,
) -> dict[str, Any]:
    spec19 = _load_module(SPEC19_MATERIALIZER, "materialize_spec19_scene_bundle_v7")
    base = spec19._load_base_module()
    workspace = workspace.expanduser().resolve()
    seed_workspace = seed_workspace.expanduser().resolve()
    source_run = source_run.expanduser().resolve()
    source_dataset = _source_dataset_root(source_run)
    if not source_dataset.exists():
        raise FileNotFoundError(f"source dataset root not found: {source_dataset}")

    base._copy_seed_workspace(seed_workspace, workspace, force=force)
    base._ensure_split_dirs(workspace)

    generated_dir = workspace / "manifests" / "generated" / "structured_atoms"
    generated_dir.mkdir(parents=True, exist_ok=True)
    render_catalog_src = _find_catalog(source_dataset, prefix)
    render_catalog_dst = generated_dir / f"{prefix}_render_catalog.json"
    base.shutil.copy2(render_catalog_src, render_catalog_dst)
    render_catalog_rows = json.loads(render_catalog_dst.read_text(encoding="utf-8"))
    if not isinstance(render_catalog_rows, list):
        raise RuntimeError("spec19 render catalog must be a JSON list")

    eval_contract_src = source_dataset / "contracts" / "eval_contract.svg.v1.json"
    if eval_contract_src.exists():
        eval_contract_dst = generated_dir / f"{prefix}_eval_contract.json"
        base.shutil.copy2(eval_contract_src, eval_contract_dst)
    else:
        probe_cmd = [
            python_exec,
            str(ROOT / "version" / "v7" / "scripts" / "generate_svg_structured_spec19_v7.py"),
            "--out-dir",
            str(generated_dir),
            "--prefix",
            prefix,
        ]
        proc = base.subprocess.run(probe_cmd, cwd=ROOT, stdout=base.subprocess.PIPE, stderr=base.subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"spec19 generator failed while reconstructing eval contract (rc={proc.returncode}):\n{proc.stdout.strip()}")

    seen_prompt_rows = _read_lines(source_dataset / "holdout" / f"{prefix}_seen_prompts.txt")
    holdout_prompt_rows = _read_lines(source_dataset / "holdout" / f"{prefix}_holdout_prompts.txt")
    holdout_prompt_dev = _read_lines(source_dataset / "holdout" / f"{prefix}_holdout_prompts_dev.txt")
    holdout_prompt_test = _read_lines(source_dataset / "holdout" / f"{prefix}_holdout_prompts_test.txt")
    hidden_seen_prompt_rows = _read_lines(source_dataset / "holdout" / f"{prefix}_hidden_seen_prompts.txt")
    hidden_holdout_prompt_rows = _read_lines(source_dataset / "holdout" / f"{prefix}_hidden_holdout_prompts.txt")

    holdout_dev_catalog_rows, holdout_test_catalog_rows = spec19._split_holdout_catalog_rows(render_catalog_rows)
    dev_rows = spec19._rows_from_catalog(holdout_dev_catalog_rows, base=base)
    test_rows = spec19._rows_from_catalog(holdout_test_catalog_rows, base=base)

    probe_report_path = _find_probe_report(source_run, probe_report)
    delta_rows, delta_manifest = _build_delta_rows(
        probe_report_path=probe_report_path,
        render_catalog_rows=render_catalog_rows,
        row_from_catalog=lambda prompt, output: spec19._row_from_catalog(prompt, output, base=base),
    )
    delta_set = set(delta_rows)
    if not delta_rows:
        raise RuntimeError("probe report produced no delta rows")

    source_pretrain_rows = [
        row for row in _read_lines(source_dataset / "pretrain" / "train" / f"{prefix}_pretrain_train.txt")
        if row not in delta_set
    ]
    source_midtrain_rows = [
        row for row in _read_lines(source_dataset / "midtrain" / "train" / f"{prefix}_midtrain_train.txt")
        if row not in delta_set
    ]
    replay_count = len(delta_rows) * max(1, int(replay_to_delta_ratio))
    pretrain_replay_rows = _sample_evenly(source_pretrain_rows, replay_count)
    midtrain_replay_rows = _sample_evenly(source_midtrain_rows, replay_count)
    pretrain_train_rows = _interleave(pretrain_replay_rows, delta_rows, replay_to_delta_ratio)
    midtrain_train_rows = _interleave(midtrain_replay_rows, delta_rows, replay_to_delta_ratio)

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
    base._write_lines(holdout_dir / f"{prefix}_seen_prompts.txt", seen_prompt_rows)
    base._write_lines(holdout_dir / f"{prefix}_hidden_seen_prompts.txt", hidden_seen_prompt_rows)
    base._write_lines(holdout_dir / f"{prefix}_hidden_holdout_prompts.txt", hidden_holdout_prompt_rows)

    tokenizer_dir = workspace / "tokenizer"
    frozen = spec19._copy_frozen_tokenizer(freeze_tokenizer_run, tokenizer_dir, prefix, base=base)
    tokenizer_corpus_rows = spec19._tokenizer_corpus_rows(render_catalog_rows, base=base)
    base.shutil.copy2(render_catalog_dst, tokenizer_dir / f"{prefix}_render_catalog.json")
    base._write_lines(tokenizer_dir / f"{prefix}_tokenizer_corpus.txt", tokenizer_corpus_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_seen_prompts.txt", seen_prompt_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_holdout_prompts.txt", holdout_prompt_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_hidden_seen_prompts.txt", hidden_seen_prompt_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_hidden_holdout_prompts.txt", hidden_holdout_prompt_rows)

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
        "line": "spec19_probe_miss_delta",
        "workspace": str(workspace),
        "frozen_from_run": str(frozen["frozen_from_run"]),
        "artifacts": {
            "tokenizer_json": base._rel(Path(frozen["tokenizer_json"]), workspace),
            "tokenizer_bin": base._rel(Path(frozen["tokenizer_bin"]), workspace),
            "render_catalog": base._rel(tokenizer_dir / f"{prefix}_render_catalog.json", workspace),
            "reserved_control_tokens": base._rel(Path(frozen["reserved_control_tokens"]), workspace),
            "tokenizer_corpus": base._rel(tokenizer_dir / f"{prefix}_tokenizer_corpus.txt", workspace),
        },
    }
    base._write_json(tokenizer_dir / f"{prefix}_tokenizer_manifest.json", tokenizer_manifest)

    manifests_dir = workspace / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    base._write_json(
        manifests_dir / f"{prefix}_delta_manifest.json",
        {
            "format": "ck.spec19_probe_miss_delta.v1",
            "workspace": str(workspace),
            "source_run": str(source_run),
            "probe_report": str(probe_report_path),
            "replay_to_delta_ratio": int(replay_to_delta_ratio),
            "weight_quantum_passthrough": int(weight_quantum),
            "delta_rows": len(delta_rows),
            "pretrain_replay_rows": len(pretrain_replay_rows),
            "midtrain_replay_rows": len(midtrain_replay_rows),
            "delta_cases": delta_manifest,
        },
    )
    base._write_json(
        manifests_dir / f"{prefix}_workspace_manifest.json",
        {
            "format": "ck.structured_svg_atoms.workspace_materialization.v1",
            "workspace": str(workspace),
            "line": "spec19_probe_miss_delta",
            "source_run": str(source_run),
            "probe_report": str(probe_report_path),
            "replay_to_delta_ratio": int(replay_to_delta_ratio),
            "weight_quantum_passthrough": int(weight_quantum),
            "stages": {
                "pretrain": {
                    "train": base._rel(pretrain_train, workspace),
                    "dev": base._rel(pretrain_dev, workspace),
                    "test": base._rel(pretrain_test, workspace),
                    "counts": {
                        "train_rows": len(pretrain_train_rows),
                        "dev_rows": len(dev_rows),
                        "test_rows": len(test_rows),
                        "delta_rows": len(delta_rows),
                        "replay_rows": len(pretrain_replay_rows),
                    },
                },
                "midtrain": {
                    "train": base._rel(midtrain_train, workspace),
                    "dev": base._rel(midtrain_dev, workspace),
                    "test": base._rel(midtrain_test, workspace),
                    "counts": {
                        "train_rows": len(midtrain_train_rows),
                        "dev_rows": len(dev_rows),
                        "test_rows": len(test_rows),
                        "delta_rows": len(delta_rows),
                        "replay_rows": len(midtrain_replay_rows),
                    },
                },
            },
        },
    )

    return {
        "pretrain_train_rows": len(pretrain_train_rows),
        "midtrain_train_rows": len(midtrain_train_rows),
        "delta_rows": len(delta_rows),
        "pretrain_replay_rows": len(pretrain_replay_rows),
        "midtrain_replay_rows": len(midtrain_replay_rows),
        "holdout_rows": len(dev_rows) + len(test_rows),
        "hidden_seen_prompts": len(hidden_seen_prompt_rows),
        "hidden_holdout_prompts": len(hidden_holdout_prompt_rows),
    }


def main() -> int:
    spec19 = _load_module(SPEC19_MATERIALIZER, "materialize_spec19_scene_bundle_v7")
    base = spec19._load_base_module()
    import argparse

    ap = argparse.ArgumentParser(description="Materialize a replay-anchored spec19 delta workspace from probe misses")
    ap.add_argument("--workspace", required=True, type=Path, help="Destination workspace")
    ap.add_argument("--seed-workspace", default=str(base.DEFAULT_SEED_WORKSPACE), type=Path, help="Seed spec workspace template to copy")
    ap.add_argument("--prefix", default="spec19_scene_bundle", help="Dataset prefix")
    ap.add_argument("--freeze-tokenizer-run", required=True, type=Path, help="Run directory whose tokenizer should be copied unchanged")
    ap.add_argument("--source-run", required=True, type=Path, help="Completed spec19 run whose probe misses should seed the delta")
    ap.add_argument("--probe-report", default=None, type=Path, help="Optional probe report override")
    ap.add_argument("--replay-to-delta-ratio", type=int, default=3, help="Replay rows to include per delta row")
    ap.add_argument("--weight-quantum", type=int, default=5, help="Accepted for shared launcher compatibility; unused by the delta materializer")
    ap.add_argument("--python-exec", default=sys.executable, help="Python executable for helpers")
    ap.add_argument("--force", action="store_true", help="Replace workspace if it exists")
    args = ap.parse_args()

    summary = materialize_workspace(
        args.workspace,
        seed_workspace=args.seed_workspace,
        prefix=args.prefix,
        freeze_tokenizer_run=args.freeze_tokenizer_run,
        source_run=args.source_run,
        probe_report=args.probe_report,
        replay_to_delta_ratio=max(1, int(args.replay_to_delta_ratio)),
        weight_quantum=max(1, int(args.weight_quantum)),
        python_exec=str(args.python_exec),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
