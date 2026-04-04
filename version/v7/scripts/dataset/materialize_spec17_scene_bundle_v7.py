#!/usr/bin/env python3
"""Materialize a cache-local spec17 bounded-intent bundle workspace."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from tokenizer_policy_v7 import sanitize_tokenizer_doc, visible_special_tokens
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tokenizer_policy_v7 import sanitize_tokenizer_doc, visible_special_tokens


ROOT = Path(__file__).resolve().parents[4]
BASE_SCRIPT = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec06_structured_atoms_v7.py"
STRUCTURED_GENERATOR = ROOT / "version" / "v7" / "scripts" / "generate_svg_structured_spec17_v7.py"
BLUEPRINT_PATH = ROOT / "version" / "v7" / "reports" / "spec17_curriculum_blueprint.json"
LINE_NAME = "spec17_scene_bundle"
STAGE_BLUEPRINT_IDS = {"pretrain": "stage_a", "midtrain": "stage_b"}


def _load_base_module():
    spec = importlib.util.spec_from_file_location("materialize_spec06_base_v7", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load base materializer: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_blueprint() -> dict[str, Any]:
    return json.loads(BLUEPRINT_PATH.read_text(encoding="utf-8"))


def _row_from_catalog(prompt: str, output_tokens: str, *, base: Any) -> str:
    return base._row_from_catalog(str(prompt or "").strip(), str(output_tokens or "").strip())


def _copy_frozen_tokenizer(freeze_run: Path, tokenizer_dir: Path, prefix: str, *, base: Any) -> dict[str, Any]:
    freeze_run = freeze_run.expanduser().resolve()
    tokenizer_json_src = next(
        (
            path
            for path in (
                freeze_run / "tokenizer.json",
                freeze_run / "dataset" / "tokenizer" / "tokenizer.json",
            )
            if path.exists()
        ),
        None,
    )
    tokenizer_bin_src = next(
        (
            path
            for path in (
                freeze_run / "tokenizer_bin",
                freeze_run / "dataset" / "tokenizer" / "tokenizer_bin",
            )
            if path.exists()
        ),
        None,
    )
    if tokenizer_json_src is None or tokenizer_bin_src is None:
        raise FileNotFoundError(f"frozen tokenizer artifacts not found under {freeze_run}")

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json_dst = tokenizer_dir / "tokenizer.json"
    tokenizer_bin_dst = tokenizer_dir / "tokenizer_bin"
    base.shutil.copy2(tokenizer_json_src, tokenizer_json_dst)
    base._copy_tree(tokenizer_bin_src, tokenizer_bin_dst)

    tokenizer_doc = json.loads(tokenizer_json_dst.read_text(encoding="utf-8"))
    tokenizer_doc, removed_special_tokens = sanitize_tokenizer_doc(tokenizer_doc)
    tokenizer_json_dst.write_text(json.dumps(tokenizer_doc, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    reserved = visible_special_tokens(tokenizer_doc)
    base._write_lines(tokenizer_dir / f"{prefix}_reserved_control_tokens.txt", reserved)
    return {
        "tokenizer_json": tokenizer_json_dst,
        "tokenizer_bin": tokenizer_bin_dst,
        "reserved_control_tokens": tokenizer_dir / f"{prefix}_reserved_control_tokens.txt",
        "frozen_from_run": freeze_run,
        "sanitized_removed_special_tokens": removed_special_tokens,
    }


def _stage_surface_repeats(stage: str, *, weight_quantum: int) -> dict[str, int]:
    blueprint = _load_blueprint()
    blueprint_stage_id = STAGE_BLUEPRINT_IDS[str(stage or "").strip().lower()]
    for row in (blueprint.get("curriculum_stages") or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("id") or "") != blueprint_stage_id:
            continue
        repeats: dict[str, int] = {}
        for item in (row.get("surface_mix") or []):
            if not isinstance(item, dict):
                continue
            surface = str(item.get("surface") or "").strip()
            weight = int(item.get("weight") or 0)
            if not surface or weight <= 0:
                continue
            repeats[surface] = max(1, int(weight) // max(1, int(weight_quantum)))
        if repeats:
            return repeats
    raise KeyError(f"missing stage surface mix for {stage}")


def _stage_surface_schedule(stage: str, *, weight_quantum: int) -> list[str]:
    repeats = _stage_surface_repeats(stage, weight_quantum=weight_quantum)
    schedule: list[str] = []
    for surface, repeat in repeats.items():
        schedule.extend([surface] * max(1, repeat))
    return schedule


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("family") or row.get("layout") or ""),
        str(row.get("profile_id") or row.get("case_id") or ""),
        str(row.get("prompt_surface") or ""),
        str(row.get("topic") or ""),
        str(row.get("goal") or ""),
        str(row.get("audience") or ""),
        str(row.get("prompt") or ""),
    )


def _build_stage_catalog_rows(
    catalog_rows: list[dict[str, Any]],
    *,
    stage: str,
    base: Any,
    weight_quantum: int,
) -> list[dict[str, Any]]:
    repeats = _stage_surface_repeats(stage, weight_quantum=weight_quantum)
    schedule = _stage_surface_schedule(stage, weight_quantum=weight_quantum)
    surface_family_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in catalog_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("split") or "") != "train":
            continue
        if not bool(row.get("training_prompt")):
            continue
        surface = str(row.get("prompt_surface") or "")
        if surface not in repeats:
            continue
        row_doc = dict(row)
        row_doc["_stage_row_text"] = _row_from_catalog(str(row.get("prompt") or ""), str(row.get("output_tokens") or ""), base=base)
        family = str(row_doc.get("family") or row_doc.get("layout") or "unknown")
        surface_family_rows[surface][family].append(row_doc)

    surface_queues: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
    for surface, family_rows in surface_family_rows.items():
        queue: list[tuple[str, list[dict[str, Any]]]] = []
        for family in sorted(family_rows):
            bucket = sorted(family_rows[family], key=_row_sort_key)
            if bucket:
                queue.append((family, bucket))
        if queue:
            surface_queues[surface] = queue

    selected: list[dict[str, Any]] = []
    while True:
        progressed = False
        for surface in schedule:
            queue = surface_queues.get(surface) or []
            while queue and not queue[0][1]:
                queue.pop(0)
            if not queue:
                surface_queues.pop(surface, None)
                continue
            family, bucket = queue.pop(0)
            selected.append(bucket.pop(0))
            progressed = True
            if bucket:
                queue.append((family, bucket))
            if queue:
                surface_queues[surface] = queue
            else:
                surface_queues.pop(surface, None)
        if not progressed:
            break
    return selected


def _rows_from_stage_catalog(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("_stage_row_text") or "").strip() for row in rows if str(row.get("_stage_row_text") or "").strip()]


def _rows_from_catalog(rows: list[dict[str, Any]], *, base: Any) -> list[str]:
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        output_tokens = str(row.get("output_tokens") or "").strip()
        if not prompt or not output_tokens:
            continue
        out.append(_row_from_catalog(prompt, output_tokens, base=base))
    return out


def _prompts_from_catalog(rows: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        out.append(prompt)
    return out


def _tokenizer_corpus_rows(catalog_rows: list[dict[str, Any]], *, base: Any) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for row in catalog_rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        output_tokens = str(row.get("output_tokens") or "").strip()
        if not prompt or not output_tokens:
            continue
        text = _row_from_catalog(prompt, output_tokens, base=base)
        if text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows


def _split_holdout_catalog_rows(catalog_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in catalog_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("split") or "") != "holdout":
            continue
        profile_id = str(row.get("profile_id") or row.get("case_id") or "").strip()
        if not profile_id:
            continue
        if profile_id not in grouped:
            grouped[profile_id] = []
            order.append(profile_id)
        grouped[profile_id].append(row)

    dev_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for idx, profile_id in enumerate(order):
        target = dev_rows if idx % 2 == 0 else test_rows
        target.extend(grouped.get(profile_id) or [])
    if order and not test_rows:
        moved = grouped.get(order[-1]) or []
        for row in moved:
            if row in dev_rows:
                dev_rows.remove(row)
        test_rows.extend(moved)
    return dev_rows, test_rows


def _summarize_catalog(catalog_rows: list[dict[str, Any]], *, split: str) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    prompt_surface_counts: Counter[str] = Counter()
    for row in catalog_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("split") or "") != split:
            continue
        family_counts[str(row.get("family") or row.get("layout") or "unknown")] += 1
        profile_counts[str(row.get("profile_id") or row.get("case_id") or "unknown")] += 1
        prompt_surface_counts[str(row.get("prompt_surface") or "unknown")] += 1
    return {
        "split": split,
        "family_counts": dict(sorted(family_counts.items())),
        "profile_counts": dict(sorted(profile_counts.items())),
        "prompt_surface_counts": dict(sorted(prompt_surface_counts.items())),
    }


def _summarize_stage_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    prompt_surface_counts: Counter[str] = Counter()
    unique_rows: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        family_counts[str(row.get("family") or row.get("layout") or "unknown")] += 1
        prompt_surface_counts[str(row.get("prompt_surface") or "unknown")] += 1
        unique_rows[str(row.get("_stage_row_text") or "")] += 1
    duplicate_items = [(text, count) for text, count in unique_rows.items() if text and count > 1]
    return {
        "total_rows": len(rows),
        "unique_rows": len(unique_rows),
        "duplicate_unique_rows": len(duplicate_items),
        "duplicate_rows_total": int(sum(count for _, count in duplicate_items)),
        "family_counts": dict(sorted(family_counts.items())),
        "prompt_surface_counts": dict(sorted(prompt_surface_counts.items())),
    }


def materialize_workspace(
    workspace: Path,
    *,
    seed_workspace: Path,
    prefix: str,
    freeze_tokenizer_run: Path,
    weight_quantum: int,
    python_exec: str,
    force: bool,
) -> dict[str, Any]:
    base = _load_base_module()
    workspace = workspace.expanduser().resolve()
    seed_workspace = seed_workspace.expanduser().resolve()
    if not seed_workspace.exists():
        raise FileNotFoundError(f"seed workspace not found: {seed_workspace}")

    base._copy_seed_workspace(seed_workspace, workspace, force=force)
    base._ensure_split_dirs(workspace)

    generated_dir = workspace / "manifests" / "generated" / "structured_atoms"
    generated_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exec,
        str(STRUCTURED_GENERATOR),
        "--out-dir",
        str(generated_dir),
        "--prefix",
        prefix,
    ]
    proc = base.subprocess.run(cmd, cwd=ROOT, stdout=base.subprocess.PIPE, stderr=base.subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"spec17 generator failed (rc={proc.returncode}):\n{proc.stdout.strip()}")

    render_catalog_rows = json.loads((generated_dir / f"{prefix}_render_catalog.json").read_text(encoding="utf-8"))
    if not isinstance(render_catalog_rows, list):
        raise RuntimeError("spec17 render catalog must be a JSON list")
    seen_prompt_rows = (generated_dir / f"{prefix}_seen_prompts.txt").read_text(encoding="utf-8").splitlines()
    holdout_prompt_rows = (generated_dir / f"{prefix}_holdout_prompts.txt").read_text(encoding="utf-8").splitlines()
    hidden_seen_prompt_rows = (generated_dir / f"{prefix}_hidden_seen_prompts.txt").read_text(encoding="utf-8").splitlines()
    hidden_holdout_prompt_rows = (generated_dir / f"{prefix}_hidden_holdout_prompts.txt").read_text(encoding="utf-8").splitlines()

    pretrain_stage_rows = _build_stage_catalog_rows(
        render_catalog_rows,
        stage="pretrain",
        base=base,
        weight_quantum=int(weight_quantum),
    )
    midtrain_stage_rows = _build_stage_catalog_rows(
        render_catalog_rows,
        stage="midtrain",
        base=base,
        weight_quantum=int(weight_quantum),
    )
    pretrain_train_rows = _rows_from_stage_catalog(pretrain_stage_rows)
    midtrain_train_rows = _rows_from_stage_catalog(midtrain_stage_rows)

    holdout_dev_catalog_rows, holdout_test_catalog_rows = _split_holdout_catalog_rows(render_catalog_rows)
    dev_rows = _rows_from_catalog(holdout_dev_catalog_rows, base=base)
    test_rows = _rows_from_catalog(holdout_test_catalog_rows, base=base)
    holdout_prompt_dev = _prompts_from_catalog(holdout_dev_catalog_rows)
    holdout_prompt_test = _prompts_from_catalog(holdout_test_catalog_rows)

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
    frozen = _copy_frozen_tokenizer(freeze_tokenizer_run, tokenizer_dir, prefix, base=base)
    tokenizer_corpus_rows = _tokenizer_corpus_rows(render_catalog_rows, base=base)
    base.shutil.copy2(generated_dir / f"{prefix}_render_catalog.json", tokenizer_dir / f"{prefix}_render_catalog.json")
    base._write_lines(tokenizer_dir / f"{prefix}_tokenizer_corpus.txt", tokenizer_corpus_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_seen_prompts.txt", seen_prompt_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_holdout_prompts.txt", holdout_prompt_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_hidden_seen_prompts.txt", hidden_seen_prompt_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_hidden_holdout_prompts.txt", hidden_holdout_prompt_rows)

    contracts_dir = workspace / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    base.shutil.copy2(generated_dir / f"{prefix}_eval_contract.json", contracts_dir / "eval_contract.svg.v1.json")
    base.shutil.copy2(BLUEPRINT_PATH, contracts_dir / "spec17_curriculum_blueprint.json")

    probe_contract_dst = contracts_dir / f"{prefix}_probe_report_contract.json"
    probe_cmd = [
        python_exec,
        str(ROOT / "version" / "v7" / "scripts" / "build_spec17_probe_contract_v7.py"),
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
        raise RuntimeError(f"spec17 probe contract build failed (rc={probe_proc.returncode}):\n{probe_proc.stdout.strip()}")
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

    workspace_manifest = {
        "format": "ck.structured_svg_atoms.workspace_materialization.v1",
        "source_seed_workspace": str(seed_workspace),
        "workspace": str(workspace),
        "line": LINE_NAME,
        "generator": str(STRUCTURED_GENERATOR),
        "prefix": prefix,
        "frozen_tokenizer_run": str(freeze_tokenizer_run.expanduser().resolve()),
        "weight_quantum": int(weight_quantum),
        "train_summary": _summarize_catalog(render_catalog_rows, split="train"),
        "holdout_summary": _summarize_catalog(render_catalog_rows, split="holdout"),
        "hidden_train_summary": _summarize_catalog(render_catalog_rows, split="probe_hidden_train"),
        "hidden_holdout_summary": _summarize_catalog(render_catalog_rows, split="probe_hidden_holdout"),
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
                "summary": _summarize_stage_rows(pretrain_stage_rows),
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
                "summary": _summarize_stage_rows(midtrain_stage_rows),
                "notes": "Spec17 keeps the frozen spec16 bundle contract and varies only the bounded intent prompt side.",
            },
        },
    }
    manifests_dir = workspace / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    base._write_json(manifests_dir / f"{prefix}_workspace_manifest.json", workspace_manifest)

    return {
        "pretrain_train_rows": len(pretrain_train_rows),
        "midtrain_train_rows": len(midtrain_train_rows),
        "holdout_rows": len(holdout_dev_catalog_rows) + len(holdout_test_catalog_rows),
        "hidden_seen_prompts": len(hidden_seen_prompt_rows),
        "hidden_holdout_prompts": len(hidden_holdout_prompt_rows),
    }


def main() -> int:
    base = _load_base_module()
    import argparse

    ap = argparse.ArgumentParser(description="Materialize a cache-local spec17 scene-bundle workspace")
    ap.add_argument("--workspace", required=True, type=Path, help="Destination workspace")
    ap.add_argument("--seed-workspace", default=str(base.DEFAULT_SEED_WORKSPACE), type=Path, help="Seed spec workspace template to copy")
    ap.add_argument("--prefix", default="spec17_scene_bundle", help="Dataset prefix")
    ap.add_argument("--freeze-tokenizer-run", required=True, type=Path, help="Run directory whose tokenizer should be copied unchanged")
    ap.add_argument("--weight-quantum", type=int, default=5, help="Convert stage weights to row repeats by dividing by this quantum")
    ap.add_argument("--python-exec", default=sys.executable, help="Python executable for generators")
    ap.add_argument("--force", action="store_true", help="Replace workspace if it exists")
    args = ap.parse_args()

    summary = materialize_workspace(
        args.workspace,
        seed_workspace=args.seed_workspace,
        prefix=args.prefix,
        freeze_tokenizer_run=args.freeze_tokenizer_run,
        weight_quantum=int(args.weight_quantum),
        python_exec=str(args.python_exec),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
