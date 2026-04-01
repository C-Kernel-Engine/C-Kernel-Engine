#!/usr/bin/env python3
"""Materialize a cache-local spec16 generalized visual scene-bundle workspace."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
BASE_SCRIPT = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec06_structured_atoms_v7.py"
STRUCTURED_GENERATOR = ROOT / "version" / "v7" / "scripts" / "generate_svg_structured_spec16_v7.py"
LINE_NAME = "spec16_scene_bundle"
PRETRAIN_PROMPT_SURFACE_ORDER = [
    "tag_canonical",
    "bridge_create",
    "bridge_count_guard",
    "bridge_bundle_only",
    "bridge_style_lock",
    "repair_family_form_lock",
    "repair_bundle_singletons",
    "repair_topology_lock",
    "repair_clean_stop",
    "repair_control_stop",
    "train_hidden_compose",
    "train_hidden_stop",
    "train_hidden_style_bundle",
    "train_hidden_clean_stop",
]
MIDTRAIN_PROMPT_SURFACE_ORDER = [
    "tag_canonical",
    "repair_family_form_lock",
    "bridge_style_lock",
    "repair_topology_lock",
    "repair_bundle_singletons",
    "repair_clean_stop",
    "repair_control_stop",
    "bridge_count_guard",
    "bridge_bundle_only",
    "train_hidden_compose",
    "train_hidden_style_bundle",
    "train_hidden_clean_stop",
    "train_hidden_stop",
]


def _load_base_module():
    spec = importlib.util.spec_from_file_location("materialize_spec06_base_v7", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load base materializer: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row_from_catalog(prompt: str, output_tokens: str, *, base: Any) -> str:
    return base._row_from_catalog(str(prompt or "").strip(), str(output_tokens or "").strip())


def _split_members(split: str) -> set[str]:
    split = str(split or "")
    if split == "train":
        # Treat authored augmentation branches as real training rows.
        return {"train", "train_aug", "style_aug", "contrast_aug", "cross_form"}
    return {split}


def _stage_prompt_surface_order(stage: str) -> list[str]:
    stage = str(stage or "").strip().lower()
    if stage == "midtrain":
        return list(MIDTRAIN_PROMPT_SURFACE_ORDER)
    return list(PRETRAIN_PROMPT_SURFACE_ORDER)


def _stage_prompt_surfaces(stage: str) -> set[str]:
    return set(_stage_prompt_surface_order(stage))


def _stage_surface_schedule(stage: str, *, canonical_repeat: int, bridge_repeat: int) -> list[str]:
    schedule: list[str] = []
    for surface in _stage_prompt_surface_order(stage):
        repeat = int(canonical_repeat) if surface == "tag_canonical" else int(bridge_repeat)
        schedule.extend([surface] * max(1, repeat))
    return schedule


def _stage_split_rank(split: str) -> int:
    return {
        "train": 0,
        "style_aug": 1,
        "train_aug": 2,
        "contrast_aug": 3,
        "cross_form": 4,
        "holdout": 5,
    }.get(str(split or ""), 99)


def _stage_family(row: dict[str, Any]) -> str:
    return str(row.get("family") or row.get("layout") or "unknown")


def _stage_case_bucket_key(row: dict[str, Any]) -> tuple[str, str, str]:
    source_asset = str(row.get("source_asset") or "").split("#", 1)[0]
    stable_case = source_asset or str(row.get("case_id") or "")
    return (
        _stage_family(row),
        str(row.get("form_token") or ""),
        stable_case,
    )


def _stage_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _stage_family(row),
        str(row.get("form_token") or ""),
        str(row.get("source_asset") or ""),
        _stage_split_rank(str(row.get("split") or "")),
        str(row.get("case_id") or ""),
        str(row.get("theme") or ""),
        str(row.get("tone") or ""),
        str(row.get("density") or ""),
        str(row.get("background") or ""),
        str(row.get("prompt_surface") or ""),
        str(row.get("prompt") or ""),
    )


def _interleave_case_surface_rows(
    surface_rows: dict[str, list[dict[str, Any]]],
    *,
    surface_schedule: list[str],
) -> list[dict[str, Any]]:
    ordered_rows: list[dict[str, Any]] = []
    buckets = {surface: list(rows) for surface, rows in surface_rows.items()}
    while True:
        progressed = False
        for surface in surface_schedule:
            bucket = buckets.get(surface) or []
            if not bucket:
                continue
            ordered_rows.append(bucket.pop(0))
            progressed = True
        if not progressed:
            break
    return ordered_rows


def _build_stage_catalog_rows(
    catalog_rows: list[dict[str, Any]],
    *,
    stage: str,
    split: str,
    canonical_repeat: int,
    bridge_repeat: int,
    base: Any,
) -> list[dict[str, Any]]:
    allowed_splits = _split_members(split)
    allowed_surfaces = _stage_prompt_surfaces(stage)
    surface_schedule = _stage_surface_schedule(
        stage,
        canonical_repeat=int(canonical_repeat),
        bridge_repeat=int(bridge_repeat),
    )
    family_case_rows: dict[str, dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]]] = defaultdict(dict)
    seen_stage_rows: set[str] = set()
    for row in catalog_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("split") or "") not in allowed_splits:
            continue
        if not bool(row.get("training_prompt")):
            continue
        prompt_surface = str(row.get("prompt_surface") or "")
        if prompt_surface not in allowed_surfaces:
            continue
        prompt = str(row.get("prompt") or "").strip()
        output_tokens = str(row.get("output_tokens") or "").strip()
        if not prompt or not output_tokens:
            continue
        stage_row = _row_from_catalog(prompt, output_tokens, base=base)
        if stage_row in seen_stage_rows:
            continue
        seen_stage_rows.add(stage_row)
        row_doc = dict(row)
        row_doc["_stage_row_text"] = stage_row
        family = _stage_family(row_doc)
        case_key = _stage_case_bucket_key(row_doc)
        case_buckets = family_case_rows.setdefault(family, {})
        surface_buckets = case_buckets.setdefault(case_key, {})
        surface_buckets.setdefault(prompt_surface, []).append(row_doc)

    family_queues: dict[str, list[tuple[tuple[str, str, str], list[dict[str, Any]]]]] = {}
    for family, case_rows in family_case_rows.items():
        queue: list[tuple[tuple[str, str, str], list[dict[str, Any]]]] = []
        for case_key in sorted(case_rows.keys()):
            surface_rows = case_rows[case_key]
            for surface, rows in surface_rows.items():
                surface_rows[surface] = sorted(rows, key=_stage_row_sort_key)
            ordered_rows = _interleave_case_surface_rows(surface_rows, surface_schedule=surface_schedule)
            if ordered_rows:
                queue.append((case_key, ordered_rows))
        if queue:
            family_queues[family] = queue

    selected_rows: list[dict[str, Any]] = []
    while family_queues:
        progressed = False
        for family in sorted(list(family_queues.keys())):
            queue = family_queues.get(family) or []
            while queue and not queue[0][1]:
                queue.pop(0)
            if not queue:
                family_queues.pop(family, None)
                continue
            case_key, case_rows = queue.pop(0)
            selected_rows.append(case_rows.pop(0))
            progressed = True
            if case_rows:
                queue.append((case_key, case_rows))
            if queue:
                family_queues[family] = queue
            else:
                family_queues.pop(family, None)
        if not progressed:
            break
    return selected_rows


def _rows_from_stage_catalog(rows: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for row in rows:
        text = str((row or {}).get("_stage_row_text") or "").strip()
        if text:
            out.append(text)
    return out


def _summarize_stage_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    form_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    prompt_surface_counts: Counter[str] = Counter()
    unique_rows: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        family_counts[_stage_family(row)] += 1
        form_counts[str(row.get("form_token") or "unknown")] += 1
        split_counts[str(row.get("split") or "unknown")] += 1
        prompt_surface_counts[str(row.get("prompt_surface") or "unknown")] += 1
        unique_rows[str(row.get("_stage_row_text") or "")] += 1
    duplicate_items = [(text, count) for text, count in unique_rows.items() if text and count > 1]
    duplicate_rows_total = int(sum(count for _, count in duplicate_items))
    return {
        "total_rows": len(rows),
        "unique_rows": len(unique_rows),
        "duplicate_unique_rows": len(duplicate_items),
        "duplicate_rows_total": duplicate_rows_total,
        "family_counts": dict(sorted(family_counts.items())),
        "form_counts": dict(sorted(form_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "prompt_surface_counts": dict(sorted(prompt_surface_counts.items())),
    }


def _summarize_catalog(catalog_rows: list[dict[str, Any]], *, split: str) -> dict[str, Any]:
    allowed_splits = _split_members(split)
    prompt_surface_counts: Counter[str] = Counter()
    form_counts: Counter[str] = Counter()
    case_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for row in catalog_rows:
        if (
            not isinstance(row, dict)
            or str(row.get("split") or "") not in allowed_splits
            or not bool(row.get("training_prompt"))
        ):
            continue
        prompt_surface_counts[str(row.get("prompt_surface") or "unknown")] += 1
        form_counts[str(row.get("form_token") or "unknown")] += 1
        case_counts[str(row.get("case_id") or "unknown")] += 1
        family_counts[str(row.get("family") or row.get("layout") or "unknown")] += 1
    return {
        "split": split,
        "prompt_surface_counts": dict(sorted(prompt_surface_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "form_counts": dict(sorted(form_counts.items())),
        "case_counts": dict(sorted(case_counts.items())),
    }


def _split_holdout_catalog_rows(
    catalog_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in catalog_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("split") or "") != "holdout":
            continue
        if not bool(row.get("training_prompt")):
            continue
        scene_id = str(row.get("case_id") or "").strip()
        if scene_id not in grouped:
            grouped[scene_id] = []
            order.append(scene_id)
        grouped[scene_id].append(row)

    dev_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for idx, scene_id in enumerate(order):
        target = dev_rows if idx % 2 == 0 else test_rows
        target.extend(grouped.get(scene_id) or [])
    if order and not test_rows:
        moved = grouped.get(order[-1]) or []
        for row in moved:
            if row in dev_rows:
                dev_rows.remove(row)
        test_rows.extend(moved)
    return dev_rows, test_rows


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
    prompts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    return prompts


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


def materialize_workspace(
    workspace: Path,
    *,
    seed_workspace: Path,
    prefix: str,
    pretrain_canonical_repeat: int,
    pretrain_bridge_repeat: int,
    midtrain_canonical_repeat: int,
    midtrain_bridge_repeat: int,
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
        raise RuntimeError(f"spec16 generator failed (rc={proc.returncode}):\n{proc.stdout.strip()}")

    render_catalog_rows = json.loads((generated_dir / f"{prefix}_render_catalog.json").read_text(encoding="utf-8"))
    seen_prompt_rows = (generated_dir / f"{prefix}_seen_prompts.txt").read_text(encoding="utf-8").splitlines()
    holdout_prompt_rows = (generated_dir / f"{prefix}_holdout_prompts.txt").read_text(encoding="utf-8").splitlines()
    hidden_seen_prompt_rows = (generated_dir / f"{prefix}_hidden_seen_prompts.txt").read_text(encoding="utf-8").splitlines()
    hidden_holdout_prompt_rows = (generated_dir / f"{prefix}_hidden_holdout_prompts.txt").read_text(encoding="utf-8").splitlines()
    reserved_token_rows = (generated_dir / f"{prefix}_reserved_control_tokens.txt").read_text(encoding="utf-8").splitlines()
    tokenizer_corpus_rows = _tokenizer_corpus_rows(render_catalog_rows, base=base)

    pretrain_stage_rows = _build_stage_catalog_rows(
        render_catalog_rows,
        stage="pretrain",
        split="train",
        canonical_repeat=int(pretrain_canonical_repeat),
        bridge_repeat=int(pretrain_bridge_repeat),
        base=base,
    )
    midtrain_stage_rows = _build_stage_catalog_rows(
        render_catalog_rows,
        stage="midtrain",
        split="train",
        canonical_repeat=int(midtrain_canonical_repeat),
        bridge_repeat=int(midtrain_bridge_repeat),
        base=base,
    )
    pretrain_train_rows = _rows_from_stage_catalog(pretrain_stage_rows)
    midtrain_train_rows = _rows_from_stage_catalog(midtrain_stage_rows)
    pretrain_train_summary = _summarize_stage_rows(pretrain_stage_rows)
    midtrain_train_summary = _summarize_stage_rows(midtrain_stage_rows)

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
    tokenizer_json_src = generated_dir / f"{prefix}_tokenizer" / "tokenizer.json"
    tokenizer_bin_src = generated_dir / f"{prefix}_tokenizer" / "tokenizer_bin"
    eval_contract_src = generated_dir / f"{prefix}_eval_contract.json"
    tokenizer_json_dst = tokenizer_dir / "tokenizer.json"
    tokenizer_bin_dst = tokenizer_dir / "tokenizer_bin"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    base.shutil.copy2(tokenizer_json_src, tokenizer_json_dst)
    base._copy_tree(tokenizer_bin_src, tokenizer_bin_dst)
    base.shutil.copy2(generated_dir / f"{prefix}_vocab.json", tokenizer_dir / f"{prefix}_vocab.json")
    base.shutil.copy2(generated_dir / f"{prefix}_render_catalog.json", tokenizer_dir / f"{prefix}_render_catalog.json")
    base._write_lines(tokenizer_dir / f"{prefix}_reserved_control_tokens.txt", reserved_token_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_tokenizer_corpus.txt", tokenizer_corpus_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_seen_prompts.txt", seen_prompt_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_holdout_prompts.txt", holdout_prompt_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_hidden_seen_prompts.txt", hidden_seen_prompt_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_hidden_holdout_prompts.txt", hidden_holdout_prompt_rows)

    contracts_dir = workspace / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    base.shutil.copy2(eval_contract_src, contracts_dir / "eval_contract.svg.v1.json")

    probe_contract_dst = contracts_dir / f"{prefix}_probe_report_contract.json"
    probe_cmd = [
        python_exec,
        str(ROOT / "version" / "v7" / "scripts" / "build_spec16_probe_contract_v7.py"),
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
    probe_proc = base.subprocess.run(
        probe_cmd,
        cwd=ROOT,
        stdout=base.subprocess.PIPE,
        stderr=base.subprocess.STDOUT,
        text=True,
    )
    if probe_proc.returncode != 0:
        raise RuntimeError(f"spec16 probe contract build failed (rc={probe_proc.returncode}):\n{probe_proc.stdout.strip()}")
    base.shutil.copy2(probe_contract_dst, tokenizer_dir / f"{prefix}_probe_report_contract.json")

    tokenizer_manifest = {
        "format": "ck.structured_svg_atoms.tokenizer_manifest.v1",
        "line": LINE_NAME,
        "workspace": str(workspace),
        "artifacts": {
            "tokenizer_json": base._rel(tokenizer_json_dst, workspace),
            "tokenizer_bin": base._rel(tokenizer_bin_dst, workspace),
            "vocab": base._rel(tokenizer_dir / f"{prefix}_vocab.json", workspace),
            "render_catalog": base._rel(tokenizer_dir / f"{prefix}_render_catalog.json", workspace),
            "reserved_control_tokens": base._rel(tokenizer_dir / f"{prefix}_reserved_control_tokens.txt", workspace),
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
        "pretrain_canonical_repeat": int(pretrain_canonical_repeat),
        "pretrain_bridge_repeat": int(pretrain_bridge_repeat),
        "midtrain_canonical_repeat": int(midtrain_canonical_repeat),
        "midtrain_bridge_repeat": int(midtrain_bridge_repeat),
        "train_summary": _summarize_catalog(render_catalog_rows, split="train"),
        "holdout_summary": _summarize_catalog(render_catalog_rows, split="holdout"),
        "stages": {
            "pretrain": {
                "train": base._rel(pretrain_train, workspace),
                "dev": base._rel(pretrain_dev, workspace),
                "test": base._rel(pretrain_test, workspace),
                "prompt_surface_order": _stage_prompt_surface_order("pretrain"),
                "counts": {
                    "train_rows": len(pretrain_train_rows),
                    "dev_rows": len(dev_rows),
                    "test_rows": len(test_rows),
                    "train_summary": pretrain_train_summary,
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
                    "train_summary": midtrain_train_summary,
                },
                "prompt_surface_order": _stage_prompt_surface_order("midtrain"),
                "notes": "Spec16 is the first generalized visual-DSL line: compact shared [bundle] output only, payload externalized, and lowering through solved family compilers. Midtrain is narrowed to family/form/style/topology repair instead of broad create prompts.",
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

    ap = argparse.ArgumentParser(description="Materialize a cache-local spec16 generalized visual scene-bundle workspace")
    ap.add_argument("--workspace", required=True, type=Path, help="Destination workspace")
    ap.add_argument("--seed-workspace", default=str(base.DEFAULT_SEED_WORKSPACE), type=Path, help="Seed spec workspace template to copy")
    ap.add_argument("--prefix", default="spec16_scene_bundle", help="Dataset prefix")
    ap.add_argument("--pretrain-canonical-repeat", type=int, default=10)
    ap.add_argument("--pretrain-bridge-repeat", type=int, default=3)
    ap.add_argument("--midtrain-canonical-repeat", type=int, default=8)
    ap.add_argument("--midtrain-bridge-repeat", type=int, default=3)
    ap.add_argument("--python-exec", default=sys.executable, help="Python executable for generators")
    ap.add_argument("--force", action="store_true", help="Replace workspace if it exists")
    args = ap.parse_args()

    summary = materialize_workspace(
        args.workspace,
        seed_workspace=args.seed_workspace,
        prefix=args.prefix,
        pretrain_canonical_repeat=int(args.pretrain_canonical_repeat),
        pretrain_bridge_repeat=int(args.pretrain_bridge_repeat),
        midtrain_canonical_repeat=int(args.midtrain_canonical_repeat),
        midtrain_bridge_repeat=int(args.midtrain_bridge_repeat),
        python_exec=str(args.python_exec),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
