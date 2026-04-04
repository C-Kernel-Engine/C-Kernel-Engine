#!/usr/bin/env python3
"""Materialize a unified spec19 workspace and populate instruction-style SFT rows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
UNIFIED_MATERIALIZER = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec19_unified_curriculum_v7.py"
SPEC19_MATERIALIZER = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec19_scene_bundle_v7.py"
DEFAULT_LINE_NAME = "spec19_sft_instruction"
DEFAULT_FORMAT_VERSION = "ck.spec19_sft_instruction.v1"

EXPLICIT_SURFACES = {
    "explicit_bundle_anchor",
    "explicit_permuted_anchor",
    "clean_stop_anchor",
}
ROUTING_SURFACES = {
    "routebook_direct",
    "routebook_direct_hint",
    "form_minimal_pair",
    "family_minimal_pair",
    "routebook_paraphrase",
    "holdout_routebook_direct",
    "holdout_style_topology_bridge",
    "hidden_routebook_paraphrase",
    "hidden_recombination",
    "recombination_bridge",
    "style_topology_bridge",
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


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


def _parse_surface_multiplier(raw_items: list[str] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in raw_items or []:
        text = str(raw or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"expected surface multiplier in surface=count form: {text}")
        surface, count_text = text.split("=", 1)
        surface = surface.strip()
        if not surface:
            raise ValueError(f"empty surface in multiplier: {text}")
        count = int(count_text.strip())
        if count < 1:
            raise ValueError(f"multiplier must be >= 1 for {surface}: {count}")
        out[surface] = count
    return dict(sorted(out.items()))


def _instruction_variants(prompt: str, *, prompt_surface: str) -> list[str]:
    prompt_text = str(prompt or "").strip()
    surface = str(prompt_surface or "").strip()
    if not prompt_text:
        return []
    variants = [prompt_text]
    if surface in EXPLICIT_SURFACES:
        variants.extend(
            [
                f"Follow this explicit bundle request exactly. Return one compiler-facing [bundle] only and stop after [/bundle]. {prompt_text}",
                f"Use the explicit request below as the only source of truth. Emit one canonical [bundle] only with no prose before or after it. {prompt_text}",
            ]
        )
    else:
        variants.extend(
            [
                f"Plan exactly one compiler-facing shared visual bundle for this request. Return one [bundle] only and stop after [/bundle]. {prompt_text}",
                f"Read the request carefully and answer with one canonical [bundle] only. Do not explain your reasoning. {prompt_text}",
            ]
        )
        if surface in ROUTING_SURFACES:
            variants.append(
                f"Keep the routing decision faithful to the request below and answer with one compiler-facing [bundle] only. {prompt_text}"
            )
    return _dedupe_preserve(variants)


def _split_train_holdout_rows(
    catalog_rows: list[dict[str, Any]],
    *,
    spec19: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [
        row for row in catalog_rows
        if isinstance(row, dict)
        and str(row.get("split") or "") == "train"
        and bool(row.get("training_prompt"))
    ]
    holdout_dev_rows, holdout_test_rows = spec19._split_holdout_catalog_rows(catalog_rows)
    return train_rows, holdout_dev_rows, holdout_test_rows


def _build_sft_rows(
    catalog_rows: list[dict[str, Any]],
    *,
    base: Any,
    max_variants: int,
    surface_multipliers: dict[str, int] | None = None,
    preserve_surface_weights: bool = False,
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    out: list[str] = []
    source_surface_counts: Counter[str] = Counter()
    instruction_surface_counts: Counter[str] = Counter()
    for row in catalog_rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        output_tokens = str(row.get("output_tokens") or "").strip()
        if not prompt or not output_tokens:
            continue
        surface = str(row.get("prompt_surface") or "unknown").strip() or "unknown"
        source_surface_counts[surface] += 1
        variants = _instruction_variants(prompt, prompt_surface=surface)[: max(1, int(max_variants))]
        multiplier = max(1, int((surface_multipliers or {}).get(surface, 1)))
        for instruction in variants:
            rendered = base._row_from_catalog(instruction, output_tokens)
            repeats = multiplier if preserve_surface_weights else 1
            for _ in range(repeats):
                out.append(rendered)
                instruction_surface_counts[surface] += 1
    materialized_rows = out if preserve_surface_weights else _dedupe_preserve(out)
    return materialized_rows, dict(sorted(source_surface_counts.items())), dict(sorted(instruction_surface_counts.items()))


def _build_sft_manifest(
    *,
    workspace: Path,
    prefix: str,
    unified_manifest: dict[str, Any],
    train_rows: int,
    dev_rows: int,
    test_rows: int,
    source_surface_counts: dict[str, int],
    instruction_surface_counts: dict[str, int],
    train_variants: int,
    eval_variants: int,
    line_name: str,
    format_version: str,
    train_surface_multipliers: dict[str, int],
) -> dict[str, Any]:
    return {
        "format": format_version,
        "line": line_name,
        "workspace": str(workspace),
        "prefix": prefix,
        "derived_from_line": str(unified_manifest.get("line") or ""),
        "source_runs": list(unified_manifest.get("source_runs") or []),
        "instruction_variant_policy": {
            "train_variants_per_prompt": int(train_variants),
            "eval_variants_per_prompt": int(eval_variants),
            "preserve_original_prompt": True,
            "preserve_train_surface_weights": bool(train_surface_multipliers),
            "train_surface_multipliers": dict(train_surface_multipliers),
        },
        "stages": {
            "sft": {
                "train_rows": int(train_rows),
                "dev_rows": int(dev_rows),
                "test_rows": int(test_rows),
                "source_prompt_surface_counts": dict(source_surface_counts),
                "instruction_row_surface_counts": dict(instruction_surface_counts),
            }
        },
    }


def _refresh_manifests(
    workspace: Path,
    *,
    prefix: str,
    sft_manifest: dict[str, Any],
    train_rows: int,
    dev_rows: int,
    test_rows: int,
    source_surface_counts: dict[str, int],
    instruction_surface_counts: dict[str, int],
    line_name: str,
) -> None:
    manifests_dir = workspace / "manifests"
    sft_manifest_path = manifests_dir / f"{prefix}_sft_instruction_manifest.json"
    _write_json(sft_manifest_path, sft_manifest)

    workspace_manifest_path = manifests_dir / f"{prefix}_workspace_manifest.json"
    workspace_manifest = _read_json(workspace_manifest_path)
    workspace_manifest["line"] = line_name
    workspace_manifest["derived_from_line"] = sft_manifest["derived_from_line"]
    stages = workspace_manifest.setdefault("stages", {})
    if not isinstance(stages, dict):
        stages = {}
        workspace_manifest["stages"] = stages
    stages["sft"] = {
        "train": f"sft/train/{prefix}_sft_train.txt",
        "dev": f"sft/dev/{prefix}_sft_dev.txt",
        "test": f"sft/test/{prefix}_sft_test.txt",
        "counts": {
            "train_rows": int(train_rows),
            "dev_rows": int(dev_rows),
            "test_rows": int(test_rows),
        },
        "summary": {
            "source_prompt_surface_counts": dict(source_surface_counts),
            "instruction_row_surface_counts": dict(instruction_surface_counts),
        },
        "notes": (
            "Instruction-style spec19 SFT rows derived from the unified curriculum. "
            "Keep the canonical [bundle] output contract unchanged while sharpening prompt fidelity on top of the frozen r3d policy."
        ),
    }
    workspace_manifest["sft_instruction_manifest"] = f"manifests/{prefix}_sft_instruction_manifest.json"
    _write_json(workspace_manifest_path, workspace_manifest)

    unified_manifest_path = manifests_dir / f"{prefix}_unified_curriculum_manifest.json"
    if unified_manifest_path.exists():
        unified_manifest = _read_json(unified_manifest_path)
        unified_manifest["sft_instruction_manifest"] = f"manifests/{prefix}_sft_instruction_manifest.json"
        _write_json(unified_manifest_path, unified_manifest)


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
    train_variants: int = 3,
    eval_variants: int = 2,
    line_name: str = DEFAULT_LINE_NAME,
    format_version: str = DEFAULT_FORMAT_VERSION,
    train_surface_multipliers: dict[str, int] | None = None,
) -> dict[str, Any]:
    unified = _load_module(UNIFIED_MATERIALIZER, "materialize_spec19_unified_curriculum_v7")
    spec19 = _load_module(SPEC19_MATERIALIZER, "materialize_spec19_scene_bundle_v7")
    base = spec19._load_base_module()

    workspace = workspace.expanduser().resolve()
    summary = unified.materialize_workspace(
        workspace,
        seed_workspace=seed_workspace,
        prefix=prefix,
        freeze_tokenizer_run=freeze_tokenizer_run,
        source_runs=source_runs,
        weight_quantum=weight_quantum,
        python_exec=python_exec,
        force=force,
    )

    render_catalog_path = workspace / "tokenizer" / f"{prefix}_render_catalog.json"
    catalog_rows = json.loads(render_catalog_path.read_text(encoding="utf-8"))
    if not isinstance(catalog_rows, list):
        raise RuntimeError(f"expected JSON list render catalog at {render_catalog_path}")
    train_catalog_rows, holdout_dev_rows, holdout_test_rows = _split_train_holdout_rows(catalog_rows, spec19=spec19)

    sft_train_rows, train_source_surface_counts, train_instruction_surface_counts = _build_sft_rows(
        train_catalog_rows,
        base=base,
        max_variants=int(train_variants),
        surface_multipliers=dict(train_surface_multipliers or {}),
        preserve_surface_weights=bool(train_surface_multipliers),
    )
    sft_dev_rows, dev_source_surface_counts, dev_instruction_surface_counts = _build_sft_rows(
        holdout_dev_rows,
        base=base,
        max_variants=int(eval_variants),
    )
    sft_test_rows, test_source_surface_counts, test_instruction_surface_counts = _build_sft_rows(
        holdout_test_rows,
        base=base,
        max_variants=int(eval_variants),
    )

    base._write_lines(workspace / "sft" / "train" / f"{prefix}_sft_train.txt", sft_train_rows)
    base._write_lines(workspace / "sft" / "dev" / f"{prefix}_sft_dev.txt", sft_dev_rows)
    base._write_lines(workspace / "sft" / "test" / f"{prefix}_sft_test.txt", sft_test_rows)

    unified_manifest = _read_json(workspace / "manifests" / f"{prefix}_unified_curriculum_manifest.json")
    source_surface_counts: Counter[str] = Counter(train_source_surface_counts)
    source_surface_counts.update(dev_source_surface_counts)
    source_surface_counts.update(test_source_surface_counts)
    instruction_surface_counts: Counter[str] = Counter(train_instruction_surface_counts)
    instruction_surface_counts.update(dev_instruction_surface_counts)
    instruction_surface_counts.update(test_instruction_surface_counts)
    sft_manifest = _build_sft_manifest(
        workspace=workspace,
        prefix=prefix,
        unified_manifest=unified_manifest,
        train_rows=len(sft_train_rows),
        dev_rows=len(sft_dev_rows),
        test_rows=len(sft_test_rows),
        source_surface_counts=dict(sorted(source_surface_counts.items())),
        instruction_surface_counts=dict(sorted(instruction_surface_counts.items())),
        train_variants=int(train_variants),
        eval_variants=int(eval_variants),
        line_name=str(line_name),
        format_version=str(format_version),
        train_surface_multipliers=dict(sorted((train_surface_multipliers or {}).items())),
    )
    _refresh_manifests(
        workspace,
        prefix=prefix,
        sft_manifest=sft_manifest,
        train_rows=len(sft_train_rows),
        dev_rows=len(sft_dev_rows),
        test_rows=len(sft_test_rows),
        source_surface_counts=dict(sorted(source_surface_counts.items())),
        instruction_surface_counts=dict(sorted(instruction_surface_counts.items())),
        line_name=str(line_name),
    )

    out = dict(summary)
    out["line"] = str(line_name)
    out["sft_train_rows"] = len(sft_train_rows)
    out["sft_dev_rows"] = len(sft_dev_rows)
    out["sft_test_rows"] = len(sft_test_rows)
    out["train_surface_multipliers"] = dict(sorted((train_surface_multipliers or {}).items()))
    return out


def main() -> int:
    spec19 = _load_module(SPEC19_MATERIALIZER, "materialize_spec19_scene_bundle_v7")
    base = spec19._load_base_module()

    ap = argparse.ArgumentParser(description="Materialize a unified spec19 curriculum and instruction-style SFT rows")
    ap.add_argument("--workspace", required=True, type=Path, help="Destination workspace")
    ap.add_argument("--seed-workspace", default=str(base.DEFAULT_SEED_WORKSPACE), type=Path, help="Seed spec workspace template to copy")
    ap.add_argument("--prefix", default="spec19_scene_bundle", help="Dataset prefix")
    ap.add_argument("--freeze-tokenizer-run", required=True, type=Path, help="Run directory whose tokenizer should be copied unchanged")
    ap.add_argument("--source-run", action="append", dest="source_runs", type=Path, required=True, help="Completed run whose stage train rows should seed the unified curriculum")
    ap.add_argument("--weight-quantum", type=int, default=5, help="Accepted for shared launcher compatibility; recorded in manifests")
    ap.add_argument("--python-exec", default=sys.executable, help="Python executable for generators")
    ap.add_argument("--train-variants", type=int, default=3, help="Instruction variants per train prompt")
    ap.add_argument("--eval-variants", type=int, default=2, help="Instruction variants per dev/test prompt")
    ap.add_argument("--line-name", default=DEFAULT_LINE_NAME, help="Recorded line name for manifests")
    ap.add_argument("--format-version", default=DEFAULT_FORMAT_VERSION, help="Recorded format version for manifests")
    ap.add_argument(
        "--train-surface-multiplier",
        action="append",
        dest="train_surface_multipliers",
        default=[],
        help="Optional weighting override in prompt_surface=count form; repeats train rows for that surface deterministically",
    )
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
        train_variants=int(args.train_variants),
        eval_variants=int(args.eval_variants),
        line_name=str(args.line_name),
        format_version=str(args.format_version),
        train_surface_multipliers=_parse_surface_multiplier(list(args.train_surface_multipliers or [])),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
