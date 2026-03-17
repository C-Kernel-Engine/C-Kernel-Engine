#!/usr/bin/env python3
"""Materialize a cache-local spec08 rich scene DSL workspace."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
BASE_SCRIPT = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec06_structured_atoms_v7.py"
STRUCTURED_GENERATOR = ROOT / "version" / "v7" / "scripts" / "generate_svg_structured_spec08_v7.py"
_FLOW_REPAIR_TOPIC = "governance_path"
_FLOW_REPAIR_SOURCE_TOPIC = "gpu_readiness"
_FLOW_REPAIR_LAYOUT = "flow-steps"
_FLOW_REPAIR_BG = "mint"
_FLOW_REPAIR_FRAME = "card"
_FLOW_REPAIR_ACCENTS = ("blue", "green")
_FLOW_REPAIR_DENSITIES = ("airy", "compact")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_lines(path: Path, rows: list[str]) -> None:
    text = "\n".join(rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _parse_prompt_tags(prompt: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for token in str(prompt or "").split():
        if not (token.startswith("[") and token.endswith("]")):
            continue
        body = token[1:-1]
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        tags[key] = value
    return tags


def _prompt_from_tags(tags: dict[str, str], *, include_edit: str | None = None) -> str:
    tokens = [
        "[task:svg]",
        f"[layout:{tags.get('layout', 'bullet-panel')}]",
        f"[topic:{tags.get('topic', '')}]",
        f"[accent:{tags.get('accent', 'orange')}]",
        f"[bg:{tags.get('bg', 'paper')}]",
        f"[frame:{tags.get('frame', 'plain')}]",
        f"[density:{tags.get('density', 'compact')}]",
    ]
    if include_edit:
        tokens.append(f"[edit:{include_edit}]")
    tokens.append("[OUT]")
    return " ".join(tokens)


def _row_from_catalog(prompt: str, output_tokens: str) -> str:
    return f"{prompt} {output_tokens}".strip()


def _render_catalog_path(workspace: Path, prefix: str) -> Path:
    candidates = [
        workspace / "manifests" / "generated" / "structured_atoms" / f"{prefix}_render_catalog.json",
        workspace / "tokenizer" / f"{prefix}_render_catalog.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"spec08 render catalog not found for prefix '{prefix}' under {workspace}")


def _is_flow_repair_source(tags: dict[str, str]) -> bool:
    return (
        str(tags.get("layout") or "") == _FLOW_REPAIR_LAYOUT
        and str(tags.get("topic") or "") == _FLOW_REPAIR_SOURCE_TOPIC
        and str(tags.get("bg") or "") == _FLOW_REPAIR_BG
        and str(tags.get("frame") or "") == _FLOW_REPAIR_FRAME
        and str(tags.get("accent") or "") in set(_FLOW_REPAIR_ACCENTS)
        and str(tags.get("density") or "") in set(_FLOW_REPAIR_DENSITIES)
    )


def _is_spectrum_exact_booster(tags: dict[str, str]) -> bool:
    return (
        str(tags.get("layout") or "") == "spectrum-band"
        and str(tags.get("topic") or "") == "capacity_math"
        and str(tags.get("accent") or "") == "blue"
        and str(tags.get("bg") or "") == "slate"
        and str(tags.get("frame") or "") == "card"
        and str(tags.get("density") or "") == "airy"
    )


def _apply_spec08_r2_repairs(workspace: Path, *, prefix: str) -> dict[str, Any]:
    catalog_path = _render_catalog_path(workspace, prefix)
    catalog_rows = _load_json(catalog_path)
    if not isinstance(catalog_rows, list):
        raise RuntimeError(f"expected list render catalog at {catalog_path}")

    prompt_to_output: dict[str, str] = {}
    for row in catalog_rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        output_tokens = str(row.get("output_tokens") or "").strip()
        if prompt and output_tokens:
            prompt_to_output[prompt] = output_tokens

    midtrain_train = workspace / "midtrain" / "train" / f"{prefix}_midtrain_train.txt"
    rows = midtrain_train.read_text(encoding="utf-8").splitlines()
    extra_rows: list[str] = []
    flow_repairs = 0
    spectrum_boosters = 0

    seen_prompts = set((workspace / "holdout" / f"{prefix}_seen_prompts.txt").read_text(encoding="utf-8").splitlines())

    for prompt in sorted(seen_prompts):
        tags = _parse_prompt_tags(prompt)
        if _is_flow_repair_source(tags):
            target_tags = dict(tags)
            target_tags["topic"] = _FLOW_REPAIR_TOPIC
            target_prompt = _prompt_from_tags(target_tags)
            target_output = prompt_to_output.get(target_prompt)
            if not target_output:
                continue
            repair_prompt = _prompt_from_tags(tags, include_edit=f"topic={_FLOW_REPAIR_TOPIC}")
            repair_row = _row_from_catalog(repair_prompt, target_output)
            for _ in range(4):
                extra_rows.append(repair_row)
                flow_repairs += 1

        if _is_spectrum_exact_booster(tags):
            output_tokens = prompt_to_output.get(prompt)
            if not output_tokens:
                continue
            exact_row = _row_from_catalog(prompt, output_tokens)
            for _ in range(3):
                extra_rows.append(exact_row)
                spectrum_boosters += 1

    if extra_rows:
        rows.extend(extra_rows)
        _write_lines(midtrain_train, rows)

    return {
        "midtrain_train_rows_added": int(len(extra_rows)),
        "flow_topic_repairs_added": int(flow_repairs),
        "spectrum_exact_boosters_added": int(spectrum_boosters),
    }


def _load_base_module():
    spec = importlib.util.spec_from_file_location("materialize_spec06_base_v7", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load base materializer: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    base = _load_base_module()
    base.STRUCTURED_GENERATOR = STRUCTURED_GENERATOR
    base.LINE_NAME = "spec08_rich_scene_dsl"

    ap = argparse.ArgumentParser(description="Materialize a cache-local spec08 rich scene DSL workspace")
    ap.add_argument("--workspace", required=True, type=Path, help="Destination workspace")
    ap.add_argument("--seed-workspace", default=str(base.DEFAULT_SEED_WORKSPACE), type=Path, help="Seed spec workspace template to copy")
    ap.add_argument("--prefix", default="spec08_rich_scene_dsl", help="Artifact prefix for generated files")
    ap.add_argument("--train-repeats", type=int, default=3, help="How many times to repeat each seen combo in pretrain/train")
    ap.add_argument("--holdout-repeats", type=int, default=1, help="How many times to repeat each holdout combo before dev/test split")
    ap.add_argument("--midtrain-edit-repeat", type=int, default=1, help="How many times to repeat each midtrain edit row in train")
    ap.add_argument("--midtrain-shuffle-seed", type=int, default=42, help="Shuffle seed for interleaving the blended midtrain curriculum")
    ap.add_argument("--python-exec", default=str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable)
    ap.add_argument("--force", action="store_true", help="Replace an existing destination workspace")
    args = ap.parse_args()

    summary = base.materialize_workspace(
        args.workspace,
        seed_workspace=args.seed_workspace,
        prefix=str(args.prefix),
        train_repeats=int(args.train_repeats),
        holdout_repeats=int(args.holdout_repeats),
        midtrain_edit_repeat=max(1, int(args.midtrain_edit_repeat)),
        midtrain_shuffle_seed=int(args.midtrain_shuffle_seed),
        python_exec=str(args.python_exec),
        force=bool(args.force),
    )
    summary["spec08_r2_repairs"] = _apply_spec08_r2_repairs(args.workspace, prefix=str(args.prefix))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
