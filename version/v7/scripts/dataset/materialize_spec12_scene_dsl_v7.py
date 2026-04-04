#!/usr/bin/env python3
"""Materialize a cache-local spec12 scene DSL workspace."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SPEC10_MATERIALIZER = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec10_scene_dsl_v7.py"
STRUCTURED_GENERATOR = ROOT / "version" / "v7" / "scripts" / "generate_svg_structured_spec12_v7.py"
LINE_NAME = "spec12_scene_dsl"


def _load_spec10_materializer():
    spec = importlib.util.spec_from_file_location("materialize_spec10_scene_dsl_v7", SPEC10_MATERIALIZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load spec10 materializer: {SPEC10_MATERIALIZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene_tokens(output_tokens: str) -> list[str]:
    return [token for token in str(output_tokens or "").split() if token]


def _continuation_row(prompt: str, prefix_tokens: list[str], target_token: str, *, base: Any) -> str | None:
    if not prompt or not target_token:
        return None
    prompt_with_prefix = " ".join([prompt, *prefix_tokens]).strip()
    if not prompt_with_prefix:
        return None
    return base._row_from_catalog(prompt_with_prefix, target_token)


def _scene_anchor_rows(prompt: str, output_tokens: str, *, base: Any) -> list[str]:
    tokens = _scene_tokens(output_tokens)
    if len(tokens) < 2 or tokens[0] != "[scene]":
        return []
    # Force the model to learn the fixed scene-header order before the first block.
    limit = min(len(tokens) - 1, 6)
    rows: list[str] = []
    for idx in range(limit):
        row = _continuation_row(prompt, tokens[: idx + 1], tokens[idx + 1], base=base)
        if row:
            rows.append(row)
    return rows


def _scene_start_rows(prompt: str, output_tokens: str, *, base: Any) -> list[str]:
    tokens = _scene_tokens(output_tokens)
    if not tokens or tokens[0] != "[scene]":
        return []
    row = _continuation_row(prompt, [], tokens[0], base=base)
    return [row] if row else []


def _is_component_open(token: str) -> bool:
    if not (token.startswith("[") and token.endswith("]")):
        return False
    if token in {"[scene]", "[/scene]"} or token.startswith("[/"):
        return False
    body = token[1:-1]
    return ":" not in body


def _block_open_rows(prompt: str, output_tokens: str, *, base: Any) -> list[str]:
    tokens = _scene_tokens(output_tokens)
    rows: list[str] = []
    for idx, token in enumerate(tokens[:-1]):
        if not _is_component_open(token):
            continue
        row = _continuation_row(prompt, tokens[: idx + 1], tokens[idx + 1], base=base)
        if row:
            rows.append(row)
    return rows


def _block_close_rows(prompt: str, output_tokens: str, *, base: Any) -> list[str]:
    tokens = _scene_tokens(output_tokens)
    rows: list[str] = []
    for idx, token in enumerate(tokens):
        if not token.startswith("[/"):
            continue
        row = _continuation_row(prompt, tokens[:idx], token, base=base)
        if row:
            rows.append(row)
    return rows


def _post_block_transition_rows(prompt: str, output_tokens: str, *, base: Any) -> list[str]:
    tokens = _scene_tokens(output_tokens)
    rows: list[str] = []
    for idx, token in enumerate(tokens[:-1]):
        if not token.startswith("[/"):
            continue
        next_token = tokens[idx + 1]
        row = _continuation_row(prompt, tokens[: idx + 1], next_token, base=base)
        if row:
            rows.append(row)
    return rows


def _scene_tail_rows(prompt: str, output_tokens: str, *, base: Any, tail_tokens: int = 8) -> list[str]:
    tokens = _scene_tokens(output_tokens)
    if len(tokens) < 2:
        return []
    start = max(0, len(tokens) - max(1, int(tail_tokens)))
    rows: list[str] = []
    for idx in range(start, len(tokens)):
        row = _continuation_row(prompt, tokens[:idx], tokens[idx], base=base)
        if row:
            rows.append(row)
    return rows


def _restart_cleanup_rows(prompt: str, output_tokens: str, *, base: Any) -> list[str]:
    tokens = _scene_tokens(output_tokens)
    rows: list[str] = []
    if len(tokens) < 6 or tokens[0] != "[scene]":
        return rows

    contamination_specs = (
        (3, tokens[3] + "<|eos|><|bos|>[task:svg"),
        (4, tokens[4] + "<|eos|><|bos|>[task:svg"),
        (4, tokens[4] + "g" + tokens[4]),
    )
    for idx, bad_token in contamination_specs:
        if idx + 1 >= len(tokens):
            continue
        row = _continuation_row(prompt, [*tokens[:idx], bad_token], tokens[idx + 1], base=base)
        if row:
            rows.append(row)
    return rows


def _layout_from_prompt(prompt: str) -> str:
    for token in str(prompt or "").split():
        if token.startswith("[layout:") and token.endswith("]"):
            return token[len("[layout:") : -1]
    return ""


def _repeat_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _probe_report_rows(
    source_run: Path | None,
    *,
    split: str,
    exact_match: bool | None,
    base: Any,
) -> tuple[list[str], dict[str, int]]:
    if source_run is None:
        return [], {}
    probe_path = source_run / "spec12_probe_report.json"
    if not probe_path.exists():
        return [], {}
    try:
        report = json.loads(probe_path.read_text(encoding="utf-8"))
    except Exception:
        return [], {}

    rows: list[str] = []
    layout_counts: dict[str, int] = {}
    seen_prompts: set[str] = set()
    for item in report.get("results") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("split") or "") != split:
            continue
        if exact_match is not None and bool(item.get("exact_match")) != bool(exact_match):
            continue
        prompt = str(item.get("prompt") or "").strip()
        output_tokens = str(item.get("expected_output") or "").strip()
        if not prompt or not output_tokens or prompt in seen_prompts:
            continue
        row = base._row_from_catalog(prompt, output_tokens)
        if not row:
            continue
        rows.append(row)
        seen_prompts.add(prompt)
        layout = _layout_from_prompt(prompt) or "unknown"
        layout_counts[layout] = int(layout_counts.get(layout, 0)) + 1
    return rows, layout_counts


def _augment_midtrain_rows(
    workspace: Path,
    *,
    prefix: str,
    midtrain_shuffle_seed: int,
    midtrain_close_repeat: int,
    midtrain_header_repeat: int,
    midtrain_block_repeat: int,
    midtrain_transition_repeat: int,
    midtrain_table_repeat: int,
    midtrain_memory_repeat: int,
    midtrain_decision_repeat: int,
    midtrain_end_repeat: int,
    midtrain_start_repeat: int,
    midtrain_restart_repeat: int,
    midtrain_exact_replay_source_run: Path | None,
    midtrain_exact_replay_repeat: int,
    midtrain_train_failure_source_run: Path | None,
    midtrain_train_failure_repeat: int,
) -> dict[str, int]:
    spec10 = _load_spec10_materializer()
    base = spec10._load_base_module()

    render_catalog_path = workspace / "tokenizer" / f"{prefix}_render_catalog.json"
    if not render_catalog_path.exists():
        return {"repair_rows_added": 0, "anchor_rows_added": 0, "block_open_rows_added": 0, "block_close_rows_added": 0}

    render_catalog_rows = json.loads(render_catalog_path.read_text(encoding="utf-8"))
    train_repairs: list[str] = []
    anchor_added = 0
    block_open_added = 0
    block_close_added = 0
    transition_added = 0
    table_bonus_added = 0
    memory_bonus_added = 0
    decision_bonus_added = 0
    scene_tail_added = 0
    scene_start_added = 0
    restart_cleanup_added = 0
    exact_replay_added = 0
    failure_focus_added = 0

    for row in render_catalog_rows:
        if not isinstance(row, dict) or str(row.get("split") or "") != "train":
            continue
        prompt = str(row.get("prompt") or "").strip()
        output_tokens = str(row.get("output_tokens") or "").strip()
        if not prompt or not output_tokens:
            continue
        layout = _layout_from_prompt(prompt)
        layout_bonus = 1
        if layout == "table_matrix":
            layout_bonus = max(layout_bonus, int(midtrain_table_repeat))
        if layout == "memory_map":
            layout_bonus = max(layout_bonus, int(midtrain_memory_repeat))
        if layout == "decision_tree":
            layout_bonus = max(layout_bonus, int(midtrain_decision_repeat))

        for repair_row in _scene_start_rows(prompt, output_tokens, base=base):
            repeats = _repeat_count(midtrain_start_repeat) * layout_bonus
            for _ in range(repeats):
                train_repairs.append(repair_row)
                scene_start_added += 1
                if layout == "table_matrix" and layout_bonus > 1:
                    table_bonus_added += 1
                if layout == "memory_map" and layout_bonus > 1:
                    memory_bonus_added += 1
                if layout == "decision_tree" and layout_bonus > 1:
                    decision_bonus_added += 1

        for repair_row in _scene_anchor_rows(prompt, output_tokens, base=base):
            repeats = _repeat_count(midtrain_header_repeat) * layout_bonus
            for _ in range(repeats):
                train_repairs.append(repair_row)
                anchor_added += 1
                if layout == "table_matrix" and layout_bonus > 1:
                    table_bonus_added += 1
                if layout == "memory_map" and layout_bonus > 1:
                    memory_bonus_added += 1
                if layout == "decision_tree" and layout_bonus > 1:
                    decision_bonus_added += 1

        for repair_row in _block_open_rows(prompt, output_tokens, base=base):
            repeats = _repeat_count(midtrain_block_repeat) * layout_bonus
            for _ in range(repeats):
                train_repairs.append(repair_row)
                block_open_added += 1
                if layout == "table_matrix" and layout_bonus > 1:
                    table_bonus_added += 1
                if layout == "memory_map" and layout_bonus > 1:
                    memory_bonus_added += 1
                if layout == "decision_tree" and layout_bonus > 1:
                    decision_bonus_added += 1

        for repair_row in _block_close_rows(prompt, output_tokens, base=base):
            repeats = _repeat_count(midtrain_close_repeat) * layout_bonus
            for _ in range(repeats):
                train_repairs.append(repair_row)
                block_close_added += 1
                if layout == "table_matrix" and layout_bonus > 1:
                    table_bonus_added += 1
                if layout == "memory_map" and layout_bonus > 1:
                    memory_bonus_added += 1
                if layout == "decision_tree" and layout_bonus > 1:
                    decision_bonus_added += 1

        for repair_row in _post_block_transition_rows(prompt, output_tokens, base=base):
            repeats = _repeat_count(midtrain_transition_repeat) * layout_bonus
            for _ in range(repeats):
                train_repairs.append(repair_row)
                transition_added += 1
                if layout == "table_matrix" and layout_bonus > 1:
                    table_bonus_added += 1
                if layout == "memory_map" and layout_bonus > 1:
                    memory_bonus_added += 1
                if layout == "decision_tree" and layout_bonus > 1:
                    decision_bonus_added += 1

        for repair_row in _scene_tail_rows(prompt, output_tokens, base=base):
            repeats = _repeat_count(midtrain_end_repeat) * layout_bonus
            for _ in range(repeats):
                train_repairs.append(repair_row)
                scene_tail_added += 1
                if layout == "table_matrix" and layout_bonus > 1:
                    table_bonus_added += 1
                if layout == "memory_map" and layout_bonus > 1:
                    memory_bonus_added += 1
                if layout == "decision_tree" and layout_bonus > 1:
                    decision_bonus_added += 1

        for repair_row in _restart_cleanup_rows(prompt, output_tokens, base=base):
            repeats = _repeat_count(midtrain_restart_repeat) * layout_bonus
            for _ in range(repeats):
                train_repairs.append(repair_row)
                restart_cleanup_added += 1
                if layout == "table_matrix" and layout_bonus > 1:
                    table_bonus_added += 1
                if layout == "memory_map" and layout_bonus > 1:
                    memory_bonus_added += 1
                if layout == "decision_tree" and layout_bonus > 1:
                    decision_bonus_added += 1

    exact_replay_rows, exact_replay_layouts = _probe_report_rows(
        midtrain_exact_replay_source_run,
        split="train",
        exact_match=True,
        base=base,
    )
    for replay_row in exact_replay_rows:
        for _ in range(_repeat_count(midtrain_exact_replay_repeat)):
            train_repairs.append(replay_row)
            exact_replay_added += 1

    failure_focus_rows, failure_focus_layouts = _probe_report_rows(
        midtrain_train_failure_source_run,
        split="train",
        exact_match=False,
        base=base,
    )
    for replay_row in failure_focus_rows:
        for _ in range(_repeat_count(midtrain_train_failure_repeat)):
            train_repairs.append(replay_row)
            failure_focus_added += 1

    if not train_repairs:
        return {
            "repair_rows_added": 0,
            "scene_start_rows_added": 0,
            "anchor_rows_added": 0,
            "block_open_rows_added": 0,
            "block_close_rows_added": 0,
            "transition_rows_added": 0,
            "table_bonus_rows_added": 0,
            "memory_bonus_rows_added": 0,
            "decision_bonus_rows_added": 0,
            "scene_tail_rows_added": 0,
            "restart_cleanup_rows_added": 0,
            "exact_replay_rows_added": 0,
            "train_failure_focus_rows_added": 0,
        }

    midtrain_train = workspace / "midtrain" / "train" / f"{prefix}_midtrain_train.txt"
    existing_rows = midtrain_train.read_text(encoding="utf-8").splitlines()
    merged_rows = existing_rows + train_repairs
    random.Random(int(midtrain_shuffle_seed) + 1012).shuffle(merged_rows)
    midtrain_train.write_text("\n".join(merged_rows) + "\n", encoding="utf-8")

    manifest_path = workspace / "manifests" / f"{prefix}_workspace_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stages = manifest.setdefault("stages", {})
        midtrain = stages.setdefault("midtrain", {})
        counts = midtrain.setdefault("counts", {})
        counts["train_rows"] = len(merged_rows)
        counts["repair_rows_added"] = len(train_repairs)
        counts["scene_start_rows_added"] = scene_start_added
        counts["scene_anchor_rows_added"] = anchor_added
        counts["block_open_rows_added"] = block_open_added
        counts["block_close_rows_added"] = block_close_added
        counts["block_transition_rows_added"] = transition_added
        counts["table_bonus_rows_added"] = table_bonus_added
        counts["memory_bonus_rows_added"] = memory_bonus_added
        counts["decision_bonus_rows_added"] = decision_bonus_added
        counts["scene_tail_rows_added"] = scene_tail_added
        counts["restart_cleanup_rows_added"] = restart_cleanup_added
        counts["exact_replay_rows_added"] = exact_replay_added
        counts["train_failure_focus_rows_added"] = failure_focus_added
        if exact_replay_layouts:
            counts["exact_replay_layout_counts"] = dict(sorted(exact_replay_layouts.items()))
        if failure_focus_layouts:
            counts["train_failure_focus_layout_counts"] = dict(sorted(failure_focus_layouts.items()))
        note = (
            "Spec12 repair rows add scene-start anchors, scene-header anchors, "
            "component-open rows, block-close rows, post-close transition rows, "
            "scene-tail anchors, restart-cleanup rows, probe-derived exact replay rows, "
            "and probe-derived train-failure focus rows; table_matrix, memory_map, "
            "and decision_tree receive targeted ordering pressure."
        )
        existing_notes = str(midtrain.get("notes") or "").strip()
        midtrain["notes"] = f"{existing_notes} {note}".strip()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    return {
        "repair_rows_added": len(train_repairs),
        "scene_start_rows_added": scene_start_added,
        "anchor_rows_added": anchor_added,
        "block_open_rows_added": block_open_added,
        "block_close_rows_added": block_close_added,
        "transition_rows_added": transition_added,
        "table_bonus_rows_added": table_bonus_added,
        "memory_bonus_rows_added": memory_bonus_added,
        "decision_bonus_rows_added": decision_bonus_added,
        "scene_tail_rows_added": scene_tail_added,
        "restart_cleanup_rows_added": restart_cleanup_added,
        "exact_replay_rows_added": exact_replay_added,
        "train_failure_focus_rows_added": failure_focus_added,
    }


def materialize_workspace(
    workspace: Path,
    *,
    seed_workspace: Path,
    prefix: str,
    train_repeats: int,
    holdout_repeats: int,
    midtrain_edit_repeat: int,
    midtrain_direct_repeat: int,
    midtrain_close_repeat: int,
    midtrain_header_repeat: int,
    midtrain_block_repeat: int,
    midtrain_transition_repeat: int,
    midtrain_table_repeat: int,
    midtrain_memory_repeat: int,
    midtrain_decision_repeat: int,
    midtrain_end_repeat: int,
    midtrain_start_repeat: int,
    midtrain_restart_repeat: int,
    midtrain_exact_replay_source_run: Path | None,
    midtrain_exact_replay_repeat: int,
    midtrain_train_failure_source_run: Path | None,
    midtrain_train_failure_repeat: int,
    midtrain_shuffle_seed: int,
    python_exec: str,
    force: bool,
) -> dict[str, object]:
    spec10 = _load_spec10_materializer()
    original_generator = spec10.STRUCTURED_GENERATOR
    original_line_name = spec10.LINE_NAME
    try:
        spec10.STRUCTURED_GENERATOR = STRUCTURED_GENERATOR
        spec10.LINE_NAME = LINE_NAME
        summary = spec10.materialize_workspace(
            workspace,
            seed_workspace=seed_workspace,
            prefix=prefix,
            train_repeats=train_repeats,
            holdout_repeats=holdout_repeats,
            midtrain_edit_repeat=midtrain_edit_repeat,
            midtrain_direct_repeat=midtrain_direct_repeat,
            midtrain_close_repeat=midtrain_close_repeat,
            midtrain_shuffle_seed=midtrain_shuffle_seed,
            python_exec=python_exec,
            force=force,
        )
        repair_stats = _augment_midtrain_rows(
            workspace,
            prefix=prefix,
            midtrain_shuffle_seed=midtrain_shuffle_seed,
            midtrain_close_repeat=midtrain_close_repeat,
            midtrain_header_repeat=midtrain_header_repeat,
            midtrain_block_repeat=midtrain_block_repeat,
            midtrain_transition_repeat=midtrain_transition_repeat,
            midtrain_table_repeat=midtrain_table_repeat,
            midtrain_memory_repeat=midtrain_memory_repeat,
            midtrain_decision_repeat=midtrain_decision_repeat,
            midtrain_end_repeat=midtrain_end_repeat,
            midtrain_start_repeat=midtrain_start_repeat,
            midtrain_restart_repeat=midtrain_restart_repeat,
            midtrain_exact_replay_source_run=midtrain_exact_replay_source_run,
            midtrain_exact_replay_repeat=midtrain_exact_replay_repeat,
            midtrain_train_failure_source_run=midtrain_train_failure_source_run,
            midtrain_train_failure_repeat=midtrain_train_failure_repeat,
        )
        summary.update(repair_stats)
        midtrain_train = workspace / "midtrain" / "train" / f"{prefix}_midtrain_train.txt"
        if midtrain_train.exists():
            summary["midtrain_train_rows"] = len(midtrain_train.read_text(encoding="utf-8").splitlines())
        return summary
    finally:
        spec10.STRUCTURED_GENERATOR = original_generator
        spec10.LINE_NAME = original_line_name


def main() -> int:
    spec10 = _load_spec10_materializer()
    base = spec10._load_base_module()

    import argparse

    ap = argparse.ArgumentParser(description="Materialize a cache-local spec12 scene DSL workspace")
    ap.add_argument("--workspace", required=True, type=Path, help="Destination workspace")
    ap.add_argument("--seed-workspace", default=str(base.DEFAULT_SEED_WORKSPACE), type=Path, help="Seed spec workspace template to copy")
    ap.add_argument("--prefix", default="spec12_scene_dsl", help="Artifact prefix for generated files")
    ap.add_argument("--train-repeats", type=int, default=4, help="How many times to repeat each seen combo in pretrain/train")
    ap.add_argument("--holdout-repeats", type=int, default=1, help="How many times to repeat each holdout combo before dev/test split")
    ap.add_argument("--midtrain-edit-repeat", type=int, default=3, help="How many times to repeat each semantic midtrain edit row in train")
    ap.add_argument("--midtrain-direct-repeat", type=int, default=6, help="Minimum number of direct scene rows to keep per train example in midtrain")
    ap.add_argument("--midtrain-close-repeat", type=int, default=8, help="How many times to repeat each explicit closing-tag continuation row in midtrain train")
    ap.add_argument("--midtrain-header-repeat", type=int, default=6, help="How many times to repeat each scene-header anchor continuation row in midtrain train")
    ap.add_argument("--midtrain-block-repeat", type=int, default=4, help="How many times to repeat each component-open continuation row in midtrain train")
    ap.add_argument("--midtrain-transition-repeat", type=int, default=4, help="How many times to repeat each post-block transition row in midtrain train")
    ap.add_argument("--midtrain-table-repeat", type=int, default=2, help="Extra repeat multiplier applied to table_matrix repair rows")
    ap.add_argument("--midtrain-memory-repeat", type=int, default=2, help="Extra repeat multiplier applied to memory_map repair rows")
    ap.add_argument("--midtrain-decision-repeat", type=int, default=2, help="Extra repeat multiplier applied to decision_tree repair rows")
    ap.add_argument("--midtrain-end-repeat", type=int, default=4, help="How many times to repeat each scene-tail continuation row in midtrain train")
    ap.add_argument("--midtrain-start-repeat", type=int, default=4, help="How many times to repeat each prompt-to-[scene] continuation row in midtrain train")
    ap.add_argument("--midtrain-restart-repeat", type=int, default=3, help="How many times to repeat each restart-cleanup continuation row in midtrain train")
    ap.add_argument("--midtrain-exact-replay-source-run", type=Path, default=None, help="Optional source run dir whose exact train probe cases are replayed as full-scene anchors")
    ap.add_argument("--midtrain-exact-replay-repeat", type=int, default=0, help="How many times to repeat each exact train probe row from the source run")
    ap.add_argument("--midtrain-train-failure-source-run", type=Path, default=None, help="Optional source run dir whose failed train probe cases are replayed as full-scene focus rows")
    ap.add_argument("--midtrain-train-failure-repeat", type=int, default=0, help="How many times to repeat each failed train probe row from the source run")
    ap.add_argument("--midtrain-shuffle-seed", type=int, default=42, help="Shuffle seed for interleaving the semantic scene curriculum")
    ap.add_argument("--python-exec", default=str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable)
    ap.add_argument("--force", action="store_true", help="Replace an existing destination workspace")
    args = ap.parse_args()

    summary = materialize_workspace(
        args.workspace,
        seed_workspace=args.seed_workspace,
        prefix=str(args.prefix),
        train_repeats=int(args.train_repeats),
        holdout_repeats=int(args.holdout_repeats),
        midtrain_edit_repeat=max(0, int(args.midtrain_edit_repeat)),
        midtrain_direct_repeat=max(0, int(args.midtrain_direct_repeat)),
        midtrain_close_repeat=max(0, int(args.midtrain_close_repeat)),
        midtrain_header_repeat=max(0, int(args.midtrain_header_repeat)),
        midtrain_block_repeat=max(0, int(args.midtrain_block_repeat)),
        midtrain_transition_repeat=max(0, int(args.midtrain_transition_repeat)),
        midtrain_table_repeat=max(0, int(args.midtrain_table_repeat)),
        midtrain_memory_repeat=max(0, int(args.midtrain_memory_repeat)),
        midtrain_decision_repeat=max(0, int(args.midtrain_decision_repeat)),
        midtrain_end_repeat=max(0, int(args.midtrain_end_repeat)),
        midtrain_start_repeat=max(0, int(args.midtrain_start_repeat)),
        midtrain_restart_repeat=max(0, int(args.midtrain_restart_repeat)),
        midtrain_exact_replay_source_run=args.midtrain_exact_replay_source_run.expanduser().resolve() if args.midtrain_exact_replay_source_run else None,
        midtrain_exact_replay_repeat=max(0, int(args.midtrain_exact_replay_repeat)),
        midtrain_train_failure_source_run=args.midtrain_train_failure_source_run.expanduser().resolve() if args.midtrain_train_failure_source_run else None,
        midtrain_train_failure_repeat=max(0, int(args.midtrain_train_failure_repeat)),
        midtrain_shuffle_seed=int(args.midtrain_shuffle_seed),
        python_exec=str(args.python_exec),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
