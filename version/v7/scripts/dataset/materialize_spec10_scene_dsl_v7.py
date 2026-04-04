#!/usr/bin/env python3
"""Materialize a cache-local spec10 asset-grounded scene DSL workspace."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
BASE_SCRIPT = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec06_structured_atoms_v7.py"
STRUCTURED_GENERATOR = ROOT / "version" / "v7" / "scripts" / "generate_svg_structured_spec10_v7.py"
LINE_NAME = "spec10_asset_scene_dsl"
LAYOUTS = (
    "poster_stack",
    "comparison_span_chart",
    "pipeline_lane",
    "dual_panel_compare",
    "dashboard_cards",
)
TOPICS = (
    "cpu_gpu_cost",
    "ethernet_equalizer",
    "memory_reality",
    "performance_balance",
    "pipeline_overview",
    "training_intuition",
)
THEMES = ("infra_dark", "paper_editorial", "signal_glow")
TONES = ("amber", "green", "blue", "purple", "mixed")
DENSITIES = ("compact", "balanced", "airy")


def _load_base_module():
    spec = importlib.util.spec_from_file_location("materialize_spec06_base_v7", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load base materializer: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _topic_phrase(topic: str) -> str:
    return str(topic or "").replace("_", " ")


def _layout_phrase(layout: str) -> str:
    return {
        "poster_stack": "stacked poster infographic",
        "comparison_span_chart": "comparison chart infographic",
        "pipeline_lane": "pipeline flow infographic",
        "dual_panel_compare": "dual panel comparison infographic",
        "dashboard_cards": "dashboard card infographic",
    }.get(str(layout or ""), "infographic")


def _style_phrase(theme: str, tone: str, density: str) -> str:
    theme_phrase = {
        "infra_dark": "an infra dark",
        "paper_editorial": "a paper editorial",
        "signal_glow": "a signal glow",
    }.get(theme, "a structured")
    density_phrase = {
        "compact": "compact",
        "balanced": "balanced",
        "airy": "airy",
    }.get(density, "balanced")
    tone_phrase = str(tone or "blue").replace("_", " ")
    return f"{theme_phrase} style with {tone_phrase} tone and {density_phrase} spacing"


def _sft_instruction_variants(prompt: str) -> list[str]:
    tags = _parse_prompt_tags(prompt)
    layout = _layout_phrase(str(tags.get("layout") or ""))
    topic = _topic_phrase(str(tags.get("topic") or ""))
    style = _style_phrase(
        str(tags.get("theme") or THEMES[0]),
        str(tags.get("tone") or TONES[0]),
        str(tags.get("density") or DENSITIES[1]),
    )
    variants = [
        f"Create a {layout} about {topic} in {style}. Return scene DSL only. [OUT]",
        f"Make a {layout} for {topic} using {style}. Output structured scene tags only. [OUT]",
        f"Create a complete {layout} about {topic} in {style}. Start with [scene] and end with [/scene]. [OUT]",
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for row in variants:
        if row not in seen:
            seen.add(row)
            deduped.append(row)
    return deduped


def _layout_repeat(layout: str) -> int:
    return {
        "poster_stack": 3,
        "comparison_span_chart": 3,
        "pipeline_lane": 3,
        "dual_panel_compare": 3,
        "dashboard_cards": 3,
    }.get(layout, 2)


def _valid_style_combo(theme: str, tone: str) -> bool:
    allowed = {
        "infra_dark": {"amber", "green", "blue", "purple", "mixed"},
        "paper_editorial": {"amber", "green", "blue", "mixed"},
        "signal_glow": {"green", "blue", "purple"},
    }
    return tone in allowed.get(theme, set())


def _prompt_from_tags(tags: dict[str, str], *, include_edit: str | None = None) -> str:
    tokens = [
        "[task:svg]",
        f"[layout:{tags.get('layout', LAYOUTS[0])}]",
        f"[topic:{tags.get('topic', TOPICS[0])}]",
        f"[theme:{tags.get('theme', THEMES[0])}]",
        f"[tone:{tags.get('tone', TONES[0])}]",
        f"[density:{tags.get('density', DENSITIES[1])}]",
    ]
    if include_edit:
        tokens.append(f"[edit:{include_edit}]")
    tokens.append("[OUT]")
    return " ".join(tokens)


def _alt_value(current: str, values: tuple[str, ...], *, limit: int) -> list[str]:
    out = [value for value in values if value != current]
    return out[: max(0, int(limit))]


def _topic_edit_targets(current: str, *, limit: int) -> list[str]:
    ordered: list[str] = []
    topics = list(TOPICS)
    if current in topics:
        idx = topics.index(current)
        for offset in (1, -1, 2, -2):
            candidate = topics[(idx + offset) % len(topics)]
            if candidate != current and candidate not in ordered:
                ordered.append(candidate)
    else:
        ordered.extend(topics)
    return ordered[: max(0, int(limit))]


def _related_layouts(layout: str) -> tuple[str, ...]:
    return {
        "poster_stack": ("dashboard_cards", "dual_panel_compare"),
        "comparison_span_chart": ("dual_panel_compare", "dashboard_cards"),
        "pipeline_lane": ("dashboard_cards", "comparison_span_chart"),
        "dual_panel_compare": ("comparison_span_chart", "poster_stack"),
        "dashboard_cards": ("poster_stack", "pipeline_lane"),
    }.get(str(layout or ""), ())


def _candidate_semantic_edits(tags: dict[str, str]) -> list[tuple[str, dict[str, str]]]:
    current_theme = str(tags.get("theme") or THEMES[0])
    current_tone = str(tags.get("tone") or TONES[0])
    current_density = str(tags.get("density") or DENSITIES[1])
    current_topic = str(tags.get("topic") or TOPICS[0])
    current_layout = str(tags.get("layout") or LAYOUTS[0])

    edits: list[tuple[str, dict[str, str]]] = []
    for density_alt in _alt_value(current_density, DENSITIES, limit=2):
        target = dict(tags)
        target["density"] = density_alt
        edits.append((f"density={density_alt}", target))

    for tone_alt in _alt_value(current_tone, TONES, limit=2):
        if not _valid_style_combo(current_theme, tone_alt):
            continue
        target = dict(tags)
        target["tone"] = tone_alt
        edits.append((f"tone={tone_alt}", target))

    theme_alts = [theme for theme in THEMES if theme != current_theme and _valid_style_combo(theme, current_tone)]
    for theme_alt in theme_alts[:1]:
        target = dict(tags)
        target["theme"] = theme_alt
        edits.append((f"theme={theme_alt}", target))

    for topic_alt in _topic_edit_targets(current_topic, limit=1):
        target = dict(tags)
        target["topic"] = topic_alt
        edits.append((f"topic={topic_alt}", target))

    for layout_alt in _related_layouts(current_layout)[:1]:
        target = dict(tags)
        target["layout"] = layout_alt
        edits.append((f"layout={layout_alt}", target))

    return edits


def _boosted_edit_repeat(edit_op: str, base_repeat: int) -> int:
    kind = str(edit_op).split("=", 1)[0]
    repeat = max(1, int(base_repeat))
    if kind in {"density", "tone", "theme"}:
        return max(repeat, 2)
    if kind == "topic":
        return max(repeat, 2)
    if kind == "layout":
        return max(repeat, 1)
    if kind == "canon":
        return max(repeat, 4)
    return repeat


def _scene_tokens(output_tokens: str) -> list[str]:
    return [token for token in str(output_tokens or "").split() if token]


def _closing_row(prompt: str, output_tokens: str, *, base) -> str | None:
    tokens = _scene_tokens(output_tokens)
    if len(tokens) < 2 or tokens[-1] != "[/scene]":
        return None
    prompt_with_prefix = f"{prompt} {' '.join(tokens[:-1])}".strip()
    return base._row_from_catalog(prompt_with_prefix, tokens[-1])


def _build_midtrain_rows(
    catalog_rows: list[dict[str, Any]],
    *,
    is_train: bool,
    shuffle_seed: int,
    direct_repeat: int,
    edit_repeat: int,
    close_repeat: int,
    base,
) -> list[str]:
    prompt_index = {str(row.get("prompt") or ""): row for row in catalog_rows}
    out_rows: list[str] = []
    for row in catalog_rows:
        prompt = str(row.get("prompt") or "").strip()
        output_tokens = str(row.get("output_tokens") or "").strip()
        if not prompt or not output_tokens:
            continue
        tags = _parse_prompt_tags(prompt)
        stage_row = base._row_from_catalog(prompt, output_tokens)
        repeats = max(_layout_repeat(str(tags.get("layout") or "")), int(direct_repeat)) if is_train else 1
        for _ in range(repeats):
            out_rows.append(stage_row)
        if is_train:
            canon_prompt = _prompt_from_tags(tags, include_edit="canon")
            canon_row = base._row_from_catalog(canon_prompt, output_tokens)
            for _ in range(_boosted_edit_repeat("canon", edit_repeat)):
                out_rows.append(canon_row)
            closing_row = _closing_row(prompt, output_tokens, base=base)
            if closing_row:
                for _ in range(max(1, int(close_repeat))):
                    out_rows.append(closing_row)
            for edit_op, target_tags in _candidate_semantic_edits(tags):
                target_prompt = _prompt_from_tags(target_tags)
                target = prompt_index.get(target_prompt)
                if target is None:
                    continue
                edit_prompt = _prompt_from_tags(tags, include_edit=edit_op)
                edit_row = base._row_from_catalog(edit_prompt, str(target.get("output_tokens") or ""))
                for _ in range(_boosted_edit_repeat(edit_op, edit_repeat)):
                    out_rows.append(edit_row)
    rng = random.Random(int(shuffle_seed))
    rng.shuffle(out_rows)
    return out_rows


def _summarize_rows(rows: list[str]) -> dict[str, Any]:
    layout_counts: Counter[str] = Counter()
    theme_counts: Counter[str] = Counter()
    tone_counts: Counter[str] = Counter()
    density_counts: Counter[str] = Counter()
    for row in rows:
        tags = _parse_prompt_tags(row)
        layout_counts[str(tags.get("layout") or "unknown")] += 1
        theme_counts[str(tags.get("theme") or "unknown")] += 1
        tone_counts[str(tags.get("tone") or "unknown")] += 1
        density_counts[str(tags.get("density") or "unknown")] += 1
    return {
        "total_rows": len(rows),
        "layout_counts": dict(sorted(layout_counts.items())),
        "theme_counts": dict(sorted(theme_counts.items())),
        "tone_counts": dict(sorted(tone_counts.items())),
        "density_counts": dict(sorted(density_counts.items())),
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
    midtrain_shuffle_seed: int,
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

    original_generator = base.STRUCTURED_GENERATOR
    try:
        base.STRUCTURED_GENERATOR = STRUCTURED_GENERATOR
        generated_dir = base._run_generator(
            workspace,
            prefix=prefix,
            train_repeats=train_repeats,
            holdout_repeats=holdout_repeats,
            python_exec=python_exec,
        )
    finally:
        base.STRUCTURED_GENERATOR = original_generator

    source_manifest = base._load_json(generated_dir / f"{prefix}_manifest.json")
    render_catalog_rows = json.loads((generated_dir / f"{prefix}_render_catalog.json").read_text(encoding="utf-8"))

    train_rows = (generated_dir / f"{prefix}_train.txt").read_text(encoding="utf-8").splitlines()
    holdout_rows = (generated_dir / f"{prefix}_holdout.txt").read_text(encoding="utf-8").splitlines()
    tokenizer_corpus_rows = (generated_dir / f"{prefix}_tokenizer_corpus.txt").read_text(encoding="utf-8").splitlines()
    seen_prompt_rows = (generated_dir / f"{prefix}_seen_prompts.txt").read_text(encoding="utf-8").splitlines()
    holdout_prompt_rows = (generated_dir / f"{prefix}_holdout_prompts.txt").read_text(encoding="utf-8").splitlines()
    hidden_seen_prompt_rows = (
        (generated_dir / f"{prefix}_hidden_seen_prompts.txt").read_text(encoding="utf-8").splitlines()
        if (generated_dir / f"{prefix}_hidden_seen_prompts.txt").exists()
        else []
    )
    hidden_holdout_prompt_rows = (
        (generated_dir / f"{prefix}_hidden_holdout_prompts.txt").read_text(encoding="utf-8").splitlines()
        if (generated_dir / f"{prefix}_hidden_holdout_prompts.txt").exists()
        else []
    )
    reserved_token_rows = (generated_dir / f"{prefix}_reserved_control_tokens.txt").read_text(encoding="utf-8").splitlines()

    dev_rows, test_rows = base._split_even(holdout_rows)
    holdout_prompt_dev, holdout_prompt_test = base._split_even(holdout_prompt_rows)

    pretrain_train = workspace / "pretrain" / "train" / f"{prefix}_pretrain_train.txt"
    pretrain_dev = workspace / "pretrain" / "dev" / f"{prefix}_pretrain_dev.txt"
    pretrain_test = workspace / "pretrain" / "test" / f"{prefix}_pretrain_test.txt"
    base._write_lines(pretrain_train, train_rows)
    base._write_lines(pretrain_dev, dev_rows)
    base._write_lines(pretrain_test, test_rows)

    sft_train_rows: list[str] = []
    sft_holdout_rows: list[str] = []
    midtrain_train_catalog: list[dict[str, Any]] = []
    midtrain_holdout_catalog: list[dict[str, Any]] = []
    for row in render_catalog_rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        output_tokens = str(row.get("output_tokens") or "").strip()
        split = str(row.get("split") or "").strip()
        tags = _parse_prompt_tags(prompt)
        layout = str(tags.get("layout") or "")
        if not prompt or not output_tokens or split not in {"train", "holdout"}:
            continue
        stage_row = base._row_from_catalog(prompt, output_tokens)
        instruction_rows = [base._row_from_catalog(instruction, output_tokens) for instruction in _sft_instruction_variants(prompt)]
        catalog_row = {
            "prompt": prompt,
            "output_tokens": output_tokens,
            "split": split,
            "tags": dict(tags),
        }
        if split == "train":
            midtrain_train_catalog.append(catalog_row)
            sft_train_rows.extend([stage_row] * _layout_repeat(layout))
            sft_train_rows.extend(instruction_rows)
        else:
            midtrain_holdout_catalog.append(catalog_row)
            sft_holdout_rows.append(stage_row)
            sft_holdout_rows.extend(instruction_rows[:1])

    midtrain_holdout_dev_catalog, midtrain_holdout_test_catalog = base._split_catalog_even_by_layout(midtrain_holdout_catalog)
    midtrain_train_rows = _build_midtrain_rows(
        midtrain_train_catalog,
        is_train=True,
        shuffle_seed=int(midtrain_shuffle_seed),
        direct_repeat=int(midtrain_direct_repeat),
        edit_repeat=int(midtrain_edit_repeat),
        close_repeat=int(midtrain_close_repeat),
        base=base,
    )
    midtrain_dev_rows = _build_midtrain_rows(
        midtrain_holdout_dev_catalog,
        is_train=False,
        shuffle_seed=int(midtrain_shuffle_seed) + 1,
        direct_repeat=1,
        edit_repeat=1,
        close_repeat=1,
        base=base,
    )
    midtrain_test_rows = _build_midtrain_rows(
        midtrain_holdout_test_catalog,
        is_train=False,
        shuffle_seed=int(midtrain_shuffle_seed) + 2,
        direct_repeat=1,
        edit_repeat=1,
        close_repeat=1,
        base=base,
    )
    sft_dev_rows, sft_test_rows = base._split_even(sft_holdout_rows)

    midtrain_train = workspace / "midtrain" / "train" / f"{prefix}_midtrain_train.txt"
    midtrain_dev = workspace / "midtrain" / "dev" / f"{prefix}_midtrain_dev.txt"
    midtrain_test = workspace / "midtrain" / "test" / f"{prefix}_midtrain_test.txt"
    base._write_lines(midtrain_train, midtrain_train_rows)
    base._write_lines(midtrain_dev, midtrain_dev_rows)
    base._write_lines(midtrain_test, midtrain_test_rows)

    sft_train = workspace / "sft" / "train" / f"{prefix}_sft_train.txt"
    sft_dev = workspace / "sft" / "dev" / f"{prefix}_sft_dev.txt"
    sft_test = workspace / "sft" / "test" / f"{prefix}_sft_test.txt"
    base._write_lines(sft_train, sft_train_rows)
    base._write_lines(sft_dev, sft_dev_rows)
    base._write_lines(sft_test, sft_test_rows)

    holdout_dir = workspace / "holdout"
    base._write_lines(holdout_dir / f"{prefix}_holdout_prompts.txt", holdout_prompt_rows)
    base._write_lines(holdout_dir / f"{prefix}_holdout_prompts_dev.txt", holdout_prompt_dev)
    base._write_lines(holdout_dir / f"{prefix}_holdout_prompts_test.txt", holdout_prompt_test)
    base._write_lines(holdout_dir / f"{prefix}_seen_prompts.txt", seen_prompt_rows)
    if hidden_seen_prompt_rows:
        base._write_lines(holdout_dir / f"{prefix}_hidden_seen_prompts.txt", hidden_seen_prompt_rows)
    if hidden_holdout_prompt_rows:
        base._write_lines(holdout_dir / f"{prefix}_hidden_holdout_prompts.txt", hidden_holdout_prompt_rows)

    tokenizer_dir = workspace / "tokenizer"
    tokenizer_json_src = generated_dir / f"{prefix}_tokenizer" / "tokenizer.json"
    tokenizer_bin_src = generated_dir / f"{prefix}_tokenizer" / "tokenizer_bin"
    probe_contract_src = generated_dir / f"{prefix}_probe_report_contract.json"
    eval_contract_src = generated_dir / f"{prefix}_eval_contract.json"
    tokenizer_json_dst = tokenizer_dir / "tokenizer.json"
    tokenizer_bin_dst = tokenizer_dir / "tokenizer_bin"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    base.shutil.copy2(tokenizer_json_src, tokenizer_json_dst)
    base._copy_tree(tokenizer_bin_src, tokenizer_bin_dst)
    base.shutil.copy2(generated_dir / f"{prefix}_vocab.json", tokenizer_dir / f"{prefix}_vocab.json")
    base.shutil.copy2(generated_dir / f"{prefix}_render_catalog.json", tokenizer_dir / f"{prefix}_render_catalog.json")
    if probe_contract_src.exists():
        base._rewrite_probe_contract(
            probe_contract_src,
            tokenizer_dir / f"{prefix}_probe_report_contract.json",
            catalog_path=f"{prefix}_render_catalog.json",
            train_path=f"{prefix}_seen_prompts.txt",
            test_path=f"{prefix}_holdout_prompts.txt",
        )
    base._write_lines(tokenizer_dir / f"{prefix}_tokenizer_corpus.txt", tokenizer_corpus_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_reserved_control_tokens.txt", reserved_token_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_seen_prompts.txt", seen_prompt_rows)
    base._write_lines(tokenizer_dir / f"{prefix}_holdout_prompts.txt", holdout_prompt_rows)
    if hidden_seen_prompt_rows:
        base._write_lines(tokenizer_dir / f"{prefix}_hidden_seen_prompts.txt", hidden_seen_prompt_rows)
    if hidden_holdout_prompt_rows:
        base._write_lines(tokenizer_dir / f"{prefix}_hidden_holdout_prompts.txt", hidden_holdout_prompt_rows)

    contracts_dir = workspace / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    if probe_contract_src.exists():
        base._rewrite_probe_contract(
            probe_contract_src,
            contracts_dir / f"{prefix}_probe_report_contract.json",
            catalog_path=f"../manifests/generated/structured_atoms/{prefix}_render_catalog.json",
            train_path=f"../holdout/{prefix}_seen_prompts.txt",
            test_path=f"../holdout/{prefix}_holdout_prompts.txt",
        )
    if eval_contract_src.exists():
        base.shutil.copy2(eval_contract_src, contracts_dir / "eval_contract.svg.v1.json")

    tokenizer_manifest = {
        "format": "ck.structured_svg_atoms.tokenizer_manifest.v1",
        "line": LINE_NAME,
        "workspace": str(workspace),
        "tokenizer_rows": len(tokenizer_corpus_rows),
        "tag_seed_rows": len(seen_prompt_rows) + len(holdout_prompt_rows),
        "kind_counts": {
            "tokenizer_corpus": len(tokenizer_corpus_rows),
            "tag_seed": len(seen_prompt_rows) + len(holdout_prompt_rows),
            "reserved_control_tokens": len(reserved_token_rows),
        },
        "artifacts": {
            "tokenizer_corpus": base._rel(tokenizer_dir / f"{prefix}_tokenizer_corpus.txt", workspace),
            "reserved_control_tokens": base._rel(tokenizer_dir / f"{prefix}_reserved_control_tokens.txt", workspace),
            "seen_prompts": base._rel(tokenizer_dir / f"{prefix}_seen_prompts.txt", workspace),
            "holdout_prompts": base._rel(tokenizer_dir / f"{prefix}_holdout_prompts.txt", workspace),
            "hidden_seen_prompts": base._rel(tokenizer_dir / f"{prefix}_hidden_seen_prompts.txt", workspace)
            if hidden_seen_prompt_rows
            else None,
            "hidden_holdout_prompts": base._rel(tokenizer_dir / f"{prefix}_hidden_holdout_prompts.txt", workspace)
            if hidden_holdout_prompt_rows
            else None,
            "tokenizer_json": base._rel(tokenizer_json_dst, workspace),
            "tokenizer_bin": base._rel(tokenizer_bin_dst, workspace),
            "vocab": base._rel(tokenizer_dir / f"{prefix}_vocab.json", workspace),
            "render_catalog": base._rel(tokenizer_dir / f"{prefix}_render_catalog.json", workspace),
            "probe_report_contract": base._rel(tokenizer_dir / f"{prefix}_probe_report_contract.json", workspace)
            if probe_contract_src.exists()
            else None,
            "eval_contract": base._rel(contracts_dir / "eval_contract.svg.v1.json", workspace)
            if eval_contract_src.exists()
            else None,
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
        "train_repeats": int(train_repeats),
        "holdout_repeats": int(holdout_repeats),
        "midtrain_edit_repeat": int(midtrain_edit_repeat),
        "midtrain_direct_repeat": int(midtrain_direct_repeat),
        "midtrain_close_repeat": int(midtrain_close_repeat),
        "midtrain_shuffle_seed": int(midtrain_shuffle_seed),
        "request_axes": {
            "layouts": list(LAYOUTS),
            "topics": list(TOPICS),
            "themes": list(THEMES),
            "tones": list(TONES),
            "densities": list(DENSITIES),
        },
        "stages": {
            "pretrain": {
                "train": base._rel(pretrain_train, workspace),
                "dev": base._rel(pretrain_dev, workspace),
                "test": base._rel(pretrain_test, workspace),
                "counts": {
                    "train_rows": len(train_rows),
                    "dev_rows": len(dev_rows),
                    "test_rows": len(test_rows),
                },
            },
            "midtrain": {
                "train": base._rel(midtrain_train, workspace),
                "dev": base._rel(midtrain_dev, workspace),
                "test": base._rel(midtrain_test, workspace),
                "notes": "Semantic scene DSL reinforcement with direct rows, canon-format rows, closure-continuation rows, and semantic edit rows over theme/tone/density/topic/layout.",
                "counts": {
                    "train_rows": len(midtrain_train_rows),
                    "dev_rows": len(midtrain_dev_rows),
                    "test_rows": len(midtrain_test_rows),
                    "promoted_holdout_rows": 0,
                    "train_summary": _summarize_rows(midtrain_train_rows),
                    "dev_summary": _summarize_rows(midtrain_dev_rows),
                    "test_summary": _summarize_rows(midtrain_test_rows),
                },
            },
            "sft": {
                "train": base._rel(sft_train, workspace),
                "dev": base._rel(sft_dev, workspace),
                "test": base._rel(sft_test, workspace),
                "notes": "Mixed DSL and semantic natural-language prompts aligned to theme/tone/density rather than legacy accent/bg/frame controls.",
                "counts": {
                    "train_rows": len(sft_train_rows),
                    "dev_rows": len(sft_dev_rows),
                    "test_rows": len(sft_test_rows),
                },
            },
        },
        "holdout": {
            "prompts_all": base._rel(holdout_dir / f"{prefix}_holdout_prompts.txt", workspace),
            "prompts_dev": base._rel(holdout_dir / f"{prefix}_holdout_prompts_dev.txt", workspace),
            "prompts_test": base._rel(holdout_dir / f"{prefix}_holdout_prompts_test.txt", workspace),
            "seen_prompts": base._rel(holdout_dir / f"{prefix}_seen_prompts.txt", workspace),
            "hidden_seen_prompts": base._rel(holdout_dir / f"{prefix}_hidden_seen_prompts.txt", workspace)
            if hidden_seen_prompt_rows
            else None,
            "hidden_holdout_prompts": base._rel(holdout_dir / f"{prefix}_hidden_holdout_prompts.txt", workspace)
            if hidden_holdout_prompt_rows
            else None,
        },
        "tokenizer": tokenizer_manifest["artifacts"],
        "source_manifest": str(generated_dir / f"{prefix}_manifest.json"),
        "source_counts": source_manifest.get("counts", {}),
    }
    base._write_json(workspace / "manifests" / f"{prefix}_workspace_manifest.json", workspace_manifest)

    return {
        "workspace": str(workspace),
        "prefix": prefix,
        "generated_dir": str(generated_dir),
        "pretrain_train_rows": len(train_rows),
        "pretrain_dev_rows": len(dev_rows),
        "pretrain_test_rows": len(test_rows),
        "midtrain_train_rows": len(midtrain_train_rows),
        "midtrain_dev_rows": len(midtrain_dev_rows),
        "midtrain_test_rows": len(midtrain_test_rows),
        "midtrain_promoted_holdout_rows": 0,
        "sft_train_rows": len(sft_train_rows),
        "sft_dev_rows": len(sft_dev_rows),
        "sft_test_rows": len(sft_test_rows),
        "holdout_prompt_rows": len(holdout_prompt_rows),
        "hidden_seen_prompt_rows": len(hidden_seen_prompt_rows),
        "hidden_holdout_prompt_rows": len(hidden_holdout_prompt_rows),
        "tokenizer_rows": len(tokenizer_corpus_rows),
        "tokenizer_json": str(tokenizer_json_dst),
        "tokenizer_bin": str(tokenizer_bin_dst),
        "workspace_manifest": str(workspace / "manifests" / f"{prefix}_workspace_manifest.json"),
    }


def main() -> int:
    base = _load_base_module()

    import argparse

    ap = argparse.ArgumentParser(description="Materialize a cache-local spec10 asset-grounded scene DSL workspace")
    ap.add_argument("--workspace", required=True, type=Path, help="Destination workspace")
    ap.add_argument("--seed-workspace", default=str(base.DEFAULT_SEED_WORKSPACE), type=Path, help="Seed spec workspace template to copy")
    ap.add_argument("--prefix", default="spec10_asset_scene_dsl", help="Artifact prefix for generated files")
    ap.add_argument("--train-repeats", type=int, default=3, help="How many times to repeat each seen combo in pretrain/train")
    ap.add_argument("--holdout-repeats", type=int, default=1, help="How many times to repeat each holdout combo before dev/test split")
    ap.add_argument("--midtrain-edit-repeat", type=int, default=2, help="How many times to repeat each semantic midtrain edit row in train")
    ap.add_argument("--midtrain-direct-repeat", type=int, default=4, help="Minimum number of direct scene rows to keep per train example in midtrain")
    ap.add_argument("--midtrain-close-repeat", type=int, default=6, help="How many times to repeat each explicit closing-tag continuation row in midtrain train")
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
        midtrain_edit_repeat=max(1, int(args.midtrain_edit_repeat)),
        midtrain_direct_repeat=max(1, int(args.midtrain_direct_repeat)),
        midtrain_close_repeat=max(1, int(args.midtrain_close_repeat)),
        midtrain_shuffle_seed=int(args.midtrain_shuffle_seed),
        python_exec=str(args.python_exec),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
