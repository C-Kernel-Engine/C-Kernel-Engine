#!/usr/bin/env python3
"""Materialize a cache-local spec06 infographic workspace."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SEED_WORKSPACE = ROOT / "version" / "v7" / "data" / "spec04"
STRUCTURED_GENERATOR = ROOT / "version" / "v7" / "scripts" / "generate_svg_structured_spec06_v7.py"
LINE_NAME = "spec06_infographics"

WORKSPACE_ENTRIES = (
    "README.md",
    "contracts",
    "manifests",
    "raw_assets",
    "normalized",
    "pretrain",
    "midtrain",
    "sft",
    "holdout",
    "tokenizer",
)

LAYOUTS = ("bullet-panel", "compare-panels", "stat-cards", "spectrum-band", "flow-steps")
ACCENTS = ("orange", "green", "blue", "purple", "gray")
BACKGROUNDS = ("paper", "mint", "slate")
TOPICS = (
    "structured_outputs",
    "platform_rollout",
    "gpu_readiness",
    "governance_path",
    "capacity_math",
    "eval_discipline",
)
SOURCE_BY_LAYOUT = {
    "bullet-panel": "memory-reality-infographic",
    "compare-panels": "performance-balance",
    "stat-cards": "activation-memory-infographic",
    "spectrum-band": "operator-spectrum-map",
    "flow-steps": "pipeline-overview",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _copy_seed_workspace(seed_workspace: Path, workspace: Path, *, force: bool) -> None:
    if workspace.exists():
        if not force:
            raise FileExistsError(f"workspace already exists: {workspace} (use --force)")
        _remove_existing(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    for name in WORKSPACE_ENTRIES:
        src = seed_workspace / name
        if not src.exists():
            continue
        dst = workspace / name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _ensure_split_dirs(workspace: Path) -> None:
    for stage in ("pretrain", "midtrain", "sft"):
        for split in ("train", "dev", "test"):
            (workspace / stage / split).mkdir(parents=True, exist_ok=True)


def _run_generator(
    workspace: Path,
    *,
    prefix: str,
    train_repeats: int,
    holdout_repeats: int,
    python_exec: str,
) -> Path:
    out_dir = workspace / "manifests" / "generated" / "structured_atoms"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exec,
        str(STRUCTURED_GENERATOR),
        "--out-dir",
        str(out_dir),
        "--prefix",
        prefix,
        "--train-repeats",
        str(train_repeats),
        "--holdout-repeats",
        str(holdout_repeats),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"structured generator failed (rc={proc.returncode}):\n{proc.stdout.strip()}")
    return out_dir


def _split_even(rows: list[str]) -> tuple[list[str], list[str]]:
    dev: list[str] = []
    test: list[str] = []
    for idx, row in enumerate(rows):
        (dev if idx % 2 == 0 else test).append(row)
    if rows and not test:
        test.append(dev.pop())
    return dev, test


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        _remove_existing(dst)
    shutil.copytree(src, dst)


def _rel(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def _row_from_catalog(prompt: str, output_tokens: str) -> str:
    return f"{prompt} {output_tokens}".strip()


def _parse_prompt_tags(prompt: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for token in prompt.split():
        if not (token.startswith("[") and token.endswith("]")):
            continue
        body = token[1:-1]
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        tags[key] = value
    return tags


def _prompt_from_tags(tags: dict[str, str], *, include_edit: str | None = None) -> str:
    layout = str(tags.get("layout") or "bullet-panel").strip()
    tokens = [
        "[task:svg]",
        f"[layout:{layout}]",
        f"[topic:{tags.get('topic', TOPICS[0])}]",
        f"[accent:{tags.get('accent', ACCENTS[0])}]",
        f"[bg:{tags.get('bg', BACKGROUNDS[0])}]",
        f"[frame:{tags.get('frame', 'plain')}]",
        f"[density:{tags.get('density', 'compact')}]",
    ]
    if include_edit:
        tokens.append(f"[edit:{include_edit}]")
    tokens.append("[OUT]")
    return " ".join(tokens)


def _topic_phrase(topic: str) -> str:
    return str(topic or "").replace("_", " ")


def _bg_phrase(bg: str) -> str:
    if bg == "paper":
        return "a paper background"
    if bg == "mint":
        return "a mint background"
    if bg == "slate":
        return "a slate background"
    return "a plain background"


def _frame_phrase(frame: str) -> str:
    return "with a card frame" if frame == "card" else "with no outer frame"


def _density_phrase(density: str) -> str:
    return "airy spacing" if density == "airy" else "compact spacing"


def _sft_instruction_variants(prompt: str) -> list[str]:
    tags = _parse_prompt_tags(prompt)
    layout = str(tags.get("layout") or "bullet-panel")
    topic = _topic_phrase(str(tags.get("topic") or TOPICS[0]))
    accent = str(tags.get("accent") or ACCENTS[0])
    bg = _bg_phrase(str(tags.get("bg") or BACKGROUNDS[0]))
    frame = _frame_phrase(str(tags.get("frame") or "plain"))
    density = _density_phrase(str(tags.get("density") or "compact"))

    if layout == "bullet-panel":
        variants = [
            f"Create a bullet panel infographic about {topic} with {accent} accents on {bg}, {frame}, using {density}. Return structured SVG atoms only. [OUT]",
            f"Make an infographic card for {topic}: headline plus bullets, {accent} accent, {bg}, {frame}, {density}. Output structured SVG atom tags only. [OUT]",
        ]
    elif layout == "compare-panels":
        variants = [
            f"Create a two panel comparison infographic about {topic} with {accent} accents on {bg}, {frame}, using {density}. Return structured SVG atoms only. [OUT]",
            f"Make a side by side comparison layout for {topic}, {accent} accent, {bg}, {frame}, {density}. Output structured SVG atom tags only. [OUT]",
        ]
    elif layout == "stat-cards":
        variants = [
            f"Create a stat card infographic about {topic} with three value cards, {accent} accents on {bg}, {frame}, using {density}. Return structured SVG atoms only. [OUT]",
            f"Make a metrics panel for {topic} using three cards, {accent} accent, {bg}, {frame}, {density}. Output structured SVG atom tags only. [OUT]",
        ]
    elif layout == "spectrum-band":
        variants = [
            f"Create a spectrum infographic about {topic} with a three segment band, {accent} accents on {bg}, {frame}, using {density}. Return structured SVG atoms only. [OUT]",
            f"Make a banded spectrum layout for {topic}, {accent} accent, {bg}, {frame}, {density}. Output structured SVG atom tags only. [OUT]",
        ]
    else:
        variants = [
            f"Create a three step flow infographic about {topic} with {accent} accents on {bg}, {frame}, using {density}. Return structured SVG atoms only. [OUT]",
            f"Make a step by step infographic for {topic}, {accent} accent, {bg}, {frame}, {density}. Output structured SVG atom tags only. [OUT]",
        ]

    deduped: list[str] = []
    seen: set[str] = set()
    for row in variants:
        if row not in seen:
            seen.add(row)
            deduped.append(row)
    return deduped


def _layout_balanced_repeats(layout: str) -> int:
    return {
        "bullet-panel": 3,
        "compare-panels": 3,
        "stat-cards": 3,
        "spectrum-band": 3,
        "flow-steps": 3,
    }.get(layout, 2)


def _is_targeted_bullet_slice(tags: dict[str, str]) -> bool:
    return (
        str(tags.get("layout") or "") == "bullet-panel"
        and str(tags.get("topic") or "") == "governance_path"
        and str(tags.get("bg") or "") == "slate"
        and str(tags.get("frame") or "") == "card"
        and str(tags.get("accent") or "") in {"orange", "green"}
    )


def _is_targeted_bullet_anchor(tags: dict[str, str]) -> bool:
    return (
        str(tags.get("layout") or "") == "bullet-panel"
        and str(tags.get("topic") or "") == "governance_path"
        and str(tags.get("bg") or "") == "slate"
        and str(tags.get("frame") or "") == "card"
        and str(tags.get("accent") or "") == "blue"
    )


def _is_targeted_bullet_layout_neighbor(tags: dict[str, str]) -> bool:
    return (
        str(tags.get("layout") or "") in {"compare-panels", "spectrum-band"}
        and str(tags.get("topic") or "") == "governance_path"
        and str(tags.get("bg") or "") == "slate"
        and str(tags.get("frame") or "") == "card"
        and str(tags.get("accent") or "") in {"orange", "green"}
    )


def _is_targeted_bullet_midtrain_promotion(tags: dict[str, str]) -> bool:
    return _is_targeted_bullet_slice(tags) or _is_targeted_bullet_anchor(tags)


def _is_targeted_spectrum_neighbor(tags: dict[str, str]) -> bool:
    layout = str(tags.get("layout") or "")
    topic = str(tags.get("topic") or "")
    accent = str(tags.get("accent") or "")
    bg = str(tags.get("bg") or "")
    density = str(tags.get("density") or "")
    if topic not in {"platform_rollout", "structured_outputs"} or bg != "slate":
        return False
    if layout == "spectrum-band":
        return (accent == "orange" and density == "compact") or (accent != "orange" and density == "airy")
    if layout in {"compare-panels", "flow-steps"}:
        return accent == "orange" and density == "airy"
    return False


def _direct_repeat(tags: dict[str, str], *, is_train: bool) -> int:
    layout = str(tags.get("layout") or "bullet-panel")
    base = _layout_balanced_repeats(layout) if is_train else 1
    if not is_train:
        return base
    if _is_targeted_bullet_slice(tags):
        return max(base, 5)
    if _is_targeted_bullet_anchor(tags):
        return max(base, 4)
    if _is_targeted_bullet_layout_neighbor(tags):
        return max(base, 4)
    if _is_targeted_spectrum_neighbor(tags):
        return max(base, 5)
    return base


def _flip_bg(bg: str) -> str | None:
    for value in BACKGROUNDS:
        if value != bg:
            return value
    return None


def _flip_density(density: str) -> str | None:
    return {"compact": "airy", "airy": "compact"}.get(str(density or "").strip())


def _flip_frame(frame: str) -> str | None:
    return {"plain": "card", "card": "plain"}.get(str(frame or "").strip())


def _alt_value(current: str, candidates: tuple[str, ...], *, limit: int | None = None) -> list[str]:
    values = [value for value in candidates if value != current]
    if limit is None:
        return values
    return values[: max(0, int(limit))]


def _topic_edit_targets(current: str, *, layout: str, limit: int) -> list[str]:
    topics = list(TOPICS)
    if current not in topics:
        return topics[: max(0, int(limit))]
    preferred = {
        "structured_outputs": ("eval_discipline", "platform_rollout", "capacity_math"),
        "platform_rollout": ("gpu_readiness", "structured_outputs", "governance_path"),
        "gpu_readiness": ("platform_rollout", "capacity_math", "governance_path"),
        "governance_path": ("eval_discipline", "capacity_math", "gpu_readiness"),
        "capacity_math": ("governance_path", "eval_discipline", "gpu_readiness"),
        "eval_discipline": ("governance_path", "capacity_math", "structured_outputs"),
    }.get(current, ())
    idx = topics.index(current)
    ordered: list[str] = []
    if layout == "bullet-panel":
        for candidate in preferred:
            if candidate != current and candidate not in ordered:
                ordered.append(candidate)
    for offset in (1, -1, 2, -2, 3):
        candidate = topics[(idx + offset) % len(topics)]
        if candidate != current and candidate not in ordered:
            ordered.append(candidate)
    return ordered[: max(0, int(limit))]


def _topic_target_count(layout: str, tags: dict[str, str]) -> int:
    if layout == "bullet-panel" and (_is_targeted_bullet_slice(tags) or _is_targeted_bullet_anchor(tags)):
        return 1
    return 3 if layout == "bullet-panel" else 2


def _related_layouts(layout: str) -> tuple[str, ...]:
    return {
        "bullet-panel": ("compare-panels", "stat-cards"),
        "compare-panels": ("spectrum-band", "bullet-panel"),
        "stat-cards": ("compare-panels", "spectrum-band"),
        "spectrum-band": ("compare-panels", "flow-steps"),
        "flow-steps": ("spectrum-band", "compare-panels"),
    }.get(layout, ())


def _candidate_midtrain_edits(tags: dict[str, str]) -> list[tuple[str, dict[str, str]]]:
    layout = str(tags.get("layout") or "bullet-panel")
    edits: list[tuple[str, dict[str, str]]] = []

    bg_alt = _flip_bg(str(tags.get("bg") or ""))
    if bg_alt:
        target = dict(tags)
        target["bg"] = bg_alt
        edits.append((f"bg={bg_alt}", target))

    density_alt = _flip_density(str(tags.get("density") or ""))
    if density_alt:
        target = dict(tags)
        target["density"] = density_alt
        edits.append((f"density={density_alt}", target))

    frame_alt = _flip_frame(str(tags.get("frame") or ""))
    if frame_alt:
        target = dict(tags)
        target["frame"] = frame_alt
        edits.append((f"frame={frame_alt}", target))

    for accent_alt in _alt_value(str(tags.get("accent") or ""), ACCENTS, limit=1):
        target = dict(tags)
        target["accent"] = accent_alt
        edits.append((f"accent={accent_alt}", target))

    for topic_alt in _topic_edit_targets(
        str(tags.get("topic") or ""),
        layout=layout,
        limit=_topic_target_count(layout, tags),
    ):
        target = dict(tags)
        target["topic"] = topic_alt
        edits.append((f"topic={topic_alt}", target))

    contrast_layouts: list[str] = []
    if layout == "bullet-panel" and (_is_targeted_bullet_slice(tags) or _is_targeted_bullet_anchor(tags)):
        contrast_layouts.extend(["compare-panels", "spectrum-band"])
    for layout_alt in _related_layouts(layout):
        if layout_alt not in contrast_layouts:
            contrast_layouts.append(layout_alt)

    if layout == "bullet-panel" and (_is_targeted_bullet_slice(tags) or _is_targeted_bullet_anchor(tags)):
        layout_limit = 2
    else:
        layout_limit = 2 if layout in {"compare-panels", "stat-cards", "spectrum-band"} else 1
    for layout_alt in contrast_layouts[:layout_limit]:
        target = dict(tags)
        target["layout"] = layout_alt
        edits.append((f"layout={layout_alt}", target))

    return edits


def _midtrain_edit_priority(layout: str, tags: dict[str, str]) -> tuple[str, ...]:
    if layout == "bullet-panel":
        if _is_targeted_bullet_slice(tags) or _is_targeted_bullet_anchor(tags):
            return ("layout", "density", "frame", "accent", "bg", "topic")
        return ("topic", "layout", "density", "frame", "accent", "bg")
    if layout == "compare-panels":
        return ("layout", "topic", "accent", "density", "frame", "bg")
    if layout == "stat-cards":
        return ("layout", "topic", "accent", "frame", "density", "bg")
    if layout == "spectrum-band":
        return ("layout", "topic", "accent", "density", "frame", "bg")
    if layout == "flow-steps":
        return ("topic", "layout", "density", "frame", "accent", "bg")
    return ()


def _midtrain_max_edits(layout: str) -> int:
    return {
        "bullet-panel": 6,
        "compare-panels": 6,
        "stat-cards": 6,
        "spectrum-band": 6,
        "flow-steps": 6,
    }.get(layout, 3)


def _select_midtrain_edits(
    layout: str,
    tags: dict[str, str],
    edits: list[tuple[str, dict[str, str]]],
) -> list[tuple[str, dict[str, str]]]:
    priority = _midtrain_edit_priority(layout, tags)
    rank = {key: idx for idx, key in enumerate(priority)}
    ordered = sorted(edits, key=lambda item: rank.get(item[0].split("=", 1)[0], len(rank)))
    return ordered[: max(1, _midtrain_max_edits(layout))]


def _boosted_edit_repeat(tags: dict[str, str], edit_op: str, base_repeat: int) -> int:
    edit_kind = str(edit_op or "").split("=", 1)[0]
    layout = str(tags.get("layout") or "")
    repeat = max(1, int(base_repeat))
    if edit_kind == "topic":
        repeat = max(repeat, 3)
        if layout == "bullet-panel":
            repeat = max(repeat, 4)
    if edit_kind == "layout":
        repeat = max(repeat, 2 if layout in {"compare-panels", "stat-cards", "spectrum-band"} else 1)
    if _is_targeted_bullet_slice(tags):
        if edit_kind == "layout":
            repeat = max(repeat, 4)
        if edit_kind == "density":
            repeat = max(repeat, 3)
        if edit_kind == "frame":
            repeat = max(repeat, 3)
        if edit_kind == "accent":
            repeat = max(repeat, 2)
        if edit_kind == "topic":
            repeat = max(repeat, 1)
    if _is_targeted_bullet_anchor(tags):
        if edit_kind == "layout":
            repeat = max(repeat, 3)
        if edit_kind == "density":
            repeat = max(repeat, 2)
    if _is_targeted_spectrum_neighbor(tags):
        if edit_kind == "layout":
            repeat = max(repeat, 5)
        if edit_kind == "topic":
            repeat = max(repeat, 4)
    return repeat


def _split_catalog_even_by_layout(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_layout: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        layout = str(row.get("tags", {}).get("layout") or "unknown")
        by_layout.setdefault(layout, []).append(row)

    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for layout in sorted(by_layout):
        group = sorted(by_layout[layout], key=lambda item: str(item.get("prompt") or ""))
        for idx, row in enumerate(group):
            (dev if idx % 2 == 0 else test).append(row)
        if group and not any(str(item.get("tags", {}).get("layout") or "") == layout for item in test):
            test.append(dev.pop())
    return dev, test


def _build_midtrain_rows(
    catalog_rows: list[dict[str, Any]],
    *,
    is_train: bool,
    edit_repeat: int,
    shuffle_seed: int,
) -> list[str]:
    prompt_index = {str(row.get("prompt") or ""): row for row in catalog_rows}
    out_rows: list[str] = []
    for row in catalog_rows:
        prompt = str(row.get("prompt") or "")
        output_tokens = str(row.get("output_tokens") or "")
        tags = dict(row.get("tags") or {})
        layout = str(tags.get("layout") or "bullet-panel")
        base_row = _row_from_catalog(prompt, output_tokens)
        base_repeat = _direct_repeat(tags, is_train=is_train)
        for _ in range(base_repeat):
            out_rows.append(base_row)

        edits = _select_midtrain_edits(layout, tags, _candidate_midtrain_edits(tags))
        for edit_op, target_tags in edits:
            target_prompt = _prompt_from_tags(target_tags)
            target = prompt_index.get(target_prompt)
            if target is None:
                continue
            edit_prompt = _prompt_from_tags(tags, include_edit=edit_op)
            edit_row = _row_from_catalog(edit_prompt, str(target.get("output_tokens") or ""))
            repeat = _boosted_edit_repeat(tags, edit_op, int(edit_repeat)) if is_train else 1
            for _ in range(repeat):
                out_rows.append(edit_row)

    rng = random.Random(int(shuffle_seed))
    rng.shuffle(out_rows)
    return out_rows


def _summarize_midtrain_rows(rows: list[str]) -> dict[str, Any]:
    layout_counts: Counter[str] = Counter()
    edit_counts: Counter[str] = Counter()
    edit_rows = 0
    for row in rows:
        tags = _parse_prompt_tags(row)
        layout_counts[str(tags.get("layout") or "unknown")] += 1
        edit_tag = str(tags.get("edit") or "").strip()
        if edit_tag:
            edit_rows += 1
            edit_counts[edit_tag.split("=", 1)[0]] += 1
    return {
        "total_rows": len(rows),
        "direct_rows": len(rows) - edit_rows,
        "edit_rows": edit_rows,
        "layout_counts": dict(sorted(layout_counts.items())),
        "edit_counts": dict(sorted(edit_counts.items())),
    }


def _rewrite_probe_contract(
    src: Path,
    dst: Path,
    *,
    catalog_path: str,
    train_path: str,
    test_path: str,
) -> None:
    payload = copy.deepcopy(_load_json(src))
    catalog = payload.get("catalog")
    if isinstance(catalog, dict):
        catalog["path"] = catalog_path
    splits = payload.get("splits")
    if isinstance(splits, list):
        for split in splits:
            if not isinstance(split, dict):
                continue
            name = str(split.get("name") or "").strip()
            if name == "train":
                split["path"] = train_path
            elif name == "test":
                split["path"] = test_path
    _write_json(dst, payload)


def materialize_workspace(
    workspace: Path,
    *,
    seed_workspace: Path,
    prefix: str,
    train_repeats: int,
    holdout_repeats: int,
    midtrain_edit_repeat: int,
    midtrain_shuffle_seed: int,
    python_exec: str,
    force: bool,
) -> dict[str, Any]:
    workspace = workspace.expanduser().resolve()
    seed_workspace = seed_workspace.expanduser().resolve()
    if not seed_workspace.exists():
        raise FileNotFoundError(f"seed workspace not found: {seed_workspace}")

    _copy_seed_workspace(seed_workspace, workspace, force=force)
    _ensure_split_dirs(workspace)

    generated_dir = _run_generator(
        workspace,
        prefix=prefix,
        train_repeats=train_repeats,
        holdout_repeats=holdout_repeats,
        python_exec=python_exec,
    )
    source_manifest = _load_json(generated_dir / f"{prefix}_manifest.json")
    render_catalog_rows = json.loads((generated_dir / f"{prefix}_render_catalog.json").read_text(encoding="utf-8"))

    train_rows = (generated_dir / f"{prefix}_train.txt").read_text(encoding="utf-8").splitlines()
    holdout_rows = (generated_dir / f"{prefix}_holdout.txt").read_text(encoding="utf-8").splitlines()
    tokenizer_corpus_rows = (generated_dir / f"{prefix}_tokenizer_corpus.txt").read_text(encoding="utf-8").splitlines()
    seen_prompt_rows = (generated_dir / f"{prefix}_seen_prompts.txt").read_text(encoding="utf-8").splitlines()
    holdout_prompt_rows = (generated_dir / f"{prefix}_holdout_prompts.txt").read_text(encoding="utf-8").splitlines()
    reserved_token_rows = (generated_dir / f"{prefix}_reserved_control_tokens.txt").read_text(encoding="utf-8").splitlines()

    dev_rows, test_rows = _split_even(holdout_rows)
    holdout_prompt_dev, holdout_prompt_test = _split_even(holdout_prompt_rows)

    pretrain_train = workspace / "pretrain" / "train" / f"{prefix}_pretrain_train.txt"
    pretrain_dev = workspace / "pretrain" / "dev" / f"{prefix}_pretrain_dev.txt"
    pretrain_test = workspace / "pretrain" / "test" / f"{prefix}_pretrain_test.txt"
    _write_lines(pretrain_train, train_rows)
    _write_lines(pretrain_dev, dev_rows)
    _write_lines(pretrain_test, test_rows)

    sft_train_rows: list[str] = []
    sft_holdout_rows: list[str] = []
    midtrain_train_catalog: list[dict[str, Any]] = []
    midtrain_holdout_catalog: list[dict[str, Any]] = []
    promoted_midtrain_holdout_rows = 0
    for row in render_catalog_rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        output_tokens = str(row.get("output_tokens") or "").strip()
        split = str(row.get("split") or "").strip()
        tags = _parse_prompt_tags(prompt)
        layout = tags.get("layout", "bullet-panel")
        if not prompt or not output_tokens or split not in {"train", "holdout"}:
            continue
        stage_row = _row_from_catalog(prompt, output_tokens)
        midtrain_row = {
            "prompt": prompt,
            "output_tokens": output_tokens,
            "split": split,
            "tags": dict(tags),
        }
        if split == "train":
            midtrain_train_catalog.append(midtrain_row)
        else:
            midtrain_holdout_catalog.append(midtrain_row)
            if _is_targeted_bullet_midtrain_promotion(tags):
                promoted_midtrain_holdout_rows += 1
                midtrain_train_catalog.append(dict(midtrain_row))
        instruction_rows = [_row_from_catalog(instruction, output_tokens) for instruction in _sft_instruction_variants(prompt)]
        if split == "train":
            sft_train_rows.extend([stage_row] * _layout_balanced_repeats(layout))
            sft_train_rows.extend(instruction_rows)
        else:
            sft_holdout_rows.append(stage_row)
            sft_holdout_rows.extend(instruction_rows[:1])

    midtrain_holdout_dev_catalog, midtrain_holdout_test_catalog = _split_catalog_even_by_layout(midtrain_holdout_catalog)
    midtrain_train_rows = _build_midtrain_rows(
        midtrain_train_catalog,
        is_train=True,
        edit_repeat=int(midtrain_edit_repeat),
        shuffle_seed=int(midtrain_shuffle_seed),
    )
    midtrain_dev_rows = _build_midtrain_rows(
        midtrain_holdout_dev_catalog,
        is_train=False,
        edit_repeat=1,
        shuffle_seed=int(midtrain_shuffle_seed) + 1,
    )
    midtrain_test_rows = _build_midtrain_rows(
        midtrain_holdout_test_catalog,
        is_train=False,
        edit_repeat=1,
        shuffle_seed=int(midtrain_shuffle_seed) + 2,
    )
    sft_dev_rows, sft_test_rows = _split_even(sft_holdout_rows)

    midtrain_train = workspace / "midtrain" / "train" / f"{prefix}_midtrain_train.txt"
    midtrain_dev = workspace / "midtrain" / "dev" / f"{prefix}_midtrain_dev.txt"
    midtrain_test = workspace / "midtrain" / "test" / f"{prefix}_midtrain_test.txt"
    _write_lines(midtrain_train, midtrain_train_rows)
    _write_lines(midtrain_dev, midtrain_dev_rows)
    _write_lines(midtrain_test, midtrain_test_rows)

    sft_train = workspace / "sft" / "train" / f"{prefix}_sft_train.txt"
    sft_dev = workspace / "sft" / "dev" / f"{prefix}_sft_dev.txt"
    sft_test = workspace / "sft" / "test" / f"{prefix}_sft_test.txt"
    _write_lines(sft_train, sft_train_rows)
    _write_lines(sft_dev, sft_dev_rows)
    _write_lines(sft_test, sft_test_rows)

    holdout_dir = workspace / "holdout"
    _write_lines(holdout_dir / f"{prefix}_holdout_prompts.txt", holdout_prompt_rows)
    _write_lines(holdout_dir / f"{prefix}_holdout_prompts_dev.txt", holdout_prompt_dev)
    _write_lines(holdout_dir / f"{prefix}_holdout_prompts_test.txt", holdout_prompt_test)
    _write_lines(holdout_dir / f"{prefix}_seen_prompts.txt", seen_prompt_rows)

    tokenizer_dir = workspace / "tokenizer"
    tokenizer_json_src = generated_dir / f"{prefix}_tokenizer" / "tokenizer.json"
    tokenizer_bin_src = generated_dir / f"{prefix}_tokenizer" / "tokenizer_bin"
    probe_contract_src = generated_dir / f"{prefix}_probe_report_contract.json"
    eval_contract_src = generated_dir / f"{prefix}_eval_contract.json"
    tokenizer_json_dst = tokenizer_dir / "tokenizer.json"
    tokenizer_bin_dst = tokenizer_dir / "tokenizer_bin"
    shutil.copy2(tokenizer_json_src, tokenizer_json_dst)
    _copy_tree(tokenizer_bin_src, tokenizer_bin_dst)
    shutil.copy2(generated_dir / f"{prefix}_vocab.json", tokenizer_dir / f"{prefix}_vocab.json")
    shutil.copy2(generated_dir / f"{prefix}_render_catalog.json", tokenizer_dir / f"{prefix}_render_catalog.json")
    if probe_contract_src.exists():
        _rewrite_probe_contract(
            probe_contract_src,
            tokenizer_dir / f"{prefix}_probe_report_contract.json",
            catalog_path=f"{prefix}_render_catalog.json",
            train_path=f"{prefix}_seen_prompts.txt",
            test_path=f"{prefix}_holdout_prompts.txt",
        )
    _write_lines(tokenizer_dir / f"{prefix}_tokenizer_corpus.txt", tokenizer_corpus_rows)
    _write_lines(tokenizer_dir / f"{prefix}_reserved_control_tokens.txt", reserved_token_rows)
    _write_lines(tokenizer_dir / f"{prefix}_seen_prompts.txt", seen_prompt_rows)
    _write_lines(tokenizer_dir / f"{prefix}_holdout_prompts.txt", holdout_prompt_rows)

    contracts_dir = workspace / "contracts"
    if probe_contract_src.exists():
        _rewrite_probe_contract(
            probe_contract_src,
            contracts_dir / f"{prefix}_probe_report_contract.json",
            catalog_path=f"../manifests/generated/structured_atoms/{prefix}_render_catalog.json",
            train_path=f"../holdout/{prefix}_seen_prompts.txt",
            test_path=f"../holdout/{prefix}_holdout_prompts.txt",
        )
    if eval_contract_src.exists():
        shutil.copy2(eval_contract_src, contracts_dir / "eval_contract.svg.v1.json")

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
            "tokenizer_corpus": _rel(tokenizer_dir / f"{prefix}_tokenizer_corpus.txt", workspace),
            "reserved_control_tokens": _rel(tokenizer_dir / f"{prefix}_reserved_control_tokens.txt", workspace),
            "seen_prompts": _rel(tokenizer_dir / f"{prefix}_seen_prompts.txt", workspace),
            "holdout_prompts": _rel(tokenizer_dir / f"{prefix}_holdout_prompts.txt", workspace),
            "tokenizer_json": _rel(tokenizer_json_dst, workspace),
            "tokenizer_bin": _rel(tokenizer_bin_dst, workspace),
            "vocab": _rel(tokenizer_dir / f"{prefix}_vocab.json", workspace),
            "render_catalog": _rel(tokenizer_dir / f"{prefix}_render_catalog.json", workspace),
            "probe_report_contract": _rel(tokenizer_dir / f"{prefix}_probe_report_contract.json", workspace)
            if probe_contract_src.exists()
            else None,
            "eval_contract": _rel(contracts_dir / "eval_contract.svg.v1.json", workspace)
            if eval_contract_src.exists()
            else None,
        },
    }
    _write_json(tokenizer_dir / f"{prefix}_tokenizer_manifest.json", tokenizer_manifest)

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
        "midtrain_shuffle_seed": int(midtrain_shuffle_seed),
        "stages": {
            "pretrain": {
                "train": _rel(pretrain_train, workspace),
                "dev": _rel(pretrain_dev, workspace),
                "test": _rel(pretrain_test, workspace),
                "counts": {
                    "train_rows": len(train_rows),
                    "dev_rows": len(dev_rows),
                    "test_rows": len(test_rows),
                },
            },
            "midtrain": {
                "train": _rel(midtrain_train, workspace),
                "dev": _rel(midtrain_dev, workspace),
                "test": _rel(midtrain_test, workspace),
                "notes": "Blended infographic template coverage plus interleaved topic and layout edit rows across asset-inspired layout families.",
                "counts": {
                    "train_rows": len(midtrain_train_rows),
                    "dev_rows": len(midtrain_dev_rows),
                    "test_rows": len(midtrain_test_rows),
                    "promoted_holdout_rows": promoted_midtrain_holdout_rows,
                    "train_summary": _summarize_midtrain_rows(midtrain_train_rows),
                    "dev_summary": _summarize_midtrain_rows(midtrain_dev_rows),
                    "test_summary": _summarize_midtrain_rows(midtrain_test_rows),
                },
            },
            "sft": {
                "train": _rel(sft_train, workspace),
                "dev": _rel(sft_dev, workspace),
                "test": _rel(sft_test, workspace),
                "notes": "Mixed DSL and natural-language infographic prompts for the same deterministic templates.",
                "counts": {
                    "train_rows": len(sft_train_rows),
                    "dev_rows": len(sft_dev_rows),
                    "test_rows": len(sft_test_rows),
                },
            },
        },
        "holdout": {
            "prompts_all": _rel(holdout_dir / f"{prefix}_holdout_prompts.txt", workspace),
            "prompts_dev": _rel(holdout_dir / f"{prefix}_holdout_prompts_dev.txt", workspace),
            "prompts_test": _rel(holdout_dir / f"{prefix}_holdout_prompts_test.txt", workspace),
            "seen_prompts": _rel(holdout_dir / f"{prefix}_seen_prompts.txt", workspace),
        },
        "tokenizer": tokenizer_manifest["artifacts"],
        "source_manifest": str(generated_dir / f"{prefix}_manifest.json"),
        "source_counts": source_manifest.get("counts", {}),
    }
    _write_json(workspace / "manifests" / f"{prefix}_workspace_manifest.json", workspace_manifest)

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
        "midtrain_promoted_holdout_rows": promoted_midtrain_holdout_rows,
        "sft_train_rows": len(sft_train_rows),
        "sft_dev_rows": len(sft_dev_rows),
        "sft_test_rows": len(sft_test_rows),
        "holdout_prompt_rows": len(holdout_prompt_rows),
        "tokenizer_rows": len(tokenizer_corpus_rows),
        "tokenizer_json": str(tokenizer_json_dst),
        "tokenizer_bin": str(tokenizer_bin_dst),
        "workspace_manifest": str(workspace / "manifests" / f"{prefix}_workspace_manifest.json"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize a cache-local spec06 infographic workspace")
    ap.add_argument("--workspace", required=True, type=Path, help="Destination workspace (recommended inside a cache run dir)")
    ap.add_argument("--seed-workspace", default=str(DEFAULT_SEED_WORKSPACE), type=Path, help="Seed spec workspace template to copy")
    ap.add_argument("--prefix", default="spec06_structured_infographics", help="Artifact prefix for generated files")
    ap.add_argument("--train-repeats", type=int, default=3, help="How many times to repeat each seen combo in pretrain/train")
    ap.add_argument("--holdout-repeats", type=int, default=1, help="How many times to repeat each holdout combo before dev/test split")
    ap.add_argument("--midtrain-edit-repeat", type=int, default=1, help="How many times to repeat each midtrain edit row in train")
    ap.add_argument("--midtrain-shuffle-seed", type=int, default=42, help="Shuffle seed for interleaving the blended midtrain curriculum")
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
        midtrain_shuffle_seed=int(args.midtrain_shuffle_seed),
        python_exec=str(args.python_exec),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
