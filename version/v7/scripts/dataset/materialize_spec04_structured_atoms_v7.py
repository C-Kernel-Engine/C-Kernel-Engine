#!/usr/bin/env python3
"""Materialize a cache-local spec04 workspace from the structured SVG toy line."""

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
STRUCTURED_TOY_GENERATOR = ROOT / "version" / "v7" / "scripts" / "generate_svg_structured_toy_v7.py"

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


def _run_structured_toy_generator(
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
        str(STRUCTURED_TOY_GENERATOR),
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
        raise RuntimeError(
            f"structured toy generator failed (rc={proc.returncode}):\n{proc.stdout.strip()}"
        )
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
    layout = str(tags.get("layout") or "single").strip()
    tokens = ["[task:svg]", f"[layout:{layout}]"]
    if layout in {"single", "pair-h", "pair-v", "badge"}:
        tokens.append(f"[shape:{tags.get('shape', 'rect')}]")
    if layout in {"pair-h", "pair-v"}:
        tokens.append(f"[shape2:{tags.get('shape2', 'rect')}]")
    if layout in {"single", "pair-h", "pair-v", "label-card", "badge"}:
        tokens.append(f"[color:{tags.get('color', 'blue')}]")
    if layout in {"pair-h", "pair-v"}:
        tokens.append(f"[color2:{tags.get('color2', 'red')}]")
    if layout in {"single", "pair-h", "pair-v"}:
        tokens.append(f"[size:{tags.get('size', 'small')}]")
    tokens.append(f"[bg:{tags.get('bg', 'none')}]")
    if layout in {"label-card", "badge"}:
        tokens.append(f"[label:{tags.get('label', 'ok')}]")
    if include_edit:
        tokens.append(f"[edit:{include_edit}]")
    tokens.append("[OUT]")
    return " ".join(tokens)


def _bg_phrase(bg: str) -> str:
    if bg == "paper":
        return "a paper background"
    if bg == "mint":
        return "a mint background"
    if bg == "slate":
        return "a slate background"
    return "no background fill"


def _shape_phrase(shape: str) -> str:
    return {"rect": "rectangle"}.get(shape, shape)


def _sft_instruction_variants(prompt: str) -> list[str]:
    tags = _parse_prompt_tags(prompt)
    layout = tags.get("layout", "single")
    shape = _shape_phrase(tags.get("shape", "rect"))
    color = tags.get("color", "blue")
    bg = _bg_phrase(tags.get("bg", "none"))
    size = tags.get("size", "small")
    label = tags.get("label", "").upper()
    shape2 = _shape_phrase(tags.get("shape2", "rect"))
    color2 = tags.get("color2", "red")

    variants: list[str]
    if layout == "single":
        variants = [
            f"Draw a {size} {color} {shape} centered on {bg}. Return structured SVG atoms only. [OUT]",
            f"Create one {color} {shape} on {bg}. Keep it {size} and output only structured SVG atom tags. [OUT]",
        ]
    elif layout == "pair-h":
        variants = [
            f"Draw two {size} shapes side by side on {bg}: a {color} {shape} on the left and a {color2} {shape2} on the right. Return structured SVG atoms only. [OUT]",
            f"Create a horizontal composition on {bg} with a left {color} {shape} and a right {color2} {shape2}. Output structured SVG atom tags only. [OUT]",
        ]
    elif layout == "pair-v":
        variants = [
            f"Draw two {size} shapes stacked vertically on {bg}: a {color} {shape} on top and a {color2} {shape2} below. Return structured SVG atoms only. [OUT]",
            f"Create a vertical composition on {bg} with a top {color} {shape} and a bottom {color2} {shape2}. Output structured SVG atom tags only. [OUT]",
        ]
    elif layout == "label-card":
        variants = [
            f"Create a rounded {color} label card on {bg} with the text {label} centered. Return structured SVG atoms only. [OUT]",
            f"Make a text card on {bg}: a {color} rectangle with {label} in the middle. Output structured SVG atom tags only. [OUT]",
        ]
    elif layout == "badge":
        variants = [
            f"Create a badge card on {bg} with a {color} {shape} icon and the text {label}. Return structured SVG atoms only. [OUT]",
            f"Draw a badge on {bg}: a card with a {color} {shape} icon on the left and {label} on the right. Output structured SVG atom tags only. [OUT]",
        ]
    else:
        variants = [f"{prompt}"]
    # Preserve order but drop duplicates if two templates collapse to the same string.
    deduped: list[str] = []
    seen: set[str] = set()
    for row in variants:
        if row not in seen:
            seen.add(row)
            deduped.append(row)
    return deduped


def _layout_balanced_repeats(layout: str) -> int:
    return {
        "single": 8,
        "pair-h": 1,
        "pair-v": 1,
        "label-card": 8,
        "badge": 3,
    }.get(layout, 1)


def _flip_bg(bg: str) -> str | None:
    return {
        "none": "paper",
        "paper": "mint",
        "mint": "paper",
        "slate": "paper",
    }.get(str(bg or "").strip())


def _flip_size(size: str) -> str | None:
    return {
        "small": "big",
        "big": "small",
    }.get(str(size or "").strip())


def _alt_value(current: str, candidates: tuple[str, ...]) -> str | None:
    for value in candidates:
        if value != current:
            return value
    return None


def _candidate_midtrain_edits(tags: dict[str, str]) -> list[tuple[str, dict[str, str]]]:
    layout = str(tags.get("layout") or "").strip()
    edits: list[tuple[str, dict[str, str]]] = []

    bg_alt = _flip_bg(str(tags.get("bg") or ""))
    if bg_alt:
        target = dict(tags)
        target["bg"] = bg_alt
        edits.append((f"bg={bg_alt}", target))

    if layout == "single":
        shape_alt = _alt_value(str(tags.get("shape") or ""), ("circle", "rect", "triangle"))
        if shape_alt:
            target = dict(tags)
            target["shape"] = shape_alt
            edits.append((f"shape={shape_alt}", target))
        color_alt = _alt_value(str(tags.get("color") or ""), ("blue", "green", "orange", "purple", "red"))
        if color_alt:
            target = dict(tags)
            target["color"] = color_alt
            edits.append((f"color={color_alt}", target))
        size_alt = _flip_size(str(tags.get("size") or ""))
        if size_alt:
            target = dict(tags)
            target["size"] = size_alt
            edits.append((f"size={size_alt}", target))
    elif layout == "pair-h":
        target = dict(tags)
        target["layout"] = "pair-v"
        edits.append(("layout=pair-v", target))
    elif layout == "pair-v":
        target = dict(tags)
        target["layout"] = "pair-h"
        edits.append(("layout=pair-h", target))

    if layout in {"pair-h", "pair-v"}:
        shape2_alt = _alt_value(str(tags.get("shape2") or ""), ("circle", "rect", "triangle"))
        if shape2_alt and shape2_alt != tags.get("shape"):
            target = dict(tags)
            target["shape2"] = shape2_alt
            edits.append((f"shape2={shape2_alt}", target))
        color2_alt = _alt_value(str(tags.get("color2") or ""), ("blue", "green", "orange", "purple", "red"))
        if color2_alt and color2_alt != tags.get("color"):
            target = dict(tags)
            target["color2"] = color2_alt
            edits.append((f"color2={color2_alt}", target))
        size_alt = _flip_size(str(tags.get("size") or ""))
        if size_alt:
            target = dict(tags)
            target["size"] = size_alt
            edits.append((f"size={size_alt}", target))
    elif layout == "label-card":
        label_alt = _alt_value(str(tags.get("label") or ""), ("ai", "data", "flow", "go", "note", "ok"))
        if label_alt:
            target = dict(tags)
            target["label"] = label_alt
            edits.append((f"label={label_alt}", target))
        color_alt = _alt_value(str(tags.get("color") or ""), ("blue", "green", "orange", "purple", "red"))
        if color_alt:
            target = dict(tags)
            target["color"] = color_alt
            edits.append((f"color={color_alt}", target))
    elif layout == "badge":
        label_alt = _alt_value(str(tags.get("label") or ""), ("ai", "data", "flow", "go", "note", "ok"))
        if label_alt:
            target = dict(tags)
            target["label"] = label_alt
            edits.append((f"label={label_alt}", target))
        shape_alt = _alt_value(str(tags.get("shape") or ""), ("circle", "rect", "triangle"))
        if shape_alt:
            target = dict(tags)
            target["shape"] = shape_alt
            edits.append((f"shape={shape_alt}", target))
        color_alt = _alt_value(str(tags.get("color") or ""), ("blue", "green", "orange", "purple", "red"))
        if color_alt:
            target = dict(tags)
            target["color"] = color_alt
            edits.append((f"color={color_alt}", target))
    return edits


def _midtrain_edit_priority(layout: str) -> tuple[str, ...]:
    if layout == "single":
        return ("bg", "size", "color", "shape")
    if layout in {"pair-h", "pair-v"}:
        return ("layout", "bg", "color2", "shape2", "size")
    if layout == "label-card":
        return ("label", "bg", "color")
    if layout == "badge":
        return ("label", "bg", "shape", "color")
    return ()


def _midtrain_max_edits(layout: str) -> int:
    return {
        "single": 2,
        "pair-h": 3,
        "pair-v": 3,
        "label-card": 2,
        "badge": 2,
    }.get(layout, 2)


def _select_midtrain_edits(layout: str, edits: list[tuple[str, dict[str, str]]]) -> list[tuple[str, dict[str, str]]]:
    priority = _midtrain_edit_priority(layout)
    rank = {key: idx for idx, key in enumerate(priority)}
    ordered = sorted(edits, key=lambda item: rank.get(item[0].split("=", 1)[0], len(rank)))
    return ordered[: max(1, _midtrain_max_edits(layout))]


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
            moved = dev.pop()
            test.append(moved)
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
        layout = str(tags.get("layout") or "single")
        base_row = _row_from_catalog(prompt, output_tokens)
        base_repeat = _layout_balanced_repeats(layout) if is_train else 1
        for _ in range(base_repeat):
            out_rows.append(base_row)

        edits = _select_midtrain_edits(layout, _candidate_midtrain_edits(tags))
        for edit_op, target_tags in edits:
            target_prompt = _prompt_from_tags(target_tags)
            target = prompt_index.get(target_prompt)
            if target is None:
                continue
            edit_prompt = _prompt_from_tags(tags, include_edit=edit_op)
            edit_row = _row_from_catalog(edit_prompt, str(target.get("output_tokens") or ""))
            repeat = max(1, int(edit_repeat)) if is_train else 1
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

    generated_dir = _run_structured_toy_generator(
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
    reserved_token_rows = (
        generated_dir / f"{prefix}_reserved_control_tokens.txt"
    ).read_text(encoding="utf-8").splitlines()

    dev_rows, test_rows = _split_even(holdout_rows)
    holdout_prompt_dev, holdout_prompt_test = _split_even(holdout_prompt_rows)

    pretrain_train = workspace / "pretrain" / "train" / f"{prefix}_pretrain_train.txt"
    pretrain_dev = workspace / "pretrain" / "dev" / f"{prefix}_pretrain_dev.txt"
    pretrain_test = workspace / "pretrain" / "test" / f"{prefix}_pretrain_test.txt"
    _write_lines(pretrain_train, train_rows)
    _write_lines(pretrain_dev, dev_rows)
    _write_lines(pretrain_test, test_rows)

    complex_train_rows: list[str] = []
    complex_holdout_rows: list[str] = []
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
        layout = tags.get("layout", "single")
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
        target = complex_train_rows if split == "train" else complex_holdout_rows
        for _ in range(_layout_balanced_repeats(layout) if split == "train" else 1):
            target.append(stage_row)
        instruction_rows = [
            _row_from_catalog(instruction, output_tokens)
            for instruction in _sft_instruction_variants(prompt)
        ]
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
        "format": "ck.spec04.tokenizer_manifest.v1",
        "line": "structured_svg_atoms",
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
        "format": "ck.spec04.workspace_materialization.v1",
        "source_seed_workspace": str(seed_workspace),
        "workspace": str(workspace),
        "line": "structured_svg_atoms",
        "generator": str(STRUCTURED_TOY_GENERATOR),
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
                "notes": "Blended base coverage plus interleaved minimal-pair edit rows across single, pair, label-card, and badge layouts.",
                "counts": {
                    "train_rows": len(midtrain_train_rows),
                    "dev_rows": len(midtrain_dev_rows),
                    "test_rows": len(midtrain_test_rows),
                    "train_summary": _summarize_midtrain_rows(midtrain_train_rows),
                    "dev_summary": _summarize_midtrain_rows(midtrain_dev_rows),
                    "test_summary": _summarize_midtrain_rows(midtrain_test_rows),
                },
            },
            "sft": {
                "train": _rel(sft_train, workspace),
                "dev": _rel(sft_dev, workspace),
                "test": _rel(sft_test, workspace),
                "notes": "Layout-balanced mixed DSL plus natural-language instruction prompts mapped to the same structured SVG atom outputs.",
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

    summary = {
        "workspace": str(workspace),
        "prefix": prefix,
        "generated_dir": str(generated_dir),
        "pretrain_train_rows": len(train_rows),
        "pretrain_dev_rows": len(dev_rows),
        "pretrain_test_rows": len(test_rows),
        "midtrain_train_rows": len(midtrain_train_rows),
        "midtrain_dev_rows": len(midtrain_dev_rows),
        "midtrain_test_rows": len(midtrain_test_rows),
        "sft_train_rows": len(sft_train_rows),
        "sft_dev_rows": len(sft_dev_rows),
        "sft_test_rows": len(sft_test_rows),
        "holdout_prompt_rows": len(holdout_prompt_rows),
        "tokenizer_rows": len(tokenizer_corpus_rows),
        "tokenizer_json": str(tokenizer_json_dst),
        "tokenizer_bin": str(tokenizer_bin_dst),
        "workspace_manifest": str(workspace / "manifests" / f"{prefix}_workspace_manifest.json"),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize a cache-local spec04 workspace from the structured SVG toy line")
    ap.add_argument("--workspace", required=True, type=Path, help="Destination workspace (recommended inside a cache run dir)")
    ap.add_argument("--seed-workspace", default=str(DEFAULT_SEED_WORKSPACE), type=Path, help="Seed spec workspace template to copy")
    ap.add_argument("--prefix", default="spec04_structured_svg_atoms", help="Artifact prefix for generated files")
    ap.add_argument("--train-repeats", type=int, default=12, help="How many times to repeat each seen combo in pretrain/train")
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
