#!/usr/bin/env python3
"""Build a composition-focused midtrain corpus for spec04 structured SVG atoms."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


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
    layout = str(tags.get("layout") or "").strip()
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


def _row_line(prompt: str, output_tokens: str) -> str:
    return f"{prompt} {output_tokens}".strip()


def _row_kind(tags: dict[str, str]) -> str:
    layout = str(tags.get("layout") or "").strip()
    if layout in {"pair-h", "pair-v"}:
        return "pair"
    if layout in {"label-card", "badge"}:
        return layout
    return layout or "unknown"


def _flip_bg(bg: str) -> str | None:
    return {
        "paper": "mint",
        "mint": "paper",
        "slate": "paper",
    }.get(str(bg or "").strip())


def _alt_value(current: str, candidates: tuple[str, ...]) -> str | None:
    for value in candidates:
        if value != current:
            return value
    return None


def _candidate_edits(tags: dict[str, str]) -> list[tuple[str, dict[str, str]]]:
    layout = str(tags.get("layout") or "").strip()
    edits: list[tuple[str, dict[str, str]]] = []
    if layout == "pair-h":
        target = dict(tags)
        target["layout"] = "pair-v"
        edits.append(("layout=pair-v", target))
    elif layout == "pair-v":
        target = dict(tags)
        target["layout"] = "pair-h"
        edits.append(("layout=pair-h", target))

    bg_alt = _flip_bg(str(tags.get("bg") or ""))
    if bg_alt:
        target = dict(tags)
        target["bg"] = bg_alt
        edits.append((f"bg={bg_alt}", target))

    if layout == "badge":
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
    elif layout in {"pair-h", "pair-v"}:
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
    return edits


def _complex_rows(render_catalog: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in render_catalog:
        if not isinstance(row, dict):
            continue
        if str(row.get("split") or "").strip() != split:
            continue
        kind = str(row.get("kind") or "").strip()
        if kind not in {"pair", "label-card", "badge"}:
            continue
        prompt = str(row.get("prompt") or "").strip()
        output_tokens = str(row.get("output_tokens") or "").strip()
        if not prompt or not output_tokens:
            continue
        tags = _parse_prompt_tags(prompt)
        rows.append(
            {
                "kind": kind,
                "prompt": prompt,
                "output_tokens": output_tokens,
                "svg_xml": str(row.get("svg_xml") or ""),
                "tags": tags,
            }
        )
    return rows


def _build_split_rows(rows: list[dict[str, Any]], *, rng: random.Random, edit_repeat: int) -> tuple[list[str], dict[str, Any]]:
    prompt_index = {str(row["prompt"]): row for row in rows}
    direct_rows: list[str] = []
    edit_rows: list[str] = []
    edit_counts: Counter[str] = Counter()
    direct_kind_counts: Counter[str] = Counter()

    for row in rows:
        prompt = str(row["prompt"])
        output_tokens = str(row["output_tokens"])
        tags = dict(row["tags"])
        kind = _row_kind(tags)
        direct_kind_counts[kind] += 1
        direct_rows.append(_row_line(prompt, output_tokens))

        for edit_op, target_tags in _candidate_edits(tags):
            target_prompt = _prompt_from_tags(target_tags)
            target = prompt_index.get(target_prompt)
            if target is None:
                continue
            edit_prompt = _prompt_from_tags(tags, include_edit=edit_op)
            line = _row_line(edit_prompt, str(target["output_tokens"]))
            for _ in range(max(1, int(edit_repeat))):
                edit_rows.append(line)
            edit_counts[edit_op.split("=", 1)[0]] += 1

    out_rows = [*direct_rows, *edit_rows]
    rng.shuffle(out_rows)
    summary = {
        "direct_rows": len(direct_rows),
        "edit_rows": len(edit_rows),
        "total_rows": len(out_rows),
        "direct_kind_counts": dict(sorted(direct_kind_counts.items())),
        "edit_counts": dict(sorted(edit_counts.items())),
    }
    return out_rows, summary


def build_midtrain_corpus(run_dir: Path, *, seed: int, edit_repeat: int, manifest_out: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    render_catalog_path = _find_existing(
        [
            run_dir / "dataset" / "manifests" / "generated" / "structured_atoms" / "spec04_structured_svg_atoms_render_catalog.json",
            run_dir / "dataset" / "tokenizer" / "spec04_structured_svg_atoms_render_catalog.json",
            run_dir / "spec04_workspace" / "manifests" / "generated" / "structured_atoms" / "spec04_structured_svg_atoms_render_catalog.json",
            run_dir / "spec04_workspace" / "tokenizer" / "spec04_structured_svg_atoms_render_catalog.json",
        ]
    )
    if render_catalog_path is None:
        raise SystemExit("spec04 render catalog not found under dataset/ or spec04_workspace/")

    render_catalog = _load_json(render_catalog_path)
    if not isinstance(render_catalog, list):
        raise SystemExit(f"expected JSON list render catalog: {render_catalog_path}")

    rng = random.Random(int(seed))
    train_rows = _complex_rows(render_catalog, "train")
    holdout_rows = _complex_rows(render_catalog, "holdout")
    holdout_rows.sort(key=lambda row: str(row["prompt"]))
    holdout_dev = [row for idx, row in enumerate(holdout_rows) if idx % 2 == 0]
    holdout_test = [row for idx, row in enumerate(holdout_rows) if idx % 2 == 1]
    if holdout_rows and not holdout_test:
        holdout_test.append(holdout_dev.pop())

    train_lines, train_summary = _build_split_rows(train_rows, rng=rng, edit_repeat=int(edit_repeat))
    dev_lines, dev_summary = _build_split_rows(holdout_dev, rng=rng, edit_repeat=1)
    test_lines, test_summary = _build_split_rows(holdout_test, rng=rng, edit_repeat=1)

    train_path = run_dir / "dataset" / "midtrain" / "train" / "spec04_structured_svg_atoms_midtrain_composition_train.txt"
    dev_path = run_dir / "dataset" / "midtrain" / "dev" / "spec04_structured_svg_atoms_midtrain_composition_dev.txt"
    test_path = run_dir / "dataset" / "midtrain" / "test" / "spec04_structured_svg_atoms_midtrain_composition_test.txt"
    _write_lines(train_path, train_lines)
    _write_lines(dev_path, dev_lines)
    _write_lines(test_path, test_lines)

    manifest = {
        "format": "ck.spec04.midtrain_composition_manifest.v1",
        "run_dir": str(run_dir),
        "seed": int(seed),
        "edit_repeat": int(edit_repeat),
        "render_catalog": str(render_catalog_path),
        "artifacts": {
            "train": str(train_path),
            "dev": str(dev_path),
            "test": str(test_path),
        },
        "splits": {
            "train": train_summary,
            "dev": dev_summary,
            "test": test_summary,
        },
    }
    manifest_path = manifest_out if manifest_out is not None else (
        run_dir / "dataset" / "manifests" / "generated" / "structured_atoms" / "spec04_structured_svg_atoms_midtrain_composition_manifest.json"
    )
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a composition-focused midtrain corpus for spec04")
    ap.add_argument("--run", required=True, type=Path, help="Run directory containing spec04 dataset artifacts")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    ap.add_argument("--edit-repeat", type=int, default=2, help="How many times to repeat each edit row in train")
    ap.add_argument("--manifest-out", type=Path, default=None, help="Optional manifest output path")
    args = ap.parse_args()

    manifest = build_midtrain_corpus(
        args.run,
        seed=int(args.seed),
        edit_repeat=max(1, int(args.edit_repeat)),
        manifest_out=args.manifest_out.expanduser().resolve() if args.manifest_out else None,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
