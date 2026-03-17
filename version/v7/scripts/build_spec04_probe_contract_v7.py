#!/usr/bin/env python3
"""Build a balanced minimal-pair probe contract for spec04 structured scenes."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _extract_prompt(line: str) -> str:
    marker = " [OUT] "
    if marker in line:
        return line.split(marker, 1)[0] + " [OUT]"
    return line.strip()


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


def _load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        prompt = _extract_prompt(raw.strip())
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    return prompts


def _selection_group_key(tags: dict[str, str]) -> tuple[str, ...]:
    layout = tags.get("layout", "")
    if layout == "single":
        return (layout, tags.get("shape", ""), tags.get("color", ""), tags.get("size", ""))
    if layout in {"pair-h", "pair-v"}:
        return (
            layout,
            tags.get("shape", ""),
            tags.get("shape2", ""),
            tags.get("color", ""),
            tags.get("color2", ""),
            tags.get("size", ""),
        )
    if layout == "label-card":
        return (layout, tags.get("color", ""), tags.get("bg", ""))
    if layout == "badge":
        return (layout, tags.get("shape", ""), tags.get("color", ""), tags.get("bg", ""))
    return (layout,)


def _label_prefix(layout: str) -> str:
    return {
        "single": "Single",
        "pair-h": "Pair-H",
        "pair-v": "Pair-V",
        "label-card": "Label",
        "badge": "Badge",
    }.get(layout, layout or "Case")


def _pick_balanced_prompts(prompts: list[str], limit: int) -> list[str]:
    by_layout: dict[str, list[str]] = defaultdict(list)
    for prompt in prompts:
        layout = _parse_prompt_tags(prompt).get("layout", "")
        by_layout[layout].append(prompt)

    ordered_layouts = ["single", "pair-h", "pair-v", "label-card", "badge"]
    selected: list[str] = []
    chosen: set[str] = set()

    for layout in ordered_layouts:
        layout_prompts = by_layout.get(layout, [])
        groups: "OrderedDict[tuple[str, ...], list[str]]" = OrderedDict()
        for prompt in layout_prompts:
            key = _selection_group_key(_parse_prompt_tags(prompt))
            groups.setdefault(key, []).append(prompt)
        pair = next((group[:2] for group in groups.values() if len(group) >= 2), None)
        if pair:
            for prompt in pair:
                if prompt not in chosen:
                    selected.append(prompt)
                    chosen.add(prompt)
            continue
        if layout_prompts:
            prompt = layout_prompts[0]
            if prompt not in chosen:
                selected.append(prompt)
                chosen.add(prompt)

    return selected[:limit]


def _case_from_prompt(
    prompt: str,
    catalog: dict[str, dict[str, Any]],
    split: str,
    split_label: str,
    index: int,
) -> dict[str, Any]:
    row = catalog.get(prompt) or {}
    layout = _parse_prompt_tags(prompt).get("layout", "")
    return {
        "id": f"{split}_{index:02d}",
        "label": f"{split_label} {_label_prefix(layout)} #{index}",
        "prompt": prompt,
        "expected_output": row.get("output_tokens"),
        "expected_rendered_output": row.get("svg_xml"),
        "expected_rendered_mime": "image/svg+xml" if row.get("svg_xml") else None,
    }


def build_contract(run_dir: Path, per_split: int) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    catalog_path = _find_existing(
        [
            run_dir / "dataset" / "manifests" / "generated" / "structured_atoms" / "spec04_structured_svg_atoms_render_catalog.json",
            run_dir / "dataset" / "tokenizer" / "spec04_structured_svg_atoms_render_catalog.json",
            run_dir / "spec04_workspace" / "manifests" / "generated" / "structured_atoms" / "spec04_structured_svg_atoms_render_catalog.json",
            run_dir / "spec04_workspace" / "tokenizer" / "spec04_structured_svg_atoms_render_catalog.json",
        ]
    )
    if catalog_path is None:
        raise SystemExit("spec04 render catalog not found under dataset/ or spec04_workspace/")

    catalog_rows = _load_json(catalog_path)
    if not isinstance(catalog_rows, list):
        raise SystemExit(f"expected JSON list render catalog: {catalog_path}")
    catalog = {
        str(row.get("prompt") or "").strip(): row
        for row in catalog_rows
        if isinstance(row, dict) and str(row.get("prompt") or "").strip()
    }

    split_defs = [
        (
            "train",
            "Train",
            run_dir / "dataset" / "pretrain" / "train" / "spec04_structured_svg_atoms_pretrain_train.txt",
        ),
        (
            "dev",
            "Dev",
            run_dir / "dataset" / "pretrain" / "dev" / "spec04_structured_svg_atoms_pretrain_dev.txt",
        ),
        (
            "test",
            "Test",
            run_dir / "dataset" / "pretrain" / "test" / "spec04_structured_svg_atoms_pretrain_test.txt",
        ),
    ]

    splits: list[dict[str, Any]] = []
    for split_name, split_label, path in split_defs:
        if not path.exists():
            raise SystemExit(f"dataset split not found: {path}")
        prompts = _load_prompts(path)
        selected_prompts = _pick_balanced_prompts(prompts, max(1, int(per_split)))
        splits.append(
            {
                "name": split_name,
                "label": f"{split_label} prompts",
                "cases": [
                    _case_from_prompt(prompt, catalog, split_name, split_label, index)
                    for index, prompt in enumerate(selected_prompts, start=1)
                ],
            }
        )

    return {
        "schema": "ck.probe_report_contract.v1",
        "title": "Spec04 Structured Scenes Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/svg]"],
            "renderer": "structured_svg_atoms.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 95,
            "temperature": 0.0,
            "stop_on_text": "<|eos|>",
        },
        "splits": splits,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a balanced minimal-pair probe contract for a spec04 run")
    ap.add_argument("--run", required=True, type=Path, help="Run directory containing spec04 dataset artifacts")
    ap.add_argument("--output", required=True, type=Path, help="Output contract JSON")
    ap.add_argument("--per-split", type=int, default=10, help="How many balanced cases to include per split")
    args = ap.parse_args()

    contract = build_contract(args.run, max(1, int(args.per_split)))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
