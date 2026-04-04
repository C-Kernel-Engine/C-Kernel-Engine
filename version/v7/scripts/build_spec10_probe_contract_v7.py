#!/usr/bin/env python3
"""Build a balanced probe contract for spec10 asset-grounded scene DSL runs."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

from build_spec06_probe_contract_v7 import _detect_prefix


LAYOUT_ORDER = (
    "poster_stack",
    "comparison_span_chart",
    "pipeline_lane",
    "dual_panel_compare",
    "dashboard_cards",
)
THEME_ORDER = ("infra_dark", "paper_editorial", "signal_glow")
TONE_ORDER = ("amber", "green", "blue", "purple", "mixed")
DENSITY_ORDER = ("compact", "balanced", "airy")
TOPIC_ORDER = (
    "memory_reality",
    "performance_balance",
    "pipeline_overview",
    "cpu_gpu_cost",
    "ethernet_equalizer",
    "training_intuition",
)


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
    return (
        tags.get("layout", ""),
        tags.get("topic", ""),
        tags.get("theme", ""),
        tags.get("tone", ""),
    )


def _label_prefix(layout: str) -> str:
    return {
        "poster_stack": "Poster",
        "comparison_span_chart": "Compare",
        "pipeline_lane": "Pipeline",
        "dual_panel_compare": "Dual",
        "dashboard_cards": "Dashboard",
    }.get(layout, layout or "Case")


def _layout_case_budget(limit: int) -> dict[str, int]:
    budgets = {layout: 1 for layout in LAYOUT_ORDER}
    remaining = max(0, int(limit) - len(LAYOUT_ORDER))
    idx = 0
    while remaining > 0:
        budgets[LAYOUT_ORDER[idx % len(LAYOUT_ORDER)]] += 1
        remaining -= 1
        idx += 1
    return budgets


def _topic_rank(topic: str) -> int:
    try:
        return TOPIC_ORDER.index(topic)
    except ValueError:
        return len(TOPIC_ORDER)


def _prompt_sort_key(prompt: str) -> tuple[int, int, int, int, int, str]:
    tags = _parse_prompt_tags(prompt)
    topic = tags.get("topic", "")
    theme = tags.get("theme", "")
    tone = tags.get("tone", "")
    density = tags.get("density", "")
    return (
        _topic_rank(topic),
        THEME_ORDER.index(theme) if theme in THEME_ORDER else len(THEME_ORDER),
        TONE_ORDER.index(tone) if tone in TONE_ORDER else len(TONE_ORDER),
        DENSITY_ORDER.index(density) if density in DENSITY_ORDER else len(DENSITY_ORDER),
        tags.get("layout", ""),
        prompt,
    )


def _pick_balanced_prompts(prompts: list[str], limit: int) -> list[str]:
    by_layout: dict[str, list[str]] = defaultdict(list)
    for prompt in prompts:
        layout = _parse_prompt_tags(prompt).get("layout", "")
        by_layout[layout].append(prompt)

    budgets = _layout_case_budget(limit)
    selected: list[str] = []
    chosen: set[str] = set()
    for layout in LAYOUT_ORDER:
        layout_prompts = sorted(by_layout.get(layout, []), key=_prompt_sort_key)
        groups: "OrderedDict[tuple[str, ...], list[str]]" = OrderedDict()
        for prompt in layout_prompts:
            key = _selection_group_key(_parse_prompt_tags(prompt))
            groups.setdefault(key, []).append(prompt)
        budget = budgets.get(layout, 0)
        while budget > 0 and groups:
            progressed = False
            for key in list(groups.keys()):
                prompt_group = groups[key]
                while prompt_group and prompt_group[0] in chosen:
                    prompt_group.pop(0)
                if not prompt_group:
                    groups.pop(key, None)
                    continue
                prompt = prompt_group.pop(0)
                selected.append(prompt)
                chosen.add(prompt)
                budget -= 1
                progressed = True
                if not prompt_group:
                    groups.pop(key, None)
                if budget <= 0:
                    break
            if not progressed:
                break
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


def build_contract(run_dir: Path, prefix: str, per_split: int) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    dataset_root = run_dir / "dataset" if (run_dir / "dataset").exists() else run_dir
    catalog_path = _find_existing(
        [
            dataset_root / "manifests" / "generated" / "structured_atoms" / f"{prefix}_render_catalog.json",
            dataset_root / "tokenizer" / f"{prefix}_render_catalog.json",
        ]
    )
    if catalog_path is None:
        raise SystemExit(f"render catalog not found for prefix '{prefix}' under dataset/")

    catalog_rows = _load_json(catalog_path)
    if not isinstance(catalog_rows, list):
        raise SystemExit(f"expected JSON list render catalog: {catalog_path}")
    catalog = {
        str(row.get("prompt") or "").strip(): row
        for row in catalog_rows
        if isinstance(row, dict) and str(row.get("prompt") or "").strip()
    }

    split_defs = [
        ("train", "Train", dataset_root / "pretrain" / "train" / f"{prefix}_pretrain_train.txt"),
        ("dev", "Dev", dataset_root / "pretrain" / "dev" / f"{prefix}_pretrain_dev.txt"),
        ("test", "Test", dataset_root / "pretrain" / "test" / f"{prefix}_pretrain_test.txt"),
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
        "title": f"{prefix} Asset-Grounded Scene DSL Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec09.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 256,
            "temperature": 0.0,
            # Keep structural close markers in the adapter layer; ck_chat.py
            # strips matched --stop-on-text strings from the returned response.
        },
        "splits": splits,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a balanced probe contract for a spec10 asset-grounded scene DSL run")
    ap.add_argument("--run", required=True, type=Path, help="Run directory containing spec10 scene DSL dataset artifacts")
    ap.add_argument("--prefix", default=None, help="Dataset prefix, auto-detected from dataset/tokenizer if omitted")
    ap.add_argument("--output", required=True, type=Path, help="Output contract JSON")
    ap.add_argument("--per-split", type=int, default=12, help="How many balanced cases to include per split")
    args = ap.parse_args()

    prefix = _detect_prefix(args.run.expanduser().resolve(), args.prefix)
    contract = build_contract(args.run, prefix, max(1, int(args.per_split)))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
