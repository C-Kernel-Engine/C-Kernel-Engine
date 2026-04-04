#!/usr/bin/env python3
"""Build a balanced probe contract for spec14a comparison-board runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pack_training_tokens_v7 import _TrueBPEHandle


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOKENIZER_LIB = ROOT / "build" / "libckernel_tokenizer.so"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _detect_prefix_local(run_dir: Path, explicit_prefix: str | None) -> str:
    if explicit_prefix:
        return str(explicit_prefix).strip()
    candidates = sorted((run_dir / "dataset" / "tokenizer").glob("*_render_catalog.json"))
    if not candidates:
        candidates = sorted((run_dir / "tokenizer").glob("*_render_catalog.json"))
    if not candidates:
        raise SystemExit("no *_render_catalog.json found under dataset/tokenizer or tokenizer")
    return candidates[0].name[: -len("_render_catalog.json")]


def _run_tokenizer_paths(run_dir: Path) -> tuple[Path | None, Path | None]:
    tokenizer_json_candidates = [run_dir / "tokenizer.json", run_dir / "dataset" / "tokenizer" / "tokenizer.json"]
    tokenizer_bin_candidates = [run_dir / "tokenizer_bin", run_dir / "dataset" / "tokenizer" / "tokenizer_bin"]
    tokenizer_json = next((p for p in tokenizer_json_candidates if p.exists()), None)
    tokenizer_bin = next((p for p in tokenizer_bin_candidates if p.exists()), None)
    return tokenizer_json, tokenizer_bin


def _run_context_len(run_dir: Path) -> int | None:
    for candidate in (run_dir / "config.json", run_dir / "weights_manifest.json", run_dir / "train_init_config.json"):
        if not candidate.exists():
            continue
        try:
            doc = _load_json(candidate)
        except Exception:
            continue
        if isinstance(doc, dict):
            arch = doc.get("architecture")
            if isinstance(arch, dict) and arch.get("context_len"):
                return int(arch.get("context_len"))
            cfg = doc.get("config")
            if isinstance(cfg, dict) and cfg.get("context_len"):
                return int(cfg.get("context_len"))
    return None


def _round_up(value: int, quantum: int) -> int:
    return ((max(1, int(value)) + int(quantum) - 1) // int(quantum)) * int(quantum)


def _recommend_decode_max_tokens(run_dir: Path, catalog: dict[str, dict[str, Any]]) -> int:
    tokenizer_json, tokenizer_bin = _run_tokenizer_paths(run_dir)
    if tokenizer_json is None or tokenizer_bin is None or not DEFAULT_TOKENIZER_LIB.exists():
        return 512
    context_len = _run_context_len(run_dir)
    max_prompt_tokens = 0
    max_expected_tokens = 0
    with _TrueBPEHandle(DEFAULT_TOKENIZER_LIB, tokenizer_bin, tokenizer_json) as handle:
        for prompt, row in catalog.items():
            max_prompt_tokens = max(max_prompt_tokens, len(handle.encode(str(prompt or ""))))
            expected_output = str((row or {}).get("output_tokens") or "").strip()
            if expected_output:
                max_expected_tokens = max(max_expected_tokens, len(handle.encode(expected_output)))
    if max_expected_tokens <= 0:
        return 512
    recommended = _round_up(max_expected_tokens + 16, 32)
    if context_len and max_prompt_tokens > 0:
        safe_cap = max(64, int(context_len) - int(max_prompt_tokens) - 1)
        recommended = min(recommended, safe_cap)
    return max(64, recommended)


def _extract_prompt(line: str) -> str:
    marker = " [OUT] "
    if marker in line:
        return line.split(marker, 1)[0] + " [OUT]"
    return line.strip()


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


def _selection_group_key(prompt: str, catalog: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    row = catalog.get(prompt) or {}
    return (
        str(row.get("layout") or ""),
        str(row.get("case_id") or ""),
        str(row.get("prompt_surface") or ""),
    )


def _prompt_sort_key(prompt: str, catalog: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    row = catalog.get(prompt) or {}
    return (
        str(row.get("layout") or ""),
        str(row.get("case_id") or ""),
        str(row.get("prompt_surface") or ""),
        prompt,
    )


def _pick_balanced_prompts(prompts: list[str], catalog: dict[str, dict[str, Any]], limit: int) -> list[str]:
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for prompt in prompts:
        groups[_selection_group_key(prompt, catalog)].append(prompt)
    for key in list(groups.keys()):
        groups[key] = sorted(groups[key], key=lambda prompt: _prompt_sort_key(prompt, catalog))
    selected: list[str] = []
    while len(selected) < int(limit) and groups:
        progressed = False
        for key in sorted(groups.keys()):
            bucket = groups.get(key) or []
            if not bucket:
                groups.pop(key, None)
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if not bucket:
                groups.pop(key, None)
            if len(selected) >= int(limit):
                break
        if not progressed:
            break
    return selected[: int(limit)]


def _label_prefix(layout: str) -> str:
    return {
        "comparison_board": "Board",
        "decision_tree": "Decision",
        "flow_graph": "Flow",
        "table_matrix": "Table",
        "memory_map": "Memory",
    }.get(layout, layout or "Case")


def _case_from_prompt(prompt: str, catalog: dict[str, dict[str, Any]], split: str, split_label: str, index: int) -> dict[str, Any]:
    row = catalog.get(prompt) or {}
    layout = str(row.get("layout") or "")
    prompt_surface = str(row.get("prompt_surface") or "prompt")
    return {
        "id": f"{split}_{index:02d}",
        "label": f"{split_label} {_label_prefix(layout)} #{index} ({prompt_surface})",
        "prompt": prompt,
        "expected_output": row.get("output_tokens"),
        "expected_rendered_output": row.get("svg_xml"),
        "expected_rendered_mime": "image/svg+xml" if row.get("svg_xml") else None,
        "content_json": row.get("content_json") if isinstance(row.get("content_json"), dict) else None,
    }


def build_contract(run_dir: Path, prefix: str, per_split: int, *, hidden_per_split: int = 0) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    dataset_root = run_dir / "dataset" if (run_dir / "dataset").exists() else run_dir
    catalog_path = _find_existing([
        dataset_root / "manifests" / "generated" / "structured_atoms" / f"{prefix}_render_catalog.json",
        dataset_root / "tokenizer" / f"{prefix}_render_catalog.json",
    ])
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
    decode_max_tokens = _recommend_decode_max_tokens(run_dir, catalog)

    split_defs = [
        ("train", "Train", dataset_root / "holdout" / f"{prefix}_seen_prompts.txt"),
        ("dev", "Dev", dataset_root / "holdout" / f"{prefix}_holdout_prompts_dev.txt"),
        ("test", "Test", dataset_root / "holdout" / f"{prefix}_holdout_prompts_test.txt"),
    ]
    if int(hidden_per_split) > 0:
        split_defs.extend(
            [
                ("hidden_train", "Hidden Train", dataset_root / "holdout" / f"{prefix}_hidden_seen_prompts.txt"),
                ("hidden_test", "Hidden Holdout", dataset_root / "holdout" / f"{prefix}_hidden_holdout_prompts.txt"),
            ]
        )

    splits: list[dict[str, Any]] = []
    for split_name, split_label, path in split_defs:
        if not path.exists():
            continue
        prompts = _load_prompts(path)
        limit = int(hidden_per_split) if split_name.startswith("hidden_") else int(per_split)
        selected = _pick_balanced_prompts(prompts, catalog, limit=limit)
        splits.append(
            {
                "name": split_name,
                "label": f"{split_label} prompts",
                "cases": [_case_from_prompt(prompt, catalog, split_name, split_label, idx) for idx, prompt in enumerate(selected, start=1)],
            }
        )

    return {
        "schema": "ck.probe_report_contract.v1",
        "title": "Spec14a Comparison-Board Probe",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec14a.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": decode_max_tokens,
            "temperature": 0.0,
            "stop_on_text": [
                "Create a comparison board",
                "Plan a comparison board",
                "Draft the structured board",
                "Compose the scene program",
                "Return the board scene",
            ],
        },
        "splits": splits,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--prefix", default=None, help="Dataset prefix (auto-detect when omitted)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--per-split", type=int, default=12)
    ap.add_argument("--hidden-per-split", type=int, default=8)
    args = ap.parse_args()

    prefix = _detect_prefix_local(args.run.expanduser().resolve(), args.prefix)
    contract = build_contract(args.run, prefix, per_split=int(args.per_split), hidden_per_split=int(args.hidden_per_split))
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
