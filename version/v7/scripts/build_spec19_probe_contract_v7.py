#!/usr/bin/env python3
"""Build a balanced probe contract for spec19 textbook-routing scene bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_spec17_probe_contract_v7 import (
    _case_from_prompt,
    _detect_prefix_local,
    _find_existing,
    _load_json,
    _load_prompts,
    _pick_balanced_prompts,
    _recommend_decode_max_tokens,
)


VISIBLE_PROMPT_SURFACE_ORDER = [
    "explicit_bundle_anchor",
    "explicit_permuted_anchor",
    "clean_stop_anchor",
    "routebook_direct",
    "routebook_direct_hint",
    "form_minimal_pair",
    "family_minimal_pair",
    "routebook_paraphrase",
    "style_topology_bridge",
    "recombination_bridge",
    "holdout_routebook_direct",
    "holdout_style_topology_bridge",
]
HIDDEN_PROMPT_SURFACE_ORDER = [
    "hidden_routebook_paraphrase",
    "hidden_recombination",
]


def build_contract(
    run_dir: Path,
    prefix: str,
    per_split: int,
    *,
    hidden_per_split: int = 0,
    repairer: str | None = "spec16_scene_bundle.v1",
) -> dict[str, Any]:
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

    import build_spec17_probe_contract_v7 as spec17_probe

    original_visible = list(spec17_probe.VISIBLE_PROMPT_SURFACE_ORDER)
    original_hidden = list(spec17_probe.HIDDEN_PROMPT_SURFACE_ORDER)
    spec17_probe.VISIBLE_PROMPT_SURFACE_ORDER = list(VISIBLE_PROMPT_SURFACE_ORDER)
    spec17_probe.HIDDEN_PROMPT_SURFACE_ORDER = list(HIDDEN_PROMPT_SURFACE_ORDER)
    try:
        splits: list[dict[str, Any]] = []
        for split_name, split_label, path in split_defs:
            if not path.exists():
                continue
            prompts = _load_prompts(path)
            limit = int(hidden_per_split) if split_name.startswith("hidden_") else int(per_split)
            selected = _pick_balanced_prompts(prompts, catalog, limit=limit, hidden=split_name.startswith("hidden_"))
            splits.append(
                {
                    "name": split_name,
                    "label": f"{split_label} prompts",
                    "cases": [_case_from_prompt(prompt, catalog, split_name, split_label, idx) for idx, prompt in enumerate(selected, start=1)],
                }
            )
    finally:
        spec17_probe.VISIBLE_PROMPT_SURFACE_ORDER = original_visible
        spec17_probe.HIDDEN_PROMPT_SURFACE_ORDER = original_hidden

    output_adapter: dict[str, Any] = {
        "name": "text_renderer",
        "stop_markers": ["[/bundle]"],
        "renderer": "structured_svg_scene_spec16.v1",
        "preview_mime": "image/svg+xml",
    }
    repairer_name = str(repairer or "").strip()
    if repairer_name:
        output_adapter["repairer"] = repairer_name

    return {
        "schema": "ck.probe_report_contract.v1",
        "title": "Spec19 Textbook Routing Scene Bundle Probe",
        "dataset_type": "svg",
        "output_adapter": output_adapter,
        "decode": {
            "max_tokens": decode_max_tokens,
            "temperature": 0.0,
            "stop_on_text": [
                "pick one shared bundle",
                "return bundle only",
                "Plan exactly one compiler-facing shared visual bundle",
            ],
        },
        "splits": splits,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a balanced probe contract for spec19 textbook-routing scene bundles.")
    ap.add_argument("--run", required=True, type=Path, help="Run directory")
    ap.add_argument("--prefix", default=None, help="Dataset prefix")
    ap.add_argument("--per-split", type=int, default=12)
    ap.add_argument("--hidden-per-split", type=int, default=6)
    ap.add_argument("--repairer", default="spec16_scene_bundle.v1", help="Optional registered repairer name")
    ap.add_argument("--no-repairer", action="store_true", help="Omit repairer from the probe contract")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    run_dir = args.run.expanduser().resolve()
    prefix = _detect_prefix_local(run_dir, args.prefix)
    repairer_name = None if args.no_repairer else str(args.repairer or "").strip()
    contract = build_contract(
        run_dir,
        prefix,
        int(args.per_split),
        hidden_per_split=int(args.hidden_per_split),
        repairer=repairer_name,
    )
    args.output.expanduser().resolve().write_text(json.dumps(contract, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
