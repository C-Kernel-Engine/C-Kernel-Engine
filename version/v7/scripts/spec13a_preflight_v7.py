#!/usr/bin/env python3
"""Preflight gates for spec13a scene-DSL training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_spec13a_probe_contract_v7 import build_contract as build_probe_contract
from spec12_preflight_v7 import (
    DEFAULT_TOKENIZER_LIB,
    _budget_stage,
    _catalog_validation,
    _safe_int,
    _stage_dataset_path,
    _validate_scene_output,
)


def _canary_validation(run_dir: Path, prefix: str, per_split: int) -> dict[str, object]:
    contract = build_probe_contract(
        run_dir,
        prefix,
        max(1, int(per_split)),
        hidden_per_split=max(1, int(per_split)),
    )
    cases: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    ok = 0
    total = 0
    for split in contract.get("splits") or []:
        split_name = str((split or {}).get("name") or "probe")
        for case in (split or {}).get("cases") or []:
            if not isinstance(case, dict):
                continue
            total += 1
            prompt = str(case.get("prompt") or "")
            expected_output = str(case.get("expected_output") or "")
            expected_svg = str(case.get("expected_rendered_output") or "")
            content_json = case.get("content_json") if isinstance(case.get("content_json"), dict) else None
            result = _validate_scene_output(expected_output, expected_svg, prompt, content_json)
            record = {
                "split": split_name,
                "label": case.get("label"),
                "prompt": prompt,
                "layout": result.get("layout"),
                "topic": result.get("topic"),
                "starts_clean": result.get("starts_clean"),
                "ends_clean": result.get("ends_clean"),
                "parse_ok": result.get("parse_ok"),
                "renderable": result.get("renderable"),
                "svg_exact": result.get("svg_exact"),
            }
            cases.append(record)
            passed = all(bool(result.get(key)) for key in ("starts_clean", "ends_clean", "parse_ok", "renderable", "svg_exact"))
            if passed:
                ok += 1
            elif len(failures) < 24:
                failed = dict(record)
                failed["parse_error"] = result.get("parse_error")
                failed["render_error"] = result.get("render_error")
                failures.append(failed)
    return {
        "count": int(total),
        "pass_count": int(ok),
        "pass_rate": float(ok) / float(max(1, total)),
        "cases": cases,
        "failures": failures,
        "pass": int(ok) == int(total),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight gates for spec13a scene-DSL training")
    ap.add_argument("--run", required=True, type=Path, help="Run directory")
    ap.add_argument("--prefix", default="spec13a_scene_dsl", help="Dataset prefix")
    ap.add_argument("--tokenizer-json", required=True, type=Path)
    ap.add_argument("--tokenizer-bin", required=True, type=Path)
    ap.add_argument("--seq-len", type=int, default=768)
    ap.add_argument("--pretrain-epochs", type=float, default=1.0)
    ap.add_argument("--midtrain-epochs", type=float, default=1.0)
    ap.add_argument("--current-pretrain-total-tokens", default=None)
    ap.add_argument("--current-midtrain-total-tokens", default=None)
    ap.add_argument("--canary-per-split", type=int, default=4)
    ap.add_argument("--json-out", required=True, type=Path)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    run_dir = args.run.expanduser().resolve()
    prefix = str(args.prefix)
    tokenizer_json = args.tokenizer_json.expanduser().resolve()
    tokenizer_bin = args.tokenizer_bin.expanduser().resolve()
    pretrain = _budget_stage(
        name="pretrain",
        dataset_path=_stage_dataset_path(run_dir, prefix, "pretrain"),
        tokenizer_json=tokenizer_json,
        tokenizer_bin=tokenizer_bin,
        tokenizer_lib=DEFAULT_TOKENIZER_LIB,
        seq_len=int(args.seq_len),
        target_epochs=float(args.pretrain_epochs),
        current_total_tokens=_safe_int(args.current_pretrain_total_tokens),
    )
    midtrain = _budget_stage(
        name="midtrain",
        dataset_path=_stage_dataset_path(run_dir, prefix, "midtrain"),
        tokenizer_json=tokenizer_json,
        tokenizer_bin=tokenizer_bin,
        tokenizer_lib=DEFAULT_TOKENIZER_LIB,
        seq_len=int(args.seq_len),
        target_epochs=float(args.midtrain_epochs),
        current_total_tokens=_safe_int(args.current_midtrain_total_tokens),
    )
    catalog = _catalog_validation(run_dir, prefix)
    canary = _canary_validation(run_dir, prefix, int(args.canary_per_split))
    payload = {
        "schema": "ck.spec13a_preflight.v1",
        "run": str(run_dir),
        "prefix": prefix,
        "seq_len": int(args.seq_len),
        "stages": {"pretrain": pretrain, "midtrain": midtrain},
        "catalog": catalog,
        "canary": canary,
        "pass": bool(pretrain["budget_gate_pass"] and midtrain["budget_gate_pass"] and catalog["pass"] and canary["pass"]),
    }
    args.json_out.expanduser().resolve().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.strict and not payload["pass"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
