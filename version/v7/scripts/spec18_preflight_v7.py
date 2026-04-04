#!/usr/bin/env python3
"""Preflight gates for spec18 routing-first scene-bundle training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_spec18_probe_contract_v7 import build_contract as build_probe_contract
from spec12_preflight_v7 import (
    DEFAULT_TOKENIZER_LIB,
    _budget_stage,
    _catalog_path,
    _load_json,
    _parse_prompt_tags,
    _safe_int,
    _stage_dataset_path,
)
from render_svg_structured_scene_spec16_v7 import render_structured_scene_spec16_svg
from spec16_scene_bundle_canonicalizer_v7 import canonicalize_scene_bundle_text, serialize_scene_bundle


def _validate_bundle_output(output_tokens: str, expected_svg: str | None, prompt: str, content_json: dict[str, Any] | None) -> dict[str, Any]:
    tokens = [tok.strip() for tok in str(output_tokens or "").split() if tok.strip()]
    starts_clean = bool(tokens) and tokens[0] == "[bundle]"
    ends_clean = bool(tokens) and tokens[-1] == "[/bundle]"
    try:
        bundle = canonicalize_scene_bundle_text(output_tokens)
        parse_ok = True
        parse_error = None
        canonical_text = serialize_scene_bundle(bundle)
    except Exception as exc:
        bundle = None
        parse_ok = False
        parse_error = str(exc)
        canonical_text = ""

    try:
        rendered_svg = render_structured_scene_spec16_svg(output_tokens, content=content_json)
        renderable = bool(rendered_svg.strip())
        render_error = None
    except Exception as exc:
        rendered_svg = ""
        renderable = False
        render_error = str(exc)

    expected_svg_norm = str(expected_svg or "").strip()
    rendered_svg_norm = rendered_svg.strip()
    svg_exact = bool(expected_svg_norm) and expected_svg_norm == rendered_svg_norm
    prompt_tags = _parse_prompt_tags(prompt)
    return {
        "prompt": prompt,
        "layout": prompt_tags.get("layout"),
        "starts_clean": bool(starts_clean),
        "ends_clean": bool(ends_clean),
        "parse_ok": bool(parse_ok),
        "parse_error": parse_error,
        "renderable": bool(renderable),
        "render_error": render_error,
        "svg_exact": bool(svg_exact),
        "bundle": None if bundle is None else bundle.to_dict(),
        "canonical_output": canonical_text,
    }


def _catalog_validation(run_dir: Path, prefix: str) -> dict[str, Any]:
    catalog_rows = _load_json(_catalog_path(run_dir, prefix))
    if not isinstance(catalog_rows, list):
        raise SystemExit("spec18 render catalog must be a JSON list")
    failures: list[dict[str, Any]] = []
    per_layout: dict[str, dict[str, int]] = {}
    ok = 0
    for row in catalog_rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "")
        output_tokens = str(row.get("output_tokens") or "")
        expected_svg = str(row.get("svg_xml") or "")
        content_json = row.get("content_json") if isinstance(row.get("content_json"), dict) else None
        result = _validate_bundle_output(output_tokens, expected_svg, prompt, content_json)
        layout = str(row.get("family") or row.get("layout") or result.get("layout") or "unknown")
        stats = per_layout.setdefault(layout, {"count": 0, "starts_clean": 0, "ends_clean": 0, "parse_ok": 0, "renderable": 0, "svg_exact": 0})
        stats["count"] += 1
        for key in ("starts_clean", "ends_clean", "parse_ok", "renderable", "svg_exact"):
            if bool(result.get(key)):
                stats[key] += 1
        passed = all(bool(result.get(key)) for key in ("starts_clean", "ends_clean", "parse_ok", "renderable", "svg_exact"))
        if passed:
            ok += 1
        elif len(failures) < 24:
            failures.append(
                {
                    "prompt": prompt,
                    "layout": layout,
                    "starts_clean": result.get("starts_clean"),
                    "ends_clean": result.get("ends_clean"),
                    "parse_ok": result.get("parse_ok"),
                    "parse_error": result.get("parse_error"),
                    "renderable": result.get("renderable"),
                    "render_error": result.get("render_error"),
                    "svg_exact": result.get("svg_exact"),
                }
            )
    total = sum(int(stats.get("count", 0)) for stats in per_layout.values())
    return {
        "count": int(total),
        "pass_count": int(ok),
        "pass_rate": float(ok) / float(max(1, total)),
        "layout_summary": per_layout,
        "failures": failures,
        "pass": int(ok) == int(total),
    }


def _canary_validation(run_dir: Path, prefix: str, per_split: int) -> dict[str, Any]:
    contract = build_probe_contract(
        run_dir,
        prefix,
        max(1, int(per_split)),
        hidden_per_split=max(1, int(per_split)),
    )
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
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
            result = _validate_bundle_output(expected_output, expected_svg, prompt, content_json)
            layout = str((content_json or {}).get("family") or result.get("layout") or "unknown")
            record = {
                "split": split_name,
                "label": case.get("label"),
                "prompt": prompt,
                "layout": layout,
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
    ap = argparse.ArgumentParser(description="Preflight gates for spec18 routing-first scene-bundle training")
    ap.add_argument("--run", required=True, type=Path, help="Run directory")
    ap.add_argument("--prefix", default="spec18_scene_bundle", help="Dataset prefix")
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
        "schema": "ck.spec18_preflight.v1",
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
