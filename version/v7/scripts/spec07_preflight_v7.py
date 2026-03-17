#!/usr/bin/env python3
"""Preflight gates for spec07 scene-DSL training.

This script does three things before a real spec07 run:
1. Computes packed one-epoch token budgets for pretrain and midtrain using the
   actual staged tokenizer and the same row-aware packing assumptions as
   training.
2. Validates scene-DSL hygiene and compiler roundtrip correctness on the full
   render catalog.
3. Builds and checks a small balanced canary probe so failures are easy to read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_spec07_probe_contract_v7 import build_contract as build_probe_contract
from pack_training_tokens_v7 import (
    _TrueBPEHandle,
    _encode_row_with_fallback,
    _pack_rows,
    _read_rows,
    _resolve_special_ids,
)
from render_svg_structured_scene_v7 import _parse_scene_document, render_structured_scene_svg


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOKENIZER_LIB = ROOT / "build" / "libckernel_tokenizer.so"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _token_value(token: str, prefix: str) -> str | None:
    if token.startswith(prefix) and token.endswith("]"):
        return token[len(prefix) : -1]
    return None


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


def _stage_dataset_path(run_dir: Path, prefix: str, stage: str) -> Path:
    return run_dir / "dataset" / stage / "train" / f"{prefix}_{stage}_train.txt"


def _catalog_path(run_dir: Path, prefix: str) -> Path:
    candidates = [
        run_dir / "dataset" / "tokenizer" / f"{prefix}_render_catalog.json",
        run_dir / "dataset" / "manifests" / "generated" / "structured_atoms" / f"{prefix}_render_catalog.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"render catalog not found for prefix '{prefix}' under {run_dir / 'dataset'}")


def _safe_int(value: int | str | None) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _budget_stage(
    *,
    name: str,
    dataset_path: Path,
    tokenizer_json: Path,
    tokenizer_bin: Path,
    tokenizer_lib: Path,
    seq_len: int,
    target_epochs: float,
    current_total_tokens: int | None,
) -> dict[str, Any]:
    rows = _read_rows(dataset_path)
    if not rows:
        raise SystemExit(f"no rows found in dataset: {dataset_path}")

    bos_id, eos_id, pad_id = _resolve_special_ids(
        tokenizer_json=tokenizer_json,
        bos_override=None,
        eos_override=None,
        pad_override=None,
    )

    dropped_rows: list[dict[str, Any]] = []
    payloads: list[list[int]] = []
    row_lengths: list[int] = []
    with _TrueBPEHandle(tokenizer_lib, tokenizer_bin, tokenizer_json) as handle:
        for idx, row in enumerate(rows, start=1):
            ids = _encode_row_with_fallback(handle, row)
            full = [int(bos_id), *[int(v) for v in ids], int(eos_id)]
            row_lengths.append(int(len(full)))
            if len(full) > int(seq_len):
                if len(dropped_rows) < 16:
                    dropped_rows.append(
                        {
                            "row_index": int(idx),
                            "token_count": int(len(full)),
                            "preview": row[:160],
                        }
                    )
                continue
            payloads.append(full)

    if not payloads:
        raise SystemExit(f"all rows exceeded seq_len={seq_len}: {dataset_path}")

    _, stats = _pack_rows(payloads, int(seq_len), int(pad_id))
    one_epoch_tokens = int(stats.get("recommended_total_tokens", 0) or 0)
    recommended_total_tokens = max(int(seq_len), int(round(one_epoch_tokens * float(target_epochs))))
    current_budget = _safe_int(current_total_tokens)
    effective_epochs_at_current = (
        float(current_budget) / float(max(1, one_epoch_tokens))
        if current_budget is not None
        else None
    )
    budget_pass = current_budget is None or current_budget >= recommended_total_tokens

    return {
        "stage": name,
        "dataset": str(dataset_path),
        "rows_total": int(len(rows)),
        "rows_kept": int(len(payloads)),
        "rows_dropped_oversize": int(len(dropped_rows)),
        "dropped_examples": dropped_rows,
        "row_tokens_min": int(min(row_lengths) if row_lengths else 0),
        "row_tokens_max": int(max(row_lengths) if row_lengths else 0),
        "pack_stats": stats,
        "target_effective_epochs": float(target_epochs),
        "one_epoch_total_tokens": int(one_epoch_tokens),
        "recommended_total_tokens": int(recommended_total_tokens),
        "current_total_tokens": int(current_budget) if current_budget is not None else None,
        "effective_epochs_at_current_budget": effective_epochs_at_current,
        "budget_gate_pass": bool(budget_pass),
    }


def _validate_scene_output(
    output_tokens: str,
    expected_svg: str | None,
    prompt: str,
) -> dict[str, Any]:
    tokens = [tok.strip() for tok in str(output_tokens or "").split() if tok.strip()]
    starts_clean = bool(tokens) and tokens[0] == "[scene]"
    ends_clean = bool(tokens) and tokens[-1] == "[/scene]"
    try:
        scene_doc = _parse_scene_document(output_tokens)
        parse_ok = True
        parse_error = None
    except Exception as exc:
        scene_doc = None
        parse_ok = False
        parse_error = str(exc)

    try:
        rendered_svg = render_structured_scene_svg(output_tokens)
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
        "topic": prompt_tags.get("topic"),
        "starts_clean": bool(starts_clean),
        "ends_clean": bool(ends_clean),
        "parse_ok": bool(parse_ok),
        "parse_error": parse_error,
        "renderable": bool(renderable),
        "render_error": render_error,
        "svg_exact": bool(svg_exact),
        "scene_doc": scene_doc,
    }


def _catalog_validation(run_dir: Path, prefix: str) -> dict[str, Any]:
    catalog_rows = _load_json(_catalog_path(run_dir, prefix))
    if not isinstance(catalog_rows, list):
        raise SystemExit("spec07 render catalog must be a JSON list")

    failures: list[dict[str, Any]] = []
    per_layout: dict[str, dict[str, int]] = {}
    ok = 0
    for row in catalog_rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "")
        output_tokens = str(row.get("output_tokens") or "")
        expected_svg = str(row.get("svg_xml") or "")
        result = _validate_scene_output(output_tokens, expected_svg, prompt)
        layout = str(result.get("layout") or "unknown")
        stats = per_layout.setdefault(
            layout,
            {
                "count": 0,
                "starts_clean": 0,
                "ends_clean": 0,
                "parse_ok": 0,
                "renderable": 0,
                "svg_exact": 0,
            },
        )
        stats["count"] += 1
        for key in ("starts_clean", "ends_clean", "parse_ok", "renderable", "svg_exact"):
            if bool(result.get(key)):
                stats[key] += 1
        passed = all(
            bool(result.get(key))
            for key in ("starts_clean", "ends_clean", "parse_ok", "renderable", "svg_exact")
        )
        if passed:
            ok += 1
        elif len(failures) < 24:
            failures.append(
                {
                    "prompt": prompt,
                    "layout": result.get("layout"),
                    "topic": result.get("topic"),
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
    contract = build_probe_contract(run_dir, prefix, max(1, int(per_split)))
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    ok = 0
    total = 0
    for split in contract.get("splits") or []:
        split_name = str(split.get("name") or "")
        for case in split.get("cases") or []:
            if not isinstance(case, dict):
                continue
            total += 1
            prompt = str(case.get("prompt") or "")
            expected_output = str(case.get("expected_output") or "")
            expected_svg = str(case.get("expected_rendered_output") or "")
            result = _validate_scene_output(expected_output, expected_svg, prompt)
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
            passed = all(
                bool(result.get(key))
                for key in ("starts_clean", "ends_clean", "parse_ok", "renderable", "svg_exact")
            )
            if passed:
                ok += 1
            elif len(failures) < 16:
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


def _status(report: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for stage_name in ("pretrain", "midtrain"):
        stage = report["stages"][stage_name]
        if int(stage.get("rows_dropped_oversize", 0)) > 0:
            reasons.append(f"{stage_name}:oversize_rows")
        if not bool(stage.get("budget_gate_pass")):
            reasons.append(f"{stage_name}:undersized_budget")
    if not bool(report["catalog"].get("pass")):
        reasons.append("catalog:scene_or_render_failures")
    if not bool(report["canary"].get("pass")):
        reasons.append("canary:scene_or_render_failures")
    return (len(reasons) == 0, reasons)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute spec07 training budgets and run scene-DSL preflight gates")
    ap.add_argument("--run", required=True, type=Path, help="Run directory containing the staged dataset")
    ap.add_argument("--prefix", default="spec07_scene_dsl", help="Dataset prefix")
    ap.add_argument("--tokenizer-json", required=True, type=Path, help="Path to tokenizer.json")
    ap.add_argument("--tokenizer-bin", required=True, type=Path, help="Path to tokenizer_bin directory")
    ap.add_argument("--tokenizer-lib", default=str(DEFAULT_TOKENIZER_LIB), type=Path, help="Path to libckernel_tokenizer.so")
    ap.add_argument("--seq-len", type=int, required=True, help="Context length used for sample packing")
    ap.add_argument("--pretrain-epochs", type=float, default=1.0, help="Target effective epochs for pretrain")
    ap.add_argument("--midtrain-epochs", type=float, default=1.0, help="Target effective epochs for midtrain")
    ap.add_argument("--current-pretrain-total-tokens", type=int, default=0, help="Optional currently planned pretrain token budget")
    ap.add_argument("--current-midtrain-total-tokens", type=int, default=0, help="Optional currently planned midtrain token budget")
    ap.add_argument("--canary-per-split", type=int, default=4, help="Balanced canary cases per split")
    ap.add_argument("--json-out", required=True, type=Path, help="Output report JSON")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if any gate fails")
    args = ap.parse_args()

    run_dir = args.run.expanduser().resolve()
    tokenizer_json = args.tokenizer_json.expanduser().resolve()
    tokenizer_bin = args.tokenizer_bin.expanduser().resolve()
    tokenizer_lib = args.tokenizer_lib.expanduser().resolve()
    out_path = args.json_out.expanduser().resolve()

    if int(args.seq_len) < 1:
        raise SystemExit("--seq-len must be >= 1")
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")
    if not tokenizer_json.exists():
        raise SystemExit(f"tokenizer.json not found: {tokenizer_json}")
    if not tokenizer_bin.is_dir():
        raise SystemExit(f"tokenizer_bin directory not found: {tokenizer_bin}")
    if not tokenizer_lib.exists():
        raise SystemExit(f"tokenizer library not found: {tokenizer_lib}")

    prefix = str(args.prefix).strip()
    report = {
        "schema": "ck.spec07_preflight.v1",
        "run_dir": str(run_dir),
        "prefix": prefix,
        "seq_len": int(args.seq_len),
        "stages": {
            "pretrain": _budget_stage(
                name="pretrain",
                dataset_path=_stage_dataset_path(run_dir, prefix, "pretrain"),
                tokenizer_json=tokenizer_json,
                tokenizer_bin=tokenizer_bin,
                tokenizer_lib=tokenizer_lib,
                seq_len=int(args.seq_len),
                target_epochs=float(args.pretrain_epochs),
                current_total_tokens=_safe_int(args.current_pretrain_total_tokens),
            ),
            "midtrain": _budget_stage(
                name="midtrain",
                dataset_path=_stage_dataset_path(run_dir, prefix, "midtrain"),
                tokenizer_json=tokenizer_json,
                tokenizer_bin=tokenizer_bin,
                tokenizer_lib=tokenizer_lib,
                seq_len=int(args.seq_len),
                target_epochs=float(args.midtrain_epochs),
                current_total_tokens=_safe_int(args.current_midtrain_total_tokens),
            ),
        },
        "catalog": _catalog_validation(run_dir, prefix),
        "canary": _canary_validation(run_dir, prefix, int(args.canary_per_split)),
    }
    passed, reasons = _status(report)
    report["status"] = {"pass": bool(passed), "reasons": reasons}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    for stage_name in ("pretrain", "midtrain"):
        stage = report["stages"][stage_name]
        current_budget = stage.get("current_total_tokens")
        eff = stage.get("effective_epochs_at_current_budget")
        eff_txt = f"{float(eff):.3f}" if isinstance(eff, (int, float)) else "-"
        print(
            "[spec07-preflight] "
            f"{stage_name} rows={stage['rows_total']} dropped={stage['rows_dropped_oversize']} "
            f"one_epoch_total_tokens={stage['one_epoch_total_tokens']} "
            f"recommended_total_tokens={stage['recommended_total_tokens']} "
            f"current_total_tokens={current_budget if current_budget is not None else '-'} "
            f"effective_epochs_at_current={eff_txt}"
        )
    print(
        "[spec07-preflight] "
        f"catalog pass={report['catalog']['pass']} "
        f"rows={report['catalog']['count']} "
        f"pass_rate={report['catalog']['pass_rate']:.4f}"
    )
    print(
        "[spec07-preflight] "
        f"canary pass={report['canary']['pass']} "
        f"cases={report['canary']['count']} "
        f"pass_rate={report['canary']['pass_rate']:.4f}"
    )

    if args.strict and not passed:
        raise SystemExit("spec07 preflight failed: " + ", ".join(reasons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
