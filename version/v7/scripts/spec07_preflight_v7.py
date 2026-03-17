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
from deterministic_preflight_adapters_v7 import (
    DEFAULT_TOKENIZER_LIB,
    DeterministicPreflightAdapter,
    DeterministicPreflightCase,
    DeterministicPreflightValidation,
    budget_stage,
    safe_int,
    stage_dataset_path,
    summarize_status,
    validate_case_collection,
)
from render_svg_structured_scene_v7 import _parse_scene_document, render_structured_scene_svg


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _catalog_path(run_dir: Path, prefix: str) -> Path:
    candidates = [
        run_dir / "dataset" / "tokenizer" / f"{prefix}_render_catalog.json",
        run_dir / "dataset" / "manifests" / "generated" / "structured_atoms" / f"{prefix}_render_catalog.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"render catalog not found for prefix '{prefix}' under {run_dir / 'dataset'}")


class Spec07SceneAdapter(DeterministicPreflightAdapter):
    name = "spec07_scene"
    group_name = "layout"

    def load_catalog_cases(self, run_dir: Path, prefix: str) -> list[DeterministicPreflightCase]:
        catalog_rows = _load_json(_catalog_path(run_dir, prefix))
        if not isinstance(catalog_rows, list):
            raise SystemExit("spec07 render catalog must be a JSON list")
        cases: list[DeterministicPreflightCase] = []
        for row in catalog_rows:
            if not isinstance(row, dict):
                continue
            cases.append(
                DeterministicPreflightCase(
                    split=str(row.get("split") or "catalog"),
                    label=None,
                    prompt=str(row.get("prompt") or ""),
                    expected_output=str(row.get("output_tokens") or ""),
                    expected_materialized=str(row.get("svg_xml") or ""),
                )
            )
        return cases

    def build_canary_cases(
        self,
        run_dir: Path,
        prefix: str,
        per_split: int,
    ) -> list[DeterministicPreflightCase]:
        contract = build_probe_contract(run_dir, prefix, max(1, int(per_split)))
        cases: list[DeterministicPreflightCase] = []
        for split in contract.get("splits") or []:
            split_name = str(split.get("name") or "")
            for case in split.get("cases") or []:
                if not isinstance(case, dict):
                    continue
                cases.append(
                    DeterministicPreflightCase(
                        split=split_name,
                        label=str(case.get("label") or "") or None,
                        prompt=str(case.get("prompt") or ""),
                        expected_output=str(case.get("expected_output") or ""),
                        expected_materialized=str(case.get("expected_rendered_output") or ""),
                        expected_materialized_mime="image/svg+xml",
                    )
                )
        return cases

    def validate_case(self, case: DeterministicPreflightCase) -> DeterministicPreflightValidation:
        output_tokens = str(case.expected_output or "")
        tokens = [tok.strip() for tok in output_tokens.split() if tok.strip()]
        starts_clean = bool(tokens) and tokens[0] == "[scene]"
        ends_clean = bool(tokens) and tokens[-1] == "[/scene]"
        try:
            _parse_scene_document(output_tokens)
            parse_ok = True
            parse_error = None
        except Exception as exc:
            parse_ok = False
            parse_error = str(exc)

        try:
            rendered_svg = render_structured_scene_svg(output_tokens)
            materialized_ok = bool(rendered_svg.strip())
            materialized_error = None
        except Exception as exc:
            rendered_svg = ""
            materialized_ok = False
            materialized_error = str(exc)

        expected_svg_norm = str(case.expected_materialized or "").strip()
        rendered_svg_norm = rendered_svg.strip()
        prompt_tags = _parse_prompt_tags(case.prompt)
        layout = str(prompt_tags.get("layout") or "unknown")
        topic = prompt_tags.get("topic")
        return DeterministicPreflightValidation(
            group=layout,
            identity={"layout": layout, "topic": topic},
            starts_clean=bool(starts_clean),
            ends_clean=bool(ends_clean),
            parse_ok=bool(parse_ok),
            parse_error=parse_error,
            materialized_ok=bool(materialized_ok),
            materialized_error=materialized_error,
            oracle_exact=bool(expected_svg_norm) and expected_svg_norm == rendered_svg_norm,
        )


def _adapt_summary(report: dict[str, Any]) -> dict[str, Any]:
    adapted = dict(report)
    group_summary = dict(adapted.pop("group_summary", {}))
    adapted["layout_summary"] = group_summary
    for failure in adapted.get("failures") or []:
        if "materialized_ok" in failure:
            failure["renderable"] = failure.pop("materialized_ok")
        if "materialized_error" in failure:
            failure["render_error"] = failure.pop("materialized_error")
        if "oracle_exact" in failure:
            failure["svg_exact"] = failure.pop("oracle_exact")
    for case in adapted.get("cases") or []:
        if "materialized_ok" in case:
            case["renderable"] = case.pop("materialized_ok")
        if "oracle_exact" in case:
            case["svg_exact"] = case.pop("oracle_exact")
    return adapted


def _status(report: dict[str, Any]) -> tuple[bool, list[str]]:
    passed, reasons = summarize_status(report)
    translated: list[str] = []
    for reason in reasons:
        if reason == "catalog:validation_failures":
            translated.append("catalog:scene_or_render_failures")
        elif reason == "canary:validation_failures":
            translated.append("canary:scene_or_render_failures")
        else:
            translated.append(reason)
    return passed, translated


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
    adapter = Spec07SceneAdapter()
    report = {
        "schema": "ck.spec07_preflight.v1",
        "run_dir": str(run_dir),
        "prefix": prefix,
        "seq_len": int(args.seq_len),
        "stages": {
            "pretrain": budget_stage(
                name="pretrain",
                dataset_path=stage_dataset_path(run_dir, prefix, "pretrain"),
                tokenizer_json=tokenizer_json,
                tokenizer_bin=tokenizer_bin,
                tokenizer_lib=tokenizer_lib,
                seq_len=int(args.seq_len),
                target_epochs=float(args.pretrain_epochs),
                current_total_tokens=safe_int(args.current_pretrain_total_tokens),
            ),
            "midtrain": budget_stage(
                name="midtrain",
                dataset_path=stage_dataset_path(run_dir, prefix, "midtrain"),
                tokenizer_json=tokenizer_json,
                tokenizer_bin=tokenizer_bin,
                tokenizer_lib=tokenizer_lib,
                seq_len=int(args.seq_len),
                target_epochs=float(args.midtrain_epochs),
                current_total_tokens=safe_int(args.current_midtrain_total_tokens),
            ),
        },
        "catalog": _adapt_summary(
            validate_case_collection(
                adapter=adapter,
                cases=adapter.load_catalog_cases(run_dir, prefix),
                include_cases=False,
                failure_limit=24,
            )
        ),
        "canary": _adapt_summary(
            validate_case_collection(
                adapter=adapter,
                cases=adapter.build_canary_cases(run_dir, prefix, int(args.canary_per_split)),
                include_cases=True,
                failure_limit=16,
            )
        ),
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
