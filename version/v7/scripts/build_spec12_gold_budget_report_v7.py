#!/usr/bin/env python3
"""Estimate token/context budget for spec12 gold scene mappings."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from pack_training_tokens_v7 import _TrueBPEHandle


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOKENIZER_LIB = ROOT / "build" / "libckernel_tokenizer.so"
DEFAULT_MANIFEST = ROOT / "version" / "v7" / "reports" / "spec12_gold_mappings" / "spec12_gold_mappings_20260318.json"
DEFAULT_SPEC11_R2 = Path.home() / ".cache" / "ck-engine-v7" / "models" / "train" / "spec11_keyed_scene_dsl_l3_d192_h384_ctx512_r2"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_tokenizer_paths(base_run: Path) -> tuple[Path, Path]:
    tokenizer_json_candidates = [
        base_run / "tokenizer.json",
        base_run / "dataset" / "tokenizer" / "tokenizer.json",
    ]
    tokenizer_bin_candidates = [
        base_run / "tokenizer_bin",
        base_run / "dataset" / "tokenizer" / "tokenizer_bin",
    ]
    tokenizer_json = next((p for p in tokenizer_json_candidates if p.exists()), None)
    tokenizer_bin = next((p for p in tokenizer_bin_candidates if p.exists()), None)
    if tokenizer_json is None or tokenizer_bin is None:
        raise SystemExit(f"Could not find tokenizer artifacts under {base_run}")
    return tokenizer_json, tokenizer_bin


def _compact_scene_text(path: Path) -> str:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return " ".join(lines)


def _context_row(*, context_len: int, prompt_tokens: int, output_tokens: int) -> dict[str, Any]:
    total = int(prompt_tokens) + int(output_tokens)
    margin = int(context_len) - total
    return {
        "context_len": int(context_len),
        "fits": total <= int(context_len),
        "prompt_tokens": int(prompt_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total),
        "margin_tokens": int(margin),
    }


def _recommend_context(max_total_tokens: int) -> int:
    if max_total_tokens <= 512:
        return 512
    if max_total_tokens <= 768:
        return 768
    return 1024


def build_report(
    *,
    manifest_path: Path,
    tokenizer_lib: Path,
    tokenizer_json: Path,
    tokenizer_bin: Path,
    prompt_tokens: int,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    mappings = manifest.get("mappings") if isinstance(manifest, dict) else None
    if not isinstance(mappings, list) or not mappings:
        raise SystemExit(f"No mappings found in {manifest_path}")

    results: list[dict[str, Any]] = []
    with _TrueBPEHandle(tokenizer_lib, tokenizer_bin, tokenizer_json) as handle:
        for row in mappings:
            if not isinstance(row, dict):
                continue
            scene_path = ROOT / str(row.get("scene_dsl") or "")
            if not scene_path.exists():
                raise SystemExit(f"Missing scene DSL: {scene_path}")
            compact = _compact_scene_text(scene_path)
            output_tokens = len(handle.encode(compact))
            context_rows = [
                _context_row(context_len=512, prompt_tokens=prompt_tokens, output_tokens=output_tokens),
                _context_row(context_len=768, prompt_tokens=prompt_tokens, output_tokens=output_tokens),
                _context_row(context_len=1024, prompt_tokens=prompt_tokens, output_tokens=output_tokens),
            ]
            results.append(
                {
                    "asset": str(row.get("asset") or ""),
                    "family": str(row.get("family") or ""),
                    "scene_dsl": str(row.get("scene_dsl") or ""),
                    "content_json": str(row.get("content_json") or ""),
                    "output_tokens": int(output_tokens),
                    "assumed_prompt_tokens": int(prompt_tokens),
                    "contexts": context_rows,
                }
            )

    max_output_tokens = max(int(row["output_tokens"]) for row in results)
    max_total_tokens = max(int(prompt_tokens) + int(row["output_tokens"]) for row in results)
    recommended_context = _recommend_context(max_total_tokens)
    tokenizer_rework_required = max_total_tokens > 1024
    return {
        "schema": "ck.spec12_gold_budget_report.v1",
        "generated_on": str(date.today()),
        "manifest": str(manifest_path),
        "tokenizer_json": str(tokenizer_json),
        "tokenizer_bin": str(tokenizer_bin),
        "assumed_prompt_tokens": int(prompt_tokens),
        "max_output_tokens": int(max_output_tokens),
        "max_total_tokens": int(max_total_tokens),
        "recommended_context_len": int(recommended_context),
        "tokenizer_rework_required": bool(tokenizer_rework_required),
        "results": results,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Spec12 Gold Token Budget Report")
    lines.append("")
    lines.append(f"- Generated on: `{report['generated_on']}`")
    lines.append(f"- Manifest: `{report['manifest']}`")
    lines.append(f"- Assumed prompt tokens: `{report['assumed_prompt_tokens']}`")
    lines.append(f"- Max output tokens: `{report['max_output_tokens']}`")
    lines.append(f"- Max total tokens: `{report['max_total_tokens']}`")
    lines.append(f"- Recommended context length: `{report['recommended_context_len']}`")
    lines.append(f"- Tokenizer rework required before training: `{str(bool(report.get('tokenizer_rework_required'))).lower()}`")
    lines.append("")
    lines.append("| Asset | Family | Output Tokens | Total @512 | Total @768 | Total @1024 |")
    lines.append("| --- | --- | ---: | --- | --- | --- |")
    for row in report["results"]:
        ctx512, ctx768, ctx1024 = row["contexts"]
        def _cell(ctx: dict[str, Any]) -> str:
            status = "fits" if ctx["fits"] else "overflow"
            margin = ctx["margin_tokens"]
            return f"{ctx['total_tokens']} ({status}, {margin:+d})"
        lines.append(
            f"| `{row['asset'].split('/')[-1]}` | `{row['family']}` | `{row['output_tokens']}` | "
            f"{_cell(ctx512)} | {_cell(ctx768)} | {_cell(ctx1024)} |"
        )
    lines.append("")
    lines.append("## Read")
    lines.append("")
    lines.append(f"- These counts were measured against the current tokenizer at `{report['tokenizer_json']}`. They are a readiness diagnostic, not a final spec12 token count.")
    if bool(report.get("tokenizer_rework_required")):
        lines.append("- Context alone is not the fix. The current tokenizer surface does not yet compress the spec12 scene language enough for training.")
        lines.append("- Next step: define the spec12 tokenizer surface after compiler parity on the gold mappings, then re-measure before choosing `ctx`.")
    elif int(report["recommended_context_len"]) <= 512:
        lines.append("- The current gold pack still fits inside `ctx=512` with the assumed prompt budget.")
    elif int(report["recommended_context_len"]) == 768:
        lines.append("- `ctx=512` is too tight for the current gold pack. `ctx=768` is the minimum sensible next step.")
    else:
        lines.append("- The current gold pack pressures beyond `ctx=768`. `ctx=1024` is the safer target.")
    lines.append("- Do not launch training from this alone. Use this together with compiler parity on the same gold mappings.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--tokenizer-lib", type=Path, default=DEFAULT_TOKENIZER_LIB)
    ap.add_argument("--tokenizer-json", type=Path, default=None)
    ap.add_argument("--tokenizer-bin", type=Path, default=None)
    ap.add_argument("--base-run", type=Path, default=DEFAULT_SPEC11_R2)
    ap.add_argument("--prompt-tokens", type=int, default=16)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--md-out", type=Path, default=None)
    args = ap.parse_args()

    tokenizer_json = args.tokenizer_json
    tokenizer_bin = args.tokenizer_bin
    if tokenizer_json is None or tokenizer_bin is None:
        tokenizer_json, tokenizer_bin = _resolve_tokenizer_paths(args.base_run)

    report = build_report(
        manifest_path=args.manifest,
        tokenizer_lib=args.tokenizer_lib,
        tokenizer_json=tokenizer_json,
        tokenizer_bin=tokenizer_bin,
        prompt_tokens=args.prompt_tokens,
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_render_markdown(report), encoding="utf-8")

    if not args.json_out and not args.md_out:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
