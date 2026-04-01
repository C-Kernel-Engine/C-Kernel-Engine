#!/usr/bin/env python3
"""Replay stored probe outputs with repair disabled and enabled."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from build_probe_report_v7 import _build_summary, _evaluate_case
except ModuleNotFoundError:  # pragma: no cover
    from version.v7.scripts.build_probe_report_v7 import _build_summary, _evaluate_case


SCHEMA = "ck.probe_repair_eval.v1"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _adapter_cfg_from_report(report_doc: dict[str, Any], *, enable_repair: bool) -> dict[str, Any]:
    cfg = report_doc.get("output_adapter_config")
    if not isinstance(cfg, dict):
        cfg = {"name": str(report_doc.get("output_adapter") or "plain_text").strip() or "plain_text"}
    normalized = dict(cfg)
    normalized["name"] = str(normalized.get("name") or report_doc.get("output_adapter") or "plain_text").strip() or "plain_text"
    if enable_repair:
        repairer = str(normalized.get("repairer") or "").strip()
        if repairer:
            normalized["repairer"] = repairer
    else:
        normalized.pop("repairer", None)
    return normalized


def _row_to_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "split": row.get("split"),
        "split_label": row.get("split_label"),
        "label": row.get("label"),
        "prompt": row.get("prompt"),
        "expected_output": row.get("expected_output"),
        "expected_rendered_output": row.get("expected_rendered_output"),
        "expected_rendered_mime": row.get("expected_rendered_mime"),
        "content_json": row.get("content_json") if isinstance(row.get("content_json"), dict) else None,
    }


def _replay_results(report_doc: dict[str, Any], *, enable_repair: bool) -> dict[str, Any]:
    adapter_cfg = _adapter_cfg_from_report(report_doc, enable_repair=enable_repair)
    replayed: list[dict[str, Any]] = []
    for row in report_doc.get("results") or []:
        if not isinstance(row, dict):
            continue
        raw_output = str(row.get("raw_output") or row.get("response_text") or "").strip()
        response_text = str(row.get("response_text") or raw_output).strip()
        replayed.append(
            _evaluate_case(
                _row_to_case(row),
                raw_output,
                response_text,
                adapter_cfg,
                budget_analyzer=None,
            )
        )
    split_summary, totals = _build_summary(replayed)
    return {
        "output_adapter_config": adapter_cfg,
        "split_summary": split_summary,
        "totals": totals,
        "results": replayed,
    }


def _delta_rows(raw_results: list[dict[str, Any]], repaired_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for raw_row, repaired_row in zip(raw_results, repaired_results):
        delta = {
            "id": repaired_row.get("id"),
            "split": repaired_row.get("split"),
            "label": repaired_row.get("label"),
            "prompt": repaired_row.get("prompt"),
            "raw_exact_match": bool(raw_row.get("exact_match")),
            "repaired_exact_match": bool(repaired_row.get("exact_match")),
            "raw_renderable": bool(raw_row.get("renderable")),
            "repaired_renderable": bool(repaired_row.get("renderable")),
            "repair_applied": bool(repaired_row.get("repair_applied")),
            "repair_note": repaired_row.get("repair_note"),
            "raw_render_error": raw_row.get("render_error"),
            "repaired_render_error": repaired_row.get("render_error"),
            "raw_parsed_output": raw_row.get("parsed_output"),
            "repaired_parsed_output": repaired_row.get("parsed_output"),
        }
        delta["exact_improved"] = (not delta["raw_exact_match"]) and delta["repaired_exact_match"]
        delta["renderable_improved"] = (not delta["raw_renderable"]) and delta["repaired_renderable"]
        delta["exact_regressed"] = delta["raw_exact_match"] and (not delta["repaired_exact_match"])
        delta["renderable_regressed"] = delta["raw_renderable"] and (not delta["repaired_renderable"])
        deltas.append(delta)
    return deltas


def _comparison_summary(raw_doc: dict[str, Any], repaired_doc: dict[str, Any], deltas: list[dict[str, Any]]) -> dict[str, Any]:
    exact_improved = [row for row in deltas if row.get("exact_improved")]
    renderable_improved = [row for row in deltas if row.get("renderable_improved")]
    exact_regressed = [row for row in deltas if row.get("exact_regressed")]
    renderable_regressed = [row for row in deltas if row.get("renderable_regressed")]
    repaired_rows = [row for row in deltas if row.get("repair_applied")]
    return {
        "case_count": len(deltas),
        "raw_exact_rate": raw_doc.get("totals", {}).get("exact_rate"),
        "repaired_exact_rate": repaired_doc.get("totals", {}).get("exact_rate"),
        "raw_renderable_rate": raw_doc.get("totals", {}).get("renderable_rate"),
        "repaired_renderable_rate": repaired_doc.get("totals", {}).get("renderable_rate"),
        "exact_improved_count": len(exact_improved),
        "renderable_improved_count": len(renderable_improved),
        "exact_regressed_count": len(exact_regressed),
        "renderable_regressed_count": len(renderable_regressed),
        "repair_applied_count": len(repaired_rows),
        "exact_improved_ids": [row.get("id") for row in exact_improved],
        "renderable_improved_ids": [row.get("id") for row in renderable_improved],
        "exact_regressed_ids": [row.get("id") for row in exact_regressed],
        "renderable_regressed_ids": [row.get("id") for row in renderable_regressed],
    }


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _build_markdown(payload: dict[str, Any]) -> str:
    source_report = payload.get("source_report")
    summary = payload.get("summary") or {}
    lines = [
        "# Probe Repair Eval",
        "",
        f"- source report: `{source_report}`",
        f"- generated at: `{payload.get('generated_at')}`",
        f"- case count: `{summary.get('case_count')}`",
        "",
        "## Totals",
        "",
        f"- raw exact: `{_pct(summary.get('raw_exact_rate'))}`",
        f"- repaired exact: `{_pct(summary.get('repaired_exact_rate'))}`",
        f"- raw renderable: `{_pct(summary.get('raw_renderable_rate'))}`",
        f"- repaired renderable: `{_pct(summary.get('repaired_renderable_rate'))}`",
        f"- repair applied count: `{summary.get('repair_applied_count')}`",
        f"- exact improved count: `{summary.get('exact_improved_count')}`",
        f"- renderable improved count: `{summary.get('renderable_improved_count')}`",
        f"- exact regressed count: `{summary.get('exact_regressed_count')}`",
        f"- renderable regressed count: `{summary.get('renderable_regressed_count')}`",
        "",
        "## Exact Improvements",
        "",
    ]
    improved_ids = summary.get("exact_improved_ids") or []
    if improved_ids:
        lines.extend(f"- `{item}`" for item in improved_ids)
    else:
        lines.append("- none")
    lines.extend(["", "## Exact Regressions", ""])
    regressed_ids = summary.get("exact_regressed_ids") or []
    if regressed_ids:
        lines.extend(f"- `{item}`" for item in regressed_ids)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def build_repair_eval(report_path: Path) -> dict[str, Any]:
    report_doc = _load_json(report_path)
    raw_doc = _replay_results(report_doc, enable_repair=False)
    repaired_doc = _replay_results(report_doc, enable_repair=True)
    deltas = _delta_rows(raw_doc.get("results") or [], repaired_doc.get("results") or [])
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_path),
        "source_run_name": report_doc.get("run_name"),
        "raw": raw_doc,
        "repaired": repaired_doc,
        "summary": _comparison_summary(raw_doc, repaired_doc, deltas),
        "deltas": deltas,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a probe report with repair disabled and enabled.")
    ap.add_argument("--report", required=True, help="Path to an existing probe report JSON.")
    ap.add_argument("--out-json", default=None, help="Optional output JSON path.")
    ap.add_argument("--out-md", default=None, help="Optional output markdown path.")
    args = ap.parse_args()

    report_path = Path(args.report).expanduser().resolve()
    payload = build_repair_eval(report_path)

    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else report_path.with_name(report_path.stem + "_repair_eval.json")
    out_md = Path(args.out_md).expanduser().resolve() if args.out_md else report_path.with_name(report_path.stem + "_repair_eval.md")

    _write_json(out_json, payload)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_build_markdown(payload), encoding="utf-8")

    summary = payload.get("summary") or {}
    print(
        "[OK] "
        f"raw exact={_pct(summary.get('raw_exact_rate'))} "
        f"repaired exact={_pct(summary.get('repaired_exact_rate'))} "
        f"repair_applied={summary.get('repair_applied_count')} "
        f"exact_improved={summary.get('exact_improved_count')} "
        f"exact_regressed={summary.get('exact_regressed_count')}"
    )
    print(f"[OK] wrote: {out_json}")
    print(f"[OK] wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
