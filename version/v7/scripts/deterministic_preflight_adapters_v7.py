#!/usr/bin/env python3
"""Reusable contracts for adapter-based deterministic preflight gates.

These helpers are meant for tasks where the model emits a structured surface
form and a deterministic system can parse, compile, render, execute, or test
that output before real training or promotion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from pack_training_tokens_v7 import (
    _TrueBPEHandle,
    _encode_row_with_fallback,
    _pack_rows,
    _read_rows,
    _resolve_special_ids,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOKENIZER_LIB = ROOT / "build" / "libckernel_tokenizer.so"


def safe_int(value: int | str | None) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    try:
        return int(value)
    except Exception:
        return None


@dataclass(frozen=True)
class DeterministicPreflightCase:
    split: str
    prompt: str
    expected_output: str
    label: str | None = None
    expected_materialized: str | None = None
    expected_materialized_mime: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeterministicPreflightValidation:
    group: str
    identity: dict[str, Any]
    starts_clean: bool
    ends_clean: bool
    parse_ok: bool
    parse_error: str | None
    materialized_ok: bool
    materialized_error: str | None
    oracle_exact: bool

    def passed(self) -> bool:
        return (
            bool(self.starts_clean)
            and bool(self.ends_clean)
            and bool(self.parse_ok)
            and bool(self.materialized_ok)
            and bool(self.oracle_exact)
        )


class DeterministicPreflightAdapter(Protocol):
    name: str
    group_name: str

    def load_catalog_cases(self, run_dir: Path, prefix: str) -> list[DeterministicPreflightCase]:
        """Return all deterministic catalog rows used for full-catalog hygiene checks."""

    def build_canary_cases(
        self,
        run_dir: Path,
        prefix: str,
        per_split: int,
    ) -> list[DeterministicPreflightCase]:
        """Return a small balanced canary set for quick preflight validation."""

    def validate_case(self, case: DeterministicPreflightCase) -> DeterministicPreflightValidation:
        """Parse and materialize one deterministic case."""


def stage_dataset_path(run_dir: Path, prefix: str, stage: str) -> Path:
    return run_dir / "dataset" / stage / "train" / f"{prefix}_{stage}_train.txt"


def budget_stage(
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
    current_budget = safe_int(current_total_tokens)
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


def validate_case_collection(
    *,
    adapter: DeterministicPreflightAdapter,
    cases: Sequence[DeterministicPreflightCase],
    include_cases: bool = False,
    failure_limit: int = 24,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    group_summary: dict[str, dict[str, int]] = {}
    ok = 0
    total = 0

    for case in cases:
        total += 1
        result = adapter.validate_case(case)
        group = str(result.group or "unknown")
        stats = group_summary.setdefault(
            group,
            {
                "count": 0,
                "starts_clean": 0,
                "ends_clean": 0,
                "parse_ok": 0,
                "materialized_ok": 0,
                "oracle_exact": 0,
            },
        )
        stats["count"] += 1
        for key in ("starts_clean", "ends_clean", "parse_ok", "materialized_ok", "oracle_exact"):
            if bool(getattr(result, key)):
                stats[key] += 1

        record = {
            "split": case.split,
            "label": case.label,
            "prompt": case.prompt,
            adapter.group_name: group,
            **dict(result.identity or {}),
            "starts_clean": bool(result.starts_clean),
            "ends_clean": bool(result.ends_clean),
            "parse_ok": bool(result.parse_ok),
            "materialized_ok": bool(result.materialized_ok),
            "oracle_exact": bool(result.oracle_exact),
        }
        if include_cases:
            records.append(record)

        if result.passed():
            ok += 1
            continue
        if len(failures) < int(max(1, failure_limit)):
            failures.append(
                {
                    **record,
                    "parse_error": result.parse_error,
                    "materialized_error": result.materialized_error,
                }
            )

    report = {
        "count": int(total),
        "pass_count": int(ok),
        "pass_rate": float(ok) / float(max(1, total)),
        "group_name": adapter.group_name,
        "group_summary": group_summary,
        "failures": failures,
        "pass": int(ok) == int(total),
    }
    if include_cases:
        report["cases"] = records
    return report


def summarize_status(
    report: dict[str, Any],
    *,
    stage_names: Sequence[str] = ("pretrain", "midtrain"),
    catalog_key: str = "catalog",
    canary_key: str = "canary",
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for stage_name in stage_names:
        stage = report["stages"][stage_name]
        if int(stage.get("rows_dropped_oversize", 0)) > 0:
            reasons.append(f"{stage_name}:oversize_rows")
        if not bool(stage.get("budget_gate_pass")):
            reasons.append(f"{stage_name}:undersized_budget")
    if not bool(report[catalog_key].get("pass")):
        reasons.append(f"{catalog_key}:validation_failures")
    if not bool(report[canary_key].get("pass")):
        reasons.append(f"{canary_key}:validation_failures")
    return (len(reasons) == 0, reasons)
