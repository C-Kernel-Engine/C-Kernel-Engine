#!/usr/bin/env python3
"""Fail the CI verdict after nightly evidence has been published."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


VALID_STATUSES = {"pass", "fail", "skip", "timeout"}


def _parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def verify_report(
    path: Path,
    *,
    not_before_epoch: float = 0.0,
    fast_regression_required: bool = False,
) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"missing nightly report: {path}"]
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read nightly report {path}: {exc}"]

    if not isinstance(payload, dict):
        return ["nightly report root must be an object"]
    try:
        timestamp = datetime.fromisoformat(str(payload["timestamp"]).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if timestamp.timestamp() + 1 < not_before_epoch:
            errors.append("nightly report predates this workflow execution")
    except (KeyError, TypeError, ValueError):
        errors.append("nightly report has no valid timestamp")

    results = payload.get("results")
    summary = payload.get("summary")
    if not isinstance(results, list) or not results:
        errors.append("nightly report has no result rows")
        results = []
    if not isinstance(summary, dict):
        errors.append("nightly report has no summary object")
        summary = {}

    counts = {status: 0 for status in VALID_STATUSES}
    seen_names: set[str] = set()
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            errors.append(f"result row {index} is not an object")
            continue
        name = str(row.get("name") or "").strip()
        status = str(row.get("status") or "").lower()
        if not name:
            errors.append(f"result row {index} has no name")
        elif name in seen_names:
            errors.append(f"duplicate result row: {name}")
        seen_names.add(name)
        if status not in VALID_STATUSES:
            errors.append(f"{name or f'row {index}'} has invalid status {status!r}")
            continue
        counts[status] += 1
        if status in {"fail", "timeout"}:
            errors.append(f"{name}: {status}")
        for subtest in row.get("sub_tests") or []:
            if isinstance(subtest, dict) and str(subtest.get("status") or "").lower() == "fail":
                errors.append(f"{name}: failed subtest {subtest.get('name') or '<unnamed>'}")

    expected = {
        "total": len(results),
        "passed": counts["pass"],
        "failed": counts["fail"],
        "skipped": counts["skip"],
        "timeout": counts["timeout"],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(
                f"summary {key} mismatch: reported={summary.get(key)!r} actual={value}"
            )

    if fast_regression_required:
        regression = payload.get("regression_fast")
        if not isinstance(regression, dict):
            errors.append("required fast regression payload is missing")
        else:
            status = str(regression.get("status") or "").lower()
            if status != "pass":
                errors.append(f"required fast regression payload status: {status or '<missing>'}")
            if not regression.get("summary_path"):
                errors.append("required fast regression summary path is missing")
            if not isinstance(regression.get("family_rows"), list) or not regression["family_rows"]:
                errors.append("required fast regression has no family rows")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--not-before-epoch", type=float, default=0.0)
    parser.add_argument("--runner-outcome", default="success")
    parser.add_argument("--fast-regression-required", type=_parse_bool, default=False)
    parser.add_argument("--fast-regression-outcome", default="skipped")
    args = parser.parse_args()

    errors = verify_report(
        args.report,
        not_before_epoch=args.not_before_epoch,
        fast_regression_required=args.fast_regression_required,
    )
    if args.runner_outcome != "success":
        errors.append(f"nightly runner process outcome: {args.runner_outcome}")
    if args.fast_regression_required and args.fast_regression_outcome != "success":
        errors.append(
            f"required fast regression outcome: {args.fast_regression_outcome}"
        )

    if errors:
        print("Nightly required verdict: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Nightly required verdict: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
