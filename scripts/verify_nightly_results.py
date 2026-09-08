#!/usr/bin/env python3
"""Fail the CI verdict after nightly evidence has been published."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator


VALID_STATUSES = {"pass", "fail", "skip", "timeout"}
CAPABILITY_STATUSES = {"pass", "fail", "error", "timeout", "not_tested"}
ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_MANIFEST = ROOT / "version" / "v8" / "testing" / "capability_cases.json"
CAPABILITY_SCHEMA = ROOT / "version" / "v8" / "schemas" / "capability_evidence_report.schema.json"


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _capability_event(event: object) -> str:
    value = str(event or "").lower()
    if value == "pull_request":
        return "pull_request"
    if value in {"push", "schedule", "workflow_dispatch"}:
        return "nightly"
    return ""


def _verify_capability_evidence(payload: dict) -> list[str]:
    report = payload.get("capability_evidence")
    if not isinstance(report, dict):
        return ["current-run capability evidence is missing"]

    errors: list[str] = []
    try:
        schema = json.loads(CAPABILITY_SCHEMA.read_text(encoding="utf-8"))
        manifest = json.loads(CAPABILITY_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot load capability contracts: {exc}"]
    schema_errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda error: list(error.absolute_path),
    )
    errors.extend(
        "capability schema:"
        f"{'.'.join(str(item) for item in error.absolute_path) or '<root>'}: "
        f"{error.message}"
        for error in schema_errors
    )
    expected_cases = {case["id"]: case for case in manifest["cases"]}
    if report.get("schema") != "cke.v8.capability_evidence_report":
        errors.append("capability evidence has an invalid schema identity")
    if report.get("schema_version") != 1 or report.get("scope") != "current_run":
        errors.append("capability evidence is not a v1 current-run report")
    source = report.get("source")
    if not isinstance(source, dict):
        errors.append("capability evidence has no source provenance")
        source = {}
    else:
        if source.get("event") != payload.get("event", "local"):
            errors.append("capability evidence event does not match the nightly report")
        commit = _git_commit()
        if commit and source.get("repository_commit") != commit:
            errors.append("capability evidence does not describe the checked-out commit")
        manifest_hash = hashlib.sha256(CAPABILITY_MANIFEST.read_bytes()).hexdigest()
        if source.get("manifest_sha256") != manifest_hash:
            errors.append("capability evidence uses a stale capability manifest")

    report_errors = report.get("errors")
    if not isinstance(report_errors, list):
        errors.append("capability evidence errors field is malformed")
    else:
        errors.extend(f"capability evidence: {message}" for message in report_errors)

    cases = report.get("cases")
    summary = report.get("summary")
    if not isinstance(cases, list):
        errors.append("capability evidence has no case rows")
        cases = []
    if not isinstance(summary, dict):
        errors.append("capability evidence has no summary")
        summary = {}

    counts = {status: 0 for status in CAPABILITY_STATUSES}
    seen: set[str] = set()
    required_event = _capability_event(payload.get("event"))
    for index, row in enumerate(cases):
        if not isinstance(row, dict):
            errors.append(f"capability row {index} is not an object")
            continue
        case_id = str(row.get("id") or "").strip()
        status = str(row.get("status") or "").lower()
        if not case_id:
            errors.append(f"capability row {index} has no ID")
        elif case_id in seen:
            errors.append(f"duplicate capability row: {case_id}")
        seen.add(case_id)
        if status not in CAPABILITY_STATUSES:
            errors.append(f"{case_id or f'capability row {index}'} has invalid status {status!r}")
            continue
        counts[status] += 1
        expected_case = expected_cases.get(case_id)
        if expected_case is None:
            errors.append(f"unregistered capability row: {case_id}")
            events = []
        else:
            for field in (
                "family",
                "circuits",
                "evidence_level",
                "phases",
                "oracle",
                "artifact",
                "schedule",
            ):
                if row.get(field) != expected_case.get(field):
                    errors.append(f"{case_id}: evidence {field} differs from manifest")
            events = expected_case["schedule"]["events"]
        if required_event and required_event in events and status != "pass":
            errors.append(
                f"required capability {case_id}: {status}"
                + (f" ({row.get('reason')})" if row.get("reason") else "")
            )

    expected = {
        "total": len(cases),
        "passed": counts["pass"],
        "failed": counts["fail"],
        "errors": counts["error"],
        "timeouts": counts["timeout"],
        "not_tested": counts["not_tested"],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(
                f"capability summary {key} mismatch: "
                f"reported={summary.get(key)!r} actual={value}"
            )
    missing = sorted(set(expected_cases) - seen)
    if missing:
        errors.append("capability evidence is missing cases: " + ", ".join(missing))
    return errors


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
    errors.extend(_verify_capability_evidence(payload))
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
