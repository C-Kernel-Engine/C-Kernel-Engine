#!/usr/bin/env python3
"""Map capability registrations onto outcomes from one concrete test run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "version" / "v8" / "testing" / "capability_cases.json"


def _execution_key(entrypoint: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    args = entrypoint.get("args") or entrypoint.get("execution_args") or []
    if not isinstance(args, list):
        args = []
    return (
        str(entrypoint.get("kind") or entrypoint.get("execution_kind") or ""),
        str(entrypoint.get("target") or entrypoint.get("execution_id") or ""),
        tuple(str(arg) for arg in args),
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _case_status(result: dict[str, Any] | None) -> tuple[str, bool, str]:
    if result is None:
        return "not_tested", False, "entry point was not selected for this run"
    status = str(result.get("status") or "").lower()
    if status == "pass":
        return "pass", True, ""
    if status == "fail":
        return "fail", True, str(result.get("error_msg") or "entry point failed")
    if status == "timeout":
        return "timeout", True, str(result.get("error_msg") or "entry point timed out")
    if status == "skip":
        return "not_tested", False, str(result.get("error_msg") or "entry point skipped")
    return "error", False, f"entry point returned invalid status {status!r}"


def build_report(
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    root: Path = ROOT,
    manifest_bytes: bytes | None = None,
    event: str = "",
) -> dict[str, Any]:
    indexed: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    for result in results:
        key = _execution_key(result)
        if key[0] and key[1]:
            indexed[key] = result

    rows: list[dict[str, Any]] = []
    for case in manifest.get("cases", []):
        entrypoint = case["entrypoint"]
        key = _execution_key(entrypoint)
        result = indexed.get(key)
        status, executed, reason = _case_status(result)
        execution = None
        if result is not None:
            execution = {
                "kind": key[0],
                "id": key[1],
                "args": list(key[2]),
                "name": str(result.get("name") or ""),
                "duration_sec": max(0.0, float(result.get("duration_sec") or 0.0)),
                "error": str(result.get("error_msg") or ""),
            }
        rows.append(
            {
                "id": case["id"],
                "family": case["family"],
                "circuits": case["circuits"],
                "evidence_level": case["evidence_level"],
                "phases": case["phases"],
                "oracle": case["oracle"],
                "artifact": case["artifact"],
                "schedule": case["schedule"],
                "status": status,
                "selected": result is not None,
                "executed": executed,
                "reason": reason,
                "execution": execution,
            }
        )

    counts = Counter(row["status"] for row in rows)
    return {
        "schema": "cke.v8.capability_evidence_report",
        "schema_version": 1,
        "scope": "current_run",
        "source": {
            "repository_commit": _git_commit(root),
            "manifest_sha256": hashlib.sha256(
                manifest_bytes
                if manifest_bytes is not None
                else json.dumps(manifest, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "event": event or os.environ.get("GITHUB_EVENT_NAME", "local"),
        },
        "summary": {
            "total": len(rows),
            "passed": counts["pass"],
            "failed": counts["fail"],
            "errors": counts["error"],
            "timeouts": counts["timeout"],
            "not_tested": counts["not_tested"],
        },
        "cases": rows,
        "errors": [],
    }


def report_from_nightly(
    nightly: dict[str, Any],
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    root: Path = ROOT,
) -> dict[str, Any]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    results = nightly.get("results")
    if not isinstance(results, list):
        raise ValueError("nightly report has no results list")
    return build_report(
        manifest,
        results,
        root=root,
        manifest_bytes=manifest_bytes,
        event=str(nightly.get("event") or ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nightly-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = report_from_nightly(
        _load_json(args.nightly_report), manifest_path=args.manifest
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if report["summary"]["failed"] or report["summary"]["errors"] or report["summary"]["timeouts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
