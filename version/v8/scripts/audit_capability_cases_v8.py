#!/usr/bin/env python3
"""Audit the declared route from model capabilities to executable evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "version" / "v8" / "testing" / "capability_cases.json"
SCHEMA = ROOT / "version" / "v8" / "schemas" / "capability_case_manifest.schema.json"
DEFAULT_REPORT = ROOT / "build" / "v8" / "capability-case-audit.json"
LONG_CONTEXT_CATALOG = ROOT / "version" / "v8" / "regression" / "long_context_models.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _make_targets(makefile: Path) -> set[str]:
    pattern = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*):(?:\s|$)")
    return {
        match.group(1)
        for line in makefile.read_text(encoding="utf-8").splitlines()
        if (match := pattern.match(line))
    }


def _nightly_make_targets(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignment = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and (
                any(isinstance(target, ast.Name) and target.id == "MAKE_TARGETS" for target in node.targets)
                if isinstance(node, ast.Assign)
                else isinstance(node.target, ast.Name) and node.target.id == "MAKE_TARGETS"
            )
        ),
        None,
    )
    if assignment is None:
        raise ValueError(f"MAKE_TARGETS is missing from {path}")
    value = assignment.value
    if not isinstance(value, ast.Dict):
        raise ValueError("nightly MAKE_TARGETS must be a dictionary literal")
    targets: set[str] = set()
    for row in value.values:
        if not isinstance(row, ast.Dict):
            continue
        for key, item in zip(row.keys, row.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "target"
                and isinstance(item, ast.Constant)
                and isinstance(item.value, str)
            ):
                targets.add(item.value)
    return targets


def _catalog_ids(path: Path) -> set[str]:
    payload = _load_json(path)
    return {
        str(row.get("id"))
        for row in payload.get("models", [])
        if isinstance(row, dict) and row.get("id")
    }


def _repository_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def audit_manifest(
    payload: dict[str, Any],
    root: Path = ROOT,
    *,
    manifest_bytes: bytes | None = None,
) -> dict[str, Any]:
    schema = _load_json(root / SCHEMA.relative_to(ROOT))
    schema_errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    errors = [
        f"schema:{'.'.join(str(item) for item in error.absolute_path) or '<root>'}: {error.message}"
        for error in schema_errors
    ]
    if schema_errors:
        return {
            "schema": "cke.v8.capability_case_audit",
            "schema_version": 1,
            "scope": "registration_only",
            "status": "fail",
            "summary": {"cases": 0, "families": 0, "errors": len(errors)},
            "coverage": {},
            "errors": errors,
        }

    make_targets = _make_targets(root / "Makefile")
    nightly_targets = _nightly_make_targets(root / "scripts" / "nightly_runner.py")
    catalog_ids = _catalog_ids(root / LONG_CONTEXT_CATALOG.relative_to(ROOT))
    seen: set[str] = set()
    coverage: dict[str, Counter[str]] = defaultdict(Counter)

    for case in payload["cases"]:
        case_id = case["id"]
        if case_id in seen:
            errors.append(f"{case_id}: duplicate case ID")
        seen.add(case_id)
        coverage[case["family"]][case["evidence_level"]] += 1

        for circuit in case["circuits"]:
            path = root / "version" / "v8" / "circuits" / f"{circuit}.json"
            if not path.is_file():
                errors.append(f"{case_id}: missing circuit {circuit}: {path.relative_to(root)}")

        target = case["entrypoint"]["target"]
        if target not in make_targets:
            errors.append(f"{case_id}: unknown Make target {target}")
        events = set(case["schedule"]["events"])
        if events & {"pull_request", "nightly"} and target not in nightly_targets:
            errors.append(f"{case_id}: {target} is not registered in nightly MAKE_TARGETS")

        artifact = case["artifact"]
        if artifact["kind"] == "real_model" and artifact["catalog_id"] not in catalog_ids:
            errors.append(
                f"{case_id}: unknown long-context catalog ID {artifact['catalog_id']}"
            )

        for evidence_path in case["evidence_paths"]:
            if not (root / evidence_path).is_file():
                errors.append(f"{case_id}: missing evidence path {evidence_path}")

    normalized_coverage = {
        family: dict(sorted(levels.items()))
        for family, levels in sorted(coverage.items())
    }
    return {
        "schema": "cke.v8.capability_case_audit",
        "schema_version": 1,
        "scope": "registration_only",
        "status": "fail" if errors else "pass",
        "source": {
            "repository_commit": _repository_commit(root),
            "manifest_sha256": (
                hashlib.sha256(manifest_bytes).hexdigest()
                if manifest_bytes is not None
                else ""
            ),
        },
        "summary": {
            "cases": len(payload["cases"]),
            "families": len(coverage),
            "errors": len(errors),
        },
        "coverage": normalized_coverage,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    try:
        manifest_bytes = args.manifest.read_bytes()
        report = audit_manifest(
            json.loads(manifest_bytes),
            manifest_bytes=manifest_bytes,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema": "cke.v8.capability_case_audit",
            "schema_version": 1,
            "scope": "registration_only",
            "status": "fail",
            "summary": {"cases": 0, "families": 0, "errors": 1},
            "coverage": {},
            "errors": [str(exc)],
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"capability cases: status={report['status']} "
        f"cases={report['summary']['cases']} "
        f"families={report['summary']['families']} "
        f"errors={report['summary']['errors']}"
    )
    for error in report["errors"]:
        print(f"FAIL: {error}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
