from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "capability_evidence.py"
SPEC = importlib.util.spec_from_file_location("capability_evidence_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)
MANIFEST = ROOT / "version" / "v8" / "testing" / "capability_cases.json"
SCHEMA = ROOT / "version" / "v8" / "schemas" / "capability_evidence_report.schema.json"


def _result(target: str, status: str, *, error: str = "") -> dict:
    return {
        "name": target,
        "status": status,
        "duration_sec": 1.25,
        "error_msg": error,
        "execution_kind": "make",
        "execution_id": target,
    }


def _build(results: list[dict]) -> dict:
    manifest_bytes = MANIFEST.read_bytes()
    return evidence.build_report(
        json.loads(manifest_bytes),
        results,
        root=ROOT,
        manifest_bytes=manifest_bytes,
        event="pull_request",
    )


def _row(report: dict, case_id: str) -> dict:
    return next(row for row in report["cases"] if row["id"] == case_id)


def test_current_run_evidence_validates_against_schema() -> None:
    report = _build([_result("test-v8-cohere-laguna-contracts", "pass")])
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(report)
    assert report["scope"] == "current_run"
    assert len(report["source"]["manifest_sha256"]) == 64


def test_one_shared_execution_can_prove_multiple_registered_cases() -> None:
    report = _build([_result("test-v8-cohere-laguna-contracts", "pass")])
    assert _row(report, "cohere2.compiler-circuit")["status"] == "pass"
    assert _row(report, "laguna.compiler-circuit")["status"] == "pass"
    assert report["summary"]["passed"] == 2


def test_failure_timeout_and_invalid_status_remain_visible() -> None:
    report = _build(
        [
            _result("test-v8-qwen38-dense-contracts", "fail", error="bad provider"),
            _result("test-v8-qwen38-flash-contracts", "timeout"),
            _result("test-v8-command-a-plus-nvfp4", "unknown"),
        ]
    )
    assert _row(report, "qwen38-dense.storage-lowering")["reason"] == "bad provider"
    assert _row(report, "qwen38-flash.provider-oracles")["status"] == "timeout"
    assert _row(report, "command-a-plus.nvfp4-oracle")["status"] == "error"
    assert report["summary"]["failed"] == 2  # Qwen3.6 and Qwen3.8 share the lane.
    assert report["summary"]["timeouts"] == 1
    assert report["summary"]["errors"] == 1


def test_skipped_and_unselected_cases_are_not_tested_not_passed() -> None:
    report = _build([_result("test-v8-qwen38-flash-contracts", "skip", error="no ISA")])
    skipped = _row(report, "qwen38-flash.provider-oracles")
    hardware = _row(report, "qwen38-dense.long-context")
    assert skipped["selected"] is True
    assert skipped["executed"] is False
    assert skipped["status"] == "not_tested"
    assert skipped["reason"] == "no ISA"
    assert hardware["selected"] is False
    assert hardware["status"] == "not_tested"


def test_test_report_renders_current_capability_evidence() -> None:
    page = (ROOT / "docs" / "site" / "_pages" / "test-report.html").read_text(
        encoding="utf-8"
    )
    assert 'id="capability-evidence-tbody"' in page
    assert "renderCapabilityEvidence(data.capability_evidence || null)" in page
    assert "countRequiredCapabilityFailures" in page
