from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "audit_capability_cases_v8.py"
SPEC = importlib.util.spec_from_file_location("audit_capability_cases_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
MANIFEST = ROOT / "version" / "v8" / "testing" / "capability_cases.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_checked_in_capability_cases_are_reachable() -> None:
    report = audit.audit_manifest(load_manifest(), ROOT)
    assert report["status"] == "pass", report["errors"]
    assert report["scope"] == "registration_only"
    assert report["summary"] == {"cases": 8, "families": 5, "errors": 0}
    assert report["coverage"]["cohere2"] == {
        "contract": 1,
        "full_artifact": 1,
    }
    assert report["coverage"]["qwen38"] == {
        "contract": 1,
        "full_artifact": 1,
    }


def test_duplicate_case_ids_fail_closed() -> None:
    payload = load_manifest()
    payload["cases"].append(copy.deepcopy(payload["cases"][0]))
    report = audit.audit_manifest(payload, ROOT)
    assert report["status"] == "fail"
    assert any("duplicate case ID" in error for error in report["errors"])


def test_missing_evidence_path_fails_closed() -> None:
    payload = load_manifest()
    payload["cases"][0]["evidence_paths"] = ["tests/does_not_exist.py"]
    report = audit.audit_manifest(payload, ROOT)
    assert report["status"] == "fail"
    assert any("missing evidence path" in error for error in report["errors"])


def test_unknown_real_model_catalog_id_fails_closed() -> None:
    payload = load_manifest()
    case = next(row for row in payload["cases"] if row["artifact"]["kind"] == "real_model")
    case["artifact"]["catalog_id"] = "missing_model"
    report = audit.audit_manifest(payload, ROOT)
    assert report["status"] == "fail"
    assert any("unknown long-context catalog ID" in error for error in report["errors"])


def test_required_event_cannot_use_an_unregistered_nightly_target() -> None:
    payload = load_manifest()
    case = payload["cases"][0]
    case["entrypoint"]["target"] = "certify-v8-engineering-quality"
    report = audit.audit_manifest(payload, ROOT)
    assert report["status"] == "fail"
    assert any("not registered in nightly MAKE_TARGETS" in error for error in report["errors"])


def test_manifest_rejects_ambiguous_evidence_levels() -> None:
    payload = load_manifest()
    payload["cases"][0]["evidence_level"] = "works"
    report = audit.audit_manifest(payload, ROOT)
    assert report["status"] == "fail"
    assert any(error.startswith("schema:") for error in report["errors"])
