#!/usr/bin/env python3
"""Structured v8 build-failure evidence and stale-artifact contracts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
BUILDER = SCRIPTS / "build_ir_v8.py"
VISUALIZER = ROOT / "version" / "v8" / "tools" / "open_ir_visualizer_v8.py"
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "v8"
    / "artifact_manifests"
    / "nemotron-nano-9b-v2-q4_k_m.json"
)
DIAGNOSTIC_SCHEMA = ROOT / "version" / "v8" / "schemas" / "build_diagnostic.schema.json"


def _load_builder():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("build_ir_v8_diagnostic_test", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _builder_command(manifest: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(BUILDER),
        "--manifest",
        str(manifest),
        "--mode",
        "prefill",
        "--context-len",
        "32",
        "--output",
        str(output_dir / "ir1_prefill.json"),
        "--layout-output",
        str(output_dir / "layout_prefill.json"),
        "--lowered-output",
        str(output_dir / "lowered_prefill.json"),
        "--call-output",
        str(output_dir / "lowered_prefill_call.json"),
    ]


def test_interface_failure_writes_structured_partial_build_artifact(tmp_path: Path) -> None:
    manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
    attention_ops = manifest["template"]["block_types"]["decoder"]["body"][
        "ops_by_kind"
    ]["attention"]
    v_proj = next(op for op in attention_ops if isinstance(op, dict) and op["op"] == "v_proj")
    v_proj["graph_slots"]["inputs"] = {"x": "main_stream_q8"}

    broken_manifest = tmp_path / "weights_manifest.json"
    broken_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    result = subprocess.run(
        _builder_command(broken_manifest, tmp_path),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "CKE-V8-CIRCUIT-INTERFACE" in result.stderr
    assert "Traceback" not in result.stderr
    diagnostic = json.loads((tmp_path / "build_diagnostic.json").read_text())
    Draft202012Validator(
        json.loads(DIAGNOSTIC_SCHEMA.read_text(encoding="utf-8"))
    ).validate(diagnostic)
    assert diagnostic["schema"] == "cke.v8.build_diagnostic"
    assert diagnostic["status"] == "failed"
    assert len(diagnostic["identity"]["manifest_sha256"]) == 64
    assert diagnostic["source_revision"]
    assert diagnostic["pipeline"] == {
        "failed_stage": "circuit_validation",
        "later_stages": "not_generated",
    }
    failure = diagnostic["failure"]
    assert failure["code"] == "CKE-V8-CIRCUIT-INTERFACE"
    assert failure["category"] == "implementation_defect"
    assert failure["location"]["circuit"] == "nemotron_h"
    assert failure["location"]["operation"] == "v_proj"
    assert failure["location"]["provider"]
    assert "unknown input ports" in failure["summary"]
    assert "A" in failure["expected"]["canonical_ports"]
    assert failure["observed"]["unknown_ports"] == ["x"]
    assert "canonical operation interface" in failure["remediation"]
    assert "CircuitInterfaceError" in diagnostic["traceback"]


def test_successful_rebuild_removes_stale_failure_artifact(tmp_path: Path) -> None:
    diagnostic = tmp_path / "build_diagnostic.json"
    diagnostic.write_text('{"status":"failed"}\n', encoding="utf-8")
    result = subprocess.run(
        _builder_command(FIXTURE, tmp_path),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not diagnostic.exists()


def test_missing_manifest_is_classified_as_user_configuration(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    result = subprocess.run(
        _builder_command(missing, tmp_path),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    diagnostic = json.loads((tmp_path / "build_diagnostic.json").read_text())
    failure = diagnostic["failure"]
    assert failure["code"] == "CKE-V8-INVALID-BUILD-INPUT"
    assert failure["category"] == "user_configuration"
    assert failure["stage"] == "input_validation"


def test_composite_provider_failure_preserves_all_rejection_reasons() -> None:
    builder = _load_builder()
    providers = []
    for provider_id, status, gate in (
        ("candidate_exact", "candidate", "q8_0"),
        ("production_wrong_dtype", "production", "q4_k"),
    ):
        providers.append(
            {
                "id": provider_id,
                "op": "moe_swiglu_shared",
                "quant": {
                    "gate_weight": gate,
                    "up_weight": gate,
                    "down_weight": gate,
                },
                "selection": {
                    "status": status,
                    "priority": 100,
                    "equivalence_group": "moe_swiglu_shared.test.v1",
                    "phases": ["prefill"],
                },
            }
        )

    with pytest.raises(builder.BuildDiagnosticError) as raised:
        builder.resolve_swiglu_moe_provider(
            {"kernels": providers},
            kernel_op="moe_swiglu_shared",
            layer_quant={
                "shared_ffn_gate": "q8_0",
                "shared_ffn_up": "q8_0",
                "shared_ffn_down": "q8_0",
            },
            weight_prefix="shared_ffn",
            mode="prefill",
            prefer_q8_activation=True,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic["code"] == "CKE-V8-COMPOSITE-PROVIDER-NOT-FOUND"
    assert diagnostic["expected"]["gate_weight"] == "q8_0"
    assert [row["reason"] for row in diagnostic["candidates"]] == [
        "status_not_production:candidate",
        "gate_weight_dtype_mismatch",
    ]


def test_visualizer_embeds_partial_build_diagnostic(tmp_path: Path) -> None:
    (tmp_path / "weights_manifest.json").write_text(
        FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    payload = {
        "schema": "cke.v8.build_diagnostic",
        "schema_version": 1,
        "status": "failed",
        "failure": {
            "code": "CKE-V8-COMPOSITE-PROVIDER-NOT-FOUND",
            "stage": "provider_resolution",
            "summary": "No production provider matches",
            "remediation": "Validate the exact storage tuple.",
            "location": {"circuit": "laguna", "operation": "shared_expert"},
            "candidates": [
                {
                    "provider": "candidate_q8",
                    "decision": "rejected",
                    "stage": "provider_selection",
                    "reason": "status_not_production:candidate",
                }
            ],
        },
        "pipeline": {
            "failed_stage": "provider_resolution",
            "later_stages": "not_generated",
        },
    }
    (tmp_path / "build_diagnostic.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    report = tmp_path / "ir_report.html"
    result = subprocess.run(
        [
            sys.executable,
            str(VISUALIZER),
            "--generate",
            "--html-only",
            "--strict-run-artifacts",
            "--run",
            str(tmp_path),
            "--output",
            str(report),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    html = report.read_text(encoding="utf-8")
    assert '"build_diagnostic": {' in html
    assert "CKE-V8-COMPOSITE-PROVIDER-NOT-FOUND" in html
    assert 'id="buildDiagnosticPanel"' in html
    assert "Later pipeline stages" in html
    assert "status_not_production:candidate" in html
