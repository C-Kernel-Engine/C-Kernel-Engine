from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "certify_model_memory_plans_v8.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("certify_model_memory_plans_v8_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase_evidence_accepts_planner_owned_write_extents(tmp_path: Path) -> None:
    module = _load_module()
    layout = tmp_path / "layout.json"
    lowered = tmp_path / "lowered.json"
    lowered.write_text(
        json.dumps({"operations": [{"outputs": {"output": {}}}, {"outputs": {}}]}),
        encoding="utf-8",
    )
    layout.write_text(
        json.dumps(
            {
                "validation": {
                    "activation_memory": {
                        "status": "PASS",
                        "arena_bytes": 4096,
                        "activation_buffer_count": 2,
                        "writes": [
                            {"required_bytes": 2920, "available_bytes": 4096}
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    evidence = module._phase_evidence(layout, lowered, "prefill")

    assert evidence["extent_validated_write_count"] == 1
    assert evidence["writable_operation_count"] == 1
    assert evidence["extent_coverage_percent"] == 100.0
    assert evidence["min_write_headroom_bytes"] == 1176


def test_phase_evidence_rejects_undersized_write(tmp_path: Path) -> None:
    module = _load_module()
    layout = tmp_path / "layout.json"
    lowered = tmp_path / "lowered.json"
    lowered.write_text(json.dumps({"operations": []}), encoding="utf-8")
    layout.write_text(
        json.dumps(
            {
                "validation": {
                    "activation_memory": {
                        "status": "PASS",
                        "writes": [
                            {"required_bytes": 292000, "available_bytes": 272000}
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="invalid write extent"):
        module._phase_evidence(layout, lowered, "prefill")


def test_extent_coverage_baseline_fails_closed(tmp_path: Path) -> None:
    module = _load_module()
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"models": {"gemma": {"prefill": 9, "decode": 9}}}),
        encoding="utf-8",
    )
    rows = [
        {
            "model_id": "gemma",
            "status": "PASS",
            "phases": [
                {"phase": "prefill", "extent_validated_write_count": 8},
                {"phase": "decode", "extent_validated_write_count": 9},
            ],
        }
    ]

    module._apply_baseline(rows, baseline)

    assert rows[0]["status"] == "FAIL"
    assert "prefill writes 8 < baseline 9" in rows[0]["reason"]


def test_nightly_registers_planner_only_model_matrix() -> None:
    source = (ROOT / "scripts" / "nightly_runner.py").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert '"v8_model_memory_plans"' in source
    assert '"target": "v8-model-memory-plans-nightly"' in source
    assert '"v8-model-memory-plans-nightly": ROOT' in source
    assert "v8-model-memory-plans-nightly:" in makefile
    assert "certify_model_memory_plans_v8.py" in makefile


def test_qwen38_optional_model_uses_full_context_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setenv("V8_QWEN38_MODEL", "/models/qwen38.bump")
    monkeypatch.delenv("V8_QWEN38_MEMORY_PLAN_CONTEXTS", raising=False)

    rows = module._load_models(ROOT / "version" / "v8" / "regression" / "families.json")
    qwen38 = next(row for row in rows if row["id"] == "qwen38")

    assert qwen38["model"] == "/models/qwen38.bump"
    assert qwen38["contexts"] == [262144]

    baseline = json.loads(
        (ROOT / "version" / "v8" / "contracts" / "model_memory_plan_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    assert baseline["models"]["qwen38"] == {"prefill": 265, "decode": 217}
