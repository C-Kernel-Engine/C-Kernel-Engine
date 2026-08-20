#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "version" / "v8" / "scripts" / "verify_gemma4_runtime_contract_v8.py"
CERTIFY = ROOT / "version" / "v8" / "scripts" / "run_gemma4_certification_v8.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gemma4_runtime_contract_test", VERIFY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify = _load_module()


def _write_runtime(root: Path, *, full_width: int = 4096) -> None:
    (root / "weights_manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "num_hidden_layers": 2,
                    "num_heads": 8,
                    "layer_attention_output_dim": [2048, 4096],
                    "layer_v_head_dim": [256, 512],
                    "layer_sliding_window": [512, 0],
                }
            }
        ),
        encoding="utf-8",
    )
    for phase in ("prefill", "decode"):
        operations = []
        for layer, width in enumerate((2048, full_width)):
            operations.extend(
                [
                    {
                        "op": "quantize_out_proj_input",
                        "layer": layer,
                        "args": [{"name": "k", "expr": str(width)}],
                    },
                    {
                        "op": "out_proj",
                        "layer": layer,
                        "args": [{"name": "K", "expr": str(width)}],
                    },
                ]
            )
        (root / f"lowered_{phase}_call.json").write_text(
            json.dumps({"operations": operations}), encoding="utf-8"
        )
        (root / f"layout_{phase}.json").write_text(
            json.dumps(
                {
                    "validation": {
                        "activation_memory": {
                            "status": "PASS",
                            "arena_bytes": 8192,
                            "writes": [
                                {"required_bytes": 4096, "available_bytes": 8192}
                            ],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )


def test_runtime_contract_accepts_mixed_widths_and_planned_writes(tmp_path: Path) -> None:
    _write_runtime(tmp_path)
    report = verify.verify_runtime(tmp_path)

    assert report["status"] == "PASS"
    assert report["attention_output_widths"] == [2048, 4096]
    assert report["phases"]["prefill"]["checked_attention_width_arguments"] == 4
    assert report["phases"]["decode"]["memory"]["checked_writes"] == 1


def test_runtime_contract_rejects_truncated_full_attention_width(tmp_path: Path) -> None:
    _write_runtime(tmp_path, full_width=2048)
    with pytest.raises(ValueError, match="layer 1 quantize_out_proj_input.k=2048"):
        verify.verify_runtime(tmp_path)


def test_runtime_contract_rejects_missing_write_evidence(tmp_path: Path) -> None:
    _write_runtime(tmp_path)
    layout = tmp_path / "layout_prefill.json"
    payload = json.loads(layout.read_text(encoding="utf-8"))
    payload["validation"]["activation_memory"]["writes"] = []
    layout.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="prefill.*no write extents"):
        verify.verify_runtime(tmp_path)


def test_runtime_contract_rejects_duplicate_width_operation(tmp_path: Path) -> None:
    _write_runtime(tmp_path)
    lowered = tmp_path / "lowered_decode_call.json"
    payload = json.loads(lowered.read_text(encoding="utf-8"))
    payload["operations"].append(dict(payload["operations"][-1]))
    lowered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="decode has duplicate out_proj for layer 1"):
        verify.verify_runtime(tmp_path)


def test_capability_gated_certification_is_registered() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "test-v8-gemma4-highmem:" in makefile
    assert "--min-memory-gb $(V8_GEMMA4_MIN_MEM_GB)" in makefile
    assert "nightly-gemma4-e2e:" in makefile


def test_certification_skips_before_model_access_when_memory_is_insufficient(
    tmp_path: Path,
) -> None:
    report = tmp_path / "summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(CERTIFY),
            "--model",
            "/does/not/exist.gguf",
            "--work-root",
            str(tmp_path / "work"),
            "--report",
            str(report),
            "--min-memory-gb",
            "999999",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert "SKIP:" in completed.stdout
    assert payload["status"] == "SKIP"
    assert payload["memory_preflight"]["available_bytes"] > 0
