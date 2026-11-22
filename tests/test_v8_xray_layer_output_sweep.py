import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "xray_layer_output_sweep_v8.py"
SPEC = importlib.util.spec_from_file_location("xray_layer_output_sweep_v8", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_layer_sweep_compares_final_row_and_discloses_row_only_oracle(tmp_path):
    subject_dir = tmp_path / "subject"
    oracle_dir = tmp_path / "oracle"
    subject_dir.mkdir()
    oracle_dir.mkdir()

    layer0 = np.arange(12, dtype=np.float32).reshape(3, 4)
    layer1 = layer0 + 10
    layer0.tofile(subject_dir / "tok_0000_layer_000_layer_out.f32")
    layer1.tofile(subject_dir / "tok_0000_layer_001_layer_out.f32")
    layer0.tofile(oracle_dir / "l_out-0-token-000002-occ-000.bin")
    (layer1[-1] + np.float32(0.25)).tofile(
        oracle_dir / "l_out-1-token-000002-occ-000.bin"
    )

    report = MODULE.build_report(
        subject_dir=subject_dir,
        oracle_dir=oracle_dir,
        parity_report={
            "threads": 4,
            "ck_runtime": {"path": "/runtime.so", "sha256": "a" * 64},
            "first_divergence": {"step": 0, "ck_next": 1, "llama_next": 2},
        },
        production_report={"first_divergence": {"step": 251}},
        model="fixture",
        layers=2,
        token_count=3,
        hidden_size=4,
        logical_token=2,
        subject_pattern="tok_0000_layer_{layer:03d}_layer_out.f32",
        oracle_pattern="l_out-{layer}-token-{token:06d}-occ-000.bin",
    )

    assert report["capture_scope"]["covered_layer_count"] == 2
    assert report["capture_scope"]["checkpoint_granularity"] == "layer_output_final_causal_row"
    assert report["comparisons"][0]["metrics"]["byte_exact"] is True
    assert report["comparisons"][1]["metrics"]["max_abs"] == 0.25
    assert report["comparisons"][1]["capture_extent"]["oracle"] == "final_row_only"
    assert report["first_non_exact_stop"] == 1
    assert report["first_divergence"]["classification"] == "OBSERVED_LAYER_OUTPUT_DIVERGENCE"
    assert report["production_baseline"]["first_divergence"]["step"] == 251


def test_layer_sweep_reports_production_decode_scope_truthfully(tmp_path):
    subject_dir = tmp_path / "subject"
    oracle_dir = tmp_path / "oracle"
    subject_dir.mkdir()
    oracle_dir.mkdir()
    values = np.arange(4, dtype=np.float32)
    values.tofile(subject_dir / "tok_0007_layer_000_layer_out.f32")
    values.tofile(oracle_dir / "l_out-0-token-000007-occ-000.bin")

    report = MODULE.build_report(
        subject_dir=subject_dir,
        oracle_dir=oracle_dir,
        parity_report={"first_divergence": None},
        production_report=None,
        model="fixture",
        layers=1,
        token_count=1,
        hidden_size=4,
        logical_token=7,
        subject_pattern="tok_{token:04d}_layer_{layer:03d}_layer_out.f32",
        oracle_pattern="l_out-{layer}-token-{token:06d}-occ-000.bin",
        phase="decode",
        comparison_mode="production_decode",
    )

    assert report["phase"] == "decode"
    assert report["run"]["comparison_mode"] == "production_decode"
    assert (
        report["capture_scope"]["comparison"]
        == "CK persistent decode vs llama.cpp production decode"
    )
    assert report["provenance"]["oracle"]["mode"] == "production_graph_tensor_dump"


def test_ordered_capture_matches_oracle_slots_instead_of_execution_index(tmp_path):
    subject_dir = tmp_path / "subject"
    oracle_dir = tmp_path / "oracle"
    subject_dir.mkdir()
    oracle_dir.mkdir()
    checkpoints = []
    for sequence_index, slot in enumerate((1, 0)):
        filename = f"actual-{sequence_index}.f32"
        values = np.full(4, slot + 1, dtype="<f4")
        values.tofile(subject_dir / filename)
        values.tofile(oracle_dir / f"l_out-{slot}-token-000007-occ-000.bin")
        checkpoints.append({
            "id": f"fixture.slot{slot}", "sequence_index": sequence_index, "state_slot": slot,
            "weight_layer": 0, "loop_pass": sequence_index, "filename": filename,
        })
    report = MODULE.build_report(
        subject_dir=subject_dir, oracle_dir=oracle_dir, parity_report={}, production_report=None,
        model="fixture", layers=2, token_count=1, hidden_size=4, logical_token=7,
        phase="decode", subject_pattern="unused", oracle_pattern="l_out-{layer}-token-{token:06d}-occ-000.bin",
        ordered_capture={"phase": "append", "position": 7, "checkpoints": checkpoints},
    )
    assert report["status"] == "pass"
    assert [row["checkpoint_id"] for row in report["comparisons"]] == ["fixture.slot1", "fixture.slot0"]
    assert [row["resolved_execution"]["state_slot"] for row in report["comparisons"]] == [1, 0]
