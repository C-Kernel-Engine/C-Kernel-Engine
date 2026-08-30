#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "xray_numerical_parity_v8.py"
SPEC = importlib.util.spec_from_file_location("command_a_plus_xray", SCRIPT)
XRAY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(XRAY)


CHECKPOINTS = [
    "decoder.layer.0.layernorm",
    "decoder.layer.0.q_proj",
    "decoder.layer.0.attention_output",
    "decoder.layer.0.router_logits",
    "decoder.layer.0.routed_moe_output",
    "decoder.layer.0.shared_moe_output",
    "decoder.layer.0.output",
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(checkpoint: str, path: Path, backend: str) -> dict:
    op = checkpoint.rsplit(".", 1)[-1]
    kernel = {
        "routed_moe_output": "moe_swiglu_expert_forward_nvfp4",
        "shared_moe_output": "moe_swiglu_shared_forward_nvfp4",
    }.get(op, op)
    contract = {
        "routed_moe_output": "nvfp4_q8_0_routed_swiglu_fp32",
        "shared_moe_output": "nvfp4_q8_0_shared_swiglu_fp32",
    }.get(op, "fp32_observed")
    return {
        "checkpoint_id": checkpoint,
        "producer": op,
        "phase": "decode",
        "layer": 0,
        "tensor_path": str(path),
        "storage_dtype": "fp32",
        "exported_dtype": "fp32",
        "logical_shape": [1, 16],
        "physical_shape": [1, 16],
        "logical_layout": "token_major",
        "axis_names": ["token", "channel"],
        "physical_axis_names": ["token", "channel"],
        "resolved_contract_id": contract,
        "kernel_id": kernel,
        "function": kernel if backend == "cke" else f"llama::{kernel}",
        "sha256": _digest(path),
    }


def _manifest(backend: str, entries: list[dict]) -> dict:
    return {
        "schema": "cke.checkpoint_manifest",
        "schema_version": 1,
        "backend": backend,
        "run": {
            "model": "command-a-plus-05-2026-w4a4",
            "phase": "decode",
            "source": "bounded-nightly-fixture",
        },
        "checkpoints": entries,
    }


def test_xray_localizes_native_nvfp4_provider_boundary(tmp_path: Path) -> None:
    cke_entries = []
    llama_entries = []
    base = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    for index, checkpoint in enumerate(CHECKPOINTS):
        expected = base + np.float32(index)
        observed = expected.copy()
        if checkpoint.endswith("routed_moe_output"):
            observed[7] += np.float32(0.125)
        elif index > CHECKPOINTS.index("decoder.layer.0.routed_moe_output"):
            observed[7] += np.float32(0.125)
        cke_path = tmp_path / f"cke-{index}.f32"
        llama_path = tmp_path / f"llama-{index}.f32"
        observed.tofile(cke_path)
        expected.tofile(llama_path)
        cke_entries.append(_entry(checkpoint, cke_path, "cke"))
        llama_entries.append(_entry(checkpoint, llama_path, "llamacpp"))

    profile = {
        "schema": "cke.parity_profile",
        "schema_version": 1,
        "name": "command-a-plus-nvfp4-bounded-nightly",
        "backend": "llamacpp",
        "contract_schema_version": 1,
        "required_match_fields": [
            "checkpoint_id",
            "producer",
            "logical_layout",
            "axis_names",
            "resolved_contract_id",
            "kernel_id",
        ],
        "dtype_thresholds": {
            "fp32": {
                "cosine_min": 0.999999,
                "rmse_max": 1.0e-6,
                "relative_rmse_max": 1.0e-6,
                "max_abs_max": 1.0e-6,
                "finite_required": True,
            }
        },
        "observed_storage": {"default": "fp32", "checkpoints": {}},
        "interval_expansions": {},
        "backend_mappings": {},
        "checkpoint_order": CHECKPOINTS,
    }
    report = XRAY.compare_manifests(
        _manifest("cke", cke_entries),
        _manifest("llamacpp", llama_entries),
        profile,
        checkpoint_order=CHECKPOINTS,
    )

    assert report["status"] == "fail"
    assert report["first_divergence"]["checkpoint_id"].endswith(
        "routed_moe_output"
    )
    assert (
        report["first_divergence"]["resolved_execution"]["kernel_id"]
        == "moe_swiglu_expert_forward_nvfp4"
    )
    assert (
        report["first_divergence"]["resolved_execution"]["resolved_contract_id"]
        == "nvfp4_q8_0_routed_swiglu_fp32"
    )


def test_xray_checkpoint_order_covers_parallel_block_joins() -> None:
    assert CHECKPOINTS.index("decoder.layer.0.attention_output") < CHECKPOINTS.index(
        "decoder.layer.0.routed_moe_output"
    )
    assert CHECKPOINTS.index("decoder.layer.0.routed_moe_output") < CHECKPOINTS.index(
        "decoder.layer.0.shared_moe_output"
    )
    assert CHECKPOINTS[-1] == "decoder.layer.0.output"


def test_bounded_nvfp4_lane_is_registered_in_demo_nightly() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    nightly = (ROOT / "scripts" / "nightly_runner.py").read_text(encoding="utf-8")
    assert "test-v8-command-a-plus-nvfp4:" in makefile
    assert '"v8_command_a_plus_nvfp4"' in nightly
    profile = nightly.split('"demo-readiness": [', 1)[1].split("],", 1)[0]
    assert '"v8_command_a_plus_nvfp4"' in profile
