"""Fail-closed contracts for the Parakeet TDT native CPU candidate."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_complete_checkpoint_mapping_has_no_silent_leftovers() -> None:
    manifest = load(ROOT / "docs/notes/artifacts/parakeet_tdt_0_6b_v3_tensor_manifest.json")
    model_map = load(ROOT / "version/v8/model_maps/safetensors_ck_map.json")["architectures"]["parakeet_tdt"]
    source = {tensor["name"] for tensor in manifest["tensors"]}
    mapped = set()
    layers = 24
    for ref in model_map["tensor_refs"]:
        for pattern in ref["sources"]:
            if "{L}" in pattern:
                mapped.update(pattern.replace("{L}", str(layer)) for layer in range(layers))
            else:
                mapped.add(pattern)
    ignored = {name for name in source if name.endswith(".num_batches_tracked")}
    assert len(source) == 723
    assert len(mapped) == 699
    assert len(ignored) == 24
    assert source == mapped | ignored


def test_circuit_binds_new_native_providers_and_declares_memory_ownership() -> None:
    circuit = load(ROOT / "version/v8/circuits/parakeet_tdt.json")
    assert circuit["kernels"] == {
        "relative_position": "audio_relative_sinusoidal_position_f32",
        "relative_attention": "audio_conformer_relative_attention_f32",
        "grouped_conv1d": "audio_conv1d_channel_major_grouped_f32",
        "batch_norm": "audio_batch_norm_inference_channel_major_f32",
        "lstm": "audio_lstm_step_f32",
    }
    invariants = circuit["contract"]["runtime_invariants"]
    assert invariants["weight_container"] == "BUMPWGT5"
    assert invariants["production_kernel_heap_allocation"] is False


def test_normal_inference_runner_has_no_reference_framework_import() -> None:
    path = ROOT / "version/v8/scripts/run_parakeet_native_v8.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level.append(node.module or "")
    assert "torch" not in top_level
    assert "transformers" not in top_level
    assert "safetensors" not in top_level


def test_new_compute_kernels_do_not_allocate_or_free() -> None:
    source = (ROOT / "src/kernels/audio_kernels.c").read_text(encoding="utf-8")
    names = (
        "audio_relative_sinusoidal_position_f32",
        "audio_batch_norm_inference_channel_major_f32",
        "audio_lstm_step_f32",
        "audio_conv1d_channel_major_grouped_f32",
        "audio_conformer_relative_attention_f32",
    )
    for name in names:
        start = source.index(f"int {name}(")
        next_function = source.find("\nint ", start + 5)
        body = source[start : next_function if next_function >= 0 else len(source)]
        assert "malloc(" not in body
        assert "calloc(" not in body
        assert "realloc(" not in body
        assert "free(" not in body
