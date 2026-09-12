"""Fail-closed contracts for the Parakeet TDT native CPU candidate."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "version/v8/scripts/run_parakeet_native_v8.py"
    spec = importlib.util.spec_from_file_location("run_parakeet_native_v8", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_five_minute_certifier():
    path = ROOT / "version/v8/scripts/certify_parakeet_long_audio_v8.py"
    spec = importlib.util.spec_from_file_location("certify_parakeet_long_audio_v8", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_long_runner():
    scripts = ROOT / "version/v8/scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "run_parakeet_long_audio_v8.py"
        spec = importlib.util.spec_from_file_location("run_parakeet_long_audio_v8", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def load_chunking_certifier():
    scripts = ROOT / "version/v8/scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "certify_parakeet_chunking_v8.py"
        spec = importlib.util.spec_from_file_location("certify_parakeet_chunking_v8", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


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
        "scaled_residual_add": "audio_scaled_residual_add_f32",
        "argmax": "audio_argmax_first_f32",
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
        "audio_scaled_residual_add_f32",
        "audio_argmax_first_f32",
    )
    for name in names:
        start = source.index(f"int {name}(")
        next_function = source.find("\nint ", start + 5)
        body = source[start : next_function if next_function >= 0 else len(source)]
        assert "malloc(" not in body
        assert "calloc(" not in body
        assert "realloc(" not in body
        assert "free(" not in body


def test_python_session_routes_residual_and_argmax_model_math_to_native() -> None:
    source = (ROOT / "version/v8/scripts/run_parakeet_native_v8.py").read_text(
        encoding="utf-8"
    )
    assert "np.argmax(" not in source
    assert "value + np.float32(0.5)" not in source
    assert "encoder_row + decoder_row" not in source


def test_native_timestamp_contract_matches_pinned_transformers_fixture() -> None:
    runner = load_runner()
    fixture_path = ROOT / "docs/notes/artifacts/parakeet_tdt_0_6b_v3_2086-149220-0033_fp32.npz"
    report = load(ROOT / "docs/notes/artifacts/parakeet_tdt_0_6b_v3_2086-149220-0033_fp32.json")
    with np.load(fixture_path, allow_pickle=False) as fixture:
        sequences = fixture["decode.sequences"][0]
        durations = fixture["decode.durations"][0]
        expected = report["decode"]["timestamps"]
        pieces = iter(item["token"] for item in expected)
        chunks = [None if int(token) in {2, 8192} else next(pieces) for token in sequences]
        actual = runner.refine_token_timestamps(
            sequences, durations, chunks,
            blank_token_id=8192,
            pad_token_id=2,
        )
    assert actual == expected


def test_native_timestamp_contract_rejects_invalid_trajectories() -> None:
    runner = load_runner()
    with pytest.raises(ValueError, match="equal-length rank-one"):
        runner.refine_token_timestamps(
            np.asarray([1, 2]), np.asarray([1]), ["a", "b"],
            blank_token_id=9, pad_token_id=0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        runner.refine_token_timestamps(
            np.asarray([1]), np.asarray([-1]), ["a"],
            blank_token_id=9, pad_token_id=0,
        )


def test_native_frontend_declares_resampling_path() -> None:
    source = (ROOT / "version/v8/scripts/run_parakeet_native_v8.py").read_text(encoding="utf-8")
    assert "audio_resampled_frame_count" in source
    assert "audio_resample_windowed_sinc_f32" in source
    assert 'info.sample_rate != 16000' in source


def test_native_full_attention_rejects_overlength_before_subsampling() -> None:
    runner = load_runner()
    session = object.__new__(runner.ParakeetSession)
    session.encoder_config = {"max_position_embeddings": 4}
    features = np.zeros((33, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="requires 5 encoder positions"):
        session.encode(features, live_frames=32)


def test_retained_five_minute_runs_are_complete_and_deterministic() -> None:
    certifier = load_five_minute_certifier()
    artifacts = ROOT / "docs/notes/artifacts"
    result = certifier.certify(
        ROOT / "version/v8/test_assets/whisper_long_audio/corpus.json",
        artifacts / "parakeet_tdt_0_6b_v3_five_minute_fp32_first.json",
        artifacts / "parakeet_tdt_0_6b_v3_five_minute_fp32.json",
    )
    assert result["status"] == "pass"
    assert result["deterministic_identity"] == {
        "sequences": True,
        "durations": True,
        "transcript": True,
        "timestamps": True,
    }
    assert result["duration_seconds"] == 300.0
    assert result["source_samples"] == 4_800_000
    assert result["word_error_rate"] == pytest.approx(0.10434782608695652)
    assert result["technical_terms"]["CKE"]["exact_in_candidate"] is False


def test_five_minute_certifier_rejects_trajectory_divergence(tmp_path: Path) -> None:
    certifier = load_five_minute_certifier()
    artifacts = ROOT / "docs/notes/artifacts"
    repeat = load(artifacts / "parakeet_tdt_0_6b_v3_five_minute_fp32.json")
    repeat["decode"]["durations"][10] += 1
    divergent = tmp_path / "divergent.json"
    divergent.write_text(json.dumps(repeat), encoding="utf-8")
    with pytest.raises(RuntimeError, match="diverged in durations"):
        certifier.certify(
            ROOT / "version/v8/test_assets/whisper_long_audio/corpus.json",
            artifacts / "parakeet_tdt_0_6b_v3_five_minute_fp32_first.json",
            divergent,
        )


def test_long_audio_window_plan_has_exact_coverage() -> None:
    runner = load_long_runner()
    windows = runner.plan_windows(25_420, 10, 300.0, 30.0)
    assert len(windows) == 10
    assert windows[0].start_frame == 0
    assert windows[-1].end_frame == 25_420
    assert windows[0].ownership_start_frame == 0
    assert windows[-1].ownership_end_frame == 25_420
    assert all(
        windows[index].ownership_end_frame == windows[index + 1].ownership_start_frame
        for index in range(len(windows) - 1)
    )


def test_long_audio_reconciliation_keeps_whole_boundary_words_once() -> None:
    runner = load_long_runner()
    first_words = runner.timestamp_tokens_to_words([
        {"token": " repeated", "start": 33.0, "end": 33.4},
        {"token": " bound", "start": 34.7, "end": 34.9},
        {"token": "ary", "start": 34.9, "end": 35.2},
        {"token": ".", "start": 35.2, "end": 35.2},
    ])
    second_words = runner.timestamp_tokens_to_words([
        {"token": " boundary", "start": 4.7, "end": 5.2},
        {"token": " repeated", "start": 6.0, "end": 6.4},
    ])
    first = runner.select_owned_words(
        first_words, window_start_seconds=0.0,
        ownership_start_seconds=0.0, ownership_end_seconds=35.0,
        final_window=False,
    )
    second = runner.select_owned_words(
        second_words, window_start_seconds=30.0,
        ownership_start_seconds=35.0, ownership_end_seconds=70.0,
        final_window=True,
    )
    assert [word["text"] for word in first + second] == [
        " repeated", " boundary.", " repeated",
    ]
    assert runner.timestamp_tokens_to_words([]) == []


def test_retained_chunked_runs_are_deterministic_and_complete() -> None:
    certifier = load_chunking_certifier()
    artifacts = ROOT / "docs/notes/artifacts"
    result = certifier.certify(
        ROOT / "version/v8/test_assets/whisper_long_audio/kernel2_mic1_5min.txt",
        artifacts / "parakeet_tdt_0_6b_v3_five_minute_fp32.json",
        artifacts / "parakeet_tdt_0_6b_v3_five_minute_chunked_first.json",
        artifacts / "parakeet_tdt_0_6b_v3_five_minute_chunked.json",
    )
    assert result["status"] == "pass"
    assert result["duration_seconds"] == 300.0
    assert result["windows"] == 2
    assert result["deterministic_word_trajectory"] is True
    assert result["chunked_word_error_rate"] == pytest.approx(0.07246376811594203)
