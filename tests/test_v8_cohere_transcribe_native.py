from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/run_cohere_transcribe_native_v8.py"
CIRCUIT = ROOT / "version/v8/circuits/cohere_transcribe.json"
CONVERTER = ROOT / "version/v8/scripts/convert_cohere_transcribe_gguf_to_bump_v8.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cohere_transcribe_native", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


native = _load_module()


def test_native_inference_has_no_reference_framework_dependency() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert "torch" not in imports
    assert "transformers" not in imports
    assert "crispasr" not in imports


def test_session_routes_attention_and_token_selection_to_cke() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "np.matmul(" not in source
    assert "np.argmax(" not in source
    assert "attention_forward_query_key_head_major_f32_decode_heads" in source
    assert "self.k.argmax_first(logits)" in source


def test_circuit_declares_the_certified_bounded_audio_envelope() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    assert circuit["status"] == "native_bounded_long_audio_e2e"
    assert circuit["contract"]["artifact"]["required_tensor_count"] == 2104
    assert circuit["contract"]["audio_encoder"]["blocks"] == 48
    assert circuit["contract"]["audio_decoder"]["blocks"] == 8
    assert circuit["contract"]["runtime_invariants"]["production_kernel_heap_allocation"] is False
    assert circuit["contract"]["runtime_invariants"]["long_audio"] == (
        "external_vad_speech_slices_up_to_30_seconds"
    )
    model_map = json.loads((ROOT / "version/v8/model_maps/gguf_ck_map.json").read_text())
    contract = model_map["architectures"]["cohere-transcribe"]
    assert contract["conversion_status"] == "dedicated_audio_converter"
    assert CONVERTER.name in contract["conversion_blocker"]


def test_decoder_head_layout_round_trip_and_f16_cache_boundary() -> None:
    value = np.arange(3 * 16, dtype=np.float32).reshape(3, 16)
    heads = native.CohereSession._heads(value, 4)
    np.testing.assert_array_equal(native.CohereSession._tokens(heads), value)
    rounded = native.CohereSession._f16_cache_round(
        np.asarray([[[1.0003, -2.0007]]], dtype=np.float32)
    )
    np.testing.assert_array_equal(rounded, rounded.astype(np.float16).astype(np.float32))


def test_sentencepiece_metadata_decodes_the_recorded_piece_contract() -> None:
    weights = native.GGUFWeights.__new__(native.GGUFWeights)
    weights.metadata = {
        "tokenizer.ggml.tokens": ["<|endoftext|>", "▁Well", ",", "▁I"],
    }
    assert weights.token_id("▁Well") == 1
    assert weights.decode([1, 2, 3]) == "Well, I"


def test_comparison_reports_bit_patterns_separately_from_tolerance() -> None:
    actual = np.asarray([1.0, 2.0], dtype=np.float32)
    reference = actual.copy()
    assert native.compare(actual, reference)["bit_exact"] is True
    reference[1] = np.nextafter(reference[1], np.float32(3.0))
    result = native.compare(actual, reference)
    assert result["bit_exact"] is False
    assert result["finite"] is True
    assert result["rmse"] > 0.0


def test_comparison_rejects_identical_nonfinite_bit_patterns() -> None:
    value = np.asarray([np.nan], dtype=np.float32)
    result = native.compare(value, value.copy())
    assert result["bit_exact"] is True
    assert result["finite"] is False


def test_native_runner_publishes_structured_error_report(tmp_path: Path) -> None:
    output = tmp_path / "failure.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            str(tmp_path / "missing.gguf"),
            "--audio",
            str(tmp_path / "missing.wav"),
            "--engine",
            str(tmp_path / "missing-engine.so"),
            "--audio-lib",
            str(tmp_path / "missing-audio.so"),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "ERROR"
    assert report["error"]["type"] in {"OSError", "FileNotFoundError"}


def test_artifact_make_target_is_fail_closed_and_path_portable() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    start = makefile.index("test-cohere-transcribe-native-auto:")
    end = makefile.index("# Policy:", start)
    target = makefile[start:end]
    assert "convert_cohere_transcribe_gguf_to_bump_v8.py" in target
    assert "run_cohere_transcribe_native_v8.py" in target
    assert "CK_COHERE_TRANSCRIBE_REFERENCE_MANIFEST" in target
    assert "CK_COHERE_TRANSCRIBE_REFERENCE_SUMMARY" in target
    assert "/data/" not in target
