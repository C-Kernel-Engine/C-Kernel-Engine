from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/run_cohere_transcribe_long_audio_v8.py"
CERTIFIER = ROOT / "version/v8/scripts/certify_cohere_transcribe_long_audio_v8.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cohere_transcribe_long_audio", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


long_audio = _load_module()


def _load_certifier():
    spec = importlib.util.spec_from_file_location("cohere_transcribe_long_certifier", CERTIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


certifier = _load_certifier()


def test_window_plan_has_exact_single_owner_coverage() -> None:
    windows = long_audio.plan_windows(1000, 10, 30.0, 5.0)
    assert [(item.start_frame, item.end_frame) for item in windows] == [
        (0, 300), (250, 550), (500, 800), (750, 1000),
    ]
    assert windows[0].ownership_start_frame == 0
    assert windows[-1].ownership_end_frame == 1000
    assert all(
        windows[index].ownership_end_frame == windows[index + 1].ownership_start_frame
        for index in range(len(windows) - 1)
    )


@pytest.mark.parametrize(
    "window,overlap",
    [(0.0, 0.0), (5.0, -1.0), (5.0, 5.0), (5.0, 6.0)],
)
def test_window_plan_rejects_invalid_geometry(window: float, overlap: float) -> None:
    with pytest.raises(ValueError):
        long_audio.plan_windows(1000, 10, window, overlap)


def test_monotonic_dtw_follows_increasing_attention_frames() -> None:
    attention = np.asarray([
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.8, 0.2, 0.0],
        [0.0, 0.0, 0.1, 0.9],
    ], dtype=np.float32)
    np.testing.assert_array_equal(
        long_audio.monotonic_dtw_path(attention), [0, 1, 3],
    )


def test_energy_windows_cut_at_the_quietest_search_region() -> None:
    samples = np.ones(600, dtype=np.float32)
    samples[250:260] = 0.0
    samples[500:510] = 0.0
    windows = long_audio.plan_energy_windows(
        samples, sample_rate=10, window_seconds=30.0,
        search_seconds=10.0, energy_window_seconds=1.0,
    )
    assert [(item.start_frame, item.end_frame) for item in windows] == [
        (0, 250), (250, 500), (500, 600),
    ]
    assert all(item.start_frame == item.ownership_start_frame for item in windows)
    assert all(item.end_frame == item.ownership_end_frame for item in windows)


def test_imported_speech_windows_are_strict_and_hashed(tmp_path: Path) -> None:
    path = tmp_path / "vad.json"
    path.write_text(json.dumps({
        "crispasr_vad": {
            "version": 1,
            "kind": "chunks",
            "sample_rate": 10,
            "num_slices": 2,
            "slices": [
                {"start": 10, "end": 100},
                {"start": 200, "end": 300},
            ],
        },
    }))
    windows, identity = long_audio.load_speech_windows(
        path, total_frames=400, sample_rate=10, max_window_seconds=10.0,
    )
    assert [(item.start_frame, item.end_frame) for item in windows] == [(10, 100), (200, 300)]
    assert identity["format"] == "crispasr_vad"
    assert len(identity["sha256"]) == 64

    document = json.loads(path.read_text())
    document["crispasr_vad"]["slices"][1]["start"] = 99
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="overlapping"):
        long_audio.load_speech_windows(
            path, total_frames=400, sample_rate=10, max_window_seconds=10.0,
        )


def test_decode_loop_detection_and_word_error_rate() -> None:
    clean = "one two three one two"
    assert long_audio.repeated_ngram_runs(clean) == []
    loop = "next slide next slide next slide"
    findings = long_audio.repeated_ngram_runs(loop)
    assert any(item["ngram_words"] == 2 and item["repetitions"] == 3 for item in findings)
    comparison = long_audio.word_error_rate("one two three", "one too three")
    assert comparison == {
        "reference_words": 3,
        "candidate_words": 3,
        "word_edits": 1,
        "word_error_rate": pytest.approx(1 / 3),
    }


def test_certifier_requires_loaded_runtime_and_exact_repeat(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"transcription": [{"text": "one two three"}]}))
    common = {
        "schema": "cke.v8.cohere_transcribe.long_audio",
        "status": "PASS",
        "model": {"id": "cohere"},
        "input": {"sha256": "audio"},
        "runtime": {
            "engine": {"sha256": "engine", "present_in_process_maps": True},
            "audio": {"sha256": "audio-lib", "present_in_process_maps": True},
        },
        "policy": {"kind": "imported_vad_speech_slices"},
        "speech_segments": {"sha256": "segments"},
        "windows": [{"generated_token_ids": [1, 2, 3]}],
        "selected_tokens": [{"id": 1, "text": "one two three", "start": 0.0, "end": 1.0}],
        "transcript": "one two three",
        "total_elapsed_seconds": 1.0,
        "real_time_factor": 0.1,
        "peak_rss_bytes": 100,
    }
    candidate, repeat = tmp_path / "candidate.json", tmp_path / "repeat.json"
    candidate.write_text(json.dumps(common))
    repeat.write_text(json.dumps(common))
    report = certifier.certify(candidate, repeat, reference, 0.1)
    assert report["status"] == "PASS"
    assert all(report["checks"].values())

    changed = json.loads(repeat.read_text())
    changed["runtime"]["engine"]["present_in_process_maps"] = False
    repeat.write_text(json.dumps(changed))
    report = certifier.certify(candidate, repeat, reference, 0.1)
    assert report["status"] == "FAIL"
    assert not report["checks"]["loaded_libraries_verified"]
    with pytest.raises(ValueError, match="distinct"):
        certifier.certify(candidate, candidate, reference, 0.1)


def test_monotonic_dtw_rejects_nonfinite_or_empty_attention() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        long_audio.monotonic_dtw_path(np.empty((0, 3), np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        long_audio.monotonic_dtw_path(np.asarray([[np.nan]], np.float32))


def test_overlap_ownership_selects_each_boundary_token_once() -> None:
    windows = long_audio.plan_windows(550, 10, 30.0, 5.0)
    first = long_audio.select_owned_tokens(
        [{
            "id": 1, "text": " first", "start": 27.0, "end": 28.0,
            "ownership_start": 27.0, "ownership_end": 28.0,
        }],
        windows[0], 10, False,
    )
    second = long_audio.select_owned_tokens(
        [{
            "id": 1, "text": " first", "start": 2.0, "end": 3.0,
            "ownership_start": 2.0, "ownership_end": 3.0,
        }],
        windows[1], 10, True,
    )
    assert len(first) + len(second) == 1


def test_caption_and_srt_output_preserve_text_and_time() -> None:
    tokens = [
        {"text": " Hello", "start": 1.0, "end": 1.2},
        {"text": " world.", "start": 1.2, "end": 2.0},
        {"text": " Again", "start": 4.0, "end": 4.5},
    ]
    captions = long_audio.build_captions(tokens)
    assert [item["text"] for item in captions] == ["Hello world.", "Again"]
    assert long_audio.render_srt(captions) == (
        "1\n00:00:01,000 --> 00:00:02,000\nHello world.\n\n"
        "2\n00:00:04,000 --> 00:00:04,500\nAgain\n"
    )


def test_long_audio_runner_declares_bounded_state_and_native_resampling() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"state": "reset_each_window"' in source
    assert '"quiet_point_half_open_window"' in source
    assert '"linear_token_byte_midpoint_at_overlap_midpoint"' in source
    assert '"timestamp_alignment": "last_decoder_layer_cross_attention_monotonic_dtw"' in source
    assert '"resampling": "cke_windowed_sinc_radius_16_before_frontend"' in source
    assert "np.matmul(" not in source
    assert "np.argmax(" in source  # DTW terminal selection, not model token selection.


def test_long_audio_make_target_is_optional_portable_and_fail_closed() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    start = makefile.index("test-cohere-transcribe-long-audio-auto:")
    end = makefile.index("# Policy:", start)
    target = makefile[start:end]
    assert "run_cohere_transcribe_long_audio_v8.py" in target
    assert "certify_cohere_transcribe_long_audio_v8.py" in target
    assert "CK_COHERE_TRANSCRIBE_SPEECH_SEGMENTS" in target
    assert "CK_COHERE_TRANSCRIBE_LONG_REFERENCE" in target
    assert "first repeat" in target
    assert "|| exit $$?" in target
    assert "/data/" not in target


def test_long_audio_runner_publishes_early_failure(tmp_path: Path) -> None:
    output = tmp_path / "error.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model", str(tmp_path / "missing-model"),
            "--audio", str(tmp_path / "missing.wav"),
            "--engine", str(tmp_path / "missing-engine.so"),
            "--audio-lib", str(tmp_path / "missing-audio.so"),
            "--output", str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    report = json.loads(output.read_text())
    assert report["status"] == "ERROR"
    assert report["error"]["type"] == "FileNotFoundError"
