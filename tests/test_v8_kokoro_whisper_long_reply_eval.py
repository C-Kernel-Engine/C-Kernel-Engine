"""Integrity and critical-fact controls for the long Kokoro/Whisper diagnostic."""

import hashlib
import importlib.util
import json
from pathlib import Path
import wave

import pytest


SCRIPT = (Path(__file__).resolve().parents[1] / "version/v8/tts"
          / "evaluate_kokoro_whisper_long_reply.py")
SPEC = importlib.util.spec_from_file_location("kokoro_whisper_long_reply", SCRIPT)
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)
SOURCE_FIXTURE = (Path(__file__).resolve().parents[1] / "version/v8/test_assets"
                  / "kokoro_long_reply/technical_summary_20261106.txt")


def test_retained_real_source_has_expected_identity():
    source = SOURCE_FIXTURE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "0bbca9d04ece5e012929adcd8093f6e8117e2c9e93a6489e54b0c4e65eb4bdf7")
    assert len(source.decode("utf-8").split()) == 1133


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _wav(path, rate, frames):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * frames)


def _fixture(tmp_path):
    source = tmp_path / "source.txt"
    plan = tmp_path / "plan.json"
    tts = tmp_path / "tts.json"
    wav = tmp_path / "tts.wav"
    asr_wav = tmp_path / "asr.wav"
    asr = tmp_path / "asr.json"
    resample = tmp_path / "resample.json"
    text = "PR 1035 changed the value from -3.5 to +3.5."
    source.write_text(text, encoding="utf-8")
    _wav(wav, 24000, 24000)
    _wav(asr_wav, 16000, 16000)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    planned_chunks = [
        {"source_start": 0, "source_end": 22,
         "source_text": text[:22], "valid_token_count": 18, "voice_row": 15},
        {"source_start": 22, "source_end": len(text),
         "source_text": text[22:], "valid_token_count": 20, "voice_row": 17},
    ]
    _write_json(plan, {"source_text": text, "chunks": planned_chunks})
    _write_json(tts, {"status": "execution_pass", "source_sha256":
                        hashlib.sha256(text.encode()).hexdigest(),
                        "wav_sha256": sha(wav), "chunks": [
                            {"index": index, **chunk, "samples": 12000}
                            for index, chunk in enumerate(planned_chunks)],
                        "sample_count": 24000, "sample_rate": 24000,
                        "numerical_waveform_parity": "not_tested_this_run",
                        "known_connected_waveform_parity": "fail_unresolved"})
    _write_json(resample, {"provider": "audio_resample_windowed_sinc_f32",
                           "source_sha256": sha(wav), "target_sha256": sha(asr_wav),
                           "source_frames": 24000, "source_rate": 24000,
                           "target_frames": 16000, "target_rate": 16000})
    _write_json(asr, {"status": "ok", "wav_sha256": sha(asr_wav),
                      "timestamps": True, "encoder_runtime_sha256": "encoder",
                      "decoder_runtime_sha256": "decoder",
                      "decoder": {"stop": "segment_complete", "transcript_text": text},
                      "segments": [{"index": 0, "source_frame_start": 0,
                                    "source_frame_window_end": 16000,
                                    "source_frame_consumed_end": 16000,
                                    "start_seconds": 0.0, "end_seconds": 1.0}]})
    paths = (source, plan, tts, wav, asr_wav, asr, resample)
    return paths


def test_complete_timestamp_round_trip_preserves_separate_verdicts(tmp_path):
    result = evaluator.evaluate(*_fixture(tmp_path))
    assert result["execution"] == "pass"
    assert result["transcript_fidelity"] == "pass"
    assert result["exact_number_fidelity"] == "pass"
    assert result["known_connected_waveform_parity"] == "fail_unresolved"
    assert result["human_listening"] == "not_tested"


def test_changed_sign_fails_critical_fact_even_when_wer_passes(tmp_path):
    paths = _fixture(tmp_path)
    report = json.loads(paths[5].read_text())
    report["decoder"]["transcript_text"] = "PR 1035 changed the value from +3.5 to +3.5."
    _write_json(paths[5], report)
    result = evaluator.evaluate(*paths, maximum_wer=0.2)
    assert result["transcript_fidelity"] == "pass"
    assert result["exact_number_fidelity"] == "fail"
    assert result["missing_exact_numbers"] == ["-3.5"]
    assert result["unexpected_exact_numbers"] == ["+3.5"]


def test_identifier_hyphen_is_not_a_negative_number(tmp_path):
    paths = _fixture(tmp_path)
    report = json.loads(paths[5].read_text())
    report["decoder"]["transcript_text"] = (
        "PR-1035 changed the value from -3.5 to +3.5.")
    _write_json(paths[5], report)
    result = evaluator.evaluate(*paths)
    assert result["transcript_fidelity"] == "pass"
    assert result["exact_number_fidelity"] == "pass"
    assert result["missing_exact_numbers"] == []


def test_identifier_rule_does_not_hide_subtraction_signs():
    assert evaluator.NUMBERS.findall(evaluator._identifier_aware("PR-1035 x-3.5")) == [
        "1035", "-3.5"]
    assert evaluator.NUMBERS.findall(evaluator._identifier_aware("PR-1035 x+3.5")) == [
        "1035", "+3.5"]


def test_swapped_numbers_fail_even_when_the_multiset_matches(tmp_path):
    paths = _fixture(tmp_path)
    report = json.loads(paths[5].read_text())
    report["decoder"]["transcript_text"] = (
        "PR 1035 changed the value from +3.5 to -3.5.")
    _write_json(paths[5], report)
    result = evaluator.evaluate(*paths)
    assert result["missing_exact_numbers"] == []
    assert result["unexpected_exact_numbers"] == []
    assert result["number_sequence_match"] is False
    assert result["exact_number_fidelity"] == "fail"


def test_intact_spoken_compounds_match_digits_but_split_numbers_fail():
    assert evaluator._number_facts("forty two, then forty-seven") == ["42", "47"]
    assert evaluator._number_facts("42, then 47") == ["42", "47"]
    assert evaluator._number_facts("40, two; then 40. Seven") != ["42", "47"]


@pytest.mark.parametrize("mutation", ["fixed_window", "early_end", "wrong_audio",
                                         "text_gap", "wrong_resample_hash",
                                         "wrong_voice_row", "short_chunk_total"])
def test_incomplete_or_mixed_run_is_rejected(tmp_path, mutation):
    paths = _fixture(tmp_path)
    if mutation == "text_gap":
        path = paths[1]
        value = json.loads(path.read_text())
        value["chunks"][1]["source_start"] += 1
    elif mutation == "wrong_resample_hash":
        path = paths[6]
        value = json.loads(path.read_text())
        value["target_sha256"] = "wrong"
    elif mutation in ("wrong_voice_row", "short_chunk_total"):
        path = paths[2]
        value = json.loads(path.read_text())
        if mutation == "wrong_voice_row":
            value["chunks"][1]["voice_row"] += 1
        else:
            value["chunks"][1]["samples"] -= 1
    else:
        path = paths[5]
        value = json.loads(path.read_text())
        if mutation == "fixed_window":
            value["timestamps"] = False
        elif mutation == "early_end":
            value["segments"][0]["source_frame_consumed_end"] = 15000
        else:
            value["wav_sha256"] = "wrong"
    _write_json(path, value)
    with pytest.raises(ValueError):
        evaluator.evaluate(*paths)
