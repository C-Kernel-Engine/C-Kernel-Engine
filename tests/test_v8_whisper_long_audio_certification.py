from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "certify_whisper_long_audio_v8.py"
MANIFEST = (
    ROOT / "version" / "v8" / "test_assets" / "whisper_long_audio" / "corpus.json"
)
SPEC = importlib.util.spec_from_file_location("whisper_long_audio", SCRIPT)
assert SPEC and SPEC.loader
CERTIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CERTIFIER)


def _report(transcript: str) -> dict:
    segments = []
    for index in range(10):
        start = index * 30.0
        end = (index + 1) * 30.0
        segments.append(
            {
                "index": index,
                "start_seconds": start,
                "end_seconds": end,
                "timestamp_events": [
                    {"global_seconds": start},
                    {"global_seconds": end},
                ],
            }
        )
    return {
        "status": "ok",
        "language": "en",
        "task": "transcribe",
        "timestamps": True,
        "segments": segments,
        "decoder": {
            "stop": "segment_complete",
            "transcript_text": transcript,
        },
    }


def test_long_audio_manifest_pins_all_whisper_sizes_and_fixture() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest["models"]) == {"tiny", "base", "small", "medium", "large-v3"}
    audio = MANIFEST.parent / manifest["fixture"]["audio"]
    assert CERTIFIER._sha256(audio) == manifest["fixture"]["flac_sha256"]
    assert manifest["fixture"]["duration_seconds"] == 300.0


def test_nightly_matrix_keeps_every_whisper_size_non_skippable() -> None:
    workflow = (ROOT / ".github" / "workflows" / "nightly.yml").read_text(
        encoding="utf-8"
    )
    job = workflow.split("  whisper-long-audio:", 1)[1]
    for model in ("tiny", "base", "small", "medium", "large-v3"):
        assert f"- model: {model}" in job
    assert "continue-on-error" not in job
    assert "if-no-files-found: error" in job


def test_long_audio_certification_accepts_complete_matching_report(
    tmp_path: Path,
) -> None:
    reference = (MANIFEST.parent / "kernel2_mic1_5min.txt").read_text(encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report(reference)), encoding="utf-8")
    summary = CERTIFIER.certify(MANIFEST, report, "base")
    assert summary["status"] == "pass"
    assert summary["word_error_rate"] == 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["segments"].pop(), "audio windows"),
        (
            lambda report: report["segments"][4].update(end_seconds=121.0),
            "segments are not contiguous",
        ),
        (
            lambda report: report["decoder"].update(transcript_text="The"),
            "transcript is too short",
        ),
    ],
)
def test_long_audio_certification_fails_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    reference = (MANIFEST.parent / "kernel2_mic1_5min.txt").read_text(encoding="utf-8")
    payload = _report(reference)
    mutation(payload)
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        CERTIFIER.certify(MANIFEST, report, "tiny")
