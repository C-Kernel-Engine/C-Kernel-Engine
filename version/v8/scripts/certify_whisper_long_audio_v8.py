#!/usr/bin/env python3
"""Certify a CKE Whisper report against the published long-audio corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _edit_distance(reference: list[str], candidate: list[str]) -> int:
    previous = list(range(len(candidate) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(candidate, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, candidate: str) -> float:
    expected = _words(reference)
    if not expected:
        raise ValueError("reference transcript contains no words")
    return _edit_distance(expected, _words(candidate)) / len(expected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def certify(
    manifest_path: Path,
    report_path: Path,
    model_name: str,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture = manifest["fixture"]
    models = manifest["models"]
    _require(model_name in models, f"model is not in corpus: {model_name}")
    policy = models[model_name]

    asset_root = manifest_path.parent
    audio_path = asset_root / fixture["audio"]
    reference_path = asset_root / fixture["reference"]
    _require(audio_path.is_file(), f"missing corpus audio: {audio_path}")
    _require(reference_path.is_file(), f"missing reference: {reference_path}")
    _require(
        _sha256(audio_path) == fixture["flac_sha256"],
        "corpus audio hash does not match corpus.json",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("status") == "ok", "Whisper report status is not ok")
    _require(report.get("language") == fixture["language"], "language mismatch")
    _require(report.get("task") == fixture["task"], "task mismatch")
    _require(report.get("timestamps") is True, "timestamp decoding is required")

    segments = report.get("segments") or []
    _require(
        len(segments) >= int(fixture["minimum_windows"]),
        f"only {len(segments)} audio windows completed",
    )
    starts: list[float] = []
    ends: list[float] = []
    timestamps: list[float] = []
    for index, segment in enumerate(segments):
        _require(segment.get("index") == index, "segment indices are not contiguous")
        start = float(segment["start_seconds"])
        end = float(segment["end_seconds"])
        _require(end > start, f"segment {index} made no forward progress")
        starts.append(start)
        ends.append(end)
        segment_timestamps = [
            float(event["global_seconds"])
            for event in segment.get("timestamp_events", [])
        ]
        _require(
            all(
                segment_timestamps[i] >= segment_timestamps[i - 1]
                for i in range(1, len(segment_timestamps))
            ),
            f"timestamps are not monotonic within segment {index}",
        )
        timestamps.extend(segment_timestamps)

    duration = float(fixture["duration_seconds"])
    _require(abs(starts[0]) <= 1e-9, "first segment does not start at zero")
    _require(abs(ends[-1] - duration) <= 0.05, "audio was not fully consumed")
    _require(
        all(starts[i] >= starts[i - 1] for i in range(1, len(starts))),
        "segment starts are not monotonic",
    )
    _require(
        all(abs(starts[i] - ends[i - 1]) <= 1e-6 for i in range(1, len(starts))),
        "segments are not contiguous",
    )
    _require(
        all(0.0 <= value <= duration + 0.05 for value in timestamps),
        "timestamp lies outside the audio duration",
    )
    # Adjacent Whisper windows intentionally overlap at a timestamp boundary,
    # so an event near the end of window N may follow an earlier boundary event
    # in window N+1. Segment coverage must be monotonic and contiguous above;
    # timestamp events must be monotonic only within their own decode window.

    decoder = report.get("decoder") or {}
    _require(decoder.get("stop") == "segment_complete", "long audio did not complete")
    transcript = str(decoder.get("transcript_text", ""))
    candidate_words = _words(transcript)
    _require(
        len(candidate_words) >= int(fixture["minimum_words"]),
        f"transcript is too short: {len(candidate_words)} words",
    )
    reference = reference_path.read_text(encoding="utf-8")
    wer = word_error_rate(reference, transcript)
    maximum_wer = float(policy["maximum_wer"])
    _require(wer <= maximum_wer, f"WER {wer:.4f} exceeds {maximum_wer:.4f}")

    summary = {
        "schema": "cke.whisper.long_audio_certification.v1",
        "status": "pass",
        "model": model_name,
        "checkpoint": policy["checkpoint"],
        "runner": policy["runner"],
        "word_error_rate": wer,
        "maximum_word_error_rate": maximum_wer,
        "reference_words": len(_words(reference)),
        "transcript_words": len(candidate_words),
        "windows": len(segments),
        "timestamps": len(timestamps),
        "duration_seconds": ends[-1],
        "audio_sha256": fixture["flac_sha256"],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = certify(args.manifest, args.report, args.model)
    encoded = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
