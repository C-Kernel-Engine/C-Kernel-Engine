#!/usr/bin/env python3
"""Assemble inspectable Whisper word timings across checked audio windows.

This is a reporting boundary, not an audio editor. A window may decode audio
past its committed seek point; words in that overlap must not be silently
published twice. Ambiguous words remain visible for review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


# A conservative review trigger, not a calibrated word-duration model.
MAX_CONSERVATIVE_WORD_SPAN_SECONDS = 2.0
_SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _seconds(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def assemble(report: dict, alignments: list[dict], report_hash: str) -> dict:
    segments = report.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("run report has no audio windows")
    if len(segments) != len(alignments):
        raise ValueError("one alignment is required for every audio window")
    declared_count = report.get("windowing", {}).get("window_count")
    if declared_count is not None and declared_count != len(segments):
        raise ValueError("run report window count disagrees with its segments")
    if not isinstance(report_hash, str) or not _SHA256.fullmatch(report_hash):
        raise ValueError("run report hash is missing")
    audio_hash = report.get("wav_sha256")
    if not isinstance(audio_hash, str) or not _SHA256.fullmatch(audio_hash):
        raise ValueError("run report audio hash is missing")

    committed: list[dict] = []
    review: list[dict] = []
    expected_start = 0
    source_rate = None
    source_total = None
    for index, (segment, alignment) in enumerate(zip(segments, alignments)):
        if _integer(segment.get("index"), "window index") != index:
            raise ValueError("run report windows are out of order")
        audio = segment.get("encoder", {}).get("audio", {})
        rate = _integer(audio.get("source_sample_rate"), "source sample rate", minimum=1)
        total = _integer(audio.get("source_frames"), "source frame count", minimum=1)
        window_frames = _integer(audio.get("window_source_frames"), "window frame count", minimum=1)
        start = _integer(segment.get("source_frame_start"), "window start")
        if _integer(audio.get("window_start_frame"), "encoder window start") != start:
            raise ValueError("encoder audio start disagrees with run window")
        end = _integer(segment.get("source_frame_consumed_end"), "consumed end", minimum=1)
        window_end = _integer(segment.get("source_frame_window_end"), "window end", minimum=1)
        if (start != expected_start or not start < end <= window_end <= total or
                window_end != min(total, start + window_frames)):
            raise ValueError("audio windows have a gap or invalid consumed extent")
        if source_rate is not None and (rate != source_rate or total != source_total):
            raise ValueError("audio geometry changes between windows")
        source_rate, source_total = rate, total
        expected_start = end

        window = alignment.get("window", {})
        provenance = alignment.get("provenance", {})
        trace = segment.get("decoder", {}).get("attention_trace") or {}
        trace_hash = trace.get("sha256")
        if not isinstance(trace_hash, str) or not _SHA256.fullmatch(trace_hash):
            raise ValueError("run window has no valid attention trace hash")
        if (alignment.get("schema") != "cke.whisper_word_alignment.v1" or
                alignment.get("status") not in {"aligned", "partial"} or
                _integer(window.get("index"), "alignment window index") != index or
                provenance.get("trace_sha256") != trace_hash or
                provenance.get("run_report_sha256") != report_hash or
                provenance.get("audio_sha256") != audio_hash):
            raise ValueError("word alignment provenance disagrees with run window")
        if (window.get("source_sample_rate") != rate or
                window.get("source_frames") != window_frames or
                abs(_seconds(window.get("start_seconds"), "alignment start") -
                    start / rate) > 1 / rate):
            raise ValueError("word alignment audio geometry disagrees with run window")
        words = alignment.get("words")
        if not isinstance(words, list):
            raise ValueError("alignment has no word list")
        previous_word_start = start / rate
        for word_index, word in enumerate(words):
            if (not isinstance(word, dict) or not isinstance(word.get("text"), str) or
                    not word["text"] or word.get("audio_window") != index):
                raise ValueError("word has invalid text or window identity")
            item = dict(word)
            item["window_word_index"] = word_index
            reasons = []
            first, last = word.get("start_seconds"), word.get("end_seconds")
            if first is None or last is None:
                if first is not None or last is not None:
                    raise ValueError("word has only one timing boundary")
                reasons.append("missing_word_timing")
            else:
                first = _seconds(first, "word start")
                last = _seconds(last, "word end")
                # The analyzer rounds to milliseconds, so permit one source frame
                # of representation error at the physical window boundaries.
                if (first < start / rate - 1 / rate or
                        last > window_end / rate + 1 / rate or first >= last):
                    raise ValueError("word timing exceeds its audio window")
                if first < previous_word_start - 1 / rate:
                    raise ValueError("word timings run backward within an audio window")
                previous_word_start = first
                if last - first > MAX_CONSERVATIVE_WORD_SPAN_SECONDS:
                    reasons.append("long_alignment_span")
                if first >= end / rate - 1 / rate and end < total:
                    reasons.append("unconsumed_window_overlap")
                elif first < end / rate < last and end < total:
                    reasons.append("crosses_consumed_boundary")
            item["assembly_review_reasons"] = reasons
            if reasons:
                review.append(item)
            else:
                committed.append(item)

    if expected_start != source_total:
        raise ValueError("audio windows do not consume the complete recording")
    return {
        "schema": "cke.whisper_word_assembly.v1",
        "status": "REVIEW_REQUIRED" if review else "ASSEMBLED_UNCALIBRATED",
        "edit_ready": False,
        "words": committed,
        "review_words": review,
        "window_count": len(segments),
        "counts": {"committed": len(committed), "review": len(review)},
        "provenance": {"run_report_sha256": report_hash, "audio_sha256": audio_hash,
                       "trace_sha256_by_window": [
                           segment["decoder"]["attention_trace"]["sha256"]
                           for segment in segments]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True,
                        help="Original recording; its hash must match the run report")
    parser.add_argument("--alignment", type=Path, action="append", required=True,
                        help="One alignment JSON per window, in window order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.run_report.read_text())
    if sha256(args.audio) != report.get("wav_sha256"):
        raise ValueError("audio hash disagrees with run report")
    alignments = [json.loads(path.read_text()) for path in args.alignment]
    result = assemble(report, alignments, sha256(args.run_report))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['status']}: {result['counts']['committed']} committed, "
          f"{result['counts']['review']} review -> {args.output}")
    return 0 if not result["review_words"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
