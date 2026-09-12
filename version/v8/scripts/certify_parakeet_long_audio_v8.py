#!/usr/bin/env python3
"""Certify two complete Parakeet runs on the retained five-minute corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
TECHNICAL_TERMS = (
    "CKE", "kernel", "GEMV", "SwiGLU", "RMSNorm", "AVX-512",
    "GEMM", "softmax", "RoPE", "PyTorch",
)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def contains_phrase(haystack: list[str], phrase: list[str]) -> bool:
    return any(
        haystack[index:index + len(phrase)] == phrase
        for index in range(len(haystack) - len(phrase) + 1)
    )


def edit_distance(reference: list[str], candidate: list[str]) -> int:
    previous = list(range(len(candidate) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(candidate, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, candidate: str) -> float:
    expected = words(reference)
    if not expected:
        raise ValueError("reference transcript contains no words")
    return edit_distance(expected, words(candidate)) / len(expected)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def certify(
    manifest_path: Path,
    first_report_path: Path,
    repeat_report_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture = manifest["fixture"]
    reference_path = manifest_path.parent / fixture["reference"]
    reference_text = reference_path.read_text(encoding="utf-8")
    first = json.loads(first_report_path.read_text(encoding="utf-8"))
    repeat = json.loads(repeat_report_path.read_text(encoding="utf-8"))

    for label, report in (("first", first), ("repeat", repeat)):
        require(report.get("status") == "pass", f"{label} run did not pass")
        frontend = report.get("audio_frontend") or {}
        require(frontend.get("source_sample_rate") == 16000, f"{label} sample rate mismatch")
        require(frontend.get("source_channels") == 1, f"{label} channel count mismatch")
        require(
            abs(float(frontend.get("source_seconds", -1)) - float(fixture["duration_seconds"])) <= 1e-9,
            f"{label} duration mismatch",
        )
        consumed = frontend.get("source_frames_consumed")
        require(
            (label == "first" and consumed is None) or consumed == frontend.get("source_samples"),
            f"{label} did not consume every source frame",
        )
        require(all((report.get("checks") or {}).values()), f"{label} contains a failed check")
        require(
            int(report.get("peak_rss_bytes", 0)) <= 8 * 1024**3,
            f"{label} exceeded the eight-GiB evidence budget",
        )
        decode = report.get("decode") or {}
        require(decode.get("termination_reason") in (None, "encoder_exhausted"), f"{label} termination mismatch")
        timestamps = decode.get("timestamps") or []
        require(bool(timestamps), f"{label} has no timestamps")
        require(
            all(float(item["start"]) <= float(item["end"]) for item in timestamps),
            f"{label} contains a negative timestamp interval",
        )
        require(
            all(float(timestamps[index]["start"]) >= float(timestamps[index - 1]["start"])
                for index in range(1, len(timestamps))),
            f"{label} timestamps are not monotonic",
        )

    first_decode = first["decode"]
    repeat_decode = repeat["decode"]
    identity_fields = ("sequences", "durations", "transcript", "timestamps")
    for field in identity_fields:
        require(
            first_decode.get(field) == repeat_decode.get(field),
            f"five-minute repeat diverged in {field}",
        )
    require(
        first["inputs"]["audio"]["sha256"] == repeat["inputs"]["audio"]["sha256"],
        "five-minute runs used different audio",
    )

    transcript = str(repeat_decode["transcript"])
    reference_words = words(reference_text)
    candidate_words = words(transcript)
    terms = {}
    for term in TECHNICAL_TERMS:
        normalized_term = words(term)
        present = contains_phrase(reference_words, normalized_term)
        terms[term] = {
            "present_in_reference": present,
            "exact_in_candidate": contains_phrase(candidate_words, normalized_term) if present else None,
        }

    return {
        "schema": "cke.parakeet.five_minute_certification.v1",
        "status": "pass",
        "model": repeat["model"],
        "audio_sha256": repeat["inputs"]["audio"]["sha256"],
        "duration_seconds": repeat["audio_frontend"]["source_seconds"],
        "source_samples": repeat["audio_frontend"]["source_samples"],
        "word_error_rate": word_error_rate(reference_text, transcript),
        "reference_words": len(reference_words),
        "transcript_words": len(candidate_words),
        "technical_terms": terms,
        "deterministic_identity": {field: True for field in identity_fields},
        "steps": repeat_decode["steps"],
        "timestamps": len(repeat_decode["timestamps"]),
        "performance": {
            "frontend_seconds": repeat["frontend_elapsed_seconds"],
            "encoder_seconds": repeat["encoder_elapsed_seconds"],
            "total_seconds": repeat["total_elapsed_seconds"],
            "real_time_factor": repeat["real_time_factor"],
            "process_cpu_seconds": repeat["process_cpu_seconds"],
            "peak_rss_bytes": repeat["peak_rss_bytes"],
        },
        "reports": {
            "first": {"sha256": sha256_file(first_report_path)},
            "repeat": {"sha256": sha256_file(repeat_report_path)},
        },
        "limitations": [
            "full-attention evidence only",
            "five-minute English fixture only",
            "generated-circuit execution not yet implemented",
            "per-call allocation instrumentation not yet collected",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--first-report", required=True, type=Path)
    parser.add_argument("--repeat-report", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = certify(args.manifest, args.first_report, args.repeat_report)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
