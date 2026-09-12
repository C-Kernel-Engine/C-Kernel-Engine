#!/usr/bin/env python3
"""Certify deterministic Parakeet overlap reconciliation on retained audio."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from certify_parakeet_long_audio_v8 import word_error_rate


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_report(report: dict[str, Any], label: str) -> None:
    require(report.get("status") == "pass", f"{label} run did not pass")
    require(all((report.get("checks") or {}).values()), f"{label} contains a failed check")
    coverage = report.get("source_coverage") or {}
    require(coverage.get("complete") is True, f"{label} source coverage is incomplete")
    require(coverage.get("start_seconds") == 0.0, f"{label} does not start at zero")
    require(
        coverage.get("completed_windows") == coverage.get("planned_windows"),
        f"{label} did not complete every window",
    )
    windows = report.get("windows") or []
    require(bool(windows), f"{label} contains no windows")
    require(windows[0]["ownership_frame_start"] == 0, f"{label} ownership does not start at zero")
    require(
        windows[-1]["ownership_seconds_end"] == coverage["end_seconds"],
        f"{label} ownership does not reach the source end",
    )
    for index, window in enumerate(windows):
        require(window["index"] == index, f"{label} window indices are not contiguous")
        require(
            window["source_frames_consumed"] ==
            window["source_frame_end"] - window["source_frame_start"],
            f"{label} window {index} did not consume its complete input",
        )
        require(window["finite_encoder"], f"{label} window {index} encoder is non-finite")
        require(window["finite_first_logits"], f"{label} window {index} logits are non-finite")
        if index:
            require(
                windows[index - 1]["ownership_frame_end"] == window["ownership_frame_start"],
                f"{label} ownership has a gap or overlap at window {index}",
            )
    words = report.get("selected_words") or []
    require(bool(words), f"{label} contains no selected words")
    require(
        all(float(words[index]["start"]) >= float(words[index - 1]["start"])
            for index in range(1, len(words))),
        f"{label} selected-word timestamps are not monotonic",
    )


def certify(
    reference_path: Path,
    full_report_path: Path,
    first_chunked_path: Path,
    repeat_chunked_path: Path,
) -> dict[str, Any]:
    reference = reference_path.read_text(encoding="utf-8")
    full = json.loads(full_report_path.read_text(encoding="utf-8"))
    first = json.loads(first_chunked_path.read_text(encoding="utf-8"))
    repeat = json.loads(repeat_chunked_path.read_text(encoding="utf-8"))
    validate_report(first, "first")
    validate_report(repeat, "repeat")
    require(first["input"]["sha256"] == repeat["input"]["sha256"], "chunk runs used different audio")
    require(first["policy"] == repeat["policy"], "chunk runs used different policies")
    require(first["transcript"] == repeat["transcript"], "chunk transcript is not deterministic")
    require(first["selected_words"] == repeat["selected_words"], "chunk word trajectory is not deterministic")
    require(full.get("status") == "pass", "full-attention comparison run did not pass")
    require(
        full["inputs"]["audio"]["sha256"] == repeat["input"]["sha256"],
        "full and chunked runs used different audio",
    )
    full_transcript = full["decode"]["transcript"]
    chunked_transcript = repeat["transcript"]
    chunked_wer = word_error_rate(reference, chunked_transcript)
    full_wer = word_error_rate(reference, full_transcript)
    trajectory_wer = word_error_rate(full_transcript, chunked_transcript)
    require(chunked_wer <= 0.15, f"chunked WER {chunked_wer:.4f} exceeds 0.15")
    require(trajectory_wer <= 0.10, f"chunk/full WER {trajectory_wer:.4f} exceeds 0.10")
    return {
        "schema": "cke.parakeet.chunking_certification.v1",
        "status": "pass",
        "audio_sha256": repeat["input"]["sha256"],
        "duration_seconds": repeat["source_coverage"]["end_seconds"],
        "policy": repeat["policy"],
        "windows": len(repeat["windows"]),
        "selected_words": len(repeat["selected_words"]),
        "deterministic_transcript": True,
        "deterministic_word_trajectory": True,
        "chunked_word_error_rate": chunked_wer,
        "full_attention_word_error_rate": full_wer,
        "chunked_vs_full_word_error_rate": trajectory_wer,
        "performance": {
            "total_seconds": repeat["total_elapsed_seconds"],
            "real_time_factor": repeat["real_time_factor"],
            "process_cpu_seconds": repeat["process_cpu_seconds"],
            "peak_rss_bytes": repeat["peak_rss_bytes"],
        },
        "reports": {
            "full_attention": sha256_file(full_report_path),
            "first_chunked": sha256_file(first_chunked_path),
            "repeat_chunked": sha256_file(repeat_chunked_path),
        },
        "limitations": [
            "five-minute English boundary fixture",
            "decoder and encoder state reset for every window",
            "word ownership uses timestamp midpoint at the overlap midpoint",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--full-report", required=True, type=Path)
    parser.add_argument("--first-chunked", required=True, type=Path)
    parser.add_argument("--repeat-chunked", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = certify(args.reference, args.full_report, args.first_chunked, args.repeat_chunked)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
