#!/usr/bin/env python3
"""Certify repeatable native Cohere Transcribe long-audio reports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_runner():
    path = Path(__file__).with_name("run_cohere_transcribe_long_audio_v8.py")
    spec = importlib.util.spec_from_file_location("cke_cohere_long_audio", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _read(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: report must be an object")
    return document


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def certify(
    candidate_path: Path,
    repeat_path: Path,
    reference_path: Path,
    maximum_word_error_rate: float,
) -> dict[str, Any]:
    if candidate_path.resolve() == repeat_path.resolve():
        raise ValueError("candidate and repeat must be distinct report files")
    candidate, repeat = _read(candidate_path), _read(repeat_path)
    reference, reference_identity = runner.load_reference_transcript(reference_path)
    for name, report in (("candidate", candidate), ("repeat", repeat)):
        if report.get("schema") != "cke.v8.cohere_transcribe.long_audio":
            raise ValueError(f"{name}: wrong report schema")
    runtime_loaded = all(
        bool(report.get("runtime", {}).get(library, {}).get("present_in_process_maps"))
        for report in (candidate, repeat)
        for library in ("engine", "audio")
    )
    matching_identity = all(
        candidate.get(field) == repeat.get(field)
        for field in ("model", "input", "runtime", "policy", "speech_segments")
    )
    candidate_ids = [item.get("generated_token_ids") for item in candidate.get("windows", [])]
    repeat_ids = [item.get("generated_token_ids") for item in repeat.get("windows", [])]
    transcript = str(candidate.get("transcript", ""))
    comparison = runner.word_error_rate(reference, transcript)
    comparison["maximum_word_error_rate"] = maximum_word_error_rate
    comparison["within_limit"] = comparison["word_error_rate"] <= maximum_word_error_rate
    checks = {
        "candidate_passed": candidate.get("status") == "PASS",
        "repeat_passed": repeat.get("status") == "PASS",
        "matching_artifact_runtime_input_policy": matching_identity,
        "loaded_libraries_verified": runtime_loaded,
        "generated_token_ids_exact": candidate_ids == repeat_ids and bool(candidate_ids),
        "selected_token_trajectory_exact": (
            candidate.get("selected_tokens") == repeat.get("selected_tokens")
            and bool(candidate.get("selected_tokens"))
        ),
        "transcript_exact": transcript == repeat.get("transcript") and bool(transcript),
        "no_repeated_phrase_loops": not any(
            int(item["ngram_words"]) > 1
            for item in runner.repeated_ngram_runs(transcript)
        ),
        "reference_word_error_rate": comparison["within_limit"],
    }
    return {
        "schema": "cke.v8.cohere_transcribe.long_audio_certification",
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "candidate": runner.native.parakeet.file_identity(candidate_path),
        "repeat": runner.native.parakeet.file_identity(repeat_path),
        "reference": reference_identity,
        "checks": checks,
        "reference_comparison": comparison,
        "performance": {
            "candidate_elapsed_seconds": candidate.get("total_elapsed_seconds"),
            "repeat_elapsed_seconds": repeat.get("total_elapsed_seconds"),
            "candidate_real_time_factor": candidate.get("real_time_factor"),
            "repeat_real_time_factor": repeat.get("real_time_factor"),
            "candidate_peak_rss_bytes": candidate.get("peak_rss_bytes"),
            "repeat_peak_rss_bytes": repeat.get("peak_rss_bytes"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--repeat", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--max-word-error-rate", default=0.10, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = certify(
            args.candidate, args.repeat, args.reference, args.max_word_error_rate,
        )
        _write(args.output, report)
    except Exception as error:
        report = {
            "schema": "cke.v8.cohere_transcribe.long_audio_certification",
            "schema_version": 1,
            "status": "ERROR",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        _write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
