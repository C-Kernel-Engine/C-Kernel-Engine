#!/usr/bin/env python3
"""Run and record native long-audio Cohere transcription through generated components."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import time
import wave
from pathlib import Path
from typing import Any


WINDOW = re.compile(
    r"^window=(\d+) source_frames=(\d+):(\d+) frontend=([0-9.]+)s "
    r"encoder=([0-9.]+)s decoder=([0-9.]+)s encoder_frames=(\d+) tokens=(\d+)$",
    re.MULTILINE,
)
TOKENS = re.compile(r"^window_token_ids\[(\d+)\]=([0-9]+(?:,[0-9]+)*)$", re.MULTILINE)
COMPLETION = re.compile(
    r"^completed_windows=(\d+) source_frames=(\d+) consumed_frames=(\d+)$",
    re.MULTILINE,
)
RUNTIME_PROFILE = re.compile(
    r"^audio_runtime_profile requested_threads=(\S+) actual_threads=(-?\d+)$",
    re.MULTILINE,
)
GEMM_PROFILE = re.compile(
    r"^gemm_profile window=(\d+) M=(\d+) N=(\d+) K=(\d+) "
    r"active_threads=(\d+) mode=(serial|parallel) calls=(\d+) elapsed_ns=(\d+)$",
    re.MULTILINE,
)
GEMM_PROFILE_SUMMARY = re.compile(
    r"^gemm_profile_summary window=(\d+) shapes=(\d+) overflow_calls=(\d+)$",
    re.MULTILINE,
)
THREADPOOL_PROFILE = re.compile(
    r"^threadpool_profile window=(\d+) dispatches=(\d+) total_ns=(\d+) "
    r"main_work_ns=(\d+) completion_wait_ns=(\d+)$",
    re.MULTILINE,
)


def identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def load_plan(path: Path) -> tuple[int, list[tuple[int, int]]]:
    rows = path.read_text(encoding="ascii").splitlines()
    if not rows:
        raise ValueError("native segment plan is empty")
    header = rows[0].split()
    if len(header) != 3 or header[0] != "cke_audio_segments_v1":
        raise ValueError("native segment plan has an unsupported schema")
    sample_rate, count = int(header[1]), int(header[2])
    if sample_rate <= 0 or count <= 0 or len(rows) != count + 1:
        raise ValueError("native segment plan count is invalid")
    segments: list[tuple[int, int]] = []
    previous_end = 0
    for index, row in enumerate(rows[1:]):
        values = row.split()
        if len(values) != 2:
            raise ValueError(f"native segment {index} is malformed")
        start, end = map(int, values)
        if start < previous_end or end <= start:
            raise ValueError(f"native segment {index} has invalid or overlapping bounds")
        segments.append((start, end))
        previous_end = end
    return sample_rate, segments


def parse_native(stderr: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    timing_matches = list(WINDOW.finditer(stderr))
    token_matches = list(TOKENS.finditer(stderr))
    timings = {
        int(match.group(1)): {
            "index": int(match.group(1)),
            "start_frame": int(match.group(2)),
            "end_frame": int(match.group(3)),
            "frontend_seconds": float(match.group(4)),
            "encoder_seconds": float(match.group(5)),
            "decoder_seconds": float(match.group(6)),
            "encoder_frames": int(match.group(7)),
            "generated_tokens": int(match.group(8)),
        }
        for match in timing_matches
    }
    token_rows = {
        int(match.group(1)): [int(value) for value in match.group(2).split(",")]
        for match in token_matches
    }
    completion_match = COMPLETION.search(stderr)
    if completion_match is None:
        raise ValueError("native host did not publish completion accounting")
    completion = {
        "completed_windows": int(completion_match.group(1)),
        "source_frames": int(completion_match.group(2)),
        "consumed_frames": int(completion_match.group(3)),
    }
    if len(timings) != len(timing_matches) or len(token_rows) != len(token_matches):
        raise ValueError("native evidence contains duplicate window identities")
    if set(timings) != set(token_rows):
        raise ValueError("native timing and token window inventories differ")
    if sorted(timings) != list(range(len(timings))):
        raise ValueError("native window inventory is not contiguous")
    windows = []
    for index in sorted(timings):
        row = timings[index]
        row["generated_token_ids"] = token_rows[index]
        if row["generated_tokens"] != len(token_rows[index]):
            raise ValueError(f"native window {index} token count is inconsistent")
        windows.append(row)
    return windows, completion


def parse_performance_profile(stderr: str, expected_windows: int) -> dict[str, Any]:
    runtime_matches = list(RUNTIME_PROFILE.finditer(stderr))
    if len(runtime_matches) != 1:
        raise ValueError("native host did not publish exactly one runtime profile")
    entries = [
        {
            "window": int(match.group(1)),
            "M": int(match.group(2)),
            "N": int(match.group(3)),
            "K": int(match.group(4)),
            "active_threads": int(match.group(5)),
            "mode": match.group(6),
            "calls": int(match.group(7)),
            "elapsed_ns": int(match.group(8)),
        }
        for match in GEMM_PROFILE.finditer(stderr)
    ]
    summary_matches = list(GEMM_PROFILE_SUMMARY.finditer(stderr))
    summaries = {
        int(match.group(1)): {
            "shapes": int(match.group(2)),
            "overflow_calls": int(match.group(3)),
        }
        for match in summary_matches
    }
    threadpool_matches = list(THREADPOOL_PROFILE.finditer(stderr))
    threadpool = {
        int(match.group(1)): {
            "dispatches": int(match.group(2)),
            "total_ns": int(match.group(3)),
            "main_work_ns": int(match.group(4)),
            "completion_wait_ns": int(match.group(5)),
        }
        for match in threadpool_matches
    }
    if (len(summary_matches) != len(summaries) or len(summaries) != expected_windows
            or set(summaries) != set(range(expected_windows))):
        raise ValueError("native GEMM profile summary inventory is incomplete")
    if (len(threadpool_matches) != len(threadpool) or len(threadpool) != expected_windows
            or set(threadpool) != set(range(expected_windows))):
        raise ValueError("native thread-pool profile inventory is incomplete")
    shape_keys = {
        (entry["window"], entry["M"], entry["N"], entry["K"],
         entry["active_threads"], entry["mode"])
        for entry in entries
    }
    if len(shape_keys) != len(entries):
        raise ValueError("native GEMM profile contains duplicate shape identities")
    for window in range(expected_windows):
        observed = sum(entry["window"] == window for entry in entries)
        if observed != summaries[window]["shapes"]:
            raise ValueError(f"native GEMM profile shape count differs for window {window}")
        if summaries[window]["overflow_calls"]:
            raise ValueError(f"native GEMM profile overflowed for window {window}")
    requested = runtime_matches[0].group(1)
    return {
        "instrumented": True,
        "requested_threads": requested if requested == "auto" else int(requested),
        "actual_threads": int(runtime_matches[0].group(2)),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "gemm_shapes": entries,
        "windows": [summaries[index] for index in range(expected_windows)],
        "threadpool_windows": [threadpool[index] for index in range(expected_windows)],
        "total_profiled_calls": sum(entry["calls"] for entry in entries),
        "total_profiled_elapsed_ns": sum(entry["elapsed_ns"] for entry in entries),
    }


def load_reference(
    path: Path, segments: list[tuple[int, int]], input_sha256: str
) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("status") != "PASS" or not isinstance(document.get("windows"), list):
        raise ValueError("reference report is not a passing long-audio result")
    reference_input = document.get("input")
    if (
        not isinstance(reference_input, dict)
        or reference_input.get("sha256") != input_sha256
    ):
        raise ValueError("reference report input identity does not match the audio")
    windows = document["windows"]
    if len(windows) != len(segments):
        raise ValueError("reference report window count does not match the segment plan")
    normalized = []
    for index, (row, segment) in enumerate(zip(windows, segments)):
        if not isinstance(row, dict) or row.get("index") != index:
            raise ValueError(f"reference window {index} has an invalid identity")
        if (row.get("start_frame"), row.get("end_frame")) != segment:
            raise ValueError(f"reference window {index} does not match the segment plan")
        tokens = row.get("generated_token_ids")
        if not isinstance(tokens, list) or not tokens or not all(
            isinstance(token, int) and token >= 0 for token in tokens
        ):
            raise ValueError(f"reference window {index} lacks a token trajectory")
        normalized.append({
            "index": index,
            "start_frame": segment[0],
            "end_frame": segment[1],
            "generated_token_ids": tokens,
        })
    return normalized


def validate_runtime_bundle(runtime_dir: Path) -> dict[str, Any]:
    path = runtime_dir / ".ck_runtime_bundle.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("inputs", {}).get("schema") != "ck-v8-runtime-bundle-v2":
        raise ValueError(f"{runtime_dir}: unsupported generated runtime bundle")
    outputs = document.get("outputs")
    required = {"libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so"}
    if not isinstance(outputs, dict) or not required.issubset(outputs):
        raise ValueError(f"{runtime_dir}: generated runtime bundle is incomplete")
    for name in sorted(required):
        expected = outputs.get(name)
        if not isinstance(expected, dict):
            raise ValueError(f"{runtime_dir}: invalid runtime identity for {name}")
        actual = identity(runtime_dir / name)
        if (
            actual["bytes"] != expected.get("size")
            or actual["sha256"] != expected.get("sha256")
        ):
            raise ValueError(f"{runtime_dir}: stale generated runtime output: {name}")
    return document


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        raise ValueError("source revision must be a full lowercase Git commit")
    paths = {
        "native_host": args.host.resolve(),
        "native_host_source": args.host_source.resolve(),
        "encoder_model_library": args.encoder_runtime.resolve() / "libmodel.so",
        "encoder_weights": args.encoder_runtime.resolve() / "weights.bump",
        "encoder_manifest_map": args.encoder_runtime.resolve() / "weights_manifest.map",
        "encoder_runtime_bundle": args.encoder_runtime.resolve() / ".ck_runtime_bundle.json",
        "decoder_model_library": args.decoder_runtime.resolve() / "libmodel.so",
        "decoder_weights": args.decoder_runtime.resolve() / "weights.bump",
        "decoder_manifest_map": args.decoder_runtime.resolve() / "weights_manifest.map",
        "decoder_runtime_bundle": args.decoder_runtime.resolve() / ".ck_runtime_bundle.json",
        "input": args.input.resolve(),
        "segment_plan": args.segment_plan.resolve(),
        "reference_report": args.reference_report.resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing native long-audio inputs: " + ", ".join(missing))
    validate_runtime_bundle(args.encoder_runtime.resolve())
    validate_runtime_bundle(args.decoder_runtime.resolve())
    sample_rate, segments = load_plan(paths["segment_plan"])
    input_identity = identity(paths["input"])
    reference_windows = load_reference(
        paths["reference_report"], segments, input_identity["sha256"]
    )
    with wave.open(str(paths["input"]), "rb") as source:
        if source.getframerate() != sample_rate:
            raise ValueError("segment plan sample rate does not match input")
        source_frames = source.getnframes()
    if segments[-1][1] > source_frames:
        raise ValueError("segment plan exceeds input frame count")
    command = [
        str(paths["native_host"]),
        str(paths["encoder_model_library"]),
        str(paths["encoder_weights"]),
        str(paths["encoder_manifest_map"]),
        str(paths["decoder_model_library"]),
        str(paths["decoder_weights"]),
        str(paths["decoder_manifest_map"]),
        str(paths["input"]),
        str(paths["segment_plan"]),
    ]
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("PYTHON") and key not in {"VIRTUAL_ENV", "CONDA_PREFIX"}
    }
    if args.profile:
        environment["CK_AUDIO_PERF_PROFILE"] = "1"
    started = time.perf_counter()
    completed = subprocess.run(
        command, env=environment, text=True, capture_output=True,
        timeout=args.timeout, check=False,
    )
    elapsed = time.perf_counter() - started
    windows, coverage = parse_native(completed.stderr) if completed.returncode == 0 else ([], {})
    performance_profile = (
        parse_performance_profile(completed.stderr, len(segments))
        if completed.returncode == 0 and args.profile else {"instrumented": False}
    )
    expected_consumed = sum(end - start for start, end in segments)
    token_mismatches = [
        row["index"] for row, reference in zip(windows, reference_windows)
        if row["generated_token_ids"] != reference["generated_token_ids"]
    ] if len(windows) == len(reference_windows) else list(range(len(reference_windows)))
    checks = {
        "native_exit_zero": completed.returncode == 0,
        "completed_every_segment": len(windows) == len(segments),
        "completion_inventory_exact": coverage.get("completed_windows") == len(segments),
        "segment_geometry_exact": [
            (row["start_frame"], row["end_frame"]) for row in windows
        ] == segments,
        "source_frame_identity": coverage.get("source_frames") == source_frames,
        "consumed_frame_accounting": coverage.get("consumed_frames") == expected_consumed,
        "all_windows_produced_tokens": bool(windows) and all(
            row["generated_token_ids"] for row in windows
        ),
        "reference_token_trajectories_exact": not token_mismatches,
        "python_environment_removed": all(
            not key.startswith("PYTHON") for key in environment
        ),
    }
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "schema": "cke.v8.cohere_generated_long_audio",
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "native persistent generated-component long-audio transcription",
        "source_revision": args.source_revision,
        "checks": checks,
        "identity": {
            name: input_identity if name == "input" else identity(path)
            for name, path in paths.items()
        },
        "policy": {
            "segment_schema": "cke_audio_segments_v1",
            "state": "generated_decoder_kv_and_position_state_reset_before_each_segment",
            "model_libraries": "loaded_once_for_all_segments",
            "quality_oracle": "exact per-window token trajectory from pinned reference report",
        },
        "coverage": {
            **coverage,
            "planned_windows": len(segments),
            "expected_consumed_frames": expected_consumed,
            "intentionally_skipped_frames": source_frames - expected_consumed,
        },
        "windows": windows,
        "reference": {
            "mismatching_window_indices": token_mismatches,
        },
        "transcript": completed.stdout.strip(),
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": int(usage.ru_maxrss) * 1024,
        "performance_profile": performance_profile,
        "measurement_policy": {
            "instrumentation": "enabled" if args.profile else "disabled",
            "window_temperature": "first_after_load_then_persistent_loaded_session",
            "timing_clock": "CLOCK_MONOTONIC_in_native_host",
            "profile_scope": "generated_encoder_fp32_gemm_and_threadpool",
        },
        "stderr": completed.stderr,
        "not_certified": [
            "timestamps", "resampling", "multilingual quality", "cancellation",
            "performance",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, type=Path)
    parser.add_argument("--host-source", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--encoder-runtime", required=True, type=Path)
    parser.add_argument("--decoder-runtime", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--segment-plan", required=True, type=Path)
    parser.add_argument("--reference-report", required=True, type=Path)
    parser.add_argument("--timeout", default=7200.0, type=float)
    parser.add_argument(
        "--profile", action="store_true",
        help="collect instrumented FP32 GEMM shape and elapsed-time evidence",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as error:
        report = {
            "schema": "cke.v8.cohere_generated_long_audio",
            "schema_version": 1,
            "status": "ERROR",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
