#!/usr/bin/env python3
"""Run bounded Parakeet full-attention windows with deterministic reconciliation."""

from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import run_parakeet_native_v8 as native


@dataclass(frozen=True)
class Window:
    index: int
    start_frame: int
    end_frame: int
    ownership_start_frame: int
    ownership_end_frame: int


def plan_windows(
    total_frames: int,
    sample_rate: int,
    window_seconds: float,
    overlap_seconds: float,
) -> list[Window]:
    if total_frames <= 0 or sample_rate <= 0:
        raise ValueError("audio geometry must be positive")
    window_frames = round(window_seconds * sample_rate)
    overlap_frames = round(overlap_seconds * sample_rate)
    if window_frames <= 0 or overlap_frames < 0 or overlap_frames >= window_frames:
        raise ValueError("window must be positive and overlap must be in [0, window)")

    spans: list[tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(start + window_frames, total_frames)
        spans.append((start, end))
        if end == total_frames:
            break
        start = end - overlap_frames

    boundaries = [0]
    for previous, current in zip(spans, spans[1:]):
        overlap_start = current[0]
        overlap_end = previous[1]
        boundaries.append((overlap_start + overlap_end) // 2)
    boundaries.append(total_frames)
    return [
        Window(index, start, end, boundaries[index], boundaries[index + 1])
        for index, (start, end) in enumerate(spans)
    ]


def timestamp_tokens_to_words(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    punctuation = set("?'¡¿-:,%/.!")
    for item in tokens:
        text = str(item["token"])
        start = float(item["start"])
        end = float(item["end"])
        starts_word = bool(text[:1].isspace())
        attaches = bool(words) and (not starts_word or text in punctuation)
        if attaches:
            words[-1]["text"] += text
            words[-1]["end"] = max(float(words[-1]["end"]), end)
        else:
            words.append({"text": text, "start": start, "end": end})
    return words


def select_owned_words(
    words: list[dict[str, Any]],
    *,
    window_start_seconds: float,
    ownership_start_seconds: float,
    ownership_end_seconds: float,
    final_window: bool,
) -> list[dict[str, Any]]:
    selected = []
    for word in words:
        global_start = window_start_seconds + float(word["start"])
        global_end = window_start_seconds + float(word["end"])
        midpoint = (global_start + global_end) * 0.5
        owned = midpoint >= ownership_start_seconds and (
            midpoint < ownership_end_seconds or
            (final_window and midpoint <= ownership_end_seconds)
        )
        if owned:
            selected.append({
                "text": word["text"],
                "start": global_start,
                "end": global_end,
            })
    return selected


def _write_window(
    path: Path,
    frames: bytes,
    *,
    channels: int,
    sample_width: int,
    sample_rate: int,
) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(sample_rate)
        output.writeframes(frames)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    usage_started = resource.getrusage(resource.RUSAGE_SELF)
    with wave.open(str(args.wav), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        total_frames = source.getnframes()
        compression = source.getcomptype()
        if sample_width != 2 or compression != "NONE":
            raise ValueError("long-audio Parakeet requires uncompressed PCM16 WAV")
        source_pcm = source.readframes(total_frames)
    bytes_per_frame = channels * sample_width
    if len(source_pcm) != total_frames * bytes_per_frame:
        raise RuntimeError("WAV payload is shorter than its declared frame count")

    windows = plan_windows(
        total_frames, sample_rate, args.window_seconds, args.overlap_seconds,
    )
    kernels = native.Kernels(args.engine, args.audio_lib)
    session = native.ParakeetSession(args.model, kernels)
    report: dict[str, Any] = {
        "schema": "cke.parakeet.long_audio.v1",
        "status": "running",
        "cke": native.git_identity(native.ROOT),
        "model": {
            "id": "nvidia/parakeet-tdt-0.6b-v3",
            "revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
            "metadata": session.model_metadata_provenance(),
        },
        "runtime": kernels.provenance(),
        "thread_policy": kernels.thread_policy(),
        "host": native.host_identity(),
        "input": {
            **native.file_identity(args.wav),
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "source_frames": total_frames,
            "source_seconds": total_frames / sample_rate,
        },
        "policy": {
            "kind": "overlapping_full_attention_windows",
            "window_seconds": args.window_seconds,
            "overlap_seconds": args.overlap_seconds,
            "state": "reset_each_window",
            "ownership": "word_midpoint_at_overlap_midpoint",
            "timestamp_offset": "source_window_start",
        },
        "windows": [],
    }
    selected_words: list[dict[str, Any]] = []
    prior_elapsed = 0.0
    prior_cpu = {"user": 0.0, "system": 0.0}
    if args.resume and args.output and args.output.is_file():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("schema") != report["schema"]:
            raise ValueError("resume report schema mismatch")
        if prior.get("input", {}).get("sha256") != report["input"]["sha256"]:
            raise ValueError("resume report input mismatch")
        if prior.get("policy") != report["policy"]:
            raise ValueError("resume report window policy mismatch")
        completed = prior.get("windows") or []
        if len(completed) > len(windows):
            raise ValueError("resume report has too many completed windows")
        for expected, actual in zip(windows, completed):
            if (
                actual.get("index") != expected.index or
                actual.get("source_frame_start") != expected.start_frame or
                actual.get("source_frame_end") != expected.end_frame
            ):
                raise ValueError("resume report window geometry mismatch")
        report["windows"] = completed
        selected_words = list(prior.get("selected_words") or [])
        prior_elapsed = float(prior.get("total_elapsed_seconds", 0.0))
        prior_cpu = prior.get("process_cpu_seconds") or prior_cpu
        report["resume_count"] = int(prior.get("resume_count", 0)) + 1
    try:
        with tempfile.TemporaryDirectory(prefix="cke-parakeet-long-") as temp_text:
            temp = Path(temp_text)
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
            for window in windows[len(report["windows"]):]:
                window_started = time.perf_counter()
                path = temp / f"window-{window.index:04d}.wav"
                begin = window.start_frame * bytes_per_frame
                end = window.end_frame * bytes_per_frame
                _write_window(
                    path, source_pcm[begin:end], channels=channels,
                    sample_width=sample_width, sample_rate=sample_rate,
                )
                frontend_started = time.perf_counter()
                features, live_frames, frontend = session.frontend(path)
                frontend_seconds = time.perf_counter() - frontend_started
                encoder_started = time.perf_counter()
                encoded = session.encode(features, live_frames)
                encoder_seconds = time.perf_counter() - encoder_started
                decoder_started = time.perf_counter()
                sequences, durations, first_logits = session.decode(encoded)
                decoder_seconds = time.perf_counter() - decoder_started
                timestamps = native.decode_token_timestamps(
                    tokenizer, sequences, durations,
                    blank_token_id=int(session.config["blank_token_id"]),
                    pad_token_id=int(session.config["pad_token_id"]),
                )
                words = timestamp_tokens_to_words(timestamps)
                owned = select_owned_words(
                    words,
                    window_start_seconds=window.start_frame / sample_rate,
                    ownership_start_seconds=window.ownership_start_frame / sample_rate,
                    ownership_end_seconds=window.ownership_end_frame / sample_rate,
                    final_window=window.index == len(windows) - 1,
                )
                selected_words.extend(owned)
                report["windows"].append({
                    "index": window.index,
                    "source_frame_start": window.start_frame,
                    "source_frame_end": window.end_frame,
                    "ownership_frame_start": window.ownership_start_frame,
                    "ownership_frame_end": window.ownership_end_frame,
                    "source_seconds_start": window.start_frame / sample_rate,
                    "source_seconds_end": window.end_frame / sample_rate,
                    "ownership_seconds_start": window.ownership_start_frame / sample_rate,
                    "ownership_seconds_end": window.ownership_end_frame / sample_rate,
                    "source_frames_consumed": frontend["source_frames_consumed"],
                    "encoder_frames": int(encoded.shape[0]),
                    "decode_steps": int(sequences.size),
                    "termination_reason": "encoder_exhausted",
                    "finite_encoder": bool(np.isfinite(encoded).all()),
                    "finite_first_logits": bool(np.isfinite(first_logits).all()),
                    "selected_words": len(owned),
                    "transcript": tokenizer.decode(sequences.tolist(), skip_special_tokens=True),
                    "elapsed_seconds": time.perf_counter() - window_started,
                    "frontend_seconds": frontend_seconds,
                    "encoder_seconds": encoder_seconds,
                    "decoder_seconds": decoder_seconds,
                })
                if args.output:
                    report["selected_words"] = selected_words
                    report["transcript"] = "".join(
                        str(word["text"]) for word in selected_words
                    ).strip()
                    report["source_coverage"] = {
                        "start_seconds": 0.0,
                        "end_seconds": total_frames / sample_rate,
                        "complete": len(report["windows"]) == len(windows),
                        "planned_windows": len(windows),
                        "completed_windows": len(report["windows"]),
                    }
                    report["total_elapsed_seconds"] = (
                        prior_elapsed + time.perf_counter() - started
                    )
                    _write_report(args.output, report)
    except BaseException as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        session.close()
        report["selected_words"] = selected_words
        report["transcript"] = "".join(str(word["text"]) for word in selected_words).strip()
        report["source_coverage"] = {
            "start_seconds": 0.0,
            "end_seconds": total_frames / sample_rate,
            "complete": len(report["windows"]) == len(windows),
            "planned_windows": len(windows),
            "completed_windows": len(report["windows"]),
        }
        report["total_elapsed_seconds"] = prior_elapsed + time.perf_counter() - started
        report["real_time_factor"] = report["total_elapsed_seconds"] / (total_frames / sample_rate)
        usage_finished = resource.getrusage(resource.RUSAGE_SELF)
        report["process_cpu_seconds"] = {
            "user": float(prior_cpu.get("user", 0.0)) + usage_finished.ru_utime - usage_started.ru_utime,
            "system": float(prior_cpu.get("system", 0.0)) + usage_finished.ru_stime - usage_started.ru_stime,
        }
        report["peak_rss_bytes"] = int(usage_finished.ru_maxrss) * 1024
        if report["status"] == "running":
            finite = all(
                window["finite_encoder"] and window["finite_first_logits"]
                for window in report["windows"]
            )
            monotonic = all(
                float(selected_words[index]["start"]) >= float(selected_words[index - 1]["start"])
                for index in range(1, len(selected_words))
            )
            report["checks"] = {
                "all_windows_completed": len(report["windows"]) == len(windows),
                "source_coverage_exact": (
                    windows[0].ownership_start_frame == 0 and
                    windows[-1].ownership_end_frame == total_frames and
                    all(windows[index].ownership_end_frame == windows[index + 1].ownership_start_frame
                        for index in range(len(windows) - 1))
                ),
                "finite_outputs": finite,
                "monotonic_word_timestamps": monotonic,
                "nonempty_transcript": bool(report["transcript"]),
            }
            report["status"] = "pass" if all(report["checks"].values()) else "fail"
        if args.output:
            _write_report(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--engine", default=Path("build/libckernel_engine.so"), type=Path)
    parser.add_argument("--audio-lib", default=Path("build/libckernel_audio.so"), type=Path)
    parser.add_argument("--window-seconds", default=300.0, type=float)
    parser.add_argument("--overlap-seconds", default=30.0, type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
