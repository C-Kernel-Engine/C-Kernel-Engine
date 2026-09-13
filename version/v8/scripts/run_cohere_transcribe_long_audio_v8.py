#!/usr/bin/env python3
"""Run Cohere Transcribe over bounded audio windows with DTW timing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import re
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_native():
    path = Path(__file__).with_name("run_cohere_transcribe_native_v8.py")
    spec = importlib.util.spec_from_file_location("cke_cohere_transcribe_native", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


native = _load_native()


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
        boundaries.append((current[0] + previous[1]) // 2)
    boundaries.append(total_frames)
    return [
        Window(index, start, end, boundaries[index], boundaries[index + 1])
        for index, (start, end) in enumerate(spans)
    ]


def find_energy_min_split(
    samples: np.ndarray,
    search_start: int,
    search_end: int,
    energy_window_frames: int,
) -> int:
    if search_end <= search_start:
        return search_start
    span = search_end - search_start
    if energy_window_frames <= 0:
        raise ValueError("energy window must be positive")
    if span <= energy_window_frames:
        return search_start + span // 2
    best = search_start
    best_energy = float("inf")
    for offset in range(0, span - energy_window_frames + 1, energy_window_frames):
        window = samples[
            search_start + offset:search_start + offset + energy_window_frames
        ].astype(np.float64)
        energy = float(np.sqrt(np.mean(window * window)))
        if energy < best_energy:
            best_energy = energy
            best = search_start + offset
    return best


def plan_energy_windows(
    samples: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    search_seconds: float,
    energy_window_seconds: float = 0.1,
) -> list[Window]:
    if samples.ndim != 1 or not samples.size or sample_rate <= 0:
        raise ValueError("mono audio geometry must be positive")
    window_frames = round(window_seconds * sample_rate)
    search_frames = round(search_seconds * sample_rate)
    energy_frames = round(energy_window_seconds * sample_rate)
    if window_frames <= 0 or search_frames <= 0 or search_frames >= window_frames:
        raise ValueError("energy search must be positive and shorter than the window")
    spans: list[tuple[int, int]] = []
    start = 0
    while start < samples.size:
        cap = min(start + window_frames, samples.size)
        if cap == samples.size:
            spans.append((start, cap))
            break
        cut = find_energy_min_split(
            samples, cap - search_frames, cap, energy_frames,
        )
        if cut <= start:
            cut = cap
        spans.append((start, cut))
        start = cut
    return [Window(index, start, end, start, end) for index, (start, end) in enumerate(spans)]


def load_speech_windows(
    path: Path,
    *,
    total_frames: int,
    sample_rate: int,
    max_window_seconds: float,
) -> tuple[list[Window], dict[str, Any]]:
    """Load and strictly validate CrispASR's versioned VAD slice format."""
    raw = path.read_bytes()
    document = json.loads(raw)
    container = document.get("crispasr_vad")
    if not isinstance(container, dict) or container.get("version") != 1:
        raise ValueError("speech segments must use crispasr_vad version 1")
    if int(container.get("sample_rate", 0)) != sample_rate:
        raise ValueError("speech segment sample rate does not match the WAV")
    slices = container.get("slices")
    if not isinstance(slices, list) or not slices:
        raise ValueError("speech segment list must be nonempty")
    if int(container.get("num_slices", -1)) != len(slices):
        raise ValueError("speech segment count does not match num_slices")
    maximum_frames = round(max_window_seconds * sample_rate)
    windows: list[Window] = []
    previous_end = 0
    for index, item in enumerate(slices):
        if not isinstance(item, dict):
            raise ValueError(f"speech segment {index} is not an object")
        start, end = int(item.get("start", -1)), int(item.get("end", -1))
        if start < previous_end or end <= start or end > total_frames:
            raise ValueError(f"speech segment {index} has invalid or overlapping bounds")
        if end - start > maximum_frames:
            raise ValueError(f"speech segment {index} exceeds the model window")
        windows.append(Window(index, start, end, start, end))
        previous_end = end
    return windows, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "format": "crispasr_vad",
        "version": 1,
        "kind": container.get("kind"),
    }


def repeated_ngram_runs(text: str, max_n: int = 16) -> list[dict[str, Any]]:
    """Find immediate decode loops using CrispASR's repetition limits."""
    words = re.findall(r"\S+", text.casefold())
    findings: list[dict[str, Any]] = []
    for size in range(min(max_n, len(words) // 2), 0, -1):
        allowed = 3 if size == 1 else 2
        index = 0
        while index + size * (allowed + 1) <= len(words):
            unit = words[index:index + size]
            repetitions = 1
            while (
                index + size * (repetitions + 1) <= len(words)
                and words[index + size * repetitions:index + size * (repetitions + 1)] == unit
            ):
                repetitions += 1
            if repetitions > allowed:
                findings.append({
                    "word_index": index,
                    "ngram_words": size,
                    "repetitions": repetitions,
                    "text": " ".join(unit),
                })
                index += size * repetitions
            else:
                index += 1
    return findings


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.casefold(), flags=re.UNICODE)


def word_error_rate(reference: str, candidate: str) -> dict[str, Any]:
    expected, actual = normalized_words(reference), normalized_words(candidate)
    if not expected:
        raise ValueError("reference transcript is empty after normalization")
    previous = list(range(len(actual) + 1))
    for row, left in enumerate(expected, 1):
        current = [row]
        for column, right in enumerate(actual, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left != right),
            ))
        previous = current
    edits = previous[-1]
    return {
        "reference_words": len(expected),
        "candidate_words": len(actual),
        "word_edits": edits,
        "word_error_rate": edits / len(expected),
    }


def load_reference_transcript(path: Path) -> tuple[str, dict[str, Any]]:
    raw = path.read_bytes()
    document = json.loads(raw)
    segments = document.get("transcription")
    if not isinstance(segments, list) or not segments:
        raise ValueError("reference transcription must contain nonempty segments")
    texts = [str(item.get("text", "")).strip() for item in segments if isinstance(item, dict)]
    transcript = " ".join(text for text in texts if text)
    if not transcript:
        raise ValueError("reference transcription has no text")
    return transcript, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def monotonic_dtw_path(attention: np.ndarray) -> np.ndarray:
    """Return the maximum-weight monotonic frame path for [token, frame]."""
    matrix = np.asarray(attention, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("attention must be a nonempty token-by-frame matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("attention contains non-finite values")
    token_count, frame_count = matrix.shape
    scores = np.empty_like(matrix)
    predecessor = np.zeros((token_count, frame_count), dtype=np.int32)
    scores[0] = matrix[0]
    for token in range(1, token_count):
        best_index = 0
        best_score = scores[token - 1, 0]
        for frame in range(frame_count):
            candidate = scores[token - 1, frame]
            if candidate > best_score:
                best_score = candidate
                best_index = frame
            scores[token, frame] = matrix[token, frame] + best_score
            predecessor[token, frame] = best_index
    path = np.empty(token_count, dtype=np.int32)
    path[-1] = int(np.argmax(scores[-1]))
    for token in range(token_count - 2, -1, -1):
        path[token] = predecessor[token + 1, path[token + 1]]
    return path


def aligned_tokens(
    session: Any,
    token_ids: list[int],
    duration_seconds: float,
) -> list[dict[str, Any]]:
    if not token_ids:
        return []
    attention = session.last_cross_attention
    if len(attention) < len(token_ids):
        raise RuntimeError("decoder did not retain attention for every generated token")
    # CrispASR aligns visible token i against the preceding generation step,
    # with token zero using the first available row.
    rows = np.stack([
        attention[max(index - 1, 0)].mean(axis=0)
        for index in range(len(token_ids))
    ])
    path = monotonic_dtw_path(rows)
    pieces = session.weights.metadata.get("tokenizer.ggml.tokens")
    if not isinstance(pieces, list):
        raise ValueError("model tokenizer pieces are unavailable")
    texts = [str(pieces[token_id]).replace("▁", " ") for token_id in token_ids]
    byte_lengths = [len(text.encode("utf-8")) for text in texts]
    total_bytes = sum(byte_lengths)
    byte_position = 0
    result = []
    for index, token_id in enumerate(token_ids):
        text = texts[index]
        start = min(float(path[index]) * 0.08, duration_seconds)
        end = (
            min(float(path[index + 1]) * 0.08, duration_seconds)
            if index + 1 < len(path)
            else duration_seconds
        )
        ownership_start = (
            byte_position / total_bytes * duration_seconds if total_bytes else start
        )
        byte_position += byte_lengths[index]
        ownership_end = (
            byte_position / total_bytes * duration_seconds if total_bytes else end
        )
        result.append({
            "id": token_id,
            "text": text,
            "start": start,
            "end": max(start, end),
            "ownership_start": ownership_start,
            "ownership_end": ownership_end,
        })
    return result


def select_owned_tokens(
    tokens: list[dict[str, Any]],
    window: Window,
    sample_rate: int,
    final_window: bool,
) -> list[dict[str, Any]]:
    selected = []
    window_start = window.start_frame / sample_rate
    owner_start = window.ownership_start_frame / sample_rate
    owner_end = window.ownership_end_frame / sample_rate
    for token in tokens:
        start = window_start + float(token["start"])
        end = window_start + float(token["end"])
        ownership_start = window_start + float(token["ownership_start"])
        ownership_end = window_start + float(token["ownership_end"])
        midpoint = (ownership_start + ownership_end) * 0.5
        if midpoint >= owner_start and (
            midpoint < owner_end or (final_window and midpoint <= owner_end)
        ):
            selected.append({
                "id": token["id"],
                "text": token["text"],
                # DTW can assign a span across the ownership boundary. Clipping
                # keeps the retained global trajectory monotonic without using
                # timestamp geometry to decide which window owns the token.
                "start": min(max(start, owner_start), owner_end),
                "end": min(max(end, owner_start), owner_end),
            })
    return selected


def build_captions(
    tokens: list[dict[str, Any]],
    *,
    maximum_seconds: float = 8.0,
    maximum_characters: int = 80,
) -> list[dict[str, Any]]:
    if maximum_seconds <= 0 or maximum_characters <= 0:
        raise ValueError("caption limits must be positive")
    captions: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for token in tokens:
        current.append(token)
        text = "".join(str(item["text"]) for item in current).strip()
        start = float(current[0]["start"])
        end = max(float(item["end"]) for item in current)
        sentence_end = bool(re.search(r"[.!?][\"']?$", text))
        if len(text) >= maximum_characters or end - start >= maximum_seconds or sentence_end:
            captions.append({
                "index": len(captions) + 1,
                "start": start,
                "end": max(end, start + 0.08),
                "text": text,
            })
            current = []
    if current:
        text = "".join(str(item["text"]) for item in current).strip()
        start = float(current[0]["start"])
        end = max(float(item["end"]) for item in current)
        captions.append({
            "index": len(captions) + 1,
            "start": start,
            "end": max(end, start + 0.08),
            "text": text,
        })
    return captions


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def render_srt(captions: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"{item['index']}\n{_srt_time(float(item['start']))} --> "
        f"{_srt_time(float(item['end']))}\n{item['text']}"
        for item in captions
    ) + ("\n" if captions else "")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_window(
    path: Path,
    payload: bytes,
    *,
    channels: int,
    sample_width: int,
    sample_rate: int,
) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(sample_rate)
        output.writeframes(payload)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    _write_text(path, json.dumps(report, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    usage_started = resource.getrusage(resource.RUSAGE_SELF)
    with wave.open(str(args.audio), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        total_frames = source.getnframes()
        compression = source.getcomptype()
        payload = source.readframes(total_frames)
    if sample_width != 2 or compression != "NONE":
        raise ValueError("Cohere long audio requires uncompressed PCM16 WAV")
    bytes_per_frame = channels * sample_width
    if len(payload) != total_frames * bytes_per_frame:
        raise RuntimeError("WAV payload is shorter than its declared frame count")

    pcm_frames = np.frombuffer(payload, dtype="<i2").reshape(total_frames, channels)
    mono_pcm = pcm_frames.astype(np.float32).mean(axis=1)
    speech_segments = None
    if args.speech_segments:
        if args.overlap_seconds != 0:
            raise ValueError("imported speech segments cannot be combined with overlap")
        windows, speech_segments = load_speech_windows(
            args.speech_segments,
            total_frames=total_frames,
            sample_rate=sample_rate,
            max_window_seconds=args.window_seconds,
        )
    elif args.boundary_search_seconds > 0:
        if args.overlap_seconds != 0:
            raise ValueError("quiet-point boundary search currently requires zero overlap")
        windows = plan_energy_windows(
            mono_pcm, sample_rate, args.window_seconds,
            args.boundary_search_seconds,
        )
    else:
        windows = plan_windows(
            total_frames, sample_rate, args.window_seconds, args.overlap_seconds,
        )
    kernels = native.CohereKernels(args.engine, args.audio_lib)
    session = native.CohereSession(args.model, kernels)
    max_clip_seconds = float(
        session.weights.metadata["cohere_transcribe.audio.max_clip_s"]
    )
    if args.window_seconds > max_clip_seconds:
        raise ValueError(
            f"window exceeds model max clip: {args.window_seconds} > {max_clip_seconds}"
        )
    report: dict[str, Any] = {
        "schema": "cke.v8.cohere_transcribe.long_audio",
        "schema_version": 1,
        "status": "RUNNING",
        "model": {
            "id": "cstr/cohere-transcribe-03-2026-GGUF",
            "artifact": (
                {
                    "format": "BUMPWGT5",
                    "weights": native.parakeet.file_identity(args.model / "weights.bump"),
                    "manifest": native.parakeet.file_identity(args.model / "weights_manifest.json"),
                    "config": native.parakeet.file_identity(args.model / "config.json"),
                }
                if args.model.is_dir()
                else {"format": "GGUF", **native.parakeet.file_identity(args.model)}
            ),
        },
        "input": {
            **native.parakeet.file_identity(args.audio),
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "source_frames": total_frames,
            "source_seconds": total_frames / sample_rate,
        },
        "runtime": kernels.provenance(),
        "thread_policy": kernels.thread_policy(),
        "policy": {
            "kind": (
                "imported_vad_speech_slices"
                if speech_segments
                else (
                    "quiet_point_bounded_full_attention_windows"
                    if args.boundary_search_seconds > 0
                    else "overlapping_bounded_full_attention_windows"
                )
            ),
            "window_seconds": args.window_seconds,
            "overlap_seconds": args.overlap_seconds,
            "boundary_search_seconds": args.boundary_search_seconds,
            "model_max_clip_seconds": max_clip_seconds,
            "state": "reset_each_window",
            "ownership": (
                "speech_slice_half_open_window"
                if speech_segments
                else (
                    "quiet_point_half_open_window"
                    if args.boundary_search_seconds > 0
                    else "linear_token_byte_midpoint_at_overlap_midpoint"
                )
            ),
            "timestamp_alignment": "last_decoder_layer_cross_attention_monotonic_dtw",
            "timestamp_resolution_seconds": 0.08,
            "resampling": "cke_windowed_sinc_radius_16_before_frontend",
            "digital_silence": "all_pcm16_samples_zero",
        },
        "windows": [],
        "selected_tokens": [],
    }
    if speech_segments:
        report["speech_segments"] = speech_segments
    prior_elapsed = 0.0
    prior_cpu = {"user": 0.0, "system": 0.0}
    prior_peak_rss = 0
    if args.resume and args.output and args.output.is_file():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("schema") != report["schema"]:
            raise ValueError("resume report schema mismatch")
        if prior.get("input", {}).get("sha256") != report["input"]["sha256"]:
            raise ValueError("resume report input mismatch")
        if prior.get("policy") != report["policy"]:
            raise ValueError("resume report policy mismatch")
        completed = prior.get("windows") or []
        for expected, actual in zip(windows, completed):
            geometry = asdict(expected)
            if any(actual.get(name) != value for name, value in geometry.items()):
                raise ValueError("resume report window geometry mismatch")
        report["windows"] = completed
        report["selected_tokens"] = prior.get("selected_tokens") or []
        report["resume_count"] = int(prior.get("resume_count", 0)) + 1
        prior_elapsed = float(prior.get("total_elapsed_seconds", 0.0))
        previous_cpu = prior.get("process_cpu_seconds") or {}
        prior_cpu = {
            "user": float(previous_cpu.get("user", 0.0)),
            "system": float(previous_cpu.get("system", 0.0)),
        }
        prior_peak_rss = int(prior.get("peak_rss_bytes", 0))

    try:
        with tempfile.TemporaryDirectory(prefix="cke-cohere-long-") as temporary_text:
            temporary = Path(temporary_text)
            for window in windows[len(report["windows"]):]:
                window_started = time.perf_counter()
                begin = window.start_frame * bytes_per_frame
                end = window.end_frame * bytes_per_frame
                window_payload = payload[begin:end]
                pcm = np.frombuffer(window_payload, dtype="<i2")
                silent = bool(not np.any(pcm))
                path = temporary / f"window-{window.index:04d}.wav"
                _write_window(
                    path, window_payload, channels=channels,
                    sample_width=sample_width, sample_rate=sample_rate,
                )
                token_ids: list[int] = []
                transcript = ""
                finite_frontend = True
                finite_encoder = True
                frontend_seconds = encoder_seconds = decoder_seconds = 0.0
                encoder_frames = 0
                if not silent:
                    stage = time.perf_counter()
                    features = session.frontend(path)
                    frontend_seconds = time.perf_counter() - stage
                    finite_frontend = bool(np.isfinite(features).all())
                    stage = time.perf_counter()
                    encoder = session.encode(features)
                    encoder_seconds = time.perf_counter() - stage
                    finite_encoder = bool(np.isfinite(encoder).all())
                    encoder_frames = int(encoder.shape[0])
                    stage = time.perf_counter()
                    token_ids, transcript = session.decode(
                        encoder, args.language, args.max_new_tokens,
                    )
                    decoder_seconds = time.perf_counter() - stage
                    local_tokens = aligned_tokens(
                        session, token_ids,
                        (window.end_frame - window.start_frame) / sample_rate,
                    )
                    owned = select_owned_tokens(
                        local_tokens, window, sample_rate,
                        window.index == len(windows) - 1,
                    )
                    report["selected_tokens"].extend(owned)
                else:
                    owned = []
                report["windows"].append({
                    **asdict(window),
                    "source_seconds_start": window.start_frame / sample_rate,
                    "source_seconds_end": window.end_frame / sample_rate,
                    "source_frames_consumed": window.end_frame - window.start_frame,
                    "resampled_frames": round(
                        (window.end_frame - window.start_frame) * 16000 / sample_rate
                    ),
                    "digital_silence": silent,
                    "encoder_frames": encoder_frames,
                    "generated_token_ids": token_ids,
                    "generated_tokens": len(token_ids),
                    "selected_tokens": len(owned),
                    "transcript": transcript,
                    "finite_frontend": finite_frontend,
                    "finite_encoder": finite_encoder,
                    "frontend_seconds": frontend_seconds,
                    "encoder_seconds": encoder_seconds,
                    "decoder_seconds": decoder_seconds,
                    "elapsed_seconds": time.perf_counter() - window_started,
                })
                if args.output:
                    report["total_elapsed_seconds"] = prior_elapsed + time.perf_counter() - started
                    _write_report(args.output, report)
    except BaseException as error:
        report["status"] = "ERROR"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["transcript"] = "".join(
            str(token["text"]) for token in report["selected_tokens"]
        ).strip()
        report["captions"] = build_captions(report["selected_tokens"])
        covered_frames = sum(window.end_frame - window.start_frame for window in windows)
        report["source_coverage"] = {
            "source_frames": total_frames,
            "source_seconds": total_frames / sample_rate,
            "scheduled_frames": covered_frames,
            "scheduled_seconds": covered_frames / sample_rate,
            "intentionally_skipped_frames": total_frames - covered_frames,
            "planned_windows": len(windows),
            "completed_windows": len(report["windows"]),
            "complete": len(report["windows"]) == len(windows),
        }
        report["total_elapsed_seconds"] = prior_elapsed + time.perf_counter() - started
        report["real_time_factor"] = report["total_elapsed_seconds"] / (total_frames / sample_rate)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report["process_cpu_seconds"] = {
            "user": prior_cpu["user"] + usage.ru_utime - usage_started.ru_utime,
            "system": prior_cpu["system"] + usage.ru_stime - usage_started.ru_stime,
        }
        report["peak_rss_bytes"] = max(prior_peak_rss, int(usage.ru_maxrss) * 1024)
        if report["status"] == "RUNNING":
            selected = report["selected_tokens"]
            loops = repeated_ngram_runs(report["transcript"])
            phrase_loops = [item for item in loops if int(item["ngram_words"]) > 1]
            checks = {
                "all_windows_completed": len(report["windows"]) == len(windows),
                "scheduled_source_coverage_exact": covered_frames == sum(
                    int(item["source_frames_consumed"]) for item in report["windows"]
                ),
                "finite_outputs": all(
                    item["finite_frontend"] and item["finite_encoder"]
                    for item in report["windows"]
                ),
                "monotonic_token_timestamps": all(
                    float(selected[index]["start"]) >= float(selected[index - 1]["start"])
                    for index in range(1, len(selected))
                ),
                "nonempty_transcript": bool(report["transcript"]),
                "no_repeated_phrase_loops": not phrase_loops,
            }
            report["quality"] = {
                "repetition_findings": loops,
                "repeated_phrase_findings": phrase_loops,
            }
            if args.reference_transcript:
                reference, identity = load_reference_transcript(args.reference_transcript)
                comparison = word_error_rate(reference, report["transcript"])
                comparison["maximum_word_error_rate"] = args.max_word_error_rate
                comparison["within_limit"] = comparison["word_error_rate"] <= args.max_word_error_rate
                report["quality"]["reference"] = identity
                report["quality"]["reference_comparison"] = comparison
                checks["reference_word_error_rate"] = comparison["within_limit"]
            report["checks"] = checks
            report["status"] = "PASS" if all(checks.values()) else "FAIL"
        if args.output:
            _write_report(args.output, report)
        if report["status"] in {"PASS", "FAIL"}:
            if args.text_output:
                _write_text(args.text_output, report["transcript"] + "\n")
            if args.srt_output:
                _write_text(args.srt_output, render_srt(report["captions"]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--engine", default=Path("build/libckernel_engine.so"), type=Path)
    parser.add_argument("--audio-lib", default=Path("build/libckernel_audio.so"), type=Path)
    parser.add_argument("--window-seconds", default=30.0, type=float)
    parser.add_argument("--overlap-seconds", default=0.0, type=float)
    parser.add_argument("--boundary-search-seconds", default=5.0, type=float)
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-new-tokens", default=256, type=int)
    parser.add_argument("--speech-segments", type=Path)
    parser.add_argument("--reference-transcript", type=Path)
    parser.add_argument("--max-word-error-rate", default=0.10, type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--text-output", type=Path)
    parser.add_argument("--srt-output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report: dict[str, Any]
    try:
        report = run(args)
    except Exception as error:
        existing = None
        if args.output and args.output.is_file():
            try:
                existing = json.loads(args.output.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = None
        if args.output and not (
            isinstance(existing, dict) and existing.get("status") == "ERROR"
        ):
            _write_report(args.output, {
                "schema": "cke.v8.cohere_transcribe.long_audio",
                "schema_version": 1,
                "status": "ERROR",
                "error": {"type": type(error).__name__, "message": str(error)},
            })
        print(f"Cohere long-audio run failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
