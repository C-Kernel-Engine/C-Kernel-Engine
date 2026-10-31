#!/usr/bin/env python3
"""Turn a generated Whisper attention trace into inspectable word timings.

Numerical alignment runs in the native CKE provider. This host script owns
artifact validation, tokenizer correspondence, time offsets and reporting.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re

import numpy as np


_WORD = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _word_tokens(tokenizer, ids: list[int]) -> tuple[str, list[dict]]:
    text = tokenizer.decode(ids, skip_special_tokens=True)
    previous = ""
    char_token: list[int] = []
    monotonic = True
    for index in range(len(ids)):
        current = tokenizer.decode(ids[:index + 1], skip_special_tokens=True)
        if not current.startswith(previous):
            monotonic = False
            break
        char_token.extend([index] * (len(current) - len(previous)))
        previous = current
    if previous != text or len(char_token) != len(text):
        monotonic = False
    words = []
    for match in _WORD.finditer(text):
        item = {"text": match.group(), "char_start": match.start(),
                "char_end": match.end(), "token_start": None,
                "token_end_exclusive": None,
                "correspondence": "monotonic_prefix" if monotonic else "ambiguous"}
        if monotonic:
            item["token_start"] = char_token[match.start()]
            item["token_end_exclusive"] = char_token[match.end() - 1] + 1
        words.append(item)
    return text, words


def align_trace(report: dict, trace_path: Path, decoder_dir: Path,
                encoder_dir: Path, library_path: Path, window_index: int) -> dict:
    segments = report.get("segments")
    if isinstance(segments, list) and segments:
        if not (0 <= window_index < len(segments)):
            raise ValueError("window index exceeds report segment count")
        segment = segments[window_index]
        encoder = segment["encoder"]
        decoder = segment["decoder"]
    elif window_index == 0:
        encoder, decoder = report["encoder"], report["decoder"]
    else:
        raise ValueError("single-window report only supports window zero")
    identity = decoder.get("attention_trace")
    if not isinstance(identity, dict) or _sha256(trace_path) != identity.get("sha256"):
        raise ValueError("attention trace hash does not match decoder report")
    with np.load(trace_path, allow_pickle=False) as data:
        attention = np.asarray(data["selected_attention"], dtype=np.float32)
        token_ids = np.asarray(data["token_ids"], dtype=np.int32)
        heads = np.asarray(data["alignment_heads"], dtype=np.int32)
        frame_capacity = int(data["frame_capacity"])
    if (attention.ndim != 3 or attention.shape !=
            (heads.shape[0], len(token_ids), frame_capacity) or
            heads.shape != (len(identity["selected_heads"]), 2) or
            heads.tolist() != identity["selected_heads"] or
            list(attention.shape) != identity.get("shape") or
            token_ids.tolist() != decoder["generated_tokens"] or
            not np.isfinite(attention).all() or
            (attention.shape[1] and not np.allclose(
                attention.sum(axis=2), 1.0, rtol=0, atol=2e-4))):
        raise ValueError("trace geometry, selected heads or tokens disagree with report")
    audio = encoder["audio"]
    encoder_config = json.loads((encoder_dir / "config.json").read_text())
    source_rate = int(audio["source_sample_rate"])
    target_rate = int(encoder_config["audio_sample_rate"])
    source_frames = int(audio["window_source_frames"])
    start_frame = int(audio["window_start_frame"])
    hop = int(encoder_config["audio_hop_length"])
    stride = int(encoder_config["audio_conv2_stride"])
    if min(source_rate, target_rate, source_frames, hop, stride) <= 0 or start_frame < 0:
        raise ValueError("invalid audio geometry for word timing")
    resampled = (source_frames * target_rate + source_rate - 1) // source_rate
    valid_frames = min(frame_capacity, (resampled + hop * stride - 1) // (hop * stride))
    if not (1 <= valid_frames <= frame_capacity):
        raise ValueError("valid encoder-frame extent is outside trace capacity")
    generation = json.loads((decoder_dir / "generation_config.json").read_text())
    timestamp_begin = int(generation["no_timestamps_token_id"]) + 1
    selected_indices = [i for i, token in enumerate(token_ids)
                        if int(token) < timestamp_begin]
    text_ids = [int(token_ids[i]) for i in selected_indices]
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(decoder_dir / "tokenizer.json"))
    text, words = _word_tokens(tokenizer, text_ids)
    seconds_per_frame = hop * stride / target_rate
    window_start_seconds = start_frame / source_rate
    aligned_frames: np.ndarray | None = None
    provider_status = None
    if len(text_ids) >= 2:
        selected = np.ascontiguousarray(
            attention[:, selected_indices, :valid_frames], dtype=np.float32)
        selected_heads, tokens, frames = selected.shape
        matrix = tokens * frames
        cells = (tokens + 1) * (frames + 1)
        score = np.zeros(matrix, np.float32)
        mean = np.zeros(matrix, np.float32)
        cost = np.zeros(cells, np.float32)
        backtrace = np.zeros(cells, np.uint8)
        aligned_frames = np.full(tokens, -1, np.int32)
        library = ctypes.CDLL(str(library_path))
        function = library.alignment_attention_dtw_f32
        function.argtypes = ([ctypes.c_void_p, ctypes.c_size_t] +
                             [ctypes.c_size_t] * 6 +
                             [ctypes.c_void_p, ctypes.c_size_t] * 4 +
                             [ctypes.c_void_p, ctypes.c_size_t])
        function.restype = ctypes.c_int
        provider_status = int(function(
            selected.ctypes.data, selected.size,
            selected_heads, tokens, frames, frames, tokens * frames, 7,
            score.ctypes.data, score.size, mean.ctypes.data, mean.size,
            cost.ctypes.data, cost.size, backtrace.ctypes.data, backtrace.size,
            aligned_frames.ctypes.data, aligned_frames.size))
        if provider_status != 0:
            aligned_frames = None
    else:
        provider_status = "insufficient_text_tokens"

    for word in words:
        first = word["token_start"]
        last = word["token_end_exclusive"]
        word.update({"start_seconds": None, "end_seconds": None,
                     "audio_window": window_index,
                     "transcription_confidence": None,
                     "boundary_reliability": "missing",
                     "quality_indicators": {"calibrated": False}})
        if aligned_frames is None or first is None or last is None:
            continue
        begin = int(aligned_frames[first])
        end = int(aligned_frames[last]) if last < len(aligned_frames) else valid_frames
        if not (0 <= begin < end <= valid_frames):
            word["boundary_reliability"] = "ambiguous"
            continue
        word["start_seconds"] = round(window_start_seconds + begin * seconds_per_frame, 3)
        word["end_seconds"] = round(window_start_seconds + end * seconds_per_frame, 3)
        word["boundary_reliability"] = "uncalibrated_alignment"
        word["quality_indicators"] = {
            "calibrated": False,
            "selected_head_peak_mean": float(attention[:, selected_indices[first], begin].mean()),
            "duration_frames": end - begin,
        }
    status = "aligned" if words and all(
        word["start_seconds"] is not None for word in words) else "partial"
    return {
        "schema": "cke.whisper_word_alignment.v1", "status": status,
        "text": text, "words": words, "token_ids": text_ids,
        "token_first_frames": aligned_frames.tolist() if aligned_frames is not None else None,
        "provider_status": provider_status,
        "window": {"index": window_index, "start_seconds": window_start_seconds,
                   "source_frames": source_frames, "source_sample_rate": source_rate,
                   "valid_attention_frames": valid_frames,
                   "attention_frame_seconds": seconds_per_frame},
        "provenance": {"trace_sha256": _sha256(trace_path),
                       "alignment_library_sha256": _sha256(library_path),
                       "tokenizer_sha256": _sha256(decoder_dir / "tokenizer.json"),
                       "generation_config_sha256": _sha256(decoder_dir / "generation_config.json"),
                       "selected_heads": heads.tolist()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--decoder-run-dir", type=Path, required=True)
    parser.add_argument("--encoder-run-dir", type=Path, required=True)
    parser.add_argument("--alignment-library", type=Path, required=True)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    result = align_trace(report, args.trace, args.decoder_run_dir,
                         args.encoder_run_dir, args.alignment_library,
                         args.window_index)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['status']}: {len(result['words'])} words -> {args.output}")
    return 0 if result["status"] == "aligned" else 2


if __name__ == "__main__":
    raise SystemExit(main())
