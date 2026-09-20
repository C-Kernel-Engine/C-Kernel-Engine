#!/usr/bin/env python3
"""Export validated CrispASR VAD slices as a native CKE audio segment plan."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


SCHEMA = "cke_audio_segments_v1"


def load_segments(path: Path) -> tuple[int, list[tuple[int, int]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    container = document.get("crispasr_vad")
    if not isinstance(container, dict) or container.get("version") != 1:
        raise ValueError("segment source must use crispasr_vad version 1")
    sample_rate = container.get("sample_rate")
    slices = container.get("slices")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("segment source has invalid sample rate")
    if not isinstance(slices, list) or not slices:
        raise ValueError("segment source has no slices")
    if container.get("num_slices") != len(slices):
        raise ValueError("segment source count does not match num_slices")
    result: list[tuple[int, int]] = []
    previous_end = 0
    for index, item in enumerate(slices):
        if not isinstance(item, dict):
            raise ValueError(f"segment {index} is not an object")
        start, end = item.get("start"), item.get("end")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < previous_end
            or end <= start
        ):
            raise ValueError(f"segment {index} has invalid or overlapping bounds")
        result.append((start, end))
        previous_end = end
    return sample_rate, result


def validate_audio(path: Path, sample_rate: int, segments: list[tuple[int, int]]) -> int:
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getcomptype() != "NONE"
        ):
            raise ValueError("audio must be uncompressed mono PCM16 WAV")
        if source.getframerate() != sample_rate:
            raise ValueError("audio sample rate does not match segment source")
        frames = source.getnframes()
    if segments[-1][1] > frames:
        raise ValueError("segment source exceeds audio frame count")
    return frames


def render(sample_rate: int, segments: list[tuple[int, int]]) -> str:
    rows = [f"{SCHEMA} {sample_rate} {len(segments)}"]
    rows.extend(f"{start} {end}" for start, end in segments)
    return "\n".join(rows) + "\n"


def write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="ascii")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vad", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sample_rate, segments = load_segments(args.vad)
    validate_audio(args.audio, sample_rate, segments)
    write_atomic(args.output, render(sample_rate, segments))
    print(f"wrote {len(segments)} native audio segments to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
