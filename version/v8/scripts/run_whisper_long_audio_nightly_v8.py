#!/usr/bin/env python3
"""Run one fail-closed CKE Whisper long-audio nightly certification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

from certify_whisper_long_audio_v8 import certify


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MANIFEST = (
    ROOT / "version" / "v8" / "test_assets" / "whisper_long_audio" / "corpus.json"
)


def _pcm_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise RuntimeError("nightly WAV must be mono PCM16")
        if audio.getframerate() != 16000:
            raise RuntimeError("nightly WAV must be 16 kHz")
        while True:
            frames = audio.readframes(65536)
            if not frames:
                break
            digest.update(frames)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.model not in manifest["models"]:
        raise RuntimeError(f"unknown corpus model: {args.model}")
    fixture = manifest["fixture"]
    policy = manifest["models"][args.model]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required; nightly may not skip audio materialization")

    work_root = args.work_root.resolve() / args.model
    output_root = args.output_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    wav_path = work_root / "kernel2_mic1_5min.wav"
    runtime_root = work_root / "runtime"
    report_path = output_root / f"whisper-{args.model}-5min-report.json"
    certification_path = output_root / f"whisper-{args.model}-5min-certification.json"

    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(manifest_path.parent / fixture["audio"]),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ],
        check=True,
    )
    actual_pcm_hash = _pcm_sha256(wav_path)
    if actual_pcm_hash != fixture["pcm_s16le_sha256"]:
        raise RuntimeError(
            "materialized PCM hash mismatch: "
            f"expected {fixture['pcm_s16le_sha256']}, got {actual_pcm_hash}"
        )

    command = [
        str(SCRIPT_DIR / "cks-v8-run"),
        "audio",
        f"hf://{policy['checkpoint']}",
        "--run",
        str(runtime_root),
        "--wav",
        str(wav_path),
        "--language",
        fixture["language"],
        "--task",
        fixture["task"],
        "--max-tokens",
        "448",
        "--timestamps",
        "--output",
        str(report_path),
    ]
    if args.force_convert:
        command.append("--force-convert")
    if args.force_compile:
        command.append("--force-compile")
    environment = os.environ.copy()
    environment.setdefault("CK_CACHE_DIR", str(args.work_root.resolve() / "model-cache"))
    subprocess.run(command, cwd=ROOT, env=environment, check=True)

    summary = certify(manifest_path, report_path, args.model)
    certification_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Whisper long-audio nightly: FAIL: {error}", file=sys.stderr)
        raise
