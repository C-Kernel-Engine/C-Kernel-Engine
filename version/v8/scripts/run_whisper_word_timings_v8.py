#!/usr/bin/env python3
"""Produce candidate word timings with a trace-enabled CKE Whisper bundle."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys

from align_whisper_trace_v8 import align_trace
from assemble_whisper_word_timings_v8 import assemble, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-run-dir", type=Path, required=True)
    parser.add_argument("--decoder-run-dir", type=Path, required=True,
                        help="Generated decoder built with attention tracing")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New private directory for the transcript, traces and timings")
    parser.add_argument("--language", default="en")
    parser.add_argument("--vocabulary-hint", default="")
    parser.add_argument("--max-tokens", default=128, type=int)
    args = parser.parse_args()

    library = (args.decoder_run_dir / "libckernel_engine.so").resolve()
    try:
        getattr(ctypes.CDLL(str(library)), "alignment_attention_dtw_f32")
    except (OSError, AttributeError) as error:
        parser.error(f"decoder engine must export alignment_attention_dtw_f32 ({error})")

    try:
        output = args.output_dir.resolve()
        output.mkdir(mode=0o700)
        report_path = output / "run.json"
        trace_dir = output / "traces"
        command = [
            sys.executable, str(Path(__file__).with_name("run_whisper_v8.py")), "run",
            "--encoder-run-dir", str(args.encoder_run_dir),
            "--decoder-run-dir", str(args.decoder_run_dir),
            "--wav", str(args.wav), "--language", args.language,
            "--max-tokens", str(args.max_tokens), "--timestamps",
            "--attention-trace-dir", str(trace_dir), "--output", str(report_path),
        ]
        if args.vocabulary_hint:
            command.append(f"--vocabulary-hint={args.vocabulary_hint}")
        result = subprocess.run(command)
        if result.returncode:
            raise RuntimeError("traced transcription failed; check the decoder trace build and selected heads")

        report = json.loads(report_path.read_text())
        report_hash = sha256(report_path)
        audio_hash = sha256(args.wav)
        if report.get("status") != "ok" or audio_hash != report.get("wav_sha256"):
            raise ValueError("trace run did not complete or audio changed")
        alignments = []
        for index, _ in enumerate(report["segments"]):
            alignment = align_trace(report, trace_dir / f"window-{index:04d}.npz",
                                    args.decoder_run_dir, args.encoder_run_dir, library, index)
            alignment["provenance"].update(run_report_sha256=report_hash, audio_sha256=audio_hash)
            (output / f"alignment-{index:04d}.json").write_text(json.dumps(alignment, indent=2) + "\n")
            alignments.append(alignment)
        words = assemble(report, alignments, report_hash)
        path = output / "words.json"
        path.write_text(json.dumps(words, indent=2) + "\n")
        counts = words["counts"]
        print(f"{words['status']}: {counts['committed']} committed, {counts['review']} review -> {path}")
        return 2 if words["review_words"] else 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"Whisper word timing failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
