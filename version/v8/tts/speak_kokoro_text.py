#!/usr/bin/env python3
"""Speak checked English text through one loaded generated Kokoro runtime.

The pinned frontend plans phoneme requests; the native host appends their
returned valid samples into one WAV. Neural operations remain in generated C.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import wave

from prepare_kokoro_accessibility_text import TextPlanError, prepare


class SpeechError(RuntimeError):
    pass


def _exchange(worker: subprocess.Popen, command: str, prefix: str) -> list[str]:
    if any(character in command for character in "\r\n"):
        raise SpeechError("invalid native command")
    assert worker.stdin is not None and worker.stdout is not None
    worker.stdin.write(command + "\n")
    worker.stdin.flush()
    response = worker.stdout.readline().rstrip("\n")
    fields = response.split("\t")
    if not fields or fields[0] != prefix:
        raise SpeechError(f"native worker rejected {command.split(chr(9))[0]}: {response or 'EOF'}")
    return fields


def _validate_wav(path: Path, samples: int) -> None:
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
                audio.getcomptype(), audio.getnframes()) != (
                    1, 2, 24000, "NONE", samples):
            raise SpeechError("native WAV format or returned sample count mismatch")
        if len(audio.readframes(samples)) != samples * 2:
            raise SpeechError("native WAV has a truncated payload")
    if path.stat().st_size != 44 + 2 * samples:
        raise SpeechError("native WAV has unexpected trailing data")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def synthesize(text: str, config: Path, bundle: Path, output: Path,
               seed: int) -> dict:
    if seed < 0 or seed > (1 << 64) - 1:
        raise SpeechError("seed must be an unsigned 64-bit integer")
    if output.exists() or output.is_symlink():
        raise SpeechError("output already exists")
    start = time.monotonic()
    plan = prepare(text, config, bundle)
    if len(plan["chunks"]) > 64:
        raise SpeechError("aggregate WAV supports at most 64 chunks; use incremental playback")
    if seed > (1 << 64) - len(plan["chunks"]):
        raise SpeechError("seed range exceeds unsigned 64-bit integer")
    planned = time.monotonic()
    dispatcher = bundle / "native-kokoro-dispatch"
    if not dispatcher.is_file():
        raise SpeechError("native Kokoro dispatcher unavailable")
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    with tempfile.TemporaryDirectory(prefix=".kokoro-speech-", dir=output.parent) as scratch:
        staged = Path(scratch) / "speech.wav"
        worker = subprocess.Popen([str(dispatcher), str(bundle), "--session"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1)
        try:
            assert worker.stdout is not None
            ready = worker.stdout.readline().strip()
            if ready != "READY\tloads=1":
                detail = worker.stderr.read().strip() if not ready and worker.stderr else ""
                raise SpeechError(f"native worker did not start: {ready or detail or 'EOF'}")
            _exchange(worker, f"BEGIN\t{staged}", "BEGUN")
            total = 0
            first = None
            for chunk in plan["chunks"]:
                ids = ",".join(map(str, chunk["phoneme_ids"]))
                fields = _exchange(worker, f"APPEND\t{seed + chunk['index']}\t{ids}",
                                   "CHUNK_READY")
                if len(fields) != 3:
                    raise SpeechError("invalid native chunk response")
                count, cumulative = map(int, fields[1:])
                if count <= 0 or cumulative != total + count:
                    raise SpeechError("native chunk sample extent mismatch")
                total = cumulative
                first = first or time.monotonic()
                chunks.append({"index": chunk["index"],
                               "source_start": chunk["source_start"],
                               "source_end": chunk["source_end"],
                               "source_text": chunk["source_text"],
                               "valid_token_count": chunk["valid_token_count"],
                               "voice_row": chunk["voice_row"],
                               "samples": count})
            fields = _exchange(worker, "END", "WAV_READY")
            if len(fields) != 3 or int(fields[1]) != total or fields[2] != str(staged):
                raise SpeechError("native final sample extent mismatch")
            _validate_wav(staged, total)
            if output.exists() or output.is_symlink():
                raise SpeechError("output appeared during synthesis")
            os.link(staged, output)
            done = time.monotonic()
            assert worker.stdin is not None
            worker.stdin.write("QUIT\n")
            worker.stdin.flush()
            worker.stdin.close()
            if worker.wait(timeout=10) != 0:
                output.unlink(missing_ok=True)
                raise SpeechError("native worker exited with failure")
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
            if worker.stdin and not worker.stdin.closed:
                worker.stdin.close()
            if worker.stdout:
                worker.stdout.close()
            if worker.stderr:
                worker.stderr.close()
    return {"status": "execution_pass", "numerical_waveform_parity": "not_tested_this_run",
            "known_connected_waveform_parity": "fail_unresolved",
            "chunks": chunks, "sample_count": total, "sample_rate": 24000,
            "source_sha256": hashlib.sha256(plan["source_text"].encode()).hexdigest(),
            "wav_sha256": _sha256(output),
            "plan_seconds": planned - start,
            "first_chunk_seconds": first - start if first is not None else None,
            "total_seconds": done - start,
            "bundle_identity_sha256": plan["bundle_identity_sha256"],
            "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
            "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text-file", type=Path)
    source.add_argument("--stdin", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    def interrupted(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.report and (args.report.exists() or args.report.is_symlink()):
            raise SpeechError("report already exists")
        text = args.text_file.read_text() if args.text_file else sys.stdin.read()
        report = synthesize(text, args.config, args.bundle_dir, args.output,
                            args.seed)
        if args.report:
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
        print(json.dumps(report, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TextPlanError, SpeechError) as error:
        print(f"Kokoro speech failed: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Kokoro speech interrupted; partial WAV discarded", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
