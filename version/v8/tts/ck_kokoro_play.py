#!/usr/bin/env python3
"""Play one checked Kokoro whole-text WAV for the local voice client.

The caller owns cancellation of this process group. The model schedule remains
in the generated runtime invoked by speak_kokoro_text.py.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave


class PlaybackError(RuntimeError):
    pass


class _Playing:
    def __init__(self, player: Path, wav: Path):
        self.wav = wav
        self.child = subprocess.Popen([str(player), str(wav)],
                                      stdin=subprocess.DEVNULL)
        self.ended = None
        self.thread = threading.Thread(target=self._wait, daemon=True)
        self.thread.start()

    def _wait(self):
        self.child.wait()
        self.ended = time.monotonic()

    def finish(self) -> float:
        self.thread.join()
        if self.child.returncode:
            raise PlaybackError(f"audio player failed ({self.child.returncode})")
        assert self.ended is not None
        return self.ended

    def stop(self):
        if self.child.poll() is None:
            try:
                self.child.kill()
            except ProcessLookupError:
                pass
        self.thread.join()


def _run(command: list[str], *, capture: bool) -> None:
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE if capture else None,
                             stderr=subprocess.PIPE if capture else None)
    try:
        stdout, stderr = child.communicate()
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    if child.returncode:
        detail = (stderr or b"").decode("utf-8", "replace")[-400:]
        raise PlaybackError(f"speech command failed ({child.returncode}): {detail}")


def _checked_wav(path: Path, samples: int) -> None:
    try:
        if type(samples) is not int or samples <= 0:
            raise ValueError("invalid sample count")
        with wave.open(str(path), "rb") as audio:
            valid = ((audio.getnchannels(), audio.getsampwidth(),
                      audio.getframerate(), audio.getcomptype()) ==
                     (1, 2, 24000, "NONE") and
                     audio.getnframes() == samples and
                     len(audio.readframes(samples)) == samples * 2)
        valid = valid and path.stat().st_size == 44 + 2 * samples
    except (OSError, ValueError, TypeError, wave.Error) as error:
        raise PlaybackError("Kokoro did not publish a valid WAV") from error
    if not valid:
        raise PlaybackError("Kokoro WAV format or sample count mismatch")


def _write_private_report(path: Path, result: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def play(text: str, config: Path, bundle: Path, player: Path,
         *, synthesizer: Path | None = None,
         technical_mode: str = "natural") -> dict:
    if not text.strip() or "\x00" in text:
        raise PlaybackError("spoken text is empty or invalid")
    if not player.is_file() or not os.access(player, os.X_OK):
        raise PlaybackError("selected audio player is unavailable")
    synthesizer = synthesizer or Path(__file__).with_name("speak_kokoro_text.py")
    with tempfile.TemporaryDirectory(prefix="cke-kokoro-play-") as folder:
        root = Path(folder)
        source, output, report = (root / "text.txt", root / "speech.wav",
                                  root / "report.json")
        source.write_text(text, encoding="utf-8")
        command = [sys.executable, str(synthesizer), "--text-file", str(source),
                   "--config", str(config), "--bundle-dir", str(bundle),
                   "--output", str(output), "--report", str(report)]
        if technical_mode != "natural":
            command.extend(("--technical-mode", technical_mode))
        _run(command, capture=True)
        try:
            result = json.loads(report.read_text(encoding="utf-8"))
            _checked_wav(output, result["sample_count"])
        except (OSError, ValueError, TypeError, KeyError, wave.Error) as error:
            raise PlaybackError("Kokoro did not publish a valid WAV and report") from error
        source_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if (result.get("status") != "execution_pass" or
                result.get("source_sha256") != source_hash):
            raise PlaybackError("Kokoro output failed its playback contract")
        _run([str(player), str(output)], capture=False)
        return result


def _exchange(worker: subprocess.Popen, command: str, prefix: str) -> list[str]:
    if "\r" in command or "\n" in command:
        raise PlaybackError("invalid native command")
    assert worker.stdin is not None and worker.stdout is not None
    worker.stdin.write(command + "\n")
    worker.stdin.flush()
    response = worker.stdout.readline().rstrip("\n")
    fields = response.split("\t")
    if fields[0] != prefix:
        raise PlaybackError(f"native worker rejected {command.split(chr(9))[0]}: "
                            f"{response or 'EOF'}")
    return fields


def play_incremental(text: str, config: Path, bundle: Path, player: Path,
                     *, seed: int = 7, synthesis_workers: int = 1,
                     technical_mode: str = "natural",
                     prepare_plan=None) -> dict:
    """Play checked chunks in order with a bounded number of loaded workers."""
    if not player.is_file() or not os.access(player, os.X_OK):
        raise PlaybackError("selected audio player is unavailable")
    if seed < 0 or seed > (1 << 64) - 1:
        raise PlaybackError("seed must be an unsigned 64-bit integer")
    if type(synthesis_workers) is not int or not 1 <= synthesis_workers <= 3:
        raise PlaybackError("synthesis workers must be an integer in 1..3")
    if prepare_plan is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from prepare_kokoro_accessibility_text import prepare as prepare_plan
        finally:
            sys.path.pop(0)
    started = time.monotonic()
    plan = (prepare_plan(text, config, bundle)
            if technical_mode == "natural" else
            prepare_plan(text, config, bundle, technical_mode=technical_mode))
    chunks = plan["chunks"]
    if not chunks or len(chunks) > 512 or seed > (1 << 64) - len(chunks):
        raise PlaybackError("incremental request exceeds checked chunk or seed capacity")
    if any(type(chunk.get("index")) is not int or chunk["index"] != position
           for position, chunk in enumerate(chunks)):
        raise PlaybackError("incremental chunks must have consecutive indices")
    dispatcher = bundle / "native-kokoro-dispatch"
    if not dispatcher.is_file():
        raise PlaybackError("native Kokoro dispatcher unavailable")
    active_workers = min(synthesis_workers, len(chunks))
    with tempfile.TemporaryDirectory(prefix="cke-kokoro-incremental-") as folder:
        workers = []
        executor = ThreadPoolExecutor(max_workers=active_workers)
        playing = None
        total_samples = 0
        played = []
        first_play = None
        try:
            for _ in range(active_workers):
                workers.append(subprocess.Popen(
                    [str(dispatcher), str(bundle), "--session"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, bufsize=1))
            for worker in workers:
                assert worker.stdout is not None
                if worker.stdout.readline().strip() != "READY\tloads=1":
                    raise PlaybackError("native Kokoro session did not become ready")

            def synthesize(chunk: dict, worker: subprocess.Popen) -> tuple:
                index = chunk["index"]
                output = Path(folder) / f"chunk-{index}.wav"
                ids = ",".join(map(str, chunk["phoneme_ids"]))
                synthesis_started = time.monotonic()
                fields = _exchange(worker, f"RUN\t{output}\t{seed + index}\t{ids}",
                                   "WAV_READY")
                if len(fields) != 3 or fields[2] != str(output):
                    raise PlaybackError("native chunk response has wrong output path")
                try:
                    samples = int(fields[1])
                except ValueError as error:
                    raise PlaybackError("native chunk response has invalid sample count") from error
                _checked_wav(output, samples)
                wav_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
                return output, samples, synthesis_started, time.monotonic(), wav_sha256

            pending = {index: executor.submit(synthesize, chunks[index], workers[index])
                       for index in range(active_workers)}
            previous_end = None
            for index, chunk in enumerate(chunks):
                output, samples, synthesis_started, ready, wav_sha256 = (
                    pending.pop(index).result())
                following = index + active_workers
                if following < len(chunks):
                    pending[following] = executor.submit(
                        synthesize, chunks[following], workers[index % active_workers])
                if playing is not None:
                    previous_end = playing.finish()
                    playing.wav.unlink()
                playing = _Playing(player, output)
                launched = time.monotonic()
                if first_play is None:
                    first_play = launched
                total_samples += samples
                played.append({"index": index, "source_start": chunk["source_start"],
                               "source_end": chunk["source_end"],
                               "source_text": chunk["source_text"],
                               "samples": samples, "audio_seconds": samples / 24000,
                               "wav_sha256": wav_sha256,
                               "synthesis_seconds": ready - synthesis_started,
                               "playback_gap_seconds": (max(0.0, launched - previous_end)
                                                        if previous_end is not None else None)})
            if playing is not None:
                playing.finish()
                playing.wav.unlink()
                playing = None
            for worker in workers:
                assert worker.stdin is not None
                worker.stdin.write("QUIT\n")
                worker.stdin.flush()
                worker.stdin.close()
            for worker in workers:
                if worker.wait(timeout=10):
                    raise PlaybackError("native Kokoro worker exited with failure")
        finally:
            try:
                if playing is not None:
                    playing.stop()
            finally:
                for worker in workers:
                    if worker.poll() is None:
                        try:
                            worker.kill()
                        except ProcessLookupError:
                            pass
                        worker.wait()
                executor.shutdown(wait=True, cancel_futures=True)
                for worker in workers:
                    if worker.stdin and not worker.stdin.closed:
                        try:
                            worker.stdin.close()
                        except BrokenPipeError:
                            pass
                    if worker.stdout:
                        worker.stdout.close()
                    if worker.stderr:
                        worker.stderr.close()
    return {"status": "execution_pass", "source_sha256": hashlib.sha256(
                plan["source_text"].encode("utf-8")).hexdigest(),
            "bundle_identity_sha256": plan["bundle_identity_sha256"],
            "bundle_manifest_sha256": plan.get("bundle_manifest_sha256"),
            "technical_mode": technical_mode,
            "pronunciation_expansions": plan.get("pronunciation_expansions", []),
            "synthesis_workers": active_workers,
            "native_model_loads": len(workers),
            "chunks": played, "sample_count": total_samples,
            "audio_seconds": total_samples / 24000,
            "playback_gap_total_seconds": sum(
                item["playback_gap_seconds"] or 0.0 for item in played),
            "first_playback_start_seconds": first_play - started,
            "total_seconds": time.monotonic() - started,
            "known_connected_waveform_parity": "fail_unresolved"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--incremental", action="store_true",
                        help="play checked chunks while preparing subsequent chunks")
    parser.add_argument("--synthesis-workers", type=int, default=1,
                        help="1..3 loaded native workers for incremental playback")
    parser.add_argument("--technical-mode", choices=("natural", "exact"),
                        default="natural",
                        help="pronounce path components naturally or spell each character")
    parser.add_argument("--report", type=Path,
                        help="write completed playback metrics and source spans")
    parser.add_argument("--text-file", type=Path,
                        help="read the complete reply from a UTF-8 file")
    parser.add_argument("text", nargs="?")
    args = parser.parse_args()

    def interrupted(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if (args.text is None) == (args.text_file is None):
            raise PlaybackError("provide text or --text-file, but not both")
        if not 1 <= args.synthesis_workers <= 3:
            raise PlaybackError("synthesis workers must be in 1..3")
        if not args.incremental and args.synthesis_workers != 1:
            raise PlaybackError("multiple synthesis workers require --incremental")
        if args.report and (args.report.exists() or args.report.is_symlink()):
            raise PlaybackError("report already exists")
        text = args.text_file.read_text(encoding="utf-8") if args.text_file else args.text
        result = (play_incremental(text, args.config, args.bundle_dir,
                                   args.player,
                                   synthesis_workers=args.synthesis_workers,
                                   technical_mode=args.technical_mode)
                  if args.incremental else
                  play(text, args.config, args.bundle_dir, args.player,
                       technical_mode=args.technical_mode))
        if args.report:
            _write_private_report(args.report, result)
        print(f"CKE Kokoro played {result['sample_count']} samples", flush=True)
        return 0
    except (PlaybackError, OSError, ValueError) as error:
        print(f"CKE Kokoro playback failed: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("CKE Kokoro playback interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
