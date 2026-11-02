#!/usr/bin/env python3
"""Play one checked Kokoro whole-text WAV for the local voice client.

The caller owns cancellation of this process group. The model schedule remains
in the generated runtime invoked by speak_kokoro_text.py.
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
import wave


class PlaybackError(RuntimeError):
    pass


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


def play(text: str, config: Path, bundle: Path, player: Path,
         *, synthesizer: Path | None = None) -> dict:
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
        _run([sys.executable, str(synthesizer), "--text-file", str(source),
              "--config", str(config), "--bundle-dir", str(bundle),
              "--output", str(output), "--report", str(report)], capture=True)
        try:
            result = json.loads(report.read_text(encoding="utf-8"))
            with wave.open(str(output), "rb") as audio:
                valid = ((audio.getnchannels(), audio.getsampwidth(),
                          audio.getframerate(), audio.getcomptype()) ==
                         (1, 2, 24000, "NONE") and
                         audio.getnframes() == result["sample_count"] > 0 and
                         len(audio.readframes(audio.getnframes())) ==
                         result["sample_count"] * 2)
            valid = valid and output.stat().st_size == 44 + 2 * result["sample_count"]
        except (OSError, ValueError, TypeError, KeyError, wave.Error) as error:
            raise PlaybackError("Kokoro did not publish a valid WAV and report") from error
        source_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if (not valid or result.get("status") != "execution_pass" or
                result.get("source_sha256") != source_hash):
            raise PlaybackError("Kokoro output failed its playback contract")
        _run([str(player), str(output)], capture=False)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("text")
    args = parser.parse_args()

    def interrupted(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        result = play(args.text, args.config, args.bundle_dir, args.player)
        print(f"CKE Kokoro played {result['sample_count']} samples", flush=True)
        return 0
    except PlaybackError as error:
        print(f"CKE Kokoro playback failed: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("CKE Kokoro playback interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
