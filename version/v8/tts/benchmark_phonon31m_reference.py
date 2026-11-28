"""Measure the pinned Phonon publisher stream on a local text file."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
import wave

import numpy as np


RELEASE = "8f7a37c94a4c58b04c8287300c53f11d2e208eae"
RUNTIME = "0.8.0"


def stream_case(tts, name, text, folder, stop_after=None):
    start = perf_counter()
    samples = chunks = 0
    first_pcm = playback_until = interrupted_at = None
    total_gap = worst_gap = 0.0
    digest = hashlib.sha256()
    with tts.stream(text, seed=42) as stream, wave.open(str(folder / f"{name}.wav"), "wb") as wav:
        rate = stream.sample_rate
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        for pcm in stream:
            arrived = perf_counter() - start
            pcm = np.ascontiguousarray(pcm, dtype="<f4")
            if pcm.ndim != 1 or not pcm.size or not np.isfinite(pcm).all():
                raise ValueError("publisher returned invalid mono PCM")
            chunks += 1
            samples += pcm.size
            if first_pcm is None:
                first_pcm = playback_until = arrived
            gap = max(0.0, arrived - playback_until)
            total_gap += gap
            worst_gap = max(worst_gap, gap)
            playback_until = max(playback_until, arrived) + pcm.size / rate
            digest.update(pcm.tobytes())
            wav.writeframesraw(np.rint(np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
            if chunks == stop_after:
                interrupted_at = perf_counter()
                break
    finished = perf_counter()
    if not samples:
        raise RuntimeError("publisher stream produced no PCM")
    duration = samples / rate
    return {
        "sample_rate": rate, "samples": samples, "chunks": chunks, "audio_seconds": duration,
        "tt_first_pcm_seconds": first_pcm, "wall_seconds": finished - start,
        "wall_per_audio_second": (finished - start) / duration,
        "simulated_playback_gap_seconds": total_gap, "simulated_worst_gap_seconds": worst_gap,
        "stream_exit_seconds": finished - interrupted_at if interrupted_at is not None else None,
        "pcm_float_sha256": digest.hexdigest(),
    }


def main():
    from importlib.metadata import version

    import ptts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if version("ptts") != RUNTIME:
        parser.error(f"expected ptts=={RUNTIME}")
    data = args.text.read_bytes()
    text = data.decode()
    if not text.strip():
        parser.error("text file is empty")
    args.output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    tts = ptts.TTS(f"hf://gradium/phonon-31m-en@{RELEASE}", lang="en", quant="q8_0", device="cpu", seed=42)
    report = {
        "model_revision": RELEASE, "publisher_runtime": RUNTIME, "quant": "q8_0",
        "platform": platform.platform(), "requested_rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
        "text_sha256": hashlib.sha256(data).hexdigest(), "words": len(text.split()),
        "load_seconds": perf_counter() - start, "cases": {},
    }
    for name, request, stop_after in (
        ("smoke", "Hello from CKE.", None),
        ("interrupt", text, 10),
        ("full", text, None),
        ("repeat_smoke", "Hello from CKE.", None),
    ):
        report["cases"][name] = stream_case(tts, name, request, args.output, stop_after)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_rss_kib"] = peak_rss // 1024 if sys.platform == "darwin" else peak_rss
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
