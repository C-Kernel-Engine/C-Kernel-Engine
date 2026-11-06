#!/usr/bin/env python3
"""Evaluate a generated Kokoro WAV transcribed by generated CKE Whisper.

This is an application-level diagnostic. It does not certify Kokoro waveform
parity, human listening quality, or that Whisper recognized every spoken word.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import wave


# A sign is meaningful after whitespace or punctuation, but the hyphen in
# identifiers such as PR-1035 must not turn 1035 into a negative quantity.
NUMBER_PATTERN = r"(?<![A-Za-z0-9])[-+]\d+(?:[,.]\d+)*|\d+(?:[,.]\d+)*"
WORDS = re.compile(NUMBER_PATTERN + r"|[a-z]+(?:'[a-z]+)?")
NUMBERS = re.compile(NUMBER_PATTERN)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(text: str) -> list[str]:
    return WORDS.findall(text.casefold())


def _distance(reference: list[str], candidate: list[str]) -> int:
    previous = list(range(len(candidate) + 1))
    for index, expected in enumerate(reference, 1):
        current = [index]
        for position, actual in enumerate(candidate, 1):
            current.append(min(current[-1] + 1, previous[position] + 1,
                               previous[position - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def _wav(path: Path) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as audio:
        rate = audio.getframerate()
        frames = audio.getnframes()
        return {"channels": audio.getnchannels(), "sample_width": audio.getsampwidth(),
                "sample_rate": rate, "sample_count": frames,
                "duration_seconds": frames / rate}


def evaluate(source_path: Path, plan_path: Path, tts_path: Path, wav_path: Path,
             asr_wav_path: Path, asr_path: Path, resample_path: Path,
             *, maximum_wer: float = 0.1) -> dict:
    if not 0 <= maximum_wer <= 1:
        raise ValueError("maximum WER must be between zero and one")
    source = source_path.read_text(encoding="utf-8").strip()
    if not source:
        raise ValueError("source text is empty")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tts = json.loads(tts_path.read_text(encoding="utf-8"))
    asr = json.loads(asr_path.read_text(encoding="utf-8"))
    resample = json.loads(resample_path.read_text(encoding="utf-8"))
    wav = _wav(wav_path)
    asr_wav = _wav(asr_wav_path)
    wav_hash = _sha256(wav_path)
    asr_wav_hash = _sha256(asr_wav_path)
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

    chunks = plan.get("chunks", [])
    cursor = 0
    for index, chunk in enumerate(chunks):
        start, end = chunk.get("source_start"), chunk.get("source_end")
        if start != cursor or not isinstance(end, int) or end <= start or end > len(source):
            raise ValueError(f"text chunk {index} leaves a gap, overlap, or invalid span")
        cursor = end
    if not chunks or cursor != len(source) or plan.get("source_text") != source:
        raise ValueError("text plan does not cover the exact source")
    if (tts.get("status") != "execution_pass" or
            tts.get("source_sha256") != source_hash or
            tts.get("wav_sha256") != wav_hash or
            len(tts.get("chunks", [])) != len(chunks) or
            tts.get("sample_count") != wav["sample_count"] or
            tts.get("sample_rate") != wav["sample_rate"] or
            wav["channels"] != 1 or wav["sample_width"] != 2):
        raise ValueError("TTS report or WAV does not match the source and plan")
    if (resample.get("source_sha256") != wav_hash or
            resample.get("target_sha256") != asr_wav_hash or
            resample.get("source_frames") != wav["sample_count"] or
            resample.get("source_rate") != wav["sample_rate"] or
            resample.get("target_frames") != asr_wav["sample_count"] or
            resample.get("target_rate") != asr_wav["sample_rate"] or
            resample.get("provider") != "audio_resample_windowed_sinc_f32"):
        raise ValueError("resampling provenance or geometry does not match the audio")
    if abs(wav["duration_seconds"] - asr_wav["duration_seconds"]) > 1 / asr_wav["sample_rate"]:
        raise ValueError("resampling changed the audio duration")
    if (asr_wav["sample_rate"] != 16000 or asr_wav["channels"] != 1 or
            asr_wav["sample_width"] != 2):
        raise ValueError("Whisper input must be 16 kHz mono PCM16")
    if (asr.get("status") != "ok" or asr.get("wav_sha256") != asr_wav_hash or
            asr.get("timestamps") is not True or
            not asr.get("encoder_runtime_sha256") or
            not asr.get("decoder_runtime_sha256") or
            asr.get("decoder", {}).get("stop") != "segment_complete"):
        raise ValueError("ASR report lacks completed timestamp-mode generated execution")
    segments = asr.get("segments", [])
    frame = 0
    for index, segment in enumerate(segments):
        start = segment.get("source_frame_start")
        end = segment.get("source_frame_consumed_end")
        window_end = segment.get("source_frame_window_end")
        if (segment.get("index") != index or start != frame or
                not isinstance(end, int) or not isinstance(window_end, int) or
                not start < end <= window_end <= asr_wav["sample_count"] or
                abs(segment.get("start_seconds", -1) - start / 16000) > 1 / 16000 or
                abs(segment.get("end_seconds", -1) - end / 16000) > 1 / 16000):
            raise ValueError(f"ASR segment {index} leaves a gap, overlap, or invalid extent")
        frame = end
    if not segments or frame != asr_wav["sample_count"]:
        raise ValueError("ASR did not consume the complete audio")

    transcript = asr["decoder"].get("transcript_text")
    if not isinstance(transcript, str):
        raise ValueError("ASR transcript is missing")
    expected = _tokens(source)
    observed = _tokens(transcript)
    wer = _distance(expected, observed) / len(expected)
    expected_numbers = NUMBERS.findall(source)
    observed_numbers = NUMBERS.findall(transcript)
    missing_numbers = list((Counter(expected_numbers) - Counter(observed_numbers)).elements())
    unexpected_numbers = list((Counter(observed_numbers) - Counter(expected_numbers)).elements())
    return {
        "schema": "cke.kokoro_whisper.long_reply_diagnostic.v1",
        "source_file_sha256": _sha256(source_path),
        "source_text_sha256": source_hash,
        "tts_wav_sha256": wav_hash,
        "asr_wav_sha256": asr_wav_hash,
        "bundle_identity_sha256": tts.get("bundle_identity_sha256"),
        "asr_encoder_runtime_sha256": asr["encoder_runtime_sha256"],
        "asr_decoder_runtime_sha256": asr["decoder_runtime_sha256"],
        "source_words": len(source.split()),
        "planned_chunks": len(chunks),
        "wav_seconds": wav["duration_seconds"],
        "asr_segments": len(segments),
        "transcript_words": len(transcript.split()),
        "word_error_rate": wer,
        "maximum_word_error_rate": maximum_wer,
        "missing_exact_numbers": missing_numbers,
        "unexpected_exact_numbers": unexpected_numbers,
        "execution": "pass",
        "transcript_fidelity": "pass" if wer <= maximum_wer else "fail",
        "exact_number_fidelity": "pass" if not missing_numbers and not unexpected_numbers else "fail",
        "waveform_numerical_parity": tts.get("numerical_waveform_parity", "not_tested"),
        "known_connected_waveform_parity": tts.get("known_connected_waveform_parity", "not_tested"),
        "human_listening": "not_tested",
        "transcript": transcript,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "plan", "tts_report", "wav", "asr_wav", "asr_report",
                 "resample_report"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--maximum-wer", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.source, args.plan, args.tts_report, args.wav,
                      args.asr_wav, args.asr_report, args.resample_report,
                      maximum_wer=args.maximum_wer)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if (result["transcript_fidelity"] == "pass" and
                 result["exact_number_fidelity"] == "pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
