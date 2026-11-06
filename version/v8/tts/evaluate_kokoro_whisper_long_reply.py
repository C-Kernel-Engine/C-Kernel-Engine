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


WORDS = re.compile(r"[+-]?\d+(?:[,.]\d+)*|[a-z]+(?:'[a-z]+)?")
NUMBERS = re.compile(r"[+-]?\d+(?:[,.]\d+)*")
IDENTIFIER_HYPHEN = re.compile(r"\b(PR|issue|RFC)[-–](?=\d)", re.IGNORECASE)
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9}
SPOKEN_NUMBER = re.compile(
    r"\b(?:(minus|negative|plus|positive)\s+)?("
    + "|".join(TENS) + r")(?:[ -]+(" + "|".join(UNITS) + r"))?\b",
    re.IGNORECASE,
)
SPOKEN_SIGN_DIGITS = re.compile(
    r"\b(minus|negative|plus|positive)\s+(\d+(?:[,.]\d+)*)\b",
    re.IGNORECASE,
)
UNSUPPORTED_NUMBER_WORDS = re.compile(
    r"\b(?:hundred|thousand|million|billion|trillion)\b|"
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|\d+)"
    r"\s+point\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|\d+)\b",
    re.IGNORECASE,
)
SPACED_SIGN = re.compile(r"(?<!\w)[+-]\s+(?=\d)")
SYMBOL_SIGN_WORD = re.compile(
    r"(?<!\w)[+-]\s*(?=zero|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b",
    re.IGNORECASE,
)


def _identifier_aware(text: str) -> str:
    # PR-1035 is an identifier. A general x-3.5 expression still keeps its
    # minus sign; only declared identifier prefixes receive this normalization.
    return IDENTIFIER_HYPHEN.sub(lambda match: match.group(1) + " ", text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(text: str) -> list[str]:
    return WORDS.findall(_identifier_aware(text).casefold())


def _number_facts(text: str) -> list[str]:
    # This deliberately supports only tens and optional units. A pause
    # rendered as "40, two" is not merged into 42.
    def spoken_value(match: re.Match[str]) -> str:
        value = TENS[match.group(2).casefold()]
        if match.group(3):
            value += UNITS[match.group(3).casefold()]
        sign = match.group(1)
        if not sign:
            return str(value)
        prefix = "-" if sign.casefold() in ("minus", "negative") else "+"
        return prefix + str(value)

    normalized = SPOKEN_NUMBER.sub(spoken_value, _identifier_aware(text))
    normalized = SPOKEN_SIGN_DIGITS.sub(
        lambda match: ("-" if match.group(1).casefold() in ("minus", "negative")
                       else "+") + match.group(2), normalized)
    return NUMBERS.findall(normalized)


def _unsupported_number_phrases(text: str) -> list[str]:
    # Do not certify a trailing supported fragment of "one hundred and
    # forty two", or silently discard an unparsed spoken decimal/sign.
    phrases = [match.group() for match in UNSUPPORTED_NUMBER_WORDS.finditer(text)]
    phrases.extend(match.group() for match in SPACED_SIGN.finditer(text))
    phrases.extend(match.group() for match in SYMBOL_SIGN_WORD.finditer(text))
    normalized = SPOKEN_NUMBER.sub("42", _identifier_aware(text))
    normalized = SPOKEN_SIGN_DIGITS.sub("-42", normalized)
    unsupported_sign = re.compile(
        r"\b(?:minus|negative|plus|positive)\s+"
        r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
        r"eighteen|nineteen|\d+)\b", re.IGNORECASE)
    phrases.extend(match.group() for match in unsupported_sign.finditer(normalized))
    return phrases


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

    max_request_ids = plan.get("max_request_ids")
    if type(max_request_ids) is not int or not 3 <= max_request_ids <= 48:
        raise ValueError("text plan lacks a valid request capacity")
    chunks = plan.get("chunks", [])
    cursor = 0
    for index, chunk in enumerate(chunks):
        if (not isinstance(chunk, dict) or
                any(type(chunk.get(field)) is not int for field in
                    ("source_start", "source_end", "valid_token_count", "voice_row")) or
                not isinstance(chunk.get("source_text"), str)):
            raise ValueError(f"text chunk {index} lacks required typed metadata")
        start, end = chunk["source_start"], chunk["source_end"]
        if (start != cursor or end <= start or end > len(source) or
                not chunk["source_text"] or
                chunk["source_text"] != source[start:end].strip() or
                not 3 <= chunk["valid_token_count"] <= max_request_ids or
                chunk["voice_row"] != chunk["valid_token_count"] - 3):
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
    sample_total = 0
    for index, (planned, rendered) in enumerate(zip(chunks, tts["chunks"])):
        if (not isinstance(rendered, dict) or
                any(type(rendered.get(field)) is not int for field in
                    ("index", "source_start", "source_end", "valid_token_count",
                     "voice_row", "samples")) or
                not isinstance(rendered.get("source_text"), str) or
                rendered["index"] != index or
                any(rendered[field] != planned[field] for field in
                    ("source_start", "source_end", "source_text",
                     "valid_token_count", "voice_row")) or
                rendered["samples"] <= 0):
            raise ValueError(f"TTS chunk {index} does not match its planned request")
        sample_total += rendered["samples"]
    if sample_total != wav["sample_count"]:
        raise ValueError("TTS chunk sample counts do not match the WAV")
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
    expected_numbers = _number_facts(source)
    observed_numbers = _number_facts(transcript)
    unsupported_numbers = (_unsupported_number_phrases(source) +
                           _unsupported_number_phrases(transcript))
    missing_numbers = list((Counter(expected_numbers) - Counter(observed_numbers)).elements())
    unexpected_numbers = list((Counter(observed_numbers) - Counter(expected_numbers)).elements())
    number_sequence_match = expected_numbers == observed_numbers
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
        "number_sequence_match": number_sequence_match,
        "unsupported_number_phrases": unsupported_numbers,
        "execution": "pass",
        "transcript_fidelity": "pass" if wer <= maximum_wer else "fail",
        "exact_number_fidelity": ("incomplete" if unsupported_numbers else
                                  "pass" if number_sequence_match else "fail"),
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
