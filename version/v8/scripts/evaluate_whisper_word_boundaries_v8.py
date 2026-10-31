#!/usr/bin/env python3
"""Score native Whisper word timing against independently checked word spans.

Reference audio and annotations stay local. This tool reports transcription,
filler recall, and matched-word boundary error as distinct measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


FILLERS = frozenset({"um", "uh", "erm", "hmm"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_word(value: str) -> str:
    return re.sub(r"[^\w'’-]", "", value.casefold()).replace("’", "'")


def edit_alignment(reference: list[str], predicted: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Return a deterministic minimum-edit path, preserving repeated-word positions."""
    rows, columns = len(reference), len(predicted)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        cost[i][0] = i
    for j in range(columns + 1):
        cost[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + (reference[i - 1] != predicted[j - 1]),
                             cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    operations = []
    i, j = rows, columns
    while i or j:
        # Exact matches take priority; otherwise preserve a deterministic path.
        if i and j and reference[i - 1] == predicted[j - 1] and cost[i][j] == cost[i - 1][j - 1]:
            operations.append(("match", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and j and cost[i][j] == cost[i - 1][j - 1] + 1:
            operations.append(("substitution", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            operations.append(("deletion", i - 1, None))
            i -= 1
        else:
            operations.append(("insertion", None, j - 1))
            j -= 1
    operations.reverse()
    return operations


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _errors_ms(values: list[float]) -> dict:
    return {"count": len(values), "mean": sum(values) / len(values) if values else None,
            "median": _percentile(values, 50), "p95": _percentile(values, 95),
            "maximum": max(values) if values else None,
            "within_100_ms": sum(value <= 100 for value in values),
            "within_200_ms": sum(value <= 200 for value in values)}


def evaluate(reference: dict, alignment: dict) -> dict:
    reference_words = reference.get("words")
    predicted_words = alignment.get("words")
    if not isinstance(reference_words, list) or not isinstance(predicted_words, list):
        raise ValueError("reference and alignment require word lists")
    ref = [normalized_word(str(word["word"])) for word in reference_words]
    pred = [normalized_word(str(word["text"])) for word in predicted_words]
    if not ref or any(not any(character.isalnum() for character in word)
                      for word in ref + pred):
        raise ValueError("empty or nonlexical word in comparison")
    path = edit_alignment(ref, pred)
    ambiguous_repeats = set()
    for op, reference_index, predicted_index in path:
        if op == "match":
            continue
        for word in (ref[reference_index] if reference_index is not None else None,
                     pred[predicted_index] if predicted_index is not None else None):
            if word is not None and (ref.count(word) > 1 or pred.count(word) > 1):
                ambiguous_repeats.add(word)
    counts = {name: sum(op == name for op, _, _ in path)
              for name in ("match", "substitution", "deletion", "insertion")}
    start_errors, end_errors, omitted_fillers, matched_fillers = [], [], [], []
    missing_timing = []
    ambiguous_repeat_timing = []
    boundary_rows = []
    for op, reference_index, predicted_index in path:
        if op == "deletion" and ref[reference_index] in FILLERS:
            omitted_fillers.append({"index": reference_index, "word": ref[reference_index]})
        if op == "substitution" and ref[reference_index] in FILLERS:
            omitted_fillers.append({"index": reference_index, "word": ref[reference_index]})
        if op != "match":
            continue
        if ref[reference_index] in FILLERS:
            matched_fillers.append({"index": reference_index, "word": ref[reference_index]})
        if ref[reference_index] in ambiguous_repeats:
            ambiguous_repeat_timing.append(predicted_index)
            continue
        expected = reference_words[reference_index]
        actual = predicted_words[predicted_index]
        start, end = actual.get("start_seconds"), actual.get("end_seconds")
        if start is None or end is None:
            missing_timing.append(predicted_index)
            continue
        if not (0 <= float(start) <= float(end)):
            raise ValueError("invalid predicted word boundary")
        expected_start = float(expected["start_ms"]) / 1000.0
        expected_end = float(expected["end_ms"]) / 1000.0
        if not (0 <= expected_start < expected_end):
            raise ValueError("invalid reference word boundary")
        start_error = abs(float(start) - expected_start) * 1000.0
        end_error = abs(float(end) - expected_end) * 1000.0
        start_errors.append(start_error)
        end_errors.append(end_error)
        boundary_rows.append({"reference_index": reference_index,
                              "predicted_index": predicted_index,
                              "word": ref[reference_index],
                              "reference_start_seconds": expected_start,
                              "reference_end_seconds": expected_end,
                              "predicted_start_seconds": float(start),
                              "predicted_end_seconds": float(end),
                              "start_error_ms": start_error,
                              "end_error_ms": end_error})
    return {
        "schema": "cke.whisper_word_boundary_evaluation.v1",
        "status": "MEASURED_NOT_CERTIFIED",
        "sample_id": reference.get("id"),
        "reference_words": len(ref), "predicted_words": len(pred),
        "transcription": {**counts,
                          "word_error_rate": (counts["substitution"] + counts["deletion"] +
                                              counts["insertion"]) / len(ref),
                          "edits": [{"operation": op, "reference_index": i,
                                     "predicted_index": j,
                                     "reference_word": ref[i] if i is not None else None,
                                     "predicted_word": pred[j] if j is not None else None}
                                    for op, i, j in path if op != "match"]},
        "fillers": {"reference_count": sum(word in FILLERS for word in ref),
                    "matched_count": len(matched_fillers),
                    "omitted": omitted_fillers},
        "boundaries": {"matched_words_with_timing": len(start_errors),
                       "matched_words_missing_timing": missing_timing,
                       "matched_words_excluded_ambiguous_repeat": ambiguous_repeat_timing,
                       "rows": boundary_rows,
                       "worst": max(boundary_rows,
                                    key=lambda row: max(row["start_error_ms"],
                                                        row["end_error_ms"])) if boundary_rows else None,
                       "start_error_ms": _errors_ms(start_errors),
                       "end_error_ms": _errors_ms(end_errors),
                       "combined_error_ms": _errors_ms(start_errors + end_errors)},
    }


def validate_provenance(run_report: dict, alignment: dict, audio_sha256: str) -> str:
    if run_report.get("wav_sha256") != audio_sha256:
        raise ValueError("run report audio hash disagrees with supplied recording")
    if alignment.get("schema") != "cke.whisper_word_alignment.v1":
        raise ValueError("unsupported alignment report schema")
    window_index = alignment.get("window", {}).get("index")
    if not isinstance(window_index, int) or isinstance(window_index, bool) or window_index < 0:
        raise ValueError("alignment report has no valid window identity")
    segments = run_report.get("segments")
    if isinstance(segments, list) and segments:
        if window_index >= len(segments):
            raise ValueError("alignment window exceeds run report")
        decoder = segments[window_index].get("decoder", {})
    elif window_index == 0:
        decoder = run_report.get("decoder", {})
    else:
        raise ValueError("single-window report cannot describe another window")
    trace_hash = alignment.get("provenance", {}).get("trace_sha256")
    if (not isinstance(trace_hash, str) or
            (decoder.get("attention_trace") or {}).get("sha256") != trace_hash):
        raise ValueError("alignment trace identity disagrees with its audio window")
    return trace_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.reference_manifest.read_text())
    samples = [sample for sample in manifest.get("samples", [])
               if sample.get("id") == args.sample_id]
    if len(samples) != 1:
        raise ValueError("sample ID must identify exactly one reference recording")
    sample = samples[0]
    if Path(sample["audio"]).name != args.audio.name:
        raise ValueError("reference sample audio name disagrees with supplied recording")
    report = json.loads(args.run_report.read_text())
    audio_hash = sha256(args.audio)
    alignment = json.loads(args.alignment.read_text())
    trace_hash = validate_provenance(report, alignment, audio_hash)
    result = evaluate(sample, alignment)
    result["provenance"] = {
        "reference_manifest_sha256": sha256(args.reference_manifest),
        "audio_sha256": audio_hash, "run_report_sha256": sha256(args.run_report),
        "alignment_report_sha256": sha256(args.alignment),
        "trace_sha256": trace_hash,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.sample_id}: WER={result['transcription']['word_error_rate']:.3f}, "
          f"matched={result['boundaries']['matched_words_with_timing']}, "
          f"boundary median={result['boundaries']['combined_error_ms']['median']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
