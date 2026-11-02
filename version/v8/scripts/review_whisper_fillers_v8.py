#!/usr/bin/env python3
"""Review Whisper filler intervals without modifying the source recording.

`propose` writes private context WAVs and an unapproved manifest. `render`
requires a separate, explicit approval file naming the reviewed previews and
the exact cut intervals. Neither alignment nor filler recognition is certified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import wave


FILLERS = frozenset({"um", "uh", "erm", "hmm"})
CONTEXT_SECONDS = 0.75
MAX_CANDIDATE_SECONDS = 5.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_open(path: Path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def _seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite time")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite time")
    return result


def _audio_info(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels() != 1 or source.getsampwidth() != 2 or
                source.getcomptype() != "NONE" or source.getframerate() <= 0):
            raise ValueError("review requires uncompressed mono PCM16 WAV")
        return source.getframerate(), source.getnframes()


def _interval(start: object, end: object, rate: int, frames: int) -> tuple[int, int]:
    first = round(_seconds(start, "start") * rate)
    last = round(_seconds(end, "end") * rate)
    if not 0 <= first < last <= frames:
        raise ValueError("filler interval exceeds audio or has no samples")
    if last - first > MAX_CANDIDATE_SECONDS * rate:
        raise ValueError("filler interval exceeds the bounded review length")
    return first, last


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("review input must be a JSON object")
    return value


def _write_json_private(path: Path, value: dict) -> None:
    with _private_open(path) as output:
        output.write((json.dumps(value, indent=2) + "\n").encode())


def _candidate(label: str, start: object, end: object, source: str,
               rate: int, frames: int, detail: dict) -> dict:
    first, last = _interval(start, end, rate, frames)
    margin = round(CONTEXT_SECONDS * rate)
    return {"label": label, "source": source,
            "proposed_frames": [first, last],
            "context_frames": [max(0, first - margin), min(frames, last + margin)],
            "detail": detail, "review_required": True}


def propose(audio_path: Path, run_report_path: Path, assembly_path: Path, output_dir: Path,
            manual_path: Path | None = None) -> dict:
    rate, frames = _audio_info(audio_path)
    audio_hash = sha256(audio_path)
    run_report = _read_json(run_report_path)
    assembly = _read_json(assembly_path)
    if (assembly.get("schema") != "cke.whisper_word_assembly.v1" or
            assembly.get("edit_ready") is not False or
            assembly.get("provenance", {}).get("audio_sha256") != audio_hash or
            assembly.get("provenance", {}).get("run_report_sha256") !=
            sha256(run_report_path) or run_report.get("wav_sha256") != audio_hash):
        raise ValueError("assembly and run report are not bound to this audio")
    candidates = []
    unlocated = []
    committed = assembly.get("words")
    review_words = assembly.get("review_words")
    if not isinstance(committed, list) or not isinstance(review_words, list):
        raise ValueError("assembly has no checked word lists")
    for word in committed + review_words:
        if not isinstance(word, dict):
            raise ValueError("assembly has a malformed word")
        label = str(word.get("text", "")).casefold().strip(".,!?;:")
        if label not in FILLERS:
            continue
        detail = {"audio_window": word.get("audio_window"),
                  "window_word_index": word.get("window_word_index"),
                  "assembly_review_reasons": word.get("assembly_review_reasons", [])}
        if word.get("start_seconds") is None or word.get("end_seconds") is None:
            unlocated.append({"label": label, **detail,
                              "reason": "transcribed_filler_missing_timing"})
            continue
        candidates.append(_candidate(label, word["start_seconds"],
                                     word["end_seconds"], "transcribed_filler",
                                     rate, frames, detail))

    manual_hash = None
    if manual_path is not None:
        flags = _read_json(manual_path)
        manual_hash = sha256(manual_path)
        if (flags.get("schema") != "cke.whisper_manual_filler_flags.v1" or
                flags.get("audio_sha256") != audio_hash or
                not isinstance(flags.get("flags"), list)):
            raise ValueError("manual filler flags disagree with source audio")
        for index, flag in enumerate(flags["flags"]):
            if not isinstance(flag, dict):
                raise ValueError("manual filler flag must be an object")
            label = flag.get("label")
            if not isinstance(label, str) or not label.strip() or len(label) > 64:
                raise ValueError("manual filler label is invalid")
            note = flag.get("note", "")
            if not isinstance(note, str) or len(note) > 256:
                raise ValueError("manual filler note is invalid")
            candidates.append(_candidate(label.strip(), flag.get("start_seconds"),
                                         flag.get("end_seconds"), "manual_flag", rate,
                                         frames, {"manual_flag_index": index,
                                                  "note": note}))

    candidates.sort(key=lambda item: (item["proposed_frames"][0], item["source"]))
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    with wave.open(str(audio_path), "rb") as source:
        for index, candidate in enumerate(candidates):
            candidate["id"] = f"f{index:04d}"
            filename = f"{candidate['id']}-context.wav"
            first, last = candidate["context_frames"]
            source.setpos(first)
            preview = output_dir / filename
            audio_bytes = source.readframes(last - first)
            if len(audio_bytes) != (last - first) * 2:
                raise ValueError("source WAV ended before its declared frame count")
            with _private_open(preview) as raw, wave.open(raw, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(rate)
                output.writeframes(audio_bytes)
            candidate["preview_file"] = filename
            candidate["preview_sha256"] = sha256(preview)
    result = {
        "schema": "cke.whisper_filler_review.v1", "status": "AWAITING_HUMAN_REVIEW",
        "audio_sha256": audio_hash, "assembly_sha256": sha256(assembly_path),
        "run_report_sha256": sha256(run_report_path),
        "manual_flags_sha256": manual_hash, "sample_rate": rate, "source_frames": frames,
        "candidates": candidates, "unlocated_transcribed_fillers": unlocated,
        "transcription_omissions": "not_measured_by_review_tool",
        "edited_audio": None,
    }
    _write_json_private(output_dir / "proposals.json", result)
    return result


def _copy_frames(source, output, count: int) -> None:
    remaining = count
    while remaining:
        length = min(remaining, 65536)
        data = source.readframes(length)
        if len(data) != length * 2:
            raise ValueError("source WAV ended before its declared frame count")
        output.writeframes(data)
        remaining -= length


def render(audio_path: Path, proposal_path: Path, approval_path: Path,
           output_path: Path, evidence_path: Path) -> dict:
    rate, frames = _audio_info(audio_path)
    audio_hash = sha256(audio_path)
    proposal = _read_json(proposal_path)
    approval = _read_json(approval_path)
    if (proposal.get("schema") != "cke.whisper_filler_review.v1" or
            proposal.get("status") != "AWAITING_HUMAN_REVIEW" or
            proposal.get("audio_sha256") != audio_hash or
            proposal.get("sample_rate") != rate or
            proposal.get("source_frames") != frames):
        raise ValueError("proposal does not belong to this audio")
    reviewer = approval.get("approved_by")
    selections = approval.get("approved")
    if (approval.get("schema") != "cke.whisper_filler_approval.v1" or
            approval.get("audio_sha256") != audio_hash or
            approval.get("proposal_sha256") != sha256(proposal_path) or
            not isinstance(reviewer, str) or not reviewer.strip() or
            not isinstance(selections, list) or not selections):
        raise ValueError("explicit approval for this audio and proposal is required")
    by_id = {item["id"]: item for item in proposal.get("candidates", [])}
    if len(by_id) != len(proposal.get("candidates", [])):
        raise ValueError("proposal candidate IDs are ambiguous")
    cuts = []
    seen = set()
    for selection in selections:
        if not isinstance(selection, dict) or selection.get("id") not in by_id:
            raise ValueError("approval names an unknown candidate")
        candidate = by_id[selection["id"]]
        if selection["id"] in seen:
            raise ValueError("candidate was approved more than once")
        seen.add(selection["id"])
        preview_name = candidate.get("preview_file")
        if (not isinstance(preview_name, str) or
                Path(preview_name).name != preview_name):
            raise ValueError("proposal preview path is invalid")
        preview_path = proposal_path.parent / preview_name
        preview_hash = candidate.get("preview_sha256")
        if (selection.get("reviewed_preview_sha256") != preview_hash or
                sha256(preview_path) != preview_hash):
            raise ValueError("approved preview identity does not match")
        first, last = _interval(selection.get("cut_start_seconds"),
                                selection.get("cut_end_seconds"), rate, frames)
        context_first, context_last = candidate["context_frames"]
        if not context_first <= first < last <= context_last:
            raise ValueError("approved cut exceeds its reviewed audio context")
        cuts.append({"id": selection["id"], "start_frame": first,
                     "end_frame": last, "reviewed_preview_sha256": preview_hash})
    cuts.sort(key=lambda item: item["start_frame"])
    if any(left["end_frame"] > right["start_frame"]
           for left, right in zip(cuts, cuts[1:])):
        raise ValueError("approved cuts overlap")
    if (output_path.resolve() == audio_path.resolve() or
            evidence_path.resolve() in {audio_path.resolve(), output_path.resolve()} or
            output_path.exists() or evidence_path.exists()):
        raise ValueError("render output paths must be new and separate from source")

    created_output = False
    created_evidence = False
    try:
        with wave.open(str(audio_path), "rb") as source:
            raw = _private_open(output_path)
            created_output = True
            with raw, wave.open(raw, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(rate)
                cursor = 0
                for cut in cuts:
                    _copy_frames(source, output, cut["start_frame"] - cursor)
                    source.setpos(cut["end_frame"])
                    cursor = cut["end_frame"]
                _copy_frames(source, output, frames - cursor)
        result = {"schema": "cke.whisper_filler_render.v1",
                  "source_audio_sha256": audio_hash,
                  "proposal_sha256": sha256(proposal_path),
                  "approval_sha256": sha256(approval_path),
                  "output_audio_sha256": sha256(output_path),
                  "approved_by": reviewer.strip(), "cuts": cuts,
                  "source_frames": frames,
                  "output_frames": frames - sum(c["end_frame"] - c["start_frame"]
                                                for c in cuts),
                  "sample_rate": rate}
        raw_evidence = _private_open(evidence_path)
        created_evidence = True
        with raw_evidence:
            raw_evidence.write((json.dumps(result, indent=2) + "\n").encode())
    except Exception:
        if created_output:
            output_path.unlink(missing_ok=True)
        if created_evidence:
            evidence_path.unlink(missing_ok=True)
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    propose_parser = sub.add_parser("propose")
    propose_parser.add_argument("--audio", type=Path, required=True)
    propose_parser.add_argument("--run-report", type=Path, required=True)
    propose_parser.add_argument("--assembly", type=Path, required=True)
    propose_parser.add_argument("--manual-flags", type=Path)
    propose_parser.add_argument("--output-dir", type=Path, required=True)
    render_parser = sub.add_parser("render")
    render_parser.add_argument("--audio", type=Path, required=True)
    render_parser.add_argument("--proposals", type=Path, required=True)
    render_parser.add_argument("--approval", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "propose":
        result = propose(args.audio, args.run_report, args.assembly,
                         args.output_dir, args.manual_flags)
        print(f"{len(result['candidates'])} proposals; "
              f"{len(result['unlocated_transcribed_fillers'])} unlocated fillers; "
              f"review {args.output_dir / 'proposals.json'}")
    else:
        result = render(args.audio, args.proposals, args.approval,
                        args.output, args.evidence)
        print(f"approved render: {result['output_frames']} frames -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
