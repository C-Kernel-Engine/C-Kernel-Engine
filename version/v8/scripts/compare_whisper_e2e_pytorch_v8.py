#!/usr/bin/env python3
"""Compare generated CKE Whisper transcription with Hugging Face PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_whisper_v8.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(checkpoint: Path) -> dict[str, dict[str, object]]:
    files = sorted(
        path
        for path in checkpoint.iterdir()
        if path.is_file()
        and path.suffix in {".json", ".safetensors", ".txt", ".model"}
    )
    return {
        str(path.relative_to(checkpoint)): {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    }


def validate_runtime_checkpoint(
    run_dir: Path,
    expected: dict[str, dict[str, object]],
    role: str,
) -> None:
    stamp_path = run_dir / ".ck-whisper-runtime.json"
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"missing or invalid Whisper {role} provenance stamp: {stamp_path}"
        ) from exc
    inputs = stamp.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("role") != role:
        raise RuntimeError(
            f"Whisper {role} provenance has the wrong runtime role: {stamp_path}"
        )
    if inputs.get("checkpoint") != expected:
        raise RuntimeError(
            f"Whisper {role} runtime was not built from the requested "
            f"checkpoint: {run_dir}"
        )


def _read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError("PyTorch Whisper oracle requires PCM16 WAV")
        rate = source.getframerate()
        channels = source.getnchannels()
        raw = source.readframes(source.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    samples = samples.reshape(-1, channels).mean(axis=1)
    return samples / 32768.0, rate


def first_token_difference(
    subject: list[int], oracle: list[int]
) -> dict[str, int] | None:
    common = min(len(subject), len(oracle))
    for index in range(common):
        if subject[index] != oracle[index]:
            return {
                "index": index,
                "subject": subject[index],
                "oracle": oracle[index],
            }
    if len(subject) != len(oracle):
        return {
            "index": common,
            "subject": subject[common] if common < len(subject) else -1,
            "oracle": oracle[common] if common < len(oracle) else -1,
        }
    return None


def load_attention_trace(captured: dict, token_ids: list[int], heads: list[list[int]]) -> np.ndarray:
    """Require identity and semantic row alignment before reference comparison."""
    path = Path(captured["path"])
    if _sha256(path) != captured["sha256"]:
        raise ValueError("attention trace identity changed")
    with np.load(path, allow_pickle=False) as trace:
        if str(trace["schema"]) != "cke.whisper_selected_attention.v2":
            raise ValueError("unsupported attention trace schema")
        if trace["token_ids"].tolist() != token_ids:
            raise ValueError("attention trace token correspondence changed")
        if trace["alignment_heads"].tolist() != heads:
            raise ValueError("attention trace alignment heads changed")
        actual = trace["selected_attention"].copy()
    if (actual.ndim != 3 or actual.shape[:2] != (len(heads), len(token_ids))
            or actual.shape[2] == 0 or not np.isfinite(actual).all()):
        raise ValueError("attention trace has invalid shape or nonfinite values")
    return actual


def _pytorch_reference(args: argparse.Namespace, cke: dict | None = None) -> dict[str, object]:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    samples, sample_rate = _read_pcm16(args.wav)
    load_started = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(
        args.checkpoint, local_files_only=True
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    load_elapsed = time.perf_counter() - load_started
    long_form = samples.size > int(processor.feature_extractor.n_samples)
    frontend_started = time.perf_counter()
    inputs = processor(
        samples,
        sampling_rate=sample_rate,
        return_tensors="pt",
        **(
            {
                "truncation": False,
                "padding": "longest",
                "return_attention_mask": True,
            }
            if long_form
            else {}
        ),
    )
    frontend_elapsed = time.perf_counter() - frontend_started
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            inputs.input_features,
            **(
                {"attention_mask": inputs.attention_mask}
                if long_form
                else {}
            ),
            language=args.language,
            task=args.task,
            return_timestamps=args.timestamps,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            **({"prompt_ids": processor.get_prompt_ids(args.vocabulary_hint, return_tensors="pt")}
               if getattr(args, "vocabulary_hint", "") else {}),
        )
    elapsed = time.perf_counter() - started
    tokens = [int(value) for value in generated[0].tolist()]
    attention_reference = None
    if getattr(args, "attention_trace_dir", None) is not None:
        if cke is None or len(cke.get("segments", [])) != 1:
            raise ValueError("attention comparison currently requires one audio window")
        decoder = cke["segments"][0]["decoder"]
        captured = decoder["attention_trace"]
        prefix = decoder["forced_prefix"]
        ids = decoder["generated_tokens"]
        if not ids:
            raise ValueError("attention comparison requires generated tokens")
        heads = model.generation_config.alignment_heads
        actual = load_attention_trace(captured, ids, heads)
        with torch.inference_mode():
            replay = model(inputs.input_features,
                           decoder_input_ids=torch.tensor([prefix + ids[:-1]]),
                           output_attentions=True)
        expected = np.stack([
            replay.cross_attentions[layer][0, head, len(prefix)-1:, :].cpu().numpy()
            for layer, head in heads])
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError("attention comparison has invalid shape or nonfinite values")
        error = actual.astype(np.float64) - expected.astype(np.float64)
        attention_reference = {
            "status": "measured_not_certified", "shape": list(actual.shape),
            "max_abs": float(np.abs(error).max()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "trace_sha256": captured["sha256"],
            "mode": "teacher-forced eager FP32; same decoder IDs, independent frontend and encoder",
            "limitations": "Not identical-input local replay or human boundary accuracy; no fitted tolerance.",
        }
    return {
        "tokens": tokens,
        "text": processor.decode(
            generated[0],
            skip_special_tokens=True,
            decode_with_timestamps=args.timestamps,
            clean_up_tokenization_spaces=False,
        ),
        "transcript_text": processor.decode(
            generated[0],
            skip_special_tokens=True,
            decode_with_timestamps=False,
            clean_up_tokenization_spaces=False,
        ),
        "long_form": long_form,
        "load_seconds": load_elapsed,
        "frontend_seconds": frontend_elapsed,
        "generation_seconds": elapsed,
        "seconds": elapsed,
        "pytorch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "model_dtype": str(next(model.parameters()).dtype),
        "attention_implementation": "eager",
        "attention_reference": attention_reference,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-run-dir", type=Path, required=True)
    parser.add_argument("--decoder-run-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--task", choices=("transcribe", "translate"), default="transcribe"
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--vocabulary-hint", default="")
    parser.add_argument("--attention-trace-dir", type=Path,
                        help="Retain traces and measure single-window teacher-forced reference error")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    args.checkpoint = args.checkpoint.resolve()
    args.encoder_run_dir = args.encoder_run_dir.resolve()
    args.decoder_run_dir = args.decoder_run_dir.resolve()
    args.wav = args.wav.resolve()
    checkpoint = checkpoint_identity(args.checkpoint)
    validate_runtime_checkpoint(args.encoder_run_dir, checkpoint, "encoder")
    validate_runtime_checkpoint(args.decoder_run_dir, checkpoint, "decoder")
    os.environ["CK_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    with tempfile.TemporaryDirectory(prefix="cke-whisper-pytorch-") as text:
        cke_path = Path(text) / "cke.json"
        subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "run",
                "--encoder-run-dir",
                str(args.encoder_run_dir),
                "--decoder-run-dir",
                str(args.decoder_run_dir),
                "--wav",
                str(args.wav),
                "--language",
                args.language,
                "--task",
                args.task,
                "--max-tokens",
                str(args.max_tokens),
                *(["--timestamps"] if args.timestamps else []),
                *([f"--vocabulary-hint={args.vocabulary_hint}"] if args.vocabulary_hint else []),
                *(["--attention-trace-dir", str(args.attention_trace_dir)] if args.attention_trace_dir else []),
                "--output",
                str(cke_path),
            ],
            check=True,
        )
        cke = json.loads(cke_path.read_text(encoding="utf-8"))

    oracle = _pytorch_reference(args, cke)
    subject_tokens = [
        int(value) for value in cke["decoder"]["generated_tokens"]
    ]
    oracle_tokens = [int(value) for value in oracle["tokens"]]
    difference = first_token_difference(subject_tokens, oracle_tokens)
    transcript_match = (
        str(cke["decoder"]["transcript_text"])
        == str(oracle["transcript_text"])
    )
    report = {
        "schema": "cke.whisper_e2e_pytorch_parity",
        "schema_version": 2,
        "verdict_scope": "generated_token_trajectory",
        "status": (
            "pass"
            if difference is None and transcript_match
            else "fail"
        ),
        "language": args.language,
        "task": args.task,
        "timestamps": bool(args.timestamps),
        "max_tokens": args.max_tokens,
        "threads": args.threads,
        "wav": str(args.wav),
        "wav_sha256": _sha256(args.wav),
        "checkpoint": str(args.checkpoint),
        "checkpoint_files": checkpoint,
        "cke": cke,
        "pytorch": oracle,
        "matched_tokens": (
            len(subject_tokens)
            if difference is None
            else int(difference["index"])
        ),
        "subject_token_count": len(subject_tokens),
        "oracle_token_count": len(oracle_tokens),
        "first_difference": difference,
        "transcript_match": transcript_match,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"status={report['status']} "
        f"matched={report['matched_tokens']}/"
        f"{max(len(subject_tokens), len(oracle_tokens))} "
        f"report={args.output}"
    )
    return 0 if difference is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
