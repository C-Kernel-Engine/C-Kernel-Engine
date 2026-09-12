#!/usr/bin/env python3
"""Capture a compact, pinned Parakeet TDT reference fixture.

This is a certification/conversion utility. It is not part of native CKE inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
TRANSFORMERS_REVISION = "66799f45cea7513712580c6170cdaa4438df702a"
EXPECTED_HASHES = {
    "model.safetensors": "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11",
    "config.json": "e747b85e1bdfd300c8b8ac63bac8dd5221f8fe9bc275b48d06c735fcd6971b6e",
    "generation_config.json": "b141de6ec6d7f982ece13f98f604e3fe1807ea9c0e839185d0ab7064604209d0",
    "processor_config.json": "8346a93a3b987fa1dec57a78f045cd0817d21786589a5a096b41a57a446fd1d7",
    "tokenizer.json": "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91",
    "tokenizer_config.json": "0b2fe0037599ee335f0b972fa682bf0ece74e4ccfec755cb7daa3405d3d3e874",
    "audio": "5fceacff0315d49cb59fcc505bcecf1ed5f2f35c2897b1e65a59f30e5d922150",
}
EXPECTED_SOURCE_HASHES = {
    "modeling_parakeet.py": "61c7e96716e20411650a8190213eb5f19916dfb05a8ffbed5e123cb8769ee69c",
    "generation_parakeet.py": "ee5ac5310d699f7d48d9c04e525bebae6b905e663f74fb418470ce514703d0ba",
    "feature_extraction_parakeet.py": "5ec8a9664456edfc0b56c64083bdcdd74dd78364504d35207ab917d1c3b9f575",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_array(value: Any) -> np.ndarray:
    if isinstance(value, (tuple, list)):
        value = value[0]
    return value.detach().cpu().contiguous().numpy()


def array_record(value: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(value)
    record: dict[str, Any] = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "finite": bool(finite.all()),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }
    if value.size:
        fp = value.astype(np.float64, copy=False)
        record.update({
            "min": float(fp.min()),
            "max": float(fp.max()),
            "mean": float(fp.mean()),
        })
    return record


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--transformers-revision", required=True)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoModelForTDT, AutoProcessor
    from transformers.models.parakeet import (
        feature_extraction_parakeet,
        generation_parakeet,
        modeling_parakeet,
    )

    if args.model_revision != MODEL_REVISION:
        raise ValueError(f"expected model revision {MODEL_REVISION}, got {args.model_revision}")
    if args.transformers_revision != TRANSFORMERS_REVISION:
        raise ValueError(
            f"expected Transformers revision {TRANSFORMERS_REVISION}, got {args.transformers_revision}"
        )
    installed_sources = {
        "modeling_parakeet.py": Path(modeling_parakeet.__file__).resolve(),
        "generation_parakeet.py": Path(generation_parakeet.__file__).resolve(),
        "feature_extraction_parakeet.py": Path(feature_extraction_parakeet.__file__).resolve(),
    }
    installed_source_hashes = {
        name: sha256_file(path) for name, path in installed_sources.items()
    }
    if installed_source_hashes != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(
            "installed Parakeet reference source does not match the pinned Transformers commit: "
            f"{installed_source_hashes}"
        )
    for name, expected in EXPECTED_HASHES.items():
        path = args.audio if name == "audio" else args.model_dir / name
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {path}: expected {expected}, got {actual}")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if audio.ndim != 1:
        audio = audio.mean(axis=-1, dtype=np.float32)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForTDT.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.float32
    ).eval()
    inputs = processor(audio=audio, sampling_rate=sample_rate, return_tensors="pt")

    arrays: dict[str, np.ndarray] = {
        "audio.samples": np.ascontiguousarray(audio, dtype=np.float32),
        "frontend.input_features": tensor_array(inputs["input_features"]),
        "frontend.attention_mask": tensor_array(inputs["attention_mask"]),
    }
    hooks = []

    def capture_once(name: str):
        def hook(_module, _inputs, output):
            if name not in arrays:
                arrays[name] = tensor_array(output)
        return hook

    hooks.append(model.encoder.subsampling.register_forward_hook(capture_once("encoder.subsampling")))
    hooks.append(model.encoder.layers[0].register_forward_hook(capture_once("encoder.layer.0")))
    hooks.append(model.encoder.layers[-1].register_forward_hook(capture_once("encoder.layer.23")))
    hooks.append(model.encoder_projector.register_forward_hook(capture_once("encoder.projected")))
    hooks.append(model.decoder.register_forward_hook(capture_once("decoder.first_output")))
    hooks.append(model.joint.register_forward_hook(capture_once("joint.first_logits")))

    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, return_dict_in_generate=True)
    elapsed = time.perf_counter() - started
    for hook in hooks:
        hook.remove()

    arrays["decode.sequences"] = tensor_array(generated.sequences)
    arrays["decode.durations"] = tensor_array(generated.durations)
    decoded = processor.decode(
        generated.sequences,
        durations=generated.durations,
        skip_special_tokens=True,
    )
    if isinstance(decoded, tuple):
        transcript, timestamps = decoded
    else:
        transcript, timestamps = decoded, None
    if isinstance(transcript, list) and len(transcript) == 1:
        transcript = transcript[0]
    if isinstance(timestamps, list) and len(timestamps) == 1:
        timestamps = timestamps[0]

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    records = {name: array_record(array) for name, array in sorted(arrays.items())}
    required = {
        "audio.samples", "frontend.input_features", "frontend.attention_mask",
        "encoder.subsampling", "encoder.layer.0", "encoder.layer.23",
        "encoder.projected", "decoder.first_output", "joint.first_logits",
        "decode.sequences", "decode.durations",
    }
    missing = sorted(required - arrays.keys())
    finite = all(record["finite"] for record in records.values())
    config_files = {}
    for name in ("config.json", "generation_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json"):
        path = args.model_dir / name
        config_files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    report = {
        "schema_version": 1,
        "kind": "cke.parakeet.reference_fixture",
        "status": "pass" if not missing and finite and generated.sequences.numel() > 1 else "fail",
        "scope": "one short FP32 CPU reference trajectory; no CKE parity claim",
        "model": {
            "id": "nvidia/parakeet-tdt-0.6b-v3",
            "revision": args.model_revision,
            "weights": {"sha256": sha256_file(args.model_dir / "model.safetensors"), "bytes": (args.model_dir / "model.safetensors").stat().st_size},
            "files": config_files,
        },
        "reference": {
            "implementation": "huggingface/transformers",
            "revision": args.transformers_revision,
            "transformers_version": transformers.__version__,
            "source_sha256": installed_source_hashes,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "threads": args.threads,
            "dtype": "float32",
        },
        "audio": {
            "path": args.audio.name,
            "source_url": "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav",
            "source_context": "NVIDIA model-card example; LibriSpeech utterance 2086-149220-0033",
            "sha256": sha256_file(args.audio),
            "sample_rate": int(sample_rate),
            "samples": int(audio.shape[0]),
            "seconds": float(audio.shape[0] / sample_rate),
            "channels": 1,
        },
        "execution": {"elapsed_seconds": elapsed},
        "arrays_file": {"path": args.output_npz.name, "sha256": sha256_file(args.output_npz), "arrays": records},
        "decode": {
            "transcript": str(transcript),
            "timestamps": json_value(timestamps),
            "steps": int(generated.sequences.shape[1]),
            "nonblank_steps": int((generated.sequences != model.config.blank_token_id).sum().item()),
            "blank_token_id": int(model.config.blank_token_id),
            "durations": list(model.config.durations),
        },
        "checks": {"required_arrays_present": not missing, "missing_arrays": missing, "all_arrays_finite": finite, "nonempty_trajectory": generated.sequences.numel() > 1},
        "limitations": [
            "This fixture records the reference path only; it does not establish native CKE parity.",
            "It covers one 7.435-second English utterance and no silence, multilingual, long-audio, chunk-boundary, or diarization behavior.",
            "Timestamps are the pinned Transformers decoder convention and still require comparison with NeMo before CKE support is promoted.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "elapsed_seconds": elapsed, "transcript": transcript, "steps": report["decode"]["steps"], "npz": str(args.output_npz), "json": str(args.output_json)}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
