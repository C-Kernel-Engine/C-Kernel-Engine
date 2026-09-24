#!/usr/bin/env python3
"""Capture one pinned Kokoro v1.0 reference utterance from local assets.

The script is oracle tooling only. It uses upstream Python at export time and
performs no network access. Runtime TTS must use native CKE operations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import wave

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np


HERE = Path(__file__).resolve().parent
PIN = json.loads((HERE / "kokoro_v1_reference.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_tensor(value, stem: str, out_dir: Path, records: dict) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        path = out_dir / f"{stem}.npy"
        np.save(path, array, allow_pickle=False)
        records[stem] = {
            "file": path.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": sha256(path),
        }
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            capture_tensor(item, f"{stem}_{index}", out_dir, records)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Local pinned HF snapshot containing config, weights and voice")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()

    fixture = PIN["fixture"]
    model_dir = args.model_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "config.json"
    weight_path = model_dir / "kokoro-v1_0.pth"
    voice_path = model_dir / "voices" / "af_heart.pt"
    required = [config_path] if args.preprocess_only else [config_path, weight_path, voice_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing pinned local asset(s): " + ", ".join(missing))

    import torch
    from kokoro import KModel, KPipeline

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.manual_seed(fixture["torch_seed"])
    pipeline = KPipeline(lang_code="a", repo_id=PIN["model"]["repository"], model=False, trf=False)
    segments = list(pipeline(fixture["text"], model=False))
    if len(segments) != 1 or not segments[0].phonemes:
        raise RuntimeError(f"expected one nonempty Kokoro segment, got {len(segments)}")
    phonemes = segments[0].phonemes
    config = json.loads(config_path.read_text())
    ids = [0, *(config["vocab"].get(phone) for phone in phonemes), 0]
    if any(token is None for token in ids):
        raise RuntimeError("phoneme outside pinned config vocabulary")
    if len(phonemes) > 510:
        raise RuntimeError("fixture exceeds Kokoro's 510-phoneme segment limit")

    record = {
        "schema_version": 1,
        "status": "preprocessing_captured" if args.preprocess_only else "full_oracle_captured",
        "pin": PIN,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {name: package_version(name) for name in (
                "kokoro", "misaki", "torch", "transformers", "spacy",
                "num2words", "phonemizer-fork", "espeakng-loader")},
        },
        "assets": {str(path.relative_to(model_dir)): sha256(path) for path in required},
        "graphemes": segments[0].graphemes,
        "phonemes": phonemes,
        "phoneme_codepoints": [f"U+{ord(char):04X}" for char in phonemes],
        "input_ids": ids,
        "tensors": {},
        "evidence": {
            "upstream_preprocessing": "CAPTURED",
            "upstream_waveform": "NOT_TESTED" if args.preprocess_only else "CAPTURED",
            "native_primitive_parity": "NOT_TESTED",
            "native_full_waveform_parity": "NOT_TESTED",
            "human_listening": "NOT_TESTED",
            "application_playback": "NOT_TESTED",
        },
    }
    if args.preprocess_only:
        (out_dir / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        return 0

    voice_pack = torch.load(voice_path, map_location="cpu", weights_only=True)
    voice_row = voice_pack[len(phonemes) - 1]
    if voice_row.ndim == 1:
        voice_row = voice_row.unsqueeze(0)
    if tuple(voice_row.shape) != (1, 256):
        raise RuntimeError(f"unexpected voice style shape {tuple(voice_row.shape)}")
    capture_tensor(voice_row, "voice_style", out_dir, record["tensors"])
    capture_tensor(voice_row[:, :128], "decoder_style", out_dir, record["tensors"])
    capture_tensor(voice_row[:, 128:], "predictor_style", out_dir, record["tensors"])
    record["voice_row_index"] = len(phonemes) - 1

    model = KModel(repo_id=PIN["model"]["repository"],
                   config=str(config_path), model=str(weight_path)).eval()
    module_names = (
        "bert", "bert_encoder", "predictor.text_encoder", "predictor.lstm",
        "predictor.duration_proj", "predictor.shared", "predictor.F0_proj",
        "predictor.N_proj", "text_encoder", "decoder", "decoder.generator",
        "decoder.generator.conv_post",
    )
    modules = dict(model.named_modules())
    hooks = []
    for name in module_names:
        module = modules.get(name)
        if module is None:
            record.setdefault("unavailable_hooks", []).append(name)
            continue
        def on_output(_module, _inputs, output, label=name):
            capture_tensor(output, label.replace(".", "_"), out_dir, record["tensors"])
        hooks.append(module.register_forward_hook(on_output))
    stft = model.decoder.generator.stft
    original_inverse = stft.inverse
    def capture_inverse(spec, phase):
        capture_tensor(spec, "istft_magnitude", out_dir, record["tensors"])
        capture_tensor(phase, "istft_phase", out_dir, record["tensors"])
        return original_inverse(spec, phase)
    stft.inverse = capture_inverse
    try:
        output = model(phonemes, voice_row, speed=fixture["speed"], return_output=True)
    finally:
        stft.inverse = original_inverse
        for hook in hooks:
            hook.remove()
    capture_tensor(output.pred_dur, "predicted_duration", out_dir, record["tensors"])
    durations = output.pred_dur.detach().cpu().reshape(-1).to(torch.int64)
    if len(durations) != len(ids) or (durations < 1).any():
        raise RuntimeError("oracle duration shape or lower bound mismatch")
    frame_to_token = torch.repeat_interleave(torch.arange(len(ids)), durations)
    capture_tensor(frame_to_token, "frame_to_token", out_dir, record["tensors"])
    capture_tensor(output.audio, "waveform_f32", out_dir, record["tensors"])
    audio = output.audio.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    if not np.isfinite(audio).all() or len(audio) == 0:
        raise RuntimeError("oracle produced empty or nonfinite waveform")
    wav_path = out_dir / "waveform_pcm16.wav"
    write_wav(wav_path, audio, fixture["sample_rate_hz"])
    record["waveform"] = {
        "frames": len(audio), "seconds": len(audio) / fixture["sample_rate_hz"],
        "min": float(audio.min()), "max": float(audio.max()),
        "pcm16_file": wav_path.name, "pcm16_sha256": sha256(wav_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
