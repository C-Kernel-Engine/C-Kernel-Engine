#!/usr/bin/env python3
"""Capture a pinned Qwen3-TTS 0.6B CustomVoice CPU oracle run.

This is reference tooling. It does not participate in CKE model execution.
The asset-backed capture is opt-in; the cheap contract tests run in nightly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODEL_REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"
REFERENCE_REPO = "https://github.com/QwenLM/Qwen3-TTS"
REFERENCE_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
DEFAULT_TEXT = "Hello, this is CKE."
DEFAULT_SPEAKER = "Ryan"
DEFAULT_LANGUAGE = "English"
MODEL_FILE_SHA256 = {
    "config.json": "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455",
    "generation_config.json": "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model.safetensors": "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb",
    "preprocessor_config.json": "efdde1022ea9d76928bf7a9cd53139138f5ba2e466e837f08f6105ab1af1c119",
    "speech_tokenizer/config.json": "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
    "speech_tokenizer/configuration.json": "6bc26d64eb5024b4d1dab5a52371958b429256d6c9d59787f1f5294a54e0cebd",
    "speech_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
    "speech_tokenizer/preprocessor_config.json": "fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb",
    "tokenizer_config.json": "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
REFERENCE_FILE_SHA256 = {
    "inference/qwen3_tts_model.py": "e1da450732857c1f5fe3e36ebab85db2f6dc6a48caaff6973f463384e30275e4",
    "core/models/modeling_qwen3_tts.py": "25c42656bcf810f06ef6bc1839bd7083f3c8cfedac3a147c4060b4262b1c96a0",
    "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py": "844e8dd8c0182ef9c6463c874631c22ef3c5a4fd1899dd657016164cc5379628",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_request(text: str, max_new_tokens: int) -> None:
    if not text or not text.strip():
        raise ValueError("text must contain spoken characters")
    if "\x00" in text:
        raise ValueError("text contains NUL")
    if max_new_tokens < 2 or max_new_tokens > 4096:
        raise ValueError("max_new_tokens must be between 2 and 4096")


def verify_model_dir(model_dir: Path) -> dict[str, str]:
    required = tuple(MODEL_FILE_SHA256)
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing pinned model files: {', '.join(missing)}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("tts_model_size") != "0b6" or config.get("tts_model_type") != "custom_voice":
        raise ValueError("model is not Qwen3-TTS 0.6B CustomVoice")
    if model_dir.parent.name == "snapshots" and model_dir.name != MODEL_REVISION:
        raise ValueError("Hugging Face snapshot revision differs from the pinned revision")
    hashes = {name: sha256_file(model_dir / name) for name in required}
    for name, expected in MODEL_FILE_SHA256.items():
        if hashes[name] != expected:
            raise ValueError(f"pinned model checksum mismatch: {name}")
    return hashes


def verify_reference_code(package_dir: Path) -> dict[str, str]:
    hashes = {}
    for name, expected in REFERENCE_FILE_SHA256.items():
        path = package_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned reference code: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"pinned reference code checksum mismatch: {name}")
        hashes[name] = actual
    return hashes


def capture(model_dir: Path, output_dir: Path, text: str, speaker: str,
            language: str, seed: int, max_new_tokens: int) -> dict:
    validate_request(text, max_new_tokens)
    model_hashes = verify_model_dir(model_dir)

    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    import qwen_tts
    from qwen_tts import Qwen3TTSModel

    reference_hashes = verify_reference_code(Path(qwen_tts.__file__).resolve().parent)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, int(os.environ.get("CKE_QWEN3_TTS_TORCH_THREADS", "4"))))
    started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(model_dir), device_map="cpu", dtype=torch.float32,
        attn_implementation="eager",
    )
    load_seconds = time.perf_counter() - started

    captured: dict[str, object] = {}
    original_generate = model.model.generate

    def record_projection(_, inputs, output):
        if inputs[0].ndim == 3 and inputs[0].shape[1] == captured.get("context_tokens"):
            # The role and single-token projections can have the same row
            # count as a short spoken context. The full-context projection is
            # the last matching call in non-streaming prompt assembly.
            captured["context_projection_input"] = inputs[0].detach().cpu().numpy().copy()
            captured["context_projection_output"] = output.detach().cpu().numpy().copy()

    def record_fc1(_, inputs, output):
        if inputs[0].ndim == 3 and inputs[0].shape[1] == captured.get("context_tokens"):
            captured["context_fc1_output"] = output.detach().cpu().numpy().copy()

    projection_hook = model.model.talker.text_projection.register_forward_hook(record_projection)
    fc1_hook = model.model.talker.text_projection.linear_fc1.register_forward_hook(record_fc1)

    def record_generate(*args, **kwargs):
        captured["text_token_ids"] = [item.detach().cpu().tolist() for item in kwargs["input_ids"]]
        captured["context_token_ids"] = kwargs["input_ids"][0][:, 3:-5].detach().cpu().tolist()
        captured["context_tokens"] = len(captured["context_token_ids"][0])
        result = original_generate(*args, **kwargs)
        captured["codec_tokens"] = [item.detach().cpu().tolist() for item in result[0]]
        return result

    model.model.generate = record_generate
    try:
        with torch.inference_mode():
            started = time.perf_counter()
            wavs, sample_rate = model.generate_custom_voice(
                text=text, speaker=speaker, language=language,
                do_sample=False, subtalker_dosample=False,
                max_new_tokens=max_new_tokens,
                non_streaming_mode=True,
            )
            generate_seconds = time.perf_counter() - started
    finally:
        model.model.generate = original_generate
        projection_hook.remove()
        fc1_hook.remove()

    if len(wavs) != 1 or not captured.get("codec_tokens") or "context_projection_output" not in captured:
        raise RuntimeError("reference did not return waveform, codec sequence and context projection")
    with torch.inference_mode():
        expected_projection_input = model.model.talker.get_text_embeddings()(
            torch.tensor(captured["context_token_ids"], device=model.model.talker.device)
        ).detach().cpu().numpy()
    if not np.array_equal(captured["context_projection_input"], expected_projection_input):
        raise RuntimeError("captured projection does not match the spoken-text IDs")
    waveform = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    if not waveform.size or not np.isfinite(waveform).all():
        raise RuntimeError("reference returned empty or nonfinite audio")
    if sample_rate <= 0:
        raise RuntimeError("reference returned invalid sample rate")

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "qwen3_tts_customvoice_reference.wav"
    codes_path = output_dir / "qwen3_tts_customvoice_reference.npz"
    sf.write(wav_path, waveform, sample_rate, subtype="PCM_16")
    np.savez_compressed(
        codes_path, waveform=waveform,
        text_token_ids=np.asarray(captured["text_token_ids"][0], dtype=np.int64),
        context_token_ids=np.asarray(captured["context_token_ids"], dtype=np.int64),
        context_projection_input=captured["context_projection_input"],
        context_fc1_output=captured["context_fc1_output"],
        context_projection_output=captured["context_projection_output"],
        codec_tokens=np.asarray(captured["codec_tokens"][0], dtype=np.int64),
    )
    report = {
        "schema": 1,
        "status": "PASS",
        "scope": "official PyTorch reference only; CKE generated execution NOT_TESTED",
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION, "files_sha256": model_hashes},
        "reference": {"repo": REFERENCE_REPO, "revision": REFERENCE_REVISION,
                      "files_sha256": reference_hashes},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "transformers": transformers.__version__, "numpy": np.__version__,
                    "device": "cpu", "dtype": "float32", "attention": "eager",
                    "threads": torch.get_num_threads(), "machine": platform.machine()},
        "request": {"text": text, "speaker": speaker, "language": language,
                    "seed": seed, "do_sample": False, "subtalker_dosample": False,
                    "max_new_tokens": max_new_tokens},
        "result": {"sample_rate": sample_rate, "samples": int(waveform.size),
                   "text_tokens": int(np.asarray(captured["text_token_ids"][0]).size),
                   "context_tokens": captured["context_tokens"],
                   "codec_frames": len(captured["codec_tokens"][0]),
                   "waveform_peak": float(np.max(np.abs(waveform))),
                   "load_seconds": load_seconds, "generate_seconds": generate_seconds,
                   "wav_sha256": sha256_file(wav_path), "npz_sha256": sha256_file(codes_path)},
    }
    (output_dir / "qwen3_tts_customvoice_reference.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    report = capture(args.model_dir, args.output_dir, args.text, args.speaker,
                     args.language, args.seed, args.max_new_tokens)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
