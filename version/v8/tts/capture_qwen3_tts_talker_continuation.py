#!/usr/bin/env python3
"""Capture pinned Qwen3-TTS talker prefill and three continuation calls.

The reference supplies exact one-token inputs for teacher-forced arithmetic
isolation. Captured codec IDs and embeddings are never production inputs in
an autonomous speech claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

if __package__:
    from .capture_qwen3_tts_customvoice_oracle import (
        DEFAULT_LANGUAGE, DEFAULT_SPEAKER, DEFAULT_TEXT, MODEL_REVISION,
        REFERENCE_REVISION, verify_model_dir, verify_reference_code,
    )
else:
    from capture_qwen3_tts_customvoice_oracle import (
        DEFAULT_LANGUAGE, DEFAULT_SPEAKER, DEFAULT_TEXT, MODEL_REVISION,
        REFERENCE_REVISION, verify_model_dir, verify_reference_code,
    )


def capture(model_dir: Path, output_dir: Path,
            text: str = DEFAULT_TEXT) -> dict:
    model_hashes = verify_model_dir(model_dir)
    import numpy as np
    import torch
    import qwen_tts
    from qwen_tts import Qwen3TTSModel

    reference_hashes = verify_reference_code(Path(qwen_tts.__file__).resolve().parent)
    torch.manual_seed(20261022)
    torch.set_num_threads(max(1, int(os.environ.get("CKE_QWEN3_TTS_TORCH_THREADS", "4"))))
    model = Qwen3TTSModel.from_pretrained(
        str(model_dir), device_map="cpu", dtype=torch.float32,
        attn_implementation="eager")
    talker = model.model.talker
    arrays: dict[str, np.ndarray] = {}
    state = {"step": -1}
    handles = []

    def save(name, tensor):
        arrays[name] = tensor.detach().cpu().contiguous().numpy().copy()

    def before_model(_, args, kwargs):
        state["step"] += 1
        step = state["step"]
        for name, key in (("input", "inputs_embeds"),
                          ("position", "position_ids"),
                          ("cache_position", "cache_position"),
                          ("attention_mask", "attention_mask")):
            save(f"step{step}_{name}", kwargs[key])

    def after_model(_, args, kwargs, output):
        step = state["step"]
        save(f"step{step}_norm", output.last_hidden_state)
        for layer, cache in enumerate(output.past_key_values.layers):
            for kind, tensor in (("k", cache.keys), ("v", cache.values)):
                name = (f"step{step}_layer{layer}_{kind}_cache" if step == 0
                        else f"step{step}_layer{layer}_{kind}_new")
                save(name, tensor if step == 0 else tensor[:, :, -1:, :])

    handles.append(talker.model.register_forward_pre_hook(
        before_model, with_kwargs=True))
    handles.append(talker.model.register_forward_hook(
        after_model, with_kwargs=True))
    for layer, module in enumerate(talker.model.layers):
        def record(_, args, output, index=layer):
            save(f"step{state['step']}_layer{index}_output", output[0])
        handles.append(module.register_forward_hook(record))
    handles.append(talker.codec_head.register_forward_hook(
        lambda _, args, output: save(f"step{state['step']}_logits", output)))
    try:
        with torch.inference_mode():
            model.generate_custom_voice(
                text=text, speaker=DEFAULT_SPEAKER,
                language=DEFAULT_LANGUAGE, do_sample=False,
                subtalker_dosample=False, max_new_tokens=4,
                non_streaming_mode=True)
    finally:
        for handle in handles:
            handle.remove()
    if state["step"] != 3 or len(talker.model.layers) != 28:
        raise RuntimeError("expected prefill and exactly three one-token continuations")
    prompt_rows = arrays["step0_input"].shape[1]
    for step in range(4):
        rows = prompt_rows if step == 0 else 1
        if arrays[f"step{step}_input"].shape != (1, rows, 1024):
            raise RuntimeError(f"unexpected step {step} input geometry")
        expected_position = list(range(prompt_rows)) if step == 0 else [prompt_rows - 1 + step]
        if arrays[f"step{step}_cache_position"].tolist() != expected_position:
            raise RuntimeError(f"unexpected step {step} cache position")
        for layer in range(28):
            suffix = "cache" if step == 0 else "new"
            if arrays[f"step{step}_layer{layer}_k_{suffix}"].shape != (
                    1, 8, rows, 128):
                raise RuntimeError(f"unexpected step {step} layer {layer} cache")
    if not all(np.isfinite(value).all() for name, value in arrays.items()
               if value.dtype.kind == "f"):
        raise RuntimeError("nonfinite continuation reference")

    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = output_dir / "qwen3_tts_talker_continuation.npz"
    np.savez_compressed(fixture, **arrays)
    manifest = {
        "schema": 1,
        "scope": ("pinned 18-row prefill and three teacher-forced continuation steps; CKE execution NOT_TESTED"
                  if text == DEFAULT_TEXT else
                  "selected-text prefill and three teacher-forced continuation steps; CKE execution NOT_TESTED"),
        "model_revision": MODEL_REVISION,
        "model_files_sha256": model_hashes,
        "reference_revision": REFERENCE_REVISION,
        "reference_files_sha256": reference_hashes,
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "numpy": np.__version__, "machine": platform.machine(),
                    "device": "cpu", "dtype": "float32", "attention": "eager",
                    "threads": torch.get_num_threads()},
        "request": {"text": text, "speaker": DEFAULT_SPEAKER,
                    "language": DEFAULT_LANGUAGE, "seed": 20261022,
                    "do_sample": False, "subtalker_dosample": False,
                    "max_new_tokens": 4, "non_streaming_mode": True},
        "steps": 4,
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
        "npz_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    }
    (output_dir / "qwen3_tts_talker_continuation.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()
    print(json.dumps(capture(args.model_dir, args.output_dir, args.text), indent=2))


if __name__ == "__main__":
    main()
