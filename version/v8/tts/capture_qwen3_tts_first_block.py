#!/usr/bin/env python3
"""Capture the first two talker-layer calls from the pinned CustomVoice oracle.

This is reference evidence, not a CKE execution path. The first call is the
prompt prefill and the next call is one continuation token. The capture stops
after those calls, so it neither claims synthesis nor writes a waveform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from capture_qwen3_tts_customvoice_oracle import (
    DEFAULT_LANGUAGE, DEFAULT_SPEAKER, DEFAULT_TEXT, MODEL_REVISION,
    REFERENCE_REVISION, verify_model_dir, verify_reference_code,
)


class CaptureComplete(Exception):
    """Stop generation after the two required layer calls."""


def capture(model_dir: Path, output_dir: Path,
            text: str = DEFAULT_TEXT) -> dict:
    model_hashes = verify_model_dir(model_dir)
    import numpy as np
    import torch
    import qwen_tts
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models import modeling_qwen3_tts as modeling

    reference_hashes = verify_reference_code(Path(qwen_tts.__file__).resolve().parent)
    torch.manual_seed(20261022)
    torch.set_num_threads(max(1, int(os.environ.get("CKE_QWEN3_TTS_TORCH_THREADS", "4"))))
    model = Qwen3TTSModel.from_pretrained(
        str(model_dir), device_map="cpu", dtype=torch.float32,
        attn_implementation="eager")
    layer = model.model.talker.model.layers[0]
    slots = {
        "input_norm": layer.input_layernorm,
        "q_proj": layer.self_attn.q_proj,
        "q_norm": layer.self_attn.q_norm,
        "k_proj": layer.self_attn.k_proj,
        "k_norm": layer.self_attn.k_norm,
        "v_proj": layer.self_attn.v_proj,
        "attention": layer.self_attn,
        "o_proj": layer.self_attn.o_proj,
        "post_attention_norm": layer.post_attention_layernorm,
        "gate_proj": layer.mlp.gate_proj,
        "up_proj": layer.mlp.up_proj,
        "down_proj": layer.mlp.down_proj,
    }
    calls: list[dict[str, np.ndarray]] = []
    current: dict[str, np.ndarray] | None = None
    handles = []

    def array(value):
        return value.detach().cpu().contiguous().numpy().copy()

    def begin(_, args, kwargs):
        nonlocal current
        if len(calls) >= 2:
            raise CaptureComplete()
        current = {"input": array(args[0])}
        position = kwargs.get("position_embeddings")
        if position is None:
            raise RuntimeError("reference omitted position embeddings")
        current["cos"] = array(position[0])
        current["sin"] = array(position[1])
        current["position_ids"] = array(kwargs["position_ids"])

    def finish(_, __, output):
        nonlocal current
        assert current is not None
        current["output"] = array(output[0])
        calls.append(current)
        current = None

    def record(name):
        def hook(_, __, output):
            assert current is not None
            value = output[0] if isinstance(output, tuple) else output
            current[name] = array(value)
        return hook


    handles.append(layer.register_forward_pre_hook(begin, with_kwargs=True))
    for name, module in slots.items():
        handles.append(module.register_forward_hook(record(name)))
    handles.append(layer.register_forward_hook(finish))
    original_rope = modeling.apply_multimodal_rotary_pos_emb

    def record_rope(*args, **kwargs):
        rotated_q, rotated_k = original_rope(*args, **kwargs)
        if current is not None:
            current["q_after_rope"] = array(rotated_q)
            current["k_after_rope"] = array(rotated_k)
        return rotated_q, rotated_k

    modeling.apply_multimodal_rotary_pos_emb = record_rope
    try:
        with torch.inference_mode():
            model.generate_custom_voice(
                text=text, speaker=DEFAULT_SPEAKER,
                language=DEFAULT_LANGUAGE, do_sample=False,
                subtalker_dosample=False, max_new_tokens=8,
                non_streaming_mode=True)
    except CaptureComplete:
        pass
    finally:
        modeling.apply_multimodal_rotary_pos_emb = original_rope
        for handle in handles:
            handle.remove()
    if len(calls) != 2 or calls[0]["input"].shape[1] <= 1 or calls[1]["input"].shape[1] != 1:
        raise RuntimeError("expected prompt prefill followed by a one-token decode")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "qwen3_tts_first_block.npz"
    np.savez_compressed(path, **{
        f"call{index}_{key}": value
        for index, call in enumerate(calls) for key, value in call.items()
    })
    report = {
        "schema": 1,
        "scope": "pinned PyTorch first talker block only; CKE execution NOT_TESTED",
        "model_revision": MODEL_REVISION,
        "model_files_sha256": model_hashes,
        "reference_revision": REFERENCE_REVISION,
        "reference_files_sha256": reference_hashes,
        "request": {"text": text, "speaker": DEFAULT_SPEAKER,
                    "language": DEFAULT_LANGUAGE, "seed": 20261022},
        "calls": [{key: list(value.shape) for key, value in call.items()}
                  for call in calls],
        "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output_dir / "qwen3_tts_first_block.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()
    print(json.dumps(capture(args.model_dir, args.output_dir, args.text), indent=2))


if __name__ == "__main__":
    main()
