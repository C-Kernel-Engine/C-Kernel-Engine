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
import platform
import sys

from capture_qwen3_tts_customvoice_oracle import (
    DEFAULT_LANGUAGE, DEFAULT_SPEAKER, DEFAULT_TEXT, MODEL_REVISION,
    REFERENCE_REVISION, verify_model_dir, verify_reference_code,
)


class CaptureComplete(Exception):
    """Stop generation after the two required layer calls."""


def capture(model_dir: Path, output_dir: Path, *, text: str = DEFAULT_TEXT,
            stem: str = "qwen3_tts_first_block") -> dict:
    if not text or not text.strip() or "\x00" in text:
        raise ValueError("text must contain spoken characters and no NUL")
    if not stem or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in stem):
        raise ValueError("stem must contain only lowercase letters, digits and underscores")
    model_hashes = verify_model_dir(model_dir)
    import numpy as np
    import torch
    import transformers
    import qwen_tts
    import qwen_tts.core.models.modeling_qwen3_tts as modeling
    from qwen_tts import Qwen3TTSModel

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
    original_rotary = modeling.apply_multimodal_rotary_pos_emb

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
        current["cache_position"] = array(kwargs["cache_position"])
        mask = kwargs.get("attention_mask")
        if mask is not None:
            current["attention_mask"] = array(mask)
        cache = kwargs.get("past_key_values")
        if cache is not None and len(cache.layers) > 0:
            cache_layer = cache.layers[0]
            if cache_layer.keys is not None:
                current["kv_cache_key"] = array(cache_layer.keys)
                current["kv_cache_value"] = array(cache_layer.values)

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

    def record_input(name):
        def hook(_, args):
            assert current is not None
            current[name] = array(args[0])
        return hook

    def record_rotary(q, k, cos, sin, sections, interleaved=False, unsqueeze_dim=1):
        q_out, k_out = original_rotary(
            q, k, cos, sin, sections, interleaved, unsqueeze_dim)
        if current is not None:
            current["q_rotary"] = array(q_out)
            current["k_rotary"] = array(k_out)
        return q_out, k_out

    handles.append(layer.register_forward_pre_hook(begin, with_kwargs=True))
    for name, module in slots.items():
        handles.append(module.register_forward_hook(record(name)))
    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
        record_input("attention_context")))
    handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
        record_input("attention_residual")))
    handles.append(layer.mlp.down_proj.register_forward_pre_hook(
        record_input("swiglu_output")))
    handles.append(layer.register_forward_hook(finish))
    modeling.apply_multimodal_rotary_pos_emb = record_rotary
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
        modeling.apply_multimodal_rotary_pos_emb = original_rotary
        for handle in handles:
            handle.remove()
    if len(calls) != 2 or calls[0]["input"].shape[1] <= 1 or calls[1]["input"].shape[1] != 1:
        raise RuntimeError("expected prompt prefill followed by a one-token decode")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.npz"
    np.savez_compressed(path, **{
        f"call{index}_{key}": value
        for index, call in enumerate(calls) for key, value in call.items()
    })
    weights = {
        name: array(tensor)
        for name, tensor in layer.state_dict().items()
    }
    weights.update({
        f"text_projection.{name}": array(tensor)
        for name, tensor in model.model.talker.text_projection.state_dict().items()
    })
    weight_hashes = {
        name: hashlib.sha256(value.tobytes()).hexdigest()
        for name, value in weights.items()
    }
    report = {
        "schema": 1,
        "scope": "pinned PyTorch first talker block only; CKE execution NOT_TESTED",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "model_file_sha256": model_hashes,
        "reference_file_sha256": reference_hashes,
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "transformers": transformers.__version__, "numpy": np.__version__,
                    "device": "cpu", "dtype": "float32", "attention": "eager",
                    "threads": torch.get_num_threads(), "machine": platform.machine()},
        "request": {"text": text, "speaker": DEFAULT_SPEAKER,
                    "language": DEFAULT_LANGUAGE, "seed": 20261022},
        "calls": [{key: list(value.shape) for key, value in call.items()}
                  for call in calls],
        "weight_shapes": {name: list(value.shape) for name, value in weights.items()},
        "effective_weight_dtype": "float32",
        "weight_hash_semantics": "loaded FP32 tensors after pinned BF16 asset conversion",
        "weight_sha256": weight_hashes,
        "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--stem", default="qwen3_tts_first_block")
    args = parser.parse_args()
    print(json.dumps(capture(args.model_dir, args.output_dir,
                             text=args.text, stem=args.stem), indent=2))


if __name__ == "__main__":
    main()
