"""Capture two text-conditioned frames through the pinned Fish Audio runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


CHECKPOINT_REVISION = "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216"
PUBLISHER_REVISION = "1ff3c3ed2f80d3106a83909dd25b83129c3d816f"
CHECKPOINT_FILES = {
    "model-00001-of-00015.safetensors": "a1b7af15210d2eda2d75a2cf976295f80e345855b745b109aa3ce49d26a44497",
    "model-00002-of-00015.safetensors": "a95e113c6f32f8fcb23ecb618cd4384fad6358f3c0ac31cfbf352daf5370db94",
    "model-00003-of-00015.safetensors": "a34d0952d6369f5c73e6ae06138ef96bc6a3af9bb259c4e41425a33706cf5b68",
    "model-00004-of-00015.safetensors": "13ac368d87441a18752e424e485b711a115ea1084e3a18ab13276e73eedef8b4",
    "model-00005-of-00015.safetensors": "c432fa0dac97471f0d625bc67eb94fe084ca2e5ecd8865f457b751479b50e363",
    "model-00006-of-00015.safetensors": "661a31433747fa9d029632a61350ea9eb105bf7276d013ff0901b6a939fb2dcb",
    "model-00007-of-00015.safetensors": "4a7574b5c238f424b37aa945a736da1994a26bd38b86bf3ec4abf524c0969cc8",
    "model-00008-of-00015.safetensors": "7fac6d1e899939eb03c9cf28a33b0bfd9c7a599298662f26d9a59d64c96d007f",
    "model-00009-of-00015.safetensors": "fcf1313744ea5978e363b2bb49c3b66748d886d79f3ec102f4346a462385de4b",
    "model-00010-of-00015.safetensors": "6db3f7cd9d49806ae4f17b6f6d17a0c8dd9686aee57728da1b50ec0c6139a629",
    "model-00011-of-00015.safetensors": "a9027ef4db30bb1875bd2bc93b80bae785c5943807efe63bf09834b66e49fc16",
    "model-00012-of-00015.safetensors": "c16fea955e973010064ff7bc907a6fa67821956e0de23f74807c31fb48339f3a",
    "model-00013-of-00015.safetensors": "64861c8e1ee13a0cbb5ec9431f402cee33edcbcbc744ff63091fe3a997f0da19",
    "model-00014-of-00015.safetensors": "ba5319475a6258984143b69af8380f166df8179af90378f2f4e02694cad87baf",
    "model-00015-of-00015.safetensors": "1e2ddf01c1be73fb19f3022c28e90d24e730c1bc759e5de78083afb39aab3fb7",
    "config.json": "aea53f0f2b9a15023fc0f8eb0f42360663671c99814037bada6867d508010111",
    "model.safetensors.index.json": "a8e9aca4e9fddca51f03fad63f52c82c470e98e28b6b0b1b1562dbe6456b50bb",
    "tokenizer.json": "0e9b9db76b2f0389b04b6ef057a2ac3b89100b7e96f2dd7034d19580a09654d6"
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_publisher(checkout):
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != PUBLISHER_REVISION:
        raise ValueError(f"Fish Speech checkout must be at {PUBLISHER_REVISION}")
    if subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "-uno"], text=True):
        raise ValueError("Fish Speech checkout has tracked modifications")
    sys.path.insert(0, str(checkout.resolve()))


def load_model(checkpoint, device, torch):
    from tokenizers import Tokenizer

    from fish_speech.models.text2semantic.fish_omni.backbone import (
        CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY, FishOmniBackbone, load_backbone_weights,
    )
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.model import (
        AUDIO_DECODER_PREFIXES, FishOmniModel, iter_checkpoint_tensors,
    )
    from fish_speech.models.text2semantic.fish_omni.tower import (
        FishOmniAudioDecoder, load_audio_decoder_weights,
    )

    if device != "cpu":
        return FishOmniModel.from_pretrained(
            checkpoint, devices=[device], tower_device=device, dtype=torch.bfloat16,
        )
    # CPU diagnostic: use the publisher's eager expert math with FP8 resident
    # weights and BF16 compute. This bypasses the CUDA-only frontend guard.
    config = FishOmniConfig.from_dict(json.loads((checkpoint / "config.json").read_text()))
    if not config.fp8_experts:
        raise ValueError("the CPU reference requires the pinned block-FP8 checkpoint")
    with torch.device("meta"):
        backbone = FishOmniBackbone(config.text, fp8_experts=True, dtype=torch.bfloat16)
        tower = FishOmniAudioDecoder(config, dtype=torch.bfloat16)
    backbone.materialize(["cpu"])
    load_backbone_weights(backbone, iter_checkpoint_tensors(checkpoint, skip_prefixes=AUDIO_DECODER_PREFIXES))
    tower.to_empty(device="cpu")
    load_audio_decoder_weights(tower, iter_checkpoint_tensors(
        checkpoint, skip_prefixes=(CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY),
    ))
    return FishOmniModel(config, backbone, tower, Tokenizer.from_file(str(checkpoint / "tokenizer.json"))).eval()


def capture(model, text, seed, torch):
    from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt, generate_drama3

    torch.manual_seed(seed)
    prompt = build_drama3_prompt(text, model.tokenizer, model.config)
    tower = model.audio_decoder
    arrays = {"prompt_rows": prompt.numpy()}
    current = None
    step_count = 0

    def bf16(tensor):
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"expected BF16 capture, found {tensor.dtype}")
        return tensor.detach().to("cpu").contiguous().view(torch.uint16).numpy().copy()

    def save_shared(_layer, args):
        hidden, code, _, position, unseen = args
        if current is not None:
            arrays[current + "shared_input"] = bf16(hidden)
            arrays[current + "semantic_offset"] = code.cpu().numpy().copy()
            arrays[current + "enrichment"] = bf16(unseen)
            arrays[current + "position"] = position.cpu().numpy().copy()

    def head_hook(index):
        def save_head(_module, _args, output):
            if current is not None:
                arrays[current + f"head{index}_scores"] = bf16(output)
        return save_head

    original = tower.decode_step
    original_prefill = tower.prefill
    prefill_hidden, prefill_positions, prefill_labels = [], [], []

    def prefill(hidden, positions, frames, next_frames):
        prefill_hidden.append(bf16(hidden))
        prefill_positions.append(positions.cpu().numpy().copy())
        prefill_labels.append(next_frames.cpu().numpy().copy())
        return original_prefill(hidden, positions, frames, next_frames)

    def decode(logits, hidden, positions, frames, history, **settings):
        nonlocal current, step_count
        current = f"step{step_count}_"
        arrays[current + "backbone_hidden"] = bf16(hidden)
        arrays[current + "input_row"] = frames.cpu().numpy().copy()
        arrays[current + "main_history"] = history.cpu().numpy().copy()
        config = model.config
        candidates = torch.tensor(sorted([config.eos_token_id, *range(
            config.semantic_start_token_id, config.semantic_end_token_id + 1,
        )]), device=logits.device)
        arrays[current + "main_candidate_ids"] = candidates.cpu().numpy()
        arrays[current + "main_scores"] = bf16(logits[:, candidates])
        result = original(logits, hidden, positions, frames, history, **settings)
        arrays[current + "sampled_row"] = result.cpu().numpy().copy()
        step_count += 1
        current = None
        return result

    hooks = [tower.shared_layers[0].register_forward_pre_hook(save_shared)]
    hooks.extend(head.register_forward_hook(head_hook(index)) for index, head in enumerate(tower.codebook_heads))
    tower.decode_step = decode
    tower.prefill = prefill
    try:
        rows, finish = generate_drama3(model, prompt, max_new_tokens=2, top_k=1, temperature=0.7, top_p=1.0)
    finally:
        tower.decode_step = original
        tower.prefill = original_prefill
        for hook in hooks:
            hook.remove()

    if len(rows) != 2 or step_count != 2:
        raise RuntimeError(f"reference stopped after {len(rows)} audio frames ({finish})")
    for layer, module in enumerate((*tower.shared_layers, *tower.residual_layers)):
        attention = module.transformer_layer.attention
        for kind, cache in (("key", attention.k_cache), ("value", attention.v_cache)):
            # Position p consumes the cache prefix [0,p) and writes one row.
            arrays[f"tower_{layer}_{kind}"] = bf16(cache[0, :, :len(prompt) + 1].transpose(0, 1))
    arrays["prefill_hidden"] = np.concatenate(prefill_hidden)
    arrays["prefill_positions"] = np.concatenate(prefill_positions)
    arrays["prefill_next_rows"] = np.concatenate(prefill_labels)
    if not np.array_equal(arrays["prefill_next_rows"], arrays["prompt_rows"][1:]):
        raise RuntimeError("publisher tower labels do not match the following prompt rows")
    arrays["frame_rows"] = rows.numpy()
    if not np.array_equal(arrays["step1_input_row"], arrays["frame_rows"][:1]):
        raise RuntimeError("publisher's second step did not consume the first sampled row")
    expected_positions = [len(prompt) - 1, len(prompt)]
    if [int(arrays[f"step{i}_position"][0]) for i in range(2)] != expected_positions:
        raise RuntimeError("unexpected tower position sequence")
    return arrays, finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fish-speech", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text", default="Hello, CKE.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    import torch

    load_publisher(args.fish_speech)
    for name, expected in CHECKPOINT_FILES.items():
        if digest(args.checkpoint / name) != expected:
            raise ValueError(f"checkpoint hash mismatch: {name}")
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    torch.set_num_threads(args.threads)
    model = load_model(args.checkpoint, args.device, torch)
    with torch.inference_mode():
        arrays, finish = capture(model, args.text, args.seed, torch)
    args.output.mkdir(parents=True)
    path = args.output / "frames.npz"
    np.savez_compressed(path, **arrays)
    report = {
        "schema": "cke.drama3.reference.two_frames.v1",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "publisher_revision": PUBLISHER_REVISION,
        "text": args.text, "seed": args.seed, "device": args.device, "torch": torch.__version__,
        "sampling": {"top_k": 1, "top_p": 1.0, "temperature": 0.7, "repetition_retry": "publisher"},
        "finish_reason": finish, "prompt_rows": len(arrays["prompt_rows"]),
        "frames": arrays["frame_rows"].tolist(), "frames_sha256": digest(path),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
