"""Capture pinned Bison BF16 decision logits with explicit independent branch masks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from version.v8.decision import groundhog_checkpoint

CHECKPOINT_FILES = {
    "config.json": "bfabc4abdd51bc421188332d28eb1de468ae21f1c1ccf35f07de9506d62b67e2",
    "model.safetensors.index.json": "06e580553feed4e5d7285eed764b19f2b38c1cd2be28cb041dab036aa0db5ae2",
    "model-00001-of-00002.safetensors": "55b01d0ec3e7e36547a32544d013ece6ff0cf4447be0f2444ad46452422fc5e7",
    "model-00002-of-00002.safetensors": "1d9f8e96e15b50c160a3f5dc69d40daf579ced5c62b628a6c7328b808f40bb62",
}
REVISION = "bf8350b2c59eb6c36e95d02a2af870aec18724a7"
PREFIX = "model.language_model."


def checkpoint_hashes(revision):
    if revision == REVISION:
        return CHECKPOINT_FILES
    if revision == groundhog_checkpoint.REVISION:
        return groundhog_checkpoint.CHECKPOINT_FILES
    raise ValueError(f"unknown pinned decision checkpoint {revision!r}")


def capture(checkpoint, request, threads, *, readout_inputs=None):
    revision = request.get("checkpoint", REVISION)
    import torch
    import transformers
    from safetensors import safe_open
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4RMSNorm, Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding,
    )

    torch.set_num_threads(threads)
    text_config = json.loads((checkpoint / "config.json").read_text())["text_config"]
    overrides = text_config.pop("per_layer_config")
    full_layers = {
        int(index): (row["head_dim"], row["num_key_value_heads"])
        for index, row in overrides.items()
    }
    if set(full_layers) != {i for i, kind in enumerate(text_config["layer_types"]) if kind == "full_attention"}:
        raise ValueError("expected overrides on every full attention layer")
    if len(set(full_layers.values())) != 1:
        raise ValueError("stock Gemma4 reference cannot express nonuniform full-attention geometry")
    head_dim, kv_heads = next(iter(full_layers.values()))
    config = Gemma4TextConfig.from_dict({
        **text_config, "global_head_dim": head_dim, "num_global_key_value_heads": kv_heads,
    })
    config._attn_implementation = "eager"
    if config.hidden_size_per_layer_input or config.num_kv_shared_layers:
        raise ValueError("pinned decision oracle requires independent layer state and no PLE")
    if config.enable_moe_block and revision != groundhog_checkpoint.REVISION:
        raise ValueError("unexpected routed expert checkpoint")
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]

    def weights(names):
        by_shard = defaultdict(list)
        for name in names:
            by_shard[weight_map[name]].append(name)
        values = {}
        for shard, keys in by_shard.items():
            with safe_open(checkpoint / shard, framework="pt", device="cpu") as source:
                for key in keys:
                    tensor = source.get_tensor(key)
                    if tensor.dtype != torch.bfloat16:
                        raise ValueError(f"unexpected publisher weight dtype: {key}")
                    values[key] = tensor
        return values

    capture_sha256 = None
    with torch.inference_mode():
        embed_key = PREFIX + "embed_tokens.weight"
        embed = weights([embed_key])[embed_key]
        rotary = Gemma4TextRotaryEmbedding(config)
        prefix = request["prefix"]
        branches = []
        for index, question in enumerate([*request["questions"], request["questions"][-1]]):
            tokens = prefix + question["tokens"]
            row = torch.arange(len(tokens))[:, None]
            col = torch.arange(len(tokens))[None, :]
            allowed = (row >= len(prefix)) | (col < len(prefix))
            if index == len(request["questions"]):
                # Deliberate error: the shared prefix sees the question.
                allowed = torch.ones_like(allowed)
            sliding = (col <= row) & (col > row - config.sliding_window)
            masks = {
                name: torch.zeros(mask.shape, dtype=torch.bfloat16).masked_fill(
                    ~mask, torch.finfo(torch.bfloat16).min
                )[None, None]
                for name, mask in [("full_attention", allowed), ("sliding_attention", sliding)]
            }
            hidden = embed[tokens].unsqueeze(0) * torch.tensor(config.hidden_size**0.5, dtype=torch.bfloat16)
            position_ids = row.T
            rope = {kind: rotary(hidden, position_ids, kind) for kind in masks}
            branches.append((hidden, masks, position_ids, rope))
        del embed

        for index, kind in enumerate(config.layer_types):
            with torch.device("meta"):
                layer = Gemma4TextDecoderLayer(config, index)
            prefix_key = f"{PREFIX}layers.{index}."
            values = weights(prefix_key + key for key in layer.state_dict())
            layer.load_state_dict({key.removeprefix(prefix_key): value for key, value in values.items()},
                                  strict=True, assign=True)
            del values
            layer.eval()
            for i, (hidden, masks, positions, rope) in enumerate(branches):
                hidden = layer(hidden, shared_kv_states={}, position_embeddings=rope[kind],
                               attention_mask=masks[kind], position_ids=positions)
                branches[i] = (hidden, masks, positions, rope)
            del layer

        norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.load_state_dict({"weight": weights([PREFIX + "norm.weight"])[PREFIX + "norm.weight"]}, assign=True)
        head = weights(["lm_head.weight"])["lm_head.weight"]
        outputs = []
        packed_hidden = []
        for index, ((hidden, _, _, _), question) in enumerate(zip(
            branches, [*request["questions"], request["questions"][-1]], strict=True,
        )):
            # The stock publisher Gemma module applies the logit softcap to BF16 logits.
            slots = [token for _, token in question["options"]]
            if readout_inputs is not None and index < len(request["questions"]):
                start = len(prefix) if index else 0
                packed_hidden.append(norm(hidden[0, start:]).clone())
            raw_logits = torch.nn.functional.linear(norm(hidden[0, -2:]), head[slots])
            logits = raw_logits
            softcap = config.final_logit_softcapping
            if softcap is not None:
                logits = softcap * torch.tanh(logits / softcap)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"nonfinite scores for {question['name']}")
            outputs.append({
                "name": question["name"],
                "option_scores": dict(zip([label for label, _ in question["options"]], logits[-1].tolist(), strict=True)),
                "raw_option_scores": dict(zip([label for label, _ in question["options"]], raw_logits[-1].tolist(), strict=True)),
                "previous_row_scores": logits[-2].tolist(),
            })
        if readout_inputs is not None:
            import numpy as np

            with readout_inputs.open("xb") as stream:
                np.save(stream, torch.cat(packed_hidden).view(torch.uint16).numpy(), allow_pickle=False)
            with readout_inputs.open("rb") as stream:
                capture_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    correct = outputs[:-1]
    return {
        "checkpoint": revision, "renderer_revision": request["renderer_revision"],
        "torch": torch.__version__, "transformers": transformers.__version__,
        "scope": "PyTorch BF16 with explicit block-bidirectional and causal sliding masks",
        "readout_inputs_sha256": capture_sha256,
        "questions": correct,
        "wrong_mask_changes_scores": outputs[-1]["option_scores"] != correct[-1]["option_scores"],
        "wrong_row_changes_scores": all(
            list(scores["option_scores"].values()) != scores["previous_row_scores"] for scores in correct
        ),
    }


def main():
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument("--checkpoint", required=True, type=Path)
    argp.add_argument("--request", required=True, type=Path)
    argp.add_argument("--out", required=True, type=Path)
    argp.add_argument("--readout-inputs", type=Path,
                      help="optional new NPY file containing the oracle's packed BF16 final hidden rows")
    argp.add_argument("--threads", default=4, type=int)
    args = argp.parse_args()
    if args.threads < 1 or args.out.exists() or (args.readout_inputs is not None and (
        args.readout_inputs.exists() or args.readout_inputs.resolve() == args.out.resolve()
    )):
        argp.error("threads must be positive and output must be new")
    request = json.loads(args.request.read_text())
    if len(request.get("questions", ())) != 2:
        argp.error("expected a pinned two-question decision fixture")
    for name, expected in checkpoint_hashes(request.get("checkpoint")).items():
        with (args.checkpoint / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"publisher digest mismatch: {name}")
    report = capture(args.checkpoint, request, args.threads, readout_inputs=args.readout_inputs)
    report["request_sha256"] = hashlib.sha256(args.request.read_bytes()).hexdigest()
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return 0 if report["wrong_mask_changes_scores"] and report["wrong_row_changes_scores"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
