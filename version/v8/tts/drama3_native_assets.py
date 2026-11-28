"""Prepare pinned Drama 3 weights for the generated speech acceptance harness."""

import hashlib
import json

import numpy as np

from version.v8.tts.build_drama3_codec_waveform_circuit import build_circuit
from version.v8.tts.capture_drama3_two_frame_reference import CHECKPOINT_FILES

CODEC_SHA256 = "74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f"
TOWER_SETTINGS = dict(shared_layers=4, residual_layers=9, residual_vocab=4096, vocabulary=4096,
                      width=2048, heads=16, kv_heads=2, dim=256, rotary=64, inner=6144, blas=True)


def verify_file(path, expected):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"pinned artifact hash mismatch: {path}")


def load_speech_weights(checkpoint):
    import torch
    from safetensors import safe_open

    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        verify_file(checkpoint / name, CHECKPOINT_FILES[name])
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    feedback_name = "audio_decoder.codebook_embeddings.weight"
    prefixes = ("audio_decoder_mtp_shared.", "audio_decoder_mtp.")
    shards = {}
    for name, shard in index.items():
        if name == feedback_name or name.startswith(prefixes):
            shards.setdefault(shard, []).append(name)

    weights, tables = {}, {}
    feedback = None
    for shard, names in sorted(shards.items()):
        verify_file(checkpoint / shard, CHECKPOINT_FILES[shard])
        with safe_open(checkpoint / shard, framework="pt") as archive:
            for name in names:
                value = archive.get_tensor(name)
                if value.dtype != torch.bfloat16:
                    raise ValueError(f"expected BF16 speech weight: {name}")
                bits = value.view(torch.uint16).numpy()
                if name == feedback_name:
                    feedback = bits
                elif name.endswith((".semantic_embedding.weight", ".codebook_embedding.weight")):
                    kind = "shared" if name.startswith(prefixes[0]) else "residual"
                    tables[kind + name.split(".")[2]] = bits
                elif name.endswith(".wqkv.weight"):
                    for part, matrix in zip(("q", "attention_gate", "k", "v"),
                                            np.split(bits, (4096, 8192, 8704)), strict=True):
                        weights[name.removesuffix("wqkv.weight") + part + ".weight"] = matrix
                else:
                    weights[name] = bits
    if feedback is None or feedback.shape != (40960, 2048):
        raise ValueError("missing pinned feedback embedding table")
    # Match the publisher’s FP32 inverse-frequency construction.
    weights["inv_freq"] = (10000000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64)).numpy()
    return weights, tables, feedback


def load_codec(checkpoint):
    import torch

    path = checkpoint / "codec.pth"
    verify_file(path, CODEC_SHA256)
    raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    graph, _ = build_circuit(1)
    names = {name for op in graph["block_types"]["rvq"]["body"]["ops"]
             for name in op.get("weight_refs", {}).values()}
    weights = {}

    def get(name):
        if name in raw:
            return raw[name]
        stem = name.removesuffix(".weight")
        if stem + ".weight_v" in raw:
            return torch._weight_norm(raw[stem + ".weight_v"], raw[stem + ".weight_g"], 0)
        params = stem + ".parametrizations.weight.original"
        return torch._weight_norm(raw[params + "1"], raw[params + "0"], 0)

    for name in sorted(names):
        if name.startswith("book"):
            book, part = name.split(".", 1)
            index = int(book.removeprefix("book"))
            kind = "quantizer" if index else "semantic_quantizer"
            source = f"quantizer.{kind}.quantizers.{max(0, index - 1)}."
            value = get(source + {"table": "codebook.weight", "weight": "out_proj.weight",
                                  "bias": "out_proj.bias"}[part])
            if part == "weight":
                value = value.squeeze(-1)
        elif name.startswith("codec_layer"):
            layer, part = name.split(".", 1)
            source = f"quantizer.post_module.layers.{layer.removeprefix('codec_layer')}."
            if part in ("attention.q.weight", "attention.k.weight", "attention.v.weight"):
                value = raw[source + "attention.wqkv.weight"].chunk(3, 0)["qkv".index(part[10])]
            else:
                value = raw[source + part]
        elif name == "codec_transformer_norm.weight":
            value = raw["quantizer.post_module.norm.weight"]
        elif name.startswith("upsample."):
            value = get("quantizer." + name)
            if name.endswith(".dwconv.conv.weight"):
                value = value.squeeze(1)
        else:
            if not name.startswith("decoder."):
                raise ValueError(f"unexpected codec weight {name}")
            value = get(name)
            if name.endswith(".alpha"):
                value = value.flatten()
                if (not torch.isfinite(value).all() or torch.any(value == 0) or
                        not torch.equal(value + 1e-9, value)):
                    raise ValueError(f"unsupported Snake denominator: {name}")
        if value.dtype != torch.float32:
            raise ValueError(f"expected FP32 codec weight: {name}")
        weights[name] = value.detach().numpy()
    return weights, raw["quantizer.post_module.freqs_cis"].float().numpy()
