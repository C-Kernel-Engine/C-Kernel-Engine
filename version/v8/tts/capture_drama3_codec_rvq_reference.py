"""Capture a development-only oracle for the Drama 3 codec RVQ prefix.

Use the pinned publisher source with its FishAudio dependencies installed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import wave

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

CODEC_SHA256 = "74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f"
FISH_SOURCE = Path("sglang_omni/models/fishaudio_s2_pro/fish_speech")
PUBLISHER_HASHES = {
    "models/dac/modded_dac.py": "9752b22178fe06a8cd5c2902510be6bcf507d7175e348def200a1acfa7698d71",
    "models/dac/rvq.py": "16cf24bf6a7481606cd78b2a4ecd6cfdeba9456dff38224f4833c6ff2735973b",
    "configs/modded_dac_vq.yaml": "e21d8c3bd72795c50941d8bb6104edfc87a6858d4c5c496d737ede47213223bc",
}


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def capture(codec_path, publisher, folder, *, include_transformer=False):
    if sha256(codec_path) != CODEC_SHA256:
        raise ValueError("codec SHA-256 does not match the pinned release")
    for name, expected in PUBLISHER_HASHES.items():
        if sha256(publisher / FISH_SOURCE / name) != expected:
            raise ValueError(f"publisher file SHA-256 mismatch: {name}")
    folder.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(publisher))
    config = OmegaConf.load(publisher / FISH_SOURCE / "configs/modded_dac_vq.yaml")
    torch.set_num_threads(4)
    codec = instantiate(config)
    weights = torch.load(codec_path, map_location="cpu", mmap=True, weights_only=True)
    load = codec.load_state_dict(weights, strict=False, assign=True)
    legacy_buffers = {
        f"{block}.{buffer}"
        for block in ("encoder.block.4.block.5", "quantizer.pre_module", "quantizer.post_module")
        for buffer in ("freqs_cis", "causal_mask")
    }
    if load.missing_keys or set(load.unexpected_keys) != legacy_buffers:
        raise ValueError(f"unexpected checkpoint load result: {load}")
    codec.eval()
    codec.fold_weight_norm()
    folded = {}
    books = [*codec.quantizer.semantic_quantizer.quantizers, *codec.quantizer.quantizer.quantizers]
    if len(books) != 10:
        raise ValueError("expected ten publisher codebooks")
    for i, book in enumerate(books):
        folded[f"book{i}.table"] = book.codebook.weight.detach().float().numpy()
        folded[f"book{i}.weight"] = book.out_proj.weight.detach().float().squeeze(-1).numpy()
        folded[f"book{i}.bias"] = book.out_proj.bias.detach().float().numpy()
    np.savez(folder / "rvq_weights.npz", **folded)

    post_module = codec.quantizer.post_module
    transformer_weights = {}
    if include_transformer:
        for i, layer in enumerate(post_module.layers):
            for name, value in layer.state_dict().items():
                if value.dtype != torch.float32:
                    raise ValueError(f"expected FP32 codec transformer parameter: {name}")
                name = f"codec_layer{i}.{name}"
                if name.endswith(".wqkv.weight"):
                    for port, tensor in zip(("q", "k", "v"), value.chunk(3, dim=0), strict=True):
                        transformer_weights[name.replace("wqkv", port)] = tensor.detach().numpy()
                else:
                    transformer_weights[name] = value.detach().numpy()
        transformer_weights["codec_transformer_norm.weight"] = post_module.norm.weight.detach().numpy()
        np.savez(folder / "transformer_weights.npz", **transformer_weights)

    frames = [1, 3, 8]
    results = []
    seen = {}

    def capture_transformer(_module, inputs, output):
        seen["quantizer_pre_transformer"] = inputs[0].detach().float().numpy()
        seen["quantizer_post_transformer"] = output.detach().float().numpy()

    def capture_layer(name):
        def hook(_module, _inputs, output):
            seen[name] = output.detach().float().numpy().copy()
        return hook

    generator = torch.Generator().manual_seed(472)
    with torch.inference_mode():
        for length in frames:
            semantic = torch.randint(4096, (1, 1, length), generator=generator)
            residual = torch.randint(1024, (1, 9, length), generator=generator)
            codes = torch.cat([semantic, residual], dim=1)
            pristine = codes.clone()
            seen.clear()
            handles = [post_module.register_forward_hook(capture_transformer)]
            if include_transformer:
                rotary = post_module.freqs_cis[:length].float().numpy()
                seen["rotary_cos"], seen["rotary_sin"] = rotary[..., 0], rotary[..., 1]
                handles.extend(layer.register_forward_hook(capture_layer(f"codec_layer{i}_out"))
                               for i, layer in enumerate(post_module.layers))
            try:
                audio = codec.from_indices(codes)
            finally:
                for handle in handles:
                    handle.remove()
            torch.testing.assert_close(pristine, codes, rtol=0, atol=0)
            if not torch.isfinite(audio).all() or audio.shape != (1, 1, length * codec.frame_length):
                raise ValueError(f"invalid decoded samples for {length} code frames: {audio.shape}")
            if not {"quantizer_pre_transformer", "quantizer_post_transformer"} <= seen.keys():
                raise ValueError("missing quantizer transformer capture")
            samples = audio.flatten().float().numpy()
            np.savez(folder / f"reference-codes-{length}.npz", codes=codes.numpy(),
                     samples=samples, **seen)
            with wave.open(str(folder / f"reference-codes-{length}.wav"), "wb") as wav:
                wav.setparams((1, 2, codec.sample_rate, 0, "NONE", "not compressed"))
                wav.writeframes((np.clip(samples, -1, 1) * 32767).round().astype("<i2").tobytes())
            results.append({"frames": length, "samples": samples.size,
                            "max_abs": float(np.abs(samples).max()),
                            "codes_sha256": hashlib.sha256(codes.numpy().tobytes()).hexdigest()})
    report = {"scope": "publisher PyTorch codec on synthetic code IDs",
              "model_revision": "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216",
              "implementation": "dfb6b43b360b04e53daa7c36aed0bc448789d3a4",
              "codec_sha256": CODEC_SHA256,
              "torch": torch.__version__, "sample_rate": codec.sample_rate,
              "omitted_legacy_buffers": load.unexpected_keys,
              "samples_per_frame": int(codec.frame_length), "cases": results}
    report["replay_sha256"] = {}
    artifacts = ["rvq_weights.npz", *(f"reference-codes-{length}.npz" for length in frames)]
    if include_transformer:
        artifacts.append("transformer_weights.npz")
        report["transformer"] = dict(layers=len(post_module.layers), heads=post_module.config.n_head,
                                     head_dim=post_module.config.head_dim,
                                     intermediate=post_module.config.intermediate_size,
                                     window=post_module.window_size)
    for name in artifacts:
        report["replay_sha256"][name] = sha256(folder / name)
    (folder / "reference-result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--publisher-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-transformer", action="store_true",
                        help="Capture all codec transformer parameters and layer outputs")
    args = parser.parse_args()
    print(json.dumps(capture(args.codec, args.publisher_root, args.out_dir,
                             include_transformer=args.include_transformer)))


if __name__ == "__main__":
    main()
