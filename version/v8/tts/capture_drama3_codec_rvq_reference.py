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


def capture(codec_path, publisher, folder):
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

    frames = [1, 3, 8]
    results = []
    seen = {}

    def capture_transformer(_module, inputs, output):
        seen["quantizer_pre_transformer"] = inputs[0].detach().float().numpy()
        seen["quantizer_post_transformer"] = output.detach().float().numpy()

    generator = torch.Generator().manual_seed(472)
    with torch.inference_mode():
        for length in frames:
            semantic = torch.randint(4096, (1, 1, length), generator=generator)
            residual = torch.randint(1024, (1, 9, length), generator=generator)
            codes = torch.cat([semantic, residual], dim=1)
            pristine = codes.clone()
            seen.clear()
            handle = codec.quantizer.post_module.register_forward_hook(capture_transformer)
            try:
                audio = codec.from_indices(codes)
            finally:
                handle.remove()
            torch.testing.assert_close(pristine, codes, rtol=0, atol=0)
            if not torch.isfinite(audio).all() or audio.shape != (1, 1, length * codec.frame_length):
                raise ValueError(f"invalid decoded samples for {length} code frames: {audio.shape}")
            if len(seen) != 2:
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
    for name in ("rvq_weights.npz", *(f"reference-codes-{length}.npz" for length in frames)):
        report["replay_sha256"][name] = sha256(folder / name)
    (folder / "reference-result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--publisher-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.codec, args.publisher_root, args.out_dir)))


if __name__ == "__main__":
    main()
