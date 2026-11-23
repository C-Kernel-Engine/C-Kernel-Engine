"""Build a PyTorch replay fixture for the pinned Sigma-3B attention ingress."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REVISION = "7eff02e0b72dace46d9084430330c192bf5a4386"
PREFIX = "encoder.layers.0."
WEIGHTS = {
    "norm_weight": ("input_layernorm.weight", "6fd06cc7bc07acecda22488f0ffd9f487934a75ca422f97ba4f73f36ea145e98"),
    "q_weight": ("self_attn.q_proj.weight", "5e0e16b1f3338f5843c6d395037287e6887670588a1c6ddfb92344db3e147293"),
    "k_weight": ("self_attn.k_proj.weight", "55ed5d3660761fa04e33f4ade22536b52573fe4c6331187f8c584d7eaf1c88d9"),
    "v_weight": ("self_attn.v_proj.weight", "fd7568e38756513b44d349aa2eef10c122b85869c6a74884559431b2b516fba6"),
}


def bits(value):
    return value.contiguous().view(torch.uint16).numpy()


def capture(checkpoint, output):
    torch.set_num_threads(2)
    indexed = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for name, (field, pinned_hash) in WEIGHTS.items():
        key = PREFIX + field
        shard = indexed[key]
        if Path(shard).name != shard:
            raise ValueError(f"unexpected shard path for {key}")
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as source:
            value = source.get_tensor(key)
        if value.dtype != torch.bfloat16 or hashlib.sha256(bits(value).tobytes()).hexdigest() != pinned_hash:
            raise ValueError(f"pinned BF16 weight mismatch for {key}")
        weights[name] = value

    rng = torch.Generator().manual_seed(3112)
    torch.randn((1, 7, 3072), generator=rng)  # Match the prefix-plus-block reference fixture.
    x = torch.randn((32, 3072), generator=rng).bfloat16()
    fp32 = x.float()
    normalized = weights["norm_weight"] * (
        fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + 1e-5)
    ).bfloat16()
    contents = {"unnormalized": bits(x), "hidden": bits(normalized)}
    contents.update({name: bits(value) for name, value in weights.items()})
    for port in ("q", "k", "v"):
        contents[port] = bits(torch.nn.functional.linear(normalized, weights[f"{port}_weight"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **contents)
    with output.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    output.with_suffix(".json").write_text(json.dumps(
        {"checkpoint": REVISION, "fixture_sha256": digest, "torch": torch.__version__},
        indent=2,
    ) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    capture(args.checkpoint, args.output)
