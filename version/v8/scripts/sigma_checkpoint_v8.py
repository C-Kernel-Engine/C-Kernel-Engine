"""Pinned Sigma-3B artifact verification and weight preparation."""
import hashlib
import json
import math

def pinned_weights(checkpoint, schema, torch, safetensors):
    configs = {
        "config.json": "a964291d6e14ee06c467e3d5edb69af178d27890faebaec2cc9acb260758cfaf",
        "model.safetensors.index.json": "7cf433a8e3923d7f87c65d5afb5660e6aa3d64247f4a475bb42817cbaabe1d7d",
        "model-00001-of-00002.safetensors": "69ea2c08d7adfeb625b7bce14752dea223edad314e0048b7d43a61742d613fbd",
        "model-00002-of-00002.safetensors": "c6dca19d68edd435c9344dbb1b8087c0c34a0980b975f58a878b27125dfd6115",
    }
    for artifact, digest in configs.items():
        with (checkpoint / artifact).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Sigma checkpoint hash mismatch: {artifact}")
    config = json.loads((checkpoint / "config.json").read_text())
    mapping = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    values = {}
    special = {"inv_freq", "time_frequencies", "normalized_embedding"}
    physical = set(schema) - special
    physical.update(("encoder.embed_tokens.weight", "replaid_embedding_matrix.matrix"))
    for shard in sorted({mapping[name] for name in physical}):
        if shard not in configs:
            raise ValueError(f"unexpected Sigma weight shard: {shard}")
        with safetensors.safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
            for name in sorted(physical):
                if mapping[name] == shard:
                    values[name] = handle.get_tensor(name)
    for name in physical - {"encoder.embed_tokens.weight", "replaid_embedding_matrix.matrix"}:
        shape, dtype = schema[name]
        value = values[name]
        if list(value.shape) != shape or value.dtype != torch.bfloat16:
            raise ValueError(f"unexpected Sigma weight shape or dtype: {name}")
        if dtype == "F32":
            values[name] = value.float()

    table = values.pop("replaid_embedding_matrix.matrix").float()
    values["normalized_embedding"] = table / (torch.linalg.vector_norm(table, dim=-1, keepdim=True) + 1e-8)
    values["time_frequencies"] = torch.linspace(-5., 5., 32, dtype=torch.float32).exp()

    dim = config["head_dim"]
    rope = config["rope_parameters"]
    base, original = rope["rope_theta"], rope["original_max_position_embeddings"]

    def correction(cycles):
        return dim * math.log(original / (cycles * 2 * math.pi)) / (2 * math.log(base))

    lo, hi = max(0, math.floor(correction(rope["beta_fast"]))), min(dim - 1, math.ceil(correction(rope["beta_slow"])))
    freq = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    fraction = ((torch.arange(dim // 2, dtype=torch.float32) - lo) / max(hi - lo, .001)).clamp(0, 1)
    values["inv_freq"] = (1.0 / freq) * (1 - fraction) + (1.0 / ((config["max_position_embeddings"] / original) * freq)) * fraction
    return values
