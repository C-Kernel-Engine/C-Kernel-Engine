"""Check repeated generated denoiser layers and full vocabulary readout."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from version.v8.scripts.build_sigma_bf16_denoiser_stack_v8 import build_circuit
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


def bits(x):
    import torch
    return x.contiguous().view(torch.uint16).numpy()


def compare(folder, torch, graph, config, schema, weights, *, rows, count, heads, kv_heads, dim, blas=False,
            check_cached=None):
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    run, arena, view, _ = compile_checked_weight_graph(
        folder, graph, {name: weight.numpy() if name == "inv_freq" else bits(weight)
                        for name, weight in weights.items()}, schema, source_config=config,
        target_isa="avx2" if blas else "portable",
    )
    source = torch.randn((rows, config["embed_dim"]), generator=torch.Generator().manual_seed(3112)).bfloat16()
    view("input", np.uint16)[:] = bits(source)
    positions = torch.arange(rows, dtype=torch.int32)
    view("positions", np.int32)[:] = positions
    view("ranges", np.int32)[:] = np.array([[0, 1, 1, 1]] + [[0, rows, rows, rows]] * (rows - 1), np.int32)
    mask = torch.ones((rows, rows), dtype=torch.bool)
    mask[0, 1:] = False
    assert run(arena, len(arena)) == 0
    if check_cached is not None:
        stack_geometry = check_cached
        cached_graph, cached_config, cached_schema = build_circuit(rows - 1, prefix_rows=1, **stack_geometry)
        cached_path = folder / "with_prefix"
        cached_path.mkdir()
        cached, cached_arena, cached_view, _ = compile_checked_weight_graph(
            cached_path, cached_graph, {k: v.numpy() if k == "inv_freq" else bits(v) for k, v in weights.items()},
            cached_schema, source_config=cached_config,
        )
        cache = {}
        for i in range(count):
            for name, tensor in (("key", "k_rotary"), ("value", "v")):
                cache[f"layer{i}_prefix_{name}"] = view(f"layer{i}_{tensor}", np.uint16)[:1].copy()
        for name, value in cache.items():
            cached_view(name, np.uint16)[:] = value
        cached_view("input", np.uint16)[:] = bits(source[1:])
        cached_view("positions", np.int32)[:] = positions[1:]
        cached_view("ranges", np.int32)[:] = view("ranges", np.int32)[1:]
        assert cached(cached_arena, len(cached_arena)) == 0
        np.testing.assert_array_equal(cached_view("diffusion_scores", np.uint16), view("diffusion_scores", np.uint16)[1:])
        for name, value in cache.items():
            np.testing.assert_array_equal(cached_view(name, np.uint16), value)
    angle = positions[:, None].float() * weights["inv_freq"]
    cos, sin = angle.cos().repeat(1, 2).bfloat16(), angle.sin().repeat(1, 2).bfloat16()

    def rotate(x, count):
        x = x.reshape(rows, count, dim).transpose(0, 1)
        a, b = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-b, a), dim=-1) * sin

    def norm(x, name):
        fp32 = x.float()
        return (fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + 1e-5)).bfloat16() * weights[name]

    def compare_at(name, reference):
        actual = view(name, np.uint16)
        f32 = (actual.astype(np.uint32) << 16).view(np.float32)
        diff = f32.astype(np.float64) - reference.float().numpy()
        return {"exact": int(np.sum(actual == bits(reference))), "count": actual.size,
                "max_abs": float(np.max(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff ** 2)))}

    x = source
    metrics = {}
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for index in range(count):
            key = f"encoder.layers.{index}."
            n = norm(x, key + "input_layernorm.weight")
            q, k, v = (F.linear(n, weights[key + f"self_attn.{port}_proj.weight"]) for port in ("q", "k", "v"))
            v = v.reshape(rows, kv_heads, dim).transpose(0, 1)
            context = F.scaled_dot_product_attention(
                rotate(q, heads), rotate(k, kv_heads).repeat_interleave(heads // kv_heads, dim=0),
                v.repeat_interleave(heads // kv_heads, dim=0), mask,
            ).transpose(0, 1).reshape(rows, heads * dim)
            x = x + F.linear(context, weights[key + "self_attn.o_proj.weight"])
            n = norm(x, key + "post_attention_layernorm.weight")
            gate = F.linear(n, weights[key + "mlp.gate_proj.weight"])
            up = F.linear(n, weights[key + "mlp.up_proj.weight"])
            x = x + F.linear(F.silu(gate) * up, weights[key + "mlp.down_proj.weight"])
            layer = f"layer{index}_output"
            metrics[layer] = compare_at(layer, x)
        x = norm(x, "encoder.norm.weight")
        logits = F.linear(x, weights["diffusion_head.weight"])
        metrics["denoised_hidden"] = compare_at("denoised_hidden", x)
        metrics["diffusion_scores"] = compare_at("diffusion_scores", logits)
        scores = view("diffusion_scores", np.uint16)
        native = (scores.astype(np.uint32) << 16).view(np.float32)
        metrics["greedy_rows_equal"] = int(np.sum(native.argmax(-1) == logits.float().numpy().argmax(-1)))
    return metrics


def test_generated_sigma_denoiser_stack(tmp_path):
    torch = pytest.importorskip("torch")
    rows, count, heads, kv_heads, dim = 4, 2, 4, 2, 6
    geometry = dict(layers=count, vocabulary=17, width=13, heads=heads, kv_heads=kv_heads, dim=dim, inner=19)
    graph, config, schema = build_circuit(rows, **geometry)
    rng = torch.Generator().manual_seed(3194)
    weights = {name: (torch.randn(shape, generator=rng) * 0.25).bfloat16()
               for name, (shape, dtype) in schema.items() if dtype == "BF16"}
    weights["inv_freq"] = torch.tensor([1.0, 0.1, 0.01])
    metrics = compare(tmp_path, torch, graph, config, schema, weights, rows=rows,
                      count=count, heads=heads, kv_heads=kv_heads, dim=dim, check_cached=geometry)
    assert metrics["greedy_rows_equal"] == rows
    assert metrics["diffusion_scores"]["max_abs"] <= 0.008, metrics


def test_pinned_sigma_denoiser_stack(tmp_path, record_property):
    checkpoint = os.environ.get("CK_SIGMA_BF16_CHECKPOINT")
    if not checkpoint or not os.environ.get("CK_BLAS_LIBRARY"):
        pytest.skip("configure CK_SIGMA_BF16_CHECKPOINT and CK_BLAS_LIBRARY for the pinned denoiser")
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors")
    torch.set_num_threads(4)
    checkpoint = Path(checkpoint)
    raw_config = (checkpoint / "config.json").read_bytes()
    assert hashlib.sha256(raw_config).hexdigest() == "a964291d6e14ee06c467e3d5edb69af178d27890faebaec2cc9acb260758cfaf"
    pinned = json.loads(raw_config)
    index = (checkpoint / "model.safetensors.index.json").read_bytes()
    assert hashlib.sha256(index).hexdigest() == "7cf433a8e3923d7f87c65d5afb5660e6aa3d64247f4a475bb42817cbaabe1d7d"
    mapping = json.loads(index)["weight_map"]
    rows, count = 3, pinned["num_hidden_layers"]
    heads, kv_heads, dim = pinned["num_attention_heads"], pinned["num_key_value_heads"], pinned["head_dim"]
    graph, config, schema = build_circuit(rows, layers=count, vocabulary=pinned["vocab_size"],
                                          width=pinned["hidden_size"], heads=heads, kv_heads=kv_heads,
                                          dim=dim, inner=pinned["intermediate_size"], blas=True)
    weights = {}
    hashes = {
        "model-00001-of-00002.safetensors": "69ea2c08d7adfeb625b7bce14752dea223edad314e0048b7d43a61742d613fbd",
        "model-00002-of-00002.safetensors": "c6dca19d68edd435c9344dbb1b8087c0c34a0980b975f58a878b27125dfd6115",
    }
    for shard, digest in hashes.items():
        with (checkpoint / shard).open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == digest
        with safetensors.safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
            for k in schema:
                if mapping.get(k) == shard:
                    weight = handle.get_tensor(k)
                    assert weight.dtype == torch.bfloat16 and list(weight.shape) == schema[k][0]
                    weights[k] = weight
    p = pinned["rope_parameters"]
    base, original = p["rope_theta"], p["original_max_position_embeddings"]

    def correction(rotations):
        return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low, high = max(0, math.floor(correction(p["beta_fast"]))), min(dim - 1, math.ceil(correction(p["beta_slow"])))
    frequency = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    blend = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 0.001)).clamp(0, 1)
    weights["inv_freq"] = (1.0 / frequency) * (1 - blend) + (original / pinned["max_position_embeddings"] / frequency) * blend
    metrics = compare(tmp_path, torch, graph, config, schema, weights, rows=rows,
                      count=count, heads=heads, kv_heads=kv_heads, dim=dim, blas=True)
    record_property("checkpoint_shards", json.dumps(hashes))
    record_property("metrics", json.dumps(metrics))
    assert metrics["greedy_rows_equal"] == rows, metrics
    assert metrics["diffusion_scores"]["rmse"] <= 0.05 and metrics["diffusion_scores"]["max_abs"] <= 0.5, metrics
