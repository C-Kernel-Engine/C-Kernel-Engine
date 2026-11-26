"""Compare Drama 3's first generated shared tower layer with PyTorch."""
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.tts.build_drama3_bf16_tower_circuit import build_circuit


def check_layer(output, torch, rows, geometry, table, weights, *, blas=False):
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    width, heads, kv_heads, dim, rotary, vocabulary = (geometry[name] for name in
        ("width", "heads", "kv_heads", "dim", "rotary", "vocabulary"))

    def bits(x):
        return x.contiguous().view(torch.uint16).numpy()

    graph, config, schema = build_circuit(rows, **geometry, blas=blas)
    storage = {name: weight.numpy() if name == "inv_freq" else bits(weight)
               for name, weight in weights.items()}
    run, arena, view, _ = compile_checked_weight_graph(
        output, graph, storage, schema, source_config=config, target_isa="avx2" if blas else "portable",
    )
    rng = torch.Generator().manual_seed(381)
    hidden = torch.randn((rows, width), generator=rng).bfloat16()
    unseen = torch.randn((rows, width), generator=rng).bfloat16() * 0.1
    codes = torch.arange(rows, dtype=torch.int32) % vocabulary
    codes[0] = vocabulary
    unseen[0] = 0
    positions = torch.arange(rows, dtype=torch.int32) + 2
    ranges = np.array([[0, i, i, i] for i in range(1, rows + 1)], dtype=np.int32)
    padded_table = torch.cat((table, torch.zeros_like(table[:1])))
    for name, tensor in (("hidden", hidden), ("unseen_sum", unseen), ("embedding_table", padded_table)):
        view(name, np.uint16)[:] = bits(tensor)
    for name, array in (("code_rows", codes.numpy()), ("positions", positions.numpy()), ("ranges", ranges)):
        view(name, np.int32)[:] = array
    assert run(arena, len(arena)) == 0

    def rmsnorm(x, key):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) *
                (1.0 + weights[key].float())).bfloat16()

    def rotate(x, count):
        value = x.reshape(rows, count, dim)
        angle = positions[:, None].float() * weights["inv_freq"]
        cos, sin = angle.cos()[:, None], angle.sin()[:, None]
        even, odd = value[..., :rotary:2].float(), value[..., 1:rotary:2].float()
        prefix = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
        return torch.cat((prefix.flatten(-2).bfloat16(), value[..., rotary:]), dim=-1).reshape(rows, -1)

    x = {"embedding": padded_table[codes]}
    x["enriched"] = x["embedding"] + unseen
    x["enorm"] = rmsnorm(x["enriched"], "enorm.weight")
    x["hnorm"] = rmsnorm(hidden, "hnorm.weight")
    x["concat"] = torch.cat((x["enorm"], x["hnorm"]), dim=-1)
    x["mixed"] = F.linear(x["concat"], weights["eh_proj.weight"])
    layer = "transformer_layer."
    attn = layer + "attention."
    x["attn_norm"] = rmsnorm(x["mixed"], layer + "attention_norm.weight")
    ports = ("q", "attention_gate", "k", "v")
    fused = torch.cat([weights[attn + port + ".weight"] for port in ports])
    pieces = F.linear(x["attn_norm"], fused).split((heads * dim, heads * dim, kv_heads * dim, kv_heads * dim), -1)
    x.update(zip(ports, pieces, strict=True))
    for port, count in (("q", heads), ("k", kv_heads)):
        x[f"{port}_norm"] = rmsnorm(x[port].reshape(rows * count, dim), attn + f"{port}_norm.weight").reshape(rows, -1)
        x[f"{port}_rotary"] = rotate(x[f"{port}_norm"], count)
    q, k, v = (x[name].reshape(rows, count, dim).transpose(0, 1)
               for name, count in (("q_rotary", heads), ("k_rotary", kv_heads), ("v", kv_heads)))
    mask = positions[:, None] >= positions
    with sdpa_kernel(SDPBackend.MATH):
        context = F.scaled_dot_product_attention(q, k.repeat_interleave(heads // kv_heads, dim=0),
                                                  v.repeat_interleave(heads // kv_heads, dim=0), mask)
    x["context"] = context.transpose(0, 1).reshape(rows, -1)
    x["gated_context"] = x["context"] * x["attention_gate"].sigmoid()
    x["o"] = F.linear(x["gated_context"], weights[attn + "wo.weight"])
    x["attn_residual"] = x["mixed"] + x["o"]
    x["ffn_norm"] = rmsnorm(x["attn_residual"], layer + "ffn_norm.weight")
    mlp = layer + "feed_forward."
    x["gate"] = F.linear(x["ffn_norm"], weights[mlp + "w1.weight"])
    x["up"] = F.linear(x["ffn_norm"], weights[mlp + "w3.weight"])
    x["swiglu"] = F.silu(x["gate"]) * x["up"]
    x["down"] = F.linear(x["swiglu"], weights[mlp + "w2.weight"])
    x["pre_final_norm"] = x["attn_residual"] + x["down"]
    x["output"] = rmsnorm(x["pre_final_norm"], "final_layernorm.weight")
    results = {}
    for name, expected in x.items():
        actual = view(name, np.uint16)
        expanded = (actual.astype(np.uint32) << 16).view(np.float32)
        err = expanded.astype(np.float64) - expected.float().numpy()
        results[name] = {"exact": int(np.sum(actual == bits(expected))), "elements": actual.size,
                         "max_abs": float(np.max(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}
    before = view("output", np.uint16).copy()
    view("code_rows", np.int32)[-1] = -1
    assert run(arena, len(arena)) != 0
    np.testing.assert_array_equal(view("output", np.uint16), before)
    view("code_rows", np.int32)[-1] = vocabulary
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("output", np.uint16)[:-1], before[:-1])
    view("code_rows", np.int32)[-1] = codes[-1]
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("output", np.uint16), before)
    return results


def test_generated_first_shared_layer(tmp_path):
    torch = pytest.importorskip("torch")
    geometry = dict(width=13, heads=4, kv_heads=2, dim=6, rotary=4, inner=17, vocabulary=7)
    _, _, schema = build_circuit(3, **geometry)
    rng = torch.Generator().manual_seed(4830)
    weights = {name: (torch.randn(shape, generator=rng) * 0.15).bfloat16()
               for name, (shape, kind) in schema.items() if kind == "BF16"}
    weights["inv_freq"] = torch.tensor([1.0, 0.01])
    table = torch.randn((7, 13), generator=rng).bfloat16()
    metrics = check_layer(tmp_path, torch, 3, geometry, table, weights)
    for name, values in metrics.items():
        assert values["max_abs"] <= .016 and values["rmse"] <= .004, (name, values)


def test_pinned_first_shared_layer(tmp_path, record_property):
    root = os.environ.get("CK_DRAMA3_TOWER0_RANGE_DIR")
    if not root or not os.environ.get("CK_BLAS_LIBRARY"):
        pytest.skip("configure CK_DRAMA3_TOWER0_RANGE_DIR and CK_BLAS_LIBRARY for the pinned replay")
    torch = pytest.importorskip("torch")
    geometry = dict(width=2048, heads=16, kv_heads=2, dim=256, rotary=64, inner=6144, vocabulary=4096)
    torch.set_num_threads(4)
    root = Path(root)
    header = (root / "shard14.header").read_bytes()
    assert hashlib.sha256(header).hexdigest() == "4e289ef05b3b3affb7b7497095db9d28f6b115a07a58b9487e13f75a29aa9020"
    with (root / "shared0.bin").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == "f7d0ddcb8a240368b5b0090d1c8faad00004e46646946f51e42007f279b3bef9"
    n, = struct.unpack_from("<Q", header)
    weight_map = json.loads(header[8:8 + n])
    prefix = "audio_decoder_mtp_shared.layers.0."
    contents = np.memmap(root / "shared0.bin", mode="r", dtype=np.uint16)
    weights = {}
    for field, meta in weight_map.items():
        if field.startswith(prefix):
            start, end = ((item - 382108672) // 2 for item in meta["data_offsets"])
            assert meta["dtype"] == "BF16" and 0 <= start < end <= contents.size
            weights[field.removeprefix(prefix)] = torch.from_numpy(
                contents[start:end].reshape(meta["shape"]).copy()).view(torch.bfloat16)
    table = weights.pop("semantic_embedding.weight")
    attention = "transformer_layer.attention."
    packed = weights.pop(attention + "wqkv.weight")
    for port, matrix in zip(("q", "attention_gate", "k", "v"), packed.split((4096, 4096, 512, 512)), strict=True):
        weights[attention + port + ".weight"] = matrix
    weights["inv_freq"] = 10000000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    metrics = check_layer(tmp_path, torch, 5, geometry, table, weights, blas=True)
    record_property("errors", json.dumps(metrics, sort_keys=True))
    assert metrics["output"]["max_abs"] <= .125 and metrics["output"]["rmse"] <= .008, metrics
