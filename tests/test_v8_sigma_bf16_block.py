"""Compare one complete generated Sigma block with independent PyTorch math."""
import numpy as np
import pytest

from version.v8.scripts.build_sigma_bf16_block_v8 import build_circuit
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


def bits(tensor):
    import torch
    return tensor.contiguous().view(torch.uint16).numpy()


def run_case(output, torch, rows, width, heads, kv_heads, dim, inner, weights, x, positions, ranges,
             *, blas=False, target_isa="portable", prefix=0):
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    graph, config, schema = build_circuit(rows, width=width, heads=heads, kv_heads=kv_heads,
                                          dim=dim, inner=inner, blas=blas)
    storage = {name: weight.numpy() if name == "inv_freq" else bits(weight)
               for name, weight in weights.items()}
    run, arena, view, _ = compile_checked_weight_graph(output, graph, storage, schema,
                                                        source_config=config, target_isa=target_isa)
    source = view("input", np.uint16)
    source[:] = bits(x)
    view("positions", np.int32)[:] = positions
    mask = torch.zeros((rows, rows), dtype=torch.bool)
    for index, (a, b, c, d) in enumerate(ranges):
        mask[index, a:b] = mask[index, c:d] = True
    visibility = view("ranges", np.int32)
    visibility[:] = ranges
    assert run(arena, len(arena)) == 0

    def norm(value, name):
        y = value.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-5)).bfloat16() * weights[name]

    def project(value, name):
        return F.linear(value, weights[f"{name}_proj.weight"])

    def rotate(value, count):
        value = value.reshape(rows, count, dim).transpose(0, 1)
        angle = positions[:, None].float() * weights["inv_freq"]
        cos, sin = angle.cos().repeat(1, 2).bfloat16(), angle.sin().repeat(1, 2).bfloat16()
        first, second = value.chunk(2, dim=-1)
        return value * cos + torch.cat((-second, first), dim=-1) * sin

    checkpoints = {}
    checkpoints["normalized"] = norm(x, "input_layernorm.weight")
    for port in ("q", "k", "v"):
        checkpoints[port] = project(checkpoints["normalized"], port)
    q = rotate(checkpoints["q"], heads)
    k = rotate(checkpoints["k"], kv_heads)
    v = checkpoints["v"].reshape(rows, kv_heads, dim).transpose(0, 1)
    checkpoints["q_rotary"] = q.transpose(0, 1).reshape(rows, -1)
    checkpoints["k_rotary"] = k.transpose(0, 1).reshape(rows, -1)
    with sdpa_kernel(SDPBackend.MATH):
        context = F.scaled_dot_product_attention(q, k.repeat_interleave(heads // kv_heads, dim=0),
                                                  v.repeat_interleave(heads // kv_heads, dim=0), mask)
    checkpoints["context"] = context.transpose(0, 1).reshape(rows, -1)
    checkpoints["o"] = project(checkpoints["context"], "o")
    checkpoints["attention_residual"] = x + checkpoints["o"]
    checkpoints["post_norm"] = norm(checkpoints["attention_residual"], "post_attention_layernorm.weight")
    checkpoints["gate"] = project(checkpoints["post_norm"], "gate")
    checkpoints["up"] = project(checkpoints["post_norm"], "up")
    checkpoints["swiglu"] = F.silu(checkpoints["gate"]) * checkpoints["up"]
    checkpoints["down"] = project(checkpoints["swiglu"], "down")
    checkpoints["output"] = checkpoints["attention_residual"] + checkpoints["down"]
    metrics = {}
    for name, expected in checkpoints.items():
        actual = view(name, np.uint16)
        different = actual != bits(expected)
        y = (actual.astype(np.uint32) << 16).view(np.float32)
        error = y.astype(np.float64) - expected.float().numpy()
        metrics[name] = {"exact": int(actual.size - np.count_nonzero(different)), "elements": actual.size,
                         "max_abs": float(np.abs(error).max()), "rmse": float(np.sqrt(np.mean(error ** 2)))}

    if prefix:
        cached_graph, cached_config, cached_schema = build_circuit(
            rows - prefix, width=width, heads=heads, kv_heads=kv_heads, dim=dim,
            inner=inner, prefix_rows=prefix, blas=blas,
        )
        (output / "cached").mkdir()
        cached_run, cached_arena, cached_view, _ = compile_checked_weight_graph(
            output / "cached", cached_graph, storage, cached_schema,
            source_config=cached_config, target_isa=target_isa,
        )
        clean = {"key": view("k_rotary", np.uint16)[:prefix].copy(),
                 "value": view("v", np.uint16)[:prefix].copy()}
        for name, data in clean.items():
            cached_view(f"prefix_{name}", np.uint16)[:] = data
        cached_view("input", np.uint16)[:] = bits(x[prefix:])
        cached_view("positions", np.int32)[:] = positions[prefix:]
        cached_view("ranges", np.int32)[:] = ranges[prefix:]
        for _ in range(2):
            assert cached_run(cached_arena, len(cached_arena)) == 0
            for name, data in clean.items():
                np.testing.assert_array_equal(cached_view(f"prefix_{name}", np.uint16), data)
            for name in ("context", "output"):
                np.testing.assert_array_equal(cached_view(name, np.uint16), view(name, np.uint16)[prefix:])

    saved = view("output", np.uint16).copy()
    visibility[0] = [0, rows + 1, rows + 1, rows + 1]
    assert run(arena, len(arena)) != 0
    np.testing.assert_array_equal(view("output", np.uint16), saved)
    visibility[:] = ranges
    source[-1] = bits(-x[-1])
    assert run(arena, len(arena)) == 0
    # Row 0 sees only its own block. The last row can belong to a future block.
    if rows - 1 not in [*range(*ranges[0][:2]), *range(*ranges[0][2:])]:
        np.testing.assert_array_equal(view("output", np.uint16)[0], saved[0])
    assert np.any(view("output", np.uint16)[-1] != saved[-1])
    source[:] = bits(x)
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("output", np.uint16), saved)
    return metrics


@pytest.mark.parametrize("rows", [3, 6])
def test_generated_sigma_block(tmp_path, rows):
    torch = pytest.importorskip("torch")
    width, heads, kv_heads, dim, inner = 13, 4, 2, 6, 9
    _, _, schema = build_circuit(rows, width=width, heads=heads, kv_heads=kv_heads, dim=dim, inner=inner)
    rng = torch.Generator().manual_seed(99)
    weights = {name: (torch.randn(shape, generator=rng) * 0.25).bfloat16()
               for name, (shape, dtype) in schema.items() if dtype == "BF16"}
    weights["inv_freq"] = torch.tensor([0.9, 0.017, 0.0])
    x = torch.randn((rows, width), generator=rng).bfloat16()
    positions = torch.arange(rows, dtype=torch.int32) * 2 + 1
    ranges = np.array([[0, 1, 1, rows - 1]] + [[0, rows, rows, rows]] * (rows - 1), dtype=np.int32)
    metrics = run_case(tmp_path, torch, rows, width, heads, kv_heads, dim, inner, weights, x, positions, ranges, prefix=1)
    for name, metric in metrics.items():
        assert metric["max_abs"] <= 0.008 and metric["rmse"] <= 0.001, (name, metric)


def test_pinned_sigma_block(tmp_path, record_property):
    import hashlib
    import json
    import math
    import os
    from pathlib import Path

    checkpoint = os.environ.get("CK_SIGMA_BF16_CHECKPOINT")
    if not checkpoint:
        pytest.skip("CK_SIGMA_BF16_CHECKPOINT is not configured")
    if not os.environ.get("CK_BLAS_LIBRARY"):
        pytest.skip("CK_BLAS_LIBRARY is required for the pinned block replay")
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors")
    from version.v8.scripts.capture_sigma_bf16_attention_ingress_v8 import PREFIX, WEIGHTS

    checkpoint = Path(checkpoint)
    config = (checkpoint / "config.json").read_bytes()
    assert hashlib.sha256(config).hexdigest() == "a964291d6e14ee06c467e3d5edb69af178d27890faebaec2cc9acb260758cfaf"
    config = json.loads(config)
    indexed = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    pinned = {
        "input_layernorm.weight": WEIGHTS["norm_weight"],
        **{f"{port}_proj.weight": WEIGHTS[f"{port}_weight"] for port in ("q", "k", "v")},
        "o_proj.weight": ("self_attn.o_proj.weight", "9de964eed1b4c1f7778b2f6d2493a902c13a113ec2fd0af6468d98aaad61554a"),
        "gate_proj.weight": ("mlp.gate_proj.weight", "d56eafa7cf1ffbbc653e06dea8075d5b8de1805f61a5940a0fe45d1f29ee4c17"),
        "up_proj.weight": ("mlp.up_proj.weight", "2897419db183ec319cb5cdf8d4fae87608dd9d9482c16618e1cc970073bbf019"),
        "down_proj.weight": ("mlp.down_proj.weight", "6859601806b7a565dcf290638abe10d4e8f285e8cd8c1ab11fd262a7f40cd226"),
        "post_attention_layernorm.weight": ("post_attention_layernorm.weight", "1f5a5599157a17d835e601d88b5bcc15e036de434b4b64a9ff826076bd181b98"),
    }
    weights = {}
    for name, (field, digest) in pinned.items():
        key = PREFIX + field
        shard = indexed[key]
        assert Path(shard).name == shard
        with safetensors.safe_open(checkpoint / shard, framework="pt", device="cpu") as archive:
            value = archive.get_tensor(key)
        assert value.dtype == torch.bfloat16 and hashlib.sha256(bits(value).tobytes()).hexdigest() == digest
        weights[name] = value

    dim = config["head_dim"]
    params = config["rope_parameters"]
    base, original = params["rope_theta"], params["original_max_position_embeddings"]
    stretch = config["max_position_embeddings"] / original
    low, high = [
        clamp(round_boundary(dim * math.log(original / (params[beta] * 2 * math.pi)) / (2 * math.log(base))), bound)
        for beta, clamp, round_boundary, bound in (
            ("beta_fast", max, math.floor, 0), ("beta_slow", min, math.ceil, dim - 1)
        )
    ]
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    mix = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 0.001)).clamp(0, 1)
    weights["inv_freq"] = pos_freqs.reciprocal() * (1 - mix) + (stretch * pos_freqs).reciprocal() * mix

    width = config["hidden_size"]
    torch.set_num_threads(4)
    prefix, rows = 7, 39
    x = torch.randn((rows, width), generator=torch.Generator().manual_seed(3112)).bfloat16()
    positions = torch.arange(rows, dtype=torch.int32)
    ranges = np.array([[0, prefix, prefix, prefix]] * prefix + [[0, rows, rows, rows]] * (rows - prefix),
                      dtype=np.int32)
    metrics = run_case(tmp_path, torch, rows, width, config["num_attention_heads"],
                       config["num_key_value_heads"], dim, config["intermediate_size"],
                       weights, x, positions, ranges, blas=True, target_isa="avx2", prefix=prefix)
    record_property("checkpoint", "7eff02e0b72dace46d9084430330c192bf5a4386")
    record_property("torch", torch.__version__)
    record_property("errors", json.dumps(metrics, sort_keys=True))
    assert metrics["output"]["max_abs"] <= 0.03125 and metrics["output"]["rmse"] <= 0.0005, metrics
