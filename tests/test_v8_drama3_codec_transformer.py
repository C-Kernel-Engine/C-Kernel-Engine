"""Compare a connected generated codec transformer against independent arithmetic."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.test_v8_drama3_codec_rvq import invoke
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.tts.build_drama3_codec_transformer_circuit import build_circuit


def compile_transformer(folder, frames, weights, **geometry):
    graph, config = build_circuit(frames, **geometry)
    declarations = {name: (list(value.shape), "F32") for name, value in weights.items()}
    # The sliding provider shares a translation unit with legacy attention providers.
    return compile_checked_weight_graph(folder, graph, weights, declarations, source_config=config,
                                        openmp=True, provider_warnings="report", c_standard="gnu11")


def set_rotary(runtime, cos, sin):
    view = runtime[2]
    view("rotary_cos")[:] = cos
    view("rotary_sin")[:] = sin


@pytest.fixture
def synthetic(tmp_path):
    rng = np.random.default_rng(9)
    frames, heads, head_dim, layers, inner, window = 4, 2, 4, 2, 10, 2
    width = heads * head_dim
    weights = {}

    def tensor(name, shape):
        weights[name] = rng.normal(0, 0.25, shape).astype(np.float32)

    for i in range(10):
        tensor(f"book{i}.table", (5, 3))
        tensor(f"book{i}.weight", (width, 3))
        tensor(f"book{i}.bias", (width,))
    for i in range(layers):
        prefix = f"codec_layer{i}"
        for name in ("attention.q", "attention.k", "attention.v", "attention.wo"):
            tensor(f"{prefix}.{name}.weight", (width, width))
        for name in ("attention_norm.weight", "ffn_norm.weight",
                     "attention_layer_scale.gamma", "ffn_layer_scale.gamma"):
            tensor(f"{prefix}.{name}", (width,))
        for name in ("w1", "w3"):
            tensor(f"{prefix}.feed_forward.{name}.weight", (inner, width))
        tensor(f"{prefix}.feed_forward.w2.weight", (width, inner))
    tensor("codec_transformer_norm.weight", (width,))
    runtime = compile_transformer(tmp_path, frames, weights, vocabularies=(5,) * 10, book_width=3,
                                  heads=heads, head_dim=head_dim, layers=layers,
                                  intermediate=inner, window=window)
    codes = rng.integers(0, 5, (10, frames), dtype=np.int32)
    angles = np.arange(frames, dtype=np.float32)[:, None] * np.array([1, 0.01], dtype=np.float32)
    cos, sin = np.cos(angles), np.sin(angles)
    set_rotary(runtime, cos, sin)
    return runtime, codes, weights, cos, sin, (frames, heads, head_dim, layers, window)


def test_connected_causal_window_rejection_and_recovery(synthetic):
    runtime, codes, _, cos, sin, _ = synthetic
    view = runtime[2]
    assert invoke(runtime, codes) == 0
    baseline = view("codec_transformer_out").copy()
    changed = codes.copy()
    changed[0, 0] = (changed[0, 0] + 1) % 5
    assert invoke(runtime, changed) == 0
    assert not np.array_equal(view("codec_transformer_out")[0], baseline[0])
    # Two layers with width two cannot propagate a change over three positions.
    np.testing.assert_array_equal(view("codec_transformer_out")[3], baseline[3])
    latest = view("codec_transformer_out").copy()
    for bad_id in (-1, 5):
        invalid = codes.copy()
        invalid[-1, -1] = bad_id
        assert invoke(runtime, invalid) != 0
        np.testing.assert_array_equal(view("codec_transformer_out"), latest)
    view("rotary_sin")[-1, 0] = np.nan
    assert invoke(runtime, codes) != 0
    np.testing.assert_array_equal(view("codec_transformer_out"), latest)
    set_rotary(runtime, cos, sin)
    assert invoke(runtime, codes) == 0
    np.testing.assert_array_equal(view("codec_transformer_out"), baseline)


def test_nonzero_reference(synthetic):
    torch = pytest.importorskip("torch")
    runtime, codes, weights, cos, sin, (frames, heads, head_dim, layers, window) = synthetic
    assert invoke(runtime, codes) == 0
    view = runtime[2]
    tensors = {name: torch.from_numpy(value) for name, value in weights.items()}

    def linear(x, prefix, bias=None):
        return torch.nn.functional.linear(x, tensors[f"{prefix}.weight"], bias)

    def norm(x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * tensors[name]

    def rotate(x):
        paired = x.reshape(frames, heads, head_dim // 2, 2)
        c, s = torch.from_numpy(cos)[:, None], torch.from_numpy(sin)[:, None]
        a, b = paired.unbind(-1)
        return torch.stack((a * c - b * s, b * c + a * s), dim=-1).flatten(2).transpose(0, 1)

    books = [linear(tensors[f"book{i}.table"][codes[i]], f"book{i}", tensors[f"book{i}.bias"])
             for i in range(10)]
    x = books[0] + sum(books[1:])
    positions = torch.arange(frames)
    lag = positions[:, None] - positions
    mask = (lag >= 0) & (lag < window)
    for i in range(layers):
        name = f"codec_layer{i}"
        normalized = norm(x, f"{name}.attention_norm.weight")
        q, k = (rotate(linear(normalized, f"{name}.attention.{port}")) for port in ("q", "k"))
        v = linear(normalized, f"{name}.attention.v").reshape(frames, heads, head_dim).transpose(0, 1)
        attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attention = linear(attention.transpose(0, 1).reshape(frames, -1), f"{name}.attention.wo")
        x = x + attention * tensors[f"{name}.attention_layer_scale.gamma"]
        mlp = norm(x, f"{name}.ffn_norm.weight")
        gated = torch.nn.functional.silu(linear(mlp, f"{name}.feed_forward.w1"))
        gated = gated * linear(mlp, f"{name}.feed_forward.w3")
        x = x + linear(gated, f"{name}.feed_forward.w2") * tensors[f"{name}.ffn_layer_scale.gamma"]
        np.testing.assert_allclose(view(f"{name}_out"), x.numpy(), atol=1e-6, rtol=1e-6)
    reference = norm(x, "codec_transformer_norm.weight")
    np.testing.assert_allclose(view("codec_transformer_out"), reference.numpy(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("frames", [1, 3, 8])
def test_pinned_codec_transformer(tmp_path, frames, record_property):
    location = os.environ.get("CK_DRAMA3_CODEC_TRANSFORMER_REFERENCE_DIR")
    if location is None:
        pytest.skip("CK_DRAMA3_CODEC_TRANSFORMER_REFERENCE_DIR is not configured")
    folder = Path(location)
    report = json.loads((folder / "reference-result.json").read_text())
    assert report["model_revision"] == "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216"
    assert report["implementation"] == "dfb6b43b360b04e53daa7c36aed0bc448789d3a4"
    assert report["codec_sha256"] == "74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f"
    geometry = report["transformer"]
    assert geometry == {"layers": 8, "heads": 16, "head_dim": 64, "intermediate": 3072, "window": 128}
    weights = {}
    for name in ("rvq_weights.npz", "transformer_weights.npz", f"reference-codes-{frames}.npz"):
        with (folder / name).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == report["replay_sha256"][name]
        if name.endswith("_weights.npz"):
            with np.load(folder / name) as archive:
                weights.update(archive)
    runtime = compile_transformer(tmp_path, frames, weights, **geometry)
    with np.load(folder / f"reference-codes-{frames}.npz") as fixture:
        set_rotary(runtime, fixture["rotary_cos"], fixture["rotary_sin"])
        assert invoke(runtime, fixture["codes"][0]) == 0
        view = runtime[2]
        for i in range(geometry["layers"]):
            name = f"codec_layer{i}_out"
            np.testing.assert_allclose(view(name), fixture[name][0], atol=1e-5, rtol=1e-5)
        expected = fixture["quantizer_post_transformer"][0].T
        actual = view("codec_transformer_out")
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        record_property("max_abs", float(np.abs(actual - expected).max()))
