"""Certify BF16 normalization and connected Q/K/V through normal checked codegen."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from version.v8.scripts.build_sigma_bf16_attention_ingress_v8 import build_circuit
from version.v8.scripts.checked_bundle_evidence_v8 import checked_target_flags
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


def compile_ingress(folder, rows, weights, *, target_isa="portable", **geometry):
    graph, config = build_circuit(rows, **geometry)
    return compile_checked_weight_graph(
        folder, graph, weights, {name: (list(array.shape), "BF16") for name, array in weights.items()},
        source_config=config, target_isa=target_isa,
    )


def expand(data):
    return (data.astype(np.uint32) << 16).view(np.float32)


def test_generated_bf16_ingress_and_recovery(tmp_path):
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(765)
    rows, width, q_width, kv_width = 3, 13, 8, 4
    source = torch.randn((rows, width), generator=generator).bfloat16()
    gamma = torch.randn(width, generator=generator).bfloat16()
    projections = {name: torch.randn((channels, width), generator=generator).bfloat16()
                   for name, channels in (("q", q_width), ("k", kv_width), ("v", kv_width))}
    weights = {"input_layernorm.weight": gamma.view(torch.uint16).numpy()}
    weights.update({f"{name}_proj.weight": value.view(torch.uint16).numpy()
                    for name, value in projections.items()})
    run, arena, view, _ = compile_ingress(tmp_path, rows, weights, width=width,
                                         q_width=q_width, kv_width=kv_width)
    inputs = view("input", np.uint16)
    inputs[:] = source.view(torch.uint16).numpy()
    assert run(arena, len(arena)) == 0
    fp32 = source.float()
    expected_norm = (fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + 1e-5)).bfloat16() * gamma
    np.testing.assert_array_equal(view("normalized", np.uint16), expected_norm.view(torch.uint16).numpy())
    snapshots = {}
    for name, weight in projections.items():
        expected = torch.nn.functional.linear(expected_norm, weight).view(torch.uint16).numpy()
        actual = view(name, np.uint16)
        np.testing.assert_array_equal(actual, expected)
        snapshots[name] = actual.copy()
    # The last input is validated before normalized rows and projections publish.
    for invalid in (0x7F80, 0x7FC0):
        inputs[-1, -1] = invalid
        assert run(arena, len(arena)) != 0
        np.testing.assert_array_equal(view("normalized", np.uint16), expected_norm.view(torch.uint16).numpy())
        for name, before in snapshots.items():
            np.testing.assert_array_equal(view(name, np.uint16), before)
    inputs[:] = source.view(torch.uint16).numpy()
    assert run(arena, len(arena)) == 0
    for name, before in snapshots.items():
        np.testing.assert_array_equal(view(name, np.uint16), before)


def test_pinned_layer0_ingress(tmp_path, record_property):
    path = os.environ.get("CK_SIGMA_BF16_LAYER0_FIXTURE")
    if path is None:
        pytest.skip("CK_SIGMA_BF16_LAYER0_FIXTURE is not configured")
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["checkpoint"] == "7eff02e0b72dace46d9084430330c192bf5a4386"
    with path.open("rb") as file:
        assert hashlib.file_digest(file, "sha256").hexdigest() == meta["fixture_sha256"]
    try:
        checked_target_flags("avx2")
    except ValueError as exc:
        pytest.skip(str(exc))
    with np.load(path) as archive:
        tensors = dict(archive)
    inputs = tensors["unnormalized"]
    weights = {"input_layernorm.weight": tensors["norm_weight"]}
    for name in ("q", "k", "v"):
        weights[f"{name}_proj.weight"] = tensors[f"{name}_weight"]
    assert inputs.shape == (32, 3072)
    run, arena, view, _ = compile_ingress(tmp_path, len(inputs), weights, target_isa="avx2")
    view("input", np.uint16)[:] = inputs
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("normalized", np.uint16), tensors["hidden"])
    for name in ("q", "k", "v"):
        actual = expand(view(name, np.uint16))
        expected = expand(tensors[name])
        np.testing.assert_allclose(actual, expected, rtol=0.008, atol=0.0001)
        record_property(f"{name}_max_abs", float(np.abs(actual - expected).max()))
