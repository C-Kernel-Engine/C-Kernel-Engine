"""Generated first Clef evidence-routing layer with independent PyTorch parity."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.test_v8_clef_flash_head_ingress import compile_case, weights_for
from tests.test_v8_attention_cross_token_major_checked import reference as attention_reference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/decision"))
from build_clef_flash_evidence0_circuit import (  # noqa: E402
    OUTPUT, build_circuit, split_packed_attention_projection)

PREFIX = "evidence_layers.0"
ASSET_HASH = "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0"
PINNED_LAYER_ATOL = 2e-6  # P3 observed worst 1.133e-6 at evidence0_v.


def synthetic_weights(channels, width, vocabulary, feedforward, *, seed=223):
    weights = weights_for(channels, width, vocabulary, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for name in ("query_norm", "memory_norm", "feedforward_norm"):
        stem = f"{PREFIX}.{name}"
        weights[f"{stem}.weight"] = rng.uniform(0.8, 1.2, width).astype("f4")
        weights[f"{stem}.bias"] = rng.normal(0, 0.05, width).astype("f4")
    for name in ("attention.q_proj", "attention.k_proj",
                 "attention.v_proj", "attention.out_proj",
                 "feedforward.0", "feedforward.3"):
        stem = f"{PREFIX}.{name}"
        out = feedforward if name == "feedforward.0" else width
        inp = feedforward if name == "feedforward.3" else width
        weights[f"{stem}.weight"] = rng.normal(0, 0.12, (out, inp)).astype("f4")
        weights[f"{stem}.bias"] = rng.normal(0, 0.03, out).astype("f4")
    return weights


def pinned_weights(asset, vocabulary=16):
    from safetensors import safe_open
    identity = json.loads((ROOT / "tests/fixtures/decision/clef_flash/identity.json").read_text())
    if (hashlib.sha256(Path(asset).read_bytes()).hexdigest() != ASSET_HASH or
            identity["joint_head_weights_sha256"] != ASSET_HASH):
        raise ValueError("pinned Clef head archive identity differs")
    with safe_open(asset, framework="pt") as archive:
        names = ("hidden_norm.weight", "hidden_norm.bias",
                 "memory_projection.weight", "question_projection.weight",
                 "option_context_projection.weight", "option_lexical_projection.weight",
                 "option_question_projection.weight", "global_projection.weight",
                 f"{PREFIX}.query_norm.weight", f"{PREFIX}.query_norm.bias",
                 f"{PREFIX}.memory_norm.weight", f"{PREFIX}.memory_norm.bias",
                 f"{PREFIX}.feedforward_norm.weight", f"{PREFIX}.feedforward_norm.bias",
                 f"{PREFIX}.attention.out_proj.weight",
                 f"{PREFIX}.attention.out_proj.bias",
                 f"{PREFIX}.feedforward.0.weight", f"{PREFIX}.feedforward.0.bias",
                 f"{PREFIX}.feedforward.3.weight", f"{PREFIX}.feedforward.3.bias")
        result = {name: archive.get_tensor(name).float().numpy() for name in names}
        packed_weight = archive.get_tensor(f"{PREFIX}.attention.in_proj_weight").float().numpy()
        packed_bias = archive.get_tensor(f"{PREFIX}.attention.in_proj_bias").float().numpy()
    result.update(split_packed_attention_projection(packed_weight, packed_bias))
    result["output_embedding.weight"] = np.random.default_rng(113).normal(
        0, 0.02, (vocabulary, 4096)).astype("f4")
    return result


def prepare(view, hidden):
    tokens = hidden.shape[0]
    view("hidden")[:] = hidden
    view("input_ids")[:] = np.arange(1, tokens + 1, dtype="i4")
    view("question_spans")[:] = [[0, 2], [tokens - 3, tokens]]
    view("option_spans")[:] = [[1, 3], [2, tokens - 1], [tokens - 1, tokens]]
    view("global_span")[:] = [[tokens - 1, tokens]]
    view("option_counts")[:] = [2, 1]


def torch_layer(queries, memory, weights, heads, prefix=PREFIX):
    import torch
    import torch.nn.functional as functional

    def tensor(name):
        return torch.from_numpy(weights[f"{prefix}.{name}"])

    def norm(x, name):
        return functional.layer_norm(x, (x.shape[-1],),
                                     tensor(f"{name}.weight"),
                                     tensor(f"{name}.bias"), eps=1e-5)

    def linear(x, name):
        return functional.linear(x, tensor(f"{name}.weight"),
                                 tensor(f"{name}.bias"))

    qn = norm(torch.from_numpy(queries), "query_norm")
    mn = norm(torch.from_numpy(memory), "memory_norm")
    q = linear(qn, "attention.q_proj")
    k = linear(mn, "attention.k_proj")
    v = linear(mn, "attention.v_proj")
    width = q.shape[-1]
    head_dim = width // heads

    def shaped(x):
        return x.reshape(1, -1, heads, head_dim).transpose(1, 2)

    context = functional.scaled_dot_product_attention(
        shaped(q), shaped(k), shaped(v), dropout_p=0.0,
        is_causal=False).transpose(1, 2).reshape_as(q)
    out = linear(context, "attention.out_proj")
    residual = torch.from_numpy(queries) + out
    ff_norm = norm(residual, "feedforward_norm")
    ff1 = linear(ff_norm, "feedforward.0")
    gelu = functional.gelu(ff1)
    ff2 = linear(gelu, "feedforward.3")
    return {
        "evidence0_query_norm": qn, "evidence0_memory_norm": mn,
        "evidence0_q": q, "evidence0_k": k, "evidence0_v": v,
        "evidence0_context": context, "evidence0_attn_out": out,
        "evidence0_attn_residual": residual,
        "evidence0_ff_norm": ff_norm, "evidence0_ff1": ff1,
        "evidence0_gelu": gelu, "evidence0_ff2": ff2,
        "evidence0_output": residual + ff2,
    }


class ClefFlashEvidence0Test(unittest.TestCase):
    def test_packed_qkv_importer_preserves_rows(self):
        packed_weight = np.arange(3 * 4 * 4, dtype="f4").reshape(12, 4)
        packed_bias = np.arange(12, dtype="f4")
        parts = split_packed_attention_projection(packed_weight, packed_bias)
        for suffix in ("weight", "bias"):
            reconstructed = np.concatenate([
                parts[f"{PREFIX}.attention.{name}_proj.{suffix}"]
                for name in ("q", "k", "v")])
            np.testing.assert_array_equal(reconstructed,
                packed_weight if suffix == "weight" else packed_bias)
        with self.assertRaises(ValueError):
            split_packed_attention_projection(packed_weight[:-1], packed_bias)

    def test_circuit_regenerates(self):
        self.assertEqual(OUTPUT.read_text(), json.dumps(build_circuit(), indent=2) + "\n")

    def test_connected_synthetic_and_recovery(self):
        graph = build_circuit(channels=8, width=8, heads=2, feedforward=16)
        weights = synthetic_weights(8, 8, 16, 16)
        with tempfile.TemporaryDirectory() as temp:
            arena, execute, view, call_ir, _loaded = compile_case(Path(temp), graph, weights)
            self.assertEqual(call_ir["errors"], [])
            self.assertEqual(len(call_ir["operations"]), 28)
            first = np.random.default_rng(101).normal(0, 0.5, (6, 8)).astype("f4")
            second = np.random.default_rng(102).normal(0, 0.5, (6, 8)).astype("f4")
            prepare(view, first)
            self.assertEqual(execute(arena, len(arena)), 0)
            np.testing.assert_allclose(view("evidence0_context"),
                attention_reference(view("evidence0_q"), view("evidence0_k"),
                                    view("evidence0_v"), 2), atol=1e-6, rtol=0)
            baseline = view("evidence0_output").copy()
            prepare(view, second)
            self.assertEqual(execute(arena, len(arena)), 0)
            self.assertFalse(np.array_equal(view("evidence0_output"), baseline))
            prepare(view, first)
            self.assertEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("evidence0_output"), baseline)
            view("option_counts")[:] = [1, 1]
            view("evidence0_output")[:] = -83
            self.assertNotEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("evidence0_output"), -83)
            view("option_counts")[:] = [2, 1]
            self.assertEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("evidence0_output"), baseline)
            layout = json.loads((Path(temp) / "layout.json").read_text())
            entry = next(item for item in layout["memory"]["weights"]["entries"]
                         if item["name"] == f"{PREFIX}.attention.q_proj.weight")
            q_weight = np.ndarray((8, 8), dtype="f4", buffer=arena,
                                  offset=entry["abs_offset"])
            original = q_weight.copy()
            q_weight[:] = 0
            self.assertEqual(execute(arena, len(arena)), 0)
            self.assertFalse(np.array_equal(view("evidence0_output"), baseline))
            q_weight[:] = original
            self.assertEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("evidence0_output"), baseline)

    def test_connected_pytorch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch comparison unavailable")
        graph = build_circuit(channels=8, width=8, heads=2, feedforward=16)
        weights = synthetic_weights(8, 8, 16, 16, seed=311)
        with tempfile.TemporaryDirectory() as temp:
            arena, execute, view, _call_ir, _loaded = compile_case(Path(temp), graph, weights)
            prepare(view, np.random.default_rng(312).normal(0, 0.5, (6, 8)).astype("f4"))
            self.assertEqual(execute(arena, len(arena)), 0)
            expected = torch_layer(view("option_queries").copy(),
                                   view("memory").copy(), weights, 2)
            for name, reference in expected.items():
                np.testing.assert_allclose(view(name), reference.numpy(),
                                           atol=3e-6, rtol=0, err_msg=name)

    def test_pinned_head_weights(self):
        asset = os.environ.get("CKE_CLEF_FLASH_HEAD_PATH")
        if not asset:
            self.skipTest("pinned Clef head weights unavailable")
        try:
            import torch
            import safetensors  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch/safetensors comparison unavailable")
        graph = build_circuit()
        weights = pinned_weights(asset)
        with tempfile.TemporaryDirectory() as temp:
            arena, execute, view, _call_ir, _loaded = compile_case(Path(temp), graph, weights)
            prepare(view, np.random.default_rng(403).normal(0, 0.2, (6, 4096)).astype("f4"))
            self.assertEqual(execute(arena, len(arena)), 0)
            expected = torch_layer(view("option_queries").copy(),
                                   view("memory").copy(), weights, 16)
            for name, reference in expected.items():
                np.testing.assert_allclose(view(name), reference.numpy(),
                                           atol=PINNED_LAYER_ATOL, rtol=0,
                                           err_msg=name)

    def test_direct_pinned_reference_layer(self):
        asset = os.environ.get("CKE_CLEF_FLASH_HEAD_PATH")
        source_path = os.environ.get("CKE_CLEF_FLASH_REFERENCE_SOURCE")
        if not asset or not source_path:
            self.skipTest("pinned Clef head weights/reference source unavailable")
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch comparison unavailable")
        identity = json.loads((ROOT / "tests/fixtures/decision/clef_flash/identity.json").read_text())
        if hashlib.sha256(Path(source_path).read_bytes()).hexdigest() != identity["joint_schema_model_sha256"]:
            self.fail("pinned Clef reference source identity differs")
        spec = importlib.util.spec_from_file_location("clef_pinned_reference", source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        weights = pinned_weights(asset)
        layer = module.EvidenceRoutingLayer(
            width=1024, heads=16, feedforward=4096, dropout=0.0)
        state = {}
        for name, value in weights.items():
            if name.startswith(PREFIX + ".") and ".attention.q_proj." not in name and ".attention.k_proj." not in name and ".attention.v_proj." not in name:
                state[name[len(PREFIX) + 1:]] = torch.from_numpy(value)
        for suffix in ("weight", "bias"):
            state[f"attention.in_proj_{suffix}"] = torch.from_numpy(np.concatenate([
                weights[f"{PREFIX}.attention.{name}_proj.{suffix}"]
                for name in ("q", "k", "v")]))
        layer.load_state_dict(state, strict=True)
        layer.eval()
        graph = build_circuit()
        with tempfile.TemporaryDirectory() as temp:
            arena, execute, view, _call_ir, _loaded = compile_case(Path(temp), graph, weights)
            prepare(view, np.random.default_rng(403).normal(0, 0.2, (6, 4096)).astype("f4"))
            self.assertEqual(execute(arena, len(arena)), 0)
            with torch.no_grad():
                reference = layer(
                    torch.from_numpy(view("option_queries").copy()).unsqueeze(0),
                    torch.from_numpy(view("memory").copy()).unsqueeze(0))[0]
            np.testing.assert_allclose(view("evidence0_output"), reference.numpy(),
                                       atol=PINNED_LAYER_ATOL, rtol=0)


if __name__ == "__main__":
    unittest.main()
