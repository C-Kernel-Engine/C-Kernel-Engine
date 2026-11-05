"""Both Clef evidence-routing layers through one ordinary generated entry."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.test_v8_clef_flash_head_ingress import compile_case
from tests.test_v8_clef_flash_evidence0 import (
    ASSET_HASH, pinned_weights, prepare, synthetic_weights, torch_layer)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/decision"))
from build_clef_flash_evidence_complete_circuit import OUTPUT, build_circuit  # noqa: E402


def both_synthetic_weights(channels, width, vocabulary, feedforward):
    first = synthetic_weights(channels, width, vocabulary, feedforward, seed=311)
    second = synthetic_weights(channels, width, vocabulary, feedforward, seed=517)
    for name, value in second.items():
        if name.startswith("evidence_layers.0."):
            first[name.replace("evidence_layers.0.", "evidence_layers.1.", 1)] = value
    return first


def both_pinned_weights(asset):
    from safetensors import safe_open

    first = pinned_weights(asset)
    if hashlib.sha256(Path(asset).read_bytes()).hexdigest() != ASSET_HASH:
        raise ValueError("pinned Clef head archive identity differs")
    prefix = "evidence_layers.1"
    with safe_open(asset, framework="pt") as archive:
        for stem in ("query_norm", "memory_norm", "feedforward_norm"):
            for suffix in ("weight", "bias"):
                name = f"{prefix}.{stem}.{suffix}"
                first[name] = archive.get_tensor(name).float().numpy()
        for stem in ("attention.out_proj", "feedforward.0", "feedforward.3"):
            for suffix in ("weight", "bias"):
                name = f"{prefix}.{stem}.{suffix}"
                first[name] = archive.get_tensor(name).float().numpy()
        packed = archive.get_tensor(f"{prefix}.attention.in_proj_weight").float().numpy()
        bias = archive.get_tensor(f"{prefix}.attention.in_proj_bias").float().numpy()
    width = packed.shape[1]
    for index, stem in enumerate(("q", "k", "v")):
        start, end = index * width, (index + 1) * width
        first[f"{prefix}.attention.{stem}_proj.weight"] = packed[start:end].copy()
        first[f"{prefix}.attention.{stem}_proj.bias"] = bias[start:end].copy()
    return first


class ClefFlashCompleteEvidenceTest(unittest.TestCase):
    def test_circuit_regenerates_and_rebinds_previous_output(self):
        self.assertEqual(OUTPUT.read_text(), json.dumps(build_circuit(), indent=2) + "\n")
        graph = build_circuit(channels=8, width=8, heads=2, feedforward=16)
        second = graph["block_types"]["evidence1"]
        self.assertEqual(second["header"][0]["graph_slots"]["inputs"],
                         {"input": "evidence0_output"})
        residual = next(op for op in second["body"]["ops"]
                        if op["id"] == "evidence1_attention_residual")
        self.assertEqual(residual["graph_slots"]["inputs"],
                         {"left": "evidence0_output", "right": "evidence1_attn_out"})
        self.assertEqual(second["header"][1]["graph_slots"]["inputs"],
                         {"input": "memory"})

    def test_connected_synthetic_recovery_and_independent_torch(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch comparison unavailable")
        graph = build_circuit(channels=8, width=8, heads=2, feedforward=16)
        weights = both_synthetic_weights(8, 8, 16, 16)
        with tempfile.TemporaryDirectory() as temp:
            arena, execute, view, call_ir, _loaded = compile_case(Path(temp), graph, weights)
            self.assertEqual(call_ir["errors"], [])
            self.assertEqual(len(call_ir["operations"]), 41)
            hidden_a = np.random.default_rng(301).normal(0, 0.5, (6, 8)).astype("f4")
            hidden_b = np.random.default_rng(302).normal(0, 0.5, (6, 8)).astype("f4")
            prepare(view, hidden_a)
            self.assertEqual(execute(arena, len(arena)), 0)
            first = torch_layer(view("option_queries").copy(),
                                view("memory").copy(), weights, 2)
            second = torch_layer(first["evidence0_output"].numpy(),
                                 view("memory").copy(), weights, 2,
                                 prefix="evidence_layers.1")
            for name, expected in second.items():
                key = name.replace("evidence0_", "evidence1_", 1)
                np.testing.assert_allclose(view(key), expected.numpy(),
                                           atol=3e-6, rtol=0, err_msg=key)
            baseline = view("evidence1_output").copy()
            prepare(view, hidden_b)
            self.assertEqual(execute(arena, len(arena)), 0)
            self.assertFalse(np.array_equal(view("evidence1_output"), baseline))
            prepare(view, hidden_a)
            self.assertEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("evidence1_output"), baseline)
            view("option_counts")[:] = [1, 1]
            view("evidence1_output")[:] = -57
            self.assertNotEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("evidence1_output"), -57)
            view("option_counts")[:] = [2, 1]
            self.assertEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("evidence1_output"), baseline)
            layout = json.loads((Path(temp) / "layout.json").read_text())
            entry = next(row for row in layout["memory"]["weights"]["entries"]
                         if row["name"] == "evidence_layers.1.attention.q_proj.weight")
            weight = np.ndarray((8, 8), dtype="f4", buffer=arena,
                                offset=entry["abs_offset"])
            saved = weight.copy()
            weight[:] = 0
            self.assertEqual(execute(arena, len(arena)), 0)
            self.assertFalse(np.array_equal(view("evidence1_output"), baseline))
            weight[:] = saved
            self.assertEqual(execute(arena, len(arena)), 0)
            np.testing.assert_array_equal(view("evidence1_output"), baseline)

    def test_eight_token_geometry_matches_independent_torch(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch comparison unavailable")
        graph = build_circuit(tokens=8, channels=8, width=8,
                              heads=2, feedforward=16)
        weights = both_synthetic_weights(8, 8, 16, 16)
        with tempfile.TemporaryDirectory() as temp:
            arena, execute, view, _call_ir, _loaded = compile_case(
                Path(temp), graph, weights)
            prepare(view, np.random.default_rng(803).normal(0, 0.5, (8, 8)).astype("f4"))
            self.assertEqual(execute(arena, len(arena)), 0)
            first = torch_layer(view("option_queries").copy(),
                                view("memory").copy(), weights, 2)
            second = torch_layer(first["evidence0_output"].numpy(),
                                 view("memory").copy(), weights, 2,
                                 prefix="evidence_layers.1")
            np.testing.assert_allclose(view("evidence1_output"),
                                       second["evidence0_output"].numpy(),
                                       atol=3e-6, rtol=0)

    def test_direct_pinned_reference_both_layers(self):
        asset = os.environ.get("CKE_CLEF_FLASH_HEAD_PATH")
        source = os.environ.get("CKE_CLEF_FLASH_REFERENCE_SOURCE")
        if not asset or not source:
            self.skipTest("pinned Clef head weights/reference source unavailable")
        try:
            import torch
            import safetensors  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch/safetensors comparison unavailable")
        identity = json.loads((ROOT / "tests/fixtures/decision/clef_flash/identity.json").read_text())
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != identity["joint_schema_model_sha256"]:
            self.fail("pinned Clef reference source identity differs")
        spec = importlib.util.spec_from_file_location("clef_pinned_reference_two_layers", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        weights = both_pinned_weights(asset)
        layers = []
        for index in range(2):
            prefix = f"evidence_layers.{index}"
            layer = module.EvidenceRoutingLayer(
                width=1024, heads=16, feedforward=4096, dropout=0.0)
            state = {}
            for name, value in weights.items():
                if name.startswith(prefix + ".") and ".attention.q_proj." not in name and ".attention.k_proj." not in name and ".attention.v_proj." not in name:
                    state[name[len(prefix) + 1:]] = torch.from_numpy(value)
            for suffix in ("weight", "bias"):
                state[f"attention.in_proj_{suffix}"] = torch.from_numpy(np.concatenate([
                    weights[f"{prefix}.attention.{stem}_proj.{suffix}"]
                    for stem in ("q", "k", "v")]))
            layer.load_state_dict(state, strict=True)
            layer.eval()
            layers.append(layer)
        with tempfile.TemporaryDirectory() as temp:
            arena, execute, view, _call_ir, _loaded = compile_case(
                Path(temp), build_circuit(), weights)
            prepare(view, np.random.default_rng(403).normal(0, 0.2, (6, 4096)).astype("f4"))
            self.assertEqual(execute(arena, len(arena)), 0)
            with torch.no_grad():
                query = torch.from_numpy(view("option_queries").copy()).unsqueeze(0)
                memory = torch.from_numpy(view("memory").copy()).unsqueeze(0)
                first = layers[0](query, memory)
                second = layers[1](first, memory)[0]
            np.testing.assert_allclose(view("evidence0_output"), first[0].numpy(),
                                       atol=2e-6, rtol=0)
            np.testing.assert_allclose(view("evidence1_output"), second.numpy(),
                                       atol=4e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
