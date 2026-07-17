#!/usr/bin/env python3
"""Structural guards for the production-order Qwen3-VL parity lane.

Leaf kernels are executed by ``make test-qwen3vl-methodical-parity``.  This
module ensures that the artifact-backed X-ray lane observes every semantic
edge in the same order as the circuit instead of silently skipping a stage.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CIRCUIT = ROOT / "version/v8/circuits/qwen3_vl_vision.json"
PROFILE = ROOT / "version/v8/parity_profiles/qwen3vl_llamacpp_q8_v1.json"
CONTRACTS = ROOT / "version/v8/contracts/numerical_execution.json"
KERNEL_MAPS = ROOT / "version/v8/kernel_maps"

EXPECTED_LAYER_ORDER = [
    "vision.frontend.patch_bias.output",
    "vision.frontend.position.output",
    "vision.layer.{layer}.norm1.output",
    "vision.layer.{layer}.q.pre_rope",
    "vision.layer.{layer}.k.pre_rope",
    "vision.layer.{layer}.v.output",
    "vision.layer.{layer}.q.post_rope",
    "vision.layer.{layer}.k.post_rope",
    "vision.layer.{layer}.attention.output",
    "vision.layer.{layer}.out_proj.output",
    "vision.layer.{layer}.residual1.output",
    "vision.layer.{layer}.norm2.output",
    "vision.layer.{layer}.mlp.up",
    "vision.layer.{layer}.mlp.activation",
    "vision.layer.{layer}.mlp.down",
    "vision.layer.{layer}.output",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Qwen3VLMethodicalParityTests(unittest.TestCase):
    def test_xray_profile_covers_the_complete_layer_in_production_order(self) -> None:
        circuit = load(CIRCUIT)
        profile = load(PROFILE)
        self.assertEqual(profile["checkpoint_order"], EXPECTED_LAYER_ORDER)

        circuit_ids = {
            checkpoint["id"]
            for edge in circuit["semantic_checkpoints"]["exports"].values()
            for checkpoint in edge["checkpoints"]
        }
        self.assertEqual(set(EXPECTED_LAYER_ORDER) - circuit_ids, set())
        self.assertEqual(
            set(EXPECTED_LAYER_ORDER) - set(profile["backend_mappings"]),
            set(),
        )

    def test_every_profile_edge_declares_layout_and_backend_tensor(self) -> None:
        profile = load(PROFILE)
        for checkpoint_id in EXPECTED_LAYER_ORDER:
            mapping = profile["backend_mappings"][checkpoint_id]
            with self.subTest(checkpoint=checkpoint_id):
                self.assertTrue(mapping["producer"])
                self.assertIn(mapping["logical_layout"], {"token_major", "head_major"})
                self.assertGreaterEqual(len(mapping["axis_names"]), 2)
                self.assertTrue(mapping["capture_tensor"])
                self.assertTrue(mapping["result_tensor"])

    def test_position_policies_resolve_to_distinct_exact_contracts(self) -> None:
        circuit = load(CIRCUIT)
        required = circuit["required_numerical_contracts"]
        half = required["vision.frontend.position.fp32"]
        aligned = required["vision.frontend.position.fp32.align_corners"]
        self.assertEqual(
            half["selector"]["config_not_equals"]["position_interpolation_policy"],
            "align_corners_bilinear",
        )
        self.assertEqual(
            aligned["selector"]["config_equals"]["position_interpolation_policy"],
            "align_corners_bilinear",
        )
        self.assertNotEqual(
            half["phases"]["prefill"]["contract_id"],
            aligned["phases"]["prefill"]["contract_id"],
        )

        contracts = load(CONTRACTS)["contracts"]
        contract_id = aligned["phases"]["prefill"]["contract_id"]
        self.assertEqual(
            contracts[contract_id]["spatial_transform"]["coordinate_transform"],
            "align_corners",
        )
        kernel = load(KERNEL_MAPS / "position_embeddings_add_tiled_2d_align_corners.json")
        capabilities = {row["contract_id"] for row in kernel["numerical_capabilities"]}
        self.assertIn(contract_id, capabilities)

    def test_q8_projection_boundaries_are_declared_by_the_circuit(self) -> None:
        circuit = load(CIRCUIT)
        defaults = circuit["contract"]["runtime_defaults"]
        self.assertTrue(defaults["prefer_q8_0_contract"])
        self.assertEqual(defaults["activation_preference_by_op"]["out_proj"], "q8_0")
        body = circuit["block_types"]["vision_encoder"]["body"]["ops"]
        ops = [op["op"] for op in body]
        for op in ("qkv_packed_proj", "out_proj", "mlp_up"):
            self.assertIn(op, ops)


if __name__ == "__main__":
    unittest.main(verbosity=2)
