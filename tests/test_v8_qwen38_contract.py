#!/usr/bin/env python3
import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import convert_gguf_to_bump_v8 as converter  # type: ignore


class Qwen38ContractTests(unittest.TestCase):
    def test_exact_artifact_metadata_selects_qwen38(self) -> None:
        metadata = {
            "general.architecture": "qwen35",
            "general.basename": "Qwen3.8-27B",
        }
        contract = converter.gguf_ck_artifact_contract("qwen35", metadata)
        self.assertEqual(contract["artifact_variant"], "qwen38-27b-unsloth")
        self.assertEqual(contract["runtime_arch"], "qwen38")
        self.assertEqual(converter.gguf_ck_template_arch("qwen35", metadata), "qwen38")

    def test_other_qwen35_artifacts_remain_qwen35(self) -> None:
        metadata = {
            "general.architecture": "qwen35",
            "general.basename": "Qwen3.6-27B",
        }
        contract = converter.gguf_ck_artifact_contract("qwen35", metadata)
        self.assertNotIn("artifact_variant", contract)
        self.assertEqual(converter.gguf_ck_template_arch("qwen35", metadata), "qwen35")

    def test_variant_match_metadata_is_retained_by_gguf_reader(self) -> None:
        self.assertIn("general.basename", converter.gguf_ck_artifact_match_keys())
        self.assertIn("general.size_label", converter.gguf_ck_artifact_match_keys())

        model_map = converter.load_gguf_ck_map()["architectures"]["qwen35"]
        variants = {row["id"]: row for row in model_map["artifact_variants"]}
        self.assertEqual(variants["qwen38-27b-unsloth"]["template"], "qwen38")
        self.assertEqual(variants["qwen38-27b-standard"]["runtime_arch"], "qwen38")

    def test_standard_qwen38_gguf_identity_selects_qwen38(self) -> None:
        metadata = {
            "general.architecture": "qwen35",
            "general.basename": "Qwen3.8",
            "general.size_label": "27B",
        }
        contract = converter.gguf_ck_artifact_contract("qwen35", metadata)
        self.assertEqual(contract["artifact_variant"], "qwen38-27b-standard")
        self.assertEqual(converter.gguf_ck_template_arch("qwen35", metadata), "qwen38")

    def test_ambiguous_artifact_variants_fail_closed(self) -> None:
        original = converter._GGUF_CK_MAP_CACHE
        model_map = copy.deepcopy(converter.load_gguf_ck_map())
        duplicate = copy.deepcopy(model_map["architectures"]["qwen35"]["artifact_variants"][0])
        duplicate["id"] = "duplicate"
        model_map["architectures"]["qwen35"]["artifact_variants"].append(duplicate)
        try:
            converter._GGUF_CK_MAP_CACHE = model_map
            with self.assertRaisesRegex(converter.GGUFError, "multiple variants"):
                converter.gguf_ck_artifact_contract(
                    "qwen35", {"general.basename": "Qwen3.8-27B"}
                )
        finally:
            converter._GGUF_CK_MAP_CACHE = original

    def test_qwen38_circuit_is_dense_and_model_owned(self) -> None:
        path = ROOT / "version" / "v8" / "circuits" / "qwen38.json"
        circuit = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(circuit["name"], "qwen38")
        self.assertEqual(circuit["family"], "qwen38")
        self.assertEqual(circuit["flags"]["mlp_topology"], "dense_swiglu")

        serialized = json.dumps(circuit)
        self.assertNotIn("qwen35moe", serialized)
        self.assertNotIn("moe_router", serialized)
        self.assertNotIn("moe_swiglu_expert_mlp", serialized)

        body = circuit["block_types"]["decoder"]["body"]["ops_by_kind"]
        dense_tail = [
            "post_attention_norm",
            "mlp_gate_up",
            "silu_mul",
            "mlp_down",
            "residual_add",
        ]
        self.assertEqual(body["recurrent"][-5:], dense_tail)
        self.assertEqual(body["full_attention"][-5:], dense_tail)

    def test_circuit_runtime_defaults_are_applied_generically(self) -> None:
        circuit = converter.load_template_for_arch("qwen38")
        config = converter.apply_circuit_runtime_defaults({}, circuit)
        self.assertTrue(config["prefer_q8_0_contract"])
        self.assertEqual(
            config["activation_preference_by_op"]["recurrent_gate_proj"],
            "q8_k",
        )


if __name__ == "__main__":
    unittest.main()
