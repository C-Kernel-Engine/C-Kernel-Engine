#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V7_BUILD_PATH = ROOT / "version" / "v7" / "scripts" / "build_ir_v7.py"
V8_BUILD_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"


def _load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_ir_v7 = _load_module("build_ir_v7_for_v8_scaffold_tests", V7_BUILD_PATH)
build_ir_v8 = _load_module("build_ir_v8_for_tests", V8_BUILD_PATH)


def _normalized_template_doc(doc: dict) -> dict:
    normalized = json.loads(json.dumps(doc))
    normalized.pop("required_contracts", None)
    normalized.pop("required_numerical_contracts", None)
    # v8 makes the seed's implicit projection source explicit. Validate these
    # declarations separately before comparing the inherited graph topology.
    normalized.pop("projection_inputs", None)
    contract = normalized.get("contract")
    if isinstance(contract, dict):
        # Runtime preferences moved out of circuit flags in v8. They are not
        # graph topology and therefore are excluded from the v7 seed check.
        contract.pop("runtime_defaults", None)
    kernels = normalized.get("kernels")
    if isinstance(kernels, dict):
        for key in list(kernels):
            if key.startswith("attn") or key == "rope_qk":
                kernels.pop(key)
        if not kernels:
            normalized.pop("kernels", None)
    flags = normalized.get("flags")
    if isinstance(flags, dict):
        for key in build_ir_v8._FORBIDDEN_TEMPLATE_FLAG_KEYS:
            flags.pop(key, None)
    attention_contract = normalized.get("contract", {}).get("attention_contract")
    if isinstance(attention_contract, dict):
        attention_contract.pop("train_runtime_contract", None)
        for key in (
            "layer_policy_config_key",
            "layer_kind_config_key",
            "state_policy_config_key",
            "attention_policy_config_key",
            "recurrent_policy_config_key",
            "kv_policy_config_key",
        ):
            attention_contract.pop(key, None)
    return normalized


class BuildIrV8ScaffoldTests(unittest.TestCase):
    def test_v8_template_root_is_isolated(self) -> None:
        self.assertEqual(build_ir_v8.V8_ROOT.name, "v8")
        self.assertTrue((build_ir_v8.V8_ROOT / "circuits" / "qwen3.json").exists())

    def test_v8_templates_match_current_v7_seed_after_runtime_policy_extraction(self) -> None:
        for name in ("gemma3", "llama", "qwen2", "qwen3", "qwen35"):
            with self.subTest(template=name):
                v7_doc = json.loads((ROOT / "version" / "v7" / "templates" / f"{name}.json").read_text(encoding="utf-8"))
                v8_doc = json.loads((ROOT / "version" / "v8" / "circuits" / f"{name}.json").read_text(encoding="utf-8"))
                projections = {"q_proj", "k_proj", "v_proj"}
                if name == "qwen35":
                    projections.remove("q_proj")
                    projections.update({
                        "q_gate_proj", "recurrent_qkv_proj", "recurrent_gate_proj",
                        "recurrent_alpha_proj", "recurrent_beta_proj",
                    })
                self.assertEqual(
                    v8_doc.get("projection_inputs"),
                    {op: {"x": "main_stream_q8"} for op in projections},
                )
                self.assertEqual(_normalized_template_doc(v8_doc), _normalized_template_doc(v7_doc))

    def test_v8_seeded_templates_do_not_embed_runtime_policy_flags(self) -> None:
        for name in ("gemma3", "llama", "qwen2", "qwen3", "qwen35"):
            with self.subTest(template=name):
                v8_doc = json.loads((ROOT / "version" / "v8" / "circuits" / f"{name}.json").read_text(encoding="utf-8"))
                flags = v8_doc.get("flags", {})
                self.assertIsInstance(flags, dict)
                for key in build_ir_v8._FORBIDDEN_TEMPLATE_FLAG_KEYS:
                    self.assertNotIn(key, flags)

    def test_v7_seed_parser_filters_conditional_graph_branches(self) -> None:
        section = [
            {"op": "dense", "when": {"config_key": "mode", "not_equals": "moe"}},
            {"op": "routed", "when": {"config_key": "mode", "equals": "moe"}},
        ]
        self.assertEqual(build_ir_v7._extract_template_ops(section, {"mode": "dense"}), ["dense"])
        self.assertEqual(build_ir_v7._extract_template_ops(section, {"mode": "moe"}), ["routed"])

        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            build_ir_v7._extract_template_ops(
                [{"op": "invalid", "when": {"config_key": "mode"}}],
                {"mode": "dense"},
            )

    def test_v8_rope_resolution_requires_explicit_compatible_kernel(self) -> None:
        cases = [
            ({}, {"rope_qk": "rope_forward_qk"}),
            ({"rope_layout": "split"}, {"rope_qk": "rope_forward_qk"}),
            ({"rope_layout": "interleaved"}, {"rope_qk": "rope_forward_qk_pairwise"}),
            ({"rope_layout": "split"}, {"rope_qk": "rope_forward_qk_custom"}),
        ]
        for config, kernels in cases:
            with self.subTest(config=config, kernels=kernels):
                self.assertEqual(
                    build_ir_v8._resolve_rope_qk_kernel(config, kernels),
                    kernels["rope_qk"],
                )

        for config, kernels in [
            ({"rope_layout": "interleaved"}, {}),
            ({"rope_layout": "split"}, {}),
            ({"rope_layout": "interleaved"}, {"rope_qk": "rope_forward_qk"}),
        ]:
            with self.subTest(config=config, kernels=kernels):
                with self.assertRaises(RuntimeError):
                    build_ir_v8._resolve_rope_qk_kernel(config, kernels)

    def test_v8_uses_local_kernel_registry(self) -> None:
        registry = build_ir_v8.load_kernel_registry()
        self.assertIsInstance(registry, dict)
        self.assertTrue(registry)
        self.assertEqual(build_ir_v8.V8_ROOT.name, "v8")
        self.assertTrue((build_ir_v8.V8_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json").exists())

    def test_circuit_owned_activation_bindings_keep_two_streams_distinct(self) -> None:
        graph = {
            "inputs": {
                "hidden": {"slot": "main_stream", "dtype": "fp32"},
                "aux": {"slot": "routed_free_stream", "dtype": "fp32"},
            },
            "outputs": {
                "next_main": {"slot": "main_stream", "dtype": "fp32"},
                "next_aux": {"slot": "routed_free_stream", "dtype": "fp32"},
            },
        }
        assignments = build_ir_v8.plan_memory(
            [{"idx": 0, "op": "future_architecture_composite", "layer": 1, "dataflow": graph}],
            slot_bindings={"routed_free_stream": "layer_output"},
        )[0]
        self.assertEqual(assignments["inputs"]["hidden"]["buffer"], "A_EMBEDDED_INPUT")
        self.assertEqual(assignments["inputs"]["aux"]["buffer"], "layer_output")
        self.assertEqual(assignments["outputs"]["next_aux"]["buffer"], "layer_output")

    def test_explicit_projection_input_slot_overrides_legacy_stream_default(self) -> None:
        graph = {
            "inputs": {
                "x": {"slot": "normalized_stream", "dtype": "fp32"},
            },
            "outputs": {
                "y": {"slot": "q_scratch", "dtype": "fp32"},
            },
        }
        assignment = build_ir_v8.plan_memory(
            [{"idx": 0, "op": "q_proj", "layer": 0, "dataflow": graph}],
            slot_bindings={"normalized_stream": "layer_input"},
        )[0]

        self.assertEqual(assignment["inputs"]["x"]["buffer"], "layer_input")

    def test_activation_binding_contract_rejects_empty_names(self) -> None:
        self.assertEqual(
            build_ir_v8._template_activation_bindings(
                {"activation_bindings": {"routed_free_stream": "layer_output"}}
            ),
            {"routed_free_stream": "layer_output"},
        )
        with self.assertRaisesRegex(RuntimeError, "activation_bindings"):
            build_ir_v8._template_activation_bindings(
                {"activation_bindings": {"routed_free_stream": ""}}
            )

    def test_circuit_owned_activation_buffers_resolve_generic_shape_expressions(self) -> None:
        specs = build_ir_v8._template_activation_buffer_specs(
            {
                "activation_buffers": {
                    "collected": {
                        "shape": [
                            {"config": "rows"},
                            {"mul": [{"config": "width"}, {"config": "slices"}]},
                        ]
                    }
                }
            },
            {"rows": 5, "width": 7, "slices": 3},
        )
        self.assertEqual(specs["collected"]["shape"], "[5, 21]")
        self.assertEqual(specs["collected"]["size"], 5 * 21 * 4)
        self.assertEqual(specs["collected"]["dtype"], "fp32")

    def test_circuit_owned_activation_buffers_reject_missing_or_zero_extents(self) -> None:
        template = {"activation_buffers": {"scratch": {"shape": [{"config": "rows"}]}}}
        with self.assertRaisesRegex(RuntimeError, "missing config key"):
            build_ir_v8._template_activation_buffer_specs(template, {})
        with self.assertRaisesRegex(RuntimeError, "non-positive extent"):
            build_ir_v8._template_activation_buffer_specs(template, {"rows": 0})

    def test_circuit_owned_activation_buffers_reject_unknown_dtype(self) -> None:
        template = {
            "activation_buffers": {
                "scratch": {"shape": [4, 8], "dtype": "mystery_float"}
            }
        }
        with self.assertRaisesRegex(RuntimeError, "unsupported storage type"):
            build_ir_v8._template_activation_buffer_specs(template, {})


if __name__ == "__main__":
    unittest.main()
