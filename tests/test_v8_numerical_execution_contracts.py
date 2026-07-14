#!/usr/bin/env python3
"""Fail-closed tests for generic v8 numerical execution contracts."""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "resolve_numerical_execution_contracts_v8.py"
SPEC = importlib.util.spec_from_file_location("resolve_numerical_execution_contracts_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)

PLANNER_SCRIPT = ROOT / "version" / "v8" / "scripts" / "plan_parity_bisection_v8.py"
PLANNER_SPEC = importlib.util.spec_from_file_location("plan_parity_bisection_v8", PLANNER_SCRIPT)
assert PLANNER_SPEC is not None and PLANNER_SPEC.loader is not None
planner = importlib.util.module_from_spec(PLANNER_SPEC)
PLANNER_SPEC.loader.exec_module(planner)


CONTRACT_ID = "bf16_weight_bf16_input_fp32_dot_fp32_output"


def circuit(validation: str = "observed"):
    return {
        "name": "contract_test",
        "required_numerical_contracts": {
            "gemm": {
                "op": "gemm",
                "template_ops": ["mlp_up"],
                "phases": {
                    "prefill": {
                        "contract_id": CONTRACT_ID,
                        "validation": validation,
                        "evidence": "tests/test_v8_numerical_execution_contracts.py",
                    }
                },
                "checkpoint": {
                    "id": "vision.layer.0.mlp.up",
                    "producer": "mlp_up",
                    "logical_layout": "token_major",
                    "axis_names": ["token", "channel"],
                },
            }
        },
    }


def mrope_circuit(contract_id: str):
    return {
        "name": "vision_mrope_contract_test",
        "required_numerical_contracts": {
            "vision_mrope": {
                "op": "rope",
                "template_ops": ["rope_qk"],
                "phases": {
                    "prefill": {
                        "contract_id": contract_id,
                        "validation": "validated",
                        "evidence": "unittest/test_vision.py::test_mrope_qk_vision_storage_matrix",
                    }
                },
                "checkpoint": {
                    "id": "vision.layer.0.q.post_rope",
                    "producer": "vision_mrope",
                    "logical_layout": "head_major",
                    "axis_names": ["head", "token", "channel"],
                },
            }
        },
    }


class NumericalExecutionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts = resolver.load_json(resolver.DEFAULT_CONTRACTS)
        cls.kernels = resolver.load_kernel_capabilities(contracts=cls.contracts)

    def test_exact_bf16_kernel_resolution_preserves_semantics(self):
        plan = resolver.resolve_contract(
            circuit(), self.contracts, self.kernels, "gemm", "prefill"
        )
        self.assertEqual(plan["kernel"]["id"], "gemm_nt_bf16")
        self.assertEqual(plan["kernel"]["function"], "gemm_nt_bf16")
        self.assertEqual(plan["contract"]["semantics"]["compute"]["input"], "bf16_rne")
        self.assertEqual(plan["contract"]["semantics"]["reduction"]["order"], "ascending_k")
        self.assertFalse(
            plan["contract"]["semantics"]["threading"]["thread_count_changes_arithmetic_order"]
        )
        self.assertEqual(plan["checkpoint"]["axis_names"], ["token", "channel"])

    def test_bf16_position_contract_resolves_exact_kernel(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen3_vl_vision.json"
        )
        plan = resolver.resolve_contract(
            circuit_doc,
            self.contracts,
            self.kernels,
            "vision.frontend.position",
            "prefill",
            mode="production",
        )
        self.assertEqual(
            plan["contract"]["id"],
            "bf16_tiled_2d_align_corners_rne_residual",
        )
        self.assertEqual(
            plan["kernel"]["id"],
            "position_embeddings_add_tiled_2d_align_corners_bf16",
        )
        self.assertEqual(
            plan["kernel"]["function"],
            "position_embeddings_add_tiled_2d_align_corners_bf16",
        )

    def test_fp32_position_contract_resolves_exact_kernel_and_evaluation_order(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen3_vl_vision.json"
        )
        plan = resolver.resolve_contract(
            circuit_doc,
            self.contracts,
            self.kernels,
            "vision.frontend.position.fp32",
            "prefill",
            mode="production",
        )
        self.assertEqual(
            plan["contract"]["id"],
            "fp32_tiled_2d_antialias_half_pixel_contracted",
        )
        self.assertEqual(plan["kernel"]["id"], "position_embeddings_add_tiled_2d")
        self.assertEqual(plan["kernel"]["function"], "position_embeddings_add_tiled_2d")
        spatial = plan["contract"]["semantics"]["spatial_transform"]
        self.assertEqual(spatial["evaluation_order"], "channel_row_column")
        self.assertEqual(spatial["contraction"], "enabled")

    def test_fp32_layernorm_contract_resolves_exact_kernel_and_reduction(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen3_vl_vision.json"
        )
        plan = resolver.resolve_contract(
            circuit_doc,
            self.contracts,
            self.kernels,
            "vision.layer.layernorm.fp32",
            "prefill",
            mode="production",
        )
        self.assertEqual(plan["kernel"]["id"], "layernorm_fp32_exact")
        self.assertEqual(
            plan["kernel"]["function"], "layernorm_naive_serial_matched_precision"
        )
        semantics = plan["contract"]["semantics"]
        self.assertEqual(semantics["compute"]["contraction"], "enabled")
        self.assertEqual(semantics["reduction"]["order"], "contract_defined_chunks")
        self.assertEqual(semantics["reduction"]["merge_order"], "ascending_chunk")

    def test_mrope_storage_contract_matrix_resolves_exact_functions(self):
        expected = {
            "vision_mrope_fp32_input_fp32_compute_fp32_output": ("mrope_qk_vision", "mrope_qk_vision"),
            "vision_mrope_fp32_input_fp32_compute_bf16_output": ("mrope_qk_vision_bf16_storage", "mrope_qk_vision_bf16_storage"),
            "vision_mrope_fp32_input_fp32_compute_fp16_output": ("mrope_qk_vision_fp16_storage", "mrope_qk_vision_fp16_storage"),
        }
        for contract_id, (kernel_id, function) in expected.items():
            with self.subTest(contract_id=contract_id):
                plan = resolver.resolve_contract(
                    mrope_circuit(contract_id), self.contracts, self.kernels, "vision_mrope", "prefill", mode="production"
                )
                self.assertEqual(plan["kernel"]["id"], kernel_id)
                self.assertEqual(plan["kernel"]["function"], function)
                self.assertEqual(plan["contract"]["semantics"]["reduction"]["kind"], "none")

    def test_qwen3vl_circuit_requests_bf16_mrope_storage(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen3_vl_vision.json"
        )
        plan = resolver.resolve_contract(
            circuit_doc,
            self.contracts,
            self.kernels,
            "vision.layer.mrope",
            "prefill",
            mode="production",
        )
        self.assertEqual(
            plan["contract"]["id"],
            "vision_mrope_fp32_input_fp32_compute_bf16_output",
        )
        self.assertEqual(plan["kernel"]["id"], "mrope_qk_vision_bf16_storage")
        self.assertEqual(plan["kernel"]["function"], "mrope_qk_vision_bf16_storage")

    def test_qwen35_circuit_resolves_partial_width_text_mrope(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        plan = resolver.resolve_contract(
            circuit_doc,
            self.contracts,
            self.kernels,
            "decoder.mrope",
            "prefill",
            mode="production",
        )
        self.assertEqual(
            plan["contract"]["id"],
            "text_mrope_fp32_input_fp32_compute_fp32_output",
        )
        self.assertEqual(plan["kernel"]["id"], "mrope_qk_text")
        self.assertEqual(plan["kernel"]["function"], "mrope_qk_text")
        position = plan["contract"]["semantics"]["position_transform"]
        self.assertEqual(position["rotary_width"], "configured_rotary_dim")
        self.assertEqual(position["position_rank"], 4)
        self.assertEqual(plan["template_ops"], ["rope_qk"])

    def test_unsupported_mrope_storage_contract_hard_fails(self):
        doc = mrope_circuit("vision_mrope_fp64_input_fp64_compute_fp64_output")
        with self.assertRaisesRegex(resolver.ContractError, "unknown requested contract"):
            resolver.resolve_contract(doc, self.contracts, self.kernels, "vision_mrope", "prefill")
    def test_qwen3vl_bf16_boundary_contracts_resolve_exact_functions(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen3_vl_vision.json"
        )
        expected = {
            "vision.layer.layernorm": "layernorm_naive_serial_bf16_storage",
            "vision.layer.qkv_projection": "gemm_nt_bf16_native_bf16_storage",
            "vision.layer.mlp_projection": "gemm_nt_bf16_native_bf16_storage",
            "vision.layer.mlp_activation": "gelu_pytorch_tanh_bf16_storage",
            "vision.layer.attention": "attention_forward_full_head_major_gqa_sdpa_bf16_storage",
            "vision.layer.out_projection": "gemm_nt_bf16_native_bf16_storage",
            "vision.layer.residual": "ck_residual_add_token_major_bf16_storage",
            "vision.projector.projection": "gemm_nt_bf16_amx_bf16_storage",
        }
        for operation, function in expected.items():
            with self.subTest(operation=operation):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    operation,
                    "prefill",
                    mode="production",
                )
                self.assertEqual(plan["kernel"]["function"], function)
                self.assertEqual(plan["contract"]["status"], "validated")
                if operation == "vision.layer.attention":
                    self.assertEqual(
                        plan["contract"]["semantics"]["threading"]["work_partition"],
                        "independent_heads",
                    )
                    self.assertEqual(plan["implementation"]["threading"]["runtime"], "ck_threadpool")
                    self.assertEqual(
                        plan["implementation"]["threading"]["dispatch"],
                        ["ck_threadpool_dispatch_n"],
                    )

    def test_zero_provider_is_hard_failure(self):
        kernels = copy.deepcopy(self.kernels)
        kernels["kernels"] = {}
        with self.assertRaisesRegex(resolver.ContractError, "resolved to 0 kernels"):
            resolver.resolve_contract(circuit(), self.contracts, kernels, "gemm", "prefill")

    def test_multiple_providers_are_hard_failure(self):
        kernels = copy.deepcopy(self.kernels)
        duplicate = copy.deepcopy(kernels["kernels"]["gemm_nt_bf16"])
        duplicate["id"] = "gemm_nt_bf16_duplicate"
        kernels["kernels"][duplicate["id"]] = duplicate
        with self.assertRaisesRegex(resolver.ContractError, "resolved to 2 kernels"):
            resolver.resolve_contract(circuit(), self.contracts, kernels, "gemm", "prefill")

    def test_decode_cannot_silently_use_prefill_provider(self):
        doc = circuit()
        doc["required_numerical_contracts"]["gemm"]["phases"]["decode"] = copy.deepcopy(
            doc["required_numerical_contracts"]["gemm"]["phases"]["prefill"]
        )
        with self.assertRaisesRegex(resolver.ContractError, "resolved to 0 kernels"):
            resolver.resolve_contract(doc, self.contracts, self.kernels, "gemm", "decode")

    def test_production_rejects_observed_contract(self):
        with self.assertRaisesRegex(resolver.ContractError, "production resolution uses unvalidated"):
            resolver.resolve_contract(
                circuit(), self.contracts, self.kernels, "gemm", "prefill", mode="production"
            )

    def test_arithmetic_capability_mismatch_is_hard_failure(self):
        capability = copy.deepcopy(
            self.kernels["kernels"]["gemm_nt_bf16"]["capabilities"][0]
        )
        capability["arithmetic"]["thread_count_changes_arithmetic_order"] = True
        with self.assertRaisesRegex(resolver.ContractError, "arithmetic metadata disagrees"):
            resolver._validate_capability_against_contract(
                "gemm_nt_bf16", capability, self.contracts["contracts"][CONTRACT_ID]
            )

    def test_multisection_rope_sections_must_mean_axis_selection(self):
        contracts = copy.deepcopy(self.contracts)
        base = copy.deepcopy(contracts["contracts"][CONTRACT_ID])
        base["position_transform"] = {
            "pairing": "multi_section",
            "rotary_width": "mrope_n_dims",
            "head_width": "head_dim",
            "position_rank": 3,
            "axis_order": ["temporal", "height", "width"],
            "section_interpretation": "contiguous_widths",
            "frequency_compute": "fp32",
            "intermediate_compute": "fp32",
            "rounding_points": ["output_store"],
            "threading": "independent_tokens",
        }
        contracts["contracts"]["qwen_mrope_invalid"] = base
        with self.assertRaisesRegex(resolver.ContractError, "redefines rotary width"):
            resolver.validate_contract_registry(contracts)

    def test_multisection_rope_accepts_interleaved_axis_selection(self):
        contracts = copy.deepcopy(self.contracts)
        base = copy.deepcopy(contracts["contracts"][CONTRACT_ID])
        base["operator_family"] = "text_mrope"
        base["position_transform"] = {
            "pairing": "multi_section",
            "rotary_width": "mrope_n_dims",
            "head_width": "head_dim",
            "position_rank": 4,
            "axis_order": ["temporal", "height", "width", "reserved"],
            "section_interpretation": "interleaved_axis_selection",
            "frequency_compute": "fp32",
            "intermediate_compute": "fp32",
            "rounding_points": [],
            "threading": "serial",
        }
        contracts["contracts"]["qwen_text_imrope"] = base
        resolver.validate_contract_registry(contracts)

    def test_mrope_width_must_match_full_rotary_width(self):
        contracts = copy.deepcopy(self.contracts)
        base = copy.deepcopy(contracts["contracts"][CONTRACT_ID])
        base["position_transform"] = {
            "pairing": "multi_section",
            "rotary_width": "mrope_n_dims",
            "head_width": "head_dim",
            "rotary_width_value": 128,
            "head_width_value": 128,
            "mrope_n_dims_value": 64,
            "position_rank": 3,
            "axis_order": ["temporal", "height", "width"],
            "section_interpretation": "axis_selection",
            "frequency_compute": "fp32",
            "intermediate_compute": "fp32",
            "rounding_points": ["output_store"],
            "threading": "independent_tokens",
        }
        contracts["contracts"]["qwen_mrope_bad_width"] = base
        with self.assertRaisesRegex(resolver.ContractError, "inconsistent M-RoPE width"):
            resolver.validate_contract_registry(contracts)

    def test_sparse_failure_produces_bounded_granular_request(self):
        profile = planner.load(
            ROOT / "version" / "v8" / "parity_profiles" / "qwen3vl_pytorch_bf16_v1.json"
        )
        report = {
            "comparisons": [
                {"checkpoint_id": "vision.frontend.position.output", "status": "pass"},
                {"checkpoint_id": "vision.layer.0.output", "status": "pass"},
                {"checkpoint_id": "vision.layer.8.output", "status": "pass"},
                {"checkpoint_id": "vision.layer.16.output", "status": "fail"},
            ]
        }
        result = planner.plan(profile, report)
        self.assertEqual(result["status"], "granular")
        self.assertEqual(result["interval"], "vision.layer.8.output->vision.layer.16.output")
        self.assertEqual(result["next_checkpoints"][0], "vision.layer.9.output")
        self.assertEqual(result["next_checkpoints"][-1], "vision.layer.15.output")

    def test_first_failing_layer_expands_only_that_block(self):
        profile = planner.load(
            ROOT / "version" / "v8" / "parity_profiles" / "qwen3vl_pytorch_bf16_v1.json"
        )
        order = ["vision.layer.8.output", "vision.layer.9.output", "vision.layer.10.output"]
        report = {"comparisons": [
            {"checkpoint_id": "vision.layer.8.output", "status": "pass"},
            {"checkpoint_id": "vision.layer.9.output", "status": "fail"},
        ]}
        result = planner.plan(profile, report, checkpoint_order=order)
        self.assertEqual(result["status"], "granular")
        self.assertEqual(result["next_checkpoints"][0], "vision.layer.9.norm1.output")
        self.assertEqual(result["next_checkpoints"][-1], "vision.layer.9.mlp.down")

    def test_graph_ir_metadata_retains_contract_and_checkpoint(self):
        scripts = ROOT / "version" / "v8" / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("build_ir_v8_contract_test", scripts / "build_ir_v8.py")
            assert spec is not None and spec.loader is not None
            build_ir = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(build_ir)
        finally:
            sys.path.pop(0)
        plan = resolver.resolve_contract(
            circuit(), self.contracts, self.kernels, "gemm", "prefill"
        )
        metadata = build_ir._graph_ir_contract_metadata(plan)
        self.assertEqual(metadata["required_contract_id"], CONTRACT_ID)
        self.assertEqual(metadata["resolved_contract_id"], CONTRACT_ID)
        self.assertEqual(metadata["kernel_id"], "gemm_nt_bf16")
        self.assertEqual(metadata["function"], "gemm_nt_bf16")
        self.assertEqual(metadata["semantics"]["rounding"]["points"], ["input_load"])
        self.assertEqual(metadata["checkpoint"]["id"], "vision.layer.0.mlp.up")


if __name__ == "__main__":
    unittest.main(verbosity=2)
