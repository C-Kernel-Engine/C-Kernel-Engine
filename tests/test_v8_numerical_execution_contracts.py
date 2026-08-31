#!/usr/bin/env python3
"""Fail-closed tests for generic v8 numerical execution contracts."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import build_ir_v8  # type: ignore

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


def yarn_rope_circuit(contract_id: str):
    return {
        "name": "yarn_rope_contract_test",
        "required_numerical_contracts": {
            "yarn_rope": {
                "op": "yarn_rope_init",
                "template_ops": ["yarn_rope_init"],
                "phases": {
                    "init": {
                        "contract_id": contract_id,
                        "validation": "validated",
                        "evidence": "unittest/test_instella_yarn_rope.py",
                    }
                },
                "checkpoint": {
                    "id": "decoder.init.yarn_rope_cache",
                    "producer": "yarn_rope_init",
                    "logical_layout": "token_major",
                    "axis_names": ["token", "rotary_pair"],
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
        self.assertEqual(
            plan["kernel"]["function"], "gemm_nt_bf16_parallel_dispatch"
        )
        self.assertEqual(plan["contract"]["semantics"]["compute"]["input"], "bf16_rne")
        self.assertEqual(plan["contract"]["semantics"]["reduction"]["order"], "ascending_k")
        self.assertFalse(
            plan["contract"]["semantics"]["threading"]["thread_count_changes_arithmetic_order"]
        )
        self.assertEqual(plan["checkpoint"]["axis_names"], ["token", "channel"])

    def test_qwen35_norms_resolve_from_hardened_kernel_interfaces(self):
        doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        expected = {
            "decoder.rmsnorm": (
                "rmsnorm.fp32.v1",
                "rmsnorm_forward_llama_production",
            ),
            "decoder.rmsnorm_bf16_pytorch": (
                "rmsnorm.fp32_bf16_values.v1",
                "rmsnorm_forward_qwen3next_pytorch_bf16_storage",
            ),
            "decoder.qk_norm": (
                "qk_norm.fp32_inplace.v1",
                "qk_norm_forward_llama_production",
            ),
            "decoder.qk_norm_bf16_pytorch": (
                "qk_norm.fp32_bf16_values_inplace.v1",
                "qk_norm_forward_pytorch_bf16_storage",
            ),
        }
        for operation, (interface_id, kernel_id) in expected.items():
            for phase in ("prefill", "decode"):
                with self.subTest(operation=operation, phase=phase):
                    plan = resolver.resolve_contract(
                        doc,
                        self.contracts,
                        self.kernels,
                        operation,
                        phase,
                        mode="production",
                    )
                    self.assertEqual(plan["operation_interface"], interface_id)
                    expected_kernel = (
                        "rmsnorm_forward_llama_production_parallel_prefill"
                        if operation == "decoder.rmsnorm" and phase == "prefill"
                        else kernel_id
                    )
                    self.assertEqual(plan["kernel"]["id"], expected_kernel)
                    self.assertEqual(
                        plan["kernel"]["interface_call_abi"], "validated"
                    )
                    scripts = ROOT / "version" / "v8" / "scripts"
                    sys.path.insert(0, str(scripts))
                    try:
                        spec = importlib.util.spec_from_file_location(
                            "build_ir_v8_rmsnorm_interface_test",
                            scripts / "build_ir_v8.py",
                        )
                        assert spec is not None and spec.loader is not None
                        build_ir = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(build_ir)
                    finally:
                        sys.path.pop(0)
                    metadata = build_ir._graph_ir_contract_metadata(plan)
                    self.assertEqual(metadata["operation_interface"], interface_id)
                    self.assertEqual(metadata["interface_call_abi"], "validated")
                    interface = self.kernels["operation_interfaces"][interface_id]
                    port_identities = [
                        (port["role"], port["name"])
                        for port in interface["ports"]
                    ]
                    if operation.startswith("decoder.qk_norm"):
                        self.assertEqual(
                            port_identities,
                            [
                                ("input", "q"),
                                ("input", "k"),
                                ("weight", "q_gamma"),
                                ("weight", "k_gamma"),
                                ("output", "q"),
                                ("output", "k"),
                            ],
                        )
                        self.assertEqual(
                            [
                                port.get("alias_of")
                                for port in interface["ports"]
                                if port["role"] == "output"
                            ],
                            ["input:q", "input:k"],
                        )
                    else:
                        self.assertEqual(
                            port_identities,
                            [
                                ("input", "input"),
                                ("weight", "gamma"),
                                ("output", "output"),
                                ("output", "rstd"),
                            ],
                        )

    def test_required_operation_interface_rejects_discrepant_provider(self):
        doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        kernels = copy.deepcopy(self.kernels)
        kernels["kernels"]["rmsnorm_forward_llama_production"][
            "operation_interface"
        ] = "rmsnorm.incompatible.v1"
        with self.assertRaisesRegex(resolver.ContractError, "resolved to 0 kernels"):
            resolver.resolve_contract(
                doc,
                self.contracts,
                kernels,
                "decoder.rmsnorm",
                "decode",
                mode="production",
            )

    def test_qwen3vl_norms_resolve_from_shared_hardened_interfaces(self):
        doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen3vl.json"
        )
        expected = {
            "decoder.rmsnorm": (
                "rmsnorm.fp32.v1",
                "rmsnorm_forward_llama_production",
            ),
            "decoder.rmsnorm.bf16": (
                "rmsnorm.fp32_bf16_values.v1",
                "rmsnorm_forward_pytorch_bf16_storage",
            ),
            "decoder.qk_norm": (
                "qk_norm.fp32_inplace.v1",
                "qk_norm_forward_llama_production",
            ),
            "decoder.qk_norm.bf16": (
                "qk_norm.fp32_bf16_values_inplace.v1",
                "qk_norm_forward_pytorch_bf16_storage",
            ),
        }
        for operation, (interface_id, kernel_id) in expected.items():
            for phase in ("prefill", "decode"):
                with self.subTest(operation=operation, phase=phase):
                    plan = resolver.resolve_contract(
                        doc,
                        self.contracts,
                        self.kernels,
                        operation,
                        phase,
                        mode="production",
                    )
                    self.assertEqual(plan["operation_interface"], interface_id)
                    expected_kernel = (
                        "rmsnorm_forward_llama_production_parallel_prefill"
                        if operation == "decoder.rmsnorm" and phase == "prefill"
                        else kernel_id
                    )
                    self.assertEqual(plan["kernel"]["id"], expected_kernel)

    def test_audio_and_vision_gelu_resolve_from_hardened_interfaces(self):
        cases = (
            (
                "audio_transformer_encoder.json",
                "audio.encoder.activation",
                "prefill",
                "gelu.fp32_inplace.v1",
                "gelu_erf_fp64_f32_inplace",
            ),
            (
                "audio_transformer_decoder.json",
                "audio.decoder.activation",
                "decode",
                "gelu.fp32_inplace.v1",
                "gelu_erf_fp64_f32_inplace",
            ),
            (
                "qwen3_vl_vision.json",
                "vision.layer.mlp_activation.fp32",
                "prefill",
                "gelu.fp32_inplace.v1",
                "gelu_ggml_inplace",
            ),
            (
                "qwen3_vl_vision.json",
                "vision.layer.mlp_activation",
                "prefill",
                "gelu.fp32_bf16_values_inplace.v1",
                "gelu_pytorch_tanh_bf16_storage",
            ),
            (
                "qwen3_vl_vision.json",
                "vision.projector.activation.pytorch_sleef_exact",
                "prefill",
                "gelu.fp32_bf16_values_inplace.v1",
                "gelu_pytorch_erf_sleef_bf16_storage",
            ),
        )
        for circuit_name, operation, phase, interface_id, kernel_id in cases:
            with self.subTest(circuit=circuit_name, operation=operation):
                doc = resolver.load_json(
                    ROOT / "version" / "v8" / "circuits" / circuit_name
                )
                plan = resolver.resolve_contract(
                    doc,
                    self.contracts,
                    self.kernels,
                    operation,
                    phase,
                    mode="production",
                )
                self.assertEqual(plan["operation_interface"], interface_id)
                self.assertEqual(plan["kernel"]["id"], kernel_id)

    def test_undeclared_operation_interface_preserves_legacy_contract_lookup(self):
        doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        del doc["required_numerical_contracts"]["decoder.rmsnorm"][
            "operation_interface"
        ]
        plan = resolver.resolve_contract(
            doc,
            self.contracts,
            self.kernels,
            "decoder.rmsnorm",
            "decode",
            mode="production",
        )
        self.assertNotIn("operation_interface", plan)
        self.assertEqual(
            plan["kernel"]["id"], "rmsnorm_forward_llama_production"
        )

    def test_hardened_kernel_interface_requires_complete_port_metadata(self):
        path = (
            ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "rmsnorm_forward_llama_production.json"
        )
        doc = json.loads(path.read_text(encoding="utf-8"))
        del doc["inputs"][0]["layout"]
        with self.assertRaisesRegex(resolver.ContractError, r"missing=\['layout'\]"):
            resolver._load_operation_interface(doc, path)

    def test_providers_cannot_disagree_on_one_operation_interface(self):
        path = (
            ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "rmsnorm_forward_llama_production.json"
        )
        first = json.loads(path.read_text(encoding="utf-8"))
        second = copy.deepcopy(first)
        second["outputs"][0]["layout"] = "head_major_contiguous"
        with tempfile.TemporaryDirectory(
            prefix=".tmp-v8-interface-", dir=ROOT
        ) as temp_dir:
            root = Path(temp_dir)
            (root / "first.json").write_text(
                json.dumps(first), encoding="utf-8"
            )
            (root / "second.json").write_text(
                json.dumps(second), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                resolver.ContractError,
                "kernel maps disagree on operation interface",
            ):
                resolver.load_kernel_capabilities(root, contracts=self.contracts)

    def test_hardened_kernel_interface_rejects_unknown_alias_target(self):
        path = (
            ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "qk_norm_forward_llama_production.json"
        )
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["outputs"][0]["alias_of"] = "input:missing"
        with self.assertRaisesRegex(resolver.ContractError, "aliases an unknown port"):
            resolver._load_operation_interface(doc, path)

    def test_hardened_kernel_interface_rejects_incompatible_alias(self):
        path = (
            ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "qk_norm_forward_llama_production.json"
        )
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["outputs"][0]["layout"] = "token_major_contiguous"
        with self.assertRaisesRegex(resolver.ContractError, "incompatible alias"):
            resolver._load_operation_interface(doc, path)

    def test_hardened_interface_rejects_missing_abi_port(self):
        path = (
            ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "rmsnorm_forward_llama_production.json"
        )
        doc = json.loads(path.read_text(encoding="utf-8"))
        del doc["call_abi"]["params"][0]["ports"]
        with self.assertRaisesRegex(
            resolver.ContractError, "does not represent every logical port"
        ):
            resolver._load_operation_interface(doc, path)

    def test_hardened_interface_rejects_unknown_abi_port(self):
        path = (
            ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "rmsnorm_forward_llama_production.json"
        )
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["call_abi"]["params"][0]["ports"] = ["input:missing"]
        with self.assertRaisesRegex(resolver.ContractError, "unknown logical ports"):
            resolver._load_operation_interface(doc, path)

    def test_hardened_interface_rejects_duplicate_abi_port_owner(self):
        path = (
            ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "rmsnorm_forward_llama_production.json"
        )
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["call_abi"]["params"][1]["ports"].append("input:input")
        with self.assertRaisesRegex(
            resolver.ContractError, "binds logical ports more than once"
        ):
            resolver._load_operation_interface(doc, path)

    def test_hardened_interface_rejects_split_inplace_alias(self):
        path = (
            ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "qk_norm_forward_llama_production.json"
        )
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["call_abi"]["params"][0]["ports"] = ["input:q"]
        doc["call_abi"]["params"].append({
            "name": "q_output",
            "source": "scratch:q_scratch",
            "ports": ["output:q"],
        })
        with self.assertRaisesRegex(
            resolver.ContractError, "splits an in-place alias"
        ):
            resolver._load_operation_interface(doc, path)

    def test_hardened_interface_rejects_null_required_port(self):
        path = (
            ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "rmsnorm_forward_llama_production.json"
        )
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["call_abi"]["params"][2]["source"] = "null"
        with self.assertRaisesRegex(resolver.ContractError, "nulls a required logical port"):
            resolver._load_operation_interface(doc, path)

    def test_hardened_interface_accepts_explicit_null_optional_port(self):
        path = (
            ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "rmsnorm_forward_llama_production.json"
        )
        doc = json.loads(path.read_text(encoding="utf-8"))
        interface = resolver._load_operation_interface(doc, path)
        self.assertEqual(interface["id"], "rmsnorm.fp32.v1")

    def test_resolver_rejects_unvalidated_interface_abi_metadata(self):
        doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        kernels = copy.deepcopy(self.kernels)
        del kernels["kernels"]["rmsnorm_forward_llama_production"][
            "interface_call_abi"
        ]
        with self.assertRaisesRegex(
            resolver.ContractError, "no validated interface-to-ABI boundary"
        ):
            resolver.resolve_contract(
                doc,
                self.contracts,
                kernels,
                "decoder.rmsnorm",
                "decode",
                mode="production",
            )

    def test_recurrent_elementwise_maps_own_validated_interfaces_and_abis(self):
        expected = {
            "attn_gate_sigmoid_mul_forward": "attn_gate_sigmoid_mul.fp32.v1",
            "recurrent_norm_gate_llama_avx2_forward": "recurrent_norm_gate.fp32.v1",
            "recurrent_norm_gate_pytorch_bf16_storage": "recurrent_norm_gate.fp32_bf16_values.v1",
            "recurrent_qk_l2_norm_forward": "recurrent_qk_l2_norm.fp32_inplace.v1",
            "recurrent_qk_l2_norm_pytorch_bf16_storage": "recurrent_qk_l2_norm.fp32_bf16_values_inplace.v1",
            "recurrent_silu_forward_ggml": "recurrent_silu.fp32.v1",
            "recurrent_silu_forward_pytorch_bf16_storage": "recurrent_silu.fp32_bf16_values.v1",
            "ssm_conv1d_forward_llama_production": "ssm_conv1d.fp32.v1",
            "ssm_conv1d_forward_llama_fma": "ssm_conv1d.fp32.v1",
            "ssm_conv1d_forward_pytorch_bf16_storage": "ssm_conv1d.fp32_bf16_values.v1",
            "swiglu_forward_ggml": "swiglu.fp32.v1",
            "swiglu_forward_pytorch_bf16_storage": "swiglu.fp32_bf16_values.v1",
        }
        for kernel_id, interface_id in expected.items():
            with self.subTest(kernel_id=kernel_id):
                kernel = self.kernels["kernels"][kernel_id]
                self.assertEqual(kernel["operation_interface"], interface_id)
                self.assertEqual(kernel["interface_call_abi"], "validated")

        inplace = resolver.load_json(
            ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "recurrent_qk_l2_norm_forward.json"
        )
        params = {
            param["name"]: param
            for param in inplace["call_abi"]["params"]
        }
        self.assertEqual(params["q"]["ports"], ["input:q", "output:q"])
        self.assertEqual(params["k"]["ports"], ["input:k", "output:k"])

    def test_kernel_interface_migration_debt_does_not_regress(self):
        scripts = ROOT / "version" / "v8" / "scripts"
        spec = importlib.util.spec_from_file_location(
            "audit_kernel_map_interfaces_v8",
            scripts / "audit_kernel_map_interfaces_v8.py",
        )
        assert spec is not None and spec.loader is not None
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        report = audit.build_report()
        baseline = audit._load(audit.BASELINE)
        audit.validate_ratchet(report, baseline)
        self.assertEqual(report["counts"]["kernel_maps"], 311)
        self.assertEqual(report["counts"]["physical_layout_maps"], 6)
        self.assertEqual(report["counts"]["resolver_governed_maps"], 117)
        self.assertEqual(report["counts"]["interface_hardened_maps"], 66)
        self.assertEqual(
            report["counts"]["interface_abi_crossvalidated_maps"], 66
        )
        self.assertEqual(report["counts"]["contract_pending_maps"], 51)
        self.assertEqual(report["counts"]["map_owned_call_abi"], 184)
        self.assertEqual(report["counts"]["legacy_interface_ready_maps"], 34)
        self.assertEqual(report["counts"]["selection_managed_maps"], 75)
        self.assertEqual(report["selection"]["legacy_selection_if_statements"], 62)
        self.assertEqual(report["selection"]["operation_specific_if_statements"], 29)

    def test_yarn_init_contracts_resolve_exact_storage_providers(self):
        expected = {
            "yarn_rope_cache_explicit_positions_fp32": (
                "yarn_rope_cache_explicit_positions_f32",
                "fp32",
            ),
            "yarn_rope_cache_explicit_positions_bf16_storage": (
                "yarn_rope_cache_explicit_positions_bf16",
                "bf16",
            ),
        }
        for contract_id, (function, output_storage) in expected.items():
            with self.subTest(contract_id=contract_id):
                plan = resolver.resolve_contract(
                    yarn_rope_circuit(contract_id),
                    self.contracts,
                    self.kernels,
                    "yarn_rope",
                    "init",
                    mode="production",
                )
                self.assertEqual(plan["kernel"]["id"], function)
                self.assertEqual(plan["kernel"]["function"], function)
                self.assertEqual(
                    plan["contract"]["semantics"]["storage"]["output"],
                    output_storage,
                )

    def test_pytorch_onednn_brgemm_contract_resolves_exact_provider(self):
        doc = circuit(validation="validated")
        requirement = doc["required_numerical_contracts"]["gemm"]["phases"]["prefill"]
        requirement["contract_id"] = (
            "bf16_weight_bf16_input_pytorch_onednn_brgemm_bf16_output"
        )
        plan = resolver.resolve_contract(
            doc, self.contracts, self.kernels, "gemm", "prefill", mode="production"
        )
        self.assertEqual(
            plan["kernel"]["id"],
            "gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage",
        )
        self.assertEqual(plan["implementation"]["threading"]["runtime"], "openmp")
        self.assertTrue(
            plan["contract"]["semantics"]["threading"]
            ["thread_count_changes_arithmetic_order"]
        )

    def test_pytorch_welford_layernorm_resolves_only_in_bringup(self):
        doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen3_vl_vision.json"
        )
        requirement = doc["required_numerical_contracts"]["vision.layer.layernorm"]
        requirement["phases"]["prefill"] = {
            "contract_id": (
                "layernorm_bf16_storage_fp32_compute_aten_avx2_welford_bf16_output"
            ),
            "validation": "observed",
            "evidence": "production-shape PyTorch 2.8 oracle",
        }
        plan = resolver.resolve_contract(
            doc,
            self.contracts,
            self.kernels,
            "vision.layer.layernorm",
            "prefill",
            mode="bringup",
        )
        self.assertEqual(plan["kernel"]["id"], "layernorm_bf16_pytorch_welford")
        self.assertEqual(
            plan["kernel"]["function"],
            "layernorm_pytorch_welford_bf16_storage",
        )
        with self.assertRaisesRegex(
            resolver.ContractError, "production resolution uses unvalidated contract"
        ):
            resolver.resolve_contract(
                doc,
                self.contracts,
                self.kernels,
                "vision.layer.layernorm",
                "prefill",
                mode="production",
            )

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

    def test_cohere_position_contract_resolves_mixed_precision_kernel(self):
        circuit_doc = build_ir_v8._load_builtin_template_doc(
            "cohere_compass_vision"
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
            "bf16_table_fp32_tiled_2d_align_corners_bf16_residual",
        )
        self.assertEqual(
            plan["kernel"]["id"],
            "position_embeddings_add_tiled_2d_align_corners_fp32_interp_bf16",
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

    def test_qwen3vl_decoder_norm_contracts_resolve_exact_providers(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen3vl.json"
        )
        expected = {
            "decoder.rmsnorm": {
                "prefill": (
                    "rmsnorm_forward_llama_production_parallel_prefill",
                    "rmsnorm_forward_llama_production_parallel_dispatch",
                ),
                "decode": (
                    "rmsnorm_forward_llama_production",
                    "rmsnorm_forward_llama_production",
                ),
            },
            "decoder.qk_norm": {
                "prefill": (
                    "qk_norm_forward_llama_production",
                    "qk_norm_forward_llama_production",
                ),
                "decode": (
                    "qk_norm_forward_llama_production",
                    "qk_norm_forward_llama_production",
                ),
            },
        }
        for operation, functions in expected.items():
            for phase in ("prefill", "decode"):
                with self.subTest(operation=operation, phase=phase):
                    plan = resolver.resolve_contract(
                        circuit_doc,
                        self.contracts,
                        self.kernels,
                        operation,
                        phase,
                        mode="production",
                    )
                    kernel_id, function = functions[phase]
                    self.assertEqual(plan["kernel"]["id"], kernel_id)
                    self.assertEqual(plan["kernel"]["function"], function)
                    semantics = plan["contract"]["semantics"]
                    self.assertEqual(semantics["compute"]["accumulator"], "fp64")
                    self.assertEqual(semantics["reduction"]["order"], "left_to_right")
                    self.assertFalse(
                        semantics["threading"]["thread_count_changes_arithmetic_order"]
                    )

    def test_qwen35_rmsnorm_resolves_llama_production_provider(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.rmsnorm",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "rmsnorm_llama_cpu_production_fp32_output",
                )
                expected_function = (
                    "rmsnorm_forward_llama_production_parallel_dispatch"
                    if phase == "prefill"
                    else "rmsnorm_forward_llama_production"
                )
                self.assertEqual(plan["kernel"]["function"], expected_function)

    def test_qwen35_recurrent_core_resolves_exact_llama_avx2_provider(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        expected = {
            "prefill": (
                "gated_deltanet_llama_fused_prefill_fp32_state",
                "gated_deltanet_llama_prefill_parallel_dispatch",
            ),
            "decode": (
                "gated_deltanet_llama_avx2_decode_fp32_state",
                "gated_deltanet_llama_avx2_parallel_forward",
            ),
        }
        for phase, (contract_id, function) in expected.items():
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.recurrent_core.prefill",
                    phase,
                    mode="production",
                )
                self.assertEqual(plan["contract"]["id"], contract_id)
                self.assertEqual(plan["kernel"]["function"], function)
                kernel_capability = self.kernels["kernels"][plan["kernel"]["id"]]
                kernel = resolver.load_json(ROOT / kernel_capability["source"])
                abi_sources = {
                    param["name"]: param["source"]
                    for param in kernel["call_abi"]["params"]
                }
                self.assertEqual(
                    abi_sources["group_count"],
                    "dim:ssm_group_count",
                )
                self.assertEqual(kernel["inputs"][0]["shape"][-2:], ["G", "D"])
                self.assertEqual(kernel["inputs"][1]["shape"][-2:], ["G", "D"])

    def test_qwen35_recurrent_core_declares_shape_aware_dynamic_head_scheduling(self):
        kernel_map = resolver.load_json(
            ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "gated_deltanet_llama_avx2_prefill_forward.json"
        )
        variants = {row["name"]: row for row in kernel_map["impl"]["variants"]}
        short = variants["avx2_short_medium_dynamic_heads"]
        long = variants["avx2_long_dynamic_heads"]

        self.assertEqual(short["shape_constraints"], {"R_max": 128})
        self.assertEqual(short["default_concurrency"], "ceil(H/2)")
        self.assertEqual(long["shape_constraints"], {"R_min": 129})
        self.assertEqual(long["default_concurrency"], "min(pool_threads,H)")
        threading = kernel_map["numerical_capabilities"][0]["implementation"]["threading"]
        self.assertIn("independent_heads", threading["work_partition"])
        self.assertEqual(threading["dispatch"], ["ck_threadpool_parallel_for_n"])
        self.assertEqual(threading["reduction_order_effect"], "none")

    def test_qwen35_full_attention_qk_norm_resolves_llama_provider(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.qk_norm",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["kernel"]["function"],
                    "qk_norm_forward_llama_production",
                )

    def test_qwen35_full_attention_qk_norm_resolves_pytorch_bf16_provider(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        required = circuit_doc["required_numerical_contracts"]
        selector_key = "decoder_qk_norm_reduction_policy"
        selector_value = "pytorch_avx2_cascade_exact"
        self.assertEqual(
            required["decoder.qk_norm"]["selector"],
            {"config_not_equals": {selector_key: selector_value}},
        )
        self.assertEqual(
            required["decoder.qk_norm_bf16_pytorch"]["selector"],
            {"config_equals": {selector_key: selector_value}},
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.qk_norm_bf16_pytorch",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["kernel"]["function"],
                    "qk_norm_forward_pytorch_bf16_storage",
                )

    def test_qwen35_recurrent_norm_gate_resolves_exact_composed_provider(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.recurrent_norm_gate",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "recurrent_norm_gate_llama_avx2_fp32_output",
                )
                expected_function = (
                    "recurrent_norm_gate_llama_avx2_parallel_dispatch"
                    if phase == "prefill"
                    else "recurrent_norm_gate_llama_avx2_forward"
                )
                self.assertEqual(plan["kernel"]["function"], expected_function)

    def test_qwen35_attention_gate_resolves_exact_llama_sigmoid_provider(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.attention_gate",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "attention_gate_llama_sigmoid_fp32_output",
                )
                self.assertEqual(
                    plan["kernel"]["function"],
                    "attn_gate_sigmoid_mul_forward",
                )

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

    def test_qwen35_circuit_resolves_exact_recurrent_qk_l2_norm(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.recurrent_qk_l2_norm",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "recurrent_qk_l2_llama_cpu_fp32_output",
                )
                self.assertEqual(
                    plan["kernel"]["id"], "recurrent_qk_l2_norm_forward"
                )
                self.assertEqual(
                    plan["kernel"]["function"], "recurrent_qk_l2_norm_forward"
                )
                semantics = plan["contract"]["semantics"]
                self.assertEqual(semantics["compute"]["accumulator"], "fp64")
                self.assertEqual(
                    semantics["compute"]["evaluation_order"],
                    "fp32_product_then_ascending_fp64_sum_then_fp32_sqrt_then_max_eps_then_reciprocal",
                )
                self.assertEqual(
                    semantics["reduction"]["order"], "left_to_right"
                )

    def test_qwen35_circuit_resolves_exact_recurrent_silu(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.recurrent_silu",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "recurrent_silu_llama_avx2_fp32_output",
                )
                self.assertEqual(
                    plan["kernel"]["id"], "recurrent_silu_forward_ggml"
                )
                self.assertEqual(
                    plan["kernel"]["function"], "recurrent_silu_forward_ggml"
                )
                semantics = plan["contract"]["semantics"]
                self.assertEqual(semantics["storage"]["weight"], "none")
                self.assertEqual(semantics["reduction"]["kind"], "none")
                self.assertEqual(
                    semantics["compute"]["evaluation_order"],
                    "llama_x86_isa_vector_expf_then_add_then_divide_with_scalar_expf_tail",
                )

    def test_qwen35_circuit_resolves_pytorch_bf16_recurrent_conv_and_silu(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        expected = {
            "decoder.recurrent_ssm_conv_bf16_pytorch": (
                "ssm_conv1d_pytorch_fp32_compute_bf16_output",
                "ssm_conv1d_forward_pytorch_bf16_storage",
            ),
            "decoder.recurrent_silu_bf16_pytorch": (
                "recurrent_silu_pytorch_sleef_bf16_input_bf16_output",
                "recurrent_silu_forward_pytorch_bf16_storage",
            ),
        }
        for operation, (contract_id, kernel_id) in expected.items():
            for phase in ("prefill", "decode"):
                with self.subTest(operation=operation, phase=phase):
                    plan = resolver.resolve_contract(
                        circuit_doc,
                        self.contracts,
                        self.kernels,
                        operation,
                        phase,
                        mode="production",
                    )
                    self.assertEqual(plan["contract"]["id"], contract_id)
                    self.assertEqual(plan["kernel"]["id"], kernel_id)
                    self.assertEqual(plan["kernel"]["function"], kernel_id)
                    self.assertEqual(
                        plan["contract"]["semantics"]["storage"]["output"],
                        "bf16",
                    )

    def test_qwen35_circuit_resolves_llama_ssm_conv(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        expected = {
            "prefill": (
                "ssm_conv1d_llama_scalar_mul_add_fp32_output",
                "ssm_conv1d_forward_llama_production",
                "ascending_kernel_rounded_multiply_then_add",
            ),
            "decode": (
                "ssm_conv1d_llama_scalar_mul_add_fp32_output",
                "ssm_conv1d_forward_llama_production",
                "ascending_kernel_rounded_multiply_then_add",
            ),
        }
        for phase, (contract_id, kernel_id, evaluation_order) in expected.items():
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.recurrent_ssm_conv",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    contract_id,
                )
                self.assertEqual(
                    plan["kernel"]["id"],
                    kernel_id,
                )
                self.assertEqual(
                    plan["kernel"]["function"],
                    kernel_id,
                )
                semantics = plan["contract"]["semantics"]
                self.assertEqual(semantics["storage"]["output"], "fp32")
                self.assertEqual(
                    semantics["compute"]["evaluation_order"],
                    evaluation_order,
                )
                self.assertEqual(
                    semantics["reduction"]["order"], "left_to_right"
                )

    def test_qwen35_circuit_resolves_pytorch_bf16_grouped_deltanet(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        expected = {
            "prefill": "gated_deltanet_pytorch_grouped_bf16_prefill_forward",
            "decode": "gated_deltanet_pytorch_grouped_bf16_forward",
        }
        for phase, kernel_id in expected.items():
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.recurrent_core_bf16_pytorch",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "gated_deltanet_pytorch_grouped_qk_fp32_state_bf16_output",
                )
                self.assertEqual(plan["kernel"]["id"], kernel_id)
                self.assertEqual(plan["kernel"]["function"], kernel_id)
                self.assertEqual(
                    plan["contract"]["semantics"]["compute"]["evaluation_order"],
                    "ascending_token_grouped_qk_repeat_interleave_group_equals_"
                    "head_div_heads_per_group_then_avx2_state_update_then_bf16_"
                    "output_store",
                )

    def test_qwen35_circuit_resolves_pytorch_bf16_recurrent_qk_l2(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.recurrent_qk_l2_norm_bf16_pytorch",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "recurrent_qk_l2_pytorch_bf16_vector_tail_bf16_output",
                )
                self.assertEqual(
                    plan["kernel"]["id"],
                    "recurrent_qk_l2_norm_pytorch_bf16_storage",
                )

    def test_qwen35_circuit_resolves_pytorch_bf16_recurrent_norm_gate(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.recurrent_norm_gate_bf16_pytorch",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "recurrent_norm_gate_pytorch_bf16_storage",
                )
                self.assertEqual(
                    plan["kernel"]["id"],
                    "recurrent_norm_gate_pytorch_bf16_storage",
                )

    def test_qwen35_circuit_resolves_pytorch_bf16_residual_add(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.residual_add_bf16_pytorch",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "residual_add_bf16_input_fp32_add_bf16_output",
                )
                self.assertEqual(
                    plan["kernel"]["id"],
                    "ck_residual_add_token_major_bf16_storage",
                )

    def test_qwen35_circuit_resolves_pytorch_bf16_rmsnorm(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.rmsnorm_bf16_pytorch",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "rmsnorm_qwen3next_pytorch_avx2_bf16_storage",
                )
                self.assertEqual(
                    plan["kernel"]["id"],
                    "rmsnorm_forward_qwen3next_pytorch_bf16_storage",
                )

    def test_qwen35_circuit_resolves_pytorch_bf16_swiglu(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.swiglu_bf16_pytorch",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "swiglu_pytorch_sleef_bf16_intermediate_bf16_output",
                )
                self.assertEqual(
                    plan["kernel"]["id"],
                    "swiglu_forward_pytorch_bf16_storage",
                )

    def test_qwen35_circuit_resolves_exact_swiglu(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.swiglu",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "swiglu_fp32_ggml_vector_exp_fp32_output",
                )
                self.assertEqual(plan["kernel"]["id"], "swiglu_forward_ggml")
                self.assertEqual(
                    plan["kernel"]["function"], "swiglu_forward_ggml"
                )

    def test_qwen2_circuit_resolves_exact_swiglu(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen2.json"
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    "decoder.swiglu",
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "swiglu_fp32_ggml_vector_exp_fp32_output",
                )
                self.assertEqual(plan["kernel"]["id"], "swiglu_forward_ggml")
                self.assertEqual(
                    plan["kernel"]["function"], "swiglu_forward_ggml"
                )

    def test_qwen35_circuit_resolves_exact_q5_recurrent_qkv_projection(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        expected = {
            "prefill": (
                "decoder.recurrent_qkv_projection.q5_k.prefill",
                "gemm_nt_q5_k",
                "gemm_nt_q5_k_parallel_dispatch_with_scratch",
            ),
            "decode": (
                "decoder.recurrent_qkv_projection.q5_k.decode",
                "gemv_q5_k",
                "gemv_q5_k",
            ),
        }
        for phase, (operation, kernel_id, function) in expected.items():
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    operation,
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "q5_k_weight_q8_k_input_avx2_fma_fp32_output",
                )
                self.assertEqual(plan["kernel"]["id"], kernel_id)
                self.assertEqual(plan["kernel"]["function"], function)
                semantics = plan["contract"]["semantics"]
                self.assertEqual(semantics["compute"]["weight"], "int5")
                self.assertEqual(semantics["reduction"]["merge_order"], "pairwise_tree")

    def test_qwen35_circuit_resolves_exact_q6_recurrent_qkv_projection(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        expected = {
            "prefill": (
                "decoder.recurrent_qkv_projection.q6_k.prefill",
                "gemm_nt_q6_k_q8_k",
            ),
            "decode": (
                "decoder.recurrent_qkv_projection.q6_k.decode",
                "gemv_q6_k_q8_k",
            ),
        }
        for phase, (operation, kernel_id) in expected.items():
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    operation,
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "q6_k_weight_q8_k_input_llama_fp32_output",
                )
                self.assertEqual(plan["kernel"]["id"], kernel_id)
                self.assertEqual(plan["kernel"]["function"], kernel_id)
                semantics = plan["contract"]["semantics"]
                self.assertEqual(semantics["compute"]["weight"], "int6")
                self.assertEqual(
                    semantics["reduction"]["merge_order"],
                    "pairwise_tree",
                )

    def test_qwen35_circuit_resolves_exact_q8_recurrent_qkv_projection(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        expected = {
            "prefill": (
                "decoder.recurrent_qkv_projection.q8_0.prefill",
                "gemm_nt_q8_0_q8_0_contract",
            ),
            "decode": (
                "decoder.recurrent_qkv_projection.q8_0.decode",
                "gemv_q8_0_q8_0_contract",
            ),
        }
        for phase, (operation, kernel_id) in expected.items():
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    operation,
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "q8_0_weight_fp32_input_internal_q8_0_llama_fp32_output",
                )
                self.assertEqual(plan["kernel"]["id"], kernel_id)
                self.assertEqual(plan["kernel"]["function"], kernel_id)
                semantics = plan["contract"]["semantics"]
                self.assertEqual(semantics["compute"]["weight"], "int8")
                self.assertEqual(semantics["reduction"]["merge_order"], "none")
                kernel_map = resolver.load_json(
                    ROOT / "version" / "v8" / "kernel_maps" / f"{kernel_id}.json"
                )
                self.assertEqual(kernel_map["call_abi"]["version"], 1)
                self.assertEqual(
                    [param["name"] for param in kernel_map["call_abi"]["params"]],
                    ["A", "B", "bias", "C", "M", "N", "K"]
                    if phase == "prefill"
                    else ["y", "W", "x", "M", "K"],
                )
        legacy_bindings = resolver.load_json(
            ROOT / "version" / "v8" / "kernel_maps" / "kernel_bindings.json"
        )["bindings"]
        for kernel_id in (
            "gemm_nt_q8_0_q8_0_contract",
            "gemv_q8_0_q8_0_contract",
        ):
            self.assertNotIn(kernel_id, legacy_bindings)

    def test_qwen35_circuit_resolves_pytorch_bf16_projection_boundary(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        expected = {
            "prefill": (
                "decoder.projections_bf16_pytorch.prefill",
                "gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage",
            ),
            "decode": (
                "decoder.projections_bf16_pytorch.decode",
                "gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage",
            ),
        }
        for phase, (operation, kernel_id) in expected.items():
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    operation,
                    phase,
                    mode="bringup",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "bf16_weight_bf16_input_pytorch_onednn_brgemm_bf16_output",
                )
                self.assertEqual(plan["kernel"]["id"], kernel_id)
                self.assertEqual(plan["kernel"]["function"], kernel_id)
                semantics = plan["contract"]["semantics"]
                self.assertEqual(semantics["compute"]["weight"], "bf16")
                self.assertEqual(
                    semantics["reduction"]["merge_order"],
                    "implementation_defined",
                )

    def test_qwen35_circuit_resolves_llama_fp32_recurrent_gate_scalars(self):
        circuit_doc = resolver.load_json(
            ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        )
        operation = "decoder.recurrent_gate_scalar_projections_llama"
        requirement = circuit_doc["required_numerical_contracts"][operation]
        self.assertEqual(
            requirement["selector"],
            {
                "config_equals": {"ssm_time_step_rank": 48},
                "config_not_equals": {"recurrent_qkv_weight_dtype": "bf16"},
            },
        )

        scripts = ROOT / "version" / "v8" / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location(
                "build_ir_v8_qwen36_gate_contract_test",
                scripts / "build_ir_v8.py",
            )
            assert spec is not None and spec.loader is not None
            build_ir = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(build_ir)
        finally:
            sys.path.pop(0)
        selector = requirement["selector"]
        self.assertTrue(
            build_ir._contract_selector_matches(
                selector,
                {
                    "ssm_time_step_rank": 48,
                    "recurrent_qkv_weight_dtype": "q6_k",
                },
                operation,
            )
        )
        self.assertFalse(
            build_ir._contract_selector_matches(
                selector,
                {
                    "ssm_time_step_rank": 16,
                    "recurrent_qkv_weight_dtype": "q5_k",
                },
                operation,
            )
        )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plan = resolver.resolve_contract(
                    circuit_doc,
                    self.contracts,
                    self.kernels,
                    operation,
                    phase,
                    mode="production",
                )
                self.assertEqual(
                    plan["contract"]["id"],
                    "fp32_weight_fp32_input_llamafile_x86_native_fp32_output",
                )
                self.assertEqual(
                    plan["kernel"]["id"], "gemm_nt_f32_llama_production"
                )
                self.assertEqual(
                    plan["kernel"]["function"],
                    "gemm_nt_f32_llama_production_parallel_dispatch",
                )
                semantics = plan["contract"]["semantics"]
                self.assertEqual(
                    semantics["reduction"]["merge_order"], "pairwise_tree"
                )
                self.assertFalse(
                    semantics["threading"][
                        "thread_count_changes_arithmetic_order"
                    ]
                )
                kernel_map = resolver.load_json(
                    ROOT
                    / "version"
                    / "v8"
                    / "kernel_maps"
                    / "gemm_nt_f32_llama_production.json"
                )
                call_sources = {
                    param["name"]: param["source"]
                    for param in kernel_map["call_abi"]["params"]
                }
                self.assertEqual(call_sources["M"], "runtime:seq_len")

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
            "vision.layer.attention": "attention_forward_full_head_major_gqa_pytorch_cpu_flash_bf16_storage",
            "vision.layer.out_projection": "gemm_nt_bf16_native_bf16_storage",
            "vision.layer.residual": "ck_residual_add_token_major_bf16_storage",
            "vision.projector.projection": "gemm_nt_bf16_amx_bf16_storage_workspace",
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
                        plan["contract"]["id"],
                        "attention_bf16_pytorch_cpu_flash_amx_exact",
                    )
                    self.assertEqual(
                        plan["kernel"]["id"],
                        "attention_forward_full_head_major_gqa_pytorch_cpu_flash_bf16_storage",
                    )
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

    def test_ranking_failure_expands_largest_sparse_drift_interval(self):
        profile = planner.load(
            ROOT / "version" / "v8" / "parity_profiles" / "qwen3vl_pytorch_bf16_v1.json"
        )
        report = {
            "comparisons": [
                {"checkpoint_id": "vision.frontend.position.output", "status": "pass", "metrics": {"relative_rmse": 0.0001}},
                {"checkpoint_id": "vision.layer.0.output", "status": "pass", "metrics": {"relative_rmse": 0.0002}},
                {"checkpoint_id": "vision.layer.8.output", "status": "pass", "metrics": {"relative_rmse": 0.0060}},
                {"checkpoint_id": "vision.layer.16.output", "status": "pass", "metrics": {"relative_rmse": 0.0065}},
            ],
            "ranking_divergence": {"classification": "RANKING_DIVERGENCE", "position": 26},
        }
        result = planner.plan(profile, report)
        self.assertEqual(result["status"], "granular_accumulated_drift")
        self.assertEqual(result["interval"], "vision.layer.0.output->vision.layer.8.output")
        self.assertEqual(result["next_checkpoints"][0], "vision.layer.1.output")
        self.assertEqual(result["next_checkpoints"][-1], "vision.layer.7.output")

    def test_ranking_failure_does_not_reopen_byte_exact_encoder(self):
        profile = planner.load(
            ROOT / "version" / "v8" / "parity_profiles" / "qwen3vl_pytorch_bf16_v1.json"
        )
        report = {
            "comparisons": [
                {"checkpoint_id": checkpoint, "status": "pass", "metrics": {"relative_rmse": 0.0}}
                for checkpoint in profile["checkpoint_order"]
            ],
            "ranking_divergence": {"classification": "RANKING_DIVERGENCE", "position": 26},
        }
        result = planner.plan(profile, report)
        self.assertEqual(result["status"], "ranking_attributed")
        self.assertEqual(result["reason"], "ranking_failed_without_nonzero_sparse_tensor_growth")
        self.assertEqual(result["next_checkpoints"], [])

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
        self.assertEqual(metadata["function"], "gemm_nt_bf16_parallel_dispatch")
        self.assertEqual(metadata["semantics"]["rounding"]["points"], ["input_load"])
        self.assertEqual(metadata["checkpoint"]["id"], "vision.layer.0.mlp.up")


if __name__ == "__main__":
    unittest.main(verbosity=2)
