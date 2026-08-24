#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
V8_ROOT = REPO_ROOT / "version" / "v8"
SCRIPT = V8_ROOT / "scripts" / "resolve_attention_contracts_v8.py"
SPEC = importlib.util.spec_from_file_location("resolve_attention_contracts_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
resolver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = resolver
SPEC.loader.exec_module(resolver)

QWEN3VL_FP16_ATTENTION = "decoder.attention.fp16_cache"


class AttentionContractV8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.circuit_path = V8_ROOT / "circuits" / "qwen3vl.json"
        cls.circuit = resolver.load_json(cls.circuit_path)
        cls.qwen35_circuit_path = V8_ROOT / "circuits" / "qwen35.json"
        cls.qwen35_circuit = resolver.load_json(cls.qwen35_circuit_path)
        cls.vision_circuit_path = V8_ROOT / "circuits" / "qwen3_vl_vision.json"
        cls.vision_circuit = resolver.load_json(cls.vision_circuit_path)
        cls.contracts = resolver.load_json(resolver.DEFAULT_CONTRACTS)
        cls.linear_contracts = resolver.load_json(resolver.DEFAULT_LINEAR_CONTRACTS)
        cls.kernels = resolver.load_kernel_capabilities()

    def resolve(self, phase: str, mode: str = "bringup"):
        return resolver.resolve_contract(
            copy.deepcopy(self.circuit),
            copy.deepcopy(self.contracts),
            copy.deepcopy(self.kernels),
            operation=QWEN3VL_FP16_ATTENTION,
            phase=phase,
            mode=mode,
            source_circuit_path=self.circuit_path,
        )

    def resolve_qwen35(self, phase: str, mode: str = "bringup"):
        return resolver.resolve_contract(
            copy.deepcopy(self.qwen35_circuit),
            copy.deepcopy(self.contracts),
            copy.deepcopy(self.kernels),
            operation="decoder.attention",
            phase=phase,
            mode=mode,
            source_circuit_path=self.qwen35_circuit_path,
        )

    def test_registry_defines_complete_semantics(self) -> None:
        resolver.validate_contract_registry(copy.deepcopy(self.contracts))

    def test_shape_dispatch_rejects_missing_reduction_contract(self) -> None:
        contracts = copy.deepcopy(self.contracts)
        contracts["contracts"]["f16_flash_auto_qtile64"]["partition"][
            "below_threshold"
        ] = "missing_short_query_contract"
        with self.assertRaisesRegex(
            resolver.ContractError,
            "(?s)HARD CONTRACT FAULT.*references missing below_threshold",
        ):
            resolver.validate_contract_registry(contracts)

    def test_shape_dispatch_rejects_unvalidated_reduction_contract(self) -> None:
        contracts = copy.deepcopy(self.contracts)
        contracts["contracts"]["f16_online_single_range"]["status"] = "observed"
        with self.assertRaisesRegex(
            resolver.ContractError,
            "(?s)HARD CONTRACT FAULT.*references unvalidated below_threshold",
        ):
            resolver.validate_contract_registry(contracts)

    def test_f16_split_contract_declares_valid_rounded_scheduling_extent(self) -> None:
        contract = self.contracts["contracts"]["f16_online_fp32_merge"]
        self.assertEqual(contract["partition"]["kind"], "kv_chunks_by_workers")
        self.assertEqual(contract["partition"]["threshold"], 512)
        self.assertEqual(contract["partition"]["extent_source"], "valid_kv_rounded")
        self.assertEqual(contract["partition"]["extent_alignment"], 256)

    def test_kernel_overlay_matches_v8_kernel_maps(self) -> None:
        resolver.validate_kernel_overlay(copy.deepcopy(self.kernels))

    def test_decode_providers_do_not_alias_distinct_reduction_math(self) -> None:
        legacy = self.kernels["kernels"][
            "attention_forward_decode_head_major_gqa_flash_f16cache"
        ]
        explicit = self.kernels["kernels"][
            "attention_forward_decode_head_major_gqa_flash_f16cache_contract"
        ]
        self.assertEqual(set(legacy["supported_reductions"]), {"fp32_online"})
        self.assertEqual(
            set(explicit["supported_reductions"]),
            {"f16_online_fp32_merge"},
        )
        self.assertNotEqual(
            legacy["supported_reductions"]["fp32_online"]["function"],
            explicit["supported_reductions"]["f16_online_fp32_merge"]["function"],
        )

    def test_kernel_overlay_rejects_function_drift(self) -> None:
        kernels = copy.deepcopy(self.kernels)
        kernels["kernels"]["attention_forward_decode_head_major_gqa_flash_f16cache_contract"][
            "supported_reductions"
        ]["f16_online_fp32_merge"]["function"] = "wrong_function"
        with self.assertRaisesRegex(resolver.ContractError, "kernel map names"):
            resolver.validate_kernel_overlay(kernels)

    def test_topology_route_rejects_call_abi_drift(self) -> None:
        kernel_id = (
            "attention_forward_causal_head_major_gqa_prefill_append_"
            "f16cache_flash_auto_qtile64"
        )
        kernels = copy.deepcopy(self.kernels)
        kernel_map = resolver.load_json(
            V8_ROOT / "kernel_maps" / f"{kernel_id}.json"
        )
        route_workers = next(
            param
            for param in kernel_map["call_abi"]["params"]
            if param["name"] == "route_workers"
        )
        route_workers["source"] = "const:8"
        with tempfile.TemporaryDirectory(prefix="cke-v8-attention-route-") as tmp:
            map_path = Path(tmp) / f"{kernel_id}.json"
            map_path.write_text(json.dumps(kernel_map), encoding="utf-8")
            kernels["kernels"][kernel_id]["base_kernel_map"] = str(map_path)
            with self.assertRaisesRegex(
                resolver.ContractError,
                "runtime route is not ABI-owned by its map",
            ):
                resolver.validate_kernel_overlay(kernels)

    def test_decode_bringup_resolves_requested_contract(self) -> None:
        result = self.resolve("decode")
        self.assertEqual(result["reduction"]["id"], "f16_online_fp32_merge")
        self.assertEqual(
            result["kernel"]["id"],
            "attention_forward_decode_head_major_gqa_flash_f16cache_contract",
        )
        self.assertTrue(result["kernel"]["explicit_selector"])
        self.assertNotIn("kernel uses legacy implicit selection", result["production_blockers"])

    def test_qwen35_decode_resolves_valid_rounded_extent_contract(self) -> None:
        result = self.resolve_qwen35("decode")
        self.assertEqual(
            result["reduction"]["id"],
            "f16_online_fp32_merge",
        )
        self.assertEqual(
            result["kernel"]["selector"],
            "CK_ATTN_REDUCTION_F16_ONLINE_FP32_MERGE",
        )
        self.assertEqual(
            result["reduction"]["semantics"]["partition"]["extent_source"],
            "valid_kv_rounded",
        )
        kernel_map = resolver.load_json(
            V8_ROOT
            / "kernel_maps"
            / "attention_forward_decode_head_major_gqa_flash_f16cache_contract.json"
        )
        reduction_arg = next(
            arg
            for arg in kernel_map["call_abi"]["params"]
            if arg["name"] == "reduction"
        )
        self.assertEqual(reduction_arg["source"], "resolved:kernel_selector")

    def test_prefill_bringup_resolves_separately(self) -> None:
        result = resolver.resolve_contract(
            copy.deepcopy(self.vision_circuit),
            copy.deepcopy(self.contracts),
            copy.deepcopy(self.kernels),
            operation="vision_encoder.attention",
            phase="prefill",
            mode="bringup",
            source_circuit_path=self.vision_circuit_path,
        )
        self.assertEqual(
            result["reduction"]["id"],
            "f16_kv_tiled64_fp32_softmax_fp64_sum_update",
        )
        self.assertEqual(
            result["kernel"]["id"],
            "attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided",
        )
        self.assertEqual(result["production_blockers"], [])
        self.assertEqual(result["phase"], "prefill")
        query_tiling = result["implementation"]["query_tiling"]
        self.assertEqual(query_tiling["selection"], "token_threshold")
        self.assertEqual(query_tiling["numerical_effect"], "none")
        self.assertEqual(query_tiling["key_tile_size"], 64)
        self.assertEqual(
            [(route["min_tokens"], route["max_tokens"], route["query_tile_size"])
             for route in query_tiling["routes"]],
            [(1, 1535, 64), (1536, None, 336)],
        )

    def test_query_tiling_rejects_shape_coverage_gap(self) -> None:
        kernels = copy.deepcopy(self.kernels)
        implementation = kernels["kernels"][
            "attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided"
        ]["implementation"]
        implementation["query_tiling"]["routes"][1]["min_tokens"] = 1537
        with self.assertRaisesRegex(
            resolver.ContractError,
            "HARD CONTRACT FAULT:.*query-tile coverage gap or overlap",
        ):
            resolver.validate_kernel_overlay(kernels)

    def test_qwen3vl_prefill_resolves_segmented_append_provider(self) -> None:
        result = self.resolve("prefill")
        self.assertEqual(
            result["kernel"]["id"],
            "attention_forward_causal_head_major_gqa_prefill_append_f16cache_flash_auto_qtile64",
        )
        self.assertEqual(
            result["requirements"]["execution.prefill_batching"],
            "segmented_append",
        )
        self.assertEqual(result["requirements"]["tensor.kv.dtype"], "fp16")
        self.assertEqual(
            result["reduction"]["id"],
            "f16_flash_auto_qtile64",
        )
        self.assertEqual(
            result["kernel"]["selector"],
            "CK_ATTN_REDUCTION_F16_FLASH_AUTO_QTILE64",
        )
        threading = self.kernels["kernels"][result["kernel"]["id"]][
            "implementation"
        ]["threading"]
        self.assertEqual(threading["runtime"], "ck_threadpool")
        self.assertEqual(
            threading["work_partition"],
            ["independent_query_blocks", "cooperative_kv_groups"],
        )
        routing = self.kernels["kernels"][result["kernel"]["id"]][
            "implementation"
        ]["work_partition_routing"]
        self.assertEqual(routing["selection"], "runtime_shape_topology")
        self.assertEqual(
            routing["routes"][0]["predicate"],
            {
                "num_heads": 24,
                "num_kv_heads": 4,
                "head_dim": 256,
                "query_tokens": 4096,
                "min_kv_tokens": 8192,
                "workers": 16,
            },
        )
        self.assertEqual(
            routing["routes"][0]["configuration"],
            {"query_tile_size": 128, "concurrent_query_tiles": 4},
        )
        self.assertEqual(threading["dispatch"], ["ck_threadpool_dispatch_n"])

    def test_qwen3vl_decode_keeps_worker_split_reduction(self) -> None:
        result = self.resolve("decode")
        self.assertEqual(result["reduction"]["id"], "f16_online_fp32_merge")
        self.assertEqual(
            result["kernel"]["selector"],
            "CK_ATTN_REDUCTION_F16_ONLINE_FP32_MERGE",
        )

    def test_production_rejects_unvalidated_circuit_request(self) -> None:
        with self.assertRaisesRegex(resolver.ContractError, "Production contract resolution rejected"):
            self.resolve("decode", mode="production")

    def test_plain_dtype_is_not_a_reduction_contract(self) -> None:
        circuit = copy.deepcopy(self.circuit)
        circuit["required_contracts"][QWEN3VL_FP16_ATTENTION]["phases"]["decode"]["requires"][
            "numerics.attention_reduction"
        ] = "fp32"
        with self.assertRaisesRegex(resolver.ContractError, "ambiguous reduction"):
            resolver.resolve_contract(
                circuit,
                copy.deepcopy(self.contracts),
                copy.deepcopy(self.kernels),
                operation=QWEN3VL_FP16_ATTENTION,
                phase="decode",
                mode="bringup",
            )

    def test_unsupported_kernel_reduction_fails_without_fallback(self) -> None:
        circuit = copy.deepcopy(self.circuit)
        kernels = copy.deepcopy(self.kernels)
        circuit["required_contracts"][QWEN3VL_FP16_ATTENTION]["phases"]["decode"]["requires"][
            "numerics.attention_reduction"
        ] = "fp32_online"
        del kernels["kernels"]["attention_forward_decode_head_major_gqa_flash_f16cache"]
        with self.assertRaisesRegex(resolver.ContractError, "HARD CONTRACT FAULT: no kernel provides"):
            resolver.resolve_contract(
                circuit,
                copy.deepcopy(self.contracts),
                kernels,
                operation=QWEN3VL_FP16_ATTENTION,
                phase="decode",
                mode="bringup",
            )

    def test_production_accepts_only_fully_validated_explicit_route(self) -> None:
        circuit = copy.deepcopy(self.circuit)
        contracts = copy.deepcopy(self.contracts)
        kernels = copy.deepcopy(self.kernels)
        circuit["required_contracts"][QWEN3VL_FP16_ATTENTION]["phases"]["decode"]["validation"] = "validated"
        contracts["contracts"]["f16_online_fp32_merge"]["status"] = "validated"
        implementation = kernels["kernels"][
            "attention_forward_decode_head_major_gqa_flash_f16cache_contract"
        ]["supported_reductions"]["f16_online_fp32_merge"]
        implementation["status"] = "validated"
        implementation["explicit_selector"] = True
        result = resolver.resolve_contract(
            circuit,
            contracts,
            kernels,
            operation=QWEN3VL_FP16_ATTENTION,
            phase="decode",
            mode="production",
        )
        self.assertEqual(result["production_blockers"], [])

    def test_multiple_matching_kernels_fail_deterministically(self) -> None:
        kernels = copy.deepcopy(self.kernels)
        duplicate = copy.deepcopy(
            kernels["kernels"]["attention_forward_decode_head_major_gqa_flash_f16cache_contract"]
        )
        with tempfile.TemporaryDirectory(prefix="cke_v8_kernel_map_") as tmp:
            map_path = Path(tmp) / "attention_decode_duplicate.json"
            base_map = resolver.load_json(
                V8_ROOT
                / "kernel_maps"
                / "attention_forward_decode_head_major_gqa_flash_f16cache_contract.json"
            )
            base_map["id"] = "attention_decode_duplicate"
            map_path.write_text(json.dumps(base_map), encoding="utf-8")
            duplicate["base_kernel_map"] = str(map_path)
            kernels["kernels"]["attention_decode_duplicate"] = duplicate
            with self.assertRaisesRegex(resolver.ContractError, "HARD CONTRACT FAULT: multiple kernels provide"):
                resolver.resolve_contract(
                    copy.deepcopy(self.circuit),
                    copy.deepcopy(self.contracts),
                    kernels,
                    operation=QWEN3VL_FP16_ATTENTION,
                    phase="decode",
                    mode="bringup",
                )

    def test_unknown_circuit_contract_field_is_a_hard_fault(self) -> None:
        circuit = copy.deepcopy(self.circuit)
        circuit["required_contracts"][QWEN3VL_FP16_ATTENTION]["phases"]["decode"]["fallback"] = "fp32"
        with self.assertRaisesRegex(resolver.ContractError, "(?s)HARD CONTRACT FAULT.*fallback"):
            resolver.resolve_contract(
                circuit,
                copy.deepcopy(self.contracts),
                copy.deepcopy(self.kernels),
                operation=QWEN3VL_FP16_ATTENTION,
                phase="decode",
                mode="bringup",
            )

    def test_missing_template_op_binding_is_a_hard_fault(self) -> None:
        circuit = copy.deepcopy(self.circuit)
        del circuit["required_contracts"][QWEN3VL_FP16_ATTENTION]["template_ops"]
        with self.assertRaisesRegex(resolver.ContractError, "(?s)HARD CONTRACT FAULT.*template_ops"):
            resolver.resolve_contract(
                circuit,
                copy.deepcopy(self.contracts),
                copy.deepcopy(self.kernels),
                operation=QWEN3VL_FP16_ATTENTION,
                phase="decode",
                mode="bringup",
            )

    def test_unknown_reduction_semantic_is_a_hard_fault(self) -> None:
        contracts = copy.deepcopy(self.contracts)
        contracts["contracts"]["fp32_online"]["magic_accumulator"] = "fast"
        with self.assertRaisesRegex(resolver.ContractError, "(?s)HARD CONTRACT FAULT.*magic_accumulator"):
            resolver.validate_contract_registry(contracts)

    def test_unknown_kernel_capability_field_is_a_hard_fault(self) -> None:
        kernels = copy.deepcopy(self.kernels)
        kernels["kernels"]["attention_forward_decode_head_major_gqa_flash_f16cache_contract"][
            "supported_reductions"
        ]["f16_online_fp32_merge"]["allow_fallback"] = True
        with self.assertRaisesRegex(resolver.ContractError, "(?s)HARD CONTRACT FAULT.*allow_fallback"):
            resolver.validate_kernel_overlay(kernels)

    def test_hard_fault_instructs_agents_not_to_bypass(self) -> None:
        circuit = copy.deepcopy(self.circuit)
        circuit["required_contracts"][QWEN3VL_FP16_ATTENTION]["phases"]["decode"]["requires"][
            "numerics.attention_reduction"
        ] = "fp32_online"
        kernels = copy.deepcopy(self.kernels)
        del kernels["kernels"]["attention_forward_decode_head_major_gqa_flash_f16cache"]
        with self.assertRaises(resolver.ContractError) as raised:
            resolver.resolve_contract(
                circuit,
                copy.deepcopy(self.contracts),
                kernels,
                operation=QWEN3VL_FP16_ATTENTION,
                phase="decode",
                mode="bringup",
            )
        self.assertIn("Do not add a fallback", str(raised.exception))

    def test_supported_v8_circuits_resolve_without_legacy_attention(self) -> None:
        cases = (
            ("gemma3", "decoder.sliding_attention", "prefill", "attention_forward_causal_head_major_gqa_flash_strided_sliding"),
            ("gemma3", "decoder.sliding_attention", "decode", "attention_forward_decode_head_major_gqa_flash_sliding"),
            ("qwen2", "decoder.attention", "prefill", "attention_forward_causal_head_major_gqa_flash_strided"),
            ("qwen2", "decoder.attention", "decode", "attention_forward_decode_head_major_gqa_flash"),
            ("qwen3", "decoder.attention", "prefill", "attention_forward_causal_head_major_gqa_flash_strided"),
            ("qwen3", "decoder.attention", "decode", "attention_forward_decode_head_major_gqa_flash"),
            ("qwen35", "decoder.attention", "prefill", "attention_forward_causal_head_major_gqa_prefill_append_f16cache_flash_auto_qtile64"),
            ("qwen35", "decoder.attention", "decode", "attention_forward_decode_head_major_gqa_flash_f16cache_contract"),
            ("qwen35", "decoder.attention_bf16_pytorch", "prefill", "attention_forward_causal_head_major_gqa_prefill_full_bf16cache_pytorch_contract"),
            ("qwen35", "decoder.attention_bf16_pytorch", "decode", "attention_forward_decode_head_major_gqa_bf16cache_pytorch_contract"),
            ("glm4", "decoder.attention", "prefill", "attention_forward_causal_head_major_gqa_prefill_append_f16cache_single_range"),
            ("glm4", "decoder.attention", "decode", "attention_forward_decode_head_major_gqa_flash_f16cache_contract"),
            ("nemotron_h", "decoder.attention", "prefill", "attention_forward_causal_head_major_gqa_flash_strided"),
            ("nemotron_h", "decoder.attention", "decode", "attention_forward_decode_head_major_gqa_flash"),
            ("llama", "decoder.attention", "prefill", "attention_forward_causal_head_major_gqa_flash_strided_f16kv"),
            ("llama", "decoder.attention", "decode", "attention_forward_decode_head_major_gqa_flash_f16kv"),
            ("qwen3vl", QWEN3VL_FP16_ATTENTION, "prefill", "attention_forward_causal_head_major_gqa_prefill_append_f16cache_flash_auto_qtile64"),
            ("qwen3vl", QWEN3VL_FP16_ATTENTION, "decode", "attention_forward_decode_head_major_gqa_flash_f16cache_contract"),
            ("qwen3vl", "decoder.attention.bf16_pytorch", "prefill", "attention_forward_causal_head_major_gqa_prefill_full_bf16cache_pytorch_contract"),
            ("qwen3vl", "decoder.attention.bf16_pytorch", "decode", "attention_forward_decode_head_major_gqa_bf16cache_pytorch_contract"),
            ("qwen3_vl_vision", "vision_encoder.attention", "prefill", "attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided"),
        )
        for circuit_name, operation, phase, expected in cases:
            with self.subTest(circuit=circuit_name, phase=phase):
                path = V8_ROOT / "circuits" / f"{circuit_name}.json"
                result = resolver.resolve_contract(
                    resolver.load_json(path),
                    copy.deepcopy(self.contracts),
                    copy.deepcopy(self.kernels),
                    operation=operation,
                    phase=phase,
                    mode="bringup",
                    source_circuit_path=path,
                )
                self.assertEqual(result["kernel"]["id"], expected)
                self.assertIn(
                    result["implementation"]["threading"]["runtime"],
                    {"serial", "ck_threadpool", "external_blas"},
                )
                self.assertTrue(result["implementation"]["threading"]["work_partition"])

    def test_f16kv_prefill_provider_owns_head_parallel_selection(self) -> None:
        kernel = resolver.load_json(
            V8_ROOT
            / "kernel_maps"
            / "attention_forward_causal_head_major_gqa_flash_strided_f16kv.json"
        )
        self.assertEqual(kernel["operation_interface"],
                         "attention.causal_gqa.fp32_f16_rounding.v1")
        self.assertEqual(kernel["selection"]["status"], "production")
        self.assertEqual(kernel["selection"]["phases"], ["prefill"])
        self.assertEqual(kernel["implementation"]["threading"]["runtime"],
                         "ck_threadpool")
        self.assertEqual(
            kernel["implementation"]["threading"]["work_partition"],
            ["independent_heads"],
        )
        self.assertEqual(
            kernel["production"]["function"],
            "attention_forward_causal_head_major_gqa_flash_strided_f16kv_workspace",
        )
        self.assertEqual(
            kernel["reference"]["function"],
            "attention_forward_causal_head_major_gqa_flash_strided_f16kv_serial",
        )
        self.assertEqual(
            [port["name"] for port in kernel["inputs"]],
            ["q", "k_cache", "v_cache"],
        )
        ports = {port["name"]: port for port in kernel["inputs"]}
        self.assertEqual(ports["k_cache"]["layout"], "head_major_contiguous")
        self.assertEqual(ports["v_cache"]["layout"], "head_major_contiguous")
        abi = {param["name"]: param for param in kernel["call_abi"]["params"]}
        self.assertEqual(abi["kv_stride_tokens"]["source"], "dim:seq_len")
        self.assertEqual(
            kernel["supported_reductions"]["f16_online_single_range"]["status"],
            "observed",
        )

    def test_qwen35_attention_storage_selectors_use_kv_contract_metadata(self) -> None:
        circuit = resolver.load_json(V8_ROOT / "circuits" / "qwen35.json")
        contracts = circuit["required_contracts"]
        self.assertEqual(
            contracts["decoder.attention"]["selector"],
            {"config_not_equals": {"decode_kv_cache_dtype": "bf16"}},
        )
        self.assertEqual(
            contracts["decoder.attention_bf16_pytorch"]["selector"],
            {"config_equals": {"decode_kv_cache_dtype": "bf16"}},
        )

    def test_execution_schema_is_operator_generic(self) -> None:
        resolver.validate_schema(
            {
                "id": "gemm_nt_q4_k_q8_k",
                "op": "gemm",
                "contract_schema_version": 1,
                "implementation": {
                    "isa_dispatch": "runtime",
                    "threading": {
                        "runtime": "ck_threadpool",
                        "work_partition": ["output_tiles"],
                        "dispatch": ["ck_threadpool_dispatch_n"],
                        "reduction_order_effect": "none",
                    },
                },
            },
            resolver.KERNEL_EXECUTION_SCHEMA,
            "synthetic GEMM execution capability",
        )

    def test_hot_quant_gemm_and_gemv_maps_declare_threading(self) -> None:
        capabilities = resolver.load_kernel_execution_capabilities()["kernels"]
        for kernel_id in (
            "gemm_nt_q4_k_q8_k",
            "gemm_nt_q6_k_q8_k",
            "gemv_q4_k_q8_k",
            "gemv_q6_k_q8_k",
        ):
            with self.subTest(kernel=kernel_id):
                threading = capabilities[kernel_id]["implementation"]["threading"]
                self.assertEqual(threading["runtime"], "ck_threadpool")
                self.assertIn("ck_threadpool_dispatch_n", threading["dispatch"])
                self.assertEqual(threading["reduction_order_effect"], "none")
                self.assertIn(capabilities[kernel_id]["numerical_contract"], self.linear_contracts["contracts"])
                reference = capabilities[kernel_id]["reference"]
                if reference["kind"] == "scalar_contract_oracle":
                    self.assertTrue(reference["function"].endswith("_ref"))
                else:
                    self.assertEqual(reference["kind"], "llama_repacked_graph_oracle")
                    self.assertEqual(
                        capabilities[kernel_id]["production"]["reference_comparison"]["requirement"],
                        "bit_exact",
                    )
                    self.assertTrue(any(
                        item["backend"] == "llama.cpp_ggml_cpu_graph"
                        and item["status"] == "validated"
                        for item in reference["validation"]["external_oracles"]
                    ))
                self.assertTrue(capabilities[kernel_id]["production"]["threaded_function"].endswith("_parallel_dispatch"))

    def test_quantized_linear_registry_defines_complete_semantics(self) -> None:
        resolver.validate_quantized_linear_contract_registry(copy.deepcopy(self.linear_contracts))

    def test_quantized_linear_production_function_drift_is_a_hard_fault(self) -> None:
        kernel = resolver.load_json(V8_ROOT / "kernel_maps" / "gemm_nt_q4_k_q8_k.json")
        kernel["production"]["function"] = "wrong_function"
        with self.assertRaisesRegex(resolver.ContractError, "production function drifts"):
            resolver.validate_quantized_linear_kernel_capability(kernel, self.linear_contracts)

    def test_quantized_linear_missing_scalar_reference_is_a_hard_fault(self) -> None:
        kernel = resolver.load_json(V8_ROOT / "kernel_maps" / "gemv_q6_k_q8_k.json")
        del kernel["reference"]["function"]
        capability = {
            key: kernel[key]
            for key in (
                "id", "op", "contract_schema_version", "numerical_contract",
                "reference", "production", "implementation", "impl",
            )
        }
        with self.assertRaisesRegex(resolver.ContractError, "(?s)HARD CONTRACT FAULT.*At reference.*function"):
            resolver.validate_schema(
                capability,
                resolver.LINEAR_KERNEL_CAPABILITY_SCHEMA,
                "mutated Q6 GEMV capability",
            )

    def test_quantized_linear_threading_must_match_reduction_contract(self) -> None:
        kernel = resolver.load_json(V8_ROOT / "kernel_maps" / "gemv_q4_k_q8_k.json")
        kernel["implementation"]["threading"]["reduction_order_effect"] = "contract_defined"
        with self.assertRaisesRegex(resolver.ContractError, "threading contradicts"):
            resolver.validate_quantized_linear_kernel_capability(kernel, self.linear_contracts)

    def test_q6_prefill_declares_map_owned_work_partition_routes(self) -> None:
        kernel = resolver.load_json(V8_ROOT / "kernel_maps" / "gemm_nt_q6_k_q8_k.json")
        routing = kernel["implementation"]["work_partition_routing"]
        self.assertEqual(routing["selection"], "generated_runtime_dispatch")
        self.assertEqual(routing["numerical_effect"], "none")
        self.assertEqual(
            routing["dispatch_function"],
            kernel["production"]["threaded_function"],
        )
        self.assertEqual(routing["policy"], "prefill_schedule")
        resolver.validate_quantized_linear_kernel_capability(kernel, self.linear_contracts)

    def test_q6_prefill_runtime_implements_map_owned_shape_route(self) -> None:
        generated = (V8_ROOT / "src" / "ck_kernel_dispatch_policy_v8.inc").read_text(encoding="utf-8")
        source = (V8_ROOT / "src" / "ck_parallel_prefill_v8.c").read_text(encoding="utf-8")
        self.assertIn("{4, 63, 10240, 10240, 5120, 5120, 16, 64", generated)
        self.assertIn("ck_policy_gemm_nt_q6_k_q8_k_prefill_schedule", source)
        self.assertNotIn("qwen36_recurrent_qkv", source)
        self.assertNotIn("CK_ENABLE_Q6K_Q8K_2D_PREFILL", source)

    def test_q6_short_wide_prefill_uses_measured_m4_provider(self) -> None:
        generated = (V8_ROOT / "src" / "ck_kernel_dispatch_policy_v8.inc").read_text(encoding="utf-8")
        source = (V8_ROOT / "src" / "ck_parallel_prefill_v8.c").read_text(encoding="utf-8")
        self.assertIn("CK_GEMM_ROUTE_OUTPUT_TILES | CK_GEMM_ROUTE_COMPACT_M4", generated)
        self.assertIn("route->flags & CK_GEMM_ROUTE_COMPACT_M4", source)
        self.assertIn("schedule->tile_m", source)
        self.assertIn("schedule->tile_n", source)
        self.assertIn("gemm_nt_q6_k_q8_k_m4_tile", source)

    def test_q6_benchmark_defaults_to_gcc_provenance(self) -> None:
        bench_path = REPO_ROOT / "benchmarks" / "bench_q6k_prefill_tile.py"
        spec = importlib.util.spec_from_file_location("bench_q6k_prefill_tile", bench_path)
        assert spec is not None and spec.loader is not None
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(
            bench.shutil, "which", return_value="/usr/bin/gcc"
        ):
            self.assertEqual(bench._compiler(), "/usr/bin/gcc")

    def test_q6_prefill_missing_fallback_is_a_hard_fault(self) -> None:
        kernel = resolver.load_json(V8_ROOT / "kernel_maps" / "gemm_nt_q6_k_q8_k.json")
        kernel["implementation"]["runtime_dispatch"]["policies"]["prefill_schedule"]["routes"].pop()
        with self.assertRaisesRegex(resolver.ContractError, "no final fallback"):
            resolver.validate_quantized_linear_kernel_capability(kernel, self.linear_contracts)

    def test_q6_prefill_mixed_exact_and_range_bounds_are_a_hard_fault(self) -> None:
        kernel = resolver.load_json(V8_ROOT / "kernel_maps" / "gemm_nt_q6_k_q8_k.json")
        route = kernel["implementation"]["runtime_dispatch"]["policies"]["prefill_schedule"]["routes"][0]
        route["min_n"] = route["n"]
        with self.assertRaisesRegex(resolver.ContractError, "mixes exact and ranged N bounds"):
            resolver.validate_quantized_linear_kernel_capability(kernel, self.linear_contracts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
