from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / "version" / "v8" / "kernel_maps"
SCHEMA = ROOT / "version" / "v8" / "schemas" / "kernel_call_abi.schema.json"
REGISTRY = MAPS / "KERNEL_REGISTRY.json"
EXCLUDED = {"KERNEL_REGISTRY.json", "kernel_bindings.json", "kernel_bindings.overlay.json"}
BUILD_IR = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
EXPECTED_GOVERNED_MAP_COUNT = 117
EXPECTED_MAP_OWNED_ABI_COUNT = 155
GLM4_PARITY_PROVIDERS = {
    "rope_forward_qk_pairwise_llama_cpu",
    "rope_precompute_cache_llama_cpu",
}
QWEN3VL_PARITY_PROVIDERS = {
    "attention_forward_decode_head_major_gqa_flash_f16cache_contract",
    "attention_forward_causal_head_major_gqa_prefill_append_f16cache_flash_auto_qtile64",
    "attention_forward_causal_head_major_gqa_prefill_append_f16cache_single_range",
    "attention_forward_causal_head_major_gqa_prefill_full_bf16cache_pytorch_contract",
    "attention_forward_decode_head_major_gqa_bf16cache_pytorch_contract",
    "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
    "patch_projection_image_bf16_pytorch_onednn_conv3d_storage",
    "qk_norm_forward_fp64_sum",
    "rmsnorm_forward_fp64_sum",
    "recurrent_silu_forward_ggml",
    "swiglu_forward_ggml",
}


if str(BUILD_IR.parent) not in sys.path:
    sys.path.insert(0, str(BUILD_IR.parent))
SPEC = importlib.util.spec_from_file_location("build_ir_v8_call_abi_tests", BUILD_IR)
assert SPEC is not None and SPEC.loader is not None
build_ir_v8 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_ir_v8
SPEC.loader.exec_module(build_ir_v8)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


class V8KernelCallABITests(unittest.TestCase):
    def test_kernel_maps_reject_duplicate_json_keys(self) -> None:
        for path in sorted(MAPS.glob("*.json")):
            if path.name in EXCLUDED:
                continue
            with self.subTest(path=path.name):
                load_json(path)

    def test_all_contract_governed_maps_own_valid_call_abi(self) -> None:
        validator = Draft202012Validator(load_json(SCHEMA))
        governed = 0
        for path in sorted(MAPS.glob("*.json")):
            if path.name in EXCLUDED:
                continue
            doc = load_json(path)
            is_governed = bool(
                doc.get("numerical_capabilities")
                or doc.get("supported_reductions")
                or ("numerical_contract" in doc and "production" in doc)
            )
            if not is_governed:
                continue
            governed += 1
            with self.subTest(kernel=doc.get("id")):
                errors = sorted(
                    validator.iter_errors(doc.get("call_abi")),
                    key=lambda error: tuple(str(part) for part in error.absolute_path),
                )
                self.assertEqual(errors, [])
        self.assertEqual(governed, EXPECTED_GOVERNED_MAP_COUNT)

    def test_map_owned_abis_do_not_exist_in_legacy_registries(self) -> None:
        call_abis = build_ir_v8.load_kernel_call_abis()
        legacy = build_ir_v8.load_kernel_bindings()
        self.assertEqual(len(call_abis), EXPECTED_MAP_OWNED_ABI_COUNT)
        self.assertTrue(QWEN3VL_PARITY_PROVIDERS.issubset(call_abis))
        self.assertTrue(GLM4_PARITY_PROVIDERS.issubset(call_abis))
        for kernel_id, entry in call_abis.items():
            with self.subTest(kernel=kernel_id):
                self.assertNotIn(kernel_id, legacy)
                self.assertNotIn(entry["function"], legacy)

    def test_generated_registry_retains_exact_map_owned_abis(self) -> None:
        registry = {
            kernel["id"]: kernel
            for kernel in load_json(REGISTRY)["kernels"]
        }
        for kernel_id, entry in build_ir_v8.load_kernel_call_abis().items():
            with self.subTest(kernel=kernel_id):
                self.assertIn(kernel_id, registry)
                self.assertEqual(registry[kernel_id].get("call_abi"), entry["call_abi"])

    def test_quantized_prefill_weight_preparation_is_map_owned_and_registered(self) -> None:
        preparations = build_ir_v8.load_kernel_weight_preparations()
        self.assertEqual(
            set(preparations),
            {
                "gemm_nt_q4_k_q8_k",
                "gemm_nt_q5_0_q8_0",
                "gemm_nt_q5_k",
                "gemm_nt_q6_k_q8_k",
                "moe_swiglu_expert_forward_q4k_q5k_bucketed",
            },
        )
        preparation = preparations["gemm_nt_q5_0_q8_0"]
        self.assertEqual(preparation["function"], "ck_q5_0_prepare_q8_0_weight")
        self.assertEqual(preparation["arguments"], {"B": "B", "N": "N", "K": "K"})
        self.assertEqual(preparation["prepared_format"], "q8_0_exact")
        registry = {
            kernel["id"]: kernel
            for kernel in load_json(REGISTRY)["kernels"]
        }
        self.assertEqual(
            registry["gemm_nt_q5_0_q8_0"]["weight_preparation"],
            preparation,
        )
        q5_k = preparations["gemm_nt_q5_k"]
        self.assertEqual(q5_k["function"], "ck_q5_k_prepare_expanded_weight")
        self.assertEqual(q5_k["prepared_format"], "q5_k_expanded_integer_metadata_v1")
        self.assertEqual(q5_k["max_total_bytes"], 268435456)
        self.assertEqual(registry["gemm_nt_q5_k"]["weight_preparation"], q5_k)
        q5_variants = {
            variant["name"]: variant
            for variant in registry["gemm_nt_q5_k"]["impl"]["variants"]
        }
        nsplit = q5_variants["avx2_prepared_nsplit_m4"]
        self.assertEqual(nsplit["shape_constraints"]["M_min"], 64)
        self.assertEqual(nsplit["shape_constraints"]["M_max"], 256)
        self.assertEqual(nsplit["work_partition"], "independent_output_columns")
        self.assertEqual(nsplit["activation_preparation"], "q8_k_once_per_call")
        self.assertEqual(nsplit["tile_n"], 64)
        q4_k = preparations["gemm_nt_q4_k_q8_k"]
        self.assertEqual(q4_k["function"], "ck_q4k_prepare_vnni_x8_weight")
        self.assertEqual(q4_k["prepared_format"], "q4_k_packed_vnni_x8")
        self.assertEqual(q4_k["max_total_bytes"], 2147483648)
        self.assertEqual(registry["gemm_nt_q4_k_q8_k"]["weight_preparation"], q4_k)
        q6_k = preparations["gemm_nt_q6_k_q8_k"]
        self.assertEqual(q6_k["function"], "ck_q6_k_prepare_expanded_weight")
        self.assertEqual(q6_k["prepared_format"], "q6_k_expanded_integer_metadata_v1")
        self.assertEqual(q6_k["max_total_bytes"], 8589934592)
        self.assertEqual(q6_k["min_remaining_memory_bytes"], 17179869184)
        self.assertEqual(registry["gemm_nt_q6_k_q8_k"]["weight_preparation"], q6_k)
        moe = preparations["moe_swiglu_expert_forward_q4k_q5k_bucketed"]
        self.assertEqual(
            moe["function"], "ck_moe_prepare_q4k_gate_up_vnni_x8"
        )
        self.assertEqual(moe["max_total_bytes"], 16 * 1024**3)
        self.assertEqual(
            registry["moe_swiglu_expert_forward_q4k_q5k_bucketed"][
                "weight_preparation"
            ],
            moe,
        )

    def test_q5_k_activation_scratch_is_planner_owned_and_exactly_sized(self) -> None:
        q5_map = load_json(MAPS / "gemm_nt_q5_k.json")
        self.assertEqual(q5_map["impl"]["function"], "gemm_nt_q5_k_parallel_dispatch_with_scratch")
        self.assertEqual(
            [param["source"] for param in q5_map["call_abi"]["params"][-2:]],
            ["scratch:activation_q8_k", "scratch_size:activation_q8_k"],
        )
        scratch = q5_map["scratch"][0]
        self.assertEqual(
            build_ir_v8._kernel_scratch_size_bytes(
                scratch,
                {"_m": 128, "_input_dim": 1024},
                {},
            ),
            128 * (1024 // 256) * 292,
        )

        source = (ROOT / "version" / "v8" / "src" / "ck_parallel_prefill_v8.c").read_text()
        body = source.split("static void gemm_nt_q5_k_parallel_dispatch_impl", 1)[1]
        body = body.split("void gemm_nt_q5_k_parallel_dispatch_with_scratch", 1)[0]
        self.assertNotIn("malloc(", body)
        self.assertNotIn("free(", body)

    def test_block_scratch_rejects_partial_storage_blocks(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "innermost extent must be divisible"):
            build_ir_v8._kernel_scratch_size_bytes(
                {
                    "dtype": "q8_k",
                    "shape": ["M", "K"],
                    "block_elements": 256,
                    "block_bytes": 292,
                },
                {"_m": 2, "_input_dim": 257},
                {},
            )

    def test_q5_k_call_ready_ir_passes_scratch_pointer_and_capacity(self) -> None:
        function = "gemm_nt_q5_k_parallel_dispatch_with_scratch"
        lowered = {
            "config": {},
            "operations": [{
                "idx": 0,
                "kernel": "gemm_nt_q5_k",
                "function": function,
                "op": "recurrent_qkv_proj",
                "layer": 0,
                "section": "body",
                "activations": {
                    "A": {"activation_offset": 64, "buffer": "layer_input"},
                },
                "outputs": {
                    "C": {"activation_offset": 4096, "buffer": "mlp_scratch"},
                },
                "weights": {
                    "recurrent_qkv_weight": {"bump_offset": 8192, "name": "recurrent_qkv_weight"},
                },
                "scratch": [{
                    "name": "activation_q8_k",
                    "scratch_offset": 16384,
                    "size": 128 * (1024 // 256) * 292,
                }],
                "params": {"_m": 128, "_output_dim": 1024, "_input_dim": 1024},
                "resolved_contract": {
                    "function": function,
                    "kernel_id": "gemm_nt_q5_k",
                    "operation": "decoder.recurrent_qkv_projection.q5_k.prefill",
                },
            }],
        }
        call = build_ir_v8.generate_ir_lower_3(lowered, "prefill")["operations"][0]
        self.assertEqual(call["errors"], [])
        args = {arg["name"]: arg["expr"] for arg in call["args"]}
        self.assertIn("16384", args["scratch"])
        self.assertEqual(args["scratch_bytes"], str(128 * (1024 // 256) * 292))

    def test_weight_preparation_rejects_unknown_size_symbols(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cke_weight_preparation_") as td:
            root = Path(td)
            (root / "synthetic.json").write_text(
                json.dumps({
                    "id": "synthetic",
                    "weight_preparation": {
                        "function": "prepare_synthetic",
                        "arguments": {"B": "B", "N": "N"},
                        "prepared_bytes": "N * UNKNOWN",
                        "max_total_bytes": 1024,
                    },
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "unknown symbols"):
                build_ir_v8.load_kernel_weight_preparations(root)

    def test_weight_preparation_rejects_invalid_memory_reserve(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cke_weight_preparation_") as td:
            root = Path(td)
            (root / "invalid.json").write_text(
                json.dumps(
                    {
                        "id": "invalid_prepare",
                        "weight_preparation": {
                            "function": "prepare_weight",
                            "arguments": {"B": "B", "N": "N", "K": "K"},
                            "prepared_bytes": "N * K",
                            "max_total_bytes": 1024,
                            "min_remaining_memory_bytes": -1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "min_remaining_memory_bytes"):
                build_ir_v8.load_kernel_weight_preparations(root)

    def test_duplicate_map_and_legacy_ownership_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cke_call_abi_duplicate_") as td:
            root = Path(td)
            (root / "synthetic.json").write_text(
                json.dumps({
                    "id": "synthetic",
                    "impl": {"function": "synthetic_fn"},
                    "call_abi": {"version": 1, "params": []},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "legacy bindings still define"):
                build_ir_v8.load_kernel_call_abis(
                    root,
                    legacy_bindings={"synthetic_fn": {"params": []}},
                )

    def test_unknown_call_source_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cke_call_abi_source_") as td:
            root = Path(td)
            (root / "synthetic.json").write_text(
                json.dumps({
                    "id": "synthetic",
                    "impl": {"function": "synthetic_fn"},
                    "call_abi": {
                        "version": 1,
                        "params": [{"name": "x", "source": "guessed:model_default"}],
                    },
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "unsupported source"):
                build_ir_v8.load_kernel_call_abis(root, legacy_bindings={})

    def test_resolved_selector_requires_resolved_contract_metadata(self) -> None:
        lowered = {
            "config": {},
            "operations": [{
                "idx": 0,
                "kernel": "attention_forward_decode_head_major_gqa_flash_f16cache_contract",
                "function": "attention_forward_decode_head_major_gqa_flash_f16cache_contract",
                "op": "attn",
                "layer": 0,
                "section": "body",
                "activations": {},
                "outputs": {},
                "scratch": [],
                "params": {},
                "resolved_contract": {
                    "function": "attention_forward_decode_head_major_gqa_flash_f16cache_contract",
                    "kernel_id": "attention_forward_decode_head_major_gqa_flash_f16cache_contract",
                },
            }],
        }
        call_ir = build_ir_v8.generate_ir_lower_3(lowered, "decode")
        errors = call_ir["operations"][0]["errors"]
        self.assertTrue(any("no explicit kernel selector" in error for error in errors))

    def test_resolved_selector_is_emitted_as_call_expression(self) -> None:
        lowered = {
            "config": {},
            "operations": [{
                "idx": 0,
                "kernel": "attention_forward_decode_head_major_gqa_flash_f16cache_contract",
                "function": "attention_forward_decode_head_major_gqa_flash_f16cache_contract",
                "op": "attn",
                "layer": 0,
                "section": "body",
                "activations": {},
                "outputs": {},
                "scratch": [],
                "params": {},
                "resolved_contract": {
                    "function": "attention_forward_decode_head_major_gqa_flash_f16cache_contract",
                    "kernel_id": "attention_forward_decode_head_major_gqa_flash_f16cache_contract",
                    "selector": "CK_ATTN_REDUCTION_F16_ONLINE_FP32_MERGE",
                },
            }],
        }
        call_ir = build_ir_v8.generate_ir_lower_3(lowered, "decode")
        reduction = next(
            arg
            for arg in call_ir["operations"][0]["args"]
            if arg["name"] == "reduction"
        )
        self.assertEqual(reduction["source"], "resolved:kernel_selector")
        self.assertEqual(
            reduction["expr"], "CK_ATTN_REDUCTION_F16_ONLINE_FP32_MERGE"
        )

    def test_malformed_optional_call_metadata_is_a_hard_failure(self) -> None:
        bad_params = [
            {"name": "x", "source": "null:guessed"},
            {"name": "x", "source": "null", "cast": ""},
            {"name": "x", "source": "null", "alt": ["x", "x"]},
            {"name": "x", "source": "null", "ports": []},
            {"name": "x", "source": "null", "ports": ["input"]},
            {"name": "x", "source": "null", "ports": ["input:x", "input:x"]},
        ]
        for index, param in enumerate(bad_params):
            with self.subTest(param=param), tempfile.TemporaryDirectory(
                prefix=f"cke_call_abi_metadata_{index}_"
            ) as td:
                root = Path(td)
                (root / "synthetic.json").write_text(
                    json.dumps({
                        "id": "synthetic",
                        "impl": {"function": "synthetic_fn"},
                        "call_abi": {"version": 1, "params": [param]},
                    }),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "HARD CALL ABI FAULT"):
                    build_ir_v8.load_kernel_call_abis(root, legacy_bindings={})

    def test_resolved_operation_cannot_fall_back_to_legacy_binding(self) -> None:
        lowered = {
            "config": {},
            "operations": [{
                "idx": 0,
                "kernel": "im2patch",
                "function": "im2patch",
                "op": "patchify",
                "layer": -1,
                "section": "header",
                "resolved_contract": {"function": "im2patch", "kernel_id": "im2patch"},
            }],
        }
        call_ir = build_ir_v8.generate_ir_lower_3(lowered, "prefill")
        self.assertIn("missing map-owned call_abi", call_ir["operations"][0]["errors"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
