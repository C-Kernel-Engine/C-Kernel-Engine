#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEGEN_PREFILL_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_prefill_v8.py"


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codegen_prefill_v8 = _load_module("codegen_prefill_v8_tests", CODEGEN_PREFILL_PATH)


class TestV8PrefillCodegen(unittest.TestCase):
    def test_hyper_prefill_exports_wide_boundaries_and_last_rows(self):
        args = {"rows": "num_tokens", "streams": "4", "hidden_dim": "2560",
                "dynamic_dim": "320", "normalized_scratch": "norm",
                "dynamic_scratch": "dynamic", "mix_scratch": "gate",
                "mixed_output": "mixed", "injection_output": "NULL", "output": "wide"}
        for op_name, label in [("hyper_mix_attn", "attn_hyper_norm"),
                               ("hyper_mix_mlp", "mlp_hyper_norm"),
                               ("hyper_mix_final", "final_hyper_norm"),
                               ("hyper_stream_expand", "hyper_stream"),
                               ("hyper_inject_attn", "after_attn_hyper"),
                               ("hyper_inject_mlp", "layer_out")]:
            with self.subTest(op=op_name):
                op = {"function": "test_provider", "op": op_name, "layer": 7,
                      "args": [{"name": k, "expr": v} for k, v in args.items()]}
                emitted = codegen_prefill_v8.emit_prefill_op(op, 0, {})
                self.assertIn(f'"{label}"', emitted)
                self.assertIn(f'"{label}_last"', emitted)
                self.assertIn("(4) * (2560)", emitted)
                self.assertNotIn('"final_injection_weights"', emitted)

    @staticmethod
    def _q4_segmented_projection_op(op_name: str = "recurrent_gate_proj") -> dict:
        return {
            "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
            "op": op_name,
            "layer": 0,
            "resolved_execution": {
                "numerical_contract": "q4_k_x_q8_k_repacked_matmul_fp32",
                "implementation": {
                    "weight_storage": {
                        "format": "q4_k",
                        "block_elements": 256,
                        "block_bytes": 144,
                    },
                    "activation_storage": {
                        "format": "q8_k",
                        "block_elements": 256,
                    },
                    "diagnostic_providers": {
                        "fp32_activation": "gemm_nt_q4_k",
                        "row_quantized": "gemv_q4_k_q8_k",
                    },
                    "segmented_row_provider": {
                        "function": "gemm_nt_q4_k_q8_k_segmented_pairwise_split_min_parallel_dispatch",
                        "segment_lengths_dtype": "i32",
                        "boundary_semantics": "restart_row_group_at_each_segment",
                        "fallback": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
                    },
                },
            },
            "args": [
                {"name": "A", "source": "activation:a", "expr": "A"},
                {"name": "B", "source": "weight:_first_weight", "expr": "B"},
                {"name": "bias", "source": "weight_f:_bias", "expr": "NULL"},
                {"name": "C", "source": "output:c", "expr": "C"},
                {"name": "M", "source": "dim:_m", "expr": "1035"},
                {"name": "N", "source": "dim:_output_dim", "expr": "6144"},
                {"name": "K", "source": "dim:_input_dim", "expr": "5120"},
            ],
        }

    @staticmethod
    def _segment_preserving_config() -> dict:
        return {
            "multimodal_bridge_contract": {
                "prefix_policy": "mixed_visual_text_prefill",
                "prefill_schedule": {
                    "projection_row_group_boundaries": {
                        "policy": "restart_each_segment",
                        "operations": [
                            "recurrent_qkv_proj",
                            "recurrent_gate_proj",
                            "mlp_gate_up",
                        ],
                    }
                },
            }
        }

    def test_q4_projection_uses_map_owned_runtime_segment_provider(self) -> None:
        emitted = codegen_prefill_v8.emit_prefill_op(
            self._q4_segmented_projection_op(),
            6,
            self._segment_preserving_config(),
            segment_plan_available=True,
        )
        self.assertIn(
            "gemm_nt_q4_k_q8_k_segmented_pairwise_split_min_parallel_dispatch(",
            emitted,
        )
        self.assertIn("g_multimodal_prefill_segment_lengths", emitted)
        self.assertIn("g_multimodal_prefill_num_segments", emitted)
        self.assertIn("gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(", emitted)

    def test_q4_projection_without_circuit_contract_keeps_unified_provider(self) -> None:
        emitted = codegen_prefill_v8.emit_prefill_op(
            self._q4_segmented_projection_op(), 6, {}
        )
        self.assertNotIn("segmented_pairwise_split_min", emitted)
        self.assertIn("gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(", emitted)

    def test_q6_projection_uses_map_owned_runtime_segment_provider(self) -> None:
        op = self._q4_segmented_projection_op("recurrent_qkv_proj")
        op["function"] = "gemm_nt_q6_k_q8_k"
        op["resolved_execution"]["implementation"].update(
            {
                "weight_storage": {
                    "format": "q6_k",
                    "block_elements": 256,
                    "block_bytes": 210,
                },
                "diagnostic_providers": {
                    "fp32_activation": "gemm_nt_q6_k",
                    "row_quantized": "gemv_q6_k_q8_k",
                },
                "segmented_row_provider": {
                    "function": "gemm_nt_q6_k_q8_k_segmented_parallel_dispatch",
                    "segment_lengths_dtype": "i32",
                    "boundary_semantics": "restart_row_group_at_each_segment",
                    "fallback": "gemm_nt_q6_k_q8_k_parallel_dispatch",
                },
            }
        )
        op["resolved_execution"]["numerical_contract"] = (
            "q6_k_weight_q8_k_input_llama_fp32_output"
        )

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            4,
            self._segment_preserving_config(),
            segment_plan_available=True,
        )

        self.assertIn("gemm_nt_q6_k_q8_k_segmented_parallel_dispatch(", emitted)
        self.assertIn("g_multimodal_prefill_segment_lengths", emitted)
        self.assertIn("gemm_nt_q6_k_q8_k(", emitted)

    def test_attention_uses_map_owned_segmented_query_provider(self) -> None:
        base_arg_names = [
            "q", "k_cache", "v_cache", "output", "num_heads", "num_kv_heads",
            "q_tokens", "past_tokens", "cache_capacity", "head_dim",
            "aligned_head_dim", "reduction", "token_workspace",
            "token_workspace_bytes",
        ]
        route_arg_names = [
            "gqa_workspace", "gqa_workspace_bytes", "route_num_heads",
            "route_num_kv_heads", "route_head_dim", "route_query_tokens",
            "route_min_kv_tokens", "route_workers", "route_query_tile_size",
            "route_concurrent_query_tiles",
        ]
        op = {
            "function": "attention_forward_causal_head_major_gqa_prefill_append_f16cache_auto_workspace",
            "op": "attn",
            "layer": 3,
            "resolved_execution": {
                "implementation": {
                    "segmented_query_provider": {
                        "function": "attention_forward_causal_head_major_gqa_prefill_segmented_f16cache_contract_workspace",
                        "base_args": base_arg_names,
                        "segment_lengths_dtype": "i32",
                        "boundary_semantics": "restart_query_tile_policy_at_each_segment",
                        "fallback": "reject_invalid_plan",
                    }
                }
            },
            "args": [
                {"name": name, "source": f"test:{name}", "expr": name}
                for name in base_arg_names + route_arg_names
            ],
        }
        emitted = codegen_prefill_v8.emit_prefill_op(
            op, 91, {}, segment_plan_available=True
        )
        self.assertIn(
            "attention_forward_causal_head_major_gqa_prefill_segmented_f16cache_contract_workspace(",
            emitted,
        )
        self.assertIn("g_multimodal_prefill_segment_lengths", emitted)
        self.assertIn("g_multimodal_prefill_num_segments", emitted)
        self.assertIn(
            "attention_forward_causal_head_major_gqa_prefill_append_f16cache_auto_workspace(",
            emitted,
        )
        segmented_call = emitted.split("} else {", 1)[0]
        self.assertNotIn("gqa_workspace,", segmented_call)
        self.assertNotIn("route_num_heads,", segmented_call)

    def test_kv_transpose_exports_attention_consumed_layout(self) -> None:
        config = {"num_kv_heads": 8, "head_dim": 128, "context_len": 1034}
        for is_k, label in ((True, "k_head_major"), (False, "v_head_major")):
            with self.subTest(label=label):
                emitted = codegen_prefill_v8.emit_prefill_op(
                    {
                        "function": "transpose_inplace",
                        "op": "transpose_kv_to_head_major",
                        "layer": 3,
                        "section": "body",
                        "_is_k": is_k,
                        "_num_kv_heads": 8,
                        "_head_dim": 128,
                    },
                    17,
                    config,
                )
                copy_back = "memcpy(buf, _temp_buf, (size_t)Hkv * num_tokens * D * sizeof(float));"
                export = f'"{label}"'
                self.assertIn(copy_back, emitted)
                self.assertIn(export, emitted)
                self.assertLess(emitted.index(copy_back), emitted.index(export))
                self.assertIn("Hkv * num_tokens * D", emitted)

    def test_last_logits_preserves_resolved_exact_gemm_provider(self) -> None:
        function = "gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage"
        op = {
            "function": function,
            "op": "logits",
            "layer": -1,
            "call_abi": {
                "version": 1,
                "owner": "kernel_map",
                "kernel_id": function,
                "last_token_dispatch": "preserve_provider",
            },
            "args": [
                {
                    "name": "A",
                    "source": "activation:a",
                    "expr": "(const float*)(model->bump + A_MAIN_STREAM)",
                },
                {
                    "name": "B",
                    "source": "weight:_first_weight",
                    "expr": "(const void*)(model->bump + W_LM_HEAD)",
                },
                {
                    "name": "C",
                    "source": "output:c",
                    "expr": "(float*)(model->bump + A_LOGITS)",
                },
                {"name": "M", "source": "dim:_m", "expr": "num_tokens"},
                {"name": "N", "source": "dim:_output_dim", "expr": "151936"},
                {"name": "K", "source": "dim:_input_dim", "expr": "4096"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            758,
            {"embed_dim": 4096, "vocab_size": 151936, "logits_layout": "last"},
        )

        self.assertIn("logits (last-only exact GEMM contract)", emitted)
        self.assertIn(function + "(", emitted)
        self.assertIn("(size_t)(num_tokens - 1) * 4096", emitted)
        self.assertNotIn(
            "gemv_bf16_pytorch_onednn_brgemm_bf16_storage",
            emitted,
        )

    def test_last_logits_quantized_provider_uses_q8_k_row_gemv(self) -> None:
        op = {
            "function": "gemm_nt_q6_k_q8_k",
            "op": "logits",
            "layer": -1,
            "call_abi": {
                "version": 1,
                "owner": "kernel_map",
                "kernel_id": "gemm_nt_q6_k_q8_k",
            },
            "resolved_execution": {
                "kernel_id": "gemm_nt_q6_k_q8_k",
                "implementation": {
                    "weight_storage": {
                        "format": "q6_k",
                        "block_elements": 256,
                        "block_bytes": 210,
                    },
                    "activation_storage": {
                        "format": "q8_k",
                        "block_elements": 256,
                    },
                    "diagnostic_providers": {
                        "fp32_activation": "gemm_nt_q6_k",
                        "row_quantized": "gemv_q6_k_q8_k",
                    },
                },
                "numerical_contract": "q6_k_x_q8_k_fp32_block_order",
            },
            "args": [
                {
                    "name": "A",
                    "source": "activation:a",
                    "expr": "(const void*)(model->bump + A_LAYER_INPUT)",
                },
                {
                    "name": "B",
                    "source": "weight:_first_weight",
                    "expr": "(const void*)(model->bump + W_OUTPUT_WEIGHT)",
                },
                {
                    "name": "bias",
                    "source": "weight_f:_bias",
                    "expr": "NULL",
                },
                {
                    "name": "C",
                    "source": "output:c",
                    "expr": "(float*)(model->bump + A_LOGITS)",
                },
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            921,
            {"embed_dim": 4096, "vocab_size": 151936, "logits_layout": "last"},
        )

        self.assertIn("logits (last-only)", emitted)
        self.assertIn("gemv_q6_k_q8_k(", emitted)
        self.assertIn("(4096 / QK_K) * sizeof(block_q8_K)", emitted)
        self.assertNotIn("(const float*)", emitted.split("ck_debug_export_hidden", 1)[0])
        self.assertNotIn("last-only exact GEMM contract", emitted)

    def test_last_logits_generic_bf16_preserves_registered_gemm(self) -> None:
        op = {
            "function": "gemm_nt_bf16",
            "op": "logits",
            "layer": -1,
            "call_abi": {
                "version": 1,
                "owner": "kernel_map",
                "kernel_id": "gemm_nt_bf16",
                "last_token_dispatch": "preserve_provider",
            },
            "args": [
                {
                    "name": "A",
                    "source": "activation:a",
                    "expr": "(const float*)(model->bump + A_MAIN_STREAM)",
                },
                {
                    "name": "B",
                    "source": "weight:_first_weight",
                    "expr": "(const void*)(model->bump + W_LM_HEAD)",
                },
                {
                    "name": "bias",
                    "source": "weight_f:_bias",
                    "expr": "NULL",
                },
                {
                    "name": "C",
                    "source": "output:c",
                    "expr": "(float*)(model->bump + A_LOGITS)",
                },
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            460,
            {"embed_dim": 2048, "vocab_size": 163840, "logits_layout": "last"},
        )

        self.assertIn("gemm_nt_bf16(", emitted)
        self.assertIn("(size_t)(num_tokens - 1) * 2048", emitted)
        self.assertNotIn("gemv_bf16(", emitted)

    def test_last_logits_q4_provider_uses_registered_row_gemv(self) -> None:
        op = {
            "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
            "op": "logits",
            "layer": -1,
            "call_abi": {
                "version": 1,
                "owner": "kernel_map",
                "kernel_id": "gemm_nt_q4_k_q8_k",
            },
            "resolved_execution": {
                "kernel_id": "gemm_nt_q4_k_q8_k",
                "implementation": {
                    "weight_storage": {
                        "format": "q4_k",
                        "block_elements": 256,
                        "block_bytes": 144,
                    },
                    "activation_storage": {
                        "format": "q8_k",
                        "block_elements": 256,
                    },
                    "diagnostic_providers": {
                        "fp32_activation": "gemm_nt_q4_k",
                        "row_quantized": "gemv_q4_k_q8_k",
                    },
                },
                "numerical_contract": "q4_k_x_q8_k_repacked_matmul_fp32",
            },
            "args": [
                {
                    "name": "A",
                    "source": "activation:a",
                    "expr": "(const void*)(model->bump + A_LAYER_INPUT)",
                },
                {
                    "name": "B",
                    "source": "weight:_first_weight",
                    "expr": "(const void*)(model->bump + W_TOKEN_EMB)",
                },
                {
                    "name": "C",
                    "source": "output:c",
                    "expr": "(float*)(model->bump + A_LOGITS)",
                },
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            1222,
            {"embed_dim": 2560, "vocab_size": 262144, "logits_layout": "last"},
        )

        self.assertIn("gemv_q4_k_q8_k(", emitted)
        self.assertNotIn("gemv_q4_k_q8_k_pairwise_split_min_parallel_dispatch", emitted)

    def test_last_logits_legacy_provider_keeps_bounded_row_dispatch(self) -> None:
        op = {
            "function": "gemm_nt_q8_0_q8_0",
            "op": "logits",
            "layer": -1,
            "call_abi": {
                "version": 0,
                "owner": "legacy_compatibility",
                "kernel_id": "gemm_nt_q8_0_q8_0",
                "source_file": "kernel_bindings*.json",
            },
            "args": [
                {
                    "name": "A",
                    "source": "activation:a",
                    "expr": "(const void*)(model->bump + A_MAIN_STREAM)",
                },
                {
                    "name": "B",
                    "source": "weight:_first_weight",
                    "expr": "(const void*)(model->bump + W_LM_HEAD)",
                },
                {
                    "name": "C",
                    "source": "output:c",
                    "expr": "(float*)(model->bump + A_LOGITS)",
                },
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            460,
            {"embed_dim": 2048, "vocab_size": 163840, "logits_layout": "last"},
        )

        self.assertIn("gemv_q8_0_q8_0(", emitted)
        self.assertNotIn("gemm_nt_q8_0_q8_0(", emitted)

    def test_residual_save_exports_prefill_layer_input_before_normalization(self) -> None:
        op = {
            "function": "memcpy",
            "op": "residual_save",
            "layer": 0,
            "op_instance_idx": 0,
            "args": [
                {"name": "dst", "expr": "RESIDUAL"},
                {"name": "src", "expr": "LAYER_INPUT"},
                {"name": "size", "source": "dim:_memcpy_bytes", "expr": "4096"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(op, 1, {"embed_dim": 4096})

        self.assertIn(
            'ck_debug_export_hidden(model, 0, "layer_input", '
            "(const float*)LAYER_INPUT, (num_tokens) * (EMBED_DIM))",
            emitted,
        )

        emitted_dump = codegen_prefill_v8.emit_prefill_op(
            op, 1, {"embed_dim": 4096}, dump=True
        )
        self.assertIn(
            'ck_dump_tensor((float*)LAYER_INPUT, 0, "layer_input", '
            "(num_tokens) * (4096))",
            emitted_dump,
        )

        op["op_instance_idx"] = 1
        emitted_after_attention = codegen_prefill_v8.emit_prefill_op(
            op, 16, {"embed_dim": 4096}
        )
        self.assertIn('"after_attn"', emitted_after_attention)
        self.assertNotIn('"layer_input"', emitted_after_attention)

    @staticmethod
    def _bridge_embedding_ops() -> list[dict]:
        return [{
            "op": "dense_embedding_lookup",
            "function": "embedding_forward_bf16_fp32",
            "args": [
                {"name": "token_ids", "expr": "TOKENS"},
                {"name": "token_embeddings", "expr": "WEIGHTS"},
                {"name": "output", "expr": "OUT"},
                {"name": "vocab_size", "expr": "VOCAB_SIZE"},
                {"name": "embed_dim", "expr": "EMBED_DIM"},
                {"name": "aligned_embed_dim", "expr": "EMBED_DIM"},
            ],
        }]

    def test_unified_mixed_bridge_emits_one_full_prefill(self) -> None:
        config = {
            "embed_dim": 16,
            "num_deepstack_layers": 3,
            "context_length": 32,
            "multimodal_bridge_contract": {
                "prefix_policy": "mixed_visual_text_prefill",
                "prefill_batching": "unified_mixed",
                "providers": {
                    "prefix_insert": {
                        "resolved_function": "ck_multimodal_prefix_insert_f32",
                    },
                    "position_builder": {
                        "resolved_function": "ck_multimodal_mrope_positions_2d",
                    },
                },
                "prefill_schedule": {
                    "segments": ["text_before", "visual", "text_after"],
                    "cache_transition": "single_pass",
                    "position_transition": "explicit_full_sequence",
                    "position_transform": {
                        "kernel_id": "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
                        "contract_id": "text_imrope_positions_bf16_input_pytorch_bf16_compute_bf16_output",
                        "resolved_function": "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
                    },
                    "deepstack_injection": {
                        "kernel_id": "ck_residual_add_token_major_bf16_storage",
                        "contract_id": "residual_add_bf16_input_fp32_add_bf16_output",
                        "resolved_function": "ck_residual_add_token_major_bf16_storage",
                        "target": "visual_rows_after_decoder_layer",
                        "layers_from_config": "num_deepstack_layers",
                    },
                },
            },
        }
        emitted = codegen_prefill_v8.emit_multimodal_bridge_api(
            self._bridge_embedding_ops(), config
        )
        self.assertEqual(
            emitted.count("ck_prefill_from_embedded(g_model, total_tokens);"), 2
        )
        self.assertNotIn(
            "ck_prefill_from_embedded_range(g_model, prefix_tokens", emitted
        )
        helpers = codegen_prefill_v8._emit_multimodal_prefill_bridge_helpers(
            config, "mrope_qk_text_imrope_bf16_pytorch_storage"
        )
        self.assertIn("mrope_qk_text_imrope_positions_bf16_pytorch_storage(q, k,", helpers)
        self.assertIn("ck_residual_add_token_major_bf16_storage(dst_row, src, dst_row,", helpers)
        self.assertNotIn("mrope_qk_imrope_positions(q, k,", helpers)

    def test_unified_mixed_bridge_rejects_unresolved_position_provider(self) -> None:
        config = {
            "embed_dim": 16,
            "num_deepstack_layers": 3,
            "multimodal_bridge_contract": {
                "prefix_policy": "mixed_visual_text_prefill",
                "prefill_batching": "unified_mixed",
                "prefill_schedule": {
                    "segments": ["text_before", "visual", "text_after"],
                    "cache_transition": "single_pass",
                    "position_transition": "explicit_full_sequence",
                },
            },
        }
        with self.assertRaisesRegex(RuntimeError, "positions-aware M-RoPE provider"):
            codegen_prefill_v8._emit_multimodal_prefill_bridge_helpers(
                config, "mrope_qk_text_imrope_bf16_pytorch_storage"
            )

    def test_segmented_bridge_retains_three_cache_preserving_prefills(self) -> None:
        config = {
            "embed_dim": 16,
            "num_deepstack_layers": 3,
            "context_length": 32,
            "multimodal_bridge_contract": {
                "prefix_policy": "mixed_visual_text_prefill",
                "prefill_batching": "segmented_append",
                "providers": {
                    "prefix_insert": {
                        "resolved_function": "ck_multimodal_prefix_insert_f32",
                    },
                },
                "prefill_schedule": {
                    "segments": ["text_before", "visual", "text_after"],
                    "cache_transition": "append_preserve",
                    "position_transition": "segment_defined",
                },
            },
        }
        emitted = codegen_prefill_v8.emit_multimodal_bridge_api(
            self._bridge_embedding_ops(), config
        )
        self.assertIn(
            "ck_prefill_from_embedded_range(g_model, prefix_tokens", emitted
        )
        self.assertEqual(
            emitted.count("ck_prefill_from_embedded(g_model, total_tokens);"), 1
        )

    def test_qwen35_post_attention_prefill_exports_full_checkpoint_extents(self) -> None:
        cases = [
            (
                {
                    "function": "attn_gate_sigmoid_mul_forward",
                    "op": "attn_gate_sigmoid_mul",
                    "layer": 3,
                    "args": [
                        {"name": "out", "expr": "ATTN"},
                        {"name": "rows", "expr": "1034"},
                        {"name": "num_heads", "expr": "8"},
                        {"name": "state_dim", "expr": "256"},
                    ],
                },
                '"attn_out", (const float*)ATTN, (8) * (num_tokens) * (256)',
            ),
            (
                {
                    "function": "rmsnorm_forward_llama_production",
                    "op": "post_attention_norm",
                    "layer": 3,
                    "args": [
                        {"name": "output", "expr": "NORM"},
                        {"name": "tokens", "expr": "1034"},
                        {"name": "d_model", "expr": "1024"},
                    ],
                },
                '"post_attn_norm", (const float*)NORM, (num_tokens) * (1024)',
            ),
            (
                {
                    "function": "swiglu_forward_ggml",
                    "op": "silu_mul",
                    "layer": 3,
                    "args": [
                        {"name": "output", "expr": "MLP"},
                        {"name": "tokens", "expr": "1034"},
                        {"name": "dim", "expr": "3584"},
                    ],
                },
                '"mlp_swiglu", (const float*)MLP, (num_tokens) * (3584)',
            ),
        ]
        for op, expected in cases:
            with self.subTest(op=op["op"]):
                emitted = codegen_prefill_v8.emit_prefill_op(op, 1, {"embed_dim": 1024})
                self.assertIn(expected, emitted)

    def test_registered_fp16_cache_batch_store_uses_physical_call_arguments(self) -> None:
        op = {
            "function": "kv_cache_store_batch_f16",
            "op": "kv_cache_store_batch_f16",
            "layer": 3,
            "section": "body",
            "args": [
                {"name": "kv_cache_k", "source": "runtime:kv_cache_k_layer_f16", "expr": "K_CACHE"},
                {"name": "kv_cache_v", "source": "runtime:kv_cache_v_layer_f16", "expr": "V_CACHE"},
                {"name": "k", "source": "activation:k_src", "expr": "K_SCRATCH"},
                {"name": "v", "source": "activation:v_src", "expr": "V_SCRATCH"},
                {"name": "start_pos", "source": "runtime:prefill_start_pos", "expr": "model->pos"},
                {"name": "num_tokens", "source": "dim:seq_len", "expr": "1034"},
                {"name": "num_kv_heads", "source": "dim:num_kv_heads", "expr": "8"},
                {"name": "head_dim", "source": "dim:head_dim", "expr": "64"},
                {"name": "max_seq_len", "source": "dim:max_seq_len", "expr": "1034"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            11,
            {
                "decode_kv_cache_dtype": "fp16",
                "num_kv_heads": 8,
                "head_dim": 64,
                "context_len": 1034,
            },
        )

        self.assertIn("kv_cache_store_batch_f16(", emitted)
        self.assertIn("K_CACHE", emitted)
        self.assertIn("V_CACHE", emitted)
        self.assertIn("K_SCRATCH", emitted)
        self.assertIn("V_SCRATCH", emitted)
        self.assertIn("prefill_start_pos", emitted)
        self.assertIn("num_tokens", emitted)

    def test_recurrent_prefill_seq_len_args_use_runtime_num_tokens(self) -> None:
        op = {
            "function": "recurrent_split_qkv_forward",
            "op": "recurrent_split_qkv",
            "layer": 0,
            "section": "body",
            "args": [
                {"name": "packed_qkv", "expr": "(const float*)(model->bump + A_RECURRENT_PACKED)"},
                {"name": "q", "expr": "(float*)(model->bump + A_RECURRENT_Q)"},
                {"name": "k", "expr": "(float*)(model->bump + A_RECURRENT_K)"},
                {"name": "v", "expr": "(float*)(model->bump + A_RECURRENT_V)"},
                {"name": "seq_len", "expr": "1034"},
                {"name": "q_dim", "expr": "2048"},
                {"name": "k_dim", "expr": "2048"},
                {"name": "v_dim", "expr": "2048"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(op, 7, {"embed_dim": 1024})

        self.assertIn("num_tokens", emitted)
        self.assertNotIn("\n        1034,", emitted)
        self.assertIn("\n        2048,", emitted)

    def test_scalar_constant_is_not_rewritten_as_runtime_token_count(self) -> None:
        op = {
            "function": "recurrent_dt_gate_forward",
            "op": "recurrent_dt_gate",
            "layer": 0,
            "section": "body",
            "args": [
                {"name": "alpha", "expr": "alpha"},
                {"name": "dt_bias", "expr": "dt_bias"},
                {"name": "a", "expr": "a"},
                {"name": "gate", "expr": "gate"},
                {"name": "rows", "source": "dim:seq_len", "expr": "1034"},
                {"name": "num_heads", "source": "dim:gate_dim", "expr": "16"},
                {"name": "state_dim", "source": "const:1", "expr": "1"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(op, 9, {"embed_dim": 1024})

        self.assertIn("\n        num_tokens,\n        16,\n        1\n", emitted)
        self.assertNotIn("\n        num_tokens,\n        16,\n        num_tokens\n", emitted)

    def test_recurrent_prefill_exports_full_batched_projection_boundary(self) -> None:
        op = {
            "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
            "op": "recurrent_gate_proj",
            "layer": 0,
            "section": "body",
            "args": [
                {"name": "A", "expr": "activation"},
                {"name": "B", "expr": "weights"},
                {"name": "bias", "expr": "NULL"},
                {"name": "C", "expr": "gate_output"},
                {"name": "M", "source": "dim:seq_len", "expr": "1034"},
                {"name": "N", "source": "dim:attn_gate_dim", "expr": "2048"},
                {"name": "K", "source": "dim:embed_dim", "expr": "1024"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(op, 3, {"embed_dim": 1024})

        self.assertIn('ck_debug_export_hidden(model, 0, "z"', emitted)
        self.assertIn("(num_tokens) * (2048)", emitted)

    def test_map_owned_weight_preparation_is_deduplicated_and_budgeted(self) -> None:
        preparation = {
            "function": "ck_q5_0_prepare_q8_0_weight",
            "arguments": {"B": "B", "N": "N", "K": "K"},
            "prepared_bytes": "N * (K / 32) * 34",
            "max_total_bytes": 1024,
            "min_remaining_memory_bytes": 256,
        }
        args = [
            {"name": "A", "expr": "input_q8"},
            {"name": "B", "expr": "model->weight_q5"},
            {"name": "bias", "expr": "NULL"},
            {"name": "C", "expr": "output"},
            {"name": "M", "expr": "num_tokens"},
            {"name": "N", "expr": "64"},
            {"name": "K", "expr": "128"},
        ]
        op = {"function": "gemm_nt_q5_0_q8_0", "args": args,
              "call_abi": {"weight_preparation": preparation}}

        emitted = codegen_prefill_v8.emit_prefill_weight_prepare_function([op, op])

        self.assertEqual(emitted.count("ck_q5_0_prepare_q8_0_weight("), 1)
        self.assertIn(
            "ck_model_preparation_budget((size_t)1024, (size_t)256)", emitted
        )
        self.assertIn("mapped_prepared_item_0_0 <= mapped_prepared_budget_0", emitted)
        self.assertIn(
            "mapped_prepared_bytes_0 <= mapped_prepared_budget_0 - mapped_prepared_item_0_0",
            emitted,
        )
        self.assertIn("mapped_prepared_skipped_0 += 1", emitted)
        self.assertIn("const int mapped_prepared_result_0_0", emitted)
        self.assertIn("if (mapped_prepared_result_0_0 > 0)", emitted)
        self.assertIn(
            "if (mapped_prepared_bytes_0 > 0 || mapped_prepared_skipped_0 > 0)",
            emitted,
        )
        self.assertIn(
            "prepared %zu bytes within runtime budget %zu (map max 1024, reserve 256); "
            "skipped %d weight(s)",
            emitted,
        )
        self.assertIn("model->weight_q5, 64, 128", emitted)

    def test_map_owned_weight_preparation_skips_only_items_beyond_budget(self) -> None:
        preparation = {
            "function": "prepare_weight",
            "arguments": {"B": "B", "N": "N", "K": "K"},
            "prepared_bytes": "N * K",
            "max_total_bytes": 1024,
        }

        def op(weight: str, n: int, k: int) -> dict:
            return {
                "function": "gemm_nt_synthetic",
                "args": [
                    {"name": "B", "expr": weight},
                    {"name": "N", "expr": str(n)},
                    {"name": "K", "expr": str(k)},
                ],
                "call_abi": {"weight_preparation": preparation},
            }

        emitted = codegen_prefill_v8.emit_prefill_weight_prepare_function(
            [op("model->small", 8, 8), op("model->large", 64, 64)]
        )

        self.assertIn("prepare_weight(model->small, 8, 8)", emitted)
        self.assertIn("prepare_weight(model->large, 64, 64)", emitted)
        self.assertEqual(emitted.count("mapped_prepared_skipped_0 += 1"), 4)
        self.assertNotIn("if (mapped_prepared_bytes_0 <=", emitted)


if __name__ == "__main__":
    unittest.main()
