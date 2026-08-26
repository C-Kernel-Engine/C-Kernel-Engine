#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEGEN_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_core_v8.py"
PREFILL_CODEGEN_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_prefill_v8.py"
sys.path.insert(0, str(CODEGEN_PATH.parent))


def _load_codegen():
    spec = importlib.util.spec_from_file_location("codegen_core_hidden_export_tests", CODEGEN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {CODEGEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


codegen = _load_codegen()


def _load_prefill_codegen():
    spec = importlib.util.spec_from_file_location(
        "codegen_prefill_hidden_export_tests", PREFILL_CODEGEN_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {PREFILL_CODEGEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prefill_codegen = _load_prefill_codegen()


def _arg(name: str, expr: str) -> dict[str, str]:
    return {"name": name, "expr": expr}


class HiddenExportExtentTests(unittest.TestCase):
    def test_fused_q4_gateup_swiglu_exports_materialized_output(self) -> None:
        gate = {
            "op": "mlp_gate_up",
            "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
            "layer": 0,
            "resolved_execution": {
                "numerical_contract": "q4_k_x_q8_k_repacked_matmul_fp32",
                "implementation": {
                    "weight_storage": {
                        "format": "q4_k",
                        "block_elements": 256,
                        "block_bytes": 144,
                    },
                    "activation_storage": {"format": "q8_k", "block_elements": 256},
                    "diagnostic_providers": {
                        "fp32_activation": "gemm_nt_q4_k",
                        "row_quantized": "gemv_q4_k_q8_k",
                    },
                },
                "reference": {
                    "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch"
                },
                "production": {
                    "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch"
                },
            },
            "args": [
                _arg("A", "A"),
                _arg("B", "B"),
                _arg("bias", "NULL"),
                _arg("C", "OUT"),
                _arg("M", "num_tokens"),
                _arg("N", "34816"),
                _arg("K", "5120"),
            ],
        }
        swiglu = {
            "op": "silu_mul",
            "function": "swiglu_forward_ggml",
            "layer": 0,
            "args": [_arg("output", "OUT")],
        }

        emitted = prefill_codegen._emit_prefill_q4_gateup_swiglu_x16(
            gate,
            swiglu,
            23,
            "fused_23",
            {},
            debug_flag_name="debug_gate",
            debug_input_name="debug_input",
        )

        self.assertIn(
            'if (fused_23) ck_debug_export_hidden(model, 0, "mlp_swiglu", '
            "(const float*)OUT, (num_tokens) * (((34816) / 2)))",
            emitted,
        )

    def test_parity_dump_captures_only_distinct_layer_input_edge(self) -> None:
        op = {
            "op": "residual_save",
            "function": "memcpy",
            "layer": 0,
            "args": [
                _arg("dst", "RESIDUAL"),
                _arg("src", "LAYER_INPUT"),
                _arg("size", "20480"),
            ],
        }
        layer_entry = codegen.emit_op(op, dump=True, op_instance_idx=0)
        after_attention = codegen.emit_op(op, dump=True, op_instance_idx=1)

        self.assertIn(
            'ck_dump_tensor((const float*)(LAYER_INPUT), 0, "layer_input", '
            "((size_t)(20480)) / sizeof(float))",
            layer_entry,
        )
        self.assertNotIn("ck_dump_tensor", after_attention)

    def test_mlp_projection_exports_cover_all_rows_and_channels(self) -> None:
        up = codegen.emit_op(
            {
                "op": "mlp_up",
                "function": "gemm_nt_bf16",
                "layer": 1,
                "args": [
                    _arg("a", "A"),
                    _arg("b", "B"),
                    _arg("bias", "BIAS"),
                    _arg("c", "UP"),
                    _arg("m", "4032"),
                    _arg("n", "4304"),
                    _arg("k", "1152"),
                ],
            }
        )
        down = codegen.emit_op(
            {
                "op": "mlp_down",
                "function": "gemm_nt_bf16",
                "layer": 1,
                "args": [
                    _arg("a", "UP"),
                    _arg("b", "B"),
                    _arg("bias", "BIAS"),
                    _arg("c", "DOWN"),
                    _arg("m", "4032"),
                    _arg("n", "1152"),
                    _arg("k", "4304"),
                ],
            }
        )

        self.assertIn('"mlp_up", (const float*)UP, (4032) * (4304)', up)
        self.assertIn('"mlp_up_last"', up)
        self.assertIn('(size_t)(4304)', up)
        self.assertIn('"mlp_down", (const float*)DOWN, (4032) * (1152)', down)
        self.assertIn('"mlp_down_last"', down)
        self.assertIn('(size_t)(1152)', down)

    def test_gelu_exports_the_full_post_activation_tensor(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "gelu",
                "function": "gelu_ggml_inplace",
                "layer": 1,
                "args": [_arg("data", "UP"), _arg("n", "17353728")],
            }
        )

        self.assertIn('"ffn_gelu", (const float*)UP, 17353728', emitted)

    def test_attention_norm_exports_the_projection_input_boundary(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "attn_norm",
                "function": "rmsnorm_forward",
                "layer": 0,
                "args": [
                    _arg("input", "X"),
                    _arg("weight", "W"),
                    _arg("output", "Y"),
                    _arg("bias", "NULL"),
                    _arg("num_tokens", "1"),
                    _arg("dim", "1024"),
                    _arg("aligned_dim", "1024"),
                    _arg("eps", "1e-6f"),
                ],
            }
        )

        self.assertIn('"attn_norm", (const float*)Y, (1) * (1024)', emitted)

    def test_recurrent_scalar_projections_export_every_output_channel(self) -> None:
        for op_name, label in (
            ("recurrent_alpha_proj", "alpha"),
            ("recurrent_beta_proj", "beta"),
        ):
            emitted = codegen.emit_op(
                {
                    "op": op_name,
                    "function": "gemm_nt_f32_llama_production",
                    "layer": 1,
                    "args": [
                        _arg("a", "A"),
                        _arg("b", "B"),
                        _arg("bias", "NULL"),
                        _arg("c", "OUT"),
                        _arg("m", "1"),
                        _arg("n", "48"),
                        _arg("k", "5120"),
                    ],
                }
            )

            self.assertIn(f'"{label}", (const float*)OUT, (1) * (48)', emitted)

    def test_qwen35_attention_exports_gate_and_pregate_boundaries(self) -> None:
        split = codegen.emit_op(
            {
                "op": "split_q_gate",
                "function": "split_q_gate_forward",
                "layer": 3,
                "args": [
                    _arg("packed_qg", "PACKED"),
                    _arg("q", "Q"),
                    _arg("gate", "GATE"),
                    _arg("rows", "1"),
                    _arg("q_dim", "2048"),
                    _arg("gate_dim", "2048"),
                ],
            }
        )
        attention = codegen.emit_op(
            {
                "op": "attn",
                "function": "attention_forward_decode_head_major_gqa_ggml_regular",
                "layer": 3,
                "args": [
                    _arg("q_token", "Q"),
                    _arg("k_cache", "K"),
                    _arg("v_cache", "V"),
                    _arg("out_token", "ATTN"),
                    _arg("num_heads", "8"),
                    _arg("num_tokens", "1"),
                    _arg("aligned_head_dim", "256"),
                ],
            }
        )
        gated = codegen.emit_op(
            {
                "op": "attn_gate_sigmoid_mul",
                "function": "attn_gate_sigmoid_mul_forward",
                "layer": 3,
                "args": [
                    _arg("x", "ATTN"),
                    _arg("gate", "GATE"),
                    _arg("out", "ATTN"),
                    _arg("rows", "18"),
                    _arg("num_heads", "8"),
                    _arg("state_dim", "256"),
                ],
            }
        )

        self.assertIn('"attn_gate", (const float*)GATE, (1) * (2048)', split)
        self.assertIn('"attn_pregate", (const float*)ATTN, (8) * (1) * (256)', attention)
        self.assertIn('"attn_out", (const float*)ATTN, (8) * (18) * (256)', gated)

    def test_mla_exports_each_semantic_attention_boundary(self) -> None:
        gate = codegen.emit_op(
            {
                "op": "attention_gate_projection",
                "function": "gemm_nt_bf16",
                "layer": 0,
                "args": [
                    _arg("output", "GATE"),
                    _arg("rows", "30"),
                    _arg("N", "2048"),
                ],
            }
        )
        kv_a = codegen.emit_op(
            {
                "op": "kv_a_proj",
                "function": "gemm_nt_bf16",
                "layer": 0,
                "args": [
                    _arg("output", "KV_A"),
                    _arg("M", "30"),
                    _arg("N", "576"),
                ],
            }
        )
        kv_norm = codegen.emit_op(
            {
                "op": "kv_a_layernorm",
                "function": "rmsnorm_forward_kv_lora",
                "layer": 0,
                "args": [
                    _arg("output", "KV_NORM"),
                    _arg("tokens", "30"),
                    _arg("d_model", "512"),
                ],
            }
        )
        decompress = codegen.emit_op(
            {
                "op": "kv_lora_decompress",
                "function": "deepseek_mla_kv_decompress_f32",
                "layer": 0,
                "args": [
                    _arg("k_nope", "K_NOPE"),
                    _arg("value", "VALUE"),
                    _arg("tokens", "30"),
                    _arg("heads", "16"),
                    _arg("qk_nope_dim", "128"),
                    _arg("v_dim", "128"),
                ],
            }
        )
        rope = codegen.emit_op(
            {
                "op": "partial_rope_concat",
                "function": "deepseek_mla_partial_rope_concat_packed_f32",
                "layer": 0,
                "args": [
                    _arg("query", "QUERY"),
                    _arg("key", "KEY"),
                    _arg("tokens", "30"),
                    _arg("heads", "16"),
                    _arg("qk_nope_dim", "128"),
                    _arg("qk_rope_dim", "64"),
                ],
            }
        )
        attention = codegen.emit_op(
            {
                "op": "mla_attention",
                "function": "deepseek_mla_attention_f32",
                "layer": 0,
                "args": [
                    _arg("output", "CONTEXT"),
                    _arg("num_tokens", "30"),
                    _arg("num_heads", "16"),
                    _arg("v_head_dim", "128"),
                ],
            }
        )

        self.assertIn('"attn_gate", (const float*)GATE, (30) * (2048)', gate)
        self.assertIn('"mla_kv_a", (const float*)KV_A, (30) * (576)', kv_a)
        self.assertIn('"mla_kv_norm", (const float*)KV_NORM, (30) * (512)', kv_norm)
        self.assertIn('"mla_k_nope", (const float*)K_NOPE, (30) * (16) * (128)', decompress)
        self.assertIn('"mla_value", (const float*)VALUE, (30) * (16) * (128)', decompress)
        self.assertIn('"mla_query", (const float*)QUERY, (30) * (16) * ((128 + 64))', rope)
        self.assertIn('"mla_key", (const float*)KEY, (30) * (16) * ((128 + 64))', rope)
        self.assertIn('"mla_context", (const float*)CONTEXT, (30) * (16) * (128)', attention)

    def test_kimi_block_norm_exports_distinguish_attention_and_ffn_inputs(self) -> None:
        first = codegen.emit_op(
            {
                "op": "block_rmsnorm",
                "function": "rmsnorm_forward",
                "layer": 0,
                "args": [
                    _arg("input", "ATTN_INPUT"),
                    _arg("output", "ATTN_NORM"),
                    _arg("rows", "30"),
                ],
            },
            op_instance_idx=0,
        )
        second = codegen.emit_op(
            {
                "op": "block_rmsnorm",
                "function": "rmsnorm_forward",
                "layer": 0,
                "args": [
                    _arg("input", "FFN_INPUT"),
                    _arg("output", "FFN_NORM"),
                    _arg("rows", "30"),
                ],
            },
            op_instance_idx=1,
        )

        self.assertIn('"block_rmsnorm", (const float*)ATTN_NORM', first)
        self.assertNotIn('"ffn_input"', first)
        self.assertNotIn('"ffn_norm"', first)
        self.assertIn('"ffn_input", (const float*)FFN_INPUT', second)
        self.assertIn('"ffn_norm", (const float*)FFN_NORM', second)
        self.assertNotIn('"block_rmsnorm"', second)

    def test_moe_exports_router_and_expert_boundaries(self) -> None:
        router = codegen.emit_op(
            {
                "op": "moe_router",
                "function": "gemm_blocked_serial",
                "layer": 1,
                "args": [
                    _arg("C", "ROUTER"),
                    _arg("M", "30"),
                    _arg("N", "64"),
                ],
            }
        )
        selection = codegen.emit_op(
            {
                "op": "group_limited_topk_router",
                "function": "group_limited_topk_router_sigmoid_f32",
                "layer": 1,
                "args": [
                    _arg("indices", "SELECTED"),
                    _arg("weights", "ROUTING"),
                    _arg("rows", "30"),
                    _arg("top_k", "6"),
                ],
            }
        )
        routed = codegen.emit_op(
            {
                "op": "moe_swiglu_expert_mlp",
                "function": "moe_swiglu_expert_forward_bf16",
                "layer": 1,
                "args": [
                    _arg("output", "ROUTED"),
                    _arg("rows", "30"),
                    _arg("hidden_dim", "2048"),
                ],
            }
        )
        combined = codegen.emit_op(
            {
                "op": "shared_swiglu_expert_mlp",
                "function": "moe_swiglu_shared_forward_bf16",
                "layer": 1,
                "args": [
                    _arg("output", "COMBINED"),
                    _arg("rows", "30"),
                    _arg("hidden_dim", "2048"),
                ],
            }
        )
        self.assertIn('"moe_router_logits", (const float*)ROUTER, (30) * (64)', router)
        self.assertIn(
            '"moe_selected_experts", (const int32_t*)SELECTED, (30) * (6)',
            selection,
        )
        self.assertIn('"moe_routing_weights", (const float*)ROUTING, (30) * (6)', selection)
        self.assertIn('"moe_routed_output", (const float*)ROUTED, (30) * (2048)', routed)
        self.assertIn('"moe_combined_output", (const float*)COMBINED, (30) * (2048)', combined)

    def test_farskip_combine_exports_both_persistent_streams(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "farskip_routed_shared_combine",
                "function": "farskip_swiglu_shared_combine_bf16",
                "layer": 1,
                "args": [
                    _arg("main_output", "MAIN"),
                    _arg("routed_free_output", "ROUTED_FREE"),
                    _arg("rows", "30"),
                    _arg("hidden_dim", "2048"),
                ],
            }
        )

        self.assertIn('"layer_out", (const float*)MAIN, (30) * (2048)', emitted)
        self.assertIn(
            '"routed_free_out", (const float*)ROUTED_FREE, (30) * (2048)',
            emitted,
        )

    def test_attention_checkpoint_name_comes_from_call_ir_contract(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "attn",
                "function": "attention_forward",
                "layer": 2,
                "args": [
                    _arg("out_token", "ATTN"),
                    _arg("num_heads", "16"),
                    _arg("num_tokens", "1008"),
                    _arg("aligned_head_dim", "72"),
                ],
                "semantic_checkpoints": [
                    {
                        "id": "vision.layer.2.attention.output",
                        "tensor": "attn_out_head_major",
                    }
                ],
            }
        )

        self.assertIn(
            '"attn_out_head_major", (const float*)ATTN, (16) * (1008) * (72)',
            emitted,
        )
        self.assertNotIn('"attn_pregate"', emitted)

    def test_qwen35_prefill_exports_full_norm_and_swiglu_extents(self) -> None:
        norm = codegen.emit_op(
            {
                "op": "post_attention_norm",
                "function": "rmsnorm_forward_llama_production",
                "layer": 3,
                "args": [
                    _arg("output", "NORM"),
                    _arg("tokens", "18"),
                    _arg("d_model", "1024"),
                ],
            }
        )
        swiglu = codegen.emit_op(
            {
                "op": "silu_mul",
                "function": "swiglu_forward_ggml",
                "layer": 3,
                "args": [
                    _arg("output", "MLP"),
                    _arg("tokens", "18"),
                    _arg("dim", "3584"),
                ],
            }
        )
        self.assertIn('"post_attn_norm", (const float*)NORM, (18) * (1024)', norm)
        self.assertIn('"mlp_swiglu", (const float*)MLP, (18) * (3584)', swiglu)

    def test_recurrent_core_exports_state_before_in_place_update(self) -> None:
        op = {
            "op": "recurrent_core",
            "function": "gated_deltanet_forward",
            "layer": 4,
            "args": [
                _arg("state_in", "STATE"),
                _arg("state_out", "STATE"),
                _arg("num_heads", "48"),
                _arg("state_dim", "128"),
                _arg("tokens", "1"),
                _arg("output", "OUTPUT"),
            ],
        }
        decode = codegen.emit_op(op)
        prefill = prefill_codegen.emit_prefill_op(op, 1, {"embed_dim": 5120})

        expected = '"state_predelta", (const float*)STATE, (48) * (128) * (128)'
        self.assertIn(expected, decode)
        self.assertIn(expected, prefill)
        self.assertLess(decode.index('"state_predelta"'), decode.index("gated_deltanet_forward("))
        self.assertLess(prefill.index('"state_predelta"'), prefill.index("gated_deltanet_forward("))

    def test_quantized_projection_exports_full_prefill_extents(self) -> None:
        resolved = {
            "numerical_contract": "q4_k_x_q8_k_repacked_matmul_fp32",
            "implementation": {
                "weight_storage": {"format": "q4_k", "block_elements": 256, "block_bytes": 144},
                "activation_storage": {"format": "q8_k", "block_elements": 256},
                "diagnostic_providers": {"fp32_activation": "gemm_nt_q4_k"},
            }
        }
        emitted = codegen.emit_op(
            {
                "op": "out_proj",
                "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
                "layer": 3,
                "resolved_execution": resolved,
                "args": [
                    _arg("A", "Q8"),
                    _arg("B", "WEIGHT"),
                    _arg("C", "OUT"),
                    _arg("M", "18"),
                    _arg("N", "1024"),
                    _arg("K", "2048"),
                ],
            }
        )
        self.assertIn('"out_proj", (const float*)OUT, (18) * (1024)', emitted)

    def test_fused_gate_up_prefill_exports_combined_row_major_matrix(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "mlp_gate_up",
                "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
                "layer": 3,
                "args": [
                    _arg("A", "Q8"),
                    _arg("B", "WEIGHT"),
                    _arg("C", "GATE_UP"),
                    _arg("M", "18"),
                    _arg("N", "7168"),
                    _arg("K", "1024"),
                ],
            }
        )
        self.assertIn('"mlp_gate_up", (const float*)GATE_UP, (18) * (7168)', emitted)
        self.assertIn('"mlp_gate_up_last"', emitted)
        self.assertIn('(size_t)(7168)', emitted)


if __name__ == "__main__":
    unittest.main()
