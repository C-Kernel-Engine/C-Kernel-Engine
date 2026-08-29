#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_ir_v8  # type: ignore
import codegen_core_v8  # type: ignore
import convert_gguf_to_bump_v8 as converter  # type: ignore


class LagunaContractTests(unittest.TestCase):
    def test_compact_moe_maps_own_parallel_execution_and_scratch(self) -> None:
        maps = ROOT / "version" / "v8" / "kernel_maps"
        expected = {
            "moe_swiglu_expert_forward_q4k_q4k.json": (
                "moe_swiglu_expert_forward_q4k_q4k_parallel_workspace",
                ["independent_rows", "independent_experts"],
            ),
            "moe_swiglu_expert_forward_q4k_q6k.json": (
                "moe_swiglu_expert_forward_q4k_q6k_parallel_workspace",
                ["independent_rows", "independent_experts"],
            ),
            "moe_swiglu_shared_forward_q4k_q4k.json": (
                "moe_swiglu_shared_forward_q4k_q4k_parallel_workspace",
                ["independent_rows"],
            ),
            "moe_swiglu_shared_forward_q4k_q6k.json": (
                "moe_swiglu_shared_forward_q4k_q6k_parallel_workspace",
                ["independent_rows"],
            ),
        }
        for filename, (function, partition) in expected.items():
            with self.subTest(map=filename):
                spec = json.loads((maps / filename).read_text(encoding="utf-8"))
                self.assertEqual(spec["impl"]["function"], function)
                self.assertEqual(
                    spec["implementation"]["threading"]["runtime"],
                    "ck_threadpool",
                )
                self.assertEqual(
                    spec["implementation"]["threading"]["work_partition"],
                    partition,
                )
                self.assertTrue(spec["scratch"][0]["size_bytes"].startswith("64 *"))

    def test_model_map_and_circuit_own_the_hybrid_graph(self) -> None:
        model_map = converter.load_gguf_ck_map()["architectures"]["laguna"]
        self.assertEqual(model_map["template"], "laguna")
        self.assertEqual(model_map["layer_kind_config_key"], "layer_kinds")
        self.assertEqual(
            model_map["metadata_map"]["attention_sliding_window"],
            "laguna.attention.sliding_window",
        )
        self.assertEqual(
            model_map["metadata_map"]["attention_layer_norm_epsilon"],
            "laguna.attention.layer_norm_rms_epsilon",
        )
        self.assertEqual(
            model_map["layer_kinds"]["moe_global_attention"]["tensor_dims"]
            ["attn_q.weight"],
            [2048, 6144],
        )
        self.assertEqual(
            model_map["layer_kinds"]["moe_sliding_attention"]["tensor_dims"]
            ["attn_q.weight"],
            [2048, 8192],
        )

        circuit = build_ir_v8._load_builtin_template_doc("laguna")
        body = circuit["block_types"]["decoder"]["body"]
        self.assertEqual(body["kind_config_key"], "layer_kinds")
        self.assertNotIn("moe_swiglu_expert_mlp", circuit["kernels"])
        self.assertNotIn("shared_swiglu_expert_mlp", circuit["kernels"])
        self.assertEqual(
            circuit["kernels"]["attn_gate_softplus_mul"],
            "attn_gate_softplus_mul_forward",
        )
        global_rope = next(
            item
            for item in body["ops_by_kind"]["moe_global_attention"]
            if item == "rope_qk"
            or (isinstance(item, dict) and item.get("op") == "rope_qk")
        )
        sliding_rope = next(
            item
            for item in body["ops_by_kind"]["moe_sliding_attention"]
            if item == "rope_qk"
            or (isinstance(item, dict) and item.get("op") == "rope_qk")
        )
        self.assertEqual(global_rope, "rope_qk")
        self.assertEqual(
            sliding_rope,
            {"op": "rope_qk", "kernel": "rope_forward_qk_split_direct"},
        )

    def test_init_ir_populates_the_global_yarn_rope_cache(self) -> None:
        circuit = build_ir_v8._load_builtin_template_doc("laguna")
        config = {
            "embed_dim": 2048,
            "num_heads": 64,
            "num_kv_heads": 8,
            "head_dim": 128,
            "context_length": 262144,
            "rotary_dim": 64,
            "max_rotary_dim": 128,
            "rope_theta": 500000.0,
            "rope_scaling_type": "yarn",
            "rope_scaling_factor": 32.0,
            "rope_original_context_length": 8192,
            "rope_beta_fast": 64.0,
            "rope_beta_slow": 1.0,
            "rope_mscale": 1.0,
            "rope_mscale_all_dim": 0.0,
        }
        init_ops = build_ir_v8.generate_init_ops(
            {"template": circuit, "config": config}, config
        )
        rope = next(op for op in init_ops if op["op"] == "yarn_rope_init")

        self.assertEqual(
            rope["kernel"], "yarn_rope_cache_contiguous_positions_f32"
        )
        self.assertEqual(rope["params"]["rotary_dim"]["value"], 64)
        self.assertEqual(rope["params"]["factor"]["value"], 32.0)
        self.assertEqual(rope["params"]["original_context"]["value"], 8192)
        self.assertEqual(rope["params"]["mscale"]["value"], 1.0)
        self.assertEqual(rope["params"]["mscale_all_dim"]["value"], 0.0)

    def test_mixed_global_and_sliding_attention_dims_are_per_layer(self) -> None:
        config = {
            "embed_dim": 2048,
            "num_heads": 64,
            "num_kv_heads": 8,
            "head_dim": 128,
            "v_head_dim": 128,
            "attn_out_dim": 8192,
            "rotary_dim": 128,
            "sliding_window": 512,
            "rope_theta": 500000.0,
            "rope_theta_swa": 10000.0,
            "layer_num_heads": [48, 64],
            "layer_q_dim": [6144, 8192],
            "layer_q_head_dim": [128, 128],
            "layer_k_head_dim": [128, 128],
            "layer_v_head_dim": [128, 128],
            "layer_attention_output_dim": [6144, 8192],
            "layer_rotary_dim": [64, 128],
            "layer_sliding_window": [0, 512],
            "layer_rope_kind": ["full", "swa"],
        }

        for op_name in (
            "qk_norm",
            "rope_qk",
            "attn",
            "attn_gate_softplus_mul",
        ):
            with self.subTest(op=op_name):
                params = {"num_heads": 64, "head_dim": 128}
                build_ir_v8.apply_layer_attention_dims(op_name, params, 0, config)
                self.assertEqual(params["num_heads"], 48)
                self.assertEqual(params["num_kv_heads"], 8)
                self.assertEqual(params["head_dim"], 128)

        projection = {}
        build_ir_v8.apply_layer_attention_dims("q_proj", projection, 0, config)
        self.assertEqual(projection["_output_dim"], 6144)
        output = {}
        build_ir_v8.apply_layer_attention_dims("out_proj", output, 0, config)
        self.assertEqual(output["_input_dim"], 6144)

    def test_generic_decode_xray_exports_rope_and_gated_attention(self) -> None:
        common_args = [
            {"name": "num_heads", "expr": "48"},
            {"name": "num_kv_heads", "expr": "8"},
            {"name": "num_tokens", "expr": "1"},
            {"name": "head_dim", "expr": "128"},
        ]
        rope = codegen_core_v8.emit_op(
            {
                "idx": 0,
                "layer": 0,
                "section": "body",
                "op": "rope_qk",
                "function": "rope_forward_qk_with_rotary_dim",
                "args": [
                    {"name": "q", "expr": "q"},
                    {"name": "k", "expr": "k"},
                    *common_args,
                ],
            },
            dump=True,
        )
        self.assertIn('"Qcur_rope", 48, 1, 128', rope)
        self.assertIn('"Kcur_rope", 8, 1, 128', rope)

        gated = codegen_core_v8.emit_op(
            {
                "idx": 1,
                "layer": 0,
                "section": "body",
                "op": "attn_gate_softplus_mul",
                "function": "attn_gate_softplus_mul_forward",
                "args": [
                    {"name": "out", "expr": "attn"},
                    {"name": "rows", "expr": "1"},
                    {"name": "num_heads", "expr": "48"},
                    {"name": "state_dim", "expr": "128"},
                ],
            },
            dump=True,
        )
        self.assertIn('"attn_gated", 48, 1, 128', gated)

    def test_compiler_and_codegen_have_no_laguna_family_branch(self) -> None:
        for relative in (
            "version/v8/scripts/build_ir_v8.py",
            "version/v8/scripts/codegen_core_v8.py",
            "version/v8/scripts/codegen_prefill_v8.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8").lower()
            self.assertNotIn("laguna", source, relative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
