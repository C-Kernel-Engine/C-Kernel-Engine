#!/usr/bin/env python3
import json
import inspect
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v7" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_ir_v7 as build_ir
import memory_planner_v7 as memory_planner


class TestQwen35TemplateGuard(unittest.TestCase):
    def test_dense_templates_remain_lowerable(self) -> None:
        for name in ("qwen2", "qwen3", "gemma3", "llama"):
            path = ROOT / "version" / "v7" / "templates" / f"{name}.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            manifest = {"config": {"model": name, "arch": name}, "template": doc}
            self.assertIsNone(
                build_ir.unsupported_template_lowering_reason(manifest),
                msg=f"{name} should stay on the dense lowering path",
            )

    def test_qwen35_template_no_longer_hits_generic_guard(self) -> None:
        path = ROOT / "version" / "v7" / "templates" / "qwen35.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        manifest = {
            "config": {
                "model": "qwen35",
                "arch": "qwen35",
                "layer_kinds": ["full_attention"],
                "full_attention_interval": 4,
            },
            "template": doc,
        }
        self.assertIsNone(build_ir.unsupported_template_lowering_reason(manifest))

    def test_body_ops_resolve_by_declared_kind(self) -> None:
        path = ROOT / "version" / "v7" / "templates" / "qwen35.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        body = doc["block_types"]["decoder"]["body"]
        resolved = build_ir._resolve_body_ops_for_layer(
            body,
            {"layer_kinds": ["full_attention"]},
            0,
        )
        self.assertEqual(
            resolved[:10],
            [
                "attn_norm",
                "q_gate_proj",
                "split_q_gate",
                "k_proj",
                "v_proj",
                "qk_norm",
                "rope_qk",
                "attn",
                "attn_gate_sigmoid_mul",
                "out_proj",
            ],
        )

    def test_qwen35_places_post_attention_norm_after_attention_residual(self) -> None:
        path = ROOT / "version" / "v7" / "templates" / "qwen35.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        body = doc["block_types"]["decoder"]["body"]
        for kind in ("recurrent", "full_attention"):
            ops = body["ops_by_kind"][kind]
            self.assertLess(ops.index("out_proj") if kind == "full_attention" else ops.index("recurrent_out_proj"), ops.index("residual_add"))
            self.assertLess(ops.index("residual_add"), ops.index("post_attention_norm"))
            self.assertNotIn("rmsnorm", ops)

    def test_template_declares_recurrent_projection_activation_preferences(self) -> None:
        path = ROOT / "version" / "v7" / "templates" / "qwen35.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        prefs = doc.get("flags", {}).get("activation_preference_by_op", {})
        self.assertEqual(
            prefs,
            {
                "recurrent_gate_proj": "fp32",
                "recurrent_alpha_proj": "fp32",
                "recurrent_beta_proj": "fp32",
            },
        )
        self.assertEqual(doc.get("flags", {}).get("prefill_policy"), "sequential_decode")

    def test_body_ops_resolve_generic_ops_by_kind_contract(self) -> None:
        body = {
            "kind_config_key": "block_kinds",
            "ops_by_kind": {
                "alpha": ["foo", "bar"],
                "beta": ["baz"],
            },
        }
        resolved = build_ir._resolve_body_ops_for_layer(
            body,
            {"block_kinds": ["beta"]},
            0,
        )
        self.assertEqual(resolved, ["baz"])

    def test_residual_save_is_inserted_only_for_branch_entry_norms(self) -> None:
        gemma_path = ROOT / "version" / "v7" / "templates" / "gemma3.json"
        gemma = json.loads(gemma_path.read_text(encoding="utf-8"))
        ops = gemma["block_types"]["decoder"]["body"]["ops"]
        self.assertTrue(build_ir.should_insert_residual_save(ops, ops.index("attn_norm")))
        self.assertFalse(build_ir.should_insert_residual_save(ops, ops.index("post_attention_norm")))
        self.assertTrue(build_ir.should_insert_residual_save(ops, ops.index("ffn_norm")))
        self.assertFalse(build_ir.should_insert_residual_save(ops, ops.index("post_ffn_norm")))

    def test_quant_aliases_resolve_from_generic_kind_contract(self) -> None:
        body = {
            "kind_config_key": "block_kinds",
            "quant_aliases_common": {
                "ln1_gamma": "attn_norm",
            },
            "quant_aliases_by_kind": {
                "full_attention": {
                    "wq": "attn_q_gate",
                },
                "recurrent": {
                    "wq": "attn_qkv",
                },
            },
        }
        aliases = build_ir._resolve_template_quant_aliases(
            body,
            {"block_kinds": ["full_attention"]},
            0,
        )
        self.assertEqual(
            aliases,
            {
                "ln1_gamma": "attn_norm",
                "wq": "attn_q_gate",
            },
        )

    def test_recurrent_matmul_dims_are_derived_from_generic_ssm_config(self) -> None:
        config = build_ir._normalize_manifest_config(
            {
                "embed_dim": 1024,
                "num_layers": 4,
                "ssm_state_size": 128,
                "ssm_group_count": 16,
                "ssm_time_step_rank": 16,
                "ssm_inner_size": 2048,
            }
        )
        self.assertEqual(build_ir.compute_matmul_dims("recurrent_qkv_proj", config), (6144, 1024))
        self.assertEqual(build_ir.compute_matmul_dims("recurrent_gate_proj", config), (2048, 1024))
        self.assertEqual(build_ir.compute_matmul_dims("recurrent_alpha_proj", config), (16, 1024))
        self.assertEqual(build_ir.compute_matmul_dims("recurrent_beta_proj", config), (16, 1024))

    def test_recurrent_out_proj_reads_recurrent_branch_by_contract(self) -> None:
        self.assertEqual(
            build_ir.OP_DATAFLOW["recurrent_out_proj"]["inputs"]["x"],
            "recurrent_normed",
        )

    def test_recurrent_normed_uses_distinct_physical_buffer(self) -> None:
        self.assertEqual(
            memory_planner.SLOT_TO_BUFFER_DEFAULT["recurrent_normed"],
            "A_RECURRENT_NORMED",
        )

    def test_recurrent_state_buffers_scale_by_layer_count(self) -> None:
        config = build_ir._normalize_manifest_config(
            {
                "num_layers": 4,
                "embed_dim": 1024,
                "ssm_state_size": 128,
                "ssm_group_count": 16,
                "ssm_time_step_rank": 16,
                "ssm_inner_size": 2048,
                "ssm_conv_history": 3,
                "ssm_conv_channels": 6144,
                "gate_dim": 16,
            }
        )
        specs = build_ir.build_activation_specs(config, mode="decode", context_len=64)
        conv_stride = build_ir._recurrent_state_stride_bytes(config, "conv")
        ssm_stride = build_ir._recurrent_state_stride_bytes(config, "ssm")
        self.assertEqual(specs["recurrent_conv_state"]["size"], 4 * conv_stride)
        self.assertEqual(specs["recurrent_ssm_state"]["size"], 4 * ssm_stride)

    def test_layer_scoped_recurrent_state_offsets_are_generic(self) -> None:
        lowered_op = {
            "layer": 3,
            "activations": {
                "state_in": {
                    "buffer": "recurrent_ssm_state",
                    "activation_offset": 1024,
                    "ptr_expr": "activations + 1024",
                },
                "conv_in": {
                    "buffer": "recurrent_conv_state",
                    "activation_offset": 2048,
                    "ptr_expr": "activations + 2048",
                },
            },
            "outputs": {
                "state_out": {
                    "buffer": "recurrent_ssm_state",
                    "activation_offset": 1024,
                    "ptr_expr": "activations + 1024",
                },
            },
        }
        config = {
            "ssm_conv_history": 3,
            "ssm_conv_channels": 6144,
            "gate_dim": 16,
            "ssm_state_size": 128,
        }
        build_ir._apply_layer_scoped_recurrent_state_offsets(lowered_op, config)
        self.assertEqual(lowered_op["activations"]["conv_in"]["activation_offset"], 2048 + 3 * 3 * 6144 * 4)
        ssm_stride = build_ir._recurrent_state_stride_bytes(config, "ssm")
        self.assertEqual(lowered_op["activations"]["state_in"]["activation_offset"], 1024 + 3 * ssm_stride)
        self.assertEqual(lowered_op["outputs"]["state_out"]["activation_offset"], 1024 + 3 * ssm_stride)

    def test_call_lower_preserves_scoped_activation_offsets(self) -> None:
        stride = 3 * 6144 * 4
        lowered_ir = {
            "config": {},
            "memory": {
                "arena": {
                    "mode": "region",
                    "weights_base": 0,
                    "activations_base": 4096,
                },
                "weights": {"entries": []},
                "activations": {
                    "buffers": [
                        {
                            "name": "recurrent_conv_state",
                            "offset": 2048,
                            "define": "A_RECURRENT_CONV_STATE",
                        },
                        {
                            "name": "recurrent_q",
                            "offset": 8192,
                            "define": "A_RECURRENT_Q",
                        },
                        {
                            "name": "recurrent_k",
                            "offset": 12288,
                            "define": "A_RECURRENT_K",
                        },
                        {
                            "name": "recurrent_v",
                            "offset": 16384,
                            "define": "A_RECURRENT_V",
                        },
                        {
                            "name": "recurrent_packed",
                            "offset": 20480,
                            "define": "A_RECURRENT_PACKED",
                        },
                    ]
                },
            },
            "operations": [
                {
                    "idx": 0,
                    "function": "recurrent_conv_state_update_forward",
                    "op": "recurrent_conv_state_update",
                    "layer": 1,
                    "section": "body",
                    "activations": {
                        "state_in": {
                            "buffer": "recurrent_conv_state",
                            "activation_offset": 2048 + stride,
                        },
                        "q": {
                            "buffer": "recurrent_q",
                            "activation_offset": 8192,
                        },
                        "k": {
                            "buffer": "recurrent_k",
                            "activation_offset": 12288,
                        },
                        "v": {
                            "buffer": "recurrent_v",
                            "activation_offset": 16384,
                        },
                    },
                    "outputs": {
                        "conv_x": {
                            "buffer": "recurrent_packed",
                            "activation_offset": 20480,
                        },
                        "state_out": {
                            "buffer": "recurrent_conv_state",
                            "activation_offset": 2048 + stride,
                        },
                    },
                    "weights": {},
                    "scratch": [],
                    "params": {
                        "ssm_conv_history": 3,
                        "num_seqs": 1,
                        "seq_len": 1,
                        "q_dim": 2048,
                        "k_dim": 2048,
                        "v_dim": 2048,
                    },
                }
            ],
        }
        lowered_call = build_ir.generate_ir_lower_3(lowered_ir, "decode")
        op = lowered_call["operations"][0]
        arg_expr = {arg["name"]: arg["expr"] for arg in op["args"]}
        self.assertEqual(
            arg_expr["state_in"],
            f"(const float*)(model->bump + (A_RECURRENT_CONV_STATE + {stride}))",
        )
        self.assertEqual(
            arg_expr["state_out"],
            f"(float*)(model->bump + (A_RECURRENT_CONV_STATE + {stride}))",
        )

    def test_post_attention_norm_is_treated_as_generic_pre_norm(self) -> None:
        # Residual-save and quantize insertion should be driven by the declared
        # stitch op type, not by model-family-specific branches.
        self.assertIn('"post_attention_norm"', inspect.getsource(build_ir))

    def test_planned_attention_scratch_preserves_declared_logical_slot(self) -> None:
        activation_buffers = {
            "q_scratch": {"offset": 0},
            "k_scratch": {"offset": 128},
            "attn_scratch": {"offset": 256},
        }
        buffer_name_map = {
            "A_ATTN_SCRATCH": "attn_scratch",
        }
        self.assertEqual(
            build_ir._resolve_logical_buffer_name(
                "A_ATTN_SCRATCH",
                "q_scratch",
                activation_buffers,
                buffer_name_map,
            ),
            "q_scratch",
        )
        self.assertEqual(
            build_ir._resolve_logical_buffer_name(
                "A_ATTN_SCRATCH",
                "attn_scratch",
                activation_buffers,
                buffer_name_map,
            ),
            "attn_scratch",
        )

    def test_recurrent_norm_gate_preserves_declared_gate_buffer(self) -> None:
        lowered_op = {"activations": {}, "outputs": {}}
        ir_op = {
            "inputs": {
                "x": {"dtype": "fp32"},
                "gate": {"dtype": "fp32"},
            },
            "outputs": {
                "out": {"dtype": "fp32"},
            },
            "weights": {
                "ssm_norm": {"name": "layer.0.ssm_norm"},
            },
        }
        activation_buffers = {
            "recurrent_packed": {"offset": 1024},
            "recurrent_z": {"offset": 2048},
            "recurrent_normed": {"offset": 4096},
        }
        build_ir._bind_recurrent_norm_gate_io(lowered_op, ir_op, activation_buffers)
        self.assertEqual(lowered_op["activations"]["x"]["buffer"], "recurrent_packed")
        self.assertEqual(lowered_op["activations"]["gate"]["buffer"], "recurrent_z")
        self.assertEqual(lowered_op["outputs"]["out"]["buffer"], "recurrent_normed")

    def test_dataflow_name_resolution_preserves_declared_gate_slot(self) -> None:
        legacy_map = {"gate": "x", "up": "x"}
        ir_op = {
            "dataflow": {
                "inputs": {
                    "x": {"slot": "attn_scratch"},
                    "gate": {"slot": "attn_gate"},
                }
            }
        }
        self.assertEqual(
            build_ir._resolve_planner_io_name("gate", True, ir_op, "inputs", legacy_map),
            "gate",
        )
        self.assertEqual(
            build_ir._resolve_planner_io_name("x", True, ir_op, "inputs", legacy_map),
            "x",
        )
        self.assertEqual(
            build_ir._resolve_planner_io_name("gate", False, ir_op, "inputs", legacy_map),
            "gate",
        )

    def test_legacy_name_resolution_still_supports_swiglu_aliases(self) -> None:
        legacy_map = {"gate": "x", "up": "x"}
        ir_op = {"dataflow": {"inputs": {"x": {"slot": "mlp_scratch"}}}}
        self.assertEqual(
            build_ir._resolve_planner_io_name("gate", False, ir_op, "inputs", legacy_map),
            "x",
        )
        self.assertEqual(
            build_ir._resolve_planner_io_name("up", False, ir_op, "inputs", legacy_map),
            "x",
        )


if __name__ == "__main__":
    unittest.main()
