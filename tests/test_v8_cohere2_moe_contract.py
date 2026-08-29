#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_ir_v8  # type: ignore
import convert_gguf_to_bump_v8 as converter  # type: ignore


def _tiny_manifest() -> dict:
    entries: list[dict] = []
    offset = 0

    def add(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        size = 1
        for dim in shape:
            size *= int(dim)
        entries.append({
            "name": name,
            "dtype": dtype,
            "offset": offset,
            "file_offset": offset,
            "size": size,
            "nbytes": size,
            "shape": shape,
        })
        offset += size

    hidden = 256
    attention_width = 512
    dense_ff = 256
    expert_ff = 256
    experts = 8
    add("token_emb", "q8_0", [512, hidden])
    for layer in range(4):
        add(f"layer.{layer}.ln1_gamma", "fp32", [hidden])
        add(f"layer.{layer}.ln1_beta", "fp32", [hidden])
        add(f"layer.{layer}.wq", "q8_0", [attention_width, hidden])
        add(f"layer.{layer}.wk", "q8_0", [64, hidden])
        add(f"layer.{layer}.wv", "q8_0", [64, hidden])
        add(f"layer.{layer}.wo", "q8_0", [hidden, attention_width])
        if layer == 0:
            add(f"layer.{layer}.w1", "q8_0", [dense_ff, hidden])
            add(f"layer.{layer}.w3", "q8_0", [dense_ff, hidden])
            add(f"layer.{layer}.w2", "q8_0", [hidden, dense_ff])
        else:
            add(f"layer.{layer}.moe_router", "fp32", [experts, hidden])
            add(
                f"layer.{layer}.moe_expert_gate",
                "q4_k",
                [experts, expert_ff, hidden],
            )
            add(
                f"layer.{layer}.moe_expert_up",
                "q4_k",
                [experts, expert_ff, hidden],
            )
            add(
                f"layer.{layer}.moe_expert_down",
                "q5_k",
                [experts, hidden, expert_ff],
            )
    add("final_ln_weight", "fp32", [hidden])
    add("final_ln_bias", "fp32", [hidden])

    config = {
        "model": "cohere2_moe",
        "arch": "cohere2_moe",
        "num_layers": 4,
        "embed_dim": hidden,
        "attn_out_dim": attention_width,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 64,
        "v_head_dim": 64,
        "intermediate_size": dense_ff,
        "moe_intermediate_size": expert_ff,
        "n_routed_experts": experts,
        "experts_per_tok": 2,
        "num_experts_per_tok": 2,
        "router_num_groups": 1,
        "router_topk_group": 1,
        "router_norm_topk_prob": False,
        "routed_scaling_factor": 1.0,
        "context_length": 64,
        "max_seq_len": 64,
        "vocab_size": 512,
        "rope_theta": 50000.0,
        "rotary_dim": 64,
        "sliding_window": 16,
        "logit_scale": 1.0,
        "layer_kinds": [
            "dense_full_attention",
            "moe_sliding_attention",
            "moe_sliding_attention",
            "moe_sliding_attention",
        ],
        "tie_word_embeddings": True,
    }
    return {
        "config": config,
        "entries": entries,
        "template": build_ir_v8._load_builtin_template_doc("cohere2_moe"),
        "quant_summary": {
            "token_emb": "q8_0",
            "lm_head": "q8_0",
            "layer.0": {
                "wq": "q8_0",
                "wk": "q8_0",
                "wv": "q8_0",
                "wo": "q8_0",
                "w1": "q8_0",
                "w3": "q8_0",
                "w2": "q8_0",
            },
            **{
                f"layer.{layer}": {
                    "wq": "q8_0",
                    "wk": "q8_0",
                    "wv": "q8_0",
                    "wo": "q8_0",
                    "moe_router": "fp32",
                    "moe_expert_gate": "q4_k",
                    "moe_expert_up": "q4_k",
                    "moe_expert_down": "q5_k",
                }
                for layer in range(1, 4)
            },
        },
    }


class Cohere2MoeContractTests(unittest.TestCase):
    def test_model_map_owns_identity_metadata_and_layer_plan(self) -> None:
        contract = converter.gguf_ck_arch_contract("cohere2moe")
        self.assertEqual(contract["template"], "cohere2_moe")
        self.assertEqual(contract["conversion_family"], "mapped_hybrid")
        self.assertEqual(
            converter.gguf_ck_layer_kinds_from_map(
                "cohere2moe",
                8,
                {"cohere2moe.leading_dense_block_count": 1},
            ),
            [
                "dense_full_attention",
                "moe_sliding_attention",
                "moe_sliding_attention",
                "moe_sliding_attention",
                "moe_full_attention",
                "moe_sliding_attention",
                "moe_sliding_attention",
                "moe_sliding_attention",
            ],
        )
        self.assertEqual(
            contract["tensor_map"]["blk.{L}.ffn_gate_exps.weight"],
            "layer.{L}.moe_expert_gate",
        )

    def test_conversion_keeps_attention_output_width_distinct_from_hidden_width(self) -> None:
        plan = converter.build_uniform_attention_width_plan(
            num_layers=4,
            num_heads=8,
            q_head_dim=64,
            v_head_dim=32,
        )
        self.assertEqual(plan["layer_q_dim"], [512] * 4)
        self.assertEqual(plan["attn_out_dim"], 256)
        self.assertEqual(plan["layer_attention_output_dim"], [256] * 4)

    def test_circuit_preserves_parallel_residual_and_rope_policy(self) -> None:
        circuit = build_ir_v8._load_builtin_template_doc("cohere2_moe")
        self.assertTrue(circuit["flags"]["parallel_attention_mlp"])
        kinds = circuit["block_types"]["decoder"]["body"]["ops_by_kind"]
        self.assertNotIn("rope_qk", kinds["dense_full_attention"])
        self.assertNotIn("rope_qk", kinds["moe_full_attention"])
        self.assertIn("rope_qk", kinds["moe_sliding_attention"])
        for kind, ops in kinds.items():
            names = [item["op"] if isinstance(item, dict) else item for item in ops]
            self.assertEqual(names[-2:], ["residual_add", "residual_add"], kind)
            self.assertEqual(
                ops[-2]["graph_slots"]["inputs"],
                {"a": "main_stream", "b": "attention_residual"},
            )
            self.assertEqual(
                ops[-1]["graph_slots"]["inputs"],
                {"a": "main_stream", "b": "attention_output"},
            )

    def test_prefill_ir_selects_dense_then_routed_blocks(self) -> None:
        manifest = _tiny_manifest()
        operations = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "cohere2_moe_manifest.synthetic.json",
            mode="prefill",
        )
        layer_zero = [op["op"] for op in operations if op.get("layer") == 0]
        layer_one = [op["op"] for op in operations if op.get("layer") == 1]
        self.assertIn("mlp_gate_up", layer_zero)
        self.assertNotIn("moe_router", layer_zero)
        self.assertIn("moe_router", layer_one)
        self.assertIn("group_limited_topk_router", layer_one)
        self.assertIn("moe_swiglu_expert_mlp", layer_one)
        self.assertIn("rope_qk", layer_one)

    def test_quantized_dense_prefix_consumes_planner_owned_q8_inputs(self) -> None:
        manifest = _tiny_manifest()
        registry = build_ir_v8.load_kernel_registry()
        ir1 = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "cohere2_moe_manifest.synthetic.json",
            mode="prefill",
        )
        lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, manifest, "prefill")
        layout = build_ir_v8.generate_memory_layout(
            lower1,
            manifest,
            registry,
            mode="prefill",
            context_len=8,
        )
        lower2 = build_ir_v8.generate_ir_lower_2(
            lower1,
            layout,
            manifest,
            registry,
            mode="prefill",
        )
        call_ir = build_ir_v8.generate_ir_lower_3(lower2, "prefill")
        layer_zero = {
            op["op"]: op
            for op in call_ir["operations"]
            if op.get("layer") == 0
        }

        for op_name in ("out_proj", "mlp_gate_up", "mlp_down"):
            op = layer_zero[op_name]
            a_arg = next(arg for arg in op["args"] if arg["name"] == "A")
            self.assertEqual(a_arg["buffer_ref"], "layer_input", op_name)

        out_proj = layer_zero["out_proj"]
        k_arg = next(arg for arg in out_proj["args"] if arg["name"] == "K")
        self.assertEqual(k_arg["expr"], str(manifest["config"]["attn_out_dim"]))
        out_quant = layer_zero["quantize_out_proj_input"]
        quant_k = next(arg for arg in out_quant["args"] if arg["name"] == "k")
        self.assertEqual(quant_k["expr"], str(manifest["config"]["attn_out_dim"]))

    def test_compiler_and_codegen_remain_model_name_free(self) -> None:
        for relative in (
            "version/v8/scripts/build_ir_v8.py",
            "version/v8/scripts/codegen_core_v8.py",
            "version/v8/scripts/codegen_prefill_v8.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8").lower()
            self.assertNotIn("cohere2moe", source, relative)
            self.assertNotIn("cohere2_moe", source, relative)


if __name__ == "__main__":
    unittest.main()
