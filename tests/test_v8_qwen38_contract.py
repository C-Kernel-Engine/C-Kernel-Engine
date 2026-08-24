#!/usr/bin/env python3
import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import convert_gguf_to_bump_v8 as converter  # type: ignore
import build_ir_v8  # type: ignore
import codegen_core_v8  # type: ignore
import codegen_prefill_v8  # type: ignore


class Qwen38ContractTests(unittest.TestCase):
    def test_exact_artifact_metadata_selects_qwen38(self) -> None:
        metadata = {
            "general.architecture": "qwen35",
            "general.basename": "Qwen3.8-27B",
        }
        contract = converter.gguf_ck_artifact_contract("qwen35", metadata)
        self.assertEqual(contract["artifact_variant"], "qwen38-27b-unsloth")
        self.assertEqual(contract["runtime_arch"], "qwen38")
        self.assertEqual(converter.gguf_ck_template_arch("qwen35", metadata), "qwen38")

    def test_other_qwen35_artifacts_remain_qwen35(self) -> None:
        metadata = {
            "general.architecture": "qwen35",
            "general.basename": "Qwen3.6-27B",
        }
        contract = converter.gguf_ck_artifact_contract("qwen35", metadata)
        self.assertNotIn("artifact_variant", contract)
        self.assertEqual(converter.gguf_ck_template_arch("qwen35", metadata), "qwen35")

    def test_variant_match_metadata_is_retained_by_gguf_reader(self) -> None:
        self.assertIn("general.basename", converter.gguf_ck_artifact_match_keys())
        self.assertIn("general.size_label", converter.gguf_ck_artifact_match_keys())

        model_map = converter.load_gguf_ck_map()["architectures"]["qwen35"]
        variants = {row["id"]: row for row in model_map["artifact_variants"]}
        self.assertEqual(variants["qwen38-27b-unsloth"]["template"], "qwen38")
        self.assertEqual(variants["qwen38-27b-standard"]["runtime_arch"], "qwen38")

    def test_standard_qwen38_gguf_identity_selects_qwen38(self) -> None:
        metadata = {
            "general.architecture": "qwen35",
            "general.basename": "Qwen3.8",
            "general.size_label": "27B",
        }
        contract = converter.gguf_ck_artifact_contract("qwen35", metadata)
        self.assertEqual(contract["artifact_variant"], "qwen38-27b-standard")
        self.assertEqual(converter.gguf_ck_template_arch("qwen35", metadata), "qwen38")

    def test_ambiguous_artifact_variants_fail_closed(self) -> None:
        original = converter._GGUF_CK_MAP_CACHE
        model_map = copy.deepcopy(converter.load_gguf_ck_map())
        duplicate = copy.deepcopy(model_map["architectures"]["qwen35"]["artifact_variants"][0])
        duplicate["id"] = "duplicate"
        model_map["architectures"]["qwen35"]["artifact_variants"].append(duplicate)
        try:
            converter._GGUF_CK_MAP_CACHE = model_map
            with self.assertRaisesRegex(converter.GGUFError, "multiple variants"):
                converter.gguf_ck_artifact_contract(
                    "qwen35", {"general.basename": "Qwen3.8-27B"}
                )
        finally:
            converter._GGUF_CK_MAP_CACHE = original

    def test_qwen38_circuit_is_dense_and_model_owned(self) -> None:
        path = ROOT / "version" / "v8" / "circuits" / "qwen38.json"
        circuit = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(circuit["name"], "qwen38")
        self.assertEqual(circuit["family"], "qwen38")
        self.assertEqual(circuit["flags"]["mlp_topology"], "dense_swiglu")

        serialized = json.dumps(circuit)
        self.assertNotIn("qwen35moe", serialized)
        self.assertNotIn("moe_router", serialized)
        self.assertNotIn("moe_swiglu_expert_mlp", serialized)

        body = circuit["block_types"]["decoder"]["body"]["ops_by_kind"]
        dense_tail = [
            "post_attention_norm",
            "mlp_gate_up",
            "silu_mul",
            "mlp_down",
            "residual_add",
        ]
        self.assertEqual(body["recurrent"][-5:], dense_tail)
        self.assertEqual(body["full_attention"][-5:], dense_tail)

    def test_circuit_runtime_defaults_are_applied_generically(self) -> None:
        circuit = converter.load_template_for_arch("qwen38")
        config = converter.apply_circuit_runtime_defaults({}, circuit)
        self.assertTrue(config["prefer_q8_0_contract"])
        self.assertEqual(config["prefill_chunk_length"], 4096)
        self.assertEqual(config["logits_layout"], "last")
        self.assertEqual(
            config["activation_preference_by_op"]["recurrent_gate_proj"],
            "q8_k",
        )

    def test_generic_q5_k_projection_remains_visible_to_xray(self) -> None:
        emitted = codegen_core_v8.emit_op({
            "idx": 1,
            "layer": 3,
            "section": "body",
            "op": "k_proj",
            "function": "gemv_q5_k",
            "args": [
                {"name": "y", "expr": "k_output"},
                {"name": "W", "expr": "k_weight"},
                {"name": "x", "expr": "hidden"},
                {"name": "M", "expr": "1024"},
                {"name": "K", "expr": "5120"},
            ],
        })
        self.assertIn(
            'ck_debug_export_hidden(model, 3, "k_proj", '
            '(const float*)k_output, 1024);',
            emitted,
        )

    def test_262k_plan_separates_capacity_from_transient_extent(self) -> None:
        context_capacity = 262_144
        config = {
            "num_layers": 64,
            "embed_dim": 5120,
            "num_heads": 24,
            "num_kv_heads": 4,
            "head_dim": 256,
            "intermediate_size": 17_408,
            "vocab_size": 248_320,
            "context_length": context_capacity,
            "decode_kv_cache_dtype": "fp16",
            "prefill_chunk_length": 4096,
            "logits_layout": "last",
            "layer_kv_policy": [
                "produce" if (layer + 1) % 4 == 0 else "none"
                for layer in range(64)
            ],
            "_template_uses_kv_cache": True,
            "_template_uses_rope": True,
            "_template_has_logits": True,
        }

        specs = build_ir_v8.build_activation_specs(
            config,
            mode="prefill",
            context_len=context_capacity,
        )

        expected_kv = 16 * 2 * 4 * context_capacity * 256 * 2
        self.assertEqual(specs["kv_cache"]["size"], expected_kv)
        self.assertEqual(specs["kv_cache"]["dtype"], "fp16")
        self.assertEqual(specs["embedded_input"]["size"], 4096 * 5120 * 4)
        self.assertEqual(specs["logits"]["size"], 248_320 * 4)
        self.assertEqual(config["kv_cache_token_stride_total"], 32_768)
        self.assertEqual(
            sum(offset >= 0 for offset in config["layer_k_cache_offset"]),
            16,
        )

    def test_compact_kv_layout_rejects_partial_or_out_of_bounds_ownership(self) -> None:
        base = {
            "num_layers": 2,
            "num_kv_heads": 2,
            "head_dim": 8,
            "layer_kv_policy": ["none", "produce"],
            "layer_k_cache_offset": [-1, 0],
            "layer_v_cache_offset": [0, 16],
            "kv_cache_token_stride_total": 32,
        }
        with self.assertRaisesRegex(RuntimeError, "ownership must be paired"):
            build_ir_v8._materialize_compact_kv_layout(copy.deepcopy(base))

        out_of_bounds = copy.deepcopy(base)
        out_of_bounds["layer_v_cache_offset"][0] = -1
        out_of_bounds["layer_v_cache_offset"][1] = 24
        with self.assertRaisesRegex(RuntimeError, "interval exceeds token stride"):
            build_ir_v8._materialize_compact_kv_layout(out_of_bounds)

    def test_chunked_prefill_codegen_preserves_absolute_position(self) -> None:
        emitted = codegen_prefill_v8.emit_prefill_function(
            [],
            {
                "context_length": 262_144,
                "prefill_chunk_length": 4096,
            },
        )
        self.assertIn(
            "static void ck_prefill_range(CKModel *model, const int32_t *tokens, "
            "int num_tokens, int prefill_start_pos)",
            emitted,
        )
        self.assertIn("int start_pos = model->pos;", emitted)
        self.assertIn(
            "ck_prefill_range(model, tokens + consumed, chunk, start_pos + consumed);",
            emitted,
        )
        self.assertIn("if (chunk == 1 && consumed > 0)", emitted)
        self.assertIn("ck_decode(model, tokens[consumed]);", emitted)
        self.assertIn("model->pos = prefill_start_pos + num_tokens;", emitted)
        self.assertNotIn("const int prefill_start_pos = 0;", emitted)

    def test_lower2_uses_chunk_extent_instead_of_context_capacity(self) -> None:
        context_capacity = 262_144
        chunk_extent = 4096
        config = {
            "context_length": context_capacity,
            "prefill_chunk_length": chunk_extent,
            "embed_dim": 8,
            "num_heads": 1,
            "num_kv_heads": 1,
            "head_dim": 8,
            "intermediate_size": 16,
            "num_layers": 1,
        }
        op = {
            "idx": 0,
            "kernel": "ck_residual_add_token_major",
            "function": "ck_residual_add_token_major",
            "op": "residual_add",
            "layer": 0,
            "section": "body",
            "inputs": {},
            "outputs": {},
            "weights": {},
            "scratch": [],
            "params": {"seq_len": context_capacity},
        }
        buffers = []
        offset = 0
        for name, size in (
            ("embedded_input", chunk_extent * 8 * 4),
            ("layer_input", chunk_extent * 8 * 4),
            ("residual", chunk_extent * 8 * 4),
            ("mlp_scratch", chunk_extent * 16 * 2 * 4),
            ("kv_cache", 64),
            ("rope_cache", 64),
            ("logits", 64),
        ):
            buffers.append({
                "name": name,
                "offset": offset,
                "size": size,
                "dtype": "fp32",
            })
            offset += size

        lowered = build_ir_v8.generate_ir_lower_2(
            [op],
            {
                "config": config,
                "memory": {
                    "weights": {"entries": []},
                    "activations": {"buffers": buffers},
                },
            },
            {
                "config": config,
                "template": {"contract": {"version": 1}},
            },
            {},
            mode="prefill",
        )
        params = lowered["operations"][0]["params"]
        self.assertEqual(params["seq_len"], chunk_extent)
        self.assertEqual(params["_m"], chunk_extent)


if __name__ == "__main__":
    unittest.main()
