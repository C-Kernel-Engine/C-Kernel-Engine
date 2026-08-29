#!/usr/bin/env python3
from __future__ import annotations

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
import convert_safetensors_to_bump_v8 as converter  # type: ignore
import run_multimodal_bridge_v8 as bridge  # type: ignore
from tests.test_v8_cohere2_contract import _make_tiny_cohere2_manifest


def _header(name: str, shape: list[int]) -> converter.HeaderTensor:
    return converter.HeaderTensor(name=name, dtype="BF16", shape=shape, shard=Path("model.safetensors"))


class CohereCompassContractTests(unittest.TestCase):
    def test_model_map_detects_official_architecture(self) -> None:
        config = {
            "model_type": "cohere_compass",
            "architectures": ["CohereCompassForConditionalGeneration"],
            "text_config": {"model_type": "cohere_compass_text"},
        }
        self.assertEqual(converter._infer_arch(config), "cohere_compass_text")
        text = converter._safetensors_arch_contract("cohere_compass_text")
        vision = converter._safetensors_arch_contract("cohere_compass_vision")
        self.assertEqual(text["config_builder"], "cohere2_text")
        self.assertEqual(vision["tensor_mapper"], "qwen3_vl_vision")

    def test_text_config_preserves_hybrid_attention_and_interleaved_mrope(self) -> None:
        model_config = {
            "model_type": "cohere_compass",
            "architectures": ["CohereCompassForConditionalGeneration"],
            "image_token_id": 255031,
            "video_token_id": 255032,
            "text_config": {
                "model_type": "cohere_compass_text",
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "max_position_embeddings": 128,
                "layer_norm_eps": 1e-5,
                "sliding_window": 8,
                "logit_scale": 0.25,
                "tie_word_embeddings": True,
                "attention_bias": False,
                "layer_types": [
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                "rope_parameters": {
                    "sliding_attention": {
                        "mrope_interleaved": True,
                        "mrope_section": [1, 1, 0],
                        "rope_theta": 50000,
                    },
                    "full_attention": None,
                },
            },
            "vision_config": {"deepstack_visual_indexes": [0, 1, 2]},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
            config = converter._build_config(root, "cohere_compass_text", None)

        self.assertEqual(config["layer_kinds"], model_config["text_config"]["layer_types"])
        self.assertEqual(config["mrope_sections"], [1, 1, 0, 0])
        self.assertTrue(config["mrope_interleaved"])
        self.assertEqual(config["rope_theta"], 50000.0)
        self.assertEqual(config["sliding_window"], 8)
        self.assertEqual(config["logit_scale"], 0.25)
        self.assertEqual(config["num_deepstack_layers"], 3)

    def test_text_tensor_map_owns_parallel_layernorm_and_swiglu_weights(self) -> None:
        prefix = "model.language_model.layers.0"
        headers = {
            item.name: item
            for item in [
                _header("model.language_model.embed_tokens.weight", [64, 16]),
                _header(f"{prefix}.input_layernorm.weight", [16]),
                _header(f"{prefix}.self_attn.q_proj.weight", [16, 16]),
                _header(f"{prefix}.self_attn.k_proj.weight", [8, 16]),
                _header(f"{prefix}.self_attn.v_proj.weight", [8, 16]),
                _header(f"{prefix}.self_attn.o_proj.weight", [16, 16]),
                _header(f"{prefix}.mlp.gate_proj.weight", [32, 16]),
                _header(f"{prefix}.mlp.up_proj.weight", [32, 16]),
                _header(f"{prefix}.mlp.down_proj.weight", [16, 32]),
                _header("model.language_model.norm.weight", [16]),
                _header("model.visual.pos_embed.weight", [4, 8]),
            ]
        }
        config = {
            "num_layers": 1,
            "embed_dim": 16,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 4,
        }
        refs = converter._refs_for_arch("cohere_compass_text", config, headers)
        by_name = {ref.ck_name: ref for ref in refs}
        self.assertEqual(
            by_name["layer.0.w1"].source_names,
            (f"{prefix}.mlp.gate_proj.weight", f"{prefix}.mlp.up_proj.weight"),
        )
        self.assertEqual(by_name["layer.0.ln1_gamma"].source_names, by_name["layer.0.ln2_gamma"].source_names)
        self.assertEqual(by_name["layer.0.ln1_beta"].synth, "zeros_fp32")
        audit = converter._build_source_audit("cohere_compass_text", headers, refs)
        self.assertEqual(audit["verdict"], "pass")
        self.assertIn(
            {
                "source": "model.visual.pos_embed.weight",
                "reason": "vision_tower_not_in_decoder_artifact",
            },
            audit["ignored_source_tensors"],
        )

    def test_circuits_reuse_providers_but_keep_cohere_identity(self) -> None:
        text = build_ir_v8._load_builtin_template_doc("cohere_compass_text")
        vision = build_ir_v8._load_builtin_template_doc("cohere_compass_vision")
        composition = build_ir_v8._load_builtin_template_doc("cohere_compass")

        self.assertEqual(text["name"], "cohere_compass_text")
        self.assertEqual(text["inherited_from"], "cohere2")
        self.assertEqual(text["kernels"]["rope_qk"], "mrope_qk_text_imrope")
        self.assertEqual(text["kernels"]["rope_qk_decode"], "mrope_qk_text_imrope")
        self.assertIn("decoder.mrope", text["required_numerical_contracts"])
        self.assertEqual(text["contract"]["block_contract"]["body_type"], "parallel_attention_swiglu")
        self.assertEqual(vision["name"], "cohere_compass_vision")
        self.assertEqual(vision["inherited_from"], "qwen3_vl_vision")
        self.assertEqual(vision["block_types"]["vision_encoder"]["branches"][0]["name"], "deepstack")
        self.assertEqual(
            composition["resolved_components"]["decoder"]["circuit"],
            "cohere_compass_text",
        )

    def test_composition_requires_three_deepstack_outputs_and_matching_width(self) -> None:
        circuit = bridge._load_explicit_composition_circuit("cohere_compass")
        evidence = bridge._validate_composition_runtime(
            circuit,
            encoder_config={
                "num_deepstack_layers": 3,
                "deepstack_layer_indices": [8, 16, 24],
                "projector_out_dim": 2048,
            },
            decoder_config={"embed_dim": 2048},
        )
        self.assertEqual(evidence["status"], "validated")
        stitch = bridge._composition_bridge_contract(circuit)
        self.assertEqual(stitch["deepstack_injections"], 3)
        self.assertEqual(stitch["position_policy"], "mrope_2d")

    def test_text_mrope_builds_call_ready_prefill_and_decode(self) -> None:
        manifest = _make_tiny_cohere2_manifest()
        manifest["config"].update(
            {
                "model": "cohere_compass_text",
                "arch": "cohere_compass_text",
                "context_length": 500_000,
                "max_seq_len": 500_000,
                "prefill_chunk_length": 8,
                "rope_layout": "multi_section_1d",
                "mrope_sections": [1, 1, 0, 0],
                "mrope_n_dims": 4,
                "mrope_interleaved": True,
            }
        )
        manifest["template"] = build_ir_v8._load_builtin_template_doc(
            "cohere_compass_text"
        )
        registry = build_ir_v8.load_kernel_registry()
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                ir1 = build_ir_v8.build_ir1_direct(
                    manifest,
                    ROOT / "tests" / "cohere_compass_manifest.synthetic.json",
                    mode=mode,
                )
                rope_ops = [op for op in ir1 if op.get("op") == "rope_qk"]
                self.assertTrue(rope_ops)
                self.assertTrue(
                    all(op["kernel"] == "mrope_qk_text_imrope" for op in rope_ops)
                )
                lower1 = build_ir_v8.generate_ir_lower_1(
                    ir1, registry, manifest, mode
                )
                layout = build_ir_v8.generate_memory_layout(
                    lower1,
                    manifest,
                    registry,
                    mode=mode,
                    context_len=8,
                )
                lower2 = build_ir_v8.generate_ir_lower_2(
                    lower1, layout, manifest, registry, mode=mode
                )
                residual_saves = [
                    op for op in lower2["operations"] if op.get("op") == "residual_save"
                ]
                self.assertTrue(residual_saves)
                expected_rows = 1 if mode == "decode" else 8
                self.assertTrue(
                    all(
                        op["params"]["_memcpy_bytes"]
                        == expected_rows * manifest["config"]["embed_dim"] * 4
                        for op in residual_saves
                    )
                )
                validation = build_ir_v8._validate_lowered_activation_memory(
                    {"operations": residual_saves}, layout
                )
                self.assertEqual(
                    validation["copy_write_count"], len(residual_saves)
                )
                oversized = json.loads(json.dumps(residual_saves))
                oversized[0]["params"]["_memcpy_bytes"] += 1
                with self.assertRaisesRegex(
                    RuntimeError, "residual_save.*only .* bytes are available"
                ):
                    build_ir_v8._validate_lowered_activation_memory(
                        {"operations": oversized}, layout
                    )
                call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode)
                self.assertFalse(
                    [op for op in call_ir["operations"] if op.get("errors")]
                )

    def test_compiler_and_codegen_do_not_branch_on_cohere_compass(self) -> None:
        for relative in (
            "version/v8/scripts/build_ir_v8.py",
            "version/v8/scripts/codegen_core_v8.py",
            "version/v8/scripts/codegen_prefill_v8.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8").lower()
            self.assertNotIn("cohere_compass", source, relative)


if __name__ == "__main__":
    unittest.main()
