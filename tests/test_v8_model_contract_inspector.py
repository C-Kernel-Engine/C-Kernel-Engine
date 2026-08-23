from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "version/v8/scripts/inspect_model_contract_v8.py"
AUDIT_SCRIPT = REPO / "version/v8/scripts/audit_safetensors_index_v8.py"


def _instella_config() -> dict:
    return {
        "architectures": ["InstellaMoEForCausalLM"],
        "model_type": "deepseek_v3",
        "farskip": True,
        "gated_attention": True,
        "hidden_size": 2048,
        "intermediate_size": 10944,
        "moe_intermediate_size": 1408,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "num_experts_per_tok": 6,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "routed_scaling_factor": 2.5,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 96,
        "qk_rope_head_dim": 32,
        "v_head_dim": 128,
        "rope_interleave": True,
        "rope_scaling": {"type": "yarn", "factor": 40},
    }


def _instella_index() -> dict:
    shard_names = [f"model-{index:05d}-of-00006.safetensors" for index in range(1, 7)]
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for layer in range(27):
        prefix = f"model.layers.{layer}"
        names.extend([
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
            f"{prefix}.self_attn.kv_a_layernorm.weight",
            f"{prefix}.self_attn.kv_b_proj.weight",
            f"{prefix}.self_attn.gate_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
        ])
        if layer == 0:
            names.extend([
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ])
            continue
        names.extend([
            f"{prefix}.mlp.gate.weight",
            f"{prefix}.mlp.gate.e_score_correction_bias",
            f"{prefix}.mlp.shared_experts.gate_proj.weight",
            f"{prefix}.mlp.shared_experts.up_proj.weight",
            f"{prefix}.mlp.shared_experts.down_proj.weight",
        ])
        for expert in range(64):
            expert_prefix = f"{prefix}.mlp.experts.{expert}"
            names.extend([
                f"{expert_prefix}.gate_proj.weight",
                f"{expert_prefix}.up_proj.weight",
                f"{expert_prefix}.down_proj.weight",
            ])
    assert len(names) == 5344
    return {
        "metadata": {"total_size": 31725581824},
        "weight_map": {
            name: shard_names[index % len(shard_names)]
            for index, name in enumerate(names)
        },
    }


class ModelContractInspectorTests(unittest.TestCase):
    def test_nemotron_h_reports_supported_config_after_mamba2_reference_kernels(self) -> None:
        cfg = {
            "architectures": ["NemotronHForCausalLM"],
            "model_type": "nemotron_h",
            "hidden_size": 4096,
            "intermediate_size": 21504,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 10,
            "hybrid_override_pattern": "MEMEM*EMEM",
            "mamba_num_heads": 128,
            "mamba_head_dim": 64,
            "ssm_state_size": 128,
            "conv_kernel": 4,
            "mlp_hidden_act": "relu2",
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["arch"], "nemotron_h")
        self.assertEqual(report["status"], "supported")
        self.assertEqual(report["layer_kind_counts"], {"attention": 1, "mamba": 5, "moe": 4})
        self.assertEqual(report["missing_ops"], [])
        self.assertEqual(report["kernel_registry"]["missing_kernel_ops"], [])
        self.assertEqual(report["template_suggestion"]["candidate_template"], "nemotron_h.json")
        self.assertIn("mamba", report["template_suggestion"]["block_sketch"]["body"]["ops_by_kind"])
        self.assertIn("moe", report["template_suggestion"]["block_sketch"]["body"]["ops_by_kind"])
        self.assertNotIn("mamba_in_proj_split", report["missing_ops"])
        self.assertNotIn("mamba_conv1d_state_update", report["missing_ops"])
        self.assertNotIn("mamba_dt_softplus", report["missing_ops"])
        self.assertNotIn("mamba_rmsnorm_gate", report["missing_ops"])
        self.assertNotIn("mamba_out_proj", report["missing_ops"])
        self.assertNotIn("relu2_mlp", report["missing_ops"])
        self.assertNotIn("shared_expert_mlp", report["missing_ops"])
        self.assertNotIn("group_limited_topk_router", report["missing_ops"])
        self.assertNotIn("moe_relu2_expert_mlp", report["missing_ops"])
        self.assertNotIn("safetensors_to_bump_mapping", report["missing_ops"])
        self.assertNotIn("v8_template_contract", report["missing_ops"])

    def test_safetensors_index_audit_classifies_nemotron_families(self) -> None:
        index = {
            "metadata": {"total_parameters": 123, "total_size": 456},
            "weight_map": {
                "backbone.embeddings.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.norm.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.in_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.conv1d.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.out_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.1.norm.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.1.mixer.q_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.1.mixer.k_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.1.mixer.v_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.1.mixer.o_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.2.norm.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.2.mixer.gate.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.2.mixer.gate.e_score_correction_bias": "model-00001-of-00001.safetensors",
                "backbone.layers.2.mixer.experts.0.up_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.2.mixer.experts.0.down_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.2.mixer.shared_experts.up_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.2.mixer.shared_experts.down_proj.weight": "model-00001-of-00001.safetensors",
                "lm_head.weight": "model-00001-of-00001.safetensors",
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.safetensors.index.json"
            path.write_text(json.dumps(index), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(AUDIT_SCRIPT), str(path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
        report = json.loads(proc.stdout)
        self.assertEqual(report["families"]["mamba"], 3)
        self.assertEqual(report["families"]["attention"], 4)
        self.assertEqual(report["families"]["moe_router"], 2)
        self.assertEqual(report["families"]["moe_expert"], 2)
        self.assertEqual(report["families"]["moe_shared_expert"], 2)
        self.assertEqual(report["layers"]["2"]["expert_count"], 1)


    def test_kimi_vl_reports_mla_moe_bringup_contract(self) -> None:
        cfg = {
            "architectures": ["KimiVLForConditionalGeneration"],
            "model_type": "kimi_vl",
            "vision_config": {
                "model_type": "moonvit",
                "patch_size": 14,
                "num_attention_heads": 16,
                "num_hidden_layers": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "merge_kernel_size": [2, 2],
            },
            "text_config": {
                "vocab_size": 163840,
                "max_position_embeddings": 131072,
                "hidden_size": 2048,
                "intermediate_size": 11264,
                "moe_intermediate_size": 1408,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "n_shared_experts": 2,
                "n_routed_experts": 64,
                "kv_lora_rank": 512,
                "q_lora_rank": None,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "qk_nope_head_dim": 128,
                "topk_method": "noaux_tc",
                "n_group": 1,
                "topk_group": 1,
                "num_experts_per_tok": 6,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
                "norm_topk_prob": True,
                "scoring_func": "sigmoid",
                "routed_scaling_factor": 2.446,
                "hidden_act": "silu",
                "rope_theta": 800000.0,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.assertEqual(proc.returncode, 2, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["arch"], "kimi_vl")
        self.assertEqual(report["status"], "bringup_required")
        self.assertEqual(report["layer_kind_counts"], {"mla_dense_mlp": 1, "mla_moe": 3})
        self.assertEqual(report["template_suggestion"]["candidate_template"], "kimi_vl.json")
        self.assertEqual(report["template_suggestion"]["modalities"], ["text", "vision"])
        self.assertIn("mla_moe", report["template_suggestion"]["block_sketch"]["body"]["ops_by_kind"])
        for op in (
            "mla_attention",
            "kv_lora_decompress",
            "group_limited_topk_router",
            "moe_swiglu_expert_mlp",
            "shared_swiglu_expert_mlp",
            "moonvit_encoder",
            "tiktoken_tokenizer",
        ):
            self.assertIn(op, report["required_ops"])
        for missing in (
            "kimi_vl_safetensors_to_bump_mapping",
            "mla_attention_contract",
            "tiktoken_tokenizer_contract",
            "moonvit_bridge_contract",
        ):
            self.assertIn(missing, report["missing_ops"])
        missing_kernel_ops = {
            row["op"] for row in report["kernel_registry"]["missing_kernel_ops"]
        }
        self.assertIn("moonvit_encoder", missing_kernel_ops)
        self.assertIn("moonvit_projector", missing_kernel_ops)
        self.assertIn("media_placeholder_merge", missing_kernel_ops)
        self.assertIn("tiktoken_tokenizer", missing_kernel_ops)
        self.assertNotIn("group_limited_topk_router", missing_kernel_ops)
        self.assertNotIn("moe_swiglu_expert_mlp", missing_kernel_ops)
        self.assertNotIn("shared_swiglu_expert_mlp", missing_kernel_ops)
        self.assertNotIn("moe_swiglu_expert_mlp", report["missing_ops"])
        self.assertNotIn("shared_swiglu_expert_mlp", report["missing_ops"])
        self.assertNotIn("kv_lora_decompress_contract", report["missing_ops"])
        self.assertNotIn("kimi_vl_template_contract", report["missing_ops"])
        self.assertNotIn("v8_template_contract", report["missing_ops"])

    def test_instella_is_not_misclassified_as_stock_deepseek_v3(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            path.write_text(json.dumps(_instella_config()), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["arch"], "instella_moe")
        self.assertEqual(report["model_type"], "deepseek_v3")
        self.assertEqual(
            report["layer_kind_counts"],
            {"mla_gated_dense_mlp": 1, "mla_gated_farskip_moe": 26},
        )
        self.assertIn("attention_gate_sigmoid_mul", report["required_ops"])
        self.assertEqual(report["status"], "supported")
        self.assertNotIn("farskip_two_stream_residual", report["required_ops"])
        self.assertNotIn("farskip_routed_shared_combine", report["missing_ops"])
        self.assertNotIn("mla_interleaved_yarn_contract", report["missing_ops"])
        self.assertNotIn("safetensors_to_bump_mapping", report["missing_ops"])
        self.assertNotIn("v8_template_contract", report["missing_ops"])

    def test_real_instella_index_shape_contract_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.safetensors.index.json"
            path.write_text(json.dumps(_instella_index()), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(AUDIT_SCRIPT), str(path), "--arch", "instella_moe"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["tensor_count"], 5344)
        self.assertEqual(report["shard_count"], 6)
        self.assertEqual(report["layer_count"], 27)
        self.assertEqual(report["required_tensor_patterns"]["missing"], [])
        self.assertEqual(report["declared_index_contract"]["failures"], [])
        self.assertEqual(report["layers"]["0"]["expert_count"], 0)
        self.assertEqual(report["layers"]["26"]["expert_count"], 64)

    def test_instella_index_contract_rejects_missing_expert(self) -> None:
        index = _instella_index()
        del index["weight_map"]["model.layers.26.mlp.experts.63.down_proj.weight"]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.safetensors.index.json"
            path.write_text(json.dumps(index), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(AUDIT_SCRIPT), str(path), "--arch", "instella_moe"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.assertEqual(proc.returncode, 2, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["status"], "fail")
        failures = report["declared_index_contract"]["failures"]
        self.assertTrue(any("tensor_count" in row for row in failures))

    def test_safetensors_index_audit_classifies_kimi_mla_moe_families(self) -> None:
        index = {
            "metadata": {"total_parameters": 123, "total_size": 456},
            "weight_map": {
                "language_model.lm_head.weight": "model-00001-of-00001.safetensors",
                "language_model.model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.input_layernorm.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.kv_a_layernorm.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.kv_a_proj_with_mqa.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.kv_b_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.mlp.gate_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.mlp.up_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.mlp.down_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.gate.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.gate.e_score_correction_bias": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.experts.0.gate_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.experts.0.up_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.experts.0.down_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.shared_experts.up_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.shared_experts.down_proj.weight": "model-00001-of-00001.safetensors",
                "vision_tower.encoder.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                "multi_modal_projector.linear_1.weight": "model-00001-of-00001.safetensors",
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.safetensors.index.json"
            path.write_text(json.dumps(index), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(AUDIT_SCRIPT), str(path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
        report = json.loads(proc.stdout)
        self.assertEqual(report["families"]["mla_attention"], 3)
        self.assertEqual(report["families"]["moe_router"], 2)
        self.assertEqual(report["families"]["moe_expert"], 3)
        self.assertEqual(report["families"]["moe_shared_expert"], 3)
        self.assertEqual(report["families"]["vision_tower"], 1)
        self.assertEqual(report["families"]["multimodal_projector"], 1)
        self.assertEqual(report["layers"]["1"]["expert_count"], 1)


    def test_kimi_safetensors_index_validates_required_model_map_patterns(self) -> None:
        index = {
            "metadata": {"total_parameters": 123, "total_size": 456},
            "weight_map": {
                "language_model.model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.kv_a_proj_with_mqa.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.kv_a_layernorm.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.kv_b_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.o_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.gate.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.experts.0.gate_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight": "model-00001-of-00001.safetensors",
                "vision_tower.encoder.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                "multi_modal_projector.linear_1.weight": "model-00001-of-00001.safetensors",
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.safetensors.index.json"
            path.write_text(json.dumps(index), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(AUDIT_SCRIPT), str(path), "--arch", "kimi_vl"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["model_map_status"], "known")
        self.assertEqual(report["required_tensor_patterns"]["missing"], [])
        expert_row = next(row for row in report["required_tensor_patterns"]["patterns"] if "experts.{E}.gate_proj" in row["pattern"])
        self.assertEqual(expert_row["expert_counts"], {"1": 1})

    def test_kimi_safetensors_index_fails_when_required_mla_tensor_missing(self) -> None:
        index = {
            "metadata": {"total_parameters": 123, "total_size": 456},
            "weight_map": {
                "language_model.model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.kv_a_proj_with_mqa.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.kv_a_layernorm.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.o_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.gate.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.experts.0.gate_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight": "model-00001-of-00001.safetensors",
                "vision_tower.encoder.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                "multi_modal_projector.linear_1.weight": "model-00001-of-00001.safetensors",
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.safetensors.index.json"
            path.write_text(json.dumps(index), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(AUDIT_SCRIPT), str(path), "--arch", "kimi_vl"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.assertEqual(proc.returncode, 2, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["status"], "fail")
        self.assertIn("language_model.model.layers.{L}.self_attn.kv_b_proj.weight", report["required_tensor_patterns"]["missing"])

    def test_qwen3_dense_config_is_supported_at_contract_level(self) -> None:
        cfg = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_hidden_layers": 4,
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
        report = json.loads(proc.stdout)
        self.assertEqual(report["arch"], "qwen3")
        self.assertEqual(report["status"], "supported")
        self.assertEqual(report["layer_kind_counts"], {"attention": 4})
        self.assertEqual(report["missing_ops"], [])
        self.assertEqual(report["kernel_registry"]["missing_kernel_ops"], [])
        self.assertEqual(report["template_suggestion"]["candidate_template"], "qwen3.json")
        self.assertIn("attention", report["template_suggestion"]["block_sketch"]["body"]["ops_by_kind"])

    def test_inspector_weights_audit_reads_safetensors_index_from_model_dir(self) -> None:
        cfg = {
            "architectures": ["KimiVLForConditionalGeneration"],
            "model_type": "kimi_vl",
            "vision_config": {"model_type": "moonvit", "num_hidden_layers": 1},
            "text_config": {
                "hidden_size": 2048,
                "intermediate_size": 11264,
                "num_hidden_layers": 2,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "first_k_dense_replace": 1,
                "moe_layer_freq": 1,
                "hidden_act": "silu",
            },
        }
        index = {
            "metadata": {"total_parameters": 123, "total_size": 456},
            "weight_map": {
                "language_model.model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.kv_a_proj_with_mqa.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.kv_a_layernorm.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.kv_b_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.0.self_attn.o_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.gate.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.experts.0.gate_proj.weight": "model-00001-of-00001.safetensors",
                "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight": "model-00001-of-00001.safetensors",
                "vision_tower.encoder.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                "multi_modal_projector.linear_1.weight": "model-00001-of-00001.safetensors",
            },
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(root), "--weights-audit"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.assertEqual(proc.returncode, 2, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["weights_audit"]["status"], "pass")
        self.assertEqual(report["weights_audit"]["model_map_status"], "known")
        self.assertEqual(report["weights_audit"]["families"]["mla_attention"], 3)
        self.assertEqual(report["weights_audit"]["required_tensor_patterns"]["missing"], [])


if __name__ == "__main__":
    unittest.main()
