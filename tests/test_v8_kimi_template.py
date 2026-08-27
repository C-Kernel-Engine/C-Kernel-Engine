#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V8_BUILD_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
V8_CONVERTER_PATH = ROOT / "version" / "v8" / "scripts" / "convert_safetensors_to_bump_v8.py"


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


build_ir_v8 = _load_module("build_ir_v8_kimi_tests", V8_BUILD_PATH)
convert_safetensors = _load_module("convert_safetensors_kimi_tests", V8_CONVERTER_PATH)


def _entry(name: str, dtype: str, shape: list[int], offset: int) -> dict:
    nbytes_per = {"fp32": 4, "bf16": 2, "fp16": 2, "q8_0": 1, "q5_0": 1, "q6_k": 1, "q4_k": 1}.get(dtype, 4)
    size = 1
    for dim in shape:
        size *= int(dim)
    return {"name": name, "dtype": dtype, "offset": offset, "shape": shape, "nbytes": size * nbytes_per}


def _make_tiny_kimi_manifest() -> dict:
    offset = 0
    entries = []

    def add(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        item = _entry(name, dtype, shape, offset)
        entries.append(item)
        offset += int(item["nbytes"])

    add("token_emb", "bf16", [32, 8])
    add("final_ln_weight", "fp32", [8])
    add("final_ln_bias", "fp32", [8])
    add("output.weight", "bf16", [32, 8])
    for layer in range(2):
        add(f"layer.{layer}.block_norm", "fp32", [8])
        add(f"layer.{layer}.post_attention_norm", "fp32", [8])
        add(f"layer.{layer}.mla_q_proj", "bf16", [8, 8])
        add(f"layer.{layer}.mla_kv_a_proj", "bf16", [6, 8])
        add(f"layer.{layer}.mla_kv_a_norm", "fp32", [4])
        add(f"layer.{layer}.mla_kv_b_proj", "bf16", [8, 4])
        add(f"layer.{layer}.mla_out_proj", "bf16", [8, 4])
    add("layer.0.mlp_gate", "bf16", [16, 8])
    add("layer.0.mlp_up", "bf16", [16, 8])
    add("layer.0.mlp_down", "bf16", [8, 16])
    add("layer.1.moe_router", "fp32", [2, 8])
    add("layer.1.moe_router_bias", "fp32", [2])
    add("layer.1.moe_expert_gate", "bf16", [2, 4, 8])
    add("layer.1.moe_expert_up", "bf16", [2, 4, 8])
    add("layer.1.moe_expert_down", "bf16", [2, 8, 4])
    add("layer.1.moe_shared_gate", "bf16", [4, 8])
    add("layer.1.moe_shared_up", "bf16", [4, 8])
    add("layer.1.moe_shared_down", "bf16", [8, 4])

    return {
        "config": {
            "model": "kimi_vl",
            "arch": "kimi_vl",
            "model_type": "kimi_vl",
            "num_layers": 2,
            "embed_dim": 8,
            "hidden_size": 8,
            "num_heads": 2,
            "num_kv_heads": 2,
            "head_dim": 4,
            "intermediate_size": 16,
            "intermediate_dim": 16,
            "moe_intermediate_size": 4,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "vocab_size": 32,
            "context_length": 16,
            "layer_kinds": ["mla_dense_mlp", "mla_moe"],
            "layer_attention_policy": ["mla", "mla"],
            "layer_moe_policy": ["none", "routed_swiglu"],
            "rope_layout": "partial_pairwise_concat",
        },
        "quant_summary": {
            "token_emb": "bf16",
            "lm_head": "bf16",
            "final_ln_weight": "fp32",
            "layer.0": {
                "mla_q_proj": "bf16",
                "mla_kv_a_proj": "bf16",
                "mla_kv_a_norm": "fp32",
                "mla_kv_b_proj": "bf16",
                "mla_out_proj": "bf16",
                "mlp_gate": "bf16",
                "mlp_up": "bf16",
                "mlp_down": "bf16",
            },
            "layer.1": {
                "mla_q_proj": "bf16",
                "mla_kv_a_proj": "bf16",
                "mla_kv_a_norm": "fp32",
                "mla_kv_b_proj": "bf16",
                "mla_out_proj": "bf16",
                "moe_router": "fp32",
                "moe_router_bias": "fp32",
                "moe_expert_gate": "bf16",
                "moe_expert_up": "bf16",
                "moe_expert_down": "bf16",
                "moe_shared_gate": "bf16",
                "moe_shared_up": "bf16",
                "moe_shared_down": "bf16",
            },
        },
        "entries": entries,
        "template": build_ir_v8._load_builtin_template_doc("kimi_vl"),
    }


class V8KimiTemplateTests(unittest.TestCase):
    def test_mla_prefill_maps_partition_independent_outputs(self) -> None:
        maps = ROOT / "version/v8/kernel_maps"
        attention = json.loads(
            (maps / "deepseek_mla_attention_f32.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            attention["impl"]["function"],
            "deepseek_mla_attention_f32_parallel_dispatch",
        )
        self.assertEqual(attention["scratch"][0]["shape"], ["H", "T"])
        self.assertEqual(
            attention["parallelization"]["preferred"]["prefill"], "token_row"
        )
        self.assertEqual(
            attention["parallelization"]["strategies"][0]["partition_dim"], "T"
        )

        decompress = json.loads(
            (maps / "deepseek_mla_kv_decompress_bf16.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            decompress["impl"]["function"],
            "deepseek_mla_kv_decompress_bf16_parallel_dispatch",
        )
        self.assertEqual(
            decompress["parallelization"]["preferred"],
            {"prefill": "token", "decode": "serial"},
        )

    def test_bf16_moe_prefill_partitions_independent_rows(self) -> None:
        maps = ROOT / "version/v8/kernel_maps"
        expected = {
            "gemm_nt_bf16.json": (
                "gemm_nt_bf16_parallel_dispatch", "M"
            ),
            "moe_swiglu_expert_forward_bf16.json": (
                "moe_swiglu_expert_forward_bf16_parallel_dispatch", "R"
            ),
            "moe_swiglu_shared_forward_bf16.json": (
                "moe_swiglu_shared_forward_bf16_parallel_dispatch", "R"
            ),
            "farskip_swiglu_shared_combine_bf16.json": (
                "farskip_swiglu_shared_combine_bf16_parallel_dispatch", "R"
            ),
        }
        for name, (function, partition_dim) in expected.items():
            with self.subTest(map=name):
                doc = json.loads((maps / name).read_text(encoding="utf-8"))
                self.assertEqual(
                    doc["parallelization"]["preferred"],
                    {"prefill": "row", "decode": "serial"},
                )
                self.assertEqual(
                    doc["parallelization"]["strategies"][0]["partition_dim"],
                    partition_dim,
                )
                self.assertEqual(doc["impl"]["function"], function)
                self.assertEqual(doc["call_abi"]["version"], 1)

    def test_kimi_nested_text_config_and_tiktoken_sidecar_are_hydrated(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck-kimi-config-") as td:
            root = Path(td)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["KimiVLForConditionalGeneration"],
                        "model_type": "kimi_vl",
                        "text_config": {
                            "num_hidden_layers": 2,
                            "hidden_size": 8,
                            "intermediate_size": 16,
                            "num_attention_heads": 2,
                            "num_key_value_heads": 2,
                            "vocab_size": 32,
                            "max_position_embeddings": 64,
                            "bos_token_id": 28,
                            "eos_token_id": 29,
                            "pad_token_id": 30,
                            "qk_nope_head_dim": 2,
                            "qk_rope_head_dim": 2,
                            "v_head_dim": 2,
                            "kv_lora_rank": 4,
                            "first_k_dense_replace": 1,
                            "moe_layer_freq": 1,
                            "n_routed_experts": 2,
                            "n_shared_experts": 1,
                            "num_experts_per_tok": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "TikTokenTokenizer",
                        "bos_token": "[BOS]",
                        "eos_token": "[EOS]",
                        "pad_token": "[PAD]",
                        "added_tokens_decoder": {
                            "28": {"content": "[BOS]"},
                            "29": {"content": "[EOS]"},
                            "30": {"content": "[PAD]"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "tiktoken.model").write_bytes(b"fixture")
            (root / "tokenization_fixture.py").write_text("class Fixture: pass\n", encoding="utf-8")
            (root / "configuration_fixture.py").write_text("class Fixture: pass\n", encoding="utf-8")

            config = convert_safetensors._build_config(root, "kimi_vl", None)
            payloads, tokenizer_contract, special = (
                convert_safetensors._tokenizer_payloads_from_json(root, 32)
            )
            output = root / "runtime"
            output.mkdir()
            convert_safetensors._copy_tokenizer_sidecars(root, output)
            staged = tuple(
                (output / "tokenizer_source" / name).exists()
                for name in (
                    "config.json",
                    "tiktoken.model",
                    "tokenization_fixture.py",
                    "configuration_fixture.py",
                )
            )

        self.assertEqual(config["bos_token_id"], 28)
        self.assertEqual(config["eos_token_id"], 29)
        self.assertEqual(config["pad_token_id"], 30)
        self.assertEqual(payloads, [])
        self.assertEqual(tokenizer_contract["tokenizer_type"], "tiktoken")
        self.assertEqual(special["bos_token_id"], 28)
        self.assertEqual(special["eos_token_id"], 29)
        self.assertEqual(special["pad_token_id"], 30)
        self.assertEqual(staged, (True, True, True, True))

    def test_kimi_selected_providers_are_registered(self) -> None:
        registry = build_ir_v8.load_kernel_registry()
        registered = {
            str(row.get("id") or row.get("name"))
            for row in registry.get("kernels", [])
        }
        required = {
            "gemm_nt_fp32_exact",
            "rmsnorm_forward_strided_f32",
            "group_limited_topk_router_sigmoid_f32",
            "deepseek_mla_attention_f32",
            "moe_swiglu_expert_forward_bf16",
            "moe_swiglu_shared_forward_bf16",
        }
        self.assertEqual(required - registered, set())

    def test_kimi_mla_moe_template_lowers_to_reference_kernels(self) -> None:
        manifest = _make_tiny_kimi_manifest()
        ops = build_ir_v8.build_ir1_direct(manifest, ROOT / "tests" / "kimi.synthetic.json", mode="decode")
        prefill_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "kimi.synthetic.json",
            mode="prefill",
        )
        by_layer_op = {(op.get("layer"), op.get("op"), op.get("instance", 0)): op for op in ops}
        prefill_by_layer_op = {
            (op.get("layer"), op.get("op"), op.get("instance", 0)): op
            for op in prefill_ops
        }
        registry = build_ir_v8.load_kernel_registry()
        lowered_decode_1 = build_ir_v8.generate_ir_lower_1(ops, registry, manifest, "decode")
        decode_layout = build_ir_v8.generate_memory_layout(
            lowered_decode_1, manifest, registry, mode="decode", context_len=16
        )
        lowered_decode_2 = build_ir_v8.generate_ir_lower_2(
            lowered_decode_1, decode_layout, manifest, registry, mode="decode"
        )
        lowered_prefill_1 = build_ir_v8.generate_ir_lower_1(
            prefill_ops, registry, manifest, "prefill"
        )
        prefill_layout = build_ir_v8.generate_memory_layout(
            lowered_prefill_1, manifest, registry, mode="prefill", context_len=16
        )
        lowered_prefill_2 = build_ir_v8.generate_ir_lower_2(
            lowered_prefill_1, prefill_layout, manifest, registry, mode="prefill"
        )
        call_decode = build_ir_v8.generate_ir_lower_3(lowered_decode_2, "decode")
        call_prefill = build_ir_v8.generate_ir_lower_3(lowered_prefill_2, "prefill")
        lowered_decode_by_layer_op = {
            (op.get("layer"), op.get("op")): op
            for op in lowered_decode_2["operations"]
        }
        lowered_prefill_by_layer_op = {
            (op.get("layer"), op.get("op")): op
            for op in lowered_prefill_2["operations"]
        }

        self.assertEqual([op["op"] for op in ops].count("residual_save"), 4)
        self.assertEqual(by_layer_op[(0, "q_proj", 0)]["kernel"], "gemv_bf16")
        self.assertEqual(prefill_by_layer_op[(0, "q_proj", 0)]["kernel"], "gemm_nt_bf16")
        self.assertEqual(lowered_decode_by_layer_op[(0, "q_proj")]["params"]["_output_dim"], 8)
        self.assertEqual(lowered_decode_by_layer_op[(0, "out_proj")]["params"]["_input_dim"], 4)
        self.assertEqual(lowered_prefill_by_layer_op[(0, "out_proj")]["params"]["_input_dim"], 4)
        self.assertEqual(by_layer_op[(0, "kv_a_proj", 0)]["kernel"], "gemv_bf16")
        self.assertEqual(
            by_layer_op[(0, "kv_a_layernorm", 0)]["kernel"],
            "rmsnorm_forward_strided_f32",
        )
        self.assertEqual(by_layer_op[(0, "kv_lora_decompress", 0)]["kernel"], "deepseek_mla_kv_decompress_bf16")
        self.assertEqual(by_layer_op[(0, "partial_rope_concat", 0)]["kernel"], "deepseek_mla_partial_rope_concat_packed_f32")
        self.assertEqual(
            by_layer_op[(0, "mla_attention", 0)]["kernel"],
            "deepseek_mla_attention_decode_f32",
        )
        self.assertEqual(
            prefill_by_layer_op[(0, "mla_attention", 0)]["kernel"],
            "deepseek_mla_attention_f32",
        )
        self.assertEqual(by_layer_op[(1, "moe_swiglu_expert_mlp", 0)]["kernel"], "moe_swiglu_expert_forward_bf16")
        self.assertEqual(by_layer_op[(1, "shared_swiglu_expert_mlp", 0)]["kernel"], "moe_swiglu_shared_forward_bf16")
        self.assertEqual(by_layer_op[(1, "moe_router", 0)]["kernel"], "gemm_nt_fp32_exact")
        self.assertEqual(
            prefill_by_layer_op[(1, "moe_router", 0)]["kernel"],
            "gemm_nt_fp32_exact",
        )
        self.assertEqual(
            by_layer_op[(1, "group_limited_topk_router", 0)]["kernel"],
            "group_limited_topk_router_sigmoid_f32",
        )

        q_source = by_layer_op[(0, "q_proj", 0)]["dataflow"]["inputs"]["x"]
        kv_source = by_layer_op[(0, "kv_a_proj", 0)]["dataflow"]["inputs"]["x"]
        router_source = by_layer_op[(1, "moe_router", 0)]["dataflow"]["inputs"]["A"]
        self.assertEqual(q_source["slot"], "layer_input")
        self.assertEqual(q_source["from_op"], by_layer_op[(0, "block_rmsnorm", 0)]["op_id"])
        self.assertEqual(kv_source["slot"], "layer_input")
        self.assertEqual(router_source["slot"], "layer_input")
        self.assertEqual(router_source["from_op"], by_layer_op[(1, "block_rmsnorm", 1)]["op_id"])

        decode_norm = next(
            op for op in call_decode["operations"]
            if op.get("layer") == 0 and op.get("op") == "kv_a_layernorm"
        )
        prefill_norm = next(
            op for op in call_prefill["operations"]
            if op.get("layer") == 0 and op.get("op") == "kv_a_layernorm"
        )
        for norm in (decode_norm, prefill_norm):
            args = {arg["name"]: arg["expr"] for arg in norm["args"]}
            self.assertEqual(args["d_model"], "4")
            self.assertEqual(args["input_stride"], "6")
            self.assertEqual(args["output_stride"], "4")

        routed_weights = by_layer_op[(1, "moe_swiglu_expert_mlp", 0)]["weights"]
        self.assertIn("moe_expert_gate", routed_weights)
        self.assertIn("moe_expert_up", routed_weights)
        self.assertIn("moe_expert_down", routed_weights)


if __name__ == "__main__":
    unittest.main()
