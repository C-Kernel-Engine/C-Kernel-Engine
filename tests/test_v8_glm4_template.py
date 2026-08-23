#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
V8_BUILD_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
V8_CODEGEN_CORE_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_core_v8.py"


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


build_ir_v8 = _load_module("build_ir_v8_glm4_tests", V8_BUILD_PATH)
codegen_core_v8 = _load_module("codegen_core_v8_glm4_tests", V8_CODEGEN_CORE_PATH)


def _entry(name: str, dtype: str, shape: list[int], offset: int) -> dict:
    nbytes_per = {"fp32": 4, "bf16": 2, "fp16": 2, "q8_0": 1, "q5_0": 1, "q6_k": 1, "q4_k": 1}.get(dtype, 4)
    size = 1
    for dim in shape:
        size *= int(dim)
    return {
        "name": name,
        "dtype": dtype,
        "offset": offset,
        "shape": shape,
        "nbytes": size * nbytes_per,
    }


class GLM4KVXrayExportTests(unittest.TestCase):
    def test_fp32_cache_export_uses_resolved_storage_contract(self) -> None:
        source = codegen_core_v8.emit_model_and_api(
            layout={"memory": {"activations": {"buffers": []}}},
            config={"decode_kv_cache_dtype": "fp32"},
        )
        self.assertIn("CK_EXPORT int ck_model_debug_export_kv(", source)
        self.assertIn("const float *layer_k = g_model->kv_cache +", source)
        self.assertIn("UINT32_C(0)", source)
        self.assertIn("sizeof(float)", source)

    def test_fp16_compatibility_export_rejects_non_fp16_cache(self) -> None:
        source = codegen_core_v8.emit_model_and_api(
            layout={"memory": {"activations": {"buffers": []}}},
            config={"decode_kv_cache_dtype": "bf16"},
        )
        self.assertIn("const uint16_t *layer_k = g_model->kv_cache_f16 +", source)
        self.assertIn("UINT32_C(2)", source)
        self.assertIn("if (UINT32_C(2) != UINT32_C(1)) return -7;", source)


class GLM4KVStorageContractTests(unittest.TestCase):
    def test_circuit_declares_llama_fp16_decode_cache(self) -> None:
        circuit = json.loads(
            (ROOT / "version" / "v8" / "circuits" / "glm4.json").read_text(
                encoding="utf-8"
            )
        )
        attention = circuit["contract"]["attention_contract"]
        self.assertEqual(attention["decode_kv_cache_dtype"], "fp16")

    def test_decode_ir_selects_fp16_store_and_attention(self) -> None:
        manifest = _make_tiny_glm4_quant_manifest()
        ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "glm4_quant_manifest.synthetic.json",
            mode="decode",
        )
        attention = next(op for op in ops if op["op"] == "attn")
        self.assertEqual(
            attention["kernel"],
            "attention_forward_decode_head_major_gqa_flash_f16cache_contract",
        )
        self.assertEqual(
            attention["resolved_contract"]["resolved_contract_id"],
            "f16_online_fp32_merge",
        )
        self.assertEqual(
            attention["resolved_contract"]["selector"],
            "CK_ATTN_REDUCTION_F16_ONLINE_FP32_MERGE",
        )


def _make_tiny_glm4_bf16_manifest() -> dict:
    offset = 0
    entries = []

    def add(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        item = _entry(name, dtype, shape, offset)
        entries.append(item)
        offset += int(item["nbytes"])

    add("token_emb", "bf16", [64, 16])
    add("layer.0.ln1_gamma", "fp32", [16])
    add("layer.0.ln2_gamma", "fp32", [16])
    add("layer.0.post_attention_norm", "fp32", [16])
    add("layer.0.post_ffn_norm", "fp32", [16])
    add("layer.0.wq", "bf16", [16, 16])
    add("layer.0.bq", "fp32", [16])
    add("layer.0.wk", "bf16", [8, 16])
    add("layer.0.bk", "fp32", [8])
    add("layer.0.wv", "bf16", [8, 16])
    add("layer.0.bv", "fp32", [8])
    add("layer.0.wo", "bf16", [16, 16])
    add("layer.0.w1", "bf16", [32, 16])
    add("layer.0.w2", "bf16", [16, 16])
    add("final_ln_weight", "fp32", [16])

    return {
        "config": {
            "model": "glm4",
            "arch": "glm4",
            "num_layers": 1,
            "embed_dim": 16,
            "num_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 4,
            "intermediate_size": 16,
            "context_length": 32,
            "max_seq_len": 32,
            "vocab_size": 64,
            "rope_dim": 4,
            "rope_partial_rotary_dim": 2,
        },
        "quant_summary": {
            "token_emb": "bf16",
            "layer.0": {
                "wq": "bf16",
                "wk": "bf16",
                "wv": "bf16",
                "wo": "bf16",
                "w1": "bf16",
                "w2": "bf16",
            },
            "final_ln_weight": "fp32",
        },
        "entries": entries,
        "template": build_ir_v8._load_builtin_template_doc("glm4"),
    }


def _make_tiny_glm4_quant_manifest() -> dict:
    manifest = _make_tiny_glm4_bf16_manifest()
    for item in manifest["entries"]:
        name = item["name"]
        if name == "token_emb":
            item["dtype"] = "q4_k"
        elif name.endswith(".wq") or name.endswith(".wk") or name.endswith(".wo") or name.endswith(".w1"):
            item["dtype"] = "q4_k"
        elif name.endswith(".wv"):
            item["dtype"] = "q6_k"
        elif name.endswith(".w2"):
            item["dtype"] = "q8_0"
    manifest["quant_summary"] = {
        "token_emb": "q4_k",
        "layer.0": {
            "wq": "q4_k",
            "wk": "q4_k",
            "wv": "q6_k",
            "wo": "q4_k",
            "w1": "q4_k",
            "w2": "q8_0",
        },
        "final_ln_weight": "fp32",
    }
    return manifest


class V8GLM4TemplateTests(unittest.TestCase):
    def test_builtin_template_declares_glm4_contract(self) -> None:
        doc = build_ir_v8._load_builtin_template_doc("glm4")
        self.assertEqual(doc["name"], "glm4")
        self.assertEqual(doc["flags"]["tokenizer"], "bpe")
        self.assertEqual(doc["contract"]["attention_contract"]["rope_layout"], "pairwise")
        self.assertTrue(doc["contract"]["attention_contract"]["partial_rotary"])
        self.assertTrue(doc["contract"]["block_contract"]["qkv_bias"])
        self.assertEqual(doc["contract"]["chat_contract"]["force_bos_text_if_tokenizer_add_bos_false"], "[gMASK]<sop>\n")
        self.assertEqual(
            doc["contract"]["chat_contract"]["assistant_generation_prefix"],
            "<|assistant|>",
        )
        self.assertEqual(
            doc["kernels"]["attn_decode"],
            "attention_forward_decode_head_major_gqa_flash",
        )
        self.assertEqual(
            doc["kernels"]["rmsnorm"],
            "rmsnorm_forward_llama_production",
        )
        self.assertNotIn("post_attention_norm", doc["kernels"])
        self.assertEqual(
            doc["kernels"]["post_ffn_norm"],
            "rmsnorm_forward_llama_production",
        )
        self.assertEqual(
            doc["kernels"]["rope_init"],
            "rope_precompute_cache_llama_cpu",
        )
        self.assertEqual(
            doc["kernels"]["rope_qk"],
            "rope_forward_qk_pairwise",
        )
        self.assertEqual(
            doc["kernels"]["rope_qk_decode"],
            "rope_forward_qk_pairwise_llama_cpu",
        )
        self.assertEqual(
            build_ir_v8._resolve_rope_qk_kernel(
                {"rope_layout": "pairwise"},
                doc["kernels"],
                "decode",
            ),
            "rope_forward_qk_pairwise_llama_cpu",
        )
        self.assertEqual(
            build_ir_v8._resolve_rope_qk_kernel(
                {"rope_layout": "pairwise"},
                doc["kernels"],
                "prefill",
            ),
            "rope_forward_qk_pairwise",
        )
        body_items = build_ir_v8._normalize_template_op_items(doc["block_types"]["decoder"]["body"]["ops"])
        self.assertEqual(
            [item["op"] for item in body_items],
            [
                "rmsnorm",
                "qkv_proj",
                "rope_qk",
                "attn",
                "out_proj",
                "post_attention_norm",
                "residual_add",
                "rmsnorm",
                "mlp_gate_up",
                "silu_mul",
                "mlp_down",
                "post_ffn_norm",
                "residual_add",
            ],
        )
        by_op = {item["op"]: item for item in body_items if isinstance(item.get("graph_slots"), dict)}
        self.assertEqual(by_op["qkv_proj"]["graph_slots"]["inputs"]["x"], "main_stream")
        self.assertEqual(by_op["out_proj"]["graph_slots"]["inputs"]["x"], "attn_scratch")
        self.assertEqual(by_op["mlp_gate_up"]["graph_slots"]["inputs"]["x"], "main_stream")
        self.assertEqual(by_op["mlp_down"]["graph_slots"]["inputs"]["A"], "mlp_scratch")
        self.assertEqual(by_op["mlp_down"]["graph_slots"]["outputs"]["C"], "main_stream")
        footer_items = build_ir_v8._normalize_template_op_items(doc["block_types"]["decoder"]["footer"])
        footer_by_op = {item["op"]: item for item in footer_items if isinstance(item.get("graph_slots"), dict)}
        self.assertEqual(footer_by_op["logits"]["graph_slots"]["inputs"]["x"], "main_stream")

    def test_attention_provider_is_phase_specific(self) -> None:
        manifest = _make_tiny_glm4_quant_manifest()
        decode_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "glm4_quant_manifest.synthetic.json",
            mode="decode",
        )
        prefill_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "glm4_quant_manifest.synthetic.json",
            mode="prefill",
        )

        decode_attention = next(op for op in decode_ops if op["op"] == "attn")
        prefill_attention = next(op for op in prefill_ops if op["op"] == "attn")
        self.assertEqual(
            decode_attention["kernel"],
            "attention_forward_decode_head_major_gqa_flash_f16cache_contract",
        )
        self.assertEqual(
            prefill_attention["kernel"],
            "attention_forward_causal_head_major_gqa_prefill_append_f16cache_single_range",
        )
        self.assertEqual(
            decode_attention["resolved_contract"]["resolved_contract_id"],
            "f16_online_fp32_merge",
        )
        self.assertEqual(
            prefill_attention["resolved_contract"]["resolved_contract_id"],
            "f16_online_single_range",
        )
        self.assertTrue(
            all(
                op["kernel"] == "rmsnorm_forward_llama_production"
                for op in decode_ops
                if op["op"] == "rmsnorm"
            )
        )
        self.assertTrue(
            all(
                op["kernel"] == "rmsnorm_forward_llama_production"
                for op in prefill_ops
                if op["op"] == "rmsnorm"
            )
        )
        for ops in (decode_ops, prefill_ops):
            self.assertTrue(
                all(
                    op["kernel"] == "rmsnorm_forward"
                    for op in ops
                    if op["op"] == "post_attention_norm"
                )
            )
            self.assertTrue(
                all(
                    op["kernel"] == "rmsnorm_forward_llama_production"
                    for op in ops
                    if op["op"] == "post_ffn_norm"
                )
            )

    def test_decode_call_ir_preserves_projection_bias_semantics(self) -> None:
        manifest = _make_tiny_glm4_quant_manifest()
        ir1 = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "glm4_quant_manifest.synthetic.json",
            mode="decode",
        )
        registry = build_ir_v8.load_kernel_registry()
        ir1 = build_ir_v8.insert_bias_add_ops(
            ir1,
            registry,
            manifest,
            mode="decode",
        )
        lowered = build_ir_v8.generate_ir_lower_1(
            ir1, registry, manifest, mode="decode"
        )
        layout = build_ir_v8.generate_memory_layout(
            lowered,
            manifest,
            registry,
            mode="decode",
            context_len=16,
        )
        lowered = build_ir_v8.generate_ir_lower_2(
            lowered, layout, manifest, registry, mode="decode"
        )
        call_ir = build_ir_v8.generate_ir_lower_3(lowered, mode="decode")
        self.assertEqual(
            [
                op.get("bias_for")
                for op in call_ir["operations"]
                if op["op"] == "bias_add"
            ][:3],
            ["q_proj", "k_proj", "v_proj"],
        )

    def test_rope_init_provider_survives_call_ir_lowering(self) -> None:
        manifest = _make_tiny_glm4_quant_manifest()
        config = build_ir_v8._normalize_manifest_config(manifest["config"])
        init_ir = build_ir_v8.generate_init_ir(manifest, config)
        layout = {
            "memory": {
                "activations": {
                    "buffers": [
                        {"name": "rope_cache", "define": "A_ROPE_CACHE"},
                    ]
                }
            }
        }

        standard_call_ir = build_ir_v8.generate_init_ir_lower_3(init_ir, layout)
        standard_rope = next(
            op for op in standard_call_ir["operations"] if op["op"] == "rope_init"
        )
        self.assertEqual(
            standard_rope["function"],
            "rope_precompute_cache_llama_cpu",
        )
        self.assertEqual(standard_rope["errors"], [])

        rope_init = next(op for op in init_ir["ops"] if op["op"] == "rope_init")
        rope_init["kernel"] = "test_map_selected_rope_provider"
        standard_abi = {
            "call_abi": {
                "params": [
                    {"source": "output:cos_cache"},
                    {"source": "output:sin_cache"},
                    {"source": "dim:max_seq_len"},
                    {"source": "dim:head_dim"},
                    {"source": "config:rope_theta"},
                    {"source": "dim:rotary_dim"},
                    {"source": "config:rope_scaling_type"},
                    {"source": "config:rope_scaling_factor"},
                ]
            }
        }
        with mock.patch.object(
            build_ir_v8,
            "load_kernel_call_abis",
            return_value={"test_map_selected_rope_provider": standard_abi},
        ):
            call_ir = build_ir_v8.generate_init_ir_lower_3(init_ir, layout)
        rope = next(op for op in call_ir["operations"] if op["op"] == "rope_init")
        self.assertEqual(rope["function"], "test_map_selected_rope_provider")
        self.assertEqual(rope["errors"], [])
        self.assertEqual(
            [arg["source"] for arg in rope["args"]],
            [
                "output:rope_cos",
                "output:rope_sin",
                "dim:max_seq_len",
                "dim:head_dim",
                "config:rope_theta",
                "dim:rotary_dim",
                "config:rope_scaling_type",
                "config:rope_scaling_factor",
            ],
        )

    def test_bf16_dense_projections_read_normed_main_stream(self) -> None:
        manifest = _make_tiny_glm4_bf16_manifest()
        ir1_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "glm4_bf16_manifest.synthetic.json",
            mode="decode",
        )

        by_op = {}
        for ir_op in ir1_ops:
            by_op.setdefault(ir_op["op"], []).append(ir_op)

        for op_name in ("q_proj", "k_proj", "v_proj"):
            source = by_op[op_name][0]["dataflow"]["inputs"]["x"]
            self.assertEqual(source["slot"], "main_stream")
            self.assertEqual(source["from_op"], by_op["rmsnorm"][0]["op_id"])

        mlp_source = by_op["mlp_gate_up"][0]["dataflow"]["inputs"]["x"]
        self.assertEqual(mlp_source["slot"], "main_stream")
        self.assertEqual(mlp_source["from_op"], by_op["rmsnorm"][1]["op_id"])


    def test_quantized_projections_read_q8_main_stream_view(self) -> None:
        manifest = _make_tiny_glm4_quant_manifest()
        ir1_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "glm4_quant_manifest.synthetic.json",
            mode="decode",
        )

        by_op = {}
        for ir_op in ir1_ops:
            by_op.setdefault(ir_op["op"], []).append(ir_op)

        for op_name in ("q_proj", "k_proj", "v_proj"):
            source = by_op[op_name][0]["dataflow"]["inputs"]["x"]
            self.assertEqual(source["slot"], "main_stream_q8")
            self.assertEqual(source["from_op"], by_op["quantize_input_0"][0]["op_id"])

        mlp_source = by_op["mlp_gate_up"][0]["dataflow"]["inputs"]["x"]
        self.assertEqual(mlp_source["slot"], "main_stream_q8")
        self.assertEqual(mlp_source["from_op"], by_op["quantize_input_2"][0]["op_id"])


if __name__ == "__main__":
    unittest.main()
