#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V8_BUILD_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"


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


build_ir_v8 = _load_module("build_ir_v8_siglip_tests", V8_BUILD_PATH)


def _make_siglip_manifest() -> dict:
    return {
        "config": {
            "model": "siglip_vit",
            "arch": "siglip_vit",
            "num_layers": 1,
            "embed_dim": 256,
            "num_heads": 8,
            "num_kv_heads": 8,
            "head_dim": 32,
            "intermediate_size": 512,
            "context_length": 196,
            "max_seq_len": 196,
            "vocab_size": 32000,
        },
        "quant_summary": {
            "layer.0": {
                "wq": "q8_0",
                "wk": "q8_0",
                "wv": "q8_0",
                "wo": "q8_0",
                "w1": "q8_0",
                "w2": "q8_0",
                "w3": "q8_0",
            }
        },
        "entries": [
            {"name": "patch_embeddings.weight", "dtype": "fp32", "offset": 0, "shape": [256, 768], "nbytes": 786432},
            {"name": "patch_embeddings.bias", "dtype": "fp32", "offset": 786432, "shape": [256], "nbytes": 1024},
            {"name": "layer.0.ln1_gamma", "dtype": "fp32", "offset": 787456, "shape": [256], "nbytes": 1024},
            {"name": "layer.0.ln2_gamma", "dtype": "fp32", "offset": 788480, "shape": [256], "nbytes": 1024},
            {"name": "layer.0.wq", "dtype": "q8_0", "offset": 789504, "shape": [256, 256], "nbytes": 65536},
            {"name": "layer.0.wk", "dtype": "q8_0", "offset": 855040, "shape": [256, 256], "nbytes": 65536},
            {"name": "layer.0.wv", "dtype": "q8_0", "offset": 920576, "shape": [256, 256], "nbytes": 65536},
            {"name": "layer.0.wo", "dtype": "q8_0", "offset": 986112, "shape": [256, 256], "nbytes": 65536},
            {"name": "layer.0.w1", "dtype": "q8_0", "offset": 1051648, "shape": [512, 256], "nbytes": 131072},
            {"name": "layer.0.w2", "dtype": "q8_0", "offset": 1182720, "shape": [256, 512], "nbytes": 131072},
            {"name": "layer.0.w3", "dtype": "q8_0", "offset": 1313792, "shape": [512, 256], "nbytes": 131072},
        ],
        "template": build_ir_v8._load_builtin_template_doc("siglip_vit"),
    }


class V8SiglipTemplateTests(unittest.TestCase):
    def test_builtin_template_declares_encoder_contract(self) -> None:
        doc = build_ir_v8._load_builtin_template_doc("siglip_vit")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["sequence"], ["vision_encoder"])
        self.assertEqual(doc["block_types"]["vision_encoder"]["header"], ["patch_embeddings"])
        vision_contract = doc["contract"]["vision_contract"]
        self.assertEqual(vision_contract["input_modality"], "image")
        self.assertEqual(vision_contract["patch_size"], 16)
        attn_contract = doc["contract"]["attention_contract"]
        self.assertFalse(attn_contract["causal"])
        self.assertEqual(attn_contract["attn_variant"], "dense_bidirectional")

    def test_siglip_template_is_lowerable_in_v8(self) -> None:
        manifest = {
            "config": {"model": "siglip_vit", "arch": "siglip_vit"},
            "template": build_ir_v8._load_builtin_template_doc("siglip_vit"),
        }
        self.assertIsNone(build_ir_v8.unsupported_template_lowering_reason(manifest))

    def test_siglip_prefill_lowering_emits_patch_frontend_and_encoder_ops(self) -> None:
        manifest = _make_siglip_manifest()
        ir1_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "siglip_manifest.synthetic.json",
            mode="prefill",
        )

        ops = [op["op"] for op in ir1_ops]
        self.assertNotIn("patch_embeddings", ops)
        self.assertIn("patchify", ops)
        self.assertIn("patch_proj", ops)
        self.assertIn("attn_norm", ops)
        self.assertTrue(
            "qkv_proj" in ops or {"q_proj", "k_proj", "v_proj"}.issubset(set(ops)),
            msg="vision encoder must lower QKV projection somehow",
        )
        self.assertIn("attn", ops)
        self.assertIn("out_proj", ops)
        self.assertIn("ffn_norm", ops)
        self.assertIn("mlp_gate_up", ops)
        self.assertIn("geglu", ops)
        self.assertIn("mlp_down", ops)
        self.assertEqual(ops.count("residual_add"), 2)
        attn_kernels = [op["kernel"] for op in ir1_ops if op["op"] == "attn"]
        self.assertEqual(attn_kernels, ["attention_forward_full_head_major_gqa_flash_strided"])

    def test_siglip_prefill_binds_expected_layer_weights(self) -> None:
        manifest = _make_siglip_manifest()
        ir1_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "siglip_manifest.synthetic.json",
            mode="prefill",
        )

        by_op = {}
        for ir_op in ir1_ops:
            by_op.setdefault(ir_op["op"], []).append(ir_op)

        self.assertEqual(by_op["patch_proj"][0]["weights"]["patch_emb"]["name"], "patch_embeddings.weight")
        self.assertEqual(by_op["patch_proj"][0]["weights"]["patch_bias"]["name"], "patch_embeddings.bias")
        self.assertEqual(by_op["attn_norm"][0]["weights"]["ln1_gamma"]["name"], "layer.0.ln1_gamma")
        self.assertEqual(by_op["ffn_norm"][0]["weights"]["ln2_gamma"]["name"], "layer.0.ln2_gamma")
        self.assertEqual(by_op["out_proj"][0]["weights"]["wo"]["name"], "layer.0.wo")
        self.assertEqual(by_op["mlp_down"][0]["weights"]["w2"]["name"], "layer.0.w2")
        self.assertEqual(by_op["mlp_gate_up"][0]["weights"]["w1"]["name"], "layer.0.w1")
        self.assertEqual(by_op["mlp_gate_up"][0]["weights"]["w3"]["name"], "layer.0.w3")

    def test_siglip_prefill_disables_decoder_only_runtime_state(self) -> None:
        manifest = _make_siglip_manifest()
        ir1_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "siglip_manifest.synthetic.json",
            mode="prefill",
        )

        config = manifest["config"]
        self.assertFalse(config["_template_uses_kv_cache"])
        self.assertFalse(config["_template_uses_rope"])
        self.assertFalse(config["_template_has_logits"])

        act_specs = build_ir_v8.build_activation_specs(config, mode="prefill", context_len=196)
        self.assertIn("image_input", act_specs)
        self.assertIn("patch_scratch", act_specs)
        self.assertNotIn("kv_cache", act_specs)
        self.assertNotIn("rope_cache", act_specs)
        self.assertNotIn("logits", act_specs)

        lowered_ir1 = build_ir_v8.generate_ir_lower_1(
            ir1_ops,
            build_ir_v8.load_kernel_registry(),
            manifest,
            "prefill",
        )
        lowered_ops = [op["op"] for op in lowered_ir1]
        self.assertNotIn("kv_cache_batch_copy", lowered_ops)
        self.assertNotIn("kv_cache_store", lowered_ops)
        self.assertNotIn("copy_last_logits", lowered_ops)

        registry = build_ir_v8.load_kernel_registry()
        layout = build_ir_v8.generate_memory_layout(lowered_ir1, manifest, registry, mode="prefill", context_len=196)
        lowered_ir2 = build_ir_v8.generate_ir_lower_2(lowered_ir1, layout, manifest, registry, mode="prefill")
        by_lowered_op = {op["op"]: op for op in lowered_ir2["operations"]}
        self.assertEqual(by_lowered_op["patchify"]["activations"]["image"]["buffer"], "image_input")
        self.assertEqual(by_lowered_op["patchify"]["outputs"]["patches"]["buffer"], "patch_scratch")
        self.assertEqual(by_lowered_op["patch_proj"]["activations"]["A"]["buffer"], "patch_scratch")
        self.assertEqual(by_lowered_op["patch_proj"]["outputs"]["C"]["buffer"], "embedded_input")


if __name__ == "__main__":
    unittest.main()
