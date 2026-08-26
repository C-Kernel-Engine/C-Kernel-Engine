#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_ir_v8  # type: ignore
import convert_gguf_to_bump_v8 as converter  # type: ignore


def _entry(name: str, dtype: str, shape: list[int], offset: int) -> dict:
    item_count = 1
    for dim in shape:
        item_count *= int(dim)
    element_size = {"fp32": 4, "q4_k": 1, "q6_k": 1}.get(dtype, 1)
    size = item_count * element_size
    return {
        "name": name,
        "dtype": dtype,
        "offset": offset,
        "shape": shape,
        "nbytes": size,
        "size": size,
    }


def _make_tiny_cohere2_manifest(*, value_weight_dtype: str = "q6_k") -> dict:
    entries: list[dict] = []
    offset = 0

    def add(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        item = _entry(name, dtype, shape, offset)
        entries.append(item)
        offset += int(item["size"])

    embed_dim = 16
    intermediate = 32
    vocab_size = 64
    add("token_emb", "q4_k", [vocab_size, embed_dim])
    for layer in range(4):
        for name in ("ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta"):
            add(f"layer.{layer}.{name}", "fp32", [embed_dim])
        for name, dtype, shape in (
            ("wq", "q4_k", [embed_dim, embed_dim]),
            ("bq", "fp32", [embed_dim]),
            ("wk", "q4_k", [8, embed_dim]),
            ("bk", "fp32", [8]),
            ("wv", value_weight_dtype, [8, embed_dim]),
            ("bv", "fp32", [8]),
            ("wo", "q4_k", [embed_dim, embed_dim]),
            ("bo", "fp32", [embed_dim]),
            ("w1", "q4_k", [2 * intermediate, embed_dim]),
            ("b1", "fp32", [2 * intermediate]),
            ("w2", "q6_k", [embed_dim, intermediate]),
            ("b2", "fp32", [embed_dim]),
        ):
            add(f"layer.{layer}.{name}", dtype, shape)
    add("final_ln_weight", "fp32", [embed_dim])
    add("final_ln_bias", "fp32", [embed_dim])

    layer_quant = {
        "wq": "q4_k",
        "wk": "q4_k",
        "wv": value_weight_dtype,
        "wo": "q4_k",
        "w1": "q4_k",
        "w2": "q6_k",
    }
    return {
        "config": {
            "model": "cohere2",
            "arch": "cohere2",
            "num_layers": 4,
            "embed_dim": embed_dim,
            "num_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 4,
            "intermediate_size": intermediate,
            "context_length": 32,
            "max_seq_len": 32,
            "vocab_size": vocab_size,
            "rope_theta": 50000.0,
            "rotary_dim": 4,
            "rms_eps": 1e-5,
            "sliding_window": 8,
            "logit_scale": 0.25,
            "layer_kinds": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            "tie_word_embeddings": True,
        },
        "entries": entries,
        "template": build_ir_v8._load_builtin_template_doc("cohere2"),
        "quant_summary": {
            "token_emb": "q4_k",
            **{f"layer.{layer}": dict(layer_quant) for layer in range(4)},
        },
    }


class Cohere2ContractTests(unittest.TestCase):
    def test_model_map_owns_metadata_pattern_and_norm_alias(self) -> None:
        metadata = converter.gguf_ck_declared_metadata_keys()
        self.assertIn("cohere2.logit_scale", metadata)
        self.assertEqual(
            converter.gguf_ck_layer_kinds_from_map("cohere2", 8),
            ["sliding_attention"] * 3
            + ["full_attention"]
            + ["sliding_attention"] * 3
            + ["full_attention"],
        )
        self.assertEqual(
            converter.gguf_ck_source_tensor_alias("cohere2", "ffn_norm.weight"),
            "attn_norm.weight",
        )
        self.assertTrue(converter.gguf_ck_synthesizes_layernorm_beta("cohere2"))

    def test_circuit_declares_parallel_branches_and_exact_residual_order(self) -> None:
        circuit = build_ir_v8._load_builtin_template_doc("cohere2")
        self.assertEqual(circuit["name"], "cohere2")
        self.assertEqual(circuit["flags"]["rope"], "rope")
        self.assertEqual(
            circuit["activation_bindings"]["normalized_stream"],
            "normalized_input",
        )
        chat = circuit["contract"]["chat_contract"]
        self.assertEqual(
            chat["conversation_prefix"],
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|>",
        )
        self.assertTrue(chat["assistant_generation_prefix"].endswith("<|START_RESPONSE|>"))
        self.assertEqual(
            [rule["pattern"] for rule in circuit["contract"]["weight_policy"]["ignore"]],
            ["layer.*.ln2_gamma", "layer.*.ln2_beta"],
        )
        body = circuit["block_types"]["decoder"]["body"]["ops_by_kind"]
        sliding = body["sliding_attention"]
        full = body["full_attention"]
        self.assertIn("rope_qk", sliding)
        self.assertNotIn("rope_qk", full)
        for ops in (sliding, full):
            op_names = [item["op"] if isinstance(item, dict) else item for item in ops]
            self.assertLess(op_names.index("out_proj"), op_names.index("mlp_gate_up"))
            self.assertEqual(op_names[-2:], ["residual_add", "residual_add"])
            first_add = ops[-2]["graph_slots"]["inputs"]
            second_add = ops[-1]["graph_slots"]["inputs"]
            self.assertEqual(first_add, {"a": "main_stream", "b": "attention_residual"})
            self.assertEqual(second_add, {"a": "main_stream", "b": "attention_output"})

    def test_compiler_has_no_cohere_family_branch(self) -> None:
        for relative in (
            "version/v8/scripts/build_ir_v8.py",
            "version/v8/scripts/codegen_core_v8.py",
            "version/v8/scripts/codegen_prefill_v8.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8").lower()
            self.assertNotIn("cohere2", source, relative)

    def test_prefill_ir_uses_three_sliding_layers_then_one_full_layer(self) -> None:
        manifest = _make_tiny_cohere2_manifest()
        operations = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "cohere2_manifest.synthetic.json",
            mode="prefill",
        )
        layer_ops = {
            layer: [op["op"] for op in operations if op.get("layer") == layer]
            for layer in range(4)
        }
        for layer in range(3):
            self.assertIn("rope_qk", layer_ops[layer])
            self.assertIn("attn_sliding", layer_ops[layer])
            self.assertNotIn("attn", layer_ops[layer])
        self.assertNotIn("rope_qk", layer_ops[3])
        self.assertIn("attn", layer_ops[3])
        self.assertNotIn("attn_sliding", layer_ops[3])

        first_quantizer = next(
            op
            for op in operations
            if op.get("layer") == 0 and op["op"] == "quantize_input_0"
        )
        self.assertEqual(
            first_quantizer["graph_slots"]["inputs"]["input"],
            "normalized_stream",
        )
        footer = [
            (op["op"], op["kernel"])
            for op in operations
            if op.get("section") == "footer"
        ]
        self.assertEqual(footer[-1], ("final_logit_scale", "final_logit_scale_f32"))

    def test_prefill_and_decode_build_complete_call_ir(self) -> None:
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                manifest = _make_tiny_cohere2_manifest()
                registry = build_ir_v8.load_kernel_registry()
                ir1 = build_ir_v8.build_ir1_direct(
                    manifest,
                    ROOT / "tests" / "cohere2_manifest.synthetic.json",
                    mode=mode,
                )
                lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, manifest, mode)
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
                call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode)
                self.assertFalse(
                    [(op.get("op"), op.get("errors")) for op in call_ir["operations"] if op.get("errors")]
                )
                if mode == "prefill":
                    layer_zero_quantizers = [
                        op
                        for op in call_ir["operations"]
                        if op.get("layer") == 0 and op.get("op") == "quantize_input_0"
                    ]
                    # Attention-output quantization reuses the shared Q8
                    # workspace, so lowering must restore the normalized
                    # stream before the parallel MLP branch consumes it.
                    self.assertGreaterEqual(len(layer_zero_quantizers), 3)
                    for quantizer in layer_zero_quantizers:
                        input_arg = next(
                            arg for arg in quantizer["args"] if arg["name"] == "x"
                        )
                        self.assertEqual(input_arg["buffer_ref"], "normalized_input")

    def test_same_contract_workspace_clobber_restores_normalized_stream(self) -> None:
        manifest = _make_tiny_cohere2_manifest(value_weight_dtype="q4_k")
        operations = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "cohere2_manifest.synthetic.json",
            mode="prefill",
        )
        layer_zero = [op for op in operations if op.get("layer") == 0]
        names = [op["op"] for op in layer_zero]
        out_proj = names.index("out_proj")
        self.assertEqual(names[out_proj + 1 : out_proj + 3], [
            "quantize_input_0",
            "mlp_gate_up",
        ])
        restored = layer_zero[out_proj + 1]
        self.assertEqual(
            restored["graph_slots"]["inputs"]["input"],
            "normalized_stream",
        )

    def test_final_logit_scale_kernel_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "liblogit_scale.so"
            subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-shared",
                    "-fPIC",
                    "-I",
                    str(ROOT / "include"),
                    str(ROOT / "src" / "kernels" / "logit_kernels.c"),
                    "-o",
                    str(library),
                ],
                check=True,
            )
            native = ctypes.CDLL(str(library))
            native.final_logit_scale_f32.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]
            values = (ctypes.c_float * 6)(-8.0, -1.0, 0.0, 1.0, 4.0, 12.0)
            native.final_logit_scale_f32(values, 2, 3, ctypes.c_float(0.25))
            self.assertEqual(list(values), [-2.0, -0.25, 0.0, 0.25, 1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
