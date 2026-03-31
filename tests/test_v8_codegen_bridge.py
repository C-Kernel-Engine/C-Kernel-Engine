#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V8_BUILD_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
V8_CODEGEN_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_v8.py"


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


build_ir_v8 = _load_module("build_ir_v8_codegen_bridge_tests", V8_BUILD_PATH)


def _entry(name: str, dtype: str, shape: list[int], offset: int) -> dict:
    nbytes_per = {"fp32": 4, "fp16": 2, "q8_0": 1, "q5_0": 1, "q6_k": 1, "q4_k": 1}.get(dtype, 4)
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


def _make_qwen3_decoder_manifest() -> dict:
    offset = 0
    entries = []

    def add(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        item = _entry(name, dtype, shape, offset)
        entries.append(item)
        offset += int(item["nbytes"])

    add("token_emb", "q8_0", [64, 16])
    add("layer.0.ln1_gamma", "fp32", [16])
    add("layer.0.ln2_gamma", "fp32", [16])
    add("layer.0.q_norm", "fp32", [4])
    add("layer.0.k_norm", "fp32", [4])
    add("layer.0.wq", "q8_0", [16, 16])
    add("layer.0.wk", "q8_0", [16, 16])
    add("layer.0.wv", "q8_0", [16, 16])
    add("layer.0.wo", "q8_0", [16, 16])
    add("layer.0.w1", "q8_0", [32, 16])
    add("layer.0.w2", "q8_0", [16, 32])
    add("layer.0.w3", "q8_0", [32, 16])
    add("final_ln_weight", "fp32", [16])

    return {
        "config": {
            "model": "qwen3",
            "arch": "qwen3",
            "num_layers": 1,
            "embed_dim": 16,
            "num_heads": 4,
            "num_kv_heads": 4,
            "head_dim": 4,
            "intermediate_size": 32,
            "context_length": 32,
            "max_seq_len": 32,
            "vocab_size": 64,
        },
        "quant_summary": {
            "token_emb": "q8_0",
            "layer.0": {
                "wq": "q8_0",
                "wk": "q8_0",
                "wv": "q8_0",
                "wo": "q8_0",
                "w1": "q8_0",
                "w2": "q8_0",
                "w3": "q8_0",
            },
            "final_ln_weight": "fp32",
        },
        "entries": entries,
        "template": build_ir_v8._load_builtin_template_doc("qwen3"),
    }


def _make_qwen2_decoder_manifest() -> dict:
    offset = 0
    entries = []

    def add(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        item = _entry(name, dtype, shape, offset)
        entries.append(item)
        offset += int(item["nbytes"])

    add("token_emb", "q8_0", [64, 16])
    add("layer.0.ln1_gamma", "fp32", [16])
    add("layer.0.ln2_gamma", "fp32", [16])
    add("layer.0.wq", "q5_0", [16, 16])
    add("layer.0.bq", "fp32", [16])
    add("layer.0.wk", "q5_0", [16, 16])
    add("layer.0.bk", "fp32", [16])
    add("layer.0.wv", "q8_0", [16, 16])
    add("layer.0.bv", "fp32", [16])
    add("layer.0.wo", "q5_0", [16, 16])
    add("layer.0.w1", "q5_0", [32, 16])
    add("layer.0.w2", "q6_k", [16, 16])
    add("final_ln_weight", "fp32", [16])

    return {
        "config": {
            "model": "qwen2",
            "arch": "qwen2",
            "num_layers": 1,
            "embed_dim": 16,
            "num_heads": 4,
            "num_kv_heads": 4,
            "head_dim": 4,
            "intermediate_size": 16,
            "context_length": 32,
            "max_seq_len": 32,
            "vocab_size": 64,
        },
        "quant_summary": {
            "token_emb": "q8_0",
            "layer.0": {
                "wq": "q5_0",
                "wk": "q5_0",
                "wv": "q8_0",
                "wo": "q5_0",
                "w1": "q5_0",
                "w2": "q6_k",
            },
            "final_ln_weight": "fp32",
        },
        "entries": entries,
        "template": build_ir_v8._load_builtin_template_doc("qwen2"),
    }


class V8CodegenBridgeTests(unittest.TestCase):
    def test_qwen2_decode_uses_fp32_mlp_kernels(self) -> None:
        manifest = _make_qwen2_decoder_manifest()

        ir1_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "qwen2_manifest.synthetic.json",
            mode="decode",
        )

        by_op = {}
        for ir_op in ir1_ops:
            by_op.setdefault(ir_op["op"], []).append(ir_op)

        ops = [op["op"] for op in ir1_ops]
        self.assertIn("quantize_input_0", ops)
        self.assertNotIn("quantize_input_1", ops)
        self.assertNotIn("quantize_mlp_down_input", ops)
        self.assertEqual(by_op["q_proj"][0]["kernel"], "gemv_q5_0_q8_0")
        self.assertEqual(by_op["k_proj"][0]["kernel"], "gemv_q5_0_q8_0")
        self.assertEqual(by_op["out_proj"][0]["kernel"], "gemv_q5_0_q8_0")
        self.assertEqual(by_op["mlp_gate_up"][0]["kernel"], "gemv_q5_0")
        self.assertEqual(by_op["mlp_down"][0]["kernel"], "gemv_q6_k")

    def test_decoder_codegen_with_prefill_emits_multimodal_bridge_api(self) -> None:
        manifest = _make_qwen3_decoder_manifest()

        with tempfile.TemporaryDirectory(prefix="v8_codegen_bridge_") as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "weights_manifest.json"
            prefill_ir1 = tmp / "ir1_prefill.json"
            prefill_layout = tmp / "layout_prefill.json"
            prefill_lowered = tmp / "lowered_prefill.json"
            prefill_call = tmp / "call_prefill.json"
            decode_ir1 = tmp / "ir1_decode.json"
            decode_layout = tmp / "layout_decode.json"
            decode_lowered = tmp / "lowered_decode.json"
            decode_call = tmp / "call_decode.json"
            c_path = tmp / "decoder_v8_bridge.c"

            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            for mode, ir1_path, layout_path, lowered_path, call_path in (
                ("prefill", prefill_ir1, prefill_layout, prefill_lowered, prefill_call),
                ("decode", decode_ir1, decode_layout, decode_lowered, decode_call),
            ):
                rc = build_ir_v8.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--mode",
                        mode,
                        "--output",
                        str(ir1_path),
                        "--layout-output",
                        str(layout_path),
                        "--lowered-output",
                        str(lowered_path),
                        "--call-output",
                        str(call_path),
                    ]
                )
                self.assertEqual(rc, 0, msg=f"build_ir_v8 failed for mode={mode}")

            result = subprocess.run(
                [
                    sys.executable,
                    str(V8_CODEGEN_PATH),
                    "--ir",
                    str(decode_call),
                    "--prefill",
                    str(prefill_call),
                    "--layout",
                    str(decode_layout),
                    "--output",
                    str(c_path),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(c_path.exists())

            text = c_path.read_text(encoding="utf-8")
            self.assertIn("#define CK_HAS_PREFILL 1", text)
            self.assertIn("static void ck_prefill_from_embedded", text)
            self.assertIn("static void ck_decode_embedded", text)
            self.assertIn("static int ck_bridge_forward_staged", text)
            self.assertIn("CK_EXPORT int ck_model_write_embeddings", text)
            self.assertIn("CK_EXPORT int ck_model_embed_tokens_at", text)
            self.assertIn("CK_EXPORT int ck_model_forward_from_embeddings", text)
            self.assertIn("CK_EXPORT int ck_model_forward_mixed", text)
            self.assertIn("CK_EXPORT intptr_t ck_model_get_named_activation_runtime_offset", text)
            self.assertIn("CK_EXPORT intptr_t ck_model_get_named_activation_nbytes", text)
            self.assertIn("CK_EXPORT uintptr_t ck_model_get_named_activation_ptr", text)
            self.assertIn("rope_precompute_cache(", text)
            self.assertNotIn("/* No pre-weights init ops */", text)
            self.assertIn("logits (last-only)", text)
            self.assertNotIn("copy_last_logits (prefill fixup)", text)

    def test_decoder_parity_dump_emits_decode_attention_kqv_dump(self) -> None:
        manifest = _make_qwen3_decoder_manifest()

        with tempfile.TemporaryDirectory(prefix="v8_codegen_bridge_dump_") as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "weights_manifest.json"
            prefill_ir1 = tmp / "ir1_prefill.json"
            prefill_layout = tmp / "layout_prefill.json"
            prefill_lowered = tmp / "lowered_prefill.json"
            prefill_call = tmp / "call_prefill.json"
            decode_ir1 = tmp / "ir1_decode.json"
            decode_layout = tmp / "layout_decode.json"
            decode_lowered = tmp / "lowered_decode.json"
            decode_call = tmp / "call_decode.json"
            c_path = tmp / "decoder_v8_bridge_dump.c"

            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            for mode, ir1_path, layout_path, lowered_path, call_path in (
                ("prefill", prefill_ir1, prefill_layout, prefill_lowered, prefill_call),
                ("decode", decode_ir1, decode_layout, decode_lowered, decode_call),
            ):
                rc = build_ir_v8.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--mode",
                        mode,
                        "--output",
                        str(ir1_path),
                        "--layout-output",
                        str(layout_path),
                        "--lowered-output",
                        str(lowered_path),
                        "--call-output",
                        str(call_path),
                    ]
                )
                self.assertEqual(rc, 0, msg=f"build_ir_v8 failed for mode={mode}")

            result = subprocess.run(
                [
                    sys.executable,
                    str(V8_CODEGEN_PATH),
                    "--ir",
                    str(decode_call),
                    "--prefill",
                    str(prefill_call),
                    "--layout",
                    str(decode_layout),
                    "--output",
                    str(c_path),
                    "--parity-dump",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            text = c_path.read_text(encoding="utf-8")
            self.assertIn('ck_dump_tensor((float*)(model->bump + A_ATTN_SCRATCH), 0, "kqv_out", NUM_HEADS * HEAD_DIM);', text)


if __name__ == "__main__":
    unittest.main()
