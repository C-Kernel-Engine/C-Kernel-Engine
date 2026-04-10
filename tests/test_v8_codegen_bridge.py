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
BUILD_DIR = ROOT / "build"
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
    add("layer.0.wq", "q4_k", [16, 16])
    add("layer.0.wk", "q4_k", [16, 16])
    add("layer.0.wv", "q6_k", [16, 16])
    add("layer.0.wo", "q4_k", [16, 16])
    add("layer.0.w1", "q4_k", [32, 16])
    add("layer.0.w2", "q6_k", [16, 32])
    add("layer.0.w3", "q4_k", [32, 16])
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
                "wq": "q4_k",
                "wk": "q4_k",
                "wv": "q6_k",
                "wo": "q4_k",
                "w1": "q4_k",
                "w2": "q6_k",
                "w3": "q4_k",
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


def _make_qwen3vl_decoder_manifest() -> dict:
    offset = 0
    entries = []

    def add(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        item = _entry(name, dtype, shape, offset)
        entries.append(item)
        offset += int(item["nbytes"])

    add("token_emb", "q8_0", [64, 16])
    add("output.weight", "q8_0", [64, 16])
    add("layer.0.ln1_gamma", "fp32", [16])
    add("layer.0.ln2_gamma", "fp32", [16])
    add("layer.0.q_norm", "fp32", [4])
    add("layer.0.k_norm", "fp32", [4])
    add("layer.0.wq", "q4_k", [16, 16])
    add("layer.0.wk", "q4_k", [16, 16])
    add("layer.0.wv", "q6_k", [16, 16])
    add("layer.0.wo", "q4_k", [16, 16])
    add("layer.0.w1", "q4_k", [32, 16])
    add("layer.0.w2", "q6_k", [16, 32])
    add("layer.0.w3", "q4_k", [32, 16])
    add("final_ln_weight", "fp32", [16])

    return {
        "config": {
            "model": "qwen3vl",
            "arch": "qwen3vl",
            "num_layers": 1,
            "embed_dim": 16,
            "num_heads": 4,
            "num_kv_heads": 4,
            "head_dim": 4,
            "intermediate_size": 32,
            "context_length": 32,
            "max_seq_len": 32,
            "vocab_size": 64,
            "tie_word_embeddings": False,
            "mrope_sections": [1, 1, 0, 0],
            "mrope_n_dims": 2,
            "num_deepstack_layers": 3,
            "rope_layout": "multi_section_1d",
        },
        "quant_summary": {
            "token_emb": "q8_0",
            "lm_head": "q8_0",
            "layer.0": {
                "wq": "q4_k",
                "wk": "q4_k",
                "wv": "q6_k",
                "wo": "q4_k",
                "w1": "q4_k",
                "w2": "q6_k",
                "w3": "q4_k",
            },
            "final_ln_weight": "fp32",
        },
        "entries": entries,
        "template": build_ir_v8._load_builtin_template_doc("qwen3vl"),
    }


class V8CodegenBridgeTests(unittest.TestCase):
    def test_builtin_qwen3vl_template_uses_text_mrope(self) -> None:
        doc = build_ir_v8._load_builtin_template_doc("qwen3vl")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["contract"]["attention_contract"]["rope_layout"], "multi_section_1d")
        self.assertEqual(doc["kernels"]["rope_qk"], "mrope_qk_text")

    def test_qwen2_decode_uses_contracted_q8_kernels(self) -> None:
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
        self.assertIn("quantize_input_1", ops)
        self.assertIn("quantize_mlp_down_input", ops)
        self.assertEqual(by_op["q_proj"][0]["kernel"], "gemv_q5_0_q8_0")
        self.assertEqual(by_op["k_proj"][0]["kernel"], "gemv_q5_0_q8_0")
        self.assertEqual(by_op["out_proj"][0]["kernel"], "gemv_q5_0_q8_0")
        self.assertEqual(by_op["mlp_gate_up"][0]["kernel"], "gemv_q5_0_q8_0")
        self.assertEqual(by_op["mlp_down"][0]["kernel"], "gemv_q6_k_q8_k")

    def test_qwen3vl_decode_uses_text_mrope_kernel(self) -> None:
        manifest = _make_qwen3vl_decoder_manifest()

        ir1_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "qwen3vl_manifest.synthetic.json",
            mode="decode",
        )

        by_op = {}
        for ir_op in ir1_ops:
            by_op.setdefault(ir_op["op"], []).append(ir_op)

        self.assertEqual(by_op["rope_qk"][0]["kernel"], "mrope_qk_text")
        self.assertEqual(by_op["attn"][0]["kernel"], "attention_forward_decode_head_major_gqa_flash_f16cache")

        with tempfile.TemporaryDirectory(prefix="v8_qwen3vl_text_mrope_") as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "weights_manifest.json"
            ir1_path = tmp / "ir1_decode.json"
            layout_path = tmp / "layout_decode.json"
            lowered_path = tmp / "lowered_decode.json"
            call_path = tmp / "call_decode.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            rc = build_ir_v8.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--mode",
                    "decode",
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
            self.assertEqual(rc, 0)

            call_doc = json.loads(call_path.read_text(encoding="utf-8"))
            layout_doc = json.loads(layout_path.read_text(encoding="utf-8"))
            rope_call = next(op for op in call_doc["operations"] if op["op"] == "rope_qk")
            attn_call = next(op for op in call_doc["operations"] if op["op"] == "attn")
            kv_store_call = next(op for op in call_doc["operations"] if op["op"] == "kv_cache_store")
            kv_buf = next(buf for buf in layout_doc["memory"]["activations"]["buffers"] if buf["name"] == "kv_cache")
            self.assertEqual(rope_call["function"], "mrope_qk_text")
            self.assertEqual(attn_call["function"], "attention_forward_decode_head_major_gqa_flash_f16cache")
            self.assertEqual(kv_store_call["function"], "kv_cache_store_f16")
            self.assertEqual(kv_buf["dtype"], "fp16")
            arg_map = {arg["name"]: arg["expr"] for arg in rope_call["args"]}
            self.assertEqual(arg_map["pos_offset"], "model->rope_pos")
            self.assertEqual(arg_map["section_0"], "1")
            self.assertEqual(arg_map["section_1"], "1")
            self.assertEqual(arg_map["section_2"], "0")
            self.assertEqual(arg_map["section_3"], "0")
            self.assertEqual(arg_map["n_dims"], "2")

    def test_qwen3vl_decoder_codegen_emits_deepstack_bridge_api(self) -> None:
        manifest = _make_qwen3vl_decoder_manifest()

        with tempfile.TemporaryDirectory(prefix="v8_codegen_qwen3vl_deepstack_bridge_") as tmpdir:
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
            c_path = tmp / "decoder_v8_qwen3vl_bridge.c"

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
                    "--prefill-layout",
                    str(prefill_layout),
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

            text = c_path.read_text(encoding="utf-8")
            self.assertIn("CK_EXPORT int ck_model_write_embeddings_ex", text)
            self.assertIn("CK_EXPORT int ck_model_forward_segments_grid_ex", text)
            self.assertIn("CK_EXPORT int ck_model_forward_mixed_ex", text)
            self.assertIn("CK_EXPORT int ck_model_forward_mixed_grid_ex", text)
            self.assertIn("if (prefix_grid_x > 0 && prefix_grid_y > 0 && prefix_grid_x * prefix_grid_y != prefix_tokens) return -10;", text)
            self.assertIn("ck_qwen3vl_prefill_bridge_prepare", text)
            self.assertIn("ck_qwen3vl_prefill_mrope_qk", text)
            self.assertIn("ck_qwen3vl_prefill_deepstack_add(CKModel *model, int layer, int num_tokens)", text)
            self.assertIn("ck_qwen3vl_prefill_deepstack_add(model, 0, num_tokens);", text)
            self.assertIn(
                """    /* Op 24: gemm_nt_q6_k_q8_k (mlp_down) layer=0 */
    if (debug_mlp_down_fp32 && ck_debug_mlp_down_fp32_input != NULL) {
        gemm_nt_q6_k(
            ck_debug_mlp_down_fp32_input,
            (const void*)(model->bump + W_LAYER_0_W2),
            NULL,
            (float*)(model->bump + A_EMBEDDED_INPUT),
            num_tokens,
            16,
            32
        );
    } else {""",
                text,
            )
            self.assertIn('const char *bridge_fp32_env = getenv("CK_V8_QWEN3VL_PREFILL_FP32");', text)
            self.assertIn("int bridge_force_fp32 = bridge_fp32_env ? (atoi(bridge_fp32_env) != 0) : 0;", text)
            self.assertIn("if (ck_qwen3vl_prefill_bridge_is_active() && bridge_force_fp32) {", text)
            self.assertIn("debug_outproj_fp32 = 1;", text)
            self.assertIn("debug_mlp_down_fp32 = 1;", text)
            self.assertIn("g_qwen3vl_prefill_total_tokens = total_tokens;", text)
            self.assertIn("g_qwen3vl_prefill_prefix_start = prefix_start;", text)
            self.assertIn("g_qwen3vl_prefill_prefix_tokens = prefix_tokens;", text)
            self.assertIn("int rc = ck_embed_tokens_at(g_model, tokens_after, tokens_after_count, tokens_before_count + prefix_tokens);", text)
            self.assertIn("ck_prefill_from_embedded(g_model, total_tokens);", text)
            self.assertIn(
                "int debug_outproj_fp32 = debug_outproj_env ? (atoi(debug_outproj_env) != 0) : 0;",
                text,
            )
            self.assertIn(
                "int debug_mlp_down_fp32 = debug_mlp_down_env ? (atoi(debug_mlp_down_env) != 0) : 0;",
                text,
            )
            self.assertIn("return ck_model_forward_mixed_ex(prefix_embeddings, prefix_tokens, (16), tokens, token_count, output);", text)
            self.assertNotIn("static int g_bridge_deepstack_active;", text)
            self.assertNotIn("static void ck_decode_embedded(CKModel *model)", text)
            self.assertNotIn("static int ck_bridge_forward_staged", text)

    def test_qwen3vl_decoder_codegen_without_prefill_layout_uses_staged_decode_bridge(self) -> None:
        manifest = _make_qwen3vl_decoder_manifest()

        with tempfile.TemporaryDirectory(prefix="v8_codegen_qwen3vl_decode_bridge_") as tmpdir:
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
            c_path = tmp / "decoder_v8_qwen3vl_decode_bridge.c"

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

            text = c_path.read_text(encoding="utf-8")
            self.assertIn("static void ck_prefill_from_embedded", text)
            self.assertIn("static void ck_decode_embedded(CKModel *model)", text)
            self.assertIn("static int ck_bridge_forward_staged(CKModel *model, int total_tokens)", text)
            self.assertIn("CK_BRIDGE_ROW_EMBED = 1", text)
            self.assertIn("ck_decode_embedded(model);", text)
            self.assertIn("static void ck_qwen3vl_runtime_mrope_qk(CKModel *model, float *q, float *k,", text)
            self.assertIn("if (model && model->bridge_has_explicit_positions)", text)
            self.assertIn("g_bridge_deepstack_slices", text)
            self.assertIn("int rc = ck_bridge_forward_staged(g_model, total_tokens);", text)
            self.assertNotIn("ck_prefill_from_embedded(g_model, prefix_tokens);", text)

            make_result = subprocess.run(
                ["make", "build/libckernel_engine.so"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(make_result.returncode, 0, msg=make_result.stderr)

            so_path = tmp / "decoder_v8_qwen3vl_decode_bridge.so"
            compile_result = subprocess.run(
                [
                    "cc",
                    "-shared",
                    "-fPIC",
                    "-O3",
                    "-fopenmp",
                    "-Iinclude",
                    "-Iversion/v8/src",
                    str(c_path),
                    "version/v8/src/ckernel_model_load_v8.c",
                    "version/v8/src/ck_parallel_decode_v8.c",
                    "version/v8/src/ck_parallel_prefill_v8.c",
                    "-Lbuild",
                    "-lckernel_engine",
                    f"-Wl,-rpath,{BUILD_DIR}",
                    "-o",
                    str(so_path),
                    "-lm",
                    "-lpthread",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(compile_result.returncode, 0, msg=compile_result.stderr)


    def test_qwen3vl_decoder_codegen_emits_deepstack_bridge_api_with_parity_dump(self) -> None:
        manifest = _make_qwen3vl_decoder_manifest()

        with tempfile.TemporaryDirectory(prefix="v8_codegen_qwen3vl_deepstack_bridge_dump_") as tmpdir:
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
            c_path = tmp / "decoder_v8_qwen3vl_bridge_dump.c"

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
                    "--prefill-layout",
                    str(prefill_layout),
                    "--layout",
                    str(decode_layout),
                    "--parity-dump",
                    "--output",
                    str(c_path),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            text = c_path.read_text(encoding="utf-8")
            self.assertIn("#ifdef CK_PARITY_DUMP", text)
            self.assertIn("ck_qwen3vl_prefill_deepstack_add(model, 0, num_tokens);", text)
            self.assertNotIn("g_bridge_deepstack_slices + 0", text)

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
                    "--prefill-layout",
                    str(prefill_layout),
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
            self.assertIn("CK_EXPORT int ck_model_write_embeddings", text)
            self.assertIn("CK_EXPORT int ck_model_write_embeddings_ex", text)
            self.assertIn("CK_EXPORT int ck_model_embed_tokens_at", text)
            self.assertIn("CK_EXPORT int ck_model_forward_from_embeddings", text)
            self.assertIn("CK_EXPORT int ck_model_forward_mixed", text)
            self.assertIn("CK_EXPORT int ck_model_forward_mixed_ex", text)
            self.assertIn("CK_EXPORT int ck_model_forward_mixed_grid_ex", text)
            self.assertIn("CK_EXPORT intptr_t ck_model_get_named_activation_runtime_offset", text)
            self.assertIn("CK_EXPORT intptr_t ck_model_get_named_activation_nbytes", text)
            self.assertIn("CK_EXPORT uintptr_t ck_model_get_named_activation_ptr", text)
            self.assertIn("rope_precompute_cache(", text)
            self.assertNotIn("/* No pre-weights init ops */", text)
            self.assertIn("logits (last-only)", text)
            self.assertNotIn("copy_last_logits (prefill fixup)", text)
            self.assertNotIn("static void ck_decode_embedded", text)
            self.assertNotIn("static int ck_bridge_forward_staged", text)

    def test_prefill_codegen_emits_multimodal_bridge_api(self) -> None:
        manifest = _make_qwen3vl_decoder_manifest()

        with tempfile.TemporaryDirectory(prefix="v8_codegen_prefill_bridge_") as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "weights_manifest.json"
            prefill_ir1 = tmp / "ir1_prefill.json"
            prefill_layout = tmp / "layout_prefill.json"
            prefill_lowered = tmp / "lowered_prefill.json"
            prefill_call = tmp / "call_prefill.json"
            c_path = tmp / "decoder_v8_prefill_bridge.c"

            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            rc = build_ir_v8.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--mode",
                    "prefill",
                    "--output",
                    str(prefill_ir1),
                    "--layout-output",
                    str(prefill_layout),
                    "--lowered-output",
                    str(prefill_lowered),
                    "--call-output",
                    str(prefill_call),
                ]
            )
            self.assertEqual(rc, 0)

            result = subprocess.run(
                [
                    sys.executable,
                    str(V8_CODEGEN_PATH),
                    "--ir",
                    str(prefill_call),
                    "--layout",
                    str(prefill_layout),
                    "--output",
                    str(c_path),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            text = c_path.read_text(encoding="utf-8")
            self.assertIn("static void ck_prefill_from_embedded", text)
            self.assertIn("CK_EXPORT int ck_model_write_embeddings", text)
            self.assertIn("CK_EXPORT int ck_model_write_embeddings_ex", text)
            self.assertIn("CK_EXPORT int ck_model_embed_tokens_at", text)
            self.assertIn("CK_EXPORT int ck_model_forward_from_embeddings", text)
            self.assertIn("CK_EXPORT int ck_model_forward_mixed", text)
            self.assertIn("CK_EXPORT int ck_model_forward_mixed_ex", text)
            self.assertIn("CK_EXPORT int ck_model_forward_mixed_grid_ex", text)
            self.assertIn("if (prefix_grid_x > 0 && prefix_grid_y > 0 && prefix_grid_x * prefix_grid_y != prefix_tokens) return -10;", text)
            self.assertIn("static void kv_cache_batch_copy(", text)
            self.assertNotIn("vocab_size * sizeof(float)", text)
            self.assertIn("transpose_v_to_head_major layer=0", text)
            self.assertIn("float *buf = (float*)(model->bump + A_V_SCRATCH);", text)
            self.assertIn("ck_qwen3vl_prefill_bridge_prepare", text)
            self.assertIn("ck_prefill_from_embedded(g_model, total_tokens);", text)
            self.assertIn("ck_decode(g_model, tokens[i]);", text)

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
                    "--prefill-layout",
                    str(prefill_layout),
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
