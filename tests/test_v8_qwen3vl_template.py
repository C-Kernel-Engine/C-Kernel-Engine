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


build_ir_v8 = _load_module("build_ir_v8_qwen3vl_tests", V8_BUILD_PATH)


def _entry(name: str, dtype: str, shape: list[int], offset: int) -> dict:
    nbytes_per = {"fp32": 4, "fp16": 2, "q8_0": 1}.get(dtype, 4)
    size = 1
    for dim in shape:
        size *= int(dim)
    size *= nbytes_per
    return {"name": name, "dtype": dtype, "offset": offset, "shape": shape, "size": size}


def _make_qwen3vl_manifest() -> dict:
    offset = 0
    entries = []
    def add(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        e = _entry(name, dtype, shape, offset)
        entries.append(e)
        offset += int(e["size"])

    add("v.patch_embd.weight", "fp32", [1152, 768])
    add("v.patch_embd.weight.1", "fp32", [1152, 768])
    add("v.patch_embd.bias", "fp32", [1152])
    add("v.position_embd.weight", "fp32", [2304, 1152])
    add("v.blk.0.ln1.weight", "fp32", [1152])
    add("v.blk.0.ln1.bias", "fp32", [1152])
    add("v.blk.0.ln2.weight", "fp32", [1152])
    add("v.blk.0.ln2.bias", "fp32", [1152])
    add("v.blk.0.attn_qkv.weight", "q8_0", [3456, 1152])
    add("v.blk.0.attn_qkv.bias", "fp32", [3456])
    add("v.blk.0.attn_out.weight", "q8_0", [1152, 1152])
    add("v.blk.0.attn_out.bias", "fp32", [1152])
    add("v.blk.0.ffn_up.weight", "q8_0", [4304, 1152])
    add("v.blk.0.ffn_up.bias", "fp32", [4304])
    add("v.blk.0.ffn_down.weight", "fp16", [1152, 4304])
    add("v.blk.0.ffn_down.bias", "fp32", [1152])
    add("v.deepstack.0.norm.weight", "fp32", [4608])
    add("v.deepstack.0.norm.bias", "fp32", [4608])
    add("v.deepstack.0.fc1.weight", "q8_0", [4608, 4608])
    add("v.deepstack.0.fc1.bias", "fp32", [4608])
    add("v.deepstack.0.fc2.weight", "q8_0", [4096, 4608])
    add("v.deepstack.0.fc2.bias", "fp32", [4096])
    add("v.post_ln.weight", "fp32", [1152])
    add("v.post_ln.bias", "fp32", [1152])
    add("mm.0.weight", "q8_0", [4608, 4608])
    add("mm.0.bias", "fp32", [4608])
    add("mm.2.weight", "q8_0", [4096, 4608])
    add("mm.2.bias", "fp32", [4096])

    return {
        "config": {
            "model": "qwen3_vl_vision",
            "arch": "qwen3_vl_vision",
            "num_layers": 1,
            "embed_dim": 1152,
            "num_heads": 16,
            "num_kv_heads": 16,
            "head_dim": 72,
            "attn_out_dim": 1152,
            "intermediate_size": 4304,
            "context_length": 2304,
            "max_seq_len": 2304,
            "image_size": 768,
            "patch_size": 16,
            "vision_channels": 3,
            "patch_dim": 768,
            "vision_grid_h": 48,
            "vision_grid_w": 48,
            "vision_num_patches": 2304,
            "spatial_merge_size": 2,
            "q_dim": 1152,
            "k_dim": 1152,
            "v_dim": 1152,
            "spatial_merge_factor": 4,
            "vision_merged_tokens": 576,
            "projector_in_dim": 4608,
            "projector_hidden_dim": 4608,
            "projector_out_dim": 4096,
            "projector_total_out_dim": 8192,
            "projection_dim": 4096,
            "deepstack_layer_indices": [0],
            "num_deepstack_layers": 1,
            "prefer_q8_activation": True
        },
        "quant_summary": {
            "layer.0": {
                "attn_qkv": "q8_0",
                "wo": "q8_0",
                "w2": "fp16",
                "w3": "q8_0"
            },
            "patch_emb": "fp32",
            "patch_emb_aux": "fp32",
            "mm0_w": "q8_0",
            "mm1_w": "q8_0",
            "deepstack.0": {
                "norm": "fp32",
                "fc1": "q8_0",
                "fc2": "q8_0"
            }
        },
        "entries": entries,
        "template": build_ir_v8._load_builtin_template_doc("qwen3_vl_vision"),
    }


class V8Qwen3VLTemplateTests(unittest.TestCase):
    def test_builtin_template_declares_qwen3vl_vision_contract(self) -> None:
        doc = build_ir_v8._load_builtin_template_doc("qwen3_vl_vision")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["version"], 3)
        self.assertEqual(doc["sequence"], ["vision_encoder"])
        self.assertIn("vision_position_contract", doc["contract"])
        self.assertEqual(doc["contract"]["attention_contract"]["rope_layout"], "multi_section_2d")
        self.assertEqual(doc["block_types"]["vision_encoder"]["header"][-1]["op"], "position_ids_2d")
        self.assertEqual(
            doc["block_types"]["vision_encoder"]["header"][3]["params"]["merge_size_from_config"],
            "spatial_merge_size",
        )
        self.assertEqual(
            doc["block_types"]["vision_encoder"]["header"][5]["params"]["merge_size_from_config"],
            "spatial_merge_size",
        )
        self.assertEqual(
            doc["block_types"]["vision_encoder"]["header"][6]["params"]["merge_size_from_config"],
            "spatial_merge_size",
        )
        self.assertEqual(doc["block_types"]["vision_encoder"]["body"]["ops"][3]["op"], "rope_qk")
        self.assertEqual(doc["block_types"]["vision_encoder"]["body"]["ops"][8]["op"], "mlp_up")
        self.assertEqual(doc["block_types"]["vision_encoder"]["body"]["ops"][9]["op"], "gelu")
        branch = doc["block_types"]["vision_encoder"]["branches"][0]
        self.assertEqual(branch["name"], "deepstack")
        self.assertEqual(branch["tap"]["from"], "body.mlp_residual.out")
        self.assertEqual(
            branch["producer"]["ops"][0]["params"]["merge_size_from_config"],
            "spatial_merge_size",
        )
        self.assertEqual(
            doc["block_types"]["vision_encoder"]["footer"][1]["params"]["merge_size_from_config"],
            "spatial_merge_size",
        )
        self.assertEqual(doc["block_types"]["vision_encoder"]["footer"][-1]["op"], "branch_concat")
        self.assertEqual(doc["kernels"]["layernorm"], "layernorm_fp32_exact")
        self.assertEqual(doc["kernels"]["branch_layernorm"], "layernorm_fp32_exact")

    def test_qwen3vl_branch_plan_reads_template_declared_layers(self) -> None:
        manifest = _make_qwen3vl_manifest()
        plan = build_ir_v8.build_template_branch_plan(manifest)
        self.assertEqual(plan["format"], "v8-template-branch-plan")
        self.assertEqual(plan["sequence"], ["vision_encoder"])
        self.assertEqual(len(plan["blocks"]), 1)
        branch = plan["blocks"][0]["branches"][0]
        self.assertEqual(branch["name"], "deepstack")
        self.assertEqual(branch["status"], "active")
        self.assertEqual(branch["layers"], [0])
        self.assertEqual(branch["tap_ref"]["section"], "body")
        self.assertEqual(branch["tap_ref"]["op_id"], "mlp_residual")
        self.assertEqual(branch["collect_contract"]["target"], "branch.deepstack")
        self.assertEqual(branch["collect_contract"]["rows"], 576)
        self.assertEqual(branch["collect_contract"]["slice_dim"], 4096)
        self.assertEqual(branch["collect_contract"]["num_slices"], 1)
        self.assertEqual(
            branch["producer_ops"],
            ["branch_spatial_merge", "branch_layernorm", "branch_fc1", "branch_gelu", "branch_fc2"],
        )
        self.assertEqual(branch["stitches"][0]["op"], "branch_concat")

    def test_qwen3vl_prefill_lowering_emits_vision_merger_ops(self) -> None:
        manifest = _make_qwen3vl_manifest()
        ir1_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "qwen3_vl_vision_manifest.synthetic.json",
            mode="prefill",
        )
        ops = [op["op"] for op in ir1_ops]
        self.assertIn("patchify", ops)
        self.assertIn("patch_proj", ops)
        self.assertIn("patch_proj_aux", ops)
        self.assertIn("add_stream", ops)
        self.assertIn("patch_bias_add", ops)
        self.assertIn("position_embeddings", ops)
        self.assertIn("position_ids_2d", ops)
        self.assertIn("qkv_packed_proj", ops)
        self.assertIn("split_qkv_packed", ops)
        self.assertIn("rope_qk", ops)
        self.assertIn("attn", ops)
        self.assertIn("mlp_up", ops)
        self.assertIn("gelu", ops)
        self.assertIn("spatial_merge", ops)
        self.assertIn("projector_fc1", ops)
        self.assertIn("projector_gelu", ops)
        self.assertIn("projector_fc2", ops)
        self.assertIn("branch_spatial_merge", ops)
        self.assertIn("branch_layernorm", ops)
        self.assertIn("branch_fc1", ops)
        self.assertIn("branch_gelu", ops)
        self.assertIn("branch_fc2", ops)
        self.assertIn("branch_concat", ops)

        by_op = {}
        for ir_op in ir1_ops:
            by_op.setdefault(ir_op["op"], []).append(ir_op)

        self.assertEqual(by_op["patch_proj"][0]["weights"]["patch_emb"]["name"], "v.patch_embd.weight")
        self.assertEqual(by_op["patch_proj_aux"][0]["weights"]["patch_emb_aux"]["name"], "v.patch_embd.weight.1")
        self.assertEqual(by_op["patch_bias_add"][0]["weights"]["patch_bias"]["name"], "v.patch_embd.bias")
        self.assertEqual(by_op["position_embeddings"][0]["weights"]["pos_emb"]["name"], "v.position_embd.weight")
        self.assertEqual(by_op["qkv_packed_proj"][0]["weights"]["attn_qkv"]["name"], "v.blk.0.attn_qkv.weight")
        self.assertEqual(by_op["projector_fc1"][0]["weights"]["mm0_w"]["name"], "mm.0.weight")
        self.assertEqual(by_op["projector_fc1"][0]["weights"]["mm0_b"]["name"], "mm.0.bias")
        self.assertEqual(by_op["projector_fc2"][0]["weights"]["mm1_w"]["name"], "mm.2.weight")
        self.assertEqual(by_op["projector_fc2"][0]["weights"]["mm1_b"]["name"], "mm.2.bias")
        self.assertEqual(by_op["mlp_up"][0]["weights"]["w3"]["name"], "v.blk.0.ffn_up.weight")
        self.assertEqual(by_op["mlp_up"][0]["weights"]["b1"]["name"], "v.blk.0.ffn_up.bias")
        self.assertEqual(by_op["branch_layernorm"][0]["weights"]["branch_norm_gamma"]["name"], "v.deepstack.0.norm.weight")
        self.assertEqual(by_op["branch_layernorm"][0]["weights"]["branch_norm_beta"]["name"], "v.deepstack.0.norm.bias")
        self.assertEqual(by_op["branch_fc1"][0]["weights"]["branch_fc1_w"]["name"], "v.deepstack.0.fc1.weight")
        self.assertEqual(by_op["branch_fc1"][0]["weights"]["branch_fc1_b"]["name"], "v.deepstack.0.fc1.bias")
        self.assertEqual(by_op["branch_fc2"][0]["weights"]["branch_fc2_w"]["name"], "v.deepstack.0.fc2.weight")
        self.assertEqual(by_op["branch_fc2"][0]["weights"]["branch_fc2_b"]["name"], "v.deepstack.0.fc2.bias")
        tapped = [
            op for op in ir1_ops
            if op.get("template_op_id") == "mlp_residual" and op.get("graph", {}).get("branch_taps")
        ]
        self.assertEqual(len(tapped), 1)
        self.assertEqual(tapped[0]["graph"]["branch_taps"][0]["name"], "deepstack")
        self.assertEqual(
            tapped[0]["graph"]["branch_taps"][0]["collect"]["target"],
            "branch.deepstack",
        )
        self.assertEqual(by_op["branch_fc2"][0]["params"]["out_dim_from_config"], "projector_out_dim")
        self.assertEqual(by_op["branch_fc2"][0]["params"]["branch_collect_target"], "branch.deepstack")
        self.assertEqual(by_op["branch_fc2"][0]["params"]["branch_collect_rows"], 576)
        self.assertEqual(by_op["branch_fc2"][0]["params"]["branch_collect_slice_dim"], 4096)
        self.assertEqual(by_op["branch_concat"][0]["params"]["main_dim_from_config"], "projector_out_dim")
        self.assertEqual(by_op["branch_concat"][0]["params"]["branch_slice_dim_from_config"], "projector_out_dim")
        self.assertEqual(by_op["branch_concat"][0]["params"]["num_branch_slices_from_config"], "num_deepstack_layers")

    def test_qwen3vl_codegen_smoke_emits_c(self) -> None:
        manifest = _make_qwen3vl_manifest()
        with tempfile.TemporaryDirectory(prefix="v8_qwen3vl_codegen_") as td:
            td_path = Path(td)
            manifest_path = td_path / "weights_manifest.json"
            ir1_path = td_path / "ir1.json"
            layout_path = td_path / "layout.json"
            lowered_path = td_path / "lowered.json"
            call_path = td_path / "call.json"
            c_path = td_path / "model_v8_qwen3vl.c"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            rc = build_ir_v8.main(
                [
                    "--manifest", str(manifest_path),
                    "--mode", "prefill",
                    "--output", str(ir1_path),
                    "--layout-output", str(layout_path),
                    "--lowered-output", str(lowered_path),
                    "--call-output", str(call_path),
                ]
            )
            self.assertEqual(rc, 0)
            call_doc = json.loads(call_path.read_text(encoding="utf-8"))
            call_ops = call_doc.get("operations", [])
            layout_doc = json.loads(layout_path.read_text(encoding="utf-8"))
            vision_positions = next(
                buf for buf in layout_doc["memory"]["activations"]["buffers"]
                if buf["name"] == "vision_positions"
            )
            self.assertEqual(vision_positions["dtype"], "i32")
            patch_bias = next(op for op in call_ops if op.get("op") == "patch_bias_add")
            patch_bias_x = next(arg for arg in patch_bias.get("args", []) if arg.get("name") == "x")
            self.assertEqual(patch_bias_x.get("buffer_ref"), "embedded_input")
            rope = next(op for op in call_ops if op.get("op") == "rope_qk")
            rope_positions = next(arg for arg in rope.get("args", []) if arg.get("name") == "positions")
            self.assertEqual(rope_positions.get("buffer_ref"), "vision_positions")
            split_qkv = next(op for op in call_ops if op.get("op") == "split_qkv_packed")
            self.assertEqual(split_qkv.get("function"), "split_qkv_packed_head_major_forward")
            qkv_packed_proj = next(op for op in call_ops if op.get("op") == "qkv_packed_proj")
            self.assertEqual(qkv_packed_proj.get("function"), "gemm_nt_q8_0_q8_0_contract")
            attn = next(op for op in call_ops if op.get("op") == "attn")
            self.assertEqual(attn.get("function"), "attention_forward_full_head_major_gqa_ggml_strided")
            attn_idx = next(i for i, op in enumerate(call_ops) if op.get("op") == "attn")
            transpose_idx = next(i for i, op in enumerate(call_ops) if op.get("op") == "transpose_attn_out_to_token_major")
            out_proj_idx = next(i for i, op in enumerate(call_ops) if op.get("op") == "out_proj")
            self.assertLess(attn_idx, transpose_idx)
            self.assertLess(transpose_idx, out_proj_idx)
            out_proj = next(op for op in call_ops if op.get("op") == "out_proj")
            self.assertEqual(out_proj.get("function"), "gemm_nt_q8_0_q8_0_contract")
            self.assertNotIn("kv_cache_batch_copy", [op.get("op") for op in call_ops])
            projector_fc1 = next(op for op in call_ops if op.get("op") == "projector_fc1")
            self.assertEqual(projector_fc1.get("function"), "gemm_nt_q8_0_q8_0_contract")
            projector_fc1_bias = next(arg for arg in projector_fc1.get("args", []) if arg.get("name") == "bias")
            self.assertEqual(projector_fc1_bias.get("weight_ref"), "mm.0.bias")
            projector_fc2 = next(op for op in call_ops if op.get("op") == "projector_fc2")
            self.assertEqual(projector_fc2.get("function"), "gemm_nt_q8_0_q8_0_contract")
            projector_fc2_bias = next(arg for arg in projector_fc2.get("args", []) if arg.get("name") == "bias")
            self.assertEqual(projector_fc2_bias.get("weight_ref"), "mm.2.bias")
            branch_fc1 = next(op for op in call_ops if op.get("op") == "branch_fc1")
            self.assertEqual(branch_fc1.get("function"), "gemm_nt_q8_0_q8_0_contract")
            branch_fc1_bias = next(arg for arg in branch_fc1.get("args", []) if arg.get("name") == "bias")
            self.assertEqual(branch_fc1_bias.get("weight_ref"), "v.deepstack.0.fc1.bias")
            branch_fc2 = next(op for op in call_ops if op.get("op") == "branch_fc2")
            self.assertEqual(branch_fc2.get("function"), "gemm_nt_q8_0_q8_0_contract")
            branch_fc2_bias = next(arg for arg in branch_fc2.get("args", []) if arg.get("name") == "bias")
            self.assertEqual(branch_fc2_bias.get("weight_ref"), "v.deepstack.0.fc2.bias")
            patch_proj = next(op for op in call_ops if op.get("op") == "patch_proj")
            self.assertEqual(patch_proj.get("function"), "gemm_naive_parallel")
            patch_proj_aux = next(op for op in call_ops if op.get("op") == "patch_proj_aux")
            self.assertEqual(patch_proj_aux.get("function"), "gemm_naive_parallel")
            attn_norm = next(op for op in call_ops if op.get("op") == "layernorm")
            self.assertEqual(attn_norm.get("function"), "layernorm_naive_serial_matched_precision")
            branch_norm = next(op for op in call_ops if op.get("op") == "branch_layernorm")
            self.assertEqual(branch_norm.get("function"), "layernorm_naive_serial_matched_precision")
            mlp_up = next(op for op in call_ops if op.get("op") == "mlp_up")
            self.assertEqual(mlp_up.get("function"), "gemm_nt_q8_0_q8_0_contract")
            mlp_up_n = next(arg for arg in mlp_up.get("args", []) if arg.get("name") == "N")
            self.assertEqual(mlp_up_n.get("expr"), str(manifest["config"]["intermediate_size"]))
            gelu = next(op for op in call_ops if op.get("op") == "gelu")
            self.assertEqual(gelu.get("function"), "gelu_ggml_inplace")
            gelu_n = next(arg for arg in gelu.get("args", []) if arg.get("name") == "n")
            self.assertEqual(
                gelu_n.get("expr"),
                str(manifest["config"]["context_length"] * manifest["config"]["intermediate_size"]),
            )
            projector_gelu = next(op for op in call_ops if op.get("op") == "projector_gelu")
            self.assertEqual(projector_gelu.get("function"), "gelu_ggml_inplace")
            branch_gelu = next(op for op in call_ops if op.get("op") == "branch_gelu")
            self.assertEqual(branch_gelu.get("function"), "gelu_ggml_inplace")
            mlp_down = next(op for op in call_ops if op.get("op") == "mlp_down")
            self.assertEqual(mlp_down.get("function"), "gemm_nt_f16")

            result = subprocess.run(
                [
                    sys.executable,
                    str(V8_CODEGEN_PATH),
                    "--ir", str(call_path),
                    "--layout", str(layout_path),
                    "--output", str(c_path),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(c_path.exists())
            text = c_path.read_text(encoding="utf-8")
            self.assertIn("gemm_naive_parallel", text)
            self.assertIn("position_embeddings_add_tiled_2d", text)
            self.assertIn("spatial_merge_contiguous_tiled", text)
            self.assertIn("add_stream_reorder_2d", text)
            self.assertIn("vision_bridge_output", text)
            self.assertIn("ck_strict_mtmd_clip_encode_planar_f32", text)
            self.assertIn("gelu_ggml_inplace", text)
            self.assertIn("layernorm_naive_serial_matched_precision", text)
            self.assertIn("feature_concat", text)
            self.assertIn("transpose_attn_out_to_token_major", text)
            self.assertNotIn("transpose_inplace();", text)

    def test_qwen3vl_lowering_requires_spatial_merge_size_config(self) -> None:
        manifest = _make_qwen3vl_manifest()
        manifest["config"].pop("spatial_merge_size", None)
        ir1_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "qwen3_vl_vision_manifest.synthetic.json",
            mode="prefill",
        )
        registry = build_ir_v8.load_kernel_registry()
        lowered_ir1 = build_ir_v8.generate_ir_lower_1(
            ir1_ops,
            registry,
            manifest,
            "prefill",
        )
        layout = build_ir_v8.generate_memory_layout(
            lowered_ir1,
            manifest,
            registry,
            mode="prefill",
            context_len=manifest["config"]["context_length"],
        )
        with self.assertRaisesRegex(RuntimeError, "spatial_merge_size"):
            build_ir_v8.generate_ir_lower_2(
                lowered_ir1,
                layout,
                manifest,
                registry,
                mode="prefill",
            )


if __name__ == "__main__":
    unittest.main()
