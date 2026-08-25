#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
V8_BUILD_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
V8_CODEGEN_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_v8.py"
V8_CODEGEN_CORE_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_core_v8.py"
V8_CONVERT_PATH = ROOT / "version" / "v8" / "scripts" / "convert_gguf_to_bump_v8.py"
V8_BRIDGE_PATH = ROOT / "version" / "v8" / "scripts" / "run_multimodal_bridge_v8.py"


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
codegen_core_v8 = _load_module("codegen_core_v8_qwen3vl_tests", V8_CODEGEN_CORE_PATH)
convert_gguf_to_bump_v8 = _load_module("convert_gguf_to_bump_v8_qwen3vl_tests", V8_CONVERT_PATH)
run_multimodal_bridge_v8 = _load_module("run_multimodal_bridge_v8_qwen3vl_tests", V8_BRIDGE_PATH)


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
            "prefer_q8_activation": True,
            "vision_mrope_n_dims": 72,
            "vision_mrope_sections": [18, 18, 0, 0],
            "vision_mrope_storage_boundary": "bf16"
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
    def test_bridge_uses_aligned_arena_activation_base(self) -> None:
        layout = {
            "memory": {
                "arena": {"activations_base": 1024},
                "weights": {"base_offset": 4, "size": 1016},
            }
        }
        self.assertEqual(run_multimodal_bridge_v8._activation_runtime_base(layout), 1024)
        self.assertEqual(
            run_multimodal_bridge_v8._activation_runtime_offset(layout, {"offset": 64}),
            1088,
        )

    def test_bridge_legacy_layout_derives_activation_base_from_weights(self) -> None:
        layout = {"memory": {"weights": {"base_offset": 4, "size": 1016}}}
        self.assertEqual(run_multimodal_bridge_v8._activation_runtime_base(layout), 1020)

    def test_bf16_decoder_norm_contracts_select_pytorch_providers(self) -> None:
        manifest = {
            "config": {
                "model": "qwen3vl",
                "arch": "qwen3vl",
                "decoder_norm_storage_boundary": "bf16",
                "decoder_norm_reduction_policy": "pytorch_avx2_cascade_exact",
                "decoder_qk_norm_reduction_policy": "pytorch_avx2_cascade_exact",
            },
            "template": build_ir_v8._load_builtin_template_doc("qwen3vl"),
        }
        plans = build_ir_v8._resolve_manifest_execution_contracts(manifest, "decode")
        selected = {plan["operation"]: plan for plan in plans}
        self.assertNotIn("decoder.rmsnorm", selected)
        self.assertNotIn("decoder.qk_norm", selected)
        self.assertEqual(
            selected["decoder.rmsnorm.bf16"]["kernel"]["function"],
            "rmsnorm_forward_pytorch_bf16_storage",
        )
        self.assertEqual(
            selected["decoder.qk_norm.bf16"]["kernel"]["function"],
            "qk_norm_forward_pytorch_bf16_storage",
        )
        self.assertEqual(
            selected["decoder.rmsnorm.bf16"]["contract"]["id"],
            "rmsnorm_pytorch_avx2_cascade_bf16_storage",
        )
        self.assertEqual(
            selected["decoder.qk_norm.bf16"]["contract"]["id"],
            "rmsnorm_pytorch_avx2_cascade_bf16_storage",
        )

    def test_bf16_decoder_projection_and_residual_contracts_cover_both_phases(self) -> None:
        manifest = {
            "config": {
                "model": "qwen3vl",
                "arch": "qwen3vl",
                "decoder_norm_storage_boundary": "bf16",
                "decoder_prefill_projection_storage_boundary": "bf16",
                "decoder_projection_reduction_policy": "pytorch_onednn_brgemm_exact",
                "decoder_residual_storage_boundary": "bf16",
            },
            "template": build_ir_v8._load_builtin_template_doc("qwen3vl"),
        }
        expected = {
            "decoder.layer.qkv_projection.bf16_pytorch_onednn",
            "decoder.layer.out_projection.bf16_pytorch_onednn",
            "decoder.layer.mlp_projection.bf16_pytorch_onednn",
            "decoder.logits.bf16_pytorch_onednn",
            "decoder.layer.residual.bf16_pytorch",
        }
        for phase in ("prefill", "decode"):
            plans = build_ir_v8._resolve_manifest_execution_contracts(manifest, phase)
            selected = {plan["operation"]: plan for plan in plans}
            self.assertTrue(expected.issubset(selected), (phase, sorted(selected)))
            for operation in expected - {"decoder.layer.residual.bf16_pytorch"}:
                self.assertEqual(
                    selected[operation]["kernel"]["function"],
                    "gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage",
                )
                self.assertEqual(
                    selected[operation]["contract"]["id"],
                    "bf16_weight_bf16_input_pytorch_onednn_brgemm_bf16_output",
                )
            residual = selected["decoder.layer.residual.bf16_pytorch"]
            self.assertEqual(
                residual["kernel"]["function"],
                "ck_residual_add_token_major_bf16_storage",
            )
            self.assertEqual(
                residual["contract"]["id"],
                "residual_add_bf16_input_fp32_add_bf16_output",
            )

    def test_bf16_decoder_mrope_contract_covers_prefill_and_decode(self) -> None:
        manifest = {
            "config": {
                "model": "qwen3vl",
                "arch": "qwen3vl",
                "decoder_mrope_storage_boundary": "pytorch_bf16_exact",
            },
            "template": build_ir_v8._load_builtin_template_doc("qwen3vl"),
        }
        for phase in ("prefill", "decode"):
            plans = build_ir_v8._resolve_manifest_execution_contracts(manifest, phase)
            selected = {
                plan["operation"]: plan
                for plan in plans
                if plan["operation"].startswith("decoder.layer.mrope")
            }
            self.assertEqual(set(selected), {"decoder.layer.mrope.bf16_pytorch"})
            plan = selected["decoder.layer.mrope.bf16_pytorch"]
            self.assertEqual(
                plan["kernel"]["function"],
                "mrope_qk_text_imrope_bf16_pytorch_storage",
            )
            self.assertEqual(
                plan["contract"]["id"],
                "text_imrope_bf16_input_pytorch_bf16_compute_bf16_output",
            )

    def test_bf16_decoder_swiglu_contract_covers_prefill_and_decode(self) -> None:
        manifest = {
            "config": {
                "model": "qwen3vl",
                "arch": "qwen3vl",
                "decoder_swiglu_storage_boundary": "pytorch_bf16_exact",
            },
            "template": build_ir_v8._load_builtin_template_doc("qwen3vl"),
        }
        for phase in ("prefill", "decode"):
            plans = build_ir_v8._resolve_manifest_execution_contracts(manifest, phase)
            selected = {
                plan["operation"]: plan
                for plan in plans
                if plan["operation"].startswith("decoder.swiglu")
            }
            self.assertEqual(set(selected), {"decoder.swiglu.bf16"})
            plan = selected["decoder.swiglu.bf16"]
            self.assertEqual(
                plan["kernel"]["function"],
                "swiglu_forward_pytorch_bf16_storage",
            )
            self.assertEqual(
                plan["contract"]["id"],
                "swiglu_pytorch_sleef_bf16_intermediate_bf16_output",
            )


    def test_gguf_vision_mrope_resolves_fp32_contract(self) -> None:
        manifest = _make_qwen3vl_manifest()
        manifest["config"]["vision_mrope_storage_boundary"] = "fp32"
        self.assertEqual(
            convert_gguf_to_bump_v8._inject_runtime_config_defaults(
                manifest["config"], "qwen3_vl_vision"
            )["vision_mrope_storage_boundary"],
            "fp32",
        )
        with tempfile.TemporaryDirectory(prefix="v8_qwen3vl_gguf_mrope_") as td:
            root = Path(td)
            manifest_path = root / "weights_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            paths = {
                name: root / f"{name}.json"
                for name in ("ir1", "layout", "lowered", "call")
            }
            args = [
                "--manifest", str(manifest_path),
                "--mode", "prefill",
                "--output", str(paths["ir1"]),
                "--layout-output", str(paths["layout"]),
                "--lowered-output", str(paths["lowered"]),
                "--call-output", str(paths["call"]),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(build_ir_v8.main(args), 0)

            for artifact in ("ir1", "lowered", "call"):
                doc = json.loads(paths[artifact].read_text())
                operations = doc if isinstance(doc, list) else doc.get("operations", doc.get("ops", []))
                mrope = next(item for item in operations if item.get("op") == "rope_qk")
                self.assertEqual(
                    mrope["resolved_contract"]["contract_id"],
                    "vision_mrope_fp32_input_fp32_compute_fp32_output",
                )
                self.assertEqual(
                    mrope["resolved_contract"]["kernel_id"],
                    "mrope_qk_vision",
                )
                if artifact == "call":
                    self.assertEqual(mrope["function"], "mrope_qk_vision")
                else:
                    self.assertEqual(mrope["kernel"], "mrope_qk_vision")

            generated_c = root / "qwen3vl_vision.c"
            result = subprocess.run(
                [
                    sys.executable,
                    str(V8_CODEGEN_PATH),
                    "--ir",
                    str(paths["call"]),
                    "--layout",
                    str(paths["layout"]),
                    "--output",
                    str(generated_c),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            generated = generated_c.read_text(encoding="utf-8")
            self.assertIn("mrope_qk_vision(", generated)
            self.assertNotIn("ck_multimodal_prefill_mrope_qk", generated)
            result = subprocess.run(
                [
                    "cc",
                    "-fsyntax-only",
                    "-fopenmp",
                    "-Iinclude",
                    "-Iversion/v8/src",
                    str(generated_c),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_vision_mrope_classification_uses_semantics_not_kernel_name(self) -> None:
        operation = {
            "op": "rope_qk",
            "kernel": "deliberately_renamed_kernel",
            "function": "unrelated_function_name",
            "resolved_contract": {
                "semantics": {
                    "operator_family": "vision_mrope",
                    "position_transform": {
                        "pairing": "multi_section",
                        "position_rank": 4,
                    },
                }
            },
        }
        self.assertTrue(build_ir_v8._is_vision_mrope_operation(operation))
        self.assertTrue(codegen_core_v8._is_vision_mrope_operation(operation))

        operation["resolved_contract"]["semantics"]["operator_family"] = "gemm"
        self.assertFalse(build_ir_v8._is_vision_mrope_operation(operation))
        self.assertFalse(codegen_core_v8._is_vision_mrope_operation(operation))

    def test_authoritative_contract_preserves_behavior_and_reaches_call_ir(self) -> None:
        manifest = _make_qwen3vl_manifest()
        with tempfile.TemporaryDirectory(prefix="v8_qwen3vl_contract_equivalence_") as td:
            root = Path(td)
            manifest_path = root / "weights_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            def generate(prefix: str) -> dict[str, object]:
                paths = {
                    name: root / f"{prefix}_{name}.json"
                    for name in ("ir1", "layout", "lowered", "call")
                }
                args = [
                    "--manifest", str(manifest_path),
                    "--mode", "prefill",
                    "--output", str(paths["ir1"]),
                    "--layout-output", str(paths["layout"]),
                    "--lowered-output", str(paths["lowered"]),
                    "--call-output", str(paths["call"]),
                ]
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(build_ir_v8.main(args), 0)
                return {name: json.loads(path.read_text()) for name, path in paths.items()}

            with mock.patch.object(build_ir_v8, "_resolve_manifest_numerical_contracts", return_value=[]):
                legacy = generate("legacy")

            # A stale legacy override must not influence a contract-bearing op.
            manifest["template"]["kernels"]["attn"] = "attention_forward_causal_head_major_gqa_flash"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            contracted = generate("contracted")

            def without_contract_metadata(value):
                if isinstance(value, dict):
                    return {
                        key: without_contract_metadata(item)
                        for key, item in value.items()
                        if key not in {"required_contract", "resolved_contract", "semantic_checkpoints"}
                    }
                if isinstance(value, list):
                    return [without_contract_metadata(item) for item in value]
                return value

            self.assertEqual(without_contract_metadata(contracted), without_contract_metadata(legacy))

            expected_kernel = "attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided"
            for artifact in ("ir1", "lowered", "call"):
                doc = contracted[artifact]
                operations = doc if isinstance(doc, list) else doc.get("operations", doc.get("ops", []))
                attn = next(item for item in operations if item.get("op") == "attn")
                self.assertEqual(
                    attn["required_contract"]["numerics.attention_reduction"],
                    "f16_kv_tiled64_fp32_softmax_fp64_sum_update",
                )
                self.assertEqual(
                    attn["resolved_contract"]["contract_id"],
                    "f16_kv_tiled64_fp32_softmax_fp64_sum_update",
                )
                self.assertEqual(attn["resolved_contract"]["kernel_id"], expected_kernel)
                if artifact == "call":
                    self.assertEqual(attn["function"], expected_kernel)
                else:
                    self.assertEqual(attn["kernel"], expected_kernel)

                mrope = next(item for item in operations if item.get("op") == "rope_qk")
                self.assertEqual(
                    mrope["resolved_contract"]["contract_id"],
                    "vision_mrope_fp32_input_fp32_compute_bf16_output",
                )
                self.assertEqual(
                    mrope["resolved_contract"]["semantics"]["operator_family"],
                    "vision_mrope",
                )
                self.assertEqual(
                    mrope["resolved_contract"]["kernel_id"],
                    "mrope_qk_vision_bf16_storage",
                )
                if artifact == "call":
                    self.assertEqual(mrope["function"], "mrope_qk_vision_bf16_storage")
                else:
                    self.assertEqual(mrope["kernel"], "mrope_qk_vision_bf16_storage")
                if artifact == "lowered":
                    self.assertEqual(mrope["params"]["n_dims"], 72)
                    self.assertEqual(
                        [mrope["params"][f"section_{idx}"] for idx in range(4)],
                        [18, 18, 0, 0],
                    )
                if artifact == "call":
                    args = {arg["name"]: arg["expr"] for arg in mrope["args"]}
                    self.assertEqual(args["n_dims"], "72")
                    self.assertEqual([args[f"section_{idx}"] for idx in range(4)], ["18", "18", "0", "0"])

                checkpoint_by_id = {
                    checkpoint["id"]: checkpoint
                    for operation in operations
                    for checkpoint in operation.get("semantic_checkpoints", [])
                }
                expected_checkpoints = {
                    "vision.frontend.position.output",
                    "vision.layer.0.norm1.output",
                    "vision.layer.0.q.pre_rope",
                    "vision.layer.0.k.pre_rope",
                    "vision.layer.0.v.output",
                    "vision.layer.0.q.post_rope",
                    "vision.layer.0.k.post_rope",
                    "vision.layer.0.attention.output",
                    "vision.layer.0.out_proj.output",
                    "vision.layer.0.residual1.output",
                    "vision.layer.0.norm2.output",
                    "vision.layer.0.mlp.up",
                    "vision.layer.0.mlp.activation",
                    "vision.layer.0.mlp.down",
                    "vision.layer.0.output",
                    "vision.spatial_merge.output",
                    "vision.projector.output",
                    "vision.prefix.output",
                }
                self.assertTrue(expected_checkpoints.issubset(checkpoint_by_id))
                for checkpoint_id in expected_checkpoints:
                    checkpoint = checkpoint_by_id[checkpoint_id]
                    self.assertTrue(checkpoint["kernel_id"])
                    self.assertTrue(checkpoint["function"])
                    self.assertEqual(len(checkpoint["axis_names"]), 2 if checkpoint["logical_layout"] == "token_major" else 3)
                for checkpoint_id in (
                    "vision.layer.0.mlp.up",
                    "vision.layer.0.mlp.activation",
                    "vision.layer.0.mlp.down",
                    "vision.spatial_merge.output",
                    "vision.projector.output",
                    "vision.prefix.output",
                ):
                    self.assertEqual(checkpoint_by_id[checkpoint_id]["storage_dtype"], "bf16")
                self.assertEqual(
                    checkpoint_by_id["vision.spatial_merge.output"]["tensor"],
                    "vision_spatial_merge",
                )
                self.assertEqual(
                    checkpoint_by_id["vision.projector.output"]["tensor"],
                    "vision_projector_out",
                )

    def test_stale_semantic_checkpoint_declaration_hard_fails(self) -> None:
        template = {
            "semantic_checkpoints": {
                "schema": "cke.semantic_checkpoint_contract",
                "schema_version": 1,
                "exports": {
                    "stale": {
                        "section": "body", "template_op_id": "missing", "op": "gelu",
                        "checkpoints": [{
                            "id": "vision.layer.{layer}.mlp.activation", "producer": "missing",
                            "tensor": "ffn_gelu", "logical_layout": "token_major",
                            "axis_names": ["token", "channel"], "storage_dtype": "fp32",
                        }],
                    }
                },
            }
        }
        registry = {"kernels": [{"id": "gelu", "impl": {"function": "gelu_forward"}}]}
        arranged = [{"kernel": "gelu", "op": "gelu", "template_op_id": "mlp_gelu", "section": "body", "layer": 0}]
        with self.assertRaisesRegex(RuntimeError, "checkpoint declarations did not bind"):
            build_ir_v8._attach_semantic_checkpoints(template, arranged, registry)

    def test_template_operation_conditions_are_config_driven(self) -> None:
        section = [
            {
                "id": "exact",
                "op": "patch_projection_image",
                "when": {"config_key": "frontend.policy", "equals": "exact"},
            },
            {
                "id": "legacy",
                "op": "patchify",
                "when": {"config_key": "frontend.policy", "not_equals": "exact"},
            },
        ]
        self.assertEqual(
            [item["id"] for item in build_ir_v8._normalize_template_op_items(
                section, config={"frontend": {"policy": "exact"}}
            )],
            ["exact"],
        )
        self.assertEqual(
            [item["id"] for item in build_ir_v8._normalize_template_op_items(
                section, config={"frontend": {"policy": "legacy"}}
            )],
            ["legacy"],
        )

    def test_invalid_template_operation_condition_hard_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "HARD CIRCUIT CONDITION FAULT"):
            build_ir_v8._normalize_template_op_items(
                [{
                    "op": "patchify",
                    "when": {"config_key": "frontend.policy", "equals": "a", "not_equals": "b"},
                }],
                config={"frontend": {"policy": "a"}},
            )

    def test_semantic_checkpoint_conditions_follow_operation_conditions(self) -> None:
        template = {
            "semantic_checkpoints": {
                "schema": "cke.semantic_checkpoint_contract",
                "schema_version": 1,
                "exports": {
                    "exact": {
                        "section": "header",
                        "template_op_id": "exact",
                        "op": "patch_projection_image",
                        "when": {"config_key": "frontend_policy", "equals": "exact"},
                        "checkpoints": [{
                            "id": "vision.frontend.output", "producer": "exact",
                            "tensor": "vision_patch_projection", "logical_layout": "token_major",
                            "axis_names": ["token", "channel"], "storage_dtype": "bf16",
                        }],
                    },
                    "legacy": {
                        "section": "header",
                        "template_op_id": "legacy",
                        "op": "patchify",
                        "when": {"config_key": "frontend_policy", "not_equals": "exact"},
                        "checkpoints": [{
                            "id": "vision.frontend.output", "producer": "legacy",
                            "tensor": "vision_patchify", "logical_layout": "token_major",
                            "axis_names": ["token", "channel"], "storage_dtype": "fp32",
                        }],
                    },
                },
            }
        }
        registry = {"kernels": [{
            "id": "exact_kernel", "impl": {"function": "exact_kernel"}
        }]}
        arranged = [{
            "kernel": "exact_kernel", "op": "patch_projection_image",
            "template_op_id": "exact", "section": "header", "layer": -1,
        }]
        build_ir_v8._attach_semantic_checkpoints(
            template, arranged, registry, {"frontend_policy": "exact"}
        )
        self.assertEqual(
            arranged[0]["semantic_checkpoints"][0]["id"],
            "vision.frontend.output",
        )

    def test_qwen3vl_invalid_vision_mrope_sections_fail_lowering(self) -> None:
        manifest = _make_qwen3vl_manifest()
        manifest["config"]["vision_mrope_n_dims"] = 4
        manifest["config"]["vision_mrope_sections"] = [2, 2, 2, 2]
        with tempfile.TemporaryDirectory(prefix="v8_qwen3vl_invalid_mrope_") as td:
            td_path = Path(td)
            manifest_path = td_path / "weights_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "vision M-RoPE sections exceed available frequency pairs",
            ):
                build_ir_v8.main(
                    [
                        "--manifest", str(manifest_path),
                        "--mode", "prefill",
                        "--output", str(td_path / "ir1.json"),
                        "--layout-output", str(td_path / "layout.json"),
                        "--lowered-output", str(td_path / "lowered.json"),
                        "--call-output", str(td_path / "call.json"),
                    ]
                )

    def test_qwen3vl_mmproj_position_grid_size_uses_square_side(self) -> None:
        class FakeTensor:
            dims = [2304, 1152]
            ne1 = 2304

        self.assertEqual(convert_gguf_to_bump_v8._derive_position_grid_size(FakeTensor()), 48)

    def test_qwen3vl_mmproj_position_grid_size_keeps_non_square_rows(self) -> None:
        class FakeTensor:
            dims = [2305, 1152]
            ne1 = 2305

        self.assertEqual(convert_gguf_to_bump_v8._derive_position_grid_size(FakeTensor()), 2305)


    def test_qwen3vl_geometry_override_ignores_stale_cached_min_pixels(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_qwen3vl_geometry_") as td:
            image_path = Path(td) / "ocr.ppm"
            image_path.write_bytes(b"P6\n512 256\n255\n" + bytes([255, 255, 255]) * 512 * 256)
            cfg = {
                "patch_size": 16,
                "spatial_merge_size": 2,
                "image_min_pixels": 1024 * 16 * 16 * 2 * 2,
                "image_max_pixels": 4096 * 16 * 16 * 2 * 2,
            }
            out = run_multimodal_bridge_v8._qwen3vl_geometry_overrides(
                cfg,
                image_path,
                image_min_tokens=128,
            )
            self.assertEqual(out["image_width"], 512)
            self.assertEqual(out["image_height"], 256)
            self.assertEqual(out["merged_grid_x"], 16)
            self.assertEqual(out["merged_grid_y"], 8)
            self.assertEqual(out["vision_merged_tokens"], 128)

    def test_qwen3vl_decoder_declares_bridge_generation_contract(self) -> None:
        doc = build_ir_v8._load_builtin_template_doc("qwen3vl")
        bridge = doc["contract"]["multimodal_bridge"]
        self.assertEqual(bridge["prefix_policy"], "mixed_visual_text_prefill")
        self.assertEqual(bridge["generation_policy"], "incremental_decode_after_prefill")
        self.assertEqual(bridge["position_policy"], "mrope_2d")
        self.assertEqual(bridge["cache_policy"], "persistent_decoder_kv")
        self.assertEqual(bridge["runtime_policy"], "decode_staged")
        self.assertEqual(
            bridge["prefill_schedules"]["segmented_append"],
            {
                "segments": ["text_before", "visual", "text_after"],
                "cache_transition": "append_preserve",
                "position_transition": "segment_defined",
                "position_transform": {
                    "kernel_id": "mrope_qk_imrope_positions",
                    "contract_id": "text_imrope_positions_fp32_input_fp32_compute_fp32_output",
                },
            },
        )
        self.assertEqual(
            bridge["prefill_schedules"]["unified_mixed"],
            {
                "segments": ["text_before", "visual", "text_after"],
                "cache_transition": "single_pass",
                "position_transition": "explicit_full_sequence",
                "deepstack_injection": {
                    "kernel_id": "ck_residual_add_token_major_bf16_storage",
                    "contract_id": "residual_add_bf16_input_fp32_add_bf16_output",
                    "target": "visual_rows_after_decoder_layer",
                    "layers_from_config": "num_deepstack_layers",
                },
                "position_transform": {
                    "kernel_id": "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
                    "contract_id": "text_imrope_positions_bf16_input_pytorch_bf16_compute_bf16_output",
                },
            },
        )
        self.assertEqual(
            run_multimodal_bridge_v8._bridge_runtime_from_policy(bridge),
            "decode-staged",
        )
        self.assertEqual(
            run_multimodal_bridge_v8._bridge_generation_mode_from_policy(bridge),
            "incremental-decode",
        )

        bf16_manifest = {
            "config": {"model": "qwen3vl", "decode_kv_cache_dtype": "bf16"},
            "template": copy.deepcopy(doc),
        }
        hydrated_bf16 = build_ir_v8._hydrate_manifest_template(bf16_manifest)
        resolved_bf16 = hydrated_bf16["config"]["multimodal_bridge_contract"]
        self.assertEqual(
            resolved_bf16["prefill_schedule"]["position_transform"]["resolved_function"],
            "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
        )
        self.assertEqual(
            resolved_bf16["prefill_schedule"]["deepstack_injection"]["resolved_function"],
            "ck_residual_add_token_major_bf16_storage",
        )
        self.assertEqual(resolved_bf16["prefill_batching"], "unified_mixed")
        self.assertEqual(resolved_bf16["prefill_schedule"]["cache_transition"], "single_pass")
        bf16_attention = build_ir_v8._resolve_manifest_numerical_contracts(
            hydrated_bf16,
            "prefill",
        )
        self.assertEqual(len(bf16_attention), 1)
        self.assertEqual(
            bf16_attention[0]["kernel"]["function"],
            "attention_forward_causal_head_major_gqa_prefill_full_bf16cache_pytorch_contract",
        )
        self.assertEqual(
            bf16_attention[0]["requirements"]["execution.attention_schedule"],
            "full_matrix_causal",
        )

        fp16_manifest = {
            "config": {"model": "qwen3vl", "decode_kv_cache_dtype": "fp16"},
            "template": copy.deepcopy(doc),
        }
        hydrated_fp16 = build_ir_v8._hydrate_manifest_template(fp16_manifest)
        resolved_fp16 = hydrated_fp16["config"]["multimodal_bridge_contract"]
        self.assertEqual(resolved_fp16["prefill_batching"], "segmented_append")
        self.assertEqual(resolved_fp16["prefill_schedule"]["cache_transition"], "append_preserve")
        fp16_attention = build_ir_v8._resolve_manifest_numerical_contracts(
            hydrated_fp16,
            "prefill",
        )
        self.assertEqual(len(fp16_attention), 1)
        self.assertEqual(
            fp16_attention[0]["kernel"]["function"],
            "attention_forward_causal_head_major_gqa_prefill_append_f16cache_auto_workspace",
        )
        self.assertEqual(
            fp16_attention[0]["kernel"]["id"],
            "attention_forward_causal_head_major_gqa_prefill_append_f16cache_flash_auto_qtile64",
        )
        self.assertNotIn(
            "execution.attention_schedule",
            fp16_attention[0]["requirements"],
        )

    def test_unknown_bridge_contract_defaults_to_safe_replay(self) -> None:
        self.assertEqual(
            run_multimodal_bridge_v8._bridge_runtime_from_policy({}, fallback="prefill"),
            "prefill",
        )
        self.assertEqual(
            run_multimodal_bridge_v8._bridge_generation_mode_from_policy({}, fallback="mixed-replay"),
            "mixed-replay",
        )

    def test_cached_manifest_refreshes_circuit_without_rewriting_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scripts = root / "scripts"
            circuits = root / "circuits"
            scripts.mkdir()
            circuits.mkdir()
            current = {"name": "fixture", "version": 2, "contract": {"mode": "current"}}
            (circuits / "fixture.json").write_text(json.dumps(current), encoding="utf-8")
            manifest_path = root / "weights_manifest.json"
            manifest = {"template": {"name": "fixture", "version": 1}}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch.object(run_multimodal_bridge_v8, "SCRIPT_DIR", scripts):
                refreshed = run_multimodal_bridge_v8._refresh_manifest_circuit_snapshot(
                    manifest,
                    manifest_path,
                )

            self.assertEqual(refreshed["template"], current)
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["template"],
                current,
            )

    def test_segmented_prefill_schedule_and_attention_requirement_are_atomic(self) -> None:
        circuit = build_ir_v8._load_builtin_template_doc("qwen3vl")
        self.assertIsNotNone(circuit)
        build_ir_v8._validate_segmented_prefill_contract(circuit, source="test:qwen3vl")

        missing_attention = copy.deepcopy(circuit)
        missing_attention["required_contracts"]["decoder.attention.fp16_cache"]["phases"][
            "prefill"
        ]["requires"]["execution.prefill_batching"] = "unknown_schedule"
        with self.assertRaisesRegex(RuntimeError, "schedules and attention providers disagree"):
            build_ir_v8._validate_segmented_prefill_contract(
                missing_attention,
                source="test:missing-attention",
            )

        malformed_schedule = copy.deepcopy(circuit)
        malformed_schedule["contract"]["multimodal_bridge"]["prefill_schedules"][
            "segmented_append"
        ]["segments"] = ["visual", "text_after"]
        with self.assertRaisesRegex(RuntimeError, "unsupported mixed-prefill schedule"):
            build_ir_v8._validate_segmented_prefill_contract(
                malformed_schedule,
                source="test:malformed-schedule",
            )

    def test_builtin_template_declares_qwen3vl_vision_contract(self) -> None:
        doc = build_ir_v8._load_builtin_template_doc("qwen3_vl_vision")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["version"], 4)
        self.assertEqual(doc["sequence"], ["vision_encoder"])
        self.assertIn("vision_position_contract", doc["contract"])
        self.assertEqual(doc["contract"]["attention_contract"]["rope_layout"], "multi_section_2d")
        self.assertEqual(doc["block_types"]["vision_encoder"]["header"][-1]["op"], "position_ids_2d")
        header_by_id = {
            item["id"]: item
            for item in doc["block_types"]["vision_encoder"]["header"]
        }
        self.assertEqual(
            header_by_id["patch_sum"]["params"]["merge_size_from_config"],
            "spatial_merge_size",
        )
        self.assertEqual(
            header_by_id["position_embeddings"]["params"]["merge_size_from_config"],
            "spatial_merge_size",
        )
        self.assertEqual(
            header_by_id["vision_position_ids"]["params"]["merge_size_from_config"],
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
        self.assertNotIn("dtype", branch["collect"])

    def test_deepstack_collect_view_must_name_a_declared_output(self) -> None:
        manifest = _make_qwen3vl_manifest()
        branch = manifest["template"]["block_types"]["vision_encoder"]["branches"][0]
        fc2 = next(item for item in branch["producer"]["ops"] if item["id"] == "fc2")
        fc2["output_view"]["output"] = "undeclared"
        with self.assertRaisesRegex(RuntimeError, "not a declared graph output"):
            build_ir_v8.build_ir1_direct(
                manifest,
                ROOT / "tests" / "qwen3_vl_vision_manifest.synthetic.json",
                mode="prefill",
            )

    def test_bf16_storage_contracts_select_exact_generated_kernels(self) -> None:
        manifest = _make_qwen3vl_manifest()
        manifest["config"].update({
            "vision_position_storage_boundary": "bf16",
            "vision_layernorm_storage_boundary": "bf16",
            "vision_projection_storage_boundary": "bf16",
            "vision_attention_storage_boundary": "bf16",
            "vision_residual_storage_boundary": "bf16",
            "vision_activation_storage_boundary": "bf16",
        })
        for entry in manifest["entries"]:
            if ".attn_qkv." in entry["name"] or ".attn_out." in entry["name"]:
                if entry["name"].endswith(".weight"):
                    entry["dtype"] = "bf16"

        legacy = build_ir_v8._resolve_manifest_numerical_contracts(manifest, "prefill")
        execution = build_ir_v8._resolve_manifest_execution_contracts(manifest, "prefill")
        self.assertEqual(legacy, [])
        selected = {plan["operation"]: plan["kernel"]["function"] for plan in execution}
        self.assertEqual(selected["vision.layer.layernorm"], "layernorm_naive_serial_bf16_storage")
        self.assertEqual(selected["vision.layer.qkv_projection"], "gemm_nt_bf16_native_bf16_storage")
        self.assertEqual(selected["vision.layer.mlp_projection"], "gemm_nt_bf16_native_bf16_storage")
        self.assertEqual(selected["vision.layer.mlp_activation"], "gelu_pytorch_tanh_bf16_storage")
        self.assertEqual(
            selected["vision.layer.attention"],
            "attention_forward_full_head_major_gqa_pytorch_cpu_flash_bf16_storage",
        )
        self.assertEqual(selected["vision.layer.out_projection"], "gemm_nt_bf16_native_bf16_storage")
        self.assertEqual(
            selected["vision.layer.residual"],
            "ck_residual_add_token_major_bf16_storage",
        )

        ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "qwen3_vl_vision_manifest.synthetic.json",
            mode="prefill",
        )
        kernels = [(op["op"], op["kernel"]) for op in ops]
        self.assertIn(("layernorm", "layernorm_bf16_storage"), kernels)
        self.assertIn(("qkv_packed_proj", "gemm_nt_bf16_native_bf16_storage"), kernels)
        self.assertIn(
            ("attn", "attention_forward_full_head_major_gqa_pytorch_cpu_flash_bf16_storage"),
            kernels,
        )
        self.assertIn(("out_proj", "gemm_nt_bf16_native_bf16_storage"), kernels)
        self.assertIn(("mlp_up", "gemm_nt_bf16_native_bf16_storage"), kernels)
        self.assertIn(("mlp_down", "gemm_nt_bf16_native_bf16_storage"), kernels)
        self.assertIn(("gelu", "gelu_pytorch_tanh_bf16_storage"), kernels)
        self.assertIn(
            ("residual_add", "ck_residual_add_token_major_bf16_storage"),
            kernels,
        )

        binding = build_ir_v8.load_kernel_call_abis()[
            "attention_forward_full_head_major_gqa_pytorch_cpu_flash_bf16_storage"
        ]["call_abi"]
        self.assertEqual(
            [param["name"] for param in binding["params"]],
            [
                "q", "k", "v", "output", "num_heads", "num_kv_heads",
                "num_tokens", "head_dim", "aligned_head_dim", "kv_stride_tokens",
            ],
        )

        registry = build_ir_v8.load_kernel_registry()
        lowered_ir1 = build_ir_v8.generate_ir_lower_1(
            ops, registry, manifest, "prefill"
        )
        layout = build_ir_v8.generate_memory_layout(
            lowered_ir1,
            manifest,
            registry,
            mode="prefill",
            context_len=manifest["config"]["context_length"],
        )
        lowered_ir2 = build_ir_v8.generate_ir_lower_2(
            lowered_ir1, layout, manifest, registry, mode="prefill"
        )
        self.assertNotIn(
            "transpose_attn_out_to_token_major",
            [op.get("op") for op in lowered_ir2["operations"]],
        )
        call_ir = build_ir_v8.generate_ir_lower_3(lowered_ir2, mode="prefill")
        attention_call = next(
            op for op in call_ir["operations"] if op.get("op") == "attn"
        )
        self.assertEqual(
            attention_call["resolved_physical_execution"]["function"],
            "attention_forward_full_head_major_gqa_pytorch_cpu_flash_bf16_storage_token_output",
        )

    def test_default_attention_contract_retains_legacy_owner(self) -> None:
        manifest = _make_qwen3vl_manifest()
        manifest["config"].pop("vision_attention_storage_boundary", None)
        legacy = build_ir_v8._resolve_manifest_numerical_contracts(manifest, "prefill")
        execution = build_ir_v8._resolve_manifest_execution_contracts(manifest, "prefill")
        self.assertEqual(len(legacy), 1)
        self.assertEqual(
            legacy[0]["kernel"]["function"],
            "attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided",
        )
        self.assertNotIn("vision.layer.attention", {plan["operation"] for plan in execution})

    def test_template_dtype_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "declares dtype/quant policy"):
            build_ir_v8._raise_on_forbidden_template_metadata({"dtype": "fp16"}, source="synthetic")

    def test_template_activation_policy_flags_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "flags.activation_preference_by_op"):
            build_ir_v8._raise_on_forbidden_template_metadata(
                {"flags": {"activation_preference_by_op": {"mlp_down": "fp32"}}},
                source="synthetic",
            )

    def test_template_q8_contract_policy_flags_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "flags.prefer_q8_0_contract"):
            build_ir_v8._raise_on_forbidden_template_metadata(
                {"flags": {"prefer_q8_0_contract": True}},
                source="synthetic",
            )
        with self.assertRaisesRegex(RuntimeError, "flags.prefer_fp32_logits"):
            build_ir_v8._raise_on_forbidden_template_metadata(
                {"flags": {"prefer_fp32_logits": True}},
                source="synthetic",
            )

    def test_hydrate_manifest_rejects_embedded_template_policy(self) -> None:
        manifest = {
            "config": {"model": "qwen3vl", "arch": "qwen3vl"},
            "template": {
                "name": "qwen3vl",
                "flags": {"prefer_fp32_mlp_matmuls": True},
            },
        }
        with self.assertRaisesRegex(RuntimeError, "flags.prefer_fp32_mlp_matmuls"):
            build_ir_v8._hydrate_manifest_template(manifest)

    def test_circuit_runtime_defaults_own_quant_policy(self) -> None:
        qwen2_cfg = build_ir_v8._apply_circuit_runtime_defaults(
            {}, build_ir_v8._load_builtin_template_doc("qwen2"), source="qwen2"
        )
        self.assertNotIn("prefer_fp32_mlp_matmuls", qwen2_cfg)

        gemma_cfg = build_ir_v8._apply_circuit_runtime_defaults(
            {}, build_ir_v8._load_builtin_template_doc("gemma3"), source="gemma3"
        )
        self.assertTrue(gemma_cfg["prefer_q8_0_contract"])
        self.assertTrue(gemma_cfg["prefer_fp32_logits"])

        vision_cfg = build_ir_v8._apply_circuit_runtime_defaults(
            {}, build_ir_v8._load_builtin_template_doc("qwen3_vl_vision"), source="vision"
        )
        self.assertTrue(vision_cfg["prefer_q8_0_contract"])
        self.assertEqual(
            vision_cfg["q8_0_contract_ops"],
            ["projector_fc2", "branch_fc1", "branch_fc2"],
        )
        self.assertEqual(
            vision_cfg["activation_preference_by_op"],
            {
                "mlp_down": "fp32",
                "out_proj": "q8_0",
                "branch_fc1": "fp32",
                "branch_fc2": "fp32",
            },
        )

    def test_circuit_runtime_defaults_are_identity_invariant(self) -> None:
        defaults = {"prefer_q8_0_contract": True, "activation_preference_by_op": {"mlp_down": "fp32"}}
        alpha = {"name": "model_alpha", "contract": {"runtime_defaults": defaults}}
        beta = {"name": "model_beta", "contract": {"runtime_defaults": defaults}}
        self.assertEqual(
            build_ir_v8._apply_circuit_runtime_defaults({}, alpha, source="alpha"),
            build_ir_v8._apply_circuit_runtime_defaults({}, beta, source="beta"),
        )

    def test_circuit_runtime_defaults_reject_unknown_policy(self) -> None:
        circuit = {"contract": {"runtime_defaults": {"model_specific_fast_path": True}}}
        with self.assertRaisesRegex(RuntimeError, "HARD CIRCUIT DEFAULT FAULT"):
            build_ir_v8._apply_circuit_runtime_defaults({}, circuit, source="invalid")

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
        self.assertEqual(branch["collect_contract"]["dtype"], "fp32")
        self.assertEqual(
            branch["producer_ops"],
            ["branch_spatial_merge", "branch_layernorm", "branch_fc1", "branch_gelu", "branch_fc2"],
        )
        self.assertEqual(branch["stitches"][0]["op"], "branch_concat")

    def test_zero_deepstack_disables_branch_and_dependent_stitch(self) -> None:
        manifest = _make_qwen3vl_manifest()
        manifest["config"]["deepstack_layer_indices"] = []
        manifest["config"]["num_deepstack_layers"] = 0
        plan = build_ir_v8.build_template_branch_plan(manifest)
        self.assertEqual(plan["blocks"][0]["branches"], [])
        block = manifest["template"]["block_types"]["vision_encoder"]
        footer = build_ir_v8._normalize_block_footer_items(block, manifest["config"])
        self.assertNotIn("branch_concat", [item["op"] for item in footer])
        self.assertEqual(
            build_ir_v8._resolve_activation_extent(
                {"max": [1, {"mul": [4096, {"config": "num_deepstack_layers"}]}]},
                manifest["config"],
                "test.branch_collect",
            ),
            1,
        )

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
        self.assertNotIn("branch_collect_offset_bytes", by_op["branch_fc2"][0]["params"])
        self.assertEqual(
            by_op["branch_fc2"][0]["output_view"],
            {
                "kind": "collect_slice",
                "output": "y",
                "offset_bytes": 0,
                "size_bytes": 576 * 4096 * 4,
                "index": 0,
            },
        )
        self.assertEqual(
            by_op["branch_fc2"][0]["graph_slots"],
            {"inputs": {"x": "branch_mlp"}, "outputs": {"y": "branch_collect"}},
        )
        self.assertEqual(by_op["branch_concat"][0]["params"]["main_dim_from_config"], "projector_out_dim")
        self.assertEqual(by_op["branch_concat"][0]["params"]["branch_slice_dim_from_config"], "projector_out_dim")
        self.assertEqual(by_op["branch_concat"][0]["params"]["num_branch_slices_from_config"], "num_deepstack_layers")

    def test_qwen3vl_codegen_smoke_emits_c(self) -> None:
        manifest = _make_qwen3vl_manifest()
        manifest["config"]["position_interpolation_policy"] = "align_corners_bilinear"
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
            pos_embed = next(op for op in call_ops if op.get("op") == "position_embeddings")
            pos_grid = next(arg for arg in pos_embed.get("args", []) if arg.get("name") == "source_grid_size")
            self.assertEqual(pos_grid.get("expr"), "48")
            self.assertEqual(
                pos_embed.get("function"),
                "position_embeddings_add_tiled_2d_align_corners",
            )
            split_qkv = next(op for op in call_ops if op.get("op") == "split_qkv_packed")
            self.assertEqual(split_qkv.get("function"), "split_qkv_packed_head_major_forward")
            qkv_packed_proj = next(op for op in call_ops if op.get("op") == "qkv_packed_proj")
            self.assertEqual(qkv_packed_proj.get("function"), "gemm_nt_q8_0_q8_0_contract")
            self.assertNotIn("quantize_input_0", [op.get("op") for op in call_ops])
            attn = next(op for op in call_ops if op.get("op") == "attn")
            self.assertEqual(
                attn.get("function"),
                "attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided",
            )
            attn_idx = next(i for i, op in enumerate(call_ops) if op.get("op") == "attn")
            transpose_idx = next(i for i, op in enumerate(call_ops) if op.get("op") == "transpose_attn_out_to_token_major")
            out_proj_idx = next(i for i, op in enumerate(call_ops) if op.get("op") == "out_proj")
            self.assertLess(attn_idx, transpose_idx)
            self.assertLess(transpose_idx, out_proj_idx)
            out_proj = next(op for op in call_ops if op.get("op") == "out_proj")
            self.assertEqual(out_proj.get("function"), "gemm_nt_q8_0_q8_0_contract")
            self.assertNotIn("quantize_out_proj_input", [op.get("op") for op in call_ops])
            self.assertNotIn("kv_cache_batch_copy", [op.get("op") for op in call_ops])
            projector_fc1 = next(op for op in call_ops if op.get("op") == "projector_fc1")
            self.assertEqual(projector_fc1.get("function"), "gemm_nt_q8_0_q8_0_contract")
            projector_fc1_bias = next(arg for arg in projector_fc1.get("args", []) if arg.get("name") == "bias")
            self.assertEqual(projector_fc1_bias.get("weight_ref"), "mm.0.bias")
            projector_fc2 = next(op for op in call_ops if op.get("op") == "projector_fc2")
            self.assertEqual(projector_fc2.get("function"), "gemm_nt_q8_0_q8_0_contract")
            projector_fc2_input = next(arg for arg in projector_fc2.get("args", []) if arg.get("name") == "A")
            self.assertEqual(projector_fc2_input.get("buffer_ref"), "mlp_scratch")
            projector_fc2_bias = next(arg for arg in projector_fc2.get("args", []) if arg.get("name") == "bias")
            self.assertEqual(projector_fc2_bias.get("weight_ref"), "mm.2.bias")
            branch_concat = next(op for op in call_ops if op.get("op") == "branch_concat")
            branch_concat_main = next(arg for arg in branch_concat.get("args", []) if arg.get("name") == "main_input")
            self.assertEqual(branch_concat_main.get("buffer_ref"), "vision_output")
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
            spatial_merge = next(op for op in call_ops if op.get("op") == "spatial_merge")
            spatial_merge_out = next(arg for arg in spatial_merge.get("args", []) if arg.get("name") == "output")
            self.assertEqual(spatial_merge_out.get("buffer_ref"), "embedded_input")
            self.assertNotIn("quantize_final_output", [op.get("op") for op in call_ops])
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
            self.assertIn("gemm_nt_q8_0_q8_0_contract", text)
            self.assertIn("add_stream_reorder_2d", text)
            self.assertIn("vision_bridge_output", text)
            self.assertIn("ck_strict_mtmd_clip_encode_planar_f32", text)
            self.assertIn("gelu_ggml_inplace", text)
            self.assertIn("layernorm_naive_serial_matched_precision", text)
            self.assertIn("ck_debug_export_hidden(model, 0, \"qkv_packed\"", text)
            self.assertIn("ck_debug_export_hidden(model, 0, \"q_proj\"", text)
            self.assertIn("ck_debug_export_hidden(model, 0, \"k_proj\"", text)
            self.assertIn("ck_debug_export_hidden(model, 0, \"v_proj\"", text)
            self.assertIn("ck_debug_export_hidden(model, 0, \"rope_q\"", text)
            self.assertIn("ck_debug_export_hidden(model, 0, \"rope_k\"", text)
            self.assertIn("ck_debug_export_hidden(model, 0, \"attn_out_head_major\"", text)
            self.assertIn("ck_debug_export_hidden(model, 0, \"out_proj\"", text)
            self.assertIn("ck_debug_export_hidden(model, 0, \"ln1\"", text)
            self.assertIn("ck_debug_export_hidden(model, 0, \"ffn_inp_normed\"", text)
            spatial_export = next(
                line for line in text.splitlines()
                if 'ck_debug_export_hidden(model, -1, "vision_spatial_merge"' in line
            )
            self.assertIn("(48) * (48) * (1152)", spatial_export)
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

    def test_deepstack_collect_slice_survives_call_ready_lowering(self) -> None:
        manifest = _make_qwen3vl_manifest()
        # Allocate a second slice so a nonzero test offset remains in bounds.
        manifest["config"]["num_deepstack_layers"] = 2
        ir1_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "qwen3_vl_vision_manifest.synthetic.json",
            mode="prefill",
        )
        branch_fc2 = next(op for op in ir1_ops if op.get("op") == "branch_fc2")
        branch_fc2["output_view"]["offset_bytes"] = 128

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
        lowered_ir2 = build_ir_v8.generate_ir_lower_2(
            lowered_ir1,
            layout,
            manifest,
            registry,
            mode="prefill",
        )
        lowered_fc2 = next(
            op for op in lowered_ir2["operations"] if op.get("op") == "branch_fc2"
        )
        collect_base = next(
            item["offset"]
            for item in layout["memory"]["activations"]["buffers"]
            if item["name"] == "branch_collect"
        )
        self.assertEqual(
            lowered_fc2["outputs"]["C"]["activation_offset"],
            collect_base + 128,
        )

        call_ir = build_ir_v8.generate_ir_lower_3(lowered_ir2, "prefill")
        call_fc2 = next(op for op in call_ir["operations"] if op.get("op") == "branch_fc2")
        output_arg = next(arg for arg in call_fc2["args"] if arg.get("name") == "C")
        self.assertIn("+ 128", output_arg["expr"])

    def test_reused_layout_cannot_replace_resolved_bridge_numerics(self) -> None:
        manifest = _make_qwen3vl_manifest()
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
        # Simulate --layout-input from an older compiler which knew the
        # schedule but not its exact positions-aware numerical provider.
        stale_bridge = {
            "prefill_batching": "unified_mixed",
            "prefill_schedule": {
                "segments": ["text_before", "visual", "text_after"],
                "cache_transition": "single_pass",
                "position_transition": "explicit_full_sequence",
            },
        }
        layout["config"]["multimodal_bridge_contract"] = stale_bridge
        manifest["config"]["multimodal_bridge_contract"] = {
            **copy.deepcopy(stale_bridge),
            "prefill_schedule": {
                **copy.deepcopy(stale_bridge["prefill_schedule"]),
                "position_transform": {
                    "kernel_id": "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
                    "contract_id": "text_imrope_positions_bf16_input_pytorch_bf16_compute_bf16_output",
                    "resolved_function": "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
                },
            },
        }

        lowered = build_ir_v8.generate_ir_lower_2(
            lowered_ir1,
            layout,
            manifest,
            registry,
            mode="prefill",
        )
        call_ir = build_ir_v8.generate_ir_lower_3(lowered, mode="prefill")
        for artifact in (lowered, call_ir):
            transform = artifact["config"]["multimodal_bridge_contract"][
                "prefill_schedule"
            ]["position_transform"]
            self.assertEqual(
                transform["kernel_id"],
                "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
            )
            self.assertEqual(
                transform["contract_id"],
                "text_imrope_positions_bf16_input_pytorch_bf16_compute_bf16_output",
            )
            self.assertEqual(
                transform["resolved_function"],
                "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
            )


if __name__ == "__main__":
    unittest.main()
