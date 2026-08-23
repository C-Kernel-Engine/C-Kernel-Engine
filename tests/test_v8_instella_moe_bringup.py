from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "convert_safetensors_to_bump_v8.py"
CIRCUIT = ROOT / "version" / "v8" / "circuits" / "instella_moe.json"


def _load_converter():
    scripts = str(SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("convert_instella_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


converter = _load_converter()


def _tiny_config() -> dict:
    return {
        "architectures": ["InstellaMoEForCausalLM"],
        "model_type": "deepseek_v3",
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "kv_lora_rank": 4,
        "q_lora_rank": None,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 2,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "routed_scaling_factor": 2.5,
        "gated_attention": True,
        "farskip": True,
        "rope_interleave": True,
        "rope_scaling": {
            "type": "yarn",
            "factor": 40,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
        "rope_theta": 8000000,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
    }


def _headers() -> dict:
    fake = Path("/tmp/instella-fixture.safetensors")
    rows: dict[str, object] = {}

    def add(name: str, shape: list[int], dtype: str = "BF16") -> None:
        rows[name] = converter.HeaderTensor(name, dtype, shape, fake)

    add("model.embed_tokens.weight", [32, 8])
    add("model.norm.weight", [8], "F32")
    add("lm_head.weight", [32, 8])
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        add(f"{prefix}.input_layernorm.weight", [8], "F32")
        add(f"{prefix}.post_attention_layernorm.weight", [8], "F32")
        add(f"{prefix}.self_attn.q_proj.weight", [8, 8])
        add(f"{prefix}.self_attn.kv_a_proj_with_mqa.weight", [6, 8])
        add(f"{prefix}.self_attn.kv_a_layernorm.weight", [4], "F32")
        add(f"{prefix}.self_attn.kv_b_proj.weight", [8, 4])
        add(f"{prefix}.self_attn.gate_proj.weight", [4, 8])
        add(f"{prefix}.self_attn.o_proj.weight", [8, 4])
    add("model.layers.0.mlp.gate_proj.weight", [16, 8])
    add("model.layers.0.mlp.up_proj.weight", [16, 8])
    add("model.layers.0.mlp.down_proj.weight", [8, 16])
    add("model.layers.1.mlp.gate.weight", [2, 8], "F32")
    add("model.layers.1.mlp.gate.e_score_correction_bias", [2], "F32")
    for expert in range(2):
        prefix = f"model.layers.1.mlp.experts.{expert}"
        add(f"{prefix}.gate_proj.weight", [4, 8])
        add(f"{prefix}.up_proj.weight", [4, 8])
        add(f"{prefix}.down_proj.weight", [8, 4])
    add("model.layers.1.mlp.shared_experts.gate_proj.weight", [8, 8])
    add("model.layers.1.mlp.shared_experts.up_proj.weight", [8, 8])
    add("model.layers.1.mlp.shared_experts.down_proj.weight", [8, 8])
    return rows


class InstellaMoEBringupTests(unittest.TestCase):
    def test_chat_contract_matches_checkpoint_role_markers(self) -> None:
        circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
        contract = circuit["contract"]["chat_contract"]

        self.assertEqual(contract["conversation_prefix"], "<｜begin▁of▁sentence｜>")
        self.assertEqual(contract["turn_prefix_by_role"]["user"], "<｜User｜>")
        self.assertEqual(
            contract["turn_prefix_by_role"]["assistant"],
            "<｜Assistant｜>",
        )
        self.assertEqual(
            contract["turn_suffix_by_role"]["assistant"],
            "<｜end▁of▁sentence｜>",
        )
        self.assertFalse(contract["raw_prompt_allowed"])

    def test_mla_latent_norm_selects_strided_bf16_storage_provider(self) -> None:
        circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
        self.assertEqual(
            circuit["kernels"]["kv_a_layernorm"],
            "rmsnorm_forward_strided_pytorch_bf16_storage",
        )

    def test_architecture_class_overrides_deepseek_model_type(self) -> None:
        self.assertEqual(converter._infer_arch(_tiny_config()), "instella_moe")

    def test_tensor_role_mapping_consumes_every_fixture_header(self) -> None:
        headers = _headers()
        refs = converter._refs_for_arch("instella_moe", _tiny_config(), headers)
        consumed = {source for ref in refs for source in ref.source_names}
        self.assertEqual(consumed, set(headers))
        by_name = {ref.ck_name: ref for ref in refs}
        self.assertIn("layer.0.mla_gate_proj", by_name)
        self.assertIn("layer.1.mla_gate_proj", by_name)
        self.assertIsNone(
            by_name["layer.0.mla_kv_b_proj"].dtype,
            "the converter must preserve the checkpoint's BF16 KV-B projection",
        )
        self.assertEqual(by_name["layer.1.moe_expert_gate"].shape, (2, 4, 8))
        self.assertEqual(by_name["layer.1.moe_shared_gate"].source_names, (
            "model.layers.1.mlp.shared_experts.gate_proj.weight",
        ))

    def test_runtime_config_preserves_farskip_and_interleaved_yarn(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text(
                json.dumps(_tiny_config()), encoding="utf-8"
            )
            config = converter._build_config(root, "instella_moe", None)
        self.assertEqual(config["model"], "instella_moe")
        self.assertEqual(
            config["layer_kinds"],
            ["mla_gated_farskip_dense_mlp", "mla_gated_farskip_moe_first"],
        )
        self.assertTrue(config["gated_attention"])
        self.assertTrue(config["farskip"])
        self.assertTrue(config["rope_interleave"])
        self.assertEqual(config["rope_layout"], "partial_interleaved_yarn")
        self.assertEqual(config["rope_scaling_type"], "yarn")
        self.assertEqual(config["rope_scaling_factor"], 40.0)
        self.assertEqual(config["rope_original_context_length"], 4096)
        self.assertEqual(config["rope_beta_fast"], 32.0)
        self.assertEqual(config["rope_beta_slow"], 1.0)
        self.assertEqual(config["rope_mscale"], 1.0)
        self.assertEqual(config["rope_mscale_all_dim"], 1.0)
        expected_mscale = 0.1 * math.log(40.0) + 1.0
        self.assertAlmostEqual(
            config["attention_scale"],
            (1.0 / math.sqrt(4.0)) * expected_mscale * expected_mscale,
        )
        self.assertEqual(config["moe_shared_intermediate_size"], 8)

    def test_synthetic_model_reaches_call_ready_ir_in_prefill_and_decode(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover - development dependency guard
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "checkpoint"
            out = root / "runtime"
            checkpoint.mkdir()
            out.mkdir()
            (checkpoint / "config.json").write_text(
                json.dumps(_tiny_config()), encoding="utf-8"
            )
            (checkpoint / "tokenizer.json").write_text(
                json.dumps({
                    "version": "1.0",
                    "model": {
                        "type": "BPE",
                        "unk_token": "<unk>",
                        "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2},
                        "merges": [],
                    },
                    "added_tokens": [],
                }),
                encoding="utf-8",
            )
            tensors = {}
            for name, header in _headers().items():
                dtype = torch.bfloat16 if header.dtype == "BF16" else torch.float32
                tensors[name] = torch.zeros(tuple(header.shape), dtype=dtype)
            save_file(tensors, checkpoint / "model.safetensors")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--checkpoint", str(checkpoint),
                    "--output", str(out / "weights.bump"),
                    "--config-out", str(out / "config.json"),
                    "--manifest-out", str(out / "weights_manifest.json"),
                    "--arch", "auto",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            builder = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
            for mode in ("prefill", "decode"):
                call_path = out / f"call_{mode}.json"
                builder_args = [
                    sys.executable,
                    str(builder),
                    "--manifest", str(out / "weights_manifest.json"),
                    "--mode", mode,
                    "--context-len", "8",
                    "--layout-mode", "packed",
                    "--output", str(out / f"ir1_{mode}.json"),
                    "--layout-output", str(out / f"layout_{mode}.json"),
                    "--lowered-output", str(out / f"lowered_{mode}.json"),
                    "--call-output", str(call_path),
                ]
                if mode == "decode":
                    builder_args.extend(["--init-output", str(out / "init.json")])
                build = subprocess.run(
                    builder_args,
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    build.returncode,
                    0,
                    build.stdout + build.stderr,
                )
                call_ir = json.loads(call_path.read_text(encoding="utf-8"))
                self.assertFalse(call_ir.get("errors"), call_ir.get("errors"))
                graph_ir = json.loads(
                    (out / f"ir1_{mode}.json").read_text(encoding="utf-8")
                )
                joined = [
                    op
                    for op in graph_ir.get("operations", graph_ir.get("ops", []))
                    if op.get("interface_validation", {}).get("status") == "validated"
                ]
                self.assertTrue(joined)
                self.assertTrue(
                    any(op.get("op") == "attn_gate_sigmoid_mul" for op in joined)
                )
                self.assertTrue(any(op.get("op") == "residual_add" for op in joined))
                ops = call_ir.get("operations", call_ir.get("ops", []))
                final_norm = next(op for op in ops if op.get("op") == "final_rmsnorm")
                self.assertEqual(
                    final_norm.get("call_abi", {}).get("kernel_id"),
                    "rmsnorm_forward_pytorch_bf16_storage",
                )
                kv_decompress = next(
                    op for op in ops
                    if op.get("layer") == 0 and op.get("op") == "kv_lora_decompress"
                )
                self.assertEqual(
                    kv_decompress.get("call_abi", {}).get("kernel_id"),
                    "deepseek_mla_kv_decompress_bf16",
                )
                self.assertEqual(
                    kv_decompress.get("call_abi", {}).get("owner"),
                    "kernel_map",
                )
                layer_zero = [op for op in ops if op.get("layer") == 0]
                self.assertEqual(
                    [op.get("op") for op in layer_zero].count("residual_save"),
                    1,
                    "explicit FarSkip dataflow must not be overwritten by a legacy residual copy",
                )
                layer_zero_norms = [
                    op for op in layer_zero if op.get("op") == "block_rmsnorm"
                ]
                self.assertEqual(len(layer_zero_norms), 2)
                ffn_input = next(
                    arg for arg in layer_zero_norms[1].get("args", [])
                    if arg.get("source") in {"activation:input", "activation:x"}
                )
                self.assertEqual(ffn_input.get("buffer_ref"), "residual")
                residual_adds = [
                    op for op in layer_zero if op.get("op") == "residual_add"
                ]
                self.assertEqual(len(residual_adds), 2)
                final_inputs = {
                    arg.get("name"): arg.get("buffer_ref")
                    for arg in residual_adds[1].get("args", [])
                    if arg.get("name") in {"a", "b"}
                }
                self.assertEqual(
                    final_inputs,
                    {"a": "embedded_input", "b": "layer_output"},
                )
                normalized_projection_ops = {
                    "q_proj",
                    "attention_gate_projection",
                    "mlp_gate_up",
                }
                bf16_storage_projections = {
                    "q_proj",
                    "kv_a_proj",
                    "attention_gate_projection",
                    "out_proj",
                    "mlp_gate_up",
                    "mlp_down",
                }
                for op in ops:
                    if op.get("op") not in bf16_storage_projections:
                        continue
                    self.assertEqual(
                        op.get("call_abi", {}).get("kernel_id"),
                        "gemm_nt_bf16_bf16_storage",
                        op,
                    )
                for op in ops:
                    if op.get("layer") != 0 or op.get("op") not in normalized_projection_ops:
                        continue
                    activation_args = [
                        arg for arg in op.get("args", [])
                        if arg.get("source") in {"activation:a", "activation:x"}
                    ]
                    self.assertEqual(len(activation_args), 1, op)
                    self.assertEqual(
                        activation_args[0].get("buffer_ref"),
                        "layer_input",
                        op,
                    )
                gate = next(
                    op for op in ops
                    if op.get("layer") == 0
                    and op.get("op") == "attention_gate_projection"
                )
                gate_outputs = [
                    arg for arg in gate.get("args", [])
                    if arg.get("source") in {"output:c", "output:y"}
                ]
                self.assertEqual(len(gate_outputs), 1, gate)
                self.assertEqual(gate_outputs[0].get("buffer_ref"), "attn_gate")
                router = next(
                    op for op in ops
                    if op.get("layer") == 1 and op.get("op") == "moe_router"
                )
                self.assertEqual(router.get("call_abi", {}).get("owner"), "kernel_map")
                router_buffers = {
                    arg.get("name"): arg.get("buffer_ref")
                    for arg in router.get("args", [])
                    if arg.get("name") in {"A", "C"}
                }
                self.assertEqual(
                    router_buffers,
                    {"A": "layer_input", "C": "mlp_scratch"},
                    "router logits must not overwrite their normalized activation input",
                )
                farskip = [op for op in ops if op.get("op") == "farskip_routed_shared_combine"]
                self.assertEqual(len(farskip), 1)
                self.assertEqual(farskip[0].get("call_abi", {}).get("owner"), "kernel_map")
                farskip_buffers = {
                    arg.get("name"): arg.get("buffer_ref")
                    for arg in farskip[0].get("args", [])
                    if arg.get("name") in {
                        "hidden", "routed", "post_attn_residual",
                        "main_output", "routed_free_output",
                    }
                }
                self.assertEqual(
                    farskip_buffers,
                    {
                        "hidden": "layer_input",
                        "routed": "mlp_scratch",
                        "post_attn_residual": "embedded_input",
                        "main_output": "embedded_input",
                        "routed_free_output": "layer_output",
                    },
                )

            init_call = json.loads((out / "init_call.json").read_text(encoding="utf-8"))
            yarn = next(
                op for op in init_call["operations"] if op["op"] == "yarn_rope_init"
            )
            self.assertEqual(
                yarn["function"], "yarn_rope_cache_contiguous_positions_f32"
            )
            self.assertEqual(yarn["errors"], [])
            self.assertEqual(
                [arg["source"] for arg in yarn["args"]],
                [
                    "output:cos_cache",
                    "output:sin_cache",
                    "dim:T",
                    "dim:Dr",
                    "param:freq_base",
                    "param:factor",
                    "param:original_context",
                    "param:beta_fast",
                    "param:beta_slow",
                    "param:mscale",
                    "param:mscale_all_dim",
                ],
            )
            # This minimal fixture does not materialize production vocabulary
            # weight macros; keep generated-C coverage focused on the map-owned
            # arithmetic init call certified by this test.
            init_call["operations"] = [yarn]
            (out / "init_call.json").write_text(
                json.dumps(init_call, indent=2), encoding="utf-8"
            )

            generated_c = out / "model_v8.c"
            codegen = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "version" / "v8" / "scripts" / "codegen_v8.py"),
                    "--ir", str(out / "call_decode.json"),
                    "--layout", str(out / "layout_decode.json"),
                    "--prefill", str(out / "call_prefill.json"),
                    "--prefill-layout", str(out / "layout_prefill.json"),
                    "--output", str(generated_c),
                    "--strict-contracts",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(codegen.returncode, 0, codegen.stdout + codegen.stderr)
            source = generated_c.read_text(encoding="utf-8")
            self.assertIn("farskip_swiglu_shared_combine_bf16", source)
            self.assertIn("yarn_rope_cache_contiguous_positions_f32(", source)
            syntax = subprocess.run(
                [
                    "cc", "-std=c11", "-fopenmp", "-fsyntax-only",
                    "-Iinclude", "-Iversion/v8/src", str(generated_c),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(syntax.returncode, 0, syntax.stdout + syntax.stderr)

    def test_codegen_fails_closed_without_resolved_yarn_init(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "codegen_v8",
            ROOT / "version" / "v8" / "scripts" / "codegen_v8.py",
        )
        assert spec is not None and spec.loader is not None
        codegen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codegen)
        layout = {
            "config": {
                "_template_uses_rope": True,
                "rope_theta": 8_000_000.0,
                "rotary_dim": 32,
                "rope_scaling_type": "yarn",
                "rope_layout": "partial_interleaved_yarn",
            },
            "memory": {
                "activations": {"buffers": [{"name": "rope_cache"}]},
            },
        }
        with self.assertRaisesRegex(RuntimeError, "resolved init_call.json provider"):
            codegen._inject_missing_rope_init(
                "static int do_init(void) {\n    /* No pre-weights init ops */\n}\n",
                layout,
                None,
            )


if __name__ == "__main__":
    unittest.main()
