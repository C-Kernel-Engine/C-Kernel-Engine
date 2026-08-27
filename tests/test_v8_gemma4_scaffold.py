import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str((REPO_ROOT / "version" / "v8" / "scripts").resolve()))

from scripts.chat_contract import build_chat_contract, load_template_chat_contract  # type: ignore
from convert_gguf_to_bump_v8 import (  # type: ignore
    TensorInfo,
    build_gemma4_attention_plan,
    classify_layer_contract,
    describe_layer_contract,
)
from build_ir_v8 import (  # type: ignore
    _apply_circuit_runtime_defaults,
    _load_builtin_template_doc,
    apply_layer_attention_dims,
    _validate_lowered_activation_memory,
    build_activation_specs,
    find_kernel,
    load_kernel_registry,
)


def _write_tiny_bpe_tokenizer(checkpoint: Path, vocab_size: int) -> None:
    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "Hello": 3,
        "world": 4,
        "!": 5,
        " test": 6,
        " code": 7,
    }
    for idx in range(len(vocab), vocab_size):
        vocab[f"<tok_{idx}>"] = idx

    (checkpoint / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "unk_token": "<unk>",
                    "vocab": vocab,
                    "merges": ["Hello world", " test  code"],
                },
                "added_tokens": [
                    {"id": 0, "content": "<unk>"},
                    {"id": 1, "content": "<s>"},
                    {"id": 2, "content": "</s>"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (checkpoint / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": {"content": "<s>"},
                "eos_token": {"content": "</s>"},
                "unk_token": {"content": "<unk>"},
                "add_bos_token": True,
                "add_eos_token": False,
                "added_tokens_decoder": {
                    "0": {"content": "<unk>"},
                    "1": {"content": "<s>"},
                    "2": {"content": "</s>"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _tensor(name: str, dims: tuple[int, ...]) -> TensorInfo:
    return TensorInfo(name=name, dims=dims, ggml_type=12, offset=0)


class V8Gemma4ScaffoldTests(unittest.TestCase):
    def test_q8_k_mlp_workspace_uses_canonical_block_size(self) -> None:
        specs = build_activation_specs(
            {
                "embed_dim": 2560,
                "intermediate_size": 10240,
                "num_layers": 42,
            },
            mode="prefill",
            context_len=25,
        )
        expected = 25 * (10240 // 256) * 292
        self.assertEqual(specs["layer_input"]["size"], expected)
        self.assertGreater(specs["layer_input"]["size"], 25 * 2560 * 4)

    def test_quantized_write_extent_rejects_neighbor_overwrite(self) -> None:
        operation = {
            "op": "quantize_mlp_down_input",
            "kernel": "quantize_row_q8_k",
            "layer": 0,
            "params": {"rows": 25, "_input_dim": 10240},
            "resolved_codegen_capability": {
                "operator_family": "activation_quantization",
                "output_storage": {
                    "format": "q8_k",
                    "block_elements": 256,
                    "block_bytes": 292,
                },
            },
            "outputs": {
                "output": {"buffer": "layer_input", "activation_offset": 0}
            },
        }
        layout = {
            "memory": {
                "activations": {
                    "size": 528000,
                    "buffers": [
                        {"name": "layer_input", "offset": 0, "size": 272000},
                        {"name": "residual", "offset": 272000, "size": 256000},
                    ],
                }
            }
        }
        with self.assertRaisesRegex(RuntimeError, "writes 292000 bytes.*only 272000"):
            _validate_lowered_activation_memory({"operations": [operation]}, layout)

        layout["memory"]["activations"] = {
            "size": 548000,
            "buffers": [
                {"name": "layer_input", "offset": 0, "size": 292000},
                {"name": "residual", "offset": 292000, "size": 256000},
            ],
        }
        report = _validate_lowered_activation_memory(
            {"operations": [operation]}, layout
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["quantized_write_count"], 1)
        self.assertEqual(report["writes"][0]["required_bytes"], 292000)
        self.assertEqual(report["writes"][0]["available_bytes"], 292000)

    def test_quantized_write_extent_rejects_provider_block_abi_drift(self) -> None:
        operation = {
            "op": "quantize_mlp_down_input",
            "kernel": "quantize_row_q8_k",
            "layer": 0,
            "params": {"rows": 25, "_input_dim": 10240},
            "resolved_codegen_capability": {
                "operator_family": "activation_quantization",
                "output_storage": {
                    "format": "q8_k",
                    "block_elements": 256,
                    "block_bytes": 272,
                },
            },
            "outputs": {
                "output": {"buffer": "layer_input", "activation_offset": 0}
            },
        }
        layout = {
            "memory": {
                "activations": {
                    "size": 292000,
                    "buffers": [
                        {"name": "layer_input", "offset": 0, "size": 292000}
                    ],
                }
            }
        }
        with self.assertRaisesRegex(
            RuntimeError,
            "declares block_bytes=272 for q8_k.*canonical ABI requires 292",
        ):
            _validate_lowered_activation_memory({"operations": [operation]}, layout)

    def test_activation_layout_rejects_overlapping_intervals(self) -> None:
        layout = {
            "memory": {
                "activations": {
                    "size": 192,
                    "buffers": [
                        {"name": "a", "offset": 0, "size": 128},
                        {"name": "b", "offset": 64, "size": 128},
                    ],
                }
            }
        }
        with self.assertRaisesRegex(RuntimeError, "b starts at 64 before a ends at 128"):
            _validate_lowered_activation_memory({"operations": []}, layout)

    def test_load_template_chat_contract_gemma4(self) -> None:
        contract = load_template_chat_contract("gemma4")
        self.assertIsNotNone(contract)
        self.assertEqual(contract["turn_prefix"], "<|turn>{role}\n")
        self.assertEqual(contract["turn_suffix"], "<turn|>\n")
        self.assertEqual(contract["assistant_generation_prefix"], "<|turn>model\n")
        self.assertEqual(contract["token_stop_markers"], ["<turn|>"])

    def test_gemma4_uses_q8_activation_logits_by_default(self) -> None:
        gemma4 = _apply_circuit_runtime_defaults(
            {}, _load_builtin_template_doc("gemma4"), source="gemma4"
        )
        gemma3 = _apply_circuit_runtime_defaults(
            {}, _load_builtin_template_doc("gemma3"), source="gemma3"
        )
        self.assertNotIn("prefer_fp32_logits", gemma4)
        self.assertTrue(gemma3["prefer_fp32_logits"])

    def test_build_chat_contract_detects_gemma4_markers(self) -> None:
        chat_template = "<|turn>user\nHello<turn|>\n<|turn>model\n"
        contract = build_chat_contract(
            template_data=None,
            chat_template=chat_template,
            finetune="it",
            model_name="Gemma-4-E4B-It",
            model_type="gemma4",
        )
        self.assertIsNotNone(contract)
        self.assertEqual(contract["name"], "gemma4")
        self.assertIn("<|turn>", contract["template_markers"])
        self.assertIn("<turn|>", contract["template_markers"])

    def test_classify_layer_contract_gemma4_hybrid(self) -> None:
        tensors = {
            "blk.0.attn_q.weight": _tensor("blk.0.attn_q.weight", (2560, 2048)),
            "blk.0.attn_k.weight": _tensor("blk.0.attn_k.weight", (2560, 512)),
            "blk.0.attn_v.weight": _tensor("blk.0.attn_v.weight", (2560, 512)),
            "blk.0.attn_output.weight": _tensor("blk.0.attn_output.weight", (2048, 2560)),
            "blk.0.attn_q_norm.weight": _tensor("blk.0.attn_q_norm.weight", (256,)),
            "blk.0.attn_k_norm.weight": _tensor("blk.0.attn_k_norm.weight", (256,)),
            "blk.0.attn_norm.weight": _tensor("blk.0.attn_norm.weight", (2560,)),
            "blk.0.post_attention_norm.weight": _tensor("blk.0.post_attention_norm.weight", (2560,)),
            "blk.0.ffn_gate.weight": _tensor("blk.0.ffn_gate.weight", (2560, 10240)),
            "blk.0.ffn_up.weight": _tensor("blk.0.ffn_up.weight", (2560, 10240)),
            "blk.0.ffn_down.weight": _tensor("blk.0.ffn_down.weight", (10240, 2560)),
            "blk.0.inp_gate.weight": _tensor("blk.0.inp_gate.weight", (2560, 256)),
            "blk.0.proj.weight": _tensor("blk.0.proj.weight", (256, 2560)),
        }
        self.assertEqual(classify_layer_contract(tensors, 0), "gemma4_hybrid")

    def test_describe_layer_contract_gemma4_hybrid_is_specific(self) -> None:
        tensors = {
            "blk.0.attn_q.weight": _tensor("blk.0.attn_q.weight", (2560, 2048)),
            "blk.0.attn_k.weight": _tensor("blk.0.attn_k.weight", (2560, 512)),
            "blk.0.attn_v.weight": _tensor("blk.0.attn_v.weight", (2560, 512)),
            "blk.0.attn_output.weight": _tensor("blk.0.attn_output.weight", (2048, 2560)),
            "blk.0.attn_q_norm.weight": _tensor("blk.0.attn_q_norm.weight", (256,)),
            "blk.0.attn_k_norm.weight": _tensor("blk.0.attn_k_norm.weight", (256,)),
            "blk.0.attn_norm.weight": _tensor("blk.0.attn_norm.weight", (2560,)),
            "blk.0.post_attention_norm.weight": _tensor("blk.0.post_attention_norm.weight", (2560,)),
            "blk.0.ffn_gate.weight": _tensor("blk.0.ffn_gate.weight", (2560, 10240)),
            "blk.0.ffn_up.weight": _tensor("blk.0.ffn_up.weight", (2560, 10240)),
            "blk.0.ffn_down.weight": _tensor("blk.0.ffn_down.weight", (10240, 2560)),
            "blk.0.inp_gate.weight": _tensor("blk.0.inp_gate.weight", (2560, 256)),
            "blk.0.proj.weight": _tensor("blk.0.proj.weight", (256, 2560)),
        }
        detail = describe_layer_contract(tensors, 0, arch="gemma4")
        self.assertIsNotNone(detail)
        self.assertIn("Gemma4 hybrid block", detail)
        self.assertIn("per-layer projection lowering", detail)

    def test_build_gemma4_attention_plan_makes_shared_kv_explicit(self) -> None:
        plan = build_gemma4_attention_plan(
            {
                "gemma4.attention.sliding_window_pattern": [True, False, True, False],
                "gemma4.attention.shared_kv_layers": 2,
                "gemma4.attention.sliding_window": 1024,
                "gemma4.attention.head_count": 16,
                "gemma4.attention.head_count_kv": 4,
                "gemma4.attention.key_length": 256,
                "gemma4.attention.value_length": 256,
                "gemma4.attention.key_length_swa": 128,
                "gemma4.attention.value_length_swa": 128,
                "gemma4.rope.dimension_count": 512,
                "gemma4.rope.dimension_count_swa": 256,
            },
            4,
        )
        self.assertEqual(
            plan["layer_kinds"],
            [
                "sliding_attention_kv",
                "full_attention_kv",
                "sliding_attention_shared_kv",
                "full_attention_shared_kv",
            ],
        )
        self.assertEqual(plan["layer_kv_policy"], ["produce", "produce", "reuse", "reuse"])
        self.assertEqual(plan["layer_kv_source"], [0, 1, 0, 1])
        self.assertEqual(plan["layer_sliding_window"], [1024, 0, 1024, 0])
        self.assertEqual(plan["layer_rope_kind"], ["swa", "full", "swa", "full"])
        self.assertEqual(plan["layer_q_head_dim"], [128, 256, 128, 256])
        self.assertEqual(plan["layer_k_head_dim"], [128, 256, 128, 256])
        self.assertEqual(plan["layer_v_head_dim"], [128, 256, 128, 256])
        self.assertEqual(plan["layer_rotary_dim"], [256, 512, 256, 512])
        self.assertEqual(plan["layer_q_dim"], [2048, 4096, 2048, 4096])
        self.assertEqual(plan["layer_kv_dim"], [512, 1024, 512, 1024])
        self.assertEqual(plan["layer_attention_output_dim"], [2048, 4096, 2048, 4096])

    def test_full_attention_output_projection_uses_per_layer_value_width(self) -> None:
        config = {
            "embed_dim": 2560,
            "num_heads": 8,
            "num_kv_heads": 2,
            "head_dim": 512,
            "attn_out_dim": 2048,
            "layer_q_head_dim": [256, 512],
            "layer_k_head_dim": [256, 512],
            "layer_v_head_dim": [256, 512],
            "layer_q_dim": [2048, 4096],
            "layer_attention_output_dim": [2048, 4096],
            "layer_rotary_dim": [256, 512],
            "layer_sliding_window": [512, 0],
            "layer_rope_kind": ["swa", "full"],
        }

        out_proj = {}
        apply_layer_attention_dims("out_proj", out_proj, 1, config)
        self.assertEqual(out_proj["_output_dim"], 2560)
        self.assertEqual(out_proj["_input_dim"], 4096)
        self.assertEqual(out_proj["input_dim"], 4096)

        quantize = {}
        apply_layer_attention_dims("quantize_out_proj_input", quantize, 1, config)
        self.assertEqual(quantize["_input_dim"], 4096)
        self.assertEqual(quantize["input_dim"], 4096)

    def test_gemma4_template_declares_q_only_shared_kv_kinds(self) -> None:
        template_path = REPO_ROOT / "version" / "v8" / "circuits" / "gemma4.json"
        import json

        template = json.loads(template_path.read_text(encoding="utf-8"))
        body = template["block_types"]["decoder"]["body"]
        self.assertEqual(body["kind_config_key"], "layer_kinds")
        self.assertIn("sliding_attention_shared_kv", body["ops_by_kind"])
        self.assertIn("full_attention_shared_kv", body["ops_by_kind"])
        shared_ops = body["ops_by_kind"]["sliding_attention_shared_kv"]
        self.assertIn("q_proj", shared_ops)
        self.assertIn("q_norm", shared_ops)
        self.assertIn("rope_q", shared_ops)
        self.assertIn("attn_sliding_shared_kv", shared_ops)
        self.assertNotIn("k_proj", shared_ops)
        self.assertNotIn("v_proj", shared_ops)

    def test_gemma4_geglu_declares_disjoint_compact_stream(self) -> None:
        template_path = REPO_ROOT / "version" / "v8" / "circuits" / "gemma4.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))

        self.assertEqual(template["activation_bindings"]["mlp_compact"], "mlp_compact")
        self.assertEqual(
            template["activation_buffers"]["mlp_compact"]["shape"],
            [
                {"config": "execution_extent"},
                {"config": "intermediate_size"},
            ],
        )
        for ops in template["block_types"]["decoder"]["body"]["ops_by_kind"].values():
            by_name = {
                item if isinstance(item, str) else item["op"]: item
                for item in ops
            }
            self.assertEqual(
                by_name["geglu"]["graph_slots"],
                {
                    "inputs": {"x": "mlp_scratch"},
                    "outputs": {"out": "mlp_compact"},
                },
            )
            self.assertEqual(
                by_name["mlp_down"]["graph_slots"]["inputs"]["x"],
                "mlp_compact",
            )

    def test_gemma4_assistant_q_only_ops_are_lowerable(self) -> None:
        import json
        import build_ir_v8  # type: ignore

        template_path = REPO_ROOT / "version" / "v8" / "circuits" / "gemma4_assistant.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        ops = build_ir_v8._collect_template_ops(
            template,
            {
                "num_layers": 2,
                "layer_kinds": [
                    "sliding_attention_q_only_k_eq_v",
                    "full_attention_q_only_k_eq_v",
                ],
            },
        )
        self.assertEqual(build_ir_v8.validate_template_ops(ops), [])
        for op in (
            "assistant_pre_projection",
            "q_norm",
            "rope_q",
            "attn_sliding_shared_kv",
            "attn_shared_kv",
            "assistant_layer_scale",
            "assistant_post_projection",
        ):
            self.assertIn(op, ops)

        for kernel_id in (
            "assistant_layer_scale_forward",
            "q_norm_forward",
            "rope_forward_q_gemma4",
            "kv_cache_store_shared_q",
            "attention_forward_causal_head_major_shared_kv_gemma4",
            "attention_forward_causal_head_major_shared_kv_sliding_gemma4",
            "attention_forward_decode_head_major_shared_kv_gemma4",
            "attention_forward_decode_head_major_shared_kv_sliding_gemma4",
        ):
            self.assertTrue((REPO_ROOT / "version" / "v8" / "kernel_maps" / f"{kernel_id}.json").exists())

    def test_gemma4_speculative_pair_template_declares_bridge_and_verifier(self) -> None:
        template_path = REPO_ROOT / "version" / "v8" / "circuits" / "gemma4_speculative_pair.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        self.assertTrue(template["experimental"])
        self.assertEqual(template["contract"]["target"]["template"], "gemma4")
        self.assertEqual(template["contract"]["draft"]["template"], "gemma4_assistant")
        self.assertEqual(template["contract"]["bridge"]["source"], "target_hidden_stream")
        self.assertEqual(template["contract"]["bridge"]["dest"], "draft_backbone_stream")
        self.assertEqual(template["contract"]["verifier"]["kernel"], "speculative_verify_greedy_f32")
        self.assertEqual(template["contract"]["committer"]["kernel"], "speculative_commit_one_i32")
        speculative = template["contract"]["speculative_contract"]
        self.assertTrue(speculative["enabled"])
        self.assertEqual(speculative["mode"], "target_draft_verify")
        self.assertEqual(speculative["draft_length"], 1)
        self.assertEqual(speculative["verify_policy"], "greedy")
        self.assertEqual(speculative["commit_policy"], "accept_one_or_target_fallback")
        self.assertEqual(speculative["bridge"]["source"], "target_hidden_stream")
        self.assertEqual(speculative["bridge"]["dest"], "draft_backbone_stream")
        self.assertEqual(speculative["token_source"], "speculative_commit_or_reject.verified_token")
        self.assertEqual(speculative["count_source"], "constant_1")
        self.assertIn("speculative_tokens_match_target_only_greedy", speculative["invariants"])
        autoregressive = template["contract"]["autoregressive_contract"]
        self.assertTrue(autoregressive["enabled"])
        self.assertEqual(autoregressive["mode"], "speculative")
        self.assertEqual(autoregressive["stream"], "main_generation")
        self.assertEqual(autoregressive["position_source"], "target_position")
        self.assertEqual(autoregressive["token_source"], "speculative_commit_or_reject.verified_token")

        ops = template["block_types"]["speculative_decode"]["ops"]
        self.assertEqual(
            [op["op"] for op in ops],
            [
                "target_decode_step",
                "target_hidden_to_draft_backbone_stream",
                "draft_decode_step",
                "speculative_verify_greedy",
                "speculative_commit_or_reject",
            ],
        )
        verify = ops[3]["graph_slots"]
        self.assertEqual(verify["inputs"]["target_logits"], "target_logits")
        self.assertEqual(verify["inputs"]["draft_token"], "draft_candidate_token")
        self.assertEqual(verify["outputs"]["accepted"], "accepted_flag")
        self.assertEqual(verify["outputs"]["verified_token"], "verified_token")
        commit = ops[4]["graph_slots"]
        self.assertEqual(ops[4]["kernel"], "speculative_commit_one_i32")
        self.assertEqual(commit["inputs"]["accepted"], "accepted_flag")
        self.assertEqual(commit["inputs"]["verified_token"], "verified_token")
        self.assertEqual(commit["inputs"]["token_count"], "token_count")
        self.assertEqual(commit["outputs"]["draft_position"], "draft_position")
        self.assertTrue((REPO_ROOT / "version" / "v8" / "kernel_maps" / "speculative_verify_greedy_f32.json").exists())
        self.assertTrue((REPO_ROOT / "version" / "v8" / "kernel_maps" / "speculative_commit_one_i32.json").exists())

    def test_gemma4_speculative_pair_probe_script_exists(self) -> None:
        script = REPO_ROOT / "version" / "v8" / "scripts" / "run_gemma4_speculative_pair_probe_v8.py"
        self.assertTrue(script.exists())
        text = script.read_text(encoding="utf-8")
        self.assertIn("target_hidden_stream", text)
        self.assertIn("draft_backbone_stream", text)
        self.assertIn("speculative_verify_greedy_f32", text)
        self.assertIn("speculative_commit_one_i32", text)

    def test_gemma4_assistant_synthetic_checkpoint_generates_runtime(self) -> None:
        try:
            import torch  # type: ignore
            import safetensors.torch as st  # type: ignore
        except Exception as exc:
            self.skipTest(f"torch/safetensors unavailable: {exc}")

        with tempfile.TemporaryDirectory(prefix="ck_gemma4_assistant_e2e_") as td:
            root = Path(td)
            checkpoint = root / "gemma4_assistant"
            runtime = root / "runtime"
            checkpoint.mkdir()
            runtime.mkdir()

            (checkpoint / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Gemma4AssistantForCausalLM"],
                        "model_type": "gemma4_assistant",
                        "backbone_hidden_size": 16,
                        "tie_word_embeddings": True,
                        "use_ordered_embeddings": True,
                        "text_config": {
                            "model_type": "gemma4_text",
                            "attention_bias": False,
                            "attention_k_eq_v": True,
                            "bos_token_id": 2,
                            "eos_token_id": 1,
                            "global_head_dim": 8,
                            "head_dim": 4,
                            "hidden_activation": "gelu_pytorch_tanh",
                            "hidden_size": 8,
                            "intermediate_size": 16,
                            "layer_types": ["sliding_attention", "full_attention"],
                            "max_position_embeddings": 128,
                            "num_attention_heads": 2,
                            "num_global_key_value_heads": 1,
                            "num_hidden_layers": 2,
                            "num_key_value_heads": 2,
                            "num_kv_shared_layers": 2,
                            "rms_norm_eps": 1e-6,
                            "rope_parameters": {
                                "full_attention": {
                                    "partial_rotary_factor": 0.25,
                                    "rope_theta": 1000000.0,
                                    "rope_type": "proportional",
                                },
                                "sliding_attention": {
                                    "rope_theta": 10000.0,
                                    "rope_type": "default",
                                },
                            },
                            "sliding_window": 32,
                            "tie_word_embeddings": True,
                            "vocab_size": 32,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _write_tiny_bpe_tokenizer(checkpoint, vocab_size=32)

            tensors = {
                "model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
                "model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
                "masked_embedding.centroids.weight": torch.randn(4, 8, dtype=torch.bfloat16),
                "masked_embedding.token_ordering": torch.arange(32, dtype=torch.int64),
                "pre_projection.weight": torch.randn(8, 16, dtype=torch.bfloat16),
                "post_projection.weight": torch.randn(16, 8, dtype=torch.bfloat16),
            }
            for layer, q_dim in enumerate((8, 16)):
                pfx = f"model.layers.{layer}"
                tensors.update(
                    {
                        f"{pfx}.input_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                        f"{pfx}.pre_feedforward_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                        f"{pfx}.post_attention_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                        f"{pfx}.post_feedforward_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                        f"{pfx}.layer_scalar": torch.ones(1, dtype=torch.bfloat16),
                        f"{pfx}.self_attn.q_proj.weight": torch.randn(q_dim, 8, dtype=torch.bfloat16),
                        f"{pfx}.self_attn.q_norm.weight": torch.ones(q_dim // 2, dtype=torch.bfloat16),
                        f"{pfx}.self_attn.o_proj.weight": torch.randn(8, q_dim, dtype=torch.bfloat16),
                        f"{pfx}.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
                        f"{pfx}.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
                        f"{pfx}.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
                    }
                )
            st.save_file(tensors, checkpoint / "model.safetensors")

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "version" / "v8" / "scripts" / "ck_run_v8.py"),
                    "run",
                    str(checkpoint),
                    "--run",
                    str(runtime),
                    "--context-len",
                    "16",
                    "--force-convert",
                    "--force-compile",
                    "--generate-only",
                    "--chat-template",
                    "none",
                    "--allow-raw-prompt",
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            cfg = json.loads((runtime / "config.json").read_text(encoding="utf-8"))
            build_dir = runtime
            self.assertEqual(cfg["model"], "gemma4_assistant")
            self.assertFalse(cfg["standalone_text_inference_supported"])
            self.assertEqual(cfg["layer_kinds"], ["sliding_attention_q_only_k_eq_v", "full_attention_q_only_k_eq_v"])
            self.assertTrue((build_dir / "model_v8.c").exists())
            self.assertTrue((build_dir / "libmodel.so").exists())
            self.assertIn("assistant_pre_projection", (build_dir / "model_v8.c").read_text(encoding="utf-8"))
            self.assertIn("attn_shared_kv", (build_dir / "lowered_decode_call.json").read_text(encoding="utf-8"))

    def test_gemma4_kv_layers_use_supported_paired_qk_ops_for_first_bringup(self) -> None:
        template_path = REPO_ROOT / "version" / "v8" / "circuits" / "gemma4.json"
        import json

        template = json.loads(template_path.read_text(encoding="utf-8"))
        body = template["block_types"]["decoder"]["body"]
        for kind in ("sliding_attention_kv", "full_attention_kv"):
            ops = body["ops_by_kind"][kind]
            self.assertIn("qk_norm", ops)
            self.assertIn("rope_qk", ops)
            self.assertNotIn("q_norm", ops)
            self.assertNotIn("rope_q", ops)

    def test_gemma4_v_norm_is_unweighted_rmsnorm(self) -> None:
        import json

        template_path = REPO_ROOT / "version" / "v8" / "circuits" / "gemma4.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        body = template["block_types"]["decoder"]["body"]
        for kind in ("sliding_attention_kv", "full_attention_kv"):
            ops = body["ops_by_kind"][kind]
            self.assertIn("v_norm", ops)
            self.assertLess(ops.index("v_proj"), ops.index("v_norm"))
            self.assertLess(ops.index("v_norm"), ops.index("qk_norm"))

        overlay_path = REPO_ROOT / "version" / "v8" / "kernel_maps" / "kernel_bindings.overlay.json"
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        binding = overlay["bindings"]["rmsnorm_forward_no_weight"]
        self.assertNotIn("gamma", {param["name"] for param in binding["params"]})
        self.assertNotIn("v_norm_gamma", json.dumps(binding))

    def test_gemma4_template_runs_per_layer_embedding_after_ffn_residual(self) -> None:
        import json

        template_path = REPO_ROOT / "version" / "v8" / "circuits" / "gemma4.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        body = template["block_types"]["decoder"]["body"]
        for ops in body["ops_by_kind"].values():
            self.assertIn("gemma4_per_layer_embed", ops)
            self.assertLess(ops.index("post_ffn_norm"), ops.index("gemma4_per_layer_embed"))

        kernel_map = json.loads(
            (REPO_ROOT / "version" / "v8" / "kernel_maps" / "gemma4_per_layer_embed_forward.json").read_text(encoding="utf-8")
        )
        self.assertEqual(kernel_map["op"], "gemma4_per_layer_embed")
        self.assertEqual(kernel_map["impl"]["function"], "gemma4_per_layer_embed_forward")
        prepare_map = json.loads(
            (REPO_ROOT / "version" / "v8" / "kernel_maps" / "gemma4_per_layer_prepare_forward.json").read_text(encoding="utf-8")
        )
        self.assertEqual(prepare_map["op"], "gemma4_per_layer_prepare")

    def test_gemma4_prepare_provider_is_selected_generically_by_storage_dtype(self) -> None:
        registry = load_kernel_registry()
        self.assertEqual(
            find_kernel(
                registry,
                op="gemma4_per_layer_prepare",
                quant={"weight": "q5_k"},
                mode="prefill",
            ),
            "gemma4_per_layer_prepare_forward",
        )
        self.assertEqual(
            find_kernel(
                registry,
                op="gemma4_per_layer_prepare",
                quant={"weight": "bf16"},
                mode="prefill",
            ),
            "gemma4_per_layer_prepare_bf16_forward",
        )

        source = (
            REPO_ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('if op == "gemma4_per_layer_prepare"', source)


if __name__ == "__main__":
    unittest.main()
