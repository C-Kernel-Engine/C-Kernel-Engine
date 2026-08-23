#!/usr/bin/env python3
import json
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "version" / "v7" / "scripts" / "parity") not in sys.path:
    sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts" / "parity"))

import ck_chat  # type: ignore
import chat_contract  # type: ignore
import compare_first_token_logits as first_token  # type: ignore


class _FakeCFunc:
    def __init__(self, fn):
        self.fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.fn(*args)


class TestCKChatRuntimeContract(unittest.TestCase):
    def test_manifest_template_contract_is_authoritative(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_manifest_contract_") as td:
            run_dir = Path(td)
            (run_dir / "weights_manifest.json").write_text(
                json.dumps(
                    {
                        "template": {
                            "name": "fixture",
                            "contract": {
                                "chat_contract": {
                                    "name": "fixture",
                                    "raw_prompt_allowed": False,
                                    "conversation_prefix": "<bos>",
                                    "turn_prefix_by_role": {"user": "<user>"},
                                    "assistant_generation_prefix": "<assistant>",
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            model._configure_chat_template("auto")

            self.assertTrue(model.use_chat_template)
            self.assertEqual(
                model.format_chat_prompt("Hello"),
                "<bos><user>Hello<assistant>",
            )

    def test_role_specific_turn_contract_matches_deepseek_style_template(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_role_contract_") as td:
            run_dir = Path(td)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "chat_contract": {
                            "name": "role_specific",
                            "raw_prompt_allowed": False,
                            "conversation_prefix": "<bos>",
                            "turn_prefix_by_role": {
                                "system": "",
                                "user": "<user>",
                                "assistant": "<assistant>",
                            },
                            "turn_suffix_by_role": {
                                "assistant": "<eos>",
                            },
                            "assistant_generation_prefix": "<assistant>",
                            "system_prompt_mode": "dedicated_turn",
                        }
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            model._configure_chat_template("auto")

            self.assertEqual(
                model.format_chat_prompt("Hello"),
                "<bos><user>Hello<assistant>",
            )
            self.assertEqual(
                model.format_chat_conversation(
                    [("user", "Hello"), ("assistant", "Hi"), ("user", "Next")],
                    system_prompt="System",
                ),
                "<bos>System<user>Hello<assistant>Hi<eos><user>Next<assistant>",
            )

    def test_glm4_contract_matches_gguf_generation_prompt_exactly(self) -> None:
        model = ck_chat.CKModel("/tmp/nonexistent")
        model._configure_chat_template("glm4")

        prompt = model.format_chat_prompt("Hello!")

        self.assertEqual(
            prompt,
            "[gMASK]<sop>\n<|user|>\nHello!\n<|assistant|>",
        )
        self.assertFalse(prompt.endswith("\n"))

    def test_glm4_gguf_jinja_compiles_to_exact_generation_prefix(self) -> None:
        contract = chat_contract.build_chat_contract(
            template_data={},
            chat_template=(
                "[gMASK]<sop>\n"
                "{% for message in messages %}"
                "{% if message.role == 'user' %}<|user|>\n{{ message.content }}"
                "{% endif %}{% endfor %}"
                "{% if add_generation_prompt %}<|assistant|>{% endif %}"
            ),
            model_type="glm4",
        )

        self.assertIsNotNone(contract)
        self.assertEqual(contract["assistant_generation_prefix"], "<|assistant|>")

    def test_generated_runtime_root_recognizes_only_supported_directory_names(self) -> None:
        model_root = Path("/tmp/model")

        self.assertEqual(
            ck_chat._model_root_for_runtime(model_root / ".ck_build"),
            model_root,
        )
        self.assertEqual(
            ck_chat._model_root_for_runtime(model_root / ".ck_build_v8"),
            model_root,
        )
        self.assertEqual(
            ck_chat._model_root_for_runtime(model_root / ".ck_builder"),
            model_root / ".ck_builder",
        )

    def test_auto_mode_uses_exported_python_tokenizer_chat_template(self) -> None:
        class _Tokenizer:
            chat_template = "exported-template"

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                self.last_call = (messages, tokenize, add_generation_prompt)
                return "<rendered>"

        model = ck_chat.CKModel("/tmp/nonexistent")
        model.tokenizer = _Tokenizer()
        model._configure_chat_template("auto")

        rendered = model.format_chat_conversation(
            [("user", "Hi")],
            system_prompt="System",
        )

        self.assertEqual(rendered, "<rendered>")
        self.assertEqual(model.chat_template_mode, "tokenizer")
        self.assertEqual(
            model.tokenizer.last_call,
            (
                [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hi"},
                ],
                False,
                True,
            ),
        )

    def test_first_token_llama_root_honors_environment(self) -> None:
        old = os.environ.get("CK_LLAMA_CPP_ROOT")
        try:
            os.environ["CK_LLAMA_CPP_ROOT"] = "/tmp/cke-test-llama-root"
            self.assertEqual(
                first_token._llama_cpp_root(),
                Path("/tmp/cke-test-llama-root").resolve(),
            )
        finally:
            if old is None:
                os.environ.pop("CK_LLAMA_CPP_ROOT", None)
            else:
                os.environ["CK_LLAMA_CPP_ROOT"] = old

    def test_ck_model_reads_named_activation_f32_when_runtime_exports_api(self) -> None:
        data = np.array([1.25, -2.5, 3.75, 4.5], dtype=np.float32)
        data_ptr = data.ctypes.data

        class FakeLib:
            def __init__(self) -> None:
                self.ck_model_get_named_activation_ptr = _FakeCFunc(
                    lambda name: data_ptr if name == b"target_hidden_stream" else 0
                )
                self.ck_model_get_named_activation_nbytes = _FakeCFunc(
                    lambda name: data.nbytes if name == b"target_hidden_stream" else -1
                )
                self.ck_model_get_named_activation_runtime_offset = _FakeCFunc(
                    lambda name: 4096 if name == b"target_hidden_stream" else -1
                )

        model = ck_chat.CKModel("/tmp/nonexistent")
        model.lib = FakeLib()
        model._setup_named_activation_api()

        self.assertTrue(model.has_named_activations)
        self.assertEqual(model.named_activation_nbytes("target_hidden_stream"), data.nbytes)
        self.assertEqual(model.named_activation_runtime_offset("target_hidden_stream"), 4096)
        self.assertEqual(model.named_activation_nbytes("missing"), -1)
        self.assertIsNone(model.read_named_activation_f32("missing"))
        np.testing.assert_allclose(
            model.read_named_activation_f32("target_hidden_stream"),
            data,
        )
        np.testing.assert_allclose(
            model.read_named_activation_f32("target_hidden_stream", max_floats=2),
            data[:2],
        )

    def test_load_model_meta_preserves_non_null_config_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_meta_") as td:
            run_dir = Path(td)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "chat_template": "<|im_start|>user\n{{ prompt }}<|im_end|>\n<|im_start|>assistant\n",
                        "model_type": "qwen35",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "weights_manifest.json").write_text(
                json.dumps({"config": {"chat_template": None, "model_type": "qwen35"}}),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            meta = model._load_model_meta()
            self.assertIn("<|im_start|>", str(meta.get("chat_template")))
            self.assertEqual(meta.get("model_type"), "qwen35")

    def test_runtime_contract_marks_recurrent_layer_kinds_as_sequential_prefill(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_contract_") as td:
            run_dir = Path(td)
            (run_dir / "config.json").write_text(
                json.dumps({"layer_kinds": ["recurrent", "full_attention"]}),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            contract = model._load_runtime_contract()
            self.assertEqual(contract.get("prefill_policy"), "sequential_decode")

    def test_runtime_contract_keeps_explicit_bf16_hybrid_prefill_batched(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_bf16_prefill_") as td:
            run_dir = Path(td)
            (run_dir / "weights_manifest.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "prefill_policy": "batched",
                            "recurrent_qkv_weight_dtype": "bf16",
                            "layer_kinds": ["recurrent", "full_attention"],
                        },
                        "template": {
                            "flags": {"prefill_policy": "sequential_decode"}
                        },
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            contract = model._load_runtime_contract()

        self.assertEqual(contract.get("prefill_policy"), "batched")

    def test_runtime_contract_keeps_q5_hybrid_prefill_sequential(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_q5_prefill_") as td:
            run_dir = Path(td)
            (run_dir / "weights_manifest.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "prefill_policy": "batched",
                            "recurrent_qkv_weight_dtype": "q5_k",
                            "layer_kinds": ["recurrent", "full_attention"],
                        },
                        "template": {
                            "flags": {"prefill_policy": "sequential_decode"}
                        },
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            contract = model._load_runtime_contract()

        self.assertEqual(contract.get("prefill_policy"), "sequential_decode")

    def test_runtime_contract_allows_explicit_batched_prefill_certification(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_force_batched_") as td:
            run_dir = Path(td)
            (run_dir / "weights_manifest.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "prefill_policy": "batched",
                            "recurrent_qkv_weight_dtype": "q5_k",
                            "layer_kinds": ["recurrent", "full_attention"],
                        },
                        "template": {
                            "flags": {"prefill_policy": "sequential_decode"}
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"CK_V8_FORCE_BATCHED_PREFILL": "1"},
                clear=False,
            ):
                model = ck_chat.CKModel(str(run_dir))
                contract = model._load_runtime_contract()

        self.assertEqual(contract.get("prefill_policy"), "batched")

    def test_runtime_contract_loads_sampler_defaults_from_config(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_sampler_") as td:
            run_dir = Path(td)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "sampler_defaults": {
                            "repeat_penalty": 1.12,
                            "repeat_last_n": 96,
                            "no_repeat_ngram_size": 4,
                        }
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            contract = model._load_runtime_contract()

        self.assertEqual(contract["sampler_defaults"]["repeat_penalty"], 1.12)
        self.assertEqual(contract["sampler_defaults"]["repeat_last_n"], 96)
        self.assertEqual(contract["sampler_defaults"]["no_repeat_ngram_size"], 4)

    def test_runtime_contract_keeps_dense_attention_on_batched_prefill(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_dense_") as td:
            run_dir = Path(td)
            (run_dir / "weights_manifest.json").write_text(
                json.dumps({"config": {"layer_kinds": ["full_attention", "full_attention"]}}),
                encoding="utf-8",
            )
            contract = first_token.load_runtime_contract(run_dir)
            self.assertEqual(contract.get("prefill_policy"), "batched")

    def test_qwen35_auto_mode_defaults_to_visible_thinking(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_qwen35_") as td:
            run_dir = Path(td)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen35",
                        "chat_template": "<|im_start|>system\n{{ system }}<|im_end|>\n<|im_start|>assistant\n",
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            model._configure_chat_template("auto")
            self.assertTrue(model.use_chat_template)
            self.assertEqual(model.chat_template_mode, "qwen35")
            prompt = model.format_chat_prompt("Hello")
            self.assertNotIn("/no_think\nHello", prompt)
            self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_qwen3_auto_mode_uses_visible_think_generation_prompt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_qwen3_") as td:
            run_dir = Path(td)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "chat_template": "<|im_start|>user\n{{ prompt }}<|im_end|>\n{%- if add_generation_prompt %}<|im_start|>assistant\n{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{%- endif %}{%- endif %}",
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            model._configure_chat_template("auto")
            self.assertTrue(model.use_chat_template)
            self.assertEqual(model.chat_template_mode, "qwen3")
            prompt = model.format_chat_prompt("Hello")
            self.assertNotIn("/no_think\nHello", prompt)
            self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_qwen35_suppressed_thinking_mode_uses_empty_think_generation_prompt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_qwen35_suppressed_") as td:
            run_dir = Path(td)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen35",
                        "chat_template": "<|im_start|>system\n{{ system }}<|im_end|>\n<|im_start|>assistant\n",
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            model.thinking_mode = "suppressed"
            model._configure_chat_template("auto")
            prompt = model.format_chat_prompt("Hello")
            self.assertIn("/no_think\nHello", prompt)
            self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_legacy_qwen35_sidecar_contract_hydrates_visible_thinking_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_qwen35_legacy_") as td:
            run_dir = Path(td)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "chat_contract": {
                            "name": "qwen35",
                            "raw_prompt_allowed": False,
                            "turn_prefix": "<|im_start|>{role}\n",
                            "turn_suffix": "<|im_end|>\n",
                            "assistant_generation_prefix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                            "role_labels": {
                                "system": "system",
                                "user": "user",
                                "assistant": "assistant",
                            },
                            "last_user_prefix": "/no_think\n",
                            "last_user_prefix_suppression_markers": ["/no_think", "/nothink", "/think"],
                            "stop_text_markers": ["<|im_end|>"],
                            "token_stop_markers": ["<|im_end|>"],
                            "template_markers": ["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            model._configure_chat_template("auto")
            prompt = model.format_chat_prompt("Hello")
            self.assertNotIn("/no_think\nHello", prompt)
            self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_extracts_static_default_system_prompt_from_chat_template(self) -> None:
        template = (
            "{% for message in messages %}"
            "{% if loop.first and messages[0]['role'] != 'system' %}"
            "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )
        extracted = chat_contract.extract_static_default_system_prompt(template)
        self.assertEqual(extracted, "You are a helpful assistant.")

    def test_llama_chatml_contract_preserves_template_default_system_prompt(self) -> None:
        template = (
            "{%- if messages[0].role == 'system' %}"
            "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
            "{%- else %}"
            "{{- '<|im_start|>system\\n你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。<|im_end|>\\n' }}"
            "{%- endif %}"
            "{{- '<|im_start|>assistant\\n' }}"
            "{{- '<think>' }}"
            "{{- '</think>' }}"
        )
        contract = chat_contract.build_chat_contract(
            template_data={"name": "llama"},
            chat_template=template,
            model_type="llama",
            model_name="Nanbeige4.1-3B",
        )
        self.assertIsNotNone(contract)
        self.assertEqual(contract.get("name"), "llama_chatml")
        self.assertEqual(
            contract.get("default_system_prompt"),
            "你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。",
        )
        self.assertTrue(bool(contract.get("inject_default_system_prompt")))
        self.assertEqual(contract.get("assistant_generation_prefix"), "<|im_start|>assistant\n")
        self.assertEqual(contract.get("stop_text_markers"), ["<|im_end|>"])
        self.assertEqual(
            contract.get("template_markers"),
            ["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
        )

    def test_qwen35_chatml_contract_still_uses_qwen35_preset(self) -> None:
        contract = chat_contract.build_chat_contract(
            template_data={"name": "qwen35"},
            chat_template="<|im_start|>user\n{{ prompt }}<|im_end|>\n<|im_start|>assistant\n",
            model_type="qwen35",
            model_name="Qwen3.5-0.8B",
        )
        self.assertIsNotNone(contract)
        self.assertEqual(contract.get("name"), "qwen35")
        self.assertIn(
            "<think>",
            str(contract.get("assistant_generation_prefix_by_thinking_mode", {}).get("suppressed", "")),
        )

    def test_explicit_sidecar_chat_contract_drives_runtime_formatting(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_sidecar_") as td:
            run_dir = Path(td)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "special_tokens": {"add_bos_token": False},
                        "chat_contract": {
                            "name": "sidecar_contract",
                            "raw_prompt_allowed": False,
                            "turn_prefix": "<{role}>",
                            "turn_suffix": "|",
                            "assistant_generation_prefix": "<bot>",
                            "role_labels": {
                                "system": "sys",
                                "user": "usr",
                                "assistant": "bot",
                            },
                            "system_prompt_mode": "dedicated_turn",
                            "system_prompt_separator": "\n\n",
                            "default_system_prompt": "SYS",
                            "inject_default_system_prompt": True,
                            "force_bos_text_if_tokenizer_add_bos_false": "<bos>",
                            "last_user_prefix": "PREFIX:",
                            "last_user_prefix_suppression_markers": ["prefix:"],
                            "stop_text_markers": ["<stop>"],
                            "token_stop_markers": ["<stop>"],
                            "template_markers": ["<sys>", "<usr>", "<bot>"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            model._configure_chat_template("auto")

            self.assertTrue(model.use_chat_template)
            self.assertEqual(model.chat_template_mode, "sidecar_contract")
            self.assertEqual(model.default_stop_text_markers(), ["<stop>"])
            self.assertEqual(
                model.format_chat_prompt("Hello"),
                "<bos><sys>SYS|<usr>PREFIX:Hello|<bot>",
            )

    def test_explicit_sidecar_chat_contract_blocks_raw_prompt_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_sidecar_raw_") as td:
            run_dir = Path(td)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "chat_contract": {
                            "name": "sidecar_contract",
                            "raw_prompt_allowed": False,
                        }
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            self.assertIn("chat-template=none", str(model._raw_prompt_mode_risk("none")))

    def test_legacy_named_sidecar_contract_hydrates_template_defaults(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_legacy_contract_") as td:
            run_dir = Path(td)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "chat_contract": {
                            "name": "gemma",
                            "raw_prompt_allowed": False,
                            "turn_prefix": "<start_of_turn>{role}\n",
                            "turn_suffix": "<end_of_turn>\n",
                            "assistant_generation_prefix": "<start_of_turn>model\n",
                            "role_labels": {
                                "system": "system",
                                "user": "user",
                                "assistant": "model",
                            },
                            "system_prompt_mode": "prepend_first_user",
                            "stop_text_markers": ["<end_of_turn>"],
                            "token_stop_markers": ["<end_of_turn>"],
                            "template_markers": ["<start_of_turn>", "<end_of_turn>"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            model._configure_chat_template("auto")

            self.assertEqual(model.chat_template_mode, "gemma")
            self.assertEqual(model.default_min_new_tokens(), 8)

    def test_raw_prompt_mode_is_blocked_for_instruction_template_without_override(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_raw_prompt_") as td:
            run_dir = Path(td)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "gemma3",
                        "finetune": "it",
                        "chat_template": "<start_of_turn>user\n{{ prompt }}<end_of_turn>\n<start_of_turn>model\n",
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            self.assertIn("chat-template=none", str(model._raw_prompt_mode_risk("none")))
            self.assertIsNone(model._raw_prompt_mode_risk("auto"))

    def test_load_passes_allow_raw_prompt_override(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_load_override_") as td:
            run_dir = Path(td)
            (run_dir / "weights.bump").write_bytes(b"\0")
            (run_dir / "libmodel.so").write_bytes(b"\0")
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "gemma3",
                        "finetune": "it",
                        "chat_template": "<start_of_turn>user\n{{ prompt }}<end_of_turn>\n<start_of_turn>model\n",
                    }
                ),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))

            class _FakeLib:
                def __init__(self):
                    self._fns = {}

                def __getattr__(self, name):
                    if name not in self._fns:
                        fn = mock.Mock()
                        if name == "ck_model_has_tokenizer":
                            fn.return_value = 0
                        elif name == "ck_model_init":
                            fn.return_value = 0
                        elif name == "ck_model_get_vocab_size":
                            fn.return_value = 256
                        elif name == "ck_model_get_context_window":
                            fn.return_value = 128
                        elif name == "ck_model_free":
                            fn.return_value = None
                        elif name == "ck_model_get_active_tokens":
                            fn.return_value = 1
                        elif name == "ck_set_gemm_schedule":
                            fn.return_value = 0
                        self._fns[name] = fn
                    return self._fns[name]

            with mock.patch.object(ck_chat.ctypes, "CDLL", return_value=_FakeLib()):
                with mock.patch.object(model, "_load_python_tokenizer", return_value=True):
                    ok = model.load(chat_template="none", allow_raw_prompt=True)
            self.assertTrue(ok)

    def test_generate_uses_cumulative_decode_for_stream_text(self) -> None:
        class FakeModel:
            has_kv_decode = False
            eos_tokens = {0}
            vocab_size = 32
            context_window = 32

            def encode(self, text: str):
                return [9]

            def is_eos_token(self, token_id: int) -> bool:
                return token_id == 0

            def forward(self, token_ids):
                return np.zeros((32,), dtype=np.float32)

            def decode(self, token_ids):
                ids = list(token_ids)
                if ids == [1]:
                    return "\uFFFD"
                if ids == [2]:
                    return "A"
                if ids == [1, 2]:
                    return "éA"
                return ""

        sample_ids = iter([1, 2, 0])
        orig_sample = ck_chat.sample_top_k
        try:
            ck_chat.sample_top_k = lambda *args, **kwargs: next(sample_ids)
            with redirect_stdout(io.StringIO()) as buf:
                out = ck_chat.generate(FakeModel(), "hi", max_tokens=3, show_stats=False)
        finally:
            ck_chat.sample_top_k = orig_sample
        self.assertEqual(out, "éA")
        self.assertNotIn("\\uFFFD", buf.getvalue())
        self.assertNotIn("\ufffd", buf.getvalue())

    def test_generate_drops_incomplete_utf8_fragment_at_eos(self) -> None:
        class FakeModel:
            has_kv_decode = False
            eos_tokens = {0}
            vocab_size = 32
            context_window = 32

            def encode(self, text: str):
                return [9]

            def is_eos_token(self, token_id: int) -> bool:
                return token_id == 0

            def forward(self, token_ids):
                return np.zeros((32,), dtype=np.float32)

            def decode(self, token_ids):
                return "Thanks! \ufffd" if token_ids else ""

        sample_ids = iter([1, 0])
        orig_sample = ck_chat.sample_top_k
        try:
            ck_chat.sample_top_k = lambda *args, **kwargs: next(sample_ids)
            with redirect_stdout(io.StringIO()) as buf:
                out = ck_chat.generate(FakeModel(), "hi", max_tokens=2, show_stats=False)
        finally:
            ck_chat.sample_top_k = orig_sample

        self.assertEqual(out, "Thanks!")
        self.assertNotIn("\\uFFFD", buf.getvalue())
        self.assertNotIn("\ufffd", buf.getvalue())

    def test_tokenizer_input_normalization_is_nfc(self) -> None:
        decomposed = "e\u0301"
        self.assertEqual(ck_chat._normalize_tokenizer_input(decomposed, "gpt2"), "é")
        self.assertEqual(
            ck_chat._normalize_tokenizer_input(decomposed, "sentencepiece"),
            decomposed,
        )

    def test_generate_trims_obvious_repeated_suffix_loop(self) -> None:
        repeated = "Is there anything you want to know?"

        class FakeModel:
            has_kv_decode = False
            eos_tokens = set()
            vocab_size = 32
            context_window = 128

            def encode(self, text: str):
                return [9]

            def is_eos_token(self, token_id: int) -> bool:
                return False

            def forward(self, token_ids):
                return np.zeros((32,), dtype=np.float32)

            def decode(self, token_ids):
                ids = tuple(token_ids)
                mapping = {
                    (1,): "Intro",
                    (1, 2): f"Intro\n\n{repeated}",
                    (1, 2, 3): f"Intro\n\n{repeated}\n\n",
                    (1, 2, 3, 4): f"Intro\n\n{repeated}\n\n{repeated}",
                    (1, 2, 3, 4, 5): f"Intro\n\n{repeated}\n\n{repeated}\n\n",
                    (1, 2, 3, 4, 5, 6): f"Intro\n\n{repeated}\n\n{repeated}\n\n{repeated}",
                }
                return mapping.get(ids, "")

        sample_ids = iter([1, 2, 3, 4, 5, 6, 7])
        orig_sample = ck_chat.sample_top_k
        try:
            ck_chat.sample_top_k = lambda *args, **kwargs: next(sample_ids)
            with redirect_stdout(io.StringIO()) as buf:
                out = ck_chat.generate(FakeModel(), "hi", max_tokens=7, show_stats=False)
        finally:
            ck_chat.sample_top_k = orig_sample

        self.assertEqual(out.count(repeated), 2)
        self.assertEqual(buf.getvalue().count(repeated), 2)

    def test_generate_collapses_repeated_leading_think_markers_for_display(self) -> None:
        class FakeModel:
            has_kv_decode = False
            eos_tokens = {0}
            vocab_size = 32
            context_window = 64

            def encode(self, text: str):
                return [9]

            def is_eos_token(self, token_id: int) -> bool:
                return token_id == 0

            def forward(self, token_ids):
                return np.zeros((32,), dtype=np.float32)

            def decode(self, token_ids):
                ids = tuple(token_ids)
                mapping = {
                    (1,): "<think>",
                    (1, 2): "<think>\n\n",
                    (1, 2, 3): "<think>\n\n<think>",
                    (1, 2, 3, 4): "<think>\n\n<think>\n\nReasoning starts",
                    (1, 2, 3, 4, 5): "<think>\n\n<think>\n\nReasoning starts</think>\nAnswer",
                }
                return mapping.get(ids, "")

        sample_ids = iter([1, 2, 3, 4, 5, 0])
        orig_sample = ck_chat.sample_top_k
        try:
            ck_chat.sample_top_k = lambda *args, **kwargs: next(sample_ids)
            with redirect_stdout(io.StringIO()) as buf:
                out = ck_chat.generate(FakeModel(), "hi", max_tokens=6, show_stats=False)
        finally:
            ck_chat.sample_top_k = orig_sample

        rendered = buf.getvalue()
        self.assertEqual(rendered.count("<think>"), 1)
        self.assertIn("Reasoning starts", rendered)
        self.assertIn("Answer", rendered)
        self.assertEqual(out.count("<think>"), 1)

    def test_terminal_thinking_style_preserves_text(self) -> None:
        text = "<think>\nReasoning\n</think>\nAnswer"
        styled = ck_chat._style_thinking_for_terminal(text, enabled=True)
        self.assertIn("\033[90m<think>", styled)
        self.assertIn("</think>\033[0m", styled)
        self.assertEqual(
            styled.replace("\033[90m", "").replace("\033[0m", ""),
            text,
        )

    def test_strip_trailing_decode_artifacts_keeps_valid_text(self) -> None:
        self.assertEqual(
            ck_chat._strip_trailing_decode_artifacts("Hello! \ufffd\u0141\u013a\u012c"),
            "Hello!",
        )
        self.assertEqual(
            ck_chat._strip_trailing_decode_artifacts("<think>\nReasoning\n</think>\nAnswer"),
            "<think>\nReasoning\n</think>\nAnswer",
        )

    def test_generate_reports_stop_reason_when_eos_token_hits(self) -> None:
        class FakeModel:
            has_kv_decode = False
            eos_tokens = {0}
            vocab_size = 32
            context_window = 32

            def encode(self, text: str):
                return [9]

            def is_eos_token(self, token_id: int) -> bool:
                return token_id == 0

            def forward(self, token_ids):
                return np.zeros((32,), dtype=np.float32)

            def decode(self, token_ids):
                ids = tuple(token_ids)
                if ids == (1,):
                    return "Paris"
                return ""

        sample_ids = iter([1, 0])
        orig_sample = ck_chat.sample_top_k
        try:
            ck_chat.sample_top_k = lambda *args, **kwargs: next(sample_ids)
            with redirect_stdout(io.StringIO()) as buf:
                out = ck_chat.generate(FakeModel(), "hi", max_tokens=4, show_stats=True)
        finally:
            ck_chat.sample_top_k = orig_sample

        self.assertEqual(out, "Paris")
        self.assertIn("stop: eos token 0", buf.getvalue())

    def test_greedy_speculative_decode_accepts_matching_draft_tokens(self) -> None:
        class FakeKVModel:
            has_kv_decode = True
            eos_tokens = {0}
            vocab_size = 16
            context_window = 32

            def __init__(self, argmax_tokens):
                self.argmax_tokens = list(argmax_tokens)
                self.pos = 0
                self.decode_calls = []
                self.reset_calls = 0

            def encode(self, text: str):
                return [9]

            def kv_cache_enable(self):
                return True

            def kv_cache_reset(self):
                self.pos = 0
                self.reset_calls += 1

            def set_parity_token_index(self, idx: int):
                pass

            def is_eos_token(self, token_id: int) -> bool:
                return int(token_id) == 0

            def _logits(self):
                token = self.argmax_tokens[min(self.pos, len(self.argmax_tokens) - 1)]
                out = np.zeros((self.vocab_size,), dtype=np.float32)
                out[int(token)] = 10.0
                return out

            def prefill(self, token_ids):
                self.pos = 0
                return self._logits()

            def decode_step(self, token_id: int):
                self.decode_calls.append(int(token_id))
                self.pos += 1
                return self._logits()

            def decode(self, token_ids):
                return "".join({1: "A", 2: "B", 3: "C"}.get(int(t), "") for t in token_ids)

        target = FakeKVModel([1, 2, 3, 0])
        draft = FakeKVModel([1, 2, 3, 0])
        with redirect_stdout(io.StringIO()) as buf:
            out = ck_chat.generate(
                target,
                "hi",
                max_tokens=3,
                temperature=0,
                show_stats=True,
                speculative=ck_chat.SpeculativeConfig(draft, draft_tokens=3),
            )

        self.assertEqual(out, "ABC")
        self.assertEqual(target.decode_calls, [1, 2, 3])
        self.assertIn("3/3 accepted", buf.getvalue())

    def test_greedy_speculative_decode_rejects_mismatch_and_uses_verifier_token(self) -> None:
        class FakeKVModel:
            has_kv_decode = True
            eos_tokens = {0}
            vocab_size = 16
            context_window = 32

            def __init__(self, argmax_tokens):
                self.argmax_tokens = list(argmax_tokens)
                self.pos = 0

            def encode(self, text: str):
                return [9]

            def kv_cache_enable(self):
                return True

            def kv_cache_reset(self):
                self.pos = 0

            def set_parity_token_index(self, idx: int):
                pass

            def is_eos_token(self, token_id: int) -> bool:
                return int(token_id) == 0

            def _logits(self):
                token = self.argmax_tokens[min(self.pos, len(self.argmax_tokens) - 1)]
                out = np.zeros((self.vocab_size,), dtype=np.float32)
                out[int(token)] = 10.0
                return out

            def prefill(self, token_ids):
                self.pos = 0
                return self._logits()

            def decode_step(self, token_id: int):
                self.pos += 1
                return self._logits()

            def decode(self, token_ids):
                return "".join({1: "A", 2: "B", 5: "V"}.get(int(t), "") for t in token_ids)

        target = FakeKVModel([1, 5, 0])
        draft = FakeKVModel([1, 2, 0])
        with redirect_stdout(io.StringIO()) as buf:
            out = ck_chat.generate(
                target,
                "hi",
                max_tokens=2,
                temperature=0,
                show_stats=True,
                speculative=ck_chat.SpeculativeConfig(draft, draft_tokens=2),
            )

        self.assertEqual(out, "AV")
        self.assertIn("1/2 accepted", buf.getvalue())
        self.assertIn("1 rejected", buf.getvalue())

    def test_decode_preserves_special_tokens_on_python_tokenizer_path(self) -> None:
        class FakeTokenizer:
            def __init__(self):
                self.calls = []

            def decode(self, token_ids, skip_special_tokens=True):
                self.calls.append((list(token_ids), bool(skip_special_tokens)))
                return "<|im_end|>" if not skip_special_tokens else ""

        model = ck_chat.CKModel("/tmp/unused")
        model.use_c_tokenizer = False
        model.tokenizer = FakeTokenizer()

        text = model.decode([123, 456], skip_special_tokens=False)

        self.assertEqual(text, "<|im_end|>")
        self.assertEqual(model.tokenizer.calls, [([123, 456], False)])

    def test_apply_python_tokenizer_contract_updates_direct_tokenizer_object(self) -> None:
        class FakeTokenizer:
            def __init__(self):
                self.add_bos = False
                self.add_eos = False
                self.add_space_prefix = False
                self.bos_id = -1
                self.eos_id = -1
                self.unk_id = -1
                self.pad_id = -1
                self.model_type = "unknown"

        model = ck_chat.CKModel("/tmp/unused")
        model.tokenizer = FakeTokenizer()

        with mock.patch.object(
            model,
            "_load_tokenizer_contract",
            return_value={
                "add_bos_token": True,
                "add_eos_token": True,
                "add_space_prefix": True,
                "bos_token_id": 11,
                "eos_token_id": 12,
                "unk_token_id": 13,
                "pad_token_id": 14,
                "tokenizer_model": "gpt2",
            },
        ):
            model._apply_python_tokenizer_contract()

        self.assertTrue(model.tokenizer.add_bos)
        self.assertTrue(model.tokenizer.add_eos)
        self.assertTrue(model.tokenizer.add_space_prefix)
        self.assertEqual(model.tokenizer.bos_id, 11)
        self.assertEqual(model.tokenizer.eos_id, 12)
        self.assertEqual(model.tokenizer.unk_id, 13)
        self.assertEqual(model.tokenizer.pad_id, 14)
        self.assertEqual(model.tokenizer.model_type, "gpt2")

    def test_load_python_tokenizer_prefers_exported_tokenizer_json_contract_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_tok_path_") as td:
            root = Path(td)
            run_dir = root / "run"
            run_dir.mkdir(parents=True)
            contract_tok = root / "tokenizer.json"
            contract_tok.write_text(json.dumps({"model": {"type": "BPE", "vocab": {}, "merges": []}}), encoding="utf-8")
            (run_dir / "config.json").write_text(
                json.dumps({"tokenizer_contract": {"tokenizer_type": "bpe", "path": str(contract_tok)}}),
                encoding="utf-8",
            )

            model = ck_chat.CKModel(str(run_dir))
            sentinel = object()

            with mock.patch.object(ck_chat, "HF_TOKENIZER_AVAILABLE", True):
                with mock.patch.object(ck_chat, "Tokenizer", create=True) as tok_mod:
                    tok_mod.from_file.return_value = sentinel
                    with mock.patch.object(model, "_apply_python_tokenizer_contract") as apply_mock:
                        ok = model._load_python_tokenizer()

            self.assertTrue(ok)
            tok_mod.from_file.assert_called_once_with(str(contract_tok))
            apply_mock.assert_called_once()
            self.assertIs(model.tokenizer, sentinel)

    def test_v8_build_runtime_discovers_tokenizer_in_source_model_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_v8_build_tok_") as td:
            model_root = Path(td) / "model"
            run_dir = model_root / ".ck_build_v8"
            run_dir.mkdir(parents=True)
            tokenizer_json = model_root / "tokenizer.json"
            tokenizer_json.write_text(
                json.dumps({"model": {"type": "BPE", "vocab": {}, "merges": []}}),
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            sentinel = object()

            with mock.patch.object(ck_chat, "HF_TOKENIZER_AVAILABLE", True):
                with mock.patch.object(ck_chat, "Tokenizer", create=True) as tok_mod:
                    tok_mod.from_file.return_value = sentinel
                    with mock.patch.object(model, "_apply_python_tokenizer_contract"):
                        ok = model._load_python_tokenizer()

            self.assertTrue(ok)
            tok_mod.from_file.assert_called_once_with(str(tokenizer_json))
            self.assertIs(model.tokenizer, sentinel)

    def test_custom_transformers_tokenizer_loads_locally_from_source_model_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_chat_custom_tok_") as td:
            model_root = Path(td) / "model"
            run_dir = model_root / ".ck_build_v8"
            run_dir.mkdir(parents=True)
            (model_root / "tokenizer_config.json").write_text(
                json.dumps({"auto_map": {"AutoTokenizer": ["tokenization_custom.Custom", None]}}),
                encoding="utf-8",
            )
            (model_root / "tiktoken.model").write_bytes(b"fixture")
            (model_root / "tokenization_custom.py").write_text(
                "# local custom tokenizer fixture\n",
                encoding="utf-8",
            )
            model = ck_chat.CKModel(str(run_dir))
            sentinel = mock.Mock()
            sentinel.chat_template = "exported-template"
            auto_tokenizer = mock.Mock()
            auto_tokenizer.from_pretrained.return_value = sentinel

            with mock.patch.object(
                ck_chat,
                "_load_transformers_auto_tokenizer",
                return_value=auto_tokenizer,
            ), mock.patch.object(model, "_apply_python_tokenizer_contract"):
                ok = model._load_python_tokenizer()

            self.assertTrue(ok)
            self.assertIs(model.tokenizer, sentinel)
            auto_tokenizer.from_pretrained.assert_called_once_with(
                str(model_root),
                trust_remote_code=True,
                local_files_only=True,
            )

    def test_chat_template_marker_support_accepts_atomic_eos_marker_even_if_decode_hides_it(self) -> None:
        model = ck_chat.CKModel("/tmp/unused")
        model.use_c_tokenizer = True
        model.use_chat_template = True
        model.chat_contract = {
            "template_markers": ["<start_of_turn>", "<end_of_turn>"],
        }

        lookup_map = {
            b"<start_of_turn>": 105,
            b"<end_of_turn>": 106,
        }
        model.lib = mock.Mock()
        model.lib.ck_model_lookup_token.side_effect = lambda raw: lookup_map.get(raw, -1)
        model.encode = mock.Mock(side_effect=lambda text: [2, 105] if "start" in text else [2, 106])
        model.decode = mock.Mock(return_value="")

        with mock.patch.object(
            model,
            "_load_tokenizer_contract",
            return_value={"add_bos_token": True, "bos_token_id": 2, "add_eos_token": False},
        ):
            self.assertTrue(model._chat_template_markers_supported())

        self.assertEqual(model.decode.call_count, 0)

    def test_ensure_interactive_stdin_reattaches_dev_tty_when_stdin_is_not_tty(self) -> None:
        class FakeStream:
            closed = False
            encoding = "utf-8"

            def isatty(self):
                return False

        class FakeTTY:
            closed = False

            def isatty(self):
                return True

            def close(self):
                self.closed = True

        fake_tty = FakeTTY()
        with mock.patch.object(ck_chat.sys, "stdin", FakeStream()):
            with mock.patch("builtins.open", return_value=fake_tty) as open_mock:
                attached = ck_chat._ensure_interactive_stdin()
                current_stdin = ck_chat.sys.stdin

        self.assertIs(attached, fake_tty)
        self.assertIs(current_stdin, fake_tty)
        open_mock.assert_called_once()

    def test_ensure_interactive_stdin_returns_none_without_tty(self) -> None:
        class FakeStream:
            closed = False
            encoding = "utf-8"

            def isatty(self):
                return False

        original_stdin = ck_chat.sys.stdin
        with mock.patch.object(ck_chat.sys, "stdin", FakeStream()):
            with mock.patch("builtins.open", side_effect=OSError("no tty")):
                attached = ck_chat._ensure_interactive_stdin()
                current_stdin = ck_chat.sys.stdin

        self.assertIsNone(attached)
        self.assertIsNot(current_stdin, original_stdin)

    def test_detect_eos_tokens_prefers_explicit_manifest_eos_for_gemma(self) -> None:
        class FakeTokenizer:
            def get_vocab(self):
                return {}

        model = ck_chat.CKModel("/tmp/unused")
        model.use_c_tokenizer = False
        model.tokenizer = FakeTokenizer()
        model.vocab_size = 262144
        model.chat_template_mode = "gemma"

        with mock.patch.object(model, "_load_tokenizer_contract", return_value={"eos_token_id": 106}):
            with mock.patch.object(model, "_load_model_meta", return_value={"model_type": "gemma3"}):
                with mock.patch.object(model, "_lookup_single_token_id", return_value=-1):
                    model._detect_eos_tokens()

        self.assertIn(106, model.eos_tokens)
        self.assertNotIn(151643, model.eos_tokens)
        self.assertNotIn(151645, model.eos_tokens)


if __name__ == "__main__":
    unittest.main()
