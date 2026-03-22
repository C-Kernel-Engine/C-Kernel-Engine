#!/usr/bin/env python3
import json
import io
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


class TestCKChatRuntimeContract(unittest.TestCase):
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
            with redirect_stdout(io.StringIO()):
                out = ck_chat.generate(FakeModel(), "hi", max_tokens=3, show_stats=False)
        finally:
            ck_chat.sample_top_k = orig_sample
        self.assertEqual(out, "éA")

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
