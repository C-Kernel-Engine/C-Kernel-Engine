#!/usr/bin/env python3
import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "version" / "v7" / "scripts" / "parity") not in sys.path:
    sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts" / "parity"))

import ck_chat  # type: ignore
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

    def test_qwen35_auto_mode_uses_qwen35_chat_template(self) -> None:
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
            self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

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


if __name__ == "__main__":
    unittest.main()
