#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from gguf_tokenizer import GGUFTokenizer  # type: ignore


class GGUFTokenizerTests(unittest.TestCase):
    def test_bytelevel_decode_restores_newlines_and_spaces(self) -> None:
        tok = GGUFTokenizer(
            {
                "tokenizer.ggml.tokens": ["<pad>", "Ċ", "Ġhello", "<|im_end|>"],
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.bos_token_id": 99,
                "tokenizer.ggml.eos_token_id": 100,
                "tokenizer.ggml.unknown_token_id": 101,
                "tokenizer.ggml.padding_token_id": 0,
                "tokenizer.ggml.special_token_ids": [3],
            }
        )

        self.assertEqual(tok.decode([1, 2]), "\n hello")
        self.assertEqual(tok.decode([1, 2, 3], skip_special=True), "\n hello")
        self.assertEqual(tok.decode([1, 2, 3], skip_special=False), "\n hello<|im_end|>")

    def test_bytelevel_encode_restores_newline_token(self) -> None:
        tok = GGUFTokenizer(
            {
                "tokenizer.ggml.tokens": ["!", "Ċ", "Hello", "<|im_end|>"],
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.bos_token_id": 99,
                "tokenizer.ggml.eos_token_id": 100,
                "tokenizer.ggml.unknown_token_id": 101,
                "tokenizer.ggml.padding_token_id": 102,
                "tokenizer.ggml.add_bos_token": False,
            }
        )

        self.assertEqual(tok.encode("\n", add_bos=False), [1])

    def test_from_json_bytelevel_disables_implicit_bos(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gguf_tok_json_") as td:
            path = Path(td) / "tokenizer.json"
            path.write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "BPE",
                            "vocab": {
                                "!": 0,
                                "H": 1,
                                "e": 2,
                                "l": 3,
                                "o": 4,
                                "Ċ": 5,
                            },
                        },
                        "added_tokens": [
                            {"id": 10, "content": "<|im_start|>", "special": True},
                            {"id": 11, "content": "<|im_end|>", "special": True},
                        ],
                        "post_processor": {
                            "type": "ByteLevel",
                            "add_prefix_space": False,
                            "trim_offsets": False,
                            "use_regex": False,
                        },
                        "decoder": {
                            "type": "ByteLevel",
                            "add_prefix_space": False,
                            "trim_offsets": False,
                            "use_regex": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            tok = GGUFTokenizer.from_json(str(path))

        self.assertEqual(tok.model_type, "gpt2")
        self.assertFalse(tok.add_bos)
        self.assertFalse(tok.add_eos)

    def test_from_json_templateprocessing_detects_bos_and_eos(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gguf_tok_template_") as td:
            path = Path(td) / "tokenizer.json"
            path.write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "BPE",
                            "vocab": {
                                "H": 0,
                            },
                        },
                        "added_tokens": [
                            {"id": 10, "content": "<|im_start|>", "special": True},
                            {"id": 11, "content": "<|im_end|>", "special": True},
                        ],
                        "post_processor": {
                            "type": "TemplateProcessing",
                            "single": [
                                {"id": 10, "type": "SpecialToken", "pattern": {"id": "<|im_start|>"}},
                                {"id": "[[ID]]", "type": "TokenType", "pattern": {"id": "[[ID]]"}},
                                {"id": 11, "type": "SpecialToken", "pattern": {"id": "<|im_end|>"}},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            tok = GGUFTokenizer.from_json(str(path))

        self.assertTrue(tok.add_bos)
        self.assertTrue(tok.add_eos)
        self.assertEqual(tok.bos_id, 10)
        self.assertEqual(tok.eos_id, 11)


if __name__ == "__main__":
    unittest.main()
