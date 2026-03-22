#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONVERT_PATH = ROOT / "version" / "v7" / "scripts" / "convert_gguf_to_bump_v7.py"
BUILD_IR_PATH = ROOT / "version" / "v7" / "scripts" / "build_ir_v7.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


convert = _load_module("convert_gguf_to_bump_v7_for_tests", CONVERT_PATH)
build_ir = _load_module("build_ir_v7_for_tests", BUILD_IR_PATH)


def _strings_by_offsets(offsets: list[int], blob: bytes) -> list[str]:
    out: list[str] = []
    for off in offsets:
        end = blob.find(b"\x00", off)
        out.append(blob[off:end].decode("utf-8", errors="replace"))
    return out


class TokenizerContractResolutionTests(unittest.TestCase):
    def test_load_tokenizer_json_merges_added_tokens_into_vocab_slots(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v7_tok_json_") as td:
            path = Path(td) / "tokenizer.json"
            path.write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "BPE",
                            "vocab": {
                                "<unk>": 0,
                                "<s>": 1,
                                "</s>": 2,
                                "hello": 3,
                            },
                            "merges": [["h", "ello"]],
                        },
                        "added_tokens": [
                            {"id": 4, "content": "<|im_start|>", "special": True},
                            {"id": 5, "content": "<|im_end|>", "special": True},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            offsets, strings, merges, scores, types = convert.load_tokenizer_json(str(path), 6)

        vocab = _strings_by_offsets(offsets, strings)
        self.assertEqual(vocab[4], "<|im_start|>")
        self.assertEqual(vocab[5], "<|im_end|>")
        self.assertEqual(len(scores), 6)
        self.assertEqual(len(types), 6)
        self.assertIsInstance(merges, list)

    def test_apply_special_tokenizer_overrides_prefers_artifact_type(self) -> None:
        special = {"tokenizer_model": "llama", "bos_token_id": 2}
        contract = {"tokenizer_type": "bpe", "source": "tokenizer_json"}

        patched = convert._apply_special_tokenizer_overrides(special, contract)

        self.assertEqual(patched["tokenizer_model"], "bpe")
        self.assertEqual(patched["bos_token_id"], 2)

    def test_apply_tokenizer_contract_overrides_updates_template(self) -> None:
        template = {
            "flags": {"tokenizer": "sentencepiece"},
            "contract": {"tokenizer_contract": {"tokenizer_type": "sentencepiece"}},
        }

        patched = convert._apply_tokenizer_contract_overrides(template, "bpe")

        self.assertEqual(patched["flags"]["tokenizer"], "bpe")
        self.assertEqual(patched["contract"]["tokenizer_contract"]["tokenizer_type"], "bpe")
        self.assertEqual(template["flags"]["tokenizer"], "sentencepiece")

    def test_build_ir_prefers_explicit_tokenizer_contract(self) -> None:
        template = {"flags": {"tokenizer": "sentencepiece"}}
        config = {"tokenizer_contract": {"tokenizer_type": "bpe"}}
        manifest = {"special_tokens": {"tokenizer_model": "llama"}}

        resolved = build_ir._resolve_tokenizer_type(template, config, manifest)

        self.assertEqual(resolved, "bpe")

    def test_bpe_codegen_emits_special_id_and_marker_registration(self) -> None:
        c_code = build_ir._generate_tokenizer_c_code(
            "bpe",
            vocab_size=8,
            num_merges=0,
            special_tokens={
                "add_bos_token": True,
                "add_eos_token": False,
                "unk_token_id": 3,
                "bos_token_id": 2,
                "eos_token_id": 106,
                "pad_token_id": 0,
            },
        )

        self.assertIsNotNone(c_code)
        init = str(c_code["init"])
        self.assertIn("ck_true_bpe_set_special_ids", init)
        self.assertIn("cfg.add_bos = true", init)
        self.assertIn("<start_of_turn>", init)
        self.assertIn("<end_of_turn>", init)

    def test_bpe_codegen_registers_chat_contract_markers_for_special_matching(self) -> None:
        c_code = build_ir._generate_tokenizer_c_code(
            "bpe",
            vocab_size=8,
            num_merges=0,
            special_tokens={"eos_token_id": 151645},
            chat_contract={
                "template_markers": ["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
                "token_stop_markers": ["<|im_end|>"],
            },
        )

        self.assertIsNotNone(c_code)
        init = str(c_code["init"])
        self.assertIn('"<think>"', init)
        self.assertIn('"</think>"', init)
        self.assertEqual(init.count('"<|im_end|>"'), 1)


if __name__ == "__main__":
    unittest.main()
