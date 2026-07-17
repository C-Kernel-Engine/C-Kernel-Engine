#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "version" / "v8" / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "version" / "v8" / "scripts"))

import build_ir_v8  # type: ignore


class TestV8TokenizerCapabilityCodegen(unittest.TestCase):
    def test_bpe_distinguishes_decode_tables_from_text_encode(self) -> None:
        generated = build_ir_v8._generate_tokenizer_c_code(
            "bpe",
            vocab_size=256,
            num_merges=0,
            special_tokens={"bos_token_id": 1, "eos_token_id": 2},
            model_type="nemotron_h",
        )
        self.assertIsNotNone(generated)
        c_code = str(generated["api_functions"])
        self.assertIn("CK_EXPORT int ck_model_has_tokenizer(void)", c_code)
        self.assertIn("#ifdef W_VOCAB_OFFSETS", c_code)
        self.assertIn("CK_DISABLE_FULL_BPE_TOKENIZER", c_code)
        self.assertNotIn("CK_ENABLE_FULL_BPE_TOKENIZER", c_code)
        self.assertIn("CK_EXPORT int ck_model_can_encode_text(void)", c_code)
        self.assertIn("return (g_model && g_model->tokenizer) ? 1 : 0;", c_code)
        self.assertIn('printf("[Tokenizer] Registered special: %s -> %d\\n",', str(generated["init"]))

    def test_sentencepiece_exports_text_encode_capability(self) -> None:
        generated = build_ir_v8._generate_tokenizer_c_code(
            "sentencepiece",
            vocab_size=256,
            num_merges=0,
            special_tokens={"bos_token_id": 1, "eos_token_id": 2},
            model_type="gemma3",
        )
        self.assertIsNotNone(generated)
        c_code = str(generated["api_functions"])
        self.assertIn("CK_EXPORT int ck_model_has_tokenizer(void)", c_code)
        self.assertIn("CK_EXPORT int ck_model_can_encode_text(void)", c_code)
        self.assertIn("return (g_model && g_model->tokenizer) ? 1 : 0;", c_code)
        self.assertIn('printf("[Tokenizer] Registered special: %s -> %d\\n",', str(generated["init"]))


if __name__ == "__main__":
    unittest.main()
