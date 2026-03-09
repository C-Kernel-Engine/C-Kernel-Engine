import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gguf_tokenizer import GGUFTokenizer  # noqa: E402


class GGUFTokenizerSpacePrefixTest(unittest.TestCase):
    def _build_vocab(self, add_space_prefix: bool) -> GGUFTokenizer:
        vocab = {
            "tokenizer.ggml.tokens": [
                "<unk>",
                "<s>",
                "</s>",
                "Hello",
                "\u2581Hello",
                "!",
            ],
            "tokenizer.ggml.scores": [0.0] * 6,
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.unknown_token_id": 0,
            "tokenizer.ggml.padding_token_id": 0,
            "tokenizer.ggml.add_bos_token": True,
            "tokenizer.ggml.add_eos_token": False,
            "tokenizer.ggml.add_space_prefix": add_space_prefix,
        }
        return GGUFTokenizer(vocab)

    def test_encode_uses_sentencepiece_dummy_prefix_when_enabled(self) -> None:
        tok = self._build_vocab(add_space_prefix=True)
        self.assertEqual(tok.encode("Hello!"), [1, 4, 5])

    def test_encode_leaves_bare_token_when_space_prefix_disabled(self) -> None:
        tok = self._build_vocab(add_space_prefix=False)
        self.assertEqual(tok.encode("Hello!"), [1, 3, 5])

    def test_legacy_json_roundtrip_preserves_space_prefix_flag(self) -> None:
        tok = self._build_vocab(add_space_prefix=True)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vocab.json"
            tok.save(str(path))
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(data["add_space_prefix"])

            reloaded = GGUFTokenizer.from_json(str(path))
            self.assertTrue(reloaded.add_space_prefix)
            self.assertEqual(reloaded.encode("Hello!"), [1, 4, 5])


if __name__ == "__main__":
    unittest.main()
