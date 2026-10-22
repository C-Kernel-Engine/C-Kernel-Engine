"""Integrity and topology checks for pinned first-block PyTorch captures.

These fixtures are reference evidence. Passing this suite does not mean that
CKE has executed the Qwen3-TTS talker.
"""

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


FIXTURES = Path(__file__).resolve().parent / "fixtures/tts/qwen3_tts_0b6_customvoice"
STEMS = ("qwen3_tts_first_block_hello", "qwen3_tts_first_block_hi")
EXPECTED_WEIGHT_NAMES = {
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "mlp.down_proj.weight", "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "text_projection.linear_fc1.weight", "text_projection.linear_fc1.bias",
    "text_projection.linear_fc2.weight", "text_projection.linear_fc2.bias",
}


class FirstBlockCaptureTest(unittest.TestCase):
    def test_two_prefill_lengths_and_one_token_decode(self):
        lengths = []
        weight_hashes = []
        for stem in STEMS:
            with self.subTest(stem=stem):
                path = FIXTURES / f"{stem}.npz"
                report = json.loads((FIXTURES / f"{stem}.json").read_text())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 report["npz_sha256"])
                self.assertIn("CKE execution NOT_TESTED", report["scope"])
                self.assertEqual(report["runtime"]["dtype"], "float32")
                self.assertEqual(report["runtime"]["attention"], "eager")
                self.assertEqual(set(report["weight_sha256"]), EXPECTED_WEIGHT_NAMES)
                weight_hashes.append(report["weight_sha256"])
                with np.load(path) as captured:
                    self.assertEqual(len(report["calls"]), 2)
                    prefill = captured["call0_input"]
                    decode = captured["call1_input"]
                    lengths.append(prefill.shape[1])
                    self.assertEqual(prefill.shape[0], decode.shape[0])
                    self.assertEqual(prefill.shape[2], decode.shape[2])
                    self.assertEqual(decode.shape[1], 1)
                    for call in (0, 1):
                        prefix = f"call{call}_"
                        for name, shape in report["calls"][call].items():
                            actual = captured[prefix + name]
                            self.assertEqual(list(actual.shape), shape)
                            self.assertTrue(np.isfinite(actual).all(), name)
                        q = captured[prefix + "q_rotary"]
                        k = captured[prefix + "k_rotary"]
                        self.assertEqual(q.shape[1:], (16, prefill.shape[1] if call == 0 else 1, 128))
                        self.assertEqual(k.shape[1:], (8, prefill.shape[1] if call == 0 else 1, 128))
                        np.testing.assert_array_equal(
                            captured[prefix + "input"] + captured[prefix + "o_proj"],
                            captured[prefix + "attention_residual"])
                        np.testing.assert_array_equal(
                            captured[prefix + "attention_residual"] + captured[prefix + "down_proj"],
                            captured[prefix + "output"])
                        gate = captured[prefix + "gate_proj"]
                        up = captured[prefix + "up_proj"]
                        swiglu = (gate / (np.float32(1) + np.exp(-gate))) * up
                        self.assertLessEqual(float(np.max(np.abs(
                            swiglu - captured[prefix + "swiglu_output"]))), 5e-7)
        self.assertNotEqual(lengths[0], lengths[1])
        self.assertEqual(weight_hashes[0], weight_hashes[1])


if __name__ == "__main__":
    unittest.main()
