#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v7" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import convert_gguf_to_bump_v7 as gguf


def tensor(name: str) -> gguf.TensorInfo:
    return gguf.TensorInfo(name=name, dims=(1,), ggml_type=gguf.GGML_TYPE_F32, offset=0)


class TestQwen35ContractDetection(unittest.TestCase):
    def test_qwen35_recurrent_hybrid_detection(self) -> None:
        tensors = {
            f"blk.0.{name}": tensor(f"blk.0.{name}")
            for name in (
                "attn_norm.weight",
                "post_attention_norm.weight",
                "attn_qkv.weight",
                "attn_gate.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
                "ssm_alpha.weight",
                "ssm_beta.weight",
                "ssm_conv1d.weight",
                "ssm_out.weight",
            )
        }
        self.assertEqual(gguf.classify_layer_contract(tensors, 0), "qwen35_recurrent_hybrid")
        detail = gguf.describe_layer_contract(tensors, 0, arch="qwen35")
        self.assertIsNotNone(detail)
        self.assertIn("qwen35 recurrent hybrid", detail)
        self.assertIn("DeltaNet/SSM-style", detail)

    def test_qwen35_full_attention_hybrid_detection(self) -> None:
        tensors = {
            f"blk.3.{name}": tensor(f"blk.3.{name}")
            for name in (
                "attn_norm.weight",
                "post_attention_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            )
        }
        self.assertEqual(gguf.classify_layer_contract(tensors, 3), "qwen35_full_attention_hybrid")
        detail = gguf.describe_layer_contract(tensors, 3, arch="qwen35")
        self.assertIsNotNone(detail)
        self.assertIn("qwen35 full-attention hybrid", detail)
        self.assertIn("no ffn_norm", detail)

    def test_dense_decoder_is_not_misclassified(self) -> None:
        tensors = {
            f"blk.1.{name}": tensor(f"blk.1.{name}")
            for name in (
                "attn_norm.weight",
                "ffn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            )
        }
        self.assertEqual(gguf.classify_layer_contract(tensors, 1), "dense_decoder")
        self.assertIsNone(gguf.describe_layer_contract(tensors, 1, arch="qwen3"))


if __name__ == "__main__":
    unittest.main()
