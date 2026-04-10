import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str((REPO_ROOT / "version" / "v8" / "scripts").resolve()))

from scripts.chat_contract import build_chat_contract, load_template_chat_contract  # type: ignore
from convert_gguf_to_bump_v8 import TensorInfo, classify_layer_contract, describe_layer_contract  # type: ignore


def _tensor(name: str, dims: tuple[int, ...]) -> TensorInfo:
    return TensorInfo(name=name, dims=dims, ggml_type=12, offset=0)


class V8Gemma4ScaffoldTests(unittest.TestCase):
    def test_load_template_chat_contract_gemma4(self) -> None:
        contract = load_template_chat_contract("gemma4")
        self.assertIsNotNone(contract)
        self.assertEqual(contract["turn_prefix"], "<|turn>{role}\n")
        self.assertEqual(contract["turn_suffix"], "<turn|>\n")
        self.assertEqual(contract["assistant_generation_prefix"], "<|turn>model\n")
        self.assertEqual(contract["token_stop_markers"], ["<turn|>"])

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


if __name__ == "__main__":
    unittest.main()
