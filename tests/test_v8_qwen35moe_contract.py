import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "version" / "v8" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from convert_gguf_to_bump_v8 import (  # type: ignore
    GGML_TYPE_F32,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q8_0,
    GGUFError,
    TensorInfo,
    audit_qwen35moe_gguf_contract,
    gguf_ck_arch_contract,
)


HIDDEN = 2048
EXPERTS = 256
TOP_K = 8
EXPERT_DIM = 512


def tensor(name: str, dims: tuple[int, ...], dtype: int) -> TensorInfo:
    return TensorInfo(name=name, dims=dims, ggml_type=dtype, offset=0)


def fixture(layer_count: int = 4) -> tuple[dict[str, TensorInfo], dict[str, object]]:
    tensors: dict[str, TensorInfo] = {}
    common = {
        "attn_norm.weight": ((HIDDEN,), GGML_TYPE_F32),
        "post_attention_norm.weight": ((HIDDEN,), GGML_TYPE_F32),
        "ffn_gate_inp.weight": ((HIDDEN, EXPERTS), GGML_TYPE_F32),
        "ffn_gate_exps.weight": ((HIDDEN, EXPERT_DIM, EXPERTS), GGML_TYPE_Q4_K),
        "ffn_up_exps.weight": ((HIDDEN, EXPERT_DIM, EXPERTS), GGML_TYPE_Q4_K),
        "ffn_down_exps.weight": ((EXPERT_DIM, HIDDEN, EXPERTS), GGML_TYPE_Q5_K),
        "ffn_gate_inp_shexp.weight": ((HIDDEN,), GGML_TYPE_F32),
        "ffn_gate_shexp.weight": ((HIDDEN, EXPERT_DIM), GGML_TYPE_Q8_0),
        "ffn_up_shexp.weight": ((HIDDEN, EXPERT_DIM), GGML_TYPE_Q8_0),
        "ffn_down_shexp.weight": ((EXPERT_DIM, HIDDEN), GGML_TYPE_Q8_0),
    }
    recurrent = {
        "attn_qkv.weight": ((HIDDEN, 8192), GGML_TYPE_Q8_0),
        "attn_gate.weight": ((HIDDEN, 4096), GGML_TYPE_Q8_0),
        "ssm_alpha.weight": ((HIDDEN, 32), GGML_TYPE_Q8_0),
        "ssm_beta.weight": ((HIDDEN, 32), GGML_TYPE_Q8_0),
        "ssm_conv1d.weight": ((4, 8192), GGML_TYPE_F32),
        "ssm_out.weight": ((4096, HIDDEN), GGML_TYPE_Q8_0),
    }
    attention = {
        "attn_q.weight": ((HIDDEN, 8192), GGML_TYPE_Q8_0),
        "attn_k.weight": ((HIDDEN, 512), GGML_TYPE_Q8_0),
        "attn_v.weight": ((HIDDEN, 512), GGML_TYPE_Q8_0),
        "attn_output.weight": ((4096, HIDDEN), GGML_TYPE_Q8_0),
        "attn_q_norm.weight": ((256,), GGML_TYPE_F32),
        "attn_k_norm.weight": ((256,), GGML_TYPE_F32),
    }
    for layer in range(layer_count):
        layer_tensors = {**common, **(attention if (layer + 1) % 4 == 0 else recurrent)}
        for suffix, (dims, dtype) in layer_tensors.items():
            name = f"blk.{layer}.{suffix}"
            tensors[name] = tensor(name, dims, dtype)
    meta: dict[str, object] = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": layer_count,
        "qwen35moe.embedding_length": HIDDEN,
        "qwen35moe.expert_count": EXPERTS,
        "qwen35moe.expert_used_count": TOP_K,
        "qwen35moe.expert_feed_forward_length": EXPERT_DIM,
        "qwen35moe.expert_shared_feed_forward_length": EXPERT_DIM,
        "qwen35moe.full_attention_interval": 4,
    }
    return tensors, meta


class Qwen35MoeContractTests(unittest.TestCase):
    def test_model_map_owns_metadata_and_all_expert_tensors(self) -> None:
        contract = gguf_ck_arch_contract("qwen35moe")
        self.assertEqual(contract["family"], "qwen35")
        self.assertEqual(contract["metadata_map"]["expert_count"], "qwen35moe.expert_count")
        tensor_map = contract["tensor_map"]
        for suffix in (
            "ffn_gate_inp.weight",
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
            "ffn_gate_inp_shexp.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        ):
            self.assertIn(f"blk.{{L}}.{suffix}", tensor_map)

    def test_real_shape_contract_reports_quant_and_layer_cadence(self) -> None:
        tensors, meta = fixture()
        report = audit_qwen35moe_gguf_contract(tensors, meta)
        self.assertEqual(
            report["layer_kinds"],
            ["recurrent_moe", "recurrent_moe", "recurrent_moe", "full_attention_moe"],
        )
        self.assertEqual(report["experts_per_token"], 8)
        self.assertEqual(report["quant_by_role"]["expert_gate"], ["q4_k"])
        self.assertEqual(report["quant_by_role"]["expert_down"], ["q5_k"])
        self.assertEqual(report["quant_by_role"]["shared_up"], ["q8_0"])

    def test_missing_shared_gate_fails_closed(self) -> None:
        tensors, meta = fixture()
        del tensors["blk.2.ffn_gate_inp_shexp.weight"]
        with self.assertRaisesRegex(GGUFError, "does not satisfy"):
            audit_qwen35moe_gguf_contract(tensors, meta)

    def test_wrong_expert_axis_fails_closed(self) -> None:
        tensors, meta = fixture()
        name = "blk.0.ffn_gate_exps.weight"
        tensors[name] = tensor(name, (HIDDEN, EXPERT_DIM, EXPERTS - 1), GGML_TYPE_Q4_K)
        with self.assertRaisesRegex(GGUFError, "does not satisfy|expected GGUF dims"):
            audit_qwen35moe_gguf_contract(tensors, meta)

    def test_invalid_top_k_fails_closed(self) -> None:
        tensors, meta = fixture()
        meta["qwen35moe.expert_used_count"] = EXPERTS + 1
        with self.assertRaisesRegex(GGUFError, "expert_used_count"):
            audit_qwen35moe_gguf_contract(tensors, meta)


if __name__ == "__main__":
    unittest.main()
