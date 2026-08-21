import json
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
    resolve_qwen35_prefill_policy,
)
from build_ir_v8 import (  # type: ignore
    TEMPLATE_OP_WEIGHTS,
    _kernel_port_size_bytes,
    _required_kernel_call_scratch_bytes,
    _kernel_scratch_size_bytes,
    _resolve_body_ops_for_layer,
    _validate_lowered_activation_memory,
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
    def test_moe_conversion_owns_safe_prefill_policy(self) -> None:
        self.assertEqual(resolve_qwen35_prefill_policy(moe=True), "sequential_decode")
        self.assertEqual(resolve_qwen35_prefill_policy(moe=False), "batched")

    def test_model_map_owns_metadata_and_all_expert_tensors(self) -> None:
        contract = gguf_ck_arch_contract("qwen35moe")
        self.assertEqual(contract["family"], "qwen35")
        self.assertEqual(contract["template"], "qwen35")
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

    def test_shared_circuit_selects_dense_or_moe_tail_mechanically(self) -> None:
        circuit_path = REPO_ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        circuit = json.loads(circuit_path.read_text(encoding="utf-8"))
        body = circuit["block_types"]["decoder"]["body"]

        dense_ops = _resolve_body_ops_for_layer(
            body,
            {"layer_kinds": ["recurrent"], "mlp_execution_mode": "dense"},
            0,
        )
        moe_ops = _resolve_body_ops_for_layer(
            body,
            {"layer_kinds": ["recurrent"], "mlp_execution_mode": "qwen35moe"},
            0,
        )

        self.assertEqual(dense_ops[-5:], [
            "post_attention_norm", "mlp_gate_up", "silu_mul", "mlp_down", "residual_add",
        ])
        self.assertEqual(moe_ops[-6:], [
            "post_attention_norm",
            "moe_router",
            "full_softmax_topk_router",
            "moe_swiglu_expert_mlp",
            "gated_shared_swiglu_expert_mlp",
            "residual_add",
        ])
        self.assertNotIn("moe_router", dense_ops)
        self.assertNotIn("mlp_gate_up", moe_ops)
        self.assertEqual(dense_ops.count("post_attention_norm"), 1)
        self.assertEqual(moe_ops.count("post_attention_norm"), 1)

    def test_shared_circuit_selects_exact_moe_provider_candidates(self) -> None:
        circuit_path = REPO_ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        kernels = json.loads(circuit_path.read_text(encoding="utf-8"))["kernels"]
        self.assertEqual(
            kernels["full_softmax_topk_router"],
            "moe_softmax_topk_router_llama_f32",
        )
        self.assertEqual(
            kernels["moe_swiglu_expert_mlp"],
            "moe_swiglu_expert_forward_q4k_q5k",
        )
        self.assertEqual(
            kernels["gated_shared_swiglu_expert_mlp"],
            "moe_swiglu_shared_forward_q8_0_gated",
        )
        self.assertEqual(
            TEMPLATE_OP_WEIGHTS["gated_shared_swiglu_expert_mlp"],
            ["moe_shared_gate", "moe_shared_up", "moe_shared_down", "moe_shared_router"],
        )

    def test_moe_tail_consumes_the_normalized_main_stream(self) -> None:
        circuit_path = REPO_ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        circuit = json.loads(circuit_path.read_text(encoding="utf-8"))
        body = circuit["block_types"]["decoder"]["body"]
        for branch in ("recurrent", "full_attention"):
            entries = {
                entry["op"]: entry
                for entry in body["ops_by_kind"][branch]
                if isinstance(entry, dict) and entry.get("op")
            }
            with self.subTest(branch=branch):
                self.assertEqual(
                    entries["moe_router"]["graph_slots"]["inputs"]["x"],
                    "normalized_stream",
                )
                self.assertEqual(
                    entries["moe_swiglu_expert_mlp"]["graph_slots"]["inputs"]["hidden"],
                    "normalized_stream",
                )
                self.assertEqual(
                    entries["gated_shared_swiglu_expert_mlp"]["graph_slots"]["inputs"]["hidden"],
                    "normalized_stream",
                )

    def test_map_owned_moe_workspace_formulas_resolve_at_real_shape(self) -> None:
        config = {
            "embed_dim": HIDDEN,
            "moe_intermediate_size": EXPERT_DIM,
            "n_routed_experts": EXPERTS,
            "experts_per_tok": TOP_K,
        }
        cases = {
            "moe_softmax_topk_router_llama_f32.json": 1024,
            "moe_swiglu_expert_forward_q4k_q5k.json": 64 * 15296,
            "moe_swiglu_shared_forward_q8_0_gated.json": 15040,
        }
        maps_dir = REPO_ROOT / "version" / "v8" / "kernel_maps"
        for filename, expected in cases.items():
            provider = json.loads((maps_dir / filename).read_text(encoding="utf-8"))
            with self.subTest(provider=provider["id"]):
                self.assertEqual(
                    _kernel_scratch_size_bytes(provider["scratch"][0], {}, config),
                    expected,
                )

    def test_moe_output_extent_resolves_from_physical_port_contract(self) -> None:
        provider_path = (
            REPO_ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "moe_swiglu_expert_forward_q4k_q5k.json"
        )
        provider = json.loads(provider_path.read_text(encoding="utf-8"))
        self.assertEqual(
            _kernel_port_size_bytes(
                provider["outputs"][0],
                {"R": 1},
                {"embed_dim": HIDDEN},
            ),
            HIDDEN * 4,
        )

    def test_required_provider_workspace_is_reserved_by_memory_planner(self) -> None:
        provider_path = (
            REPO_ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "moe_swiglu_expert_forward_q4k_q5k.json"
        )
        provider = json.loads(provider_path.read_text(encoding="utf-8"))
        config = {
            "embed_dim": HIDDEN,
            "moe_intermediate_size": EXPERT_DIM,
            "n_routed_experts": EXPERTS,
            "experts_per_tok": TOP_K,
            "context_length": 512,
        }
        self.assertEqual(
            _required_kernel_call_scratch_bytes(
                [
                    {
                        "kernel": provider["id"],
                        "params": {},
                        "scratch": provider["scratch"],
                    }
                ],
                config,
                1,
            ),
            64 * 15296,
        )

    def test_required_q5_workspace_uses_logical_projection_dimensions(self) -> None:
        provider_path = (
            REPO_ROOT
            / "version"
            / "v8"
            / "kernel_maps"
            / "gemm_nt_q5_k.json"
        )
        provider = json.loads(provider_path.read_text(encoding="utf-8"))
        self.assertEqual(
            _required_kernel_call_scratch_bytes(
                [
                    {
                        "op": "mlp_down",
                        "kernel": provider["id"],
                        "params": {},
                        "scratch": provider["scratch"],
                    }
                ],
                {
                    "embed_dim": HIDDEN,
                    "intermediate_size": EXPERT_DIM,
                    "context_length": 512,
                },
                3,
            ),
            3 * (EXPERT_DIM // 256) * 292,
        )

    def test_unresolved_required_provider_workspace_fails_planning(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "planner cannot resolve required workspace"
        ):
            _required_kernel_call_scratch_bytes(
                [
                    {
                        "kernel": "synthetic_required_workspace",
                        "scratch": [
                            {
                                "name": "workspace",
                                "size_resolution": "required",
                                "shape": ["missing_extent"],
                            }
                        ],
                    }
                ],
                {},
                1,
            )

    def test_required_moe_workspace_must_not_alias_live_output(self) -> None:
        layout = {
            "memory": {
                "arena": {"total_size": 32768},
                "activations": {
                    "size": 32768,
                    "buffers": [
                        {"name": "mlp_scratch", "offset": 0, "size": 32768},
                    ],
                },
            },
        }
        lowered = {
            "operations": [
                {
                    "op": "moe_swiglu_expert_mlp",
                    "layer": 0,
                    "kernel": "moe_swiglu_expert_forward_q4k_q5k",
                    "scratch": [
                        {
                            "name": "workspace",
                            "scratch_offset": 0,
                            "size": 15296,
                            "disjoint_from": [
                                {
                                    "kind": "output",
                                    "name": "output",
                                    "offset": 0,
                                    "size": HIDDEN * 4,
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "HARD SCRATCH ALIAS FAULT"):
            _validate_lowered_activation_memory(lowered, layout)

        lowered["operations"][0]["scratch"][0]["scratch_offset"] = HIDDEN * 4
        report = _validate_lowered_activation_memory(lowered, layout)
        self.assertEqual(report["scratch_contract_count"], 1)
        self.assertEqual(report["scratch"][0]["disjoint_port_count"], 1)

    def test_required_scratch_expression_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsupported size_bytes"):
            _kernel_scratch_size_bytes(
                {
                    "size_bytes": "unknown_size(H)",
                    "size_resolution": "required",
                },
                {},
                {"embed_dim": HIDDEN},
            )

    def test_legacy_optional_scratch_expression_remains_unresolved(self) -> None:
        self.assertIsNone(
            _kernel_scratch_size_bytes(
                {"size_bytes": "min(M, 8) * sizeof(block_q8_1)"},
                {"_m": 8},
                {},
            )
        )

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
