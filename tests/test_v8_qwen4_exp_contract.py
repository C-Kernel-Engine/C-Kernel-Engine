from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_qwen4_exp_q4k_q8_0_moe_scratch_includes_worker_repacking():
    provider = json.loads(
        (
            ROOT
            / "version/v8/kernel_maps/moe_swiglu_expert_forward_q4k_q8_0.json"
        ).read_text(encoding="utf-8")
    )

    resolved = build_ir_v8._kernel_scratch_size_bytes(
        provider["scratch"][0],
        {"R": 40, "H": 2560, "I": 640, "E": 512, "K": 10},
        {},
    )

    assert resolved == 62_246_912
sys.path.insert(0, str(ROOT / "version" / "v8" / "scripts"))

import build_ir_v8  # noqa: E402
import convert_gguf_to_bump_v8 as gguf_converter  # noqa: E402
import resolve_numerical_execution_contracts_v8 as numerical_resolver  # noqa: E402
from memory_planner_v8 import plan_memory  # noqa: E402


MAPS = ROOT / "version" / "v8" / "kernel_maps"
CIRCUIT = ROOT / "version" / "v8" / "circuits" / "qwen4_exp.json"
ENGINE_HEADER = ROOT / "include" / "ckernel_engine.h"
GGUF_MODEL_MAP = ROOT / "version" / "v8" / "model_maps" / "gguf_ck_map.json"


def _load(name: str) -> dict:
    return json.loads((MAPS / name).read_text(encoding="utf-8"))


def _qwen4exp_plan_fixture() -> tuple[dict, dict]:
    contract = json.loads(GGUF_MODEL_MAP.read_text(encoding="utf-8"))["architectures"]["qwen4exp"]
    tensors: dict[str, gguf_converter.TensorInfo] = {}

    def add(name: str, dims: tuple[int, ...] = (32,)) -> None:
        tensors[name] = gguf_converter.TensorInfo(
            name=name,
            dims=dims,
            ggml_type=gguf_converter.GGML_TYPE_F32,
            offset=len(tensors) * 128,
        )

    for name in contract["required_global_tensors"]:
        add(name, (32, 64) if name.endswith("weight") else (32,))
    for layer, kind in enumerate(contract["layer_kind_pattern"]):
        suffixes = set(contract["required_common_layer_suffixes"])
        suffixes.update(contract["layer_kinds"][kind]["required_suffixes"])
        if layer == 1:
            suffixes.update(contract["required_ple_layer_suffixes"])
        for suffix in suffixes:
            add(f"blk.{layer}.{suffix}", (32, 64) if suffix.endswith("weight") else (32,))

    metadata_keys = contract["metadata_map"]
    meta = {
        metadata_keys["block_count"]: 4,
        metadata_keys["ple_layers"]: [1],
        metadata_keys["ple_ngram_size"]: 3,
        metadata_keys["ple_heads_per_ngram"]: 2,
        metadata_keys["ple_layer_multipliers"]: [11, 13, 17],
        metadata_keys["ple_head_offsets"]: [0, 10, 20, 30],
        metadata_keys["ple_head_vocab_sizes"]: [10, 10, 10, 10],
    }
    return tensors, meta


def test_qwen4exp_gguf_contract_maps_to_dedicated_circuit() -> None:
    contract = json.loads(GGUF_MODEL_MAP.read_text(encoding="utf-8"))["architectures"]["qwen4exp"]
    assert contract["template"] == "qwen4_exp"
    assert contract["runtime_arch"] == "qwen4_exp"
    assert contract["conversion_family"] == "qwen4_exp"
    assert contract["layer_kind_pattern"] == [
        "recurrent",
        "recurrent",
        "recurrent",
        "sparse_attention",
    ]
    assert contract["tensor_map"]["per_layer_token_embd.weight"] == (
        "layer.{PLE}.ple_ngram_embedding"
    )


def test_qwen4_exp_declares_phase_specific_full_attention_providers() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    kernels = circuit["kernels"]
    assert kernels["attn_prefill"] == (
        "attention_forward_causal_head_major_gqa_flash_compact_token_output"
    )
    assert kernels["attn_decode"] == (
        "attention_forward_decode_head_major_gqa_regular"
    )


def test_qwen4_exp_router_slots_match_fp32_gemm_interface() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    variants = circuit["block_types"]["decoder"]["body"]["ops_by_kind"]
    routers = [
        entry
        for ops in variants.values()
        for entry in ops
        if isinstance(entry, dict) and entry.get("op") == "moe_router"
    ]
    assert len(routers) == len(variants)
    for router in routers:
        assert router["graph_slots"] == {
            "inputs": {"A": "main_stream"},
            "outputs": {"C": "mlp_scratch"},
        }


def test_compact_prefill_attention_uses_token_extent_as_kv_stride() -> None:
    provider = _load(
        "attention_forward_causal_head_major_gqa_flash_compact_token_output.json"
    )
    assert provider["impl"]["function"] == (
        "attention_forward_causal_head_major_gqa_flash_strided_token_output"
    )
    assert [port["layout"] for port in provider["inputs"]] == [
        "head_major_contiguous",
        "head_major_contiguous",
        "head_major_contiguous",
    ]
    stride = next(
        param for param in provider["call_abi"]["params"]
        if param["name"] == "kv_stride_tokens"
    )
    assert stride["source"] == "dim:seq_len"


def test_qsa_weights_are_inactive_only_when_compression_is_disabled() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    weights = {
        "layer.3.index_q",
        "layer.3.index_k",
        "layer.3.index_q_norm",
        "layer.3.index_k_norm",
        "layer.3.attn_q_gate",
    }
    ignored = build_ir_v8._ignored_manifest_weights(
        circuit, {"indexer_compress_ratio": 0}, weights
    )
    assert set(ignored) == {
        "layer.3.index_q",
        "layer.3.index_k",
        "layer.3.index_q_norm",
        "layer.3.index_k_norm",
    }
    assert build_ir_v8._ignored_manifest_weights(
        circuit, {"indexer_compress_ratio": 4}, weights
    ) == {}


def test_qwen4exp_gguf_plan_is_complete_and_materializes_ple_metadata() -> None:
    tensors, meta = _qwen4exp_plan_fixture()
    result = gguf_converter.build_qwen4_exp_gguf_plan(tensors, meta)
    assert result["coverage"] == {
        "arch": "qwen4exp",
        "total_source_tensors": len(tensors),
        "consumed_source_tensors": len(tensors),
        "unconsumed_source_tensors": [],
        "pass": True,
        "map_path": str(GGUF_MODEL_MAP),
    }
    assert result["ple_owner_layer"] == 1
    metadata_entries = [
        entry for entry in result["plan"] if entry["source_dtype"] == "gguf_metadata_i64"
    ]
    assert [entry["name"] for entry in metadata_entries] == [
        "layer.1.ple_layer_multipliers",
        "layer.1.ple_ngram_heads_offsets",
        "layer.1.ple_ngram_heads_vocab_sizes",
    ]


def test_qwen4exp_quant_summary_groups_layer_entries() -> None:
    tensors, meta = _qwen4exp_plan_fixture()
    plan = gguf_converter.build_qwen4_exp_gguf_plan(tensors, meta)["plan"]
    summary = gguf_converter.build_qwen4_exp_quant_summary(plan, "Q4_K", "Q6_K")

    assert summary["source"] == "gguf"
    assert summary["token_emb"] == "Q4_K"
    assert summary["lm_head"] == "Q6_K"
    assert summary["layer.0"]
    assert "attn_qkv" in summary["layer.0"]


def test_qwen4exp_gguf_plan_rejects_ple_tensor_on_non_owner_layer() -> None:
    tensors, meta = _qwen4exp_plan_fixture()
    name = "blk.0.ple_key.weight"
    tensors[name] = gguf_converter.TensorInfo(
        name=name,
        dims=(32, 64),
        ggml_type=gguf_converter.GGML_TYPE_F32,
        offset=999,
    )
    with pytest.raises(gguf_converter.GGUFError, match="non-owner layer 0"):
        gguf_converter.build_qwen4_exp_gguf_plan(tensors, meta)


def test_hyper_scratch_extents_are_required_and_disjoint_when_planned() -> None:
    provider = _load("hyper_connection_mix_bf16.json")
    config = {"seq_len": 3, "embed_dim": 256, "hc_count": 4, "hc_lowrank": 320}
    sizes = [
        build_ir_v8._kernel_scratch_size_bytes(scratch, {}, config)
        for scratch in provider["scratch"]
    ]
    assert sizes == [3 * 4 * 256 * 4, 3 * 320 * 4, 3 * 4 * 256 * 4]
    assert all(scratch["size_resolution"] == "required" for scratch in provider["scratch"])

    cursor = 0
    intervals = []
    for size in sizes:
        cursor = (cursor + 63) & ~63
        intervals.append((cursor, cursor + size))
        cursor += size
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))


def test_hyper_scratch_fails_closed_without_stream_dimensions() -> None:
    scratch = _load("hyper_connection_mix_bf16.json")["scratch"][0]
    assert build_ir_v8._kernel_scratch_size_bytes(
        scratch, {}, {"seq_len": 1, "embed_dim": 256}
    ) is None


def test_ple_scratch_extents_are_required_and_disjoint_when_planned() -> None:
    provider = _load("ple_gate_conv_inject_bf16.json")
    config = {"seq_len": 3, "embed_dim": 256, "hc_count": 4}
    sizes = [
        build_ir_v8._kernel_scratch_size_bytes(scratch, {}, config)
        for scratch in provider["scratch"]
    ]
    assert sizes == [3 * 4 * 256 * 4] * 4
    assert all(scratch["size_resolution"] == "required" for scratch in provider["scratch"])

    cursor = 0
    intervals = []
    for size in sizes:
        cursor = (cursor + 63) & ~63
        intervals.append((cursor, cursor + size))
        cursor += size
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))


def test_qsa_index_scratch_extents_are_required_and_disjoint_when_planned() -> None:
    provider = _load("qsa_index_select_bf16.json")
    config = {
        "indexer_n_heads": 4,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
    }
    sizes = [
        build_ir_v8._kernel_scratch_size_bytes(scratch, {}, config)
        for scratch in provider["scratch"]
    ]
    assert sizes == [4 * 128 * 4, 128 * 4, 512 * 4, 512 * 4]
    assert all(scratch["size_resolution"] == "required" for scratch in provider["scratch"])

    cursor = 0
    intervals = []
    for size in sizes:
        cursor = (cursor + 63) & ~63
        intervals.append((cursor, cursor + size))
        cursor += size
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))


def test_qsa_index_scratch_fails_closed_without_model_dimensions() -> None:
    scratch = _load("qsa_index_select_bf16.json")["scratch"][0]
    assert build_ir_v8._kernel_scratch_size_bytes(scratch, {}, {}) is None


def test_qwen4_exp_resolves_one_gated_bf16_shared_expert_provider() -> None:
    registry = build_ir_v8.load_kernel_registry()
    provider_id = build_ir_v8.resolve_swiglu_moe_provider(
        registry,
        kernel_op="gated_shared_swiglu_expert_mlp",
        layer_quant={
            "moe_shared_gate": "bf16",
            "moe_shared_up": "bf16",
            "moe_shared_down": "bf16",
        },
        weight_prefix="moe_shared",
        mode="decode",
        prefer_q8_activation=False,
    )
    provider = _load(f"{provider_id}.json")
    assert provider["op"] == "gated_shared_swiglu_expert_mlp"
    assert provider["impl"]["function"] == (
        "moe_swiglu_shared_forward_bf16_gated_parallel_dispatch"
    )
    assert [weight["name"] for weight in provider["weights"]] == [
        "moe_shared_gate",
        "moe_shared_up",
        "moe_shared_down",
        "moe_shared_router",
    ]


@pytest.mark.parametrize(
    ("template_op", "layer_quant", "header_quant", "expected"),
    [
        (
            "hyper_mix_attn",
            {
                "attn_hyper_mix_down": "bf16",
                "attn_hyper_mix_up": "bf16",
                "attn_hyper_inject": "bf16",
            },
            {},
            "hyper_connection_mix_bf16",
        ),
        (
            "hyper_mix_attn",
            {
                "attn_hyper_mix_down": "q4_k",
                "attn_hyper_mix_up": "q5_0",
                "attn_hyper_inject": "q4_k",
            },
            {},
            "hyper_connection_mix_q4k_q5_0_q4k",
        ),
        (
            "hyper_mix_mlp",
            {
                "mlp_hyper_mix_down": "q6_k",
                "mlp_hyper_mix_up": "q5_0",
                "mlp_hyper_inject": "q4_k",
            },
            {},
            "hyper_connection_mix_q6k_q5_0_q4k",
        ),
        (
            "hyper_mix_final",
            {},
            {
                "final_hyper_mix_down": "q4_k",
                "final_hyper_mix_up": "q5_0",
            },
            "hyper_connection_final_mix_q4k_q5_0",
        ),
    ],
)
def test_hyper_connection_provider_resolution_uses_all_weight_roles(
    template_op: str,
    layer_quant: dict,
    header_quant: dict,
    expected: str,
) -> None:
    registry = build_ir_v8.load_kernel_registry()
    assert build_ir_v8.resolve_hyper_connection_provider(
        registry,
        template_op=template_op,
        layer_quant=layer_quant,
        header_quant=header_quant,
        mode="decode",
    ) == expected


def test_hyper_connection_provider_resolution_rejects_unsupported_tuple() -> None:
    registry = build_ir_v8.load_kernel_registry()
    with pytest.raises(RuntimeError, match="no hyper-connection provider"):
        build_ir_v8.resolve_hyper_connection_provider(
            registry,
            template_op="hyper_mix_attn",
            layer_quant={
                "attn_hyper_mix_down": "q4_k",
                "attn_hyper_mix_up": "q8_0",
                "attn_hyper_inject": "q4_k",
            },
            header_quant={},
            mode="decode",
        )


def test_qwen4_exp_quant_aliases_bridge_manifest_names_to_circuit_slots() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    body = circuit["block_types"]["decoder"]["body"]
    effective = build_ir_v8._apply_layer_quant_aliases(
        {
            "attn_hyper.mix_down": "q4_k",
            "attn_hyper.mix_up": "q5_0",
            "attn_hyper.inject": "q4_k",
            "mlp_hyper.mix_down": "q6_k",
            "mlp_hyper.mix_up": "q5_0",
            "mlp_hyper.inject": "q4_k",
        },
        body,
        {"circuit_layer_kinds": ["recurrent"]},
        0,
    )
    assert effective["attn_hyper_mix_down"] == "q4_k"
    assert effective["attn_hyper_mix_up"] == "q5_0"
    assert effective["attn_hyper_inject"] == "q4_k"
    assert effective["mlp_hyper_mix_down"] == "q6_k"
    assert effective["mlp_hyper_mix_up"] == "q5_0"
    assert effective["mlp_hyper_inject"] == "q4_k"


def test_header_quant_aliases_resolve_final_hyper_storage_from_entries() -> None:
    effective = build_ir_v8._apply_header_quant_aliases(
        {"source": "gguf"},
        {
            "hyper.norm": "fp32",
            "hyper.mix_down": "q4_k",
            "hyper.mix_up": "q5_0",
        },
    )
    assert effective["final_hyper_norm"] == "fp32"
    assert effective["final_hyper_mix_down"] == "q4_k"
    assert effective["final_hyper_mix_up"] == "q5_0"


def test_qwen4_exp_ple_providers_follow_weight_storage() -> None:
    template = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    assert "ple_ngram_embed" not in template["kernels"]
    assert "ple_gate_conv_inject" not in template["kernels"]

    registry = build_ir_v8.load_kernel_registry()
    assert build_ir_v8.find_kernel(
        registry,
        op="ple_ngram_embed",
        quant={"weight": "q5_0"},
        mode="decode",
        prefer_q8_activation=False,
    ) == "ple_ngram_embed_q5_0"
    assert build_ir_v8.find_kernel(
        registry,
        op="ple_gate_conv_inject",
        quant={"conv_weight": "fp16"},
        mode="decode",
        prefer_q8_activation=False,
    ) == "ple_gate_conv_inject_fp16"


def test_qwen4_exp_ple_quantizes_embedding_before_q4_projection() -> None:
    template = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    recurrent_ple = template["block_types"]["decoder"]["body"]["ops_by_kind"][
        "recurrent_ple"
    ]
    items = [item if isinstance(item, dict) else {"op": item} for item in recurrent_ple]
    by_op = {item["op"]: item for item in items}

    quantize = by_op["activation_quantize"]
    assert quantize["kernel"] == "quantize_row_q8_k"
    assert quantize["params"]["_input_dim_from_config"] == "ple_embed_dim"
    assert quantize["graph_slots"] == {
        "inputs": {"input": "ple_embedding"},
        "outputs": {"output": "ple_embedding_q8"},
    }
    assert by_op["ple_key_proj"]["graph_slots"]["inputs"]["x"] == "ple_embedding_q8"
    assert by_op["ple_value_proj"]["graph_slots"]["inputs"]["x"] == "ple_embedding_q8"
    assert template["activation_bindings"]["ple_embedding_q8"] == "main_stream_q8"


def test_decode_buffer_validation_rejects_fp32_input_to_q8_provider() -> None:
    lowered = {
        "operations": [
            {
                "op": "ple_key_proj",
                "layer": 1,
                "kernel": "gemv_q4_k_q8_k",
                "weights": {},
                "activations": {
                    "x_q8": {"buffer": "ple_embedding", "dtype": "fp32"}
                },
                "outputs": {},
            }
        ]
    }
    with pytest.raises(RuntimeError, match="HARD ACTIVATION STORAGE FAULT"):
        build_ir_v8.validate_buffer_assignments(lowered)


def test_decode_buffer_validation_rejects_q8_label_on_fp32_region() -> None:
    lowered = {
        "memory": {
            "activations": {
                "buffers": [
                    {"name": "ple_embedding", "dtype": "fp32", "size": 10240}
                ]
            }
        },
        "operations": [
            {
                "op": "ple_key_proj",
                "layer": 1,
                "kernel": "gemv_q4_k_q8_k",
                "weights": {},
                "activations": {
                    "x_q8": {"buffer": "ple_embedding", "dtype": "q8_k"}
                },
                "outputs": {},
            }
        ],
    }
    with pytest.raises(RuntimeError, match="physical='fp32'"):
        build_ir_v8.validate_buffer_assignments(lowered)


def test_qwen4_exp_projection_contract_selects_bf16_storage_by_phase() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    contracts = numerical_resolver.load_json(numerical_resolver.DEFAULT_CONTRACTS)
    kernels = numerical_resolver.load_kernel_capabilities(contracts=contracts)

    expected = {
        "prefill": (
            "decoder.projections_bf16_storage.prefill",
            "gemm_nt_bf16_pytorch_onednn_3_12_brgemm_bf16_storage",
        ),
        "decode": (
            "decoder.projections_bf16_storage.decode",
            "gemm_nt_bf16_pytorch_onednn_3_12_brgemm_bf16_storage",
        ),
    }
    for phase, (operation, kernel_id) in expected.items():
        plan = numerical_resolver.resolve_contract(
            circuit,
            contracts,
            kernels,
            operation=operation,
            phase=phase,
            source_circuit_path=CIRCUIT,
        )
        assert plan["kernel"]["id"] == kernel_id
        assert plan["contract"]["id"] == (
            "bf16_weight_bf16_input_pytorch_onednn_3_12_brgemm_bf16_output"
        )


def test_qwen4_exp_projection_provider_has_public_c_abi() -> None:
    provider = _load(
        "gemm_nt_bf16_pytorch_onednn_3_12_brgemm_bf16_storage.json"
    )
    declaration = provider["impl"]["c_declaration"]
    header = ENGINE_HEADER.read_text(encoding="utf-8")
    function = provider["impl"]["function"]
    assert function in declaration
    assert function in header


def test_qwen4_exp_logits_select_exact_pytorch_bf16_provider() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    contracts = numerical_resolver.load_json(numerical_resolver.DEFAULT_CONTRACTS)
    kernels = numerical_resolver.load_kernel_capabilities(contracts=contracts)

    for phase in ("prefill", "decode"):
        plan = numerical_resolver.resolve_contract(
            circuit,
            contracts,
            kernels,
            operation="decoder.logits_bf16_storage",
            phase=phase,
            source_circuit_path=CIRCUIT,
        )
        assert plan["kernel"]["id"] == (
            "gemm_nt_bf16_pytorch_onednn_3_12_brgemm_bf16_storage"
        )
        assert plan["contract"]["id"] == (
            "bf16_weight_bf16_input_pytorch_onednn_3_12_brgemm_bf16_output"
        )


def test_qwen4_exp_recurrent_contracts_select_pytorch_bf16_providers() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    contracts = numerical_resolver.load_json(numerical_resolver.DEFAULT_CONTRACTS)
    kernels = numerical_resolver.load_kernel_capabilities(contracts=contracts)
    expected = {
        "decoder.recurrent_dt_gate_fp32_pytorch": {
            "prefill": "recurrent_dt_gate_forward_pytorch_fp32",
            "decode": "recurrent_dt_gate_forward_pytorch_fp32",
        },
        "decoder.recurrent_ssm_conv_bf16_pytorch": {
            "prefill": "ssm_conv1d_forward_pytorch_bf16_storage",
            "decode": "ssm_conv1d_forward_pytorch_bf16_storage",
        },
        "decoder.recurrent_silu_bf16_pytorch": {
            "prefill": "recurrent_silu_forward_pytorch_bf16_storage",
            "decode": "recurrent_silu_forward_pytorch_bf16_storage",
        },
        "decoder.recurrent_qk_l2_norm_bf16_pytorch": {
            "prefill": "recurrent_qk_l2_norm_pytorch_fp32_output",
            "decode": "recurrent_qk_l2_norm_pytorch_fp32_output",
        },
        "decoder.recurrent_core_bf16_pytorch": {
            "prefill": "gated_deltanet_pytorch_grouped_bf16_prefill_forward",
            "decode": "gated_deltanet_pytorch_grouped_bf16_forward",
        },
        "decoder.recurrent_norm_gate_bf16_pytorch": {
            "prefill": "recurrent_norm_sigmoid_gate_pytorch_bf16_storage",
            "decode": "recurrent_norm_sigmoid_gate_pytorch_bf16_storage",
        },
        "decoder.moe_router_bf16_pytorch": {
            "prefill": "moe_softmax_topk_router_pytorch_bf16",
            "decode": "moe_softmax_topk_router_pytorch_bf16",
        },
    }
    for operation, phase_providers in expected.items():
        for phase, provider_id in phase_providers.items():
            plan = numerical_resolver.resolve_contract(
                circuit,
                contracts,
                kernels,
                operation=operation,
                phase=phase,
                source_circuit_path=CIRCUIT,
            )
            assert plan["kernel"]["id"] == provider_id
            provider = _load(f"{provider_id}.json")
            assert plan["kernel"]["function"] == provider["impl"]["function"]


def test_qwen4_exp_quantized_recurrent_core_selects_grouped_llama_providers() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    contracts = numerical_resolver.load_json(numerical_resolver.DEFAULT_CONTRACTS)
    kernels = numerical_resolver.load_kernel_capabilities(contracts=contracts)
    operation = "decoder.recurrent_core_llama_grouped"
    requirement = circuit["required_numerical_contracts"][operation]
    assert requirement["selector"] == {
        "config_not_equals": {"recurrent_qkv_weight_dtype": "bf16"}
    }

    expected = {
        "prefill": (
            "gated_deltanet_llama_fused_prefill_fp32_state",
            "gated_deltanet_llama_avx2_prefill_forward",
            "gated_deltanet_llama_prefill_parallel_dispatch",
        ),
        "decode": (
            "gated_deltanet_llama_avx2_decode_fp32_state",
            "gated_deltanet_llama_avx2_forward",
            "gated_deltanet_llama_avx2_parallel_forward",
        ),
    }
    for phase, (contract_id, provider_id, function) in expected.items():
        plan = numerical_resolver.resolve_contract(
            circuit,
            contracts,
            kernels,
            operation=operation,
            phase=phase,
            source_circuit_path=CIRCUIT,
        )
        assert plan["contract"]["id"] == contract_id
        assert plan["kernel"]["id"] == provider_id
        assert plan["kernel"]["function"] == function
        provider = _load(f"{provider_id}.json")
        q_input, k_input, v_input = provider["inputs"][:3]
        assert q_input["shape"][-2:] == ["G", "D"]
        assert k_input["shape"][-2:] == ["G", "D"]
        assert v_input["shape"][-2:] == ["H", "D"]


def test_qwen4_exp_quantized_recurrent_silu_selects_llama_provider() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    contracts = numerical_resolver.load_json(numerical_resolver.DEFAULT_CONTRACTS)
    kernels = numerical_resolver.load_kernel_capabilities(contracts=contracts)
    operation = "decoder.recurrent_silu_llama"
    requirement = circuit["required_numerical_contracts"][operation]
    assert requirement["selector"] == {
        "config_not_equals": {"recurrent_qkv_weight_dtype": "bf16"}
    }

    for phase in ("prefill", "decode"):
        plan = numerical_resolver.resolve_contract(
            circuit,
            contracts,
            kernels,
            operation=operation,
            phase=phase,
            source_circuit_path=CIRCUIT,
        )
        assert plan["contract"]["id"] == "recurrent_silu_llama_avx2_fp32_output"
        assert plan["kernel"]["id"] == "recurrent_silu_forward_ggml"
        assert plan["kernel"]["function"] == "recurrent_silu_forward_ggml"


def test_qwen4_exp_quantized_recurrent_norm_selects_sigmoid_llama_providers() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    contracts = numerical_resolver.load_json(numerical_resolver.DEFAULT_CONTRACTS)
    kernels = numerical_resolver.load_kernel_capabilities(contracts=contracts)
    operation = "decoder.recurrent_norm_sigmoid_gate_llama"
    requirement = circuit["required_numerical_contracts"][operation]
    assert requirement["selector"] == {
        "config_not_equals": {"recurrent_qkv_weight_dtype": "bf16"}
    }

    expected = {
        "prefill": (
            "recurrent_norm_sigmoid_gate_llama_avx2_parallel_prefill",
            "recurrent_norm_sigmoid_gate_llama_avx2_parallel_dispatch",
        ),
        "decode": (
            "recurrent_norm_sigmoid_gate_llama_avx2_forward",
            "recurrent_norm_sigmoid_gate_llama_avx2_forward",
        ),
    }
    for phase, (provider_id, function) in expected.items():
        plan = numerical_resolver.resolve_contract(
            circuit,
            contracts,
            kernels,
            operation=operation,
            phase=phase,
            source_circuit_path=CIRCUIT,
        )
        assert plan["contract"]["id"] == (
            "recurrent_norm_sigmoid_gate_llama_fp32_output"
        )
        assert plan["kernel"]["id"] == provider_id
        assert plan["kernel"]["function"] == function


def test_qwen4_exp_quantized_hyper_injection_selects_fp32_llama_provider() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    contracts = numerical_resolver.load_json(numerical_resolver.DEFAULT_CONTRACTS)
    kernels = numerical_resolver.load_kernel_capabilities(contracts=contracts)
    operation = "decoder.hyper_stream_inject_llama"
    requirement = circuit["required_numerical_contracts"][operation]
    assert requirement["selector"] == {
        "config_not_equals": {"recurrent_qkv_weight_dtype": "bf16"}
    }
    assert requirement["template_ops"] == ["hyper_inject_attn", "hyper_inject_mlp"]

    for phase in ("prefill", "decode"):
        plan = numerical_resolver.resolve_contract(
            circuit,
            contracts,
            kernels,
            operation=operation,
            phase=phase,
            source_circuit_path=CIRCUIT,
        )
        assert plan["contract"]["id"] == "hyper_stream_inject_llama_fp32_output"
        assert plan["kernel"]["id"] == "hyper_stream_inject_f32"
        assert plan["kernel"]["function"] == "hyper_stream_inject_f32"


def test_qwen4_exp_quantized_router_selects_llama_fp32_provider() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    contracts = numerical_resolver.load_json(numerical_resolver.DEFAULT_CONTRACTS)
    kernels = numerical_resolver.load_kernel_capabilities(contracts=contracts)
    operation = "decoder.moe_router_llama_fp32"
    requirement = circuit["required_numerical_contracts"][operation]
    assert requirement["selector"] == {
        "config_not_equals": {"recurrent_qkv_weight_dtype": "bf16"}
    }
    for phase in ("prefill", "decode"):
        plan = numerical_resolver.resolve_contract(
            circuit,
            contracts,
            kernels,
            operation=operation,
            phase=phase,
            source_circuit_path=CIRCUIT,
        )
        assert plan["contract"]["id"] == (
            "llama_full_softmax_topk_selected_renorm_fp32_avx2"
        )
        assert plan["kernel"]["id"] == "moe_softmax_topk_router_llama_f32"
        assert plan["kernel"]["function"] == (
            "moe_softmax_topk_router_llama_f32_workspace"
        )


def test_qwen4_exp_recurrent_contract_selectors_are_mutually_exclusive() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    requirements = circuit["required_numerical_contracts"]
    quantized = requirements["decoder.recurrent_core_llama_grouped"]["selector"]
    bf16 = requirements["decoder.recurrent_core_bf16_pytorch"]["selector"]

    assert build_ir_v8._contract_selector_matches(
        quantized, {"recurrent_qkv_weight_dtype": "q6_k"}, "recurrent_core"
    )
    assert not build_ir_v8._contract_selector_matches(
        bf16, {"recurrent_qkv_weight_dtype": "q6_k"}, "recurrent_core"
    )
    assert not build_ir_v8._contract_selector_matches(
        quantized, {"recurrent_qkv_weight_dtype": "bf16"}, "recurrent_core"
    )
    assert build_ir_v8._contract_selector_matches(
        bf16, {"recurrent_qkv_weight_dtype": "bf16"}, "recurrent_core"
    )
    quantized_norm = requirements[
        "decoder.recurrent_norm_sigmoid_gate_llama"
    ]["selector"]
    bf16_norm = requirements[
        "decoder.recurrent_norm_gate_bf16_pytorch"
    ]["selector"]
    assert build_ir_v8._contract_selector_matches(
        quantized_norm,
        {"recurrent_qkv_weight_dtype": "q6_k"},
        "recurrent_norm_gate",
    )
    assert not build_ir_v8._contract_selector_matches(
        bf16_norm,
        {"recurrent_qkv_weight_dtype": "q6_k"},
        "recurrent_norm_gate",
    )
    assert not build_ir_v8._contract_selector_matches(
        quantized_norm,
        {"recurrent_qkv_weight_dtype": "bf16"},
        "recurrent_norm_gate",
    )
    assert build_ir_v8._contract_selector_matches(
        bf16_norm,
        {"recurrent_qkv_weight_dtype": "bf16"},
        "recurrent_norm_gate",
    )
    quantized_inject = requirements[
        "decoder.hyper_stream_inject_llama"
    ]["selector"]
    assert build_ir_v8._contract_selector_matches(
        quantized_inject,
        {"recurrent_qkv_weight_dtype": "q6_k"},
        "hyper_stream_inject",
    )
    assert not build_ir_v8._contract_selector_matches(
        quantized_inject,
        {"recurrent_qkv_weight_dtype": "bf16"},
        "hyper_stream_inject",
    )
    quantized_router = requirements["decoder.moe_router_llama_fp32"]["selector"]
    bf16_router = requirements["decoder.moe_router_bf16_pytorch"]["selector"]
    assert build_ir_v8._contract_selector_matches(
        quantized_router,
        {"recurrent_qkv_weight_dtype": "q6_k"},
        "full_softmax_topk_router",
    )
    assert not build_ir_v8._contract_selector_matches(
        bf16_router,
        {"recurrent_qkv_weight_dtype": "q6_k"},
        "full_softmax_topk_router",
    )
    assert not build_ir_v8._contract_selector_matches(
        quantized_router,
        {"recurrent_qkv_weight_dtype": "bf16"},
        "full_softmax_topk_router",
    )
    assert build_ir_v8._contract_selector_matches(
        bf16_router,
        {"recurrent_qkv_weight_dtype": "bf16"},
        "full_softmax_topk_router",
    )


def test_qwen4_exp_qsa_qk_norm_selects_storage_contract() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    contracts = numerical_resolver.load_json(numerical_resolver.DEFAULT_CONTRACTS)
    kernels = numerical_resolver.load_kernel_capabilities(contracts=contracts)
    cases = (
        (
            "decoder.qk_norm_llama_fp32",
            "qk_norm_forward_llama_production",
            "rmsnorm_llama_cpu_production_fp32_output",
        ),
        (
            "decoder.qk_norm_bf16_pytorch",
            "qk_norm_forward_qwen4_pytorch_bf16_storage",
            "rmsnorm_qwen3next_pytorch_avx2_bf16_storage",
        ),
    )
    for operation, provider_id, contract_id in cases:
        for phase in ("prefill", "decode"):
            plan = numerical_resolver.resolve_contract(
                circuit,
                contracts,
                kernels,
                operation=operation,
                phase=phase,
                source_circuit_path=CIRCUIT,
            )
            assert plan["kernel"]["id"] == provider_id
            assert plan["contract"]["id"] == contract_id


def test_qwen4_exp_qsa_rope_selects_storage_contract() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    contracts = numerical_resolver.load_json(numerical_resolver.DEFAULT_CONTRACTS)
    kernels = numerical_resolver.load_kernel_capabilities(contracts=contracts)
    cases = (
        (
            "decoder.rope_qk_llama_fp32",
            "mrope_qk_text_imrope",
            "text_imrope_fp32_input_fp32_compute_fp32_output",
        ),
        (
            "decoder.rope_qk_bf16_pytorch",
            "mrope_qk_text_imrope_bf16_pytorch_storage",
            "text_imrope_bf16_input_pytorch_bf16_compute_bf16_output",
        ),
    )
    for operation, provider_id, contract_id in cases:
        for phase in ("prefill", "decode"):
            plan = numerical_resolver.resolve_contract(
                circuit,
                contracts,
                kernels,
                operation=operation,
                phase=phase,
                source_circuit_path=CIRCUIT,
            )
            assert plan["kernel"]["id"] == provider_id
            assert plan["contract"]["id"] == contract_id


def test_qwen4_exp_attention_selects_cache_storage_contract() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    quantized = circuit["required_contracts"]["decoder.attention"]
    bf16 = circuit["required_contracts"]["decoder.attention_bf16_pytorch"]
    assert quantized["selector"] == {
        "config_not_equals": {"decode_kv_cache_dtype": "bf16"}
    }
    assert bf16["selector"] == {
        "config_equals": {"decode_kv_cache_dtype": "bf16"}
    }
    for phase in ("prefill", "decode"):
        assert quantized["phases"][phase]["requires"]["tensor.kv.dtype"] == "fp16"
        assert bf16["phases"][phase]["requires"]["tensor.kv.dtype"] == "bf16"


def test_qwen4_exp_qsa_gate_selects_pytorch_bf16_storage() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    contracts = numerical_resolver.load_json(numerical_resolver.DEFAULT_CONTRACTS)
    kernels = numerical_resolver.load_kernel_capabilities(contracts=contracts)
    requirement = circuit["required_numerical_contracts"][
        "decoder.attn_gate_bf16_pytorch"
    ]
    assert requirement["selector"] == {
        "config_equals": {"decoder_norm_storage_boundary": "bf16"}
    }
    for phase in ("prefill", "decode"):
        plan = numerical_resolver.resolve_contract(
            circuit,
            contracts,
            kernels,
            operation="decoder.attn_gate_bf16_pytorch",
            phase=phase,
            source_circuit_path=CIRCUIT,
        )
        assert plan["kernel"]["id"] == (
            "attn_gate_sigmoid_mul_pytorch_bf16_storage"
        )
        assert plan["contract"]["id"] == (
            "attention_gate_pytorch_sleef_bf16_storage"
        )


def test_hyper_injection_edges_have_dedicated_activation_storage() -> None:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    bindings = circuit["activation_bindings"]
    buffers = circuit["activation_buffers"]

    assert bindings["hyper_stream"] == "hyper_stream"
    assert buffers["hyper_stream"]["shape"] == [
        {"config": "execution_extent"},
        {"mul": [{"config": "hc_count"}, {"config": "embed_dim"}]},
    ]

    for slot in ("attn_hyper_injection", "mlp_hyper_injection"):
        assert bindings[slot] == slot
        assert buffers[slot]["shape"] == [
            {"config": "execution_extent"},
            {"config": "hc_count"},
        ]
        assert bindings[slot] != "layer_input"

    ops = [
        {
            "idx": 0,
            "op": "hyper_mix_attn",
            "layer": 0,
            "dataflow": {
                "inputs": {"hyper_input": {"slot": "hyper_stream", "dtype": "fp32"}},
                "outputs": {
                    "mixed_output": {"slot": "main_stream", "dtype": "fp32"},
                    "injection_output": {
                        "slot": "attn_hyper_injection",
                        "dtype": "fp32",
                    },
                },
            },
        },
        {
            "idx": 1,
            "op": "quantize_input_0",
            "layer": 0,
            "dataflow": {
                "inputs": {"input": {"slot": "main_stream", "dtype": "fp32"}},
                "outputs": {
                    "output": {"slot": "main_stream_q8", "dtype": "q8_k"}
                },
            },
        },
        {
            "idx": 2,
            "op": "hyper_inject_attn",
            "layer": 0,
            "dataflow": {
                "inputs": {
                    "hyper_input": {"slot": "hyper_stream", "dtype": "fp32"},
                    "injection_weight": {
                        "slot": "attn_hyper_injection",
                        "dtype": "fp32",
                    },
                },
                "outputs": {"output": {"slot": "hyper_stream", "dtype": "fp32"}},
            },
        },
    ]
    assignments = plan_memory(ops, slot_bindings=bindings)
    saved_stream = assignments[2]["inputs"]["hyper_input"]["buffer"]
    injection = assignments[0]["outputs"]["injection_output"]["buffer"]
    quantized = assignments[1]["outputs"]["output"]["buffer"]
    assert saved_stream != injection
    assert saved_stream != quantized
    assert quantized == "A_MAIN_STREAM_Q8"
    assert assignments[2]["inputs"]["injection_weight"]["buffer"] == injection


def test_hyper_attention_mix_is_a_quantization_anchor() -> None:
    assert "hyper_mix_attn" in build_ir_v8.ACTIVATION_QUANTIZATION_ANCHOR_OP_NAMES
    assert "hyper_mix_attn" not in build_ir_v8.PRE_NORM_OP_NAMES


def test_qwen4_exp_ple_provider_uses_explicit_ple_eos_token() -> None:
    provider = json.loads(
        (ROOT / "version/v8/kernel_maps/ple_ngram_embed_bf16.json").read_text()
    )
    eos = next(
        param
        for param in provider["call_abi"]["params"]
        if param["name"] == "eos_token_id"
    )
    assert eos["source"] == "dim:ple_eos_token_id"
