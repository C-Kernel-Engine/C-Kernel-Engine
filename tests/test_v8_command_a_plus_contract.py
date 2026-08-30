#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_ir_v8  # type: ignore


def _tiny_manifest() -> dict:
    entries: list[dict] = []
    offset = 0

    def add(name: str, dtype: str, shape: list[int], nbytes: int | None = None) -> None:
        nonlocal offset
        elements = 1
        for dim in shape:
            elements *= dim
        if nbytes is None:
            nbytes = elements * (4 if dtype == "fp32" else 2)
        entries.append({
            "name": name,
            "dtype": dtype,
            "offset": offset,
            "file_offset": offset,
            "size": nbytes,
            "nbytes": nbytes,
            "shape": shape,
        })
        offset += nbytes

    hidden = 256
    intermediate = 256
    shared_intermediate = 512
    experts = 4
    add("token_emb", "bf16", [512, hidden])
    for layer in range(2):
        add(f"layer.{layer}.ln1_gamma", "fp32", [hidden])
        add(f"layer.{layer}.ln1_beta", "fp32", [hidden])
        add(f"layer.{layer}.wq", "bf16", [512, hidden])
        add(f"layer.{layer}.wk", "bf16", [64, hidden])
        add(f"layer.{layer}.wv", "bf16", [64, hidden])
        add(f"layer.{layer}.wo", "bf16", [hidden, 512])
        add(f"layer.{layer}.bo", "fp32", [hidden])
        add(f"layer.{layer}.moe_router", "bf16", [experts, hidden])
        add(f"layer.{layer}.moe_router_bias", "fp32", [experts])
        for projection, shape in (
            ("moe_expert_gate", [experts, intermediate, hidden]),
            ("moe_expert_up", [experts, intermediate, hidden]),
            ("moe_expert_down", [experts, hidden, intermediate]),
        ):
            add(f"layer.{layer}.{projection}", "nvfp4", shape, 36 * (experts * intermediate * hidden // 64))
            add(f"layer.{layer}.{projection}_scale", "fp32", [experts])
        for projection, shape in (
            ("moe_shared_gate", [shared_intermediate, hidden]),
            ("moe_shared_up", [shared_intermediate, hidden]),
            ("moe_shared_down", [hidden, shared_intermediate]),
        ):
            add(f"layer.{layer}.{projection}", "nvfp4", shape, 36 * (shared_intermediate * hidden // 64))
            add(f"layer.{layer}.{projection}_scale", "fp32", [1])
    add("final_ln_weight", "fp32", [hidden])
    add("final_ln_bias", "fp32", [hidden])

    config = {
        "model": "cohere_command_a_plus_text",
        "arch": "cohere_command_a_plus_text",
        "num_layers": 2,
        "embed_dim": hidden,
        "attn_out_dim": 512,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 64,
        "v_head_dim": 64,
        "intermediate_size": intermediate,
        "moe_intermediate_size": intermediate,
        "moe_shared_expert_intermediate_size": shared_intermediate,
        "n_routed_experts": experts,
        "num_experts": experts,
        "experts_per_tok": 2,
        "num_experts_per_tok": 2,
        "router_num_groups": 1,
        "router_topk_group": 1,
        "router_norm_topk_prob": 1,
        "routed_scaling_factor": 1.0,
        "moe_shared_combination_scale": 0.5,
        "context_length": 64,
        "max_seq_len": 64,
        "vocab_size": 512,
        "rope_theta": 50000.0,
        "rotary_dim": 64,
        "rms_eps": 1e-5,
        "rms_norm_eps": 1e-5,
        "sliding_window": 16,
        "logit_scale": 1.0,
        "layer_kinds": ["moe_sliding_attention", "moe_full_attention"],
        "tie_word_embeddings": True,
    }
    return {
        "config": config,
        "entries": entries,
        "template": build_ir_v8._load_builtin_template_doc("cohere_command_a_plus_text"),
        "quant_summary": {
            "token_emb": "bf16",
            "lm_head": "bf16",
            **{
                f"layer.{layer}": {
                    "wq": "bf16",
                    "wk": "bf16",
                    "wv": "bf16",
                    "wo": "bf16",
                    "moe_router": "bf16",
                    "moe_expert_gate": "nvfp4",
                    "moe_expert_up": "nvfp4",
                    "moe_expert_down": "nvfp4",
                    "moe_shared_gate": "nvfp4",
                    "moe_shared_up": "nvfp4",
                    "moe_shared_down": "nvfp4",
                }
                for layer in range(2)
            },
        },
    }


def test_circuit_owns_command_a_plus_algebra() -> None:
    circuit = build_ir_v8._load_builtin_template_doc("cohere_command_a_plus_text")
    assert circuit["contract"]["block_contract"]["norm_type"] == "layernorm"
    assert circuit["contract"]["attention_contract"]["rope_type"] == "rope_on_sliding_layers_only"
    assert circuit["kernels"]["moe_swiglu_expert_mlp"] == "moe_swiglu_expert_forward_nvfp4"
    assert circuit["kernels"]["shared_swiglu_expert_mlp"] == "moe_swiglu_shared_forward_nvfp4"

    kinds = circuit["block_types"]["decoder"]["body"]["ops_by_kind"]
    sliding = [op["op"] if isinstance(op, dict) else op for op in kinds["moe_sliding_attention"]]
    full = [op["op"] if isinstance(op, dict) else op for op in kinds["moe_full_attention"]]
    assert "rope_qk" in sliding
    assert "rope_qk" not in full
    for ops in (sliding, full):
        assert ops.index("moe_swiglu_expert_mlp") < ops.index("shared_swiglu_expert_mlp")
        assert ops[-2:] == ["residual_add", "residual_add"]


def test_native_nvfp4_providers_survive_call_lowering() -> None:
    manifest = _tiny_manifest()
    registry = build_ir_v8.load_kernel_registry()
    ir1 = build_ir_v8.build_ir1_direct(
        manifest,
        ROOT / "tests" / "command_a_plus.synthetic.json",
        mode="decode",
    )
    lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, manifest, "decode")
    layout = build_ir_v8.generate_memory_layout(
        lower1, manifest, registry, mode="decode", context_len=8
    )
    lower2 = build_ir_v8.generate_ir_lower_2(
        lower1, layout, manifest, registry, mode="decode"
    )
    call_ir = build_ir_v8.generate_ir_lower_3(lower2, "decode")

    layer_zero = [op for op in call_ir["operations"] if op.get("layer") == 0]
    for projection in ("q_proj", "k_proj", "v_proj"):
        call = next(op for op in layer_zero if op["op"] == projection)
        activation = next(arg for arg in call["args"] if arg["name"] == "x")
        assert activation["buffer_ref"] == "normalized_input"
    routed = next(op for op in layer_zero if op["op"] == "moe_swiglu_expert_mlp")
    shared = next(op for op in layer_zero if op["op"] == "shared_swiglu_expert_mlp")
    assert routed["function"] == "moe_swiglu_expert_forward_nvfp4_workspace"
    assert shared["function"] == "moe_swiglu_shared_forward_nvfp4_workspace"
    assert any(arg["name"] == "expert_gate_scales" for arg in routed["args"])
    assert any(arg["name"] == "shared_gate_scale" for arg in shared["args"])
    combination = next(arg for arg in shared["args"] if arg["name"] == "combination_scale")
    assert combination["expr"] == "0.5"
    shared_workspace = next(arg for arg in shared["args"] if arg["name"] == "workspace_bytes")
    assert int(shared_workspace["expr"]) == 6016
