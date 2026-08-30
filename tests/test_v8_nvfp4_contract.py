from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "version" / "v8" / "scripts" / "convert_safetensors_to_bump_v8.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_safetensors_nvfp4_test", CONVERTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expected_pack(weight: np.ndarray, scale_codes: np.ndarray) -> bytes:
    rows, packed_cols = weight.shape
    blocks16 = packed_cols // 8
    packed16 = weight.reshape(rows, blocks16, 8)
    values = np.stack((packed16 & 0x0F, packed16 >> 4), axis=-1).reshape(
        rows, blocks16, 16
    )
    pairs = values[:, :, :8] | (values[:, :, 8:] << 4)
    blocks64 = blocks16 // 4
    return np.concatenate(
        (
            (scale_codes & 0x7F).reshape(rows, blocks64, 4),
            pairs.reshape(rows, blocks64, 32),
        ),
        axis=-1,
    ).astype(np.uint8).tobytes()


def test_nvfp4_repack_preserves_packed_values_and_scale_codes() -> None:
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks float8_e4m3fn storage")
    converter = _load_converter()

    weight = np.arange(64, dtype=np.uint8).reshape(2, 32)
    scale_codes = np.array(
        [[0x08, 0x17, 0x38, 0x47], [0x01, 0x10, 0x27, 0x6E]], dtype=np.uint8
    )
    weight_tensor = torch.from_numpy(weight.copy())
    scale_tensor = torch.from_numpy(scale_codes.copy()).view(torch.float8_e4m3fn)

    data, shape = converter._nvfp4_pack_tensors(weight_tensor, scale_tensor)

    assert shape == [2, 64]
    assert data == _expected_pack(weight, scale_codes)
    assert converter._dtype_code("nvfp4") == 14


def test_nvfp4_transform_size_and_reciprocal_scale(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks float8_e4m3fn storage")
    converter = _load_converter()

    weight = torch.arange(32, dtype=torch.uint8).reshape(1, 32)
    scale_codes = torch.tensor([0x08, 0x17, 0x38, 0x47], dtype=torch.uint8)
    scale = scale_codes.view(torch.float8_e4m3fn).reshape(1, 4)
    safetensors.save_file(
        {
            "projection.weight_packed": weight,
            "projection.weight_scale": scale,
            "projection.weight_global_scale": torch.tensor(4.0, dtype=torch.bfloat16),
        },
        tmp_path / "model.safetensors",
    )
    headers = converter._load_safetensors_headers(tmp_path)

    packed_ref = converter.TensorRef(
        "projection", ("projection.weight_packed", "projection.weight_scale"),
        transform="nvfp4_pack",
    )
    assert converter._entry_size_from_header(
        packed_ref, headers, "preserve"
    ) == ("nvfp4", 36, [1, 64])
    output = io.BytesIO()
    writer = converter.HashingWriter(output)
    assert converter._write_ref(
        writer, tmp_path, headers, packed_ref, "preserve", "test"
    ) == ("nvfp4", 36, [1, 64])
    assert output.getvalue() == _expected_pack(
        weight.numpy(), scale_codes.numpy().reshape(1, 4)
    )

    global_ref = converter.TensorRef(
        "projection.scale", ("projection.weight_global_scale",),
        transform="reciprocal_fp32",
    )
    output = io.BytesIO()
    writer = converter.HashingWriter(output)
    assert converter._write_ref(
        writer, tmp_path, headers, global_ref, "preserve", "test"
    ) == ("fp32", 4, [1])
    assert np.frombuffer(output.getvalue(), dtype=np.float32).tolist() == [0.25]


def test_command_a_plus_contract_maps_packed_experts_declaratively() -> None:
    converter = _load_converter()
    shard = Path("model.safetensors")

    def header(name: str, dtype: str, shape: list[int]):
        return converter.HeaderTensor(name, dtype, shape, shard)

    root = "model.language_model"
    layer = f"{root}.layers.0"
    entries = [
        header(f"{root}.embed_tokens.weight", "BF16", [128, 64]),
        header(f"{layer}.input_layernorm.weight", "BF16", [64]),
        header(f"{layer}.input_layernorm.bias", "BF16", [64]),
        header(f"{layer}.self_attn.q_proj.weight", "BF16", [64, 64]),
        header(f"{layer}.self_attn.k_proj.weight", "BF16", [16, 64]),
        header(f"{layer}.self_attn.v_proj.weight", "BF16", [16, 64]),
        header(f"{layer}.self_attn.o_proj.weight", "BF16", [64, 64]),
        header(f"{layer}.self_attn.o_proj.bias", "BF16", [64]),
        header(f"{layer}.self_attn.rotary_emb.inv_freq", "F32", [32]),
        header(f"{layer}.mlp.gate.weight", "BF16", [2, 64]),
        header(f"{root}.norm.weight", "BF16", [64]),
        header(f"{root}.norm.bias", "BF16", [64]),
        header("model.vision_tower.patch.weight", "BF16", [8, 8]),
    ]
    for expert in range(2):
        for projection, rows, cols in (
            ("gate_proj", 64, 64), ("up_proj", 64, 64), ("down_proj", 64, 64)
        ):
            prefix = f"{layer}.mlp.experts.{expert}.{projection}"
            entries.extend([
                header(f"{prefix}.weight_packed", "U8", [rows, cols // 2]),
                header(f"{prefix}.weight_scale", "F8_E4M3", [rows, cols // 16]),
                header(f"{prefix}.weight_global_scale", "BF16", []),
                header(f"{prefix}.input_global_scale", "BF16", []),
            ])
    for projection, rows, cols in (
        ("gate_proj", 128, 64), ("up_proj", 128, 64), ("down_proj", 64, 128)
    ):
        prefix = f"{layer}.mlp.shared_experts.{projection}"
        entries.extend([
            header(f"{prefix}.weight_packed", "U8", [rows, cols // 2]),
            header(f"{prefix}.weight_scale", "F8_E4M3", [rows, cols // 16]),
            header(f"{prefix}.weight_global_scale", "BF16", []),
            header(f"{prefix}.input_global_scale", "BF16", [1]),
        ])
    headers = {entry.name: entry for entry in entries}
    config = {
        "num_layers": 1,
        "embed_dim": 64,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_experts": 2,
        "num_shared_experts": 2,
    }

    refs = converter._refs_for_arch(
        "cohere_command_a_plus_text", config, headers
    )
    by_name = {ref.ck_name: ref for ref in refs}
    assert by_name["layer.0.moe_expert_gate"].transform == "nvfp4_pack"
    assert by_name["layer.0.moe_expert_gate"].shape == (2, 64, 64)
    assert by_name["layer.0.moe_expert_gate_scale"].transform == "reciprocal_fp32"
    assert by_name["layer.0.moe_shared_gate"].shape == (128, 64)
    assert by_name["layer.0.ln1_beta"].source_names == (
        f"{layer}.input_layernorm.bias",
    )
    audit = converter._build_source_audit(
        "cohere_command_a_plus_text", headers, refs
    )
    assert audit["verdict"] == "pass"
    reasons = {row["reason"] for row in audit["ignored_source_tensors"]}
    assert "native_cpu_provider_uses_dynamic_q8_0_activations" in reasons
    assert "vision_tower_not_in_decoder_artifact" in reasons


def test_command_a_plus_config_retains_moe_and_average_contract(tmp_path: Path) -> None:
    converter = _load_converter()
    (tmp_path / "config.json").write_text(
        __import__("json").dumps({
            "model_type": "cohere2_vision",
            "text_config": {
                "model_type": "cohere2_moe",
                "num_hidden_layers": 4,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "vocab_size": 128,
                "max_position_embeddings": 200000,
                "sliding_window": 4096,
                "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "num_shared_experts": 4,
                "expert_selection_fn": "sigmoid",
                "norm_topk_prob": True,
                "shared_expert_combination_strategy": "average",
                "layer_norm_eps": 1e-5,
                "rope_theta": 50000,
            },
        }),
        encoding="utf-8",
    )
    config = converter._build_config(
        tmp_path, "cohere_command_a_plus_text", None
    )
    assert config["layer_kinds"] == ["moe_sliding_attention"] * 3 + ["moe_full_attention"]
    assert config["n_routed_experts"] == 8
    assert config["experts_per_tok"] == 2
    assert config["moe_shared_expert_intermediate_size"] == 512
    assert config["moe_shared_combination_scale"] == 0.5
    assert config["router_num_groups"] == 1
