from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = ROOT / "version" / "v8" / "scripts" / "convert_safetensors_to_bump_v8.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location(
        "convert_safetensors_to_bump_v8_contract_test", CONVERTER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audio_runtime_policy_and_artifact_ownership_are_contract_driven() -> None:
    converter = _load_converter()
    contract = converter._safetensors_arch_contract("whisper_encoder")

    fp32_config: dict[str, object] = {}
    converter._apply_linear_weight_runtime_config(
        fp32_config, contract, "preserve"
    )
    assert fp32_config == {
        "audio_encoder_attention_reduction_policy": "ordered_fp32_packed_k",
        "audio_runtime_topology_policy": "all_allowed_cpus",
    }

    fp16_config: dict[str, object] = {}
    converter._apply_linear_weight_runtime_config(fp16_config, contract, "fp16")
    assert fp16_config == {
        "audio_encoder_attention_reduction_policy": "tiled_f16kv_online_softmax",
        "audio_runtime_topology_policy": "performance_core_smt_on_hybrid",
    }
    assert (
        converter._contract_ignored_source_tensor(
            contract, "model.decoder.layers.0.fc1.weight"
        )
        == "decoder_not_in_encoder_artifact"
    )
    assert (
        converter._contract_ignored_source_tensor(contract, "proj_out.weight")
        == "decoder_not_in_encoder_artifact"
    )


def test_contract_selection_does_not_depend_on_audio_architecture_name() -> None:
    converter = _load_converter()
    original = converter._SAFETENSORS_CK_MAP_CACHE
    try:
        converter._SAFETENSORS_CK_MAP_CACHE = {
            "version": 1,
            "architectures": {
                "future_audio_encoder": {
                    "runtime_config_by_linear_weight_dtype": {
                        "default": {"provider_policy": "reference"},
                        "fp16": {"provider_policy": "tiled"},
                    },
                    "ignored_source_tensors": [
                        {"prefix": "other_tower.", "reason": "separate_artifact"}
                    ],
                }
            },
        }
        contract = converter._safetensors_arch_contract("future_audio_encoder")
        config: dict[str, object] = {}
        converter._apply_linear_weight_runtime_config(config, contract, "fp16")
        assert config == {"provider_policy": "tiled"}
        assert (
            converter._ignored_source_tensor(
                "future_audio_encoder", "other_tower.layer.weight"
            )
            == "separate_artifact"
        )
    finally:
        converter._SAFETENSORS_CK_MAP_CACHE = original


def test_vision_numerics_profiles_are_explicit() -> None:
    converter = _load_converter()

    exact: dict[str, object] = {}
    converter._apply_vision_numerics_profile(
        exact, "cohere_compass_vision", "pytorch_exact"
    )
    assert exact == {
        "vision_patch_frontend": "integrated_temporal2",
        "vision_patch_projection_reduction_policy": "pytorch_onednn_conv3d_exact",
        "vision_mrope_reduction_policy": "pytorch_mkl_exact",
        "vision_layernorm_reduction_policy": "pytorch_welford_exact",
        "vision_projection_reduction_policy": "pytorch_onednn_brgemm_exact",
        "vision_attention_reduction_policy": "pytorch_amx_exact",
        "vision_projector_activation_reduction_policy": "pytorch_sleef_exact",
    }

    native: dict[str, object] = {}
    converter._apply_vision_numerics_profile(
        native, "cohere_compass_vision", "native"
    )
    assert native == {
        "vision_patch_frontend": "integrated_temporal2",
        "vision_patch_projection_reduction_policy": "native_pair_dot",
        "vision_mrope_reduction_policy": "portable_fp32_reference",
        "vision_layernorm_reduction_policy": "pytorch_welford_exact",
        "vision_projection_reduction_policy": "native_pair_dot",
        "vision_attention_reduction_policy": "portable_tiled_sdpa",
        "vision_projector_activation_reduction_policy": "portable_libm_erf",
    }

    with pytest.raises(SystemExit, match="only valid for a vision encoder"):
        converter._apply_vision_numerics_profile({}, "qwen3", "native")


def _require_torch_safetensors() -> tuple[object, object]:
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    return torch, st


def _qwen4_exp_tiny_config() -> dict[str, object]:
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": 4,
            "hidden_size": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "hc_count": 4,
            "hc_lowrank": 320,
            "num_experts": 8,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 128,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "ngram_size": 3,
            "ngram_vocab_size_base": 2048,
            "heads_per_ngram": 8,
            "ple_embed_dim": 256,
            "ple_conv_kernel_size": 4,
            "split_ngram_parts": 2,
            "ple_layer_ids": [2],
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
            "full_attention_interval": 4,
            "norm_topk_prob": True,
            "mtp_num_hidden_layers": 1,
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "max_position_embeddings": 262144,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "qwen_sparse_attention",
            ],
        },
    }


def _qwen4_exp_tiny_headers(converter) -> dict[str, object]:
    names = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
        "lm_head.weight",
        "model.visual.patch_embed.proj.weight",
    }
    layer_types = [
        "linear_attention", "linear_attention", "linear_attention", "qwen_sparse_attention"
    ]
    for layer, layer_type in enumerate(layer_types):
        pfx = f"model.language_model.layers.{layer}"
        for stage in ("attn", "mlp"):
            hp = f"{pfx}.{stage}_hyper_connection"
            names.update({
                f"{hp}.hc_norm.weight",
                f"{hp}.input_mix_weight_down.weight",
                f"{hp}.input_mix_weight_up.weight",
                f"{hp}.block_inject_weight.weight",
            })
        mlp = f"{pfx}.mlp"
        names.update({
            f"{mlp}.gate.weight",
            f"{mlp}.experts.gate_up_proj",
            f"{mlp}.experts.down_proj",
            f"{mlp}.shared_expert.gate_proj.weight",
            f"{mlp}.shared_expert.up_proj.weight",
            f"{mlp}.shared_expert.down_proj.weight",
            f"{mlp}.shared_expert_gate.weight",
        })
        if layer_type == "linear_attention":
            attn = f"{pfx}.linear_attn"
            names.update({
                f"{attn}.in_proj_qkv.weight", f"{attn}.in_proj_z.weight",
                f"{attn}.in_proj_a.weight", f"{attn}.in_proj_b.weight",
                f"{attn}.conv1d.weight", f"{attn}.dt_bias", f"{attn}.A_log",
                f"{attn}.norm.weight", f"{attn}.out_proj.weight",
            })
        else:
            attn = f"{pfx}.self_attn"
            names.update({
                f"{attn}.q_proj.weight", f"{attn}.k_proj.weight",
                f"{attn}.v_proj.weight", f"{attn}.o_proj.weight",
                f"{attn}.q_norm.weight", f"{attn}.k_norm.weight",
                f"{attn}.indexer.index_qk_proj.weight",
                f"{attn}.indexer.q_layernorm.weight",
                f"{attn}.indexer.k_layernorm.weight",
            })
    ple = "model.language_model.layers.1.ple"
    names.update({
        f"{ple}.key_proj.weight", f"{ple}.value_proj.weight",
        f"{ple}.conv1d.weight", f"{ple}.norm_key.weight",
        f"{ple}.norm_query.weight", f"{ple}.norm_conv.weight",
        f"{ple}.ple_embedding.layer_multipliers",
        f"{ple}.ple_embedding.ngram_heads_offsets",
        f"{ple}.ple_embedding.ngram_heads_vocab_sizes",
        f"{ple}.ple_embedding.ngram_embedding.shard_0.weight",
        f"{ple}.ple_embedding.ngram_embedding.shard_1.weight",
    })
    return {
        name: converter.HeaderTensor(
            name=name,
            dtype="I64" if name.endswith(("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")) else "BF16",
            shape=[2, 16] if ".ngram_embedding.shard_" in name else [1],
            shard=Path("model.safetensors"),
        )
        for name in names
    }


def test_qwen4_exp_import_contract_preserves_flash_next_topology(
    tmp_path: Path, monkeypatch
) -> None:
    converter = _load_converter()
    hf = _qwen4_exp_tiny_config()
    assert converter._infer_arch(hf) == "qwen4_exp"

    architecture = converter._qwen4_exp_architecture_metadata(hf)
    assert architecture["layer_kinds"] == [
        "recurrent", "recurrent", "recurrent", "sparse_attention"
    ]
    assert architecture["ple_layer_ids"] == [2]
    assert architecture["ple_owner_layers"] == [1]
    assert architecture["rotary_dim"] == 64
    assert architecture["mrope_sections"] == [11, 11, 10, 0]

    headers = _qwen4_exp_tiny_headers(converter)
    refs = converter._qwen4_exp_text_refs(hf, headers)
    by_name = {ref.ck_name: ref for ref in refs}
    assert "layer.3.index_qk" in by_name
    assert "layer.1.ple_ngram_embedding" in by_name
    assert by_name["layer.1.ple_ngram_embedding"].shape == (4, 16)
    assert len(by_name["layer.1.ple_ngram_embedding"].source_names) == 2
    assert by_name["layer.1.ple_layer_multipliers"].dtype == "i64"
    assert converter._ref_transform("qwen4_exp", by_name["layer.3.attn_q_norm"]) == "qwen4_exp_norm_plus_one"
    for name in ("ple_norm_key", "ple_norm_query", "ple_norm_conv"):
        assert converter._ref_transform("qwen4_exp", by_name[f"layer.1.{name}"]) == (
            "qwen4_exp_norm_plus_one"
        )
    assert converter._ref_transform("qwen4_exp", by_name["layer.0.ssm_norm"]) is None

    audit = converter._build_source_audit("qwen4_exp", headers, refs)
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert audit["ignored_source_tensors"] == [{
        "source": "model.visual.patch_embed.proj.weight",
        "reason": "vision_tower_not_in_text_artifact",
    }]

    checkpoint = tmp_path / "qwen4_exp"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(hf), encoding="utf-8")
    monkeypatch.setattr(converter, "_load_safetensors_headers", lambda _: headers)
    config = converter._build_config(checkpoint, "qwen4_exp", None)
    assert config["model"] == "qwen4_exp"
    assert config["layer_attention_policy"] == ["none", "none", "none", "qsa_sparse"]
    assert config["circuit_layer_kinds"] == [
        "recurrent", "recurrent_ple", "recurrent", "sparse_attention"
    ]
    assert config["n_routed_experts"] == 8
    assert config["experts_per_tok"] == 4
    assert config["recurrent_qkv_weight_dtype"] == "bf16"
    assert config["ple_eos_token_id"] == 248044
    assert config["layer_execution_plan"][1]["ple"] is True
    assert config["layer_execution_plan"][2]["ple"] is False




def _write_tiny_bpe_tokenizer(checkpoint: Path, vocab_size: int) -> None:
    vocab: dict[str, int] = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "Hello": 3,
        "world": 4,
        "!": 5,
        "Ġtest": 6,
        "Ġcode": 7,
        "Helloworld": 8,
        "ĠtestĠcode": 9,
    }
    for idx in range(len(vocab), vocab_size):
        vocab[f"<tok_{idx}>"] = idx

    (checkpoint / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "unk_token": "<unk>",
                    "vocab": vocab,
                    "merges": ["Hello world", "Ġtest Ġcode"],
                },
                "added_tokens": [
                    {"id": 0, "content": "<unk>"},
                    {"id": 1, "content": "<s>"},
                    {"id": 2, "content": "</s>"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (checkpoint / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": {"content": "<s>"},
                "eos_token": {"content": "</s>"},
                "unk_token": {"content": "<unk>"},
                "add_bos_token": True,
                "add_eos_token": False,
                "added_tokens_decoder": {
                    "0": {"content": "<unk>"},
                    "1": {"content": "<s>"},
                    "2": {"content": "</s>"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_tokenizer_contract_preserves_unicode_isolated_split_profile(tmp_path: Path) -> None:
    converter = _load_converter()
    checkpoint = tmp_path / "tokenizer_profile"
    checkpoint.mkdir()
    _write_tiny_bpe_tokenizer(checkpoint, vocab_size=32)
    path = checkpoint / "tokenizer.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["pre_tokenizer"] = {
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {"Regex": r"\d{1,3}(?=(?:\d{3})*\b)"},
                "behavior": "Isolated",
                "invert": False,
            },
            {
                "type": "Split",
                "pattern": {
                    "Regex": r"[^\r\n\p{L}\p{N}]?[\p{Lu}]*[\p{Ll}]+|\p{N}{1,3}"
                },
                "behavior": "Isolated",
                "invert": False,
            },
            {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": False,
            },
        ],
    }
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")

    payloads, contract, _special = converter._tokenizer_payloads_from_json(
        checkpoint, 32
    )
    assert payloads
    assert contract is not None
    assert contract["tokenizer_type"] == "bpe"
    assert contract["pretokenizer"] == "unicode_split_isolated"

def test_qwen3_safetensors_to_bump_smoke(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "qwen3"
    out = tmp_path / "out"
    checkpoint.mkdir()
    out.mkdir()

    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_hidden_layers": 1,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
                "max_position_embeddings": 64,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tiny_bpe_tokenizer(checkpoint, vocab_size=32)

    tensors = {
        "model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.float32),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(8, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_norm.weight": torch.ones(4, dtype=torch.float32),
        "model.layers.0.self_attn.k_norm.weight": torch.ones(4, dtype=torch.float32),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(8, dtype=torch.float32),
        "lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
    }
    st.save_file(tensors, checkpoint / "model.safetensors")

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    names = [entry["name"] for entry in manifest["entries"]]
    assert manifest["model"] == "qwen3"
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert names[0] == "token_emb"
    assert "layer.0.q_norm" in names
    assert "layer.0.k_norm" in names
    assert "output.weight" in names
    assert {"vocab_offsets", "vocab_strings", "vocab_merges"}.issubset(set(names))
    assert names[-3:] == ["vocab_offsets", "vocab_strings", "vocab_merges"]
    entries = {entry["name"]: entry for entry in manifest["entries"]}
    assert entries["vocab_offsets"]["dtype"] == "i32"
    assert entries["vocab_strings"]["dtype"] == "u8"
    assert entries["vocab_merges"]["dtype"] == "i32"
    assert entries["vocab_offsets"]["shape"] == [32]
    assert entries["vocab_strings"]["size"] > 0
    assert entries["vocab_merges"]["shape"] == [6]
    assert entries["vocab_merges"]["size"] == 24
    assert manifest["tokenizer_contract"]["tokenizer_type"] == "bpe"
    assert "pretokenizer" not in manifest["tokenizer_contract"]
    assert manifest["config"]["tokenizer_contract"]["tokenizer_type"] == "bpe"
    assert manifest["special_tokens"]["bos_token"] == "<s>"
    assert manifest["special_tokens"]["bos_token_id"] == 1
    assert manifest["special_tokens"]["eos_token"] == "</s>"
    assert manifest["special_tokens"]["eos_token_id"] == 2
    assert manifest["special_tokens"]["unk_token"] == "<unk>"
    assert manifest["special_tokens"]["unk_token_id"] == 0
    assert manifest["template"]["flags"]["tokenizer"] == "bpe"
    assert manifest["template"]["contract"]["tokenizer_contract"]["tokenizer_type"] == "bpe"
    assert (out / "weights.bump").stat().st_size > 0


def test_whisper_encoder_safetensors_maps_and_generates_call_ir(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "whisper_tiny"
    out = tmp_path / "out_whisper_tiny"
    checkpoint.mkdir()
    out.mkdir()

    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["WhisperForConditionalGeneration"],
                "model_type": "whisper",
                "d_model": 8,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "max_source_positions": 4,
                "num_mel_bins": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    tensors = {
        "model.encoder.conv1.weight": torch.randn(8, 4, 3),
        "model.encoder.conv1.bias": torch.randn(8),
        "model.encoder.conv2.weight": torch.randn(8, 8, 3),
        "model.encoder.conv2.bias": torch.randn(8),
        "model.encoder.embed_positions.weight": torch.randn(4, 8),
        "model.encoder.layers.0.self_attn_layer_norm.weight": torch.randn(8),
        "model.encoder.layers.0.self_attn_layer_norm.bias": torch.randn(8),
        "model.encoder.layers.0.final_layer_norm.weight": torch.randn(8),
        "model.encoder.layers.0.final_layer_norm.bias": torch.randn(8),
        "model.encoder.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
        "model.encoder.layers.0.self_attn.q_proj.bias": torch.randn(8),
        "model.encoder.layers.0.self_attn.k_proj.weight": torch.randn(8, 8),
        "model.encoder.layers.0.self_attn.v_proj.weight": torch.randn(8, 8),
        "model.encoder.layers.0.self_attn.v_proj.bias": torch.randn(8),
        "model.encoder.layers.0.self_attn.out_proj.weight": torch.randn(8, 8),
        "model.encoder.layers.0.self_attn.out_proj.bias": torch.randn(8),
        "model.encoder.layers.0.fc1.weight": torch.randn(16, 8),
        "model.encoder.layers.0.fc1.bias": torch.randn(16),
        "model.encoder.layers.0.fc2.weight": torch.randn(8, 16),
        "model.encoder.layers.0.fc2.bias": torch.randn(8),
        "model.encoder.layer_norm.weight": torch.randn(8),
        "model.encoder.layer_norm.bias": torch.randn(8),
        # A complete Whisper checkpoint contains decoder tensors. The
        # encoder artifact must classify them explicitly instead of silently
        # accepting or accidentally binding them.
        "model.decoder.embed_tokens.weight": torch.randn(32, 8),
        "proj_out.weight": torch.randn(32, 8),
    }
    st.save_file(tensors, checkpoint / "model.safetensors")

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in manifest["entries"]}
    assert manifest["model"] == "whisper_encoder"
    assert manifest["template"]["name"] == "audio_transformer_encoder"
    assert manifest["config"]["artifact_scope"] == "encoder_only"
    assert manifest["config"]["audio_feature_channels"] == 4
    assert manifest["config"]["audio_feature_frames"] == 8
    assert manifest["config"]["audio_conv1_elements"] == 64
    assert manifest["config"]["audio_conv2_output_frames"] == 4
    assert manifest["config"]["audio_conv2_elements"] == 32
    assert manifest["config"]["head_dim"] == 4
    assert manifest["config"]["attention_scale"] == 0.5
    assert manifest["tokenizer_contract"] is None
    assert not any(name.startswith("vocab_") for name in entries)
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert audit["synthetic_entries"] == ["layer.0.bk"]
    assert entries["layer.0.bk"]["source_name"] == "synthetic:zeros_fp32"
    assert entries["layer.0.bk"]["shape"] == [8]
    assert entries["layer.0.bk"]["dtype"] == "fp32"
    assert all(entry["file_offset"] % 64 == 0 for entry in manifest["entries"])
    ignored = {
        row["source"]: row["reason"] for row in audit["ignored_source_tensors"]
    }
    assert ignored == {
        "model.decoder.embed_tokens.weight": "decoder_not_in_encoder_artifact",
        "proj_out.weight": "decoder_not_in_encoder_artifact",
    }
    assert (out / "weights.bump").stat().st_size > 0

    build_ir = Path("version/v8/scripts/build_ir_v8.py")
    lowered = out / "lowered_encoder.json"
    call = out / "call_encoder.json"
    subprocess.run(
        [
            sys.executable,
            str(build_ir),
            "--manifest",
            str(out / "weights_manifest.json"),
            "--mode",
            "prefill",
            "--output",
            str(out / "ir1_encoder.json"),
            "--layout-output",
            str(out / "layout_encoder.json"),
            "--lowered-output",
            str(lowered),
            "--call-output",
            str(call),
            "--context-len",
            "4",
        ],
        check=True,
    )
    call_ops = json.loads(call.read_text(encoding="utf-8"))["operations"]
    assert not [op for op in call_ops if op.get("errors")]
    functions = {op["op"]: op["function"] for op in call_ops}
    assert functions["audio_wav_decode"] == "audio_wav_decode_memory_pcm16_mono_window_f32"
    assert functions["audio_resample"] == "audio_resample_windowed_sinc_f32"
    assert functions["audio_pad_or_truncate"] == "audio_pad_or_truncate_f32"
    assert functions["audio_stft_tables"] == "audio_stft_precompute_tables_f32"
    assert functions["audio_stft"] == "audio_stft_power_fft400_f32"
    assert functions["audio_mel_filters"] == "audio_whisper_mel_filters_slaney_f32"
    assert (
        functions["audio_log_mel"]
        == "audio_whisper_log_mel_from_power_reference_f32"
    )
    assert (
        functions["audio_feature_window"]
        == "audio_whisper_log_mel_window_wav_pcm16_f32"
    )
    assert functions["audio_conv1d_stem_1"] == "audio_conv1d_channel_major_f32"
    assert functions["audio_conv1d_stem_2"] == "audio_conv1d_channel_major_f32"
    assert functions["layout_channel_to_token"] == "audio_transpose_channel_to_token_f32"
    assert (
        functions["attn"]
        == "attention_forward_query_key_head_major_f32_packed_k"
    )
    attn_op = next(op for op in call_ops if op["op"] == "attn")
    attn_args = {arg["name"]: arg["expr"] for arg in attn_op["args"]}
    assert attn_args["score_scratch"] != attn_args["key_transpose_scratch"]
    layout_doc = json.loads(
        (out / "layout_encoder.json").read_text(encoding="utf-8")
    )
    assert layout_doc["memory"]["arena"]["activations_base"] % 64 == 0
    assert all(
        row["abs_offset"] % 64 == 0
        for row in layout_doc["memory"]["activations"]["buffers"]
        if row["dtype"] in {"fp32", "bf16", "fp16"}
    )

    generated_c = out / "whisper_encoder_v8.c"
    subprocess.run(
        [
            sys.executable,
            "version/v8/scripts/codegen_v8.py",
            "--ir",
            str(call),
            "--layout",
            str(out / "layout_encoder.json"),
            "--output",
            str(generated_c),
            "--strict-contracts",
        ],
        check=True,
    )
    generated = generated_c.read_text(encoding="utf-8")
    assert "audio_conv1d_channel_major_f32" in generated
    assert "attention_forward_query_key_head_major_f32" in generated
    assert "CK_EXPORT int ck_model_run_encoder(void)" in generated
    assert "CK_EXPORT int ck_model_run_audio_wav(" in generated
    assert "audio_wav_decode_memory_pcm16_mono_window_f32(" in generated
    assert "CK_EXPORT int ck_model_run_audio_wav_window(" in generated
    assert "CK_EXPORT int ck_model_prepare_audio_wav_window(" in generated
    assert "audio_resample_windowed_sinc_f32(" in generated
    assert "audio_pad_or_truncate_f32(" in generated
    assert "audio_stft_precompute_tables_f32(" in generated
    assert "audio_stft_power_fft400_f32(" in generated
    assert "audio_whisper_mel_filters_slaney_f32(" in generated
    assert "audio_whisper_log_mel_from_power_reference_f32(" in generated
    assert "audio_whisper_log_mel_window_wav_pcm16_f32(" in generated
    subprocess.run(
        [
            "cc",
            "-fsyntax-only",
            "-fopenmp",
            "-Iinclude",
            "-Iversion/v8/src",
            str(generated_c),
        ],
        check=True,
    )

    fp16_out = tmp_path / "out_whisper_encoder_fp16"
    fp16_out.mkdir()
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(fp16_out / "weights.bump"),
            "--config-out",
            str(fp16_out / "config.json"),
            "--manifest-out",
            str(fp16_out / "weights_manifest.json"),
            "--arch",
            "whisper_encoder",
            "--linear-weight-dtype",
            "fp16",
            "--dry-run",
        ],
        check=True,
    )
    fp16_manifest = json.loads(
        (fp16_out / "weights_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        fp16_manifest["config"]["audio_encoder_attention_reduction_policy"]
        == "tiled_f16kv_online_softmax"
    )
    assert (
        fp16_manifest["config"]["audio_runtime_topology_policy"]
        == "performance_core_smt_on_hybrid"
    )
    fp16_entries = fp16_manifest["entries"]
    projection_entries = [
        entry for entry in fp16_entries if entry.get("role") == "linear_weight"
    ]
    assert len(projection_entries) == 6
    assert {entry["dtype"] for entry in projection_entries} == {"fp16"}
    assert {
        entry["dtype"]
        for entry in fp16_entries
        if entry.get("role") != "linear_weight"
    } == {"fp32"}
    fp16_call = fp16_out / "call_encoder.json"
    subprocess.run(
        [
            sys.executable,
            str(build_ir),
            "--manifest",
            str(fp16_out / "weights_manifest.json"),
            "--mode",
            "prefill",
            "--output",
            str(fp16_out / "ir1_encoder.json"),
            "--layout-output",
            str(fp16_out / "layout_encoder.json"),
            "--lowered-output",
            str(fp16_out / "lowered_encoder.json"),
            "--call-output",
            str(fp16_call),
            "--context-len",
            "4",
        ],
        check=True,
    )
    fp16_ops = json.loads(fp16_call.read_text(encoding="utf-8"))["operations"]
    projection_ops = {
        "q_proj", "k_proj", "v_proj", "out_proj", "mlp_up", "mlp_down"
    }
    assert {
        op["function"] for op in fp16_ops if op["op"] in projection_ops
    } == {"gemm_nt_f16"}
    assert {
        op["function"] for op in fp16_ops if op["op"] == "attn"
    } == {"attention_forward_query_key_head_major_tiled_f16kv_fp32"}

    fp16_real_out = tmp_path / "out_whisper_encoder_fp16_real"
    fp16_real_out.mkdir()
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(fp16_real_out / "weights.bump"),
            "--config-out",
            str(fp16_real_out / "config.json"),
            "--manifest-out",
            str(fp16_real_out / "weights_manifest.json"),
            "--arch",
            "whisper_encoder",
            "--linear-weight-dtype",
            "fp16",
        ],
        check=True,
    )
    fp16_real_manifest = json.loads(
        (fp16_real_out / "weights_manifest.json").read_text(encoding="utf-8")
    )
    wq_entry = next(
        entry for entry in fp16_real_manifest["entries"]
        if entry["name"] == "layer.0.wq"
    )
    assert wq_entry["dtype"] == "fp16"
    assert wq_entry["role"] == "linear_weight"
    assert wq_entry["size"] == 2 * math.prod(wq_entry["shape"])
    with (fp16_real_out / "weights.bump").open("rb") as weights_file:
        weights_file.seek(wq_entry["file_offset"])
        payload = weights_file.read(wq_entry["size"])
    expected = tensors["model.encoder.layers.0.self_attn.q_proj.weight"]
    actual = np.frombuffer(payload, dtype=np.float16).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected.numpy().astype(np.float16))


def test_whisper_decoder_safetensors_keeps_self_and_cross_attention_distinct(
    tmp_path: Path,
) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "whisper_tiny"
    out = tmp_path / "out_whisper_decoder"
    checkpoint.mkdir()
    out.mkdir()
    _write_tiny_bpe_tokenizer(checkpoint, vocab_size=32)

    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["WhisperForConditionalGeneration"],
                "model_type": "whisper",
                "d_model": 8,
                "decoder_layers": 2,
                "decoder_attention_heads": 2,
                "decoder_ffn_dim": 16,
                "max_source_positions": 4,
                "max_target_positions": 8,
                "num_mel_bins": 4,
                "vocab_size": 32,
                "tie_word_embeddings": True,
                "decoder_start_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def matrix(rows: int, cols: int):
        return torch.randn(rows, cols)

    tensors = {
        "model.decoder.embed_tokens.weight": matrix(32, 8),
        "model.decoder.embed_positions.weight": matrix(8, 8),
        "model.decoder.layer_norm.weight": torch.randn(8),
        "model.decoder.layer_norm.bias": torch.randn(8),
        "model.decoder.layers.0.self_attn_layer_norm.weight": torch.randn(8),
        "model.decoder.layers.0.self_attn_layer_norm.bias": torch.randn(8),
        "model.decoder.layers.0.self_attn.q_proj.weight": matrix(8, 8),
        "model.decoder.layers.0.self_attn.q_proj.bias": torch.randn(8),
        "model.decoder.layers.0.self_attn.k_proj.weight": matrix(8, 8),
        "model.decoder.layers.0.self_attn.v_proj.weight": matrix(8, 8),
        "model.decoder.layers.0.self_attn.v_proj.bias": torch.randn(8),
        "model.decoder.layers.0.self_attn.out_proj.weight": matrix(8, 8),
        "model.decoder.layers.0.self_attn.out_proj.bias": torch.randn(8),
        "model.decoder.layers.0.encoder_attn_layer_norm.weight": torch.randn(8),
        "model.decoder.layers.0.encoder_attn_layer_norm.bias": torch.randn(8),
        "model.decoder.layers.0.encoder_attn.q_proj.weight": matrix(8, 8),
        "model.decoder.layers.0.encoder_attn.q_proj.bias": torch.randn(8),
        "model.decoder.layers.0.encoder_attn.k_proj.weight": matrix(8, 8),
        "model.decoder.layers.0.encoder_attn.v_proj.weight": matrix(8, 8),
        "model.decoder.layers.0.encoder_attn.v_proj.bias": torch.randn(8),
        "model.decoder.layers.0.encoder_attn.out_proj.weight": matrix(8, 8),
        "model.decoder.layers.0.encoder_attn.out_proj.bias": torch.randn(8),
        "model.decoder.layers.0.final_layer_norm.weight": torch.randn(8),
        "model.decoder.layers.0.final_layer_norm.bias": torch.randn(8),
        "model.decoder.layers.0.fc1.weight": matrix(16, 8),
        "model.decoder.layers.0.fc1.bias": torch.randn(16),
        "model.decoder.layers.0.fc2.weight": matrix(8, 16),
        "model.decoder.layers.0.fc2.bias": torch.randn(8),
        "model.encoder.layer_norm.weight": torch.randn(8),
    }
    for name, tensor in list(tensors.items()):
        if "model.decoder.layers.0." in name:
            tensors[name.replace("layers.0.", "layers.1.")] = tensor.clone()
    st.save_file(tensors, checkpoint / "model.safetensors")

    subprocess.run(
        [
            sys.executable,
            "version/v8/scripts/convert_safetensors_to_bump_v8.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "whisper_decoder",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in manifest["entries"]}
    assert manifest["model"] == "whisper_decoder"
    assert manifest["template"]["name"] == "audio_transformer_decoder"
    assert manifest["config"]["artifact_scope"] == "decoder_only"
    assert manifest["config"]["context_length"] == 8
    assert manifest["config"]["encoder_memory_length"] == 4
    assert manifest["config"]["uses_cross_attention"] is True
    assert entries["layer.0.wq"]["source_name"].endswith("self_attn.q_proj.weight")
    assert entries["layer.0.cross_wq"]["source_name"].endswith(
        "encoder_attn.q_proj.weight"
    )
    assert entries["layer.0.wk"]["source_name"].endswith("self_attn.k_proj.weight")
    assert entries["layer.0.cross_wk"]["source_name"].endswith(
        "encoder_attn.k_proj.weight"
    )
    assert entries["layer.0.bk"]["source_name"] == "synthetic:zeros_fp32"
    assert entries["layer.0.cross_bk"]["source_name"] == "synthetic:zeros_fp32"
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert {
        row["source"]: row["reason"] for row in audit["ignored_source_tensors"]
    } == {
        "model.encoder.layer_norm.weight": "encoder_not_in_decoder_artifact"
    }

    for mode in ("prefill", "decode"):
        subprocess.run(
            [
                sys.executable,
                "version/v8/scripts/build_ir_v8.py",
                "--manifest",
                str(out / "weights_manifest.json"),
                "--mode",
                mode,
                "--output",
                str(out / f"lowered_{mode}.json"),
                "--layout-output",
                str(out / f"layout_{mode}.json"),
                "--call-output",
                str(out / f"lowered_{mode}_call.json"),
                "--context-len",
                "8",
            ],
            check=True,
        )

    calls = {
        mode: json.loads(
            (out / f"lowered_{mode}_call.json").read_text(encoding="utf-8")
        )
        for mode in ("prefill", "decode")
    }
    assert calls["prefill"]["errors"] == []
    assert calls["decode"]["errors"] == []

    def first_call(mode: str, op_name: str, layer: int = 0) -> dict:
        return next(
            row
            for row in calls[mode]["operations"]
            if row["op"] == op_name and int(row.get("layer", -1)) == layer
        )

    def args_by_name(call: dict) -> dict:
        return {arg["name"]: arg for arg in call["args"]}

    decode_position = first_call("decode", "position_embeddings", layer=-1)
    decode_position_args = args_by_name(decode_position)
    assert decode_position["function"] == "position_embeddings_add_at_offset"
    assert decode_position_args["start_position"]["expr"] == "model->pos"

    for op_name in (
        "q_proj",
        "v_proj",
        "out_proj",
        "cross_q_proj",
        "cross_out_proj",
        "mlp_up",
        "mlp_down",
    ):
        projection_args = args_by_name(first_call("decode", op_name))
        assert projection_args["bias"]["expr"] == "NULL"
    decode_bias_adds = [
        row for row in calls["decode"]["operations"] if row["op"] == "bias_add"
    ]
    assert len(decode_bias_adds) == 14
    assert all(args_by_name(row)["b"]["expr"] != "NULL" for row in decode_bias_adds)
    cross_v_args = args_by_name(first_call("prefill", "cross_v_proj"))
    assert cross_v_args["bias"]["expr"] != "NULL"
    decode_layer_ops = [
        row
        for row in calls["decode"]["operations"]
        if int(row.get("layer", -1)) == 0
    ]
    v_projection_index = next(
        i for i, row in enumerate(decode_layer_ops) if row["op"] == "v_proj"
    )
    v_bias_index = next(
        i
        for i, row in enumerate(decode_layer_ops)
        if row["op"] == "bias_add"
        and args_by_name(row)["a"].get("buffer_ref") == "v_scratch"
    )
    kv_store_index = next(
        i for i, row in enumerate(decode_layer_ops) if row["op"] == "kv_cache_store"
    )
    attention_index = next(
        i for i, row in enumerate(decode_layer_ops) if row["op"] == "attn"
    )
    assert v_projection_index < v_bias_index < kv_store_index < attention_index
    decode_attention_args = args_by_name(decode_layer_ops[attention_index])
    assert (
        decode_layer_ops[attention_index]["function"]
        == "attention_forward_decode_head_major_gqa_flash"
    )
    assert decode_attention_args["kv_tokens"]["expr"] == "model->pos + 1"
    assert decode_attention_args["cache_capacity"]["expr"] == "8"
    assert "model->kv_cache" in decode_attention_args["k_cache"]["expr"]
    assert "model->kv_cache" in decode_attention_args["v_cache"]["expr"]
    assert "model->bump" not in decode_attention_args["k_cache"]["expr"]
    assert "model->bump" not in decode_attention_args["v_cache"]["expr"]

    prefill_attention = first_call("prefill", "attn")
    assert (
        prefill_attention["function"]
        == "attention_forward_causal_head_major_gqa_flash_strided"
    )

    decode_ops = [row["op"] for row in calls["decode"]["operations"]]
    assert "cross_k_proj" not in decode_ops
    assert "cross_v_proj" not in decode_ops
    assert "transpose_cross_kv_to_head_major" not in decode_ops

    for mode, query_tokens in (("prefill", "8"), ("decode", "1")):
        cross_attn = args_by_name(first_call(mode, "cross_attn"))
        assert cross_attn["query_tokens"]["expr"] == query_tokens
        assert cross_attn["query_tokens"]["source"] == "runtime:query_tokens"
        assert cross_attn["key_tokens"]["expr"] == "4"
        assert cross_attn["query"]["buffer_ref"] == "cross_q_scratch"
        assert cross_attn["key"]["buffer_ref"] == "cross_k_cache"
        assert cross_attn["value"]["buffer_ref"] == "cross_v_cache"

    for op_name, cache_name in (
        ("cross_k_proj", "cross_k_cache"),
        ("cross_v_proj", "cross_v_cache"),
    ):
        projection = args_by_name(first_call("prefill", op_name))
        assert projection["A"]["buffer_ref"] == "encoder_memory"
        assert projection["C"]["buffer_ref"] == cache_name
        assert projection["M"]["expr"] == "4"
        assert projection["M"]["source"] == "dim:encoder_memory_length"

    prefill_ops = [
        row["op"]
        for row in calls["prefill"]["operations"]
        if int(row.get("layer", -1)) == 0
    ]
    assert prefill_ops.index("cross_q_proj") < prefill_ops.index(
        "transpose_cross_q_to_head_major"
    )
    assert prefill_ops.count("transpose_cross_kv_to_head_major") == 2
    assert prefill_ops.index("cross_attn") < prefill_ops.index(
        "transpose_cross_attn_out_to_token_major"
    )

    layout_decode = json.loads((out / "layout_decode.json").read_text(encoding="utf-8"))
    activation_buffers = {
        row["name"]: row
        for row in layout_decode["memory"]["activations"]["buffers"]
    }
    assert activation_buffers["cross_k_cache"]["shape"] == "[2, 2, 4, 4]"
    assert activation_buffers["cross_v_cache"]["shape"] == "[2, 2, 4, 4]"
    layer_stride_bytes = 2 * 4 * 4 * 4
    for mode in ("prefill", "decode"):
        for op_name, macro in (
            ("cross_attn", "A_CROSS_K_CACHE"),
            ("cross_attn", "A_CROSS_V_CACHE"),
        ):
            args = args_by_name(first_call(mode, op_name, layer=1))
            cache_arg = "key" if macro.endswith("K_CACHE") else "value"
            assert args[cache_arg]["expr"] == (
                f"(const float*)(model->bump + ({macro} + {layer_stride_bytes}))"
            )

    generated_c = out / "whisper_decoder_v8.c"
    subprocess.run(
        [
            sys.executable,
            "version/v8/scripts/codegen_v8.py",
            "--ir",
            str(out / "lowered_decode_call.json"),
            "--prefill",
            str(out / "lowered_prefill_call.json"),
            "--prefill-layout",
            str(out / "layout_prefill.json"),
            "--layout",
            str(out / "layout_decode.json"),
            "--output",
            str(generated_c),
            "--strict-contracts",
        ],
        check=True,
    )
    generated = generated_c.read_text(encoding="utf-8")
    assert "position_embeddings_add_at_offset" in generated
    assert "attention_forward_query_key_head_major_f32" in generated
    assert "CK_EXPORT int ck_model_set_encoder_memory(" in generated
    assert "if (tokens != 4 || dim != 8) return -2;" in generated
    assert "g_model->encoder_kv_ready = 0;" in generated
    assert "if (!g_model->encoder_kv_ready) return -2;" in generated
    assert generated.count("model->encoder_kv_ready = 1;") == 2
    assert "g_model->bump + A_ENCODER_MEMORY" in generated
    encoder_projection_calls = generated.count(
        "(const float*)(model->bump + A_ENCODER_MEMORY),"
    )
    assert encoder_projection_calls == 8
    assert generated.count(
        "(float*)(model->bump + A_CROSS_K_CACHE),\n"
        "        4,\n"
        "        8,\n"
        "        8"
    ) == 2
    assert generated.count(
        "(float*)(model->bump + A_CROSS_V_CACHE),\n"
        "        4,\n"
        "        8,\n"
        "        8"
    ) == 2
    assert generated.count(
        "        2,\n"
        "        num_tokens,\n"
        "        4,\n"
        "        4,\n"
        "        0.5"
    ) == 4
    subprocess.run(
        [
            "cc",
            "-fsyntax-only",
            "-fopenmp",
            "-Iinclude",
            "-Iversion/v8/src",
            str(generated_c),
        ],
        check=True,
    )


def test_qwen35_safetensors_to_bump_smoke(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "qwen35"
    out = tmp_path / "out_qwen35"
    checkpoint.mkdir()
    out.mkdir()

    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "tie_word_embeddings": True,
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "num_hidden_layers": 4,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "vocab_size": 32,
                    "max_position_embeddings": 64,
                    "full_attention_interval": 4,
                    "layer_types": layer_types,
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 4,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 2,
                    "linear_value_head_dim": 4,
                    "rms_norm_eps": 1e-6,
                    "tie_word_embeddings": True,
                    "rope_parameters": {
                        "rope_theta": 10000000.0,
                        "partial_rotary_factor": 1.0,
                        "mrope_interleaved": True,
                        "mrope_section": [1, 1, 0],
                        "rope_type": "default",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    tensors = {
        "model.language_model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "model.language_model.norm.weight": torch.ones(8, dtype=torch.float32),
    }
    for layer, kind in enumerate(layer_types):
        prefix = f"model.language_model.layers.{layer}"
        tensors[f"{prefix}.input_layernorm.weight"] = torch.ones(8, dtype=torch.float32)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(8, dtype=torch.float32)
        tensors[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(16, 8, dtype=torch.bfloat16)
        tensors[f"{prefix}.mlp.up_proj.weight"] = torch.randn(16, 8, dtype=torch.bfloat16)
        tensors[f"{prefix}.mlp.down_proj.weight"] = torch.randn(8, 16, dtype=torch.bfloat16)
        if kind == "linear_attention":
            tensors[f"{prefix}.linear_attn.in_proj_qkv.weight"] = torch.randn(24, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.linear_attn.in_proj_z.weight"] = torch.randn(8, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.linear_attn.in_proj_a.weight"] = torch.randn(8, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.linear_attn.in_proj_b.weight"] = torch.randn(8, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.linear_attn.conv1d.weight"] = torch.randn(24, 1, 4, dtype=torch.float32)
            tensors[f"{prefix}.linear_attn.dt_bias"] = torch.randn(8, dtype=torch.float32)
            tensors[f"{prefix}.linear_attn.A_log"] = torch.randn(8, 4, dtype=torch.float32)
            tensors[f"{prefix}.linear_attn.norm.weight"] = torch.ones(8, dtype=torch.float32)
            tensors[f"{prefix}.linear_attn.out_proj.weight"] = torch.randn(8, 8, dtype=torch.bfloat16)
        else:
            tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(16, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(4, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(4, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(8, 16, dtype=torch.bfloat16)
            tensors[f"{prefix}.self_attn.q_norm.weight"] = torch.ones(4, dtype=torch.float32)
            tensors[f"{prefix}.self_attn.k_norm.weight"] = torch.ones(4, dtype=torch.float32)

    st.save_file(tensors, checkpoint / "model.safetensors")

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    names = [entry["name"] for entry in manifest["entries"]]
    assert manifest["model"] == "qwen35"
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert any(row["target"] == "layer.0.ssm_a" and row["transform"] == "neg_exp_a_log" for row in audit["transforms"])
    assert any(row["target"] == "layer.0.attn_norm" and row["transform"] == "qwen35_norm_plus_one" for row in audit["transforms"])
    assert any(row["target"] == "layer.3.attn_q_norm" and row["transform"] == "qwen35_norm_plus_one" for row in audit["transforms"])
    assert any(row["target"] == "final_ln_weight" and row["transform"] == "qwen35_norm_plus_one" for row in audit["transforms"])
    assert not any(row["target"] == "layer.0.ssm_norm" and row.get("transform") == "qwen35_norm_plus_one" for row in audit["transforms"])
    assert config["layer_kinds"] == ["recurrent", "recurrent", "recurrent", "full_attention"]
    assert config["layer_recurrent_policy"] == ["deltanet", "deltanet", "deltanet", "none"]
    assert config["attn_q_gate_proj_dim"] == 16
    assert config["attn_out_dim"] == 8
    assert config["q_dim"] == 8
    assert config["k_dim"] == 8
    assert config["v_dim"] == 8
    assert config["gate_dim"] == 8
    assert config["recurrent_qkv_weight_dtype"] == "bf16"
    assert config["decoder_norm_storage_boundary"] == "bf16"
    assert config["decoder_qk_norm_reduction_policy"] == "pytorch_avx2_cascade_exact"
    assert config["decode_kv_cache_dtype"] == "bf16"
    assert config["rotary_dim"] == 4
    assert config["mrope_sections"] == [1, 1, 0, 0]
    assert config["mrope_n_dims"] == 4
    assert config["mrope_interleaved"] is True
    dtypes = {entry["name"]: entry["dtype"] for entry in manifest["entries"]}
    assert dtypes["layer.0.ssm_conv1d"] == "fp32"
    assert "layer.0.attn_qkv" in names
    assert "layer.0.ssm_alpha" in names
    assert "layer.0.ssm_beta" in names
    assert "layer.0.ssm_conv1d" in names
    assert "layer.3.attn_q_gate" in names
    assert "layer.3.attn_q_norm" in names
    assert "final_ln_weight" in names
    assert "output.weight" not in names
    assert (out / "weights.bump").stat().st_size > 0



def test_nemotron_h_safetensors_to_bump_dry_run_maps_hybrid_mamba_attention_moe(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "nemotron_h"
    out = tmp_path / "out_nemotron_h"
    checkpoint.mkdir()
    out.mkdir()

    config = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "num_hidden_layers": 4,
        "hidden_size": 8,
        "intermediate_size": 6,
        "moe_intermediate_size": 6,
        "moe_shared_expert_intermediate_size": 12,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "hybrid_override_pattern": "M*E-",
        "mamba_num_heads": 2,
        "mamba_head_dim": 4,
        "ssm_state_size": 3,
        "conv_kernel": 4,
        "n_groups": 2,
        "chunk_size": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "mlp_hidden_act": "relu2",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "rope_theta": 10000.0,
        "layer_norm_epsilon": 1e-5,
    }
    (checkpoint / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")

    tensors = {
        "backbone.embeddings.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "backbone.norm_f.weight": torch.ones(8, dtype=torch.float32),
        "lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
    }
    # Layer 0: Mamba2
    tensors.update({
        "backbone.layers.0.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.0.mixer.in_proj.weight": torch.randn(34, 8, dtype=torch.bfloat16),
        "backbone.layers.0.mixer.conv1d.weight": torch.randn(20, 1, 4, dtype=torch.float32),
        "backbone.layers.0.mixer.conv1d.bias": torch.randn(20, dtype=torch.float32),
        "backbone.layers.0.mixer.dt_bias": torch.randn(2, dtype=torch.float32),
        "backbone.layers.0.mixer.A_log": torch.randn(2, dtype=torch.float32),
        "backbone.layers.0.mixer.D": torch.randn(2, dtype=torch.float32),
        "backbone.layers.0.mixer.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.0.mixer.out_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
    })
    # Layer 1: attention
    tensors.update({
        "backbone.layers.1.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.1.mixer.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "backbone.layers.1.mixer.k_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "backbone.layers.1.mixer.v_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "backbone.layers.1.mixer.o_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
    })
    # Layer 2: MoE
    tensors.update({
        "backbone.layers.2.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.2.mixer.gate.weight": torch.randn(4, 8, dtype=torch.float32),
        "backbone.layers.2.mixer.gate.e_score_correction_bias": torch.randn(4, dtype=torch.float32),
        "backbone.layers.2.mixer.shared_experts.up_proj.weight": torch.randn(12, 8, dtype=torch.bfloat16),
        "backbone.layers.2.mixer.shared_experts.down_proj.weight": torch.randn(8, 12, dtype=torch.bfloat16),
    })
    for expert in range(4):
        tensors[f"backbone.layers.2.mixer.experts.{expert}.up_proj.weight"] = torch.randn(6, 8, dtype=torch.bfloat16)
        tensors[f"backbone.layers.2.mixer.experts.{expert}.down_proj.weight"] = torch.randn(8, 6, dtype=torch.bfloat16)
    # Layer 3: dense ReLU2 MLP
    tensors.update({
        "backbone.layers.3.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.3.mixer.up_proj.weight": torch.randn(6, 8, dtype=torch.bfloat16),
        "backbone.layers.3.mixer.down_proj.weight": torch.randn(8, 6, dtype=torch.bfloat16),
    })

    st.save_file(tensors, checkpoint / "model.safetensors")
    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    names = [entry["name"] for entry in manifest["entries"]]
    assert manifest["model"] == "nemotron_h"
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert cfg["layer_kinds"] == ["mamba", "attention", "moe", "mlp"]
    assert cfg["layer_state_policy"] == ["mamba2", "none", "none", "none"]
    assert cfg["ssm_conv_kernel"] == 4
    assert cfg["ssm_conv_history"] == 4
    assert cfg["layer_moe_policy"] == ["none", "none", "routed_relu2", "none"]
    assert "layer.0.mamba_in_proj" in names
    assert "layer.0.mamba_conv1d" in names
    assert "layer.1.attn_q" in names
    assert "layer.2.moe_router" in names
    assert "layer.2.moe_router_bias" in names
    assert "layer.2.moe_expert.3.up" in names
    assert "layer.2.moe_shared_up" in names
    assert "layer.3.mlp_up" in names
    assert "output.weight" in names
    assert any(row["target"] == "layer.0.mamba_a" and row["transform"] == "neg_exp_a_log" for row in audit["transforms"])


def test_nemotron_h_safetensors_to_bump_dry_run_maps_dense_mamba_attention_relu2_without_moe(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "nemotron_h_dense"
    out = tmp_path / "out_nemotron_h_dense"
    checkpoint.mkdir()
    out.mkdir()

    config = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "num_hidden_layers": 3,
        "hidden_size": 8,
        "intermediate_size": 6,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "hybrid_override_pattern": "M*-",
        "mamba_num_heads": 2,
        "mamba_head_dim": 4,
        "ssm_state_size": 3,
        "conv_kernel": 4,
        "n_groups": 2,
        "chunk_size": 8,
        "mlp_hidden_act": "relu2",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "rope_theta": 10000.0,
        "layer_norm_epsilon": 1e-5,
    }
    (checkpoint / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")

    tensors = {
        "backbone.embeddings.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "backbone.norm_f.weight": torch.ones(8, dtype=torch.float32),
        "lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "backbone.layers.0.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.0.mixer.in_proj.weight": torch.randn(34, 8, dtype=torch.bfloat16),
        "backbone.layers.0.mixer.conv1d.weight": torch.randn(20, 1, 4, dtype=torch.float32),
        "backbone.layers.0.mixer.conv1d.bias": torch.randn(20, dtype=torch.float32),
        "backbone.layers.0.mixer.dt_bias": torch.randn(2, dtype=torch.float32),
        "backbone.layers.0.mixer.A_log": torch.randn(2, dtype=torch.float32),
        "backbone.layers.0.mixer.D": torch.randn(2, dtype=torch.float32),
        "backbone.layers.0.mixer.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.0.mixer.out_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "backbone.layers.1.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.1.mixer.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "backbone.layers.1.mixer.k_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "backbone.layers.1.mixer.v_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "backbone.layers.1.mixer.o_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "backbone.layers.2.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.2.mixer.up_proj.weight": torch.randn(6, 8, dtype=torch.bfloat16),
        "backbone.layers.2.mixer.down_proj.weight": torch.randn(8, 6, dtype=torch.bfloat16),
    }

    st.save_file(tensors, checkpoint / "model.safetensors")
    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    names = [entry["name"] for entry in manifest["entries"]]
    assert manifest["model"] == "nemotron_h"
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert cfg["layer_kinds"] == ["mamba", "attention", "mlp"]
    assert cfg["layer_state_policy"] == ["mamba2", "none", "none"]
    assert cfg["layer_moe_policy"] == ["none", "none", "none"]
    assert cfg["layer_mlp_policy"] == ["none", "none", "relu2"]
    attention_ops = manifest["template"]["block_types"]["decoder"]["body"]["ops_by_kind"]["attention"]
    assert "rope_qk" not in attention_ops
    assert cfg["ssm_conv_kernel"] == 4
    assert cfg["ssm_conv_history"] == 4
    assert cfg["recurrent_num_heads"] == 2
    assert cfg["recurrent_head_dim"] == 4
    assert cfg["recurrent_state_heads"] == 2
    assert cfg["recurrent_state_rows"] == 4
    assert cfg["recurrent_state_cols"] == 3
    assert manifest["config"]["recurrent_state_heads"] == 2
    assert manifest["config"]["recurrent_state_rows"] == 4
    assert manifest["config"]["recurrent_state_cols"] == 3
    assert "layer.0.mamba_in_proj" in names
    assert "layer.1.attn_q" in names
    assert "layer.2.mlp_up" in names
    assert "layer.2.mlp_down" in names
    assert "output.weight" in names
    assert not any(".moe_" in name or name.endswith("moe_router") for name in names)
    assert any(row["target"] == "layer.0.mamba_a" and row["transform"] == "neg_exp_a_log" for row in audit["transforms"])

    build_ir = Path("version/v8/scripts/build_ir_v8.py")
    lowered = out / "lowered_decode.json"
    subprocess.run(
        [
            sys.executable,
            str(build_ir),
            "--manifest",
            str(out / "weights_manifest.json"),
            "--mode",
            "decode",
            "--output",
            str(out / "ir1_decode.json"),
            "--layout-output",
            str(out / "layout_decode.json"),
            "--lowered-output",
            str(lowered),
            "--call-output",
            str(out / "lowered_decode_call.json"),
            "--context-len",
            "8",
        ],
        check=True,
    )
    layout = json.loads((out / "layout_decode.json").read_text(encoding="utf-8"))
    conv_state = next(
        buf for buf in layout["memory"]["activations"]["buffers"] if buf["name"] == "recurrent_conv_state"
    )
    ssm_state = next(
        buf for buf in layout["memory"]["activations"]["buffers"] if buf["name"] == "recurrent_ssm_state"
    )
    assert conv_state["shape"] == "[3, 4, 20]"
    assert ssm_state["shape"] == "[3, 2, 4, 3]"

    ir1_ops = json.loads((out / "ir1_decode.json").read_text(encoding="utf-8"))["ops"]
    ir1_by_op = {(op["layer"], op["op"]): op for op in ir1_ops if "layer" in op}
    ir1_mamba_in = ir1_by_op[(0, "mamba_in_proj")]
    assert ir1_mamba_in["dataflow"]["inputs"]["x"]["slot"] == "layer_input"
    assert ir1_mamba_in["dataflow"]["inputs"]["x"]["from_op"] == ir1_by_op[(0, "block_rmsnorm")]["op_id"]

    lowered_ops = json.loads(lowered.read_text(encoding="utf-8"))["operations"]
    by_op = {(op["layer"], op["op"]): op for op in lowered_ops}

    attention_layer_ops = [op["op"] for op in lowered_ops if op.get("layer") == 1]
    assert "rope_qk" not in attention_layer_ops
    assert "kv_cache_store" in attention_layer_ops
    assert attention_layer_ops.index("v_proj") < attention_layer_ops.index("kv_cache_store") < attention_layer_ops.index("attn")
    attn = by_op[(1, "attn")]
    assert attn["kernel"] == "attention_forward_decode_head_major_gqa_flash"
    assert attn["_kv_cache_read_layer"] == 1
    attn_inputs = attn.get("inputs") or attn.get("activations") or {}
    assert attn_inputs["k_cache"]["buffer"] == "kv_cache"
    assert attn_inputs["v_cache"]["buffer"] == "kv_cache"

    mamba_in = by_op[(0, "mamba_in_proj")]
    assert mamba_in["kernel"] == "gemv_bf16"
    assert mamba_in["activations"]["x"]["buffer"] == "layer_input"

    mlp_up = by_op[(2, "mlp_up")]
    relu2 = by_op[(2, "relu2")]
    mlp_down = by_op[(2, "mlp_down")]
    assert mlp_up["kernel"] == "gemv_bf16"
    assert mlp_up["activations"]["x"]["buffer"] == "layer_input"
    assert relu2["activations"]["x"]["buffer"] == "mlp_scratch"
    assert relu2["outputs"]["out"]["buffer"] == "mlp_scratch"
    assert mlp_down["activations"]["x"]["buffer"] == "mlp_scratch"


def test_glm4_safetensors_to_bump_uses_declarative_source_map(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "glm4"
    out = tmp_path / "out_glm4"
    checkpoint.mkdir()
    out.mkdir()

    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4ForCausalLM"],
                "model_type": "glm4",
                "num_hidden_layers": 1,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "partial_rotary_factor": 0.5,
                "vocab_size": 32,
                "max_position_embeddings": 64,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-5,
                "attention_bias": True,
                "tie_word_embeddings": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tiny_bpe_tokenizer(checkpoint, vocab_size=32)

    tensors = {
        "model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.float32),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(8, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.bias": torch.randn(8, dtype=torch.float32),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.k_proj.bias": torch.randn(4, dtype=torch.float32),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.v_proj.bias": torch.randn(4, dtype=torch.float32),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.o_proj.bias": torch.randn(8, dtype=torch.float32),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(8, dtype=torch.float32),
        "lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
    }
    st.save_file(tensors, checkpoint / "model.safetensors")

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    names = [entry["name"] for entry in manifest["entries"]]
    entries = {entry["name"]: entry for entry in manifest["entries"]}

    assert manifest["model"] == "glm4"
    assert manifest["template"]["name"] == "glm4"
    assert (
        manifest["template"]["contract"]["chat_contract"]
        ["force_bos_text_if_tokenizer_add_bos_false"]
        == "[gMASK]<sop>\n"
    )
    assert manifest["has_attention_biases"] is True
    assert manifest["has_qk_norm"] is False
    assert manifest["config"]["rotary_dim"] == 2
    assert manifest["config"]["partial_rotary_factor"] == 0.5
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert "layer.0.bq" in names and "layer.0.bo" in names
    assert entries["layer.0.w1"]["source_name"].endswith("gate_proj.weight+model.layers.0.mlp.up_proj.weight")
    assert entries["layer.0.w1"]["shape"] == [16, 8]
    assert entries["layer.0.b1"]["shape"] == [32]
    assert "output.weight" in names


def test_gemma4_assistant_safetensors_to_bump_maps_q_only_drafter(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "gemma4_assistant"
    out = tmp_path / "out_gemma4_assistant"
    checkpoint.mkdir()
    out.mkdir()

    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4AssistantForCausalLM"],
                "model_type": "gemma4_assistant",
                "backbone_hidden_size": 16,
                "tie_word_embeddings": True,
                "use_ordered_embeddings": False,
                "text_config": {
                    "model_type": "gemma4_text",
                    "attention_bias": False,
                    "attention_k_eq_v": True,
                    "bos_token_id": 2,
                    "eos_token_id": 1,
                    "global_head_dim": 8,
                    "head_dim": 4,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "max_position_embeddings": 128,
                    "num_attention_heads": 2,
                    "num_global_key_value_heads": 1,
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 2,
                    "num_kv_shared_layers": 2,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                        "full_attention": {
                            "partial_rotary_factor": 0.25,
                            "rope_theta": 1000000.0,
                            "rope_type": "proportional",
                        },
                        "sliding_attention": {
                            "rope_theta": 10000.0,
                            "rope_type": "default",
                        },
                    },
                    "sliding_window": 32,
                    "tie_word_embeddings": True,
                    "vocab_size": 32,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tiny_bpe_tokenizer(checkpoint, vocab_size=32)

    tensors = {
        "model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
        "pre_projection.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "post_projection.weight": torch.randn(16, 8, dtype=torch.bfloat16),
    }
    q_dims = [8, 16]
    for layer, q_dim in enumerate(q_dims):
        pfx = f"model.layers.{layer}"
        tensors.update(
            {
                f"{pfx}.input_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                f"{pfx}.pre_feedforward_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                f"{pfx}.post_attention_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                f"{pfx}.post_feedforward_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                f"{pfx}.layer_scalar": torch.ones(1, dtype=torch.bfloat16),
                f"{pfx}.self_attn.q_proj.weight": torch.randn(q_dim, 8, dtype=torch.bfloat16),
                f"{pfx}.self_attn.q_norm.weight": torch.ones(q_dim // 2, dtype=torch.bfloat16),
                f"{pfx}.self_attn.o_proj.weight": torch.randn(8, q_dim, dtype=torch.bfloat16),
                f"{pfx}.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
                f"{pfx}.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
                f"{pfx}.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
            }
        )
    st.save_file(tensors, checkpoint / "model.safetensors")

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    names = {entry["name"] for entry in manifest["entries"]}

    assert manifest["model"] == "gemma4_assistant"
    assert manifest["template"]["name"] == "gemma4_assistant"
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert cfg["attention_k_eq_v"] is True
    assert cfg["assistant_role"] == "mtp_drafter"
    assert cfg["assistant_projection_mode"] == "mtp_bridge"
    assert cfg["assistant_layer_scalar_mode"] == "layer_output_scale"
    assert cfg["standalone_text_inference_supported"] is False
    assert cfg["layer_kinds"] == ["sliding_attention_q_only_k_eq_v", "full_attention_q_only_k_eq_v"]
    assert cfg["layer_q_dim"] == [8, 16]
    assert cfg["layer_q_norm_dim"] == [4, 8]
    assert cfg["layer_q_head_dim"] == [4, 8]
    assert cfg["layer_k_head_dim"] == [4, 8]
    assert cfg["layer_v_head_dim"] == [4, 8]
    assert cfg["layer_rotary_dim"] == [4, 8]
    assert "assistant.pre_projection" in names
    assert "assistant.post_projection" in names
    assert "layer.0.wq" in names
    assert "layer.0.q_norm" in names
    assert "layer.0.wk" not in names
    assert "layer.0.wv" not in names


def test_kimi_vl_safetensors_to_bump_dry_run_maps_text_decoder(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "kimi_vl"
    out = tmp_path / "out_kimi_vl"
    checkpoint.mkdir()
    out.mkdir()

    config = {
        "architectures": ["KimiVLForConditionalGeneration"],
        "model_type": "kimi_vl",
        "text_config": {
            "num_hidden_layers": 2,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "kv_lora_rank": 4,
            "q_lora_rank": None,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.446,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "rope_theta": 800000.0,
            "tie_word_embeddings": False,
        },
        "vision_config": {
            "model_type": "moonvit",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 14,
        },
    }
    (checkpoint / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")

    tensors = {
        "language_model.model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "language_model.model.norm.weight": torch.ones(8, dtype=torch.float32),
        "language_model.lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
    }
    for layer in range(2):
        pfx = f"language_model.model.layers.{layer}"
        tensors.update(
            {
                f"{pfx}.input_layernorm.weight": torch.ones(8, dtype=torch.float32),
                f"{pfx}.post_attention_layernorm.weight": torch.ones(8, dtype=torch.float32),
                f"{pfx}.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
                f"{pfx}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(6, 8, dtype=torch.bfloat16),
                f"{pfx}.self_attn.kv_a_layernorm.weight": torch.ones(4, dtype=torch.float32),
                f"{pfx}.self_attn.kv_b_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
                f"{pfx}.self_attn.o_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
            }
        )
    tensors.update(
        {
            "language_model.model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
            "language_model.model.layers.0.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
            "language_model.model.layers.0.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
            "language_model.model.layers.1.mlp.gate.weight": torch.randn(2, 8, dtype=torch.float32),
            "language_model.model.layers.1.mlp.gate.e_score_correction_bias": torch.randn(2, dtype=torch.float32),
            "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
            "language_model.model.layers.1.mlp.shared_experts.up_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
            "language_model.model.layers.1.mlp.shared_experts.down_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        }
    )
    for expert in range(2):
        tensors.update(
            {
                f"language_model.model.layers.1.mlp.experts.{expert}.gate_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
                f"language_model.model.layers.1.mlp.experts.{expert}.up_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
                f"language_model.model.layers.1.mlp.experts.{expert}.down_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
            }
        )
    st.save_file(tensors, checkpoint / "model.safetensors")

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    names = {entry["name"] for entry in manifest["entries"]}
    entries = {entry["name"]: entry for entry in manifest["entries"]}

    assert manifest["model"] == "kimi_vl"
    assert manifest["template"]["name"] == "kimi_vl"
    assert manifest["config"]["layer_kinds"] == ["mla_dense_mlp", "mla_moe"]
    assert manifest["config"]["rotary_dim"] == 2
    assert manifest["config"]["mla_q_head_dim"] == 4
    assert manifest["config"]["layer_moe_policy"] == ["none", "routed_swiglu"]
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert entries["layer.1.moe_expert_gate"]["shape"] == [2, 4, 8]
    assert entries["layer.1.moe_expert_up"]["shape"] == [2, 4, 8]
    assert entries["layer.1.moe_expert_down"]["shape"] == [2, 8, 4]
    assert entries["layer.0.mla_kv_b_proj"]["dtype"] == "bf16"
    assert {
        "token_emb",
        "layer.0.mla_q_proj",
        "layer.0.mla_kv_a_proj",
        "layer.0.mla_kv_a_norm",
        "layer.0.mla_kv_b_proj",
        "layer.0.mlp_gate",
        "layer.1.moe_router",
        "layer.1.moe_router_bias",
        "layer.1.moe_expert_gate",
        "layer.1.moe_expert_up",
        "layer.1.moe_expert_down",
        "layer.1.moe_shared_gate",
        "final_ln_weight",
        "final_ln_bias",
        "output.weight",
    }.issubset(names)



def _write_tiny_qwen3vl_checkpoint(checkpoint: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
                "image_token_id": 151655,
                "vision_start_token_id": 151652,
                "vision_end_token_id": 151653,
                "tie_word_embeddings": False,
                "text_config": {
                    "model_type": "qwen3_vl_text",
                    "num_hidden_layers": 1,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "vocab_size": 32,
                    "max_position_embeddings": 64,
                    "rope_theta": 5000000.0,
                    "rms_norm_eps": 1e-6,
                    "rope_scaling": {
                        "mrope_interleaved": True,
                        "mrope_section": [1, 1, 2],
                        "rope_type": "default",
                    },
                },
                "vision_config": {
                    "model_type": "qwen3_vl",
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 12,
                    "num_heads": 2,
                    "out_hidden_size": 8,
                    "patch_size": 2,
                    "temporal_patch_size": 2,
                    "spatial_merge_size": 2,
                    "num_position_embeddings": 4,
                    "deepstack_visual_indexes": [0],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (checkpoint / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_mean": [0.1, 0.2, 0.3],
                "image_std": [0.4, 0.5, 0.6],
                "min_pixels": 16,
                "max_pixels": 4096,
                "size": {"shortest_edge": 4},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tiny_bpe_tokenizer(checkpoint, vocab_size=32)

    tensors = {
        "model.language_model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
        "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "model.language_model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "model.visual.patch_embed.proj.weight": torch.randn(8, 3, 2, 2, 2, dtype=torch.bfloat16),
        "model.visual.patch_embed.proj.bias": torch.randn(8, dtype=torch.bfloat16),
        "model.visual.pos_embed.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.visual.blocks.0.norm1.weight": torch.ones(8, dtype=torch.bfloat16),
        "model.visual.blocks.0.norm1.bias": torch.zeros(8, dtype=torch.bfloat16),
        "model.visual.blocks.0.norm2.weight": torch.ones(8, dtype=torch.bfloat16),
        "model.visual.blocks.0.norm2.bias": torch.zeros(8, dtype=torch.bfloat16),
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(24, 8, dtype=torch.bfloat16),
        "model.visual.blocks.0.attn.qkv.bias": torch.randn(24, dtype=torch.bfloat16),
        "model.visual.blocks.0.attn.proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.visual.blocks.0.attn.proj.bias": torch.randn(8, dtype=torch.bfloat16),
        "model.visual.blocks.0.mlp.linear_fc1.weight": torch.randn(12, 8, dtype=torch.bfloat16),
        "model.visual.blocks.0.mlp.linear_fc1.bias": torch.randn(12, dtype=torch.bfloat16),
        "model.visual.blocks.0.mlp.linear_fc2.weight": torch.randn(8, 12, dtype=torch.bfloat16),
        "model.visual.blocks.0.mlp.linear_fc2.bias": torch.randn(8, dtype=torch.bfloat16),
        "model.visual.merger.norm.weight": torch.ones(8, dtype=torch.bfloat16),
        "model.visual.merger.norm.bias": torch.zeros(8, dtype=torch.bfloat16),
        "model.visual.merger.linear_fc1.weight": torch.randn(32, 32, dtype=torch.bfloat16),
        "model.visual.merger.linear_fc1.bias": torch.randn(32, dtype=torch.bfloat16),
        "model.visual.merger.linear_fc2.weight": torch.randn(8, 32, dtype=torch.bfloat16),
        "model.visual.merger.linear_fc2.bias": torch.randn(8, dtype=torch.bfloat16),
        "model.visual.deepstack_merger_list.0.norm.weight": torch.ones(32, dtype=torch.bfloat16),
        "model.visual.deepstack_merger_list.0.norm.bias": torch.zeros(32, dtype=torch.bfloat16),
        "model.visual.deepstack_merger_list.0.linear_fc1.weight": torch.randn(32, 32, dtype=torch.bfloat16),
        "model.visual.deepstack_merger_list.0.linear_fc1.bias": torch.randn(32, dtype=torch.bfloat16),
        "model.visual.deepstack_merger_list.0.linear_fc2.weight": torch.randn(8, 32, dtype=torch.bfloat16),
        "model.visual.deepstack_merger_list.0.linear_fc2.bias": torch.randn(8, dtype=torch.bfloat16),
    }
    st.save_file(tensors, checkpoint / "model.safetensors")


def test_qwen3vl_safetensors_auto_text_ignores_vision(tmp_path: Path) -> None:
    _require_torch_safetensors()
    checkpoint = tmp_path / "qwen3vl"
    out = tmp_path / "out_qwen3vl_text"
    out.mkdir()
    _write_tiny_qwen3vl_checkpoint(checkpoint)

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    names = [entry["name"] for entry in manifest["entries"]]
    assert manifest["model"] == "qwen3vl"
    assert manifest["config"]["mrope_sections"] == [1, 1, 2, 0]
    assert manifest["config"]["mrope_interleaved"] is True
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert any(row["reason"] == "vision_tower_not_in_decoder_pass" for row in audit["ignored_source_tensors"])
    assert "layer.0.q_norm" in names
    assert "layer.0.k_norm" in names
    assert "output.weight" in names
    assert manifest["config"]["tie_word_embeddings"] is False
    assert manifest["config"]["num_deepstack_layers"] == 1
    assert manifest["config"]["decoder_norm_storage_boundary"] == "bf16"
    assert manifest["config"]["decoder_norm_reduction_policy"] == "pytorch_avx2_cascade_exact"
    assert manifest["config"]["decoder_qk_norm_reduction_policy"] == "pytorch_avx2_cascade_exact"
    assert manifest["config"]["decoder_prefill_projection_storage_boundary"] == "bf16"
    assert manifest["config"]["decoder_projection_reduction_policy"] == "pytorch_onednn_brgemm_exact"
    assert manifest["config"]["decoder_residual_storage_boundary"] == "bf16"
    assert manifest["config"]["decoder_mrope_storage_boundary"] == "pytorch_bf16_exact"
    assert manifest["config"]["decoder_swiglu_storage_boundary"] == "pytorch_bf16_exact"
    assert manifest["config"]["decode_kv_cache_dtype"] == "bf16"
    assert "decoder_decode_projection_storage_boundary" not in manifest["config"]

    real_out = tmp_path / "out_qwen3vl_text_real"
    real_out.mkdir()
    subprocess.run(
        [
            sys.executable, str(script),
            "--checkpoint", str(checkpoint),
            "--output", str(real_out / "weights.bump"),
            "--config-out", str(real_out / "config.json"),
            "--manifest-out", str(real_out / "weights_manifest.json"),
            "--arch", "auto",
        ],
        check=True,
    )
    real_manifest = json.loads((real_out / "weights_manifest.json").read_text(encoding="utf-8"))
    preview_offsets = {entry["name"]: entry["file_offset"] for entry in manifest["entries"]}
    real_offsets = {entry["name"]: entry["file_offset"] for entry in real_manifest["entries"]}
    assert preview_offsets == real_offsets
    assert (real_out / "weights.bump").stat().st_size > 0

    build_ir = Path("version/v8/scripts/build_ir_v8.py")
    lowered = out / "lowered_text.json"
    call = out / "call_text.json"
    layout = out / "layout_text.json"
    subprocess.run(
        [
            sys.executable, str(build_ir),
            "--manifest", str(out / "weights_manifest.json"),
            "--mode", "decode",
            "--output", str(out / "ir1_text.json"),
            "--layout-output", str(layout),
            "--lowered-output", str(lowered),
            "--call-output", str(call),
            "--context-len", "4",
        ],
        check=True,
    )
    text_ops = json.loads(lowered.read_text(encoding="utf-8"))["operations"]
    layer0_projection_ops = {
        op["op"]: op["kernel"] for op in text_ops
        if op.get("layer") == 0 and op.get("op") in {"q_proj", "k_proj", "v_proj"}
    }
    assert layer0_projection_ops == {
        "q_proj": "gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage",
        "k_proj": "gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage",
        "v_proj": "gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage",
    }
    rope_op = next(op for op in text_ops if op.get("layer") == 0 and op.get("op") == "rope_qk")
    assert rope_op["kernel"] == "mrope_qk_text_imrope_bf16_pytorch_storage"
    assert rope_op["resolved_contract"]["resolved_contract_id"] == (
        "text_imrope_bf16_input_pytorch_bf16_compute_bf16_output"
    )
    layer0_norm_ops = {
        op["op"]: op["kernel"] for op in text_ops
        if op.get("layer") == 0 and op.get("op") in {"rmsnorm", "qk_norm"}
    }
    assert layer0_norm_ops == {
        "rmsnorm": "rmsnorm_forward_pytorch_bf16_storage",
        "qk_norm": "qk_norm_forward_pytorch_bf16_storage",
    }
    attention_op = next(
        op for op in text_ops if op.get("layer") == 0 and op.get("op") == "attn"
    )
    assert attention_op["kernel"] == (
        "attention_forward_decode_head_major_gqa_bf16cache_pytorch_contract"
    )
    assert attention_op["required_contract"]["tensor.kv.dtype"] == "bf16"
    cache_store = next(
        op for op in text_ops if op.get("layer") == 0 and op.get("op") == "kv_cache_store"
    )
    assert cache_store["kernel"] == "kv_cache_store_bf16"
    layout_doc = json.loads(layout.read_text(encoding="utf-8"))
    kv_cache = next(
        item
        for item in layout_doc["memory"]["activations"]["buffers"]
        if item["name"] == "kv_cache"
    )
    assert kv_cache["dtype"] == "bf16"
    assert not any(op.get("op") == "qkv_proj" for op in text_ops)

    prefill_lowered = out / "lowered_text_prefill.json"
    subprocess.run(
        [
            sys.executable, str(build_ir),
            "--manifest", str(out / "weights_manifest.json"),
            "--mode", "prefill",
            "--output", str(out / "ir1_text_prefill.json"),
            "--layout-output", str(out / "layout_text_prefill.json"),
            "--lowered-output", str(prefill_lowered),
            "--call-output", str(out / "call_text_prefill.json"),
            "--context-len", "4",
        ],
        check=True,
    )
    prefill_ops = json.loads(prefill_lowered.read_text(encoding="utf-8"))["operations"]
    prefill_attention = next(
        op for op in prefill_ops if op.get("layer") == 0 and op.get("op") == "attn"
    )
    assert prefill_attention["kernel"] == (
        "attention_forward_causal_head_major_gqa_prefill_full_bf16cache_pytorch_contract"
    )
    assert prefill_attention["required_contract"]["tensor.kv.dtype"] == "bf16"
    assert any(op.get("kernel") == "kv_cache_store_batch_bf16" for op in prefill_ops)


def test_qwen3vl_safetensors_vision_maps_temporal_patch_split(tmp_path: Path) -> None:
    _require_torch_safetensors()
    checkpoint = tmp_path / "qwen3vl"
    out = tmp_path / "out_qwen3vl_vision"
    out.mkdir()
    _write_tiny_qwen3vl_checkpoint(checkpoint)

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "qwen3_vl_vision",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in manifest["entries"]}
    assert manifest["model"] == "qwen3_vl_vision"
    assert manifest["config"]["deepstack_layer_indices"] == [0]
    assert manifest["config"]["projector_total_out_dim"] == 16
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert entries["v.patch_embd.weight"]["shape"] == [8, 12]
    assert entries["v.patch_embd.weight.1"]["shape"] == [8, 12]
    assert entries["v.patch_embd.weight"]["transform"] == "qwen3vl_patch_temporal_0"
    assert entries["v.patch_embd.weight.1"]["transform"] == "qwen3vl_patch_temporal_1"
    assert entries["v.patch_embd.weight"]["size"] == 8 * 12 * 2
    assert entries["v.position_embd.weight"]["dtype"] == "fp32"
    assert entries["v.deepstack.0.fc1.weight"]["shape"] == [32, 32]
    assert "model.visual.patch_embed.proj.weight" in audit["source_to_targets"]
    assert audit["source_to_targets"]["model.visual.patch_embed.proj.weight"] == [
        "v.patch_embd.weight",
        "v.patch_embd.weight.1",
    ]
    assert manifest["config"]["rope_layout"] == "multi_section_2d"
    assert manifest["config"]["vision_mrope_n_dims"] == 4
    assert manifest["config"]["vision_mrope_sections"] == [1, 1, 0, 0]
    assert manifest["config"]["vision_mrope_storage_boundary"] == "bf16"
    assert manifest["config"]["position_interpolation_policy"] == "align_corners_bilinear"
    assert manifest["config"]["vision_position_storage_boundary"] == "bf16"
    assert manifest["config"]["vision_layernorm_storage_boundary"] == "bf16"
    assert manifest["config"]["vision_layernorm_reduction_policy"] == "pytorch_welford_exact"
    assert manifest["config"]["vision_projection_storage_boundary"] == "bf16"
    assert manifest["config"]["vision_attention_storage_boundary"] == "bf16"
    assert manifest["config"]["vision_residual_storage_boundary"] == "bf16"
    assert manifest["config"]["vision_activation_storage_boundary"] == "bf16"
    assert manifest["config"]["vision_patch_projection_reduction_policy"] == "pytorch_onednn_conv3d_exact"
    assert (out / "weights.bump").stat().st_size > 0

    build_ir = Path("version/v8/scripts/build_ir_v8.py")
    codegen = Path("version/v8/scripts/codegen_v8.py")
    lowered = out / "lowered_vision.json"
    call = out / "lowered_vision_call.json"
    layout = out / "layout_vision.json"
    generated_c = out / "generated_vision.c"
    subprocess.run(
        [
            sys.executable,
            str(build_ir),
            "--manifest",
            str(out / "weights_manifest.json"),
            "--mode",
            "prefill",
            "--output",
            str(out / "ir1_vision.json"),
            "--layout-output",
            str(layout),
            "--lowered-output",
            str(lowered),
            "--call-output",
            str(call),
            "--context-len",
            "4",
        ],
        check=True,
    )
    lowered_ops = json.loads(lowered.read_text(encoding="utf-8"))["operations"]
    kernels_by_op = {op["op"]: op.get("kernel") for op in lowered_ops}
    assert kernels_by_op["position_embeddings"] == "position_embeddings_add_tiled_2d_align_corners_bf16"
    call_ops = json.loads(call.read_text(encoding="utf-8"))["operations"]
    position_call = next(op for op in call_ops if op["op"] == "position_embeddings")
    assert position_call["function"] == "position_embeddings_add_tiled_2d_align_corners_bf16"
    assert position_call["resolved_contract"]["resolved_contract_id"] == "bf16_tiled_2d_align_corners_rne_residual"
    assert position_call["resolved_contract"]["kernel_id"] == "position_embeddings_add_tiled_2d_align_corners_bf16"
    rope_call = next(op for op in call_ops if op["op"] == "rope_qk")
    rope_args = {arg["name"]: arg["expr"] for arg in rope_call["args"]}
    assert rope_call["function"] == "mrope_qk_vision_bf16_pytorch_storage"
    assert rope_call["resolved_contract"]["resolved_contract_id"] == "vision_mrope_pytorch_mkl_fp32_input_fp32_compute_bf16_output"
    assert rope_call["resolved_contract"]["kernel_id"] == "mrope_qk_vision_bf16_pytorch_storage"
    assert rope_args["n_dims"] == "4"
    assert [rope_args[f"section_{i}"] for i in range(4)] == ["1", "1", "0", "0"]
    assert kernels_by_op["patch_projection_image"] == "patch_projection_image_bf16_pytorch_onednn_conv3d_storage"
    assert "patchify" not in kernels_by_op
    assert "patch_proj" not in kernels_by_op
    assert "patch_proj_aux" not in kernels_by_op
    assert "add_stream" not in kernels_by_op
    assert "patch_bias_add" not in kernels_by_op
    patch_projection_call = next(
        op for op in call_ops if op["op"] == "patch_projection_image"
    )
    assert patch_projection_call["function"] == "patch_projection_image_bf16_pytorch_onednn_conv3d_storage"
    assert patch_projection_call["resolved_contract"]["resolved_contract_id"] == "patch_projection_bf16_pytorch_onednn_conv3d_storage"
    assert patch_projection_call["resolved_contract"]["kernel_id"] == "patch_projection_image_bf16_pytorch_onednn_conv3d_storage"
    assert [
        checkpoint["id"]
        for checkpoint in patch_projection_call["semantic_checkpoints"]
    ] == ["vision.frontend.patch_bias.output"]
    for op_name in (
        "qkv_packed_proj",
        "out_proj",
        "mlp_up",
        "mlp_down",
        "projector_fc1",
        "projector_fc2",
    ):
        assert kernels_by_op[op_name] == "gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage", op_name

    subprocess.run(
        [
            sys.executable,
            str(codegen),
            "--ir",
            str(call),
            "--layout",
            str(layout),
            "--output",
            str(generated_c),
        ],
        check=True,
    )
    generated = generated_c.read_text(encoding="utf-8")
    assert "patch_projection_image_bf16_pytorch_onednn_conv3d_storage(" in generated
    assert "gemm_nt_bf16(" not in generated
    assert "gemm_naive_parallel(" not in generated
    assert "gemm_blocked_serial(" not in generated

    native_manifest = json.loads(
        (out / "weights_manifest.json").read_text(encoding="utf-8")
    )
    converter = _load_converter()
    converter._apply_vision_numerics_profile(
        native_manifest["config"], "qwen3_vl_vision", "native"
    )
    native_manifest_path = out / "weights_manifest_native.json"
    native_manifest_path.write_text(
        json.dumps(native_manifest, indent=2) + "\n", encoding="utf-8"
    )
    native_lowered = out / "lowered_vision_native.json"
    native_call = out / "lowered_vision_native_call.json"
    subprocess.run(
        [
            sys.executable,
            str(build_ir),
            "--manifest",
            str(native_manifest_path),
            "--mode",
            "prefill",
            "--output",
            str(out / "ir1_vision_native.json"),
            "--layout-output",
            str(out / "layout_vision_native.json"),
            "--lowered-output",
            str(native_lowered),
            "--call-output",
            str(native_call),
            "--context-len",
            "4",
        ],
        check=True,
    )
    native_calls = json.loads(native_call.read_text(encoding="utf-8"))["operations"]
    native_functions = {op.get("function") for op in native_calls}
    assert "patch_projection_image_bf16_native_storage" in native_functions
    assert "mrope_qk_vision_bf16_storage" in native_functions
    assert "gemm_nt_bf16_native_bf16_storage" in native_functions
    assert (
        "attention_forward_full_head_major_gqa_sdpa_bf16_storage"
        in native_functions
    )
    assert "gelu_erf_bf16_storage" in native_functions
    assert not any(
        function
        and any(marker in function for marker in ("onednn", "mkl", "amx", "sleef"))
        for function in native_functions
    )
