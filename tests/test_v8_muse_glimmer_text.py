import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = ROOT / "version" / "v8" / "scripts" / "convert_safetensors_to_bump_v8.py"
BUILDER_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


converter = _load("muse_converter_test", CONVERTER_PATH)
builder = _load("muse_builder_test", BUILDER_PATH)


def _text_config(num_layers: int = 4) -> dict:
    layer_types = [
        "full_attention" if (layer + 1) % 4 == 0 else "sliding_attention"
        for layer in range(num_layers)
    ]
    return {
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "model_type": "muse_glimmer",
        "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": 6656,
            "intermediate_size": 19968,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 202048,
            "max_position_embeddings": 131072,
            "sliding_window": 2048,
            "layer_types": layer_types,
            "layer_rope_theta": [0.0 if kind == "full_attention" else 500000.0 for kind in layer_types],
            "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0},
            "rms_norm_eps": 1.0e-5,
            "post_norm_eps": 1.0e-8,
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.19611613513818404,
            "final_logit_softcapping": 20.0,
            "tie_word_embeddings": False,
        },
    }


def test_muse_config_keeps_projection_gate_and_position_contracts_independent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp)
        (checkpoint / "config.json").write_text(json.dumps(_text_config()), encoding="utf-8")
        config = converter._build_config(checkpoint, "muse_glimmer_text", None)

    assert config["embed_dim"] == 6656
    assert config["attn_out_dim"] == 4096
    assert config["k_dim"] == config["v_dim"] == 256
    assert config["attn_gate_dim"] == 4096
    assert config["layer_sliding_window"] == [2048, 2048, 2048, 0]
    assert config["layer_rope_kind"] == ["swa", "swa", "swa", "none"]
    assert config["layer_rotary_dim"] == [128, 128, 128, 0]
    assert config["post_norm_eps"] == 1.0e-8
    assert config["qk_scale_factor"] == 3.87
    assert config["tie_word_embeddings"] is False


def test_muse_gate_lowering_uses_attention_width_instead_of_head_count() -> None:
    config = {
        "embed_dim": 6656,
        "num_heads": 32,
        "num_kv_heads": 2,
        "head_dim": 128,
        "attn_out_dim": 4096,
        "attn_gate_dim": 4096,
        "layer_attention_gate_dim": [4096],
    }
    projection = {}
    builder.apply_layer_attention_dims("attention_gate_projection", projection, 0, config)
    assert projection["_input_dim"] == 6656
    assert projection["_output_dim"] == 4096

    gate = {}
    builder.apply_layer_attention_dims("attn_gate_sigmoid_mul", gate, 0, config)
    assert gate["num_heads"] == 32
    assert gate["state_dim"] == 128


def test_muse_circuit_preserves_ordered_text_numerics() -> None:
    circuit = builder._load_builtin_template_doc("muse_glimmer_text")
    assert circuit is not None
    body = circuit["block_types"]["decoder"]["body"]["ops_by_kind"]
    sliding = [item if isinstance(item, str) else item["op"] for item in body["sliding_attention"]]
    full = [item if isinstance(item, str) else item["op"] for item in body["full_attention"]]

    assert "rope_qk" in sliding
    assert "rope_qk" not in full
    assert sliding.index("qk_norm_no_weight_scaled") < sliding.index("rope_qk")
    assert sliding.index("attn_sliding") < sliding.index("attn_gate_sigmoid_mul") < sliding.index("out_proj")
    assert sliding.count("post_attention_norm") == 1
    assert sliding.count("post_ffn_norm") == 1

    footer = [
        item if isinstance(item, str) else item["op"]
        for item in circuit["block_types"]["decoder"]["footer"]
    ]
    assert footer[-2:] == ["final_logit_scale", "final_logit_softcap"]
    assert circuit["contract"]["logits_contract"]["lm_head"] == "separate_output_weight"


def test_muse_model_map_accounts_for_text_and_defers_vision_explicitly() -> None:
    contract = converter._safetensors_arch_contract("muse_glimmer_text")
    assert contract["template"] == "muse_glimmer_text"
    targets = {row["target"] for row in contract["tensor_refs"]}
    assert {
        "token_emb",
        "layer.{L}.wq",
        "layer.{L}.wk",
        "layer.{L}.wv",
        "layer.{L}.mla_gate_proj",
        "layer.{L}.wo",
        "layer.{L}.post_attention_norm",
        "layer.{L}.post_ffn_norm",
        "final_ln_weight",
        "lm_head",
    } <= targets
    ignored = contract["ignored_source_tensors"]
    assert any(row.get("prefix") == "model.vision_tower." for row in ignored)


def test_muse_tiny_checkpoint_converts_lowers_and_emits_strict_c(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    checkpoint.mkdir()
    output.mkdir()

    hidden, heads, kv_heads, head_dim = 16, 2, 1, 8
    intermediate, layers, vocab = 32, 4, 32
    config = _text_config(layers)
    text = config["text_config"]
    text.update(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=vocab,
        max_position_embeddings=64,
        sliding_window=16,
        bos_token_id=1,
        eos_token_id=2,
    )
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tokenizer = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2}
            | {f"t{i}": i for i in range(3, vocab)},
            "merges": [],
        },
        "added_tokens": [
            {"id": 0, "content": "<unk>"},
            {"id": 1, "content": "<s>"},
            {"id": 2, "content": "</s>"},
        ],
    }
    (checkpoint / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    (checkpoint / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "add_bos_token": True,
                "add_eos_token": False,
            }
        ),
        encoding="utf-8",
    )

    def matrix(rows: int, columns: int):
        return torch.randn(rows, columns, dtype=torch.bfloat16)

    tensors = {
        "model.language_model.embed_tokens.weight": matrix(vocab, hidden),
        "model.language_model.norm.weight": torch.randn(hidden, dtype=torch.bfloat16),
        "lm_head.weight": matrix(vocab, hidden),
    }
    for layer in range(layers):
        prefix = f"model.language_model.layers.{layer}"
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            tensors[f"{prefix}.{norm}.weight"] = torch.randn(
                hidden, dtype=torch.bfloat16
            )
        tensors[f"{prefix}.self_attn.q_proj.weight"] = matrix(
            heads * head_dim, hidden
        )
        tensors[f"{prefix}.self_attn.k_proj.weight"] = matrix(
            kv_heads * head_dim, hidden
        )
        tensors[f"{prefix}.self_attn.v_proj.weight"] = matrix(
            kv_heads * head_dim, hidden
        )
        tensors[f"{prefix}.self_attn.gate_proj.weight"] = matrix(
            heads * head_dim, hidden
        )
        tensors[f"{prefix}.self_attn.o_proj.weight"] = matrix(
            hidden, heads * head_dim
        )
        tensors[f"{prefix}.mlp.gate_proj.weight"] = matrix(intermediate, hidden)
        tensors[f"{prefix}.mlp.up_proj.weight"] = matrix(intermediate, hidden)
        tensors[f"{prefix}.mlp.down_proj.weight"] = matrix(hidden, intermediate)
    safetensors.save_file(tensors, checkpoint / "model.safetensors")

    python = sys.executable
    subprocess.run(
        [
            python,
            str(CONVERTER_PATH),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output / "weights.bump"),
            "--config-out",
            str(output / "config.json"),
            "--manifest-out",
            str(output / "weights_manifest.json"),
            "--arch",
            "muse_glimmer_text",
        ],
        cwd=ROOT,
        check=True,
    )
    audit = json.loads((output / "conversion_audit.json").read_text())
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    manifest = json.loads((output / "weights_manifest.json").read_text())
    entries = {entry["name"]: entry for entry in manifest["entries"]}
    assert entries["layer.0.w1"]["shape"] == [2 * intermediate, hidden]
    assert entries["layer.0.w1"]["source_name"].endswith(
        "mlp.gate_proj.weight+model.language_model.layers.0.mlp.up_proj.weight"
    )
    assert "layer.0.w3" not in entries

    for mode in ("decode", "prefill"):
        command = [
            python,
            str(BUILDER_PATH),
            "--manifest",
            str(output / "weights_manifest.json"),
            "--mode",
            mode,
            "--context-len",
            "32",
            "--output",
            str(output / f"{mode}.ir1.json"),
            "--layout-output",
            str(output / f"{mode}.layout.json"),
            "--call-output",
            str(output / f"{mode}.call.json"),
            "--no-fusion",
        ]
        if mode == "prefill":
            command += ["--prefill-chunk-len", "8"]
        subprocess.run(command, cwd=ROOT, check=True)
        lowered = json.loads((output / f"{mode}.call.json").read_text())
        assert lowered["errors"] == []
        gate = next(
            op
            for op in lowered["operations"]
            if op["op"] == "attention_gate_projection" and op["layer"] == 0
        )
        gate_input = next(
            arg
            for arg in gate["args"]
            if arg["source"] in {"activation:x", "activation:a"}
        )
        assert gate_input["buffer_ref"] == "embedded_input"

    generated = output / "muse_tiny.c"
    subprocess.run(
        [
            python,
            str(ROOT / "version/v8/scripts/codegen_v8.py"),
            "--ir",
            str(output / "decode.call.json"),
            "--layout",
            str(output / "decode.layout.json"),
            "--prefill",
            str(output / "prefill.call.json"),
            "--prefill-layout",
            str(output / "prefill.layout.json"),
            "--output",
            str(generated),
            "--strict-contracts",
        ],
        cwd=ROOT,
        check=True,
    )
    emitted = generated.read_text(encoding="utf-8")
    assert "qk_norm_forward_muse_unweighted_scaled_pytorch_bf16_storage" in emitted
    assert "rmsnorm_forward_muse_centered_pytorch_bf16_storage" in emitted
    assert "rmsnorm_forward_muse_weighted_pytorch_bf16_storage" in emitted
    assert "rope_forward_qk_split_direct_muse_pytorch_bf16_storage" in emitted
    assert "swiglu_forward_pytorch_bf16_storage" in emitted
    assert "ck_residual_add_token_major_bf16_storage" in emitted
    assert "final_logit_scale_muse_pytorch_bf16_storage" in emitted
    assert "final_logit_softcap_muse_pytorch_bf16_storage" in emitted
    assert "gemm_nt_bf16_pytorch_onednn_3_12_brgemm_bf16_storage" in emitted
