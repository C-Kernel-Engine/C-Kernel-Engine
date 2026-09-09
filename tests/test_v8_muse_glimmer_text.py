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
CERTIFIER_PATH = (
    ROOT / "version" / "v8" / "scripts" / "certify_muse_glimmer_text_v8.py"
)
CERTIFICATION_ARTIFACT = (
    ROOT
    / "docs"
    / "notes"
    / "artifacts"
    / "muse_glimmer_text_parity_128_2026-09-09.json"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


converter = _load("muse_converter_test", CONVERTER_PATH)
builder = _load("muse_builder_test", BUILDER_PATH)
certifier = _load("muse_certifier_test", CERTIFIER_PATH)


def test_committed_muse_certification_artifact_is_complete_and_consumable() -> None:
    report = json.loads(CERTIFICATION_ARTIFACT.read_text(encoding="utf-8"))
    assert report["schema"] == "cke.v8.muse_glimmer_text_parity"
    assert report["status"] == "pass"
    assert report["comparison"] == {
        "numeric": "forced-reference-token history",
        "trajectory": "free-running greedy history",
        "exactness": "float32 IEEE-754 bit-pattern equality",
    }

    provenance = report["provenance"]
    required_hashes = {
        "libmodel_sha256",
        "engine_sha256",
        "generated_c_sha256",
        "runtime_bundle_sha256",
        "reference_manifest_sha256",
        "loaded_engine_sha256",
    }
    for field in required_hashes:
        assert len(provenance[field]) == 64
        int(provenance[field], 16)
    assert provenance["loaded_engine_sha256"] == provenance["engine_sha256"]
    assert Path(provenance["loaded_engine_path"]).name == "libckernel_engine.so"

    cases = report["cases"]
    assert [case["name"] for case in cases] == list(certifier.CASES)
    assert sum(case["generated_tokens"] for case in cases) == 384
    for case in cases:
        expected = case["generated_tokens"]
        assert case["status"] == "pass"
        assert case["prompt_tokens"] > 0
        assert expected == 128
        assert len(case["reference_ids"]) == expected
        assert certifier._case_passes(
            case["free_running_history"],
            case["forced_reference_history"],
            expected,
        )
        assert case["free_running_history"]["actual_ids"] == case["reference_ids"]
        assert case["forced_reference_history"]["actual_ids"] == case["reference_ids"]


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
    assert circuit["kernels"]["attn"].endswith("muse_eager_bf16_storage")
    assert circuit["kernels"]["attn_decode"].endswith("muse_eager_bf16_storage")
    assert circuit["kernels"]["attn_sliding"].endswith(
        "muse_eager_bf16_storage_sliding"
    )
    assert circuit["kernels"]["attn_sliding_decode"].endswith(
        "muse_eager_bf16_storage_sliding"
    )

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
        if mode == "decode":
            layer_three = [
                op["op"] for op in lowered["operations"] if op["layer"] == 3
            ]
            assert layer_three.index("qk_norm_no_weight_scaled") < layer_three.index(
                "kv_cache_store"
            ) < layer_three.index("attn")

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
    assert "attention_forward_causal_head_major_gqa_muse_eager_bf16_storage" in emitted
    assert "attention_forward_causal_head_major_gqa_muse_eager_bf16_storage_sliding" in emitted
    assert "attention_forward_decode_head_major_gqa_muse_eager_bf16_storage" in emitted
    assert "attention_forward_decode_head_major_gqa_muse_eager_bf16_storage_sliding" in emitted


def test_muse_eager_attention_workspace_budget_is_enforced() -> None:
    provider = json.loads(
        (
            ROOT
            / "version/v8/kernel_maps/attention_forward_causal_head_major_gqa_muse_eager_bf16_storage.json"
        ).read_text(encoding="utf-8")
    )
    op = {
        "kernel": provider["id"],
        "op": "attention",
        "layer": 0,
        "scratch": provider["scratch"],
        "params": {"num_heads": 32, "num_kv_heads": 2, "head_dim": 128},
    }
    with pytest.raises(RuntimeError, match="HARD SCRATCH BUDGET FAULT"):
        builder._required_kernel_call_scratch_bytes(
            [op], {"context_length": 131072}, 2048
        )


def test_muse_attention_adds_no_kernel_allocation_debt() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "version/v8/scripts/audit_kernel_allocations_v8.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )
    source = (ROOT / "src/kernels/attention_kernels.c").read_text(encoding="utf-8")
    body = source.split("static void ck_attention_muse_eager_bf16_storage_impl", 1)[1]
    body = body.split("void attention_forward_causal_head_major_gqa_muse", 1)[0]
    assert "malloc(" not in body
    assert "free(" not in body


def _reference_manifest() -> dict:
    return {
        "schema": "cke.v8.muse_glimmer_text_reference",
        "max_tokens": 128,
        "cases": [
            {
                "name": name,
                "prompt_tokens": 2,
                "generated_tokens": 2,
                "output": "fixture",
            }
            for name in certifier.CASES
        ],
    }


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda manifest: manifest.update(cases=[]), "nonempty cases"),
        (
            lambda manifest: manifest.update(cases=manifest["cases"][:-1]),
            "do not match",
        ),
        (lambda manifest: manifest.update(max_tokens=0), "positive integer"),
    ],
)
def test_muse_certifier_rejects_incomplete_manifests(mutate, match: str) -> None:
    manifest = _reference_manifest()
    mutate(manifest)
    with pytest.raises(ValueError, match=match):
        certifier._validate_reference_manifest(manifest)


def test_muse_certifier_validates_trajectory_arrays(tmp_path: Path) -> None:
    case = _reference_manifest()["cases"][0]
    fixture = tmp_path / "case.npz"
    import numpy as np

    np.savez(
        fixture,
        prompt_ids=np.asarray([1, 2], dtype=np.int32),
        generated_ids=np.asarray([3, 4], dtype=np.int32),
        logits=np.zeros((2, 8), dtype=np.float32),
    )
    with np.load(fixture, allow_pickle=False) as arrays:
        prompt, generated, logits = certifier._validate_reference_arrays(
            case, arrays, 8
        )
    assert prompt.tolist() == [1, 2]
    assert generated.tolist() == [3, 4]
    assert logits.shape == (2, 8)

    np.savez(
        fixture,
        prompt_ids=np.asarray([1, 2], dtype=np.int32),
        generated_ids=np.asarray([3, 4], dtype=np.int32),
        logits=np.asarray([[0.0] * 8, [float("nan")] * 8], dtype=np.float32),
    )
    with np.load(fixture, allow_pickle=False) as arrays, pytest.raises(
        ValueError, match="non-finite"
    ):
        certifier._validate_reference_arrays(case, arrays, 8)

    np.savez(
        fixture,
        prompt_ids=np.asarray([1, 2], dtype=np.int32),
        generated_ids=np.asarray([], dtype=np.int32),
        logits=np.empty((0, 8), dtype=np.float32),
    )
    with np.load(fixture, allow_pickle=False) as arrays, pytest.raises(
        ValueError, match="generated_ids must be nonempty"
    ):
        certifier._validate_reference_arrays(case, arrays, 8)


def test_muse_certifier_requires_all_runtime_logits_to_be_finite() -> None:
    free = {
        "first_token_divergence": None,
        "finite_logit_rows": 2,
    }
    forced = {
        "first_logit_divergence": None,
        "finite_logit_rows": 2,
    }
    assert certifier._case_passes(free, forced, 2)
    forced["finite_logit_rows"] = 1
    assert not certifier._case_passes(free, forced, 2)
    forced["finite_logit_rows"] = 2
    free["finite_logit_rows"] = 1
    assert not certifier._case_passes(free, forced, 2)


def test_muse_certifier_rejects_a_different_loaded_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = tmp_path / "requested" / "libckernel_engine.so"
    loaded = tmp_path / "loaded" / "libckernel_engine.so"
    requested.parent.mkdir()
    loaded.parent.mkdir()
    requested.write_bytes(b"requested")
    loaded.write_bytes(b"loaded")
    monkeypatch.setattr(certifier, "_resolved_symbol_library", lambda _lib, _symbol: loaded)
    with pytest.raises(RuntimeError, match="different CK engine"):
        certifier._verify_loaded_engine(object(), requested)


def test_muse_certifier_publishes_failure_for_empty_reference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = tmp_path / "runtime"
    reference = tmp_path / "reference"
    report = tmp_path / "report.json"
    runtime.mkdir()
    reference.mkdir()
    (reference / "reference.json").write_text(
        json.dumps(
            {
                "schema": "cke.v8.muse_glimmer_text_reference",
                "max_tokens": 128,
                "cases": [],
            }
        ),
        encoding="utf-8",
    )
    assert certifier.compare_cke(runtime, reference, report) == 3
    published = json.loads(report.read_text(encoding="utf-8"))
    assert published["status"] == "fail"
    assert "nonempty cases" in published["error"]
    capsys.readouterr()
