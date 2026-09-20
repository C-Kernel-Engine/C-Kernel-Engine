from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "version" / "v8"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


converter = _load(
    "cohere_generated_decoder_converter",
    V8 / "scripts" / "convert_cohere_transcribe_gguf_to_bump_v8.py",
)
certifier = _load(
    "cohere_generated_decoder_certifier",
    V8 / "scripts" / "certify_cohere_generated_decoder_v8.py",
)


def test_decoder_component_selects_only_decoder_tensors() -> None:
    tensors = {
        "fe.window": object(),
        "enc.blk.0.attn.q.weight": object(),
        "dec.emb.weight": object(),
        "dec.blk.0.attn_q.weight": object(),
    }

    selected = converter._component_tensors(tensors, "audio_decoder")

    assert [name for name, _ in selected] == [
        "dec.emb.weight",
        "dec.blk.0.attn_q.weight",
    ]
    assert converter._component_tensors(tensors, "audio_encoder") == list(
        tensors.items()
    )


def test_decoder_weights_map_to_shared_decoder_roles() -> None:
    expected = {
        "dec.emb.weight": "token_emb",
        "dec.pos.weight": "pos_emb",
        "dec.emb_ln.weight": "embedding_ln_weight",
        "dec.emb_ln.bias": "embedding_ln_bias",
        "dec.out_ln.weight": "final_ln_weight",
        "dec.out_ln.bias": "final_ln_bias",
        "dec.head.weight": "lm_head",
        "dec.head.bias": "lm_head_bias",
        "dec.blk.3.attn_q.weight": "layer.3.wq",
        "dec.blk.3.cross_k.bias": "layer.3.cross_bk",
        "dec.blk.3.ffn_up.weight": "layer.3.w3",
        "dec.blk.3.ffn_down.bias": "layer.3.b2",
    }
    assert {
        source: converter._component_weight_name(source, "audio_decoder")
        for source in expected
    } == expected


def test_cohere_special_token_lookup_comes_from_gguf_vocabulary() -> None:
    metadata = {
        "tokenizer.ggml.tokens": ["<unk>", "<pad>", "<|endoftext|>"]
    }
    assert converter._token_id(metadata, "<unk>") == 0
    assert converter._token_id(metadata, "<|endoftext|>") == 2
    assert converter._token_id(metadata, "<missing>") == -1


def test_shared_decoder_declares_cohere_variants_without_family_dispatch() -> None:
    circuit = json.loads(
        (V8 / "circuits" / "audio_transformer_decoder.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (V8 / "schemas" / "numerical_required_contracts.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(
        {"required_numerical_contracts": circuit["required_numerical_contracts"]}
    )

    decoder = circuit["block_types"]["decoder"]
    header = [row for row in decoder["header"] if isinstance(row, dict)]
    body = [row for row in decoder["body"]["ops"] if isinstance(row, dict)]
    footer = [row for row in decoder["footer"] if isinstance(row, dict)]
    assert header[0]["when"] == {
        "config_key": "decoder_embedding_layernorm",
        "equals": True,
    }
    assert {row["op"] for row in body} == {"gelu", "audio_tdt_relu"}
    assert footer[0]["op"] == "bias_add"
    assert footer[0]["when"] == {
        "config_key": "decoder_output_bias",
        "equals": True,
    }
    assert "cohere" not in json.dumps(circuit).lower()


def test_decoder_artifact_selects_shared_decoder_without_mutating_family_graph() -> None:
    assert converter._component_circuit_name("audio_decoder") == (
        "audio_transformer_decoder.json"
    )
    assert converter._component_circuit_name("audio_encoder") == (
        "cohere_transcribe.json"
    )
    circuit = json.loads(
        (V8 / "circuits" / "cohere_transcribe.json").read_text(encoding="utf-8")
    )
    assert "resolved_components" not in circuit
    assert circuit["block_types"]["decoder"]["sequence"] == [
        "prompt",
        "causal_self_attention",
        "cross_attention",
        "relu_ffn",
        "lm_head",
        "greedy_eos",
    ]


def test_decoder_prompt_is_artifact_metadata_not_host_policy() -> None:
    source = (V8 / "scripts" / "convert_cohere_transcribe_gguf_to_bump_v8.py").read_text(
        encoding="utf-8"
    )
    assert '"audio_decoder_prompt_tokens"' in source
    assert source.count('"<|en|>"') >= 2


def test_decoder_certifier_requires_explicit_valid_prompt_ids() -> None:
    assert certifier._parse_prompt_ids("1, 2,3").tolist() == [1, 2, 3]
    for value in ("", "-1", "one,2"):
        try:
            certifier._parse_prompt_ids(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid prompt IDs were accepted: {value!r}")


def test_decoder_certifier_distinguishes_generated_and_emitted_tokens(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"generated_token_ids": [7, 8]}), encoding="utf-8")
    assert certifier._expected_tokens(summary, 3) == [7, 8, 3]
    summary.write_text(
        json.dumps({"decode": {"emitted_token_ids": [7, 8, 4]}}),
        encoding="utf-8",
    )
    assert certifier._expected_tokens(summary, 3) == [7, 8, 4]


def test_decoder_certifier_rejects_malformed_encoder_fixture(tmp_path: Path) -> None:
    import numpy as np

    fixture = tmp_path / "encoder.npz"
    np.savez(fixture, **{certifier.ENCODER_CHECKPOINT: np.ones((2, 3), np.float16)})
    try:
        certifier._encoder_fixture(fixture)
    except ValueError as error:
        assert "rank-2 float32" in str(error)
    else:
        raise AssertionError("FP16 encoder fixture was accepted")
