"""Tests for the full-vocabulary scorer used by DaisyRose acceptance."""

import json

import numpy as np
import pytest

from server.daisyrose_bf16_acceptance import compare_logits, publisher_logits


def test_comparison_checks_entire_vocabulary_at_each_step():
    oracle = np.array([[2, -2, 0], [0, 3, 1]], dtype=np.float32)
    native = oracle.copy()
    native[0, 2] += 0.125
    native[1, 2] += 0.375
    rows = compare_logits(native, oracle, max_abs_limit=0.25, rmse_limit=0.2)
    assert [(row["phase"], row["numeric_pass"]) for row in rows] == [
        ("prefill", True), ("teacher_forced_decode", False),
    ]
    assert [row["native_argmax"] for row in rows] == [0, 1]
    assert [row["oracle_argmax"] for row in rows] == [0, 1]


def test_comparison_rejects_invalid_arrays():
    with pytest.raises(ValueError, match="geometry"):
        compare_logits(np.ones((2, 2)), np.ones((2, 3)), max_abs_limit=1, rmse_limit=1)
    with pytest.raises(ValueError, match="nonfinite"):
        compare_logits(np.array([[np.inf]]), np.zeros((1, 1)), max_abs_limit=1, rmse_limit=1)


def test_oracle_records_and_executes_the_selected_attention(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, layer_types=["full_attention"],
    )
    (tmp_path / "config.json").write_text(json.dumps({"text_config": config.to_dict()}))
    rng = torch.Generator().manual_seed(418)
    prefix = "model.language_model."
    layer = Qwen3_5DecoderLayer(config, 0)
    weights = {
        f"{prefix}layers.0.{name}": (torch.randn(value.shape, generator=rng) / 4).bfloat16()
        for name, value in layer.state_dict().items()
    }
    for name in (prefix + "embed_tokens.weight", "lm_head.weight"):
        weights[name] = (torch.randn(64, 32, generator=rng) / 4).bfloat16()
    weights[prefix + "norm.weight"] = torch.zeros(32, dtype=torch.bfloat16)
    save_file(weights, str(tmp_path / "small.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "small.safetensors")})
    )
    cases = [{"prompt_ids": [2, 5, 8, 11, 13], "answer_ids": [21]}]
    math, math_identity = publisher_logits(tmp_path, cases, 1)
    eager, eager_identity = publisher_logits(tmp_path, cases, 1, attention="eager")
    assert math_identity["attention"] == "math-sdpa"
    assert eager_identity["attention"] == "eager"
    assert math[0].shape == eager[0].shape == (2, 64)
    assert np.isfinite(math[0]).all() and np.isfinite(eager[0]).all()
    assert not np.array_equal(math[0], eager[0])
