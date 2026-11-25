"""Decision block geometry and optional pinned slot projection."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from version.v8.decision.packed_question_layout import Question, pack_questions
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.scripts.replay_bf16_projection_bundle_v8 import projection_graph


def test_window_keeps_branch_local_positions_and_rejects_bad_slots():
    questions = (Question("first", (5, 6, 7), (("a", 8), ("b", 9))),
                 Question("second", (0, 1), (("only", 8),)))
    layout = pack_questions((2, 3, 4), questions, vocab_size=10, sliding_window=3, capacity=8)
    assert layout.token_ids == (2, 3, 4, 5, 6, 7, 0, 1)
    assert layout.position_ids == (0, 1, 2, 3, 4, 5, 3, 4)
    assert layout.full_ranges[6] == (0, 3, 6, 8)
    assert layout.sliding_ranges[6:] == ((1, 3, 6, 7), (2, 3, 6, 8))
    assert layout.sliding_ranges[5] == (3, 3, 3, 6)
    invalid = (Question("wrong", (5,), (("a", 8), ("b", 8))),)
    with pytest.raises(ValueError, match="distinct"):
        pack_questions((2, 3, 4), invalid, vocab_size=10, sliding_window=3, capacity=8)


def test_pinned_bison_readout_rows_preserve_option_mapping(tmp_path):
    source = os.environ.get("CK_BLOCK_DECISION_BISON_SLOT_WEIGHTS")
    if not source:
        pytest.skip("CK_BLOCK_DECISION_BISON_SLOT_WEIGHTS is not configured")
    torch = pytest.importorskip("torch")
    weights = np.load(source, allow_pickle=False)
    assert weights.shape == (3, 5376) and weights.dtype == np.uint16
    assert hashlib.sha256(weights.tobytes()).hexdigest() == (
        "41511eda30b253ddade4dc0601d0a192ec79bf23201db6b597f293fcfa94793a"
    )
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/decision/block_packed_bison_request.json").read_text()
    )
    questions = tuple(
        Question(row["name"], tuple(row["tokens"]), tuple(map(tuple, row["options"])))
        for row in fixture["questions"]
    )
    layout = pack_questions(tuple(fixture["prefix"]), questions, vocab_size=262144,
                            sliding_window=1024, capacity=256)
    slot_ids = [236776, 236799, 236780]
    width = weights.shape[-1]
    hidden = torch.randn(
        len(layout.token_ids), width, generator=torch.Generator().manual_seed(185)
    ).bfloat16()
    readout = hidden[[row for _, row, _ in layout.answers]]
    invoke, arena, view, _ = compile_checked_weight_graph(
        tmp_path, projection_graph(2, width, 3),
        {"projection.weight": weights}, {"projection.weight": ([3, width], "BF16")},
        source_config={"num_layers": 1, "embed_dim": width, "num_heads": 1,
                       "num_kv_heads": 1, "head_dim": width, "intermediate_size": width,
                       "context_length": 2, "max_seq_len": 2,
                       "activation_buffer_dtypes": {"input": "bf16", "output": "bf16"}},
    )

    def run(rows):
        view("input", np.uint16)[:] = rows.view(torch.uint16).numpy()
        assert invoke(arena, len(arena)) == 0
        return view("output", np.uint16).copy()

    expected = torch.nn.functional.linear(readout, torch.from_numpy(weights).view(torch.bfloat16))
    output = run(readout)
    np.testing.assert_array_equal(output, expected.view(torch.uint16).numpy())
    scores = {
        name: {option: int(output[i, slot_ids.index(slot)]) for option, slot in options}
        for i, (name, _, options) in enumerate(layout.answers)
    }
    assert list(scores["refund"]) == ["yes", "no"]
    assert list(scores["team"]) == ["billing", "technical", "fraud"]
    assert scores["refund"]["yes"] != scores["refund"]["no"]
    wrong_row = run(hidden[[layout.answers[0][1] - 1, layout.answers[1][1]]])
    assert not np.array_equal(wrong_row[0, :2], output[0, :2])
    np.testing.assert_array_equal(wrong_row[1], output[1])


def test_bison_oracle_runs_both_masks_and_reaches_readout(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    from safetensors.torch import save_file
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

    from version.v8.decision.capture_bison_block_oracle import PREFIX, capture

    config = Gemma4TextConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=3, vocab_size=14,
        num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=2,
        head_dim=4, global_head_dim=4, hidden_size_per_layer_input=0,
        layer_types=["sliding_attention", "full_attention", "full_attention"], sliding_window=3,
        final_logit_softcapping=30.0, pad_token_id=0,
    )
    config_dict = config.to_dict()
    config_dict["per_layer_config"] = {
        "01": {"head_dim": 4, "num_key_value_heads": 2},
        "02": {"head_dim": 4, "num_key_value_heads": 2},
    }
    (tmp_path / "config.json").write_text(json.dumps({"text_config": config_dict}))
    rng = torch.Generator().manual_seed(13)
    weights = {name: torch.randn(14, 16, generator=rng).bfloat16()
               for name in (PREFIX + "embed_tokens.weight", "lm_head.weight")}
    weights[PREFIX + "norm.weight"] = torch.ones(16).bfloat16()
    for i in range(3):
        for key, tensor in Gemma4TextDecoderLayer(config, i).state_dict().items():
            weights[f"{PREFIX}layers.{i}.{key}"] = (
                torch.randn(tensor.shape, generator=rng).bfloat16() * 0.5
                if tensor.ndim == 2 else torch.ones_like(tensor).bfloat16()
            )
    save_file(weights, str(tmp_path / "tiny.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": dict.fromkeys(weights, "tiny.safetensors"),
    }))
    result = capture(tmp_path, {
        "prefix": [2, 3, 4], "renderer_revision": "test",
        "questions": [{"name": "first", "tokens": [5, 6], "options": [["yes", 7], ["no", 8]]},
                      {"name": "second", "tokens": [10, 11, 12], "options": [["a", 7], ["b", 8], ["c", 9]]}],
    }, threads=1)
    assert [list(q["option_scores"]) for q in result["questions"]] == [["yes", "no"], ["a", "b", "c"]]
    assert result["wrong_mask_changes_scores"] and result["wrong_row_changes_scores"]
