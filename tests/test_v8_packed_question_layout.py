"""Decision block geometry and optional pinned slot projection."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from version.v8.decision.build_packed_option_readout import option_readout_graph
from version.v8.decision.packed_question_layout import Question, pack_questions
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


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


def _build_readout(folder, hidden, weights):
    folder.mkdir(exist_ok=True)
    rows, width = hidden.shape
    labels = weights.shape[0]
    invoke, arena, view, _ = compile_checked_weight_graph(
        folder, option_readout_graph(rows, 2, width, labels),
        {"option.weight": weights}, {"option.weight": ([labels, width], "BF16")},
        source_config={"num_layers": 1, "embed_dim": width, "num_heads": 1,
                       "num_kv_heads": 1, "head_dim": width, "intermediate_size": width,
                       "context_length": rows, "max_seq_len": rows,
                       "activation_buffer_dtypes": {"hidden": "bf16", "selected": "bf16",
                                                    "answer_rows": "i32", "scores": "bf16"}},
    )
    view("hidden", np.uint16)[:] = hidden

    def run(indices):
        view("answer_rows", np.int32)[:] = indices
        return invoke(arena, len(arena))

    return view("scores", np.uint16), run


def test_generated_readout_rejects_out_of_range_rows_without_publishing(tmp_path):
    # Exact BF16 representations of [[1, 2], [2, 3], [3, 4]] and an identity projection.
    hidden = np.array([[0x3F80, 0x4000], [0x4000, 0x4040], [0x4040, 0x4080]], dtype=np.uint16)
    weights = np.array([[0x3F80, 0], [0, 0x3F80]], dtype=np.uint16)
    output, run = _build_readout(tmp_path, hidden, weights)
    assert run([2, 0]) == 0
    np.testing.assert_array_equal(output, hidden[[2, 0]])
    assert run([0, 3]) == -2
    assert run([-1, 2]) == -2
    np.testing.assert_array_equal(output, hidden[[2, 0]])
    assert run([1, 1]) == 0
    np.testing.assert_array_equal(output, hidden[[1, 1]])


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
    oracle_capture = os.environ.get("CK_BLOCK_DECISION_BISON_ORACLE")
    if oracle_capture:
        capture_dir = Path(oracle_capture)
        hidden_bits = np.load(capture_dir / "bison-readout-inputs.npy", allow_pickle=False)
        assert hidden_bits.dtype == np.uint16 and hidden_bits.shape == (len(layout.token_ids), width)
        hidden = torch.from_numpy(hidden_bits).view(torch.bfloat16)
    else:
        hidden = torch.randn(
            len(layout.token_ids), width, generator=torch.Generator().manual_seed(185)
        ).bfloat16()
    readout = hidden[[row for _, row, _ in layout.answers]]
    rows = np.array([row for _, row, _ in layout.answers], dtype=np.int32)
    output, run = _build_readout(tmp_path, hidden.view(torch.uint16).numpy(), weights)
    assert run(rows) == 0
    expected = torch.nn.functional.linear(readout, torch.from_numpy(weights).view(torch.bfloat16))
    np.testing.assert_array_equal(output, expected.view(torch.uint16).numpy())
    before = output.copy()
    scores = {
        name: {option: int(output[i, slot_ids.index(slot)]) for option, slot in options}
        for i, (name, _, options) in enumerate(layout.answers)
    }
    assert list(scores["refund"]) == ["yes", "no"]
    assert list(scores["team"]) == ["billing", "technical", "fraud"]
    assert scores["refund"]["yes"] != scores["refund"]["no"]
    if oracle_capture:
        report = json.loads((capture_dir / "bison-oracle.json").read_text())
        assert report["checkpoint"] == fixture["checkpoint"]
        assert report["renderer_revision"] == fixture["renderer_revision"]
        assert report["readout_inputs_sha256"] == hashlib.sha256(
            (capture_dir / "bison-readout-inputs.npy").read_bytes()
        ).hexdigest()
        assert report["wrong_mask_changes_scores"] and report["wrong_row_changes_scores"]
        for i, (_, _, options) in enumerate(layout.answers):
            values = output[i, [slot_ids.index(slot) for _, slot in options]]
            raw = torch.from_numpy(values).view(torch.bfloat16).float().tolist()
            assert raw == list(report["questions"][i]["raw_option_scores"].values())
    changed_rows = rows.copy()
    changed_rows[0] -= 1
    assert run(changed_rows) == 0
    assert not np.array_equal(output[0, :2], before[0, :2])
    np.testing.assert_array_equal(output[1], before[1])


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
    readout_inputs = tmp_path / "readout_inputs.npy"
    result = capture(tmp_path, {
        "prefix": [2, 3, 4], "renderer_revision": "test",
        "questions": [{"name": "first", "tokens": [5, 6], "options": [["yes", 7], ["no", 8]]},
                      {"name": "second", "tokens": [10, 11, 12], "options": [["a", 7], ["b", 8], ["c", 9]]}],
    }, threads=1, readout_inputs=readout_inputs)
    assert [list(q["option_scores"]) for q in result["questions"]] == [["yes", "no"], ["a", "b", "c"]]
    assert result["wrong_mask_changes_scores"] and result["wrong_row_changes_scores"]
    hidden = np.load(readout_inputs, allow_pickle=False)
    assert hidden.dtype == np.uint16 and hidden.shape == (8, 16)
    weight_rows = weights["lm_head.weight"][[7, 8, 9]].view(torch.uint16).numpy()
    output, run = _build_readout(tmp_path / "native", hidden, weight_rows)
    assert run([4, 7]) == 0
    raw = torch.from_numpy(output).view(torch.bfloat16).float().tolist()
    for scores, question in zip(raw, result["questions"], strict=True):
        expected = list(question["raw_option_scores"].values())
        assert scores[:len(expected)] == expected
