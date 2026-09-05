"""Fail-closed terminal selection and generated position bookkeeping."""
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
from build_ir_v8 import _contract_selector_matches
from plan_terminal_rows_v8 import plan_terminal_prefill_rows
from codegen_prefill_v8 import emit_prefill_function, emit_prefill_from_embedded_function


def fixture():
    return {
        "mode": "prefill",
        "config": {"logits_layout": "last", "prefill_chunk_length": 8, "context_length": 64,
                   "contract": {"prefill_terminal_rows": {
                       "version": 1, "selector": {"config_not_equals": {"storage": "bf16"}},
                       "start_before": "projection", "suffix": [["projection"], ["logits"]],
                       "live_inputs": [{"op": "projection", "argument": "A", "width_args": ["K"]}],
                   }}},
        "memory": {"activations": {"buffers": [
            {"name": "hidden", "dtype": "f32", "define": "A_HIDDEN", "size": 128},
        ]}},
        "operations": [
            {"op": "projection", "function": "project", "layer": 1, "args": [
                {"name": "A", "source": "activation:a", "buffer_ref": "hidden",
                 "expr": "(const float*)(model->bump + A_HIDDEN)"},
                {"name": "K", "source": "dim:K", "expr": "4"},
                {"name": "M", "source": "dim:M", "expr": "8"},
                {"name": "C", "source": "output:c", "buffer_ref": "result", "expr": "result"},
            ]},
            {"op": "logits", "function": "head", "layer": -1, "args": [
                {"name": "A", "source": "activation:a", "buffer_ref": "result", "expr": "result"},
            ]},
        ],
    }


def plan(ir):
    plan_terminal_prefill_rows(ir, _contract_selector_matches)


def test_selects_last_layer_and_is_repeatable():
    ir = fixture()
    earlier = copy.deepcopy(ir["operations"][0])
    earlier["layer"] = 0
    ir["operations"].insert(0, earlier)
    plan(ir)
    assert "prefill_row_selection" not in earlier
    expected = copy.deepcopy(ir)
    plan(ir)
    assert ir == expected
    assert ir["operations"][1]["prefill_row_selection"]["copies"] == [
        {"buffer": "hidden", "define": "A_HIDDEN", "row_elements": 4}]


@pytest.mark.parametrize("change", ["full", "decode", "bf16", "no_contract"])
def test_unselected_paths_remove_stale_plan(change):
    ir = fixture()
    plan(ir)
    if change == "full":
        ir["config"]["logits_layout"] = "full"
    elif change == "decode":
        ir["mode"] = "decode"
    elif change == "bf16":
        ir["config"]["storage"] = "bf16"
    else:
        ir["config"]["contract"] = {}
    plan(ir)
    assert all("prefill_row_selection" not in op for op in ir["operations"])


@pytest.mark.parametrize("change", ["bounds", "dtype", "base", "dimension", "suffix", "dependency", "duplicate"])
def test_rejects_changed_contract_or_live_input(change):
    ir = fixture()
    buf = ir["memory"]["activations"]["buffers"][0]
    args = ir["operations"][0]["args"]
    if change == "bounds":
        buf["size"] -= 1
    elif change == "dtype":
        buf["dtype"] = "bf16"
    elif change == "base":
        args[0]["expr"] += " + 1"
    elif change == "dimension":
        args[1]["expr"] = "dynamic_width"
    elif change == "suffix":
        ir["operations"].append({"op": "kv_store", "layer": 1, "args": []})
    elif change == "dependency":
        ir["operations"][1]["args"][0]["buffer_ref"] = "undeclared_skip"
    else:
        inputs = ir["config"]["contract"]["prefill_terminal_rows"]["live_inputs"]
        inputs.append(copy.deepcopy(inputs[0]))
    with pytest.raises(ValueError, match="HARD TERMINAL ROW CONTRACT FAULT"):
        plan(ir)


@pytest.mark.parametrize("emitter", [emit_prefill_function, emit_prefill_from_embedded_function])
def test_codegen_preserves_consumed_count(emitter):
    ir = fixture()
    plan(ir)
    code = emitter(ir["operations"], config=ir["config"])
    assert "const int prefill_original_num_tokens = num_tokens;" in code
    assert "(size_t)(num_tokens - 1) * 4" in code
    assert code.index("num_tokens = 1;") < code.index("num_tokens = prefill_original_num_tokens;")
    assert code.index("num_tokens = prefill_original_num_tokens;") < code.index("model->pos = prefill_start_pos + num_tokens;")
    ir["config"]["logits_layout"] = "full"
    plan(ir)
    code = emitter(ir["operations"], config=ir["config"])
    assert "prefill_original_num_tokens" not in code
