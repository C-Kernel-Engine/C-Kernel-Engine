#!/usr/bin/env python3
"""Connect the first generated talker continuation to a second codebook frame.

The first frame, its feedback, the next talker ID and the second predictor
frame are circuit edges. EOS loop control and waveform decoding remain separate.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_generated_talker_continuation_circuit import (
        ROOT, HIDDEN, build_circuit as first_continuation,
    )
    from .build_qwen3_tts_predictor_full_frame_circuit import (
        build_circuit as predictor_frame,
    )
else:
    from build_qwen3_tts_generated_talker_continuation_circuit import (
        ROOT, HIDDEN, build_circuit as first_continuation,
    )
    from build_qwen3_tts_predictor_full_frame_circuit import (
        build_circuit as predictor_frame,
    )

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_second_generated_frame_bounded.json"
TALKER_VOCAB = 3072
TALKER_EOS_ID = 2150
PREFIX = "second_"


def copy_ids(op_id: str, source: str, destination: str,
             source_elements: int, destination_elements: int,
             *, source_offset: int = 0, destination_offset: int = 0,
             runtime_destination_offset: str | None = None) -> dict:
    op = {
        "id": op_id, "op": "copy_i32_span_checked",
        "kernel": "copy_i32_span_checked", "returns_status": True,
        "graph_slots": {"inputs": {"source": source},
                        "outputs": {"destination": destination}},
        "params": {"S": source_elements, "D": destination_elements,
                   "call_constants": {
                       "source_elements": source_elements,
                       "source_offset": source_offset,
                       "destination_elements": destination_elements,
                       "count": 1}},
    }
    if runtime_destination_offset is None:
        op["params"]["call_constants"]["destination_offset"] = destination_offset
    else:
        op["consumes_runtime_lengths"] = [runtime_destination_offset]
        op["runtime_scalar_bindings"] = {
            "destination_offset": runtime_destination_offset}
    return op


def build_circuit(*, context_tokens: int = 7, talker_layers: int = 28,
                  continuation_steps: int = 14,
                  max_generated_frames: int = 64) -> dict:
    if (isinstance(max_generated_frames, bool)
            or not isinstance(max_generated_frames, int)
            or not 2 <= max_generated_frames <= 64):
        raise ValueError("max_generated_frames must be in 2..64")
    cache_capacity = context_tokens + 10 + max_generated_frames
    max_step_index = max_generated_frames - 2
    graph = first_continuation(context_tokens=context_tokens,
                               talker_layers=talker_layers,
                               continuation_steps=continuation_steps,
                               max_generated_frames=max_generated_frames)
    predictor = predictor_frame(continuation_steps=continuation_steps)
    full = talker_layers == 28 and continuation_steps == 14
    graph["name"] = ("qwen3_tts_second_generated_frame_bounded" if full else
                     f"qwen3_tts_second_frame_{talker_layers}_layer_"
                     f"{continuation_steps + 1}_code_fixture")
    graph["native_entry"]["function"] = "ck_" + graph["name"]
    graph["contract"]["runtime_invariants"].update({
        "generated_speech_code_frames": 2,
        "second_frame_requires_non_eos_talker_id": True,
        "second_frame_pinned_talker_position": context_tokens + 11,
        "later_continuation_positions_safety_only": [19, cache_capacity - 1],
        "generation_policy": "greedy_fixed_non_eos_prefix_with_repetition_penalty",
        "repetition_penalty": "105/100 sign-dependent on generated-code history",
        "minimum_new_frames": 2,
        "maximum_new_frames": max_generated_frames,
        "codec_eos_token_id": TALKER_EOS_ID,
        "oracle_fed_model_intermediates": False,
        "complete_text_to_wav": False,
    })
    shapes, bindings = graph["activation_buffers"], graph["activation_bindings"]
    for name, shape in (
        ("second_talker_adjusted_logits", [TALKER_VOCAB]),
        ("second_talker_first_code_id", [1]),
        ("second_talker_first_code_embedding", [1, HIDDEN]),
        ("second_bridge_left_indices", [2]),
        ("second_bridge_right_indices", [2]),
        ("second_pred_block_input", [2, HIDDEN]),
    ):
        shapes[name] = {"shape": shape}
        bindings[name] = name
    graph["sequence"].append("second_talker_predictor_bridge")
    graph["block_types"]["second_talker_predictor_bridge"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": [
            {"id": "second_talker_repetition_penalty",
             "op": "repetition_penalty_last_row_checked",
             "kernel": "repetition_penalty_last_row_checked_f32",
             "returns_status": True,
             "graph_slots": {"inputs": {"values": "cont_talker_logits",
                                        "history": "talker_first_code_id"},
                             "outputs": {"output": "second_talker_adjusted_logits"}},
             "params": {"R": 1, "C": TALKER_VOCAB, "H": 1,
                        "call_constants": {
                            "values_elements": TALKER_VOCAB, "rows": 1,
                            "channels": TALKER_VOCAB,
                            "row_stride": TALKER_VOCAB,
                            "history_elements": 1, "history_count": 1,
                            "penalty_numerator": 105,
                            "penalty_denominator": 100,
                            "output_elements": TALKER_VOCAB}}},
            {"id": "second_talker_select_first_code",
             "op": "select_next_id_checked",
             "kernel": "select_next_id_checked_f32",
             "returns_status": True,
             "graph_slots": {"inputs": {"values": "second_talker_adjusted_logits"},
                             "outputs": {"index": "second_talker_first_code_id"}},
             "params": {"R": 1, "C": TALKER_VOCAB, "call_constants": {
                 "values_elements": TALKER_VOCAB, "rows": 1,
                 "channels": TALKER_VOCAB, "row_stride": TALKER_VOCAB,
                 "emitted_frames": 1, "min_frames": 2,
                 "max_frames": max_generated_frames, "stop_id": TALKER_EOS_ID,
                 "suppressed_begin": TALKER_VOCAB - 1024,
                 "suppressed_end": TALKER_VOCAB,
                 "selected_elements": 1}}},
            {"id": "second_talker_first_code_embedding_lookup",
             "op": "embedding_lookup_token_major_checked",
             "kernel": "audio_text_embedding_token_major_f32",
             "returns_status": True,
             "weight_refs": {"table": "talker.model.codec_embedding.weight"},
             "graph_slots": {"inputs": {"ids": "second_talker_first_code_id"},
                             "outputs": {"output": "second_talker_first_code_embedding"}},
             "params": {"T": 1, "V": TALKER_VOCAB, "C": HIDDEN,
                        "call_constants": {
                            "id_elements": 1, "table_elements": TALKER_VOCAB * HIDDEN,
                            "vocabulary": TALKER_VOCAB, "channels": HIDDEN,
                            "table_stride": HIDDEN, "output_elements": HIDDEN,
                            "tokens": 1, "output_stride": HIDDEN}}},
            {"id": "second_predictor_input_rows",
             "op": "indexed_sum_rows_checked",
             "kernel": "indexed_sum_rows_checked_f32", "returns_status": True,
             "graph_slots": {"inputs": {
                 "left": "cont_talker_final_norm",
                 "right": "second_talker_first_code_embedding",
                 "left_indices": "external:second_bridge_left_indices",
                 "right_indices": "external:second_bridge_right_indices"},
                 "outputs": {"output": "second_pred_block_input"}},
             "params": {"L": 1, "K": 1, "R": 2, "C": HIDDEN,
                        "call_constants": {
                            "left_elements": HIDDEN, "left_stride": HIDDEN,
                            "left_rows": 1, "right_elements": HIDDEN,
                            "right_stride": HIDDEN, "right_rows": 1,
                            "left_index_elements": 2, "right_index_elements": 2,
                            "output_elements": 2 * HIDDEN,
                            "output_stride": HIDDEN, "rows": 2,
                            "channels": HIDDEN}}},
        ]}, "footer": [],
    }

    def slot(value: str) -> str:
        external = value.startswith("external:")
        name = value.removeprefix("external:")
        if name == "block_input":
            return "second_pred_block_input"
        return ("external:" if external else "") + PREFIX + name

    for name, shape in predictor["activation_buffers"].items():
        if name == "block_input":
            continue
        mapped = PREFIX + name
        if mapped in shapes:
            raise ValueError(f"second predictor slot collides: {mapped}")
        shapes[mapped] = copy.deepcopy(shape)
        bindings[mapped] = mapped
    if predictor["runtime_constants"] != graph["runtime_constants"]:
        raise ValueError("second predictor runtime constants differ")
    for name, item in predictor["runtime_lengths"].items():
        mapped = copy.deepcopy(item)
        mapped["producer"] = PREFIX + item["producer"]
        graph["runtime_lengths"][PREFIX + name] = mapped
    for block_name in predictor["sequence"]:
        block = copy.deepcopy(predictor["block_types"][block_name])
        for op in (block.get("header", []) + block.get("body", {}).get("ops", []) +
                   block.get("footer", [])):
            op["id"] = PREFIX + op["id"]
            for direction in ("inputs", "outputs"):
                op["graph_slots"][direction] = {
                    port: slot(value)
                    for port, value in op["graph_slots"][direction].items()}
            if op.get("consumes_runtime_lengths"):
                op["consumes_runtime_lengths"] = [
                    PREFIX + name for name in op["consumes_runtime_lengths"]]
            if op.get("produces_runtime_lengths"):
                op["produces_runtime_lengths"] = {
                    PREFIX + name: result for name, result in
                    op["produces_runtime_lengths"].items()}
            if op.get("runtime_scalar_bindings"):
                op["runtime_scalar_bindings"] = {
                    argument: PREFIX + name for argument, name in
                    op["runtime_scalar_bindings"].items()}
        mapped_name = PREFIX + block_name
        graph["sequence"].append(mapped_name)
        graph["block_types"][mapped_name] = block
    # A failed first frame must not leave a prior second-frame extent valid.
    graph["block_types"][graph["sequence"][0]]["header"].insert(0, {
        "id": "clear_second_predictor_code_count",
        "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
        "returns_status": True,
        "params": {"call_constants": {
            "extent_affine_source": 0, "extent_affine_factor": 1,
            "extent_affine_offset": 0,
            "extent_affine_capacity": continuation_steps + 1}},
        "graph_slots": {"inputs": {},
                        "outputs": {"valid_extent":
                                    "second_predictor_valid_code_count"}},
    })
    # The first entry ends after the initial frame and cache seed. Every later
    # frame must calculate fresh embeddings, append KV at its checked position,
    # and select a fresh predictor frame from the previous generated IDs.
    first_ids = (["talker_first_code_id", "pred_predictor_codebook0_selected_id"] +
                 [f"pred_step{i}_selected_id" for i in range(1, continuation_steps + 1)])
    next_ids = (["second_talker_first_code_id",
                 "second_predictor_codebook0_selected_id"] +
                [f"second_step{i}_selected_id" for i in range(1, continuation_steps + 1)])
    state_ids = [f"rolling_code_state_{i}" for i in range(len(first_ids))]
    if len(first_ids) != continuation_steps + 2:
        raise ValueError("speech codebook state geometry differs")
    for name in state_ids:
        shapes[name] = {"shape": [1], "lifetime": "sequence"}
        bindings[name] = name
    shapes["talker_history_ids"] = {"shape": [max_generated_frames], "lifetime": "sequence"}
    bindings["talker_history_ids"] = "talker_history_ids"
    shapes["generation_state_valid"] = {"shape": [1], "lifetime": "sequence"}
    bindings["generation_state_valid"] = "generation_state_valid"
    shapes["expected_step_index"] = {"shape": [1], "lifetime": "sequence"}
    bindings["expected_step_index"] = "expected_step_index"
    for layer in range(talker_layers):
        for kind in ("k", "v"):
            shapes[f"talker_layer{layer}_cache_{kind}"]["lifetime"] = "sequence"
    shapes["step_control_verified"] = {"shape": [1]}
    bindings["step_control_verified"] = "step_control_verified"
    shapes["history_count_checked"] = {"shape": [1]}
    bindings["history_count_checked"] = "history_count_checked"
    shapes["talker_step_limit_scalar"] = {"shape": [1]}
    bindings["talker_step_limit_scalar"] = "talker_step_limit_scalar"
    graph["runtime_lengths"].update({
        "generation_state_ready": {
            "producer": "check_generation_state_ready", "result": "valid_extent",
            "capacity": 1, "allow_zero": False},
        "history_valid_count": {
            "producer": "check_history_count", "result": "valid_extent",
            "capacity": max_generated_frames - 1, "allow_zero": False},
        "talker_step_limit": {
            "producer": "set_talker_step_limit", "result": "valid_extent",
            "capacity": max_step_index, "allow_zero": True},
    })
    prefill_header = graph["block_types"][graph["sequence"][0]]["header"]
    step_ids = {"clear_talker_continuation_valid", "check_talker_step_raw",
                "check_talker_step_index", "check_talker_decode_position"}
    step_header = [op for op in prefill_header if op["id"] in step_ids]
    if len(step_header) != 4:
        raise ValueError("talker step checks differ from expected graph")
    prefill_header[:] = [op for op in prefill_header if op["id"] not in step_ids]
    prefill_clear_continuation = copy.deepcopy(step_header[0])
    prefill_clear_continuation["id"] = "clear_continuation_on_prefill"
    prefill_header.insert(0, prefill_clear_continuation)
    clear_state = {
        "id": "clear_generation_state_valid",
        "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
        "returns_status": True,
        "params": {"call_constants": {
            "extent_affine_source": 0, "extent_affine_factor": 1,
            "extent_affine_offset": 0, "extent_affine_capacity": 1}},
        "graph_slots": {"inputs": {},
                        "outputs": {"valid_extent": "generation_state_valid"}},
    }
    prefill_header.insert(0, clear_state)
    state_publish_ops = [copy_ids(
        f"seed_rolling_code_{index}", source, "external:" + destination,
        1, 1) for index, (source, destination) in
        enumerate(zip(first_ids, state_ids))]
    state_publish_ops.extend([
        copy_ids("seed_talker_history", first_ids[0],
                 "external:talker_history_ids", 1, max_generated_frames),
        {"id": "initialize_expected_step_index",
         "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
         "returns_status": True,
         "params": {"call_constants": {
             "extent_affine_source": 0, "extent_affine_factor": 1,
             "extent_affine_offset": 0, "extent_affine_capacity": max_step_index + 1}},
         "graph_slots": {"inputs": {},
                         "outputs": {"valid_extent": "external:expected_step_index"}}},
        {"id": "publish_generation_state_valid",
         "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
         "returns_status": True,
         "params": {"call_constants": {
             "extent_affine_source": 0, "extent_affine_factor": 1,
             "extent_affine_offset": 1, "extent_affine_capacity": 1}},
         "graph_slots": {"inputs": {},
                         "outputs": {"valid_extent": "generation_state_valid"}}},
    ])
    graph["sequence"].insert(graph["sequence"].index("talker_code_feedback"),
                             "publish_prefill_frame_state")
    graph["block_types"]["publish_prefill_frame_state"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": state_publish_ops}, "footer": [],
    }

    feedback = graph["block_types"]["talker_code_feedback"]
    feedback["header"] = [
        {"id": "check_generation_state_ready", "op": "runtime_extent_sum",
         "kernel": "runtime_extent_sum_i32", "returns_status": True,
         "produces_runtime_lengths": {"generation_state_ready": "valid_extent"},
         "graph_slots": {"inputs": {"values": "generation_state_valid"},
                         "outputs": {"valid_extent": "generation_state_ready_scalar"}},
         "params": {"T": 1}},
        {"id": "check_expected_step_index",
         "op": "equal_i32_scalar_checked",
         "kernel": "equal_i32_scalar_checked", "returns_status": True,
         "consumes_runtime_lengths": ["generation_state_ready"],
         "graph_slots": {"inputs": {
             "requested": "external:talker_step_index_input",
             "expected": "external:expected_step_index"},
             "outputs": {"checked": "step_control_verified"}},
         "params": {"call_constants": {
             "requested_elements": 1, "expected_elements": 1,
             "checked_elements": 1}}},
        copy.deepcopy(clear_state), *step_header,
        {"id": "check_history_count", "op": "runtime_extent_affine",
         "kernel": "runtime_extent_affine_i32", "returns_status": True,
         "consumes_runtime_lengths": ["talker_step_index"],
         "produces_runtime_lengths": {"history_valid_count": "valid_extent"},
         "runtime_scalar_bindings": {"extent_affine_source": "talker_step_index"},
         "params": {"call_constants": {
             "extent_affine_factor": 1, "extent_affine_offset": 1,
             "extent_affine_capacity": max_generated_frames - 1}},
         "graph_slots": {"inputs": {},
                         "outputs": {"valid_extent": "history_count_checked"}}},
    ]
    feedback["header"][2]["id"] = "invalidate_generation_state_for_step"
    feedback["header"][2]["consumes_runtime_lengths"] = ["generation_state_ready"]
    step_raw = next(op for op in feedback["header"]
                    if op["id"] == "check_talker_step_raw")
    step_raw["graph_slots"]["inputs"]["values"] = "step_control_verified"
    # The map's default max_duration belongs to the predictor and is 14 for
    # this geometry. Bind the talker check to its own declared frame limit.
    step_raw["consumes_runtime_lengths"] = ["talker_step_limit"]
    step_raw["runtime_scalar_bindings"] = {
        "max_duration": "talker_step_limit",
        "expanded_capacity": "talker_step_limit"}
    feedback["header"].insert(feedback["header"].index(step_raw), {
        "id": "set_talker_step_limit",
        "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
        "returns_status": True,
        "produces_runtime_lengths": {"talker_step_limit": "valid_extent"},
        "params": {"call_constants": {
            "extent_affine_source": 0, "extent_affine_factor": 1,
            "extent_affine_offset": max_step_index,
            "extent_affine_capacity": max_step_index}},
        "graph_slots": {"inputs": {},
                        "outputs": {"valid_extent": "talker_step_limit_scalar"}},
    })
    shapes["generation_state_ready_scalar"] = {"shape": [1]}
    bindings["generation_state_ready_scalar"] = "generation_state_ready_scalar"

    feedback_ops = feedback["body"]["ops"]
    if feedback_ops[0]["id"] != "feedback_last_embedding_lookup":
        raise ValueError("feedback embedding order differs")
    feedback_ops.pop(0)
    rolling_embeddings = []
    for index, selected in enumerate(state_ids):
        name = f"rolling_embedding_{index}"
        rolling_embeddings.append(name)
        shapes[name] = {"shape": [1, HIDDEN]}
        bindings[name] = name
        vocabulary = TALKER_VOCAB if index == 0 else 2048
        table = ("talker.model.codec_embedding.weight" if index == 0 else
                 f"talker.code_predictor.model.codec_embedding.{index - 1}.weight")
        feedback_ops.insert(index, {
            "id": f"rolling_embedding_lookup_{index}",
            "op": "embedding_lookup_token_major_checked",
            "kernel": "audio_text_embedding_token_major_f32",
            "returns_status": True,
            "weight_refs": {"table": table},
            "graph_slots": {"inputs": {"ids": "external:" + selected},
                            "outputs": {"output": name}},
            "params": {"T": 1, "V": vocabulary, "C": HIDDEN,
                       "call_constants": {
                           "id_elements": 1,
                           "table_elements": vocabulary * HIDDEN,
                           "vocabulary": vocabulary, "channels": HIDDEN,
                           "table_stride": HIDDEN,
                           "output_elements": HIDDEN,
                           "tokens": 1, "output_stride": HIDDEN}},
        })
    for index in range(1, len(first_ids)):
        op = next(item for item in feedback_ops
                  if item["id"] == f"feedback_code_sum_{index}")
        op["graph_slots"]["inputs"] = {
            "left": rolling_embeddings[0] if index == 1 else
                    f"feedback_code_sum_{index - 1}",
            "right": rolling_embeddings[index],
        }

    # Keep the original full-call ABI as a checked wrapper while also exposing
    # prefill and repeatable frame entries over the same planned arena.
    entry = graph["native_entry"]
    entry["segments"] = [
        {"function": entry["function"] + "_prefill",
         "last_op": "publish_generation_state_valid"},
        {"function": entry["function"] + "_frame",
         "last_op": "second_predictor_publish_code_count"},
    ]
    graph["contract"]["runtime_invariants"].update({
        "shared_arena_entries": [item["function"] for item in entry["segments"]],
        "second_entry_requires_first_entry_valid": True,
        "repeatable_generation_step": True,
        "supported_frame_positions": [18, cache_capacity - 1],
    })
    clear = copy.deepcopy(next(
        op for op in graph["block_types"][graph["sequence"][0]]["header"]
        if op["id"] == "clear_second_predictor_code_count"))
    clear["id"] = "clear_second_count_on_frame_entry"
    bridge = graph["block_types"]["second_talker_predictor_bridge"]
    feedback["header"].insert(0, clear)
    shapes["two_phase_ready_scalar"] = {"shape": [1]}
    bindings["two_phase_ready_scalar"] = "two_phase_ready_scalar"
    graph["runtime_lengths"]["two_phase_ready"] = {
        "producer": "check_two_phase_ready", "result": "valid_extent",
        "capacity": 1, "allow_zero": False,
    }
    bridge["header"].append({
        "id": "check_two_phase_ready", "op": "runtime_extent_sum",
        "kernel": "runtime_extent_sum_i32", "returns_status": True,
        "produces_runtime_lengths": {"two_phase_ready": "valid_extent"},
        "graph_slots": {"inputs": {"values": "talker_continuation_valid"},
                        "outputs": {"valid_extent": "two_phase_ready_scalar"}},
        "params": {"T": 1},
    })
    bridge["body"]["ops"][0]["consumes_runtime_lengths"] = ["two_phase_ready"]

    penalty = bridge["body"]["ops"][0]
    penalty["graph_slots"]["inputs"]["history"] = "external:talker_history_ids"
    penalty["params"]["H"] = max_generated_frames
    penalty["params"]["call_constants"]["history_elements"] = max_generated_frames
    penalty["params"]["call_constants"].pop("history_count")
    penalty["consumes_runtime_lengths"].append("history_valid_count")
    penalty["runtime_scalar_bindings"] = {"history_count": "history_valid_count"}
    selector = bridge["body"]["ops"][1]
    selector["params"]["call_constants"].pop("emitted_frames")
    selector["consumes_runtime_lengths"] = ["history_valid_count"]
    selector["runtime_scalar_bindings"] = {"emitted_frames": "history_valid_count"}

    publish = None
    for block_name in graph["sequence"]:
        block = graph["block_types"][block_name]
        for container in (block["header"], block["body"]["ops"], block["footer"]):
            for op in container:
                if op["id"] == "second_predictor_publish_code_count":
                    publish = op
                    container.remove(op)
                    break
    if publish is None:
        raise ValueError("second frame publisher missing")
    commit_ops = [copy_ids("append_talker_history", next_ids[0],
                           "external:talker_history_ids", 1, max_generated_frames,
                           runtime_destination_offset="history_valid_count")]
    for index, (source, destination) in enumerate(zip(next_ids, state_ids)):
        commit_ops.append(copy_ids(f"commit_rolling_code_{index}", source,
                                   "external:" + destination, 1, 1))
    commit_ops.extend([
        {"id": "advance_expected_step_index",
         "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
         "returns_status": True,
         "consumes_runtime_lengths": ["talker_step_index"],
         "runtime_scalar_bindings": {
             "extent_affine_source": "talker_step_index"},
         "params": {"call_constants": {
             "extent_affine_factor": 1, "extent_affine_offset": 1,
             "extent_affine_capacity": max_generated_frames - 1}},
         "graph_slots": {"inputs": {},
                         "outputs": {"valid_extent":
                                     "external:expected_step_index"}}},
        {"id": "republish_generation_state_valid",
         "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
         "returns_status": True,
         "params": {"call_constants": {
             "extent_affine_source": 0, "extent_affine_factor": 1,
             "extent_affine_offset": 1, "extent_affine_capacity": 1}},
         "graph_slots": {"inputs": {},
                         "outputs": {"valid_extent": "generation_state_valid"}}},
        publish,
    ])
    graph["sequence"].append("commit_rolling_frame_state")
    graph["block_types"]["commit_rolling_frame_state"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": commit_ops}, "footer": [],
    }
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("second-frame circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
