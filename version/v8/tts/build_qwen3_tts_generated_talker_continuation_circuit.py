#!/usr/bin/env python3
"""Connect a generated speech-code frame to a checked talker continuation.

This bounded circuit reuses the predictor's fifteen distinct codebook
embeddings, adds the projected TTS pad row, and executes the next talker step
with KV state seeded by the generated prefill. It does not yet loop to EOS.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_talker_predictor_frame_circuit import (
        ROOT, HIDDEN, build_circuit as generated_frame,
    )
    from .build_qwen3_tts_talker_prefill_cache_circuit import build_circuit as cached_prefill
    from .build_qwen3_tts_talker_decode_circuit import build_circuit as talker_decode
else:
    from build_qwen3_tts_talker_predictor_frame_circuit import (
        ROOT, HIDDEN, build_circuit as generated_frame,
    )
    from build_qwen3_tts_talker_prefill_cache_circuit import build_circuit as cached_prefill
    from build_qwen3_tts_talker_decode_circuit import build_circuit as talker_decode

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_generated_talker_continuation_bounded.json"
VOCAB = 2048


def build_circuit(*, context_tokens: int = 7, talker_layers: int = 28,
                  continuation_steps: int = 14,
                  max_generated_frames: int = 4) -> dict:
    if (isinstance(max_generated_frames, bool)
            or not isinstance(max_generated_frames, int)
            or not 2 <= max_generated_frames <= 64):
        raise ValueError("max_generated_frames must be in 2..64")
    cache_capacity = context_tokens + 10 + max_generated_frames
    max_step_index = max_generated_frames - 2
    graph = generated_frame(context_tokens=context_tokens,
                            talker_layers=talker_layers,
                            continuation_steps=continuation_steps)
    graph["name"] = ("qwen3_tts_generated_talker_continuation_bounded" if
                     talker_layers == 28 else
                     f"qwen3_tts_generated_talker_continuation_{talker_layers}_layer_fixture")
    graph["native_entry"]["function"] = "ck_" + graph["name"]
    graph["contract"]["runtime_invariants"].update({
        "generated_talker_continuation_steps": 1,
        "continuation_positions": [context_tokens + 11, cache_capacity - 1],
        "codebook_feedback_rows": continuation_steps + 2,
        "oracle_fed_model_intermediates": False,
        "complete_text_to_wav": False,
    })
    shapes, bindings = graph["activation_buffers"], graph["activation_bindings"]
    shapes["talker_continuation_valid"] = {"shape": [1]}
    bindings["talker_continuation_valid"] = "talker_continuation_valid"
    for name in ("talker_step_index_input", "talker_step_raw_checked",
                 "talker_step_index_checked",
                 "talker_decode_position_checked"):
        shapes[name] = {"shape": [1]}
        bindings[name] = name
    graph["runtime_lengths"].update({
        "talker_step_raw": {
            "producer": "check_talker_step_raw", "result": "valid_extent",
            "capacity": max(continuation_steps, max_step_index), "allow_zero": True},
        "talker_step_index": {
            "producer": "check_talker_step_index", "result": "valid_extent",
            "capacity": max_step_index, "allow_zero": True},
        "talker_decode_position": {
            "producer": "check_talker_decode_position", "result": "valid_extent",
            "capacity": cache_capacity - 1, "allow_zero": False},
    })
    graph["block_types"][graph["sequence"][0]]["header"].insert(0, {
        "id": "clear_talker_continuation_valid",
        "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
        "returns_status": True,
        "params": {"call_constants": {
            "extent_affine_source": 0, "extent_affine_factor": 1,
            "extent_affine_offset": 0, "extent_affine_capacity": 1}},
        "graph_slots": {"inputs": {},
                        "outputs": {"valid_extent": "talker_continuation_valid"}},
    })
    graph["block_types"][graph["sequence"][0]]["header"][1:1] = [
        {"id": "check_talker_step_raw", "op": "runtime_extent_sum",
         "kernel": "runtime_extent_sum_i32", "returns_status": True,
         "produces_runtime_lengths": {"talker_step_raw": "valid_extent"},
         "graph_slots": {"inputs": {"values": "external:talker_step_index_input"},
                         "outputs": {"valid_extent": "talker_step_raw_checked"}}},
        {"id": "check_talker_step_index", "op": "runtime_extent_affine",
         "kernel": "runtime_extent_affine_i32", "returns_status": True,
         "consumes_runtime_lengths": ["talker_step_raw"],
         "produces_runtime_lengths": {"talker_step_index": "valid_extent"},
         "runtime_scalar_bindings": {"extent_affine_source": "talker_step_raw"},
         "params": {"call_constants": {
             "extent_affine_factor": 1, "extent_affine_offset": 0,
             "extent_affine_capacity": max_step_index}},
         "graph_slots": {"inputs": {},
                         "outputs": {"valid_extent": "talker_step_index_checked"}}},
        {"id": "check_talker_decode_position", "op": "runtime_extent_affine",
         "kernel": "runtime_extent_affine_i32", "returns_status": True,
         "consumes_runtime_lengths": ["talker_step_index"],
         "produces_runtime_lengths": {"talker_decode_position": "valid_extent"},
         "runtime_scalar_bindings": {"extent_affine_source": "talker_step_index"},
         "params": {"call_constants": {
             "extent_affine_factor": 1,
             "extent_affine_offset": context_tokens + 11,
             "extent_affine_capacity": cache_capacity - 1}},
         "graph_slots": {"inputs": {},
                         "outputs": {"valid_extent": "talker_decode_position_checked"}}},
    ]

    # Seed state immediately after the prefill. Its Q/K/V producers must remain
    # alive until this point, before predictor work can reuse their arena space.
    seed = cached_prefill(context_tokens=context_tokens,
                          cache_capacity=cache_capacity)["block_types"]["talker_cache_seed"]
    seed_ops = copy.deepcopy(seed["body"]["ops"][:talker_layers])
    for layer in range(talker_layers):
        for kind in ("k", "v"):
            name = f"talker_layer{layer}_cache_{kind}"
            shapes[name] = {"shape": [8, cache_capacity, 128]}
            bindings[name] = name
    graph["block_types"]["talker_cache_seed"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": seed_ops}, "footer": [],
    }
    graph["sequence"].insert(graph["sequence"].index("talker_predictor_bridge"),
                             "talker_cache_seed")

    # The last predictor ID was selected but not embedded, because predictor
    # execution ends there. Feedback requires the fifteenth distinct table.
    shapes["feedback_last_embedding"] = {"shape": [1, HIDDEN]}
    bindings["feedback_last_embedding"] = "feedback_last_embedding"
    ops = [{
        "id": "feedback_last_embedding_lookup",
        "op": "embedding_lookup_token_major_checked",
        "kernel": "audio_text_embedding_token_major_f32", "returns_status": True,
        "weight_refs": {"table": f"talker.code_predictor.model.codec_embedding.{continuation_steps}.weight"},
        "graph_slots": {"inputs": {"ids": f"pred_step{continuation_steps}_selected_id"},
                        "outputs": {"output": "feedback_last_embedding"}},
        "params": {"T": 1, "V": VOCAB, "C": HIDDEN, "call_constants": {
            "id_elements": 1, "table_elements": VOCAB * HIDDEN,
            "vocabulary": VOCAB, "channels": HIDDEN, "table_stride": HIDDEN,
            "output_elements": HIDDEN, "tokens": 1, "output_stride": HIDDEN}},
    }]
    embeddings = (["talker_first_code_embedding", "pred_feedback_embedding"] +
                  [f"pred_step{i}_embedding" for i in range(2, continuation_steps + 1)] +
                  ["feedback_last_embedding"])
    assert len(embeddings) == continuation_steps + 2
    previous = embeddings[0]
    for index, embedding in enumerate(embeddings[1:], 1):
        summed = f"feedback_code_sum_{index}"
        shapes[summed] = {"shape": [1, HIDDEN]}
        bindings[summed] = summed
        ops.append({
            "id": summed, "op": "add_rows_checked",
            "kernel": "add_rows_checked_f32", "returns_status": True,
            "graph_slots": {"inputs": {"left": previous, "right": embedding},
                            "outputs": {"output": summed}},
            "params": {"N": HIDDEN, "call_constants": {
                "left_elements": HIDDEN, "right_elements": HIDDEN,
                "output_elements": HIDDEN, "elements": HIDDEN}},
        })
        previous = summed
    for name, values in (("feedback_left_indices", [0]),
                         ("feedback_right_indices", [3])):
        shapes[name] = {"shape": [len(values)]}
        bindings[name] = name
    shapes["feedback_next_input"] = {"shape": [1, HIDDEN]}
    bindings["feedback_next_input"] = "feedback_next_input"
    ops.append({
        "id": "feedback_add_projected_pad", "op": "indexed_sum_rows_checked",
        "kernel": "indexed_sum_rows_checked_f32", "returns_status": True,
        "graph_slots": {"inputs": {
            "left": previous, "right": "projection_output",
            "left_indices": "external:feedback_left_indices",
            "right_indices": "external:feedback_right_indices"},
            "outputs": {"output": "feedback_next_input"}},
        "params": {"L": 1, "K": context_tokens + 6, "R": 1, "C": HIDDEN,
                   "call_constants": {
                       "left_elements": HIDDEN, "left_stride": HIDDEN,
                       "left_rows": 1,
                       "right_elements": (context_tokens + 6) * HIDDEN,
                       "right_stride": HIDDEN,
                       "right_rows": context_tokens + 6,
                       "left_index_elements": 1, "right_index_elements": 1,
                       "output_elements": HIDDEN, "output_stride": HIDDEN,
                       "rows": 1, "channels": HIDDEN}},
    })
    graph["sequence"].append("talker_code_feedback")
    graph["block_types"]["talker_code_feedback"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": ops}, "footer": [],
    }

    continuation = talker_decode(layers=talker_layers,
                                 cache_capacity=cache_capacity)
    caches = {f"talker_layer{layer}_cache_{kind}"
              for layer in range(talker_layers) for kind in ("k", "v")}

    def slot(value: str) -> str:
        external = value.startswith("external:")
        name = value.removeprefix("external:")
        if name == "block_input":
            return "feedback_next_input"
        if name in caches:
            return "external:" + name
        return ("external:" if external else "") + "cont_" + name

    for name, shape in continuation["activation_buffers"].items():
        if name in caches or name in ("block_input", "step_index_input",
                                      "step_index_checked", "position_checked"):
            continue
        mapped = slot(name)
        shapes[mapped] = copy.deepcopy(shape)
        bindings[mapped] = mapped
    for block_name in continuation["sequence"]:
        block = copy.deepcopy(continuation["block_types"][block_name])
        block["header"] = []  # checked position is bound to each consumer below
        for op in block.get("body", {}).get("ops", []):
            op["id"] = "cont_" + op["id"]
            for direction in ("inputs", "outputs"):
                op["graph_slots"][direction] = {
                    port: slot(name) for port, name in
                    op["graph_slots"][direction].items()}
            if op["op"] == "rope_qk_bounded":
                op["params"]["call_constants"].pop("pos_offset", None)
                op["consumes_runtime_lengths"] = ["talker_decode_position"]
                op["runtime_scalar_bindings"] = {
                    "pos_offset": "talker_decode_position"}
            if op["op"] == "attention_decode_append_checked":
                op["params"]["call_constants"].pop("position", None)
                op["consumes_runtime_lengths"] = ["talker_decode_position"]
                op["runtime_scalar_bindings"] = {
                    "position": "talker_decode_position"}
        mapped = "cont_" + block_name
        graph["sequence"].append(mapped)
        graph["block_types"][mapped] = block
    graph["sequence"].append("publish_talker_continuation")
    graph["block_types"]["publish_talker_continuation"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": [{
            "id": "publish_talker_continuation_valid",
            "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
            "returns_status": True,
            "params": {"call_constants": {
                "extent_affine_source": 0, "extent_affine_factor": 1,
                "extent_affine_offset": 1, "extent_affine_capacity": 1}},
            "graph_slots": {"inputs": {},
                            "outputs": {"valid_extent": "talker_continuation_valid"}},
        }]}, "footer": [],
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
            raise SystemExit("generated continuation circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
