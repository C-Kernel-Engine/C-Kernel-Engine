#!/usr/bin/env python3
"""Carry checked Qwen prompt rows through generated prefill and first continuation.

Prefill and frame are separate native segments. The frame segment rechecks the
request's context count before adding it to the native loop's checked step.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_runtime_prompt_cache_circuit import (
        ROOT, build_circuit as runtime_prefill,
    )
    from .build_qwen3_tts_second_generated_frame_circuit import (
        build_circuit as generated_frames,
    )
else:
    from build_qwen3_tts_runtime_prompt_cache_circuit import (
        ROOT, build_circuit as runtime_prefill,
    )
    from build_qwen3_tts_second_generated_frame_circuit import (
        build_circuit as generated_frames,
    )


OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_runtime_second_frame_bounded.json"


def _copy_runtime_bindings(target: list[dict], source: list[dict]) -> None:
    reference = {op["id"]: op for op in source}
    for op in target:
        matched = reference.get(op["id"])
        if matched is None:
            raise ValueError("runtime prefill operation is missing: " + op["id"])
        for key in ("consumes_runtime_lengths", "runtime_scalar_bindings"):
            if key in matched:
                op[key] = copy.deepcopy(matched[key])
        for argument in matched.get("runtime_scalar_bindings", {}):
            if argument not in op["params"]["call_constants"]:
                raise ValueError("runtime scalar was not a call constant: " + op["id"])
            del op["params"]["call_constants"][argument]


def build_circuit(*, context_capacity: int = 7,
                  talker_layers: int = 28,
                  continuation_steps: int = 14,
                  max_generated_frames: int = 64) -> dict:
    graph = generated_frames(context_tokens=context_capacity,
                             talker_layers=talker_layers,
                             continuation_steps=continuation_steps,
                             max_generated_frames=max_generated_frames)
    cache_capacity = context_capacity + 10 + max_generated_frames
    dynamic = runtime_prefill(context_capacity=context_capacity,
                              cache_capacity=cache_capacity,
                              layers=talker_layers)
    graph["name"] = ("qwen3_tts_runtime_second_frame_bounded" if
                     talker_layers == 28 and continuation_steps == 14 else
                     f"qwen3_tts_runtime_second_frame_{talker_layers}_layer_fixture")
    entry = graph["native_entry"]
    entry["function"] = "ck_" + graph["name"]
    for index, segment in enumerate(entry["segments"]):
        segment["function"] = entry["function"] + ("_prefill" if index == 0 else "_frame")
    graph["contract"]["runtime_invariants"].update({
        "context_capacity": context_capacity,
        "supported_context_tokens": f"1..{context_capacity}",
        "prefill_rows": "context_tokens+11",
        "continuation_position": "context_tokens+11+step_index",
        "complete_text_to_wav": False,
    })
    shapes, bindings = graph["activation_buffers"], graph["activation_bindings"]
    for name in dynamic["runtime_lengths"] | {"request_context_tokens": None}:
        shapes[name] = copy.deepcopy(dynamic["activation_buffers"][name])
        bindings[name] = name
    shapes["request_context_tokens"]["lifetime"] = "sequence"
    graph["runtime_lengths"].update(copy.deepcopy(dynamic["runtime_lengths"]))
    for name in ("max_duration", "expanded_capacity"):
        graph["runtime_constants"][name] = max(
            graph["runtime_constants"][name], dynamic["runtime_constants"][name])
    shapes["published_prompt_rows"] = {"shape": [1], "lifetime": "sequence"}
    bindings["published_prompt_rows"] = "published_prompt_rows"

    projection = graph["block_types"]["text_projection"]
    projection["header"].extend(copy.deepcopy(
        dynamic["block_types"]["text_projection"]["header"]))
    for block_name in ("text_projection", "first_block", "talker_head"):
        _copy_runtime_bindings(
            graph["block_types"][block_name]["body"]["ops"],
            dynamic["block_types"][block_name]["body"]["ops"])
    graph["block_types"]["talker_cache_seed"] = copy.deepcopy(
        dynamic["block_types"]["talker_cache_seed"])
    for layer in range(1, talker_layers):
        name = f"talker_layer_{layer}"
        _copy_runtime_bindings(
            graph["block_types"][name]["body"]["ops"],
            dynamic["block_types"][name]["body"]["ops"])

    bridge = graph["block_types"]["talker_predictor_bridge"]["body"]["ops"]
    for op in bridge:
        if op["id"] == "talker_select_first_code":
            op["consumes_runtime_lengths"] = ["valid_prompt_rows"]
            op["runtime_scalar_bindings"] = {"rows": "valid_prompt_rows"}
            del op["params"]["call_constants"]["rows"]
        elif op["id"] == "talker_predictor_input_rows":
            op["consumes_runtime_lengths"] = ["valid_prompt_rows"]
            op["runtime_scalar_bindings"] = {"left_rows": "valid_prompt_rows"}
            del op["params"]["call_constants"]["left_rows"]

    prefill_publish = graph["block_types"]["publish_prefill_frame_state"]["body"]["ops"]
    prefill_publish.insert(-1, {
        "id": "publish_prefill_prompt_rows", "op": "runtime_extent_affine",
        "kernel": "runtime_extent_affine_i32", "returns_status": True,
        "consumes_runtime_lengths": ["valid_prompt_rows"],
        "runtime_scalar_bindings": {"extent_affine_source": "valid_prompt_rows"},
        "params": {"call_constants": {
            "extent_affine_factor": 1, "extent_affine_offset": 0,
            "extent_affine_capacity": context_capacity + 11}},
        "graph_slots": {"inputs": {},
                        "outputs": {"valid_extent": "external:published_prompt_rows"}},
    })

    for name in ("frame_context_tokens", "frame_prompt_rows",
                 "frame_prompt_rows_value", "prompt_rows_verified"):
        shapes[name] = {"shape": [1]}
        bindings[name] = name
    graph["runtime_lengths"].update({
        "frame_context_tokens": {
            "producer": "check_frame_context_tokens", "result": "valid_extent",
            "capacity": context_capacity, "allow_zero": False},
        "frame_prompt_rows": {
            "producer": "check_frame_prompt_rows", "result": "valid_extent",
            "capacity": context_capacity + 11, "allow_zero": False},
    })
    header = graph["block_types"]["talker_code_feedback"]["header"]
    position = next(op for op in header
                    if op["id"] == "check_talker_decode_position")
    position["consumes_runtime_lengths"] = ["talker_step_index", "frame_prompt_rows"]
    position["runtime_scalar_bindings"] = {
        "extent_affine_source": "talker_step_index",
        "extent_affine_offset": "frame_prompt_rows"}
    del position["params"]["call_constants"]["extent_affine_offset"]
    at = header.index(position)
    header[at:at] = [
        {"id": "check_frame_context_tokens", "op": "runtime_extent_sum",
         "kernel": "runtime_extent_sum_i32", "returns_status": True,
         "produces_runtime_lengths": {"frame_context_tokens": "valid_extent"},
         "graph_slots": {"inputs": {"values": "external:request_context_tokens"},
                         "outputs": {"valid_extent": "frame_context_tokens"}}},
        {"id": "check_frame_prompt_rows", "op": "runtime_extent_affine",
         "kernel": "runtime_extent_affine_i32", "returns_status": True,
         "consumes_runtime_lengths": ["frame_context_tokens"],
         "produces_runtime_lengths": {"frame_prompt_rows": "valid_extent"},
         "runtime_scalar_bindings": {"extent_affine_source": "frame_context_tokens"},
         "params": {"call_constants": {
             "extent_affine_factor": 1, "extent_affine_offset": 11,
             "extent_affine_capacity": context_capacity + 11}},
         "graph_slots": {"inputs": {},
                         "outputs": {"valid_extent": "frame_prompt_rows"}}},
        {"id": "materialize_frame_prompt_rows", "op": "runtime_extent_affine",
         "kernel": "runtime_extent_affine_i32", "returns_status": True,
         "consumes_runtime_lengths": ["frame_prompt_rows"],
         "runtime_scalar_bindings": {"extent_affine_source": "frame_prompt_rows"},
         "params": {"call_constants": {
             "extent_affine_factor": 1, "extent_affine_offset": 0,
             "extent_affine_capacity": context_capacity + 11}},
         "graph_slots": {"inputs": {},
                         "outputs": {"valid_extent": "frame_prompt_rows_value"}}},
        {"id": "verify_prefill_frame_prompt_rows",
         "op": "equal_i32_scalar_checked", "kernel": "equal_i32_scalar_checked",
         "returns_status": True,
         "graph_slots": {"inputs": {
             "requested": "frame_prompt_rows_value",
             "expected": "external:published_prompt_rows"},
             "outputs": {"checked": "prompt_rows_verified"}},
         "params": {"call_constants": {
             "requested_elements": 1, "expected_elements": 1,
             "checked_elements": 1}}},
    ]
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("runtime continuation circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
