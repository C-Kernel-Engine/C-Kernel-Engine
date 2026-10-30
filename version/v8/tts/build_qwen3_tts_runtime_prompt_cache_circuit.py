#!/usr/bin/env python3
"""Seed one Qwen talker KV layer from checked runtime prompt rows.

The bounded first-block graph produces packed valid K/V rows. The existing
cache provider copies those rows into capacity-strided caller-owned state.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_runtime_prompt_first_block_circuit import (
        ROOT, build_circuit as runtime_prefix,
    )
    from .build_qwen3_tts_talker_prefill_circuit import build_circuit as full_prefill
else:
    from build_qwen3_tts_runtime_prompt_first_block_circuit import (
        ROOT, build_circuit as runtime_prefix,
    )
    from build_qwen3_tts_talker_prefill_circuit import build_circuit as full_prefill


OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_runtime_prompt_cache_bounded.json"
KV_HEADS = 8
HEAD_DIM = 128
LAYERS = 28


def build_circuit(*, context_capacity: int = 7,
                  cache_capacity: int | None = None,
                  layers: int = LAYERS) -> dict:
    if isinstance(layers, bool) or not isinstance(layers, int) or not 1 <= layers <= LAYERS:
        raise ValueError("layers must be an integer in 1..28")
    runtime = runtime_prefix(context_capacity=context_capacity)
    graph = full_prefill(context_tokens=context_capacity, layers=layers)
    prompt_capacity = context_capacity + 11
    if cache_capacity is None:
        cache_capacity = prompt_capacity + 3
    if (isinstance(cache_capacity, bool) or not isinstance(cache_capacity, int)
            or not prompt_capacity < cache_capacity <= 256):
        raise ValueError("cache_capacity must exceed prompt capacity and be at most 256")
    graph["name"] = ("qwen3_tts_runtime_prompt_cache_bounded" if layers == LAYERS
                     else f"qwen3_tts_runtime_prompt_cache_{layers}_layer_fixture")
    graph["native_entry"]["function"] = "ck_" + graph["name"]
    graph["runtime_constants"] = copy.deepcopy(runtime["runtime_constants"])
    graph["runtime_lengths"] = copy.deepcopy(runtime["runtime_lengths"])
    for name in runtime["runtime_lengths"] | {"request_context_tokens": None}:
        graph["activation_buffers"][name] = {"shape": [1]}
        graph["activation_bindings"][name] = name
    for block_name in ("text_projection", "first_block"):
        graph["block_types"][block_name] = copy.deepcopy(runtime["block_types"][block_name])
    first_ops = {
        op["id"]: op for op in runtime["block_types"]["first_block"]["body"]["ops"]
    }
    for layer in range(1, layers):
        for op in graph["block_types"][f"talker_layer_{layer}"]["body"]["ops"]:
            first = first_ops[op["id"].removeprefix(f"layer{layer}_")]
            for key in ("consumes_runtime_lengths", "runtime_scalar_bindings"):
                op[key] = copy.deepcopy(first[key])
            for argument in first["runtime_scalar_bindings"]:
                op["params"]["call_constants"].pop(argument)
    for op in graph["block_types"]["talker_head"]["body"]["ops"]:
        argument = "rows" if op["id"] == "talker_final_norm" else "linear_rows"
        op["consumes_runtime_lengths"] = ["valid_prompt_rows"]
        op["runtime_scalar_bindings"] = {argument: "valid_prompt_rows"}
        op["params"]["call_constants"].pop(argument)
    graph["contract"]["runtime_invariants"].update({
        "cache_capacity": cache_capacity,
        "talker_layers": layers,
        "cache_valid_rows_after_success": "valid_prompt_rows",
        "cache_requires_reset_after_failed_entry": True,
        "complete_synthesis": False,
    })
    cache_elements = KV_HEADS * cache_capacity * HEAD_DIM
    source_elements = KV_HEADS * prompt_capacity * HEAD_DIM
    ops = []
    for layer in range(layers):
        prefix = "" if layer == 0 else f"layer{layer}_"
        for kind in ("k", "v"):
            name = f"talker_layer{layer}_cache_{kind}"
            graph["activation_buffers"][name] = {
                "shape": [KV_HEADS, cache_capacity, HEAD_DIM]}
            graph["activation_bindings"][name] = name
        ops.append({
            "id": f"talker_layer{layer}_cache_seed",
            "op": "kv_cache_seed_checked",
            "kernel": "kv_cache_seed_checked_f32",
            "returns_status": True,
            "consumes_runtime_lengths": ["valid_prompt_rows"],
            "runtime_scalar_bindings": {"tokens": "valid_prompt_rows"},
            "graph_slots": {
                "inputs": {"k": prefix + "k_head", "v": prefix + "v_head",
                           "cache_k": f"external:talker_layer{layer}_cache_k",
                           "cache_v": f"external:talker_layer{layer}_cache_v"},
                "outputs": {"cache_k": f"external:talker_layer{layer}_cache_k",
                            "cache_v": f"external:talker_layer{layer}_cache_v"},
            },
            "params": {"KV": KV_HEADS, "T": prompt_capacity,
                       "S": cache_capacity, "D": HEAD_DIM,
                       "call_constants": {
                           "k_elements": source_elements,
                           "v_elements": source_elements,
                           "k_cache_elements": cache_elements,
                           "v_cache_elements": cache_elements,
                           "kv_heads": KV_HEADS,
                           "cache_capacity": cache_capacity,
                           "head_dim": HEAD_DIM}},
        })
    graph["sequence"].append("talker_cache_seed")
    graph["block_types"]["talker_cache_seed"] = {
        "sequence": ["header", "body", "footer"],
        "header": [],
        "body": {"type": "dense", "ops": ops},
        "footer": [],
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
            raise SystemExit("runtime prompt-cache circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
