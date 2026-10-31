#!/usr/bin/env python3
"""Author an opt-in Whisper decoder trace circuit from a pinned manifest.

This build-time transformation declares observation edges and buffer geometry.
The resulting neural execution still runs through ordinary v8 generated C.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def trace_manifest(source: dict, generation: dict, *, emit_observer: bool = True) -> dict:
    manifest = copy.deepcopy(source)
    config = manifest.get("config")
    template = manifest.get("template")
    if not isinstance(config, dict) or not isinstance(template, dict):
        raise ValueError("decoder manifest requires config and circuit template")
    heads = int(config.get("num_heads", 0))
    layers = int(config.get("num_layers", 0))
    frames = int(config.get("encoder_memory_length", 0))
    dim = int(config.get("head_dim", 0))
    if min(heads, layers, frames, dim) <= 0:
        raise ValueError("decoder manifest lacks positive attention dimensions")
    raw_heads = generation.get("alignment_heads")
    if not isinstance(raw_heads, list) or not raw_heads:
        raise ValueError("pinned generation config has no alignment heads")
    selected = []
    for pair in raw_heads:
        if (not isinstance(pair, list) or len(pair) != 2 or
                any(not isinstance(x, int) or isinstance(x, bool) for x in pair)):
            raise ValueError("alignment head must be an integer [layer, head] pair")
        layer, head = pair
        if not (0 <= layer < layers and 0 <= head < heads):
            raise ValueError("alignment head exceeds decoder geometry")
        selected.append([layer, head])
    if len({tuple(pair) for pair in selected}) != len(selected):
        raise ValueError("alignment head list contains duplicates")

    block = template.get("block_types", {}).get("decoder", {})
    body = block.get("body")
    if not isinstance(body, dict) or not isinstance(body.get("ops"), list):
        raise ValueError("decoder body operation list is missing")
    ops = body["ops"]
    matches = [index for index, item in enumerate(ops)
               if item == "cross_attn" or
               isinstance(item, dict) and item.get("op") == "cross_attn"]
    if len(matches) != 1:
        raise ValueError("decoder must contain exactly one cross-attention edge")
    if any(isinstance(item, dict) and item.get("id") == "attention_trace"
           for item in ops):
        raise ValueError("decoder already contains an attention trace")
    constants = {
        "query_elements": heads * dim,
        "key_elements": heads * frames * dim,
        "trace_elements": layers * heads * frames,
        "score_elements": heads * frames,
        "layers": layers, "heads": heads,
        "query_tokens": 1, "key_tokens": frames,
        "frame_capacity": frames, "head_dim": dim,
    }
    config["alignment_trace_call_constants"] = constants
    config["alignment_heads"] = selected
    template.setdefault("activation_buffers", {})["attention_trace"] = {
        "shape": [{"config": "num_layers"}, {"config": "num_heads"},
                  {"config": "encoder_memory_length"}],
        "lifetime": "sequence", "mutable": True,
    }
    template.setdefault("activation_bindings", {})["attention_trace"] = "attention_trace"
    observer = {
        "id": "attention_trace", "op": "attention_last_query_prob_trace",
        "kernel": "attention_last_query_prob_trace_f32", "returns_status": True,
        "params": {"H_from_config": "num_heads", "Tq": 1,
                   "D_from_config": "head_dim", "L_from_config": "num_layers",
                   "F_capacity_from_config": "encoder_memory_length",
                   "call_constants_from_config": "alignment_trace_call_constants"},
        "graph_slots": {
            "inputs": {"query": "cross_q_scratch", "key": "cross_k_cache"},
            "outputs": {"trace": "attention_trace"}},
    }
    if emit_observer:
        ops.insert(matches[0] + 1, observer)
    template.setdefault("contract", {})["alignment_trace"] = {
        "layout": "layer_head_frame_capacity", "selected_heads": selected,
        "observation": ("last_query_probability_row" if emit_observer
                        else "reserved_decode_trace_buffer"),
        "valid_frames": "encoder_memory_length",
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-output", type=Path,
                        help="Write a matching prefill layout without the decode observer")
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    generation = json.loads(args.generation_config.read_text())
    output = trace_manifest(source, generation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if args.prefill_output is not None:
        prefill = trace_manifest(source, generation, emit_observer=False)
        args.prefill_output.parent.mkdir(parents=True, exist_ok=True)
        args.prefill_output.write_text(json.dumps(prefill, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
