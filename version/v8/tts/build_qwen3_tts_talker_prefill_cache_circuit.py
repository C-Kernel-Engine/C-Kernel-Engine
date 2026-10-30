#!/usr/bin/env python3
"""Declare Qwen3-TTS prefill with checked caller-owned per-layer KV seeding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_talker_prefill_circuit import ROOT, build_circuit as prefill
else:
    from build_qwen3_tts_talker_prefill_circuit import ROOT, build_circuit as prefill

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_talker_prefill_cache_bounded.json"
LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128


def build_circuit(*, context_tokens: int = 7,
                  cache_capacity: int | None = None) -> dict:
    graph = prefill(context_tokens=context_tokens)
    prompt_rows = context_tokens + 11
    if cache_capacity is None:
        cache_capacity = prompt_rows + 3
    if (isinstance(cache_capacity, bool) or not isinstance(cache_capacity, int)
            or cache_capacity < prompt_rows + 1):
        raise ValueError("cache_capacity must exceed prompt rows")
    graph["name"] = "qwen3_tts_talker_prefill_cache_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_talker_prefill_cache_bounded"
    graph["contract"]["runtime_invariants"].update({
        "cache_capacity": cache_capacity,
        "cache_valid_rows_after_success": prompt_rows,
        "cache_requires_reset_after_failed_entry": True,
        "autonomous_code_generation": False,
    })
    cache_elements = KV_HEADS * cache_capacity * HEAD_DIM
    source_elements = KV_HEADS * prompt_rows * HEAD_DIM
    ops = []
    for layer in range(LAYERS):
        prefix = "" if layer == 0 else f"layer{layer}_"
        cache_k = f"talker_layer{layer}_cache_k"
        cache_v = f"talker_layer{layer}_cache_v"
        for name in (cache_k, cache_v):
            graph["activation_buffers"][name] = {
                "shape": [KV_HEADS, cache_capacity, HEAD_DIM]}
            graph["activation_bindings"][name] = name
        ops.append({
            "id": f"talker_layer{layer}_cache_seed",
            "op": "kv_cache_seed_checked",
            "kernel": "kv_cache_seed_checked_f32",
            "returns_status": True,
            "graph_slots": {
                "inputs": {"k": prefix + "k_head", "v": prefix + "v_head",
                           "cache_k": "external:" + cache_k,
                           "cache_v": "external:" + cache_v},
                "outputs": {"cache_k": "external:" + cache_k,
                            "cache_v": "external:" + cache_v},
            },
            "params": {"KV": KV_HEADS, "T": prompt_rows,
                       "S": cache_capacity, "D": HEAD_DIM,
                       "call_constants": {
                           "k_elements": source_elements,
                           "v_elements": source_elements,
                           "k_cache_elements": cache_elements,
                           "v_cache_elements": cache_elements,
                           "kv_heads": KV_HEADS, "tokens": prompt_rows,
                           "cache_capacity": cache_capacity,
                           "head_dim": HEAD_DIM}},
        })
    graph["sequence"].append("talker_cache_seed")
    graph["block_types"]["talker_cache_seed"] = {
        "sequence": ["header", "body", "footer"],
        "header": [], "body": {"type": "dense", "ops": ops}, "footer": [],
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
            raise SystemExit("cached prefill circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
