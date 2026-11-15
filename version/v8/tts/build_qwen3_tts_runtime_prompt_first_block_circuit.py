#!/usr/bin/env python3
"""Declare checked runtime prompt rows through Qwen text and talker block 0.

The circuit keeps capacity-sized storage and binds the request's valid rows to
ordinary providers. This is a connected prefill prefix, not full speech.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_prompt_first_block_circuit import (
        ROOT, build_circuit as fixed_prompt,
    )
else:
    from build_qwen3_tts_prompt_first_block_circuit import (
        ROOT, build_circuit as fixed_prompt,
    )


OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_runtime_prompt_first_block_bounded.json"


from version.v8.scripts.checked_circuit_authoring_v8 import (
    bind_runtime_scalars as _bind, runtime_extent as _extent,
)

def build_circuit(*, context_capacity: int = 7) -> dict:
    if isinstance(context_capacity, bool) or not isinstance(context_capacity, int) or not 1 <= context_capacity <= 117:
        raise ValueError("context_capacity must be an integer in 1..117")
    graph = fixed_prompt(context_capacity)
    text_capacity = context_capacity + 6
    prompt_capacity = context_capacity + 11
    graph["name"] = "qwen3_tts_runtime_prompt_first_block_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_runtime_prompt_first_block_bounded"
    graph["contract"]["runtime_invariants"].update({
        "context_capacity": context_capacity,
        "supported_context_tokens": f"1..{context_capacity}",
        "runtime_prompt_rows": "valid_context_tokens+11",
        "complete_synthesis": False,
    })
    graph["runtime_constants"] = {
        "value_elements": 1, "phoneme_count": 1,
        "max_duration": context_capacity,
        "expanded_capacity": context_capacity,
    }
    capacities = {
        "valid_context_tokens": context_capacity,
        "valid_text_rows": text_capacity,
        "valid_prompt_rows": prompt_capacity,
        "valid_q_norm_rows": prompt_capacity * 16,
        "valid_k_norm_rows": prompt_capacity * 8,
        "valid_hidden_elements": prompt_capacity * 1024,
        "valid_mlp_elements": prompt_capacity * 3072,
    }
    names = {name: name for name in capacities}
    names["request_context_tokens"] = "request_context_tokens"
    graph["activation_buffers"].update(
        {name: {"shape": [1]} for name in names})
    graph["activation_bindings"].update(names)
    graph["runtime_lengths"] = {
        name: {"producer": "check_" + name, "result": "valid_extent",
               "capacity": capacity, "allow_zero": False}
        for name, capacity in capacities.items()
    }
    header = graph["block_types"]["text_projection"]["header"]
    header.extend([
        _extent("check_valid_context_tokens", "runtime_extent_sum",
                "request_context_tokens", "valid_context_tokens",
                "valid_context_tokens", capacity=context_capacity),
        _extent("check_valid_text_rows", "runtime_extent_affine",
                "valid_context_tokens", "valid_text_rows", "valid_text_rows",
                offset=6, capacity=text_capacity),
        _extent("check_valid_prompt_rows", "runtime_extent_affine",
                "valid_context_tokens", "valid_prompt_rows", "valid_prompt_rows",
                offset=11, capacity=prompt_capacity),
        _extent("check_valid_q_norm_rows", "runtime_extent_scale",
                "valid_prompt_rows", "valid_q_norm_rows", "valid_q_norm_rows",
                factor=16, capacity=prompt_capacity * 16),
        _extent("check_valid_k_norm_rows", "runtime_extent_scale",
                "valid_prompt_rows", "valid_k_norm_rows", "valid_k_norm_rows",
                factor=8, capacity=prompt_capacity * 8),
        _extent("check_valid_hidden_elements", "runtime_extent_scale",
                "valid_prompt_rows", "valid_hidden_elements", "valid_hidden_elements",
                factor=1024, capacity=prompt_capacity * 1024),
        _extent("check_valid_mlp_elements", "runtime_extent_scale",
                "valid_prompt_rows", "valid_mlp_elements", "valid_mlp_elements",
                factor=3072, capacity=prompt_capacity * 3072),
    ])
    projection = graph["block_types"]["text_projection"]["body"]["ops"]
    projection_bindings = {
        "text_embedding": {"tokens": "valid_text_rows"},
        "embedding_transpose": {"transpose_columns": "valid_text_rows"},
        "projection_fc1": {"linear_rows": "valid_text_rows"},
        "projection_active": {"silu_rows": "valid_text_rows"},
        "projection_fc2": {"linear_rows": "valid_text_rows"},
        "prompt_assembly": {"left_rows": "valid_text_rows",
                            "rows": "valid_prompt_rows"},
    }
    block = graph["block_types"]["first_block"]["body"]["ops"]
    block_bindings = {
        **{name: {"rows": "valid_prompt_rows"} for name in (
            "input_norm", "post_attention_norm")},
        **{name: {"linear_rows": "valid_prompt_rows"} for name in (
            "q_proj", "k_proj", "v_proj", "attention_proj",
            "gate_proj", "up_proj", "down_proj")},
        "q_norm": {"rows": "valid_q_norm_rows"},
        "k_norm": {"rows": "valid_k_norm_rows"},
        **{name: {"tokens": "valid_prompt_rows"} for name in (
            "q_head", "k_head", "v_head", "rope_qk",
            "attention_head", "attention_token")},
        **{name: {"elements": "valid_hidden_elements"} for name in (
            "post_attention", "block_output")},
        "gated_mlp": {"elements": "valid_mlp_elements"},
    }
    for operations, expected in ((projection, projection_bindings),
                                 (block, block_bindings)):
        found = set()
        for op in operations:
            if op["id"] in expected:
                _bind(op, **expected[op["id"]])
                found.add(op["id"])
            elif op["id"] not in ("codec_embedding", "codec_embedding_transpose"):
                raise ValueError("unbound runtime row consumer: " + op["id"])
        if found != set(expected):
            raise ValueError("missing runtime row consumer: " + str(set(expected) - found))
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("runtime prompt circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
