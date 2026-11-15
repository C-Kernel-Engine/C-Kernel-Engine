#!/usr/bin/env python3
"""Author checked Qwen3-TTS text ID → two-layer talker projection graph.

This is the connected ingress to the talker, not a complete decoder block or
speech generator. The same operation contracts apply to other FP32 embeddings,
linear projections, and SiLU activations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_text_projection_bounded.json"
VOCAB = 151936
TEXT = 2048
HIDDEN = 1024


from version.v8.scripts.checked_circuit_authoring_v8 import linear

def build_circuit(*, tokens: int = 7, vocab: int = VOCAB,
                  text_width: int = TEXT, hidden_width: int = HIDDEN) -> dict:
    for name, value in (("tokens", tokens), ("vocab", vocab),
                        ("text_width", text_width), ("hidden_width", hidden_width)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if tokens > 1024:
        raise ValueError("bounded projection supports at most 1024 tokens")
    shapes = {
        "text_ids": [tokens], "embedding_channel": [text_width, tokens],
        "embedding_token": [tokens, text_width],
        "projection_fc1": [tokens, text_width],
        "projection_active": [tokens, text_width],
        "projection_output": [tokens, hidden_width],
    }
    ops = [
        {"id": "text_embedding", "op": "embedding_lookup_checked",
         "kernel": "audio_text_embedding_channel_major_f32",
         "returns_status": True,
         "weight_refs": {"table": "talker.model.text_embedding.weight"},
         "graph_slots": {"inputs": {"ids": "external:text_ids"},
                         "outputs": {"output": "embedding_channel"}},
         "params": {"T": tokens, "V": vocab, "C": text_width,
                    "T_capacity": tokens, "call_constants": {
                        "id_elements": tokens,
                        "table_elements": vocab * text_width,
                        "vocabulary": vocab, "channels": text_width,
                        "table_stride": text_width,
                        "output_elements": text_width * tokens,
                        "tokens": tokens, "output_stride": tokens}}},
        {"id": "embedding_transpose", "op": "transpose_strided_checked",
         "kernel": "transpose_strided_f32_checked", "returns_status": True,
         "graph_slots": {"inputs": {"input": "embedding_channel"},
                         "outputs": {"output": "embedding_token"}},
         "params": {"R": text_width, "C": tokens, "call_constants": {
             "transpose_input_elements": text_width * tokens,
             "transpose_input_stride": tokens,
             "transpose_output_elements": text_width * tokens,
             "transpose_output_stride": text_width,
             "transpose_rows": text_width,
             "transpose_columns": tokens}}},
        linear("projection_fc1", "embedding_token", "projection_fc1",
               text_width, text_width, tokens,
               "talker.text_projection.linear_fc1"),
        {"id": "projection_active", "op": "silu_strided_checked",
         "kernel": "silu_strided_f32_checked", "returns_status": True,
         "graph_slots": {"inputs": {"input": "projection_fc1"},
                         "outputs": {"output": "projection_active"}},
         "params": {"R": tokens, "C": text_width, "call_constants": {
             "silu_input_elements": tokens * text_width,
             "silu_input_stride": text_width,
             "silu_output_elements": tokens * text_width,
             "silu_output_stride": text_width,
             "silu_rows": tokens, "silu_columns": text_width}}},
        linear("projection_fc2", "projection_active", "projection_output",
               text_width, hidden_width, tokens,
               "talker.text_projection.linear_fc2"),
    ]
    return {
        "version": 3, "name": "qwen3_tts_text_projection_bounded",
        "family": "bounded_graph", "checked_native_entry": True,
        "contract": {"runtime_invariants": {
            "inference_only": True, "production_kernel_heap_allocation": False,
            "tokens": tokens, "complete_talker_block": False,
            "complete_synthesis": False}},
        "activation_buffers": {name: {"shape": shape}
                               for name, shape in shapes.items()},
        "activation_bindings": {name: name for name in shapes},
        "native_entry": {"function": "ck_qwen3_tts_text_projection_bounded",
                         "params": [{"c_type": "uint8_t *", "name": "arena"},
                                    {"c_type": "size_t", "name": "arena_bytes"}],
                         "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
        "sequence": ["text_projection"],
        "block_types": {"text_projection": {
            "sequence": ["header", "body", "footer"],
            "header": [], "body": {"type": "dense", "ops": ops},
            "footer": []}},
        "semantic_checkpoints": {
            "schema": "cke.semantic_checkpoint_contract", "schema_version": 1,
            "exports": {op["id"]: {
                "section": "body", "template_op_id": op["id"],
                "op": op["op"], "checkpoints": [{
                    "id": f"qwen3_tts.talker.{op['id']}",
                    "producer": op["id"],
                    "tensor": next(iter(op["graph_slots"]["outputs"].values())),
                    "logical_layout": ("channel_major" if op["id"] == "text_embedding"
                                       else "token_major"),
                    "axis_names": (["channel", "token"] if op["id"] == "text_embedding"
                                   else ["token", "channel"]),
                    "storage_dtype": "fp32"}]}
                for op in ops}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("text projection circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
