"""Extend Clef's generated head ingress through its first evidence-routing layer.

The packed PyTorch MHA input projection is split into effective Q/K/V weights
and biases by the model importer. The split retains the source archive identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_clef_flash_head_ingress_circuit import build_circuit as build_ingress

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/clef_flash_evidence0_bounded.json"
PREFIX = "evidence_layers.0"


def split_packed_attention_projection(packed_weight, packed_bias):
    """Import PyTorch's packed Q/K/V rows as three declared projection weights.

    This is an asset transform, not runtime model arithmetic. Each slice is a
    byte-preserving copy after the caller chooses its declared storage dtype.
    """
    if (len(packed_weight.shape) != 2 or
            packed_weight.shape[0] != 3 * packed_weight.shape[1] or
            packed_bias.shape != (packed_weight.shape[0],)):
        raise ValueError("packed MHA projection must be [3W,W] with [3W] bias")
    width = packed_weight.shape[1]
    result = {}
    for index, name in enumerate(("q", "k", "v")):
        stem = f"{PREFIX}.attention.{name}_proj"
        start, end = index * width, (index + 1) * width
        result[f"{stem}.weight"] = packed_weight[start:end].copy()
        result[f"{stem}.bias"] = packed_bias[start:end].copy()
    return result


def build_circuit(*, tokens=6, questions=2, options=3, channels=4096,
                  width=1024, vocabulary=16, heads=None, feedforward=None):
    if heads is None:
        heads = 16 if width == 1024 else 1
    if feedforward is None:
        feedforward = 4 * width
    if heads < 1 or feedforward < 1 or width % heads:
        raise ValueError("positive feedforward and heads dividing width required")
    graph = build_ingress(tokens=tokens, questions=questions, options=options,
                          channels=channels, width=width, vocabulary=vocabulary)
    shapes = graph["activation_buffers"]
    bindings = graph["activation_bindings"]

    def buffer(name, rows, cols):
        shapes[name] = {"shape": [rows, cols]}
        bindings[name] = name

    for name, rows, cols in (
        ("evidence0_query_norm", options, width),
        ("evidence0_memory_norm", tokens, width),
        ("evidence0_q", options, width),
        ("evidence0_k", tokens, width),
        ("evidence0_v", tokens, width),
        ("evidence0_context", options, width),
        ("evidence0_attn_out", options, width),
        ("evidence0_attn_residual", options, width),
        ("evidence0_ff_norm", options, width),
        ("evidence0_ff1", options, feedforward),
        ("evidence0_gelu", options, feedforward),
        ("evidence0_ff2", options, width),
        ("evidence0_output", options, width),
    ):
        buffer(name, rows, cols)

    def norm(name, source, output, rows, stem):
        count = rows * width
        return {
            "id": name, "op": "layernorm",
            "kernel": "layernorm_rows_checked_param_epsilon_f32",
            "returns_status": True,
            "weight_refs": {"gamma": f"{stem}.weight", "beta": f"{stem}.bias"},
            "graph_slots": {"inputs": {"input": source},
                            "outputs": {"output": output}},
            "params": {"M": rows, "C": width, "Q": rows * (width + 2),
                       "epsilon": 1e-5, "call_constants": {
                "norm_input_elements": count, "norm_input_stride": width,
                "norm_gamma_elements": width, "norm_beta_elements": width,
                "norm_output_elements": count, "norm_output_stride": width,
                "norm_scratch_elements": rows * (width + 2),
                "norm_rows": rows, "norm_channels": width}},
        }

    def linear(name, source, output, rows, input_width, output_width, stem):
        return {
            "id": name, "op": "linear_rows_checked",
            "kernel": "linear_rows_checked_f32", "returns_status": True,
            "weight_refs": {"weight": f"{stem}.weight", "bias": f"{stem}.bias"},
            "graph_slots": {"inputs": {"input": source},
                            "outputs": {"output": output}},
            "params": {"M": rows, "K": input_width, "N": output_width,
                       "call_constants": {
                "linear_input_elements": rows * input_width,
                "linear_input_stride": input_width,
                "linear_weight_elements": output_width * input_width,
                "linear_weight_stride": input_width,
                "linear_bias_elements": output_width,
                "linear_output_elements": rows * output_width,
                "linear_output_stride": output_width,
                "linear_rows": rows,
                "linear_input_channels": input_width,
                "linear_output_channels": output_width}},
        }

    def add(name, left, right, output):
        count = options * width
        return {
            "id": name, "op": "add_rows_checked",
            "kernel": "add_rows_checked_f32", "returns_status": True,
            "graph_slots": {"inputs": {"left": left, "right": right},
                            "outputs": {"output": output}},
            "params": {"N": count, "call_constants": {
                "left_elements": count, "right_elements": count,
                "output_elements": count, "elements": count}},
        }

    header = [
        norm("evidence0_query_norm_op", "option_queries",
             "evidence0_query_norm", options, f"{PREFIX}.query_norm"),
        norm("evidence0_memory_norm_op", "memory",
             "evidence0_memory_norm", tokens, f"{PREFIX}.memory_norm"),
    ]
    body = [
        linear("evidence0_q_projection", "evidence0_query_norm",
               "evidence0_q", options, width, width,
               f"{PREFIX}.attention.q_proj"),
        linear("evidence0_k_projection", "evidence0_memory_norm",
               "evidence0_k", tokens, width, width,
               f"{PREFIX}.attention.k_proj"),
        linear("evidence0_v_projection", "evidence0_memory_norm",
               "evidence0_v", tokens, width, width,
               f"{PREFIX}.attention.v_proj"),
        {"id": "evidence0_cross_attention",
         "op": "attention_cross_token_major_checked",
         "kernel": "attention_cross_token_major_checked_f32",
         "returns_status": True,
         "graph_slots": {"inputs": {
             "query": "evidence0_q", "key": "evidence0_k",
             "value": "evidence0_v"},
             "outputs": {"output": "evidence0_context"}},
         "params": {"Q": options, "S": tokens, "H": heads,
                    "D": width // heads, "X": tokens + options * width,
                    "call_constants": {
             "query_elements": options * width,
             "key_elements": tokens * width,
             "value_elements": tokens * width,
             "output_elements": options * width,
             "scratch_elements": tokens + options * width,
             "query_tokens": options, "memory_tokens": tokens,
             "heads": heads, "head_dim": width // heads}}},
        linear("evidence0_output_projection", "evidence0_context",
               "evidence0_attn_out", options, width, width,
               f"{PREFIX}.attention.out_proj"),
        add("evidence0_attention_residual", "option_queries",
            "evidence0_attn_out", "evidence0_attn_residual"),
        norm("evidence0_feedforward_norm_op", "evidence0_attn_residual",
             "evidence0_ff_norm", options, f"{PREFIX}.feedforward_norm"),
        linear("evidence0_feedforward1", "evidence0_ff_norm",
               "evidence0_ff1", options, width, feedforward,
               f"{PREFIX}.feedforward.0"),
        {"id": "evidence0_gelu_op", "op": "gelu",
         "kernel": "gelu_rows_erf_checked_f32", "returns_status": True,
         "graph_slots": {"inputs": {"input": "evidence0_ff1"},
                         "outputs": {"output": "evidence0_gelu"}},
         "params": {"M": options, "C": feedforward,
                    "Q": options * feedforward, "call_constants": {
             "gelu_input_elements": options * feedforward,
             "gelu_input_stride": feedforward,
             "gelu_output_elements": options * feedforward,
             "gelu_output_stride": feedforward,
             "gelu_scratch_elements": options * feedforward,
             "gelu_rows": options, "gelu_channels": feedforward}}},
        linear("evidence0_feedforward2", "evidence0_gelu",
               "evidence0_ff2", options, feedforward, width,
               f"{PREFIX}.feedforward.3"),
    ]
    footer = [add("evidence0_feedforward_residual", "evidence0_attn_residual",
                  "evidence0_ff2", "evidence0_output")]
    graph["name"] = "clef_flash_evidence0_bounded"
    graph["native_entry"]["function"] = "ck_clef_evidence0"
    graph["sequence"].append("evidence0")
    graph["block_types"]["evidence0"] = {
        "sequence": ["header", "body", "footer"],
        "header": header, "body": {"type": "dense", "ops": body},
        "footer": footer}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if OUTPUT.read_text() != generated:
            raise SystemExit("Clef evidence0 circuit differs from canonical authoring")
    else:
        OUTPUT.write_text(generated)


if __name__ == "__main__":
    main()
