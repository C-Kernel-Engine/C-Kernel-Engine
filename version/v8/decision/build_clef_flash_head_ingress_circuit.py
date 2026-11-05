"""Declare the Clef decision head's hidden and lexical ingress through v8 maps.

This bounded component stops before evidence routing and scoring. Its external
inputs are backbone hidden states, input IDs, and the backbone output embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/clef_flash_head_ingress_bounded.json"


def build_circuit(*, tokens=6, questions=2, options=3, channels=4096,
                  width=1024, vocabulary=16):
    if min(tokens, questions, options, channels, width, vocabulary) < 1:
        raise ValueError("all capacities and dimensions must be positive")
    shapes = {
        "hidden": [tokens, channels], "input_ids": [tokens],
        "question_spans": [questions, 2], "option_spans": [options, 2],
        "global_span": [1, 2],
        "normalized_hidden": [tokens, channels],
        "lexical_token_rows": [tokens, channels],
        "question_vectors": [questions, channels],
        "option_context_vectors": [options, channels],
        "lexical_option_vectors": [options, channels],
        "global_vector": [1, channels],
        "memory": [tokens, width],
        "question_projected": [questions, width],
        "option_context_projected": [options, width],
        "option_lexical_projected": [options, width],
        "option_question_projected": [questions, width],
        "global_projected": [1, width],
    }

    def norm():
        count = tokens * channels
        return {
            "id": "hidden_norm", "op": "layernorm",
            "kernel": "layernorm_rows_checked_param_epsilon_f32",
            "returns_status": True,
            "weight_refs": {"gamma": "hidden_norm.weight",
                            "beta": "hidden_norm.bias"},
            "graph_slots": {"inputs": {"input": "external:hidden"},
                            "outputs": {"output": "normalized_hidden"}},
            "params": {"M": tokens, "C": channels, "epsilon": 1e-5,
                       "Q": tokens * (channels + 2),
                       "call_constants": {
                "norm_input_elements": count, "norm_input_stride": channels,
                "norm_gamma_elements": channels, "norm_beta_elements": channels,
                "norm_output_elements": count, "norm_output_stride": channels,
                "norm_scratch_elements": tokens * (channels + 2),
                "norm_rows": tokens, "norm_channels": channels}},
        }

    def embedding():
        return {
            "id": "output_embedding_lookup",
            "op": "embedding_lookup_token_major_checked",
            "kernel": "audio_text_embedding_token_major_f32",
            "returns_status": True,
            "weight_refs": {"table": "output_embedding.weight"},
            "graph_slots": {"inputs": {"ids": "external:input_ids"},
                            "outputs": {"output": "lexical_token_rows"}},
            "params": {"T": tokens, "V": vocabulary, "C": channels,
                       "call_constants": {
                "id_elements": tokens, "table_elements": vocabulary * channels,
                "vocabulary": vocabulary, "channels": channels,
                "table_stride": channels, "output_elements": tokens * channels,
                "tokens": tokens, "output_stride": channels}},
        }

    def span(name, source, spans, output, count):
        return {
            "id": name, "op": "token_span_mean_checked",
            "kernel": "token_span_mean_checked_f32", "returns_status": True,
            "graph_slots": {"inputs": {"input": source, "spans": f"external:{spans}"},
                            "outputs": {"output": output}},
            "params": {"T": tokens, "S": count, "C": channels,
                       "call_constants": {
                "input_elements": tokens * channels, "input_stride": channels,
                "token_capacity": tokens, "valid_tokens": tokens,
                "channels": channels, "span_elements": 2 * count,
                "span_count": count, "output_elements": count * channels,
                "output_stride": channels}},
        }

    def linear(name, source, output, rows, weight):
        return {
            "id": name, "op": "linear_rows_checked_no_bias",
            "kernel": "linear_rows_checked_no_bias_f32", "returns_status": True,
            "weight_refs": {"weight": f"{weight}.weight"},
            "graph_slots": {"inputs": {"input": source},
                            "outputs": {"output": output}},
            "params": {"M": rows, "K": channels, "N": width,
                       "call_constants": {
                "linear_input_elements": rows * channels,
                "linear_input_stride": channels,
                "linear_weight_elements": width * channels,
                "linear_weight_stride": channels,
                "linear_output_elements": rows * width,
                "linear_output_stride": width, "linear_rows": rows,
                "linear_input_channels": channels,
                "linear_output_channels": width}},
        }

    def final_row():
        return {
            "id": "global_final_row", "op": "last_valid_row_checked",
            "kernel": "last_valid_row_checked_f32", "returns_status": True,
            "graph_slots": {"inputs": {
                "input": "normalized_hidden", "span": "external:global_span"},
                "outputs": {"output": "global_vector"}},
            "params": {"T": tokens, "C": channels,
                       "call_constants": {
                "input_elements": tokens * channels,
                "input_stride": channels,
                "token_capacity": tokens,
                "valid_tokens": tokens,
                "channels": channels,
                "span_elements": 2,
                "output_elements": channels}},
        }

    header = [norm(), embedding()]
    body = [
        span("question_pool", "normalized_hidden", "question_spans",
             "question_vectors", questions),
        span("option_context_pool", "normalized_hidden", "option_spans",
             "option_context_vectors", options),
        span("option_lexical_pool", "lexical_token_rows", "option_spans",
             "lexical_option_vectors", options),
        final_row(),
    ]
    footer = [
        linear("memory_projection", "normalized_hidden", "memory", tokens,
               "memory_projection"),
        linear("question_projection", "question_vectors", "question_projected",
               questions, "question_projection"),
        linear("option_context_projection", "option_context_vectors",
               "option_context_projected", options, "option_context_projection"),
        linear("option_lexical_projection", "lexical_option_vectors",
               "option_lexical_projected", options, "option_lexical_projection"),
        linear("option_question_projection", "question_vectors",
               "option_question_projected", questions,
               "option_question_projection"),
        linear("global_projection", "global_vector", "global_projected", 1,
               "global_projection"),
    ]
    return {
        "version": 3, "name": "clef_flash_head_ingress_bounded",
        "family": "decision_head", "checked_native_entry": True,
        "contract": {"runtime_invariants": {"inference_only": True,
                                            "production_kernel_heap_allocation": False}},
        "activation_buffers": {key: {"shape": value} for key, value in shapes.items()},
        "activation_bindings": {key: key for key in shapes},
        "native_entry": {"function": "ck_clef_head_ingress", "params": [
            {"c_type": "uint8_t *", "name": "arena"},
            {"c_type": "size_t", "name": "arena_bytes"}],
            "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
        "sequence": ["head_ingress"],
        "block_types": {"head_ingress": {"sequence": ["header", "body", "footer"],
                        "header": header, "body": {"type": "dense", "ops": body},
                        "footer": footer}},
        "runtime_constants": {}, "runtime_lengths": {},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if OUTPUT.read_text() != generated:
            raise SystemExit("Clef head-ingress circuit differs from canonical authoring")
    else:
        OUTPUT.write_text(generated)


if __name__ == "__main__":
    main()
