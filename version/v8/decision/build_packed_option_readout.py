"""Author the generated option readout shared by packed decision requests."""


def option_readout_graph(token_rows: int, questions: int, width: int, labels: int):
    if min(token_rows, questions, width, labels) <= 0:
        raise ValueError("readout dimensions must be positive")

    gather = "gather_rows_bf16_checked"
    projection = "linear_rows_bf16_checked"
    return {
        "version": 3,
        "name": "packed_option_readout",
        "family": "bounded_graph",
        "checked_native_entry": True,
        "contract": {"runtime_invariants": {"inference_only": True}},
        "activation_buffers": {
            "hidden": {"shape": [token_rows, width]},
            "answer_rows": {"shape": [questions]},
            "selected": {"shape": [questions, width]},
            "scores": {"shape": [questions, labels]},
        },
        "activation_bindings": {
            name: name for name in ("hidden", "answer_rows", "selected", "scores")
        },
        "native_entry": {
            "function": "ck_packed_option_readout",
            "params": [{"c_type": "uint8_t *", "name": "arena"},
                       {"c_type": "size_t", "name": "arena_bytes"}],
            "arena": {"pointer": "arena", "bytes": "arena_bytes"},
        },
        "sequence": ["block"],
        "block_types": {"block": {
            "sequence": ["header", "body", "footer"], "header": [], "footer": [],
            "body": {"type": "dense", "ops": [
                {
                    "id": "answer_rows", "op": gather, "kernel": gather, "returns_status": True,
                    "graph_slots": {
                        "inputs": {"input": "external:hidden", "indices": "external:answer_rows"},
                        "outputs": {"output": "selected"},
                    },
                    "params": {"R": token_rows, "S": questions, "D": width, "call_constants": {
                        "input_elements": token_rows * width, "index_elements": questions,
                        "output_elements": questions * width, "input_rows": token_rows,
                        "output_rows": questions, "width": width,
                    }},
                },
                {
                    "id": "option_projection", "op": projection, "kernel": projection,
                    "returns_status": True, "weight_refs": {"weight": "option.weight"},
                    "graph_slots": {
                        "inputs": {"input": "selected"}, "outputs": {"output": "scores"},
                    },
                    "params": {"M": questions, "K": width, "N": labels, "call_constants": {
                        "input_elements": questions * width, "weight_elements": labels * width,
                        "output_elements": questions * labels, "staged_elements": questions * labels,
                        "rows": questions, "input_width": width, "output_width": labels,
                    }},
                },
            ]},
        }},
    }
