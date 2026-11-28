"""Declare Karpo's FP32 readout from supplied final backbone marker rows."""

from version.v8.scripts.replay_bf16_projection_bundle_v8 import projection_graph


def build_circuit(questions, width=1024, labels=255):
    if any(type(size) is not int or size < 1 for size in (questions, width, labels)):
        raise ValueError("readout dimensions must be positive")
    graph = projection_graph(questions, width, labels)
    graph["name"] = "karpo_marker_readout"
    graph["native_entry"]["function"] = "ck_karpo_marker_readout"
    shapes = {"hidden": [questions, width], "scores": [questions, labels],
              "choice_counts": [questions], "probabilities": [questions, labels]}
    graph["activation_buffers"] = {name: {"shape": shape} for name, shape in shapes.items()}
    graph["activation_bindings"] = {name: name for name in shapes}
    projection = "linear_rows_checked_no_bias"
    softmax = "prefix_softmax_rows_checked"
    graph["block_types"]["block"]["body"]["ops"] = [
        {
            "id": "readout", "op": projection, "kernel": projection + "_f32", "returns_status": True,
            "weight_refs": {"weight": "readout.weight"},
            "graph_slots": {"inputs": {"input": "external:hidden"}, "outputs": {"output": "scores"}},
            "params": {"M": questions, "K": width, "N": labels, "call_constants": {
                "linear_input_elements": questions * width, "linear_input_stride": width,
                "linear_weight_elements": labels * width, "linear_weight_stride": width,
                "linear_output_elements": questions * labels, "linear_output_stride": labels,
                "linear_rows": questions, "linear_input_channels": width, "linear_output_channels": labels,
            }},
        },
        {
            "id": "probabilities", "op": softmax, "kernel": softmax + "_f32", "returns_status": True,
            "graph_slots": {"inputs": {"logits": "scores", "counts": "external:choice_counts"},
                            "outputs": {"output": "probabilities"}},
            "params": {"R": questions, "C": labels, "call_constants": {
                "logits_elements": questions * labels, "count_elements": questions,
                "output_elements": questions * labels, "rows": questions, "columns": labels,
            }},
        },
    ]
    config = {"num_layers": 1, "embed_dim": width, "num_heads": 1, "num_kv_heads": 1,
              "head_dim": width, "intermediate_size": width, "context_length": questions,
              "max_seq_len": questions, "activation_buffer_dtypes": {"choice_counts": "i32"}}
    return graph, config, {"readout.weight": ([labels, width], "F32")}
