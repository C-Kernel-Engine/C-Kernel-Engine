"""Declare the embedding feedback from one Drama 3 semantic frame."""

import math

from version.v8.scripts.replay_bf16_projection_bundle_v8 import projection_graph


def build_circuit(width=2048, vocabulary=4096):
    """Mix the first residual embedding into the backbone input.

    The remaining eight embeddings are summed in FP32 and rounded once for
    the tower. The caller binds the token embedding and the nine codebook
    indices into the shared embedding table.
    """
    if any(type(value) is not int or value < 1 for value in (width, vocabulary)):
        raise ValueError("invalid Drama 3 feedback geometry")
    graph = projection_graph(1, width, width)
    graph.update(name="drama3_frame_feedback", activation_buffers={}, activation_bindings={})
    graph["native_entry"]["function"] = "ck_drama3_frame_feedback"
    ops = graph["block_types"]["block"]["body"]["ops"]
    ops.clear()
    config = {
        "num_layers": 1, "embed_dim": width, "num_heads": 1, "num_kv_heads": 1,
        "head_dim": width, "intermediate_size": width, "context_length": 1,
        "max_seq_len": 1, "activation_buffer_dtypes": {},
    }

    def buffer(name, shape, dtype):
        graph["activation_buffers"][name] = {"shape": shape}
        graph["activation_bindings"][name] = name
        config["activation_buffer_dtypes"][name] = dtype

    def emit(name, op, inputs, *, kernel=None, dtype="bf16", port="out", dims=None, constants=None, **params):
        buffer(name, [1, width], dtype)
        ops.append({
            "id": name, "op": op, "kernel": kernel or op, "returns_status": True,
            "graph_slots": {"inputs": inputs, "outputs": {port: name}},
            "params": {**({"N": width} if dims is None else dims), **params,
                       "call_constants": {"elements": width} if constants is None else constants},
        })

    table_rows = 10 * vocabulary
    buffer("text_embedding", [1, width], "bf16")
    buffer("codebook_embeddings", [table_rows, width], "bf16")
    for index in range(1, 10):
        buffer(f"row{index}", [1], "i32")
        emit(f"code{index}", "gather_rows_bf16_checked",
             {"input": "external:codebook_embeddings", "indices": f"external:row{index}"},
             port="output", dims={"R": table_rows, "S": 1, "D": width},
             constants={"input_elements": table_rows * width, "index_elements": 1,
                        "output_elements": width, "input_rows": table_rows, "output_rows": 1,
                        "width": width})
    emit("mixed", "add_bf16_checked", {"left": "external:text_embedding", "right": "code1"})
    emit("scaled", "scale_bf16_checked", {"input": "mixed"},
         kernel="scale_scalar_f32_bf16_checked", scale=1 / math.sqrt(2))
    emit("backbone_embedding", "widen_bf16_checked", {"input": "scaled"}, dtype="fp32")

    for index in range(2, 10):
        emit(f"wide{index}", "widen_bf16_checked", {"input": f"code{index}"}, dtype="fp32")
    previous = "wide2"
    for index in range(3, 10):
        emit(f"sum{index}", "add_rows_checked", {"left": previous, "right": f"wide{index}"},
             kernel="add_rows_checked_f32", dtype="fp32", port="output",
             constants={"left_elements": width, "right_elements": width,
                        "output_elements": width, "elements": width})
        previous = f"sum{index}"
    emit("tower_enrichment", "narrow_f32_bf16_checked", {"input": previous})
    return graph, config
