"""Declare Sigma's compact BF16 normalization and Q/K/V projection boundary."""
from version.v8.scripts.replay_bf16_projection_bundle_v8 import projection_graph


def build_circuit(rows: int, *, width: int = 3072, q_width: int = 4096, kv_width: int = 1024):
    if any(type(n) is not int or n < 1 for n in (rows, width, q_width, kv_width)):
        raise ValueError("BF16 attention ingress dimensions must be positive integers")
    graph = projection_graph(rows, width, q_width)
    graph["name"] = "sigma_bf16_attention_ingress"
    graph["native_entry"]["function"] = "ck_sigma_bf16_attention_ingress"
    block = graph["block_types"]["block"]["body"]
    buffers = graph["activation_buffers"]
    buffers.pop("output")
    buffers["normalized"] = {"shape": [rows, width]}
    count = rows * width
    block["ops"] = [{
        "id": "normalize", "op": "rmsnorm_rows_bf16_checked",
        "kernel": "rmsnorm_rows_bf16_checked", "returns_status": True,
        "graph_slots": {"inputs": {"input": "external:input"}, "outputs": {"output": "normalized"}},
        "weight_refs": {"gamma": "input_layernorm.weight"},
        "params": {"M": rows, "K": width, "epsilon": 1e-5, "call_constants": {
            "input_elements": count, "gamma_elements": width, "output_elements": count,
            "work_elements": count + width, "rows": rows, "width": width}},
    }]
    for port, channels in (("q", q_width), ("k", kv_width), ("v", kv_width)):
        fragment = projection_graph(rows, width, channels)
        projection = fragment["block_types"]["block"]["body"]["ops"][0]
        projection.update(id=port, weight_refs={"weight": f"{port}_proj.weight"},
                          graph_slots={"inputs": {"input": "normalized"}, "outputs": {"output": port}})
        block["ops"].append(projection)
        buffers[port] = {"shape": [rows, channels]}
    graph["activation_bindings"] = {name: name for name in buffers}
    config = {"num_layers": 1, "embed_dim": width, "num_heads": 1, "num_kv_heads": 1,
              "head_dim": width, "intermediate_size": width, "context_length": rows, "max_seq_len": rows,
              "activation_buffer_dtypes": dict.fromkeys(buffers, "bf16")}
    return graph, config
