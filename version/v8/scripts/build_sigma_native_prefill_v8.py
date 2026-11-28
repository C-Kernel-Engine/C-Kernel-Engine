"""Build the clean-token encoder entry that supplies Sigma's read-only prefix K/V."""
from version.v8.scripts.build_sigma_bf16_denoiser_stack_v8 import build_circuit as encoder


def build_circuit(rows, *, vocabulary=131072, **geometry):
    graph, config, schema = encoder(rows, vocabulary=vocabulary, readout=False, **geometry)
    graph["name"] = "sigma_native_clean_prefix"
    graph["native_entry"]["function"] = "ck_sigma_native_clean_prefix"
    width = config["embed_dim"]
    graph["activation_buffers"]["prefix_tokens"] = {"shape": [rows]}
    graph["activation_bindings"]["prefix_tokens"] = "prefix_tokens"
    config["activation_buffer_dtypes"]["prefix_tokens"] = "i32"
    schema["encoder.embed_tokens.weight"] = ([vocabulary, width], "BF16")
    ops = graph["block_types"]["block"]["body"]["ops"]
    for node in ops:
        inputs = node["graph_slots"]["inputs"]
        for port, source in inputs.items():
            if source == "external:input":
                inputs[port] = "input"
    ops.insert(0, {
        "id": "prefix_token_lookup", "op": "gather_rows_bf16_checked",
        "kernel": "gather_weight_rows_bf16_checked", "returns_status": True,
        "weight_refs": {"input": "encoder.embed_tokens.weight"},
        "graph_slots": {
            "inputs": {"indices": "external:prefix_tokens"},
            "outputs": {"output": "input"},
        },
        "params": {"R": vocabulary, "S": rows, "D": width, "call_constants": {
            "input_elements": vocabulary * width, "index_elements": rows, "output_elements": rows * width,
            "input_rows": vocabulary, "output_rows": rows, "width": width,
        }},
    })
    return graph, config, schema
