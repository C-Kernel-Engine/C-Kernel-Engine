"""Author one teacher-forced Drama 3 shared codebook-tower layer."""
from version.v8.scripts.replay_bf16_projection_bundle_v8 import projection_graph


def build_circuit(rows, *, width=2048, heads=16, kv_heads=2, dim=256, rotary=64,
                  inner=6144, vocabulary=4096, blas=False):
    if (any(type(n) is not int or n < 1 for n in (rows, width, heads, kv_heads, dim, rotary, inner, vocabulary))
            or heads % kv_heads or rotary > dim or rotary % 2):
        raise ValueError("invalid codebook-tower geometry")
    graph = projection_graph(rows, width, width)
    graph.update(name="drama3_bf16_shared_tower_layer", activation_buffers={}, activation_bindings={})
    graph["native_entry"]["function"] = "ck_drama3_bf16_shared_tower_layer"
    ops = graph["block_types"]["block"]["body"]["ops"] = []
    config = {"num_layers": 1, "embed_dim": width, "num_heads": 1, "num_kv_heads": 1,
              "head_dim": width, "intermediate_size": inner, "context_length": rows,
              "max_seq_len": rows, "activation_buffer_dtypes": {}}
    weights = {"inv_freq": ([rotary // 2], "F32")}

    def buffer(name, shape, dtype="bf16"):
        graph["activation_buffers"][name] = {"shape": shape}
        graph["activation_bindings"][name] = name
        config["activation_buffer_dtypes"][name] = dtype

    def emit(name, op, inputs, params, channels, *, output="out", provider=None, weight_refs=None):
        buffer(name, [rows, channels])
        layer = {"id": name, "op": op, "kernel": provider or op, "returns_status": True,
                 "graph_slots": {"inputs": inputs, "outputs": {output: name}}, "params": params}
        if weight_refs:
            layer["weight_refs"] = weight_refs
        ops.append(layer)

    def project(name, source, source_width, out_width, weight):
        provider = "linear_rows_bf16_blas_checked" if blas else "linear_rows_bf16_checked"
        params = projection_graph(rows, source_width, out_width)["block_types"]["block"]["body"]["ops"][0]["params"]
        if blas:
            params["call_constants"]["work_elements"] = rows * (source_width + out_width) + source_width * out_width
        weights[weight] = ([out_width, source_width], "BF16")
        emit(name, "linear_rows_bf16_checked", {"input": source}, params, out_width,
             output="output", provider=provider, weight_refs={"weight": weight})

    def normalize(name, source, channels, weight, *, count=1):
        elements = rows * channels * count
        weights[weight] = ([channels], "BF16")
        emit(name, "rmsnorm_rows_bf16_checked", {"input": source},
             {"M": rows * count, "K": channels, "epsilon": 1e-6, "call_constants": {
                 "input_elements": elements, "gamma_elements": channels, "output_elements": elements,
                 "work_elements": elements + channels, "rows": rows * count, "width": channels,
             }}, channels * count, output="output", provider="rmsnorm_rows_centered_single_round_bf16_checked",
             weight_refs={"gamma": weight})

    def pointwise(name, op, inputs, channels):
        elements = rows * channels
        emit(name, op, inputs, {"N": elements, "call_constants": {"elements": elements}}, channels)

    for name in ("hidden", "unseen_sum"):
        buffer(name, [rows, width])
    buffer("code_rows", [rows], "i32")
    buffer("positions", [rows], "i32")
    buffer("ranges", [rows, 4], "i32")
    table_rows = vocabulary + 1
    buffer("embedding_table", [table_rows, width])
    emit("embedding", "gather_rows_bf16_checked",
         {"input": "external:embedding_table", "indices": "external:code_rows"},
         {"R": table_rows, "S": rows, "D": width, "call_constants": {
             "input_elements": table_rows * width, "index_elements": rows,
             "output_elements": rows * width, "input_rows": table_rows, "output_rows": rows, "width": width,
         }}, width, output="output")
    pointwise("enriched", "add_bf16_checked", {"left": "embedding", "right": "external:unseen_sum"}, width)
    normalize("enorm", "enriched", width, "enorm.weight")
    normalize("hnorm", "external:hidden", width, "hnorm.weight")
    emit("concat", "concat_features_bf16_checked", {"left": "enorm", "right": "hnorm"},
         {"R": rows, "A": width, "B": width, "C": 2 * width, "call_constants": {
             "rows": rows, "left_width": width, "right_width": width, "out_width": 2 * width,
             "left_elements": rows * width, "right_elements": rows * width, "out_elements": 2 * rows * width,
         }}, 2 * width)
    project("mixed", "concat", 2 * width, width, "eh_proj.weight")
    layer = "transformer_layer."
    attn = layer + "attention."
    normalize("attn_norm", "mixed", width, layer + "attention_norm.weight")
    query_width, key_width = heads * dim, kv_heads * dim
    for port, channels in (("q", query_width), ("attention_gate", query_width), ("k", key_width), ("v", key_width)):
        project(port, "attn_norm", width, channels, attn + port + ".weight")
    for port, count in (("q", heads), ("k", kv_heads)):
        normalize(f"{port}_norm", port, dim, attn + f"{port}_norm.weight", count=count)
        emit(f"{port}_rotary", "rope_interleaved_partial_bf16_checked",
             {"input": f"{port}_norm", "positions": "external:positions"},
             {"T": rows, "H": count, "D": dim, "F": rotary // 2, "call_constants": {
                 "tokens": rows, "heads": count, "dim": dim, "elements": rows * count * dim,
                 "frequencies": rotary // 2,
             }}, count * dim, weight_refs={"inv_freq": "inv_freq"})
    emit("context", "attention_ranges_checked",
         {"query": "q_rotary", "key": "k_rotary", "value": "v", "ranges": "external:ranges"},
         {"Q": rows, "S": rows, "H": heads, "G": kv_heads, "D": dim, "X": rows * query_width + rows,
          "scale": dim ** -0.5, "call_constants": {
              "query_elements": rows * query_width, "key_elements": rows * key_width,
              "value_elements": rows * key_width, "range_elements": rows * 4,
              "output_elements": rows * query_width, "scratch_elements": rows * query_width + rows,
              "query_tokens": rows, "memory_tokens": rows, "heads": heads, "kv_heads": kv_heads, "head_dim": dim,
          }}, query_width, output="output", provider="attention_ranges_checked_bf16_math")
    pointwise("gated_context", "sigmoid_mul_bf16_checked", {"gate": "attention_gate", "up": "context"}, query_width)
    project("o", "gated_context", query_width, width, attn + "wo.weight")
    pointwise("attn_residual", "add_bf16_checked", {"left": "mixed", "right": "o"}, width)
    normalize("ffn_norm", "attn_residual", width, layer + "ffn_norm.weight")
    mlp = layer + "feed_forward."
    project("gate", "ffn_norm", width, inner, mlp + "w1.weight")
    project("up", "ffn_norm", width, inner, mlp + "w3.weight")
    pointwise("swiglu", "swiglu_bf16_checked", {"gate": "gate", "up": "up"}, inner)
    project("down", "swiglu", inner, width, mlp + "w2.weight")
    pointwise("pre_final_norm", "add_bf16_checked", {"left": "attn_residual", "right": "down"}, width)
    normalize("output", "pre_final_norm", width, "final_layernorm.weight")
    return graph, config, weights
