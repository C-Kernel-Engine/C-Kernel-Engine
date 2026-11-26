"""Declare one compact Sigma denoiser block with explicit key visibility."""
from version.v8.scripts.build_sigma_bf16_attention_ingress_v8 import build_circuit as ingress
from version.v8.scripts.replay_bf16_projection_bundle_v8 import projection_graph


def build_circuit(rows, *, width=3072, heads=32, kv_heads=8, dim=128, inner=9216, prefix_rows=0, blas=False):
    if any(type(size) is not int or size < 1 for size in (rows, width, heads, kv_heads, dim, inner)):
        raise ValueError("Sigma block requires positive integer geometry")
    if type(prefix_rows) is not int or prefix_rows < 0 or prefix_rows + rows > 4096:
        raise ValueError("unsupported prefix and block length")
    if dim % 2 or heads % kv_heads:
        raise ValueError("unsupported Sigma block geometry")
    memory_rows = prefix_rows + rows
    q_width, kv_width = heads * dim, kv_heads * dim
    graph, config = ingress(rows, width=width, q_width=q_width, kv_width=kv_width)
    graph.update(name="sigma_bf16_block")
    graph["native_entry"]["function"] = "ck_sigma_bf16_block"
    ops = graph["block_types"]["block"]["body"]["ops"]
    weights = {"input_layernorm.weight": ([width], "BF16")}

    def buffer(name, shape, dtype="bf16"):
        graph["activation_buffers"][name] = {"shape": shape}
        graph["activation_bindings"][name] = name
        config["activation_buffer_dtypes"][name] = dtype

    def emit(name, op, inputs, params, channels, *, output="out", provider=None, weight_refs=None, shape=None):
        buffer(name, shape or [rows, channels])
        node = {"id": name, "op": op, "kernel": provider or op, "returns_status": True,
                "graph_slots": {"inputs": inputs, "outputs": {output: name}}, "params": params}
        if weight_refs:
            node["weight_refs"] = weight_refs
        ops.append(node)

    def projection_provider(op, source_width, channels):
        if blas:
            op["kernel"] = "linear_rows_bf16_blas_checked"
            op["params"]["call_constants"]["work_elements"] = (
                rows * source_width + channels * source_width + rows * channels
            )

    def project(name, source, source_width, channels):
        fragment = projection_graph(rows, source_width, channels)
        op = fragment["block_types"]["block"]["body"]["ops"][0]
        op.update(id=name, weight_refs={"weight": f"{name}_proj.weight"},
                  graph_slots={"inputs": {"input": source}, "outputs": {"output": name}})
        projection_provider(op, source_width, channels)
        buffer(name, [rows, channels])
        weights[f"{name}_proj.weight"] = ([channels, source_width], "BF16")
        ops.append(op)

    for op, channels in zip(ops[1:], (q_width, kv_width, kv_width), strict=True):
        weights[op["weight_refs"]["weight"]] = ([channels, width], "BF16")
        projection_provider(op, width, channels)
    buffer("positions", [rows], "i32")
    buffer("ranges", [rows, 4], "i32")
    weights["inv_freq"] = ([dim // 2], "F32")
    for port, count in (("q", heads), ("k", kv_heads)):
        emit(f"{port}_rotary", "rope_split_positions_bf16_checked",
             {"input": port, "positions": "external:positions"},
             {"T": rows, "H": count, "D": dim, "call_constants": {
                 "tokens": rows, "heads": count, "dim": dim, "elements": rows * count * dim,
                 "frequencies": dim // 2,
             }}, count * dim, weight_refs={"inv_freq": "inv_freq"})

    key, value = "k_rotary", "v"
    if prefix_rows:
        lengths = {"left": prefix_rows * kv_width, "right": rows * kv_width,
                   "out": memory_rows * kv_width}
        for name, tail in (("key", key), ("value", value)):
            buffer(f"prefix_{name}", [prefix_rows, kv_width])
            emit(f"combined_{name}", "concat_features_bf16_checked",
                 {"left": f"external:prefix_{name}", "right": tail},
                 {"R": 1, "A": lengths["left"], "B": lengths["right"], "C": lengths["out"],
                  "call_constants": {"rows": 1, **{f"{side}_width": n for side, n in lengths.items()},
                                     **{f"{side}_elements": n for side, n in lengths.items()}}},
                 kv_width, shape=[memory_rows, kv_width])
        key, value = "combined_key", "combined_value"

    emit("context", "attention_ranges_checked",
         {"query": "q_rotary", "key": key, "value": value, "ranges": "external:ranges"},
         {"Q": rows, "S": memory_rows, "H": heads, "G": kv_heads, "D": dim,
          "X": rows * q_width + memory_rows, "scale": dim ** -0.5, "call_constants": {
              "query_elements": rows * q_width, "key_elements": memory_rows * kv_width,
              "value_elements": memory_rows * kv_width, "output_elements": rows * q_width,
              "scratch_elements": rows * q_width + memory_rows, "range_elements": rows * 4,
              "query_tokens": rows, "memory_tokens": memory_rows, "heads": heads,
              "kv_heads": kv_heads, "head_dim": dim,
          }}, q_width, provider="attention_ranges_checked_bf16_math", output="output")
    project("o", "context", q_width, width)

    def pointwise(name, op, ports, channels):
        count = rows * channels
        emit(name, op, ports, {"N": count, "call_constants": {"elements": count}}, channels)

    pointwise("attention_residual", "add_bf16_checked", {"left": "external:input", "right": "o"}, width)
    elements = rows * width
    weights["post_attention_layernorm.weight"] = ([width], "BF16")
    emit("post_norm", "rmsnorm_rows_bf16_checked", {"input": "attention_residual"},
         {"M": rows, "K": width, "epsilon": 1e-5, "call_constants": {
             "input_elements": elements, "gamma_elements": width, "output_elements": elements,
             "work_elements": elements + width, "rows": rows, "width": width,
         }}, width, output="output", weight_refs={"gamma": "post_attention_layernorm.weight"})
    project("gate", "post_norm", width, inner)
    project("up", "post_norm", width, inner)
    pointwise("swiglu", "swiglu_bf16_checked", {"gate": "gate", "up": "up"}, inner)
    project("down", "swiglu", inner, width)
    pointwise("output", "add_bf16_checked", {"left": "attention_residual", "right": "down"}, width)
    return graph, config, weights
