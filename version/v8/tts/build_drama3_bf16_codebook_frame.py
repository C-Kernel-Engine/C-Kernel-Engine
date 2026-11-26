"""Compose one Drama 3 frame with its semantic ID and hidden row supplied."""
from version.v8.scripts.replay_bf16_projection_bundle_v8 import projection_graph
from version.v8.tts.build_drama3_bf16_tower_circuit import build_circuit as tower_layer


def build_circuit(*, shared_layers=4, residual_layers=9, residual_vocab=1024, **geometry):
    vocabulary = geometry.get("vocabulary", 4096)
    if (any(type(n) is not int or n < 1 for n in (shared_layers, residual_layers, residual_vocab))
            or residual_vocab > vocabulary):
        raise ValueError("invalid codebook-frame geometry")
    graph, config, declarations = tower_layer(1, semantic_only=True, **geometry)
    shared_inputs = ("hidden", "unseen_sum", "code_rows", "positions", "ranges")
    graph.update(name="drama3_bf16_codebook_frame",
                 activation_buffers={k: graph["activation_buffers"][k] for k in shared_inputs},
                 activation_bindings={k: k for k in shared_inputs})
    graph["native_entry"]["function"] = "ck_drama3_bf16_codebook_frame"
    config["activation_buffer_dtypes"] = {k: config["activation_buffer_dtypes"][k] for k in shared_inputs}
    ops = graph["block_types"]["block"]["body"]["ops"] = []
    weights = {"inv_freq": declarations["inv_freq"]}
    width = config["embed_dim"]
    hidden, code, packed = "external:hidden", "external:code_rows", "external:code_rows"

    def buffer(name, shape, dtype="bf16"):
        graph["activation_buffers"][name] = {"shape": shape}
        graph["activation_bindings"][name] = name
        config["activation_buffer_dtypes"][name] = dtype

    for i in range(shared_layers + residual_layers):
        shared = i < shared_layers
        index = i if shared else i - shared_layers
        prefix = f"{'shared' if shared else 'residual'}{index}_"
        layer_key = f"audio_decoder_mtp{'_shared' if shared else ''}.layers.{index}."
        fragment, settings, layer_weights = tower_layer(1, semantic_only=True, enrich=i == 0, **geometry)
        for name, spec in fragment["activation_buffers"].items():
            if name not in shared_inputs:
                buffer(prefix + name, spec["shape"], settings["activation_buffer_dtypes"][name])
        for node in fragment["block_types"]["block"]["body"]["ops"]:
            node["id"] = prefix + node["id"]
            slots = node["graph_slots"]
            external = {"external:hidden": hidden, "external:code_rows": code,
                        "external:embedding_table": f"external:{prefix}embedding_table"}
            for port, source in slots["inputs"].items():
                slots["inputs"][port] = external.get(source, source if source.startswith("external:") else prefix + source)
            slots["outputs"] = {port: prefix + name for port, name in slots["outputs"].items()}
            for role, name in node.get("weight_refs", {}).items():
                if name != "inv_freq":
                    node["weight_refs"][role] = layer_key + name
                    weights[layer_key + name] = layer_weights[name]
            ops.append(node)
        hidden = prefix + "output"
        if shared:
            continue

        logits, code = prefix + "scores", prefix + "code"
        buffer(logits, [1, residual_vocab])
        buffer(code, [1], "i32")
        head = projection_graph(1, width, residual_vocab)["block_types"]["block"]["body"]["ops"][0]
        name = f"audio_decoder_mtp.heads.{index}.weight"
        weights[name] = ([residual_vocab, width], "BF16")
        head.update(id=logits, weight_refs={"weight": name},
                    graph_slots={"inputs": {"input": hidden}, "outputs": {"output": logits}})
        if geometry.get("blas"):
            head["kernel"] = "linear_rows_bf16_blas_checked"
            head["params"]["call_constants"]["work_elements"] = width + width * residual_vocab + residual_vocab
        ops.append(head)
        op = "argmax_values_bf16_checked"
        ops.append({"id": code, "op": op, "kernel": op, "returns_status": True,
                    "graph_slots": {"inputs": {"values": logits}, "outputs": {"index": code}},
                    "params": {"N": residual_vocab, "call_constants": {"elements": residual_vocab, "index_elements": 1}}})
        codes = "frame_codes" if index + 1 == residual_layers else prefix + "codes"
        size = index + 1
        buffer(codes, [size + 1], "i32")
        op = "concat_i32_checked"
        ops.append({"id": codes, "op": op, "kernel": op, "returns_status": True,
                    "graph_slots": {"inputs": {"left": packed, "right": code}, "outputs": {"output": codes}},
                    "params": {"L": size, "R": 1, "O": size + 1, "call_constants": {
                        "left_elements": size, "left_count": size, "right_elements": 1,
                        "right_count": 1, "output_elements": size + 1,
                    }}})
        packed = codes
    return graph, config, weights
