"""Compose one Drama 3 frame with its semantic ID and hidden row supplied."""
from version.v8.scripts.replay_bf16_projection_bundle_v8 import projection_graph
from version.v8.tts.build_drama3_bf16_tower_circuit import build_circuit as tower_layer


def _layer_label(shared, index):
    return f"{'shared' if shared else 'residual'}{index}"


def _compose_layer(graph, config, weights, ops, *, rows, shared, index, first,
                   hidden, code, frame_inputs, semantic_only, geometry):
    label = _layer_label(shared, index)
    prefix = label + "_"
    layer_key = f"audio_decoder_mtp{'_shared' if shared else ''}.layers.{index}."
    fragment, settings, layer_weights = tower_layer(
        rows, semantic_only=semantic_only, enrich=first, **geometry,
    )
    for name, spec in fragment["activation_buffers"].items():
        if name not in frame_inputs:
            target = prefix + name
            graph["activation_buffers"][target] = spec
            graph["activation_bindings"][target] = target
            config["activation_buffer_dtypes"][target] = settings["activation_buffer_dtypes"][name]

    external = {"external:hidden": hidden, "external:code_rows": code,
                "external:embedding_table": f"external:{prefix}embedding_table"}
    if geometry.get("prefix_rows"):
        external.update({f"external:prefix_{name}": f"external:{prefix}prefix_{name}"
                         for name in ("key", "value")})
    for node in fragment["block_types"]["block"]["body"]["ops"]:
        node["id"] = prefix + node["id"]
        slots = node["graph_slots"]
        slots["inputs"] = {port: external.get(source, source if source.startswith("external:") else prefix + source)
                           for port, source in slots["inputs"].items()}
        slots["outputs"] = {port: prefix + name for port, name in slots["outputs"].items()}
        for role, name in node.get("weight_refs", {}).items():
            if name != "inv_freq":
                node["weight_refs"][role] = layer_key + name
                weights[layer_key + name] = layer_weights[name]
        ops.append(node)
    return prefix


def _empty_tower(rows, name, semantic_only, geometry):
    graph, config, declarations = tower_layer(rows, semantic_only=semantic_only, **geometry)
    frame_inputs = ("hidden", "unseen_sum", "code_rows", "positions", "ranges")
    graph.update(name=name,
                 activation_buffers={key: graph["activation_buffers"][key] for key in frame_inputs},
                 activation_bindings={key: key for key in frame_inputs})
    graph["native_entry"]["function"] = "ck_" + name
    config["activation_buffer_dtypes"] = {key: config["activation_buffer_dtypes"][key] for key in frame_inputs}
    ops = graph["block_types"]["block"]["body"]["ops"] = []
    return graph, config, {"inv_freq": declarations["inv_freq"]}, ops, frame_inputs


def build_prefill_circuit(rows, *, shared_layers=4, residual_layers=9, **geometry):
    """Teacher-force tower rows with externally supplied next-frame labels.

    The code row for every layer uses the following frame's semantic mask.
    The caller supplies the reserved zero-table sentinel for text labels.
    Enrichment and backbone hidden states belong to the current row.
    """
    if any(type(n) is not int or n < 1 for n in (shared_layers, residual_layers)):
        raise ValueError("invalid codebook-prefill geometry")
    graph, config, weights, ops, frame_inputs = _empty_tower(
        rows, "drama3_bf16_tower_prefill", False, geometry,
    )
    hidden = "external:hidden"
    for i in range(shared_layers + residual_layers):
        shared = i < shared_layers
        index = i if shared else i - shared_layers
        code = "code_rows" if shared or index == 0 else f"residual{index}_code_rows"
        if code != "code_rows":
            graph["activation_buffers"][code] = {"shape": [rows]}
            graph["activation_bindings"][code] = code
            config["activation_buffer_dtypes"][code] = "i32"
        prefix = _compose_layer(
            graph, config, weights, ops, rows=rows, shared=shared, index=index, first=i == 0,
            hidden=hidden, code="external:" + code, frame_inputs=frame_inputs,
            semantic_only=False, geometry=geometry,
        )
        hidden = prefix + "output"
    return graph, config, weights


def build_circuit(*, shared_layers=4, residual_layers=9, residual_vocab=1024, forced_feedback=False, **geometry):
    vocabulary = geometry.get("vocabulary", 4096)
    if (any(type(n) is not int or n < 1 for n in (shared_layers, residual_layers, residual_vocab))
            or residual_vocab > vocabulary):
        raise ValueError("invalid codebook-frame geometry")
    if type(forced_feedback) is not bool:
        raise ValueError("forced_feedback must be boolean")
    graph, config, weights, ops, frame_inputs = _empty_tower(
        1, "drama3_bf16_codebook_frame", True, geometry,
    )
    width = config["embed_dim"]
    hidden, code, packed = "external:hidden", "external:code_rows", "external:code_rows"

    def buffer(name, shape, dtype="bf16"):
        graph["activation_buffers"][name] = {"shape": shape}
        graph["activation_bindings"][name] = name
        config["activation_buffer_dtypes"][name] = dtype

    for i in range(shared_layers + residual_layers):
        shared = i < shared_layers
        index = i if shared else i - shared_layers
        prefix = _compose_layer(
            graph, config, weights, ops, rows=1, shared=shared, index=index, first=i == 0,
            hidden=hidden, code=code, frame_inputs=frame_inputs, semantic_only=True, geometry=geometry,
        )
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
        if forced_feedback and index + 1 < residual_layers:
            forced = prefix + "forced_code"
            buffer(forced, [1], "i32")
            code = "external:" + forced
    return graph, config, weights
