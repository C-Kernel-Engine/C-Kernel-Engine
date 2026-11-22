"""Author Inkling decoder blocks through ordinary checked lowering."""
import math

from version.v8.scripts.build_inkling2_mini_projection_circuit import (
    LAYER, _project, routed_expert_mixture, sliding_attention_replay,
)

from version.v8.scripts.build_inkling2_cross_pass_v8 import append_cross_pass_controls

ATTENTION_WEIGHTS = {
    "attn.wq_du.weight": "attn_q.weight",
    "attn.wk_dv.weight": "attn_k.weight",
    "attn.wv_dv.weight": "attn_v.weight",
    "attn.wr_du.weight": "attn_rel.weight",
    "attn.attn_sink_proj.weight": "attn_sink_proj.weight",
    "attn.gate_attn.weight": "attn_gate.weight",
    "attn.wo_ud.weight": "attn_output.weight",
    "attn.q_l2_clip.scale": "attn_q_clip.weight",
    "attn.rel_l2_clip.scale": "attn_rel_clip.weight",
    "attn.rel_logits_proj.proj": "attn_rel_bias.weight",
    "attn.k_sconv.weight": "attn_k_conv.weight",
    "attn.v_sconv.weight": "attn_v_conv.weight",
}

PACKED_PROJECTIONS = {
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_rel.weight",
    "attn_sink_proj.weight", "attn_gate.weight", "attn_output.weight",
}


def _bind_q8_projection(node, q8_activations):
    p = node["params"]
    width, channels, rows = p["K"], p["N"], p["M"]
    node.update(op="linear_rows_q8_0_checked", kernel="linear_rows_q8_0_checked_f32")
    p["call_constants"] = {
        "rows": rows, "width": width, "channels": channels,
        "input_elements": rows * width, "output_elements": rows * channels,
        "weight_bytes": width // 32 * 34 * channels,
    }
    if q8_activations:
        node["kernel"] = "linear_rows_q8_0_checked_q8_activation"
        p["call_constants"]["activation_bytes"] = width // 32 * 34


def first_decoder_block(prefill_tokens=4, width=2304, kv_heads=8, query_heads=24, dim=128,
                        relative_dim=16, band=1024, window=1024, inner=1152, experts=128,
                        top_k=8, slots=128, taps=4, epsilon=1e-6, q8_activations=False):
    """Keep the layer-zero entry available to the pinned replay."""
    return sliding_decoder_block(prefill_tokens, width, kv_heads, query_heads, dim, relative_dim,
                                 band, window, inner, experts, top_k, slots, taps, epsilon,
                                 q8_activations)


def sliding_decoder_block(prefill_tokens=4, width=2304, kv_heads=8, query_heads=24, dim=128,
                          relative_dim=16, band=1024, window=1024, inner=1152, experts=128,
                          top_k=8, slots=128, taps=4, epsilon=1e-6, q8_activations=False,
                          *, layer=0, state_slot=0, depth_states=0, previous_pass_states=0,
                          block_boundary=True, control_rank=64):
    """Compose one sliding block with explicit incoming residual and pass state."""
    return decoder_block(prefill_tokens, width, kv_heads, query_heads, dim, relative_dim,
                         band, window, inner, experts, top_k, slots, taps, epsilon, q8_activations,
                         layer=layer, state_slot=state_slot, depth_states=depth_states,
                         previous_pass_states=previous_pass_states, block_boundary=block_boundary,
                         control_rank=control_rank)


def decoder_block(prefill_tokens, width, kv_heads, query_heads, dim, relative_dim,
                  band, window, inner, experts, top_k, slots, taps, epsilon, q8_activations,
                  *, layer, state_slot, depth_states, previous_pass_states, block_boundary,
                  control_rank, full_attention=False, log_floor=8192, log_alpha=0.1):
    """Select the attention topology before lowering a single decoder occurrence."""
    if full_attention and (not layer or window or type(log_floor) is not int or log_floor < 1
                           or not math.isfinite(log_alpha) or log_alpha < 0):
        raise ValueError("invalid full-attention window or log position scale")
    if not (0 <= layer < 24 and 0 <= state_slot < 48 and 0 <= depth_states <= 5
            and 0 <= previous_pass_states <= 5
            and (not previous_pass_states or 1 <= control_rank <= width * 2)):
        raise ValueError("invalid decoder state or control geometry")
    if (layer != 0 and depth_states == 0) or (layer == 0 and not block_boundary):
        raise ValueError("invalid incoming depth bank for sliding block")
    if previous_pass_states and not 6 <= layer < 18:
        raise ValueError("cross-pass controls require a core layer")
    area = f"blk.{layer}."
    if width % 32 or query_heads * dim % 32:
        raise ValueError("packed Q8_0 projections require full input blocks")
    graph = sliding_attention_replay(
        prefill_tokens=prefill_tokens, width=width, kv_width=kv_heads * dim, taps=taps,
        slots=(state_slot,), kv_heads=kv_heads, query_heads=query_heads, relative_dim=relative_dim,
        band=band, window=window,
    )
    body = graph["block_types"]["block"]["body"]
    exports = graph["semantic_checkpoints"]["exports"]
    graph["name"] = "inkling2_mini_first_q8_decoder_block"
    if layer or state_slot:
        attention_kind = "full" if full_attention else "sliding"
        graph["name"] = f"inkling2_mini_{attention_kind}_layer{layer}_slot{state_slot}"
    if q8_activations:
        graph["name"] += "_q8_activation"
    graph["native_entry"]["function"] = "ck_" + graph["name"]
    graph["contract"]["runtime_invariants"].update(
        inner=inner, physical_layer=layer, depth_states=depth_states, previous_pass_states=previous_pass_states,
        block_boundary=block_boundary, attention="full" if full_attention else "sliding",
        scope="one decoder layer with packed GGUF weights"
    )
    for export in exports.values():
        for check in export["checkpoints"]:
            check["id"] = check["id"].replace("inkling2.layer6.", f"inkling2.layer{layer}.", 1)

    def buffer(name, shape, *, sequence=False):
        graph["activation_buffers"][name] = {"shape": shape}
        if sequence:
            graph["activation_buffers"][name]["lifetime"] = "sequence"
        graph["activation_bindings"][name] = name
        return name

    buffer("router_bias", [experts])
    buffer("expert_slots", [experts])
    for name in ("attn_post_conv", "ffn_post_conv"):
        buffer("history_" + name, [taps - 1, width], sequence=True)

    original = {phase: [op for op in body["ops"] if op["id"].startswith(f"inkling2_{phase}_slot{state_slot}_")]
                for phase in ("prefill", "append")}
    body["ops"] = []
    for phase, rows in (("prefill", prefill_tokens), ("append", 1)):
        entry = f"inkling2_{phase}_slot{state_slot}"
        resid = buffer(entry + "_hidden", [rows, width])
        count = rows * width

        def add(suffix, op, inputs, *, weights=None, params=None, checkpoint=False, shape=None):
            name = buffer(entry + "_" + suffix, shape or [rows, width])
            node = {
                "id": name, "op": op, "kernel": op + "_f32", "returns_status": True,
                "graph_slots": {"inputs": inputs, "outputs": {"output": name}},
                "params": params or {},
            }
            if weights:
                node["weight_refs"] = weights
            body["ops"].append(node)
            if checkpoint:
                exports[name] = {
                    "section": "body", "template_op_id": name, "op": op,
                    "checkpoints": [{"id": f"inkling2.layer{layer}.{name}", "producer": name, "tensor": name,
                                     "logical_layout": "token_major_contiguous",
                                     "axis_names": ["token", "channel"], "storage_dtype": "fp32"}],
                }
            return node

        def rms(suffix, source, weight):
            return add(suffix, "rmsnorm_rows_checked", {"input": source},
                       weights={"gamma": area + weight}, params={
                           "R": rows, "C": width, "eps": epsilon, "call_constants": {
                               "rows": rows, "width": width, "gamma_elements": width,
                               "input_stride": width, "output_stride": width,
                               "input_elements": count, "output_elements": count,
                           }}, checkpoint=True)["id"]

        def conv(weight, source):
            history = "external:history_" + weight
            node = add(weight, "causal_depthwise_residual_checked", {"input": source, "history": history},
                       weights={"weight": area + weight + ".weight"}, params={
                           "T": rows, "C": width, "K": taps, "H": taps - 1, "N": width,
                           "call_constants": {
                               "input_elements": count, "weight_elements": taps * width,
                               "history_elements": (taps - 1) * width, "output_elements": count,
                               "tokens": rows, "channels": width, "taps": taps, "history_rows": taps - 1,
                           }}, checkpoint=True)
            node["graph_slots"]["outputs"]["history"] = history
            return node["id"]

        def depth_mix(suffix, bank, current, states, weight):
            return add(suffix, "depth_residual_mix_checked", {"bank": bank, "current": current},
                       weights={"query": area + weight}, params={
                           "R": rows, "S": states, "C": width, "epsilon": epsilon,
                           "call_constants": {
                               "rows": rows, "states": states, "width": width, "bank_elements": states * count,
                               "current_elements": count, "query_elements": width, "output_elements": count,
                           }}, checkpoint=True)["id"]

        def sum_rows(suffix, left, right):
            return add(suffix, "add_rows_checked", {"left": left, "right": right},
                       params={"N": count, "call_constants": {
                           "left_elements": count, "right_elements": count,
                           "output_elements": count, "elements": count,
                       }}, checkpoint=True)["id"]

        def control(source, ports):
            shapes = {}
            controlled = append_cross_pass_controls(
                body["ops"], shapes, tokens=rows, width=width, rank=control_rank,
                states=previous_pass_states, layer=layer, bank=previous_bank, previous=previous_out,
                attention_input=source, ffn_input=source, ports=ports, prefix=entry + "_loop_", epsilon=epsilon,
            )
            for name, shape in shapes.items():
                buffer(name, shape)
            return controlled

        incoming = "external:" + resid
        ports = ("q", "k", "v")
        current = dict.fromkeys(ports, incoming)
        if depth_states:
            prior_depth = "external:" + buffer(entry + "_depth_bank", [rows, depth_states, width])
            if full_attention:
                current = {
                    port: depth_mix(port + "_attention_depth_mix", prior_depth, incoming, depth_states,
                                    "attn_res_query." + port)
                    for port in ports
                }
            else:
                mixed = depth_mix("attention_depth_mix", prior_depth, incoming, depth_states, "attn_res_query.weight")
                current = dict.fromkeys(ports, mixed)
        if previous_pass_states:
            previous_bank = "external:" + buffer(entry + "_previous_bank", [rows, previous_pass_states, width])
            previous_out = "external:" + buffer(entry + "_previous_output", [rows, width])
            if full_attention:
                current = {port: control(current[port], (port,))[port] for port in ports}
            else:
                current = control(current["q"], ports)
        if full_attention or previous_pass_states:
            normalized = {
                port: rms(port + "_attn_norm", source,
                          f"attn_{port}_inp_norm.weight" if full_attention else "attn_norm.weight")
                for port, source in current.items()
            }
        else:
            norm = rms("attn_norm", current["q"], "attn_norm.weight")
            normalized = dict.fromkeys(ports, norm)
        replaces = {f"external:{entry}_{port}_input": normalized[branch]
                    for port, branch in (("query", "q"), ("k", "k"), ("v", "v"))}
        for old in replaces:
            graph["activation_buffers"].pop(old.removeprefix("external:"))
            graph["activation_bindings"].pop(old.removeprefix("external:"))
        def log_scale(source, channels):
            output = add(source.removeprefix(entry + "_") + "_log_scale", "log_position_scale_rows_checked",
                         {"input": source}, shape=graph["activation_buffers"][source]["shape"], params={
                             "R": rows, "C": channels, "alpha": log_alpha,
                             "call_constants": {
                                 "rows": rows, "width": channels, "floor": log_floor,
                                 "position": 0 if phase == "prefill" else prefill_tokens,
                                 "input_elements": rows * channels, "output_elements": rows * channels,
                             }})["id"]
            return output

        scaled = {}
        for node in original[phase]:
            bind = node["graph_slots"]
            bind["inputs"] = {port: scaled.get(source, replaces.get(source, source))
                              for port, source in bind["inputs"].items()}
            if full_attention and node["op"] == "causal_depthwise_residual_checked":
                kind = "k" if node["id"] == entry + "_k_output" else "v"
                dynamic_width = kv_heads * taps
                name = buffer(entry + f"_{kind}_dynamic", [rows, dynamic_width])
                weight = f"attn_{kind}_conv_proj.weight"
                projection = _project(name, normalized[kind], width, dynamic_width, rows,
                                      weight, weight_prefix=area)
                _bind_q8_projection(projection, q8_activations)
                body["ops"].append(projection)
                node.update(op="causal_dynamic_depthwise_residual_checked",
                            kernel="causal_dynamic_depthwise_residual_checked_f32")
                node["params"].update(A=dynamic_width, KV=kv_heads)
                node["params"]["call_constants"].update(
                    heads=kv_heads, dynamic_width=dynamic_width, dynamic_elements=rows * dynamic_width,
                )
                bind["inputs"]["dynamic"] = name
            for port, name in node.get("weight_refs", {}).items():
                suffix = ATTENTION_WEIGHTS[name.removeprefix(LAYER)]
                node["weight_refs"][port] = area + suffix
                if suffix not in PACKED_PROJECTIONS:
                    continue
                _bind_q8_projection(node, q8_activations)
                if node["id"] in exports:
                    exports[node["id"]]["op"] = node["op"]
            body["ops"].append(node)
            if full_attention:
                name = node["id"]
                if name == entry + "_query":
                    scaled[name] = log_scale(name, query_heads * dim)
                elif name == entry + "_relative_scaled":
                    scaled[name] = log_scale(name, query_heads * relative_dim)
                if name in exports:
                    exports[name]["op"] = node["op"]

        residual = conv("attn_post_conv", entry + "_projected_attention")
        bank_count = depth_states
        if block_boundary:
            bank = incoming
            bank_count += 1
            if depth_states:
                left_width = depth_states * width
                joined_width = bank_count * width
                bank = add("next_depth_bank", "concat_feature_rows_checked", {"left": prior_depth, "right": incoming},
                           params={"R": rows, "L": left_width, "C": width, "J": joined_width, "call_constants": {
                               "rows": rows, "left_width": left_width, "right_width": width, "width": joined_width,
                               "left_elements": rows * left_width, "right_elements": count,
                               "output_elements": rows * joined_width,
                           }}, shape=[rows, bank_count, width])["id"]
        else:
            bank = prior_depth
            residual = sum_rows("attention_residual", incoming, residual)
        mix = depth_mix("depth_mix", bank, residual, bank_count, "ffn_res_query.weight")
        if previous_pass_states:
            mix = control(mix, ("ffn",))["ffn"]
        mlp_input = rms("ffn_norm", mix, "ffn_norm.weight")
        moe = routed_expert_mixture(rows, width, inner, experts, top_k, slots,
                                   layer=layer, storage="q8_0", q8_activations=q8_activations)
        names = {"hidden": mlp_input, "expert_slots": "external:expert_slots",
                 "correction_bias": "external:router_bias"}
        for name, spec in moe["activation_buffers"].items():
            if name not in names:
                names[name] = buffer(entry + "_" + name, spec["shape"])
        for node in moe["block_types"]["block"]["body"]["ops"]:
            node["id"] = names[node["id"]]
            for bindings in node["graph_slots"].values():
                for port, source in bindings.items():
                    bindings[port] = names[source.removeprefix("external:")]
            if node["op"] == "linear_rows_checked_no_bias":
                node["weight_refs"]["weight"] = area + "ffn_gate_inp.weight"
            elif node["op"] == "routed_softcapped_experts_checked":
                node["weight_refs"] = {port: area + f"expert_bank.{port}" for port in node["weight_refs"]}
            body["ops"].append(node)
        mlp_output = conv("ffn_post_conv", names["mixture"])
        out = sum_rows("layer_output", residual, mlp_output)
        for segment in graph["native_entry"]["segments"]:
            if segment["function"] == "ck_" + entry:
                segment["last_op"] = out
    return graph


def first_block_weights(graph):
    """Describe the logical tensors loaded into the bounded native block."""
    result = {}
    for op in graph["block_types"]["block"]["body"]["ops"]:
        p = op["params"]
        for port, name in op.get("weight_refs", {}).items():
            mode = "F32"
            if op["op"] == "routed_softcapped_experts_checked":
                shape = [p["S"], p["H"], p["I"]] if port == "down" else [p["S"], p["I"], p["H"]]
                mode = "Q8_0"
            elif port in ("gamma", "query"):
                shape = [p["C"]]
            else:
                shape = [p["N"], p["K"]]
                if op["op"] == "linear_rows_q8_0_checked":
                    mode = "Q8_0"
            previous = result.setdefault(name, (shape, mode))
            if previous != (shape, mode):
                raise ValueError(f"inconsistent phase weight binding for {name}")
    return result
