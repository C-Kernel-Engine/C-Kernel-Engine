"""Bounded Inkling layer-6 projection replay circuits using checked providers."""

from version.v8.scripts.checked_circuit_authoring_v8 import linear

SOURCE_REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"
TRANSFORMERS_REVISION = "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3"
LAYER = "model.llm.layers.6."


def _project(name, source, width, channels, tokens, weight, *, weight_prefix=LAYER):
    op = linear(name, source, name, width, channels, tokens, "unused")
    op.update(op="linear_rows_checked_no_bias", kernel="linear_rows_checked_no_bias_f32",
              weight_refs={"weight": weight_prefix + weight})
    del op["params"]["call_constants"]["linear_bias_elements"]
    return op


def _graph(kind, tokens, width, shapes, ops, *, layer=6, scope=None):
    if type(tokens) is not int or not 1 <= tokens <= 16:
        raise ValueError("expected 1..16 fixed token rows")
    if any(type(n) is not int or not 1 <= n <= 8192 for shape in shapes.values() for n in shape):
        raise ValueError("unsupported projection geometry")
    name = f"inkling2_mini_layer{layer}_{kind}_fp32"
    checks = {}
    for op in ops:
        operation = op["id"]
        checks[operation] = {
            "section": "body", "template_op_id": operation, "op": op["op"],
            "checkpoints": [{
                "id": f"inkling2.layer{layer}.{kind}.{operation}", "producer": operation,
                "tensor": operation, "logical_layout": "token_major_contiguous",
                "axis_names": ["token", "channel"], "storage_dtype": "fp32",
            }],
        }
    return {
        "version": 3, "name": name, "family": "bounded_graph", "checked_native_entry": True,
        "contract": {
            "runtime_invariants": {
                "inference_only": True, "tokens": tokens, "width": width, "heads": 1, "mult": 1,
                "scope": scope or "supplied projection operands, FP32 output, isolated layer-6 component",
            },
            "provenance": {"checkpoint_revision": SOURCE_REVISION, "transformers_revision": TRANSFORMERS_REVISION},
        },
        "activation_buffers": {key: {"shape": shape} for key, shape in shapes.items()},
        "activation_bindings": {key: key for key in shapes},
        "native_entry": {
            "function": f"ck_{name}",
            "params": [{"c_type": "uint8_t *", "name": "arena"}, {"c_type": "size_t", "name": "arena_bytes"}],
            "arena": {"pointer": "arena", "bytes": "arena_bytes"},
        },
        "sequence": ["block"],
        "block_types": {"block": {
            "sequence": ["header", "body", "footer"], "header": [], "footer": [],
            "body": {"type": "dense", "ops": ops},
        }},
        "semantic_checkpoints": {
            "schema": "cke.semantic_checkpoint_contract", "schema_version": 1,
            "exports": checks,
        },
    }


def attention_projections(tokens=8, width=2304, q=3072, kv=1024, relative=384, heads=24):
    """Q/K/V, relative-logit and sink projections from supplied normalized rows."""
    bindings = {
        "q": ("query_input", q, "attn.wq_du.weight"),
        "k": ("key_input", kv, "attn.wk_dv.weight"),
        "v": ("value_input", kv, "attn.wv_dv.weight"),
        "relative": ("query_input", relative, "attn.wr_du.weight"),
        "sink": ("query_input", heads, "attn.attn_sink_proj.weight"),
    }
    shapes = {key: [tokens, width] for key in ("query_input", "key_input", "value_input")}
    ops = []
    for name, (source, channels, weight) in bindings.items():
        ops.append(_project(name, "external:" + source, width, channels, tokens, weight))
        shapes[name] = [tokens, channels]
    return _graph("attention_projections", tokens, width, shapes, ops)


def controller_gates(tokens=8, width=2304, rank=64):
    """Query-controller forget and gain gates from the supplied bottleneck."""
    shapes = {"bottleneck": [tokens, rank]}
    ops = []
    for gate in ("forget", "gain"):
        projection = gate + "_logits"
        ops.append(_project(projection, "external:bottleneck", rank, width, tokens,
                            f"controllers.attn.{gate}_weight"))
        shapes[projection] = shapes[gate] = [tokens, width]
        ops.append({
            "id": gate, "op": "tanh_strided_checked", "kernel": "tanh_strided_f32_checked",
            "returns_status": True,
            "graph_slots": {"inputs": {"input": projection}, "outputs": {"output": gate}},
            "params": {"R": tokens, "C": width, "call_constants": {
                "tanh_input_elements": tokens * width, "tanh_input_stride": width,
                "tanh_output_elements": tokens * width, "tanh_output_stride": width,
                "tanh_rows": tokens, "tanh_columns": width,
            }},
        })
    return _graph("controller_gates", tokens, width, shapes, ops)


def causal_kv_replay(prefill_tokens=8, width=2304, kv_width=1024, taps=4,
                     slots=(6, 24, 36), kv_heads=8, key_clip=22.627416997969522):
    """K/V projections with independent caller-owned convolution histories."""
    if type(taps) is not int or not 2 <= taps <= 256:
        raise ValueError("unsupported convolution tap count")
    if not slots or len(set(slots)) != len(slots) or any(type(s) is not int or s < 0 for s in slots):
        raise ValueError("each pass needs an independent state slot")

    if type(kv_heads) is not int or kv_heads < 1 or kv_width % kv_heads:
        raise ValueError("K/V widths must partition into whole heads")
    dim = kv_width // kv_heads
    capacity = prefill_tokens + 1
    ops, shapes, segments, state_buffers = [], {}, [], []
    for slot in slots:
        for kind in ("k", "v"):
            history = f"history_{kind}_{slot}"
            shapes[history] = [taps - 1, kv_width]
            cache = f"cache_{kind}_{slot}"
            shapes[cache] = [kv_heads, capacity, dim]
            state_buffers.extend((history, cache))
        for phase, rows in (("prefill", prefill_tokens), ("append", 1)):
            entry = f"inkling2_{phase}_slot{slot}"
            for kind in ("k", "v"):
                source, projection, output = (f"{entry}_{kind}_{suffix}" for suffix in
                                             ("input", "projection", "output"))
                history = f"history_{kind}_{slot}"
                shapes[source] = [rows, width]
                shapes[projection] = shapes[output] = [rows, kv_width]
                ops.append(_project(projection, "external:" + source, width, kv_width, rows,
                                    f"attn.w{kind}_dv.weight"))
                ops.append({
                    "id": output, "op": "causal_depthwise_residual_checked",
                    "kernel": "causal_depthwise_residual_checked_f32", "returns_status": True,
                    "graph_slots": {
                        "inputs": {"input": projection, "history": "external:" + history},
                        "outputs": {"output": output, "history": "external:" + history},
                    },
                    "weight_refs": {"weight": LAYER + f"attn.{kind}_sconv.weight"},
                    "params": {
                        "T": rows, "C": kv_width, "K": taps, "H": taps - 1, "N": kv_width,
                        "call_constants": {
                            "input_elements": rows * kv_width, "weight_elements": taps * kv_width,
                            "history_elements": (taps - 1) * kv_width, "output_elements": rows * kv_width,
                            "tokens": rows, "channels": kv_width, "taps": taps, "history_rows": taps - 1,
                        },
                    },
                })
            commit = f"{entry}_commit_kv"
            cache_bindings = {f"cache_{kind}": f"external:cache_{kind}_{slot}" for kind in ("k", "v")}
            ops.append({
                "id": commit, "op": "kv_cache_l2_rms_append_checked", "kernel": "kv_cache_l2_rms_append_checked_f32",
                "returns_status": True,
                "graph_slots": {
                    "inputs": {"k": f"{entry}_k_output", "v": f"{entry}_v_output", **cache_bindings},
                    "outputs": cache_bindings,
                },
                "params": {"T": rows, "KV": kv_heads, "D": dim, "S": capacity,
                           "key_clip": key_clip, "epsilon": 1e-6, "call_constants": {
                    "key_elements": rows * kv_width, "value_elements": rows * kv_width,
                    "k_cache_elements": capacity * kv_width, "v_cache_elements": capacity * kv_width,
                    "tokens": rows, "heads": kv_heads, "dim": dim, "capacity": capacity,
                    "position": 0 if phase == "prefill" else prefill_tokens,
                }},
            })
            segments.append({"function": f"ck_{entry}", "last_op": commit})
    graph = _graph("causal_kv", prefill_tokens, width, shapes, ops)
    scope = graph["contract"]["runtime_invariants"]
    scope["scope"] = "supplied K/V input rows through FP32 projection, static convolution, normalization and KV storage"
    scope["state_slots"] = list(slots)
    graph["native_entry"]["segments"] = segments
    for buffer in state_buffers:
        graph["activation_buffers"][buffer]["lifetime"] = "sequence"
    exports = graph["semantic_checkpoints"]["exports"]
    for segment in segments:
        # Cache writes use the explicit state buffers, rather than a tensor
        # named after the operation. Compare their logical extents in replay.
        del exports[segment["last_op"]]
    return graph


def relative_attention_replay(prefill_tokens=8, width=2304, kv_width=1024, taps=4,
                              slots=(6, 24, 36), kv_heads=8, query_heads=24,
                              band=1024, window=1024):
    """Read the generated cache through causal GQA with relative logits and sinks."""
    graph = causal_kv_replay(prefill_tokens=prefill_tokens, width=width, kv_width=kv_width,
                             taps=taps, slots=slots, kv_heads=kv_heads)
    if (any(type(value) is not int for value in (query_heads, band, window))
            or not 1 <= query_heads <= 256 or query_heads % kv_heads
            or not 1 <= band <= 16384 or not 0 <= window <= 16384):
        raise ValueError("unsupported attention geometry")
    dim, capacity = kv_width // kv_heads, prefill_tokens + 1
    ops, shapes, new_ops = graph["block_types"]["block"]["body"]["ops"], graph["activation_buffers"], {}
    for slot in slots:
        for phase, tokens in (("prefill", prefill_tokens), ("append", 1)):
            prefix = f"inkling2_{phase}_slot{slot}"
            output = prefix + "_attention"
            q, bias, sink = (prefix + "_" + port for port in ("query", "relative_band", "sink"))
            shapes.update({
                q: {"shape": [tokens, query_heads, dim]},
                bias: {"shape": [tokens, query_heads, band]},
                sink: {"shape": [tokens, query_heads]},
                output: {"shape": [tokens, query_heads, dim]},
            })
            for name in (q, bias, sink, output):
                graph["activation_bindings"][name] = name
            new_ops[prefix + "_commit_kv"] = {
                "id": output, "op": "gqa_relative_sink_checked", "kernel": "gqa_relative_sink_checked_f32",
                "returns_status": True,
                "graph_slots": {
                    "inputs": {
                        "query": "external:" + q, "key": f"external:cache_k_{slot}",
                        "value": f"external:cache_v_{slot}", "relative": "external:" + bias,
                        "sink": "external:" + sink,
                    },
                    "outputs": {"output": output},
                },
                "params": {
                    "T": tokens, "H": query_heads, "KV": kv_heads, "D": dim, "S": capacity, "B": band,
                    "call_constants": {
                        "query_elements": tokens * query_heads * dim,
                        "key_elements": capacity * kv_width, "value_elements": capacity * kv_width,
                        "relative_elements": tokens * query_heads * band, "sink_elements": tokens * query_heads,
                        "output_elements": tokens * query_heads * dim, "work_elements": dim,
                        "tokens": tokens, "heads": query_heads, "kv_heads": kv_heads,
                        "dim": dim, "capacity": capacity, "position": 0 if phase == "prefill" else prefill_tokens,
                        "relative_extent": band, "window": window,
                    },
                },
            }

    for segment in graph["native_entry"]["segments"]:
        segment["last_op"] = new_ops[segment["last_op"]]["id"]
    graph["block_types"]["block"]["body"]["ops"] = [
        child for op in ops for child in ([op, new_ops[op["id"]]] if op["id"] in new_ops else [op])
    ]
    for op in new_ops.values():
        name = op["id"]
        graph["semantic_checkpoints"]["exports"][name] = {
            "section": "body", "template_op_id": name, "op": op["op"],
            "checkpoints": [{
                "id": f"inkling2.layer6.{name}", "producer": name, "tensor": name,
                "logical_layout": "token_major_contiguous",
                "axis_names": ["token", "head", "channel"], "storage_dtype": "fp32",
            }],
        }
    graph["name"] = "inkling2_mini_layer6_relative_attention_fp32"
    graph["native_entry"]["function"] = "ck_inkling2_mini_layer6_relative_attention_fp32"
    graph["contract"]["runtime_invariants"]["scope"] = (
        "generated K/V projection and static convolution through attention over stored cache; supplied Q, bias and sinks"
    )
    return graph


def selected_expert(tokens=3, width=2304, inner=1152, *, layer=6, expert=7,
                    gate_cap=4.0, up_cap=25.0):
    """Generated FP32 path for one preselected expert with expanded weights."""
    if type(layer) is not int or not 0 <= layer < 24 or type(expert) is not int or not 0 <= expert < 128:
        raise ValueError("invalid checkpoint layer or expert")
    sizes = {"hidden": width, "gate": inner, "up": inner, "activated": inner, "down": width}
    shapes = {name: [tokens, columns] for name, columns in sizes.items()}
    prefix = f"model.llm.layers.{layer}.mlp.experts.expanded.{expert}."
    ops = [_project(name, "external:hidden", width, inner, tokens, name, weight_prefix=prefix)
           for name in ("gate", "up")]
    elements = tokens * inner
    ops.append({
        "id": "activated", "op": "softcapped_swiglu_rows_checked",
        "kernel": "softcapped_swiglu_rows_checked_f32", "returns_status": True,
        "graph_slots": {"inputs": {"gate": "gate", "up": "up"}, "outputs": {"output": "activated"}},
        "params": {"N": elements, "gate_cap": gate_cap, "up_cap": up_cap, "call_constants": {
            "gate_elements": elements, "up_elements": elements, "output_elements": elements,
            "elements": elements,
        }},
    })
    ops.append(_project("down", "activated", inner, width, tokens, "down", weight_prefix=prefix))
    return _graph("selected_expert", tokens, width, shapes, ops, layer=layer,
                  scope="one preselected expert, expanded checkpoint weights, FP32 operands and outputs")


def expert_router(tokens=4, width=2304, experts=128, top_k=8, *, layer=6, route_scale=8.0):
    """Checked no-group routing from synthetic or captured post-normalization rows."""
    if type(experts) is not int or type(top_k) is not int or not 1 <= top_k <= experts <= 1024:
        raise ValueError("invalid router dimensions")
    prefix = f"model.llm.layers.{layer}.mlp.gate."
    shapes = {"hidden": [tokens, width], "logits": [tokens, experts],
              "correction_bias": [experts], "indices": [tokens, top_k],
              "route_weights": [tokens, top_k]}
    ops = [_project("logits", "external:hidden", width, experts, tokens, "weight", weight_prefix=prefix)]
    ops.append({
        "id": "indices", "op": "sigmoid_topk_router_checked",
        "kernel": "sigmoid_topk_router_checked_f32", "returns_status": True,
        "graph_slots": {"inputs": {"logits": "logits", "bias": "external:correction_bias"},
                        "outputs": {"indices": "indices", "weights": "route_weights"}},
        "params": {"R": tokens, "E": experts, "K": top_k, "route_scale": route_scale,
                   "call_constants": {
                       "logits_elements": tokens * experts, "bias_elements": experts,
                       "indices_elements": tokens * top_k, "weights_elements": tokens * top_k,
                       "rows": tokens, "experts": experts, "top_k": top_k,
                   }},
    })
    graph = _graph("expert_router", tokens, width, shapes, ops, layer=layer,
                   scope="ungrouped sigmoid router; FP32 projection and scores; no expert dispatch")
    checkpoints = graph["semantic_checkpoints"]["exports"]["indices"]["checkpoints"]
    checkpoints[0].update(id=f"inkling2.layer{layer}.expert_router.weights",
                          tensor="route_weights", axis_names=["token", "route"])
    return graph


def routed_expert_mixture(tokens, width, inner, experts, top_k, slots, *,
                          layer=6, storage="f32", route_scale=8.0, gate_cap=4.0, up_cap=25.0):
    """Connect generated routing, checked expert dispatch and weighted mixture."""
    if (type(slots) is not int or not 1 <= slots <= experts
            or storage not in {"f32", "q8_0", "iq4_nl"}
            or storage != "f32" and (width % 32 or inner % 32)):
        raise ValueError("unsupported routed expert bank")
    graph = expert_router(tokens, width, experts, top_k, layer=layer, route_scale=route_scale)
    graph["name"] = f"inkling2_mini_layer{layer}_routed_experts_{storage}"
    graph["native_entry"]["function"] = "ck_" + graph["name"]
    graph["contract"]["runtime_invariants"].update(
        inner=inner, scope="checked routed MoE with caller-provided expert bank"
    )
    graph["activation_buffers"].update(
        expert_slots={"shape": [experts]},
        expert_output={"shape": [tokens * top_k, width]},
        mixture={"shape": [tokens, width]},
    )
    graph["activation_bindings"].update({p: p for p in ("expert_slots", "expert_output", "mixture")})
    work = 3 * inner + tokens * (top_k + 1) * width
    elements = slots * inner * width
    block_bytes = {"q8_0": 34, "iq4_nl": 18}
    prefix = f"model.llm.layers.{layer}.mlp.experts.bank."
    mixture = {
        "id": "mixture", "op": "routed_softcapped_experts_checked",
        "kernel": f"routed_softcapped_experts_checked_{storage}", "returns_status": True,
        "graph_slots": {
            "inputs": {"hidden": "external:hidden", "ids": "indices",
                       "route_weights": "route_weights", "expert_slots": "external:expert_slots"},
            "outputs": {"expert_output": "expert_output", "output": "mixture"},
        },
        "weight_refs": {port: prefix + port for port in ("gate", "up", "down")},
        "params": {
            "R": tokens, "H": width, "I": inner, "E": experts, "S": slots, "K": top_k,
            "P": tokens * top_k, "W": work, "gate_cap": gate_cap, "up_cap": up_cap,
            "call_constants": {
                "rows": tokens, "hidden_dim": width, "inner": inner, "experts": experts,
                "slots": slots, "top_k": top_k,
                "matrix_bytes": 4 * elements if storage == "f32" else elements // 32 * block_bytes[storage],
                "hidden_elements": tokens * width, "id_elements": tokens * top_k,
                "route_elements": tokens * top_k, "expert_slot_elements": experts,
                "expert_output_elements": tokens * top_k * width,
                "output_capacity": tokens * width, "work_elements": work,
            },
        },
    }
    graph["block_types"]["block"]["body"]["ops"].append(mixture)
    graph["semantic_checkpoints"]["exports"]["mixture"] = {
        "section": "body", "template_op_id": "mixture", "op": mixture["op"],
        "checkpoints": [
            {"id": f"inkling2.layer{layer}.routed_experts.{name}", "producer": "mixture",
             "tensor": name, "logical_layout": "token_major_contiguous",
             "axis_names": [axis, "channel"], "storage_dtype": "fp32"}
            for name, axis in (("expert_output", "route"), ("mixture", "token"))
        ],
    }
    return graph
