"""Bounded Inkling layer-6 projection replay circuits using checked providers."""

from version.v8.scripts.checked_circuit_authoring_v8 import linear

SOURCE_REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"
TRANSFORMERS_REVISION = "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3"
LAYER = "model.llm.layers.6."


def _project(name, source, width, channels, tokens, weight):
    op = linear(name, source, name, width, channels, tokens, "unused")
    op.update(op="linear_rows_checked_no_bias", kernel="linear_rows_checked_no_bias_f32",
              weight_refs={"weight": LAYER + weight})
    del op["params"]["call_constants"]["linear_bias_elements"]
    return op


def _graph(kind, tokens, width, shapes, ops):
    if type(tokens) is not int or not 1 <= tokens <= 16:
        raise ValueError("expected 1..16 fixed token rows")
    if any(type(n) is not int or not 1 <= n <= 8192 for shape in shapes.values() for n in shape):
        raise ValueError("unsupported projection geometry")
    name = f"inkling2_mini_layer6_{kind}_fp32"
    checks = {}
    for op in ops:
        operation = op["id"]
        checks[operation] = {
            "section": "body", "template_op_id": operation, "op": op["op"],
            "checkpoints": [{
                "id": f"inkling2.layer6.{kind}.{operation}", "producer": operation,
                "tensor": operation, "logical_layout": "token_major_contiguous",
                "axis_names": ["token", "channel"], "storage_dtype": "fp32",
            }],
        }
    return {
        "version": 3, "name": name, "family": "bounded_graph", "checked_native_entry": True,
        "contract": {
            "runtime_invariants": {
                "inference_only": True, "tokens": tokens, "width": width, "heads": 1, "mult": 1,
                "scope": "supplied projection operands, FP32 output, isolated layer-6 component",
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
