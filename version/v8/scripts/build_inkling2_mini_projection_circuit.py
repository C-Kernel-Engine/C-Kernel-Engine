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
