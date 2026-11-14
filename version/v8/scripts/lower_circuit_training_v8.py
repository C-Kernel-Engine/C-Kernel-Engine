"""Bounded checked-circuit adapter for the shared native training backend.

This is operation-level lowering, independent of model names and topology.
Unsupported providers fail closed. Existing gradient rules own derivatives;
this adapter still supplies provider-specific ABI/save bindings explicitly.
"""

from copy import deepcopy
import math
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def lower_circuit_training(
    graph, weights, *, inputs, bindings, frozen, config, loss, activation_dtypes=None
):
    """Lower declared buffers/edges and explicit input/parameter ownership to IR1.

    `inputs` describes runtime inputs; `bindings` maps circuit external names to
    those inputs or frozen weights. Neither supplies model arithmetic.
    """
    from version.v8.scripts.native_entry_binding_v8 import resolve_entry_argument

    provider_maps = {m["id"]: m for m in training_registry()["kernels"]}
    native_contexts = []
    activation_dtypes = activation_dtypes or {}
    frozen = set(frozen)
    if frozen - set(weights):
        raise ValueError("unknown frozen parameter")
    for name, value in weights.items():
        if str(value.dtype) != "float32":
            raise ValueError("circuit adapter requires FP32 weights: " + name)
    for name, spec in inputs.items():
        if spec.get("dtype") not in ("i32", "fp32"):
            raise ValueError("runtime input binding supports i32 or fp32 only: " + name)
        if not spec.get("shape") or any(
            type(d) is not int or d <= 0 for d in spec["shape"]
        ):
            raise ValueError("invalid input extent: " + name)
    tensors = {}
    ops = []
    integer_inputs = []
    float_inputs = []

    def ref(tid, shape, kind="activation", dtype="fp32", grad=True):
        if tid in tensors:
            return dict(tensors[tid], tensor=tid)
        m = dict(
            shape=list(shape),
            numel=math.prod(shape),
            kind=kind,
            dtype=dtype,
            requires_grad=grad,
            persistent=kind == "weight",
        )
        tensors[tid] = m
        return dict(m, tensor=tid)

    for name, a in weights.items():
        ref("weight." + name, list(a.shape) or [1], "weight", grad=name not in frozen)

    def weight(n):
        return dict(ref("weight." + n, weights[n].shape), name=n)

    for name, spec in inputs.items():
        ref("input." + name, spec["shape"], "input", spec["dtype"], False)
        if spec["dtype"] == "i32":
            integer_inputs.append("input." + name)
        else:
            float_inputs.append("input." + name)
    integer_inputs.sort()

    def act(n):
        n = n.removeprefix("external:")
        if n in bindings:
            target = bindings[n]
            if target.startswith("weight."):
                return weight(target.removeprefix("weight."))
            if target not in tensors:
                raise ValueError("undeclared input binding: " + target)
            return dict(tensors[target], tensor=target)
        dtype = activation_dtypes.get(n, "fp32")
        return ref(
            "act." + n,
            graph["activation_buffers"][n]["shape"],
            dtype=dtype,
            grad=dtype != "i32",
        )

    def emit(
        name,
        kernel,
        inputs,
        output,
        constants,
        rule=None,
        params=None,
        ws=None,
        saved=None,
        scratch=0,
    ):
        outputs = dict(output=output)
        if scratch:
            outputs["scratch"] = ref("aux." + name, [scratch], grad=False)
        o = dict(
            op_id=len(ops),
            op=name,
            kernel_id=kernel,
            phase="forward",
            section="body",
            layer=0,
            requires_grad=rule is not None,
            map_call=True,
            weights=ws or {},
            params=dict(params or {}, call_constants=constants),
            dataflow=dict(inputs=inputs, outputs=outputs),
            save_for_backward=saved or inputs,
        )
        if rule:
            o["grad_rule"] = rule
        ops.append(o)
        return o

    def gather(name, table, ids, out, rows, capacity, cols, as_weight=False):
        const = dict(
            table_elements=capacity * cols,
            id_elements=rows,
            output_elements=rows * cols,
            rows=rows,
            capacity=capacity,
            channels=cols,
            d_output_elements=rows * cols,
            d_table_elements=capacity * cols,
            scratch_elements=capacity * cols,
        )
        return emit(
            name,
            "indexed_rows_weight_train_f32" if as_weight else "indexed_rows_train_f32",
            dict(ids=ids) if as_weight else dict(table=table, ids=ids),
            out,
            const,
            "indexed_rows_weight_backward" if as_weight else "indexed_rows_backward",
            ws=dict(table=table) if as_weight else {},
            saved=dict(ids=ids),
        )

    for section in graph["sequence"]:
        block = graph["block_types"][section]
        for item in (
            block.get("header", [])
            + block.get("body", {}).get("ops", [])
            + block.get("footer", [])
        ):
            name = item["id"]
            kernel = item["kernel"]
            p = deepcopy(item["params"])
            c = p.pop("call_constants", {})
            inputs = {k: act(v) for k, v in item["graph_slots"]["inputs"].items()}
            outnames = item["graph_slots"]["outputs"]
            out = act(next(iter(outnames.values())))
            ws = {k: weight(v) for k, v in item.get("weight_refs", {}).items()}
            if kernel == "expand_group_row_indices_checked_i32":
                o = emit(name, kernel, inputs, out, c)
                o["dataflow"]["outputs"] = {k: act(v) for k, v in outnames.items()}
                continue
            provider = provider_maps.get(kernel, {})
            contract = provider.get("training_contract", {})
            if contract.get("frozen_forward") is not None:
                if (
                    contract["frozen_forward"]
                    != {
                        "inputs": "nondifferentiable",
                        "outputs": "stop_gradient",
                        "parameters": "external_frozen",
                    }
                    or p.pop("stop_gradient", None) is not True
                    or provider.get("modes", {}).get("training") is not True
                    or ws
                    or any(x.get("requires_grad") is not False for x in inputs.values())
                    or set(outnames)
                    != {port["name"] for port in provider.get("outputs", [])}
                    or len(outnames) != 1
                    or out.get("dtype") != "fp32"
                ):
                    raise ValueError(
                        "frozen component requires explicit nondifferentiable boundary"
                    )
                entry = graph.get("native_entry", {})
                abi = {
                    a["name"]: a for a in provider.get("call_abi", {}).get("params", [])
                }
                declared_contexts = contract.get("borrowed_contexts")
                if not isinstance(declared_contexts, list) or not declared_contexts:
                    raise ValueError(
                        "frozen component requires borrowed context declarations"
                    )
                for context in declared_contexts:
                    if not isinstance(context, dict) or set(context) != {
                        "pointer",
                        "bytes",
                    }:
                        raise ValueError("invalid frozen component context pair")
                    names = {}
                    for field, expected in (
                        ("pointer", "constvoid*"),
                        ("bytes", "size_t"),
                    ):
                        argument = abi.get(context[field])
                        if argument is None:
                            raise ValueError("missing native context argument")
                        bound_name = resolve_entry_argument(entry, provider, argument)
                        parameter = next(
                            p for p in entry["params"] if p["name"] == bound_name
                        )
                        if parameter["c_type"].replace(" ", "") != expected:
                            raise ValueError("invalid borrowed native context type")
                        names[field] = bound_name
                    if names not in native_contexts:
                        native_contexts.append(names)
                tensors[out["tensor"]]["requires_grad"] = False
                out["requires_grad"] = False
                operation = emit(name, kernel, inputs, out, c, params=p)
                symbols = provider.get("call_constant_contract", {}).get(
                    "shape_symbols", {}
                )
                for scratch in provider.get("scratch", []):
                    shape = [
                        c.get(symbols.get(dim, "")) if isinstance(dim, str) else dim
                        for dim in scratch.get("shape", [])
                    ]
                    port = scratch.get("name")
                    if (
                        scratch.get("dtype") != "fp32"
                        or not shape
                        or any(type(d) is not int or d <= 0 for d in shape)
                        or math.prod(shape) > 2147483647
                        or not isinstance(port, str)
                        or port in operation["dataflow"]["outputs"]
                    ):
                        raise ValueError("invalid frozen component scratch contract")
                    operation["dataflow"]["outputs"][port] = ref(
                        "aux." + name + "." + port, shape, grad=False
                    )
                continue
            if kernel.startswith("linear_rows_checked"):
                rows = c["linear_rows"]
                ic = c["linear_input_channels"]
                oc = c["linear_output_channels"]
                c.update(T=rows, aligned_in=ic, aligned_out=oc, num_threads=1)
                emit(
                    name,
                    kernel,
                    inputs,
                    out,
                    c,
                    (
                        "linear_rows_train_backward"
                        if "bias" in ws
                        else "linear_rows_train_backward_nobias"
                    ),
                    ws=ws,
                    saved=dict(input=inputs["input"], weight=ws["weight"]),
                )
                continue
            if kernel == "layernorm_rows_checked_param_epsilon_f32":
                rows = c["norm_rows"]
                cols = c["norm_channels"]
                saved = dict(input=inputs["input"], gamma=ws["gamma"])
                saved.update(
                    mean_cache=ref("aux." + name + ".mean", [rows], grad=False),
                    rstd_cache=ref("aux." + name + ".rstd", [rows], grad=False),
                )
                o = emit(
                    name,
                    "layernorm_train_forward_f32",
                    inputs,
                    out,
                    dict(rows=rows, channels=cols),
                    "layernorm_cached_backward",
                    params=dict(epsilon=p["epsilon"]),
                    ws=ws,
                    saved=saved,
                )
                o["dataflow"]["outputs"].update(
                    mean_cache=saved["mean_cache"], rstd_cache=saved["rstd_cache"]
                )
                continue
            if kernel == "audio_text_embedding_token_major_f32":
                gather(
                    name,
                    ws["table"],
                    inputs["ids"],
                    out,
                    c["tokens"],
                    c["vocabulary"],
                    c["channels"],
                    True,
                )
                continue
            if kernel == "indexed_sum_rows_checked_f32":
                rows = out["shape"][0]
                cols = out["shape"][1]
                gathers = []
                for side in ("left", "right"):
                    table = inputs[side]
                    g = ref("act." + name + "." + side, [rows, cols])
                    gather(
                        name + "." + side,
                        table,
                        inputs[side + "_indices"],
                        g,
                        rows,
                        table["shape"][0],
                        cols,
                    )
                    gathers.append(g)
                emit(
                    name,
                    "add_rows_checked_f32",
                    dict(left=gathers[0], right=gathers[1]),
                    out,
                    dict(
                        left_elements=rows * cols,
                        right_elements=rows * cols,
                        output_elements=rows * cols,
                        elements=rows * cols,
                    ),
                    "add_rows_checked_backward",
                )
                continue
            if kernel == "broadcast_add_rows_checked_f32":
                rows, cols = out["shape"]
                if act("broadcast_zeros")["shape"] != [rows]:
                    raise ValueError("broadcast index capacity differs")
                g = ref("act." + name + ".broadcast", [rows, cols])
                gather(
                    name + ".broadcast",
                    inputs["right"],
                    act("broadcast_zeros"),
                    g,
                    rows,
                    1,
                    cols,
                )
                emit(
                    name,
                    "add_rows_checked_f32",
                    dict(left=inputs["left"], right=g),
                    out,
                    dict(
                        left_elements=rows * cols,
                        right_elements=rows * cols,
                        output_elements=rows * cols,
                        elements=rows * cols,
                    ),
                    "add_rows_checked_backward",
                )
                continue
            if kernel in ("token_span_mean_checked_f32", "last_valid_row_checked_f32"):
                rows = out["shape"][0]
                cols = c["channels"]
                capacity = c["token_capacity"]
                c.update(
                    span_count=rows,
                    d_output_elements=rows * cols,
                    d_output_stride=cols,
                    d_input_elements=capacity * cols,
                    d_input_stride=cols,
                    scratch_elements=capacity * cols,
                )
                if kernel == "last_valid_row_checked_f32":
                    saved = dict(spans=inputs["span"])
                    c["output_stride"] = cols
                else:
                    saved = dict(spans=inputs["spans"])
                emit(
                    name,
                    kernel,
                    inputs,
                    out,
                    c,
                    "token_span_mean_checked_backward",
                    saved=saved,
                )
                continue
            if kernel == "gelu_rows_erf_checked_f32":
                rows = c["gelu_rows"]
                cols = c["gelu_channels"]
                c.update(
                    input_elements=rows * cols,
                    input_stride=cols,
                    d_output_elements=rows * cols,
                    d_output_stride=cols,
                    d_input_elements=rows * cols,
                    d_input_stride=cols,
                    scratch_elements=rows * cols,
                    rows=rows,
                    channels=cols,
                )
                emit(
                    name,
                    kernel,
                    inputs,
                    out,
                    c,
                    "gelu_erf_checked_backward",
                    scratch=rows * cols,
                )
                continue
            if kernel == "attention_cross_token_major_checked_f32":
                q, s, h, d = (
                    c["query_tokens"],
                    c["memory_tokens"],
                    c["heads"],
                    c["head_dim"],
                )
                c["backward_scratch_elements"] = q * h * d + 2 * s * h * d + 2 * s
                emit(
                    name,
                    kernel,
                    inputs,
                    out,
                    c,
                    "attention_cross_backward",
                    scratch=c["scratch_elements"],
                )
                continue
            if kernel == "grouped_softmax_weighted_rows_checked_f32":
                groups, rows, cols = c["groups"], c["option_rows"], c["channels"]
                # The shared forward/backward scratch bound covers both providers.
                c.update(
                    d_output_elements=groups * cols,
                    d_output_stride=cols,
                    d_options_elements=rows * cols,
                    d_options_stride=cols,
                    d_fields_elements=groups * cols,
                    d_fields_stride=cols,
                )
                c["scratch_elements"] = max(
                    c["scratch_elements"], (rows + 3 * groups) * cols + 2 * rows
                )
                emit(
                    name,
                    kernel,
                    inputs,
                    out,
                    c,
                    "grouped_weighted_checked_backward",
                    scratch=c["scratch_elements"],
                )
                continue
            if kernel in (
                "indexed_cosine_rows_checked_f32",
                "indexed_pair_features_checked_f32",
            ):
                if kernel == "indexed_cosine_rows_checked_f32" and item.get(
                    "training", {}
                ).get("vjp") not in ("normalized_dot", "cosine_similarity"):
                    raise ValueError("indexed cosine requires an explicit training VJP")
                groups, rows, cols = c["groups"], c["rows"], c["channels"]
                c.update(
                    d_output_elements=c["output_elements"],
                    d_left_elements=groups * cols,
                    d_right_elements=rows * cols,
                )
                c["scratch_elements"] = max(
                    c["scratch_elements"], (groups + 2 * rows) * cols
                )
                rule = (
                    "indexed_pair_features_checked_backward"
                    if "pair" in kernel
                    else (
                        "indexed_normalized_dot_checked_backward"
                        if item.get("training", {}).get("vjp") == "normalized_dot"
                        else "indexed_cosine_rows_checked_backward"
                    )
                )
                emit(
                    name,
                    kernel,
                    inputs,
                    out,
                    c,
                    rule,
                    params=p,
                    scratch=c["scratch_elements"],
                )
                continue
            if kernel.startswith("scalar_"):
                c.update(d_output_elements=1, d_input_elements=1)
                emit(
                    name,
                    kernel,
                    inputs,
                    out,
                    c,
                    kernel.removesuffix("_f32") + "_backward",
                    params=p,
                    ws=ws,
                )
                continue
            if kernel == "scale_vector_checked_f32":
                rows = c["rows"]
                c.update(
                    d_output_elements=rows,
                    d_input_elements=rows,
                    d_gain_elements=1,
                    scratch_elements=rows + 1,
                    backward_scratch_elements=rows + 1,
                )
                emit(
                    name,
                    kernel,
                    inputs,
                    out,
                    c,
                    "scale_vector_checked_backward",
                    scratch=rows + 1,
                )
                continue
            if kernel == "add_rows_checked_f32":
                emit(name, kernel, inputs, out, c, "add_rows_checked_backward")
                continue
            raise ValueError("unsupported circuit training provider: " + kernel)
    return dict(
        format="ir1-train-v7",
        version=1,
        num_layers=0,
        config=dict(
            config,
            training_integer_inputs=integer_inputs,
            training_float_inputs=sorted(float_inputs),
            training_native_contexts=native_contexts,
        ),
        ops=ops,
        tensors=tensors,
        loss=deepcopy(loss),
    )


def training_registry():
    registry = json.loads(
        (ROOT / "version/v7/kernel_maps/KERNEL_REGISTRY.json").read_text()
    )
    maps = {k["id"]: k for k in registry["kernels"]}
    for path in (ROOT / "version/v8/kernel_maps").glob("*.json"):
        m = json.loads(path.read_text())
        if "id" in m:
            maps[m["id"]] = m
    registry["kernels"] = list(maps.values())
    return registry
