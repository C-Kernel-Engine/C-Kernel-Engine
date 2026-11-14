"""Bounded checked-circuit adapter for the shared native training backend.

This is operation-level lowering, independent of model names and topology.
Unsupported providers fail closed. Existing gradient rules own derivatives;
direct FP32 projections/additions declare ABI/save bindings in their maps.
Other operation groups retain explicit bounded translation rules.
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
    from version.v8.scripts.map_training_binding_v8 import resolve_training_binding

    provider_maps = {m["id"]: m for m in training_registry()["kernels"]}
    gradient_rules = json.loads(
        (ROOT / "version/v8/scripts/grad_rules_v8.json").read_text()
    )["rules"]
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
    integer_constants = {}

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
            save_for_backward=saved if saved is not None else inputs,
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
            if kernel not in provider_maps:
                raise ValueError("unsupported circuit training provider: " + kernel)
            binding = resolve_training_binding(
                provider_maps[kernel],
                gradient_rules,
                inputs=inputs,
                weights=ws,
                output=out,
                constants=c,
            )
            if binding is not None:
                if set(outnames) != {
                    port["name"] for port in provider_maps[kernel].get("outputs", [])
                }:
                    raise ValueError("training output ports disagree with provider")
                rule, bound_constants, saved = binding
                emit(
                    name, kernel, inputs, out, bound_constants, rule, ws=ws, saved=saved
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
                constant_id = "const." + name + ".broadcast_rows"
                if constant_id in tensors:
                    raise ValueError("duplicate compiler-owned broadcast constant")
                indices = ref(constant_id, [rows], kind="aux", dtype="i32", grad=False)
                tensors[constant_id]["persistent"] = True
                indices["persistent"] = True
                integer_constants[constant_id] = [0] * rows
                g = ref("act." + name + ".broadcast", [rows, cols])
                gather(
                    name + ".broadcast",
                    inputs["right"],
                    indices,
                    g,
                    rows,
                    1,
                    cols,
                    as_weight=inputs["right"].get("kind") == "weight",
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
            raise ValueError("unsupported circuit training provider: " + kernel)
    return dict(
        format="ir1-train-v7",
        version=1,
        num_layers=0,
        config=dict(
            config,
            training_integer_inputs=integer_inputs,
            training_integer_constants=integer_constants,
            training_float_inputs=sorted(float_inputs),
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
