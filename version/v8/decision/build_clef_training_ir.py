"""Bounded FP32 head-only training authoring from the canonical Clef circuit.

This is an experimental fixed-feature training graph, not a public job recipe.
No forward/backward arithmetic or layer schedule is executed in Python.
"""

from copy import deepcopy
import json
import math
import numpy as np
from pathlib import Path
from .package_clef_head import build_circuit, effective_weights

ROOT = Path(__file__).resolve().parents[3]


def build_training_ir(
    raw, hidden, *, questions, options, heads, feedforward, frozen=()
):
    if (
        not isinstance(hidden, np.ndarray)
        or hidden.ndim != 2
        or hidden.dtype != np.float32
    ):
        raise ValueError("frozen hidden features must be a rank-two FP32 array")
    if not np.isfinite(hidden).all():
        raise ValueError("frozen hidden features must be finite")
    for name, value in (
        ("questions", questions),
        ("options", options),
        ("heads", heads),
        ("feedforward", feedforward),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(name + " must be a positive integer")
    tokens, channels = hidden.shape
    width = raw["question_projection.weight"].shape[0]
    graph = build_circuit(
        tokens=tokens,
        questions=questions,
        options=options,
        channels=channels,
        width=width,
        vocabulary=len(raw["output_embedding.weight"]),
        heads=heads,
        feedforward=feedforward,
    )
    weights = effective_weights(graph, raw)
    weights["__features.hidden"] = hidden.copy()
    frozen = set(frozen) | {"output_embedding.weight", "__features.hidden"}
    if frozen - set(weights):
        raise ValueError("unknown frozen parameter")
    tensors = {}
    ops = []
    integer_inputs = []

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

    integers = {
        "input_ids",
        "question_spans",
        "option_spans",
        "global_span",
        "option_counts",
        "question_type_ids",
    }
    for n in integers:
        ref(
            "input." + n, graph["activation_buffers"][n]["shape"], "input", "i32", False
        )
        integer_inputs.append("input." + n)
    ref("input.labels", [questions], "input", "i32", False)
    integer_inputs.append("input.labels")
    ref("input.broadcast_zeros", [questions], "input", "i32", False)
    integer_inputs.append("input.broadcast_zeros")
    integer_inputs.sort()

    def act(n):
        n = n.removeprefix("external:")
        if n == "hidden":
            return weight("__features.hidden")
        if n in integers or n in ("labels", "broadcast_zeros"):
            return ref("input." + n, [])
        dtype = (
            "i32" if n in ("option_row_indices", "option_question_indices") else "fp32"
        )
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
                        if name == "prior_cosine"
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
            raise ValueError("unsupported head training provider: " + kernel)
    loss = dict(
        kernel_id="grouped_cross_entropy_checked_f32",
        inputs=dict(
            logits="act.option_logits",
            counts="input.option_counts",
            targets="input.labels",
        ),
        params=dict(
            call_constants=dict(
                logits_elements=options,
                count_elements=questions,
                target_elements=questions,
                gradient_elements=options,
                loss_elements=1,
                scratch_elements=options + 1,
                groups=questions,
                options=options,
            )
        ),
        scratch=dict(scratch=options + 1),
    )
    ir = dict(
        format="ir1-train-v7",
        version=1,
        num_layers=0,
        config=dict(
            hidden_size=width,
            intermediate_size=feedforward,
            num_heads=heads,
            num_kv_heads=heads,
            head_dim=width // heads,
            vocab_size=len(raw["output_embedding.weight"]),
            train_tokens=tokens,
            train_batch_policy="fixed",
            training_integer_inputs=integer_inputs,
        ),
        ops=ops,
        tensors=tensors,
        loss=loss,
    )
    return ir, weights, graph


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
