"""Resolve the first, deliberately narrow generated two-row decode contract.

This is an IR/layout capability check. It does not identify models by name and
does not rewrite emitted C. Unsupported graphs retain their ordinary decoder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sequence_state_contract_v8 import resolve_sequence_state_contract


_WEIGHT = re.compile(r"^\(const void\*\)\(model->bump \+ (W_[A-Z0-9_]+)\)$")
_BASE = re.compile(r"^\((?:const )?(?:float|void|int32_t|uint8_t)\*\)\(model->bump \+ (A_[A-Z0-9_]+)\)$")


def _args(op: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {arg["name"]: arg for arg in op.get("args", []) if isinstance(arg, dict) and isinstance(arg.get("name"), str)}


def _writes(op: dict[str, Any]) -> set[str]:
    return {str(arg["buffer_ref"]) for arg in op.get("args", [])
            if isinstance(arg, dict) and str(arg.get("source", "")).startswith("output:")
            and isinstance(arg.get("buffer_ref"), str)}


def _crossing_values(ops: list[dict[str, Any]], buffers: dict[str, dict[str, Any]]) -> set[str]:
    """Find prefix-produced call buffers read before the suffix overwrites them."""
    return _crossing_at(ops, 5, _certified_full_writes(ops, buffers))


def _unaliased(ref: str, buffers: dict[str, dict[str, Any]]) -> bool:
    row = buffers.get(ref)
    if not row or not isinstance(row.get("abs_offset"), int) or not isinstance(row.get("size"), int):
        return False
    start, end = row["abs_offset"], row["abs_offset"] + row["size"]
    return all(
        other == ref or item.get("lifetime") != "call" or
        (isinstance(item.get("abs_offset"), int) and isinstance(item.get("size"), int)
         and (end <= item["abs_offset"] or item["abs_offset"] + item["size"] <= start))
        for other, item in buffers.items()
    )


def _certified_full_writes(
    ops: list[dict[str, Any]], buffers: dict[str, dict[str, Any]],
) -> frozenset[tuple[int, str]]:
    """Recognize exact-base writes covering the complete decode-row value.

    A hybrid prefill/decode arena can reserve much more than one decode row.
    Match the operation's declared row extent, never the arena capacity.
    """
    spans: dict[str, int] = {}
    conflicting: set[str] = set()

    def record(ref: Any, size: int) -> None:
        if not isinstance(ref, str) or size <= 0:
            return
        if ref in spans and spans[ref] != size:
            conflicting.add(ref)
        else:
            spans[ref] = size

    # The first Q projection consumes the decoder hidden row. Projection
    # outputs and attention output have explicit per-row dimensions.
    for op in ops:
        name, args = op.get("op"), _args(op)
        try:
            if name == "q_proj":
                record(args["x"].get("buffer_ref"), int(args["K"]["expr"]) * 4)
            if name in {"q_proj", "k_proj", "v_proj"}:
                record(args["y"].get("buffer_ref"), int(args["M"]["expr"]) * 4)
            elif name == "residual_save":
                record(args["dst"].get("buffer_ref"), int(args["size"]["expr"]))
            elif name == "out_proj":
                record(args["x"].get("buffer_ref"), int(args["K"]["expr"]) * 4)
        except (KeyError, TypeError, ValueError):
            continue
    complete: set[tuple[int, str]] = set()
    for index, op in enumerate(ops):
        name = op.get("op")
        args = _args(op)
        output_name = (
            "y" if name in {"q_proj", "k_proj", "v_proj", "out_proj", "mlp_down"}
            else "dst" if name == "residual_save" else None
        )
        if output_name is None or output_name not in args:
            continue
        output = args[output_name]
        ref = output.get("buffer_ref")
        row = buffers.get(ref)
        if (not row or ref in conflicting or row.get("lifetime") != "call"
                or not _unaliased(ref, buffers)):
            continue
        try:
            written = int(args["size"]["expr"]) if name == "residual_save" else int(args["M"]["expr"]) * 4
        except (KeyError, TypeError, ValueError):
            continue
        if (written == spans.get(ref) and isinstance(row.get("size"), int)
                and row["size"] >= written and _exact_base(output, row)):
            complete.add((index, ref))
    # Attention's complete output extent is proved by the next projection's
    # input width. Its scratch ports do not themselves prove an overwrite.
    for index, op in enumerate(ops[:-1]):
        if op.get("op") != "attn_sliding" or ops[index + 1].get("op") != "out_proj":
            continue
        out = _args(op).get("output", {})
        following = _args(ops[index + 1])
        ref = out.get("buffer_ref")
        row = buffers.get(ref)
        try:
            width = int(following["K"]["expr"]) * 4
        except (KeyError, TypeError, ValueError):
            continue
        if (row and ref not in conflicting and row.get("lifetime") == "call"
                and spans.get(ref) == width and isinstance(row.get("size"), int)
                and row["size"] >= width
                and following.get("x", {}).get("buffer_ref") == ref
                and _exact_base(out, row) and _exact_base(following["x"], row)
                and _unaliased(ref, buffers)):
            complete.add((index, ref))
    return frozenset(complete)


def _crossing_at(
    ops: list[dict[str, Any]], cut: int,
    full_replacements: frozenset[tuple[int, str]] = frozenset(),
) -> set[str]:
    """Find call values live across a cut, independent of argument order.

    An output port alone does not prove a full write. Only a separately
    validated replacement may kill an incoming value; unknown and partial
    writes conservatively leave it live and therefore prevent admission.
    """
    produced = set().union(*(_writes(op) for op in ops[:cut])) | {"token_ids"}
    overwritten: set[str] = set()
    crossing: set[str] = set()
    for index, op in enumerate(ops[cut:], cut):
        reads: set[str] = set()
        writes: set[str] = set()
        for arg in op.get("args", []):
            if not isinstance(arg, dict) or arg.get("buffer_ref") not in produced:
                continue
            ref = arg["buffer_ref"]
            source = str(arg.get("source", ""))
            if source.startswith("output:"):
                writes.add(ref)
            else:
                reads.add(ref)
        crossing.update(reads - overwritten)
        for ref in writes:
            if (index, ref) in full_replacements:
                overwritten.add(ref)
    return crossing


def _layer_extension(
    ops: list[dict[str, Any]], buffers: dict[str, dict[str, Any]],
    refs: dict[str, str], input_dim: int, saved_bytes: int,
    selected: str, batch_function: str, max_input_dim: int,
) -> dict[str, Any] | None:
    """Admit a second shared projection after sequence-local attention.

    Cutpoints and live values come from the lowered graph. The supported
    arithmetic remains selected by the same map as Q/K; no model name chooses
    the schedule. Unknown uses retain the original Q/K-only batch entry.
    """
    layer = ops[3].get("layer")
    candidates = [(index, op) for index, op in enumerate(ops[5:], 5)
                  if op.get("op") == "mlp_gate_up" and op.get("layer") == layer]
    if len(candidates) != 1:
        return None
    index, op = candidates[0]
    if index <= 5 or any(item.get("layer") != layer for item in ops[5:index + 1]):
        return None
    args = _args(op)
    if set(args) != {"x", "y", "W", "M", "K"}:
        return None
    if args["x"].get("buffer_ref") != refs["input"]:
        return None
    try:
        output_dim, width = int(args["M"]["expr"]), int(args["K"]["expr"])
    except (KeyError, TypeError, ValueError):
        return None
    if width != input_dim or width > max_input_dim or output_dim <= 0 or output_dim % 2:
        return None
    if ((op.get("call_abi") or {}).get("kernel_id") != selected
            or op.get("function") != ops[3].get("function")):
        return None
    output_ref = args["y"].get("buffer_ref")
    output = buffers.get(output_ref)
    if (not output or output_ref in refs.values()
            or output.get("lifetime") != "call" or output.get("mutable") is not True
            or int(output.get("size", 0)) < output_dim * 4
            or not re.fullmatch(r"A_[A-Z0-9_]+", str(output.get("define", "")))
            or not _exact_base(args["x"], buffers[refs["input"]])
            or not _exact_base(args["y"], output)):
        return None
    weight = _WEIGHT.fullmatch(str(args["W"].get("expr", "")))
    if not weight or index + 1 >= len(ops):
        return None
    consumer = ops[index + 1]
    consumer_args = _args(consumer)
    try:
        consumer_dim = int(consumer_args["dim"]["expr"])
    except (KeyError, TypeError, ValueError):
        return None
    if (consumer.get("op") != "geglu" or consumer.get("layer") != layer
            or consumer_dim * 2 != output_dim
            or consumer_args.get("tokens", {}).get("expr") != "1"
            or consumer_args.get("x", {}).get("buffer_ref") != output_ref
            or consumer_args.get("out", {}).get("buffer_ref") != output_ref
            or not _exact_base(consumer_args["x"], output)
            or not _exact_base(consumer_args["out"], output)):
        return None
    # The admitted gate/up kernel writes the exact contiguous span consumed by
    # the immediately following GeGLU. Other output ports remain unknown
    # writes for crossing analysis, including partial/in-place operations.
    replacements = _certified_full_writes(ops, buffers) | {(index, output_ref)}
    if (_crossing_at(ops, index, replacements) != {refs["input"], refs["residual"]}
            or _crossing_at(ops, index + 1, replacements) != {output_ref, refs["residual"]}):
        return None
    for ref, row in buffers.items():
        if ref == output_ref or row.get("lifetime") != "call":
            continue
        if (int(row["abs_offset"]) < int(output["abs_offset"]) + int(output["size"])
                and int(output["abs_offset"]) < int(row["abs_offset"]) + int(row["size"])):
            return None
    return {
        "cut": index, "post_cut": index + 1,
        "input_define": buffers[refs["input"]]["define"],
        "residual_define": buffers[refs["residual"]]["define"],
        "output_define": output["define"],
        "input_bytes": input_dim * 4, "residual_bytes": saved_bytes,
        "output_bytes": output_dim * 4, "output_dim": output_dim,
        "weight": weight.group(1), "gemm_function": batch_function,
    }


def _exact_base(arg: dict[str, Any], buffer: dict[str, Any]) -> bool:
    match = _BASE.fullmatch(str(arg.get("expr", "")))
    return bool(match and match.group(1) == buffer.get("define"))


def _mapped_two_row_projection(
    op: dict[str, Any], providers: dict[str, dict[str, Any]],
    buffers: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve an exact-base projection through its selected decode map."""
    args = _args(op)
    input_arg = args.get("x") or args.get("x_q8")
    if set(args) not in ({"x", "y", "W", "M", "K"}, {"x_q8", "y", "W", "M", "K"}) or not input_arg:
        return None
    try:
        width, height = int(args["K"]["expr"]), int(args["M"]["expr"])
    except (KeyError, ValueError, TypeError):
        return None
    if width <= 0 or height <= 0:
        return None
    input_ref, output_ref = input_arg.get("buffer_ref"), args["y"].get("buffer_ref")
    if input_ref not in buffers or output_ref not in buffers or input_ref == output_ref:
        return None
    if not _exact_base(input_arg, buffers[input_ref]) or not _exact_base(args["y"], buffers[output_ref]):
        return None
    weight = _WEIGHT.fullmatch(str(args["W"].get("expr", "")))
    if not weight:
        return None
    selected = (op.get("call_abi") or {}).get("kernel_id")
    decode = providers.get(selected, {})
    batch_id = decode.get("batch_decode_two_rows", {}).get("provider_id")
    provider = providers.get(batch_id, {})
    binding = provider.get("batch_decode_two_rows", {})
    max_input_dim = binding.get("max_input_dim")
    alignment = decode.get("constraints", {}).get("alignment", {}).get("K", 1)
    if (not isinstance(max_input_dim, int) or max_input_dim <= 0
            or not isinstance(alignment, int) or alignment <= 0
            or width % alignment):
        return None
    if (decode.get("impl", {}).get("function") != op.get("function")
            or binding.get("decode_provider_id") != selected
            or binding.get("decode_function") != op.get("function")
            or binding.get("rows") != 2
            or binding.get("numerical_contract") != provider.get("numerical_contract")
            or any(decode.get("quant", {}).get(key) != provider.get("quant", {}).get(key)
                   for key in ("weight", "activation", "output"))
            or width > max_input_dim):
        return None
    function = binding.get("function")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", str(function or "")):
        return None
    dtype = binding.get("input_dtype", "fp32")
    if dtype not in {"fp32", "q8_k"} or (dtype == "q8_k") != ("x_q8" in args):
        return None
    if dtype == "q8_k" and width % 256:
        return None
    input_bytes = width * 4 if dtype == "fp32" else width // 256 * 292
    output_bytes = height * 4
    input_capacity = buffers[input_ref].get("size")
    output_capacity = buffers[output_ref].get("size")
    if (not isinstance(input_capacity, int) or input_capacity < input_bytes
            or not isinstance(output_capacity, int) or output_capacity < output_bytes):
        return None
    return {"input": input_ref, "output": output_ref, "input_bytes": input_bytes,
            "output_bytes": output_bytes, "input_dim": width, "output_dim": height,
            "weight": weight.group(1), "function": function, "dtype": dtype}


def _layer_projection_groups(
    ops: list[dict[str, Any]], start: int, end: int, layer: int,
    buffers: dict[str, dict[str, Any]], providers: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[list[dict[str, Any]]] | None:
    """Admit one layer's map-selected, ordered row-wise projections."""
    if str(config.get("out_proj_input_policy") or "").strip().lower() in {
        "fp32", "fp32_input", "force_fp32"
    }:
        return None
    for key in ("out_proj_fp32_layers", "out_proj_fp32_layer",
                "attn_proj_fp32_layers", "attn_qkv_fp32_layers", "attn_proj_fp32_layer",
                "mlp_gate_up_fp32_layers", "mlp_gate_up_fp32_layer",
                "mlp_gate_up_q8_contract_layers", "mlp_gate_up_q8_contract_layer"):
        raw = config.get(key)
        values = raw if isinstance(raw, (list, tuple, set)) else (raw,)
        if any(value is not None and str(value) == str(layer) for value in values):
            return None
    projection_names = ("q_proj", "k_proj", "v_proj", "out_proj", "mlp_gate_up", "mlp_down")
    indexed = [(i, ops[i]) for i in range(start, end) if ops[i].get("op") in projection_names]
    if [op.get("op") for _, op in indexed] != list(projection_names):
        return None
    if any(op.get("layer") != layer for _, op in indexed):
        return None
    projections = []
    for index, op in indexed:
        projection = _mapped_two_row_projection(op, providers, buffers)
        if projection is None:
            return None
        projections.append({"index": index, **projection})
    # Only contiguous projections with one identical packed input may share a
    # stage. Stateful attention and other graph operations stay sequence-local.
    groups: list[list[dict[str, Any]]] = []
    for projection in projections:
        if (groups and projection["index"] == groups[-1][-1]["index"] + 1
                and projection["input"] == groups[-1][0]["input"]
                and projection["input_bytes"] == groups[-1][0]["input_bytes"]):
            groups[-1].append(projection)
        else:
            groups.append([projection])
    if [len(group) for group in groups] != [3, 1, 1, 1]:
        return None
    gate = groups[2][0]
    gate_after = ops[gate["index"] + 1]
    gate_args = _args(gate_after)
    try:
        gate_consumer_dim = int(gate_args.get("dim", {}).get("expr", 0))
    except (TypeError, ValueError):
        return None
    if (gate_after.get("op") != "geglu" or gate_after.get("layer") != layer
            or gate_args.get("x", {}).get("buffer_ref") != gate["output"]
            or gate_args.get("out", {}).get("buffer_ref") != gate["output"]
            or gate_args.get("tokens", {}).get("expr") != "1"
            or gate_consumer_dim * 2 != gate["output_dim"]):
        return None
    down = groups[3][0]
    if down["dtype"] == "q8_k":
        quantize = ops[down["index"] - 1]
        qargs = _args(quantize)
        if (quantize.get("op") != "quantize_mlp_down_input"
                or quantize.get("layer") != layer
                or qargs.get("y", {}).get("buffer_ref") != down["input"]
                or not _exact_base(qargs.get("y", {}), buffers[down["input"]])
                or qargs.get("rows", {}).get("expr") != "1"
                or qargs.get("k", {}).get("expr") != str(down["input_dim"])):
            return None
    elif (down["input"] != gate["output"]
          or down["index"] != gate["index"] + 2
          or gate_consumer_dim != down["input_dim"]
          or not _exact_base(gate_args["out"], buffers[down["input"]])):
        return None
    for group in groups:
        outputs = [projection["output"] for projection in group]
        if len(outputs) != len(set(outputs)) or group[0]["input"] in outputs:
            return None
    return groups


def _full_layer_plan(
    ops: list[dict[str, Any]], buffers: dict[str, dict[str, Any]],
    providers: dict[str, dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any] | None:
    """Extend checked shared projections across contiguous supported layers.

    Each layer supplies the same bounded operation contract. The cuts, live
    values, extents and selected providers come from the complete lowered IR.
    The first unsupported layer and everything after it execute locally.
    """
    if len(ops) < 22:
        return None
    first_layer = ops[3].get("layer")
    if not isinstance(first_layer, int) or first_layer < 0:
        return None
    layer_limit = config.get("batch_decode_max_shared_layers")
    if layer_limit is not None and (type(layer_limit) is not int or layer_limit < 1):
        return None
    groups: list[list[dict[str, Any]]] = []
    shared_layers: list[int] = []
    start, layer, selected_end = 0, first_layer, 0
    while start < len(ops):
        end = next((i for i in range(max(start + 1, 4), len(ops))
                    if ops[i].get("layer") != layer), None)
        if end is None:
            break
        layer_groups = _layer_projection_groups(
            ops, start, end, layer, buffers, providers, config,
        )
        if layer_groups is None:
            break
        groups.extend(layer_groups)
        shared_layers.append(layer)
        selected_end = end
        start = end
        if layer_limit is not None and len(shared_layers) >= layer_limit:
            break
        if ops[start].get("layer") != layer + 1:
            break
        layer += 1
    if not shared_layers:
        return None
    stages = []
    cursor = 0
    for group in groups:
        start, stop = group[0]["index"], group[-1]["index"] + 1
        if cursor < start:
            stages.append({"kind": "local", "start": cursor, "stop": start})
        stages.append({"kind": "shared", "start": start, "stop": stop, "projections": group})
        cursor = stop
    stages.append({"kind": "local", "start": cursor, "stop": len(ops)})
    # The final local stage runs every unsupported later layer. Liveness at
    # every cut includes the complete remaining decode sequence.
    extents: dict[str, int] = {}
    for projection in (projection for group in groups for projection in group):
        for ref, count in ((projection["input"], projection["input_bytes"]),
                           (projection["output"], projection["output_bytes"])):
            extents[ref] = max(extents.get(ref, 0), count)
    for op in ops[:selected_end]:
        if op.get("op") != "residual_save":
            continue
        save = _args(op)
        try:
            residual = save["dst"]["buffer_ref"]
            count = int(save["size"]["expr"])
        except (KeyError, ValueError, TypeError):
            return None
        extents[residual] = max(extents.get(residual, 0), count)
    replacements = _certified_full_writes(ops, buffers) | {
        (groups[index][0]["index"], groups[index][0]["output"])
        for index in range(2, len(groups), 4)
    }
    for stage in stages[:-1]:
        cut = stage["stop"]
        live = _crossing_at(ops, cut, replacements)
        if not live or any(ref not in extents for ref in live):
            return None
        for ref in live:
            row = buffers[ref]
            capacity = row.get("size")
            if (row.get("lifetime") != "call" or row.get("mutable") is not True
                    or not isinstance(capacity, int) or capacity < extents[ref]
                    or not _unaliased(ref, buffers)
                    or not re.fullmatch(r"A_[A-Z0-9_]+", str(row.get("define", "")))):
                return None
        stage["live_after"] = sorted(live)
    for before, shared in zip(stages, stages[1:]):
        if shared["kind"] == "shared":
            if any(projection["input"] not in before.get("live_after", [])
                   or projection["output"] not in shared["live_after"]
                   for projection in shared["projections"]):
                return None
    # Every call-local access in shared layers must use the declared base. This
    # rejects offset expressions that snapshots could not preserve.
    for op in ops[:selected_end]:
        for arg in op.get("args", []):
            ref = arg.get("buffer_ref") if isinstance(arg, dict) else None
            if (ref in extents and not str(arg.get("source", "")).startswith("scratch:")
                    and not _exact_base(arg, buffers[ref])):
                return None
    return {"stages": stages, "buffers": {ref: buffers[ref]["define"] for ref in extents},
            "extents": extents, "layer_end": selected_end,
            "shared_layers": shared_layers, "projection_groups": len(groups)}


def _first_recurrent_projection_plan(
    ops: list[dict[str, Any]], buffers: dict[str, dict[str, Any]],
    providers: dict[str, dict[str, Any]], state: dict[str, int],
    config: dict[str, Any], row_buffers: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Admit one mapped projection before the first recurrent state update.

    This is deliberately a one-group contract. The gate and subsequent layers
    execute through the ordinary sequence-local generated calls. Snapshot the
    complete decode-row extents at both cuts, including values whose old
    contents a later operation could read after a partial write. In a combined
    runtime the prefill arena may be much larger than the decode-row plan.
    """
    if (config.get("batch_decode_recurrent_first_projection") is not True
            or "recurrent_conv_state_bytes" not in state or len(ops) < 7):
        return None
    if [op.get("op") for op in ops[:5]] != [
        "dense_embedding_lookup", "residual_save", "attn_norm",
        "quantize_input_0", "recurrent_qkv_proj",
    ] or ops[4].get("layer") != 0:
        return None
    logits = buffers.get("logits")
    vocab = config.get("vocab_size")
    if (not logits or type(vocab) is not int or vocab <= 0
            or logits.get("size") != vocab * 4):
        return None
    projection = _mapped_two_row_projection(ops[4], providers, buffers)
    if projection is None:
        return None
    replacements = _certified_full_writes(ops, buffers)
    before = _crossing_at(ops, 4, replacements)
    after = _crossing_at(ops, 5, replacements)
    refs = before | after | {projection["input"], projection["output"]}
    if not before or not after:
        return None
    extents: dict[str, int] = {}
    for ref in refs:
        row = buffers.get(ref)
        decode_row = row_buffers.get(ref) if row_buffers is not None else row
        if (not row or row.get("lifetime") != "call" or row.get("mutable") is not True
                or type(row.get("size")) is not int or row["size"] <= 0
                or not re.fullmatch(r"A_[A-Z0-9_]+", str(row.get("define", "")))
                or not _unaliased(ref, buffers)
                or not decode_row or decode_row.get("lifetime") != "call"
                or decode_row.get("mutable") is not True
                or decode_row.get("define") != row["define"]
                or type(decode_row.get("size")) is not int
                or not 0 < decode_row["size"] <= row["size"]
                or (row_buffers is not None and not _unaliased(ref, row_buffers))):
            return None
        extents[ref] = decode_row["size"]
    if (extents[projection["input"]] < projection["input_bytes"]
            or extents[projection["output"]] < projection["output_bytes"]):
        return None
    # No hidden alias into a snapshotted buffer may bypass the planned base.
    for op in ops:
        for arg in op.get("args", []):
            if (isinstance(arg, dict) and arg.get("buffer_ref") in refs
                    and not _exact_base(arg, buffers[arg["buffer_ref"]])):
                return None
    return {
        "stages": [
            {"kind": "local", "start": 0, "stop": 4, "live_after": sorted(before)},
            {"kind": "shared", "start": 4, "stop": 5,
             "projections": [{"index": 4, **projection}], "live_after": sorted(after)},
            {"kind": "local", "start": 5, "stop": len(ops)},
        ],
        "buffers": {ref: buffers[ref]["define"] for ref in extents},
        "extents": extents, "layer_end": 5,
        "shared_layers": [0], "projection_groups": 1,
        "recurrent_state_bytes": {
            "conv": state["recurrent_conv_state_bytes"],
            "ssm": state["recurrent_ssm_state_bytes"],
        },
    }


def resolve_two_row_batch_contract(
    ops: list[dict[str, Any]], layout: dict[str, Any], config: dict[str, Any],
    *, row_layout: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Admit a checked, map-selected two-row decode projection boundary.

    The initial split is intentionally exact: embedding, residual save, norm,
    Q/K projections, then the rest of the normal generated decode. The two
    projections must have the same input and the same map-declared M=2 provider.
    A second map-compatible gate/up projection can be shared after a checked
    sequence-local attention interval. Cache writes remain sequence-local.
    """
    state = resolve_sequence_state_contract(layout, config, ops)
    if state is None or len(ops) < 6:
        return None
    if "recurrent_conv_state_bytes" in state:
        buffers = {
            row.get("name"): row
            for row in layout.get("memory", {}).get("activations", {}).get("buffers", [])
            if isinstance(row, dict)
        }
        registry = json.loads((Path(__file__).resolve().parents[1] / "kernel_maps" / "KERNEL_REGISTRY.json").read_text())
        providers = {item["id"]: item for item in registry["kernels"]}
        row_buffers = None
        if row_layout is not None:
            row_buffers = {
                row.get("name"): row
                for row in row_layout.get("memory", {}).get("activations", {}).get("buffers", [])
                if isinstance(row, dict)
            }
        plan = _first_recurrent_projection_plan(
            ops, buffers, providers, state, config, row_buffers)
        return {"full_layer_plan": plan} if plan else None
    if [op.get("op") for op in ops[:5]] != [
        "dense_embedding_lookup", "residual_save", "attn_norm", "q_proj", "k_proj"
    ]:
        return None
    q, k = ops[3:5]
    if q.get("layer") != k.get("layer") or q.get("function") != k.get("function"):
        return None
    qargs, kargs = _args(q), _args(k)
    if set(qargs) != {"x", "y", "W", "M", "K"} or set(kargs) != set(qargs):
        return None
    if qargs["x"].get("buffer_ref") != kargs["x"].get("buffer_ref"):
        return None
    if qargs["K"].get("expr") != kargs["K"].get("expr"):
        return None
    try:
        input_dim = int(qargs["K"]["expr"])
        q_dim, k_dim = int(qargs["M"]["expr"]), int(kargs["M"]["expr"])
    except (TypeError, ValueError, KeyError):
        return None
    if min(input_dim, q_dim, k_dim) <= 0 or input_dim % 32:
        return None
    buffers = {
        row.get("name"): row
        for row in layout.get("memory", {}).get("activations", {}).get("buffers", [])
        if isinstance(row, dict)
    }
    refs = {
        "input": qargs["x"].get("buffer_ref"),
        "q": qargs["y"].get("buffer_ref"),
        "k": kargs["y"].get("buffer_ref"),
        "residual": _args(ops[1]).get("dst", {}).get("buffer_ref"),
    }
    if len(set(refs.values())) != 4 or any(ref not in buffers for ref in refs.values()):
        return None
    emb, save, norm = (_args(op) for op in ops[:3])
    if (not {"token_ids", "output", "token_count"} <= emb.keys()
            or not {"dst", "src", "size"} <= save.keys()
            or not {"input", "output"} <= norm.keys()
            or [_writes(op) for op in ops[:3]] != [
                {refs["input"]}, {refs["residual"]}, {refs["input"]}]
            or emb["output"].get("buffer_ref") != refs["input"]
            or save["src"].get("buffer_ref") != refs["input"]
            or norm["input"].get("buffer_ref") != refs["input"]
            or norm["output"].get("buffer_ref") != refs["input"]
            or emb["token_count"].get("expr") != "1"):
        return None
    if _crossing_values(ops, buffers) != set(refs.values()):
        return None
    try:
        saved_bytes = int(save["size"].get("expr", -1))
    except (TypeError, ValueError):
        return None
    if saved_bytes != input_dim * 4 or saved_bytes % 4:
        return None
    sizes = {name: input_dim * 4 if name == "input" else
             q_dim * 4 if name == "q" else
             k_dim * 4 if name == "k" else
             saved_bytes for name, ref in refs.items()}
    for name, ref in refs.items():
        row = buffers[ref]
        if row.get("lifetime") != "call" or row.get("mutable") is not True:
            return None
        if int(row.get("size", 0)) < sizes[name] or not re.fullmatch(r"A_[A-Z0-9_]+", str(row.get("define", ""))):
            return None
    for op in ops:
        for arg in op.get("args", []):
            if isinstance(arg, dict) and arg.get("buffer_ref") in refs.values():
                if not _exact_base(arg, buffers[arg["buffer_ref"]]):
                    return None
    for first, left in enumerate(refs.values()):
        a = buffers[left]
        for right in list(refs.values())[first + 1:]:
            b = buffers[right]
            if int(a["abs_offset"]) < int(b["abs_offset"]) + int(b["size"]) and int(b["abs_offset"]) < int(a["abs_offset"]) + int(a["size"]):
                return None
    weights = []
    for args in (qargs, kargs):
        match = _WEIGHT.fullmatch(str(args["W"].get("expr", "")))
        if not match:
            return None
        weights.append(match.group(1))
    # The output path must be last-only; a per-position logits buffer requires
    # a separate destination contract before batched decode can be advertised.
    logits = buffers.get("logits")
    vocab = config.get("vocab_size")
    if not logits or not isinstance(vocab, int) or logits.get("size") != vocab * 4:
        return None
    selected = (q.get("call_abi") or {}).get("kernel_id")
    if not isinstance(selected, str) or (k.get("call_abi") or {}).get("kernel_id") != selected:
        return None
    registry = json.loads((Path(__file__).resolve().parents[1] / "kernel_maps" / "KERNEL_REGISTRY.json").read_text())
    providers = {item["id"]: item for item in registry["kernels"]}
    decode = providers.get(selected, {})
    batch_id = decode.get("batch_decode_two_rows", {}).get("provider_id")
    if not isinstance(batch_id, str):
        return None
    provider = providers.get(batch_id, {})
    binding = provider.get("batch_decode_two_rows", {})
    if (decode.get("impl", {}).get("function") != q.get("function")
            or decode.get("quant", {}).get("weight") != provider.get("quant", {}).get("weight")
            or decode.get("quant", {}).get("activation") != provider.get("quant", {}).get("activation")
            or decode.get("quant", {}).get("output") != provider.get("quant", {}).get("output")
            or binding.get("decode_provider_id") != selected
            or binding.get("decode_function") != q.get("function")
            or binding.get("numerical_contract") != provider.get("numerical_contract")
            or binding.get("rows") != 2):
        return None
    function = binding.get("function")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", str(function or "")):
        return None
    max_input_dim = binding.get("max_input_dim")
    if not isinstance(max_input_dim, int) or max_input_dim <= 0 or input_dim > max_input_dim:
        return None
    contract = {
        "prefix_len": 3,
        "suffix_start": 5,
        "input_dim": input_dim,
        "q_dim": q_dim,
        "k_dim": k_dim,
        "buffers": {name: buffers[ref]["define"] for name, ref in refs.items()},
        "sizes": sizes,
        "weights": weights,
        "gemm_function": function,
    }
    extension = _layer_extension(
        ops, buffers, refs, input_dim, saved_bytes, selected, function,
        max_input_dim,
    )
    if extension:
        contract["layer_extension"] = extension
        full_layer = _full_layer_plan(ops, buffers, providers, config)
        if full_layer:
            contract["full_layer_plan"] = full_layer
    return contract


def resolve_mixed_prefill_local_stages(
    decode_ops: list[dict[str, Any]],
    prefill_phase_ops: list[dict[str, Any]] | None,
    plan: dict[str, Any],
) -> bool:
    """Admit phase-aware local stages without changing shared projections.

    The prefill-phase operations have already been resolved against selected
    provider maps. The mixed entry may change only local function bindings;
    changing an argument, live buffer, or shared projection needs a distinct
    batch provider contract and must retain decode-replay execution.
    """
    if prefill_phase_ops is None or len(decode_ops) != len(prefill_phase_ops):
        return False
    local_indices = {index for stage in plan.get("stages", [])
                     if stage.get("kind") == "local"
                     for index in range(stage["start"], stage["stop"])}
    if not local_indices or len(local_indices) == len(decode_ops):
        return False
    changed = False
    for index, (ordinary, prefill) in enumerate(zip(decode_ops, prefill_phase_ops)):
        if ordinary == prefill:
            continue
        if index not in local_indices:
            return False
        ordinary_other = {key: value for key, value in ordinary.items()
                          if key != "function"}
        prefill_other = {key: value for key, value in prefill.items()
                         if key != "function"}
        alternate = ordinary.get("prefill_continuation")
        if (ordinary_other != prefill_other or
                not isinstance(alternate, dict) or
                alternate.get("function") != prefill.get("function") or
                ordinary.get("function") == prefill.get("function")):
            return False
        changed = True
    return changed
