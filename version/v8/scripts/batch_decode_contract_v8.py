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


def _crossing_values(ops: list[dict[str, Any]]) -> set[str]:
    """Find prefix-produced call buffers read before the suffix overwrites them."""
    return _crossing_at(ops, 5)


def _crossing_at(ops: list[dict[str, Any]], cut: int) -> set[str]:
    """Find call values live across a generated execution cutpoint."""
    produced = set().union(*(_writes(op) for op in ops[:cut])) | {"token_ids"}
    overwritten: set[str] = set()
    crossing: set[str] = set()
    for op in ops[cut:]:
        for arg in op.get("args", []):
            if not isinstance(arg, dict) or arg.get("buffer_ref") not in produced:
                continue
            ref = arg["buffer_ref"]
            if not str(arg.get("source", "")).startswith("output:") and ref not in overwritten:
                crossing.add(ref)
            elif str(arg.get("source", "")).startswith("output:"):
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
    if (_crossing_at(ops, index) != {refs["input"], refs["residual"]}
            or _crossing_at(ops, index + 1) != {output_ref, refs["residual"]}):
        return None
    for ref in refs.values():
        row = buffers[ref]
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


def resolve_two_row_batch_contract(
    ops: list[dict[str, Any]], layout: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any] | None:
    """Admit a KV-only decoder with two equivalent first-layer projections.

    The initial split is intentionally exact: embedding, residual save, norm,
    Q/K projections, then the rest of the normal generated decode. The two
    projections must have the same input and the same map-declared M=2 provider.
    A second map-compatible gate/up projection can be shared after a checked
    sequence-local attention interval. Cache writes remain sequence-local.
    """
    if resolve_sequence_state_contract(layout, config) is None or len(ops) < 6:
        return None
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
    if _crossing_values(ops) != set(refs.values()):
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
    return contract
