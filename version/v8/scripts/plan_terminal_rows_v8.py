"""Validate circuit-owned last-output row selection after persistent-state work."""

import math
import re


def plan_terminal_prefill_rows(ir, selector_matches):
    for op in ir.get("operations", []):
        op.pop("prefill_row_selection", None)
    config = ir.get("config", {})
    contract = config.get("contract", {}).get("prefill_terminal_rows")
    if not contract or ir.get("mode") != "prefill" or config.get("logits_layout") != "last":
        return

    def fault(message):
        raise ValueError("HARD TERMINAL ROW CONTRACT FAULT: " + message)

    if set(contract) != {"version", "selector", "start_before", "suffix", "live_inputs"} or contract["version"] != 1:
        fault("unsupported declaration")
    if not selector_matches(contract["selector"], config, contract["start_before"]):
        return
    ops = ir["operations"]
    last_layer = max((int(op.get("layer", -1)) for op in ops), default=-1)
    starts = [i for i, op in enumerate(ops)
              if int(op.get("layer", -1)) == last_layer and op.get("op") == contract["start_before"]]
    if last_layer < 0 or len(starts) != 1:
        fault("start must resolve exactly once in the last layer")
    start = starts[0]
    suffix = ops[start:]
    expected = contract["suffix"]
    if not isinstance(expected, list) or len(suffix) != len(expected):
        fault("terminal suffix length changed")
    for op, choices in zip(suffix, expected):
        if not isinstance(choices, list) or not choices or op.get("op") not in choices:
            fault("terminal suffix operation changed: " + str(op.get("op")))
        if int(op.get("layer", -1)) not in {last_layer, -1}:
            fault("terminal suffix crosses another layer")

    buffers = {b["name"]: b for b in ir["memory"]["activations"]["buffers"]}
    copies = []
    available = set()
    context_extent = int(config.get("context_length") or 0)
    chunk_extent = int(config.get("prefill_chunk_length") or context_extent)
    if context_extent <= 0 or chunk_extent <= 0:
        fault("missing positive prefill extent")
    extent = min(context_extent, chunk_extent)
    if not isinstance(contract["live_inputs"], list) or not contract["live_inputs"]:
        fault("live inputs must be declared")
    for item in contract["live_inputs"]:
        if set(item) != {"op", "argument", "width_args"}:
            fault("unsupported live input declaration")
        matches = [op for op in suffix if op.get("op") == item["op"]]
        if len(matches) != 1:
            fault("live input consumer must be unique")
        args = {a["name"]: a for a in matches[0]["args"]}
        arg = args.get(item["argument"], {})
        name = arg.get("buffer_ref")
        buf = buffers.get(name, {})
        if not str(arg.get("source", "")).startswith("activation:") or buf.get("dtype") not in {"f32", "fp32"}:
            fault("live input must reference an FP32 activation buffer")
        define = buf.get("define", "")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", define):
            fault("invalid activation define")
        if arg.get("expr") != f"(const float*)(model->bump + {define})":
            fault("live input must reference the full buffer base")
        widths = item["width_args"]
        if not isinstance(widths, list) or not widths:
            fault("missing row width arguments")
        factors = []
        for width_arg in widths:
            dim = args.get(width_arg, {})
            value = str(dim.get("expr", ""))
            if not str(dim.get("source", "")).startswith("dim:") or not value.isdecimal() or int(value) <= 0:
                fault("row widths must resolve from positive dimension arguments")
            factors.append(int(value))
        width = math.prod(factors)
        required_bytes = width * extent * 4
        planned_bytes = int(buf.get("size", 0))
        if required_bytes > planned_bytes:
            fault(
                "live input exceeds planned buffer capacity: "
                f"{item['op']}.{item['argument']} buffer={name} "
                f"requires={required_bytes} planned={planned_bytes} "
                f"rows={extent} row_elements={width}"
            )
        if name in available:
            fault("duplicate live input buffer")
        available.add(name)
        copies.append({"buffer": name, "define": define, "row_elements": width})

    # Every activation read must be a declared live-in or a preceding suffix output.
    # This prevents moving selection past a residual/skip input without moving it too.
    live_used = set()
    produced = set()
    for op in suffix:
        for arg in op["args"]:
            if str(arg.get("source", "")).startswith("activation:"):
                name = arg.get("buffer_ref")
                if name not in available and name not in produced:
                    fault("undeclared live input: " + str(name))
                if name not in produced:
                    live_used.add(name)
        produced.update(a["buffer_ref"] for a in op["args"]
                        if str(a.get("source", "")).startswith("output:") and a.get("buffer_ref"))
    if live_used != available:
        fault("unused live input declaration")
    ops[start]["prefill_row_selection"] = {"version": 1, "selection": "last", "copies": copies}
