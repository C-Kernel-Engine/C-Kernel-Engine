"""Explicit embedded-input entry and layer-output connection planning.

Layer boundaries come from declared section/layer membership, never attention,
MLP, residual counts, model names or emitted C comments.
"""

from copy import deepcopy
import json
from pathlib import Path
import re


def layer_output_boundaries(ops):
    boundaries = {}
    closed = set()
    active = None
    for index, op in enumerate(ops):
        layer = op.get("layer") if op.get("section") == "body" else None
        if layer is not None and (type(layer) is not int or layer < 0):
            raise ValueError(
                "body operations need explicit nonnegative layer identities"
            )
        if layer != active:
            if active is not None:
                closed.add(active)
            if layer in closed:
                raise ValueError("noncontiguous layer boundary is ambiguous")
            active = layer
        if layer is not None:
            boundaries[layer] = index
    return boundaries


def plan_embedded_decode_entry(ops, config):
    if (
        not ops
        or ops[0].get("op") != "dense_embedding_lookup"
        or ops[0].get("section") != "header"
    ):
        raise ValueError(
            "embedded entry requires a declared leading embedding operation"
        )
    output = [
        a for a in ops[0].get("args", []) if a.get("source", "").startswith("output:")
    ]
    if len(output) != 1 or output[0].get("buffer_ref") != "embedded_input":
        raise ValueError("embedded entry requires a unique declared embedding output")
    if any(o.get("op") == "dense_embedding_lookup" for o in ops[1:]):
        raise ValueError("multiple embedding producers cannot be bypassed")
    bridge = config.get("multimodal_bridge_contract") or {}
    count = config.get("num_deepstack_layers", 0) if bridge else 0
    width = config.get("embed_dim")
    if type(count) is not int or count < 0 or type(width) is not int or width <= 0:
        raise ValueError("invalid embedded-input capacity")
    after = {}
    if count:
        # Reuse the resolved FP32 residual provider already admitted by this graph.
        # A bridge-specific arithmetic contract requires its own explicit provider.
        candidates = [o for o in ops if o.get("op") == "residual_add"]
        identities = {
            (o.get("function"), o.get("call_abi", {}).get("kernel_id"))
            for o in candidates
        }
        if len(identities) != 1 or not candidates or not all(next(iter(identities))):
            raise ValueError(
                "embedded residual connection needs one resolved residual provider"
            )
        function, provider = next(iter(identities))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", provider):
            raise ValueError("invalid residual provider identity")
        map_path = (
            Path(__file__).resolve().parents[1] / "kernel_maps" / (provider + ".json")
        )
        if not map_path.is_file():
            raise ValueError("residual provider map unavailable")
        kernel_map = json.loads(map_path.read_text())
        if (
            kernel_map.get("id") != provider
            or kernel_map.get("op") != "residual_add"
            or kernel_map.get("impl", {}).get("function") != function
            or any(
                o.get("call_abi", {}).get("owner") != "kernel_map" for o in candidates
            )
        ):
            raise ValueError("residual call disagrees with its provider map")
        for layer, index in layer_output_boundaries(ops).items():
            if layer >= count:
                continue
            writers = [
                a
                for a in ops[index].get("args", [])
                if a.get("source", "").startswith("output:")
                and a.get("buffer_ref") == "embedded_input"
            ]
            if len(writers) != 1:
                raise ValueError(
                    "declared layer output does not publish the embedded stream"
                )
            injection = deepcopy(candidates[0])
            args = {a["name"]: a for a in injection.get("args", [])}
            if set(args) != {"a", "b", "out", "tokens", "aligned_embed_dim"}:
                raise ValueError("unsupported residual connection ABI")
            for name in ("a", "out"):
                args[name].update(expr=writers[0]["expr"], buffer_ref="embedded_input")
            args["b"].update(
                expr=f"g_bridge_deepstack_slices + {layer * width}",
                buffer_ref="bridge_deepstack_slices",
            )
            args["tokens"]["expr"] = "1"
            args["aligned_embed_dim"]["expr"] = str(width)
            injection.update(
                idx=len(ops) + layer,
                layer=layer,
                connection="visual_rows_after_decoder_layer",
            )
            after[str(index)] = injection
    return dict(
        schema="cke.embedded-decode-entry.v1",
        name="ck_decode_embedded",
        input_buffer="embedded_input",
        input_contract=dict(
            shape=[1, width], dtype="fp32", layout="token_major_contiguous"
        ),
        skip_indices=[0],
        after_operations=after,
        deepstack_elements=max(1, count * width),
    )


def validate_embedded_decode_entry(plan, ops, config):
    if plan != plan_embedded_decode_entry(ops, config):
        raise ValueError(
            "embedded entry disagrees with declared operations and connections"
        )
    return plan
