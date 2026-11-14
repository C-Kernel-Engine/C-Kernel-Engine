"""Author a frozen native decoder feature producer before an existing circuit.

This describes the connection; token execution and feature transport are native.
The supplied component's weights, ABI identity and numerical certification remain
separate package responsibilities. This does not synthesize backbone gradients.
"""

from copy import deepcopy
import re


def compose_decoder_features(
    circuit, *, feature_input="hidden", token_input="backbone_ids"
):
    if any(
        not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
        for name in (feature_input, token_input)
    ):
        raise ValueError("component buffer names must be identifiers")
    if circuit.get("checked_native_entry") is not True:
        raise ValueError("component composition requires a checked native entry")
    graph = deepcopy(circuit)
    buffers = graph["activation_buffers"]
    shape = buffers[feature_input]["shape"]
    if (
        len(shape) != 2
        or any(type(n) is not int or n <= 0 for n in shape)
        or token_input in buffers
        or "native_feature_source" in graph["block_types"]
    ):
        raise ValueError(
            "feature composition requires bounded positive geometry and unique inputs"
        )
    tokens, channels = shape
    if graph["native_entry"].get("segments"):
        raise ValueError("component composition requires an unsegmented entry")
    parameters = graph["native_entry"]["params"]
    if any(
        p["name"] in {"decoder_context", "decoder_context_bytes"} for p in parameters
    ):
        raise ValueError("component entry parameters already declared")
    matches = 0
    for block in graph["block_types"].values():
        for section in ("header", "body", "footer"):
            ops = block.get(section, [])
            if isinstance(ops, dict):
                ops = ops.get("ops", [])
            for op in ops:
                outputs = op.get("graph_slots", {}).get("outputs", {})
                if feature_input in outputs.values():
                    raise ValueError("feature input already has a circuit producer")
                inputs = op.get("graph_slots", {}).get("inputs", {})
                for name, edge in list(inputs.items()):
                    if edge == "external:" + feature_input:
                        inputs[name] = feature_input
                        matches += 1
    if not matches:
        raise ValueError("no declared external feature edge to connect")
    parameters.extend(
        [
            {"name": "decoder_context", "c_type": "const void *"},
            {"name": "decoder_context_bytes", "c_type": "size_t"},
        ]
    )
    buffers[token_input] = {"shape": [tokens]}
    graph["activation_bindings"][token_input] = token_input
    operation = {
        "id": "collect_native_features",
        "op": "decoder_features_sequence",
        "kernel": "decoder_features_sequence_checked_f32",
        "returns_status": True,
        "graph_slots": {
            "inputs": {"ids": "external:" + token_input},
            "outputs": {"output": feature_input},
        },
        "params": {
            "T": tokens,
            "C": channels,
            "Q": tokens * channels,
            "call_constants": {
                "id_elements": tokens,
                "feature_tokens": tokens,
                "feature_channels": channels,
                "feature_elements": tokens * channels,
            },
        },
    }
    graph["block_types"]["native_feature_source"] = {
        "sequence": ["header", "body", "footer"],
        "header": [],
        "footer": [],
        "body": {"type": "dense", "ops": [operation]},
    }
    graph["sequence"].insert(0, "native_feature_source")
    return graph
