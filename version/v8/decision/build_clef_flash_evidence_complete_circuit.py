"""Compose both pinned Clef evidence-routing layers in one generated entry.

The second layer reuses the first layer's declarative operation template. Its
query and residual inputs are explicitly rebound to the first layer's output;
the shared memory branch remains unchanged.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from build_clef_flash_evidence0_circuit import build_circuit as build_first

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/clef_flash_evidence_complete_bounded.json"


def _replace_layer_name(value):
    if isinstance(value, str):
        return value.replace("evidence_layers.0", "evidence_layers.1").replace(
            "evidence0_", "evidence1_")
    if isinstance(value, list):
        return [_replace_layer_name(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_layer_name(item) for key, item in value.items()}
    return value


def build_circuit(*, tokens=6, questions=2, options=3, channels=4096,
                  width=1024, vocabulary=16, heads=None, feedforward=None):
    graph = build_first(tokens=tokens, questions=questions, options=options,
                        channels=channels, width=width, vocabulary=vocabulary,
                        heads=heads, feedforward=feedforward)
    first = graph["block_types"]["evidence0"]
    second = _replace_layer_name(deepcopy(first))
    # These are the only two uses of the previous query stream in this block.
    # Check them before rebinding so a template change cannot silently create a
    # stale edge into layer one.
    header = second["header"]
    body = second["body"]["ops"]
    residuals = [op for op in body if op["id"] == "evidence1_attention_residual"]
    if (len(residuals) != 1 or
            header[0]["graph_slots"]["inputs"] != {"input": "option_queries"} or
            residuals[0]["graph_slots"]["inputs"] != {
                "left": "option_queries", "right": "evidence1_attn_out"}):
        raise ValueError("evidence layer query edges changed; review layer-two binding")
    header[0]["graph_slots"]["inputs"]["input"] = "evidence0_output"
    residuals[0]["graph_slots"]["inputs"]["left"] = "evidence0_output"
    for name, shape in list(graph["activation_buffers"].items()):
        if name.startswith("evidence0_"):
            new_name = name.replace("evidence0_", "evidence1_", 1)
            graph["activation_buffers"][new_name] = deepcopy(shape)
            graph["activation_bindings"][new_name] = new_name
    graph["name"] = "clef_flash_evidence_complete_bounded"
    graph["native_entry"]["function"] = "ck_clef_evidence_complete"
    graph["sequence"].append("evidence1")
    graph["block_types"]["evidence1"] = second
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if OUTPUT.read_text() != generated:
            raise SystemExit("Clef complete evidence circuit differs from canonical authoring")
    else:
        OUTPUT.write_text(generated)


if __name__ == "__main__":
    main()
