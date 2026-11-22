#!/usr/bin/env python3
"""Replay a BF16 projection packet through a freshly generated checked C entry."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.scripts.xray_operation_bundle_v8 import BundleError, load_bundle, sha256

PROVIDER = "linear_rows_bf16_checked"
CONTRACT = "projection.row_major.bf16.rne.checked.v1"


def projection_graph(rows, width, channels):
    sizes = {
        "input_elements": rows * width,
        "weight_elements": channels * width,
        "output_elements": rows * channels,
        "staged_elements": rows * channels,
        "rows": rows,
        "input_width": width,
        "output_width": channels,
    }
    op = {
        "id": "projection", "op": PROVIDER, "kernel": PROVIDER, "returns_status": True,
        "weight_refs": {"weight": "projection.weight"},
        "graph_slots": {"inputs": {"input": "external:input"}, "outputs": {"output": "output"}},
        "params": {"M": rows, "K": width, "N": channels, "call_constants": sizes},
    }
    return {
        "version": 3, "name": "xray_bf16_projection", "family": "bounded_graph",
        "checked_native_entry": True, "contract": {"runtime_invariants": {"inference_only": True}},
        "activation_buffers": {"input": {"shape": [rows, width]},
                               "output": {"shape": [rows, channels]}},
        "activation_bindings": {"input": "input", "output": "output"},
        "native_entry": {
            "function": "ck_xray_bf16_projection",
            "params": [{"c_type": "uint8_t *", "name": "arena"},
                       {"c_type": "size_t", "name": "arena_bytes"}],
            "arena": {"pointer": "arena", "bytes": "arena_bytes"},
        },
        "sequence": ["block"],
        "block_types": {"block": {
            "sequence": ["header", "body", "footer"], "header": [], "footer": [],
            "body": {"type": "dense", "ops": [op]},
        }},
    }


def replay(packet, output):
    manifest, tensors = load_bundle(packet)
    if (manifest["operation"] != PROVIDER or manifest["provider"] != PROVIDER
            or manifest["contract"] != CONTRACT):
        raise BundleError("packet does not declare the checked BF16 projection")
    expected = {
        "activation": ("bf16_bits", ["token", "feature"]),
        "weight": ("bf16_bits", ["channel", "feature"]),
        "cke": ("fp32", ["token", "channel"]),
        "oracle": ("fp32", ["token", "channel"]),
    }
    if set(tensors) != set(expected):
        raise BundleError("projection packet tensor roles differ from the ABI")
    for name, (dtype, axes) in expected.items():
        desc = manifest["tensors"][name]
        if desc["dtype"] != dtype or desc["axes"] != axes:
            raise BundleError(f"{name} storage or layout disagrees with the projection ABI")
    x, w, recorded, oracle = (tensors[name] for name in expected)
    rows, width = x.shape
    channels, weight_width = w.shape
    if (width != weight_width or recorded.shape != (rows, channels)
            or oracle.shape != recorded.shape or rows * width * channels > (1 << 27)):
        raise BundleError("projection geometry disagrees with the packet or exceeds the replay bound")

    output.mkdir(parents=True)
    graph = projection_graph(rows, width, channels)
    config = {"num_layers": 1, "embed_dim": width, "num_heads": 1, "num_kv_heads": 1,
              "head_dim": width, "intermediate_size": channels, "context_length": rows,
              "max_seq_len": rows, "activation_buffer_dtypes": {"input": "bf16", "output": "bf16"}}
    run, arena, view, _ = compile_checked_weight_graph(
        output, graph, {"projection.weight": w},
        {"projection.weight": ([channels, width], "BF16")}, source_config=config)
    view("input", np.uint16)[:] = x
    if run(arena, len(arena)):
        raise RuntimeError("generated projection failed")
    observed = (view("output", np.uint16).astype(np.uint32) << 16).view(np.float32)
    matches_capture = bool(np.array_equal(observed.view("u4"), recorded.view("u4")))
    exact = int(np.count_nonzero(observed.view("u4") == oracle.view("u4")))
    report = {
        "capture_reproduced": matches_capture,
        "oracle_exact_count": exact, "output_count": observed.size,
        "oracle_max_abs": float(np.max(np.abs(observed.astype(np.float64) - oracle))),
        "packet_sha256": sha256(packet), "new_runtime_sha256": sha256(output / "generated.so"),
    }
    (output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = replay(args.packet, args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["capture_reproduced"] and result["oracle_exact_count"] == result["output_count"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
