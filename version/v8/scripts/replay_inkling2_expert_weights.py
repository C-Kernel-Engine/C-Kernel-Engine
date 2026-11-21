#!/usr/bin/env python3
"""Replay selected real NVFP4 and MXFP8 Inkling experts through generated FP32 C."""
import argparse
import json
import math
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from version.v8.scripts.build_inkling2_mini_projection_circuit import selected_expert
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph
from version.v8.scripts.inkling2_expert_weights_v8 import decode_expert_rows, split_gate_up_rows
from version.v8.scripts.replay_inkling2_mini_projections import (
    HEADER_HASH, PAYLOAD_BYTES, SHARD_URL, SOURCE_REVISION, digest,
)


def read_tile(header, name, index, shape, dtype):
    entry = json.loads(header)[name]
    tensor_shape = entry["shape"]
    start, end = entry["data_offsets"]
    itemsize = 4 if dtype == "F32" else 1
    if (entry["dtype"] != dtype or len(index) + len(shape) != len(tensor_shape)
            or any(not 0 <= coord < size for coord, size in zip(index, tensor_shape))
            or any(not 0 < count <= size for count, size in zip(shape, tensor_shape[len(index):]))
            or list(shape[1:]) != tensor_shape[len(index) + 1:]
            or not 0 <= start < end <= PAYLOAD_BYTES
            or end - start != math.prod(tensor_shape) * itemsize):
        raise ValueError(f"invalid expert extent: {name}")
    offset = sum(coord * math.prod(tensor_shape[axis + 1:]) for axis, coord in enumerate(index))
    size = math.prod(shape) * itemsize
    if size > 16 * 1024 * 1024:
        raise ValueError(f"expert tile exceeds the read limit: {name}")
    base = 8 + len(header)
    first = base + start + offset * itemsize
    last = first + size - 1
    req = Request(SHARD_URL, headers={"Range": f"bytes={first}-{last}"})
    with urlopen(req, timeout=60) as response:
        if (response.status != 206
                or response.headers.get("Content-Range") != f"bytes {first}-{last}/{base + PAYLOAD_BYTES}"):
            raise ValueError(f"wrong HTTP range for expert tile: {name}")
        raw = response.read(size + 1)
    if len(raw) != size:
        raise ValueError(f"wrong HTTP extent for expert tile: {name}")
    return np.frombuffer(raw, "<f4" if dtype == "F32" else "u1").reshape(shape), {
        "sha256": digest(raw), "range_inclusive": [first, last],
    }


def torch_reference(torch, weights, scales, mode, amax, denominator):
    def as_byte_tensor(values):
        return torch.tensor(values, dtype=torch.uint8)

    stored = as_byte_tensor(weights)
    scale_codes = as_byte_tensor(scales)
    if mode == "nvfp4":
        # Construct each alternating column from the independently tabulated E2M1 values.
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
        values = torch.empty((weights.shape[0], 2 * weights.shape[1]), dtype=torch.float32)
        values[:, ::2] = lut[stored.remainder(16).long()]
        values[:, 1::2] = lut[stored.div(16, rounding_mode="floor").long()]
        scale = scale_codes.view(torch.float8_e4m3fn).float()
        return (values * scale.repeat_interleave(16, dim=1)) * (
            torch.tensor(float(amax)) / torch.tensor(float(denominator))
        )
    return stored.view(torch.float8_e4m3fn).float() * torch.exp2(
        scale_codes.float() - 127
    ).repeat_interleave(32, dim=1)


def replay_tile(header, torch, layer, expert, matrix, width, mode, rows):
    name = f"model.llm.layers.{layer}.mlp.experts.{matrix}_weight"
    sources = {}

    def read(suffix, index, shape, dtype):
        result, metadata = read_tile(header, name + suffix, index, shape, dtype)
        sources[name + suffix] = metadata
        return result

    is_fp4 = mode == "nvfp4"
    weights = read("", (expert,), (rows, width // 2 if is_fp4 else width), "U8" if is_fp4 else "F8_E4M3")
    scales = read("_scale", (expert,), (rows, width // (16 if is_fp4 else 32)), "F8_E4M3" if is_fp4 else "U8")
    amax = read("_amax", (), (expert + 1,), "F32")[-1] if is_fp4 else None
    denominator = read("_outer_scale_denominator", (), (1,), "F32")[0] if is_fp4 else None
    actual = decode_expert_rows(weights, scales, format_name=mode, amax=amax, denominator=denominator)
    expected = torch_reference(torch, weights, scales, mode, amax, denominator).numpy()
    np.testing.assert_array_equal(actual, expected)
    controls = {}
    if matrix == "w13":
        gate, up = split_gate_up_rows(actual)
        np.testing.assert_array_equal(gate, expected[::2])
        np.testing.assert_array_equal(up, expected[1::2])
        controls["swapped_gate_up_changed_elements"] = int(np.count_nonzero(gate != up))
    if is_fp4:
        wrong = decode_expert_rows(weights, scales, format_name=mode, amax=1, denominator=1)
        controls["dropped_outer_scale_changed_elements"] = int(np.count_nonzero(wrong != expected))
    else:
        wrong = decode_expert_rows(weights, np.tile(scales[0], (rows, 1)), format_name=mode)
        controls["broadcast_row_scales_changed_elements"] = int(np.count_nonzero(wrong != expected))
    if not all(controls.values()):
        raise AssertionError(f"insensitive negative control: {name}")
    return actual, expected, {"layer": layer, "expert": expert, "matrix": matrix, "format": mode,
                              "shape": list(actual.shape), "equal_elements": actual.size,
                              "controls": controls, "source_ranges": sources}


def replay_expert(output, header, torch, layer, expert, mode):
    actual_gate_up, reference_gate_up, gate_up_report = replay_tile(
        header, torch, layer, expert, "w13", 2304, mode, 2304,
    )
    actual_down, reference_down, down_report = replay_tile(
        header, torch, layer, expert, "w2", 1152, mode, 2304,
    )
    gate, up = split_gate_up_rows(actual_gate_up)
    graph = selected_expert(layer=layer, expert=expert)
    ops = graph["block_types"]["block"]["body"]["ops"]
    weights = {op["weight_refs"]["weight"]: value
               for op, value in zip((ops[0], ops[1], ops[3]), (gate, up, actual_down))}
    folder = output / f"layer-{layer}-expert-{expert}"
    folder.mkdir()
    run, arena, view, _ = compile_declared_f32_graph(folder, graph, weights)
    rng = np.random.default_rng(703)
    hidden = rng.standard_normal((3, 2304), dtype=np.float32)
    hidden *= np.array([0.25, 1.0, 2.0], dtype=np.float32)[:, None]
    hidden = torch.from_numpy(hidden).bfloat16().float()
    view("hidden")[:] = hidden.numpy()
    if run(arena, len(arena)):
        raise RuntimeError("generated expert entry failed")

    gate_ref, up_ref = torch.nn.functional.linear(
        hidden, torch.from_numpy(reference_gate_up)
    ).reshape(3, 1152, 2).unbind(dim=-1)
    activation = (gate_ref.div(4.0).tanh() * 4.0 * gate_ref.sigmoid()) * (up_ref.div(25.0).tanh() * 25.0)
    down_ref = torch.nn.functional.linear(activation, torch.from_numpy(reference_down))
    checkpoints = {}
    for name, ref in (("gate", gate_ref), ("up", up_ref), ("activated", activation), ("down", down_ref)):
        target = ref.numpy()
        actual = view(name)
        np.testing.assert_allclose(actual, target, rtol=4e-4, atol=5e-5, err_msg=name)
        delta = actual.astype(np.float64) - target
        checkpoints[name] = {"max_abs": float(np.abs(delta).max()), "rmse": float(np.sqrt(np.mean(delta**2)))}
    capped_gate = gate_ref.div(4.0).tanh() * 4.0
    wrong = (capped_gate * capped_gate.sigmoid()) * (up_ref.div(25.0).tanh() * 25.0)
    control = (wrong - activation).abs().max().item()
    if control < 1e-3:
        raise AssertionError("capped-gate sigmoid control is insensitive")
    identity = json.loads((folder / "build-identity.json").read_text())
    return {
        "layer": layer, "expert": expert, "format": mode,
        "weights": [gate_up_report, down_report], "checkpoints": checkpoints,
        "capped_gate_sigmoid_max_abs_error": control,
        "build_identity_sha256": identity["build_identity_sha256"],
        "library_sha256": digest((folder / "generated.so").read_bytes()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    header = args.header.read_bytes()
    if digest(header) != HEADER_HASH:
        parser.error("safetensors header differs from the pinned audit")
    import torch

    torch.set_num_threads(2)
    args.output.mkdir(mode=0o700)
    cases = [(6, 7, "nvfp4"), (20, 11, "mxfp8")]
    results = [replay_expert(args.output, header, torch, *case) for case in cases]
    report = {
        "checkpoint_revision": SOURCE_REVISION, "header_sha256": HEADER_HASH,
        "oracle": f"PyTorch {torch.__version__} FP32 storage expansion",
        "scope": "two selected experts on synthetic rows; FP32 expansion and execution; routing and activation quantization not tested",
        "status": "PASS", "results": results,
    }
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
