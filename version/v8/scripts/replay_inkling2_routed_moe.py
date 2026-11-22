#!/usr/bin/env python3
"""Replay native Inkling MoE with publisher weights or packed GGUF expert ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from version.v8.scripts.build_inkling2_mini_projection_circuit import routed_expert_mixture
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.scripts.convert_gguf_to_bump_v8 import (
    GGUFReader, TensorInfo, GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
    _gguf_read_value, _gguf_skip_value, align_up, ggml_row_bytes,
)
from version.v8.scripts.inkling2_expert_weights_v8 import decode_expert_rows
from version.v8.scripts.replay_inkling2_expert_weights import read_tile, torch_reference
from version.v8.scripts.replay_inkling2_mini_projections import HEADER_HASH, SOURCE_REVISION, digest, read_bf16

GGUF_REPO = "ggml-org/Inkling-2-Mini-GGUF"
GGUF_REVISION = "e8a2cecf0e78914f07cc27b9be711a9b2c7561d9"
GGUF_FILES = {
    "q8_0": ("Inkling-2-Mini-Q8_0.gguf", 27_820_099_264),
    "iq4_nl": ("Inkling-2-Mini-IQ4_NL.gguf", 17_627_940_768),
}
PACKED_FORMATS = {GGML_TYPE_Q8_0: "q8_0", GGML_TYPE_IQ4_NL: "iq4_nl"}
IQ4_LEVELS = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32,
)
WIDTH, INNER, EXPERTS, TOP_K = 2304, 1152, 128, 8


def fetch_gguf_range(file, first, count):
    name, size = file
    if not 0 < count <= 16 * 1024 * 1024 or not 0 <= first <= size - count:
        raise ValueError("invalid bounded GGUF range")
    last = first + count - 1
    url = f"https://huggingface.co/{GGUF_REPO}/resolve/{GGUF_REVISION}/{name}"
    with urlopen(Request(url, headers={"Range": f"bytes={first}-{last}"}), timeout=60) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {first}-{last}/{size}":
            raise ValueError("publisher did not honor the pinned GGUF range")
        raw = response.read(count + 1)
    if len(raw) != count:
        raise ValueError("wrong GGUF range byte count")
    return raw, {"sha256": digest(raw), "range_inclusive": [first, last]}


def read_gguf_header(header):
    r = GGUFReader(io.BytesIO(header))
    if r._read_exact(4) != b"GGUF" or r.u32() != 3:
        raise ValueError("expected GGUF v3")
    n_tensor, n_meta = r.u64(), r.u64()
    if not 0 < n_tensor <= 4096 or not 0 < n_meta <= 4096:
        raise ValueError("unexpected GGUF header sizes")
    meta = {}
    for _ in range(n_meta):
        name, kind = r.key_str(), r.u32()
        if name in {"general.architecture", "general.alignment"}:
            meta[name] = _gguf_read_value(r, kind)
        else:
            _gguf_skip_value(r, kind)
    if meta.get("general.architecture") != "inkling2_loop":
        raise ValueError("GGUF architecture is not inkling2_loop")
    alignment = meta.get("general.alignment", 32)
    if type(alignment) is not int or not 0 < alignment <= 4096 or alignment & (alignment - 1):
        raise ValueError("invalid GGUF data alignment")
    tensors = {}
    for _ in range(n_tensor):
        name, rank = r.key_str(), r.u32()
        if name in tensors or not 1 <= rank <= 4:
            raise ValueError("invalid GGUF tensor identity or rank")
        dims = tuple(r.u64() for _ in range(rank))
        kind, offset = r.u32(), r.u64()
        if not all(dims) or offset % alignment:
            raise ValueError("invalid GGUF tensor range")
        tensors[name] = TensorInfo(name, dims, kind, offset)
    start = align_up(r.tell(), alignment)
    return tensors, start


def gguf_tensor(file, tensors, data_start, name, dims, kind, expert=None):
    info = tensors[name]
    if info.dims != dims or info.ggml_type != kind:
        raise ValueError(f"unexpected GGUF shape or dtype: {name}")
    row_bytes = ggml_row_bytes(kind, dims[0])
    count = row_bytes * (dims[1] if len(dims) > 1 else 1)
    offset = data_start + info.offset
    if expert is not None:
        if len(dims) != 3 or not 0 <= expert < dims[2]:
            raise ValueError("expert outside GGUF tensor")
        offset += expert * count
    raw, report = fetch_gguf_range(file, offset, count)
    return raw, dict(report, name=name, ggml_type=kind)


def unpack_gguf(packed, shape, dtype):
    blocks = packed.reshape(-1, 34 if dtype == GGML_TYPE_Q8_0 else 18)
    scales = blocks[:, :2].copy().view("<f2").astype(np.float32)
    codes = blocks[:, 2:]
    if dtype == GGML_TYPE_Q8_0:
        values = codes.view(np.int8)
    elif dtype == GGML_TYPE_IQ4_NL:
        values = IQ4_LEVELS[np.concatenate((codes % 16, codes // 16), axis=-1)]
    else:
        raise ValueError("unsupported reference expert dtype")
    return (scales * values).reshape(shape)


def gguf_expert_type(tensors, layer):
    prefix = f"blk.{layer}."
    kind = tensors[prefix + "ffn_gate_up_exps.weight"].ggml_type
    if kind not in PACKED_FORMATS or tensors[prefix + "ffn_down_exps.weight"].ggml_type != kind:
        raise ValueError("layer has no supported uniform expert tensor dtype")
    return kind


def from_gguf(file, tensors, data_start, layer, expert=None):
    prefix = f"blk.{layer}."

    def tensor(name, dims, kind):
        return gguf_tensor(file, tensors, data_start, prefix + name, dims, kind, expert)

    if expert is None:
        router, r = tensor("ffn_gate_inp.weight", (WIDTH, EXPERTS), GGML_TYPE_F32)
        bias, b = tensor("exp_probs_b.bias", (EXPERTS,), GGML_TYPE_F32)
        return (np.frombuffer(router, "<f4").reshape(EXPERTS, WIDTH), np.frombuffer(bias, "<f4")), [r, b]
    dtype = gguf_expert_type(tensors, layer)
    gate_up, gu = tensor("ffn_gate_up_exps.weight", (WIDTH, 2 * INNER, EXPERTS), dtype)
    down, d = tensor("ffn_down_exps.weight", (INNER, WIDTH, EXPERTS), dtype)
    gate_up = np.frombuffer(gate_up, np.uint8)
    gate, up = np.split(gate_up, 2)
    packed = (gate, up, np.frombuffer(down, np.uint8))
    oracle = tuple(unpack_gguf(data, shape, dtype) for data, shape in
                   zip(packed, ((INNER, WIDTH), (INNER, WIDTH), (WIDTH, INNER))))
    return packed, oracle, [gu, d]


def from_publisher(header, torch, layer, expert=None):
    prefix = f"model.llm.layers.{layer}.mlp."
    if expert is None:
        name = prefix + "gate.weight"
        weight, w = read_bf16(header, name, [EXPERTS, WIDTH])
        bias, b = read_tile(header, prefix + "gate.bias", (), (EXPERTS,), "F32")
        return (weight, bias), [dict(w, name=name), dict(b, name=prefix + "gate.bias")]
    source_format = "nvfp4" if layer == 6 else "mxfp8"
    is_fp4 = source_format == "nvfp4"
    reports = []

    def decode(kind, rows, columns):
        name = prefix + f"experts.{kind}_weight"

        def read(suffix, index, shape, dtype):
            value, record = read_tile(header, name + suffix, index, shape, dtype)
            reports.append(dict(record, name=name + suffix))
            return value

        raw = read("", (expert,), (rows, columns // 2 if is_fp4 else columns), "U8" if is_fp4 else "F8_E4M3")
        scales = read("_scale", (expert,), (rows, columns // (16 if is_fp4 else 32)), "F8_E4M3" if is_fp4 else "U8")
        amax = read("_amax", (expert,), (), "F32").item() if is_fp4 else None
        denominator = read("_outer_scale_denominator", (), (1,), "F32")[0] if is_fp4 else None
        expanded = decode_expert_rows(raw, scales, format_name=source_format, amax=amax, denominator=denominator)
        reference = torch_reference(torch, raw, scales, source_format, amax, denominator).numpy()
        np.testing.assert_array_equal(expanded, reference)
        return expanded, reference

    gu, ref_gu = decode("w13", 2 * INNER, WIDTH)
    down, ref_down = decode("w2", WIDTH, INNER)
    return (gu[::2].copy(), gu[1::2].copy(), down), (ref_gu[::2], ref_gu[1::2], ref_down), reports


def replay(output, torch, load, layer, storage):
    (router, bias), router_sources = load(layer)
    seed = 384
    rng = np.random.default_rng(seed)
    hidden = torch.from_numpy(rng.standard_normal((1, WIDTH), dtype=np.float32)).bfloat16().float() * .5
    scores = torch.nn.functional.linear(hidden, torch.tensor(router)).sigmoid()
    selected = (scores + torch.tensor(bias)).topk(TOP_K + 1, dim=-1)
    ids = selected.indices[:, :-1]
    margin = (selected.values[0, -2] - selected.values[0, -1]).item()
    if margin <= 0:
        raise ValueError("ambiguous oracle top-k boundary")
    raw_weights = scores.gather(1, ids)
    route_weights = raw_weights / (raw_weights.sum(-1, keepdim=True) + 1e-20) * 8
    expert_ids = sorted(ids[0].tolist())
    expert_slots = np.full(EXPERTS, -1, dtype=np.int32)
    expert_slots[expert_ids] = np.arange(TOP_K, dtype=np.int32)

    # Pin only the experts this reference row selects. Native routing still
    # computes every ID and rejects if it selects anything outside the bank.
    with ThreadPoolExecutor(max_workers=4) as executor:
        experts = list(executor.map(lambda expert: load(layer, expert), expert_ids))
    graph = routed_expert_mixture(1, WIDTH, INNER, EXPERTS, TOP_K, TOP_K, layer=layer, storage=storage)
    ops = graph["block_types"]["block"]["body"]["ops"]
    router_name = ops[0]["weight_refs"]["weight"]
    weights = {router_name: router}
    declarations = {router_name: ([EXPERTS, WIDTH], "F32")}
    for i, (port, name) in enumerate(ops[-1]["weight_refs"].items()):
        shape = [TOP_K, *([WIDTH, INNER] if port == "down" else [INNER, WIDTH])]
        if storage != "f32":
            weights[name] = np.concatenate([matrices[i] for matrices, _, _ in experts])
        else:
            weights[name] = np.stack([matrices[i] for matrices, _, _ in experts])
        declarations[name] = (shape, storage.upper() if storage != "f32" else "F32")
    run, arena, view, _ = compile_checked_weight_graph(output, graph, weights, declarations)
    view("hidden")[:] = hidden.numpy()
    view("correction_bias")[:] = bias
    view("expert_slots", np.int32)[:] = expert_slots
    if run(arena, len(arena)):
        raise RuntimeError("generated route-to-mixture entry rejected replay")
    np.testing.assert_array_equal(view("indices", np.int32), ids.numpy())
    mixed = torch.zeros_like(hidden)
    outputs = torch.empty((TOP_K, WIDTH), dtype=torch.float32)
    for expert_id, (_, (gate, up, down), _) in zip(expert_ids, experts):
        route = ids[0].tolist().index(expert_id)
        g = torch.nn.functional.linear(hidden, torch.tensor(gate))
        u = torch.nn.functional.linear(hidden, torch.tensor(up))
        activation = (4 * (g / 4).tanh() * g.sigmoid()) * (25 * (u / 25).tanh())
        contribution = torch.nn.functional.linear(activation, torch.tensor(down))
        outputs[route] = contribution[0]
        mixed += contribution * route_weights[0, route]

    errors = {}
    for name, tensor in (("route_weights", route_weights), ("expert_output", outputs), ("mixture", mixed)):
        expected = tensor.numpy()
        np.testing.assert_allclose(view(name), expected, rtol=4e-4, atol=5e-5, err_msg=name)
        errors[name] = float(np.max(np.abs(view(name) - expected)))
    before = view("mixture").copy()
    view("expert_slots", np.int32)[expert_ids[0]] = -1
    if not run(arena, len(arena)):
        raise AssertionError("missing selected expert was accepted")
    np.testing.assert_array_equal(view("mixture"), before)
    view("expert_slots", np.int32)[:] = expert_slots
    if run(arena, len(arena)):
        raise RuntimeError("generated layer did not recover")
    np.testing.assert_array_equal(view("mixture"), before)
    identity = json.loads((output / "build-identity.json").read_text())
    return {
        "status": "PASS", "layer": layer, "storage": storage, "seed": seed, "tokens": 1,
        "minimum_top_k_margin": margin, "route_ids": ids[0].tolist(), "max_abs_errors": errors,
        "missing_expert_rejection_and_recovery": "PASS",
        "router_sources": router_sources,
        "expert_sources": {str(i): expert[2] for i, expert in zip(expert_ids, experts)},
        "build_identity_sha256": identity["build_identity_sha256"],
        "library_sha256": digest((output / "generated.so").read_bytes()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--publisher-header", type=Path)
    source.add_argument("--gguf", choices=sorted(GGUF_FILES))
    parser.add_argument("--layer", type=int, choices=(6, 20), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    if args.publisher_header:
        header = args.publisher_header.read_bytes()
        if digest(header) != HEADER_HASH:
            parser.error("publisher header differs from the pinned audit")

        def load(layer, expert=None):
            return from_publisher(header, torch, layer, expert)

        source = {"revision": SOURCE_REVISION, "header_sha256": HEADER_HASH,
                  "serialized_expert_format": "NVFP4" if args.layer == 6 else "MXFP8"}
        storage = "f32"
    else:
        file = GGUF_FILES[args.gguf]
        header, record = fetch_gguf_range(file, 0, 16 * 1024 * 1024)
        tensors, data_start = read_gguf_header(header)

        def load(layer, expert=None):
            return from_gguf(file, tensors, data_start, layer, expert)

        source = {"repo": GGUF_REPO, "revision": GGUF_REVISION, "file": file[0], "header_range": record}
        storage = PACKED_FORMATS[gguf_expert_type(tensors, args.layer)]
    args.output.mkdir(mode=0o700)
    report = replay(args.output, torch, load, args.layer, storage)
    report.update(source=source, oracle=f"PyTorch {torch.__version__} FP32",
                  scope="routed MoE on synthetic hidden rows, one bounded checkpoint expert bank")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
