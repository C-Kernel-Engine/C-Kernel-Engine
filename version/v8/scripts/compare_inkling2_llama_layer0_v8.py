#!/usr/bin/env python3
"""Compare an Inkling native layer replay with a pinned CPU llama.cpp capture."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from version.v8.scripts.convert_gguf_to_bump_v8 import GGML_TYPE_Q8_0, ggml_tensor_bytes
from version.v8.scripts.build_inkling2_mini_projection_circuit import routed_expert_mixture
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.scripts.replay_inkling2_routed_moe import (
    EXPERTS, GGUF_FILES, INNER, TOP_K, WIDTH, from_gguf, make_local_gguf_range_reader,
    read_gguf_header, unpack_gguf,
)


def distance(actual, expected):
    diff = actual.astype(np.float64) - expected
    if not np.isfinite(diff).all():
        raise ValueError("comparison contains nonfinite elements")
    return {"max_abs": float(np.abs(diff).max()), "rmse": float(np.sqrt(np.mean(diff**2)))}


def load(path, shape, dtype="<f4"):
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != size:
        raise ValueError(f"wrong capture size: {path}")
    return np.fromfile(path, dtype=dtype).reshape(shape)


def pack_reference(values):
    blocks = values.reshape(*values.shape[:-1], -1, 32)
    peak = np.max(np.abs(blocks), axis=-1)
    scale = (peak / np.float32(127)).astype("<f2")
    inverse = np.divide(np.float32(127), peak, where=scale != 0, out=np.zeros_like(peak))
    codes = np.rint(blocks * inverse[..., None]).clip(-127, 127).astype(np.int8)
    return scale, codes


def replay_captured_ffn(gguf, output, capture, expected_ids):
    """Invoke the generated MoE on llama.cpp's exact FFN operand."""
    file = GGUF_FILES["q8_0"]
    source = make_local_gguf_range_reader(gguf)
    tensors, base = read_gguf_header(source(file, 0, 16 * 1024**2)[0])
    rows = len(expected_ids)
    hidden = capture(".full.f32.ffn-input.f32", (rows, WIDTH))
    expected = capture(".full.f32.ffn_moe_out-0", (rows, WIDTH))
    active = sorted(set(np.asarray(expected_ids).flatten().tolist()))
    graph = routed_expert_mixture(rows, WIDTH, INNER, EXPERTS, TOP_K, len(active),
                                  layer=0, storage="q8_0", q8_activations=True)
    nodes = graph["block_types"]["block"]["body"]["ops"]
    (router, bias), _ = from_gguf(file, tensors, base, 0, read_range=source)
    router_name = nodes[0]["weight_refs"]["weight"]
    weights = {router_name: router}
    specs = {router_name: ([EXPERTS, WIDTH], "F32")}
    bank = [from_gguf(file, tensors, base, 0, expert, read_range=source)[0] for expert in active]
    for index, (port, name) in enumerate(nodes[-1]["weight_refs"].items()):
        weights[name] = np.concatenate([slot[index] for slot in bank])
        matrix = [WIDTH, INNER] if port == "down" else [INNER, WIDTH]
        specs[name] = ([len(active), *matrix], "Q8_0")
    output.mkdir(mode=0o700)
    run, arena, view, _ = compile_checked_weight_graph(output, graph, weights, specs)
    view("hidden")[:] = hidden
    view("correction_bias")[:] = bias
    mapping = view("expert_slots", np.int32)
    mapping.fill(-1)
    mapping[active] = np.arange(len(active), dtype=np.int32)
    if run(arena, len(arena)):
        raise RuntimeError("native MoE rejected captured FFN input")
    identity = json.loads((output / "build-identity.json").read_text())
    return {
        "route_ids_exact": np.array_equal(view("indices", np.int32), expected_ids),
        "output_vs_llama": distance(view("mixture"), expected),
        "input_sha256": hashlib.sha256(hidden.tobytes()).hexdigest(),
        "build_identity_sha256": identity["build_identity_sha256"],
        "library_sha256": identity["library_sha256"],
    }


def compare(gguf, case, llama_prefix, *, ffn_replay=None):
    def capture(suffix, shape, dtype="<f4"):
        return load(llama_prefix.with_name(llama_prefix.name + suffix), shape, dtype)

    native = load(case / "layer-output.f32", (5, 2304))
    source = json.loads((case / "report.json").read_text())
    replay_input = load(case / "hidden.f32", (5, 2304))
    if (hashlib.sha256(replay_input.tobytes()).hexdigest() != source["input_sha256"]
            or not np.array_equal(replay_input, capture(".input.f32", (5, 2304)))):
        raise ValueError("llama.cpp and CKE inputs do not match")
    llama_full = capture(".full.f32", (5, 2304))
    llama_split = np.concatenate((capture(".prefill.f32", (4, 2304)), capture(".append.f32", (1, 2304))))
    native_routes = source["exact_route_ids"]
    llama_routes = capture(".full.f32.ffn_moe_topk-0", (5, 8), "<i4").tolist()
    swaps = [{"row": row, "native_only": sorted(set(a) - set(b)), "llama_only": sorted(set(b) - set(a))}
             for row, (a, b) in enumerate(zip(native_routes, llama_routes)) if set(a) != set(b)]

    with gguf.open("rb") as f:
        tensors, data_start = read_gguf_header(f.read(16 * 1024**2))
        info = tensors["blk.0.attn_q.weight"]
        if info.dims != (2304, 3072) or info.ggml_type != GGML_TYPE_Q8_0:
            raise ValueError("projection does not have the pinned Q8_0 layout")
        f.seek(data_start + info.offset)
        payload = f.read(ggml_tensor_bytes(info))
    if hashlib.sha256(payload).hexdigest() != source["source"]["ranges"][info.name]["sha256"]:
        raise ValueError("query weights differ from the native replay")
    packed = np.frombuffer(payload, np.uint8)
    query_input = capture(".full.f32.q-input.f32", (5, 2304))
    llama_dot = capture(".full.f32.q-dot.f32", (5, 3072))
    f32 = query_input @ unpack_gguf(packed, (3072, 2304), GGML_TYPE_Q8_0).T

    blocks = packed.reshape(3072, 72, 34)
    weight_scale = blocks[:, :, :2].copy().view("<f2").reshape(3072, 72).astype(np.float32)
    weight_codes = blocks[:, :, 2:].view(np.int8)
    scale, codes = pack_reference(query_input)
    products = np.einsum("tbi,obi->tob", codes, weight_codes, dtype=np.int32)
    quant = products * (weight_scale[None, :, :] * scale.astype(np.float32)[:, None, :])
    q8 = quant.astype(np.float32).sum(axis=-1)
    result = {
        "native_vs_llama": distance(native, llama_full),
        "llama_split_vs_full": distance(llama_split, llama_full),
        "native_vs_llama_max_abs_by_row": np.abs(native.astype(np.float64) - llama_full).max(axis=1).tolist(),
        "changed_route_membership": swaps,
        "llama_routes": llama_routes,
        "identical_query_input": {
            "sha256": hashlib.sha256(query_input.tobytes()).hexdigest(),
            "packed_weight_sha256": hashlib.sha256(payload).hexdigest(),
            "f32_activation_vs_llama": distance(f32, llama_dot),
            "q8_0_activation_vs_llama": distance(q8, llama_dot),
        },
    }
    trace_path = case / "native-trace.npz"
    if trace_path.is_file():
        boundaries = {
            "attn_norm": ".full.f32.q-input.f32",
            "query_raw": ".full.f32.q-dot.f32",
            "gated_attention": ".full.f32.o-input.f32",
            "projected_attention": ".full.f32.o-dot.f32",
            "attn_post_conv": ".full.f32.ffn_inp-0",
            "ffn_norm": ".full.f32.ffn-input.f32",
            "mixture": ".full.f32.ffn_moe_out-0",
            "ffn_post_conv": ".full.f32.ffn_out-0",
        }
        with np.load(trace_path) as trace:
            result["captured_boundaries"] = {
                name: distance(trace[name], capture(suffix, trace[name].shape))
                for name, suffix in boundaries.items()
            }
            result["activation_pack_differences"] = {}
            for name in ("gated_attention", "ffn_norm"):
                values = trace[name]
                native_scale, native_codes = pack_reference(values)
                ref_scale, ref_codes = pack_reference(capture(boundaries[name], values.shape))
                result["activation_pack_differences"][name] = {
                    "scale_blocks_changed": int(np.count_nonzero(native_scale.view("<u2") != ref_scale.view("<u2"))),
                    "codes_changed": int(np.count_nonzero(native_codes != ref_codes)),
                    "blocks": int(native_scale.size),
                }
    if ffn_replay:
        result["identical_ffn_input"] = replay_captured_ffn(gguf, ffn_replay, capture, llama_routes)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--native-replay", type=Path, required=True)
    parser.add_argument("--llama-prefix", type=Path, required=True)
    parser.add_argument("--ffn-replay", type=Path, help="new output directory for identical-input native MoE")
    args = parser.parse_args()
    print(json.dumps(compare(args.gguf, args.native_replay, args.llama_prefix, ffn_replay=args.ffn_replay), indent=2))


if __name__ == "__main__":
    main()
