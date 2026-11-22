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
from version.v8.scripts.replay_inkling2_routed_moe import read_gguf_header, unpack_gguf


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


def compare(gguf, case, llama_prefix):
    def capture(suffix, shape, dtype="<f4"):
        return load(llama_prefix.with_name(llama_prefix.name + suffix), shape, dtype)

    native = load(case / "layer-output.f32", (5, 2304))
    source = json.loads((case / "report.json").read_text())
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
    x = query_input.reshape(5, 72, 32)
    scale = np.max(np.abs(x), axis=-1) / np.float32(127)
    recip = np.divide(np.float32(1), scale, where=scale != 0, out=np.zeros_like(scale))
    codes = np.rint(x * recip[..., None]).clip(-127, 127).astype(np.int8)
    products = np.einsum("tbi,obi->tob", codes, weight_codes, dtype=np.int32)
    quant = (products * (weight_scale[None, :, :] * scale.astype("<f2").astype(np.float32)[:, None, :]))
    q8 = quant.astype(np.float32).sum(axis=-1)
    return {
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--native-replay", type=Path, required=True)
    parser.add_argument("--llama-prefix", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.gguf, args.native_replay, args.llama_prefix), indent=2))


if __name__ == "__main__":
    main()
