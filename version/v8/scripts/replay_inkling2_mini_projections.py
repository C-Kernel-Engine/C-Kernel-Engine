#!/usr/bin/env python3
"""Replay isolated Inkling layer-6 projections with ranged publisher weights."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from version.v8.scripts.build_inkling2_mini_projection_circuit import (  # noqa: E402
    SOURCE_REVISION, TRANSFORMERS_REVISION, attention_projections, controller_gates,
)
from version.v8.scripts.checked_circuit_authoring_v8 import declared_f32_weight_shapes  # noqa: E402
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph  # noqa: E402

SHARD = "model-00001-of-00001.safetensors"
HEADER_HASH = "4b1fa8297cb1a0f6484c707bef69e7f0f81d08325ccf2c5b25b86d528f64acf2"
PAYLOAD_BYTES = 19_525_870_752
SHARD_URL = f"https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/{SOURCE_REVISION}/{SHARD}?download=true"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_bf16(header, name, shape):
    info = json.loads(header)[name]
    start, end = info["data_offsets"]
    data_bytes = 2 * math.prod(shape)
    if (info["dtype"] != "BF16" or info["shape"] != shape
            or not 0 <= start < end <= PAYLOAD_BYTES or end - start != data_bytes
            or data_bytes > 16 * 1024 * 1024):
        raise ValueError(f"unexpected projection tensor: {name}")
    base = 8 + len(header)
    first, last = base + start, base + end - 1
    req = Request(SHARD_URL, headers={"Range": f"bytes={first}-{last}"})
    with urlopen(req, timeout=60) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {first}-{last}/{base + PAYLOAD_BYTES}":
            raise ValueError(f"publisher did not honor the pinned tensor range: {name}")
        payload = response.read(data_bytes + 1)
    if len(payload) != data_bytes:
        raise ValueError(f"short tensor response: {name}")
    weights = (np.frombuffer(payload, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if not np.isfinite(weights).all():
        raise ValueError(f"nonfinite weights: {name}")
    return weights, {"raw_bf16_sha256": digest(payload), "bytes": data_bytes,
                     "source_shape": shape, "shard_byte_range_inclusive": [first, last]}


def replay(folder, graph, weights, rng, torch):
    fn, arena, view, _ = compile_declared_f32_graph(folder, graph, weights)
    ops = graph["block_types"]["block"]["body"]["ops"]
    expected = {}
    for op in ops:
        source = op["graph_slots"]["inputs"]["input"]
        key = source.removeprefix("external:")
        if source.startswith("external:") and key not in expected:
            raw = rng.standard_normal(view(key).shape, dtype=np.float32) * np.float32(0.2)
            # Fixed synthetic operands use the same BF16-representable input on both paths.
            expected[key] = torch.from_numpy(raw).bfloat16().float()
            view(key)[:] = expected[key].numpy()
        if "weight_refs" in op:
            weight = torch.from_numpy(weights[op["weight_refs"]["weight"]])
            expected[op["id"]] = torch.nn.functional.linear(expected[key], weight)
        else:
            expected[op["id"]] = torch.tanh(expected[key])

    if fn(arena, len(arena)):
        raise RuntimeError("native projection entry failed")
    errors = {}
    for op in ops:
        name = op["id"]
        got, want = view(name), expected[name].numpy()
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=3e-5, err_msg=name)
        diff = got.astype("f8") - want.astype("f8")
        errors[name] = {"max_abs": float(abs(diff).max()), "rmse": float(np.sqrt(np.mean(diff * diff)))}
    identity = json.loads((folder / "build-identity.json").read_text())
    return {"tokens": graph["contract"]["runtime_invariants"]["tokens"], "checkpoints": errors,
            "loaded_build_identity_sha256": identity["build_identity_sha256"],
            "library_sha256": digest((folder / "generated.so").read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, required=True,
                        help="safetensors-header.json from the pinned metadata audit")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    header = args.header.read_bytes()
    if digest(header) != HEADER_HASH:
        parser.error("safetensors header differs from the pinned audit")
    args.output.mkdir(mode=0o700)

    import torch
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    report = {"status": "RUNNING", "checkpoint_revision": SOURCE_REVISION,
              "transformers_source_revision": TRANSFORMERS_REVISION,
              "header_sha256": HEADER_HASH, "shard": SHARD,
              "scope": "real BF16 weights expanded to FP32; synthetic BF16-representable inputs; FP32 projection and tanh oracle",
              "seed": 414, "physical_layer": 6, "oracle": f"PyTorch {torch.__version__} FP32 eager",
              "weights": {}, "results": {}}
    target = args.output / "report.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    rng = np.random.default_rng(report["seed"])
    try:
        for builder in (attention_projections, controller_gates):
            graph = builder()
            weights = {}
            for name, shape in declared_f32_weight_shapes(graph).items():
                weights[name], report["weights"][name] = read_bf16(header, name, shape)
            for tokens in (1, 8):
                folder = args.output / f"{builder.__name__}-{tokens}"
                folder.mkdir()
                report["results"][folder.name] = replay(folder, builder(tokens=tokens), weights, rng, torch)
        report["status"] = "PASS_COMPONENT_REPLAY"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
