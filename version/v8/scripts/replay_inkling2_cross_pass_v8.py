#!/usr/bin/env python3
"""Replay real GGUF cross-pass controller weights from bounded synthetic states."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from version.v8.scripts.build_inkling2_cross_pass_v8 import CONTROL_PORTS, control_weight_shapes, cross_pass_controls
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.scripts.replay_inkling2_routed_moe import (
    GGML_TYPE_F32, GGUF_FILES, GGUF_REPO, GGUF_REVISION, fetch_gguf_range, gguf_tensor, read_gguf_header,
)


def replay(torch, output, layer, file, tensors, base):
    graph = cross_pass_controls(layer=layer)
    declarations = {name: (shape, "F32") for name, shape in control_weight_shapes(graph).items()}
    weights, ranges = {}, {}
    query_name = f"blk.{layer}.loop_msg_query.weight"
    raw, ranges[query_name] = gguf_tensor(file, tensors, base, query_name, (2304, 4), GGML_TYPE_F32)
    queries = np.frombuffer(raw, "<f4").reshape(4, -1)
    for index, port in enumerate(CONTROL_PORTS):
        weights[f"blk.{layer}.loop_msg_query.{port}"] = queries[index]

    def load(item):
        name, (shape, _) = item
        data, proof = gguf_tensor(file, tensors, base, name, tuple(reversed(shape)), GGML_TYPE_F32)
        return name, np.frombuffer(data, "<f4").reshape(shape), proof

    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, weight, proof in pool.map(load, ((n, spec) for n, spec in declarations.items() if n not in weights)):
            weights[name], ranges[name] = weight, proof

    output.mkdir()
    fn, arena, view, _ = compile_checked_weight_graph(output, graph, weights, declarations)
    rng = np.random.default_rng(143)
    states = {name: torch.from_numpy(rng.standard_normal(view(name).shape, dtype=np.float32) * 0.2)
              for name in ("prior_bank", "prior_output", "attention_input", "ffn_input")}
    for name, value in states.items():
        view(name)[:] = value.numpy()
    if fn(arena, len(arena)):
        raise RuntimeError("native generated control entry rejected valid rows")
    F = torch.nn.functional

    def norm(x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)

    bank = norm(torch.cat((states["prior_bank"], states["prior_output"][:, None]), dim=1))
    comparison, first = {}, {}

    def check(name, target):
        actual = view(name)
        expected = target.numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=4e-5, err_msg=name)
        comparison[name] = float(np.abs(actual - expected).max())

    for index, port in enumerate(CONTROL_PORTS):
        query = torch.tensor(queries[index])
        probabilities = (bank @ query / np.sqrt(2304)).softmax(dim=-1)
        message = (bank * probabilities[..., None]).sum(dim=1)
        source = states["ffn_input" if port == "ffn" else "attention_input"]
        joined = torch.cat((norm(source), norm(message)), dim=-1)
        stem = f"blk.{layer}.loop_{port}_ctrl_"
        down = F.linear(joined, torch.tensor(weights[stem + "down.weight"]))
        forget = F.linear(down, torch.tensor(weights[stem + "forget.weight"])).tanh()
        gain = F.linear(down, torch.tensor(weights[stem + "gain.weight"])).tanh()
        controlled = (1 - forget) * source + gain * message
        for name, value in (("message", message), ("join", joined), ("down", down),
                            ("forget", forget), ("gain", gain), ("controlled", controlled)):
            check(f"{port}_{name}", value)
        first[port] = view(port + "_controlled").copy()

    for name in ("prior_bank", "prior_output"):
        view(name)[:] = states[name].numpy() * -1
    if fn(arena, len(arena)):
        raise RuntimeError("alternate bank unexpectedly rejected")
    effect = {port: float(np.abs(first[port] - view(port + "_controlled")).max()) for port in CONTROL_PORTS}
    if min(effect.values()) < 1e-3:
        raise ValueError("wrong-bank control did not affect every controller")
    for name in ("prior_bank", "prior_output"):
        view(name)[:] = states[name].numpy()
    if fn(arena, len(arena)):
        raise RuntimeError("restored bank rejected")
    for port, target in first.items():
        np.testing.assert_array_equal(view(port + "_controlled"), target)

    identity = json.loads((output / "build-identity.json").read_text())
    return {
        "physical_layer": layer, "status": "PASS", "seed": 143, "rows": 4, "history_states": 1,
        "max_abs_by_checkpoint": comparison, "wrong_bank_max_abs_by_port": effect, "aba_exact": True,
        "ranges": ranges, "build_identity_sha256": identity["build_identity_sha256"],
        "library_sha256": identity["library_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    file = GGUF_FILES["q8_0"]
    header, proof = fetch_gguf_range(file, 0, 16 * 1024 * 1024)
    tensors, base = read_gguf_header(header)
    args.output.mkdir(mode=0o700)
    cases = [replay(torch, args.output / f"layer-{layer}", layer, file, tensors, base) for layer in (6, 11)]
    report = {"scope": "four cross-pass controls with synthetic banks and inputs; no downstream attention or model loop",
              "oracle": f"PyTorch {torch.__version__} FP32",
              "source": {"repository": GGUF_REPO, "revision": GGUF_REVISION, "file": file[0], "header": proof},
              "cases": cases}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
