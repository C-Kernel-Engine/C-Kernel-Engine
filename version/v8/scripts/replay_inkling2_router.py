#!/usr/bin/env python3
"""Replay real Inkling router weights through generated C and an FP32 oracle."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from version.v8.scripts.build_inkling2_mini_projection_circuit import expert_router
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph
from version.v8.scripts.replay_inkling2_expert_weights import read_tile
from version.v8.scripts.replay_inkling2_mini_projections import HEADER_HASH, SOURCE_REVISION, digest, read_bf16


def replay(header, output, torch, layer):
    prefix = f"model.llm.layers.{layer}.mlp.gate."
    weight, weight_source = read_bf16(header, prefix + "weight", [128, 2304])
    bias, bias_source = read_tile(header, prefix + "bias", (), (128,), "F32")
    tokens = 8
    scale = 8.0
    graph = expert_router(tokens, layer=layer, route_scale=scale)
    run, arena, view, _ = compile_declared_f32_graph(output, graph, {prefix + "weight": weight})
    seed = 384
    rng = np.random.default_rng(seed)
    hidden = rng.standard_normal((tokens, 2304), dtype=np.float32)
    hidden *= np.geomspace(0.05, 2.0, tokens).astype(np.float32)[:, None]
    hidden = torch.from_numpy(hidden).bfloat16().float()
    view("hidden")[:] = hidden.numpy()
    view("correction_bias")[:] = bias
    if run(arena, len(arena)):
        raise RuntimeError("generated router rejected pinned operands")
    logits = torch.nn.functional.linear(hidden, torch.from_numpy(weight))
    scores = logits.sigmoid()
    corrected = scores + torch.tensor(bias)
    choice = corrected.topk(9, dim=1)
    ids = choice.indices[:, :8]
    margin = (choice.values[:, 7] - choice.values[:, 8]).min().item()
    if margin <= 0:
        raise RuntimeError("oracle route boundary ties in pinned replay")
    values = scores.gather(1, ids)
    expected_weights = (values / (values.sum(dim=-1, keepdim=True) + 1e-20)) * scale
    np.testing.assert_allclose(view("logits"), logits.numpy(), rtol=3e-4, atol=1e-5)
    np.testing.assert_array_equal(view("indices", np.int32), ids.numpy())
    np.testing.assert_allclose(view("route_weights"), expected_weights.numpy(), rtol=3e-5, atol=2e-6)

    weights_max_error = np.abs(view("route_weights") - expected_weights.numpy()).max().item()
    logits_max_error = np.abs(view("logits") - logits.numpy()).max().item()
    view("correction_bias")[:] = 0
    if run(arena, len(arena)):
        raise RuntimeError("generated router rejected zero-bias control")
    no_bias = view("indices", np.int32).copy()
    changed = int(np.count_nonzero(np.sort(no_bias, axis=1) != np.sort(ids.numpy(), axis=1)))
    if not changed:
        raise AssertionError("zero-bias control did not change selected experts")
    identity = json.loads((output / "build-identity.json").read_text())
    return {
        "checkpoint_revision": SOURCE_REVISION, "header_sha256": HEADER_HASH,
        "status": "PASS", "scope": "one real router with synthetic hidden rows and FP32 math",
        "layer": layer, "seed": seed, "tokens": tokens, "oracle": f"PyTorch {torch.__version__} FP32",
        "max_router_logit_abs_error": logits_max_error, "max_route_weight_abs_error": weights_max_error,
        "minimum_top_k_margin": margin, "exact_routed_ids": ids.numel(), "routed_ids": ids.tolist(),
        "zero_bias_control_changed_sorted_entries": changed,
        "weight_source": weight_source, "bias_source": bias_source,
        "build_identity_sha256": identity["build_identity_sha256"],
        "library_sha256": digest((output / "generated.so").read_bytes()),
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
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(mode=0o700)
    report = replay(header, args.output, torch, layer=6)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
