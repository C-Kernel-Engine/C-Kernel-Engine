#!/usr/bin/env python3
"""Replay Inkling layer zero from real packed GGUF ranges and supplied hidden rows."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from version.v8.scripts.build_inkling2_decoder_block_v8 import first_block_weights, first_decoder_block
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.scripts.replay_inkling2_mini_causal_kv import run
from version.v8.scripts.replay_inkling2_mini_projections import digest
from version.v8.scripts.replay_inkling2_routed_moe import (
    EXPERTS, GGML_TYPE_F32, GGML_TYPE_Q8_0, GGUF_FILES, GGUF_REPO, GGUF_REVISION, TOP_K, WIDTH,
    fetch_gguf_range, from_gguf, gguf_tensor, read_gguf_header, unpack_gguf,
)


def attention_reference(torch, hidden, weights):
    F = torch.nn.functional

    def weight(name):
        return torch.tensor(weights["blk.0." + name])

    def norm(x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)

    def conv(x, name):
        filters = weight(name).unsqueeze(1)
        history = F.pad(x.T.unsqueeze(0), (filters.shape[-1] - 1, 0))
        return (x + F.conv1d(history, filters, groups=x.shape[1])[0].T) / np.sqrt(5)

    def clip(x, maximum):
        return x * (maximum / (x.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1)

    rows = hidden.shape[0]
    normalized = norm(hidden) * weight("attn_norm.weight")
    query = clip(
        F.linear(normalized, weight("attn_q.weight")).view(rows, 24, 128) * weight("attn_q_clip.weight").view(24, 128),
        22.627416610717773,
    )
    raw_key = F.linear(normalized, weight("attn_k.weight"))
    raw_value = F.linear(normalized, weight("attn_v.weight"))
    key = clip(conv(raw_key, "attn_k_conv.weight").reshape(rows, 8, 128), 22.627416610717773)
    value = norm(conv(raw_value, "attn_v_conv.weight").reshape(rows, 8, 128))
    relative = clip(
        F.linear(normalized, weight("attn_rel.weight")).view(rows, 24, 16) * weight("attn_rel_clip.weight").view(24, 16),
        8,
    )
    band = F.linear(relative, weight("attn_rel_bias.weight"))
    sink = F.linear(normalized, weight("attn_sink_proj.weight"))
    distance = torch.arange(rows)[:, None] - torch.arange(rows)[None, :]
    bias = band.gather(-1, distance.clamp(min=0)[:, None, :].expand(-1, 24, -1))
    scores = torch.einsum("thd,shd->ths", query, key.repeat_interleave(3, dim=1)) / np.sqrt(128)
    scores = (scores + bias).masked_fill(distance[:, None, :] < 0, -torch.inf)
    probabilities = torch.cat((scores, sink.unsqueeze(-1)), dim=-1).softmax(dim=-1)[..., :-1]
    attention = torch.einsum("ths,shd->thd", probabilities, value.repeat_interleave(3, dim=1)).reshape(rows, 3072)
    gate = F.linear(normalized, weight("attn_gate.weight")).sigmoid()
    projected = F.linear(attention * gate, weight("attn_output.weight"))
    residual = conv(projected, "attn_post_conv.weight")
    states = norm(torch.stack((hidden, residual), dim=1))
    mix_weights = (F.linear(states, weight("ffn_res_query.weight").unsqueeze(0)).squeeze(-1) / np.sqrt(WIDTH))
    mixed = (states * mix_weights.softmax(dim=-1).unsqueeze(-1)).sum(dim=1)
    ffn_input = norm(mixed) * weight("ffn_norm.weight")
    return {
        "attn_norm": normalized, "projected_attention": projected, "attn_post_conv": residual,
        "depth_mix": mixed, "ffn_norm": ffn_input,
    }, {
        "cache_k_0": key.transpose(0, 1), "cache_v_0": value.transpose(0, 1),
        "history_k_0": raw_key[-3:], "history_v_0": raw_value[-3:],
        "history_attn_post_conv": projected[-3:],
    }, conv


def replay(folder, torch):
    file = GGUF_FILES["q8_0"]
    header, header_proof = fetch_gguf_range(file, 0, 16 * 1024 * 1024)
    tensors, data_start = read_gguf_header(header)
    rows = 5
    options = dict(slots=TOP_K)
    shape_graph = first_decoder_block(prefill_tokens=rows, **options)
    declarations = first_block_weights(shape_graph)
    references = {}
    weights = {}
    sources = {}
    for name, (shape, mode) in declarations.items():
        if ".expert_bank." not in name:
            kind = GGML_TYPE_Q8_0 if mode == "Q8_0" else GGML_TYPE_F32
            gguf_dims = tuple(shape[::-1])
            if name in {"blk.0.attn_q_clip.weight", "blk.0.attn_rel_clip.weight"}:
                gguf_dims = (shape[0] // 24, 24)
            raw, sources[name] = gguf_tensor(file, tensors, data_start, name, gguf_dims, kind)
            if mode == "Q8_0":
                weights[name] = np.frombuffer(raw, np.uint8)
                references[name] = unpack_gguf(weights[name], shape, kind)
            else:
                weights[name] = references[name] = np.frombuffer(raw, "<f4").reshape(shape)

    (router, bias), bias_source = from_gguf(file, tensors, data_start, 0)
    np.testing.assert_array_equal(router, weights["blk.0.ffn_gate_inp.weight"])
    sources["blk.0.exp_probs_b.bias"] = bias_source[-1]
    seed = 162
    hidden = torch.from_numpy(np.random.default_rng(seed).normal(0., .2, (rows, WIDTH)).astype(np.float32))
    expected, expected_state, conv = attention_reference(torch, hidden, references)
    F = torch.nn.functional
    ff = expected["ffn_norm"]
    scores = F.linear(ff, torch.tensor(router)).sigmoid()
    top = (scores + torch.tensor(bias)).topk(TOP_K + 1, dim=-1)
    ids = top.indices[:, :TOP_K]
    route_stability = (top.values[:, -2] - top.values[:, -1]).min().item()
    if route_stability <= 0:
        raise ValueError("oracle has a tied top-k boundary")
    probs = scores.gather(1, ids)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20) * 8
    mixture = torch.zeros_like(ff)
    active_ids = sorted(set(ids.flatten().tolist()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        bank = list(pool.map(lambda expert: from_gguf(file, tensors, data_start, 0, expert), active_ids))
    expert_map = np.full(EXPERTS, -1, dtype=np.int32)
    for slot, expert in enumerate(active_ids):
        expert_map[expert] = slot
        _, matrices, sources[f"expert_{expert}"] = bank[slot]
        row_indices, slot_indices = torch.where(ids == expert)
        g, u, d = (torch.tensor(mat) for mat in matrices)
        g = F.linear(ff[row_indices], g)
        u = F.linear(ff[row_indices], u)
        act = (4 * (g / 4).tanh() * g.sigmoid()) * (25 * (u / 25).tanh())
        mixture[row_indices] += F.linear(act, d) * probs[row_indices, slot_indices, None]
    expected["mixture"] = mixture
    expected["ffn_post_conv"] = conv(mixture, "ffn_post_conv.weight")
    expected["layer_output"] = expected["attn_post_conv"] + expected["ffn_post_conv"]
    expected_state["history_ffn_post_conv"] = mixture[-3:]
    for i, name in enumerate(("gate", "up", "down")):
        weights["blk.0.expert_bank." + name] = np.concatenate([record[0][i] for record in bank])

    report = {
        "status": "PASS", "physical_layer": 0, "state_slot": 0,
        "contract": "packed Q8_0 weights, FP32 activations and state, ordered FP64 projection accumulation",
        "scope": "synthetic layer inputs, compact bank of routed experts, no whole model load",
        "oracle": f"PyTorch {torch.__version__} with FP32 expansion of the same GGUF weights",
        "source": {"repo": GGUF_REPO, "revision": GGUF_REVISION, "file": file[0],
                   "header_range": header_proof, "ranges": sources},
        "seed": seed, "input_sha256": digest(hidden.numpy().tobytes()), "tokens": rows,
        "exact_route_ids": ids.tolist(), "minimum_route_margin": route_stability, "builds": {},
    }
    views = {}
    for name, length in (("split", rows - 1), ("full", rows)):
        path = folder / name
        path.mkdir()
        graph = first_decoder_block(prefill_tokens=length, slots=len(bank))
        bundle = compile_checked_weight_graph(path, graph, weights, first_block_weights(graph))
        view = bundle[2]
        view("router_bias")[:] = bias
        view("expert_slots", np.int32)[:] = expert_map
        view("inkling2_prefill_slot0_hidden")[:] = hidden[:length].numpy()
        run(bundle, "prefill", 0)
        if name == "split":
            view("inkling2_append_slot0_hidden")[:] = hidden[length:].numpy()
            run(bundle, "append", 0)
        views[name] = view
        identity = json.loads((path / "build-identity.json").read_text())
        report["builds"][name] = {
            "build_identity_sha256": identity["build_identity_sha256"],
            "library_sha256": identity["library_sha256"],
            "selected_providers": sorted(identity["selected_provider_sha256"]),
        }

    def compare(name, a, b, oracle):
        np.testing.assert_array_equal(a, b, err_msg=name)
        np.testing.assert_allclose(a, oracle.numpy(), atol=2e-5, rtol=5e-5, err_msg=name)
        return {"split_exact": True, "max_abs_vs_oracle": float(np.abs(a - oracle.numpy()).max())}

    split, full = views["split"], views["full"]
    report["activations"] = {}
    for name, target in expected.items():
        sequence = np.concatenate([split(f"inkling2_{phase}_slot0_{name}") for phase in ("prefill", "append")])
        report["activations"][name] = compare(name, sequence, full("inkling2_prefill_slot0_" + name), target)
    report["state"] = {}
    for name, target in expected_state.items():
        states = split(name), full(name)
        if name.startswith("cache"):
            states = states[0], states[1][:, :rows]
        report["state"][name] = compare(name, *states, target)
    split_ids = np.concatenate([split(f"inkling2_{phase}_slot0_indices", np.int32) for phase in ("prefill", "append")])
    np.testing.assert_array_equal(split_ids, full("inkling2_prefill_slot0_indices", np.int32))
    np.testing.assert_array_equal(split_ids, ids.numpy())
    hidden.numpy().astype("<f4", copy=False).tofile(folder / "hidden.f32")
    full("inkling2_prefill_slot0_layer_output").astype("<f4", copy=False).tofile(folder / "layer-output.f32")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(mode=0o700)
    report = replay(args.output, torch)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
