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

from version.v8.scripts.build_inkling2_decoder_block_v8 import (
    PACKED_PROJECTIONS, first_block_weights, first_decoder_block,
)
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.scripts.replay_inkling2_mini_causal_kv import run
from version.v8.scripts.replay_inkling2_mini_projections import digest
from version.v8.scripts.replay_inkling2_routed_moe import (
    EXPERTS, GGML_TYPE_F32, GGML_TYPE_Q8_0, GGUF_FILES, GGUF_REPO, GGUF_REVISION, TOP_K, WIDTH,
    fetch_gguf_range, from_gguf, gguf_tensor, make_local_gguf_range_reader, read_gguf_header, unpack_gguf,
)


def quantized_activation(torch, tensor):
    blocks = tensor.reshape(*tensor.shape[:-1], -1, 32)
    peak = blocks.abs().amax(dim=-1, keepdim=True)
    scale = (peak / 127).half().float()
    inverse = torch.where(scale > 0, torch.full_like(peak, 127) / peak, 0)
    codes = (blocks * inverse).round().clamp(-127, 127).int()
    return codes, scale.squeeze(-1)


def expand_packed_activations(torch, tensor):
    """Independent expanded FP32 reference for nearest-even Q8_0 activation."""
    codes, scale = quantized_activation(torch, tensor)
    return (codes.float() * scale[..., None]).reshape_as(tensor)


def packed_linear_reference(torch, values, packed):
    """Model the Q8 block scales, integer dot groups, and FP32 lane reduction."""
    codes, scales = quantized_activation(torch, values)
    rows, blocks, _ = codes.shape
    weight = packed.reshape(-1, blocks, 34)
    weight_scales = torch.from_numpy(weight[:, :, :2].copy().view("<f2").squeeze(-1)).float()
    weight_codes = torch.from_numpy(weight[:, :, 2:].copy().view("i1")).int()
    products = torch.einsum(
        "rblc,oblc->robl", codes.reshape(rows, blocks, 8, 4),
        weight_codes.reshape(weight.shape[0], blocks, 8, 4),
    )
    scale = scales[:, None] * weight_scales[None, :]
    lanes = torch.zeros((rows, weight.shape[0], 8), dtype=torch.float32)
    for block in range(blocks):
        # Widen the integer product and accumulator to model one FP32 rounding
        # at the FMA boundary instead of an FP32 multiply followed by addition.
        product = scale[:, :, block, None].double() * products[:, :, block].double()
        lanes = (lanes.double() + product).float()
    a, b, c, d, e, f, g, h = lanes.unbind(-1)
    return ((a + e) + (c + g)) + ((b + f) + (d + h))


def attention_reference(torch, hidden, weights, packed_weights, *, q8_activations=False, packed_oracle=False):
    F = torch.nn.functional

    def weight(name):
        return torch.tensor(weights["blk.0." + name])

    def project(x, name):
        if q8_activations and name in PACKED_PROJECTIONS:
            if packed_oracle:
                return packed_linear_reference(torch, x, packed_weights["blk.0." + name])
            x = expand_packed_activations(torch, x)
        return F.linear(x, weight(name))

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
        project(normalized, "attn_q.weight").view(rows, 24, 128) * weight("attn_q_clip.weight").view(24, 128),
        22.627416610717773,
    )
    raw_key = project(normalized, "attn_k.weight")
    raw_value = project(normalized, "attn_v.weight")
    key = clip(conv(raw_key, "attn_k_conv.weight").reshape(rows, 8, 128), 22.627416610717773)
    value = norm(conv(raw_value, "attn_v_conv.weight").reshape(rows, 8, 128))
    relative = clip(
        project(normalized, "attn_rel.weight").view(rows, 24, 16) * weight("attn_rel_clip.weight").view(24, 16),
        8,
    )
    band = F.linear(relative, weight("attn_rel_bias.weight"))
    sink = project(normalized, "attn_sink_proj.weight")
    distance = torch.arange(rows)[:, None] - torch.arange(rows)[None, :]
    bias = band.gather(-1, distance.clamp(min=0)[:, None, :].expand(-1, 24, -1))
    scores = torch.einsum("thd,shd->ths", query, key.repeat_interleave(3, dim=1)) / np.sqrt(128)
    scores = (scores + bias).masked_fill(distance[:, None, :] < 0, -torch.inf)
    probabilities = torch.cat((scores, sink.unsqueeze(-1)), dim=-1).softmax(dim=-1)[..., :-1]
    attention = torch.einsum("ths,shd->thd", probabilities, value.repeat_interleave(3, dim=1)).reshape(rows, 3072)
    gate = project(normalized, "attn_gate.weight").sigmoid()
    projected = project(attention * gate, "attn_output.weight")
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


def replay(folder, torch, *, q8_activations=False, packed_oracle=False, local_gguf=None):
    read_range = make_local_gguf_range_reader(local_gguf) if local_gguf else fetch_gguf_range
    file = GGUF_FILES["q8_0"]
    header, header_proof = read_range(file, 0, 16 * 1024 * 1024)
    tensors, data_start = read_gguf_header(header)
    if local_gguf:
        prior = json.loads((ROOT / "docs/notes/artifacts/inkling2_mini_q8_first_decoder_block.json").read_text())
        directory_hash = prior["llama_cpp_cpu_comparison"]["artifact_header_sha256"]
        if local_gguf.stat().st_size != file[1] or digest(header[:data_start]) != directory_hash:
            raise ValueError("local GGUF file size or tensor directory differs from the pinned Q8_0 artifact")
    rows = 5
    options = dict(slots=TOP_K, q8_activations=q8_activations)
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
            raw, sources[name] = gguf_tensor(file, tensors, data_start, name, gguf_dims, kind, read_range=read_range)
            if mode == "Q8_0":
                weights[name] = np.frombuffer(raw, np.uint8)
                references[name] = unpack_gguf(weights[name], shape, kind)
            else:
                weights[name] = references[name] = np.frombuffer(raw, "<f4").reshape(shape)

    (router, bias), bias_source = from_gguf(file, tensors, data_start, 0, read_range=read_range)
    np.testing.assert_array_equal(router, weights["blk.0.ffn_gate_inp.weight"])
    sources["blk.0.exp_probs_b.bias"] = bias_source[-1]
    seed = 162
    hidden = torch.from_numpy(np.random.default_rng(seed).normal(0., .2, (rows, WIDTH)).astype(np.float32))
    expected, expected_state, conv = attention_reference(
        torch, hidden, references, weights, q8_activations=q8_activations, packed_oracle=packed_oracle)
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
    active_ids = sorted(set(ids.flatten().tolist()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        bank = list(pool.map(lambda expert: from_gguf(file, tensors, data_start, 0, expert, read_range=read_range), active_ids))
    expert_map = np.full(EXPERTS, -1, dtype=np.int32)
    for slot, expert in enumerate(active_ids):
        expert_map[expert] = slot
        sources[f"expert_{expert}"] = bank[slot][2]

    def expert_mixture(inputs, route_ids, routing):
        mixture = torch.zeros_like(inputs)
        for expert, (packed, matrices, _) in zip(active_ids, bank):
            row_indices, slot_indices = torch.where(route_ids == expert)
            if not row_indices.numel():
                continue

            def project(values, index):
                if packed_oracle:
                    return packed_linear_reference(torch, values, packed[index])
                if q8_activations:
                    values = expand_packed_activations(torch, values)
                return F.linear(values, torch.tensor(matrices[index]))

            ff_rows = inputs[row_indices]
            g, u = project(ff_rows, 0), project(ff_rows, 1)
            act = (4 * (g / 4).tanh() * g.sigmoid()) * (25 * (u / 25).tanh())
            mixture[row_indices] += project(act, 2) * routing[row_indices, slot_indices, None]
        return mixture

    mixture = expert_mixture(ff, ids, probs)
    expected["mixture"] = mixture
    expected["ffn_post_conv"] = conv(mixture, "ffn_post_conv.weight")
    expected["layer_output"] = expected["attn_post_conv"] + expected["ffn_post_conv"]
    expected_state["history_ffn_post_conv"] = mixture[-3:]
    for i, name in enumerate(("gate", "up", "down")):
        weights["blk.0.expert_bank." + name] = np.concatenate([record[0][i] for record in bank])

    report = {
        "status": "PASS", "physical_layer": 0, "state_slot": 0,
        "contract": ("packed Q8_0 weights and nearest-even Q8_0 activations, eight-lane FP32 FMA reduction, FP32 state"
                     if q8_activations else
                     "packed Q8_0 weights, FP32 activations and state, ordered FP64 projection accumulation"),
        "scope": "synthetic layer inputs, compact bank of routed experts, no whole model load",
        "oracle": (
            f"PyTorch {torch.__version__} with ordered Q8 block-dot arithmetic"
            if packed_oracle else
            f"PyTorch {torch.__version__} with FP32 expansion of the same GGUF weights"
            + (" and explicitly quantized activations" if q8_activations else "")
        ),
        "source": {"repo": GGUF_REPO, "revision": GGUF_REVISION, "file": file[0],
                   "header_range": header_proof, "ranges": sources,
                   "read_mode": "local_ranges" if local_gguf else "publisher_http_ranges"},
        "seed": seed, "input_sha256": digest(hidden.numpy().tobytes()), "tokens": rows,
        "exact_route_ids": ids.tolist(), "minimum_route_margin": route_stability, "builds": {},
    }
    views = {}
    for name, length in (("split", rows - 1), ("full", rows)):
        path = folder / name
        path.mkdir()
        graph = first_decoder_block(prefill_tokens=length, slots=len(bank), q8_activations=q8_activations)
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
        ref = oracle.numpy()
        if not np.isfinite(a).all() or not np.isfinite(ref).all():
            raise ValueError(f"nonfinite data at {name}")
        delta = np.abs(a - ref)
        mismatches = int(np.count_nonzero(delta > 2e-5 + 5e-5 * np.abs(ref)))
        return {"split_exact": True, "max_abs_vs_oracle": float(delta.max()),
                "oracle_compare": "FAIL" if mismatches else "PASS",
                "oracle_mismatched_elements": mismatches}

    split, full = views["split"], views["full"]
    # Capture fixed logical rows before the comparisons so a failure can be replayed.
    hidden.numpy().astype("<f4", copy=False).tofile(folder / "hidden.f32")
    full("inkling2_prefill_slot0_layer_output").astype("<f4", copy=False).tofile(folder / "layer-output.f32")
    full("inkling2_prefill_slot0_indices", np.int32).astype("<i4", copy=False).tofile(folder / "route-ids.i32")
    if q8_activations:
        trace_names = ("attn_norm", "query_raw", "query", "gated_attention", "projected_attention", "attn_post_conv",
                       "depth_mix", "ffn_norm", "logits", "mixture", "ffn_post_conv")
        np.savez(folder / "native-trace.npz",
                 **{name: full("inkling2_prefill_slot0_" + name) for name in trace_names})
        np.savez(folder / "oracle-trace.npz", **{name: value.numpy() for name, value in expected.items()})
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
    exact_routes = np.array_equal(split_ids, ids.numpy())
    if q8_activations:
        native_input = torch.from_numpy(full("inkling2_prefill_slot0_ffn_norm"))
        native_codes, native_scales = quantized_activation(torch, native_input)
        oracle_codes, oracle_scales = quantized_activation(torch, ff)
        report["ffn_input_quantization"] = {
            "different_scales": int((native_scales != oracle_scales).sum()),
            "scales": native_scales.numel(),
            "different_codes": int((native_codes != oracle_codes).sum()),
            "codes": native_codes.numel(),
        }
        if exact_routes:
            native_weights = torch.from_numpy(full("inkling2_prefill_slot0_route_weights"))
            same_input = expert_mixture(native_input, torch.from_numpy(split_ids), native_weights)
            native_mix = full("inkling2_prefill_slot0_mixture")
            report["routed_experts_same_input"] = {
                **compare("routed experts", native_mix, native_mix, same_input),
                "bitwise_equal": native_mix.tobytes() == same_input.numpy().tobytes(),
            }
    comparisons = [*report["activations"].values(), *report["state"].values()]
    report.update(
        status="PASS" if exact_routes and all(c["oracle_compare"] == "PASS" for c in comparisons) else "FAIL",
        native_split_continuity="PASS", route_oracle="PASS" if exact_routes else "FAIL",
        oracle_tolerances={"atol": 2e-5, "rtol": 5e-5},
        exact_route_ids=split_ids.tolist(),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q8-activations", action="store_true")
    parser.add_argument("--torch-q8-block-contract", action="store_true",
                        help="use the packed block and FP32 lane order in the PyTorch oracle")
    parser.add_argument("--gguf", type=Path, help="read pinned layer-zero ranges from a local GGUF")
    args = parser.parse_args()
    if args.torch_q8_block_contract and not args.q8_activations:
        parser.error("the PyTorch Q8 block contract requires --q8-activations")
    import torch

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(mode=0o700)
    report = replay(args.output, torch, q8_activations=args.q8_activations,
                    packed_oracle=args.torch_q8_block_contract, local_gguf=args.gguf)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "PASS":
        parser.exit(1, "composed reference comparison failed; see report.json and captured rows\n")


if __name__ == "__main__":
    main()
