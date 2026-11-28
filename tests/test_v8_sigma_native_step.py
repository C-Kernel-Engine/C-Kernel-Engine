"""Replay the complete generated Sigma DDPM step against independent Torch math."""
import ctypes as C
import json
from pathlib import Path
import os
import subprocess

import numpy as np
import pytest

from version.v8.scripts.build_sigma_native_prefill_v8 import build_circuit as build_prefill
from version.v8.scripts.build_sigma_native_step_v8 import build_circuit
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.scripts.sigma_checkpoint_v8 import pinned_weights


def bits(tensor):
    import torch
    return tensor.contiguous().view(torch.uint16).numpy()


def widen(array):
    return (array.astype(np.uint32) << 16).view(np.float32)


def encoder(hidden, positions, visible, weights, geometry, cache=None):
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    rows = hidden.shape[0]
    heads, kv_heads, dim = (geometry[name] for name in ("heads", "kv_heads", "dim"))
    groups = heads // kv_heads
    angle = positions[:, None].float() * weights["inv_freq"]
    cos, sin = angle.cos().repeat(1, 2).bfloat16(), angle.sin().repeat(1, 2).bfloat16()

    def norm(x, name):
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)).bfloat16() * weights[name]

    def rotate(x, count):
        x = x.reshape(rows, count, dim).transpose(0, 1)
        left, right = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-right, left), dim=-1) * sin

    made = []
    with sdpa_kernel(SDPBackend.MATH):
        for i in range(geometry["layers"]):
            key = f"encoder.layers.{i}."

            def project(data, weight):
                return F.linear(data, weights[key + weight + ".weight"])

            n = norm(hidden, key + "input_layernorm.weight")
            q = rotate(project(n, "self_attn.q_proj"), heads)
            k = rotate(project(n, "self_attn.k_proj"), kv_heads)
            v = project(n, "self_attn.v_proj").reshape(rows, kv_heads, dim).transpose(0, 1)
            made.append((k, v))
            if cache is not None:
                k, v = (torch.cat((prev, current), dim=1) for prev, current in zip(cache[i], (k, v), strict=True))
            context = F.scaled_dot_product_attention(
                q, k.repeat_interleave(groups, dim=0), v.repeat_interleave(groups, dim=0), visible,
            ).transpose(0, 1).reshape(rows, heads * dim)
            hidden = hidden + project(context, "self_attn.o_proj")
            post = norm(hidden, key + "post_attention_layernorm.weight")
            hidden = hidden + project(F.silu(project(post, "mlp.gate_proj")) * project(post, "mlp.up_proj"), "mlp.down_proj")
    return norm(hidden, "encoder.norm.weight"), made


def reference(weights, geometry, z, noise, self_cond, gamma, scale, positions, cond_mask, cache):
    import torch
    from torch.nn import functional as F

    alpha_sq, sigma_sq = torch.sigmoid(-gamma[0]), torch.sigmoid(gamma[0])
    normalized = (z / (alpha_sq / z.shape[-1] + sigma_sq).sqrt()).float()
    angle = gamma[0].float() * weights["time_frequencies"]
    time = torch.cat((angle.sin(), angle.cos())).expand(len(z), -1)
    projected = F.linear(normalized, weights["replaid_input_linear.weight"])
    sc = F.linear(self_cond * (z.shape[-1] ** 0.5), weights["replaid_selfcond_linear.weight"])
    t = F.linear(time, weights["replaid_gamma_linear.weight"])
    prior = F.linear((alpha_sq.sqrt() / sigma_sq).float() * z.float(), weights["normalized_embedding"])
    results = {}
    for label, inp, mask, memory in (
        ("cond", (projected + sc) + t, cond_mask, cache),
        ("uncond", projected + t, None, None),
    ):
        hidden, _ = encoder(inp.bfloat16(), positions, mask, weights, geometry, memory)
        scores = F.linear(hidden, weights["diffusion_head.weight"]).float() + prior
        mean = scores.softmax(-1) @ weights["normalized_embedding"]
        results.update({f"{label}_input": inp.bfloat16(), f"{label}_logits": scores,
                        f"{label}_mean": mean, f"{label}_ids": scores.argmax(-1)})
    a, b = results["cond_mean"], results["uncond_mean"]
    estimate = (a if scale == 1 else b + scale * (a - b)).double()
    alpha_next = torch.sigmoid(-gamma[1])
    blend = -torch.expm1(gamma[1] - gamma[0])
    updated = ((1 - blend) * (alpha_next / alpha_sq).sqrt() * z + blend * alpha_next.sqrt() * estimate +
               (blend * (1 - alpha_next)).sqrt() * noise)
    results.update(prior=prior, normalized=normalized, guided=estimate, next_latent=updated)
    return results


def native_prefix(folder, weights, geometry, tokens, blas):
    rows = len(tokens)
    graph, config, schema = build_prefill(rows, **geometry, blas=blas)
    folder.mkdir()
    run, arena, view, _ = compile_checked_weight_graph(
        folder, graph, {name: bits(weights[name]) if dtype == "BF16" else weights[name].numpy()
                        for name, (_, dtype) in schema.items()},
        schema, source_config=config, target_isa="avx2" if blas else "portable",
    )
    position = np.arange(rows, dtype=np.int32)
    view("positions", np.int32)[:] = position
    view("ranges", np.int32)[:] = np.stack((np.zeros(rows, dtype=np.int32), position + 1,
                                          position + 1, position + 1), axis=1)
    ids = view("prefix_tokens", np.int32)
    ids[:] = tokens
    ids[-1] = geometry["vocabulary"]
    assert run(arena, len(arena)) != 0
    ids[:] = tokens
    assert run(arena, len(arena)) == 0
    return [
        tuple(view(f"layer{i}_{name}", np.uint16).copy() for name in ("k_rotary", "v"))
        for i in range(geometry["layers"])
    ]


def check_native_schedule(folder, step, arena, view, initial, schedule, noises, expected, vocabulary, *, cancel_trial):
    fields = ("latent", "next_latent", "noise", "gamma", "self_condition", "cond_mean", "cond_ids")
    types = (np.float64,) * 4 + (np.float32, np.float32, np.int32)

    class Bindings(C.Structure):
        _fields_ = [(name, C.c_size_t) for name in fields]

    class Request(C.Structure):
        _fields_ = [
            ("step", C.c_void_p), ("arena", C.c_void_p), ("arena_bytes", C.c_size_t),
            ("bindings", Bindings), ("latent_elements", C.c_size_t), ("token_count", C.c_size_t),
            ("vocabulary", C.c_size_t), ("initial_latent", C.c_void_p), ("schedule", C.c_void_p),
            ("schedule_count", C.c_size_t), ("noise", C.c_void_p), ("noise_elements", C.c_size_t),
            ("output_ids", C.c_void_p), ("output_capacity", C.c_size_t),
            ("should_cancel", C.c_void_p), ("cancel_context", C.c_void_p),
        ]

    root = Path(__file__).resolve().parents[1]
    runtime = folder / "schedule.so"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    "-I", str(root / "include"), str(root / "src/runtime/checked_diffusion_schedule.c"),
                    "-o", str(runtime)], check=True)
    invoke = C.CDLL(str(runtime)).ck_run_checked_diffusion_schedule
    invoke.argtypes = [C.POINTER(Request)]
    invoke.restype = C.c_int
    address = C.addressof(arena)
    bindings = Bindings(*(view("step_" + name, kind).ctypes.data - address
                          for name, kind in zip(fields, types, strict=True)))
    output = np.full(len(expected) + 1, -91, dtype=np.int32)
    request = Request(C.cast(step, C.c_void_p), address, len(arena), bindings, initial.size, len(expected), vocabulary,
                      initial.ctypes.data, schedule.ctypes.data, schedule.size, noises.ctypes.data, noises.size,
                      output.ctypes.data, output.size, None, None)
    request.output_capacity = len(expected) - 1
    assert invoke(C.byref(request)) != 0
    np.testing.assert_array_equal(output, -91)
    request.output_capacity = len(output)
    if cancel_trial:
        # Neither malformed offsets nor a repeated transition may write IDs.
        request.bindings.cond_ids = request.bindings.latent
        assert invoke(C.byref(request)) == -1
        request.bindings.cond_ids = bindings.cond_ids
        bad_schedule = schedule.copy()
        bad_schedule[1] = bad_schedule[0]
        request.schedule = bad_schedule.ctypes.data
        assert invoke(C.byref(request)) == -1
        request.schedule = schedule.ctypes.data
        np.testing.assert_array_equal(output, -91)
        polls = 0

        @C.CFUNCTYPE(C.c_int, C.c_void_p)
        def after_first_step(_):
            nonlocal polls
            polls += 1
            return int(polls == 3)

        request.should_cancel = C.cast(after_first_step, C.c_void_p)
        assert invoke(C.byref(request)) == -7
        assert polls == 3
        np.testing.assert_array_equal(output, -91)
        request.should_cancel = None
    assert invoke(C.byref(request)) == 0
    np.testing.assert_array_equal(output[:-1], expected)
    assert output[-1] == -91
    return output[:-1].tolist()


def run_case(folder, *, weights, tokens, geometry, bounds, blas=False):
    import torch

    rows, latent, n = 3, geometry["latent_width"], len(tokens)
    model_geometry = {name: value for name, value in geometry.items() if name not in {"latent_width", "bands"}}
    prepared = native_prefix(folder / "prefill", weights, model_geometry, tokens, blas)
    graph, config, schema = build_circuit(rows, prefix_rows=n, **geometry, blas=blas)
    storage = {name: bits(weights[name]) if dtype == "BF16" else weights[name].contiguous().numpy()
               for name, (_, dtype) in schema.items()}
    run, arena, view, _ = compile_checked_weight_graph(
        folder, graph, storage, schema, source_config=config, target_isa="avx2" if blas else "portable",
    )
    rng = torch.Generator().manual_seed(19057)
    z = torch.randn((rows, latent), dtype=torch.float64, generator=rng)
    clean = torch.arange(n, dtype=torch.int32)
    positions = torch.arange(n, n + rows, dtype=torch.int32)
    cond_mask = torch.ones((rows, n + rows), dtype=torch.bool)
    ranges = np.tile([0, n + rows, n + rows, n + rows], (rows, 1))
    if not blas and n >= 3:
        # Drop one clean key while keeping the current block fully visible.
        cond_mask[:, 1] = False
        ranges[:] = [0, 1, 2, n + rows]
    prefix = weights["encoder.embed_tokens.weight"][list(tokens)]
    _, oracle_cache = encoder(prefix, clean, clean[:, None] >= clean[None, :], weights, geometry)
    native_cache = []
    prefix_error = 0.0
    view("step_positions", np.int32)[:] = positions
    view("step_cond_ranges", np.int32)[:] = ranges
    view("step_uncond_ranges", np.int32)[:] = [0, rows, rows, rows]
    stored = {}
    for i, pair in enumerate(prepared):
        current = []
        for kind, data in zip(("key", "value"), pair, strict=True):
            name = f"step_cond_layer{i}_prefix_{kind}"
            stored[name] = data
            view(name, np.uint16)[:] = data
            current.append(torch.from_numpy(data).view(torch.bfloat16)
                           .reshape(n, geometry["kv_heads"], geometry["dim"]).transpose(0, 1))
        prefix_error = max(prefix_error, *(float((a.float() - b.float()).abs().max())
                                           for a, b in zip(current, oracle_cache[i], strict=True)))
        native_cache.append(tuple(current))

    def bind(name, tensor):
        array = tensor.numpy()
        view("step_" + name, array.dtype)[:] = array

    def decode(name):
        if name.endswith("_input"):
            return widen(view("step_" + name, np.uint16))
        if name.endswith("_ids"):
            return view("step_" + name, np.int32)
        dtype = np.float64 if name in {"guided", "next_latent"} else np.float32
        return view("step_" + name, dtype)

    endpoints = [2.0, 0.5, -0.75, -2.5, -2.5]
    self_cond = torch.zeros(z.shape, dtype=torch.float32)
    initial, expected_z, expected_self = z, z, self_cond
    trace, noises = [], []
    for i in range(len(endpoints) - 1):
        gamma = torch.tensor(endpoints[i:i+2], dtype=torch.float64)
        noise = torch.randn(z.shape, dtype=torch.float64, generator=rng)
        scale = 1.75
        if gamma[0] != gamma[1]:
            noises.append(noise.numpy())
        for name, data in (("latent", z), ("noise", noise), ("self_condition", self_cond),
                           ("gamma", gamma), ("guidance", torch.tensor([scale]))):
            bind(name, data)
        assert run(arena, len(arena)) == 0
        same_input = reference(weights, geometry, z, noise, self_cond, gamma, scale, positions, cond_mask, native_cache)
        trajectory = reference(weights, geometry, expected_z, noise, expected_self, gamma, scale, positions, cond_mask, oracle_cache)
        record = {}
        for name, expected in same_input.items():
            actual = decode(name)
            oracle = expected.float().numpy() if expected.dtype == torch.bfloat16 else expected.numpy()
            if name.endswith("_ids"):
                differing = np.flatnonzero(actual != oracle)
                record[name] = int(actual.size - differing.size)
                logits_name = name.replace("_ids", "_logits")
                oracle_logits = same_input[logits_name].numpy()
                record[name + "_mismatches"] = [
                    {"row": int(row), "native": int(actual[row]), "oracle": int(oracle[row]),
                     "oracle_gap": float(oracle_logits[row, oracle[row]] - oracle_logits[row, actual[row]])}
                    for row in differing
                ]
                continue
            delta = actual.astype(np.float64) - oracle
            record[name] = {"max": float(np.abs(delta).max()), "rmse": float(np.sqrt(np.mean(delta ** 2)))}
            limit = bounds.get(name)
            if limit is not None:
                assert record[name]["max"] <= limit, (i, name, record[name])
            if name.endswith("_logits") and "logits_rmse" in bounds:
                assert record[name]["rmse"] <= bounds["logits_rmse"], (i, name, record[name])
        next_z = decode("next_latent").copy()
        record["trajectory_max"] = float(np.max(np.abs(next_z - trajectory["next_latent"].numpy())))
        assert record["trajectory_max"] <= bounds["trajectory"], record
        record["trajectory_ids"] = int(np.count_nonzero(decode("cond_ids") == trajectory["cond_ids"].numpy()))
        trace.append(record)
        if gamma[0] == gamma[1]:
            np.testing.assert_array_equal(next_z, z.numpy())
        if i == 0:
            first = next_z.copy()
            first_noise = noise
        z = torch.from_numpy(next_z)
        self_cond = torch.from_numpy(decode("cond_mean").copy())
        expected_z, expected_self = trajectory["next_latent"], trajectory["cond_mean"]

    final_ids = decode("cond_ids").tolist()
    before_error = decode("next_latent").copy()
    for name, value, good in (
        ("gamma", torch.tensor([0.0, 1.0], dtype=torch.float64), torch.tensor([-1.0, -2.0], dtype=torch.float64)),
        ("noise", torch.full(z.shape, torch.nan, dtype=torch.float64), first_noise),
    ):
        bind(name, value)
        assert run(arena, len(arena)) != 0
        np.testing.assert_array_equal(decode("next_latent"), before_error)
        bind(name, good)
    for name, data in (("latent", initial), ("noise", first_noise),
                       ("gamma", torch.tensor(endpoints[:2], dtype=torch.float64)),
                       ("self_condition", torch.zeros_like(self_cond))):
        bind(name, data)
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(decode("next_latent"), first)
    final_ids = check_native_schedule(folder, run, arena, view, initial.numpy(), np.array(endpoints[:-1]),
                                      np.stack(noises), final_ids, geometry["vocabulary"], cancel_trial=not blas)
    for name, before in stored.items():
        np.testing.assert_array_equal(view(name, np.uint16), before)
    return {"steps": trace, "prefix_max": prefix_error, "prompt_ids": list(tokens),
            "final_ids": final_ids}


def test_sigma_native_step_and_trajectory(tmp_path):
    torch = pytest.importorskip("torch")
    geometry = dict(layers=2, vocabulary=19, width=11, heads=4, kv_heads=2, dim=6, inner=17,
                    latent_width=5, bands=4)
    _, _, schema = build_circuit(3, prefix_rows=3, **geometry)
    rng = torch.Generator().manual_seed(573)
    weights = {}
    for name, (shape, storage) in schema.items():
        tensor = torch.randn(shape, generator=rng) * 0.23
        weights[name] = tensor.bfloat16() if storage == "BF16" else tensor
    weights["normalized_embedding"] = torch.nn.functional.normalize(weights["normalized_embedding"], dim=-1)
    weights["inv_freq"] = torch.tensor([1.0, 0.1, 0.001])
    weights["encoder.embed_tokens.weight"] = torch.randn((19, 11), generator=rng).bfloat16()
    report = run_case(tmp_path, weights=weights, tokens=(1, 7, 12), geometry=geometry, bounds={"cond_logits": .015625, "uncond_logits": .015625,
                                                 "next_latent": .006, "trajectory": .008})
    assert all(r["cond_ids"] == r["uncond_ids"] == r["trajectory_ids"] == 3 for r in report["steps"])



def test_pinned_sigma_native_step(tmp_path, record_property):
    checkpoint = os.environ.get("CK_SIGMA_BF16_CHECKPOINT")
    if not checkpoint or not os.environ.get("CK_BLAS_LIBRARY"):
        pytest.skip("configure CK_SIGMA_BF16_CHECKPOINT and CK_BLAS_LIBRARY for the pinned replay")
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors")
    torch.set_num_threads(4)
    geometry = dict(layers=26, vocabulary=131072, width=3072, heads=32, kv_heads=8, dim=128, inner=9216,
                    latent_width=16, bands=32)
    tokens = (1, 1784, 8961, 1307, 5498, 1395)
    _, _, schema = build_circuit(3, prefix_rows=len(tokens), **geometry, blas=True)
    weights = pinned_weights(Path(checkpoint), schema, torch, safetensors)
    report = run_case(tmp_path, weights=weights, tokens=tokens, geometry=geometry, blas=True,
                      bounds={"cond_logits": 0.5, "uncond_logits": 0.5, "logits_rmse": .08,
                              "cond_mean": .02, "uncond_mean": .02, "guided": .04,
                              "next_latent": .06, "trajectory": .08})
    record_property("scope", "pinned BF16 weights, generated clean prefix and native fixed-noise DDPM schedule")
    record_property("metrics", json.dumps(report))
    assert report["prefix_max"] <= 0.25, report
    assert all(r["cond_ids"] == r["trajectory_ids"] == 3 for r in report["steps"]), report
