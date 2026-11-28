"""Exercise two committed Sigma blocks on one compiled native arena."""
import ctypes as C
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.test_v8_sigma_native_step import bits, encoder, pinned_weights, reference, widen
from version.v8.scripts.build_sigma_native_session_v8 import build_circuit
from version.v8.scripts.checked_execution_exports_v8 import load_exports
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph

SIZE, PTR = C.c_size_t, C.c_void_p
ENTRY = C.CFUNCTYPE(C.c_int, PTR, SIZE)
CANCEL = C.CFUNCTYPE(C.c_int, PTR)


class Plane(C.Structure):
    _fields_ = [(name, SIZE) for name in ("cache", "prefill_rows", "append_rows")]


class Clean(C.Structure):
    _fields_ = [("entry", PTR)] + [(name, SIZE) for name in ("token_ids", "positions", "ranges")]


STEP_FIELDS = ("latent", "next_latent", "noise", "gamma", "self_condition", "cond_mean", "cond_ids")


class Step(C.Structure):
    _fields_ = [(name, SIZE) for name in STEP_FIELDS]


class Session(C.Structure):
    _fields_ = [
        ("arena", PTR), ("arena_bytes", SIZE), ("prefill", Clean), ("append", Clean),
        ("step", PTR), ("step_bindings", Step),
        *((name, SIZE) for name in ("positions", "conditional_ranges", "unconditional_ranges", "guidance")),
        ("planes", C.POINTER(Plane)),
        *((name, SIZE) for name in ("plane_count", "kv_width", "capacity", "prompt_rows", "block_rows",
                                   "latent_elements", "vocabulary", "valid_rows")),
    ]


class Block(C.Structure):
    _fields_ = [
        *((name, PTR) for name in ("latent", "schedule", "noise")),
        *((name, SIZE) for name in ("latent_elements", "schedule_count", "noise_elements")),
        ("guidance", C.c_float), ("staged_ids", PTR), ("output_ids", PTR),
        ("staged_capacity", SIZE), ("output_capacity", SIZE), ("should_cancel", PTR), ("cancel_context", PTR),
    ]


def run_session(tmp_path, torch, geometry, weights, prompt_ids, *, blas=False):
    prompt_rows, rows = len(prompt_ids), 3
    capacity = prompt_rows + 2 * rows
    vocabulary, kv_heads, dim = (geometry[name] for name in ("vocabulary", "kv_heads", "dim"))
    graph, config, schema = build_circuit(prompt_rows, rows, cache_capacity=capacity, **geometry, blas=blas)
    inputs = {name: bits(weights[name]) if dtype == "BF16" else weights[name].contiguous().numpy()
              for name, (_, dtype) in schema.items()}
    exports = ("ck_prefill_diffusion_session", "ck_append_diffusion_session", "ck_generate_diffusion_block")
    _, arena, view, library = compile_checked_weight_graph(
        tmp_path, graph, inputs, schema, source_config=config, target_isa="avx2" if blas else "portable",
        runtime_sources=("src/runtime/checked_diffusion_session.c", "src/runtime/checked_diffusion_schedule.c"),
        extra_exports=exports,
    )
    address = C.addressof(arena)
    checked = load_exports(tmp_path, library)
    assert checked["properties"]["cache_capacity"] == capacity
    assert checked["arena_bytes"] == len(arena)
    types = {np.uint16: "bf16", np.float64: "fp64", np.float32: "fp32", np.int32: "i32"}

    def offset(name, kind):
        port = checked["ports"][name]
        allocation = view(name, kind)
        assert port["dtype"] == types[kind] and port["bytes"] == allocation.nbytes
        assert port["offset"] == allocation.ctypes.data - address
        return port["offset"]

    def clean(kind):
        entry = getattr(library, checked["entries"][kind])
        return Clean(C.cast(entry, PTR), *(offset(kind + "_" + name, np.int32)
                                          for name in ("prefix_tokens", "positions", "ranges")))

    planes, caches = [], []
    for i in range(geometry["layers"]):
        for kind, produced in (("key", "k_rotary"), ("value", "v")):
            name = f"step_cond_layer{i}_prefix_{kind}"
            assert all(name not in writes for writes in checked["entry_writes"].values())
            for phase in ("prefill", "append"):
                assert f"{phase}_layer{i}_{produced}" in checked["entry_writes"][checked["entries"][phase]]
            planes.append(Plane(offset(name, np.uint16), offset(f"prefill_layer{i}_{produced}", np.uint16),
                                offset(f"append_layer{i}_{produced}", np.uint16)))
            caches.append(view(name, np.uint16))
    plane_array = (Plane * len(planes))(*planes)
    prefill, append, generate = (getattr(library, name) for name in exports)
    prefill.argtypes = append.argtypes = [C.POINTER(Session), PTR, SIZE, PTR, PTR]
    generate.argtypes = [C.POINTER(Session), C.POINTER(Block)]
    step = getattr(library, checked["entries"]["step"])
    step.argtypes = [PTR, SIZE]
    step.restype = C.c_int
    step_pointer = C.cast(step, PTR)
    traced = []

    @ENTRY
    def traced_step(ptr, length):
        status = step(ptr, length)
        if status == 0:
            traced.append((view("step_next_latent", np.float64).copy(), view("step_cond_ids", np.int32).copy()))
        return status

    kinds = (np.float64,) * 4 + (np.float32,) * 2 + (np.int32,)
    state = Session(
        address, len(arena), clean("prefill"), clean("append"), C.cast(traced_step, PTR),
        Step(*(offset("step_" + name, dtype) for name, dtype in zip(STEP_FIELDS, kinds, strict=True))),
        *(offset("step_" + name, dtype) for name, dtype in (
            ("positions", np.int32), ("cond_ranges", np.int32), ("uncond_ranges", np.int32), ("guidance", np.float32))),
        plane_array, len(planes), kv_heads * dim, capacity, prompt_rows, rows,
        rows * geometry["latent_width"], vocabulary, 0,
    )
    s = C.byref(state)
    prompt = np.array(prompt_ids, np.int32)
    assert append(s, prompt.ctypes.data, rows, None, None) == -1
    assert prefill(s, prompt.ctypes.data, len(prompt), None, None) == 0
    assert state.valid_rows == prompt_rows
    for cache in caches:
        np.testing.assert_array_equal(cache[prompt_rows:], 0)

    def snapshot():
        return [cache.copy() for cache in caches]

    def compare_cache(old):
        for cache, before in zip(caches, old, strict=True):
            np.testing.assert_array_equal(cache, before)

    before = snapshot()
    invalid = np.array([0, 1, vocabulary], np.int32)
    assert append(s, invalid.ctypes.data, rows, None, None) == -1
    invalid_prompt = prompt.copy()
    invalid_prompt[-1] = vocabulary
    assert prefill(s, invalid_prompt.ctypes.data, prompt_rows, None, None) == -1
    assert state.valid_rows == prompt_rows
    compare_cache(before)

    weights_table = weights["encoder.embed_tokens.weight"]
    p = torch.arange(prompt_rows, dtype=torch.int32)
    _, reference_cache = encoder(weights_table[list(prompt_ids)], p, p[:, None] >= p[None, :], weights, geometry)
    rng = torch.Generator().manual_seed(19057)
    gamma = torch.tensor([2.0, 0.5, -0.75, -2.5], dtype=torch.float64)
    schedule = gamma.numpy()
    blocks = []
    all_ids, native_inputs = [], []
    for i in range(2):
        width = geometry["latent_width"]
        latent = torch.randn((rows, width), dtype=torch.float64, generator=rng)
        noises = torch.randn((len(schedule) - 1, rows, width), dtype=torch.float64, generator=rng)
        raw_latent, raw_noises = latent.numpy(), noises.numpy()
        output, staging = np.full(rows + 1, -91, np.int32), np.full(rows, -31, np.int32)
        request = Block(raw_latent.ctypes.data, schedule.ctypes.data, raw_noises.ctypes.data,
                        raw_latent.size, len(schedule), raw_noises.size, 1.75,
                        staging.ctypes.data, output.ctypes.data, len(staging), len(output), None, None)
        r = C.byref(request)
        valid = state.valid_rows
        previous = snapshot()
        if not blas:
            if i == 0:
                request.output_capacity = rows - 1
                assert generate(s, r) == -1
                request.output_capacity = len(output)
                compare_cache(previous)
                np.testing.assert_array_equal(output, -91)

                append_status = []

                @ENTRY
                def failed_append(ptr, length):
                    append_status.append(successful_append(ptr, length))
                    return -6

                saved_append = state.append.entry
                successful_append = ENTRY(saved_append)
                state.append.entry = C.cast(failed_append, PTR)
                assert generate(s, r) == -6
                assert append_status == [0]
                state.append.entry = saved_append
                compare_cache(previous)
                assert state.valid_rows == valid
                np.testing.assert_array_equal(output, -91)
            else:
                append_status = []
                saved_append = state.append.entry
                successful_append = ENTRY(saved_append)

                @ENTRY
                def observed_append(ptr, length):
                    status = successful_append(ptr, length)
                    append_status.append(status)
                    return status

                @CANCEL
                def after_append(_):
                    return int(bool(append_status))

                state.append.entry = C.cast(observed_append, PTR)
                request.should_cancel = C.cast(after_append, PTR)
                assert generate(s, r) == -7
                request.should_cancel = None
                state.append.entry = saved_append
                assert append_status == [0]
                compare_cache(previous)
                assert state.valid_rows == valid
                np.testing.assert_array_equal(output, -91)
        traced.clear()
        assert generate(s, r) == 0
        assert state.valid_rows == valid + rows
        assert len(traced) == len(schedule)
        assert output[-1] == -91
        for cache, old in zip(caches, previous, strict=True):
            np.testing.assert_array_equal(cache[:valid], old[:valid])
            np.testing.assert_array_equal(cache[valid + rows:], 0)

        absolute = torch.arange(valid, valid + rows, dtype=torch.int32)
        causal = absolute[:, None] >= torch.arange(valid + rows, dtype=torch.int32)
        # Compare an independent iterative trajectory using only the oracle's prefix.
        z, self_condition = latent, torch.zeros_like(latent).float()
        visible = torch.ones((rows, valid + rows), dtype=torch.bool)
        deltas = []
        for j, (native_z, native_ids) in enumerate(traced):
            last = j == len(schedule) - 1
            pair = gamma[j:j+1].repeat(2) if last else gamma[j:j+2]
            result = reference(weights, geometry, z, torch.zeros_like(z) if last else noises[j],
                               self_condition, pair, 1.75, absolute, visible, reference_cache)
            z, self_condition = result["next_latent"], result["cond_mean"]
            deltas.append({"latent_max": float(np.max(np.abs(native_z - z.numpy()))),
                           "conditional_ids": int(np.sum(native_ids == result["cond_ids"].numpy()))})
        expected_ids = result["cond_ids"].tolist()
        _, new_rows = encoder(weights_table[expected_ids], absolute, causal, weights, geometry, reference_cache)
        prefix_error = max(float(np.abs(widen(native[valid:valid+rows]) -
                           expected.transpose(0, 1).reshape(rows, -1).float().numpy()).max())
                           for native, expected in zip(caches, (x for pair in new_rows for x in pair), strict=True))
        reference_cache = [tuple(torch.cat((a, b), dim=1) for a, b in zip(old, new, strict=True))
                           for old, new in zip(reference_cache, new_rows, strict=True)]
        all_ids.extend(output[:-1].tolist())
        blocks.append({"ids": output[:-1].tolist(), "oracle_ids": expected_ids, "appended_kv_max": prefix_error,
                       "trajectory": deltas, "cache_rows": int(state.valid_rows)})
        native_inputs.append((raw_latent, raw_noises))

    full = snapshot()
    output[:] = -91
    assert generate(s, r) == -1
    compare_cache(full)
    np.testing.assert_array_equal(output, -91)
    assert state.valid_rows == capacity

    # Repeat both blocks with direct native dispatch after a different prompt.
    state.step = step_pointer
    other = (prompt + 1) % vocabulary
    assert prefill(s, other.ctypes.data, len(prompt), None, None) == 0
    assert any(not np.array_equal(cache, old) for cache, old in zip(caches, before, strict=True))
    if not blas:
        assert generate(s, r) == 0
    assert prefill(s, prompt.ctypes.data, len(prompt), None, None) == 0
    compare_cache(before)
    for (initial, noise), block in zip(native_inputs, blocks, strict=True):
        request.latent, request.noise = initial.ctypes.data, noise.ctypes.data
        assert generate(s, r) == 0
        np.testing.assert_array_equal(output[:-1], block["ids"])
    compare_cache(full)
    return {"blocks": blocks, "committed_ids": all_ids, "request_recovery": True}


def test_native_sigma_two_blocks(tmp_path):
    torch = pytest.importorskip("torch")
    geometry = dict(layers=2, vocabulary=19, width=11, heads=4, kv_heads=2, dim=6, inner=17,
                    latent_width=5, bands=4)
    _, _, schema = build_circuit(3, 3, cache_capacity=9, **geometry)
    rng = torch.Generator().manual_seed(573)
    weights = {}
    for name, (shape, dtype) in schema.items():
        values = torch.randn(shape, generator=rng) * 0.23
        weights[name] = values.bfloat16() if dtype == "BF16" else values
    weights["normalized_embedding"] = torch.nn.functional.normalize(weights["normalized_embedding"], dim=-1)
    weights["inv_freq"] = torch.tensor([1.0, 0.1, 0.001])
    report = run_session(tmp_path, torch, geometry, weights, (1, 7, 12))
    for block in report["blocks"]:
        assert block["ids"] == block["oracle_ids"], report
        assert block["appended_kv_max"] <= .015625, report
        assert max(d["latent_max"] for d in block["trajectory"]) <= .008, report
        assert all(d["conditional_ids"] == 3 for d in block["trajectory"]), report


def test_pinned_sigma_two_blocks(tmp_path, record_property):
    checkpoint = os.environ.get("CK_SIGMA_BF16_CHECKPOINT")
    if not checkpoint or not os.environ.get("CK_BLAS_LIBRARY"):
        pytest.skip("configure CK_SIGMA_BF16_CHECKPOINT and CK_BLAS_LIBRARY for the pinned replay")
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors")
    torch.set_num_threads(4)
    geometry = dict(layers=26, vocabulary=131072, width=3072, heads=32, kv_heads=8, dim=128, inner=9216,
                    latent_width=16, bands=32)
    ids = (1, 1784, 8961, 1307, 5498, 1395)
    _, _, schema = build_circuit(len(ids), 3, cache_capacity=len(ids) + 6, **geometry, blas=True)
    weights = pinned_weights(Path(checkpoint), schema, torch, safetensors)
    report = run_session(tmp_path, torch, geometry, weights, ids, blas=True)
    record_property("metrics", json.dumps(report))
    for block in report["blocks"]:
        assert block["ids"] == block["oracle_ids"], report
        assert block["appended_kv_max"] <= .25, report
        assert max(d["latent_max"] for d in block["trajectory"]) <= .08, report
        assert all(d["conditional_ids"] == 3 for d in block["trajectory"]), report
