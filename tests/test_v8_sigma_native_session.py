"""Exercise two committed Sigma blocks on one compiled native arena."""
import ctypes as C
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.test_v8_audio_noise_counter_oracle import oracle as gaussian_oracle
from tests.test_v8_sigma_native_step import bits, encoder, pinned_weights, reference, widen
from version.v8.scripts.build_sigma_native_session_v8 import build_circuit
from version.v8.scripts.checked_execution_exports_v8 import load_exports
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph

from version.v8.scripts.run_sigma_text_v8 import SigmaTextSession
from version.v8.scripts.sigma_session_v8 import Block, CANCEL, ENTRY, PTR, SIZE, Seeded, Session, bind_session


def run_session(tmp_path, torch, geometry, weights, prompt_ids, *, blas=False):
    prompt_rows, rows = len(prompt_ids), 3
    capacity = prompt_rows + 2 * rows
    vocabulary = geometry["vocabulary"]
    graph, config, schema = build_circuit(prompt_rows, rows, cache_capacity=capacity, **geometry, blas=blas)
    inputs = {name: bits(weights[name]) if dtype == "BF16" else weights[name].contiguous().numpy()
              for name, (_, dtype) in schema.items()}
    exports = ("ck_prefill_diffusion_session", "ck_append_diffusion_session", "ck_generate_diffusion_block")
    _, arena, view, library = compile_checked_weight_graph(
        tmp_path, graph, inputs, schema, source_config=config, target_isa="avx2" if blas else "portable",
        runtime_sources=("src/runtime/checked_diffusion_session.c", "src/runtime/checked_diffusion_schedule.c"),
        extra_exports=(*exports, "ck_generate_seeded_diffusion"),
    )
    checked = load_exports(tmp_path, library)
    state, _ = bind_session(tmp_path, arena, library, geometry)
    assert state.capacity == capacity
    caches = [view(f"step_cond_layer{i}_prefix_{kind}", np.uint16)
              for i in range(geometry["layers"]) for kind in ("key", "value")]
    prefill, append, generate = (getattr(library, name) for name in exports)
    prefill.argtypes = append.argtypes = [C.POINTER(Session), PTR, SIZE, PTR, PTR]
    generate.argtypes = [C.POINTER(Session), C.POINTER(Block)]
    seeded = library.ck_generate_seeded_diffusion
    seeded.argtypes = [C.POINTER(Session), C.POINTER(Seeded)]
    step = getattr(library, checked["entries"]["step"])
    step.argtypes = [PTR, SIZE]
    step.restype = C.c_int
    step_pointer = C.cast(step, PTR)
    traced = []
    workspace = np.full((4, rows, geometry["latent_width"]), -71., np.float64)
    seeded_inputs = []
    capture_seeded = False

    @ENTRY
    def traced_step(ptr, length):
        if capture_seeded and len(traced) % len(workspace) == 0:
            seeded_inputs.append(workspace.copy())
        status = step(ptr, length)
        if status == 0:
            traced.append((view("step_next_latent", np.float64).copy(), view("step_cond_ids", np.int32).copy()))
        return status

    state.step = C.cast(traced_step, PTR)
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
    def compare_block(valid, latent, noises, reference_cache, traced, generated):
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
        return {"ids": generated.tolist(), "oracle_ids": expected_ids, "appended_kv_max": prefix_error,
                "trajectory": deltas, "cache_rows": valid + rows}, reference_cache

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

        block, reference_cache = compare_block(valid, latent, noises, reference_cache, traced, output[:-1])
        blocks.append(block)
        all_ids.extend(block["ids"])
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
    seeded_output = np.full(2 * rows + 1, -81, np.int32)
    seeded_request = Seeded(
        prompt.ctypes.data, len(prompt), schedule.ctypes.data, len(schedule), 19057, 2 * rows, 1.75,
        workspace.ctypes.data, workspace.size, staging.ctypes.data, seeded_output.ctypes.data,
        len(staging), len(seeded_output), 99, None, None,
    )
    request_ptr = C.byref(seeded_request)
    if not blas:
        ascending = schedule[::-1].copy()
        for field, invalid in (("token_budget", 2 * rows + 1), ("output_capacity", rows - 1),
                               ("workspace_elements", workspace.size - 1),
                               ("workspace", workspace.ctypes.data + 1), ("output_ids", prompt.ctypes.data),
                               ("prompt_ids", invalid_prompt.ctypes.data), ("prompt_rows", 0),
                               ("schedule", ascending.ctypes.data)):
            original = getattr(seeded_request, field)
            setattr(seeded_request, field, invalid)
            assert seeded(s, request_ptr) == -1
            setattr(seeded_request, field, original)
            assert seeded_request.generated_rows == 99
            assert state.valid_rows == capacity
            np.testing.assert_array_equal(seeded_output, -81)
            np.testing.assert_array_equal(workspace, -71.)
            compare_cache(full)

    def seeded_trial(ids, budget, seed):
        nonlocal capture_seeded
        seeded_request.prompt_ids, seeded_request.prompt_rows = ids.ctypes.data, len(ids)
        seeded_request.token_budget, seeded_request.seed = budget, seed
        seeded_output.fill(-81)
        traced.clear()
        seeded_inputs.clear()
        state.step = C.cast(traced_step, PTR)
        capture_seeded = True
        status = seeded(s, request_ptr)
        capture_seeded = False
        state.step = step_pointer
        assert status == 0
        assert seeded_request.generated_rows == budget
        assert state.valid_rows == len(ids) + budget
        np.testing.assert_array_equal(seeded_output[budget:], -81)
        for cache in caches:
            np.testing.assert_array_equal(cache[state.valid_rows:], 0)
        assert len(seeded_inputs) == budget // rows
        assert len(traced) == len(seeded_inputs) * len(schedule)
        pos = torch.arange(len(ids), dtype=torch.int32)
        _, oracle_cache = encoder(weights_table[ids.tolist()], pos, pos[:, None] >= pos[None, :],
                                  weights, geometry)
        reports = []
        for i, noise_inputs in enumerate(seeded_inputs):
            np.testing.assert_allclose(
                noise_inputs.ravel(), gaussian_oracle(noise_inputs.size, seed, i, dtype=np.float64),
                atol=2e-14, rtol=0,
            )
            expected_trace = traced[i * len(schedule):(i + 1) * len(schedule)]
            block, oracle_cache = compare_block(
                len(ids) + i * rows, torch.from_numpy(noise_inputs[0]), torch.from_numpy(noise_inputs[1:]),
                oracle_cache, expected_trace, seeded_output[i * rows:(i + 1) * rows],
            )
            reports.append(block)
        return reports, snapshot(), np.stack(seeded_inputs)

    first, seed_cache, seed_inputs = seeded_trial(prompt, 2 * rows, 19057)
    assert not np.array_equal(seed_inputs[0], seed_inputs[1])
    shorter = prompt[:-1].copy()
    second, _, other_inputs = seeded_trial(shorter, rows, 151)
    assert not np.array_equal(seed_inputs[0], other_inputs[0])

    if not blas:
        # Stop in the second block after its first denoise step. A committed
        # block must survive and the partially generated block stays private.
        calls = []

        @ENTRY
        def count_step(ptr, length):
            calls.append(None)
            return step(ptr, length)

        @CANCEL
        def cancel_second(_):
            return int(len(calls) > len(schedule))

        seeded_request.should_cancel = C.cast(cancel_second, PTR)
        seeded_request.prompt_ids, seeded_request.prompt_rows = prompt.ctypes.data, len(prompt)
        seeded_request.seed, seeded_request.token_budget = 19057, 2 * rows
        seeded_output.fill(-81)
        state.step = C.cast(count_step, PTR)
        assert seeded(s, request_ptr) == -7
        assert len(calls) == len(schedule) + 1
        assert seeded_request.generated_rows == rows
        assert state.valid_rows == len(prompt) + rows
        np.testing.assert_array_equal(seeded_output[:rows], first[0]["ids"])
        np.testing.assert_array_equal(seeded_output[rows:], -81)
        for cache, expected in zip(caches, seed_cache, strict=True):
            np.testing.assert_array_equal(cache[:state.valid_rows], expected[:state.valid_rows])
            np.testing.assert_array_equal(cache[state.valid_rows:], 0)
        seeded_request.should_cancel = None

        # Repeat through the direct generated entry with no host callbacks.
        state.step = step_pointer
        assert seeded(s, request_ptr) == 0
        assert state.valid_rows == capacity
        assert seeded_request.generated_rows == 2 * rows
        np.testing.assert_array_equal(seeded_output, [*first[0]["ids"], *first[1]["ids"], -81])
        compare_cache(seed_cache)
        np.testing.assert_array_equal(workspace, seed_inputs[-1])

    return {"blocks": blocks, "seeded_blocks": first + second, "committed_ids": all_ids,
            "request_recovery": True}


def synthetic_weights(torch, schema):
    rng = torch.Generator().manual_seed(573)
    weights = {}
    for name, (shape, dtype) in schema.items():
        values = torch.randn(shape, generator=rng) * 0.23
        weights[name] = values.bfloat16() if dtype == "BF16" else values
    weights["normalized_embedding"] = torch.nn.functional.normalize(weights["normalized_embedding"], dim=-1)
    weights["inv_freq"] = torch.tensor([1.0, 0.1, 0.001])
    return weights


SMALL = dict(layers=2, vocabulary=19, width=11, heads=4, kv_heads=2, dim=6, inner=17,
             latent_width=5, bands=4)


def test_native_sigma_two_blocks(tmp_path):
    torch = pytest.importorskip("torch")
    _, _, schema = build_circuit(3, 3, cache_capacity=9, **SMALL)
    report = run_session(tmp_path, torch, SMALL, synthetic_weights(torch, schema), (1, 7, 12))
    for block in report["blocks"] + report["seeded_blocks"]:
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
    for block in report["blocks"] + report["seeded_blocks"]:
        assert block["ids"] == block["oracle_ids"], report
        assert block["appended_kv_max"] <= .25, report
        assert max(d["latent_max"] for d in block["trajectory"]) <= .08, report
        assert all(d["conditional_ids"] == 3 for d in block["trajectory"]), report


def compare_ddim_text(folder, session, torch, geometry, weights, text, seed, temperature):
    rows = session.state.block_rows
    gamma = torch.from_numpy(session.schedule)
    native_pointer = session.state.step
    native = ENTRY(native_pointer)
    trajectory = []
    layout = json.loads((folder / "layout.json").read_text())
    guided = next(item["abs_offset"] for item in layout["memory"]["activations"]["buffers"]
                  if item["name"] == "step_guided")

    @ENTRY
    def traced(ptr, size):
        status = native(ptr, size)
        if status == 0:
            bindings = session.state.step_bindings
            shape = (rows, geometry["latent_width"])
            latent_t = C.c_double * (rows * geometry["latent_width"])
            tensors = tuple(np.array(latent_t.from_address(ptr + offset)).reshape(shape)
                            for offset in (bindings.latent, guided, bindings.next_latent))
            ids = (C.c_int32 * rows).from_address(ptr + bindings.cond_ids)
            trajectory.append((*tensors, np.array(ids)))
        return status

    session.state.step = C.cast(traced, PTR)
    first = session.generate(text, seed=seed, budget=rows)
    assert first["finish_reason"] == "budget"
    assert len(trajectory) == len(gamma)
    prompt_ids = first["prompt_ids"]
    prompt_rows = len(prompt_ids)
    positions = torch.arange(prompt_rows, dtype=torch.int32)
    _, cache = encoder(weights["encoder.embed_tokens.weight"][prompt_ids], positions,
                       positions[:, None] >= positions[None, :], weights, geometry)
    block_positions = torch.arange(prompt_rows, prompt_rows + rows, dtype=torch.int32)
    z = torch.from_numpy(gaussian_oracle(rows * geometry["latent_width"], seed, 0, dtype=np.float64))
    z = z.reshape(rows, geometry["latent_width"])
    metrics = []
    self_condition = torch.zeros_like(z).float()
    for i, (native_in, native_estimate, next_latent, ids) in enumerate(trajectory):
        final = i == len(gamma) - 1
        pair = gamma[i:i + 1].repeat(2) if final else gamma[i:i + 2]
        oracle = reference(weights, geometry, z, torch.zeros_like(z), self_condition, pair, session.guidance,
                           block_positions, torch.ones((rows, prompt_rows + rows), dtype=torch.bool), cache)
        alpha, sigma = torch.sigmoid(-pair).sqrt(), torch.sigmoid(pair).sqrt()

        def transition(latent, estimate):
            if final:
                return latent
            epsilon = (latent - alpha[0] * estimate) / sigma[0].clamp(min=1e-8) / temperature
            corrected = (latent - sigma[0] * epsilon) / alpha[0].clamp(min=1e-8)
            return alpha[1] * corrected + sigma[1] * epsilon

        same_input = transition(torch.from_numpy(native_in), torch.from_numpy(native_estimate))
        np.testing.assert_allclose(next_latent, same_input.numpy(), atol=4e-13, rtol=2e-15)
        z = transition(z, oracle["guided"])
        self_condition = oracle["cond_mean"]
        error = float(np.max(np.abs(next_latent - z.numpy())))
        local = float(np.max(np.abs(next_latent - same_input.numpy())))
        estimate_error = float(np.max(np.abs(native_estimate - oracle["guided"].numpy())))
        agreeing = int(np.sum(ids == oracle["cond_ids"].numpy()))
        metrics.append({"latent_max": error, "transition_max": local, "guided_max": estimate_error,
                        "agreeing_ids": agreeing, "rows": rows})
    assert ids.tolist() == first["generated_ids"]
    session.state.step = native_pointer
    return first, metrics



def test_sigma_text_ddim_and_request_controls(tmp_path):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    rows, temperature, guidance, seed = 3, 0.3, 3.0, 145
    graph, config, schema = build_circuit(
        rows, rows, cache_capacity=9, score_temperature_milli=300, **SMALL,
    )
    weights = synthetic_weights(torch, schema)
    payloads = {name: bits(weights[name]) if dtype == "BF16" else weights[name].contiguous().numpy()
                for name, (_, dtype) in schema.items()}
    _, arena, _, library = compile_checked_weight_graph(
        tmp_path, graph, payloads, schema, source_config=config, target_isa="portable",
        runtime_sources=("src/runtime/checked_diffusion_session.c", "src/runtime/checked_diffusion_schedule.c"),
        extra_exports=("ck_generate_seeded_diffusion",),
    )

    class TextCodec:
        def encode(self, text):
            return SimpleNamespace(ids=[int(word) for word in text.split()])

        def decode(self, tokens, *, skip_special_tokens):
            assert skip_special_tokens
            return " ".join(str(token) for token in tokens)

    gamma = torch.tensor([2.0, .5, -.75, -2.5], dtype=torch.float64)
    session = SigmaTextSession(tmp_path, arena, library, SMALL, TextCodec(), gamma.numpy(), guidance)
    assert session.state.step_bindings.noise == SIZE(-1).value
    first, metrics = compare_ddim_text(tmp_path, session, torch, SMALL, weights, "1 7 12", seed, temperature)
    assert all(step["latent_max"] <= .01 and step["agreeing_ids"] == rows for step in metrics), metrics

    with pytest.raises(ValueError, match="capacity"):
        session.generate("1 2 3 4", seed=seed, budget=rows)
    with pytest.raises(ValueError, match="Budget"):
        session.generate("1 7", seed=seed, budget=1)
    short = session.generate("1 7", seed=seed, budget=rows)
    assert session.state.valid_rows == 5
    assert short["build_identity"] == first["build_identity"]
    stopped = session.generate("1 7 12", seed=seed, budget=2 * rows, stop_strings=[first["text"]])
    assert stopped["finish_reason"] == "stop_string"
    assert stopped["text"] == "" and stopped["generated_ids"] == first["generated_ids"]
    assert session.state.valid_rows == 6

    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls == 5

    cancelled = session.generate("1 7", seed=seed, budget=rows, should_cancel=cancel)
    assert cancelled["finish_reason"] == "cancelled" and cancelled["generated_ids"] == []
    recovered = session.generate("1 7 12", seed=seed, budget=rows)
    assert recovered["generated_ids"] == first["generated_ids"]


def test_pinned_sigma_text_ddim(tmp_path, record_property):
    from types import SimpleNamespace

    from version.v8.scripts.run_sigma_text_v8 import GEOMETRY, build_session

    checkpoint = os.environ.get("CK_SIGMA_BF16_CHECKPOINT")
    if not checkpoint or not os.environ.get("CK_BLAS_LIBRARY"):
        pytest.skip("configure CK_SIGMA_BF16_CHECKPOINT and CK_BLAS_LIBRARY for the pinned text harness")
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors")
    pytest.importorskip("tokenizers")
    args = SimpleNamespace(checkpoint=Path(checkpoint), threads=4, prompt_capacity=9, block=3, cache=12,
                           score_temperature_milli=300, sampler="ddim", steps=3, guidance=3.0, isa="avx2")
    session = build_session(args, tmp_path)
    _, _, schema = build_circuit(args.prompt_capacity, args.block, cache_capacity=args.cache,
                                  score_temperature_milli=300, **GEOMETRY, blas=True)
    weights = pinned_weights(args.checkpoint, schema, torch, safetensors)
    result, metrics = compare_ddim_text(tmp_path, session, torch, GEOMETRY, weights, "The capital of France is", 41, .3)
    record_property("ddim_metrics", json.dumps({"text_result": result, "trajectory": metrics}))
    # The DDIM score correction can amplify BF16 model drift. The local
    # transition has its own same-input gate. Report end-to-end latent drift
    # alongside the independent model's token-ID verdict.
    assert all(step["agreeing_ids"] == args.block for step in metrics), metrics
