"""Compare generated greedy codebook feedback for a single Drama frame."""
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.tts.build_drama3_bf16_codebook_frame import build_circuit, build_prefill_circuit


def bits(x):
    import torch
    return x.contiguous().view(torch.uint16).numpy()


def oracle_layer(hidden, code, table, weights, key, embed_extra, heads, kv_heads, dim, trace=None, operands=None, history=None):
    import torch
    from torch.nn import functional as F

    def norm(x, name):
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) *
                (1 + weights[key + name].float())).bfloat16()

    def projection(x, name):
        return F.linear(x, weights[key + name])

    def checkpoint(name, x):
        if trace is not None:
            trace[name] = x
        return operands(name) if operands is not None else x

    inp = checkpoint("embedding", table[code].unsqueeze(0))
    if embed_extra is not None:
        inp = checkpoint("enriched", inp + embed_extra)
    e = checkpoint("enorm", norm(inp, "enorm.weight"))
    h = checkpoint("hnorm", norm(hidden, "hnorm.weight"))
    concat = checkpoint("concat", torch.cat((e, h), dim=-1))
    hidden = checkpoint("mixed", projection(concat, "eh_proj.weight"))
    layer = "transformer_layer."
    attn = layer + "attention."
    normalized = checkpoint("attn_norm", norm(hidden, layer + "attention_norm.weight"))
    # With one causal memory row, each value head is broadcast to its query group.
    # The first-layer regression separately exercises Q/K and multiple keys.
    value = checkpoint("v", projection(normalized, attn + "v.weight"))
    context = value.reshape(kv_heads, dim).repeat_interleave(heads // kv_heads, dim=0).reshape(1, -1)
    if history is not None:
        prefix_key, prefix_value, position = history
        frequencies = weights["inv_freq"]
        angles = position * frequencies
        rotary = 2 * len(frequencies)

        def rotate(port, count):
            raw = projection(normalized, attn + port + ".weight").reshape(count, dim)
            x = norm(raw, attn + port + "_norm.weight")
            even, odd = x[..., :rotary:2].float(), x[..., 1:rotary:2].float()
            a = torch.stack((even * angles.cos() - odd * angles.sin(),
                             odd * angles.cos() + even * angles.sin()), dim=-1)
            return torch.cat((a.flatten(-2).bfloat16(), x[..., rotary:]), dim=-1)

        query, current_key = rotate("q", heads), rotate("k", kv_heads)
        value = value.reshape(kv_heads, dim)
        keys = torch.cat((prefix_key, current_key[None]), dim=0).transpose(0, 1)
        values = torch.cat((prefix_value, value[None]), dim=0).transpose(0, 1)
        ratio = heads // kv_heads
        context = F.scaled_dot_product_attention(
            query[:, None, :], keys.repeat_interleave(ratio, dim=0),
            values.repeat_interleave(ratio, dim=0),
        ).reshape(1, -1)
    context = checkpoint("context", context)
    attn_gate = checkpoint("attention_gate", projection(normalized, attn + "attention_gate.weight"))
    context = checkpoint("gated_context", context * attn_gate.sigmoid())
    projected = checkpoint("o", projection(context, attn + "wo.weight"))
    hidden = checkpoint("attn_residual", hidden + projected)
    normalized = checkpoint("ffn_norm", norm(hidden, layer + "ffn_norm.weight"))
    mlp = layer + "feed_forward."
    gate = checkpoint("gate", projection(normalized, mlp + "w1.weight"))
    up = checkpoint("up", projection(normalized, mlp + "w3.weight"))
    product = checkpoint("swiglu", F.silu(gate) * up)
    out = checkpoint("down", projection(product, mlp + "w2.weight"))
    hidden = checkpoint("pre_final_norm", hidden + out)
    return checkpoint("output", norm(hidden, "final_layernorm.weight"))


def difference(native, oracle):
    error = native.double() - oracle.double()
    return {"max_abs": float(error.abs().max()), "rmse": float(error.square().mean().sqrt()),
            "different": int((native != oracle).sum()), "elements": oracle.numel()}


def score_comparison(native, oracle):
    native_id, oracle_id = int(native.argmax()), int(oracle.argmax())
    ids = sorted({native_id, oracle_id})
    margins = {}
    for name, scores in (("native", native), ("oracle", oracle)):
        best = scores.float().topk(2).values.flatten()
        margins[name] = float(best[0] - best[1])
    return {"native_id": native_id, "oracle_id": oracle_id, "top_two_margin": margins,
            "competing_scores": {str(i): [float(native[0, i]), float(oracle[0, i])] for i in ids}}



def projection_receipt(native, oracle, source, weight, packet, key):
    # Retain one row and its input so the largest discrepancy can be replayed
    # on the very same BF16 operands without re-running the tower.
    channel = int((native.float() - oracle.float()).abs().argmax())
    result_native, result_oracle = float(native[0, channel]), float(oracle[0, channel])
    row = weight[channel]
    exact = float((source[0].double() * row.double()).sum())
    packet[key + "_input"] = bits(source[0])
    packet[key + "_weight"] = bits(row)
    return {"channel": channel, "native": result_native, "pytorch": result_oracle,
            "fp64_dot": exact, "midpoint": (result_native + result_oracle) * .5, "operand_key": key}



def run_frame(folder, torch, weights, tables, settings, *, check_continuation=False):
    from torch.nn import functional as F

    graph, config, schema = build_circuit(**settings)
    shared, residual = settings["shared_layers"], settings["residual_layers"]
    limit = settings["residual_vocab"]
    width, heads, kv_heads, dim = (settings[name] for name in ("width", "heads", "kv_heads", "dim"))
    stored = {name: weights[name].numpy() if name == "inv_freq" else bits(weights[name][:shape[0]])
              for name, (shape, _) in schema.items()}
    run, arena, view, _ = compile_checked_weight_graph(
        folder, graph, stored, schema, source_config=config,
        target_isa="avx2" if settings.get("blas") else "portable",
    )
    for name, table in tables.items():
        view(f"{name}_embedding_table", np.uint16)[:] = bits(table)
    rng = torch.Generator().manual_seed(2157)
    hidden = torch.randn((1, width), generator=rng).bfloat16()
    extra = (torch.randn((1, width), generator=rng) * 0.03).bfloat16()
    view("hidden", np.uint16)[:] = bits(hidden)
    view("unseen_sum", np.uint16)[:] = bits(extra)
    view("positions", np.int32)[:] = [0]
    view("ranges", np.int32)[:] = [[0, 1, 1, 1]]
    code_buffer = view("code_rows", np.int32)
    codes, reference_codes, metrics, traces, first_layer = [], [], [], [], []
    same_history = []
    replay_operands = {}
    for trial, code0 in enumerate((2, 3, 2)):
        code_buffer[0] = code0
        assert run(arena, len(arena)) == 0
        code = code0
        h = hidden
        expected = [code]
        stats, checkpoints = [], []
        native_h = h
        native_code = aligned_code = code
        aligned_h = hidden
        aligned_layers = []
        for i in range(shared + residual):
            common = i < shared
            index = i if common else i - shared
            label = f"{'shared' if common else 'residual'}{index}"
            key = f"audio_decoder_mtp{'_shared' if common else ''}.layers.{index}."
            enrichment = extra if i == 0 else None
            internal = {} if i == 0 else None
            h = oracle_layer(h, code, tables[label], weights, key, enrichment, heads, kv_heads, dim, internal)
            if internal is not None:
                first_layer.append({name: {"different": int((view(f"{label}_{name}", np.uint16) != bits(y)).sum()),
                                           "elements": y.numel()} for name, y in internal.items()})
            local = oracle_layer(native_h, native_code, tables[label], weights, key, enrichment, heads, kv_heads, dim)
            native_h = torch.from_numpy(view(f"{label}_output", np.uint16).copy()).view(torch.bfloat16)
            checkpoints.append({"layer": label, "chain_error": float((native_h.float() - h.float()).abs().max()),
                                "local_error": float((native_h.float() - local.float()).abs().max())})

            # Both paths consume CKE's preceding selections. A mismatch here
            # cannot be caused by feeding different code IDs into the tower.
            if trial < 2:
                def capture(name):
                    return torch.from_numpy(view(f"{label}_{name}", np.uint16).copy()).view(torch.bfloat16)

                chain, isolated = {}, {}
                aligned_h = oracle_layer(aligned_h, aligned_code, tables[label], weights, key, enrichment,
                                         heads, kv_heads, dim, chain)
                prior_hidden = hidden if i == 0 else previous_native_h
                oracle_layer(prior_hidden, aligned_code, tables[label], weights, key, enrichment,
                             heads, kv_heads, dim, isolated, capture)
                stages = {name: {"same_history": difference(capture(name), candidate),
                                 "identical_operands": difference(capture(name), isolated[name])}
                          for name, candidate in chain.items()}
                aligned_layer = {"layer": label, "input_code": aligned_code, "stages": stages}
                if common and index < 2:
                    for name, input_name, weight_suffix in (
                        ("mixed", "concat", "eh_proj.weight"),
                        ("gate", "ffn_norm", "transformer_layer.feed_forward.w1.weight"),
                        ("up", "ffn_norm", "transformer_layer.feed_forward.w3.weight"),
                    ):
                        if stages[name]["identical_operands"]["different"]:
                            operand_key = f"semantic{code0}_{label}_{name}"
                            stages[name]["rounding"] = projection_receipt(
                                capture(name), isolated[name], capture(input_name), weights[key + weight_suffix],
                                replay_operands, operand_key,
                            )
                aligned_layers.append(aligned_layer)
                previous_native_h = native_h
            if common:
                continue
            oracle = F.linear(h, weights[f"audio_decoder_mtp.heads.{index}.weight"])[..., :limit]
            code = int(oracle.argmax(-1).item())
            expected.append(code)
            native_code = int(view(f"residual{index}_code", np.int32)[0])
            score = view(f"residual{index}_scores", np.uint16)
            native = (score.astype(np.uint32) << 16).view(np.float32)
            if trial < 2:
                head = weights[f"audio_decoder_mtp.heads.{index}.weight"][:limit]
                native_scores = torch.from_numpy(score.copy()).view(torch.bfloat16)
                aligned_scores = F.linear(aligned_h, head)
                local_scores = F.linear(native_h, head)
                aligned_layer["readout"] = {
                    **score_comparison(native_scores, aligned_scores),
                    "identical_operands": difference(native_scores, local_scores),
                }
                aligned_code = native_code
            error = native.astype(np.float64) - oracle.float().numpy()
            stats.append({"max_abs": float(np.max(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
                          "greedy_margin": float(torch.diff(oracle.float().topk(2).values.flip(-1)).item())})
        actual = view("frame_codes", np.int32).copy()
        codes.append(actual.tolist())
        reference_codes.append(expected)
        metrics.extend(stats)
        traces.append(checkpoints)
        if trial < 2:
            same_history.append({"semantic_id": code0, "layers": aligned_layers})
    assert codes[0] == codes[2]
    saved = view("frame_codes", np.int32).copy()
    code_buffer[0] = settings["vocabulary"]
    assert run(arena, len(arena)) != 0
    np.testing.assert_array_equal(view("frame_codes", np.int32), saved)
    code_buffer[0] = 2
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("frame_codes", np.int32), saved)
    if check_continuation:
        check_next_frame(folder, torch, weights, tables, settings, stored, view, hidden, extra, saved)
    if replay_operands:
        np.savez(folder / "same-history-projection-operands.npz", **replay_operands)
    return {
        "codes": codes[:2], "reference_codes": reference_codes[:2], "layers": traces[:2],
        "first_layer": first_layer[:2], "scores": metrics, "same_history": same_history,
        "max_abs": max(v["max_abs"] for v in metrics), "worst_rmse": max(v["rmse"] for v in metrics),
    }


def tower_labels(settings):
    return [f"{kind}{index}" for kind in ("shared", "residual") for index in range(settings[f"{kind}_layers"])]


def check_next_frame(folder, torch, weights, tables, settings, stored, previous, hidden, extra, frame):
    from torch.nn import functional as F

    prefill_dir = folder / "teacher_forced_prefix"
    prefill_dir.mkdir()
    geometry = {name: value for name, value in settings.items() if name != "residual_vocab"}
    graph, config, schema = build_prefill_circuit(1, **geometry)
    prefill, prefill_arena, prepared, _ = compile_checked_weight_graph(
        prefill_dir, graph, {name: stored[name] for name in schema}, schema, source_config=config,
    )
    for name in ("hidden", "unseen_sum"):
        prepared(name, np.uint16)[:] = previous(name, np.uint16)
    for name in ("positions", "ranges", "code_rows"):
        prepared(name, np.int32)[:] = previous(name, np.int32)
    for label, table in tables.items():
        prepared(f"{label}_embedding_table", np.uint16)[:-1] = bits(table)
        prepared(f"{label}_embedding_table", np.uint16)[-1] = 0
    for index in range(1, settings["residual_layers"]):
        prepared(f"residual{index}_code_rows", np.int32)[:] = [frame[index]]
    assert prefill(prefill_arena, len(prefill_arena)) == 0
    for label in tables:
        for output in ("k_rotary", "v", "output"):
            name = f"{label}_{output}"
            np.testing.assert_array_equal(prepared(name, np.uint16), previous(name, np.uint16))

    continuation = folder / "next_frame"
    continuation.mkdir()
    graph, config, schema = build_circuit(prefix_rows=2, **settings)
    run, arena, view, _ = compile_checked_weight_graph(continuation, graph, stored, schema, source_config=config)
    heads, kv_heads, dim = (settings[name] for name in ("heads", "kv_heads", "dim"))
    next_hidden = hidden.flip(-1).contiguous()
    next_extra = extra.neg()
    semantic = 3  # a separately supplied semantic input for this synthetic frame
    view("hidden", np.uint16)[:] = bits(next_hidden)
    view("unseen_sum", np.uint16)[:] = bits(next_extra)
    view("positions", np.int32)[:] = [1]
    view("ranges", np.int32)[:] = [[0, 1, 2, 3]]
    view("code_rows", np.int32)[:] = [semantic]
    prefixes = {}
    for label, table in tables.items():
        view(f"{label}_embedding_table", np.uint16)[:] = bits(table)
        pair = []
        for kind, name in (("key", "k_rotary"), ("value", "v")):
            plane = view(f"{label}_prefix_{kind}", np.uint16)
            plane[0] = prepared(f"{label}_{name}", np.uint16)[0]
            plane[1] = 0x4400
            pair.append((plane, plane.copy()))
        prefixes[label] = pair

    assert run(arena, len(arena)) == 0
    h, code = next_hidden, semantic
    expected = [semantic]
    limit = settings["residual_vocab"]
    for i, label in enumerate(tower_labels(settings)):
        table = tables[label]
        shared = label.startswith("shared")
        index = int(label.removeprefix("shared" if shared else "residual"))
        key = f"audio_decoder_mtp{'_shared' if shared else ''}.layers.{index}."
        k, v = (torch.from_numpy(saved[:1].copy()).view(torch.bfloat16).reshape(1, kv_heads, dim)
                for _, saved in prefixes[label])
        h = oracle_layer(h, code, table, weights, key, next_extra if i == 0 else None,
                         heads, kv_heads, dim, history=(k, v, 1))
        if not shared:
            oracle = F.linear(h, weights[f"audio_decoder_mtp.heads.{index}.weight"][:limit])
            native = torch.from_numpy(view(f"{label}_scores", np.uint16).copy()).view(torch.bfloat16)
            torch.testing.assert_close(native.float(), oracle.float(), atol=.016, rtol=0)
            code = int(oracle.argmax(-1).item())
            expected.append(code)

    result = view("frame_codes", np.int32).copy()
    np.testing.assert_array_equal(result, expected)
    view("ranges", np.int32)[0, -1] = 4
    assert run(arena, len(arena)) != 0
    np.testing.assert_array_equal(view("frame_codes", np.int32), result)
    view("ranges", np.int32)[0, -1] = 3
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("frame_codes", np.int32), result)
    for pair in prefixes.values():
        for plane, saved in pair:
            np.testing.assert_array_equal(plane, saved)

    forced_dir = folder / "forced_next_frame"
    forced_dir.mkdir()
    graph, cfg, schema = build_circuit(prefix_rows=2, forced_feedback=True, **settings)
    force, force_arena, bound, _ = compile_checked_weight_graph(forced_dir, graph, stored, schema, source_config=cfg)
    layer_inputs = ("embedding_table", "prefix_key", "prefix_value")
    for name in ("hidden", "unseen_sum", *(f"{label}_{name}" for label in tables for name in layer_inputs)):
        bound(name, np.uint16)[:] = view(name, np.uint16)
    for name in ("positions", "ranges", "code_rows"):
        bound(name, np.int32)[:] = view(name, np.int32)
    for index, code in enumerate(result[1:-1]):
        bound(f"residual{index}_forced_code", np.int32)[:] = [code]
    assert force(force_arena, len(force_arena)) == 0
    np.testing.assert_array_equal(bound("frame_codes", np.int32), result)
    bound("residual0_forced_code", np.int32)[:] = [settings["vocabulary"]]
    assert force(force_arena, len(force_arena)) != 0
    np.testing.assert_array_equal(bound("frame_codes", np.int32), result)


def test_generated_codebook_feedback(tmp_path):
    torch = pytest.importorskip("torch")
    settings = dict(shared_layers=2, residual_layers=3, residual_vocab=5, vocabulary=7,
                    width=11, heads=4, kv_heads=2, dim=6, rotary=4, inner=13)
    _, _, schema = build_circuit(**settings)
    rng = torch.Generator().manual_seed(1826)
    weights = {name: (torch.randn(shape, generator=rng) * 0.17).bfloat16()
               for name, (shape, dtype) in schema.items() if dtype == "BF16"}
    weights["inv_freq"] = torch.tensor([1.0, 0.01])
    tables = {f"{kind}{index}": torch.randn((7, 11), generator=rng).bfloat16()
              for kind, count in (("shared", 2), ("residual", 3)) for index in range(count)}
    report = run_frame(tmp_path, torch, weights, tables, settings, check_continuation=True)
    assert report["codes"] == report["reference_codes"], report
    assert report["max_abs"] <= 0.015625, report
    for codes, aligned in zip(report["codes"], report["same_history"], strict=True):
        inputs = [layer["input_code"] for layer in aligned["layers"]]
        assert inputs == [codes[0]] * (settings["shared_layers"] + 1) + codes[1:-1]
        last = aligned["layers"][-1]
        assert last["readout"]["native_id"] == codes[-1]
        assert last["stages"]["embedding"]["identical_operands"]["different"] == 0


def pinned_tower_weights(root, torch):
    settings = dict(shared_layers=4, residual_layers=9, residual_vocab=1024, vocabulary=4096,
                    width=2048, heads=16, kv_heads=2, dim=256, rotary=64, inner=6144, blas=True)
    artifacts = {
        14: (168607328, "4e289ef05b3b3affb7b7497095db9d28f6b115a07a58b9487e13f75a29aa9020",
             "ea53c55090ea8cbf6f1bef84fe1436f9343cb100e07dd69022a9829707a5e81d"),
        15: (15128, "384f507bb157270941fe1300114bcefb0875924ee1404fd747c87f34f2dd9286",
             "133f9be280d3895c71aef0f09430a044ef41b0f0beac8e99521e288772b2970d"),
    }
    root = Path(root)
    weights, tables = {}, {}
    for shard, (offset, header_sha, payload_sha) in artifacts.items():
        header = (root / f"shard{shard}.header").read_bytes()
        assert hashlib.sha256(header).hexdigest() == header_sha
        data = root / f"tower-shard{shard}.bin"
        with data.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == payload_sha
        n, = struct.unpack_from("<Q", header)
        contents = np.memmap(data, mode="r", dtype=np.uint16)
        for key, value in json.loads(header[8:8+n]).items():
            if not key.startswith(("audio_decoder_mtp_shared.", "audio_decoder_mtp.")):
                continue
            start, end = ((a + n + 8 - offset) // 2 for a in value["data_offsets"])
            assert value["dtype"] == "BF16" and 0 <= start < end <= contents.size
            t = torch.from_numpy(contents[start:end].reshape(value["shape"]).copy()).view(torch.bfloat16)
            if key.endswith(("semantic_embedding.weight", "codebook_embedding.weight")):
                label = "shared" if key.startswith("audio_decoder_mtp_shared") else "residual"
                tables[label + key.split(".")[2]] = t
            elif key.endswith(".wqkv.weight"):
                for port, matrix in zip(("q", "attention_gate", "k", "v"), t.split((4096, 4096, 512, 512)), strict=True):
                    weights[key.removesuffix("wqkv.weight") + port + ".weight"] = matrix
            else:
                weights[key] = t
    weights["inv_freq"] = 10000000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    return weights, tables, settings


def test_pinned_codebook_frame(tmp_path, record_property):
    root = os.environ.get("CK_DRAMA3_TOWER_RANGES")
    if not root or not os.environ.get("CK_BLAS_LIBRARY"):
        pytest.skip("configure CK_DRAMA3_TOWER_RANGES and CK_BLAS_LIBRARY for the pinned tower")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(4)
    weights, tables, settings = pinned_tower_weights(root, torch)
    report = run_frame(tmp_path, torch, weights, tables, settings)
    record_property("reports", json.dumps(report))
    assert report["codes"] == report["reference_codes"]
    assert report["max_abs"] <= 0.5 and report["worst_rmse"] <= 0.05


def test_publisher_text_conditioned_tower(tmp_path, record_property):
    """Check the tower with publisher text history and then native K/V feedback."""
    from version.v8.tts.capture_drama3_two_frame_reference import (
        CHECKPOINT_REVISION, PUBLISHER_REVISION, digest,
    )

    root = os.environ.get("CK_DRAMA3_TOWER_RANGES")
    capture = os.environ.get("CK_DRAMA3_TWO_FRAME_REFERENCE")
    if not root or not capture or not os.environ.get("CK_BLAS_LIBRARY"):
        pytest.skip("configure tower ranges, LP64 BLAS and a captured publisher text request")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(4)
    capture = Path(capture)
    receipt = json.loads((capture / "report.json").read_text())
    assert receipt["schema"] == "cke.drama3.reference.two_frames.v1"
    assert receipt["checkpoint_revision"] == CHECKPOINT_REVISION
    assert receipt["publisher_revision"] == PUBLISHER_REVISION
    assert receipt["frames_sha256"] == digest(capture / "frames.npz")
    with np.load(capture / "frames.npz", allow_pickle=False) as ref:
        rows = ref["frame_rows"]
        assert rows.shape == (2, 11) and np.issubdtype(rows.dtype, np.integer)
        assert rows.tolist() == receipt["frames"]
        assert np.array_equal(rows[:, 0] - 248099, rows[:, 1])
        assert ((rows[:, 1:] >= 0) & (rows[:, 1:] < 4096)).all()
        prompt_rows = len(ref["prompt_rows"])
        assert prompt_rows == receipt["prompt_rows"] and prompt_rows > 1
        assert np.array_equal(ref["step0_input_row"], ref["prompt_rows"][-1:])
        assert np.array_equal(ref["step1_input_row"], rows[:1])
        weights, tables, settings = pinned_tower_weights(root, torch)
        settings["residual_vocab"] = settings["vocabulary"]  # publisher scores every raw tower ID
        graph, cfg, schema = build_circuit(prefix_rows=prompt_rows, forced_feedback=True, **settings)
        stored = {name: weights[name].numpy() if name == "inv_freq" else bits(weights[name])
                  for name in schema}
        run, arena, view, _ = compile_checked_weight_graph(
            tmp_path, graph, stored, schema, source_config=cfg, target_isa="avx2",
        )
        for label, table in tables.items():
            view(f"{label}_embedding_table", np.uint16)[:] = bits(table)

        def set_bf16(name, values):
            slot = view(name, np.uint16)
            assert values.dtype == slot.dtype and values.shape == slot.shape
            slot[:] = values

        # The capture uses a text-only prompt, so each following prompt row
        # is a nonsemantic teacher label. The sentinel selects the zero row.
        prefill_rows = prompt_rows - 1
        assert not np.any((ref["prompt_rows"][:, 0] >= 248099) & (ref["prompt_rows"][:, 0] <= 252194))
        np.testing.assert_array_equal(ref["prefill_next_rows"], ref["prompt_rows"][1:])
        np.testing.assert_array_equal(ref["prefill_positions"], np.arange(prefill_rows))
        prefill_dir = tmp_path / "prefill"
        prefill_dir.mkdir()
        prefill_args = {name: value for name, value in settings.items() if name != "residual_vocab"}
        graph, cfg, schema = build_prefill_circuit(prefill_rows, **prefill_args)
        prefill, prefill_arena, prepared, _ = compile_checked_weight_graph(
            prefill_dir, graph, {name: stored[name] for name in schema}, schema,
            source_config=cfg, target_isa="avx2",
        )
        prepared("hidden", np.uint16)[:] = ref["prefill_hidden"]
        prepared("unseen_sum", np.uint16).fill(0)
        prepared("positions", np.int32)[:] = ref["prefill_positions"]
        prepared("ranges", np.int32)[:] = [[0, p + 1, p + 1, p + 1] for p in range(prefill_rows)]
        prepared("code_rows", np.int32).fill(settings["vocabulary"])
        for index in range(1, settings["residual_layers"]):
            prepared(f"residual{index}_code_rows", np.int32).fill(settings["vocabulary"])
        for label, table in tables.items():
            prepared(f"{label}_embedding_table", np.uint16)[:-1] = bits(table)
            prepared(f"{label}_embedding_table", np.uint16)[-1].fill(0)
        assert prefill(prefill_arena, len(prefill_arena)) == 0
        prefill_drift = {}
        for i, label in enumerate(tower_labels(settings)):
            for kind, current in (("key", "k_rotary"), ("value", "v")):
                actual = prepared(f"{label}_{current}", np.uint16)
                expected = ref[f"tower_{i}_{kind}"][:prefill_rows].reshape(actual.shape)
                prefill_drift[f"{label}_{kind}"] = difference(
                    torch.from_numpy(actual).view(torch.bfloat16), torch.from_numpy(expected).view(torch.bfloat16),
                )

        def replay(step, native_feedback=None, *, native_prefill=False):
            expected = rows[step, 1:]
            prefix = f"step{step}_"
            position = int(ref[prefix + "position"][0])
            assert position == prompt_rows - 1 + step
            set_bf16("hidden", ref[prefix + "backbone_hidden"])
            set_bf16("unseen_sum", ref[prefix + "enrichment"])
            view("code_rows", np.int32)[:] = [expected[0]]
            view("positions", np.int32)[:] = [position]
            view("ranges", np.int32)[:] = [[0, position, prompt_rows, prompt_rows + 1]]
            for index in range(settings["residual_layers"] - 1):
                view(f"residual{index}_forced_code", np.int32)[:] = [expected[index + 1]]
            planes = {}
            for i, label in enumerate(tower_labels(settings)):
                for kind, current in (("key", "k_rotary"), ("value", "v")):
                    data = ref[f"tower_{i}_{kind}"]
                    assert data.dtype == np.uint16 and data.shape == (prompt_rows + 1, 2, 256)
                    plane = view(f"{label}_prefix_{kind}", np.uint16)
                    plane.fill(0)
                    plane[:position] = data[:position].reshape(position, -1)
                    if native_prefill:
                        plane[:prefill_rows] = prepared(f"{label}_{current}", np.uint16)
                    if native_feedback is not None:
                        plane[position - 1] = native_feedback[label, current][0]
                    planes[label, current] = (plane, plane.copy())
            assert run(arena, len(arena)) == 0
            scores = []
            for index in range(settings["residual_layers"]):
                native = torch.from_numpy(view(f"residual{index}_scores", np.uint16).copy()).view(torch.bfloat16)
                oracle = torch.from_numpy(ref[prefix + f"head{index}_scores"].copy()).view(torch.bfloat16)
                scores.append({**difference(native, oracle), **score_comparison(native, oracle)})
            for plane, before in planes.values():
                np.testing.assert_array_equal(plane, before)
            result = {"frame": step, "native_prefix_tail": native_feedback is not None, "native_prefill": native_prefill,
                      "expected_ids": expected.tolist(), "native_ids": view("frame_codes", np.int32).tolist(),
                      "scores": scores}
            return result, {(label, current): view(f"{label}_{current}", np.uint16).copy()
                            for label, current in planes}

        first, native_tail = replay(0)
        second, _ = replay(1)
        repeated_with_tail, _ = replay(1, native_tail)
        native_first, native_tail = replay(0, native_prefill=True)
        native_second, _ = replay(1, native_tail, native_prefill=True)
        results = [first, second, repeated_with_tail, native_first, native_second]
    record_property("text_conditioned_tower", json.dumps({"frames": results, "prefill_kv": prefill_drift}))
    for result in results:
        assert result["native_ids"] == result["expected_ids"]
        assert all(score["max_abs"] <= .5 and score["rmse"] <= .05 for score in result["scores"])
