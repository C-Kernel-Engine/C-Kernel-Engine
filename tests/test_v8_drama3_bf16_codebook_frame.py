"""Compare generated greedy codebook feedback for a single Drama frame."""
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.tts.build_drama3_bf16_codebook_frame import build_circuit


def bits(x):
    import torch
    return x.contiguous().view(torch.uint16).numpy()


def oracle_layer(hidden, code, table, weights, key, embed_extra, heads, kv_heads, dim, trace=None):
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
        return x

    inp = table[code].unsqueeze(0)
    if embed_extra is not None:
        inp = inp + embed_extra
    e = checkpoint("enorm", norm(inp, "enorm.weight"))
    h = checkpoint("hnorm", norm(hidden, "hnorm.weight"))
    hidden = checkpoint("mixed", projection(torch.cat((e, h), dim=-1), "eh_proj.weight"))
    layer = "transformer_layer."
    attn = layer + "attention."
    normalized = checkpoint("attn_norm", norm(hidden, layer + "attention_norm.weight"))
    # With one causal memory row, each value head is broadcast to its query group.
    # The first-layer regression separately exercises Q/K and multiple keys.
    value = checkpoint("v", projection(normalized, attn + "v.weight"))
    context = value.reshape(kv_heads, dim).repeat_interleave(heads // kv_heads, dim=0).reshape(1, -1)
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


def run_frame(folder, torch, weights, tables, settings):
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
    for code0 in (2, 3, 2):
        code_buffer[0] = code0
        assert run(arena, len(arena)) == 0
        code = code0
        h = hidden
        expected = [code]
        stats, checkpoints = [], []
        native_h = h
        native_code = code
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
            if common:
                continue
            oracle = F.linear(h, weights[f"audio_decoder_mtp.heads.{index}.weight"])[..., :limit]
            code = int(oracle.argmax(-1).item())
            expected.append(code)
            native_code = int(view(f"residual{index}_code", np.int32)[0])
            score = view(f"residual{index}_scores", np.uint16)
            native = (score.astype(np.uint32) << 16).view(np.float32)
            error = native.astype(np.float64) - oracle.float().numpy()
            stats.append({"max_abs": float(np.max(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
                          "greedy_margin": float(torch.diff(oracle.float().topk(2).values.flip(-1)).item())})
        actual = view("frame_codes", np.int32).copy()
        codes.append(actual.tolist())
        reference_codes.append(expected)
        metrics.extend(stats)
        traces.append(checkpoints)
    assert codes[0] == codes[2]
    saved = view("frame_codes", np.int32).copy()
    code_buffer[0] = settings["vocabulary"]
    assert run(arena, len(arena)) != 0
    np.testing.assert_array_equal(view("frame_codes", np.int32), saved)
    code_buffer[0] = 2
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("frame_codes", np.int32), saved)
    return {"codes": codes[:2], "reference_codes": reference_codes[:2], "layers": traces[:2], "first_layer": first_layer[:2], "scores": metrics,
            "max_abs": max(v["max_abs"] for v in metrics), "worst_rmse": max(v["rmse"] for v in metrics)}


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
    report = run_frame(tmp_path, torch, weights, tables, settings)
    assert report["codes"] == report["reference_codes"], report
    assert report["max_abs"] <= 0.015625, report


def test_pinned_codebook_frame(tmp_path, record_property):
    root = os.environ.get("CK_DRAMA3_TOWER_RANGES")
    if not root or not os.environ.get("CK_BLAS_LIBRARY"):
        pytest.skip("configure CK_DRAMA3_TOWER_RANGES and CK_BLAS_LIBRARY for the pinned tower")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(4)
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
    report = run_frame(tmp_path, torch, weights, tables, settings)
    record_property("reports", json.dumps(report))
    assert report["codes"] == report["reference_codes"], report
    assert report["max_abs"] <= 0.5 and report["worst_rmse"] <= 0.05, report
