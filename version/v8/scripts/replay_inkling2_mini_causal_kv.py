#!/usr/bin/env python3
"""Replay layer-6 K/V projection and causal convolution prefill and append."""

import argparse
import ctypes
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from version.v8.scripts.build_inkling2_mini_projection_circuit import (  # noqa: E402
    SOURCE_REVISION, TRANSFORMERS_REVISION, LAYER, causal_kv_replay, relative_attention_replay, sliding_attention_replay,
)
from version.v8.scripts.checked_circuit_authoring_v8 import declared_f32_weight_shapes  # noqa: E402
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph  # noqa: E402
from version.v8.scripts.replay_inkling2_mini_projections import HEADER_HASH, digest, read_bf16  # noqa: E402


def run(bundle, phase, slot):
    _, arena, _, lib = bundle
    fn = getattr(lib, f"ck_inkling2_{phase}_slot{slot}")
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    fn.restype = ctypes.c_int
    if fn(arena, len(arena)):
        raise RuntimeError(f"generated {phase} rejected slot {slot}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--attention", action="store_true",
                       help="connect scoped relative-bias and sink attention to the stored K/V")
    scope.add_argument("--sliding-output", action="store_true",
                       help="run the complete sliding-attention projection and gate path")
    args = parser.parse_args()
    header = args.header.read_bytes()
    if digest(header) != HEADER_HASH:
        parser.error("header does not match the pinned shard")
    args.output.mkdir(mode=0o700)

    import torch
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    report = {
        "status": "RUNNING", "source_revision": SOURCE_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION, "header_sha256": HEADER_HASH,
        "scope": "Layer 6 K/V projections and static residual convolution and normalized KV storage, isolated FP32 with supplied synthetic rows",
        "oracle": f"PyTorch {torch.__version__} eager FP32 linear, grouped conv1d and per-head norms",
        "slots": [6, 24, 36], "prefill_tokens": 8, "append_tokens": 1, "seed": 491,
        "weights": {}, "comparisons": {},
    }
    builder = relative_attention_replay if args.attention else causal_kv_replay
    if args.sliding_output:
        builder = sliding_attention_replay
        report["scope"] += "; generated clipped query, relative bias, sinks, gate and output projection"
    elif args.attention:
        report["scope"] += "; connected attention with supplied Q and sink plus a band from the publisher's bias table"
    target = args.output / "report.json"
    rng = np.random.default_rng(report["seed"])
    try:
        split_graph = builder()
        weights = {}
        for name, shape in declared_f32_weight_shapes(split_graph).items():
            source_shape = [shape[0], 1, shape[1]] if name.endswith("_sconv.weight") else shape
            if args.sliding_output:
                if name.endswith("q_l2_clip.scale"):
                    source_shape = [24, 128]
                elif name.endswith("rel_l2_clip.scale"):
                    source_shape = [24, 16]
                elif name.endswith("rel_logits_proj.proj"):
                    source_shape = [16, 1024]
            weight, report["weights"][name] = read_bf16(header, name, source_shape)
            if args.sliding_output and name.endswith("rel_logits_proj.proj"):
                weight = weight.T
            weights[name] = weight.reshape(shape)
        bundles, identities = {}, {}
        if args.attention:
            name = LAYER + "attn.rel_logits_proj.proj"
            bias_table, report["weights"][name] = read_bf16(header, name, [16, 1024])
        query_inputs = {}
        for key, graph in (("split", split_graph), ("full", builder(prefill_tokens=9))):
            path = args.output / key
            path.mkdir()
            bundles[key] = compile_declared_f32_graph(path, graph, weights)
            identity = json.loads((path / "build-identity.json").read_text())
            identities[key] = {
                "build_identity_sha256": identity["build_identity_sha256"],
                "generated_library_sha256": digest((path / "generated.so").read_bytes()),
            }
        report["builds"] = identities
        split_view, full_view = bundles["split"][2], bundles["full"][2]
        prefill, oracle, oracle_cache = {}, {}, {}

        for slot in report["slots"]:
            for kind in ("k", "v"):
                rows = rng.standard_normal((9, 2304), dtype=np.float32) * np.float32(0.2)
                initial = np.zeros((3, 1024), dtype=np.float32)
                split_view(f"history_{kind}_{slot}")[:] = initial
                full_view(f"history_{kind}_{slot}")[:] = initial
                split_view(f"cache_{kind}_{slot}").fill(-777)
                full_view(f"cache_{kind}_{slot}").fill(-777)
                split_view(f"inkling2_prefill_slot{slot}_{kind}_input")[:] = rows[:8]
                split_view(f"inkling2_append_slot{slot}_{kind}_input")[:] = rows[8:]
                full_view(f"inkling2_prefill_slot{slot}_{kind}_input")[:] = rows
                projection = torch.from_numpy(weights[LAYER + f"attn.w{kind}_dv.weight"])
                filters = torch.from_numpy(weights[LAYER + f"attn.{kind}_sconv.weight"]).unsqueeze(1)
                raw = torch.nn.functional.linear(torch.from_numpy(rows), projection)
                expanded = torch.cat((torch.from_numpy(initial), raw)).T.unsqueeze(0)
                convolved = torch.nn.functional.conv1d(expanded, filters, groups=1024)[0].T
                output = (convolved + raw) / np.sqrt(5)
                oracle[slot, kind] = output.numpy()
                heads = output.reshape(9, 8, 128)
                if kind == "k":
                    scale = (22.627416997969522 / (torch.linalg.vector_norm(heads, dim=-1, keepdim=True) + 1e-6)).clamp(max=1)
                else:
                    scale = torch.rsqrt(heads.square().mean(dim=-1, keepdim=True) + 1e-6)
                oracle_cache[slot, kind] = (heads * scale).transpose(0, 1).numpy()
            if args.sliding_output:
                rows = rng.standard_normal((9, 2304), dtype=np.float32) * np.float32(0.2)
                split_view(f"inkling2_prefill_slot{slot}_query_input")[:] = rows[:8]
                split_view(f"inkling2_append_slot{slot}_query_input")[:] = rows[8:]
                full_view(f"inkling2_prefill_slot{slot}_query_input")[:] = rows
                hidden = torch.from_numpy(rows)

                def project(name):
                    return torch.nn.functional.linear(hidden, torch.from_numpy(weights[LAYER + "attn." + name]))

                def clip(values, name, dim, threshold):
                    gamma = torch.from_numpy(weights[LAYER + "attn." + name]).reshape(24, dim)
                    scaled = values.reshape(9, 24, dim) * gamma
                    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
                    return scaled * (threshold / (norm + 1e-6)).clamp(max=1)

                query = clip(project("wq_du.weight"), "q_l2_clip.scale", 128, 22.627416997969522)
                relative_state = clip(project("wr_du.weight"), "rel_l2_clip.scale", 16, 8.0)
                band = relative_state @ torch.from_numpy(weights[LAYER + "attn.rel_logits_proj.proj"]).T
                sinks = project("attn_sink_proj.weight")
                gate = project("gate_attn.weight").sigmoid()
                query_inputs[slot] = (query.numpy(), band, sinks.numpy(), gate)
            elif args.attention:
                query = rng.standard_normal((9, 24, 128), dtype=np.float32) * 0.2
                relative_state = rng.standard_normal((9, 24, 16), dtype=np.float32) * 0.2
                band = torch.from_numpy(relative_state) @ torch.from_numpy(bias_table)
                sinks = rng.standard_normal((9, 24), dtype=np.float32)
                query_inputs[slot] = (query, band, sinks)
                for name, data in (("query", query), ("relative_band", band.numpy()), ("sink", sinks)):
                    split_view(f"inkling2_prefill_slot{slot}_{name}")[:] = data[:8]
                    split_view(f"inkling2_append_slot{slot}_{name}")[:] = data[8:]
                    full_view(f"inkling2_prefill_slot{slot}_{name}")[:] = data
            run(bundles["full"], "prefill", slot)
            run(bundles["split"], "prefill", slot)
            for kind in ("k", "v"):
                prefill[slot, kind] = split_view(f"history_{kind}_{slot}").copy()
                if not np.all(split_view(f"cache_{kind}_{slot}")[:, 8] == -777):
                    raise ValueError("prefill touched the uninitialized cache tail")

        # Feed the wrong slot's key history to slot 24 and require a numerical failure.
        split_view("history_k_24")[:] = prefill[6, "k"]
        run(bundles["split"], "append", 24)
        wrong = split_view("inkling2_append_slot24_k_output")
        correct = full_view("inkling2_prefill_slot24_k_output")[-1:]
        error = float(abs(wrong - correct).max())
        if error <= 1e-3:
            raise ValueError("wrong-slot control did not change the continuation")
        report["wrong_slot_max_abs"] = error
        for kind in ("k", "v"):
            split_view(f"history_{kind}_24")[:] = prefill[24, kind]

        for slot in (36, 6, 24):
            run(bundles["split"], "append", slot)
            for kind in ("k", "v"):
                first = split_view(f"inkling2_prefill_slot{slot}_{kind}_output")
                tail = split_view(f"inkling2_append_slot{slot}_{kind}_output")
                full = full_view(f"inkling2_prefill_slot{slot}_{kind}_output")
                output = np.concatenate((first, tail))
                history = split_view(f"history_{kind}_{slot}")
                if (output.tobytes() != full.tobytes() or
                        history.tobytes() != full_view(f"history_{kind}_{slot}").tobytes()):
                    raise ValueError(f"split execution changed output or history in slot {slot} {kind}")
                np.testing.assert_allclose(output, oracle[slot, kind], rtol=1e-5, atol=3e-6)
                cache = split_view(f"cache_{kind}_{slot}")
                full_cache = full_view(f"cache_{kind}_{slot}")[:, :9]
                if cache.tobytes() != full_cache.tobytes():
                    raise ValueError(f"split execution changed the stored cache in slot {slot} {kind}")
                np.testing.assert_allclose(cache, oracle_cache[slot, kind], rtol=1e-5, atol=5e-6)
                report["comparisons"][f"{slot}_{kind}"] = {
                    "split_exact": True, "history_exact": True, "stored_kv_exact": True,
                    "oracle_cache_max_abs": float(abs(cache - oracle_cache[slot, kind]).max()),
                    "oracle_max_abs": float(abs(output - oracle[slot, kind]).max()),
                }
        if args.attention or args.sliding_output:
            report["attention"] = {}
            positions = torch.arange(9)
            distance = positions[:, None] - positions[None, :]
            for slot in report["slots"]:
                before = split_view(f"inkling2_prefill_slot{slot}_attention")
                after = split_view(f"inkling2_append_slot{slot}_attention")
                full = full_view(f"inkling2_prefill_slot{slot}_attention")
                split_output = np.concatenate((before, after))
                if split_output.tobytes() != full.tobytes():
                    raise ValueError(f"split attention changed output in slot {slot}")
                query, band, sinks = query_inputs[slot][:3]
                key = torch.from_numpy(oracle_cache[slot, "k"]).repeat_interleave(3, dim=0)
                value = torch.from_numpy(oracle_cache[slot, "v"]).repeat_interleave(3, dim=0)
                score = torch.einsum("thd,hsd->ths", torch.from_numpy(query), key) / np.sqrt(128)
                bias = band.gather(-1, distance.clamp(min=0)[:, None, :].expand(-1, 24, -1))
                score = (score + bias).masked_fill(distance[:, None, :] < 0, -torch.inf)
                probs = torch.cat((score, torch.from_numpy(sinks).unsqueeze(-1)), dim=-1)
                probs = torch.softmax(probs, dim=-1)[..., :-1]
                expected = torch.einsum("ths,hsd->thd", probs, value).numpy()
                np.testing.assert_allclose(split_output, expected, rtol=2e-5, atol=8e-6)
                report["attention"][slot] = {
                    "split_exact": True,
                    "oracle_max_abs": float(abs(split_output - expected).max()),
                }
                if args.sliding_output:
                    name = "projected_attention"
                    projected = np.concatenate((split_view(f"inkling2_prefill_slot{slot}_{name}"),
                                                split_view(f"inkling2_append_slot{slot}_{name}")))
                    np.testing.assert_array_equal(projected, full_view(f"inkling2_prefill_slot{slot}_{name}"))
                    weight = torch.from_numpy(weights[LAYER + "attn.wo_ud.weight"])
                    expected_gated = torch.from_numpy(expected.reshape(9, 3072)) * query_inputs[slot][3]
                    expected_projected = torch.nn.functional.linear(expected_gated, weight).numpy()
                    np.testing.assert_allclose(projected, expected_projected, rtol=4e-5, atol=1e-5)
                    report["attention"][slot].update(
                        projected_split_exact=True,
                        projected_oracle_max_abs=float(abs(projected - expected_projected).max()),
                    )
        if args.sliding_output:
            report["status"] = "PASS_SLIDING_ATTENTION_OUTPUT_COMPONENT"
        else:
            report["status"] = "PASS_CACHED_ATTENTION_COMPONENT" if args.attention else "PASS_CAUSAL_KV_COMPONENT"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
