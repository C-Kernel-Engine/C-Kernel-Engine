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
    SOURCE_REVISION, TRANSFORMERS_REVISION, LAYER, causal_kv_replay,
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
    target = args.output / "report.json"
    rng = np.random.default_rng(report["seed"])
    try:
        split_graph = causal_kv_replay()
        weights = {}
        for name, shape in declared_f32_weight_shapes(split_graph).items():
            source_shape = [shape[0], 1, shape[1]] if name.endswith("_sconv.weight") else shape
            weight, report["weights"][name] = read_bf16(header, name, source_shape)
            weights[name] = weight.reshape(shape)
        bundles, identities = {}, {}
        for key, graph in (("split", split_graph), ("full", causal_kv_replay(prefill_tokens=9))):
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
        report["status"] = "PASS_CAUSAL_KV_COMPONENT"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
