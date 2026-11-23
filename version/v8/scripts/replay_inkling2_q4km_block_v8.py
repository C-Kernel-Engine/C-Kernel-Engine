#!/usr/bin/env python3
"""Compare two real-weight mixed Inkling blocks against pinned llama.cpp captures."""

import argparse
import ctypes
import hashlib
import json
import mmap
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from version.v8.scripts.build_inkling2_decoder_block_v8 import (
    bind_mixed_decoder_weights, sliding_decoder_block,
)
from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph
from version.v8.scripts.replay_inkling2_routed_moe import read_gguf_header
from version.v8.scripts.replay_inkling2_text_loop_v8 import populate_arena, weight_entries


GGUF_SHA256 = "dac3702abcb5684cbcb1fbd148d94b2d531283be974630ac61964d194db483de"
CONFIG_SHA256 = "a7a349601f0984c4abf999c480ef775d8567d21f5f84b4e8a3e035b44f288dc2"
WIDTH = 2304
ROWS = 5


def read_rows(path, rows):
    data = np.fromfile(path, "<f4", count=rows * WIDTH + 1)
    if data.size != rows * WIDTH or not np.isfinite(data).all():
        raise ValueError(f"invalid oracle tensor: {path.name}")
    return data.reshape(rows, WIDTH)


def run_block(folder, stream, tensors, base, config, layer, input_rows, phases):
    folder.mkdir()
    built_rows = phases[0][1]
    graph = bind_mixed_decoder_weights(sliding_decoder_block(
        prefill_tokens=built_rows, layer=layer, state_slot=layer, depth_states=int(layer != 0),
    ), {name: t.ggml_type for name, t in tensors.items()})
    entries = weight_entries(graph)
    path = folder / "circuit.json"
    path.write_text(json.dumps(graph))
    source = {"template": graph, "entries": entries, "quant_summary": {}, "config": {
        "model": graph["name"], "arch": graph["name"], "num_layers": 1,
        "embed_dim": WIDTH, "num_heads": 24, "num_kv_heads": 8, "head_dim": 128,
        "intermediate_size": 1152, "context_length": built_rows+1, "max_seq_len": built_rows+1,
        "vocab_size": 1,
    }}
    layout, _, _, lib, _ = compile_native_graph(
        folder, source, path, extra_exports=("ck_set_num_threads",),
    )
    lib.ck_set_num_threads.argtypes = [ctypes.c_int]
    lib.ck_set_num_threads.restype = None
    lib.ck_set_num_threads(4)
    offsets = {item["name"]: item["abs_offset"] for item in layout["memory"]["activations"]["buffers"]}
    size = layout["memory"]["arena"]["total_size"]
    outputs = []
    with mmap.mmap(-1, size) as arena:
        addr = ctypes.addressof(ctypes.c_char.from_buffer(arena))

        def view(name, dtype="<f4"):
            return np.ndarray(tuple(graph["activation_buffers"][name]["shape"]), dtype, buffer=arena,
                              offset=offsets[name])

        populate_arena(stream, arena, layout["memory"]["weights"]["entries"], entries, tensors, base, config)
        bias = tensors[f"blk.{layer}.exp_probs_b.bias"]
        if bias.dims != (128,) or bias.ggml_type != 0:
            raise ValueError("invalid Q4_K_M router bias")
        stream.seek(base + bias.offset)
        view("router_bias")[:] = np.frombuffer(stream.read(128 * 4), "<f4")
        view("expert_slots", "<i4")[:] = np.arange(128, dtype="<i4")
        offset = 0
        for phase, rows, oracle in phases:
            prefix = f"inkling2_{phase}_slot{layer}"
            hidden = read_rows(Path(str(oracle) + ".l_out-5"), rows) if layer else input_rows[offset:offset+rows]
            view(prefix + "_hidden")[:] = hidden
            if layer:
                view(prefix + "_depth_bank")[:, 0] = input_rows[offset:offset+rows]
            run = getattr(lib, "ck_" + prefix)
            run.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            run.restype = ctypes.c_int
            rc = run(addr, size)
            if rc:
                raise RuntimeError(f"generated {prefix} failed: {rc}")
            outputs.append(view(prefix + "_layer_output").copy())
            offset += rows
        state = {name: hashlib.sha256(view(name)[:, :ROWS].tobytes() if name.startswith("cache") else
                                     view(name).tobytes()).hexdigest()
                 for name in (f"cache_k_{layer}", f"cache_v_{layer}", f"history_k_{layer}",
                              f"history_v_{layer}", "history_attn_post_conv", "history_ffn_post_conv")}
    return np.concatenate(outputs), state, {
        "weight_storage": {e["name"]: e["dtype"] for e in entries},
        "build_identity": json.loads((folder / "build-identity.json").read_text())["build_identity_sha256"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("gguf", "config", "input", "oracle-dir", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir()
    report = {"scope": "Q4_K_M stored weights, FP32 activations, ordered FP64 projection dots",
              "status": "INCOMPLETE", "gguf_sha256": GGUF_SHA256, "layers": {},
              "schedule_comparison": {}}
    try:
        config_bytes = args.config.read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != CONFIG_SHA256:
            raise ValueError("unexpected Inkling config identity")
        config = json.loads(config_bytes)["text_config"]
        inputs = read_rows(args.input, ROWS)
        report["input_sha256"] = hashlib.sha256(inputs.tobytes()).hexdigest()
        with args.gguf.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != GGUF_SHA256:
                raise ValueError("unexpected Q4_K_M artifact identity")
            stream.seek(0)
            tensors, base = read_gguf_header(stream.read(16 << 20))
            for layer in (0, 6):
                reference = args.oracle_dir / f"layer{layer}"
                if not np.array_equal(read_rows(Path(str(reference) + ".input.f32"), ROWS), inputs):
                    raise ValueError("oracle used different synthetic embeddings")
                full_path = Path(str(reference) + ".full.f32")
                prefill_path = Path(str(reference) + ".prefill.f32")
                append_path = Path(str(reference) + ".append.f32")
                summary = report["layers"][str(layer)] = {}
                scheduled_outputs = {}
                for schedule, phases in (
                    ("full", [("prefill", ROWS, full_path)]),
                    ("split", [("prefill", ROWS - 1, prefill_path), ("append", 1, append_path)]),
                ):
                    values, states, identity = run_block(
                        args.output / f"layer{layer}-{schedule}", stream, tensors, base, config, layer, inputs, phases,
                    )
                    expected = np.concatenate([read_rows(path, rows) for _, rows, path in phases])
                    if not np.isfinite(values).all():
                        raise ValueError("nonfinite generated block output")
                    delta = values.astype("f8") - expected
                    summary[schedule] = {
                        **identity, "state_sha256": states,
                        "oracle_output_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                        "oracle_bitwise_equal": values.tobytes() == expected.tobytes(),
                        "oracle_rmse": float(np.sqrt(np.mean(delta**2))),
                        "oracle_max_abs": float(np.max(np.abs(delta))),
                    }
                    scheduled_outputs[schedule] = values
                    np.save(args.output / f"layer{layer}-{schedule}.npy", values)
                # Layer 6 inherits separate upstream oracle captures for each schedule.
                report["schedule_comparison"][str(layer)] = {
                    "upstream_inputs": "same" if layer == 0 else "captured_per_schedule",
                    "output_bitwise_equal": (
                        scheduled_outputs["full"].tobytes() == scheduled_outputs["split"].tobytes()
                    ),
                    "differing_state_regions": [
                        name for name, digest in summary["full"]["state_sha256"].items()
                        if summary["split"]["state_sha256"][name] != digest
                    ],
                }
        report["status"] = "PASS_EXECUTION"
        report["oracle_status"] = "PASS_BITWISE" if all(
            case["oracle_bitwise_equal"] for layer in report["layers"].values() for case in layer.values()
        ) else "DIFFERENT"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
