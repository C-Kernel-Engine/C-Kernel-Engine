#!/usr/bin/env python3
"""Run the pinned Inkling Q8_0 text graph with real token IDs and a mapped arena."""

import argparse
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from version.v8.scripts.build_inkling2_cross_pass_v8 import CONTROL_PORTS
from version.v8.scripts.build_inkling2_decoder_block_v8 import first_block_weights
from version.v8.scripts.build_inkling2_text_loop_v8 import resident_text_loop, text_token_loop
from version.v8.scripts.checked_execution_exports_v8 import load_exports
from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph
from version.v8.scripts.replay_inkling2_routed_moe import (
    GGML_TYPE_F32, GGML_TYPE_Q8_0, GGUF_FILES, GGUF_REVISION, read_gguf_header,
)


GGUF_SHA256 = "d57542f8aea5d7639beba5fc4ae7f584754916f63ac5d5a69328c27f652f45da"
TRANSFER_BYTES = 8 * 1024 * 1024


def weight_entries(graph):
    entries, offset = [], 4096
    for name, (shape, dtype) in first_block_weights(graph).items():
        elements = int(np.prod(shape, dtype=object))
        size = elements * 4 if dtype == "F32" else elements // 32 * 34
        offset = (offset + 63) & ~63
        entries.append(dict(name=name, shape=shape, dtype=dtype, size=size, file_offset=offset))
        offset += size
    return entries


def gguf_copy_spans(tensors, data_start, entry, config):
    """Translate one compiled weight into exact GGUF payload spans."""
    name, shape, size = entry["name"], entry["shape"], entry["size"]
    dtype = GGML_TYPE_F32 if entry["dtype"] == "F32" else GGML_TYPE_Q8_0
    dims = tuple(reversed(shape))
    offset, stride, count, width = 0, size, 1, size
    if ".expert_bank." in name:
        layer, kind = name.split("expert_bank.")
        if kind == "down":
            name = layer + "ffn_down_exps.weight"
        else:
            name = layer + "ffn_gate_up_exps.weight"
            cols, rows, count = dims
            dims = (cols, rows * 2, count)
            width = size // count
            stride = width * 2
            offset = width if kind == "up" else 0
    elif name.endswith((".attn_q_clip.weight", ".attn_rel_clip.weight")):
        heads = config["num_attention_heads"]
        physical_layer = int(name.split(".")[1])
        if physical_layer in config["local_layer_ids"]:
            heads = config["swa_num_attention_heads"]
        dims = (shape[0] // heads, heads)
    elif not name.endswith(".weight") and any(key in name for key in (".loop_msg_query.", ".attn_res_query.")):
        prefix, port = name.rsplit(".", 1)
        order = CONTROL_PORTS if ".loop_msg_query." in name else CONTROL_PORTS[:3]
        name = prefix + ".weight"
        dims = (shape[0], len(order))
        offset = order.index(port) * size
    elif name == "output.weight":
        dims = (dims[0], config["vocab_size"])
    info = tensors[name]
    if info.dims != dims or info.ggml_type != dtype:
        raise ValueError(f"GGUF tensor differs from compiled contract: {name} {info.dims} {info.ggml_type}; expected {dims} {dtype}")
    for i in range(count):
        yield data_start + info.offset + offset + i * stride, i * width, width


def populate_arena(source, arena, layout_entries, manifest_entries, tensors, base, config):
    declaration = {entry["name"]: entry for entry in manifest_entries}
    for planned in layout_entries:
        entry = declaration[planned["name"]]
        if planned["size"] != entry["size"]:
            raise ValueError(f"arena footprint differs from the GGUF contract: {entry['name']}")
        destination = planned["abs_offset"]
        for src, displacement, count in gguf_copy_spans(tensors, base, entry, config):
            for first in range(0, count, TRANSFER_BYTES):
                amount = min(TRANSFER_BYTES, count - first)
                source.seek(src + first)
                data = source.read(amount)
                if len(data) != amount:
                    raise ValueError(f"truncated GGUF while copying {entry['name']}")
                start = destination + displacement + first
                arena[start:start + amount] = data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokens-file", type=Path, required=True)
    parser.add_argument("--append-token", type=int)
    parser.add_argument("--f64-norm-sum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--greedy-tokens", type=int, default=0)
    parser.add_argument("--stop-id", type=int, action="append", default=[])
    parser.add_argument("--context-capacity", type=int)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)["text_config"]
    with args.tokens_file.open("rb") as token_file:
        raw_ids = token_file.read((1 << 20) + 1)
    if len(raw_ids) > 1 << 20:
        parser.error("token ID file exceeds 1 MiB")
    ids = [int(token) for token in raw_ids.split()]
    resident = args.context_capacity is not None
    capacity = args.context_capacity if resident else len(ids) + 1
    chunk_tokens = args.chunk_tokens if resident else len(ids)
    if (not 1 <= chunk_tokens <= 16 or not chunk_tokens < capacity <= 65536 or not ids
            or len(ids) + (args.append_token is not None) > capacity
            or any(not 0 <= token < config["unpadded_vocab_size"] for token in ids)):
        parser.error("invalid prompt IDs, chunk size or context capacity")
    if args.append_token is not None and not 0 <= args.append_token < config["unpadded_vocab_size"]:
        parser.error("invalid continuation ID")
    if (args.greedy_tokens < 0 or (args.greedy_tokens and (not resident or args.append_token is not None))
            or len(ids) + args.greedy_tokens > capacity
            or any(not 0 <= token < config["unpadded_vocab_size"] for token in args.stop_id)):
        parser.error("greedy generation requires a resident capacity, valid stop IDs and room for output")
    args.output.mkdir(mode=0o700)
    report = {"status": "INCOMPLETE", "gguf_revision": GGUF_REVISION, "prompt_ids": ids,
              "append_token": args.append_token, "scope": "bounded native Q8_0 token replay",
              "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "context_capacity": capacity, "chunk_tokens": chunk_tokens, "completed_prompt_tokens": 0,
              "f64_norm_sum": args.f64_norm_sum}
    report_path = args.output / "report.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        with args.gguf.open("rb") as gguf:
            if os.fstat(gguf.fileno()).st_size != GGUF_FILES["q8_0"][1]:
                raise ValueError("GGUF file length differs from the pinned Q8_0 artifact")
            report["gguf_sha256"] = hashlib.file_digest(gguf, "sha256").hexdigest()
            if report["gguf_sha256"] != GGUF_SHA256:
                raise ValueError("GGUF bytes differ from the pinned Q8_0 artifact")
            gguf.seek(0)
            tensors, base = read_gguf_header(gguf.read(16 * 1024 * 1024))
            if resident:
                graph = resident_text_loop(config, chunk_tokens, capacity, f64_norm_sum=args.f64_norm_sum)
            else:
                graph = text_token_loop(config, len(ids), f64_norm_sum=args.f64_norm_sum)
            entries = weight_entries(graph)
            for item in entries:
                tuple(gguf_copy_spans(tensors, base, item, config))
            circuit = args.output / "circuit.json"
            circuit.write_text(json.dumps(graph))
            source = {"template": graph, "entries": entries, "quant_summary": {}, "config": {
                "model": graph["name"], "arch": graph["name"], "num_layers": 1,
                "embed_dim": config["hidden_size"], "num_heads": 1,
                "num_kv_heads": 1, "head_dim": config["hidden_size"],
                "intermediate_size": config["intermediate_size"], "context_length": capacity,
                "max_seq_len": capacity, "vocab_size": config["vocab_size"],
            }}
            begin = time.monotonic()
            layout, _, _, library, _ = compile_native_graph(args.output, source, circuit)
            total = layout["memory"]["arena"]["total_size"]
            binding = load_exports(args.output, library)
            buffers = binding["ports"]
            report["build_seconds"] = time.monotonic() - begin
            report["arena_bytes"] = total
            report["build_identity"] = json.loads((args.output / "build-identity.json").read_text())
            save()
            with (args.output / "arena.bin").open("w+b") as file:
                file.truncate(total)
                os.posix_fallocate(file.fileno(), 0, total)
                begin = time.monotonic()
                with mmap.mmap(file.fileno(), total, access=mmap.ACCESS_WRITE) as weights:
                    populate_arena(gguf, weights, layout["memory"]["weights"]["entries"],
                                   entries, tensors, base, config)
                    weights.flush()
                report["arena_pack_seconds"] = time.monotonic() - begin
                save()
                with mmap.mmap(file.fileno(), total, access=mmap.ACCESS_COPY) as arena:
                    pointer = ctypes.addressof(ctypes.c_char.from_buffer(arena))

                    def view(name):
                        port = buffers[name]
                        dtype = {"fp32": np.float32, "i32": np.int32}[port["dtype"]]
                        return np.ndarray(tuple(port["shape"]), dtype, buffer=arena, offset=port["offset"])

                    view("expert_slots")[:] = np.arange(config["n_routed_experts"], dtype=np.int32)
                    for layer in range(config["num_hidden_layers"]):
                        dtype = GGML_TYPE_F32
                        tensor = tensors[f"blk.{layer}.exp_probs_b.bias"]
                        if tensor.dims != (config["n_routed_experts"],) or tensor.ggml_type != dtype:
                            raise ValueError("unexpected router correction layout")
                        gguf.seek(base + tensor.offset)
                        view(f"router_bias_{layer}")[:] = np.frombuffer(
                            gguf.read(config["n_routed_experts"] * 4), "<f4")

                    def run(phase, tokens, position):
                        if resident:
                            view(f"{phase}_request_position")[:] = position
                        view(f"{phase}_token_ids")[:] = tokens
                        function = getattr(library, binding["entries"][phase])
                        function.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                        function.restype = ctypes.c_int
                        begin = time.monotonic()
                        status = function(pointer, total)
                        elapsed = time.monotonic() - begin
                        if status:
                            raise RuntimeError(f"generated {phase} returned {status}")
                        logits = view(f"{phase}_logits").copy()
                        if not np.isfinite(logits).all():
                            raise ValueError(f"nonfinite {phase} logits")
                        return logits, elapsed

                    total_prefill, offset = 0.0, 0
                    while offset < len(ids):
                        use_chunk = len(ids) - offset >= chunk_tokens
                        piece = ids[offset:offset + (chunk_tokens if use_chunk else 1)]
                        logits, elapsed = run("prefill" if use_chunk else "append", piece, offset)
                        offset += len(piece)
                        total_prefill += elapsed
                        report.update(prefill_seconds=total_prefill, completed_prompt_tokens=offset)
                        save()
                    np.save(args.output / "prefill-logits.npy", logits if not resident else logits[-1:])
                    report["prefill_argmax"] = logits.argmax(axis=1).tolist()
                    if args.append_token is not None:
                        logits, elapsed = run("append", [args.append_token], len(ids))
                        np.save(args.output / "append-logits.npy", logits)
                        report.update(append_seconds=elapsed, append_argmax=logits.argmax(axis=1).tolist())
                    if args.greedy_tokens:
                        generated, generation_seconds = [], 0.0
                        stop_ids = set(args.stop_id)
                        finish_reason = "length"
                        for step in range(args.greedy_tokens):
                            token = int(logits[-1].argmax())
                            generated.append(token)
                            if token in stop_ids:
                                finish_reason = "stop"
                                break
                            if step + 1 < args.greedy_tokens:
                                logits, elapsed = run("append", [token], len(ids) + step)
                                generation_seconds += elapsed
                            report.update(generated_ids=generated, generation_seconds=generation_seconds)
                            save()
                        report.update(generated_ids=generated, generation_seconds=generation_seconds,
                                      finish_reason=finish_reason)
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
