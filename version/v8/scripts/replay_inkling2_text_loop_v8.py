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


def capture_layer_rows(directory, graph, view, phase, position, row):
    """Write final causal rows in loop execution order for the X-Ray sweep."""
    directory.mkdir(mode=0o700, exist_ok=True)
    checkpoints = []
    exports = graph["semantic_checkpoints"]["exports"]
    schedule = graph["contract"]["runtime_invariants"]["layer_schedule"]
    for index, occurrence in enumerate(schedule):
        slot = occurrence["state_slot"]
        port = f"inkling2_{phase}_slot{slot}_layer_output"
        check, = exports[port]["checkpoints"]
        filename = f"tok_{position:06d}_layer_{index:03d}_layer_out.f32"
        values = np.asarray(view(check["tensor"])[row], dtype="<f4")
        values.tofile(directory / filename)
        checkpoints.append({**check, **occurrence, "sequence_index": index, "filename": filename})
    (directory / "index.json").write_text(json.dumps({
        "phase": phase, "position": position, "checkpoints": checkpoints,
    }, indent=2) + "\n")


def read_token_ids(path):
    with path.open("rb") as token_file:
        raw_ids = token_file.read((1 << 20) + 1)
    if len(raw_ids) > 1 << 20:
        raise ValueError("token ID file exceeds 1 MiB")
    return [int(token) for token in raw_ids.split()]


def compare_oracle_row(actual, expected, position, phase):
    """Compare one complete vocabulary row from each native runtime."""
    if actual.shape != expected.shape or not np.isfinite(expected).all():
        raise ValueError("oracle row has invalid geometry or nonfinite logits")
    mismatches = np.flatnonzero(actual.view("<u4") != expected.view("<u4"))
    delta = actual.astype(np.float64) - expected
    return {
        "position": position, "phase": phase, "elements": actual.size,
        "exact_elements": actual.size - mismatches.size,
        "first_different_index": int(mismatches[0]) if mismatches.size else None,
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "native_argmax": int(actual.argmax()), "oracle_argmax": int(expected.argmax()),
        "status": "PASS" if not mismatches.size else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokens-file", type=Path, required=True)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--append-token", type=int)
    continuation.add_argument("--append-tokens-file", type=Path,
                              help="whitespace-separated teacher-forced token IDs")
    parser.add_argument("--oracle-logits", type=Path,
                        help="little-endian FP32 final prompt row followed by one row per forced token")
    parser.add_argument("--f64-norm-sum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--greedy-tokens", type=int, default=0)
    parser.add_argument("--stop-id", type=int, action="append", default=[])
    parser.add_argument("--context-capacity", type=int)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument("--fp32-static-conv", action="store_true",
                        help="use ordered FP32 arithmetic for static short convolutions")
    parser.add_argument("--reference-moe", action="store_true",
                        help="use reciprocal-scaled SiTU and router-slot accumulation")
    parser.add_argument("--last-prefill-logit", action="store_true",
                        help="project only the final hidden row in each prefill chunk")
    parser.add_argument("--threads", type=int, default=1,
                        help="OpenMP worker count for independent Q8_0 rows")
    parser.add_argument("--xray-token-position", type=int,
                        help="capture layer outputs at this zero-based processed token")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("threads must be in 1..64")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)["text_config"]
    try:
        ids = read_token_ids(args.tokens_file)
        if args.append_tokens_file is not None:
            forced = read_token_ids(args.append_tokens_file)
            if not forced:
                parser.error("empty teacher-forced token file")
        else:
            forced = [] if args.append_token is None else [args.append_token]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    resident = args.context_capacity is not None
    capacity = args.context_capacity if resident else len(ids) + 1
    vocab = config["unpadded_vocab_size"]
    chunk_tokens = args.chunk_tokens if resident else len(ids)
    if (not 1 <= chunk_tokens <= 16 or not chunk_tokens < capacity <= 65536 or not ids
            or len(ids) + len(forced) > capacity
            or any(not 0 <= token < vocab for token in ids)):
        parser.error("invalid prompt IDs, chunk size or context capacity")
    if any(not 0 <= token < vocab for token in forced):
        parser.error("invalid continuation ID")
    if (args.greedy_tokens < 0 or (args.greedy_tokens and (not resident or forced))
            or len(ids) + args.greedy_tokens > capacity
            or any(not 0 <= token < config["unpadded_vocab_size"] for token in args.stop_id)):
        parser.error("greedy generation requires a resident capacity, valid stop IDs and room for output")
    if args.xray_token_position is not None:
        processed = len(ids) + len(forced) + max(0, args.greedy_tokens - 1)
        if not 0 <= args.xray_token_position < processed:
            parser.error("X-Ray token position is outside the replay")
    oracle = None
    if args.oracle_logits is not None:
        if args.greedy_tokens:
            parser.error("oracle comparison requires teacher-forced decode")
        expected_shape = (len(forced) + 1, vocab)
        try:
            if args.oracle_logits.stat().st_size != expected_shape[0] * expected_shape[1] * 4:
                parser.error("oracle file size differs from the required complete logit rows")
            oracle = np.memmap(args.oracle_logits, mode="r", dtype="<f4", shape=expected_shape)
        except OSError as error:
            parser.error(str(error))
    args.output.mkdir(mode=0o700)
    report = {"status": "INCOMPLETE", "gguf_revision": GGUF_REVISION, "prompt_ids": ids,
              "append_token": args.append_token, "scope": "bounded native Q8_0 token replay",
              "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "context_capacity": capacity, "chunk_tokens": chunk_tokens, "completed_prompt_tokens": 0,
              "threads": args.threads,
              "f64_norm_sum": args.f64_norm_sum, "last_prefill_logit": args.last_prefill_logit,
              "reference_moe": args.reference_moe, "fp32_static_conv": args.fp32_static_conv}
    if args.append_tokens_file is not None:
        report.update(append_ids=forced, completed_append_tokens=0)
    if oracle is not None:
        with args.oracle_logits.open("rb") as file:
            report["oracle_logits_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
        report["oracle_comparison"] = {"status": "INCOMPLETE", "rows": []}
    report_path = args.output / "report.json"

    def save():
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_path)

    def compare(logits, index, position, phase):
        if oracle is not None:
            report["oracle_comparison"]["rows"].append(compare_oracle_row(logits, oracle[index], position, phase))

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
            options = {"f64_norm_sum": args.f64_norm_sum, "last_prefill_logit": args.last_prefill_logit,
                       "reference_moe": args.reference_moe, "fp32_static_conv": args.fp32_static_conv}
            if resident:
                graph = resident_text_loop(config, chunk_tokens, capacity, **options)
            else:
                graph = text_token_loop(config, len(ids), **options)
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
            layout, _, _, library, _ = compile_native_graph(
                args.output, source, circuit, openmp=args.threads > 1)
            if args.threads > 1:
                library.omp_set_dynamic.argtypes = [ctypes.c_int]
                library.omp_set_dynamic.restype = None
                library.omp_set_dynamic(0)
                library.omp_set_num_threads.argtypes = [ctypes.c_int]
                library.omp_set_num_threads.restype = None
                library.omp_set_num_threads(args.threads)
            total = layout["memory"]["arena"]["total_size"]
            binding = load_exports(args.output, library)
            buffers = binding["ports"]
            activation_offsets = {
                entry["name"]: entry["abs_offset"]
                for entry in layout["memory"]["activations"]["buffers"]
            }
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

                    def view_checkpoint(name):
                        return np.ndarray(tuple(graph["activation_buffers"][name]["shape"]),
                                          dtype="<f4", buffer=arena, offset=activation_offsets[name])

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
                        capture_position = args.xray_token_position
                        if capture_position is not None and position <= capture_position < position + len(tokens):
                            capture_dir = args.output / "xray"
                            row = capture_position - position
                            capture_layer_rows(capture_dir, graph, view_checkpoint, phase, capture_position, row)
                            if logits.shape[0] == len(tokens):
                                np.asarray(logits[row], dtype="<f4").tofile(capture_dir / "logits.f32")
                            elif row == len(tokens) - 1:
                                np.asarray(logits[-1], dtype="<f4").tofile(capture_dir / "logits.f32")
                            report["xray_token_position"] = capture_position
                        return logits, elapsed

                    total_prefill, offset = 0.0, 0
                    last_report = time.monotonic()
                    while offset < len(ids):
                        use_chunk = len(ids) - offset >= chunk_tokens
                        piece = ids[offset:offset + (chunk_tokens if use_chunk else 1)]
                        logits, elapsed = run("prefill" if use_chunk else "append", piece, offset)
                        offset += len(piece)
                        total_prefill += elapsed
                        report.update(prefill_seconds=total_prefill, completed_prompt_tokens=offset)
                        now = time.monotonic()
                        if offset == len(ids) or now - last_report >= 2.0:
                            save()
                            last_report = now
                    np.save(args.output / "prefill-logits.npy", logits if not resident else logits[-1:])
                    report["prefill_argmax"] = logits.argmax(axis=1).tolist()
                    compare(logits[-1], 0, len(ids) - 1, "prefill")
                    if forced:
                        append_rows = np.lib.format.open_memmap(
                            args.output / "append-logits.npy", mode="w+", dtype="<f4", shape=(len(forced), vocab))
                        append_seconds, append_argmax = 0.0, []
                        for index, token in enumerate(forced):
                            position = len(ids) + index
                            logits, elapsed = run("append", [token], position)
                            append_rows[index] = logits[-1]
                            append_seconds += elapsed
                            append_argmax.append(int(logits[-1].argmax()))
                            compare(logits[-1], index + 1, position, "decode")
                            report.update(append_seconds=append_seconds, append_argmax=append_argmax,
                                          completed_append_tokens=index + 1)
                            save()
                        append_rows.flush()
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
        if args.xray_token_position is not None and "xray_token_position" not in report:
            raise ValueError("the requested X-Ray token was not processed")
        report["execution_status"] = "PASS"
        if oracle is None:
            report["status"] = "PASS"
        else:
            comparison = report["oracle_comparison"]
            comparison["status"] = ("PASS" if all(row["status"] == "PASS" for row in comparison["rows"])
                                    else "FAIL")
            report["status"] = comparison["status"]
    except Exception as error:
        report.update(status="FAIL", execution_status="FAIL", error=str(error))
        raise
    finally:
        save()
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
