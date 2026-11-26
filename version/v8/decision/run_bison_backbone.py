"""Build and execute the pinned Bison packed decision graph through generated C."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import mmap
import shutil
from pathlib import Path
import struct
import time

import numpy as np

from version.v8.decision import groundhog_checkpoint
from version.v8.decision.build_bison_backbone import Backbone, Layer
from version.v8.decision.build_groundhog_moe import Experts
from version.v8.decision.capture_bison_block_oracle import PREFIX, checkpoint_hashes
from version.v8.decision.packed_question_layout import Question, pack_questions
from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bf16_bits(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def widen(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


class Checkpoint:
    def __init__(self, directory, expected_hashes):
        self.directory = directory
        self.expected_hashes = expected_hashes
        self.index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.shards = {}
        self.verified = {}

    def tensor(self, name, shape):
        shard = self.index[name]
        if shard not in self.shards:
            path = self.directory / shard
            self.verify(shard)
            with path.open("rb") as stream:
                length, = struct.unpack("<Q", stream.read(8))
                self.shards[shard] = (path, length + 8, json.loads(stream.read(length)))
        path, payload, tensors = self.shards[shard]
        entry = tensors[name]
        start, end = entry["data_offsets"]
        if (entry["dtype"] != "BF16" or entry["shape"] != list(shape) or
            start < 0 or end - start != math.prod(shape) * 2 or payload + end > path.stat().st_size):
            raise ValueError(f"unexpected publisher storage for {name}")
        return np.memmap(path, offset=payload + start, dtype="<u2", shape=tuple(shape), mode="r")

    def verify(self, name):
        if name not in self.verified:
            digest = file_sha256(self.directory / name)
            if digest != self.expected_hashes[name]:
                raise ValueError(f"publisher digest mismatch: {name}")
            self.verified[name] = digest


def rotary_frequencies(text, layer):
    params = text["rope_parameters"][layer.visibility + "_attention"]
    dim = layer.head_dim
    kind = params["rope_type"]
    if kind == "default":
        pairs = dim // 2
    elif kind == "proportional":
        pairs = int(params["partial_rotary_factor"] * dim // 2)
    else:
        raise ValueError(f"unsupported rotary policy {kind}")
    exponent = np.arange(0, 2 * pairs, 2, dtype=np.float32) / np.float32(dim)
    result = np.zeros(dim // 2, dtype=np.float32)
    result[:pairs] = 1.0 / np.power(np.float32(params["rope_theta"]), exponent)
    return result


def layer_geometry(text):
    if (text["num_kv_shared_layers"] or text["hidden_size_per_layer_input"]
        or text["attention_bias"] or text["hidden_activation"] != "gelu_pytorch_tanh"):
        raise ValueError("expected a biasless graph with independent layer state and tanh GELU")
    kinds = text["layer_types"]
    overrides = {int(index): record for index, record in text["per_layer_config"].items()}
    if len(kinds) != text["num_hidden_layers"]:
        raise ValueError("invalid layer schedule")
    layers = []
    for index, kind in enumerate(kinds):
        if kind not in {"sliding_attention", "full_attention"}:
            raise ValueError(f"unknown attention visibility for layer {index}")
        dimensions = overrides.get(index, {})
        layers.append(Layer(
            dimensions.get("num_key_value_heads", text["num_key_value_heads"]),
            dimensions.get("head_dim", text["head_dim"]),
            kind.removesuffix("_attention"),
            kind == "full_attention" and text["attention_k_eq_v"],
        ))
    return layers


def manifest(graph):
    entries = []
    offset = 4096
    for name, (shape, dtype) in graph.weights.items():
        offset = (offset + 63) & ~63
        size = math.prod(shape) * (2 if dtype == "BF16" else 4)
        entries.append({"name": name, "shape": shape, "dtype": dtype, "size": size, "file_offset": offset})
        offset += size
    return {"config": graph.config, "entries": entries, "quant_summary": {}, "template": graph.graph}


def run(checkpoint, request_path, output, *, blas=True, limit=None, keep_arena=False,
        oracle_path=None, max_abs=None, max_rmse=None, cblas=False, partition_topk=False):
    request = json.loads(request_path.read_text())
    revision = request.get("checkpoint")
    hashes = checkpoint_hashes(revision)
    request_sha = file_sha256(request_path)
    expected = None
    if oracle_path is not None:
        expected = json.loads(oracle_path.read_text())
        if (limit is not None or expected.get("checkpoint") != revision or
            expected.get("request_sha256") != request_sha or
            any(value is None or not math.isfinite(value) or value < 0 for value in (max_abs, max_rmse))):
            raise ValueError("oracle requires a matching full request and finite nonnegative error limits")
    weights = Checkpoint(checkpoint, hashes)
    weights.verify("config.json")
    weights.verify("model.safetensors.index.json")
    text = json.loads((checkpoint / "config.json").read_text())["text_config"]
    layers = layer_geometry(text)
    if text["enable_moe_block"] != (revision == groundhog_checkpoint.REVISION):
        raise ValueError("checkpoint expert configuration differs from its pinned policy")
    experts = None
    if text["enable_moe_block"]:
        experts = Experts(text["num_experts"], text["top_k_experts"], text["moe_intermediate_size"], partition_topk)
    elif partition_topk:
        raise ValueError("partition top-k requires routed experts")
    full = limit is None
    if limit is not None:
        if not 1 <= limit < len(layers):
            raise ValueError("partial replay must execute fewer than the full model's layers")
        layers = layers[:limit]
    questions = tuple(Question(q["name"], tuple(q["tokens"]), tuple(map(tuple, q["options"])))
                      for q in request["questions"])
    packed = pack_questions(tuple(request["prefix"]), questions, vocab_size=text["vocab_size"],
                            sliding_window=text["sliding_window"], capacity=512)
    tokens, inverse = np.unique(packed.token_ids, return_inverse=True)
    option_ids = sorted({token for _, _, options in packed.answers for _, token in options})
    option_column = {token: i for i, token in enumerate(option_ids)}
    width = text["hidden_size"]
    graph = Backbone(len(inverse), len(tokens), width, text["num_attention_heads"],
                     text["intermediate_size"], text["rms_norm_eps"], blas=blas, cblas=cblas)
    hidden = "input"
    frequencies = {}
    for index, layer in enumerate(layers):
        frequencies.setdefault(f"rope.{layer.visibility}.{layer.head_dim}", rotary_frequencies(text, layer))
        hidden = graph.layer(index, layer, hidden, moe=experts)
    if full:
        graph.readout(hidden, len(questions), len(option_ids))
    output.mkdir(parents=True)
    circuit = output / "circuit.json"
    circuit.write_text(json.dumps(graph.graph))
    layout, _, library, _, invoke = compile_native_graph(
        output, manifest(graph), circuit, target_isa="avx2",
    )
    entries = layout["memory"]["weights"]["entries"]
    memory = layout["memory"]["arena"]["total_size"]
    activations = {item["name"]: item["abs_offset"] for item in layout["memory"]["activations"]["buffers"]}
    if memory > shutil.disk_usage(output).free:
        raise ValueError(f"planned arena requires {memory} free bytes in {output}")
    arena_path = output / "arena.bin"
    file = arena_path.open("x+b")
    try:
        with file:
            file.truncate(memory)
            with mmap.mmap(file.fileno(), memory) as arena:
                for entry in entries:
                    name, offset = entry["name"], entry["abs_offset"]
                    shape, dtype = graph.weights[name]
                    target = np.ndarray(shape, "<u2" if dtype == "BF16" else "<f4", buffer=arena, offset=offset)
                    try:
                        if name.startswith("unit."):
                            target.fill(0x3F80)
                        elif name in frequencies:
                            target[:] = frequencies[name]
                        elif name == "option.weight":
                            target[:] = weights.tensor("lm_head.weight", [text["vocab_size"], width])[option_ids]
                        else:
                            stored = weights.tensor(name, shape)
                            target[:] = widen(stored) if dtype == "F32" else stored
                    finally:
                        del target

                def view(name, dtype):
                    shape = graph.graph["activation_buffers"][name]["shape"]
                    return np.ndarray(shape, dtype, buffer=arena, offset=activations[name])

                embedding = weights.tensor(PREFIX + "embed_tokens.weight", [text["vocab_size"], width])
                scale = widen(bf16_bits(math.sqrt(width)))
                view("embedding_table", np.uint16)[:] = bf16_bits(widen(embedding[tokens]) * scale)
                view("token_rows", np.int32)[:] = inverse
                view("positions", np.int32)[:] = packed.position_ids
                kinds = {layer.visibility for layer in layers}
                for kind in kinds:
                    name = kind + "_ranges"
                    view(name, np.int32)[:] = getattr(packed, name)
                if full:
                    view("answer_rows", np.int32)[:] = [row for _, row, _ in packed.answers]
                start = time.monotonic()
                native_arena = (ctypes.c_uint8 * memory).from_buffer(arena)
                try:
                    result = invoke(native_arena, memory)
                finally:
                    del native_arena
                seconds = time.monotonic() - start
                if result:
                    raise RuntimeError(f"generated backbone rejected the request with status {result}")
                observed = widen(view("scores" if full else hidden, np.uint16))
    finally:
        if not keep_arena:
            arena_path.unlink()
    report = {
        "checkpoint": revision, "verified_files": weights.verified, "request_sha256": request_sha,
        "layers": len(layers), "tokens": len(packed.token_ids), "library_sha256": file_sha256(library),
        "projection": "bf16_cblas" if cblas else "fp64_blas" if blas else "fp64_ordered_scalar",
        "routing": "partition_topk" if partition_topk else "default", "native_seconds": seconds,
    }
    if full:
        report["questions"] = [
            {"name": name, "raw_option_scores": {label: float(observed[i, option_column[token]])
                                                for label, token in options}}
            for i, (name, _, options) in enumerate(packed.answers)
        ]
    else:
        np.save(output / "hidden.npy", observed, allow_pickle=False)
    if expected is not None:
        if len(expected["questions"]) != len(questions):
            raise ValueError("oracle and generated request differ")
        differences = []
        winners_match = True
        for actual, reference in zip(report["questions"], expected["questions"], strict=True):
            scores, reference_scores = actual["raw_option_scores"], reference["raw_option_scores"]
            if actual["name"] != reference["name"] or scores.keys() != reference_scores.keys():
                raise ValueError("oracle question or option identities differ")
            if not all(isinstance(score, (int, float)) and not isinstance(score, bool)
                       and math.isfinite(score) for score in reference_scores.values()):
                raise ValueError("oracle scores must be finite numbers")
            differences.extend(scores[label] - reference_scores[label] for label in scores)
            winners_match &= max(scores, key=scores.get) == max(reference_scores, key=reference_scores.get)
        maximum = max(map(abs, differences))
        rmse = math.sqrt(sum(error * error for error in differences) / len(differences))
        report["comparison"] = {
            "pass": winners_match and maximum <= max_abs and rmse <= max_rmse,
            "limits": {"max_abs": max_abs, "rmse": max_rmse},
            "rmse": rmse,
            "oracle_sha256": file_sha256(oracle_path), "differences": differences,
            "max_abs": maximum, "winners_match": winners_match,
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, help="compare a prefix of the backbone without the readout")
    parser.add_argument("--oracle", type=Path, help="compare labeled raw option scores on a full run")
    parser.add_argument("--max-abs", type=float)
    parser.add_argument("--max-rmse", type=float)
    projection = parser.add_mutually_exclusive_group()
    projection.add_argument("--scalar", action="store_true", help="use the ordered scalar FP64 projection")
    projection.add_argument("--bf16-cblas", action="store_true", help="use native BF16 CBLAS for row projections")
    parser.add_argument("--partition-topk", action="store_true", help="use partition-based expert tie selection")
    parser.add_argument("--keep-arena", action="store_true")
    args = parser.parse_args()
    report = run(args.checkpoint, args.request, args.output, blas=not args.scalar,
                 limit=args.layers, keep_arena=args.keep_arena, oracle_path=args.oracle,
                 max_abs=args.max_abs, max_rmse=args.max_rmse, cblas=args.bf16_cblas,
                 partition_topk=args.partition_topk)
    print(json.dumps(report, indent=2))
    return 0 if report.get("comparison", {"pass": True})["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
