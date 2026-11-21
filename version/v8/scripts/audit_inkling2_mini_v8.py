#!/usr/bin/env python3
"""Audit Inkling 2 Mini's source tensor layout and fixed text execution schedule."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from urllib.request import Request, urlopen

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
_FILENAMES = ("config.json", "model.safetensors.index.json", "safetensors-header.json", "hf_quant_config.json")
_MAX_METADATA = 2 * 1024 * 1024
_LAYER = "model.llm.layers."
_ITEM_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}


def text_schedule(config):
    """Physical weights are shared. Every occurrence owns its sequence state."""
    loop = config["loop_config"]
    depth, block = config["num_hidden_layers"], config["n_layers_per_attn_res_block"]
    start, end, passes = (loop[k] for k in ("core_start", "core_end", "num_passes"))
    if (any(type(x) is not int for x in (depth, block, start, end, passes, loop["gate_rank"]))
            or not 0 < start < end < depth <= 256
            or not 1 <= block <= depth or start % block or end % block
            or not 2 <= passes <= 8 or loop["gate_rank"] <= 0
            or loop["is_varloop"] is not False):
        raise ValueError("unsupported fixed loop geometry")
    local = config["local_layer_ids"]
    if (not isinstance(local, list) or any(type(x) is not int or not 0 <= x < depth for x in local)
            or len(local) != len(set(local))):
        raise ValueError("invalid sliding-layer list")

    def layer(weight, slot, phase, loop_pass=None):
        return dict(weight_layer=weight, state_slot=slot, phase=phase, loop_pass=loop_pass,
                    attention="sliding" if weight in local else "full",
                    takes_prior_pass_message=loop_pass is not None and loop_pass > 0)

    schedule = [layer(i, i, "prelude") for i in range(start)]
    width = end - start
    for pass_idx in range(passes):
        for i in range(start, end):
            slot = i if pass_idx == 0 else depth + (pass_idx - 1) * width + i - start
            schedule.append(layer(i, slot, "core", pass_idx))
    schedule.extend(layer(i, i, "epilogue") for i in range(end, depth))
    return schedule


def _storage_bytes(header, index):
    if set(header) != set(index["weight_map"]):
        raise ValueError("tensor index and shard header disagree")
    if len(set(index["weight_map"].values())) != 1:
        raise ValueError("the audit expects one main shard")
    spans = []
    for name, descriptor in header.items():
        shape = descriptor["shape"]
        start, end = descriptor["data_offsets"]
        dtype = descriptor["dtype"]
        if (dtype not in _ITEM_BYTES or any(type(n) is not int or n <= 0 for n in shape)
                or type(start) is not int or type(end) is not int or start < 0
                or end - start != math.prod(shape) * _ITEM_BYTES[dtype]):
            raise ValueError(f"invalid tensor storage extent: {name}")
        spans.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(spans):
        if start != cursor:
            raise ValueError(f"noncontiguous or overlapping tensor storage: {name}")
        cursor = end
    if cursor != index["metadata"]["total_size"]:
        raise ValueError("payload size disagrees with tensor index")
    return cursor


def audit(config, index, header, quant):
    if config.get("model_type") != "inkling2_loop" or quant.get("quant_method") != "inkling2":
        raise ValueError("expected inkling2_loop checkpoint and inkling2 quantization metadata")
    header = {name: meta for name, meta in header.items() if name != "__metadata__"}
    payload_bytes = _storage_bytes(header, index)
    text = config["text_config"]
    schedule = text_schedule(text)
    depth, hidden, inner, experts = (
        text[key] for key in ("num_hidden_layers", "hidden_size", "intermediate_size", "n_routed_experts")
    )

    def tensor(name, dtype, shape):
        meta = header.get(name, {})
        if meta.get("dtype") != dtype or meta.get("shape") != shape:
            raise ValueError(f"unexpected dtype or shape: {name}")

    for name in ("embed", "unembed"):
        tensor(f"model.llm.{name}.weight", "BF16", [text["vocab_size"], hidden])
    start, end, rank = (text["loop_config"][key] for key in ("core_start", "core_end", "gate_rank"))
    for name in ("middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"):
        tensor(f"model.llm.{name}", "F32", [2 * (end - start), hidden])

    expected_weights = {
        f"{_LAYER}{i}.mlp.experts.{kind}_weight": (i, dims)
        for i in range(depth)
        for kind, dims in (("w13", [experts, 2 * inner, hidden]), ("w2", [experts, hidden, inner]))
    }
    matched = set()
    expert_layers = {}
    for group in quant["model"]["parameter_groups"]:
        mode = group["compute_dtype"]
        if mode not in ("nvfp4", "mxfp8"):
            continue
        is_fp4 = mode == "nvfp4"
        block = 16 if is_fp4 else 32
        expected_quant = (
            {"amax_type": "per_token", "block_size": 16, "format": "nvfp4_e2m1",
             "rounding": "nearest", "scale_dtype": "float8_e4m3fn",
             "scale_selection_rule": "fouroversix_mae"}
            if is_fp4 else
            {"block_size": 32, "format": "mxfp8_e4m3", "rounding": "nearest", "scale_dtype": "ue8m0"}
        )
        if (group["weight_block_shape"] != ([1, 16] if is_fp4 else [32, 32])
                or group["weight_transform"] != f"{mode}_nearest"
                or group["required_activation_quantization"] != expected_quant):
            raise ValueError(f"unsupported {mode} quantization contract")
        for name in group["tensors"]:
            if name not in expected_weights or name in matched:
                raise ValueError(f"unexpected or repeated quantized expert tensor: {name}")
            i, dims = expected_weights[name]
            if (expert_layers.setdefault(i, mode) != mode or dims[-1] % block
                    or (not is_fp4 and dims[-2] % block)):
                raise ValueError(f"inconsistent expert quantization: {name}")
            matched.add(name)
            tensor(name, "U8" if is_fp4 else "F8_E4M3", [*dims[:-1], dims[-1] // 2] if is_fp4 else dims)
            tensor(name + "_scale", "F8_E4M3" if is_fp4 else "U8", [*dims[:-1], dims[-1] // block])
            if is_fp4:
                tensor(name + "_amax", "F32", [experts])
                tensor(name + "_outer_scale_denominator", "F32", [1])
    if matched != set(expected_weights):
        raise ValueError("missing one or more declared expert matrices")

    for i in range(depth):
        weight_prefix = f"{_LAYER}{i}.controllers."
        if i not in range(start, end):
            if any(name.startswith(weight_prefix) for name in header):
                raise ValueError("controller storage outside the repeated core")
            continue
        for controller in ("attn", "k", "v", "mlp"):
            for name, shape in (("down", [rank, 2 * hidden]), ("gain", [hidden, rank]),
                                ("forget", [hidden, rank])):
                tensor(f"{weight_prefix}{controller}.{name}_weight", "BF16", shape)

    return {
        "schema": "cke.inkling2_mini.import_audit.v1",
        "status": "METADATA_VALIDATED",
        "source_payload_bytes": payload_bytes,
        "tensor_count": len(header),
        "storage_dtypes": dict(sorted(Counter(x["dtype"] for x in header.values()).items())),
        "physical_text_layers": depth,
        "text_state_slots": len(schedule),
        "physical_expert_layers_by_format": {
            mode: [i for i in range(depth) if expert_layers[i] == mode] for mode in ("nvfp4", "mxfp8")
        },
        "executed_expert_layers_by_format": dict(sorted(Counter(
            expert_layers[step["weight_layer"]] for step in schedule).items())),
        "state_per_slot": ["kv_cache", "causal_convolution_history"],
        "text_schedule": schedule,
        "execution": "NOT_TESTED",
    }


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"repeated JSON key: {key}")
        obj[key] = value
    return obj


def fetch_metadata(revision):
    """Fetch publisher metadata and only the header bytes of the main shard."""
    base = f"https://huggingface.co/{MODEL_ID}/resolve/{revision}/"
    assets = {}
    for name in ("config.json", "model.safetensors.index.json", "hf_quant_config.json"):
        with urlopen(base + name, timeout=20) as response:
            content = response.read(_MAX_METADATA + 1)
        if len(content) > _MAX_METADATA:
            raise ValueError(f"metadata exceeds the download limit: {name}")
        assets[name] = content
    index = json.loads(assets["model.safetensors.index.json"], object_pairs_hook=_unique_object)
    shards = set(index["weight_map"].values())
    if len(shards) != 1:
        raise ValueError("expected exactly one publisher shard")
    shard = shards.pop()
    if not re.fullmatch(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", shard):
        raise ValueError("unexpected publisher shard filename")

    def read_range(first, last):
        request = Request(base + shard + "?download=true", headers={"Range": f"bytes={first}-{last}"})
        with urlopen(request, timeout=20) as response:
            match = re.fullmatch(rf"bytes {first}-{last}/([0-9]+)", response.headers.get("Content-Range") or "")
            if response.status != 206 or match is None:
                raise ValueError("publisher did not honor the requested safetensors byte range")
            length = last - first + 1
            content = response.read(length + 1)
            if len(content) != length:
                raise ValueError("incomplete safetensors range")
            return content, int(match[1])

    size, total = read_range(0, 7)
    header_bytes, = struct.unpack("<Q", size)
    if not 0 < header_bytes <= _MAX_METADATA or header_bytes + 8 + index["metadata"]["total_size"] != total:
        raise ValueError("invalid safetensors header size or shard total")
    content, observed_total = read_range(8, 7 + header_bytes)
    if observed_total != total:
        raise ValueError("safetensors shard size changed during fetch")
    assets["safetensors-header.json"] = content
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="directory with raw metadata and extracted safetensors-header.json")
    parser.add_argument("--revision", required=True, help="publisher revision used to fetch these inputs")
    parser.add_argument("--fetch", action="store_true", help="download metadata into a new directory")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full publisher commit hash")
    try:
        if args.fetch and args.snapshot.exists():
            raise ValueError("--fetch requires a new snapshot directory")
        assets = (fetch_metadata(args.revision) if args.fetch else
                  {name: (args.snapshot / name).read_bytes() for name in _FILENAMES})
        config, index, header, quant = (
            json.loads(assets[name], object_pairs_hook=_unique_object) for name in _FILENAMES
        )
        report = audit(config, index, header, quant)
        if args.fetch:
            args.snapshot.mkdir(mode=0o700)
            for name, content in assets.items():
                (args.snapshot / name).write_bytes(content)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    report["source_model"] = MODEL_ID
    report["source_revision"] = args.revision
    report["source_metadata_sha256"] = {name: hashlib.sha256(content).hexdigest()
                                         for name, content in assets.items()}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
