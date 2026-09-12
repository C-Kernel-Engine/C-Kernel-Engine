#!/usr/bin/env python3
"""Convert a complete Cohere Transcribe F16/F32 GGUF into BUMPWGT5."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _load_converter():
    path = Path(__file__).with_name("convert_gguf_to_bump_v8.py")
    spec = importlib.util.spec_from_file_location("cke_gguf_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


c = _load_converter()


def parse_gguf(path: Path):
    handle = path.open("rb")
    reader = c.GGUFReader(handle)
    if reader._read_exact(4) != b"GGUF":
        raise c.GGUFError(f"{path}: invalid GGUF magic")
    version = reader.u32()
    if version < 2:
        raise c.GGUFError(f"{path}: GGUF v{version} is unsupported")
    tensor_count, metadata_count = reader.u64(), reader.u64()
    metadata = {}
    for _ in range(metadata_count):
        key = reader.key_str()
        metadata[key] = c._gguf_read_value(reader, reader.u32())
    tensors = {}
    for _ in range(tensor_count):
        name = reader.key_str()
        dimensions = tuple(int(reader.u64()) for _ in range(reader.u32()))
        ggml_type, offset = reader.u32(), reader.u64()
        tensors[name] = c.TensorInfo(name=name, dims=dimensions, ggml_type=ggml_type, offset=offset)
    data_start = c.align_up(reader.tell(), int(metadata.get("general.alignment", 32)))
    return handle, metadata, tensors, data_start


def positive(metadata: dict[str, object], key: str) -> int:
    value = int(metadata.get(key, 0))
    if value <= 0:
        raise c.GGUFError(f"missing positive metadata {key}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    source = args.gguf.resolve()
    output_dir = args.output_dir.resolve()
    handle, metadata, tensors, source_data_start = parse_gguf(source)
    try:
        if metadata.get("general.architecture") != "cohere-transcribe":
            raise c.GGUFError(f"{source}: expected cohere-transcribe architecture")
        inventory = c.gguf_ck_tensor_inventory_report(
            "cohere-transcribe", metadata, tensors.keys()
        )
        if not inventory or inventory.get("status") != "complete":
            raise c.GGUFError(f"incomplete Cohere tensor inventory: {inventory}")
        unsupported = sorted(
            info.name for info in tensors.values()
            if info.ggml_type not in {c.GGML_TYPE_F16, c.GGML_TYPE_F32}
        )
        if unsupported:
            raise c.GGUFError(
                "correctness bundle supports only F16/F32 source tensors; "
                f"unsupported={unsupported[:16]}"
            )
        config = {
            key: value for key, value in metadata.items()
            if key.startswith("cohere_transcribe.") or key.startswith("tokenizer.ggml.")
        }
        config.update({
            "model": "cohere_transcribe",
            "model_type": "cohere_transcribe",
            "source_architecture": "cohere-transcribe",
            "tensor_count": len(tensors),
        })
        entries = []
        cursor = c.DATA_START + 4 + len(tensors)
        for name, info in tensors.items():
            cursor = c.align_up(cursor, 32)
            size = c.ggml_tensor_bytes(info)
            entries.append({
                "name": name,
                "dtype": "fp16" if info.ggml_type == c.GGML_TYPE_F16 else "fp32",
                "file_offset": cursor,
                "size": size,
                "source_name": name,
                "source_dtype": c.ggml_type_name(info.ggml_type),
                "shape": [int(value) for value in reversed(info.dims)],
            })
            cursor += size
        circuit = json.loads((ROOT / "version/v8/circuits/cohere_transcribe.json").read_text(encoding="utf-8"))
        manifest = {
            "version": 5,
            "model": "cohere_transcribe",
            "source_arch": "cohere-transcribe",
            "source_format": "gguf",
            "bump_layout": {
                "header_size": c.HEADER_SIZE,
                "ext_metadata_size": c.EXT_METADATA_SIZE,
                "data_start": c.DATA_START,
            },
            "config": config,
            "template": circuit,
            "source_tensor_coverage": {
                "total_source_tensors": len(tensors),
                "consumed_source_tensors": len(entries),
                "unconsumed_source_tensors": [],
                "pass": True,
            },
            "num_layers": positive(metadata, "cohere_transcribe.encoder.n_layers"),
            "embed_dim": positive(metadata, "cohere_transcribe.encoder.d_model"),
            "num_heads": positive(metadata, "cohere_transcribe.encoder.n_heads"),
            "head_dim": positive(metadata, "cohere_transcribe.encoder.head_dim"),
            "intermediate_size": positive(metadata, "cohere_transcribe.encoder.ffn_dim"),
            "vocab_size": positive(metadata, "cohere_transcribe.vocab_size"),
            "context_length": positive(metadata, "cohere_transcribe.decoder.max_ctx"),
            "entries": entries,
        }
        print(
            f"[cohere->bump] tensors={len(entries)} bytes={cursor - c.DATA_START} "
            f"plan_only={args.plan_only}"
        )
        if args.plan_only:
            return 0
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / "weights.bump"
        config_path = output_dir / "config.json"
        manifest_path = output_dir / "weights_manifest.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with weights_path.open("w+b") as output:
            output.write(b"\x00" * c.HEADER_SIZE)
            output.write(b"\x00" * c.EXT_METADATA_SIZE)
            writer = c.HashingWriter(output)
            writer.write(struct.pack("<I", len(tensors)))
            writer.write(bytes(
                c.CK_DT_FP16 if info.ggml_type == c.GGML_TYPE_F16 else c.CK_DT_FP32
                for info in tensors.values()
            ))
            position = c.DATA_START + 4 + len(tensors)
            for entry, info in zip(entries, tensors.values()):
                if int(entry["file_offset"]) > position:
                    writer.write(b"\x00" * (int(entry["file_offset"]) - position))
                c.copy_bytes_stream(
                    handle, source_data_start + info.offset, int(entry["size"]), writer
                )
                position = int(entry["file_offset"]) + int(entry["size"])
            checksum = writer.digest()
            manifest_hash = c.calculate_manifest_hash(manifest)
            metadata_blob = c.build_bumpv5_metadata(
                circuit,
                config,
                {"source": "gguf", "storage": "mixed_fp16_fp32"},
                manifest_hash,
                Path(__file__).name,
            )
            metadata_blob["template_hash"] = c.calculate_template_hash(circuit)
            metadata_bytes = c._canonical_json_bytes(metadata_blob)
            metadata_hash = c.calculate_metadata_hash(metadata_blob)
            output.seek(0)
            output.write(b"BUMPWGT5")
            output.write(struct.pack("<I", c.BUMP_VERSION_V5))
            output.write(struct.pack("<I", 1))
            for value in (
                manifest["num_layers"], manifest["vocab_size"], manifest["embed_dim"],
                manifest["intermediate_size"], manifest["context_length"], manifest["num_heads"],
                manifest["num_heads"], manifest["head_dim"],
            ):
                output.write(struct.pack("<I", int(value)))
            for value in (
                manifest["embed_dim"], manifest["head_dim"],
                manifest["intermediate_size"], manifest["context_length"],
            ):
                output.write(struct.pack("<Q", int(value)))
            output.write(struct.pack("<I", 0))
            output.write(struct.pack("<I", 0))
            output.write(checksum)
            output.seek(0, os.SEEK_END)
            output.write(metadata_bytes)
            c.write_bumpv5_footer(output, len(metadata_bytes), metadata_hash)
        print(f"[cohere->bump] wrote {weights_path}")
        return 0
    finally:
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
