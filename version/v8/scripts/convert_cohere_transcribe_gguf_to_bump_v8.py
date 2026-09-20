#!/usr/bin/env python3
"""Convert a complete Cohere Transcribe F16/F32 GGUF into BUMPWGT5."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
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


COMPONENT_SCOPES = (
    "audio_frontend",
    "audio_subsampling",
    "audio_encoder_block",
    "audio_encoder",
    "audio_decoder",
)


_DECODER_LAYER_WEIGHT_ROLES = {
    "attn_ln.weight": "ln1_gamma",
    "attn_ln.bias": "ln1_beta",
    "attn_q.weight": "wq",
    "attn_q.bias": "bq",
    "attn_k.weight": "wk",
    "attn_k.bias": "bk",
    "attn_v.weight": "wv",
    "attn_v.bias": "bv",
    "attn_o.weight": "wo",
    "attn_o.bias": "bo",
    "cross_ln.weight": "cross_ln_gamma",
    "cross_ln.bias": "cross_ln_beta",
    "cross_q.weight": "cross_wq",
    "cross_q.bias": "cross_bq",
    "cross_k.weight": "cross_wk",
    "cross_k.bias": "cross_bk",
    "cross_v.weight": "cross_wv",
    "cross_v.bias": "cross_bv",
    "cross_o.weight": "cross_wo",
    "cross_o.bias": "cross_bo",
    "ffn_ln.weight": "ln2_gamma",
    "ffn_ln.bias": "ln2_beta",
    "ffn_up.weight": "w3",
    "ffn_up.bias": "b1",
    "ffn_down.weight": "w2",
    "ffn_down.bias": "b2",
}


def _component_weight_name(name: str, artifact_scope: str) -> str:
    if artifact_scope != "audio_decoder":
        return name
    global_names = {
        "dec.emb.weight": "token_emb",
        "dec.pos.weight": "pos_emb",
        "dec.emb_ln.weight": "embedding_ln_weight",
        "dec.emb_ln.bias": "embedding_ln_bias",
        "dec.out_ln.weight": "final_ln_weight",
        "dec.out_ln.bias": "final_ln_bias",
        "dec.head.weight": "lm_head",
        "dec.head.bias": "lm_head_bias",
    }
    if name in global_names:
        return global_names[name]
    match = re.fullmatch(r"dec\.blk\.(\d+)\.(.+)", name)
    if match and match.group(2) in _DECODER_LAYER_WEIGHT_ROLES:
        return f"layer.{match.group(1)}.{_DECODER_LAYER_WEIGHT_ROLES[match.group(2)]}"
    return name


def _component_tensors(tensors, artifact_scope: str):
    if artifact_scope == "audio_decoder":
        return [(name, info) for name, info in tensors.items() if name.startswith("dec.")]
    return list(tensors.items())


def _component_circuit_name(artifact_scope: str) -> str:
    return (
        "audio_transformer_decoder.json"
        if artifact_scope == "audio_decoder"
        else "cohere_transcribe.json"
    )


def _promote_to_fp32(name: str, ggml_type: int, artifact_scope: str) -> bool:
    if ggml_type != c.GGML_TYPE_F16:
        return False
    if name == "fe.mel_fb":
        return artifact_scope in {"audio_frontend", "audio_subsampling", "audio_encoder"}
    if name.startswith("enc.pre."):
        return artifact_scope in {"audio_subsampling", "audio_encoder"}
    if name.startswith("enc.blk."):
        return artifact_scope in {"audio_encoder_block", "audio_encoder"}
    if name.startswith("enc.proj."):
        return artifact_scope == "audio_encoder"
    if name.startswith("dec."):
        return artifact_scope == "audio_decoder"
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-scope",
        choices=COMPONENT_SCOPES,
        default="audio_frontend",
        help="Generated component to retain and prepare in the BUMP artifact",
    )
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
        sample_rate = positive(metadata, "cohere_transcribe.audio.sample_rate")
        hop_length = positive(metadata, "cohere_transcribe.audio.hop_length")
        max_source_frames = (
            positive(metadata, "cohere_transcribe.audio.max_clip_s") * sample_rate
        )
        n_fft = positive(metadata, "cohere_transcribe.audio.n_fft")
        encoder_dim = positive(metadata, "cohere_transcribe.encoder.d_model")
        encoder_heads = positive(metadata, "cohere_transcribe.encoder.n_heads")
        encoder_head_dim = positive(metadata, "cohere_transcribe.encoder.head_dim")
        encoder_layers = positive(metadata, "cohere_transcribe.encoder.n_layers")
        encoder_ffn_dim = positive(metadata, "cohere_transcribe.encoder.ffn_dim")
        encoder_conv_kernel = positive(metadata, "cohere_transcribe.encoder.conv_kernel")
        decoder_dim = positive(metadata, "cohere_transcribe.decoder.d_model")
        decoder_heads = positive(metadata, "cohere_transcribe.decoder.n_heads")
        decoder_layers = positive(metadata, "cohere_transcribe.decoder.n_layers")
        decoder_ffn_dim = positive(metadata, "cohere_transcribe.decoder.ffn_dim")
        vocab_size = positive(metadata, "cohere_transcribe.vocab_size")
        decoder_context = positive(metadata, "cohere_transcribe.decoder.max_ctx")
        feature_frames = max_source_frames // hop_length + 1
        subsampling_frames = (feature_frames + 7) // 8
        conv0 = tensors.get("enc.pre.conv.0.weight")
        if conv0 is None or len(conv0.dims) != 4:
            raise c.GGUFError("enc.pre.conv.0.weight must be a rank-4 tensor")
        subsampling_kernel = int(conv0.dims[0])
        if int(conv0.dims[1]) != subsampling_kernel or subsampling_kernel <= 0:
            raise c.GGUFError("Cohere subsampling requires a square convolution kernel")
        conv_channels = int(conv0.dims[-1])
        if encoder_heads * encoder_head_dim != encoder_dim:
            raise c.GGUFError(
                "Cohere encoder head geometry does not match the model width"
            )
        if decoder_dim % decoder_heads != 0:
            raise c.GGUFError(
                "Cohere decoder width is not divisible by its attention heads"
            )
        stage0_frames = (feature_frames + 1) // 2
        feature_channels = positive(metadata, "cohere_transcribe.audio.n_mels")
        stage0_width = (feature_channels + 1) // 2
        subsampling_workspace_elements = (
            2 * conv_channels * stage0_frames * stage0_width
        )
        block_token_elements = subsampling_frames * encoder_dim
        block_large_elements = max(
            subsampling_frames * encoder_ffn_dim,
            (2 * subsampling_frames - 1) * encoder_dim,
            2 * block_token_elements,
        )
        block_workspace_elements = (
            5 * block_token_elements
            + block_large_elements
            + encoder_heads * subsampling_frames
        )
        config.update({
            "model": "cohere_transcribe",
            "model_type": "cohere_transcribe",
            "source_architecture": "cohere-transcribe",
            "tensor_count": len(tensors),
            "embed_dim": encoder_dim,
            "hidden_size": encoder_dim,
            "num_heads": encoder_heads,
            "num_attention_heads": encoder_heads,
            "num_kv_heads": encoder_heads,
            "head_dim": encoder_head_dim,
            "num_layers": encoder_layers,
            "intermediate_size": encoder_ffn_dim,
            "vocab_size": vocab_size,
            "context_length": decoder_context,
            "artifact_scope": args.artifact_scope,
            "audio_include_frontend": args.artifact_scope in {
                "audio_frontend", "audio_subsampling", "audio_encoder"
            },
            "audio_include_subsampling": args.artifact_scope in {
                "audio_subsampling", "audio_encoder"
            },
            "audio_include_encoder_blocks": args.artifact_scope in {
                "audio_encoder_block", "audio_encoder"
            },
            "external_activation_slots": (
                ["audio_encoder_tokens"]
                if args.artifact_scope == "audio_encoder_block"
                else []
            ),
            "audio_activation_profile": "circuit_declared",
            "audio_frontend_asset_policy": "model_assets",
            "audio_sample_rate": sample_rate,
            "audio_max_source_frames": max_source_frames,
            "audio_hop_length": hop_length,
            "audio_n_fft": n_fft,
            "audio_power_bins": n_fft // 2 + 1,
            "audio_window_length": positive(metadata, "cohere_transcribe.audio.win_length"),
            "audio_feature_channels": feature_channels,
            "audio_feature_frames": feature_frames,
            "audio_feature_live_frames": feature_frames,
            "audio_subsampling_conv_channels": conv_channels,
            "audio_subsampling_kernel_size": subsampling_kernel,
            "audio_subsampling_stride": 2,
            "audio_subsampling_factor": 8,
            "audio_subsampling_output_frames": subsampling_frames,
            "audio_subsampling_workspace_elements": subsampling_workspace_elements,
            "audio_subsampling_workspace_bytes": subsampling_workspace_elements * 4,
            "audio_relative_position_frames": 2 * subsampling_frames - 1,
            "audio_fastconformer_conv_kernel_size": encoder_conv_kernel,
            "audio_fastconformer_layer_norm_epsilon": 1.0e-5,
            "audio_fastconformer_batch_norm_epsilon": 1.0e-5,
            "audio_fastconformer_block_workspace_elements": block_workspace_elements,
            "audio_fastconformer_block_workspace_bytes": block_workspace_elements * 4,
            "audio_encoder_projection_size": decoder_dim,
            "audio_include_encoder_projection": args.artifact_scope == "audio_encoder",
            "audio_preemphasis_coefficient": 0.97,
            "audio_log_epsilon": 2.0 ** -24,
            "audio_normalization_epsilon": 1.0e-5,
            "audio_normalization_frame_policy": "all_stft_frames",
        })
        if args.artifact_scope == "audio_decoder":
            config.update({
                "embed_dim": decoder_dim,
                "hidden_size": decoder_dim,
                "num_heads": decoder_heads,
                "num_attention_heads": decoder_heads,
                "num_kv_heads": decoder_heads,
                "num_key_value_heads": decoder_heads,
                "head_dim": decoder_dim // decoder_heads,
                "attention_scale": (decoder_dim // decoder_heads) ** -0.5,
                "num_layers": decoder_layers,
                "num_hidden_layers": decoder_layers,
                "intermediate_size": decoder_ffn_dim,
                "encoder_memory_length": subsampling_frames,
                "dynamic_encoder_memory_length": True,
                "uses_cross_attention": True,
                "decode_kv_cache_dtype": "fp16",
                "decoder_activation": "relu",
                "decoder_embedding_layernorm": True,
                "decoder_output_bias": True,
                "tie_word_embeddings": False,
                "has_attention_biases": True,
                "rms_eps": 1.0e-5,
                "prefer_q8_activation": False,
                "prefill_policy": "encoder_decoder",
                "audio_include_frontend": False,
                "audio_include_subsampling": False,
                "audio_include_encoder_blocks": False,
                "audio_include_encoder_projection": False,
            })
        component_tensors = _component_tensors(tensors, args.artifact_scope)
        if not component_tensors:
            raise c.GGUFError(
                f"no tensors selected for artifact scope {args.artifact_scope}"
            )
        component_source_names = {name for name, _ in component_tensors}
        config["source_tensor_count"] = len(tensors)
        config["tensor_count"] = len(component_tensors)
        entries = []
        cursor = c.DATA_START + 4 + len(component_tensors)
        for name, info in component_tensors:
            cursor = c.align_up(cursor, 32)
            promote_to_fp32 = _promote_to_fp32(
                info.name, info.ggml_type, args.artifact_scope
            )
            size = (
                math.prod(info.dims) * 4
                if promote_to_fp32
                else c.ggml_tensor_bytes(info)
            )
            entries.append({
                "name": _component_weight_name(name, args.artifact_scope),
                "dtype": (
                    "fp32"
                    if promote_to_fp32 or info.ggml_type == c.GGML_TYPE_F32
                    else "fp16"
                ),
                "file_offset": cursor,
                "size": size,
                "source_name": name,
                "source_dtype": c.ggml_type_name(info.ggml_type),
                "conversion": (
                    "fp16_to_fp32_exact" if promote_to_fp32 else "identity"
                ),
                "shape": [int(value) for value in reversed(info.dims)],
            })
            cursor += size
        circuit_name = _component_circuit_name(args.artifact_scope)
        circuit = json.loads(
            (ROOT / "version/v8/circuits" / circuit_name).read_text(encoding="utf-8")
        )
        if args.artifact_scope == "audio_decoder":
            circuit.setdefault("contract", {}).setdefault("artifact", {}).update({
                "source_architecture": "cohere-transcribe",
                "component": "decoder",
            })
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
                "unconsumed_source_tensors": [
                    name for name in tensors if name not in component_source_names
                ],
                "pass": True,
                "scope": args.artifact_scope,
            },
            "num_layers": int(config["num_layers"]),
            "embed_dim": int(config["embed_dim"]),
            "num_heads": int(config["num_heads"]),
            "head_dim": int(config["head_dim"]),
            "intermediate_size": int(config["intermediate_size"]),
            "vocab_size": vocab_size,
            "context_length": decoder_context,
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
            writer.write(struct.pack("<I", len(entries)))
            writer.write(bytes(
                c.CK_DT_FP32 if entry["dtype"] == "fp32" else c.CK_DT_FP16
                for entry in entries
            ))
            position = c.DATA_START + 4 + len(entries)
            for entry, (_, info) in zip(entries, component_tensors):
                if int(entry["file_offset"]) > position:
                    writer.write(b"\x00" * (int(entry["file_offset"]) - position))
                if entry.get("conversion") == "fp16_to_fp32_exact":
                    c.copy_f16_to_f32_stream(
                        handle,
                        source_data_start + info.offset,
                        math.prod(info.dims),
                        writer,
                    )
                else:
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
