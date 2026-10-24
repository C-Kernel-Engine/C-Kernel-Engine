#!/usr/bin/env python3
"""Connect the first codec ConvNeXt output to the second causal upsample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen3_tts_codec_convnext_complete_circuit import build_circuit as first_block_circuit
from build_qwen3_tts_codec_rvq_circuit import checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_codec_second_upsample_bounded.json"


def build_circuit(*, frames: int = 46, vocabulary: int = 2048,
                  codebook_width: int = 256, output_width: int = 512,
                  acoustic_codebooks: int = 15, latent_width: int = 1024,
                  hidden_width: int = 512, heads: int = 16,
                  head_dim: int = 64, sliding_window: int = 72,
                  intermediate_width: int = 1024, layers: int = 8,
                  depthwise_kernel: int = 7, expansion: int = 4) -> dict:
    graph = first_block_circuit(
        frames=frames, vocabulary=vocabulary, codebook_width=codebook_width,
        output_width=output_width, acoustic_codebooks=acoustic_codebooks,
        latent_width=latent_width, hidden_width=hidden_width, heads=heads,
        head_dim=head_dim, sliding_window=sliding_window,
        intermediate_width=intermediate_width, layers=layers,
        depthwise_kernel=depthwise_kernel, expansion=expansion)
    graph["name"] = "qwen3_tts_codec_second_upsample_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_codec_second_upsample_bounded"
    graph["contract"]["runtime_invariants"]["second_codec_upsample_rate"] = 2
    input_frames = frames * 2
    output_frames = input_frames * 2
    name = "second_upsample"
    graph["activation_buffers"][name] = {
        "shape": [latent_width, output_frames]}
    graph["activation_bindings"][name] = name
    op = {
        "id": name, "op": "audio_conv_transpose1d_dense_checked",
        "kernel": "audio_conv_transpose1d_dense_channel_major_f32_checked",
        "returns_status": True,
        "weight_refs": {
            "weight": "codec.decoder.upsample.1.0.conv.weight",
            "bias": "codec.decoder.upsample.1.0.conv.bias"},
        "graph_slots": {"inputs": {"input": "convnext_output"},
                        "outputs": {"output": name}},
        "params": {
            "I": latent_width, "O": latent_width, "T": input_frames,
            "K": 2, "U": output_frames,
            "Q": latent_width * output_frames,
            "call_constants": {
                "dense_deconv_input_elements": latent_width * input_frames,
                "dense_deconv_input_stride": input_frames,
                "dense_deconv_weight_elements": latent_width * latent_width * 2,
                "dense_deconv_bias_elements": latent_width,
                "dense_deconv_output_elements": latent_width * output_frames,
                "dense_deconv_output_stride": output_frames,
                "dense_deconv_scratch_elements": latent_width * output_frames,
                "dense_deconv_input_channels": latent_width,
                "dense_deconv_output_channels": latent_width,
                "dense_deconv_input_frames": input_frames,
                "dense_deconv_kernel_size": 2,
                "dense_deconv_stride": 2,
                "dense_deconv_padding": 0,
                "dense_deconv_output_padding": 0,
                "dense_deconv_output_frames": output_frames}}}
    graph["block_types"]["codec_semantic"]["body"]["ops"].append(op)
    graph["semantic_checkpoints"]["exports"][name] = checkpoint(
        op, "channel_major")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("codec second-upsample circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
