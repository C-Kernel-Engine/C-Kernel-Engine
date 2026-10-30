#!/usr/bin/env python3
"""Declare a checked Qwen code-loop to codec handoff on one planned arena.

The native shell repeats the prefill/frame segments. This circuit owns the
frame-major to codebook-major connections and the downstream codec schedule.
The full-entry wrapper is not a synthesis loop; callers use its three segments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_second_generated_frame_circuit import (
        ROOT, build_circuit as generated_frames, copy_ids,
    )
    from .build_qwen3_tts_first_frame_waveform_circuit import attach_codec
else:
    from build_qwen3_tts_second_generated_frame_circuit import (
        ROOT, build_circuit as generated_frames, copy_ids,
    )
    from build_qwen3_tts_first_frame_waveform_circuit import attach_codec

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_loop_codec_bounded.json"
SCALAR_CODEC_CONV = "audio_conv1d_dilated_checked_channel_major_f32"
AVX2_CODEC_CONV = SCALAR_CODEC_CONV + "_avx2_fma"


def build_circuit(*, context_tokens: int = 7, frames: int = 5,
                  max_generated_frames: int = 64,
                  talker_layers: int = 28, continuation_steps: int = 14,
                  codec_kwargs: dict | None = None,
                  codec_conv_provider: str = "scalar") -> dict:
    if (isinstance(frames, bool) or not isinstance(frames, int)
            or not 2 <= frames <= max_generated_frames):
        raise ValueError("frames must be in 2..max_generated_frames")
    codebooks = continuation_steps + 2
    if codec_kwargs is not None and codec_kwargs.get("acoustic_codebooks", 15) != codebooks - 1:
        raise ValueError("codec acoustic codebooks must match generated codes")
    if codec_conv_provider not in ("scalar", "avx2_fma"):
        raise ValueError("unsupported codec Conv1D provider")
    graph = generated_frames(
        context_tokens=context_tokens, talker_layers=talker_layers,
        continuation_steps=continuation_steps,
        max_generated_frames=max_generated_frames)
    canonical = (context_tokens == 7 and frames == 5
                 and max_generated_frames == 64 and
                 talker_layers == 28 and continuation_steps == 14 and
                 codec_kwargs is None)
    graph["name"] = ("qwen3_tts_loop_codec_bounded" if canonical else
                     "qwen3_tts_loop_codec_bounded_synthetic")
    entry = graph["native_entry"]
    entry["function"] = "ck_qwen3_tts_loop_codec_bounded"
    for index, segment in enumerate(entry["segments"]):
        segment["function"] = entry["function"] + ("_prefill" if index == 0 else "_frame")
    graph["contract"]["runtime_invariants"].update({
        "loop_codec_frames": frames,
        "loop_codec_input_layout": "frame_major_i32",
        "loop_codec_segments": ["prefill", "frame", "codec"],
        "natural_eos_certified": False,
        "complete_text_to_wav": False,
    })
    shapes, bindings = graph["activation_buffers"], graph["activation_bindings"]
    source = "loop_frame_major_codes"
    shapes[source] = {"shape": [frames * codebooks], "lifetime": "sequence"}
    bindings[source] = source
    ids = {}
    ops = []
    for codebook in range(codebooks):
        name = ("semantic_ids" if codebook == 0 else
                f"acoustic_ids_{codebook - 1}")
        packed = "packed_loop_" + name
        shapes[packed] = {"shape": [frames]}
        bindings[packed] = packed
        ids[name] = packed
        for frame in range(frames):
            ops.append(copy_ids(
                f"pack_loop_{name}_{frame}", "external:" + source,
                packed if frame == 0 else "external:" + packed,
                frames * codebooks, frames,
                source_offset=frame * codebooks + codebook,
                destination_offset=frame))
    graph["sequence"].append("pack_loop_codec_ids")
    graph["block_types"]["pack_loop_codec_ids"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": ops}, "footer": [],
    }
    graph = attach_codec(graph, frames=frames, ids=ids,
                         codec_kwargs=codec_kwargs)
    if codec_conv_provider == "avx2_fma":
        selected = 0
        for block_name in graph["sequence"]:
            if not block_name.startswith("codec_"):
                continue
            block = graph["block_types"][block_name]
            for op in (block.get("header", []) +
                       block.get("body", {}).get("ops", []) +
                       block.get("footer", [])):
                if op.get("kernel") == SCALAR_CODEC_CONV:
                    op["kernel"] = AVX2_CODEC_CONV
                    selected += 1
        if selected == 0:
            raise ValueError("codec has no selectable Conv1D operations")
        graph["name"] += "_avx2_fma"
        graph["contract"]["runtime_invariants"]["codec_conv_provider"] = AVX2_CODEC_CONV
    entry["segments"].append({
        "function": entry["function"] + "_codec",
        "last_op": "publish_waveform_sample_count",
    })
    # Reduced synthetic geometries inherit the canonical checkpoint catalogue.
    # Retain only declarations for operations present in this circuit; the
    # full pinned geometry keeps the entire catalogue unchanged.
    operation_ids = {op["id"] for block in graph["sequence"]
                     for op in graph["block_types"][block].get("body", {}).get("ops", [])}
    operation_ids.update(op["id"] for block in graph["sequence"]
                         for section in ("header", "footer")
                         for op in graph["block_types"][block].get(section, []))
    exports = graph["semantic_checkpoints"]["exports"]
    graph["semantic_checkpoints"]["exports"] = {
        name: declaration for name, declaration in exports.items()
        if declaration["template_op_id"] in operation_ids
    }
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("loop-to-codec circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
