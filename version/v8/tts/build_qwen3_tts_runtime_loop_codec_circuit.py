#!/usr/bin/env python3
"""Join checked Qwen prompt/continuation lengths to the existing codec graph."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .build_qwen3_tts_loop_codec_circuit import (
        ROOT, build_circuit as fixed_loop,
    )
    from .build_qwen3_tts_runtime_second_frame_circuit import (
        build_circuit as runtime_frames,
    )
else:
    from build_qwen3_tts_loop_codec_circuit import (
        ROOT, build_circuit as fixed_loop,
    )
    from build_qwen3_tts_runtime_second_frame_circuit import (
        build_circuit as runtime_frames,
    )


OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_runtime_loop_codec_bounded.json"


def build_circuit(*, context_capacity: int = 16,
                  frames: int = 46, max_generated_frames: int = 64,
                  talker_layers: int = 28, continuation_steps: int = 14,
                  codec_kwargs: dict | None = None,
                  codec_conv_provider: str = "scalar") -> dict:
    graph = fixed_loop(context_tokens=context_capacity, frames=frames,
                       max_generated_frames=max_generated_frames,
                       talker_layers=talker_layers,
                       continuation_steps=continuation_steps,
                       codec_kwargs=codec_kwargs,
                       codec_conv_provider=codec_conv_provider)
    dynamic = runtime_frames(context_capacity=context_capacity,
                             talker_layers=talker_layers,
                             continuation_steps=continuation_steps,
                             max_generated_frames=max_generated_frames)
    prefix = dynamic["sequence"]
    if graph["sequence"][:len(prefix)] != prefix:
        raise ValueError("codec graph no longer preserves generated frame prefix")
    for index in range(2):
        if (graph["native_entry"]["segments"][index]["last_op"] !=
                dynamic["native_entry"]["segments"][index]["last_op"]):
            raise ValueError("codec graph segment boundary differs")
    for name in prefix:
        graph["block_types"][name] = copy.deepcopy(dynamic["block_types"][name])
    graph["activation_buffers"].update(copy.deepcopy(dynamic["activation_buffers"]))
    graph["activation_bindings"].update(copy.deepcopy(dynamic["activation_bindings"]))
    graph["runtime_lengths"].update(copy.deepcopy(dynamic["runtime_lengths"]))
    graph["runtime_constants"] = copy.deepcopy(dynamic["runtime_constants"])
    graph["contract"]["runtime_invariants"].update(
        dynamic["contract"]["runtime_invariants"])
    graph["contract"]["runtime_invariants"]["complete_text_to_wav"] = False
    graph["name"] = ("qwen3_tts_runtime_loop_codec_bounded" if
                     talker_layers == 28 and continuation_steps == 14 and
                     frames == 46 and max_generated_frames == 64 and
                     codec_kwargs is None else
                     "qwen3_tts_runtime_loop_codec_bounded_synthetic")
    entry = graph["native_entry"]
    entry["function"] = "ck_" + graph["name"]
    for index, segment in enumerate(entry["segments"]):
        segment["function"] = entry["function"] + (
            "_prefill" if index == 0 else "_frame" if index == 1 else "_codec")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("runtime loop-codec circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
