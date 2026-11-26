#!/usr/bin/env python3
"""Connect the checked runtime decoder to Kokoro's first generator upsample."""

import argparse
import json
from pathlib import Path

from build_kokoro_generator_stage0_circuit import (
    OUTPUT_CAPACITY, build_circuit as generator_stage0_circuit)
from build_kokoro_runtime_decoder_complete_circuit import (
    UPSAMPLED_FRAMES, build_circuit as runtime_decoder_circuit)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_runtime_generator_stage0_bounded.json'


def build_circuit():
    graph = generator_stage0_circuit(
        decoder_graph=runtime_decoder_circuit(),
        input_capacity=UPSAMPLED_FRAMES,
        output_capacity=OUTPUT_CAPACITY)
    graph['name'] = 'kokoro_runtime_generator_stage0_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_runtime_generator_stage0_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_complete_acoustic_decoder=True,
        generated_generator_main_upsample_stage0=True,
        complete_generator=False, complete_waveform=False)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('runtime generator stage0 differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
