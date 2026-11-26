#!/usr/bin/env python3
"""Connect runtime Kokoro prosody F0 to the source projection beside stage0."""

import argparse
import json
from pathlib import Path

from build_kokoro_generated_f0_source_circuit import (
    build_circuit as source_projection_circuit)
from build_kokoro_runtime_generator_stage0_circuit import (
    build_circuit as runtime_stage0_circuit)
from build_kokoro_runtime_decoder_complete_circuit import UPSAMPLED_FRAMES


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_runtime_source_projection_bounded.json'


def build_circuit():
    graph = source_projection_circuit(
        prosody_graph=runtime_stage0_circuit(),
        f0_capacity=UPSAMPLED_FRAMES)
    graph['name'] = 'kokoro_runtime_source_projection_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_runtime_source_projection_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_runtime_source_projection=True,
        captured_gaussian_input=True, native_rng=False,
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
            raise SystemExit('runtime source projection differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
