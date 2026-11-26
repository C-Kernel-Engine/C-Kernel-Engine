#!/usr/bin/env python3
"""Connect runtime Kokoro source samples to STFT and first source convolution."""

import argparse
import json
from pathlib import Path

from build_kokoro_generated_source_stft_conv_circuit import (
    build_circuit as source_stft_conv_circuit)
from build_kokoro_runtime_source_projection_circuit import (
    build_circuit as runtime_source_projection_circuit)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_runtime_source_stft_conv_bounded.json'


def build_circuit():
    graph = source_stft_conv_circuit(
        source_graph=runtime_source_projection_circuit())
    graph['name'] = 'kokoro_runtime_source_stft_conv_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_runtime_source_stft_conv_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_runtime_source_stft_conv=True,
        scalar_raw_phase_compatibility=False,
        captured_gaussian_input=True,
        native_rng=False, complete_generator=False, complete_waveform=False)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('runtime source STFT/convolution differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
