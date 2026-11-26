#!/usr/bin/env python3
"""Declare Kokoro's complete acoustic decoder at checked runtime frame lengths."""

import argparse
import json
from pathlib import Path

from build_kokoro_decoder_complete_circuit import (
    INPUT_CHANNELS, decoder_join, upsampling_block_ops)
from build_kokoro_decoder_encode_circuit import (
    add_decoder_block, decoder_residual_block_ops)
from build_kokoro_runtime_decoder_encode_circuit import (
    build_circuit as decoder_encode_circuit, FRAMES)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_runtime_decoder_complete_bounded.json'
UPSAMPLED_FRAMES = 2 * FRAMES


def build_circuit():
    graph = decoder_encode_circuit()
    graph['name'] = 'kokoro_runtime_decoder_complete_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_runtime_decoder_complete_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_complete_acoustic_decoder=True, complete_decoder=True,
        complete_waveform=False)
    previous = 'decoder_encode_output'
    for index in range(4):
        stem = f'decoder_decode{index}'
        joins = decoder_join(index, previous, FRAMES)
        block = (decoder_residual_block_ops(
            stem, f'{stem}_input', f'waveform_decoder.decode.{index}',
            INPUT_CHANNELS, 1024, FRAMES)
            if index < 3 else upsampling_block_ops(FRAMES, UPSAMPLED_FRAMES))
        add_decoder_block(graph, stem, joins + block)
        previous = f'{stem}_output'
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('runtime complete decoder differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
