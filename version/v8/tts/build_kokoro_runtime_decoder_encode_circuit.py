#!/usr/bin/env python3
"""Declare the first decoder residual block behind checked frame lengths."""
import argparse
import json
from pathlib import Path

from build_kokoro_runtime_decoder_ingress_circuit import (
    build_circuit as ingress_circuit, FRAMES)
from build_kokoro_decoder_encode_circuit import (
    STYLE, PREFIX, INPUT_CHANNELS, OUTPUT_CHANNELS,
    add_decoder_block, decoder_residual_block_ops)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_runtime_decoder_encode_bounded.json'


def build_circuit():
    graph = ingress_circuit()
    graph['name'] = 'kokoro_runtime_decoder_encode_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_runtime_decoder_encode_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_decoder_encode=True, complete_decoder=False,
        complete_waveform=False)
    graph['activation_buffers']['decoder_style'] = {'shape': [STYLE]}
    graph['activation_bindings']['decoder_style'] = 'decoder_style'
    add_decoder_block(graph, 'decoder_encode', decoder_residual_block_ops(
        'decoder_encode', 'decoder_joined', PREFIX,
        INPUT_CHANNELS, OUTPUT_CHANNELS, FRAMES))
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('runtime decoder encode differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
