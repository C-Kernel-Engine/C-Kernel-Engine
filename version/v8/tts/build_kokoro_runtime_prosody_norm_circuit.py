#!/usr/bin/env python3
"""Join variable-length shared prosody to the first F0 and noise AdaIN stages."""
import argparse
import copy
import json
from pathlib import Path

from build_kokoro_runtime_prosody_shared_circuit import (
    build_circuit as runtime_shared, FRAMES)
from build_kokoro_prosody_norm_circuit import build_circuit as fixed_norm

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_runtime_prosody_norm_bounded.json'


def build_circuit():
    graph = runtime_shared()
    fixed = fixed_norm()
    graph['name'] = 'kokoro_runtime_prosody_norm_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_runtime_prosody_norm_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_first_f0_noise_norm=True, complete_prosody=False,
        complete_waveform=False)
    block = copy.deepcopy(fixed['block_types']['prosody_norm'])
    transpose = block['header'][0]
    transpose['params']['R'] = FRAMES
    transpose['params']['call_constants'].update(
        transpose_input_elements=512 * FRAMES,
        transpose_output_elements=512 * FRAMES,
        transpose_output_stride=FRAMES)
    for op in block['footer']:
        params = op['params']
        params['T'] = FRAMES
        params['call_constants'].update(
            adain_input_elements=512 * FRAMES,
            adain_input_stride=FRAMES,
            adain_output_elements=512 * FRAMES,
            adain_output_stride=FRAMES)
    graph['block_types']['prosody_norm'] = block
    graph['sequence'].append('prosody_norm')
    for name in ('prosody_shared_channel', 'f0_norm0_output',
                 'n_norm0_output'):
        graph['activation_buffers'][name] = {'shape': [512, FRAMES]}
        graph['activation_bindings'][name] = name
    for name in ('f0_norm0_style', 'n_norm0_style'):
        graph['activation_buffers'][name] = {'shape': [1024]}
        graph['activation_bindings'][name] = name
    for op in (block['header'] + block['body']['ops'] + block['footer']):
        if op['id'] in fixed['semantic_checkpoints']['exports']:
            graph['semantic_checkpoints']['exports'][op['id']] = copy.deepcopy(
                fixed['semantic_checkpoints']['exports'][op['id']])
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('runtime prosody norm circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
