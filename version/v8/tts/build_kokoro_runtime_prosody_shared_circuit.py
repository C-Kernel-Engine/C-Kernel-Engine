#!/usr/bin/env python3
"""Join bounded runtime text/duration expansion to the shared prosody scan."""
import argparse
import copy
import json
from pathlib import Path

from build_kokoro_runtime_text_expansion_circuit import (
    build_circuit as expanded_circuit, FRAMES)
from build_kokoro_prosody_shared_circuit import build_circuit as fixed_shared

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_runtime_prosody_shared_bounded.json'


def build_circuit():
    graph = expanded_circuit()
    fixed = fixed_shared()
    graph['name'] = 'kokoro_runtime_prosody_shared_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_runtime_prosody_shared_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_prosody_shared_scan=True, complete_prosody=False,
        complete_waveform=False)
    block = copy.deepcopy(fixed['block_types']['prosody_shared'])
    transpose = block['header'][0]
    scan = block['body']['ops'][0]
    transpose['params']['C'] = FRAMES
    transpose['params']['call_constants'].update(
        transpose_input_elements=640 * FRAMES,
        transpose_input_stride=FRAMES,
        transpose_output_elements=640 * FRAMES)
    scan['params']['T'] = FRAMES
    scan['params']['call_constants'].update(
        scan_input_elements=FRAMES * 640,
        scan_output_elements=FRAMES * 512)
    graph['block_types']['prosody_shared'] = block
    graph['sequence'].append('prosody_shared')
    for name, channels in (('prosody_input_token', 640),
                           ('prosody_shared_features', 512)):
        graph['activation_buffers'][name] = {'shape': [FRAMES, channels]}
        graph['activation_bindings'][name] = name
    for name in ('prosody_input_transpose', 'prosody_shared_scan'):
        graph['semantic_checkpoints']['exports'][name] = copy.deepcopy(
            fixed['semantic_checkpoints']['exports'][name])
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('runtime prosody circuit differs from authoring source')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
