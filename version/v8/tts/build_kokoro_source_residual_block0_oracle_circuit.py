#!/usr/bin/env python3
"""Declare complete source residual block with diagnostic direct-convolution input."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_source_residual_block0_circuit import (
    build_circuit as connected_circuit, ROOT)
from build_kokoro_source_residual_first_pair_oracle_circuit import (
    build_circuit as first_pair_oracle_circuit)


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_source_residual_block0_oracle_input_bounded.json'


def build_circuit():
    graph = first_pair_oracle_circuit()
    connected = connected_circuit()
    graph['name'] = 'kokoro_source_residual_block0_oracle_input_bounded'
    graph['native_entry']['function'] = \
        'ck_kokoro_source_residual_block0_oracle_input_bounded'
    for pair in (1, 2):
        name = f'source_residual_pair{pair}'
        graph['sequence'].append(name)
        block = copy.deepcopy(connected['block_types'][name])
        graph['block_types'][name] = block
        for op in block['header'] + block['body']['ops'] + block['footer']:
            key = op['id']
            graph['activation_buffers'][key] = connected['activation_buffers'][key]
            graph['activation_bindings'][key] = key
            graph['semantic_checkpoints']['exports'][key] = \
                connected['semantic_checkpoints']['exports'][key]
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('source residual block oracle-input circuit differs')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
