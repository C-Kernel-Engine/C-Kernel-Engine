#!/usr/bin/env python3
"""Declare an input-isolation circuit for Kokoro's first source residual ops.

This is a numerical diagnostic with a supplied source-convolution tensor, not
the connected phoneme-to-source acceptance graph.
"""

import argparse
import json
from pathlib import Path

from build_kokoro_source_residual_prefix_circuit import (
    build_circuit as connected_circuit, CAPACITY, CHANNELS, ROOT)


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_source_residual_oracle_input_bounded.json'


def build_circuit():
    connected = connected_circuit()
    block = connected['block_types']['source_residual_prefix']
    block['body']['ops'][0]['graph_slots']['inputs']['input'] = \
        'external:source_conv0'
    names = ('source_conv0', 'decoder_style', 'source_res0_style0',
             'source_res0_norm0', 'source_res0_snake0', 'source_res0_conv0')
    graph = {
        'version': 3, 'name': 'kokoro_source_residual_oracle_input_bounded',
        'family': 'bounded_graph', 'checked_native_entry': True,
        'contract': {'runtime_invariants': {
            'inference_only': True, 'oracle_supplied_source_conv': True,
            'complete_waveform': False}},
        'activation_buffers': {key: connected['activation_buffers'][key]
                               for key in names},
        'activation_bindings': {key: key for key in names},
        'runtime_constants': {'value_elements': 1, 'phoneme_count': 1,
                              'max_duration': CAPACITY,
                              'expanded_capacity': CAPACITY},
        'runtime_lengths': {'source_conv_frames': {
            'producer': 'source_conv_extent', 'result': 'valid_extent',
            'capacity': CAPACITY, 'allow_zero': False}},
        'native_entry': {
            'function': 'ck_kokoro_source_residual_oracle_input_bounded',
            'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                       {'c_type': 'size_t', 'name': 'arena_bytes'},
                       {'c_type': 'int32_t *', 'name': 'out_frames'}],
            'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
            'runtime_length_outputs': {'source_conv_frames': 'out_frames'}},
        'sequence': ['source_residual_prefix'],
        'block_types': {'source_residual_prefix': {
            'sequence': ['header', 'body', 'footer'],
            'header': [{
                'id': 'source_conv_extent', 'op': 'runtime_extent_sum',
                'kernel': 'runtime_extent_sum_i32', 'returns_status': True,
                'produces_runtime_lengths': {
                    'source_conv_frames': 'valid_extent'},
                'params': {'T': 1},
                'graph_slots': {'inputs': {
                    'values': 'external:source_conv_frame_count'},
                    'outputs': {'valid_extent': 'source_conv_extent_value'}}},
                *block['header']],
            'body': block['body'], 'footer': block['footer']}},
        'semantic_checkpoints': {'schema': 'cke.semantic_checkpoint_contract',
            'schema_version': 1, 'exports': {
            key: connected['semantic_checkpoints']['exports'][key]
            for key in names if key.startswith('source_res0_')}},
    }
    graph['activation_buffers']['source_conv_frame_count'] = {'shape': [1]}
    graph['activation_buffers']['source_conv_extent_value'] = {'shape': [1]}
    graph['activation_bindings']['source_conv_frame_count'] = \
        'source_conv_frame_count'
    graph['activation_bindings']['source_conv_extent_value'] = \
        'source_conv_extent_value'
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('source residual oracle-input circuit differs')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
