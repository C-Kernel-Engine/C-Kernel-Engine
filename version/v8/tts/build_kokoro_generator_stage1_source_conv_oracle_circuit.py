#!/usr/bin/env python3
"""Isolate the second source convolution with direct pinned STFT channels."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_generator_stage1_source_conv_circuit import (
    ROOT, INPUT_CHANNELS, OUTPUT_CHANNELS, PADDED_CAPACITY,
    build_circuit as connected_circuit,
)


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generator_stage1_source_conv_oracle_input_bounded.json'


def build_circuit():
    connected = connected_circuit()
    block = copy.deepcopy(connected['block_types']['generator_stage1_source_conv'])
    op = block['body']['ops'][0]
    op['graph_slots']['inputs']['input'] = 'external:source_stft_channels'
    op['consumes_runtime_lengths'] = ['source_stft_frames']
    graph = {
        'version': 3,
        'name': 'kokoro_generator_stage1_source_conv_oracle_input_bounded',
        'family': 'bounded_graph', 'checked_native_entry': True,
        'contract': {'runtime_invariants': {
            'inference_only': True, 'oracle_supplied_stft_channels': True,
            'complete_waveform': False}},
        'activation_buffers': {
            'source_stft_channels': {'shape': [INPUT_CHANNELS, PADDED_CAPACITY]},
            'source_frame_count': {'shape': [1]},
            'source_extent_value': {'shape': [1]},
            'generator_stage1_source_conv': {
                'shape': [OUTPUT_CHANNELS, PADDED_CAPACITY]}},
        'activation_bindings': {name: name for name in (
            'source_stft_channels', 'source_frame_count',
            'source_extent_value', 'generator_stage1_source_conv')},
        'runtime_constants': {'value_elements': 1, 'phoneme_count': 1,
                              'max_duration': PADDED_CAPACITY,
                              'expanded_capacity': PADDED_CAPACITY},
        'runtime_lengths': {'source_stft_frames': {
            'producer': 'source_extent', 'result': 'valid_extent',
            'capacity': PADDED_CAPACITY, 'allow_zero': False}},
        'native_entry': {
            'function': 'ck_kokoro_generator_stage1_source_conv_oracle_input_bounded',
            'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                       {'c_type': 'size_t', 'name': 'arena_bytes'},
                       {'c_type': 'int32_t *', 'name': 'out_source_frames'}],
            'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
            'runtime_length_outputs': {
                'source_stft_frames': 'out_source_frames'}},
        'sequence': ['source_extent_component', 'generator_stage1_source_conv'],
        'block_types': {
            'source_extent_component': {
                'sequence': ['header', 'body', 'footer'],
                'header': [{
                    'id': 'source_extent', 'op': 'runtime_extent_sum',
                    'kernel': 'runtime_extent_sum_i32', 'returns_status': True,
                    'produces_runtime_lengths': {
                        'source_stft_frames': 'valid_extent'},
                    'params': {'T': 1},
                    'graph_slots': {'inputs': {
                        'values': 'external:source_frame_count'},
                        'outputs': {'valid_extent': 'source_extent_value'}}}],
                'body': {'type': 'dense', 'ops': []}, 'footer': []},
            'generator_stage1_source_conv': block},
        'semantic_checkpoints': {
            'schema': 'cke.semantic_checkpoint_contract',
            'schema_version': 1, 'exports': {
                'generator_stage1_source_conv': copy.deepcopy(
                    connected['semantic_checkpoints']['exports'][
                        'generator_stage1_source_conv'])}}}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('source convolution oracle circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
