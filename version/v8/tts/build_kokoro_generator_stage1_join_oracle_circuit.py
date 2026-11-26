#!/usr/bin/env python3
"""Isolate second source residual and join using direct pinned branch inputs."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_generator_stage1_join_circuit import (
    ROOT, CHANNELS, PADDED_CAPACITY, SOURCE_BLOCKS,
    build_circuit as connected_circuit,
)


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generator_stage1_join_oracle_input_bounded.json'


def build_circuit():
    connected = connected_circuit()
    names = [f'stage1_{name}' for name in SOURCE_BLOCKS]
    names.append('generator_stage1_join')
    blocks = {name: copy.deepcopy(connected['block_types'][name])
              for name in names}
    blocks['stage1_source_residual_prefix']['body']['ops'][0][
        'graph_slots']['inputs']['input'] = \
        'external:generator_stage1_source_conv'
    blocks['stage1_source_residual_first_pair']['footer'][0][
        'graph_slots']['inputs']['right'] = \
        'external:generator_stage1_source_conv'
    join = blocks['generator_stage1_join']['body']['ops'][0]
    join['graph_slots']['inputs']['left'] = \
        'external:generator_stage1_reflection'
    join['consumes_runtime_lengths'] = ['source_stft_frames']
    join['runtime_scalar_bindings']['sum_columns'] = 'source_stft_frames'
    buffers = {
        'generator_stage1_source_conv': {
            'shape': [CHANNELS, PADDED_CAPACITY]},
        'generator_stage1_reflection': {
            'shape': [CHANNELS, PADDED_CAPACITY]},
        'decoder_style': {'shape': [128]},
        'source_frame_count': {'shape': [1]},
        'source_extent_value': {'shape': [1]},
    }
    checkpoints = {}
    for block in blocks.values():
        for op in block['header'] + block['body']['ops'] + block['footer']:
            buffers[op['id']] = copy.deepcopy(
                connected['activation_buffers'][op['id']])
            checkpoints[op['id']] = copy.deepcopy(
                connected['semantic_checkpoints']['exports'][op['id']])
    blocks['source_extent_component'] = {
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
        'body': {'type': 'dense', 'ops': []}, 'footer': []}
    return {
        'version': 3,
        'name': 'kokoro_generator_stage1_join_oracle_input_bounded',
        'family': 'bounded_graph', 'checked_native_entry': True,
        'contract': {'runtime_invariants': {
            'inference_only': True, 'oracle_supplied_source_conv': True,
            'oracle_supplied_main_reflection': True,
            'complete_waveform': False}},
        'activation_buffers': buffers,
        'activation_bindings': {name: name for name in buffers},
        'runtime_constants': {'value_elements': 1, 'phoneme_count': 1,
                              'max_duration': PADDED_CAPACITY,
                              'expanded_capacity': PADDED_CAPACITY},
        'runtime_lengths': {'source_stft_frames': {
            'producer': 'source_extent', 'result': 'valid_extent',
            'capacity': PADDED_CAPACITY, 'allow_zero': False}},
        'native_entry': {
            'function': 'ck_kokoro_generator_stage1_join_oracle_input_bounded',
            'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                       {'c_type': 'size_t', 'name': 'arena_bytes'},
                       {'c_type': 'int32_t *', 'name': 'out_source_frames'}],
            'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
            'runtime_length_outputs': {
                'source_stft_frames': 'out_source_frames'}},
        'sequence': ['source_extent_component', *names],
        'block_types': blocks,
        'semantic_checkpoints': {
            'schema': 'cke.semantic_checkpoint_contract',
            'schema_version': 1, 'exports': checkpoints},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('stage-one join oracle-input circuit differs')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
