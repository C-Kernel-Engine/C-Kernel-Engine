#!/usr/bin/env python3
"""Isolate second-stage main residual pool with pinned join input."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_generator_stage1_pool_circuit import (
    ROOT, CHANNELS, PADDED_CAPACITY, build_circuit as connected_circuit,
)


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generator_stage1_pool_oracle_input_bounded.json'


def build_circuit():
    connected = connected_circuit()
    names = [f'generator_stage1_main_resblock{block}_pair{pair}'
             for block in (3, 4, 5) for pair in range(3)]
    names += ['generator_stage1_pool', 'generator_stage1_final_activation']
    blocks = {name: copy.deepcopy(connected['block_types'][name])
              for name in names}
    for block in (3, 4, 5):
        first = blocks[f'generator_stage1_main_resblock{block}_pair0']
        first['body']['ops'][0]['graph_slots']['inputs']['input'] = \
            'external:generator_stage1_join'
        first['footer'][0]['graph_slots']['inputs']['right'] = \
            'external:generator_stage1_join'
    first_block = blocks[names[0]]
    first_block['header'].insert(0, {
        'id': 'main_res_extent', 'op': 'runtime_extent_sum',
        'kernel': 'runtime_extent_sum_i32', 'returns_status': True,
        'produces_runtime_lengths': {
            'generator_stage1_frames': 'valid_extent'},
        'params': {'T': 1},
        'graph_slots': {'inputs': {
            'values': 'external:main_res_frame_count'},
            'outputs': {'valid_extent': 'main_res_extent_value'}}})
    buffers = {
        'decoder_style': {'shape': [128]},
        'generator_stage1_join': {
            'shape': [CHANNELS, PADDED_CAPACITY]},
        'main_res_frame_count': {'shape': [1]},
        'main_res_extent_value': {'shape': [1]},
    }
    checkpoints = {}
    for block in blocks.values():
        for op in block['header'] + block['body']['ops'] + block['footer']:
            if op['id'] == 'main_res_extent':
                continue
            buffers[op['id']] = copy.deepcopy(
                connected['activation_buffers'][op['id']])
            checkpoints[op['id']] = copy.deepcopy(
                connected['semantic_checkpoints']['exports'][op['id']])
    return {
        'version': 3,
        'name': 'kokoro_generator_stage1_pool_oracle_input_bounded',
        'family': 'bounded_graph', 'checked_native_entry': True,
        'contract': {'runtime_invariants': {
            'inference_only': True, 'oracle_supplied_stage1_join': True,
            'complete_waveform': False}},
        'activation_buffers': buffers,
        'activation_bindings': {name: name for name in buffers},
        'runtime_constants': {'value_elements': 1, 'phoneme_count': 1,
                              'max_duration': PADDED_CAPACITY,
                              'expanded_capacity': PADDED_CAPACITY},
        'runtime_lengths': {'generator_stage1_frames': {
            'producer': 'main_res_extent', 'result': 'valid_extent',
            'capacity': PADDED_CAPACITY, 'allow_zero': False}},
        'native_entry': {
            'function': 'ck_kokoro_generator_stage1_pool_oracle_input_bounded',
            'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                       {'c_type': 'size_t', 'name': 'arena_bytes'},
                       {'c_type': 'int32_t *', 'name': 'out_frames'}],
            'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
            'runtime_length_outputs': {
                'generator_stage1_frames': 'out_frames'}},
        'sequence': names,
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
            raise SystemExit('stage-one pool oracle-input circuit differs')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
