#!/usr/bin/env python3
"""Isolate first-stage generator blocks with the direct pinned join input."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_generator_stage0_pool_circuit import (
    build_circuit as connected_circuit, ROOT, CAPACITY,
)


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generator_stage0_pool_oracle_input_bounded.json'


def build_circuit():
    connected = connected_circuit()
    names = [f'generator_main_residual_pair{pair}' for pair in range(3)]
    names += [f'generator_main_resblock{block}_pair{pair}'
              for block in (1, 2) for pair in range(3)]
    names += ['generator_stage0_pool']
    blocks = {name: copy.deepcopy(connected['block_types'][name]) for name in names}
    for name in (names[0], names[3], names[6]):
        first = blocks[name]
        first['body']['ops'][0]['graph_slots']['inputs']['input'] = \
            'external:generator_stage0_join'
        first['footer'][0]['graph_slots']['inputs']['right'] = \
            'external:generator_stage0_join'
    blocks[names[0]]['header'].insert(0, {
        'id': 'main_res_extent', 'op': 'runtime_extent_sum',
        'kernel': 'runtime_extent_sum_i32', 'returns_status': True,
        'produces_runtime_lengths': {'generator_stage0_frames': 'valid_extent'},
        'params': {'T': 1},
        'graph_slots': {'inputs': {'values': 'external:main_res_frame_count'},
                        'outputs': {'valid_extent': 'main_res_extent_value'}}})
    produced = [op['id'] for block in blocks.values()
                for op in block['header'] + block['body']['ops'] + block['footer']
                if op['id'] != 'main_res_extent']
    buffers = ('decoder_style', 'generator_stage0_join', *produced)
    graph = {
        'version': 3, 'name': 'kokoro_generator_stage0_pool_oracle_input_bounded',
        'family': 'bounded_graph', 'checked_native_entry': True,
        'contract': {'runtime_invariants': {
            'inference_only': True, 'oracle_supplied_generator_join': True,
            'complete_waveform': False}},
        'activation_buffers': {key: connected['activation_buffers'][key]
                               for key in buffers},
        'activation_bindings': {key: key for key in buffers},
        'runtime_constants': {'value_elements': 1, 'phoneme_count': 1,
                              'max_duration': CAPACITY,
                              'expanded_capacity': CAPACITY},
        'runtime_lengths': {'generator_stage0_frames': {
            'producer': 'main_res_extent', 'result': 'valid_extent',
            'capacity': CAPACITY, 'allow_zero': False}},
        'native_entry': {
            'function': 'ck_kokoro_generator_stage0_pool_oracle_input_bounded',
            'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                       {'c_type': 'size_t', 'name': 'arena_bytes'},
                       {'c_type': 'int32_t *', 'name': 'out_frames'}],
            'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
            'runtime_length_outputs': {'generator_stage0_frames': 'out_frames'}},
        'sequence': names, 'block_types': blocks,
        'semantic_checkpoints': {'schema': 'cke.semantic_checkpoint_contract',
            'schema_version': 1, 'exports': {
                key: connected['semantic_checkpoints']['exports'][key]
                for key in produced}},
    }
    for key in ('main_res_frame_count', 'main_res_extent_value'):
        graph['activation_buffers'][key] = {'shape': [1]}
        graph['activation_bindings'][key] = key
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('stage0 pool oracle-input circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
