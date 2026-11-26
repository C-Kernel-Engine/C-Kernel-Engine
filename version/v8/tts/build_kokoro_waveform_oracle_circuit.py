#!/usr/bin/env python3
"""Isolate the generated spectral tail with a pinned final-generator input."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_waveform_circuit import (
    ROOT, PADDED_CAPACITY, WAVEFORM_CAPACITY,
    build_circuit as waveform_circuit,
)


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_waveform_oracle_input_bounded.json'


def build_circuit():
    connected = waveform_circuit()
    blocks = {name: copy.deepcopy(connected['block_types'][name]) for name in (
        'generator_spectral_projection', 'generator_inverse_stft')}
    conv = blocks['generator_spectral_projection']['body']['ops'][0]
    conv['graph_slots']['inputs']['input'] = \
        'external:generator_stage1_final_activation'
    for block in blocks.values():
        for op in block['header'] + block['body']['ops'] + block['footer']:
            op['consumes_runtime_lengths'] = [
                'head_frames' if name == 'source_stft_frames' else
                'head_samples' if name == 'source_samples' else name
                for name in op.get('consumes_runtime_lengths', [])]
            op['runtime_scalar_bindings'] = {
                scalar: 'head_frames' if name == 'source_stft_frames' else
                'head_samples' if name == 'source_samples' else name
                for scalar, name in op.get('runtime_scalar_bindings', {}).items()}
    extents = []
    for ident, source, length, capacity in (
        ('head_frame_extent', 'head_frame_count', 'head_frames',
         PADDED_CAPACITY),
        ('head_sample_extent', 'head_sample_count', 'head_samples',
         WAVEFORM_CAPACITY),
    ):
        extents.append({
            'id': ident, 'op': 'runtime_extent_sum',
            'kernel': 'runtime_extent_sum_i32', 'returns_status': True,
            'produces_runtime_lengths': {length: 'valid_extent'},
            'params': {'T': 1},
            'graph_slots': {'inputs': {'values': f'external:{source}'},
                'outputs': {'valid_extent': f'{ident}_value'}}})
    blocks['head_extents'] = {
        'sequence': ['header', 'body', 'footer'], 'header': extents,
        'body': {'type': 'dense', 'ops': []}, 'footer': []}
    buffers = {
        'generator_stage1_final_activation': {
            'shape': [128, PADDED_CAPACITY]},
        'head_frame_count': {'shape': [1]},
        'head_sample_count': {'shape': [1]},
        'head_frame_extent_value': {'shape': [1]},
        'head_sample_extent_value': {'shape': [1]},
    }
    checkpoints = {}
    for block in blocks.values():
        for op in block['header'] + block['body']['ops'] + block['footer']:
            if op['id'] in ('head_frame_extent', 'head_sample_extent'):
                continue
            for tensor in op['graph_slots']['outputs'].values():
                buffers[tensor] = copy.deepcopy(
                    connected['activation_buffers'][tensor])
            checkpoints[op['id']] = copy.deepcopy(
                connected['semantic_checkpoints']['exports'][op['id']])
    return {
        'version': 3,
        'name': 'kokoro_waveform_oracle_input_bounded',
        'family': 'bounded_graph', 'checked_native_entry': True,
        'contract': {'runtime_invariants': {
            'inference_only': True,
            'oracle_supplied_final_generator_features': True,
            'complete_model': False}},
        'activation_buffers': buffers,
        'activation_bindings': {name: name for name in buffers},
        'runtime_constants': {'value_elements': 1, 'phoneme_count': 1,
                              'max_duration': WAVEFORM_CAPACITY,
                              'expanded_capacity': WAVEFORM_CAPACITY},
        'runtime_lengths': {
            'head_frames': {'producer': 'head_frame_extent',
                'result': 'valid_extent', 'capacity': PADDED_CAPACITY,
                'allow_zero': False},
            'head_samples': {'producer': 'head_sample_extent',
                'result': 'valid_extent', 'capacity': WAVEFORM_CAPACITY,
                'allow_zero': False}},
        'native_entry': {
            'function': 'ck_kokoro_waveform_oracle_input_bounded',
            'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                       {'c_type': 'size_t', 'name': 'arena_bytes'},
                       {'c_type': 'int32_t *', 'name': 'out_frames'},
                       {'c_type': 'int32_t *', 'name': 'out_samples'}],
            'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
            'runtime_length_outputs': {
                'head_frames': 'out_frames',
                'head_samples': 'out_samples'}},
        'sequence': ['head_extents', 'generator_spectral_projection',
                     'generator_inverse_stft'],
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
            raise SystemExit('oracle-input waveform circuit differs')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
