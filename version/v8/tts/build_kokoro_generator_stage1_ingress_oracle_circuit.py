#!/usr/bin/env python3
"""Isolate the second generator upsample with direct stage-zero pool input."""

import argparse
import json
from pathlib import Path

from build_kokoro_generator_stage1_ingress_circuit import (
    ROOT, INPUT_CHANNELS, INPUT_CAPACITY,
    add_ingress,
)


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generator_stage1_ingress_oracle_input_bounded.json'


def build_circuit():
    graph = {
        'version': 3,
        'name': 'kokoro_generator_stage1_ingress_oracle_input_bounded',
        'family': 'bounded_graph', 'checked_native_entry': True,
        'contract': {'runtime_invariants': {
            'inference_only': True, 'oracle_supplied_stage0_pool': True,
            'complete_waveform': False}},
        'activation_buffers': {
            'generator_stage0_residual_mean':
                {'shape': [INPUT_CHANNELS, INPUT_CAPACITY]},
            'stage0_frame_count': {'shape': [1]},
            'stage0_extent_value': {'shape': [1]}},
        'activation_bindings': {
            'generator_stage0_residual_mean': 'generator_stage0_residual_mean',
            'stage0_frame_count': 'stage0_frame_count',
            'stage0_extent_value': 'stage0_extent_value'},
        'runtime_constants': {'value_elements': 1, 'phoneme_count': 1,
                              'max_duration': INPUT_CAPACITY,
                              'expanded_capacity': INPUT_CAPACITY},
        'runtime_lengths': {'generator_stage0_frames': {
            'producer': 'stage0_extent', 'result': 'valid_extent',
            'capacity': INPUT_CAPACITY, 'allow_zero': False}},
        'native_entry': {
            'function': 'ck_kokoro_generator_stage1_ingress_oracle_input_bounded',
            'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                       {'c_type': 'size_t', 'name': 'arena_bytes'},
                       {'c_type': 'int32_t *', 'name': 'out_stage0_frames'}],
            'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
            'runtime_length_outputs': {
                'generator_stage0_frames': 'out_stage0_frames'}},
        'sequence': ['stage0_extent_component'],
        'block_types': {'stage0_extent_component': {
            'sequence': ['header', 'body', 'footer'],
            'header': [{
                'id': 'stage0_extent', 'op': 'runtime_extent_sum',
                'kernel': 'runtime_extent_sum_i32', 'returns_status': True,
                'produces_runtime_lengths': {
                    'generator_stage0_frames': 'valid_extent'},
                'params': {'T': 1},
                'graph_slots': {'inputs': {
                    'values': 'external:stage0_frame_count'},
                    'outputs': {'valid_extent': 'stage0_extent_value'}}}],
            'body': {'type': 'dense', 'ops': []}, 'footer': []}},
        'semantic_checkpoints': {
            'schema': 'cke.semantic_checkpoint_contract',
            'schema_version': 1, 'exports': {}}}
    graph = add_ingress(graph)
    graph['block_types']['generator_stage1_ingress']['body']['ops'][0][
        'graph_slots']['inputs']['input'] = \
        'external:generator_stage0_residual_mean'
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('stage1 oracle-input circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
