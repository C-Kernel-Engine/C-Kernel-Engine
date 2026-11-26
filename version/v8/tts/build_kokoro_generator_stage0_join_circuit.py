#!/usr/bin/env python3
"""Join Kokoro's first generated main upsample and source residual block."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_generator_stage0_circuit import build_circuit as main_circuit
from build_kokoro_source_residual_block0_circuit import build_circuit as source_circuit


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generator_stage0_join_bounded.json'
CHANNELS = 256
CAPACITY = 2560


def build_circuit(tokens=36):
    graph = main_circuit(tokens)
    source = source_circuit(tokens)
    for name in graph['sequence']:
        if name in source['block_types'] and graph['block_types'][name] != source['block_types'][name]:
            raise ValueError(f'shared Kokoro block differs: {name}')
    for name, shape in source['activation_buffers'].items():
        if name in graph['activation_buffers'] and graph['activation_buffers'][name] != shape:
            raise ValueError(f'shared Kokoro buffer differs: {name}')
    for name, contract in source['runtime_lengths'].items():
        if name in graph['runtime_lengths'] and graph['runtime_lengths'][name] != contract:
            raise ValueError(f'shared Kokoro runtime length differs: {name}')
    graph['name'] = 'kokoro_generator_stage0_join_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_generator_stage0_join_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_generator_stage0_source_join=True,
        source_phase_numerical_compatibility=False,
        captured_gaussian_input=True, native_rng=False,
        complete_waveform=False)
    for name in ('source_samples', 'source_stft_frames'):
        graph['runtime_lengths'][name] = source['runtime_lengths'][name]
    # Both branches are 10 * checked upsampled_frames. Keep one producer and
    # bind the source convolution and every residual consumer to its result.
    for name in source['sequence']:
        if name in graph['block_types']:
            continue
        block = copy.deepcopy(source['block_types'][name])
        if name == 'source_stft_conv':
            block['header'] = [op for op in block['header']
                               if op['id'] != 'source_conv_extent']
        for op in block['header'] + block['body']['ops'] + block['footer']:
            op['consumes_runtime_lengths'] = [
                'generator_stage0_frames' if length == 'source_conv_frames'
                else length for length in op.get('consumes_runtime_lengths', [])]
            for scalar, length in op.get('runtime_scalar_bindings', {}).items():
                if length == 'source_conv_frames':
                    op['runtime_scalar_bindings'][scalar] = 'generator_stage0_frames'
        graph['sequence'].append(name)
        graph['block_types'][name] = block
    for name, shape in source['activation_buffers'].items():
        if name != 'source_conv_extent_value':
            graph['activation_buffers'].setdefault(name, shape)
            graph['activation_bindings'].setdefault(
                name, source['activation_bindings'][name])
    for name, checkpoint in source['semantic_checkpoints']['exports'].items():
        if name in graph['semantic_checkpoints']['exports']:
            if graph['semantic_checkpoints']['exports'][name] != checkpoint:
                raise ValueError(f'shared Kokoro checkpoint differs: {name}')
        else:
            graph['semantic_checkpoints']['exports'][name] = checkpoint
    for name, argument in (('source_samples', 'out_source_samples'),
                           ('source_stft_frames', 'out_stft_frames')):
        graph['native_entry']['params'].append(
            {'c_type': 'int32_t *', 'name': argument})
        graph['native_entry']['runtime_length_outputs'][name] = argument
    graph['activation_buffers']['generator_stage0_join'] = {
        'shape': [CHANNELS, CAPACITY]}
    graph['activation_bindings']['generator_stage0_join'] = 'generator_stage0_join'
    join = {
        'id': 'generator_stage0_join',
        'op': 'audio_scaled_sum_strided_checked',
        'kernel': 'audio_scaled_sum_strided_f32_checked',
        'returns_status': True,
        'consumes_runtime_lengths': ['generator_stage0_frames'],
        'runtime_scalar_bindings': {'sum_columns': 'generator_stage0_frames'},
        'params': {'R': CHANNELS, 'C': CAPACITY, 'scale': 1.0,
            'call_constants': {
                'sum_left_elements': CHANNELS * CAPACITY,
                'sum_left_stride': CAPACITY,
                'sum_right_elements': CHANNELS * CAPACITY,
                'sum_right_stride': CAPACITY,
                'sum_output_elements': CHANNELS * CAPACITY,
                'sum_output_stride': CAPACITY,
                'sum_rows': CHANNELS}},
        'graph_slots': {'inputs': {
            'left': 'generator_stage0_upsample',
            'right': 'source_res_pair2_output'},
            'outputs': {'output': 'generator_stage0_join'}}}
    graph['sequence'].append('generator_stage0_join')
    graph['block_types']['generator_stage0_join'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [], 'body': {'type': 'dense', 'ops': [join]}, 'footer': []}
    graph['semantic_checkpoints']['exports']['generator_stage0_join'] = {
        'section': 'body', 'template_op_id': 'generator_stage0_join',
        'op': join['op'], 'checkpoints': [{
            'id': 'kokoro.generator.stage0_join',
            'producer': 'generator_stage0_join',
            'tensor': 'generator_stage0_join',
            'logical_layout': 'channel_major',
            'axis_names': ['channel', 'frame'], 'storage_dtype': 'fp32'}]}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('generator stage0 join circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
