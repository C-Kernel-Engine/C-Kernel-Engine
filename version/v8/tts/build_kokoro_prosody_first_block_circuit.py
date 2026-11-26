#!/usr/bin/env python3
"""Extend generated F0/noise paths through their first residual blocks."""
import argparse
import json
import math
from pathlib import Path

from build_kokoro_prosody_norm_circuit import build_circuit as norm_circuit

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_prosody_first_block_bounded.json'
CHANNELS = 512
CAPACITY = 128
STYLE = 128


def operation(ident, op, kernel, source, output, params, weights=None,
              extra_inputs=None, runtime_args=None):
    inputs = {'input': source} if source is not None else {}
    if extra_inputs:
        inputs.update(extra_inputs)
    result = {'id': ident, 'op': op, 'kernel': kernel,
        'returns_status': True, 'params': params,
        'graph_slots': {'inputs': inputs, 'outputs': {'output': output}}}
    if weights:
        result['weight_refs'] = weights
    if runtime_args:
        result['consumes_runtime_lengths'] = ['expanded_frames']
        result['runtime_scalar_bindings'] = {
            name: 'expanded_frames' for name in runtime_args}
    return result


def branch_ops(branch, capacity=CAPACITY):
    stem = branch.lower()
    prefix = f'duration_prosody.{branch}.0'
    source = f'{stem}_norm0_output'
    ops = []
    def add(ident, op, kernel, input_name, params, weights=None,
            extra_inputs=None, runtime_args=None):
        ops.append(operation(ident, op, kernel, input_name, ident, params,
            weights, extra_inputs, runtime_args))
    def activation(name, input_name):
        add(name, 'leaky_relu_strided_checked',
            'leaky_relu_strided_f32_checked', input_name,
            {'R': CHANNELS, 'C': capacity, 'negative_slope': .2,
             'call_constants': {'leaky_input_elements': CHANNELS * capacity,
                'leaky_input_stride': capacity,
                'leaky_output_elements': CHANNELS * capacity,
                'leaky_output_stride': capacity,
                'leaky_rows': CHANNELS}},
            runtime_args=['leaky_columns'])
    def convolution(name, input_name, index):
        weight_prefix = f'{prefix}.conv{index}'
        add(name, 'audio_conv1d_checked',
            'audio_conv1d_checked_channel_major_f32', input_name,
            {'I': CHANNELS, 'O': CHANNELS, 'T': capacity, 'K': 3,
             'U': capacity, 'Q': CHANNELS * capacity,
             'call_constants': {'conv_input_elements': CHANNELS * capacity,
                'conv_input_stride': capacity,
                'conv_weight_elements': CHANNELS * CHANNELS * 3,
                'conv_bias_elements': CHANNELS,
                'conv_output_elements': CHANNELS * capacity,
                'conv_output_stride': capacity,
                'conv_scratch_elements': CHANNELS * capacity,
                'conv_input_channels': CHANNELS,
                'conv_output_channels': CHANNELS,
                'conv_kernel_size': 3, 'conv_stride': 1,
                'conv_padding': 1}},
            {'weight': f'{weight_prefix}.weight',
             'bias': f'{weight_prefix}.bias'},
            runtime_args=['conv_input_frames', 'conv_output_frames'])
    activation(f'{stem}_act0', source)
    convolution(f'{stem}_conv0', f'{stem}_act0', 1)
    norm2 = f'{prefix}.norm2'
    add(f'{stem}_norm1_style', 'linear_rows_checked',
        'linear_rows_checked_f32', 'external:predictor_style',
        {'M': 1, 'K': STYLE, 'N': 2 * CHANNELS,
         'call_constants': {'linear_input_elements': STYLE,
            'linear_input_stride': STYLE,
            'linear_weight_elements': 2 * CHANNELS * STYLE,
            'linear_weight_stride': STYLE,
            'linear_bias_elements': 2 * CHANNELS,
            'linear_output_elements': 2 * CHANNELS,
            'linear_output_stride': 2 * CHANNELS,
            'linear_rows': 1, 'linear_input_channels': STYLE,
            'linear_output_channels': 2 * CHANNELS}},
        {'weight': f'{norm2}.fc.weight', 'bias': f'{norm2}.fc.bias'})
    add(f'{stem}_norm1_output', 'audio_adain_instance_norm',
        'audio_adain_instance_norm_f32', f'{stem}_conv0',
        {'C': CHANNELS, 'T': capacity, 'normalization_epsilon': 1e-5,
         'call_constants': {'adain_input_elements': CHANNELS * capacity,
            'adain_input_stride': capacity,
            'adain_norm_weight_elements': CHANNELS,
            'adain_norm_bias_elements': CHANNELS,
            'adain_style_affine_elements': 2 * CHANNELS,
            'adain_output_elements': CHANNELS * capacity,
            'adain_output_stride': capacity,
            'adain_channels': CHANNELS}},
        {'norm_weight': f'{norm2}.norm.weight',
         'norm_bias': f'{norm2}.norm.bias'},
        {'style_affine': f'{stem}_norm1_style'},
        runtime_args=['adain_frames'])
    activation(f'{stem}_act1', f'{stem}_norm1_output')
    convolution(f'{stem}_conv1', f'{stem}_act1', 2)
    # Kokoro's block scales the entire residual sum, unlike the existing
    # audio_scaled_residual_add provider that scales only its branch operand.
    ops.append({'id': f'{stem}_block0_output',
        'op': 'audio_scaled_sum_strided_checked',
        'kernel': 'audio_scaled_sum_strided_f32_checked',
        'returns_status': True,
        'consumes_runtime_lengths': ['expanded_frames'],
        'runtime_scalar_bindings': {'sum_columns': 'expanded_frames'},
        'params': {'R': CHANNELS, 'C': capacity,
            'scale': 1.0 / math.sqrt(2.0), 'call_constants': {
                'sum_left_elements': CHANNELS * capacity,
                'sum_left_stride': capacity,
                'sum_right_elements': CHANNELS * capacity,
                'sum_right_stride': capacity,
                'sum_output_elements': CHANNELS * capacity,
                'sum_output_stride': capacity, 'sum_rows': CHANNELS}},
        'graph_slots': {'inputs': {'left': 'prosody_shared_channel',
            'right': f'{stem}_conv1'},
            'outputs': {'output': f'{stem}_block0_output'}}})
    return ops


def build_circuit(tokens=36):
    graph = norm_circuit(tokens)
    graph['name'] = 'kokoro_prosody_first_block_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_prosody_first_block_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_first_f0_noise_residual_blocks=True,
        complete_prosody=False, complete_waveform=False)
    ops = []
    for branch in ('F0', 'N'):
        ops.extend(branch_ops(branch))
    for item in ops:
        name = item['id']
        graph['activation_buffers'][name] = {'shape':
            [2 * CHANNELS] if name.endswith('_style') else
            [CHANNELS, CAPACITY]}
        graph['activation_bindings'][name] = name
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'body', 'template_op_id': name, 'op': item['op'],
            'checkpoints': [{'id': f'kokoro.prosody.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': ('feature_contiguous' if
                    name.endswith('_style') else 'channel_major'),
                'axis_names': (['channel'] if name.endswith('_style') else
                    ['channel', 'frame']), 'storage_dtype': 'fp32'}]}
    graph['sequence'].append('prosody_first_block')
    graph['block_types']['prosody_first_block'] = {
        'sequence': ['header', 'body', 'footer'], 'header': [],
        'body': {'type': 'dense', 'ops': ops}, 'footer': []}
    graph['required_numerical_contracts']['audio_scaled_sum_strided_checked'] = {
        'op': 'audio_scaled_sum_strided_checked',
        'template_ops': ['audio_scaled_sum_strided_checked'],
        'phases': {'prefill': {'contract_id':
            'audio_scaled_sum_strided_sum_then_scale_fp32',
            'validation': 'validated',
            'evidence': 'tests/test_v8_audio_scaled_sum_strided.py'}},
        'checkpoint': {'id': 'kokoro.prosody.f0_block0_output',
            'producer': 'f0_block0_output',
            'logical_layout': 'channel_major',
            'axis_names': ['channel', 'frame']}}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('first block circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
