#!/usr/bin/env python3
"""Declare Kokoro's first connected source residual AdaIN, Snake and conv."""

import argparse
import json
from pathlib import Path

from build_kokoro_generated_source_stft_conv_circuit import build_circuit as source_circuit


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_source_residual_prefix_bounded.json'
CHANNELS = 256
CAPACITY = 2560
STYLE = 128
PREFIX = 'waveform_decoder.generator.noise_res.0'


def build_circuit(tokens=36):
    graph = source_circuit(tokens)
    graph['name'] = 'kokoro_source_residual_prefix_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_source_residual_prefix_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_source_residual_prefix=True,
        source_phase_numerical_compatibility=False,
        complete_waveform=False)
    buffers = {
        'decoder_style': [STYLE],
        'source_res0_style0': [2 * CHANNELS],
        'source_res0_norm0': [CHANNELS, CAPACITY],
        'source_res0_snake0': [CHANNELS, CAPACITY],
        'source_res0_conv0': [CHANNELS, CAPACITY],
    }
    for name, shape in buffers.items():
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
    style = {
        'id': 'source_res0_style0', 'op': 'linear_rows_checked',
        'kernel': 'linear_rows_checked_f32', 'returns_status': True,
        'weight_refs': {'weight': f'{PREFIX}.adain1.0.fc.weight',
                        'bias': f'{PREFIX}.adain1.0.fc.bias'},
        'params': {'M': 1, 'K': STYLE, 'N': 2 * CHANNELS,
            'call_constants': {'linear_input_elements': STYLE,
                'linear_input_stride': STYLE,
                'linear_weight_elements': 2 * CHANNELS * STYLE,
                'linear_weight_stride': STYLE,
                'linear_bias_elements': 2 * CHANNELS,
                'linear_output_elements': 2 * CHANNELS,
                'linear_output_stride': 2 * CHANNELS,
                'linear_rows': 1, 'linear_input_channels': STYLE,
                'linear_output_channels': 2 * CHANNELS}},
        'graph_slots': {'inputs': {'input': 'external:decoder_style'},
                        'outputs': {'output': 'source_res0_style0'}}}
    norm = {
        'id': 'source_res0_norm0', 'op': 'audio_adain_instance_norm',
        'kernel': 'audio_adain_instance_norm_f32', 'returns_status': True,
        'consumes_runtime_lengths': ['source_conv_frames'],
        'runtime_scalar_bindings': {'adain_frames': 'source_conv_frames'},
        'weight_refs': {'norm_weight': f'{PREFIX}.adain1.0.norm.weight',
                        'norm_bias': f'{PREFIX}.adain1.0.norm.bias'},
        'params': {'C': CHANNELS, 'T': CAPACITY,
            'normalization_epsilon': 1e-5, 'call_constants': {
                'adain_input_elements': CHANNELS * CAPACITY,
                'adain_input_stride': CAPACITY,
                'adain_norm_weight_elements': CHANNELS,
                'adain_norm_bias_elements': CHANNELS,
                'adain_style_affine_elements': 2 * CHANNELS,
                'adain_output_elements': CHANNELS * CAPACITY,
                'adain_output_stride': CAPACITY,
                'adain_channels': CHANNELS}},
        'graph_slots': {'inputs': {'input': 'source_conv0',
            'style_affine': 'source_res0_style0'},
            'outputs': {'output': 'source_res0_norm0'}}}
    snake = {
        'id': 'source_res0_snake0', 'op': 'audio_snake_strided_checked',
        'kernel': 'audio_snake_strided_f32_checked',
        'returns_status': True,
        'consumes_runtime_lengths': ['source_conv_frames'],
        'runtime_scalar_bindings': {'snake_frames': 'source_conv_frames'},
        'weight_refs': {'alpha': f'{PREFIX}.alpha1.0.channel'},
        'params': {'C': CHANNELS, 'T': CAPACITY,
            'call_constants': {'snake_input_elements': CHANNELS * CAPACITY,
                'snake_input_stride': CAPACITY,
                'snake_alpha_elements': CHANNELS,
                'snake_output_elements': CHANNELS * CAPACITY,
                'snake_output_stride': CAPACITY,
                'snake_channels': CHANNELS}},
        'graph_slots': {'inputs': {'input': 'source_res0_norm0'},
                        'outputs': {'output': 'source_res0_snake0'}}}
    convolution = {
        'id': 'source_res0_conv0', 'op': 'audio_conv1d_dilated_checked',
        'kernel': 'audio_conv1d_dilated_checked_channel_major_f32',
        'returns_status': True,
        'consumes_runtime_lengths': ['source_conv_frames'],
        'runtime_scalar_bindings': {
            'conv_input_frames': 'source_conv_frames',
            'conv_output_frames': 'source_conv_frames'},
        'weight_refs': {'weight': f'{PREFIX}.convs1.0.weight',
                        'bias': f'{PREFIX}.convs1.0.bias'},
        'params': {'I': CHANNELS, 'O': CHANNELS,
            'T': CAPACITY, 'K': 7, 'U': CAPACITY,
            'Q': CHANNELS * CAPACITY,
            'call_constants': {
                'conv_input_elements': CHANNELS * CAPACITY,
                'conv_input_stride': CAPACITY,
                'conv_weight_elements': CHANNELS * CHANNELS * 7,
                'conv_bias_elements': CHANNELS,
                'conv_output_elements': CHANNELS * CAPACITY,
                'conv_output_stride': CAPACITY,
                'conv_scratch_elements': CHANNELS * CAPACITY,
                'conv_input_channels': CHANNELS,
                'conv_output_channels': CHANNELS,
                'conv_kernel_size': 7, 'conv_stride': 1,
                'conv_padding': 3, 'conv_dilation': 1}},
        'graph_slots': {'inputs': {'input': 'source_res0_snake0'},
                        'outputs': {'output': 'source_res0_conv0'}}}
    graph['sequence'].append('source_residual_prefix')
    graph['block_types']['source_residual_prefix'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [style],
        'body': {'type': 'dense', 'ops': [norm, snake]},
        'footer': [convolution]}
    for item in (style, norm, snake, convolution):
        name = item['id']
        style_tensor = name == 'source_res0_style0'
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'header' if style_tensor else
                ('footer' if name == 'source_res0_conv0' else 'body'),
            'template_op_id': name, 'op': item['op'],
            'checkpoints': [{'id': f'kokoro.source.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'feature_contiguous' if style_tensor
                    else 'channel_major',
                'axis_names': ['channel'] if style_tensor
                    else ['channel', 'frame'],
                'storage_dtype': 'fp32'}]}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('source residual prefix circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
