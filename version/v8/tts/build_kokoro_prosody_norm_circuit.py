#!/usr/bin/env python3
"""Connect both first F0/noise AdaIN stages to generated shared prosody.

This is a bounded branch-prefix circuit. Convolution, residual blocks,
upsampling, final F0/N projection, and waveform remain separate work.
"""
import argparse
import json
from pathlib import Path

from build_kokoro_prosody_shared_circuit import build_circuit as shared_circuit

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_prosody_norm_bounded.json'
CAPACITY = 128
CHANNELS = 512
STYLE = 128


def build_circuit(tokens=36):
    graph = shared_circuit(tokens)
    graph['name'] = 'kokoro_prosody_norm_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_prosody_norm_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_first_f0_noise_norm=True, complete_prosody=False,
        complete_waveform=False)
    graph['activation_buffers']['prosody_shared_channel'] = {
        'shape': [CHANNELS, CAPACITY]}
    graph['activation_bindings']['prosody_shared_channel'] = \
        'prosody_shared_channel'
    transpose = {'id': 'prosody_shared_channel_transpose',
        'op': 'transpose_strided_checked',
        'kernel': 'transpose_strided_f32_checked', 'returns_status': True,
        'consumes_runtime_lengths': ['expanded_frames'],
        'runtime_scalar_bindings': {'transpose_rows': 'expanded_frames'},
        'params': {'R': CAPACITY, 'C': CHANNELS, 'call_constants': {
            'transpose_input_elements': CAPACITY * CHANNELS,
            'transpose_input_stride': CHANNELS,
            'transpose_output_elements': CAPACITY * CHANNELS,
            'transpose_output_stride': CAPACITY,
            'transpose_columns': CHANNELS}},
        'graph_slots': {'inputs': {'input': 'prosody_shared_features'},
                        'outputs': {'output': 'prosody_shared_channel'}}}
    body = []
    footer = []
    for branch in ('F0', 'N'):
        stem = branch.lower()
        prefix = f'duration_prosody.{branch}.0.norm1'
        style_name = f'{stem}_norm0_style'
        output_name = f'{stem}_norm0_output'
        graph['activation_buffers'][style_name] = {'shape': [2 * CHANNELS]}
        graph['activation_buffers'][output_name] = {
            'shape': [CHANNELS, CAPACITY]}
        graph['activation_bindings'][style_name] = style_name
        graph['activation_bindings'][output_name] = output_name
        projection = {'id': style_name, 'op': 'linear_rows_checked',
            'kernel': 'linear_rows_checked_f32', 'returns_status': True,
            'weight_refs': {'weight': f'{prefix}.fc.weight',
                            'bias': f'{prefix}.fc.bias'},
            'params': {'M': 1, 'K': STYLE, 'N': 2 * CHANNELS,
                'call_constants': {
                'linear_input_elements': STYLE,
                'linear_input_stride': STYLE,
                'linear_weight_elements': 2 * CHANNELS * STYLE,
                'linear_weight_stride': STYLE,
                'linear_bias_elements': 2 * CHANNELS,
                'linear_output_elements': 2 * CHANNELS,
                'linear_output_stride': 2 * CHANNELS,
                'linear_rows': 1, 'linear_input_channels': STYLE,
                'linear_output_channels': 2 * CHANNELS}},
            'graph_slots': {'inputs': {'input': 'external:predictor_style'},
                            'outputs': {'output': style_name}}}
        normalization = {'id': output_name,
            'op': 'audio_adain_instance_norm',
            'kernel': 'audio_adain_instance_norm_f32',
            'returns_status': True,
            'consumes_runtime_lengths': ['expanded_frames'],
            'runtime_scalar_bindings': {'adain_frames': 'expanded_frames'},
            'weight_refs': {
                'norm_weight': f'{prefix}.norm.weight',
                'norm_bias': f'{prefix}.norm.bias'},
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
            'graph_slots': {'inputs': {
                'input': 'prosody_shared_channel',
                'style_affine': style_name},
                'outputs': {'output': output_name}}}
        body.append(projection)
        footer.append(normalization)
        for operation, tensor, layout, axes, section in (
                (projection, style_name, 'feature_contiguous',
                 ['channel'], 'body'),
                (normalization, output_name, 'channel_major',
                 ['channel', 'frame'], 'footer')):
            graph['semantic_checkpoints']['exports'][operation['id']] = {
                'section': section, 'template_op_id': operation['id'],
                'op': operation['op'], 'checkpoints': [{
                    'id': f'kokoro.prosody.{operation["id"]}',
                    'producer': operation['id'], 'tensor': tensor,
                    'logical_layout': layout, 'axis_names': axes,
                    'storage_dtype': 'fp32'}]}
    graph['sequence'].append('prosody_norm')
    graph['block_types']['prosody_norm'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [transpose], 'body': {'type': 'dense', 'ops': body},
        'footer': footer}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('prosody norm circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
