#!/usr/bin/env python3
"""Declare Kokoro's first style-conditioned acoustic-decoder residual block."""

import argparse
import json
import math
from pathlib import Path

from build_kokoro_decoder_ingress_circuit import build_circuit as ingress_circuit
from build_kokoro_prosody_second_block_circuit import (
    activation, convolution, declared, norm)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_decoder_encode_bounded.json'
FRAMES = 128
STYLE = 128
INPUT_CHANNELS = 514
OUTPUT_CHANNELS = 1024
PREFIX = 'waveform_decoder.encode'


def style(name, prefix, channels):
    return declared(name, 'linear_rows_checked', 'linear_rows_checked_f32',
        {'input': 'external:decoder_style'}, name, [2 * channels],
        {'M': 1, 'K': STYLE, 'N': 2 * channels, 'call_constants': {
            'linear_input_elements': STYLE, 'linear_input_stride': STYLE,
            'linear_weight_elements': 2 * channels * STYLE,
            'linear_weight_stride': STYLE,
            'linear_bias_elements': 2 * channels,
            'linear_output_elements': 2 * channels,
            'linear_output_stride': 2 * channels, 'linear_rows': 1,
            'linear_input_channels': STYLE,
            'linear_output_channels': 2 * channels}},
        {'weight': f'{prefix}.fc.weight',
         'bias': f'{prefix}.fc.bias'})


def decoder_residual_block_ops(stem, source, prefix, input_channels,
                               output_channels, frame_capacity=FRAMES):
    """Declare one non-upsampling AdainResBlk1d; caller owns graph edges."""
    ops = []
    add = ops.append
    add(style(f'{stem}_norm0_style', f'{prefix}.norm1', input_channels))
    add(norm(f'{stem}_norm0', source,
             f'{stem}_norm0_style', f'{prefix}.norm1',
             input_channels, frame_capacity, 'expanded_frames'))
    add(activation(f'{stem}_act0', f'{stem}_norm0',
                   input_channels, frame_capacity, 'expanded_frames'))
    add(convolution(f'{stem}_conv0', f'{stem}_act0',
        f'{prefix}.conv1', input_channels, output_channels,
        frame_capacity, 3, 'expanded_frames'))
    add(style(f'{stem}_norm1_style', f'{prefix}.norm2', output_channels))
    add(norm(f'{stem}_norm1', f'{stem}_conv0',
             f'{stem}_norm1_style', f'{prefix}.norm2',
             output_channels, frame_capacity, 'expanded_frames'))
    add(activation(f'{stem}_act1', f'{stem}_norm1',
                   output_channels, frame_capacity, 'expanded_frames'))
    add(convolution(f'{stem}_conv1', f'{stem}_act1',
        f'{prefix}.conv2', output_channels, output_channels,
        frame_capacity, 3, 'expanded_frames'))
    add(convolution(f'{stem}_shortcut', source,
        f'{prefix}.conv1x1', input_channels, output_channels,
        frame_capacity, 1, 'expanded_frames', zero_bias=True))
    add(declared(f'{stem}_output',
        'audio_scaled_sum_strided_checked',
        'audio_scaled_sum_strided_f32_checked',
        {'left': f'{stem}_conv1',
         'right': f'{stem}_shortcut'},
        f'{stem}_output', [output_channels, frame_capacity],
        {'R': output_channels, 'C': frame_capacity,
         'scale': 1.0 / math.sqrt(2.0), 'call_constants': {
            'sum_left_elements': output_channels * frame_capacity,
            'sum_left_stride': frame_capacity,
            'sum_right_elements': output_channels * frame_capacity,
            'sum_right_stride': frame_capacity,
            'sum_output_elements': output_channels * frame_capacity,
            'sum_output_stride': frame_capacity,
            'sum_rows': output_channels}},
        None, ('expanded_frames',),
        {'sum_columns': 'expanded_frames'}))
    return ops


def add_decoder_block(graph, block_name, ops):
    for item, shape in ops:
        name = item['id']
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
        style_tensor = name.endswith('_style')
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'body', 'template_op_id': name,
            'op': item['op'],
            'checkpoints': [{'id': f'kokoro.decoder.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'feature_contiguous' if style_tensor else
                                  'channel_major',
                'axis_names': ['channel'] if style_tensor else
                              ['channel', 'frame'],
                'storage_dtype': 'fp32'}]}
    graph['sequence'].append(block_name)
    graph['block_types'][block_name] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [], 'body': {'type': 'dense',
                              'ops': [item for item, _ in ops]},
        'footer': []}


def build_circuit(tokens=36):
    graph = ingress_circuit(tokens)
    graph['name'] = 'kokoro_decoder_encode_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_decoder_encode_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_decoder_encode=True, complete_decoder=False,
        complete_waveform=False)
    graph['activation_buffers']['decoder_style'] = {'shape': [STYLE]}
    graph['activation_bindings']['decoder_style'] = 'decoder_style'
    add_decoder_block(graph, 'decoder_encode', decoder_residual_block_ops(
        'decoder_encode', 'decoder_joined', PREFIX,
        INPUT_CHANNELS, OUTPUT_CHANNELS))
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('decoder encode circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
