#!/usr/bin/env python3
"""Declare Kokoro's complete acoustic decoder before waveform generation."""

import argparse
import json
import math
from pathlib import Path

from build_kokoro_decoder_encode_circuit import (
    add_decoder_block, build_circuit as encode_circuit,
    decoder_residual_block_ops, style)
from build_kokoro_decoder_ingress_circuit import concat
from build_kokoro_prosody_second_block_circuit import (
    activation, convolution, declared, norm)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_decoder_complete_bounded.json'
INPUT_FRAMES = 128
OUTPUT_FRAMES = 256
INPUT_CHANNELS = 1090
OUTPUT_CHANNELS = 512


def decoder_join(index, source, frame_capacity=INPUT_FRAMES):
    stem = f'decoder_decode{index}'
    return [
        (concat(f'{stem}_join_asr', source,
                'decoder_asr_res', 1024, 64, frame_capacity),
         [1088, frame_capacity]),
        (concat(f'{stem}_join_f0', f'{stem}_join_asr',
                'decoder_f0_downsample', 1088, 1, frame_capacity),
         [1089, frame_capacity]),
        (concat(f'{stem}_input', f'{stem}_join_f0',
                'decoder_n_downsample', 1089, 1, frame_capacity),
         [1090, frame_capacity]),
    ]


def upsampling_block_ops(frame_capacity=INPUT_FRAMES,
                         upsampled_capacity=OUTPUT_FRAMES):
    stem = 'decoder_decode3'
    source = f'{stem}_input'
    prefix = 'waveform_decoder.decode.3'
    ops = []
    add = ops.append
    add(style(f'{stem}_norm0_style', f'{prefix}.norm1', INPUT_CHANNELS))
    add(norm(f'{stem}_norm0', source, f'{stem}_norm0_style',
        f'{prefix}.norm1', INPUT_CHANNELS, frame_capacity, 'expanded_frames'))
    add(activation(f'{stem}_act0', f'{stem}_norm0',
        INPUT_CHANNELS, frame_capacity, 'expanded_frames'))
    add(declared(f'{stem}_pool',
        'audio_conv_transpose1d_depthwise_checked',
        'audio_conv_transpose1d_depthwise_channel_major_f32_checked',
        {'input': f'{stem}_act0'}, f'{stem}_pool',
        [INPUT_CHANNELS, upsampled_capacity],
        {'C': INPUT_CHANNELS, 'T': frame_capacity, 'K': 3,
         'U': upsampled_capacity, 'Q': INPUT_CHANNELS * upsampled_capacity,
         'call_constants': {
            'deconv_input_elements': INPUT_CHANNELS * frame_capacity,
            'deconv_input_stride': frame_capacity,
            'deconv_weight_elements': INPUT_CHANNELS * 3,
            'deconv_bias_elements': INPUT_CHANNELS,
            'deconv_output_elements': INPUT_CHANNELS * upsampled_capacity,
            'deconv_output_stride': upsampled_capacity,
            'deconv_scratch_elements': INPUT_CHANNELS * upsampled_capacity,
            'deconv_channels': INPUT_CHANNELS, 'deconv_kernel_size': 3,
            'deconv_stride': 2, 'deconv_padding': 1,
            'deconv_output_padding': 1}},
        {'weight': f'{prefix}.pool.weight',
         'bias': f'{prefix}.pool.bias'},
        ('expanded_frames', 'upsampled_frames'),
        {'deconv_input_frames': 'expanded_frames',
         'deconv_output_frames': 'upsampled_frames'}))
    add(convolution(f'{stem}_conv0', f'{stem}_pool',
        f'{prefix}.conv1', INPUT_CHANNELS, OUTPUT_CHANNELS,
        upsampled_capacity, 3, 'upsampled_frames'))
    add(style(f'{stem}_norm1_style', f'{prefix}.norm2', OUTPUT_CHANNELS))
    add(norm(f'{stem}_norm1', f'{stem}_conv0', f'{stem}_norm1_style',
        f'{prefix}.norm2', OUTPUT_CHANNELS, upsampled_capacity,
        'upsampled_frames'))
    add(activation(f'{stem}_act1', f'{stem}_norm1',
        OUTPUT_CHANNELS, upsampled_capacity, 'upsampled_frames'))
    add(convolution(f'{stem}_conv1', f'{stem}_act1',
        f'{prefix}.conv2', OUTPUT_CHANNELS, OUTPUT_CHANNELS,
        upsampled_capacity, 3, 'upsampled_frames'))
    add(declared(f'{stem}_shortcut_upsample',
        'audio_upsample_nearest_checked',
        'audio_upsample_nearest_channel_major_f32_checked',
        {'input': source}, f'{stem}_shortcut_upsample',
        [INPUT_CHANNELS, upsampled_capacity],
        {'C': INPUT_CHANNELS, 'T': frame_capacity, 'U': upsampled_capacity,
         'call_constants': {
            'nearest_input_elements': INPUT_CHANNELS * frame_capacity,
            'nearest_input_stride': frame_capacity,
            'nearest_output_elements': INPUT_CHANNELS * upsampled_capacity,
            'nearest_output_stride': upsampled_capacity,
            'nearest_channels': INPUT_CHANNELS, 'nearest_factor': 2}},
        None, ('expanded_frames', 'upsampled_frames'),
        {'nearest_input_frames': 'expanded_frames',
         'nearest_output_frames': 'upsampled_frames'}))
    add(convolution(f'{stem}_shortcut', f'{stem}_shortcut_upsample',
        f'{prefix}.conv1x1', INPUT_CHANNELS, OUTPUT_CHANNELS,
        upsampled_capacity, 1, 'upsampled_frames', zero_bias=True))
    add(declared(f'{stem}_output',
        'audio_scaled_sum_strided_checked',
        'audio_scaled_sum_strided_f32_checked',
        {'left': f'{stem}_conv1', 'right': f'{stem}_shortcut'},
        f'{stem}_output', [OUTPUT_CHANNELS, upsampled_capacity],
        {'R': OUTPUT_CHANNELS, 'C': upsampled_capacity,
         'scale': 1.0 / math.sqrt(2.0), 'call_constants': {
            'sum_left_elements': OUTPUT_CHANNELS * upsampled_capacity,
            'sum_left_stride': upsampled_capacity,
            'sum_right_elements': OUTPUT_CHANNELS * upsampled_capacity,
            'sum_right_stride': upsampled_capacity,
            'sum_output_elements': OUTPUT_CHANNELS * upsampled_capacity,
            'sum_output_stride': upsampled_capacity,
            'sum_rows': OUTPUT_CHANNELS}},
        None, ('upsampled_frames',),
        {'sum_columns': 'upsampled_frames'}))
    return ops


def build_circuit(tokens=36):
    graph = encode_circuit(tokens)
    graph['name'] = 'kokoro_decoder_complete_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_decoder_complete_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_complete_acoustic_decoder=True, complete_decoder=True,
        complete_waveform=False)
    previous = 'decoder_encode_output'
    for index in range(4):
        stem = f'decoder_decode{index}'
        joins = decoder_join(index, previous)
        block = (decoder_residual_block_ops(stem, f'{stem}_input',
            f'waveform_decoder.decode.{index}', INPUT_CHANNELS, 1024)
            if index < 3 else upsampling_block_ops())
        add_decoder_block(graph, stem, joins + block)
        previous = f'{stem}_output'
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('complete decoder circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
