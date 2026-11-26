#!/usr/bin/env python3
"""Declare complete generated Kokoro F0 and noise prediction branches."""
import argparse
import json
import math
from pathlib import Path

from build_kokoro_prosody_second_block_circuit import (
    build_circuit as second_circuit, declared, style, norm, activation,
    convolution, OUTPUT_CAPACITY, OUTPUT_CHANNELS)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_prosody_complete_bounded.json'


def final_branch(branch, output_capacity=OUTPUT_CAPACITY):
    stem = branch.lower()
    prefix = f'duration_prosody.{branch}.2'
    source = f'{stem}_block1_output'
    ops = []
    add = ops.append
    add(style(f'{stem}_block2_norm0_style', f'{prefix}.norm1', OUTPUT_CHANNELS))
    add(norm(f'{stem}_block2_norm0_output', source,
             f'{stem}_block2_norm0_style', f'{prefix}.norm1',
             OUTPUT_CHANNELS, output_capacity, 'upsampled_frames'))
    add(activation(f'{stem}_block2_act0', f'{stem}_block2_norm0_output',
                   OUTPUT_CHANNELS, output_capacity, 'upsampled_frames'))
    add(convolution(f'{stem}_block2_conv0', f'{stem}_block2_act0',
        f'{prefix}.conv1', OUTPUT_CHANNELS, OUTPUT_CHANNELS,
        output_capacity, 3, 'upsampled_frames'))
    add(style(f'{stem}_block2_norm1_style', f'{prefix}.norm2', OUTPUT_CHANNELS))
    add(norm(f'{stem}_block2_norm1_output', f'{stem}_block2_conv0',
             f'{stem}_block2_norm1_style', f'{prefix}.norm2',
             OUTPUT_CHANNELS, output_capacity, 'upsampled_frames'))
    add(activation(f'{stem}_block2_act1', f'{stem}_block2_norm1_output',
                   OUTPUT_CHANNELS, output_capacity, 'upsampled_frames'))
    add(convolution(f'{stem}_block2_conv1', f'{stem}_block2_act1',
        f'{prefix}.conv2', OUTPUT_CHANNELS, OUTPUT_CHANNELS,
        output_capacity, 3, 'upsampled_frames'))
    add(declared(f'{stem}_block2_output',
        'audio_scaled_sum_strided_checked',
        'audio_scaled_sum_strided_f32_checked',
        {'left': source, 'right': f'{stem}_block2_conv1'},
        f'{stem}_block2_output', [OUTPUT_CHANNELS, output_capacity],
        {'R': OUTPUT_CHANNELS, 'C': output_capacity,
         'scale': 1.0 / math.sqrt(2.0), 'call_constants': {
            'sum_left_elements': OUTPUT_CHANNELS * output_capacity,
            'sum_left_stride': output_capacity,
            'sum_right_elements': OUTPUT_CHANNELS * output_capacity,
            'sum_right_stride': output_capacity,
            'sum_output_elements': OUTPUT_CHANNELS * output_capacity,
            'sum_output_stride': output_capacity,
            'sum_rows': OUTPUT_CHANNELS}},
        None, ('upsampled_frames',),
        {'sum_columns': 'upsampled_frames'}))
    add(convolution(f'{stem}_output', f'{stem}_block2_output',
        f'duration_prosody.{branch}_proj', OUTPUT_CHANNELS, 1,
        output_capacity, 1, 'upsampled_frames'))
    return ops


def build_circuit(tokens=36):
    graph = second_circuit(tokens)
    graph['name'] = 'kokoro_prosody_complete_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_prosody_complete_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_complete_f0_noise_outputs=True,
        complete_prosody=True, complete_waveform=False)
    ops = [entry for branch in ('F0', 'N') for entry in final_branch(branch)]
    for item, shape in ops:
        name = item['id']
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
        style_tensor = name.endswith('_style')
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'body', 'template_op_id': name, 'op': item['op'],
            'checkpoints': [{'id': f'kokoro.prosody.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'feature_contiguous' if style_tensor else
                                  'channel_major',
                'axis_names': ['channel'] if style_tensor else
                              ['channel', 'frame'],
                'storage_dtype': 'fp32'}]}
    graph['sequence'].append('prosody_final')
    graph['block_types']['prosody_final'] = {
        'sequence': ['header', 'body', 'footer'], 'header': [],
        'body': {'type': 'dense', 'ops': [item for item, _ in ops]},
        'footer': []}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('complete prosody circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
