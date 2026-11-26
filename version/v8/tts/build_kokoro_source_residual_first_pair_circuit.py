#!/usr/bin/env python3
"""Declare the first complete conv pair in Kokoro's source residual block."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_source_residual_prefix_circuit import (
    build_circuit as prefix_circuit, CAPACITY, CHANNELS, ROOT)


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_source_residual_first_pair_bounded.json'


def build_circuit(tokens=36):
    graph = prefix_circuit(tokens)
    graph['name'] = 'kokoro_source_residual_first_pair_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_source_residual_first_pair_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_source_residual_first_pair=True, complete_waveform=False)
    old = graph['block_types']['source_residual_prefix']
    style = copy.deepcopy(old['header'][0])
    norm, snake = copy.deepcopy(old['body']['ops'])
    convolution = copy.deepcopy(old['footer'][0])
    for op in (style, norm, snake, convolution):
        op['id'] = op['id'].replace('source_res0_', 'source_res1_')
        op['weight_refs'] = {key: value.replace('adain1.0', 'adain2.0')
            .replace('alpha1.0', 'alpha2.0')
            .replace('convs1.0', 'convs2.0')
            for key, value in op['weight_refs'].items()}
        op['graph_slots']['outputs']['output'] = op['id']
    norm['graph_slots']['inputs'] = {
        'input': 'source_res0_conv0', 'style_affine': style['id']}
    snake['graph_slots']['inputs']['input'] = norm['id']
    convolution['graph_slots']['inputs']['input'] = snake['id']
    pair = {
        'id': 'source_res0_pair0',
        'op': 'audio_scaled_sum_strided_checked',
        'kernel': 'audio_scaled_sum_strided_f32_checked',
        'returns_status': True,
        'consumes_runtime_lengths': ['source_conv_frames'],
        'runtime_scalar_bindings': {'sum_columns': 'source_conv_frames'},
        'params': {'R': CHANNELS, 'C': CAPACITY, 'scale': 1.0,
            'call_constants': {
                'sum_left_elements': CHANNELS * CAPACITY,
                'sum_left_stride': CAPACITY,
                'sum_right_elements': CHANNELS * CAPACITY,
                'sum_right_stride': CAPACITY,
                'sum_output_elements': CHANNELS * CAPACITY,
                'sum_output_stride': CAPACITY,
                'sum_rows': CHANNELS}},
        'graph_slots': {'inputs': {'left': convolution['id'],
                                   'right': 'source_conv0'},
                        'outputs': {'output': 'source_res0_pair0'}}}
    for op in (style, norm, snake, convolution, pair):
        shape = [2 * CHANNELS] if op is style else [CHANNELS, CAPACITY]
        graph['activation_buffers'][op['id']] = {'shape': shape}
        graph['activation_bindings'][op['id']] = op['id']
        graph['semantic_checkpoints']['exports'][op['id']] = {
            'section': 'footer' if op is pair else
                ('header' if op is style else 'body'),
            'template_op_id': op['id'], 'op': op['op'],
            'checkpoints': [{'id': f'kokoro.source.{op["id"]}',
                'producer': op['id'], 'tensor': op['id'],
                'logical_layout': 'feature_contiguous' if op is style
                    else 'channel_major',
                'axis_names': ['channel'] if op is style else
                    ['channel', 'frame'], 'storage_dtype': 'fp32'}]}
    graph['sequence'].append('source_residual_first_pair')
    graph['block_types']['source_residual_first_pair'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [style],
        'body': {'type': 'dense', 'ops': [norm, snake, convolution]},
        'footer': [pair]}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('source residual first-pair circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
