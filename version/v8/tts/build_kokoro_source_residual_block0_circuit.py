#!/usr/bin/env python3
"""Declare Kokoro's complete first source residual block through v8 circuits."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_source_residual_first_pair_circuit import (
    build_circuit as first_pair_circuit)
from build_kokoro_source_residual_prefix_circuit import (
    CAPACITY, CHANNELS, PREFIX, ROOT)


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_source_residual_block0_bounded.json'


def append_pair(graph, pair, input_name):
    prefix_block = graph['block_types']['source_residual_prefix']
    first_pair_block = graph['block_types']['source_residual_first_pair']
    c1_templates = (prefix_block['header'][0],
                    *prefix_block['body']['ops'], prefix_block['footer'][0])
    c2_templates = (first_pair_block['header'][0],
                    *first_pair_block['body']['ops'][:2],
                    first_pair_block['body']['ops'][2])
    result = []
    for side, templates in ((1, c1_templates), (2, c2_templates)):
        base = f'source_res_pair{pair}'
        stem = f'{base}_c{side}'
        style, norm, snake, convolution = (copy.deepcopy(op)
            for op in templates)
        for op, suffix in ((style, 'style'), (norm, 'norm'),
                           (snake, 'snake'), (convolution, 'conv')):
            op['id'] = f'{stem}_{suffix}'
            op['graph_slots']['outputs']['output'] = op['id']
        adain = f'{PREFIX}.adain{side}.{pair}'
        style['weight_refs'] = {'weight': f'{adain}.fc.weight',
                                'bias': f'{adain}.fc.bias'}
        norm['weight_refs'] = {'norm_weight': f'{adain}.norm.weight',
                               'norm_bias': f'{adain}.norm.bias'}
        snake['weight_refs'] = {'alpha':
            f'{PREFIX}.alpha{side}.{pair}.channel'}
        convolution['weight_refs'] = {
            'weight': f'{PREFIX}.convs{side}.{pair}.weight',
            'bias': f'{PREFIX}.convs{side}.{pair}.bias'}
        norm['graph_slots']['inputs'] = {
            'input': input_name if side == 1 else result[-1]['id'],
            'style_affine': style['id']}
        snake['graph_slots']['inputs']['input'] = norm['id']
        convolution['graph_slots']['inputs']['input'] = snake['id']
        dilation = (1, 3, 5)[pair] if side == 1 else 1
        convolution['params']['call_constants']['conv_dilation'] = dilation
        convolution['params']['call_constants']['conv_padding'] = 3 * dilation
        result.extend((style, norm, snake, convolution))
    output_name = f'source_res_pair{pair}_output'
    residual = copy.deepcopy(first_pair_block['footer'][0])
    residual['id'] = output_name
    residual['graph_slots']['inputs'] = {
        'left': result[-1]['id'], 'right': input_name}
    residual['graph_slots']['outputs']['output'] = output_name
    result.append(residual)
    for op in result:
        name = op['id']
        style_tensor = '_style' in name
        graph['activation_buffers'][name] = {'shape':
            [2 * CHANNELS] if style_tensor else [CHANNELS, CAPACITY]}
        graph['activation_bindings'][name] = name
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'footer' if op is residual else
                ('header' if style_tensor else 'body'),
            'template_op_id': name, 'op': op['op'],
            'checkpoints': [{'id': f'kokoro.source.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'feature_contiguous' if style_tensor
                    else 'channel_major',
                'axis_names': ['channel'] if style_tensor else
                    ['channel', 'frame'], 'storage_dtype': 'fp32'}]}
    block_name = f'source_residual_pair{pair}'
    graph['sequence'].append(block_name)
    graph['block_types'][block_name] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [result[0], result[4]],
        'body': {'type': 'dense', 'ops': [*result[1:4], *result[5:8]]},
        'footer': [residual]}
    return output_name


def build_circuit(tokens=36):
    graph = first_pair_circuit(tokens)
    graph['name'] = 'kokoro_source_residual_block0_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_source_residual_block0_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_source_residual_block0=True, complete_waveform=False)
    previous = 'source_res0_pair0'
    for pair in (1, 2):
        previous = append_pair(graph, pair, previous)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('source residual block circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
