#!/usr/bin/env python3
"""Declare the first main generator residual block after the source/main join."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_generator_stage0_join_circuit import build_circuit as join_circuit


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generator_main_resblock0_bounded.json'
CHANNELS = 256
CAPACITY = 2560
KERNEL = 3
PREFIX = 'waveform_decoder.generator.resblocks.0'
FRAMES = 'generator_stage0_frames'


def build_circuit(tokens=36):
    graph = join_circuit(tokens)
    graph['name'] = 'kokoro_generator_main_resblock0_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_generator_main_resblock0_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_generator_main_resblock0=True,
        source_phase_numerical_compatibility=False,
        captured_gaussian_input=True, native_rng=False,
        complete_waveform=False)
    source = graph['block_types']['source_residual_pair1']
    templates = {}
    for op in source['header'] + source['body']['ops'] + source['footer']:
        if op['id'] == 'source_res_pair1_output':
            templates['sum'] = op
        else:
            suffix = op['id'].split('_')[-1]
            side = 1 if '_c1_' in op['id'] else 2
            templates[(side, suffix)] = op

    previous = 'generator_stage0_join'
    for pair in range(3):
        stages = {}
        for side in (1, 2):
            for suffix in ('style', 'norm', 'snake', 'conv'):
                op = copy.deepcopy(templates[(side, suffix)])
                name = f'main_res_pair{pair}_c{side}_{suffix}'
                op['id'] = name
                op['graph_slots']['outputs']['output'] = name
                op['consumes_runtime_lengths'] = [
                    FRAMES if length == 'source_conv_frames' else length
                    for length in op.get('consumes_runtime_lengths', [])]
                op['runtime_scalar_bindings'] = {
                    scalar: FRAMES if length == 'source_conv_frames' else length
                    for scalar, length in op.get('runtime_scalar_bindings', {}).items()}
                if suffix == 'style':
                    stem = f'{PREFIX}.adain{side}.{pair}.fc'
                    op['weight_refs'] = {'weight': f'{stem}.weight',
                                         'bias': f'{stem}.bias'}
                elif suffix == 'norm':
                    stem = f'{PREFIX}.adain{side}.{pair}.norm'
                    op['weight_refs'] = {'norm_weight': f'{stem}.weight',
                                         'norm_bias': f'{stem}.bias'}
                    op['graph_slots']['inputs'] = {
                        'input': previous if side == 1 else
                            stages[(1, 'conv')]['id'],
                        'style_affine': stages[(side, 'style')]['id']}
                elif suffix == 'snake':
                    op['weight_refs'] = {
                        'alpha': f'{PREFIX}.alpha{side}.{pair}.channel'}
                    op['graph_slots']['inputs']['input'] = \
                        stages[(side, 'norm')]['id']
                else:
                    stem = f'{PREFIX}.convs{side}.{pair}'
                    op['weight_refs'] = {'weight': f'{stem}.weight',
                                         'bias': f'{stem}.bias'}
                    op['graph_slots']['inputs']['input'] = \
                        stages[(side, 'snake')]['id']
                    dilation = (1, 3, 5)[pair] if side == 1 else 1
                    op['params']['K'] = KERNEL
                    constants = op['params']['call_constants']
                    constants['conv_weight_elements'] = CHANNELS * CHANNELS * KERNEL
                    constants['conv_kernel_size'] = KERNEL
                    constants['conv_dilation'] = dilation
                    constants['conv_padding'] = dilation
                stages[(side, suffix)] = op

        residual = copy.deepcopy(templates['sum'])
        residual['id'] = f'main_res_pair{pair}_output'
        residual['graph_slots']['inputs'] = {
            'left': stages[(2, 'conv')]['id'], 'right': previous}
        residual['graph_slots']['outputs']['output'] = residual['id']
        residual['consumes_runtime_lengths'] = [FRAMES]
        residual['runtime_scalar_bindings']['sum_columns'] = FRAMES
        ordered = [stages[(side, suffix)] for side in (1, 2)
                   for suffix in ('style', 'norm', 'snake', 'conv')]
        for op in (*ordered, residual):
            name = op['id']
            is_style = name.endswith('_style')
            graph['activation_buffers'][name] = {'shape':
                [2 * CHANNELS] if is_style else [CHANNELS, CAPACITY]}
            graph['activation_bindings'][name] = name
            graph['semantic_checkpoints']['exports'][name] = {
                'section': 'footer' if op is residual else
                    ('header' if is_style else 'body'),
                'template_op_id': name, 'op': op['op'],
                'checkpoints': [{'id': f'kokoro.generator.{name}',
                    'producer': name, 'tensor': name,
                    'logical_layout': 'feature_contiguous' if is_style
                        else 'channel_major',
                    'axis_names': ['channel'] if is_style else
                        ['channel', 'frame'], 'storage_dtype': 'fp32'}]}
        block_name = f'generator_main_residual_pair{pair}'
        graph['sequence'].append(block_name)
        graph['block_types'][block_name] = {
            'sequence': ['header', 'body', 'footer'],
            'header': [stages[(1, 'style')], stages[(2, 'style')]],
            'body': {'type': 'dense', 'ops': [
                stages[(side, suffix)] for side in (1, 2)
                for suffix in ('norm', 'snake', 'conv')]},
            'footer': [residual]}
        previous = residual['id']
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('main generator residual circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
