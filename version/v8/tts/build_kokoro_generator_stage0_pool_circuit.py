#!/usr/bin/env python3
"""Declare all three parallel first-stage Kokoro generator blocks and their mean."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_generator_main_resblock0_circuit import (
    build_circuit as first_block_circuit, ROOT, CHANNELS, CAPACITY, PREFIX,
)


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generator_stage0_pool_bounded.json'


def build_circuit(tokens=36):
    graph = first_block_circuit(tokens)
    graph['name'] = 'kokoro_generator_stage0_pool_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_generator_stage0_pool_bounded'
    graph['contract']['runtime_invariants'].update(
        complete_generator_stage0=True, complete_waveform=False)

    for block in (1, 2):
        kernel = (3, 7, 11)[block]
        old_prefix = 'main_res_pair'
        new_prefix = f'main_resblock{block}_pair'
        old_ids = {op['id'] for pair in range(3)
                   for section in ('header', 'footer')
                   for op in graph['block_types'][f'generator_main_residual_pair{pair}'][section]}
        old_ids.update(op['id'] for pair in range(3)
                       for op in graph['block_types'][f'generator_main_residual_pair{pair}']['body']['ops'])

        def remap(value):
            return value.replace(old_prefix, new_prefix) if value in old_ids else value

        for pair in range(3):
            original = graph['block_types'][f'generator_main_residual_pair{pair}']
            cloned = copy.deepcopy(original)
            for op in cloned['header'] + cloned['body']['ops'] + cloned['footer']:
                old_id = op['id']
                op['id'] = remap(old_id)
                slots = op['graph_slots']
                slots['inputs'] = {port: remap(value)
                                   for port, value in slots['inputs'].items()}
                slots['outputs'] = {port: remap(value)
                                    for port, value in slots['outputs'].items()}
                op['weight_refs'] = {port: value.replace(PREFIX,
                    f'waveform_decoder.generator.resblocks.{block}')
                    for port, value in op.get('weight_refs', {}).items()}
                if old_id.endswith('_conv'):
                    dilation = (1, 3, 5)[pair] if '_c1_' in old_id else 1
                    op['params']['K'] = kernel
                    constants = op['params']['call_constants']
                    constants['conv_weight_elements'] = CHANNELS * CHANNELS * kernel
                    constants['conv_kernel_size'] = kernel
                    constants['conv_padding'] = dilation * (kernel - 1) // 2
                graph['activation_buffers'][op['id']] = copy.deepcopy(
                    graph['activation_buffers'][old_id])
                graph['activation_bindings'][op['id']] = op['id']
                checkpoint = copy.deepcopy(
                    graph['semantic_checkpoints']['exports'][old_id])
                checkpoint['template_op_id'] = op['id']
                for item in checkpoint['checkpoints']:
                    item['id'] = f'kokoro.generator.{op["id"]}'
                    item['producer'] = op['id']
                    item['tensor'] = op['id']
                graph['semantic_checkpoints']['exports'][op['id']] = checkpoint
            name = f'generator_main_resblock{block}_pair{pair}'
            graph['block_types'][name] = cloned
            graph['sequence'].append(name)

    template = copy.deepcopy(graph['block_types'][
        'generator_main_residual_pair2']['footer'][0])
    sums = []
    for name, left, right, scale in (
        ('generator_stage0_residual_sum01', 'main_res_pair2_output',
         'main_resblock1_pair2_output', 1.0),
        ('generator_stage0_residual_mean', 'generator_stage0_residual_sum01',
         'main_resblock2_pair2_output', 1.0 / 3.0),
    ):
        op = copy.deepcopy(template)
        op['id'] = name
        op['params']['scale'] = scale
        op['graph_slots']['inputs'] = {'left': left, 'right': right}
        op['graph_slots']['outputs']['output'] = name
        graph['activation_buffers'][name] = {'shape': [CHANNELS, CAPACITY]}
        graph['activation_bindings'][name] = name
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'body', 'template_op_id': name, 'op': op['op'],
            'checkpoints': [{'id': f'kokoro.generator.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'channel_major',
                'axis_names': ['channel', 'frame'],
                'storage_dtype': 'fp32'}]}
        sums.append(op)
    graph['block_types']['generator_stage0_pool'] = {
        'sequence': ['header', 'body', 'footer'], 'header': [],
        'body': {'type': 'dense', 'ops': sums}, 'footer': []}
    graph['sequence'].append('generator_stage0_pool')
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('stage0 pool circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
