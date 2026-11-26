#!/usr/bin/env python3
"""Declare Kokoro's three second-stage main residual blocks and activation."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_generator_stage1_join_circuit import (
    ROOT, CHANNELS, PADDED_CAPACITY, _resize,
    build_circuit as stage1_join_circuit,
)
from build_kokoro_prosody_second_block_circuit import declared


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generator_stage1_pool_bounded.json'
SOURCE_STEMS = ('main_res_pair', 'main_resblock1_pair',
                'main_resblock2_pair')


def _all_ops(block):
    return block['header'] + block['body']['ops'] + block['footer']


def add_stage1_pool(graph):
    for block_index, old_stem in enumerate(SOURCE_STEMS):
        target = block_index + 3
        new_stem = f'stage1_main_resblock{target}_pair'
        source_names = (
            [f'generator_main_residual_pair{pair}' for pair in range(3)]
            if block_index == 0 else
            [f'generator_main_resblock{block_index}_pair{pair}'
             for pair in range(3)])
        ids = {op['id'] for name in source_names
               for op in _all_ops(graph['block_types'][name])}
        renamed = {name: name.replace(old_stem, new_stem) for name in ids}
        renamed['generator_stage0_join'] = 'generator_stage1_join'
        for pair, source_name in enumerate(source_names):
            block = copy.deepcopy(graph['block_types'][source_name])
            for op in _all_ops(block):
                old_id = op['id']
                op['id'] = renamed[old_id]
                for side in ('inputs', 'outputs'):
                    op['graph_slots'][side] = {
                        port: renamed.get(tensor, tensor)
                        for port, tensor in op['graph_slots'][side].items()}
                op['weight_refs'] = {
                    port: name.replace(
                        f'waveform_decoder.generator.resblocks.{block_index}.',
                        f'waveform_decoder.generator.resblocks.{target}.')
                    for port, name in op.get('weight_refs', {}).items()}
                op['consumes_runtime_lengths'] = [
                    'generator_stage1_frames' if length ==
                    'generator_stage0_frames' else length
                    for length in op.get('consumes_runtime_lengths', [])]
                op['runtime_scalar_bindings'] = {
                    scalar: 'generator_stage1_frames' if length ==
                    'generator_stage0_frames' else length
                    for scalar, length in op.get(
                        'runtime_scalar_bindings', {}).items()}
                _resize(op)
                if op['op'] == 'audio_conv1d_dilated_checked':
                    kernel = (3, 7, 11)[block_index]
                    constants = op['params']['call_constants']
                    op['params']['K'] = kernel
                    constants['conv_weight_elements'] = \
                        CHANNELS * CHANNELS * kernel
                    constants['conv_kernel_size'] = kernel
                    constants['conv_padding'] = \
                        constants['conv_dilation'] * (kernel - 1) // 2
                style = op['op'] == 'linear_rows_checked'
                graph['activation_buffers'][op['id']] = {
                    'shape': [2 * CHANNELS] if style else
                        [CHANNELS, PADDED_CAPACITY]}
                graph['activation_bindings'][op['id']] = op['id']
                checkpoint = copy.deepcopy(
                    graph['semantic_checkpoints']['exports'][old_id])
                checkpoint['template_op_id'] = op['id']
                for point in checkpoint['checkpoints']:
                    point['id'] = f'kokoro.generator.{op["id"]}'
                    point['producer'] = op['id']
                    point['tensor'] = op['id']
                graph['semantic_checkpoints']['exports'][op['id']] = checkpoint
            name = f'generator_stage1_main_resblock{target}_pair{pair}'
            graph['block_types'][name] = block
            graph['sequence'].append(name)

    sums = []
    template = graph['block_types']['generator_stage1_join']['body']['ops'][0]
    for name, left, right, scale in (
        ('generator_stage1_residual_sum34',
         'stage1_main_resblock3_pair2_output',
         'stage1_main_resblock4_pair2_output', 1.0),
        ('generator_stage1_residual_mean',
         'generator_stage1_residual_sum34',
         'stage1_main_resblock5_pair2_output', 1.0 / 3.0),
    ):
        op = copy.deepcopy(template)
        op['id'] = name
        op['params']['scale'] = scale
        op['consumes_runtime_lengths'] = ['generator_stage1_frames']
        op['runtime_scalar_bindings'] = {
            scalar: 'generator_stage1_frames'
            for scalar in op.get('runtime_scalar_bindings', {})}
        op['graph_slots'] = {'inputs': {'left': left, 'right': right},
                             'outputs': {'output': name}}
        graph['activation_buffers'][name] = {
            'shape': [CHANNELS, PADDED_CAPACITY]}
        graph['activation_bindings'][name] = name
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'body', 'template_op_id': name, 'op': op['op'],
            'checkpoints': [{'id': f'kokoro.generator.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'channel_major',
                'axis_names': ['channel', 'frame'], 'storage_dtype': 'fp32'}]}
        sums.append(op)
    graph['block_types']['generator_stage1_pool'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [], 'body': {'type': 'dense', 'ops': sums}, 'footer': []}
    graph['sequence'].append('generator_stage1_pool')
    activation, _ = declared(
        'generator_stage1_final_activation',
        'leaky_relu_strided_checked', 'leaky_relu_strided_f32_checked',
        {'input': 'generator_stage1_residual_mean'},
        'generator_stage1_final_activation',
        [CHANNELS, PADDED_CAPACITY],
        {'R': CHANNELS, 'C': PADDED_CAPACITY, 'negative_slope': .01,
         'call_constants': {
             'leaky_input_elements': CHANNELS * PADDED_CAPACITY,
             'leaky_input_stride': PADDED_CAPACITY,
             'leaky_output_elements': CHANNELS * PADDED_CAPACITY,
             'leaky_output_stride': PADDED_CAPACITY,
             'leaky_rows': CHANNELS}},
        None, ('generator_stage1_frames',),
        {'leaky_columns': 'generator_stage1_frames'})
    graph['activation_buffers'][activation['id']] = {
        'shape': [CHANNELS, PADDED_CAPACITY]}
    graph['activation_bindings'][activation['id']] = activation['id']
    graph['semantic_checkpoints']['exports'][activation['id']] = {
        'section': 'footer', 'template_op_id': activation['id'],
        'op': activation['op'], 'checkpoints': [{
            'id': 'kokoro.generator.stage1_final_activation',
            'producer': activation['id'], 'tensor': activation['id'],
            'logical_layout': 'channel_major',
            'axis_names': ['channel', 'frame'], 'storage_dtype': 'fp32'}]}
    graph['block_types']['generator_stage1_final_activation'] = {
        'sequence': ['header', 'body', 'footer'], 'header': [],
        'body': {'type': 'dense', 'ops': []}, 'footer': [activation]}
    graph['sequence'].append('generator_stage1_final_activation')
    return graph


def build_circuit(tokens=36):
    graph = stage1_join_circuit(tokens)
    graph['name'] = 'kokoro_generator_stage1_pool_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_generator_stage1_pool_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_generator_stage1_pool=True,
        complete_generator=False, complete_waveform=False)
    return add_stage1_pool(graph)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('stage-one pool circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
