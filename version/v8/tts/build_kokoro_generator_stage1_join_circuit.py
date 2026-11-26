#!/usr/bin/env python3
"""Declare the second Kokoro source residual block and generator join."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_generator_stage1_source_conv_circuit import (
    ROOT, PADDED_CAPACITY, build_circuit as source_conv_circuit,
)


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generator_stage1_join_bounded.json'
CHANNELS = 128
STYLE = 128
KERNEL = 11
SOURCE_BLOCKS = (
    'source_residual_prefix', 'source_residual_first_pair',
    'source_residual_pair1', 'source_residual_pair2',
)


def _resize(op):
    """Specialize reusable source-block declarations to pinned stage-one geometry."""
    params = op['params']
    constants = params.get('call_constants', {})
    kind = op['op']
    if kind == 'linear_rows_checked':
        params.update(M=1, K=STYLE, N=2 * CHANNELS)
        constants.update(linear_weight_elements=2 * CHANNELS * STYLE,
                         linear_bias_elements=2 * CHANNELS,
                         linear_output_elements=2 * CHANNELS,
                         linear_output_stride=2 * CHANNELS,
                         linear_output_channels=2 * CHANNELS)
    elif kind == 'audio_adain_instance_norm':
        params.update(C=CHANNELS, T=PADDED_CAPACITY)
        constants.update(adain_input_elements=CHANNELS * PADDED_CAPACITY,
                         adain_input_stride=PADDED_CAPACITY,
                         adain_norm_weight_elements=CHANNELS,
                         adain_norm_bias_elements=CHANNELS,
                         adain_style_affine_elements=2 * CHANNELS,
                         adain_output_elements=CHANNELS * PADDED_CAPACITY,
                         adain_output_stride=PADDED_CAPACITY,
                         adain_channels=CHANNELS)
    elif kind == 'audio_snake_strided_checked':
        params.update(C=CHANNELS, T=PADDED_CAPACITY)
        constants.update(snake_input_elements=CHANNELS * PADDED_CAPACITY,
                         snake_input_stride=PADDED_CAPACITY,
                         snake_alpha_elements=CHANNELS,
                         snake_output_elements=CHANNELS * PADDED_CAPACITY,
                         snake_output_stride=PADDED_CAPACITY,
                         snake_channels=CHANNELS)
    elif kind == 'audio_conv1d_dilated_checked':
        params.update(I=CHANNELS, O=CHANNELS, T=PADDED_CAPACITY,
                      K=KERNEL, U=PADDED_CAPACITY,
                      Q=CHANNELS * PADDED_CAPACITY)
        constants.update(conv_input_elements=CHANNELS * PADDED_CAPACITY,
                         conv_input_stride=PADDED_CAPACITY,
                         conv_weight_elements=CHANNELS * CHANNELS * KERNEL,
                         conv_bias_elements=CHANNELS,
                         conv_output_elements=CHANNELS * PADDED_CAPACITY,
                         conv_output_stride=PADDED_CAPACITY,
                         conv_scratch_elements=CHANNELS * PADDED_CAPACITY,
                         conv_input_channels=CHANNELS,
                         conv_output_channels=CHANNELS,
                         conv_kernel_size=KERNEL, conv_stride=1,
                         conv_padding=5 * constants['conv_dilation'])
    elif kind == 'audio_scaled_sum_strided_checked':
        params.update(R=CHANNELS, C=PADDED_CAPACITY)
        constants.update(sum_left_elements=CHANNELS * PADDED_CAPACITY,
                         sum_left_stride=PADDED_CAPACITY,
                         sum_right_elements=CHANNELS * PADDED_CAPACITY,
                         sum_right_stride=PADDED_CAPACITY,
                         sum_output_elements=CHANNELS * PADDED_CAPACITY,
                         sum_output_stride=PADDED_CAPACITY,
                         sum_rows=CHANNELS)
    else:
        raise ValueError(f'unsupported source residual operation: {kind}')


def add_source_residual_and_join(graph):
    extents = {op['id']: op for block in graph['block_types'].values()
               for op in block['header'] + block['body']['ops'] + block['footer']
               if op['id'] in ('source_stft_extent', 'generator_stage0_extent',
                                'generator_stage1_deconv_extent',
                                'generator_stage1_pad_extent')}
    source = extents['source_stft_extent']['params']['call_constants']
    first = extents['generator_stage0_extent']['params']['call_constants']
    second = extents['generator_stage1_deconv_extent']['params']['call_constants']
    pad = extents['generator_stage1_pad_extent']['params']['call_constants']
    if (source['extent_affine_factor'] !=
            first['extent_scale_factor'] * second['extent_scale_factor'] *
            pad['extent_affine_factor'] or
        source['extent_affine_offset'] != pad['extent_affine_offset'] or
        source['extent_affine_capacity'] != pad['extent_affine_capacity']):
        raise ValueError('second source and reflected main extents differ')
    old_ids = {op['id'] for block_name in SOURCE_BLOCKS
               for block in (graph['block_types'][block_name],)
               for op in block['header'] + block['body']['ops'] + block['footer']}
    renamed = {name: f'stage1_{name}' for name in old_ids}
    renamed['source_conv0'] = 'generator_stage1_source_conv'
    for block_name in SOURCE_BLOCKS:
        block = copy.deepcopy(graph['block_types'][block_name])
        for op in block['header'] + block['body']['ops'] + block['footer']:
            old_id = op['id']
            op['id'] = renamed[old_id]
            for side in ('inputs', 'outputs'):
                op['graph_slots'][side] = {
                    slot: renamed.get(tensor, tensor)
                    for slot, tensor in op['graph_slots'][side].items()}
            op['weight_refs'] = {
                slot: name.replace('.noise_res.0.', '.noise_res.1.')
                for slot, name in op.get('weight_refs', {}).items()}
            op['consumes_runtime_lengths'] = [
                'source_stft_frames' if length in (
                    'source_conv_frames', 'generator_stage0_frames') else length
                for length in op.get('consumes_runtime_lengths', [])]
            op['runtime_scalar_bindings'] = {
                scalar: 'source_stft_frames' if length in (
                    'source_conv_frames', 'generator_stage0_frames') else length
                for scalar, length in op.get('runtime_scalar_bindings', {}).items()}
            _resize(op)
            style = op['op'] == 'linear_rows_checked'
            graph['activation_buffers'][op['id']] = {
                'shape': [2 * CHANNELS] if style else
                    [CHANNELS, PADDED_CAPACITY]}
            graph['activation_bindings'][op['id']] = op['id']
            checkpoint = copy.deepcopy(
                graph['semantic_checkpoints']['exports'][old_id])
            checkpoint['template_op_id'] = op['id']
            for point in checkpoint['checkpoints']:
                point['id'] = f'kokoro.source.{op["id"]}'
                point['producer'] = op['id']
                point['tensor'] = op['id']
            graph['semantic_checkpoints']['exports'][op['id']] = checkpoint
        name = f'stage1_{block_name}'
        graph['block_types'][name] = block
        graph['sequence'].append(name)

    join = copy.deepcopy(graph['block_types']['generator_stage0_join']
                         ['body']['ops'][0])
    join['id'] = 'generator_stage1_join'
    join['consumes_runtime_lengths'] = [
        'generator_stage1_frames', 'source_stft_frames']
    join['runtime_scalar_bindings'] = {
        'sum_columns': 'generator_stage1_frames'}
    join['graph_slots'] = {'inputs': {
        'left': 'generator_stage1_reflection',
        'right': renamed['source_res_pair2_output']},
        'outputs': {'output': join['id']}}
    _resize(join)
    graph['activation_buffers'][join['id']] = {
        'shape': [CHANNELS, PADDED_CAPACITY]}
    graph['activation_bindings'][join['id']] = join['id']
    graph['semantic_checkpoints']['exports'][join['id']] = {
        'section': 'body', 'template_op_id': join['id'],
        'op': join['op'], 'checkpoints': [{
            'id': 'kokoro.generator.stage1_join',
            'producer': join['id'], 'tensor': join['id'],
            'logical_layout': 'channel_major',
            'axis_names': ['channel', 'frame'], 'storage_dtype': 'fp32'}]}
    graph['block_types']['generator_stage1_join'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [], 'body': {'type': 'dense', 'ops': [join]}, 'footer': []}
    graph['sequence'].append('generator_stage1_join')
    return graph


def build_circuit(tokens=36):
    graph = source_conv_circuit(tokens)
    graph['name'] = 'kokoro_generator_stage1_join_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_generator_stage1_join_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_generator_stage1_source_residual=True,
        generated_generator_stage1_join=True,
        complete_generator=False, complete_waveform=False)
    return add_source_residual_and_join(graph)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('stage-one source residual/join circuit differs')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
