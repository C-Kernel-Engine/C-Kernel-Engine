#!/usr/bin/env python3
"""Declare Kokoro's second main upsample and checked left reflection."""

import argparse
import json
from pathlib import Path

from build_kokoro_generator_stage0_pool_circuit import (
    build_circuit as stage0_circuit, ROOT,
)
from build_kokoro_prosody_second_block_circuit import declared


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generator_stage1_ingress_bounded.json'
INPUT_CHANNELS = 256
OUTPUT_CHANNELS = 128
INPUT_CAPACITY = 2560
DECONV_CAPACITY = 15360
PADDED_CAPACITY = DECONV_CAPACITY + 1
UPSAMPLE = 6
KERNEL = 12
PADDING = 3


def add_ingress(graph):
    graph['runtime_lengths']['generator_stage1_deconv_frames'] = {
        'producer': 'generator_stage1_deconv_extent', 'result': 'valid_extent',
        'capacity': DECONV_CAPACITY, 'allow_zero': False}
    graph['runtime_lengths']['generator_stage1_frames'] = {
        'producer': 'generator_stage1_pad_extent', 'result': 'valid_extent',
        'capacity': PADDED_CAPACITY, 'allow_zero': False}
    for name in ('generator_stage1_deconv_extent_value',
                 'generator_stage1_pad_extent_value'):
        graph['activation_buffers'][name] = {'shape': [1]}
        graph['activation_bindings'][name] = name
    graph['native_entry']['params'].extend((
        {'c_type': 'int32_t *', 'name': 'out_stage1_deconv_frames'},
        {'c_type': 'int32_t *', 'name': 'out_stage1_frames'}))
    graph['native_entry']['runtime_length_outputs'].update({
        'generator_stage1_deconv_frames': 'out_stage1_deconv_frames',
        'generator_stage1_frames': 'out_stage1_frames'})

    scale = {
        'id': 'generator_stage1_deconv_extent', 'op': 'runtime_extent_scale',
        'kernel': 'runtime_extent_scale_i32', 'returns_status': True,
        'consumes_runtime_lengths': ['generator_stage0_frames'],
        'produces_runtime_lengths': {
            'generator_stage1_deconv_frames': 'valid_extent'},
        'runtime_scalar_bindings': {
            'extent_scale_source': 'generator_stage0_frames'},
        'params': {'call_constants': {
            'extent_scale_factor': UPSAMPLE,
            'extent_scale_capacity': DECONV_CAPACITY}},
        'graph_slots': {'inputs': {}, 'outputs': {
            'valid_extent': 'generator_stage1_deconv_extent_value'}}}
    affine = {
        'id': 'generator_stage1_pad_extent', 'op': 'runtime_extent_affine',
        'kernel': 'runtime_extent_affine_i32', 'returns_status': True,
        'consumes_runtime_lengths': ['generator_stage1_deconv_frames'],
        'produces_runtime_lengths': {'generator_stage1_frames': 'valid_extent'},
        'runtime_scalar_bindings': {
            'extent_affine_source': 'generator_stage1_deconv_frames'},
        'params': {'call_constants': {
            'extent_affine_factor': 1, 'extent_affine_offset': 1,
            'extent_affine_capacity': PADDED_CAPACITY}},
        'graph_slots': {'inputs': {}, 'outputs': {
            'valid_extent': 'generator_stage1_pad_extent_value'}}}

    activation, _ = declared(
        'generator_stage1_activation', 'leaky_relu_strided_checked',
        'leaky_relu_strided_f32_checked',
        {'input': 'generator_stage0_residual_mean'},
        'generator_stage1_activation',
        [INPUT_CHANNELS, INPUT_CAPACITY],
        {'R': INPUT_CHANNELS, 'C': INPUT_CAPACITY, 'negative_slope': .1,
         'call_constants': {
             'leaky_input_elements': INPUT_CHANNELS * INPUT_CAPACITY,
             'leaky_input_stride': INPUT_CAPACITY,
             'leaky_output_elements': INPUT_CHANNELS * INPUT_CAPACITY,
             'leaky_output_stride': INPUT_CAPACITY,
             'leaky_rows': INPUT_CHANNELS}},
        None, ('generator_stage0_frames',),
        {'leaky_columns': 'generator_stage0_frames'})
    deconv, _ = declared(
        'generator_stage1_upsample',
        'audio_conv_transpose1d_dense_checked',
        'audio_conv_transpose1d_dense_channel_major_f32_checked',
        {'input': 'generator_stage1_activation'},
        'generator_stage1_upsample',
        [OUTPUT_CHANNELS, DECONV_CAPACITY],
        {'I': INPUT_CHANNELS, 'O': OUTPUT_CHANNELS, 'T': INPUT_CAPACITY,
         'K': KERNEL, 'U': DECONV_CAPACITY,
         'Q': OUTPUT_CHANNELS * DECONV_CAPACITY,
         'call_constants': {
             'dense_deconv_input_elements': INPUT_CHANNELS * INPUT_CAPACITY,
             'dense_deconv_input_stride': INPUT_CAPACITY,
             'dense_deconv_weight_elements':
                 INPUT_CHANNELS * OUTPUT_CHANNELS * KERNEL,
             'dense_deconv_bias_elements': OUTPUT_CHANNELS,
             'dense_deconv_output_elements': OUTPUT_CHANNELS * DECONV_CAPACITY,
             'dense_deconv_output_stride': DECONV_CAPACITY,
             'dense_deconv_scratch_elements': OUTPUT_CHANNELS * DECONV_CAPACITY,
             'dense_deconv_input_channels': INPUT_CHANNELS,
             'dense_deconv_output_channels': OUTPUT_CHANNELS,
             'dense_deconv_kernel_size': KERNEL,
             'dense_deconv_stride': UPSAMPLE,
             'dense_deconv_padding': PADDING,
             'dense_deconv_output_padding': 0}},
        {'weight': 'waveform_decoder.generator.ups.1.weight',
         'bias': 'waveform_decoder.generator.ups.1.bias'},
        ('generator_stage0_frames', 'generator_stage1_deconv_frames'),
        {'dense_deconv_input_frames': 'generator_stage0_frames',
         'dense_deconv_output_frames': 'generator_stage1_deconv_frames'})
    pad, _ = declared(
        'generator_stage1_reflection',
        'audio_reflect_pad1d_left_checked',
        'audio_reflect_pad1d_left_channel_major_f32_checked',
        {'input': 'generator_stage1_upsample'},
        'generator_stage1_reflection',
        [OUTPUT_CHANNELS, PADDED_CAPACITY],
        {'C': OUTPUT_CHANNELS, 'T': DECONV_CAPACITY, 'U': PADDED_CAPACITY,
         'call_constants': {
             'reflect_pad_input_elements': OUTPUT_CHANNELS * DECONV_CAPACITY,
             'reflect_pad_input_stride': DECONV_CAPACITY,
             'reflect_pad_output_elements': OUTPUT_CHANNELS * PADDED_CAPACITY,
             'reflect_pad_output_stride': PADDED_CAPACITY,
             'reflect_pad_channels': OUTPUT_CHANNELS,
             'reflect_pad_left_padding': 1}},
        None, ('generator_stage1_deconv_frames', 'generator_stage1_frames'),
        {'reflect_pad_input_frames': 'generator_stage1_deconv_frames',
         'reflect_pad_output_frames': 'generator_stage1_frames'})
    for op, shape, section in (
        (activation, [INPUT_CHANNELS, INPUT_CAPACITY], 'body'),
        (deconv, [OUTPUT_CHANNELS, DECONV_CAPACITY], 'body'),
        (pad, [OUTPUT_CHANNELS, PADDED_CAPACITY], 'footer'),
    ):
        name = op['id']
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
        graph['semantic_checkpoints']['exports'][name] = {
            'section': section, 'template_op_id': name, 'op': op['op'],
            'checkpoints': [{'id': f'kokoro.generator.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'channel_major',
                'axis_names': ['channel', 'frame'],
                'storage_dtype': 'fp32'}]}
    graph['block_types']['generator_stage1_ingress'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [scale, affine],
        'body': {'type': 'dense', 'ops': [activation, deconv]},
        'footer': [pad]}
    graph['sequence'].append('generator_stage1_ingress')
    return graph


def build_circuit(tokens=36):
    graph = stage0_circuit(tokens)
    graph['name'] = 'kokoro_generator_stage1_ingress_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_generator_stage1_ingress_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_generator_stage1_ingress=True,
        complete_generator=False, complete_waveform=False)
    return add_ingress(graph)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('stage1 ingress circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
