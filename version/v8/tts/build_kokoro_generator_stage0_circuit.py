#!/usr/bin/env python3
"""Declare the first main-path Kokoro generator upsampling stage."""

import argparse
import json
from pathlib import Path

from build_kokoro_decoder_complete_circuit import build_circuit as decoder_circuit
from build_kokoro_prosody_second_block_circuit import declared


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generator_stage0_bounded.json'
INPUT_CHANNELS = 512
OUTPUT_CHANNELS = 256
INPUT_CAPACITY = 256
OUTPUT_CAPACITY = 2560
UPSAMPLE = 10
KERNEL = 20
PADDING = 5


def build_circuit(tokens=36, *, decoder_graph=None,
                  input_capacity=INPUT_CAPACITY,
                  output_capacity=OUTPUT_CAPACITY):
    graph = decoder_graph if decoder_graph is not None else decoder_circuit(tokens)
    graph['name'] = 'kokoro_generator_stage0_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_generator_stage0_bounded'
    graph['native_entry']['params'].append(
        {'c_type': 'int32_t *', 'name': 'out_generator_frames'})
    graph['native_entry']['runtime_length_outputs']['generator_stage0_frames'] = \
        'out_generator_frames'
    graph['contract']['runtime_invariants'].update(
        generated_generator_main_upsample_stage0=True,
        complete_generator=False, complete_waveform=False)
    graph['runtime_lengths']['generator_stage0_frames'] = {
        'producer': 'generator_stage0_extent', 'result': 'valid_extent',
        'capacity': output_capacity, 'allow_zero': False}
    graph['activation_buffers']['generator_stage0_extent_value'] = {'shape': [1]}
    graph['activation_bindings']['generator_stage0_extent_value'] = \
        'generator_stage0_extent_value'
    extent = {
        'id': 'generator_stage0_extent', 'op': 'runtime_extent_scale',
        'kernel': 'runtime_extent_scale_i32', 'returns_status': True,
        'consumes_runtime_lengths': ['upsampled_frames'],
        'produces_runtime_lengths': {'generator_stage0_frames': 'valid_extent'},
        'runtime_scalar_bindings': {'extent_scale_source': 'upsampled_frames'},
        'params': {'call_constants': {'extent_scale_factor': UPSAMPLE,
                                     'extent_scale_capacity': output_capacity}},
        'graph_slots': {'inputs': {}, 'outputs': {
            'valid_extent': 'generator_stage0_extent_value'}},
    }
    activation, activation_shape = declared(
        'generator_stage0_activation', 'leaky_relu_strided_checked',
        'leaky_relu_strided_f32_checked',
        {'input': 'decoder_decode3_output'}, 'generator_stage0_activation',
        [INPUT_CHANNELS, input_capacity],
        {'R': INPUT_CHANNELS, 'C': input_capacity,
         'negative_slope': .1,
         'call_constants': {
            'leaky_input_elements': INPUT_CHANNELS * input_capacity,
            'leaky_input_stride': input_capacity,
            'leaky_output_elements': INPUT_CHANNELS * input_capacity,
            'leaky_output_stride': input_capacity,
            'leaky_rows': INPUT_CHANNELS}},
        None, ('upsampled_frames',),
        {'leaky_columns': 'upsampled_frames'})
    deconv, deconv_shape = declared(
        'generator_stage0_upsample', 'audio_conv_transpose1d_dense_checked',
        'audio_conv_transpose1d_dense_channel_major_f32_checked',
        {'input': 'generator_stage0_activation'},
        'generator_stage0_upsample', [OUTPUT_CHANNELS, output_capacity],
        {'I': INPUT_CHANNELS, 'O': OUTPUT_CHANNELS, 'T': input_capacity,
         'K': KERNEL, 'U': output_capacity,
         'Q': OUTPUT_CHANNELS * output_capacity,
         'call_constants': {
            'dense_deconv_input_elements': INPUT_CHANNELS * input_capacity,
            'dense_deconv_input_stride': input_capacity,
            'dense_deconv_weight_elements': INPUT_CHANNELS * OUTPUT_CHANNELS * KERNEL,
            'dense_deconv_bias_elements': OUTPUT_CHANNELS,
            'dense_deconv_output_elements': OUTPUT_CHANNELS * output_capacity,
            'dense_deconv_output_stride': output_capacity,
            'dense_deconv_scratch_elements': OUTPUT_CHANNELS * output_capacity,
            'dense_deconv_input_channels': INPUT_CHANNELS,
            'dense_deconv_output_channels': OUTPUT_CHANNELS,
            'dense_deconv_kernel_size': KERNEL,
            'dense_deconv_stride': UPSAMPLE,
            'dense_deconv_padding': PADDING,
            'dense_deconv_output_padding': 0}},
        {'weight': 'waveform_decoder.generator.ups.0.weight',
         'bias': 'waveform_decoder.generator.ups.0.bias'},
        ('upsampled_frames', 'generator_stage0_frames'),
        {'dense_deconv_input_frames': 'upsampled_frames',
         'dense_deconv_output_frames': 'generator_stage0_frames'})
    for item, shape in ((activation, activation_shape),
                        (deconv, deconv_shape)):
        name = item['id']
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'body' if name == 'generator_stage0_activation' else 'footer',
            'template_op_id': name, 'op': item['op'],
            'checkpoints': [{'id': f'kokoro.generator.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'channel_major',
                'axis_names': ['channel', 'frame'], 'storage_dtype': 'fp32'}]}
    graph['sequence'].append('generator_stage0')
    graph['block_types']['generator_stage0'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [extent], 'body': {'type': 'dense', 'ops': [activation]},
        'footer': [deconv]}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('generator stage0 circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
