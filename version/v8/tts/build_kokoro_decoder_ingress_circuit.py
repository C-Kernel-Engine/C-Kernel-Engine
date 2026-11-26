#!/usr/bin/env python3
"""Declare Kokoro's generated acoustic-decoder input convolutions and joins."""

import argparse
import json
from pathlib import Path

from build_kokoro_prosody_complete_circuit import build_circuit as prosody_circuit


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_decoder_ingress_bounded.json'
FRAME_CAPACITY = 128
UPSAMPLED_CAPACITY = 256


def convolution(name, source, prefix, channels_in, channels_out,
                input_capacity, output_capacity, kernel_size, stride,
                padding, input_length, output_length):
    constants = {
        'conv_input_elements': channels_in * input_capacity,
        'conv_input_stride': input_capacity,
        'conv_weight_elements': channels_out * channels_in * kernel_size,
        'conv_bias_elements': channels_out,
        'conv_output_elements': channels_out * output_capacity,
        'conv_output_stride': output_capacity,
        'conv_scratch_elements': channels_out * output_capacity,
        'conv_input_channels': channels_in,
        'conv_output_channels': channels_out,
        'conv_kernel_size': kernel_size,
        'conv_stride': stride,
        'conv_padding': padding,
    }
    return {
        'id': name, 'op': 'audio_conv1d_checked',
        'kernel': 'audio_conv1d_checked_channel_major_f32',
        'returns_status': True,
        'consumes_runtime_lengths': list(dict.fromkeys(
            (input_length, output_length))),
        'runtime_scalar_bindings': {
            'conv_input_frames': input_length,
            'conv_output_frames': output_length},
        'params': {'I': channels_in, 'O': channels_out,
            'T': input_capacity, 'K': kernel_size,
            'U': output_capacity, 'Q': channels_out * output_capacity,
            'call_constants': constants},
        'weight_refs': {'weight': f'{prefix}.weight',
                        'bias': f'{prefix}.bias'},
        'graph_slots': {'inputs': {'input': source},
                        'outputs': {'output': name}},
    }


def concat(name, left, right, left_channels, right_channels,
           frame_capacity=FRAME_CAPACITY):
    output_channels = left_channels + right_channels
    return {
        'id': name, 'op': 'audio_concat_channels_checked',
        'kernel': 'audio_concat_channels_checked_f32',
        'returns_status': True,
        'consumes_runtime_lengths': ['expanded_frames'],
        'runtime_scalar_bindings': {'concat_frames': 'expanded_frames'},
        'params': {'L': left_channels, 'R': right_channels,
            'O': output_channels, 'T': frame_capacity,
            'call_constants': {
                'concat_left_elements': left_channels * frame_capacity,
                'concat_left_stride': frame_capacity,
                'concat_right_elements': right_channels * frame_capacity,
                'concat_right_stride': frame_capacity,
                'concat_output_elements': output_channels * frame_capacity,
                'concat_output_stride': frame_capacity,
                'concat_left_channels': left_channels,
                'concat_right_channels': right_channels,
                'concat_output_channels': output_channels}},
        'graph_slots': {'inputs': {'left': left, 'right': right},
                        'outputs': {'output': name}},
    }


def decoder_ingress_ops(frame_capacity=FRAME_CAPACITY,
                        upsampled_capacity=UPSAMPLED_CAPACITY):
    return [
        convolution('decoder_f0_downsample', 'f0_output',
            'waveform_decoder.F0_conv', 1, 1, upsampled_capacity,
            frame_capacity, 3, 2, 1, 'upsampled_frames', 'expanded_frames'),
        convolution('decoder_n_downsample', 'n_output',
            'waveform_decoder.N_conv', 1, 1, upsampled_capacity,
            frame_capacity, 3, 2, 1, 'upsampled_frames', 'expanded_frames'),
        concat('decoder_text_f0', 'text_expanded',
               'decoder_f0_downsample', 512, 1, frame_capacity),
        concat('decoder_joined', 'decoder_text_f0',
               'decoder_n_downsample', 513, 1, frame_capacity),
        convolution('decoder_asr_res', 'text_expanded',
            'waveform_decoder.asr_res.0', 512, 64, frame_capacity,
            frame_capacity, 1, 1, 0, 'expanded_frames', 'expanded_frames'),
    ]


def add_decoder_ingress(graph, frame_capacity=FRAME_CAPACITY,
                        upsampled_capacity=UPSAMPLED_CAPACITY):
    operations = decoder_ingress_ops(frame_capacity, upsampled_capacity)
    for operation in operations:
        name = operation['id']
        channels = {'decoder_text_f0': 513, 'decoder_joined': 514,
                    'decoder_asr_res': 64}.get(name, 1)
        graph['activation_buffers'][name] = {
            'shape': [channels, frame_capacity]}
        graph['activation_bindings'][name] = name
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'body', 'template_op_id': name,
            'op': operation['op'],
            'checkpoints': [{'id': f'kokoro.decoder.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'channel_major',
                'axis_names': ['channel', 'frame'],
                'storage_dtype': 'fp32'}]}
    graph['sequence'].append('decoder_ingress')
    graph['block_types']['decoder_ingress'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [], 'body': {'type': 'dense', 'ops': operations},
        'footer': []}
    graph['required_numerical_contracts']['audio_concat_channels_checked'] = {
        'op': 'audio_concat_channels_checked',
        'template_ops': ['audio_concat_channels_checked'],
        'phases': {'prefill': {'contract_id':
            'audio_concat_channels_exact_copy_fp32',
            'validation': 'validated',
            'evidence': 'tests/test_v8_audio_concat_channels_checked.py'}},
        'checkpoint': {'id': 'kokoro.decoder.decoder_joined',
            'producer': 'decoder_joined',
            'logical_layout': 'channel_major',
            'axis_names': ['channel', 'frame']}}
    return graph


def build_circuit(tokens=36):
    graph = prosody_circuit(tokens)
    graph['name'] = 'kokoro_decoder_ingress_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_decoder_ingress_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_decoder_ingress=True, complete_decoder=False,
        complete_waveform=False)
    return add_decoder_ingress(graph)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('decoder ingress circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
