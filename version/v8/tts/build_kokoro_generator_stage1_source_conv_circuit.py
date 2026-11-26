#!/usr/bin/env python3
"""Connect Kokoro's second source convolution to its generated STFT."""

import argparse
import json
from pathlib import Path

from build_kokoro_generator_stage1_ingress_circuit import (
    ROOT, PADDED_CAPACITY, build_circuit as ingress_circuit,
)


OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generator_stage1_source_conv_bounded.json'
INPUT_CHANNELS = 22
OUTPUT_CHANNELS = 128


def add_source_conv(graph):
    name = 'generator_stage1_source_conv'
    graph['activation_buffers'][name] = {
        'shape': [OUTPUT_CHANNELS, PADDED_CAPACITY]}
    graph['activation_bindings'][name] = name
    convolution = {
        'id': name, 'op': 'audio_conv1d_checked',
        'kernel': 'audio_conv1d_checked_channel_major_f32',
        'returns_status': True,
        'consumes_runtime_lengths': ['source_stft_frames'],
        'runtime_scalar_bindings': {
            'conv_input_frames': 'source_stft_frames',
            'conv_output_frames': 'source_stft_frames'},
        'weight_refs': {
            'weight': 'waveform_decoder.generator.noise_convs.1.weight',
            'bias': 'waveform_decoder.generator.noise_convs.1.bias'},
        'params': {
            'I': INPUT_CHANNELS, 'O': OUTPUT_CHANNELS,
            'T': PADDED_CAPACITY, 'K': 1, 'U': PADDED_CAPACITY,
            'Q': OUTPUT_CHANNELS * PADDED_CAPACITY,
            'call_constants': {
                'conv_input_elements': INPUT_CHANNELS * PADDED_CAPACITY,
                'conv_input_stride': PADDED_CAPACITY,
                'conv_weight_elements': OUTPUT_CHANNELS * INPUT_CHANNELS,
                'conv_bias_elements': OUTPUT_CHANNELS,
                'conv_output_elements': OUTPUT_CHANNELS * PADDED_CAPACITY,
                'conv_output_stride': PADDED_CAPACITY,
                'conv_scratch_elements': OUTPUT_CHANNELS * PADDED_CAPACITY,
                'conv_input_channels': INPUT_CHANNELS,
                'conv_output_channels': OUTPUT_CHANNELS,
                'conv_kernel_size': 1, 'conv_stride': 1,
                'conv_padding': 0}},
        'graph_slots': {
            'inputs': {'input': 'source_stft_channels'},
            'outputs': {'output': name}}}
    graph['block_types']['generator_stage1_source_conv'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [], 'body': {'type': 'dense', 'ops': [convolution]},
        'footer': []}
    graph['sequence'].append('generator_stage1_source_conv')
    graph['semantic_checkpoints']['exports'][name] = {
        'section': 'body', 'template_op_id': name,
        'op': convolution['op'], 'checkpoints': [{
            'id': 'kokoro.generator.stage1_source_conv',
            'producer': name, 'tensor': name,
            'logical_layout': 'channel_major',
            'axis_names': ['channel', 'frame'], 'storage_dtype': 'fp32'}]}
    return graph


def build_circuit(tokens=36):
    graph = add_source_conv(ingress_circuit(tokens))
    graph['name'] = 'kokoro_generator_stage1_source_conv_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_generator_stage1_source_conv_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_generator_stage1_source_conv=True,
        complete_generator=False, complete_waveform=False)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('stage1 source convolution circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
