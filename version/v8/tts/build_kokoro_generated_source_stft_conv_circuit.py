#!/usr/bin/env python3
"""Declare the bounded Kokoro generated source, STFT, and first source convolution."""

import argparse
import json
from pathlib import Path

from build_kokoro_generated_f0_source_circuit import build_circuit as source_circuit

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generated_source_stft_conv_bounded.json'
F0_CAPACITY = 256
SAMPLE_CAPACITY = 76800
STFT_FRAMES_CAPACITY = SAMPLE_CAPACITY // 5 + 1
STFT_CHANNELS = 22
CONV_CHANNELS = 256
CONV_FRAMES_CAPACITY = F0_CAPACITY * 10


def build_circuit(tokens=36, *, source_graph=None):
    graph = source_graph if source_graph is not None else source_circuit(tokens)
    graph['name'] = 'kokoro_generated_source_stft_conv_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_generated_source_stft_conv_bounded'
    graph['native_entry']['params'].extend([
        {'c_type': 'int32_t *', 'name': 'out_stft_frames'},
        {'c_type': 'int32_t *', 'name': 'out_source_conv_frames'},
    ])
    graph['native_entry']['runtime_length_outputs'].update(
        source_stft_frames='out_stft_frames',
        source_conv_frames='out_source_conv_frames')
    graph['contract']['runtime_invariants'].update(
        generated_source_stft_conv=True, scalar_raw_phase_compatibility=False,
        complete_waveform=False)
    graph['runtime_lengths'].update(
        source_stft_frames={'producer': 'source_stft_extent',
            'result': 'valid_extent', 'capacity': STFT_FRAMES_CAPACITY,
            'allow_zero': False},
        source_conv_frames={'producer': 'source_conv_extent',
            'result': 'valid_extent', 'capacity': CONV_FRAMES_CAPACITY,
            'allow_zero': False})
    for name, shape in (
            ('source_stft_extent_value', [1]),
            ('source_conv_extent_value', [1]),
            ('source_stft_window', [20]),
            ('source_stft_cos', [11, 20]),
            ('source_stft_sin', [11, 20]),
            ('source_stft_channels', [STFT_CHANNELS, STFT_FRAMES_CAPACITY]),
            ('source_conv0', [CONV_CHANNELS, CONV_FRAMES_CAPACITY])):
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
    stft_extent = {
        'id': 'source_stft_extent', 'op': 'runtime_extent_affine',
        'kernel': 'runtime_extent_affine_i32', 'returns_status': True,
        'consumes_runtime_lengths': ['upsampled_frames'],
        'produces_runtime_lengths': {'source_stft_frames': 'valid_extent'},
        'runtime_scalar_bindings': {'extent_affine_source': 'upsampled_frames'},
        'params': {'call_constants': {'extent_affine_factor': 60,
            'extent_affine_offset': 1,
            'extent_affine_capacity': STFT_FRAMES_CAPACITY}},
        'graph_slots': {'inputs': {}, 'outputs': {
            'valid_extent': 'source_stft_extent_value'}},
    }
    conv_extent = {
        'id': 'source_conv_extent', 'op': 'runtime_extent_scale',
        'kernel': 'runtime_extent_scale_i32', 'returns_status': True,
        'consumes_runtime_lengths': ['upsampled_frames'],
        'produces_runtime_lengths': {'source_conv_frames': 'valid_extent'},
        'runtime_scalar_bindings': {'extent_scale_source': 'upsampled_frames'},
        'params': {'call_constants': {'extent_scale_factor': 10,
            'extent_scale_capacity': CONV_FRAMES_CAPACITY}},
        'graph_slots': {'inputs': {}, 'outputs': {
            'valid_extent': 'source_conv_extent_value'}},
    }
    stft = {
        'id': 'source_stft_channels', 'op': 'audio_stft_mag_phase_checked',
        'kernel': 'audio_stft_mag_phase_checked_f32', 'returns_status': True,
        'consumes_runtime_lengths': ['source_samples', 'source_stft_frames'],
        'runtime_scalar_bindings': {'stft_samples_count': 'source_samples',
            'stft_frames': 'source_stft_frames'},
        'params': {'N': SAMPLE_CAPACITY, 'FFT': 20, 'B': 11,
            'F': STFT_FRAMES_CAPACITY,
            'Q': STFT_CHANNELS * STFT_FRAMES_CAPACITY,
            'n_fft': 20, 'hop': 5,
            'call_constants': {'stft_samples_capacity': SAMPLE_CAPACITY,
                'stft_window_capacity': 20, 'stft_table_capacity': 220,
                'stft_output_capacity': STFT_CHANNELS * STFT_FRAMES_CAPACITY,
                'stft_output_stride': STFT_FRAMES_CAPACITY,
                'stft_scratch_capacity': STFT_CHANNELS * STFT_FRAMES_CAPACITY}},
        'graph_slots': {'inputs': {'samples': 'source_waveform',
            'window': 'external:source_stft_window',
            'cos_table': 'external:source_stft_cos',
            'sin_table': 'external:source_stft_sin'},
            'outputs': {'output': 'source_stft_channels'}},
    }
    convolution = {
        'id': 'source_conv0', 'op': 'audio_conv1d_checked',
        'kernel': 'audio_conv1d_checked_channel_major_f32',
        'returns_status': True,
        'consumes_runtime_lengths': ['source_stft_frames', 'source_conv_frames'],
        'runtime_scalar_bindings': {'conv_input_frames': 'source_stft_frames',
            'conv_output_frames': 'source_conv_frames'},
        'weight_refs': {
            'weight': 'waveform_decoder.generator.noise_convs.0.weight',
            'bias': 'waveform_decoder.generator.noise_convs.0.bias'},
        'params': {'I': STFT_CHANNELS, 'O': CONV_CHANNELS,
            'T': STFT_FRAMES_CAPACITY, 'K': 12, 'U': CONV_FRAMES_CAPACITY,
            'Q': CONV_CHANNELS * CONV_FRAMES_CAPACITY,
            'call_constants': {
                'conv_input_elements': STFT_CHANNELS * STFT_FRAMES_CAPACITY,
                'conv_input_stride': STFT_FRAMES_CAPACITY,
                'conv_weight_elements': CONV_CHANNELS * STFT_CHANNELS * 12,
                'conv_bias_elements': CONV_CHANNELS,
                'conv_output_elements': CONV_CHANNELS * CONV_FRAMES_CAPACITY,
                'conv_output_stride': CONV_FRAMES_CAPACITY,
                'conv_scratch_elements': CONV_CHANNELS * CONV_FRAMES_CAPACITY,
                'conv_input_channels': STFT_CHANNELS,
                'conv_output_channels': CONV_CHANNELS,
                'conv_kernel_size': 12, 'conv_stride': 6,
                'conv_padding': 3}},
        'graph_slots': {'inputs': {'input': 'source_stft_channels'},
            'outputs': {'output': 'source_conv0'}},
    }
    graph['sequence'].append('source_stft_conv')
    graph['block_types']['source_stft_conv'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [stft_extent, conv_extent],
        'body': {'type': 'dense', 'ops': [stft]},
        'footer': [convolution]}
    for item in (stft, convolution):
        name = item['id']
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'body' if name == 'source_stft_channels' else 'footer',
            'template_op_id': name, 'op': item['op'],
            'checkpoints': [{'id': f'kokoro.source.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'channel_major',
                'axis_names': ['channel', 'frame'], 'storage_dtype': 'fp32'}]}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('generated source STFT/convolution circuit differs')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
