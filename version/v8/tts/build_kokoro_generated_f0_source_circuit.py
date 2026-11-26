#!/usr/bin/env python3
"""Declare generated Kokoro F0 through bounded harmonic source projection."""

import argparse
import json
from pathlib import Path

from build_kokoro_prosody_complete_circuit import build_circuit as prosody_circuit


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_generated_f0_source_bounded.json'
F0_CAPACITY = 256
UPSAMPLE = 300
HARMONICS = 9
SAMPLES_CAPACITY = F0_CAPACITY * UPSAMPLE
HARMONIC_ELEMENTS = SAMPLES_CAPACITY * HARMONICS
SCRATCH_ELEMENTS = F0_CAPACITY * HARMONICS + HARMONIC_ELEMENTS


def build_circuit(tokens=36, *, prosody_graph=None,
                  f0_capacity=F0_CAPACITY):
    graph = prosody_graph if prosody_graph is not None else prosody_circuit(tokens)
    scratch_elements = f0_capacity * HARMONICS + HARMONIC_ELEMENTS
    graph['name'] = 'kokoro_generated_f0_source_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_generated_f0_source_bounded'
    graph['native_entry']['params'].append(
        {'c_type': 'int32_t *', 'name': 'out_source_samples'})
    graph['native_entry']['runtime_length_outputs']['source_samples'] = \
        'out_source_samples'
    graph['contract']['runtime_invariants'].update(
        generated_f0_source_projection=True, captured_gaussian_input=True,
        native_rng=False, complete_waveform=False)
    graph['runtime_lengths']['source_samples'] = {
        'producer': 'source_samples_extent', 'result': 'valid_extent',
        'capacity': SAMPLES_CAPACITY, 'allow_zero': False}
    for name, shape in (
            ('source_samples_extent_value', [1]),
            ('source_gaussian', [SAMPLES_CAPACITY, HARMONICS]),
            ('source_harmonics', [SAMPLES_CAPACITY, HARMONICS]),
            ('source_projected', [SAMPLES_CAPACITY, 1]),
            ('source_waveform', [SAMPLES_CAPACITY, 1])):
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
    extent = {
        'id': 'source_samples_extent', 'op': 'runtime_extent_scale',
        'kernel': 'runtime_extent_scale_i32', 'returns_status': True,
        'consumes_runtime_lengths': ['upsampled_frames'],
        'produces_runtime_lengths': {'source_samples': 'valid_extent'},
        'runtime_scalar_bindings': {'extent_scale_source': 'upsampled_frames'},
        'params': {'call_constants': {'extent_scale_factor': UPSAMPLE,
            'extent_scale_capacity': SAMPLES_CAPACITY}},
        'graph_slots': {'inputs': {}, 'outputs': {
            'valid_extent': 'source_samples_extent_value'}},
    }
    harmonic = {
        'id': 'source_harmonics', 'op': 'audio_harmonic_source_checked',
        'kernel': 'audio_harmonic_source_weighted_fma_checked_f32',
        'returns_status': True,
        'consumes_runtime_lengths': ['upsampled_frames', 'source_samples'],
        'runtime_scalar_bindings': {'source_frames': 'upsampled_frames'},
        'params': {'F': f0_capacity, 'S': SAMPLES_CAPACITY, 'H': HARMONICS,
            'Q': scratch_elements, 'upsample': UPSAMPLE,
            'harmonics': HARMONICS, 'sample_rate': 24000.,
            'voiced_threshold': 10., 'sine_amp': .1, 'noise_std': .003,
            'call_constants': {'source_f0_capacity': f0_capacity,
                'source_gaussian_capacity': HARMONIC_ELEMENTS,
                'source_output_capacity': HARMONIC_ELEMENTS,
                'source_scratch_capacity': scratch_elements}},
        'graph_slots': {'inputs': {'f0': 'f0_output',
            'gaussian': 'external:source_gaussian'},
            'outputs': {'output': 'source_harmonics'}},
    }
    linear = {
        'id': 'source_projected', 'op': 'linear_rows_checked',
        'kernel': 'linear_rows_checked_f32', 'returns_status': True,
        'consumes_runtime_lengths': ['source_samples'],
        'runtime_scalar_bindings': {'linear_rows': 'source_samples'},
        'weight_refs': {
            'weight': 'waveform_decoder.generator.m_source.l_linear.weight',
            'bias': 'waveform_decoder.generator.m_source.l_linear.bias'},
        'params': {'M': SAMPLES_CAPACITY, 'K': HARMONICS, 'N': 1,
            'call_constants': {'linear_input_elements': HARMONIC_ELEMENTS,
                'linear_input_stride': HARMONICS,
                'linear_weight_elements': HARMONICS,
                'linear_weight_stride': HARMONICS,
                'linear_bias_elements': 1,
                'linear_output_elements': SAMPLES_CAPACITY,
                'linear_output_stride': 1,
                'linear_input_channels': HARMONICS,
                'linear_output_channels': 1}},
        'graph_slots': {'inputs': {'input': 'source_harmonics'},
            'outputs': {'output': 'source_projected'}},
    }
    tanh = {
        'id': 'source_waveform', 'op': 'tanh_strided_checked',
        'kernel': 'tanh_strided_f32_checked', 'returns_status': True,
        'consumes_runtime_lengths': ['source_samples'],
        'runtime_scalar_bindings': {'tanh_rows': 'source_samples'},
        'params': {'R': SAMPLES_CAPACITY, 'C': 1,
            'call_constants': {'tanh_input_elements': SAMPLES_CAPACITY,
                'tanh_input_stride': 1,
                'tanh_output_elements': SAMPLES_CAPACITY,
                'tanh_output_stride': 1,
                'tanh_columns': 1}},
        'graph_slots': {'inputs': {'input': 'source_projected'},
            'outputs': {'output': 'source_waveform'}},
    }
    for item in (harmonic, linear, tanh):
        name = item['id']
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'footer' if name == 'source_waveform' else 'body',
            'template_op_id': name, 'op': item['op'],
            'checkpoints': [{'id': f'kokoro.source.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'sample_major',
                'axis_names': ['sample', 'harmonic'] if name == 'source_harmonics'
                              else ['sample', 'channel'],
                'storage_dtype': 'fp32'}]}
    graph['sequence'].append('source_projection')
    graph['block_types']['source_projection'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [extent], 'body': {'type': 'dense', 'ops': [harmonic, linear]},
        'footer': [tanh]}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('generated F0/source circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
