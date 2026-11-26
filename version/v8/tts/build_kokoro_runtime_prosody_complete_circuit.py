#!/usr/bin/env python3
"""Complete both Kokoro prosody branches behind checked runtime lengths."""
import argparse
import copy
import json
from pathlib import Path

from build_kokoro_runtime_prosody_norm_circuit import (
    build_circuit as runtime_norm, FRAMES)
from build_kokoro_prosody_first_block_circuit import (
    CHANNELS, branch_ops as first_branch_ops,
    build_circuit as fixed_first_block)
from build_kokoro_prosody_second_block_circuit import (
    branch_ops as second_branch_ops)
from build_kokoro_prosody_complete_circuit import final_branch

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_runtime_prosody_complete_bounded.json'
DOUBLED_FRAMES = 2 * FRAMES


def add_outputs(graph, operations):
    for op, shape in operations:
        name = op['id']
        style = name.endswith('_style')
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'body', 'template_op_id': name, 'op': op['op'],
            'checkpoints': [{'id': f'kokoro.prosody.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'feature_contiguous' if style else 'channel_major',
                'axis_names': ['channel'] if style else ['channel', 'frame'],
                'storage_dtype': 'fp32'}]}


def build_circuit():
    graph = runtime_norm()
    graph['name'] = 'kokoro_runtime_prosody_complete_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_runtime_prosody_complete_bounded'
    graph['native_entry']['params'].append(
        {'c_type': 'int32_t *', 'name': 'out_upsampled_frames'})
    graph['native_entry']['runtime_length_outputs']['upsampled_frames'] = \
        'out_upsampled_frames'
    graph['contract']['runtime_invariants'].update(
        generated_first_f0_noise_residual_blocks=True,
        generated_second_f0_noise_residual_blocks=True,
        generated_complete_f0_noise_outputs=True,
        complete_prosody=True, complete_waveform=False)
    graph['runtime_lengths']['upsampled_frames'] = {
        'producer': 'prosody_double_frames', 'result': 'valid_extent',
        'capacity': DOUBLED_FRAMES, 'allow_zero': False}
    graph['activation_buffers']['runtime_upsampled_extent'] = {'shape': [1]}
    graph['activation_bindings']['runtime_upsampled_extent'] = \
        'runtime_upsampled_extent'
    first = [(op, [2 * CHANNELS] if op['id'].endswith('_style')
              else [CHANNELS, FRAMES])
             for branch in ('F0', 'N') for op in first_branch_ops(branch, FRAMES)]
    add_outputs(graph, first)
    graph['sequence'].append('prosody_first_block')
    graph['block_types']['prosody_first_block'] = {
        'sequence': ['header', 'body', 'footer'], 'header': [],
        'body': {'type': 'dense', 'ops': [op for op, _ in first]},
        'footer': []}
    graph['required_numerical_contracts']['audio_scaled_sum_strided_checked'] = \
        copy.deepcopy(fixed_first_block()['required_numerical_contracts'][
            'audio_scaled_sum_strided_checked'])
    extent = {'id': 'prosody_double_frames', 'op': 'runtime_extent_scale',
        'kernel': 'runtime_extent_scale_i32', 'returns_status': True,
        'consumes_runtime_lengths': ['expanded_frames'],
        'produces_runtime_lengths': {'upsampled_frames': 'valid_extent'},
        'runtime_scalar_bindings': {'extent_scale_source': 'expanded_frames'},
        'params': {'call_constants': {'extent_scale_factor': 2,
                                     'extent_scale_capacity': DOUBLED_FRAMES}},
        'graph_slots': {'inputs': {},
                        'outputs': {'valid_extent': 'runtime_upsampled_extent'}}}
    second = [entry for branch in ('F0', 'N') for entry in
              second_branch_ops(branch, FRAMES, DOUBLED_FRAMES)]
    add_outputs(graph, second)
    graph['sequence'].append('prosody_second_block')
    graph['block_types']['prosody_second_block'] = {
        'sequence': ['header', 'body', 'footer'], 'header': [extent],
        'body': {'type': 'dense', 'ops': [op for op, _ in second]},
        'footer': []}
    final = [entry for branch in ('F0', 'N') for entry in
             final_branch(branch, DOUBLED_FRAMES)]
    add_outputs(graph, final)
    graph['sequence'].append('prosody_final')
    graph['block_types']['prosody_final'] = {
        'sequence': ['header', 'body', 'footer'], 'header': [],
        'body': {'type': 'dense', 'ops': [op for op, _ in final]},
        'footer': []}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('runtime complete prosody circuit differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
