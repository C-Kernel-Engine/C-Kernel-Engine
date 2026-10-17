#!/usr/bin/env python3
"""Join Kokoro's runtime-length first main and source generator branches."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_generator_stage0_join_circuit import (
    build_circuit as fixed_join_circuit)
from build_kokoro_runtime_source_stft_conv_circuit import (
    build_circuit as runtime_source_stft_circuit)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'version/v8/circuits/kokoro_runtime_stage0_join_bounded.json'
JOIN_BLOCKS = ('source_residual_prefix', 'source_residual_first_pair',
               'source_residual_pair1', 'source_residual_pair2',
               'generator_stage0_join')


def build_circuit():
    graph = runtime_source_stft_circuit()
    fixed = fixed_join_circuit()
    # Source convolution and main upsampling both produce 10 * checked 2A
    # frames. Bind the source convolution and all suffix consumers to the
    # main branch's single checked producer before any output is written.
    source = graph['block_types']['source_stft_conv']
    source['header'] = [op for op in source['header']
                        if op['id'] != 'source_conv_extent']
    convolution = source['footer'][0]
    convolution['consumes_runtime_lengths'] = [
        'generator_stage0_frames' if name == 'source_conv_frames' else name
        for name in convolution['consumes_runtime_lengths']]
    convolution['runtime_scalar_bindings']['conv_output_frames'] = \
        'generator_stage0_frames'
    del graph['runtime_lengths']['source_conv_frames']
    del graph['activation_buffers']['source_conv_extent_value']
    del graph['activation_bindings']['source_conv_extent_value']
    del graph['native_entry']['runtime_length_outputs']['source_conv_frames']
    graph['native_entry']['params'] = [
        item for item in graph['native_entry']['params']
        if item['name'] != 'out_source_conv_frames']

    for name in JOIN_BLOCKS:
        block = copy.deepcopy(fixed['block_types'][name])
        for op in block['header'] + block['body']['ops'] + block['footer']:
            for tensor in op.get('graph_slots', {}).get('outputs', {}).values():
                if tensor in graph['activation_buffers']:
                    raise ValueError(f'duplicate runtime join output: {tensor}')
                graph['activation_buffers'][tensor] = copy.deepcopy(
                    fixed['activation_buffers'][tensor])
                graph['activation_bindings'][tensor] = \
                    fixed['activation_bindings'][tensor]
                graph['semantic_checkpoints']['exports'][tensor] = \
                    copy.deepcopy(fixed['semantic_checkpoints']['exports'][tensor])
        graph['sequence'].append(name)
        graph['block_types'][name] = block
    graph['name'] = 'kokoro_runtime_stage0_join_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_runtime_stage0_join_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_runtime_stage0_join=True,
        shared_checked_stage0_frames=True,
        source_phase_numerical_compatibility=False,
        captured_gaussian_input=True,
        native_rng=False, complete_generator=False, complete_waveform=False)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('runtime stage0 join differs from authoring')
    else:
        args.output.write_text(content)


if __name__ == '__main__':
    main()
