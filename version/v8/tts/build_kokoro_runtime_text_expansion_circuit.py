#!/usr/bin/env python3
"""Declare one checked-token Kokoro duration and acoustic-text expansion graph.

This is authoring only. Its blocks and edges are executed by normal v8
lowering and generated C, with capacity separate from each request's extent.
"""
import argparse
import copy
import json
from pathlib import Path

from build_kokoro_duration_circuit import build_circuit as runtime_duration
from build_kokoro_text_encoder_circuit import build_circuit as fixed_text_encoder

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_runtime_text_expansion_bounded.json'
TOKENS = 64
FRAMES = TOKENS * 50
CHANNELS = 512


def _bind(op, *names):
    op['consumes_runtime_lengths'] = list(dict.fromkeys(
        op.get('consumes_runtime_lengths', []) + ['valid_tokens']))
    op['runtime_scalar_bindings'] = {name: 'valid_tokens' for name in names}
    for name in names:
        op['params']['call_constants'].pop(name, None)


def _specialize_op(op):
    """Set physical capacity and bind only the provider's valid row counts."""
    kind = op['op']
    params = op['params']
    values = params['call_constants']
    if kind == 'embedding_lookup_checked':
        params.update(T=TOKENS, T_capacity=TOKENS)
        values.update(id_elements=TOKENS, output_elements=CHANNELS * TOKENS,
                      output_stride=TOKENS)
        _bind(op, 'tokens')
    elif kind == 'audio_conv1d_checked':
        params.update(T=TOKENS, U=TOKENS, Q=CHANNELS * TOKENS)
        values.update(conv_input_elements=CHANNELS * TOKENS,
                      conv_input_stride=TOKENS,
                      conv_output_elements=CHANNELS * TOKENS,
                      conv_output_stride=TOKENS,
                      conv_scratch_elements=CHANNELS * TOKENS)
        _bind(op, 'conv_input_frames', 'conv_output_frames')
    elif kind == 'transpose_strided_checked':
        channel_to_token = op['id'].endswith('_token')
        params.update(R=CHANNELS if channel_to_token else TOKENS,
                      C=TOKENS if channel_to_token else CHANNELS)
        values.update(transpose_input_elements=CHANNELS * TOKENS,
                      transpose_input_stride=TOKENS if channel_to_token else CHANNELS,
                      transpose_output_elements=CHANNELS * TOKENS,
                      transpose_output_stride=CHANNELS if channel_to_token else TOKENS)
        _bind(op, 'transpose_columns' if channel_to_token else 'transpose_rows')
    elif kind == 'layernorm_rows_param_checked':
        params.update(M=TOKENS, Q=TOKENS * (CHANNELS + 2))
        values.update(norm_input_elements=TOKENS * CHANNELS,
                      norm_output_elements=TOKENS * CHANNELS,
                      norm_scratch_elements=TOKENS * (CHANNELS + 2))
        _bind(op, 'norm_rows')
    elif kind == 'leaky_relu_strided_checked':
        params['R'] = TOKENS
        values.update(leaky_input_elements=TOKENS * CHANNELS,
                      leaky_output_elements=TOKENS * CHANNELS)
        _bind(op, 'leaky_rows')
    elif kind == 'audio_lstm_bidirectional_scan':
        params['T'] = TOKENS
        values.update(scan_input_elements=TOKENS * CHANNELS,
                      scan_output_elements=TOKENS * CHANNELS)
        _bind(op, 'tokens')
    elif kind == 'audio_duration_expand':
        channels = CHANNELS if op['id'] == 'expand_generated_text_encoder' else 640
        params.update(T=TOKENS, C=channels, A_capacity=FRAMES,
                      channels=channels)
        values.update(input_elements=channels * TOKENS,
                      input_stride=channels,
                      output_elements=channels * FRAMES,
                      output_stride=FRAMES)
        _bind(op, 'phoneme_count')
    else:
        raise ValueError('unhandled acoustic text operation: ' + kind)


def build_circuit():
    graph = runtime_duration(TOKENS, runtime_tokens=True)
    fixed = fixed_text_encoder(36)
    graph['name'] = 'kokoro_runtime_text_expansion'
    graph['native_entry']['function'] = 'ck_kokoro_runtime_text_expansion'
    graph['contract']['runtime_invariants'].update(
        generated_acoustic_text_encoder=True,
        generated_duration_expansion=True,
        complete_text_encoder=True, complete_waveform=False,
        text_encoder_oracle_fed=False, max_expanded_frames=FRAMES)
    for block_name in ('text_embedding', 'text_encoder', 'duration_expansion'):
        block = copy.deepcopy(fixed['block_types'][block_name])
        for op in block['header'] + block['body']['ops'] + block['footer']:
            _specialize_op(op)
        graph['block_types'][block_name] = block
        graph['sequence'].append(block_name)
    for name, spec in fixed['activation_buffers'].items():
        if name in graph['activation_buffers']:
            continue
        shape = list(spec['shape'])
        if name in ('duration_expanded', 'text_expanded'):
            shape[-1] = FRAMES
        elif name.endswith('_token'):
            shape[0] = TOKENS
        elif name == 'text_features' or name.startswith('text_conv') or name.endswith('_channel'):
            shape[-1] = TOKENS
        elif name.startswith(('text_norm', 'text_act')) or name == 'text_encoder_features':
            shape[0] = TOKENS
        else:
            raise ValueError('unhandled acoustic text buffer: ' + name)
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
    for name, contract in fixed['required_numerical_contracts'].items():
        if name not in graph['required_numerical_contracts']:
            graph['required_numerical_contracts'][name] = copy.deepcopy(contract)
    for name, checkpoint in fixed['semantic_checkpoints']['exports'].items():
        if name not in graph['semantic_checkpoints']['exports']:
            graph['semantic_checkpoints']['exports'][name] = copy.deepcopy(checkpoint)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    contents = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != contents:
            raise SystemExit('runtime text expansion circuit differs from authoring source')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents)


if __name__ == '__main__':
    main()
