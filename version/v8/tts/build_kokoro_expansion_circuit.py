#!/usr/bin/env python3
"""Declare generated Kokoro durations and duration features through two expansions.

The duration stream is generated from phoneme IDs. The independent text-encoder
stream remains an explicit oracle-fed input for this bounded composition test.
"""
import argparse
import json
from pathlib import Path

from build_kokoro_duration_circuit import build_circuit as duration_circuit

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_duration_expansion_bounded.json'
TOKENS = 36
CAPACITY = 128


def expansion_op(name, provider, features, output, channels, input_elements,
                 input_stride, tokens=TOKENS):
    return {'id': name, 'op': 'audio_duration_expand', 'kernel': provider,
        'returns_status': True,
        'consumes_runtime_lengths': ['expanded_frames'],
        'params': {'T': tokens, 'C': channels, 'A_capacity': CAPACITY,
                   'channels': channels, 'call_constants': {
            'input_elements': input_elements, 'phoneme_count': tokens,
            'input_stride': input_stride,
            'output_elements': 640 * CAPACITY, 'output_stride': CAPACITY}},
        'graph_slots': {'inputs': {'features': features,
                                   'durations': 'runtime_values'},
                        'outputs': {'output': output}}}


def build_circuit(tokens=TOKENS):
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 2 <= tokens <= 40:
        raise ValueError('expansion tokens must be an integer between 2 and 40')
    graph = duration_circuit(tokens)
    graph['name'] = 'kokoro_duration_expansion_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_duration_expansion'
    graph['contract']['runtime_invariants'].update(
        generated_duration_expansion=True, text_encoder_oracle_fed=True,
        complete_waveform=False)
    graph['activation_buffers'].update({
        'text_features': {'shape': [640, 40]},
        'duration_expanded': {'shape': [640, CAPACITY]},
        'text_expanded': {'shape': [640, CAPACITY]}})
    graph['activation_bindings'].update({name: name for name in
        ('text_features', 'duration_expanded', 'text_expanded')})
    graph['sequence'].append('duration_expansion')
    operations = [
        expansion_op('expand_generated_duration',
            'audio_duration_expand_token_major_f32', 'predictor_features',
            'duration_expanded', 640, tokens * 640, 640, tokens),
        expansion_op('expand_oracle_text',
            'audio_duration_expand_channel_major_f32', 'external:text_features',
            'text_expanded', 512, 640 * 40, 40, tokens)]
    graph['block_types']['duration_expansion'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [], 'body': {'type': 'dense', 'ops': operations}, 'footer': []}
    for operation in operations:
        tensor = operation['graph_slots']['outputs']['output']
        graph['semantic_checkpoints']['exports'][operation['id']] = {
            'section': 'body', 'template_op_id': operation['id'],
            'op': operation['op'],
            'checkpoints': [{'id': f'kokoro.expansion.{tensor}',
                'producer': operation['id'], 'tensor': tensor,
                'logical_layout': 'channel_major',
                'axis_names': ['channel', 'frame'], 'storage_dtype': 'fp32'}]}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    contents = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != contents:
            raise SystemExit('expansion circuit differs from authoring source')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents)


if __name__ == '__main__':
    main()
