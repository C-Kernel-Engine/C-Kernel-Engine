#!/usr/bin/env python3
"""Declare the first generated Kokoro prosody stage after duration expansion.

The duration stream, including its checked frame count, is produced by the
existing circuit. This addition transposes its valid region and runs the
reference's shared bidirectional LSTM. F0/N branches remain separate work.
"""
import argparse
import json
from pathlib import Path

from build_kokoro_text_encoder_circuit import build_circuit as text_circuit

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_prosody_shared_bounded.json'
FRAME_CAPACITY = 128
INPUT_CHANNELS = 640
OUTPUT_CHANNELS = 512
HIDDEN = 256


def build_circuit(tokens=36):
    graph = text_circuit(tokens)
    graph['name'] = 'kokoro_prosody_shared_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_prosody_shared_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_prosody_shared_scan=True, complete_prosody=False,
        complete_waveform=False)
    graph['activation_buffers'].update({
        'prosody_input_token': {'shape': [FRAME_CAPACITY, INPUT_CHANNELS]},
        'prosody_shared_features': {'shape': [FRAME_CAPACITY, OUTPUT_CHANNELS]},
    })
    for name in ('prosody_input_token', 'prosody_shared_features'):
        graph['activation_bindings'][name] = name
    transpose = {
        'id': 'prosody_input_transpose', 'op': 'transpose_strided_checked',
        'kernel': 'transpose_strided_f32_checked', 'returns_status': True,
        'consumes_runtime_lengths': ['expanded_frames'],
        'runtime_scalar_bindings': {'transpose_columns': 'expanded_frames'},
        'params': {'R': INPUT_CHANNELS, 'C': FRAME_CAPACITY,
                   'call_constants': {
            'transpose_input_elements': INPUT_CHANNELS * FRAME_CAPACITY,
            'transpose_input_stride': FRAME_CAPACITY,
            'transpose_output_elements': INPUT_CHANNELS * FRAME_CAPACITY,
            'transpose_output_stride': INPUT_CHANNELS,
            'transpose_rows': INPUT_CHANNELS}},
        'graph_slots': {'inputs': {'input': 'duration_expanded'},
                        'outputs': {'output': 'prosody_input_token'}},
    }
    prefix = 'duration_prosody.shared_scan'
    scan = {
        'id': 'prosody_shared_scan', 'op': 'audio_lstm_bidirectional_scan',
        'kernel': 'audio_lstm_bidirectional_scan_f32', 'returns_status': True,
        'consumes_runtime_lengths': ['expanded_frames'],
        'runtime_scalar_bindings': {'tokens': 'expanded_frames'},
        'weight_refs': {kind: f'{prefix}.{kind}' for kind in
                        ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')},
        'params': {'T': FRAME_CAPACITY, 'I': INPUT_CHANNELS, 'H': HIDDEN,
                   'input_size': INPUT_CHANNELS, 'hidden_size': HIDDEN,
                   'call_constants': {
            'scan_input_elements': FRAME_CAPACITY * INPUT_CHANNELS,
            'scan_weight_ih_elements': 2 * 4 * HIDDEN * INPUT_CHANNELS,
            'scan_weight_hh_elements': 2 * 4 * HIDDEN * HIDDEN,
            'scan_bias_ih_elements': 2 * 4 * HIDDEN,
            'scan_bias_hh_elements': 2 * 4 * HIDDEN,
            'scan_output_elements': FRAME_CAPACITY * OUTPUT_CHANNELS,
            'scan_hidden_state_elements': OUTPUT_CHANNELS,
            'scan_cell_state_elements': OUTPUT_CHANNELS,
            'scan_input_stride': INPUT_CHANNELS,
            'scan_output_stride': OUTPUT_CHANNELS}},
        'graph_slots': {'inputs': {'input': 'prosody_input_token'},
                        'outputs': {'output': 'prosody_shared_features'}},
    }
    graph['sequence'].append('prosody_shared')
    graph['block_types']['prosody_shared'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [transpose], 'body': {'type': 'dense', 'ops': [scan]},
        'footer': [],
    }
    for op, tensor in (
            (transpose, 'prosody_input_token'),
            (scan, 'prosody_shared_features')):
        graph['semantic_checkpoints']['exports'][op['id']] = {
            'section': 'header' if op is transpose else 'body',
            'template_op_id': op['id'], 'op': op['op'],
            'checkpoints': [{'id': f"kokoro.prosody.{op['id']}",
                'producer': op['id'], 'tensor': tensor,
                'logical_layout': 'token_major',
                'axis_names': ['frame', 'channel'], 'storage_dtype': 'fp32'}],
        }
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('prosody shared circuit differs from authoring source')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
