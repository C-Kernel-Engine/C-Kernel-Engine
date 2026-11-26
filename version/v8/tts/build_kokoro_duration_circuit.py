#!/usr/bin/env python3
"""Declare a bounded Kokoro encoder → duration predictor graph."""
import argparse
import json
from pathlib import Path

from build_kokoro_encoder_circuit import build_circuit as encoder_circuit
from build_kokoro_runtime_encoder_circuit import build_circuit as runtime_encoder_circuit

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_duration_predictor_bounded.json'
TOKENS = 36
CHANNELS = 512
STYLE = 128
HIDDEN = 256
SCAN_WIDTH = CHANNELS + STYLE
BINS = 50


def concat_op(index, source, tokens):
    name = f'duration_concat{index}'
    output = f'scan{index}_input' if index < 3 else 'predictor_features'
    return {'id': name, 'op': 'feature_concat_broadcast_rows',
        'kernel': 'feature_concat_broadcast_rows_f32', 'returns_status': True,
        'params': {'T': tokens, 'C': CHANNELS, 'S': STYLE, 'N': SCAN_WIDTH,
                   'call_constants': {
            'concat_input_elements': tokens * CHANNELS,
            'concat_input_stride': CHANNELS,
            'concat_feature_elements': STYLE,
            'concat_output_elements': tokens * SCAN_WIDTH,
            'concat_output_stride': SCAN_WIDTH,
            'concat_rows': tokens, 'concat_input_channels': CHANNELS,
            'concat_feature_channels': STYLE,
            'concat_output_channels': SCAN_WIDTH}},
        'graph_slots': {'inputs': {'input': source,
                                  'feature': 'external:predictor_style'},
                        'outputs': {'output': output}}}


def scan_op(index, source, tokens, final=False):
    prefix = ('duration_prosody.head_scan' if final else
              f'duration_prosody.text_encoder.scan{index}')
    output = 'head_scan_output' if final else f'scan{index}_output'
    return {'id': 'duration_head_scan' if final else f'duration_scan{index}',
        'op': 'audio_lstm_bidirectional_scan',
        'kernel': 'audio_lstm_bidirectional_scan_f32', 'returns_status': True,
        'weight_refs': {kind: f'{prefix}.{kind}' for kind in
                        ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')},
        'params': {'T': tokens, 'I': SCAN_WIDTH, 'H': HIDDEN,
                   'input_size': SCAN_WIDTH, 'hidden_size': HIDDEN,
                   'call_constants': {
            'scan_input_elements': tokens * SCAN_WIDTH,
            'scan_weight_ih_elements': 2 * 4 * HIDDEN * SCAN_WIDTH,
            'scan_weight_hh_elements': 2 * 4 * HIDDEN * HIDDEN,
            'scan_bias_ih_elements': 2 * 4 * HIDDEN,
            'scan_bias_hh_elements': 2 * 4 * HIDDEN,
            'scan_output_elements': tokens * CHANNELS,
            'scan_hidden_state_elements': 2 * HIDDEN,
            'scan_cell_state_elements': 2 * HIDDEN,
            'scan_input_stride': SCAN_WIDTH,
            'scan_output_stride': CHANNELS}},
        'graph_slots': {'inputs': {'input': source},
                        'outputs': {'output': output}}}


def norm_op(index, tokens):
    prefix = f'duration_prosody.text_encoder.norm{index}'
    return {'id': f'duration_norm{index}', 'op': 'audio_adaptive_layer_norm',
        'kernel': 'audio_adaptive_layer_norm_f32', 'returns_status': True,
        'weight_refs': {kind: f'{prefix}.{kind}' for kind in
                        ('projection_weight', 'projection_bias')},
        'params': {'T': tokens, 'C': CHANNELS, 'S': STYLE,
                   'channels': CHANNELS, 'style_dim': STYLE,
                   'normalization_epsilon': 1e-5,
                   'call_constants': {
            'adaln_input_elements': tokens * CHANNELS,
            'adaln_style_elements': STYLE,
            'adaln_projection_weight_elements': 2 * CHANNELS * STYLE,
            'adaln_projection_bias_elements': 2 * CHANNELS,
            'adaln_output_elements': tokens * CHANNELS,
            'adaln_input_stride': CHANNELS,
            'adaln_output_stride': CHANNELS}},
        'graph_slots': {'inputs': {'input': f'scan{index}_output',
                                  'style': 'external:predictor_style'},
                        'outputs': {'output': f'norm{index}_output'}}}


def build_circuit(tokens=TOKENS, *, runtime_tokens=False):
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 2 <= tokens <= 512:
        raise ValueError('duration tokens must be an integer between 2 and 512')
    if runtime_tokens and tokens != 64:
        raise ValueError('runtime duration circuit uses capacity 64')
    graph = runtime_encoder_circuit() if runtime_tokens else encoder_circuit(tokens)
    graph['name'] = ('kokoro_runtime_duration_predictor' if runtime_tokens else
                     'kokoro_duration_predictor_bounded')
    if runtime_tokens:
        graph['native_entry']['function'] = 'ck_kokoro_runtime_duration_predictor'
        graph['native_entry']['params'].append(
            {'c_type': 'int32_t *', 'name': 'out_frames'})
        graph['native_entry']['runtime_length_outputs']['expanded_frames'] = 'out_frames'
    else:
        graph['native_entry'] = {'function': 'ck_kokoro_duration_predictor',
            'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                       {'c_type': 'size_t', 'name': 'arena_bytes'},
                       {'c_type': 'int32_t *', 'name': 'out_frames'}],
            'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
            'runtime_length_outputs': {'expanded_frames': 'out_frames'}}
    frame_capacity = tokens * BINS if runtime_tokens else 128
    graph['contract']['runtime_invariants'].update(
        generated_duration_predictor=True, complete_waveform=False,
        all_tokens_valid=True, fixed_voice='af_heart',
        max_expanded_frames=frame_capacity)
    graph['activation_buffers']['predictor_style'] = {'shape': [STYLE]}
    graph['activation_buffers']['duration_logits'] = {'shape': [tokens, BINS]}
    graph['activation_buffers']['runtime_values'] = {'shape': [tokens]}
    graph['activation_buffers']['runtime_valid_extent'] = {'shape': [1]}
    graph['activation_bindings'].update({name: name for name in
        ('predictor_style', 'duration_logits', 'runtime_values', 'runtime_valid_extent')})
    graph.setdefault('runtime_lengths', {})['expanded_frames'] = {
        'producer': 'duration_frames', 'result': 'expanded_frames',
        'capacity': frame_capacity, 'allow_zero': False}
    if runtime_tokens:
        graph['runtime_constants'].update({
            'logit_elements': tokens * BINS, 'logit_stride': BINS,
            'duration_capacity': tokens})
    else:
        graph['runtime_constants'].update({
            'logit_elements': tokens * BINS, 'phoneme_count': tokens,
            'logit_stride': BINS, 'duration_capacity': tokens,
            'max_duration': BINS, 'expanded_capacity': frame_capacity})
    header = [concat_op(0, 'phoneme_features', tokens)]
    body = []
    for i in range(3):
        graph['activation_buffers'][f'scan{i}_input'] = {'shape': [tokens, SCAN_WIDTH]}
        graph['activation_buffers'][f'scan{i}_output'] = {'shape': [tokens, CHANNELS]}
        graph['activation_buffers'][f'norm{i}_output'] = {'shape': [tokens, CHANNELS]}
        body.extend([scan_op(i, f'scan{i}_input', tokens), norm_op(i, tokens),
                     concat_op(i+1, f'norm{i}_output', tokens)])
    graph['activation_buffers']['predictor_features'] = {'shape': [tokens, SCAN_WIDTH]}
    graph['activation_buffers']['head_scan_output'] = {'shape': [tokens, CHANNELS]}
    for name in graph['activation_buffers']:
        graph['activation_bindings'][name] = name
    projection = {'id': 'duration_projection', 'op': 'linear_rows_checked',
        'kernel': 'linear_rows_checked_f32', 'returns_status': True,
        'weight_refs': {'weight': 'duration_prosody.duration_head.weight',
                        'bias': 'duration_prosody.duration_head.bias'},
        'params': {'M': tokens, 'K': CHANNELS, 'N': BINS,
                   'call_constants': {
            'linear_input_elements': tokens * CHANNELS,
            'linear_input_stride': CHANNELS,
            'linear_weight_elements': BINS * CHANNELS,
            'linear_weight_stride': CHANNELS,
            'linear_bias_elements': BINS,
            'linear_output_elements': tokens * BINS,
            'linear_output_stride': BINS,
            'linear_rows': tokens, 'linear_input_channels': CHANNELS,
            'linear_output_channels': BINS}},
        'graph_slots': {'inputs': {'input': 'head_scan_output'},
                        'outputs': {'output': 'duration_logits'}}}
    frames = {'id': 'duration_frames', 'op': 'audio_duration_logits_to_frames',
        'kernel': 'audio_duration_logits_to_frames_f32', 'returns_status': True,
        'produces_runtime_lengths': {'expanded_frames': 'expanded_frames'},
        'params': {'T': tokens, 'B': BINS, 'bins': BINS},
        'graph_slots': {'inputs': {'logits': 'duration_logits'},
                        'outputs': {'durations': 'runtime_values',
                                    'expanded_frames': 'runtime_valid_extent'}}}
    footer = [scan_op(3, 'predictor_features', tokens, final=True), projection, frames]
    if runtime_tokens:
        for op in header + body + footer:
            kind = op['op']
            binding = {
                'feature_concat_broadcast_rows': 'concat_rows',
                'audio_lstm_bidirectional_scan': 'tokens',
                'audio_adaptive_layer_norm': 'tokens',
                'linear_rows_checked': 'linear_rows',
                'audio_duration_logits_to_frames': 'phoneme_count',
            }[kind]
            op['consumes_runtime_lengths'] = ['valid_tokens']
            op['runtime_scalar_bindings'] = {binding: 'valid_tokens'}
            op['params'].get('call_constants', {}).pop(binding, None)
            if kind == 'audio_duration_logits_to_frames':
                op['params']['call_constants'] = {
                    'logit_elements': tokens * BINS,
                    'logit_stride': BINS,
                    'duration_capacity': tokens,
                    'max_duration': BINS,
                    'expanded_capacity': frame_capacity,
                }
    graph['sequence'].append('duration_predictor')
    graph['block_types']['duration_predictor'] = {'sequence': ['header', 'body', 'footer'],
        'header': header, 'body': {'type': 'dense', 'ops': body}, 'footer': footer}
    for op, family, contract in (
        ('feature_concat_broadcast_rows', 'feature_concat_broadcast_rows',
         'feature_concat_broadcast_rows_copy_fp32'),
        ('audio_lstm_bidirectional_scan', 'audio_lstm',
         'audio_lstm_bidirectional_scan_pytorch_ifgo_fp32'),
        ('audio_adaptive_layer_norm', 'audio_adaptive_layer_norm',
         'audio_adaptive_layer_norm_style_linear_fp32')):
        first = next(item for item in header + body + footer if item['op'] == op)
        graph['required_numerical_contracts'][op] = {
            'op': family, 'template_ops': [op], 'phases': {'prefill': {
                'contract_id': contract, 'validation': 'validated',
                'evidence': 'tests/test_v8_kokoro_generated_duration.py'}},
            'checkpoint': {'id': f'kokoro.duration.{first["id"]}',
                'producer': first['id'], 'logical_layout': 'token_major',
                'axis_names': ['token', 'channel']}}
    exports = graph['semantic_checkpoints']['exports']
    for section, ops in (('header', header), ('body', body), ('footer', footer)):
        for op in ops:
            if op['id'] == 'duration_frames':
                continue  # Runtime length and int32 durations have separate checks.
            tensor = op['graph_slots']['outputs']['output']
            exports[op['id']] = {
                'section': section, 'template_op_id': op['id'], 'op': op['op'],
                'checkpoints': [{'id': f'kokoro.duration.{tensor}',
                    'producer': op['id'], 'tensor': tensor,
                    'logical_layout': 'token_major',
                    'axis_names': ['token', 'channel'], 'storage_dtype': 'fp32'}]}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    contents = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != contents:
            raise SystemExit('duration circuit differs from authoring source')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents)


if __name__ == '__main__':
    main()
