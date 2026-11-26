#!/usr/bin/env python3
"""Declare the first generated Kokoro acoustic text-encoder edge.

The complete phoneme/duration graph remains circuit-owned. This bounded graph
adds only the text embedding producer; convolutions and BiLSTM are subsequent
stages, so the expanded embedding is not a complete text-encoder feature.
"""
import argparse
import json
from pathlib import Path

from build_kokoro_expansion_circuit import build_circuit as expansion_circuit

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_text_embedding_bounded.json'


def build_circuit(tokens=36):
    graph = expansion_circuit(tokens)
    graph['name'] = 'kokoro_text_embedding_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_text_embedding_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_text_embedding=True, text_encoder_oracle_fed=False,
        complete_text_encoder=False, complete_waveform=False)
    graph['activation_buffers']['text_features'] = {'shape': [512, 40]}
    graph['activation_buffers']['text_expanded'] = {'shape': [512, 128]}
    graph['sequence'].insert(-1, 'text_embedding')
    text_expand = graph['block_types']['duration_expansion']['body']['ops'][1]
    text_expand['id'] = 'expand_generated_text_embedding'
    text_expand['graph_slots']['inputs']['features'] = 'text_features'
    text_expand['params']['call_constants']['input_elements'] = 512 * 40
    text_expand['params']['call_constants']['output_elements'] = 512 * 128
    # Preserve a non-Kokoro semantic operation and an explicit producer edge.
    lookup = {'id': 'text_embedding', 'op': 'embedding_lookup_checked',
        'kernel': 'audio_text_embedding_channel_major_f32',
        'returns_status': True,
        'weight_refs': {'table': 'acoustic_text_encoder.embedding.weight'},
        'params': {'T': tokens, 'V': 178, 'C': 512, 'T_capacity': 40,
                   'call_constants': {'id_elements': tokens,
            'table_elements': 178 * 512, 'vocabulary': 178, 'channels': 512,
            'table_stride': 512, 'output_elements': 512 * 40,
            'tokens': tokens, 'output_stride': 40}},
        'graph_slots': {'inputs': {'ids': 'external:word_ids'},
                        'outputs': {'output': 'text_features'}}}
    graph['block_types']['text_embedding'] = {'sequence': ['header', 'body', 'footer'],
        'header': [lookup], 'body': {'type': 'dense', 'ops': []}, 'footer': []}
    graph['semantic_checkpoints']['exports']['text_embedding'] = {
        'section': 'header', 'template_op_id': 'text_embedding',
        'op': 'embedding_lookup_checked',
        'checkpoints': [{'id': 'kokoro.text_encoder.embedding',
            'producer': 'text_embedding', 'tensor': 'text_features',
            'logical_layout': 'channel_major',
            'axis_names': ['channel', 'token'], 'storage_dtype': 'fp32'}]}
    graph['required_numerical_contracts']['embedding_lookup_checked'] = {
        'op': 'embedding_lookup_checked',
        'template_ops': ['embedding_lookup_checked'],
        'phases': {'prefill': {
            'contract_id': 'audio_text_embedding_exact_channel_major_fp32',
            'validation': 'validated',
            'evidence': 'tests/test_v8_kokoro_generated_text_embedding.py'}},
        'checkpoint': {'id': 'kokoro.text_encoder.embedding',
            'producer': 'text_embedding', 'logical_layout': 'channel_major',
            'axis_names': ['channel', 'token']}}
    graph['semantic_checkpoints']['exports'].pop('expand_oracle_text', None)
    graph['semantic_checkpoints']['exports'][text_expand['id']] = {
        'section': 'body', 'template_op_id': text_expand['id'],
        'op': text_expand['op'],
        'checkpoints': [{'id': 'kokoro.text_encoder.embedding_expanded',
            'producer': text_expand['id'], 'tensor': 'text_expanded',
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
            raise SystemExit('text embedding circuit differs from authoring source')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents)


if __name__ == '__main__':
    main()
