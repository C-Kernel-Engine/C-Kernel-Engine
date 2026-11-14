#!/usr/bin/env python3
"""Bounded FP32 Ming flow update with supplied predictions/noise, not synthesis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'version/v8/circuits/ming3_tts_flow_step_bounded.json'
SOURCE_REVISION = 'a39b80660bdc5a4394d5e1d41af6dbfd9c443a54'
CHECKPOINT_REVISION = '7247f4c94a5b2e987404ed28a5cfec41b1a4ca61'


def build_circuit(elements=256):
    if type(elements) is not int or not 1 <= elements <= 65536:
        raise ValueError('elements must be an integer in [1, 65536]')
    n = elements
    buffers = {name: {'shape': [n]} for name in
        ('latent', 'prediction', 'null_prediction', 'noise', 'negative_null',
         'difference', 'guidance', 'velocity', 'delta', 'euler', 'stochastic', 'output')}
    buffers.update({name: {'shape': [1]} for name in ('neg_one', 'cfg', 'dt', 'noise_amplitude')})
    def scale(name, value, gain, output):
        return {'id': name, 'op': 'scale_vector_checked', 'kernel': 'scale_vector_checked_f32',
            'returns_status': True,
            'graph_slots': {'inputs': {'input': value, 'gain': 'external:'+gain}, 'outputs': {'output': output}},
            'params': {'R': n, 'X': n, 'call_constants': {'input_elements': n,
                'gain_elements': 1, 'output_elements': n, 'scratch_elements': n, 'rows': n}}}
    def add(name, left, right, output):
        return {'id': name, 'op': 'audio_scaled_sum_strided_checked',
            'kernel': 'audio_scaled_sum_strided_f32_checked', 'returns_status': True,
            'graph_slots': {'inputs': {'left': left, 'right': right}, 'outputs': {'output': output}},
            'params': {'R': 1, 'C': n, 'scale': 1.0, 'call_constants': {
                'sum_left_elements': n, 'sum_left_stride': n,
                'sum_right_elements': n, 'sum_right_stride': n,
                'sum_output_elements': n, 'sum_output_stride': n,
                'sum_rows': 1, 'sum_columns': n}}}
    ops = [scale('negate_null', 'external:null_prediction', 'neg_one', 'negative_null'),
        add('prediction_difference', 'external:prediction', 'negative_null', 'difference'),
        scale('apply_guidance', 'difference', 'cfg', 'guidance'),
        add('guided_velocity', 'external:prediction', 'guidance', 'velocity'),
        scale('euler_delta', 'velocity', 'dt', 'delta'),
        add('euler_update', 'external:latent', 'delta', 'euler'),
        scale('noise_delta', 'external:noise', 'noise_amplitude', 'stochastic'),
        add('stochastic_update', 'euler', 'stochastic', 'output')]
    return {'version': 3, 'name': 'ming3_tts_flow_step_bounded', 'family': 'bounded_graph',
        'checked_native_entry': True,
        'contract': {'runtime_invariants': {'inference_only': True,
            'production_kernel_heap_allocation': False, 'complete_synthesis': False,
            'complete_dit': False, 'complete_sampler': False, 'publisher_bf16_parity': False,
            'reference_scope': 'FP32 supplied predictions, noise and coefficients; one flow step',
            'neg_one_required_value': -1, 'native_rng': False, 'elements': n},
            'provenance': {'source_revision': SOURCE_REVISION,
                'checkpoint_revision': CHECKPOINT_REVISION, 'source_file': 'fm/CFM.py'}},
        'activation_buffers': buffers, 'activation_bindings': {name: name for name in buffers},
        'native_entry': {'function': 'ck_ming3_flow_step_bounded',
            'params': [{'c_type': 'uint8_t *', 'name': 'arena'}, {'c_type': 'size_t', 'name': 'arena_bytes'}],
            'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'}},
        'sequence': ['flow_step'], 'block_types': {'flow_step': {
            'sequence': ['header', 'body', 'footer'], 'header': [],
            'body': {'type': 'dense', 'ops': ops}, 'footer': []}},
        'semantic_checkpoints': {'schema': 'cke.semantic_checkpoint_contract', 'schema_version': 1,
            'exports': {op['id']: {'section': 'body', 'template_op_id': op['id'], 'op': op['op'],
                'checkpoints': [{'id': 'ming3.flow.'+op['id'], 'producer': op['id'],
                    'tensor': next(iter(op['graph_slots']['outputs'].values())),
                    'logical_layout': 'contiguous', 'axis_names': ['element'], 'storage_dtype': 'fp32'}]}
                for op in ops}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2)+'\n'
    if args.check:
        if args.output.read_text() != content: raise SystemExit('Ming flow circuit is stale')
    else: args.output.write_text(content)

if __name__ == '__main__': main()
