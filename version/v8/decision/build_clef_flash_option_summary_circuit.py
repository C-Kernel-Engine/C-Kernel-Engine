"""Connect evidence routing to question-conditioned option summaries.

This remains a head prefix: field decoding and decision scores are not included.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from build_clef_flash_evidence_complete_circuit import build_circuit as build_evidence

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'version/v8/circuits/clef_flash_option_summary_bounded.json'


def summary_operation(groups, options, width):
    """Generic operation declaration, also used by the model-neutral test."""
    return {
        'id': 'option_summary_pool', 'op': 'grouped_softmax_weighted_rows_checked',
        'kernel': 'grouped_softmax_weighted_rows_checked_f32', 'returns_status': True,
        'graph_slots': {'inputs': {'options': 'evidence1_output',
                                  'fields': 'question_projected',
                                  'group_counts': 'external:option_counts'},
                        'outputs': {'output': 'option_summary_raw'}},
        'params': {'G': groups, 'O': options, 'C': width,
                   'X': options + groups * width, 'call_constants': {
            'option_elements': options * width, 'option_stride': width,
            'field_elements': groups * width, 'field_stride': width,
            'count_elements': groups, 'output_elements': groups * width,
            'output_stride': width, 'scratch_elements': options + groups * width,
            'groups': groups, 'option_rows': options, 'channels': width}},
    }


def build_circuit(*, tokens=6, questions=2, options=3, channels=4096,
                  width=1024, vocabulary=16, heads=None, feedforward=None):
    graph = build_evidence(tokens=tokens, questions=questions, options=options,
                           channels=channels, width=width, vocabulary=vocabulary,
                           heads=heads, feedforward=feedforward)
    for name in ('option_summary_raw', 'option_summary_normalized'):
        graph['activation_buffers'][name] = {'shape': [questions, width]}
        graph['activation_bindings'][name] = name
    # Reuse the existing operation-local epsilon normalization declaration.
    norm = deepcopy(graph['block_types']['evidence0']['header'][0])
    norm.update(id='option_summary_norm', weight_refs={
        'gamma': 'option_summary_norm.weight', 'beta': 'option_summary_norm.bias'},
        graph_slots={'inputs': {'input': 'option_summary_raw'},
                     'outputs': {'output': 'option_summary_normalized'}})
    norm['params'].update(M=questions, Q=questions * (width + 2))
    norm['params']['call_constants'].update(
        norm_input_elements=questions * width, norm_output_elements=questions * width,
        norm_scratch_elements=questions * (width + 2), norm_rows=questions)
    graph['name'] = 'clef_flash_option_summary_bounded'
    graph['native_entry']['function'] = 'ck_clef_option_summary'
    graph['sequence'].append('option_summary')
    graph['block_types']['option_summary'] = {
        'sequence': ['header', 'body', 'footer'], 'header': [],
        'body': {'type': 'dense', 'ops': [summary_operation(questions, options, width)]},
        'footer': [norm]}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    generated = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if OUTPUT.read_text() != generated:
            raise SystemExit('Clef option-summary circuit differs from canonical authoring')
    else:
        OUTPUT.write_text(generated)


if __name__ == '__main__':
    main()
