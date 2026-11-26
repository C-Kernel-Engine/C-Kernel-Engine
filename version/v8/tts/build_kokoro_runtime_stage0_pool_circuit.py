#!/usr/bin/env python3
"""Continue runtime-length Kokoro through the first generator residual pool."""

import argparse
import copy
import json
from pathlib import Path

from build_kokoro_generator_stage0_pool_circuit import (
    build_circuit as fixed_pool_circuit)
from build_kokoro_runtime_stage0_join_circuit import (
    build_circuit as runtime_join_circuit)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_runtime_stage0_pool_bounded.json'


def build_circuit():
    graph = runtime_join_circuit()
    fixed = fixed_pool_circuit()
    join = fixed['sequence'].index('generator_stage0_join')
    for name in fixed['sequence'][join + 1:]:
        block = copy.deepcopy(fixed['block_types'][name])
        for op in block['header'] + block['body']['ops'] + block['footer']:
            for tensor in op.get('graph_slots', {}).get('outputs', {}).values():
                if tensor in graph['activation_buffers']:
                    raise ValueError(f'duplicate runtime pool output: {tensor}')
                graph['activation_buffers'][tensor] = copy.deepcopy(
                    fixed['activation_buffers'][tensor])
                graph['activation_bindings'][tensor] = \
                    fixed['activation_bindings'][tensor]
                graph['semantic_checkpoints']['exports'][tensor] = \
                    copy.deepcopy(fixed['semantic_checkpoints']['exports'][tensor])
        graph['sequence'].append(name)
        graph['block_types'][name] = block
    graph['name'] = 'kokoro_runtime_stage0_pool_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_runtime_stage0_pool_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_runtime_stage0_pool=True,
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
            raise SystemExit('runtime stage0 pool differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
