#!/usr/bin/env python3
"""Continue runtime-length Kokoro through the second generator ingress."""

import argparse
import json
from pathlib import Path

from build_kokoro_generator_stage1_ingress_circuit import add_ingress
from build_kokoro_runtime_stage0_pool_circuit import (
    build_circuit as runtime_pool_circuit)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'build/kokoro-diagnostics/kokoro_runtime_stage1_ingress_bounded.json'


def build_circuit():
    graph = add_ingress(runtime_pool_circuit())
    graph['name'] = 'kokoro_runtime_stage1_ingress_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_runtime_stage1_ingress_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_runtime_stage1_ingress=True,
        checked_stage1_deconv_and_reflection_lengths=True,
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
            raise SystemExit('runtime stage1 ingress differs from authoring')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)


if __name__ == '__main__':
    main()
