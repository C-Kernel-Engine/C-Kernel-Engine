#!/usr/bin/env python3
"""Small end-to-end example for the ck.nn -> v7 adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ckernel_engine as ck  # noqa: E402


def build_demo_model() -> ck.nn.Sequential:
    return ck.models.qwen3_tiny(
        vocab=256,
        dim=128,
        layers=2,
        hidden=256,
        heads=8,
        kv_heads=4,
        context_len=128,
        init='xavier_uniform',
        name='tiny_qwen3_module_api',
    )


def main() -> int:
    ap = argparse.ArgumentParser(description='Example: ck.nn graph -> v7 compile/train/viewer.')
    ap.add_argument('--run-name', default='python-module-api-demo', help='Run name under the canonical v7 cache train root.')
    ap.add_argument('--text', default='C-Kernel-Engine module API example.', help='Inline training text for the tiny demo run.')
    args = ap.parse_args()

    model = build_demo_model()
    run = ck.v7.compile(
        model,
        run_name=str(args.run_name),
        family='qwen3',
        init='xavier_uniform',
        config=ck.CompileConfig(
            target=ck.TargetConfig(name='cpu', isa='auto'),
            vectorize=True,
            pack_weights=True,
            unroll=1,
            dump_pass_trace=True,
            kernel_policy='fp32_reference_first',
        ),
        tokenizer_notes='Example keeps tokenizer ownership in the existing v7 runtime.',
    )

    print(run.show_graph())
    print(run.show_compile_config())
    materialize_result = run.materialize()
    train_result = run.train(
        str(args.text),
        ck.v7.TrainConfig(
            backend='ck',
            strict=True,
            epochs=1,
            seq_len=8,
            total_tokens=64,
            grad_accum=2,
            lr=5e-4,
            parity_regimen='suggest',
            memory_check=False,
        ),
    )
    viewer_artifacts = run.prepare_viewers()

    print(f'run_dir={materialize_result.run_dir}')
    print(f'graph={run.graph_path}')
    print(f'compile_config={run.compile_config_path}')
    print(f'pass_trace={run.pass_trace_path}')
    print(f'report={train_result.report_path}')
    print(f'plan={run.project_plan_path}')
    print(f'ir_report={viewer_artifacts.ir_report}')
    print(f'dataset_viewer={viewer_artifacts.dataset_viewer}')
    print(f'ir_hub={viewer_artifacts.ir_hub}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
