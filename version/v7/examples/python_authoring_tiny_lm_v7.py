#!/usr/bin/env python3
"""Small end-to-end example for the experimental v7 Python authoring layer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ckernel_engine.v7 import (  # noqa: E402
    DataSource,
    MaterializeOptions,
    TemplateSpec,
    TinyModelSpec,
    TokenizerPlan,
    TrainConfig,
    TrainingProject,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Example: author and launch a tiny v7 training run from Python.")
    ap.add_argument("--run-name", default="python-ui-demo", help="Run name under the canonical v7 cache train root.")
    ap.add_argument("--text", default="C-Kernel-Engine from Python.", help="Inline training text for the tiny demo run.")
    args = ap.parse_args()

    project = TrainingProject(
        run_name=str(args.run_name),
        model=TinyModelSpec(
            init="xavier_uniform",
            layers=2,
            vocab_size=256,
            embed_dim=128,
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=4,
            context_len=128,
        ),
        template=TemplateSpec.builtin_template("qwen3"),
        tokenizer=TokenizerPlan(family="runtime_default", notes="Example keeps tokenizer ownership in the existing v7 runtime."),
    )

    project.materialize(
        MaterializeOptions(
            generate_ir=True,
            generate_runtime=True,
            strict=True,
        )
    )
    result = project.train(
        DataSource.inline_text(str(args.text), description="Example inline training text"),
        TrainConfig(
            backend="ck",
            strict=True,
            epochs=1,
            seq_len=8,
            total_tokens=64,
            grad_accum=2,
            lr=5e-4,
            parity_regimen="suggest",
            memory_check=False,
        ),
    )
    viewer_artifacts = project.prepare_viewers()

    print(f"run_dir={result.run_dir}")
    print(f"report={result.report_path}")
    print(f"plan={result.project_plan_path}")
    print(f"ir_report={viewer_artifacts.ir_report}")
    print(f"dataset_viewer={viewer_artifacts.dataset_viewer}")
    print(f"ir_hub={viewer_artifacts.ir_hub}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
