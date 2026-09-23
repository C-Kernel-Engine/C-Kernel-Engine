#!/usr/bin/env python3
"""Author and optionally execute the certified v8 generated-C training workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = ROOT / "version" / "v7"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import ckernel_engine as cke  # noqa: E402


def build_experiment(run_dir: Path):
    model = cke.models.qwen3_tiny(
        vocab=384, dim=32, layers=4, hidden=64, heads=4, kv_heads=2,
        context_len=32, init="normal_0p02", dtype="float32", name="v8_python_english_fixture",
    )
    return cke.v8.compile(
        model, run_name="v8-python-english-fixture", run_dir=run_dir,
        dataset=cke.v8.DatasetConfig(
            corpus=ROOT / "version" / "v8" / "training" / "english_byte_v1.json",
            max_train_tokens=320, max_validation_tokens=64,
        ),
        tokenizer=cke.v8.TokenizerConfig(kind="bpe", vocab_size=384),
        training=cke.v8.TrainingConfig(epochs=2, grad_accum=2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "version" / "v8" / ".cache" / "python_authoring" / "english_fixture",
    )
    parser.add_argument("--execute", action="store_true", help="Run generated-C training after preflight")
    args = parser.parse_args()

    experiment = build_experiment(args.run_dir.expanduser().resolve())
    preflight = experiment.preflight()
    print(
        f"preflight={preflight['status']} "
        f"can_launch={preflight['can_launch_generated_workflow']} artifact={experiment.preflight_path}"
    )
    print(f"experiment={experiment.experiment_path}")
    print("command=" + " ".join(experiment.command()))
    if args.execute:
        report = experiment.run()
        print(f"training={report['status']} report={experiment.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
