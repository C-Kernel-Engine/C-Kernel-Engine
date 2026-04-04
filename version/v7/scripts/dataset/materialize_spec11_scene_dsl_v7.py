#!/usr/bin/env python3
"""Materialize a cache-local spec11 keyed scene DSL workspace."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SPEC10_MATERIALIZER = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec10_scene_dsl_v7.py"
STRUCTURED_GENERATOR = ROOT / "version" / "v7" / "scripts" / "generate_svg_structured_spec11_v7.py"
LINE_NAME = "spec11_keyed_scene_dsl"


def _load_spec10_materializer():
    spec = importlib.util.spec_from_file_location("materialize_spec10_scene_dsl_v7", SPEC10_MATERIALIZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load spec10 materializer: {SPEC10_MATERIALIZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize_workspace(
    workspace: Path,
    *,
    seed_workspace: Path,
    prefix: str,
    train_repeats: int,
    holdout_repeats: int,
    midtrain_edit_repeat: int,
    midtrain_direct_repeat: int,
    midtrain_close_repeat: int,
    midtrain_shuffle_seed: int,
    python_exec: str,
    force: bool,
) -> dict[str, object]:
    spec10 = _load_spec10_materializer()
    original_generator = spec10.STRUCTURED_GENERATOR
    original_line_name = spec10.LINE_NAME
    try:
        spec10.STRUCTURED_GENERATOR = STRUCTURED_GENERATOR
        spec10.LINE_NAME = LINE_NAME
        return spec10.materialize_workspace(
            workspace,
            seed_workspace=seed_workspace,
            prefix=prefix,
            train_repeats=train_repeats,
            holdout_repeats=holdout_repeats,
            midtrain_edit_repeat=midtrain_edit_repeat,
            midtrain_direct_repeat=midtrain_direct_repeat,
            midtrain_close_repeat=midtrain_close_repeat,
            midtrain_shuffle_seed=midtrain_shuffle_seed,
            python_exec=python_exec,
            force=force,
        )
    finally:
        spec10.STRUCTURED_GENERATOR = original_generator
        spec10.LINE_NAME = original_line_name


def main() -> int:
    spec10 = _load_spec10_materializer()
    base = spec10._load_base_module()

    import argparse

    ap = argparse.ArgumentParser(description="Materialize a cache-local spec11 keyed scene DSL workspace")
    ap.add_argument("--workspace", required=True, type=Path, help="Destination workspace")
    ap.add_argument("--seed-workspace", default=str(base.DEFAULT_SEED_WORKSPACE), type=Path, help="Seed spec workspace template to copy")
    ap.add_argument("--prefix", default="spec11_keyed_scene_dsl", help="Artifact prefix for generated files")
    ap.add_argument("--train-repeats", type=int, default=3, help="How many times to repeat each seen combo in pretrain/train")
    ap.add_argument("--holdout-repeats", type=int, default=1, help="How many times to repeat each holdout combo before dev/test split")
    ap.add_argument("--midtrain-edit-repeat", type=int, default=2, help="How many times to repeat each semantic midtrain edit row in train")
    ap.add_argument("--midtrain-direct-repeat", type=int, default=4, help="Minimum number of direct scene rows to keep per train example in midtrain")
    ap.add_argument("--midtrain-close-repeat", type=int, default=6, help="How many times to repeat each explicit closing-tag continuation row in midtrain train")
    ap.add_argument("--midtrain-shuffle-seed", type=int, default=42, help="Shuffle seed for interleaving the semantic scene curriculum")
    ap.add_argument("--python-exec", default=str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable)
    ap.add_argument("--force", action="store_true", help="Replace an existing destination workspace")
    args = ap.parse_args()

    summary = materialize_workspace(
        args.workspace,
        seed_workspace=args.seed_workspace,
        prefix=str(args.prefix),
        train_repeats=int(args.train_repeats),
        holdout_repeats=int(args.holdout_repeats),
        midtrain_edit_repeat=max(1, int(args.midtrain_edit_repeat)),
        midtrain_direct_repeat=max(1, int(args.midtrain_direct_repeat)),
        midtrain_close_repeat=max(1, int(args.midtrain_close_repeat)),
        midtrain_shuffle_seed=int(args.midtrain_shuffle_seed),
        python_exec=str(args.python_exec),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
