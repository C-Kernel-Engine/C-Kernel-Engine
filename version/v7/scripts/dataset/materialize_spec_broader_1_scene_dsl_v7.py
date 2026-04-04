#!/usr/bin/env python3
"""Materialize a cache-local broader scene-DSL bootstrap workspace."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
BASE_SCRIPT = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec10_scene_dsl_v7.py"
STRUCTURED_GENERATOR = ROOT / "version" / "v7" / "scripts" / "generate_svg_structured_spec_broader_1_v7.py"
LINE_NAME = "spec_broader_1_scene_dsl"
DEFAULT_SEED_WORKSPACE = ROOT / "version" / "v7" / "data" / "spec04"


def _load_base_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("materialize_spec10_scene_dsl_v7", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load base materializer: {BASE_SCRIPT}")
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
) -> dict[str, Any]:
    base = _load_base_module()
    original_generator = base.STRUCTURED_GENERATOR
    original_line_name = base.LINE_NAME
    try:
        base.STRUCTURED_GENERATOR = STRUCTURED_GENERATOR
        base.LINE_NAME = LINE_NAME
        return base.materialize_workspace(
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
        base.STRUCTURED_GENERATOR = original_generator
        base.LINE_NAME = original_line_name


def main() -> int:
    base = _load_base_module()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--seed-workspace", type=Path, default=DEFAULT_SEED_WORKSPACE)
    ap.add_argument("--prefix", default="spec_broader_1_scene_dsl")
    ap.add_argument("--train-repeats", type=int, default=3)
    ap.add_argument("--holdout-repeats", type=int, default=1)
    ap.add_argument("--midtrain-edit-repeat", type=int, default=1)
    ap.add_argument("--midtrain-direct-repeat", type=int, default=4)
    ap.add_argument("--midtrain-close-repeat", type=int, default=6)
    ap.add_argument("--midtrain-shuffle-seed", type=int, default=19)
    ap.add_argument("--python-exec", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    py_exec = str(args.python_exec or Path(base.sys.executable).resolve())
    manifest = materialize_workspace(
        args.workspace,
        seed_workspace=args.seed_workspace,
        prefix=str(args.prefix),
        train_repeats=int(args.train_repeats),
        holdout_repeats=int(args.holdout_repeats),
        midtrain_edit_repeat=int(args.midtrain_edit_repeat),
        midtrain_direct_repeat=int(args.midtrain_direct_repeat),
        midtrain_close_repeat=int(args.midtrain_close_repeat),
        midtrain_shuffle_seed=int(args.midtrain_shuffle_seed),
        python_exec=py_exec,
        force=bool(args.force),
    )
    print(args.workspace.expanduser().resolve())
    print(manifest["workspace_manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
