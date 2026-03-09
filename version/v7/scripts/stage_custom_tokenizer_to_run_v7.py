#!/usr/bin/env python3
"""Copy a prebuilt tokenizer.json + tokenizer_bin into a v7 run directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage custom tokenizer artifacts into a v7 run dir")
    ap.add_argument("--run", required=True, help="Run dir under ~/.cache/ck-engine-v7/models/train")
    ap.add_argument("--tokenizer-json", required=True, help="Path to tokenizer.json")
    ap.add_argument("--tokenizer-bin", required=True, help="Path to tokenizer_bin directory")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    tokenizer_json = Path(args.tokenizer_json).expanduser().resolve()
    tokenizer_bin = Path(args.tokenizer_bin).expanduser().resolve()
    ck_build = run_dir / ".ck_build"

    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")
    if not tokenizer_json.exists():
        raise SystemExit(f"tokenizer.json not found: {tokenizer_json}")
    if not tokenizer_bin.is_dir():
        raise SystemExit(f"tokenizer_bin not found: {tokenizer_bin}")

    shutil.copy2(tokenizer_json, run_dir / "tokenizer.json")
    _copy_tree(tokenizer_bin, run_dir / "tokenizer_bin")

    if ck_build.exists():
        shutil.copy2(tokenizer_json, ck_build / "tokenizer.json")
        _copy_tree(tokenizer_bin, ck_build / "tokenizer_bin")

    print(f"staged tokenizer into {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
