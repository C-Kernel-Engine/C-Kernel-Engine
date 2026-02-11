#!/usr/bin/env python3
"""
Resolve v6.6 model input into concrete model cache directory.

This is used by Make targets that need direct paths for ck-cli-v6.6:
  <model_dir>/libmodel.so
  <model_dir>/weights.bump
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ck_run_v6_6 import CACHE_DIR, detect_input_type


def resolve_model_dir(model_input: str) -> Path:
    input_type, info = detect_input_type(model_input)

    if input_type == "hf_gguf":
        return CACHE_DIR / info["repo_id"].replace("/", "--")
    if input_type == "hf_id":
        return CACHE_DIR / info["model_id"].replace("/", "--")
    if input_type == "gguf":
        return CACHE_DIR / info["path"].stem
    if input_type == "local_dir":
        return Path(info["path"]).resolve()
    if input_type == "local_config":
        return Path(info["path"]).resolve().parent

    raise ValueError(f"Unsupported model input type: {input_type}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve model dir for v6.6 input")
    parser.add_argument("--model-input", required=True, help="Same model input accepted by ck_run_v6_6.py")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_input)
    print(str(model_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

