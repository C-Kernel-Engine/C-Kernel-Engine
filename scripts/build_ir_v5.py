#!/usr/bin/env python3
"""
build_ir_v5.py
==============

Thin wrapper over build_ir_v4 with v5 manifest-first rules:
  - Requires a weights manifest unless --weight-dtype is explicitly provided.
  - Disallows overriding mixed-quant manifests.
  - Allows q4_k_m as "mixed" (manifest-driven) only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import build_ir_v4 as v4


def parse_arg_value(argv: list[str], flag: str) -> Optional[str]:
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    for arg in argv:
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def load_manifest_dtypes(manifest_path: Path) -> set[str]:
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    dtypes = {str(e.get("dtype", "")).lower() for e in data.get("entries", [])}
    return {dt for dt in dtypes if dt and dt not in {"fp32", "f32", "bf16", "fp16"}}


def validate_manifest_override(manifest_path: Path, weight_dtype: str) -> None:
    non_fp = load_manifest_dtypes(manifest_path)
    if not non_fp:
        return
    if len(non_fp) > 1:
        types = ", ".join(sorted(non_fp))
        raise SystemExit(
            f"Manifest has mixed quant dtypes ({types}); omit --weight-dtype or use --weight-dtype=q4_k_m."
        )
    only = next(iter(non_fp))
    if weight_dtype not in (only, "q4_k_m"):
        raise SystemExit(f"Manifest dtype is {only}; --weight-dtype={weight_dtype} is incompatible.")


def main(argv: list[str]) -> int:
    manifest_path = parse_arg_value(argv, "--weights-manifest")
    weight_dtype = parse_arg_value(argv, "--weight-dtype")

    if weight_dtype:
        weight_dtype = weight_dtype.lower()

    if weight_dtype == "q4_k_m":
        if not manifest_path:
            raise SystemExit("--weight-dtype=q4_k_m requires --weights-manifest")

    if not manifest_path and not weight_dtype:
        raise SystemExit("v5 requires --weights-manifest unless --weight-dtype is provided")

    if manifest_path and weight_dtype and weight_dtype != "q4_k_m":
        validate_manifest_override(Path(manifest_path), weight_dtype)

    return v4.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
