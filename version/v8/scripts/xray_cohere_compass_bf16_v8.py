#!/usr/bin/env python3
"""Run Cohere Compass BF16 PyTorch-vs-CK vision X-ray diagnosis."""

from __future__ import annotations

import sys
from pathlib import Path

import xray_qwen3vl_bf16_v8 as shared_adapter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE = SCRIPT_DIR.parent / "parity_profiles" / "cohere_compass_pytorch_bf16_v1.json"


def main(argv: list[str] | None = None) -> int:
    forwarded = [
        "--model", "cohere_compass",
        "--architecture", "cohere_compass",
        "--model-so-name", "libcohere_encoder.so",
        "--profile", str(DEFAULT_PROFILE),
        "--output-dir", "build/xray/cohere_compass_bf16",
        *(list(argv) if argv is not None else sys.argv[1:]),
    ]
    return shared_adapter.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
