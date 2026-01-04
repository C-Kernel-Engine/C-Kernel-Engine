#!/usr/bin/env python3
"""
ck_run_v5.py - C-Kernel-Engine v5 Pipeline Runner

Thin wrapper around the v4 pipeline with v5 manifest-first semantics.
"""

import os
import sys

os.environ["CK_V5"] = "1"

import ck_run_v4  # noqa: E402


def main() -> int:
    return ck_run_v4.main()


if __name__ == "__main__":
    sys.exit(main())
