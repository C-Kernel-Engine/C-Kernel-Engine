#!/usr/bin/env python3
"""Fail when unexpected tracked entries appear at the repository root.

The root layout was cleaned up in the 2026-08 root-cleanup series so that it
carries only project structure: code, tests, docs, build entry points, and
standard metadata. This check compares tracked top-level entries against an
explicit allowlist so new clutter cannot silently accumulate again.

To add a legitimate new root entry, add it to ALLOWED below in the same PR.
"""
from __future__ import annotations

import subprocess
import sys

ALLOWED_FILES = {
    ".gitignore",
    ".gitmessage",
    ".gitmodules",
    ".mailmap",
    "CONTRIBUTING.md",
    "CONTRIBUTORS.md",
    "LICENSING.md",
    "Makefile",
    "README.md",
    "RELEASE_NOTES.md",
    "requirements-nightly-constraints.txt",
    "requirements-v7.txt",
    "requirements-v8.txt",
    "requirements.txt",
}

ALLOWED_DIRS = {
    ".githooks",
    ".github",
    "assets",
    "benchmarks",
    "docs",
    "include",
    "llama.cpp",
    "patches",
    "research",
    "scripts",
    "server",
    "src",
    "tests",
    "tools",
    "unittest",
    "version",
}


def main() -> int:
    out = subprocess.check_output(
        ["git", "ls-tree", "--name-only", "HEAD"], text=True
    )
    unexpected = []
    for entry in out.splitlines():
        if entry in ALLOWED_FILES or entry in ALLOWED_DIRS:
            continue
        unexpected.append(entry)
    if unexpected:
        print("Unexpected tracked root entries:")
        for entry in unexpected:
            print(f"  {entry}")
        print(
            "\nMove the entry into docs/, version/, scripts/, or another "
            "existing directory, or add it to the allowlist in "
            "scripts/check_root_layout.py with justification."
        )
        return 1
    print(f"Root layout OK ({len(ALLOWED_FILES)} files, {len(ALLOWED_DIRS)} dirs allowed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
