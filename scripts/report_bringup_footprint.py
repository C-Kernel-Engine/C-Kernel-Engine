#!/usr/bin/env python3
"""Inventory a PR's file and line changes by ownership area (merge base to head)."""
import argparse
from collections import defaultdict
import json
from pathlib import PurePosixPath
import subprocess


def area(path, compiler_paths):
    parts = PurePosixPath(path).parts
    name = parts[-1]
    if path in compiler_paths:
        return "Policy-scoped compiler"
    if path.startswith("docs/"):
        return "Documentation"
    if "fixtures" in parts or "test_assets" in parts:
        return "Fixtures"
    if "tests" in parts or "test" in parts or name.startswith("test_"):
        return "Tests"
    if path.startswith(("version/v8/circuits/", "version/v8/model_maps/")) or (
        name.startswith("build_") and name.endswith("_circuit.py")
    ):
        return "Circuit authoring and model maps"
    if path.startswith(("version/v8/kernel_maps/", "src/kernels/")):
        return "Kernel implementations and maps"
    if path.startswith(("src/runtime/", "version/v8/src/", "server/")):
        return "Runtime and server"
    if path.startswith(("version/v8/tts/", "version/v8/decision/", "version/v8/scripts/")):
        return "Other v8 model code and tooling"
    return "Other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("head")
    parser.add_argument("--json", action="store_true", help="include file-level counts")
    args = parser.parse_args()
    policy = json.loads(subprocess.check_output(
        ["git", "show", f"{args.head}:version/v8/dsl_policy.json"]
    ))
    compiler_paths = set(policy["compiler_files"]) | set(policy["compiler_functions"])
    # With rename detection disabled a move appears as a removal and an addition.
    raw = subprocess.check_output([
        "git", "diff", "--no-renames", "--numstat", "-z", f"{args.base}...{args.head}", "--"
    ])
    files = []
    totals = defaultdict(lambda: [0, 0, 0, 0])
    for record in raw.split(b"\0"):
        if not record:
            continue
        added, removed, raw_path = record.split(b"\t", 2)
        path = raw_path.decode("utf-8", "surrogateescape")
        group = area(path, compiler_paths)
        binary = added == b"-"
        add, delete = (None, None) if binary else (int(added), int(removed))
        files.append(dict(path=path, area=group, added=add, removed=delete))
        counts = totals[group]
        counts[0] += 1
        counts[1] += add or 0
        counts[2] += delete or 0
        counts[3] += binary
    if args.json:
        print(json.dumps(files, indent=2))
        return
    print("### Bring-up change footprint\n")
    print("| Ownership area | Files | Lines added | Lines removed | Binary files |")
    print("|---|---:|---:|---:|---:|")
    totals.setdefault("Policy-scoped compiler", [0, 0, 0, 0])
    for group, counts in sorted(totals.items()):
        print("|", group, "|", " | ".join(map(str, counts)), "|")
    print("\nPath-based churn. Review model host code for execution outside generated C. "
          "Test lines measure changes, not passing cases.")


if __name__ == "__main__":
    main()
