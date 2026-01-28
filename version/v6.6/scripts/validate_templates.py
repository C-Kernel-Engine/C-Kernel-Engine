#!/usr/bin/env python3
"""
=============================================================================
EXPERIMENTAL/FUTURE - NOT USED BY CURRENT v6.6 PIPELINE
=============================================================================
This file is a standalone validation utility for template schemas.
It is NOT called by ck_run_v6_6.py or the current build pipeline.

Part of the experimental template system for future IR2 work.
Current pipeline uses: build_ir_v6_6.py with kernel maps directly.
=============================================================================

validate_templates.py
=====================

Basic schema and consistency checks for v6.6 graph templates.
This does not validate kernel availability; IR build handles that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple


DEFAULT_DIR = os.path.join("version", "v6.6", "templates")


def _is_primitive(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_paths(paths: List[str]) -> List[str]:
    collected: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for name in files:
                    if name.endswith(".json"):
                        collected.append(os.path.join(root, name))
        else:
            collected.append(p)
    return sorted(set(collected))


def validate_template(path: str, data: Dict[str, Any], strict: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    def err(msg: str) -> None:
        errors.append(f"{path}: {msg}")

    def warn(msg: str) -> None:
        warnings.append(f"{path}: {msg}")

    if not isinstance(data, dict):
        err("root must be an object")
        return errors, warnings

    required = ["version", "name", "block_types", "layer_map"]
    for key in required:
        if key not in data:
            err(f"missing required key '{key}'")

    version = data.get("version")
    if version is not None and not isinstance(version, int):
        err("version must be an integer")

    name = data.get("name")
    if name is not None and not isinstance(name, str):
        err("name must be a string")

    modes = data.get("modes")
    if modes is not None:
        if not isinstance(modes, list) or any(not isinstance(m, str) for m in modes):
            err("modes must be a list of strings when present")
            modes = None

    block_types = data.get("block_types")
    if not isinstance(block_types, dict):
        err("block_types must be an object")
        block_types = {}

    for block_name, block in block_types.items():
        if not isinstance(block, dict):
            err(f"block_types.{block_name} must be an object")
            continue
        if modes is None:
            seq = block.get("ops")
            if not isinstance(seq, list) or any(not isinstance(op, str) for op in seq):
                err(f"block_types.{block_name}.ops must be a list of op strings")
            elif not seq:
                warn(f"block_types.{block_name}.ops is empty")
            if any(k != "ops" for k in block.keys()):
                warn(f"block_types.{block_name} has extra keys besides 'ops'")
        else:
            for mode in modes:
                if mode not in block:
                    err(f"block_types.{block_name} missing mode '{mode}'")
                    continue
                seq = block.get(mode)
                if not isinstance(seq, list) or any(not isinstance(op, str) for op in seq):
                    err(f"block_types.{block_name}.{mode} must be a list of op strings")
                elif not seq:
                    warn(f"block_types.{block_name}.{mode} is empty")
            for mode in block.keys():
                if mode not in modes:
                    warn(f"block_types.{block_name} has unknown mode '{mode}'")

    layer_map = data.get("layer_map")
    if not isinstance(layer_map, dict):
        err("layer_map must be an object")
        layer_map = {}

    default_block = layer_map.get("default")
    if default_block is None or not isinstance(default_block, str):
        err("layer_map.default must be a string")
    elif default_block not in block_types:
        err(f"layer_map.default '{default_block}' not in block_types")

    overrides = layer_map.get("overrides", {})
    if overrides is not None and not isinstance(overrides, dict):
        err("layer_map.overrides must be an object if present")
        overrides = {}

    if isinstance(overrides, dict):
        for layer_key, block_name in overrides.items():
            try:
                int(layer_key)
            except Exception:
                warn(f"layer_map.overrides key '{layer_key}' is not an integer string")
            if not isinstance(block_name, str):
                err(f"layer_map.overrides.{layer_key} must be a string")
            elif block_name not in block_types:
                err(f"layer_map.overrides.{layer_key} refers to unknown block '{block_name}'")

    flags = data.get("flags")
    if flags is not None:
        if not isinstance(flags, dict):
            err("flags must be an object if present")
        else:
            for k, v in flags.items():
                if isinstance(v, list):
                    if any(not _is_primitive(x) for x in v):
                        warn(f"flags.{k} list has non-primitive values")
                elif not _is_primitive(v):
                    warn(f"flags.{k} is not a primitive value")

    op_defs = data.get("op_defs")
    if op_defs is not None:
        if not isinstance(op_defs, dict):
            err("op_defs must be an object if present")
        else:
            defined_ops = set(op_defs.keys())
            for block_name, block in block_types.items():
                if not isinstance(block, dict):
                    continue
                if modes is None:
                    seq = block.get("ops")
                    if not isinstance(seq, list):
                        continue
                    for op in seq:
                        if op not in defined_ops:
                            msg = f"op '{op}' used in {block_name}.ops not in op_defs"
                            if strict:
                                err(msg)
                            else:
                                warn(msg)
                else:
                    for mode in modes:
                        seq = block.get(mode)
                        if not isinstance(seq, list):
                            continue
                        for op in seq:
                            if op not in defined_ops:
                                msg = f"op '{op}' used in {block_name}.{mode} not in op_defs"
                                if strict:
                                    err(msg)
                                else:
                                    warn(msg)

    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate v6.6 graph templates")
    ap.add_argument("paths", nargs="*", help="Template file(s) or directory")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = ap.parse_args()

    paths = args.paths or [DEFAULT_DIR]
    json_paths = _collect_paths(paths)
    if not json_paths:
        print("No template JSON files found.", file=sys.stderr)
        return 2

    total_errors: List[str] = []
    total_warnings: List[str] = []
    for path in json_paths:
        try:
            data = _load_json(path)
        except Exception as exc:
            total_errors.append(f"{path}: failed to load JSON: {exc}")
            continue
        errors, warnings = validate_template(path, data, strict=args.strict)
        total_errors.extend(errors)
        total_warnings.extend(warnings)

    if total_errors:
        print("Template validation errors:")
        for e in total_errors:
            print(f"  - {e}")
    if total_warnings:
        print("Template validation warnings:")
        for w in total_warnings:
            print(f"  - {w}")

    if total_errors or (args.strict and total_warnings):
        return 1

    print(f"OK: {len(json_paths)} template file(s) validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
