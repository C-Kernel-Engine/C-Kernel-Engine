#!/usr/bin/env python3
"""
Validate v6.6 kernel map JSON files.

Usage:
  python3 version/v6.6/scripts/validate_kernel_maps.py
  python3 version/v6.6/scripts/validate_kernel_maps.py --dir version/v6.6/kernel_maps
  python3 version/v6.6/scripts/validate_kernel_maps.py --check-paths
  python3 version/v6.6/scripts/validate_kernel_maps.py --strict
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple


ALLOWED_SHAPE_SYMBOLS = {
    "T", "E", "AE", "H", "KV", "D", "AD", "I", "AI", "max_T", "V", "M", "N", "K", "S"
}

# Files to skip (meta/index files, not kernel maps)
SKIP_FILES = {
    "KERNEL_REGISTRY.json",  # Catalog of all kernels, different format
    "README.md",
}


def _shape_symbols(expr: str) -> List[str]:
    tokens = re.split(r"[^A-Za-z0-9_]+", expr)
    return [t for t in tokens if t and not t.isdigit()]


def _is_list_of_str(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _is_list_of_str_or_empty(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _is_bool(v: Any) -> bool:
    return isinstance(v, bool)


def _err(errors: List[str], msg: str) -> None:
    errors.append(msg)


def _warn(warnings: List[str], msg: str) -> None:
    warnings.append(msg)


def _validate_buffer(buf: Dict[str, Any],
                     ctx: str,
                     errors: List[str],
                     warnings: List[str],
                     dims: List[str]) -> None:
    if not isinstance(buf, dict):
        _err(errors, f"{ctx}: buffer is not an object")
        return
    if "name" not in buf or not isinstance(buf["name"], str) or not buf["name"]:
        _err(errors, f"{ctx}: buffer missing name")
    if "dtype" not in buf or not isinstance(buf["dtype"], str) or not buf["dtype"]:
        _err(errors, f"{ctx}: buffer missing dtype")
    has_shape = "shape" in buf
    has_size = "size_bytes" in buf
    if not has_shape and not has_size:
        _err(errors, f"{ctx}: buffer missing shape or size_bytes")
    if has_shape:
        shape = buf["shape"]
        if not isinstance(shape, list) or not shape:
            _err(errors, f"{ctx}: shape must be non-empty list")
        else:
            for s in shape:
                if isinstance(s, int):
                    continue
                if isinstance(s, str):
                    for sym in _shape_symbols(s):
                        if sym not in dims and sym not in ALLOWED_SHAPE_SYMBOLS:
                            _warn(warnings, f"{ctx}: unknown shape symbol '{sym}'")
                else:
                    _err(errors, f"{ctx}: shape entries must be str or int")


def _validate_parallelization(par: Dict[str, Any],
                              ctx: str,
                              errors: List[str],
                              warnings: List[str]) -> None:
    if not isinstance(par, dict):
        _err(errors, f"{ctx}: parallelization must be object")
        return
    supported = par.get("supported")
    strategies = par.get("strategies")
    if not _is_list_of_str(supported):
        _err(errors, f"{ctx}: parallelization.supported must be list of strings")
    if not isinstance(strategies, list) or not strategies:
        _err(errors, f"{ctx}: parallelization.strategies must be non-empty list")
        return
    for i, strat in enumerate(strategies):
        sctx = f"{ctx}: strategies[{i}]"
        if not isinstance(strat, dict):
            _err(errors, f"{sctx} must be object")
            continue
        for key in ("name", "partition_dim", "param_style"):
            if key not in strat or not isinstance(strat[key], str) or not strat[key]:
                _err(errors, f"{sctx} missing {key}")
        if strat.get("param_style") == "range":
            rng = strat.get("range")
            if not isinstance(rng, dict):
                _err(errors, f"{sctx} missing range for param_style=range")
            else:
                if "min_chunk" not in rng:
                    _err(errors, f"{sctx}: range.min_chunk required")
                if "align_bytes" in rng and not isinstance(rng["align_bytes"], int):
                    _err(errors, f"{sctx}: range.align_bytes must be int")


def _validate_impl(impl: Dict[str, Any],
                   ctx: str,
                   errors: List[str],
                   warnings: List[str],
                   check_paths: bool) -> None:
    if not isinstance(impl, dict):
        _err(errors, f"{ctx}: impl must be object")
        return
    if "function" not in impl or not isinstance(impl["function"], str) or not impl["function"]:
        _err(errors, f"{ctx}: impl.function required")
    sources = impl.get("sources")
    if not _is_list_of_str_or_empty(sources):
        _err(errors, f"{ctx}: impl.sources must be list of strings")
    elif check_paths:
        for src in sources:
            if not os.path.exists(src):
                _warn(warnings, f"{ctx}: impl.sources missing: {src}")
    variants = impl.get("variants")
    if not isinstance(variants, list) or not variants:
        _err(errors, f"{ctx}: impl.variants must be non-empty list")
        return
    for i, var in enumerate(variants):
        vctx = f"{ctx}: impl.variants[{i}]"
        if not isinstance(var, dict):
            _err(errors, f"{vctx} must be object")
            continue
        if "name" not in var or not isinstance(var["name"], str) or not var["name"]:
            _err(errors, f"{vctx}: name required")
        if "requires" in var and not _is_list_of_str_or_empty(var["requires"]):
            _err(errors, f"{vctx}: requires must be list of strings")
        if "compile_flags" in var and not _is_list_of_str_or_empty(var["compile_flags"]):
            _err(errors, f"{vctx}: compile_flags must be list of strings")


def _validate_tests(tests: Dict[str, Any],
                    ctx: str,
                    errors: List[str],
                    warnings: List[str],
                    check_paths: bool) -> None:
    if not isinstance(tests, dict):
        _err(errors, f"{ctx}: tests must be object")
        return
    for key in ("unit", "bench"):
        if key in tests and not _is_list_of_str_or_empty(tests[key]):
            _err(errors, f"{ctx}: tests.{key} must be list of strings")
        if check_paths and key in tests:
            for p in tests[key]:
                if not os.path.exists(p):
                    _warn(warnings, f"{ctx}: tests.{key} missing: {p}")
    parity = tests.get("parity")
    if parity is not None:
        if not isinstance(parity, list):
            _err(errors, f"{ctx}: tests.parity must be list")
        else:
            for i, item in enumerate(parity):
                pctx = f"{ctx}: tests.parity[{i}]"
                if not isinstance(item, dict):
                    _err(errors, f"{pctx} must be object")
                    continue
                for key in ("kind", "command"):
                    if key not in item or not isinstance(item[key], str) or not item[key]:
                        _err(errors, f"{pctx}: missing {key}")
                if check_paths and item.get("command", "").startswith(("python ", "python3 ")):
                    path = item["command"].split()[1]
                    if not os.path.exists(path):
                        _warn(warnings, f"{pctx}: parity command path missing: {path}")


def is_meta_file(path: str) -> bool:
    """Check if a JSON file is a meta/index file (has _meta field)."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return "_meta" in data
    except Exception:
        return False


def validate_kernel_map(path: str, check_paths: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return [f"{path}: JSON error: {e}"], []

    # Skip meta/index files automatically
    if "_meta" in data:
        return [], []

    ctx = os.path.basename(path)
    for key in ("id", "op", "variant", "quant", "inputs", "outputs", "scratch", "dims", "impl"):
        if key not in data:
            _err(errors, f"{ctx}: missing '{key}'")
    if "id" in data and isinstance(data["id"], str):
        base = os.path.splitext(os.path.basename(path))[0]
        if base != data["id"]:
            _warn(warnings, f"{ctx}: file name '{base}' != id '{data['id']}'")

    quant = data.get("quant")
    if isinstance(quant, dict):
        for qk in ("weight", "activation", "output"):
            if qk not in quant or not isinstance(quant[qk], str):
                _err(errors, f"{ctx}: quant.{qk} must be string")
    else:
        _err(errors, f"{ctx}: quant must be object")

    dims = data.get("dims", [])
    if not isinstance(dims, list) or not all(isinstance(d, str) for d in dims):
        _err(errors, f"{ctx}: dims must be list of strings")

    for key in ("inputs", "outputs", "scratch"):
        buf_list = data.get(key, [])
        if not isinstance(buf_list, list):
            _err(errors, f"{ctx}: {key} must be list")
        else:
            for i, buf in enumerate(buf_list):
                _validate_buffer(buf, f"{ctx}: {key}[{i}]", errors, warnings, dims)

    if "parallelization" in data:
        _validate_parallelization(data["parallelization"], ctx, errors, warnings)

    if "constraints" in data:
        c = data["constraints"]
        if not isinstance(c, dict):
            _err(errors, f"{ctx}: constraints must be object")
        elif "requires" in c:
            _err(errors, f"{ctx}: constraints.requires is not allowed (use impl.variants.requires)")

    if "modes" in data:
        modes = data["modes"]
        if not isinstance(modes, dict):
            _err(errors, f"{ctx}: modes must be object")
        else:
            for mk in ("inference", "training", "backward"):
                if mk in modes and not _is_bool(modes[mk]):
                    _err(errors, f"{ctx}: modes.{mk} must be boolean")

    if "params" in data:
        params = data["params"]
        if not isinstance(params, list):
            _err(errors, f"{ctx}: params must be list")
        else:
            for i, p in enumerate(params):
                pctx = f"{ctx}: params[{i}]"
                if not isinstance(p, dict):
                    _err(errors, f"{pctx} must be object")
                    continue
                if "name" not in p or not isinstance(p["name"], str):
                    _err(errors, f"{pctx}: name required")
                if "dtype" not in p or not isinstance(p["dtype"], str):
                    _err(errors, f"{pctx}: dtype required")

    if "fuses" in data:
        if not _is_list_of_str_or_empty(data["fuses"]):
            _err(errors, f"{ctx}: fuses must be list of strings")

    if "impl" in data:
        _validate_impl(data["impl"], ctx, errors, warnings, check_paths)

    if "tests" in data:
        _validate_tests(data["tests"], ctx, errors, warnings, check_paths)

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate kernel map JSON files")
    parser.add_argument("--dir", default="version/v6.6/kernel_maps",
                        help="Kernel map directory (default: version/v6.6/kernel_maps)")
    parser.add_argument("--check-paths", action="store_true",
                        help="Check that referenced files exist")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show skipped files and validation progress")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[error] directory not found: {args.dir}")
        return 2

    all_json = sorted(f for f in os.listdir(args.dir) if f.endswith(".json"))

    # Separate kernel maps from meta/index files
    kernel_maps: List[str] = []
    skipped: List[Tuple[str, str]] = []  # (filename, reason)
    for fname in all_json:
        path = os.path.join(args.dir, fname)
        # Check _meta field first (more descriptive reason)
        if is_meta_file(path):
            skipped.append((fname, "has _meta field (index/catalog file)"))
        elif fname in SKIP_FILES:
            skipped.append((fname, "in SKIP_FILES"))
        else:
            kernel_maps.append(fname)

    if args.verbose:
        print(f"[info] found {len(all_json)} JSON files")
        print(f"[info] validating {len(kernel_maps)} kernel maps")

    if skipped:
        print(f"[skipped] {len(skipped)} meta/index file(s):")
        for fname, reason in skipped:
            print(f"  - {fname}: {reason}")

    if not kernel_maps:
        print("[error] no kernel maps found")
        return 2

    total_errors: List[str] = []
    total_warnings: List[str] = []
    for fname in kernel_maps:
        path = os.path.join(args.dir, fname)
        errors, warnings = validate_kernel_map(path, args.check_paths)
        total_errors.extend(errors)
        total_warnings.extend(warnings)

    if total_errors:
        print("[errors]")
        for e in total_errors:
            print(" -", e)
    if total_warnings:
        print("[warnings]")
        for w in total_warnings:
            print(" -", w)

    if total_errors or (args.strict and total_warnings):
        return 1

    print(f"[ok] {len(kernel_maps)} kernel maps valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
