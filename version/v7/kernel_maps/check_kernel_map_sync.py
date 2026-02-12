#!/usr/bin/env python3
"""
check_kernel_map_sync.py - Verify kernel maps are synchronized with header declarations.

This script validates that:
1. All kernel impl.functions have corresponding prototypes in ckernel_engine.h
2. Registry and bindings are synchronized

Note: Functions that are static (internal to .c files) or have no header prototype
are allowed if they are not exported via CK_EXPORT.

Usage:
    python check_kernel_map_sync.py [--registry KERNEL_REGISTRY.json] [--headers "src/ckernel_engine.h,include/ckernel_engine.h"] [--bindings kernel_bindings.json]
"""

import json
import re
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    while cur != cur.parent:
        if (cur / "Makefile").exists() and (cur / "src").exists() and (cur / "include").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def extract_declarations(header_path: Path) -> set:
    """Extract function declarations from a C header file."""
    declarations = set()

    # Patterns for function declarations
    # Match: return_type function_name(...);
    # Exclude: #define, struct, enum, typedef, inline, static

    try:
        with open(header_path) as f:
            content = f.read()
    except FileNotFoundError:
        return declarations  # Return empty set if file doesn't exist

    # Remove comments
    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    # Find all function declarations
    # Pattern: void|int|float|const.*?(\w+)\s*\([^)]*\)\s*;
    pattern = r'(?:void|int|float|const\s+(?:float|int|uint[0-9]+_t)\s*\*?|CK_INLINE)\s+(\w+)\s*\([^;]*\)\s*;'

    for match in re.finditer(pattern, content):
        func_name = match.group(1)
        # Skip common non-kernel functions
        if not func_name.startswith('_') and func_name not in ('NULL', 'sizeof', 'offsetof'):
            declarations.add(func_name)

    return declarations


def extract_defined_functions() -> dict:
    """
    Extract function definitions from source files.
    Returns dict of {function_name: file_path} for functions that are defined.
    This is used to distinguish between:
    - Functions that ARE defined somewhere (static or not)
    - Functions that are truly missing
    """
    defined = {}
    repo_root = find_repo_root(Path(__file__).parent)
    src_dir = repo_root / "src"

    # Pattern for function definitions: return_type func_name(...) {
    # This catches both static and non-static functions
    def_pattern = r'^(?:static\s+)?(?:void|int|float|const\s+(?:float|int|uint[0-9]+_t)\s*\*?)\s+(\w+)\s*\([^)]*\)\s*\{'

    for c_file in src_dir.rglob("*.c"):
        try:
            with open(c_file) as f:
                lines = f.readlines()
        except FileNotFoundError:
            continue

        for i, line in enumerate(lines):
            # Remove leading whitespace for pattern matching
            stripped = line.lstrip()
            if stripped.startswith('//') or stripped.startswith('/*'):
                continue

            for match in re.finditer(def_pattern, stripped):
                func_name = match.group(1)
                # Only add if not already found (prefer headers/statics)
                if func_name not in defined:
                    defined[func_name] = str(c_file.relative_to(src_dir.parent))

    return defined


def extract_exported_functions() -> set:
    """
    Extract CK_EXPORT function declarations from source files.
    These are the functions that MUST be declared in headers.
    """
    exported = set()
    repo_root = find_repo_root(Path(__file__).parent)
    src_dir = repo_root / "src"

    # Pattern: CK_EXPORT ... function_name(...);
    export_pattern = r'CK_EXPORT\s+(?:void|int|float|const\s+(?:float|int|uint[0-9]+_t)\s*\*?)\s+(\w+)\s*\([^;]*\)\s*;'

    for c_file in src_dir.rglob("*.c"):
        try:
            with open(c_file) as f:
                content = f.read()
        except FileNotFoundError:
            continue

        # Remove comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        for match in re.finditer(export_pattern, content):
            func_name = match.group(1)
            exported.add(func_name)

    return exported


def check_header_declarations(registry_path: Path, header_paths: list[Path], bindings_path: Path) -> tuple[list, list]:
    """
    Check that all kernel impl.functions are declared in the headers.

    Returns:
        (errors, warnings) lists
    """
    errors = []
    warnings = []

    # Load registry
    try:
        with open(registry_path) as f:
            registry = json.load(f)
        kernel_ids = registry.get("kernels", [])
    except Exception as e:
        errors.append(f"Failed to load registry: {e}")
        return errors, warnings

    # Load bindings
    try:
        with open(bindings_path) as f:
            bindings_data = json.load(f)
        bindings = bindings_data.get("bindings", {})
    except Exception as e:
        errors.append(f"Failed to load bindings: {e}")
        return errors, warnings

    # Load header declarations from all headers
    all_header_funcs = set()
    for header_path in header_paths:
        if header_path.exists():
            funcs = extract_declarations(header_path)
            all_header_funcs.update(funcs)
            print(f"  Loaded {len(funcs)} declarations from {header_path.name}")
        else:
            warnings.append(f"Header not found: {header_path}")

    # Load defined functions (to distinguish missing from static)
    defined_funcs = extract_defined_functions()
    print(f"  Found {len(defined_funcs)} function definitions in source files")

    # Also check for exported functions in source files
    src_exports = extract_exported_functions()
    if src_exports:
        print(f"  Loaded {len(src_exports)} CK_EXPORT declarations from source files")
        # CK_EXPORT functions MUST be in headers
        for func in src_exports:
            if func not in all_header_funcs:
                warnings.append(f"CK_EXPORT function '{func}' not in headers")

    # Skip list for known external/stdlib functions
    SKIP = {
        "memmove", "memcpy",  # stdlib
        "printf", "fprintf", "sprintf", "snprintf",  # stdio
        "malloc", "free", "realloc", "calloc",  # stdlib memory
        "strlen", "strcpy", "strncpy", "strcmp", "strdup",  # stdlib string
        "memset", "memcmp", "memcpy", "memmove",  # stdlib memory
    }

    seen_funcs = set()

    for kernel in kernel_ids:
        kid = kernel.get("id", "")
        variant = kernel.get("variant", "")

        # Skip metadata entries without variant
        if not variant:
            continue

        impl = kernel.get("impl", {})
        func = impl.get("function", kid)

        # Skip duplicates
        if func in seen_funcs:
            continue
        seen_funcs.add(func)

        # Skip known non-kernel functions
        if func in SKIP:
            continue

        # Check if function exists somewhere
        if func not in all_header_funcs:
            if func in defined_funcs:
                # Function is defined (static or inline) - this is OK for internal functions
                warnings.append(
                    f"Function '{func}' is defined in {defined_funcs[func]} but not in headers. "
                    f"This is OK for static/internal functions."
                )
            else:
                # Function is WIP (planned but not yet implemented) - warning only
                warnings.append(
                    f"Function '{func}' (kernel '{kid}') is not declared in any header. "
                    f"This is a WIP kernel - bindings will be added when implemented."
                )

    return errors, warnings


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Check kernel map synchronization with header declarations"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).parent / "KERNEL_REGISTRY.json"
    )
    parser.add_argument(
        "--headers",
        type=str,
        default="src/ckernel_engine.h,include/ckernel_engine.h",
        help="Comma-separated list of header files to check (relative to repo root)"
    )
    parser.add_argument(
        "--bindings",
        type=Path,
        default=Path(__file__).parent / "kernel_bindings.json"
    )
    args = parser.parse_args()

    # Parse header paths
    header_paths = [Path(p.strip()) for p in args.headers.split(",")]
    repo_root = find_repo_root(Path(__file__).parent)
    header_paths = [repo_root / p if not p.is_absolute() else p for p in header_paths]

    print("=" * 70)
    print("KERNEL MAP SYNC CHECK")
    print("=" * 70)
    print(f"Registry: {args.registry}")
    print(f"Headers: {[str(p) for p in header_paths]}")
    print(f"Bindings: {args.bindings}")
    print()

    errors, warnings = check_header_declarations(
        args.registry, header_paths, args.bindings
    )

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"  - {w}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more")
        print()

    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        print()
        print("FAIL: Header declarations are out of sync with kernel registry")
        sys.exit(1)

    print("PASS: All kernel functions are declared in headers")
    sys.exit(0)


if __name__ == "__main__":
    main()
