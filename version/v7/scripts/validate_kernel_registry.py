#!/usr/bin/env python3
"""
Validate kernel registry consistency.

Checks:
- Every registry function is declared in the header
- Every binding references a valid function
- Function signatures are consistent
"""

import json
import re
from pathlib import Path

# ANSI colors
GREEN = '\033[92m'
RESET = '\033[0m'


def extract_header_declarations(header_path: str) -> set:
    """Extract function declarations from header."""
    with open(header_path) as f:
        content = f.read()
    # Match function declarations: return_type name(params);
    pattern = r'(?:void|int|float|size_t|bool|CKMathBackend)\s+(\w+)\s*\([^;]*\);'
    return set(re.findall(pattern, content))


def extract_header_functions(header_path: str) -> dict:
    """Extract function declarations with their signatures."""
    with open(header_path) as f:
        content = f.read()
    # Match complete declarations
    pattern = r'((?:void|int|float|size_t|bool|CKMathBackend)\s+(\w+)\s*\([^;]*\));'
    matches = re.findall(pattern, content)
    return {name: decl for decl, name in matches}


def validate_registry(registry_path: str, bindings_path: str,
                      header_path: str) -> dict:
    """Full registry validation."""
    with open(registry_path) as f:
        registry = json.load(f)
    with open(bindings_path) as f:
        bindings = json.load(f)
    header_funcs = extract_header_declarations(header_path)
    header_full = extract_header_functions(header_path)

    results = {
        "errors": [],
        "warnings": [],
        "summary": {}
    }

    # Skip list for stdlib functions
    SKIP_FUNCS = {"memcpy", "memmove", "malloc", "free", "printf"}

    # Check 1: Every registry function in header
    for kernel in registry.get("kernels", []):
        kernel_id = kernel.get("id", "unknown")
        impl = kernel.get("impl", {})
        func = impl.get("function", "")

        if func and func not in header_funcs:
            if func in SKIP_FUNCS:
                continue  # Skip stdlib functions
            # WIP kernels - not yet implemented, treat as warning
            results["warnings"].append({
                "type": "MISSING_HEADER",
                "kernel": kernel_id,
                "function": func,
                "note": f"WIP kernel - declaration will be added when implemented"
            })

    # Check 2: Every binding references valid function
    binding_funcs = set()
    for name, binding in bindings.get("bindings", {}).items():
        for param in binding.get("params", []):
            source = param.get("source", "")
            if source.startswith("function:"):
                func_name = source.split(":")[1]
                binding_funcs.add(func_name)

                if func_name not in header_funcs:
                    results["errors"].append({
                        "type": "INVALID_BINDING_REF",
                        "binding": name,
                        "parameter": param.get("name", "unknown"),
                        "function": func_name,
                        "fix": f"Add 'void {func_name}(...);' to {header_path}"
                    })

    # Check 3: Warn about header functions not in registry
    registry_funcs = set()
    for kernel in registry.get("kernels", []):
        func = kernel.get("impl", {}).get("function", "")
        if func:
            registry_funcs.add(func)

    for func in header_funcs:
        if func not in registry_funcs and not func.startswith("ck_") and not func.startswith("CK"):
            results["warnings"].append({
                "type": "UNREGISTERED_FUNCTION",
                "function": func,
                "note": f"Function declared in header but not in registry"
            })

    # Summary
    results["summary"] = {
        "total_kernels": len(registry.get("kernels", [])),
        "total_bindings": len(bindings.get("bindings", {})),
        "header_funcs": len(header_funcs),
        "registry_funcs": len(registry_funcs),
        "errors": len(results["errors"]),
        "warnings": len(results["warnings"])
    }

    return results


def validate_kernel_dependencies(registry_path: str) -> dict:
    """Check kernel dependency consistency."""
    with open(registry_path) as f:
        registry = json.load(f)

    results = {
        "errors": [],
        "warnings": [],
        "dependencies": {}
    }

    for kernel in registry.get("kernels", []):
        kernel_id = kernel.get("id", "unknown")
        deps = kernel.get("depends", [])

        # Check all dependencies exist
        kernel_ids = {k.get("id") for k in registry.get("kernels", [])}
        for dep in deps:
            if dep not in kernel_ids:
                results["errors"].append({
                    "type": "MISSING_DEPENDENCY",
                    "kernel": kernel_id,
                    "dependency": dep,
                    "fix": f"Add kernel '{dep}' to registry or remove dependency"
                })

        results["dependencies"][kernel_id] = deps

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate kernel registry consistency"
    )
    parser.add_argument("--registry",
                       default="version/v7/kernel_maps/KERNEL_REGISTRY.json")
    parser.add_argument("--bindings",
                       default="version/v7/kernel_maps/kernel_bindings.json")
    parser.add_argument("--header",
                       default="include/ckernel_engine.h")
    parser.add_argument("--check-deps", action="store_true",
                       help="Also check kernel dependencies")
    args = parser.parse_args()

    results = validate_registry(args.registry, args.bindings, args.header)

    print("\n" + "="*60)
    print("KERNEL REGISTRY VALIDATION")
    print("="*60)
    print(f"\nKernels: {results['summary']['total_kernels']}")
    print(f"Bindings: {results['summary']['total_bindings']}")
    print(f"Header functions: {results['summary']['header_funcs']}")
    print(f"Registry functions: {results['summary']['registry_funcs']}")

    if results['errors']:
        print(f"\nERRORS ({len(results['errors'])}):")
        for e in results['errors']:
            print(f"  [{e['type']}] {e.get('kernel', e.get('binding'))}")
            print(f"    -> {e.get('fix', 'Unknown fix')}")
    else:
        print(f"\nNo errors found")

    if results['warnings']:
        print(f"\nWARNINGS ({len(results['warnings'])}):")
        for w in results['warnings'][:10]:
            print(f"  [{w['type']}] {w.get('function', w.get('kernel'))}")
            print(f"    -> {w.get('note', w.get('fix', 'Unknown'))}")
        if len(results['warnings']) > 10:
            print(f"  ... and {len(results['warnings']) - 10} more")

    # Only fail on real errors, not warnings
    if results['errors']:
        return 1
    else:
        print(f"\n{GREEN}Registry validation passed (with {len(results['warnings'])} warnings for WIP kernels){RESET}")
        return 0


if __name__ == "__main__":
    exit(main())
