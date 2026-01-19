#!/usr/bin/env python3
"""
Source scanner for kernel entrypoints.

This generates a source inventory JSON (KERNEL_SOURCES.json) by scanning
src/kernels/**/*.c. It does NOT replace the kernel maps registry; use
`gen_kernel_registry_from_maps.py` for the map-driven registry.

Usage:
  python3 version/v6.6/scripts/gen_kernel_registry.py
  python3 version/v6.6/scripts/gen_kernel_registry.py --output version/v6.6/kernel_maps/KERNEL_SOURCES.json
  python3 version/v6.6/scripts/gen_kernel_registry.py --check
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

from kernel_source_scan import scan_kernel_sources


def check_registry(registry_path: str, new_registry: Dict) -> Tuple[List[str], List[str]]:
    """Compare existing registry with newly generated one."""
    if not os.path.exists(registry_path):
        return ['Registry file does not exist'], []

    try:
        with open(registry_path, 'r') as f:
            existing = json.load(f)
    except Exception as e:
        return [f'Could not read existing registry: {e}'], []

    errors: List[str] = []
    warnings: List[str] = []

    new_counts = new_registry.get('_meta', {}).get('counts', {})
    old_counts = existing.get('_meta', {}).get('counts', {})

    if new_counts and old_counts:
        for key in new_counts:
            if key in old_counts and new_counts[key] != old_counts[key]:
                diff = new_counts[key] - old_counts[key]
                sign = '+' if diff > 0 else ''
                warnings.append(f"{key}: {old_counts[key]} → {new_counts[key]} ({sign}{diff})")
    else:
        warnings.append("Registry format differs; consider regenerating")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description='Scan kernel sources and generate a source registry')
    parser.add_argument('--output', '-o',
                        default='version/v6.6/kernel_maps/KERNEL_SOURCES.json',
                        help='Output path for source scan JSON')
    parser.add_argument('--check', action='store_true',
                        help='Check if existing output is up-to-date (do not write)')
    parser.add_argument('--root', default='src/kernels',
                        help='Kernel source root (default: src/kernels)')
    args = parser.parse_args()

    print(f'[scan] scanning {args.root} for kernel functions...')
    registry = scan_kernel_sources(root=args.root, generated_by=os.path.basename(__file__))

    counts = registry['_meta']['counts']
    print(f'[found] {counts["total"]} kernels:')
    print(f'  - inference:    {counts["inference"]}')
    print(f'  - training:     {counts["training"]}')
    print(f'  - optimizer:    {counts["optimizer"]}')
    print(f'  - fusion:       {counts["fusion"]}')
    print(f'  - quantization: {counts["quantization"]}')
    print(f'  - utility:      {counts["utility"]}')

    if args.check:
        print(f'\n[check] comparing with {args.output}...')
        errors, warnings = check_registry(args.output, registry)

        if errors:
            print('[errors]')
            for e in errors:
                print(f'  - {e}')
            return 1

        if warnings:
            print('[changes detected]')
            for w in warnings:
                print(f'  - {w}')
            print('\nRun without --check to update the source registry.')
            return 1

        print('[ok] source registry is up-to-date')
        return 0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(registry, f, indent=2)

    print(f'\n[ok] wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
