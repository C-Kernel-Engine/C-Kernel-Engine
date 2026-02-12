#!/usr/bin/env python3
"""
Generate a kernel registry from v7 kernel maps (JSON).

Usage:
  python3 version/v7/scripts/gen_kernel_registry_from_maps.py
  python3 version/v7/scripts/gen_kernel_registry_from_maps.py --dir version/v7/kernel_maps \
      --output version/v7/kernel_maps/KERNEL_REGISTRY.json
  python3 version/v7/scripts/gen_kernel_registry_from_maps.py --check
  python3 version/v7/scripts/gen_kernel_registry_from_maps.py --validate --strict
"""

import argparse
import json
import os
from typing import Dict, List, Tuple


def _is_map_file(name: str) -> bool:
    if not name.endswith('.json'):
        return False
    upper = name.upper()
    if upper.startswith('KERNEL_'):
        return False
    return True


def load_kernel_maps(kernel_dir: str) -> List[Dict]:
    maps: List[Dict] = []
    for fname in sorted(os.listdir(kernel_dir)):
        if not _is_map_file(fname):
            continue
        path = os.path.join(kernel_dir, fname)
        with open(path, 'r') as f:
            data = json.load(f)
        entry = dict(data)
        entry.setdefault('id', data.get('id', fname[:-5]))
        entry.setdefault('name', entry['id'])
        entry['_source_file'] = fname
        maps.append(entry)
    return maps


def summarize_maps(maps: List[Dict]) -> Dict:
    by_op: Dict[str, int] = {}
    for k in maps:
        op = k.get('op', 'unknown')
        by_op[op] = by_op.get(op, 0) + 1
    return {
        'total': len(maps),
        'by_op': dict(sorted(by_op.items())),
    }


def check_registry(path: str, new_registry: Dict) -> Tuple[List[str], List[str]]:
    if not os.path.exists(path):
        return ['Registry file does not exist'], []

    try:
        with open(path, 'r') as f:
            existing = json.load(f)
    except Exception as e:
        return [f'Could not read existing registry: {e}'], []

    errors: List[str] = []
    warnings: List[str] = []

    new_ids = {k.get('name') for k in new_registry.get('kernels', [])}
    old_ids = {k.get('name') for k in existing.get('kernels', [])}

    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)

    if added:
        warnings.append(f'added: {len(added)}')
        warnings.extend([f'  + {x}' for x in added[:20]])
    if removed:
        warnings.append(f'removed: {len(removed)}')
        warnings.extend([f'  - {x}' for x in removed[:20]])

    return errors, warnings


def maybe_validate(kernel_dir: str, strict: bool) -> int:
    try:
        from validate_kernel_maps import validate_kernel_map
    except Exception:
        print('[warn] validate_kernel_maps import failed; skipping validation')
        return 0

    errors = []
    warnings = []
    for fname in sorted(os.listdir(kernel_dir)):
        if not _is_map_file(fname):
            continue
        path = os.path.join(kernel_dir, fname)
        e, w = validate_kernel_map(path, check_paths=False)
        errors.extend(e)
        warnings.extend(w)

    if errors:
        print('[errors]')
        for e in errors:
            print(' -', e)
    if warnings:
        print('[warnings]')
        for w in warnings:
            print(' -', w)

    if errors or (strict and warnings):
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate kernel registry from kernel maps')
    parser.add_argument('--dir', default='version/v7/kernel_maps',
                        help='Kernel map directory')
    parser.add_argument('--output', '-o',
                        default='version/v7/kernel_maps/KERNEL_REGISTRY.json',
                        help='Output path for registry JSON')
    parser.add_argument('--check', action='store_true',
                        help='Check if existing registry is up-to-date (do not write)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate kernel maps before generating registry')
    parser.add_argument('--strict', action='store_true',
                        help='Treat validation warnings as errors')
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f'[error] kernel map directory not found: {args.dir}')
        return 2

    if args.validate:
        rc = maybe_validate(args.dir, args.strict)
        if rc != 0:
            return rc

    maps = load_kernel_maps(args.dir)
    if not maps:
        print('[error] no kernel maps found')
        return 2

    summary = summarize_maps(maps)
    registry = {
        '_meta': {
            'description': 'Kernel registry generated from v7 kernel maps',
            'version': 'v7',
            'generated_by': os.path.basename(__file__),
            'source_dir': args.dir,
            'counts': summary,
        },
        'kernels': maps,
    }

    print(f'[maps] {summary["total"]} kernel maps loaded')

    if args.check:
        print(f'[check] comparing with {args.output}...')
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
            print('\nRun without --check to update the registry.')
            return 1

        print('[ok] registry is up-to-date')
        return 0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(registry, f, indent=2)

    print(f'[ok] wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
