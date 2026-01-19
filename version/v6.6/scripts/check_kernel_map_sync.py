#!/usr/bin/env python3
"""
Check kernel map IDs against kernel source implementations.

Usage:
  python3 version/v6.6/scripts/check_kernel_map_sync.py
  python3 version/v6.6/scripts/check_kernel_map_sync.py --strict
  python3 version/v6.6/scripts/check_kernel_map_sync.py --dir version/v6.6/kernel_maps --root src/kernels
"""

import argparse
import json
import os
import re
from typing import Dict, List, Set, Tuple

from kernel_source_scan import scan_function_names


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
        data['_source_file'] = fname
        maps.append(data)
    return maps


def filter_names(names: Set[str], ignore_patterns: List[str]) -> Set[str]:
    if not ignore_patterns:
        return names
    out = set()
    for n in names:
        if any(re.search(p, n) for p in ignore_patterns):
            continue
        out.add(n)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description='Check kernel maps vs source implementations')
    parser.add_argument('--dir', default='version/v6.6/kernel_maps',
                        help='Kernel map directory')
    parser.add_argument('--root', default='src/kernels',
                        help='Kernel source root')
    parser.add_argument('--strict', action='store_true',
                        help='Fail if any source kernel lacks a map')
    parser.add_argument('--ignore', action='append', default=[],
                        help='Regex patterns to ignore when checking missing maps')
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f'[error] kernel map directory not found: {args.dir}')
        return 2

    maps = load_kernel_maps(args.dir)
    map_ids = set()
    map_impls = []
    for m in maps:
        mid = m.get('id')
        if isinstance(mid, str):
            map_ids.add(mid)
        impl = m.get('impl', {})
        fn = impl.get('function') if isinstance(impl, dict) else None
        if isinstance(fn, str):
            map_impls.append((m.get('_source_file', '<unknown>'), fn))

    source_funcs = scan_function_names(root=args.root)
    source_funcs = filter_names(source_funcs, args.ignore)

    missing_in_source = sorted([fn for _, fn in map_impls if fn not in source_funcs])
    missing_in_maps = sorted(source_funcs - map_ids)

    if missing_in_source:
        print('[errors] map impls missing in source:')
        for fn in missing_in_source:
            print(' -', fn)

    if missing_in_maps:
        print('[warnings] source kernels missing maps:')
        for fn in missing_in_maps[:50]:
            print(' -', fn)
        if len(missing_in_maps) > 50:
            print(f' ... and {len(missing_in_maps) - 50} more')

    if missing_in_source:
        return 1
    if args.strict and missing_in_maps:
        return 1

    if not missing_in_source and not missing_in_maps:
        print('[ok] kernel maps and sources are in sync')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
