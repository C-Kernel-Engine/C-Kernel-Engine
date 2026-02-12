#!/usr/bin/env python3
"""
compare_activations.py - Compare CK vs llama.cpp activation dumps

Usage:
    python version/v7/scripts/parity/compare_activations.py [--ck-dir DIR] [--llama-dir DIR] [--tol TOL]

Example:
    python version/v7/scripts/parity/compare_activations.py \
        --ck-dir ~/.cache/ck-engine-v7/models/.../ck_build/ck_parity_dumps \
        --llama-dir ./layer_dumps
"""

import numpy as np
from pathlib import Path
import argparse
import sys

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'


def load_bin(path, dtype=np.float32):
    """Load binary file as numpy array."""
    return np.fromfile(path, dtype=dtype)


def compare_tensor(name, ck_data, llama_data, tol=1e-3, verbose=True):
    """Compare two tensors and return comparison stats."""
    result = {
        'name': name,
        'pass': False,
        'max_diff': float('inf'),
        'mean_diff': float('inf'),
        'rms_diff': float('inf'),
        'first_div_idx': -1,
        'ck_shape': None,
        'llama_shape': None,
    }

    if ck_data is None:
        result['error'] = 'CK data missing'
        return result
    if llama_data is None:
        result['error'] = 'llama.cpp data missing'
        return result

    result['ck_shape'] = ck_data.shape
    result['llama_shape'] = llama_data.shape

    if ck_data.shape != llama_data.shape:
        result['error'] = f'Shape mismatch: CK {ck_data.shape} vs llama {llama_data.shape}'
        return result

    # Compute differences
    diff = np.abs(ck_data - llama_data)
    result['max_diff'] = float(np.max(diff))
    result['mean_diff'] = float(np.mean(diff))
    result['rms_diff'] = float(np.sqrt(np.mean(diff ** 2)))

    # Relative error (avoid div by zero)
    llama_max = np.max(np.abs(llama_data))
    if llama_max > 1e-10:
        result['rel_error'] = result['max_diff'] / llama_max
    else:
        result['rel_error'] = 0.0

    # Find first divergence
    div_mask = diff > tol
    if np.any(div_mask):
        result['first_div_idx'] = int(np.argmax(div_mask))
    else:
        result['first_div_idx'] = -1

    result['pass'] = result['max_diff'] < tol

    return result


def print_result(result, verbose=True):
    """Print comparison result."""
    name = result['name']

    if 'error' in result:
        print(f"  {name}: {YELLOW}ERROR{RESET} - {result['error']}")
        return

    status = "PASS" if result['pass'] else "FAIL"
    color = GREEN if result['pass'] else RED

    print(f"  {name}: {color}{status}{RESET} "
          f"max={result['max_diff']:.2e} "
          f"mean={result['mean_diff']:.2e} "
          f"rms={result['rms_diff']:.2e}")

    if verbose and not result['pass']:
        if result['first_div_idx'] >= 0:
            print(f"    First divergence at index {result['first_div_idx']}")


def main():
    parser = argparse.ArgumentParser(description='Compare CK vs llama.cpp activations')
    parser.add_argument('--ck-dir', type=Path, default=Path('ck_parity_dumps'),
                        help='Directory with CK dumps')
    parser.add_argument('--llama-dir', type=Path, default=Path('layer_dumps'),
                        help='Directory with llama.cpp dumps')
    parser.add_argument('--tol', type=float, default=1e-3,
                        help='Tolerance for comparison (default: 1e-3)')
    parser.add_argument('--layer', type=int, default=0,
                        help='Layer to compare (default: 0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()

    # Define checkpoint mappings: (CK filename pattern, llama.cpp filename pattern)
    # Format: (ck_name, llama_name, size_or_None)
    checkpoints = [
        # Embedding
        ('embedding', 'inp_embd', None),

        # Layer N attention
        (f'L{args.layer}_attn_norm', f'attn_norm-{args.layer}', None),
        (f'L{args.layer}_q_proj', f'Qcur-{args.layer}', None),
        (f'L{args.layer}_k_proj', f'Kcur-{args.layer}', None),
        (f'L{args.layer}_v_proj', f'Vcur-{args.layer}', None),
        (f'L{args.layer}_q_post_norm', f'Qcur-{args.layer}', None),  # Post QK-norm
        (f'L{args.layer}_k_post_norm', f'Kcur-{args.layer}', None),
        (f'L{args.layer}_q_post_rope', f'Qcur_rope-{args.layer}', None),
        (f'L{args.layer}_k_post_rope', f'Kcur_rope-{args.layer}', None),
        (f'L{args.layer}_attn_out', f'attn_out-{args.layer}', None),
        (f'L{args.layer}_out_proj', f'inpSA-{args.layer}', None),  # Post out_proj

        # Layer N FFN
        (f'L{args.layer}_ffn_norm', f'ffn_norm-{args.layer}', None),
        (f'L{args.layer}_mlp_gate_up', f'ffn_gate-{args.layer}', None),
        (f'L{args.layer}_geglu_out', f'ffn_up-{args.layer}', None),
        (f'L{args.layer}_mlp_down', f'ffn_down-{args.layer}', None),
        (f'L{args.layer}_post_ffn', f'inpFF-{args.layer}', None),
    ]

    print("=" * 70)
    print(f"Activation Parity: CK vs llama.cpp (Layer {args.layer})")
    print(f"CK dir:    {args.ck_dir}")
    print(f"llama dir: {args.llama_dir}")
    print(f"Tolerance: {args.tol}")
    print("=" * 70)

    all_pass = True
    first_fail = None
    results = []

    for ck_name, llama_name, _ in checkpoints:
        ck_path = args.ck_dir / f'{ck_name}.bin'
        llama_path = args.llama_dir / f'{llama_name}.bin'

        # Load data
        ck_data = load_bin(ck_path) if ck_path.exists() else None
        llama_data = load_bin(llama_path) if llama_path.exists() else None

        # Compare
        result = compare_tensor(ck_name, ck_data, llama_data, tol=args.tol)
        results.append(result)

        # Print
        print_result(result, verbose=args.verbose)

        # Track first failure
        if not result['pass'] and first_fail is None:
            first_fail = ck_name
        all_pass = all_pass and result['pass']

    print("=" * 70)
    if all_pass:
        print(f"{GREEN}All checkpoints PASS{RESET}")
    else:
        print(f"{RED}First failure: {first_fail}{RESET}")
        print(f"\nInvestigate the op that produces '{first_fail}'")
        print(f"Use: CK_STOP_OP=N to stop before/after that op")

    # Summary stats
    passed = sum(1 for r in results if r['pass'])
    total = len(results)
    print(f"\nSummary: {passed}/{total} checkpoints passed")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
