#!/usr/bin/env python3
"""
analyze_parity.py - Analyze C-Kernel-Engine parity dumps

Usage:
    python scripts/analyze_parity.py /path/to/parity/dir
    python scripts/analyze_parity.py /path/to/parity/dir --layer 0
    python scripts/analyze_parity.py /path/to/parity/dir --compare-tokens
    python scripts/analyze_parity.py /path/to/parity/dir --logits
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def load_parity(path: str) -> np.ndarray:
    """Load a parity dump file."""
    return np.fromfile(path, dtype=np.float32)

def analyze_buffer(data: np.ndarray, name: str) -> dict:
    """Analyze a single buffer."""
    return {
        'name': name,
        'size': len(data),
        'mean': float(data.mean()),
        'std': float(data.std()),
        'min': float(data.min()),
        'max': float(data.max()),
        'has_nan': bool(np.isnan(data).any()),
        'has_inf': bool(np.isinf(data).any()),
        'zero_count': int((data == 0).sum()),
    }

def print_buffer_stats(stats: dict, warn_threshold: float = 100.0):
    """Print buffer statistics with warnings."""
    name = stats['name']

    # Warning flags
    flags = []
    if stats['has_nan']:
        flags.append(f"{RED}NaN{RESET}")
    if stats['has_inf']:
        flags.append(f"{RED}Inf{RESET}")
    if abs(stats['max']) > warn_threshold or abs(stats['min']) > warn_threshold:
        flags.append(f"{YELLOW}LARGE{RESET}")

    flag_str = ' '.join(flags) if flags else ''

    print(f"  {name:35s} size={stats['size']:6d} "
          f"mean={stats['mean']:10.4f} std={stats['std']:10.4f} "
          f"range=[{stats['min']:10.2f}, {stats['max']:10.2f}] {flag_str}")

def analyze_logits(parity_dir: Path, tokenizer_path: str = None):
    """Analyze logits and show top predictions."""
    print(f"\n{BOLD}=== Logits Analysis ==={RESET}\n")

    # Try to load tokenizer
    tokenizer = None
    if tokenizer_path and os.path.exists(tokenizer_path):
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(tokenizer_path)
        except:
            pass

    for tok_file in sorted(parity_dir.glob("logits_tok*.bin")):
        tok_idx = tok_file.stem.split('_tok')[1]
        logits = load_parity(str(tok_file))

        print(f"{CYAN}logits_tok{tok_idx}:{RESET}")
        print(f"  Shape: {logits.shape}, Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")
        print(f"  Range: [{logits.min():.4f}, {logits.max():.4f}]")

        # Top 5 predictions
        top5 = np.argsort(logits)[-5:][::-1]
        print(f"  Top 5 predictions:")
        for i, tid in enumerate(top5):
            decoded = ""
            if tokenizer:
                try:
                    decoded = f" -> '{tokenizer.decode([int(tid)])}'"
                except:
                    pass
            print(f"    {i+1}. token_id={tid:6d} logit={logits[tid]:8.4f}{decoded}")
        print()

def analyze_layer(parity_dir: Path, layer_id: int, tok_idx: int = 0):
    """Analyze a specific layer."""
    print(f"\n{BOLD}=== Layer {layer_id} (tok{tok_idx}) ==={RESET}\n")

    buffers = ['ln1_out', 'q_proj', 'k_proj', 'v_proj', 'q_rope', 'k_rope',
               'attn', 'attn_proj', 'residual1', 'ln2_out', 'fc1', 'swiglu',
               'mlp', 'output']

    for buf_name in buffers:
        path = parity_dir / f"layer_{layer_id}_{buf_name}_tok{tok_idx}.bin"
        if path.exists():
            data = load_parity(str(path))
            stats = analyze_buffer(data, buf_name)
            print_buffer_stats(stats)

def compare_tokens(parity_dir: Path):
    """Compare tok0 vs tok1 across all layers."""
    print(f"\n{BOLD}=== Token Comparison (tok0 vs tok1) ==={RESET}\n")

    # Find all tok0 files
    tok0_files = sorted(parity_dir.glob("*_tok0.bin"))

    print(f"{'Buffer':<40s} {'tok0_std':>10s} {'tok1_std':>10s} {'ratio':>8s} {'tok0_max':>10s} {'tok1_max':>10s}")
    print("-" * 90)

    for tok0_file in tok0_files:
        name = tok0_file.stem.replace('_tok0', '')
        tok1_file = parity_dir / f"{name}_tok1.bin"

        if tok1_file.exists():
            t0 = load_parity(str(tok0_file))
            t1 = load_parity(str(tok1_file))

            t0_std = t0.std()
            t1_std = t1.std()
            ratio = t1_std / t0_std if t0_std > 0 else 0

            # Flag suspicious ratios
            flag = ""
            if ratio > 5 or ratio < 0.2:
                flag = f" {YELLOW}!{RESET}"

            print(f"{name:<40s} {t0_std:>10.4f} {t1_std:>10.4f} {ratio:>8.2f} "
                  f"{abs(t0).max():>10.2f} {abs(t1).max():>10.2f}{flag}")

def main():
    parser = argparse.ArgumentParser(description="Analyze C-Kernel-Engine parity dumps")
    parser.add_argument('parity_dir', help='Path to parity directory')
    parser.add_argument('--layer', type=int, help='Analyze specific layer')
    parser.add_argument('--tok', type=int, default=0, help='Token index (default: 0)')
    parser.add_argument('--compare-tokens', action='store_true', help='Compare tok0 vs tok1')
    parser.add_argument('--logits', action='store_true', help='Analyze logits')
    parser.add_argument('--tokenizer', help='Path to tokenizer.json for decoding')
    parser.add_argument('--all', action='store_true', help='Show all buffers')

    args = parser.parse_args()

    parity_dir = Path(args.parity_dir)
    if not parity_dir.exists():
        print(f"{RED}Error: {parity_dir} does not exist{RESET}")
        sys.exit(1)

    # Auto-detect tokenizer
    tokenizer_path = args.tokenizer
    if not tokenizer_path:
        parent = parity_dir.parent
        if (parent / "tokenizer.json").exists():
            tokenizer_path = str(parent / "tokenizer.json")

    print(f"{BOLD}Parity Directory: {parity_dir}{RESET}")
    print(f"Files found: {len(list(parity_dir.glob('*.bin')))}")

    if args.logits:
        analyze_logits(parity_dir, tokenizer_path)
    elif args.compare_tokens:
        compare_tokens(parity_dir)
    elif args.layer is not None:
        analyze_layer(parity_dir, args.layer, args.tok)
    elif args.all:
        # Show all buffers for specified token
        for f in sorted(parity_dir.glob(f"*_tok{args.tok}.bin")):
            data = load_parity(str(f))
            stats = analyze_buffer(data, f.stem)
            print_buffer_stats(stats)
    else:
        # Default: show summary
        print(f"\n{BOLD}=== Summary ==={RESET}\n")

        # Count layers
        layer_files = list(parity_dir.glob("layer_*_output_tok0.bin"))
        print(f"Layers found: {len(layer_files)}")

        # Show logits summary
        analyze_logits(parity_dir, tokenizer_path)

        print(f"\n{CYAN}Usage examples:{RESET}")
        print(f"  python scripts/analyze_parity.py {parity_dir} --logits")
        print(f"  python scripts/analyze_parity.py {parity_dir} --layer 0")
        print(f"  python scripts/analyze_parity.py {parity_dir} --compare-tokens")
        print(f"  python scripts/analyze_parity.py {parity_dir} --all --tok 1")

if __name__ == "__main__":
    main()
