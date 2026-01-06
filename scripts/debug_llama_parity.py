#!/usr/bin/env python3
"""
debug_llama_parity.py
=====================

Compares activation dumps from hacked llama.cpp against C-Kernel-Engine parity dumps.

Usage:
    1. Run hacked llama.cpp to generate dumps in 'llama_dump/'
    2. Run C-Kernel-Engine with --parity to generate dumps in 'parity/'
    3. Run this script:
       python3 scripts/debug_llama_parity.py --llama llama_dump --ck parity --layer 0

"""

import argparse
import os
import struct
import sys
import numpy as np
from pathlib import Path

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Mapping from llama.cpp tensor names to C-Kernel-Engine parity names (and expected shapes if needed)
# NOTE: llama.cpp tensor names usually look like 'blk.0.attn_q'
# C-Kernel parity names usually look like 'layer_0_q_proj_tok0'
NAME_MAP = {
    # Layer 0
    "blk.0.attn_norm": "layer_0_ln1_out",
    "blk.0.attn_q": "layer_0_q_proj",
    "blk.0.attn_k": "layer_0_k_proj", 
    "blk.0.attn_v": "layer_0_v_proj",
    "blk.0.attn_output": "layer_0_attn_out", # After projection
    "blk.0.ffn_norm": "layer_0_ln2_out",
    "blk.0.ffn_gate": "layer_0_fc1_gate",    # If split
    "blk.0.ffn_up": "layer_0_fc1_up",        # If split
    # Note: llama.cpp usually fuses gate*up for SwiGLU, might be just 'ffn_gate' or similar depending on implementation
    "blk.0.ffn_down": "layer_0_mlp",
    
    # Common
    "result_norm": "final_out",
}

def load_bin(path, dtype=np.float32):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = f.read()
    return np.frombuffer(data, dtype=dtype)

def find_ck_file(ck_dir, base_name, tok_idx=0):
    """Find the C-Kernel dump file for a given base name."""
    # Try exact match first
    candidates = [
        f"{base_name}_tok{tok_idx}.bin",
        f"{base_name}.bin"
    ]
    
    for c in candidates:
        p = ck_dir / c
        if p.exists():
            return p
    
    # Try fuzzy match if specific mapping failed
    # e.g. if we are looking for 'layer_0_q_proj' but only have 'layer_0_q'
    for f in ck_dir.glob(f"*{base_name}*tok{tok_idx}.bin"):
        return f
        
    return None

def compare_tensors(name, llama_data, ck_data, tol=1e-4):
    if llama_data is None or ck_data is None:
        return False, 0.0, "Missing file"

    # Truncate to smaller size (alignment differences)
    size = min(llama_data.size, ck_data.size)
    l_trim = llama_data[:size]
    c_trim = ck_data[:size]

    diff = np.abs(l_trim - c_trim)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Relative error for large values
    max_val = np.max(np.abs(l_trim))
    rel_error = max_diff / (max_val + 1e-9)

    passed = max_diff < tol
    
    status_str = f"max_diff={max_diff:.2e} rel={rel_error:.2e} size={size}"
    
    # Check for NaN/Inf
    if np.isnan(l_trim).any():
        status_str += f" {RED}NaN in llama{RESET}"
        passed = False
    if np.isnan(c_trim).any():
        status_str += f" {RED}NaN in ck{RESET}"
        passed = False

    return passed, max_diff, status_str

def main():
    parser = argparse.ArgumentParser(description="Compare llama.cpp vs ck-engine tensors")
    parser.add_argument("--llama", required=True, help="Directory with llama.cpp dumps")
    parser.add_argument("--ck", required=True, help="Directory with ck-engine parity dumps")
    parser.add_argument("--layer", type=int, default=0, help="Layer to inspect")
    parser.add_argument("--tok", type=int, default=0, help="Token index to compare")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance")
    args = parser.parse_args()

    llama_dir = Path(args.llama)
    ck_dir = Path(args.ck)

    print(f"{BOLD}Comparing Layer {args.layer} (Token {args.tok}){RESET}")
    print(f"Llama: {llama_dir}")
    print(f"CK:    {ck_dir}\n")

    # Define check order - following data flow
    checkpoints = [
        # Input to layer (from prev layer or embedding)
        (f"blk.{args.layer}.attn_norm", f"layer_{args.layer}_ln1_out"),
        
        # Attention Projections
        (f"blk.{args.layer}.attn_q", f"layer_{args.layer}_q_proj"),
        (f"blk.{args.layer}.attn_k", f"layer_{args.layer}_k_proj"),
        (f"blk.{args.layer}.attn_v", f"layer_{args.layer}_v_proj"),
        
        # RoPE (llama.cpp modifies q/k in place often, or outputs to a buffer)
        # Note: matching RoPE output is tricky as llama.cpp might not dump the post-RoPE buffer distinctly
        # without extra hacks. We skip specific RoPE output unless we know the node name.
        
        # Attention Output
        (f"blk.{args.layer}.attn_output", f"layer_{args.layer}_attn_out"),
        
        # MLP
        (f"blk.{args.layer}.ffn_norm", f"layer_{args.layer}_ln2_out"),
        
        # SwiGLU inputs (Gate/Up)
        # Llama.cpp might fuse these or name them differently.
        # Check standard names:
        (f"blk.{args.layer}.ffn_gate", f"layer_{args.layer}_mlp_gate"), 
        (f"blk.{args.layer}.ffn_up", f"layer_{args.layer}_mlp_up"),
        
        # MLP Output
        (f"blk.{args.layer}.ffn_down", f"layer_{args.layer}_mlp_out"),
    ]

    diverged = False
    
    for llama_name, ck_base in checkpoints:
        # Load Llama
        l_path = llama_dir / f"{llama_name}.bin"
        l_data = load_bin(l_path)
        
        # Load CK
        c_path = find_ck_file(ck_dir, ck_base, args.tok)
        c_data = load_bin(c_path) if c_path else None

        # Format Name
        display_name = f"{llama_name:<25} vs {ck_base:<25}"
        
        if l_data is None:
            print(f"{YELLOW}[SKIP]{RESET} {display_name}: Llama file missing")
            continue
        
        if c_data is None:
            print(f"{YELLOW}[SKIP]{RESET} {display_name}: CK file missing")
            continue

        passed, max_diff, info = compare_tensors(display_name, l_data, c_data, args.tol)
        
        if passed:
            print(f"{GREEN}[PASS]{RESET} {display_name} | {info}")
        else:
            print(f"{RED}[FAIL]{RESET} {display_name} | {info}")
            if not diverged:
                print(f"\n{RED}>>> DIVERGENCE POINT DETECTED <<<{RESET}")
                # Print sample
                print("Llama sample (first 10):", l_data[:10])
                print("CK    sample (first 10):", c_data[:10])
                diverged = True

    if not diverged:
        print(f"\n{GREEN}Layer {args.layer} parity OK!{RESET}")
    else:
        print(f"\n{RED}Layer {args.layer} has divergence.{RESET}")

if __name__ == "__main__":
    main()
