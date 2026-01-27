#!/usr/bin/env python3
"""
Update fusion patterns in KERNEL_REGISTRY.json to use GEMM kernels for prefill.

The issue: Fusion patterns with "prefill" in their names were using GEMV (decode) kernels.
This script updates them to use GEMM (prefill) kernels instead.
"""

import json
from pathlib import Path

def main():
    registry_path = Path(__file__).parent.parent / "kernel_maps" / "KERNEL_REGISTRY.json"

    print(f"Loading registry: {registry_path}")
    with open(registry_path, 'r') as f:
        registry = json.load(f)

    # Mapping from GEMV (decode) to GEMM (prefill)
    gemv_to_gemm = {
        "gemv_q5_0_q8_0": "gemm_nt_q5_0_q8_0",
        "gemv_q8_0_q8_0": "gemm_nt_q8_0_q8_0",
        "gemv_q4_k_q8_k": "gemm_nt_q4_k_q8_k",
        "gemv_q6_k_q8_k": "gemm_nt_q6_k_q8_k",
    }

    updates_made = []

    for kernel in registry['kernels']:
        if 'fuses' not in kernel:
            continue

        kernel_id = kernel['id']

        # Only update kernels with "prefill" in the name that use gemv
        if 'prefill' not in kernel_id.lower():
            continue

        fuses = kernel['fuses']
        has_gemv = any(k in gemv_to_gemm for k in fuses)

        if not has_gemv:
            print(f"✓ {kernel_id}: Already uses correct kernels")
            continue

        # Update fusion sequence
        new_fuses = []
        for k in fuses:
            if k in gemv_to_gemm:
                new_k = gemv_to_gemm[k]
                new_fuses.append(new_k)
                print(f"  {k} → {new_k}")
            else:
                new_fuses.append(k)

        kernel['fuses'] = new_fuses
        updates_made.append({
            'kernel': kernel_id,
            'old': fuses,
            'new': new_fuses
        })

    if not updates_made:
        print("\n✓ No updates needed - registry already correct")
        return

    # Save updated registry
    print(f"\n✓ Updating {len(updates_made)} fusion patterns")
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)

    print(f"\n✓ Updated registry: {registry_path}")

    # Show summary
    print("\n" + "="*80)
    print("FUSION PATTERN UPDATES")
    print("="*80)
    for update in updates_made:
        print(f"\n{update['kernel']}:")
        print(f"  Before: {len(update['old'])} kernels (GEMV - decode)")
        print(f"  After:  {len(update['new'])} kernels (GEMM - prefill)")

if __name__ == "__main__":
    main()
