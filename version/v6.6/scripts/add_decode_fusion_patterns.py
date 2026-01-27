#!/usr/bin/env python3
"""
Add separate decode-mode fusion patterns to KERNEL_REGISTRY.json.

The issue: We have fusion patterns for prefill (using GEMM) but decode mode needs
separate patterns using GEMV. We need BOTH sets of patterns.

Solution: Create duplicate fusion kernel entries with _decode suffix.
"""

import json
from pathlib import Path

def main():
    registry_path = Path(__file__).parent.parent / "kernel_maps" / "KERNEL_REGISTRY.json"

    print(f"Loading registry: {registry_path}")
    with open(registry_path, 'r') as f:
        registry = json.load(f)

    # Mapping from GEMM (prefill) back to GEMV (decode)
    gemm_to_gemv = {
        "gemm_nt_q5_0_q8_0": "gemv_q5_0_q8_0",
        "gemm_nt_q8_0_q8_0": "gemv_q8_0_q8_0",
        "gemm_nt_q4_k_q8_k": "gemv_q4_k_q8_k",
        "gemm_nt_q6_k_q8_k": "gemv_q6_k_q8_k",
    }

    new_kernels = []

    for kernel in registry['kernels']:
        if 'fuses' not in kernel:
            continue

        kernel_id = kernel['id']

        # Only process kernels with "prefill" in the name
        if 'prefill' not in kernel_id.lower():
            continue

        fuses = kernel['fuses']
        has_gemm = any(k in gemm_to_gemv for k in fuses)

        if not has_gemm:
            continue

        # Create decode version
        decode_id = kernel_id.replace('_prefill', '_decode')

        # Convert GEMM → GEMV for decode
        decode_fuses = []
        for k in fuses:
            if k in gemm_to_gemv:
                decode_fuses.append(gemm_to_gemv[k])
            else:
                decode_fuses.append(k)

        # Create new kernel entry for decode
        decode_kernel = kernel.copy()
        decode_kernel['id'] = decode_id
        decode_kernel['fuses'] = decode_fuses

        # Update impl if present
        if 'impl' in decode_kernel:
            impl = decode_kernel['impl'].copy()
            impl['function'] = impl.get('function', kernel_id).replace('_prefill', '_decode')
            decode_kernel['impl'] = impl

        new_kernels.append(decode_kernel)
        print(f"\n✓ Created decode version: {decode_id}")
        print(f"  Prefill fuses: {len(fuses)} kernels (GEMM)")
        print(f"  Decode fuses:  {len(decode_fuses)} kernels (GEMV)")

    if not new_kernels:
        print("\n✓ No new decode patterns needed")
        return

    # Add new kernels to registry
    registry['kernels'].extend(new_kernels)

    # Update counts
    registry['_meta']['counts']['total'] += len(new_kernels)

    # Save updated registry
    print(f"\n✓ Adding {len(new_kernels)} decode fusion patterns")
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)

    print(f"\n✓ Updated registry: {registry_path}")

    # Show summary
    print("\n" + "="*80)
    print("DECODE FUSION PATTERNS ADDED")
    print("="*80)
    for kernel in new_kernels:
        print(f"\n{kernel['id']}:")
        print(f"  Fuses {len(kernel['fuses'])} kernels:")
        for i, k in enumerate(kernel['fuses'][:3]):
            print(f"    {i}: {k}")
        if len(kernel['fuses']) > 3:
            print(f"    ... ({len(kernel['fuses']) - 3} more)")

if __name__ == "__main__":
    main()
