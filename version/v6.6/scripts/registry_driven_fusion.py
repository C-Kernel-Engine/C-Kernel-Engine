#!/usr/bin/env python3
"""
registry_driven_fusion.py - Auto-generate fusion patterns from kernel registry.

FUSION PATTERN GENERATION:
  1. Load KERNEL_REGISTRY.json
  2. Find entries with "fuses" field
  3. Convert to fusion pattern format
  4. Merge with manual patterns for non-fused ops

FUSION MATCHING (in apply_fusion_pass):
  - Match sequences by op["kernel"] (kernel ID) from lowered IR
  - Replace sequence with fused kernel
  - Record in fusion report
"""

from typing import Dict, List, Optional, Any


def build_fusion_patterns_from_registry(registry: Dict) -> List[Dict]:
    """
    Auto-generate fusion patterns from registry entries with "fuses" field.

    Example registry entry:
        {
            "id": "mega_fused_attention_prefill",
            "op": "fused_attention_block",
            "fuses": ["rmsnorm_forward", "gemv_q5_0_q8_0", ...]
        }

    Returns fusion pattern:
        {
            "name": "mega_fused_attention_prefill",
            "priority": 100,
            "mode": ["prefill", "decode"],
            "sequence": ["rmsnorm_forward", "gemv_q5_0_q8_0", ...],  # Kernel IDs to match
            "fused_op": "fused_attention_block",
            "fused_kernel": "mega_fused_attention_prefill",
            "description": "Auto-generated from registry"
        }
    """
    patterns = []

    for kernel_entry in registry.get("kernels", []):
        if "fuses" not in kernel_entry:
            continue

        # Extract fusion info
        kernel_id = kernel_entry.get("id")
        op_type = kernel_entry.get("op", "")
        fuses = kernel_entry.get("fuses", [])

        # Skip if no fuses defined
        if not fuses:
            continue

        # Determine mode from kernel ID or fuses
        mode = ["prefill", "decode"]
        if "prefill" in kernel_id.lower():
            mode = ["prefill"]
        elif "decode" in kernel_id.lower():
            mode = ["decode"]

        # Build pattern
        pattern = {
            "name": kernel_id,
            "priority": 100,  # Fused kernels have highest priority
            "mode": mode,
            "sequence": fuses,  # Kernel IDs to match (not op names!)
            "fused_op": op_type,
            "fused_kernel": kernel_id,
            "remove_buffers": [],  # Inferred during fusion
            "description": f"Auto-generated from registry: fuses {len(fuses)} ops",
        }

        patterns.append(pattern)

    return patterns


def merge_registry_and_manual_patterns(
    registry_patterns: List[Dict],
    manual_patterns: List[Dict]
) -> List[Dict]:
    """
    Merge auto-generated registry patterns with manual patterns.

    Registry patterns (fused kernels) get highest priority.
    Manual patterns cover ops without registry fusion.
    """
    # Start with registry patterns (exact kernel matches, highest priority)
    combined = list(registry_patterns)

    # Add manual patterns for ops not covered by registry
    registry_kernel_ids = {p["fused_kernel"] for p in registry_patterns}

    for pattern in manual_patterns:
        fused_kernel = pattern.get("fused_kernel")
        if fused_kernel not in registry_kernel_ids:
            combined.append(pattern)

    # Sort by priority (highest first)
    return sorted(combined, key=lambda x: -x.get("priority", 0))


def find_fusion_candidates(
    lowered_ops: List[Dict[str, Any]],
    pattern: Dict[str, Any]
) -> Optional[int]:
    """
    Find fusion candidates in lowered IR ops by matching kernel sequences.

    Args:
        lowered_ops: List of ops from lowered IR, each with "kernel" field
        pattern: Fusion pattern with "sequence" (list of kernel IDs to match)

    Returns:
        Start index of matching sequence, or None if no match

    Example:
        ops = [
            {"name": "ln1", "kernel": "rmsnorm_forward"},
            {"name": "qkv", "kernel": "ck_qkv_project_head_major_quant"},
            ...
        ]
        pattern = {
            "sequence": ["rmsnorm_forward", "ck_qkv_project_head_major_quant", ...]
        }

        find_fusion_candidates(ops, pattern) -> 0 (match starts at index 0)
    """
    sequence = pattern.get("sequence", [])
    if not sequence:
        return None

    # Slide window over ops to find matching kernel sequence
    for start_idx in range(len(lowered_ops) - len(sequence) + 1):
        match = True
        for seq_idx, kernel_id in enumerate(sequence):
            op_idx = start_idx + seq_idx
            if op_idx >= len(lowered_ops):
                match = False
                break
            op_kernel = lowered_ops[op_idx].get("kernel")
            if op_kernel != kernel_id:
                match = False
                break

        if match:
            return start_idx

    return None


def apply_fusion_from_pattern(
    lowered_layer: Dict[str, Any],
    pattern: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Apply a single fusion pattern to a layer's lowered IR.

    Args:
        lowered_layer: Layer from lowered IR with "ops" list
        pattern: Fusion pattern to apply

    Returns:
        Modified layer with fusion applied, or None if pattern didn't match
    """
    ops = lowered_layer.get("ops", [])
    if not ops:
        return None

    # Find matching sequence
    start_idx = find_fusion_candidates(ops, pattern)
    if start_idx is None:
        return None  # No match

    sequence = pattern.get("sequence", [])
    num_ops = len(sequence)

    # Create fused op
    first_op = ops[start_idx]
    fused_op = {
        "name": f"{pattern['name']}_{start_idx}",
        "op": pattern["fused_op"],
        "kernel": pattern["fused_kernel"],
        "fused_from": [op.get("name", f"op_{i}") for i in range(start_idx, start_idx + num_ops)],
        "inputs": first_op.get("inputs", []),
        "outputs": ops[start_idx + num_ops - 1].get("outputs", []),
        "weights": [],
        "biases": [],
        "scratch": [],
        "params": first_op.get("params", {}),
    }

    # Replace sequence with fused op
    new_ops = []
    new_ops.extend(ops[:start_idx])  # Ops before sequence
    new_ops.append(fused_op)  # Fused op
    new_ops.extend(ops[start_idx + num_ops:])  # Ops after sequence

    # Update layer
    lowered_layer["ops"] = new_ops

    return {
        "layer_id": lowered_layer.get("id", 0),
        "pattern": pattern["name"],
        "ops_fused": num_ops,
        "fused_kernel": pattern["fused_kernel"],
        "fused_from": fused_op["fused_from"],
    }


# Example usage:
if __name__ == "__main__":
    # Load registry
    import json
    from pathlib import Path

    registry_path = Path(__file__).parent.parent / "kernel_maps" / "KERNEL_REGISTRY.json"
    with open(registry_path) as f:
        registry = json.load(f)

    # Build patterns
    patterns = build_fusion_patterns_from_registry(registry)

    print(f"Auto-generated {len(patterns)} fusion patterns from registry:")
    for p in patterns:
        print(f"  {p['name']}: fuses {p['sequence']}")
