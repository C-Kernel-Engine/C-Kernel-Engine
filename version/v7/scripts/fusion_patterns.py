#!/usr/bin/env python3
"""
fusion_patterns.py - Fusion pattern definitions for IR v4 optimization

Patterns are matched in priority order (highest first). Each pattern defines:
- sequence: List of op names to match consecutively
- fused_op: The replacement op name
- fused_kernel: Template for the fused kernel name ({dtype} is replaced)
- mode: List of modes where this pattern applies
- remove_buffers: Buffer name patterns to remove (intermediate results)
- constraints: Optional validation functions
"""

from typing import Dict, List, Optional, Callable

# ---------------------------------------------------------------------------
# Fusion Pattern Definitions
# ---------------------------------------------------------------------------

FUSION_PATTERNS = [
    # -------------------------------------------------------------------------
    # MLP Fusion (highest priority for decode - eliminates 2 intermediate buffers)
    # Pattern: mlp_up -> swiglu -> mlp_down
    # Fuses: gate projection + up projection + SwiGLU activation + down projection
    # -------------------------------------------------------------------------
    {
        "name": "fused_mlp_decode",
        "priority": 100,
        "mode": ["decode"],
        "sequence": ["mlp_up", "swiglu", "mlp_down"],
        "fused_op": "fused_mlp",
        "fused_kernel": "mlp_fused_decode_{dtype}",
        "remove_buffers": ["fc1_out", "swiglu_out"],
        "description": "Full MLP fusion for single-token decode (gate+up+swiglu+down)",
    },

    # -------------------------------------------------------------------------
    # GEMM + SwiGLU Fusion (for prefill where full MLP fusion may not apply)
    # Pattern: mlp_up -> swiglu
    # Keeps intermediate result in registers, avoids writing fc1_out to DRAM
    # -------------------------------------------------------------------------
    {
        "name": "gemm_swiglu_fused",
        "priority": 90,
        "mode": ["prefill", "decode"],
        "sequence": ["mlp_up", "swiglu"],
        "fused_op": "gemm_swiglu",
        "fused_kernel": "gemm_swiglu_fused_{dtype}",
        "remove_buffers": ["fc1_out"],
        "description": "Fused gate+up projection with SwiGLU activation",
    },

    # -------------------------------------------------------------------------
    # Residual + RMSNorm Fusion
    # Pattern: residual_add -> rmsnorm
    # Common at layer boundaries, reduces activation memory traffic
    # -------------------------------------------------------------------------
    {
        "name": "residual_rmsnorm_fused",
        "priority": 80,
        "mode": ["prefill", "decode"],
        "sequence": ["residual_add", "rmsnorm"],
        "fused_op": "residual_rmsnorm",
        "fused_kernel": "residual_rmsnorm_fused_{dtype}",
        "remove_buffers": [],  # residual output often needed for skip connection
        "description": "Fused residual add + RMSNorm",
    },

    # -------------------------------------------------------------------------
    # GEMM + GELU Fusion
    # Pattern: linear -> gelu (for non-gated MLPs like GPT-2)
    # -------------------------------------------------------------------------
    {
        "name": "gemm_gelu_fused",
        "priority": 70,
        "mode": ["prefill", "decode"],
        "sequence": ["linear", "gelu"],
        "fused_op": "gemm_gelu",
        "fused_kernel": "gemm_bias_gelu_fused_{dtype}",
        "remove_buffers": [],
        "description": "Fused linear + GELU activation",
    },

    # -------------------------------------------------------------------------
    # GEMM + ReLU Fusion
    # Pattern: linear -> relu
    # -------------------------------------------------------------------------
    {
        "name": "gemm_relu_fused",
        "priority": 70,
        "mode": ["prefill", "decode"],
        "sequence": ["linear", "relu"],
        "fused_op": "gemm_relu",
        "fused_kernel": "gemm_bias_relu_fused_{dtype}",
        "remove_buffers": [],
        "description": "Fused linear + ReLU activation",
    },

    # -------------------------------------------------------------------------
    # Attention + Output Projection Fusion (decode mode)
    # Pattern: attention -> attn_proj
    # Keeps attention output in cache, avoids intermediate write
    # -------------------------------------------------------------------------
    {
        "name": "attention_proj_fused_decode",
        "priority": 85,
        "mode": ["decode"],
        "sequence": ["attention", "attn_proj"],
        "fused_op": "attention_proj_fused",
        "fused_kernel": "attention_proj_fused_decode_{dtype}",
        "remove_buffers": ["attn_out"],
        "description": "Fused attention + output projection for decode",
    },

    # =========================================================================
    # BACKWARD PASS FUSION PATTERNS
    # =========================================================================

    # -------------------------------------------------------------------------
    # MLP Backward Fusion (backward pass)
    # Pattern: mlp_down_backward -> swiglu_backward -> mlp_up_backward
    # Fuses: down proj grad + swiglu grad + up proj grad
    # -------------------------------------------------------------------------
    {
        "name": "fused_mlp_backward",
        "priority": 100,
        "mode": ["backward"],
        "sequence": ["gemm_backward", "swiglu_backward", "gemm_backward"],
        "fused_op": "fused_mlp_backward",
        "fused_kernel": "mlp_fused_backward_{dtype}",
        "remove_buffers": ["d_swiglu_out", "d_fc1_out"],
        "description": "Full MLP backward fusion (down_grad+swiglu_grad+up_grad)",
    },

    # -------------------------------------------------------------------------
    # GEMM + Activation Backward Fusion
    # Pattern: gemm_backward -> swiglu_backward
    # -------------------------------------------------------------------------
    {
        "name": "gemm_swiglu_backward_fused",
        "priority": 90,
        "mode": ["backward"],
        "sequence": ["gemm_backward", "swiglu_backward"],
        "fused_op": "gemm_swiglu_backward",
        "fused_kernel": "gemm_swiglu_backward_fused_{dtype}",
        "remove_buffers": ["d_swiglu_out"],
        "description": "Fused GEMM backward + SwiGLU backward",
    },

    # -------------------------------------------------------------------------
    # RMSNorm + Residual Backward Fusion
    # Pattern: rmsnorm_backward -> add_backward
    # -------------------------------------------------------------------------
    {
        "name": "rmsnorm_residual_backward_fused",
        "priority": 80,
        "mode": ["backward"],
        "sequence": ["rmsnorm_backward", "add_backward"],
        "fused_op": "rmsnorm_residual_backward",
        "fused_kernel": "rmsnorm_residual_backward_fused_{dtype}",
        "remove_buffers": [],
        "description": "Fused RMSNorm backward + residual backward",
    },

    # -------------------------------------------------------------------------
    # Attention Backward + QKV Backward Fusion
    # Pattern: attention_backward -> qkv_backward
    # -------------------------------------------------------------------------
    {
        "name": "attention_qkv_backward_fused",
        "priority": 85,
        "mode": ["backward"],
        "sequence": ["attention_backward", "qkv_backward"],
        "fused_op": "attention_qkv_backward",
        "fused_kernel": "attention_qkv_backward_fused_{dtype}",
        "remove_buffers": ["d_q", "d_k", "d_v"],
        "description": "Fused attention backward + QKV projection backward",
    },
]


# ---------------------------------------------------------------------------
# Pattern Utilities
# ---------------------------------------------------------------------------

def get_patterns_for_mode(mode: str) -> List[Dict]:
    """Get fusion patterns applicable to the given mode, sorted by priority."""
    applicable = [p for p in FUSION_PATTERNS if mode in p.get("mode", [])]
    return sorted(applicable, key=lambda p: -p["priority"])


def get_pattern_by_name(name: str) -> Optional[Dict]:
    """Get a specific pattern by name."""
    for p in FUSION_PATTERNS:
        if p["name"] == name:
            return p
    return None


def pattern_removes_buffer(pattern: Dict, buffer_name: str) -> bool:
    """Check if a pattern removes a buffer (by suffix match)."""
    for suffix in pattern.get("remove_buffers", []):
        if buffer_name.endswith(suffix):
            return True
    return False


# ---------------------------------------------------------------------------
# Pattern Matching Helpers
# ---------------------------------------------------------------------------

def ops_match_sequence(ops: List[Dict], sequence: List[str]) -> bool:
    """Check if a list of ops matches a pattern sequence."""
    if len(ops) != len(sequence):
        return False
    for op, expected in zip(ops, sequence):
        if op.get("op") != expected:
            return False
    return True


def validate_data_flow(ops: List[Dict]) -> bool:
    """
    Validate that ops form a valid data flow chain.
    Each op's output should feed into the next op's input.
    """
    if len(ops) < 2:
        return True

    for i in range(len(ops) - 1):
        curr_outputs = set(ops[i].get("outputs", []))
        next_inputs = set(ops[i + 1].get("inputs", []))
        # At least one output should connect to next input
        if not curr_outputs.intersection(next_inputs):
            return False
    return True


def merge_op_ios(ops: List[Dict], pattern: Dict, dtype: str) -> Dict:
    """
    Merge multiple ops into a single fused op.
    - Takes inputs from first op (excluding intermediate connections)
    - Takes outputs from last op
    - Takes all weights from all ops
    - Removes intermediate buffers
    """
    first_op = ops[0]
    last_op = ops[-1]

    # Collect all intermediate buffers (outputs that become inputs later)
    intermediate = set()
    for i, op in enumerate(ops[:-1]):
        for out in op.get("outputs", []):
            for later_op in ops[i + 1:]:
                if out in later_op.get("inputs", []):
                    intermediate.add(out)

    removed = set(pattern.get("remove_buffers", []))

    fused_outputs = []
    seen_outputs = set()
    for op in ops:
        for out in op.get("outputs", []):
            if any(out.endswith(suffix) for suffix in removed):
                continue
            if out in seen_outputs:
                continue
            fused_outputs.append(out)
            seen_outputs.add(out)

    # Build fused op
    fused = {
        "op": pattern["fused_op"],
        "name": f"fused_{first_op.get('name', 'op')}",
        "kernel": pattern["fused_kernel"].format(dtype=dtype),
        "kernel_dtype": dtype,
        "inputs": [i for i in first_op.get("inputs", []) if i not in intermediate],
        "outputs": fused_outputs or last_op.get("outputs", []),
        "weights": [],
        "scratch": [],
        "fused_from": [op.get("op") for op in ops],
    }

    # Collect all weights
    for op in ops:
        for w in op.get("weights", []):
            if w not in fused["weights"]:
                fused["weights"].append(w)

    # Collect scratch (excluding intermediates)
    for op in ops:
        for s in op.get("scratch", []):
            if s not in intermediate and s not in fused["scratch"]:
                fused["scratch"].append(s)

    return fused


# ---------------------------------------------------------------------------
# Fusion Statistics
# ---------------------------------------------------------------------------

class FusionStats:
    """Track fusion statistics for reporting."""

    def __init__(self):
        self.fusions_applied = []
        self.ops_removed = 0
        self.buffers_removed = 0
        self.bytes_saved = 0

    def record_fusion(self, layer_id: int, pattern: Dict, ops_count: int,
                      buffers: List[str], bytes_saved: int = 0):
        self.fusions_applied.append({
            "layer": layer_id,
            "pattern": pattern["name"],
            "ops_fused": ops_count,
            "buffers_removed": len(buffers),
            "removed_buffers": buffers,
        })
        self.ops_removed += ops_count - 1  # Fused into 1 op
        self.buffers_removed += len(buffers)
        self.bytes_saved += bytes_saved

    def to_dict(self) -> Dict:
        return {
            "total_fusions": len(self.fusions_applied),
            "ops_removed": self.ops_removed,
            "buffers_removed": self.buffers_removed,
            "bytes_saved": self.bytes_saved,
            "fusions": self.fusions_applied,
        }


# ---------------------------------------------------------------------------
# Registry-Driven Fusion (Phase 3)
# ---------------------------------------------------------------------------
# Auto-generate fusion patterns from kernel registry entries with "fuses" field.
# Registry patterns use kernel IDs for matching (from lowered IR).
# Manual patterns use op names (from graph IR).

from pathlib import Path
from typing import Dict, List, Optional, Any


def load_kernel_registry(registry_path: Optional[Path] = None) -> Dict:
    """Load kernel registry from JSON file."""
    if registry_path is None:
        registry_path = Path(__file__).parent.parent / "kernel_maps" / "KERNEL_REGISTRY.json"

    if not registry_path.exists():
        raise FileNotFoundError(f"Kernel registry not found: {registry_path}")

    import json
    with open(registry_path) as f:
        return json.load(f)


def build_fusion_patterns_from_registry(registry: Dict) -> List[Dict]:
    """
    Auto-generate fusion patterns from registry entries with "fuses" field.

    Registry entries with "fuses" define what ops they replace.
    Pattern sequence matches by kernel IDs from lowered IR.
    """
    patterns = []

    for kernel_entry in registry.get("kernels", []):
        if "fuses" not in kernel_entry:
            continue

        kernel_id = kernel_entry.get("id")
        op_type = kernel_entry.get("op", "")
        fuses = kernel_entry.get("fuses", [])

        if not fuses:
            continue

        # Determine mode from kernel name
        mode = ["prefill", "decode"]
        if "prefill" in kernel_id.lower():
            mode = ["prefill"]
        elif "decode" in kernel_id.lower():
            mode = ["decode"]

        # Build pattern
        pattern = {
            "name": kernel_id,
            "priority": 100,  # Highest priority for fused kernels
            "mode": mode,
            "sequence": fuses,  # Kernel IDs to match
            "fused_op": op_type,
            "fused_kernel": kernel_id,
            "remove_buffers": [],
            "description": f"Auto-generated from registry: {kernel_id}",
        }

        patterns.append(pattern)

    return patterns


def get_registry_driven_patterns(
    mode: str,
    registry_path: Optional[Path] = None
) -> List[Dict]:
    """
    Get fusion patterns for a mode, merged from registry and manual patterns.

    Registry patterns (fused kernels) have highest priority.
    Manual patterns cover ops without registry fusion.
    """
    # Load registry
    registry = load_kernel_registry(registry_path)

    # Build registry patterns
    registry_patterns = build_fusion_patterns_from_registry(registry)

    # Get manual patterns for this mode
    manual_patterns = get_patterns_for_mode(mode)

    # Merge: registry patterns (exact kernel matches) + manual patterns
    combined = list(registry_patterns)
    registry_kernel_ids = {p["fused_kernel"] for p in registry_patterns}

    for pattern in manual_patterns:
        if pattern.get("fused_kernel") not in registry_kernel_ids:
            combined.append(pattern)

    # Sort by priority (highest first)
    return sorted(combined, key=lambda x: -x.get("priority", 0))


def match_pattern_by_kernel_ids(
    lowered_ops: List[Dict[str, Any]],
    pattern: Dict[str, Any]
) -> Optional[int]:
    """
    Match fusion pattern against lowered IR ops by kernel IDs.

    Args:
        lowered_ops: Ops from lowered IR with "kernel" field
        pattern: Fusion pattern with "sequence" (list of kernel IDs)

    Returns:
        Start index of matching sequence, or None
    """
    sequence = pattern.get("sequence", [])
    if not sequence:
        return None

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


# ---------------------------------------------------------------------------
# Main exports
# ---------------------------------------------------------------------------

__all__ = [
    "FUSION_PATTERNS",
    "get_patterns_for_mode",
    "get_pattern_by_name",
    "pattern_removes_buffer",
    "ops_match_sequence",
    "validate_data_flow",
    "merge_op_ios",
    "FusionStats",
    "load_kernel_registry",
    "build_fusion_patterns_from_registry",
    "get_registry_driven_patterns",
    "match_pattern_by_kernel_ids",
]
