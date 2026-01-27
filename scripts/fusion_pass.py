#!/usr/bin/env python3
"""
Fusion Pass - Detect and apply fusion patterns to IR.

Usage:
    python scripts/fusion_pass.py --ir ir/layer_00.json --kernel-maps version/v6.6/kernel_maps
    python scripts/fusion_pass.py --ir-dir ir/ --output build/fused_ir/
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class FusionPattern:
    """A fusion pattern to detect."""
    name: str
    ops: List[str]  # Required ops in sequence
    kernel_map_ids: List[str]  # Kernel map IDs that this fusion provides
    min_ops: int = 2  # Minimum ops to trigger fusion

    def matches(self, op_sequence: List[str]) -> Tuple[bool, int]:
        """Check if op_sequence matches this pattern. Returns (match, length_matched)."""
        if len(op_sequence) < self.min_ops:
            return False, 0

        # Check if sequence starts with our pattern
        for i in range(len(op_sequence) - self.min_ops + 1):
            window = op_sequence[i:i + len(self.ops)]
            if window == self.ops:
                return True, len(self.ops)

        # Partial match - try to match prefix
        for i, op in enumerate(self.ops):
            if i >= len(op_sequence):
                break
            if op_sequence[i] != op:
                break
        else:
            return True, len(op_sequence)  # Full prefix match

        return False, 0


@dataclass
class FusionResult:
    """Result of applying fusion to a layer."""
    original_ops: List[str] = field(default_factory=list)
    fused_ops: List[str] = field(default_factory=list)
    replacements: List[Dict] = field(default_factory=list)
    fusion_log: List[Dict] = field(default_factory=list)


# Fusion patterns - sequences of operations
FUSION_PATTERNS = [
    # Holy grail attention fusion
    FusionPattern(
        name="mega_fused_attention_prefill",
        ops=["CK_OP_RMSNORM", "CK_OP_LINEAR_QKV", "CK_OP_ROPE", "CK_OP_ATTENTION", "CK_OP_LINEAR_OUTPROJ", "CK_OP_ADD"],
        kernel_map_ids=["mega_fused_attention_prefill"],
        min_ops=4,
    ),
    # OutProj + MLP fusion
    FusionPattern(
        name="mega_fused_outproj_mlp_prefill",
        ops=["CK_OP_LINEAR_OUTPROJ", "CK_OP_ADD", "CK_OP_RMSNORM", "CK_OP_GEMM", "CK_OP_SWIGLU", "CK_OP_GEMM"],
        kernel_map_ids=["mega_fused_outproj_mlp_prefill"],
        min_ops=4,
    ),
    # QKV fusion (Q, K, V projections)
    FusionPattern(
        name="fused_qkv_projection",
        ops=["CK_OP_LINEAR_Q", "CK_OP_LINEAR_K", "CK_OP_LINEAR_V"],
        kernel_map_ids=["ck_qkv_project_head_major"],
        min_ops=3,
    ),
    # SwiGLU fusion (gate * swish)
    FusionPattern(
        name="swiglu_fusion",
        ops=["CK_OP_SIGMOID", "CK_OP_MUL"],
        kernel_map_ids=["swiglu_forward"],
        min_ops=2,
    ),
    # RMSNorm + residual add
    FusionPattern(
        name="rmsnorm_residual",
        ops=["CK_OP_RMSNORM", "CK_OP_ADD"],
        kernel_map_ids=["rmsnorm_forward", "ck_residual_add_token_major"],
        min_ops=2,
    ),
]


def load_kernel_maps(kernel_maps_dir: Path) -> Dict[str, Dict]:
    """Load all kernel maps for fusion lookup."""
    kernel_maps = {}
    for json_file in kernel_maps_dir.glob("*.json"):
        if json_file.name == "KERNEL_REGISTRY.json" or json_file.name.startswith("_"):
            continue
        try:
            data = json.loads(json_file.read_text())
            if "id" in data:
                kernel_maps[data["id"]] = data
        except Exception as e:
            print(f"[warn] Could not load {json_file}: {e}")
    return kernel_maps


def extract_op_sequence(ir: Dict) -> List[str]:
    """Extract operation sequence from IR."""
    ops = []
    if "ops" in ir:
        for op in ir["ops"]:
            if isinstance(op, dict) and "op" in op:
                ops.append(op["op"])
            elif isinstance(op, str):
                ops.append(op)
    return ops


def find_fusions(op_sequence: List[str], patterns: List[FusionPattern]) -> List[Tuple[FusionPattern, int, int]]:
    """Find all fusion opportunities in an op sequence.

    Returns list of (pattern, start_idx, end_idx) tuples.
    """
    fusions = []

    for pattern in patterns:
        matched = pattern.matches(op_sequence)
        if matched[0]:
            # Find where it matches
            for i in range(len(op_sequence) - pattern.min_ops + 1):
                window = op_sequence[i:i + len(pattern.ops)]
                if window == pattern.ops:
                    fusions.append((pattern, i, i + len(pattern.ops)))
                    break

    return fusions


def apply_fusions(ir: Dict, patterns: List[FusionPattern], kernel_maps: Dict) -> FusionResult:
    """Apply fusion patterns to IR."""
    result = FusionResult()

    # Extract original ops
    original_ops = extract_op_sequence(ir)
    result.original_ops = original_ops.copy()

    # Find fusions
    fusions = find_fusions(original_ops, patterns)

    if not fusions:
        result.fused_ops = original_ops.copy()
        return result

    # Apply fusions (greedy - apply first match, then re-scan)
    fused_ops = []
    i = 0
    fusion_applied = []

    while i < len(original_ops):
        matched = False

        for pattern, start, end in fusions:
            if start == i:
                # Apply fusion
                fused_ops.append(pattern.name)
                fusion_applied.append({
                    "pattern": pattern.name,
                    "original_ops": original_ops[start:end],
                    "replaced_with": pattern.kernel_map_ids[0],
                    "start_idx": start,
                    "end_idx": end,
                })
                result.replacements.extend([
                    {
                        "from": op,
                        "to": pattern.name,
                        "pattern": pattern.name,
                    }
                    for op in original_ops[start:end]
                ])
                i = end
                matched = True
                break

        if not matched:
            fused_ops.append(original_ops[i])
            i += 1

    result.fused_ops = fused_ops
    result.fusion_log = fusion_applied

    return result


def fuse_layer(ir: Dict, kernel_maps_dir: Path, output_dir: Optional[Path] = None) -> FusionResult:
    """Fuse a single layer IR."""
    patterns = FUSION_PATTERNS
    kernel_maps = load_kernel_maps(Path(kernel_maps_dir)) if kernel_maps_dir else {}

    result = apply_fusions(ir, patterns, kernel_maps)

    # Update IR with fused ops
    if "ops" in ir:
        for i, op in enumerate(ir["ops"]):
            if isinstance(op, dict) and "op" in op:
                if i < len(result.fused_ops):
                    # Check if this op was fused
                    fused_op = result.fused_ops[i]
                    if fused_op in [p.name for p in patterns]:
                        op["op"] = fused_op
                        op["_fused"] = True
                        op["_fused_from"] = result.original_ops[i]

    # Write fusion log
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fusion_log_path = output_dir / "fusion_log.json"
        with open(fusion_log_path, "w") as f:
            json.dump(result.fusion_log, f, indent=2)

    return result


def fuse_dir(ir_dir: Path, kernel_maps_dir: Path, output_dir: Path) -> List[FusionResult]:
    """Fuse all IR files in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for ir_file in sorted(ir_dir.glob("layer_*.json")):
        ir = json.loads(ir_file.read_text())
        result = fuse_layer(ir, kernel_maps_dir, output_dir)

        # Write fused IR
        output_path = output_dir / ir_file.name
        with open(output_path, "w") as f:
            json.dump(ir, f, indent=2)

        results.append(result)

        # Print summary
        if result.fusion_log:
            print(f"[fuse] {ir_file.name}: {len(result.fusion_log)} fusion(s)")
            for log in result.fusion_log:
                print(f"       {log['pattern']}: {' + '.join(log['original_ops'])} → {log['replaced_with']}")
        else:
            print(f"[skip] {ir_file.name}: no fusions")

    # Write combined fusion log
    combined_log = {
        "summary": {
            "total_layers": len(results),
            "total_fusions": sum(len(r.fusion_log) for r in results),
        },
        "layers": []
    }
    for i, r in enumerate(results):
        combined_log["layers"].append({
            "layer": f"layer_{i:02d}",
            "fusions": r.fusion_log,
            "replacements": r.replacements,
        })

    with open(output_dir / "fusion_summary.json", "w") as f:
        json.dump(combined_log, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Apply fusion patterns to IR")
    parser.add_argument("--ir", help="Single IR file to fuse")
    parser.add_argument("--ir-dir", help="Directory of IR files")
    parser.add_argument("--kernel-maps", "-k", default="version/v6.6/kernel_maps",
                        help="Kernel maps directory")
    parser.add_argument("--output", "-o", default="build/fused_ir",
                        help="Output directory for fused IR")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.ir and not args.ir_dir:
        parser.error("Either --ir or --ir-dir required")

    output_dir = Path(args.output)

    if args.ir:
        ir = json.loads(Path(args.ir).read_text())
        result = fuse_layer(ir, args.kernel_maps, output_dir)

        # Write fused IR
        output_path = output_dir / Path(args.ir).name
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(ir, f, indent=2)

        if result.fusion_log:
            print(f"[fuse] Applied {len(result.fusion_log)} fusion(s)")
            for log in result.fusion_log:
                print(f"       {log['pattern']}: {' + '.join(log['original_ops'])}")
        else:
            print("[skip] No fusions applied")

    else:
        results = fuse_dir(Path(args.ir_dir), args.kernel_maps, output_dir)
        total = sum(len(r.fusion_log) for r in results)
        print(f"\n[ok] Applied {total} fusions to {len(results)} layers")
        print(f"[ok] Written to {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
