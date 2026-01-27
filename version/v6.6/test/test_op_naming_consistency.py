#!/usr/bin/env python3
"""
test_op_naming_consistency.py - Validate operation naming is consistent

This test ensures op names are consistent across:
- IR (graph, lowered, fused)
- Kernel maps
- Generated code

BUGS THIS CATCHES:
- Bug 1: Op naming inconsistency (atten vs attention)

Usage:
    python test_op_naming_consistency.py
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

# Paths
V66_ROOT = Path(__file__).parent.parent
GENERATED_DIR = V66_ROOT / "src" / "generated"
KERNEL_MAPS_DIR = V66_ROOT / "kernel_maps"
CACHE_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"


# Known valid operation names (canonical)
CANONICAL_OPS = {
    # Attention block
    "rmsnorm", "rms_norm", "layer_norm",
    "qkv_proj", "qkv_projection", "q_proj", "k_proj", "v_proj",
    "attention", "attn", "self_attn", "self_attention",
    "out_proj", "output_proj", "o_proj",
    "rope", "rope_qk", "rotary_embedding",

    # MLP block
    "mlp", "feed_forward", "ffn",
    "mlp_gate_up", "gate_proj", "up_proj",
    "mlp_down", "down_proj",
    "silu", "swiglu", "gelu", "silu_mul",

    # Misc
    "embedding", "token_embed", "embed", "dense_embedding_lookup",
    "residual", "residual_add", "add",
    "logits", "lm_head",
    "kv_cache", "kv_cache_store",

    # Fused
    "mega_fused_attention", "mega_fused_attention_prefill", "mega_fused_attention_decode",
    "mega_fused_outproj_mlp", "mega_fused_outproj_mlp_prefill",
}

# Common typos/inconsistencies to detect
KNOWN_ISSUES = {
    "atten": "attention",
    "atention": "attention",
    "attenton": "attention",
    "embeding": "embedding",
    "embdding": "embedding",
    "rmsnrom": "rmsnorm",
    "reisdual": "residual",
    "resiudal": "residual",
    "logit": "logits",
}


@dataclass
class NamingIssue:
    """A naming consistency issue."""
    severity: str
    source: str
    message: str
    found: str
    suggested: str = ""


class NamingValidator:
    """Validates operation naming consistency."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues: List[NamingIssue] = []
        self.ops_by_source: Dict[str, Set[str]] = defaultdict(set)

    def load_ir_ops(self, path: Path) -> Set[str]:
        """Extract op names from IR JSON."""
        ops = set()

        if not path.exists():
            return ops

        with open(path) as f:
            ir = json.load(f)

        # Try various structures
        def extract_ops(data, prefix=""):
            if isinstance(data, dict):
                if "op" in data:
                    ops.add(data["op"])
                if "kernel" in data and data["kernel"]:
                    ops.add(data["kernel"])
                if "function" in data:
                    ops.add(data["function"])

                for key, value in data.items():
                    extract_ops(value, f"{prefix}.{key}")

            elif isinstance(data, list):
                for item in data:
                    extract_ops(item, prefix)

        extract_ops(ir)
        return ops

    def load_kernel_map_ops(self) -> Set[str]:
        """Extract op names from kernel maps."""
        ops = set()

        if not KERNEL_MAPS_DIR.exists():
            return ops

        for json_file in KERNEL_MAPS_DIR.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Get kernel name from file
                ops.add(json_file.stem)

                # Get op types if defined
                if "op" in data:
                    ops.add(data["op"])
                if "ops" in data:
                    ops.update(data["ops"])

            except (json.JSONDecodeError, IOError):
                continue

        return ops

    def load_c_code_ops(self) -> Set[str]:
        """Extract function names from generated C code."""
        ops = set()

        for base_dir in [GENERATED_DIR, CACHE_DIR]:
            c_path = base_dir / "ck-kernel-inference.c"
            if c_path.exists():
                code = c_path.read_text()

                # Find function calls
                pattern = r'\b(\w+)_(?:forward|backward|decode|prefill|project)\b'
                for match in re.finditer(pattern, code):
                    ops.add(match.group(0))

                # Find op comments
                pattern = r'/\*\s*Op\s*\d+:\s*(\w+)'
                for match in re.finditer(pattern, code):
                    ops.add(match.group(1))

                break

        return ops

    def check_typos(self, ops: Set[str], source: str) -> None:
        """Check for known typos."""
        for op in ops:
            op_lower = op.lower()
            for typo, correct in KNOWN_ISSUES.items():
                if typo in op_lower:
                    self.issues.append(NamingIssue(
                        severity="ERROR",
                        source=source,
                        message=f"Possible typo in op name",
                        found=op,
                        suggested=op.lower().replace(typo, correct)
                    ))

    def check_consistency(self) -> None:
        """Check naming consistency across sources."""
        # Normalize names for comparison
        def normalize(name: str) -> str:
            return name.lower().replace("_", "").replace("-", "")

        # Build normalized lookup
        all_normalized: Dict[str, List[tuple]] = defaultdict(list)
        for source, ops in self.ops_by_source.items():
            for op in ops:
                norm = normalize(op)
                all_normalized[norm].append((source, op))

        # Find inconsistencies (same normalized name, different spelling)
        for norm, occurrences in all_normalized.items():
            unique_spellings = set(op for _, op in occurrences)
            if len(unique_spellings) > 1:
                sources = {src for src, _ in occurrences}
                self.issues.append(NamingIssue(
                    severity="WARNING",
                    source=", ".join(sources),
                    message=f"Inconsistent naming across sources",
                    found=", ".join(unique_spellings)
                ))

    def run_all_tests(self) -> bool:
        """Run all naming validations."""
        print("\n" + "="*70)
        print("OPERATION NAMING CONSISTENCY")
        print("="*70)

        # Load ops from various sources
        print("\n" + "-"*70)
        print("Loading operation names from sources...")
        print("-"*70)

        # IR files
        for base_dir in [GENERATED_DIR, CACHE_DIR]:
            for ir_file in ["lowered_decode.json", "graph.json", "fused_decode.json"]:
                path = base_dir / ir_file
                if path.exists():
                    ops = self.load_ir_ops(path)
                    self.ops_by_source[ir_file] = ops
                    print(f"  {ir_file}: {len(ops)} ops")
                    self.check_typos(ops, ir_file)
                    break

        # Kernel maps
        kernel_ops = self.load_kernel_map_ops()
        self.ops_by_source["kernel_maps"] = kernel_ops
        print(f"  kernel_maps: {len(kernel_ops)} ops")
        self.check_typos(kernel_ops, "kernel_maps")

        # Generated C code
        c_ops = self.load_c_code_ops()
        self.ops_by_source["generated_c"] = c_ops
        print(f"  generated_c: {len(c_ops)} ops")
        self.check_typos(c_ops, "generated_c")

        # Check consistency
        print("\n" + "-"*70)
        print("Checking consistency...")
        print("-"*70)
        self.check_consistency()

        # Summary
        print("\n" + "="*70)
        print("NAMING ISSUES")
        print("="*70)

        if not self.issues:
            print("No naming issues found!")
        else:
            for severity in ["ERROR", "WARNING"]:
                issues = [i for i in self.issues if i.severity == severity]
                if issues:
                    print(f"\n{severity} ({len(issues)}):")
                    for issue in issues:
                        print(f"  [{issue.source}]: {issue.message}")
                        print(f"    Found: {issue.found}")
                        if issue.suggested:
                            print(f"    Suggested: {issue.suggested}")

        # Verdict
        print("\n" + "="*70)
        errors = len([i for i in self.issues if i.severity == "ERROR"])
        if errors > 0:
            print(f"VERDICT: FAIL - {errors} naming errors")
        else:
            print("VERDICT: PASS - Naming is consistent")
        print("="*70)

        return errors == 0


def main():
    parser = argparse.ArgumentParser(description="Validate op naming")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    validator = NamingValidator(verbose=args.verbose)
    success = validator.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
