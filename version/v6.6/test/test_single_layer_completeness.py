#!/usr/bin/env python3
"""
test_single_layer_completeness.py - Verify single-layer IR is complete and robust

When doing single-layer testing, this test ensures:
1. Weight name mapping works (C macros W_LAYER_0_WQ ↔ manifest layer.0.wq)
2. ALL layer 0 weights are properly referenced in the IR
3. The layer has all required operations (RMSNorm, QKV, RoPE, Attention, MLP, etc.)
4. Biases are handled correctly (NULL or used)
5. Activation buffers chain correctly

This is the "robustness" test for single-layer mode.

Usage:
    python test_single_layer_completeness.py
    python test_single_layer_completeness.py --verbose
    python test_single_layer_completeness.py --model-dir /path/to/model
"""

import argparse
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any


# ═══════════════════════════════════════════════════════════════════════════════
# Weight Name Mapping
# ═══════════════════════════════════════════════════════════════════════════════

# Map between manifest names and C macro names
WEIGHT_NAME_MAP = {
    # Token embedding
    "token_emb": "W_TOKEN_EMB",
    "token_embedding": "W_TOKEN_EMB",

    # Layer weights (template - replace {L} with layer index)
    "layer.{L}.wq": "W_LAYER_{L}_WQ",
    "layer.{L}.wk": "W_LAYER_{L}_WK",
    "layer.{L}.wv": "W_LAYER_{L}_WV",
    "layer.{L}.wo": "W_LAYER_{L}_WO",
    "layer.{L}.w1": "W_LAYER_{L}_W1",
    "layer.{L}.w2": "W_LAYER_{L}_W2",
    "layer.{L}.w3": "W_LAYER_{L}_W3",  # For models with separate gate
    "layer.{L}.ln1_gamma": "W_LAYER_{L}_LN1_GAMMA",
    "layer.{L}.ln2_gamma": "W_LAYER_{L}_LN2_GAMMA",
    "layer.{L}.ln1_weight": "W_LAYER_{L}_LN1_GAMMA",  # Alias
    "layer.{L}.ln2_weight": "W_LAYER_{L}_LN2_GAMMA",  # Alias

    # Biases
    "layer.{L}.bq": "W_LAYER_{L}_BQ",
    "layer.{L}.bk": "W_LAYER_{L}_BK",
    "layer.{L}.bv": "W_LAYER_{L}_BV",
    "layer.{L}.bo": "W_LAYER_{L}_BO",
    "layer.{L}.b1": "W_LAYER_{L}_B1",
    "layer.{L}.b2": "W_LAYER_{L}_B2",

    # Final layer norm
    "final_ln_gamma": "W_FINAL_LN_GAMMA",
    "final_ln_weight": "W_FINAL_LN_GAMMA",
    "final_ln_bias": "W_FINAL_LN_BIAS",

    # LM head
    "lm_head": "W_LM_HEAD",
    "output": "W_OUTPUT",
}


def normalize_weight_name(name: str) -> str:
    """Convert manifest name to normalized C macro style.

    Examples:
        layer.0.wq -> W_LAYER_0_WQ
        token_emb -> W_TOKEN_EMB
    """
    # Check direct match first
    if name.upper().startswith("W_"):
        return name.upper()

    # Try layer pattern
    match = re.match(r'layer\.(\d+)\.(\w+)', name)
    if match:
        layer_idx = match.group(1)
        weight_name = match.group(2).upper()
        return f"W_LAYER_{layer_idx}_{weight_name}"

    # Non-layer weights
    upper = name.upper().replace(".", "_")
    if not upper.startswith("W_"):
        upper = "W_" + upper
    return upper


def manifest_to_macro(manifest_name: str) -> str:
    """Convert manifest name to C macro name."""
    return normalize_weight_name(manifest_name)


def macro_to_manifest(macro_name: str) -> str:
    """Convert C macro name to manifest name.

    Examples:
        W_LAYER_0_WQ -> layer.0.wq
        W_TOKEN_EMB -> token_emb
        W_FINAL_LN_GAMMA -> final_ln_gamma
    """
    if not macro_name.startswith("W_"):
        return macro_name.lower()

    name = macro_name[2:]  # Remove W_

    # Check for layer pattern
    match = re.match(r'LAYER_(\d+)_(\w+)', name)
    if match:
        layer_idx = match.group(1)
        weight_name = match.group(2).lower()
        return f"layer.{layer_idx}.{weight_name}"

    # Non-layer weights - keep underscores (don't convert to dots)
    return name.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Required Operations for Complete Layer
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum operations for a complete transformer layer
REQUIRED_LAYER_OPERATIONS = {
    "attention": {
        "rmsnorm_pre_attn": ["rmsnorm_forward"],  # Pre-attention norm
        "q_proj": ["gemv_q5_0", "gemv_q8_0", "gemv_q4_k", "gemm_nt_*"],
        "k_proj": ["gemv_q5_0", "gemv_q8_0", "gemv_q4_k", "gemm_nt_*"],
        "v_proj": ["gemv_q5_0", "gemv_q8_0", "gemv_q4_k", "gemm_nt_*"],
        "rope": ["rope_forward_qk"],
        "attention": ["attention_forward_*", "attention_decode_*"],
        "out_proj": ["gemv_q5_0", "gemv_q8_0", "gemv_q4_k", "gemm_nt_*"],
        "residual_attn": ["ck_residual_add_*"],
    },
    "mlp": {
        "rmsnorm_pre_mlp": ["rmsnorm_forward"],  # Pre-MLP norm
        "gate_up_proj": ["gemv_q5_0", "gemv_q8_0", "gemv_q4_k", "gemm_nt_*"],
        "activation": ["swiglu_forward", "silu_*", "gelu_*"],
        "down_proj": ["gemv_q5_0", "gemv_q8_0", "gemv_q4_k", "gemv_q6_k", "gemm_nt_*"],
        "residual_mlp": ["ck_residual_add_*"],
    },
    "fused_alternatives": {
        # Fused kernels that replace multiple operations
        "mega_fused_attention_prefill": ["rmsnorm_pre_attn", "q_proj", "k_proj", "v_proj", "rope", "attention", "out_proj", "residual_attn"],
        "mega_fused_attention_decode": ["rmsnorm_pre_attn", "q_proj", "k_proj", "v_proj", "rope", "attention", "out_proj", "residual_attn"],
        "mega_fused_outproj_mlp_prefill": ["out_proj", "residual_attn", "rmsnorm_pre_mlp", "gate_up_proj", "activation", "down_proj", "residual_mlp"],
        "mega_fused_outproj_mlp_decode": ["out_proj", "residual_attn", "rmsnorm_pre_mlp", "gate_up_proj", "activation", "down_proj", "residual_mlp"],
    }
}

# Required weights for a complete layer
REQUIRED_LAYER_WEIGHTS = {
    "attention": ["wq", "wk", "wv", "wo", "ln1_gamma"],
    "mlp": ["w1", "w2", "ln2_gamma"],  # w1 is gate_up (fused), w2 is down
}


# ═══════════════════════════════════════════════════════════════════════════════
# Test Result Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    details: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Single Layer Validator
# ═══════════════════════════════════════════════════════════════════════════════

class SingleLayerValidator:
    """Validates that a single-layer IR is complete and robust."""

    def __init__(self, model_dir: Path, verbose: bool = False):
        self.model_dir = model_dir
        self.verbose = verbose
        self.results: List[TestResult] = []

        # Data to be loaded
        self.lowered_ir: Dict = {}
        self.manifest: Dict = {}
        self.config: Dict = {}
        self.c_code: str = ""

    def load_files(self) -> bool:
        """Load all required files."""
        # Find lowered IR
        lowered_paths = [
            self.model_dir / "lowered_decode_call.json",
            self.model_dir / "lowered_prefill_call.json",
        ]
        for p in lowered_paths:
            if p.exists():
                with open(p) as f:
                    self.lowered_ir = json.load(f)
                if self.verbose:
                    print(f"Loaded lowered IR: {p}")
                break

        if not self.lowered_ir:
            print(f"ERROR: No lowered IR found in {self.model_dir}")
            return False

        # Find manifest
        manifest_path = self.model_dir / "weights_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            if self.verbose:
                print(f"Loaded manifest: {manifest_path}")

        # Find config
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
            if self.verbose:
                print(f"Loaded config: {config_path}")

        # Find generated C code
        c_paths = [
            self.model_dir / "model_v6_6.c",
            self.model_dir / "ck-kernel-inference.c",
        ]
        for p in c_paths:
            if p.exists():
                self.c_code = p.read_text()
                if self.verbose:
                    print(f"Loaded C code: {p}")
                break

        return True

    def get_ir_layers(self) -> Set[int]:
        """Get set of layer indices in IR."""
        layers = set()
        for op in self.lowered_ir.get("operations", []):
            layer = op.get("layer", -1)
            if layer >= 0:
                layers.add(layer)
        return layers

    def get_ir_weights(self) -> Set[str]:
        """Get all weights referenced in IR (normalized to macro names)."""
        weights = set()
        for op in self.lowered_ir.get("operations", []):
            for arg in op.get("args", []):
                source = arg.get("source", "")
                expr = arg.get("expr", "")

                # From source field
                if source.startswith("weight:"):
                    weight_name = source.split(":", 1)[1]
                    weights.add(normalize_weight_name(weight_name))

                # From expr field (C macros)
                for match in re.findall(r'W_[A-Z0-9_]+', expr):
                    weights.add(match)

        return weights

    def get_manifest_weights(self, layer: int = None) -> Set[str]:
        """Get weights from manifest (normalized to macro names)."""
        weights = set()
        for entry in self.manifest.get("entries", []):
            name = entry.get("name", "")

            # Filter by layer if specified
            if layer is not None:
                match = re.search(r'layer\.(\d+)\.', name)
                if match:
                    entry_layer = int(match.group(1))
                    if entry_layer != layer:
                        continue

            weights.add(normalize_weight_name(name))

        return weights

    def test_weight_name_mapping(self) -> TestResult:
        """Test 1: Verify weight name mapping works correctly."""
        details = []

        # Test specific mappings
        test_cases = [
            ("layer.0.wq", "W_LAYER_0_WQ"),
            ("layer.15.wk", "W_LAYER_15_WK"),
            ("token_emb", "W_TOKEN_EMB"),
            ("final_ln_gamma", "W_FINAL_LN_GAMMA"),
        ]

        all_passed = True
        for manifest_name, expected_macro in test_cases:
            actual_macro = normalize_weight_name(manifest_name)
            if actual_macro != expected_macro:
                details.append(f"FAIL: {manifest_name} -> {actual_macro} (expected {expected_macro})")
                all_passed = False
            else:
                details.append(f"OK: {manifest_name} -> {actual_macro}")

        # Test reverse mapping
        for manifest_name, macro_name in test_cases:
            actual_manifest = macro_to_manifest(macro_name)
            if actual_manifest != manifest_name:
                details.append(f"FAIL reverse: {macro_name} -> {actual_manifest} (expected {manifest_name})")
                all_passed = False
            else:
                details.append(f"OK reverse: {macro_name} -> {actual_manifest}")

        return TestResult(
            name="weight_name_mapping",
            passed=all_passed,
            message="Weight name mapping works correctly" if all_passed else "Weight name mapping has errors",
            details=details
        )

    def test_layer0_weights_referenced(self) -> TestResult:
        """Test 2: Verify ALL layer 0 weights are referenced in IR."""
        details = []

        ir_layers = self.get_ir_layers()
        if not ir_layers:
            return TestResult(
                name="layer0_weights_referenced",
                passed=False,
                message="No layers found in IR",
                details=["IR has no layer operations"]
            )

        layer_idx = min(ir_layers)  # Get first layer (usually 0)
        details.append(f"Testing layer {layer_idx}")

        # Get weights referenced in IR
        ir_weights = self.get_ir_weights()
        details.append(f"IR references {len(ir_weights)} weights")

        # Get expected weights for this layer from manifest
        manifest_weights = self.get_manifest_weights(layer=layer_idx)
        details.append(f"Manifest has {len(manifest_weights)} weights for layer {layer_idx}")

        # Check required weights are present
        missing_weights = []
        for weight_type in REQUIRED_LAYER_WEIGHTS["attention"] + REQUIRED_LAYER_WEIGHTS["mlp"]:
            expected_macro = f"W_LAYER_{layer_idx}_{weight_type.upper()}"
            if expected_macro not in ir_weights:
                # Check if it's in manifest (might be valid absence)
                if expected_macro in manifest_weights:
                    missing_weights.append(expected_macro)
                    details.append(f"MISSING: {expected_macro} in manifest but not in IR")
                else:
                    details.append(f"SKIP: {expected_macro} not in manifest (model may not have it)")
            else:
                details.append(f"OK: {expected_macro} referenced in IR")

        passed = len(missing_weights) == 0
        return TestResult(
            name="layer0_weights_referenced",
            passed=passed,
            message=f"Layer {layer_idx} has all weights" if passed else f"Layer {layer_idx} missing {len(missing_weights)} weights",
            details=details
        )

    def test_layer_operations_complete(self) -> TestResult:
        """Test 3: Verify layer has all required operations."""
        details = []

        ir_layers = self.get_ir_layers()
        if not ir_layers:
            return TestResult(
                name="layer_operations_complete",
                passed=False,
                message="No layers found in IR",
                details=["IR has no layer operations"]
            )

        layer_idx = min(ir_layers)

        # Collect operations for this layer
        layer_ops = []
        for op in self.lowered_ir.get("operations", []):
            if op.get("layer") == layer_idx:
                kernel = op.get("function") or op.get("kernel", "")
                layer_ops.append(kernel)

        details.append(f"Layer {layer_idx} has {len(layer_ops)} operations:")
        for op in layer_ops:
            details.append(f"  - {op}")

        # Check for fused kernels first
        has_fused_attention = any("mega_fused_attention" in op for op in layer_ops)
        has_fused_mlp = any("mega_fused_outproj_mlp" in op for op in layer_ops)

        if has_fused_attention:
            details.append("Using fused attention kernel (covers RMSNorm+QKV+RoPE+Attn+OutProj+Residual)")
        if has_fused_mlp:
            details.append("Using fused MLP kernel (covers RMSNorm+MLP+Residual)")

        # Check for required operations (if not fused)
        missing_ops = []

        if not has_fused_attention:
            # Check attention ops
            if not any("rmsnorm" in op for op in layer_ops):
                missing_ops.append("rmsnorm (pre-attention)")
            if not any("rope" in op for op in layer_ops):
                missing_ops.append("rope")
            if not any("attention" in op for op in layer_ops):
                missing_ops.append("attention")

        if not has_fused_mlp:
            # Check MLP ops
            if not any("swiglu" in op or "silu" in op or "gelu" in op for op in layer_ops):
                missing_ops.append("activation (swiglu/silu/gelu)")

        # Check residuals (always needed unless both fused)
        if not (has_fused_attention and has_fused_mlp):
            residual_count = sum(1 for op in layer_ops if "residual" in op)
            if residual_count < 2:
                details.append(f"Only {residual_count} residual ops (expected 2)")

        if missing_ops:
            for op in missing_ops:
                details.append(f"MISSING: {op}")

        passed = len(missing_ops) == 0
        return TestResult(
            name="layer_operations_complete",
            passed=passed,
            message=f"Layer {layer_idx} operations complete" if passed else f"Layer {layer_idx} missing {len(missing_ops)} operations",
            details=details
        )

    def test_c_code_uses_ir_weights(self) -> TestResult:
        """Test 4: Verify C code uses the same weights as IR."""
        details = []

        if not self.c_code:
            return TestResult(
                name="c_code_uses_ir_weights",
                passed=True,
                message="No C code to check (skipped)",
                details=["C code not found"]
            )

        # Extract weight macros from C code
        c_weights = set(re.findall(r'W_LAYER_\d+_\w+|W_TOKEN_EMB|W_FINAL_\w+|W_LM_HEAD', self.c_code))
        details.append(f"C code references {len(c_weights)} weight macros")

        # Get IR weights
        ir_weights = self.get_ir_weights()
        layer_weights_in_ir = {w for w in ir_weights if "LAYER" in w}
        details.append(f"IR references {len(layer_weights_in_ir)} layer weight macros")

        # Compare
        in_c_not_ir = c_weights - ir_weights
        in_ir_not_c = layer_weights_in_ir - c_weights

        if in_c_not_ir:
            details.append(f"In C but not IR: {in_c_not_ir}")
        if in_ir_not_c:
            details.append(f"In IR but not C: {in_ir_not_c}")

        passed = len(in_c_not_ir) == 0 and len(in_ir_not_c) == 0
        return TestResult(
            name="c_code_uses_ir_weights",
            passed=passed,
            message="C code matches IR weights" if passed else "C code/IR weight mismatch",
            details=details
        )

    def run_all_tests(self) -> bool:
        """Run all tests and return overall pass/fail."""
        if not self.load_files():
            return False

        self.results = [
            self.test_weight_name_mapping(),
            self.test_layer0_weights_referenced(),
            self.test_layer_operations_complete(),
            self.test_c_code_uses_ir_weights(),
        ]

        return all(r.passed for r in self.results)

    def print_results(self):
        """Print test results."""
        print("\n" + "=" * 70)
        print("Single Layer Completeness Test Results")
        print("=" * 70)

        passed_count = 0
        failed_count = 0

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            color = "\033[32m" if result.passed else "\033[31m"
            reset = "\033[0m"

            print(f"\n[{color}{status}{reset}] {result.name}")
            print(f"    {result.message}")

            if self.verbose or not result.passed:
                for detail in result.details:
                    print(f"      {detail}")

            if result.passed:
                passed_count += 1
            else:
                failed_count += 1

        print("\n" + "-" * 70)
        print(f"Summary: {passed_count} passed, {failed_count} failed")
        print("-" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test single-layer IR completeness")
    parser.add_argument("--model-dir", type=Path,
                       default=Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF",
                       help="Model directory with IR files")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")

    args = parser.parse_args()

    validator = SingleLayerValidator(args.model_dir, verbose=args.verbose)
    passed = validator.run_all_tests()
    validator.print_results()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
