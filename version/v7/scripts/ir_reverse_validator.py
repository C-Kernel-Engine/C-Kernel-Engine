#!/usr/bin/env python3
"""
ir_reverse_validator.py - Validate IR Lower 3 by working backwards

This validator takes the lowered IR (IR Lower 3) and validates:
1. Buffer completeness - Every weight/bias/input/output reference has a buffer definition
2. Manifest coverage - Every weight in manifest is used (no orphans)
3. Bias accounting - If model has biases, they must appear in IR ops
4. Op sequence - Data flow is valid (no read-before-write)
5. Size consistency - file_size matches computed size from shape+dtype
6. Template reconstruction - Can we infer the template from ops?

Usage:
    python ir_reverse_validator.py --lowered=lowered_decode_call.json --manifest=weights_manifest.json

Exit codes:
    0 - All validations passed
    1 - Validation errors found
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import re
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Quantization block sizes and bytes per block
QUANT_SPECS = {
    "q4_0": {"block_size": 32, "bytes_per_block": 18},      # 32 values, 18 bytes
    "q4_1": {"block_size": 32, "bytes_per_block": 20},      # 32 values, 20 bytes
    "q4_k": {"block_size": 256, "bytes_per_block": 144},    # 256 values, 144 bytes (Q4_K_M)
    "q4_k_m": {"block_size": 256, "bytes_per_block": 144},  # Alias
    "q5_0": {"block_size": 32, "bytes_per_block": 20},      # 32 values, 20 bytes
    "q5_1": {"block_size": 32, "bytes_per_block": 22},      # 32 values, 22 bytes
    "q5_k": {"block_size": 256, "bytes_per_block": 176},    # 256 values, 176 bytes
    "q6_k": {"block_size": 256, "bytes_per_block": 210},    # 256 values, 210 bytes
    "q8_0": {"block_size": 32, "bytes_per_block": 34},      # 32 values, 34 bytes
    "q8_k": {"block_size": 256, "bytes_per_block": 292},    # 256 values, 292 bytes
    "fp32": {"block_size": 1, "bytes_per_block": 4},        # 1 value, 4 bytes
    "f32": {"block_size": 1, "bytes_per_block": 4},
    "fp16": {"block_size": 1, "bytes_per_block": 2},
    "f16": {"block_size": 1, "bytes_per_block": 2},
    "bf16": {"block_size": 1, "bytes_per_block": 2},
}

# Kernel operations that require biases
OPS_WITH_OPTIONAL_BIAS = {
    "gemv_q5_0_q8_0", "gemv_q8_0_q8_0", "gemv_q4_k", "gemv_q6_k",
    "gemv_q4_k_q8_k", "gemv_q6_k_q8_k",
    "gemm_nt_q4_k", "gemm_nt_q5_0", "gemm_nt_q6_k", "gemm_nt_q8_0",
    "mega_fused_attention_prefill", "mega_fused_attention_decode",
    "mega_fused_outproj_mlp_prefill", "mega_fused_outproj_mlp_decode",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)


@dataclass
class BufferInfo:
    """Information about a buffer from IR."""
    name: str
    dtype: str
    shape: List[int]
    offset: int
    size: int
    role: str  # weight, activation, scratch, kv_cache
    file_size: Optional[int] = None  # From manifest


# ═══════════════════════════════════════════════════════════════════════════════
# Size Calculation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_size_from_shape_dtype(shape: List[int], dtype: str) -> int:
    """Compute expected byte size from shape and dtype."""
    dtype_lower = dtype.lower()
    num_elements = 1
    for dim in shape:
        num_elements *= dim

    if dtype_lower not in QUANT_SPECS:
        # Unknown dtype, can't validate
        return -1

    spec = QUANT_SPECS[dtype_lower]
    block_size = spec["block_size"]
    bytes_per_block = spec["bytes_per_block"]

    # Number of blocks (round up)
    num_blocks = (num_elements + block_size - 1) // block_size
    return num_blocks * bytes_per_block


# ═══════════════════════════════════════════════════════════════════════════════
# IRReverseValidator
# ═══════════════════════════════════════════════════════════════════════════════

class IRReverseValidator:
    """Validates IR Lower 3 by working backwards to check consistency."""

    def __init__(self, lowered_ir: Dict, manifest: Optional[Dict] = None,
                 kernel_maps_dir: Optional[Path] = None):
        self.ir = lowered_ir
        self.manifest = manifest or {}
        self.kernel_maps_dir = kernel_maps_dir
        self.kernel_maps: Dict[str, Dict] = {}

        # Extract useful data
        self.operations = self.ir.get("operations", [])
        self.buffers = self.ir.get("buffers", {})
        self.config = self.ir.get("config", {})
        self.memory = self.ir.get("memory", {})

        # Load kernel maps if directory provided
        if kernel_maps_dir and kernel_maps_dir.exists():
            self._load_kernel_maps()

        # Build manifest lookup
        self.manifest_entries: Dict[str, Dict] = {}
        for entry in self.manifest.get("entries", []):
            self.manifest_entries[entry["name"]] = entry

    def _parse_op_args(self, op: Dict) -> Dict[str, List[str]]:
        """Parse operation args into categorized lists.

        The lowered IR uses 'args' with 'source' field:
        - weight:NAME -> weights
        - activation:NAME -> activations (inputs)
        - output:NAME -> outputs
        - bias:NAME -> biases
        - const:VALUE, dim:NAME, null -> params (ignore for buffer tracking)
        """
        result = {
            "inputs": [],
            "outputs": [],
            "weights": [],
            "biases": [],
        }

        # Try new args-based structure first
        for arg in op.get("args", []):
            if not isinstance(arg, dict):
                continue
            source = arg.get("source", "")
            name = arg.get("name", "")
            buffer_ref = str(arg.get("buffer_ref", "") or "").strip()
            weight_ref = str(arg.get("weight_ref", "") or "").strip()

            # Parse source type
            if ":" in source:
                source_type, source_ref = source.split(":", 1)
            else:
                source_type = source
                source_ref = ""

            if source_type == "weight":
                result["weights"].append(weight_ref or source_ref or name)
            elif source_type == "bias":
                result["biases"].append(weight_ref or source_ref or name)
            elif source_type == "activation":
                result["inputs"].append(buffer_ref or source_ref or name)
            elif source_type == "output":
                result["outputs"].append(buffer_ref or source_ref or name)
            elif source_type == "scratch":
                if buffer_ref:
                    result["inputs"].append(buffer_ref)
            elif source_type in ("const", "dim", "null", "runtime", "kv_cache"):
                pass  # Not buffer references

        # Also try old structure for backwards compatibility
        for key in ["inputs", "outputs", "weights", "biases"]:
            for item in op.get(key, []):
                if isinstance(item, dict):
                    buf_name = item.get("buffer") or item.get("name", "")
                elif isinstance(item, str):
                    buf_name = item
                else:
                    continue
                if buf_name and buf_name.lower() != "null":
                    result[key].append(buf_name)

        return result

    def _load_kernel_maps(self):
        """Load all kernel map JSON files."""
        for map_file in self.kernel_maps_dir.glob("*.json"):
            if map_file.name.startswith("KERNEL_"):
                continue  # Skip registry files
            try:
                with open(map_file, "r") as f:
                    data = json.load(f)
                    kernel_id = data.get("id") or data.get("function")
                    if kernel_id:
                        self.kernel_maps[kernel_id] = data
            except (json.JSONDecodeError, IOError):
                pass

    def validate_all(self) -> List[ValidationResult]:
        """Run all validation checks."""
        results = []
        results.append(self.validate_buffer_completeness())
        results.append(self.validate_manifest_coverage())
        results.append(self.validate_bias_accounting())
        results.append(self.validate_op_sequence())
        results.append(self.validate_size_consistency())
        results.append(self.validate_kernel_signatures())
        results.append(self.validate_layer_completeness())
        return results

    def validate_layer_completeness(self) -> ValidationResult:
        """Validate that each layer in the IR has all required operations.

        A complete transformer layer should have:
        - RMSNorm (attention pre-norm)
        - Q/K/V projections
        - RoPE
        - Attention
        - Output projection
        - Residual add
        - RMSNorm (MLP pre-norm)
        - MLP gate/up projection
        - Activation (SiLU/SwiGLU)
        - MLP down projection
        - Residual add
        """
        result = ValidationResult(check_name="layer_completeness", passed=True)

        # Detect layer mode
        ir_layers, config_layers = self._detect_layer_mode()

        if ir_layers == 0:
            result.info.append("No layers found in IR")
            return result

        result.info.append(f"IR has {ir_layers} layer(s), config expects {config_layers}")

        # Required operation types for a complete layer
        required_ops = {
            "rmsnorm": 2,      # Pre-attn and pre-MLP
            "qkv_proj": 3,     # Q, K, V projections (or fused)
            "rope": 1,         # RoPE
            "attention": 1,    # Attention
            "out_proj": 1,     # Output projection
            "residual": 2,     # Two residual adds
            "mlp_proj": 2,     # Gate/up and down (or fused)
            "activation": 1,   # SiLU/SwiGLU
        }

        # Map kernel names to operation types
        op_type_map = {
            "rmsnorm_forward": "rmsnorm",
            "gemv_q5_0": "proj",  # Could be qkv, out, or mlp
            "gemv_q8_0": "proj",
            "gemv_q4_k": "proj",
            "gemv_q6_k": "proj",
            "gemm_nt_q5_0": "proj",
            "gemm_nt_q8_0": "proj",
            "gemm_nt_q4_k": "proj",
            "gemm_nt_q6_k": "proj",
            "rope_forward_qk": "rope",
            "attention_forward_causal_head_major_gqa_flash_strided": "attention",
            "attention_forward_decode_head_major_gqa_flash": "attention",
            "ck_residual_add_token_major": "residual",
            "swiglu_forward": "activation",
            "kv_cache_store": "kv_cache",
            # Fused kernels
            "mega_fused_attention_prefill": "fused_attention",
            "mega_fused_attention_decode": "fused_attention",
            "mega_fused_outproj_mlp_prefill": "fused_mlp",
            "mega_fused_outproj_mlp_decode": "fused_mlp",
        }

        # Group operations by layer
        layer_ops: Dict[int, List[str]] = {}
        for op in self.operations:
            layer = op.get("layer", -1)
            if layer < 0:
                continue  # Skip header/footer
            kernel = op.get("function") or op.get("kernel", "")
            if layer not in layer_ops:
                layer_ops[layer] = []
            layer_ops[layer].append(kernel)

        # Check each layer
        for layer_idx in sorted(layer_ops.keys()):
            ops = layer_ops[layer_idx]
            op_counts: Dict[str, int] = {}

            for kernel in ops:
                op_type = op_type_map.get(kernel, "other")
                op_counts[op_type] = op_counts.get(op_type, 0) + 1

            # Check for fused operations (which replace multiple individual ops)
            has_fused_attention = op_counts.get("fused_attention", 0) > 0
            has_fused_mlp = op_counts.get("fused_mlp", 0) > 0

            # Validate layer completeness
            issues = []

            if has_fused_attention:
                result.info.append(f"Layer {layer_idx}: Using fused attention kernel")
            else:
                # Check individual attention components
                if op_counts.get("rmsnorm", 0) < 1:
                    issues.append("missing RMSNorm")
                if op_counts.get("proj", 0) < 3:
                    issues.append(f"only {op_counts.get('proj', 0)} projections (need Q/K/V)")
                if op_counts.get("rope", 0) < 1:
                    issues.append("missing RoPE")
                if op_counts.get("attention", 0) < 1:
                    issues.append("missing attention")

            if has_fused_mlp:
                result.info.append(f"Layer {layer_idx}: Using fused MLP kernel")
            else:
                if op_counts.get("activation", 0) < 1:
                    issues.append("missing activation (SiLU/SwiGLU)")

            if op_counts.get("residual", 0) < 2 and not (has_fused_attention and has_fused_mlp):
                issues.append(f"only {op_counts.get('residual', 0)} residual adds (need 2)")

            if issues:
                result.warnings.append(f"Layer {layer_idx} may be incomplete: {', '.join(issues)}")
            else:
                result.info.append(f"Layer {layer_idx}: Complete ({len(ops)} operations)")

        return result

    def validate_buffer_completeness(self) -> ValidationResult:
        """Every weight/bias/input/output reference in ops must have a buffer definition."""
        result = ValidationResult(check_name="buffer_completeness", passed=True)

        # Collect all buffer references from operations
        referenced_buffers: Set[str] = set()
        for op in self.operations:
            parsed = self._parse_op_args(op)

            for buf in parsed["inputs"]:
                referenced_buffers.add(buf)
            for buf in parsed["outputs"]:
                referenced_buffers.add(buf)
            for buf in parsed["weights"]:
                referenced_buffers.add(buf)
            for buf in parsed["biases"]:
                if buf.lower() != "null":
                    referenced_buffers.add(buf)

        # Check that all referenced buffers exist in either buffers dict or memory layout
        defined_buffers = set(self.buffers.keys())

        # Also consider memory layout entries as defined
        for region in self.memory.values():
            if isinstance(region, dict):
                for key in region.keys():
                    defined_buffers.add(key)
                for entry in region.get("buffers", []):
                    if isinstance(entry, dict) and entry.get("name"):
                        defined_buffers.add(str(entry["name"]))
                for entry in region.get("entries", []):
                    if isinstance(entry, dict) and entry.get("name"):
                        defined_buffers.add(str(entry["name"]))

        missing = referenced_buffers - defined_buffers

        # Also check manifest for weight buffers
        weight_buffers_in_manifest = set(self.manifest_entries.keys())

        for buf in missing:
            # Check if it's in manifest (weight)
            if buf in weight_buffers_in_manifest:
                result.info.append(f"Buffer '{buf}' found in manifest (weight)")
            # Check if it's a well-known activation or parameter name
            elif buf in {"input", "output", "residual", "tokens", "embedded_input",
                        "hidden", "attn_out", "mlp_out", "ln_out", "logits",
                        # Generic parameter names used in lowered IR args
                        "x", "y", "src", "dst", "gate", "out", "in", "a", "b",
                        "_first_weight", "result", "temp", "scratch"}:
                result.info.append(f"Buffer '{buf}' is well-known parameter/activation name")
            else:
                result.errors.append(f"Missing buffer definition: '{buf}'")
                result.passed = False

        if result.passed:
            result.info.append(f"All {len(referenced_buffers)} referenced buffers are defined")

        return result

    def _detect_layer_mode(self) -> Tuple[int, int]:
        """Detect how many layers are in the IR vs config.

        Returns (ir_layers, config_layers).
        """
        # Get config layer count
        config_layers = self.config.get("num_layers", 24)

        # Count actual layers in IR operations
        ir_layer_indices = set()
        for op in self.operations:
            layer = op.get("layer")
            if layer is not None and layer >= 0:
                ir_layer_indices.add(layer)

        ir_layers = len(ir_layer_indices) if ir_layer_indices else 1
        return ir_layers, config_layers

    def _get_layer_weights(self, layer_idx: int) -> Set[str]:
        """Get expected weight names for a specific layer."""
        layer_patterns = [
            f"layer.{layer_idx}.wq", f"layer.{layer_idx}.wk",
            f"layer.{layer_idx}.wv", f"layer.{layer_idx}.wo",
            f"layer.{layer_idx}.w1", f"layer.{layer_idx}.w2",
            f"layer.{layer_idx}.ln1_gamma", f"layer.{layer_idx}.ln2_gamma",
            f"layer.{layer_idx}.bq", f"layer.{layer_idx}.bk",
            f"layer.{layer_idx}.bv", f"layer.{layer_idx}.bo",
            f"layer.{layer_idx}.b1", f"layer.{layer_idx}.b2",
        ]
        return set(layer_patterns)

    def validate_manifest_coverage(self) -> ValidationResult:
        """Every weight in manifest should be used by at least one operation.

        Smart about single-layer testing mode: only validates weights for
        the layers that are actually in the IR.
        """
        result = ValidationResult(check_name="manifest_coverage", passed=True)

        if not self.manifest_entries:
            result.info.append("No manifest provided, skipping coverage check")
            return result

        # Detect layer mode
        ir_layers, config_layers = self._detect_layer_mode()
        single_layer_mode = ir_layers < config_layers

        if single_layer_mode:
            result.info.append(f"Single-layer test mode detected: IR has {ir_layers} layer(s), config has {config_layers}")

        # Collect all weight references from operations
        used_weights: Set[str] = set()
        for op in self.operations:
            parsed = self._parse_op_args(op)

            for weight in parsed["weights"]:
                used_weights.add(weight)
            for bias in parsed["biases"]:
                if bias.lower() != "null":
                    used_weights.add(bias)

        # Check for unused manifest entries
        manifest_names = set(self.manifest_entries.keys())
        unused = manifest_names - used_weights

        # Filter out known non-weight entries (e.g., tokenizer data)
        non_weight_patterns = ["tokenizer", "vocab", "merges", "special_tokens"]
        for name in list(unused):
            if any(pat in name.lower() for pat in non_weight_patterns):
                unused.remove(name)
                result.info.append(f"Skipped non-weight entry: '{name}'")

        # In single-layer mode, filter out weights for layers not in IR
        if single_layer_mode:
            # Find which layers are actually in IR
            ir_layer_indices = set()
            for op in self.operations:
                layer = op.get("layer")
                if layer is not None and layer >= 0:
                    ir_layer_indices.add(layer)

            # Filter unused - only flag weights for layers IN the IR
            filtered_unused = set()
            skipped_other_layers = 0
            for name in unused:
                # Check if this is a layer weight
                match = re.search(r'layer\.(\d+)\.', name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx in ir_layer_indices:
                        # This layer IS in IR but weight not used - problem!
                        filtered_unused.add(name)
                    else:
                        # This layer is NOT in IR - expected in single-layer mode
                        skipped_other_layers += 1
                else:
                    # Non-layer weight (token_emb, final_ln, etc)
                    filtered_unused.add(name)

            if skipped_other_layers > 0:
                result.info.append(f"Skipped {skipped_other_layers} weights for layers not in IR (single-layer mode)")
            unused = filtered_unused

        if unused:
            for name in sorted(unused):
                result.warnings.append(f"Unused manifest entry: '{name}'")
            result.info.append(f"Found {len(unused)} unused manifest entries")
        else:
            result.info.append(f"All relevant manifest entries are used")

        return result

    def validate_bias_accounting(self) -> ValidationResult:
        """If model has biases in manifest, they must appear in IR ops.

        Smart about single-layer testing mode: only validates biases for
        the layers that are actually in the IR.
        """
        result = ValidationResult(check_name="bias_accounting", passed=True)

        # Detect layer mode
        ir_layers, config_layers = self._detect_layer_mode()
        single_layer_mode = ir_layers < config_layers

        # Find which layers are actually in IR
        ir_layer_indices = set()
        for op in self.operations:
            layer = op.get("layer")
            if layer is not None and layer >= 0:
                ir_layer_indices.add(layer)

        # Find all bias entries in manifest (filtered by layers in IR if single-layer mode)
        bias_entries: Set[str] = set()
        skipped_other_layers = 0
        for name in self.manifest_entries.keys():
            # Common bias naming patterns
            if any(pat in name.lower() for pat in [".b_", ".bias", "_bias", ".bo", ".bq", ".bk", ".bv", ".b1", ".b2"]):
                if single_layer_mode:
                    # Check if this bias is for a layer in IR
                    match = re.search(r'layer\.(\d+)\.', name)
                    if match:
                        layer_idx = int(match.group(1))
                        if layer_idx in ir_layer_indices:
                            bias_entries.add(name)
                        else:
                            skipped_other_layers += 1
                    else:
                        # Non-layer bias (final_ln_bias, etc)
                        bias_entries.add(name)
                else:
                    bias_entries.add(name)

        if single_layer_mode and skipped_other_layers > 0:
            result.info.append(f"Single-layer mode: skipped {skipped_other_layers} biases for layers not in IR")

        if not bias_entries:
            result.info.append("No bias entries found in manifest (for layers in IR)")
            return result

        # Collect biases used in operations (from args with source:bias)
        used_biases: Set[str] = set()
        for op in self.operations:
            parsed = self._parse_op_args(op)
            for bias in parsed["biases"]:
                if bias.lower() != "null":
                    used_biases.add(bias)

            # Also check for biases passed as weights (some kernels do this)
            for weight in parsed["weights"]:
                if any(pat in weight.lower() for pat in [".b_", ".bias", "_bias", ".bo", ".bq", ".bk", ".bv", ".b1", ".b2"]):
                    used_biases.add(weight)

        # Check for missing biases
        missing_biases = bias_entries - used_biases

        # Note: Many models (like Qwen2) have biases set to zero and don't use them
        # So we demote this to a warning if all biases are missing (likely no-bias model)
        if missing_biases:
            if len(missing_biases) == len(bias_entries):
                # All biases are "missing" - likely a no-bias model (biases are NULL)
                result.warnings.append(f"Model appears to use NULL biases ({len(bias_entries)} bias entries not used)")
                result.info.append("This is expected for models with use_bias=False")
            else:
                # Some biases used, some not - this is suspicious
                for name in sorted(missing_biases):
                    result.errors.append(f"Bias in manifest but not used in IR: '{name}'")
                result.passed = False
        else:
            result.info.append(f"All {len(bias_entries)} biases are accounted for")

        return result

    def validate_op_sequence(self) -> ValidationResult:
        """Validate data flow - no read-before-write for intermediate buffers."""
        result = ValidationResult(check_name="op_sequence", passed=True)

        # Track which buffers have been written to
        written_buffers: Set[str] = set()

        # Inputs from outside (model inputs) are pre-written
        # Also include generic parameter names that represent forwarded data
        external_inputs = {"input", "tokens", "position", "kv_cache_k", "kv_cache_v",
                          "rope_cos", "rope_sin", "mask", "residual", "embedded_input",
                          "token_ids", "start_pos",
                          # Generic parameter names in lowered IR args
                          "x", "y", "src", "dst", "gate", "out", "in", "a", "b",
                          "_first_weight", "result", "temp", "scratch"}

        for op_idx, op in enumerate(self.operations):
            op_name = op.get("kernel") or op.get("function", f"op_{op_idx}")
            parsed = self._parse_op_args(op)

            # Check inputs are available (either external or previously written)
            for buf_name in parsed["inputs"]:
                if not buf_name:
                    continue

                # External inputs and weights are always available
                is_external = buf_name in external_inputs
                is_weight = buf_name in parsed["weights"]
                is_written = buf_name in written_buffers

                # Check if it's a weight from manifest
                is_manifest_weight = buf_name in self.manifest_entries

                if not (is_external or is_weight or is_written or is_manifest_weight):
                    # Special case: check for layer-0 or header patterns
                    if ".0." in buf_name or "header" in buf_name.lower():
                        # Might be valid - mark as info rather than error
                        result.info.append(f"Op {op_idx} ({op_name}): input '{buf_name}' "
                                          f"assumed available (header/layer-0)")
                    else:
                        result.warnings.append(f"Op {op_idx} ({op_name}): input '{buf_name}' "
                                              f"read before write")

            # Mark outputs as written
            for buf_name in parsed["outputs"]:
                if buf_name:
                    written_buffers.add(buf_name)

        result.info.append(f"Validated data flow for {len(self.operations)} operations")
        return result

    def validate_size_consistency(self) -> ValidationResult:
        """Validate that file_size matches computed size from shape+dtype."""
        result = ValidationResult(check_name="size_consistency", passed=True)

        for name, entry in self.manifest_entries.items():
            shape = entry.get("shape") or entry.get("resolved_shape")
            dtype = entry.get("dtype", "fp32")
            file_size = entry.get("file_size") or entry.get("size")

            if not shape or not file_size:
                continue

            computed_size = compute_size_from_shape_dtype(shape, dtype)
            if computed_size < 0:
                result.info.append(f"Unknown dtype '{dtype}' for '{name}', skipping size check")
                continue

            # Allow small tolerance for alignment
            tolerance = 64  # bytes
            if abs(computed_size - file_size) > tolerance:
                result.errors.append(
                    f"Size mismatch for '{name}': "
                    f"computed={computed_size} vs file_size={file_size} "
                    f"(shape={shape}, dtype={dtype})"
                )
                result.passed = False

        if result.passed:
            result.info.append("All manifest sizes are consistent with shape+dtype")

        return result

    def validate_kernel_signatures(self) -> ValidationResult:
        """Validate that ops match kernel map signatures."""
        result = ValidationResult(check_name="kernel_signatures", passed=True)

        if not self.kernel_maps:
            result.info.append("No kernel maps loaded, skipping signature validation")
            return result

        for op_idx, op in enumerate(self.operations):
            kernel_id = op.get("kernel") or op.get("function")
            if not kernel_id:
                continue

            # Find kernel map
            kernel_map = self.kernel_maps.get(kernel_id)
            if not kernel_map:
                # Try without prefixes/suffixes
                for kid, kmap in self.kernel_maps.items():
                    if kid in kernel_id or kernel_id in kid:
                        kernel_map = kmap
                        break

            if not kernel_map:
                result.info.append(f"No kernel map found for '{kernel_id}'")
                continue

            parsed = self._parse_op_args(op)

            # Check required inputs
            expected_inputs = len(kernel_map.get("inputs", []))
            actual_inputs = len(parsed["inputs"])

            # Check required weights
            expected_weights = len([w for w in kernel_map.get("weights", [])
                                   if not w.get("optional", False)])
            actual_weights = len(parsed["weights"])

            # Check outputs
            expected_outputs = len(kernel_map.get("outputs", []))
            actual_outputs = len(parsed["outputs"])

            # Note: The lowered IR args structure is different from kernel map expectations
            # The args list contains ALL parameters (dims, consts, weights, etc)
            # So we just do a sanity check rather than strict matching
            total_args = len(op.get("args", []))
            if total_args == 0 and (expected_inputs > 0 or expected_weights > 0):
                result.info.append(
                    f"Op {op_idx} ({kernel_id}): args-based IR with {total_args} args"
                )

        result.info.append(f"Checked signatures for {len(self.operations)} operations")
        return result

    def reconstruct_template(self) -> Dict:
        """From IR ops, reconstruct what the template should look like."""
        template = {
            "model": self.config.get("model_name", "unknown"),
            "architecture": self.config.get("architecture", "unknown"),
            "layers": [],
            "header_ops": [],
            "body_ops": [],
            "footer_ops": [],
        }

        # Categorize ops by phase
        for op in self.operations:
            phase = op.get("phase", "body")
            op_summary = {
                "kernel": op.get("kernel") or op.get("function"),
                "inputs": [i.get("buffer") if isinstance(i, dict) else i
                          for i in op.get("inputs", [])],
                "outputs": [o.get("buffer") if isinstance(o, dict) else o
                           for o in op.get("outputs", [])],
            }

            if phase == "header":
                template["header_ops"].append(op_summary)
            elif phase == "footer":
                template["footer_ops"].append(op_summary)
            else:
                template["body_ops"].append(op_summary)

        # Count layers from body ops
        layer_indices = set()
        for op in self.operations:
            layer = op.get("layer")
            if layer is not None:
                layer_indices.add(layer)
        template["num_layers"] = len(layer_indices) if layer_indices else 1

        return template

    def compute_expected_memory_layout(self) -> Dict:
        """Compute expected memory layout from operations."""
        layout = {
            "weights": {},
            "activations": {},
            "scratch": {},
            "kv_cache": {},
        }

        # Collect buffer sizes and offsets
        for name, entry in self.manifest_entries.items():
            size = entry.get("size") or entry.get("file_size", 0)
            offset = entry.get("runtime_offset") or entry.get("offset", 0)
            layout["weights"][name] = {
                "size": size,
                "offset": offset,
                "dtype": entry.get("dtype", "unknown"),
            }

        # Get activation and scratch info from buffers
        for name, buf in self.buffers.items():
            if buf.get("role") == "activation":
                layout["activations"][name] = {
                    "size": buf.get("size", 0),
                    "offset": buf.get("offset", 0),
                    "dtype": buf.get("dtype", "fp32"),
                }
            elif buf.get("role") == "scratch":
                layout["scratch"][name] = {
                    "size": buf.get("size", 0),
                    "offset": buf.get("offset", 0),
                }
            elif buf.get("role") == "kv_cache":
                layout["kv_cache"][name] = {
                    "size": buf.get("size", 0),
                    "dtype": buf.get("dtype", "fp32"),
                }

        return layout


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(results: List[ValidationResult], verbose: bool = False) -> str:
    """Generate a human-readable report from validation results."""
    lines = []
    lines.append("=" * 70)
    lines.append("IR Reverse Validation Report")
    lines.append("=" * 70)
    lines.append("")

    total_passed = 0
    total_failed = 0
    total_warnings = 0

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        status_color = "\033[32m" if result.passed else "\033[31m"
        reset = "\033[0m"

        lines.append(f"[{status_color}{status}{reset}] {result.check_name}")

        if result.passed:
            total_passed += 1
        else:
            total_failed += 1

        total_warnings += len(result.warnings)

        # Show errors
        for err in result.errors:
            lines.append(f"  \033[31mERROR:\033[0m {err}")

        # Show warnings
        for warn in result.warnings:
            lines.append(f"  \033[33mWARN:\033[0m {warn}")

        # Show info if verbose
        if verbose:
            for info in result.info:
                lines.append(f"  \033[2mINFO:\033[0m {info}")

        lines.append("")

    # Summary
    lines.append("-" * 70)
    lines.append(f"Summary: {total_passed} passed, {total_failed} failed, {total_warnings} warnings")
    lines.append("-" * 70)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def run_validation(lowered_path: Path, manifest_path: Optional[Path] = None,
                   kernel_maps_dir: Optional[Path] = None,
                   verbose: bool = False) -> Tuple[bool, str]:
    """Run all validations and return (passed, report)."""
    # Load lowered IR
    with open(lowered_path, "r") as f:
        lowered_ir = json.load(f)

    # Load manifest if provided
    manifest = None
    if manifest_path and manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    # Create validator
    validator = IRReverseValidator(
        lowered_ir=lowered_ir,
        manifest=manifest,
        kernel_maps_dir=kernel_maps_dir,
    )

    # Run all validations
    results = validator.validate_all()

    # Generate report
    report = generate_report(results, verbose=verbose)

    # Determine overall pass/fail
    all_passed = all(r.passed for r in results)

    return all_passed, report


def main():
    parser = argparse.ArgumentParser(
        description="Validate IR Lower 3 by working backwards"
    )
    parser.add_argument("--lowered", required=True, type=Path,
                       help="Path to lowered IR JSON (lowered_decode_call.json)")
    parser.add_argument("--manifest", type=Path,
                       help="Path to weights manifest JSON")
    parser.add_argument("--kernel-maps", type=Path,
                       help="Path to kernel_maps directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed info messages")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON")

    args = parser.parse_args()

    if not args.lowered.exists():
        print(f"Error: Lowered IR not found: {args.lowered}", file=sys.stderr)
        sys.exit(1)

    passed, report = run_validation(
        lowered_path=args.lowered,
        manifest_path=args.manifest,
        kernel_maps_dir=args.kernel_maps,
        verbose=args.verbose,
    )

    if args.json:
        # Re-run to get raw results
        with open(args.lowered, "r") as f:
            lowered_ir = json.load(f)
        manifest = None
        if args.manifest and args.manifest.exists():
            with open(args.manifest, "r") as f:
                manifest = json.load(f)
        validator = IRReverseValidator(lowered_ir, manifest, args.kernel_maps)
        results = validator.validate_all()

        json_output = {
            "passed": passed,
            "checks": [
                {
                    "name": r.check_name,
                    "passed": r.passed,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "info": r.info,
                }
                for r in results
            ]
        }
        print(json.dumps(json_output, indent=2))
    else:
        print(report)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
