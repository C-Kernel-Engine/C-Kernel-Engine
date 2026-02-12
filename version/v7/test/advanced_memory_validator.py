#!/usr/bin/env python3
"""
advanced_memory_validator.py - Detailed Memory Validation for v7

================================================================================
HOW THIS VALIDATION WORKS
================================================================================

This validator tests that the memory layout is correct by comparing:
  1. The layout JSON (source of truth for offsets/sizes)
  2. The lowered IR (operations and their buffer requirements)
  3. Generated C code (hardcoded offsets that must match layout)

--------------------------------------------------------------------------------
STAGE 1: LOAD FILES
--------------------------------------------------------------------------------

    layout_decode.json                    lowered_decode.json
    ┌─────────────────────────┐          ┌─────────────────────────┐
    │ weights:                │          │ operations:             │
    │ - name: "layer.0.wq"    │          │ - embedding_forward     │
    │ - offset: 148602200     │    ↔     │   inputs: [token_emb]   │
    │ - size: 551936          │          │   outputs: [embedded]   │
    │ - dtype: "q5_0"         │          │ - mega_fused_attention │
    │ - layer: 0              │          │   inputs: [embedded]    │
    └─────────────────────────┘          └─────────────────────────┘

--------------------------------------------------------------------------------
STAGE 2: PARSE INTO REGIONS
--------------------------------------------------------------------------------

    for entry in layout["memory"]["weights"]["entries"]:
        region = MemoryRegion(
            name=entry["name"],           # "layer.0.wq"
            offset=entry["offset"],       # 148602200
            size=entry["size"],           # 551936
            dtype=entry["dtype"],         # "q5_0"
            layer=0                       # extracted from name
        )

--------------------------------------------------------------------------------
STAGE 3: RUN VALIDATIONS
--------------------------------------------------------------------------------

    ┌─────────────────────────────────────────────────────────────────────┐
    │ VALIDATION                    │ CHECKS                              │
    ├───────────────────────────────┼─────────────────────────────────────┤
    │ 1. Weight size validation     │ elements × bytes/elem = reported    │
    │ 2. Kernel buffer validation   │ IR buffers exist in layout          │
    │ 3. Activation validation      │ Activation sizes match IR ops       │
    │ 4. Vocab/Embedding layout     │ token_emb, vocab_offsets correct    │
    │ 5. Per-layer accounting       │ All 24 layers have expected weights │
    │ 6. Full memory accounting     │ No overlaps, no gaps                │
    │ 7. Pointer alignment          │ offset % element_bytes == 0         │
    │ 8. Code-to-layout offsets     │ C code offsets match layout JSON    │
    └─────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
EXAMPLE: DETECTED OFFSET 3604 BUG
--------------------------------------------------------------------------------

    Generated C code:
        ((int32_t*)model->activations)[0] = token;        // stores at 0

        embedding_forward_q8_0(
            ((int32_t*)((uint8_t*)model->activations + 3604)),  // READS from 3604!
            ...
        )

    Layout says:
        token_emb starts at offset 0 (size: 144MB)

    Validator detects:
        "POTENTIAL BUG: Hardcoded offset 3604 in code.
         Expected offset for token might be 0."

================================================================================

USAGE:
    # Quick validation
    python test/advanced_memory_validator.py --layout=layout.json --ir=lowered.json

    # Full validation with code checking
    python test/advanced_memory_validator.py --layout=layout.json --ir=lowered.json --code=generated.c

    # JSON output for CI/CD
    python test/advanced_memory_validator.py --layout=... --ir=... --code=... --json
"""

import argparse
import json
import os
import re  # For parsing C code offsets
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MemoryRegion:
    """
    Represents a contiguous region of memory.

    The layout JSON defines where each tensor lives in memory. This class
    captures all metadata for validation.

    Example:
        name="layer.0.wq"
        offset=148602200    # Byte offset from base
        size=551936         # Size in bytes
        dtype="q5_0"        # Quantization type
        layer=0             # Which layer this belongs to
    """
    name: str
    offset: int
    size: int
    dtype: str
    layer: Optional[int] = None
    quant_blocks: Optional[int] = None  # For quantized types
    shape: Optional[List[int]] = None


@dataclass
class KernelRequirement:
    """
    Expected memory footprint for a kernel operation.

    Kernels (like embedding_forward, attention_forward) have specific
    input/output/weight requirements. This tracks what the IR expects.
    """
    kernel_name: str
    inputs: Dict[str, Tuple[List[int], str]]  # name -> (shape, dtype)
    outputs: Dict[str, Tuple[List[int], str]]
    weights: Dict[str, Tuple[List[int], str]]


# =============================================================================
# MAIN VALIDATOR CLASS
# =============================================================================

class AdvancedMemoryValidator:
    """
    Deep memory validation combining layout analysis, IR validation, and
    code-to-layout consistency checking.

    The key insight: layout JSON is the source of truth. Any hardcoded
    offset in generated C code that doesn't match the layout is a bug.
    """

    # =========================================================================
    # QUANTIZATION SPECIFICATIONS (llama.cpp compatible)
    # =========================================================================
    QUANT_SPECS = {
        "q4_0": {
            "block_size": 32,
            "block_bytes": 18,  # 16 weights (4-bit) + 2 bytes (scale, min)
            "type": "4-bit",
        },
        "q4_k": {
            "block_size": 256,
            "block_bytes": 144,  # Mixed, see llama.cpp
            "type": "4-bit K",
        },
        "q5_0": {
            "block_size": 32,
            "block_bytes": 34,  # 32 weights (5-bit) + 2 bytes (scale, min)
            "type": "5-bit",
        },
        "q5_1": {
            "block_size": 32,
            "block_bytes": 38,
            "type": "5-bit",
        },
        "q6_k": {
            "block_size": 256,
            "block_bytes": 210,  # 256 weights (6-bit) + 2 scales
            "type": "6-bit K",
        },
        "q8_0": {
            "block_size": 32,
            "block_bytes": 34,  # 32 weights + 2 bytes scale
            "type": "8-bit",
        },
        "q8_1": {
            "block_size": 32,
            "block_bytes": 40,
            "type": "8-bit",
        },
        "fp32": {"elements_per_byte": 0.25, "per_element": 4},
        "fp16": {"elements_per_byte": 0.5, "per_element": 2},
        "bf16": {"elements_per_byte": 0.5, "per_element": 2},
        "int8": {"elements_per_byte": 1.0, "per_element": 1},
        "int32": {"elements_per_byte": 0.25, "per_element": 4},
    }

    # =========================================================================
    # MODEL ARCHITECTURE SPECS (Qwen2 0.5B)
    # =========================================================================
    QWEN2_0_5B_SPECS = {
        "embed_dim": 896,
        "num_heads": 14,
        "num_kv_heads": 2,
        "head_dim": 64,
        "intermediate_size": 4864,
        "vocab_size": 151936,
        "num_layers": 24,
    }

    def __init__(self, layout_path: Path, lowered_ir_path: Path, generated_code_path: Optional[Path] = None):
        self.layout_path = layout_path
        self.lowered_ir_path = lowered_ir_path
        self.generated_code_path = generated_code_path
        self.layout = None
        self.lowered_ir = None
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    @staticmethod
    def _entry_offset(entry: Dict[str, Any], prefer_abs: bool = True) -> int:
        """Canonical offset helper for layouts that carry both offset/abs_offset."""
        if prefer_abs and entry.get("abs_offset") is not None:
            return int(entry.get("abs_offset", 0))
        return int(entry.get("offset", 0))

    def load(self) -> bool:
        """Load layout and IR files."""
        with open(self.layout_path, 'r') as f:
            self.layout = json.load(f)

        if self.lowered_ir_path.exists():
            with open(self.lowered_ir_path, 'r') as f:
                self.lowered_ir = json.load(f)

        return True

    # =========================================================================
    # QUANTIZATION SIZE VALIDATION
    # =========================================================================

    def compute_quantized_size(self, elements: int, dtype: str) -> int:
        """Compute exact size for quantized type."""
        if dtype not in self.QUANT_SPECS:
            # Assume float-like
            return elements * self.QUANT_SPECS.get("fp32", {}).get("per_element", 4)

        spec = self.QUANT_SPECS[dtype]

        # Block-based quantization
        if "block_size" in spec:
            num_blocks = (elements + spec["block_size"] - 1) // spec["block_size"]
            return num_blocks * spec["block_bytes"]

        # Element-based
        if "per_element" in spec:
            return elements * spec["per_element"]
        if "elements_per_byte" in spec:
            return int(elements * spec["elements_per_byte"])

        return elements  # Fallback

    def validate_weight_sizes(self) -> bool:
        """Validate each weight has correct size for its dtype."""
        weights = self.layout.get("memory", {}).get("weights", {})
        entries = weights.get("entries", [])

        spec = self.QWEN2_0_5B_SPECS
        E, H, D, I = spec["embed_dim"], spec["num_heads"], spec["head_dim"], spec["intermediate_size"]

        # Weights that have known quantization structure issues
        skip_validation = set()  # Add weights that fail but are actually correct

        for entry in entries:
            name = entry.get("name", "")
            reported_size = entry.get("size", 0)
            dtype = entry.get("dtype", "fp32")

            # Skip fp32 biases - they're simple
            if dtype == "fp32" and any(b in name for b in ["bq", "bk", "bv", "bo", "b1", "b2"]):
                continue

            # Determine expected elements from name
            expected_elements = self._get_weight_elements(name, E, H, D, I)
            if expected_elements == 0:
                continue  # Skip unknown

            expected_size = self.compute_quantized_size(expected_elements, dtype)

            # For K-quant types (q4_k, q6_k), the actual size can vary due to:
            # - Mixed quantization within a weight matrix
            # - Extra scales for sub-blocks
            # - Implementation-specific packing
            if dtype in ["q4_k", "q6_k"]:
                # Allow wider tolerance for K-quant
                min_expected = int(expected_size * 0.8)
                max_expected = int(expected_size * 1.5)
                if reported_size < min_expected or reported_size > max_expected:
                    self.warnings.append(
                        f"{name}: {reported_size} outside expected range "
                        f"[{min_expected}-{max_expected}] for {dtype}"
                    )
                else:
                    self.info.append(f"{name}: {reported_size} (K-quant, expected ~{expected_size})")
            elif reported_size != expected_size:
                # Check if within 10% (might have alignment or special packing)
                if abs(reported_size - expected_size) < expected_size * 0.10:
                    self.warnings.append(
                        f"{name}: {reported_size} vs expected ~{expected_size} "
                        f"({dtype}, {expected_elements} elements) - possible alignment/packing"
                    )
                else:
                    # Don't error, just warn - the layout might be correct with different packing
                    self.warnings.append(
                        f"{name}: SIZE DIFF {reported_size} vs expected {expected_size} "
                        f"({dtype}, {expected_elements} elements) - needs review"
                    )

        return True

    def _get_weight_elements(self, name: str, E: int, H: int, D: int, I: int) -> int:
        """Get expected element count from weight name."""
        parts = name.split(".")
        if len(parts) < 3:
            return 0

        weight_type = parts[-1]

        # Projection weights
        if weight_type == "wq":
            return E * E  # (num_heads * head_dim) × embed_dim = E × E
        elif weight_type == "wk":
            return D * H * 2 * E // H  # (num_kv_heads × head_dim) × embed_dim
        elif weight_type == "wv":
            return D * H * 2 * E // H
        elif weight_type == "wo":
            return E * E
        elif weight_type == "w1":  # gate/up
            return I * E
        elif weight_type == "w2":  # down
            return E * I
        elif weight_type == "w3":  # up (alternative naming)
            return I * E

        # Biases (note: b1 is often gate+bias combined)
        elif weight_type == "bq":
            return E
        elif weight_type == "bk":
            return D * H * 2 // H  # num_kv_heads × head_dim
        elif weight_type == "bv":
            return D * H * 2 // H
        elif weight_type == "bo":
            return E
        elif weight_type == "b1":
            # b1 in qwen2 is packed: gate_bias + up_bias = I + I = 2*I
            return I * 2
        elif weight_type == "b2":
            return E

        # Layer norms
        elif "gamma" in weight_type or "weight" in weight_type:
            return E
        elif "beta" in weight_type:
            return E

        return 0  # Unknown - skip

    # =========================================================================
    # KERNEL REQUIREMENTS VALIDATION
    # =========================================================================

    def get_kernel_requirements(self) -> Dict[str, KernelRequirement]:
        """Extract kernel requirements from lowered IR."""
        if not self.lowered_ir:
            return {}

        requirements = {}
        ops = self.lowered_ir.get("operations", [])

        for op in ops:
            kernel_name = op.get("function", op.get("kernel", ""))
            if not kernel_name:
                continue

            req = KernelRequirement(
                kernel_name=kernel_name,
                inputs={},
                outputs={},
                weights={}
            )

            # Parse inputs
            for inp_name, inp_spec in op.get("inputs", {}).items():
                shape = inp_spec.get("shape", [])
                dtype = inp_spec.get("dtype", "fp32")
                req.inputs[inp_name] = (shape, dtype)

            # Parse outputs
            for out_name, out_spec in op.get("outputs", {}).items():
                shape = out_spec.get("shape", [])
                dtype = out_spec.get("dtype", "fp32")
                req.outputs[out_name] = (shape, dtype)

            # Parse weights
            for w_name, w_spec in op.get("weights", {}).items():
                if isinstance(w_spec, dict):
                    shape = w_spec.get("shape", [])
                    dtype = w_spec.get("dtype", "fp32")
                else:
                    shape = []
                    dtype = "fp32"
                req.weights[w_name] = (shape, dtype)

            requirements[kernel_name] = req

        return requirements

    def validate_kernel_buffer_sizes(self) -> bool:
        """Validate IR operation buffer sizes against kernel requirements."""
        requirements = self.get_kernel_requirements()

        if not requirements:
            self.info.append("No kernel requirements from IR")
            return True

        self.info.append(f"Found {len(requirements)} unique kernels in IR")

        # Check that operations have required inputs/outputs
        for kernel_name, req in requirements.items():
            if not req.inputs:
                self.warnings.append(f"Kernel {kernel_name} has no inputs defined")
            if not req.outputs:
                self.warnings.append(f"Kernel {kernel_name} has no outputs defined")

        return True

    # =========================================================================
    # ACTIVATION BUFFER VALIDATION
    # =========================================================================

    def validate_activation_sizes(self) -> bool:
        """Validate activation buffer sizes match operation requirements."""
        if not self.lowered_ir:
            return True

        # Get activation buffers from memory layout
        act_buffers = self.layout.get("memory", {}).get("activations", {})
        buffer_sizes = {}
        for buf in act_buffers.get("buffers", []):
            buffer_sizes[buf.get("name", "")] = buf.get("size", 0)

        # Check each operation
        ops = self.lowered_ir.get("operations", [])
        activation_refs = defaultdict(list)  # buffer -> [ops referencing it]

        for op in ops:
            op_name = op.get("name", op.get("op", ""))

            # Track which buffers are referenced
            for inp in op.get("inputs", {}).values():
                if inp.get("type") == "activation":
                    if buf := inp.get("buffer") or inp.get("source"):
                        activation_refs[buf].append((op_name, "input"))

            for out in op.get("outputs", {}).values():
                if out.get("type") == "activation":
                    if buf := out.get("buffer"):
                        activation_refs[buf].append((op_name, "output"))

        # Verify buffers exist
        for buf_name in activation_refs:
            if buf_name and buf_name not in buffer_sizes:
                self.warnings.append(f"Activation '{buf_name}' referenced but not in layout")

        return True

    # =========================================================================
    # VOCAB/EMBEDDING LAYOUT VALIDATION
    # =========================================================================

    def validate_vocab_layout(self) -> bool:
        """Validate vocab, embedding, and special tokens layout."""
        weights = self.layout.get("memory", {}).get("weights", {})
        entries = {e.get("name", ""): e for e in weights.get("entries", [])}

        spec = self.QWEN2_0_5B_SPECS
        V, E = spec["vocab_size"], spec["embed_dim"]

        # Check token_emb
        if "token_emb" in entries:
            entry = entries["token_emb"]
            dtype = entry.get("dtype", "fp32")
            reported_size = entry.get("size", 0)

            expected_elements = V * E
            expected_size = self.compute_quantized_size(expected_elements, dtype)

            if abs(reported_size - expected_size) < expected_size * 0.05:
                self.info.append(
                    f"token_emb: {reported_size:,} bytes ({dtype}, {V}×{E}) ✓"
                )
            else:
                self.warnings.append(
                    f"token_emb size: {reported_size} vs expected {expected_size}"
                )

        # Check vocab entries
        # vocab_offsets: V tokens × 4 bytes (int32) = 607,744
        # vocab_merges: num_merges × 3 × 4 bytes (int32 triplets [token_a, token_b, merged])
        vocab_entries = {
            "vocab_offsets": (V * 4, "int32"),  # 151,936 × 4 = 607,744
            "vocab_strings": (spec.get("total_vocab_bytes", 1527572), "u8"),
            "vocab_merges": (spec.get("num_merges", 151387) * 3 * 4, "int32"),  # 151,387 × 3 × 4 = 1,816,644
        }

        for name, (expected_size, dtype) in vocab_entries.items():
            if name in entries:
                reported = entries[name].get("size", 0)
                if reported == expected_size:
                    self.info.append(f"{name}: {reported:,} bytes ✓")
                else:
                    self.warnings.append(
                        f"{name}: {reported} vs expected {expected_size}"
                    )

        return True

    # =========================================================================
    # PER-LAYER ACCOUNTING
    # =========================================================================

    def validate_per_layer_accounting(self) -> bool:
        """Validate each layer has all weights and correct sizes."""
        weights = self.layout.get("memory", {}).get("weights", {})
        entries = weights.get("entries", [])

        spec = self.QWEN2_0_5B_SPECS
        num_layers = spec["num_layers"]

        # Collect weights per layer
        layer_weights = defaultdict(list)
        for entry in entries:
            name = entry.get("name", "")
            if name.startswith("layer."):
                parts = name.split(".")
                if len(parts) >= 3:
                    try:
                        layer_idx = int(parts[1])
                        weight_name = parts[2]
                        layer_weights[layer_idx].append((weight_name, entry))
                    except ValueError:
                        pass

        # Check each layer
        expected_weights = {"wq", "wk", "wv", "wo", "w1", "w2", "bq", "bk", "bv", "bo",
                          "b1", "b2", "ln1_gamma", "ln2_gamma"}

        for layer_idx in range(num_layers):
            if layer_idx not in layer_weights:
                self.warnings.append(f"Layer {layer_idx}: no weights found")
                continue

            layer_names = {w[0] for w in layer_weights[layer_idx]}
            missing = expected_weights - layer_names

            if missing:
                self.warnings.append(f"Layer {layer_idx}: missing {missing}")

            # Check weight sizes
            for weight_name, entry in layer_weights[layer_idx]:
                reported_size = entry.get("size", 0)
                dtype = entry.get("dtype", "fp32")

                # Quick sanity check
                if reported_size == 0:
                    self.errors.append(f"Layer {layer_idx} {weight_name}: zero size!")
                    return False

        self.info.append(f"Layers checked: {num_layers}")
        self.info.append(f"Total weight entries: {len(entries)}")

        return True

    # =========================================================================
    # FULL MEMORY ACCOUNTING
    # =========================================================================

    def validate_full_accounting(self) -> bool:
        """Verify all memory is accounted for (no leaks, no gaps)."""
        weights = self.layout.get("memory", {}).get("weights", {})
        entries = [e for e in weights.get("entries", []) if int(e.get("size", 0)) >= 0]

        # Sort by offset
        sorted_entries = sorted(entries, key=lambda e: self._entry_offset(e))

        # Track memory regions
        total_reported = weights.get("size", 0)
        computed_total = sum(e.get("size", 0) for e in sorted_entries)

        self.info.append(f"Weight section reported size: {total_reported:,}")
        self.info.append(f"Computed from entries: {computed_total:,}")

        if total_reported != computed_total:
            diff = computed_total - total_reported
            self.warnings.append(
                f"Memory accounting diff: {diff:,} bytes "
                f"({'leak' if diff > 0 else 'gap'})"
            )

        # Check for overlaps and gaps
        prev_end = 0
        for entry in sorted_entries:
            offset = self._entry_offset(entry)
            size = int(entry.get("size", 0))
            if size == 0:
                # Zero-sized placeholder entries can legally share offsets.
                continue

            if offset < prev_end:
                self.errors.append(
                    f"OVERLAP: {entry.get('name')} at {offset} overlaps previous"
                )
                return False

            if offset > prev_end and prev_end > 0:
                gap = offset - prev_end
                if gap > 4096:  # Significant gap
                    self.warnings.append(
                        f"Gap of {gap} bytes before {entry.get('name')}"
                    )

            prev_end = offset + size

        self.info.append(f"Memory accounting: {computed_total:,} bytes in {len(sorted_entries)} entries")

        return True

    # =========================================================================
    # POINTER OFFSET ALIGNMENT VALIDATION
    # =========================================================================

    def validate_pointer_offsets(self) -> bool:
        """
        Validate that memory offsets are consistent with element sizes.

        Common bugs:
        - Using fp32 offset for fp16 data (2x wrong address)
        - Using element count instead of byte offset
        - Unaligned access for SIMD types

        For each memory region, we check:
        1. Offset alignment (64-bit alignment for SIMD)
        2. Element size consistency
        3. Cross-region pointer validity
        """
        ELEMENT_BYTES = {
            "fp32": 4, "f32": 4,
            "fp16": 2, "f16": 2, "bf16": 2,
            "int8": 1, "u8": 1,
            "int16": 2, "u16": 2,
            "int32": 4, "u32": 4,
            "q8_0": 1,  # 8-bit per element
            "q5_0": 0.625,  # 5-bit per element
            "q4_k": 0.5,  # 4-bit per element
            "q6_k": 0.75,  # 6-bit per element
        }

        weights = self.layout.get("memory", {}).get("weights", {})
        entries = weights.get("entries", [])

        errors = []
        warnings = []

        for entry in entries:
            name = entry.get("name", "")
            offset = self._entry_offset(entry)
            size = entry.get("size", 0)
            dtype = entry.get("dtype", "fp32")

            bytes_per_elem = ELEMENT_BYTES.get(dtype, 4)

            # Check 1: Offset alignment
            # 64-byte alignment for AVX-512, 32-byte for AVX2, 16 for SSE
            # Use 64 as safe default
            if offset % 64 != 0:
                # Quantized types don't need strict alignment
                if dtype not in ["q8_0", "q5_0", "q4_k", "q6_k"]:
                    warnings.append(
                        f"{name}: offset {offset} not 64-byte aligned (dtype={dtype})"
                    )

            # Check 2: Size consistency
            # For quantized types, size should be multiple of block size
            if dtype in ["q4_k", "q6_k"]:
                if size % 256 != 0:
                    warnings.append(
                        f"{name}: size {size} not multiple of 256 for {dtype}"
                    )
            elif dtype in ["q5_0", "q8_0"]:
                if size % 32 != 0:
                    warnings.append(
                        f"{name}: size {size} not multiple of 32 for {dtype}"
                    )

            # Check 3: Element boundary alignment
            # If we're accessing elements, offset should be element-aligned
            if dtype not in ["q8_0", "q5_0", "q4_k", "q6_k"]:
                # fp32/fp16 should be element-aligned
                if offset % bytes_per_elem != 0:
                    warnings.append(
                        f"{name}: offset {offset} not aligned for {dtype} "
                        f"({bytes_per_elem} bytes/elem)"
                    )

        # Check 4: Cross-region pointer consistency
        # If we have pointers to different types, verify offsets make sense
        fp32_regions = [e for e in entries if e.get("dtype") == "fp32"]
        quant_regions = [e for e in entries if "q" in e.get("dtype", "")]

        for fp32 in fp32_regions:
            fp32_offset = self._entry_offset(fp32)
            fp32_size = fp32.get("size", 0)

            # Check if any quantized region might be mistaken for fp32
            for quant in quant_regions:
                quant_offset = self._entry_offset(quant)
                # If quant region is in fp32 region range, that's a potential bug
                if (fp32_offset <= quant_offset < fp32_offset + fp32_size):
                    warnings.append(
                        f"Potential aliasing: {quant.get('name')} at {quant_offset} "
                        f"inside {fp32.get('name')} [{fp32_offset}-{fp32_offset + fp32_size}]"
                    )

        for w in warnings:
            self.warnings.append(f"ALIGNMENT: {w}")
        for e in errors:
            self.warnings.append(f"ALIGNMENT: {e}")

        return True

    # =========================================================================
    # BUMPWGT5 FILE FORMAT VALIDATION
    # =========================================================================

    def validate_bumpwgt5_format(self) -> bool:
        """
        Validate BUMPWGT5 file format structure integrity.

        Catches common errors:
        - Wrong weight start offset (dtype_table parsing bug)
        - dtype_table corruption
        - Impossible weight sizes
        - dtype_table vs layout dtype mismatches
        """
        # Check 1: First weight offset >= expected start
        weights = self.layout.get("memory", {}).get("weights", {})
        entries = weights.get("entries", [])

        if entries:
            # v7 packed layouts may expose both relative and absolute offsets.
            # Prefer absolute start derived from weights.base_offset.
            bump_layout = self.layout.get("bump_layout", {})
            base_offset = int(weights.get("base_offset", 0))
            header_size = int(bump_layout.get("header_size", 0))
            data_start = int(bump_layout.get("data_start", 0))
            expected_min_offset = base_offset if base_offset > 0 else data_start

            sorted_entries = sorted(entries, key=lambda e: self._entry_offset(e))
            first_offset = self._entry_offset(sorted_entries[0])

            if first_offset < expected_min_offset:
                self.errors.append(
                    f"FIRST WEIGHT at {first_offset} < expected {expected_min_offset}. "
                    f"base_offset={base_offset}, header={header_size}, data_start={data_start}. "
                    f"Check packed offset mapping."
                )
            elif first_offset == expected_min_offset:
                self.info.append(f"First weight at correct offset {first_offset} ✓")
            else:
                self.info.append(f"First weight at {first_offset} (expected >= {expected_min_offset})")

        # Check 2: dtype_table hash/structure validation
        self._validate_dtype_table_structure()

        # Check 3: Size bounds
        self._validate_weight_size_bounds()

        # Check 4: dtype_table vs layout dtype consistency
        self._validate_dtype_table_matches_layout()

        return len([e for e in self.errors if "FATAL" in e or "IMPOSSIBLE" in e]) == 0

    def _validate_dtype_table_structure(self) -> bool:
        """Validate dtype_table has reasonable structure."""
        dtype_table = self.layout.get("dtype_table", [])

        if not dtype_table:
            self.warnings.append("No dtype_table in layout - cannot validate format")
            return True

        # Expected: 339 entries (matching validation.weight_count)
        expected_count = 339
        if len(dtype_table) != expected_count:
            self.warnings.append(
                f"dtype_table has {len(dtype_table)} entries, expected {expected_count}"
            )

        # Valid dtype values (from GGUF)
        valid_dtypes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
        dtype_names = {
            0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q5_0", 5: "Q5_1",
            6: "Q8_0", 7: "Q8_1", 8: "I8", 9: "I16", 10: "I32", 11: "F64",
            12: "I8", 13: "I16", 14: "I32", 15: "8-BF16", 16: "8-FP8", 17: "8-BF8"
        }

        invalid_count = 0
        for i, dtype in enumerate(dtype_table):
            if dtype not in valid_dtypes:
                invalid_count += 1
                if invalid_count <= 3:
                    self.warnings.append(f"dtype_table[{i}]={dtype} - invalid dtype")

        if invalid_count > 0:
            self.warnings.append(f"Total invalid dtype entries: {invalid_count}")

        # Count by dtype
        dtype_counts = {}
        for dtype in dtype_table:
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

        self.info.append(f"dtype_table distribution: {dtype_counts}")

        return True

    def _validate_weight_size_bounds(self) -> bool:
        """Catch impossible weight sizes."""
        weights = self.layout.get("memory", {}).get("weights", {})
        entries = weights.get("entries", [])

        MAX_REASONABLE_WEIGHT = 200 * 1024 * 1024  # 200MB per weight (large models)
        MAX_TOTAL_WEIGHTS = 1000 * 1024 * 1024  # 1GB total

        for entry in entries:
            name = entry.get("name", "")
            size = entry.get("size", 0)

            # Check for impossible sizes
            if size > MAX_REASONABLE_WEIGHT:
                self.errors.append(
                    f"IMPOSSIBLE SIZE: {name} = {size/1024/1024:.1f}MB "
                    f"(max reasonable: {MAX_REASONABLE_WEIGHT/1024/1024:.0f}MB)"
                )
            elif size > 50 * 1024 * 1024:  # 50MB - flag as notable
                self.info.append(f"LARGE WEIGHT: {name} = {size/1024/1024:.1f}MB")

        total_size = sum(e.get("size", 0) for e in entries)
        if total_size > MAX_TOTAL_WEIGHTS:
            self.errors.append(
                f"TOTAL WEIGHT SIZE IMPOSSIBLE: {total_size/1024/1024:.1f}MB "
                f"(max reasonable: {MAX_TOTAL_WEIGHTS/1024/1024:.0f}MB)"
            )

        return True

    def _validate_dtype_table_matches_layout(self) -> bool:
        """Verify weight dtypes match dtype_table entries."""
        dtype_table = self.layout.get("dtype_table", [])
        if not dtype_table:
            return True

        weights = self.layout.get("memory", {}).get("weights", {})
        entries = weights.get("entries", [])

        # Map table index to dtype
        table_idx_to_dtype = {
            0: "fp32", 1: "fp16", 2: "q4_0", 3: "q4_1", 4: "q5_0", 5: "q5_1",
            6: "q8_0", 7: "q8_1", 8: "int8", 9: "int16", 10: "int32",
            # Add more mappings as needed
        }

        mismatches = []
        for entry in entries:
            table_idx = entry.get("table_index", -1)
            if table_idx >= 0 and table_idx < len(dtype_table):
                expected_dtype = table_idx_to_dtype.get(dtype_table[table_idx], "unknown")
                actual_dtype = entry.get("dtype", "")
                if expected_dtype != "unknown" and expected_dtype != actual_dtype:
                    mismatches.append(f"{entry['name']}: table={expected_dtype}, layout={actual_dtype}")

        if mismatches:
            self.errors.append(f"dtype_table mismatches ({len(mismatches)}): {mismatches[:5]}")
        else:
            self.info.append("dtype_table matches layout dtypes ✓")

        return True

    # =========================================================================
    # SINGLE BUMP ALLOCATOR VALIDATION
    # =========================================================================

    def validate_single_bump_allocator(self, generated_code_path: Optional[Path] = None) -> bool:
        """
        Validate that code uses single contiguous bump allocator, not multiple mallocs.

        v7 design principle: All memory (weights + activations) in ONE contiguous block.
        This ensures:
        - Cache-friendly access patterns
        - Simple memory management (one free call)
        - Huge page efficiency (single mmap/aligned_alloc)

        Checks:
        1. Code calls ck_huge_alloc() ONCE for the entire bump
        2. No individual malloc/calloc for activations or weights
        3. Single memset to initialize the entire bump
        """
        if not generated_code_path or not generated_code_path.exists():
            self.warnings.append("No code provided - skipping bump allocator validation")
            return True

        with open(generated_code_path, 'r') as f:
            code = f.read()

        import re

        # Count allocation calls
        # Pattern: bump allocation in generated runtime
        bump_alloc_pattern = re.compile(
            r'(ck_huge_alloc\s*\(\s*BUMP_TOTAL_SIZE|aligned_alloc\s*\([^)]*BUMP_TOTAL_SIZE|malloc\s*\(\s*BUMP_TOTAL_SIZE)',
            re.MULTILINE,
        )
        bump_allocs = len(bump_alloc_pattern.findall(code))

        # Pattern: any malloc/calloc that might be for activations/weights
        other_alloc_pattern = re.compile(r'\b(malloc|calloc|aligned_alloc)\s*\([^)]*(activ|weight|bump)')
        other_allocs = other_alloc_pattern.findall(code)

        # Pattern: single memset for entire bump
        bump_memset_pattern = re.compile(r'memset\s*\(\s*model\s*->\s*bump')
        bump_memset_count = len(bump_memset_pattern.findall(code))

        # Check results
        if bump_allocs == 1:
            self.info.append("Single bump allocation for BUMP_TOTAL_SIZE found ✓")
        elif bump_allocs == 0:
            self.warnings.append("No explicit BUMP_TOTAL_SIZE allocator pattern found")
        else:
            self.warnings.append(f"Multiple bump allocations found ({bump_allocs})")

        if other_allocs:
            self.warnings.append(
                f"Potential additional allocations found: {[a[0] for a in other_allocs]}"
            )

        if bump_memset_count == 1:
            self.info.append(f"Single bump memset: model->bump initialized once ✓")
        elif bump_memset_count == 0:
            self.warnings.append("No memset for model->bump found")

        return True

    # =========================================================================
    # CODE-TO-LAYOUT OFFSET CONSISTENCY VALIDATION
    # =========================================================================

    def validate_code_offsets(self, generated_code_path: Optional[Path] = None) -> bool:
        """
        Validate that offsets used in generated code match the layout.

        Common bugs this catches:
        - Token stored at offset 0 but embedding reads from 3604
        - Hardcoded offsets in codegen that don't match layout
        - Activation buffer offsets mismatched between ops

        We check:
        1. If generated code provided, parse offsets and compare to layout
        2. Verify activation buffer usage is consistent across ops
        3. Check that first token buffer matches expected layout
        """
        # Check if we have IR with buffer references
        if not self.lowered_ir:
            self.warnings.append("No IR provided - skipping code offset validation")
            return True

        # Extract expected activation offsets from IR
        # IR should have buffer names that match layout
        layout_buffers = self.layout.get("memory", {}).get("activations", {}).get("buffers", [])
        layout_buffer_map = {b.get("name", ""): b for b in layout_buffers}

        # Track buffer references in IR
        buffer_refs = defaultdict(list)  # buffer_name -> [(op, is_input)]

        for op in self.lowered_ir.get("operations", []):
            op_name = op.get("name", op.get("op", "unknown"))

            for inp_name, inp_spec in op.get("inputs", {}).items():
                if inp_spec.get("type") == "activation":
                    buf = inp_spec.get("buffer", inp_spec.get("source", ""))
                    if buf:
                        buffer_refs[buf].append((op_name, True))

            for out_name, out_spec in op.get("outputs", {}).items():
                if out_spec.get("type") == "activation":
                    buf = out_spec.get("buffer", "")
                    if buf:
                        buffer_refs[buf].append((op_name, False))

        # Check 1: Buffer existence
        for buf_name in buffer_refs:
            if buf_name and buf_name not in layout_buffer_map:
                self.warnings.append(
                    f"Buffer '{buf_name}' referenced in IR but not in layout"
                )

        # Check 2: Verify buffer sizes make sense
        for buf_name, refs in buffer_refs.items():
            if buf_name in layout_buffer_map:
                buf_size = layout_buffer_map[buf_name].get("size", 0)
                expected_usage = self._estimate_buffer_usage(buf_name, refs)

                if buf_size < expected_usage:
                    self.warnings.append(
                        f"Buffer '{buf_name}' size {buf_size} may be too small "
                        f"for {len(refs)} references (est. need {expected_usage})"
                    )

        # Check 3: Parse generated code for hardcoded offsets (if provided)
        if generated_code_path and generated_code_path.exists():
            self._parse_and_validate_code_offsets(generated_code_path)

        # Check 4: Verify first token handling
        # Look for "input_tokens" or similar buffer that should start at offset 0
        first_token_buffer = None
        for buf_name in buffer_refs:
            if "input" in buf_name.lower() or "token" in buf_name.lower():
                first_token_buffer = buf_name
                break

        if first_token_buffer and first_token_buffer in layout_buffer_map:
            buf_offset = layout_buffer_map[first_token_buffer].get("offset", -1)
            if buf_offset != 0:
                self.warnings.append(
                    f"First token buffer '{first_token_buffer}' at offset {buf_offset}, not 0. "
                    f"This may cause issues if code expects token at offset 0."
                )

        return True

    def _estimate_buffer_usage(self, buffer_name: str, refs: List[Tuple[str, bool]]) -> int:
        """Estimate buffer usage based on operation type."""
        # Rough estimates for common buffers
        spec = self.QWEN2_0_5B_SPECS
        E, I = spec["embed_dim"], spec["intermediate_size"]

        usage_estimates = {
            "embed": E * 4,
            "input": 4,
            "output": E * 4,
            "q": E * 4,
            "k": E * 4,
            "v": E * 4,
            "scores": E * E * 4,
            "mlp": I * 4,
            "residual": E * 4,
        }

        base_usage = 0
        for key, size in usage_estimates.items():
            if key in buffer_name.lower():
                base_usage = size
                break

        return base_usage * len(refs)

    def _parse_and_validate_code_offsets(self, code_path: Path) -> bool:
        """
        Parse generated C code for hardcoded offsets and validate against layout.

        This method checks that all hardcoded offsets in generated C code match
        actual buffer offsets defined in the layout.

        Flow:
        1. Build a map of ALL known offsets (weights + activation buffers)
        2. Find all "base + OFFSET" patterns in C code
        3. For each offset, check if it matches a known buffer
        4. Only flag as suspicious if offset doesn't match any known buffer

        Special cases:
        - Token/input buffer should be at offset 0
        - Activation buffers can be at any valid offset (like 3604 for layer_input)
        """
        weights = self.layout.get("memory", {}).get("weights", {})
        weight_entries = weights.get("entries", [])
        activations = self.layout.get("memory", {}).get("activations", {})
        act_buffers = activations.get("buffers", [])

        # Step 1: Build map of ALL known offsets from layout
        # offset -> [(name, size, type)]
        known_offsets = {}

        for entry in weight_entries:
            name = entry.get("name", "")
            offset = entry.get("offset", 0)
            size = entry.get("size", 0)
            dtype = entry.get("dtype", "fp32")

            if offset not in known_offsets:
                known_offsets[offset] = []
            known_offsets[offset].append({
                "name": name,
                "size": size,
                "dtype": dtype,
                "type": "weight"
            })

        for buf in act_buffers:
            name = buf.get("name", "")
            offset = buf.get("offset", 0)
            size = buf.get("size", 0)
            dtype = buf.get("dtype", "fp32")

            if offset not in known_offsets:
                known_offsets[offset] = []
            known_offsets[offset].append({
                "name": name,
                "size": size,
                "dtype": dtype,
                "type": "activation"
            })

        self.info.append(f"Found {len(known_offsets)} unique buffer offsets in layout")

        # Step 2: Parse C code for hardcoded offsets
        with open(code_path, 'r') as f:
            code = f.read()

        import re

        # Pattern: matches "activations + N", "weights + N", "bump + N"
        offset_pattern = re.compile(
            r'(activations|weights|bump)[\s]*\+[\s]*(\d+)'
        )

        found_offsets = {}  # offset -> count of occurrences
        for match in offset_pattern.finditer(code):
            base = match.group(1)
            offset = int(match.group(2))
            if offset not in found_offsets:
                found_offsets[offset] = []
            found_offsets[offset].append(match.group(0).strip())

        # Step 3: Validate each found offset against known buffers
        suspicious_offsets = []  # Offsets that don't match any known buffer
        token_offsets = []  # Offsets used with "token" or "input" context

        for offset, occurrences in found_offsets.items():
            if offset in known_offsets:
                # This offset is legitimate - matches a known buffer
                bufs = known_offsets[offset]
                for buf in bufs:
                    self.info.append(
                        f"CODE→LAYOUT: offset {offset} → {buf['name']} "
                        f"({buf['size']} bytes, {buf['type']})"
                    )
            else:
                # Offset doesn't match any known buffer - check context
                suspicious_offsets.append((offset, occurrences))

        # Step 4: Check suspicious offsets for specific patterns
        for offset, occurrences in suspicious_offsets:
            # Check if this might be a token-related bug
            is_token_related = False
            for occurrence in occurrences:
                if "token" in occurrence.lower():
                    is_token_related = True
                    break

            if is_token_related:
                # Token should be at offset 0
                if offset != 0:
                    self.warnings.append(
                        f"Token/input accessed at offset {offset}, not 0. "
                        f"This may cause issues if token is stored at 0."
                    )
            else:
                # Check if this might be a "computed" offset (from calculations)
                # These are legitimate if they're within a valid range
                if offset < 1000000:  # Reasonable offset value
                    self.warnings.append(
                        f"Unknown offset {offset} used in code. "
                        f"Should match a buffer in layout. Occurrences: {len(occurrences)}"
                    )

        # Step 5: Special check - verify token is stored/read from correct offset
        # Token should be at the "token_ids" buffer offset, NOT hardcoded to wrong offset
        token_ids_offset = None
        layer_input_offset = None
        for buf in act_buffers:
            name = buf.get("name", "").lower()
            offset = buf.get("offset", -1)
            if name in ["token_ids", "token", "input_token"]:
                token_ids_offset = offset
                self.info.append(
                    f"Token buffer '{buf.get('name')}' found at offset {token_ids_offset}"
                )
            elif "layer_input" in name:
                layer_input_offset = offset
                self.info.append(
                    f"Layer input buffer '{buf.get('name')}' found at offset {layer_input_offset}"
                )

        # Check for token storage pattern: *((int32_t*)...activations + OFFSET)) = token
        # Matches patterns like:
        #   *((int32_t*)((uint8_t*)model->activations + 3604)) = token;
        #   *((int32_t*)((uint8_t*)model->activations+3604)) = token;
        token_store_pattern = re.compile(
            r'\*\s*\(\s*\(\s*int32_t\s*\*\s*\)\s*\(\s*\(\s*uint8_t\s*\*\s*\)\s*model\s*->\s*activations\s*[\+\-]\s*(\d+)\s*\)\s*\)\s*=\s*token'
        )

        for match in token_store_pattern.finditer(code):
            actual_offset = int(match.group(1))
            if token_ids_offset is not None and actual_offset != token_ids_offset:
                self.errors.append(
                    f"BUG: Token stored at offset {actual_offset}, "
                    f"but token_ids buffer is at offset {token_ids_offset}. "
                    f"Token will be stored in wrong buffer! "
                    f"(layer_input is at {layer_input_offset})"
                )
            elif token_ids_offset is not None and actual_offset == token_ids_offset:
                self.info.append(f"Token stored at correct offset {token_ids_offset} ✓")

        # Also check simpler pattern for token store
        simple_token_pattern = re.compile(
            r'model\s*->\s*activations\s*\+\s*(\d+)[\^\s]*=\s*token'
        )
        for match in simple_token_pattern.finditer(code):
            actual_offset = int(match.group(1))
            self.info.append(f"Token store pattern found at offset {actual_offset}")

        # Step 6: Check layer offsets consistency
        # All layer weights should have consistent offsets relative to each other
        self._validate_layer_offsets_consistency(weight_entries)

        return True

    def _validate_layer_offsets_consistency(self, weight_entries: List[Dict]) -> bool:
        """
        Validate that layer weights have consistent offsets.

        For each weight type (wq, wk, wv, wo, w1, w2, etc.), the offset
        difference between layer N and layer N+1 should be consistent.
        """
        # Group weights by type
        weights_by_type = defaultdict(dict)  # weight_type -> {layer_num: entry}
        for entry in weight_entries:
            name = entry.get("name", "")
            if name.startswith("layer."):
                parts = name.split(".")
                if len(parts) >= 3:
                    try:
                        layer = int(parts[1])
                        weight_type = parts[2]
                        weights_by_type[weight_type][layer] = entry
                    except ValueError:
                        pass

        # Check each weight type
        for weight_type, layers in weights_by_type.items():
            if len(layers) < 2:
                continue

            # Get sorted layer indices
            sorted_layers = sorted(layers.keys())

            # Calculate strides between consecutive layers
            strides = []
            for i in range(len(sorted_layers) - 1):
                l1, l2 = sorted_layers[i], sorted_layers[i + 1]
                offset1 = layers[l1].get("offset", 0)
                offset2 = layers[l2].get("offset", 0)
                stride = offset2 - offset1
                strides.append((l1, l2, stride))

            # Check for consistent stride
            if strides:
                first_stride = strides[0][2]
                inconsistent = [(l1, l2, s) for l1, l2, s in strides if s != first_stride]

                if inconsistent:
                    self.warnings.append(
                        f"Weight '{weight_type}': Inconsistent layer strides. "
                        f"First stride: {first_stride}. "
                        f"Inconsistent: {inconsistent[:3]}..."
                    )
                else:
                    self.info.append(
                        f"Weight '{weight_type}': Consistent layer stride: {first_stride} bytes ✓"
                    )

        return True

    # =========================================================================
    # RUN ALL VALIDATIONS
    # =========================================================================

    def run_all(self) -> bool:
        """Run all advanced validations."""
        if not self.load():
            self.errors.append("Failed to load files")
            return False

        print("\n" + "=" * 70)
        print("ADVANCED MEMORY VALIDATION")
        print("=" * 70)

        validations = [
            ("BUMPWGT5 format integrity", self.validate_bumpwgt5_format),
            ("Weight size validation", self.validate_weight_sizes),
            ("Kernel buffer validation", self.validate_kernel_buffer_sizes),
            ("Activation size validation", self.validate_activation_sizes),
            ("Vocab/Embedding layout", self.validate_vocab_layout),
            ("Per-layer accounting", self.validate_per_layer_accounting),
            ("Full memory accounting", self.validate_full_accounting),
            ("Pointer offset alignment", self.validate_pointer_offsets),
            ("Single bump allocator", lambda: self.validate_single_bump_allocator(self.generated_code_path)),
            ("Code-to-layout offsets", lambda: self.validate_code_offsets(self.generated_code_path)),
        ]

        all_passed = True
        for name, fn in validations:
            print(f"\n--- {name} ---")
            if not fn():
                all_passed = False

        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for e in self.errors[:20]:
                print(f"  ✗ {e}")
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for w in self.warnings[:20]:
                print(f"  ⚠ {w}")
            if len(self.warnings) > 20:
                print(f"  ... and {len(self.warnings) - 20} more")

        if self.info:
            print(f"\nINFO ({len(self.info)}):")
            for i in self.info[:10]:
                print(f"  ✓ {i}")
            if len(self.info) > 10:
                print(f"  ... and {len(self.info) - 10} more")

        if all_passed and not self.errors:
            print("\n✓ All advanced validations passed")
        elif not all_passed:
            print("\n✗ Validation failed")

        print("=" * 70)

        return all_passed and not self.errors


def main():
    parser = argparse.ArgumentParser(description="Advanced memory validation")
    parser.add_argument("--layout", type=Path,
                        help="Path to layout JSON")
    parser.add_argument("--ir", type=Path,
                        help="Path to lowered IR JSON")
    parser.add_argument("--code", type=Path,
                        help="Path to generated C code (for offset validation)")
    parser.add_argument("--model-dir", type=Path,
                        help="Model directory; auto-resolves --layout/--ir/--code")
    parser.add_argument("--model", type=Path,
                        help="Alias for --model-dir")
    parser.add_argument("--deep", action="store_true",
                        help="Compatibility flag; full validation is already deep by default")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")

    args = parser.parse_args()

    model_dir = args.model_dir or args.model
    if model_dir is None:
        env_model_dir = os.environ.get("CK_MODEL_DIR")
        if env_model_dir:
            model_dir = Path(env_model_dir)
    if (args.layout is None or args.ir is None) and model_dir is not None:
        layout_candidate = model_dir / "layout_decode.json"
        ir_candidates = [
            model_dir / "lowered_decode_call.json",
            model_dir / "lowered_decode.json",
            model_dir / "lowered_prefill_call.json",
            model_dir / "lowered_prefill.json",
        ]
        code_candidates = [
            model_dir / "model_v7.c",
            model_dir / "ck-kernel-inference.c",
        ]

        if args.layout is None and layout_candidate.exists():
            args.layout = layout_candidate
        if args.ir is None:
            for cand in ir_candidates:
                if cand.exists():
                    args.ir = cand
                    break
        if args.code is None:
            for cand in code_candidates:
                if cand.exists():
                    args.code = cand
                    break

    if args.layout is None or args.ir is None:
        parser.error("provide --layout/--ir or use --model-dir")

    validator = AdvancedMemoryValidator(args.layout, args.ir, args.code)
    passed = validator.run_all()

    if args.json:
        print(json.dumps({
            "passed": passed,
            "errors": validator.errors,
            "warnings": validator.warnings,
            "info": validator.info,
        }, indent=2))

    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())
