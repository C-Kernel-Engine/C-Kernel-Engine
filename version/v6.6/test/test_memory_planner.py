#!/usr/bin/env python3
"""
test_memory_planner.py - Memory Layout Quick Validator for v6.6

================================================================================
HOW THIS VALIDATION WORKS
================================================================================

This is a QUICK validation test for memory layouts. It checks the most
important invariants without deep IR or code analysis.

--------------------------------------------------------------------------------
STAGE 1: LOAD LAYOUT
--------------------------------------------------------------------------------

    layout_decode.json
    ┌────────────────────────────────────────┐
    │ memory:                                 │
    │   weights:                              │
    │     - name: "layer.0.wq"               │
    │     - offset: 148602200                 │
    │     - size: 551936                      │
    │     - dtype: "q5_0"                     │
    │   config:                               │
    │     - num_layers: 24                    │
    │     - embed_dim: 896                    │
    └────────────────────────────────────────┘

--------------------------------------------------------------------------------
STAGE 2: PARSE WEIGHTS
--------------------------------------------------------------------------------

    for entry in layout["memory"]["weights"]["entries"]:
        region = MemoryRegion(
            name=entry["name"],       # "layer.0.wq"
            offset=entry["offset"],   # 148602200
            size=entry["size"],       # 551936
            dtype=entry["dtype"],     # "q5_0"
            mem_type=MemType.WEIGHT,  # Classified from name
            layer=0                   # Extracted from name
        )

--------------------------------------------------------------------------------
STAGE 3: RUN QUICK TESTS
--------------------------------------------------------------------------------

    ┌─────────────────────────────────────────────────────────────────────┐
    │ TEST                          │ HOW IT WORKS                        │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 1. No overlaps                │ Sort by offset, check gaps          │
    │ 2. Contiguity                 │ Report gaps between regions         │
    │ 3. Quant sizes                │ elements × bytes/elem = size        │
    │ 4. Quant summary matches      │ layout quant_summary vs actual dtypes│
    │ 5. Layer stride               │ layer[N].offset - layer[N-1].offset │
    │ 6. Layer offsets              │ All 24 layers present               │
    │ 7. Activation buffers         │ IR buffers exist in layout          │
    │ 8. Canary markers             │ Check canary positions if present   │
    │ 9. Data flow                  │ No dangling tensors                 │
    └─────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
EXAMPLE: NO-OVERLAP CHECK
--------------------------------------------------------------------------------

    Sort regions by offset:
        [A: 0-100], [B: 100-200], [C: 200-350]

    Check each adjacent pair:
        A.end (100) <= B.offset (100)? ✓
        B.end (200) <= C.offset (200)? ✓

    If overlap detected:
        ERROR: A [0-100] overlaps B [80-180]

================================================================================

USAGE:
    # Quick validation (no IR needed)
    python test_memory_planner.py --layout=layout.json

    # With IR for activation validation
    python test_memory_planner.py --layout=layout.json --ir=ir.json

    # JSON output for CI/CD
    python test_memory_planner.py --layout=... --json
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum


class MemType(Enum):
    """Memory region type classification."""
    WEIGHT = "weight"       # Model weights
    ACTIVATION = "activation"  # Intermediate activations
    SCRATCH = "scratch"     # Temporary buffers
    KV_CACHE = "kv_cache"   # Key-value cache
    ROPE = "rope"           # RoPE tables
    LOGITS = "logits"       # Output logits
    VOCAB = "vocab"         # Vocabulary data


@dataclass
class MemoryRegion:
    """
    Represents a contiguous region of memory.

    Key methods:
        end()       - Returns offset + size (region end address)
        overlaps()  - Checks if this region overlaps another
    """
    name: str
    offset: int
    size: int
    dtype: str
    mem_type: MemType
    layer: Optional[int] = None
    owner: Optional[str] = None  # Which operation owns this

    def end(self) -> int:
        """Return the end offset of this region."""
        return self.offset + self.size

    def overlaps(self, other: 'MemoryRegion') -> bool:
        """
        Check if this region overlaps with another.

        Two regions overlap if:
            A.offset < B.end AND A.end > B.offset

        Example (overlap):
            A: [0, 100)
            B: [80, 180)
            80 < 100 AND 180 > 0 → OVERLAP!

        Example (no overlap):
            A: [0, 100)
            B: [100, 200)
            100 < 100? NO → No overlap
        """
        return (self.offset < other.end() and self.end() > other.offset)


@dataclass
class ValidationResult:
    """Result of memory validation."""
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    regions: List[MemoryRegion] = field(default_factory=list)


class MemoryPlannerTest:
    """Memory planner test case runner."""

    # Quantization byte sizes (per element, not per block)
    QUANT_BYTES = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "q8_0": 1,      # 8-bit per weight
        "q5_0": 0.625,  # 5-bit per weight (5/8)
        "q4_k": 0.5,    # 4-bit K-quant
        "q6_k": 0.75,   # 6-bit K-quant
        "q8_k": 1.0,
        "int32": 4,
        "int16": 2,
        "int8": 1,
    }

    # Standard activation sizes (per token, fp32)
    ACTIVATION_BYTES = {
        "embed": 4,           # fp32
        "logits": 4,          # fp32 per vocab
        "rmsnorm": 4,         # fp32 per dim
        "q": 4,               # fp32 per head * dim
        "k": 4,               # fp32 per head * dim
        "v": 4,               # fp32 per head * dim
        "scores": 4,          # fp32 attention scores
        "attn_out": 4,        # fp32 output
        "mlp_intermediate": 4, # fp32 intermediate
        "output": 4,          # fp32 final output
    }

    def __init__(self, layout_path: Path, ir_path: Optional[Path] = None):
        self.layout_path = layout_path
        self.ir_path = ir_path
        self.layout = None
        self.ir = None
        self.result = ValidationResult()

    def load(self) -> bool:
        """Load layout and IR files."""
        # Load layout
        with open(self.layout_path, 'r') as f:
            self.layout = json.load(f)

        # Load IR if provided
        if self.ir_path and self.ir_path.exists():
            with open(self.ir_path, 'r') as f:
                self.ir = json.load(f)

        return True

    def compute_weight_size(self, name: str, dtype: str, shape: List[int]) -> int:
        """Compute expected weight size in bytes."""
        elements = 1
        for dim in shape:
            elements *= dim

        bytes_per_elem = self.QUANT_BYTES.get(dtype, 4)
        return int(elements * bytes_per_elem)

    def compute_activation_size(self, shape: List[int], dtype: str = "fp32") -> int:
        """Compute expected activation size in bytes."""
        elements = 1
        for dim in shape:
            elements *= dim

        bytes_per_elem = self.QUANT_BYTES.get(dtype, 4)
        return elements * bytes_per_elem

    def parse_weight_entry(self, entry: Dict) -> MemoryRegion:
        """Parse a weight entry from layout."""
        name = entry["name"]
        dtype = entry.get("dtype", "fp32")
        offset = entry.get("offset", 0)
        size = entry.get("size", 0)

        # Determine layer
        layer = None
        if name.startswith("layer."):
            parts = name.split(".")
            if len(parts) >= 2:
                try:
                    layer = int(parts[1])
                except ValueError:
                    pass

        # Determine memory type
        if "ln" in name and "gamma" in name:
            mem_type = MemType.WEIGHT
        elif "w" in name or "b" in name:
            mem_type = MemType.WEIGHT
        elif "vocab" in name:
            mem_type = MemType.VOCAB
        elif "logits" in name:
            mem_type = MemType.LOGITS
        else:
            mem_type = MemType.WEIGHT

        return MemoryRegion(
            name=name,
            offset=offset,
            size=size,
            dtype=dtype,
            mem_type=mem_type,
            layer=layer
        )

    def parse_activation_entry(self, name: str, entry: Dict) -> MemoryRegion:
        """Parse an activation entry from IR or layout."""
        buf = entry.get("buffer", "")
        dtype = entry.get("dtype", "fp32")
        shape = entry.get("shape", [])

        # Compute expected size
        size = self.compute_activation_size(shape, dtype)

        # Determine layer
        layer = None
        if "layer." in name:
            parts = name.split(".")
            if len(parts) >= 2:
                try:
                    layer = int(parts[1])
                except ValueError:
                    pass

        # Determine memory type
        if "kv_cache" in name or "k_cache" in name or "v_cache" in name:
            mem_type = MemType.KV_CACHE
        elif "rope" in name:
            mem_type = MemType.ROPE
        elif "scratch" in name or "scores" in name:
            mem_type = MemType.SCRATCH
        else:
            mem_type = MemType.ACTIVATION

        return MemoryRegion(
            name=name,
            offset=-1,  # May not be in layout
            size=size,
            dtype=dtype,
            mem_type=mem_type,
            layer=layer
        )

    def extract_regions_from_layout(self) -> List[MemoryRegion]:
        """Extract all memory regions from layout file."""
        regions = []

        # Extract weights
        weights = self.layout.get("memory", {}).get("weights", {})
        for entry in weights.get("entries", []):
            region = self.parse_weight_entry(entry)
            regions.append(region)

        return regions

    def validate_quant_sizes(self) -> bool:
        """Validate that quantization sizes match actual data."""
        weights = self.layout.get("memory", {}).get("weights", {})
        quant_summary = self.layout.get("quant_summary", {})

        for entry in weights.get("entries", []):
            name = entry["name"]
            dtype = entry.get("dtype", "fp32")
            reported_size = entry.get("size", 0)

            # Extract expected shape from name and quant type
            # This is simplified - real impl would check manifest
            if "wq" in name or "wk" in name or "wv" in name or "wo" in name:
                # Projection weight
                expected_size = self._get_proj_weight_size(name, dtype)
            elif "w1" in name or "w2" in name or "w3" in name:
                # MLP weight
                expected_size = self._get_mlp_weight_size(name, dtype)
            elif "ln" in name and "gamma" in name:
                # Layer norm weight
                expected_size = 896 * self.QUANT_BYTES.get(dtype, 4)
            elif "ln" in name and "beta" in name:
                expected_size = 896 * self.QUANT_BYTES.get(dtype, 4)
            else:
                expected_size = reported_size  # Skip unknown

            if expected_size > 0 and abs(expected_size - reported_size) > reported_size * 0.01:
                self.result.warnings.append(
                    f"Size mismatch for {name}: reported={reported_size}, expected~={expected_size}"
                )

        return True

    def _get_proj_weight_size(self, name: str, dtype: str) -> int:
        """Get expected projection weight size."""
        # Pattern: layer.N.wq, layer.N.wk, etc.
        parts = name.split(".")
        if len(parts) < 3:
            return 0

        layer_num = parts[1]
        weight_type = parts[2]

        # Qwen2 0.5B dimensions
        embed_dim = 896
        num_heads = 14
        num_kv_heads = 2
        head_dim = 64

        bytes_per_elem = self.QUANT_BYTES.get(dtype, 4)

        if weight_type == "wq":
            # Q: (num_heads * head_dim, embed_dim) = (14*64, 896) = (896, 896)
            return int(embed_dim * embed_dim * bytes_per_elem)
        elif weight_type == "wk":
            # K: (num_kv_heads * head_dim, embed_dim) = (2*64, 896) = (128, 896)
            return int(num_kv_heads * head_dim * embed_dim * bytes_per_elem)
        elif weight_type == "wv":
            # V: (num_kv_heads * head_dim, embed_dim)
            return int(num_kv_heads * head_dim * embed_dim * bytes_per_elem)
        elif weight_type == "wo":
            # O: (embed_dim, num_heads * head_dim) = (896, 896)
            return int(embed_dim * embed_dim * bytes_per_elem)
        elif weight_type == "b1":
            # MLP gate/up proj bias: (intermediate_size,)
            return 4864 * 4  # fp32
        elif weight_type == "b2":
            # MLP down proj bias: (embed_dim,)
            return 896 * 4  # fp32
        elif weight_type == "bq":
            # Q proj bias
            return 896 * 4  # fp32
        elif weight_type == "bk":
            # K proj bias
            return 128 * 4  # fp32
        elif weight_type == "bv":
            # V proj bias
            return 128 * 4  # fp32
        elif weight_type == "bo":
            # O proj bias
            return 896 * 4  # fp32

        return 0

    def _get_mlp_weight_size(self, name: str, dtype: str) -> int:
        """Get expected MLP weight size."""
        parts = name.split(".")
        if len(parts) < 3:
            return 0

        weight_type = parts[2]

        embed_dim = 896
        intermediate = 4864
        bytes_per_elem = self.QUANT_BYTES.get(dtype, 4)

        if weight_type == "w1":
            # Gate proj: (intermediate, embed_dim) = (4864, 896)
            return int(intermediate * embed_dim * bytes_per_elem)
        elif weight_type == "w2":
            # Down proj: (embed_dim, intermediate) = (896, 4864)
            return int(embed_dim * intermediate * bytes_per_elem)
        elif weight_type == "w3":
            # Up proj: (intermediate, embed_dim)
            return int(intermediate * embed_dim * bytes_per_elem)

        return 0

    def validate_no_overlaps(self, regions: List[MemoryRegion]) -> bool:
        """Check that no memory regions overlap."""
        # Sort by offset
        sorted_regions = sorted(regions, key=lambda r: r.offset)

        for i, r1 in enumerate(sorted_regions):
            for r2 in sorted_regions[i+1:]:
                if r1.overlaps(r2):
                    self.result.errors.append(
                        f"OVERLAP: {r1.name} [{r1.offset:#x}-{r1.end():#x}] overlaps "
                        f"{r2.name} [{r2.offset:#x}-{r2.end():#x}]"
                    )
                    self.result.passed = False

        return self.result.passed

    def validate_contiguity(self, regions: List[MemoryRegion]) -> bool:
        """Check that regions are properly packed (no gaps where possible)."""
        # Sort by offset
        sorted_regions = sorted(regions, key=lambda r: r.offset)

        total_gap = 0
        for i in range(len(sorted_regions) - 1):
            r1 = sorted_regions[i]
            r2 = sorted_regions[i + 1]

            gap = r2.offset - r1.end()
            if gap > 0:
                # Gap might be intentional (alignment, different sections)
                if gap < 4096:  # Small gap - might be alignment
                    self.result.warnings.append(
                        f"Small gap ({gap} bytes) between {r1.name} and {r2.name}"
                    )
                total_gap += gap

        return True

    def validate_layer_offsets(self) -> bool:
        """Validate that layer offsets follow expected pattern."""
        config = self.layout.get("config", {})
        num_layers = config.get("num_layers", 24)
        embed_dim = config.get("embed_dim", 896)

        # Extract layer offsets from layout
        layers = self.layout.get("memory", {}).get("layers", {})

        for layer_idx in range(num_layers):
            layer_key = f"layer.{layer_idx}"
            if layer_key not in layers:
                self.result.warnings.append(f"Layer {layer_idx} not found in layout")

        return True

    def validate_activations_match_ops(self) -> bool:
        """Validate that activation buffers match IR operations."""
        if not self.ir:
            self.result.warnings.append("No IR provided, skipping activation validation")
            return True

        # Extract expected activations from IR operations
        op_activations = {}  # op_name -> list of activation names

        for op in self.ir.get("operations", []):
            op_name = op.get("name", op.get("op", "unknown"))

            # Collect inputs
            for inp in op.get("inputs", {}).values():
                if inp.get("type") == "activation":
                    act_name = inp.get("buffer", inp.get("source", ""))
                    if act_name:
                        if act_name not in op_activations:
                            op_activations[act_name] = []
                        op_activations[act_name].append(op_name)

            # Collect outputs
            for out in op.get("outputs", {}).values():
                if out.get("type") == "activation":
                    act_name = out.get("buffer", "")
                    if act_name:
                        if act_name not in op_activations:
                            op_activations[act_name] = []
                        op_activations[act_name].append(op_name)

        # Check that activations referenced in IR exist in layout
        layout_buffers = self.layout.get("memory", {}).get("activations", {}).get("buffers", [])
        layout_buffer_names = {b.get("name", "") for b in layout_buffers}

        for act_name in op_activations:
            if act_name and act_name not in layout_buffer_names:
                # Might be a scratch buffer not in layout
                if "scratch" not in act_name and "scores" not in act_name:
                    self.result.warnings.append(
                        f"Activation '{act_name}' referenced in IR but not in layout"
                    )

        return True

    def validate_quant_summary_consistency(self) -> bool:
        """Validate that quant_summary matches actual weight types."""
        quant_summary = self.layout.get("quant_summary", {})
        weights = self.layout.get("memory", {}).get("weights", {})
        weight_dict = {e.get("name", ""): e for e in weights.get("entries", [])}

        for layer_key, layer_quants in quant_summary.items():
            if not layer_key.startswith("layer."):
                continue

            for weight_name, expected_dtype in layer_quants.items():
                full_name = f"{layer_key}.{weight_name}"
                if full_name in weight_dict:
                    actual_dtype = weight_dict[full_name].get("dtype", "")
                    if actual_dtype != expected_dtype:
                        self.result.errors.append(
                            f"Dtype mismatch: {full_name} is {actual_dtype} but "
                            f"quant_summary says {expected_dtype}"
                        )
                        self.result.passed = False

        return self.result.passed

    def compute_memory_summary(self) -> Dict:
        """Compute total memory usage breakdown."""
        regions = self.extract_regions_from_layout()

        summary = {
            "total_weight_bytes": 0,
            "total_activation_bytes": 0,
            "total_kv_cache_bytes": 0,
            "total_rope_bytes": 0,
            "by_dtype": {},
            "by_layer": {},
        }

        for region in regions:
            if region.size <= 0:
                continue

            summary["total_weight_bytes"] += region.size

            # By dtype
            if region.dtype not in summary["by_dtype"]:
                summary["by_dtype"][region.dtype] = 0
            summary["by_dtype"][region.dtype] += region.size

            # By layer
            if region.layer is not None:
                if region.layer not in summary["by_layer"]:
                    summary["by_layer"][region.layer] = 0
                summary["by_layer"][region.layer] += region.size

        return summary

    # =========================================================================
    # TEST 4: LAYER STRIDE VALIDATION
    # =========================================================================

    def get_layer_stride(self) -> Optional[int]:
        """Compute expected layer stride from consecutive layers."""
        layer0_offsets = {}
        layer1_offsets = {}

        for region in self.result.regions:
            if region.layer == 0:
                weight_name = region.name.split(".")[-1]
                layer0_offsets[weight_name] = region.offset
            elif region.layer == 1:
                weight_name = region.name.split(".")[-1]
                layer1_offsets[weight_name] = region.offset

        for weight in layer0_offsets:
            if weight in layer1_offsets:
                return layer1_offsets[weight] - layer0_offsets[weight]

        return None

    def validate_layer_stride(self) -> bool:
        """Validate layer offsets - variable stride is OK in v6.6."""
        base_stride = self.get_layer_stride()
        if base_stride is None:
            self.result.warnings.append("Cannot compute layer stride")
            return True

        self.result.warnings.append(f"Base layer stride: {base_stride:,} bytes ({base_stride/1024:.1f} KB)")

        regions_by_layer = {}
        for region in self.result.regions:
            if region.layer is not None:
                if region.layer not in regions_by_layer:
                    regions_by_layer[region.layer] = {}
                weight_name = region.name.split(".")[-1]
                regions_by_layer[region.layer][weight_name] = region.offset

        config = self.layout.get("config", {})
        num_layers = config.get("num_layers", 24)

        # Track unique strides
        unique_strides = set()

        for layer_idx in range(1, num_layers):
            if layer_idx not in regions_by_layer or layer_idx - 1 not in regions_by_layer:
                continue
            for weight in regions_by_layer[layer_idx]:
                if weight in regions_by_layer[layer_idx - 1]:
                    actual_stride = (regions_by_layer[layer_idx][weight] -
                                   regions_by_layer[layer_idx - 1][weight])
                    unique_strides.add(actual_stride)

        # v6.6 uses variable stride due to mixed quantization
        if len(unique_strides) > 1:
            self.result.warnings.append(
                f"Variable layer stride detected ({len(unique_strides)} unique values): "
                f"{sorted(unique_strides)[:5]}... This is expected in v6.6 with mixed quantization"
            )

        return True

    # =========================================================================
    # TEST 5: ACTIVATION BUFFER VALIDATION
    # =========================================================================

    def validate_activation_sizes(self) -> bool:
        """Validate that activation buffers match IR operation requirements."""
        if not self.ir:
            self.result.warnings.append("No IR - skipping activation validation")
            return True

        config = self.layout.get("config", {})
        embed_dim = config.get("embed_dim", 896)

        # Track referenced and defined buffers
        referenced = set()
        defined = set()

        layout_buffers = self.layout.get("memory", {}).get("activations", {}).get("buffers", [])
        for b in layout_buffers:
            if name := b.get("name"):
                defined.add(name)

        for op in self.ir.get("operations", []):
            for inp in op.get("inputs", {}).values():
                if inp.get("type") == "activation":
                    if buf := inp.get("buffer") or inp.get("source"):
                        referenced.add(buf)
            for out in op.get("outputs", {}).values():
                if buf := out.get("buffer"):
                    referenced.add(buf)

        undefined = referenced - defined
        for buf in undefined:
            if buf and "scratch" not in buf and "scores" not in buf:
                self.result.warnings.append(f"Undeclared activation: '{buf}'")

        return True

    # =========================================================================
    # TEST 6: CANARY MARKER GENERATION
    # =========================================================================

    def generate_canary_positions(self) -> List[Dict]:
        """Generate expected canary positions for all memory boundaries."""
        regions = sorted(self.result.regions, key=lambda r: r.offset)
        canaries = []

        canaries.append({"name": "header_start", "offset": 0})

        for i, region in enumerate(regions):
            canaries.append({"name": f"{region.name}_start", "offset": region.offset})
            end_offset = regions[i + 1].offset if i < len(regions) - 1 else region.end()
            canaries.append({"name": f"{region.name}_end", "offset": region.end()})

        if regions:
            canaries.append({"name": "footer_end", "offset": regions[-1].end()})

        return canaries

    def validate_canaries(self) -> bool:
        """Validate canary positions."""
        canaries = self.layout.get("canaries", [])
        if not canaries:
            generated = self.generate_canary_positions()
            self.result.warnings.append(f"No canaries in layout. Expected {len(generated)}:")
            for c in generated[:5]:
                self.result.warnings.append(f"  {c['name']}: {c['offset']}")
            return True

        # Check key boundaries
        regions = self.result.regions
        if regions:
            first, last = regions[0].offset, regions[-1].end()
            canary_offsets = {c.get("offset") for c in canaries}
            if first not in canary_offsets:
                self.result.warnings.append(f"No canary at header_start ({first})")
            if last not in canary_offsets:
                self.result.warnings.append(f"No canary at footer_end ({last})")

        return True

    # =========================================================================
    # TEST 7: DATA FLOW VALIDATION
    # =========================================================================

    def validate_data_flow(self) -> bool:
        """Validate tensor dependencies (no dangling tensors)."""
        if not self.ir:
            return True

        sources = {}  # tensor -> producing op
        consumers = {}  # tensor -> [consuming ops]

        for op in self.ir.get("operations", []):
            op_name = op.get("name", op.get("op", "unknown"))
            for out in op.get("outputs", {}):
                sources[out] = op_name
                consumers[out] = consumers.get(out, [])
            for inp in op.get("inputs", {}).values():
                if inp.get("type") == "activation":
                    if src := inp.get("source"):
                        consumers[src] = consumers.get(src, [])
                        if op_name not in consumers[src]:
                            consumers[src].append(op_name)

        # Check dangling tensors
        for tensor, producer in sources.items():
            if not consumers.get(tensor):
                if "output" not in tensor and "logits" not in tensor:
                    self.result.warnings.append(f"Dangling: {tensor} produced by {producer}")

        # Check missing sources
        for tensor, ops in consumers.items():
            if tensor not in sources and "input" not in tensor and "tokens" not in tensor:
                self.result.errors.append(f"Missing: {tensor} consumed by {ops}")
                self.result.passed = False

        return self.result.passed

    # =========================================================================
    # MAIN TEST RUNNER
    # =========================================================================

    def run_all_tests(self) -> ValidationResult:
        """Run all memory validation tests."""
        if not self.load():
            self.result.errors.append("Failed to load layout/IR files")
            return self.result

        # Extract regions
        regions = self.extract_regions_from_layout()
        self.result.regions = regions

        print("Running tests...")

        # Test 1: No overlaps
        print("  [1/9] Checking for memory overlaps...")
        self.validate_no_overlaps(regions)

        # Test 2: Contiguity
        print("  [2/9] Checking memory contiguity...")
        self.validate_contiguity(regions)

        # Test 3: Quant size consistency
        print("  [3/9] Validating quantization sizes...")
        self.validate_quant_sizes()

        # Test 4: Quant summary matches weights
        print("  [4/9] Checking quant_summary consistency...")
        self.validate_quant_summary_consistency()

        # Test 5: Layer stride
        print("  [5/9] Validating layer-to-layer stride...")
        self.validate_layer_stride()

        # Test 6: Layer offsets (basic)
        print("  [6/9] Checking layer offsets...")
        self.validate_layer_offsets()

        # Test 7: Activations match ops
        print("  [7/9] Validating activation buffers...")
        self.validate_activation_sizes()
        self.validate_activations_match_ops()

        # Test 8: Canary markers
        print("  [8/9] Checking canary markers...")
        self.validate_canaries()

        # Test 9: Data flow
        print("  [9/9] Validating data flow...")
        self.validate_data_flow()

        return self.result

    def print_report(self):
        """Print validation report."""
        print("\n" + "=" * 70)
        print("MEMORY PLANNER VALIDATION REPORT")
        print("=" * 70)
        print(f"Layout: {self.layout_path}")
        if self.ir_path:
            print(f"IR: {self.ir_path}")

        # Memory summary
        summary = self.compute_memory_summary()
        print(f"\n--- Memory Summary ---")
        print(f"Total weight bytes: {summary['total_weight_bytes']:,} ({summary['total_weight_bytes']/1024/1024:.2f} MB)")
        print(f"By dtype:")
        for dtype, size in sorted(summary["by_dtype"].items()):
            print(f"  {dtype}: {size:,} ({size/1024/1024:.2f} MB)")

        # Test results
        print(f"\n--- Test Results ---")
        if self.result.passed:
            print("PASSED: All memory validation tests")
        else:
            print("FAILED: Errors found")
            for err in self.result.errors:
                print(f"  ERROR: {err}")

        if self.result.warnings:
            print(f"\nWarnings ({len(self.result.warnings)}):")
            for warn in self.result.warnings[:10]:  # Limit output
                print(f"  WARN: {warn}")
            if len(self.result.warnings) > 10:
                print(f"  ... and {len(self.result.warnings) - 10} more")

        # Region listing
        print(f"\n--- Memory Regions ({len(self.result.regions)}) ---")
        sorted_regions = sorted(self.result.regions, key=lambda r: r.offset)
        for region in sorted_regions[:20]:
            print(f"  [{region.offset:12d}] {region.name:30s} {region.size:10d} bytes {region.dtype}")
        if len(sorted_regions) > 20:
            print(f"  ... and {len(sorted_regions) - 20} more regions")

        print("=" * 70)

        return self.result.passed


def main():
    parser = argparse.ArgumentParser(description="Memory layout validator")
    parser.add_argument("--layout", type=Path, required=True,
                        help="Path to layout JSON file")
    parser.add_argument("--ir", type=Path,
                        help="Path to IR JSON file")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print summary")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")

    args = parser.parse_args()

    test = MemoryPlannerTest(args.layout, args.ir)
    result = test.run_all_tests()

    if not args.quiet:
        test.print_report()
    elif args.json:
        print(json.dumps({
            "passed": result.passed,
            "errors": result.errors,
            "warnings": result.warnings,
        }, indent=2))
    else:
        if result.passed:
            print("PASSED")
        else:
            print("FAILED")
            for err in result.errors:
                print(f"  {err}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    exit(main())
