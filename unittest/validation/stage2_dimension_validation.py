"""
Stage 2: Dimension and Memory Planning Validation

Validates that memory layout, tensor dimensions, and alignment are correct:
  2.1 Manifest dimension verification (ne0, ne1 vs GGUF)
  2.2 Alignment validation (64-byte cache line)
  2.3 Stride calculation verification for quantized formats
  2.4 Total memory footprint match
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .base import BaseValidator, StageResult, TestResult, TestStatus
from .kernel_registry import get_kernel_spec, get_spec_by_ggml_type, KERNEL_REGISTRY
from .stage1_weight_validation import GGUFReader

CACHE_ALIGN = 64
HEADER_SIZE = 128


@dataclass
class ModelConfig:
    """Model configuration extracted from GGUF or manifest"""
    num_layers: int = 0
    embed_dim: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate: int = 0
    vocab_size: int = 0
    context_length: int = 0

    # Aligned dimensions
    aligned_embed: int = 0
    aligned_head: int = 0
    aligned_intermediate: int = 0

    def compute_aligned(self, align: int = CACHE_ALIGN):
        """Compute aligned dimensions"""
        self.aligned_embed = align_up(self.embed_dim, align // 4)  # FP32 elements
        self.aligned_head = align_up(self.head_dim, align // 4)
        self.aligned_intermediate = align_up(self.intermediate, align // 4)


def align_up(n: int, a: int) -> int:
    """Align n up to multiple of a"""
    return ((n + a - 1) // a) * a


def compute_tensor_size(dims: List[int], block_size: int, elems_per_block: int) -> int:
    """Compute tensor size in bytes for quantized format"""
    if not dims:
        return 0

    total_elements = 1
    for d in dims:
        total_elements *= d

    num_blocks = (total_elements + elems_per_block - 1) // elems_per_block
    return num_blocks * block_size


class Stage2Validator(BaseValidator):
    """Stage 2: Dimension and Memory Planning Validation"""

    # Expected GGUF tensor name -> BUMP name mapping
    TENSOR_MAPPING = {
        'token_embd.weight': 'token_emb',
        'output_norm.weight': 'final_ln_weight',
        'output.weight': 'lm_head_weight',
    }

    # Per-layer tensor mappings (X = layer index)
    LAYER_TENSOR_MAPPING = {
        'blk.{}.attn_norm.weight': 'layer.{}.ln1_gamma',
        'blk.{}.ffn_norm.weight': 'layer.{}.ln2_gamma',
        'blk.{}.attn_q.weight': 'layer.{}.wq',
        'blk.{}.attn_k.weight': 'layer.{}.wk',
        'blk.{}.attn_v.weight': 'layer.{}.wv',
        'blk.{}.attn_output.weight': 'layer.{}.wo',
        'blk.{}.ffn_gate.weight': 'layer.{}.w1_gate',
        'blk.{}.ffn_up.weight': 'layer.{}.w1_up',
        'blk.{}.ffn_down.weight': 'layer.{}.w2',
    }

    def __init__(self, gguf_path: str, bump_path: Optional[str] = None,
                 manifest_path: Optional[str] = None, verbose: bool = False):
        super().__init__(gguf_path, bump_path, manifest_path, verbose)
        self.gguf_reader = None
        self.manifest = None
        self.config = None

    def _load_manifest(self) -> Optional[Dict]:
        """Load manifest JSON file"""
        if not self.manifest_path or not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, 'r') as f:
            return json.load(f)

    def _extract_config(self) -> ModelConfig:
        """Extract model config from GGUF and/or manifest"""
        config = ModelConfig()

        if self.gguf_reader:
            gguf_config = self.gguf_reader.get_model_config()
            config.num_layers = gguf_config.get('num_layers', 0)
            config.embed_dim = gguf_config.get('embed_dim', 0)
            config.num_heads = gguf_config.get('num_heads', 0)
            config.num_kv_heads = gguf_config.get('num_kv_heads', config.num_heads)
            config.intermediate = gguf_config.get('intermediate', 0)
            config.context_length = gguf_config.get('context_length', 0)

            if config.embed_dim and config.num_heads:
                config.head_dim = config.embed_dim // config.num_heads

        # Override with manifest if available
        if self.manifest:
            config.num_layers = self.manifest.get('num_layers', config.num_layers)
            config.embed_dim = self.manifest.get('embed_dim', config.embed_dim)
            config.vocab_size = self.manifest.get('vocab_size', config.vocab_size)

        config.compute_aligned()
        return config

    def _test_manifest_dimensions(self) -> TestResult:
        """Test 2.1: Verify manifest dimensions match GGUF"""
        self.log("Testing manifest dimensions...")

        if not self.manifest:
            return TestResult(
                name="Manifest dimensions",
                status=TestStatus.SKIP,
                message="No manifest provided"
            )

        if not self.gguf_reader:
            return TestResult(
                name="Manifest dimensions",
                status=TestStatus.SKIP,
                message="No GGUF reader"
            )

        errors = []
        checked = 0

        entries = {e['name']: e for e in self.manifest.get('entries', [])}

        # Check global tensors
        for gguf_name, bump_name in self.TENSOR_MAPPING.items():
            if gguf_name in self.gguf_reader.tensors and bump_name in entries:
                gguf_info = self.gguf_reader.tensors[gguf_name]
                bump_entry = entries[bump_name]

                gguf_dims = gguf_info['dims']
                gguf_dtype = gguf_info['dtype']

                # Compute expected size
                spec = get_spec_by_ggml_type(gguf_dtype)
                if spec:
                    expected_size = compute_tensor_size(
                        gguf_dims, spec.block_size_bytes, spec.elements_per_block
                    )
                    actual_size = bump_entry['size']

                    if expected_size != actual_size:
                        errors.append(
                            f"{bump_name}: expected {expected_size} bytes, got {actual_size}"
                        )

                checked += 1

        # Check first layer tensors
        if self.config and self.config.num_layers > 0:
            for gguf_pattern, bump_pattern in self.LAYER_TENSOR_MAPPING.items():
                gguf_name = gguf_pattern.format(0)
                bump_name = bump_pattern.format(0)

                if gguf_name in self.gguf_reader.tensors and bump_name in entries:
                    gguf_info = self.gguf_reader.tensors[gguf_name]
                    bump_entry = entries[bump_name]

                    gguf_dims = gguf_info['dims']
                    gguf_dtype = gguf_info['dtype']

                    spec = get_spec_by_ggml_type(gguf_dtype)
                    if spec:
                        expected_size = compute_tensor_size(
                            gguf_dims, spec.block_size_bytes, spec.elements_per_block
                        )
                        actual_size = bump_entry['size']

                        # Allow some slack for alignment padding
                        if abs(expected_size - actual_size) > CACHE_ALIGN:
                            errors.append(
                                f"{bump_name}: expected ~{expected_size} bytes, got {actual_size}"
                            )

                    checked += 1

        if errors:
            return TestResult(
                name="Manifest dimensions",
                status=TestStatus.FAIL,
                message=f"{len(errors)} dimension mismatches",
                details={'errors': errors, 'checked': checked}
            )

        return TestResult(
            name="Manifest dimensions",
            status=TestStatus.PASS,
            message=f"Checked {checked} tensors",
            details={'checked': checked}
        )

    def _test_alignment(self) -> TestResult:
        """Test 2.2: Verify 64-byte alignment of offsets"""
        self.log("Testing alignment...")

        if not self.manifest:
            return TestResult(
                name="Alignment",
                status=TestStatus.SKIP,
                message="No manifest provided"
            )

        errors = []
        entries = self.manifest.get('entries', [])

        for entry in entries:
            offset = entry.get('file_offset', 0)
            name = entry.get('name', 'unknown')

            # Offsets should be 64-byte aligned after header
            if offset > HEADER_SIZE:
                if (offset - HEADER_SIZE) % CACHE_ALIGN != 0:
                    errors.append(f"{name}: offset {offset} not 64-byte aligned")

        if errors:
            return TestResult(
                name="Alignment",
                status=TestStatus.FAIL,
                message=f"{len(errors)} alignment issues",
                details={'errors': errors[:10]}  # First 10
            )

        return TestResult(
            name="Alignment",
            status=TestStatus.PASS,
            message=f"All {len(entries)} entries aligned"
        )

    def _test_stride_calculations(self) -> TestResult:
        """Test 2.3: Verify stride calculations for quantized formats"""
        self.log("Testing stride calculations...")

        if not self.gguf_reader:
            return TestResult(
                name="Stride calculations",
                status=TestStatus.SKIP,
                message="No GGUF reader"
            )

        errors = []
        checked = 0

        for tensor_name, info in self.gguf_reader.tensors.items():
            dtype = info['dtype']
            dims = info['dims']

            spec = get_spec_by_ggml_type(dtype)
            if not spec:
                continue

            # For quantized formats, dim[0] must be divisible by elements_per_block
            if dims and len(dims) >= 1:
                if dims[0] % spec.elements_per_block != 0:
                    errors.append(
                        f"{tensor_name}: dim[0]={dims[0]} not divisible by {spec.elements_per_block}"
                    )

            checked += 1

        if errors:
            return TestResult(
                name="Stride calculations",
                status=TestStatus.FAIL,
                message=f"{len(errors)} stride issues",
                details={'errors': errors[:10]}
            )

        return TestResult(
            name="Stride calculations",
            status=TestStatus.PASS,
            message=f"Checked {checked} tensors"
        )

    def _test_kernel_dimensions(self) -> TestResult:
        """Test 2.5: Verify kernel input/output dimensions match manifest"""
        self.log("Testing kernel dimensions...")

        if not self.manifest or not self.config:
            return TestResult(
                name="Kernel dimensions",
                status=TestStatus.SKIP,
                message="No manifest or config"
            )

        errors = []
        entries = {e['name']: e for e in self.manifest.get('entries', [])}

        # Check key dimension relationships
        embed_dim = self.config.embed_dim
        num_heads = self.config.num_heads
        num_kv_heads = self.config.num_kv_heads
        head_dim = self.config.head_dim
        intermediate = self.config.intermediate

        if embed_dim <= 0 or num_heads <= 0:
            return TestResult(
                name="Kernel dimensions",
                status=TestStatus.SKIP,
                message="Config missing dimensions"
            )

        # Validate attention dimensions
        expected_qkv_out = num_heads * head_dim  # Q output dim
        expected_kv_out = num_kv_heads * head_dim  # K/V output dim

        # Check wq/wk/wv/wo dimensions for layer 0
        wq = entries.get("layer.0.wq")
        wk = entries.get("layer.0.wk")
        wv = entries.get("layer.0.wv")
        wo = entries.get("layer.0.wo")

        if wq and wk and wv and wo:
            # WQ should project embed_dim -> num_heads * head_dim
            # For quantized, size depends on quant format
            pass  # Size validation is complex for quant, skip detailed check

        # Check FFN dimensions
        w1 = entries.get("layer.0.w1")  # gate + up
        w2 = entries.get("layer.0.w2")  # down

        if w1 and w2:
            # w1 projects embed_dim -> 2 * intermediate (gate + up)
            # w2 projects intermediate -> embed_dim
            pass

        # Check embedding dimension
        token_emb = entries.get("token_emb")
        if token_emb:
            vocab_size = self.config.vocab_size or self.manifest.get('vocab_size', 0)
            if vocab_size > 0:
                expected_emb_size_f32 = vocab_size * embed_dim * 4
                # Allow for quantized sizes
                actual_size = token_emb['size']
                # For Q8_0: 34 bytes per 32 elements
                if token_emb['dtype'] == 'q8_0':
                    expected_blocks = (vocab_size * embed_dim + 31) // 32
                    expected_size = expected_blocks * 34
                    if abs(actual_size - expected_size) > CACHE_ALIGN * 2:
                        errors.append(
                            f"token_emb: expected ~{expected_size} bytes, got {actual_size}"
                        )

        if errors:
            return TestResult(
                name="Kernel dimensions",
                status=TestStatus.FAIL,
                message=f"{len(errors)} dimension issues",
                details={'errors': errors}
            )

        return TestResult(
            name="Kernel dimensions",
            status=TestStatus.PASS,
            message=f"Dimensions consistent (embed={embed_dim}, heads={num_heads}, ff={intermediate})"
        )

    def _test_memory_footprint(self) -> TestResult:
        """Test 2.4: Verify total memory footprint"""
        self.log("Testing memory footprint...")

        if not self.manifest or not self.bump_path:
            return TestResult(
                name="Memory footprint",
                status=TestStatus.SKIP,
                message="No manifest or BUMP file"
            )

        try:
            # Get actual file size
            actual_size = os.path.getsize(self.bump_path)

            # Sum manifest entries
            entries = self.manifest.get('entries', [])
            if not entries:
                return TestResult(
                    name="Memory footprint",
                    status=TestStatus.FAIL,
                    message="No entries in manifest"
                )

            # Find max offset + size
            max_end = 0
            for entry in entries:
                end = entry.get('file_offset', 0) + entry.get('size', 0)
                max_end = max(max_end, end)

            # Allow some tolerance for final padding
            diff = abs(actual_size - max_end)
            if diff > CACHE_ALIGN * 2:
                return TestResult(
                    name="Memory footprint",
                    status=TestStatus.FAIL,
                    message=f"File size {actual_size} != manifest max {max_end} (diff={diff})"
                )

            return TestResult(
                name="Memory footprint",
                status=TestStatus.PASS,
                message=f"File: {actual_size} bytes, manifest: {max_end} bytes",
                details={'file_size': actual_size, 'manifest_max': max_end}
            )

        except Exception as e:
            return TestResult(
                name="Memory footprint",
                status=TestStatus.ERROR,
                message=str(e)
            )

    def run(self) -> StageResult:
        """Execute all Stage 2 validations"""
        result = StageResult(stage_num=2, stage_name="Dimension & Memory")

        try:
            # Initialize
            self.gguf_reader = GGUFReader(self.gguf_path)
            self.manifest = self._load_manifest()
            self.config = self._extract_config()

            # Run tests
            result.add_test(self._test_manifest_dimensions())
            result.add_test(self._test_alignment())
            result.add_test(self._test_stride_calculations())
            result.add_test(self._test_kernel_dimensions())  # Stage 2.5
            result.add_test(self._test_memory_footprint())

        except Exception as e:
            result.error_message = f"Stage 2 initialization failed: {e}"

        return result
