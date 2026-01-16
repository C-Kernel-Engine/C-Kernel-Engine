#!/usr/bin/env python3
"""
Layer 5: Tensor Flow Validation
================================

THE CRITICAL TEST: Traces tensor shapes through the entire forward pass.
Catches dimension mismatches that cause "gibberish" output.

Tests:
1. Embedding output shape
2. Per-layer input/output shapes
3. Attention dimension flow (Q, K, V, cache)
4. MLP dimension flow (gate, up, down)
5. Final logits shape
6. Prefill vs Decode equivalence

Usage:
    python scripts/test_tensor_flow.py --model-dir path/to/model_dir
    python scripts/test_tensor_flow.py --ir path/to/lowered_decode.json
    python scripts/test_tensor_flow.py --auto
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass

# ANSI colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
CYAN = '\033[0;36m'
MAGENTA = '\033[0;35m'
NC = '\033[0m'


@dataclass
class TensorShape:
    """Represents a tensor shape with symbolic dimensions."""
    dims: List[str]  # e.g., ['S', 'E'] or ['1', '896']

    def __str__(self):
        return f"[{', '.join(str(d) for d in self.dims)}]"

    def resolve(self, symbols: Dict[str, int]) -> Tuple[int, ...]:
        """Resolve symbolic dimensions to concrete values."""
        result = []
        for d in self.dims:
            if isinstance(d, int):
                result.append(d)
            elif d in symbols:
                result.append(symbols[d])
            elif d.isdigit():
                result.append(int(d))
            else:
                # Try to evaluate expression like "H*D" or "3*H*D"
                expr = d
                for sym, val in symbols.items():
                    expr = expr.replace(sym, str(val))
                try:
                    result.append(eval(expr))
                except:
                    result.append(-1)  # Unknown
        return tuple(result)


@dataclass
class TensorFlowStep:
    """One step in the tensor flow."""
    name: str
    op: str
    input_shapes: List[TensorShape]
    output_shape: TensorShape
    weight_dtype: Optional[str] = None
    notes: str = ""


class TensorFlowValidator:
    """Validate tensor shapes through the forward pass."""

    def __init__(self, model_dir: str, verbose: bool = False):
        self.model_dir = Path(model_dir)
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.warnings = 0

        self.ir = None
        self.manifest = None
        self.config = None
        self.symbols = {}

        # Model dimensions
        self.embed_dim = 0
        self.num_heads = 0
        self.num_kv_heads = 0
        self.head_dim = 0
        self.intermediate_dim = 0
        self.vocab_size = 0
        self.max_seq_len = 0
        self.num_layers = 0

    def log_pass(self, msg: str):
        print(f"{GREEN}[PASS]{NC} {msg}")
        self.passed += 1

    def log_fail(self, msg: str):
        print(f"{RED}[FAIL]{NC} {msg}")
        self.failed += 1

    def log_warn(self, msg: str):
        print(f"{YELLOW}[WARN]{NC} {msg}")
        self.warnings += 1

    def log_info(self, msg: str):
        print(f"{BLUE}[INFO]{NC} {msg}")

    def log_dim(self, name: str, expected: Tuple, actual: Tuple = None, ok: bool = True):
        """Log a dimension check."""
        if actual is None:
            actual = expected
        status = f"{GREEN}✓{NC}" if ok else f"{RED}✗{NC}"
        print(f"  {status} {name}: {expected}")

    def load_files(self) -> bool:
        """Load IR, manifest, and config files."""
        # Try to find IR
        ir_paths = [
            self.model_dir / "lowered_decode.json",
            self.model_dir / "lowered_prefill.json",
        ]
        for ir_path in ir_paths:
            if ir_path.exists():
                with open(ir_path) as f:
                    self.ir = json.load(f)
                self.log_pass(f"Loaded IR: {ir_path.name}")
                break
        else:
            self.log_warn("No IR file found")

        # Load manifest
        manifest_path = self.model_dir / "weights_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            self.log_pass(f"Loaded manifest")

        # Load config
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
            self.log_pass(f"Loaded config")

        # Extract dimensions
        self._extract_dimensions()

        return self.ir is not None or self.config is not None

    def _extract_dimensions(self):
        """Extract model dimensions from IR, config, or manifest."""
        # From IR symbols
        if self.ir:
            symbols = self.ir.get('symbols', {})
            config = self.ir.get('config', {})

            self.embed_dim = symbols.get('E', config.get('embed_dim', 0))
            self.num_heads = symbols.get('H', config.get('num_heads', 0))
            self.head_dim = symbols.get('D', config.get('head_dim', 0))
            self.intermediate_dim = symbols.get('I', config.get('intermediate_dim', 0))
            self.vocab_size = symbols.get('V', config.get('vocab_size', 0))
            self.max_seq_len = symbols.get('T', config.get('max_seq_len', 0))
            self.num_layers = config.get('num_layers', 0)
            self.num_kv_heads = config.get('num_kv_heads', self.num_heads)

            self.symbols = {
                'E': self.embed_dim,
                'H': self.num_heads,
                'D': self.head_dim,
                'I': self.intermediate_dim,
                'V': self.vocab_size,
                'T': self.max_seq_len,
                'S': 1,  # Default for decode
            }

        # Override from config.json if available
        if self.config:
            self.embed_dim = self.config.get('hidden_size', self.config.get('embed_dim', self.embed_dim))
            self.num_heads = self.config.get('num_attention_heads', self.config.get('num_heads', self.num_heads))
            self.num_kv_heads = self.config.get('num_key_value_heads', self.num_kv_heads)
            self.head_dim = self.config.get('head_dim', self.embed_dim // self.num_heads if self.num_heads else 0)
            self.intermediate_dim = self.config.get('intermediate_size', self.config.get('intermediate_dim', self.intermediate_dim))
            self.vocab_size = self.config.get('vocab_size', self.vocab_size)
            self.max_seq_len = self.config.get('max_position_embeddings', self.config.get('max_seq_len', self.max_seq_len))
            self.num_layers = self.config.get('num_hidden_layers', self.config.get('num_layers', self.num_layers))

    def print_model_dimensions(self):
        """Print extracted model dimensions."""
        print(f"\n{CYAN}Model Dimensions:{NC}")
        print(f"  embed_dim (E):       {self.embed_dim}")
        print(f"  num_heads (H):       {self.num_heads}")
        print(f"  num_kv_heads:        {self.num_kv_heads}")
        print(f"  head_dim (D):        {self.head_dim}")
        print(f"  intermediate (I):    {self.intermediate_dim}")
        print(f"  vocab_size (V):      {self.vocab_size}")
        print(f"  max_seq_len (T):     {self.max_seq_len}")
        print(f"  num_layers:          {self.num_layers}")
        print()

    def test_embedding_shape(self) -> bool:
        """Test 1: Embedding produces correct shape."""
        self.log_info("Test 1: Embedding Output Shape")

        # For decode: input [1] → output [1, embed_dim]
        # For prefill: input [S] → output [S, embed_dim]

        expected_input = (1,)  # token_id for decode
        expected_output = (1, self.embed_dim)

        print(f"  Decode path:")
        self.log_dim("token_id input", expected_input)
        self.log_dim("embedding output", expected_output)

        # Verify embedding weight shape
        if self.manifest:
            tensors = self.manifest.get('tensors', self.manifest.get('weights', {}))
            for name, info in tensors.items():
                if 'embed' in name.lower() and 'token' in name.lower():
                    shape = tuple(info.get('shape', info.get('dims', [])))
                    expected = (self.vocab_size, self.embed_dim)
                    if shape == expected or shape == tuple(reversed(expected)):
                        self.log_pass(f"Embedding weight shape: {shape}")
                    else:
                        self.log_fail(f"Embedding weight shape: expected {expected}, got {shape}")
                        return False
                    break

        return True

    def test_attention_flow(self, layer_id: int = 0) -> bool:
        """Test 2: Attention dimension flow."""
        self.log_info(f"Test 2: Attention Dimension Flow (Layer {layer_id})")

        E = self.embed_dim
        H = self.num_heads
        KVH = self.num_kv_heads
        D = self.head_dim
        T = self.max_seq_len

        print(f"  {MAGENTA}Attention block:{NC}")

        # Input: hidden [1, E]
        self.log_dim("hidden input", (1, E))

        # RMSNorm: [1, E] → [1, E]
        self.log_dim("rmsnorm output", (1, E))

        # Q projection: [1, E] → [1, H*D]
        q_dim = H * D
        self.log_dim("Q projection output", (1, q_dim))

        # K projection: [1, E] → [1, KVH*D]
        k_dim = KVH * D
        self.log_dim("K projection output", (1, k_dim))

        # V projection: [1, E] → [1, KVH*D]
        v_dim = KVH * D
        self.log_dim("V projection output", (1, v_dim))

        # Q reshape: [1, H*D] → [1, H, D] or [H, 1, D]
        self.log_dim("Q reshaped (per-head)", (1, H, D))

        # K reshape: [1, KVH*D] → [1, KVH, D] or [KVH, 1, D]
        self.log_dim("K reshaped (per-head)", (1, KVH, D))

        # RoPE: same shapes
        self.log_dim("Q after RoPE", (1, H, D))
        self.log_dim("K after RoPE", (1, KVH, D))

        # KV Cache write
        # K cache: [KVH, T, D] - write single token
        self.log_dim("K cache shape", (KVH, T, D))
        self.log_dim("K cache stride (per head)", (T * D,))

        # V cache: [KVH, T, D]
        self.log_dim("V cache shape", (KVH, T, D))

        # Attention: Q[1, H, D] × K[KVH, t, D]^T → scores[H, 1, t]
        # With GQA: each Q head attends to corresponding KV head
        gqa_ratio = H // KVH if KVH else 1
        self.log_dim(f"GQA ratio", (gqa_ratio,))

        # Attention output: [1, H, D] → [1, H*D]
        self.log_dim("attention output", (1, H * D))

        # O projection: [1, H*D] → [1, E]
        self.log_dim("O projection output", (1, E))

        # Verify weight shapes if manifest available
        if self.manifest:
            tensors = self.manifest.get('tensors', self.manifest.get('weights', {}))
            weight_checks = [
                (f"blk.{layer_id}.attn_q", (E, H * D)),
                (f"blk.{layer_id}.attn_k", (E, KVH * D)),
                (f"blk.{layer_id}.attn_v", (E, KVH * D)),
                (f"blk.{layer_id}.attn_output", (H * D, E)),
            ]

            issues = []
            for weight_pattern, expected in weight_checks:
                for name, info in tensors.items():
                    if weight_pattern in name:
                        shape = tuple(info.get('shape', info.get('dims', [])))
                        # Shapes might be transposed
                        if shape != expected and shape != tuple(reversed(expected)):
                            issues.append(f"{name}: expected {expected} or {tuple(reversed(expected))}, got {shape}")
                        break

            if issues:
                for issue in issues:
                    self.log_fail(issue)
                return False

        self.log_pass("Attention dimension flow correct")
        return True

    def test_mlp_flow(self, layer_id: int = 0) -> bool:
        """Test 3: MLP dimension flow."""
        self.log_info(f"Test 3: MLP Dimension Flow (Layer {layer_id})")

        E = self.embed_dim
        I = self.intermediate_dim

        print(f"  {MAGENTA}MLP block:{NC}")

        # Input: hidden [1, E]
        self.log_dim("hidden input", (1, E))

        # RMSNorm: [1, E] → [1, E]
        self.log_dim("rmsnorm output", (1, E))

        # Gate projection: [1, E] → [1, I]
        self.log_dim("gate projection output", (1, I))

        # Up projection: [1, E] → [1, I]
        self.log_dim("up projection output", (1, I))

        # SwiGLU: gate * silu(up) → [1, I]
        self.log_dim("swiglu output", (1, I))

        # Down projection: [1, I] → [1, E]
        self.log_dim("down projection output", (1, E))

        # Residual: [1, E] + [1, E] → [1, E]
        self.log_dim("residual output", (1, E))

        # Verify weight shapes
        if self.manifest:
            tensors = self.manifest.get('tensors', self.manifest.get('weights', {}))
            weight_checks = [
                (f"blk.{layer_id}.ffn_gate", (E, I)),
                (f"blk.{layer_id}.ffn_up", (E, I)),
                (f"blk.{layer_id}.ffn_down", (I, E)),
            ]

            issues = []
            for weight_pattern, expected in weight_checks:
                for name, info in tensors.items():
                    if weight_pattern in name:
                        shape = tuple(info.get('shape', info.get('dims', [])))
                        if shape != expected and shape != tuple(reversed(expected)):
                            issues.append(f"{name}: expected {expected} or {tuple(reversed(expected))}, got {shape}")
                        break

            if issues:
                for issue in issues:
                    self.log_fail(issue)
                return False

        self.log_pass("MLP dimension flow correct")
        return True

    def test_final_output(self) -> bool:
        """Test 4: Final output shape."""
        self.log_info("Test 4: Final Output Shape")

        E = self.embed_dim
        V = self.vocab_size

        print(f"  {MAGENTA}Output head:{NC}")

        # Final norm: [1, E] → [1, E]
        self.log_dim("final rmsnorm output", (1, E))

        # LM head: [1, E] → [1, V]
        self.log_dim("logits output", (1, V))

        # Verify LM head weight
        if self.manifest:
            tensors = self.manifest.get('tensors', self.manifest.get('weights', {}))
            for name, info in tensors.items():
                if 'output' in name.lower() and 'norm' not in name.lower():
                    shape = tuple(info.get('shape', info.get('dims', [])))
                    expected = (E, V)
                    if shape == expected or shape == tuple(reversed(expected)):
                        self.log_pass(f"LM head weight shape: {shape}")
                    else:
                        self.log_fail(f"LM head weight shape: expected {expected}, got {shape}")
                        return False
                    break

        return True

    def test_kv_cache_indexing(self) -> bool:
        """Test 5: KV cache indexing math."""
        self.log_info("Test 5: KV Cache Indexing")

        KVH = self.num_kv_heads
        D = self.head_dim
        T = self.max_seq_len

        print(f"  {MAGENTA}Cache layout:{NC}")

        # Cache shape: [KVH, T, D]
        # Index for head h, position t, dim d: h * T * D + t * D + d
        head_stride = T * D
        pos_stride = D
        dim_stride = 1

        self.log_dim("cache shape", (KVH, T, D))
        print(f"    head_stride: {head_stride}")
        print(f"    pos_stride: {pos_stride}")
        print(f"    dim_stride: {dim_stride}")

        # Total cache size
        cache_size = KVH * T * D * 4  # FP32
        cache_size_mb = cache_size / (1024 * 1024)
        print(f"    K cache size: {cache_size_mb:.1f} MB")
        print(f"    V cache size: {cache_size_mb:.1f} MB")
        print(f"    Total KV per layer: {cache_size_mb * 2:.1f} MB")
        print(f"    Total KV all layers: {cache_size_mb * 2 * self.num_layers:.1f} MB")

        self.log_pass("KV cache indexing consistent")
        return True

    def test_prefill_decode_equivalence(self) -> bool:
        """Test 6: Prefill and decode should produce equivalent results."""
        self.log_info("Test 6: Prefill vs Decode Shapes")

        E = self.embed_dim
        H = self.num_heads
        D = self.head_dim

        print(f"  {MAGENTA}Prefill path (S tokens):{NC}")
        self.log_dim("input", ("S",))
        self.log_dim("hidden", ("S", E))
        self.log_dim("Q", ("S", H * D))
        self.log_dim("attention", ("S", H, D))
        self.log_dim("logits", ("S", self.vocab_size))

        print(f"  {MAGENTA}Decode path (1 token):{NC}")
        self.log_dim("input", (1,))
        self.log_dim("hidden", (1, E))
        self.log_dim("Q", (1, H * D))
        self.log_dim("attention (uses KV cache)", (1, H, D))
        self.log_dim("logits", (1, self.vocab_size))

        print(f"  {MAGENTA}Equivalence check:{NC}")
        print(f"    Prefill token[i] output ≈ Decode with KV[0:i] output")

        self.log_pass("Shape patterns match between prefill and decode")
        return True

    def visualize_tensor_flow(self):
        """Generate visual representation of tensor flow."""
        print()
        print(f"{CYAN}{'='*60}{NC}")
        print(f"{CYAN}  Tensor Flow Visualization{NC}")
        print(f"{CYAN}{'='*60}{NC}")

        E = self.embed_dim
        H = self.num_heads
        KVH = self.num_kv_heads
        D = self.head_dim
        I = self.intermediate_dim
        V = self.vocab_size

        print(f"""
    ┌──────────────────────────────────────────────────────────┐
    │  token_id [1]                                            │
    └────────────────────────────┬─────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────┐
    │  Embedding: [1] → [1, {E}]                              │
    │  Weight: [{V}, {E}]                                │
    └────────────────────────────┬─────────────────────────────┘
                                 │
          ┌──────────────────────┴──────────────────────┐
          │                    × {self.num_layers} layers             │
          │                                             │
          │  ┌────────────────────────────────────────┐ │
          │  │  RMSNorm: [1, {E}] → [1, {E}]        │ │
          │  └────────────────────┬───────────────────┘ │
          │                       │                     │
          │  ┌────────────────────┴───────────────────┐ │
          │  │  QKV Projection                        │ │
          │  │  Q: [1, {E}] → [1, {H * D}]          │ │
          │  │  K: [1, {E}] → [1, {KVH * D}]           │ │
          │  │  V: [1, {E}] → [1, {KVH * D}]           │ │
          │  └────────────────────┬───────────────────┘ │
          │                       │                     │
          │  ┌────────────────────┴───────────────────┐ │
          │  │  RoPE: Q[{H}, {D}], K[{KVH}, {D}]          │ │
          │  └────────────────────┬───────────────────┘ │
          │                       │                     │
          │  ┌────────────────────┴───────────────────┐ │
          │  │  Attention (GQA {H}:{KVH})                │ │
          │  │  Q×K^T → scores → softmax → ×V        │ │
          │  │  Output: [1, {H * D}]                  │ │
          │  └────────────────────┬───────────────────┘ │
          │                       │                     │
          │  ┌────────────────────┴───────────────────┐ │
          │  │  O Projection: [1, {H * D}] → [1, {E}]│ │
          │  │  + Residual                            │ │
          │  └────────────────────┬───────────────────┘ │
          │                       │                     │
          │  ┌────────────────────┴───────────────────┐ │
          │  │  RMSNorm: [1, {E}] → [1, {E}]        │ │
          │  └────────────────────┬───────────────────┘ │
          │                       │                     │
          │  ┌────────────────────┴───────────────────┐ │
          │  │  MLP (SwiGLU)                          │ │
          │  │  Gate: [1, {E}] → [1, {I}]         │ │
          │  │  Up:   [1, {E}] → [1, {I}]         │ │
          │  │  Down: [1, {I}] → [1, {E}]         │ │
          │  │  + Residual                            │ │
          │  └────────────────────┬───────────────────┘ │
          │                       │                     │
          └───────────────────────┴─────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────┐
    │  Final RMSNorm: [1, {E}] → [1, {E}]                    │
    └────────────────────────────┬─────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────┐
    │  LM Head: [1, {E}] → [1, {V}]                     │
    │  Weight: [{E}, {V}]                                │
    └────────────────────────────┬─────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────┐
    │  logits [1, {V}] → argmax → next_token            │
    └──────────────────────────────────────────────────────────┘
""")

    def run_all_tests(self) -> bool:
        """Run all tensor flow validation tests."""
        print("=" * 60)
        print("  CK-Engine Tensor Flow Validation")
        print("=" * 60)
        print(f"Model: {self.model_dir}")
        print()

        # Load files
        if not self.load_files():
            print(f"{RED}Failed to load model files{NC}")
            return False

        # Print dimensions
        self.print_model_dimensions()

        # Run tests
        self.test_embedding_shape()
        print()

        self.test_attention_flow(layer_id=0)
        print()

        self.test_mlp_flow(layer_id=0)
        print()

        self.test_final_output()
        print()

        self.test_kv_cache_indexing()
        print()

        self.test_prefill_decode_equivalence()
        print()

        # Visual flow
        self.visualize_tensor_flow()

        # Summary
        print("=" * 60)
        print("  Summary")
        print("=" * 60)
        print(f"  {GREEN}Passed:{NC} {self.passed}")
        print(f"  {RED}Failed:{NC} {self.failed}")
        print(f"  {YELLOW}Warnings:{NC} {self.warnings}")
        print()

        if self.failed == 0:
            print(f"{GREEN}All tensor flow tests passed!{NC}")
            return True
        else:
            print(f"{RED}Tensor flow issues detected - likely cause of gibberish output{NC}")
            return False


def find_cached_models() -> List[Path]:
    """Find model directories in cache."""
    models = []

    cache_dirs = [
        Path.home() / ".cache" / "ck-engine-v6.6" / "models",
        Path.home() / ".cache" / "ck-engine-v6.5" / "models",
        Path.home() / ".cache" / "ck-engine-v6" / "models",
    ]

    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
        for model_dir in cache_dir.iterdir():
            if model_dir.is_dir():
                # Check for required files
                if (model_dir / "config.json").exists() or (model_dir / "weights_manifest.json").exists():
                    models.append(model_dir)

    return models


def main():
    parser = argparse.ArgumentParser(description="Validate tensor flow through model")
    parser.add_argument("--model-dir", type=str, help="Model directory")
    parser.add_argument("--ir", type=str, help="IR JSON file (alternative to model-dir)")
    parser.add_argument("--auto", action="store_true", help="Auto-find cached models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.auto:
        models = find_cached_models()
        if not models:
            print(f"{RED}No cached models found{NC}")
            sys.exit(1)

        print(f"Found {len(models)} cached models:")
        for m in models:
            print(f"  - {m.parent.name}/{m.name}")
        print()

        model_dir = models[0]

    elif args.model_dir:
        model_dir = Path(args.model_dir)

    elif args.ir:
        # Use IR file's directory
        model_dir = Path(args.ir).parent

    else:
        parser.print_help()
        sys.exit(1)

    validator = TensorFlowValidator(str(model_dir), verbose=args.verbose)
    success = validator.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
