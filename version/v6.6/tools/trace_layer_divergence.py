#!/usr/bin/env python3
"""
trace_layer_divergence.py - Find exactly where v6.6 diverges from reference

This tool runs both v6.5 and v6.6 with instrumentation to dump intermediate
tensors at each operation, then compares them to find the first divergence point.

Stages traced:
1. Embedding output
2. Layer N RMSNorm1 output
3. Layer N Q/K/V projections
4. Layer N RoPE output
5. Layer N Attention output
6. Layer N Out projection
7. Layer N Residual1
8. Layer N RMSNorm2 output
9. Layer N MLP (gate, up, down)
10. Layer N Residual2
11. Final RMSNorm
12. Logits

Usage:
    python trace_layer_divergence.py --v65-lib <path> --v66-lib <path> --prompt "Hello"
    python trace_layer_divergence.py --v66-only --stop-at-layer 0
"""

import argparse
import ctypes
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TensorDump:
    """A dumped tensor with metadata."""
    name: str
    stage: str
    layer: int
    shape: Tuple[int, ...]
    data: np.ndarray
    offset: int = 0

    def summary(self, max_vals: int = 5) -> str:
        """Return a summary string."""
        flat = self.data.flatten()
        preview = flat[:max_vals]
        return (f"{self.name} @ layer {self.layer}: "
                f"shape={self.shape}, "
                f"range=[{flat.min():.6f}, {flat.max():.6f}], "
                f"mean={flat.mean():.6f}, "
                f"first{max_vals}={preview}")


class ModelInstrumenter:
    """Instruments a CK model to dump intermediate tensors."""

    def __init__(self, lib_path: str, weights_path: str):
        self.lib_path = lib_path
        self.weights_path = weights_path
        self.lib = None
        self.dumps: Dict[str, TensorDump] = {}

    def load(self):
        """Load the model library."""
        self.lib = ctypes.CDLL(self.lib_path)

        # Set up function signatures
        self.lib.ck_model_init.argtypes = [ctypes.c_char_p]
        self.lib.ck_model_init.restype = ctypes.c_int

        self.lib.ck_model_embed_tokens.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int
        ]
        self.lib.ck_model_embed_tokens.restype = ctypes.c_int

        self.lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
        self.lib.ck_model_get_vocab_size.restype = ctypes.c_int
        self.lib.ck_model_get_context_window.restype = ctypes.c_int
        self.lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

        self.lib.ck_model_kv_cache_reset.argtypes = []
        self.lib.ck_model_kv_cache_reset.restype = None

        # Initialize
        ret = self.lib.ck_model_init(self.weights_path.encode())
        if ret != 0:
            raise RuntimeError(f"Failed to init model: {ret}")

        self.base_ptr = self.lib.ck_model_get_base_ptr()
        self.vocab_size = self.lib.ck_model_get_vocab_size()

    def unload(self):
        """Unload the model."""
        if self.lib:
            try:
                self.lib.ck_model_free()
            except:
                pass

    def read_buffer(self, offset: int, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """Read a buffer from model memory."""
        ptr = ctypes.cast(
            self.base_ptr + offset,
            ctypes.POINTER(ctypes.c_float)
        )
        size = int(np.prod(shape))
        arr = np.ctypeslib.as_array(ptr, shape=(size,))
        return arr.reshape(shape).copy()

    def run_with_stop(self, tokens: List[int], stop_op: int) -> Dict[str, TensorDump]:
        """Run inference stopping at a specific op."""
        os.environ['CK_STOP_OP'] = str(stop_op)

        # Reset KV cache
        self.lib.ck_model_kv_cache_reset()

        # Run
        token_arr = (ctypes.c_int32 * len(tokens))(*tokens)
        self.lib.ck_model_embed_tokens(token_arr, len(tokens))

        # Clear stop
        if 'CK_STOP_OP' in os.environ:
            del os.environ['CK_STOP_OP']

        return {}


def compare_tensors(t1: np.ndarray, t2: np.ndarray, name: str, atol: float = 1e-4, rtol: float = 1e-3) -> Dict:
    """Compare two tensors and return comparison metrics."""
    if t1.shape != t2.shape:
        return {
            'name': name,
            'match': False,
            'error': f'Shape mismatch: {t1.shape} vs {t2.shape}'
        }

    diff = np.abs(t1 - t2)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Find first divergence point
    divergence_idx = None
    flat1 = t1.flatten()
    flat2 = t2.flatten()
    for i in range(len(flat1)):
        if abs(flat1[i] - flat2[i]) > atol:
            divergence_idx = i
            break

    # Correlation
    if t1.std() > 1e-10 and t2.std() > 1e-10:
        corr = np.corrcoef(flat1, flat2)[0, 1]
    else:
        corr = 1.0 if np.allclose(t1, t2) else 0.0

    match = np.allclose(t1, t2, atol=atol, rtol=rtol)

    return {
        'name': name,
        'match': match,
        'max_diff': float(max_diff),
        'mean_diff': float(mean_diff),
        'correlation': float(corr),
        'divergence_idx': divergence_idx,
        't1_range': (float(flat1.min()), float(flat1.max())),
        't2_range': (float(flat2.min()), float(flat2.max())),
        't1_mean': float(flat1.mean()),
        't2_mean': float(flat2.mean()),
    }


def load_layout_offsets(model_dir: Path, mode: str = 'decode') -> Dict[str, int]:
    """Load activation buffer offsets from layout JSON."""
    import json

    layout_path = model_dir / f"layout_{mode}.json"
    if not layout_path.exists():
        return {}

    with open(layout_path) as f:
        layout = json.load(f)

    offsets = {}
    arena_base = layout.get('memory', {}).get('arena', {}).get('activations_base', 0)

    for buf in layout.get('memory', {}).get('activations', {}).get('buffers', []):
        name = buf.get('name', '')
        offset = buf.get('offset', 0)
        offsets[name] = arena_base + offset

    return offsets


def trace_single_model(lib_path: str, weights_path: str, model_dir: Path,
                       tokens: List[int], config: Dict) -> Dict[str, np.ndarray]:
    """Trace a single model and dump intermediate tensors."""
    import json

    # Load the model
    lib = ctypes.CDLL(lib_path)

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int

    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int

    lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    lib.ck_model_kv_cache_reset.argtypes = []
    lib.ck_model_kv_cache_reset.restype = None

    ret = lib.ck_model_init(weights_path.encode())
    if ret != 0:
        raise RuntimeError(f"Failed to init model from {lib_path}")

    base_ptr = lib.ck_model_get_base_ptr()
    vocab_size = lib.ck_model_get_vocab_size()

    # Load layout offsets
    offsets = load_layout_offsets(model_dir)

    # Helper to read buffer
    def read_buf(offset: int, shape: Tuple) -> np.ndarray:
        ptr = ctypes.cast(base_ptr + offset, ctypes.POINTER(ctypes.c_float))
        size = int(np.prod(shape))
        arr = np.ctypeslib.as_array(ptr, shape=(size,))
        return arr.reshape(shape).copy()

    dumps = {}
    embed_dim = config.get('embed_dim', 896)
    num_tokens = len(tokens)

    # We'll trace by running with CK_STOP_OP at different points
    stages = [
        (0, 'embedding', (num_tokens, embed_dim), 'embedded_input'),
        (1, 'layer0_rmsnorm1', (num_tokens, embed_dim), 'layer_input'),
    ]

    # For more detailed tracing, we need to know the op sequence
    # This is a simplified version that dumps at key activation buffers

    # Run full inference
    lib.ck_model_kv_cache_reset()
    token_arr = (ctypes.c_int32 * len(tokens))(*tokens)
    lib.ck_model_embed_tokens(token_arr, len(tokens))

    # Dump key buffers
    buffer_shapes = {
        'embedded_input': (num_tokens, embed_dim),
        'layer_input': (num_tokens, embed_dim),
        'layer_output': (num_tokens, embed_dim),
        'logits': (num_tokens, vocab_size) if num_tokens == 1 else (vocab_size,),
    }

    for buf_name, shape in buffer_shapes.items():
        if buf_name in offsets:
            try:
                dumps[buf_name] = read_buf(offsets[buf_name], shape)
            except Exception as e:
                print(f"  Warning: Could not read {buf_name}: {e}")

    # Get logits directly
    logits_ptr = lib.ck_model_get_logits()
    if logits_ptr:
        logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
        dumps['logits'] = logits.copy()

    lib.ck_model_free()

    return dumps


def main():
    parser = argparse.ArgumentParser(description="Trace layer divergence between models")
    parser.add_argument('--v65-lib', help='Path to v6.5 .so file')
    parser.add_argument('--v66-lib', help='Path to v6.6 .so file')
    parser.add_argument('--v66-dir', help='Path to v6.6 model cache directory')
    parser.add_argument('--weights', help='Path to weights file')
    parser.add_argument('--prompt', default="Hello", help='Test prompt')
    parser.add_argument('--token', type=int, help='Single token ID to test')
    parser.add_argument('--compare-buffers', action='store_true',
                       help='Compare activation buffers between runs')

    args = parser.parse_args()

    # Default paths
    if not args.v66_dir:
        args.v66_dir = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
    else:
        args.v66_dir = Path(args.v66_dir)

    if not args.v66_lib:
        args.v66_lib = args.v66_dir / "model_v6_6.so"

    if not args.weights:
        args.weights = args.v66_dir / "model.bump"

    # Load config
    import json
    config_path = args.v66_dir / "lowered_decode.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
            config = data.get('config', {})

    print(f"Model config: embed_dim={config.get('embed_dim')}, layers={config.get('num_layers')}")

    # Simple token for testing
    tokens = [args.token] if args.token else [9707]  # "Hello" token typically

    print(f"\nTesting with tokens: {tokens}")
    print(f"Using v6.6 lib: {args.v66_lib}")

    if Path(args.v66_lib).exists():
        print("\nTracing v6.6...")
        try:
            v66_dumps = trace_single_model(
                str(args.v66_lib),
                str(args.weights),
                args.v66_dir,
                tokens,
                config
            )

            print("\nv6.6 buffer dumps:")
            for name, arr in v66_dumps.items():
                flat = arr.flatten()
                print(f"  {name}: shape={arr.shape}, "
                      f"range=[{flat.min():.6f}, {flat.max():.6f}], "
                      f"mean={flat.mean():.6f}")
                print(f"    first 10: {flat[:10]}")

                # Check for NaN/Inf
                if np.any(np.isnan(flat)):
                    print(f"    [WARNING] Contains NaN!")
                if np.any(np.isinf(flat)):
                    print(f"    [WARNING] Contains Inf!")

            # Analyze logits
            if 'logits' in v66_dumps:
                logits = v66_dumps['logits']
                top_k = 10
                top_indices = np.argsort(logits.flatten())[-top_k:][::-1]
                print(f"\n  Top {top_k} logits:")
                for idx in top_indices:
                    print(f"    token {idx}: {logits.flatten()[idx]:.4f}")

        except Exception as e:
            print(f"Error tracing v6.6: {e}")
            import traceback
            traceback.print_exc()

    # Compare with v6.5 if available
    if args.v65_lib and Path(args.v65_lib).exists():
        print("\nTracing v6.5...")
        # Similar code for v6.5...

    print("\n" + "=" * 60)
    print("DIVERGENCE ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
