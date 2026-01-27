#!/usr/bin/env python3
"""
test_layer_by_layer.py - Layer-by-layer numerical validation for v6.6

Compares v6.6 output against llama.cpp (or PyTorch) at each checkpoint
to find the FIRST point of divergence.

USAGE:
    # First time: build reference
    python test_layer_by_layer.py --model ~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF --build-reference

    # Run comparison
    python test_layer_by_layer.py --model ~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF --token 25

    # Compare specific layer
    python test_layer_by_layer.py --model <path> --token 25 --layer 0

CHECKPOINTS:
    - tokenizer: token encoding
    - embedding: dense embedding lookup
    - layer_{i}_ln1: pre-attention RMSNorm
    - layer_{i}_q: Q projection
    - layer_{i}_k: K projection
    - layer_{i}_v: V projection
    - layer_{i}_q_rope: Q after RoPE
    - layer_{i}_k_rope: K after RoPE
    - layer_{i}_attn_out: attention output
    - layer_{i}_attn_residual: after attention residual
    - layer_{i}_ln2: pre-MLP RMSNorm
    - layer_{i}_mlp_out: MLP output
    - layer_{i}_output: layer output (after MLP residual)
    - final_ln: final layer norm
    - logits: output logits
"""

import argparse
import ctypes
import json
import numpy as np
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import struct


# ============================================================================
# CHECKPOINT DEFINITIONS
# ============================================================================

@dataclass
class Checkpoint:
    """A checkpoint in the inference pipeline."""
    name: str
    description: str
    shape_expr: str  # e.g., "[embed_dim]" or "[num_heads, head_dim]"
    tolerance: float = 1e-4  # max allowed absolute difference


# Define all checkpoints for a transformer layer
def get_checkpoints(num_layers: int, config: Dict) -> List[Checkpoint]:
    """Generate checkpoint list for model."""
    embed_dim = config.get("embed_dim", 896)
    num_heads = config.get("num_heads", 14)
    num_kv_heads = config.get("num_kv_heads", 2)
    head_dim = config.get("head_dim", 64)
    intermediate = config.get("intermediate_size", 4864)
    vocab_size = config.get("vocab_size", 151936)

    checkpoints = [
        Checkpoint("embedding", "Token embedding lookup", f"[{embed_dim}]"),
    ]

    for i in range(num_layers):
        layer_checks = [
            Checkpoint(f"layer_{i}_ln1", f"Layer {i} pre-attention RMSNorm", f"[{embed_dim}]"),
            Checkpoint(f"layer_{i}_q", f"Layer {i} Q projection", f"[{num_heads * head_dim}]"),
            Checkpoint(f"layer_{i}_k", f"Layer {i} K projection", f"[{num_kv_heads * head_dim}]"),
            Checkpoint(f"layer_{i}_v", f"Layer {i} V projection", f"[{num_kv_heads * head_dim}]"),
            Checkpoint(f"layer_{i}_q_rope", f"Layer {i} Q after RoPE", f"[{num_heads * head_dim}]"),
            Checkpoint(f"layer_{i}_k_rope", f"Layer {i} K after RoPE", f"[{num_kv_heads * head_dim}]"),
            Checkpoint(f"layer_{i}_attn_out", f"Layer {i} attention output", f"[{embed_dim}]"),
            Checkpoint(f"layer_{i}_attn_residual", f"Layer {i} attention + residual", f"[{embed_dim}]"),
            Checkpoint(f"layer_{i}_ln2", f"Layer {i} pre-MLP RMSNorm", f"[{embed_dim}]"),
            Checkpoint(f"layer_{i}_mlp_out", f"Layer {i} MLP output", f"[{embed_dim}]"),
            Checkpoint(f"layer_{i}_output", f"Layer {i} final output", f"[{embed_dim}]"),
        ]
        checkpoints.extend(layer_checks)

    checkpoints.extend([
        Checkpoint("final_ln", "Final layer norm", f"[{embed_dim}]"),
        Checkpoint("logits", "Output logits", f"[{vocab_size}]", tolerance=1e-3),
    ])

    return checkpoints


# ============================================================================
# REFERENCE EXTRACTORS
# ============================================================================

class LlamaCppReference:
    """Extract reference values from llama.cpp with debug patch."""

    def __init__(self, model_path: Path, llamacpp_path: Optional[Path] = None):
        self.model_path = model_path
        self.llamacpp_path = llamacpp_path or Path.home() / "llama.cpp"
        self.debug_binary = self.llamacpp_path / "llama-cli"

    def is_available(self) -> bool:
        """Check if debug build is available."""
        return self.debug_binary.exists()

    def build_debug(self) -> bool:
        """Build llama.cpp with debug hooks."""
        print("Building llama.cpp with v6.6 debug hooks...")

        # Apply patch
        patch_path = Path(__file__).parent.parent / "patches" / "llamacpp_debug_v6_6.patch"
        if patch_path.exists():
            try:
                subprocess.run(
                    ["git", "apply", str(patch_path)],
                    cwd=self.llamacpp_path,
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError:
                print("  Patch may already be applied, continuing...")

        # Build with debug flag
        try:
            subprocess.run(
                ["make", "clean"],
                cwd=self.llamacpp_path,
                check=True,
                capture_output=True
            )
            subprocess.run(
                ["make", "-j", "LLAMA_V66_DEBUG=1"],
                cwd=self.llamacpp_path,
                check=True,
                capture_output=True
            )
            print("  Build successful!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  Build failed: {e}")
            return False

    def run(self, token: int) -> Dict[str, np.ndarray]:
        """Run inference and extract checkpoint values."""
        if not self.is_available():
            raise RuntimeError("llama.cpp debug build not available. Run with --build-reference first.")

        # Find GGUF model
        gguf_path = None
        for f in self.model_path.iterdir():
            if f.suffix == ".gguf":
                gguf_path = f
                break

        if not gguf_path:
            # Try to find original GGUF
            gguf_path = self.model_path / "model.gguf"
            if not gguf_path.exists():
                raise RuntimeError(f"No GGUF model found in {self.model_path}")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            debug_file = f.name

        try:
            env = os.environ.copy()
            env["LLAMA_DEBUG_FILE"] = debug_file

            # Run llama.cpp with single token
            result = subprocess.run(
                [
                    str(self.debug_binary),
                    "-m", str(gguf_path),
                    "-p", f"<|token_{token}|>",  # Placeholder, we'll inject token directly
                    "-n", "1",
                    "--temp", "0",
                ],
                cwd=self.llamacpp_path,
                env=env,
                capture_output=True,
                timeout=60
            )

            # Parse debug output
            if os.path.exists(debug_file):
                with open(debug_file) as f:
                    data = json.load(f)
                return self._parse_debug_output(data)
            else:
                print(f"  Warning: Debug file not created")
                return {}

        finally:
            if os.path.exists(debug_file):
                os.unlink(debug_file)

    def _parse_debug_output(self, data: Dict) -> Dict[str, np.ndarray]:
        """Parse debug JSON into numpy arrays."""
        result = {}
        for name, info in data.items():
            if isinstance(info, dict) and "first_32" in info:
                # Reconstruct array from first/last values
                arr = np.array(info["first_32"], dtype=np.float32)
                result[name] = arr
        return result


class PyTorchReference:
    """Extract reference values from PyTorch/HuggingFace."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def is_available(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            return True
        except ImportError:
            return False

    def load(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        """Load model from HuggingFace."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading PyTorch model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use FP32 for comparison
            trust_remote_code=True
        )
        self.model.eval()

    def run(self, token: int) -> Dict[str, np.ndarray]:
        """Run inference and extract checkpoint values."""
        import torch

        if self.model is None:
            self.load()

        results = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                results[name] = output.detach().cpu().numpy().flatten()
            return hook

        # Register hooks for each layer
        for i, layer in enumerate(self.model.model.layers):
            hooks.append(layer.input_layernorm.register_forward_hook(make_hook(f"layer_{i}_ln1")))
            hooks.append(layer.self_attn.q_proj.register_forward_hook(make_hook(f"layer_{i}_q")))
            hooks.append(layer.self_attn.k_proj.register_forward_hook(make_hook(f"layer_{i}_k")))
            hooks.append(layer.self_attn.v_proj.register_forward_hook(make_hook(f"layer_{i}_v")))
            hooks.append(layer.post_attention_layernorm.register_forward_hook(make_hook(f"layer_{i}_ln2")))
            hooks.append(layer.mlp.register_forward_hook(make_hook(f"layer_{i}_mlp_out")))

        # Hook for embedding
        hooks.append(self.model.model.embed_tokens.register_forward_hook(make_hook("embedding")))

        # Hook for final layer norm
        hooks.append(self.model.model.norm.register_forward_hook(make_hook("final_ln")))

        try:
            # Run forward pass
            input_ids = torch.tensor([[token]], dtype=torch.long)
            with torch.no_grad():
                outputs = self.model(input_ids)
                results["logits"] = outputs.logits[0, 0].cpu().numpy()

        finally:
            # Remove hooks
            for h in hooks:
                h.remove()

        return results


# ============================================================================
# V6.6 EXTRACTOR (with debug hooks)
# ============================================================================

class V66Extractor:
    """Extract checkpoint values from v6.6 inference."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.lib = None
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load model config."""
        config_path = self.model_path / "lowered_decode.json"
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
                return data.get("config", {})
        return {}

    def load(self):
        """Load the v6.6 library."""
        # Load kernel engine first
        engine_path = self.model_path / "libckernel_engine.so"
        if engine_path.exists():
            ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

        lib_path = self.model_path / "libmodel.so"
        if not lib_path.exists():
            lib_path = self.model_path / "ck-kernel-inference.so"

        self.lib = ctypes.CDLL(str(lib_path))

        # Setup function signatures
        self.lib.ck_model_init.argtypes = [ctypes.c_char_p]
        self.lib.ck_model_init.restype = ctypes.c_int
        self.lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        self.lib.ck_model_decode.restype = ctypes.c_int
        self.lib.ck_model_get_logits.argtypes = []
        self.lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.ck_model_free.argtypes = []
        self.lib.ck_model_free.restype = None

        # Check for debug functions
        try:
            self.lib.ck_debug_get_activation.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            self.lib.ck_debug_get_activation.restype = ctypes.c_int
            self.has_debug = True
        except:
            self.has_debug = False

    def run(self, token: int) -> Dict[str, np.ndarray]:
        """Run inference and extract checkpoint values."""
        if self.lib is None:
            self.load()

        weights_path = self.model_path / "weights.bump"
        result = self.lib.ck_model_init(str(weights_path).encode())
        if result != 0:
            raise RuntimeError(f"v6.6 init failed: {result}")

        vocab_size = self.config.get("vocab_size", 151936)
        output = (ctypes.c_float * vocab_size)()

        result = self.lib.ck_model_decode(token, output)
        if result != 0:
            print(f"  Warning: decode returned {result}")

        results = {}

        # Get logits
        logits_ptr = self.lib.ck_model_get_logits()
        if logits_ptr:
            results["logits"] = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,)).copy()

        # Get intermediate activations if debug functions available
        if self.has_debug:
            embed_dim = self.config.get("embed_dim", 896)
            num_layers = self.config.get("num_layers", 24)

            checkpoints_to_extract = [
                ("embedding", embed_dim),
                ("final_ln", embed_dim),
            ]
            for i in range(num_layers):
                checkpoints_to_extract.extend([
                    (f"layer_{i}_ln1", embed_dim),
                    (f"layer_{i}_q", self.config.get("num_heads", 14) * self.config.get("head_dim", 64)),
                    (f"layer_{i}_k", self.config.get("num_kv_heads", 2) * self.config.get("head_dim", 64)),
                    (f"layer_{i}_v", self.config.get("num_kv_heads", 2) * self.config.get("head_dim", 64)),
                    (f"layer_{i}_attn_out", embed_dim),
                    (f"layer_{i}_mlp_out", embed_dim),
                    (f"layer_{i}_output", embed_dim),
                ])

            for name, size in checkpoints_to_extract:
                buf = (ctypes.c_float * size)()
                ret = self.lib.ck_debug_get_activation(name.encode(), buf, size)
                if ret == 0:
                    results[name] = np.array(buf[:], dtype=np.float32)

        self.lib.ck_model_free()
        return results


# ============================================================================
# COMPARISON ENGINE
# ============================================================================

@dataclass
class ComparisonResult:
    """Result of comparing a checkpoint."""
    checkpoint: str
    passed: bool
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    first_diff_idx: int = -1
    expected_sample: List[float] = field(default_factory=list)
    actual_sample: List[float] = field(default_factory=list)
    message: str = ""


def compare_arrays(name: str, expected: np.ndarray, actual: np.ndarray,
                   tolerance: float = 1e-4) -> ComparisonResult:
    """Compare two arrays and return detailed result."""
    result = ComparisonResult(checkpoint=name, passed=True)

    # Check shapes
    if expected.shape != actual.shape:
        result.passed = False
        result.message = f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
        return result

    # Compute differences
    abs_diff = np.abs(expected - actual)
    result.max_abs_diff = float(np.max(abs_diff))

    # Relative difference (avoid div by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = abs_diff / (np.abs(expected) + 1e-10)
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)
    result.max_rel_diff = float(np.max(rel_diff))

    # Find first significant difference
    diff_mask = abs_diff > tolerance
    if np.any(diff_mask):
        result.first_diff_idx = int(np.argmax(diff_mask))
        result.passed = False

        # Sample values around the difference
        idx = result.first_diff_idx
        start = max(0, idx - 2)
        end = min(len(expected), idx + 3)
        result.expected_sample = expected[start:end].tolist()
        result.actual_sample = actual[start:end].tolist()

        result.message = f"Values differ at index {idx}: expected {expected[idx]:.6f}, got {actual[idx]:.6f}"
    else:
        result.message = f"PASS (max_diff={result.max_abs_diff:.2e})"

    return result


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

class LayerByLayerTest:
    """Main test runner for layer-by-layer comparison."""

    def __init__(self, model_path: Path, reference: str = "llamacpp"):
        self.model_path = Path(model_path)
        self.reference_type = reference

        # Load config
        config_path = self.model_path / "lowered_decode.json"
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
                self.config = data.get("config", {})
        else:
            self.config = {}

        self.num_layers = self.config.get("num_layers", 24)

        # Setup extractors
        self.v66 = V66Extractor(self.model_path)

        if reference == "llamacpp":
            self.ref = LlamaCppReference(self.model_path)
        elif reference == "pytorch":
            self.ref = PyTorchReference(self.model_path)
        else:
            raise ValueError(f"Unknown reference: {reference}")

    def run(self, token: int, stop_on_fail: bool = True,
            start_layer: int = 0, end_layer: Optional[int] = None) -> List[ComparisonResult]:
        """Run layer-by-layer comparison."""

        print(f"\n{'='*70}")
        print(f"LAYER-BY-LAYER TEST: token={token}")
        print(f"{'='*70}\n")

        # Get reference values
        print("Running reference...")
        try:
            ref_values = self.ref.run(token)
            print(f"  Got {len(ref_values)} checkpoints from reference")
        except Exception as e:
            print(f"  Reference failed: {e}")
            ref_values = {}

        # Get v6.6 values
        print("Running v6.6...")
        try:
            v66_values = self.v66.run(token)
            print(f"  Got {len(v66_values)} checkpoints from v6.6")
        except Exception as e:
            print(f"  v6.6 failed: {e}")
            v66_values = {}

        # Compare at each checkpoint
        results = []
        checkpoints = get_checkpoints(self.num_layers, self.config)

        if end_layer is None:
            end_layer = self.num_layers

        print(f"\n{'='*70}")
        print("COMPARISON RESULTS")
        print(f"{'='*70}\n")

        first_failure = None

        for cp in checkpoints:
            # Filter by layer range
            if cp.name.startswith("layer_"):
                layer_num = int(cp.name.split("_")[1])
                if layer_num < start_layer or layer_num >= end_layer:
                    continue

            if cp.name not in ref_values:
                print(f"[SKIP] {cp.name}: not in reference")
                continue

            if cp.name not in v66_values:
                print(f"[SKIP] {cp.name}: not in v6.6")
                continue

            result = compare_arrays(
                cp.name,
                ref_values[cp.name],
                v66_values[cp.name],
                cp.tolerance
            )
            results.append(result)

            if result.passed:
                print(f"[PASS] {cp.name}: {result.message}")
            else:
                print(f"[FAIL] {cp.name}: {result.message}")
                print(f"       Expected: {result.expected_sample}")
                print(f"       Actual:   {result.actual_sample}")

                if first_failure is None:
                    first_failure = result

                if stop_on_fail:
                    print(f"\n{'='*70}")
                    print("STOPPING AT FIRST FAILURE")
                    print(f"{'='*70}")
                    break

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")

        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")

        if first_failure:
            print(f"\n  FIRST FAILURE: {first_failure.checkpoint}")
            print(f"  This is where to start debugging.")
            print(f"\n  Suggested fixes:")
            self._suggest_fix(first_failure)

        return results

    def _suggest_fix(self, failure: ComparisonResult):
        """Suggest where to look for the bug."""
        name = failure.checkpoint

        if name == "embedding":
            print("    - Check build_ir_v6_6.py: embedding operation binding")
            print("    - Check kernel_maps/embedding_forward_q8_0.json")
            print("    - Verify token_emb weight offset in layout")

        elif "_ln1" in name or "_ln2" in name or name == "final_ln":
            layer = name.split("_")[1] if "_" in name else "final"
            print(f"    - Check kernel_maps/rmsnorm_forward.json")
            print(f"    - Verify gamma weight offset for layer {layer}")
            print(f"    - Check if epsilon value matches (1e-5 vs 1e-6)")

        elif "_q" in name or "_k" in name or "_v" in name:
            layer = name.split("_")[1]
            proj = name.split("_")[2]  # q, k, or v
            print(f"    - Check kernel_maps/gemv_*.json for {proj.upper()} projection")
            print(f"    - Verify w{proj} weight offset for layer {layer}")
            print(f"    - Check if bias is correctly applied (or NULL)")

        elif "_rope" in name:
            print("    - Check kernel_maps/rope_forward_qk.json")
            print("    - Verify rope_cos/rope_sin cache offsets")
            print("    - Check theta value (10000 vs 1000000)")

        elif "_attn" in name:
            print("    - Check kernel_maps/attention_forward_*.json")
            print("    - Verify KV cache offsets and stride")
            print("    - Check head_dim alignment")

        elif "_mlp" in name:
            print("    - Check kernel_maps/gemv_*.json for MLP")
            print("    - Check kernel_maps/swiglu_forward.json")
            print("    - Verify w1, w2 weight offsets")
            print("    - Check intermediate_size alignment")

        elif name == "logits":
            print("    - Check lm_head / output projection")
            print("    - May be tied weights with embedding")
            print("    - Check final_ln output first")


def main():
    parser = argparse.ArgumentParser(description="Layer-by-layer numerical validation")
    parser.add_argument("--model", type=Path, required=True, help="Model cache path")
    parser.add_argument("--reference", choices=["llamacpp", "pytorch"], default="pytorch",
                        help="Reference implementation to compare against")
    parser.add_argument("--token", type=int, default=25, help="Token ID to test")
    parser.add_argument("--layer", type=int, help="Test specific layer only")
    parser.add_argument("--all-layers", action="store_true", help="Don't stop on first failure")
    parser.add_argument("--build-reference", action="store_true",
                        help="Build llama.cpp with debug hooks")

    args = parser.parse_args()

    # Set library path
    os.environ["LD_LIBRARY_PATH"] = str(args.model) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    test = LayerByLayerTest(args.model, args.reference)

    if args.build_reference and args.reference == "llamacpp":
        test.ref.build_debug()
        return

    start_layer = args.layer if args.layer is not None else 0
    end_layer = args.layer + 1 if args.layer is not None else None

    results = test.run(
        token=args.token,
        stop_on_fail=not args.all_layers,
        start_layer=start_layer,
        end_layer=end_layer
    )

    # Exit with error if any failures
    if any(not r.passed for r in results):
        exit(1)


if __name__ == "__main__":
    main()
