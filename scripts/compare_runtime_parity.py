#!/usr/bin/env python3
"""
compare_runtime_parity.py - Full Model Tensor-by-Tensor Parity

Compares CK-Engine and llama.cpp tensor outputs during inference.
Works with ANY GGUF model - extracts tensor names from manifest.

Usage:
    python scripts/compare_runtime_parity.py \
        --gguf model.gguf \
        --ck-weights weights.bump \
        --manifest weights_manifest.json \
        --prompt "Hello" \
        --tolerance 1e-3

Prerequisites:
    1. Convert GGUF to bump format:
       python scripts/convert_gguf_to_bump.py \
           --gguf model.gguf \
           --output weights.bump \
           --config-out config.json \
           --manifest-out weights_manifest.json

    2. Run hacked llama.cpp to generate tensor dumps:
       LD_LIBRARY_PATH=llama.cpp/build/lib llama.cpp/build/bin/llama-cli \
           -m model.gguf -p "Hello" -n 1

    3. Run CK with --parity flag to generate dumps:
       python scripts/ck_run_v5.py run model_dir --parity
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'


def load_tensor(path: Path, dtype=np.float32) -> Optional[np.ndarray]:
    """Load a binary tensor file."""
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        data = f.read()
    return np.frombuffer(data, dtype=dtype)


def find_tensor_file(dump_dir: Path, patterns: List[str]) -> Optional[Path]:
    """Find a tensor file matching any of the given patterns."""
    for pattern in patterns:
        matches = list(dump_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


class TensorNameMapper:
    """Maps between CK tensor names and llama.cpp tensor names."""

    # Mapping from CK parity dump names to llama.cpp dump names
    # CK format: layer_{id}_{op}_tok{tok}.bin or layer_{id}_{op}.bin
    # llama.cpp format: {op}-{layer}.bin or {op}.bin

    CK_TO_LLAMA = {
        # RMSNorm
        "ln1_out": "attn_norm",
        "ln2_out": "ffn_norm",
        "final_ln": "result_norm",

        # Attention projections
        "q_proj": "Qcur",
        "k_proj": "Kcur",
        "v_proj": "Vcur",
        "q_rope": "Qcur",  # Post-RoPE Q
        "k_rope": "Kcur",  # Post-RoPE K

        # Attention output
        "attn_out": "attn_out",
        "attn_proj": "attn_out",

        # MLP
        "mlp_gate": "ffn_gate",
        "mlp_up": "ffn_up",
        "mlp_out": "ffn_out",
        "mlp": "ffn_down",

        # Residuals
        "residual1": "ffn_inp",
        "residual2": "l_out",

        # Final output
        "output": "result_output",
        "logits": "result_output",
    }

    def __init__(self, manifest_path: Optional[str] = None):
        """Initialize mapper with optional manifest for model config."""
        self.manifest = None
        self.n_layers = 24  # Default
        self.config = {}

        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            self.config = self.manifest.get("config", {})
            self.n_layers = self.config.get("num_hidden_layers",
                                            self.config.get("n_layer", 24))

    def get_llama_name(self, ck_name: str, layer: int) -> List[str]:
        """Get possible llama.cpp tensor names for a CK tensor."""
        patterns = []

        # Extract operation from CK name
        for ck_op, llama_op in self.CK_TO_LLAMA.items():
            if ck_op in ck_name:
                # llama.cpp format: {op}-{layer}.bin
                if layer >= 0:
                    patterns.append(f"{llama_op}-{layer}.bin")
                else:
                    patterns.append(f"{llama_op}.bin")
                break

        return patterns

    def get_ck_patterns(self, layer: int, op: str, tok: int = 0) -> List[str]:
        """Get possible CK tensor file patterns."""
        patterns = [
            f"layer_{layer}_{op}_tok{tok}.bin",
            f"layer_{layer}_{op}.bin",
            f"layer_{layer}_{op}*.bin",
        ]
        return patterns


class RuntimeParityTester:
    """Compare tensor outputs between CK and llama.cpp."""

    def __init__(self, llama_dump_dir: str, ck_dump_dir: str,
                 manifest_path: Optional[str] = None, tol: float = 1e-3):
        self.llama_dir = Path(llama_dump_dir)
        self.ck_dir = Path(ck_dump_dir)
        self.tol = tol
        self.results = []

        self.mapper = TensorNameMapper(manifest_path)
        self.n_layers = self.mapper.n_layers

    def compare_tensors(self, name: str, llama_data: np.ndarray,
                        ck_data: np.ndarray) -> Tuple[bool, float, str]:
        """Compare two tensors and return (passed, max_diff, info)."""
        if llama_data is None or ck_data is None:
            return False, float('inf'), "Missing data"

        # Truncate to smaller size if needed
        size = min(llama_data.size, ck_data.size)
        l_trim = llama_data[:size]
        c_trim = ck_data[:size]

        diff = np.abs(l_trim - c_trim)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))

        # Relative error
        max_val = np.max(np.abs(l_trim))
        rel_error = max_diff / (max_val + 1e-9)

        passed = max_diff < self.tol

        info = f"max_diff={max_diff:.2e} rel={rel_error:.2e} size={size}"

        # Check for NaN/Inf
        if np.isnan(l_trim).any():
            info += f" {RED}NaN in llama{RESET}"
            passed = False
        if np.isnan(c_trim).any():
            info += f" {RED}NaN in ck{RESET}"
            passed = False

        return passed, max_diff, info

    def check_layer(self, layer: int, tok: int = 0) -> Dict[str, bool]:
        """Check all tensor outputs for a single layer."""
        layer_results = {}

        # Checkpoints to compare (in data flow order)
        checkpoints = [
            ("ln1_out", "attn_norm"),
            ("q_proj", "Qcur"),
            ("k_proj", "Kcur"),
            ("v_proj", "Vcur"),
            ("attn_out", "attn_out"),
            ("ln2_out", "ffn_norm"),
            ("mlp_out", "ffn_out"),
        ]

        for ck_op, llama_op in checkpoints:
            # Find CK tensor
            ck_patterns = self.mapper.get_ck_patterns(layer, ck_op, tok)
            ck_path = find_tensor_file(self.ck_dir, ck_patterns)
            ck_data = load_tensor(ck_path) if ck_path else None

            # Find llama.cpp tensor
            llama_path = self.llama_dir / f"{llama_op}-{layer}.bin"
            llama_data = load_tensor(llama_path)

            name = f"layer_{layer}_{ck_op}"

            if ck_data is None:
                print(f"{YELLOW}[SKIP]{RESET} {name}: CK file missing")
                continue

            if llama_data is None:
                print(f"{YELLOW}[SKIP]{RESET} {name}: llama.cpp file missing")
                continue

            passed, max_diff, info = self.compare_tensors(name, llama_data, ck_data)
            self.results.append((name, layer, passed, max_diff, info))
            layer_results[ck_op] = passed

            status = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
            print(f"{status} {name}: {info}")

            if not passed:
                # Show samples
                print(f"       llama: {llama_data[:5]}")
                print(f"       ck:    {ck_data[:5]}")

        return layer_results

    def check_all_layers(self, tok: int = 0):
        """Check tensor parity for all layers."""
        print(f"\n{'='*70}")
        print(f"{BOLD}FULL MODEL PARITY TEST ({self.n_layers} layers){RESET}")
        print(f"{'='*70}")
        print(f"Tolerance: {self.tol}")
        print(f"llama.cpp dumps: {self.llama_dir}")
        print(f"CK dumps: {self.ck_dir}")
        print()

        divergence_layer = None

        for layer in range(self.n_layers):
            print(f"\n--- Layer {layer} ---")
            layer_results = self.check_layer(layer, tok)

            if layer_results and not all(layer_results.values()):
                if divergence_layer is None:
                    divergence_layer = layer
                    print(f"\n{RED}>>> DIVERGENCE POINT DETECTED at layer {layer} <<<{RESET}")

        # Check final output
        print(f"\n--- Final Output ---")
        self._check_final_output()

        # Summary
        self._print_summary(divergence_layer)

    def _check_final_output(self):
        """Check final norm and output tensors."""
        final_checks = [
            ("final_ln", "result_norm"),
            ("output", "result_output"),
        ]

        for ck_op, llama_op in final_checks:
            # CK final tensors might not have layer prefix
            ck_patterns = [f"{ck_op}.bin", f"final_{ck_op}.bin", f"*{ck_op}*.bin"]
            ck_path = find_tensor_file(self.ck_dir, ck_patterns)
            ck_data = load_tensor(ck_path) if ck_path else None

            llama_path = self.llama_dir / f"{llama_op}.bin"
            llama_data = load_tensor(llama_path)

            name = f"final_{ck_op}"

            if ck_data is None or llama_data is None:
                print(f"{YELLOW}[SKIP]{RESET} {name}: Missing file")
                continue

            passed, max_diff, info = self.compare_tensors(name, llama_data, ck_data)
            self.results.append((name, -1, passed, max_diff, info))

            status = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
            print(f"{status} {name}: {info}")

    def _print_summary(self, divergence_layer: Optional[int]):
        """Print test summary."""
        print(f"\n{'='*70}")
        print(f"{BOLD}PARITY TEST SUMMARY{RESET}")
        print(f"{'='*70}")

        if not self.results:
            print(f"{YELLOW}No tensors were compared. Check dump directories.{RESET}")
            return

        passed = sum(1 for r in self.results if r[2])
        total = len(self.results)

        # Per-layer summary
        layer_stats = {}
        for name, layer, ok, max_diff, _ in self.results:
            if layer not in layer_stats:
                layer_stats[layer] = {"passed": 0, "total": 0, "worst": 0}
            layer_stats[layer]["total"] += 1
            if ok:
                layer_stats[layer]["passed"] += 1
            layer_stats[layer]["worst"] = max(layer_stats[layer]["worst"], max_diff)

        print(f"\nPer-layer summary:")
        for layer in sorted(layer_stats.keys()):
            if layer < 0:
                label = "Final"
            else:
                label = f"Layer {layer:2d}"
            s = layer_stats[layer]
            status = "PASS" if s["passed"] == s["total"] else "FAIL"
            color = GREEN if s["passed"] == s["total"] else RED
            print(f"  {label}: [{color}{status}{RESET}] {s['passed']}/{s['total']} ops, "
                  f"worst_diff={s['worst']:.2e}")

        print(f"\nOverall: {passed}/{total} operations passed")

        if divergence_layer is not None:
            print(f"\n{RED}First divergence at layer {divergence_layer}{RESET}")
            print("Recommendation: Debug this layer's inputs/weights first")

        if passed == total:
            print(f"\n{GREEN}All tensors match within tolerance!{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Full model tensor-by-tensor parity test"
    )
    parser.add_argument("--llama-dump", default="llama_dump",
                        help="Directory with llama.cpp tensor dumps")
    parser.add_argument("--ck-dump", default="parity",
                        help="Directory with CK tensor dumps")
    parser.add_argument("--manifest", default=None,
                        help="Weights manifest JSON for model config")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Tolerance for comparison (default: 1e-3)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Test specific layer only")
    parser.add_argument("--tok", type=int, default=0,
                        help="Token index to compare")
    args = parser.parse_args()

    tester = RuntimeParityTester(
        args.llama_dump, args.ck_dump,
        manifest_path=args.manifest, tol=args.tol
    )

    if args.layer is not None:
        print(f"Testing layer {args.layer} only")
        tester.check_layer(args.layer, args.tok)
        tester._print_summary(None)
    else:
        tester.check_all_layers(args.tok)


if __name__ == "__main__":
    main()
