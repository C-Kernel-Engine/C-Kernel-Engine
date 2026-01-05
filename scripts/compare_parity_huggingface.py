#!/usr/bin/env python3
"""
Compare C-Kernel-Engine parity dumps against HuggingFace model outputs.

This script:
1. Loads a HuggingFace model (Qwen2, etc.)
2. Runs inference with hooks to capture intermediate buffers
3. Loads C-Kernel-Engine parity dumps
4. Compares layer by layer to find divergence

Usage:
    python scripts/compare_parity_huggingface.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --parity-dir /path/to/parity_dumps \
        --prompt "Hello"
"""

import argparse
import os
import sys
import struct
import numpy as np
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("ERROR: PyTorch and transformers are required")
    print("  pip install torch transformers")
    sys.exit(1)


class HuggingFaceCapture:
    """Capture intermediate activations from HuggingFace model."""

    def __init__(self, model):
        self.model = model
        self.captures = {}
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to capture intermediate activations."""

        # Embedding output
        def capture_embed(module, input, output):
            self.captures['embed_out'] = output.detach().cpu().numpy().flatten()

        self.hooks.append(
            self.model.model.embed_tokens.register_forward_hook(capture_embed)
        )

        # Per-layer captures
        for layer_id, layer in enumerate(self.model.model.layers):
            # Input to layer (after first RMSNorm - attention input)
            def make_ln1_hook(lid):
                def hook(module, input, output):
                    self.captures[f'layer_{lid}_ln1_out'] = output.detach().cpu().numpy().flatten()
                return hook
            self.hooks.append(
                layer.input_layernorm.register_forward_hook(make_ln1_hook(layer_id))
            )

            # Q/K/V projections
            def make_q_hook(lid):
                def hook(module, input, output):
                    self.captures[f'layer_{lid}_q_proj'] = output.detach().cpu().numpy().flatten()
                return hook
            self.hooks.append(
                layer.self_attn.q_proj.register_forward_hook(make_q_hook(layer_id))
            )

            def make_k_hook(lid):
                def hook(module, input, output):
                    self.captures[f'layer_{lid}_k_proj'] = output.detach().cpu().numpy().flatten()
                return hook
            self.hooks.append(
                layer.self_attn.k_proj.register_forward_hook(make_k_hook(layer_id))
            )

            def make_v_hook(lid):
                def hook(module, input, output):
                    self.captures[f'layer_{lid}_v_proj'] = output.detach().cpu().numpy().flatten()
                return hook
            self.hooks.append(
                layer.self_attn.v_proj.register_forward_hook(make_v_hook(layer_id))
            )

            # Post-attention LayerNorm (MLP input)
            def make_ln2_hook(lid):
                def hook(module, input, output):
                    self.captures[f'layer_{lid}_ln2_out'] = output.detach().cpu().numpy().flatten()
                return hook
            self.hooks.append(
                layer.post_attention_layernorm.register_forward_hook(make_ln2_hook(layer_id))
            )

            # MLP gate projection
            def make_gate_hook(lid):
                def hook(module, input, output):
                    self.captures[f'layer_{lid}_gate_proj'] = output.detach().cpu().numpy().flatten()
                return hook
            self.hooks.append(
                layer.mlp.gate_proj.register_forward_hook(make_gate_hook(layer_id))
            )

            # MLP up projection
            def make_up_hook(lid):
                def hook(module, input, output):
                    self.captures[f'layer_{lid}_up_proj'] = output.detach().cpu().numpy().flatten()
                return hook
            self.hooks.append(
                layer.mlp.up_proj.register_forward_hook(make_up_hook(layer_id))
            )

            # MLP down projection (final MLP output)
            def make_down_hook(lid):
                def hook(module, input, output):
                    self.captures[f'layer_{lid}_mlp'] = output.detach().cpu().numpy().flatten()
                return hook
            self.hooks.append(
                layer.mlp.down_proj.register_forward_hook(make_down_hook(layer_id))
            )

        # Final layer norm
        def capture_final_ln(module, input, output):
            self.captures['final_out'] = output.detach().cpu().numpy().flatten()
        self.hooks.append(
            self.model.model.norm.register_forward_hook(capture_final_ln)
        )

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear_captures(self):
        """Clear captured activations."""
        self.captures.clear()


def load_parity_file(path):
    """Load a parity dump file (binary float32)."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = f.read()
    count = len(data) // 4
    return np.frombuffer(data, dtype=np.float32, count=count)


def compare_buffers(name, hf_buf, ck_buf, tolerance=1e-4, max_show=10):
    """Compare two buffers and report differences."""
    if hf_buf is None:
        return {"status": "skip", "reason": "HuggingFace buffer missing"}
    if ck_buf is None:
        return {"status": "skip", "reason": "C-Kernel-Engine buffer missing"}

    # Handle size mismatch (due to alignment)
    min_len = min(len(hf_buf), len(ck_buf))
    hf_trim = hf_buf[:min_len]
    ck_trim = ck_buf[:min_len]

    # Compute differences
    abs_diff = np.abs(hf_trim - ck_trim)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    # Relative difference (avoid div by zero)
    hf_abs = np.abs(hf_trim) + 1e-10
    rel_diff = abs_diff / hf_abs
    max_rel_diff = float(np.max(rel_diff))

    # Find worst positions
    worst_idx = np.argsort(abs_diff)[-max_show:][::-1]
    worst_cases = []
    for idx in worst_idx:
        worst_cases.append({
            "index": int(idx),
            "hf": float(hf_trim[idx]),
            "ck": float(ck_trim[idx]),
            "diff": float(abs_diff[idx])
        })

    status = "pass" if max_diff < tolerance else "fail"

    return {
        "status": status,
        "hf_size": len(hf_buf),
        "ck_size": len(ck_buf),
        "compared": min_len,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "max_rel_diff": max_rel_diff,
        "worst_cases": worst_cases
    }


def run_comparison(args):
    """Run the full comparison."""
    print("=" * 70)
    print("C-Kernel-Engine vs HuggingFace Parity Comparison")
    print("=" * 70)
    print()

    # Check parity directory
    parity_dir = Path(args.parity_dir)
    if not parity_dir.exists():
        print(f"ERROR: Parity directory not found: {parity_dir}")
        return False

    # Find token index from parity files
    parity_files = list(parity_dir.glob("*_tok*.bin"))
    if not parity_files:
        print(f"ERROR: No parity files found in {parity_dir}")
        return False

    # Extract token indices
    token_indices = set()
    for f in parity_files:
        name = f.stem
        if "_tok" in name:
            tok_part = name.split("_tok")[-1]
            try:
                token_indices.add(int(tok_part))
            except ValueError:
                pass

    print(f"Found parity dumps for token indices: {sorted(token_indices)}")

    # Load model
    print(f"\nLoading HuggingFace model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # Use FP32 for best comparison
        device_map="cpu"  # CPU for reproducibility
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Get model config
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"  Layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")

    # Tokenize prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Tokens: {input_ids.tolist()[0]}")

    # Set up capture
    capture = HuggingFaceCapture(model)
    capture.register_hooks()

    # Run inference
    print("\nRunning HuggingFace inference...")
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits.cpu().numpy().flatten()
    capture.captures['logits'] = logits

    print(f"Captured {len(capture.captures)} buffers from HuggingFace")

    # Compare for each token index
    all_passed = True
    for tok_idx in sorted(token_indices):
        print(f"\n{'='*70}")
        print(f"Comparing Token Index: {tok_idx}")
        print(f"{'='*70}")

        results = []

        # Compare embedding
        ck_embed = load_parity_file(parity_dir / f"embed_out_tok{tok_idx}.bin")
        hf_embed = capture.captures.get('embed_out')
        result = compare_buffers("embed_out", hf_embed, ck_embed, args.tolerance)
        results.append(("embed_out", result))

        # Compare each layer
        for layer_id in range(num_layers):
            # Buffers to compare (name in HF, name in CK)
            layer_buffers = [
                (f"layer_{layer_id}_ln1_out", f"layer_{layer_id}_ln1_out"),
                (f"layer_{layer_id}_q_proj", f"layer_{layer_id}_q_proj"),
                (f"layer_{layer_id}_k_proj", f"layer_{layer_id}_k_proj"),
                (f"layer_{layer_id}_v_proj", f"layer_{layer_id}_v_proj"),
                (f"layer_{layer_id}_ln2_out", f"layer_{layer_id}_ln2_out"),
                (f"layer_{layer_id}_mlp", f"layer_{layer_id}_mlp"),
            ]

            for hf_name, ck_name in layer_buffers:
                hf_buf = capture.captures.get(hf_name)
                ck_buf = load_parity_file(parity_dir / f"{ck_name}_tok{tok_idx}.bin")
                result = compare_buffers(hf_name, hf_buf, ck_buf, args.tolerance)
                results.append((hf_name, result))

        # Compare final output
        ck_final = load_parity_file(parity_dir / f"final_out_tok{tok_idx}.bin")
        hf_final = capture.captures.get('final_out')
        result = compare_buffers("final_out", hf_final, ck_final, args.tolerance)
        results.append(("final_out", result))

        # Compare logits
        ck_logits = load_parity_file(parity_dir / f"logits_tok{tok_idx}.bin")
        hf_logits = capture.captures.get('logits')
        result = compare_buffers("logits", hf_logits, ck_logits, args.tolerance)
        results.append(("logits", result))

        # Print results
        passed = 0
        failed = 0
        skipped = 0

        for name, result in results:
            status = result['status']
            if status == 'pass':
                passed += 1
                if args.verbose:
                    print(f"  ✓ {name}: max_diff={result['max_diff']:.2e}")
            elif status == 'fail':
                failed += 1
                all_passed = False
                print(f"  ✗ {name}: max_diff={result['max_diff']:.2e} (FAIL)")
                if args.verbose:
                    print(f"      HF size: {result['hf_size']}, CK size: {result['ck_size']}")
                    print(f"      Worst cases:")
                    for wc in result['worst_cases'][:3]:
                        print(f"        [{wc['index']}]: HF={wc['hf']:.6f}, CK={wc['ck']:.6f}, diff={wc['diff']:.2e}")
            else:
                skipped += 1
                if args.verbose:
                    print(f"  - {name}: {result['reason']}")

        print(f"\nToken {tok_idx} Summary: {passed} passed, {failed} failed, {skipped} skipped")

    # Cleanup
    capture.remove_hooks()

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All comparisons PASSED")
    else:
        print("✗ Some comparisons FAILED - check output above")
    print("=" * 70)

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Compare C-Kernel-Engine parity dumps against HuggingFace"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="HuggingFace model name or path (e.g., Qwen/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--parity-dir", "-p",
        required=True,
        help="Directory containing parity dump files"
    )
    parser.add_argument(
        "--prompt",
        default="Hello",
        help="Input prompt to test (default: 'Hello')"
    )
    parser.add_argument(
        "--tolerance", "-t",
        type=float,
        default=1e-3,
        help="Maximum absolute difference to consider a match (default: 1e-3)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed comparison results"
    )

    args = parser.parse_args()

    success = run_comparison(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
