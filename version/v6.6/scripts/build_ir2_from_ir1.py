#!/usr/bin/env python3
"""
build_ir2_from_ir1.py - Add tensor metadata to IR1

IR2 adds tensor information to the kernel sequence:
- Input/output shapes
- Data types
- Buffer names
- Weight references

This allows the next stage (fusion) to understand data dependencies.

IR2 Format:
{
    "format": "ir2-kernel-ops",
    "version": 1,
    "mode": "decode" | "prefill",
    "config": {...model config...},
    "ops": [
        {
            "kernel": "rmsnorm_forward",
            "op": "rmsnorm",
            "inputs": [{"name": "layer.0.attn_input", "shape": "[E]", "dtype": "fp32"}],
            "outputs": [{"name": "layer.0.attn_norm_out", "shape": "[E]", "dtype": "fp32"}],
            "weights": [{"name": "layer.0.ln1_weight", "dtype": "fp32"}],
            "scratch": [],
            "params": {"embed_dim": 896}
        },
        ...
    ]
}
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Script directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_ir1(ir1_path: Path) -> Dict:
    """Load IR1 JSON file."""
    with open(ir1_path, 'r') as f:
        return json.load(f)


def load_manifest(manifest_path: Path) -> Dict:
    """Load weights manifest."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def load_kernel_registry() -> Dict:
    """Load kernel registry."""
    registry_path = PROJECT_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"
    with open(registry_path, 'r') as f:
        return json.load(f)


def find_kernel_metadata(registry: Dict, kernel_id: str) -> Optional[Dict]:
    """Find kernel metadata from registry by ID."""
    for kernel in registry["kernels"]:
        if kernel["id"] == kernel_id:
            return kernel
    return None


def infer_tensor_shape(
    op_type: str,
    io_type: str,  # "input" or "output"
    mode: str,
    config: Dict
) -> str:
    """
    Infer tensor shape based on operation type and mode.

    Shapes use symbolic dimensions:
    - E: embed_dim
    - H: num_heads
    - D: head_dim
    - V: vocab_size
    - I: intermediate_size
    - T: seq_len (1 for decode, variable for prefill)
    - B: batch_size (always 1 for now)
    """
    embed_dim = config.get("embed_dim", 0)

    # Decode mode: single token processing (shapes are vectors)
    if mode == "decode":
        if op_type in ["rmsnorm", "residual_add"]:
            return "[E]"  # Vector
        elif op_type in ["gemv", "qkv_projection"]:
            if io_type == "input":
                return "[E]"
            else:
                return "[E]"  # Output projection
        elif op_type == "attention":
            return "[E]"
        elif op_type == "swiglu":
            return "[I]"  # Intermediate dimension
        else:
            return "[E]"  # Default

    # Prefill mode: multi-token processing (shapes are matrices)
    else:
        if op_type in ["rmsnorm", "residual_add"]:
            return "[T,E]"  # Matrix (T tokens)
        elif op_type in ["gemm", "qkv_projection"]:
            if io_type == "input":
                return "[T,E]"
            else:
                return "[T,E]"
        elif op_type == "attention":
            return "[T,E]"
        elif op_type == "swiglu":
            return "[T,I]"
        else:
            return "[T,E]"  # Default


def build_ir2_from_ir1(
    ir1: Dict,
    manifest: Dict,
    registry: Dict
) -> Dict:
    """
    Transform IR1 (kernel sequence) to IR2 (ops with tensor metadata).

    Args:
        ir1: IR1 dict with "kernels" list
        manifest: Weights manifest with config and quant_summary
        registry: Kernel registry for metadata

    Returns:
        IR2 dict with "ops" list containing tensor metadata
    """
    mode = ir1.get("mode", "decode")
    config = manifest.get("config", {})
    quant_summary = manifest.get("quant_summary", {})
    template = manifest.get("template", {})

    # Extract template ops sequence
    block_name = template["sequence"][0]
    block = template["block_types"][block_name]
    template_ops = block["body"]["ops"] if isinstance(block["body"], dict) else block["body"]

    num_layers = config.get("num_layers", 0)
    kernels = ir1.get("kernels", [])

    ops = []
    kernel_idx = 0

    # Process each layer
    for layer_idx in range(num_layers):
        layer_key = f"layer.{layer_idx}"

        # Track previous output for data flow
        prev_output = f"{layer_key}.input"

        # Process each template op in sequence
        for template_op in template_ops:
            if kernel_idx >= len(kernels):
                break

            kernel_id = kernels[kernel_idx]
            kernel_meta = find_kernel_metadata(registry, kernel_id)

            if not kernel_meta:
                print(f"Warning: No metadata for kernel {kernel_id}")
                kernel_idx += 1
                continue

            op_type = kernel_meta.get("op", "unknown")

            # Build op with tensor metadata
            op = {
                "kernel": kernel_id,
                "op": op_type,
                "template_op": template_op,
                "layer": layer_idx,
            }

            # Infer input/output shapes
            input_shape = infer_tensor_shape(op_type, "input", mode, config)
            output_shape = infer_tensor_shape(op_type, "output", mode, config)

            # Build input tensors
            op["inputs"] = [{
                "name": prev_output,
                "shape": input_shape,
                "dtype": "fp32"  # Activations are fp32
            }]

            # Build output tensors
            output_name = f"{layer_key}.{template_op}_out"
            op["outputs"] = [{
                "name": output_name,
                "shape": output_shape,
                "dtype": "fp32"
            }]

            # Add weight references
            op["weights"] = []
            if "weight" in kernel_meta.get("quant", {}):
                weight_quant = kernel_meta["quant"]["weight"]
                if weight_quant != "none":
                    # Infer weight name from template op
                    if "qkv" in template_op:
                        op["weights"] = [
                            {"name": f"{layer_key}.wq", "dtype": weight_quant},
                            {"name": f"{layer_key}.wk", "dtype": weight_quant},
                            {"name": f"{layer_key}.wv", "dtype": weight_quant},
                        ]
                    elif "out_proj" in template_op:
                        op["weights"] = [{"name": f"{layer_key}.wo", "dtype": weight_quant}]
                    elif "mlp_down" in template_op:
                        op["weights"] = [{"name": f"{layer_key}.w2", "dtype": weight_quant}]
                    elif "mlp_gate_up" in template_op:
                        op["weights"] = [{"name": f"{layer_key}.w1", "dtype": weight_quant}]
                    elif "rmsnorm" in template_op:
                        op["weights"] = [{"name": f"{layer_key}.ln_weight", "dtype": "fp32"}]

            # Add params from config
            op["params"] = {
                "embed_dim": config.get("embed_dim", 896),
                "num_heads": config.get("num_heads", 14),
                "head_dim": config.get("head_dim", 64),
                "intermediate_size": config.get("intermediate_size", 4864),
            }

            ops.append(op)
            prev_output = output_name
            kernel_idx += 1

    # Build IR2
    ir2 = {
        "format": "ir2-kernel-ops",
        "version": 1,
        "mode": mode,
        "config": config,
        "ops": ops,
    }

    return ir2


def main(args: List[str]) -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build IR2: Add tensor metadata to IR1"
    )
    parser.add_argument("--ir1", type=Path, required=True, help="Path to IR1 JSON file")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to weights manifest")
    parser.add_argument("--output", type=Path, help="Output IR2 JSON file")

    parsed_args = parser.parse_args(args)

    # Load inputs
    print(f"Loading IR1: {parsed_args.ir1}")
    ir1 = load_ir1(parsed_args.ir1)

    print(f"Loading manifest: {parsed_args.manifest}")
    manifest = load_manifest(parsed_args.manifest)

    print("Loading kernel registry...")
    registry = load_kernel_registry()

    # Build IR2
    print("\nBuilding IR2...")
    ir2 = build_ir2_from_ir1(ir1, manifest, registry)

    print(f"✓ Generated {len(ir2['ops'])} ops with tensor metadata")

    # Output
    if parsed_args.output:
        with open(parsed_args.output, 'w') as f:
            json.dump(ir2, f, indent=2)
        print(f"\n✓ Wrote IR2 to: {parsed_args.output}")
    else:
        print(f"\nIR2 (first 3 ops):")
        for i, op in enumerate(ir2["ops"][:3]):
            print(f"\n  Op {i}:")
            print(f"    kernel: {op['kernel']}")
            print(f"    inputs: {[inp['name'] for inp in op['inputs']]}")
            print(f"    outputs: {[out['name'] for out in op['outputs']]}")
            print(f"    weights: {[w['name'] for w in op.get('weights', [])]}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
