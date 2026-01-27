#!/usr/bin/env python3
"""
test_pipeline_v2.py - Comprehensive Test for Template v2 Pipeline

Tests the complete flow:
  Template v2 JSON → OpNodes → GraphIR → Validation

USAGE:
    python test_pipeline_v2.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add parent to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from parse_template_v2 import TemplateParser
from op_builders_v6_6 import OpContext, build_op_from_template, check_template_ops
from ir_types_v6_6 import GraphIR, LayerIR


def load_test_manifest():
    """Load test manifest with v2 template."""
    manifest_path = Path("/tmp/test_manifest_v2.json")
    if not manifest_path.exists():
        print(f"❌ Test manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        return json.load(f)


def validate_graph_ir(graph_ir: GraphIR, expected_ops: int, expected_layers: int):
    """Validate generated GraphIR."""
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    errors = []

    # Check layers
    if len(graph_ir.layers) != expected_layers:
        errors.append(f"Expected {expected_layers} layers, got {len(graph_ir.layers)}")

    # Count total ops
    total_ops = sum(len(layer.ops) for layer in graph_ir.layers)
    if total_ops != expected_ops:
        errors.append(f"Expected {expected_ops} ops, got {total_ops}")

    # Check op structure
    for i, layer in enumerate(graph_ir.layers):
        for j, op in enumerate(layer.ops):
            # Check required fields
            if not op.op:
                errors.append(f"Layer {i} op {j}: missing 'op' type")
            if not op.name:
                errors.append(f"Layer {i} op {j}: missing 'name'")
            if not op.inputs and op.op != "tokenizer":
                errors.append(f"Layer {i} op {j} ({op.op}): missing inputs")
            if not op.outputs:
                errors.append(f"Layer {i} op {j} ({op.op}): missing outputs")

    # Check tensor flow continuity
    for i, layer in enumerate(graph_ir.layers):
        if i == 0:
            continue  # Skip header

        # Check that layer input comes from previous layer
        first_op = layer.ops[0]
        if first_op.inputs:
            input_tensor = first_op.inputs[0]
            if not input_tensor.startswith(f"layer.{i-1}"):
                errors.append(f"Layer {i} first op input '{input_tensor}' doesn't come from layer {i-1}")

    if errors:
        print("\n❌ VALIDATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ VALIDATION PASSED")
        return True


def print_graph_ir_summary(graph_ir: GraphIR):
    """Print summary of generated GraphIR."""
    print("\n" + "="*60)
    print("GRAPH IR SUMMARY")
    print("="*60)

    print(f"\nModel: {graph_ir.model_name}")
    print(f"Layers: {len(graph_ir.layers)}")

    total_ops = sum(len(layer.ops) for layer in graph_ir.layers)
    print(f"Total ops: {total_ops}")

    print("\nLayer breakdown:")
    for i, layer in enumerate(graph_ir.layers):
        print(f"  Layer {i}: {len(layer.ops)} ops")

    # Op type distribution
    op_types = {}
    for layer in graph_ir.layers:
        for op in layer.ops:
            op_types[op.op] = op_types.get(op.op, 0) + 1

    print("\nOp type distribution:")
    for op_type, count in sorted(op_types.items()):
        print(f"  {op_type:20s}: {count}")

    # Weight references
    weights_used = set()
    for layer in graph_ir.layers:
        for op in layer.ops:
            weights_used.update(op.weights)

    print(f"\nUnique weights referenced: {len(weights_used)}")

    # Tensor flow
    print("\nTensor flow (first layer body):")
    if len(graph_ir.layers) > 1:
        layer_1 = graph_ir.layers[1]  # First body layer
        for op in layer_1.ops[:5]:  # Show first 5 ops
            inputs_str = ", ".join(op.inputs)
            outputs_str = ", ".join(op.outputs)
            print(f"  {op.name:25s}: {inputs_str:30s} → {outputs_str}")


def print_detailed_ops(graph_ir: GraphIR, max_ops: int = 10):
    """Print detailed op information."""
    print("\n" + "="*60)
    print(f"DETAILED OPS (first {max_ops})")
    print("="*60)

    count = 0
    for layer in graph_ir.layers:
        for op in layer.ops:
            if count >= max_ops:
                return

            print(f"\n[{count+1}] {op.name}")
            print(f"    Op type: {op.op}")
            print(f"    Inputs:  {op.inputs}")
            print(f"    Outputs: {op.outputs}")
            if op.weights:
                print(f"    Weights: {op.weights}")
            if op.scratch:
                print(f"    Scratch: {op.scratch}")
            if op.params:
                print(f"    Params:  {op.params}")

            count += 1


def check_kernel_availability(manifest: Dict, template_ops: set) -> bool:
    """Check if all required ops have kernel implementations."""
    # Import mapping to convert template ops to kernel ops
    from op_builders_hybrid_v6_6 import TEMPLATE_TO_KERNEL_MAP

    # Load actual kernel registry
    import json
    from pathlib import Path
    registry_path = Path(__file__).parent.parent / 'kernel_maps' / 'KERNEL_REGISTRY.json'

    with open(registry_path, 'r') as f:
        registry = json.load(f)

    # Get available kernel ops
    available_kernel_ops = set()
    for kernel in registry['kernels']:
        available_kernel_ops.add(kernel['op'])

    # Also include metadata-only ops
    available_kernel_ops.update(['tokenizer', 'weight_tying'])

    # Map template ops to kernel ops and check availability
    missing_kernels = []
    kernel_ops_needed = set()

    for t_op in template_ops:
        if t_op not in TEMPLATE_TO_KERNEL_MAP:
            missing_kernels.append(f"{t_op} (no mapping)")
            continue

        k_op = TEMPLATE_TO_KERNEL_MAP[t_op]
        kernel_ops_needed.add(k_op)

        if k_op not in available_kernel_ops:
            missing_kernels.append(f"{t_op} → {k_op} (no kernel)")

    if missing_kernels:
        print(f"\n  ❌ Missing kernel implementations:")
        for op in sorted(missing_kernels):
            print(f"     - {op}")
        print(f"\n  This is a HARD FAULT - GraphIR cannot be executed without kernels.")
        return False
    else:
        print(f"  ✅ All template ops map to available kernels")
        print(f"     Template ops: {len(template_ops)}")
        print(f"     Kernel ops needed: {len(kernel_ops_needed)}")
        print(f"     Kernel ops available: {len(available_kernel_ops)}")
        return True


def main():
    print("="*60)
    print("TEMPLATE V2 PIPELINE TEST")
    print("="*60)

    # Load manifest
    print("\n[1/7] Loading test manifest...")
    manifest = load_test_manifest()
    template = manifest["template"]
    config_dict = manifest["config"]

    print(f"  Template: {template['name']} (version {template['version']})")
    print(f"  Model: {config_dict.get('model', 'unknown')}")
    print(f"  Layers: {config_dict.get('num_layers', 0)}")

    # Parse template
    print("\n[2/7] Parsing template...")
    parser = TemplateParser(template, config_dict)
    exec_sequence = parser.build_execution_sequence()

    flat_ops = exec_sequence.get_flat_ops()
    print(f"  Generated {len(flat_ops)} op nodes")
    print(f"  Blocks: {len(exec_sequence.blocks)}")

    # Check template ops
    print("\n[3/7] Checking op builders...")
    unique_ops = parser.get_all_op_ids()
    print(f"  Unique op types: {len(unique_ops)}")
    print(f"  Op types: {', '.join(sorted(unique_ops))}")

    missing_builders = check_template_ops(unique_ops)
    if missing_builders:
        print(f"\n  ❌ Missing op builders for: {', '.join(missing_builders)}")
        return 1
    else:
        print(f"  ✅ All template ops have builders")

    # Check kernel availability (HARD REQUIREMENT)
    print("\n[4/7] Checking kernel availability...")
    has_kernels = check_kernel_availability(manifest, unique_ops)
    if not has_kernels:
        print(f"\n❌ PIPELINE TEST FAILED: Missing required kernels")
        return 1

    # Build GraphIR
    print("\n[5/7] Building GraphIR...")

    # Group ops by layer
    layers_dict = {}  # layer_index → list of ops

    ctx = OpContext(config_dict)

    for block in exec_sequence.blocks:
        for op_node in block.ops:
            # Reset context for new layers
            if op_node.phase == "body" and op_node.layer_index is not None:
                if op_node.layer_index not in layers_dict:
                    ctx.reset_layer()

            # Build op
            ir_op = build_op_from_template(op_node, ctx, manifest)

            # Group by layer
            if op_node.phase == "header":
                layer_idx = -1  # Header
            elif op_node.phase == "footer":
                layer_idx = -2  # Footer
            else:
                layer_idx = op_node.layer_index if op_node.layer_index is not None else 0

            if layer_idx not in layers_dict:
                layers_dict[layer_idx] = []
            layers_dict[layer_idx].append(ir_op)

    # Create LayerIR objects
    layers_ir = []

    # Header
    if -1 in layers_dict:
        header_layer = LayerIR(
            layer_id=-1,
            ops=layers_dict[-1]
        )
        layers_ir.append(header_layer)

    # Body layers
    num_layers = config_dict.get("num_layers", 0)
    for i in range(num_layers):
        if i in layers_dict:
            body_layer = LayerIR(
                layer_id=i,
                ops=layers_dict[i]
            )
            layers_ir.append(body_layer)

    # Footer
    if -2 in layers_dict:
        footer_layer = LayerIR(
            layer_id=-2,
            ops=layers_dict[-2]
        )
        layers_ir.append(footer_layer)

    # Create GraphIR
    graph_ir = GraphIR(
        model_name=config_dict.get("model", "unknown"),
        config=config_dict,
        template=template,
        layers=layers_ir
    )

    total_ops = sum(len(layer.ops) for layer in layers_ir)
    print(f"  Generated {total_ops} ops across {len(layers_ir)} layers")

    # Validate
    print("\n[6/7] Validating GraphIR...")
    # Expected: header (2 ops) + body (2 layers × 11 ops) + footer (2 ops) = 26 ops
    expected_ops = 2 + (num_layers * 11) + 2
    expected_layers = 1 + num_layers + 1  # header + body + footer

    is_valid = validate_graph_ir(graph_ir, expected_ops, expected_layers)

    # Print summary
    print("\n[7/7] Generating report...")
    print_graph_ir_summary(graph_ir)
    print_detailed_ops(graph_ir, max_ops=10)

    # Final status
    print("\n" + "="*60)
    if is_valid:
        print("✅ PIPELINE TEST PASSED")
        print("="*60)
        print("\nNext steps:")
        print("  1. Test with real Qwen2 manifest")
        print("  2. Implement lowering stage (kernel selection)")
        print("  3. Implement memory planning")
        print("  4. Implement code generation")
        return 0
    else:
        print("❌ PIPELINE TEST FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
