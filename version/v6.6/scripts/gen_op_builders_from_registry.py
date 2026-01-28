#!/usr/bin/env python3
"""
=============================================================================
EXPERIMENTAL/FUTURE - NOT USED BY CURRENT v6.6 PIPELINE
=============================================================================
This file generates op_builders_auto.py for future IR2 work.
It is NOT called by ck_run_v6_6.py or the current build pipeline.

Related files: op_builders_auto.py, op_builders_v6_6.py, parse_template_v2.py
Current pipeline uses: build_ir_v6_6.py with kernel maps directly.
=============================================================================

gen_op_builders_from_registry.py - Auto-generate op builders from kernel registry

This script analyzes the kernel registry and generates op builder functions that
can create GraphIR Op objects from template OpNodes.

USAGE:
    python gen_op_builders_from_registry.py --output=op_builders_auto.py
    python gen_op_builders_from_registry.py --dry-run  # Show what would be generated

BENEFITS:
    - Perfect alignment between kernels and ops
    - Automatic updates when kernels change
    - No manual maintenance of op builders
    - Validates that all kernels have required metadata
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Any

# Add parent to path
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def load_kernel_registry(registry_path: Path) -> Dict:
    """Load kernel registry JSON."""
    with open(registry_path, 'r') as f:
        return json.load(f)


def analyze_kernel_metadata(registry: Dict) -> Dict[str, Any]:
    """
    Analyze kernel registry to extract op metadata.

    Returns:
        Dict mapping op_type -> {
            'inputs': [input_names],
            'outputs': [output_names],
            'scratch': [scratch_names],
            'weights': [inferred from inputs],
            'params': [inferred from dims]
        }
    """
    op_metadata = {}

    for kernel in registry['kernels']:
        op_type = kernel['op']

        if op_type not in op_metadata:
            # First kernel of this type - extract metadata
            metadata = {
                'inputs': [inp['name'] for inp in kernel.get('inputs', [])],
                'outputs': [out['name'] for out in kernel.get('outputs', [])],
                'scratch': [scr['name'] for scr in kernel.get('scratch', [])],
                'dims': kernel.get('dims', []),
                'kernel_id': kernel['id']
            }

            # Infer weights from inputs (typically weight tensors)
            weights = []
            for inp in kernel.get('inputs', []):
                # Weight tensors typically have specific naming patterns
                if any(w in inp['name'].lower() for w in ['weight', 'wq', 'wk', 'wv', 'wo', 'w1', 'w2']):
                    weights.append(inp['name'])
                # Or are 2D/3D tensors (not activations)
                elif len(inp.get('shape', [])) >= 2:
                    desc = inp.get('desc', '').lower()
                    if 'weight' in desc or 'matrix' in desc:
                        weights.append(inp['name'])

            metadata['weights'] = weights

            # Infer params from dims
            params = []
            for dim in kernel.get('dims', []):
                # Common dimension names map to config params
                dim_to_param = {
                    'H': 'num_heads',
                    'KV': 'num_kv_heads',
                    'D': 'head_dim',
                    'E': 'embed_dim',
                    'V': 'vocab_size',
                    'I': 'intermediate_size',
                    'L': 'num_layers'
                }
                if dim in dim_to_param:
                    params.append(dim_to_param[dim])

            metadata['params'] = params
            op_metadata[op_type] = metadata

    return op_metadata


def generate_op_builder_function(op_type: str, metadata: Dict) -> str:
    """Generate Python code for an op builder function."""

    # Function name
    func_name = f"build_{op_type.replace('-', '_')}_op"

    # Generate input handling
    if metadata['inputs']:
        input_setup = []
        for i, inp in enumerate(metadata['inputs']):
            if i == 0:
                input_setup.append(f"    input_name = ctx.prev_output or make_tensor_name(op_node, 'input')")
            else:
                input_setup.append(f"    {inp}_name = make_tensor_name(op_node, '{inp}')")
        input_list = ", ".join([
            "input_name" if i == 0 else f"{inp}_name"
            for i, inp in enumerate(metadata['inputs'])
        ])
    else:
        input_setup = ["    # No inputs"]
        input_list = ""

    # Generate output handling
    if len(metadata['outputs']) == 1:
        output_setup = [
            f"    output_name = make_tensor_name(op_node, '{metadata['outputs'][0]}')"
        ]
        output_list = "[output_name]"
    else:
        output_setup = []
        for out in metadata['outputs']:
            output_setup.append(f"    {out}_name = make_tensor_name(op_node, '{out}')")
        output_list = "[" + ", ".join([f"{out}_name" for out in metadata['outputs']]) + "]"

    # Generate weight handling
    if metadata['weights']:
        weight_setup = []
        for w in metadata['weights']:
            weight_setup.append(f"    {w}_weight = make_weight_name(op_node, '{w}')")
        weight_list = "[" + ", ".join([f"{w}_weight" for w in metadata['weights']]) + "]"
    else:
        weight_setup = ["    # No weights"]
        weight_list = "[]"

    # Generate scratch handling
    if metadata['scratch']:
        scratch_list = "[" + ", ".join([f"'{s}'" for s in metadata['scratch']]) + "]"
    else:
        scratch_list = "[]"

    # Generate params handling
    if metadata['params']:
        params_dict = "{\n" + "\n".join([
            f"        '{param}': ctx.config.get('{param}', 0),"
            for param in metadata['params']
        ]) + "\n    }"
    else:
        params_dict = "{}"

    # Build function code
    code = f'''
def {func_name}(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build {op_type} op (auto-generated from kernel registry)."""
{chr(10).join(input_setup)}
{chr(10).join(output_setup)}
{chr(10).join(weight_setup)}

    op = Op(
        op="{op_type}",
        name=make_tensor_name(op_node, "{op_type}"),
        inputs=[{input_list}],
        outputs={output_list},
        weights={weight_list},
        scratch={scratch_list},
        params={params_dict}
    )
    ctx.set_output({output_list}[0] if {output_list} else None)
    return op
'''

    return code


def generate_op_builders_file(op_metadata: Dict, output_path: Path):
    """Generate complete op_builders file."""

    header = '''#!/usr/bin/env python3
"""
op_builders_auto.py - Auto-generated Op Builders from Kernel Registry

GENERATED BY: gen_op_builders_from_registry.py
DO NOT EDIT MANUALLY - Changes will be overwritten!

This file is automatically generated from the kernel registry to ensure
perfect alignment between kernels and IR ops.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ir_types_v6_6 import Op
from parse_template_v2 import OpNode


# =============================================================================
# OP CONTEXT: State Tracker
# =============================================================================

class OpContext:
    """
    Tracks state during op building.

    Maintains:
    - Previous op outputs (for input chaining)
    - Op counters (for unique naming)
    - Current tensor (for residual connections)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prev_output: Optional[str] = None
        self.residual_input: Optional[str] = None

        # Per-layer counters (reset each layer)
        self.rmsnorm_count = 0
        self.residual_count = 0

    def set_output(self, output: str):
        """Update previous output for next op."""
        self.prev_output = output

    def save_residual(self):
        """Save current output for residual connection."""
        self.residual_input = self.prev_output

    def reset_layer(self):
        """Reset counters for new layer."""
        self.rmsnorm_count = 0
        self.residual_count = 0
        self.residual_input = None


# =============================================================================
# TENSOR NAMING HELPERS
# =============================================================================

def make_tensor_name(op_node: OpNode, suffix: str) -> str:
    """Generate symbolic tensor name."""
    if op_node.phase == "body" and op_node.layer_index is not None:
        return f"layer.{op_node.layer_index}.{suffix}"
    else:
        # Header/footer: global names
        return suffix


def make_weight_name(op_node: OpNode, weight: str) -> str:
    """Generate weight tensor name."""
    if op_node.phase == "body" and op_node.layer_index is not None:
        return f"layer.{op_node.layer_index}.{weight}"
    else:
        return weight


# =============================================================================
# AUTO-GENERATED OP BUILDERS
# =============================================================================
'''

    # Generate all op builder functions
    builders = []
    for op_type, metadata in sorted(op_metadata.items()):
        builder_code = generate_op_builder_function(op_type, metadata)
        builders.append(builder_code)

    # Generate registry
    registry_code = '''

# =============================================================================
# OP BUILDER REGISTRY
# =============================================================================

OP_BUILDERS = {
'''

    for op_type in sorted(op_metadata.keys()):
        func_name = f"build_{op_type.replace('-', '_')}_op"
        registry_code += f'    "{op_type}": {func_name},\n'

    registry_code += '''}


# =============================================================================
# PUBLIC API
# =============================================================================

def build_op_from_template(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Optional[Op]:
    """
    Build GraphIR Op from template OpNode.

    Args:
        op_node: OpNode from parse_template_v2
        ctx: OpContext for state tracking
        manifest: Weights manifest

    Returns:
        Op object or None if op should be skipped

    Raises:
        ValueError: If op_id has no builder
    """
    if op_node.op_id not in OP_BUILDERS:
        raise ValueError(
            f"No builder for template op: '{op_node.op_id}'. "
            f"Supported ops: {', '.join(sorted(OP_BUILDERS.keys()))}"
        )

    builder = OP_BUILDERS[op_node.op_id]
    return builder(op_node, ctx, manifest)


def get_supported_ops() -> List[str]:
    """Get list of supported template op IDs."""
    return sorted(OP_BUILDERS.keys())


def check_template_ops(template_ops: List[str]) -> List[str]:
    """
    Check which template ops are missing builders.

    Args:
        template_ops: List of op IDs from template

    Returns:
        List of unsupported op IDs
    """
    supported = set(OP_BUILDERS.keys())
    return [op for op in template_ops if op not in supported]
'''

    # Write file
    with open(output_path, 'w') as f:
        f.write(header)
        for builder in builders:
            f.write(builder)
        f.write(registry_code)

    print(f"✓ Generated {len(op_metadata)} op builders → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate op builders from kernel registry",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--registry',
        type=Path,
        default=Path(__file__).parent.parent / 'kernel_maps' / 'KERNEL_REGISTRY.json',
        help='Path to kernel registry JSON'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent / 'op_builders_auto.py',
        help='Output file for generated op builders'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without writing file'
    )

    args = parser.parse_args()

    # Load registry
    print(f"Loading kernel registry: {args.registry}")
    registry = load_kernel_registry(args.registry)
    print(f"  Found {len(registry['kernels'])} kernels")

    # Analyze metadata
    print(f"\nAnalyzing kernel metadata...")
    op_metadata = analyze_kernel_metadata(registry)
    print(f"  Extracted metadata for {len(op_metadata)} op types:")
    for op_type, meta in sorted(op_metadata.items()):
        print(f"    - {op_type}: {len(meta['inputs'])} inputs, {len(meta['outputs'])} outputs")

    # Generate or show
    if args.dry_run:
        print(f"\n[DRY RUN] Would generate to: {args.output}")
        print(f"\nOp types that would be generated:")
        for op_type in sorted(op_metadata.keys()):
            func_name = f"build_{op_type.replace('-', '_')}_op"
            print(f"  - {func_name}()")
    else:
        print(f"\nGenerating op builders...")
        generate_op_builders_file(op_metadata, args.output)
        print(f"\n✓ Done! Import with:")
        print(f"    from op_builders_auto import build_op_from_template, OpContext")

    return 0


if __name__ == '__main__':
    sys.exit(main())
