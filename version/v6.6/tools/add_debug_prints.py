#!/usr/bin/env python3
"""
add_debug_prints.py - Add debug print statements to generated C code

This creates a modified version of model_v6_6.c with debug prints at key points
to trace where values go wrong.

Usage:
    python add_debug_prints.py <model_cache_dir>
    python add_debug_prints.py <model_cache_dir> --layer 0 --ops 0-10
"""

import argparse
import re
import sys
from pathlib import Path


DEBUG_PRINT_TEMPLATE = '''
    {{ // DEBUG: {name}
        float *_dbg = (float*)(model->bump + {offset});
        float _sum = 0, _min = 1e30, _max = -1e30;
        int _nan = 0, _inf = 0;
        for(int _i = 0; _i < {size}; _i++) {{
            float v = _dbg[_i];
            if(__builtin_isnan(v)) _nan++;
            else if(__builtin_isinf(v)) _inf++;
            else {{ _sum += v; if(v < _min) _min = v; if(v > _max) _max = v; }}
        }}
        printf("[{name}] sum=%.4f min=%.4f max=%.4f nan=%d inf=%d first=[%.4f,%.4f,%.4f]\\n",
               _sum, _min, _max, _nan, _inf, _dbg[0], _dbg[1], _dbg[2]);
        if(_nan || _inf) printf("  WARNING: {name} has %d NaN, %d Inf!\\n", _nan, _inf);
    }}
'''

def add_debug_prints(c_path: Path, output_path: Path, layer_filter: int = None, op_range: tuple = None):
    """Add debug prints after each operation."""
    content = c_path.read_text()

    # Extract activation offsets
    offsets = {}
    for match in re.finditer(r'#define\s+(A_\w+)\s+(\d+)', content):
        offsets[match.group(1)] = int(match.group(2))

    # Find operation comments and insert debug prints
    op_pattern = re.compile(
        r'/\*\s*Op\s+(\d+):\s+(\w+).*?layer=(-?\d+).*?\*/(.*?)(?=/\*\s*Op|\Z)',
        re.DOTALL
    )

    insertions = []

    for match in op_pattern.finditer(content):
        op_num = int(match.group(1))
        op_name = match.group(2)
        layer = int(match.group(3))
        op_body = match.group(4)

        # Apply filters
        if layer_filter is not None and layer != layer_filter:
            continue
        if op_range is not None and (op_num < op_range[0] or op_num > op_range[1]):
            continue

        # Determine which buffer was written
        buffer_name = None
        buffer_size = 896  # Default to embed_dim

        if 'embedding' in op_name:
            buffer_name = 'A_EMBEDDED_INPUT'
        elif 'rmsnorm' in op_name:
            buffer_name = 'A_LAYER_INPUT' if 'ln1' in op_body or 'LN1' in op_body else 'A_LAYER_OUTPUT'
        elif 'q_proj' in op_name or 'Q_' in op_body:
            buffer_name = 'A_Q_SCRATCH'
            buffer_size = 14 * 64  # num_heads * head_dim
        elif 'k_proj' in op_name or 'K_' in op_body:
            buffer_name = 'A_K_SCRATCH'
            buffer_size = 2 * 64  # num_kv_heads * head_dim
        elif 'v_proj' in op_name or 'V_' in op_body:
            buffer_name = 'A_V_SCRATCH'
            buffer_size = 2 * 64
        elif 'attention' in op_name:
            buffer_name = 'A_ATTN_SCRATCH'
        elif 'out_proj' in op_name or 'wo' in op_body.lower():
            buffer_name = 'A_ATTN_SCRATCH'
        elif 'residual' in op_name:
            buffer_name = 'A_LAYER_OUTPUT'
        elif 'swiglu' in op_name:
            buffer_name = 'A_MLP_SCRATCH'
            buffer_size = 4864  # intermediate_size

        if buffer_name and buffer_name in offsets:
            debug_code = DEBUG_PRINT_TEMPLATE.format(
                name=f"Op{op_num}_{op_name}_L{layer}",
                offset=offsets[buffer_name],
                size=min(buffer_size, 1000)  # Limit check size for speed
            )

            # Find end of op (before next op or function end)
            insert_pos = match.end(4)
            # Adjust to be after the last semicolon
            last_semi = op_body.rfind(';')
            if last_semi != -1:
                insert_pos = match.start(4) + last_semi + 1

            insertions.append((insert_pos, debug_code))

    # Apply insertions in reverse order to preserve positions
    modified = content
    for pos, code in reversed(insertions):
        modified = modified[:pos] + code + modified[pos:]

    # Add necessary includes at top
    if '#include <math.h>' not in modified:
        modified = '#include <math.h>\n' + modified

    output_path.write_text(modified)
    print(f"Added {len(insertions)} debug print statements")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Add debug prints to generated C")
    parser.add_argument('model_dir', help='Path to model cache directory')
    parser.add_argument('--layer', type=int, help='Only add prints for this layer')
    parser.add_argument('--ops', help='Op range (e.g., 0-10)')
    parser.add_argument('-o', '--output', help='Output file path')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    c_path = model_dir / "model_v6_6.c"

    if not c_path.exists():
        print(f"Error: {c_path} not found")
        sys.exit(1)

    output_path = Path(args.output) if args.output else model_dir / "model_v6_6_debug.c"

    op_range = None
    if args.ops:
        parts = args.ops.split('-')
        op_range = (int(parts[0]), int(parts[1]))

    add_debug_prints(c_path, output_path, args.layer, op_range)


if __name__ == '__main__':
    main()
