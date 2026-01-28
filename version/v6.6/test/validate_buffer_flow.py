#!/usr/bin/env python3
"""
Validate buffer flow through all ops in generated code.
Ensures each op reads from where the previous op wrote.
"""

import re
import sys
from pathlib import Path

def parse_generated_code(code_path):
    """Parse generated C code to extract op buffer assignments."""
    with open(code_path) as f:
        lines = f.readlines()

    ops = []
    i = 0
    in_decode = False

    while i < len(lines):
        line = lines[i]

        # Track when we're in ck_decode
        if 'static void ck_decode(' in line:
            in_decode = True
        if in_decode and line.strip() == '}' and i > 0 and 'model->pos' in lines[i-1]:
            in_decode = False

        # Look for op comments
        if in_decode and '/* Op ' in line and ':' in line:
            # Parse: /* Op 2: quantize_row_q8_0 (quantize_input_0) layer=0 section=body */
            match = re.search(r'/\* Op (\d+): (\S+) \((\S+)\) layer=(-?\d+)', line)
            if match:
                op_idx = int(match.group(1))
                kernel = match.group(2)
                op_name = match.group(3)
                layer = int(match.group(4))

                # Get the function call (next few lines)
                func_lines = []
                j = i + 1
                while j < len(lines) and ');' not in lines[j]:
                    func_lines.append(lines[j].strip())
                    j += 1
                if j < len(lines):
                    func_lines.append(lines[j].strip())

                func_code = ' '.join(func_lines)

                # Extract buffer references
                inputs = []
                outputs = []

                # Find all A_XXX references
                buf_refs = re.findall(r'\((const )?(float|void)\s*\*\)\s*\([^)]+\+\s*(A_\w+)\)', func_code)
                for is_const, dtype, buf_name in buf_refs:
                    if is_const:
                        inputs.append(buf_name)
                    else:
                        # First non-const is usually output
                        if not outputs:
                            outputs.append(buf_name)
                        else:
                            inputs.append(buf_name)

                ops.append({
                    'idx': op_idx,
                    'kernel': kernel,
                    'op': op_name,
                    'layer': layer,
                    'inputs': inputs,
                    'outputs': outputs,
                    'code': func_code[:100]
                })
        i += 1

    return ops


def validate_buffer_flow(ops):
    """Validate buffer flow and detect issues."""
    issues = []
    buffer_contents = {}  # buffer -> (last_writer_op_idx, data_type)

    print(f"\n{'='*80}")
    print("BUFFER FLOW VALIDATION (First 30 ops)")
    print('='*80)

    for op in ops[:30]:
        op_idx = op['idx']
        op_name = op['op']
        kernel = op['kernel']
        inputs = op['inputs']
        outputs = op['outputs']

        is_quantize = 'quantize' in kernel
        expects_quantized = '_q8_0' in kernel or '_q8_k' in kernel

        print(f"\nOp {op_idx}: {op_name} ({kernel})")
        print(f"  IN:  {inputs}")
        print(f"  OUT: {outputs}")

        # Validate inputs
        for buf in inputs:
            if buf in buffer_contents:
                writer_idx, dtype = buffer_contents[buf]
                print(f"    {buf} <- written by Op {writer_idx} ({dtype})")
            else:
                print(f"    {buf} <- UNINITIALIZED or from embedding")

        # Check for input/output overlap (corruption risk)
        # Some ops are intentionally in-place: rmsnorm, residual_add, bias_add, silu_mul
        in_place_ops = {'rmsnorm', 'residual_add', 'bias_add', 'silu_mul', 'add_inplace', 'memcpy'}
        overlap = set(inputs) & set(outputs)
        is_inplace_allowed = any(iop in kernel or iop in op_name for iop in in_place_ops)
        if overlap and not is_inplace_allowed:
            issue = f"Op {op_idx} ({op_name}): IN/OUT overlap {overlap} - potential corruption!"
            issues.append(issue)
            print(f"    ⚠️  {issue}")
        elif overlap:
            print(f"    ✓ In-place op (expected overlap)")

        # Update buffer contents
        for buf in outputs:
            dtype = 'q8' if is_quantize else 'fp32'
            buffer_contents[buf] = (op_idx, dtype)

    return issues


def check_quantize_flow(ops):
    """Check that quantize ops have correct input/output."""
    print(f"\n{'='*80}")
    print("QUANTIZE OP VALIDATION")
    print('='*80)

    issues = []

    for i, op in enumerate(ops):
        if 'quantize' in op['op']:
            inputs = op['inputs']
            outputs = op['outputs']

            print(f"\n{op['op']} (Op {op['idx']}, layer {op['layer']}):")
            print(f"  Reads from: {inputs}")
            print(f"  Writes to:  {outputs}")

            # Check input != output
            if inputs and outputs:
                if inputs[0] == outputs[0]:
                    issue = f"{op['op']}: reads and writes same buffer {inputs[0]}!"
                    issues.append(issue)
                    print(f"  ❌ {issue}")
                else:
                    print(f"  ✓ Different input/output buffers")

            # Find what the next consumer reads
            for j in range(i+1, min(i+5, len(ops))):
                next_op = ops[j]
                if next_op['inputs']:
                    print(f"  Next consumer: {next_op['op']} reads {next_op['inputs']}")
                    # Check if next op reads from quantize output
                    if outputs and outputs[0] in next_op['inputs']:
                        print(f"  ✓ Next op reads from quantize output")
                    elif outputs:
                        print(f"  ⚠️  Next op does NOT read from quantize output {outputs}")
                    break

    return issues


def main():
    cache_dir = Path.home() / ".cache/ck-engine-v6.6/models"
    model_dirs = list(cache_dir.glob("*Qwen*")) + list(cache_dir.glob("*qwen*"))

    if not model_dirs:
        print("Error: No Qwen model found in cache")
        sys.exit(1)

    model_dir = model_dirs[0]
    code_path = model_dir / "model_v6_6.c"

    print(f"Analyzing: {code_path}")

    ops = parse_generated_code(code_path)
    print(f"Found {len(ops)} ops in ck_decode")

    if not ops:
        print("\nNo ops found - check parser")
        # Debug: show some lines from the file
        with open(code_path) as f:
            content = f.read()
        if 'ck_decode' in content:
            print("ck_decode function exists")
            # Show first op comment
            match = re.search(r'/\* Op \d+:.*?\*/', content)
            if match:
                print(f"Sample op comment: {match.group(0)}")
        return

    # Run validations
    flow_issues = validate_buffer_flow(ops)
    quantize_issues = check_quantize_flow(ops)

    # Summary
    all_issues = flow_issues + quantize_issues
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)

    if all_issues:
        print(f"\n❌ {len(all_issues)} ISSUES FOUND:")
        for issue in all_issues:
            print(f"  • {issue}")
        sys.exit(1)
    else:
        print("\n✓ No buffer flow issues detected in analyzed ops")


if __name__ == '__main__':
    main()
