#!/usr/bin/env python3
"""
IR Debug and Visualization Tool for C-Kernel-Engine v6.6

Features:
1. Memory layout table with offsets and sizes
2. Kernel flow visualization
3. Buffer overlap detection
4. Parity testing with llama.cpp/PyTorch
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# ANSI colors
class C:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def format_bytes(b: int) -> str:
    if b >= 1024**3: return f"{b/1024**3:.2f} GB"
    if b >= 1024**2: return f"{b/1024**2:.2f} MB"
    if b >= 1024: return f"{b/1024:.2f} KB"
    return f"{b} B"


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def show_memory_layout(layout: Dict, show_weights: bool = True, show_activations: bool = True):
    """Display memory layout in a detailed table."""
    print(f"\n{C.BOLD}{'='*80}{C.END}")
    print(f"{C.BOLD}MEMORY LAYOUT ANALYSIS{C.END}")
    print(f"{'='*80}")

    memory = layout.get("memory", layout)
    weights = memory.get("weights", {})
    activations = memory.get("activations", {})
    arena = memory.get("arena", {})

    # Arena info
    print(f"\n{C.CYAN}Arena Configuration:{C.END}")
    print(f"  Mode: {arena.get('mode', 'unknown')}")
    print(f"  Weights base: 0x{arena.get('weights_base', 0):x}")
    print(f"  Activations base: 0x{arena.get('activations_base', 0):x}")
    print(f"  Total size: {format_bytes(arena.get('total_size', 0))}")

    if show_weights:
        entries = weights.get("entries", [])
        if entries:
            print(f"\n{C.YELLOW}Weights ({len(entries)} entries):{C.END}")
            print(f"  {'Name':<40} {'Offset':>12} {'Size':>12} {'Dtype':<8} Shape")
            print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*8} {'-'*20}")

            total_size = 0
            for e in entries[:50]:  # Limit output
                name = (e.get("name") or e.get("key") or "-")[:40]
                offset = e.get("offset", 0)
                size = e.get("size", 0)
                dtype = e.get("dtype", "-")
                shape = e.get("shape", [])
                total_size += size

                # Color by dtype
                dtype_color = {
                    "q8_0": C.RED, "q5_0": C.MAGENTA, "q4_k": C.YELLOW,
                    "q6_k": C.BLUE, "fp32": C.GREEN
                }.get(dtype, C.DIM)

                print(f"  {name:<40} {C.BLUE}0x{offset:>10x}{C.END} {C.GREEN}{format_bytes(size):>12}{C.END} {dtype_color}{dtype:<8}{C.END} {shape}")

            if len(entries) > 50:
                print(f"  ... and {len(entries) - 50} more entries")
            print(f"\n  {C.BOLD}Total weights: {format_bytes(total_size)}{C.END}")

    if show_activations:
        buffers = activations.get("buffers", [])
        if buffers:
            print(f"\n{C.GREEN}Activation Buffers ({len(buffers)} buffers):{C.END}")
            print(f"  {'Buffer':<25} {'Offset':>12} {'Size':>12} {'Define':<30} Shape")
            print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*30} {'-'*30}")

            total_size = 0
            overlaps = []

            for i, b in enumerate(buffers):
                name = (b.get("name") or "-")[:25]
                offset = b.get("offset", 0)
                size = b.get("size", 0)
                define = (b.get("define") or "-")[:30]
                shape = b.get("shape", [])
                total_size += size

                # Check for overlaps
                for j, other in enumerate(buffers):
                    if i != j:
                        o_offset = other.get("offset", 0)
                        o_size = other.get("size", 0)
                        if offset < o_offset + o_size and offset + size > o_offset:
                            if (i, j) not in overlaps and (j, i) not in overlaps:
                                overlaps.append((i, j))

                # Highlight important buffers
                if "kv_cache" in name:
                    color = C.MAGENTA
                elif "scratch" in name:
                    color = C.YELLOW
                else:
                    color = ""

                print(f"  {color}{name:<25}{C.END} {C.BLUE}0x{offset:>10x}{C.END} {C.GREEN}{format_bytes(size):>12}{C.END} {define:<30} {shape}")

            print(f"\n  {C.BOLD}Total activations: {format_bytes(total_size)}{C.END}")

            if overlaps:
                print(f"\n  {C.RED}WARNING: Buffer overlaps detected:{C.END}")
                for i, j in overlaps:
                    b1, b2 = buffers[i], buffers[j]
                    print(f"    - {b1.get('name')} overlaps with {b2.get('name')}")


def show_kernel_flow(ir: Dict, layer: Optional[int] = None, show_args: bool = False):
    """Display kernel flow with connections."""
    print(f"\n{C.BOLD}{'='*80}{C.END}")
    print(f"{C.BOLD}KERNEL FLOW ANALYSIS{C.END}")
    print(f"{'='*80}")

    ops = ir.get("operations", ir.get("ops", []))
    config = ir.get("config", {})

    print(f"\n{C.CYAN}Configuration:{C.END}")
    print(f"  Mode: {ir.get('mode', 'unknown')}")
    print(f"  Layers: {config.get('num_layers', '-')}")
    print(f"  Embed dim: {config.get('embed_dim', '-')}")
    print(f"  Heads: {config.get('num_heads', '-')} / KV heads: {config.get('num_kv_heads', '-')}")

    # Group by layer
    layers = {}
    for op in ops:
        l = op.get("layer", -1)
        section = op.get("section", "body")
        key = f"Header" if section == "header" else f"Footer" if section == "footer" else f"Layer {l}"
        if layer is not None and l != layer and section == "body":
            continue
        if key not in layers:
            layers[key] = []
        layers[key].append(op)

    for layer_name, layer_ops in layers.items():
        print(f"\n{C.YELLOW}{layer_name} ({len(layer_ops)} ops){C.END}")

        for i, op in enumerate(layer_ops):
            idx = op.get("idx", i)
            op_name = op.get("op", "-")
            func = op.get("function", op.get("kernel_id", "-"))

            # Color by kernel type
            if "quantize" in func:
                color = C.RED
                marker = "⚡"
            elif "gemv" in func or "gemm" in func:
                color = C.GREEN
                marker = "▣"
            elif "attention" in func:
                color = C.MAGENTA
                marker = "◆"
            elif "rmsnorm" in func:
                color = C.CYAN
                marker = "○"
            else:
                color = C.DIM
                marker = "•"

            # Get dimensions
            params = op.get("params", {})
            dims = []
            if params.get("_input_dim"):
                dims.append(f"in:{params['_input_dim']}")
            if params.get("_output_dim"):
                dims.append(f"out:{params['_output_dim']}")

            errors = op.get("errors", [])
            error_mark = f" {C.RED}[ERRORS]{C.END}" if errors else ""

            print(f"  {C.DIM}[{idx:3d}]{C.END} {marker} {color}{op_name:<30}{C.END} → {C.BLUE}{func}{C.END} {C.DIM}{' | '.join(dims)}{C.END}{error_mark}")

            if show_args and op.get("args"):
                for arg in op["args"]:
                    expr = arg.get("expr", arg.get("source", "-"))
                    if len(expr) > 60:
                        expr = expr[:57] + "..."
                    src_type = "📥" if "input" in arg.get("source", "") or "activation" in arg.get("source", "") else "📤" if "output" in arg.get("source", "") else "⚖️"
                    print(f"       {src_type} {C.DIM}{arg.get('name', '-')}: {expr}{C.END}")

            if errors:
                for err in errors:
                    print(f"       {C.RED}⚠ {err}{C.END}")


def check_quantize_buffer_flow(ir: Dict):
    """Check that quantize ops have proper buffer connections."""
    print(f"\n{C.BOLD}{'='*80}{C.END}")
    print(f"{C.BOLD}QUANTIZE BUFFER FLOW CHECK{C.END}")
    print(f"{'='*80}")

    ops = ir.get("operations", ir.get("ops", []))

    quantize_ops = [op for op in ops if "quantize" in op.get("function", "")]
    print(f"\nFound {len(quantize_ops)} quantize operations")

    issues = []
    for op in quantize_ops:
        idx = op.get("idx", -1)
        args = op.get("args", [])

        input_arg = next((a for a in args if a.get("name") == "x"), None)
        output_arg = next((a for a in args if a.get("name") == "y"), None)
        size_arg = next((a for a in args if a.get("name") == "k"), None)

        print(f"\n  Op {idx}: {op.get('op')} ({op.get('function')})")
        print(f"    Layer: {op.get('layer', '-')}")

        if input_arg:
            print(f"    Input: {input_arg.get('expr', 'MISSING')}")
        else:
            issues.append(f"Op {idx}: Missing input argument 'x'")
            print(f"    {C.RED}Input: MISSING{C.END}")

        if output_arg:
            print(f"    Output: {output_arg.get('expr', 'MISSING')}")
        else:
            issues.append(f"Op {idx}: Missing output argument 'y'")
            print(f"    {C.RED}Output: MISSING{C.END}")

        if size_arg:
            print(f"    Size: {size_arg.get('expr', 'MISSING')}")
        else:
            issues.append(f"Op {idx}: Missing size argument 'k'")
            print(f"    {C.RED}Size: MISSING{C.END}")

        # Check for buffer aliasing issues
        if input_arg and output_arg:
            in_buf = input_arg.get("expr", "")
            out_buf = output_arg.get("expr", "")
            if in_buf == out_buf:
                issues.append(f"Op {idx}: Input and output buffers are the same!")
                print(f"    {C.RED}WARNING: Same buffer for input/output{C.END}")

    if issues:
        print(f"\n{C.RED}Issues found:{C.END}")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n{C.GREEN}No issues found in quantize buffer flow{C.END}")


def show_buffer_usage(ir: Dict, layout: Dict):
    """Show which buffers are used by which ops."""
    print(f"\n{C.BOLD}{'='*80}{C.END}")
    print(f"{C.BOLD}BUFFER USAGE ANALYSIS{C.END}")
    print(f"{'='*80}")

    ops = ir.get("operations", ir.get("ops", []))
    memory = layout.get("memory", layout)
    buffers = memory.get("activations", {}).get("buffers", [])

    # Map buffer defines to usage
    buffer_usage = {b.get("name"): [] for b in buffers}

    for op in ops:
        for arg in op.get("args", []):
            expr = arg.get("expr", "")
            for buf in buffers:
                define = buf.get("define", "")
                if define and define in expr:
                    buffer_usage[buf.get("name")].append({
                        "op_idx": op.get("idx"),
                        "op": op.get("op"),
                        "arg": arg.get("name"),
                        "type": "read" if "input" in arg.get("source", "") or "activation" in arg.get("source", "") else "write"
                    })

    for buf_name, usages in buffer_usage.items():
        if not usages:
            continue
        reads = [u for u in usages if u["type"] == "read"]
        writes = [u for u in usages if u["type"] == "write"]
        print(f"\n{C.CYAN}{buf_name}:{C.END}")
        print(f"  Reads: {len(reads)}, Writes: {len(writes)}")
        for u in usages[:10]:
            marker = "📥" if u["type"] == "read" else "📤"
            print(f"    {marker} Op {u['op_idx']}: {u['op']}.{u['arg']}")
        if len(usages) > 10:
            print(f"    ... and {len(usages) - 10} more")


def generate_parity_test(ir: Dict, output_path: Path):
    """Generate C code for parity testing."""
    print(f"\n{C.BOLD}{'='*80}{C.END}")
    print(f"{C.BOLD}GENERATING PARITY TEST CODE{C.END}")
    print(f"{'='*80}")

    ops = ir.get("operations", ir.get("ops", []))

    code = '''// Auto-generated parity test code
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float min_f(const float* data, int n) {
    float m = data[0];
    for (int i = 1; i < n; i++) if (data[i] < m) m = data[i];
    return m;
}

static float max_f(const float* data, int n) {
    float m = data[0];
    for (int i = 1; i < n; i++) if (data[i] > m) m = data[i];
    return m;
}

static void dump_tensor(const char* name, const float* data, int size, int layer) {
    char filename[256];
    snprintf(filename, sizeof(filename), "parity_%s_L%d.bin", name, layer);
    FILE* f = fopen(filename, "wb");
    if (f) {
        fwrite(&size, sizeof(int), 1, f);  // Write size first
        fwrite(data, sizeof(float), size, f);
        fclose(f);
    }
    printf("PARITY: %s layer=%d size=%d min=%.6f max=%.6f\\n",
           name, layer, size, min_f(data, size), max_f(data, size));
}

// Call these after each major operation in ck_decode/ck_prefill:
'''

    # Generate dump calls for key operations
    dump_points = {
        "embedding": ("embedded_input", "embed_dim"),
        "rmsnorm": ("layer_input", "embed_dim"),
        "q_proj": ("q_scratch", "num_heads * head_dim"),
        "k_proj": ("k_scratch", "num_kv_heads * head_dim"),
        "v_proj": ("v_scratch", "num_kv_heads * head_dim"),
        "attention": ("attn_scratch", "num_heads * head_dim"),
        "out_proj": ("embedded_input", "embed_dim"),
        "mlp_gate_up": ("mlp_scratch", "intermediate_size * 2"),
        "mlp_down": ("layer_input", "embed_dim"),
        "logits": ("logits", "vocab_size"),
    }

    for i, (op_name, (buffer, size_expr)) in enumerate(dump_points.items()):
        code += f'''
// After {op_name}:
// dump_tensor("{op_name}", (float*)(model->bump + A_{buffer.upper()}), {size_expr}, layer);
'''

    code += '''
// Python comparison script:
/*
import numpy as np
import struct

def load_parity_tensor(path):
    with open(path, 'rb') as f:
        size = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(size * 4), dtype=np.float32)
    return data

def compare_tensors(ck_path, ref_path, tolerance=1e-4):
    ck = load_parity_tensor(ck_path)
    ref = np.load(ref_path) if ref_path.endswith('.npy') else load_parity_tensor(ref_path)

    if ck.shape != ref.shape:
        print(f"Shape mismatch: CK={ck.shape} vs Ref={ref.shape}")
        return False

    diff = np.abs(ck - ref)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
    print(f"CK range: [{ck.min():.4f}, {ck.max():.4f}]")
    print(f"Ref range: [{ref.min():.4f}, {ref.max():.4f}]")

    if max_diff > tolerance:
        # Find top differences
        top_idx = np.argsort(diff)[-10:][::-1]
        print("Top differences:")
        for idx in top_idx:
            print(f"  [{idx}]: CK={ck[idx]:.6f} Ref={ref[idx]:.6f} Diff={diff[idx]:.6f}")
        return False
    return True
*/
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(code)

    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="IR Debug and Visualization Tool")
    parser.add_argument("--model-dir", type=Path,
                       default=Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF",
                       help="Model cache directory")
    parser.add_argument("--mode", choices=["decode", "prefill"], default="decode",
                       help="Mode to analyze")
    parser.add_argument("--layout", type=Path, help="Layout JSON file")
    parser.add_argument("--ir", type=Path, help="Lowered IR JSON file (call-ready preferred)")

    parser.add_argument("--memory", action="store_true", help="Show memory layout")
    parser.add_argument("--kernels", action="store_true", help="Show kernel flow")
    parser.add_argument("--layer", type=int, help="Filter to specific layer")
    parser.add_argument("--args", action="store_true", help="Show kernel arguments")
    parser.add_argument("--quantize", action="store_true", help="Check quantize buffer flow")
    parser.add_argument("--buffers", action="store_true", help="Show buffer usage")
    parser.add_argument("--parity", type=Path, help="Generate parity test code to file")
    parser.add_argument("--all", action="store_true", help="Run all analyses")

    args = parser.parse_args()

    # Auto-load files from model dir
    layout_path = args.layout or args.model_dir / f"layout_{args.mode}.json"
    ir_path = args.ir or args.model_dir / f"lowered_{args.mode}_call.json"

    layout = None
    ir = None

    if layout_path.exists():
        layout = load_json(layout_path)
        print(f"{C.GREEN}Loaded layout: {layout_path}{C.END}")
    else:
        print(f"{C.YELLOW}Layout not found: {layout_path}{C.END}")

    if ir_path.exists():
        ir = load_json(ir_path)
        print(f"{C.GREEN}Loaded IR: {ir_path}{C.END}")
    else:
        print(f"{C.YELLOW}IR not found: {ir_path}{C.END}")

    # Run requested analyses
    if args.all or args.memory:
        if layout:
            show_memory_layout(layout)
        else:
            print(f"{C.RED}Cannot show memory layout: no layout file{C.END}")

    if args.all or args.kernels:
        if ir:
            show_kernel_flow(ir, layer=args.layer, show_args=args.args)
        else:
            print(f"{C.RED}Cannot show kernel flow: no IR file{C.END}")

    if args.all or args.quantize:
        if ir:
            check_quantize_buffer_flow(ir)
        else:
            print(f"{C.RED}Cannot check quantize flow: no IR file{C.END}")

    if args.all or args.buffers:
        if ir and layout:
            show_buffer_usage(ir, layout)
        else:
            print(f"{C.RED}Cannot show buffer usage: need both IR and layout{C.END}")

    if args.parity:
        if ir:
            generate_parity_test(ir, args.parity)
        else:
            print(f"{C.RED}Cannot generate parity test: no IR file{C.END}")

    # If nothing specified, show summary
    if not any([args.memory, args.kernels, args.quantize, args.buffers, args.parity, args.all]):
        print(f"\n{C.CYAN}Usage examples:{C.END}")
        print(f"  python debug_ir.py --memory          # Show memory layout")
        print(f"  python debug_ir.py --kernels         # Show kernel flow")
        print(f"  python debug_ir.py --kernels --layer 0 --args  # Layer 0 with args")
        print(f"  python debug_ir.py --quantize        # Check quantize buffer flow")
        print(f"  python debug_ir.py --all             # Run all analyses")
        print(f"  python debug_ir.py --parity out.c    # Generate parity test code")


if __name__ == "__main__":
    main()
