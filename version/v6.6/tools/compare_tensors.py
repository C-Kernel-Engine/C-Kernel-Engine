#!/usr/bin/env python3
"""
compare_tensors.py - Tensor Comparison Tool for CK-Engine v6.6

Compares intermediate tensors between CK-Engine and a reference (llama.cpp, PyTorch, etc.)

WORKFLOW:
1. Add tensor dump points in CK-Engine generated code
2. Run CK-Engine to generate dump files
3. Generate reference dumps from llama.cpp/PyTorch
4. Run this tool to compare

Usage:
    # Compare single tensor
    python compare_tensors.py ck_embed.bin ref_embed.bin --shape 1,896

    # Compare all tensors in directories
    python compare_tensors.py --ck-dir ./ck_dumps --ref-dir ./ref_dumps

    # Generate comparison report
    python compare_tensors.py --ck-dir ./ck_dumps --ref-dir ./ref_dumps --report report.html
"""

import argparse
import json
import numpy as np
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def read_tensor_bin(path: str, shape: Tuple[int, ...] = None, dtype: str = "float32") -> np.ndarray:
    """Read raw binary tensor file."""
    np_dtype = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "int8": np.int8,
        "uint8": np.uint8,
    }.get(dtype, np.float32)

    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np_dtype)

    if shape:
        data = data.reshape(shape)

    return data


def read_tensor_json(path: str) -> np.ndarray:
    """Read tensor from JSON format (with metadata)."""
    with open(path, 'r') as f:
        obj = json.load(f)

    shape = tuple(obj.get("shape", []))
    dtype = obj.get("dtype", "float32")
    data = np.array(obj.get("data", []), dtype=np.float32)

    if shape:
        data = data.reshape(shape)

    return data


def read_tensor(path: str, shape: Tuple[int, ...] = None, dtype: str = "float32") -> np.ndarray:
    """Read tensor from file (auto-detect format)."""
    p = Path(path)
    if p.suffix == '.json':
        return read_tensor_json(path)
    elif p.suffix == '.npy':
        return np.load(path)
    else:
        return read_tensor_bin(path, shape, dtype)


def compare_tensors(a: np.ndarray, b: np.ndarray) -> Dict:
    """Compare two tensors and return metrics."""
    # Ensure same shape
    if a.shape != b.shape:
        # Try to broadcast or flatten
        try:
            a_flat = a.flatten()
            b_flat = b.flatten()
            if len(a_flat) != len(b_flat):
                return {
                    "error": f"Shape mismatch: {a.shape} vs {b.shape}",
                    "match": False
                }
            a = a_flat
            b = b_flat
        except Exception as e:
            return {"error": str(e), "match": False}

    diff = a - b
    abs_diff = np.abs(diff)

    # Handle edge cases
    a_is_zero = np.abs(a) < 1e-10
    b_is_zero = np.abs(b) < 1e-10
    both_zero = a_is_zero & b_is_zero
    rel_denom = np.maximum(np.abs(a), np.abs(b))
    rel_denom[rel_denom == 0] = 1
    rel_diff = abs_diff / rel_denom
    rel_diff[both_zero] = 0

    # Statistics
    stats = {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "elements": int(a.size),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "std_abs_diff": float(np.std(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
        "mse": float(np.mean(diff ** 2)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "correlation": float(np.corrcoef(a.flatten(), b.flatten())[0, 1]) if a.size > 1 else 1.0,
        "a_min": float(np.min(a)),
        "a_max": float(np.max(a)),
        "a_mean": float(np.mean(a)),
        "b_min": float(np.min(b)),
        "b_max": float(np.max(b)),
        "b_mean": float(np.mean(b)),
    }

    # Determine if match
    # Use tolerance thresholds
    rtol = 1e-3  # Relative tolerance
    atol = 1e-5  # Absolute tolerance

    stats["match"] = stats["max_rel_diff"] < rtol or stats["max_abs_diff"] < atol
    stats["close"] = stats["correlation"] > 0.99

    # Find worst elements
    flat_diff = abs_diff.flatten()
    worst_idx = np.argmax(flat_diff)
    stats["worst_idx"] = int(worst_idx)
    stats["worst_a"] = float(a.flatten()[worst_idx])
    stats["worst_b"] = float(b.flatten()[worst_idx])

    return stats


def print_comparison(name: str, stats: Dict, verbose: bool = False):
    """Print comparison results."""
    if "error" in stats:
        print(f"[ERROR] {name}: {stats['error']}")
        return

    match_str = "[MATCH]" if stats["match"] else ("[CLOSE]" if stats["close"] else "[DIFF]")
    color = "\033[92m" if stats["match"] else ("\033[93m" if stats["close"] else "\033[91m")
    reset = "\033[0m"

    print(f"{color}{match_str}{reset} {name}")
    print(f"  Shape: {stats['shape_a']}")
    print(f"  Max abs diff: {stats['max_abs_diff']:.6e}")
    print(f"  Max rel diff: {stats['max_rel_diff']:.6e}")
    print(f"  MSE: {stats['mse']:.6e}")
    print(f"  Correlation: {stats['correlation']:.6f}")

    if verbose:
        print(f"  A: min={stats['a_min']:.6f} max={stats['a_max']:.6f} mean={stats['a_mean']:.6f}")
        print(f"  B: min={stats['b_min']:.6f} max={stats['b_max']:.6f} mean={stats['b_mean']:.6f}")
        print(f"  Worst at idx {stats['worst_idx']}: a={stats['worst_a']:.6f} b={stats['worst_b']:.6f}")


def find_divergence_point(comparisons: List[Tuple[str, Dict]]) -> Optional[str]:
    """Find where divergence starts in a sequence of comparisons."""
    last_good = None
    first_bad = None

    for name, stats in comparisons:
        if stats.get("match", False) or stats.get("close", False):
            last_good = name
        elif first_bad is None:
            first_bad = name

    if first_bad:
        return f"Divergence starts at: {first_bad} (last good: {last_good or 'none'})"
    return None


def generate_dump_code_c() -> str:
    """Generate C code snippet for dumping tensors."""
    return '''
/* Add to ck-kernel-inference.c to dump tensors */

static void dump_tensor_fp32(const char* name, const float* data, int count) {
    char path[256];
    snprintf(path, sizeof(path), "./ck_dumps/%s.bin", name);

    FILE* f = fopen(path, "wb");
    if (f) {
        fwrite(data, sizeof(float), count, f);
        fclose(f);
        fprintf(stderr, "Dumped %s: %d floats\\n", name, count);
    }
}

/* Example: dump after embedding lookup */
// dump_tensor_fp32("embed_out", embedded_input, seq_len * EMBED_DIM);

/* Example: dump after layer 0 attention */
// dump_tensor_fp32("layer0_attn_out", attn_out, seq_len * EMBED_DIM);
'''


def generate_dump_code_llamacpp() -> str:
    """Generate llama.cpp code snippet for dumping tensors."""
    return '''
/* Add to llama.cpp inference to dump tensors for comparison */

#include <fstream>

void dump_tensor(const char* name, const float* data, size_t count) {
    std::string path = std::string("./ref_dumps/") + name + ".bin";
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), count * sizeof(float));
    fprintf(stderr, "Dumped %s: %zu floats\\n", name, count);
}

/* In llama_decode_internal(), after embedding lookup: */
// dump_tensor("embed_out", inp_embd->data, n_tokens * n_embd);

/* After each layer's attention: */
// dump_tensor("layer0_attn_out", cur->data, n_tokens * n_embd);
'''


def generate_html_report(comparisons: List[Tuple[str, Dict]], output_path: str):
    """Generate HTML comparison report."""
    html = '''<!DOCTYPE html>
<html>
<head>
    <title>Tensor Comparison Report</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #e94560; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #444; padding: 10px; text-align: left; }
        th { background: #16213e; }
        .match { background: #00b894; color: white; }
        .close { background: #fdcb6e; color: black; }
        .diff { background: #d63031; color: white; }
        .stat { font-family: monospace; }
    </style>
</head>
<body>
    <h1>CK-Engine vs Reference Tensor Comparison</h1>
    <table>
        <tr>
            <th>Tensor</th>
            <th>Status</th>
            <th>Shape</th>
            <th>Max Abs Diff</th>
            <th>Max Rel Diff</th>
            <th>Correlation</th>
            <th>MSE</th>
        </tr>
'''

    for name, stats in comparisons:
        if "error" in stats:
            html += f'''<tr class="diff">
                <td>{name}</td>
                <td>ERROR</td>
                <td colspan="5">{stats["error"]}</td>
            </tr>'''
        else:
            status_class = "match" if stats["match"] else ("close" if stats["close"] else "diff")
            status_text = "MATCH" if stats["match"] else ("CLOSE" if stats["close"] else "DIFF")
            html += f'''<tr>
                <td>{name}</td>
                <td class="{status_class}">{status_text}</td>
                <td class="stat">{stats["shape_a"]}</td>
                <td class="stat">{stats["max_abs_diff"]:.2e}</td>
                <td class="stat">{stats["max_rel_diff"]:.2e}</td>
                <td class="stat">{stats["correlation"]:.4f}</td>
                <td class="stat">{stats["mse"]:.2e}</td>
            </tr>'''

    # Find divergence
    div = find_divergence_point(comparisons)

    html += f'''
    </table>
    <h2>Analysis</h2>
    <p>{div or "All tensors match within tolerance."}</p>

    <h2>Dump Code Snippets</h2>
    <h3>CK-Engine (C)</h3>
    <pre style="background: #16213e; padding: 15px; overflow-x: auto;">{generate_dump_code_c()}</pre>

    <h3>llama.cpp (C++)</h3>
    <pre style="background: #16213e; padding: 15px; overflow-x: auto;">{generate_dump_code_llamacpp()}</pre>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Report saved to: {output_path}")


def compare_directories(ck_dir: str, ref_dir: str) -> List[Tuple[str, Dict]]:
    """Compare all matching tensor files in two directories."""
    ck_path = Path(ck_dir)
    ref_path = Path(ref_dir)

    comparisons = []

    for ck_file in sorted(ck_path.glob("*.bin")):
        name = ck_file.stem
        ref_file = ref_path / f"{name}.bin"

        if ref_file.exists():
            try:
                a = read_tensor(str(ck_file))
                b = read_tensor(str(ref_file))
                stats = compare_tensors(a, b)
                comparisons.append((name, stats))
            except Exception as e:
                comparisons.append((name, {"error": str(e)}))
        else:
            comparisons.append((name, {"error": f"No reference file: {ref_file}"}))

    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Compare tensors between CK-Engine and reference")
    parser.add_argument("ck_tensor", nargs="?", help="CK-Engine tensor file")
    parser.add_argument("ref_tensor", nargs="?", help="Reference tensor file")
    parser.add_argument("--shape", help="Tensor shape (e.g., 1,896)")
    parser.add_argument("--dtype", default="float32", help="Data type")
    parser.add_argument("--ck-dir", help="CK-Engine dumps directory")
    parser.add_argument("--ref-dir", help="Reference dumps directory")
    parser.add_argument("--report", help="Generate HTML report to this path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--gen-code", action="store_true", help="Print dump code snippets")

    args = parser.parse_args()

    if args.gen_code:
        print("=== CK-Engine Dump Code (C) ===")
        print(generate_dump_code_c())
        print("\n=== llama.cpp Dump Code (C++) ===")
        print(generate_dump_code_llamacpp())
        return 0

    comparisons = []

    if args.ck_dir and args.ref_dir:
        comparisons = compare_directories(args.ck_dir, args.ref_dir)
    elif args.ck_tensor and args.ref_tensor:
        shape = tuple(int(x) for x in args.shape.split(",")) if args.shape else None
        try:
            a = read_tensor(args.ck_tensor, shape, args.dtype)
            b = read_tensor(args.ref_tensor, shape, args.dtype)
            stats = compare_tensors(a, b)
            comparisons.append((Path(args.ck_tensor).stem, stats))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 1

    # Print results
    for name, stats in comparisons:
        print_comparison(name, stats, args.verbose)

    # Generate report if requested
    if args.report:
        generate_html_report(comparisons, args.report)

    # Print divergence analysis
    div = find_divergence_point(comparisons)
    if div:
        print(f"\n{div}")

    # Return error if any tensor doesn't match
    has_errors = any(not s.get("match", False) and not s.get("close", False) for _, s in comparisons)
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
