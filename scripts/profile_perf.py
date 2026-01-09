#!/usr/bin/env python3
"""
Profile C-Kernel performance with perf and flamegraph.
This script compiles the litmus generated model and runs perf record on it.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# FlameGraph tools path
FLAMEGRAPH_DIR = Path.home() / "Programs" / "FlameGraph"
STACKCOLLAPSE = FLAMEGRAPH_DIR / "stackcollapse-perf.pl"
FLAMEGRAPH = FLAMEGRAPH_DIR / "flamegraph.pl"

ROOT = Path(__file__).parent.parent
BUILD_DIR = ROOT / "build"

def run_cmd(cmd, check=True, allow_fail=False):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=(check and not allow_fail))
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        # Only print stderr if not allowing failure or if check failed
        if allow_fail and result.returncode == 0:
            pass
        else:
            print(result.stderr, file=sys.stderr)
    return result

def main():
    parser = argparse.ArgumentParser(description="Profile C-Kernel with perf")
    parser.add_argument("--output", type=str, default="perf_profile",
                       help="Output prefix for perf data and flamegraph")
    parser.add_argument("--event", type=str, default="cycles,cache-misses",
                       help="Perf events to record (comma-separated)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Just show commands, don't run them")
    parser.add_argument("--iters", type=int, default=100,
                       help="Number of times to run (we'll use a loop in wrapper script)")
    args = parser.parse_args()

    # Paths
    litmus_c = BUILD_DIR / "litmus_generated.c"
    litmus_bin = BUILD_DIR / "litmus_profile"
    perf_data = BUILD_DIR / f"{args.output}.perf"

    if not litmus_c.exists():
        print(f"Error: {litmus_c} not found. Run 'make litmus' first.")
        return 1

    # Find kernel sources (read from .kernels manifest)
    kernels_manifest = BUILD_DIR / "litmus_generated.c.kernels"
    kernel_sources = []
    if kernels_manifest.exists():
        with open(kernels_manifest, 'r') as f:
            kernel_sources = [line.strip() for line in f if line.strip()]
    else:
        print("Warning: .kernels manifest not found, will try without")

    # Build command
    build_cmd = [
        "gcc", "-O3", "-g", "-march=native",
        "-Iinclude",
        str(litmus_c),
        "-o", str(litmus_bin)
    ]
    if kernel_sources:
        build_cmd.extend(kernel_sources)
    build_cmd.extend(["-lm"])

    if args.dry_run:
        print(f"BUILD CMD: {' '.join(build_cmd)}")
    else:
        # Compile
        print("\n" + "="*60)
        print("COMPILING WITH OPTIMIZATIONS")
        print("="*60)
        run_cmd(build_cmd)

    # Run perf record
    # Note: perf often requires root, so we'll try and show instructions if it fails
    perf_cmd = [
        "perf", "record",
        "-F", "999",  # Lower frequency for better accuracy
        "-e", args.event,
        "-g",  # Call graph
        "-o", str(perf_data),
        "--", str(litmus_bin),
        "--litmus"
    ]

    if args.dry_run:
        print(f"PERF CMD: {' '.join(perf_cmd)}")
    else:
        print("\n" + "="*60)
        print("RECORDING PERF DATA")
        print("="*60)
        print("NOTE: perf usually requires root permissions.")
        print("If this fails, try: sudo perf record ...")
        print()
        try:
            run_cmd(perf_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\nPerf failed (exit code {e.returncode}). This is likely a permissions issue.")
            print(f"Try running: sudo perf record -F 999 -e {args.event} -g -o {perf_data} -- {litmus_bin} --litmus")
            print("\nOr run the binary directly to test it:")
            print(f"  {litmus_bin} --litmus")
            return 1

    # Generate flamegraph
    flame_svg = BUILD_DIR / f"{args.output}.svg"
    perf_script_cmd = ["perf", "script", "-i", str(perf_data)]
    stackcollapse_cmd = [str(STACKCOLLAPSE)]
    flamegraph_cmd = [str(FLAMEGRAPH)]

    if args.dry_run:
        print(f"PERF SCRIPT: {' '.join(perf_script_cmd)} | {' '.join(stackcollapse_cmd)} | {' '.join(flamegraph_cmd)} > {flame_svg}")
    else:
        print("\n" + "="*60)
        print("GENERATING FLAMEGRAPH")
        print("="*60)

        # Create the flamegraph
        with open(perf_data, 'r') as perf_file:
            collapse_proc = subprocess.Popen(
                stackcollapse_cmd,
                stdin=perf_file,
                stdout=subprocess.PIPE,
                text=True
            )

            with open(flame_svg, 'w') as flame_file:
                subprocess.run(
                    flamegraph_cmd,
                    stdin=collapse_proc.stdout,
                    stdout=flame_file,
                    check=True
                )

        print(f"\nFlamegraph saved to: {flame_svg}")
        print(f"\nOpen in browser: file://{flame_svg.absolute()}")

    # Summary
    print("\n" + "="*60)
    print("PROFILING SUMMARY")
    print("="*60)
    print(f"Binary: {litmus_bin}")
    print(f"Perf data: {perf_data}")
    print(f"Flamegraph: {flame_svg}")
    print(f"Iterations: 1 (single litmus run)")
    print(f"Events: {args.event}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
