#!/usr/bin/env python3
"""
Compare perf results from baseline and mega-fused attention.
This is THE critical test - does mega-fusion actually reduce DRAM pressure?
"""

import sys
import json
from pathlib import Path


def parse_perf_file(filepath):
    """Parse perf stat output."""
    results = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    value_str = parts[0].replace(',', '')
                    value = float(value_str)
                    metric = parts[-1]
                    if metric not in ['seconds', 'msec', 'usec', 'nsec', 'K/sec', 'M/sec']:
                        results[metric] = value
                except (ValueError, IndexError):
                    continue
    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_perf_results.py <baseline.txt> <megafused.txt>")
        print("\nThis compares DRAM pressure between unfused and mega-fused attention.")
        print("Expected: 10-100x reduction in cache-misses for mega-fused!")
        sys.exit(1)

    baseline_file = Path(sys.argv[1])
    megafused_file = Path(sys.argv[2])

    if not baseline_file.exists():
        print(f"Error: {baseline_file} not found")
        sys.exit(1)

    if not megafused_file.exists():
        print(f"Error: {megafused_file} not found")
        sys.exit(1)

    baseline = parse_perf_file(baseline_file)
    megafused = parse_perf_file(megafused_file)

    print("=" * 70)
    print("DRAM PRESSURE COMPARISON: BASELINE vs MEGA-FUSED")
    print("=" * 70)
    print()

    key_metrics = [
        ('cache-misses', 'Cache Misses'),
        ('LLC-loads', 'L3 Cache Loads'),
        ('cycles', 'CPU Cycles'),
        ('instructions', 'Instructions'),
    ]

    print(f"{'Metric':<25} {'Baseline':>15} {'Mega-Fused':>15} {'Reduction':>12}")
    print("-" * 70)

    results = {}
    for metric, name in key_metrics:
        b = baseline.get(metric, 0)
        m = megafused.get(metric, 0)

        if b > 0:
            reduction = (b - m) / b * 100
            b_str = f"{b/1e6:.2f}M" if b > 1e6 else f"{b/1e3:.2f}K"
            m_str = f"{m/1e6:.2f}M" if m > 1e6 else f"{m/1e3:.2f}K"

            print(f"{name:<25} {b_str:>15} {m_str:>15} {reduction:>8.1f}%")

            if reduction > 50:
                print(f"  {'✓ EXCELLENT! Fusion working!':<40}")
            elif reduction > 0:
                print(f"  {'~ Some improvement':<40}")
            else:
                print(f"  {'✗ WARNING: No improvement!':<40}")

            results[metric] = {'baseline': b, 'megafused': m, 'reduction': reduction}

    print()
    print("=" * 70)
    print("THE CRITICAL METRIC: DRAM Traffic Reduction")
    print("=" * 70)
    print()

    if 'cache-misses' in results:
        cm_reduction = results['cache-misses']['reduction']
        print(f"Cache Miss Reduction: {cm_reduction:.1f}%")
        print()
        print("Interpretation:")
        print("  - Cache misses = DRAM traffic")
        print("  - 10% reduction = 10% less memory bandwidth used")
        print("  - 90% reduction = 10x less memory bandwidth (EXCELLENT)")
        print()

        if cm_reduction > 50:
            print("\033[92mMEGA-FUSION IS WORKING! DRAM traffic significantly reduced!\033[0m")
        elif cm_reduction > 0:
            print("\033[93mPartial improvement detected. Check implementation.\033[0m")
        else:
            print("\033[91mNo improvement. Mega-fusion may not be enabled.\033[0m")

    # Save results
    output = {
        'baseline': baseline,
        'megafused': megafused,
        'comparison': results
    }

    with open("test_results/dram_comparison.json", "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved to: test_results/dram_comparison.json")


if __name__ == "__main__":
    main()
