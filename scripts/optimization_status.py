#!/usr/bin/env python3
"""
Optimization Status Report Generator

Reads version/meta/kernel_meta.json and generates reports showing:
- Kernel implementation status by dtype
- Optimization levels achieved
- Inference/training optimization roadmap
- Performance targets vs current state

Usage:
    python3 scripts/optimization_status.py              # Full report
    python3 scripts/optimization_status.py --kernels    # Kernel status only
    python3 scripts/optimization_status.py --inference  # Inference optimizations
    python3 scripts/optimization_status.py --training   # Training optimizations
    python3 scripts/optimization_status.py --pending    # What's not done
    python3 scripts/optimization_status.py --json       # Raw JSON output
    python3 scripts/optimization_status.py --markdown   # Markdown format
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Find meta directory
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
META_FILE = ROOT / "version" / "meta" / "kernel_meta.json"


def load_meta():
    """Load the kernel metadata JSON."""
    if not META_FILE.exists():
        print(f"Error: {META_FILE} not found", file=sys.stderr)
        sys.exit(1)
    with open(META_FILE) as f:
        return json.load(f)


def status_icon(status):
    """Return icon for status."""
    icons = {
        "done": "[OK]",
        "partial": "[~~]",
        "pending": "[..]",
        "not_started": "[  ]",
    }
    return icons.get(status, "[??]")


def status_color(status):
    """Return ANSI color for status."""
    colors = {
        "done": "\033[92m",      # Green
        "partial": "\033[93m",   # Yellow
        "pending": "\033[93m",   # Yellow
        "not_started": "\033[91m",  # Red
    }
    reset = "\033[0m"
    return colors.get(status, ""), reset


def opt_level_short(levels):
    """Abbreviate optimization levels."""
    if not levels:
        return "-"
    abbrevs = {
        "scalar": "S",
        "simd_avx": "A1",
        "simd_avx2": "A2",
        "simd_avx512": "A5",
        "simd_avx512bf16": "BF",
        "simd_amx": "AM",
        "fused": "FU",
        "blocked": "BL",
        "parallel": "||",
    }
    return ",".join(abbrevs.get(l, l[:2].upper()) for l in levels)


def get_best_opt_indicator(opt_levels, compact=False):
    """Return a compact indicator for optimization level.

    If compact=True, show only the highest-tier SIMD + one modifier.
    """
    if not opt_levels:
        return ""

    # SIMD tier (pick highest)
    simd = ""
    if "simd_amx" in opt_levels:
        simd = "AM"
    elif "simd_avx512bf16" in opt_levels:
        simd = "BF"
    elif "simd_avx512" in opt_levels:
        simd = "A5"
    elif "simd_avx2" in opt_levels:
        simd = "A2"
    elif "simd_avx" in opt_levels:
        simd = "A1"

    # Modifiers
    mods = []
    if "blocked" in opt_levels:
        mods.append("BL")
    if "parallel" in opt_levels:
        mods.append("||")
    if "fused" in opt_levels:
        mods.append("FU")

    if compact:
        # Show SIMD + first modifier only
        if simd and mods:
            return f"{simd}+"
        elif simd:
            return simd
        elif mods:
            return mods[0]
        else:
            return "S"
    else:
        # Full list
        indicators = [simd] if simd else []
        indicators.extend(mods)
        if not indicators:
            return "S"
        return ",".join(indicators)


def print_kernel_table(meta, markdown=False):
    """Print kernel implementation status table with optimization indicators."""
    kernels = meta["kernels"]
    dtypes = list(meta["dtypes"].keys())

    if markdown:
        print("\n## Kernel Implementation Status\n")
        print("Legend: F=Forward, B=Backward | S=Scalar, A5=AVX512, BL=Blocked, ||=Parallel\n")
        header = "| Kernel |"
        sep = "|--------|"
        for dt in dtypes:
            header += f" {dt} |"
            sep += ":---:|"
        print(header)
        print(sep)
    else:
        print("\n" + "=" * 100)
        print("  KERNEL IMPLEMENTATION STATUS")
        print("=" * 100)
        print("\nLegend: F=Forward, B=Backward")
        print("Opts:   S=Scalar, A1=AVX1, A2=AVX2, A5=AVX512, BF=BF16, AM=AMX, BL=Blocked, ||=Parallel, FU=Fused\n")

        # Build header with wider column widths for optimization info
        col_width = 11
        header = f"{'Kernel':<12}|"
        separator = "-" * 12 + "+"
        for dt in dtypes:
            header += f"{dt:^{col_width}}|"
            separator += "-" * col_width + "+"
        print(separator)
        print(header)
        print(separator)

    for kname, kinfo in sorted(kernels.items()):
        if markdown:
            row = f"| {kname} |"
        else:
            row = f"{kname:<12}|"

        for dt in dtypes:
            if dt in kinfo.get("variants", {}):
                v = kinfo["variants"][dt]
                fwd = v.get("forward", {})
                bwd = v.get("backward", v.get("backward_dx", {}))

                fwd_done = fwd.get("status") == "done"
                bwd_done = bwd.get("status") == "done"

                fwd_opt = get_best_opt_indicator(fwd.get("opt_level", []), compact=True) if fwd_done else ""
                bwd_opt = get_best_opt_indicator(bwd.get("opt_level", []), compact=True) if bwd_done else ""

                if fwd_done and bwd_done:
                    # Show optimization level if they differ or are interesting
                    if fwd_opt == bwd_opt:
                        cell = f"F/B:{fwd_opt}" if fwd_opt else "F/B"
                    else:
                        cell = f"F:{fwd_opt} B:{bwd_opt}"
                elif fwd_done:
                    cell = f"F:{fwd_opt}" if fwd_opt else "F"
                elif bwd_done:
                    cell = f"B:{bwd_opt}" if bwd_opt else "B"
                else:
                    cell = "-"
            else:
                cell = "-"

            if markdown:
                row += f" {cell} |"
            else:
                row += f"{cell:^{col_width}}|"

        print(row)

    if not markdown:
        print(separator)


def print_optimization_details(meta, markdown=False):
    """Print detailed optimization levels per kernel."""
    kernels = meta["kernels"]
    dtypes = list(meta["dtypes"].keys())

    if markdown:
        print("\n## Optimization Levels\n")
        print("S=Scalar, A1=AVX1, A2=AVX2, A5=AVX512, BF=BF16, AM=AMX, BL=Blocked, ||=Parallel, FU=Fused\n")
    else:
        print("\n" + "=" * 80)
        print("  OPTIMIZATION LEVELS BY KERNEL")
        print("=" * 80)
        print("\nLegend: S=Scalar, A1=AVX1, A2=AVX2, A5=AVX512, BF=BF16, AM=AMX, BL=Blocked, ||=Parallel, FU=Fused\n")

    for kname, kinfo in sorted(kernels.items()):
        has_opts = False
        for dt in dtypes:
            if dt in kinfo.get("variants", {}):
                v = kinfo["variants"][dt]
                fwd = v.get("forward", {})
                bwd = v.get("backward", v.get("backward_dx", {}))
                if fwd.get("opt_level") or bwd.get("opt_level"):
                    has_opts = True
                    break

        if not has_opts:
            continue

        if markdown:
            print(f"\n### {kname}")
        else:
            print(f"\n  {kname.upper()}")
            print("  " + "-" * 40)

        for dt in dtypes:
            if dt not in kinfo.get("variants", {}):
                continue

            v = kinfo["variants"][dt]
            fwd = v.get("forward", {})
            bwd = v.get("backward", v.get("backward_dx", {}))

            fwd_opt = opt_level_short(fwd.get("opt_level", []))
            bwd_opt = opt_level_short(bwd.get("opt_level", []))

            if fwd_opt == "-" and bwd_opt == "-":
                continue

            if markdown:
                print(f"- **{dt}**: Forward={fwd_opt}, Backward={bwd_opt}")
            else:
                print(f"    {dt:<6}  Forward: {fwd_opt:<12}  Backward: {bwd_opt}")


def print_kernel_summary(meta):
    """Print summary statistics for kernels."""
    kernels = meta["kernels"]

    total_possible = 0
    total_done = 0
    by_category = defaultdict(lambda: {"done": 0, "total": 0})

    for kname, kinfo in kernels.items():
        cat = kinfo.get("category", "other")
        for dt, variants in kinfo.get("variants", {}).items():
            for direction in ["forward", "backward", "backward_dx", "backward_dw"]:
                if direction in variants:
                    total_possible += 1
                    by_category[cat]["total"] += 1
                    if variants[direction].get("status") == "done":
                        total_done += 1
                        by_category[cat]["done"] += 1

    print(f"\nKernel Coverage: {total_done}/{total_possible} ({100*total_done/total_possible:.1f}%)")
    print("\nBy Category:")
    for cat, stats in sorted(by_category.items()):
        pct = 100 * stats["done"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {cat:<15}: {stats['done']:>3}/{stats['total']:<3} ({pct:>5.1f}%)")


def print_inference_optimizations(meta, markdown=False):
    """Print inference optimization roadmap."""
    opts = meta.get("inference_optimizations", {})

    if markdown:
        print("\n## Inference Optimization Roadmap\n")
        print("| Optimization | Status | Priority | Dependencies |")
        print("|--------------|--------|----------|--------------|")
    else:
        print("\n" + "=" * 80)
        print("  INFERENCE OPTIMIZATION ROADMAP")
        print("=" * 80)
        print(f"\n{'Optimization':<25} {'Status':<12} {'Priority':<10} Dependencies")
        print("-" * 80)

    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    for name, info in sorted(opts.items(), key=lambda x: priority_order.get(x[1].get("priority", "low"), 4)):
        status = info.get("status", "not_started")
        priority = info.get("priority", "low")
        deps = ", ".join(info.get("depends_on", [])) or "-"
        desc = info.get("description", "")

        if markdown:
            icon = {"done": "done", "partial": "partial", "not_started": "todo"}.get(status, status)
            print(f"| **{name}** | {icon} | {priority} | {deps} |")
        else:
            color, reset = status_color(status)
            icon = status_icon(status)
            print(f"{color}{icon}{reset} {name:<22} {status:<12} {priority:<10} {deps}")

    if not markdown:
        print()


def print_training_optimizations(meta, markdown=False):
    """Print training optimization roadmap."""
    opts = meta.get("training_optimizations", {})

    if markdown:
        print("\n## Training Optimization Roadmap\n")
        print("| Optimization | Status | Priority |")
        print("|--------------|--------|----------|")
    else:
        print("\n" + "=" * 80)
        print("  TRAINING OPTIMIZATION ROADMAP")
        print("=" * 80)
        print(f"\n{'Optimization':<30} {'Status':<12} {'Priority':<10}")
        print("-" * 60)

    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    for name, info in sorted(opts.items(), key=lambda x: priority_order.get(x[1].get("priority", "low"), 4)):
        status = info.get("status", "not_started")
        priority = info.get("priority", "low")

        if markdown:
            print(f"| **{name}** | {status} | {priority} |")
        else:
            color, reset = status_color(status)
            icon = status_icon(status)
            print(f"{color}{icon}{reset} {name:<27} {status:<12} {priority:<10}")


def print_pending_work(meta):
    """Print what's not done yet, prioritized."""
    print("\n" + "=" * 80)
    print("  HIGH-PRIORITY PENDING WORK")
    print("=" * 80)

    # Critical inference optimizations
    print("\n[CRITICAL] Inference:")
    for name, info in meta.get("inference_optimizations", {}).items():
        if info.get("priority") == "critical" and info.get("status") != "done":
            print(f"  - {name}: {info.get('description', '')}")

    # High priority kernel optimizations
    print("\n[HIGH] Kernel Optimizations:")
    for kname, kinfo in meta.get("kernels", {}).items():
        for opt in kinfo.get("optimizations_pending", []):
            if opt.get("priority") in ["critical", "high"]:
                print(f"  - {kname}/{opt['name']}: {opt.get('description', '')}")

    # Missing backward passes
    print("\n[MEDIUM] Missing Backward Passes:")
    for kname, kinfo in meta.get("kernels", {}).items():
        for dt, variants in kinfo.get("variants", {}).items():
            if variants.get("forward", {}).get("status") == "done":
                bwd = variants.get("backward", variants.get("backward_dx", {}))
                if bwd.get("status") != "done":
                    print(f"  - {kname} ({dt})")


def print_single_core_priorities(meta):
    """Print single-core optimization priorities."""
    print("\n" + "=" * 80)
    print("  SINGLE-CORE OPTIMIZATION PRIORITIES")
    print("=" * 80)
    print("\nOptimize single-core first, then parallelize:\n")

    sco = meta.get("single_core_optimizations", {})
    for item in sco.get("priorities", []):
        status = item.get("status", "not_started")
        color, reset = status_color(status)
        icon = status_icon(status)
        print(f"{color}{icon}{reset} {item['name']:<25} {item.get('description', '')}")


def print_performance_targets(meta):
    """Print performance targets."""
    print("\n" + "=" * 80)
    print("  PERFORMANCE TARGETS")
    print("=" * 80)

    targets = meta.get("performance_targets", {})
    print(f"\n{'Component':<20} {'Metric':<15} {'Target':<30} {'Current':<20}")
    print("-" * 90)

    for name, info in targets.items():
        metric = info.get("metric", "")
        target = info.get("target", "")
        current = info.get("current", "unknown")
        print(f"{name:<20} {metric:<15} {target:<30} {current:<20}")


def main():
    parser = argparse.ArgumentParser(description="Optimization status report generator")
    parser.add_argument("--kernels", action="store_true", help="Show kernel status (compact F/B table)")
    parser.add_argument("--optimizations", "--opt", action="store_true", help="Show optimization levels per kernel")
    parser.add_argument("--inference", action="store_true", help="Show inference optimizations")
    parser.add_argument("--training", action="store_true", help="Show training optimizations")
    parser.add_argument("--pending", action="store_true", help="Show pending work")
    parser.add_argument("--single-core", action="store_true", help="Show single-core priorities")
    parser.add_argument("--targets", action="store_true", help="Show performance targets")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--markdown", "--md", action="store_true", help="Output as Markdown")
    args = parser.parse_args()

    meta = load_meta()

    if args.json:
        print(json.dumps(meta, indent=2))
        return

    # If no specific section requested, show all
    show_all = not any([args.kernels, args.optimizations, args.inference, args.training,
                        args.pending, args.single_core, args.targets])

    if not args.markdown:
        print("\n" + "=" * 80)
        print("  C-KERNEL-ENGINE OPTIMIZATION STATUS")
        print("=" * 80)

    if show_all or args.kernels:
        print_kernel_table(meta, markdown=args.markdown)
        if not args.markdown:
            print_kernel_summary(meta)

    if show_all or args.optimizations:
        print_optimization_details(meta, markdown=args.markdown)

    if show_all or args.inference:
        print_inference_optimizations(meta, markdown=args.markdown)

    if show_all or args.training:
        print_training_optimizations(meta, markdown=args.markdown)

    if show_all or args.pending:
        print_pending_work(meta)

    if show_all or args.single_core:
        print_single_core_priorities(meta)

    if show_all or args.targets:
        print_performance_targets(meta)

    print()


if __name__ == "__main__":
    main()
