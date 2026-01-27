#!/usr/bin/env python3
"""
diagnose_v6_6.py - Diagnostic Tool for CK-Engine v6.6 Build Issues

Checks for common problems:
1. Context length mismatch between layout and generated code
2. Interleaved memory layout issues
3. Stale cached files
4. Missing or corrupted intermediate files

Usage:
    python diagnose_v6_6.py /path/to/model/cache/dir
    python diagnose_v6_6.py ~/.ck_cache/models/Qwen--Qwen2-0.5B-Instruct-GGUF
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime


def format_bytes(size: int) -> str:
    """Format byte size to human readable."""
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    elif size >= 1024**2:
        return f"{size / 1024**2:.2f} MB"
    elif size >= 1024:
        return f"{size / 1024:.2f} KB"
    return f"{size} B"


def color(text: str, code: str) -> str:
    """Color text for terminal output."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }
    return f"{colors.get(code, '')}{text}{colors['reset']}"


def check_file_exists(path: Path, name: str) -> bool:
    """Check if file exists and report."""
    if path.exists():
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        size = path.stat().st_size
        print(f"  {color('[OK]', 'green')} {name}: {path.name} ({format_bytes(size)}, modified: {mtime})")
        return True
    else:
        print(f"  {color('[MISSING]', 'red')} {name}: {path.name}")
        return False


def load_json(path: Path) -> dict:
    """Load JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  {color('[ERROR]', 'red')} Failed to load {path}: {e}")
        return {}


def check_context_length(cache_dir: Path) -> list:
    """Check for context length mismatches."""
    issues = []

    layout_files = list(cache_dir.glob("layout_*.json"))
    lowered_files = list(cache_dir.glob("lowered_*.json"))
    c_files = list(cache_dir.glob("*.c"))

    # Extract context lengths from different sources
    context_lengths = {}

    for lf in layout_files:
        layout = load_json(lf)
        if layout:
            ctx = layout.get("config", {}).get("context_length")
            if ctx:
                context_lengths[f"{lf.name}"] = int(ctx)

    for lf in lowered_files:
        ir = load_json(lf)
        if ir:
            ctx = ir.get("config", {}).get("context_length")
            if ctx:
                context_lengths[f"{lf.name}"] = int(ctx)

    # Check C files for MAX_SEQ_LEN
    for cf in c_files:
        try:
            content = cf.read_text()
            match = re.search(r'#define\s+\w*MAX_SEQ_LEN\s+(\d+)', content)
            if match:
                context_lengths[f"{cf.name} (MAX_SEQ_LEN)"] = int(match.group(1))
        except:
            pass

    if len(set(context_lengths.values())) > 1:
        issues.append(f"Context length mismatch across files: {context_lengths}")

    return context_lengths, issues


def check_memory_layout(cache_dir: Path) -> list:
    """Check memory layout for issues."""
    issues = []

    for layout_file in cache_dir.glob("layout_*.json"):
        layout = load_json(layout_file)
        if not layout:
            continue

        arena = layout.get("memory", {}).get("arena", {})
        weights = layout.get("memory", {}).get("weights", {})
        activations = layout.get("memory", {}).get("activations", {})

        mode = arena.get("mode", "unknown")
        weights_base = arena.get("weights_base", 0)
        weights_size = weights.get("size", 0)
        act_base = arena.get("activations_base", 0)
        total_size = arena.get("total_size", 0)

        print(f"\n  Memory layout ({layout_file.name}):")
        print(f"    Mode: {mode}")
        print(f"    Weights: {format_bytes(weights_size)} @ offset 0x{weights_base:X}")
        print(f"    Activations base: 0x{act_base:X}")
        print(f"    Total size: {format_bytes(total_size)}")

        # Check for interleaving
        if act_base < weights_base + weights_size:
            issues.append(f"{layout_file.name}: INTERLEAVED LAYOUT - activations (0x{act_base:X}) start before weights end (0x{weights_base + weights_size:X})")
            print(f"    {color('[WARNING]', 'yellow')} INTERLEAVED LAYOUT DETECTED!")

        # Check for excessive memory
        if total_size > 20 * 1024**3:  # > 20GB
            issues.append(f"{layout_file.name}: Excessive memory ({format_bytes(total_size)}) - check context_length")
            print(f"    {color('[WARNING]', 'yellow')} Excessive memory allocation!")

        # Check buffer overlaps
        all_regions = []
        for e in weights.get("entries", []):
            all_regions.append((e.get("offset", 0), e.get("offset", 0) + e.get("size", 0), f"W:{e.get('name')}"))
        for b in activations.get("buffers", []):
            all_regions.append((b.get("abs_offset", b.get("offset", 0)), b.get("abs_offset", b.get("offset", 0)) + b.get("size", 0), f"A:{b.get('name')}"))

        all_regions.sort()
        for i in range(len(all_regions) - 1):
            _, end1, name1 = all_regions[i]
            start2, _, name2 = all_regions[i + 1]
            if start2 < end1:
                issues.append(f"{layout_file.name}: Buffer overlap: {name1} and {name2}")

    return issues


def check_generated_code(cache_dir: Path) -> list:
    """Check generated C code for issues."""
    issues = []

    for c_file in cache_dir.glob("*.c"):
        try:
            content = c_file.read_text()

            # Check for hardcoded offsets that might be wrong
            offset_matches = re.findall(r'activations\s*\+\s*(\d+)', content)
            if offset_matches:
                offsets = [int(m) for m in offset_matches]
                max_offset = max(offsets) if offsets else 0
                print(f"\n  Generated code ({c_file.name}):")
                print(f"    Found {len(offsets)} activation offset references")
                print(f"    Max offset: 0x{max_offset:X} ({format_bytes(max_offset)})")

                # Check if offsets are suspiciously large
                if max_offset > 10 * 1024**3:  # > 10GB
                    issues.append(f"{c_file.name}: Very large activation offsets (max: {format_bytes(max_offset)})")
                    print(f"    {color('[WARNING]', 'yellow')} Large offsets suggest wrong context_length!")

            # Check for NULL pointer issues
            null_patterns = re.findall(r'NULL.*NULL|NULL, NULL|NULL,\s*NULL', content)
            if len(null_patterns) > 10:
                issues.append(f"{c_file.name}: Many NULL pointer patterns found ({len(null_patterns)})")

        except Exception as e:
            issues.append(f"Failed to analyze {c_file.name}: {e}")

    return issues


def suggest_fixes(issues: list) -> list:
    """Generate fix suggestions based on issues."""
    fixes = []

    for issue in issues:
        if "context length mismatch" in issue.lower():
            fixes.append("Run with --force-compile to regenerate all files with consistent context_length")
        if "interleaved" in issue.lower():
            fixes.append("Use --layout-mode=region to separate weights and activations")
        if "excessive memory" in issue.lower() or "large offsets" in issue.lower():
            fixes.append("Check --context-len parameter - it may be using model default (32768) instead of your value")
            fixes.append("Delete cache and rebuild: rm -rf ~/.ck_cache/models/<model> && python ck_run_v6_6.py run ... --force-compile")

    return list(set(fixes))


def main():
    parser = argparse.ArgumentParser(description="Diagnose CK-Engine v6.6 build issues")
    parser.add_argument("cache_dir", help="Path to model cache directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"Error: Cache directory not found: {cache_dir}")
        return 1

    print(f"\n{'='*70}")
    print(f" CK-Engine v6.6 Diagnostic Report")
    print(f" Cache directory: {cache_dir}")
    print(f"{'='*70}")

    all_issues = []

    # Check required files
    print(f"\n{color('1. File Check', 'blue')}")
    required_files = [
        ("weights_manifest.json", "Model manifest"),
        ("model.bump", "Weight file"),
        ("layout_prefill.json", "Prefill layout"),
        ("layout_decode.json", "Decode layout"),
        ("lowered_prefill.json", "Prefill IR"),
        ("lowered_decode.json", "Decode IR"),
        ("ck-kernel-inference.c", "Generated decode code"),
    ]
    for filename, desc in required_files:
        check_file_exists(cache_dir / filename, desc)

    # Check context lengths
    print(f"\n{color('2. Context Length Check', 'blue')}")
    context_lengths, ctx_issues = check_context_length(cache_dir)
    all_issues.extend(ctx_issues)

    if context_lengths:
        print("  Context lengths found:")
        for source, length in sorted(context_lengths.items()):
            print(f"    {source}: {length:,}")

    # Check memory layout
    print(f"\n{color('3. Memory Layout Check', 'blue')}")
    layout_issues = check_memory_layout(cache_dir)
    all_issues.extend(layout_issues)

    # Check generated code
    print(f"\n{color('4. Generated Code Check', 'blue')}")
    code_issues = check_generated_code(cache_dir)
    all_issues.extend(code_issues)

    # Summary
    print(f"\n{'='*70}")
    print(f" Summary")
    print(f"{'='*70}")

    if all_issues:
        print(f"\n{color('Issues Found:', 'red')}")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")

        fixes = suggest_fixes(all_issues)
        if fixes:
            print(f"\n{color('Suggested Fixes:', 'green')}")
            for i, fix in enumerate(fixes, 1):
                print(f"  {i}. {fix}")
    else:
        print(f"\n{color('No issues found!', 'green')}")

    # Quick command to rebuild
    print(f"\n{color('Quick Rebuild Command:', 'blue')}")
    print(f"  rm -rf {cache_dir}")
    print(f"  python version/v6.6/scripts/ck_run_v6_6.py run <model> --context-len <your_length> --force-compile --layout-mode=region")

    return 1 if all_issues else 0


if __name__ == "__main__":
    sys.exit(main())
