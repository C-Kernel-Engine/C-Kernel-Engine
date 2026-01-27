#!/usr/bin/env python3
"""
visualize_memory_layout.py - Memory Layout Visualization Tool for CK-Engine v6.6

Reads layout_prefill.json/layout_decode.json and generates:
1. Detailed tables of weights and activations with offsets
2. ASCII memory map showing buffer positions
3. Overlap detection between regions
4. Memory statistics and potential issues

Usage:
    python visualize_memory_layout.py layout_prefill.json
    python visualize_memory_layout.py layout_prefill.json --html output.html
    python visualize_memory_layout.py layout_prefill.json --check-overlaps
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


def format_bytes(size: int) -> str:
    """Format byte size to human readable."""
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    elif size >= 1024**2:
        return f"{size / 1024**2:.2f} MB"
    elif size >= 1024:
        return f"{size / 1024:.2f} KB"
    return f"{size} B"


def load_layout(path: str) -> Dict:
    """Load layout JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def print_table(headers: List[str], rows: List[List[str]], title: str = None):
    """Print a formatted table."""
    if title:
        print(f"\n{'='*80}")
        print(f" {title}")
        print(f"{'='*80}")

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def print_weights_table(layout: Dict):
    """Print weights table."""
    weights = layout.get("memory", {}).get("weights", {})
    entries = weights.get("entries", [])

    headers = ["Name", "Offset", "Size", "End", "Dtype", "Define"]
    rows = []

    for e in sorted(entries, key=lambda x: x.get("offset", 0)):
        offset = e.get("offset", 0)
        size = e.get("size", 0)
        end = offset + size
        rows.append([
            e.get("name", "?"),
            f"0x{offset:X} ({offset:,})",
            format_bytes(size),
            f"0x{end:X}",
            e.get("dtype", "?"),
            e.get("define", "")
        ])

    print_table(headers, rows, f"WEIGHTS ({len(entries)} entries, {format_bytes(weights.get('size', 0))} total)")


def print_activations_table(layout: Dict):
    """Print activations table."""
    activations = layout.get("memory", {}).get("activations", {})
    buffers = activations.get("buffers", [])

    headers = ["Name", "Offset", "Abs Offset", "Size", "End", "Shape", "Dtype"]
    rows = []

    for b in sorted(buffers, key=lambda x: x.get("offset", 0)):
        offset = b.get("offset", 0)
        abs_offset = b.get("abs_offset", offset)
        size = b.get("size", 0)
        end = abs_offset + size
        rows.append([
            b.get("name", "?"),
            f"0x{offset:X}",
            f"0x{abs_offset:X} ({abs_offset:,})",
            format_bytes(size),
            f"0x{end:X}",
            b.get("shape", "?"),
            b.get("dtype", "fp32")
        ])

    print_table(headers, rows, f"ACTIVATIONS ({len(buffers)} buffers, {format_bytes(activations.get('size', 0))} total)")


def print_memory_map(layout: Dict, scale: int = 100):
    """Print ASCII memory map."""
    print(f"\n{'='*80}")
    print(" MEMORY MAP (Visual)")
    print(f"{'='*80}")

    arena = layout.get("memory", {}).get("arena", {})
    total_size = arena.get("total_size", 0)
    weights_base = arena.get("weights_base", 0)
    act_base = arena.get("activations_base", 0)

    # Collect all regions
    regions = []

    # Add weight entries
    for e in layout.get("memory", {}).get("weights", {}).get("entries", []):
        regions.append({
            "name": e.get("name", "?"),
            "start": e.get("offset", 0),
            "end": e.get("offset", 0) + e.get("size", 0),
            "type": "weight",
            "dtype": e.get("dtype", "?")
        })

    # Add activation buffers
    for b in layout.get("memory", {}).get("activations", {}).get("buffers", []):
        regions.append({
            "name": b.get("name", "?"),
            "start": b.get("abs_offset", b.get("offset", 0)),
            "end": b.get("abs_offset", b.get("offset", 0)) + b.get("size", 0),
            "type": "activation",
            "dtype": b.get("dtype", "fp32")
        })

    # Sort by start
    regions = sorted(regions, key=lambda r: r["start"])

    # Print summary bar
    if total_size > 0:
        bar_width = 60
        print(f"\nTotal: {format_bytes(total_size)}")
        print(f"[{'W'*int(bar_width * weights_base / total_size)}{'|'}{'A'*(bar_width - int(bar_width * act_base / total_size))}]")
        print(f" ^Weights                              ^Activations")

    # Print regions
    print(f"\nRegions (sorted by offset):")
    print("-" * 100)

    prev_end = 0
    for r in regions[:50]:  # Limit to first 50 to avoid spam
        # Check for gap
        if r["start"] > prev_end:
            gap = r["start"] - prev_end
            print(f"  [GAP: {format_bytes(gap)}]")

        # Check for overlap
        overlap_marker = ""
        if r["start"] < prev_end:
            overlap = prev_end - r["start"]
            overlap_marker = f" [OVERLAP: {format_bytes(overlap)}!]"

        type_marker = "W" if r["type"] == "weight" else "A"
        print(f"  [{type_marker}] 0x{r['start']:012X} - 0x{r['end']:012X}: {r['name']:<30} ({format_bytes(r['end'] - r['start'])}, {r['dtype']}){overlap_marker}")

        prev_end = max(prev_end, r["end"])

    if len(regions) > 50:
        print(f"  ... and {len(regions) - 50} more regions")


def check_overlaps(layout: Dict) -> List[Tuple[str, str, int]]:
    """Check for overlapping regions."""
    overlaps = []

    # Collect all regions
    regions = []

    for e in layout.get("memory", {}).get("weights", {}).get("entries", []):
        regions.append({
            "name": f"W:{e.get('name', '?')}",
            "start": e.get("offset", 0),
            "end": e.get("offset", 0) + e.get("size", 0)
        })

    for b in layout.get("memory", {}).get("activations", {}).get("buffers", []):
        regions.append({
            "name": f"A:{b.get('name', '?')}",
            "start": b.get("abs_offset", b.get("offset", 0)),
            "end": b.get("abs_offset", b.get("offset", 0)) + b.get("size", 0)
        })

    # Check all pairs
    for i, r1 in enumerate(regions):
        for r2 in regions[i+1:]:
            # Check if they overlap
            if r1["start"] < r2["end"] and r2["start"] < r1["end"]:
                overlap = min(r1["end"], r2["end"]) - max(r1["start"], r2["start"])
                overlaps.append((r1["name"], r2["name"], overlap))

    return overlaps


def print_config(layout: Dict):
    """Print model configuration."""
    config = layout.get("config", {})

    print(f"\n{'='*80}")
    print(" MODEL CONFIGURATION")
    print(f"{'='*80}")

    headers = ["Parameter", "Value"]
    rows = []
    for key, value in sorted(config.items()):
        rows.append([key, str(value)])

    print_table(headers, rows)


def print_arena_summary(layout: Dict):
    """Print arena/allocation summary."""
    arena = layout.get("memory", {}).get("arena", {})
    weights = layout.get("memory", {}).get("weights", {})
    activations = layout.get("memory", {}).get("activations", {})

    print(f"\n{'='*80}")
    print(" ARENA SUMMARY")
    print(f"{'='*80}")

    print(f"Mode: {arena.get('mode', 'unknown')}")
    print(f"Weights base: 0x{arena.get('weights_base', 0):X}")
    print(f"Activations base: 0x{arena.get('activations_base', 0):X}")
    print(f"Total size: {format_bytes(arena.get('total_size', 0))}")
    print()
    print(f"Weights size: {format_bytes(weights.get('size', 0))}")
    print(f"Activations size: {format_bytes(activations.get('size', 0))}")

    # Check if interleaved
    act_base = arena.get("activations_base", 0)
    weights_size = weights.get("size", 0)

    if act_base < weights_size:
        print(f"\n[WARNING] INTERLEAVED LAYOUT DETECTED!")
        print(f"  Activations start at 0x{act_base:X} which is INSIDE the weights region")
        print(f"  Weights extend to 0x{weights_size:X}")
        print(f"  This may cause aliasing issues!")


def generate_html(layout: Dict, output_path: str):
    """Generate HTML visualization."""
    config = layout.get("config", {})
    arena = layout.get("memory", {}).get("arena", {})
    weights = layout.get("memory", {}).get("weights", {})
    activations = layout.get("memory", {}).get("activations", {})

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>CK-Engine Memory Layout Visualization</title>
    <style>
        body {{ font-family: 'Consolas', monospace; margin: 20px; background: #1a1a2e; color: #eee; }}
        h1, h2 {{ color: #0f3460; background: #e94560; padding: 10px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #444; padding: 8px; text-align: left; }}
        th {{ background: #16213e; }}
        tr:nth-child(even) {{ background: #1a1a2e; }}
        tr:nth-child(odd) {{ background: #16213e; }}
        .weight {{ color: #00ff88; }}
        .activation {{ color: #ff6b6b; }}
        .warning {{ background: #ff6b6b; color: white; padding: 5px; }}
        .memory-bar {{ display: flex; height: 30px; margin: 20px 0; border: 2px solid #444; }}
        .memory-bar .weights {{ background: #00ff88; }}
        .memory-bar .activations {{ background: #ff6b6b; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0; }}
        .stat-box {{ background: #16213e; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #e94560; }}
        .region {{ margin: 5px 0; padding: 5px; border-left: 4px solid; }}
        .region.weight {{ border-color: #00ff88; }}
        .region.activation {{ border-color: #ff6b6b; }}
    </style>
</head>
<body>
    <h1>CK-Engine v6.6 Memory Layout</h1>

    <h2>Configuration</h2>
    <div class="stats">
        <div class="stat-box"><div class="stat-value">{config.get('num_layers', 'N/A')}</div>Layers</div>
        <div class="stat-box"><div class="stat-value">{config.get('embed_dim', 'N/A')}</div>Embed Dim</div>
        <div class="stat-box"><div class="stat-value">{config.get('num_heads', 'N/A')}</div>Heads</div>
        <div class="stat-box"><div class="stat-value">{config.get('vocab_size', 'N/A'):,}</div>Vocab Size</div>
        <div class="stat-box"><div class="stat-value">{config.get('context_length', 'N/A'):,}</div>Context Len</div>
        <div class="stat-box"><div class="stat-value">{format_bytes(arena.get('total_size', 0))}</div>Total Memory</div>
        <div class="stat-box"><div class="stat-value">{format_bytes(weights.get('size', 0))}</div>Weights</div>
        <div class="stat-box"><div class="stat-value">{format_bytes(activations.get('size', 0))}</div>Activations</div>
    </div>

    <h2>Memory Layout</h2>
    <div class="memory-bar">
        <div class="weights" style="width: {100 * weights.get('size', 1) / max(arena.get('total_size', 1), 1):.1f}%"></div>
        <div class="activations" style="width: {100 * activations.get('size', 1) / max(arena.get('total_size', 1), 1):.1f}%"></div>
    </div>
    <p>
        <span class="weight">Green: Weights (0x0 - 0x{weights.get('size', 0):X})</span> |
        <span class="activation">Red: Activations (0x{arena.get('activations_base', 0):X} - 0x{arena.get('total_size', 0):X})</span>
    </p>

    {"<div class='warning'>WARNING: Interleaved layout detected - activations overlap with weights region!</div>" if arena.get('activations_base', 0) < weights.get('size', 0) else ""}

    <h2>Weights ({len(weights.get('entries', []))} entries)</h2>
    <table>
        <tr><th>Name</th><th>Offset</th><th>Size</th><th>Dtype</th><th>Define</th></tr>
        {"".join(f"<tr><td>{e.get('name', '?')}</td><td>0x{e.get('offset', 0):X}</td><td>{format_bytes(e.get('size', 0))}</td><td>{e.get('dtype', '?')}</td><td>{e.get('define', '')}</td></tr>" for e in sorted(weights.get('entries', []), key=lambda x: x.get('offset', 0)))}
    </table>

    <h2>Activations ({len(activations.get('buffers', []))} buffers)</h2>
    <table>
        <tr><th>Name</th><th>Abs Offset</th><th>Size</th><th>Shape</th><th>Dtype</th></tr>
        {"".join(f"<tr><td>{b.get('name', '?')}</td><td>0x{b.get('abs_offset', 0):X}</td><td>{format_bytes(b.get('size', 0))}</td><td>{b.get('shape', '?')}</td><td>{b.get('dtype', 'fp32')}</td></tr>" for b in sorted(activations.get('buffers', []), key=lambda x: x.get('abs_offset', 0)))}
    </table>

</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"HTML visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize CK-Engine memory layout")
    parser.add_argument("layout_file", help="Path to layout JSON file")
    parser.add_argument("--html", help="Generate HTML output to this path")
    parser.add_argument("--check-overlaps", action="store_true", help="Check for overlapping regions")
    parser.add_argument("--weights-only", action="store_true", help="Show only weights table")
    parser.add_argument("--activations-only", action="store_true", help="Show only activations table")

    args = parser.parse_args()

    try:
        layout = load_layout(args.layout_file)
    except Exception as e:
        print(f"Error loading layout: {e}", file=sys.stderr)
        return 1

    if args.html:
        generate_html(layout, args.html)
        return 0

    # Print standard output
    print_config(layout)
    print_arena_summary(layout)

    if args.weights_only:
        print_weights_table(layout)
    elif args.activations_only:
        print_activations_table(layout)
    else:
        print_weights_table(layout)
        print_activations_table(layout)
        print_memory_map(layout)

    if args.check_overlaps:
        overlaps = check_overlaps(layout)
        if overlaps:
            print(f"\n{'='*80}")
            print(" OVERLAP WARNINGS")
            print(f"{'='*80}")
            for r1, r2, size in overlaps:
                print(f"  {r1} overlaps with {r2} by {format_bytes(size)}")
        else:
            print("\nNo overlaps detected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
