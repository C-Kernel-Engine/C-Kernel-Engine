#!/usr/bin/env python3
"""
DSL Planner Visualizer Generator for C-Kernel-Engine v7

Reads probe_report.json and probe_autopsy.json from a training run directory
and generates a standalone HTML visualizer with embedded data.

Usage:
    python3 version/v7/tools/open_dsl_planner.py --run <run_dir>
    python3 version/v7/tools/open_dsl_planner.py --run <run_dir> --output report.html
    python3 version/v7/tools/open_dsl_planner.py --run <run_dir> --open
    python3 version/v7/tools/open_dsl_planner.py --list

Examples:
    python3 version/v7/tools/open_dsl_planner.py --run ~/.cache/ck-engine-v7/models/train/spec19_scene_bundle_l3_d192_h384_ctx768_r3c_cumulative_neighbors
    python3 version/v7/tools/open_dsl_planner.py --list
"""

import os
import sys
import json
import html as html_mod
import argparse
import webbrowser
import glob as globmod
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
V7_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = V7_ROOT.parent.parent
VISUALIZER = SCRIPT_DIR / "dsl_planner_visualizer.html"

DEFAULT_CACHE = Path.home() / ".cache" / "ck-engine-v7" / "models" / "train"


def find_probe_files(run_dir: Path) -> tuple[Path | None, Path | None]:
    """Find probe_report.json and probe_autopsy.json in a run directory."""
    report = None
    autopsy = None

    for f in run_dir.iterdir():
        name = f.name
        if name.endswith("_probe_report.json") or name == "probe_report.json":
            report = f
        elif name.endswith("_probe_autopsy.json") or name == "probe_autopsy.json":
            autopsy = f

    return report, autopsy


def list_runs():
    """List available training runs with probe data."""
    if not DEFAULT_CACHE.exists():
        print(f"Cache directory not found: {DEFAULT_CACHE}")
        return

    runs = []
    for d in sorted(DEFAULT_CACHE.iterdir()):
        if not d.is_dir():
            continue
        report, autopsy = find_probe_files(d)
        if report:
            runs.append((d.name, bool(report), bool(autopsy)))

    if not runs:
        print("No runs with probe data found.")
        return

    print(f"\n  {'Run Name':<70} {'Report':>8} {'Autopsy':>8}")
    print(f"  {'─' * 70} {'─' * 8} {'─' * 8}")
    for name, has_report, has_autopsy in runs:
        r = "✓" if has_report else "✗"
        a = "✓" if has_autopsy else "✗"
        print(f"  {name:<70} {r:>8} {a:>8}")
    print()


def generate(run_dir: Path, output_path: Path | None = None, auto_open: bool = False):
    """Generate standalone HTML report from probe data."""
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    report_path, autopsy_path = find_probe_files(run_dir)

    if not report_path:
        print(f"Error: No probe_report.json found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"  Run: {run_dir.name}")
    print(f"  Report: {report_path.name}")

    with open(report_path) as f:
        probe_report = json.load(f)

    probe_autopsy = None
    if autopsy_path:
        print(f"  Autopsy: {autopsy_path.name}")
        with open(autopsy_path) as f:
            probe_autopsy = json.load(f)
    else:
        print("  Autopsy: not found (visualization will be partial)")

    # Build embedded data payload
    data = {
        "probe_report": probe_report,
        "probe_autopsy": probe_autopsy,
        "generated_by": "open_dsl_planner.py",
        "generated_at": datetime.now().isoformat(),
    }

    # Read visualizer template
    if not VISUALIZER.exists():
        print(f"Error: Visualizer template not found: {VISUALIZER}", file=sys.stderr)
        sys.exit(1)

    with open(VISUALIZER) as f:
        html = f.read()

    # Embed data — same technique as ir_visualizer
    data_json = json.dumps(data)
    data_json = data_json.replace("</", "<\\/")
    data_json = data_json.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

    data_js = (
        f"window.__DSL_PLANNER_DATA__ = {data_json};"
        "window.dispatchEvent(new Event('ckDslPlannerDataLoaded'));"
    )
    html = html.replace("</body>", f"<script>{data_js}</script></body>")

    # Update title
    run_name = probe_report.get("run_name", run_dir.name)
    html = html.replace(
        "<title>DSL Planner | C-Kernel-Engine</title>",
        f"<title>DSL Planner | {html_mod.escape(run_name)} | C-Kernel-Engine</title>",
    )

    # Output path
    if output_path is None:
        output_path = run_dir / "dsl_planner_report.html"

    with open(output_path, "w") as f:
        f.write(html)

    total = probe_report.get("totals", {}).get("count", 0)
    exact_rate = probe_report.get("totals", {}).get("exact_rate", 0)
    exact = round(exact_rate * total)
    misses = (probe_autopsy or {}).get("miss_cases", total - exact)

    print(f"\n  ✓ Generated: {output_path}")
    print(f"  ✓ {exact}/{total} exact ({exact_rate*100:.1f}%), {misses} misses")
    print(f"  ✓ Size: {output_path.stat().st_size / 1024:.0f} KB")

    if auto_open:
        url = output_path.as_uri()
        print(f"  → Opening: {url}")
        webbrowser.open(url)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="DSL Planner Visualizer — C-Kernel-Engine v7",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 version/v7/tools/open_dsl_planner.py --list
    python3 version/v7/tools/open_dsl_planner.py --run ~/.cache/ck-engine-v7/models/train/spec19_...
    python3 version/v7/tools/open_dsl_planner.py --run spec19_scene_bundle_l3_d192_h384_ctx768_r3c_cumulative_neighbors
    python3 version/v7/tools/open_dsl_planner.py --run spec19_... --open
    python3 version/v7/tools/open_dsl_planner.py --run spec19_... --output ~/reports/spec19.html
        """,
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available training runs with probe data.",
    )
    parser.add_argument(
        "--run", "-r",
        type=str,
        help="Training run directory (full path or name under default cache).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output HTML file path. Defaults to <run_dir>/dsl_planner_report.html.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated report in a browser.",
    )

    args = parser.parse_args()

    if args.list:
        list_runs()
        return

    if not args.run:
        parser.print_help()
        return

    run_dir = Path(args.run)
    if not run_dir.is_absolute() and not run_dir.exists():
        # Try as a name under the default cache
        candidate = DEFAULT_CACHE / args.run
        if candidate.exists():
            run_dir = candidate

    output_path = Path(args.output) if args.output else None
    generate(run_dir, output_path, auto_open=args.open)


if __name__ == "__main__":
    main()
