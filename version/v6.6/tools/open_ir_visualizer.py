#!/usr/bin/env python3
"""
IR Visualizer Launcher for C-Kernel-Engine v6.6

Usage:
    python version/v6.6/tools/open_ir_visualizer.py              # Open visualizer
    python version/v6.6/tools/open_ir_visualizer.py --list       # List available models
    python version/v6.6/tools/open_ir_visualizer.py <model>      # Generate and open report
    python version/v6.6/tools/open_ir_visualizer.py --generate <model>  # Generate only
"""
import sys
import json
import webbrowser
import argparse
from pathlib import Path

# Path construction:
# Script is at: version/v6.6/tools/open_ir_visualizer.py
SCRIPT_DIR = Path(__file__).parent              # .../version/v6.6/tools
V66_ROOT = SCRIPT_DIR.parent                    # .../version/v6.6
PROJECT_ROOT = V66_ROOT.parent                   # .../Workspace/C-Kernel-Engine

sys.path.insert(0, str(V66_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_PATH = Path.home() / ".cache" / "ck-engine-v6.6" / "models"
VISUALIZER = SCRIPT_DIR / "ir_visualizer.html"


def list_available_models():
    """List all models in cache."""
    if not CACHE_PATH.exists():
        return []

    models = []
    for model_dir in CACHE_PATH.iterdir():
        if not model_dir.is_dir():
            continue

        ck_build = model_dir / "ck_build"
        has_data = (
            ck_build.exists() and
            (ck_build / "ir1_decode.json").exists()
        )

        models.append({
            "name": model_dir.name,
            "path": str(ck_build),
            "has_data": has_data
        })

    return sorted(models, key=lambda m: m["name"])


def load_model_data(ck_build_path: Path) -> dict:
    """Load all IR data for a model."""
    # Define required vs optional files
    REQUIRED_FILES = [
        "ir1_decode",
        "layout_decode",
        "lowered_decode_call",
    ]
    OPTIONAL_FILES = [
        "ir1_prefill",
        "layout_prefill",
        "lowered_prefill_call",
        "manifest",
    ]

    data_files = {
        "ir1_decode": ck_build_path / "ir1_decode.json",
        "ir1_prefill": ck_build_path / "ir1_prefill.json",
        "layout_decode": ck_build_path / "layout_decode.json",
        "layout_prefill": ck_build_path / "layout_prefill.json",
        "lowered_decode_call": ck_build_path / "lowered_decode_call.json",
        "lowered_prefill_call": ck_build_path / "lowered_prefill_call.json",
        "manifest": ck_build_path / "weights_manifest.json",
    }

    data = {
        "meta": {
            "model": ck_build_path.parent.name,
            "path": str(ck_build_path),
            "warnings": [],
        },
        "files": {}
    }

    loaded = []
    missing_required = []
    missing_optional = []

    for key, path in data_files.items():
        if path.exists():
            try:
                with open(path, "r") as f:
                    data["files"][key] = json.load(f)
                loaded.append(key)
            except Exception as e:
                print(f"  ! {key}: {e}")
        else:
            if key in REQUIRED_FILES:
                missing_required.append(key)
            else:
                missing_optional.append(key)

    # Report missing files
    if missing_required:
        print(f"  ! Missing required files: {missing_required}")
        data["meta"]["warnings"].append(f"Missing required: {missing_required}")

    if missing_optional:
        print(f"  - Missing optional files: {missing_optional}")

    print(f"  Loaded {len(loaded)} files")
    return data


def generate_html_report(ck_build_path: Path, output_path: Path = None):
    """Generate standalone HTML report."""
    from datetime import datetime

    model_name = ck_build_path.parent.name
    print(f"Generating report for: {model_name}")

    # Load data
    data = load_model_data(ck_build_path)
    data["meta"]["generated_at"] = datetime.now().isoformat()
    data["meta"]["engine_version"] = "v6.6"

    # Read visualizer template
    if not VISUALIZER.exists():
        raise FileNotFoundError(f"Visualizer not found: {VISUALIZER}")

    with open(VISUALIZER, "r") as f:
        html = f.read()

    # Embed data
    data_js = f"window.EMBEDDED_IR_DATA = {json.dumps(data)};"
    html = html.replace('</body>', f'<script>{data_js}</script></body>')

    # Update title
    html = html.replace(
        '<title>IR Visualizer | C-Kernel-Engine</title>',
        f'<title>IR Visualizer | {model_name} | C-Kernel-Engine</title>'
    )

    # Write output
    if output_path is None:
        output_path = ck_build_path / "ir_report.html"

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="IR Visualizer Launcher for C-Kernel-Engine v6.6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python version/v6.6/tools/open_ir_visualizer.py              # Open visualizer
    python version/v6.6/tools/open_ir_visualizer.py --list       # List available models
    python version/v6.6/tools/open_ir_visualizer.py gemma3       # Generate and open report
    python version/v6.6/tools/open_ir_visualizer.py --generate gemma3  # Generate only
        """
    )

    parser.add_argument(
        "model",
        nargs="?",
        help="Model name (e.g., gemma3) or path to ck_build directory"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models in cache"
    )
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Generate HTML report without opening browser"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for generated HTML (default: ck_build/ir_report.html)"
    )

    args = parser.parse_args()

    if args.list:
        print("Available models in cache:")
        for m in list_available_models():
            status = "OK" if m["has_data"] else "no data"
            print(f"  - {m['name']} ({status})")
        return

    if args.model:
        # Determine path
        if Path(args.model).exists():
            ck_build = Path(args.model)
        else:
            ck_build = CACHE_PATH / args.model / "ck_build"
            if not ck_build.exists():
                print(f"Error: Model not found: {args.model}")
                print("\nAvailable models:")
                for m in list_available_models():
                    print(f"  - {m['name']}")
                return

        # Generate report
        output = args.output or ck_build / "ir_report.html"
        report_path = generate_html_report(ck_build, output)
        print(f"\nGenerated: {report_path}")

        if not args.generate:
            webbrowser.open(f"file://{report_path}")
    else:
        # Open visualizer
        if VISUALIZER.exists():
            webbrowser.open(f"file://{VISUALIZER}")
        else:
            print(f"Error: Visualizer not found: {VISUALIZER}")


if __name__ == "__main__":
    main()
