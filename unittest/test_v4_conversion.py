#!/usr/bin/env python3
"""
V4 conversion sanity test (GGUF -> bump -> IR).

This is an integration-style test that:
  1) Converts a GGUF file to v4 bump weights with --max-layers
  2) Generates v4 IR/codegen from the emitted config + manifest
  3) Verifies layer counts and manifest coverage

Usage:
  CK_TEST_GGUF=/path/to/model.gguf python unittest/test_v4_conversion.py
  python unittest/test_v4_conversion.py --gguf /path/to/model.gguf --layers 1,2 --validate
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from test_utils import print_system_info


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd, verbose=False):
    if verbose:
        print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_num_layers(cfg: dict) -> int:
    for key in ("num_hidden_layers", "num_layers"):
        if key in cfg:
            return int(cfg[key])
    raise AssertionError("Config missing num_layers/num_hidden_layers")


def ensure_layers(label: str, got: int, expected: int) -> None:
    if got != expected:
        raise AssertionError(f"{label}: expected {expected} layers, got {got}")


def run_case(gguf_path: Path, layers: int, validate: bool, verbose: bool) -> None:
    out_dir = PROJECT_ROOT / "build" / f"v4_conversion_l{layers}"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / "weights.bump"
    manifest_in = out_dir / "weights_manifest_input.json"
    config_path = out_dir / "config.json"

    convert_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "v4" / "convert_gguf_to_bump_v4.py"),
        "--gguf",
        str(gguf_path),
        "--output",
        str(weights_path),
        "--manifest-out",
        str(manifest_in),
        "--config-out",
        str(config_path),
        "--max-layers",
        str(layers),
    ]
    if validate:
        convert_cmd.append("--validate")
    run(convert_cmd, verbose)

    cfg = load_json(config_path)
    ensure_layers("config.json", get_num_layers(cfg), layers)

    ir_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "v4" / "build_ir_v4.py"),
        "--config",
        str(config_path),
        "--name",
        f"v4_conv_l{layers}",
        "--prefix",
        str(out_dir),
        "--modes",
        "decode",
        "--dtype",
        "fp32",
        "--weights-manifest",
        str(manifest_in),
        "--max-layers",
        str(layers),
        "--emit",
        "lib",
    ]
    run(ir_cmd, verbose)

    layout = load_json(out_dir / "layout_decode.json")
    sections = layout.get("sections", [])
    if not sections:
        raise AssertionError("layout_decode.json missing sections")
    layer_list = sections[0].get("layers", [])
    ensure_layers("layout_decode.json", len(layer_list), layers)

    merged_manifest_path = out_dir / "weights_manifest.json"
    if merged_manifest_path.exists():
        merged = load_json(merged_manifest_path)
        missing = merged.get("missing", [])
        if missing:
            raise AssertionError(f"weights_manifest.json missing {len(missing)} weights")


def main() -> None:
    ap = argparse.ArgumentParser(description="V4 GGUF conversion sanity test.")
    ap.add_argument("--gguf", help="Path to GGUF model")
    ap.add_argument("--layers", default="1,2", help="Layer counts to test (comma-separated)")
    ap.add_argument("--validate", action="store_true", help="Enable converter validation checks")
    ap.add_argument("--verbose", action="store_true", help="Print subprocess commands")
    args = ap.parse_args()

    gguf_path = args.gguf or os.environ.get("CK_TEST_GGUF") or os.environ.get("CK_TEST_GGUF_PATH")
    if not gguf_path:
        print("[SKIPPED] Set CK_TEST_GGUF=/path/to/model.gguf to run this test.")
        return

    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        raise SystemExit(f"GGUF file not found: {gguf_path}")

    print_system_info()
    print("================================================================================")
    print("  TEST: V4 GGUF Conversion Sanity")
    print("================================================================================")
    print(f"  GGUF:   {gguf_path}")
    print(f"  Layers: {args.layers}")
    if args.validate:
        print("  Validate: on")
    print()

    layer_counts = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    if not layer_counts:
        raise SystemExit("--layers must include at least one integer")

    for layers in layer_counts:
        run_case(gguf_path, layers, args.validate, args.verbose)
        print(f"  [PASS] layers={layers}")

    print("\n  ALL TESTS PASSED")


if __name__ == "__main__":
    main()
