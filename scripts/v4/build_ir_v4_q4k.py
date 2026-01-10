#!/usr/bin/env python3
"""
build_ir_v4_q4k.py

Generate v4 IR + codegen for Q4_K weights (FP32 activations).
This is a convenience wrapper around:
  1) convert_hf_to_bump_v4.py --dtype q4_k --manifest-out
  2) build_ir_v4.py --dtype fp32 --weights-manifest

It can also convert GGUF inputs via convert_gguf_to_bump_v4.py.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPT_DIR.parent
_V3_DIR = _ROOT_DIR / "v3"
_V4_DIR = _SCRIPT_DIR
for path in (_V4_DIR, _ROOT_DIR, _V3_DIR):
    if path.is_dir():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

import build_ir_v3 as v3
import build_ir_v4 as v4


def run(cmd, verbose=False):
    if verbose:
        print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Build IR v4 for Q4_K weights (HF checkpoint).")
    ap.add_argument("--checkpoint", help="HF checkpoint directory (local)")
    ap.add_argument("--gguf", help="Input GGUF model (Q4_K/Q6_K/F32/F16/BF16)")
    ap.add_argument("--config", help="Optional config.json (defaults to checkpoint/config.json)")
    ap.add_argument("--preset", help="Preset name (qwen2-0.5b, smollm-135)")
    ap.add_argument("--prefix", help="Output directory (default: build/<name>_v4_q4k)")
    ap.add_argument("--modes", default="prefill,decode", help="Modes to emit (default: prefill,decode)")
    ap.add_argument("--tokens", type=int, help="Override tokens for prefill/backward")
    ap.add_argument("--context", type=int, help="Override context length for weights conversion")
    ap.add_argument("--fusion", default="off", choices=["on", "off", "auto"], help="Fusion mode (default: off)")
    ap.add_argument("--emit", default="lib", choices=["lib", "exe"], help="Emit lib or exe (default: lib)")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    args = ap.parse_args()

    if args.preset and args.config:
        raise SystemExit("Use either --preset or --config, not both.")
    if bool(args.checkpoint) == bool(args.gguf):
        raise SystemExit("Specify exactly one of --checkpoint or --gguf.")
    if args.gguf and args.preset:
        raise SystemExit("--preset is only supported with --checkpoint.")

    config_path = None
    model_name = None
    if args.preset:
        preset = v4.PRESETS.get(args.preset)
        if not preset:
            raise SystemExit(f"Unknown preset: {args.preset}")
        config_path = preset["config"]
        model_name = preset.get("name")
    elif args.config:
        config_path = args.config

    if args.gguf:
        base = os.path.basename(args.gguf)
        model_name = os.path.splitext(base)[0]
    elif not model_name:
        model_name = v3.model_id_to_name(os.path.basename(args.checkpoint))

    safe_name = model_name.replace("-", "_").replace(".", "_")
    output_dir = args.prefix or os.path.join("build", f"{safe_name}_v4_q4k")

    os.makedirs(output_dir, exist_ok=True)

    weights_path = os.path.join(output_dir, "weights_v4_q4k.bump")
    manifest_path = os.path.join(output_dir, "weights_v4_q4k.manifest.json")
    config_out = os.path.join(output_dir, "config.json")

    if args.gguf:
        convert_cmd = [
            sys.executable, "scripts/v4/convert_gguf_to_bump_v4.py",
            "--gguf", args.gguf,
            "--output", weights_path,
            "--manifest-out", manifest_path,
            "--config-out", config_out,
        ]
        if args.context:
            convert_cmd.extend(["--context", str(args.context)])
        run(convert_cmd, args.verbose)
        if not config_path:
            config_path = config_out
    else:
        if not config_path:
            config_path = os.path.join(args.checkpoint, "config.json")
        if not os.path.exists(config_path):
            raise SystemExit(f"config.json not found: {config_path}")
        convert_cmd = [
            sys.executable, "scripts/v4/convert_hf_to_bump_v4.py",
            "--checkpoint", args.checkpoint,
            "--output", weights_path,
            "--dtype", "q4_k",
            "--manifest-out", manifest_path,
        ]
        if args.config:
            convert_cmd.extend(["--config", args.config])
        if args.context:
            convert_cmd.extend(["--context", str(args.context)])
        run(convert_cmd, args.verbose)

    cfg = v3.parse_config(config_path)
    embed_dim = cfg.get("embed_dim")
    intermediate = cfg.get("intermediate_dim")

    if embed_dim % 256 != 0 or intermediate % 256 != 0:
        raise SystemExit(
            f"Q4_K requires embed_dim and intermediate_size to be multiples of 256 "
            f"(got embed_dim={embed_dim}, intermediate={intermediate})."
        )

    ir_cmd = [
        sys.executable, "scripts/v4/build_ir_v4.py",
        "--config", config_path,
        "--name", model_name,
        "--prefix", output_dir,
        "--dtype", "fp32",
        "--weight-dtype", "q4_k",
        "--fusion", args.fusion,
        "--emit", args.emit,
        "--modes", args.modes,
        "--weights-manifest", manifest_path,
    ]
    if args.tokens:
        ir_cmd.extend(["--tokens", str(args.tokens)])

    run(ir_cmd, args.verbose)

    print(f"[ok] Q4_K v4 artifacts written to {output_dir}")
    print(f"[ok] weights: {weights_path}")
    print(f"[ok] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
