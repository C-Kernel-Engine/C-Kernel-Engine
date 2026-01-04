#!/usr/bin/env python3
"""
test_v4_q4k_pipeline.py
=======================

End-to-end sanity pipeline for v4 Q4_K:
  1) Convert GGUF -> bump (1 layer)
  2) Build IR v4 (1 layer) + compile libmodel.so
  3) Run smoke test + PyTorch parity (random weights)
  4) Repeat for 2 layers
  5) Optionally convert full model if --full is set
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd, verbose=False, env=None) -> None:
    if verbose:
        print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def compile_generated(out_dir: Path, verbose: bool) -> Path:
    c_files = sorted(out_dir.glob("generated_*_decode.c"))
    if not c_files:
        raise SystemExit(f"No generated decode C file found in {out_dir}")
    c_file = c_files[0]
    so_path = out_dir / "libmodel.so"
    cmd = [
        "gcc",
        "-shared",
        "-fPIC",
        "-O2",
        "-Iinclude",
        f"-I{out_dir}",
        str(c_file),
        "-Lbuild",
        "-lckernel_engine",
        "-lm",
        "-o",
        str(so_path),
    ]
    run(cmd, verbose)
    return so_path


def run_case(gguf: Path, layers: int, validate: bool, verbose: bool, run_parity: bool) -> None:
    out_dir = ROOT / "build" / f"v4_q4k_l{layers}"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = out_dir / "weights.bump"
    manifest = out_dir / "weights_manifest_input.json"
    config = out_dir / "config.json"

    convert_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "convert_gguf_to_bump_v4.py"),
        "--gguf",
        str(gguf),
        "--output",
        str(weights),
        "--manifest-out",
        str(manifest),
        "--config-out",
        str(config),
        "--max-layers",
        str(layers),
    ]
    if validate:
        convert_cmd.append("--validate")
    run(convert_cmd, verbose)

    ir_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_ir_v4.py"),
        "--config",
        str(config),
        "--name",
        f"v4_q4k_l{layers}",
        "--prefix",
        str(out_dir),
        "--modes",
        "decode",
        "--dtype",
        "fp32",
        "--weight-dtype",
        "q4_k",
        "--weights-manifest",
        str(manifest),
        "--max-layers",
        str(layers),
        "--emit",
        "lib",
    ]
    run(ir_cmd, verbose)

    compile_generated(out_dir, verbose)

    run_env = os.environ.copy()
    lib_dir = str(ROOT / "build")
    run_env["LD_LIBRARY_PATH"] = lib_dir + (
        (":" + run_env["LD_LIBRARY_PATH"]) if run_env.get("LD_LIBRARY_PATH") else ""
    )

    smoke_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ck_model_smoke_v4.py"),
        "--model-dir",
        str(out_dir),
        "--weights",
        str(weights),
        "--prompt-len",
        "4",
        "--decode-steps",
        "2",
    ]
    run(smoke_cmd, verbose, env=run_env)

    if run_parity:
        parity_cmd = [
            sys.executable,
            str(ROOT / "unittest" / "test_multi_layer_parity.py"),
            "--layers",
            str(layers),
        ]
        run(parity_cmd, verbose, env=run_env)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v4 Q4_K pipeline tests.")
    ap.add_argument("--gguf", required=True, help="Path to GGUF model file")
    ap.add_argument("--layers", default="1,2", help="Layer counts to test (comma-separated)")
    ap.add_argument("--validate", action="store_true", help="Enable converter validation checks")
    ap.add_argument("--no-parity", action="store_true", help="Skip PyTorch parity test")
    ap.add_argument("--full", action="store_true", help="Convert full model after 1/2 layer checks")
    ap.add_argument("--verbose", action="store_true", help="Print subprocess commands")
    args = ap.parse_args()

    gguf = Path(args.gguf)
    if not gguf.exists():
        raise SystemExit(f"GGUF file not found: {gguf}")

    layer_counts = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    if not layer_counts:
        raise SystemExit("--layers must include at least one integer")

    run_parity = not args.no_parity

    for layers in layer_counts:
        print(f"[v4-q4k] testing {layers} layer(s)")
        run_case(gguf, layers, args.validate, args.verbose, run_parity)

    if args.full:
        out_dir = ROOT / "build" / "v4_q4k_full"
        out_dir.mkdir(parents=True, exist_ok=True)
        weights = out_dir / "weights.bump"
        manifest = out_dir / "weights_manifest_input.json"
        config = out_dir / "config.json"

        convert_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "convert_gguf_to_bump_v4.py"),
            "--gguf",
            str(gguf),
            "--output",
            str(weights),
            "--manifest-out",
            str(manifest),
            "--config-out",
            str(config),
        ]
        if args.validate:
            convert_cmd.append("--validate")
        run(convert_cmd, args.verbose)

        ir_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "build_ir_v4.py"),
            "--config",
            str(config),
            "--name",
            "v4_q4k_full",
            "--prefix",
            str(out_dir),
            "--modes",
            "decode",
            "--dtype",
            "fp32",
            "--weight-dtype",
            "q4_k",
            "--weights-manifest",
            str(manifest),
            "--emit",
            "lib",
        ]
        run(ir_cmd, args.verbose)
        compile_generated(out_dir, args.verbose)

    print("[v4-q4k] done")


if __name__ == "__main__":
    main()
