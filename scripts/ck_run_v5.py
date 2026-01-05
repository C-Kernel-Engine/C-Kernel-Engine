#!/usr/bin/env python3
"""
ck_run_v5.py - C-Kernel-Engine v5 Pipeline Runner

Full v5 pipeline with explicit codegen:
  1. Download from HuggingFace (GGUF)
  2. Convert to bump format + weights manifest
  3. Verify kernel coverage
  4. Build IR v5 (graph → lowered → layout)
  5. Generate v5 explicit C code (per-layer unrolled)
  6. Compile to shared library

Usage:
  python scripts/ck_run_v5.py run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf
  python scripts/ck_run_v5.py run ./model.gguf --weight-dtype=q4_k_m
  python scripts/ck_run_v5.py run MODEL --debug --parity
  python scripts/ck_run_v5.py list
  python scripts/ck_run_v5.py clean

v5 Features:
  - Explicit per-layer kernel calls (gemm_nt_q5_0, gemm_nt_q8_0, etc.)
  - Each layer has separate function (qwen2_layer_0_decode, qwen2_layer_1_decode)
  - Easy to insert debug hooks for PyTorch parity testing
  - --debug: Print buffer stats after each operation
  - --parity: Save layer outputs to .bin files for comparison
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path.home() / ".cache" / "ck-engine-v5" / "models"
SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent

# Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_ORANGE = "\033[38;5;214m"
C_GREEN = "\033[38;5;114m"
C_RED = "\033[38;5;203m"


def log(msg: str, color: str = ""):
    if color:
        print(f"{color}{msg}{C_RESET}")
    else:
        print(msg)


def log_step(step: int, msg: str):
    print(f"{C_ORANGE}[{step}/6]{C_RESET} {C_BOLD}{msg}{C_RESET}")


def log_error(msg: str):
    print(f"{C_RED}Error:{C_RESET} {msg}", file=sys.stderr)


def run_cmd(cmd: list, capture: bool = False):
    try:
        if capture:
            return subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            return subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log_error(f"Command failed: {' '.join(cmd)}")
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Input Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_input(model_input: str) -> tuple[str, dict]:
    """Detect input type: hf_gguf, gguf, local_dir."""
    # hf://org/repo/file.gguf
    if model_input.startswith("hf://") and model_input.endswith(".gguf"):
        parts = model_input[5:].split("/")
        if len(parts) >= 3:
            return "hf_gguf", {"repo_id": f"{parts[0]}/{parts[1]}", "filename": "/".join(parts[2:])}

    # Local GGUF file
    if model_input.endswith(".gguf") and Path(model_input).exists():
        return "gguf", {"path": Path(model_input).resolve()}

    # Local directory
    if Path(model_input).is_dir() and (Path(model_input) / "config.json").exists():
        return "local_dir", {"path": Path(model_input).resolve()}

    # HuggingFace model ID
    if "/" in model_input:
        org, name = model_input.split("/", 1)
        return "hf_id", {"model_id": model_input, "org": org, "name": name}

    log_error(f"Unknown input format: {model_input}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ═══════════════════════════════════════════════════════════════════════════════

def step_download_gguf(repo_id: str, filename: str, force: bool = False) -> Path:
    """Download GGUF from HuggingFace."""
    log_step(1, f"Downloading {filename} from {repo_id}")

    model_dir = CACHE_DIR / repo_id.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = model_dir / Path(filename).name

    if gguf_path.exists() and not force:
        log(f"  Using cached: {gguf_path}", C_DIM)
        return gguf_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log_error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(model_dir))
    log(f"  Downloaded: {gguf_path}", C_GREEN)
    return gguf_path


def step_convert_gguf(gguf_path: Path, output_dir: Path, force: bool = False) -> tuple[Path, Path, Path]:
    """Convert GGUF to bump format."""
    log_step(2, "Converting GGUF to bump format")

    weights_path = output_dir / "weights.bump"
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "weights_manifest.json"

    if weights_path.exists() and config_path.exists() and manifest_path.exists() and not force:
        log(f"  Using cached: {weights_path}", C_DIM)
        return weights_path, config_path, manifest_path

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "convert_gguf_to_bump_v4.py"),
        f"--gguf={gguf_path}",
        f"--output={weights_path}",
        f"--config-out={config_path}",
        f"--manifest-out={manifest_path}",
    ]
    run_cmd(cmd)
    log(f"  Created: {weights_path}", C_GREEN)
    return weights_path, config_path, manifest_path


def step_build_ir_v5(config_path: Path, output_dir: Path, manifest_path: Path,
                     weight_dtype: str = None, debug: bool = False,
                     parity: bool = False, force: bool = False) -> Path:
    """Build IR and generate v5 explicit C code."""
    log_step(3, "Building IR v5 (explicit codegen)")

    layout_path = output_dir / "layout_decode.json"

    if layout_path.exists() and not force:
        log(f"  Using cached: {layout_path}", C_DIM)
        return layout_path

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "build_ir_v4.py"),
        f"--config={config_path}",
        f"--prefix={output_dir}",
        f"--weights-manifest={manifest_path}",
        "--modes=prefill,decode",
        "--emit=lib",
        "--dtype=fp32",
        "--codegen=v5",      # v5 explicit codegen
        "--fusion=off",       # v5 doesn't support fusion yet
    ]

    if weight_dtype:
        cmd.append(f"--weight-dtype={weight_dtype}")

    if debug:
        cmd.append("--debug")

    if parity:
        cmd.append("--parity")

    run_cmd(cmd)
    log(f"  Created: {layout_path}", C_GREEN)
    return layout_path


def step_codegen_wrapper(output_dir: Path, force: bool = False) -> Path:
    """Generate wrapper C code using v4 codegen (compatible with v5 generated code)."""
    log_step(4, "Generating wrapper C code")

    model_c = output_dir / "model.c"
    if model_c.exists() and not force:
        log(f"  Using cached: {model_c}", C_DIM)
        return model_c

    # Import and use the v4 step_codegen
    sys.path.insert(0, str(SCRIPTS_DIR))
    import ck_run_v4
    layout_path = output_dir / "layout_decode.json"
    model_c = ck_run_v4.step_codegen(layout_path, output_dir, force=force)
    log(f"  Created: {model_c}", C_GREEN)
    return model_c


def step_compile(output_dir: Path, force: bool = False) -> Path:
    """Compile generated C to shared library."""
    log_step(5, "Compiling to shared library")

    lib_path = output_dir / "libmodel.so"
    model_c = output_dir / "model.c"

    if not model_c.exists():
        log_error("model.c not found - run step_codegen_wrapper first")
        sys.exit(1)

    if lib_path.exists() and not force:
        log(f"  Using cached: {lib_path}", C_DIM)
        return lib_path

    # Import and use the v4 step_compile
    sys.path.insert(0, str(SCRIPTS_DIR))
    import ck_run_v4
    lib_path = ck_run_v4.step_compile(model_c, output_dir, force=force)
    log(f"  Created: {lib_path}", C_GREEN)
    return lib_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    print(f"\n{C_ORANGE}C-Kernel-Engine v5{C_RESET} (Explicit Codegen)")

    input_type, info = detect_input(args.model)
    log(f"Input: {args.model} ({input_type})", C_DIM)

    # Step 1: Download
    if input_type == "hf_gguf":
        gguf_path = step_download_gguf(info["repo_id"], info["filename"], args.force_download)
        work_dir = gguf_path.parent
    elif input_type == "gguf":
        gguf_path = info["path"]
        work_dir = CACHE_DIR / gguf_path.stem
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_error(f"Unsupported input type: {input_type}")
        sys.exit(1)

    # Step 2: Convert
    weights_path, config_path, manifest_path = step_convert_gguf(
        gguf_path, work_dir, args.force_convert
    )

    # Step 3-4: Build IR + Generate v5 C code
    force_ir = args.force_compile or args.debug or args.parity
    layout_path = step_build_ir_v5(
        config_path, work_dir, manifest_path,
        weight_dtype=args.weight_dtype,
        debug=args.debug,
        parity=args.parity,
        force=force_ir,
    )

    # Step 4: Generate wrapper
    model_c = step_codegen_wrapper(work_dir, args.force_compile)

    # Step 5: Compile
    lib_path = step_compile(work_dir, args.force_compile)

    # Summary
    print(f"\n{C_GREEN}Generated (v5 Explicit):{C_RESET}")
    print(f"  Decode C:  {work_dir}/generated_*_decode.c")
    print(f"  Prefill C: {work_dir}/generated_*_prefill.c")
    print(f"  Library:   {lib_path}")

    if args.generate_only:
        print(f"\n{C_DIM}Use --generate-only was set, skipping inference.{C_RESET}")
        return

    # Step 6: Run (placeholder)
    log_step(6, "Ready for inference")
    print(f"  {C_DIM}Inference not implemented yet. Use the generated library.{C_RESET}")


def list_models():
    print(f"\n{C_ORANGE}Cached v5 Models:{C_RESET}")
    if not CACHE_DIR.exists():
        print("  (none)")
        return

    for model_dir in sorted(CACHE_DIR.iterdir()):
        if model_dir.is_dir():
            lib = model_dir / "libmodel.so"
            status = C_GREEN + "compiled" + C_RESET if lib.exists() else C_DIM + "not compiled" + C_RESET
            print(f"  {model_dir.name}: {status}")


def clean_models(model: Optional[str]):
    if model:
        model_dir = CACHE_DIR / model.replace("/", "--")
        if model_dir.exists():
            shutil.rmtree(model_dir)
            print(f"Cleaned: {model_dir}")
        else:
            print(f"Not found: {model_dir}")
    else:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"Cleaned all v5 models")


def main():
    parser = argparse.ArgumentParser(
        description="C-Kernel-Engine v5 Pipeline (Explicit Codegen)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ck_run_v5.py run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf
  python scripts/ck_run_v5.py run MODEL --weight-dtype=q4_k_m --debug
  python scripts/ck_run_v5.py run MODEL --parity --generate-only
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run v5 pipeline")
    run_parser.add_argument("model", help="Model: hf://org/repo/file.gguf or local path")
    run_parser.add_argument("--weight-dtype",
                           choices=["q4_k", "q4_k_m", "q5_0", "q6_k", "q8_0"],
                           help="Weight dtype (default: auto from manifest)")
    run_parser.add_argument("--debug", action="store_true",
                           help="Insert debug prints (buffer stats after each op)")
    run_parser.add_argument("--parity", action="store_true",
                           help="Save layer outputs for PyTorch comparison")
    run_parser.add_argument("--generate-only", action="store_true",
                           help="Generate code only, don't run inference")
    run_parser.add_argument("--force-download", action="store_true",
                           help="Re-download model")
    run_parser.add_argument("--force-convert", action="store_true",
                           help="Re-convert weights")
    run_parser.add_argument("--force-compile", action="store_true",
                           help="Re-generate and recompile")

    # List command
    subparsers.add_parser("list", help="List cached v5 models")

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean cached models")
    clean_parser.add_argument("model", nargs="?", help="Model to clean (or all)")

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(args)
    elif args.command == "list":
        list_models()
    elif args.command == "clean":
        clean_models(getattr(args, "model", None))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
