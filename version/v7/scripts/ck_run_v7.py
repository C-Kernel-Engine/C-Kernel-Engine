#!/usr/bin/env python3
"""
ck_run_v7.py - C-Kernel-Engine v7 Pipeline Runner (standalone)

Unified CLI that chains: download -> convert -> IR -> codegen -> compile -> run

v7 features:
  - Manifest-first approach (requires weights manifest)
  - Explicit unrolled codegen (per-layer, explicit kernels)
  - Mixed-quant support via per-tensor dtypes

Usage:
  python scripts/v7/ck_run_v7.py run HuggingFaceTB/SmolLM-135M
  python scripts/v7/ck_run_v7.py run ./model.gguf
  python scripts/v7/ck_run_v7.py run ./local/config.json
  python scripts/v7/ck_run_v7.py run Qwen/Qwen2-0.5B --weight-dtype=q4_k
"""

import argparse
import ctypes
import json
import math
import os
import random
import shutil
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

V7_MODE = True  # Always v7 in this standalone script

SCRIPTS_DIR = Path(__file__).parent  # version/v7/scripts/
V7_ROOT = SCRIPTS_DIR.parent        # version/v7/
PROJECT_ROOT = SCRIPTS_DIR.parents[2]  # C-Kernel-Engine/
ROOT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"  # Main scripts (for ck_chat.py etc)
BUILD_DIR = PROJECT_ROOT / "build"
HEADER_SIZE = 128
KERNEL_MAPS_DIR = V7_ROOT / "kernel_maps"
KERNEL_REGISTRY_PATH = KERNEL_MAPS_DIR / "KERNEL_REGISTRY.json"

def _get_cache_dir() -> Path:
    """Pick a writable cache dir (env override, default ~/.cache, fallback to repo)."""
    env = os.environ.get("CK_CACHE_DIR")
    if env:
        path = Path(env).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    default = Path.home() / ".cache/ck-engine-v7/models"
    try:
        default.mkdir(parents=True, exist_ok=True)
        probe = default / ".ck_write_probe"
        with open(probe, "w") as f:
            f.write("ok")
        probe.unlink()
        return default
    except Exception:
        fallback = PROJECT_ROOT / ".ck_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

CACHE_DIR = _get_cache_dir()


def _get_default_report_dir() -> Path:
    """Resolve writable v7 report directory (env override + ignored cache default)."""
    env = os.environ.get("CK_V7_REPORT_DIR")
    report_dir = Path(env).expanduser() if env else (V7_ROOT / ".cache" / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


DEFAULT_REPORT_DIR = _get_default_report_dir()

# Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_ORANGE = "\033[38;5;214m"
C_GREEN = "\033[38;5;114m"
C_BLUE = "\033[38;5;75m"
C_RED = "\033[38;5;203m"
C_CYAN = "\033[38;5;87m"


def log(msg: str, color: str = ""):
    """Print colored log message."""
    if color:
        print(f"{color}{msg}{C_RESET}")
    else:
        print(msg)


def log_step(step: int, msg: str):
    """Print pipeline step."""
    print(f"{C_ORANGE}[{step}/6]{C_RESET} {C_BOLD}{msg}{C_RESET}")


def log_error(msg: str):
    """Print error message."""
    print(f"{C_RED}Error:{C_RESET} {msg}", file=sys.stderr)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(cmd: list, cwd: Path = None, capture: bool = False) -> subprocess.CompletedProcess:
    """Run command with error handling."""
    try:
        if capture:
            return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        else:
            return subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        log_error(f"Command failed: {' '.join(cmd)}")
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)


def run_cmd_allow_fail(cmd: list, cwd: Path = None) -> subprocess.CompletedProcess:
    """Run command without exiting on non-zero status."""
    return subprocess.run(cmd, cwd=cwd)


def _sync_runtime_lib(src: Path, dst: Path, label: str) -> None:
    """Copy runtime shared library into model dir, overwriting stale copies."""
    if not src.exists():
        return
    try:
        # Always overwrite: stale sidecar libs in cache can segfault after refactors.
        shutil.copy2(src, dst)
        log(f"  Refreshed {label} at {dst}", C_DIM)
    except Exception as e:
        log(f"  Warning: failed to refresh {label}: {e}", C_ORANGE)


def _detect_default_ck_threads() -> int:
    """Best-effort physical core count with sensible fallback."""
    logical = None
    try:
        logical = len(os.sched_getaffinity(0))
    except Exception:
        logical = os.cpu_count() or 1

    physical = 0
    try:
        pairs = set()
        phys_id = None
        core_id = None
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    if phys_id is not None and core_id is not None:
                        pairs.add((phys_id, core_id))
                    phys_id = None
                    core_id = None
                    continue
                if line.startswith("physical id"):
                    phys_id = int(line.split(":", 1)[1].strip())
                elif line.startswith("core id"):
                    core_id = int(line.split(":", 1)[1].strip())
        if phys_id is not None and core_id is not None:
            pairs.add((phys_id, core_id))
        physical = len(pairs)
    except Exception:
        physical = 0

    # Match C-side behavior: if physical core detection is unreliable, use logical.
    if physical <= 1 and (logical or 1) > 1:
        return int(logical)
    if physical > 1:
        return min(int(physical), int(logical or physical))
    return int(logical or 1)


def load_manifest_non_fp_dtypes(manifest_path: Path) -> set[str]:
    """Return non-FP *weight* dtype set from a manifest.

    Notes:
    - Ignore tokenizer/metadata payload dtypes (e.g. i32/u8 vocab tables).
    - Treat only quantized/model-weight dtypes as "non-fp" for dtype override checks.
    """
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()

    float_like = {"fp32", "f32", "bf16", "bfloat16", "fp16", "f16"}
    # Common non-weight/metadata payload dtypes that should not block fp32 override.
    metadata_like = {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "int8", "int16", "int32", "int64"}

    non_fp: set[str] = set()
    for entry in data.get("entries", []):
        if not isinstance(entry, dict):
            continue
        dt = str(entry.get("dtype", "")).lower()
        if not dt:
            continue
        if dt in float_like or dt in metadata_like:
            continue
        # Quantized/model storage dtypes are represented as q* in this codebase.
        if dt.startswith("q"):
            non_fp.add(dt)
            continue
        # Unknown non-float dtype: keep it conservative so we don't silently mis-handle.
        non_fp.add(dt)
    return non_fp


def normalize_weight_dtype(weight_dtype: Optional[str], manifest_path: Optional[Path]) -> Optional[str]:
    """Normalize weight dtype and guard against mixed-quant overrides.

    Returns None when per-tensor dtypes from manifest should be used (mixed quant mode).
    Returns a dtype string when a uniform dtype should be applied.
    """
    # Load manifest dtypes upfront if available
    non_fp = set()
    if manifest_path and manifest_path.exists():
        non_fp = load_manifest_non_fp_dtypes(manifest_path)

    # No explicit dtype specified
    if not weight_dtype:
        # If manifest has mixed quant types, use per-tensor mode automatically
        if len(non_fp) > 1:
            types = ", ".join(sorted(non_fp))
            log(f"  Mixed quant detected ({types}); using per-tensor dtypes from manifest", C_DIM)
            return None
        return None

    dtype = weight_dtype.lower()
    if dtype == "float32":
        dtype = "f32"

    # Explicit q4_k_m means mixed quant mode
    if dtype == "q4_k_m":
        if not manifest_path or not manifest_path.exists():
            log_error("q4_k_m requires a weights manifest (GGUF conversion).")
            sys.exit(1)
        log("  q4_k_m is mixed quant; using per-weight dtypes from manifest", C_DIM)
        return None

    # Check manifest compatibility
    if non_fp:
        # Don't allow forcing float on quantized weights
        if dtype in {"f32", "bf16"}:
            log_error("Manifest has quantized weights; do not force float --weight-dtype.")
            sys.exit(1)

        # Mixed quant in manifest - accept and use per-tensor mode
        if len(non_fp) > 1:
            types = ", ".join(sorted(non_fp))
            log(f"  Mixed quant detected ({types}); using per-tensor dtypes from manifest", C_DIM)
            return None

        # Single quant type - verify it matches
        only = next(iter(non_fp))
        if dtype != only:
            log_error(f"Manifest dtype is {only}; --weight-dtype={dtype} is incompatible.")
            sys.exit(1)

    return dtype


# ═══════════════════════════════════════════════════════════════════════════════
# Input Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_input_type(model_input: str) -> tuple[str, dict]:
    """
    Detect input type and return (type, info).
    Types: 'hf_gguf', 'hf_id', 'hf_url', 'gguf', 'local_dir', 'local_config'
    """
    # HuggingFace single file URL: hf://org/repo/file.gguf
    # This downloads just the GGUF file, not the entire repo
    if model_input.startswith('hf://') and model_input.endswith('.gguf'):
        # Parse: hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf
        parts = model_input[5:].split('/')  # Remove 'hf://'
        if len(parts) >= 3:
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = '/'.join(parts[2:])  # Handle nested paths
            return 'hf_gguf', {'repo_id': repo_id, 'filename': filename}

    # Local GGUF file
    if model_input.endswith('.gguf') and Path(model_input).exists():
        return 'gguf', {'path': Path(model_input).resolve()}

    # Local config.json
    if model_input.endswith('.json') and Path(model_input).exists():
        return 'local_config', {'path': Path(model_input).resolve()}

    # Local directory with config.json
    local_path = Path(model_input)
    if local_path.is_dir() and (local_path / "config.json").exists():
        return 'local_dir', {'path': local_path.resolve()}

    # HuggingFace URL
    if model_input.startswith('https://huggingface.co/'):
        # Extract org/model from URL
        parts = model_input.replace('https://huggingface.co/', '').strip('/').split('/')
        if len(parts) >= 2:
            model_id = f"{parts[0]}/{parts[1]}"
            return 'hf_id', {'model_id': model_id, 'org': parts[0], 'name': parts[1]}

    # HuggingFace model ID (org/model or just model)
    if '/' in model_input:
        org, name = model_input.split('/', 1)
        return 'hf_id', {'model_id': model_input, 'org': org, 'name': name}

    # Assume single name is HF model (search common orgs)
    return 'hf_id', {'model_id': model_input, 'org': '', 'name': model_input}


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ═══════════════════════════════════════════════════════════════════════════════

def step_download(model_id: str, cache_dir: Path, force: bool = False) -> Path:
    """Download model from HuggingFace Hub."""
    log_step(1, f"Downloading {model_id}")

    model_dir = cache_dir / model_id.replace('/', '--')
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    config_path = model_dir / "config.json"
    if config_path.exists() and not force:
        log(f"  Using cached model at {model_dir}", C_DIM)
        return model_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log_error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    log(f"  Downloading to {model_dir}", C_DIM)
    snapshot_download(
        model_id,
        local_dir=str(model_dir),
        ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot"],  # Skip non-safetensors
    )

    return model_dir


def _strip_gguf_suffix(model_id: str) -> str:
    lower = model_id.lower()
    for suffix in ("-gguf", "_gguf", ".gguf"):
        if lower.endswith(suffix):
            return model_id[:-len(suffix)]
    return model_id


def ensure_tokenizer_files(model_id: str, work_dir: Path) -> None:
    """Ensure tokenizer.json exists in work_dir (fetch from base repo if needed)."""
    tokenizer_path = work_dir / "tokenizer.json"
    if tokenizer_path.exists():
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log_error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    candidates = []
    base_id = _strip_gguf_suffix(model_id)
    if base_id != model_id:
        candidates.append(base_id)
    candidates.append(model_id)

    for repo_id in candidates:
        log(f"  Fetching tokenizer.json from {repo_id}", C_DIM)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename="tokenizer.json",
                local_dir=str(work_dir),
            )
            if tokenizer_path.exists():
                return
        except Exception:
            pass

    log(f"  Warning: tokenizer.json not found for {model_id}", C_DIM)


def step_download_gguf(repo_id: str, filename: str, cache_dir: Path, force: bool = False) -> Path:
    """Download a single GGUF file from HuggingFace Hub."""
    log_step(1, f"Downloading {filename} from {repo_id}")

    # Create cache directory based on repo
    model_dir = cache_dir / repo_id.replace('/', '--')
    model_dir.mkdir(parents=True, exist_ok=True)

    gguf_path = model_dir / Path(filename).name

    # Check if already downloaded
    if gguf_path.exists() and not force:
        log(f"  Using cached GGUF at {gguf_path}", C_DIM)
        return gguf_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log_error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    log(f"  Downloading to {gguf_path}", C_DIM)
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(model_dir),
    )

    # hf_hub_download might put it in a subdirectory, move to expected location
    downloaded = Path(downloaded_path)
    if downloaded != gguf_path:
        shutil.move(str(downloaded), str(gguf_path))

    log(f"  Downloaded {gguf_path.stat().st_size / 1e6:.1f} MB", C_GREEN)
    return gguf_path


def step_convert_hf(model_dir: Path,
                    output_dir: Path,
                    weight_dtype: str = "float32",
                    force: bool = False,
                    tokenizer_json: Optional[Path] = None) -> Path:
    """Convert HF safetensors to bump format."""
    log_step(2, f"Converting weights to bump format ({weight_dtype})")

    weights_path = output_dir / "weights.bump"
    manifest_path = output_dir / "weights_manifest.json"

    if weights_path.exists() and manifest_path.exists() and not force:
        log(f"  Using cached weights at {weights_path}", C_DIM)
        return weights_path

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "convert_hf_to_bump_v7.py"),
        f"--checkpoint={model_dir}",
        f"--output={weights_path}",
        f"--dtype={weight_dtype}",
        f"--manifest-out={manifest_path}",
    ]
    if tokenizer_json and tokenizer_json.exists():
        cmd.append(f"--tokenizer-json={tokenizer_json}")

    run_cmd(cmd)
    log(f"  Created {weights_path}", C_GREEN)
    return weights_path


def validate_gguf_kernel_coverage(gguf_path: Path) -> bool:
    """Validate that all GGUF tensor types have kernel support. Returns True if all supported."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "convert_gguf_to_bump_v7.py"),
        f"--gguf={gguf_path}",
        "--inspect",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check for missing kernel warnings
    if "❌ MISSING" in result.stdout:
        log(f"{C_RED}[validate]{C_RESET} Kernel coverage check failed:")
        for line in result.stdout.split('\n'):
            if "kernel coverage:" in line or "MISSING" in line or "→" in line:
                print(f"  {line.strip()}")
        return False
    elif "✓" in result.stdout:
        log(f"  Kernel coverage: {C_GREEN}OK{C_RESET}", C_DIM)
    return True


def step_convert_gguf(gguf_path: Path,
                      output_dir: Path,
                      force: bool = False,
                      validate: bool = True) -> tuple[Path, Path]:
    """Convert GGUF to bump format.

    Note: GGUF contains complete vocab including special tokens.
    We no longer pass tokenizer.json to the converter - that was causing
    vocab corruption where gaps in tokenizer.json created <|ck_missing_N|>
    placeholders that overwrote proper GGUF vocab entries like <|im_end|>.
    """
    log_step(2, f"Converting GGUF to bump format")

    # Validate kernel coverage first
    if validate:
        if not validate_gguf_kernel_coverage(gguf_path):
            log_error("GGUF contains unsupported tensor types. See above for details.")
            log(f"  {C_DIM}Tip: Check src/kernels/ for available quant kernels{C_RESET}")
            sys.exit(1)

    weights_path = output_dir / "weights.bump"
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "weights_manifest.json"

    if weights_path.exists() and config_path.exists() and manifest_path.exists() and not force:
        log(f"  Using cached weights at {weights_path}", C_DIM)
        return weights_path, config_path
    if weights_path.exists() and config_path.exists() and not manifest_path.exists() and not force:
        log(f"  Missing manifest; re-running conversion", C_DIM)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use GGUF vocab directly - it has complete vocab including special tokens
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "convert_gguf_to_bump_v7.py"),
        f"--gguf={gguf_path}",
        f"--output={weights_path}",
        f"--config-out={config_path}",
        f"--manifest-out={manifest_path}",
    ]

    run_cmd(cmd)
    log(f"  Created {weights_path}", C_GREEN)
    return weights_path, config_path


def step_regenerate_kernel_registry(force: bool = False) -> Path:
    """Regenerate KERNEL_REGISTRY.json from kernel maps if needed."""
    if not KERNEL_MAPS_DIR.exists():
        log(f"  Warning: kernel_maps directory not found at {KERNEL_MAPS_DIR}", C_DIM)
        return KERNEL_REGISTRY_PATH

    # Check if registry needs regeneration
    registry_mtime = 0
    if KERNEL_REGISTRY_PATH.exists() and not force:
        registry_mtime = KERNEL_REGISTRY_PATH.stat().st_mtime

    # Check if any kernel map file is newer than registry
    needs_regen = not KERNEL_REGISTRY_PATH.exists() or force
    if not needs_regen:
        for map_file in KERNEL_MAPS_DIR.glob("*.json"):
            # Skip registry files themselves
            if map_file.name.upper().startswith("KERNEL_"):
                continue
            if map_file.stat().st_mtime > registry_mtime:
                needs_regen = True
                log(f"  {map_file.name} is newer than registry", C_DIM)
                break

    if not needs_regen:
        return KERNEL_REGISTRY_PATH

    log(f"  Regenerating kernel registry from kernel maps", C_DIM)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "gen_kernel_registry_from_maps.py"),
        f"--dir={KERNEL_MAPS_DIR}",
        f"--output={KERNEL_REGISTRY_PATH}",
    ]
    run_cmd(cmd)
    log(f"  Updated {KERNEL_REGISTRY_PATH.name}", C_GREEN)
    return KERNEL_REGISTRY_PATH


def step_inspect_weights_v7(input_type: str, model_dir: Optional[Path], gguf_path: Optional[Path],
                            output_dir: Path, force: bool = False) -> Path:
    """Emit a lightweight weights manifest for v7 (no conversion)."""
    manifest_path = output_dir / "weights_manifest_input.json"
    if manifest_path.exists() and not force:
        log(f"  Using cached manifest at {manifest_path}", C_DIM)
        return manifest_path

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "inspect_weights_v7.py"),
        f"--manifest-out={manifest_path}",
    ]

    if gguf_path is not None:
        cmd.append(f"--gguf={gguf_path}")
        cmd.append(f"--config-out={output_dir / 'config.json'}")
    elif model_dir is not None:
        cmd.append(f"--checkpoint={model_dir}")
    else:
        log_error("inspect requires gguf or checkpoint directory")
        sys.exit(1)

    run_cmd(cmd)
    log(f"  Created {manifest_path}", C_GREEN)
    return manifest_path


def _prefill_codegen_is_stub(output_dir: Path) -> bool:
    prefill_path = output_dir / "ck-kernel-prefill.c"
    if not prefill_path.exists():
        return True
    try:
        text = prefill_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return True
    if "prefill not yet implemented" in text:
        return True
    return "EXPLICIT PER-LAYER PREFILL FUNCTIONS" not in text


def step_build_ir(config_path: Path, output_dir: Path, manifest_path: Path = None,
                  bump_path: Path = None,
                  weight_dtype: str = None, modes: list = None, force: bool = False,
                  debug: bool = False, parity: bool = False,
                  codegen_version: str = "v7",
                  int8_activations: bool = False,
                  context_len: int = None,
                  logits_layout: str = None,
                  no_fusion: bool = False,
                  layout_mode: str = "region",
                  layer_limit: int = None,
                  profile: bool = False,
                  parallel_decode: bool = False) -> Path:
    """Build IR1: Direct template + quant → kernel IDs (v7 new pipeline).

    Args:
        manifest_path: Path to weights_manifest.json (required for v7).
        modes: Execution modes (generates IR1 for all requested modes).
        force: If True, regenerate even if cached IR1 exists.

    Returns:
        Path to primary IR1 file (decode mode).

    Note:
        v7 pipeline stages (only IR1 is implemented):
        - IR1: Template + Quant → Kernel IDs (current)
        - IR2: Add tensor metadata (shapes, memory layout) - TODO
        - Memory Planning: Allocate buffers, plan reuse - TODO
        - Code Generation: Generate C code that calls kernels - TODO
    """
    log_step(3, "Building IR1 (Template + Quant → Kernel IDs)")

    # Validate that manifest exists (required for v7)
    if not manifest_path or not manifest_path.exists():
        log_error("Manifest path required for v7 pipeline")
        sys.exit(1)

    # Determine which modes to generate (default: both prefill and decode)
    target_modes = modes if modes else ["prefill", "decode"]
    # Use a shared layout across modes to avoid offset mismatches
    shared_layout_mode = None
    if len(target_modes) > 1:
        shared_layout_mode = "prefill" if "prefill" in target_modes else target_modes[0]
        # Ensure the layout-producing mode runs first
        if target_modes[0] != shared_layout_mode:
            target_modes = [shared_layout_mode] + [m for m in target_modes if m != shared_layout_mode]
    log(f"  Generating IR1 for modes: {', '.join(target_modes)}", C_DIM)

    output_dir.mkdir(parents=True, exist_ok=True)

    ir1_paths = {}

    # Generate IR1 for each mode
    shared_layout_path = output_dir / f"layout_{shared_layout_mode}.json" if shared_layout_mode else None
    for mode in target_modes:
        ir1_path = output_dir / f"ir1_{mode}.json"
        layout_path = output_dir / f"layout_{mode}.json"
        lowered_path = output_dir / f"lowered_{mode}.json"
        lowered_call_path = output_dir / f"lowered_{mode}_call.json"
        manifest_map_path = output_dir / "weights_manifest.map"
        init_path = output_dir / "init.json"  # Init ops (rope_init, etc.) - shared across modes
        layout_input = None
        if shared_layout_mode and mode != shared_layout_mode:
            layout_input = shared_layout_path

        # Check if we can reuse existing IR outputs
        if not force:
            outputs_exist = all([
                ir1_path.exists(),
                layout_path.exists(),
                lowered_path.exists(),
                lowered_call_path.exists(),
                manifest_map_path.exists(),
            ])
            if layout_input and not layout_input.exists():
                outputs_exist = False
            manifest_newer = False
            context_len_mismatch = False
            if outputs_exist:
                manifest_mtime = manifest_path.stat().st_mtime
                layout_input_mtime = 0
                if layout_input and layout_input.exists():
                    layout_input_mtime = layout_input.stat().st_mtime
                manifest_newer = (
                    manifest_mtime > ir1_path.stat().st_mtime or
                    manifest_mtime > layout_path.stat().st_mtime or
                    manifest_mtime > lowered_path.stat().st_mtime or
                    manifest_mtime > lowered_call_path.stat().st_mtime or
                    manifest_mtime > manifest_map_path.stat().st_mtime or
                    layout_input_mtime > ir1_path.stat().st_mtime or
                    layout_input_mtime > layout_path.stat().st_mtime or
                    layout_input_mtime > lowered_path.stat().st_mtime or
                    layout_input_mtime > lowered_call_path.stat().st_mtime
                )
                # Check if context_len or layout_mode/logits_layout matches cached layout
                if layout_path.exists():
                    try:
                        import json
                        with open(layout_path, 'r') as f:
                            cached_layout = json.load(f)
                        # Check context_len
                        if context_len is not None:
                            cached_ctx = cached_layout.get("config", {}).get("context_length")
                            if cached_ctx is not None and cached_ctx != context_len:
                                context_len_mismatch = True
                                log(f"  Context length changed: cached={cached_ctx}, requested={context_len}", C_YELLOW)
                        # Check layout_mode
                        cached_mode = cached_layout.get("memory", {}).get("arena", {}).get("mode")
                        if cached_mode and cached_mode != layout_mode:
                            context_len_mismatch = True  # Reuse flag for any config change
                            log(f"  Layout mode changed: cached={cached_mode}, requested={layout_mode}", C_YELLOW)
                        if logits_layout and logits_layout != "auto":
                            cached_logits_layout = cached_layout.get("config", {}).get("logits_layout")
                            if cached_logits_layout and cached_logits_layout != logits_layout:
                                context_len_mismatch = True
                                log(f"  Logits layout changed: cached={cached_logits_layout}, requested={logits_layout}", C_YELLOW)
                            elif cached_logits_layout is None:
                                context_len_mismatch = True
                                log(f"  Logits layout set to {logits_layout}; cached layout has no setting", C_YELLOW)
                    except Exception:
                        pass

            if outputs_exist and not manifest_newer and not context_len_mismatch:
                log(f"  Using cached IR outputs for {mode} at {ir1_path}", C_DIM)
                ir1_paths[mode] = ir1_path
                continue
            if manifest_newer:
                log(f"  Manifest updated, rebuilding IR outputs for {mode}", C_DIM)
            if context_len_mismatch:
                log(f"  Context length changed, rebuilding IR outputs for {mode}", C_DIM)

        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "build_ir_v7.py"),
            f"--manifest={manifest_path}",
            f"--mode={mode}",
            f"--output={ir1_path}",
            f"--layout-output={layout_path}",
            f"--lowered-output={lowered_path}",
            f"--call-output={lowered_call_path}",
            f"--manifest-map-output={manifest_map_path}",
            f"--layout-mode={layout_mode}",
        ]
        # Generate init.json only once (first mode) - it's shared across modes
        if not init_path.exists() or force:
            cmd.append(f"--init-output={init_path}")
        if layout_input:
            cmd.append(f"--layout-input={layout_input}")
        if layer_limit is not None:
            cmd.append(f"--layer-limit={layer_limit}")
        if context_len is not None:
            cmd.append(f"--context-len={context_len}")
        if logits_layout:
            cmd.append(f"--logits-layout={logits_layout}")
        if no_fusion:
            cmd.append("--no-fusion")

        if profile:
            cmd.append("--profile")

        # ADR: OpenMP parallel pass is SUPERSEDED by persistent pthread thread pool.
        #
        # Previously, --parallel passed OpenMP pragma annotations into the IR via
        # parallel_pass.py. However, codegen_v7.py never consumed these annotations.
        # Actual decode parallelization uses ck_parallel_decode.h which macro-redirects
        # serial GEMV calls to thread pool dispatch wrappers — always enabled, no flag needed.
        #
        # Why thread pool over OpenMP:
        #   - OpenMP fork/join creates N threads per #pragma region (~15-50us overhead each)
        #   - Thread pool keeps N-1 pthreads alive, spin-waiting on atomics (~0.1us wake)
        #   - OpenMP + thread pool together causes 2N threads competing for N cores
        #   - Thread pool gives deterministic (ith, nth) work splitting to kernels
        #
        # The --parallel / --parallel-decode flags fed the OpenMP pass which is now
        # commented out in build_ir_v7.py. Flags accepted but no longer have effect.
        if parallel_decode:
            log("  [WARNING] --parallel-decode is deprecated. Thread pool dispatch "
                "(ck_parallel_decode.h) is always enabled. OpenMP parallel pass removed "
                "due to fork/join overhead and core oversubscription. Flag ignored.", C_YELLOW)

        log(f"  Generating IR1 + {'NO fusion' if no_fusion else 'fusion'} + layout + lowered + call IR for mode: {mode}", C_DIM)
        run_cmd(cmd)
        log(f"  Created IR1 for {mode} at {ir1_path}", C_GREEN)
        log(f"  Created layout for {mode} at {layout_path}", C_GREEN)
        log(f"  Created lowered IR for {mode} at {lowered_path}", C_GREEN)
        log(f"  Created call-ready IR for {mode} at {lowered_call_path}", C_GREEN)
        if init_path.exists():
            log(f"  Created init IR at {init_path}", C_GREEN)

        # Generate human-readable .map file
        map_path = output_dir / f"layout_{mode}.map"
        map_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "generate_memory_map_v7.py"),
            str(layout_path),
            "-o", str(map_path)
        ]
        run_cmd(map_cmd)
        log(f"  Created memory map at {map_path}", C_GREEN)

        ir1_paths[mode] = ir1_path

    # Return decode IR1 as primary (for compatibility)
    return ir1_paths.get("decode", ir1_paths[target_modes[0]])


def step_codegen(
    ir1_path: Path,
    output_dir: Path,
    force: bool = False,
    profile: bool = False,
    dump: bool = False,
) -> Path:
    """Generate v7 C code from lowered IR.

    The lowered IR contains everything needed for codegen:
    - Explicit pointer expressions for weights and activations
    - Function names for each kernel
    - Model config parameters
    """
    log_step(4, "Generating C code")

    # Check for call-ready lowered IR files
    lowered_decode = output_dir / "lowered_decode_call.json"
    lowered_prefill = output_dir / "lowered_prefill_call.json"
    model_c_path = output_dir / "model_v7.c"

    if not lowered_decode.exists():
        log_error(f"Lowered IR not found: {lowered_decode}")
        log_error("Run step_build_ir first to generate call-ready lowered IR")
        sys.exit(1)

    # Skip codegen if model_v7.c already exists and is newer than lowered IR
    if not force and model_c_path.exists():
        src_mtime = lowered_decode.stat().st_mtime
        c_mtime = model_c_path.stat().st_mtime
        if c_mtime > src_mtime:
            log(f"  Using cached: {model_c_path}", C_GREEN)
            return model_c_path

    # Show stats
    import json
    with open(lowered_decode, 'r') as f:
        decode_ir = json.load(f)
    decode_ops = len(decode_ir.get('operations', []))

    prefill_ops = 0
    if lowered_prefill.exists():
        with open(lowered_prefill, 'r') as f:
            prefill_ir = json.load(f)
        prefill_ops = len(prefill_ir.get('operations', []))

    log(f"  Decode: {decode_ops} ops", C_DIM)
    log(f"  Prefill: {prefill_ops} ops", C_DIM)

    # Call codegen
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "codegen_v7.py"),
        f"--decode={lowered_decode}",
        f"--prefill={lowered_prefill}" if lowered_prefill.exists() else "",
        f"--output={model_c_path}"
    ]
    if profile:
        cmd.append("--profile")
    if dump:
        cmd.append("--parity-dump")
    # Filter empty args
    cmd = [c for c in cmd if c]

    run_cmd(cmd)
    log(f"  Created C code at {model_c_path}", C_GREEN)

    return model_c_path


def step_compile(model_c_path: Path, output_dir: Path, force: bool = False) -> Path:
    """Compile C code to shared library linked against libckernel_engine.so."""
    log_step(5, "Compiling to shared library")

    # Output library name (ck_chat.py expects libmodel.so or ck-kernel-inference.so)
    lib_path = output_dir / "libmodel.so"
    kernel_lib = BUILD_DIR / "libckernel_engine.so"
    tokenizer_lib = BUILD_DIR / "libckernel_tokenizer.so"

    log(f"  Source: {model_c_path}", C_DIM)
    log(f"  Lines: {sum(1 for _ in open(model_c_path))}", C_DIM)

    # Ensure runtime libs exist before compile.
    missing_targets = []
    if not kernel_lib.exists():
        missing_targets.append(str(kernel_lib))
    if not tokenizer_lib.exists():
        missing_targets.append(str(tokenizer_lib))
    if missing_targets:
        log(f"  Building missing runtime libs: {', '.join(Path(t).name for t in missing_targets)}", C_DIM)
        run_cmd(["make"] + missing_targets, cwd=PROJECT_ROOT)
        still_missing = [t for t in missing_targets if not Path(t).exists()]
        if still_missing:
            log(f"  Missing required runtime libs after build: {', '.join(Path(t).name for t in still_missing)}", C_RED)
            return model_c_path

    # Skip if already compiled and not forcing
    if lib_path.exists() and not force:
        src_mtime = model_c_path.stat().st_mtime
        lib_mtime = lib_path.stat().st_mtime
        if lib_mtime > src_mtime:
            log(f"  Using cached: {lib_path}", C_GREEN)
            return lib_path

    # Compile to shared library
    # -fvisibility=default ensures CK_EXPORT symbols are exported
    include_dir = PROJECT_ROOT / "include"
    v7_include = V7_ROOT / "include"
    v7_src = V7_ROOT / "src"
    loader_src = V7_ROOT / "src" / "ckernel_model_load_v7.c"

    # Detect compiler for OpenMP flag
    # Override with CK_V7_COMPILER=gcc|icx|clang when needed (e.g., profiling portability).
    import shutil
    compiler = "gcc"
    requested_compiler = os.environ.get("CK_V7_COMPILER", "").strip()
    if requested_compiler:
        if not shutil.which(requested_compiler):
            log_error(f"Requested CK_V7_COMPILER not found in PATH: {requested_compiler}")
            sys.exit(1)
        compiler = requested_compiler
    elif shutil.which("icx"):
        compiler = "icx"

    omp_flag = "-qopenmp" if compiler == "icx" else "-fopenmp"

    cmd = [
        compiler,
        "-shared", "-fPIC",
        "-mcmodel=large",  # Handle large static data in v7 models
        "-O3", "-march=native",
        "-std=c11",
        "-fvisibility=default",  # Export CK_EXPORT symbols
        omp_flag,  # OpenMP for parallelization
        f"-I{include_dir}",
        f"-I{v7_include}",
        f"-I{v7_src}",
        "-o", str(lib_path),
        str(model_c_path),
        str(loader_src),
        str(v7_src / "ck_parallel_decode.c"),  # Thread-pool parallel GEMV dispatch
        str(v7_src / "ck_parallel_prefill.c"),  # Thread-pool parallel GEMM dispatch (prefill)
        f"-L{BUILD_DIR}",
        f"-L{output_dir}",  # Also look in output_dir for libckernel_engine.so
        "-lckernel_tokenizer",  # BPE tokenizer library
        # Keep tokenizer before engine: both export legacy ck_tokenizer_* symbols,
        # and Gemma's generated code must bind to the tokenizer ABI.
        "-lckernel_engine",
        "-lm",
        f"-Wl,-rpath,$ORIGIN",  # Use $ORIGIN so library can find deps in same dir
        f"-Wl,-rpath,{BUILD_DIR}",
    ]

    # Allow caller to inject extra compile/link flags for perf tooling.
    # Example: CK_V7_EXTRA_CFLAGS="-fno-omit-frame-pointer -g"
    extra_cflags = os.environ.get("CK_V7_EXTRA_CFLAGS", "").strip()
    extra_ldflags = os.environ.get("CK_V7_EXTRA_LDFLAGS", "").strip()
    if extra_cflags:
        try:
            cmd.extend(shlex.split(extra_cflags))
        except ValueError:
            log_error(f"Invalid CK_V7_EXTRA_CFLAGS: {extra_cflags}")
            sys.exit(1)
    if extra_ldflags:
        try:
            cmd.extend(shlex.split(extra_ldflags))
        except ValueError:
            log_error(f"Invalid CK_V7_EXTRA_LDFLAGS: {extra_ldflags}")
            sys.exit(1)

    # Add profiling define if requested
    if os.environ.get("CK_PROFILE") == "1":
        cmd.append("-DCK_PROFILE")
    # Enable ck_dump_tensor instrumentation in generated C when requested.
    if os.environ.get("CK_PARITY_DUMP") == "1":
        cmd.append("-DCK_PARITY_DUMP")

    log(f"  Compiling...", C_DIM)
    try:
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            log(f"  Compiled: {lib_path}", C_GREEN)
            _sync_runtime_lib(kernel_lib, output_dir / "libckernel_engine.so", "libckernel_engine.so")
            _sync_runtime_lib(tokenizer_lib, output_dir / "libckernel_tokenizer.so", "libckernel_tokenizer.so")
            # Create symlink for ck-kernel-inference.so (legacy name)
            symlink_path = output_dir / "ck-kernel-inference.so"
            try:
                if symlink_path.exists() or symlink_path.is_symlink():
                    os.unlink(symlink_path)
                os.symlink("libmodel.so", symlink_path)
                log(f"  Created symlink: ck-kernel-inference.so -> libmodel.so", C_DIM)
            except Exception:
                pass  # Symlink creation is optional
            return lib_path
        else:
            log(f"  Compilation failed:", C_RED)
            # Show first few errors
            for line in result.stderr.split('\n')[:10]:
                if line.strip():
                    log(f"    {line}", C_DIM)
            log(f"  Falling back to syntax check...", C_ORANGE)
    except Exception as e:
        log(f"  Compilation error: {e}", C_RED)

    # Fallback: syntax check only (include all paths)
    cmd = ["gcc", "-fsyntax-only", "-std=c11",
           f"-I{include_dir}", f"-I{v7_include}", f"-I{v7_src}",
           str(model_c_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            log(f"  Syntax check: PASSED", C_GREEN)
        else:
            log(f"  Syntax check failed:", C_ORANGE)
            for line in result.stderr.split('\n')[:5]:
                if line.strip():
                    log(f"    {line}", C_DIM)
    except Exception:
        pass

    return model_c_path


def _layout_weight_buffers(layout: dict) -> list[dict]:
    # New v7 layout format (flat memory spec).
    mem = layout.get("memory")
    if isinstance(mem, dict):
        weights = mem.get("weights", {})
        entries = weights.get("entries", [])
        if isinstance(entries, list) and entries:
            out = []
            for ent in entries:
                if not isinstance(ent, dict):
                    continue
                size = int(ent.get("size", 0))
                if size <= 0:
                    continue
                out.append(
                    {
                        "name": ent.get("name", ""),
                        "dtype": ent.get("dtype", "fp32"),
                        # Offset within weight arena for runtime debug mapping.
                        "offset": ent.get("offset", 0),
                        "size": size,
                    }
                )
            if out:
                return out

    # Legacy layout format (section/layer buffers).
    section = layout["sections"][0]
    buffers = []
    buffers.extend(section["header"]["buffers"])
    for layer in section["layers"]:
        buffers.extend(layer["buffers"])
    buffers.extend(section["footer"]["buffers"])
    weights = []
    for buf in buffers:
        if buf.get("role") != "weight":
            continue
        if buf.get("tied_to"):
            continue
        if int(buf.get("size", 0)) <= 0:
            continue
        weights.append(buf)
    return weights


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def build_dummy_weights(layout_path: Path, output_dir: Path) -> Path:
    with layout_path.open("r") as f:
        layout = json.load(f)
    weights = _layout_weight_buffers(layout)
    entries = []
    file_off = HEADER_SIZE
    for buf in weights:
        file_off = _align_up(file_off, 64)
        entries.append({
            "name": buf["name"],
            "dtype": buf.get("dtype", "fp32"),
            "file_offset": file_off,
            "size": int(buf["size"]),
            "runtime_offset": int(buf["offset"], 16) if isinstance(buf["offset"], str) else int(buf["offset"]),
        })
        file_off += int(buf["size"])

    weights_path = output_dir / "weights_dummy.bump"
    with open(weights_path, "wb") as f:
        f.write(b"BUMPWGT4")
        if HEADER_SIZE > 8:
            f.write(b"\x00" * (HEADER_SIZE - 8))
        f.truncate(file_off)

    model_meta = layout.get("model", {})
    if isinstance(model_meta, str):
        model_name = model_meta
    else:
        model_name = model_meta.get("name", "dummy")
    manifest = {
        "format": "ck-bumpwgt4-dummy-v1",
        "generated": "dummy",
        "model": model_name,
        "missing": [],
        "entries": entries,
    }
    json_path = output_dir / "weights_manifest.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    map_path = output_dir / "weights_manifest.map"
    with open(map_path, "w") as f:
        f.write("# ck-bumpwgt4-manifest-map v1\n")
        f.write("# name|dtype|file_offset|size|runtime_offset\n")
        for e in entries:
            f.write(
                f"{e['name']}|{e['dtype']}|0x{e['file_offset']:016X}|0x{e['size']:016X}|0x{e['runtime_offset']:016X}\n"
            )

    return weights_path


def _backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    backup = path.with_name(path.name + ".real")
    if backup.exists():
        backup = path.with_name(path.name + ".real.bak")
    path.replace(backup)
    return backup


def _restore_file(path: Path, backup: Optional[Path]) -> None:
    if backup and backup.exists():
        if path.exists():
            path.unlink()
        backup.replace(path)


def run_smoke_test(model_dir: Path, weights_path: Path, use_valgrind: bool) -> None:
    script = SCRIPTS_DIR / "ck_model_smoke_v7.py"
    cmd = [
        sys.executable,
        str(script),
        "--model-dir",
        str(model_dir),
        "--weights",
        str(weights_path),
        "--prompt-len",
        "4",
        "--decode-steps",
        "2",
    ]
    if use_valgrind:
        suppression = PROJECT_ROOT / "valgrind.supp"
        cmd = [
            "valgrind",
            "--tool=memcheck",
            "--leak-check=full",
        ] + cmd
        if suppression.exists():
            cmd.insert(3, f"--suppressions={suppression}")
        result = run_cmd_allow_fail(cmd)
        if result.returncode != 0:
            log(f"  Warning: valgrind returned {result.returncode}", C_DIM)
        return
    run_cmd(cmd)


def run_parity_tests() -> None:
    """Run parity tests with auto-escalation to DEBUG mode on failure.

    Test registry - each test covers specific regression cases:
    - test_kv_cache_layer_decode.py:
        - Non-contiguous cache stride bug (numpy view with guard region)
        - KV cache write offset calculation
        - Prefill → decode KV handoff
    - test_fused_attention_decode.py:
        - Fused attention kernel correctness
    - test_multi_layer_parity.py:
        - Progressive layer test (1 → 2 → 4 layers)
        - Inter-layer data handoff
        - Residual connection across layers
    """
    tests = [
        PROJECT_ROOT / "unittest" / "test_kv_cache_layer_decode.py",
        PROJECT_ROOT / "unittest" / "test_fused_attention_decode.py",
        PROJECT_ROOT / "unittest" / "test_multi_layer_parity.py",
    ]
    failed_tests = []

    for test in tests:
        if not test.exists():
            log(f"  Skipping missing test: {test}", C_DIM)
            continue

        cmd = [sys.executable, str(test)]
        env = os.environ.copy()
        ld_path = str(PROJECT_ROOT / "build")
        env["LD_LIBRARY_PATH"] = f"{ld_path}:{env.get('LD_LIBRARY_PATH', '')}"

        # First run: normal mode
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                               capture_output=True, text=True)

        if result.returncode != 0:
            log_error(f"Test failed: {test.name}")
            log(f"  {C_ORANGE}Auto-escalating to DEBUG mode...{C_RESET}")

            # Second run: DEBUG mode to show diagnostics
            env_debug = env.copy()
            env_debug["DEBUG"] = "1"
            subprocess.run(cmd, cwd=PROJECT_ROOT, env=env_debug)

            failed_tests.append(test.name)
        else:
            # Show success from captured output
            for line in result.stdout.splitlines():
                if "PASS" in line or "TEST:" in line:
                    print(line)

    if failed_tests:
        log_error(f"Failed tests: {', '.join(failed_tests)}")
        log(f"  {C_DIM}Tip: Run with DEBUG=1 to see detailed diagnostics{C_RESET}")
        sys.exit(1)


def _materialize_train_telemetry(summary_json: Path, profile_meta: Optional[dict] = None) -> None:
    if not summary_json.exists():
        return
    with summary_json.open("r", encoding="utf-8") as f:
        s = json.load(f)

    report_dir = DEFAULT_REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    step = int(s.get("steps", 0) or 0)
    loss_ck = float(s.get("final_ck_loss", 0.0) or 0.0)
    loss_pt = float(s.get("final_torch_loss", 0.0) or 0.0)
    lr = float(s.get("lr", 0.0) or 0.0)
    max_loss = float(s.get("max_loss_abs_diff", 0.0) or 0.0)
    max_param = float(s.get("final_param_max_abs_diff", 0.0) or 0.0)

    raw_curve = s.get("loss_curve") if isinstance(s, dict) else None
    if isinstance(raw_curve, list) and raw_curve:
        training_loss_curve = {
            "steps": raw_curve,
            "source": "train_e2e_detailed",
        }
    else:
        training_loss_curve = {
            "steps": [
                {"step": step, "loss_ck": loss_ck, "loss_pt": loss_pt, "lr": lr, "grad_norm": 0.0}
            ],
            "source": "train_e2e_summary",
        }

    raw_parity = s.get("parity_steps") if isinstance(s, dict) else None
    if isinstance(raw_parity, list) and raw_parity:
        training_parity = {
            "steps": raw_parity,
            "source": "train_e2e_detailed",
        }
    else:
        training_parity = {
            "steps": [
                {"step": step, "loss_diff": max_loss, "max_param_diff": max_param, "worst_param": "aggregate"}
            ],
            "source": "train_e2e_summary",
        }

    grad_series = s.get("grad_norm_series") if isinstance(s.get("grad_norm_series"), dict) else {}
    training_grad_norms = {
        "steps": grad_series.get("steps", [row.get("step", step) for row in training_loss_curve.get("steps", [])]),
        "global": grad_series.get("global", [row.get("grad_norm", 0.0) for row in training_loss_curve.get("steps", [])]),
        "params": grad_series.get("params", {}),
        "source": "train_e2e_detailed" if grad_series else "train_e2e_summary",
    }

    step_profile = s.get("step_profile") if isinstance(s.get("step_profile"), dict) else {}
    train_tok_s = step_profile.get("train_tok_s")
    decode_tok_s = step_profile.get("decode_tok_s", train_tok_s)
    training_step_profile = {
        "steps": int(step_profile.get("steps", step) or step),
        "micro_steps": int(step_profile.get("micro_steps", s.get("micro_steps", 0)) or 0),
        "tokens_per_update": int(step_profile.get("tokens_per_update", s.get("tokens_per_update", 0)) or 0),
        "processed_tokens": int(step_profile.get("processed_tokens", 0) or 0),
        "ck_total_ms": float(step_profile.get("ck_total_ms", 0.0) or 0.0),
        "torch_total_ms": float(step_profile.get("torch_total_ms", 0.0) or 0.0),
        "ck_avg_step_ms": float(step_profile.get("ck_avg_step_ms", 0.0) or 0.0),
        "torch_avg_step_ms": float(step_profile.get("torch_avg_step_ms", 0.0) or 0.0),
        "train_tok_s": float(train_tok_s or 0.0) if train_tok_s is not None else None,
        "decode_tok_s": float(decode_tok_s or 0.0) if decode_tok_s is not None else None,
        "external_profiles": profile_meta or {},
    }

    ckpt_info = s.get("checkpoints") if isinstance(s.get("checkpoints"), dict) else {}
    training_checkpoint_policy = {
        "policy": "step_interval" if bool(ckpt_info.get("enabled")) else "none",
        "source": "train_e2e",
        "checkpointing": bool(ckpt_info.get("enabled", False)),
        "save_every": int(ckpt_info.get("save_every", 0) or 0),
        "save_final": bool(ckpt_info.get("save_final", False)),
        "count": int(ckpt_info.get("count", 0) or 0),
        "latest_step": int(ckpt_info.get("latest_step", 0) or 0),
        "files": ckpt_info.get("files", []),
    }

    payloads = {
        "training_loss_curve_latest.json": training_loss_curve,
        "training_parity_latest.json": training_parity,
        "training_grad_norms_latest.json": training_grad_norms,
        "training_step_profile_latest.json": training_step_profile,
        "training_checkpoint_policy_latest.json": training_checkpoint_policy,
    }
    for name, payload in payloads.items():
        with (report_dir / name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def _resolve_train_text(args: argparse.Namespace) -> Optional[str]:
    """Resolve training text from data file, explicit text, or prompt."""
    train_data = getattr(args, "train_data", None)
    if train_data:
        data_path = Path(train_data)
        if not data_path.exists():
            log_error(f"Training data file not found: {data_path}")
            sys.exit(2)
        try:
            data = data_path.read_text(encoding="utf-8")
        except Exception as e:
            log_error(f"Failed to read training data from {data_path}: {e}")
            sys.exit(2)
        if not data.strip():
            log_error(f"Training data file is empty: {data_path}")
            sys.exit(2)
        return data

    train_text = getattr(args, "train_text", None)
    if train_text:
        return str(train_text)

    prompt = getattr(args, "prompt", None)
    return str(prompt) if prompt else None


def _resolve_train_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "train_mode", "pretrain") or "pretrain").lower()
    if getattr(args, "pretraining", False):
        mode = "pretrain"
    if mode not in ("pretrain", "sft"):
        log_error(f"Unsupported train mode: {mode}")
        sys.exit(2)
    return mode


def _resolve_train_backend(args: argparse.Namespace) -> str:
    backend = getattr(args, "backend", None)
    if backend is None:
        backend = getattr(args, "train_backend", "both")
    backend = str(backend or "both").lower()
    if backend == "torch":
        backend = "pytorch"
    if backend not in ("ck", "pytorch", "both"):
        log_error(f"Unsupported train backend: {backend}")
        sys.exit(2)
    if backend == "ck":
        log("  backend=ck: running generated v7 C training runtime", C_DIM)
    elif backend == "pytorch":
        log("  backend=pytorch: running PyTorch parity harness only", C_DIM)
    return backend



def _build_train_token_batches(train_text: Optional[str], total_tokens: int, seq_len: int, vocab: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Build deterministic token/target batches for CK runtime train stepping."""
    if seq_len < 1:
        return []
    needed = max(int(total_tokens) + 1, int(seq_len) + 1)
    if train_text:
        raw = train_text.encode("utf-8", errors="ignore")
        if len(raw) < 2:
            raw = b"hello"
        ids = [int(b) % int(vocab) for b in raw]
        repeats = (needed + len(ids) - 1) // len(ids)
        stream = (ids * repeats)[:needed]
    else:
        rng = random.Random(int(seed))
        stream = [rng.randrange(int(vocab)) for _ in range(needed)]

    batches: list[tuple[list[int], list[int]]] = []
    for i in range(0, max(1, int(total_tokens) - int(seq_len) + 1), int(seq_len)):
        x = stream[i:i + int(seq_len)]
        y = stream[i + 1:i + int(seq_len) + 1]
        if len(x) == int(seq_len) and len(y) == int(seq_len):
            batches.append((x, y))
    if not batches:
        batches.append((stream[:int(seq_len)], stream[1:int(seq_len) + 1]))
    return batches



def _manifest_entries_map(manifest: dict) -> dict[str, dict]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Invalid manifest: missing entries[]")
    out: dict[str, dict] = {}
    for e in entries:
        if isinstance(e, dict) and e.get("name"):
            out[str(e.get("name"))] = e
    return out


def _build_ck_runtime_init_payload(run_dir: Path, runtime_summary: dict) -> dict:
    """Build flattened fp32 weight payload for ck_train_init from run_dir bump+manifest."""
    manifest_path = run_dir / "weights_manifest.json"
    bump_path = run_dir / "weights.bump"
    if not manifest_path.exists() or not bump_path.exists():
        raise RuntimeError(f"Missing run_dir weights artifacts: {manifest_path} / {bump_path}")

    order = runtime_summary.get("init_weight_order") if isinstance(runtime_summary, dict) else None
    expected_numel = runtime_summary.get("init_weight_numel") if isinstance(runtime_summary, dict) else None
    if not isinstance(order, list) or not order:
        raise RuntimeError("generated_train_runtime_summary_v7.json missing init_weight_order")
    if not isinstance(expected_numel, list):
        expected_numel = [0] * len(order)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _manifest_entries_map(manifest)
    bump_blob = bump_path.read_bytes()

    payload = bytearray()
    manifest_sizes: list[int] = []
    loaded: list[dict] = []

    for i, wname_raw in enumerate(order):
        wname = str(wname_raw)
        entry = entries.get(wname)
        if entry is None and ("tiny." + wname) in entries:
            entry = entries.get("tiny." + wname)
        if entry is None:
            raise RuntimeError(f"Runtime init weight not found in manifest: {wname}")

        dtype = str(entry.get("dtype", "")).lower()
        if dtype not in ("fp32", "f32"):
            raise RuntimeError(f"Runtime init only supports fp32 weights ({wname}: {dtype})")

        off = int(entry.get("offset", 0) or 0)
        size = int(entry.get("size", 0) or 0)
        if off < 0 or size <= 0 or (off + size) > len(bump_blob):
            raise RuntimeError(f"Invalid bump span for {wname}: off={off} size={size}")

        src_numel = size // 4
        exp = 0
        if i < len(expected_numel):
            try:
                exp = int(expected_numel[i] or 0)
            except Exception:
                exp = 0
        copy_numel = src_numel
        if exp > 0:
            copy_numel = min(src_numel, exp)

        payload.extend(bump_blob[off:off + copy_numel * 4])
        manifest_sizes.append(int(copy_numel))
        loaded.append({
            "weight": wname,
            "manifest_name": str(entry.get("name", wname)),
            "src_numel": int(src_numel),
            "copy_numel": int(copy_numel),
        })

    total_floats = len(payload) // 4
    if total_floats <= 0:
        float_buf = (ctypes.c_float * 1)(0.0)
    else:
        float_buf = (ctypes.c_float * total_floats).from_buffer_copy(payload)

    if manifest_sizes:
        sizes_buf = (ctypes.c_int * len(manifest_sizes))(*manifest_sizes)
    else:
        sizes_buf = (ctypes.c_int * 1)(0)

    return {
        "float_buffer": float_buf,
        "sizes_buffer": sizes_buf,
        "num_params": len(manifest_sizes),
        "total_floats": int(total_floats),
        "loaded": loaded,
    }


def _decode_memory_diagnostic(diag_rc: int, runtime_summary: dict, diag_meta: Optional[dict] = None) -> dict:
    # Decode negative diagnostic return codes into a stable, operator-friendly
    # phase classification. This keeps CLI error reporting deterministic and
    # allows runbook automation to branch by phase/index/op_id.
    payload: dict = {
        "rc": int(diag_rc),
        "ok": bool(diag_rc >= 0),
        "phase": "unknown",
        "index": None,
    }
    if diag_rc >= 0:
        payload["phase"] = "pass"
        return payload

    idx = None
    if diag_rc <= -5000:
        payload["phase"] = "backward_trace_canary"
    elif diag_rc <= -400:
        payload["phase"] = "optimizer_canary"
        idx = -400 - int(diag_rc)
    elif diag_rc <= -300:
        payload["phase"] = "backward_canary"
        idx = -300 - int(diag_rc)
    elif diag_rc <= -200:
        payload["phase"] = "weights_readonly"
        idx = -200 - int(diag_rc)
    elif diag_rc <= -100:
        payload["phase"] = "forward_canary"
        idx = -100 - int(diag_rc)
    elif diag_rc <= -10:
        payload["phase"] = "plant_canary"
        idx = -10 - int(diag_rc)
    else:
        payload["phase"] = "runtime_error"

    if idx is not None and idx >= 0:
        payload["index"] = int(idx)

    canary_ranges = runtime_summary.get("canary_ranges") if isinstance(runtime_summary, dict) else None
    tensor_slots = runtime_summary.get("tensor_slots") if isinstance(runtime_summary, dict) else None

    if payload["phase"].endswith("canary") and isinstance(idx, int) and idx >= 0:
        if isinstance(canary_ranges, list) and idx < len(canary_ranges):
            payload["range"] = canary_ranges[idx]
        else:
            tail_len = int(runtime_summary.get("canary_tail_floats", 0) or 0) if isinstance(runtime_summary, dict) else 0
            tail_idx = idx - (len(canary_ranges) if isinstance(canary_ranges, list) else 0)
            if tail_len > 0 and tail_idx >= 0:
                payload["tail_canary"] = {
                    "tail_index": int(tail_idx),
                    "tail_length": int(tail_len),
                }

    if payload["phase"] == "weights_readonly" and isinstance(idx, int) and idx >= 0 and isinstance(tensor_slots, list) and idx < len(tensor_slots):
        payload["slot"] = tensor_slots[idx]

    if payload["phase"] == "backward_trace_canary" and isinstance(diag_meta, dict):
        op_id = diag_meta.get("failed_op_id")
        canary_idx = diag_meta.get("failed_canary_idx")
        if isinstance(op_id, int) and op_id >= 0:
            payload["failed_op_id"] = int(op_id)
            if isinstance(runtime_summary, dict):
                trace_rows = runtime_summary.get("backward_op_trace")
                if isinstance(trace_rows, list):
                    for row in trace_rows:
                        try:
                            if int(row.get("op_id", -1)) == int(op_id):
                                payload["failed_op"] = row
                                break
                        except Exception:
                            continue
        if isinstance(canary_idx, int) and canary_idx >= 0:
            payload["index"] = int(canary_idx)
            if isinstance(canary_ranges, list) and canary_idx < len(canary_ranges):
                payload["range"] = canary_ranges[canary_idx]
            else:
                tail_len = int(runtime_summary.get("canary_tail_floats", 0) or 0) if isinstance(runtime_summary, dict) else 0
                tail_idx = canary_idx - (len(canary_ranges) if isinstance(canary_ranges, list) else 0)
                if tail_len > 0 and tail_idx >= 0:
                    payload["tail_canary"] = {
                        "tail_index": int(tail_idx),
                        "tail_length": int(tail_len),
                    }

    return payload


def _compute_parity_check_steps(total_steps: int, profile: str, parity_every: int) -> set[int]:
    if total_steps <= 0:
        return set()
    if parity_every and parity_every > 0:
        return {s for s in range(1, total_steps + 1) if (s % parity_every) == 0}

    profile = str(profile or "balanced").lower()
    steps: set[int] = set()
    if profile == "debug":
        steps = {s for s in range(1, total_steps + 1) if (s % 10) == 0 or s == 1}
    elif profile == "light":
        steps.update({1, 10, 100})
        steps.update({s for s in range(1, total_steps + 1) if s >= 100 and (s % 500) == 0})
    else:  # balanced
        for s in range(1, total_steps + 1):
            if s <= 100 and (s % 10) == 0:
                steps.add(s)
            elif s <= 1000 and (s % 50) == 0:
                steps.add(s)
            elif s % 500 == 0:
                steps.add(s)
        steps.add(1)
    return {s for s in steps if 1 <= s <= total_steps}


def _run_ck_oracle_reference(
    args: argparse.Namespace,
    run_dir: Path,
    train_text: Optional[str],
    max_steps: Optional[int] = None,
) -> Optional[dict]:
    """Run periodic PyTorch oracle reference once and return parsed JSON payload."""
    train_script = SCRIPTS_DIR / "train_parity_epochs_v7.py"
    if not train_script.exists():
        log("  Warning: parity oracle script missing; skipping oracle replay", C_ORANGE)
        return None

    parity_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(parity_python) if parity_python.exists() else sys.executable
    oracle_json = run_dir / "oracle_reference_latest.json"

    cmd = [
        python_exec,
        str(train_script),
        "--epochs", str(getattr(args, "train_epochs", 3)),
        "--seq-len", str(getattr(args, "train_seq_len", 16)),
        "--total-tokens", str(getattr(args, "train_total_tokens", 1024)),
        "--grad-accum", str(getattr(args, "train_grad_accum", 8)),
        "--optimizer", str(getattr(args, "train_optimizer", "adamw")),
        "--lr", str(getattr(args, "train_lr", 1e-3)),
        "--seed", str(getattr(args, "train_seed", 42)),
        "--vocab", str(getattr(args, "train_vocab", 256)),
        "--d-model", str(getattr(args, "train_d_model", 64)),
        "--hidden", str(getattr(args, "train_hidden", 128)),
        "--loss-tol", str(getattr(args, "train_loss_tol", 2e-5)),
        "--param-tol", str(getattr(args, "train_param_tol", 3e-5)),
        "--json-out", str(oracle_json),
    ]

    if max_steps is not None and int(max_steps) > 0:
        cmd.extend(["--max-steps", str(int(max_steps))])

    bump = run_dir / "weights.bump"
    manifest = run_dir / "weights_manifest.json"
    if bump.exists() and manifest.exists():
        cmd.extend(["--weights-bump", str(bump), "--weights-manifest", str(manifest)])
    if train_text:
        cmd.extend(["--train-text", train_text])

    rc = run_cmd_allow_fail(cmd, cwd=PROJECT_ROOT).returncode
    if rc != 0 or not oracle_json.exists():
        log("  Warning: parity oracle replay failed; continuing without oracle data", C_ORANGE)
        return None

    try:
        return json.loads(oracle_json.read_text(encoding="utf-8"))
    except Exception:
        log("  Warning: failed to parse oracle_reference_latest.json", C_ORANGE)
        return None


def _ck_export_runtime_weight_snapshot(lib: ctypes.CDLL) -> Optional[tuple[object, int]]:
    """Export current CK runtime weights into a contiguous float snapshot buffer."""
    if not hasattr(lib, "ck_train_get_weight_snapshot_numel") or not hasattr(lib, "ck_train_export_weight_snapshot"):
        return None
    try:
        numel = int(lib.ck_train_get_weight_snapshot_numel())
    except Exception:
        return None
    if numel <= 0 or numel > (1 << 30):
        return None
    buf = (ctypes.c_float * numel)()
    try:
        wrote = int(lib.ck_train_export_weight_snapshot(buf, ctypes.c_int(numel)))
    except Exception:
        return None
    if wrote <= 0:
        return None
    if wrote < numel:
        # Keep deterministic sizing for import/replay: truncate to returned size.
        trunc = (ctypes.c_float * wrote)()
        ctypes.memmove(ctypes.addressof(trunc), ctypes.addressof(buf), wrote * ctypes.sizeof(ctypes.c_float))
        return trunc, int(wrote)
    return buf, int(numel)


def _ck_import_runtime_weight_snapshot(lib: ctypes.CDLL, snapshot_buf: object, snapshot_numel: int) -> int:
    """Import a previously exported CK runtime weight snapshot."""
    if not hasattr(lib, "ck_train_import_weight_snapshot"):
        return -1
    try:
        return int(lib.ck_train_import_weight_snapshot(snapshot_buf, ctypes.c_int(int(snapshot_numel))))
    except Exception:
        return -2


def _ck_export_runtime_activation_snapshot(lib: ctypes.CDLL) -> Optional[tuple[object, int]]:
    """Export current runtime activations/saved tensors snapshot."""
    if not hasattr(lib, "ck_train_get_activation_snapshot_numel") or not hasattr(lib, "ck_train_export_activation_snapshot"):
        return None
    try:
        numel = int(lib.ck_train_get_activation_snapshot_numel())
    except Exception:
        return None
    if numel <= 0 or numel > (1 << 30):
        return None
    buf = (ctypes.c_float * numel)()
    try:
        wrote = int(lib.ck_train_export_activation_snapshot(buf, ctypes.c_int(numel)))
    except Exception:
        return None
    if wrote <= 0:
        return None
    if wrote < numel:
        trunc = (ctypes.c_float * wrote)()
        ctypes.memmove(ctypes.addressof(trunc), ctypes.addressof(buf), wrote * ctypes.sizeof(ctypes.c_float))
        return trunc, int(wrote)
    return buf, int(numel)


def _write_ck_weight_snapshot_artifact(
    run_dir: Path,
    step: int,
    snapshot_buf: object,
    snapshot_numel: int,
    *,
    reason: str,
) -> Optional[Path]:
    """Persist CK weight snapshot for drift/replay triage."""
    try:
        snap_dir = run_dir / "oracle_ck_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        bin_path = snap_dir / f"step_{int(step):08d}.f32bin"
        meta_path = snap_dir / f"step_{int(step):08d}.json"
        raw = ctypes.string_at(ctypes.addressof(snapshot_buf), int(snapshot_numel) * ctypes.sizeof(ctypes.c_float))
        bin_path.write_bytes(raw)
        meta = {
            "generated_at": _utc_now_iso(),
            "step": int(step),
            "numel": int(snapshot_numel),
            "bytes": int(len(raw)),
            "reason": str(reason),
            "file": str(bin_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return bin_path
    except Exception:
        return None


def _write_ck_activation_snapshot_artifact(
    run_dir: Path,
    step: int,
    snapshot_buf: object,
    snapshot_numel: int,
    *,
    reason: str,
) -> Optional[Path]:
    """Persist CK activation snapshot for drift/replay triage."""
    try:
        snap_dir = run_dir / "oracle_ck_activations"
        snap_dir.mkdir(parents=True, exist_ok=True)
        bin_path = snap_dir / f"step_{int(step):08d}.f32bin"
        meta_path = snap_dir / f"step_{int(step):08d}.json"
        raw = ctypes.string_at(ctypes.addressof(snapshot_buf), int(snapshot_numel) * ctypes.sizeof(ctypes.c_float))
        bin_path.write_bytes(raw)
        meta = {
            "generated_at": _utc_now_iso(),
            "step": int(step),
            "numel": int(snapshot_numel),
            "bytes": int(len(raw)),
            "reason": str(reason),
            "file": str(bin_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return bin_path
    except Exception:
        return None




def _write_ck_weight_checkpoint_bump(
    run_dir: Path,
    runtime_summary: dict,
    snapshot_buf: object,
    snapshot_numel: int,
    *,
    step: int,
    reason: str,
) -> Optional[dict]:
    """Write a resumable weights checkpoint (.bump + manifest) from runtime snapshot."""
    try:
        init_order = runtime_summary.get("init_weight_order") if isinstance(runtime_summary, dict) else None
        init_numel = runtime_summary.get("init_weight_numel") if isinstance(runtime_summary, dict) else None
        if not isinstance(init_order, list) or not isinstance(init_numel, list):
            return None
        if len(init_order) != len(init_numel) or len(init_order) == 0:
            return None

        expected_total = 0
        for n in init_numel:
            expected_total += int(n or 0)
        if expected_total <= 0 or int(snapshot_numel) < expected_total:
            return None

        raw = ctypes.string_at(
            ctypes.addressof(snapshot_buf),
            int(snapshot_numel) * ctypes.sizeof(ctypes.c_float),
        )
        if len(raw) < expected_total * 4:
            return None

        src_manifest = {}
        src_manifest_path = run_dir / "weights_manifest.json"
        if src_manifest_path.exists():
            try:
                src_manifest = json.loads(src_manifest_path.read_text(encoding="utf-8"))
            except Exception:
                src_manifest = {}
        src_entries = src_manifest.get("entries") if isinstance(src_manifest, dict) else None
        src_entry_by_name: dict[str, dict] = {}
        if isinstance(src_entries, list):
            for row in src_entries:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name", "") or "")
                if name:
                    src_entry_by_name[name] = row

        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        stem = f"weights_step_{int(step):08d}"
        bump_path = ckpt_dir / f"{stem}.bump"
        manifest_path = ckpt_dir / f"{stem}_manifest.json"

        blob = bytearray()
        entries = []
        cursor_bytes = 0
        for idx, wname in enumerate(init_order):
            numel = int(init_numel[idx] or 0)
            if numel <= 0:
                continue
            nbytes = numel * 4
            if cursor_bytes + nbytes > len(raw):
                return None
            offset = _align_up(len(blob), 64)
            if offset > len(blob):
                blob.extend(b"\x00" * (offset - len(blob)))
            blob.extend(raw[cursor_bytes:cursor_bytes + nbytes])
            cursor_bytes += nbytes

            src = src_entry_by_name.get(str(wname), {})
            entry = {
                "name": str(wname),
                "offset": int(offset),
                "size": int(nbytes),
                "dtype": str(src.get("dtype", "fp32") or "fp32"),
            }
            shape = src.get("shape")
            if isinstance(shape, list) and shape:
                entry["shape"] = shape
            else:
                entry["shape"] = [int(numel)]
            entries.append(entry)

        manifest_out = {
            "version": 1,
            "format": "weights_manifest_v7_runtime_checkpoint",
            "source": "ck_runtime_snapshot",
            "step": int(step),
            "reason": str(reason),
            "entries": entries,
        }
        if isinstance(src_manifest, dict):
            if isinstance(src_manifest.get("config"), dict):
                manifest_out["config"] = src_manifest["config"]
            if isinstance(src_manifest.get("template"), dict):
                manifest_out["template"] = src_manifest["template"]

        bump_path.write_bytes(bytes(blob))
        manifest_path.write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
        return {
            "step": int(step),
            "reason": str(reason),
            "bump": str(bump_path),
            "manifest": str(manifest_path),
            "weights": int(len(entries)),
            "floats": int(expected_total),
            "bytes": int(len(blob)),
        }
    except Exception:
        return None

def _extract_activation_slots_ordered(
    runtime_summary: dict,
    snapshot: object,
    snapshot_numel: int,
):
    """Decode flattened activation snapshot into ordered slot payloads."""
    try:
        import numpy as _np
    except Exception:
        return None

    rows = runtime_summary.get("tensor_slots") if isinstance(runtime_summary, dict) else None
    if not isinstance(rows, list):
        return None

    act_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        # Keep strict slot comparison scoped to forward activation tensors only.
        # saved.* and aux.* are intentionally excluded because they are not stable
        # forward-equivalent observability surfaces across kernels/backends.
        if str(row.get("section", "")) not in ("activations",):
            continue
        try:
            act_rows.append({
                "name": str(row.get("name", "")),
                "offset": int(row.get("offset", 0) or 0),
                "numel": int(row.get("numel", 0) or 0),
            })
        except Exception:
            continue
    act_rows.sort(key=lambda r: (r["offset"], r["name"]))

    snap_np = _np.ctypeslib.as_array(snapshot, shape=(int(snapshot_numel),)).astype(_np.float32, copy=False)

    cursor = 0
    out = []
    for row in act_rows:
        numel = int(row.get("numel", 0))
        if numel <= 0:
            continue
        if cursor + numel > snap_np.size:
            return None
        out.append(
            {
                "name": str(row.get("name", "")),
                "offset": int(row.get("offset", 0)),
                "numel": int(numel),
                "flat": snap_np[cursor:cursor + numel].copy(),
            }
        )
        cursor += numel
    return out


def _extract_activation_slot_flat(
    runtime_summary: dict,
    snapshot: object,
    snapshot_numel: int,
    *,
    slot_name_exact: str,
    slot_name_contains: Optional[str] = None,
):
    """Extract one activation tensor from flattened activation snapshot order."""
    rows = _extract_activation_slots_ordered(runtime_summary, snapshot, snapshot_numel)
    if not isinstance(rows, list):
        return None, None

    for row in rows:
        name = str(row.get("name", ""))
        hit = (name == slot_name_exact)
        if not hit and slot_name_contains:
            hit = slot_name_contains in name
        if hit:
            return row.get("flat"), name
    return None, None


def _slot_op_hint_from_name(name: str) -> str:
    """Best-effort op hint from activation slot name."""
    try:
        m = re.match(r"^act\.L(\d+)\.([^.]+)\.", str(name))
        if m:
            return f"layer_{int(m.group(1))}:{m.group(2)}"
        m = re.match(r"^act\.S([^.]+)\.([^.]+)\.", str(name))
        if m:
            return f"stage_{m.group(1).lower()}:{m.group(2)}"
        m = re.match(r"^saved\.op(\d+)\.([^.]+)", str(name))
        if m:
            return f"op{int(m.group(1))}:{m.group(2)}"
    except Exception:
        pass
    return str(name)


def _infer_runtime_num_tokens(runtime_summary: dict, d_model_hint: int, vocab_hint: int) -> int:
    """Infer compile-time CK_NUM_TOKENS from activation slot shapes."""
    rows = runtime_summary.get("tensor_slots") if isinstance(runtime_summary, dict) else None
    if not isinstance(rows, list):
        return 1

    d_model = max(1, int(d_model_hint or 0))
    vocab = max(1, int(vocab_hint or 0))

    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        numel = int(row.get("numel", 0) or 0)
        if numel <= 0:
            continue
        if name == "act.Sheader.dense_embedding_lookup.0.out" and (numel % d_model) == 0:
            return max(1, numel // d_model)
        if name == "act.Sfooter.logits.0.y" and (numel % vocab) == 0:
            return max(1, numel // vocab)
    return 1


def _compile_train_runtime_variant(
    run_dir: Path,
    c_src: Path,
    suffix: str,
    *,
    defines: Optional[dict] = None,
    asan: bool = False,
) -> Path:
    out_so = run_dir / f"libtrain_{suffix}.so"
    cc = os.environ.get("CC") or "gcc"
    cmd = [
        cc,
        "-shared",
        "-fPIC",
        "-O1" if asan else "-O3",
        str(c_src),
        "-o",
        str(out_so),
        "-I",
        str(PROJECT_ROOT / "include"),
        "-I",
        str(PROJECT_ROOT),
        "-L",
        str(BUILD_DIR),
        "-lckernel_engine",
        "-lm",
        f"-Wl,-rpath,{BUILD_DIR}",
    ]
    if asan:
        cmd.extend(["-fsanitize=address", "-fno-omit-frame-pointer"])
    for k, v in sorted((defines or {}).items()):
        cmd.append(f"-D{k}={int(v)}")
    run_cmd(cmd, cwd=PROJECT_ROOT)
    return out_so


def _run_ck_memory_diag_direct(lib_path: Path, init_payload: dict) -> dict:
    lib = ctypes.CDLL(str(lib_path))
    if not hasattr(lib, "ck_train_init") or not hasattr(lib, "ck_train_memory_diagnostic"):
        raise RuntimeError(f"Missing diagnostic symbols in {lib_path}")

    lib.ck_train_init.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.ck_train_init.restype = ctypes.c_int
    lib.ck_train_memory_diagnostic.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float]
    lib.ck_train_memory_diagnostic.restype = ctypes.c_int

    has_op = bool(hasattr(lib, "ck_train_get_last_diag_failed_op"))
    has_canary = bool(hasattr(lib, "ck_train_get_last_diag_failed_canary"))
    if has_op:
        lib.ck_train_get_last_diag_failed_op.argtypes = []
        lib.ck_train_get_last_diag_failed_op.restype = ctypes.c_int
    if has_canary:
        lib.ck_train_get_last_diag_failed_canary.argtypes = []
        lib.ck_train_get_last_diag_failed_canary.restype = ctypes.c_int

    float_ptr = ctypes.cast(init_payload["float_buffer"], ctypes.POINTER(ctypes.c_float))
    size_ptr = ctypes.cast(init_payload["sizes_buffer"], ctypes.POINTER(ctypes.c_int))
    init_rc = int(lib.ck_train_init(float_ptr, size_ptr, ctypes.c_int(init_payload["num_params"])))
    if init_rc < 0:
        return {"init_rc": init_rc, "diag_rc": None, "failed_op_id": None, "failed_canary_idx": None}

    diag_rc = int(lib.ck_train_memory_diagnostic(None, None, ctypes.c_float(0.0)))
    return {
        "init_rc": init_rc,
        "diag_rc": diag_rc,
        "failed_op_id": int(lib.ck_train_get_last_diag_failed_op()) if has_op else None,
        "failed_canary_idx": int(lib.ck_train_get_last_diag_failed_canary()) if has_canary else None,
    }


def _probe_ck_runtime_loss_curve(lib_path: Path, init_payload: dict, batches: list, steps: int, lr: float) -> dict:
    lib = ctypes.CDLL(str(lib_path))
    if not hasattr(lib, "ck_train_step") or not hasattr(lib, "ck_train_init"):
        raise RuntimeError(f"Missing training symbols in {lib_path}")

    lib.ck_train_init.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.ck_train_init.restype = ctypes.c_int
    lib.ck_train_step.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_float]
    lib.ck_train_step.restype = ctypes.c_int

    float_ptr = ctypes.cast(init_payload["float_buffer"], ctypes.POINTER(ctypes.c_float))
    size_ptr = ctypes.cast(init_payload["sizes_buffer"], ctypes.POINTER(ctypes.c_int))
    init_rc = int(lib.ck_train_init(float_ptr, size_ptr, ctypes.c_int(init_payload["num_params"])))
    out = {"init_rc": init_rc, "step_rcs": [], "losses": []}
    if init_rc < 0:
        return out

    limit = max(1, min(int(steps), len(batches)))
    for i in range(limit):
        x_vals, y_vals = batches[i]
        x_buf = (ctypes.c_int32 * len(x_vals))(*x_vals)
        y_buf = (ctypes.c_int32 * len(y_vals))(*y_vals)
        loss_out = ctypes.c_float(0.0)
        rc = int(lib.ck_train_step(x_buf, y_buf, ctypes.byref(loss_out), ctypes.c_float(lr)))
        out["step_rcs"].append(int(rc))
        out["losses"].append(float(loss_out.value))
        if rc < 0:
            break
    return out


def _run_asan_diag_subprocess(run_dir: Path, lib_path: Path) -> dict:
    py = sys.executable
    script = r'''
import ctypes, json, pathlib, sys
run = pathlib.Path(sys.argv[1])
lib_path = pathlib.Path(sys.argv[2])
summary = json.loads((run / "generated_train_runtime_summary_v7.json").read_text(encoding="utf-8"))
manifest = json.loads((run / "weights_manifest.json").read_text(encoding="utf-8"))
bump_blob = (run / "weights.bump").read_bytes()
entries = {e.get("name"): e for e in (manifest.get("entries") or []) if isinstance(e, dict) and e.get("name")}
payload = bytearray()
sizes = []
for i, wname in enumerate(summary.get("init_weight_order") or []):
    e = entries.get(wname) or entries.get("tiny." + str(wname))
    if not isinstance(e, dict):
        raise RuntimeError(f"missing manifest entry for {wname}")
    off = int(e.get("offset", 0) or 0)
    size = int(e.get("size", 0) or 0)
    src_numel = size // 4
    nums = summary.get("init_weight_numel") or []
    exp = int(nums[i] or 0) if i < len(nums) else src_numel
    copy_numel = min(src_numel, exp if exp > 0 else src_numel)
    payload.extend(bump_blob[off:off + copy_numel * 4])
    sizes.append(copy_numel)
FloatArray = ctypes.c_float * (len(payload) // 4 if payload else 1)
float_buf = FloatArray.from_buffer_copy(payload) if payload else FloatArray(0.0)
IntArray = ctypes.c_int * (len(sizes) if sizes else 1)
size_buf = IntArray(*sizes) if sizes else IntArray(0)
lib = ctypes.CDLL(str(lib_path))
lib.ck_train_init.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.ck_train_init.restype = ctypes.c_int
lib.ck_train_memory_diagnostic.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float]
lib.ck_train_memory_diagnostic.restype = ctypes.c_int
init_rc = int(lib.ck_train_init(ctypes.cast(float_buf, ctypes.POINTER(ctypes.c_float)), ctypes.cast(size_buf, ctypes.POINTER(ctypes.c_int)), ctypes.c_int(len(sizes))))
diag_rc = int(lib.ck_train_memory_diagnostic(None, None, ctypes.c_float(0.0))) if init_rc >= 0 else -9999
print(json.dumps({"init_rc": init_rc, "diag_rc": diag_rc}))
'''
    env = dict(os.environ)
    asan_lib = subprocess.check_output(["gcc", "-print-file-name=libasan.so"], text=True).strip()
    if asan_lib and Path(asan_lib).exists():
        preload = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = asan_lib if not preload else f"{asan_lib}:{preload}"
    env.setdefault("ASAN_OPTIONS", "detect_leaks=0")
    proc = subprocess.run([py, "-c", script, str(run_dir), str(lib_path)], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    stderr = (proc.stderr or "")[-4000:]
    stdout = (proc.stdout or "").strip()
    parsed = None
    if stdout:
        try:
            parsed = json.loads(stdout.splitlines()[-1])
        except Exception:
            parsed = None
    return {
        "returncode": int(proc.returncode),
        "stdout": stdout[-1000:],
        "stderr": stderr,
        "parsed": parsed,
        "asan_detected": ("AddressSanitizer" in stderr) or ("heap-buffer-overflow" in stderr),
    }


def _run_pr37_memory_verification(
    args: argparse.Namespace,
    run_dir: Path,
    c_src: Path,
    runtime_summary: dict,
    init_payload: dict,
    batches: list,
    lr: float,
) -> dict:
    steps = max(1, int(getattr(args, "train_verify_steps", 4) or 4))
    fault_op = int(getattr(args, "train_verify_fault_op_id", -1) or -1)
    if fault_op < 0:
        trace = runtime_summary.get("backward_op_trace") if isinstance(runtime_summary, dict) else None
        if isinstance(trace, list) and trace:
            gemm_ops = [
                int(row.get("op_id", -1) or -1)
                for row in trace
                if str(row.get("kernel_id", "")) == "gemm_backward_f32"
            ]
            candidates = [x for x in gemm_ops if x >= 0]
            if not candidates:
                candidates = [int(row.get("op_id", -1) or -1) for row in trace if int(row.get("op_id", -1) or -1) >= 0]
            fault_op = max(candidates) if candidates else 1
        else:
            fault_op = 1

    report: dict = {
        "generated_at": _utc_now_iso(),
        "run_dir": str(run_dir),
        "steps": steps,
        "fault_op_id": fault_op,
        "checks": {},
    }

    lib_off = _compile_train_runtime_variant(run_dir, c_src, "verify_canary_off", defines={"CK_RUNTIME_CANARY_CHECKS": 0, "CK_RUNTIME_BOUNDS_ASSERT": 1})
    lib_on = _compile_train_runtime_variant(run_dir, c_src, "verify_canary_on", defines={"CK_RUNTIME_CANARY_CHECKS": 1, "CK_RUNTIME_BOUNDS_ASSERT": 1})
    probe_off = _probe_ck_runtime_loss_curve(lib_off, init_payload, batches, steps, lr)
    probe_on = _probe_ck_runtime_loss_curve(lib_on, init_payload, batches, steps, lr)
    losses_off = probe_off.get("losses") or []
    losses_on = probe_on.get("losses") or []
    n = min(len(losses_off), len(losses_on))
    max_diff = max((abs(float(losses_off[i]) - float(losses_on[i])) for i in range(n)), default=0.0)
    toggle_ok = (
        int(probe_off.get("init_rc", -1)) >= 0 and
        int(probe_on.get("init_rc", -1)) >= 0 and
        all(int(x) >= 0 for x in (probe_off.get("step_rcs") or [])) and
        all(int(x) >= 0 for x in (probe_on.get("step_rcs") or [])) and
        n > 0 and
        max_diff <= 1e-12
    )
    report["checks"]["toggle_diff"] = {
        "ok": bool(toggle_ok),
        "max_loss_abs_diff": float(max_diff),
        "samples_compared": int(n),
        "off": probe_off,
        "on": probe_on,
    }

    lib_fault = _compile_train_runtime_variant(
        run_dir,
        c_src,
        "verify_fault_canary",
        defines={
            "CK_RUNTIME_CANARY_CHECKS": 1,
            "CK_RUNTIME_BOUNDS_ASSERT": 1,
            "CK_RUNTIME_FAULT_INJECT": 1,
            "CK_FAULT_INJECT_OP_ID": int(fault_op),
        },
    )
    fault_diag = _run_ck_memory_diag_direct(lib_fault, init_payload)
    fault_decoded = _decode_memory_diagnostic(int(fault_diag.get("diag_rc") or -9999), runtime_summary, {
        "failed_op_id": fault_diag.get("failed_op_id"),
        "failed_canary_idx": fault_diag.get("failed_canary_idx"),
    })
    fault_ok = (
        isinstance(fault_diag.get("diag_rc"), int) and int(fault_diag["diag_rc"]) < 0 and
        str(fault_decoded.get("phase")) == "backward_trace_canary" and
        int(fault_decoded.get("failed_op_id", -1)) == int(fault_op)
    )
    report["checks"]["intentional_plus1"] = {
        "ok": bool(fault_ok),
        "diag": fault_diag,
        "decoded": fault_decoded,
    }

    asan_clean = _compile_train_runtime_variant(run_dir, c_src, "verify_asan_clean", defines={"CK_RUNTIME_CANARY_CHECKS": 1, "CK_RUNTIME_BOUNDS_ASSERT": 1}, asan=True)
    asan_fault = _compile_train_runtime_variant(
        run_dir,
        c_src,
        "verify_asan_fault",
        defines={
            "CK_RUNTIME_CANARY_CHECKS": 1,
            "CK_RUNTIME_BOUNDS_ASSERT": 1,
            "CK_RUNTIME_FAULT_INJECT": 1,
            "CK_FAULT_INJECT_OP_ID": int(fault_op),
        },
        asan=True,
    )
    asan_clean_res = _run_asan_diag_subprocess(run_dir, asan_clean)
    asan_fault_res = _run_asan_diag_subprocess(run_dir, asan_fault)
    clean_diag = int((asan_clean_res.get("parsed") or {}).get("diag_rc", -1)) if isinstance(asan_clean_res.get("parsed"), dict) else -1
    fault_diag = int((asan_fault_res.get("parsed") or {}).get("diag_rc", 1)) if isinstance(asan_fault_res.get("parsed"), dict) else 1
    asan_ok = (
        int(asan_clean_res.get("returncode", 1)) == 0 and
        clean_diag >= 0 and
        (
            (int(asan_fault_res.get("returncode", 1)) == 0 and fault_diag < 0) or
            (int(asan_fault_res.get("returncode", 0)) != 0 and bool(asan_fault_res.get("asan_detected")))
        )
    )
    report["checks"]["asan_agreement"] = {
        "ok": bool(asan_ok),
        "clean": asan_clean_res,
        "fault": asan_fault_res,
    }

    bounds_present = bool(runtime_summary.get("tensor_slot_count", 0))
    bounds_ok = bool(bounds_present and int(probe_on.get("init_rc", -1)) >= 0)
    report["checks"]["bounds_assertions"] = {
        "ok": bool(bounds_ok),
        "enabled_in_variants": True,
        "tensor_slot_count": int(runtime_summary.get("tensor_slot_count", 0) or 0),
    }

    report["ok"] = all(bool(v.get("ok")) for v in (report.get("checks") or {}).values())
    return report



def _ensure_train_runtime_artifacts(run_dir: Path, python_exec: str, strict: bool, runtime_defines: Optional[dict] = None) -> tuple[Path, Path]:
    """Ensure run_dir has IR1/IR2/layout/audits and compiled libtrain.so."""
    # This is the train-runtime artifact chain in one place:
    # manifest -> IR1 -> IR2 -> invariants -> layout -> layout audit -> codegen -> libtrain.so
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "weights_manifest.json"
    if not manifest.exists():
        raise RuntimeError(f"Missing run_dir manifest: {manifest} (run `cks-v7-run init --generate-ir --generate-runtime` first)")

    ir1 = run_dir / "ir1_train_forward.json"
    ir1_report = run_dir / "ir1_train_report.json"
    ir2 = run_dir / "ir2_train_backward.json"
    ir2_summary = run_dir / "ir2_train_summary.json"
    inv = run_dir / "ir_train_invariants.json"
    layout_train = run_dir / "layout_train.json"
    layout_audit = run_dir / "layout_train_audit.json"
    c_src = run_dir / "generated_train_runtime_v7.c"
    c_summary = run_dir / "generated_train_runtime_summary_v7.json"

    build_ir_script = SCRIPTS_DIR / "build_ir_train_v7.py"
    lower_ir_script = SCRIPTS_DIR / "lower_ir2_backward_v7.py"
    inv_script = SCRIPTS_DIR / "validate_ir_train_invariants_v7.py"
    layout_script = SCRIPTS_DIR / "generate_train_layout_v7.py"
    layout_audit_script = SCRIPTS_DIR / "validate_train_memory_layout_v7.py"

    # Each stage only regenerates when inputs are newer. This keeps CLI reruns fast
    # while still guaranteeing that libtrain is rebuilt after contract changes.
    needs_ir1 = (
        (not ir1.exists())
        or (manifest.exists() and manifest.stat().st_mtime > ir1.stat().st_mtime)
        or (build_ir_script.exists() and build_ir_script.stat().st_mtime > ir1.stat().st_mtime)
    )
    if needs_ir1:
        cmd = [
            python_exec,
            str(build_ir_script),
            "--manifest", str(manifest),
            "--output", str(ir1),
            "--report-out", str(ir1_report),
        ]
        if strict:
            cmd.append("--strict")
        run_cmd(cmd, cwd=PROJECT_ROOT)

    needs_ir2 = (
        (not ir2.exists())
        or strict
        or (ir1.exists() and ir1.stat().st_mtime > ir2.stat().st_mtime)
        or (lower_ir_script.exists() and lower_ir_script.stat().st_mtime > ir2.stat().st_mtime)
    )
    if needs_ir2:
        cmd = [
            python_exec,
            str(lower_ir_script),
            "--ir1", str(ir1),
            "--output", str(ir2),
            "--summary-out", str(ir2_summary),
        ]
        if strict:
            cmd.append("--strict")
        else:
            cmd.append("--allow-partial")
        run_cmd(cmd, cwd=PROJECT_ROOT)

    needs_inv = (
        (not inv.exists())
        or strict
        or (ir2.exists() and ir2.stat().st_mtime > inv.stat().st_mtime)
        or (inv_script.exists() and inv_script.stat().st_mtime > inv.stat().st_mtime)
    )
    if needs_inv:
        cmd = [
            python_exec,
            str(inv_script),
            "--ir1", str(ir1),
            "--ir2", str(ir2),
            "--output", str(inv),
        ]
        if strict:
            cmd.append("--strict-unresolved")
        else:
            cmd.append("--allow-partial")
        run_cmd(cmd, cwd=PROJECT_ROOT)

    needs_layout = (
        (not layout_train.exists())
        or (ir2.exists() and ir2.stat().st_mtime > layout_train.stat().st_mtime)
        or (manifest.exists() and manifest.stat().st_mtime > layout_train.stat().st_mtime)
        or (layout_script.exists() and layout_script.stat().st_mtime > layout_train.stat().st_mtime)
    )
    if needs_layout:
        cmd = [
            python_exec,
            str(layout_script),
            "--ir2", str(ir2),
            "--manifest", str(manifest),
            "--output", str(layout_train),
            "--align-bytes", "64",
        ]
        if strict:
            cmd.append("--strict")
        run_cmd(cmd, cwd=PROJECT_ROOT)

    needs_layout_audit = (
        (not layout_audit.exists())
        or strict
        or (layout_train.exists() and layout_train.stat().st_mtime > layout_audit.stat().st_mtime)
        or (ir2.exists() and ir2.stat().st_mtime > layout_audit.stat().st_mtime)
        or (layout_audit_script.exists() and layout_audit_script.stat().st_mtime > layout_audit.stat().st_mtime)
    )
    if needs_layout_audit:
        cmd = [
            python_exec,
            str(layout_audit_script),
            "--layout", str(layout_train),
            "--ir2", str(ir2),
            "--output", str(layout_audit),
        ]
        if strict:
            cmd.append("--strict")
        run_cmd(cmd, cwd=PROJECT_ROOT)

    codegen_script = SCRIPTS_DIR / "codegen_train_runtime_v7.py"
    regen_codegen = (
        (not c_src.exists())
        or (not c_summary.exists())
        or (ir2.exists() and ir2.stat().st_mtime > c_src.stat().st_mtime)
        or (codegen_script.exists() and codegen_script.stat().st_mtime > c_src.stat().st_mtime)
    )
    if regen_codegen:
        cmd = [
            python_exec,
            str(codegen_script),
            "--ir2", str(ir2),
            "--manifest", str(manifest),
            "--layout", str(layout_train),
            "--output", str(c_src),
            "--summary-out", str(c_summary),
        ]
        run_cmd(cmd, cwd=PROJECT_ROOT)

    lib_ck = BUILD_DIR / "libckernel_engine.so"
    if not lib_ck.exists():
        run_cmd(["make", "--no-print-directory", str(lib_ck)], cwd=PROJECT_ROOT)

    libtrain_so = run_dir / "libtrain.so"
    defines = dict(runtime_defines or {})
    needs_compile = (not libtrain_so.exists()) or (c_src.stat().st_mtime > libtrain_so.stat().st_mtime)
    if defines:
        needs_compile = True
    if needs_compile:
        cc = os.environ.get("CC") or "gcc"
        cmd = [
            cc,
            "-shared",
            "-fPIC",
            "-O3",
            str(c_src),
            "-o",
            str(libtrain_so),
            "-I",
            str(PROJECT_ROOT / "include"),
            "-I",
            str(PROJECT_ROOT),
            "-L",
            str(BUILD_DIR),
            "-lckernel_engine",
            "-lm",
            f"-Wl,-rpath,{BUILD_DIR}",
        ]
        for k, v in sorted(defines.items()):
            cmd.append(f"-D{k}={int(v)}")
        run_cmd(cmd, cwd=PROJECT_ROOT)

    return c_src, libtrain_so


def _run_ck_train_runtime(
    args: argparse.Namespace,
    run_dir: Path,
    json_out: Path,
    train_text: Optional[str],
    train_mode: str,
    train_backend: str,
    profile_meta: dict,
) -> Path:
    """Execute generated training runtime directly via ctypes (PR3 CK backend path)."""
    # PR3 scope: ck backend executes generated C train step and optional oracle checks.
    # Full long-run trainer features (rich profiling, real grad telemetry) are still layered on top.
    parity_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(parity_python) if parity_python.exists() else sys.executable

    runtime_defines: dict = {}
    if bool(getattr(args, "train_runtime_canary_checks", False)):
        runtime_defines["CK_RUNTIME_CANARY_CHECKS"] = 1
    if bool(getattr(args, "train_runtime_bounds_assert", False)):
        runtime_defines["CK_RUNTIME_BOUNDS_ASSERT"] = 1
    fault_op_id = int(getattr(args, "train_runtime_fault_op_id", -1) or -1)
    if fault_op_id >= 0:
        runtime_defines["CK_RUNTIME_FAULT_INJECT"] = 1
        runtime_defines["CK_FAULT_INJECT_OP_ID"] = int(fault_op_id)

    c_src, libtrain_so = _ensure_train_runtime_artifacts(
        run_dir=run_dir,
        python_exec=python_exec,
        strict=bool(getattr(args, "train_strict", False)),
        runtime_defines=runtime_defines,
    )

    runtime_summary_path = run_dir / "generated_train_runtime_summary_v7.json"
    if not runtime_summary_path.exists():
        raise RuntimeError(f"Missing runtime summary: {runtime_summary_path}")
    runtime_summary = json.loads(runtime_summary_path.read_text(encoding="utf-8"))

    init_payload = _build_ck_runtime_init_payload(run_dir, runtime_summary)

    lib = ctypes.CDLL(str(libtrain_so))
    if not hasattr(lib, "ck_train_step"):
        raise RuntimeError(f"Missing ck_train_step symbol in {libtrain_so}")
    if not hasattr(lib, "ck_train_init"):
        raise RuntimeError(f"Missing ck_train_init symbol in {libtrain_so}")

    lib.ck_train_init.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.ck_train_init.restype = ctypes.c_int

    lib.ck_train_step.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
    ]
    lib.ck_train_step.restype = ctypes.c_int

    diag_info = {
        "available": bool(hasattr(lib, "ck_train_memory_diagnostic")),
        "ran": False,
        "rc": None,
        "strict_only": True,
    }
    has_diag_op_getter = bool(hasattr(lib, "ck_train_get_last_diag_failed_op"))
    has_diag_canary_getter = bool(hasattr(lib, "ck_train_get_last_diag_failed_canary"))
    has_snapshot_numel = bool(hasattr(lib, "ck_train_get_weight_snapshot_numel"))
    has_snapshot_export = bool(hasattr(lib, "ck_train_export_weight_snapshot"))
    has_snapshot_import = bool(hasattr(lib, "ck_train_import_weight_snapshot"))
    has_act_snapshot_numel = bool(hasattr(lib, "ck_train_get_activation_snapshot_numel"))
    has_act_snapshot_export = bool(hasattr(lib, "ck_train_export_activation_snapshot"))
    if hasattr(lib, "ck_train_memory_diagnostic"):
        lib.ck_train_memory_diagnostic.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
        ]
        lib.ck_train_memory_diagnostic.restype = ctypes.c_int
    if has_diag_op_getter:
        lib.ck_train_get_last_diag_failed_op.argtypes = []
        lib.ck_train_get_last_diag_failed_op.restype = ctypes.c_int
    if has_diag_canary_getter:
        lib.ck_train_get_last_diag_failed_canary.argtypes = []
        lib.ck_train_get_last_diag_failed_canary.restype = ctypes.c_int
    if has_snapshot_numel:
        lib.ck_train_get_weight_snapshot_numel.argtypes = []
        lib.ck_train_get_weight_snapshot_numel.restype = ctypes.c_int
    if has_snapshot_export:
        lib.ck_train_export_weight_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ck_train_export_weight_snapshot.restype = ctypes.c_int
    if has_snapshot_import:
        lib.ck_train_import_weight_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ck_train_import_weight_snapshot.restype = ctypes.c_int
    if has_act_snapshot_numel:
        lib.ck_train_get_activation_snapshot_numel.argtypes = []
        lib.ck_train_get_activation_snapshot_numel.restype = ctypes.c_int
    if has_act_snapshot_export:
        lib.ck_train_export_activation_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ck_train_export_activation_snapshot.restype = ctypes.c_int

    float_ptr = ctypes.cast(init_payload["float_buffer"], ctypes.POINTER(ctypes.c_float))
    size_ptr = ctypes.cast(init_payload["sizes_buffer"], ctypes.POINTER(ctypes.c_int))
    rc = int(lib.ck_train_init(float_ptr, size_ptr, ctypes.c_int(init_payload["num_params"])))
    if rc < 0:
        raise RuntimeError(f"ck_train_init failed with code {rc}")

    # Strict mode executes memory diagnostics before training micro-steps.
    # If this fails, we abort early with decoded phase/index/op metadata to
    # localize corruption before optimizer updates obscure root cause.
    if bool(getattr(args, "train_strict", False)) and bool(diag_info["available"]):
        diag_rc = int(lib.ck_train_memory_diagnostic(None, None, ctypes.c_float(0.0)))
        diag_info["ran"] = True
        diag_info["rc"] = int(diag_rc)
        # Optional metadata getters are emitted by generated runtime; when present
        # they provide first failing op/canary for backward trace localization.
        diag_meta = {
            "failed_op_id": int(lib.ck_train_get_last_diag_failed_op()) if has_diag_op_getter else None,
            "failed_canary_idx": int(lib.ck_train_get_last_diag_failed_canary()) if has_diag_canary_getter else None,
        }
        decoded = _decode_memory_diagnostic(diag_rc, runtime_summary, diag_meta=diag_meta)
        diag_info["decoded"] = decoded
        diag_path = run_dir / "memory_diagnostic_latest.json"
        diag_payload = {
            "generated_at": _utc_now_iso(),
            "run_dir": str(run_dir),
            "runtime_summary": str(runtime_summary_path),
            "diagnostic": decoded,
            "meta": diag_meta,
        }
        diag_path.write_text(json.dumps(diag_payload, indent=2), encoding="utf-8")
        profile_meta.setdefault("artifacts", []).append({"label": "memory_diagnostic", "path": str(diag_path)})
        if diag_rc < 0:
            phase = str(decoded.get("phase", "unknown"))
            idx = decoded.get("index")
            bad_op_id = decoded.get("failed_op_id")
            bad_op = decoded.get("failed_op") if isinstance(decoded.get("failed_op"), dict) else None
            if isinstance(bad_op, dict):
                raise RuntimeError(
                    "ck_train_memory_diagnostic failed "
                    f"(phase={phase}, index={idx}, op_id={bad_op_id}, op={bad_op.get('op')}, kernel={bad_op.get('kernel_id')}, rc={diag_rc})"
                )
            raise RuntimeError(f"ck_train_memory_diagnostic failed (phase={phase}, index={idx}, op_id={bad_op_id}, rc={diag_rc})")
        rc = int(lib.ck_train_init(float_ptr, size_ptr, ctypes.c_int(init_payload["num_params"])))
        if rc < 0:
            raise RuntimeError(f"ck_train_init (post-diagnostic) failed with code {rc}")

    epochs = int(getattr(args, "train_epochs", 3) or 3)
    seq_len = int(getattr(args, "train_seq_len", 16) or 16)
    total_tokens = int(getattr(args, "train_total_tokens", 1024) or 1024)
    grad_accum = int(getattr(args, "train_grad_accum", 8) or 8)
    lr = float(getattr(args, "train_lr", 1e-3) or 1e-3)
    seed = int(getattr(args, "train_seed", 42) or 42)
    vocab = int(getattr(args, "train_vocab", 256) or 256)
    optimizer = str(getattr(args, "train_optimizer", "adamw") or "adamw")

    parity_on = bool(getattr(args, "parity_on", False))
    parity_profile = str(getattr(args, "parity_profile", "balanced") or "balanced")
    parity_every = int(getattr(args, "parity_every", 50) or 0)
    train_loss_tol = float(getattr(args, "train_loss_tol", 2e-5) or 2e-5)
    activation_tol = max(train_loss_tol * 10.0, 1e-5)
    parity_replay_on_check = bool(getattr(args, "parity_replay_on_check", False))
    parity_replay_tol = float(getattr(args, "parity_replay_tol", 1e-7) or 1e-7)
    replay_weight_tol = float(getattr(args, "train_param_tol", 3e-5) or 3e-5)
    train_save_every = int(getattr(args, "train_save_every", 0) or 0)
    train_save_final = bool(getattr(args, "train_save_final", True))
    replay_auto_enabled = False
    has_weight_snapshot_api = bool(has_snapshot_numel and has_snapshot_export and has_snapshot_import)
    runtime_num_tokens = _infer_runtime_num_tokens(runtime_summary, d_model_hint=int(getattr(args, "train_d_model", 0) or 0), vocab_hint=vocab)

    batches = _build_train_token_batches(train_text, total_tokens, seq_len, vocab, seed)
    total_steps = epochs * len(batches)

    if bool(getattr(args, "train_verify_memory", False)):
        verify_report = _run_pr37_memory_verification(
            args=args,
            run_dir=run_dir,
            c_src=c_src,
            runtime_summary=runtime_summary,
            init_payload=init_payload,
            batches=batches,
            lr=lr,
        )
        verify_path = run_dir / "memory_verification_latest.json"
        verify_path.write_text(json.dumps(verify_report, indent=2), encoding="utf-8")
        profile_meta.setdefault("artifacts", []).append({"label": "memory_verification", "path": str(verify_path)})
        if not bool(verify_report.get("ok")):
            raise RuntimeError(f"PR3.7 memory verification failed: {verify_path}")

    # Oracle checks run at sampled steps to keep parity cost bounded for long runs.
    check_steps = _compute_parity_check_steps(total_steps, parity_profile, parity_every) if parity_on else set()

    oracle_payload = None
    oracle_loss_by_step: dict[int, float] = {}
    oracle_max_steps_used = 0
    oracle_source = "none"
    snapshot_oracle_error = None
    snapshot_oracle_enabled = False
    snapshot_oracle_fn = None
    oracle_strict = False

    if parity_on:
        grad_accum_for_oracle = max(1, int(getattr(args, "train_grad_accum", 1) or 1))
        oracle_max_steps = max(check_steps) if check_steps else 0
        if oracle_max_steps > 0 and grad_accum_for_oracle > 1:
            oracle_max_steps = int(math.ceil(float(oracle_max_steps) / float(grad_accum_for_oracle)))
        oracle_max_steps_used = int(oracle_max_steps)

        if has_weight_snapshot_api:
            try:
                import oracle_snapshot_torch_v7 as _snapshot_oracle_mod
                from oracle_snapshot_torch_v7 import compute_loss_logits_and_slots_from_snapshot_array

                if getattr(_snapshot_oracle_mod, "torch", None) is None:
                    raise RuntimeError("torch is required for snapshot oracle")
                snapshot_oracle_fn = compute_loss_logits_and_slots_from_snapshot_array
                snapshot_oracle_enabled = True
                oracle_source = "torch_snapshot_step"
                oracle_strict = True
            except Exception as e:
                snapshot_oracle_error = str(e)
                snapshot_oracle_enabled = False

        # Fallback reference only when snapshot-step oracle is unavailable.
        if not snapshot_oracle_enabled:
            oracle_payload = _run_ck_oracle_reference(
                args,
                run_dir,
                train_text,
                max_steps=oracle_max_steps if oracle_max_steps > 0 else None,
            )
            if isinstance(oracle_payload, dict):
                oracle_source = "tiny_reference_harness"
                for row in oracle_payload.get("loss_curve", []) if isinstance(oracle_payload.get("loss_curve"), list) else []:
                    try:
                        st = int(row.get("step", 0) or 0)
                        if st < 1:
                            continue
                        loss_pt = float(row.get("loss_pt", row.get("loss_ck", 0.0)) or 0.0)
                        # Optimizer-step index (native to train_parity_epochs_v7.py).
                        oracle_loss_by_step[st] = loss_pt
                        # Micro-step index used by CK runtime path when grad_accum > 1.
                        oracle_loss_by_step[st * grad_accum_for_oracle] = loss_pt
                    except Exception:
                        continue

    # When strict snapshot-oracle is active, force replay checks so parity
    # covers full CK step determinism (backward + optimizer + weight update),
    # not just forward activation/logit comparisons.
    if parity_on and snapshot_oracle_enabled and has_weight_snapshot_api and not parity_replay_on_check:
        parity_replay_on_check = True
        replay_auto_enabled = True

    replay_lib = None
    replay_runtime_error = None
    replay_has_forward_api = False
    replay_has_set_batch_api = False
    replay_has_act_snapshot_api = False
    if (parity_replay_on_check or snapshot_oracle_enabled) and has_weight_snapshot_api:
        try:
            replay_so = run_dir / "libtrain_replay.so"
            shutil.copy2(libtrain_so, replay_so)
            replay_lib = ctypes.CDLL(str(replay_so))
            if not hasattr(replay_lib, "ck_train_step") or not hasattr(replay_lib, "ck_train_init"):
                raise RuntimeError("missing ck_train_step/ck_train_init in replay runtime")
            if not hasattr(replay_lib, "ck_train_import_weight_snapshot"):
                raise RuntimeError("missing ck_train_import_weight_snapshot in replay runtime")

            replay_lib.ck_train_init.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            replay_lib.ck_train_init.restype = ctypes.c_int
            replay_lib.ck_train_step.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_float]
            replay_lib.ck_train_step.restype = ctypes.c_int
            replay_lib.ck_train_import_weight_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            replay_lib.ck_train_import_weight_snapshot.restype = ctypes.c_int

            replay_has_forward_api = bool(hasattr(replay_lib, "ck_train_forward_step"))
            replay_has_set_batch_api = bool(hasattr(replay_lib, "ck_train_set_batch"))
            replay_has_act_snapshot_api = bool(
                hasattr(replay_lib, "ck_train_get_activation_snapshot_numel")
                and hasattr(replay_lib, "ck_train_export_activation_snapshot")
            )
            if replay_has_forward_api:
                replay_lib.ck_train_forward_step.argtypes = []
                replay_lib.ck_train_forward_step.restype = ctypes.c_int
            if replay_has_set_batch_api:
                replay_lib.ck_train_set_batch.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
                replay_lib.ck_train_set_batch.restype = ctypes.c_int
            if replay_has_act_snapshot_api:
                replay_lib.ck_train_get_activation_snapshot_numel.argtypes = []
                replay_lib.ck_train_get_activation_snapshot_numel.restype = ctypes.c_int
                replay_lib.ck_train_export_activation_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
                replay_lib.ck_train_export_activation_snapshot.restype = ctypes.c_int

            replay_init_rc = int(replay_lib.ck_train_init(float_ptr, size_ptr, ctypes.c_int(init_payload["num_params"])))
            if replay_init_rc < 0:
                raise RuntimeError(f"replay ck_train_init failed with code {replay_init_rc}")
        except Exception as e:
            replay_runtime_error = str(e)
            replay_lib = None
            replay_has_forward_api = False
            replay_has_set_batch_api = False
            replay_has_act_snapshot_api = False

    step = 0
    micro_steps = 0
    processed_tokens = 0
    total_ck_ms = 0.0
    loss_curve: list[dict] = []
    parity_steps: list[dict] = []
    grad_steps: list[int] = []
    grad_global: list[float] = []
    parity_failures: list[dict] = []
    replay_failures: list[dict] = []
    checked_diffs: list[float] = []
    snapshot_artifacts: list[str] = []
    activation_snapshot_artifacts: list[str] = []
    checkpoint_artifacts: list[dict] = []
    last_checkpoint_step = 0
    oracle_points = 0
    last_oracle_loss = None

    for _ in range(epochs):
        for x_vals, y_vals in batches:
            step += 1
            micro_steps += 1
            x_buf = (ctypes.c_int32 * len(x_vals))(*x_vals)
            y_buf = (ctypes.c_int32 * len(y_vals))(*y_vals)
            loss_out = ctypes.c_float(0.0)

            need_check_snapshot = bool(
                parity_on
                and (step in check_steps)
                and (
                    snapshot_oracle_enabled
                    or parity_replay_on_check
                    or bool(getattr(args, "dump_on_drift", False))
                )
            )
            pre_snapshot = None
            pre_snapshot_numel = 0
            if need_check_snapshot and has_weight_snapshot_api:
                snap = _ck_export_runtime_weight_snapshot(lib)
                if snap is not None:
                    pre_snapshot, pre_snapshot_numel = snap

            t0 = time.perf_counter()
            calls = int(lib.ck_train_step(x_buf, y_buf, ctypes.byref(loss_out), ctypes.c_float(lr)))
            t1 = time.perf_counter()
            if calls < 0:
                raise RuntimeError(f"ck_train_step failed at step {step} (calls={calls})")

            step_ms = (t1 - t0) * 1000.0
            total_ck_ms += step_ms
            processed_tokens += min(len(x_vals), int(runtime_num_tokens))
            loss_val = float(loss_out.value)

            post_snapshot = None
            post_snapshot_numel = 0
            if parity_replay_on_check and (step in check_steps) and has_weight_snapshot_api:
                post_snap = _ck_export_runtime_weight_snapshot(lib)
                if post_snap is not None:
                    post_snapshot, post_snapshot_numel = post_snap

            if has_weight_snapshot_api and train_save_every > 0 and (step % train_save_every) == 0:
                ckpt_snap = (post_snapshot, post_snapshot_numel) if (post_snapshot is not None and post_snapshot_numel > 0) else _ck_export_runtime_weight_snapshot(lib)
                if ckpt_snap is not None:
                    ckpt_buf, ckpt_numel = ckpt_snap
                    ckpt_meta = _write_ck_weight_checkpoint_bump(
                        run_dir,
                        runtime_summary,
                        ckpt_buf,
                        int(ckpt_numel),
                        step=step,
                        reason="periodic",
                    )
                    if isinstance(ckpt_meta, dict):
                        checkpoint_artifacts.append(ckpt_meta)
                        last_checkpoint_step = int(step)

            replay_loss = None
            replay_diff = None
            replay_ok = None
            replay_weight_max_abs_diff = None
            replay_weight_mean_abs_diff = None
            replay_weight_error = None
            if parity_replay_on_check and (step in check_steps):
                if replay_lib is None:
                    replay_failures.append({
                        "step": step,
                        "loss_ck": loss_val,
                        "loss_replay": None,
                        "replay_diff": None,
                        "threshold": parity_replay_tol,
                        "weight_max_abs_diff": None,
                        "weight_mean_abs_diff": None,
                        "weight_threshold": replay_weight_tol,
                        "weight_error": "replay_runtime_unavailable",
                        "reason": f"replay_runtime_unavailable:{replay_runtime_error}",
                    })
                elif pre_snapshot is not None and pre_snapshot_numel > 0:
                    import_rc = _ck_import_runtime_weight_snapshot(replay_lib, pre_snapshot, pre_snapshot_numel)
                    if import_rc >= 0:
                        replay_loss_out = ctypes.c_float(0.0)
                        replay_calls = int(replay_lib.ck_train_step(x_buf, y_buf, ctypes.byref(replay_loss_out), ctypes.c_float(lr)))
                        if replay_calls < 0:
                            raise RuntimeError(f"ck_train_step replay failed at step {step} (calls={replay_calls})")
                        replay_loss = float(replay_loss_out.value)
                        replay_diff = abs(replay_loss - loss_val)

                        if post_snapshot is not None and post_snapshot_numel > 0:
                            replay_post = _ck_export_runtime_weight_snapshot(replay_lib)
                            if replay_post is not None:
                                replay_post_buf, replay_post_numel = replay_post
                                if int(replay_post_numel) == int(post_snapshot_numel):
                                    try:
                                        import numpy as _np
                                        ck_post_np = _np.ctypeslib.as_array(post_snapshot, shape=(int(post_snapshot_numel),)).astype(_np.float32, copy=False)
                                        replay_post_np = _np.ctypeslib.as_array(replay_post_buf, shape=(int(replay_post_numel),)).astype(_np.float32, copy=False)
                                        delta_np = _np.abs(ck_post_np - replay_post_np)
                                        replay_weight_max_abs_diff = float(_np.max(delta_np)) if delta_np.size else 0.0
                                        replay_weight_mean_abs_diff = float(_np.mean(delta_np)) if delta_np.size else 0.0
                                    except Exception as e:
                                        replay_weight_error = f"replay_weight_compare_failed:{e}"
                                else:
                                    replay_weight_error = f"replay_post_snapshot_size_mismatch:{replay_post_numel}!={post_snapshot_numel}"
                            else:
                                replay_weight_error = "replay_post_snapshot_unavailable"
                        else:
                            replay_weight_error = "post_snapshot_unavailable"

                        replay_ok = bool(
                            (replay_diff is not None)
                            and (replay_diff <= parity_replay_tol)
                            and (replay_weight_error is None)
                            and (replay_weight_max_abs_diff is not None)
                            and (float(replay_weight_max_abs_diff) <= float(replay_weight_tol))
                        )
                        if not replay_ok:
                            replay_failures.append({
                                "step": step,
                                "loss_ck": loss_val,
                                "loss_replay": replay_loss,
                                "replay_diff": replay_diff,
                                "threshold": parity_replay_tol,
                                "weight_max_abs_diff": replay_weight_max_abs_diff,
                                "weight_mean_abs_diff": replay_weight_mean_abs_diff,
                                "weight_threshold": replay_weight_tol,
                                "weight_error": replay_weight_error,
                            })
                    else:
                        replay_failures.append({
                            "step": step,
                            "loss_ck": loss_val,
                            "loss_replay": None,
                            "replay_diff": None,
                            "threshold": parity_replay_tol,
                            "weight_max_abs_diff": None,
                            "weight_mean_abs_diff": None,
                            "weight_threshold": replay_weight_tol,
                            "weight_error": "snapshot_import_failed",
                            "reason": f"snapshot_import_failed:{import_rc}",
                        })
                else:
                    replay_failures.append({
                        "step": step,
                        "loss_ck": loss_val,
                        "loss_replay": None,
                        "replay_diff": None,
                        "threshold": parity_replay_tol,
                        "weight_max_abs_diff": None,
                        "weight_mean_abs_diff": None,
                        "weight_threshold": replay_weight_tol,
                        "weight_error": "snapshot_unavailable",
                        "reason": "snapshot_unavailable",
                    })

            oracle_loss = None
            oracle_error = None
            oracle_logits_max_abs_diff = None
            oracle_logits_slot = None
            oracle_first_bad_tensor = None
            oracle_first_bad_diff = None
            oracle_first_bad_op = None
            oracle_slots_compared = 0
            oracle_slots_matched = 0
            if parity_on and (step in check_steps) and snapshot_oracle_enabled and snapshot_oracle_fn is not None:
                if pre_snapshot is not None and pre_snapshot_numel > 0:
                    try:
                        import numpy as _np
                        snap_np = _np.ctypeslib.as_array(pre_snapshot, shape=(int(pre_snapshot_numel),)).astype(_np.float32, copy=True)
                        oracle_x_vals = list(x_vals[: int(runtime_num_tokens)]) if int(runtime_num_tokens) > 0 else list(x_vals)
                        oracle_y_vals = list(y_vals[: int(runtime_num_tokens)]) if int(runtime_num_tokens) > 0 else list(y_vals)
                        oracle_loss_val, _oracle_logits, oracle_slot_map = snapshot_oracle_fn(
                            run_dir,
                            runtime_summary,
                            snap_np,
                            oracle_x_vals,
                            oracle_y_vals,
                        )
                        oracle_loss = float(oracle_loss_val)

                        ck_slot_rows = None
                        if replay_lib is not None and replay_has_forward_api and replay_has_set_batch_api and replay_has_act_snapshot_api:
                            import_rc = _ck_import_runtime_weight_snapshot(replay_lib, pre_snapshot, pre_snapshot_numel)
                            if import_rc < 0:
                                oracle_error = f"snapshot_import_failed:{import_rc}"
                            else:
                                set_rc = int(replay_lib.ck_train_set_batch(x_buf, y_buf))
                                if set_rc < 0:
                                    oracle_error = f"replay_set_batch_failed:{set_rc}"
                                else:
                                    fwd_rc = int(replay_lib.ck_train_forward_step())
                                    if fwd_rc < 0:
                                        oracle_error = f"replay_forward_failed:{fwd_rc}"
                                    else:
                                        act_numel = int(replay_lib.ck_train_get_activation_snapshot_numel())
                                        if act_numel > 0:
                                            act_buf = (ctypes.c_float * act_numel)()
                                            wrote = int(replay_lib.ck_train_export_activation_snapshot(act_buf, ctypes.c_int(act_numel)))
                                            if wrote > 0:
                                                ck_slot_rows = _extract_activation_slots_ordered(runtime_summary, act_buf, int(wrote))
                                            else:
                                                oracle_error = f"replay_activation_export_failed:{wrote}"
                                        else:
                                            oracle_error = "replay_activation_numel_invalid"
                        elif has_act_snapshot_export:
                            act_snap = _ck_export_runtime_activation_snapshot(lib)
                            if act_snap is not None:
                                act_buf, act_numel = act_snap
                                ck_slot_rows = _extract_activation_slots_ordered(runtime_summary, act_buf, act_numel)
                            else:
                                oracle_error = "activation_snapshot_unavailable"

                        if isinstance(ck_slot_rows, list):
                            for row in ck_slot_rows:
                                name = str(row.get("name", ""))
                                ck_flat = _np.asarray(row.get("flat"), dtype=_np.float32).reshape(-1)
                                oracle_flat = oracle_slot_map.get(name)
                                if oracle_flat is None:
                                    continue
                                oracle_slots_compared += 1
                                ref = _np.asarray(oracle_flat, dtype=_np.float32).reshape(-1)
                                if ck_flat.size != ref.size:
                                    diff = float("inf")
                                elif ck_flat.size == 0:
                                    diff = 0.0
                                else:
                                    diff = float(_np.max(_np.abs(ck_flat - ref)))
                                    oracle_slots_matched += 1

                                if name == "act.Sfooter.logits.0.y" or ".logits." in name:
                                    oracle_logits_max_abs_diff = diff
                                    oracle_logits_slot = name

                                if (oracle_first_bad_tensor is None) and (diff > float(activation_tol)):
                                    oracle_first_bad_tensor = name
                                    oracle_first_bad_diff = diff
                                    oracle_first_bad_op = _slot_op_hint_from_name(name)

                            if oracle_logits_max_abs_diff is None:
                                maybe_logits = oracle_slot_map.get("act.Sfooter.logits.0.y")
                                if maybe_logits is not None:
                                    oracle_logits_slot = "act.Sfooter.logits.0.y"
                                    oracle_logits_max_abs_diff = None
                        elif oracle_error is None:
                            oracle_error = "activation_slot_decode_failed"
                    except Exception as e:
                        oracle_error = str(e)
                else:
                    oracle_error = "snapshot_unavailable"

            if oracle_loss is None:
                oracle_loss = oracle_loss_by_step.get(step)

            if oracle_loss is not None:
                oracle_points += 1
                last_oracle_loss = float(oracle_loss)

            loss_diff = abs(loss_val - oracle_loss) if oracle_loss is not None else 0.0
            snapshot_path = None
            activation_snapshot_path = None
            if parity_on and (step in check_steps):
                if oracle_loss is not None:
                    checked_diffs.append(loss_diff)
                    fail_loss = bool(oracle_strict and (loss_diff > train_loss_tol))
                    fail_logits = bool(
                        oracle_strict
                        and oracle_logits_max_abs_diff is not None
                        and float(oracle_logits_max_abs_diff) > float(activation_tol)
                    )
                    fail_slots = bool(
                        oracle_strict
                        and oracle_first_bad_tensor is not None
                        and oracle_first_bad_diff is not None
                        and float(oracle_first_bad_diff) > float(activation_tol)
                    )
                    if fail_loss or fail_logits or fail_slots:
                        if bool(getattr(args, "dump_on_drift", False)) and pre_snapshot is not None and pre_snapshot_numel > 0:
                            snap_path = _write_ck_weight_snapshot_artifact(
                                run_dir,
                                step,
                                pre_snapshot,
                                pre_snapshot_numel,
                                reason="parity_drift",
                            )
                            if snap_path is not None:
                                snapshot_path = str(snap_path)
                                snapshot_artifacts.append(snapshot_path)
                        if bool(getattr(args, "dump_on_drift", False)) and has_act_snapshot_export:
                            act_snap = _ck_export_runtime_activation_snapshot(lib)
                            if act_snap is not None:
                                act_buf, act_numel = act_snap
                                act_path = _write_ck_activation_snapshot_artifact(
                                    run_dir,
                                    step,
                                    act_buf,
                                    act_numel,
                                    reason="parity_drift",
                                )
                                if act_path is not None:
                                    activation_snapshot_path = str(act_path)
                                    activation_snapshot_artifacts.append(activation_snapshot_path)
                        parity_failures.append({
                            "step": step,
                            "loss_ck": loss_val,
                            "loss_oracle": oracle_loss,
                            "loss_diff": loss_diff,
                            "threshold": train_loss_tol,
                            "logits_max_abs_diff": oracle_logits_max_abs_diff,
                            "logits_threshold": activation_tol,
                            "logits_slot": oracle_logits_slot,
                            "first_bad_tensor": oracle_first_bad_tensor,
                            "first_bad_diff": oracle_first_bad_diff,
                            "first_bad_op": oracle_first_bad_op,
                            "activation_threshold": activation_tol,
                            "slots_compared": oracle_slots_compared,
                            "slots_matched": oracle_slots_matched,
                            "snapshot": snapshot_path,
                            "activation_snapshot": activation_snapshot_path,
                            "oracle_error": oracle_error,
                        })
                else:
                    if oracle_strict:
                        parity_failures.append({
                            "step": step,
                            "loss_ck": loss_val,
                            "loss_oracle": None,
                            "loss_diff": None,
                            "threshold": train_loss_tol,
                            "snapshot": None,
                            "activation_snapshot": None,
                            "reason": oracle_error or "oracle_unavailable",
                        })

            # grad_norm is a placeholder until runtime exports per-step grad telemetry.
            loss_curve.append(
                {
                    "step": step,
                    "micro_steps": 1,
                    "tokens": len(x_vals),
                    "loss_ck": loss_val,
                    "loss_pt": oracle_loss if oracle_loss is not None else loss_val,
                    "logits_max_abs_diff": oracle_logits_max_abs_diff,
                    "logits_slot": oracle_logits_slot,
                    "first_bad_tensor": oracle_first_bad_tensor,
                    "first_bad_diff": oracle_first_bad_diff,
                    "first_bad_op": oracle_first_bad_op,
                    "slots_compared": oracle_slots_compared,
                    "slots_matched": oracle_slots_matched,
                    "lr": lr,
                    "grad_norm": 0.0,
                    "forward_ms": 0.0,
                    "backward_ms": 0.0,
                    "optimizer_ms": 0.0,
                    "step_ms": step_ms,
                    "torch_forward_ms": 0.0,
                    "torch_backward_ms": 0.0,
                    "torch_optimizer_ms": 0.0,
                    "torch_step_ms": 0.0,
                }
            )
            parity_steps.append(
                {
                    "step": step,
                    "loss_diff": loss_diff,
                    "max_param_diff": float(replay_weight_max_abs_diff or 0.0),
                    "worst_param": None,
                    "mean_param_diff": float(replay_weight_mean_abs_diff or 0.0),
                    "checked": bool(parity_on and (step in check_steps) and (oracle_loss is not None)),
                    "oracle_available": oracle_loss is not None,
                    "oracle_source": oracle_source,
                    "oracle_error": oracle_error,
                    "logits_max_abs_diff": oracle_logits_max_abs_diff,
                    "logits_threshold": activation_tol,
                    "logits_slot": oracle_logits_slot,
                    "first_bad_tensor": oracle_first_bad_tensor,
                    "first_bad_diff": oracle_first_bad_diff,
                    "first_bad_op": oracle_first_bad_op,
                    "slots_compared": oracle_slots_compared,
                    "slots_matched": oracle_slots_matched,
                    "replay_enabled": bool(parity_replay_on_check),
                    "replay_available": bool(pre_snapshot is not None and pre_snapshot_numel > 0),
                    "replay_loss": replay_loss,
                    "replay_diff": replay_diff,
                    "replay_ok": replay_ok,
                    "replay_weight_max_abs_diff": replay_weight_max_abs_diff,
                    "replay_weight_mean_abs_diff": replay_weight_mean_abs_diff,
                    "replay_weight_threshold": replay_weight_tol,
                    "replay_weight_error": replay_weight_error,
                }
            )
            grad_steps.append(step)
            grad_global.append(0.0)

    if has_weight_snapshot_api and train_save_final and step > 0 and int(last_checkpoint_step) != int(step):
        final_snap = _ck_export_runtime_weight_snapshot(lib)
        if final_snap is not None:
            final_buf, final_numel = final_snap
            final_ckpt = _write_ck_weight_checkpoint_bump(
                run_dir,
                runtime_summary,
                final_buf,
                int(final_numel),
                step=step,
                reason="final",
            )
            if isinstance(final_ckpt, dict):
                checkpoint_artifacts.append(final_ckpt)
                last_checkpoint_step = int(step)

    replay_weight_max_values = [
        float(row.get("replay_weight_max_abs_diff"))
        for row in parity_steps
        if isinstance(row, dict) and row.get("replay_weight_max_abs_diff") is not None
    ]
    replay_weight_mean_values = [
        float(row.get("replay_weight_mean_abs_diff"))
        for row in parity_steps
        if isinstance(row, dict) and row.get("replay_weight_mean_abs_diff") is not None
    ]

    final_ck_loss = float(loss_curve[-1]["loss_ck"]) if loss_curve else 0.0
    final_oracle_loss = float(last_oracle_loss) if last_oracle_loss is not None else final_ck_loss
    train_tok_s = (processed_tokens / (total_ck_ms / 1000.0)) if total_ck_ms > 0 else 0.0
    avg_ck_step_ms = (total_ck_ms / step) if step > 0 else 0.0

    if parity_on and oracle_points == 0:
        parity_failures.append({
            "step": 0,
            "loss_ck": final_ck_loss,
            "loss_oracle": None,
            "loss_diff": None,
            "threshold": train_loss_tol,
            "reason": snapshot_oracle_error or "oracle_unavailable",
        })

    max_loss_abs_diff = max(checked_diffs) if checked_diffs else 0.0
    mean_loss_abs_diff = (sum(checked_diffs) / len(checked_diffs)) if checked_diffs else 0.0
    pass_parity = (len(parity_failures) == 0) and (not parity_replay_on_check or len(replay_failures) == 0)

    if parity_on and parity_failures and bool(getattr(args, "dump_on_drift", False)):
        topk = int(getattr(args, "drift_topk", 20) or 20)
        ranked = sorted(
            parity_failures,
            key=lambda r: float(r.get("loss_diff") or 0.0),
            reverse=True,
        )
        drift_report = {
            "generated_at": _utc_now_iso(),
            "backend": train_backend,
            "oracle": str(getattr(args, "oracle", "pytorch") or "pytorch"),
            "oracle_source": oracle_source,
            "threshold": train_loss_tol,
            "logits_threshold": activation_tol,
            "first_failure": ranked[0] if ranked else None,
            "failures": ranked[:topk],
            "replay_failures": replay_failures[:topk],
            "snapshot_files": snapshot_artifacts[:topk],
            "activation_snapshot_files": activation_snapshot_artifacts[:topk],
        }
        drift_path = run_dir / "drift_report.json"
        drift_path.write_text(json.dumps(drift_report, indent=2), encoding="utf-8")
        profile_meta.setdefault("artifacts", []).append({"label": "drift_report", "path": str(drift_path)})

    summary = {
        "epochs": epochs,
        "seq_len": seq_len,
        "total_tokens": total_tokens,
        "grad_accum": grad_accum,
        "optimizer": optimizer,
        "lr": lr,
        "steps": step,
        "micro_steps": micro_steps,
        "tokens_per_update": seq_len * grad_accum,
        "max_loss_abs_diff": max_loss_abs_diff,
        "mean_loss_abs_diff": mean_loss_abs_diff,
        "final_ck_loss": final_ck_loss,
        "final_torch_loss": final_oracle_loss,
        "final_param_max_abs_diff": float(max(replay_weight_max_values) if replay_weight_max_values else 0.0),
        "final_param_mean_abs_diff": float((sum(replay_weight_mean_values) / len(replay_weight_mean_values)) if replay_weight_mean_values else 0.0),
        "pass_parity": pass_parity,
        "loss_curve": loss_curve,
        "parity_steps": parity_steps,
        "grad_norm_series": {
            "steps": grad_steps,
            "global": grad_global,
            "params": {},
        },
        "step_profile": {
            "steps": step,
            "micro_steps": micro_steps,
            "tokens_per_update": seq_len * grad_accum,
            "processed_tokens": processed_tokens,
            "ck_total_ms": total_ck_ms,
            "torch_total_ms": 0.0,
            "ck_avg_step_ms": avg_ck_step_ms,
            "torch_avg_step_ms": 0.0,
            "train_tok_s": train_tok_s,
            "decode_tok_s": train_tok_s,
        },
        "backend": train_backend,
        "train_mode": train_mode,
        "source": "ck_runtime_generated",
        "runtime_init": {
            "num_params": int(init_payload.get("num_params", 0)),
            "total_floats": int(init_payload.get("total_floats", 0)),
            "runtime_num_tokens": int(runtime_num_tokens),
            "loaded": init_payload.get("loaded", []),
            "memory_diagnostic": diag_info,
        },
        "checkpoints": {
            "enabled": bool(has_weight_snapshot_api and (train_save_every > 0 or train_save_final)),
            "save_every": int(train_save_every),
            "save_final": bool(train_save_final),
            "count": int(len(checkpoint_artifacts)),
            "latest_step": int(last_checkpoint_step),
            "files": checkpoint_artifacts,
        },
        "oracle": {
            "enabled": parity_on,
            "profile": parity_profile,
            "every": parity_every,
            "checks": sorted(check_steps),
            "max_steps": int(oracle_max_steps_used),
            "source": oracle_source,
            "strict": bool(oracle_strict),
            "available": bool(oracle_points > 0),
            "snapshot_torch_enabled": bool(snapshot_oracle_enabled),
            "snapshot_torch_error": snapshot_oracle_error,
            "failures": parity_failures,
            "replay_on_check": bool(parity_replay_on_check),
            "replay_auto_enabled": bool(replay_auto_enabled),
            "replay_tol": float(parity_replay_tol),
            "replay_weight_tol": float(replay_weight_tol),
            "logits_tol": float(activation_tol),
            "replay_failures": replay_failures,
            "snapshot_api_available": bool(has_weight_snapshot_api),
            "activation_snapshot_api_available": bool(has_act_snapshot_numel and has_act_snapshot_export),
            "replay_runtime_error": replay_runtime_error,
            "snapshot_files": snapshot_artifacts,
            "activation_snapshot_files": activation_snapshot_artifacts,
        },
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    profile_meta.setdefault("artifacts", []).append({"label": "ck_runtime_lib", "path": str(libtrain_so)})
    profile_meta.setdefault("artifacts", []).append({"label": "runtime_summary", "path": str(runtime_summary_path)})
    profile_meta.setdefault("artifacts", []).append({"label": "train_summary", "path": str(json_out)})
    layout_path = run_dir / "layout_train.json"
    layout_audit_path = run_dir / "layout_train_audit.json"
    if layout_path.exists():
        profile_meta.setdefault("artifacts", []).append({"label": "layout_train", "path": str(layout_path)})
    if layout_audit_path.exists():
        profile_meta.setdefault("artifacts", []).append({"label": "layout_train_audit", "path": str(layout_audit_path)})
    if parity_on and (run_dir / "oracle_reference_latest.json").exists():
        profile_meta.setdefault("artifacts", []).append({"label": "oracle_reference", "path": str(run_dir / "oracle_reference_latest.json")})
    if snapshot_artifacts:
        profile_meta.setdefault("artifacts", []).append({"label": "oracle_ck_snapshots", "path": str(run_dir / "oracle_ck_snapshots")})
    if activation_snapshot_artifacts:
        profile_meta.setdefault("artifacts", []).append({"label": "oracle_ck_activations", "path": str(run_dir / "oracle_ck_activations")})
    if checkpoint_artifacts:
        profile_meta.setdefault("artifacts", []).append({"label": "train_checkpoints", "path": str(run_dir / "checkpoints")})

    return json_out


def _export_train_telemetry_to_run_dir(summary_json: Path, run_dir: Path) -> None:
    """Write viewer-friendly training telemetry files into run_dir."""
    if not summary_json.exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    with summary_json.open("r", encoding="utf-8") as f:
        s = json.load(f)

    step = int(s.get("steps", 0) or 0)
    loss_ck = float(s.get("final_ck_loss", 0.0) or 0.0)
    loss_pt = float(s.get("final_torch_loss", 0.0) or 0.0)
    lr = float(s.get("lr", 0.0) or 0.0)
    max_loss = float(s.get("max_loss_abs_diff", 0.0) or 0.0)
    max_param = float(s.get("final_param_max_abs_diff", 0.0) or 0.0)

    raw_curve = s.get("loss_curve") if isinstance(s, dict) else None
    if isinstance(raw_curve, list) and raw_curve:
        training_loss_curve = {"steps": raw_curve, "source": "train_e2e_detailed"}
    else:
        training_loss_curve = {
            "steps": [{"step": step, "loss_ck": loss_ck, "loss_pt": loss_pt, "lr": lr, "grad_norm": 0.0}],
            "source": "train_e2e_summary",
        }

    raw_parity = s.get("parity_steps") if isinstance(s, dict) else None
    if isinstance(raw_parity, list) and raw_parity:
        training_parity = {"steps": raw_parity, "source": "train_e2e_detailed"}
    else:
        training_parity = {
            "steps": [{"step": step, "loss_diff": max_loss, "max_param_diff": max_param, "worst_param": "aggregate"}],
            "source": "train_e2e_summary",
        }

    grad_series = s.get("grad_norm_series") if isinstance(s.get("grad_norm_series"), dict) else {}
    training_grad_norms = {
        "steps": grad_series.get("steps", [row.get("step", step) for row in training_loss_curve.get("steps", [])]),
        "global": grad_series.get("global", [row.get("grad_norm", 0.0) for row in training_loss_curve.get("steps", [])]),
        "params": grad_series.get("params", {}),
        "source": "train_e2e_detailed" if grad_series else "train_e2e_summary",
    }

    step_profile = s.get("step_profile") if isinstance(s.get("step_profile"), dict) else {}
    training_step_profile = {
        "steps": int(step_profile.get("steps", step) or step),
        "micro_steps": int(step_profile.get("micro_steps", s.get("micro_steps", 0)) or 0),
        "tokens_per_update": int(step_profile.get("tokens_per_update", s.get("tokens_per_update", 0)) or 0),
        "processed_tokens": int(step_profile.get("processed_tokens", 0) or 0),
        "ck_total_ms": float(step_profile.get("ck_total_ms", 0.0) or 0.0),
        "torch_total_ms": float(step_profile.get("torch_total_ms", 0.0) or 0.0),
        "ck_avg_step_ms": float(step_profile.get("ck_avg_step_ms", 0.0) or 0.0),
        "torch_avg_step_ms": float(step_profile.get("torch_avg_step_ms", 0.0) or 0.0),
        "train_tok_s": float(step_profile.get("train_tok_s", 0.0) or 0.0),
        "decode_tok_s": float(step_profile.get("decode_tok_s", step_profile.get("train_tok_s", 0.0)) or 0.0),
    }

    ckpt_info = s.get("checkpoints") if isinstance(s.get("checkpoints"), dict) else {}
    training_checkpoint_policy = {
        "policy": "step_interval" if bool(ckpt_info.get("enabled")) else "none",
        "source": "train_e2e",
        "checkpointing": bool(ckpt_info.get("enabled", False)),
        "save_every": int(ckpt_info.get("save_every", 0) or 0),
        "save_final": bool(ckpt_info.get("save_final", False)),
        "count": int(ckpt_info.get("count", 0) or 0),
        "latest_step": int(ckpt_info.get("latest_step", 0) or 0),
        "files": ckpt_info.get("files", []),
    }

    payloads = {
        "training_loss_curve.json": training_loss_curve,
        "training_parity.json": training_parity,
        "training_grad_norms.json": training_grad_norms,
        "training_step_profile.json": training_step_profile,
        "training_checkpoint_policy.json": training_checkpoint_policy,
    }
    for name, payload in payloads.items():
        (run_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    dst = run_dir / "train_e2e_latest.json"
    try:
        same = summary_json.resolve() == dst.resolve()
    except Exception:
        same = str(summary_json) == str(dst)
    if not same:
        shutil.copy2(summary_json, dst)


def _run_train_strict_preflight(python_exec: str) -> None:
    """Run strict static training contract checks before executing train commands."""
    validate_script = SCRIPTS_DIR / "validate_v7_contracts.py"
    if not validate_script.exists():
        log_error(f"Strict preflight requested but script not found: {validate_script}")
        sys.exit(1)
    out = DEFAULT_REPORT_DIR / "contract_report_latest.json"
    cmd = [
        python_exec,
        str(validate_script),
        "--strict",
        "--training-mode",
        "--json-out",
        str(out),
    ]
    log("  train-strict preflight: validate_v7_contracts.py", C_DIM)
    run_cmd(cmd, cwd=PROJECT_ROOT)


def step_run_train_e2e(args: argparse.Namespace) -> Path:
    """Run v7 train E2E using selected backend (ck runtime or parity harness)."""
    log_step(1, "Running train E2E")

    train_script = SCRIPTS_DIR / "train_parity_epochs_v7.py"
    if not train_script.exists():
        log_error(f"Training parity script not found: {train_script}")
        sys.exit(1)

    train_text = _resolve_train_text(args)
    train_mode = _resolve_train_mode(args)
    train_backend = _resolve_train_backend(args)

    run_dir_arg = getattr(args, "run_dir", None)
    run_dir = Path(run_dir_arg) if run_dir_arg else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    json_out = getattr(args, "train_json_out", None)
    if not json_out:
        if run_dir is not None:
            json_out = run_dir / "train_e2e_latest.json"
        else:
            json_out = DEFAULT_REPORT_DIR / "train_e2e_latest.json"
    else:
        json_out = Path(json_out)

    parity_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(parity_python) if parity_python.exists() else sys.executable

    if bool(getattr(args, "train_strict", False)):
        _run_train_strict_preflight(python_exec)

    train_vocab = int(getattr(args, "train_vocab", 256) or 256)
    train_d_model = int(getattr(args, "train_d_model", 64) or 64)
    train_hidden = int(getattr(args, "train_hidden", 128) or 128)
    train_loss_tol = float(getattr(args, "train_loss_tol", 2e-5) or 2e-5)
    train_param_tol = float(getattr(args, "train_param_tol", 3e-5) or 3e-5)

    cmd = [
        python_exec,
        str(train_script),
        "--epochs", str(getattr(args, "train_epochs", 3)),
        "--seq-len", str(getattr(args, "train_seq_len", 16)),
        "--total-tokens", str(getattr(args, "train_total_tokens", 1024)),
        "--grad-accum", str(getattr(args, "train_grad_accum", 8)),
        "--optimizer", str(getattr(args, "train_optimizer", "adamw")),
        "--lr", str(getattr(args, "train_lr", 1e-3)),
        "--seed", str(getattr(args, "train_seed", 42)),
        "--vocab", str(train_vocab),
        "--d-model", str(train_d_model),
        "--hidden", str(train_hidden),
        "--loss-tol", str(train_loss_tol),
        "--param-tol", str(train_param_tol),
        "--json-out", str(json_out),
    ]

    if run_dir is not None and bool(getattr(args, "train_use_init_bump", True)):
        bump_path = run_dir / "weights.bump"
        manifest_path = run_dir / "weights_manifest.json"
        if bump_path.exists() and manifest_path.exists():
            cmd.extend(["--weights-bump", str(bump_path), "--weights-manifest", str(manifest_path)])
            log(f"  init weights: {bump_path.name} + {manifest_path.name}", C_DIM)

    if train_text:
        cmd.extend(["--train-text", train_text])

    log(
        f"  mode={train_mode} backend={train_backend} epochs={getattr(args, 'train_epochs', 3)} "
        f"seq_len={getattr(args, 'train_seq_len', 16)} total_tokens={getattr(args, 'train_total_tokens', 1024)}",
        C_DIM,
    )
    log(
        f"  d_model={train_d_model} hidden={train_hidden} vocab={train_vocab} "
        f"grad_accum={getattr(args, 'train_grad_accum', 8)} optimizer={getattr(args, 'train_optimizer', 'adamw')}",
        C_DIM,
    )
    if train_text:
        short = train_text if len(train_text) <= 64 else train_text[:61] + "..."
        log(f"  train_text={short}", C_DIM)

    parity_on = bool(getattr(args, "parity_on", False))
    parity_profile = str(getattr(args, "parity_profile", "balanced") or "balanced")
    parity_every = int(getattr(args, "parity_every", 50) or 0)
    oracle = str(getattr(args, "oracle", "pytorch") or "pytorch")
    analysis_mode = str(getattr(args, "analysis_checkpoints", "log") or "log")
    train_save_every = int(getattr(args, "train_save_every", 0) or 0)
    train_save_final = bool(getattr(args, "train_save_final", True))
    if parity_on:
        cadence = f"every={parity_every}" if parity_every > 0 else f"profile={parity_profile}"
        log(f"  parity oracle: on ({oracle}, {cadence})", C_DIM)
    else:
        log("  parity oracle: off", C_DIM)
    if bool(getattr(args, "dump_on_drift", False)):
        log(f"  drift dumps: on (topk={int(getattr(args, 'drift_topk', 20) or 20)})", C_DIM)
    if train_save_every > 0 or train_save_final:
        cadence = f"every={train_save_every}" if train_save_every > 0 else "off"
        final_txt = "on" if train_save_final else "off"
        log(f"  runtime checkpoints: cadence={cadence}, final={final_txt}", C_DIM)
    log(f"  analysis checkpoints: {analysis_mode}", C_DIM)

    profile_mode = str(getattr(args, "profile_train", "none") or "none").lower()
    profile_dir_arg = getattr(args, "train_profile_dir", None)
    if profile_dir_arg:
        profile_dir = Path(profile_dir_arg)
    elif run_dir is not None:
        profile_dir = run_dir / "profile_train_latest"
    else:
        profile_dir = DEFAULT_REPORT_DIR / "profile_train_latest"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_meta = {
        "mode": profile_mode,
        "train_mode": train_mode,
        "backend": train_backend,
        "parity_on": parity_on,
        "parity_profile": parity_profile,
        "parity_every": parity_every,
        "oracle": oracle,
        "dump_on_drift": bool(getattr(args, "dump_on_drift", False)),
        "drift_topk": int(getattr(args, "drift_topk", 20) or 20),
        "analysis_checkpoints": analysis_mode,
        "train_save_every": train_save_every,
        "train_save_final": train_save_final,
        "artifacts": [],
    }

    def _note_artifact(label: str, path: Path) -> None:
        profile_meta["artifacts"].append({"label": label, "path": str(path)})

    if train_backend == "ck":
        if run_dir is None:
            log_error("--backend ck requires --run <run_dir> so runtime artifacts can be generated")
            sys.exit(2)
        if profile_mode in ("perf", "vtune"):
            log(f"  Warning: --profile-train={profile_mode} for backend=ck is not wired yet; running direct CK runtime", C_ORANGE)
        _run_ck_train_runtime(
            args=args,
            run_dir=run_dir,
            json_out=Path(json_out),
            train_text=train_text,
            train_mode=train_mode,
            train_backend=train_backend,
            profile_meta=profile_meta,
        )
    else:
        if profile_mode == "perf":
            perf_bin = shutil.which("perf")
            if perf_bin:
                perf_stat = profile_dir / "perf_train.stat.txt"
                log(f"  Profiling training with perf stat -> {perf_stat}", C_DIM)
                perf_cmd = [perf_bin, "stat", "-d", "-d", "-d", "-o", str(perf_stat), "--"] + cmd
                prof = run_cmd_allow_fail(perf_cmd, cwd=PROJECT_ROOT)
                if prof.returncode != 0:
                    log(f"  Warning: perf profiling failed ({prof.returncode}); running train-e2e without external profiler", C_ORANGE)
                    run_cmd(cmd, cwd=PROJECT_ROOT)
                else:
                    _note_artifact("perf_stat", perf_stat)
                    perf_artifacts_script = SCRIPTS_DIR / "perf_artifacts_v7.py"
                    run_cmd_allow_fail([
                        python_exec,
                        str(perf_artifacts_script),
                        "--out-dir", str(DEFAULT_REPORT_DIR),
                        "--perf-stat", str(perf_stat),
                    ], cwd=PROJECT_ROOT)
            else:
                log("  Warning: perf not found; running without external profiler", C_ORANGE)
                run_cmd(cmd, cwd=PROJECT_ROOT)
        elif profile_mode == "vtune":
            vtune_bin = shutil.which("vtune")
            if vtune_bin:
                result_dir = profile_dir / "vtune_hotspots"
                text_out = profile_dir / "vtune_hotspots.txt"
                csv_out = profile_dir / "vtune_hotspots.csv"
                log(f"  Profiling training with VTune hotspots -> {result_dir}", C_DIM)
                vtune_cmd = [vtune_bin, "-collect", "hotspots", "-result-dir", str(result_dir), "--"] + cmd
                prof = run_cmd_allow_fail(vtune_cmd, cwd=PROJECT_ROOT)
                if prof.returncode != 0:
                    log(f"  Warning: VTune profiling failed ({prof.returncode}); running train-e2e without external profiler", C_ORANGE)
                    run_cmd(cmd, cwd=PROJECT_ROOT)
                else:
                    run_cmd_allow_fail([vtune_bin, "-report", "hotspots", "-r", str(result_dir), "-format", "text", "-report-output", str(text_out)], cwd=PROJECT_ROOT)
                    run_cmd_allow_fail([vtune_bin, "-report", "hotspots", "-r", str(result_dir), "-format", "csv", "-report-output", str(csv_out)], cwd=PROJECT_ROOT)
                    _note_artifact("vtune_result_dir", result_dir)
                    _note_artifact("vtune_hotspots_text", text_out)
                    _note_artifact("vtune_hotspots_csv", csv_out)
                    vtune_artifacts_script = SCRIPTS_DIR / "vtune_artifacts_v7.py"
                    run_cmd_allow_fail([
                        python_exec,
                        str(vtune_artifacts_script),
                        "--out-dir", str(DEFAULT_REPORT_DIR),
                        "--result-dir", str(result_dir),
                        "--report-text", str(text_out),
                        "--report-csv", str(csv_out),
                    ], cwd=PROJECT_ROOT)
            else:
                log("  Warning: vtune not found; running without external profiler", C_ORANGE)
                run_cmd(cmd, cwd=PROJECT_ROOT)
        else:
            if profile_mode not in ("none", ""):
                log(f"  Warning: unknown --profile-train mode '{profile_mode}', using none", C_ORANGE)
            run_cmd(cmd, cwd=PROJECT_ROOT)

    try:
        _materialize_train_telemetry(Path(json_out), profile_meta=profile_meta)
    except Exception as e:
        log(f"  Warning: telemetry materialization failed: {e}", C_ORANGE)
    if run_dir is not None:
        try:
            _export_train_telemetry_to_run_dir(Path(json_out), run_dir)
        except Exception as e:
            log(f"  Warning: run_dir telemetry export failed: {e}", C_ORANGE)
    log(f"  Train parity report: {json_out}", C_GREEN)
    return Path(json_out)


def step_run_train_init(args: argparse.Namespace) -> None:
    """Initialize a tiny v7 training run directory (weights.bump + manifest)."""
    log_step(1, "Initializing tiny v7 training run")

    init_script = SCRIPTS_DIR / "init_tiny_train_model_v7.py"
    if not init_script.exists():
        log_error(f"Init script not found: {init_script}")
        sys.exit(1)

    parity_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(parity_python) if parity_python.exists() else sys.executable

    run_name = str(getattr(args, "run_name", "tiny_init") or "tiny_init").strip()
    out_dir_arg = getattr(args, "output_dir", None)
    out_dir = Path(out_dir_arg) if out_dir_arg else (V7_ROOT / "runs" / run_name)

    cmd = [
        python_exec,
        str(init_script),
        "--output-dir", str(out_dir),
        "--seed", str(getattr(args, "train_seed", 42)),
        "--init", str(getattr(args, "init", "normal_0p02")),
        "--layers", str(getattr(args, "layers", 2)),
        "--vocab-size", str(getattr(args, "vocab_size", 256)),
        "--embed-dim", str(getattr(args, "embed_dim", 128)),
        "--hidden-dim", str(getattr(args, "hidden_dim", 256)),
        "--num-heads", str(getattr(args, "num_heads", 8)),
        "--num-kv-heads", str(getattr(args, "num_kv_heads", 4)),
        "--context-len", str(getattr(args, "context_len", 128)),
        "--rope-theta", str(getattr(args, "rope_theta", 1_000_000.0)),
        "--kernel-policy", str(getattr(args, "kernel_policy", "fp32_reference_first")),
        "--template", str(getattr(args, "template", "qwen3")),
    ]
    template_file = getattr(args, "template_file", None)
    if template_file:
        cmd.extend(["--template-file", str(template_file)])
    run_cmd(cmd, cwd=PROJECT_ROOT)

    # Run-dir aliases for operator workflows.
    try:
        bump = out_dir / "weights.bump"
        bump_alias = out_dir / "weights_init.bump"
        if bump.exists() and not bump_alias.exists():
            shutil.copy2(bump, bump_alias)

        cfg_src = out_dir / "train_init_config.json"
        cfg_alias = out_dir / "config.json"
        if cfg_src.exists() and not cfg_alias.exists():
            shutil.copy2(cfg_src, cfg_alias)
    except Exception as e:
        log(f"  Warning: failed to create run-dir aliases: {e}", C_ORANGE)

    if getattr(args, "generate_ir", False):
        manifest_path = out_dir / "weights_manifest.json"

        ir1_script = SCRIPTS_DIR / "build_ir_train_v7.py"
        ir1_out = out_dir / "ir1_train_forward.json"
        ir1_report = out_dir / "ir1_train_report.json"
        ir1_cmd = [
            python_exec,
            str(ir1_script),
            "--manifest", str(manifest_path),
            "--output", str(ir1_out),
            "--report-out", str(ir1_report),
        ]
        if getattr(args, "strict", False):
            ir1_cmd.append("--strict")
        run_cmd(ir1_cmd, cwd=PROJECT_ROOT)

        ir2_script = SCRIPTS_DIR / "lower_ir2_backward_v7.py"
        ir2_out = out_dir / "ir2_train_backward.json"
        ir2_summary = out_dir / "ir2_train_summary.json"
        ir2_cmd = [
            python_exec,
            str(ir2_script),
            "--ir1", str(ir1_out),
            "--output", str(ir2_out),
            "--summary-out", str(ir2_summary),
        ]
        if getattr(args, "strict", False):
            ir2_cmd.append("--strict")
        else:
            ir2_cmd.append("--allow-partial")
        run_cmd(ir2_cmd, cwd=PROJECT_ROOT)

        inv_script = SCRIPTS_DIR / "validate_ir_train_invariants_v7.py"
        inv_out = out_dir / "ir_train_invariants.json"
        inv_cmd = [
            python_exec,
            str(inv_script),
            "--ir1", str(ir1_out),
            "--ir2", str(ir2_out),
            "--output", str(inv_out),
        ]
        if getattr(args, "strict", False):
            inv_cmd.append("--strict-unresolved")
        else:
            inv_cmd.append("--allow-partial")
        run_cmd(inv_cmd, cwd=PROJECT_ROOT)

        layout_script = SCRIPTS_DIR / "generate_train_layout_v7.py"
        layout_out = out_dir / "layout_train.json"
        layout_cmd = [
            python_exec,
            str(layout_script),
            "--ir2", str(ir2_out),
            "--manifest", str(manifest_path),
            "--output", str(layout_out),
            "--align-bytes", "64",
        ]
        if getattr(args, "strict", False):
            layout_cmd.append("--strict")
        run_cmd(layout_cmd, cwd=PROJECT_ROOT)

        layout_audit_script = SCRIPTS_DIR / "validate_train_memory_layout_v7.py"
        layout_audit_out = out_dir / "layout_train_audit.json"
        layout_audit_cmd = [
            python_exec,
            str(layout_audit_script),
            "--layout", str(layout_out),
            "--ir2", str(ir2_out),
            "--output", str(layout_audit_out),
        ]
        if getattr(args, "strict", False):
            layout_audit_cmd.append("--strict")
        run_cmd(layout_audit_cmd, cwd=PROJECT_ROOT)

        if getattr(args, "generate_runtime", False):
            rt_script = SCRIPTS_DIR / "codegen_train_runtime_v7.py"
            rt_out = out_dir / "generated_train_runtime_v7.c"
            rt_summary = out_dir / "generated_train_runtime_summary_v7.json"
            rt_cmd = [
                python_exec,
                str(rt_script),
                "--ir2", str(ir2_out),
                "--manifest", str(manifest_path),
                "--layout", str(layout_out),
                "--output", str(rt_out),
                "--summary-out", str(rt_summary),
            ]
            run_cmd(rt_cmd, cwd=PROJECT_ROOT)

        log(f"  Generated train IR: {ir1_out}", C_GREEN)
        log(f"  Generated backward IR: {ir2_out}", C_GREEN)
        log(f"  Generated training layout: {layout_out}", C_GREEN)
        log(f"  Training memory audit: {layout_audit_out}", C_GREEN)


    mode = "pretrain" if getattr(args, "pretraining", False) else str(getattr(args, "train_mode", "pretrain"))
    meta = {
        "generated_at": _utc_now_iso(),
        "mode": mode,
        "init": str(getattr(args, "init", "normal_0p02")),
        "template": str(getattr(args, "template", "qwen3")),
        "template_file": str(getattr(args, "template_file", "") or ""),
        "paths": {
            "run_dir": str(out_dir),
            "weights": str(out_dir / "weights.bump"),
            "weights_init": str(out_dir / "weights_init.bump"),
            "manifest": str(out_dir / "weights_manifest.json"),
            "config": str(out_dir / "config.json"),
        },
    }
    (out_dir / "operator_train_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"  Run directory: {out_dir}", C_GREEN)


def step_run_train_sanity(args: argparse.Namespace) -> None:
    """Quick sanity gate: tiny train parity + optional loss-drop assertion."""
    log_step(1, "Running v7 training sanity gate")

    if not getattr(args, "train_json_out", None):
        if getattr(args, "run_dir", None):
            args.train_json_out = str(Path(getattr(args, "run_dir")) / "train_sanity_latest.json")
        else:
            args.train_json_out = str(DEFAULT_REPORT_DIR / "train_sanity_latest.json")

    json_out = step_run_train_e2e(args)
    payload = json.loads(Path(json_out).read_text(encoding="utf-8"))

    pass_parity = bool(payload.get("pass_parity", False))
    loss_curve = payload.get("loss_curve") if isinstance(payload.get("loss_curve"), list) else []
    min_loss_drop = float(getattr(args, "min_loss_drop", 0.0) or 0.0)

    loss_drop_ok = True
    observed_drop = 0.0
    if loss_curve and len(loss_curve) >= 2:
        try:
            first_loss = float(loss_curve[0].get("loss_ck", 0.0))
            last_loss = float(loss_curve[-1].get("loss_ck", 0.0))
            observed_drop = first_loss - last_loss
        except Exception:
            observed_drop = 0.0
    if min_loss_drop > 0.0:
        loss_drop_ok = observed_drop >= min_loss_drop

    summary = {
        "generated_at": _utc_now_iso(),
        "pass_parity": pass_parity,
        "min_loss_drop": min_loss_drop,
        "observed_loss_drop": observed_drop,
        "loss_drop_pass": loss_drop_ok,
        "train_json": str(json_out),
    }
    out = (Path(getattr(args, "run_dir")) / "sanity_overfit.json") if getattr(args, "run_dir", None) else (DEFAULT_REPORT_DIR / "train_sanity_summary_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if pass_parity and loss_drop_ok:
        log(f"  Sanity: PASS (loss_drop={observed_drop:.6f})", C_GREEN)
        log(f"  Summary: {out}", C_GREEN)
        return

    if not pass_parity:
        log_error("Sanity failed: parity check did not pass")
    elif not loss_drop_ok:
        log_error(f"Sanity failed: observed loss drop {observed_drop:.6f} < required {min_loss_drop:.6f}")
    log_error(f"Sanity summary: {out}")
    sys.exit(1)


def step_run_train_parity(args: argparse.Namespace) -> None:
    """Run parity-focused gate bundle: train parity + optional FD/replay checks."""
    log_step(1, "Running v7 training parity gate")

    if not getattr(args, "train_json_out", None):
        if getattr(args, "run_dir", None):
            args.train_json_out = str(Path(getattr(args, "run_dir")) / "train_parity_latest.json")
        else:
            args.train_json_out = str(DEFAULT_REPORT_DIR / "train_parity_latest.json")
    json_out = step_run_train_e2e(args)

    parity_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(parity_python) if parity_python.exists() else sys.executable

    report_dir = Path(getattr(args, "run_dir")) if getattr(args, "run_dir", None) else (DEFAULT_REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    failures = []

    if getattr(args, "with_fd", False):
        fd_script = SCRIPTS_DIR / "check_fd_gradients_v7.py"
        fd_json = report_dir / "fd_gradients_latest.json"
        fd_cmd = [
            python_exec,
            str(fd_script),
            "--seed", str(getattr(args, "train_seed", 42)),
            "--seq-len", str(getattr(args, "train_seq_len", 16)),
            "--total-tokens", str(getattr(args, "train_total_tokens", 1024)),
            "--vocab", str(getattr(args, "train_vocab", 256)),
            "--d-model", str(getattr(args, "train_d_model", 64)),
            "--hidden", str(getattr(args, "train_hidden", 128)),
            "--json-out", str(fd_json),
        ]
        log("  Running finite-difference gradient check", C_DIM)
        rc = run_cmd_allow_fail(fd_cmd, cwd=PROJECT_ROOT).returncode
        if rc != 0:
            failures.append("fd_gradients")

    if getattr(args, "with_replay", False):
        replay_script = SCRIPTS_DIR / "check_replay_determinism_v7.py"
        replay_json = report_dir / "replay_determinism_latest.json"
        replay_cmd = [
            python_exec,
            str(replay_script),
            "--epochs", str(getattr(args, "train_epochs", 3)),
            "--seq-len", str(getattr(args, "train_seq_len", 16)),
            "--total-tokens", str(getattr(args, "train_total_tokens", 1024)),
            "--vocab", str(getattr(args, "train_vocab", 256)),
            "--d-model", str(getattr(args, "train_d_model", 64)),
            "--hidden", str(getattr(args, "train_hidden", 128)),
            "--grad-accum", str(getattr(args, "train_grad_accum", 8)),
            "--optimizer", str(getattr(args, "train_optimizer", "adamw")),
            "--lr", str(getattr(args, "train_lr", 1e-3)),
            "--seed", str(getattr(args, "train_seed", 42)),
            "--json-out", str(replay_json),
        ]
        log("  Running deterministic replay check", C_DIM)
        rc = run_cmd_allow_fail(replay_cmd, cwd=PROJECT_ROOT).returncode
        if rc != 0:
            failures.append("replay")

    payload = json.loads(Path(json_out).read_text(encoding="utf-8"))
    parity_pass = bool(payload.get("pass_parity", False))
    if not parity_pass:
        failures.append("train_parity")

    gate = {
        "generated_at": _utc_now_iso(),
        "train_json": str(json_out),
        "pass_train_parity": parity_pass,
        "with_fd": bool(getattr(args, "with_fd", False)),
        "with_replay": bool(getattr(args, "with_replay", False)),
        "failures": failures,
        "passed": len(failures) == 0,
    }
    gate_out = report_dir / "parity_report.json"
    gate_out.write_text(json.dumps(gate, indent=2), encoding="utf-8")

    if failures:
        log_error(f"Parity gate failed: {', '.join(failures)}")
        log_error(f"Gate report: {gate_out}")
        sys.exit(1)

    log(f"  Parity gate: PASS", C_GREEN)
    log(f"  Gate report: {gate_out}", C_GREEN)




def step_run_train_suite(args: argparse.Namespace) -> None:
    """Run stability sweep epochs (1,3,5,10 by default) + optional spot profiling.

    Outputs:
      - <run_dir>/train_e{E}.json for each sweep epoch (or version/v7/.cache/reports if --run not set)
      - <run_dir>/training_epoch_sweep_latest.json consolidated table source
      - training_* telemetry materialized from final sweep epoch
    """
    log_step(1, "Running training stability suite")

    epochs_raw = str(getattr(args, "epochs_list", "1,3,5,10") or "1,3,5,10")
    try:
        epochs = []
        for tok in [x.strip() for x in epochs_raw.split(",") if x.strip()]:
            v = int(tok)
            if v < 1:
                raise ValueError
            if v not in epochs:
                epochs.append(v)
    except Exception:
        log_error(f"Invalid --epochs-list: {epochs_raw}")
        sys.exit(2)
    if not epochs:
        log_error("--epochs-list produced no valid epochs")
        sys.exit(2)

    report_dir = Path(getattr(args, "run_dir")) if getattr(args, "run_dir", None) else (DEFAULT_REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    sweep_outputs = []

    for ep in epochs:
        out_json = report_dir / f"train_e{ep}.json"
        run_args = argparse.Namespace(**vars(args))
        run_args.train_epochs = ep
        run_args.train_json_out = str(out_json)
        run_args.profile_train = "none"
        step_run_train_e2e(run_args)

        data = json.loads(out_json.read_text(encoding="utf-8"))
        step_profile = data.get("step_profile") if isinstance(data.get("step_profile"), dict) else {}
        rows.append({
            "epoch": ep,
            "pass_parity": bool(data.get("pass_parity", False)),
            "final_ck_loss": float(data.get("final_ck_loss", 0.0) or 0.0),
            "final_torch_loss": float(data.get("final_torch_loss", 0.0) or 0.0),
            "max_loss_abs_diff": float(data.get("max_loss_abs_diff", 0.0) or 0.0),
            "final_param_max_abs_diff": float(data.get("final_param_max_abs_diff", 0.0) or 0.0),
            "steps": int(data.get("steps", 0) or 0),
            "micro_steps": int(data.get("micro_steps", 0) or 0),
            "train_tok_s": float(step_profile.get("train_tok_s", 0.0) or 0.0),
            "json_out": str(out_json),
        })
        sweep_outputs.append(out_json)

    profile_mode = str(getattr(args, "profile_train", "none") or "none").lower()
    profile_epoch = int(getattr(args, "profile_epoch", 3) or 3)
    profile_row = None

    if profile_mode != "none":
        profile_json = report_dir / f"train_profile_e{profile_epoch}_{profile_mode}.json"
        prof_args = argparse.Namespace(**vars(args))
        prof_args.train_epochs = profile_epoch
        prof_args.train_json_out = str(profile_json)
        prof_args.profile_train = profile_mode
        step_run_train_e2e(prof_args)

        pdata = json.loads(profile_json.read_text(encoding="utf-8"))
        pstep = pdata.get("step_profile") if isinstance(pdata.get("step_profile"), dict) else {}
        profile_row = {
            "epoch": profile_epoch,
            "mode": profile_mode,
            "pass_parity": bool(pdata.get("pass_parity", False)),
            "final_ck_loss": float(pdata.get("final_ck_loss", 0.0) or 0.0),
            "final_param_max_abs_diff": float(pdata.get("final_param_max_abs_diff", 0.0) or 0.0),
            "train_tok_s": float(pstep.get("train_tok_s", 0.0) or 0.0),
            "json_out": str(profile_json),
        }

    # Keep training_*_latest stable to final sweep epoch for dashboard continuity.
    last_sweep_json = sweep_outputs[-1]
    try:
        _materialize_train_telemetry(last_sweep_json, profile_meta={"mode": "none", "artifacts": []})
    except Exception as e:
        log(f"  Warning: failed to rematerialize latest training telemetry: {e}", C_ORANGE)

    payload = {
        "generated_at": _utc_now_iso(),
        "epochs": epochs,
        "runs": rows,
        "profile_run": profile_row,
        "source": "ck_run_v7 train-suite",
    }
    sweep_path = report_dir / "training_epoch_sweep_latest.json"
    sweep_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    passed = sum(1 for r in rows if r.get("pass_parity"))
    log(f"  Sweep summary: {passed}/{len(rows)} epochs parity-pass", C_GREEN if passed == len(rows) else C_ORANGE)
    log(f"  Sweep report: {sweep_path}", C_GREEN)




def run_reverse_validation(work_dir: Path, verbose: bool = False) -> bool:
    """Run IR reverse validation to check IR Lower 3 consistency.

    Validates:
    - Buffer completeness (all references have definitions)
    - Manifest coverage (all weights are used)
    - Bias accounting (biases in manifest appear in ops)
    - Op sequence (no read-before-write)
    - Size consistency (file_size matches shape+dtype)
    - Kernel signatures (ops match kernel map signatures)

    Returns True if all checks pass.
    """
    log(f"{C_ORANGE}[reverse-test]{C_RESET} Running IR reverse validation")

    # Import the validator
    try:
        from ir_reverse_validator import run_validation
    except ImportError:
        # Try relative import
        validator_path = SCRIPTS_DIR / "ir_reverse_validator.py"
        if validator_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("ir_reverse_validator", validator_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            run_validation = module.run_validation
        else:
            log_error("ir_reverse_validator.py not found")
            return False

    # Find lowered IR files
    lowered_decode = work_dir / "lowered_decode_call.json"
    lowered_prefill = work_dir / "lowered_prefill_call.json"
    manifest_path = work_dir / "weights_manifest.json"

    all_passed = True

    # Validate decode IR
    if lowered_decode.exists():
        log(f"  Validating decode IR: {lowered_decode.name}", C_DIM)
        passed, report = run_validation(
            lowered_path=lowered_decode,
            manifest_path=manifest_path if manifest_path.exists() else None,
            kernel_maps_dir=KERNEL_MAPS_DIR,
            verbose=verbose,
        )
        if not passed:
            all_passed = False
            log(f"  {C_RED}Decode validation FAILED{C_RESET}")
        else:
            log(f"  {C_GREEN}Decode validation PASSED{C_RESET}")

        if verbose or not passed:
            print(report)
    else:
        log(f"  {C_DIM}Skipping decode validation (no lowered_decode_call.json){C_RESET}")

    # Validate prefill IR
    if lowered_prefill.exists():
        log(f"  Validating prefill IR: {lowered_prefill.name}", C_DIM)
        passed, report = run_validation(
            lowered_path=lowered_prefill,
            manifest_path=manifest_path if manifest_path.exists() else None,
            kernel_maps_dir=KERNEL_MAPS_DIR,
            verbose=verbose,
        )
        if not passed:
            all_passed = False
            log(f"  {C_RED}Prefill validation FAILED{C_RESET}")
        else:
            log(f"  {C_GREEN}Prefill validation PASSED{C_RESET}")

        if verbose or not passed:
            print(report)
    else:
        log(f"  {C_DIM}Skipping prefill validation (no lowered_prefill_call.json){C_RESET}")

    return all_passed


def step_run_chat(model_dir: Path, args: argparse.Namespace, gguf_path: Path = None):
    """Run chat interface."""
    log_step(6, "Starting chat")

    # Ensure libckernel_engine.so is findable
    kernel_lib = BUILD_DIR / "libckernel_engine.so"
    dst_lib = model_dir / "libckernel_engine.so"
    _sync_runtime_lib(kernel_lib, dst_lib, "libckernel_engine.so")

    # Ensure libckernel_tokenizer.so is findable (for BPE tokenizer)
    tokenizer_lib = BUILD_DIR / "libckernel_tokenizer.so"
    dst_tok = model_dir / "libckernel_tokenizer.so"
    _sync_runtime_lib(tokenizer_lib, dst_tok, "libckernel_tokenizer.so")

    if kernel_lib.exists():
        # Also set LD_LIBRARY_PATH for the subprocess
        ld_path = str(BUILD_DIR)
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{ld_path}:{model_dir}:{current_ld}"

    # Threading policy:
    # - CK threadpool handles parallelism (CK_NUM_THREADS).
    # - Leave OpenMP effectively serial by default to avoid oversubscription/races.
    #   Users can still override explicitly in their shell env.
    if not os.environ.get("CK_NUM_THREADS"):
        os.environ["CK_NUM_THREADS"] = str(_detect_default_ck_threads())
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")

    cmd = [
        sys.executable,
        str(ROOT_SCRIPTS_DIR / "ck_chat.py"),
        "--model-dir",
        str(model_dir),
    ]

    if gguf_path:
        cmd.extend(["--gguf", str(gguf_path)])
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.max_tokens:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])
    if getattr(args, "no_chat_template", False):
        cmd.append("--no-chat-template")
    elif getattr(args, "chat_template", None):
        cmd.extend(["--chat-template", args.chat_template])
    if getattr(args, "python_tokenizer", False):
        cmd.append("--python-tokenizer")
    if getattr(args, 'parity', False):
        cmd.append("--parity")

    # If profiling, run as subprocess so post-run summary can execute
    # Otherwise, replace current process (original behavior)
    if os.environ.get("CK_PROFILE") == "1":
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
    else:
        os.execvp(sys.executable, cmd)


def step_run_c_cli_smoke(lib_path: Path, weights_path: Path, prompt: str, max_tokens: int):
    """Run native v7 CLI once to validate true-BPE end-to-end."""
    log(f"{C_ORANGE}[smoke]{C_RESET} Running native v7 CLI (true-BPE)")

    cli_path = PROJECT_ROOT / "build" / "ck-cli-v7"
    if not cli_path.exists():
        run_cmd(["make", "ck-cli-v7"], cwd=PROJECT_ROOT)

    env = os.environ.copy()
    ld_path = str(PROJECT_ROOT / "build")
    env["LD_LIBRARY_PATH"] = f"{ld_path}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        str(cli_path),
        str(lib_path),
        str(weights_path),
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            input="",
            text=True,
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as e:
        log_error("Native v7 CLI smoke test timed out")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        log_error("Native v7 CLI smoke test failed")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)

    log(f"  {C_GREEN}Native v7 CLI smoke test: OK{C_RESET}", C_DIM)


def step_run_c_cli_parity_dump(
    lib_path: Path,
    weights_path: Path,
    prompt: str,
    max_tokens: int,
    ctx_size: Optional[int],
) -> None:
    """Run native ck-cli to generate CK parity dumps (dump.bin)."""
    log(f"{C_ORANGE}[parity]{C_RESET} Running ck-cli-v7 for CK dumps")

    cli_path = PROJECT_ROOT / "build" / "ck-cli-v7"
    if not cli_path.exists():
        run_cmd(["make", "build/ck-cli-v7"], cwd=PROJECT_ROOT)

    env = os.environ.copy()
    ld_path = str(PROJECT_ROOT / "build")
    env["LD_LIBRARY_PATH"] = f"{ld_path}:{env.get('LD_LIBRARY_PATH', '')}"

    parity_dump_dir = lib_path.parent / "ck_parity_dumps"
    parity_dump_dir.mkdir(parents=True, exist_ok=True)
    env["CK_PARITY_DUMP"] = "1"
    env["CK_PARITY_DIR"] = str(parity_dump_dir)

    if max_tokens <= 0:
        max_tokens = 1

    cmd = [
        str(cli_path),
        str(lib_path),
        str(weights_path),
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--no-chat-template",
    ]
    if ctx_size and ctx_size > 0:
        cmd.extend(["--context", str(ctx_size)])

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            input="",
            text=True,
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as e:
        log_error("ck-cli parity dump timed out")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return

    if result.returncode != 0:
        log_error("ck-cli parity dump failed")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return

    dump_file = parity_dump_dir / "dump.bin"
    if not dump_file.exists() or dump_file.stat().st_size == 0:
        log_error("ck-cli parity dump empty")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return

    log(f"  {C_GREEN}ck-cli parity dump: OK{C_RESET}", C_DIM)


def _run_llamacpp_parity(
    work_dir: Path,
    prompt: str,
    max_tokens: int,
    ctx_size: Optional[int] = None,
    temperature: Optional[float] = None,
    llama_filter: Optional[str] = None,
    llama_layer: Optional[int] = None,
    llama_stop_after: Optional[int] = None,
    llama_include_global: bool = False,
    llama_timeout: Optional[int] = None,
    gguf_path_hint: Optional[Path] = None,
) -> bool:
    """Run llama.cpp parity binary to generate reference dumps."""
    log(f"\n{C_ORANGE}[llamacpp-parity]{C_RESET} Running llama.cpp for reference dumps")

    # Prefer patched parity binary; fallback to local llama.cpp main.
    llm_path = PROJECT_ROOT / "build" / "llama-parity"
    if not llm_path.exists():
        llm_path = PROJECT_ROOT / "llama.cpp" / "main"
    if not llm_path.exists():
        log_error("llama.cpp parity binary not found (expected build/llama-parity or llama.cpp/main)")
        return False

    gguf_path = gguf_path_hint if gguf_path_hint and gguf_path_hint.exists() else None
    if not gguf_path:
        for pattern in ("*.gguf", "*/*.gguf"):
            found = next(iter(work_dir.glob(pattern)), None)
            if found:
                gguf_path = found
                break
    if not gguf_path:
        gguf_path = next(iter(work_dir.parent.glob("*.gguf")), None)
    if not gguf_path:
        log_error("GGUF file not found for llama parity run")
        return False

    ref_dump_dir = work_dir / "llama_parity_dumps"
    ref_dump_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CKDMP_DIR"] = str(ref_dump_dir)
    if llama_layer is None:
        env["CKDMP_ALL_LAYERS"] = "1"
    if llama_filter:
        env["CKDMP_FILTER"] = llama_filter
    if llama_layer is not None:
        env["CKDMP_LAYER"] = str(llama_layer)
    if llama_stop_after is not None:
        env["CKDMP_STOP_AFTER"] = str(llama_stop_after)
    if llama_include_global:
        env["CKDMP_INCLUDE_GLOBAL"] = "1"

    temp = 0.0 if temperature is None else float(temperature)
    cmd = [
        str(llm_path),
        "-m",
        str(gguf_path),
        "-p",
        prompt,
        "-no-cnv",
        "--simple-io",
        "--no-warmup",
        "--temp",
        f"{temp}",
        "-n",
        str(max_tokens),
    ]
    if ctx_size and ctx_size > 0:
        cmd.extend(["--ctx-size", str(ctx_size)])

    try:
        timeout_sec = 600 if llama_timeout is None else int(llama_timeout)
        if timeout_sec <= 0:
            timeout_sec = None
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        log_error("llama.cpp parity run timed out")
        return False
    except Exception as e:
        log_error(f"llama.cpp parity run error: {e}")
        return False

    ref_dump = ref_dump_dir / "dump.bin"
    ref_index = ref_dump_dir / "index.json"
    has_dump = ref_dump.exists() and ref_dump.stat().st_size > 0
    has_index = ref_index.exists() and ref_index.stat().st_size > 0

    if result.returncode == 0 and has_dump:
        log(f"  Reference dump: {ref_dump}", C_GREEN)
        return True

    # CKDMP_STOP_AFTER intentionally exits non-zero after enough dumps.
    if result.returncode != 0 and has_dump and has_index:
        log(f"  llama.cpp exited with code {result.returncode} after writing dumps", C_ORANGE)
        log(f"  Reference dump: {ref_dump}", C_GREEN)
        return True

    log_error("llama.cpp parity dump failed")
    if result.stderr:
        log(f"  {result.stderr[:300]}", C_DIM)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Profiling Summary
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_profile_summary(work_dir: Path):
    """Parse profile CSV and generate summary JSON for ir_visualizer."""
    import csv as csv_mod
    csv_path = work_dir / "profile_decode.csv"
    if not csv_path.exists():
        log(f"  No profile CSV found at {csv_path}", C_DIM)
        return

    entries = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            entries.append(row)

    if not entries:
        log(f"  Profile CSV is empty", C_DIM)
        return

    # Aggregate by op (legacy view: token_id == 0, typically prefill)
    by_op = {}
    by_layer = {}
    total_us = 0
    for e in entries:
        if e.get('token_id', '0') != '0':  # Only first token
            continue
        op = e.get('op', 'unknown')
        layer = int(e.get('layer', -1))
        us = float(e.get('time_us', 0))
        total_us += us
        by_op[op] = by_op.get(op, 0) + us
        if layer >= 0:
            if layer not in by_layer:
                by_layer[layer] = {}
            by_layer[layer][op] = by_layer[layer].get(op, 0) + us

    # Also aggregate by explicit mode across all token ids.
    by_mode = {}
    for e in entries:
        mode = e.get('mode', 'unknown')
        op = e.get('op', 'unknown')
        us = float(e.get('time_us', 0))
        mode_bucket = by_mode.setdefault(mode, {"total_us": 0.0, "by_op": {}})
        mode_bucket["total_us"] += us
        mode_bucket["by_op"][op] = mode_bucket["by_op"].get(op, 0.0) + us

    summary = {
        "total_us": total_us,
        "total_ms": total_us / 1000,
        "by_op": by_op,
        "by_layer": {str(k): v for k, v in sorted(by_layer.items())},
        "by_mode": by_mode,
        "entries": entries,
    }

    summary_path = work_dir / "profile_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n{C_GREEN}Profile summary:{C_RESET}")
    log(f"  Total: {total_us / 1000:.2f} ms")
    log(f"  CSV:   {csv_path}")
    log(f"  JSON:  {summary_path}")

    # Show top 5 hotspots for token_id==0 view
    sorted_ops = sorted(by_op.items(), key=lambda x: x[1], reverse=True)
    if sorted_ops:
        log(f"  Top hotspots (token_id=0):")
        for op, us in sorted_ops[:5]:
            pct = us / total_us * 100 if total_us > 0 else 0
            log(f"    {op:20s} {us/1000:8.2f} ms ({pct:5.1f}%)")

    # Decode-specific view across all decode tokens (what operators usually want).
    decode_bucket = by_mode.get("decode", {})
    decode_total = float(decode_bucket.get("total_us", 0.0))
    decode_by_op = decode_bucket.get("by_op", {})
    if decode_total > 0 and decode_by_op:
        sorted_decode = sorted(decode_by_op.items(), key=lambda x: x[1], reverse=True)
        log(f"  Top decode hotspots (all decode tokens):")
        for op, us in sorted_decode[:5]:
            pct = us / decode_total * 100 if decode_total > 0 else 0
            log(f"    {op:20s} {us/1000:8.2f} ms ({pct:5.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace):
    """Run the full v7 pipeline."""
    model_input = args.model
    weights_path = None
    manifest_input_path = None
    gguf_path_for_tokenizer = None  # Track GGUF path for tokenizer extraction
    v7_mode = V7_MODE

    # Optional fast path: training parity harness (CK vs PyTorch).
    # This path validates training numerics and does not require model conversion.
    if getattr(args, "train_e2e", False):
        if getattr(args, "model", None):
            log("  --train-e2e uses tiny training harness; model input is ignored for now", C_DIM)
        step_run_train_e2e(args)
        return

    # Step 0: Regenerate kernel registry from kernel maps if needed
    step_regenerate_kernel_registry(force=getattr(args, 'force_compile', False))

    # Detect input type
    input_type, info = detect_input_type(model_input)
    version_tag = "v7" if v7_mode else "v4"
    log(f"{C_ORANGE}C-Kernel-Engine {version_tag}{C_RESET}")
    log(f"Input: {model_input} ({input_type})", C_DIM)

    # Determine working directory
    if input_type == 'hf_id':
        model_id = info['model_id']
        work_dir = CACHE_DIR / model_id.replace('/', '--')
        model_dir = step_download(model_id, CACHE_DIR, force=args.force_download)

        # Check if this is a GGUF-only repo (no safetensors)
        has_safetensors = list(model_dir.glob("*.safetensors")) or list(model_dir.glob("model*.safetensors"))
        gguf_files = list(model_dir.glob("*.gguf"))

        if gguf_files and not has_safetensors:
            # GGUF-only repo - pick the best GGUF file
            log(f"  Detected GGUF-only repo with {len(gguf_files)} GGUF files", C_DIM)

            # Prefer Q4_K_M if available, otherwise first file
            gguf_path = None
            for pattern in ["*q4_k_m*", "*q4_k*", "*q6_k*", "*"]:
                matches = list(model_dir.glob(pattern + ".gguf"))
                if matches:
                    gguf_path = matches[0]
                    break

            if not gguf_path:
                gguf_path = gguf_files[0]

            gguf_path_for_tokenizer = gguf_path
            log(f"  Using GGUF: {gguf_path.name}", C_GREEN)
            if v7_mode:
                manifest_input_path = step_inspect_weights_v7(
                    input_type, model_dir, gguf_path, work_dir, force=args.force_inspect
                )
                if args.inspect_only:
                    return
            # ensure_tokenizer_files still fetches tokenizer.json for ck_chat.py BPE encoding
            # but we no longer pass it to the converter - GGUF has complete vocab
            ensure_tokenizer_files(model_id, work_dir)
            weights_path, config_path = step_convert_gguf(
                gguf_path, work_dir,
                force=args.force_convert,
            )
            manifest_path = work_dir / "weights_manifest.json"
        else:
            # Standard HF repo with safetensors
            config_path = model_dir / "config.json"
            if v7_mode:
                manifest_input_path = step_inspect_weights_v7(
                    input_type, model_dir, None, work_dir, force=args.force_inspect
                )
                if args.inspect_only:
                    return
            weights_path = step_convert_hf(
                model_dir, work_dir,
                weight_dtype=args.weight_dtype or "float32",
                force=args.force_convert,
                tokenizer_json=(model_dir / "tokenizer.json") if (model_dir / "tokenizer.json").exists() else None
            )
            manifest_path = work_dir / "weights_manifest.json"

    elif input_type == 'gguf':
        gguf_path = info['path']
        gguf_path_for_tokenizer = gguf_path
        work_dir = CACHE_DIR / gguf_path.stem
        if v7_mode:
            manifest_input_path = step_inspect_weights_v7(
                input_type, None, gguf_path, work_dir, force=args.force_inspect
            )
            if args.inspect_only:
                return
        # Local GGUF - use GGUF vocab directly (no tokenizer.json override)
        weights_path, config_path = step_convert_gguf(
            gguf_path, work_dir,
            force=args.force_convert,
        )
        manifest_path = work_dir / "weights_manifest.json"

    elif input_type == 'local_dir':
        model_dir = info['path']
        work_dir = model_dir / ".ck_build"
        config_path = model_dir / "config.json"

        # Convert weights
        if v7_mode:
            manifest_input_path = step_inspect_weights_v7(
                input_type, model_dir, None, work_dir, force=args.force_inspect
            )
            if args.inspect_only:
                return
        weights_path = step_convert_hf(
            model_dir, work_dir,
            weight_dtype=args.weight_dtype or "float32",
            force=args.force_convert,
            tokenizer_json=(model_dir / "tokenizer.json") if (model_dir / "tokenizer.json").exists() else None
        )
        manifest_path = work_dir / "weights_manifest.json"

    elif input_type == 'local_config':
        config_path = info['path']
        work_dir = config_path.parent / ".ck_build"
        manifest_path = None
        # No weight conversion for config-only (assume weights.bump exists)
        if v7_mode and not args.weight_dtype:
            log_error("v7 requires a weights manifest for config-only runs (or pass --weight-dtype)")
            sys.exit(1)
        if v7_mode and args.inspect_only:
            log_error("inspect-only requires a weights source (GGUF or checkpoint directory)")
            sys.exit(1)

    elif input_type == 'hf_gguf':
        # Download single GGUF file from HuggingFace
        repo_id = info['repo_id']
        filename = info['filename']
        work_dir = CACHE_DIR / repo_id.replace('/', '--')

        gguf_path = step_download_gguf(repo_id, filename, CACHE_DIR, force=args.force_download)
        gguf_path_for_tokenizer = gguf_path
        if v7_mode:
            manifest_input_path = step_inspect_weights_v7(
                input_type, None, gguf_path, work_dir, force=args.force_inspect
            )
            if args.inspect_only:
                return
        # ensure_tokenizer_files still fetches tokenizer.json for ck_chat.py BPE encoding
        # but we no longer pass it to the converter - GGUF has complete vocab
        ensure_tokenizer_files(repo_id, work_dir)
        weights_path, config_path = step_convert_gguf(
            gguf_path, work_dir,
            force=args.force_convert,
        )
        manifest_path = work_dir / "weights_manifest.json"

    else:
        log_error(f"Unknown input type: {input_type}")
        sys.exit(1)

    if v7_mode and not manifest_input_path and not args.weight_dtype and manifest_path is None:
        log_error("v7 requires a weights manifest (inspect-only or conversion) unless --weight-dtype is provided")
        sys.exit(1)

    # Build IR
    # If debug/parity/profiling is enabled, force recompile for instrumentation.
    parity_dump = bool(getattr(args, "parity_dump", False))
    detailed_llama_parity = bool(getattr(args, "detailed_llamacpp_parity", False))
    if detailed_llama_parity:
        parity_dump = True
    force_for_debug = (
        args.force_compile
        or getattr(args, "debug", False)
        or getattr(args, "parity", False)
        or getattr(args, "profile", False)
        or parity_dump
        or detailed_llama_parity
    )
    manifest_for_dtype = None
    
    # Vocabulary metadata (prefer non-zero values from config/manifest)
    num_merges = 0
    total_vocab_bytes = 0
    cfg_data = None

    if config_path and config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg_data = json.load(f)
            num_merges = int(cfg_data.get("num_merges") or 0)
            total_vocab_bytes = int(cfg_data.get("total_vocab_bytes") or 0)
        except Exception:
            cfg_data = None

    def _maybe_update_vocab_meta(path: Optional[Path]) -> None:
        nonlocal num_merges, total_vocab_bytes
        if not path or not path.exists():
            return
        try:
            with open(path, "r") as f:
                mdata = json.load(f)
            m_merges = int(mdata.get("num_merges") or 0)
            m_bytes = int(mdata.get("total_vocab_bytes") or 0)
            if m_merges > 0:
                num_merges = m_merges
            if m_bytes > 0:
                total_vocab_bytes = m_bytes
        except Exception:
            return

    if manifest_path and manifest_path.exists():
        manifest_for_dtype = manifest_path
        _maybe_update_vocab_meta(manifest_path)
        _maybe_update_vocab_meta(manifest_input_path)
    elif manifest_input_path and manifest_input_path.exists():
        manifest_for_dtype = manifest_input_path
        _maybe_update_vocab_meta(manifest_input_path)

    # Inject metadata into config.json for build_ir_v7
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg_data = json.load(f)
            cfg_data["num_merges"] = num_merges
            cfg_data["total_vocab_bytes"] = total_vocab_bytes
            with open(config_path, "w") as f:
                json.dump(cfg_data, f, indent=2)
        except Exception:
            pass

    weight_dtype = normalize_weight_dtype(args.weight_dtype, manifest_for_dtype)
    bump_path = work_dir / "weights.bump"
    layout_path = step_build_ir(
        config_path, work_dir,
        manifest_path=manifest_path or manifest_input_path,
        bump_path=bump_path if bump_path.exists() else None,
        weight_dtype=weight_dtype,
        force=force_for_debug,
        debug=getattr(args, 'debug', False),
        parity=getattr(args, 'parity', False),
        codegen_version=getattr(args, 'codegen', 'v7'),
        int8_activations=getattr(args, 'int8_activations', False),
        context_len=getattr(args, 'context_len', None),
        logits_layout=getattr(args, 'logits_layout', None),
        no_fusion=getattr(args, 'no_fusion', False),
        layout_mode=getattr(args, 'layout_mode', 'region'),
        layer_limit=getattr(args, 'layer_limit', None),
        profile=getattr(args, 'profile', False),
        parallel_decode=getattr(args, 'parallel_decode', False),
    )

    # Generate C code
    model_c_path = step_codegen(
        layout_path,
        work_dir,
        force=force_for_debug,
        profile=getattr(args, 'profile', False),
        dump=parity_dump,
    )

    # Run reverse validation if requested (validates IR Lower 3 before compile)
    if getattr(args, 'reverse_test', False):
        reverse_verbose = getattr(args, 'reverse_test_verbose', False)
        if not run_reverse_validation(work_dir, verbose=reverse_verbose):
            log_error("IR reverse validation failed - IR Lower 3 has consistency issues")
            if not getattr(args, 'force_compile', False):
                log(f"  {C_DIM}Use --force-compile to continue anyway{C_RESET}")
                sys.exit(1)
            else:
                log(f"  {C_ORANGE}Continuing due to --force-compile{C_RESET}")

    # Set up profiling environment before compile
    if getattr(args, 'profile', False):
        os.environ["CK_PROFILE"] = "1"
        os.environ["CK_PROFILE_CSV"] = str(work_dir / "profile_decode.csv")
        os.environ["CK_PROFILE_JSON"] = str(work_dir / "profile_decode.json")
        log(f"  Profiling enabled: CSV → {work_dir / 'profile_decode.csv'}", C_DIM)
    else:
        os.environ.pop("CK_PROFILE", None)
        os.environ.pop("CK_PROFILE_CSV", None)
        os.environ.pop("CK_PROFILE_JSON", None)

    if parity_dump:
        parity_dump_dir = work_dir / "ck_parity_dumps"
        parity_dump_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CK_PARITY_DUMP"] = "1"
        os.environ["CK_PARITY_DIR"] = str(parity_dump_dir)
        log(f"  Parity dumping enabled: {parity_dump_dir}", C_DIM)
    else:
        os.environ.pop("CK_PARITY_DUMP", None)
        os.environ.pop("CK_PARITY_DIR", None)

    # Compile
    lib_path = step_compile(model_c_path, work_dir, force=force_for_debug)

    # Copy tokenizer if available
    tokenizer_src = None
    if input_type in ('hf_id', 'local_dir'):
        tokenizer_src = (model_dir if input_type == 'hf_id' else info['path']) / "tokenizer.json"
    if tokenizer_src and tokenizer_src.exists():
        tokenizer_dst = work_dir / "tokenizer.json"
        if not tokenizer_dst.exists():
            shutil.copy(tokenizer_src, tokenizer_dst)

    if args.test:
        log(f"{C_ORANGE}[test]{C_RESET} Running v7 smoke tests")
        layout_decode = work_dir / "layout_decode.json"
        if not layout_decode.exists():
            log_error(f"layout_decode.json not found in {work_dir}")
            sys.exit(1)
        manifest_map = work_dir / "weights_manifest.map"
        manifest_json = work_dir / "weights_manifest.json"
        backup_map = _backup_file(manifest_map)
        backup_json = _backup_file(manifest_json)
        try:
            dummy_weights = build_dummy_weights(layout_decode, work_dir)
            use_valgrind = shutil.which("valgrind") is not None
            run_smoke_test(work_dir, dummy_weights, use_valgrind)
        finally:
            _restore_file(manifest_map, backup_map)
            _restore_file(manifest_json, backup_json)

        if weights_path and Path(weights_path).exists():
            run_smoke_test(work_dir, Path(weights_path), False)

        lib_engine = PROJECT_ROOT / "build" / "libckernel_engine.so"
        if lib_engine.exists():
            try:
                __import__("torch")
                run_parity_tests()
            except ImportError:
                log("  Skipping parity tests (torch not installed)", C_DIM)
        else:
            log("  Skipping parity tests (build/libckernel_engine.so missing)", C_DIM)

    # Determine effective context length for parity tools (prefer explicit CLI override).
    effective_ctx = getattr(args, "context_len", None)

    # Run chat (unless generate-only)
    if args.generate_only or args.test_only:
        log(f"\n{C_GREEN}Generated:{C_RESET}")
        log(f"  Layout:  {layout_path}")
        log(f"  C code:  {model_c_path}")
        log(f"  Library: {lib_path}")
    else:
        print()
        if getattr(args, "c_cli_smoke", False):
            weights_bump = Path(weights_path) if weights_path else (work_dir / "weights.bump")
            if not weights_bump.exists():
                log_error(f"weights.bump not found at {weights_bump}")
                sys.exit(1)
            prompt = getattr(args, "c_cli_prompt", None) or "Hello"
            max_tokens = int(getattr(args, "c_cli_max_tokens", 16) or 16)
            step_run_c_cli_smoke(lib_path, weights_bump, prompt, max_tokens)
        if detailed_llama_parity:
            weights_bump = Path(weights_path) if weights_path else (work_dir / "weights.bump")
            if not weights_bump.exists():
                log_error(f"weights.bump not found at {weights_bump}")
                sys.exit(1)
            parity_prompt = getattr(args, "prompt", None) or "Hello"
            parity_max_tokens = int(getattr(args, "max_tokens", 1) or 1)
            step_run_c_cli_parity_dump(lib_path, weights_bump, parity_prompt, parity_max_tokens, effective_ctx)
            _run_llamacpp_parity(
                work_dir,
                prompt=parity_prompt,
                max_tokens=parity_max_tokens,
                ctx_size=effective_ctx,
                temperature=getattr(args, "temperature", None),
                llama_filter=getattr(args, "llama_filter", None),
                llama_layer=getattr(args, "llama_layer", None),
                llama_stop_after=getattr(args, "llama_stop_after", None),
                llama_include_global=getattr(args, "llama_include_global", False),
                llama_timeout=getattr(args, "llama_timeout", None),
                gguf_path_hint=gguf_path_for_tokenizer,
            )
        else:
            step_run_chat(work_dir, args, gguf_path=gguf_path_for_tokenizer)

    # Generate profile summary if profiling was enabled
    if getattr(args, 'profile', False):
        _generate_profile_summary(work_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive Model Selector
# ═══════════════════════════════════════════════════════════════════════════════

def find_available_models() -> list[dict]:
    """Find all available GGUF models (local + cached)."""
    models = []

    # 1. Local GGUF files in project root
    for gguf in PROJECT_ROOT.glob("*.gguf"):
        size_mb = gguf.stat().st_size / 1e6
        if size_mb < 50:  # Skip very small files (vocab only)
            continue
        models.append({
            'type': 'local',
            'name': gguf.stem,
            'path': str(gguf),
            'size_mb': size_mb,
            'display': f"{gguf.name} ({size_mb:.1f} MB)",
        })

    # 2. Cached models
    if CACHE_DIR.exists():
        for model_dir in CACHE_DIR.iterdir():
            if not model_dir.is_dir():
                continue

            # Check for GGUF files
            gguf_files = list(model_dir.glob("*.gguf"))
            if gguf_files:
                gguf = gguf_files[0]
                size_mb = gguf.stat().st_size / 1e6
                models.append({
                    'type': 'cached_gguf',
                    'name': model_dir.name.replace('--', '/'),
                    'path': str(gguf),
                    'size_mb': size_mb,
                    'display': f"{model_dir.name.replace('--', '/')} ({size_mb:.1f} MB)",
                })
            # Check for config.json (HF models)
            elif (model_dir / "config.json").exists():
                size_mb = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / 1e6
                models.append({
                    'type': 'cached_hf',
                    'name': model_dir.name.replace('--', '/'),
                    'path': str(model_dir),
                    'size_mb': size_mb,
                    'display': f"{model_dir.name.replace('--', '/')} ({size_mb:.1f} MB)",
                })

    # Sort by size (largest first)
    models.sort(key=lambda x: -x['size_mb'])
    return models


def interactive_model_select() -> Optional[str]:
    """Interactive model selection menu."""
    print()
    print(f"{C_ORANGE}╔══════════════════════════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_ORANGE}║{C_RESET}  {C_BOLD}C-Kernel-Engine v7 - Interactive Model Selector{C_RESET}                     {C_ORANGE}║{C_RESET}")
    print(f"{C_ORANGE}╚══════════════════════════════════════════════════════════════════════╝{C_RESET}")
    print()

    models = find_available_models()

    if not models:
        print(f"  {C_DIM}No models found.{C_RESET}")
        print()
        print(f"  {C_BOLD}Options:{C_RESET}")
        print(f"    1. Download a model from HuggingFace:")
        print(f"       {C_CYAN}./ck-v7 run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf{C_RESET}")
        print()
        print(f"    2. Download a GGUF file manually:")
        print(f"       {C_CYAN}huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir .{C_RESET}")
        print()
        return None

    print(f"  {C_BOLD}Available Models:{C_RESET}")
    print()

    for i, model in enumerate(models, 1):
        type_tag = {
            'local': f"{C_GREEN}[local]{C_RESET}",
            'cached_gguf': f"{C_BLUE}[cached]{C_RESET}",
            'cached_hf': f"{C_CYAN}[HF]{C_RESET}",
        }.get(model['type'], '')
        print(f"    {C_BOLD}[{i}]{C_RESET} {type_tag} {model['display']}")

    print()
    print(f"    {C_DIM}[h] Download from HuggingFace{C_RESET}")
    print(f"    {C_DIM}[q] Quit{C_RESET}")
    print()

    try:
        choice = input(f"  {C_BOLD}Select model (1-{len(models)}):{C_RESET} ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return None

    if choice == 'q' or choice == '':
        return None

    if choice == 'h':
        print()
        print(f"  {C_BOLD}Enter HuggingFace model (examples):{C_RESET}")
        print(f"    - hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf")
        print(f"    - Qwen/Qwen2.5-3B-Instruct-GGUF")
        print(f"    - HuggingFaceTB/SmolLM-135M")
        print()
        try:
            hf_input = input(f"  {C_BOLD}Model:{C_RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return None
        return hf_input if hf_input else None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]['path']
    except ValueError:
        pass

    print(f"  {C_RED}Invalid selection{C_RESET}")
    return None


def run_interactive(args: argparse.Namespace):
    """Run interactive model selection and pipeline."""
    model_path = interactive_model_select()

    if not model_path:
        return

    print()
    log(f"Selected: {model_path}", C_GREEN)
    print()

    # Create args for run_pipeline
    args.model = model_path
    args.force_download = False
    args.force_convert = False
    args.force_compile = False
    args.force_inspect = False
    args.generate_only = False
    args.test = False
    args.test_only = False
    args.inspect_only = False
    args.debug = False
    args.parity = False
    args.codegen = 'v7'
    args.prompt = None

    run_pipeline(args)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="C-Kernel-Engine v7 Pipeline Runner (standalone, manifest-first)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  ./ck-v7

  # Inference
  ./ck-v7 run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf

  # Training run-dir flow
  ./ck-v7 init --run ./version/v7/runs/exp1 --init xavier_uniform
  ./ck-v7 train --run ./version/v7/runs/exp1 --data ./train.txt --train-epochs 3
  ./ck-v7 sanity --run ./version/v7/runs/exp1 --data ./train.txt --train-epochs 1
  ./ck-v7 parity --run ./version/v7/runs/exp1 --data ./train.txt --with-fd --with-replay
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    def _add_train_common_args(sp: argparse.ArgumentParser, *, include_profile: bool = True) -> None:
        sp.add_argument('--run', dest='run_dir', default=None,
                        help='Run directory for artifacts (single source of truth)')
        sp.add_argument('--data', dest='train_data', default=None,
                        help='Path to UTF-8 training text file (repeated to fill token budget)')
        sp.add_argument('--prompt', dest='train_text', default='Hello!',
                        help='Inline training text (used when --data is not set)')
        sp.add_argument('--train-mode', choices=['pretrain', 'sft'], default='pretrain',
                        help='Training mode label (pretrain/sft)')
        sp.add_argument('--pretraining', action='store_true',
                        help='Shortcut: force --train-mode=pretrain')
        sp.add_argument('--backend', choices=['ck', 'pytorch', 'torch', 'both'], default='both',
                        help='Training backend selector: ck=generated C runtime, pytorch=parity harness only, both=CK+PyTorch parity harness')
        sp.add_argument('--train-backend', choices=['ck', 'pytorch', 'torch', 'both'], default='both',
                        help='Deprecated alias for --backend (kept for compatibility)')
        sp.add_argument('--train-strict', action='store_true',
                        help='Enable strict training preflight checks before running training commands')
        sp.add_argument('--parity-on', action='store_true',
                        help='Enable scheduled oracle parity checks (metadata/config for training pipeline)')
        sp.add_argument('--oracle', choices=['pytorch'], default='pytorch',
                        help='Oracle backend used for parity checks (default: pytorch)')
        sp.add_argument('--parity-profile', choices=['debug', 'balanced', 'light'], default='balanced',
                        help='Parity cadence profile (debug/balanced/light)')
        sp.add_argument('--parity-every', type=int, default=50,
                        help='Fixed parity cadence in steps (<=0 keeps profile-driven cadence)')
        sp.add_argument('--parity-replay-on-check', action='store_true',
                        help='Replay checked CK steps from exported weight snapshots to verify one-step determinism')
        sp.add_argument('--parity-replay-tol', type=float, default=1e-7,
                        help='Allowed absolute loss delta for CK replay-on-check')
        sp.add_argument('--dump-on-drift', action='store_true',
                        help='On parity mismatch, dump drift artifacts for triage')
        sp.add_argument('--drift-topk', type=int, default=20,
                        help='Top-K tensor diffs to include in drift report')
        sp.add_argument('--analysis-checkpoints', choices=['log', 'off'], default='log',
                        help='Training analysis checkpoint cadence mode')
        sp.add_argument('--train-save-every', type=int, default=0,
                        help='Write runtime weight checkpoints every N train steps (0 disables)')
        sp.set_defaults(train_save_final=True)
        sp.add_argument('--no-train-save-final', dest='train_save_final', action='store_false',
                        help='Do not write final runtime weight checkpoint at end of training')
        sp.add_argument('--train-runtime-canary-checks', action='store_true',
                        help='Compile CK train runtime with CK_RUNTIME_CANARY_CHECKS=1 (step-level canary checks)')
        sp.add_argument('--train-runtime-bounds-assert', action='store_true',
                        help='Compile CK train runtime with CK_RUNTIME_BOUNDS_ASSERT=1 (pointer-span assertions)')
        sp.add_argument('--train-runtime-fault-op-id', type=int, default=-1,
                        help='Compile CK train runtime with fault injection at op_id (>=0 writes +1 past output)')
        sp.add_argument('--train-verify-memory', action='store_true',
                        help='Run PR3.7 memory verification suite (toggle diff, intentional +1, ASan agreement, bounds)')
        sp.add_argument('--train-verify-steps', type=int, default=4,
                        help='Number of deterministic steps used in toggle-diff verification')
        sp.add_argument('--train-verify-fault-op-id', type=int, default=-1,
                        help='Fault op_id for PR3.7 verification (default: max backward op_id)')
        sp.set_defaults(train_use_init_bump=True)
        sp.add_argument('--no-train-use-init-bump', dest='train_use_init_bump', action='store_false',
                        help='Do not load tiny parity init from run_dir/weights.bump')

        sp.add_argument('--train-epochs', type=int, default=3)
        sp.add_argument('--train-seq-len', type=int, default=16)
        sp.add_argument('--train-total-tokens', type=int, default=1024)
        sp.add_argument('--train-grad-accum', type=int, default=8)
        sp.add_argument('--train-optimizer', choices=['adamw', 'sgd'], default='adamw')
        sp.add_argument('--train-lr', type=float, default=1e-3)
        sp.add_argument('--train-seed', type=int, default=42)

        sp.add_argument('--train-vocab', type=int, default=256,
                        help='Tiny harness vocab size (default: 256)')
        sp.add_argument('--train-d-model', type=int, default=64,
                        help='Tiny harness d_model (default: 64)')
        sp.add_argument('--train-hidden', type=int, default=128,
                        help='Tiny harness hidden size (default: 128)')
        sp.add_argument('--train-loss-tol', type=float, default=2e-5,
                        help='Parity tolerance for max loss abs diff')
        sp.add_argument('--train-param-tol', type=float, default=3e-5,
                        help='Parity tolerance for max param abs diff')

        sp.add_argument('--train-json-out', default=None,
                        help='Optional JSON output path (default: run_dir/train_e2e_latest.json or v7/.cache/reports)')

        if include_profile:
            sp.add_argument('--profile-train', choices=['none', 'perf', 'vtune'], default='none',
                            help='Optional external profiler for training command')
            sp.add_argument('--train-profile-dir', default=None,
                            help='Output directory for train profiler artifacts')

    # Run command (inference pipeline + optional tiny train-e2e fast path)
    run_parser = subparsers.add_parser('run', help='Run model')
    run_parser.add_argument('model', help='Model ID, URL, GGUF file, or local path')
    run_parser.add_argument('--weight-dtype',
                           choices=['float32', 'bf16', 'q4_0', 'q4_1', 'q4_k', 'q4_k_m',
                                    'q5_0', 'q5_1', 'q6_k', 'q8_0'],
                           help='Weight dtype override (default: auto). q4_k_m uses mixed GGUF dtypes.')
    run_parser.add_argument('--context-len', type=int, default=None,
                           help='Context length for generation (default: from model config, max 32768). '
                                'All buffers (KV cache, activations, RoPE) sized accordingly.')
    run_parser.add_argument('--logits-layout', choices=['auto', 'last', 'full'], default='auto',
                           help='Logits buffer layout (auto=decode last/prefill full)')
    run_parser.add_argument('--temperature', type=float, default=0.7,
                           help='Sampling temperature (default: 0.7)')
    run_parser.add_argument('--max-tokens', type=int, default=512,
                           help='Max tokens to generate (default: 512)')
    run_parser.add_argument('--prompt', help='Single prompt (non-interactive)')
    run_parser.add_argument('--train-e2e', action='store_true',
                           help='Run tiny training parity E2E (CK vs PyTorch) and exit')
    run_parser.add_argument('--run', dest='run_dir', default=None,
                           help='Optional run directory for train-e2e artifact output')
    run_parser.add_argument('--train-data', default=None,
                           help='Training text file for --train-e2e (UTF-8)')
    run_parser.add_argument('--train-text', type=str, default=None,
                           help='Optional training text (UTF-8) for --train-e2e; falls back to --prompt')
    run_parser.add_argument('--train-mode', choices=['pretrain', 'sft'], default='pretrain')
    run_parser.add_argument('--pretraining', action='store_true')
    run_parser.add_argument('--backend', choices=['ck', 'pytorch', 'torch', 'both'], default='both',
                           help='Training backend selector for --train-e2e (ck runtime, pytorch parity harness, or both)')
    run_parser.add_argument('--train-backend', choices=['ck', 'pytorch', 'torch', 'both'], default='both',
                           help='Deprecated alias for --backend')
    run_parser.add_argument('--train-strict', action='store_true',
                           help='Enable strict training preflight checks before --train-e2e')
    run_parser.add_argument('--parity-on', action='store_true',
                           help='Enable scheduled oracle parity checks metadata for --train-e2e')
    run_parser.add_argument('--oracle', choices=['pytorch'], default='pytorch')
    run_parser.add_argument('--parity-profile', choices=['debug', 'balanced', 'light'], default='balanced')
    run_parser.add_argument('--parity-every', type=int, default=50)
    run_parser.add_argument('--parity-replay-on-check', action='store_true')
    run_parser.add_argument('--parity-replay-tol', type=float, default=1e-7)
    run_parser.add_argument('--dump-on-drift', action='store_true')
    run_parser.add_argument('--drift-topk', type=int, default=20)
    run_parser.add_argument('--analysis-checkpoints', choices=['log', 'off'], default='log')
    run_parser.add_argument('--train-runtime-canary-checks', action='store_true')
    run_parser.add_argument('--train-runtime-bounds-assert', action='store_true')
    run_parser.add_argument('--train-runtime-fault-op-id', type=int, default=-1)
    run_parser.add_argument('--train-verify-memory', action='store_true')
    run_parser.add_argument('--train-verify-steps', type=int, default=4)
    run_parser.add_argument('--train-verify-fault-op-id', type=int, default=-1)
    run_parser.set_defaults(train_use_init_bump=True)
    run_parser.add_argument('--no-train-use-init-bump', dest='train_use_init_bump', action='store_false',
                           help='Do not load tiny parity init from run_dir/weights.bump')
    run_parser.add_argument('--train-epochs', type=int, default=3,
                           help='Epochs for --train-e2e (default: 3)')
    run_parser.add_argument('--train-seq-len', type=int, default=16,
                           help='Sequence length for --train-e2e (default: 16)')
    run_parser.add_argument('--train-total-tokens', type=int, default=1024,
                           help='Total tokens for --train-e2e (default: 1024)')
    run_parser.add_argument('--train-grad-accum', type=int, default=8,
                           help='Gradient accumulation steps for --train-e2e (default: 8)')
    run_parser.add_argument('--train-optimizer', choices=['adamw', 'sgd'], default='adamw',
                           help='Optimizer for --train-e2e (default: adamw)')
    run_parser.add_argument('--train-lr', type=float, default=1e-3,
                           help='Learning rate for --train-e2e (default: 1e-3)')
    run_parser.add_argument('--train-seed', type=int, default=42,
                           help='Random seed for --train-e2e (default: 42)')
    run_parser.add_argument('--train-vocab', type=int, default=256)
    run_parser.add_argument('--train-d-model', type=int, default=64)
    run_parser.add_argument('--train-hidden', type=int, default=128)
    run_parser.add_argument('--train-loss-tol', type=float, default=2e-5)
    run_parser.add_argument('--train-param-tol', type=float, default=3e-5)
    run_parser.add_argument('--train-json-out', default=None,
                           help='Optional JSON output path for --train-e2e (default: run_dir/train_e2e_latest.json or version/v7/.cache/reports/train_e2e_latest.json)')
    run_parser.add_argument('--profile-train', choices=['none', 'perf', 'vtune'], default='none',
                           help='Optional external profiler for --train-e2e (none, perf, vtune)')
    run_parser.add_argument('--train-profile-dir', default=None,
                           help='Output directory for train profiler artifacts (default: run_dir/profile_train_latest)')
    run_parser.add_argument('--chat-template', choices=['auto', 'none', 'qwen', 'gemma'], default='auto',
                           help='Chat template mode passed to ck_chat.py (auto, none, qwen, gemma)')
    run_parser.add_argument('--no-chat-template', action='store_true',
                           help='Disable chat template formatting (same as --chat-template=none)')
    run_parser.add_argument('--python-tokenizer', action='store_true',
                           help='Force Python tokenizer in ck_chat.py (skip C tokenizer)')
    run_parser.add_argument('--force-download', action='store_true',
                           help='Re-download model files')
    run_parser.add_argument('--force-convert', action='store_true',
                           help='Re-convert weights')
    run_parser.add_argument('--force-compile', action='store_true',
                           help='Re-generate and recompile')
    run_parser.add_argument('--generate-only', action='store_true',
                           help='Generate C code only, do not run')
    run_parser.add_argument('--test', action='store_true',
                           help='Run smoke tests after build')
    run_parser.add_argument('--test-only', action='store_true',
                           help='Run tests and exit (skip chat)')
    run_parser.add_argument('--inspect-only', action='store_true',
                           help='Inspect weights and emit manifest only (v7)')
    run_parser.add_argument('--force-inspect', action='store_true',
                           help='Re-run inspect step even if cached (v7)')
    run_parser.add_argument('--debug', action='store_true',
                           help='Emit debug prints in generated C code to trace NaN/zero issues')
    run_parser.add_argument('--parity', action='store_true',
                           help='Save intermediate buffers for parity comparison with PyTorch')
    run_parser.add_argument('--codegen', choices=['v4', 'v7'], default='v7',
                           help='Codegen version: v7=explicit unrolled (default), v4=loop-based (legacy)')
    run_parser.add_argument('--int8-activations', action='store_true',
                           help='Use INT8 activation path (5-15x faster for Q5_0/Q8_0/Q4_K models)')
    run_parser.add_argument('--no-fusion', action='store_true',
                           help='Disable kernel fusion (use unfused ops for debugging)')
    run_parser.add_argument('--layout-mode', choices=['region', 'packed'], default='region',
                           help='Memory layout mode (region=weights+activations, packed=single arena)')
    run_parser.add_argument('--layer-limit', type=int, default=None,
                           help='Limit to first N layers (packed layout prototype)')
    run_parser.add_argument('--c-cli-smoke', action='store_true',
                           help='Run native v7 CLI once (true-BPE smoke test)')
    run_parser.add_argument('--c-cli-prompt', default='Hello',
                           help='Prompt for native v7 CLI smoke test (default: Hello)')
    run_parser.add_argument('--c-cli-max-tokens', type=int, default=16,
                           help='Max tokens for native v7 CLI smoke test (default: 16)')
    run_parser.add_argument('--profile', action='store_true',
                           help='Enable per-kernel timing profiling (CK_PROFILE)')
    run_parser.add_argument('--parity-dump', action='store_true',
                           help='Dump CK tensors to work_dir/ck_parity_dumps/dump.bin')
    run_parser.add_argument('--detailed-llamacpp-parity', action='store_true',
                           help='Run CK parity dump + llama.cpp parity dump for reference comparison')
    run_parser.add_argument('--llama-filter', type=str, default=None,
                           help='llama.cpp CKDMP filter (comma-separated tensor name substrings)')
    run_parser.add_argument('--llama-layer', type=int, default=None,
                           help='llama.cpp CKDMP layer filter (dump only this layer id)')
    run_parser.add_argument('--llama-stop-after', type=int, default=None,
                           help='llama.cpp CKDMP stop after N dumps')
    run_parser.add_argument('--llama-include-global', action='store_true',
                           help='Include global tensors when using --llama-layer')
    run_parser.add_argument('--llama-timeout', type=int, default=None,
                           help='llama.cpp parity timeout in seconds (default 600)')
    run_parser.add_argument('--parallel-decode', action='store_true',
                           help='[DEPRECATED] Flag accepted for compatibility only.')
    run_parser.add_argument('--reverse-test', action='store_true',
                           help='Run IR reverse validation after codegen (validates IR Lower 3 consistency)')
    run_parser.add_argument('--reverse-test-verbose', action='store_true',
                           help='Show detailed info from reverse validation')

    # Interactive command (also default when no command given)
    interactive_parser = subparsers.add_parser('interactive', aliases=['i'],
                                               help='Interactive model selector (default)')
    interactive_parser.add_argument('--weight-dtype',
                                   choices=['float32', 'bf16', 'q4_0', 'q4_1', 'q4_k', 'q4_k_m',
                                            'q5_0', 'q5_1', 'q6_k', 'q8_0'],
                                   help='Weight dtype override')
    interactive_parser.add_argument('--int8-activations', action='store_true',
                                   help='Use INT8 activation path (5-15x faster for Q5_0/Q8_0/Q4_K models)')
    interactive_parser.add_argument('--temperature', type=float, default=0.7)
    interactive_parser.add_argument('--max-tokens', type=int, default=512)

    # Init command (tiny training run bootstrap)
    init_parser = subparsers.add_parser('init', help='Initialize tiny v7 training run directory')
    init_parser.add_argument('--run', dest='output_dir', default=None,
                             help='Output run directory (default: version/v7/runs/tiny_init)')
    init_parser.add_argument('--run-name', default='tiny_init',
                             help='Run name used when --run is not set')
    init_parser.add_argument('--init', choices=['normal_0p02', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'zeros'],
                             default='normal_0p02', help='Weight initialization method')
    init_parser.add_argument('--train-mode', choices=['pretrain', 'sft'], default='pretrain')
    init_parser.add_argument('--pretraining', action='store_true')
    init_parser.add_argument('--train-seed', type=int, default=42)
    init_parser.add_argument('--layers', type=int, default=2)
    init_parser.add_argument('--vocab-size', type=int, default=256)
    init_parser.add_argument('--embed-dim', type=int, default=128)
    init_parser.add_argument('--hidden-dim', type=int, default=256)
    init_parser.add_argument('--num-heads', type=int, default=8)
    init_parser.add_argument('--num-kv-heads', type=int, default=4)
    init_parser.add_argument('--context-len', type=int, default=128)
    init_parser.add_argument('--rope-theta', type=float, default=1_000_000.0)
    init_parser.add_argument('--kernel-policy', default='fp32_reference_first')
    init_parser.add_argument('--template', default='qwen3',
                             help='Training graph template name (built-ins: qwen3, qwen2, gemma3)')
    init_parser.add_argument('--template-file', default=None,
                             help='Optional custom template JSON path (embedded into weights_manifest.json)')
    init_parser.add_argument('--generate-ir', action='store_true',
                             help='Also generate train IR artifacts (IR1 + IR2 + invariants)')
    init_parser.add_argument('--generate-runtime', action='store_true',
                             help='With --generate-ir, also emit generated_train_runtime_v7.c')
    init_parser.add_argument('--strict', action='store_true',
                             help='Strict train IR build checks when --generate-ir is used')

    # Train command (alias for train-e2e parity harness)
    train_parser = subparsers.add_parser('train-e2e', aliases=['train'],
                                        help='Run tiny training parity E2E (CK vs PyTorch)')
    _add_train_common_args(train_parser, include_profile=True)

    # Training sanity gate
    sanity_parser = subparsers.add_parser('sanity', help='Run quick v7 training sanity gate')
    _add_train_common_args(sanity_parser, include_profile=True)
    sanity_parser.add_argument('--min-loss-drop', type=float, default=0.0,
                               help='Require (first_loss - last_loss) >= threshold')

    # Training parity gate bundle
    parity_parser = subparsers.add_parser('parity', help='Run parity gate (train + optional FD/replay)')
    _add_train_common_args(parity_parser, include_profile=True)
    parity_parser.add_argument('--with-fd', action='store_true',
                               help='Also run finite-difference gradient check')
    parity_parser.add_argument('--with-replay', action='store_true',
                               help='Also run deterministic replay check')

    # Training profile command
    profile_parser = subparsers.add_parser('profile', help='Run train-e2e with profiler enabled')
    _add_train_common_args(profile_parser, include_profile=True)
    profile_parser.set_defaults(profile_train='perf')

    suite_parser = subparsers.add_parser('train-suite', aliases=['train-observe'],
                                        help='Run epoch sweep (1,3,5,10) and optional spot profiling')
    _add_train_common_args(suite_parser, include_profile=True)
    suite_parser.add_argument('--epochs-list', default='1,3,5,10',
                             help='Comma-separated epoch checkpoints to run parity/stability (default: 1,3,5,10)')
    suite_parser.add_argument('--profile-epoch', type=int, default=3,
                             help='Epoch count for spot profile run (default: 3)')

    list_parser = subparsers.add_parser('list', help='List cached models')

    clean_parser = subparsers.add_parser('clean', help='Clean cached models')
    clean_parser.add_argument('model', nargs='?', help='Model to clean (or all)')

    args = parser.parse_args()

    if args.command is None:
        args.weight_dtype = None
        args.temperature = 0.7
        args.max_tokens = 512
        run_interactive(args)
    elif args.command == 'run':
        run_pipeline(args)
    elif args.command in ('interactive', 'i'):
        run_interactive(args)
    elif args.command == 'init':
        step_run_train_init(args)
    elif args.command in ('train-e2e', 'train'):
        args.train_e2e = True
        step_run_train_e2e(args)
    elif args.command == 'sanity':
        step_run_train_sanity(args)
    elif args.command == 'parity':
        step_run_train_parity(args)
    elif args.command == 'profile':
        if not getattr(args, 'profile_train', None):
            args.profile_train = 'perf'
        step_run_train_e2e(args)
    elif args.command in ('train-suite', 'train-observe'):
        step_run_train_suite(args)
    elif args.command == 'list':
        if CACHE_DIR.exists():
            models = list(CACHE_DIR.iterdir())
            if models:
                log(f"Cached models in {CACHE_DIR}:")
                for m in sorted(models):
                    if m.is_dir():
                        size = sum(f.stat().st_size for f in m.rglob('*') if f.is_file())
                        log(f"  {m.name.replace('--', '/')} ({size / 1e6:.1f} MB)")
            else:
                log("No cached models")
        else:
            log("No cached models")
    elif args.command == 'clean':
        if args.model:
            model_dir = CACHE_DIR / args.model.replace('/', '--')
            if model_dir.exists():
                shutil.rmtree(model_dir)
                log(f"Removed {args.model}")
            else:
                log_error(f"Model not found: {args.model}")
        else:
            if CACHE_DIR.exists():
                shutil.rmtree(CACHE_DIR)
                log("Cleaned all cached models")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
