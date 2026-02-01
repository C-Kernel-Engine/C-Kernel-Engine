#!/usr/bin/env python3
"""
ck_run_v6.py - C-Kernel-Engine v6 Pipeline Runner (standalone)

Unified CLI that chains: download -> convert -> IR -> codegen -> compile -> run

v6 features:
  - Manifest-first approach (requires weights manifest)
  - Explicit unrolled codegen (per-layer, explicit kernels)
  - Mixed-quant support via per-tensor dtypes

Usage:
  python scripts/v6/ck_run_v6.py run HuggingFaceTB/SmolLM-135M
  python scripts/v6/ck_run_v6.py run ./model.gguf
  python scripts/v6/ck_run_v6.py run ./local/config.json
  python scripts/v6/ck_run_v6.py run Qwen/Qwen2-0.5B --weight-dtype=q4_k
"""

import argparse
import json
import os
import shutil
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

V6_MODE = True  # Always v6 in this standalone script

SCRIPTS_DIR = Path(__file__).parent  # version/v6.6/scripts/
V66_ROOT = SCRIPTS_DIR.parent        # version/v6.6/
PROJECT_ROOT = SCRIPTS_DIR.parents[2]  # C-Kernel-Engine/
ROOT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"  # Main scripts (for ck_chat.py etc)
BUILD_DIR = PROJECT_ROOT / "build"
HEADER_SIZE = 128
KERNEL_MAPS_DIR = V66_ROOT / "kernel_maps"
KERNEL_REGISTRY_PATH = KERNEL_MAPS_DIR / "KERNEL_REGISTRY.json"

def _get_cache_dir() -> Path:
    """Pick a writable cache dir (env override, default ~/.cache, fallback to repo)."""
    env = os.environ.get("CK_CACHE_DIR")
    if env:
        path = Path(env).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    default = Path.home() / ".cache/ck-engine-v6.6/models"
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


def load_manifest_non_fp_dtypes(manifest_path: Path) -> set[str]:
    """Return non-FP dtype set from a v4 weights manifest."""
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()

    dtypes = {str(entry.get("dtype", "")).lower() for entry in data.get("entries", [])}
    return {dt for dt in dtypes if dt and dt not in {"fp32", "f32", "bf16", "fp16"}}


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
        str(SCRIPTS_DIR / "convert_hf_to_bump_v6_6.py"),
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
        str(SCRIPTS_DIR / "convert_gguf_to_bump_v6_6.py"),
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
        str(SCRIPTS_DIR / "convert_gguf_to_bump_v6_6.py"),
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


def step_inspect_weights_v6(input_type: str, model_dir: Optional[Path], gguf_path: Optional[Path],
                            output_dir: Path, force: bool = False) -> Path:
    """Emit a lightweight weights manifest for v6 (no conversion)."""
    manifest_path = output_dir / "weights_manifest_input.json"
    if manifest_path.exists() and not force:
        log(f"  Using cached manifest at {manifest_path}", C_DIM)
        return manifest_path

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "inspect_weights_v6_6.py"),
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
                  codegen_version: str = "v6",
                  int8_activations: bool = False,
                  context_len: int = None,
                  logits_layout: str = None,
                  no_fusion: bool = False,
                  layout_mode: str = "region",
                  layer_limit: int = None,
                  profile: bool = False,
                  parallel_decode: bool = False) -> Path:
    """Build IR1: Direct template + quant → kernel IDs (v6.6 new pipeline).

    Args:
        manifest_path: Path to weights_manifest.json (required for v6.6).
        modes: Execution modes (generates IR1 for all requested modes).
        force: If True, regenerate even if cached IR1 exists.

    Returns:
        Path to primary IR1 file (decode mode).

    Note:
        v6.6 pipeline stages (only IR1 is implemented):
        - IR1: Template + Quant → Kernel IDs (current)
        - IR2: Add tensor metadata (shapes, memory layout) - TODO
        - Memory Planning: Allocate buffers, plan reuse - TODO
        - Code Generation: Generate C code that calls kernels - TODO
    """
    log_step(3, "Building IR1 (Template + Quant → Kernel IDs)")

    # Validate that manifest exists (required for v6.6)
    if not manifest_path or not manifest_path.exists():
        log_error("Manifest path required for v6.6 pipeline")
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
            str(SCRIPTS_DIR / "build_ir_v6_6.py"),
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
        # parallel_pass.py. However, codegen_v6_6.py never consumed these annotations.
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
        # commented out in build_ir_v6_6.py. Flags accepted but no longer have effect.
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
            str(SCRIPTS_DIR / "generate_memory_map_v6_6.py"),
            str(layout_path),
            "-o", str(map_path)
        ]
        run_cmd(map_cmd)
        log(f"  Created memory map at {map_path}", C_GREEN)

        ir1_paths[mode] = ir1_path

    # Return decode IR1 as primary (for compatibility)
    return ir1_paths.get("decode", ir1_paths[target_modes[0]])


def step_codegen(ir1_path: Path, output_dir: Path, force: bool = False, profile: bool = False) -> Path:
    """Generate v6.6 C code from lowered IR.

    The lowered IR contains everything needed for codegen:
    - Explicit pointer expressions for weights and activations
    - Function names for each kernel
    - Model config parameters
    """
    log_step(4, "Generating C code")

    # Check for call-ready lowered IR files
    lowered_decode = output_dir / "lowered_decode_call.json"
    lowered_prefill = output_dir / "lowered_prefill_call.json"
    model_c_path = output_dir / "model_v6_6.c"

    if not lowered_decode.exists():
        log_error(f"Lowered IR not found: {lowered_decode}")
        log_error("Run step_build_ir first to generate call-ready lowered IR")
        sys.exit(1)

    # Skip codegen if model_v6_6.c already exists and is newer than lowered IR
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
        str(SCRIPTS_DIR / "codegen_v6_6.py"),
        f"--decode={lowered_decode}",
        f"--prefill={lowered_prefill}" if lowered_prefill.exists() else "",
        f"--output={model_c_path}"
    ]
    if profile:
        cmd.append("--profile")
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

    log(f"  Source: {model_c_path}", C_DIM)
    log(f"  Lines: {sum(1 for _ in open(model_c_path))}", C_DIM)

    # Check if kernel library exists
    if not kernel_lib.exists():
        log(f"  Kernel library not found: {kernel_lib}", C_RED)
        log(f"  Run 'make' in project root to build libckernel_engine.so", C_DIM)
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
    v66_include = V66_ROOT / "include"
    v66_src = V66_ROOT / "src"
    loader_src = V66_ROOT / "src" / "ckernel_model_load_v6_6.c"

    # Detect compiler for OpenMP flag
    import shutil
    compiler = "gcc"
    omp_flag = "-fopenmp"
    if shutil.which("icx"):
        compiler = "icx"
        omp_flag = "-qopenmp"

    cmd = [
        compiler,
        "-shared", "-fPIC",
        "-mcmodel=large",  # Handle large static data in v6.6 models
        "-O3", "-march=native",
        "-std=c11",
        "-fvisibility=default",  # Export CK_EXPORT symbols
        omp_flag,  # OpenMP for parallelization
        f"-I{include_dir}",
        f"-I{v66_include}",
        f"-I{v66_src}",
        "-o", str(lib_path),
        str(model_c_path),
        str(loader_src),
        str(v66_src / "ck_parallel_decode.c"),  # Thread-pool parallel GEMV dispatch
        str(v66_src / "ck_parallel_prefill.c"),  # Thread-pool parallel GEMM dispatch (prefill)
        f"-L{BUILD_DIR}",
        f"-L{output_dir}",  # Also look in output_dir for libckernel_engine.so
        "-lckernel_engine",
        "-lckernel_tokenizer",  # BPE tokenizer library
        "-lm",
        f"-Wl,-rpath,$ORIGIN",  # Use $ORIGIN so library can find deps in same dir
        f"-Wl,-rpath,{BUILD_DIR}",
    ]

    # Add profiling define if requested
    if os.environ.get("CK_PROFILE") == "1":
        cmd.append("-DCK_PROFILE")

    log(f"  Compiling...", C_DIM)
    try:
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            log(f"  Compiled: {lib_path}", C_GREEN)
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
           f"-I{include_dir}", f"-I{v66_include}", f"-I{v66_src}",
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
    script = SCRIPTS_DIR / "ck_model_smoke_v6_6.py"
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
    if kernel_lib.exists():
        # Copy to model dir so it's always available
        dst_lib = model_dir / "libckernel_engine.so"
        if not dst_lib.exists():
            import shutil
            shutil.copy(kernel_lib, dst_lib)
            log(f"  Copied libckernel_engine.so to {model_dir}", C_DIM)

    # Ensure libckernel_tokenizer.so is findable (for BPE tokenizer)
    tokenizer_lib = BUILD_DIR / "libckernel_tokenizer.so"
    if tokenizer_lib.exists():
        dst_tok = model_dir / "libckernel_tokenizer.so"
        if not dst_tok.exists():
            import shutil
            shutil.copy(tokenizer_lib, dst_tok)
            log(f"  Copied libckernel_tokenizer.so to {model_dir}", C_DIM)

    if kernel_lib.exists():
        # Also set LD_LIBRARY_PATH for the subprocess
        ld_path = str(BUILD_DIR)
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{ld_path}:{model_dir}:{current_ld}"

    cmd = [
        sys.executable,
        str(ROOT_SCRIPTS_DIR / "ck_chat.py"),
        "--model-dir",
        str(model_dir),
    ]

    if gguf_path:
        cmd.extend(["--gguf", str(gguf_path)])
    if args.temperature:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.max_tokens:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])
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
    """Run native v6 CLI once to validate true-BPE end-to-end."""
    log(f"{C_ORANGE}[smoke]{C_RESET} Running native v6 CLI (true-BPE)")

    cli_path = PROJECT_ROOT / "build" / "ck-cli-v6"
    if not cli_path.exists():
        run_cmd(["make", "ck-cli-v6"], cwd=PROJECT_ROOT)

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
        log_error("Native v6 CLI smoke test timed out")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        log_error("Native v6 CLI smoke test failed")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)

    log(f"  {C_GREEN}Native v6 CLI smoke test: OK{C_RESET}", C_DIM)


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

    # Aggregate by op (across all layers, first decode token only)
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

    summary = {
        "total_us": total_us,
        "total_ms": total_us / 1000,
        "by_op": by_op,
        "by_layer": {str(k): v for k, v in sorted(by_layer.items())},
        "entries": entries,
    }

    summary_path = work_dir / "profile_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n{C_GREEN}Profile summary:{C_RESET}")
    log(f"  Total: {total_us / 1000:.2f} ms")
    log(f"  CSV:   {csv_path}")
    log(f"  JSON:  {summary_path}")

    # Show top 5 hotspots
    sorted_ops = sorted(by_op.items(), key=lambda x: x[1], reverse=True)
    if sorted_ops:
        log(f"  Top hotspots:")
        for op, us in sorted_ops[:5]:
            pct = us / total_us * 100 if total_us > 0 else 0
            log(f"    {op:20s} {us/1000:8.2f} ms ({pct:5.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace):
    """Run the full v6 pipeline."""
    model_input = args.model
    weights_path = None
    manifest_input_path = None
    gguf_path_for_tokenizer = None  # Track GGUF path for tokenizer extraction
    v6_mode = V6_MODE

    # Step 0: Regenerate kernel registry from kernel maps if needed
    step_regenerate_kernel_registry(force=getattr(args, 'force_compile', False))

    # Detect input type
    input_type, info = detect_input_type(model_input)
    version_tag = "v6" if v6_mode else "v4"
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
            if v6_mode:
                manifest_input_path = step_inspect_weights_v6(
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
            if v6_mode:
                manifest_input_path = step_inspect_weights_v6(
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
        if v6_mode:
            manifest_input_path = step_inspect_weights_v6(
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
        if v6_mode:
            manifest_input_path = step_inspect_weights_v6(
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
        if v6_mode and not args.weight_dtype:
            log_error("v6 requires a weights manifest for config-only runs (or pass --weight-dtype)")
            sys.exit(1)
        if v6_mode and args.inspect_only:
            log_error("inspect-only requires a weights source (GGUF or checkpoint directory)")
            sys.exit(1)

    elif input_type == 'hf_gguf':
        # Download single GGUF file from HuggingFace
        repo_id = info['repo_id']
        filename = info['filename']
        work_dir = CACHE_DIR / repo_id.replace('/', '--')

        gguf_path = step_download_gguf(repo_id, filename, CACHE_DIR, force=args.force_download)
        gguf_path_for_tokenizer = gguf_path
        if v6_mode:
            manifest_input_path = step_inspect_weights_v6(
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

    if v6_mode and not manifest_input_path and not args.weight_dtype and manifest_path is None:
        log_error("v6 requires a weights manifest (inspect-only or conversion) unless --weight-dtype is provided")
        sys.exit(1)

    # Build IR
    # If debug or parity is enabled, force recompile to ensure special code is generated
    force_for_debug = args.force_compile or getattr(args, 'debug', False) or getattr(args, 'parity', False) or getattr(args, 'profile', False)
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

    # Inject metadata into config.json for build_ir_v6
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
        codegen_version=getattr(args, 'codegen', 'v6'),
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
    model_c_path = step_codegen(layout_path, work_dir, force=force_for_debug, profile=getattr(args, 'profile', False))

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
        log(f"{C_ORANGE}[test]{C_RESET} Running v6 smoke tests")
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
    print(f"{C_ORANGE}║{C_RESET}  {C_BOLD}C-Kernel-Engine v6 - Interactive Model Selector{C_RESET}                     {C_ORANGE}║{C_RESET}")
    print(f"{C_ORANGE}╚══════════════════════════════════════════════════════════════════════╝{C_RESET}")
    print()

    models = find_available_models()

    if not models:
        print(f"  {C_DIM}No models found.{C_RESET}")
        print()
        print(f"  {C_BOLD}Options:{C_RESET}")
        print(f"    1. Download a model from HuggingFace:")
        print(f"       {C_CYAN}./ck-v6 run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf{C_RESET}")
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
    args.codegen = 'v6'
    args.prompt = None

    run_pipeline(args)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="C-Kernel-Engine v6 Pipeline Runner (standalone, manifest-first)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default - just run with no args)
  ./ck-v6
  python scripts/v6/ck_run_v6.py

  # Download GGUF directly (recommended for quantized models)
  ./ck-v6 run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf

  # Local GGUF file
  ./ck-v6 run ./model.gguf

  # Full HuggingFace model (downloads all files)
  ./ck-v6 run HuggingFaceTB/SmolLM-135M

  # Generate code only (inspect before running)
  ./ck-v6 run Qwen/Qwen2-0.5B --generate-only

  # Single prompt mode
  ./ck-v6 run ./model.gguf --prompt "What is 2+2?" --max-tokens 50
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run command
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
                           help='Inspect weights and emit manifest only (v6)')
    run_parser.add_argument('--force-inspect', action='store_true',
                           help='Re-run inspect step even if cached (v6)')
    run_parser.add_argument('--debug', action='store_true',
                           help='Emit debug prints in generated C code to trace NaN/zero issues')
    run_parser.add_argument('--parity', action='store_true',
                           help='Save intermediate buffers for parity comparison with PyTorch')
    run_parser.add_argument('--codegen', choices=['v4', 'v6'], default='v6',
                           help='Codegen version: v6=explicit unrolled (default), v4=loop-based (legacy)')
    run_parser.add_argument('--int8-activations', action='store_true',
                           help='Use INT8 activation path (5-15x faster for Q5_0/Q8_0/Q4_K models)')
    run_parser.add_argument('--no-fusion', action='store_true',
                           help='Disable kernel fusion (use unfused ops for debugging)')
    run_parser.add_argument('--layout-mode', choices=['region', 'packed'], default='region',
                           help='Memory layout mode (region=weights+activations, packed=single arena)')
    run_parser.add_argument('--layer-limit', type=int, default=None,
                           help='Limit to first N layers (packed layout prototype)')
    run_parser.add_argument('--c-cli-smoke', action='store_true',
                           help='Run native v6 CLI once (true-BPE smoke test)')
    run_parser.add_argument('--c-cli-prompt', default='Hello',
                           help='Prompt for native v6 CLI smoke test (default: Hello)')
    run_parser.add_argument('--c-cli-max-tokens', type=int, default=16,
                           help='Max tokens for native v6 CLI smoke test (default: 16)')
    run_parser.add_argument('--profile', action='store_true',
                           help='Enable per-kernel timing profiling (CK_PROFILE)')
    run_parser.add_argument('--parallel-decode', action='store_true',
                           help='[DEPRECATED] Was: OpenMP parallel GEMV for decode. '
                                'Superseded by persistent pthread thread pool '
                                '(ck_parallel_decode.h) which is always enabled. '
                                'Thread pool avoids OpenMP fork/join overhead and '
                                'core oversubscription. Flag accepted but ignored.')
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

    # List command
    list_parser = subparsers.add_parser('list', help='List cached models')

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean cached models')
    clean_parser.add_argument('model', nargs='?', help='Model to clean (or all)')

    args = parser.parse_args()

    # Default to interactive mode if no command given
    if args.command is None:
        # Set defaults for interactive mode
        args.weight_dtype = None
        args.temperature = 0.7
        args.max_tokens = 512
        run_interactive(args)
    elif args.command == 'run':
        run_pipeline(args)
    elif args.command in ('interactive', 'i'):
        run_interactive(args)
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
