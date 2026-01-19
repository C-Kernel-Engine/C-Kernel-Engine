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
CACHE_DIR = Path.home() / ".cache/ck-engine-v6.6/models"
SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parents[1]
ROOT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BUILD_DIR = PROJECT_ROOT / "build"
HEADER_SIZE = 128

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
                      validate: bool = True,
                      tokenizer_json: Optional[Path] = None) -> tuple[Path, Path]:
    """Convert GGUF to bump format."""
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

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "convert_gguf_to_bump_v6_6.py"),
        f"--gguf={gguf_path}",
        f"--output={weights_path}",
        f"--config-out={config_path}",
        f"--manifest-out={manifest_path}",
    ]
    if tokenizer_json and tokenizer_json.exists():
        cmd.append(f"--tokenizer-json={tokenizer_json}")

    run_cmd(cmd)
    log(f"  Created {weights_path}", C_GREEN)
    return weights_path, config_path


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
                  weight_dtype: str = None, modes: list = None, force: bool = False,
                  debug: bool = False, parity: bool = False,
                  codegen_version: str = "v6",
                  int8_activations: bool = False) -> Path:
    """Build IR and generate layout JSON.

    Args:
        debug: If True, emit debug prints in generated C code to trace NaN/zero issues.
        parity: If True, save intermediate buffers for parity comparison with PyTorch.
        codegen_version: "v6" for explicit unrolled (default), "v4" for loop-based (legacy).
        int8_activations: If True, use INT8 activation path (Q5_0×Q8_0 kernels).
    """
    log_step(3, f"Building IR v6 and layout (codegen={codegen_version})")

    preferred_mode = "decode" if not modes or "decode" in modes else modes[0]
    layout_path = output_dir / f"layout_{preferred_mode}.json"

    if layout_path.exists() and not force:
        if _prefill_codegen_is_stub(output_dir):
            log("  Cached prefill stub detected; regenerating IR/codegen", C_DIM)
        else:
            log(f"  Using cached layout at {layout_path}", C_DIM)
            return layout_path

    output_dir.mkdir(parents=True, exist_ok=True)

    build_script = "build_ir_v6_6.py" if V6_MODE else "v4/build_ir_v4.py"
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / build_script),
        f"--config={config_path}",
        f"--prefix={output_dir}",
        "--emit=lib",
        "--dtype=fp32",
    ]

    if manifest_path and manifest_path.exists():
        cmd.append(f"--weights-manifest={manifest_path}")

    if weight_dtype:
        cmd.append(f"--weight-dtype={weight_dtype}")

    if modes:
        cmd.append(f"--modes={','.join(modes)}")
    else:
        cmd.append("--modes=prefill,decode")

    if debug:
        cmd.append("--debug")

    if parity:
        cmd.append("--parity")

    # INT8 activations are enabled by default in v6.6
    # The --int8-activations flag is now a no-op (always on in v6.6)

    # v6 codegen: explicit unrolled kernels (requires --fusion=off)
    if codegen_version == "v6":
        cmd.append("--codegen=v6")
        cmd.append("--fusion=off")
        log(f"  Using v6 explicit codegen (fusion disabled)", C_DIM)

    run_cmd(cmd)
    log(f"  Created {layout_path}", C_GREEN)
    return layout_path


def step_codegen(layout_path: Path, output_dir: Path, force: bool = False) -> Path:
    """Generate v6 wrapper C code that exposes the ck_model_* API."""
    log_step(4, "Generating C code")

    model_c_path = output_dir / "model.c"
    if model_c_path.exists() and not force:
        log(f"  Using cached C code at {model_c_path}", C_DIM)
        return model_c_path

    inference_header_path = output_dir / "ck-kernel-inference.h"
    inference_source_path = output_dir / "ck-kernel-inference.c"
    decode_header_path = output_dir / "ck-kernel-decode.h"
    decode_source_path = output_dir / "ck-kernel-decode.c"
    if inference_header_path.exists() and inference_source_path.exists():
        kernel_header_path = inference_header_path
        kernel_source_path = inference_source_path
    elif decode_header_path.exists() and decode_source_path.exists():
        kernel_header_path = decode_header_path
        kernel_source_path = decode_source_path
    else:
        log_error("Missing ck-kernel-inference/ck-kernel-decode C/H files. Run build_ir_v6_6.py first.")
        sys.exit(1)

    model_name = None
    try:
        with layout_path.open("r", encoding="utf-8") as f:
            layout_json = json.load(f)
        model_name = layout_json.get("model")
    except Exception:
        model_name = None

    if not model_name:
        log_error(f"Unable to resolve model name from {layout_path}")
        sys.exit(1)

    safe_name = re.sub(r"[^A-Za-z0-9]", "_", model_name).upper()
    prefix = safe_name.lower()
    kernel_header = kernel_header_path.name
    kernel_source = kernel_source_path.name

    wrapper = f"""\ 
// AUTO-GENERATED v6 wrapper: {model_name}
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "{kernel_header}"
#include "ckernel_model_load_v4.h"

#include "{kernel_source}"

static {safe_name}Model g_model;
static int g_initialized = 0;
static int g_active_tokens = 0;
static int g_kv_cache_enabled = 0;
static int g_kv_cache_capacity = {safe_name}_MAX_SEQ_LEN;
static int g_kv_cache_tokens = 0;

static int32_t *g_tokens = NULL;
static int g_tokens_cap = 0;

static float *g_logits = NULL;
static size_t g_logits_cap = 0;

static int ensure_tokens_capacity(int n) {{
    if (n <= g_tokens_cap) return 0;
    int32_t *buf = (int32_t *)realloc(g_tokens, (size_t)n * sizeof(int32_t));
    if (!buf) return -1;
    g_tokens = buf;
    g_tokens_cap = n;
    return 0;
}}

static int ensure_logits_capacity(int n) {{
    size_t needed = (size_t)n * (size_t){safe_name}_VOCAB_SIZE;
    if (needed <= g_logits_cap) return 0;
    float *buf = (float *)realloc(g_logits, needed * sizeof(float));
    if (!buf) return -1;
    g_logits = buf;
    g_logits_cap = needed;
    return 0;
}}

static const char *manifest_path_from_weights(const char *weights_path,
                                             char *out,
                                             size_t out_len) {{
    if (!weights_path || !out || out_len == 0) return NULL;
    const char *slash = strrchr(weights_path, '/');
    size_t dir_len = slash ? (size_t)(slash - weights_path + 1) : 0;
    const char *fname = "weights_manifest.map";
    size_t need = dir_len + strlen(fname) + 1;
    if (need > out_len) return NULL;
    if (dir_len) {{
        memcpy(out, weights_path, dir_len);
    }}
    strcpy(out + dir_len, fname);
    return out;
}}

int ck_model_init(const char *weights_path) {{
    if (g_initialized) return 0;
    if ({prefix}_model_allocate(&g_model) != 0) return -1;
    char manifest[4096];
    const char *manifest_path = manifest_path_from_weights(weights_path, manifest, sizeof(manifest));
    if (!manifest_path) return -2;
    if (ck_load_weights_manifest_v4(g_model.base, weights_path, manifest_path) != 0) return -3;
    {prefix}_precompute_rope(&g_model);
    g_initialized = 1;
    return 0;
}}

void ck_model_free(void) {{
    if (!g_initialized) return;
    {prefix}_model_free(&g_model);
    free(g_tokens);
    g_tokens = NULL;
    g_tokens_cap = 0;
    free(g_logits);
    g_logits = NULL;
    g_logits_cap = 0;
    g_initialized = 0;
    g_active_tokens = 0;
    g_kv_cache_tokens = 0;
}}

int ck_model_embed_tokens(const int32_t *tokens, int num_tokens) {{
    if (!g_initialized || !tokens) return -1;
    int cap = {safe_name}_MAX_SEQ_LEN;
    if (g_kv_cache_enabled && g_kv_cache_capacity > 0 && g_kv_cache_capacity < cap) {{
        cap = g_kv_cache_capacity;
    }}
    if (num_tokens > cap) num_tokens = cap;
    if (num_tokens < 1) num_tokens = 1;
    if (ensure_tokens_capacity(num_tokens) != 0) return -2;
    memcpy(g_tokens, tokens, (size_t)num_tokens * sizeof(int32_t));
    g_active_tokens = num_tokens;
    if (g_kv_cache_enabled) {{
        g_kv_cache_tokens = 0;
    }}
    return 0;
}}

int ck_model_forward(float *logits_out) {{
    if (!g_initialized) return -1;
    if (!g_tokens || g_active_tokens <= 0) return -2;
    if (ensure_logits_capacity(g_active_tokens) != 0) return -3;
    {prefix}_forward(&g_model, (const int *)g_tokens, g_active_tokens);
    float *model_logits = {safe_name}_PTR(&g_model, {safe_name}_FOOTER.logits);
    memcpy(g_logits,
           model_logits,
           (size_t)g_active_tokens * (size_t){safe_name}_VOCAB_SIZE * sizeof(float));
    if (g_kv_cache_enabled) {{
        g_kv_cache_tokens = g_active_tokens;
    }}
    if (logits_out) {{
        memcpy(logits_out, g_logits,
               (size_t)g_active_tokens * (size_t){safe_name}_VOCAB_SIZE * sizeof(float));
    }}
    return 0;
}}

int ck_model_kv_cache_enable(int capacity) {{
    if (!g_initialized) return -1;
    g_kv_cache_enabled = 1;
    int cap = capacity;
    if (cap <= 0 || cap > {safe_name}_MAX_SEQ_LEN) cap = {safe_name}_MAX_SEQ_LEN;
    g_kv_cache_capacity = cap;
    g_kv_cache_tokens = 0;
    g_active_tokens = 0;
    return 0;
}}

void ck_model_kv_cache_reset(void) {{
    g_kv_cache_tokens = 0;
    g_active_tokens = 0;
}}

int ck_model_decode(int32_t token, float *logits_out) {{
    if (!g_initialized) return -1;
    int token_index = g_kv_cache_tokens;
    if (token_index < 0 || token_index >= g_kv_cache_capacity) return -2;
    if (ensure_logits_capacity(token_index + 1) != 0) return -3;
    {prefix}_decode(&g_model, (const int *)&token, token_index);
    float *model_logits = {safe_name}_PTR(&g_model, {safe_name}_FOOTER.logits);
    memcpy(g_logits + (size_t)token_index * {safe_name}_VOCAB_SIZE,
           model_logits,
           (size_t){safe_name}_VOCAB_SIZE * sizeof(float));
    g_kv_cache_tokens = token_index + 1;
    g_active_tokens = g_kv_cache_tokens;
    if (logits_out) {{
        memcpy(logits_out,
               g_logits + (size_t)token_index * {safe_name}_VOCAB_SIZE,
               (size_t){safe_name}_VOCAB_SIZE * sizeof(float));
    }}
    return 0;
}}

float *ck_model_get_logits(void) {{
    return g_logits;
}}

int ck_model_get_vocab_size(void) {{
    return {safe_name}_VOCAB_SIZE;
}}

int ck_model_get_num_merges(void) {{
    return {safe_name}_NUM_MERGES;
}}

int ck_model_get_vocab_strings_size(void) {{
    return {safe_name}_TOTAL_VOCAB_BYTES;
}}

int ck_model_get_context_window(void) {{
    return {safe_name}_MAX_SEQ_LEN;
}}

int ck_model_get_active_tokens(void) {{
    return g_active_tokens;
}}

int ck_model_sample_argmax(void) {{
    if (!g_initialized || !g_logits) return -1;
    int vocab_size = {safe_name}_VOCAB_SIZE;
    float *logits = g_logits + (size_t)(g_active_tokens - 1) * vocab_size;
    int best_token = 0;
    float max_logit = -1e30f;
    for (int i = 0; i < vocab_size; i++) {{
        if (logits[i] > max_logit) {{
            max_logit = logits[i];
            best_token = i;
        }}
    }}
    return best_token;
}}

void *ck_model_get_vocab_offsets(void) {{
    if (!g_initialized) return NULL;
    return {safe_name}_PTR(&g_model, {safe_name}_HEADER.vocab_offsets);
}}

void *ck_model_get_vocab_strings(void) {{
    if (!g_initialized) return NULL;
    return {safe_name}_PTR(&g_model, {safe_name}_HEADER.vocab_strings);
}}

void *ck_model_get_vocab_merges(void) {{
    if (!g_initialized) return NULL;
    return {safe_name}_PTR(&g_model, {safe_name}_HEADER.vocab_merges);
}}

int ck_model_verify_canaries(void) {{
    if (!g_initialized) return -1;
    return {prefix}_verify_canaries(&g_model);
}}
"""

    model_c_path.write_text(wrapper)
    log(f"  Created {model_c_path}", C_GREEN)
    return model_c_path


def step_compile(model_c_path: Path, output_dir: Path, force: bool = False) -> Path:
    """Compile C code to shared library."""
    log_step(5, "Compiling to shared library")

    if (output_dir / "ck-kernel-inference.c").exists():
        lib_path = output_dir / "ck-kernel-inference.so"
        cmd_cache_path = output_dir / "ck-kernel-inference.build.cmd"
    else:
        lib_path = output_dir / "ck-kernel-decode.so"
        cmd_cache_path = output_dir / "ck-kernel-decode.build.cmd"

    # Find kernel sources
    kernel_list_path = model_c_path.with_suffix('.c.kernels')
    kernel_sources = []
    if kernel_list_path.exists():
        kernel_sources = kernel_list_path.read_text().strip().split('\n')

    # Default kernel sources if not specified
    if not kernel_sources:
        src_dir = PROJECT_ROOT / "src" / "kernels"
        kernel_sources = [str(f) for f in src_dir.glob("*.c")]
    extra_sources = [
        PROJECT_ROOT / "src" / "v4_legacy" / "ckernel_model_load_v4.c",
        PROJECT_ROOT / "src" / "ckernel_orchestration.c",
        PROJECT_ROOT / "src" / "ckernel_strict.c",
        PROJECT_ROOT / "src" / "cpu_features.c",
    ]
    existing = set(kernel_sources)
    for src in extra_sources:
        src_str = str(src)
        if src_str not in existing:
            kernel_sources.append(src_str)
            existing.add(src_str)

    # Build command
    cflags = ["-O3", "-march=native", "-mtune=native", "-DNDEBUG"]
    if os.environ.get("CK_V6_FAST_MATH") == "1":
        cflags += ["-ffast-math", "-funroll-loops"]
    extra_cflags = os.environ.get("CK_V6_EXTRA_CFLAGS", "").strip()
    if extra_cflags:
        cflags += extra_cflags.split()

    cmd = [
        "gcc", *cflags, "-fPIC", "-fopenmp", "-shared",
        f"-I{PROJECT_ROOT / 'include'}",
        "-o", str(lib_path),
        str(model_c_path),
    ] + kernel_sources + ["-lm"]

    cmd_str = " ".join(cmd)

    if lib_path.exists() and not force:
        try:
            lib_mtime = lib_path.stat().st_mtime
            src_mtime = max([model_c_path.stat().st_mtime] + [Path(s).stat().st_mtime for s in kernel_sources])
            cached_cmd = cmd_cache_path.read_text() if cmd_cache_path.exists() else ""
            if cached_cmd == cmd_str and src_mtime <= lib_mtime:
                log(f"  Using cached library at {lib_path}", C_DIM)
                return lib_path
        except Exception:
            pass

    log(f"  Compiling with {len(kernel_sources)} kernel sources", C_DIM)
    run_cmd(cmd)
    cmd_cache_path.write_text(cmd_str)
    log(f"  Created {lib_path}", C_GREEN)
    return lib_path


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


def step_run_chat(model_dir: Path, args: argparse.Namespace, gguf_path: Path = None):
    """Run chat interface."""
    log_step(6, "Starting chat")

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

    # Replace current process with chat
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
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace):
    """Run the full v6 pipeline."""
    model_input = args.model
    weights_path = None
    manifest_input_path = None
    gguf_path_for_tokenizer = None  # Track GGUF path for tokenizer extraction
    v6_mode = V6_MODE

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
            ensure_tokenizer_files(model_id, work_dir)
            tokenizer_json = work_dir / "tokenizer.json"
            weights_path, config_path = step_convert_gguf(
                gguf_path, work_dir,
                force=args.force_convert,
                tokenizer_json=tokenizer_json if tokenizer_json.exists() else None
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
        tokenizer_json = gguf_path.parent / "tokenizer.json"
        if not tokenizer_json.exists():
            tokenizer_json = work_dir / "tokenizer.json"
        weights_path, config_path = step_convert_gguf(
            gguf_path, work_dir,
            force=args.force_convert,
            tokenizer_json=tokenizer_json if tokenizer_json.exists() else None
        )
        manifest_path = work_dir / "weights_manifest.json"
        # Local GGUF has no repo to fetch tokenizer; user must supply it.

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
        ensure_tokenizer_files(repo_id, work_dir)
        tokenizer_json = work_dir / "tokenizer.json"
        weights_path, config_path = step_convert_gguf(
            gguf_path, work_dir,
            force=args.force_convert,
            tokenizer_json=tokenizer_json if tokenizer_json.exists() else None
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
    force_for_debug = args.force_compile or getattr(args, 'debug', False) or getattr(args, 'parity', False)
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
    layout_path = step_build_ir(
        config_path, work_dir,
        manifest_path=manifest_path or manifest_input_path,
        weight_dtype=weight_dtype,
        force=force_for_debug,
        debug=getattr(args, 'debug', False),
        parity=getattr(args, 'parity', False),
        codegen_version=getattr(args, 'codegen', 'v6'),
        int8_activations=getattr(args, 'int8_activations', False),
    )

    # Generate C code
    model_c_path = step_codegen(layout_path, work_dir, force=force_for_debug)

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
    run_parser.add_argument('--c-cli-smoke', action='store_true',
                           help='Run native v6 CLI once (true-BPE smoke test)')
    run_parser.add_argument('--c-cli-prompt', default='Hello',
                           help='Prompt for native v6 CLI smoke test (default: Hello)')
    run_parser.add_argument('--c-cli-max-tokens', type=int, default=16,
                           help='Max tokens for native v6 CLI smoke test (default: 16)')

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
