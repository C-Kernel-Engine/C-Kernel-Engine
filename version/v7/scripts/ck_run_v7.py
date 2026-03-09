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
import copy
import ctypes
import hashlib
import importlib.util
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
from typing import Any, Optional, Sequence

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
V7_REQUIREMENTS_PATH = PROJECT_ROOT / "requirements-v7.txt"

def _get_cache_dir() -> Path:
    """Pick the canonical writable cache dir for all generated v7 artifacts."""
    env = os.environ.get("CK_CACHE_DIR")
    path = Path(env).expanduser() if env else (Path.home() / ".cache" / "ck-engine-v7" / "models")
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".ck_write_probe"
        with open(probe, "w") as f:
            f.write("ok")
        probe.unlink()
        return path
    except Exception as e:
        location = str(path)
        raise RuntimeError(
            "v7 cache root must be writable so all generated artifacts stay under one "
            f"operator-visible tree.\n  cache_root={location}\n"
            "Set CK_CACHE_DIR to a writable cache path if you need an override.\n"
            "Repo-local .ck_cache fallback is disabled for v7 training workflows."
        ) from e

CACHE_DIR = _get_cache_dir()


def _get_default_train_root() -> Path:
    """Canonical run root for operator-visible training artifacts."""
    return CACHE_DIR / "train"


DEFAULT_TRAIN_ROOT = _get_default_train_root()


def _cache_root_hint() -> str:
    env = os.environ.get("CK_CACHE_DIR")
    if env:
        return str(Path(env).expanduser())
    return "~/.cache/ck-engine-v7/models"


def _cache_train_root_hint() -> str:
    return f"{_cache_root_hint()}/train"


def _get_default_report_dir() -> Path:
    """Resolve writable v7 report directory under the operator-visible cache tree."""
    env = os.environ.get("CK_V7_REPORT_DIR")
    report_dir = Path(env).expanduser() if env else (CACHE_DIR / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


DEFAULT_REPORT_DIR = _get_default_report_dir()

# Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_ORANGE = "\033[38;5;214m"
C_YELLOW = C_ORANGE
C_GREEN = "\033[38;5;114m"
C_BLUE = "\033[38;5;75m"
C_RED = "\033[38;5;203m"
C_CYAN = "\033[38;5;87m"

TEMPLATE_AUDIT_TOKENIZER_REF_ALLOWLIST = {
    "vocab_offsets",
    "vocab_strings",
    "vocab_scores",
    "vocab_types",
    "vocab_merges",
}


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


def _parse_v7_requirement_packages() -> list[str]:
    packages: list[str] = []
    if not V7_REQUIREMENTS_PATH.exists():
        return packages
    for raw in V7_REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line.startswith("-e ") or "://" in line:
            continue
        pkg = re.split(r"[<>=!~\[\]\s]", line, maxsplit=1)[0].strip()
        if pkg:
            packages.append(pkg)
    return packages


def _missing_v7_python_packages() -> list[str]:
    missing: list[str] = []
    for pkg in _parse_v7_requirement_packages():
        module_name = pkg.replace("-", "_")
        if importlib.util.find_spec(module_name) is None:
            missing.append(pkg)
    return missing


def _ensure_v7_python_requirements(command: Optional[str]) -> None:
    commands_requiring_v7_env = {
        "run",
        "init",
        "train-e2e",
        "train",
        "sanity",
        "parity",
        "profile",
        "train-suite",
        "train-observe",
        "template-audit",
        "v7-template-audit",
    }
    if command not in commands_requiring_v7_env:
        return

    missing = _missing_v7_python_packages()
    if not missing:
        return

    reqs = " ".join(_parse_v7_requirement_packages())
    log_error(
        f"Missing v7 Python dependencies for '{command}': {', '.join(missing)}"
    )
    print(f"Required Python packages: {reqs}", file=sys.stderr)
    print("Supported bootstrap:", file=sys.stderr)
    print("  make v7-init", file=sys.stderr)
    print("  make v7-doctor", file=sys.stderr)
    print("Manual environment (pip example):", file=sys.stderr)
    print("  python3 -m venv .venv", file=sys.stderr)
    print("  . .venv/bin/activate", file=sys.stderr)
    print(f"  python -m pip install -r {V7_REQUIREMENTS_PATH.name}", file=sys.stderr)
    print(
        "If you prefer uv/conda, install the same package set into the interpreter you plan to use.",
        file=sys.stderr,
    )
    sys.exit(2)


def _classify_v7_failure(
    command: Optional[str], exc: Optional[BaseException]
) -> tuple[str, list[str]]:
    if exc is None:
        return ("v7 command failed", ["make v7-doctor"])

    message = str(exc)
    cmd_text = ""
    if isinstance(exc, subprocess.CalledProcessError):
        if isinstance(exc.cmd, (list, tuple)):
            cmd_text = " ".join(str(part) for part in exc.cmd)
        elif exc.cmd is not None:
            cmd_text = str(exc.cmd)
    haystack = f"{message} {cmd_text}".lower()

    if isinstance(exc, FileNotFoundError):
        if any(tok in haystack for tok in (".gguf", "config.json", "weights", "token file", "run dir")):
            return (
                "model, run directory, or training input path is missing",
                [
                    "Check the model/run path you passed in.",
                    "make v7-doctor",
                ],
            )
        if any(tok in haystack for tok in ("git", "make", "python3", "gcc", "clang", "icx", "perf", "valgrind", "vtune", "advisor")):
            return (
                "a required host tool is missing from this machine",
                [
                    "make v7-doctor",
                    "Install the missing host tool, then retry.",
                ],
            )
        return (
            "a required file or tool was not found",
            [
                "make v7-doctor",
                "Check the path printed above and retry.",
            ],
        )

    if isinstance(exc, subprocess.CalledProcessError):
        if any(tok in haystack for tok in ("profile-v7", "perf ", "flamegraph", "cachegrind", "vtune", "advisor")):
            return (
                "profiling tool, permissions, or host profiling setup is incomplete",
                [
                    "make v7-doctor",
                    "make v7-capture-artifacts V7_MODEL=<model>  # skip profiling and verify runtime first",
                ],
            )
        if any(tok in haystack for tok in ("convert_gguf", "huggingface", "hf://", ".gguf")):
            return (
                "model download or GGUF conversion failed",
                [
                    "version/v7/scripts/cks-v7-run run <hf://...|model.gguf> [...]",
                    "make v7-doctor",
                ],
            )
        if any(tok in haystack for tok in ("gcc", "clang", "icx", "make ", "codegen", "build_ir", "compile")):
            return (
                "compile, codegen, or host toolchain step failed",
                [
                    "make v7-doctor",
                    "make v7-demo-runtime V7_MODEL=hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
                ],
            )
        return (
            "a subprocess inside the v7 pipeline failed",
            [
                "make v7-doctor",
                "Retry with version/v7/scripts/cks-v7-run ... if this is a first run.",
            ],
        )

    if any(tok in haystack for tok in ("hfvalidationerror", "repo id must be in the form", "hf://")):
        return (
            "model identifier is invalid or the local GGUF path does not exist",
            [
                "Use hf://<repo>/<file>.gguf for Hugging Face models.",
                "Use an existing local .gguf path for local runs, then retry.",
            ],
        )

    if command in {"train-e2e", "train", "sanity", "parity", "profile", "train-suite", "train-observe", "init"}:
        return (
            "training setup, staged data, or train runtime configuration is inconsistent",
            [
                "make v7-doctor",
                "Review the runbook bootstrap section, then retry.",
            ],
        )

    return (
        f"unexpected {type(exc).__name__} while running v7",
        [
            "make v7-doctor",
            "Retry with version/v7/scripts/cks-v7-run ... for guided bootstrap.",
        ],
    )


def _print_v7_next_steps(command: Optional[str], exc: Optional[BaseException] = None) -> None:
    likely_cause, next_commands = _classify_v7_failure(command, exc)
    print("\nLikely cause:", file=sys.stderr)
    print(f"  {likely_cause}", file=sys.stderr)
    print("Next command:", file=sys.stderr)
    for item in next_commands:
        print(f"  {item}", file=sys.stderr)
    print("Fallbacks:", file=sys.stderr)
    print("  make v7-init", file=sys.stderr)
    print("  make v7-doctor", file=sys.stderr)
    if command == "run":
        print("  version/v7/scripts/cks-v7-run run <model-or-run-dir> [...]", file=sys.stderr)


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
        raise


def run_cmd_allow_fail(cmd: list, cwd: Path = None) -> subprocess.CompletedProcess:
    """Run command without exiting on non-zero status."""
    return subprocess.run(cmd, cwd=cwd)


def _cc_supports_flags(cc: str, flags: Sequence[str]) -> bool:
    """Return True when compiler accepts all flags."""
    if not flags:
        return True
    try:
        probe = subprocess.run(
            [cc, "-Werror", "-x", "c", "-fsyntax-only", *list(flags), "-"],
            input="int main(void) { return 0; }\n",
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return int(probe.returncode) == 0
    except Exception:
        return False


def _bitwise_parity_compile_flags(cc: str) -> list[str]:
    """
    Compiler flags for tighter FP reproducibility in generated train runtime.
    We probe support per-flag to stay portable across gcc/clang variants.
    """
    candidates = [
        "-O1",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fno-unsafe-math-optimizations",
        "-fno-associative-math",
        "-fno-reciprocal-math",
        "-fno-finite-math-only",
        "-frounding-math",
        "-fexcess-precision=standard",
        "-fno-tree-vectorize",
    ]
    selected: list[str] = []
    for f in candidates:
        if _cc_supports_flags(cc, [f]):
            selected.append(f)
    return selected


def _path_to_make_target(path: Path) -> str:
    """
    Convert an absolute filesystem path to a Make target string.

    Make targets in this repo are declared relative to PROJECT_ROOT (e.g. build/libx.so),
    so passing absolute paths can produce "No rule to make target ..." failures.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


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
    """Best-effort physical core count; prefer physical cores on HT systems."""
    try:
        logical = len(os.sched_getaffinity(0))
    except Exception:
        logical = os.cpu_count() or 1
    logical = max(1, int(logical or 1))

    physical = 0
    threads_per_core = 0

    # 1) Linux /proc/cpuinfo (best source when available)
    try:
        pairs = set()
        sockets = set()
        phys_id = None
        core_id = None
        cpu_cores_hint = 0
        siblings_hint = 0
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
                    sockets.add(phys_id)
                elif line.startswith("core id"):
                    core_id = int(line.split(":", 1)[1].strip())
                elif line.startswith("cpu cores"):
                    cpu_cores_hint = max(cpu_cores_hint, int(line.split(":", 1)[1].strip()))
                elif line.startswith("siblings"):
                    siblings_hint = max(siblings_hint, int(line.split(":", 1)[1].strip()))
        if phys_id is not None and core_id is not None:
            pairs.add((phys_id, core_id))

        physical = len(pairs)
        if siblings_hint > 0 and cpu_cores_hint > 0 and siblings_hint >= cpu_cores_hint:
            threads_per_core = max(1, siblings_hint // cpu_cores_hint)

        if physical <= 1 and cpu_cores_hint > 0:
            if sockets:
                physical = cpu_cores_hint * max(1, len(sockets))
            elif threads_per_core > 1:
                physical = max(1, logical // threads_per_core)
    except Exception:
        physical = 0

    # 2) lscpu fallback (works on many container/VM setups)
    if physical <= 1:
        try:
            out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
            kv = {}
            for raw in out.splitlines():
                if ":" not in raw:
                    continue
                k, v = raw.split(":", 1)
                kv[k.strip().lower()] = v.strip()

            sockets = int(kv.get("socket(s)", "0") or 0)
            cores_per_socket = int(kv.get("core(s) per socket", "0") or 0)
            tpc = int(kv.get("thread(s) per core", "0") or 0)
            if tpc > 0 and threads_per_core <= 0:
                threads_per_core = tpc

            if sockets > 0 and cores_per_socket > 0:
                physical = max(1, sockets * cores_per_socket)
            elif tpc > 1:
                physical = max(1, logical // tpc)
        except Exception:
            pass

    if physical > 1:
        return min(int(physical), int(logical))
    if threads_per_core > 1:
        return max(1, int(logical) // int(threads_per_core))
    return int(logical)


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


def _as_positive_int(value: object) -> Optional[int]:
    """Best-effort integer parse that returns None for non-positive/invalid values."""
    try:
        iv = int(value)
    except Exception:
        return None
    return iv if iv > 0 else None


def _as_finite_float(value: object) -> Optional[float]:
    """Best-effort float parse that returns None for invalid/non-finite values."""
    try:
        fv = float(value)
    except Exception:
        return None
    return fv if math.isfinite(fv) else None


def _load_train_adamw_from_run_manifest(run_dir: Optional[Path]) -> Optional[dict]:
    """Extract AdamW defaults from run_dir/weights_manifest.json when available."""
    if run_dir is None:
        return None
    manifest_path = run_dir / "weights_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    cfg = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    training_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    optimizer_cfg = (
        training_cfg.get("optimizer")
        if isinstance(training_cfg.get("optimizer"), dict)
        else {}
    )
    adamw_cfg = {}
    if isinstance(optimizer_cfg.get("adamw"), dict):
        adamw_cfg = optimizer_cfg.get("adamw") or {}
    elif isinstance(training_cfg.get("adamw"), dict):
        adamw_cfg = training_cfg.get("adamw") or {}
    else:
        adamw_cfg = optimizer_cfg

    out: dict[str, object] = {"manifest": str(manifest_path)}
    beta1 = _as_finite_float(adamw_cfg.get("beta1"))
    beta2 = _as_finite_float(adamw_cfg.get("beta2"))
    eps = _as_finite_float(adamw_cfg.get("eps"))
    weight_decay = _as_finite_float(adamw_cfg.get("weight_decay"))

    if beta1 is not None:
        out["beta1"] = float(beta1)
    if beta2 is not None:
        out["beta2"] = float(beta2)
    if eps is not None:
        out["eps"] = float(eps)
    if weight_decay is not None:
        out["weight_decay"] = float(weight_decay)

    if len(out) <= 1:
        return None
    return out


def _validate_adamw_hparams(beta1: float, beta2: float, eps: float, weight_decay: float) -> None:
    """Validate AdamW hyperparameters used by generated runtime and parity harness."""
    if not (0.0 <= float(beta1) < 1.0):
        raise ValueError(f"--train-adamw-beta1 must be in [0, 1): got {beta1}")
    if not (0.0 <= float(beta2) < 1.0):
        raise ValueError(f"--train-adamw-beta2 must be in [0, 1): got {beta2}")
    if not (float(eps) > 0.0):
        raise ValueError(f"--train-adamw-eps must be > 0: got {eps}")
    if not (float(weight_decay) >= 0.0):
        raise ValueError(f"--train-adamw-weight-decay must be >= 0: got {weight_decay}")


def _resolve_train_adamw_hparams(args: argparse.Namespace, run_dir: Optional[Path]) -> dict:
    """Resolve AdamW hparams with precedence: CLI override -> run manifest -> defaults."""
    defaults = {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.01,
    }
    requested_raw: dict[str, Optional[float]] = {}
    cli_arg_map = {
        "beta1": "train_adamw_beta1",
        "beta2": "train_adamw_beta2",
        "eps": "train_adamw_eps",
        "weight_decay": "train_adamw_weight_decay",
    }
    for key, attr in cli_arg_map.items():
        raw = getattr(args, attr, None)
        if raw is None:
            requested_raw[key] = None
            continue
        fv = _as_finite_float(raw)
        if fv is None:
            raise ValueError(f"--{attr.replace('_', '-')} must be finite")
        requested_raw[key] = float(fv)
    effective = dict(defaults)
    source = "defaults"
    manifest_path = None

    manifest_cfg = _load_train_adamw_from_run_manifest(run_dir)
    if isinstance(manifest_cfg, dict):
        for key in ("beta1", "beta2", "eps", "weight_decay"):
            fv = _as_finite_float(manifest_cfg.get(key))
            if fv is not None:
                effective[key] = float(fv)
        manifest_path = manifest_cfg.get("manifest")
        source = "run_manifest"

    for key in ("beta1", "beta2", "eps", "weight_decay"):
        rv = requested_raw.get(key)
        if rv is not None:
            effective[key] = float(rv)
            source = "cli"

    _validate_adamw_hparams(
        beta1=float(effective["beta1"]),
        beta2=float(effective["beta2"]),
        eps=float(effective["eps"]),
        weight_decay=float(effective["weight_decay"]),
    )

    return {
        "requested": requested_raw,
        "effective": effective,
        "source": source,
        "manifest": manifest_path,
    }


def _load_train_dims_from_run_manifest(run_dir: Optional[Path]) -> Optional[dict]:
    """Extract training dims from run_dir/weights_manifest.json when available."""
    if run_dir is None:
        return None
    manifest_path = run_dir / "weights_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    cfg = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    training_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    tiny_cfg = (
        training_cfg.get("tiny_parity")
        if isinstance(training_cfg.get("tiny_parity"), dict)
        else {}
    )

    dims: dict[str, object] = {"manifest": str(manifest_path)}
    vocab = _as_positive_int(cfg.get("vocab_size")) or _as_positive_int(tiny_cfg.get("vocab"))
    d_model = (
        _as_positive_int(cfg.get("embed_dim"))
        or _as_positive_int(cfg.get("hidden_size"))
        or _as_positive_int(tiny_cfg.get("d_model"))
    )
    hidden = (
        _as_positive_int(cfg.get("intermediate_size"))
        or _as_positive_int(tiny_cfg.get("hidden"))
    )
    num_layers = (
        _as_positive_int(cfg.get("num_layers"))
        or _as_positive_int(cfg.get("num_hidden_layers"))
        or _as_positive_int(tiny_cfg.get("num_layers"))
    )

    if vocab is not None:
        dims["vocab"] = int(vocab)
    if d_model is not None:
        dims["d_model"] = int(d_model)
    if hidden is not None:
        dims["hidden"] = int(hidden)
    if num_layers is not None:
        dims["num_layers"] = int(num_layers)

    if len(dims) <= 1:
        return None
    return dims


def _resolve_train_dims_for_run(args: argparse.Namespace, run_dir: Optional[Path]) -> dict:
    """Resolve requested vs effective train dims, preferring run-dir manifest values."""
    requested = {
        "vocab": int(getattr(args, "train_vocab", 256) or 256),
        "d_model": int(getattr(args, "train_d_model", 64) or 64),
        "hidden": int(getattr(args, "train_hidden", 128) or 128),
        "num_layers": int(getattr(args, "num_layers", 1) or 1),
    }
    effective = dict(requested)
    source = "cli"
    manifest_path = None

    manifest_dims = _load_train_dims_from_run_manifest(run_dir)
    if isinstance(manifest_dims, dict):
        for key in ("vocab", "d_model", "hidden", "num_layers"):
            val = _as_positive_int(manifest_dims.get(key))
            if val is not None:
                effective[key] = int(val)
        manifest_path = manifest_dims.get("manifest")
        source = "run_manifest"

    mismatches = {
        key: {"requested": int(requested[key]), "effective": int(effective[key])}
        for key in ("vocab", "d_model", "hidden", "num_layers")
        if int(requested[key]) != int(effective[key])
    }

    return {
        "requested": requested,
        "effective": effective,
        "source": source,
        "manifest": manifest_path,
        "mismatches": mismatches,
    }


def _is_ck_runtime_dir(path: Path) -> bool:
    """Detect local dirs that already contain runnable CK artifacts."""
    return bool((path / "weights.bump").exists() and (path / "weights_manifest.json").exists())


def _load_builtin_template_doc(template_name: Optional[str]) -> Optional[dict]:
    name = str(template_name or "").strip().lower()
    if not name:
        return None
    path = V7_ROOT / "templates" / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _merge_template_defaults(
    default_doc: dict[str, Any],
    override_doc: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Merge stale embedded templates onto the latest built-in defaults."""
    merged = copy.deepcopy(default_doc)
    if not isinstance(override_doc, dict):
        return merged
    for key, value in override_doc.items():
        if value is None:
            continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_template_defaults(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _hydrate_manifest_template(
    template_doc: Optional[dict[str, Any]],
    cfg: dict[str, Any],
) -> Optional[dict[str, Any]]:
    template_name = ""
    if isinstance(template_doc, dict):
        template_name = str(template_doc.get("name", "") or "").strip().lower()
    if not template_name:
        template_name = str(cfg.get("model", "") or "").strip().lower()
    built_in = _load_builtin_template_doc(template_name)
    if built_in and isinstance(template_doc, dict):
        return _merge_template_defaults(built_in, template_doc)
    if built_in:
        return copy.deepcopy(built_in)
    if isinstance(template_doc, dict):
        return copy.deepcopy(template_doc)
    return None


def _manifest_entry_offset(entry: dict) -> int:
    try:
        return int(entry.get("file_offset", entry.get("offset", 0)) or 0)
    except Exception:
        return 0


def _manifest_entry_size(entry: dict) -> int:
    try:
        return int(entry.get("size", entry.get("size_bytes", 0)) or 0)
    except Exception:
        return 0


def _manifest_entry_name_set(entries: Sequence[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in entries:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "") or "").strip()
        if name:
            out.add(name)
    return out


def _desired_template_lm_head_mode(cfg: dict[str, Any], entry_names: set[str]) -> str:
    tie_cfg = _coerce_bool(cfg.get("tie_word_embeddings"))
    has_untied_head = any(
        name in entry_names for name in ("output.weight", "lm_head.weight", "lm_head_weight", "lm_head")
    )
    if tie_cfg is False:
        return "output_weight"
    if tie_cfg is None and has_untied_head:
        return "output_weight"
    return "weight_tying"


def _normalize_manifest_template_contract(
    template_doc: dict[str, Any],
    cfg: dict[str, Any],
    entry_names: set[str],
) -> dict[str, Any]:
    patched = copy.deepcopy(template_doc)
    contract = patched.setdefault("contract", {})
    logits_contract = contract.setdefault("logits_contract", {})
    logits_contract["lm_head"] = _desired_template_lm_head_mode(cfg, entry_names)
    return patched


def _normalize_manifest_for_inference(src_manifest: dict) -> dict:
    """
    Normalize train/runtime manifests into build_ir_v7-compatible shape.

    - Adds file_offset fallback from offset
    - Drops tiny parity-only tensors from inference manifest
    - Ensures template and quant_summary are present
    """
    out = dict(src_manifest or {})
    cfg = out.get("config") if isinstance(out.get("config"), dict) else {}
    cfg = dict(cfg)
    # Normalize config aliases expected by build_ir/codegen.
    context_len = cfg.get("context_length", cfg.get("context_len", cfg.get("max_seq_len")))
    if context_len is not None:
        try:
            context_len_i = int(context_len)
            cfg["context_length"] = context_len_i
            cfg.setdefault("context_len", context_len_i)
            cfg.setdefault("max_seq_len", context_len_i)
        except Exception:
            pass
    if "rms_eps" not in cfg:
        eps = cfg.get("rms_norm_eps", cfg.get("layer_norm_eps", 1e-5))
        try:
            eps_f = float(eps)
            cfg["rms_eps"] = eps_f
            cfg.setdefault("rms_norm_eps", eps_f)
        except Exception:
            cfg["rms_eps"] = 1e-5
            cfg.setdefault("rms_norm_eps", 1e-5)
    if "intermediate_size" not in cfg and cfg.get("hidden_size") is not None:
        cfg["intermediate_size"] = int(cfg["hidden_size"])
    out["config"] = cfg
    entries_in = out.get("entries") if isinstance(out.get("entries"), list) else []

    tiny_state_names = set()
    training_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    tiny_cfg = training_cfg.get("tiny_parity") if isinstance(training_cfg.get("tiny_parity"), dict) else {}
    state_tensors = tiny_cfg.get("state_tensors") if isinstance(tiny_cfg.get("state_tensors"), dict) else {}
    for v in state_tensors.values():
        if isinstance(v, str) and v:
            tiny_state_names.add(v)

    entries_out: list[dict] = []
    for row in entries_in:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "") or "")
        if not name:
            continue
        # tiny.* tensors are parity harness state, not inference model weights.
        if name.startswith("tiny.") or name in tiny_state_names:
            continue

        e = dict(row)
        e["file_offset"] = _manifest_entry_offset(e)
        if "offset" not in e:
            e["offset"] = int(e["file_offset"])
        e["size"] = _manifest_entry_size(e)
        entries_out.append(e)
    out["entries"] = entries_out

    entry_names = _manifest_entry_name_set(entries_out)

    template = _hydrate_manifest_template(
        out.get("template") if isinstance(out.get("template"), dict) else None,
        cfg,
    )
    if isinstance(template, dict):
        out["template"] = template
    if isinstance(template, dict):
        out["template"] = _normalize_manifest_template_contract(template, cfg, entry_names)

    quant_summary = out.get("quant_summary")
    if not isinstance(quant_summary, dict) or not quant_summary:
        entry_dtype: dict[str, str] = {}
        for e in entries_out:
            name = str(e.get("name", "") or "")
            if not name:
                continue
            entry_dtype[name] = str(e.get("dtype", "fp32") or "fp32").lower()

        inferred: dict[str, object] = {}
        tok_dtype = entry_dtype.get("token_emb")
        if tok_dtype:
            inferred["token_emb"] = tok_dtype
        lm_dtype = (
            entry_dtype.get("lm_head.weight")
            or entry_dtype.get("output.weight")
            or entry_dtype.get("lm_head")
            or tok_dtype
        )
        if lm_dtype:
            inferred["lm_head"] = lm_dtype

        num_layers = int(cfg.get("num_layers", 0) or 0) if isinstance(cfg, dict) else 0
        layer_keys = ("wq", "wk", "wv", "wo", "w1", "w2", "w3")
        for layer_idx in range(max(0, num_layers)):
            layer_q: dict[str, str] = {}
            for key in layer_keys:
                dt = entry_dtype.get(f"layer.{layer_idx}.{key}")
                if dt:
                    layer_q[key] = dt
            if layer_q:
                inferred[f"layer.{layer_idx}"] = layer_q

        out["quant_summary"] = inferred

    return out


def _normalize_manifest_file_for_inference(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    raw_doc = _load_json_dict(manifest_path)
    if not isinstance(raw_doc, dict) or not raw_doc:
        return False
    normalized = _normalize_manifest_for_inference(raw_doc)
    if _canonical_json_bytes(raw_doc) == _canonical_json_bytes(normalized):
        return False
    manifest_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return True


def _link_or_copy_file(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _prepare_runtime_dir_from_local_ck_artifacts(model_dir: Path, work_dir: Path) -> tuple[Path, Path, Path]:
    """Materialize build inputs in work_dir from a pre-existing local CK runtime dir."""
    src_manifest = model_dir / "weights_manifest.json"
    src_bump = model_dir / "weights.bump"
    src_config = model_dir / "config.json"
    if not src_manifest.exists() or not src_bump.exists():
        raise RuntimeError(f"Missing local runtime artifacts in {model_dir}")

    work_dir.mkdir(parents=True, exist_ok=True)

    dst_bump = work_dir / "weights.bump"
    bump_shift = 0
    with src_bump.open("rb") as f:
        magic = f.read(8)
    if magic in (b"BUMPWGT4", b"BUMPWGT5"):
        _link_or_copy_file(src_bump, dst_bump)
    else:
        # Tiny train run artifacts are raw contiguous fp32 blobs.
        # Wrap them with a BUMPWGT4 header so inference loader accepts them.
        if dst_bump.exists():
            dst_bump.unlink()
        with src_bump.open("rb") as fin, dst_bump.open("wb") as fout:
            fout.write(b"BUMPWGT4")
            if HEADER_SIZE > 8:
                fout.write(b"\x00" * (HEADER_SIZE - 8))
            shutil.copyfileobj(fin, fout)
        bump_shift = HEADER_SIZE

    manifest_doc = json.loads(src_manifest.read_text(encoding="utf-8"))
    normalized = _normalize_manifest_for_inference(manifest_doc)
    if bump_shift:
        for row in normalized.get("entries", []):
            if not isinstance(row, dict):
                continue
            try:
                base_off = int(row.get("file_offset", row.get("offset", 0)) or 0)
            except Exception:
                base_off = 0
            shifted = base_off + int(bump_shift)
            row["file_offset"] = shifted
            row["offset"] = shifted
    dst_manifest = work_dir / "weights_manifest.json"
    dst_manifest.write_text(json.dumps(normalized, indent=2), encoding="utf-8")

    dst_config = work_dir / "config.json"
    if src_config.exists():
        shutil.copy2(src_config, dst_config)
    else:
        cfg = normalized.get("config") if isinstance(normalized.get("config"), dict) else {}
        dst_config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    src_tok = model_dir / "tokenizer.json"
    if not src_tok.exists():
        pipe_dir = model_dir / ".ck_pipeline"
        if pipe_dir.exists():
            candidates = []
            for p in pipe_dir.glob("*/tokenizer.json"):
                if p.is_file():
                    candidates.append(p)
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                src_tok = candidates[0]
    if src_tok.exists():
        dst_tok = work_dir / "tokenizer.json"
        if not dst_tok.exists():
            shutil.copy2(src_tok, dst_tok)

    # Copy CK true_bpe binary artifacts when available so Python chat fallback
    # can use the exact same tokenizer path as training.
    bpe_candidates: list[Path] = []
    for p in (model_dir / "tokenizer_bin", model_dir / "bpe_bin"):
        if p.is_dir():
            bpe_candidates.append(p)
    pipe_dir = model_dir / ".ck_pipeline"
    if pipe_dir.exists():
        for patt in ("*/tokenizer_bin", "*/bpe_bin"):
            for p in pipe_dir.glob(patt):
                if p.is_dir():
                    bpe_candidates.append(p)
    if bpe_candidates:
        bpe_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        src_bin = bpe_candidates[0]
        dst_bin = work_dir / "tokenizer_bin"
        dst_bin.mkdir(parents=True, exist_ok=True)
        for name in ("tokenizer_meta.json", "vocab_offsets.bin", "vocab_strings.bin", "vocab_merges.bin"):
            src_file = src_bin / name
            if src_file.exists():
                shutil.copy2(src_file, dst_bin / name)

    return dst_bump, dst_config, dst_manifest


def _find_local_gguf(model_dir: Path) -> Optional[Path]:
    """Return a deterministic GGUF candidate from a local model directory."""
    candidates: list[Path] = []
    for patt in ("*.gguf", "*/*.gguf"):
        for p in model_dir.glob(patt):
            if p.is_file():
                candidates.append(p.resolve())
    if not candidates:
        return None
    candidates = sorted(set(candidates))
    return candidates[0]


def _manifest_lm_head_state(manifest_path: Path) -> tuple[Optional[bool], bool]:
    """Extract (tie_word_embeddings, has_untied_lm_head_entry) from manifest."""
    doc = _load_json_dict(manifest_path)
    cfg = doc.get("config") if isinstance(doc.get("config"), dict) else {}
    tie_cfg_raw = cfg.get("tie_word_embeddings")
    tie_cfg: Optional[bool]
    if isinstance(tie_cfg_raw, bool):
        tie_cfg = tie_cfg_raw
    elif isinstance(tie_cfg_raw, (int, float)):
        tie_cfg = bool(tie_cfg_raw)
    else:
        tie_cfg = None

    entries = doc.get("entries") if isinstance(doc.get("entries"), list) else []
    names = {str(e.get("name", "")) for e in entries if isinstance(e, dict)}
    has_untied_lm_head = any(
        n in names for n in ("output.weight", "lm_head.weight", "lm_head_weight", "lm_head")
    )
    return tie_cfg, has_untied_lm_head


def _inspect_gguf_tie_word_embeddings(gguf_path: Path) -> Optional[bool]:
    """Best-effort GGUF tie-word-embeddings probe using inspect_weights_v7."""
    try:
        from inspect_weights_v7 import inspect_gguf
        cfg, _ = inspect_gguf(gguf_path, max_layers=1)
        tie = cfg.get("tie_word_embeddings")
        if isinstance(tie, bool):
            return tie
        if isinstance(tie, (int, float)):
            return bool(tie)
    except Exception:
        return None
    return None


def _gguf_manifest_lm_head_contract_ok(
    gguf_path: Path,
    manifest_path: Path,
    *,
    verbose: bool = True,
) -> bool:
    """
    Validate untied LM-head contract between source GGUF and converted manifest.

    Contract:
    - If GGUF is untied (tie_word_embeddings=false), manifest must include output/lm_head weight.
    - If manifest config declares untied, it must include output/lm_head weight.
    """
    if not manifest_path.exists():
        if verbose:
            log(f"  Missing manifest for LM-head contract check: {manifest_path}", C_ORANGE)
        return False

    tie_cfg, has_untied_lm_head = _manifest_lm_head_state(manifest_path)
    if tie_cfg is False and not has_untied_lm_head:
        if verbose:
            log_error(
                f"Manifest contract failed: tie_word_embeddings=false but output/lm_head weight is missing ({manifest_path})"
            )
        return False

    tie_src = _inspect_gguf_tie_word_embeddings(gguf_path)
    if tie_src is False and not has_untied_lm_head:
        if verbose:
            log_error(
                f"GGUF contract failed: source GGUF is untied but manifest has no output/lm_head weight ({manifest_path})"
            )
        return False
    if tie_src is False and tie_cfg is True:
        if verbose:
            log_error(
                f"GGUF contract failed: source GGUF untied but manifest config says tie_word_embeddings=true ({manifest_path})"
            )
        return False
    return True


def _template_manifest_semantic_check(manifest_path: Path) -> tuple[bool, dict[str, Any]]:
    """
    Fail-fast semantic checks between manifest weights/config and template contract.

    This catches silent mismatches that otherwise show up later as gibberish output.
    """
    manifest_doc = _normalize_manifest_for_inference(_load_json_dict(manifest_path) or {})
    cfg = manifest_doc.get("config") if isinstance(manifest_doc.get("config"), dict) else {}
    template_doc = manifest_doc.get("template") if isinstance(manifest_doc.get("template"), dict) else {}
    contract_doc = template_doc.get("contract") if isinstance(template_doc.get("contract"), dict) else {}

    seq = template_doc.get("sequence") if isinstance(template_doc.get("sequence"), list) else []
    block_types = template_doc.get("block_types") if isinstance(template_doc.get("block_types"), dict) else {}
    block_name = str(seq[0]) if seq else ""
    block_doc = block_types.get(block_name) if isinstance(block_types.get(block_name), dict) else {}
    body_doc = block_doc.get("body")
    body_ops = body_doc.get("ops") if isinstance(body_doc, dict) else body_doc
    body_ops = [str(op) for op in (body_ops or []) if isinstance(op, str)]
    footer_ops = [str(op) for op in (block_doc.get("footer") or []) if isinstance(op, str)]
    has_template_graph = bool(body_ops or footer_ops)

    entries = manifest_doc.get("entries") if isinstance(manifest_doc.get("entries"), list) else []
    entry_names = {
        str(e.get("name", "")).strip()
        for e in entries
        if isinstance(e, dict) and str(e.get("name", "")).strip()
    }

    errors: list[str] = []
    warnings: list[str] = []

    tie_cfg = _coerce_bool(cfg.get("tie_word_embeddings"))
    has_untied_head = any(
        name in entry_names for name in ("output.weight", "lm_head.weight", "lm_head_weight", "lm_head")
    )
    logits_contract = contract_doc.get("logits_contract") if isinstance(contract_doc.get("logits_contract"), dict) else {}
    template_lm_head_mode = str(logits_contract.get("lm_head", "")).strip().lower()
    tied_modes = {"weight_tying", "tied", "token_emb"}
    strict_untied_modes = {"output_weight", "output.weight", "untied"}

    if tie_cfg is False and template_lm_head_mode in tied_modes:
        errors.append(
            "Manifest says tie_word_embeddings=false but template logits_contract.lm_head indicates weight tying. "
            "Use an untied head contract (lm_head/output.weight)."
        )
    if tie_cfg is False and not has_untied_head:
        errors.append(
            "Manifest says tie_word_embeddings=false but no output/lm_head weight entry was found."
        )
    if tie_cfg is True and template_lm_head_mode in strict_untied_modes and not has_untied_head:
        errors.append(
            "Template requires untied logits head, but manifest has no untied lm_head/output.weight entry."
        )
    if tie_cfg is True and has_untied_head and template_lm_head_mode in tied_modes:
        errors.append(
            "Manifest includes untied lm_head/output.weight while template requests weight tying. "
            "This would strand the untied head and leave output.weight/lm_head.weight unused at lowering time."
        )

    has_qkv_bias = any(name.endswith((".bq", ".bk", ".bv")) for name in entry_names)
    has_qk_norm = bool(
        manifest_doc.get("has_qk_norm")
        or cfg.get("has_qk_norm")
        or any(name.endswith((".q_norm", ".k_norm")) for name in entry_names)
    )
    attention_contract = contract_doc.get("attention_contract") if isinstance(contract_doc.get("attention_contract"), dict) else {}
    contract_qk_norm = _coerce_bool(attention_contract.get("qk_norm"))
    template_has_qk_norm_op = "qk_norm" in body_ops
    if has_template_graph:
        template_has_qkv_proj = any(op in {"qkv_proj", "q_proj", "k_proj", "v_proj"} for op in body_ops)
        if has_qkv_bias and not template_has_qkv_proj:
            errors.append(
                "Manifest has Q/K/V bias tensors but template body has no qkv projection op to consume them."
            )

        if has_qk_norm and not template_has_qk_norm_op:
            errors.append("Manifest has qk_norm tensors but template body is missing qk_norm op.")
        if has_qk_norm and contract_qk_norm is False:
            errors.append("Manifest has qk_norm tensors but template contract attention_contract.qk_norm=false.")
        if (not has_qk_norm) and template_has_qk_norm_op:
            warnings.append("Template includes qk_norm op but manifest does not expose q_norm/k_norm tensors.")
    elif has_qkv_bias or has_qk_norm:
        warnings.append("Template graph ops are unavailable; skipped qkv-bias/qk-norm semantic checks.")

    special_tokens = manifest_doc.get("special_tokens")
    if not isinstance(special_tokens, dict):
        special_tokens = cfg.get("special_tokens") if isinstance(cfg.get("special_tokens"), dict) else {}
    tok_model = str(special_tokens.get("tokenizer_model", "")).strip().lower()
    manifest_tok_type: Optional[str] = None
    if tok_model in {"gpt2", "bpe"}:
        manifest_tok_type = "bpe"
    elif tok_model in {"llama", "sentencepiece", "spm"}:
        manifest_tok_type = "sentencepiece"
    template_tok_type = ""
    tok_contract = contract_doc.get("tokenizer_contract")
    if isinstance(tok_contract, dict):
        template_tok_type = str(tok_contract.get("tokenizer_type", "")).strip().lower()
    if manifest_tok_type and template_tok_type and manifest_tok_type != template_tok_type:
        errors.append(
            f"Tokenizer contract mismatch: manifest tokenizer={manifest_tok_type} but template tokenizer={template_tok_type}."
        )

    sliding_window = cfg.get("sliding_window", manifest_doc.get("sliding_window"))
    has_sliding_window = isinstance(sliding_window, (int, float)) and int(sliding_window) > 0
    template_has_sliding = "attn_sliding" in body_ops
    if has_template_graph:
        if has_sliding_window and not template_has_sliding:
            warnings.append("Manifest advertises sliding_window but template has no attn_sliding op.")
        if (not has_sliding_window) and template_has_sliding:
            warnings.append("Template has attn_sliding op but manifest has no sliding_window value.")

    return (
        len(errors) == 0,
        {
            "errors": errors,
            "warnings": warnings,
            "tie_word_embeddings": tie_cfg,
            "template_lm_head_mode": template_lm_head_mode,
            "has_untied_lm_head_entry": has_untied_head,
            "has_qk_norm": has_qk_norm,
            "template_has_qk_norm_op": template_has_qk_norm_op,
            "manifest_tokenizer_type": manifest_tok_type,
            "template_tokenizer_type": template_tok_type,
            "body_ops": body_ops,
            "footer_ops": footer_ops,
        },
    )


def _gguf_manifest_weight_category_check(
    gguf_path: Path,
    manifest_path: Path,
    *,
    verbose: bool = True,
    report_out: Optional[Path] = None,
) -> tuple[bool, list[str]]:
    """
    Compare GGUF expected converted categories vs manifest categories.

    This catches silent drops where conversion omits required categories.
    """
    if not manifest_path.exists():
        if verbose:
            log_error(f"Missing manifest for category check: {manifest_path}")
        return False, ["<manifest_missing>"]

    try:
        import importlib.util
        inspect_path = SCRIPTS_DIR / "inspect_weights_v7.py"
        spec = importlib.util.spec_from_file_location("inspect_weights_v7", inspect_path)
        if spec is None or spec.loader is None:
            reason = "inspect loader unavailable"
            if verbose:
                log_error(f"GGUF category check unavailable: {reason}")
            if report_out is not None:
                report_out.parent.mkdir(parents=True, exist_ok=True)
                report_out.write_text(
                    json.dumps(
                        {
                            "ok": False,
                            "reason": reason,
                            "gguf_path": str(gguf_path),
                            "manifest_path": str(manifest_path),
                            "missing_categories": ["<category_check_unavailable>"],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            return False, ["<category_check_unavailable>"]
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        inspect_gguf = getattr(mod, "inspect_gguf", None)
        if inspect_gguf is None:
            reason = "inspect_gguf missing in inspect_weights_v7.py"
            if verbose:
                log_error(f"GGUF category check unavailable: {reason}")
            if report_out is not None:
                report_out.parent.mkdir(parents=True, exist_ok=True)
                report_out.write_text(
                    json.dumps(
                        {
                            "ok": False,
                            "reason": reason,
                            "gguf_path": str(gguf_path),
                            "manifest_path": str(manifest_path),
                            "missing_categories": ["<category_check_unavailable>"],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            return False, ["<category_check_unavailable>"]
        _, expected_entries = inspect_gguf(gguf_path, max_layers=None)
    except Exception as exc:
        reason = f"inspect_gguf failed: {exc}"
        if verbose:
            log_error(f"GGUF category check failed: {exc}")
        if report_out is not None:
            report_out.parent.mkdir(parents=True, exist_ok=True)
            report_out.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "reason": reason,
                        "gguf_path": str(gguf_path),
                        "manifest_path": str(manifest_path),
                        "missing_categories": ["<category_check_error>"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return False, ["<category_check_error>"]

    expected_names = {
        str(e.get("name", "")) for e in expected_entries if isinstance(e, dict) and e.get("name")
    }
    manifest_doc = _load_json_dict(manifest_path)
    actual_names = {
        str(e.get("name", ""))
        for e in (manifest_doc.get("entries") or [])
        if isinstance(e, dict) and e.get("name")
    }

    aliases = {
        "lm_head_weight": ("lm_head_weight", "lm_head.weight", "output.weight", "lm_head"),
    }

    missing: list[str] = []
    for name in sorted(expected_names):
        alts = aliases.get(name, (name,))
        if not any(alt in actual_names for alt in alts):
            missing.append(name)

    if verbose:
        log(
            f"  GGUF category check: expected={len(expected_names)} manifest_entries={len(actual_names)} missing={len(missing)}",
            C_DIM,
        )
        if missing:
            preview = ", ".join(missing[:8])
            if len(missing) > 8:
                preview += ", ..."
            log_error(f"Missing converted categories: {preview}")

    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(
            json.dumps(
                {
                    "ok": len(missing) == 0,
                    "gguf_path": str(gguf_path),
                    "manifest_path": str(manifest_path),
                    "expected_category_count": len(expected_names),
                    "manifest_entry_count": len(actual_names),
                    "missing_category_count": len(missing),
                    "missing_categories": missing,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return (len(missing) == 0), missing


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
    coverage_report_path = output_dir / "gguf_weight_category_coverage.json"

    if weights_path.exists() and config_path.exists() and manifest_path.exists() and not force:
        categories_ok, _ = _gguf_manifest_weight_category_check(
            gguf_path,
            manifest_path,
            verbose=False,
            report_out=coverage_report_path,
        )
        if categories_ok:
            log(f"  Using cached weights at {weights_path}", C_DIM)
            log(f"  Category coverage report: {coverage_report_path}", C_DIM)
            return weights_path, config_path
        log("  Cached GGUF conversion category mismatch; regenerating weights", C_ORANGE)
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
    categories_ok, missing_categories = _gguf_manifest_weight_category_check(
        gguf_path,
        manifest_path,
        verbose=True,
        report_out=coverage_report_path,
    )
    if not categories_ok:
        raise RuntimeError(
            "GGUF conversion category coverage failed; see "
            f"{coverage_report_path} (missing={len(missing_categories)})"
        )
    log(f"  Category coverage report: {coverage_report_path}", C_GREEN)
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


def _build_ir_contract_signature(
    manifest_path: Path,
    *,
    target_modes: list[str],
    context_len: Optional[int],
    logits_layout: Optional[str],
    layout_mode: str,
    layer_limit: Optional[int],
    no_fusion: bool,
) -> dict[str, Any]:
    manifest_doc = _load_json_dict(manifest_path) or {}
    cfg = manifest_doc.get("config") if isinstance(manifest_doc.get("config"), dict) else {}
    template_doc = manifest_doc.get("template") if isinstance(manifest_doc.get("template"), dict) else {}
    contract_doc = template_doc.get("contract") if isinstance(template_doc.get("contract"), dict) else {}
    entries = manifest_doc.get("entries") if isinstance(manifest_doc.get("entries"), list) else []
    entry_names = [
        str(e.get("name", "")).strip()
        for e in entries
        if isinstance(e, dict) and str(e.get("name", "")).strip()
    ]
    entry_name_set = set(entry_names)
    qkv_bias_count = sum(1 for n in entry_names if n.endswith((".bq", ".bk", ".bv")))
    qk_norm_count = sum(1 for n in entry_names if n.endswith((".q_norm", ".k_norm")))

    seq = template_doc.get("sequence") if isinstance(template_doc.get("sequence"), list) else []
    block_types = template_doc.get("block_types") if isinstance(template_doc.get("block_types"), dict) else {}
    block_name = str(seq[0]) if seq else ""
    block_doc = block_types.get(block_name) if isinstance(block_types.get(block_name), dict) else {}
    body_doc = block_doc.get("body")
    body_ops = body_doc.get("ops") if isinstance(body_doc, dict) else body_doc

    signature = {
        "schema": "ck.v7.ir_contract.v1",
        "manifest_sha256": _hash_sha256_file(manifest_path),
        "model": {
            "model": cfg.get("model"),
            "arch": cfg.get("arch"),
            "tie_word_embeddings": _coerce_bool(cfg.get("tie_word_embeddings")),
            "vocab_size": cfg.get("vocab_size"),
            "embed_dim": cfg.get("embed_dim"),
            "num_layers": cfg.get("num_layers"),
            "num_heads": cfg.get("num_heads"),
            "num_kv_heads": cfg.get("num_kv_heads"),
            "head_dim": cfg.get("head_dim"),
            "rope_theta": cfg.get("rope_theta"),
            "rope_scaling": cfg.get("rope_scaling"),
            "sliding_window": cfg.get("sliding_window"),
        },
        "template": {
            "name": template_doc.get("name"),
            "sequence": seq,
            "body_ops": body_ops if isinstance(body_ops, list) else [],
            "footer_ops": block_doc.get("footer") if isinstance(block_doc.get("footer"), list) else [],
            "contract": contract_doc,
        },
        "entries": {
            "count": len(entry_names),
            "has_token_emb": "token_emb" in entry_name_set,
            "has_output_weight": "output.weight" in entry_name_set,
            "has_lm_head_weight": "lm_head.weight" in entry_name_set,
            "qkv_bias_count": qkv_bias_count,
            "qk_norm_count": qk_norm_count,
        },
        "build_args": {
            "modes": list(target_modes),
            "context_len": context_len,
            "logits_layout": logits_layout,
            "layout_mode": layout_mode,
            "layer_limit": layer_limit,
            "no_fusion": bool(no_fusion),
        },
    }
    return signature


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

    if _normalize_manifest_file_for_inference(manifest_path):
        log("  Normalized stale manifest/template semantics in weights_manifest.json", C_YELLOW)

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

    semantic_ok, semantic_details = _template_manifest_semantic_check(manifest_path)
    if not semantic_ok:
        for msg in semantic_details.get("errors", []):
            log_error(f"Template/manifest contract mismatch: {msg}")
        sys.exit(1)
    for msg in semantic_details.get("warnings", []):
        log(f"  [contract warning] {msg}", C_ORANGE)

    contract_signature = _build_ir_contract_signature(
        manifest_path,
        target_modes=target_modes,
        context_len=context_len,
        logits_layout=logits_layout,
        layout_mode=layout_mode,
        layer_limit=layer_limit,
        no_fusion=no_fusion,
    )
    contract_hash = _hash_sha256_bytes(_canonical_json_bytes(contract_signature))
    contract_hash_path = output_dir / "ir_build_contract_hash.json"
    cached_contract_hash = None
    cached_contract_doc = _load_json_dict(contract_hash_path)
    if isinstance(cached_contract_doc, dict):
        raw_hash = cached_contract_doc.get("hash")
        if isinstance(raw_hash, str) and raw_hash.strip():
            cached_contract_hash = raw_hash.strip()
    contract_hash_mismatch = bool(cached_contract_hash and cached_contract_hash != contract_hash)
    if not force and cached_contract_hash is None and contract_hash_path.exists():
        contract_hash_mismatch = True
    if not force and contract_hash_mismatch:
        log("  Contract hash changed (tie/head/template semantics), rebuilding IR outputs", C_YELLOW)

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
            contract_mismatch = contract_hash_mismatch
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

            if outputs_exist and not manifest_newer and not context_len_mismatch and not contract_mismatch:
                log(f"  Using cached IR outputs for {mode} at {ir1_path}", C_DIM)
                ir1_paths[mode] = ir1_path
                continue
            if manifest_newer:
                log(f"  Manifest updated, rebuilding IR outputs for {mode}", C_DIM)
            if context_len_mismatch:
                log(f"  Context length changed, rebuilding IR outputs for {mode}", C_DIM)
            if contract_mismatch:
                log(f"  Contract hash changed, rebuilding IR outputs for {mode}", C_DIM)

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
        if int8_activations:
            cmd.append("--prefer-q8-activation")

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

    contract_hash_path.write_text(
        json.dumps(
            {
                "schema": "ck.v7.ir_contract_hash.v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "hash": contract_hash,
                "signature": contract_signature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Return decode IR1 as primary (for compatibility)
    return ir1_paths.get("decode", ir1_paths[target_modes[0]])


def step_codegen(
    ir1_path: Path,
    output_dir: Path,
    force: bool = False,
    profile: bool = False,
    dump: bool = False,
    strict_contracts: bool = False,
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
    if strict_contracts:
        cmd.append("--strict-contracts")
    # Filter empty args
    cmd = [c for c in cmd if c]

    run_cmd(cmd)
    log(f"  Created C code at {model_c_path}", C_GREEN)

    return model_c_path


def _latest_tree_mtime(paths: Sequence[Path]) -> float:
    latest = 0.0
    for root in paths:
        if not root.exists():
            continue
        if root.is_file():
            latest = max(latest, root.stat().st_mtime)
            continue
        for path in root.rglob("*"):
            if path.is_file():
                latest = max(latest, path.stat().st_mtime)
    return latest


def _runtime_lib_needs_rebuild(lib_path: Path, source_roots: Sequence[Path]) -> bool:
    if not lib_path.exists():
        return True
    try:
        return _latest_tree_mtime(source_roots) > lib_path.stat().st_mtime
    except Exception:
        return True


def step_compile(model_c_path: Path, output_dir: Path, force: bool = False) -> Path:
    """Compile C code to shared library linked against libckernel_engine.so."""
    log_step(5, "Compiling to shared library")

    # Output library name (ck_chat.py expects libmodel.so or ck-kernel-inference.so)
    lib_path = output_dir / "libmodel.so"
    kernel_lib = BUILD_DIR / "libckernel_engine.so"
    tokenizer_lib = BUILD_DIR / "libckernel_tokenizer.so"

    log(f"  Source: {model_c_path}", C_DIM)
    log(f"  Lines: {sum(1 for _ in open(model_c_path))}", C_DIM)

    runtime_targets: list[Path] = []
    kernel_source_roots = [
        PROJECT_ROOT / "include",
        PROJECT_ROOT / "src",
        V7_ROOT / "include",
        V7_ROOT / "src",
    ]
    tokenizer_source_roots = [
        PROJECT_ROOT / "include",
        PROJECT_ROOT / "src" / "ck_tokenizer.c",
    ]
    if _runtime_lib_needs_rebuild(kernel_lib, kernel_source_roots):
        runtime_targets.append(kernel_lib)
    if _runtime_lib_needs_rebuild(tokenizer_lib, tokenizer_source_roots):
        runtime_targets.append(tokenizer_lib)
    if runtime_targets:
        verb = "missing/stale" if any(not p.exists() for p in runtime_targets) else "stale"
        log(f"  Building {verb} runtime libs: {', '.join(p.name for p in runtime_targets)}", C_DIM)
        make_targets = [_path_to_make_target(path) for path in runtime_targets]
        run_cmd(["make"] + make_targets, cwd=PROJECT_ROOT)
        still_missing = [path for path in runtime_targets if not path.exists()]
        if still_missing:
            log(f"  Missing required runtime libs after build: {', '.join(path.name for path in still_missing)}", C_RED)
            return model_c_path

    # Skip if already compiled and not forcing
    if lib_path.exists() and not force:
        src_mtime = model_c_path.stat().st_mtime
        lib_mtime = lib_path.stat().st_mtime
        if lib_mtime > src_mtime:
            _sync_runtime_lib(kernel_lib, output_dir / "libckernel_engine.so", "libckernel_engine.so")
            _sync_runtime_lib(tokenizer_lib, output_dir / "libckernel_tokenizer.so", "libckernel_tokenizer.so")
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


def _hash_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _hash_sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_json_dict(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    return None


def _resolve_dataset_id_from_source(data_source: dict[str, Any]) -> str:
    """Derive a stable dataset identifier from source metadata."""
    path = str(data_source.get("source_path") or "").strip()
    if path:
        return Path(path).stem or "unknown"
    name = str(data_source.get("dataset_name") or "").strip()
    if name:
        return Path(name).stem or name
    return "unknown"


def _build_corpus_sampling_log_payload(summary: dict[str, Any], run_dir: Optional[Path]) -> dict[str, Any]:
    epochs = summary.get("corpus_sampling_epochs")
    norm_epochs: list[dict[str, Any]] = []
    if isinstance(epochs, list):
        for row in epochs:
            if isinstance(row, dict):
                norm_epochs.append(dict(row))
    run_id = str(summary.get("run_id") or (run_dir.name if isinstance(run_dir, Path) else "unknown"))
    return {
        "schema": "ck.corpus_sampling_log.v1",
        "run_id": run_id,
        "updated_at": _utc_now_iso(),
        "epochs": norm_epochs,
    }


def _build_training_pipeline_payload(summary: dict, run_dir: Optional[Path]) -> dict:
    mode = str(summary.get("train_mode") or summary.get("mode") or "pretrain").strip().lower()
    if not mode:
        mode = "pretrain"

    stage_order = ["pretrain", "midtrain", "sft", "dpo", "grpo", "ppo"]
    if mode not in stage_order:
        stage_order = [mode] + stage_order
    active_idx = stage_order.index(mode)
    stage_ranges: dict[str, dict[str, Any]] = {}
    raw_sampling = summary.get("corpus_sampling_epochs")
    if isinstance(raw_sampling, list):
        for row in raw_sampling:
            if not isinstance(row, dict):
                continue
            stage = str(row.get("stage_id") or "").strip().lower()
            if not stage:
                continue
            step_start = row.get("step_start")
            step_end = row.get("step_end")
            epoch_id = row.get("epoch")
            cur = stage_ranges.get(stage)
            if cur is None:
                cur = {
                    "step_start": int(step_start) if isinstance(step_start, (int, float)) else None,
                    "step_end": int(step_end) if isinstance(step_end, (int, float)) else None,
                    "epoch_start": int(epoch_id) if isinstance(epoch_id, (int, float)) else None,
                    "epoch_end": int(epoch_id) if isinstance(epoch_id, (int, float)) else None,
                }
                stage_ranges[stage] = cur
            else:
                if isinstance(step_start, (int, float)):
                    s = int(step_start)
                    cur["step_start"] = s if cur.get("step_start") is None else min(int(cur["step_start"]), s)
                if isinstance(step_end, (int, float)):
                    e = int(step_end)
                    cur["step_end"] = e if cur.get("step_end") is None else max(int(cur["step_end"]), e)
                if isinstance(epoch_id, (int, float)):
                    ep = int(epoch_id)
                    cur["epoch_start"] = ep if cur.get("epoch_start") is None else min(int(cur["epoch_start"]), ep)
                    cur["epoch_end"] = ep if cur.get("epoch_end") is None else max(int(cur["epoch_end"]), ep)
    stage_timeline = []
    for idx, stage in enumerate(stage_order):
        if idx < active_idx:
            status = "completed"
        elif idx == active_idx:
            status = "active"
        else:
            status = "planned"
        row = {
            "stage": stage,
            "order": idx,
            "status": status,
            "active": stage == mode,
        }
        rng = stage_ranges.get(stage)
        if isinstance(rng, dict):
            if rng.get("step_start") is not None:
                row["step_start"] = int(rng["step_start"])
            if rng.get("step_end") is not None:
                row["step_end"] = int(rng["step_end"])
            if rng.get("epoch_start") is not None:
                row["epoch_start"] = int(rng["epoch_start"])
            if rng.get("epoch_end") is not None:
                row["epoch_end"] = int(rng["epoch_end"])
        stage_timeline.append(row)

    step_profile = summary.get("step_profile") if isinstance(summary.get("step_profile"), dict) else {}
    processed_tokens = int(step_profile.get("processed_tokens", summary.get("total_tokens", 0)) or 0)
    seq_len = int(summary.get("seq_len", 0) or 0)
    grad_accum = int(summary.get("grad_accum", 0) or 0)
    tokens_per_update = int(summary.get("tokens_per_update", seq_len * max(1, grad_accum)) or 0)
    seed = int(summary.get("seed", 0) or 0)

    data_entries: list[dict] = []
    raw_data_entries = summary.get("data_provenance")
    if isinstance(raw_data_entries, list):
        data_entries = [row for row in raw_data_entries if isinstance(row, dict)]
    if not data_entries:
        data_source = summary.get("data_source") if isinstance(summary.get("data_source"), dict) else {}
        if data_source:
            data_entries = [
                {
                    "stage": mode,
                    "dataset_name": data_source.get("dataset_name") or data_source.get("name") or "train_data",
                    "source_uri": data_source.get("source_uri"),
                    "source_path": data_source.get("source_path"),
                    "split": data_source.get("split") or "train",
                    "token_count": int(data_source.get("token_count", processed_tokens) or 0),
                    "hash": {
                        "algo": "sha256",
                        "value": data_source.get("sha256") or data_source.get("text_sha256"),
                    },
                    "sampling": data_source.get("sampling")
                    if isinstance(data_source.get("sampling"), dict)
                    else {
                        "strategy": "repeat_to_budget",
                        "seed": seed,
                        "shuffle": False,
                    },
                    "packing": data_source.get("packing")
                    if isinstance(data_source.get("packing"), dict)
                    else {
                        "seq_len": seq_len,
                        "grad_accum": grad_accum,
                        "tokens_per_update": tokens_per_update,
                    },
                }
            ]
    if not data_entries:
        data_entries = [
            {
                "stage": mode,
                "dataset_name": "unspecified",
                "source_uri": None,
                "source_path": None,
                "split": "train",
                "token_count": processed_tokens,
                "hash": {"algo": "sha256", "value": None},
                "sampling": {"strategy": "repeat_to_budget", "seed": seed, "shuffle": False},
                "packing": {
                    "seq_len": seq_len,
                    "grad_accum": grad_accum,
                    "tokens_per_update": tokens_per_update,
                },
            }
        ]

    tokenizer_lineage = summary.get("tokenizer_lineage") if isinstance(summary.get("tokenizer_lineage"), dict) else {}
    if not tokenizer_lineage:
        config_doc = _load_json_dict(run_dir / "config.json") if run_dir else None
        manifest_doc = _load_json_dict(run_dir / "weights_manifest.json") if run_dir else None
        train_init_doc = _load_json_dict(run_dir / "train_init_config.json") if run_dir else None
        operator_doc = _load_json_dict(run_dir / "operator_train_run.json") if run_dir else None
        manifest_cfg = manifest_doc.get("config") if isinstance(manifest_doc, dict) and isinstance(manifest_doc.get("config"), dict) else {}
        train_dims = summary.get("train_dims") if isinstance(summary.get("train_dims"), dict) else {}
        effective_dims = train_dims.get("effective") if isinstance(train_dims.get("effective"), dict) else {}

        vocab_size = (
            effective_dims.get("vocab")
            or (config_doc.get("vocab_size") if isinstance(config_doc, dict) else None)
            or manifest_cfg.get("vocab_size")
        )
        template_name = None
        if isinstance(operator_doc, dict):
            template_name = operator_doc.get("template")
        if not template_name and isinstance(train_init_doc, dict):
            template_name = train_init_doc.get("template")
        if not template_name and isinstance(manifest_cfg, dict):
            template_name = manifest_cfg.get("model")

        tokenizer_path = run_dir / "tokenizer.json" if run_dir else None
        tokenizer_hash = _hash_sha256_file(tokenizer_path) if tokenizer_path else None
        tokenizer_lineage = {
            "source": "run_dir" if run_dir else "summary_fallback",
            "type": "synthetic_utf8_mod_vocab",
            "vocab_size": int(vocab_size) if vocab_size is not None else None,
            "bos_token_id": (
                config_doc.get("bos_token_id")
                if isinstance(config_doc, dict)
                else manifest_cfg.get("bos_token_id")
            ),
            "eos_token_id": (
                config_doc.get("eos_token_id")
                if isinstance(config_doc, dict)
                else manifest_cfg.get("eos_token_id")
            ),
            "pad_token_id": (
                config_doc.get("pad_token_id")
                if isinstance(config_doc, dict)
                else manifest_cfg.get("pad_token_id")
            ),
            "chat_template": (
                config_doc.get("chat_template")
                if isinstance(config_doc, dict)
                else None
            ),
            "template": template_name,
            "tokenizer_path": str(tokenizer_path) if tokenizer_path and tokenizer_path.exists() else None,
            "tokenizer_sha256": tokenizer_hash,
        }

    optimizer_hparams = summary.get("optimizer_hparams") if isinstance(summary.get("optimizer_hparams"), dict) else {}
    train_dims_payload = summary.get("train_dims") if isinstance(summary.get("train_dims"), dict) else {}
    effective_dims = train_dims_payload.get("effective") if isinstance(train_dims_payload.get("effective"), dict) else {}
    requested_dims = train_dims_payload.get("requested") if isinstance(train_dims_payload.get("requested"), dict) else {}
    resolved_dims = train_dims_payload.get("resolved") if isinstance(train_dims_payload.get("resolved"), dict) else {}

    def _dim(*names: str, default: int = 0) -> int:
        for src in (effective_dims, resolved_dims, requested_dims):
            if not isinstance(src, dict):
                continue
            for name in names:
                val = src.get(name)
                if isinstance(val, (int, float)):
                    return int(val)
        return int(default)

    model_contract = {
        "family": str(tokenizer_lineage.get("template") or "unknown"),
        "layers": _dim("num_layers", "layers"),
        "embed_dim": _dim("d_model", "embed_dim"),
        "hidden_dim": _dim("hidden", "hidden_dim"),
        "num_heads": _dim("num_heads"),
        "num_kv_heads": _dim("num_kv_heads"),
        "context_len": _dim("context_length", "context_len", "context"),
        "vocab_size": _dim("vocab", "vocab_size"),
    }

    tokenizer_inputs = []
    corpora = tokenizer_lineage.get("tokenizer_corpora")
    if isinstance(corpora, list):
        for row in corpora:
            if not isinstance(row, dict):
                continue
            tokenizer_inputs.append(
                {
                    "name": row.get("name"),
                    "path": row.get("source_path"),
                    "rows": row.get("rows"),
                    "tokens": row.get("token_count"),
                    "bytes": row.get("byte_size"),
                    "sha256": row.get("sha256"),
                    "source": "tokenizer_lineage.tokenizer_corpora",
                }
            )
    if not tokenizer_inputs:
        for row in data_entries:
            if not isinstance(row, dict):
                continue
            tokenizer_inputs.append(
                {
                    "name": row.get("dataset_name"),
                    "path": row.get("source_path"),
                    "rows": row.get("rows"),
                    "tokens": row.get("token_count"),
                    "bytes": row.get("byte_size"),
                    "sha256": (row.get("hash") or {}).get("value") if isinstance(row.get("hash"), dict) else None,
                    "source": "data_provenance",
                }
            )
            break
    data_by_stage = {}
    for row in data_entries:
        if not isinstance(row, dict):
            continue
        st = str(row.get("stage") or "").strip().lower()
        if st and st not in data_by_stage:
            data_by_stage[st] = row

    # Preserve previously emitted stage contracts so non-active stages are not
    # wiped when the current run only reports active-stage provenance.
    existing_stage_by_name: dict[str, dict[str, Any]] = {}
    if isinstance(run_dir, Path):
        existing_doc = _load_json_dict(run_dir / "training_pipeline_latest.json")
        existing_pipeline = (
            existing_doc.get("pipeline")
            if isinstance(existing_doc.get("pipeline"), dict)
            else {}
        )
        existing_rows = (
            existing_pipeline.get("stages")
            if isinstance(existing_pipeline.get("stages"), list)
            else []
        )
        for prev in existing_rows:
            if not isinstance(prev, dict):
                continue
            stage_name = str(prev.get("stage") or prev.get("name") or "").strip().lower()
            if stage_name:
                existing_stage_by_name[stage_name] = dict(prev)

    data_prep_stage = {
        "stage": "data_preparation",
        "stage_id": 0,
        "status": "done",
        "type": "dataset_qc_ascii_prepare",
        "datasets": [
            {
                "name": data_entries[0].get("dataset_name"),
                "path": data_entries[0].get("source_path"),
                "rows": data_entries[0].get("rows"),
                "tokens": data_entries[0].get("token_count"),
                "bytes": data_entries[0].get("byte_size"),
                "sha256": (data_entries[0].get("hash") or {}).get("value")
                if isinstance(data_entries[0].get("hash"), dict)
                else None,
                "source": "data_provenance",
            }
        ] if data_entries else [],
        "ops": ["dataset_qc", "dataset_profile", "tokenizer_roundtrip"],
    }
    existing_data_prep = existing_stage_by_name.get("data_preparation")
    if isinstance(existing_data_prep, dict):
        data_prep_stage = dict(existing_data_prep)
        data_prep_stage["stage"] = "data_preparation"
        data_prep_stage["stage_id"] = 0

    tokenizer_stage = {
        "stage": "tokenizer",
        "stage_id": 1,
        "status": "built_or_reused",
        "type": str(tokenizer_lineage.get("type") or "unknown"),
        "tokenizer": {
            "type": tokenizer_lineage.get("type"),
            "vocab_size": tokenizer_lineage.get("vocab_size"),
            "path": tokenizer_lineage.get("tokenizer_path"),
            "sha256": tokenizer_lineage.get("tokenizer_sha256"),
            "reused_run_tokenizer": tokenizer_lineage.get("reused_run_tokenizer"),
        },
        "datasets": tokenizer_inputs,
        "coverage": {
            "active_dataset_in_corpus": tokenizer_lineage.get("active_dataset_in_tokenizer_corpus"),
            "status": tokenizer_lineage.get("coverage_status"),
            "note": tokenizer_lineage.get("coverage_note"),
        },
        "ops": ["tokenizer_build_or_reuse"],
    }
    existing_tokenizer = existing_stage_by_name.get("tokenizer")
    if isinstance(existing_tokenizer, dict):
        tokenizer_stage = dict(existing_tokenizer)
        tokenizer_stage["stage"] = "tokenizer"
        tokenizer_stage["stage_id"] = 1
        if tokenizer_stage.get("status") in (None, ""):
            tokenizer_stage["status"] = "built_or_reused"

    pipeline_stages = [data_prep_stage, tokenizer_stage]
    next_stage_id = 2
    for row in stage_timeline:
        if not isinstance(row, dict):
            continue
        st = str(row.get("stage") or "").strip().lower()
        if not st:
            continue
        prov = data_by_stage.get(st, {})
        datasets = []
        if isinstance(prov, dict) and prov:
            datasets.append(
                {
                    "name": prov.get("dataset_name"),
                    "path": prov.get("source_path"),
                    "rows": prov.get("rows"),
                    "tokens": prov.get("token_count"),
                    "bytes": prov.get("byte_size"),
                    "sha256": (prov.get("hash") or {}).get("value") if isinstance(prov.get("hash"), dict) else None,
                    "source": "data_provenance",
                }
            )
        prev_stage = existing_stage_by_name.get(st)
        prev_datasets = (
            prev_stage.get("datasets")
            if isinstance(prev_stage, dict) and isinstance(prev_stage.get("datasets"), list)
            else None
        )
        if st != mode and prev_datasets is not None:
            datasets = prev_datasets
        elif st == mode and not datasets and prev_datasets is not None:
            datasets = prev_datasets
        tokenizer_coverage = (
            prev_stage.get("tokenizer_coverage")
            if isinstance(prev_stage, dict) and isinstance(prev_stage.get("tokenizer_coverage"), dict)
            else {}
        )
        pipeline_stages.append(
            {
                "stage": st,
                "stage_id": int(next_stage_id),
                "status": str(row.get("status") or "planned"),
                "type": "training_stage",
                "datasets": datasets,
                "tokenizer_coverage": tokenizer_coverage,
                "ops": ["train"] if st == mode else [],
            }
        )
        next_stage_id += 1

    payload = {
        "schema": "ck.training_pipeline.v1",
        "generated_at": _utc_now_iso(),
        "model": model_contract,
        "pipeline": {
            "schema": "ck.training_pipeline_contract.v1",
            "source_of_truth": "training_pipeline_latest.json",
            "active_stage": mode,
            "stages": pipeline_stages,
        },
        "active_stage": mode,
        "stage_timeline": stage_timeline,
        "backend": str(summary.get("backend") or ""),
        "optimizer": {
            "name": str(summary.get("optimizer") or ""),
            "lr": float(summary.get("lr", 0.0) or 0.0),
            "hparams": optimizer_hparams,
        },
        "execution": {
            "epochs": int(summary.get("epochs", 0) or 0),
            "steps": int(summary.get("steps", 0) or 0),
            "micro_steps": int(summary.get("micro_steps", 0) or 0),
            "optimizer_steps": int(summary.get("optimizer_steps", 0) or 0),
            "seq_len": seq_len,
            "grad_accum": grad_accum,
            "tokens_total": int(summary.get("total_tokens", 0) or 0),
            "tokens_per_update": tokens_per_update,
            "processed_tokens": processed_tokens,
        },
        "train_dims": train_dims_payload,
        "data_provenance": data_entries,
        "tokenizer_lineage": tokenizer_lineage,
        "sources": {
            "summary": "train_e2e_latest.json",
            "run_dir": str(run_dir) if run_dir else None,
        },
    }
    return payload


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
    training_pipeline = _build_training_pipeline_payload(s, None)
    corpus_sampling_log = _build_corpus_sampling_log_payload(s, None)

    payloads = {
        "training_loss_curve_latest.json": training_loss_curve,
        "training_parity_latest.json": training_parity,
        "training_grad_norms_latest.json": training_grad_norms,
        "training_step_profile_latest.json": training_step_profile,
        "training_checkpoint_policy_latest.json": training_checkpoint_policy,
        "training_pipeline_latest.json": training_pipeline,
        "corpus_sampling_log_latest.json": corpus_sampling_log,
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


def _resolve_train_token_stream(args: argparse.Namespace) -> tuple[Optional[list[int]], Optional[Path]]:
    """Resolve optional pre-tokenized integer stream file."""
    token_file = getattr(args, "train_token_file", None)
    if not token_file:
        return None, None
    path = Path(token_file).expanduser().resolve()
    if not path.exists():
        log_error(f"Training token file not found: {path}")
        sys.exit(2)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log_error(f"Failed to read training token file from {path}: {e}")
        sys.exit(2)
    # Parse one integer per line, ignoring blank lines and '#' comments.
    vals: list[int] = []
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            vals.append(int(line))
        except ValueError:
            log_error(
                f"Training token file {path}:{line_no}: expected one integer per line, got: {raw_line!r}"
            )
            sys.exit(2)
    if any(v < 0 for v in vals):
        bad = next(v for v in vals if v < 0)
        log_error(f"Training token file contains negative token id ({bad}): {path}")
        sys.exit(2)
    if len(vals) <= 1:
        log_error(f"Training token file must contain at least 2 integers: {path}")
        sys.exit(2)
    return vals, path


def _resolve_train_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "train_mode", "pretrain") or "pretrain").lower()
    if getattr(args, "pretraining", False):
        mode = "pretrain"
    if mode not in ("pretrain", "midtrain", "sft", "dpo", "grpo", "ppo"):
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


def _assess_train_safety(args: argparse.Namespace, train_backend: str) -> dict:
    """Evaluate known-risk train configs and enforce policy when requested."""
    optimizer = str(getattr(args, "train_optimizer", "adamw") or "adamw").lower()
    lr = float(getattr(args, "train_lr", 1e-3) or 1e-3)
    max_grad_norm = float(getattr(args, "train_max_grad_norm", 0.0) or 0.0)
    unsafe_lr_threshold = float(getattr(args, "train_unsafe_adamw_lr_threshold", 1e-3) or 1e-3)
    if unsafe_lr_threshold <= 0.0:
        log_error("--train-unsafe-adamw-lr-threshold must be > 0")
        sys.exit(2)

    allow_unsafe = bool(getattr(args, "allow_unsafe_adamw_lr", False))
    enforce = bool(getattr(args, "enforce_production_safety", False) or getattr(args, "train_strict", False))

    risky = bool(
        optimizer == "adamw"
        and lr >= unsafe_lr_threshold
        and train_backend in ("ck", "pytorch", "both")
    )

    status = "ok"
    message = ""
    if risky:
        status = "unsafe"
        message = (
            "AdamW long-horizon profile is high-risk: lr >= threshold "
            "(known all-C drift around step ~800 at lr=1e-3). "
            "Production path should lower --train-lr below threshold. "
            "Use --allow-unsafe-adamw-lr only for diagnostics."
        )
        if enforce and not allow_unsafe:
            log_error(message)
            sys.exit(2)
        if allow_unsafe:
            status = "unsafe_allowed"
            message = "Unsafe AdamW LR profile explicitly allowed by CLI flag."
        log(f"  Warning: {message}", C_ORANGE)

    return {
        "status": status,
        "message": message,
        "optimizer": optimizer,
        "lr": lr,
        "train_backend": train_backend,
        "max_grad_norm": max_grad_norm,
        "grad_clip_configured": bool(max_grad_norm > 0.0),
        "unsafe_adamw_lr_threshold": unsafe_lr_threshold,
        "enforce_production_safety": enforce,
        "allow_unsafe_adamw_lr": allow_unsafe,
        "risky": risky,
    }



def _build_train_token_batches(
    train_text: Optional[str],
    total_tokens: int,
    seq_len: int,
    vocab: int,
    seed: int,
    token_stream: Optional[list[int]] = None,
) -> list[tuple[list[int], list[int], int]]:
    """Build deterministic token/target batches for CK runtime train stepping."""
    if seq_len < 1:
        return []
    total_tokens_i = max(1, int(total_tokens))
    seq_len_i = max(1, int(seq_len))
    windows = max(1, int(math.ceil(float(total_tokens_i) / float(seq_len_i))))
    needed = max(total_tokens_i + 1, windows * seq_len_i + 1, seq_len_i + 1)
    if token_stream:
        base = [int(v) for v in token_stream]
        bad = next((v for v in base if v < 0 or v >= int(vocab)), None)
        if bad is not None:
            raise ValueError(
                f"train token id out of range for model vocab: token={bad}, vocab={int(vocab)}"
            )
        repeats = (needed + len(base) - 1) // len(base)
        stream = (base * repeats)[:needed]
    elif train_text:
        raw = train_text.encode("utf-8", errors="ignore")
        if len(raw) < 2:
            raw = b"hello"
        ids = [int(b) % int(vocab) for b in raw]
        repeats = (needed + len(ids) - 1) // len(ids)
        stream = (ids * repeats)[:needed]
    else:
        rng = random.Random(int(seed))
        stream = [rng.randrange(int(vocab)) for _ in range(needed)]

    batches: list[tuple[list[int], list[int], int]] = []
    for w in range(windows):
        i = w * seq_len_i
        remaining = total_tokens_i - i
        valid_tokens = seq_len_i if remaining >= seq_len_i else max(1, remaining)
        x = stream[i:i + seq_len_i]
        y = stream[i + 1:i + seq_len_i + 1]
        if len(x) == seq_len_i and len(y) == seq_len_i:
            batches.append((x, y, int(valid_tokens)))
    if not batches:
        batches.append((stream[:seq_len_i], stream[1:seq_len_i + 1], seq_len_i))
    return batches


def _unpack_train_batch(batch: tuple) -> tuple[list[int], list[int], int]:
    if isinstance(batch, tuple) and len(batch) >= 3:
        x_vals = list(batch[0])
        y_vals = list(batch[1])
        valid_tokens = int(batch[2])
    elif isinstance(batch, tuple) and len(batch) == 2:
        x_vals = list(batch[0])
        y_vals = list(batch[1])
        valid_tokens = min(len(x_vals), len(y_vals))
    else:
        raise ValueError(f"Invalid batch shape: {type(batch)}")
    if valid_tokens < 1:
        valid_tokens = 1
    if valid_tokens > len(x_vals):
        valid_tokens = len(x_vals)
    if valid_tokens > len(y_vals):
        valid_tokens = len(y_vals)
    return x_vals, y_vals, int(valid_tokens)



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
    train_vocab: Optional[int] = None,
    train_d_model: Optional[int] = None,
    train_hidden: Optional[int] = None,
    train_num_layers: Optional[int] = None,
) -> Optional[dict]:
    """Run periodic PyTorch oracle reference once and return parsed JSON payload."""
    train_script = SCRIPTS_DIR / "train_parity_epochs_v7.py"
    if not train_script.exists():
        log("  Warning: parity oracle script missing; skipping oracle replay", C_ORANGE)
        return None

    parity_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(parity_python) if parity_python.exists() else sys.executable
    oracle_json = run_dir / "oracle_reference_latest.json"
    try:
        adamw_cfg = _resolve_train_adamw_hparams(args, run_dir)
    except ValueError as e:
        log_error(str(e))
        return None
    adamw_effective = dict(adamw_cfg.get("effective") or {})

    resolved_vocab = int(train_vocab) if _as_positive_int(train_vocab) is not None else int(getattr(args, "train_vocab", 256))
    resolved_d_model = int(train_d_model) if _as_positive_int(train_d_model) is not None else int(getattr(args, "train_d_model", 64))
    resolved_hidden = int(train_hidden) if _as_positive_int(train_hidden) is not None else int(getattr(args, "train_hidden", 128))
    resolved_num_layers = (
        int(train_num_layers)
        if _as_positive_int(train_num_layers) is not None
        else int(getattr(args, "num_layers", 1) or 1)
    )

    cmd = [
        python_exec,
        str(train_script),
        "--epochs", str(getattr(args, "train_epochs", 3)),
        "--seq-len", str(getattr(args, "train_seq_len", 16)),
        "--total-tokens", str(getattr(args, "train_total_tokens", 1024)),
        "--grad-accum", str(getattr(args, "train_grad_accum", 8)),
        "--optimizer", str(getattr(args, "train_optimizer", "adamw")),
        "--lr", str(getattr(args, "train_lr", 1e-3)),
        "--adamw-beta1", str(float(adamw_effective.get("beta1", 0.9))),
        "--adamw-beta2", str(float(adamw_effective.get("beta2", 0.999))),
        "--adamw-eps", str(float(adamw_effective.get("eps", 1e-8))),
        "--adamw-weight-decay", str(float(adamw_effective.get("weight_decay", 0.01))),
        "--seed", str(getattr(args, "train_seed", 42)),
        "--vocab", str(resolved_vocab),
        "--d-model", str(resolved_d_model),
        "--hidden", str(resolved_hidden),
        "--num-layers", str(resolved_num_layers),
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


def _ck_export_runtime_optimizer_state_snapshot(lib: ctypes.CDLL) -> Optional[tuple[object, int]]:
    """Export current CK runtime AdamW moment-state snapshot buffer."""
    if (
        not hasattr(lib, "ck_train_get_optimizer_state_snapshot_numel")
        or not hasattr(lib, "ck_train_export_optimizer_state_snapshot")
    ):
        return None
    try:
        numel = int(lib.ck_train_get_optimizer_state_snapshot_numel())
    except Exception:
        return None
    if numel <= 0 or numel > (1 << 30):
        return None
    buf = (ctypes.c_float * numel)()
    try:
        wrote = int(lib.ck_train_export_optimizer_state_snapshot(buf, ctypes.c_int(numel)))
    except Exception:
        return None
    if wrote <= 0:
        return None
    if wrote < numel:
        trunc = (ctypes.c_float * wrote)()
        ctypes.memmove(ctypes.addressof(trunc), ctypes.addressof(buf), wrote * ctypes.sizeof(ctypes.c_float))
        return trunc, int(wrote)
    return buf, int(numel)


def _ck_import_runtime_optimizer_state_snapshot(lib: ctypes.CDLL, snapshot_buf: object, snapshot_numel: int) -> int:
    """Import a previously exported CK runtime AdamW moment-state snapshot."""
    if not hasattr(lib, "ck_train_import_optimizer_state_snapshot"):
        return -1
    try:
        return int(lib.ck_train_import_optimizer_state_snapshot(snapshot_buf, ctypes.c_int(int(snapshot_numel))))
    except Exception:
        return -2


def _ck_export_runtime_accum_snapshot(lib: ctypes.CDLL) -> Optional[tuple[object, int]]:
    """Export current CK runtime accumulation buffers (grad + grad_act)."""
    if (
        not hasattr(lib, "ck_train_get_accum_snapshot_numel")
        or not hasattr(lib, "ck_train_export_accum_snapshot")
    ):
        return None
    try:
        numel = int(lib.ck_train_get_accum_snapshot_numel())
    except Exception:
        return None
    if numel < 0 or numel > (1 << 30):
        return None
    if numel == 0:
        return ((ctypes.c_float * 1)(), 0)
    buf = (ctypes.c_float * numel)()
    try:
        wrote = int(lib.ck_train_export_accum_snapshot(buf, ctypes.c_int(numel)))
    except Exception:
        return None
    if wrote < 0:
        return None
    if wrote == 0 and numel == 0:
        return ((ctypes.c_float * 1)(), 0)
    if wrote == 0 and numel > 0:
        return None
    if wrote < numel:
        trunc = (ctypes.c_float * wrote)()
        ctypes.memmove(ctypes.addressof(trunc), ctypes.addressof(buf), wrote * ctypes.sizeof(ctypes.c_float))
        return trunc, int(wrote)
    return buf, int(numel)


def _ck_import_runtime_accum_snapshot(lib: ctypes.CDLL, snapshot_buf: object, snapshot_numel: int) -> int:
    """Import CK runtime accumulation buffers (grad + grad_act)."""
    if not hasattr(lib, "ck_train_import_accum_snapshot"):
        return -1
    try:
        return int(lib.ck_train_import_accum_snapshot(snapshot_buf, ctypes.c_int(int(snapshot_numel))))
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
    tag: Optional[str] = None,
) -> Optional[Path]:
    """Persist CK weight snapshot for drift/replay triage."""
    try:
        snap_dir = run_dir / "oracle_ck_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        stem = f"step_{int(step):08d}"
        if tag:
            stem = f"{stem}_{str(tag)}"
        bin_path = snap_dir / f"{stem}.f32bin"
        meta_path = snap_dir / f"{stem}.json"
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
    tag: Optional[str] = None,
) -> Optional[Path]:
    """Persist CK activation snapshot for drift/replay triage."""
    try:
        snap_dir = run_dir / "oracle_ck_activations"
        snap_dir.mkdir(parents=True, exist_ok=True)
        stem = f"step_{int(step):08d}"
        if tag:
            stem = f"{stem}_{str(tag)}"
        bin_path = snap_dir / f"{stem}.f32bin"
        meta_path = snap_dir / f"{stem}.json"
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


def _write_ck_optimizer_state_snapshot_artifact(
    run_dir: Path,
    step: int,
    snapshot_buf: object,
    snapshot_numel: int,
    *,
    reason: str,
    tag: Optional[str] = None,
) -> Optional[Path]:
    """Persist CK optimizer-state snapshot for runtime parity triage."""
    try:
        snap_dir = run_dir / "oracle_ck_optimizer_state"
        snap_dir.mkdir(parents=True, exist_ok=True)
        stem = f"step_{int(step):08d}"
        if tag:
            stem = f"{stem}_{str(tag)}"
        bin_path = snap_dir / f"{stem}.f32bin"
        meta_path = snap_dir / f"{stem}.json"
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


def _write_ck_accum_snapshot_artifact(
    run_dir: Path,
    step: int,
    snapshot_buf: object,
    snapshot_numel: int,
    *,
    reason: str,
    tag: Optional[str] = None,
) -> Optional[Path]:
    """Persist CK grad-accum snapshot for runtime parity triage."""
    try:
        snap_dir = run_dir / "oracle_ck_accum"
        snap_dir.mkdir(parents=True, exist_ok=True)
        stem = f"step_{int(step):08d}"
        if tag:
            stem = f"{stem}_{str(tag)}"
        bin_path = snap_dir / f"{stem}.f32bin"
        meta_path = snap_dir / f"{stem}.json"
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
    has_step_ex = bool(hasattr(lib, "ck_train_step_ex"))

    lib.ck_train_init.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.ck_train_init.restype = ctypes.c_int
    lib.ck_train_step.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_float]
    lib.ck_train_step.restype = ctypes.c_int
    if has_step_ex:
        lib.ck_train_step_ex.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
        ]
        lib.ck_train_step_ex.restype = ctypes.c_int

    float_ptr = ctypes.cast(init_payload["float_buffer"], ctypes.POINTER(ctypes.c_float))
    size_ptr = ctypes.cast(init_payload["sizes_buffer"], ctypes.POINTER(ctypes.c_int))
    init_rc = int(lib.ck_train_init(float_ptr, size_ptr, ctypes.c_int(init_payload["num_params"])))
    out = {"init_rc": init_rc, "step_rcs": [], "losses": []}
    if init_rc < 0:
        return out

    limit = max(1, min(int(steps), len(batches)))
    for i in range(limit):
        x_vals, y_vals, valid_tokens = _unpack_train_batch(batches[i])
        x_buf = (ctypes.c_int32 * len(x_vals))(*x_vals)
        y_buf = (ctypes.c_int32 * len(y_vals))(*y_vals)
        loss_out = ctypes.c_float(0.0)
        if has_step_ex:
            rc = int(lib.ck_train_step_ex(x_buf, y_buf, ctypes.c_int(int(valid_tokens)), ctypes.byref(loss_out), ctypes.c_float(lr)))
        else:
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



def _ensure_train_runtime_artifacts(
    run_dir: Path,
    python_exec: str,
    strict: bool,
    runtime_defines: Optional[dict] = None,
    train_tokens: int = 1,
    extra_cflags: Optional[Sequence[str]] = None,
) -> tuple[Path, Path]:
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
    exec_plan = run_dir / "train_exec_plan.json"
    c_src = run_dir / "generated_train_runtime_v7.c"
    c_summary = run_dir / "generated_train_runtime_summary_v7.json"

    build_ir_script = SCRIPTS_DIR / "build_ir_train_v7.py"
    lower_ir_script = SCRIPTS_DIR / "lower_ir2_backward_v7.py"
    inv_script = SCRIPTS_DIR / "validate_ir_train_invariants_v7.py"
    layout_script = SCRIPTS_DIR / "generate_train_layout_v7.py"
    layout_audit_script = SCRIPTS_DIR / "validate_train_memory_layout_v7.py"
    exec_plan_script = SCRIPTS_DIR / "generate_train_exec_plan_v7.py"

    desired_tokens = max(1, int(train_tokens or 1))

    def _read_ir1_tokens(path: Path) -> Optional[int]:
        if not path.exists():
            return None
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            tensors = doc.get("tensors") if isinstance(doc, dict) else None
            if isinstance(tensors, dict):
                for tid, meta in tensors.items():
                    if not isinstance(tid, str) or not tid.startswith("act.Sheader.dense_embedding_lookup"):
                        continue
                    if isinstance(meta, dict):
                        shape = meta.get("shape")
                        if isinstance(shape, list) and shape:
                            tok = int(shape[0])
                            if tok > 0:
                                return tok
            cfg = doc.get("config") if isinstance(doc, dict) else None
            if isinstance(cfg, dict):
                tok = cfg.get("train_tokens", cfg.get("tokens"))
                if tok is not None:
                    tok = int(tok)
                    if tok > 0:
                        return tok
        except Exception:
            return None
        return None

    # Each stage only regenerates when inputs are newer. This keeps CLI reruns fast
    # while still guaranteeing that libtrain is rebuilt after contract changes.
    existing_ir1_tokens = _read_ir1_tokens(ir1)
    needs_ir1 = (
        (not ir1.exists())
        or (manifest.exists() and manifest.stat().st_mtime > ir1.stat().st_mtime)
        or (build_ir_script.exists() and build_ir_script.stat().st_mtime > ir1.stat().st_mtime)
        or (existing_ir1_tokens is not None and int(existing_ir1_tokens) != int(desired_tokens))
    )
    if needs_ir1:
        cmd = [
            python_exec,
            str(build_ir_script),
            "--manifest", str(manifest),
            "--output", str(ir1),
            "--report-out", str(ir1_report),
            "--tokens", str(desired_tokens),
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

    needs_exec_plan = (
        (not exec_plan.exists())
        or (ir2.exists() and ir2.stat().st_mtime > exec_plan.stat().st_mtime)
        or (exec_plan_script.exists() and exec_plan_script.stat().st_mtime > exec_plan.stat().st_mtime)
    )
    if needs_exec_plan:
        cmd = [
            python_exec,
            str(exec_plan_script),
            "--ir2", str(ir2),
            "--output", str(exec_plan),
            "--mode", "deterministic",
        ]
        run_cmd(cmd, cwd=PROJECT_ROOT)

    codegen_script = SCRIPTS_DIR / "codegen_train_runtime_v7.py"
    regen_codegen = (
        (not c_src.exists())
        or (not c_summary.exists())
        or (ir2.exists() and ir2.stat().st_mtime > c_src.stat().st_mtime)
        or (exec_plan.exists() and exec_plan.stat().st_mtime > c_src.stat().st_mtime)
        or (codegen_script.exists() and codegen_script.stat().st_mtime > c_src.stat().st_mtime)
    )
    if regen_codegen:
        cmd = [
            python_exec,
            str(codegen_script),
            "--ir2", str(ir2),
            "--manifest", str(manifest),
            "--layout", str(layout_train),
            "--exec-plan", str(exec_plan),
            "--output", str(c_src),
            "--summary-out", str(c_summary),
        ]
        run_cmd(cmd, cwd=PROJECT_ROOT)

    lib_ck = BUILD_DIR / "libckernel_engine.so"
    if not lib_ck.exists():
        run_cmd(["make", "--no-print-directory", str(lib_ck)], cwd=PROJECT_ROOT)

    libtrain_so = run_dir / "libtrain.so"
    defines = dict(runtime_defines or {})
    cflags = [str(f) for f in (extra_cflags or []) if str(f).strip()]
    needs_compile = (not libtrain_so.exists()) or (c_src.stat().st_mtime > libtrain_so.stat().st_mtime)
    if defines:
        needs_compile = True
    if cflags:
        needs_compile = True
    if needs_compile:
        cc = os.environ.get("CC") or "gcc"
        cmd = [
            cc,
            "-shared",
            "-fPIC",
            "-O3",
            *cflags,
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
            if isinstance(v, bool):
                dval = "1" if v else "0"
            elif isinstance(v, int):
                dval = str(v)
            elif isinstance(v, float):
                dval = f"{v:.9g}"
                if ("." not in dval) and ("e" not in dval.lower()):
                    dval += ".0"
            else:
                dval = str(v)
            cmd.append(f"-D{k}={dval}")
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

    seq_len = int(getattr(args, "train_seq_len", 16) or 16)
    if seq_len < 1:
        log_error(f"--train-seq-len must be >= 1, got {seq_len}")
        sys.exit(2)
    grad_accum = int(getattr(args, "train_grad_accum", 8) or 8)
    if grad_accum < 1:
        log_error(f"--train-grad-accum must be >= 1, got {grad_accum}")
        sys.exit(2)

    runtime_defines: dict = {
        "CK_NUM_TOKENS": max(1, int(seq_len)),
        "CK_GRAD_ACCUM_STEPS": int(grad_accum),
    }
    bitwise_parity_enabled = bool(getattr(args, "bitwise_parity", False))
    bitwise_compile_flags: list[str] = []
    bitwise_runtime_env: dict[str, str] = {}
    bitwise_runtime_env_prev: dict[str, Optional[str]] = {}
    ck_loss_backend = str(getattr(args, "ck_loss_backend", "c") or "c").strip().lower()
    runtime_ce_backend = "default"
    # Generated runtime cannot call into Python/PyTorch kernels directly.
    # For backend=ck, route torch parity CE request to the strict C ptref CE path.
    if ck_loss_backend in ("c_ptref", "torch"):
        runtime_defines["CK_TRAIN_USE_CE_PTREF"] = 1
        runtime_ce_backend = "ptref"
    else:
        runtime_defines["CK_TRAIN_USE_CE_PTREF"] = 0
    if ck_loss_backend == "torch":
        log("  backend=ck CE note: --ck-loss-backend=torch maps to ptref C CE in generated runtime", C_ORANGE)
    max_grad_norm = float(getattr(args, "train_max_grad_norm", 0.0) or 0.0)
    runtime_defines["CK_MAX_GRAD_NORM"] = f"{max_grad_norm:.9g}"
    # Strict memory diagnostics can snapshot all weights via malloc().
    # Production runs may disable this while keeping the diagnostic entrypoint.
    if bool(getattr(args, "train_disable_diag_snapshot", False) or getattr(args, "enforce_production_safety", False)):
        runtime_defines["CK_TRAIN_DIAG_WEIGHT_SNAPSHOT"] = 0
    try:
        adamw_cfg = _resolve_train_adamw_hparams(args, run_dir)
    except ValueError as e:
        raise RuntimeError(str(e)) from e
    adamw_effective = dict(adamw_cfg.get("effective") or {})
    runtime_defines["CK_ADAMW_BETA1"] = f"{float(adamw_effective.get('beta1', 0.9)):.9g}"
    runtime_defines["CK_ADAMW_BETA2"] = f"{float(adamw_effective.get('beta2', 0.999)):.9g}"
    runtime_defines["CK_ADAMW_EPS"] = f"{float(adamw_effective.get('eps', 1e-8)):.9g}"
    runtime_defines["CK_ADAMW_WEIGHT_DECAY"] = f"{float(adamw_effective.get('weight_decay', 0.01)):.9g}"
    if isinstance(profile_meta, dict):
        profile_meta["optimizer_hparams"] = {
            "source": str(adamw_cfg.get("source") or "defaults"),
            "manifest": adamw_cfg.get("manifest"),
            "adamw": {
                "beta1": float(adamw_effective.get("beta1", 0.9)),
                "beta2": float(adamw_effective.get("beta2", 0.999)),
                "eps": float(adamw_effective.get("eps", 1e-8)),
                "weight_decay": float(adamw_effective.get("weight_decay", 0.01)),
            },
        }
    if bool(getattr(args, "ablate_attention_backward", False)):
        runtime_defines["CK_ABLATE_ATTENTION_BACKWARD"] = 1
    if bool(getattr(args, "ablate_rope_backward_qk", False)):
        runtime_defines["CK_ABLATE_ROPE_BACKWARD_QK"] = 1
    if bool(getattr(args, "ablate_qk_norm_backward", False)):
        runtime_defines["CK_ABLATE_QK_NORM_BACKWARD"] = 1
    if bool(getattr(args, "train_runtime_canary_checks", False)):
        runtime_defines["CK_RUNTIME_CANARY_CHECKS"] = 1
    if bool(getattr(args, "train_runtime_bounds_assert", False)):
        runtime_defines["CK_RUNTIME_BOUNDS_ASSERT"] = 1
    fault_op_id = int(getattr(args, "train_runtime_fault_op_id", -1) or -1)
    if fault_op_id >= 0:
        runtime_defines["CK_RUNTIME_FAULT_INJECT"] = 1
        runtime_defines["CK_FAULT_INJECT_OP_ID"] = int(fault_op_id)
    if bitwise_parity_enabled:
        # Bitwise-parity mode is explicitly slower: force deterministic runtime
        # scheduling + conservative FP compilation for the generated train runtime.
        runtime_defines["CK_TRAIN_BITWISE_PARITY"] = 1
        cc = os.environ.get("CC") or "gcc"
        bitwise_compile_flags = _bitwise_parity_compile_flags(cc)
        bitwise_runtime_env = {
            "CK_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "CK_QK_NORM_BACKWARD_ISA": "scalar",
        }
        for env_k, env_v in bitwise_runtime_env.items():
            bitwise_runtime_env_prev[env_k] = os.environ.get(env_k)
            os.environ[env_k] = str(env_v)
        if bitwise_compile_flags:
            log(
                "  bitwise parity mode: on "
                f"(env=deterministic, cflags={' '.join(bitwise_compile_flags)})",
                C_DIM,
            )
        else:
            log(
                "  bitwise parity mode: on (env=deterministic, no extra compiler flags accepted)",
                C_ORANGE,
            )
        if isinstance(profile_meta, dict):
            profile_meta["bitwise_parity"] = {
                "enabled": True,
                "compile_flags": list(bitwise_compile_flags),
                "runtime_env": dict(bitwise_runtime_env),
            }
    if any(
        bool(getattr(args, name, False))
        for name in ("ablate_attention_backward", "ablate_rope_backward_qk", "ablate_qk_norm_backward")
    ):
        log(
            "  runtime ablations: "
            f"attention_backward={int(bool(getattr(args, 'ablate_attention_backward', False)))} "
            f"rope_backward_qk={int(bool(getattr(args, 'ablate_rope_backward_qk', False)))} "
            f"qk_norm_backward={int(bool(getattr(args, 'ablate_qk_norm_backward', False)))}",
            C_ORANGE,
        )

    try:  # ensure bitwise_runtime_env_prev is restored on any exception
        _json_out = _run_ck_train_runtime_body(
            args=args,
            run_dir=run_dir,
            json_out=json_out,
            train_text=train_text,
            train_mode=train_mode,
            train_backend=train_backend,
            profile_meta=profile_meta,
            seq_len=seq_len,
            grad_accum=grad_accum,
            runtime_defines=runtime_defines,
            bitwise_parity_enabled=bitwise_parity_enabled,
            bitwise_compile_flags=bitwise_compile_flags,
            bitwise_runtime_env=bitwise_runtime_env,
            runtime_ce_backend=runtime_ce_backend,
            ck_loss_backend=ck_loss_backend,
            python_exec=python_exec,
            parity_python=parity_python,
            adamw_cfg=adamw_cfg,
            adamw_effective=adamw_effective,
        )
    finally:
        if bitwise_runtime_env_prev:
            for env_k, prev_v in bitwise_runtime_env_prev.items():
                if prev_v is None:
                    os.environ.pop(env_k, None)
                else:
                    os.environ[env_k] = str(prev_v)
    return _json_out


def _run_ck_train_runtime_body(
    args: argparse.Namespace,
    run_dir: Path,
    json_out: Path,
    train_text: Optional[str],
    train_mode: str,
    train_backend: str,
    profile_meta: dict,
    seq_len: int,
    grad_accum: int,
    runtime_defines: dict,
    bitwise_parity_enabled: bool,
    bitwise_compile_flags: list[str],
    bitwise_runtime_env: dict[str, str],
    runtime_ce_backend: str,
    ck_loss_backend: str,
    python_exec: str,
    parity_python: Path,
    adamw_cfg: dict[str, Any],
    adamw_effective: dict[str, Any],
) -> Path:
    """Inner body of _run_ck_train_runtime (extracted for try/finally env guard)."""
    parity_on = bitwise_parity_enabled  # alias for existing references

    c_src, libtrain_so = _ensure_train_runtime_artifacts(
        run_dir=run_dir,
        python_exec=python_exec,
        strict=bool(getattr(args, "train_strict", False)),
        runtime_defines=runtime_defines,
        train_tokens=seq_len,
        extra_cflags=bitwise_compile_flags,
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
    has_step_ex = bool(hasattr(lib, "ck_train_step_ex"))

    def _set_runtime_strict_parity(lib_handle, enabled: bool) -> bool:
        """Best-effort strict kernel parity toggle for generated runtime."""
        if not hasattr(lib_handle, "ck_set_strict_parity"):
            return False
        try:
            lib_handle.ck_set_strict_parity.argtypes = [ctypes.c_int]
            lib_handle.ck_set_strict_parity.restype = None
            lib_handle.ck_set_strict_parity(1 if enabled else 0)
            return True
        except Exception:
            return False

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
    if has_step_ex:
        lib.ck_train_step_ex.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
        ]
        lib.ck_train_step_ex.restype = ctypes.c_int

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
    has_opt_state_snapshot_numel = bool(hasattr(lib, "ck_train_get_optimizer_state_snapshot_numel"))
    has_opt_state_snapshot_export = bool(hasattr(lib, "ck_train_export_optimizer_state_snapshot"))
    has_opt_state_snapshot_import = bool(hasattr(lib, "ck_train_import_optimizer_state_snapshot"))
    has_accum_snapshot_numel = bool(hasattr(lib, "ck_train_get_accum_snapshot_numel"))
    has_accum_snapshot_export = bool(hasattr(lib, "ck_train_export_accum_snapshot"))
    has_accum_snapshot_import = bool(hasattr(lib, "ck_train_import_accum_snapshot"))
    has_act_snapshot_numel = bool(hasattr(lib, "ck_train_get_activation_snapshot_numel"))
    has_act_snapshot_export = bool(hasattr(lib, "ck_train_export_activation_snapshot"))
    has_flush_optimizer_api = bool(hasattr(lib, "ck_train_flush_optimizer"))
    has_accum_counter_api = bool(hasattr(lib, "ck_train_get_accum_counter"))
    has_accum_steps_api = bool(hasattr(lib, "ck_train_get_accum_steps"))
    has_opt_step_getter_api = bool(hasattr(lib, "ck_train_get_opt_step"))
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
    if has_opt_state_snapshot_numel:
        lib.ck_train_get_optimizer_state_snapshot_numel.argtypes = []
        lib.ck_train_get_optimizer_state_snapshot_numel.restype = ctypes.c_int
    if has_opt_state_snapshot_export:
        lib.ck_train_export_optimizer_state_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ck_train_export_optimizer_state_snapshot.restype = ctypes.c_int
    if has_opt_state_snapshot_import:
        lib.ck_train_import_optimizer_state_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ck_train_import_optimizer_state_snapshot.restype = ctypes.c_int
    if has_accum_snapshot_numel:
        lib.ck_train_get_accum_snapshot_numel.argtypes = []
        lib.ck_train_get_accum_snapshot_numel.restype = ctypes.c_int
    if has_accum_snapshot_export:
        lib.ck_train_export_accum_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ck_train_export_accum_snapshot.restype = ctypes.c_int
    if has_accum_snapshot_import:
        lib.ck_train_import_accum_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ck_train_import_accum_snapshot.restype = ctypes.c_int
    if has_act_snapshot_numel:
        lib.ck_train_get_activation_snapshot_numel.argtypes = []
        lib.ck_train_get_activation_snapshot_numel.restype = ctypes.c_int
    if has_act_snapshot_export:
        lib.ck_train_export_activation_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ck_train_export_activation_snapshot.restype = ctypes.c_int
    if has_flush_optimizer_api:
        lib.ck_train_flush_optimizer.argtypes = [ctypes.c_float]
        lib.ck_train_flush_optimizer.restype = ctypes.c_int
    if has_accum_counter_api:
        lib.ck_train_get_accum_counter.argtypes = []
        lib.ck_train_get_accum_counter.restype = ctypes.c_int
    if has_accum_steps_api:
        lib.ck_train_get_accum_steps.argtypes = []
        lib.ck_train_get_accum_steps.restype = ctypes.c_int
    if has_opt_step_getter_api:
        lib.ck_train_get_opt_step.argtypes = []
        lib.ck_train_get_opt_step.restype = ctypes.c_int

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
    total_tokens = int(getattr(args, "train_total_tokens", 1024) or 1024)
    lr = float(getattr(args, "train_lr", 1e-3) or 1e-3)
    seed = int(getattr(args, "train_seed", 42) or 42)
    resolved_train_dims = _resolve_train_dims_for_run(args, run_dir)
    requested_train_dims = dict(resolved_train_dims.get("requested") or {})
    effective_train_dims = dict(resolved_train_dims.get("effective") or {})
    if resolved_train_dims.get("mismatches"):
        mismatch_txt = ", ".join(
            f"{k}:{v.get('requested')}->{v.get('effective')}"
            for k, v in (resolved_train_dims.get("mismatches") or {}).items()
            if isinstance(v, dict)
        )
        manifest_src = str(resolved_train_dims.get("manifest") or "")
        log(
            f"  ck runtime train dims: using run-dir manifest ({mismatch_txt})"
            + (f" [{manifest_src}]" if manifest_src else ""),
            C_ORANGE,
        )
    vocab = int(effective_train_dims.get("vocab", 256) or 256)
    d_model_hint = int(effective_train_dims.get("d_model", 64) or 64)
    hidden_hint = int(effective_train_dims.get("hidden", 128) or 128)
    train_num_layers = int(effective_train_dims.get("num_layers", 1) or 1)
    optimizer = str(getattr(args, "train_optimizer", "adamw") or "adamw")

    train_data_arg = getattr(args, "train_data", None)
    train_data_path = Path(train_data_arg).expanduser().resolve() if train_data_arg else None
    token_stream_vals, token_stream_path = _resolve_train_token_stream(args)
    if token_stream_path is not None and train_text:
        log("  train-token-file provided: ignoring --data/--prompt text tokenization path", C_DIM)
        train_text = None
    if token_stream_vals is not None and len(token_stream_vals) > 0:
        tok_min = int(min(token_stream_vals))
        tok_max = int(max(token_stream_vals))
        if tok_min < 0 or tok_max >= int(vocab):
            log_error(
                "Token stream has ids outside model vocab range: "
                f"min={tok_min} max={tok_max} vocab={int(vocab)}"
            )
            log_error(
                "Re-init run-dir with vocab >= tokenizer vocab, or provide a token file "
                "that matches run vocab."
            )
            sys.exit(2)
    train_data_kind = (
        "token_file"
        if token_stream_path
        else ("file" if train_data_path else ("inline_text" if train_text else "synthetic"))
    )
    if token_stream_path:
        dataset_name = token_stream_path.name
        source_uri = token_stream_path.as_uri()
        source_path = str(token_stream_path)
    elif train_data_path:
        dataset_name = train_data_path.name
        source_uri = train_data_path.as_uri()
        source_path = str(train_data_path)
    elif train_text:
        if getattr(args, "train_text", None):
            dataset_name = "inline_train_text"
            source_uri = "inline://train_text"
        elif getattr(args, "prompt", None):
            dataset_name = "prompt_text"
            source_uri = "inline://prompt"
        else:
            dataset_name = "inline_text"
            source_uri = "inline://text"
        source_path = None
    else:
        dataset_name = "synthetic_seeded"
        source_uri = "synthetic://seeded"
        source_path = None
    train_text_bytes = train_text.encode("utf-8", errors="ignore") if train_text else b""
    train_text_hash = _hash_sha256_bytes(train_text_bytes) if train_text_bytes else None
    train_data_source = {
        "kind": train_data_kind,
        "dataset_name": dataset_name,
        "source_uri": source_uri,
        "source_path": source_path,
        "split": "train",
        "token_count": int(total_tokens),
        "text_chars": int(len(train_text)) if train_text else 0,
        "text_sha256": train_text_hash,
        "sampling": {
            "strategy": "repeat_to_budget",
            "seed": int(seed),
            "shuffle": False,
        },
        "packing": {
            "seq_len": int(seq_len),
            "grad_accum": int(grad_accum),
            "tokens_per_update": int(seq_len * grad_accum),
        },
    }
    config_doc = _load_json_dict(run_dir / "config.json") or {}
    tokenizer_path = run_dir / "tokenizer.json"
    tokenizer_lineage = {
        "source": "run_dir",
        "type": "pretokenized_stream" if token_stream_path else "synthetic_utf8_mod_vocab",
        "vocab_size": int(vocab),
        "bos_token_id": config_doc.get("bos_token_id"),
        "eos_token_id": config_doc.get("eos_token_id"),
        "pad_token_id": config_doc.get("pad_token_id"),
        "chat_template": config_doc.get("chat_template"),
        "template": None,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path.exists() else None,
        "tokenizer_sha256": _hash_sha256_file(tokenizer_path) if tokenizer_path.exists() else None,
    }
    run_meta = _load_json_dict(run_dir / "operator_train_run.json") or {}
    if isinstance(run_meta, dict) and run_meta.get("template"):
        tokenizer_lineage["template"] = run_meta.get("template")
    elif isinstance(config_doc, dict) and config_doc.get("model"):
        tokenizer_lineage["template"] = config_doc.get("model")

    parity_on = bool(getattr(args, "parity_on", False))
    parity_profile = str(getattr(args, "parity_profile", "balanced") or "balanced")
    parity_every = int(getattr(args, "parity_every", 50) or 0)
    train_loss_tol = float(getattr(args, "train_loss_tol", 2e-5) or 2e-5)
    activation_tol = max(train_loss_tol * 10.0, 1e-5)
    parity_replay_on_check = bool(getattr(args, "parity_replay_on_check", False))
    parity_replay_tol = float(getattr(args, "parity_replay_tol", 1e-7) or 1e-7)
    bruteforce_debug = bool(getattr(args, "bruteforce_debug", False))
    dump_on_drift = bool(getattr(args, "dump_on_drift", False))
    dump_on_check = bool(getattr(args, "dump_on_check", False))
    dump_check_topk = max(1, int(getattr(args, "dump_check_topk", 200) or 200))
    replay_weight_tol = float(getattr(args, "train_param_tol", 3e-5) or 3e-5)
    train_save_every = int(getattr(args, "train_save_every", 0) or 0)
    train_save_final = bool(getattr(args, "train_save_final", True))

    if bruteforce_debug:
        if not parity_on:
            parity_on = True
        parity_profile = "debug"
        parity_every = 1
        if not parity_replay_on_check:
            parity_replay_on_check = True
        if not dump_on_check:
            dump_on_check = True
        log(
            "  generated-runtime brute-force debug: parity_on + parity_every=1 + parity_replay_on_check + dump_on_check",
            C_DIM,
        )

    replay_auto_enabled = False
    has_weight_snapshot_api = bool(has_snapshot_numel and has_snapshot_export and has_snapshot_import)
    has_optimizer_state_snapshot_api = bool(
        has_opt_state_snapshot_numel and has_opt_state_snapshot_export and has_opt_state_snapshot_import
    )
    has_accum_snapshot_api = bool(
        has_accum_snapshot_numel and has_accum_snapshot_export and has_accum_snapshot_import
    )
    runtime_num_tokens = _infer_runtime_num_tokens(
        runtime_summary,
        d_model_hint=int(d_model_hint),
        vocab_hint=vocab,
    )
    runtime_grad_accum_steps = int(lib.ck_train_get_accum_steps()) if has_accum_steps_api else int(grad_accum)

    try:
        batches = _build_train_token_batches(
            train_text,
            total_tokens,
            seq_len,
            vocab,
            seed,
            token_stream=token_stream_vals,
        )
    except ValueError as e:
        log_error(str(e))
        sys.exit(2)
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

    # For oracle parity runs, force strict kernel math by default.
    # Can also be enabled explicitly for non-oracle CK runs via --kernel-strict-math.
    strict_runtime_enabled = bool(
        bitwise_parity_enabled or parity_on or bool(getattr(args, "kernel_strict_math", False))
    )
    strict_runtime_bound = _set_runtime_strict_parity(lib, strict_runtime_enabled)

    oracle_payload = None
    oracle_loss_by_step: dict[int, float] = {}
    oracle_max_steps_used = 0
    oracle_source = "none"
    snapshot_oracle_error = None
    snapshot_oracle_enabled = False
    snapshot_oracle_fn = None
    oracle_strict = False

    if parity_on:
        grad_accum_for_oracle = max(1, int(runtime_grad_accum_steps))
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
                train_vocab=vocab,
                train_d_model=d_model_hint,
                train_hidden=hidden_hint,
                train_num_layers=train_num_layers,
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

    # Auto-enable replay checks only when oracle checks run every micro-step.
    # Sparse parity cadence (e.g. every=8) can skip intermediate runtime state
    # that replay-on-check currently expects, so keep it opt-in there.
    checks_cover_all_steps = bool(parity_on and total_steps > 0 and len(check_steps) == int(total_steps))

    # Auto-enable replay checks:
    # - grad_accum=1: weight snapshots are sufficient.
    # - grad_accum>1: require accumulation-buffer snapshots + counter state.
    if (
        parity_on
        and checks_cover_all_steps
        and snapshot_oracle_enabled
        and has_weight_snapshot_api
        and (not parity_replay_on_check)
        and (
            int(runtime_grad_accum_steps) <= 1
            or (
                has_accum_snapshot_api
                and has_accum_counter_api
                and has_optimizer_state_snapshot_api
                and has_opt_step_getter_api
            )
        )
    ):
        parity_replay_on_check = True
        replay_auto_enabled = True

    replay_lib = None
    replay_runtime_error = None
    replay_has_step_ex = False
    replay_has_forward_api = False
    replay_has_set_batch_api = False
    replay_has_act_snapshot_api = False
    replay_has_set_accum_counter_api = False
    replay_has_get_accum_counter_api = False
    replay_has_get_opt_step_api = False
    replay_has_set_opt_step_api = False
    replay_has_opt_state_snapshot_api = False
    replay_has_accum_snapshot_api = False
    if (parity_replay_on_check or snapshot_oracle_enabled) and has_weight_snapshot_api:
        try:
            replay_so = run_dir / "libtrain_replay.so"
            shutil.copy2(libtrain_so, replay_so)
            replay_lib = ctypes.CDLL(str(replay_so))
            if strict_runtime_bound:
                _set_runtime_strict_parity(replay_lib, strict_runtime_enabled)
            if not hasattr(replay_lib, "ck_train_step") or not hasattr(replay_lib, "ck_train_init"):
                raise RuntimeError("missing ck_train_step/ck_train_init in replay runtime")
            if not hasattr(replay_lib, "ck_train_import_weight_snapshot"):
                raise RuntimeError("missing ck_train_import_weight_snapshot in replay runtime")

            replay_lib.ck_train_init.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            replay_lib.ck_train_init.restype = ctypes.c_int
            replay_lib.ck_train_step.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_float]
            replay_lib.ck_train_step.restype = ctypes.c_int
            replay_has_step_ex = bool(hasattr(replay_lib, "ck_train_step_ex"))
            if replay_has_step_ex:
                replay_lib.ck_train_step_ex.argtypes = [
                    ctypes.POINTER(ctypes.c_int32),
                    ctypes.POINTER(ctypes.c_int32),
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_float,
                ]
                replay_lib.ck_train_step_ex.restype = ctypes.c_int
            replay_lib.ck_train_import_weight_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            replay_lib.ck_train_import_weight_snapshot.restype = ctypes.c_int

            replay_has_opt_state_snapshot_api = bool(
                hasattr(replay_lib, "ck_train_get_optimizer_state_snapshot_numel")
                and hasattr(replay_lib, "ck_train_export_optimizer_state_snapshot")
                and hasattr(replay_lib, "ck_train_import_optimizer_state_snapshot")
            )
            replay_has_accum_snapshot_api = bool(
                hasattr(replay_lib, "ck_train_get_accum_snapshot_numel")
                and hasattr(replay_lib, "ck_train_export_accum_snapshot")
                and hasattr(replay_lib, "ck_train_import_accum_snapshot")
            )
            replay_has_forward_api = bool(hasattr(replay_lib, "ck_train_forward_step"))
            replay_has_set_batch_api = bool(hasattr(replay_lib, "ck_train_set_batch"))
            replay_has_act_snapshot_api = bool(
                hasattr(replay_lib, "ck_train_get_activation_snapshot_numel")
                and hasattr(replay_lib, "ck_train_export_activation_snapshot")
            )
            replay_has_set_accum_counter_api = bool(hasattr(replay_lib, "ck_train_set_accum_counter"))
            replay_has_get_accum_counter_api = bool(hasattr(replay_lib, "ck_train_get_accum_counter"))
            replay_has_get_opt_step_api = bool(hasattr(replay_lib, "ck_train_get_opt_step"))
            replay_has_set_opt_step_api = bool(hasattr(replay_lib, "ck_train_set_opt_step"))
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
            if replay_has_set_accum_counter_api:
                replay_lib.ck_train_set_accum_counter.argtypes = [ctypes.c_int]
                replay_lib.ck_train_set_accum_counter.restype = ctypes.c_int
            if replay_has_get_accum_counter_api:
                replay_lib.ck_train_get_accum_counter.argtypes = []
                replay_lib.ck_train_get_accum_counter.restype = ctypes.c_int
            if replay_has_get_opt_step_api:
                replay_lib.ck_train_get_opt_step.argtypes = []
                replay_lib.ck_train_get_opt_step.restype = ctypes.c_int
            if replay_has_set_opt_step_api:
                replay_lib.ck_train_set_opt_step.argtypes = [ctypes.c_int]
                replay_lib.ck_train_set_opt_step.restype = ctypes.c_int
            if replay_has_opt_state_snapshot_api:
                replay_lib.ck_train_get_optimizer_state_snapshot_numel.argtypes = []
                replay_lib.ck_train_get_optimizer_state_snapshot_numel.restype = ctypes.c_int
                replay_lib.ck_train_export_optimizer_state_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
                replay_lib.ck_train_export_optimizer_state_snapshot.restype = ctypes.c_int
                replay_lib.ck_train_import_optimizer_state_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
                replay_lib.ck_train_import_optimizer_state_snapshot.restype = ctypes.c_int
            if replay_has_accum_snapshot_api:
                replay_lib.ck_train_get_accum_snapshot_numel.argtypes = []
                replay_lib.ck_train_get_accum_snapshot_numel.restype = ctypes.c_int
                replay_lib.ck_train_export_accum_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
                replay_lib.ck_train_export_accum_snapshot.restype = ctypes.c_int
                replay_lib.ck_train_import_accum_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
                replay_lib.ck_train_import_accum_snapshot.restype = ctypes.c_int

            replay_init_rc = int(replay_lib.ck_train_init(float_ptr, size_ptr, ctypes.c_int(init_payload["num_params"])))
            if replay_init_rc < 0:
                raise RuntimeError(f"replay ck_train_init failed with code {replay_init_rc}")
        except Exception as e:
            replay_runtime_error = str(e)
            replay_lib = None
            replay_has_step_ex = False
            replay_has_forward_api = False
            replay_has_set_batch_api = False
            replay_has_act_snapshot_api = False
            replay_has_set_accum_counter_api = False
            replay_has_get_accum_counter_api = False
            replay_has_get_opt_step_api = False
            replay_has_set_opt_step_api = False
            replay_has_opt_state_snapshot_api = False
            replay_has_accum_snapshot_api = False

    step = 0
    micro_steps = 0
    optimizer_steps = 0
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
    optimizer_snapshot_artifacts: list[str] = []
    accum_snapshot_artifacts: list[str] = []
    check_dump_artifacts: list[dict] = []
    checkpoint_artifacts: list[dict] = []
    last_checkpoint_step = 0
    oracle_points = 0
    last_oracle_loss = None
    corpus_sampling_epochs: list[dict] = []
    source_stage = str(train_mode or "pretrain").strip().lower() or "pretrain"

    for epoch_idx in range(epochs):
        epoch_num = int(epoch_idx) + 1
        epoch_step_start = int(step) + 1
        epoch_loss_start: Optional[float] = None
        epoch_loss_end: Optional[float] = None
        epoch_rows_sampled = 0
        epoch_tokens_consumed = 0
        for batch in batches:
            x_vals, y_vals, valid_tokens = _unpack_train_batch(batch)
            step += 1
            micro_steps += 1
            epoch_rows_sampled += 1
            x_buf = (ctypes.c_int32 * len(x_vals))(*x_vals)
            y_buf = (ctypes.c_int32 * len(y_vals))(*y_vals)
            loss_out = ctypes.c_float(0.0)

            need_check_snapshot = bool(
                parity_on
                and (step in check_steps)
                and (
                    snapshot_oracle_enabled
                    or parity_replay_on_check
                    or dump_on_drift
                    or dump_on_check
                )
            )
            pre_replay_accum_counter = int(lib.ck_train_get_accum_counter()) if has_accum_counter_api else None
            pre_replay_opt_step = int(lib.ck_train_get_opt_step()) if has_opt_step_getter_api else None
            pre_snapshot = None
            pre_snapshot_numel = 0
            pre_optimizer_state_snapshot = None
            pre_optimizer_state_snapshot_numel = 0
            pre_accum_snapshot = None
            pre_accum_snapshot_numel = 0
            if need_check_snapshot and has_weight_snapshot_api:
                snap = _ck_export_runtime_weight_snapshot(lib)
                if snap is not None:
                    pre_snapshot, pre_snapshot_numel = snap
            if need_check_snapshot and (parity_replay_on_check or dump_on_check) and has_optimizer_state_snapshot_api:
                opt_snap = _ck_export_runtime_optimizer_state_snapshot(lib)
                if opt_snap is not None:
                    pre_optimizer_state_snapshot, pre_optimizer_state_snapshot_numel = opt_snap
            if need_check_snapshot and (parity_replay_on_check or dump_on_check) and has_accum_snapshot_api:
                accum_snap = _ck_export_runtime_accum_snapshot(lib)
                if accum_snap is not None:
                    pre_accum_snapshot, pre_accum_snapshot_numel = accum_snap

            t0 = time.perf_counter()
            if has_step_ex:
                calls = int(lib.ck_train_step_ex(x_buf, y_buf, ctypes.c_int(int(valid_tokens)), ctypes.byref(loss_out), ctypes.c_float(lr)))
            else:
                calls = int(lib.ck_train_step(x_buf, y_buf, ctypes.byref(loss_out), ctypes.c_float(lr)))
            t1 = time.perf_counter()
            if calls < 0:
                raise RuntimeError(f"ck_train_step failed at step {step} (calls={calls})")

            accum_now = None
            if has_accum_counter_api:
                accum_now = int(lib.ck_train_get_accum_counter())
                if accum_now == 0:
                    optimizer_steps += 1
            elif (micro_steps % grad_accum) == 0:
                optimizer_steps += 1
            opt_step_now = int(lib.ck_train_get_opt_step()) if has_opt_step_getter_api else None

            step_ms = (t1 - t0) * 1000.0
            total_ck_ms += step_ms
            consumed_tokens = min(int(valid_tokens), int(runtime_num_tokens))
            processed_tokens += consumed_tokens
            epoch_tokens_consumed += consumed_tokens
            loss_val = float(loss_out.value)
            if epoch_loss_start is None:
                epoch_loss_start = float(loss_val)
            epoch_loss_end = float(loss_val)

            post_snapshot = None
            post_snapshot_numel = 0
            post_optimizer_state_snapshot = None
            post_optimizer_state_snapshot_numel = 0
            post_accum_snapshot = None
            post_accum_snapshot_numel = 0
            if (parity_replay_on_check or dump_on_check) and (step in check_steps) and has_weight_snapshot_api:
                post_snap = _ck_export_runtime_weight_snapshot(lib)
                if post_snap is not None:
                    post_snapshot, post_snapshot_numel = post_snap
            if (parity_replay_on_check or dump_on_check) and (step in check_steps) and has_optimizer_state_snapshot_api:
                post_opt_snap = _ck_export_runtime_optimizer_state_snapshot(lib)
                if post_opt_snap is not None:
                    post_optimizer_state_snapshot, post_optimizer_state_snapshot_numel = post_opt_snap
            if (parity_replay_on_check or dump_on_check) and (step in check_steps) and has_accum_snapshot_api:
                post_accum_snap = _ck_export_runtime_accum_snapshot(lib)
                if post_accum_snap is not None:
                    post_accum_snapshot, post_accum_snapshot_numel = post_accum_snap

            if dump_on_check and (step in check_steps):
                dump_row: dict = {"step": int(step)}
                if pre_snapshot is not None and pre_snapshot_numel > 0:
                    wpre = _write_ck_weight_snapshot_artifact(
                        run_dir,
                        step,
                        pre_snapshot,
                        pre_snapshot_numel,
                        reason="parity_check",
                        tag="pre",
                    )
                    if wpre is not None:
                        snapshot_artifacts.append(str(wpre))
                        dump_row["weight_pre"] = str(wpre)
                if post_snapshot is not None and post_snapshot_numel > 0:
                    wpost = _write_ck_weight_snapshot_artifact(
                        run_dir,
                        step,
                        post_snapshot,
                        post_snapshot_numel,
                        reason="parity_check",
                        tag="post",
                    )
                    if wpost is not None:
                        snapshot_artifacts.append(str(wpost))
                        dump_row["weight_post"] = str(wpost)
                if pre_optimizer_state_snapshot is not None and pre_optimizer_state_snapshot_numel > 0:
                    opre = _write_ck_optimizer_state_snapshot_artifact(
                        run_dir,
                        step,
                        pre_optimizer_state_snapshot,
                        pre_optimizer_state_snapshot_numel,
                        reason="parity_check",
                        tag="pre",
                    )
                    if opre is not None:
                        optimizer_snapshot_artifacts.append(str(opre))
                        dump_row["optimizer_pre"] = str(opre)
                if post_optimizer_state_snapshot is not None and post_optimizer_state_snapshot_numel > 0:
                    opost = _write_ck_optimizer_state_snapshot_artifact(
                        run_dir,
                        step,
                        post_optimizer_state_snapshot,
                        post_optimizer_state_snapshot_numel,
                        reason="parity_check",
                        tag="post",
                    )
                    if opost is not None:
                        optimizer_snapshot_artifacts.append(str(opost))
                        dump_row["optimizer_post"] = str(opost)
                if pre_accum_snapshot is not None and pre_accum_snapshot_numel >= 0:
                    apre = _write_ck_accum_snapshot_artifact(
                        run_dir,
                        step,
                        pre_accum_snapshot,
                        pre_accum_snapshot_numel,
                        reason="parity_check",
                        tag="pre",
                    )
                    if apre is not None:
                        accum_snapshot_artifacts.append(str(apre))
                        dump_row["accum_pre"] = str(apre)
                if post_accum_snapshot is not None and post_accum_snapshot_numel >= 0:
                    apost = _write_ck_accum_snapshot_artifact(
                        run_dir,
                        step,
                        post_accum_snapshot,
                        post_accum_snapshot_numel,
                        reason="parity_check",
                        tag="post",
                    )
                    if apost is not None:
                        accum_snapshot_artifacts.append(str(apost))
                        dump_row["accum_post"] = str(apost)
                if has_act_snapshot_export:
                    act_snap = _ck_export_runtime_activation_snapshot(lib)
                    if act_snap is not None:
                        act_buf, act_numel = act_snap
                        adump = _write_ck_activation_snapshot_artifact(
                            run_dir,
                            step,
                            act_buf,
                            act_numel,
                            reason="parity_check",
                            tag="post",
                        )
                        if adump is not None:
                            activation_snapshot_artifacts.append(str(adump))
                            dump_row["activation_post"] = str(adump)
                if len(dump_row) > 1:
                    check_dump_artifacts.append(dump_row)

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
            replay_optimizer_state_max_abs_diff = None
            replay_optimizer_state_mean_abs_diff = None
            replay_optimizer_state_error = None
            replay_optimizer_state_tol = replay_weight_tol
            replay_accum_snapshot_max_abs_diff = None
            replay_accum_snapshot_mean_abs_diff = None
            replay_accum_snapshot_error = None
            replay_accum_snapshot_tol = replay_weight_tol
            replay_pre_optimizer_state_import_max_abs_diff = None
            replay_pre_optimizer_state_import_error = None
            replay_pre_accum_import_max_abs_diff = None
            replay_pre_accum_import_error = None
            replay_post_accum_counter = None
            replay_post_opt_step = None
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
                        "optimizer_state_max_abs_diff": None,
                        "optimizer_state_mean_abs_diff": None,
                        "optimizer_state_threshold": replay_optimizer_state_tol,
                        "optimizer_state_error": ("replay_runtime_unavailable" if has_optimizer_state_snapshot_api else None),
                        "accum_snapshot_max_abs_diff": None,
                        "accum_snapshot_mean_abs_diff": None,
                        "accum_snapshot_threshold": replay_accum_snapshot_tol,
                        "accum_snapshot_error": ("replay_runtime_unavailable" if has_accum_snapshot_api else None),
                        "reason": f"replay_runtime_unavailable:{replay_runtime_error}",
                    })
                elif pre_snapshot is not None and pre_snapshot_numel > 0:
                    import_rc = _ck_import_runtime_weight_snapshot(replay_lib, pre_snapshot, pre_snapshot_numel)
                    if import_rc >= 0:
                        if has_optimizer_state_snapshot_api:
                            if not replay_has_opt_state_snapshot_api:
                                replay_optimizer_state_error = "replay_optimizer_state_snapshot_api_unavailable"
                            elif pre_optimizer_state_snapshot is None or pre_optimizer_state_snapshot_numel <= 0:
                                replay_optimizer_state_error = "optimizer_state_snapshot_unavailable"
                            else:
                                opt_import_rc = _ck_import_runtime_optimizer_state_snapshot(
                                    replay_lib,
                                    pre_optimizer_state_snapshot,
                                    pre_optimizer_state_snapshot_numel,
                                )
                                if opt_import_rc < 0:
                                    replay_optimizer_state_error = f"optimizer_state_snapshot_import_failed:{opt_import_rc}"
                        if has_accum_snapshot_api:
                            if not replay_has_accum_snapshot_api:
                                replay_accum_snapshot_error = "replay_accum_snapshot_api_unavailable"
                            elif pre_accum_snapshot is None or pre_accum_snapshot_numel < 0:
                                replay_accum_snapshot_error = "accum_snapshot_unavailable"
                            else:
                                accum_import_rc = _ck_import_runtime_accum_snapshot(
                                    replay_lib,
                                    pre_accum_snapshot,
                                    pre_accum_snapshot_numel,
                                )
                                if accum_import_rc < 0:
                                    replay_accum_snapshot_error = f"accum_snapshot_import_failed:{accum_import_rc}"
                        if (pre_replay_accum_counter is not None) and (int(pre_replay_accum_counter) > 0) and (not replay_has_set_accum_counter_api):
                            if replay_accum_snapshot_error is None:
                                replay_accum_snapshot_error = "replay_set_accum_counter_api_unavailable"
                        if (pre_replay_opt_step is not None) and (int(pre_replay_opt_step) > 0) and (not replay_has_set_opt_step_api):
                            if replay_optimizer_state_error is None:
                                replay_optimizer_state_error = "replay_set_opt_step_api_unavailable"
                        if replay_has_set_accum_counter_api and pre_replay_accum_counter is not None:
                            _ = int(replay_lib.ck_train_set_accum_counter(ctypes.c_int(int(pre_replay_accum_counter))))
                        if replay_has_set_opt_step_api and pre_replay_opt_step is not None:
                            _ = int(replay_lib.ck_train_set_opt_step(ctypes.c_int(int(pre_replay_opt_step))))
                        if has_optimizer_state_snapshot_api and pre_optimizer_state_snapshot is not None and pre_optimizer_state_snapshot_numel > 0:
                            if replay_has_opt_state_snapshot_api:
                                replay_pre_opt = _ck_export_runtime_optimizer_state_snapshot(replay_lib)
                                if replay_pre_opt is not None:
                                    replay_pre_opt_buf, replay_pre_opt_numel = replay_pre_opt
                                    if int(replay_pre_opt_numel) == int(pre_optimizer_state_snapshot_numel):
                                        try:
                                            import numpy as _np
                                            src_pre_opt_np = _np.ctypeslib.as_array(
                                                pre_optimizer_state_snapshot,
                                                shape=(int(pre_optimizer_state_snapshot_numel),),
                                            ).astype(_np.float32, copy=False)
                                            replay_pre_opt_np = _np.ctypeslib.as_array(
                                                replay_pre_opt_buf,
                                                shape=(int(replay_pre_opt_numel),),
                                            ).astype(_np.float32, copy=False)
                                            pre_opt_delta_np = _np.abs(src_pre_opt_np - replay_pre_opt_np)
                                            replay_pre_optimizer_state_import_max_abs_diff = float(_np.max(pre_opt_delta_np)) if pre_opt_delta_np.size else 0.0
                                        except Exception as e:
                                            replay_pre_optimizer_state_import_error = f"pre_optimizer_state_compare_failed:{e}"
                                    else:
                                        replay_pre_optimizer_state_import_error = (
                                            f"pre_optimizer_state_size_mismatch:{replay_pre_opt_numel}!={pre_optimizer_state_snapshot_numel}"
                                        )
                                else:
                                    replay_pre_optimizer_state_import_error = "pre_optimizer_state_export_unavailable"
                        if has_accum_snapshot_api and pre_accum_snapshot is not None and pre_accum_snapshot_numel >= 0:
                            if replay_has_accum_snapshot_api:
                                replay_pre_accum = _ck_export_runtime_accum_snapshot(replay_lib)
                                if replay_pre_accum is not None:
                                    replay_pre_accum_buf, replay_pre_accum_numel = replay_pre_accum
                                    if int(replay_pre_accum_numel) == int(pre_accum_snapshot_numel):
                                        try:
                                            import numpy as _np
                                            src_pre_accum_np = _np.ctypeslib.as_array(
                                                pre_accum_snapshot,
                                                shape=(int(pre_accum_snapshot_numel),),
                                            ).astype(_np.float32, copy=False)
                                            replay_pre_accum_np = _np.ctypeslib.as_array(
                                                replay_pre_accum_buf,
                                                shape=(int(replay_pre_accum_numel),),
                                            ).astype(_np.float32, copy=False)
                                            pre_accum_delta_np = _np.abs(src_pre_accum_np - replay_pre_accum_np)
                                            replay_pre_accum_import_max_abs_diff = float(_np.max(pre_accum_delta_np)) if pre_accum_delta_np.size else 0.0
                                        except Exception as e:
                                            replay_pre_accum_import_error = f"pre_accum_compare_failed:{e}"
                                    else:
                                        replay_pre_accum_import_error = (
                                            f"pre_accum_size_mismatch:{replay_pre_accum_numel}!={pre_accum_snapshot_numel}"
                                        )
                                else:
                                    replay_pre_accum_import_error = "pre_accum_export_unavailable"
                        replay_loss_out = ctypes.c_float(0.0)
                        if replay_has_step_ex:
                            replay_calls = int(
                                replay_lib.ck_train_step_ex(
                                    x_buf,
                                    y_buf,
                                    ctypes.c_int(int(valid_tokens)),
                                    ctypes.byref(replay_loss_out),
                                    ctypes.c_float(lr),
                                )
                            )
                        else:
                            replay_calls = int(replay_lib.ck_train_step(x_buf, y_buf, ctypes.byref(replay_loss_out), ctypes.c_float(lr)))
                        if replay_calls < 0:
                            raise RuntimeError(f"ck_train_step replay failed at step {step} (calls={replay_calls})")
                        replay_loss = float(replay_loss_out.value)
                        replay_diff = abs(replay_loss - loss_val)
                        if replay_has_get_accum_counter_api:
                            replay_post_accum_counter = int(replay_lib.ck_train_get_accum_counter())
                        if replay_has_get_opt_step_api:
                            replay_post_opt_step = int(replay_lib.ck_train_get_opt_step())

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

                        if has_optimizer_state_snapshot_api and replay_optimizer_state_error is None:
                            if post_optimizer_state_snapshot is not None and post_optimizer_state_snapshot_numel > 0:
                                replay_post_opt = _ck_export_runtime_optimizer_state_snapshot(replay_lib)
                                if replay_post_opt is not None:
                                    replay_post_opt_buf, replay_post_opt_numel = replay_post_opt
                                    if int(replay_post_opt_numel) == int(post_optimizer_state_snapshot_numel):
                                        try:
                                            import numpy as _np
                                            ck_post_opt_np = _np.ctypeslib.as_array(
                                                post_optimizer_state_snapshot,
                                                shape=(int(post_optimizer_state_snapshot_numel),),
                                            ).astype(_np.float32, copy=False)
                                            replay_post_opt_np = _np.ctypeslib.as_array(
                                                replay_post_opt_buf,
                                                shape=(int(replay_post_opt_numel),),
                                            ).astype(_np.float32, copy=False)
                                            opt_delta_np = _np.abs(ck_post_opt_np - replay_post_opt_np)
                                            replay_optimizer_state_max_abs_diff = float(_np.max(opt_delta_np)) if opt_delta_np.size else 0.0
                                            replay_optimizer_state_mean_abs_diff = float(_np.mean(opt_delta_np)) if opt_delta_np.size else 0.0
                                        except Exception as e:
                                            replay_optimizer_state_error = f"replay_optimizer_state_compare_failed:{e}"
                                    else:
                                        replay_optimizer_state_error = (
                                            f"replay_post_optimizer_state_snapshot_size_mismatch:"
                                            f"{replay_post_opt_numel}!={post_optimizer_state_snapshot_numel}"
                                        )
                                else:
                                    replay_optimizer_state_error = "replay_post_optimizer_state_snapshot_unavailable"
                            else:
                                replay_optimizer_state_error = "post_optimizer_state_snapshot_unavailable"

                        if has_accum_snapshot_api and replay_accum_snapshot_error is None:
                            if post_accum_snapshot is not None and post_accum_snapshot_numel >= 0:
                                replay_post_accum = _ck_export_runtime_accum_snapshot(replay_lib)
                                if replay_post_accum is not None:
                                    replay_post_accum_buf, replay_post_accum_numel = replay_post_accum
                                    if int(replay_post_accum_numel) == int(post_accum_snapshot_numel):
                                        try:
                                            import numpy as _np
                                            ck_post_accum_np = _np.ctypeslib.as_array(
                                                post_accum_snapshot,
                                                shape=(int(post_accum_snapshot_numel),),
                                            ).astype(_np.float32, copy=False)
                                            replay_post_accum_np = _np.ctypeslib.as_array(
                                                replay_post_accum_buf,
                                                shape=(int(replay_post_accum_numel),),
                                            ).astype(_np.float32, copy=False)
                                            accum_delta_np = _np.abs(ck_post_accum_np - replay_post_accum_np)
                                            replay_accum_snapshot_max_abs_diff = float(_np.max(accum_delta_np)) if accum_delta_np.size else 0.0
                                            replay_accum_snapshot_mean_abs_diff = float(_np.mean(accum_delta_np)) if accum_delta_np.size else 0.0
                                        except Exception as e:
                                            replay_accum_snapshot_error = f"replay_accum_snapshot_compare_failed:{e}"
                                    else:
                                        replay_accum_snapshot_error = (
                                            f"replay_post_accum_snapshot_size_mismatch:"
                                            f"{replay_post_accum_numel}!={post_accum_snapshot_numel}"
                                        )
                                else:
                                    replay_accum_snapshot_error = "replay_post_accum_snapshot_unavailable"
                            else:
                                replay_accum_snapshot_error = "post_accum_snapshot_unavailable"

                        replay_optimizer_state_ok = True
                        if has_optimizer_state_snapshot_api:
                            replay_optimizer_state_ok = bool(
                                (replay_optimizer_state_error is None)
                                and (replay_optimizer_state_max_abs_diff is not None)
                                and (float(replay_optimizer_state_max_abs_diff) <= float(replay_optimizer_state_tol))
                            )
                        replay_accum_snapshot_ok = True
                        if has_accum_snapshot_api:
                            replay_accum_snapshot_ok = bool(
                                (replay_accum_snapshot_error is None)
                                and (replay_accum_snapshot_max_abs_diff is not None)
                                and (float(replay_accum_snapshot_max_abs_diff) <= float(replay_accum_snapshot_tol))
                            )

                        replay_ok = bool(
                            (replay_diff is not None)
                            and (replay_diff <= parity_replay_tol)
                            and (replay_weight_error is None)
                            and (replay_weight_max_abs_diff is not None)
                            and (float(replay_weight_max_abs_diff) <= float(replay_weight_tol))
                            and replay_optimizer_state_ok
                            and replay_accum_snapshot_ok
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
                                "optimizer_state_max_abs_diff": replay_optimizer_state_max_abs_diff,
                                "optimizer_state_mean_abs_diff": replay_optimizer_state_mean_abs_diff,
                                "optimizer_state_threshold": replay_optimizer_state_tol,
                                "optimizer_state_error": replay_optimizer_state_error,
                                "accum_snapshot_max_abs_diff": replay_accum_snapshot_max_abs_diff,
                                "accum_snapshot_mean_abs_diff": replay_accum_snapshot_mean_abs_diff,
                                "accum_snapshot_threshold": replay_accum_snapshot_tol,
                                "accum_snapshot_error": replay_accum_snapshot_error,
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
                            "optimizer_state_max_abs_diff": None,
                            "optimizer_state_mean_abs_diff": None,
                            "optimizer_state_threshold": replay_optimizer_state_tol,
                            "optimizer_state_error": ("skipped_due_to_weight_import_failure" if has_optimizer_state_snapshot_api else None),
                            "accum_snapshot_max_abs_diff": None,
                            "accum_snapshot_mean_abs_diff": None,
                            "accum_snapshot_threshold": replay_accum_snapshot_tol,
                            "accum_snapshot_error": ("skipped_due_to_weight_import_failure" if has_accum_snapshot_api else None),
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
                        "optimizer_state_max_abs_diff": None,
                        "optimizer_state_mean_abs_diff": None,
                        "optimizer_state_threshold": replay_optimizer_state_tol,
                        "optimizer_state_error": ("skipped_due_to_weight_snapshot_unavailable" if has_optimizer_state_snapshot_api else None),
                        "accum_snapshot_max_abs_diff": None,
                        "accum_snapshot_mean_abs_diff": None,
                        "accum_snapshot_threshold": replay_accum_snapshot_tol,
                        "accum_snapshot_error": ("skipped_due_to_weight_snapshot_unavailable" if has_accum_snapshot_api else None),
                        "reason": "snapshot_unavailable",
                    })

            oracle_loss = None
            oracle_error = None
            oracle_logits_max_abs_diff = None
            oracle_logits_slot = None
            oracle_first_bad_tensor = None
            oracle_first_bad_diff = None
            oracle_first_bad_op = None
            oracle_top_tensor = None
            oracle_top_diff = None
            oracle_top_op = None
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

                                if (oracle_top_diff is None) or (float(diff) > float(oracle_top_diff)):
                                    oracle_top_tensor = name
                                    oracle_top_diff = float(diff)
                                    oracle_top_op = _slot_op_hint_from_name(name)

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
                        failure_modes = []
                        if fail_loss:
                            failure_modes.append("loss")
                        if fail_logits:
                            failure_modes.append("logits")
                        if fail_slots:
                            failure_modes.append("slots")
                        if fail_loss and (not fail_logits) and (not fail_slots):
                            drift_signature = "loss_only"
                        elif fail_logits and (not fail_slots):
                            drift_signature = "logits"
                        elif fail_slots:
                            drift_signature = "tensor_slot"
                        else:
                            drift_signature = "mixed"
                        if dump_on_drift and pre_snapshot is not None and pre_snapshot_numel > 0:
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
                        if dump_on_drift and has_act_snapshot_export:
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
                        if dump_on_drift and pre_optimizer_state_snapshot is not None and pre_optimizer_state_snapshot_numel > 0:
                            drift_opt_path = _write_ck_optimizer_state_snapshot_artifact(
                                run_dir,
                                step,
                                pre_optimizer_state_snapshot,
                                pre_optimizer_state_snapshot_numel,
                                reason="parity_drift",
                            )
                            if drift_opt_path is not None:
                                optimizer_snapshot_artifacts.append(str(drift_opt_path))
                        if dump_on_drift and pre_accum_snapshot is not None and pre_accum_snapshot_numel >= 0:
                            drift_accum_path = _write_ck_accum_snapshot_artifact(
                                run_dir,
                                step,
                                pre_accum_snapshot,
                                pre_accum_snapshot_numel,
                                reason="parity_drift",
                            )
                            if drift_accum_path is not None:
                                accum_snapshot_artifacts.append(str(drift_accum_path))
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
                            "failure_modes": failure_modes,
                            "drift_signature": drift_signature,
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
                    "epoch": epoch_num,
                    "source_stage": source_stage,
                    "micro_steps": 1,
                    "tokens": len(x_vals),
                    "loss_ck": loss_val,
                    "loss_pt": oracle_loss if oracle_loss is not None else loss_val,
                    "logits_max_abs_diff": oracle_logits_max_abs_diff,
                    "logits_slot": oracle_logits_slot,
                    "first_bad_tensor": oracle_first_bad_tensor,
                    "first_bad_diff": oracle_first_bad_diff,
                    "first_bad_op": oracle_first_bad_op,
                    "top_tensor": oracle_top_tensor,
                    "top_diff": oracle_top_diff,
                    "top_op": oracle_top_op,
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
                    "top_tensor": oracle_top_tensor,
                    "top_diff": oracle_top_diff,
                    "top_op": oracle_top_op,
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
                    "replay_optimizer_state_max_abs_diff": replay_optimizer_state_max_abs_diff,
                    "replay_optimizer_state_mean_abs_diff": replay_optimizer_state_mean_abs_diff,
                    "replay_optimizer_state_threshold": replay_optimizer_state_tol,
                    "replay_optimizer_state_error": replay_optimizer_state_error,
                    "replay_accum_snapshot_max_abs_diff": replay_accum_snapshot_max_abs_diff,
                    "replay_accum_snapshot_mean_abs_diff": replay_accum_snapshot_mean_abs_diff,
                    "replay_accum_snapshot_threshold": replay_accum_snapshot_tol,
                    "replay_accum_snapshot_error": replay_accum_snapshot_error,
                    "pre_accum_counter": pre_replay_accum_counter,
                    "post_accum_counter": accum_now,
                    "pre_opt_step": pre_replay_opt_step,
                    "post_opt_step": opt_step_now,
                    "replay_post_accum_counter": replay_post_accum_counter,
                    "replay_post_opt_step": replay_post_opt_step,
                    "replay_has_set_accum_counter_api": replay_has_set_accum_counter_api,
                    "replay_has_get_accum_counter_api": replay_has_get_accum_counter_api,
                    "replay_has_set_opt_step_api": replay_has_set_opt_step_api,
                    "replay_has_get_opt_step_api": replay_has_get_opt_step_api,
                    "replay_pre_optimizer_state_import_max_abs_diff": replay_pre_optimizer_state_import_max_abs_diff,
                    "replay_pre_optimizer_state_import_error": replay_pre_optimizer_state_import_error,
                    "replay_pre_accum_import_max_abs_diff": replay_pre_accum_import_max_abs_diff,
                    "replay_pre_accum_import_error": replay_pre_accum_import_error,
                }
            )
            grad_steps.append(step)
            grad_global.append(0.0)

        if epoch_rows_sampled > 0:
            rows_total = train_data_source.get("rows_total")
            if not isinstance(rows_total, int) or rows_total <= 0:
                rows_total = len(batches)
            coverage_pct = None
            if isinstance(rows_total, int) and rows_total > 0:
                coverage_pct = min(100.0, (float(epoch_rows_sampled) * 100.0) / float(rows_total))
            corpus_sampling_epochs.append(
                {
                    "epoch": epoch_num,
                    "stage_id": source_stage,
                    "step_start": epoch_step_start,
                    "step_end": int(step),
                    "loss_start": float(epoch_loss_start if epoch_loss_start is not None else 0.0),
                    "loss_end": float(epoch_loss_end if epoch_loss_end is not None else 0.0),
                    "rows_sampled": int(epoch_rows_sampled),
                    "tokens_consumed": int(epoch_tokens_consumed),
                    "datasets": [
                        {
                            "dataset_id": _resolve_dataset_id_from_source(train_data_source),
                            "label": str(train_data_source.get("dataset_name") or "train_data"),
                            "path": train_data_source.get("source_path"),
                            "rows_sampled": int(epoch_rows_sampled),
                            "rows_total": int(rows_total) if isinstance(rows_total, int) and rows_total > 0 else None,
                            "tokenized": True,
                            "coverage_pct": float(coverage_pct) if coverage_pct is not None else None,
                        }
                    ],
                }
            )

    pending_accum = int(lib.ck_train_get_accum_counter()) if has_accum_counter_api else int(micro_steps % grad_accum)
    if pending_accum > 0:
        if not has_flush_optimizer_api:
            raise RuntimeError(
                f"ck_train runtime ended with pending grad accumulation ({pending_accum}) but ck_train_flush_optimizer is unavailable"
            )
        t0 = time.perf_counter()
        flush_calls = int(lib.ck_train_flush_optimizer(ctypes.c_float(lr)))
        t1 = time.perf_counter()
        if flush_calls < 0:
            raise RuntimeError(f"ck_train_flush_optimizer failed after step {step} (calls={flush_calls})")
        optimizer_steps += 1
        total_ck_ms += (t1 - t0) * 1000.0

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
    replay_optimizer_state_max_values = [
        float(row.get("replay_optimizer_state_max_abs_diff"))
        for row in parity_steps
        if isinstance(row, dict) and row.get("replay_optimizer_state_max_abs_diff") is not None
    ]
    replay_optimizer_state_mean_values = [
        float(row.get("replay_optimizer_state_mean_abs_diff"))
        for row in parity_steps
        if isinstance(row, dict) and row.get("replay_optimizer_state_mean_abs_diff") is not None
    ]
    replay_accum_snapshot_max_values = [
        float(row.get("replay_accum_snapshot_max_abs_diff"))
        for row in parity_steps
        if isinstance(row, dict) and row.get("replay_accum_snapshot_max_abs_diff") is not None
    ]
    replay_accum_snapshot_mean_values = [
        float(row.get("replay_accum_snapshot_mean_abs_diff"))
        for row in parity_steps
        if isinstance(row, dict) and row.get("replay_accum_snapshot_mean_abs_diff") is not None
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

    if parity_on and parity_failures and dump_on_drift:
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
            "optimizer_snapshot_files": optimizer_snapshot_artifacts[:topk],
            "accum_snapshot_files": accum_snapshot_artifacts[:topk],
            "check_dump_files": check_dump_artifacts[:topk],
        }
        drift_path = run_dir / "drift_report.json"
        drift_path.write_text(json.dumps(drift_report, indent=2), encoding="utf-8")
        profile_meta.setdefault("artifacts", []).append({"label": "drift_report", "path": str(drift_path)})

    summary = {
        "run_id": str(run_dir.name),
        "epochs": epochs,
        "seq_len": seq_len,
        "total_tokens": total_tokens,
        "grad_accum": grad_accum,
        "seed": seed,
        "optimizer": optimizer,
        "optimizer_hparams": {
            "source": str(adamw_cfg.get("source") or "defaults"),
            "manifest": adamw_cfg.get("manifest"),
            "adamw": {
                "beta1": float(adamw_effective.get("beta1", 0.9)),
                "beta2": float(adamw_effective.get("beta2", 0.999)),
                "eps": float(adamw_effective.get("eps", 1e-8)),
                "weight_decay": float(adamw_effective.get("weight_decay", 0.01)),
            },
        },
        "lr": lr,
        "ck_loss_backend": ck_loss_backend,
        "runtime_ce_backend": runtime_ce_backend,
        "bitwise_parity": {
            "enabled": bool(bitwise_parity_enabled),
            "compile_flags": list(bitwise_compile_flags),
            "runtime_env": dict(bitwise_runtime_env),
            "strict_runtime_forced": bool(strict_runtime_enabled),
        },
        "steps": step,
        "micro_steps": micro_steps,
        "optimizer_steps": optimizer_steps,
        "tokens_per_update": seq_len * grad_accum,
        "max_loss_abs_diff": max_loss_abs_diff,
        "mean_loss_abs_diff": mean_loss_abs_diff,
        "final_ck_loss": final_ck_loss,
        "final_torch_loss": final_oracle_loss,
        "final_param_max_abs_diff": float(max(replay_weight_max_values) if replay_weight_max_values else 0.0),
        "final_param_mean_abs_diff": float((sum(replay_weight_mean_values) / len(replay_weight_mean_values)) if replay_weight_mean_values else 0.0),
        "final_optimizer_state_max_abs_diff": float(max(replay_optimizer_state_max_values) if replay_optimizer_state_max_values else 0.0),
        "final_optimizer_state_mean_abs_diff": float(
            (sum(replay_optimizer_state_mean_values) / len(replay_optimizer_state_mean_values))
            if replay_optimizer_state_mean_values else 0.0
        ),
        "final_accum_snapshot_max_abs_diff": float(max(replay_accum_snapshot_max_values) if replay_accum_snapshot_max_values else 0.0),
        "final_accum_snapshot_mean_abs_diff": float(
            (sum(replay_accum_snapshot_mean_values) / len(replay_accum_snapshot_mean_values))
            if replay_accum_snapshot_mean_values else 0.0
        ),
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
            "optimizer_steps": optimizer_steps,
            "tokens_per_update": seq_len * grad_accum,
            "processed_tokens": processed_tokens,
            "ck_total_ms": total_ck_ms,
            "torch_total_ms": 0.0,
            "ck_avg_step_ms": avg_ck_step_ms,
            "torch_avg_step_ms": 0.0,
            "train_tok_s": train_tok_s,
            "decode_tok_s": train_tok_s,
        },
        "train_dims": {
            "source": str(resolved_train_dims.get("source") or "cli"),
            "manifest": resolved_train_dims.get("manifest"),
            "requested": requested_train_dims,
            "effective": effective_train_dims,
            "mismatches": resolved_train_dims.get("mismatches"),
        },
        "data_source": train_data_source,
        "data_provenance": [
            {
                "stage": train_mode,
                "dataset_name": train_data_source.get("dataset_name"),
                "source_uri": train_data_source.get("source_uri"),
                "source_path": train_data_source.get("source_path"),
                "split": train_data_source.get("split"),
                "token_count": train_data_source.get("token_count"),
                "hash": {"algo": "sha256", "value": train_data_source.get("text_sha256")},
                "sampling": train_data_source.get("sampling"),
                "packing": train_data_source.get("packing"),
            }
        ],
        "tokenizer_lineage": tokenizer_lineage,
        "backend": train_backend,
        "train_mode": train_mode,
        "source": "ck_runtime_generated",
        "safety": dict(profile_meta.get("train_safety", {})) if isinstance(profile_meta, dict) else {},
        "runtime_init": {
            "num_params": int(init_payload.get("num_params", 0)),
            "total_floats": int(init_payload.get("total_floats", 0)),
            "runtime_num_tokens": int(runtime_num_tokens),
            "runtime_grad_accum_steps": int(runtime_grad_accum_steps),
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
            "runtime_strict_math": bool(strict_runtime_bound and strict_runtime_enabled),
            "available": bool(oracle_points > 0),
            "snapshot_torch_enabled": bool(snapshot_oracle_enabled),
            "snapshot_torch_error": snapshot_oracle_error,
            "failures": parity_failures,
            "replay_on_check": bool(parity_replay_on_check),
            "replay_auto_enabled": bool(replay_auto_enabled),
            "replay_tol": float(parity_replay_tol),
            "replay_weight_tol": float(replay_weight_tol),
            "replay_optimizer_state_tol": float(replay_weight_tol),
            "replay_accum_snapshot_tol": float(replay_weight_tol),
            "logits_tol": float(activation_tol),
            "bruteforce_debug": bool(bruteforce_debug),
            "dump_on_drift": bool(dump_on_drift),
            "dump_on_check": bool(dump_on_check),
            "dump_check_topk": int(dump_check_topk),
            "replay_failures": replay_failures,
            "snapshot_api_available": bool(has_weight_snapshot_api),
            "optimizer_state_snapshot_api_available": bool(has_optimizer_state_snapshot_api),
            "accum_snapshot_api_available": bool(has_accum_snapshot_api),
            "activation_snapshot_api_available": bool(has_act_snapshot_numel and has_act_snapshot_export),
            "replay_optimizer_state_snapshot_api_available": bool(replay_has_opt_state_snapshot_api),
            "replay_accum_snapshot_api_available": bool(replay_has_accum_snapshot_api),
            "replay_runtime_error": replay_runtime_error,
            "snapshot_files": snapshot_artifacts,
            "activation_snapshot_files": activation_snapshot_artifacts,
            "optimizer_snapshot_files": optimizer_snapshot_artifacts,
            "accum_snapshot_files": accum_snapshot_artifacts,
            "check_dump_files": check_dump_artifacts[:dump_check_topk],
        },
        "corpus_sampling_epochs": corpus_sampling_epochs,
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
    if optimizer_snapshot_artifacts:
        profile_meta.setdefault("artifacts", []).append({"label": "oracle_ck_optimizer_state", "path": str(run_dir / "oracle_ck_optimizer_state")})
    if accum_snapshot_artifacts:
        profile_meta.setdefault("artifacts", []).append({"label": "oracle_ck_accum", "path": str(run_dir / "oracle_ck_accum")})
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
    training_pipeline = _build_training_pipeline_payload(s, run_dir)
    corpus_sampling_log = _build_corpus_sampling_log_payload(s, run_dir)

    payloads = {
        "training_loss_curve.json": training_loss_curve,
        "training_parity.json": training_parity,
        "training_grad_norms.json": training_grad_norms,
        "training_step_profile.json": training_step_profile,
        "training_checkpoint_policy.json": training_checkpoint_policy,
        "training_pipeline.json": training_pipeline,
        "corpus_sampling_log.json": corpus_sampling_log,
        "corpus_sampling_log_latest.json": corpus_sampling_log,
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


def _run_ck_profile_via_cli(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    json_out: Path,
    profile_mode: str,
    profile_dir: Path,
    profile_meta: dict,
) -> Path:
    """Run native CK training profiler flow through ck-cli-v7 profile."""
    tool = str(profile_mode or "none").strip().lower()
    if tool not in {"perf", "vtune", "advisor", "cachegrind", "asan"}:
        raise ValueError(f"unsupported profile tool '{tool}'")

    token_file_arg = getattr(args, "train_token_file", None)
    if not token_file_arg:
        raise ValueError("--profile-train with backend=ck requires --train-token-file")
    token_file = Path(str(token_file_arg)).expanduser().resolve()
    if not token_file.exists():
        raise FileNotFoundError(f"Training token file not found: {token_file}")

    profile_dir.mkdir(parents=True, exist_ok=True)

    # Profilers like VTune refuse to reuse non-empty result dirs.
    stale_result_dirs = {
        "vtune": ("vtune_hotspots", "vtune_memory"),
        "advisor": ("advisor_run",),
    }.get(tool, ())
    for rel in stale_result_dirs:
        stale = profile_dir / rel
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)

    # Keep ck-cli-v7 current so profiling always reflects latest runtime plumbing.
    run_cmd(["make", "--no-print-directory", "ck-cli-v7"], cwd=PROJECT_ROOT)
    ck_cli = BUILD_DIR / "ck-cli-v7"
    if not ck_cli.exists():
        raise RuntimeError(f"Native profiler runner missing after build: {ck_cli}")

    cli_cmd = [
        str(ck_cli),
        "profile",
        "--run",
        str(run_dir),
        "--tool",
        tool,
        "--output-dir",
        str(profile_dir),
        "--train-token-file",
        str(token_file),
        "--train-epochs",
        str(int(getattr(args, "train_epochs", 3) or 3)),
        "--train-seq-len",
        str(int(getattr(args, "train_seq_len", 16) or 16)),
        "--train-total-tokens",
        str(int(getattr(args, "train_total_tokens", 1024) or 1024)),
        "--train-grad-accum",
        str(max(1, int(getattr(args, "train_grad_accum", 8) or 8))),
        "--train-lr",
        str(float(getattr(args, "train_lr", 1e-3) or 1e-3)),
    ]
    if bool(getattr(args, "train_strict", False)):
        cli_cmd.append("--train-strict")

    ck_threads = os.environ.get("CK_NUM_THREADS", "").strip()
    if ck_threads:
        try:
            threads_n = int(ck_threads)
            if threads_n > 0:
                cli_cmd.extend(["--threads", str(threads_n)])
        except ValueError:
            log(f"  Warning: ignoring invalid CK_NUM_THREADS='{ck_threads}' for ck-cli-v7 profile", C_ORANGE)

    log(f"  backend=ck profiler: ck-cli-v7 profile --tool {tool}", C_DIM)
    run_cmd(cli_cmd, cwd=PROJECT_ROOT)

    summary_src = run_dir / "train_e2e_latest.json"
    if not summary_src.exists():
        raise RuntimeError(f"ck-cli-v7 profile completed but summary missing: {summary_src}")

    json_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        same = summary_src.resolve() == json_out.resolve()
    except Exception:
        same = str(summary_src) == str(json_out)
    if not same:
        shutil.copy2(summary_src, json_out)

    profile_meta["driver"] = "ck-cli-v7 profile"
    profile_meta.setdefault("artifacts", []).append({"label": "ck_cli_profile_output_dir", "path": str(profile_dir)})

    # Persist profiler summaries + primary human-readable artifacts for dashboards.
    summary_candidates = {
        "perf": ["perf_stat_summary.json", "flamegraph_manifest.json"],
        "vtune": ["vtune_summary.json"],
        "advisor": ["advisor_summary.json"],
        "cachegrind": ["cachegrind_summary.json"],
        "asan": ["asan_summary.json", "memory_verification_latest.json", "memory_diagnostic_latest.json"],
    }.get(tool, [])
    for name in summary_candidates:
        p = run_dir / name
        if p.exists():
            profile_meta.setdefault("artifacts", []).append({"label": name, "path": str(p)})

    for name in ("advisor_roofline.html", "advisor_roofline.txt", "advisor_roofline.csv"):
        p = profile_dir / name
        if p.exists():
            profile_meta.setdefault("artifacts", []).append({"label": name, "path": str(p)})
    for name in ("vtune_hotspots.txt", "vtune_hotspots.csv", "vtune_memory_summary.txt", "vtune_memory_summary.csv"):
        p = profile_dir / name
        if p.exists():
            profile_meta.setdefault("artifacts", []).append({"label": name, "path": str(p)})

    return json_out


def _maybe_run_training_parity_regimen(args: argparse.Namespace, *, run_dir: Optional[Path]) -> None:
    """Optionally suggest/run/require staged training parity regimen after train-e2e."""
    mode = str(getattr(args, "parity_regimen", "suggest") or "suggest").strip().lower()
    if mode not in {"off", "suggest", "run", "require"}:
        mode = "suggest"
    if mode == "off":
        return

    regimen_script = SCRIPTS_DIR / "run_training_parity_regimen_v7.py"
    if not regimen_script.exists():
        msg = f"training parity regimen script missing: {regimen_script}"
        if mode == "require":
            log_error(msg)
            sys.exit(2)
        log(f"  Warning: {msg}", C_ORANGE)
        return

    if run_dir is None:
        if mode == "suggest":
            log("  parity regimen: skipped suggestion (no --run dir set for this train command)", C_DIM)
            return
        msg = "--parity-regimen run/require needs --run <run_dir> so artifacts can be persisted per run"
        if mode == "require":
            log_error(msg)
            sys.exit(2)
        log(f"  Warning: {msg}", C_ORANGE)
        return

    regimen_json = run_dir / "training_parity_regimen_latest.json"
    regimen_md = run_dir / "training_parity_regimen_latest.md"
    parity_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(parity_python) if parity_python.exists() else sys.executable
    cmd = [
        python_exec,
        str(regimen_script),
        "--run-dir",
        str(run_dir),
        "--json-out",
        str(regimen_json),
        "--md-out",
        str(regimen_md),
    ]

    if mode == "suggest":
        if regimen_json.exists():
            try:
                payload = json.loads(regimen_json.read_text(encoding="utf-8"))
                summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
                passed = bool(summary.get("passed", False))
                if passed:
                    log(f"  parity regimen: present/pass ({regimen_json})", C_DIM)
                else:
                    log(f"  parity regimen: present but not passing ({regimen_json})", C_ORANGE)
                    log(f"  parity regimen: rerun with --parity-regimen run or command below", C_DIM)
                    log(f"    {' '.join(cmd)}", C_DIM)
            except Exception:
                log(f"  parity regimen: unreadable report at {regimen_json}; rerun recommended", C_ORANGE)
                log(f"    {' '.join(cmd)}", C_DIM)
        else:
            log("  parity regimen: not found for this run (recommended before long CK-only training)", C_ORANGE)
            log("  parity regimen options:", C_DIM)
            log("    --parity-regimen run      (run now, continue on failure)", C_DIM)
            log("    --parity-regimen require  (run now, fail command if regimen fails)", C_DIM)
            log(f"  parity regimen command: {' '.join(cmd)}", C_DIM)
        return

    log(f"  parity regimen: running ({mode})", C_DIM)
    rc = run_cmd_allow_fail(cmd, cwd=PROJECT_ROOT).returncode
    if rc == 0:
        log(f"  parity regimen: PASS ({regimen_json})", C_GREEN)
        return

    if mode == "require":
        log_error(f"parity regimen failed (mode=require): {regimen_json}")
        sys.exit(1)
    log(f"  Warning: parity regimen failed (mode=run), continuing: {regimen_json}", C_ORANGE)


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

    train_dims = _resolve_train_dims_for_run(args, run_dir)
    train_vocab = int(train_dims["effective"]["vocab"])
    train_d_model = int(train_dims["effective"]["d_model"])
    train_hidden = int(train_dims["effective"]["hidden"])
    train_num_layers = int(train_dims["effective"]["num_layers"])
    if train_dims.get("mismatches"):
        mismatch_txt = ", ".join(
            f"{k}:{v.get('requested')}->{v.get('effective')}"
            for k, v in (train_dims.get("mismatches") or {}).items()
            if isinstance(v, dict)
        )
        manifest_src = str(train_dims.get("manifest") or "")
        log(
            f"  train dims: using run-dir manifest ({mismatch_txt})"
            + (f" [{manifest_src}]" if manifest_src else ""),
            C_ORANGE,
        )
    train_loss_tol = float(getattr(args, "train_loss_tol", 2e-5) or 2e-5)
    train_param_tol = float(getattr(args, "train_param_tol", 3e-5) or 3e-5)
    train_max_grad_norm = float(getattr(args, "train_max_grad_norm", 0.0) or 0.0)
    try:
        train_adamw_cfg = _resolve_train_adamw_hparams(args, run_dir)
    except ValueError as e:
        log_error(str(e))
        sys.exit(1)
    train_adamw_effective = dict(train_adamw_cfg.get("effective") or {})
    train_safety = _assess_train_safety(args, train_backend)

    cmd = [
        python_exec,
        str(train_script),
        "--epochs", str(getattr(args, "train_epochs", 3)),
        "--seq-len", str(getattr(args, "train_seq_len", 16)),
        "--total-tokens", str(getattr(args, "train_total_tokens", 1024)),
        "--grad-accum", str(getattr(args, "train_grad_accum", 8)),
        "--optimizer", str(getattr(args, "train_optimizer", "adamw")),
        "--lr", str(getattr(args, "train_lr", 1e-3)),
        "--adamw-beta1", str(float(train_adamw_effective.get("beta1", 0.9))),
        "--adamw-beta2", str(float(train_adamw_effective.get("beta2", 0.999))),
        "--adamw-eps", str(float(train_adamw_effective.get("eps", 1e-8))),
        "--adamw-weight-decay", str(float(train_adamw_effective.get("weight_decay", 0.01))),
        "--seed", str(getattr(args, "train_seed", 42)),
        "--vocab", str(train_vocab),
        "--d-model", str(train_d_model),
        "--hidden", str(train_hidden),
        "--num-layers", str(train_num_layers),
        "--loss-tol", str(train_loss_tol),
        "--param-tol", str(train_param_tol),
        "--max-grad-norm", str(train_max_grad_norm),
        "--unsafe-adamw-lr-threshold", str(float(getattr(args, "train_unsafe_adamw_lr_threshold", 1e-3) or 1e-3)),
        "--ck-loss-backend", str(getattr(args, "ck_loss_backend", "c") or "c"),
        "--json-out", str(json_out),
    ]
    if bool(train_safety.get("enforce_production_safety")):
        cmd.append("--enforce-production-safety")
    if bool(train_safety.get("allow_unsafe_adamw_lr")):
        cmd.append("--allow-unsafe-adamw-lr")

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
        f"  d_model={train_d_model} hidden={train_hidden} vocab={train_vocab} layers={train_num_layers} "
        f"grad_accum={getattr(args, 'train_grad_accum', 8)} optimizer={getattr(args, 'train_optimizer', 'adamw')}",
        C_DIM,
    )
    if str(getattr(args, "train_optimizer", "adamw") or "adamw").lower() == "adamw":
        log(
            "  adamw: "
            f"beta1={float(train_adamw_effective.get('beta1', 0.9))} "
            f"beta2={float(train_adamw_effective.get('beta2', 0.999))} "
            f"eps={float(train_adamw_effective.get('eps', 1e-8))} "
            f"weight_decay={float(train_adamw_effective.get('weight_decay', 0.01))} "
            f"source={train_adamw_cfg.get('source', 'defaults')}",
            C_DIM,
        )
    log(
        f"  train safety: status={train_safety.get('status')} "
        f"lr={train_safety.get('lr')} max_grad_norm={train_safety.get('max_grad_norm')} "
        f"threshold={train_safety.get('unsafe_adamw_lr_threshold')}",
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
    bruteforce_debug = bool(getattr(args, "bruteforce_debug", False))
    dump_on_check = bool(getattr(args, "dump_on_check", False))
    dump_check_topk = max(1, int(getattr(args, "dump_check_topk", 200) or 200))
    train_save_every = int(getattr(args, "train_save_every", 0) or 0)
    train_save_final = bool(getattr(args, "train_save_final", True))
    if bruteforce_debug:
        log("  generated-runtime brute-force debug: on", C_DIM)
    if parity_on:
        cadence = f"every={parity_every}" if parity_every > 0 else f"profile={parity_profile}"
        log(f"  parity oracle: on ({oracle}, {cadence})", C_DIM)
    else:
        log("  parity oracle: off", C_DIM)
    if bool(getattr(args, "kernel_strict_math", False)):
        log("  kernel strict math: on", C_DIM)
    if bool(getattr(args, "bitwise_parity", False)):
        if train_backend == "ck":
            log("  bitwise parity mode: on", C_DIM)
        else:
            log("  bitwise parity mode: ignored (backend must be ck)", C_ORANGE)
    if bool(getattr(args, "dump_on_drift", False)):
        log(f"  drift dumps: on (topk={int(getattr(args, 'drift_topk', 20) or 20)})", C_DIM)
    if dump_on_check:
        log(f"  check dumps: on (metadata topk={dump_check_topk})", C_DIM)
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
        "bruteforce_debug": bruteforce_debug,
        "bitwise_parity": bool(getattr(args, "bitwise_parity", False)),
        "dump_on_drift": bool(getattr(args, "dump_on_drift", False)),
        "dump_on_check": dump_on_check,
        "dump_check_topk": dump_check_topk,
        "drift_topk": int(getattr(args, "drift_topk", 20) or 20),
        "analysis_checkpoints": analysis_mode,
        "train_save_every": train_save_every,
        "train_save_final": train_save_final,
        "train_safety": train_safety,
        "artifacts": [],
    }

    def _note_artifact(label: str, path: Path) -> None:
        profile_meta["artifacts"].append({"label": label, "path": str(path)})

    if train_backend == "ck":
        if run_dir is None:
            log_error("--backend ck requires --run <run_dir> so runtime artifacts can be generated")
            sys.exit(2)
        if profile_mode in ("perf", "vtune", "advisor", "cachegrind", "asan"):
            try:
                _run_ck_profile_via_cli(
                    args=args,
                    run_dir=run_dir,
                    json_out=Path(json_out),
                    profile_mode=profile_mode,
                    profile_dir=profile_dir,
                    profile_meta=profile_meta,
                )
            except Exception as e:
                log_error(f"CK profiler dispatch failed (--profile-train={profile_mode}): {e}")
                sys.exit(2)
        else:
            if profile_mode not in ("none", ""):
                log(f"  Warning: unknown --profile-train mode '{profile_mode}', using none", C_ORANGE)
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
        elif profile_mode == "advisor":
            log("  Warning: --profile-train=advisor is only wired for backend=ck; running without external profiler", C_ORANGE)
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
    _maybe_run_training_parity_regimen(args, run_dir=run_dir)
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
    # Keep default run dirs under the cache tree so IR + dataset + train artifacts stay together.
    default_train_root = DEFAULT_TRAIN_ROOT
    out_dir = Path(out_dir_arg) if out_dir_arg else (default_train_root / run_name)

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
        "--adamw-beta1", str(getattr(args, "adamw_beta1", 0.9)),
        "--adamw-beta2", str(getattr(args, "adamw_beta2", 0.999)),
        "--adamw-eps", str(getattr(args, "adamw_eps", 1e-8)),
        "--adamw-weight-decay", str(getattr(args, "adamw_weight_decay", 0.01)),
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

        exec_plan_script = SCRIPTS_DIR / "generate_train_exec_plan_v7.py"
        exec_plan_out = out_dir / "train_exec_plan.json"
        exec_plan_cmd = [
            python_exec,
            str(exec_plan_script),
            "--ir2", str(ir2_out),
            "--output", str(exec_plan_out),
            "--mode", "deterministic",
        ]
        run_cmd(exec_plan_cmd, cwd=PROJECT_ROOT)

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
                "--exec-plan", str(exec_plan_out),
                "--output", str(rt_out),
                "--summary-out", str(rt_summary),
            ]
            run_cmd(rt_cmd, cwd=PROJECT_ROOT)

        log(f"  Generated train IR: {ir1_out}", C_GREEN)
        log(f"  Generated backward IR: {ir2_out}", C_GREEN)
        log(f"  Generated training layout: {layout_out}", C_GREEN)
        log(f"  Training memory audit: {layout_audit_out}", C_GREEN)
        log(f"  Generated train exec plan: {exec_plan_out}", C_GREEN)


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

    dataset_workspace = getattr(args, "dataset_workspace", None)
    if dataset_workspace:
        stage_script = SCRIPTS_DIR / "dataset" / "stage_dataset_workspace_v7.py"
        if not stage_script.exists():
            log_error(f"Dataset staging script not found: {stage_script}")
            sys.exit(1)
        # Repo workspaces are seed templates only. The operator-facing working copy must
        # live under the same cache run-dir as IR, checkpoints, and training telemetry.
        dataset_cmd = [
            python_exec,
            str(stage_script),
            "--workspace", str(Path(dataset_workspace)),
            "--run-dir", str(out_dir),
            "--mode", str(getattr(args, "dataset_stage_mode", "copy")),
        ]
        if getattr(args, "dataset_stage_force", False):
            dataset_cmd.append("--force")
        run_cmd(dataset_cmd, cwd=PROJECT_ROOT)
        log(f"  Dataset snapshot: {out_dir / 'dataset'}", C_GREEN)
        log(f"  Dataset viewer: {out_dir / 'dataset_viewer.html'}", C_GREEN)

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
      - <run_dir>/train_e{E}.json for each sweep epoch (or <cache>/reports if --run not set)
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
    if getattr(args, "top_k", None) is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if getattr(args, "top_p", None) is not None:
        cmd.extend(["--top-p", str(args.top_p)])
    if getattr(args, "min_p", None) is not None:
        cmd.extend(["--min-p", str(args.min_p)])
    if getattr(args, "repeat_penalty", None) is not None:
        cmd.extend(["--repeat-penalty", str(args.repeat_penalty)])
    if getattr(args, "repeat_last_n", None) is not None:
        cmd.extend(["--repeat-last-n", str(args.repeat_last_n)])
    # Keep ck_run compatible with baseline ck_chat.py CLI.
    # Optional anti-markup controls are only available in local experimental ck_chat variants.
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
    # Debug stop-points from the caller shell can short-circuit decode before dumps.
    env.pop("CK_STOP_OP", None)

    # Need at least one decode step after prefill to emit decode-path parity dumps.
    if max_tokens <= 1:
        max_tokens = 2

    cmd = [
        str(cli_path),
        str(lib_path),
        str(weights_path),
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--no-chat-template",
        "--ignore-eos",
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
    parity_family: str = "llama",
    require_token_aware_dumps: bool = False,
    allow_raw_fallback: bool = True,
) -> bool:
    """Run llama.cpp parity binary to generate reference dumps."""
    log(f"\n{C_ORANGE}[llamacpp-parity]{C_RESET} Running llama.cpp for reference dumps")

    # Prefer patched parity binary; then common llama.cpp executable locations.
    llm_candidates = [
        PROJECT_ROOT / "build" / "llama-parity",
        PROJECT_ROOT / "llama.cpp" / "build" / "bin" / "llama-completion",
        PROJECT_ROOT / "llama.cpp" / "build" / "bin" / "llama-cli",
        PROJECT_ROOT / "llama.cpp" / "main",
        PROJECT_ROOT / "llama.cpp" / "build" / "bin" / "main",
    ]
    llm_path = next((p for p in llm_candidates if p.exists()), None)
    if llm_path is None:
        log_error(
            "llama.cpp parity binary not found "
            "(expected build/llama-parity, llama.cpp/build/bin/llama-cli, "
            "llama.cpp/build/bin/llama-completion, or llama.cpp/main)"
        )
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
    ref_dump = ref_dump_dir / "dump.bin"
    ref_index = ref_dump_dir / "index.json"

    def _purge_ref_dump() -> None:
        for p in (ref_dump, ref_index):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    # Never reuse stale reference dumps across runs.
    _purge_ref_dump()

    def _token_audit(index_path: Path) -> tuple[bool, str]:
        if not index_path.exists():
            return False, "missing index.json"
        try:
            obj = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"index parse error: {e}"
        if not isinstance(obj, list):
            return False, "index is not a list"
        token_ids: list[int] = []
        for row in obj:
            if not isinstance(row, dict):
                continue
            if "token_id" not in row:
                continue
            try:
                token_ids.append(int(row.get("token_id", 0)))
            except Exception:
                continue
        if not token_ids:
            return False, "index has no token_id entries"
        unique = sorted(set(token_ids))
        # Raw fallback conversions typically collapse everything to token_id=0.
        collapsed_zero = len(token_ids) >= 8 and len(unique) == 1 and unique[0] == 0
        if collapsed_zero:
            return False, f"collapsed token ids (all zero across {len(token_ids)} dumps)"
        return True, f"entries={len(token_ids)} unique_tokens={unique[:16]}"

    def _looks_like_flag_error(stderr_text: str) -> bool:
        s = (stderr_text or "").lower()
        needles = (
            "unknown argument",
            "unrecognized option",
            "invalid option",
            "unknown option",
            "did you mean",
        )
        return any(tok in s for tok in needles)

    def _short_tail(text: str, limit: int = 500) -> str:
        if not text:
            return ""
        return text[-limit:]

    def _run_once(
        run_cmd: list[str],
        *,
        run_env: dict[str, str],
        run_cwd: Path,
        timeout_sec: Optional[int],
    ) -> tuple[Optional[subprocess.CompletedProcess[str]], Optional[str]]:
        try:
            proc = subprocess.run(
                run_cmd,
                cwd=run_cwd,
                env=run_env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return proc, None
        except subprocess.TimeoutExpired:
            return None, "timeout"
        except Exception as e:
            return None, str(e)

    def _is_fresh_nonempty(path: Path, started_at: float, *, slack_sec: float = 1.5) -> bool:
        try:
            if not path.exists():
                return False
            st = path.stat()
            if st.st_size <= 0:
                return False
            return st.st_mtime >= (started_at - slack_sec)
        except Exception:
            return False

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
    ctx_arg = int(ctx_size) if (ctx_size is not None and int(ctx_size) > 0) else 0
    if ctx_arg <= 0:
        raw_default_ctx = str(os.environ.get("CK_LLAMA_PARITY_DEFAULT_CTX", "1024")).strip()
        try:
            ctx_arg = int(raw_default_ctx)
        except Exception:
            ctx_arg = 1024
        if ctx_arg > 0:
            log(
                f"  llama parity ctx-size not set; using -c {ctx_arg} "
                f"(override with --context-len or CK_LLAMA_PARITY_DEFAULT_CTX)",
                C_DIM,
            )

    exe_name = llm_path.name.lower()
    cmd = [
        str(llm_path),
        "-m",
        str(gguf_path),
        "-p",
        prompt,
    ]
    # Keep llama parity invocations non-interactive across binary variants.
    if "llama-cli" in exe_name:
        cmd.extend(["-no-cnv", "--single-turn", "--simple-io"])
    elif ("llama-completion" in exe_name) or (exe_name == "main") or ("llama-parity" in exe_name):
        cmd.extend(["-no-cnv", "--simple-io"])
    else:
        cmd.extend(["--simple-io"])
    cmd.extend(
        [
            "--no-warmup",
            "--temp",
            f"{temp}",
            "-n",
            str(max_tokens),
        ]
    )
    if ctx_arg > 0:
        cmd.extend(["-c", str(ctx_arg)])

    timeout_sec = 600 if llama_timeout is None else int(llama_timeout)
    if timeout_sec <= 0:
        timeout_sec = None

    attempt_diag: list[dict[str, Any]] = []
    _purge_ref_dump()
    ckdmp_started = time.time()
    result, run_err = _run_once(cmd, run_env=env, run_cwd=work_dir, timeout_sec=timeout_sec)
    if run_err is not None:
        log_error(f"llama.cpp parity run failed: {run_err}")
        return False
    if result is None:
        log_error("llama.cpp parity run failed (no process result)")
        return False
    attempt_diag.append(
        {
            "phase": "ckdmp",
            "cwd": str(work_dir),
            "cmd": " ".join(shlex.quote(x) for x in cmd),
            "rc": int(result.returncode),
            "stderr_tail": _short_tail(result.stderr, 360),
        }
    )

    # Retry with compatibility flags only when the binary explicitly rejects args.
    if result.returncode != 0 and _looks_like_flag_error(result.stderr):
        flag_variants: list[list[str]] = []
        if "--simple-io" in cmd:
            flag_variants.append([x for x in cmd if x != "--simple-io"])
        if "--no-warmup" in cmd:
            flag_variants.append([x for x in cmd if x != "--no-warmup"])
        if "-no-cnv" in cmd:
            flag_variants.append([x for x in cmd if x != "-no-cnv"])
        if "--single-turn" in cmd:
            flag_variants.append([x for x in cmd if x != "--single-turn"])
        # Last resort: minimal prompt-only invocation.
        minimal = [str(llm_path), "-m", str(gguf_path), "-p", prompt, "--temp", f"{temp}", "-n", str(max_tokens)]
        if ctx_arg > 0:
            minimal.extend(["-c", str(ctx_arg)])
        flag_variants.append(minimal)

        seen_variants: set[tuple[str, ...]] = set()
        for alt_cmd in flag_variants:
            key = tuple(alt_cmd)
            if key in seen_variants:
                continue
            seen_variants.add(key)
            _purge_ref_dump()
            ckdmp_started = time.time()
            alt_res, alt_err = _run_once(alt_cmd, run_env=env, run_cwd=work_dir, timeout_sec=timeout_sec)
            if alt_err is not None or alt_res is None:
                attempt_diag.append(
                    {
                        "phase": "ckdmp-retry",
                        "cwd": str(work_dir),
                        "cmd": " ".join(shlex.quote(x) for x in alt_cmd),
                        "rc": None,
                        "stderr_tail": f"runner error: {alt_err}",
                    }
                )
                continue
            result = alt_res
            attempt_diag.append(
                {
                    "phase": "ckdmp-retry",
                    "cwd": str(work_dir),
                    "cmd": " ".join(shlex.quote(x) for x in alt_cmd),
                    "rc": int(result.returncode),
                    "stderr_tail": _short_tail(result.stderr, 280),
                }
            )
            has_dump_alt = _is_fresh_nonempty(ref_dump, ckdmp_started)
            if has_dump_alt:
                break
            if not _looks_like_flag_error(result.stderr):
                break

    has_dump = _is_fresh_nonempty(ref_dump, ckdmp_started)
    has_index = _is_fresh_nonempty(ref_index, ckdmp_started)

    if result.returncode == 0 and has_dump:
        ok_tokens, token_msg = _token_audit(ref_index)
        if require_token_aware_dumps and not ok_tokens:
            log_error(f"llama CKDMP token audit failed: {token_msg}")
            if not allow_raw_fallback:
                _purge_ref_dump()
                return False
        else:
            log(f"  Reference dump: {ref_dump}", C_GREEN)
            if ok_tokens:
                log(f"  CKDMP token audit: {token_msg}", C_DIM)
            return True

    # CKDMP_STOP_AFTER intentionally exits non-zero after enough dumps.
    if result.returncode != 0 and has_dump and has_index:
        ok_tokens, token_msg = _token_audit(ref_index)
        if require_token_aware_dumps and not ok_tokens:
            log_error(f"llama CKDMP token audit failed: {token_msg}")
            if not allow_raw_fallback:
                _purge_ref_dump()
                return False
        else:
            log(f"  llama.cpp exited with code {result.returncode} after writing dumps", C_ORANGE)
            log(f"  Reference dump: {ref_dump}", C_GREEN)
            if ok_tokens:
                log(f"  CKDMP token audit: {token_msg}", C_DIM)
            return True

    # Fallback for local llama.cpp builds that expose LLAMA_DUMP_LAYER0 raw dumps
    # but are not instrumented with CKDMP output.
    raw_converter = SCRIPTS_DIR / "parity" / "llama_to_ckdmp_converter.py"
    if allow_raw_fallback and raw_converter.exists():
        env_raw = os.environ.copy()
        env_raw["LLAMA_DUMP_LAYER0"] = "1"
        for key in (
            "CKDMP_DIR",
            "CKDMP_ALL_LAYERS",
            "CKDMP_FILTER",
            "CKDMP_LAYER",
            "CKDMP_STOP_AFTER",
            "CKDMP_INCLUDE_GLOBAL",
        ):
            env_raw.pop(key, None)
        raw_parent_candidates: list[Path] = []
        for parent in (
            work_dir,
            work_dir / ".ck_build",
            gguf_path.parent,
            gguf_path.parent / ".ck_build",
            work_dir.parent,
            work_dir.parent / ".ck_build",
        ):
            if parent not in raw_parent_candidates:
                raw_parent_candidates.append(parent)

        for raw_parent in raw_parent_candidates:
            raw_dump_dir = raw_parent / "llama_dump"
            try:
                raw_dump_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                attempt_diag.append(
                    {
                        "phase": "raw",
                        "cwd": str(raw_parent),
                        "cmd": "<mkdir llama_dump>",
                        "rc": None,
                        "stderr_tail": f"skip: cannot create raw dump dir ({e})",
                    }
                )
                continue
            if not os.access(raw_parent, os.W_OK | os.X_OK) or not os.access(raw_dump_dir, os.W_OK | os.X_OK):
                attempt_diag.append(
                    {
                        "phase": "raw",
                        "cwd": str(raw_parent),
                        "cmd": "<raw fallback>",
                        "rc": None,
                        "stderr_tail": "skip: raw dump dir not writable",
                    }
                )
                continue

            for stale in raw_dump_dir.glob("*.bin"):
                try:
                    stale.unlink()
                except OSError:
                    pass
            raw_index = raw_dump_dir / "index.json"
            try:
                if raw_index.exists():
                    raw_index.unlink()
            except OSError:
                pass

            raw_started = time.time()
            raw_res, raw_err = _run_once(cmd, run_env=env_raw, run_cwd=raw_parent, timeout_sec=timeout_sec)
            if raw_err is not None:
                attempt_diag.append(
                    {
                        "phase": "raw",
                        "cwd": str(raw_parent),
                        "cmd": " ".join(shlex.quote(x) for x in cmd),
                        "rc": None,
                        "stderr_tail": f"runner error: {raw_err}",
                    }
                )
                continue
            if raw_res is None:
                continue
            raw_bins = sorted(
                p for p in raw_dump_dir.glob("*.bin")
                if _is_fresh_nonempty(p, raw_started)
            )
            attempt_diag.append(
                {
                    "phase": "raw",
                    "cwd": str(raw_parent),
                    "cmd": " ".join(shlex.quote(x) for x in cmd),
                    "rc": int(raw_res.returncode),
                    "stderr_tail": _short_tail(raw_res.stderr, 260),
                    "raw_bins": int(len(raw_bins)),
                }
            )
            if not raw_bins:
                continue

            conv_cmd = [
                sys.executable,
                str(raw_converter),
                "-i",
                str(raw_dump_dir),
                "-o",
                str(ref_dump),
                "--model",
                str(parity_family or "llama"),
                "--index",
            ]
            conv_res = subprocess.run(
                conv_cmd,
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
            )
            if conv_res.returncode == 0 and ref_dump.exists() and ref_dump.stat().st_size > 0:
                if raw_res.returncode != 0:
                    log(
                        f"  llama.cpp raw dump run exited with {raw_res.returncode}; converted raw dumps anyway",
                        C_ORANGE,
                    )
                log(f"  Converted raw llama_dump ({raw_dump_dir}) -> {ref_dump}", C_GREEN)
                return True

    if require_token_aware_dumps and not allow_raw_fallback:
        log_error("token-aware llama CKDMP dump required; raw LLAMA_DUMP_LAYER0 fallback disabled")
        _purge_ref_dump()
        return False

    log_error("llama.cpp parity dump failed")
    log(f"  binary: {llm_path}", C_DIM)
    log(f"  gguf:   {gguf_path}", C_DIM)
    log(f"  run:    {work_dir}", C_DIM)
    if result is not None:
        log(
            f"  ckdmp rc={result.returncode} dump={has_dump} index={has_index}",
            C_DIM,
        )
        if result.returncode == -9:
            log("  hint: process killed (rc=-9). Try --context-len 1024 or lower.", C_ORANGE)
        if result.stderr:
            log(f"  ckdmp stderr tail:\n{_short_tail(result.stderr, 420)}", C_DIM)
    for d in attempt_diag[-6:]:
        rc_txt = "n/a" if d.get("rc") is None else str(d.get("rc"))
        log(
            f"  attempt[{d.get('phase')}] cwd={d.get('cwd')} rc={rc_txt} "
            f"cmd={d.get('cmd')}",
            C_DIM,
        )
        if d.get("stderr_tail"):
            log(f"    {d.get('stderr_tail')}", C_DIM)
    return False


def _infer_parity_family(model_input: Optional[str], work_dir: Path) -> str:
    """Best-effort model family inference for parity tooling."""
    text = " ".join(
        [
            str(model_input or ""),
            str(work_dir.name or ""),
            str(work_dir.parent.name or ""),
        ]
    ).lower()
    config_paths = [
        work_dir / "config.json",
        work_dir / ".ck_build" / "config.json",
    ]
    for cfg_path in config_paths:
        try:
            if not cfg_path.exists():
                continue
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            model_type = str(cfg.get("model_type") or "").strip().lower()
            if model_type in ("llama", "gemma", "qwen2", "qwen3", "qwen"):
                return model_type
        except Exception:
            pass
    if "nanbeige" in text:
        return "llama"
    if "qwen3" in text:
        return "qwen3"
    if "qwen2" in text:
        return "qwen2"
    if "qwen" in text:
        return "qwen"
    if "gemma" in text:
        return "gemma"
    if "mistral" in text:
        return "mistral"
    if "llama" in text:
        return "llama"
    return "qwen3"


def _run_detailed_inference_parity_reports(
    work_dir: Path,
    family: str,
    *,
    model_uri: Optional[str],
    context_len: Optional[int],
    max_tokens: int,
    prompt: str,
) -> None:
    """
    Generate JSON artifacts consumed by IR visualizer inference parity cockpit.
    This runs after parity dumps are present in the same run directory.
    """
    parity_model_uri = str(model_uri or work_dir)
    detail_script = SCRIPTS_DIR / "detailed_parity_analysis.py"
    if detail_script.exists():
        cmd = [
            sys.executable,
            str(detail_script),
            "--model-uri",
            parity_model_uri,
            "--output-dir",
            str(work_dir),
            "--family",
            family,
            "--prompt",
            prompt,
            "--max-tokens",
            str(max_tokens),
            "--skip-ck-run",
            "--skip-exhaustive-llama",
            "--report-prefix",
            "detailed_parity_analysis_latest",
        ]
        if context_len and int(context_len) > 0:
            cmd.extend(["--context-len", str(int(context_len))])
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            log(f"  Detailed parity report: {work_dir / 'detailed_parity_analysis_latest.json'}", C_GREEN)
        else:
            log("  Detailed parity analysis failed; continuing", C_ORANGE)
            if proc.stderr:
                log(f"    {proc.stderr[-300:]}", C_DIM)
    else:
        log("  detailed_parity_analysis.py not found; skipping inference parity summary", C_DIM)

    autopsy_script = SCRIPTS_DIR / "parity" / "parity_autopsy.py"
    if autopsy_script.exists():
        cmd = [
            sys.executable,
            str(autopsy_script),
            "--model-uri",
            parity_model_uri,
            "--output-dir",
            str(work_dir),
            "--family",
            family,
            "--pass",
            "decode",
            "--skip-run",
            "--report-prefix",
            "parity_autopsy_latest",
        ]
        if context_len and int(context_len) > 0:
            cmd.extend(["--context-len", str(int(context_len))])
        if max_tokens and int(max_tokens) > 0:
            cmd.extend(["--max-tokens", str(int(max_tokens))])
        if prompt:
            cmd.extend(["--prompt", prompt])
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            log(f"  Parity autopsy report: {work_dir / 'parity_autopsy_latest.json'}", C_GREEN)
        else:
            log("  Parity autopsy failed; continuing", C_ORANGE)
            if proc.stderr:
                log(f"    {proc.stderr[-300:]}", C_DIM)
    else:
        log("  parity/parity_autopsy.py not found; skipping autopsy summary", C_DIM)


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


def _generate_visualizer_html(work_dir: Path) -> Path:
    """Generate ir_report.html for a run directory without running probes/profile."""
    vis_script = V7_ROOT / "tools" / "open_ir_visualizer.py"
    if not vis_script.exists():
        raise RuntimeError(f"Visualizer script not found: {vis_script}")
    cmd = [
        sys.executable,
        str(vis_script),
        "--generate",
        "--run",
        str(work_dir),
        "--html-only",
        "--strict-run-artifacts",
    ]
    run_cmd(cmd, cwd=PROJECT_ROOT)
    report_path = work_dir / "ir_report.html"
    if report_path.exists():
        log(f"  Visualizer: {report_path}", C_GREEN)
    else:
        log(f"  Warning: visualizer generation completed but report missing at {report_path}", C_ORANGE)
    return report_path


def _template_audit_write_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _template_audit_extract_weight_refs(op: dict[str, Any]) -> set[str]:
    refs: set[str] = set()

    def _push_ref(value: Any) -> None:
        if isinstance(value, str):
            name = value.strip()
            if name:
                refs.add(name)

    _push_ref(op.get("weight_ref"))
    weight_refs = op.get("weight_refs")
    if isinstance(weight_refs, list):
        for value in weight_refs:
            _push_ref(value)

    args = op.get("args")
    arg_items: list[Any]
    if isinstance(args, list):
        arg_items = args
    elif isinstance(args, dict):
        arg_items = [args]
    else:
        arg_items = []

    for item in arg_items:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        if isinstance(source, str) and ":" in source:
            prefix, value = source.split(":", 1)
            if prefix.strip().lower() in {"weight", "manifest"}:
                _push_ref(value)
        _push_ref(item.get("weight_ref"))
        nested = item.get("weight_refs")
        if isinstance(nested, list):
            for value in nested:
                _push_ref(value)

    return refs


TEMPLATE_CONTRACT_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "tokenizer_contract": ("tokenizer_type", "special_tokens"),
    "attention_contract": ("rope_type", "kv_layout"),
    "block_contract": ("norm_type", "mlp_formula", "activation"),
    "logits_contract": ("final_norm", "lm_head"),
    "quant_contract": ("kernel_select",),
    "runtime_invariants": ("required_call_args",),
}


def _template_audit_validate_contract(contract_doc: Any) -> tuple[bool, dict[str, Any]]:
    missing_sections: list[str] = []
    missing_fields: dict[str, list[str]] = {}
    wrong_types: dict[str, str] = {}

    if not isinstance(contract_doc, dict):
        return False, {
            "error": "contract is missing or not an object",
            "missing_sections": list(TEMPLATE_CONTRACT_REQUIRED_FIELDS.keys()),
        }

    for section, fields in TEMPLATE_CONTRACT_REQUIRED_FIELDS.items():
        section_doc = contract_doc.get(section)
        if not isinstance(section_doc, dict):
            missing_sections.append(section)
            continue
        for field in fields:
            value = section_doc.get(field)
            if value is None:
                missing_fields.setdefault(section, []).append(field)
            elif field == "required_call_args" and not isinstance(value, dict):
                wrong_types[f"{section}.{field}"] = type(value).__name__
            elif field == "special_tokens" and not isinstance(value, dict):
                wrong_types[f"{section}.{field}"] = type(value).__name__

    ok = not missing_sections and not missing_fields and not wrong_types
    return ok, {
        "missing_sections": missing_sections,
        "missing_fields": missing_fields,
        "wrong_types": wrong_types,
    }


def _template_audit_ir_stitching_report(work_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_entries = manifest_doc.get("entries") if isinstance(manifest_doc.get("entries"), list) else []
    manifest_names = {
        str(entry.get("name", "")).strip()
        for entry in manifest_entries
        if isinstance(entry, dict) and str(entry.get("name", "")).strip()
    }

    mode_stats: dict[str, dict[str, Any]] = {}
    aggregate_refs: set[str] = set()
    aggregate_top_errors = 0
    aggregate_ops_with_errors = 0
    aggregate_ops = 0

    for mode in ("decode", "prefill"):
        path = work_dir / f"lowered_{mode}_call.json"
        if not path.exists():
            mode_stats[mode] = {"present": False}
            continue

        doc = json.loads(path.read_text(encoding="utf-8"))
        operations = doc.get("operations") if isinstance(doc.get("operations"), list) else []
        top_errors = doc.get("errors") if isinstance(doc.get("errors"), list) else []
        ops_with_errors = 0
        mode_refs: set[str] = set()

        for op in operations:
            if not isinstance(op, dict):
                continue
            errs = op.get("errors") if isinstance(op.get("errors"), list) else []
            if errs:
                ops_with_errors += 1
            mode_refs.update(_template_audit_extract_weight_refs(op))

        mode_stats[mode] = {
            "present": True,
            "path": str(path),
            "ops": int(len(operations)),
            "top_level_errors": int(len(top_errors)),
            "ops_with_errors": int(ops_with_errors),
            "unique_weight_refs": int(len(mode_refs)),
        }
        aggregate_refs.update(mode_refs)
        aggregate_top_errors += int(len(top_errors))
        aggregate_ops_with_errors += int(ops_with_errors)
        aggregate_ops += int(len(operations))

    unknown_refs = sorted(
        ref
        for ref in aggregate_refs
        if ref
        and not ref.startswith("_")
        and ref not in manifest_names
        and ref not in TEMPLATE_AUDIT_TOKENIZER_REF_ALLOWLIST
    )
    tokenizer_refs = sorted(
        ref for ref in aggregate_refs if ref in TEMPLATE_AUDIT_TOKENIZER_REF_ALLOWLIST
    )

    passed = bool(
        aggregate_ops > 0
        and len(unknown_refs) == 0
        and aggregate_top_errors == 0
        and aggregate_ops_with_errors == 0
    )

    return {
        "pass": passed,
        "ops_total": int(aggregate_ops),
        "top_level_errors_total": int(aggregate_top_errors),
        "ops_with_errors_total": int(aggregate_ops_with_errors),
        "unique_weight_refs_total": int(len(aggregate_refs)),
        "unknown_weight_refs": unknown_refs,
        "tokenizer_weight_refs": tokenizer_refs,
        "manifest_entries": int(len(manifest_names)),
        "mode_stats": mode_stats,
    }


def _template_audit_prepare_artifacts(
    args: argparse.Namespace,
    input_type: str,
    info: dict[str, Any],
    work_dir: Path,
) -> dict[str, Any]:
    gguf_path: Optional[Path] = None
    model_dir: Optional[Path] = None

    if input_type == "hf_id":
        model_id = info["model_id"]
        model_dir = step_download(model_id, CACHE_DIR, force=getattr(args, "force_download", False))
        has_safetensors = list(model_dir.glob("*.safetensors")) or list(model_dir.glob("model*.safetensors"))
        gguf_files = list(model_dir.glob("*.gguf"))

        if gguf_files and not has_safetensors:
            gguf_path = next(iter(sorted(gguf_files)), gguf_files[0])
            ensure_tokenizer_files(model_id, work_dir)
            weights_path, config_path = step_convert_gguf(
                gguf_path,
                work_dir,
                force=getattr(args, "force_convert", False),
                validate=True,
            )
            manifest_path = work_dir / "weights_manifest.json"
        else:
            tokenizer_json = model_dir / "tokenizer.json"
            weights_path = step_convert_hf(
                model_dir,
                work_dir,
                weight_dtype=(getattr(args, "weight_dtype", None) or "float32"),
                force=getattr(args, "force_convert", False),
                tokenizer_json=tokenizer_json if tokenizer_json.exists() else None,
            )
            config_path = model_dir / "config.json"
            manifest_path = work_dir / "weights_manifest.json"

    elif input_type == "hf_gguf":
        repo_id = info["repo_id"]
        gguf_path = step_download_gguf(repo_id, info["filename"], CACHE_DIR, force=getattr(args, "force_download", False))
        ensure_tokenizer_files(repo_id, work_dir)
        weights_path, config_path = step_convert_gguf(
            gguf_path,
            work_dir,
            force=getattr(args, "force_convert", False),
            validate=True,
        )
        manifest_path = work_dir / "weights_manifest.json"

    elif input_type == "gguf":
        gguf_path = info["path"]
        weights_path, config_path = step_convert_gguf(
            gguf_path,
            work_dir,
            force=getattr(args, "force_convert", False),
            validate=True,
        )
        manifest_path = work_dir / "weights_manifest.json"

    elif input_type == "local_dir":
        model_dir = info["path"]
        local_gguf = _find_local_gguf(model_dir)
        if _is_ck_runtime_dir(model_dir):
            if getattr(args, "force_convert", False) and local_gguf is not None:
                gguf_path = local_gguf
                weights_path, config_path = step_convert_gguf(
                    local_gguf,
                    work_dir,
                    force=True,
                    validate=True,
                )
                manifest_path = work_dir / "weights_manifest.json"
            else:
                weights_path, config_path, manifest_path = _prepare_runtime_dir_from_local_ck_artifacts(model_dir, work_dir)
        else:
            tokenizer_json = model_dir / "tokenizer.json"
            weights_path = step_convert_hf(
                model_dir,
                work_dir,
                weight_dtype=(getattr(args, "weight_dtype", None) or "float32"),
                force=getattr(args, "force_convert", False),
                tokenizer_json=tokenizer_json if tokenizer_json.exists() else None,
            )
            config_path = model_dir / "config.json"
            manifest_path = work_dir / "weights_manifest.json"

    elif input_type == "local_config":
        config_path = info["path"]
        model_dir = config_path.parent
        local_gguf = _find_local_gguf(model_dir)
        if _is_ck_runtime_dir(model_dir):
            if getattr(args, "force_convert", False) and local_gguf is not None:
                gguf_path = local_gguf
                weights_path, config_path = step_convert_gguf(
                    local_gguf,
                    work_dir,
                    force=True,
                    validate=True,
                )
                manifest_path = work_dir / "weights_manifest.json"
            else:
                weights_path, config_path, manifest_path = _prepare_runtime_dir_from_local_ck_artifacts(model_dir, work_dir)
        else:
            raise RuntimeError(
                f"config-only input requires colocated weights.bump + weights_manifest.json: {config_path.parent}"
            )

    else:
        raise RuntimeError(f"Unsupported input type for template audit: {input_type}")

    if not Path(manifest_path).exists():
        raise RuntimeError(f"Missing manifest after preparation: {manifest_path}")
    if not Path(config_path).exists():
        raise RuntimeError(f"Missing config after preparation: {config_path}")
    if not Path(weights_path).exists():
        raise RuntimeError(f"Missing weights bump after preparation: {weights_path}")

    return {
        "work_dir": work_dir,
        "manifest_path": Path(manifest_path),
        "config_path": Path(config_path),
        "weights_path": Path(weights_path),
        "gguf_path": gguf_path,
        "model_dir": model_dir,
    }


def _template_audit_run_reverse_validation_on_path(
    lowered_path: Path,
    manifest_path: Path,
    *,
    verbose: bool = False,
) -> tuple[bool, str]:
    try:
        from ir_reverse_validator import run_validation
    except ImportError:
        validator_path = SCRIPTS_DIR / "ir_reverse_validator.py"
        if not validator_path.exists():
            raise RuntimeError("ir_reverse_validator.py not found")
        import importlib.util
        spec = importlib.util.spec_from_file_location("ir_reverse_validator", validator_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to load ir_reverse_validator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run_validation = module.run_validation

    return run_validation(
        lowered_path=lowered_path,
        manifest_path=manifest_path if manifest_path.exists() else None,
        kernel_maps_dir=KERNEL_MAPS_DIR,
        verbose=verbose,
    )


def step_run_template_audit(args: argparse.Namespace) -> None:
    """Run fail-fast pre-compile template audit gates for v7 onboarding."""
    # TODO(contract): evolve this into full semantic bring-up gate runner:
    # 1) template/manifest semantic contract resolved,
    # 2) lowered-call invariants pass,
    # 3) first-token logits parity pass,
    # 4) post-attn chain check pass,
    # 5) MLP contract check pass.
    # Chat run should be blocked on any failed strict gate.
    model_input = str(getattr(args, "model", "") or "").strip()
    if not model_input:
        log_error("template-audit requires a model input")
        sys.exit(1)

    requested_run_dir_raw = getattr(args, "run_dir", None)
    requested_run_dir = Path(requested_run_dir_raw).expanduser().resolve() if requested_run_dir_raw else None

    input_type, info = detect_input_type(model_input)
    if input_type == "hf_id":
        default_work_dir = CACHE_DIR / info["model_id"].replace("/", "--")
    elif input_type == "hf_gguf":
        default_work_dir = CACHE_DIR / info["repo_id"].replace("/", "--")
    elif input_type == "gguf":
        default_work_dir = CACHE_DIR / info["path"].stem
    elif input_type == "local_dir":
        default_work_dir = info["path"] / ".ck_build"
    elif input_type == "local_config":
        default_work_dir = info["path"].parent / ".ck_build"
    else:
        default_work_dir = CACHE_DIR / "template_audit"

    work_dir = requested_run_dir if requested_run_dir is not None else default_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    report_path = (
        Path(getattr(args, "audit_json_out", "")).expanduser().resolve()
        if getattr(args, "audit_json_out", None)
        else (work_dir / "template_audit_latest.json")
    )

    report: dict[str, Any] = {
        "command": "v7-template-audit",
        "model_input": model_input,
        "input_type": input_type,
        "run_dir": str(work_dir),
        "started_at": _utc_now_iso(),
        "status": "running",
        "gates": [],
    }

    def _record_gate(name: str, passed: bool, details: dict[str, Any], message: str) -> None:
        report["gates"].append(
            {
                "name": name,
                "pass": bool(passed),
                "timestamp": _utc_now_iso(),
                "message": message,
                "details": details,
            }
        )

    def _fail_gate(name: str, message: str, details: dict[str, Any]) -> None:
        _record_gate(name, False, details, message)
        report["status"] = "fail"
        report["failed_gate"] = name
        report["completed_at"] = _utc_now_iso()
        _template_audit_write_report(report, report_path)
        log_error(f"[template-audit] {name} failed: {message}")
        log(f"  Report: {report_path}", C_DIM)
        raise SystemExit(1)

    log(f"{C_ORANGE}[template-audit]{C_RESET} v7 pre-compile audit", C_BOLD)
    log(f"  Input: {model_input} ({input_type})", C_DIM)
    log(f"  Run dir: {work_dir}", C_DIM)

    step_regenerate_kernel_registry(force=bool(getattr(args, "force_compile", False)))

    # Gate 1: source+weights contract (download/convert + manifest/config existence).
    try:
        prepared = _template_audit_prepare_artifacts(args, input_type, info, work_dir)
    except SystemExit as exc:
        _fail_gate(
            "gate1_weights_contract",
            "failed to resolve/convert model artifacts",
            {"error": f"conversion exited with code {exc.code}"},
        )
        return
    except Exception as exc:
        _fail_gate(
            "gate1_weights_contract",
            "failed to resolve/convert model artifacts",
            {"error": str(exc)},
        )
        return
    _record_gate(
        "gate1_weights_contract",
        True,
        {
            "manifest_path": str(prepared["manifest_path"]),
            "weights_path": str(prepared["weights_path"]),
            "config_path": str(prepared["config_path"]),
            "gguf_path": str(prepared["gguf_path"]) if prepared.get("gguf_path") else None,
        },
        "weights/config/manifest ready",
    )

    # Gate 2: template contract (embedded template or resolvable builtin template).
    manifest_path = prepared["manifest_path"]
    try:
        manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail_gate(
            "gate2_template_contract",
            "manifest unreadable",
            {"manifest_path": str(manifest_path), "error": str(exc)},
        )
        return

    cfg = manifest_doc.get("config") if isinstance(manifest_doc.get("config"), dict) else {}
    template_doc = manifest_doc.get("template") if isinstance(manifest_doc.get("template"), dict) else None
    embedded_ok = bool(template_doc and isinstance(template_doc.get("sequence"), list) and template_doc.get("sequence"))
    arch_candidates = []
    for key in ("model", "arch"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            arch_candidates.append(value.strip())
    built_in_path = None
    for cand in arch_candidates:
        path = V7_ROOT / "templates" / f"{cand.lower()}.json"
        if path.exists():
            built_in_path = path
            break
    builtin_doc = None
    if built_in_path:
        try:
            builtin_doc = json.loads(built_in_path.read_text(encoding="utf-8"))
        except Exception:
            builtin_doc = None

    if not embedded_ok and built_in_path is None:
        _fail_gate(
            "gate2_template_contract",
            "missing template contract (no embedded template and no builtin match)",
            {
                "manifest_path": str(manifest_path),
                "arch_candidates": arch_candidates,
                "expected_templates": [str(V7_ROOT / "templates" / f"{cand.lower()}.json") for cand in arch_candidates],
            },
        )
        return

    strict_contracts = not bool(getattr(args, "no_strict_contracts", False))
    contract_source = None
    contract_doc = None
    if template_doc and isinstance(template_doc.get("contract"), dict):
        contract_source = "embedded_template"
        contract_doc = template_doc.get("contract")
    elif builtin_doc and isinstance(builtin_doc.get("contract"), dict):
        contract_source = "builtin_template"
        contract_doc = builtin_doc.get("contract")

    contract_pass = True
    contract_details: dict[str, Any] = {}
    if strict_contracts:
        contract_pass, contract_details = _template_audit_validate_contract(contract_doc)
        if not contract_pass:
            _fail_gate(
                "gate2_template_contract",
                "strict contract missing/invalid (required semantic sections not present)",
                {
                    "manifest_path": str(manifest_path),
                    "embedded_template": bool(embedded_ok),
                    "builtin_template_path": str(built_in_path) if built_in_path else None,
                    "contract_source": contract_source,
                    "contract_validation": contract_details,
                },
            )
            return

    semantic_pass, semantic_details = _template_manifest_semantic_check(manifest_path)
    if strict_contracts and not semantic_pass:
        _fail_gate(
            "gate2_template_contract",
            "template/manifest semantic mismatch",
            {
                "manifest_path": str(manifest_path),
                "contract_source": contract_source,
                "semantic_checks": semantic_details,
            },
        )
        return

    _record_gate(
        "gate2_template_contract",
        True,
        {
            "embedded_template": bool(embedded_ok),
            "builtin_template_path": str(built_in_path) if built_in_path else None,
            "arch_candidates": arch_candidates,
            "manifest_path": str(manifest_path),
            "strict_contracts": bool(strict_contracts),
            "contract_source": contract_source,
            "contract_validation": contract_details,
            "semantic_checks": semantic_details,
        },
        "template contract resolved",
    )

    # Gate 3: IR stitching + reverse validation.
    # TODO(contract): split into:
    # - structural stitching gate
    # - semantic per-op contract gate (rope/norm/mlp/logits/kv invariants)
    try:
        weight_dtype = normalize_weight_dtype(getattr(args, "weight_dtype", None), manifest_path)
        ir1_path = step_build_ir(
            prepared["config_path"],
            work_dir,
            manifest_path=manifest_path,
            bump_path=prepared["weights_path"],
            weight_dtype=weight_dtype,
            force=bool(getattr(args, "force_compile", False)),
            codegen_version="v7",
            context_len=getattr(args, "context_len", None),
            logits_layout=getattr(args, "logits_layout", None),
            no_fusion=bool(getattr(args, "no_fusion", False)),
            layout_mode=str(getattr(args, "layout_mode", "region") or "region"),
            layer_limit=getattr(args, "layer_limit", None),
        )
    except SystemExit:
        _fail_gate(
            "gate3_ir_stitching",
            "build_ir failed",
            {"run_dir": str(work_dir)},
        )
        return

    stitching = _template_audit_ir_stitching_report(work_dir, manifest_path)
    if not bool(stitching.get("pass", False)):
        _fail_gate(
            "gate3_ir_stitching",
            "lowered IR has unresolved errors or unbound weights",
            stitching,
        )
        return

    reverse_modes = ["decode", "prefill"] if bool(getattr(args, "reverse_all_modes", False)) else ["decode"]
    reverse_results: dict[str, dict[str, Any]] = {}
    reverse_verbose = bool(getattr(args, "reverse_test_verbose", False))
    reverse_ok = True
    for mode in reverse_modes:
        lowered_path = work_dir / f"lowered_{mode}_call.json"
        if not lowered_path.exists():
            reverse_results[mode] = {"present": False, "pass": False}
            reverse_ok = False
            continue
        try:
            passed, report_text = _template_audit_run_reverse_validation_on_path(
                lowered_path,
                manifest_path,
                verbose=reverse_verbose,
            )
        except Exception as exc:
            passed = False
            report_text = str(exc)
        reverse_results[mode] = {"present": True, "pass": bool(passed)}
        if (not passed) and reverse_verbose and report_text:
            print(report_text)
        reverse_ok = reverse_ok and bool(passed)

    if not reverse_ok:
        _fail_gate(
            "gate3_ir_stitching",
            "reverse validation failed",
            {
                **stitching,
                "reverse_results": reverse_results,
                "reverse_modes": reverse_modes,
            },
        )
        return

    _record_gate(
        "gate3_ir_stitching",
        True,
        {
            **stitching,
            "ir1_path": str(ir1_path),
            "reverse_results": reverse_results,
            "reverse_modes": reverse_modes,
        },
        "IR lower/call stitching validated",
    )

    # Gate 4: codegen preflight (no compile/run).
    try:
        model_c_path = step_codegen(
            ir1_path,
            work_dir,
            force=bool(getattr(args, "force_compile", False)),
            profile=False,
            dump=False,
            strict_contracts=True,
        )
    except SystemExit:
        _fail_gate(
            "gate4_codegen_preflight",
            "codegen failed",
            {"run_dir": str(work_dir)},
        )
        return

    if not model_c_path.exists():
        _fail_gate(
            "gate4_codegen_preflight",
            "codegen did not produce model_v7.c",
            {"expected": str(model_c_path)},
        )
        return

    line_count = 0
    try:
        with model_c_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in handle:
                line_count += 1
    except Exception:
        line_count = 0

    _record_gate(
        "gate4_codegen_preflight",
        True,
        {
            "model_c_path": str(model_c_path),
            "line_count": int(line_count),
        },
        "codegen preflight passed",
    )

    report["status"] = "pass"
    report["completed_at"] = _utc_now_iso()
    _template_audit_write_report(report, report_path)
    log(f"  {C_GREEN}v7-template-audit PASS{C_RESET}")
    log(f"  Report: {report_path}", C_GREEN)


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

    requested_run_dir = None
    requested_run_dir_raw = getattr(args, "run_dir", None)
    if requested_run_dir_raw:
        requested_run_dir = Path(requested_run_dir_raw).expanduser().resolve()
        requested_run_dir.mkdir(parents=True, exist_ok=True)
        log(f"  Using explicit run directory: {requested_run_dir}", C_DIM)

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
        default_work_dir = CACHE_DIR / model_id.replace('/', '--')
        work_dir = requested_run_dir if requested_run_dir is not None else default_work_dir
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
        default_work_dir = CACHE_DIR / gguf_path.stem
        work_dir = requested_run_dir if requested_run_dir is not None else default_work_dir
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
        local_gguf = _find_local_gguf(model_dir)
        if local_gguf is not None:
            gguf_path_for_tokenizer = local_gguf
        default_work_dir = model_dir / ".ck_build"
        work_dir = requested_run_dir if requested_run_dir is not None else default_work_dir
        config_path = model_dir / "config.json"

        # Local CK runtime/train run dir: reuse existing bump+manifest directly.
        if _is_ck_runtime_dir(model_dir):
            if requested_run_dir is not None:
                try:
                    same_dir = work_dir.resolve() == model_dir.resolve()
                except Exception:
                    same_dir = str(work_dir) == str(model_dir)
                if same_dir:
                    log_error(
                        "For local CK runtime dirs, --run must not equal the model dir.\n"
                        "Omit --run to build into <model_dir>/.ck_build, or choose a different output dir."
                    )
                    sys.exit(2)
            should_reconvert = bool(args.force_convert)
            if not should_reconvert and local_gguf is not None:
                coverage_ok, _ = _gguf_manifest_weight_category_check(
                    local_gguf,
                    model_dir / "weights_manifest.json",
                    verbose=False,
                    report_out=model_dir / "gguf_weight_category_coverage.json",
                )
                should_reconvert = not coverage_ok
                if should_reconvert:
                    log("  Local runtime manifest category mismatch; reconverting from GGUF", C_ORANGE)

            try:
                if should_reconvert and local_gguf is not None:
                    weights_path, config_path = step_convert_gguf(
                        local_gguf,
                        work_dir,
                        force=True,
                        validate=True,
                    )
                    manifest_path = work_dir / "weights_manifest.json"
                else:
                    if should_reconvert and local_gguf is None:
                        log("  force-convert requested but no local GGUF found; using existing CK runtime artifacts", C_ORANGE)
                    weights_path, config_path, manifest_path = _prepare_runtime_dir_from_local_ck_artifacts(
                        model_dir, work_dir
                    )
            except Exception as e:
                log_error(f"failed to prepare local CK runtime dir: {e}")
                sys.exit(1)
            if args.inspect_only:
                log(f"  Local CK runtime manifest: {manifest_path}", C_GREEN)
                return
        else:
            # Convert local HF checkpoint weights
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
        default_work_dir = config_path.parent / ".ck_build"
        work_dir = requested_run_dir if requested_run_dir is not None else default_work_dir
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
        default_work_dir = CACHE_DIR / repo_id.replace('/', '--')
        work_dir = requested_run_dir if requested_run_dir is not None else default_work_dir

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

    # Optional: generate IR visualizer HTML in the same run directory.
    # Must run before chat because non-profile chat path exec()s and does not return.
    if getattr(args, "generate_visualizer", False):
        log(f"\n{C_ORANGE}[viz]{C_RESET} Generating IR visualizer HTML", C_DIM)
        _generate_visualizer_html(work_dir)

    # Determine effective context length for parity tools (prefer explicit CLI override).
    effective_ctx = getattr(args, "context_len", None)
    parity_artifacts_generated = False

    # Detailed llama.cpp parity can run independently of chat/generate-only mode.
    if detailed_llama_parity:
        weights_bump = Path(weights_path) if weights_path else (work_dir / "weights.bump")
        if not weights_bump.exists():
            log_error(f"weights.bump not found at {weights_bump}")
            sys.exit(1)
        parity_prompt = getattr(args, "prompt", None) or "Hello"
        parity_max_tokens = int(getattr(args, "max_tokens", 1) or 1)
        step_run_c_cli_parity_dump(lib_path, weights_bump, parity_prompt, parity_max_tokens, effective_ctx)
        parity_family = _infer_parity_family(getattr(args, "model", None), work_dir)
        llama_ok = _run_llamacpp_parity(
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
            parity_family=parity_family,
            require_token_aware_dumps=bool(getattr(args, "llama_require_token_aware_dumps", False)),
            allow_raw_fallback=not bool(getattr(args, "llama_no_raw_fallback", False)),
        )
        if not llama_ok:
            log_error("llama.cpp reference dump failed; cannot complete --detailed-llamacpp-parity")
            sys.exit(1)
        parity_model_uri = str(gguf_path_for_tokenizer) if gguf_path_for_tokenizer else str(getattr(args, "model", ""))
        _run_detailed_inference_parity_reports(
            work_dir,
            parity_family,
            model_uri=parity_model_uri,
            context_len=effective_ctx,
            max_tokens=parity_max_tokens,
            prompt=parity_prompt,
        )
        parity_artifacts_generated = True

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
        if not detailed_llama_parity:
            step_run_chat(work_dir, args, gguf_path=gguf_path_for_tokenizer)

    # Generate profile summary if profiling was enabled
    if getattr(args, 'profile', False):
        _generate_profile_summary(work_dir)

    # Keep report in sync when parity artifacts were generated after initial report emit.
    if parity_artifacts_generated and getattr(args, "generate_visualizer", False):
        log(f"\n{C_ORANGE}[viz]{C_RESET} Refreshing IR visualizer HTML (with parity artifacts)", C_DIM)
        _generate_visualizer_html(work_dir)

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
  ./ck-v7 init --run-name exp1 --init xavier_uniform
  ./ck-v7 train --run ~/.cache/ck-engine-v7/models/train/exp1 --data ./train.txt --train-epochs 3
  ./ck-v7 sanity --run ~/.cache/ck-engine-v7/models/train/exp1 --data ./train.txt --train-epochs 1
  ./ck-v7 parity --run ~/.cache/ck-engine-v7/models/train/exp1 --data ./train.txt --with-fd --with-replay
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    def _add_train_common_args(sp: argparse.ArgumentParser, *, include_profile: bool = True) -> None:
        sp.add_argument('--run', dest='run_dir', default=None,
                        help='Run directory for artifacts (single source of truth)')
        sp.add_argument('--data', dest='train_data', default=None,
                        help='Path to UTF-8 training text file (repeated to fill token budget)')
        sp.add_argument('--train-token-file', dest='train_token_file', default=None,
                        help='Path to pre-tokenized integer stream file (overrides --data/--prompt tokenization)')
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
        sp.add_argument('--train-disable-diag-snapshot', action='store_true',
                        help='Disable strict memory-diagnostic weight snapshot malloc path in generated runtime')
        sp.add_argument('--parity-on', action='store_true',
                        help='Enable scheduled oracle parity checks (metadata/config for training pipeline)')
        sp.add_argument('--kernel-strict-math', action='store_true',
                        help='Force strict kernel math in CK runtime (exact sigmoid/SwiGLU + strict parity math paths)')
        sp.add_argument('--bitwise-parity', action='store_true',
                        help='CK runtime only: force deterministic single-thread + strict FP compile flags for near-bitwise parity diagnostics')
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
        sp.add_argument('--bruteforce-debug', action='store_true',
                        help='Generated-runtime only: force parity_every=1 + replay checks + check-step dumps')
        sp.add_argument('--dump-on-drift', action='store_true',
                        help='On parity mismatch, dump drift artifacts for triage')
        sp.add_argument('--dump-on-check', action='store_true',
                        help='Dump runtime snapshots on every checked parity step (generated-runtime backend)')
        sp.add_argument('--dump-check-topk', type=int, default=200,
                        help='Cap check-dump entries retained in JSON metadata (default: 200)')
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
        sp.add_argument('--ablate-attention-backward', action='store_true',
                        help='Compile CK runtime with CK_ABLATE_ATTENTION_BACKWARD=1 (diagnostic only)')
        sp.add_argument('--ablate-rope-backward-qk', action='store_true',
                        help='Compile CK runtime with CK_ABLATE_ROPE_BACKWARD_QK=1 (diagnostic only)')
        sp.add_argument('--ablate-qk-norm-backward', action='store_true',
                        help='Compile CK runtime with CK_ABLATE_QK_NORM_BACKWARD=1 (diagnostic only)')
        sp.add_argument('--train-verify-memory', action='store_true',
                        help='Run PR3.7 memory verification suite (toggle diff, intentional +1, ASan agreement, bounds)')
        sp.add_argument('--train-verify-steps', type=int, default=4,
                        help='Number of deterministic steps used in toggle-diff verification')
        sp.add_argument('--train-verify-fault-op-id', type=int, default=-1,
                        help='Fault op_id for PR3.7 verification (default: max backward op_id)')
        sp.set_defaults(train_use_init_bump=True)
        sp.add_argument('--no-train-use-init-bump', dest='train_use_init_bump', action='store_false',
                        help='Do not load tiny parity init from run_dir/weights.bump')
        sp.add_argument('--parity-regimen', choices=['off', 'suggest', 'run', 'require'], default='suggest',
                        help='Post-train operator gate: suggest/run/require staged CK-vs-PyTorch parity regimen')

        sp.add_argument('--train-epochs', type=int, default=3)
        sp.add_argument('--train-seq-len', type=int, default=16)
        sp.add_argument('--train-total-tokens', type=int, default=1024)
        sp.add_argument('--train-grad-accum', type=int, default=8)
        sp.add_argument('--train-optimizer', choices=['adamw', 'sgd'], default='adamw')
        sp.add_argument('--ck-loss-backend', choices=['c', 'c_ptref', 'torch'], default='c',
                        help='CK CE backend for parity harness (c/c_ptref/torch)')
        sp.add_argument('--train-lr', type=float, default=1e-3)
        sp.add_argument('--train-adamw-beta1', type=float, default=None,
                        help='AdamW beta1 for CK runtime/parity harness (default: 0.9)')
        sp.add_argument('--train-adamw-beta2', type=float, default=None,
                        help='AdamW beta2 for CK runtime/parity harness (default: 0.999)')
        sp.add_argument('--train-adamw-eps', type=float, default=None,
                        help='AdamW epsilon for CK runtime/parity harness (default: 1e-8)')
        sp.add_argument('--train-adamw-weight-decay', type=float, default=None,
                        help='AdamW weight decay for CK runtime/parity harness (default: 0.01)')
        sp.add_argument('--train-max-grad-norm', type=float, default=0.0,
                        help='Global grad norm clip for CK runtime (0 disables clipping; default: 0.0)')
        sp.add_argument('--enforce-production-safety', action='store_true',
                        help='Fail fast on known-unsafe long-horizon AdamW settings')
        sp.add_argument('--allow-unsafe-adamw-lr', action='store_true',
                        help='Bypass AdamW LR safety guard (use only for diagnostics)')
        sp.add_argument('--train-unsafe-adamw-lr-threshold', type=float, default=1e-3,
                        help='LR threshold used by production safety guard (default: 1e-3)')
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
                        help='Optional JSON output path (default: run_dir/train_e2e_latest.json or <cache>/reports/train_e2e_latest.json)')

        if include_profile:
            sp.add_argument('--profile-train', choices=['none', 'perf', 'vtune', 'advisor'], default='none',
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
    run_parser.add_argument('--top-k', type=int, default=40,
                           help='Top-k sampling size passed to ck_chat.py (default: 40)')
    run_parser.add_argument('--top-p', type=float, default=1.0,
                           help='Top-p nucleus sampling passed to ck_chat.py (default: 1.0)')
    run_parser.add_argument('--min-p', type=float, default=0.0,
                           help='Min-p filter passed to ck_chat.py (default: 0.0)')
    run_parser.add_argument('--repeat-penalty', type=float, default=1.0,
                           help='Repeat penalty passed to ck_chat.py (default: 1.0)')
    run_parser.add_argument('--repeat-last-n', type=int, default=64,
                           help='Repeat penalty history window passed to ck_chat.py (default: 64)')
    run_parser.add_argument('--prompt', help='Single prompt (non-interactive)')
    run_parser.add_argument('--train-e2e', action='store_true',
                           help='Run tiny training parity E2E (CK vs PyTorch) and exit')
    run_parser.add_argument('--run', dest='run_dir', default=None,
                           help='Optional explicit run directory for inference artifacts (also used by --train-e2e)')
    run_parser.add_argument('--train-data', default=None,
                           help='Training text file for --train-e2e (UTF-8)')
    run_parser.add_argument('--train-token-file', default=None,
                           help='Pre-tokenized integer stream file for --train-e2e (overrides --train-data/--train-text)')
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
    run_parser.add_argument('--train-disable-diag-snapshot', action='store_true',
                           help='Disable strict memory-diagnostic weight snapshot malloc path in generated runtime')
    run_parser.add_argument('--parity-on', action='store_true',
                           help='Enable scheduled oracle parity checks metadata for --train-e2e')
    run_parser.add_argument('--kernel-strict-math', action='store_true',
                           help='Force strict kernel math in CK runtime (exact sigmoid/SwiGLU + strict parity math paths)')
    run_parser.add_argument('--bitwise-parity', action='store_true',
                           help='CK runtime only: force deterministic single-thread + strict FP compile flags for near-bitwise parity diagnostics')
    run_parser.add_argument('--oracle', choices=['pytorch'], default='pytorch')
    run_parser.add_argument('--parity-profile', choices=['debug', 'balanced', 'light'], default='balanced')
    run_parser.add_argument('--parity-every', type=int, default=50)
    run_parser.add_argument('--parity-replay-on-check', action='store_true')
    run_parser.add_argument('--parity-replay-tol', type=float, default=1e-7)
    run_parser.add_argument('--bruteforce-debug', action='store_true',
                           help='Generated-runtime only: force parity_every=1 + replay checks + check-step dumps')
    run_parser.add_argument('--dump-on-drift', action='store_true')
    run_parser.add_argument('--dump-on-check', action='store_true')
    run_parser.add_argument('--dump-check-topk', type=int, default=200)
    run_parser.add_argument('--drift-topk', type=int, default=20)
    run_parser.add_argument('--analysis-checkpoints', choices=['log', 'off'], default='log')
    run_parser.add_argument('--train-runtime-canary-checks', action='store_true')
    run_parser.add_argument('--train-runtime-bounds-assert', action='store_true')
    run_parser.add_argument('--train-runtime-fault-op-id', type=int, default=-1)
    run_parser.add_argument('--ablate-attention-backward', action='store_true')
    run_parser.add_argument('--ablate-rope-backward-qk', action='store_true')
    run_parser.add_argument('--ablate-qk-norm-backward', action='store_true')
    run_parser.add_argument('--train-verify-memory', action='store_true')
    run_parser.add_argument('--train-verify-steps', type=int, default=4)
    run_parser.add_argument('--train-verify-fault-op-id', type=int, default=-1)
    run_parser.set_defaults(train_use_init_bump=True)
    run_parser.add_argument('--no-train-use-init-bump', dest='train_use_init_bump', action='store_false',
                           help='Do not load tiny parity init from run_dir/weights.bump')
    run_parser.add_argument('--parity-regimen', choices=['off', 'suggest', 'run', 'require'], default='suggest',
                           help='With --train-e2e: suggest/run/require staged CK-vs-PyTorch parity regimen')
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
    run_parser.add_argument('--ck-loss-backend', choices=['c', 'c_ptref', 'torch'], default='c',
                           help='CK CE backend for parity harness (c/c_ptref/torch)')
    run_parser.add_argument('--train-lr', type=float, default=1e-3,
                           help='Learning rate for --train-e2e (default: 1e-3)')
    run_parser.add_argument('--train-adamw-beta1', type=float, default=None,
                           help='AdamW beta1 for CK runtime/parity harness (default: 0.9)')
    run_parser.add_argument('--train-adamw-beta2', type=float, default=None,
                           help='AdamW beta2 for CK runtime/parity harness (default: 0.999)')
    run_parser.add_argument('--train-adamw-eps', type=float, default=None,
                           help='AdamW epsilon for CK runtime/parity harness (default: 1e-8)')
    run_parser.add_argument('--train-adamw-weight-decay', type=float, default=None,
                           help='AdamW weight decay for CK runtime/parity harness (default: 0.01)')
    run_parser.add_argument('--train-max-grad-norm', type=float, default=0.0,
                           help='Global grad norm clip for CK runtime (0 disables clipping; default: 0.0)')
    run_parser.add_argument('--enforce-production-safety', action='store_true',
                           help='Fail fast on known-unsafe long-horizon AdamW settings')
    run_parser.add_argument('--allow-unsafe-adamw-lr', action='store_true',
                           help='Bypass AdamW LR safety guard (use only for diagnostics)')
    run_parser.add_argument('--train-unsafe-adamw-lr-threshold', type=float, default=1e-3,
                           help='LR threshold used by production safety guard (default: 1e-3)')
    run_parser.add_argument('--train-seed', type=int, default=42,
                           help='Random seed for --train-e2e (default: 42)')
    run_parser.add_argument('--train-vocab', type=int, default=256)
    run_parser.add_argument('--train-d-model', type=int, default=64)
    run_parser.add_argument('--train-hidden', type=int, default=128)
    run_parser.add_argument('--train-loss-tol', type=float, default=2e-5)
    run_parser.add_argument('--train-param-tol', type=float, default=3e-5)
    run_parser.add_argument('--train-json-out', default=None,
                           help='Optional JSON output path for --train-e2e (default: run_dir/train_e2e_latest.json or <cache>/reports/train_e2e_latest.json)')
    run_parser.add_argument('--profile-train', choices=['none', 'perf', 'vtune', 'advisor'], default='none',
                           help='Optional external profiler for --train-e2e (none, perf, vtune, advisor)')
    run_parser.add_argument('--train-profile-dir', default=None,
                           help='Output directory for train profiler artifacts (default: run_dir/profile_train_latest)')
    run_parser.add_argument('--chat-template', choices=['auto', 'none', 'qwen', 'gemma', 'llama'], default='auto',
                           help='Chat template mode passed to ck_chat.py (auto, none, qwen, gemma, llama)')
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
    run_parser.add_argument('--generate-visualizer', action='store_true',
                           help='Generate ir_report.html in the same run directory after pipeline completion')
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
    run_parser.add_argument('--llama-require-token-aware-dumps', action='store_true',
                           help='Require token-aware llama CKDMP index (reject collapsed token_id=0 reference dumps)')
    run_parser.add_argument('--llama-no-raw-fallback', action='store_true',
                           help='Disable LLAMA_DUMP_LAYER0 raw fallback conversion during llama parity dump generation')
    run_parser.add_argument('--parallel-decode', action='store_true',
                           help='[DEPRECATED] Flag accepted for compatibility only.')
    run_parser.add_argument('--reverse-test', action='store_true',
                           help='Run IR reverse validation after codegen (validates IR Lower 3 consistency)')
    run_parser.add_argument('--reverse-test-verbose', action='store_true',
                           help='Show detailed info from reverse validation')

    template_audit_parser = subparsers.add_parser(
        'template-audit',
        aliases=['v7-template-audit'],
        help='Run fail-fast pre-compile v7 onboarding audit (4 gates)',
    )
    template_audit_parser.add_argument('model', help='Model ID, URL, GGUF file, or local path')
    template_audit_parser.add_argument('--run', dest='run_dir', default=None,
                                       help='Explicit run directory for audit artifacts/report')
    template_audit_parser.add_argument('--weight-dtype',
                                       choices=['float32', 'bf16', 'q4_0', 'q4_1', 'q4_k', 'q4_k_m',
                                                'q5_0', 'q5_1', 'q6_k', 'q8_0'],
                                       help='Weight dtype override for HF checkpoint conversion path')
    template_audit_parser.add_argument('--context-len', type=int, default=None,
                                       help='Context length passed to IR build audit')
    template_audit_parser.add_argument('--logits-layout', choices=['auto', 'last', 'full'], default='auto',
                                       help='Logits buffer layout passed to IR build audit')
    template_audit_parser.add_argument('--no-fusion', action='store_true',
                                       help='Disable kernel fusion during IR audit build')
    template_audit_parser.add_argument('--layout-mode', choices=['region', 'packed'], default='region',
                                       help='Memory layout mode for IR audit build')
    template_audit_parser.add_argument('--layer-limit', type=int, default=None,
                                       help='Limit to first N layers during audit build')
    template_audit_parser.add_argument('--force-download', action='store_true',
                                       help='Re-download model files for audit')
    template_audit_parser.add_argument('--force-convert', action='store_true',
                                       help='Re-convert weights for audit')
    template_audit_parser.add_argument('--force-compile', action='store_true',
                                       help='Rebuild IR/codegen artifacts during audit')
    template_audit_parser.add_argument('--reverse-test-verbose', action='store_true',
                                       help='Print detailed reverse-validation output')
    template_audit_parser.add_argument('--reverse-all-modes', action='store_true',
                                       help='Require reverse-validation pass for both decode and prefill (default: decode only)')
    template_audit_parser.add_argument('--no-strict-contracts', action='store_true',
                                       help='Allow template-audit gate2 without required semantic contract sections')
    template_audit_parser.add_argument('--audit-json-out', default=None,
                                       help='Override JSON output path (default: <run_dir>/template_audit_latest.json)')

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
                             help=f'Optional explicit run directory. Prefer omitting this and using --run-name so the default {_cache_train_root_hint()}/<run-name> location is used.')
    init_parser.add_argument('--run-name', default='tiny_init',
                             help=f'Preferred run selector. When --run is omitted, the run is created under {_cache_train_root_hint()}/<run-name>.')
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
    init_parser.add_argument('--adamw-beta1', type=float, default=0.9,
                             help='AdamW beta1 embedded into run manifest training defaults')
    init_parser.add_argument('--adamw-beta2', type=float, default=0.999,
                             help='AdamW beta2 embedded into run manifest training defaults')
    init_parser.add_argument('--adamw-eps', type=float, default=1e-8,
                             help='AdamW epsilon embedded into run manifest training defaults')
    init_parser.add_argument('--adamw-weight-decay', type=float, default=0.01,
                             help='AdamW weight_decay embedded into run manifest training defaults')
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
    init_parser.add_argument('--dataset-workspace', default=None,
                             help='Optional dataset workspace to snapshot into the run dir (e.g. version/v7/data/spec03)')
    init_parser.add_argument('--dataset-stage-mode', choices=['copy', 'symlink'], default='copy',
                             help='How to stage the dataset workspace into the run dir when --dataset-workspace is set')
    init_parser.add_argument('--dataset-stage-force', action='store_true',
                             help='Replace existing run_dir/dataset snapshot and dataset_viewer.html during init')

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
    _ensure_v7_python_requirements(args.command)

    try:
        if args.command is None:
            args.weight_dtype = None
            args.temperature = 0.7
            args.max_tokens = 512
            run_interactive(args)
        elif args.command == 'run':
            run_pipeline(args)
        elif args.command in ('template-audit', 'v7-template-audit'):
            step_run_template_audit(args)
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
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        _print_v7_next_steps(args.command, exc)
        raise SystemExit(1)
    except Exception as exc:
        log_error(f"{type(exc).__name__}: {exc}")
        _print_v7_next_steps(args.command, exc)
        if os.environ.get("CK_V7_DEBUG"):
            raise
        raise SystemExit(1)


if __name__ == "__main__":
    main()
