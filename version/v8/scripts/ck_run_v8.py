#!/usr/bin/env python3
from __future__ import annotations

"""
Inference-only v8 pipeline runner.

This is the v8 copy of the high-level run surface. It intentionally covers the
inference lane only:

  download GGUF -> convert -> build IR -> codegen -> compile -> chat

and adds a v8-native multimodal route:

  decoder GGUF + mmproj GGUF + image -> run_multimodal_bridge_v8.py

and a generated audio-transformer route:

  generated encoder + generated decoder + WAV -> ck_model_run_audio_wav
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Optional


SCRIPTS_DIR = Path(__file__).resolve().parent
V8_ROOT = SCRIPTS_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parents[2]
ROOT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BUILD_DIR = PROJECT_ROOT / "build"
KERNEL_MAPS_DIR = V8_ROOT / "kernel_maps"
KERNEL_REGISTRY_PATH = KERNEL_MAPS_DIR / "KERNEL_REGISTRY.json"
V8_REQUIREMENTS_PATH = PROJECT_ROOT / "requirements-v8.txt"
V8_VISUALIZER_PATH = V8_ROOT / "tools" / "open_ir_visualizer_v8.py"
REPO_VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ck-engine-v8" / "models"
AUTO_MMPROJ_SPECS = (
    {
        "match_any": (
            "qwen/qwen3-vl-8b-instruct-gguf",
            "qwen3-vl-8b-instruct-gguf",
            "qwen3-vl-8b-instruct",
            "qwen3vl-8b-instruct",
        ),
        "mmproj": "hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
    },
)
CODEGEN_STAMP_NAME = ".ck_codegen_bundle.json"
IR_STAMP_NAME = ".ck_ir_bundle.json"
RUNTIME_STAMP_NAME = ".ck_runtime_bundle.json"
RUNTIME_BUNDLE_SCHEMA = "ck-v8-runtime-bundle-v2"


def _configure_scratch_environment() -> Path:
    """Keep compiler and helper temporaries in persistent, writable storage."""
    configured = os.environ.get("CK_V8_TMPDIR") or os.environ.get("TMPDIR")
    if configured:
        scratch = Path(configured).expanduser()
    else:
        cache_home = Path(
            os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        ).expanduser()
        scratch = cache_home / "ck-engine-v8" / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    if not _can_write_dir(scratch):
        raise RuntimeError(f"v8 scratch directory is not writable: {scratch}")
    scratch = scratch.resolve()
    os.environ["TMPDIR"] = str(scratch)
    tempfile.tempdir = None
    return scratch


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _optional_file_identity(path: Path) -> dict[str, Any]:
    if path.is_file():
        return _file_identity(path)
    return {"path": str(path), "missing": True}


def _tree_identity(paths: list[Path]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for root in paths:
        if root.is_file():
            rows[str(root.resolve())] = _sha256_file(root)
            continue
        if not root.is_dir():
            rows[str(root.resolve())] = "missing"
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rows[str(path.resolve())] = _sha256_file(path)
    return rows


def _read_bundle_stamp(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write_bundle_stamp(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _bundle_is_current(
    stamp_path: Path,
    *,
    inputs: dict[str, Any],
    outputs: dict[str, Path],
) -> bool:
    stamp = _read_bundle_stamp(stamp_path)
    if stamp is None or stamp.get("inputs") != inputs:
        return False
    recorded_outputs = stamp.get("outputs")
    if not isinstance(recorded_outputs, dict):
        return False
    for name, path in outputs.items():
        if not path.is_file():
            return False
        if recorded_outputs.get(name) != _file_identity(path):
            return False
    return True


def _can_write_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".ck_write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _fallback_cache_dir() -> Path:
    return Path(tempfile.gettempdir()) / "ck-engine-v8" / "models"


def _get_cache_dir() -> Path:
    env = os.environ.get("CK_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    if _can_write_dir(DEFAULT_CACHE_DIR):
        return DEFAULT_CACHE_DIR
    return _fallback_cache_dir()


def _cache_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for root in (CACHE_DIR, DEFAULT_CACHE_DIR, LEGACY_CACHE_DIR):
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def _hf_hub_cache_dir(cache_dir: Path) -> Path:
    return cache_dir / ".hf-hub"


# Marker shared with run_regression_v8.py: when a model/tokenizer download
# fails due to HuggingFace rate limiting (HTTP 429) or an HTML error page, the
# regression runner classifies the failure as environment_unavailable rather
# than build_failure. Keep this string in sync with ENVIRONMENT_FAILURE_MARKER
# in version/v8/scripts/run_regression_v8.py.
DOWNLOAD_ERROR_MARKER = "CK_V8_DOWNLOAD_FAILED"


class V8DownloadError(RuntimeError):
    """Download failed after retries or yielded an error page, not a code bug."""


def _download_failure(repo_id: str, filename: str, detail: str) -> V8DownloadError:
    return V8DownloadError(
        f"{DOWNLOAD_ERROR_MARKER}: download failed for {repo_id}/{filename}: {detail} "
        "(HuggingFace rate limit (HTTP 429) or error page received)"
    )


def _validate_gguf_magic(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def _validate_json_file(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except (OSError, UnicodeDecodeError, ValueError):
        return False


class _DownloadErrorPage(Exception):
    """Transient: the server returned an HTML/error page instead of the artifact."""


def _download_retry_config() -> tuple[int, float]:
    """(outer retries, backoff base seconds); overridable for CI tuning."""
    try:
        retries = int(os.environ.get("CK_V8_DOWNLOAD_RETRIES", "") or 3)
    except ValueError:
        retries = 3
    try:
        base = float(os.environ.get("CK_V8_DOWNLOAD_RETRY_BASE_SEC", "") or 15.0)
    except ValueError:
        base = 15.0
    return max(0, retries), max(0.0, base)


def _walk_error_chain(exc: BaseException):
    seen: set[int] = set()
    stack = [exc]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        yield cur
        for attr in ("__cause__", "__context__"):
            chained = getattr(cur, attr, None)
            if isinstance(chained, BaseException):
                stack.append(chained)


def _http_status_from_error(exc: BaseException) -> int | None:
    """HTTP status from requests/httpx/urllib-style exceptions, if any."""
    for cur in _walk_error_chain(exc):
        response = getattr(cur, "response", None)
        for source in (response, cur):
            if source is None:
                continue
            for attr in ("status_code", "status", "code"):
                value = getattr(source, attr, None)
                if isinstance(value, int):
                    return value
    return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Server Retry-After (integer seconds) from the exception, if present."""
    for cur in _walk_error_chain(exc):
        response = getattr(cur, "response", None)
        headers = None
        for source in (response, cur):
            if source is not None:
                headers = getattr(source, "headers", None)
                if headers is not None:
                    break
        if headers is None:
            continue
        try:
            value = headers.get("Retry-After") or headers.get("retry-after")
        except AttributeError:
            value = None
        if value is None:
            continue
        try:
            return float(int(str(value).strip()))
        except (TypeError, ValueError):
            continue
    return None


# Transient network exception names from requests/urllib3/httpx/httpcore and
# huggingface_hub. LocalEntryNotFoundError is how huggingface_hub reports
# "hub unreachable and nothing cached" — a connection failure, not a 404.
_TRANSIENT_NETWORK_ERROR_NAMES = {
    "ConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
    "Timeout",
    "ChunkedEncodingError",
    "ProtocolError",
    "ConnectError",
    "ReadError",
    "WriteError",
    "TimeoutException",
    "NetworkError",
    "RemoteProtocolError",
    "LocalEntryNotFoundError",
}
_TRANSIENT_NETWORK_MODULES = ("requests", "urllib3", "httpx", "httpcore", "huggingface_hub")


def _is_transient_network_error(exc: BaseException) -> bool:
    for cur in _walk_error_chain(exc):
        if isinstance(cur, (TimeoutError, ConnectionError)):
            return True
        if isinstance(cur, urllib.error.URLError) and not isinstance(cur, urllib.error.HTTPError):
            return True
        name = type(cur).__name__
        module = type(cur).__module__ or ""
        if name in _TRANSIENT_NETWORK_ERROR_NAMES and module.startswith(_TRANSIENT_NETWORK_MODULES):
            return True
    return False


def _classify_download_error(exc: BaseException) -> tuple[bool, str]:
    """(retryable, short reason) for a failed download attempt.

    Only transient server-side causes retry: HTTP 429/5xx, connection
    errors/timeouts, and error-page payloads. HTTP 404 (file genuinely
    absent) and other permanent errors fail immediately.
    """
    for cur in _walk_error_chain(exc):
        if isinstance(cur, _DownloadErrorPage):
            return True, "error page received"
    status = _http_status_from_error(exc)
    if status == 404:
        return False, "HTTP 404 (not found)"
    if status is not None:
        if status == 429 or 500 <= status < 600:
            return True, f"HTTP {status}"
        return False, f"HTTP {status}"
    if _is_transient_network_error(exc):
        return True, f"connection error ({type(exc).__name__})"
    return False, type(exc).__name__


def _run_with_download_retries(repo_id: str, filename: str, fetch: Callable[[], Any]) -> Any:
    """Run fetch() with rate-limit-aware outer retries around network work.

    Retries transient failures with exponential backoff (base doubling, plus
    0-5s jitter), honoring the server Retry-After header when larger. Any
    terminal failure raises the CK_V8_DOWNLOAD_FAILED-marked V8DownloadError
    so the regression runner classifies it as environment_unavailable.
    """
    retries, base = _download_retry_config()
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return fetch()
        except Exception as exc:
            retryable, reason = _classify_download_error(exc)
            if not retryable or attempt >= attempts:
                raise _download_failure(
                    repo_id, filename, f"{reason} after attempt {attempt}/{attempts}: {exc}"
                ) from exc
            backoff = base * (2 ** (attempt - 1)) + random.uniform(0.0, 5.0)
            retry_after = _retry_after_seconds(exc)
            sleep_s = max(backoff, retry_after) if retry_after is not None else backoff
            log(
                f"  {repo_id}/{filename}: download attempt {attempt}/{attempts} "
                f"failed ({reason}); sleeping {sleep_s:.1f}s before retry",
                C_DIM,
            )
            time.sleep(sleep_s)
    raise AssertionError("unreachable: download retry loop exited without result")


def _is_arm_machine(machine: str) -> bool:
    return machine in {"aarch64", "arm64", "armv7l", "armv8l"}


def _infer_auto_mmproj_spec(model_input: str) -> str | None:
    normalized = str(model_input or "").strip().lower().replace("\\", "/")
    if not normalized:
        return None
    for row in AUTO_MMPROJ_SPECS:
        if any(token in normalized for token in row["match_any"]):
            return str(row["mmproj"])
    return None


CACHE_DIR = _get_cache_dir()
LEGACY_CACHE_DIR = Path.home() / ".cache" / "ck-engine-v7" / "models"


C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_ORANGE = "\033[38;5;214m"
C_GREEN = "\033[38;5;114m"
C_RED = "\033[38;5;203m"


def log(msg: str, color: str = "") -> None:
    if color:
        print(f"{color}{msg}{C_RESET}")
    else:
        print(msg)


def log_step(step: int, msg: str) -> None:
    print(f"{C_ORANGE}[{step}/6]{C_RESET} {C_BOLD}{msg}{C_RESET}")


def log_error(msg: str) -> None:
    print(f"{C_RED}Error:{C_RESET} {msg}", file=sys.stderr)


def _detect_default_ck_threads() -> int:
    """Best-effort physical core count; prefer physical cores on HT systems."""
    try:
        logical = len(os.sched_getaffinity(0))
    except Exception:
        logical = os.cpu_count() or 1
    logical = max(1, int(logical or 1))

    physical = 0
    threads_per_core = 0

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

    if physical <= 1:
        try:
            out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
            kv: dict[str, str] = {}
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


def _host_machine() -> str:
    return (os.uname().machine if hasattr(os, "uname") else "").lower()


def _is_x86_machine(machine: str) -> bool:
    return machine in {"x86_64", "amd64", "i386", "i486", "i586", "i686"}


def _arch_defines(machine: str) -> list[str]:
    if _is_x86_machine(machine):
        return ["-DCK_TARGET_X86=1"]
    if _is_arm_machine(machine):
        return ["-DCK_TARGET_ARM=1"]
    return []


def _hf_resolve_url(repo_id: str, filename: str) -> str:
    quoted = urllib.parse.quote(str(filename).lstrip("/"), safe="/")
    return f"https://huggingface.co/{repo_id}/resolve/main/{quoted}?download=true"


def _direct_hf_download_gguf(repo_id: str, filename: str, dst: Path) -> bool:
    url = _hf_resolve_url(repo_id, filename)
    tmp_dst = dst.with_suffix(dst.suffix + ".part")
    token = os.environ.get("HF_TOKEN", "").strip()

    def _promote() -> bool:
        # Never promote an HTML error page (e.g. an HTTP 429 body) to the
        # final artifact path: validate the GGUF magic before the rename.
        if tmp_dst.exists() and _validate_gguf_magic(tmp_dst):
            tmp_dst.replace(dst)
            return True
        tmp_dst.unlink(missing_ok=True)
        return False

    wget = shutil.which("wget")
    if wget:
        cmd = [wget, "-c", url, "-O", str(tmp_dst)]
        if token:
            cmd[1:1] = ["--header", f"Authorization: Bearer {token}"]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0 and _promote():
            return True

    curl = shutil.which("curl")
    if curl:
        # --fail: without it curl exits 0 on HTTP 4xx/5xx and saves the error
        # page body, which then poisons the GGUF cache path.
        cmd = [curl, "-fL", "--continue-at", "-", url, "-o", str(tmp_dst)]
        if token:
            cmd[1:1] = ["-H", f"Authorization: Bearer {token}"]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0 and _promote():
            return True

    return False
def _compiler_supports_openmp(compiler: str, omp_flag: str) -> bool:
    probe = tempfile.NamedTemporaryFile("w", suffix=".c", delete=False)
    try:
        probe.write("#include <omp.h>\nint main(void){return omp_get_max_threads() > 0 ? 0 : 0;}\n")
        probe.flush()
        probe.close()
        out_path = probe.name + ".out"
        cmd = [compiler, probe.name, omp_flag, "-o", out_path]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            return bool(shutil.which(compiler))
        try:
            if os.path.exists(out_path):
                os.unlink(out_path)
        except OSError:
            pass
        return proc.returncode == 0
    finally:
        try:
            os.unlink(probe.name)
        except OSError:
            pass
def _parse_requirement_packages() -> list[str]:
    packages: list[str] = []
    if not V8_REQUIREMENTS_PATH.exists():
        return packages
    for raw in V8_REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--") or line.startswith("-e "):
            continue
        if "://" in line:
            continue
        pkg = line
        for sep in ("<", ">", "=", "!", "~", "[", " ", "\t"):
            if sep in pkg:
                pkg = pkg.split(sep, 1)[0]
        pkg = pkg.strip()
        if pkg:
            packages.append(pkg)
    return packages


def _missing_python_packages() -> list[str]:
    missing: list[str] = []
    for pkg in _parse_requirement_packages():
        if importlib.util.find_spec(pkg.replace("-", "_")) is None:
            missing.append(pkg)
    return missing


def _reexec_into_repo_venv_if_possible(missing: list[str]) -> bool:
    if os.environ.get("CK_V8_REEXECED") == "1":
        return False
    if not REPO_VENV_PY.exists():
        return False
    current_python = Path(sys.executable).expanduser()
    if not current_python.is_absolute():
        current_python = (Path.cwd() / current_python).resolve()
    target_python = REPO_VENV_PY.expanduser()
    if not target_python.is_absolute():
        target_python = (Path.cwd() / target_python).resolve()
    try:
        if current_python == target_python:
            return False
    except OSError:
        return False
    env = os.environ.copy()
    env["CK_V8_REEXECED"] = "1"
    os.execve(str(REPO_VENV_PY), [str(REPO_VENV_PY), __file__, *sys.argv[1:]], env)
    return True


def _ensure_v8_python_requirements(command: Optional[str]) -> None:
    if command not in {None, "run"}:
        return
    missing = _missing_python_packages()
    if not missing:
        return
    _reexec_into_repo_venv_if_possible(missing)
    joined = ", ".join(missing)
    raise SystemExit(
        "Missing Python packages for v8 inference: "
        f"{joined}\nUse version/v8/scripts/cks-v8-run or install requirements-v8.txt."
    )


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    shown = " ".join(str(part) for part in cmd)
    log(f"  $ {shown}", C_DIM)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def _path_to_make_target(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def _sync_runtime_lib(src: Path, dst: Path, label: str) -> None:
    if not src.exists():
        raise RuntimeError(f"missing required runtime lib: {src}")
    if dst.exists():
        try:
            if (
                dst.stat().st_size == src.stat().st_size
                and _sha256_file(dst) == _sha256_file(src)
            ):
                src_stat = src.stat()
                os.utime(
                    dst,
                    ns=(src_stat.st_atime_ns, src_stat.st_mtime_ns),
                )
                log(f"  Refreshed identical {label} -> {dst}", C_DIM)
                return
        except OSError:
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=dst.parent,
        prefix=f".{dst.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(src, temporary)
        os.replace(temporary, dst)
    finally:
        temporary.unlink(missing_ok=True)
    log(f"  Synced {label} -> {dst}", C_DIM)


def _validate_runtime_bundle(bundle_dir: Path) -> None:
    """Load the assembled model exactly as chat will before publishing it."""
    model_lib = bundle_dir / "libmodel.so"
    engine_lib = bundle_dir / "libckernel_engine.so"
    tokenizer_lib = bundle_dir / "libckernel_tokenizer.so"
    missing = [
        path.name
        for path in (model_lib, engine_lib, tokenizer_lib)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "incomplete v8 runtime bundle; missing " + ", ".join(missing)
        )

    probe = (
        "import ctypes, pathlib, sys\n"
        "bundle = pathlib.Path(sys.argv[1])\n"
        "ctypes.CDLL(str(bundle / 'libmodel.so'), mode=ctypes.RTLD_GLOBAL)\n"
    )
    env = os.environ.copy()
    current_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        str(bundle_dir)
        if not current_ld_path
        else f"{bundle_dir}{os.pathsep}{current_ld_path}"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, str(bundle_dir)],
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "dynamic loader failed").strip()
        raise RuntimeError(
            "v8 runtime bundle failed dynamic-load validation before publication:\n"
            f"{detail}"
        )


def detect_input_type(model_input: str) -> tuple[str, dict[str, Any]]:
    text = str(model_input).strip()
    path = Path(text).expanduser()

    if text.startswith("hf://") and text.endswith(".gguf"):
        parts = text[5:].split("/")
        if len(parts) >= 3:
            return "hf_gguf", {"repo_id": f"{parts[0]}/{parts[1]}", "filename": "/".join(parts[2:])}

    if text.startswith("hf://"):
        repo_id = text[5:].strip("/")
        if repo_id:
            return "hf_id", {"model_id": repo_id}

    if path.is_file() and path.suffix.lower() == ".gguf":
        return "gguf", {"path": path.resolve()}

    if path.is_dir():
        return "local_dir", {"path": path.resolve()}

    if path.is_file() and path.suffix.lower() == ".json":
        return "local_config", {"path": path.resolve()}

    if text.startswith("https://huggingface.co/"):
        parts = text.replace("https://huggingface.co/", "").strip("/").split("/")
        if len(parts) >= 2:
            return "hf_id", {"model_id": f"{parts[0]}/{parts[1]}"}

    if "/" in text:
        return "hf_id", {"model_id": text}

    return "hf_id", {"model_id": text}


def _is_map_file(name: str) -> bool:
    if not name.endswith(".json"):
        return False
    return not name.upper().startswith("KERNEL_")


def step_regenerate_kernel_registry(force: bool = False) -> Path:
    if not KERNEL_MAPS_DIR.exists():
        raise RuntimeError(f"kernel map directory not found: {KERNEL_MAPS_DIR}")

    registry_mtime = 0.0
    if KERNEL_REGISTRY_PATH.exists() and not force:
        registry_mtime = KERNEL_REGISTRY_PATH.stat().st_mtime

    needs_regen = not KERNEL_REGISTRY_PATH.exists() or force
    if not needs_regen:
        for path in KERNEL_MAPS_DIR.glob("*.json"):
            if not _is_map_file(path.name):
                continue
            if path.stat().st_mtime > registry_mtime:
                needs_regen = True
                break

    if not needs_regen:
        return KERNEL_REGISTRY_PATH

    log("  Regenerating v8 kernel registry from local kernel maps", C_DIM)
    maps: list[dict[str, Any]] = []
    for path in sorted(KERNEL_MAPS_DIR.glob("*.json")):
        if not _is_map_file(path.name):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        entry = dict(data)
        entry.setdefault("id", data.get("id", path.stem))
        entry.setdefault("name", entry["id"])
        entry["_source_file"] = path.name
        maps.append(entry)
    if not maps:
        raise RuntimeError(f"no kernel maps found in {KERNEL_MAPS_DIR}")

    by_op: dict[str, int] = {}
    for row in maps:
        op = str(row.get("op", "unknown"))
        by_op[op] = by_op.get(op, 0) + 1

    registry = {
        "_meta": {
            "description": "Kernel registry generated from v8 kernel maps",
            "version": "v8",
            "generated_by": "ck_run_v8.py",
            "source_dir": str(KERNEL_MAPS_DIR.relative_to(PROJECT_ROOT)),
            "counts": {"total": len(maps), "by_op": dict(sorted(by_op.items()))},
        },
        "kernels": maps,
    }
    KERNEL_REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return KERNEL_REGISTRY_PATH


def step_download(model_id: str, cache_dir: Path, force: bool = False) -> Path:
    log_step(1, f"Downloading {model_id}")
    if not force:
        repo_dir = model_id.replace("/", "--")
        for root in _cache_roots():
            candidate_dir = root / repo_dir
            if candidate_dir.exists() and (
                any(candidate_dir.glob("*.gguf"))
                or any(candidate_dir.glob("*.safetensors"))
                or (candidate_dir / "model.safetensors.index.json").exists()
            ):
                if root == cache_dir:
                    log(f"  Using cached model at {candidate_dir}", C_DIM)
                else:
                    log(f"  Reusing cached model at {candidate_dir}", C_DIM)
                return candidate_dir
    model_dir = cache_dir / model_id.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub not installed") from exc
    def _fetch() -> None:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            cache_dir=str(_hf_hub_cache_dir(cache_dir)),
            ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot"],
        )

    _run_with_download_retries(model_id, "(snapshot)", _fetch)
    return model_dir


def step_download_gguf(repo_id: str, filename: str, cache_dir: Path, force: bool = False) -> Path:
    log_step(1, f"Downloading {filename} from {repo_id}")
    if not force:
        repo_dir = repo_id.replace("/", "--")
        filename_only = Path(filename).name
        for root in _cache_roots():
            candidate = root / repo_dir / filename_only
            if candidate.exists():
                if not _validate_gguf_magic(candidate):
                    log(f"  Discarding invalid cached GGUF at {candidate}", C_DIM)
                    candidate.unlink(missing_ok=True)
                    continue
                if root == cache_dir:
                    log(f"  Using cached GGUF at {candidate}", C_DIM)
                else:
                    log(f"  Reusing cached GGUF at {candidate}", C_DIM)
                return candidate
    model_dir = cache_dir / repo_id.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = model_dir / Path(filename).name
    machine = _host_machine()

    # On low-memory ARM boards, prefer external downloaders because the Python
    # HF path has been observed to get OOM-killed on multi-GB GGUF fetches.
    if _is_arm_machine(machine):
        if gguf_path.exists() and force:
            gguf_path.unlink()
        if _direct_hf_download_gguf(repo_id, filename, gguf_path):
            return gguf_path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub not installed") from exc
    # Embedded ARM boards are more stable with the plain HTTP path than the
    # optional transfer/Xet backends.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    download_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": filename,
        "local_dir": str(model_dir),
        "cache_dir": str(_hf_hub_cache_dir(cache_dir)),
    }
    try:
        sig = inspect.signature(hf_hub_download)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        params = sig.parameters
        if "force_download" in params:
            download_kwargs["force_download"] = force
        if "resume_download" in params:
            download_kwargs["resume_download"] = True
        if "local_dir_use_symlinks" in params:
            download_kwargs["local_dir_use_symlinks"] = False
        if "token" in params:
            token = os.environ.get("HF_TOKEN")
            if token:
                download_kwargs["token"] = token
    def _fetch() -> Path:
        try:
            downloaded = Path(
                hf_hub_download(**download_kwargs)
            )
        except Exception:
            if _direct_hf_download_gguf(repo_id, filename, gguf_path):
                return gguf_path
            raise
        if not _validate_gguf_magic(downloaded):
            # The hub returned a non-GGUF payload (e.g. a rate-limit error
            # page). Drop it instead of leaving it at the final artifact
            # path, and treat it as transient so the outer retry can pause.
            try:
                downloaded.unlink()
            except OSError:
                pass
            raise _DownloadErrorPage(
                f"server returned a non-GGUF payload for {repo_id}/{filename}"
            )
        if downloaded.resolve() != gguf_path.resolve():
            shutil.move(str(downloaded), str(gguf_path))
        return gguf_path

    return _run_with_download_retries(repo_id, filename, _fetch)


def _strip_gguf_suffix(model_id: str) -> str:
    lower = model_id.lower()
    for suffix in ("-gguf", "_gguf", ".gguf"):
        if lower.endswith(suffix):
            return model_id[:-len(suffix)]
    return model_id


def _find_cached_tokenizer_json(repo_id: str) -> Path | None:
    repo_dir = repo_id.replace("/", "--")
    for root in _cache_roots():
        candidate = root / repo_dir / "tokenizer.json"
        if candidate.exists():
            if _validate_json_file(candidate):
                return candidate
            candidate.unlink(missing_ok=True)
        nested = root / repo_dir / ".ck_build" / "tokenizer.json"
        if nested.exists():
            if _validate_json_file(nested):
                return nested
            nested.unlink(missing_ok=True)
    return None


def ensure_tokenizer_files(model_id: str, work_dir: Path) -> None:
    tokenizer_path = work_dir / "tokenizer.json"
    if tokenizer_path.exists():
        if _validate_json_file(tokenizer_path):
            return
        tokenizer_path.unlink(missing_ok=True)
    candidates = []
    base_id = _strip_gguf_suffix(model_id)
    if base_id != model_id:
        candidates.append(base_id)
    candidates.append(model_id)

    for repo_id in candidates:
        cached = _find_cached_tokenizer_json(repo_id)
        if cached is not None:
            tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, tokenizer_path)
            return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return
    transient_error: V8DownloadError | None = None
    for repo_id in candidates:
        try:
            def _fetch() -> None:
                hf_hub_download(
                    repo_id=repo_id,
                    filename="tokenizer.json",
                    local_dir=str(work_dir),
                    cache_dir=str(_hf_hub_cache_dir(CACHE_DIR)),
                )
                if tokenizer_path.exists() and not _validate_json_file(tokenizer_path):
                    # A rate-limit/error-page payload must not linger at the
                    # tokenizer path; drop it and treat the fetch as transient.
                    tokenizer_path.unlink(missing_ok=True)
                    raise _DownloadErrorPage(
                        f"server returned a non-JSON payload for {repo_id}/tokenizer.json"
                    )

            _run_with_download_retries(repo_id, "tokenizer.json", _fetch)
            if tokenizer_path.exists():
                return
        except V8DownloadError as exc:
            cause = exc.__cause__ if isinstance(exc.__cause__, BaseException) else exc
            retryable, _ = _classify_download_error(cause)
            if retryable:
                transient_error = exc
            # A tokenizer may legitimately live only in the base repository,
            # so permanent errors (notably 404) remain optional here.
            continue
        except Exception:
            continue
    if transient_error is not None:
        raise transient_error


def _find_local_gguf(model_dir: Path) -> Optional[Path]:
    candidates: list[Path] = []
    for pattern in ("*.gguf", "*/*.gguf"):
        for path in model_dir.glob(pattern):
            if path.is_file():
                candidates.append(path.resolve())
    if not candidates:
        return None
    preferred = []
    for needle in ("q4_k_m", "q4_k", "q6_k", "q8_0"):
        for path in candidates:
            if needle in path.name.lower():
                preferred.append(path)
        if preferred:
            break
    pool = preferred if preferred else sorted(set(candidates))
    return pool[0]


def _is_safetensors_checkpoint_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").exists()
        and (
            any(path.glob("*.safetensors"))
            or (path / "model.safetensors.index.json").exists()
        )
    )


def _is_runtime_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "weights.bump").exists()
        and (path / "weights_manifest.json").exists()
        and (path / "config.json").exists()
    )


def _copy_optional(src: Path, dst: Path) -> None:
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _prepare_runtime_dir_from_local_artifacts(model_dir: Path, work_dir: Path) -> tuple[Path, Path, Path]:
    src_bump = model_dir / "weights.bump"
    src_manifest = model_dir / "weights_manifest.json"
    src_config = model_dir / "config.json"
    if not src_bump.exists() or not src_manifest.exists() or not src_config.exists():
        raise RuntimeError(f"missing runtime artifacts in {model_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    dst_bump = work_dir / "weights.bump"
    dst_manifest = work_dir / "weights_manifest.json"
    dst_config = work_dir / "config.json"
    if model_dir.resolve() != work_dir.resolve():
        shutil.copy2(src_bump, dst_bump)
        shutil.copy2(src_manifest, dst_manifest)
        shutil.copy2(src_config, dst_config)
        _copy_optional(model_dir / "weights_manifest.map", work_dir / "weights_manifest.map")
        _copy_optional(model_dir / "tokenizer.json", work_dir / "tokenizer.json")
        if (model_dir / "tokenizer_bin").is_dir() and not (work_dir / "tokenizer_bin").exists():
            shutil.copytree(model_dir / "tokenizer_bin", work_dir / "tokenizer_bin")
    else:
        dst_bump = src_bump
        dst_manifest = src_manifest
        dst_config = src_config
    return dst_bump, dst_config, dst_manifest


def _stage_safetensors_tokenizer_assets(checkpoint_dir: Path, output_dir: Path) -> None:
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "tiktoken.model",
        "tokenizer.model",
    ):
        _copy_optional(checkpoint_dir / name, output_dir / name)
    for path in checkpoint_dir.glob("tokenization_*.py"):
        _copy_optional(path, output_dir / path.name)
    if (checkpoint_dir / "tokenizer_bin").is_dir() and not (output_dir / "tokenizer_bin").exists():
        shutil.copytree(checkpoint_dir / "tokenizer_bin", output_dir / "tokenizer_bin")


def step_convert_safetensors(
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    log_step(2, "Converting safetensors checkpoint to bump format")
    weights_path = output_dir / "weights.bump"
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "weights_manifest.json"
    if weights_path.exists() and config_path.exists() and manifest_path.exists() and not force:
        log(f"  Using cached weights at {weights_path}", C_DIM)
        _stage_safetensors_tokenizer_assets(checkpoint_dir, output_dir)
        return weights_path, config_path, manifest_path
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "convert_safetensors_to_bump_v8.py"),
        "--checkpoint",
        str(checkpoint_dir),
        "--output",
        str(weights_path),
        "--config-out",
        str(config_path),
        "--manifest-out",
        str(manifest_path),
        "--arch",
        "auto",
    ]
    run_cmd(cmd, cwd=PROJECT_ROOT)
    _stage_safetensors_tokenizer_assets(checkpoint_dir, output_dir)
    return weights_path, config_path, manifest_path


def step_convert_gguf(
    gguf_path: Path,
    output_dir: Path,
    *,
    force: bool = False,
    context_len: int | None = None,
) -> tuple[Path, Path, Path]:
    log_step(2, "Converting GGUF to bump format")
    weights_path = output_dir / "weights.bump"
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "weights_manifest.json"
    if weights_path.exists() and config_path.exists() and manifest_path.exists() and not force:
        log(f"  Using cached weights at {weights_path}", C_DIM)
        return weights_path, config_path, manifest_path
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "convert_gguf_to_bump_v8.py"),
        "--gguf",
        str(gguf_path),
        "--output",
        str(weights_path),
        "--config-out",
        str(config_path),
        "--manifest-out",
        str(manifest_path),
    ]
    # Preserve model-native context / RoPE metadata during conversion.
    # The active runtime window is applied later in build_ir_v8 via --context-len,
    # matching the v7 text path and avoiding qwen35 long-context regressions.
    _ = context_len
    run_cmd(cmd, cwd=PROJECT_ROOT)
    return weights_path, config_path, manifest_path


def step_build_ir(
    manifest_path: Path,
    output_dir: Path,
    *,
    force: bool = False,
    context_len: int | None = None,
    prefill_chunk_len: int | None = None,
    logits_layout: str | None = None,
    prefill_policy: str = "auto",
) -> dict[str, Path]:
    log_step(3, "Building v8 IR")
    output_dir.mkdir(parents=True, exist_ok=True)
    step_regenerate_kernel_registry(force=force)
    force = _refresh_manifest_circuit_snapshot(manifest_path) or force

    outputs = {
        "init_ir": output_dir / "init.json",
        "init_call": output_dir / "init_call.json",
        "prefill_ir": output_dir / "ir1_prefill.json",
        "prefill_layout": output_dir / "layout_prefill.json",
        "prefill_lowered": output_dir / "lowered_prefill.json",
        "prefill_call": output_dir / "lowered_prefill_call.json",
        "decode_ir": output_dir / "ir1_decode.json",
        "decode_layout": output_dir / "layout_decode.json",
        "decode_lowered": output_dir / "lowered_decode.json",
        "decode_call": output_dir / "lowered_decode_call.json",
        "manifest_map": output_dir / "weights_manifest.map",
    }

    decode_ready = all(path.exists() for path in outputs.values())
    stamp_path = output_dir / IR_STAMP_NAME
    ir_inputs = {
        "schema": "ck-v8-ir-bundle-v1",
        "manifest": _file_identity(manifest_path),
        "context_len": int(context_len) if context_len is not None else None,
        "prefill_chunk_len": int(prefill_chunk_len) if prefill_chunk_len is not None else None,
        "logits_layout": logits_layout,
        "prefill_policy": str(prefill_policy),
        "planner_sources": _tree_identity(
            [
                SCRIPTS_DIR / "build_ir_v8.py",
                SCRIPTS_DIR / "memory_planner_v8.py",
                SCRIPTS_DIR / "validate_circuit_interfaces_v8.py",
                SCRIPTS_DIR / "resolve_layout_chain_v8.py",
                SCRIPTS_DIR / "resolve_attention_contracts_v8.py",
                SCRIPTS_DIR / "resolve_numerical_execution_contracts_v8.py",
                V8_ROOT / "contracts",
                V8_ROOT / "schemas",
                KERNEL_REGISTRY_PATH,
            ]
        ),
    }
    if decode_ready and not force and _bundle_is_current(
        stamp_path,
        inputs=ir_inputs,
        outputs=outputs,
    ):
        log(f"  Using cached IR artifacts in {output_dir}", C_DIM)
        return outputs
    if decode_ready and not force:
        log("  Cached IR provenance is stale; rebuilding", C_DIM)

    def _build_mode(mode: str) -> None:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "build_ir_v8.py"),
            "--manifest",
            str(manifest_path),
            "--mode",
            mode,
            "--output",
            str(outputs[f"{mode}_ir"]),
            "--layout-output",
            str(outputs[f"{mode}_layout"]),
            "--lowered-output",
            str(outputs[f"{mode}_lowered"]),
            "--call-output",
            str(outputs[f"{mode}_call"]),
        ]
        if mode == "decode":
            cmd.extend(["--manifest-map-output", str(outputs["manifest_map"])])
            cmd.extend(["--init-output", str(outputs["init_ir"])])
        if context_len is not None:
            cmd.extend(["--context-len", str(int(context_len))])
        if prefill_chunk_len is not None:
            cmd.extend(["--prefill-chunk-len", str(int(prefill_chunk_len))])
        if logits_layout:
            cmd.extend(["--logits-layout", logits_layout])
        cmd.extend(["--prefill-policy-override", str(prefill_policy)])
        run_cmd(cmd, cwd=PROJECT_ROOT)

    _build_mode("prefill")
    _build_mode("decode")
    _write_bundle_stamp(
        stamp_path,
        {
            "inputs": ir_inputs,
            "outputs": {
                name: _file_identity(path)
                for name, path in sorted(outputs.items())
            },
        },
    )
    return outputs


def _refresh_manifest_circuit_snapshot(manifest_path: Path) -> bool:
    """Refresh cached graph policy without reconverting immutable weights."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded = manifest.get("template") if isinstance(manifest.get("template"), dict) else {}
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    circuit_name = str(embedded.get("name") or config.get("model") or "").strip().lower()
    if not circuit_name:
        return False
    circuit_path = V8_ROOT / "circuits" / f"{circuit_name}.json"
    if not circuit_path.is_file():
        return False
    current = json.loads(circuit_path.read_text(encoding="utf-8"))
    if not isinstance(current, dict):
        raise RuntimeError(f"versioned circuit is not a JSON object: {circuit_path}")
    if embedded == current:
        return False
    manifest["template"] = current
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"  Refreshed cached circuit snapshot: {circuit_name}", C_DIM)
    return True


def step_codegen(output_dir: Path, ir_paths: dict[str, Path], *, force: bool = False, profile: bool = False) -> Path:
    log_step(4, "Generating C code")
    model_c_path = output_dir / "model_v8.c"
    stamp_path = output_dir / CODEGEN_STAMP_NAME
    generation_config_path = output_dir / "generation_config.json"
    codegen_artifacts = dict(ir_paths)
    if generation_config_path.is_file():
        codegen_artifacts["generation_config"] = generation_config_path
    codegen_inputs = {
        "schema": "ck-v8-codegen-bundle-v1",
        "profile": bool(profile),
        "artifacts": {
            name: _file_identity(path)
            for name, path in sorted(codegen_artifacts.items())
            if path.is_file()
        },
        "generator_sources": _tree_identity(
            [
                SCRIPTS_DIR / "codegen_v8.py",
                SCRIPTS_DIR / "codegen_core_v8.py",
                SCRIPTS_DIR / "codegen_prefill_v8.py",
                PROJECT_ROOT / "include" / "ck_model_abi_v8.h",
                KERNEL_REGISTRY_PATH,
            ]
        ),
    }
    if not force and _bundle_is_current(
        stamp_path,
        inputs=codegen_inputs,
        outputs={"model_v8.c": model_c_path},
    ):
        log(f"  Using cached C code at {model_c_path}", C_DIM)
        return model_c_path
    if model_c_path.exists() and not force:
        log("  Cached C code provenance is stale; regenerating", C_DIM)

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "codegen_v8.py"),
        "--ir",
        str(ir_paths["decode_call"]),
        "--prefill",
        str(ir_paths["prefill_call"]),
        "--prefill-layout",
        str(ir_paths["prefill_layout"]),
        "--layout",
        str(ir_paths["decode_layout"]),
        "--output",
        str(model_c_path),
    ]
    if profile:
        cmd.append("--profile")
    if generation_config_path.is_file():
        cmd.extend(["--generation-config", str(generation_config_path)])
    run_cmd(cmd, cwd=PROJECT_ROOT)
    _write_bundle_stamp(
        stamp_path,
        {
            "inputs": codegen_inputs,
            "outputs": {"model_v8.c": _file_identity(model_c_path)},
        },
    )
    return model_c_path


def step_compile(
    model_c_path: Path,
    output_dir: Path,
    *,
    force: bool = False,
    profile: bool = False,
) -> Path:
    log_step(5, "Compiling shared library")
    lib_path = output_dir / "libmodel.so"
    kernel_lib = BUILD_DIR / "libckernel_engine.so"
    tokenizer_lib = BUILD_DIR / "libckernel_tokenizer.so"

    requested_compiler = (os.environ.get("CK_V8_COMPILER", "") or os.environ.get("CK_V7_COMPILER", "")).strip()
    compiler = requested_compiler or "gcc"
    compiler_path_value = shutil.which(compiler)
    if not compiler_path_value:
        log_error(f"Requested CK_V8_COMPILER not found in PATH: {compiler}")
        sys.exit(1)
    compiler_path = Path(compiler_path_value)

    machine = _host_machine()
    omp_flag = "-qopenmp" if compiler == "icx" else "-fopenmp"
    extra_cflags = (os.environ.get("CK_V8_EXTRA_CFLAGS", "") or os.environ.get("CK_V7_EXTRA_CFLAGS", "")).strip()
    extra_ldflags = (os.environ.get("CK_V8_EXTRA_LDFLAGS", "") or os.environ.get("CK_V7_EXTRA_LDFLAGS", "")).strip()
    compile_inputs = {
        "schema": RUNTIME_BUNDLE_SCHEMA,
        "model_source": _file_identity(model_c_path),
        "compiler": _optional_file_identity(compiler_path),
        "compiler_name": compiler,
        "machine": machine,
        "openmp_flag": omp_flag,
        "profile": bool(profile),
        "extra_cflags": extra_cflags,
        "extra_ldflags": extra_ldflags,
        "engine_sources": _tree_identity(
            [
                PROJECT_ROOT / "Makefile",
                PROJECT_ROOT / "include",
                PROJECT_ROOT / "src" / "kernels",
                PROJECT_ROOT / "src" / "ckernel_alloc.c",
                V8_ROOT / "src",
                KERNEL_REGISTRY_PATH,
            ]
        ),
        "linked_libraries": {
            name: _file_identity(path)
            for name, path in (
                ("engine", kernel_lib),
                ("tokenizer", tokenizer_lib),
            )
            if path.is_file()
        },
    }
    stamp_path = output_dir / RUNTIME_STAMP_NAME
    runtime_outputs = {
        "libmodel.so": lib_path,
        "libckernel_engine.so": output_dir / "libckernel_engine.so",
        "libckernel_tokenizer.so": output_dir / "libckernel_tokenizer.so",
    }
    if not force and _bundle_is_current(
        stamp_path,
        inputs=compile_inputs,
        outputs=runtime_outputs,
    ):
        symlink_path = output_dir / "ck-kernel-inference.so"
        try:
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to("libmodel.so")
        except Exception:
            pass
        log(f"  Using provenance-verified runtime at {lib_path}", C_DIM)
        return lib_path

    if lib_path.exists() and not force:
        log("  Cached runtime provenance is stale; rebuilding", C_DIM)
    targets = [_path_to_make_target(kernel_lib), _path_to_make_target(tokenizer_lib)]
    # A stale bundle means source identity changed even when checkout/build
    # mtimes make the old libraries look newer. Force both runtime libraries
    # through Make so generated model code can never be paired with an engine
    # from another branch or worktree.
    make_cmd = ["make", "-B", "--no-print-directory", f"CC={compiler}"]
    run_cmd([*make_cmd, *targets], cwd=PROJECT_ROOT)

    include_dir = PROJECT_ROOT / "include"
    v8_src = V8_ROOT / "src"
    use_openmp = _compiler_supports_openmp(compiler, omp_flag)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir,
        prefix=".ck-runtime-stage-",
    ) as stage_name:
        stage_dir = Path(stage_name)
        staged_model = stage_dir / "libmodel.so"
        cmd = [
            compiler,
            "-shared",
            "-fPIC",
            "-O3",
            "-march=native",
            "-std=c11",
            "-fvisibility=default",
            *_arch_defines(machine),
            f"-I{include_dir}",
            f"-I{v8_src}",
            "-o",
            str(staged_model),
            str(model_c_path),
            str(PROJECT_ROOT / "src" / "ckernel_alloc.c"),
            str(v8_src / "ckernel_model_load_v8.c"),
            str(v8_src / "ck_parallel_decode_v8.c"),
            str(v8_src / "ck_parallel_prefill_v8.c"),
            f"-L{BUILD_DIR}",
            "-lckernel_tokenizer",
            "-lckernel_engine",
            "-lm",
            "-lpthread",
            "-Wl,-rpath,$ORIGIN",
        ]
        if _is_x86_machine(machine):
            cmd.insert(3, "-mcmodel=large")
        if use_openmp:
            cmd.insert(8 if _is_x86_machine(machine) else 7, omp_flag)
        if profile:
            cmd.append("-DCK_PROFILE=1")
        if extra_cflags:
            try:
                cmd.extend(shlex.split(extra_cflags))
            except ValueError:
                log_error("Invalid CK_V8_EXTRA_CFLAGS value")
                sys.exit(1)
        if extra_ldflags:
            try:
                cmd.extend(shlex.split(extra_ldflags))
            except ValueError:
                log_error("Invalid CK_V8_EXTRA_LDFLAGS value")
                sys.exit(1)
        run_cmd(cmd, cwd=PROJECT_ROOT)
        shutil.copy2(kernel_lib, stage_dir / "libckernel_engine.so")
        shutil.copy2(tokenizer_lib, stage_dir / "libckernel_tokenizer.so")
        _validate_runtime_bundle(stage_dir)
        _sync_runtime_lib(
            stage_dir / "libckernel_engine.so",
            output_dir / "libckernel_engine.so",
            "libckernel_engine.so",
        )
        _sync_runtime_lib(
            stage_dir / "libckernel_tokenizer.so",
            output_dir / "libckernel_tokenizer.so",
            "libckernel_tokenizer.so",
        )
        # Publish the generated model last. It is the bundle consumer and may
        # require symbols that exist only in the newly staged dependencies.
        _sync_runtime_lib(staged_model, lib_path, "libmodel.so")
    symlink_path = output_dir / "ck-kernel-inference.so"
    try:
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to("libmodel.so")
    except Exception:
        pass
    compile_inputs["linked_libraries"] = {
        "engine": _file_identity(kernel_lib),
        "tokenizer": _file_identity(tokenizer_lib),
    }
    _write_bundle_stamp(
        stamp_path,
        {
            "inputs": compile_inputs,
            "outputs": {
                name: _file_identity(path)
                for name, path in runtime_outputs.items()
            },
        },
    )
    return lib_path


def _generate_visualizer_html(work_dir: Path) -> Path:
    """Generate ir_report.html for a v8 run directory without running extra probes."""
    if not V8_VISUALIZER_PATH.exists():
        raise RuntimeError(f"Visualizer script not found: {V8_VISUALIZER_PATH}")
    cmd = [
        sys.executable,
        str(V8_VISUALIZER_PATH),
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


def step_sweep_kernels(work_dir: Path, ir_paths: dict[str, Path], *, quick: bool = True) -> Path:
    log(f"\n{C_ORANGE}[tune]{C_RESET} Sweeping kernel dispatch choices", C_DIM)
    report_path = work_dir / "kernel_tuning.json"
    engine_lib = work_dir / "libckernel_engine.so"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "benchmarks" / "sweep_q6k_prefill_dispatch.py"),
        "--from-lowered",
        str(ir_paths["prefill_call"]),
        "--engine-lib",
        str(engine_lib),
        "--json-out",
        str(report_path),
        "--threads",
        str(_detect_default_ck_threads()),
    ]
    if quick:
        cmd.append("--quick")
    run_cmd(cmd, cwd=PROJECT_ROOT)
    log(f"  Kernel tuning: {report_path}", C_GREEN)
    return report_path



def _qwen3vl_ocr_fast_profile_enabled(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    alias = env.get("CK_QWEN3VL_OCR_FAST", "")
    if alias and alias != "0":
        return True
    profile = env.get("CK_SPEED_PROFILE", "")
    return profile in {"qwen3vl_ocr_xeon_avx512", "qwen3vl_ocr_fast", "qwen3vl_ocr"}


def _apply_qwen3vl_ocr_fast_defaults(env: dict[str, str]) -> None:
    if not _qwen3vl_ocr_fast_profile_enabled(env):
        return
    env.setdefault("CK_ENABLE_Q80_FP32_M4N4", "1")
    env.setdefault("CK_Q4K_GATEUP_SWIGLU_X16_THREAD_CAP", "16")
    env.setdefault("CK_ATTENTION_QBLOCK4", "1")
    env.setdefault("CK_ATTENTION_THREAD_CAP", "16")
    env.setdefault("CK_Q4K_PACKED_META_X8_MAX_M", "2048")
    env.setdefault("CK_NUM_THREADS", "20")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMP_DYNAMIC", "FALSE")

def step_run_chat(work_dir: Path, args: argparse.Namespace, *, gguf_path: Path | None) -> None:
    log_step(6, "Starting chat")
    kernel_lib = BUILD_DIR / "libckernel_engine.so"
    tokenizer_lib = BUILD_DIR / "libckernel_tokenizer.so"
    _sync_runtime_lib(kernel_lib, work_dir / "libckernel_engine.so", "libckernel_engine.so")
    _sync_runtime_lib(tokenizer_lib, work_dir / "libckernel_tokenizer.so", "libckernel_tokenizer.so")

    env = os.environ.copy()
    ld_items = [str(BUILD_DIR), str(work_dir)]
    if env.get("LD_LIBRARY_PATH"):
        ld_items.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_items)
    _apply_qwen3vl_ocr_fast_defaults(env)
    env.setdefault("CK_NUM_THREADS", str(_detect_default_ck_threads()))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMP_DYNAMIC", "FALSE")

    cmd = [
        sys.executable,
        str(ROOT_SCRIPTS_DIR / "ck_chat.py"),
        "--model-dir",
        str(work_dir),
    ]
    if gguf_path is not None:
        cmd.extend(["--gguf", str(gguf_path)])
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.max_tokens is not None:
        cmd.extend(["--max-tokens", str(int(args.max_tokens))])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(int(args.top_k))])
    if args.top_p is not None:
        cmd.extend(["--top-p", str(float(args.top_p))])
    if args.min_p is not None:
        cmd.extend(["--min-p", str(float(args.min_p))])
    if args.repeat_penalty is not None:
        cmd.extend(["--repeat-penalty", str(float(args.repeat_penalty))])
    if args.repeat_last_n is not None:
        cmd.extend(["--repeat-last-n", str(int(args.repeat_last_n))])
    no_repeat_ngram_size = getattr(args, "no_repeat_ngram_size", None)
    if no_repeat_ngram_size is not None:
        cmd.extend(["--no-repeat-ngram-size", str(int(no_repeat_ngram_size))])
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])
    if args.no_chat_template:
        cmd.append("--no-chat-template")
    else:
        cmd.extend(["--chat-template", args.chat_template])
    if args.allow_raw_prompt:
        cmd.append("--allow-raw-prompt")
    if args.thinking_mode:
        cmd.extend(["--thinking-mode", args.thinking_mode])
    if args.python_tokenizer:
        cmd.append("--python-tokenizer")
    if args.memory:
        cmd.append("--memory")
    if getattr(args, "multiline_input", False):
        cmd.append("--multiline-input")
    if getattr(args, "speculative_draft_model_dir", None):
        cmd.extend(["--speculative-draft-model-dir", str(args.speculative_draft_model_dir)])
    if getattr(args, "speculative_draft_tokens", None) is not None:
        cmd.extend(["--speculative-draft-tokens", str(int(args.speculative_draft_tokens))])
    cmd.extend(["--gemm-schedule", str(getattr(args, "gemm_schedule", "auto"))])
    os.execvpe(sys.executable, cmd, env)


def _resolve_gguf_input(model_input: str, *, force_download: bool) -> tuple[Path, str | None]:
    input_type, info = detect_input_type(model_input)
    if input_type == "hf_gguf":
        gguf_path = step_download_gguf(info["repo_id"], info["filename"], CACHE_DIR, force=force_download)
        return gguf_path, info["repo_id"]
    if input_type == "gguf":
        return info["path"], None
    if input_type == "local_dir":
        local_gguf = _find_local_gguf(info["path"])
        if local_gguf is None:
            raise RuntimeError(f"no GGUF found in {info['path']}")
        return local_gguf, None
    if input_type == "hf_id":
        model_dir = step_download(info["model_id"], CACHE_DIR, force=force_download)
        local_gguf = _find_local_gguf(model_dir)
        if local_gguf is None:
            raise RuntimeError(
                f"HF repo {info['model_id']} has no GGUF. "
                "The inference-only v8 runner currently supports GGUF sources only."
            )
        return local_gguf, info["model_id"]
    raise RuntimeError(f"unsupported GGUF input: {model_input}")


def _resolve_run_dir(model_input: str, input_type: str, info: dict[str, Any], requested: str | None) -> Path:
    if requested:
        path = Path(requested).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    if input_type == "hf_gguf":
        return CACHE_DIR / info["repo_id"].replace("/", "--")
    if input_type == "hf_id":
        return CACHE_DIR / info["model_id"].replace("/", "--") / ".ck_build_v8"
    if input_type == "gguf":
        return CACHE_DIR / info["path"].stem
    if input_type == "local_dir":
        local_dir = info["path"]
        if _is_runtime_dir(local_dir):
            return local_dir
        return local_dir / ".ck_build_v8"
    if input_type == "local_config":
        return info["path"].parent / ".ck_build_v8"
    return CACHE_DIR / "unknown"


def run_pipeline(args: argparse.Namespace) -> int:
    model_input = str(args.model)
    input_type, info = detect_input_type(model_input)
    work_dir = _resolve_run_dir(model_input, input_type, info, args.run_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    log(f"{C_ORANGE}C-Kernel-Engine v8{C_RESET}")
    log(f"Input: {model_input} ({input_type})", C_DIM)
    log(f"Run dir: {work_dir}", C_DIM)

    if args.mmproj is not None or args.image_path is not None or args.synthetic_prefix_tokens > 0:
        decoder_gguf, decoder_repo_id = _resolve_gguf_input(model_input, force_download=args.force_download)
        encoder_gguf = None
        mmproj_spec = str(args.mmproj or "").strip() or _infer_auto_mmproj_spec(model_input)
        auto_mmproj = not bool(args.mmproj) and bool(mmproj_spec)
        if mmproj_spec:
            if auto_mmproj:
                log(f"  Auto-selected mmproj: {mmproj_spec}", C_DIM)
            encoder_gguf, _ = _resolve_gguf_input(mmproj_spec, force_download=args.force_download)
        elif args.image_path is not None:
            raise RuntimeError(
                "vision run requested with --image-path but no mmproj was provided and no known default companion "
                f"was found for {model_input}. Pass --mmproj /path/to/mmproj.gguf or --mmproj hf://repo/file.gguf."
            )
        if decoder_repo_id:
            ensure_tokenizer_files(decoder_repo_id, work_dir)
        bridge_dir = work_dir / "multimodal_bridge"
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "run_multimodal_bridge_v8.py"),
            "--decoder-gguf",
            str(decoder_gguf),
            "--workdir",
            str(bridge_dir),
            "--prompt",
            args.prompt or "Describe the image.",
            "--image-mode",
            args.image_mode,
            "--report-top-k",
            str(int(args.vision_top_k)),
            "--gemm-schedule",
            str(getattr(args, "gemm_schedule", "auto")),
        ]
        if args.max_tokens is not None:
            cmd.extend(["--max-tokens", str(int(args.max_tokens))])
        if args.temperature is not None:
            cmd.extend(["--temperature", str(float(args.temperature))])
        if args.top_k is not None:
            cmd.extend(["--sample-top-k", str(int(args.top_k))])
        if args.top_p is not None:
            cmd.extend(["--top-p", str(float(args.top_p))])
        if args.min_p is not None:
            cmd.extend(["--min-p", str(float(args.min_p))])
        if args.repeat_penalty is not None:
            cmd.extend(["--repeat-penalty", str(float(args.repeat_penalty))])
        if args.repeat_last_n is not None:
            cmd.extend(["--repeat-last-n", str(int(args.repeat_last_n))])
        no_repeat_ngram_size = getattr(args, "no_repeat_ngram_size", None)
        if no_repeat_ngram_size is not None:
            cmd.extend(["--no-repeat-ngram-size", str(int(no_repeat_ngram_size))])
        for override in list(getattr(args, "vision_activation_pref", []) or []):
            cmd.extend(["--vision-activation-pref", str(override)])
        if args.no_chat_template:
            cmd.append("--no-chat-template")
        else:
            cmd.extend(["--chat-template", args.chat_template])
        if args.allow_raw_prompt:
            cmd.append("--allow-raw-prompt")
        if args.thinking_mode:
            cmd.extend(["--thinking-mode", args.thinking_mode])
        if encoder_gguf is not None:
            cmd.extend(["--encoder-gguf", str(encoder_gguf)])
        if args.image_path is not None:
            cmd.extend(["--image-path", str(Path(args.image_path).expanduser().resolve())])
        if args.image_min_tokens is not None:
            cmd.extend(["--image-min-tokens", str(int(args.image_min_tokens))])
        if args.image_max_tokens is not None:
            cmd.extend(["--image-max-tokens", str(int(args.image_max_tokens))])
        if args.synthetic_prefix_tokens > 0:
            cmd.extend(["--synthetic-prefix-tokens", str(int(args.synthetic_prefix_tokens))])
        if args.bridge_runtime is not None:
            cmd.extend(["--bridge-runtime", str(args.bridge_runtime)])
        if args.bridge_generation_mode is not None:
            cmd.extend(["--bridge-generation-mode", str(args.bridge_generation_mode)])
        if getattr(args, "profile", False):
            cmd.append("--profile-decoder")
            cmd.append("--profile-encoder")
        if args.context_len is not None:
            cmd.extend(["--decoder-context-len", str(int(args.context_len))])
        run_cmd(cmd, cwd=PROJECT_ROOT)
        log(f"  Wrote bridge report to {bridge_dir / 'bridge_report.json'}", C_GREEN)
        return 0

    gguf_path: Path | None = None
    repo_id_for_tokenizer: str | None = None
    if input_type == "local_dir" and _is_runtime_dir(info["path"]) and not args.force_convert:
        weights_path, config_path, manifest_path = _prepare_runtime_dir_from_local_artifacts(info["path"], work_dir)
        local_gguf = _find_local_gguf(info["path"])
        if local_gguf is not None:
            gguf_path = local_gguf
    elif input_type == "local_dir" and _is_safetensors_checkpoint_dir(info["path"]):
        weights_path, config_path, manifest_path = step_convert_safetensors(
            info["path"],
            work_dir,
            force=args.force_convert,
        )
    elif input_type == "local_config":
        model_dir = info["path"].parent
        if not _is_runtime_dir(model_dir):
            raise RuntimeError(
                f"config-only input requires colocated weights.bump + weights_manifest.json: {model_dir}"
            )
        weights_path, config_path, manifest_path = _prepare_runtime_dir_from_local_artifacts(model_dir, work_dir)
        local_gguf = _find_local_gguf(model_dir)
        if local_gguf is not None:
            gguf_path = local_gguf
    elif input_type == "hf_id":
        model_dir = step_download(info["model_id"], CACHE_DIR, force=args.force_download)
        if _is_safetensors_checkpoint_dir(model_dir):
            weights_path, config_path, manifest_path = step_convert_safetensors(
                model_dir,
                work_dir,
                force=args.force_convert,
            )
        else:
            local_gguf = _find_local_gguf(model_dir)
            if local_gguf is None:
                raise RuntimeError(
                    f"HF repo {info['model_id']} has no GGUF or safetensors checkpoint artifacts. "
                    "Expected *.gguf, *.safetensors, or model.safetensors.index.json."
                )
            gguf_path = local_gguf
            repo_id_for_tokenizer = info["model_id"]
            ensure_tokenizer_files(repo_id_for_tokenizer, work_dir)
            weights_path, config_path, manifest_path = step_convert_gguf(
                gguf_path,
                work_dir,
                force=args.force_convert,
                context_len=args.context_len,
            )
    else:
        gguf_path, repo_id_for_tokenizer = _resolve_gguf_input(model_input, force_download=args.force_download)
        if repo_id_for_tokenizer:
            ensure_tokenizer_files(repo_id_for_tokenizer, work_dir)
        weights_path, config_path, manifest_path = step_convert_gguf(
            gguf_path,
            work_dir,
            force=args.force_convert,
            context_len=args.context_len,
        )

    ir_paths = step_build_ir(
        manifest_path,
        work_dir,
        force=args.force_compile,
        context_len=args.context_len,
        prefill_chunk_len=getattr(args, "prefill_chunk_len", None),
        logits_layout=args.logits_layout,
        prefill_policy=getattr(args, "prefill_policy", "auto"),
    )
    if getattr(args, "plan_only", False):
        log(f"  Memory plan certified without code generation: {work_dir}", C_GREEN)
        return 0
    model_c_path = step_codegen(
        work_dir,
        ir_paths,
        force=args.force_compile,
        profile=getattr(args, "profile", False),
    )
    lib_path = step_compile(
        model_c_path,
        work_dir,
        force=args.force_compile,
        profile=getattr(args, "profile", False),
    )

    if getattr(args, "sweep_kernels", False):
        step_sweep_kernels(
            work_dir,
            ir_paths,
            quick=not getattr(args, "sweep_kernels_full", False),
        )

    if getattr(args, "generate_visualizer", False):
        log(f"\n{C_ORANGE}[viz]{C_RESET} Generating IR visualizer HTML", C_DIM)
        _generate_visualizer_html(work_dir)

    if args.generate_only:
        log(f"\n{C_GREEN}Generated:{C_RESET}")
        log(f"  Weights: {weights_path}")
        log(f"  Config:  {config_path}")
        log(f"  Layout:  {ir_paths['decode_layout']}")
        log(f"  C code:  {model_c_path}")
        log(f"  Library: {lib_path}")
        return 0

    step_run_chat(work_dir, args, gguf_path=gguf_path)
    return 0


def _whisper_checkpoint_identity(checkpoint_dir: Path) -> dict[str, Any]:
    files = sorted(
        path
        for path in checkpoint_dir.iterdir()
        if path.is_file()
        and path.suffix in {".json", ".safetensors", ".txt", ".model"}
    )
    return {
        str(path.relative_to(checkpoint_dir)): _file_identity(path)
        for path in files
    }


def _whisper_build_inputs(
    checkpoint_dir: Path,
    role: str,
    *,
    linear_weight_dtype: str = "preserve",
) -> dict[str, Any]:
    return {
        "schema": "ck-v8-whisper-runtime-inputs-v1",
        "role": role,
        "linear_weight_dtype": linear_weight_dtype,
        "checkpoint": _whisper_checkpoint_identity(checkpoint_dir),
        "sources": _tree_identity(
            [
                SCRIPTS_DIR / "convert_safetensors_to_bump_v8.py",
                SCRIPTS_DIR / "build_ir_v8.py",
                SCRIPTS_DIR / "codegen_v8.py",
                SCRIPTS_DIR / "codegen_core_v8.py",
                SCRIPTS_DIR / "codegen_prefill_v8.py",
                PROJECT_ROOT / "include" / "ck_model_abi_v8.h",
                V8_ROOT / "circuits" / f"audio_transformer_{role}.json",
                V8_ROOT / "model_maps" / "safetensors_ck_map.json",
                KERNEL_REGISTRY_PATH,
            ]
        ),
    }


def _whisper_runtime_outputs(run_dir: Path) -> dict[str, Path]:
    return {
        name: run_dir / name
        for name in (
            "weights.bump",
            "weights_manifest.json",
            "weights_manifest.map",
            "config.json",
            "model_v8.c",
            "libmodel.so",
            "libckernel_engine.so",
            "libckernel_tokenizer.so",
        )
    }


def _build_whisper_role(
    checkpoint_dir: Path,
    run_dir: Path,
    role: str,
    *,
    force_convert: bool,
    force_compile: bool,
    linear_weight_dtype: str = "preserve",
) -> None:
    if role not in {"encoder", "decoder"}:
        raise ValueError(f"unsupported Whisper role: {role}")
    run_dir.mkdir(parents=True, exist_ok=True)
    step_regenerate_kernel_registry(force=False)
    inputs = _whisper_build_inputs(
        checkpoint_dir,
        role,
        linear_weight_dtype=linear_weight_dtype,
    )
    stamp = run_dir / ".ck-whisper-runtime.json"
    if (
        not force_convert
        and not force_compile
        and _bundle_is_current(
            stamp,
            inputs=inputs,
            outputs=_whisper_runtime_outputs(run_dir),
        )
    ):
        log(f"  Using provenance-verified Whisper {role}: {run_dir}", C_DIM)
        return

    weights = run_dir / "weights.bump"
    config = run_dir / "config.json"
    manifest = run_dir / "weights_manifest.json"
    convert_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "convert_safetensors_to_bump_v8.py"),
            "--checkpoint",
            str(checkpoint_dir),
            "--output",
            str(weights),
            "--config-out",
            str(config),
            "--manifest-out",
            str(manifest),
            "--arch",
            f"whisper_{role}",
        ]
    if linear_weight_dtype != "preserve":
        convert_cmd.extend(["--linear-weight-dtype", linear_weight_dtype])
    run_cmd(convert_cmd, cwd=PROJECT_ROOT)
    _stage_safetensors_tokenizer_assets(checkpoint_dir, run_dir)

    if role == "encoder":
        ir = run_dir / "ir1.json"
        layout = run_dir / "layout.json"
        lowered = run_dir / "lowered.json"
        call = run_dir / "call.json"
        run_cmd(
            [
                sys.executable,
                str(SCRIPTS_DIR / "build_ir_v8.py"),
                "--manifest",
                str(manifest),
                "--mode",
                "prefill",
                "--output",
                str(ir),
                "--layout-output",
                str(layout),
                "--lowered-output",
                str(lowered),
                "--call-output",
                str(call),
                "--manifest-map-output",
                str(run_dir / "weights_manifest.map"),
                "--init-output",
                str(run_dir / "init.json"),
            ],
            cwd=PROJECT_ROOT,
        )
        model_c = run_dir / "model_v8.c"
        run_cmd(
            [
                sys.executable,
                str(SCRIPTS_DIR / "codegen_v8.py"),
                "--ir",
                str(call),
                "--layout",
                str(layout),
                "--output",
                str(model_c),
                "--strict-contracts",
            ],
            cwd=PROJECT_ROOT,
        )
    else:
        ir_paths = step_build_ir(manifest, run_dir, force=True)
        model_c = step_codegen(run_dir, ir_paths, force=True)

    step_compile(model_c, run_dir, force=force_compile)
    outputs = _whisper_runtime_outputs(run_dir)
    _write_bundle_stamp(
        stamp,
        {
            "inputs": inputs,
            "outputs": {
                name: _file_identity(path) for name, path in outputs.items()
            },
        },
    )


def step_build_whisper_runtimes(
    model_input: str,
    *,
    run_dir: Path | None,
    force_download: bool,
    force_convert: bool,
    force_compile: bool,
    encoder_linear_weight_dtype: str = "preserve",
) -> tuple[Path, Path]:
    input_type, info = detect_input_type(model_input)
    if input_type == "hf_id":
        checkpoint_dir = step_download(
            info["model_id"], CACHE_DIR, force=force_download
        )
        default_name = info["model_id"].replace("/", "--")
    elif input_type == "local_dir":
        checkpoint_dir = info["path"]
        default_name = checkpoint_dir.name
    else:
        raise RuntimeError(
            "audio checkpoint must be a Hugging Face model ID or local "
            "safetensors directory"
        )
    if not _is_safetensors_checkpoint_dir(checkpoint_dir):
        raise RuntimeError(
            f"Whisper checkpoint has no safetensors weights: {checkpoint_dir}"
        )

    root = (
        run_dir.expanduser().resolve()
        if run_dir is not None
        else CACHE_DIR / f"{default_name}--audio"
    )
    encoder = root / "encoder"
    decoder = root / "decoder"
    log_step(2, "Building generated Whisper encoder and decoder")
    _build_whisper_role(
        checkpoint_dir,
        encoder,
        "encoder",
        force_convert=force_convert,
        force_compile=force_compile,
        linear_weight_dtype=encoder_linear_weight_dtype,
    )
    _build_whisper_role(
        checkpoint_dir,
        decoder,
        "decoder",
        force_convert=force_convert,
        force_compile=force_compile,
    )
    return encoder, decoder


def run_audio_pipeline(args: argparse.Namespace) -> int:
    encoder_run_dir = args.encoder_run_dir
    decoder_run_dir = args.decoder_run_dir
    model_input = getattr(args, "model", None)
    if model_input:
        encoder_run_dir, decoder_run_dir = step_build_whisper_runtimes(
            str(model_input),
            run_dir=getattr(args, "run_dir", None),
            force_download=bool(getattr(args, "force_download", False)),
            force_convert=bool(getattr(args, "force_convert", False)),
            force_compile=bool(getattr(args, "force_compile", False)),
            encoder_linear_weight_dtype=str(
                getattr(args, "encoder_linear_weight_dtype", "preserve")
            ),
        )
    elif encoder_run_dir is None or decoder_run_dir is None:
        raise RuntimeError(
            "audio requires either a Whisper checkpoint or both "
            "--encoder-run-dir and --decoder-run-dir"
        )

    module_path = SCRIPTS_DIR / "run_whisper_v8.py"
    spec = importlib.util.spec_from_file_location("ck_v8_audio_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v8 audio runtime: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [
        "run",
        "--encoder-run-dir",
        str(encoder_run_dir),
        "--decoder-run-dir",
        str(decoder_run_dir),
        "--wav",
        str(args.wav),
        "--language",
        str(args.language),
        "--task",
        str(args.task),
        "--max-tokens",
        str(int(args.max_tokens)),
    ]
    if args.timestamps:
        argv.append("--timestamps")
    if args.output is not None:
        argv.extend(["--output", str(args.output)])
    return int(module.main(argv))


def main(argv: list[str] | None = None) -> int:
    _configure_scratch_environment()
    parser = argparse.ArgumentParser(
        description="C-Kernel-Engine v8 inference runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  version/v8/scripts/cks-v8-run run hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf \\
    --context-len 1024 --force-convert --force-compile

  version/v8/scripts/cks-v8-run run hf://unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \\
    --context-len 1034 --force-convert --force-compile

  version/v8/scripts/cks-v8-run run hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf \\
    --mmproj hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf \\
    --image-path version/v8/test_assets/v8_vision_doc_card_72.png --prompt "Explain this image."

  version/v8/scripts/cks-v8-run audio hf://openai/whisper-base \\
    --wav /path/to/audio.wav
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    run_parser = subparsers.add_parser("run", help="Run model")
    run_parser.add_argument("model", help="GGUF source: hf://repo/file.gguf, local GGUF, or local runtime dir")
    run_parser.add_argument("--run", dest="run_dir", default=None, help="Optional explicit run directory")
    run_parser.add_argument("--context-len", type=int, default=None, help="Context length override")
    run_parser.add_argument(
        "--prefill-chunk-len",
        type=int,
        default=None,
        help="Maximum transient token rows per prefill call (default: circuit policy or context length)",
    )
    run_parser.add_argument("--logits-layout", choices=["auto", "last", "full"], default="auto")
    run_parser.add_argument(
        "--prefill-policy",
        choices=["auto", "batched", "sequential_decode"],
        default="auto",
        help=(
            "Generated-runtime prefill policy. auto preserves the model contract; "
            "use an explicit override only for certification or diagnostics."
        ),
    )
    run_parser.add_argument("--temperature", type=float, default=0.7)
    run_parser.add_argument("--max-tokens", type=int, default=512)
    run_parser.add_argument("--top-k", type=int, default=40)
    run_parser.add_argument("--top-p", type=float, default=1.0)
    run_parser.add_argument("--min-p", type=float, default=0.0)
    run_parser.add_argument("--repeat-penalty", type=float, default=None)
    run_parser.add_argument("--repeat-last-n", type=int, default=None)
    run_parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    run_parser.add_argument("--prompt", default=None, help="Single prompt (non-interactive if set)")
    run_parser.add_argument("--chat-template", choices=["auto", "none", "qwen", "qwen2", "qwen3", "qwen35", "qwen3vl", "gemma", "gemma3", "gemma4", "glm4", "llama"], default="auto")
    run_parser.add_argument("--no-chat-template", action="store_true")
    run_parser.add_argument("--allow-raw-prompt", action="store_true")
    run_parser.add_argument("--thinking-mode", choices=["auto", "visible", "suppressed"], default="auto")
    run_parser.add_argument("--python-tokenizer", action="store_true")
    run_parser.add_argument("--memory", action="store_true")
    run_parser.add_argument(
        "-mli",
        "--multiline-input",
        action="store_true",
        help="Accept pasted multiline turns; submit with /send or / on its own line",
    )
    run_parser.add_argument("--speculative-draft-model-dir", default=None,
                            help="Path to a compiled draft model directory for greedy speculative decoding")
    run_parser.add_argument("--speculative-draft-tokens", type=int, default=4,
                            help="Number of draft tokens to propose per speculative batch")
    run_parser.add_argument("--force-download", action="store_true")
    run_parser.add_argument("--force-convert", action="store_true")
    run_parser.add_argument("--force-compile", action="store_true")
    run_parser.add_argument("--generate-visualizer", action="store_true")
    run_parser.add_argument("--generate-only", action="store_true")
    run_parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Download/convert as needed and certify prefill/decode memory plans without code generation or execution",
    )
    run_parser.add_argument("--profile", action="store_true", help="Emit CK_PROFILE timing wrappers into the generated runtime")
    run_parser.add_argument(
        "--gemm-schedule",
        choices=("auto", "static", "dynamic"),
        default="auto",
        help="GEMM tile scheduling policy; auto dynamically balances independent tiles (default)",
    )
    run_parser.add_argument("--sweep-kernels", action="store_true", help="Sweep kernel dispatch choices and write kernel_tuning.json")
    run_parser.add_argument("--sweep-kernels-full", action="store_true", help="Use the full kernel sweep instead of the quick deployment sweep")

    run_parser.add_argument(
        "--mmproj",
        default=None,
        help="Optional encoder/mmproj GGUF for vision runs; accepts local paths or hf://repo/file.gguf",
    )
    run_parser.add_argument("--image-path", default=None, help="Optional real image path for multimodal runs")
    run_parser.add_argument("--image-min-tokens", type=int, default=None, help="Minimum merged visual tokens for smart-resized Qwen3-VL images")
    run_parser.add_argument("--image-max-tokens", type=int, default=None, help="Maximum merged visual tokens for smart-resized Qwen3-VL images")
    run_parser.add_argument("--image-mode", choices=["checker", "gradient", "gray"], default="checker")
    run_parser.add_argument("--synthetic-prefix-tokens", type=int, default=0)
    run_parser.add_argument("--bridge-runtime", choices=["prefill", "decode-staged"], default=None, help="Multimodal bridge decoder runtime; decode-staged enables incremental generation diagnostics")
    run_parser.add_argument("--bridge-generation-mode", choices=["incremental-decode", "mixed-replay"], default=None, help="Multimodal generation mode after visual/text prefill")
    run_parser.add_argument("--vision-top-k", type=int, default=8)
    run_parser.add_argument(
        "--vision-activation-pref",
        action="append",
        default=[],
        help="Optional vision encoder activation override(s) in op=dtype form, e.g. out_proj=q8",
    )

    audio_parser = subparsers.add_parser(
        "audio", help="Transcribe or translate WAV audio with generated runtimes"
    )
    audio_parser.add_argument(
        "model",
        nargs="?",
        help="Optional hf:// repository or local Whisper safetensors checkpoint",
    )
    audio_parser.add_argument("--run", dest="run_dir", type=Path)
    audio_parser.add_argument("--encoder-run-dir", type=Path)
    audio_parser.add_argument("--decoder-run-dir", type=Path)
    audio_parser.add_argument("--wav", type=Path, required=True)
    audio_parser.add_argument("--language", default="en")
    audio_parser.add_argument(
        "--task", choices=("transcribe", "translate"), default="transcribe"
    )
    audio_parser.add_argument("--max-tokens", type=int, default=128)
    audio_parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Generate monotonic paired Whisper timestamp tokens",
    )
    audio_parser.add_argument("--output", type=Path)
    audio_parser.add_argument("--force-download", action="store_true")
    audio_parser.add_argument("--force-convert", action="store_true")
    audio_parser.add_argument("--force-compile", action="store_true")
    audio_parser.add_argument(
        "--encoder-linear-weight-dtype",
        choices=("preserve", "fp16", "bf16", "fp32"),
        default="preserve",
        help=(
            "storage dtype for encoder tensors declared as linear_weight; "
            "other encoder tensors and the decoder remain unchanged"
        ),
    )

    subparsers.add_parser("list", help="List cached models")

    clean_parser = subparsers.add_parser("clean", help="Clean cached models")
    clean_parser.add_argument("model", nargs="?", help="Model cache dir to remove (or all)")

    args = parser.parse_args(argv)
    if args.command in {"run", "audio"} and os.environ.get("CK_SOURCE_INTEGRITY_CHECKED") != "1":
        integrity_script = ROOT_SCRIPTS_DIR / "check_source_integrity.py"
        result = subprocess.run(
            [sys.executable, str(integrity_script)],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            return result.returncode
    _ensure_v8_python_requirements(args.command)

    try:
        if args.command == "run":
            return run_pipeline(args)
        if args.command == "audio":
            return run_audio_pipeline(args)
        if args.command == "list":
            if CACHE_DIR.exists():
                models = sorted(path for path in CACHE_DIR.iterdir() if path.is_dir())
                if not models:
                    log("No cached models")
                else:
                    log(f"Cached models in {CACHE_DIR}:")
                    for path in models:
                        log(f"  {path.name}")
            else:
                log("No cached models")
            return 0
        if args.command == "clean":
            if args.model:
                target = CACHE_DIR / args.model.replace("/", "--")
                if target.exists():
                    shutil.rmtree(target)
                    log(f"Removed {target}")
                else:
                    log_error(f"model not found: {args.model}")
                    return 1
            elif CACHE_DIR.exists():
                shutil.rmtree(CACHE_DIR)
                log("Cleaned all cached models")
            return 0
        parser.print_help()
        return 0
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        log_error(str(exc))
        return 1
    except Exception as exc:
        log_error(f"{type(exc).__name__}: {exc}")
        if os.environ.get("CK_V8_DEBUG"):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
