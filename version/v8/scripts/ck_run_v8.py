#!/usr/bin/env python3
"""
Inference-only v8 pipeline runner.

This is the v8 copy of the high-level run surface. It intentionally covers the
inference lane only:

  download GGUF -> convert -> build IR -> codegen -> compile -> chat

and adds a v8-native multimodal route:

  decoder GGUF + mmproj GGUF + image -> run_multimodal_bridge_v8.py
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Optional


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

    wget = shutil.which("wget")
    if wget:
        cmd = [wget, "-c", url, "-O", str(tmp_dst)]
        if token:
            cmd[1:1] = ["--header", f"Authorization: Bearer {token}"]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0 and tmp_dst.exists():
            tmp_dst.replace(dst)
            return True

    curl = shutil.which("curl")
    if curl:
        cmd = [curl, "-L", "--continue-at", "-", url, "-o", str(tmp_dst)]
        if token:
            cmd[1:1] = ["-H", f"Authorization: Bearer {token}"]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0 and tmp_dst.exists():
            tmp_dst.replace(dst)
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
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
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
            if dst.stat().st_mtime >= src.stat().st_mtime and dst.stat().st_size == src.stat().st_size:
                return
        except OSError:
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    log(f"  Synced {label} -> {dst}", C_DIM)


def detect_input_type(model_input: str) -> tuple[str, dict[str, Any]]:
    text = str(model_input).strip()
    path = Path(text).expanduser()

    if text.startswith("hf://") and text.endswith(".gguf"):
        parts = text[5:].split("/")
        if len(parts) >= 3:
            return "hf_gguf", {"repo_id": f"{parts[0]}/{parts[1]}", "filename": "/".join(parts[2:])}

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
            "source_dir": str(KERNEL_MAPS_DIR),
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
            if candidate_dir.exists() and any(candidate_dir.glob("*.gguf")):
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
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        cache_dir=str(_hf_hub_cache_dir(cache_dir)),
        ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot"],
    )
    return model_dir


def step_download_gguf(repo_id: str, filename: str, cache_dir: Path, force: bool = False) -> Path:
    log_step(1, f"Downloading {filename} from {repo_id}")
    if not force:
        repo_dir = repo_id.replace("/", "--")
        filename_only = Path(filename).name
        for root in _cache_roots():
            candidate = root / repo_dir / filename_only
            if candidate.exists():
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
    try:
        downloaded = Path(
            hf_hub_download(**download_kwargs)
        )
    except Exception:
        if _direct_hf_download_gguf(repo_id, filename, gguf_path):
            return gguf_path
        raise
    if downloaded.resolve() != gguf_path.resolve():
        shutil.move(str(downloaded), str(gguf_path))
    return gguf_path


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
            return candidate
        nested = root / repo_dir / ".ck_build" / "tokenizer.json"
        if nested.exists():
            return nested
    return None


def ensure_tokenizer_files(model_id: str, work_dir: Path) -> None:
    tokenizer_path = work_dir / "tokenizer.json"
    if tokenizer_path.exists():
        return
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
    for repo_id in candidates:
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename="tokenizer.json",
                local_dir=str(work_dir),
                cache_dir=str(_hf_hub_cache_dir(CACHE_DIR)),
            )
            if tokenizer_path.exists():
                return
        except Exception:
            continue


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
    logits_layout: str | None = None,
) -> dict[str, Path]:
    log_step(3, "Building v8 IR")
    output_dir.mkdir(parents=True, exist_ok=True)
    step_regenerate_kernel_registry(force=force)

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
    if decode_ready and not force:
        log(f"  Using cached IR artifacts in {output_dir}", C_DIM)
        return outputs

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
        if logits_layout:
            cmd.extend(["--logits-layout", logits_layout])
        run_cmd(cmd, cwd=PROJECT_ROOT)

    _build_mode("prefill")
    _build_mode("decode")
    return outputs


def step_codegen(output_dir: Path, ir_paths: dict[str, Path], *, force: bool = False) -> Path:
    log_step(4, "Generating C code")
    model_c_path = output_dir / "model_v8.c"
    if (
        model_c_path.exists()
        and not force
        and model_c_path.stat().st_mtime >= ir_paths["decode_call"].stat().st_mtime
        and model_c_path.stat().st_mtime >= ir_paths["prefill_call"].stat().st_mtime
    ):
        log(f"  Using cached C code at {model_c_path}", C_DIM)
        return model_c_path

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
    run_cmd(cmd, cwd=PROJECT_ROOT)
    return model_c_path


def step_compile(model_c_path: Path, output_dir: Path, *, force: bool = False) -> Path:
    log_step(5, "Compiling shared library")
    lib_path = output_dir / "libmodel.so"
    kernel_lib = BUILD_DIR / "libckernel_engine.so"
    tokenizer_lib = BUILD_DIR / "libckernel_tokenizer.so"

    if force or not kernel_lib.exists() or not tokenizer_lib.exists():
        targets = [_path_to_make_target(kernel_lib), _path_to_make_target(tokenizer_lib)]
        run_cmd(["make", "--no-print-directory", *targets], cwd=PROJECT_ROOT)

    if lib_path.exists() and not force and lib_path.stat().st_mtime >= model_c_path.stat().st_mtime:
        _sync_runtime_lib(kernel_lib, output_dir / "libckernel_engine.so", "libckernel_engine.so")
        _sync_runtime_lib(tokenizer_lib, output_dir / "libckernel_tokenizer.so", "libckernel_tokenizer.so")
        symlink_path = output_dir / "ck-kernel-inference.so"
        try:
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to("libmodel.so")
        except Exception:
            pass
        return lib_path

    include_dir = PROJECT_ROOT / "include"
    v8_src = V8_ROOT / "src"
    compiler = "gcc"
    requested_compiler = (os.environ.get("CK_V8_COMPILER", "") or os.environ.get("CK_V7_COMPILER", "")).strip()
    if requested_compiler:
        if not shutil.which(requested_compiler):
            log_error(f"Requested CK_V8_COMPILER not found in PATH: {requested_compiler}")
            sys.exit(1)
        compiler = requested_compiler
    elif shutil.which("icx"):
        compiler = "icx"
    omp_flag = "-qopenmp" if compiler == "icx" else "-fopenmp"
    machine = _host_machine()
    use_openmp = _compiler_supports_openmp(compiler, omp_flag)
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
        str(lib_path),
        str(model_c_path),
        str(v8_src / "ckernel_model_load_v8.c"),
        str(v8_src / "ck_parallel_decode_v8.c"),
        str(v8_src / "ck_parallel_prefill_v8.c"),
        f"-L{BUILD_DIR}",
        f"-L{output_dir}",
        "-lckernel_tokenizer",
        "-lckernel_engine",
        "-lm",
        "-lpthread",
        "-Wl,-rpath,$ORIGIN",
        f"-Wl,-rpath,{BUILD_DIR}",
    ]
    if _is_x86_machine(machine):
        cmd.insert(3, "-mcmodel=large")
    if use_openmp:
        cmd.insert(8 if _is_x86_machine(machine) else 7, omp_flag)
    extra_cflags = (os.environ.get("CK_V8_EXTRA_CFLAGS", "") or os.environ.get("CK_V7_EXTRA_CFLAGS", "")).strip()
    extra_ldflags = (os.environ.get("CK_V8_EXTRA_LDFLAGS", "") or os.environ.get("CK_V7_EXTRA_LDFLAGS", "")).strip()
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
    _sync_runtime_lib(kernel_lib, output_dir / "libckernel_engine.so", "libckernel_engine.so")
    _sync_runtime_lib(tokenizer_lib, output_dir / "libckernel_tokenizer.so", "libckernel_tokenizer.so")
    symlink_path = output_dir / "ck-kernel-inference.so"
    try:
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to("libmodel.so")
    except Exception:
        pass
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
        return CACHE_DIR / info["model_id"].replace("/", "--")
    if input_type == "gguf":
        return CACHE_DIR / info["path"].stem
    if input_type == "local_dir":
        return info["path"] / ".ck_build_v8"
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
        if args.synthetic_prefix_tokens > 0:
            cmd.extend(["--synthetic-prefix-tokens", str(int(args.synthetic_prefix_tokens))])
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
        logits_layout=args.logits_layout,
    )
    model_c_path = step_codegen(work_dir, ir_paths, force=args.force_compile)
    lib_path = step_compile(model_c_path, work_dir, force=args.force_compile)

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


def main(argv: list[str] | None = None) -> int:
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
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    run_parser = subparsers.add_parser("run", help="Run model")
    run_parser.add_argument("model", help="GGUF source: hf://repo/file.gguf, local GGUF, or local runtime dir")
    run_parser.add_argument("--run", dest="run_dir", default=None, help="Optional explicit run directory")
    run_parser.add_argument("--context-len", type=int, default=None, help="Context length override")
    run_parser.add_argument("--logits-layout", choices=["auto", "last", "full"], default="auto")
    run_parser.add_argument("--temperature", type=float, default=0.7)
    run_parser.add_argument("--max-tokens", type=int, default=512)
    run_parser.add_argument("--top-k", type=int, default=40)
    run_parser.add_argument("--top-p", type=float, default=1.0)
    run_parser.add_argument("--min-p", type=float, default=0.0)
    run_parser.add_argument("--repeat-penalty", type=float, default=1.0)
    run_parser.add_argument("--repeat-last-n", type=int, default=64)
    run_parser.add_argument("--prompt", default=None, help="Single prompt (non-interactive if set)")
    run_parser.add_argument("--chat-template", choices=["auto", "none", "qwen", "qwen2", "qwen3", "qwen35", "qwen3vl", "gemma", "gemma3", "llama"], default="auto")
    run_parser.add_argument("--no-chat-template", action="store_true")
    run_parser.add_argument("--allow-raw-prompt", action="store_true")
    run_parser.add_argument("--thinking-mode", choices=["auto", "visible", "suppressed"], default="auto")
    run_parser.add_argument("--python-tokenizer", action="store_true")
    run_parser.add_argument("--memory", action="store_true")
    run_parser.add_argument("--force-download", action="store_true")
    run_parser.add_argument("--force-convert", action="store_true")
    run_parser.add_argument("--force-compile", action="store_true")
    run_parser.add_argument("--generate-visualizer", action="store_true")
    run_parser.add_argument("--generate-only", action="store_true")

    run_parser.add_argument(
        "--mmproj",
        default=None,
        help="Optional encoder/mmproj GGUF for vision runs; accepts local paths or hf://repo/file.gguf",
    )
    run_parser.add_argument("--image-path", default=None, help="Optional real image path for multimodal runs")
    run_parser.add_argument("--image-mode", choices=["checker", "gradient", "gray"], default="checker")
    run_parser.add_argument("--synthetic-prefix-tokens", type=int, default=0)
    run_parser.add_argument("--vision-top-k", type=int, default=8)
    run_parser.add_argument(
        "--vision-activation-pref",
        action="append",
        default=[],
        help="Optional vision encoder activation override(s) in op=dtype form, e.g. out_proj=q8",
    )

    subparsers.add_parser("list", help="List cached models")

    clean_parser = subparsers.add_parser("clean", help="Clean cached models")
    clean_parser.add_argument("model", nargs="?", help="Model cache dir to remove (or all)")

    args = parser.parse_args(argv)
    _ensure_v8_python_requirements(args.command)

    try:
        if args.command == "run":
            return run_pipeline(args)
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
