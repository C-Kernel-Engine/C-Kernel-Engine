#!/usr/bin/env python3
"""
IR Visualizer Launcher for C-Kernel-Engine v8

Usage:
    python3 version/v8/tools/open_ir_visualizer_v8.py              # Open visualizer
    python3 version/v8/tools/open_ir_visualizer_v8.py --list       # List available models
    python3 version/v8/tools/open_ir_visualizer_v8.py <model>      # Generate and open report
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate <model>  # Generate full report (profile+probes)
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate <model> --interactive
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate <model> --html-only  # Generate HTML only
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate <model> --with-profile
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate <model> --with-probes
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate <model> --with-probes --weight-dtype float32
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate <model> --with-probes --perf-runtime cli
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate <model> --no-vtune
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate --run ./runs/exp1_baseline --with-probes --advisor
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate <model> --with-probes --run-model hf://... --chat-template none
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate --run ./runs/exp1_baseline
"""
import os
import sys
import json
import base64
import re
import html as html_mod
import shutil
import platform
import webbrowser
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Path construction:
# Script is at: version/v8/tools/open_ir_visualizer_v8.py
SCRIPT_DIR = Path(__file__).parent              # .../version/v8/tools
V8_ROOT = SCRIPT_DIR.parent                    # .../version/v8
PROJECT_ROOT = V8_ROOT.parent.parent           # .../Workspace/C-Kernel-Engine

sys.path.insert(0, str(V8_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_PATH = Path(os.environ.get("CK_CACHE_DIR", str(Path.home() / ".cache" / "ck-engine-v8" / "models"))).expanduser()
VISUALIZER = SCRIPT_DIR / "ir_visualizer.html"
CK_RUN_SCRIPT = V8_ROOT / "scripts" / "ck_run_v8.py"
MEMORY_SIGNOFF_SCRIPT = V8_ROOT / "scripts" / "memory_signoff_v8.py"
V8_REPORT_PATH = Path(os.environ.get("CK_V8_REPORT_DIR", str(CACHE_PATH / "reports"))).expanduser()
V8_REPORT_PATH_LEGACY = V8_ROOT / "reports"


def _read_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.exists():
        return {}
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"')
    return data


def _profile_install_hints() -> list[str]:
    if not sys.platform.startswith("linux"):
        return [
            "# Full profiling workflow is Linux-first.",
            "# For local smoke tests on macOS/WSL, use the runtime and artifact commands first.",
            "make v8-doctor",
        ]

    os_release = _read_os_release()
    distro_id = (os_release.get("ID") or "").lower()
    distro_like = (os_release.get("ID_LIKE") or "").lower()

    if distro_id in {"ubuntu", "debian", "linuxmint", "pop"} or any(x in distro_like for x in ("debian", "ubuntu")):
        return [
            "sudo apt-get update",
            "sudo apt-get install -y linux-tools-common linux-tools-generic linux-tools-$(uname -r) perl git valgrind",
            "git clone https://github.com/brendangregg/FlameGraph.git",
            "chmod +x FlameGraph/stackcollapse-perf.pl FlameGraph/flamegraph.pl",
            'printf "kernel.perf_event_paranoid=1\\nkernel.kptr_restrict=0\\n" | sudo tee /etc/sysctl.d/99-perf.conf',
            "sudo sysctl -p /etc/sysctl.d/99-perf.conf",
            "# Intel hosts: install Intel oneAPI Base Toolkit for icx/vtune/advisor",
            "# VTune on Linux may require:",
            "sudo sysctl -w kernel.yama.ptrace_scope=0",
        ]
    if distro_id in {"arch", "manjaro", "endeavouros"} or "arch" in distro_like:
        return [
            "sudo pacman -S base-devel git perf valgrind flamegraph",
            "# Intel hosts: sudo pacman -S intel-oneapi-basekit",
            "mkdir -p FlameGraph && ln -sf /usr/bin/flamegraph FlameGraph/flamegraph.pl && ln -sf /usr/bin/stackcollapse-perf FlameGraph/stackcollapse-perf.pl",
            "sudo sysctl -w kernel.perf_event_paranoid=1 kernel.kptr_restrict=0",
            "# VTune on Linux may require:",
            "sudo sysctl -w kernel.yama.ptrace_scope=0",
        ]
    if distro_id in {"fedora", "rhel", "centos", "rocky", "almalinux"} or any(x in distro_like for x in ("fedora", "rhel")):
        return [
            "sudo dnf install -y perf perl git valgrind",
            "git clone https://github.com/brendangregg/FlameGraph.git",
            "chmod +x FlameGraph/stackcollapse-perf.pl FlameGraph/flamegraph.pl",
            "sudo sysctl -w kernel.perf_event_paranoid=1 kernel.kptr_restrict=0",
            "# Intel hosts: install Intel oneAPI Base Toolkit for icx/vtune/advisor",
            "# VTune on Linux may require:",
            "sudo sysctl -w kernel.yama.ptrace_scope=0",
        ]
    if distro_id in {"opensuse", "opensuse-tumbleweed", "opensuse-leap"} or "suse" in distro_like:
        return [
            "sudo zypper install -y perf perl git valgrind",
            "git clone https://github.com/brendangregg/FlameGraph.git",
            "chmod +x FlameGraph/stackcollapse-perf.pl FlameGraph/flamegraph.pl",
            "sudo sysctl -w kernel.perf_event_paranoid=1 kernel.kptr_restrict=0",
            "# Intel hosts: install Intel oneAPI Base Toolkit for icx/vtune/advisor",
            "# VTune on Linux may require:",
            "sudo sysctl -w kernel.yama.ptrace_scope=0",
        ]
    return [
        "make v8-doctor",
        "# Install perf, valgrind, and FlameGraph for your Linux distribution, then retry.",
        "# Intel hosts: install Intel oneAPI Base Toolkit for icx/vtune/advisor",
    ]


def _format_mem_gib(kb: int | None) -> str | None:
    if not isinstance(kb, int) or kb <= 0:
        return None
    gib = kb / (1024.0 * 1024.0)
    return f"{gib:.1f} GiB"


def collect_host_context() -> dict[str, object]:
    context: dict[str, object] = {
        "host_platform": sys.platform,
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count() or 0,
    }

    os_release = _read_os_release()
    pretty_name = os_release.get("PRETTY_NAME") or os_release.get("NAME")
    if pretty_name:
        context["os_pretty_name"] = pretty_name
    distro_id = os_release.get("ID")
    if distro_id:
        context["os_id"] = distro_id

    try:
        cpuinfo_text = Path("/proc/cpuinfo").read_text(errors="ignore")
        processors: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for raw in cpuinfo_text.splitlines():
            line = raw.strip()
            if not line:
                if current:
                    processors.append(current)
                    current = {}
                continue
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            current[k.strip().lower()] = v.strip()
        if current:
            processors.append(current)

        model_names: set[str] = set()
        physical_ids: set[str] = set()
        core_ids_per_socket: dict[str, set[str]] = {}
        flags: set[str] = set()

        for proc in processors:
            if "model name" in proc:
                model_names.add(proc["model name"])
            if "flags" in proc:
                flags.update(proc["flags"].split())
            phys_id = proc.get("physical id", "0")
            core_id = proc.get("core id", proc.get("processor", "0"))
            physical_ids.add(phys_id)
            core_ids_per_socket.setdefault(phys_id, set()).add(core_id)

        if model_names:
            context["cpu_model"] = next(iter(model_names))
        if processors:
            context["logical_cpus"] = len(processors)
        physical_cores = sum(len(v) for v in core_ids_per_socket.values())
        if physical_cores > 0:
            context["physical_cores"] = physical_cores
        if physical_ids:
            context["sockets"] = len(physical_ids)

        isa_tags: list[str] = []
        if "avx512f" in flags:
            isa_tags.append("AVX-512")
        elif "avx2" in flags:
            isa_tags.append("AVX2")
        elif "avx" in flags:
            isa_tags.append("AVX")
        if "fma" in flags:
            isa_tags.append("FMA")
        if "avx512_vnni" in flags or "avx512vnni" in flags:
            isa_tags.append("VNNI")
        if "amx_bf16" in flags:
            isa_tags.append("AMX-BF16")
        if isa_tags:
            context["isa_tags"] = isa_tags
    except Exception:
        pass

    if not context.get("cpu_model"):
        try:
            out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
            kv: dict[str, str] = {}
            for raw in out.splitlines():
                if ":" not in raw:
                    continue
                k, v = raw.split(":", 1)
                kv[k.strip().lower()] = v.strip()
            model = kv.get("model name")
            if model:
                context["cpu_model"] = model
            sockets = kv.get("socket(s)")
            cores_per_socket = kv.get("core(s) per socket")
            if sockets and cores_per_socket:
                try:
                    context["physical_cores"] = int(sockets) * int(cores_per_socket)
                    context["sockets"] = int(sockets)
                except ValueError:
                    pass
        except Exception:
            pass

    try:
        for line in Path("/proc/meminfo").read_text(errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    mem_total_kb = int(parts[1])
                    context["mem_total_kb"] = mem_total_kb
                    human = _format_mem_gib(mem_total_kb)
                    if human:
                        context["mem_total_human"] = human
                break
    except Exception:
        pass

    try:
        nodes = [p for p in Path("/sys/devices/system/node").glob("node[0-9]*") if p.is_dir()]
        if nodes:
            context["numa_nodes"] = len(nodes)
    except Exception:
        pass

    try:
        thp_path = Path("/sys/kernel/mm/transparent_hugepage/enabled")
        if thp_path.exists():
            content = thp_path.read_text().strip()
            m = re.search(r"\[(\w+)\]", content)
            context["thp_status"] = m.group(1) if m else "unknown"
    except Exception:
        pass

    return context


def run_cmd(cmd: list[str], cwd: Path, extra_env: dict | None = None):
    print(f"[run] {' '.join(cmd)}")
    env = None
    if extra_env:
        env = dict(os.environ)
        env.update(extra_env)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def try_run_cmd(step: str, cmd: list[str], cwd: Path, extra_env: dict | None = None) -> bool:
    try:
        run_cmd(cmd, cwd=cwd, extra_env=extra_env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: {step} failed (continuing): {e}")
        return False


def collect_profile_tool_status() -> dict[str, object]:
    flamegraph_dir = PROJECT_ROOT / "FlameGraph"
    return {
        "host_platform": sys.platform,
        "install_hints": _profile_install_hints(),
        "perf": shutil.which("perf") is not None,
        "valgrind": shutil.which("valgrind") is not None,
        "cg_annotate": shutil.which("cg_annotate") is not None,
        "vtune": shutil.which("vtune") is not None,
        "advisor": shutil.which("advisor") is not None,
        "xdg_open": shutil.which("xdg-open") is not None,
        "flamegraph": (
            (flamegraph_dir / "stackcollapse-perf.pl").exists()
            and (flamegraph_dir / "flamegraph.pl").exists()
        ),
        "flamegraph_dir": str(flamegraph_dir),
    }


def resolve_model_target(model_arg: str) -> tuple[Path, Path]:
    """Resolve a model argument to ck_build/model dir and model root directory."""
    candidate = Path(model_arg)
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"Model path is not a directory: {candidate}")
        if candidate.name in {"ck_build", ".ck_build"}:
            ck_build = candidate
            model_root = candidate.parent
        else:
            if (candidate / "ck_build").exists():
                ck_build = candidate / "ck_build"
            elif (candidate / ".ck_build").exists():
                ck_build = candidate / ".ck_build"
            else:
                ck_build = candidate
            model_root = candidate
        return ck_build, model_root

    ck_build = CACHE_PATH / model_arg / "ck_build"
    if ck_build.exists():
        return ck_build, CACHE_PATH / model_arg

    dot_ck_build = CACHE_PATH / model_arg / ".ck_build"
    if dot_ck_build.exists():
        return dot_ck_build, CACHE_PATH / model_arg

    model_dir = CACHE_PATH / model_arg
    if model_dir.exists():
        return model_dir, model_dir

    raise ValueError(f"Model not found: {model_arg}")


def resolve_run_target(run_arg: Path) -> tuple[Path, Path]:
    """Resolve a run directory to (preferred_build_dir, run_root)."""
    run_root = run_arg.expanduser().resolve()
    if not run_root.exists():
        raise ValueError(f"Run directory not found: {run_root}")
    if not run_root.is_dir():
        raise ValueError(f"Run path is not a directory: {run_root}")

    if (run_root / "ck_build").exists():
        return run_root / "ck_build", run_root
    if (run_root / ".ck_build").exists():
        return run_root / ".ck_build", run_root
    return run_root, run_root


def has_local_runnable_source(path: Path) -> bool:
    """Whether path can be used as ck_run_v8 local input source."""
    if not path.exists() or not path.is_dir():
        return False
    if any(path.glob("*.gguf")):
        return True
    if any(path.glob("*.safetensors")):
        return True
    if any(path.glob("*.safetensors.index.json")):
        return True
    if (path / "model.safetensors.index.json").exists():
        return True
    return False


def has_local_compiled_runtime(path: Path) -> bool:
    """Whether path already has compiled runtime outputs usable by ck-cli."""
    if not (
        path.exists()
        and path.is_dir()
        and (path / "libmodel.so").exists()
        and (path / "weights.bump").exists()
    ):
        return False
    # Reject runtimes with unresolved shared-lib deps (e.g., libimf.so missing).
    lib_path = path / "libmodel.so"
    try:
        result = subprocess.run(
            ["ldd", str(lib_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        text = f"{result.stdout}\n{result.stderr}"
        if "=> not found" in text:
            return False
    except Exception:
        # If ldd is unavailable, keep best-effort behavior.
        pass
    return True


def resolve_local_runnable_input(path: Path) -> str:
    """
    Prefer concrete local file inputs where possible.
    - If directory contains a GGUF, return that file path (ck_run handles gguf file directly).
    - Otherwise return directory path.
    """
    if path.exists() and path.is_dir():
        ggufs = sorted(path.glob("*.gguf"))
        if ggufs:
            return str(ggufs[0])
    return str(path)


def infer_run_model_input(model_root: Path, weight_dtype: str | None = None) -> str | None:
    """
    Best-effort mapping from cache model folder names to hf:// model inputs.
    Keeps launcher ergonomic for common v8 model aliases.
    For training-oriented dtypes (float32/bf16), prefer source checkpoints.
    """
    name = model_root.name.lower()
    wants_source_ckpt = weight_dtype in {"float32", "bf16"}
    if "gemma-3-270m-it" in name:
        return "hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf"
    if "qwen2-0.5b-instruct" in name or "qwen2-0_5b-instruct" in name:
        if wants_source_ckpt:
            return "Qwen/Qwen2-0.5B-Instruct"
        return "hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf"
    if "qwen3-0.6b" in name:
        if wants_source_ckpt:
            return "Qwen/Qwen3-0.6B"
        return "hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
    return None


def infer_chat_template(run_model_input: str, model_root: Path) -> str:
    probe = f"{run_model_input} {model_root.name}".lower()
    # Gemma GGUF chat templates are often not compatible with our runtime path;
    # prefer raw prompt mode unless the operator overrides explicitly.
    if "gemma" in probe:
        return "none"
    return "auto"


def prompt_with_default(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    return raw


def maybe_interactive_configure(args: argparse.Namespace, model_root: Path) -> None:
    if not getattr(args, "interactive", False):
        return
    if not sys.stdin.isatty():
        print("Warning: --interactive requested, but stdin is not a TTY. Skipping prompts.")
        return

    print("\nInteractive report options (press Enter to keep default):")

    suggested_model = args.run_model or ""
    if not suggested_model:
        if has_local_runnable_source(model_root):
            suggested_model = resolve_local_runnable_input(model_root)
        else:
            suggested_model = infer_run_model_input(model_root, args.weight_dtype) or ""
    if suggested_model:
        args.run_model = prompt_with_default("Runtime source (--run-model)", suggested_model)
    else:
        manual = input("Runtime source (--run-model) [leave blank for auto-infer]: ").strip()
        if manual:
            args.run_model = manual

    weight_default = args.weight_dtype if args.weight_dtype else "auto"
    allowed_weight = {"auto", "float32", "bf16", "q4_0", "q4_1", "q4_k", "q4_k_m", "q5_0", "q5_1", "q6_k", "q8_0"}
    while True:
        chosen = prompt_with_default("Weight dtype", weight_default)
        if chosen in allowed_weight:
            args.weight_dtype = None if chosen == "auto" else chosen
            break
        print(f"Invalid weight dtype: {chosen}")

    chat_default = args.chat_template if args.chat_template else "auto"
    allowed_chat = {"auto", "none", "qwen", "gemma"}
    while True:
        chosen = prompt_with_default("Chat template", chat_default)
        if chosen in allowed_chat:
            args.chat_template = chosen
            break
        print(f"Invalid chat template: {chosen}")

    perf_default = args.perf_runtime if args.perf_runtime else "cli"
    allowed_perf = {"cli", "python"}
    while True:
        chosen = prompt_with_default("Perf runtime", perf_default)
        if chosen in allowed_perf:
            args.perf_runtime = chosen
            break
        print(f"Invalid perf runtime: {chosen}")

    print(
        "Selected: "
        f"run-model={args.run_model or '<auto>'}, "
        f"weight-dtype={args.weight_dtype or 'auto'}, "
        f"chat-template={args.chat_template}, "
        f"perf-runtime={args.perf_runtime}"
    )


def detect_model_dir_from_input(model_input: str) -> Path | None:
    try:
        from ck_run_v8 import CACHE_DIR, detect_input_type  # type: ignore
    except Exception:
        return None

    input_type, info = detect_input_type(model_input)
    if input_type == "hf_gguf":
        return CACHE_DIR / info["repo_id"].replace("/", "--")
    if input_type == "hf_id":
        return CACHE_DIR / info["model_id"].replace("/", "--")
    if input_type == "gguf":
        return CACHE_DIR / info["path"].stem
    if input_type == "local_dir":
        local = Path(info["path"]).resolve()
        return local.parent if local.name == ".ck_build" else local
    if input_type == "local_config":
        cfg_parent = Path(info["path"]).resolve().parent
        return cfg_parent.parent if cfg_parent.name == ".ck_build" else cfg_parent
    return None


def maybe_localize_hf_gguf(model_input: str) -> str:
    """
    If model_input is hf://repo/file.gguf and a local .ck_cache copy exists,
    use the local file path to avoid network dependence.
    """
    if not model_input.startswith("hf://"):
        return model_input
    try:
        body = model_input[len("hf://"):]
        repo_id, filename = body.rsplit("/", 1)
        if not filename.lower().endswith(".gguf"):
            return model_input
        local = PROJECT_ROOT / ".ck_cache" / repo_id.replace("/", "--") / filename
        if local.exists():
            print(f"Using local cached GGUF: {local}")
            return str(local)
    except Exception:
        return model_input
    return model_input


def copy_artifacts_if_needed(src_model_dir: Path, dst_model_dir: Path) -> None:
    if src_model_dir.resolve() == dst_model_dir.resolve():
        return
    artifact_names = [
        "profile_summary.json",
        "perf_stat_summary.json",
        "flamegraph_manifest.json",
        "cachegrind_summary.json",
        "asan_summary.json",
        "vtune_summary.json",
        "advisor_summary.json",
        "memory_signoff.json",
        "memory_verification_latest.json",
        "perf_gate_report.json",
        "ir1_train_forward.json",
        "ir2_train_backward.json",
        "ir2_train_summary.json",
        "ir_train_invariants.json",
        "contract_report_latest.json",
        "parity_1token_latest.json",
        "qk_norm_backward_parity_latest.json",
        "fd_gradients_latest.json",
        "parity_autopsy_latest.json",
        "autopsy_report.json",
        "autopsy_report_latest.json",
        "detailed_parity_analysis.json",
        "detailed_parity_analysis_latest.json",
        "detailed_parity_analysis_prefill.json",
        "detailed_parity_analysis_prefill_latest.json",
        "detailed_parity_analysis_decode.json",
        "detailed_parity_analysis_decode_latest.json",
        "train_parity_epochs_3_latest.json",
        "train_parity_epochs_5_latest.json",
        "replay_determinism_latest.json",
        "backprop_stitch_runtime_latest.json",
        "regimen_backend_xray.json",
        "training_canary_summary.json",
        "regression_ledger.json",
    ]
    copied = 0
    dst_model_dir.mkdir(parents=True, exist_ok=True)
    src_roots = [src_model_dir, src_model_dir / ".ck_build"]
    for name in artifact_names:
        src = None
        for root in src_roots:
            candidate = root / name
            if candidate.exists():
                src = candidate
                break
        if src is None:
            continue
        shutil.copy2(src, dst_model_dir / name)
        copied += 1
    if copied:
        print(f"Copied {copied} artifact(s) from {src_model_dir} -> {dst_model_dir}")


def has_model_artifact(model_root: Path, ck_build: Path, name: str) -> bool:
    return (
        (model_root / name).exists()
        or (model_root / ".ck_build" / name).exists()
        or (ck_build / name).exists()
    )


def validate_artifact_set(
    model_root: Path,
    ck_build: Path,
    expect_perf: bool,
    expect_vtune: bool,
    expect_advisor: bool = False,
    expect_cachegrind: bool = False,
    expect_asan: bool = False,
) -> list[str]:
    missing: list[str] = []
    base_required = ["memory_signoff.json", "profile_summary.json"]
    for name in base_required:
        if not has_model_artifact(model_root, ck_build, name):
            missing.append(name)
    if expect_perf:
        for name in ["perf_stat_summary.json", "flamegraph_manifest.json", "perf_gate_report.json"]:
            if not has_model_artifact(model_root, ck_build, name):
                missing.append(name)
    if expect_vtune and not has_model_artifact(model_root, ck_build, "vtune_summary.json"):
        missing.append("vtune_summary.json")
    if expect_advisor and not has_model_artifact(model_root, ck_build, "advisor_summary.json"):
        missing.append("advisor_summary.json")
    if expect_cachegrind and not has_model_artifact(model_root, ck_build, "cachegrind_summary.json"):
        missing.append("cachegrind_summary.json")
    if expect_asan and not has_model_artifact(model_root, ck_build, "asan_summary.json"):
        missing.append("asan_summary.json")
    return missing


def has_train_runtime_artifacts(run_dir: Path) -> bool:
    if not run_dir.exists() or not run_dir.is_dir():
        return False
    has_lib = (run_dir / "libtrain.so").exists() or (run_dir / ".ck_build" / "libtrain.so").exists()
    has_weights = (run_dir / "weights.bump").exists()
    has_manifest = (run_dir / "weights_manifest.json").exists()
    has_summary = (
        (run_dir / "generated_train_runtime_summary.json").exists()
        or (run_dir / "generated_train_runtime_summary_v7.json").exists()
    )
    return bool(has_lib and has_weights and has_manifest and has_summary)


def has_inference_runtime_artifacts(run_dir: Path) -> bool:
    if not run_dir.exists() or not run_dir.is_dir():
        return False
    has_lib = (
        (run_dir / "libmodel.so").exists()
        or (run_dir / ".ck_build" / "libmodel.so").exists()
        or (run_dir / "ck_build" / "libmodel.so").exists()
        or (run_dir / ".ck_build" / "ck-kernel-inference.so").exists()
    )
    has_weights = (run_dir / "weights.bump").exists()
    has_manifest = (run_dir / "weights_manifest.json").exists()
    return bool(has_lib and has_weights and has_manifest)


def infer_train_vocab_size(run_dir: Path, default_vocab: int = 1024) -> int:
    candidates = [
        run_dir / "train_init_config.json",
        run_dir / "config.json",
    ]
    keys = ("train_vocab", "vocab_size", "vocab", "n_vocab")
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        # Direct keys.
        for key in keys:
            value = payload.get(key)
            if isinstance(value, int) and value > 0:
                return value
        # Nested config dictionary.
        cfg = payload.get("config")
        if isinstance(cfg, dict):
            for key in keys:
                value = cfg.get(key)
                if isinstance(value, int) and value > 0:
                    return value
    return default_vocab


def ensure_train_token_file(run_dir: Path, token_count: int = 8192) -> Path:
    vocab = max(2, infer_train_vocab_size(run_dir))
    token_path = run_dir / ".ck_profile_tokens_viz.txt"
    if token_path.exists():
        return token_path
    seq = " ".join(str(i % vocab) for i in range(token_count))
    token_path.write_text(seq)
    return token_path


def _resolve_asset_path(raw_path: str, ck_build_path: Path, model_root: Path) -> Path | None:
    if not raw_path:
        return None
    p = Path(raw_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(model_root / p)
        candidates.append(ck_build_path / p)
        candidates.append(PROJECT_ROOT / p)
    for c in candidates:
        if c.exists():
            return c
    return None


def _encode_image_data_uri(path: Path) -> str | None:
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix)
    if mime is None:
        return None
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _resolve_multimodal_bridge_dirs(
    run_dir: Path | None,
    model_root: Path,
) -> dict[str, Path | None]:
    for base in (run_dir, model_root):
        if base is None:
            continue
        bridge_root = base / "multimodal_bridge"
        if not bridge_root.exists() or not bridge_root.is_dir():
            continue
        decoder_root = bridge_root / "decoder"
        encoder_root = bridge_root / "encoder"
        return {
            "root": bridge_root,
            "decoder": decoder_root if decoder_root.exists() and decoder_root.is_dir() else None,
            "encoder": encoder_root if encoder_root.exists() and encoder_root.is_dir() else None,
        }
    return {"root": None, "decoder": None, "encoder": None}


def _enrich_multimodal_bridge_payload(
    payload: dict[str, Any],
    *,
    ck_build_path: Path,
    model_root: Path,
) -> dict[str, Any]:
    out = dict(payload)
    encoder_report = out.get("encoder_report")
    if isinstance(encoder_report, dict):
        encoder_copy = dict(encoder_report)
        raw_image_path = str(encoder_copy.get("image_path") or "").strip()
        if raw_image_path:
            resolved = _resolve_asset_path(raw_image_path, ck_build_path, model_root)
            if resolved is not None:
                encoder_copy["image_path_resolved"] = str(resolved)
                data_uri = _encode_image_data_uri(resolved)
                if data_uri:
                    encoder_copy["image_data_uri"] = data_uri
        out["encoder_report"] = encoder_copy
    out["bridge_mode"] = "encoder_decoder" if isinstance(out.get("encoder_runtime"), dict) else "decoder_only"
    return out


def _trim_memory_signoff_payload(payload: dict, max_items: int = 120) -> dict:
    """Trim very large warning/error arrays to keep standalone report size manageable."""
    out = dict(payload)
    # Preserve original totals (if present) for drill-down/debug.
    if "warning_count" in out:
        out["warning_count_raw"] = out.get("warning_count", 0)
    if "error_count" in out:
        out["error_count_raw"] = out.get("error_count", 0)
    checks = out.get("checks")
    if not isinstance(checks, dict):
        return out
    trimmed_checks = {}
    total_errors = 0
    total_warnings = 0
    for key, check in checks.items():
        if not isinstance(check, dict):
            trimmed_checks[key] = check
            continue
        c2 = dict(check)
        for list_key in ("warnings", "errors", "info"):
            arr = c2.get(list_key)
            if isinstance(arr, list) and len(arr) > max_items:
                c2[f"{list_key}_truncated"] = len(arr) - max_items
                c2[list_key] = arr[:max_items]
        errs = c2.get("errors")
        warns = c2.get("warnings")
        if isinstance(errs, list):
            total_errors += len(errs)
        if isinstance(warns, list):
            total_warnings += len(warns)
        trimmed_checks[key] = c2
    out["checks"] = trimmed_checks
    # Keep headline counts consistent with what is rendered in each row.
    out["error_count"] = total_errors
    out["warning_count"] = total_warnings
    return out


def _summarize_call_ir_payload(payload: dict[str, Any], max_samples: int = 6) -> dict[str, Any]:
    """Summarize call-IR errors/warnings so operator snapshot can show hard failures."""
    top_errors = payload.get("errors")
    top_warnings = payload.get("warnings")
    ops = payload.get("operations")

    top_error_count = len(top_errors) if isinstance(top_errors, list) else 0
    top_warning_count = len(top_warnings) if isinstance(top_warnings, list) else 0
    op_error_count = 0
    op_warning_count = 0
    sample_errors: list[dict[str, Any]] = []
    sample_warnings: list[dict[str, Any]] = []

    if isinstance(ops, list):
        for i, op in enumerate(ops):
            if not isinstance(op, dict):
                continue
            idx = op.get("idx", i)
            op_name = op.get("op") or op.get("type") or ""
            fn_name = op.get("function") or op.get("kernel") or ""
            errs = op.get("errors")
            warns = op.get("warnings")
            if isinstance(errs, list) and errs:
                op_error_count += len(errs)
                if len(sample_errors) < max_samples:
                    sample_errors.append(
                        {
                            "idx": idx,
                            "op": str(op_name),
                            "function": str(fn_name),
                            "message": str(errs[0]),
                        }
                    )
            if isinstance(warns, list) and warns:
                op_warning_count += len(warns)
                if len(sample_warnings) < max_samples:
                    sample_warnings.append(
                        {
                            "idx": idx,
                            "op": str(op_name),
                            "function": str(fn_name),
                            "message": str(warns[0]),
                        }
                    )

    return {
        "top_error_count": top_error_count,
        "top_warning_count": top_warning_count,
        "op_error_count": op_error_count,
        "op_warning_count": op_warning_count,
        "error_count": top_error_count + op_error_count,
        "warning_count": top_warning_count + op_warning_count,
        "sample_errors": sample_errors,
        "sample_warnings": sample_warnings,
    }


def _collect_call_ir_health(files: dict[str, Any]) -> dict[str, Any] | None:
    modes: dict[str, Any] = {}
    for mode, key in (("decode", "lowered_decode_call"), ("prefill", "lowered_prefill_call")):
        payload = files.get(key)
        if not isinstance(payload, dict):
            continue
        modes[mode] = _summarize_call_ir_payload(payload)

    if not modes:
        return None

    totals = {"error_count": 0, "warning_count": 0}
    for summary in modes.values():
        totals["error_count"] += int(summary.get("error_count") or 0)
        totals["warning_count"] += int(summary.get("warning_count") or 0)

    return {
        "schema": "ck.call_ir_health.v1",
        "modes": modes,
        "totals": totals,
    }


def _piece_display(piece: str, limit: int = 64) -> str:
    disp = piece.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if len(disp) > limit:
        disp = disp[:limit] + "..."
    return disp


def _bytes_hex_preview(text: str, limit: int = 12) -> str:
    raw = text.encode("utf-8", errors="replace")
    chunk = raw[:limit]
    hx = " ".join(f"{b:02X}" for b in chunk)
    if len(raw) > limit:
        hx += " ..."
    return hx


def _parse_merge_row(row) -> tuple[str, str]:
    if isinstance(row, list) and len(row) >= 2:
        return str(row[0]), str(row[1])
    if isinstance(row, str):
        parts = row.split(" ", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return "", ""


def _build_tokenizer_preview(
    files: dict,
    ck_build_path: Path,
    model_root: Path,
    run_dir: Path | None = None,
) -> dict | None:
    pipeline = files.get("training_pipeline") if isinstance(files.get("training_pipeline"), dict) else {}
    data_lab = pipeline.get("data_lab") if isinstance(pipeline, dict) and isinstance(pipeline.get("data_lab"), dict) else {}
    tokenizer_lineage = pipeline.get("tokenizer_lineage") if isinstance(pipeline, dict) and isinstance(pipeline.get("tokenizer_lineage"), dict) else {}
    roundtrip = files.get("tokenizer_roundtrip") if isinstance(files.get("tokenizer_roundtrip"), dict) else {}
    sample_rows = roundtrip.get("sample_rows") if isinstance(roundtrip.get("sample_rows"), list) else []

    roundtrip_max_token_id: int | None = None
    for row in sample_rows:
        if not isinstance(row, dict):
            continue
        ids = row.get("token_ids")
        if not isinstance(ids, list):
            continue
        for tid in ids:
            if isinstance(tid, int) and tid >= 0:
                if roundtrip_max_token_id is None or tid > roundtrip_max_token_id:
                    roundtrip_max_token_id = tid

    candidate_values: list[str] = []
    runbook_tokenizer_aliases: list[str] = []
    if run_dir is not None:
        sibling_preview = run_dir.parent / f"{run_dir.name}_tokenize_preview" / "tokenizer.json"
        nested_preview = run_dir / "tokenize_preview" / "tokenizer.json"
        runbook_tokenizer_aliases.extend([str(sibling_preview), str(nested_preview)])
    # Prefer run-scoped tokenizer first; data-lab lineage can point to sub-runs.
    for raw in (
        str((run_dir / "tokenizer.json")) if run_dir is not None else None,
        str(model_root / "tokenizer.json"),
        str(ck_build_path / "tokenizer.json"),
        data_lab.get("tokenizer_json_path"),
        tokenizer_lineage.get("tokenizer_path"),
        *runbook_tokenizer_aliases,
    ):
        if isinstance(raw, str) and raw.strip():
            candidate_values.append(raw.strip())

    candidate_paths: list[Path] = []
    seen_paths: set[str] = set()
    for raw in candidate_values:
        resolved = _resolve_asset_path(raw, ck_build_path, model_root)
        if not resolved or not resolved.exists():
            continue
        key = str(resolved)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        candidate_paths.append(resolved)
    if not candidate_paths:
        return None

    def _vocab_max_id(payload_obj: dict) -> int | None:
        model_obj = payload_obj.get("model") if isinstance(payload_obj, dict) else {}
        if not isinstance(model_obj, dict):
            return None
        vocab_obj = model_obj.get("vocab")
        if not isinstance(vocab_obj, dict):
            return None
        max_id: int | None = None
        for idx in vocab_obj.values():
            if isinstance(idx, int) and idx >= 0:
                if max_id is None or idx > max_id:
                    max_id = int(idx)
        return max_id

    tokenizer_json_path: Path | None = None
    payload: dict | None = None
    first_parse_error: str | None = None
    fallback_path: Path | None = None
    fallback_payload: dict | None = None

    for candidate in candidate_paths:
        try:
            parsed = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as e:
            if first_parse_error is None:
                first_parse_error = f"{candidate}: {e}"
            continue
        if fallback_payload is None:
            fallback_path = candidate
            fallback_payload = parsed
        vocab_max = _vocab_max_id(parsed)
        if roundtrip_max_token_id is None:
            tokenizer_json_path = candidate
            payload = parsed
            break
        if isinstance(vocab_max, int) and vocab_max >= roundtrip_max_token_id:
            tokenizer_json_path = candidate
            payload = parsed
            break

    if payload is None and fallback_payload is not None:
        tokenizer_json_path = fallback_path
        payload = fallback_payload

    if payload is None or tokenizer_json_path is None:
        return {
            "status": "error",
            "path": str(candidate_paths[0]),
            "error": f"failed_to_parse_tokenizer_json: {first_parse_error or 'unknown_error'}",
        }

    model = payload.get("model") if isinstance(payload, dict) else {}
    if not isinstance(model, dict):
        model = {}
    model_type = str(model.get("type") or "unknown")
    vocab_raw = model.get("vocab")
    vocab_items: list[tuple[int, str]] = []
    id_to_piece: dict[int, str] = {}
    piece_to_id: dict[str, int] = {}
    if isinstance(vocab_raw, dict):
        for piece, idx in vocab_raw.items():
            if isinstance(piece, str) and isinstance(idx, int) and idx >= 0:
                vocab_items.append((int(idx), piece))
                id_to_piece[int(idx)] = piece
                piece_to_id[piece] = int(idx)
    vocab_items.sort(key=lambda kv: kv[0])

    merges_raw = model.get("merges")
    merge_samples: list[dict] = []
    merge_count = 0
    merge_piece_map: dict[str, tuple[str, str, int]] = {}
    if isinstance(merges_raw, list):
        merge_count = len(merges_raw)
        for i, row in enumerate(merges_raw):
            left, right = _parse_merge_row(row)
            if not left and not right:
                continue
            merged = left + right
            if merged not in merge_piece_map:
                merge_piece_map[merged] = (left, right, int(i))
            if i >= 48:
                continue
            merge_samples.append(
                {
                    "rank": int(i),
                    "left": _piece_display(left),
                    "right": _piece_display(right),
                    "merged_hint": _piece_display(merged),
                }
            )

    non_ascii_piece_count = 0
    for _, piece in vocab_items:
        if any(ord(ch) > 127 for ch in piece):
            non_ascii_piece_count += 1

    vocab_samples = [
        {
            "id": int(idx),
            "piece": _piece_display(piece),
            "bytes_hex": _bytes_hex_preview(piece),
            "byte_len": len(piece.encode("utf-8", errors="replace")),
            "char_len": len(piece),
        }
        for idx, piece in vocab_items[:48]
    ]
    token_lookup_row_limit = min(len(vocab_items), 50000)
    token_lookup_rows = [
        {
            "id": int(idx),
            "piece": _piece_display(piece),
            "bytes_hex": _bytes_hex_preview(piece),
            "byte_len": len(piece.encode("utf-8", errors="replace")),
            "char_len": len(piece),
        }
        for idx, piece in vocab_items[:token_lookup_row_limit]
    ]

    token_index_dir = run_dir if isinstance(run_dir, Path) else tokenizer_json_path.parent
    token_index_path = token_index_dir / "tokenizer_vocab_index.jsonl"
    token_index_error: str | None = None
    try:
        token_index_dir.mkdir(parents=True, exist_ok=True)
        with token_index_path.open("w", encoding="utf-8") as fh:
            for idx, piece in vocab_items:
                row = {
                    "id": int(idx),
                    "piece": _piece_display(piece, limit=512),
                    "bytes_hex": _bytes_hex_preview(piece, limit=64),
                    "byte_len": len(piece.encode("utf-8", errors="replace")),
                    "char_len": len(piece),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        token_index_error = str(e)

    added_tokens = payload.get("added_tokens")
    special_tokens: list[dict] = []
    if isinstance(added_tokens, list):
        for row in added_tokens[:32]:
            if not isinstance(row, dict):
                continue
            content = row.get("content")
            token_id = row.get("id")
            if isinstance(content, str):
                special_tokens.append(
                    {
                        "id": int(token_id) if isinstance(token_id, int) else None,
                        "content": _piece_display(content),
                        "special": bool(row.get("special")),
                    }
                )

    special_ids = {
        int(row["id"])
        for row in special_tokens
        if isinstance(row.get("id"), int)
    }

    merge_depth_cache: dict[str, int] = {}

    def _merge_depth(piece: str, stack: set[str] | None = None) -> int:
        cached = merge_depth_cache.get(piece)
        if cached is not None:
            return cached
        parent = merge_piece_map.get(piece)
        if parent is None:
            merge_depth_cache[piece] = 0
            return 0
        if stack is None:
            stack = set()
        if piece in stack:
            # Defensive cycle break for malformed merge tables.
            merge_depth_cache[piece] = 0
            return 0
        stack.add(piece)
        left, right, _ = parent
        depth = 1 + max(_merge_depth(left, stack), _merge_depth(right, stack))
        stack.remove(piece)
        merge_depth_cache[piece] = depth
        return depth

    source_counts = {"base": 0, "merged": 0, "special": 0}
    for idx, piece in vocab_items:
        if idx in special_ids:
            source_counts["special"] += 1
        elif piece in merge_piece_map:
            source_counts["merged"] += 1
        else:
            source_counts["base"] += 1

    token_table_rows: list[dict] = []
    seen_token_ids: set[int] = set()

    def _append_token_row(token_id: int) -> None:
        if token_id in seen_token_ids:
            return
        piece = id_to_piece.get(int(token_id))
        if not isinstance(piece, str):
            return
        seen_token_ids.add(int(token_id))
        parent = merge_piece_map.get(piece)
        source = "special" if token_id in special_ids else ("merged" if parent is not None else "base")
        merge_depth = _merge_depth(piece) if parent is not None else 0
        token_table_rows.append(
            {
                "id": int(token_id),
                "piece": _piece_display(piece),
                "bytes_hex": _bytes_hex_preview(piece),
                "char_class": "ascii" if all(ord(ch) <= 127 for ch in piece) else "non_ascii",
                "source": source,
                "merge_depth": int(merge_depth),
                "merge_rank": int(parent[2]) if parent is not None else None,
                "merge_left": _piece_display(parent[0]) if parent is not None else None,
                "merge_right": _piece_display(parent[1]) if parent is not None else None,
            }
        )

    # Build a representative, stable table:
    # - head IDs, then early merges, then special tokens, then tail IDs.
    for idx, _ in vocab_items[:96]:
        _append_token_row(int(idx))

    merged_ranked_ids: list[int] = []
    for merged_piece, (_, _, rank) in sorted(merge_piece_map.items(), key=lambda item: item[1][2]):
        tok_id = piece_to_id.get(merged_piece)
        if isinstance(tok_id, int):
            merged_ranked_ids.append(int(tok_id))
        if len(merged_ranked_ids) >= 160:
            break
    for token_id in merged_ranked_ids[:96]:
        _append_token_row(token_id)

    for token_id in sorted(special_ids):
        _append_token_row(token_id)

    for idx, _ in vocab_items[-24:]:
        _append_token_row(int(idx))

    # Fill a little more coverage using stride sampling if needed.
    if len(token_table_rows) < 160 and len(vocab_items) > 0:
        stride = max(1, len(vocab_items) // 64)
        for pos in range(0, len(vocab_items), stride):
            _append_token_row(int(vocab_items[pos][0]))
            if len(token_table_rows) >= 192:
                break

    token_table_row_limit = 220
    token_table_rows = token_table_rows[:token_table_row_limit]

    pre_tokenizer = payload.get("pre_tokenizer")
    decoder = payload.get("decoder")
    pre_tokenizer_type = pre_tokenizer.get("type") if isinstance(pre_tokenizer, dict) else None
    decoder_type = decoder.get("type") if isinstance(decoder, dict) else None

    tokenizer_config_path: Path | None = None
    tokenizer_class: str | None = None
    tokenizer_config_candidates = [
        tokenizer_json_path.parent / "tokenizer_config.json",
        ck_build_path / "tokenizer_config.json",
        model_root / "tokenizer_config.json",
    ]
    if run_dir is not None:
        tokenizer_config_candidates.insert(0, run_dir / "tokenizer_config.json")
    for candidate in tokenizer_config_candidates:
        if candidate.exists():
            tokenizer_config_path = candidate
            break
    if tokenizer_config_path is not None:
        try:
            tokenizer_config_payload = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            if isinstance(tokenizer_config_payload, dict):
                raw_class = tokenizer_config_payload.get("tokenizer_class")
                if isinstance(raw_class, str) and raw_class.strip():
                    tokenizer_class = raw_class.strip()
        except Exception:
            tokenizer_class = None

    sentencepiece_model_path: Path | None = None
    sentencepiece_candidates = [
        tokenizer_json_path.parent / "tokenizer.model",
        ck_build_path / "tokenizer.model",
        model_root / "tokenizer.model",
    ]
    if run_dir is not None:
        sentencepiece_candidates.insert(0, run_dir / "tokenizer.model")
    for candidate in sentencepiece_candidates:
        if candidate.exists():
            sentencepiece_model_path = candidate
            break

    example = None
    for row in sample_rows:
        if not isinstance(row, dict):
            continue
        ids = row.get("token_ids")
        if not isinstance(ids, list) or not ids:
            continue
        flow = []
        for tid in ids[:48]:
            if not isinstance(tid, int):
                continue
            piece = id_to_piece.get(int(tid))
            flow.append(
                {
                    "id": int(tid),
                    "piece": _piece_display(piece) if isinstance(piece, str) else "<missing>",
                }
            )
        example = {
            "line_no": row.get("line_no"),
            "source": row.get("source"),
            "decoded": row.get("decoded"),
            "exact_match": row.get("exact_match"),
            "token_flow": flow,
        }
        break

    return {
        "status": "ok",
        "path": str(tokenizer_json_path),
        "model_type": model_type,
        "vocab_size": int(len(vocab_items)),
        "merge_count": int(merge_count),
        "pre_tokenizer_type": pre_tokenizer_type,
        "decoder_type": decoder_type,
        "bytelevel_mode": bool(pre_tokenizer_type == "ByteLevel" or decoder_type == "ByteLevel"),
        "tokenizer_config_path": str(tokenizer_config_path) if tokenizer_config_path is not None else None,
        "tokenizer_class": tokenizer_class,
        "sentencepiece_model_path": str(sentencepiece_model_path) if sentencepiece_model_path is not None else None,
        "sentencepiece_model_present": bool(sentencepiece_model_path is not None),
        "ascii_piece_count": int(len(vocab_items) - non_ascii_piece_count),
        "non_ascii_piece_count": int(non_ascii_piece_count),
        "vocab_samples": vocab_samples,
        "merge_samples": merge_samples,
        "special_tokens": special_tokens,
        "token_lookup_rows": token_lookup_rows,
        "token_lookup_meta": {
            "rows_returned": int(len(token_lookup_rows)),
            "row_limit": int(token_lookup_row_limit),
            "total_vocab_size": int(len(vocab_items)),
            "truncated": bool(len(vocab_items) > token_lookup_row_limit),
            "sample_policy": "head_by_token_id_capped",
        },
        "token_index_path": str(token_index_path),
        "token_index_status": "ok" if token_index_error is None else "error",
        "token_index_error": token_index_error,
        "token_index_rows": int(len(vocab_items)),
        "token_table_rows": token_table_rows,
        "token_table_meta": {
            "total_vocab_size": int(len(vocab_items)),
            "rows_returned": int(len(token_table_rows)),
            "row_limit": int(token_table_row_limit),
            "sample_policy": "head_ids + early_merges + special_tokens + tail_ids + stride_fill",
            "source_counts": source_counts,
        },
        "encode_decode_example": example,
    }


def list_available_models():
    """List all models in cache."""
    models = []
    for cache_root in [CACHE_PATH]:
        if not cache_root.exists():
            continue
        for model_dir in cache_root.iterdir():
            if not model_dir.is_dir():
                continue

            ck_build = model_dir / "ck_build"
            has_data = (
                (ck_build.exists() and (ck_build / "ir1_decode.json").exists()) or
                (model_dir / "ir1_decode.json").exists()
            )

            models.append({
                "name": model_dir.name,
                "path": str(ck_build),
                "has_data": has_data
            })

    return sorted(models, key=lambda m: m["name"])


def collect_analysis_checkpoints(search_roots: list[Path]) -> dict | None:
    """Load and merge analysis_checkpoint_step_*.json files by step."""
    step_to_path: dict[int, Path] = {}
    # Preserve root priority: earlier roots win for the same step.
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for candidate in sorted(root.glob("analysis_checkpoint_step_*.json")):
            m = re.search(r"_step_(\d+)\.json$", candidate.name)
            if not m:
                continue
            step = int(m.group(1))
            if step not in step_to_path:
                step_to_path[step] = candidate

    checkpoints: list[dict] = []
    for step in sorted(step_to_path.keys()):
        p = step_to_path[step]
        try:
            with open(p, "r") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                if "step" not in payload:
                    payload["step"] = step
                payload.setdefault("_source", str(p))
                checkpoints.append(payload)
        except Exception as e:
            print(f"  ! analysis checkpoint {p.name}: {e}")

    if not checkpoints:
        return None

    return {
        "schema_version": "ck.analysis.v1",
        "count": len(checkpoints),
        "checkpoints": checkpoints,
    }


def collect_training_logbook(
    search_roots: list[Path],
    *,
    run_dir: Path | None = None,
    strict_run_scope: bool = False,
) -> dict | None:
    """Load an operator-facing training logbook markdown file."""
    repo_logbook = V8_ROOT / "reports" / "TRAINING_LOGBOOK.md"
    names = [
        "training_logbook.md",
        "TRAINING_LOGBOOK.md",
        "training_logbook_latest.md",
    ]
    candidates: list[Path] = []
    seen: set[str] = set()

    def push(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    for root in search_roots:
        for name in names:
            push(root / name)
    # Keep repo logbook as a final fallback even in strict run scope:
    # it is operator guidance, not mutable model telemetry.
    push(repo_logbook)

    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"  ! training logbook {path}: {e}")
            continue
        if not text.strip():
            continue

        source = "run_local"
        if path == repo_logbook:
            source = "repo_default"
        elif run_dir is not None:
            try:
                path.relative_to(run_dir)
                source = "run_local"
            except Exception:
                source = "external"

        return {
            "path": str(path),
            "source": source,
            "line_count": len(text.splitlines()),
            "markdown": text,
        }

    return None


def collect_training_checkpoint_history(run_root: Path | None) -> dict | None:
    """Collect pass-level and checkpoint-level history for training dashboards."""
    if run_root is None or not run_root.exists() or not run_root.is_dir():
        return None

    history: dict[str, Any] = {
        "schema": "ck.training_checkpoint_history.v1",
        "run_dir": str(run_root),
        "passes": [],
        "checkpoints": [],
    }

    def _mtime_fields(path: Path) -> dict[str, Any]:
        try:
            ts = float(path.stat().st_mtime)
            return {
                "mtime_unix": ts,
                "mtime_iso": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            }
        except Exception:
            return {}

    pipeline_root = run_root / ".ck_pipeline"
    if pipeline_root.exists() and pipeline_root.is_dir():
        workdirs = sorted(
            [p for p in pipeline_root.iterdir() if p.is_dir() and p.name.startswith("ascii_bpe_")],
            key=lambda p: p.name,
        )
        for order, work_dir in enumerate(workdirs, start=1):
            row: dict[str, Any] = {
                "order": order,
                "id": work_dir.name,
                "work_dir": str(work_dir),
                "train_ck_json": str(work_dir / "train_ck.json"),
                "train_torch_ref_json": str(work_dir / "train_torch_ref.json"),
                "has_torch_ref": (work_dir / "train_torch_ref.json").exists(),
            }

            ck_payload = _load_json_loose(work_dir / "train_ck.json")
            if isinstance(ck_payload, dict):
                row["source"] = ck_payload.get("source")
                curve = ck_payload.get("loss_curve")
                if isinstance(curve, list):
                    row["steps"] = len(curve)
                    losses: list[float] = []
                    grad_norms: list[float] = []
                    lrs: list[float] = []
                    for item in curve:
                        if not isinstance(item, dict):
                            continue
                        loss = item.get("loss_ck")
                        if not isinstance(loss, (int, float)):
                            loss = item.get("loss")
                        if isinstance(loss, (int, float)):
                            losses.append(float(loss))
                        grad = item.get("grad_norm")
                        if isinstance(grad, (int, float)):
                            grad_norms.append(float(grad))
                        lr = item.get("lr")
                        if isinstance(lr, (int, float)):
                            lrs.append(float(lr))

                    if losses:
                        row["loss_first"] = losses[0]
                        row["loss_final"] = losses[-1]
                        row["loss_min"] = min(losses)
                    if grad_norms:
                        row["grad_norm_final"] = grad_norms[-1]
                    if lrs:
                        row["lr_final"] = lrs[-1]

            row.update(_mtime_fields(work_dir / "train_ck.json" if (work_dir / "train_ck.json").exists() else work_dir))
            history["passes"].append(row)

    checkpoint_by_step: dict[int, dict[str, Any]] = {}
    checkpoints_dir = run_root / "checkpoints"
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        for manifest_path in sorted(checkpoints_dir.glob("weights_step_*_manifest.json")):
            m = re.search(r"weights_step_(\d+)_manifest\.json$", manifest_path.name)
            if not m:
                continue
            step = int(m.group(1))
            bump_path = checkpoints_dir / f"weights_step_{m.group(1)}.bump"
            row: dict[str, Any] = checkpoint_by_step.get(step, {"step": step})
            row["manifest"] = str(manifest_path)
            row["bump"] = str(bump_path)
            row["bump_exists"] = bump_path.exists()
            row.update(_mtime_fields(manifest_path))

            manifest_payload = _load_json_loose(manifest_path)
            if isinstance(manifest_payload, dict):
                for key in ("weights", "floats", "bytes", "reason"):
                    if key in manifest_payload:
                        row[key] = manifest_payload.get(key)
            checkpoint_by_step[step] = row

    policy_candidates = [
        run_root / "training_checkpoint_policy_latest.json",
        run_root / "training_checkpoint_policy.json",
    ]
    for policy_path in policy_candidates:
        policy_payload = _load_json_loose(policy_path)
        if not isinstance(policy_payload, dict):
            continue
        files = policy_payload.get("files")
        if not isinstance(files, list):
            continue
        for item in files:
            if not isinstance(item, dict):
                continue
            step = item.get("step")
            if not isinstance(step, int):
                continue
            row = checkpoint_by_step.get(step, {"step": step})
            row["policy_source"] = str(policy_path)
            for key in ("reason", "weights", "floats", "bytes", "bump", "manifest"):
                if key in item:
                    row[key] = item.get(key)
            checkpoint_by_step[step] = row

    history["checkpoints"] = [checkpoint_by_step[k] for k in sorted(checkpoint_by_step.keys())]
    history["summary"] = {
        "passes": len(history["passes"]),
        "checkpoints": len(history["checkpoints"]),
    }

    if history["summary"]["passes"] == 0 and history["summary"]["checkpoints"] == 0:
        return None
    return history


def _has_renderable_profile_payload(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    entries = payload.get("entries")
    if isinstance(entries, list) and len(entries) > 0:
        return True
    by_op = payload.get("by_op")
    if isinstance(by_op, dict) and len(by_op) > 0:
        return True
    by_mode = payload.get("by_mode")
    if isinstance(by_mode, dict) and len(by_mode) > 0:
        return True
    return False


def _is_profile_summary_stub(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    if _has_renderable_profile_payload(payload):
        return False
    return str(payload.get("schema") or "") == "ck.profile.summary.v1"


def _derive_profile_alias_roots(run_dir: Path | None, model_root: Path) -> list[Path]:
    """
    Derive sibling cache roots from local GGUF filenames.
    Example:
      run dir: .../Qwen--Qwen3-0.6B-GGUF
      gguf:    Qwen3-0.6B-Q8_0.gguf
      alias:   .../Qwen3-0.6B-Q8_0
    """
    bases: list[Path] = []
    if run_dir is not None:
        bases.append(run_dir)
    bases.append(model_root)

    roots: list[Path] = []
    seen: set[str] = set()

    def push(p: Path) -> None:
        key = str(p)
        if key in seen:
            return
        seen.add(key)
        roots.append(p)

    for base in bases:
        if not base.exists() or not base.is_dir():
            continue
        parent = base.parent
        for gguf in sorted(base.glob("*.gguf")):
            stem = gguf.stem
            if not stem:
                continue
            alias_root = parent / stem
            if alias_root.exists() and alias_root.is_dir():
                push(alias_root)
                push(alias_root / "ck_build")
                push(alias_root / ".ck_build")
    return roots


def _derive_runbook_alias_roots(run_dir: Path | None) -> list[Path]:
    """Runbook convenience aliases so a single root report can show staged artifacts."""
    if run_dir is None:
        return []
    aliases: list[Path] = []
    sibling_preview = run_dir.parent / f"{run_dir.name}_tokenize_preview"
    if sibling_preview.exists() and sibling_preview.is_dir():
        aliases.append(sibling_preview)
    nested_preview = run_dir / "tokenize_preview"
    if nested_preview.exists() and nested_preview.is_dir():
        aliases.append(nested_preview)
    for idx in (1, 2):
        row_dir = run_dir / f"parity_svg_row{idx}"
        if row_dir.exists() and row_dir.is_dir():
            aliases.append(row_dir)
            ck_build = row_dir / "ck_build"
            dot_ck_build = row_dir / ".ck_build"
            if ck_build.exists() and ck_build.is_dir():
                aliases.append(ck_build)
            if dot_ck_build.exists() and dot_ck_build.is_dir():
                aliases.append(dot_ck_build)
    return aliases


def _resolve_path_loose(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except TypeError:
        # Python <3.10 compatibility (strict kw may be unavailable in some envs)
        return path.resolve()
    except Exception:
        return path


def _path_under_any_root(path: Path, roots: list[Path]) -> bool:
    target = _resolve_path_loose(path)
    for root in roots:
        base = _resolve_path_loose(root)
        if target == base or base in target.parents:
            return True
    return False


def _load_json_loose(path: Path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _manifest_rows_from_payload(payload: dict) -> int | None:
    for key in ("output_rows", "num_rows", "num_train", "num_samples", "rows", "train_rows"):
        value = payload.get(key)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and value == value and value not in (float("inf"), float("-inf")):
            return int(value)
    return None


def _infer_dataset_stage_from_name(name: str, default_stage: str) -> str:
    probe = str(name or "").lower()
    if "dpo" in probe:
        return "dpo"
    if "grpo" in probe:
        return "grpo"
    if "ppo" in probe or "rl" in probe:
        return "ppo"
    if any(tok in probe for tok in ("stage_b", "midtrain", "instruction", "sft")):
        return "midtrain"
    if any(tok in probe for tok in ("stage_a", "bridge", "assets", "ascii", "svg")):
        return "pretrain"
    return default_stage


def _derive_dataset_catalog(pipeline: dict, run_dir: Path | None) -> list[dict]:
    data_lab = pipeline.get("data_lab") if isinstance(pipeline.get("data_lab"), dict) else {}
    active_stage = str(pipeline.get("active_stage") or "pretrain")
    qc = data_lab.get("dataset_qc") if isinstance(data_lab.get("dataset_qc"), dict) else {}

    entries: list[dict] = []
    seen_paths: set[str] = set()

    def _resolve_manifest_out_path(raw_path: str, manifest_path: Path | None = None) -> tuple[str, str]:
        """
        Resolve manifest out_path values that may be stale after dataset relocation.
        Returns (resolved_path, remap_note_suffix).
        """
        p = _resolve_path_loose(Path(raw_path).expanduser())
        if p.exists():
            return str(p), ""

        name = Path(raw_path).name
        probes: list[Path] = []
        if manifest_path is not None:
            probes.append(manifest_path.parent / name)
        probes.extend(
            [
                V8_ROOT / "data" / "generated" / name,
                V8_ROOT / "data" / name,
            ]
        )
        for cand in probes:
            if cand.exists():
                return str(cand), f"remapped from stale out_path ({raw_path})"
        return raw_path, ""

    def _add_entry(
        *,
        stage: str,
        kind: str,
        name: str,
        path: str,
        rows: int | None = None,
        note: str = "",
        source: str = "local",
    ) -> None:
        key = str(path)
        if not key or key in seen_paths:
            return
        seen_paths.add(key)
        entries.append(
            {
                "stage": str(stage),
                "kind": str(kind),
                "name": str(name),
                "path": key,
                "rows": int(rows) if isinstance(rows, int) else None,
                "note": str(note or ""),
                "source": str(source),
            }
        )

    dataset_path = data_lab.get("dataset_path")
    if isinstance(dataset_path, str) and dataset_path.strip():
        active_rows = qc.get("non_empty_lines")
        _add_entry(
            stage=active_stage,
            kind="active_dataset",
            name=Path(dataset_path).name,
            path=dataset_path,
            rows=int(active_rows) if isinstance(active_rows, int) else None,
            note="dataset used for this report",
        )

    manifest_candidates: list[Path] = []
    ds_dir_raw = data_lab.get("dataset_dir")
    if isinstance(ds_dir_raw, str) and ds_dir_raw.strip():
        ds_dir = Path(ds_dir_raw).expanduser()
        if ds_dir.exists():
            manifest_candidates.extend(sorted(ds_dir.glob("*manifest.json")))
    repo_data_dir = V8_ROOT / "data"
    if repo_data_dir.exists():
        manifest_candidates.extend(sorted(repo_data_dir.glob("svg*_manifest.json")))
    if run_dir is not None:
        ap_dir = run_dir / "autopilot"
        if ap_dir.exists():
            manifest_candidates.extend(sorted(ap_dir.glob("iter_*/*manifest*.json")))
            manifest_candidates.extend(sorted(ap_dir.glob("iter_*/**/*manifest*.json")))

    seen_manifest: set[str] = set()
    for manifest_path in manifest_candidates:
        key = str(_resolve_path_loose(manifest_path))
        if key in seen_manifest:
            continue
        seen_manifest.add(key)
        payload = _load_json_loose(manifest_path)
        if not isinstance(payload, dict):
            continue
        rows = _manifest_rows_from_payload(payload)
        out_path = payload.get("out_path")
        dataset_name = payload.get("dataset_name")
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            if isinstance(out_path, str) and out_path.strip():
                dataset_name = Path(out_path).name
            else:
                dataset_name = manifest_path.stem
        stage = _infer_dataset_stage_from_name(manifest_path.name, active_stage)
        note_parts: list[str] = []
        if isinstance(payload.get("source_files"), int):
            note_parts.append(f"source_files={payload.get('source_files')}")
        mapped = payload.get("mapped_common_symbols_total")
        if isinstance(mapped, int):
            note_parts.append(f"mapped_symbols={mapped}")
        tcounts = payload.get("type_counts")
        if isinstance(tcounts, dict):
            note_parts.append(f"type_families={len(tcounts)}")
        _add_entry(
            stage=stage,
            kind="manifest",
            name=dataset_name,
            path=str(manifest_path),
            rows=rows,
            note=", ".join(note_parts),
            source="manifest",
        )
        if isinstance(out_path, str) and out_path.strip():
            resolved_out_path, remap_note = _resolve_manifest_out_path(out_path, manifest_path)
            out_note = f"derived from {manifest_path.name}"
            if remap_note:
                out_note = f"{out_note}; {remap_note}"
            _add_entry(
                stage=stage,
                kind="generated_dataset",
                name=Path(resolved_out_path).name,
                path=resolved_out_path,
                rows=rows,
                note=out_note,
                source="manifest_output",
            )

    return entries


def _derive_stage_artifacts(pipeline: dict, loaded_paths: dict[str, str], run_dir: Path | None) -> list[dict]:
    timeline = pipeline.get("stage_timeline")
    if not isinstance(timeline, list) or not timeline:
        timeline = [{"stage": str(pipeline.get("active_stage") or "pretrain"), "status": "active", "active": True}]
    active_stage = str(pipeline.get("active_stage") or "pretrain")
    data_provenance = pipeline.get("data_provenance")
    if not isinstance(data_provenance, list):
        data_provenance = []
    prov_by_stage: dict[str, dict] = {}
    for row in data_provenance:
        if isinstance(row, dict):
            stage = row.get("stage")
            if isinstance(stage, str) and stage not in prov_by_stage:
                prov_by_stage[stage] = row

    data_lab = pipeline.get("data_lab") if isinstance(pipeline.get("data_lab"), dict) else {}
    artifacts = data_lab.get("artifacts") if isinstance(data_lab.get("artifacts"), dict) else {}
    active_artifacts: list[dict] = []

    def _push(label: str, path: str | None, required: bool = False) -> None:
        if not isinstance(path, str) or not path.strip():
            return
        active_artifacts.append(
            {
                "label": str(label),
                "path": str(path),
                "required": bool(required),
                "exists": bool(Path(path).expanduser().exists()),
            }
        )

    _push("dataset_path", data_lab.get("dataset_path"), required=True)
    _push("tokenizer_json_path", data_lab.get("tokenizer_json_path"), required=True)
    _push("dataset_qc_json", artifacts.get("dataset_qc_json"), required=True)
    _push("dataset_profile_json", artifacts.get("dataset_profile_json"), required=True)
    _push("tokenizer_roundtrip_json", artifacts.get("tokenizer_roundtrip_json"), required=True)
    _push("post_train_eval_json", artifacts.get("post_train_eval_json"), required=False)
    _push("training_pipeline_json", loaded_paths.get("training_pipeline"), required=True)
    _push("corpus_sampling_log_json", loaded_paths.get("corpus_sampling_log"), required=False)

    if run_dir is not None:
        for label, rel in (
            ("training_loss_curve", "training_loss_curve_latest.json"),
            ("training_parity", "training_parity_latest.json"),
            ("ir1_train", "ir1_train_forward.json"),
            ("ir2_train", "ir2_train_backward.json"),
            ("layout_train", "layout_train.json"),
            ("train_exec_plan", "train_exec_plan.json"),
        ):
            p = run_dir / rel
            if p.exists():
                _push(label, str(p), required=False)

    rows: list[dict] = []
    for row in timeline:
        if not isinstance(row, dict):
            continue
        stage = str(row.get("stage") or "")
        if not stage:
            continue
        prov = prov_by_stage.get(stage, {})
        rows.append(
            {
                "stage": stage,
                "status": str(row.get("status") or ("active" if stage == active_stage else "planned")),
                "active": bool(row.get("active") is True or stage == active_stage),
                "dataset_name": prov.get("dataset_name"),
                "token_count": prov.get("token_count"),
                "source_path": prov.get("source_path"),
                "artifacts": list(active_artifacts) if stage == active_stage else [],
            }
        )
    return rows


def _normalize_curriculum_stage_name(raw_stage: Any) -> str | None:
    stage = str(raw_stage or "").strip().lower()
    if not stage:
        return None
    aliases = {
        "stage_a": "pretrain",
        "stage_b": "midtrain",
        "pretrain": "pretrain",
        "midtrain": "midtrain",
        "sft": "sft",
        "dpo": "dpo",
        "grpo": "grpo",
        "ppo": "ppo",
    }
    return aliases.get(stage, stage)


def _extract_stage_loss_series(payload: Any) -> list[dict[str, float]]:
    rows: list[Any] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("loss_curve"), list):
            rows = payload.get("loss_curve") or []
        elif isinstance(payload.get("steps"), list):
            rows = payload.get("steps") or []
    elif isinstance(payload, list):
        rows = payload

    series: list[dict[str, float]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        step_raw = row.get("step")
        if not isinstance(step_raw, (int, float)):
            step_raw = idx + 1
        loss_raw = row.get("loss_ck")
        if not isinstance(loss_raw, (int, float)):
            loss_raw = row.get("loss")
        if not isinstance(loss_raw, (int, float)):
            continue
        step = float(step_raw)
        loss = float(loss_raw)
        if not (step >= 0):
            continue
        series.append({"step": step, "loss": loss})
    return series


def _extract_stage_loss_series_by_stage(payload: Any) -> dict[str, list[dict[str, float]]]:
    rows: list[Any] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("loss_curve"), list):
            rows = payload.get("loss_curve") or []
        elif isinstance(payload.get("steps"), list):
            rows = payload.get("steps") or []
    elif isinstance(payload, list):
        rows = payload

    by_stage: dict[str, list[dict[str, float]]] = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        step_raw = row.get("step")
        if not isinstance(step_raw, (int, float)):
            step_raw = idx + 1
        loss_raw = row.get("loss_ck")
        if not isinstance(loss_raw, (int, float)):
            loss_raw = row.get("loss")
        if not isinstance(loss_raw, (int, float)):
            continue
        stage_raw = row.get("source_stage") or row.get("stage") or row.get("curriculum_stage")
        stage = _normalize_curriculum_stage_name(stage_raw) or "unassigned"
        step = float(step_raw)
        loss = float(loss_raw)
        if not (step >= 0):
            continue
        by_stage.setdefault(stage, []).append({"step": step, "loss": loss})
    return by_stage


def _sample_stage_loss_series(series: list[dict[str, float]], max_points: int = 160) -> list[dict[str, float]]:
    if len(series) <= max_points:
        return list(series)
    stride = max(1, len(series) // max_points)
    sampled = series[::stride]
    if sampled[-1] != series[-1]:
        sampled.append(series[-1])
    return sampled


def _derive_stage_loss_history(
    pipeline: dict[str, Any],
    files: dict[str, Any],
    loaded_paths: dict[str, str],
    run_dir: Path | None,
) -> dict[str, Any]:
    active_stage = _normalize_curriculum_stage_name(pipeline.get("active_stage")) or "pretrain"
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    stage_refs: list[dict[str, Any]] = []
    active_prov: dict[str, Any] = {}
    prov_by_stage: dict[str, dict[str, Any]] = {}

    prov_rows = pipeline.get("data_provenance")
    if isinstance(prov_rows, list):
        for row in prov_rows:
            if not isinstance(row, dict):
                continue
            row_stage = _normalize_curriculum_stage_name(row.get("stage") or row.get("curriculum_stage"))
            if row_stage and row_stage not in prov_by_stage:
                prov_by_stage[row_stage] = row
            if row_stage == active_stage:
                active_prov = row
                break

    if run_dir is not None:
        # Prefer latest snapshot first; only fall back to legacy file when latest
        # is absent or malformed so stale stage ordering does not leak into stage refs.
        for cand in (run_dir / "training_pipeline_latest.json", run_dir / "training_pipeline.json"):
            snap = _load_json_loose(cand)
            if not isinstance(snap, dict):
                continue
            prov = snap.get("data_provenance")
            if not isinstance(prov, list):
                continue
            for row in prov:
                if not isinstance(row, dict):
                    continue
                stage_name = _normalize_curriculum_stage_name(row.get("stage") or row.get("curriculum_stage"))
                if not stage_name:
                    continue
                stage_refs.append(
                    {
                        "stage": stage_name,
                        "token_count": row.get("token_count"),
                        "dataset_name": row.get("dataset_name"),
                        "source_path": row.get("source_path"),
                    }
                )

    def _infer_stage_ref(train_ck: dict[str, Any]) -> dict[str, Any] | None:
        if not stage_refs:
            return None
        total_tokens = train_ck.get("total_tokens")
        if isinstance(total_tokens, (int, float)):
            token_matches = [r for r in stage_refs if isinstance(r.get("token_count"), (int, float)) and int(r.get("token_count")) == int(total_tokens)]
            if len(token_matches) == 1:
                return token_matches[0]
        data_source = train_ck.get("data_source")
        ds_name = None
        if isinstance(data_source, dict) and isinstance(data_source.get("dataset_name"), str):
            ds_name = data_source.get("dataset_name")
        if isinstance(ds_name, str) and ds_name.strip():
            ds_name_l = ds_name.strip().lower()
            for ref in stage_refs:
                ref_name = ref.get("dataset_name")
                if isinstance(ref_name, str) and ref_name.strip().lower() == ds_name_l:
                    return ref
        return None

    def _ingest(
        stage_hint: Any,
        payload: Any,
        *,
        source: str,
        path_hint: str | None = None,
        run_id: str | None = None,
        meta: dict[str, Any] | None = None,
        series_override: list[dict[str, float]] | None = None,
    ) -> None:
        stage = _normalize_curriculum_stage_name(stage_hint) or "unassigned"
        if isinstance(series_override, list) and series_override:
            series = list(series_override)
        else:
            series = _extract_stage_loss_series(payload)
        if not series:
            return

        path_key = str(path_hint or "")
        dedupe_key = (stage, path_key or (run_id or source))
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)

        losses = [p["loss"] for p in series]
        first_loss = losses[0]
        final_loss = losses[-1]
        drop_abs = first_loss - final_loss
        drop_pct = (drop_abs / first_loss * 100.0) if abs(first_loss) > 1e-12 else 0.0
        points = _sample_stage_loss_series(series, max_points=180)

        entry: dict[str, Any] = {
            "stage": stage,
            "source": source,
            "run_id": run_id or source,
            "path": path_hint,
            "steps": len(series),
            "first_loss": first_loss,
            "final_loss": final_loss,
            "min_loss": min(losses),
            "max_loss": max(losses),
            "drop_abs": drop_abs,
            "drop_pct": drop_pct,
            "points": points,
        }
        if isinstance(meta, dict):
            for key, value in meta.items():
                if value is not None:
                    entry[key] = value
        entries.append(entry)

    # Current active curve loaded in files.
    if "training_loss_curve" in files:
        current_payload = files.get("training_loss_curve")
        stage_series = _extract_stage_loss_series_by_stage(current_payload)
        if stage_series:
            # Always use active_stage from the pipeline plan rather than the per-step
            # source_stage labels in training_loss_curve.json.  Those labels can be
            # stale or mislabeled (e.g. showing "pretrain" when the run was SFT) because
            # the training script copies source_stage from the dataset provenance field
            # rather than the actual curriculum_stage of the current run.
            # training_loss_curve.json is the live artifact for the currently-active
            # training stage, so active_stage is always authoritative here.
            combined: list[dict] = []
            for series in stage_series.values():
                combined.extend(series)
            combined.sort(key=lambda p: p["step"])
            if combined:
                prov = prov_by_stage.get(active_stage, active_prov)
                _ingest(
                    active_stage,
                    current_payload,
                    source="training_loss_curve",
                    path_hint=loaded_paths.get("training_loss_curve"),
                    run_id=f"current:{active_stage}",
                    series_override=combined,
                    meta={
                        "dataset_name": prov.get("dataset_name") if isinstance(prov.get("dataset_name"), str) else None,
                        "source_path": prov.get("source_path") if isinstance(prov.get("source_path"), str) else None,
                        "raw_source_path": prov.get("source_path") if isinstance(prov.get("source_path"), str) else None,
                        "token_stream_path": None,
                        "total_tokens": prov.get("token_count"),
                        "dataset_provenance": "data_provenance",
                    },
                )
        else:
            _ingest(
                active_stage,
                current_payload,
                source="training_loss_curve",
                path_hint=loaded_paths.get("training_loss_curve"),
                run_id="current",
                meta={
                    "dataset_name": active_prov.get("dataset_name") if isinstance(active_prov.get("dataset_name"), str) else None,
                    "source_path": active_prov.get("source_path") if isinstance(active_prov.get("source_path"), str) else None,
                    "raw_source_path": active_prov.get("source_path") if isinstance(active_prov.get("source_path"), str) else None,
                    "token_stream_path": None,
                    "total_tokens": active_prov.get("token_count"),
                    "dataset_provenance": "data_provenance",
                },
            )

    # Stage-artifact curve pointers (often from *_latest snapshots).
    stage_artifacts = pipeline.get("stage_artifacts")
    if isinstance(stage_artifacts, list):
        for row in stage_artifacts:
            if not isinstance(row, dict):
                continue
            stage_name = row.get("stage")
            artifacts = row.get("artifacts")
            if not isinstance(artifacts, list):
                continue
            for art in artifacts:
                if not isinstance(art, dict):
                    continue
                if str(art.get("label") or "") != "training_loss_curve":
                    continue
                path_raw = art.get("path")
                if not isinstance(path_raw, str) or not path_raw.strip():
                    continue
                p = Path(path_raw).expanduser()
                payload = _load_json_loose(p)
                if isinstance(payload, dict):
                    _ingest(
                        stage_name,
                        payload,
                        source="stage_artifact",
                        path_hint=str(p),
                        run_id=f"artifact:{p.name}",
                    )

    # Build ledger index from files.run_ledger for historical run attribution.
    ledger_by_run_id: dict[str, dict] = {}
    _run_ledger_payload = files.get("run_ledger")
    if isinstance(_run_ledger_payload, dict):
        for _led_rec in (_run_ledger_payload.get("entries") or []):
            if isinstance(_led_rec, dict) and _led_rec.get("run_id"):
                ledger_by_run_id[str(_led_rec["run_id"])] = _led_rec
    # Also attempt direct read from run_dir if not loaded via files dict.
    if not ledger_by_run_id and run_dir is not None:
        _ledger_path = run_dir / "run_ledger.jsonl"
        if _ledger_path.exists():
            try:
                for _line in _ledger_path.read_text(encoding="utf-8").splitlines():
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _rec = json.loads(_line)
                    except Exception:
                        continue
                    if isinstance(_rec, dict) and _rec.get("run_id"):
                        ledger_by_run_id[str(_rec["run_id"])] = _rec
            except OSError:
                pass

    # Historical train runs under .ck_pipeline/ascii_bpe_*/train_ck.json.
    if run_dir is not None:
        pipe_root = run_dir / ".ck_pipeline"
        if pipe_root.exists():
            for sub in sorted(pipe_root.glob("ascii_bpe_*")):
                train_ck_path = sub / "train_ck.json"
                if not train_ck_path.exists():
                    continue
                train_ck = _load_json_loose(train_ck_path)
                if not isinstance(train_ck, dict):
                    continue

                stage_name: str | None = None
                extra_meta: dict = {}

                # ── 1. run_ledger (primary) ────────────────────────────────
                led = ledger_by_run_id.get(sub.name)
                if isinstance(led, dict):
                    stage_name = _normalize_curriculum_stage_name(led.get("stage_id"))
                    extra_meta = {
                        "stage_pass": led.get("stage_pass"),
                        "phase_label": led.get("phase_label"),
                        "run_order": led.get("run_order"),
                    }

                # ── 2. pipeline_report.json (legacy fallback) ──────────────
                report = _load_json_loose(sub / "pipeline_report.json")
                if isinstance(report, dict):
                    if not stage_name:
                        stage_name = _normalize_curriculum_stage_name(report.get("curriculum_stage") or report.get("stage"))
                if not stage_name:
                    prov = train_ck.get("data_provenance")
                    if isinstance(prov, list) and prov and isinstance(prov[0], dict):
                        stage_name = _normalize_curriculum_stage_name(prov[0].get("stage") or prov[0].get("curriculum_stage"))
                inferred_ref = _infer_stage_ref(train_ck)
                if not stage_name and isinstance(inferred_ref, dict):
                    stage_name = _normalize_curriculum_stage_name(inferred_ref.get("stage"))

                # ── 3. train_token_pack.json -> training_plan.json cross-ref ──
                if not stage_name or stage_name == "unassigned":
                    pack = _load_json_loose(sub / "train_token_pack.json")
                    if isinstance(pack, dict):
                        ds_raw = str(pack.get("dataset") or "").strip()
                        if ds_raw:
                            pack_ds_name = Path(ds_raw).name
                            for plan_cand in (
                                run_dir / "training_plan.json",
                                run_dir / "training_pipeline_latest.json",
                            ):
                                snap = _load_json_loose(plan_cand)
                                if not isinstance(snap, dict):
                                    continue
                                for ps in snap.get("stages") or []:
                                    if not isinstance(ps, dict):
                                        continue
                                    for pds in ps.get("datasets") or []:
                                        if not isinstance(pds, dict):
                                            continue
                                        pds_name = Path(str(pds.get("name") or pds.get("path") or "")).name
                                        if pds_name and pds_name == pack_ds_name:
                                            _resolved = _normalize_curriculum_stage_name(str(ps.get("stage") or ""))
                                            if _resolved:
                                                stage_name = _resolved
                                                break
                                    if stage_name and stage_name != "unassigned":
                                        break
                                if stage_name and stage_name != "unassigned":
                                    break

                if not stage_name:
                    stage_name = "unassigned"

                try:
                    ended_at = datetime.fromtimestamp(train_ck_path.stat().st_mtime, tz=timezone.utc).isoformat()
                except Exception:
                    ended_at = None

                dataset_name = None
                source_path = None
                dataset_provenance = None
                data_source = train_ck.get("data_source") if isinstance(train_ck.get("data_source"), dict) else {}
                if isinstance(data_source.get("dataset_name"), str) and data_source.get("dataset_name", "").strip():
                    dataset_name = data_source.get("dataset_name").strip()
                    dataset_provenance = "train_ck.data_source"
                if isinstance(data_source.get("source_path"), str) and data_source.get("source_path", "").strip():
                    source_path = data_source.get("source_path").strip()
                    dataset_provenance = dataset_provenance or "train_ck.data_source"
                if not source_path and isinstance(data_source.get("source_uri"), str):
                    source_uri = data_source.get("source_uri").strip()
                    if source_uri.startswith("file://") and len(source_uri) > 7:
                        source_path = source_uri[7:]
                        dataset_provenance = dataset_provenance or "train_ck.data_source_uri"

                if isinstance(report, dict):
                    if not dataset_name and isinstance(report.get("dataset_name"), str) and report.get("dataset_name", "").strip():
                        dataset_name = report.get("dataset_name").strip()
                        dataset_provenance = dataset_provenance or "pipeline_report"
                    if not source_path and isinstance(report.get("dataset_path"), str) and report.get("dataset_path", "").strip():
                        source_path = report.get("dataset_path").strip()
                        dataset_provenance = dataset_provenance or "pipeline_report"
                    report_source = report.get("data_source")
                    if isinstance(report_source, dict):
                        if not dataset_name and isinstance(report_source.get("dataset_name"), str) and report_source.get("dataset_name", "").strip():
                            dataset_name = report_source.get("dataset_name").strip()
                            dataset_provenance = dataset_provenance or "pipeline_report.data_source"
                        if not source_path and isinstance(report_source.get("source_path"), str) and report_source.get("source_path", "").strip():
                            source_path = report_source.get("source_path").strip()
                            dataset_provenance = dataset_provenance or "pipeline_report.data_source"

                if isinstance(inferred_ref, dict):
                    if not dataset_name and isinstance(inferred_ref.get("dataset_name"), str) and inferred_ref.get("dataset_name", "").strip():
                        dataset_name = inferred_ref.get("dataset_name").strip()
                        dataset_provenance = dataset_provenance or "stage_refs"
                    if not source_path and isinstance(inferred_ref.get("source_path"), str) and inferred_ref.get("source_path", "").strip():
                        source_path = inferred_ref.get("source_path").strip()
                        dataset_provenance = dataset_provenance or "stage_refs"

                if not source_path:
                    corpus_dir = sub / "corpus"
                    if corpus_dir.exists():
                        corpus_files = sorted([p for p in corpus_dir.iterdir() if p.is_file()])
                        if len(corpus_files) == 1:
                            source_path = str(corpus_files[0])
                            dataset_provenance = dataset_provenance or "ck_pipeline.corpus"
                            if not dataset_name:
                                dataset_name = corpus_files[0].name

                if stage_name == "unassigned":
                    hint_blob = f"{dataset_name or ''} {source_path or ''}".lower()
                    if "stage_a" in hint_blob or "pretrain" in hint_blob:
                        stage_name = "pretrain"
                    elif "stage_b" in hint_blob or "midtrain" in hint_blob:
                        stage_name = "midtrain"

                token_stream_path = None
                raw_source_path = None
                if source_path:
                    base_name = Path(source_path).name.lower()
                    if base_name in {"train_tokens.txt", "tokens.txt"} or "token" in base_name:
                        token_stream_path = source_path
                    else:
                        raw_source_path = source_path
                if token_stream_path and isinstance(inferred_ref, dict):
                    ref_path = inferred_ref.get("source_path")
                    if isinstance(ref_path, str) and ref_path.strip():
                        ref_path_s = ref_path.strip()
                        ref_name = Path(ref_path_s).name.lower()
                        if ref_path_s != token_stream_path and "token" not in ref_name:
                            raw_source_path = ref_path_s
                # Fallback: read train_token_pack.json for the original corpus path
                # when we only have a token stream and inferred_ref didn't resolve it.
                if token_stream_path and not raw_source_path:
                    try:
                        pack = _load_json_loose(sub / "train_token_pack.json")
                        if isinstance(pack, dict):
                            ds_raw = str(pack.get("dataset") or "").strip()
                            if ds_raw:
                                ds_base = Path(ds_raw).name.lower()
                                if ds_base not in {"train_tokens.txt", "tokens.txt"} and "token" not in ds_base:
                                    raw_source_path = ds_raw
                    except Exception:
                        pass
                if not raw_source_path and source_path and not token_stream_path:
                    raw_source_path = source_path

                _ingest(
                    stage_name,
                    train_ck,
                    source="ck_pipeline",
                    path_hint=str(train_ck_path),
                    run_id=sub.name,
                    meta={
                        "seq_len": train_ck.get("seq_len"),
                        "total_tokens": train_ck.get("total_tokens"),
                        "epochs": train_ck.get("epochs"),
                        "lr": train_ck.get("lr"),
                        "grad_accum": train_ck.get("grad_accum"),
                        "ended_at": ended_at,
                        "dataset_name": dataset_name,
                        "source_path": source_path,
                        "raw_source_path": raw_source_path,
                        "token_stream_path": token_stream_path,
                        "dataset_provenance": dataset_provenance,
                        **extra_meta,
                    },
                )

    # Sort: prefer run_order from ledger when available, else fall back to stage+ended_at.
    def _sort_key(item: dict) -> tuple:
        ro = item.get("run_order")
        if isinstance(ro, int):
            return (0, ro, str(item.get("stage") or ""), str(item.get("ended_at") or ""))
        return (1, 0, str(item.get("stage") or ""), str(item.get("ended_at") or ""))

    entries.sort(key=_sort_key)
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        stage = str(entry.get("stage") or "unassigned")
        by_stage.setdefault(stage, []).append(entry)

    return {
        "schema": "ck.stage_loss_history.v1",
        "active_stage": active_stage,
        "entries": entries,
        "by_stage": by_stage,
    }


def _canon_path_str(path_like: str) -> str:
    try:
        return str(Path(path_like).expanduser().resolve())
    except Exception:
        return str(path_like)


def _derive_tokenizer_dataset_coverage(pipeline: dict) -> dict:
    lineage = pipeline.get("tokenizer_lineage") if isinstance(pipeline.get("tokenizer_lineage"), dict) else {}
    provenance = pipeline.get("data_provenance") if isinstance(pipeline.get("data_provenance"), list) else []
    active = provenance[0] if provenance and isinstance(provenance[0], dict) else {}
    active_path = active.get("source_path") if isinstance(active.get("source_path"), str) else ""
    active_hash = None
    if isinstance(active.get("hash"), dict):
        hv = active["hash"].get("value")
        if isinstance(hv, str):
            active_hash = hv

    corpora = lineage.get("tokenizer_corpora")
    if not isinstance(corpora, list) or not corpora:
        reused = bool(lineage.get("reused_run_tokenizer", False))
        if not reused and active:
            corpora = [
                {
                    "dataset_name": active.get("dataset_name"),
                    "source_path": active_path,
                    "stage": active.get("stage"),
                    "curriculum_stage": active.get("curriculum_stage"),
                    "rows": active.get("rows"),
                    "token_count": active.get("token_count"),
                    "byte_size": active.get("byte_size"),
                    "sha256": active_hash,
                    "origin": "derived_current_run",
                }
            ]
        else:
            corpora = []
    if isinstance(lineage, dict) and "tokenizer_corpora" not in lineage and corpora:
        lineage["tokenizer_corpora"] = corpora
        pipeline["tokenizer_lineage"] = lineage

    active_in = False
    active_path_c = _canon_path_str(active_path) if active_path else ""
    active_hash_s = str(active_hash) if isinstance(active_hash, str) else ""
    for row in corpora:
        if not isinstance(row, dict):
            continue
        rh = row.get("sha256")
        rp = row.get("source_path")
        if isinstance(rh, str) and rh and active_hash_s and rh == active_hash_s:
            active_in = True
            break
        if isinstance(rp, str) and rp and active_path_c and _canon_path_str(rp) == active_path_c:
            active_in = True
            break

    reused = bool(lineage.get("reused_run_tokenizer", False))
    if active_in:
        status = "pass"
        note = "active dataset is covered by tokenizer corpus metadata"
    elif reused and corpora:
        status = "warn"
        note = "active dataset is not in tokenizer corpus metadata (tokenizer reused)"
    elif reused and not corpora:
        status = "unknown"
        note = "tokenizer corpus metadata unavailable"
    else:
        status = "pass"
        note = "tokenizer built in current run from active dataset"

    return {
        "status": status,
        "active_dataset_in_corpus": bool(active_in),
        "active_dataset_name": active.get("dataset_name"),
        "active_dataset_path": active_path or None,
        "active_dataset_sha256": active_hash_s or None,
        "note": note,
    }


def _derive_stage_dataset_bindings_from_pipeline(pipeline: dict) -> list[dict]:
    timeline = pipeline.get("stage_timeline") if isinstance(pipeline.get("stage_timeline"), list) else []
    catalog = pipeline.get("dataset_catalog") if isinstance(pipeline.get("dataset_catalog"), list) else []
    lineage = pipeline.get("tokenizer_lineage") if isinstance(pipeline.get("tokenizer_lineage"), dict) else {}
    corpora = lineage.get("tokenizer_corpora") if isinstance(lineage.get("tokenizer_corpora"), list) else []

    tok_path_set = {
        _canon_path_str(str(r.get("source_path")))
        for r in corpora
        if isinstance(r, dict) and isinstance(r.get("source_path"), str) and r.get("source_path")
    }
    tok_hash_set = {
        str(r.get("sha256"))
        for r in corpora
        if isinstance(r, dict) and isinstance(r.get("sha256"), str) and r.get("sha256")
    }

    out: list[dict] = []
    for row in timeline:
        if not isinstance(row, dict):
            continue
        stage = str(row.get("stage") or "")
        if not stage:
            continue
        datasets: list[dict] = []
        for entry in catalog:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("stage") or "") != stage:
                continue
            kind = str(entry.get("kind") or "")
            if kind not in {"active_dataset", "generated_dataset"}:
                continue
            pth = entry.get("path")
            sha = entry.get("sha256")
            in_tok = False
            if isinstance(sha, str) and sha and sha in tok_hash_set:
                in_tok = True
            elif isinstance(pth, str) and pth and _canon_path_str(pth) in tok_path_set:
                in_tok = True
            datasets.append(
                {
                    "name": entry.get("name"),
                    "path": pth,
                    "rows": entry.get("rows"),
                    "kind": kind,
                    "status": entry.get("status"),
                    "source": entry.get("source"),
                    "sha256": sha if isinstance(sha, str) and sha else None,
                    "in_tokenizer_corpus": bool(in_tok),
                }
            )
        rows_total = 0
        rows_known = False
        for ds in datasets:
            rv = ds.get("rows")
            if isinstance(rv, int):
                rows_total += int(rv)
                rows_known = True
        out.append(
            {
                "stage": stage,
                "order": row.get("order"),
                "status": row.get("status"),
                "active": bool(row.get("active") is True),
                "datasets": datasets,
                "dataset_count": len(datasets),
                "rows_total": rows_total if rows_known else None,
                "tokenizer_coverage": {
                    "in_corpus": sum(1 for ds in datasets if ds.get("in_tokenizer_corpus") is True),
                    "not_in_corpus": sum(1 for ds in datasets if ds.get("in_tokenizer_corpus") is not True),
                },
            }
        )
    return out


def _derive_stage_sequence(pipeline: dict) -> dict[str, Any]:
    """
    Build an explicit stage sequence contract for UI ordering.
    This keeps stage flow deterministic and driven by JSON metadata, not hardcoded UI order.
    """
    active_stage = _normalize_curriculum_stage_name(pipeline.get("active_stage")) or "pretrain"
    timeline = pipeline.get("stage_timeline") if isinstance(pipeline.get("stage_timeline"), list) else []
    stage_loss_history = pipeline.get("stage_loss_history") if isinstance(pipeline.get("stage_loss_history"), dict) else {}
    stage_loss_entries = stage_loss_history.get("entries") if isinstance(stage_loss_history.get("entries"), list) else []
    stage_artifacts = pipeline.get("stage_artifacts") if isinstance(pipeline.get("stage_artifacts"), list) else []
    provenance = pipeline.get("data_provenance") if isinstance(pipeline.get("data_provenance"), list) else []
    existing = pipeline.get("stage_sequence") if isinstance(pipeline.get("stage_sequence"), dict) else {}
    existing_entries = existing.get("entries") if isinstance(existing.get("entries"), list) else []

    def _to_int(raw: Any) -> int | None:
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float) and raw.is_integer():
            return int(raw)
        if isinstance(raw, str):
            text = raw.strip()
            if text and re.fullmatch(r"-?\d+", text):
                try:
                    return int(text)
                except Exception:
                    return None
        return None

    rows: dict[str, dict[str, Any]] = {}
    insertion_order = 0
    fallback_order = {"pretrain": 0, "midtrain": 1, "sft": 2, "dpo": 3, "grpo": 4, "ppo": 5}
    status_rank = {"active": 3, "completed": 2, "pass": 2, "planned": 1, "missing": 0}

    def _ingest(
        raw_stage: Any,
        *,
        source: str,
        declared_seq: Any = None,
        status: Any = None,
        active: Any = None,
    ) -> None:
        nonlocal insertion_order
        stage = _normalize_curriculum_stage_name(raw_stage)
        if not stage:
            return

        seq = _to_int(declared_seq)
        if isinstance(seq, int) and seq <= 0:
            seq = None
        status_s = str(status).strip().lower() if isinstance(status, str) else ""
        active_b = bool(active is True or stage == active_stage)

        row = rows.get(stage)
        if row is None:
            row = {
                "stage": stage,
                "declared_seq": seq,
                "status": status_s if status_s else ("active" if active_b else "planned"),
                "active": active_b,
                "sources": [source],
                "_seen": insertion_order,
            }
            insertion_order += 1
            rows[stage] = row
        else:
            if isinstance(seq, int):
                if not isinstance(row.get("declared_seq"), int) or int(seq) < int(row["declared_seq"]):
                    row["declared_seq"] = seq
            if status_s:
                prev = str(row.get("status") or "").lower()
                if status_rank.get(status_s, 0) > status_rank.get(prev, 0):
                    row["status"] = status_s
            if active_b:
                row["active"] = True
                row["status"] = "active"
            if source not in row["sources"]:
                row["sources"].append(source)

    # 1) Existing explicit stage_sequence (if present).
    for idx, row in enumerate(existing_entries):
        if not isinstance(row, dict):
            continue
        seq = row.get("seq")
        if seq is None:
            seq = _to_int(row.get("order"))
            if isinstance(seq, int):
                seq += 1
        if seq is None:
            seq = _to_int(row.get("index"))
            if isinstance(seq, int):
                seq += 1
        if seq is None:
            seq = idx + 1
        _ingest(
            row.get("stage"),
            source="stage_sequence",
            declared_seq=seq,
            status=row.get("status"),
            active=row.get("active"),
        )

    # 2) Stage timeline.
    for idx, row in enumerate(timeline):
        if not isinstance(row, dict):
            continue
        seq = _to_int(row.get("seq"))
        if seq is None:
            order = _to_int(row.get("order"))
            if isinstance(order, int):
                seq = order + 1
        if seq is None:
            index = _to_int(row.get("index"))
            if isinstance(index, int):
                seq = index + 1
        if seq is None:
            seq = idx + 1
        _ingest(
            row.get("stage"),
            source="stage_timeline",
            declared_seq=seq,
            status=row.get("status"),
            active=row.get("active"),
        )

    # 3) Data provenance (active datasets by stage).
    for row in provenance:
        if not isinstance(row, dict):
            continue
        _ingest(
            row.get("stage") or row.get("curriculum_stage"),
            source="data_provenance",
            status=row.get("status"),
        )

    # 4) Stage artifacts.
    for row in stage_artifacts:
        if not isinstance(row, dict):
            continue
        _ingest(
            row.get("stage"),
            source="stage_artifacts",
            status=row.get("status"),
            active=row.get("active"),
        )

    # 5) Stage loss history (often includes legacy/unassigned).
    for row in stage_loss_entries:
        if not isinstance(row, dict):
            continue
        _ingest(
            row.get("stage"),
            source="stage_loss_history",
        )

    # Always include current active stage.
    _ingest(active_stage, source="active_stage", status="active", active=True)

    items = list(rows.values())
    items.sort(
        key=lambda row: (
            int(row["declared_seq"]) if isinstance(row.get("declared_seq"), int) else 10_000 + fallback_order.get(str(row.get("stage") or ""), 100),
            int(row.get("_seen", 0)),
            str(row.get("stage") or ""),
        )
    )

    entries: list[dict[str, Any]] = []
    for idx, row in enumerate(items):
        entry = {
            "stage": str(row.get("stage") or ""),
            "seq": idx + 1,
            "status": str(row.get("status") or ("active" if row.get("active") else "planned")),
            "active": bool(row.get("active") is True),
            "source": ",".join(row.get("sources") or []),
        }
        if isinstance(row.get("declared_seq"), int):
            entry["declared_seq"] = int(row["declared_seq"])
        entries.append(entry)

    return {
        "schema": "ck.stage_sequence.v1",
        "active_stage": active_stage,
        "entries": entries,
    }


def _extract_loss_series(payload, preferred_keys: list[str]) -> list[float]:
    if not isinstance(payload, dict):
        return []
    curve = payload.get("loss_curve")
    if not isinstance(curve, list):
        return []
    out: list[float] = []
    for row in curve:
        if not isinstance(row, dict):
            continue
        val = None
        for k in preferred_keys:
            v = row.get(k)
            if isinstance(v, (int, float)):
                val = float(v)
                break
        if val is not None:
            out.append(val)
    return out


def _find_latest_ascii_bpe_workdir(row_dir: Path) -> Path | None:
    pipeline_root = row_dir / ".ck_pipeline"
    if not pipeline_root.exists() or not pipeline_root.is_dir():
        return None
    workdirs = sorted(
        [p for p in pipeline_root.iterdir() if p.is_dir() and p.name.startswith("ascii_bpe_")],
        key=lambda p: p.name,
    )
    if not workdirs:
        return None
    return workdirs[-1]


def _build_training_canary_summary(run_root: Path | None) -> dict | None:
    if run_root is None:
        return None
    expected = [1, 2]
    rows: list[dict] = []
    any_present = False
    th_max = 1e-4
    th_mean = 5e-5
    th_param = 1e-4

    for idx in expected:
        row_dir = run_root / f"parity_svg_row{idx}"
        row: dict = {
            "row": idx,
            "run_dir": str(row_dir),
            "status": "missing",
            "pass": None,
        }
        if not row_dir.exists() or not row_dir.is_dir():
            row["reason"] = "missing_run_dir"
            rows.append(row)
            continue

        any_present = True
        work_dir = _find_latest_ascii_bpe_workdir(row_dir)
        if work_dir is None:
            row.update({"status": "missing", "reason": "missing_pipeline_workdir"})
            rows.append(row)
            continue

        ck_path = work_dir / "train_ck.json"
        torch_path = work_dir / "train_torch_ref.json"
        row["work_dir"] = str(work_dir)
        row["train_ck_json"] = str(ck_path)
        row["train_torch_ref_json"] = str(torch_path)

        ck = _load_json_loose(ck_path) if ck_path.exists() else None
        pt = _load_json_loose(torch_path) if torch_path.exists() else None
        if not isinstance(ck, dict) or not isinstance(pt, dict):
            row.update({
                "status": "missing",
                "reason": "missing_parity_json",
                "missing_ck": not isinstance(ck, dict),
                "missing_torch_ref": not isinstance(pt, dict),
            })
            rows.append(row)
            continue

        ck_losses = _extract_loss_series(ck, ["loss_ck", "loss"])
        pt_losses = _extract_loss_series(pt, ["loss", "loss_pt", "loss_ck"])
        n = min(len(ck_losses), len(pt_losses))
        if n <= 0:
            row.update({
                "status": "missing",
                "reason": "no_overlapping_loss_curve",
                "ck_steps": len(ck_losses),
                "torch_steps": len(pt_losses),
            })
            rows.append(row)
            continue

        diffs = [abs(ck_losses[i] - pt_losses[i]) for i in range(n)]
        max_abs = max(diffs) if diffs else 0.0
        mean_abs = (sum(diffs) / len(diffs)) if diffs else 0.0
        final_param = float(ck.get("final_param_max_abs_diff", 0.0) or 0.0)

        passed = bool(max_abs <= th_max and mean_abs <= th_mean and final_param <= th_param)
        row.update(
            {
                "status": "pass" if passed else "fail",
                "pass": passed,
                "steps_compared": n,
                "max_abs_loss_diff": max_abs,
                "mean_abs_loss_diff": mean_abs,
                "final_param_max_abs_diff": final_param,
                "thresholds": {
                    "max_abs_loss_diff": th_max,
                    "mean_abs_loss_diff": th_mean,
                    "final_param_max_abs_diff": th_param,
                },
                "loss_ck_first": ck_losses[0],
                "loss_ck_final": ck_losses[n - 1],
                "loss_torch_first": pt_losses[0],
                "loss_torch_final": pt_losses[n - 1],
            }
        )

        post_eval = _load_json_loose(row_dir / "post_train_eval.json")
        if isinstance(post_eval, dict):
            row["post_train_eval"] = {
                "status": post_eval.get("status"),
                "reason": post_eval.get("reason"),
                "valid_svg_rate": post_eval.get("valid_svg_rate"),
                "note": post_eval.get("note"),
            }

        rows.append(row)

    if not rows:
        return None
    if not any_present:
        return None

    compared = [r for r in rows if isinstance(r.get("pass"), bool)]
    pass_count = sum(1 for r in compared if r.get("pass") is True)
    fail_count = sum(1 for r in compared if r.get("pass") is False)
    missing_count = len(rows) - len(compared)
    if fail_count > 0:
        status = "fail"
    elif pass_count == len(expected):
        status = "pass"
    elif pass_count > 0 and missing_count > 0:
        status = "partial"
    else:
        status = "missing"

    return {
        "schema": "ck.training_canary_summary.v1",
        "run_dir": str(run_root),
        "expected_rows": len(expected),
        "rows_present": len([r for r in rows if r.get("reason") != "missing_run_dir"]),
        "rows_compared": len(compared),
        "rows_passed": pass_count,
        "rows_failed": fail_count,
        "rows_missing": missing_count,
        "status": status,
        "rows": rows,
    }


def load_model_data(
    ck_build_path: Path,
    run_dir: Path | None = None,
    strict_run_artifacts: bool = False,
) -> dict:
    """Load all IR/profile/training data for a model or run directory."""
    strict_run_scope = bool(strict_run_artifacts and run_dir is not None)
    if run_dir is not None:
        model_root = run_dir
        model_name = run_dir.name
    else:
        model_name = ck_build_path.parent.name if ck_build_path.name in {"ck_build", ".ck_build"} else ck_build_path.name
        model_root = ck_build_path.parent if ck_build_path.name in {"ck_build", ".ck_build"} else ck_build_path
    train_runtime_available = has_train_runtime_artifacts(run_dir if run_dir is not None else model_root)
    inference_runtime_available = has_inference_runtime_artifacts(run_dir if run_dir is not None else model_root)
    bridge_dirs = _resolve_multimodal_bridge_dirs(run_dir, model_root)
    bridge_root = bridge_dirs.get("root")
    bridge_decoder_root = bridge_dirs.get("decoder")
    bridge_encoder_root = bridge_dirs.get("encoder")

    search_roots: list[Path] = []
    if bridge_decoder_root is not None:
        search_roots.append(bridge_decoder_root)
    if run_dir is not None:
        search_roots.extend([run_dir, run_dir / "ck_build", run_dir / ".ck_build"])
    search_roots.extend([ck_build_path, model_root, model_root / ".ck_build"])
    if bridge_root is not None:
        search_roots.append(bridge_root)
    if bridge_encoder_root is not None:
        search_roots.append(bridge_encoder_root)

    deduped_roots: list[Path] = []
    seen_roots: set[str] = set()
    for root in search_roots:
        key = str(root)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        deduped_roots.append(root)
    search_roots = deduped_roots

    # Include canonical sibling cache roots derived from local GGUF stems.
    # This handles cases like:
    #   run_dir = .../Qwen--Qwen3-0.6B-GGUF
    #   profile outputs in .../Qwen3-0.6B-Q8_0
    profile_alias_roots = _derive_profile_alias_roots(run_dir, model_root)
    for root in profile_alias_roots:
        key = str(root)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        search_roots.append(root)
    runbook_alias_roots = _derive_runbook_alias_roots(run_dir)
    for root in runbook_alias_roots:
        key = str(root)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        search_roots.append(root)

    # Define required vs optional files
    REQUIRED_FILES = [
        "ir1_decode",
        "layout_decode",
        "lowered_decode_call",
    ]
    # Run-directory workflows can be training-only; avoid false missing-required warnings.
    if run_dir is not None:
        REQUIRED_FILES = []
    OPTIONAL_FILES = [
        "ir1_prefill",
        "layout_prefill",
        "lowered_prefill_call",
        "lowered_decode",
        "lowered_prefill",
        "ir1_train",
        "ir2_train",
        "ir_train_invariants",
        "ir2_train_summary",
        "layout_train",
        "layout_train_audit",
        "train_exec_plan",
        "memory_diagnostic",
        "memory_verification",
        "generated_train_runtime_summary",
        "training_loss_curve",
        "training_grad_norms",
        "training_parity",
        "training_parity_regimen",
        "training_parity_xray",
        "training_canary_summary",
        "training_step_profile",
        "training_checkpoint_policy",
        "training_pipeline",
        "corpus_sampling_log",
        "training_logbook",
        "dataset_qc",
        "dataset_profile",
        "tokenizer_roundtrip",
        "post_train_eval",
        "training_epoch_sweep",
        "analysis_checkpoints",
        "train_e2e",
        "run_index",
        "run_config",
        "sanity_overfit",
        "parity_report",
        "profile_latest",
        "contract_report",
        "parity_1token",
        "qk_norm_backward_parity",
        "inference_llama_parity",
        "inference_llama_parity_prefill",
        "inference_llama_parity_decode",
        "inference_llama_autopsy",
        "inference_llama_dump_index",
        "fd_gradients",
        "train_parity_epochs_3",
        "train_parity_epochs_5",
        "train_runtime_parity_realistic",
        "train_runtime_parity_stress",
        "replay_determinism",
        "backprop_stitch_runtime",
        "grad_rules",
        "manifest",
        "kernel_registry",
        "profile_summary",
        "perf_stat_summary",
        "flamegraph_manifest",
        "cachegrind_summary",
        "asan_summary",
        "vtune_summary",
        "advisor_summary",
        "memory_signoff",
        "perf_gate_report",
        "regression_ledger",
        "embedding_dump",
        "multimodal_bridge_report",
        "full_network_graph",
        "bridge_encoder_ir1",
        "bridge_encoder_layout",
        "bridge_encoder_call",
        "bridge_encoder_lowered",
    ]

    def model_candidates(name: str) -> list[Path]:
        return [root / name for root in search_roots]

    data_files = {
        "ir1_decode": model_candidates("ir1_decode.json"),
        "ir1_prefill": model_candidates("ir1_prefill.json"),
        "layout_decode": model_candidates("layout_decode.json"),
        "layout_prefill": model_candidates("layout_prefill.json"),
        "lowered_decode": model_candidates("lowered_decode.json"),
        "lowered_prefill": model_candidates("lowered_prefill.json"),
        "lowered_decode_call": model_candidates("lowered_decode_call.json") + model_candidates("lowered_decode.json"),
        "lowered_prefill_call": model_candidates("lowered_prefill_call.json") + model_candidates("lowered_prefill.json"),
        "ir1_train": model_candidates("ir1_train_forward.json") + model_candidates("ir1_train.json") + [V8_REPORT_PATH / "ir1_train_forward_latest.json", V8_REPORT_PATH_LEGACY / "ir1_train_forward_latest.json"],
        "ir2_train": model_candidates("ir2_train_backward.json") + model_candidates("ir2_train.json") + [V8_REPORT_PATH / "ir2_train_backward_latest.json", V8_REPORT_PATH_LEGACY / "ir2_train_backward_latest.json"],
        "ir_train_invariants": model_candidates("ir_train_invariants.json") + [V8_REPORT_PATH / "ir_train_invariants_latest.json", V8_REPORT_PATH_LEGACY / "ir_train_invariants_latest.json"],
        "ir2_train_summary": model_candidates("ir2_train_summary.json") + [V8_REPORT_PATH / "ir2_train_summary_latest.json", V8_REPORT_PATH_LEGACY / "ir2_train_summary_latest.json"],
        "layout_train": model_candidates("layout_train.json") + model_candidates("layout_train_latest.json") + [V8_REPORT_PATH / "layout_train_latest.json", V8_REPORT_PATH_LEGACY / "layout_train_latest.json"],
        "layout_train_audit": model_candidates("layout_train_audit.json") + model_candidates("layout_train_audit_latest.json") + [V8_REPORT_PATH / "layout_train_audit_latest.json", V8_REPORT_PATH_LEGACY / "layout_train_audit_latest.json"],
        "train_exec_plan": model_candidates("train_exec_plan.json") + model_candidates("train_exec_plan_latest.json") + [V8_REPORT_PATH / "train_exec_plan_latest.json", V8_REPORT_PATH_LEGACY / "train_exec_plan_latest.json"],
        "memory_diagnostic": model_candidates("memory_diagnostic_latest.json") + model_candidates("memory_diagnostic.json") + [V8_REPORT_PATH / "memory_diagnostic_latest.json", V8_REPORT_PATH_LEGACY / "memory_diagnostic_latest.json"],
        "memory_verification": model_candidates("memory_verification_latest.json") + model_candidates("memory_verification.json") + [V8_REPORT_PATH / "memory_verification_latest.json", V8_REPORT_PATH_LEGACY / "memory_verification_latest.json"],
        "generated_train_runtime_summary": model_candidates("generated_train_runtime_summary_v7.json") + model_candidates("generated_train_runtime_summary.json") + [V8_REPORT_PATH / "generated_train_runtime_summary_v7.json", V8_REPORT_PATH_LEGACY / "generated_train_runtime_summary_v7.json"],
        "training_loss_curve": model_candidates("training_loss_curve.json") + model_candidates("training_loss_curve_latest.json") + [V8_REPORT_PATH / "training_loss_curve_latest.json", V8_REPORT_PATH_LEGACY / "training_loss_curve_latest.json"],
        "training_grad_norms": model_candidates("training_grad_norms.json") + model_candidates("training_grad_norms_latest.json") + [V8_REPORT_PATH / "training_grad_norms_latest.json", V8_REPORT_PATH_LEGACY / "training_grad_norms_latest.json"],
        "training_parity": model_candidates("training_parity.json") + model_candidates("training_parity_latest.json") + [V8_REPORT_PATH / "training_parity_latest.json", V8_REPORT_PATH_LEGACY / "training_parity_latest.json"],
        "training_parity_regimen": model_candidates("training_parity_regimen_latest.json") + model_candidates("training_parity_regimen.json") + [V8_REPORT_PATH / "training_parity_regimen_latest.json", V8_REPORT_PATH_LEGACY / "training_parity_regimen_latest.json"],
        "training_parity_xray": model_candidates("regimen_backend_xray.json") + model_candidates("training_parity_xray.json") + [V8_REPORT_PATH / "training_parity_xray_latest.json", V8_REPORT_PATH_LEGACY / "training_parity_xray_latest.json"],
        "training_canary_summary": model_candidates("training_canary_summary.json"),
        "training_step_profile": model_candidates("training_step_profile.json") + model_candidates("training_step_profile_latest.json") + [V8_REPORT_PATH / "training_step_profile_latest.json", V8_REPORT_PATH_LEGACY / "training_step_profile_latest.json"],
        "training_checkpoint_policy": model_candidates("training_checkpoint_policy.json") + model_candidates("training_checkpoint_policy_latest.json") + [V8_REPORT_PATH / "training_checkpoint_policy_latest.json", V8_REPORT_PATH_LEGACY / "training_checkpoint_policy_latest.json"],
        "training_pipeline": model_candidates("training_pipeline_latest.json") + model_candidates("training_pipeline.json") + [V8_REPORT_PATH / "training_pipeline_latest.json", V8_REPORT_PATH_LEGACY / "training_pipeline_latest.json"],
        "training_plan": model_candidates("training_plan.json") + [V8_REPORT_PATH / "training_plan.json", V8_REPORT_PATH_LEGACY / "training_plan.json"],
        "run_ledger": model_candidates("run_ledger.jsonl"),
        "corpus_sampling_log": model_candidates("corpus_sampling_log_latest.json") + model_candidates("corpus_sampling_log.json") + [V8_REPORT_PATH / "corpus_sampling_log_latest.json", V8_REPORT_PATH_LEGACY / "corpus_sampling_log_latest.json"],
        "dataset_qc": model_candidates("dataset_qc.json"),
        "dataset_profile": model_candidates("dataset_profile.json"),
        "tokenizer_roundtrip": model_candidates("tokenizer_roundtrip.json"),
        "post_train_eval": model_candidates("post_train_eval.json"),
        "stage_eval_matrix": model_candidates("stage_eval_matrix.json"),
        "training_epoch_sweep": model_candidates("training_epoch_sweep.json") + model_candidates("training_epoch_sweep_latest.json") + [V8_REPORT_PATH / "training_epoch_sweep_latest.json", V8_REPORT_PATH_LEGACY / "training_epoch_sweep_latest.json"],
        "train_e2e": model_candidates("train_e2e.json") + model_candidates("train_e2e_latest.json") + [V8_REPORT_PATH / "train_e2e_latest.json", V8_REPORT_PATH_LEGACY / "train_e2e_latest.json"],
        "run_index": model_candidates("run_index.json"),
        "run_config": model_candidates("config.json"),
        "sanity_overfit": model_candidates("sanity_overfit.json"),
        "parity_report": model_candidates("parity_report.json"),
        "profile_latest": model_candidates("profile_latest.json"),
        "contract_report": model_candidates("contract_report_latest.json") + [V8_REPORT_PATH / "contract_report_latest.json", V8_REPORT_PATH_LEGACY / "contract_report_latest.json"],
        "parity_1token": model_candidates("parity_1token_latest.json") + [V8_REPORT_PATH / "parity_1token_latest.json", V8_REPORT_PATH_LEGACY / "parity_1token_latest.json"],
        "qk_norm_backward_parity": model_candidates("qk_norm_backward_parity_latest.json") + [V8_REPORT_PATH / "qk_norm_backward_parity_latest.json", V8_REPORT_PATH_LEGACY / "qk_norm_backward_parity_latest.json"],
        "inference_llama_parity": (
            model_candidates("detailed_parity_analysis_latest.json")
            + model_candidates("detailed_parity_analysis.json")
            + model_candidates("llamacpp_parity_latest.json")
            + model_candidates("llamacpp_parity.json")
            + [V8_REPORT_PATH / "detailed_parity_analysis_latest.json", V8_REPORT_PATH_LEGACY / "detailed_parity_analysis_latest.json"]
        ),
        "inference_llama_parity_prefill": (
            model_candidates("detailed_parity_analysis_prefill_latest.json")
            + model_candidates("detailed_parity_analysis_prefill.json")
            + [V8_REPORT_PATH / "detailed_parity_analysis_prefill_latest.json", V8_REPORT_PATH_LEGACY / "detailed_parity_analysis_prefill_latest.json"]
        ),
        "inference_llama_parity_decode": (
            model_candidates("detailed_parity_analysis_decode_latest.json")
            + model_candidates("detailed_parity_analysis_decode.json")
            + [V8_REPORT_PATH / "detailed_parity_analysis_decode_latest.json", V8_REPORT_PATH_LEGACY / "detailed_parity_analysis_decode_latest.json"]
        ),
        "inference_llama_autopsy": (
            model_candidates("parity_autopsy_latest.json")
            + model_candidates("autopsy_report_latest.json")
            + model_candidates("autopsy_report.json")
            + [V8_REPORT_PATH / "parity_autopsy_latest.json", V8_REPORT_PATH_LEGACY / "parity_autopsy_latest.json"]
        ),
        "inference_llama_dump_index": model_candidates("llama_parity_dumps/index.json"),
        "fd_gradients": model_candidates("fd_gradients_latest.json") + [V8_REPORT_PATH / "fd_gradients_latest.json", V8_REPORT_PATH_LEGACY / "fd_gradients_latest.json"],
        "train_parity_epochs_3": model_candidates("train_parity_epochs_3_latest.json") + [V8_REPORT_PATH / "train_parity_epochs_3_latest.json", V8_REPORT_PATH_LEGACY / "train_parity_epochs_3_latest.json"],
        "train_parity_epochs_5": model_candidates("train_parity_epochs_5_latest.json") + [V8_REPORT_PATH / "train_parity_epochs_5_latest.json", V8_REPORT_PATH_LEGACY / "train_parity_epochs_5_latest.json"],
        "train_runtime_parity_realistic": model_candidates("train_runtime_parity_realistic_latest.json") + [V8_REPORT_PATH / "train_runtime_parity_realistic_latest.json", V8_REPORT_PATH_LEGACY / "train_runtime_parity_realistic_latest.json"],
        "train_runtime_parity_stress": model_candidates("train_runtime_parity_stress_latest.json") + [V8_REPORT_PATH / "train_runtime_parity_stress_latest.json", V8_REPORT_PATH_LEGACY / "train_runtime_parity_stress_latest.json"],
        "replay_determinism": model_candidates("replay_determinism_latest.json") + [V8_REPORT_PATH / "replay_determinism_latest.json", V8_REPORT_PATH_LEGACY / "replay_determinism_latest.json"],
        "backprop_stitch_runtime": model_candidates("backprop_stitch_runtime_latest.json") + model_candidates("backprop_stitch_runtime.json") + [V8_REPORT_PATH / "backprop_stitch_runtime_latest.json", V8_REPORT_PATH_LEGACY / "backprop_stitch_runtime_latest.json"],
        "grad_rules": [V8_ROOT / "scripts" / "grad_rules_v8.json"],
        "manifest": model_candidates("weights_manifest.json"),
        "profile_summary": model_candidates("profile_summary.json"),
        "perf_stat_summary": model_candidates("perf_stat_summary.json") + [V8_REPORT_PATH / "perf_stat_summary.json", V8_REPORT_PATH_LEGACY / "perf_stat_summary.json"],
        "flamegraph_manifest": model_candidates("flamegraph_manifest.json") + [V8_REPORT_PATH / "flamegraph_manifest.json", V8_REPORT_PATH_LEGACY / "flamegraph_manifest.json"],
        "cachegrind_summary": model_candidates("cachegrind_summary.json") + [V8_REPORT_PATH / "cachegrind_summary.json", V8_REPORT_PATH_LEGACY / "cachegrind_summary.json"],
        "asan_summary": model_candidates("asan_summary.json") + [V8_REPORT_PATH / "asan_summary.json", V8_REPORT_PATH_LEGACY / "asan_summary.json"],
        "vtune_summary": model_candidates("vtune_summary.json") + [V8_REPORT_PATH / "vtune_summary.json", V8_REPORT_PATH_LEGACY / "vtune_summary.json"],
        "advisor_summary": model_candidates("advisor_summary.json") + [V8_REPORT_PATH / "advisor_summary.json", V8_REPORT_PATH_LEGACY / "advisor_summary.json"],
        "memory_signoff": model_candidates("memory_signoff.json") + [V8_REPORT_PATH / "memory_signoff.json", V8_REPORT_PATH_LEGACY / "memory_signoff.json"],
        "perf_gate_report": model_candidates("perf_gate_report.json") + [V8_REPORT_PATH / "perf_gate_report.json", V8_REPORT_PATH_LEGACY / "perf_gate_report.json"],
        "embedding_dump": model_candidates("embedding_dump.json") + model_candidates("embedding_dump_latest.json"),
        "regression_ledger": (
            model_candidates("regression_ledger.json")
            + [V8_REPORT_PATH / "regression_ledger_latest.json", V8_REPORT_PATH_LEGACY / "regression_ledger_latest.json"]
            + [V8_ROOT / "reports" / "REGRESSION_LEDGER.json"]
        ),
        "kernel_registry": [V8_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"],
        "multimodal_bridge_report": [bridge_root / "bridge_report.json"] if bridge_root is not None else [],
        "full_network_graph": [bridge_root / "full_network_graph.json"] if bridge_root is not None else [],
        "bridge_encoder_ir1": [bridge_encoder_root / "ir1.json"] if bridge_encoder_root is not None else [],
        "bridge_encoder_layout": [bridge_encoder_root / "layout.json"] if bridge_encoder_root is not None else [],
        "bridge_encoder_call": [bridge_encoder_root / "call.json"] if bridge_encoder_root is not None else [],
        "bridge_encoder_lowered": [bridge_encoder_root / "lowered.json"] if bridge_encoder_root is not None else [],
    }

    if strict_run_scope:
        # Keep run reports self-consistent: load artifacts only from the explicit
        # run/model roots (plus derived alias roots), not global latest caches.
        strict_exempt = {"grad_rules", "kernel_registry", "regression_ledger"}
        for key, candidates in list(data_files.items()):
            if key in strict_exempt:
                continue
            data_files[key] = [p for p in candidates if _path_under_any_root(p, search_roots)]

    data = {
        "meta": {
            "model": model_name,
            "path": str(ck_build_path),
            "run_dir": str(run_dir) if run_dir is not None else None,
            "project_root": str(PROJECT_ROOT),
            "has_train_runtime": bool(train_runtime_available),
            "has_inference_runtime": bool(inference_runtime_available),
            "profile_tool_status": collect_profile_tool_status(),
            "host_context": collect_host_context(),
            "strict_run_artifacts": strict_run_scope,
            "warnings": [],
        },
        "files": {}
    }

    loaded = []
    loaded_paths: dict[str, str] = {}
    missing_required = []
    missing_optional = []

    for key, candidates in data_files.items():
        picked_path = None
        picked_payload = None
        fallback_stub_path = None
        fallback_stub_payload = None
        first_error = None

        for path in candidates:
            if not path.exists():
                continue
            try:
                if path.suffix == ".jsonl":
                    # Parse newline-delimited JSON; last record per run_id wins.
                    _by_run: dict[str, dict] = {}
                    for _line in path.read_text(encoding="utf-8").splitlines():
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _rec = json.loads(_line)
                        except Exception:
                            continue
                        if isinstance(_rec, dict) and _rec.get("run_id"):
                            _by_run[str(_rec["run_id"])] = _rec
                    payload = {"entries": sorted(_by_run.values(), key=lambda r: int(r.get("run_order") or 0))}
                else:
                    with open(path, "r") as f:
                        payload = json.load(f)
            except Exception as e:
                if first_error is None:
                    first_error = (path, e)
                continue

            # Prefer renderable profile payloads over marker stubs.
            if key == "profile_summary" and _is_profile_summary_stub(payload):
                if fallback_stub_payload is None:
                    fallback_stub_path = path
                    fallback_stub_payload = payload
                continue

            picked_path = path
            picked_payload = payload
            break

        if picked_payload is None and fallback_stub_payload is not None:
            picked_path = fallback_stub_path
            picked_payload = fallback_stub_payload

        if picked_payload is not None:
            data["files"][key] = picked_payload
            loaded.append(key)
            loaded_paths[key] = str(picked_path)
        else:
            if first_error is not None:
                err_path, err = first_error
                print(f"  ! {key} ({err_path}): {err}")
            if key in REQUIRED_FILES:
                missing_required.append(key)
            else:
                missing_optional.append(key)

    analysis_roots = list(search_roots)
    if not strict_run_scope:
        analysis_roots.extend([V8_REPORT_PATH, V8_REPORT_PATH_LEGACY])
    analysis_payload = collect_analysis_checkpoints(analysis_roots)
    if analysis_payload is not None:
        data["files"]["analysis_checkpoints"] = analysis_payload
        loaded.append("analysis_checkpoints")
    else:
        missing_optional.append("analysis_checkpoints")

    training_logbook_payload = collect_training_logbook(
        search_roots,
        run_dir=run_dir,
        strict_run_scope=strict_run_scope,
    )
    if training_logbook_payload is not None:
        data["files"]["training_logbook"] = training_logbook_payload
        loaded.append("training_logbook")
    else:
        missing_optional.append("training_logbook")

    checkpoint_history_payload = collect_training_checkpoint_history(run_dir if run_dir is not None else model_root)
    if checkpoint_history_payload is not None:
        data["files"]["training_checkpoint_history"] = checkpoint_history_payload
        loaded.append("training_checkpoint_history(derived)")
    else:
        missing_optional.append("training_checkpoint_history")

    if "training_canary_summary" not in data["files"]:
        canary_summary = _build_training_canary_summary(run_dir if run_dir is not None else model_root)
        if isinstance(canary_summary, dict):
            data["files"]["training_canary_summary"] = canary_summary
            loaded.append("training_canary_summary(derived)")
            missing_optional = [k for k in missing_optional if k != "training_canary_summary"]

    # If only runtime parity reports are present, derive dashboard-friendly
    # training_* aliases so the viewer renders without manual file renaming.
    runtime_payload = None
    for key in ("train_runtime_parity_realistic", "train_runtime_parity_stress"):
        candidate = data["files"].get(key)
        if isinstance(candidate, dict):
            runtime_payload = candidate
            break
    if isinstance(runtime_payload, dict):
        files = data["files"]
        if "train_e2e" not in files:
            files["train_e2e"] = runtime_payload
            loaded.append("train_e2e(runtime)")
        if "training_loss_curve" not in files and isinstance(runtime_payload.get("loss_curve"), list):
            files["training_loss_curve"] = {"steps": runtime_payload.get("loss_curve")}
            loaded.append("training_loss_curve(runtime)")
        if "training_parity" not in files and isinstance(runtime_payload.get("parity_steps"), list):
            files["training_parity"] = {"steps": runtime_payload.get("parity_steps")}
            loaded.append("training_parity(runtime)")
        if "training_step_profile" not in files and isinstance(runtime_payload.get("step_profile"), dict):
            files["training_step_profile"] = runtime_payload.get("step_profile")
            loaded.append("training_step_profile(runtime)")
        if "training_grad_norms" not in files and isinstance(runtime_payload.get("grad_norm_series"), dict):
            files["training_grad_norms"] = runtime_payload.get("grad_norm_series")
            loaded.append("training_grad_norms(runtime)")
        if "training_pipeline" not in files:
            active_stage = str(runtime_payload.get("train_mode") or runtime_payload.get("mode") or "pretrain").strip().lower()
            if not active_stage:
                active_stage = "pretrain"
            stages = ["pretrain", "midtrain", "sft", "dpo", "grpo", "ppo"]
            if active_stage not in stages:
                stages = [active_stage] + stages
            active_idx = stages.index(active_stage)
            stage_timeline = []
            for idx, stage in enumerate(stages):
                if idx < active_idx:
                    status = "completed"
                elif idx == active_idx:
                    status = "active"
                else:
                    status = "planned"
                stage_timeline.append({"stage": stage, "order": idx, "status": status, "active": stage == active_stage})
            files["training_pipeline"] = {
                "schema": "ck.training_pipeline.v1",
                "generated_at": runtime_payload.get("generated_at"),
                "active_stage": active_stage,
                "stage_timeline": stage_timeline,
                "backend": runtime_payload.get("backend"),
                "optimizer": {
                    "name": runtime_payload.get("optimizer"),
                    "lr": runtime_payload.get("lr"),
                    "hparams": runtime_payload.get("optimizer_hparams"),
                },
                "execution": {
                    "epochs": runtime_payload.get("epochs"),
                    "steps": runtime_payload.get("steps"),
                    "micro_steps": runtime_payload.get("micro_steps"),
                    "optimizer_steps": runtime_payload.get("optimizer_steps"),
                    "seq_len": runtime_payload.get("seq_len"),
                    "grad_accum": runtime_payload.get("grad_accum"),
                    "tokens_total": runtime_payload.get("total_tokens"),
                    "tokens_per_update": runtime_payload.get("tokens_per_update"),
                },
                "train_dims": runtime_payload.get("train_dims"),
                "data_provenance": runtime_payload.get("data_provenance")
                if isinstance(runtime_payload.get("data_provenance"), list)
                else [],
                "tokenizer_lineage": runtime_payload.get("tokenizer_lineage")
                if isinstance(runtime_payload.get("tokenizer_lineage"), dict)
                else {},
                "sources": {"summary": "runtime_parity", "run_dir": str(run_dir) if run_dir is not None else None},
            }
            loaded.append("training_pipeline(runtime)")
        if "corpus_sampling_log" not in files and isinstance(runtime_payload.get("corpus_sampling_epochs"), list):
            files["corpus_sampling_log"] = {
                "schema": "ck.corpus_sampling_log.v1",
                "run_id": str(runtime_payload.get("run_id") or (run_dir.name if run_dir is not None else "unknown")),
                "updated_at": runtime_payload.get("generated_at"),
                "epochs": runtime_payload.get("corpus_sampling_epochs"),
            }
            loaded.append("corpus_sampling_log(runtime)")

    # Some operators keep decode/prefill parity analyses in split JSON files.
    # Synthesize a combined payload so the viewer can render both together.
    files = data["files"]
    if "inference_llama_parity" not in files:
        parity_prefill = files.get("inference_llama_parity_prefill")
        parity_decode = files.get("inference_llama_parity_decode")
        if isinstance(parity_prefill, dict) or isinstance(parity_decode, dict):
            files["inference_llama_parity"] = {
                "prefill": parity_prefill if isinstance(parity_prefill, dict) else None,
                "decode": parity_decode if isinstance(parity_decode, dict) else None,
                "source": "split_inference_parity",
            }
            loaded.append("inference_llama_parity(split_alias)")

    # Training-only run directories often lack decode/prefill filenames.
    # Add decode/prefill aliases so shared viewer tabs (Memory/Kernel/Stats)
    # still render from training IR/layout/exec-plan artifacts.
    files = data["files"]
    aliased_optional: list[str] = []
    if "ir1_decode" not in files and isinstance(files.get("ir1_train"), dict):
        files["ir1_decode"] = files["ir1_train"]
        loaded.append("ir1_decode(train_alias)")
        aliased_optional.append("ir1_decode")
    if "layout_decode" not in files and isinstance(files.get("layout_train"), dict):
        files["layout_decode"] = files["layout_train"]
        loaded.append("layout_decode(train_alias)")
        aliased_optional.append("layout_decode")
    if "lowered_decode_call" not in files and isinstance(files.get("train_exec_plan"), dict):
        files["lowered_decode_call"] = files["train_exec_plan"]
        loaded.append("lowered_decode_call(train_alias)")
        aliased_optional.append("lowered_decode_call")
    if "lowered_decode" not in files and isinstance(files.get("train_exec_plan"), dict):
        files["lowered_decode"] = files["train_exec_plan"]
        loaded.append("lowered_decode(train_alias)")
        aliased_optional.append("lowered_decode")
    if "ir1_prefill" not in files and isinstance(files.get("ir1_train"), dict):
        files["ir1_prefill"] = files["ir1_train"]
        loaded.append("ir1_prefill(train_alias)")
        aliased_optional.append("ir1_prefill")
    if "layout_prefill" not in files and isinstance(files.get("layout_train"), dict):
        files["layout_prefill"] = files["layout_train"]
        loaded.append("layout_prefill(train_alias)")
        aliased_optional.append("layout_prefill")
    if "lowered_prefill_call" not in files and isinstance(files.get("train_exec_plan"), dict):
        files["lowered_prefill_call"] = files["train_exec_plan"]
        loaded.append("lowered_prefill_call(train_alias)")
        aliased_optional.append("lowered_prefill_call")
    if "lowered_prefill" not in files and isinstance(files.get("train_exec_plan"), dict):
        files["lowered_prefill"] = files["train_exec_plan"]
        loaded.append("lowered_prefill(train_alias)")
        aliased_optional.append("lowered_prefill")
    if aliased_optional:
        missing_optional = [k for k in missing_optional if k not in aliased_optional]

    tokenizer_preview = _build_tokenizer_preview(data["files"], ck_build_path, model_root, run_dir=run_dir)
    if isinstance(tokenizer_preview, dict):
        data["files"]["tokenizer_preview"] = tokenizer_preview
        loaded.append("tokenizer_preview(derived)")

    # Merge stand-alone data-lab artifacts into training_pipeline so the UI can
    # render a single structured view regardless of file-loading path.
    pipeline = files.get("training_pipeline")
    if isinstance(pipeline, dict):
        data_lab = pipeline.get("data_lab")
        if not isinstance(data_lab, dict):
            data_lab = {}
        for key in ("dataset_qc", "dataset_profile", "tokenizer_roundtrip", "post_train_eval", "embedding_dump"):
            payload = files.get(key)
            if isinstance(payload, dict):
                data_lab[key] = payload
        artifacts = data_lab.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
        for src_key, dst_key in (
            ("dataset_qc", "dataset_qc_json"),
            ("dataset_profile", "dataset_profile_json"),
            ("tokenizer_roundtrip", "tokenizer_roundtrip_json"),
            ("post_train_eval", "post_train_eval_json"),
            ("embedding_dump", "embedding_dump_json"),
        ):
            src_path = loaded_paths.get(src_key)
            if isinstance(src_path, str) and src_path:
                artifacts[dst_key] = src_path
        if artifacts:
            data_lab["artifacts"] = artifacts

        if "dataset_path" not in data_lab:
            qc = data_lab.get("dataset_qc")
            if isinstance(qc, dict) and isinstance(qc.get("path"), str):
                data_lab["dataset_path"] = qc.get("path")
        if "dataset_dir" not in data_lab:
            ds_path = data_lab.get("dataset_path")
            if isinstance(ds_path, str) and ds_path:
                data_lab["dataset_dir"] = str(Path(ds_path).parent)

        # Derive Step 0.55 scale-up provenance from dataset_dir manifests so
        # the Data tab can explain baseline vs scaled corpus inputs.
        step_055 = data_lab.get("step_055")
        if not isinstance(step_055, dict):
            step_055 = {}
        existing_sources = step_055.get("sources")
        if not isinstance(existing_sources, list):
            existing_sources = []
        if not existing_sources:
            ds_dir_raw = data_lab.get("dataset_dir")
            if isinstance(ds_dir_raw, str) and ds_dir_raw.strip():
                ds_dir = Path(ds_dir_raw).expanduser()
                manifest_specs = [
                    ("repo_assets_ascii", "Repo SVG assets (ASCII)", ds_dir / "svg_assets_docs_ascii_manifest.json"),
                    ("synthetic_aug", "Synthetic instruction->SVG augmentation", ds_dir / "svg_instruction_aug_manifest.json"),
                    ("repo_assets_utf8", "Repo SVG assets (UTF-8 baseline)", ds_dir / "svg_assets_docs_utf8_manifest.json"),
                    ("ascii_all", "Merged ASCII corpus", ds_dir / "svg_assets_ascii_all_manifest.json"),
                    ("ascii_le4096", "ASCII corpus filtered <=4096 chars", ds_dir / "svg_assets_ascii_le4096_vocab2048_manifest.json"),
                ]
                derived_sources: list[dict] = []
                for source_key, label, manifest_path in manifest_specs:
                    payload = _load_json_loose(manifest_path)
                    if not isinstance(payload, dict):
                        continue
                    rows = payload.get("output_rows")
                    if rows is None:
                        rows = payload.get("num_train")
                    if rows is None:
                        rows = payload.get("num_samples")
                    note_parts: list[str] = []
                    if source_key == "repo_assets_ascii":
                        mapped = payload.get("mapped_common_symbols_total")
                        if isinstance(mapped, int):
                            note_parts.append(f"mapped_symbols={mapped}")
                    if source_key == "synthetic_aug":
                        type_counts = payload.get("type_counts")
                        if isinstance(type_counts, dict):
                            note_parts.append(f"type_families={len(type_counts)}")
                    if source_key == "ascii_le4096":
                        note_parts.append("long rows trimmed for stable seq windows")
                    derived_sources.append(
                        {
                            "name": source_key,
                            "label": label,
                            "rows": rows if isinstance(rows, int) else rows,
                            "path": str(manifest_path),
                            "note": ", ".join(note_parts) if note_parts else "",
                        }
                    )
                if derived_sources:
                    step_055["detected"] = True
                    step_055["dataset_dir"] = str(ds_dir)
                    step_055["sources"] = derived_sources
                    loaded.append("step_055(derived)")
        if step_055:
            data_lab["step_055"] = step_055

        # Derive Step 0.56 bridge-pack metadata (optional Stage-A syntax bridge).
        step_056 = data_lab.get("step_056")
        if not isinstance(step_056, dict):
            step_056 = {}
        if not step_056.get("manifest_path"):
            ds_dir_raw = data_lab.get("dataset_dir")
            if isinstance(ds_dir_raw, str) and ds_dir_raw.strip():
                ds_dir = Path(ds_dir_raw).expanduser()
                bridge_candidates: list[Path] = [
                    ds_dir / "svg_stage_a_bridge_small_manifest.json",
                    ds_dir / "svg_stage_a_bridge_pack_manifest.json",
                ]
                bridge_candidates.extend(sorted(ds_dir.glob("svg_stage_a_bridge*_manifest.json")))
                seen_candidates: set[str] = set()
                for cand in bridge_candidates:
                    cand_key = str(cand.resolve()) if cand.exists() else str(cand)
                    if cand_key in seen_candidates:
                        continue
                    seen_candidates.add(cand_key)
                    payload = _load_json_loose(cand)
                    if not isinstance(payload, dict):
                        continue
                    bridge_rows = payload.get("bridge_rows")
                    if bridge_rows is None:
                        bridge_rows = payload.get("num_rows")
                    missing_before = payload.get("missing_features_from_stage_a")
                    if not isinstance(missing_before, list):
                        missing_before = []
                    missing_after = payload.get("missing_features_after_bridge_only")
                    if not isinstance(missing_after, list):
                        missing_after = []
                    step_056 = {
                        "detected": True,
                        "dataset_dir": str(ds_dir),
                        "manifest_path": str(cand),
                        "stage_a_path": payload.get("stage_a_path"),
                        "stage_b_path": payload.get("stage_b_path"),
                        "out_path": payload.get("out_path"),
                        "bridge_rows": bridge_rows,
                        "missing_features_from_stage_a": missing_before,
                        "missing_features_after_bridge_only": missing_after,
                    }
                    loaded.append("step_056(derived)")
                    break
        if step_056:
            data_lab["step_056"] = step_056

        dataset_catalog = pipeline.get("dataset_catalog")
        derived_catalog = _derive_dataset_catalog(pipeline, run_dir)
        if not isinstance(dataset_catalog, list) or not dataset_catalog:
            if derived_catalog:
                pipeline["dataset_catalog"] = derived_catalog
                loaded.append("dataset_catalog(derived)")
                dataset_catalog = derived_catalog
        elif derived_catalog:
            # Merge existing + derived to keep historical rows while repairing stale manifest paths.
            existing_rows = [r for r in dataset_catalog if isinstance(r, dict)]
            merged_rows: list[dict] = [dict(r) for r in existing_rows]
            index: dict[tuple[str, str, str], int] = {}
            for i, row in enumerate(merged_rows):
                key = (
                    str(row.get("stage") or ""),
                    str(row.get("kind") or ""),
                    str(row.get("name") or ""),
                )
                if any(key):
                    index[key] = i
            changed = False
            for row in derived_catalog:
                if not isinstance(row, dict):
                    continue
                key = (
                    str(row.get("stage") or ""),
                    str(row.get("kind") or ""),
                    str(row.get("name") or ""),
                )
                if key in index:
                    cur = merged_rows[index[key]]
                    cur_path = str(cur.get("path") or "")
                    new_path = str(row.get("path") or "")
                    cur_exists = bool(cur_path and Path(cur_path).expanduser().exists())
                    new_exists = bool(new_path and Path(new_path).expanduser().exists())
                    if new_path and (not cur_path or (not cur_exists and new_exists)) and new_path != cur_path:
                        cur["path"] = new_path
                        changed = True
                    if row.get("note") and row.get("note") != cur.get("note"):
                        cur["note"] = row.get("note")
                        changed = True
                else:
                    merged_rows.append(dict(row))
                    changed = True
            if changed:
                pipeline["dataset_catalog"] = merged_rows
                loaded.append("dataset_catalog(merged)")
                dataset_catalog = merged_rows

        stage_artifacts = pipeline.get("stage_artifacts")
        if not isinstance(stage_artifacts, list) or not stage_artifacts:
            derived_stage_artifacts = _derive_stage_artifacts(pipeline, loaded_paths, run_dir)
            if derived_stage_artifacts:
                pipeline["stage_artifacts"] = derived_stage_artifacts
                loaded.append("stage_artifacts(derived)")

        if not isinstance(pipeline.get("tokenizer_dataset_coverage"), dict):
            derived_tok_cov = _derive_tokenizer_dataset_coverage(pipeline)
            pipeline["tokenizer_dataset_coverage"] = derived_tok_cov
            loaded.append("tokenizer_dataset_coverage(derived)")

        stage_bindings = pipeline.get("stage_dataset_bindings")
        if not isinstance(stage_bindings, list) or not stage_bindings:
            derived_stage_bindings = _derive_stage_dataset_bindings_from_pipeline(pipeline)
            if derived_stage_bindings:
                pipeline["stage_dataset_bindings"] = derived_stage_bindings
                loaded.append("stage_dataset_bindings(derived)")

        stage_loss_history = pipeline.get("stage_loss_history")
        derived_stage_loss_history = _derive_stage_loss_history(
            pipeline,
            files,
            loaded_paths,
            run_dir,
        )
        if isinstance(derived_stage_loss_history, dict) and isinstance(derived_stage_loss_history.get("entries"), list):
            if not isinstance(stage_loss_history, dict) or not isinstance(stage_loss_history.get("entries"), list):
                if derived_stage_loss_history.get("entries"):
                    pipeline["stage_loss_history"] = derived_stage_loss_history
                    loaded.append("stage_loss_history(derived)")
            else:
                existing_entries = stage_loss_history.get("entries") if isinstance(stage_loss_history.get("entries"), list) else []
                # Build set of run_ids that the derived scan resolved — derived stage
                # attribution is always authoritative over stale JSON entries (which can
                # have wrong stage labels, e.g. 171735 mis-labeled "pretrain").
                _derived_run_ids: set[str] = {
                    str(r.get("run_id") or r.get("path") or "")
                    for r in derived_stage_loss_history.get("entries", [])
                    if isinstance(r, dict) and (r.get("run_id") or r.get("path"))
                }
                merged_map: dict[tuple[str, str], dict[str, Any]] = {}
                for row in existing_entries:
                    if not isinstance(row, dict):
                        continue
                    row_run_id = str(row.get("run_id") or row.get("path") or "")
                    # Skip stale JSON entries whose run_id is covered by derived scan.
                    if row_run_id and row_run_id in _derived_run_ids:
                        continue
                    key = (str(row.get("stage") or ""), row_run_id)
                    merged_map[key] = dict(row)
                for row in derived_stage_loss_history.get("entries", []):
                    if not isinstance(row, dict):
                        continue
                    key = (str(row.get("stage") or ""), str(row.get("run_id") or row.get("path") or ""))
                    cur = merged_map.get(key, {})
                    authoritative_keys = {
                        "dataset_name",
                        "source_path",
                        "raw_source_path",
                        "token_stream_path",
                        "dataset_provenance",
                    }
                    for k, v in row.items():
                        if k in authoritative_keys:
                            cur[k] = v
                            continue
                        if v is None:
                            continue
                        if k not in cur or cur.get(k) in (None, "", [], {}):
                            cur[k] = v
                    if row.get("token_stream_path") and not row.get("raw_source_path"):
                        # Keep token-stream-only runs explicit: don't let stale merged history
                        # backfill raw_source_path with the token stream path.
                        cur["raw_source_path"] = None
                    merged_map[key] = cur
                merged_entries = list(merged_map.values())
                merged_entries.sort(key=lambda item: (str(item.get("stage") or ""), str(item.get("ended_at") or ""), str(item.get("run_id") or "")))
                merged_by_stage: dict[str, list[dict[str, Any]]] = {}
                for item in merged_entries:
                    st = str(item.get("stage") or "unassigned")
                    merged_by_stage.setdefault(st, []).append(item)
                merged_history = {
                    "schema": str(stage_loss_history.get("schema") or derived_stage_loss_history.get("schema") or "ck.stage_loss_history.v1"),
                    "active_stage": stage_loss_history.get("active_stage") or derived_stage_loss_history.get("active_stage"),
                    "entries": merged_entries,
                    "by_stage": merged_by_stage,
                }
                if merged_history != stage_loss_history:
                    pipeline["stage_loss_history"] = merged_history
                    loaded.append("stage_loss_history(merged)")

        stage_sequence = _derive_stage_sequence(pipeline)
        if isinstance(stage_sequence, dict) and isinstance(stage_sequence.get("entries"), list) and stage_sequence.get("entries"):
            prev_stage_sequence = pipeline.get("stage_sequence")
            pipeline["stage_sequence"] = stage_sequence
            if not isinstance(prev_stage_sequence, dict) or prev_stage_sequence != stage_sequence:
                loaded.append("stage_sequence(derived)")

        run_sequence = pipeline.get("run_sequence")
        if not isinstance(run_sequence, list) or not run_sequence:
            data_lab_local = pipeline.get("data_lab") if isinstance(pipeline.get("data_lab"), dict) else {}
            tok_cov = pipeline.get("tokenizer_dataset_coverage") if isinstance(pipeline.get("tokenizer_dataset_coverage"), dict) else {}
            tok_round = data_lab_local.get("tokenizer_roundtrip") if isinstance(data_lab_local.get("tokenizer_roundtrip"), dict) else {}
            exec_meta = pipeline.get("execution") if isinstance(pipeline.get("execution"), dict) else {}
            post_eval = data_lab_local.get("post_train_eval") if isinstance(data_lab_local.get("post_train_eval"), dict) else {}
            steps = exec_meta.get("steps") if isinstance(exec_meta.get("steps"), (int, float)) else 0
            pipeline["run_sequence"] = [
                {"order": 1, "op": "dataset_qc", "status": "done" if data_lab_local.get("dataset_qc") else "unknown"},
                {"order": 2, "op": "tokenizer_build_or_reuse", "status": "reused" if tok_cov.get("status") == "warn" else "built_or_unknown"},
                {"order": 3, "op": "tokenizer_roundtrip", "status": str(tok_round.get("status") or "unknown")},
                {"order": 4, "op": "train", "status": "done" if int(steps) > 0 else "missing"},
                {"order": 5, "op": "post_train_eval", "status": "done" if post_eval else "skipped"},
            ]
            loaded.append("run_sequence(derived)")

        if "tokenizer_json_path" not in data_lab:
            tok = pipeline.get("tokenizer_lineage")
            if isinstance(tok, dict) and isinstance(tok.get("tokenizer_path"), str):
                data_lab["tokenizer_json_path"] = tok.get("tokenizer_path")
        if isinstance(tokenizer_preview, dict) and tokenizer_preview:
            data_lab["tokenizer_preview"] = tokenizer_preview

        pipeline["data_lab"] = data_lab

    # Enrich artifacts for standalone report portability.
    flame = data["files"].get("flamegraph_manifest")
    if isinstance(flame, dict):
        def _enrich_flame_entry(entry: dict) -> None:
            svg_path = entry.get("svg_path")
            if not isinstance(svg_path, str):
                return
            resolved = _resolve_asset_path(svg_path, ck_build_path, model_root)
            if resolved and resolved.suffix.lower() == ".svg":
                try:
                    entry["svg_inline"] = resolved.read_text(errors="ignore")
                    entry["svg_resolved_path"] = str(resolved)
                except Exception as e:
                    data["meta"]["warnings"].append(f"Failed to inline flamegraph SVG: {e}")

        # Legacy compatibility: old manifests only had decode-level fields at top-level.
        # Synthesize mode-aware entries so Profile dropdown can switch decode/prefill.
        by_mode = flame.get("by_mode")
        if not isinstance(by_mode, dict) or len(by_mode) == 0:
            decode_entry: dict[str, Any] = {}
            for key in ("svg_path", "perf_data_path", "folded_path", "top_symbols", "generated_at"):
                if key in flame:
                    decode_entry[key] = flame.get(key)

            synthesized: dict[str, dict[str, Any]] = {}
            if decode_entry:
                synthesized["decode"] = dict(decode_entry)

            decode_svg = decode_entry.get("svg_path")
            decode_perf = decode_entry.get("perf_data_path")
            decode_folded = decode_entry.get("folded_path")

            def _replace_suffix(raw: Any, old: str, new: str) -> str | None:
                if not isinstance(raw, str):
                    return None
                return raw.replace(old, new)

            prefill_entry: dict[str, Any] = {}
            prefill_svg_path = (
                flame.get("prefill_svg_path")
                if isinstance(flame.get("prefill_svg_path"), str)
                else (
                    _replace_suffix(decode_svg, "flamegraph_v8.svg", "flamegraph_v8_prefill.svg")
                    or _replace_suffix(decode_svg, "flamegraph_v7.svg", "flamegraph_v7_prefill.svg")
                )
            )
            prefill_perf_path = (
                flame.get("prefill_perf_data_path")
                if isinstance(flame.get("prefill_perf_data_path"), str)
                else (
                    _replace_suffix(decode_perf, "ck_v8_perf.data", "ck_v8_perf_prefill.data")
                    or _replace_suffix(decode_perf, "ck_v7_perf.data", "ck_v7_perf_prefill.data")
                )
            )
            prefill_folded_path = (
                flame.get("prefill_folded_path")
                if isinstance(flame.get("prefill_folded_path"), str)
                else (
                    _replace_suffix(decode_folded, "ck_v8_perf.folded", "ck_v8_perf_prefill.folded")
                    or _replace_suffix(decode_folded, "ck_v7_perf.folded", "ck_v7_perf_prefill.folded")
                )
            )
            prefill_top_symbols = flame.get("prefill_top_symbols")

            if isinstance(prefill_svg_path, str) and prefill_svg_path:
                prefill_entry["svg_path"] = prefill_svg_path
            if isinstance(prefill_perf_path, str) and prefill_perf_path:
                prefill_entry["perf_data_path"] = prefill_perf_path
            if isinstance(prefill_folded_path, str) and prefill_folded_path:
                prefill_entry["folded_path"] = prefill_folded_path
            if isinstance(prefill_top_symbols, list):
                prefill_entry["top_symbols"] = prefill_top_symbols

            if prefill_entry.get("svg_path"):
                resolved_prefill = _resolve_asset_path(str(prefill_entry["svg_path"]), ck_build_path, model_root)
                if resolved_prefill:
                    synthesized["prefill"] = prefill_entry

            if synthesized:
                for payload in synthesized.values():
                    _enrich_flame_entry(payload)
                flame["by_mode"] = synthesized
                flame["available_modes"] = sorted(synthesized.keys())
                preferred_mode = "decode" if "decode" in synthesized else next(iter(synthesized.keys()))
                preferred_entry = synthesized.get(preferred_mode, {})
                if isinstance(preferred_entry, dict):
                    for key in ("svg_path", "perf_data_path", "folded_path", "top_symbols", "svg_inline", "svg_resolved_path"):
                        if key in preferred_entry:
                            flame[key] = preferred_entry[key]

        _enrich_flame_entry(flame)
        by_mode = flame.get("by_mode")
        if isinstance(by_mode, dict):
            mode_keys: list[str] = []
            for mode_name, payload in by_mode.items():
                if not isinstance(mode_name, str) or not isinstance(payload, dict):
                    continue
                mode_keys.append(mode_name)
                _enrich_flame_entry(payload)
            if mode_keys:
                flame["available_modes"] = sorted(set(mode_keys))

    vtune = data["files"].get("vtune_summary")
    if isinstance(vtune, dict):
        path_keys = [
            "result_dir",
            "svg_path",
            "png_path",
            "image_path",
            "report_path",
            "csv_path",
            "hotspots_svg",
            "hotspots_png",
        ]
        for key in path_keys:
            raw = vtune.get(key)
            if not isinstance(raw, str):
                continue
            resolved = _resolve_asset_path(raw, ck_build_path, model_root)
            if not resolved:
                continue
            vtune[f"{key}_resolved"] = str(resolved)
            if resolved.suffix.lower() == ".svg":
                try:
                    vtune[f"{key}_inline"] = resolved.read_text(errors="ignore")
                except Exception:
                    pass
            else:
                data_uri = _encode_image_data_uri(resolved)
                if data_uri:
                    vtune[f"{key}_data_uri"] = data_uri

        artifacts = vtune.get("artifacts")
        if isinstance(artifacts, list):
            enriched = []
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                item2 = dict(item)
                raw = item2.get("path")
                if isinstance(raw, str):
                    resolved = _resolve_asset_path(raw, ck_build_path, model_root)
                    if resolved:
                        item2["resolved_path"] = str(resolved)
                        if resolved.suffix.lower() == ".svg":
                            try:
                                item2["inline"] = resolved.read_text(errors="ignore")
                            except Exception:
                                pass
                        else:
                            data_uri = _encode_image_data_uri(resolved)
                            if data_uri:
                                item2["data_uri"] = data_uri
                enriched.append(item2)
            vtune["artifacts"] = enriched

        analyses = vtune.get("analyses")
        if isinstance(analyses, list):
            enriched_analyses = []
            for entry in analyses:
                if not isinstance(entry, dict):
                    continue
                entry2 = dict(entry)
                for key in ("result_dir", "report_text", "report_csv"):
                    raw = entry2.get(key)
                    if not isinstance(raw, str):
                        continue
                    resolved = _resolve_asset_path(raw, ck_build_path, model_root)
                    if resolved:
                        entry2[f"{key}_resolved"] = str(resolved)
                enriched_analyses.append(entry2)
            vtune["analyses"] = enriched_analyses

    advisor = data["files"].get("advisor_summary")
    if isinstance(advisor, dict):
        for key in ("project_dir", "project_path", "report_path", "csv_path", "html_path"):
            raw = advisor.get(key)
            if not isinstance(raw, str):
                continue
            resolved = _resolve_asset_path(raw, ck_build_path, model_root)
            if resolved:
                advisor[f"{key}_resolved"] = str(resolved)
        artifacts = advisor.get("artifacts")
        if isinstance(artifacts, list):
            enriched = []
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                item2 = dict(item)
                raw = item2.get("path")
                if isinstance(raw, str):
                    resolved = _resolve_asset_path(raw, ck_build_path, model_root)
                    if resolved:
                        item2["resolved_path"] = str(resolved)
                enriched.append(item2)
            advisor["artifacts"] = enriched

    cachegrind = data["files"].get("cachegrind_summary")
    if isinstance(cachegrind, dict):
        for key in ("cachegrind_out", "annotate_path"):
            raw = cachegrind.get(key)
            if not isinstance(raw, str):
                continue
            resolved = _resolve_asset_path(raw, ck_build_path, model_root)
            if resolved:
                cachegrind[f"{key}_resolved"] = str(resolved)

    asan = data["files"].get("asan_summary")
    if isinstance(asan, dict):
        for key in ("verify_report_path", "memory_diagnostic_path"):
            raw = asan.get(key)
            if not isinstance(raw, str):
                continue
            resolved = _resolve_asset_path(raw, ck_build_path, model_root)
            if resolved:
                asan[f"{key}_resolved"] = str(resolved)

    mem = data["files"].get("memory_signoff")
    if isinstance(mem, dict):
        data["files"]["memory_signoff"] = _trim_memory_signoff_payload(mem)

    call_ir_health = _collect_call_ir_health(data["files"])
    if isinstance(call_ir_health, dict):
        data["files"]["call_ir_health"] = call_ir_health
        loaded.append("call_ir_health(derived)")
        totals = call_ir_health.get("totals", {})
        total_err = int(totals.get("error_count") or 0)
        total_warn = int(totals.get("warning_count") or 0)
        if total_err > 0 or total_warn > 0:
            decode = call_ir_health.get("modes", {}).get("decode", {})
            prefill = call_ir_health.get("modes", {}).get("prefill", {})
            de = int(decode.get("error_count") or 0)
            dw = int(decode.get("warning_count") or 0)
            pe = int(prefill.get("error_count") or 0)
            pw = int(prefill.get("warning_count") or 0)
            data["meta"]["warnings"].append(
                f"ERROR: Call-IR issues detected. decode(e={de}, w={dw}) prefill(e={pe}, w={pw})."
            )
            for mode_name, mode_payload in (("decode", decode), ("prefill", prefill)):
                sample_errs = mode_payload.get("sample_errors")
                if isinstance(sample_errs, list) and sample_errs:
                    s0 = sample_errs[0]
                    data["meta"]["warnings"].append(
                        "ERROR: "
                        f"{mode_name} op[{s0.get('idx')}:{s0.get('op')}] "
                        f"{s0.get('message')}"
                    )

    # Report missing files
    if missing_required:
        print(f"  ! Missing required files: {missing_required}")
        data["meta"]["warnings"].append(f"Missing required: {missing_required}")

    if missing_optional:
        print(f"  - Missing optional files: {missing_optional}")

    if loaded_paths:
        data["meta"]["loaded_paths"] = loaded_paths
    if profile_alias_roots:
        data["meta"]["profile_roots"] = [str(p) for p in profile_alias_roots]
    if strict_run_scope:
        data["meta"]["artifact_roots"] = [str(p) for p in search_roots]
    if bridge_root is not None:
        data["meta"]["multimodal_bridge_root"] = str(bridge_root)
    if bridge_decoder_root is not None:
        data["meta"]["multimodal_decoder_root"] = str(bridge_decoder_root)
    if bridge_encoder_root is not None:
        data["meta"]["multimodal_encoder_root"] = str(bridge_encoder_root)

    bridge_payload = data["files"].get("multimodal_bridge_report")
    if isinstance(bridge_payload, dict):
        enriched_bridge = _enrich_multimodal_bridge_payload(
            bridge_payload,
            ck_build_path=ck_build_path,
            model_root=model_root,
        )
        data["files"]["multimodal_bridge_report"] = enriched_bridge
        data["meta"]["multimodal_mode"] = str(enriched_bridge.get("bridge_mode") or "encoder_decoder")

    print(f"  Loaded {len(loaded)} files")
    return data


def generate_html_report(
    ck_build_path: Path,
    output_path: Path = None,
    run_dir: Path | None = None,
    strict_run_artifacts: bool = False,
):
    """Generate standalone HTML report."""
    from datetime import datetime

    model_name = ck_build_path.parent.name if ck_build_path.name in {"ck_build", ".ck_build"} else ck_build_path.name
    print(f"Generating report for: {model_name}")

    # Load data
    data = load_model_data(
        ck_build_path,
        run_dir=run_dir,
        strict_run_artifacts=strict_run_artifacts,
    )
    data["meta"]["generated_at"] = datetime.now().isoformat()
    data["meta"]["engine_version"] = "v8"

    # Read visualizer template
    if not VISUALIZER.exists():
        raise FileNotFoundError(f"Visualizer not found: {VISUALIZER}")

    with open(VISUALIZER, "r") as f:
        html = f.read()

    # Embed data safely for inline <script> context.
    # Prevent accidental </script> termination from embedded payloads (e.g., inline SVG/JS).
    data_json = json.dumps(data)
    data_json = data_json.replace("</", "<\\/")
    data_json = data_json.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

    # Embed data
    data_js = (
        f"window.EMBEDDED_IR_DATA = {data_json};"
        "window.dispatchEvent(new Event('ckEmbeddedDataLoaded'));"
        "if (window.bootstrapFromEmbeddedData) { window.bootstrapFromEmbeddedData(); }"
    )
    html = html.replace('</body>', f'<script>{data_js}</script></body>')

    # Update title
    html = html.replace(
        '<title>IR Visualizer | C-Kernel-Engine</title>',
        f'<title>IR Visualizer | {html_mod.escape(model_name)} | C-Kernel-Engine</title>'
    )

    # Write output
    if output_path is None:
        output_path = ck_build_path / "ir_report.html"

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def serve_live(run_dir: Path, html_path: Path, port: int = 7700, interval_ms: int = 5000) -> None:
    """
    Start a lightweight HTTP server for live training monitoring.

    Serves ir_report.html at GET / and a JSON snapshot of all small training
    metric files at GET /api/snapshot.  The visualizer JS auto-enables polling
    when served over HTTP (or when window.CK_LIVE_MODE is present).

    Tip: this function is optional.  A plain static server also works:
        python3 -m http.server 7700 -d <run_dir>
    then open  http://localhost:7700/ir_report.html  and the visualizer will
    auto-detect HTTP and start polling the JSON artifact files directly.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse
    import mimetypes

    LIVE_FILES = [
        "training_pipeline_latest.json",
        "training_loss_curve_latest.json",
        "training_grad_norms_latest.json",
        "training_parity_latest.json",
        "training_parity_regimen_latest.json",
        "training_step_profile_latest.json",
        "training_checkpoint_policy_latest.json",
        "corpus_sampling_log_latest.json",
        "run_index.json",
    ]

    # Inject window.CK_LIVE_MODE before </body> so the visualizer uses the
    # /api/snapshot endpoint (one request per cycle instead of N file fetches).
    with open(html_path, "r", encoding="utf-8") as f:
        base_html = f.read()
    live_cfg = json.dumps({"pollUrl": "/api/snapshot", "fsUrl": "/api/fs", "intervalMs": interval_ms})
    live_inject = f"<script>window.CK_LIVE_MODE = {live_cfg};</script>"
    served_html = base_html.replace("</body>", live_inject + "</body>")
    served_html_bytes = served_html.encode("utf-8")

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Only print errors; suppress per-request success noise
            if len(args) >= 2 and str(args[1]).startswith(("4", "5")):
                print(f"[live] {args[0]} -> {args[1]}")

        def _cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html", "/ir_report.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(served_html_bytes)))
                self._cors_headers()
                self.end_headers()
                self.wfile.write(served_html_bytes)

            elif path == "/api/snapshot":
                snapshot: dict = {}
                for fname in LIVE_FILES:
                    fpath = run_dir / fname
                    if not fpath.exists():
                        continue
                    try:
                        stat = fpath.stat()
                        content = json.loads(fpath.read_text(encoding="utf-8"))
                        snapshot[fname] = {"mtime": stat.st_mtime, "size": stat.st_size, "content": content}
                    except Exception:
                        pass
                body = json.dumps(snapshot).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self._cors_headers()
                self.end_headers()
                self.wfile.write(body)

            elif path == "/api/fs":
                params = parse_qs(parsed.query or "")
                raw_path = params.get("path", [""])[0]
                if not isinstance(raw_path, str) or not raw_path.strip():
                    self.send_error(400, "missing path")
                    return
                req_path = Path(raw_path).expanduser()
                resolved = _resolve_path_loose(req_path)
                allowed_roots = [
                    _resolve_path_loose(run_dir),
                    _resolve_path_loose(V8_ROOT),
                    _resolve_path_loose(PROJECT_ROOT),
                ]
                if not _path_under_any_root(resolved, allowed_roots):
                    self.send_error(403, "path outside allowed roots")
                    return
                if not resolved.exists() or not resolved.is_file():
                    self.send_error(404, "file not found")
                    return
                try:
                    body = resolved.read_bytes()
                except Exception:
                    self.send_error(500, "failed to read file")
                    return
                mime, _ = mimetypes.guess_type(str(resolved))
                if not mime:
                    mime = "text/plain; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self._cors_headers()
                self.end_headers()
                self.wfile.write(body)

            else:
                self.send_error(404)

    server = HTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"[live] Serving at {url}")
    print(f"[live] Polling artifacts from: {run_dir}")
    print(f"[live] Poll interval: {interval_ms}ms  |  Ctrl+C to stop")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[live] Stopped.")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(
        description="IR Visualizer Launcher for C-Kernel-Engine v8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 version/v8/tools/open_ir_visualizer_v8.py              # Open visualizer
    python3 version/v8/tools/open_ir_visualizer_v8.py --list       # List available models
    python3 version/v8/tools/open_ir_visualizer_v8.py gemma3       # Generate and open report
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate gemma3  # Generate full report (profile+probes)
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate gemma3 --interactive
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate gemma3 --html-only  # Generate HTML only
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate gemma3 --with-profile
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate gemma3 --with-probes --force-compile
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate Qwen--Qwen3-0.6B --with-probes --run-model Qwen/Qwen3-0.6B --weight-dtype float32
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate gemma3 --with-probes --perf-runtime cli
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate gemma3 --no-vtune
    python3 version/v8/tools/open_ir_visualizer_v8.py --generate gemma3 --with-probes --run-model hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf --chat-template none
        """
    )

    parser.add_argument(
        "model",
        nargs="?",
        help="Model name (e.g., gemma3) or path to ck_build directory"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models in cache"
    )
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Generate HTML report without opening browser"
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Generate HTML only (skip profile/probe capture)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for generated HTML (default: ck_build/ir_report.html)"
    )
    parser.add_argument(
        "--run",
        type=Path,
        help="Run directory produced by cks-v8-run (single source of training/profile artifacts)"
    )
    parser.add_argument(
        "--run-model",
        type=str,
        help="Explicit model input for runtime probes/profile (hf://..., .gguf, or source checkpoint dir)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Prompt for runtime/probe options after model selection"
    )
    parser.add_argument(
        "--with-profile", "--profile",
        dest="with_profile",
        action="store_true",
        help="Run the stable runtime prep path with profiling before generating the report"
    )
    parser.add_argument(
        "--with-probes", "--probes",
        dest="with_probes",
        action="store_true",
        help="Run memory sign-off + perf probes (perf stat/flamegraph + perf budget gate) before report generation"
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=1024,
        help="Context length for --with-profile/--with-probes runs (default: 1024)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Max tokens for --with-profile/--with-probes runs (default: 16)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Prompt text for --with-profile/--with-probes runs"
    )
    parser.add_argument(
        "--force-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force compile runtime artifacts for --with-profile/--with-probes (default: enabled, use --no-force-compile to reuse existing binaries)"
    )
    parser.add_argument(
        "--chat-template",
        choices=["auto", "none", "qwen", "gemma"],
        default="auto",
        help="Chat template mode for runtime probe/profile commands"
    )
    parser.add_argument(
        "--perf-runtime",
        choices=["cli", "python"],
        default="cli",
        help="Runtime used by --with-probes perf targets (default: cli for pure C runtime)"
    )
    parser.add_argument(
        "--weight-dtype",
        choices=["float32", "bf16", "q4_0", "q4_1", "q4_k", "q4_k_m", "q5_0", "q5_1", "q6_k", "q8_0"],
        default=None,
        help="Weight dtype override forwarded to the prep runner for profile/probe preparation (e.g., float32 for training-focused builds)"
    )
    parser.add_argument(
        "--vtune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture VTune artifacts in probe flow (default: enabled, use --no-vtune to skip)"
    )
    parser.add_argument(
        "--cachegrind",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture cachegrind artifacts for run-dir native train probes when available (default: enabled)"
    )
    parser.add_argument(
        "--asan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture ASan verification artifacts for run-dir native train probes when available (default: enabled)"
    )
    parser.add_argument(
        "--advisor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture Advisor roofline artifacts for run-dir native train probes when available (default: enabled, use --no-advisor to skip)"
    )
    parser.add_argument(
        "--strict-run-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="When using --run, load artifacts only from run/model roots (default: enabled for --run)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help=(
            "Start live polling server after generating the report. "
            "Requires --run or a model argument. Opens a browser window at "
            "http://localhost:<port>/  and polls training metric files every "
            "--live-interval ms. "
            "Tip: a plain  python3 -m http.server <port> -d <run_dir>  also works "
            "without this flag — the visualizer auto-detects HTTP and starts polling."
        ),
    )
    parser.add_argument(
        "--live-port",
        type=int,
        default=7700,
        metavar="PORT",
        help="Port for the --live server (default: 7700)",
    )
    parser.add_argument(
        "--live-interval",
        type=int,
        default=5000,
        metavar="MS",
        help="Poll interval in milliseconds for --live mode (default: 5000)",
    )

    args = parser.parse_args()

    if args.strict_run_artifacts is None:
        args.strict_run_artifacts = bool(args.run)

    # Operator-first default:
    # `--generate <model>` should produce a fully populated report unless explicitly disabled.
    if args.html_only:
        if args.with_profile or args.with_probes:
            print("Note: --html-only ignores --with-profile/--with-probes and only renders existing artifacts.")
        args.with_profile = False
        args.with_probes = False
    elif args.generate and (args.model or args.run) and not args.with_profile and not args.with_probes:
        if args.run and not args.model:
            print("Defaulting to artifact-only generation for --run (use --with-probes/--with-profile for active capture).")
        else:
            args.with_probes = True
            if not args.force_compile:
                args.force_compile = True
            print("Defaulting to full generation: enabling --with-probes --force-compile (use --html-only for quick HTML-only output).")

    if args.list:
        print("Available models in cache:")
        for m in list_available_models():
            status = "OK" if m["has_data"] else "no data"
            print(f"  - {m['name']} ({status})")
        return

    if (args.with_profile or args.with_probes) and not (args.model or args.run):
        parser.error("--with-profile/--with-probes require --run or a model argument")

    run_dir: Path | None = None

    if args.model or args.run:

        if args.run:
            try:
                ck_build, run_dir = resolve_run_target(args.run)
                model_root = run_dir
            except ValueError as e:
                print(f"Error: {e}")
                return
            if args.model:
                print("Note: --run provided; positional model is ignored for artifact loading.")
        else:
            try:
                ck_build, model_root = resolve_model_target(args.model)
            except ValueError as e:
                print(f"Error: {e}")
                print("\nAvailable models:")
                for m in list_available_models():
                    print(f"  - {m['name']}")
                return

        maybe_interactive_configure(args, model_root)

        if args.with_profile or args.with_probes:
            run_model_input = args.run_model
            run_model_compiled_only = False
            if not run_model_input:
                # Prefer source/HF inputs so probe flow can always prep via ck_run_v8.py first.
                if has_local_runnable_source(model_root):
                    run_model_input = resolve_local_runnable_input(model_root)
                    if run_model_input != str(model_root):
                        print(f"Using local runnable file input: {run_model_input}")
                else:
                    inferred = infer_run_model_input(model_root, args.weight_dtype)
                    if inferred:
                        run_model_input = inferred
                        print(f"Inferred runtime model input: {run_model_input}")
                    elif args.perf_runtime == "cli" and has_local_compiled_runtime(model_root):
                        run_model_input = str(model_root)
                        run_model_compiled_only = True
                        print(
                            f"Using local compiled runtime for CLI probes (no source detected): "
                            f"{run_model_input}"
                        )
                    else:
                        raise SystemExit(
                            "Error: report model directory is not a runnable source checkpoint.\n"
                            "Use --run-model with hf://... (or a .gguf/source checkpoint path)."
                        )
            else:
                run_model_path = Path(run_model_input)
                if not run_model_input.startswith("hf://"):
                    # If caller explicitly provided a local path-like run-model, fail fast when it doesn't exist.
                    looks_like_local = (
                        "/" in run_model_input
                        or run_model_input.startswith(".")
                        or run_model_path.suffix.lower() in {".gguf", ".json", ".safetensors"}
                    )
                    if looks_like_local and not run_model_path.exists():
                        raise SystemExit(
                            f"Error: --run-model local path not found: {run_model_input}\n"
                            "Tip: use a single-line absolute path, or quote it if needed."
                        )
                if run_model_path.exists() and run_model_path.is_dir() and not has_local_runnable_source(run_model_path):
                    if args.perf_runtime == "cli" and has_local_compiled_runtime(run_model_path):
                        run_model_compiled_only = True
                    else:
                        raise SystemExit(
                            "Error: --run-model points to a local directory without .gguf/.safetensors files.\n"
                            "Use an hf://... GGUF source, a .gguf file path, or a source checkpoint directory."
                        )
            run_model_input = maybe_localize_hf_gguf(run_model_input)

            if args.weight_dtype and run_model_input.lower().endswith(".gguf"):
                fallback_source = infer_run_model_input(model_root, args.weight_dtype)
                if fallback_source and not fallback_source.lower().endswith(".gguf"):
                    print(
                        "Switching runtime model input to source checkpoint "
                        f"for --weight-dtype={args.weight_dtype}: {fallback_source}"
                    )
                    run_model_input = fallback_source
                else:
                    raise SystemExit(
                        "Error: --weight-dtype is not compatible with GGUF inputs.\n"
                        "Use --run-model <HF model id or local safetensors dir> when requesting float32/bf16 conversion."
                    )

            effective_chat_template = args.chat_template
            if effective_chat_template == "auto":
                inferred_chat = infer_chat_template(run_model_input, model_root)
                if inferred_chat != "auto":
                    effective_chat_template = inferred_chat
                    print(f"Inferred chat template for probes/profile: {effective_chat_template}")

            chat_args: list[str] = []
            if effective_chat_template and effective_chat_template != "auto":
                chat_args = ["--chat-template", effective_chat_template]

            make_overrides = [
                f"V8_MODEL={run_model_input}",
                f"V8_PERF_RUNTIME={args.perf_runtime}",
                f"V8_WITH_VTUNE={1 if args.vtune else 0}",
                f"V8_FORCE_COMPILE={1 if args.force_compile else 0}",
                f"V8_PREP_WITH_PYTHON={0 if run_model_compiled_only else 1}",
            ]
            if args.weight_dtype:
                make_overrides.append(f"V8_WEIGHT_DTYPE={args.weight_dtype}")
            if effective_chat_template and effective_chat_template != "auto":
                make_overrides.append(f"V8_CHAT_TEMPLATE={effective_chat_template}")
            if effective_chat_template == "none":
                make_overrides.append("V8_CLI_ARGS=--no-chat-template")

            profile_cmd = [
                sys.executable,
                str(CK_RUN_SCRIPT),
                "run",
                run_model_input,
                "--context-len",
                str(args.context_len),
                "--max-tokens",
                str(args.max_tokens),
                "--prompt",
                args.prompt,
                "--profile",
            ]
            profile_cmd.extend(chat_args)
            if args.weight_dtype:
                profile_cmd.extend(["--weight-dtype", args.weight_dtype])
            if args.force_compile:
                profile_cmd.append("--force-compile")

            if args.with_profile:
                if args.perf_runtime == "cli":
                    print("Preparing runtime via ck_run_v8.py...")
                    try_run_cmd(
                        "runtime prep",
                        ["make", "--no-print-directory", "profile-v8-prepare-runtime", *make_overrides],
                        PROJECT_ROOT,
                    )
                    print("Running CLI profile capture...")
                    try_run_cmd(
                        "cli profile capture",
                        ["make", "--no-print-directory", "profile-v8-decode", *make_overrides],
                        PROJECT_ROOT,
                    )
                else:
                    print("Running profile capture...")
                    try_run_cmd("profile capture", profile_cmd, PROJECT_ROOT)

            if args.with_probes:
                report_model_dir = model_root
                has_layout = (report_model_dir / "layout_decode.json").exists()
                has_lowered = (report_model_dir / "lowered_decode_call.json").exists() or (report_model_dir / "lowered_decode.json").exists()
                need_probe_prep = args.force_compile or not (has_layout and has_lowered)
                if need_probe_prep and run_model_compiled_only:
                    print(
                        "Skipping probe prep via ck_run_v8.py: runtime-only input has no source checkpoint "
                        "(use --run-model hf://... or local .gguf/.safetensors to enable prep)."
                    )
                elif need_probe_prep:
                    print("Running probe prep (generate-only)...")
                    prep_cmd = [
                        sys.executable,
                        str(CK_RUN_SCRIPT),
                        "run",
                        run_model_input,
                        "--generate-only",
                        "--profile",
                        "--context-len",
                        str(max(args.context_len, 128)),
                        "--max-tokens",
                        "1",
                        "--prompt",
                        "Hello",
                    ]
                    prep_cmd.extend(chat_args)
                    if args.weight_dtype:
                        prep_cmd.extend(["--weight-dtype", args.weight_dtype])
                    if args.force_compile:
                        prep_cmd.append("--force-compile")
                    try_run_cmd("probe prep (generate-only)", prep_cmd, PROJECT_ROOT)
                else:
                    print("Skipping probe prep (layout/lowered artifacts already present).")

                if not args.with_profile:
                    # Ensure profile_summary.json exists for report profile charts.
                    if args.perf_runtime == "cli":
                        print("Preparing runtime via ck_run_v8.py (required for probes)...")
                        try_run_cmd(
                            "runtime prep (required for probes)",
                            ["make", "--no-print-directory", "profile-v8-prepare-runtime", *make_overrides],
                            PROJECT_ROOT,
                        )
                        print("Running CLI profile capture (required for probes)...")
                        try_run_cmd(
                            "cli profile capture (required for probes)",
                            ["make", "--no-print-directory", "profile-v8-decode", *make_overrides],
                            PROJECT_ROOT,
                        )
                    else:
                        print("Running profile capture (required for probes)...")
                        try_run_cmd("profile capture (required for probes)", profile_cmd, PROJECT_ROOT)

                print("Running memory sign-off...")
                try_run_cmd(
                    "memory sign-off",
                    [
                        sys.executable,
                        str(MEMORY_SIGNOFF_SCRIPT),
                        "--model-dir",
                        str(report_model_dir),
                    ],
                    PROJECT_ROOT,
                )

                print("Running perf probes + budget gate...")
                perf_available = shutil.which("perf") is not None
                vtune_available = shutil.which("vtune") is not None
                advisor_available = shutil.which("advisor") is not None
                valgrind_available = shutil.which("valgrind") is not None
                flamegraph_ok = (
                    (PROJECT_ROOT / "FlameGraph" / "stackcollapse-perf.pl").exists()
                    and (PROJECT_ROOT / "FlameGraph" / "flamegraph.pl").exists()
                )
                if not perf_available:
                    print("Skipping perf probes (perf not installed).")
                elif not flamegraph_ok:
                    print("Skipping perf probes (FlameGraph tools missing).")
                else:
                    try_run_cmd(
                        "perf probe gate",
                        ["make", "--no-print-directory", "v8-perf-gate", *make_overrides],
                        PROJECT_ROOT,
                    )

                expect_advisor = False
                expect_cachegrind = False
                expect_asan = False
                if run_dir is not None and has_train_runtime_artifacts(run_dir):
                    ck_cli = PROJECT_ROOT / "build" / "ck-cli-v8"
                    thread_hint = os.environ.get("CK_NUM_THREADS", "8")
                    token_file = ensure_train_token_file(run_dir)
                    try_run_cmd("build ck-cli-v8 (native profile probes)", ["make", "--no-print-directory", "ck-cli-v8"], PROJECT_ROOT)

                    native_probe_prefix = [
                        str(ck_cli),
                        "profile",
                        "--run",
                        str(run_dir),
                        "--train-token-file",
                        str(token_file),
                        "--train-epochs",
                        "1",
                        "--train-seq-len",
                        "8",
                        "--train-total-tokens",
                        "2048",
                        "--train-grad-accum",
                        "8",
                        "--threads",
                        thread_hint,
                    ]

                    if args.vtune:
                        if vtune_available:
                            print("Running native train VTune probe (ck-cli-v8 profile)...")
                            try_run_cmd(
                                "native train VTune probe",
                                [*native_probe_prefix, "--tool", "vtune"],
                                PROJECT_ROOT,
                                extra_env={"CK_NUM_THREADS": thread_hint},
                            )
                        else:
                            print("Skipping native train VTune probe (vtune not installed).")

                    if args.cachegrind:
                        if valgrind_available:
                            expect_cachegrind = True
                            print("Running native train cachegrind probe (ck-cli-v8 profile)...")
                            try_run_cmd(
                                "native train cachegrind probe",
                                [*native_probe_prefix, "--tool", "cachegrind"],
                                PROJECT_ROOT,
                                extra_env={"CK_NUM_THREADS": thread_hint},
                            )
                        else:
                            print("Skipping native train cachegrind probe (valgrind not installed).")

                    if args.asan:
                        expect_asan = True
                        print("Running native train ASan verification probe (ck-cli-v8 profile)...")
                        try_run_cmd(
                            "native train ASan probe",
                            [*native_probe_prefix, "--tool", "asan"],
                            PROJECT_ROOT,
                            extra_env={"CK_NUM_THREADS": thread_hint},
                        )

                    if args.advisor:
                        if advisor_available:
                            expect_advisor = True
                            print("Running native train Advisor probe (ck-cli-v8 profile)...")
                            try_run_cmd(
                                "native train Advisor probe",
                                [*native_probe_prefix, "--tool", "advisor"],
                                PROJECT_ROOT,
                                extra_env={"CK_NUM_THREADS": thread_hint},
                            )
                        else:
                            print("Skipping native train Advisor probe (advisor not installed).")
                elif args.advisor and run_dir is not None:
                    print("Skipping native train Advisor probe (run directory does not contain compiled train runtime artifacts).")
                if run_dir is None and (args.advisor or args.cachegrind or args.asan):
                    print("Skipping native train probes (no --run directory).")

                runtime_model_dir = detect_model_dir_from_input(run_model_input)
                if runtime_model_dir and runtime_model_dir.exists():
                    copy_artifacts_if_needed(runtime_model_dir, report_model_dir)
                elif runtime_model_dir and not runtime_model_dir.exists():
                    print(f"Warning: runtime artifact directory not found, skip copy: {runtime_model_dir}")

                missing = validate_artifact_set(
                    model_root=report_model_dir,
                    ck_build=ck_build,
                    expect_perf=perf_available and flamegraph_ok,
                    expect_vtune=bool(args.vtune and vtune_available),
                    expect_advisor=expect_advisor,
                    expect_cachegrind=expect_cachegrind,
                    expect_asan=expect_asan,
                )
                if missing:
                    print(f"Warning: missing expected artifacts after probe run: {missing}")
                    print("Tip: rerun with explicit model source or disable optional stages (e.g., --no-vtune/--no-advisor).")

        # Generate report
        output = args.output or ((run_dir or model_root) / "ir_report.html")
        report_path = generate_html_report(
            ck_build,
            output,
            run_dir=run_dir,
            strict_run_artifacts=bool(args.strict_run_artifacts),
        )
        print(f"\nGenerated: {report_path}")

        if getattr(args, "live", False):
            live_run_dir = run_dir or ck_build or model_root
            serve_live(
                run_dir=live_run_dir,
                html_path=report_path,
                port=args.live_port,
                interval_ms=args.live_interval,
            )
        elif not args.generate:
            webbrowser.open(f"file://{report_path}")
    else:
        # Open visualizer (no model specified)
        if getattr(args, "live", False):
            print("Error: --live requires --run or a model argument so artifacts can be located.")
            return
        if VISUALIZER.exists():
            webbrowser.open(f"file://{VISUALIZER}")
        else:
            print(f"Error: Visualizer not found: {VISUALIZER}")


if __name__ == "__main__":
    main()
