#!/usr/bin/env python3
"""
v7_doctor.py - environment and dependency checker for v7 workflows.

This script is package-manager agnostic. It reports:
  - supported platform boundary
  - required system tools
  - required Python packages for the supported v7 workflow bundle
  - optional profiling tools
"""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements-v7.txt"
VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
USE_COLOR = sys.stdout.isatty()

C_RESET = "\033[0m" if USE_COLOR else ""
C_GREEN = "\033[32m" if USE_COLOR else ""
C_RED = "\033[31m" if USE_COLOR else ""
C_YELLOW = "\033[33m" if USE_COLOR else ""
C_CYAN = "\033[36m" if USE_COLOR else ""

REQUIRED_SYSTEM_TOOLS = ("git", "make", "python3")
COMPILER_CANDIDATES = ("gcc", "clang", "icx")
OPTIONAL_PROFILING_TOOLS = (
    ("perf", "perf stat + flamegraph capture"),
    ("valgrind", "cachegrind capture"),
    ("cg_annotate", "cachegrind annotation"),
    ("vtune", "Intel VTune hotspot analysis"),
    ("advisor", "Intel Advisor roofline analysis"),
    ("flamegraph.pl", "Brendan Gregg FlameGraph rendering"),
    ("stackcollapse-perf.pl", "Brendan Gregg FlameGraph stack collapse"),
)


def _parse_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []
    packages: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line.startswith("-e ") or "://" in line:
            continue
        pkg = re.split(r"[<>=!~\\[\\]\\s]", line, maxsplit=1)[0].strip()
        if pkg:
            packages.append(pkg)
    return packages


def _find_python_module(pkg: str) -> bool:
    module_name = pkg.replace("-", "_")
    return importlib.util.find_spec(module_name) is not None


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _print_status(label: str, ok: bool, detail: str = "") -> None:
    color = C_GREEN if ok else C_RED
    status = f"{color}{'PASS' if ok else 'MISS'}{C_RESET}"
    suffix = f"  {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")


def _detect_oneapi_root() -> str | None:
    env = os.environ.get("ONEAPI_ROOT")
    if env:
        return env
    default = Path("/opt/intel/oneapi")
    if default.exists():
        return str(default)
    return None


def _resolve_tool(tool: str) -> str | None:
    resolved = shutil.which(tool)
    if resolved:
        return resolved

    common_roots = (
        PROJECT_ROOT / "FlameGraph",
        Path.cwd() / "FlameGraph",
        Path.home() / "Programs" / "FlameGraph",
        Path.home() / "FlameGraph",
    )
    for root in common_roots:
        candidate = root / tool
        if candidate.exists():
            return f"{candidate} (not on PATH)"
    return None


def _read_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.exists():
        return {}
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        data[key] = value.strip().strip('"')
    return data


def _install_hints() -> list[str]:
    os_release = _read_os_release()
    distro_id = (os_release.get("ID") or "").lower()
    distro_like = (os_release.get("ID_LIKE") or "").lower()

    if distro_id in {"ubuntu", "debian", "linuxmint", "pop"} or any(x in distro_like for x in ("debian", "ubuntu")):
        return [
            "sudo apt install build-essential git make python3 python3-venv python3-pip",
            "sudo apt install valgrind linux-tools-common linux-tools-generic linux-tools-$(uname -r)",
            "git clone https://github.com/brendangregg/FlameGraph.git ~/Programs/FlameGraph",
            "# Intel hosts: install Intel oneAPI Base Toolkit for icx/vtune/advisor",
        ]
    if distro_id in {"arch", "manjaro", "endeavouros"} or "arch" in distro_like:
        return [
            "sudo pacman -S base-devel git make python python-pip python-virtualenv",
            "sudo pacman -S valgrind perf flamegraph",
            "sudo pacman -S intel-oneapi-basekit   # Intel hosts",
        ]
    if distro_id in {"fedora", "rhel", "centos", "rocky", "almalinux"} or any(x in distro_like for x in ("fedora", "rhel")):
        return [
            "sudo dnf install gcc gcc-c++ make git python3 python3-pip python3-virtualenv",
            "sudo dnf install valgrind perf",
            "git clone https://github.com/brendangregg/FlameGraph.git ~/Programs/FlameGraph",
            "# Intel hosts: install Intel oneAPI Base Toolkit for icx/vtune/advisor",
        ]
    if distro_id in {"opensuse", "opensuse-tumbleweed", "opensuse-leap"} or "suse" in distro_like:
        return [
            "sudo zypper install gcc gcc-c++ make git python3 python3-pip python3-virtualenv",
            "sudo zypper install valgrind perf",
            "git clone https://github.com/brendangregg/FlameGraph.git ~/Programs/FlameGraph",
            "# Intel hosts: install Intel oneAPI Base Toolkit for icx/vtune/advisor",
        ]
    return [
        "Install: git, make, python3, python3-venv, python3-pip, and a C compiler (gcc/clang/icx).",
        "Optional profiling: perf, valgrind, FlameGraph, and Intel oneAPI Base Toolkit (icx/vtune/advisor on Intel hosts).",
    ]


def main() -> int:
    packages = _parse_requirements(REQUIREMENTS_PATH)
    platform_ok = sys.platform.startswith("linux")
    current_python = Path(sys.executable).resolve()

    print("=== v7 doctor ===")
    print(f"Interpreter: {current_python}")
    print(f"Platform:    {platform.platform()}")
    print(f"Repo root:    {PROJECT_ROOT}")
    if VENV_PY.exists() and current_python != VENV_PY.resolve():
        print(f"Repo venv:    {VENV_PY} (current interpreter is different)")

    _print_header("Support boundary")
    _print_status("Linux operator path", platform_ok, "supported" if platform_ok else "best-effort only outside Linux")

    _print_header("Required system tools")
    missing_required_tools: list[str] = []
    for tool in REQUIRED_SYSTEM_TOOLS:
        resolved = shutil.which(tool)
        ok = resolved is not None
        if not ok:
            missing_required_tools.append(tool)
        _print_status(tool, ok, resolved or "not found")

    available_compilers = [(name, shutil.which(name)) for name in COMPILER_CANDIDATES]
    compiler_paths = [f"{name}={path}" for name, path in available_compilers if path]
    compiler_ok = bool(compiler_paths)
    compiler_detail = ", ".join(compiler_paths) if compiler_paths else "need gcc, clang, or icx"
    if not compiler_ok:
        missing_required_tools.append("compiler")
    _print_status("compiler", compiler_ok, compiler_detail)

    _print_header("Required Python packages (supported v7 workflows)")
    missing_packages: list[str] = []
    for pkg in packages:
        ok = _find_python_module(pkg)
        if not ok:
            missing_packages.append(pkg)
        _print_status(pkg, ok)

    _print_header("Optional profiling tools")
    for tool, purpose in OPTIONAL_PROFILING_TOOLS:
        resolved = _resolve_tool(tool)
        _print_status(tool, resolved is not None, purpose if resolved is None else f"{resolved}  {purpose}")

    _print_header("Intel toolchain (optional, recommended on Intel hosts)")
    for tool in ("icx",):
        resolved = shutil.which(tool)
        _print_status(tool, resolved is not None, resolved or "not found")
    oneapi_root = _detect_oneapi_root()
    _print_status("oneAPI root", oneapi_root is not None, oneapi_root or "not found")

    print("\nRequired Python package set:")
    print("  " + " ".join(packages))
    print("  This is the supported v7 bundle for run/train/parity workflows.")
    print("  Not every package is used on every execution path.")

    print("\nSupported bootstrap:")
    print("  make v7-init")
    print("  make v7-doctor")

    print("\nManual environment (pip example):")
    print("  python3 -m venv .venv")
    print("  . .venv/bin/activate")
    print("  python -m pip install --upgrade pip")
    print(f"  python -m pip install -r {REQUIREMENTS_PATH.name}")
    print("\nSuggested host install commands:")
    for hint in _install_hints():
        print(f"  {hint}")

    print(f"\n{C_CYAN}How these tools help{C_RESET}")
    print(f"  {C_CYAN}perf{C_RESET}: CPU counters and timing")
    print(f"  {C_CYAN}FlameGraph{C_RESET}: visual hotspot stacks from perf samples")
    print(f"  {C_CYAN}valgrind/cg_annotate{C_RESET}: cachegrind miss attribution")
    print(f"  {C_CYAN}vtune{C_RESET}: microarchitectural hotspot and threading analysis")
    print(f"  {C_CYAN}advisor{C_RESET}: roofline and memory-vs-compute analysis")

    if missing_required_tools or missing_packages:
        print("\nResult: missing required dependencies.")
        return 1

    print("\nResult: required dependencies look ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
