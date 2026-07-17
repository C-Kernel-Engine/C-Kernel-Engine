#!/usr/bin/env python3
"""Probe current llama.cpp HEAD without changing CK's pinned parity oracle."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
LLAMA_REPO = "https://github.com/ggerganov/llama.cpp.git"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def run(
    command: list[str],
    *,
    cwd: pathlib.Path = ROOT,
    check: bool = True,
    stdout: Any = subprocess.PIPE,
    stderr: Any = subprocess.STDOUT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        stdout=stdout,
        stderr=stderr,
        env=env,
    )


def git_output(args: list[str], *, cwd: pathlib.Path = ROOT) -> str:
    return run(["git", *args], cwd=cwd).stdout.strip()


def pinned_commit() -> str:
    row = git_output(["ls-tree", "HEAD", "llama.cpp"])
    fields = row.split()
    if len(fields) < 3:
        raise RuntimeError("HEAD does not contain a llama.cpp gitlink")
    return fields[2]


def remote_commit(ref: str) -> str:
    rows = git_output(["ls-remote", LLAMA_REPO, ref]).splitlines()
    if not rows:
        raise RuntimeError(f"llama.cpp ref not found: {ref}")
    return rows[0].split()[0]


def clone_for_probe(path: pathlib.Path, rolling: str, pinned: str) -> None:
    if (path / ".git").exists():
        current = git_output(["rev-parse", "HEAD"], cwd=path)
        has_pin = run(["git", "cat-file", "-e", pinned], cwd=path, check=False).returncode == 0
        if current == rolling and has_pin:
            return
    shutil.rmtree(path, ignore_errors=True)
    run(["git", "clone", "--filter=blob:none", "--no-checkout", LLAMA_REPO, str(path)])
    run(["git", "fetch", "origin", rolling, pinned], cwd=path)
    run(["git", "checkout", "--detach", rolling], cwd=path)


def patch_status(probe: pathlib.Path, relative_path: str) -> dict[str, Any]:
    patch = ROOT / relative_path
    if not patch.exists():
        return {"path": relative_path, "status": "missing"}
    result = run(
        ["git", "apply", "--check", str(patch)],
        cwd=probe,
        check=False,
    )
    return {
        "path": relative_path,
        "status": "applies" if result.returncode == 0 else "incompatible",
        "detail": result.stdout.strip()[-2000:],
    }


def compatibility_status(phases: dict[str, Any]) -> str:
    """Treat executable compatibility as authoritative and patch drift as advisory."""
    for phase_name in ("ck_build", "quick_parity"):
        status = phases.get(phase_name, {}).get("status")
        if status in {"fail", "error"}:
            return "fail"
    return "pass"


def upstream_delta(probe: pathlib.Path, pinned: str, rolling: str, limit: int) -> tuple[int, list[dict[str, str]]]:
    count = int(git_output(["rev-list", "--count", f"{pinned}..{rolling}"], cwd=probe))
    lines = git_output(
        ["log", f"--max-count={limit}", "--format=%H%x09%s", f"{pinned}..{rolling}"],
        cwd=probe,
    ).splitlines()
    commits = []
    for line in lines:
        sha, _, subject = line.partition("\t")
        commits.append({"commit": sha, "subject": subject})
    return count, commits


def write_report(path: pathlib.Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_quick_log(path: pathlib.Path) -> dict[str, Any]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    summaries = re.findall(r"Passed:\s+(\d+)\s+Failed:\s+(\d+)\s+Skipped:\s+(\d+)", text)
    performance = []
    for speedup, ck_gflops, llama_gflops in re.findall(
        r"Average CK/llama\.cpp speedup:\s+([0-9.]+)x.*?"
        r"Average CK GFLOPS:\s+([0-9.]+).*?"
        r"Average llama\.cpp GFLOPS:\s+([0-9.]+)",
        text,
        flags=re.DOTALL,
    ):
        performance.append(
            {
                "ck_over_llama_speedup": float(speedup),
                "ck_gflops": float(ck_gflops),
                "llama_gflops": float(llama_gflops),
            }
        )
    result: dict[str, Any] = {"performance_summaries": performance}
    if summaries:
        passed, failed, skipped = summaries[-1]
        result["smoketest"] = {"passed": int(passed), "failed": int(failed), "skipped": int(skipped)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD", help="Rolling llama.cpp ref to probe")
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "build" / "llamacpp_rolling")
    parser.add_argument("--commit-limit", type=int, default=50)
    parser.add_argument("--resolve-only", action="store_true", help="Resolve and inspect upstream without building")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    probe = output_dir / "llama.cpp"
    report_path = output_dir / "report.json"
    pinned = pinned_commit()
    rolling = remote_commit(args.ref)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cke_commit": git_output(["rev-parse", "HEAD"]),
        "llama_cpp": {"repository": LLAMA_REPO, "pinned_commit": pinned, "rolling_commit": rolling},
        "authoritative_oracle_changed": False,
        "status": "running",
        "phases": {},
    }

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        clone_for_probe(probe, rolling, pinned)
        count, commits = upstream_delta(probe, pinned, rolling, args.commit_limit)
        report["llama_cpp"]["commits_ahead_of_pin"] = count
        report["llama_cpp"]["recent_delta"] = commits
        patches = [
            patch_status(probe, "patches/llama.patch"),
            patch_status(probe, "patches/ck-engine-parity-bench.patch"),
        ]
        patch_drift = any(item["status"] != "applies" for item in patches)
        report["phases"]["patch_compatibility"] = {
            "status": "warn" if patch_drift else "pass",
            "blocking": False,
            "reason": (
                "Optional tensor-dump and benchmark patches drifted; rolling build and "
                "quick parity determine compatibility."
                if patch_drift
                else "Optional patches still apply cleanly."
            ),
            "patches": patches,
        }

        if args.resolve_only:
            report["phases"]["quick_parity"] = {"status": "skip", "reason": "resolve-only"}
            report["status"] = "pass"
            write_report(report_path, report)
            print(report_path)
            return 0

        log_path = output_dir / "quick-parity.log"
        build_log_path = output_dir / "ck-build.log"
        environment = os.environ.copy()
        environment["LLAMA_CPP_COMMIT"] = rolling
        environment["LLAMA_CPP_DIR"] = str(probe)
        environment.setdefault("PYTHON_BIN", str(ROOT / ".venv" / "bin" / "python"))
        with build_log_path.open("w", encoding="utf-8") as log:
            build_result = run(
                ["make", "build/libckernel_engine.so", "build/libck_parity.so"],
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        report["phases"]["ck_build"] = {
            "status": "pass" if build_result.returncode == 0 else "fail",
            "returncode": build_result.returncode,
            "log": str(build_log_path),
        }
        if build_result.returncode != 0:
            report["phases"]["quick_parity"] = {"status": "skip", "reason": "CK build failed"}
            report["status"] = "fail"
            write_report(report_path, report)
            print(report_path)
            return 1
        with log_path.open("w", encoding="utf-8") as log:
            result = run(
                [str(ROOT / "scripts" / "run_parity_smoketest.sh"), "--quick"],
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        report["phases"]["quick_parity"] = {
            "status": "pass" if result.returncode == 0 else "fail",
            "returncode": result.returncode,
            "log": str(log_path),
            "summary": parse_quick_log(log_path),
        }
        report["status"] = compatibility_status(report["phases"])
    except Exception as exc:  # Preserve diagnostics even when upstream changes break setup.
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"

    write_report(report_path, report)
    print(report_path)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
