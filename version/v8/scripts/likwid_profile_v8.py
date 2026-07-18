#!/usr/bin/env python3
"""Capture optional LIKWID wrapper profiles and emit a stable v8 artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "cke.profile.likwid.v1"
AUTO_GROUP_PREFERENCE = (
    "MEM",
    "CACHE",
    "L3",
    "L2",
    "CLOCK",
    "BRANCH",
    "FLOPS_DP",
    "FLOPS_SP",
)

WORKLOAD_STATUS_WRAPPER_NAME = "cke-likwid-workload"
WORKLOAD_STATUS_SHELL = (
    'status_file=$1; shift; "$@"; status=$?; '
    'printf "%s\\n" "$status" > "$status_file"; exit "$status"'
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_number(value: str) -> float | None:
    text = value.strip().replace(",", "")
    if not text or text.lower() in {"nan", "inf", "-inf", "n/a", "-"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_available_groups(text: str) -> list[dict[str, str]]:
    """Parse `likwid-perfctr -a` across old and new LIKWID table formats."""
    groups: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("-", "Group name", "Groups")):
            continue
        if re.match(r"^No\s+groups?\s+defined\b", line, flags=re.IGNORECASE):
            continue
        match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s+(.*)$", line)
        if not match:
            continue
        name, description = match.groups()
        upper = name.upper()
        if upper in {"AVAILABLE", "GROUP", "EVENT"} or upper in seen:
            continue
        seen.add(upper)
        groups.append({"name": upper, "description": description.strip()})
    return groups


def choose_groups(
    available: Iterable[dict[str, str]], requested: str, max_groups: int
) -> list[str]:
    available_names = {
        str(item.get("name", "")).upper()
        for item in available
        if item.get("name")
    }
    if requested.strip().lower() == "auto":
        selected = [name for name in AUTO_GROUP_PREFERENCE if name in available_names]
    else:
        wanted = [part.strip().upper() for part in requested.split(",") if part.strip()]
        selected = [name for name in wanted if name in available_names]
    return selected[:max_groups] if max_groups > 0 else selected


def default_cpu_ids(thread_count: int) -> list[int]:
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = list(range(os.cpu_count() or 1))
    return allowed[: max(1, min(thread_count, len(allowed)))]


def parse_cpu_ids(value: str, thread_count: int) -> list[int]:
    if value.strip().lower() == "auto":
        return default_cpu_ids(thread_count)
    cpus: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid descending CPU range: {token}")
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(token))
    if not cpus:
        raise ValueError("CPU selection is empty")
    return list(dict.fromkeys(cpus))


def _metric_key(name: str) -> str | None:
    normalized = re.sub(r"\s+", " ", name.strip()).lower()
    mappings = (
        ("runtime (rdtsc)", "runtime_seconds"),
        ("runtime unhalted", "runtime_unhalted_seconds"),
        ("memory bandwidth", "memory_bandwidth_mbytes_per_second"),
        ("memory data volume", "memory_data_volume_gbytes"),
        ("cpi", "cpi"),
        ("ipc", "ipc"),
        ("branch misprediction rate", "branch_misprediction_rate"),
        ("l2 bandwidth", "l2_bandwidth_mbytes_per_second"),
        ("l3 bandwidth", "l3_bandwidth_mbytes_per_second"),
        ("mflop/s", "mflops"),
    )
    for needle, key in mappings:
        if needle in normalized:
            return key
    return None


def parse_likwid_csv(path: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Extract LIKWID metric rows without depending on a processor's group set."""
    if not path.exists():
        return [], {}
    rows: list[dict[str, Any]] = []
    normalized_values: dict[str, list[float]] = {}
    with path.open("r", errors="ignore", newline="") as handle:
        for row in csv.reader(handle):
            cells = [cell.strip() for cell in row]
            if len(cells) < 2:
                continue
            label_index = 1 if cells[0].upper() in {"METRIC", "EVENT"} else 0
            label = cells[label_index]
            values = [
                value
                for value in (parse_number(cell) for cell in cells[label_index + 1 :])
                if value is not None
            ]
            if not label or not values:
                continue
            entry = {
                "name": label,
                "values": values,
                "average": sum(values) / len(values),
            }
            rows.append(entry)
            key = _metric_key(label)
            if key:
                normalized_values.setdefault(key, []).extend(values)
    normalized = {
        key: sum(values) / len(values)
        for key, values in normalized_values.items()
        if values
    }
    return rows, normalized


def run_metadata_command(command: list[str], output: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    output.write_text(completed.stdout + completed.stderr)
    return completed


def wrap_workload_with_status(command: list[str], status_path: Path) -> list[str]:
    """Run a workload while preserving its exit status independently of LIKWID."""
    return [
        "/bin/sh",
        "-c",
        WORKLOAD_STATUS_SHELL,
        WORKLOAD_STATUS_WRAPPER_NAME,
        str(status_path),
        *command,
    ]


def read_workload_status(path: Path) -> int | None:
    """Return a recorded child status, failing closed on missing/malformed evidence."""
    try:
        value = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    return value if 0 <= value <= 255 else None


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def register_plot_artifacts(
    paths: Iterable[Path], artifact_dir: Path, summary: dict[str, Any]
) -> None:
    """Preserve user-exported LIKWID/perfscope plots beside counter evidence."""
    supported = {".svg", ".png", ".jpg", ".jpeg", ".webp"}
    for source in paths:
        resolved = source.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"LIKWID plot artifact does not exist: {source}")
        if resolved.suffix.lower() not in supported:
            raise ValueError(
                f"unsupported LIKWID plot format {resolved.suffix!r}; "
                "use SVG, PNG, JPEG, or WebP"
            )
        destination = artifact_dir / f"plot_{resolved.name}"
        if resolved != destination.resolve():
            shutil.copy2(resolved, destination)
        suffix = destination.suffix.lower()
        media_type = {
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }[suffix]
        summary["artifacts"].append(
            {
                "kind": "plot",
                "path": str(destination),
                "media_type": media_type,
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", default="auto")
    parser.add_argument("--max-groups", type=int, default=2)
    parser.add_argument("--cpus", default="auto")
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, int(os.environ.get("CK_NUM_THREADS", "1"))),
    )
    parser.add_argument("--summary-name", default="likwid_summary.json")
    parser.add_argument(
        "--plot-artifact",
        type=Path,
        action="append",
        default=[],
        help="preserve an exported LIKWID/perfscope SVG or raster plot (repeatable)",
    )
    parser.add_argument(
        "--require-output",
        action="append",
        default=[],
        help=(
            "require this literal text in the wrapped workload output before "
            "reporting pass (repeatable)"
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a workload command is required after --")

    output_dir = args.output_dir.resolve()
    artifact_dir = output_dir / "likwid"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / args.summary_name
    executable = shutil.which("likwid-perfctr")
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now_iso(),
        "status": "skip",
        "reason": "",
        "tool": {
            "name": "likwid-perfctr",
            "path": executable,
        },
        "requested_groups": args.groups,
        "available_groups": [],
        "selected_groups": [],
        "cpu_ids": [],
        "command": command,
        "required_output": args.require_output,
        "runs": [],
        "normalized": {},
        "artifacts": [],
        "limitations": [
            "Wrapper mode measures activity on the pinned CPUs, not only the CKE process.",
            "Counter groups and metric names vary by processor and LIKWID version.",
        ],
    }
    try:
        register_plot_artifacts(args.plot_artifact, artifact_dir, summary)
    except ValueError as exc:
        summary.update(status="fail", reason=str(exc))
        write_summary(summary_path, summary)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not executable:
        summary["reason"] = "likwid-perfctr is not installed"
        write_summary(summary_path, summary)
        print(f"SKIP: {summary['reason']} ({summary_path})")
        return 0

    try:
        cpu_ids = parse_cpu_ids(args.cpus, args.threads)
    except ValueError as exc:
        summary.update(status="fail", reason=str(exc))
        write_summary(summary_path, summary)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    cpu_spec = ",".join(str(cpu) for cpu in cpu_ids)
    summary["cpu_ids"] = cpu_ids

    info_path = artifact_dir / "likwid_info.txt"
    groups_path = artifact_dir / "available_groups.txt"
    version_path = artifact_dir / "version.txt"
    version_run = run_metadata_command([executable, "-v"], version_path)
    info_run = run_metadata_command([executable, "-i"], info_path)
    groups_run = run_metadata_command([executable, "-a"], groups_path)
    summary["tool"]["version"] = version_run.stdout.strip().splitlines()[:1]
    summary["tool"]["info_returncode"] = info_run.returncode
    for kind, path in (
        ("version", version_path),
        ("cpu_info", info_path),
        ("available_groups", groups_path),
    ):
        summary["artifacts"].append({"kind": kind, "path": str(path)})

    available = parse_available_groups(groups_run.stdout)
    selected = choose_groups(available, args.groups, args.max_groups)
    summary["available_groups"] = available
    summary["selected_groups"] = selected
    if groups_run.returncode != 0:
        summary["reason"] = "LIKWID could not enumerate counter groups"
        write_summary(summary_path, summary)
        print(f"SKIP: {summary['reason']} ({summary_path})")
        return 0
    if not selected:
        cpu_info = info_path.read_text(errors="ignore")
        if re.search(r"^CPU type:\s*nil\s*$", cpu_info, flags=re.MULTILINE):
            summary["reason"] = (
                "installed LIKWID does not recognize this processor; "
                "no counter groups are available"
            )
        else:
            summary["reason"] = "none of the requested LIKWID groups are available"
        write_summary(summary_path, summary)
        print(f"SKIP: {summary['reason']} ({summary_path})")
        return 0

    normalized_by_group: dict[str, dict[str, float]] = {}
    for group in selected:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", group.lower())
        csv_path = artifact_dir / f"{slug}.csv"
        stdout_path = artifact_dir / f"{slug}.stdout.txt"
        stderr_path = artifact_dir / f"{slug}.stderr.txt"
        status_path = artifact_dir / f"{slug}.workload_status.txt"
        status_path.unlink(missing_ok=True)
        workload = wrap_workload_with_status(command, status_path)
        wrapped = [
            executable,
            "-C",
            cpu_spec,
            "-g",
            group,
            "-o",
            str(csv_path),
            "--",
            *workload,
        ]
        completed = subprocess.run(wrapped, text=True, capture_output=True, check=False)
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        workload_returncode = read_workload_status(status_path)
        combined_output = f"{completed.stdout}\n{completed.stderr}"
        missing_output = [
            required
            for required in args.require_output
            if required not in combined_output
        ]
        metrics, normalized = parse_likwid_csv(csv_path)
        normalized_by_group[group] = normalized
        run = {
            "group": group,
            "returncode": completed.returncode,
            "tool_returncode": completed.returncode,
            "workload_returncode": workload_returncode,
            "missing_required_output": missing_output,
            "command": wrapped,
            "csv_path": str(csv_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "metrics": metrics,
            "normalized": normalized,
        }
        summary["runs"].append(run)
        summary["artifacts"].extend(
            [
                {"kind": f"{group} csv", "path": str(csv_path)},
                {"kind": f"{group} stdout", "path": str(stdout_path)},
                {"kind": f"{group} stderr", "path": str(stderr_path)},
            ]
        )
        if status_path.is_file():
            summary["artifacts"].append(
                {"kind": f"{group} workload status", "path": str(status_path)}
            )

    failed_tool = [run for run in summary["runs"] if run["tool_returncode"] != 0]
    failed_workload = [
        run for run in summary["runs"] if run["workload_returncode"] != 0
    ]
    failed_output = [run for run in summary["runs"] if run["missing_required_output"]]
    failed = failed_tool or failed_workload or failed_output
    if failed:
        summary["status"] = "fail"
        reasons: list[str] = []
        if failed_tool:
            reasons.append("one or more LIKWID counter runs failed")
        if failed_workload:
            reasons.append(
                "one or more wrapped workloads failed or did not record an exit status"
            )
        if failed_output:
            missing = sorted(
                {
                    item
                    for run in failed_output
                    for item in run["missing_required_output"]
                }
            )
            reasons.append(
                "wrapped workload output is missing required text: "
                + ", ".join(repr(item) for item in missing)
            )
        summary["reason"] = "; ".join(reasons)
    else:
        summary["status"] = "pass"
    summary["normalized"] = normalized_by_group
    write_summary(summary_path, summary)
    print(f"LIKWID profile: {summary['status']} ({summary_path})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
