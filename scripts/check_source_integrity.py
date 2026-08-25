#!/usr/bin/env python3
"""Reject unresolved merges and conflict markers in production source files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    ".github",
    ".githooks",
    "include",
    "scripts",
    "src",
    "tests",
    "unittest",
    "version/v7",
    "version/v8",
    "Makefile",
)
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".mk",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
SOURCE_NAMES = {"Makefile", "cks-v8-run"}
MARKER_PREFIXES = ("<" * 7, "|" * 7, ">" * 7)


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    for name in (
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_WORK_TREE",
    ):
        env.pop(name, None)
    return subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
        check=False,
    )


def _is_git_checkout(root: Path) -> bool:
    return _run_git(root, "rev-parse", "--is-inside-work-tree").returncode == 0


def _is_source(path: Path) -> bool:
    return path.name in SOURCE_NAMES or path.suffix.lower() in SOURCE_SUFFIXES


def _tracked_sources(root: Path) -> Iterable[Path]:
    result = _run_git(root, "ls-files", "-z", "--", *SCAN_ROOTS)
    if result.returncode != 0:
        return ()
    relative_paths = {
        raw.decode("utf-8", errors="surrogateescape")
        for raw in result.stdout.split(b"\0")
        if raw
        and _is_source(Path(raw.decode("utf-8", errors="surrogateescape")))
    }
    return (root / relative for relative in sorted(relative_paths))


def _filesystem_sources(root: Path) -> Iterable[Path]:
    for relative in SCAN_ROOTS:
        candidate = root / relative
        if candidate.is_file() and _is_source(candidate):
            yield candidate
        elif candidate.is_dir():
            for path in candidate.rglob("*"):
                if path.is_file() and _is_source(path):
                    yield path


def _unmerged_paths(root: Path) -> list[str]:
    result = _run_git(root, "ls-files", "-u", "-z", "--", *SCAN_ROOTS)
    if result.returncode != 0 or not result.stdout:
        return []
    paths = set()
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        text = entry.decode("utf-8", errors="replace")
        if "\t" in text:
            paths.add(text.split("\t", 1)[1])
    return sorted(paths)


def _is_conflict_marker(line: str) -> bool:
    stripped = line.rstrip("\r\n")
    return any(
        stripped == prefix or stripped.startswith(prefix + " ")
        for prefix in MARKER_PREFIXES
    )


def audit(root: Path = ROOT) -> dict[str, object]:
    root = root.resolve()
    git_checkout = _is_git_checkout(root)
    unmerged = _unmerged_paths(root) if git_checkout else []
    sources = _tracked_sources(root) if git_checkout else _filesystem_sources(root)
    markers: list[dict[str, object]] = []
    scanned = 0

    for path in sources:
        if not path.is_file():
            continue
        scanned += 1
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if _is_conflict_marker(line):
                        markers.append(
                            {
                                "path": str(path.relative_to(root)),
                                "line": line_number,
                                "marker": line.rstrip("\r\n")[:80],
                            }
                        )
        except OSError as exc:
            markers.append(
                {
                    "path": str(path.relative_to(root)),
                    "line": 0,
                    "marker": f"unreadable source: {exc}",
                }
            )

    return {
        "schema": "ck.source_integrity.v1",
        "status": "fail" if unmerged or markers else "pass",
        "root": str(root),
        "git_checkout": git_checkout,
        "scanned_files": scanned,
        "unmerged_paths": unmerged,
        "conflict_markers": markers,
    }


def format_failure(report: dict[str, object]) -> str:
    lines = ["source integrity check failed"]
    for path in report["unmerged_paths"]:
        lines.append(f"  unresolved index: {path}")
    for finding in report["conflict_markers"]:
        lines.append(
            f"  conflict marker: {finding['path']}:{finding['line']}: "
            f"{finding['marker']}"
        )
    lines.append("Resolve the merge completely before conversion, codegen, or execution.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = audit(args.root)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "pass":
        print(format_failure(report), file=sys.stderr)
        return 1
    print(f"source integrity passed: {report['scanned_files']} tracked source files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
