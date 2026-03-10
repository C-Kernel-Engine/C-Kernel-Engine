#!/usr/bin/env python3
"""Validate that submodule gitlinks are reachable on their configured remotes."""

from __future__ import annotations

import argparse
import configparser
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", "-C", str(repo), *args], check=check)


def normalize_url(url: str) -> str:
    text = url.strip()
    if text.endswith(".git"):
        text = text[:-4]
    return text.rstrip("/")


def parse_gitmodules(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    rows: list[dict[str, str]] = []
    for section in parser.sections():
        if not section.startswith("submodule "):
            continue
        rows.append(
            {
                "name": section[len("submodule ") :].strip().strip('"'),
                "path": parser.get(section, "path"),
                "url": parser.get(section, "url"),
            }
        )
    return rows


def pinned_gitlink_sha(repo_root: Path, treeish: str, submodule_path: str) -> str:
    proc = git(repo_root, "ls-tree", treeish, submodule_path)
    line = proc.stdout.strip()
    if not line:
        raise RuntimeError(f"submodule path not present in HEAD: {submodule_path}")
    parts = line.split()
    if len(parts) < 3 or parts[1] != "commit":
        raise RuntimeError(f"path is not a gitlink in HEAD: {submodule_path}")
    return parts[2]


def resolve_remote_name(submodule_dir: Path, expected_url: str) -> str:
    remotes = [line.strip() for line in git(submodule_dir, "remote").stdout.splitlines() if line.strip()]
    if not remotes:
        raise RuntimeError(f"{submodule_dir}: no git remotes configured")
    expected = normalize_url(expected_url)
    for remote in remotes:
        url = git(submodule_dir, "remote", "get-url", remote).stdout.strip()
        if normalize_url(url) == expected:
            return remote
    if "origin" in remotes:
        return "origin"
    raise RuntimeError(f"{submodule_dir}: no remote matches {expected_url}")


def fetch_remote_refs(submodule_dir: Path, remote: str) -> None:
    proc = git(submodule_dir, "fetch", "--quiet", "--tags", "--prune", remote, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"git fetch {remote} failed"
        raise RuntimeError(f"{submodule_dir}: {detail}")


def collect_containing_refs(submodule_dir: Path, sha: str, remote: str) -> list[str]:
    branch_proc = git(
        submodule_dir,
        "for-each-ref",
        f"--contains={sha}",
        f"refs/remotes/{remote}",
        "--format=%(refname:short)",
    )
    tag_proc = git(
        submodule_dir,
        "for-each-ref",
        f"--contains={sha}",
        "refs/tags",
        "--format=tag:%(refname:short)",
    )
    containing = [line.strip() for line in branch_proc.stdout.splitlines() if line.strip()]
    containing.extend(line.strip() for line in tag_proc.stdout.splitlines() if line.strip())
    return containing


def validate_submodule(repo_root: Path, treeish: str, item: dict[str, str], fetch: bool) -> tuple[bool, str]:
    submodule_dir = repo_root / item["path"]
    sha = pinned_gitlink_sha(repo_root, treeish, item["path"])

    if not submodule_dir.exists():
        return False, f"{item['path']}: missing local checkout; run git submodule update --init --recursive"
    probe = git(submodule_dir, "rev-parse", "--is-inside-work-tree", check=False)
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return False, f"{item['path']}: not a git worktree; run git submodule update --init --recursive"

    if git(submodule_dir, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode != 0:
        return False, f"{item['path']}: pinned commit {sha} is missing locally"

    remote = resolve_remote_name(submodule_dir, item["url"])
    if fetch:
        fetch_remote_refs(submodule_dir, remote)

    containing = collect_containing_refs(submodule_dir, sha, remote)
    if not containing:
        return False, (
            f"{item['path']}: pinned commit {sha} is not contained in any fetched ref from "
            f"{remote} ({item['url']})"
        )
    preview = ", ".join(containing[:3])
    if len(containing) > 3:
        preview += ", ..."
    return True, f"{item['path']}: {sha[:12]} reachable via {preview}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--treeish", default="HEAD", help="Commit/tree to validate (default: HEAD)")
    parser.add_argument("--no-fetch", action="store_true", help="Use existing remote-tracking refs without fetching")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    items = parse_gitmodules(repo_root / ".gitmodules")
    if not items:
        print("No submodules configured.")
        return 0

    failures = 0
    for item in items:
        try:
            ok, message = validate_submodule(repo_root, args.treeish, item, fetch=not args.no_fetch)
        except Exception as exc:
            ok = False
            message = f"{item['path']}: {exc}"
        prefix = "OK" if ok else "FAIL"
        print(f"[{prefix}] {message}")
        if not ok:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
