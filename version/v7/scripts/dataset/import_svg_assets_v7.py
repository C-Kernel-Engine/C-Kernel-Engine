#!/usr/bin/env python3
"""
Import SVG assets into a spec workspace raw_assets/ tree and write an inventory.

Raw assets are the immutable source-of-truth import layer. They preserve provenance
and should not be pre-split into pretrain/midtrain/SFT buckets.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _expand_patterns(patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in patterns:
        p = Path(raw).expanduser()
        candidates: list[Path] = []
        if p.exists() and p.is_dir():
            candidates = sorted(p.rglob("*.svg"))
        else:
            candidates = [Path(x) for x in sorted(glob.glob(str(p), recursive=True)) if x.lower().endswith(".svg")]
        for cand in candidates:
            try:
                key = str(cand.resolve())
            except Exception:
                key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
    return out


def _safe_name(name: str) -> str:
    keep = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("-")
    out = "".join(keep).strip("-")
    return out or "source"


def _parse_source_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise SystemExit(f"ERROR: --source must be NAME=PATH_OR_GLOB, got: {raw}")
    name, pattern = raw.split("=", 1)
    name = _safe_name(name)
    pattern = pattern.strip()
    if not pattern:
        raise SystemExit(f"ERROR: empty path/glob in --source {raw}")
    return name, pattern


def main() -> int:
    ap = argparse.ArgumentParser(description="Import SVG assets into a spec workspace raw_assets/ tree")
    ap.add_argument("--workspace", required=True, help="Spec workspace root, e.g. version/v7/data/spec03")
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source definition NAME=PATH_OR_GLOB (repeatable)",
    )
    ap.add_argument("--manifest", default=None, help="Optional manifest path (default: WORKSPACE/manifests/raw_assets_inventory.json)")
    ap.add_argument("--copy", action="store_true", help="Physically copy files into raw_assets/ (default)")
    ap.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying")
    ap.add_argument("--force", action="store_true", help="Overwrite existing imported file names")
    args = ap.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    raw_root = workspace / "raw_assets"
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else (workspace / "manifests" / "raw_assets_inventory.json")
    use_symlink = bool(args.symlink)
    if use_symlink and bool(args.copy):
        raise SystemExit("ERROR: choose at most one of --copy or --symlink")

    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    hash_to_first: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    imported = 0
    duplicate_count = 0

    for raw_source in args.source:
        source_name, pattern = _parse_source_arg(raw_source)
        matches = _expand_patterns([pattern])
        source_dir = raw_root / source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        source_counts[source_name] = len(matches)

        for src in matches:
            rel_name = _safe_name(src.name)
            dst = source_dir / rel_name
            if dst.exists() and not args.force:
                stem = _safe_name(src.stem)
                dst = source_dir / f"{stem}-{_safe_name(src.parent.name)}{src.suffix.lower()}"
            if dst.exists() and not args.force:
                raise SystemExit(f"ERROR: destination exists, use --force: {dst}")

            if use_symlink:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)

            sha = _sha256_file(src)
            dup_of = hash_to_first.get(sha)
            if dup_of is None:
                hash_to_first[sha] = str(src.resolve())
            else:
                duplicate_count += 1

            row = {
                "source_name": source_name,
                "source_path": str(src.resolve()),
                "imported_path": str(dst.resolve()),
                "filename": src.name,
                "bytes": int(src.stat().st_size),
                "sha256": sha,
                "duplicate_of_source_path": dup_of,
            }
            all_rows.append(row)
            imported += 1

    manifest = {
        "schema": "ck.svg_raw_assets_inventory.v1",
        "workspace": str(workspace),
        "raw_assets_root": str(raw_root),
        "mode": "symlink" if use_symlink else "copy",
        "sources": [{"name": k, "matched_files": int(v)} for k, v in sorted(source_counts.items())],
        "imported_files": int(imported),
        "unique_hashes": int(len(hash_to_first)),
        "duplicate_files": int(duplicate_count),
        "entries": all_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] imported={imported} unique={len(hash_to_first)} duplicates={duplicate_count}")
    print(f"[OK] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
