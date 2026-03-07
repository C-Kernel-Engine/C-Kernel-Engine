#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ENTRIES = (
    "README.md",
    "contracts",
    "manifests",
    "raw_assets",
    "normalized",
    "pretrain",
    "midtrain",
    "sft",
    "holdout",
    "tokenizer",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _detect_dataset_type(workspace: Path) -> str:
    contracts_dir = workspace / "contracts"
    if contracts_dir.exists():
        for contract in sorted(contracts_dir.glob("*.json")):
            obj = _load_json(contract)
            if isinstance(obj, dict):
                dataset_type = obj.get("dataset_type")
                if isinstance(dataset_type, str) and dataset_type.strip():
                    return dataset_type.strip().lower()
    return "unknown"


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _stage_entry(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        target = src.resolve()
        dst.symlink_to(target, target_is_directory=src.is_dir())
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _build_dataset_viewer(snapshot_root: Path, output_path: Path, dataset_type: str) -> tuple[str, str]:
    scripts_dir = Path(__file__).resolve().parent
    if dataset_type == "svg":
        builder = scripts_dir / "build_svg_dataset_visualizer_v7.py"
        cmd = [
            sys.executable,
            str(builder),
            "--workspace",
            str(snapshot_root),
            "--output",
            str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"dataset viewer build failed (rc={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip() or 'no output'}"
            )
        return ("svg", "ok")
    return (dataset_type, "unsupported")


def stage_workspace(workspace: Path, run_dir: Path, *, mode: str, force: bool) -> dict[str, Any]:
    workspace = workspace.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"workspace not found: {workspace}")
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    snapshot_root = run_dir / "dataset"
    if snapshot_root.exists():
        if not force:
            raise FileExistsError(f"dataset snapshot already exists: {snapshot_root} (use --force)")
        _remove_existing(snapshot_root)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    dataset_type = _detect_dataset_type(workspace)
    staged_entries: list[str] = []
    missing_entries: list[str] = []
    for name in WORKSPACE_ENTRIES:
        src = workspace / name
        if not src.exists():
            missing_entries.append(name)
            continue
        dst = snapshot_root / name
        _stage_entry(src, dst, mode)
        staged_entries.append(name)

    viewer_path = run_dir / "dataset_viewer.html"
    viewer_status = "not_requested"
    viewer_adapter = dataset_type
    if viewer_path.exists():
        if force:
            viewer_path.unlink()
        else:
            raise FileExistsError(f"dataset viewer already exists: {viewer_path} (use --force)")
    viewer_adapter, viewer_status = _build_dataset_viewer(snapshot_root, viewer_path, dataset_type)

    # Gallery is now integrated into dataset_viewer.html — record path for hub discovery
    gallery_path = run_dir / "svg_gallery.html"
    gallery_status = "integrated"

    snapshot = {
        "schema": "ck.dataset_snapshot.v1",
        "generated_at": _utc_now_iso(),
        "source_workspace": str(workspace),
        "source_workspace_role": "seed_template",
        "working_workspace": str(snapshot_root),
        "snapshot_root": str(snapshot_root),
        "run_dir": str(run_dir),
        "dataset_type": dataset_type,
        "stage_mode": mode,
        "staged_entries": staged_entries,
        "missing_entries": missing_entries,
        "viewer_path": str(viewer_path),
        "viewer_adapter": viewer_adapter,
        "viewer_status": viewer_status,
        "gallery_path": str(gallery_path) if gallery_status == "ok" else None,
        "gallery_status": gallery_status,
    }
    (snapshot_root / "dataset_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage a dataset workspace into a run dir and generate dataset_viewer.html")
    ap.add_argument("--workspace", required=True, type=Path, help="Source dataset workspace (e.g. version/v7/data/spec03)")
    ap.add_argument("--run-dir", required=True, type=Path, help="Run directory to receive the dataset snapshot")
    ap.add_argument("--mode", choices=("copy", "symlink"), default="copy", help="How to stage workspace entries into the run dir")
    ap.add_argument("--force", action="store_true", help="Replace existing run_dir/dataset snapshot and dataset_viewer.html")
    args = ap.parse_args()

    snapshot = stage_workspace(args.workspace, args.run_dir, mode=str(args.mode), force=bool(args.force))
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
