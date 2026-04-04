#!/usr/bin/env python3
"""Materialize a unified spec19 curriculum for fresh retraining from the frozen base seed."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
ROUTE_RECOVERY_MATERIALIZER = ROOT / "version" / "v7" / "scripts" / "dataset" / "materialize_spec19_route_recovery_replay_v7.py"
LINE_NAME = "spec19_unified_curriculum"
FORMAT_VERSION = "ck.spec19_unified_curriculum.v1"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _build_unified_manifest(
    *,
    workspace: Path,
    prefix: str,
    route_recovery_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "line": LINE_NAME,
        "workspace": str(workspace),
        "prefix": prefix,
        "curriculum_mode": "fresh_retrain_from_frozen_base_seed",
        "derived_from_line": str(route_recovery_manifest.get("line") or ""),
        "source_runs": list(route_recovery_manifest.get("source_runs") or []),
        "stages": dict(route_recovery_manifest.get("stages") or {}),
        "eval_collision_filter": dict(route_recovery_manifest.get("eval_collision_filter") or {}),
    }


def _refresh_manifests(workspace: Path, *, prefix: str) -> dict[str, Any]:
    manifests_dir = workspace / "manifests"
    route_manifest_path = manifests_dir / f"{prefix}_route_recovery_manifest.json"
    workspace_manifest_path = manifests_dir / f"{prefix}_workspace_manifest.json"
    mixture_manifest_path = manifests_dir / f"{prefix}_mixture_manifest.json"
    coherent_manifest_path = manifests_dir / f"{prefix}_coherent_replay_manifest.json"

    route_manifest = _read_json(route_manifest_path)
    workspace_manifest = _read_json(workspace_manifest_path)
    mixture_manifest = _read_json(mixture_manifest_path)
    coherent_manifest = _read_json(coherent_manifest_path)

    unified_manifest = _build_unified_manifest(
        workspace=workspace,
        prefix=prefix,
        route_recovery_manifest=route_manifest,
    )
    unified_manifest_path = manifests_dir / f"{prefix}_unified_curriculum_manifest.json"
    _write_json(unified_manifest_path, unified_manifest)

    route_manifest["line"] = LINE_NAME
    route_manifest["curriculum_mode"] = unified_manifest["curriculum_mode"]
    route_manifest["derived_from_line"] = unified_manifest["derived_from_line"]
    _write_json(route_manifest_path, route_manifest)

    workspace_manifest["line"] = LINE_NAME
    workspace_manifest["unified_curriculum_manifest"] = f"manifests/{prefix}_unified_curriculum_manifest.json"
    for stage_name in ("pretrain", "midtrain"):
        stage = (workspace_manifest.get("stages") or {}).get(stage_name)
        if isinstance(stage, dict):
            stage["notes"] = (
                "Fresh unified spec19 retrain: keep the cumulative winner curriculum intact, append the balanced generalized "
                "recovery delta, dedupe the full corpus, and train again from the frozen spec16 r9 seed instead of continuing "
                "from a later winner rung."
            )
    _write_json(workspace_manifest_path, workspace_manifest)

    coherent_manifest["line"] = LINE_NAME
    _write_json(coherent_manifest_path, coherent_manifest)

    mixture_manifest["line"] = LINE_NAME
    for stage_name in ("pretrain", "midtrain"):
        stage = (mixture_manifest.get("stages") or {}).get(stage_name)
        if isinstance(stage, dict):
            stage["curriculum_mode"] = unified_manifest["curriculum_mode"]
            stage["unified_curriculum_manifest"] = f"manifests/{prefix}_unified_curriculum_manifest.json"
    _write_json(mixture_manifest_path, mixture_manifest)

    return unified_manifest


def materialize_workspace(
    workspace: Path,
    *,
    seed_workspace: Path,
    prefix: str,
    freeze_tokenizer_run: Path,
    source_runs: list[Path],
    weight_quantum: int,
    python_exec: str,
    force: bool,
) -> dict[str, Any]:
    route_recovery = _load_module(ROUTE_RECOVERY_MATERIALIZER, "materialize_spec19_route_recovery_replay_v7")
    workspace = workspace.expanduser().resolve()
    summary = route_recovery.materialize_workspace(
        workspace,
        seed_workspace=seed_workspace,
        prefix=prefix,
        freeze_tokenizer_run=freeze_tokenizer_run,
        source_runs=source_runs,
        weight_quantum=weight_quantum,
        python_exec=python_exec,
        force=force,
    )
    unified_manifest = _refresh_manifests(workspace, prefix=prefix)
    out = dict(summary)
    out["line"] = LINE_NAME
    out["curriculum_mode"] = unified_manifest["curriculum_mode"]
    out["source_runs"] = list(unified_manifest.get("source_runs") or [])
    return out


def main() -> int:
    route_recovery = _load_module(ROUTE_RECOVERY_MATERIALIZER, "materialize_spec19_route_recovery_replay_v7")
    spec19 = route_recovery._load_module(route_recovery.SPEC19_MATERIALIZER, "materialize_spec19_scene_bundle_v7")
    base = spec19._load_base_module()

    ap = argparse.ArgumentParser(description="Materialize a unified spec19 curriculum for fresh retraining")
    ap.add_argument("--workspace", required=True, type=Path, help="Destination workspace")
    ap.add_argument("--seed-workspace", default=str(base.DEFAULT_SEED_WORKSPACE), type=Path, help="Seed spec workspace template to copy")
    ap.add_argument("--prefix", default="spec19_scene_bundle", help="Dataset prefix")
    ap.add_argument("--freeze-tokenizer-run", required=True, type=Path, help="Run directory whose tokenizer should be copied unchanged")
    ap.add_argument("--source-run", action="append", dest="source_runs", type=Path, required=True, help="Completed run whose stage train rows should seed the cumulative curriculum")
    ap.add_argument("--weight-quantum", type=int, default=5, help="Accepted for shared launcher compatibility; recorded in manifests")
    ap.add_argument("--python-exec", default=sys.executable, help="Python executable for generators")
    ap.add_argument("--force", action="store_true", help="Replace workspace if it exists")
    args = ap.parse_args()

    summary = materialize_workspace(
        args.workspace,
        seed_workspace=args.seed_workspace,
        prefix=args.prefix,
        freeze_tokenizer_run=args.freeze_tokenizer_run,
        source_runs=list(args.source_runs or []),
        weight_quantum=int(args.weight_quantum),
        python_exec=str(args.python_exec),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
