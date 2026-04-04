#!/usr/bin/env python3
"""Copy tokenizer artifacts and sidecars into a v7 run directory."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    from tokenizer_policy_v7 import sanitize_tokenizer_doc, visible_special_tokens
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tokenizer_policy_v7 import sanitize_tokenizer_doc, visible_special_tokens


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(str(row).strip() for row in rows if str(row).strip())
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _patch_train_init_artifacts(run_dir: Path, sidecar: dict[str, object]) -> None:
    config_path = run_dir / "train_init_config.json"
    if not config_path.exists():
        return
    try:
        doc = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(doc, dict):
        return
    artifacts = doc.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        doc["artifacts"] = artifacts
    artifacts["tokenizer_json"] = "tokenizer.json"
    artifacts["tokenizer_bin"] = "tokenizer_bin"
    artifacts["tokenizer_sidecar"] = "tokenizer_sidecar.json"
    if (run_dir / "template_train.json").exists():
        artifacts["template_train"] = "template_train.json"
    tokenizer_manifest = sidecar.get("tokenizer_manifest")
    if isinstance(tokenizer_manifest, str) and tokenizer_manifest.strip():
        artifacts["tokenizer_manifest"] = tokenizer_manifest
    reserved_control_tokens = sidecar.get("reserved_control_tokens")
    if isinstance(reserved_control_tokens, str) and reserved_control_tokens.strip():
        artifacts["reserved_control_tokens"] = reserved_control_tokens
    config_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _read_visible_special_tokens(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _discover_tokenizer_manifest(tokenizer_json: Path) -> Path | None:
    matches = sorted(tokenizer_json.parent.glob("*_tokenizer_manifest.json"))
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_manifest_artifact(manifest_doc: dict, manifest_path: Path, key: str) -> Path | None:
    artifacts = manifest_doc.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    raw = artifacts.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None
    artifact_path = Path(raw).expanduser()
    if artifact_path.is_absolute():
        return artifact_path.resolve()
    workspace = manifest_doc.get("workspace")
    if isinstance(workspace, str) and workspace.strip():
        return (Path(workspace).expanduser().resolve() / artifact_path).resolve()
    return (manifest_path.parent / artifact_path).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage custom tokenizer artifacts into a v7 run dir")
    ap.add_argument("--run", required=True, help="Run dir under ~/.cache/ck-engine-v7/models/train")
    ap.add_argument("--tokenizer-json", required=True, help="Path to tokenizer.json")
    ap.add_argument("--tokenizer-bin", required=True, help="Path to tokenizer_bin directory")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    tokenizer_json = Path(args.tokenizer_json).expanduser().resolve()
    tokenizer_bin = Path(args.tokenizer_bin).expanduser().resolve()
    ck_build = run_dir / ".ck_build"

    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")
    if not tokenizer_json.exists():
        raise SystemExit(f"tokenizer.json not found: {tokenizer_json}")
    if not tokenizer_bin.is_dir():
        raise SystemExit(f"tokenizer_bin not found: {tokenizer_bin}")

    tokenizer_doc = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    sanitized_doc, removed_special_tokens = sanitize_tokenizer_doc(tokenizer_doc)
    visible_tokens = visible_special_tokens(sanitized_doc)

    _write_json(run_dir / "tokenizer.json", sanitized_doc)
    _copy_tree(tokenizer_bin, run_dir / "tokenizer_bin")

    tokenizer_manifest = _discover_tokenizer_manifest(tokenizer_json)
    reserved_control_tokens: Path | None = None
    sidecar: dict[str, object] = {
        "format": "ck.tokenizer_sidecar.v1",
        "tokenizer_json": "tokenizer.json",
        "tokenizer_bin": "tokenizer_bin",
        "tokenizer_manifest": None,
        "reserved_control_tokens": None,
        "visible_special_tokens": list(visible_tokens),
        "sanitized_removed_special_tokens": list(removed_special_tokens),
    }

    if tokenizer_manifest is not None and tokenizer_manifest.exists():
        manifest_doc = json.loads(tokenizer_manifest.read_text(encoding="utf-8"))
        _copy_file(tokenizer_manifest, run_dir / "tokenizer_manifest.json")
        sidecar["tokenizer_manifest"] = "tokenizer_manifest.json"
        reserved_control_tokens = _resolve_manifest_artifact(manifest_doc, tokenizer_manifest, "reserved_control_tokens")

    if reserved_control_tokens is None:
        sibling_matches = sorted(tokenizer_json.parent.glob("*_reserved_control_tokens.txt"))
        if len(sibling_matches) == 1:
            reserved_control_tokens = sibling_matches[0]

    if visible_tokens or (reserved_control_tokens is not None and reserved_control_tokens.exists()):
        _write_lines(run_dir / "reserved_control_tokens.txt", visible_tokens)
        sidecar["reserved_control_tokens"] = "reserved_control_tokens.txt"

    _write_json(run_dir / "tokenizer_sidecar.json", sidecar)
    _patch_train_init_artifacts(run_dir, sidecar)

    if ck_build.exists():
        _write_json(ck_build / "tokenizer.json", sanitized_doc)
        _copy_tree(tokenizer_bin, ck_build / "tokenizer_bin")
        if tokenizer_manifest is not None and tokenizer_manifest.exists():
            _copy_file(tokenizer_manifest, ck_build / "tokenizer_manifest.json")
        if visible_tokens or (reserved_control_tokens is not None and reserved_control_tokens.exists()):
            _write_lines(ck_build / "reserved_control_tokens.txt", visible_tokens)
        _write_json(ck_build / "tokenizer_sidecar.json", sidecar)

    print(f"staged tokenizer into {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
