#!/usr/bin/env python3
"""
prepare_run_viewer.py

One-shot operator script that prepares all viewer artifacts for a v7 run:
  1. Export token embeddings  → embeddings.json
  2. Export attention matrices → attention.json
  3. Build dataset viewer     → dataset_viewer.html  (if manifests exist)

After running this, the IR hub will show the Dataset, 🧬 Embeddings, and
🔭 Attention buttons for the run.

Usage:
    python3 version/v7/tools/prepare_run_viewer.py <run_dir>
    python3 version/v7/tools/prepare_run_viewer.py <run_dir> --dry-run
    python3 version/v7/tools/prepare_run_viewer.py <run_dir> --force
    python3 version/v7/tools/prepare_run_viewer.py --all             # every train run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MODELS_ROOT = Path.home() / ".cache" / "ck-engine-v7" / "models"


# ── Checks ────────────────────────────────────────────────────────────────────

def _has_weights(run_dir: Path) -> bool:
    return (run_dir / "weights_manifest.json").exists()


def _has_tokenizer(run_dir: Path) -> bool:
    return (run_dir / "tokenizer.json").exists()


def _has_probe_report(run_dir: Path) -> bool:
    return (run_dir / "probe_report.json").exists()


def _dataset_workspace(run_dir: Path) -> Path | None:
    """Return the dataset workspace if manifests exist."""
    for candidate in [run_dir / "dataset", run_dir]:
        manifest_dir = candidate / "manifests"
        if not manifest_dir.is_dir():
            continue
        required = ["asset_classification_manifest.json"]
        if all((manifest_dir / f).exists() for f in required):
            return candidate
    # Also check dataset_snapshot.json for workspace pointer
    snap = run_dir / "dataset" / "dataset_snapshot.json"
    if snap.exists():
        try:
            data = json.loads(snap.read_text())
            for key in ("working_workspace", "snapshot_root", "source_workspace"):
                ws = data.get(key)
                if ws:
                    ws_path = Path(ws)
                    manifest_dir = ws_path / "manifests"
                    if manifest_dir.is_dir():
                        required = ["asset_classification_manifest.json"]
                        if all((manifest_dir / f).exists() for f in required):
                            return ws_path
        except Exception:
            pass
    return None


# ── Steps ─────────────────────────────────────────────────────────────────────

def run_step(label: str, cmd: list[str], dry_run: bool) -> bool:
    """Run a subprocess step. Returns True on success."""
    cmd_str = " ".join(cmd)
    if dry_run:
        print(f"  [DRY-RUN] {label}")
        print(f"    $ {cmd_str}")
        return True
    print(f"  ▸ {label}")
    print(f"    $ {cmd_str}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"    ✕ FAILED (exit {result.returncode})")
        return False
    print(f"    ✓ done")
    return True


def prepare_run(run_dir: Path, *, force: bool = False, dry_run: bool = False) -> dict:
    """Prepare all viewer artifacts for a single run. Returns status dict."""
    run_dir = run_dir.resolve()
    name = run_dir.name
    status = {"name": name, "embeddings": "skip", "attention": "skip", "viewer": "skip"}

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  {run_dir}")
    print(f"{'='*70}")

    has_wt = _has_weights(run_dir)
    has_tok = _has_tokenizer(run_dir)
    has_probe = _has_probe_report(run_dir)
    ws = _dataset_workspace(run_dir)

    # Diagnostics
    print(f"  weights_manifest.json : {'✓' if has_wt else '✕'}")
    print(f"  tokenizer.json        : {'✓' if has_tok else '✕'}")
    print(f"  probe_report.json     : {'✓' if has_probe else '✕'}")
    print(f"  dataset workspace     : {ws or '✕ (no manifests found)'}")
    print(f"  embeddings.json       : {'exists' if (run_dir / 'embeddings.json').exists() else 'missing'}")
    print(f"  attention.json        : {'exists' if (run_dir / 'attention.json').exists() else 'missing'}")
    print(f"  dataset_viewer.html   : {'exists' if (run_dir / 'dataset_viewer.html').exists() else 'missing'}")
    print()

    # ── 1. Embeddings ─────────────────────────────────────────────────────
    emb_path = run_dir / "embeddings.json"
    if has_wt and (force or not emb_path.exists()):
        ok = run_step(
            "Export embeddings → embeddings.json",
            [sys.executable, str(SCRIPT_DIR / "export_embeddings.py"), str(run_dir)],
            dry_run,
        )
        status["embeddings"] = "ok" if ok else "fail"
    elif emb_path.exists():
        print(f"  ⏭ embeddings.json already exists (use --force to regenerate)")
        status["embeddings"] = "exists"
    else:
        print(f"  ⏭ skipping embeddings: no weights_manifest.json")

    # ── 2. Attention ──────────────────────────────────────────────────────
    attn_path = run_dir / "attention.json"
    if has_wt and has_tok and (force or not attn_path.exists()):
        cmd = [sys.executable, str(SCRIPT_DIR / "export_attention.py"), str(run_dir)]
        if has_probe:
            cmd.append("--probe")
        ok = run_step(
            "Export attention → attention.json",
            cmd,
            dry_run,
        )
        status["attention"] = "ok" if ok else "fail"
    elif attn_path.exists():
        print(f"  ⏭ attention.json already exists (use --force to regenerate)")
        status["attention"] = "exists"
    else:
        missing = []
        if not has_wt: missing.append("weights_manifest.json")
        if not has_tok: missing.append("tokenizer.json")
        print(f"  ⏭ skipping attention: missing {', '.join(missing)}")

    # ── 3. Dataset viewer ─────────────────────────────────────────────────
    viewer_path = run_dir / "dataset_viewer.html"
    if ws and (force or not viewer_path.exists()):
        ok = run_step(
            "Build dataset viewer → dataset_viewer.html",
            [
                sys.executable,
                str(REPO_ROOT / "version" / "v7" / "scripts" / "build_svg_dataset_visualizer_v7.py"),
                "--workspace", str(ws),
                "--output", str(viewer_path),
            ],
            dry_run,
        )
        status["viewer"] = "ok" if ok else "fail"
    elif viewer_path.exists():
        print(f"  ⏭ dataset_viewer.html already exists (use --force to regenerate)")
        status["viewer"] = "exists"
    else:
        print(f"  ⏭ skipping dataset viewer: no workspace with manifests found")
        print(f"    hint: run the materialize pipeline first if this run has dataset/")
        snap = run_dir / "dataset" / "dataset_snapshot.json"
        if snap.exists():
            try:
                data = json.loads(snap.read_text())
                src_ws = data.get("source_workspace") or data.get("working_workspace")
                if src_ws:
                    print(f"    hint: dataset_snapshot.json points to workspace: {src_ws}")
                    print(f"    hint: try: python3 version/v7/scripts/materialize_svg_stage_artifacts_v7.py --workspace {src_ws} --force")
            except Exception:
                pass

    return status


def discover_train_runs(models_root: Path) -> list[Path]:
    """Find all train run dirs."""
    train_root = models_root / "train"
    if not train_root.is_dir():
        return []
    return sorted(
        d for d in train_root.iterdir()
        if d.is_dir() and (d / "weights_manifest.json").exists()
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prepare all viewer artifacts (embeddings, attention, dataset viewer) for v7 runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare one run:
  python3 version/v7/tools/prepare_run_viewer.py ~/.cache/ck-engine-v7/models/train/toy_svg_atoms_ctx512_d64_h128

  # Dry-run to see what would happen:
  python3 version/v7/tools/prepare_run_viewer.py ~/.cache/ck-engine-v7/models/train/toy_svg_atoms_ctx512_d64_h128 --dry-run

  # Force-regenerate everything:
  python3 version/v7/tools/prepare_run_viewer.py ~/.cache/ck-engine-v7/models/train/toy_svg_atoms_ctx512_d64_h128 --force

  # Prepare ALL train runs:
  python3 version/v7/tools/prepare_run_viewer.py --all

  # Regenerate IR hub after preparing:
  python3 version/v7/tools/open_ir_hub.py --open
""",
    )
    ap.add_argument("run_dir", nargs="?", help="Path to a v7 run directory")
    ap.add_argument("--all", action="store_true", help="Process all train runs under ~/.cache/ck-engine-v7/models/")
    ap.add_argument("--force", action="store_true", help="Regenerate even if artifacts already exist")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be done without running anything")
    ap.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT, help="Models root (default: ~/.cache/ck-engine-v7/models)")
    args = ap.parse_args()

    if not args.run_dir and not args.all:
        ap.error("provide a run_dir or use --all")

    if args.all:
        runs = discover_train_runs(args.models_root)
        if not runs:
            print(f"No train runs found under {args.models_root}")
            return 1
        print(f"Found {len(runs)} train runs under {args.models_root}")
    else:
        run_path = Path(args.run_dir).expanduser().resolve()
        if not run_path.is_dir():
            print(f"ERROR: not a directory: {run_path}")
            return 1
        runs = [run_path]

    results = []
    for run_dir in runs:
        status = prepare_run(run_dir, force=args.force, dry_run=args.dry_run)
        results.append(status)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {len(results)} runs processed")
    print(f"{'='*70}")
    for r in results:
        emb = r["embeddings"]
        attn = r["attention"]
        viewer = r["viewer"]
        flags = []
        if emb in ("ok", "exists"): flags.append("🧬")
        if attn in ("ok", "exists"): flags.append("🔭")
        if viewer in ("ok", "exists"): flags.append("📦")
        failed = [k for k in ("embeddings", "attention", "viewer") if r[k] == "fail"]
        status_str = " ".join(flags) if flags else "—"
        fail_str = f" ✕ failed: {', '.join(failed)}" if failed else ""
        print(f"  {r['name']:<55} {status_str}{fail_str}")

    any_changes = any(r[k] in ("ok",) for r in results for k in ("embeddings", "attention", "viewer"))
    if any_changes and not args.dry_run:
        print(f"\n  Tip: regenerate the IR hub to pick up new artifacts:")
        print(f"    python3 version/v7/tools/open_ir_hub.py --open")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
