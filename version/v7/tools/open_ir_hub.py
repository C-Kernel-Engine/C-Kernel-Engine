#!/usr/bin/env python3
"""
open_ir_hub.py

Generate a parent "Run Hub" HTML page that indexes v7 run directories under
~/.cache/ck-engine-v7/models (or a custom root). The hub links to each run's
ir_report.html so operators can jump between runs quickly.

Usage:
  python3 version/v7/tools/open_ir_hub.py
  python3 version/v7/tools/open_ir_hub.py --models-root ~/.cache/ck-engine-v7/models
  python3 version/v7/tools/open_ir_hub.py --output /tmp/ir_hub.html --index-out /tmp/runs_index.json
  python3 version/v7/tools/open_ir_hub.py --open

┌──────────────────────────────────────────────────────────────────────────────┐
│  ⚠️  FEATURE MANIFEST — DO NOT DELETE OR REGRESS ANY OF THESE FEATURES  ⚠️   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  This file generates the IR Run Hub HTML. Every feature below is             │
│  operator-critical. Future agents: you may ENHANCE, but NEVER remove.        │
│                                                                              │
│  PYTHON FEATURES:                                                            │
│   • RunRecord dataclass — all run metadata, artifact paths, commands         │
│   • Artifact discovery: IR report, dataset viewer, embeddings, attention     │
│   • Command builders: dataset viewer, export embeddings, export attention    │
│   • _find_embeddings_path / _find_attention_path — artifact finders          │
│   • _build_dataset_viewer_cmd — always returns a command (never None)        │
│   • build_index() — aggregates all runs into hub payload JSON                │
│                                                                              │
│  HTML/JS FEATURES:                                                           │
│   • Antsand brand badge (CK logo + "C-Kernel-Engine v7")                     │
│   • Spotlight card for latest run with full details                          │
│   • Run cards grid with kind badges (train/inference/eval)                   │
│   • Table view toggle for dense listing                                      │
│   • Hero metric ribbon (runs, specs, embeddings, attention counts)           │
│   • Button variants: .btn.dataset (cyan), .btn.emb (purple), .btn.attn      │
│   • ⚡ Operator Commands panel (renderCommandsPanel) with 5 commands:        │
│     - 📦 Build Dataset Viewer                                                │
│     - 🧬 Export Embeddings                                                   │
│     - 🔭 Export Attention Matrices                                           │
│     - 🔍 Open IR Visualizer                                                  │
│     - 📊 Open Dataset Viewer                                                 │
│   • Deep-links: embViewerLink() / attnViewerLink() → repo-root viewer       │
│   • cmdBlock(cmd, label, desc) — copyable command blocks with descriptions   │
│   • Search/filter by run name                                                │
│   • CSS dark theme with orange/cyan/purple/teal accents                      │
│                                                                              │
│  If adding features, follow the existing patterns and update this manifest.  │
│                                                                              │
│  HOW TO EDIT:                                                                │
│   • Python side: RunRecord dataclass + build_index() → hub payload JSON      │
│   • HTML/JS is embedded as a Python f-string in generate_hub_html()          │
│   • Run cards: add fields to RunRecord, surface in card template             │
│   • Commands: add to _build_*_cmd() helpers, surface in renderCommandsPanel  │
│                                                                              │
│  WHAT NOT TO BREAK:                                                          │
│   • run-card CSS class — L1 health checks for this marker                    │
│   • ir_report.html links — L3 validates hub links to ir_report               │
│   • dataset_viewer.html links — hub must show dataset button for runs        │
│   • RunRecord.to_dict() — serialised into embedded JSON for JS               │
│                                                                              │
│  TESTING:                                                                    │
│   make v7-visualizer-health          # L1 checks hub source structure        │
│   make v7-visualizer-generated-e2e   # L3 generates hub + validates          │
│   Pre-push:  .githooks/pre-push step [0.5/6]                                │
│   Nightly:   nightly_runner.py → v7_visualizer_health + generated_e2e       │
│                                                                              │
│  The hub has no JSON contract file (unlike ir_visualizer / dataset_viewer)   │
│  because it's a Python generator, not an HTML template. The L1 checker       │
│  validates: file_exists, hub_marker:run-card, hub_marker:ir_report.html,     │
│  hub_marker:dataset_viewer.html.                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

V7_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V7_ROOT.parent.parent
KERNEL_REGISTRY_PATH = V7_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"

MARKER_FILES = {
    "run_index.json",
    "ir_report.html",
    "dataset_viewer.html",
    "weights_manifest.json",
    "training_pipeline_latest.json",
    "training_parity_regimen_latest.json",
}

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".ck_pipeline",
    "checkpoints",
    "tokenizer_bin",
    "bpe_bin",
}

CKPT_STEP_RE = re.compile(r"weights_step_(\d+)\.bump$")


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _epoch_to_iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _file_mtime(path: Path) -> float | None:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return None


def _infer_kind(run_dir: Path, models_root: Path) -> str:
    rel = _to_rel(run_dir, models_root)
    parts = rel.split("/")
    if parts and parts[0] == "train":
        return "train"
    return "inference"


def _build_generate_report_cmd(run_dir: Path) -> str:
    run_q = shlex.quote(str(run_dir))
    out_q = shlex.quote(str(run_dir / "ir_report.html"))
    return (
        "python3 version/v7/tools/open_ir_visualizer.py "
        f"--generate --run {run_q} --html-only --strict-run-artifacts --output {out_q}"
    )


def _build_export_embeddings_cmd(run_dir: Path) -> str:
    run_q = shlex.quote(str(run_dir))
    return f"python3 version/v7/tools/export_embeddings.py {run_q}"


def _build_export_attention_cmd(run_dir: Path) -> str:
    run_q = shlex.quote(str(run_dir))
    return f"python3 version/v7/tools/export_attention.py {run_q} --probe"


def _build_run_make_cmd(target: str, run_dir: Path) -> str:
    run_q = shlex.quote(str(run_dir))
    return f"make {target} RUN={run_q}"


def _build_prepare_all_cmd(run_dir: Path) -> str:
    run_q = shlex.quote(str(run_dir))
    return f"python3 version/v7/tools/prepare_run_viewer.py {run_q} --force"


def _build_model_make_cmd(target: str, run_dir: Path) -> str:
    run_q = shlex.quote(str(run_dir))
    return f"make {target} V7_MODEL={run_q}"


def _find_report_path(run_dir: Path) -> Path | None:
    direct = run_dir / "ir_report.html"
    if direct.exists():
        return direct
    ck_build = run_dir / ".ck_build" / "ir_report.html"
    if ck_build.exists():
        return ck_build
    return None


def _find_dataset_viewer_path(run_dir: Path) -> Path | None:
    direct = run_dir / "dataset_viewer.html"
    if direct.exists():
        return direct
    nested = run_dir / "dataset" / "dataset_viewer.html"
    if nested.exists():
        return nested
    return None


def _find_embeddings_path(run_dir: Path) -> Path | None:
    for candidate in [run_dir / "embeddings.json", run_dir / "dataset" / "embeddings.json"]:
        if candidate.exists():
            return candidate
    return None


def _find_attention_path(run_dir: Path) -> Path | None:
    for candidate in [run_dir / "attention.json", run_dir / "dataset" / "attention.json"]:
        if candidate.exists():
            return candidate
    return None


def _find_gallery_path(run_dir: Path) -> Path | None:
    direct = run_dir / "svg_gallery.html"
    if direct.exists():
        return direct
    nested = run_dir / "dataset" / "svg_gallery.html"
    if nested.exists():
        return nested
    return None


def _find_dataset_snapshot_path(run_dir: Path) -> Path | None:
    candidate = run_dir / "dataset" / "dataset_snapshot.json"
    if candidate.exists():
        return candidate
    return None


def _dir_has_materialized_files(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_dir():
            return False
        for child in path.rglob("*"):
            if child.is_file() and child.name != ".gitkeep":
                return True
        return False
    except Exception:
        return False


def _build_dataset_materialize_cmd(dataset_workspace: str | None, dataset_type: str | None) -> str | None:
    if not dataset_workspace or not dataset_type:
        return None
    if dataset_type != "svg":
        return None
    ws_q = shlex.quote(str(dataset_workspace))
    return (
        "python3 version/v7/scripts/materialize_svg_stage_artifacts_v7.py "
        f"--workspace {ws_q} --force"
    )


def _build_dataset_viewer_cmd(dataset_workspace: str | None, dataset_type: str | None, run_dir: Path) -> str:
    """Always returns a build command. Uses the detected workspace when available,
    falls back to run_dir/dataset so the operator always has something to copy."""
    if dataset_workspace and dataset_type == "svg":
        ws_q = shlex.quote(str(dataset_workspace))
    else:
        ws_q = shlex.quote(str(run_dir / "dataset"))
    out_q = shlex.quote(str(run_dir / "dataset_viewer.html"))
    return (
        "python3 version/v7/scripts/build_svg_dataset_visualizer_v7.py "
        f"--workspace {ws_q} --output {out_q}"
    )


def _build_dataset_checklist(run_dir: Path, dataset_snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(dataset_snapshot, dict):
        return []
    dataset_root = run_dir / "dataset"
    manifests_root = dataset_root / "manifests"
    checks = [
        (
            "raw_import",
            "Raw asset import",
            _dir_has_materialized_files(dataset_root / "raw_assets")
            and (manifests_root / "raw_assets_inventory.json").exists(),
            "Need source SVG/assets imported into raw_assets and inventoried.",
        ),
        (
            "normalize",
            "Normalize + placeholders",
            _dir_has_materialized_files(dataset_root / "normalized")
            and (manifests_root / "normalized_assets_manifest.json").exists(),
            "Need normalized SVGs with placeholderized text and normalization manifest.",
        ),
        (
            "classify",
            "Classification / split planning",
            (manifests_root / "asset_classification_manifest.json").exists(),
            "Need SVG family/role classification before deriving stage corpora.",
        ),
        (
            "pretrain",
            "Pretrain corpus materialized",
            _dir_has_materialized_files(dataset_root / "pretrain"),
            "Need actual pretrain text/manifests, not just an empty folder.",
        ),
        (
            "midtrain",
            "Midtrain transform corpus materialized",
            _dir_has_materialized_files(dataset_root / "midtrain"),
            "Need transform/edit pairs for layout/style conditioning.",
        ),
        (
            "sft",
            "SFT supervision corpus materialized",
            _dir_has_materialized_files(dataset_root / "sft"),
            "Need tag/spec -> pure SVG supervision rows.",
        ),
        (
            "tokenizer",
            "Tokenizer corpus + fit audit",
            _dir_has_materialized_files(dataset_root / "tokenizer"),
            "Need tokenizer corpus or fit reports for ctx 512/2048.",
        ),
        (
            "holdout",
            "Holdout / canary set finalized",
            _dir_has_materialized_files(dataset_root / "holdout"),
            "Need held-out canary prompts/assets before overnight training.",
        ),
    ]
    return [
        {"key": key, "label": label, "ready": ready, "hint": hint}
        for key, label, ready, hint in checks
    ]


def _find_run_artifact(run_dir: Path, *relative_paths: str) -> Path | None:
    for rel in relative_paths:
        for base in (run_dir, run_dir / ".ck_build"):
            candidate = base / rel
            if candidate.exists():
                return candidate
    return None


def _has_run_artifact(run_dir: Path, *relative_paths: str) -> bool:
    return _find_run_artifact(run_dir, *relative_paths) is not None


def _extract_dims(weights_manifest: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if weights_manifest is None:
        return out
    obj = _safe_read_json(weights_manifest)
    if not isinstance(obj, dict):
        return out
    cfg = obj.get("config")
    if not isinstance(cfg, dict):
        return out
    for key in (
        "num_layers",
        "embed_dim",
        "hidden_size",
        "vocab_size",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "context_len",
    ):
        if key in cfg:
            out[key] = cfg.get(key)
    return out


def _extract_manifest_info(weights_manifest: Path | None) -> tuple[int | None, str | None]:
    if weights_manifest is None:
        return None, None
    obj = _safe_read_json(weights_manifest)
    if not isinstance(obj, dict):
        return None, None
    step = obj.get("step")
    reason = obj.get("reason")
    step_out = int(step) if isinstance(step, (int, float)) else None
    reason_out = str(reason) if isinstance(reason, str) else None
    return step_out, reason_out


def _shape_signature(dims: dict[str, Any]) -> str | None:
    keys = ("vocab_size", "embed_dim", "hidden_size", "num_layers", "num_heads", "num_kv_heads", "head_dim")
    if not all(k in dims for k in keys):
        return None
    return (
        f"v{dims['vocab_size']}_d{dims['embed_dim']}_h{dims['hidden_size']}_"
        f"L{dims['num_layers']}_nh{dims['num_heads']}_kv{dims['num_kv_heads']}_hd{dims['head_dim']}"
    )


def _latest_checkpoint(run_dir: Path) -> tuple[int | None, Path | None, Path | None, int]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None, None, None, 0
    latest_step: int | None = None
    latest_bump: Path | None = None
    count = 0
    for bump in ckpt_dir.glob("weights_step_*.bump"):
        m = CKPT_STEP_RE.match(bump.name)
        if not m:
            continue
        count += 1
        step = int(m.group(1))
        if latest_step is None or step > latest_step:
            latest_step = step
            latest_bump = bump
    if latest_bump is None:
        return None, None, None, count
    latest_manifest = latest_bump.with_name(latest_bump.stem + "_manifest.json")
    if not latest_manifest.exists():
        latest_manifest = None
    return latest_step, latest_bump, latest_manifest, count


def _extract_final_loss(path: Path | None) -> float | None:
    if path is None:
        return None
    obj = _safe_read_json(path)
    if not isinstance(obj, dict):
        return None
    steps = obj.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    last = steps[-1]
    if not isinstance(last, dict):
        return None
    for key in ("loss_ck", "loss", "final_loss"):
        v = last.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _extract_loss_curve_summary(path: Path | None) -> dict[str, Any]:
    """Extract a compact summary of the training loss curve for cross-run comparison.

    Returns dict with start_loss, final_loss, total_steps, convergence_rate,
    and a downsampled sparkline (max 50 points) for hub-level overlay charts.
    """
    empty: dict[str, Any] = {"available": False}
    if path is None:
        return empty
    obj = _safe_read_json(path)
    if not isinstance(obj, dict):
        return empty
    steps = obj.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return empty

    first = steps[0]
    last = steps[-1]
    start_loss = None
    final_loss = None
    for key in ("loss_ck", "loss"):
        if start_loss is None:
            v = first.get(key)
            if isinstance(v, (int, float)):
                start_loss = float(v)
        if final_loss is None:
            v = last.get(key)
            if isinstance(v, (int, float)):
                final_loss = float(v)

    if start_loss is None or final_loss is None:
        return empty

    total_steps = len(steps)
    convergence_rate = (start_loss - final_loss) / total_steps if total_steps > 0 else 0.0
    reduction = start_loss / final_loss if final_loss > 0 else 0.0

    # Sparkline: downsample to ~50 points
    spark = []
    stride = max(1, total_steps // 50)
    for i in range(0, total_steps, stride):
        s = steps[i]
        loss = s.get("loss_ck", s.get("loss"))
        step_num = s.get("step", i)
        if isinstance(loss, (int, float)):
            spark.append({"s": step_num, "l": round(float(loss), 6)})
    # Always include last point
    if spark and spark[-1]["s"] != last.get("step"):
        spark.append({"s": last.get("step", total_steps), "l": round(float(final_loss), 6)})

    return {
        "available": True,
        "start_loss": round(start_loss, 6),
        "final_loss": round(final_loss, 6),
        "total_steps": total_steps,
        "convergence_rate": round(convergence_rate, 8),
        "reduction": round(reduction, 2),
        "final_lr": float(last.get("lr", 0)) if isinstance(last.get("lr"), (int, float)) else None,
        "sparkline": spark,
    }


def _extract_valid_svg_rate(path: Path | None) -> float | None:
    if path is None:
        return None
    obj = _safe_read_json(path)
    if not isinstance(obj, dict):
        return None
    v = obj.get("valid_svg_rate")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _extract_parity_regimen(path: Path | None) -> dict[str, Any]:
    status = {"status": "MISSING", "passed": None, "generated_at": None}
    if path is None:
        return status
    obj = _safe_read_json(path)
    if not isinstance(obj, dict):
        return status
    summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else {}
    passed = summary.get("passed")
    skipped = obj.get("skipped")
    if isinstance(passed, bool):
        status["passed"] = passed
        if passed:
            status["status"] = "PASS" if not skipped else "PASS_REUSED"
        else:
            status["status"] = "FAIL"
    elif skipped:
        status["status"] = "SKIP"
    status["generated_at"] = obj.get("generated_at")
    return status


def _best_eval_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None

    def _score(entry: dict[str, Any]) -> tuple[float, float, float]:
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        return (
            _safe_float(metrics.get("adherence")) or 0.0,
            _safe_float(metrics.get("tag_adherence")) or 0.0,
            _safe_float(metrics.get("valid_svg_rate")) or 0.0,
        )

    return max(entries, key=_score)


def _extract_tokenizer_summary(run_dir: Path) -> dict[str, Any]:
    path = _find_run_artifact(run_dir, "tokenizer_roundtrip.json")
    if path is None:
        return {}
    obj = _safe_read_json(path)
    if not isinstance(obj, dict):
        return {}
    line_eval = obj.get("line_eval") if isinstance(obj.get("line_eval"), dict) else {}
    return {
        "path": str(path),
        "tokenizer_mode": obj.get("tokenizer_mode"),
        "status": obj.get("status"),
        "exact_match": obj.get("exact_match"),
        "byte_match_rate": _safe_float(obj.get("byte_match_rate")),
        "line_match_rate": _safe_float(obj.get("line_match_rate")),
        "token_count": int(obj["token_count"]) if isinstance(obj.get("token_count"), (int, float)) else None,
        "input_lines": int(obj["input_lines"]) if isinstance(obj.get("input_lines"), (int, float)) else None,
        "input_bytes": int(obj["input_bytes"]) if isinstance(obj.get("input_bytes"), (int, float)) else None,
        "coverage_rate": _safe_float(line_eval.get("coverage_rate")),
        "exact_match_rate": _safe_float(line_eval.get("exact_match_rate")),
    }


def _extract_stage_eval_summary(run_dir: Path) -> dict[str, Any]:
    path = _find_run_artifact(run_dir, "stage_eval_matrix.json", "stage_eval_matrix_latest.json")
    if path is None:
        return {}
    obj = _safe_read_json(path)
    if not isinstance(obj, dict):
        return {}
    entries = obj.get("entries")
    if not isinstance(entries, list) or not entries:
        return {}
    sorted_entries = sorted(entries, key=lambda item: int(item.get("run_order") or 0))
    latest = sorted_entries[-1]
    best = _best_eval_entry(sorted_entries)

    def _numeric_metrics(entry: dict[str, Any] | None) -> dict[str, float]:
        metrics = entry.get("metrics") if isinstance(entry, dict) and isinstance(entry.get("metrics"), dict) else {}
        out: dict[str, float] = {}
        for key, value in metrics.items():
            numeric = _safe_float(value)
            if numeric is not None:
                out[key] = numeric
        return out

    return {
        "path": str(path),
        "entry_count": len(sorted_entries),
        "latest_phase": latest.get("phase_label"),
        "best_phase": best.get("phase_label") if isinstance(best, dict) else None,
        "latest_metrics": _numeric_metrics(latest),
        "best_metrics": _numeric_metrics(best),
    }


def _extract_probe_summary(run_dir: Path) -> dict[str, Any]:
    candidates = sorted(run_dir.glob("*probe_report.json"))
    best: dict[str, Any] | None = None
    best_score: tuple[int, int, int] | None = None
    for path in candidates:
        obj = _safe_read_json(path)
        if not isinstance(obj, dict):
            continue
        rows = obj.get("results")
        if not isinstance(rows, list) or not rows:
            continue
        typed_rows = [row for row in rows if isinstance(row, dict)]
        if not typed_rows:
            continue
        holdout_splits = {"holdout", "test", "dev", "val", "validation"}
        holdouts = [row for row in typed_rows if str(row.get("split") or "").strip().lower() in holdout_splits]
        exact = sum(1 for row in typed_rows if row.get("exact_match"))
        renderable = sum(
            1
            for row in typed_rows
            if row.get("renderable") or row.get("materialized_output") or row.get("rendered_svg")
        )
        holdout_exact = sum(1 for row in holdouts if row.get("exact_match"))
        probe_count = len(typed_rows)
        holdout_count = len(holdouts)
        metrics: dict[str, float] = {
            "exact_rate": exact / probe_count if probe_count else 0.0,
            "renderable_rate": renderable / probe_count if probe_count else 0.0,
        }
        if holdout_count:
            metrics["holdout_exact_rate"] = holdout_exact / holdout_count
        summary = {
            "path": str(path),
            "kind": path.stem,
            "probe_count": probe_count,
            "holdout_count": holdout_count,
            "metrics": metrics,
        }
        score = (probe_count, holdout_count, len(metrics))
        if best_score is None or score > best_score:
            best = summary
            best_score = score
    if best is not None:
        return best
    return {}


def _infer_compare_family(run_dir: Path, dataset_type: str | None, kind: str) -> str:
    if dataset_type:
        return str(dataset_type).lower()
    name = run_dir.name.lower()
    hints = [
        ("toy_svg", "svg"),
        ("svg", "svg"),
        ("sql", "sql"),
        ("sqlite", "sql"),
        ("bash", "bash"),
        ("shell", "bash"),
        ("python", "python"),
        ("py_", "python"),
        ("_py", "python"),
        ("c_lang", "c"),
        ("c_code", "c"),
        ("clang", "c"),
        ("llama", "llm"),
        ("qwen", "llm"),
    ]
    for needle, label in hints:
        if needle in name:
            return label
    return kind


@dataclass
class RunRecord:
    run_dir: Path
    rel_path: str
    name: str
    kind: str
    compare_family: str
    report_path: Path | None
    dataset_viewer_path: Path | None
    embeddings_path: Path | None
    attention_path: Path | None
    gallery_path: Path | None
    dataset_snapshot_path: Path | None
    dataset_workspace: str | None
    dataset_type: str | None
    dataset_stage_mode: str | None
    dataset_staged_entries: list[str]
    dataset_missing_entries: list[str]
    dataset_refresh_cmd: str | None
    dataset_rebuild_viewer_cmd: str
    dataset_prep_checklist: list[dict[str, Any]]
    tokenizer_summary: dict[str, Any]
    eval_summary: dict[str, Any]
    probe_summary: dict[str, Any]
    dims: dict[str, Any]
    parity_regimen: dict[str, Any]
    final_loss: float | None
    loss_curve_summary: dict[str, Any]
    valid_svg_rate: float | None
    checkpoint_count: int
    latest_checkpoint_step: int | None
    latest_checkpoint_bump: Path | None
    latest_checkpoint_manifest: Path | None
    weights_step: int | None
    weights_reason: str | None
    shape_signature: str | None
    generate_report_cmd: str
    export_embeddings_cmd: str
    export_attention_cmd: str
    prepare_all_cmd: str
    artifact_sections: list[dict[str, Any]]
    coverage_summary: dict[str, Any]
    next_actions: list[dict[str, str]]
    updated_epoch: float
    updated_iso: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "run_uri": self.run_dir.resolve().as_uri(),
            "rel_path": self.rel_path,
            "name": self.name,
            "kind": self.kind,
            "compare_family": self.compare_family,
            "report_path": str(self.report_path) if self.report_path else None,
            "report_uri": self.report_path.resolve().as_uri() if self.report_path else None,
            "dataset_viewer_path": str(self.dataset_viewer_path) if self.dataset_viewer_path else None,
            "dataset_viewer_uri": self.dataset_viewer_path.resolve().as_uri() if self.dataset_viewer_path else None,
            "embeddings_path": str(self.embeddings_path) if self.embeddings_path else None,
            "embeddings_uri": self.embeddings_path.resolve().as_uri() if self.embeddings_path else None,
            "attention_path": str(self.attention_path) if self.attention_path else None,
            "attention_uri": self.attention_path.resolve().as_uri() if self.attention_path else None,
            "gallery_path": str(self.gallery_path) if self.gallery_path else None,
            "gallery_uri": self.gallery_path.resolve().as_uri() if self.gallery_path else None,
            "dataset_snapshot_path": str(self.dataset_snapshot_path) if self.dataset_snapshot_path else None,
            "dataset_snapshot_uri": self.dataset_snapshot_path.resolve().as_uri() if self.dataset_snapshot_path else None,
            "dataset_workspace": self.dataset_workspace,
            "dataset_type": self.dataset_type,
            "dataset_stage_mode": self.dataset_stage_mode,
            "dataset_staged_entries": self.dataset_staged_entries,
            "dataset_missing_entries": self.dataset_missing_entries,
            "dataset_refresh_cmd": self.dataset_refresh_cmd,
            "dataset_rebuild_viewer_cmd": self.dataset_rebuild_viewer_cmd,
            "dataset_prep_checklist": self.dataset_prep_checklist,
            "tokenizer_summary": self.tokenizer_summary,
            "eval_summary": self.eval_summary,
            "probe_summary": self.probe_summary,
            "dims": self.dims,
            "parity_regimen": self.parity_regimen,
            "final_loss": self.final_loss,
            "loss_curve_summary": self.loss_curve_summary,
            "valid_svg_rate": self.valid_svg_rate,
            "checkpoint_count": self.checkpoint_count,
            "latest_checkpoint_step": self.latest_checkpoint_step,
            "latest_checkpoint_bump": str(self.latest_checkpoint_bump) if self.latest_checkpoint_bump else None,
            "latest_checkpoint_bump_uri": self.latest_checkpoint_bump.resolve().as_uri() if self.latest_checkpoint_bump else None,
            "latest_checkpoint_manifest": str(self.latest_checkpoint_manifest) if self.latest_checkpoint_manifest else None,
            "latest_checkpoint_manifest_uri": self.latest_checkpoint_manifest.resolve().as_uri() if self.latest_checkpoint_manifest else None,
            "weights_step": self.weights_step,
            "weights_reason": self.weights_reason,
            "shape_signature": self.shape_signature,
            "generate_report_cmd": self.generate_report_cmd,
            "export_embeddings_cmd": self.export_embeddings_cmd,
            "export_attention_cmd": self.export_attention_cmd,
            "prepare_all_cmd": self.prepare_all_cmd,
            "artifact_sections": self.artifact_sections,
            "coverage_summary": self.coverage_summary,
            "next_actions": self.next_actions,
            "updated_epoch": self.updated_epoch,
            "updated_iso": self.updated_iso,
        }


def _build_section(
    key: str,
    title: str,
    items: list[tuple[str, bool]],
    *,
    core: bool,
    optional: bool = False,
) -> dict[str, Any]:
    payload_items = [{"label": label, "ready": ready} for label, ready in items]
    present = sum(1 for _, ready in items if ready)
    total = len(items)
    return {
        "key": key,
        "title": title,
        "core": core,
        "optional": optional,
        "present": present,
        "total": total,
        "items": payload_items,
    }


def _coverage_counts(sections: list[dict[str, Any]], *, core: bool) -> tuple[int, int]:
    target = [section for section in sections if bool(section.get("core")) is core]
    present = sum(int(section.get("present") or 0) for section in target)
    total = sum(int(section.get("total") or 0) for section in target)
    return present, total


def _coverage_pct(present: int, total: int) -> int:
    if total <= 0:
        return 0
    return max(0, min(100, round((present / total) * 100)))


def _build_run_coverage(
    run_dir: Path,
    kind: str,
    report_ready: bool,
    dataset_viewer_ready: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    if kind == "train":
        sections = [
            _build_section("surface", "Report Surface", [("report", report_ready)], core=True),
            _build_section(
                "compile",
                "Compile-time",
                [
                    ("ir1_train", _has_run_artifact(run_dir, "ir1_train_forward.json")),
                    ("ir2_train", _has_run_artifact(run_dir, "ir2_train_backward.json", "ir2_train_summary.json")),
                    ("layout_train", _has_run_artifact(run_dir, "layout_train.json")),
                    ("train_exec_plan", _has_run_artifact(run_dir, "train_exec_plan.json")),
                    ("gen_runtime", _has_run_artifact(run_dir, "generated_train_runtime_summary.json")),
                ],
                core=True,
            ),
            _build_section(
                "data_pipeline",
                "Data / Pipeline",
                [
                    ("training_pipeline", _has_run_artifact(run_dir, "training_pipeline_latest.json")),
                    ("dataset_qc", _has_run_artifact(run_dir, "dataset_qc.json")),
                    ("tokenizer_roundtrip", _has_run_artifact(run_dir, "tokenizer_roundtrip.json")),
                    ("run_ledger", _has_run_artifact(run_dir, "run_ledger.jsonl")),
                    ("stage_eval_matrix", _has_run_artifact(run_dir, "stage_eval_matrix_latest.json", "stage_eval_matrix.json")),
                    ("analysis_checkpoints", _has_run_artifact(run_dir, "analysis_checkpoints_latest.json")),
                ],
                core=True,
            ),
            _build_section(
                "training_runs",
                "Training Runs",
                [
                    ("loss_curve", _has_run_artifact(run_dir, "training_loss_curve_latest.json")),
                    ("grad_norms", _has_run_artifact(run_dir, "training_grad_norms_latest.json")),
                    ("parity", _has_run_artifact(run_dir, "training_parity_latest.json")),
                    ("parity_regimen", _has_run_artifact(run_dir, "training_parity_regimen_latest.json")),
                    ("post_train_eval", _has_run_artifact(run_dir, "post_train_eval.json")),
                ],
                core=True,
            ),
            _build_section(
                "runtime_perf",
                "Runtime / Perf",
                [
                    ("step_profile", _has_run_artifact(run_dir, "training_step_profile.json")),
                    ("backprop_stitch", _has_run_artifact(run_dir, "backprop_stitch_runtime_latest.json")),
                    ("replay_determinism", _has_run_artifact(run_dir, "replay_determinism_latest.json")),
                    ("replay_accum", _has_run_artifact(run_dir, "replay_accum_latest.json")),
                    ("memory_diag", _has_run_artifact(run_dir, "memory_diagnostic_latest.json")),
                    ("train_e2e", _has_run_artifact(run_dir, "train_e2e_latest.json")),
                    ("layout_audit", _has_run_artifact(run_dir, "layout_train_audit.json")),
                ],
                core=False,
            ),
            _build_section(
                "dataset_surface",
                "Dataset Surface",
                [("dataset_viewer", dataset_viewer_ready)],
                core=False,
                optional=True,
            ),
        ]
        next_actions: list[dict[str, str]] = []
        if not report_ready:
            next_actions.append({"label": "Generate report", "cmd": _build_generate_report_cmd(run_dir)})
        if any(section["present"] < section["total"] for section in sections if section["core"]):
            next_actions.append({"label": "Refresh core training artifacts", "cmd": _build_run_make_cmd("v7-capture-artifacts-run", run_dir)})
        if any(section["present"] < section["total"] for section in sections if section["key"] == "runtime_perf"):
            next_actions.append({"label": "Capture training runtime telemetry", "cmd": _build_run_make_cmd("v7-profile-dashboard-run", run_dir)})
    else:
        sections = [
            _build_section("surface", "Report Surface", [("report", report_ready)], core=True),
            _build_section(
                "compile",
                "Compile-time",
                [
                    ("ir1", _has_run_artifact(run_dir, "ir1_decode.json", "ir1_prefill.json")),
                    ("layout", _has_run_artifact(run_dir, "layout_decode.json", "layout_prefill.json")),
                    ("lowered", _has_run_artifact(run_dir, "lowered_decode_call.json", "lowered_decode.json", "lowered_prefill_call.json", "lowered_prefill.json")),
                    ("manifest", _has_run_artifact(run_dir, "weights_manifest.json")),
                    ("kernel_registry", KERNEL_REGISTRY_PATH.exists()),
                ],
                core=True,
            ),
            _build_section(
                "basic_profiling",
                "Basic Profiling",
                [
                    ("profile", _has_run_artifact(run_dir, "profile_summary.json")),
                    ("perf_stat", _has_run_artifact(run_dir, "perf_stat_summary.json")),
                    ("flamegraph", _has_run_artifact(run_dir, "flamegraph_manifest.json")),
                ],
                core=True,
            ),
            _build_section(
                "correctness",
                "Correctness",
                [
                    ("memory_signoff", _has_run_artifact(run_dir, "memory_signoff.json")),
                    ("perf_gate", _has_run_artifact(run_dir, "perf_gate_report.json")),
                ],
                core=False,
            ),
            _build_section(
                "deep_profiling",
                "Deep Profiling",
                [
                    ("cachegrind", _has_run_artifact(run_dir, "cachegrind_summary.json")),
                    ("vtune", _has_run_artifact(run_dir, "vtune_summary.json")),
                    ("advisor", _has_run_artifact(run_dir, "advisor_summary.json")),
                ],
                core=False,
            ),
            _build_section(
                "optional_train_runtime",
                "Optional Train-runtime",
                [("asan", _has_run_artifact(run_dir, "asan_summary.json"))],
                core=False,
                optional=True,
            ),
        ]
        next_actions = []
        if not report_ready:
            next_actions.append({"label": "Generate report", "cmd": _build_generate_report_cmd(run_dir)})
        compile_section = next(section for section in sections if section["key"] == "compile")
        basic_section = next(section for section in sections if section["key"] == "basic_profiling")
        correctness_section = next(section for section in sections if section["key"] == "correctness")
        if compile_section["present"] < compile_section["total"]:
            next_actions.append({"label": "Refresh compile artifacts", "cmd": _build_run_make_cmd("v7-capture-artifacts-run", run_dir)})
        if basic_section["present"] < basic_section["total"]:
            next_actions.append({"label": "Capture profiling dashboard", "cmd": _build_run_make_cmd("v7-profile-dashboard-run", run_dir)})
        if not _has_run_artifact(run_dir, "memory_signoff.json"):
            next_actions.append({"label": "Run memory signoff", "cmd": _build_model_make_cmd("v7-memory-signoff", run_dir)})
        if not _has_run_artifact(run_dir, "perf_gate_report.json"):
            next_actions.append({"label": "Evaluate perf gate", "cmd": _build_model_make_cmd("v7-perf-gate-evaluate", run_dir)})

    core_present, core_total = _coverage_counts(sections, core=True)
    advanced_present, advanced_total = _coverage_counts(sections, core=False)
    coverage_summary = {
        "core_present": core_present,
        "core_total": core_total,
        "core_pct": _coverage_pct(core_present, core_total),
        "advanced_present": advanced_present,
        "advanced_total": advanced_total,
        "advanced_pct": _coverage_pct(advanced_present, advanced_total) if advanced_total else None,
        "core_label": "core dashboard coverage",
        "advanced_label": "advanced checks",
    }
    return sections, coverage_summary, next_actions


def discover_run_dirs(models_root: Path) -> list[Path]:
    runs: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(models_root):
        # Prune noisy/non-run subtrees.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        if not (MARKER_FILES.intersection(filenames)):
            continue
        p = Path(dirpath).resolve()
        # If marker sits in .ck_build, map to parent run dir.
        if p.name == ".ck_build":
            p = p.parent
        # Exclude nested non-run internals.
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        runs.add(p)
    return sorted(runs)


def collect_run_record(run_dir: Path, models_root: Path) -> RunRecord:
    rel = _to_rel(run_dir, models_root)
    report = _find_report_path(run_dir)
    dataset_viewer = _find_dataset_viewer_path(run_dir)
    embeddings = _find_embeddings_path(run_dir)
    attention = _find_attention_path(run_dir)
    gallery = _find_gallery_path(run_dir)
    dataset_snapshot_path = _find_dataset_snapshot_path(run_dir)
    dataset_snapshot = _safe_read_json(dataset_snapshot_path) if dataset_snapshot_path else None
    wm = _find_run_artifact(run_dir, "weights_manifest.json")
    parity = _find_run_artifact(run_dir, "training_parity_regimen_latest.json")
    loss = _find_run_artifact(run_dir, "training_loss_curve_latest.json", "training_loss_curve.json")
    post_eval = _find_run_artifact(run_dir, "post_train_eval.json")
    run_index = run_dir / "run_index.json"

    dims = _extract_dims(wm)
    weights_step, weights_reason = _extract_manifest_info(wm)
    parity_status = _extract_parity_regimen(parity)
    final_loss = _extract_final_loss(loss)
    loss_curve_summary = _extract_loss_curve_summary(loss)
    valid_svg_rate = _extract_valid_svg_rate(post_eval)
    latest_ckpt_step, latest_ckpt_bump, latest_ckpt_manifest, ckpt_count = _latest_checkpoint(run_dir)

    mtimes = []
    for p in (report, wm, parity, loss, post_eval, run_index):
        m = _file_mtime(p)
        if m is not None:
            mtimes.append(m)
    updated_epoch = max(mtimes) if mtimes else 0.0
    updated_iso = _epoch_to_iso(updated_epoch)
    kind = _infer_kind(run_dir, models_root)
    artifact_sections, coverage_summary, next_actions = _build_run_coverage(
        run_dir,
        kind,
        bool(report),
        bool(dataset_viewer),
    )
    dataset_workspace = None
    if isinstance(dataset_snapshot, dict):
        for key in ("working_workspace", "snapshot_root", "source_workspace"):
            value = dataset_snapshot.get(key)
            if value:
                dataset_workspace = str(value)
                break
    dataset_type = str(dataset_snapshot.get("dataset_type")) if isinstance(dataset_snapshot, dict) and dataset_snapshot.get("dataset_type") else None
    dataset_stage_mode = str(dataset_snapshot.get("stage_mode")) if isinstance(dataset_snapshot, dict) and dataset_snapshot.get("stage_mode") else None
    dataset_staged_entries = [str(v) for v in dataset_snapshot.get("staged_entries", [])] if isinstance(dataset_snapshot, dict) else []
    dataset_missing_entries = [str(v) for v in dataset_snapshot.get("missing_entries", [])] if isinstance(dataset_snapshot, dict) else []
    dataset_refresh_cmd = _build_dataset_materialize_cmd(dataset_workspace, dataset_type)
    dataset_rebuild_viewer_cmd = _build_dataset_viewer_cmd(dataset_workspace, dataset_type, run_dir)
    dataset_prep_checklist = _build_dataset_checklist(run_dir, dataset_snapshot)
    compare_family = _infer_compare_family(run_dir, dataset_type, kind)
    tokenizer_summary = _extract_tokenizer_summary(run_dir)
    eval_summary = _extract_stage_eval_summary(run_dir)
    probe_summary = _extract_probe_summary(run_dir)

    if dataset_refresh_cmd:
        next_actions.append({"label": "Materialize staged dataset artifacts", "cmd": dataset_refresh_cmd})
    if dataset_rebuild_viewer_cmd:
        next_actions.append({"label": "Rebuild dataset viewer", "cmd": dataset_rebuild_viewer_cmd})

    return RunRecord(
        run_dir=run_dir,
        rel_path=rel,
        name=run_dir.name,
        kind=kind,
        compare_family=compare_family,
        report_path=report,
        dataset_viewer_path=dataset_viewer,
        embeddings_path=embeddings,
        attention_path=attention,
        gallery_path=gallery,
        dataset_snapshot_path=dataset_snapshot_path,
        dataset_workspace=dataset_workspace,
        dataset_type=dataset_type,
        dataset_stage_mode=dataset_stage_mode,
        dataset_staged_entries=dataset_staged_entries,
        dataset_missing_entries=dataset_missing_entries,
        dataset_refresh_cmd=dataset_refresh_cmd,
        dataset_rebuild_viewer_cmd=dataset_rebuild_viewer_cmd,
        dataset_prep_checklist=dataset_prep_checklist,
        tokenizer_summary=tokenizer_summary,
        eval_summary=eval_summary,
        probe_summary=probe_summary,
        dims=dims,
        parity_regimen=parity_status,
        final_loss=final_loss,
        loss_curve_summary=loss_curve_summary,
        valid_svg_rate=valid_svg_rate,
        checkpoint_count=ckpt_count,
        latest_checkpoint_step=latest_ckpt_step,
        latest_checkpoint_bump=latest_ckpt_bump,
        latest_checkpoint_manifest=latest_ckpt_manifest,
        weights_step=weights_step,
        weights_reason=weights_reason,
        shape_signature=_shape_signature(dims),
        generate_report_cmd=_build_generate_report_cmd(run_dir),
        export_embeddings_cmd=_build_export_embeddings_cmd(run_dir),
        export_attention_cmd=_build_export_attention_cmd(run_dir),
        prepare_all_cmd=_build_prepare_all_cmd(run_dir),
        artifact_sections=artifact_sections,
        coverage_summary=coverage_summary,
        next_actions=next_actions,
        updated_epoch=updated_epoch,
        updated_iso=updated_iso,
    )


def build_index(models_root: Path) -> dict[str, Any]:
    runs = [collect_run_record(r, models_root) for r in discover_run_dirs(models_root)]
    runs.sort(key=lambda r: r.updated_epoch, reverse=True)
    payload_runs = [r.to_json() for r in runs]
    train_count = sum(1 for r in payload_runs if r.get("kind") == "train")
    report_count = sum(1 for r in payload_runs if r.get("report_path"))
    dataset_viewer_count = sum(1 for r in payload_runs if r.get("dataset_viewer_path"))
    embeddings_count = sum(1 for r in payload_runs if r.get("embeddings_path"))
    attention_count = sum(1 for r in payload_runs if r.get("attention_path"))
    pass_count = sum(1 for r in payload_runs if (r.get("parity_regimen") or {}).get("status") in ("PASS", "PASS_REUSED"))

    global_viewer = REPO_ROOT / "dataset_viewer.html"
    global_viewer_uri = global_viewer.resolve().as_uri() if global_viewer.exists() else None

    return {
        "schema": "ck.ir.hub.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_root": str(models_root),
        "global_viewer_uri": global_viewer_uri,
        "summary": {
            "runs_total": len(payload_runs),
            "runs_train": train_count,
            "runs_with_report": report_count,
            "runs_with_dataset_viewer": dataset_viewer_count,
            "runs_with_embeddings": embeddings_count,
            "runs_with_attention": attention_count,
            "runs_parity_pass": pass_count,
        },
        "runs": payload_runs,
    }


def render_html(index_payload: dict[str, Any]) -> str:
    data_json = json.dumps(index_payload, ensure_ascii=False)
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CK v7 Run Hub</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-0: #171717;
      --bg-1: #1d1d1d;
      --bg-2: #232323;
      --panel: rgba(27, 27, 27, 0.84);
      --panel-2: rgba(34, 34, 34, 0.88);
      --panel-3: rgba(23, 23, 23, 0.94);
      --line: rgba(255, 180, 0, 0.14);
      --line-strong: rgba(7, 173, 248, 0.26);
      --text: #f5f5f5;
      --muted: #c3c3c3;
      --dim: #8a8a8a;
      --gold: #ffb400;
      --gold-soft: rgba(255, 180, 0, 0.14);
      --cyan: #07adf8;
      --cyan-soft: rgba(7, 173, 248, 0.14);
      --violet: #787878;
      --violet-soft: rgba(120, 120, 120, 0.16);
      --green: #57d89c;
      --green-soft: rgba(87, 216, 156, 0.16);
      --red: #ff6f7f;
      --red-soft: rgba(255, 111, 127, 0.16);
      --warn: #ffd071;
      --shadow-xl: 0 30px 90px rgba(0, 0, 0, 0.48);
      --shadow-lg: 0 20px 56px rgba(0, 0, 0, 0.34);
      --radius-xl: 32px;
      --radius-lg: 24px;
      --radius-md: 18px;
      --radius-sm: 14px;
    }

    * { box-sizing: border-box; }

    html { scroll-behavior: smooth; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
      background:
        radial-gradient(circle at 12% 14%, rgba(7, 173, 248, 0.08), transparent 24%),
        radial-gradient(circle at 83% 12%, rgba(255, 180, 0, 0.08), transparent 26%),
        radial-gradient(circle at 76% 82%, rgba(255, 180, 0, 0.035), transparent 22%),
        linear-gradient(180deg, #141414 0%, var(--bg-1) 42%, #111111 100%);
      line-height: 1.5;
      overflow-x: hidden;
    }

    body::before,
    body::after {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
    }

    body::before {
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.032) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.032) 1px, transparent 1px);
      background-size: 44px 44px;
      mask-image: radial-gradient(circle at center, black 28%, transparent 88%);
      opacity: 0.14;
    }

    body::after {
      background:
        radial-gradient(circle at 50% 0%, rgba(255, 255, 255, 0.035), transparent 28%),
        radial-gradient(circle at 50% 100%, rgba(70, 219, 255, 0.025), transparent 30%);
      opacity: 0.22;
    }

    body.table-mode {
      background:
        linear-gradient(180deg, #151515 0%, #1b1b1b 38%, #121212 100%);
    }

    body.table-mode::before {
      background-size: 64px 64px;
      mask-image: none;
      opacity: 0.06;
    }

    body.table-mode::after {
      opacity: 0.12;
    }

    body.compact-mode {
      line-height: 1.38;
    }

    a { color: inherit; text-decoration: none; }

    .page-shell {
      position: relative;
      z-index: 1;
      width: min(1520px, calc(100vw - 40px));
      margin: 28px auto 52px auto;
    }

    body.table-mode .page-shell {
      width: min(1760px, calc(100vw - 24px));
      margin-top: 16px;
    }

    .hero,
    .panel,
    .run-card {
      border: 1px solid var(--line);
      box-shadow: var(--shadow-lg);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
    }

    .hero {
      position: relative;
      overflow: hidden;
      padding: 30px;
      border-radius: var(--radius-xl);
      background:
        linear-gradient(140deg, rgba(40, 40, 40, 0.96) 0%, rgba(29, 29, 29, 0.97) 54%, rgba(22, 22, 22, 0.96) 100%);
      box-shadow: var(--shadow-xl);
      isolation: isolate;
    }

    .hero::before {
      content: "";
      position: absolute;
      inset: auto -18% -42% 28%;
      height: 400px;
      background: radial-gradient(circle, rgba(255, 180, 0, 0.09), transparent 64%);
      opacity: 0.4;
      z-index: -1;
    }

    .hero::after {
      content: "";
      position: absolute;
      inset: -10% 56% auto -14%;
      height: 420px;
      background: radial-gradient(circle, rgba(7, 173, 248, 0.08), transparent 62%);
      opacity: 0.34;
      z-index: -1;
    }

    body.table-mode .hero {
      padding: 18px 20px;
      border-radius: 24px;
      box-shadow: 0 12px 34px rgba(0, 0, 0, 0.26);
    }

    body.table-mode .hero::before,
    body.table-mode .hero::after {
      opacity: 0.22;
    }

    .brand-badge {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 18px;
    }
    .brand-mark {
      width: 34px;
      height: 34px;
      border-radius: 8px;
      background: var(--gold);
      color: #111;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      font-weight: 900;
      font-family: 'Space Grotesk', sans-serif;
      flex-shrink: 0;
    }
    .brand-label {
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--muted);
      letter-spacing: 0.06em;
    }
    .brand-label span { color: var(--gold); }

    .hero-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.88fr);
      gap: 22px;
      align-items: stretch;
    }

    .eyebrow {
      margin: 0 0 12px 0;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 0.72rem;
      font-weight: 700;
      color: var(--cyan);
    }

    .headline {
      margin: 0;
      font-size: clamp(2.15rem, 3.7vw, 3.8rem);
      line-height: 0.92;
      letter-spacing: -0.05em;
      max-width: 10ch;
    }

    body.table-mode .headline {
      font-size: clamp(1.9rem, 2.4vw, 2.85rem);
      max-width: none;
    }

    .headline .accent {
      display: block;
      background: linear-gradient(135deg, #e8e8e8 0%, #d2c08f 46%, #86bfd6 100%);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }

    .lede {
      margin: 14px 0 0 0;
      max-width: 58ch;
      color: var(--muted);
      font-size: 0.88rem;
    }

    body.table-mode .lede {
      margin-top: 10px;
      max-width: 72ch;
      font-size: 0.82rem;
    }

    .hero-tags,
    .hero-meta,
    .result-pills,
    .run-badges,
    .spotlight-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .hero-tags { margin-top: 16px; }
    .hero-meta { margin-top: 16px; }

    body.table-mode .hero-tags {
      display: none;
    }

    body.table-mode .hero-meta {
      margin-top: 10px;
      gap: 8px;
    }

    .tag,
    .badge,
    .result-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 26px;
      padding: 0 9px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.022);
      color: var(--muted);
      font-size: 0.65rem;
      font-weight: 600;
      letter-spacing: 0.03em;
      white-space: nowrap;
    }

    .tag strong,
    .result-pill strong {
      color: var(--text);
      font-weight: 700;
    }

    .hero-orbital {
      position: relative;
      min-height: 370px;
      border-radius: 28px;
      border: 1px solid rgba(255, 255, 255, 0.06);
      background:
        linear-gradient(180deg, rgba(44, 44, 44, 0.88), rgba(24, 24, 24, 0.96)),
        linear-gradient(135deg, rgba(255,180,0,0.028), rgba(7,173,248,0.032));
      overflow: hidden;
    }

    body.table-mode .hero-grid {
      grid-template-columns: minmax(0, 1fr) 260px;
      gap: 16px;
    }

    body.table-mode .hero-orbital {
      min-height: 210px;
      border-radius: 22px;
      background:
        linear-gradient(180deg, rgba(34, 34, 34, 0.9), rgba(23, 23, 23, 0.96));
    }

    .hero-orbital::before,
    .hero-orbital::after {
      content: "";
      position: absolute;
      border-radius: 50%;
      border: 1px solid rgba(255, 255, 255, 0.08);
      inset: 50%;
      transform: translate(-50%, -50%);
    }

    .hero-orbital::before {
      width: 320px;
      height: 320px;
      animation: spin 24s linear infinite;
    }

    .hero-orbital::after {
      width: 220px;
      height: 220px;
      border-style: dashed;
      animation: spinReverse 18s linear infinite;
    }

    body.table-mode .hero-orbital::before {
      width: 220px;
      height: 220px;
    }

    body.table-mode .hero-orbital::after {
      width: 150px;
      height: 150px;
    }

    .orbital-core {
      position: absolute;
      inset: 50%;
      transform: translate(-50%, -50%);
      width: 164px;
      height: 164px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      text-align: center;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background:
        radial-gradient(circle at 30% 30%, rgba(70, 219, 255, 0.1), transparent 46%),
        radial-gradient(circle at 70% 70%, rgba(255, 189, 74, 0.1), transparent 42%),
        rgba(10, 18, 34, 0.94);
      box-shadow:
        0 0 0 14px rgba(255, 255, 255, 0.015),
        0 14px 34px rgba(0, 0, 0, 0.34);
    }

    body.table-mode .orbital-core {
      width: 120px;
      height: 120px;
      box-shadow:
        0 0 0 10px rgba(255, 255, 255, 0.015),
        0 12px 30px rgba(0, 0, 0, 0.28);
    }

    .orbital-core .value {
      display: block;
      font-size: 2.8rem;
      line-height: 1;
      letter-spacing: -0.05em;
      font-weight: 700;
    }

    body.table-mode .orbital-core .value {
      font-size: 2.1rem;
    }

    .orbital-core .label {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
    }

    .orbit-node {
      position: absolute;
      min-width: 122px;
      padding: 10px 12px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(8, 15, 28, 0.84);
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
      animation: bob 5.6s ease-in-out infinite;
    }

    body.table-mode .orbit-node {
      min-width: 96px;
      padding: 8px 10px;
      border-radius: 13px;
      box-shadow: none;
    }

    .orbit-node:nth-child(3) { animation-delay: -1.2s; }
    .orbit-node:nth-child(4) { animation-delay: -2.8s; }
    .orbit-node:nth-child(5) { animation-delay: -4.3s; }

    .orbit-node .k {
      color: var(--dim);
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      margin-bottom: 4px;
      font-weight: 700;
    }

    .orbit-node .v {
      font-size: 1rem;
      font-weight: 700;
      line-height: 1.1;
    }

    .node-a { top: 40px; right: 26px; }
    .node-b { bottom: 42px; right: 50px; }
    .node-c { left: 30px; bottom: 66px; }

    body.table-mode .node-a { top: 20px; right: 18px; }
    body.table-mode .node-b { bottom: 18px; right: 26px; }
    body.table-mode .node-c { left: 18px; bottom: 34px; }

    .metric-ribbon {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 14px;
      margin-top: 24px;
    }

    body.compact-mode .metric-ribbon {
      gap: 10px;
      margin-top: 18px;
    }

    body.table-mode .metric-ribbon {
      gap: 12px;
      margin-top: 18px;
    }

    .metric-card {
      position: relative;
      overflow: hidden;
      padding: 18px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.014));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }

    body.compact-mode .metric-card {
      padding: 13px 14px;
    }

    body.table-mode .metric-card {
      padding: 14px 16px;
      box-shadow: none;
      background: linear-gradient(180deg, rgba(255,255,255,0.022), rgba(255,255,255,0.01));
    }

    .metric-card::before {
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 3px;
      background: linear-gradient(180deg, rgba(255, 180, 0, 0.78), rgba(7, 173, 248, 0.72));
      opacity: 0.82;
    }

    .metric-label {
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.15em;
      font-weight: 700;
    }

    .metric-value {
      font-size: clamp(1.5rem, 2.2vw, 2.4rem);
      line-height: 1;
      letter-spacing: -0.05em;
      font-weight: 700;
    }

    body.table-mode .metric-value {
      font-size: clamp(1.2rem, 1.55vw, 1.7rem);
    }

    .metric-note {
      margin-top: 8px;
      color: var(--dim);
      font-size: 0.72rem;
    }

    body.compact-mode .metric-note {
      font-size: 0.68rem;
      margin-top: 6px;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 22px;
      margin-top: 26px;
    }

    body.compact-mode .workspace {
      gap: 16px;
      margin-top: 20px;
    }

    body.table-mode .workspace {
      grid-template-columns: 1fr;
      gap: 16px;
      margin-top: 18px;
    }

    .stack {
      display: grid;
      gap: 22px;
      align-content: start;
    }

    body.compact-mode .stack {
      gap: 16px;
    }

    .panel {
      border-radius: var(--radius-lg);
      background: linear-gradient(180deg, rgba(36, 36, 36, 0.9), rgba(24, 24, 24, 0.94));
      overflow: hidden;
    }

    body.table-mode .panel {
      box-shadow: none;
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      background: linear-gradient(180deg, rgba(31, 31, 31, 0.94), rgba(21, 21, 21, 0.96));
    }

    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      flex-wrap: wrap;
      padding: 20px 22px 0 22px;
    }

    .panel-head h2,
    .panel-head h3 {
      margin: 0;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--muted);
    }

    .panel-sub {
      margin: 8px 0 0 0;
      padding: 0 22px;
      color: var(--muted);
      font-size: 0.88rem;
    }

    body.table-mode .panel-head {
      padding: 16px 18px 0 18px;
    }

    body.table-mode .panel-sub {
      padding: 0 18px;
      font-size: 0.82rem;
    }

    .toolbar-body,
    .spotlight-body,
    .rail-body {
      padding: 20px 22px 22px 22px;
    }

    body.compact-mode .toolbar-body,
    body.compact-mode .spotlight-body,
    body.compact-mode .rail-body {
      padding: 16px 17px 18px 17px;
    }

    body.table-mode .toolbar-body,
    body.table-mode .spotlight-body,
    body.table-mode .rail-body {
      padding: 16px 18px 18px 18px;
    }

    .control-grid {
      display: grid;
      grid-template-columns: 1.35fr repeat(7, minmax(0, 1fr));
      gap: 10px;
    }

    body.table-mode .control-grid {
      grid-template-columns: 1.7fr repeat(7, minmax(110px, 1fr));
      gap: 8px;
    }

    body.compact-mode .control-grid {
      gap: 8px;
    }

    .field {
      display: grid;
      gap: 6px;
    }

    .field label {
      color: var(--muted);
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-weight: 700;
    }

    .field input,
    .field select {
      width: 100%;
      min-width: 0;
      padding: 13px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.035);
      color: var(--text);
      font: inherit;
      outline: none;
      transition: 160ms ease;
    }

    body.compact-mode .field input,
    body.compact-mode .field select {
      padding: 11px 12px;
      border-radius: 12px;
      font-size: 0.82rem;
    }

    body.table-mode .field input,
    body.table-mode .field select {
      padding: 11px 12px;
      border-radius: 12px;
      font-size: 0.84rem;
    }

    .field input:focus,
    .field select:focus {
      border-color: rgba(255, 189, 74, 0.78);
      box-shadow: 0 0 0 3px rgba(255, 189, 74, 0.12);
      background: rgba(255, 255, 255, 0.05);
    }

    .result-pills {
      margin-top: 14px;
    }

    body.compact-mode .result-pills {
      margin-top: 10px;
      gap: 8px;
    }

    .result-pill {
      background: rgba(255, 255, 255, 0.03);
    }

    .detail-toggle {
      border-top: 1px solid rgba(255,255,255,0.04);
      padding-top: 14px;
      margin-top: 2px;
    }

    .detail-toggle summary {
      list-style: none;
      cursor: pointer;
      color: var(--cyan);
      font-size: 0.74rem;
      font-weight: 600;
      letter-spacing: 0.03em;
      opacity: 0.8;
      transition: opacity 160ms;
    }

    .detail-toggle summary:hover {
      opacity: 1;
    }

    .detail-toggle summary::-webkit-details-marker {
      display: none;
    }

    .detail-toggle[open] summary {
      margin-bottom: 12px;
    }

    .spotlight-shell {
      display: grid;
      grid-template-columns: minmax(0, 1.06fr) 290px;
      gap: 18px;
    }

    body.table-mode #spotlightPanel {
      display: none;
    }

    .spotlight-main {
      display: grid;
      gap: 14px;
    }

    .spotlight-title-row,
    .run-header {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      flex-wrap: wrap;
    }

    .spotlight-title {
      margin: 0;
      font-size: clamp(1.6rem, 2.4vw, 2.4rem);
      line-height: 0.96;
      letter-spacing: -0.05em;
      word-break: break-word;
    }

    .spotlight-path,
    .run-path,
    .run-footer {
      color: var(--muted);
      font-size: 0.84rem;
      word-break: break-word;
    }

    .spotlight-summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }

    .summary-tile,
    .mini-tile {
      padding: 14px 15px;
      border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      background: rgba(255, 255, 255, 0.025);
    }

    .summary-tile .k,
    .mini-tile .k,
    .run-stat .k,
    .kv-row .k {
      color: var(--dim);
      font-size: 0.64rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-weight: 700;
      margin-bottom: 5px;
    }

    .summary-tile .v,
    .mini-tile .v,
    .run-stat .v {
      font-size: 0.96rem;
      line-height: 1.2;
      font-weight: 700;
      word-break: break-word;
    }

    .spotlight-command {
      margin-top: 4px;
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.05);
      background: rgba(4, 8, 16, 0.78);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.77rem;
      color: #dce6fb;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .spotlight-side {
      padding: 16px;
      border-radius: 20px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      background:
        linear-gradient(180deg, rgba(14, 21, 36, 0.82), rgba(9, 14, 23, 0.92));
      display: grid;
      gap: 12px;
      align-content: start;
    }

    .health-ring {
      --ring: 0deg;
      width: 150px;
      height: 150px;
      margin: 4px auto 2px auto;
      border-radius: 50%;
      background:
        radial-gradient(circle at center, rgba(7, 12, 22, 0.94) 0 55%, transparent 56%),
        conic-gradient(from -90deg, rgba(255, 180, 0, 0.82) 0deg, rgba(7, 173, 248, 0.78) var(--ring), rgba(255,255,255,0.07) var(--ring), rgba(255,255,255,0.07) 360deg);
      display: grid;
      place-items: center;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
    }

    .health-core {
      width: 100px;
      height: 100px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: rgba(8, 13, 24, 0.94);
      border: 1px solid rgba(255, 255, 255, 0.06);
      text-align: center;
    }

    .health-core strong {
      display: block;
      font-size: 2rem;
      line-height: 1;
      letter-spacing: -0.05em;
    }

    .health-core span {
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      margin-top: 5px;
    }

    .spotlight-side .mini-grid {
      display: grid;
      gap: 10px;
    }

    .action-row,
    .run-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 38px;
      padding: 0 13px;
      border-radius: 12px;
      border: 1px solid rgba(255, 180, 0, 0.12);
      background: rgba(255,255,255,0.02);
      color: #e0e0e0;
      font: inherit;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: 160ms ease;
    }

    .btn:hover {
      transform: translateY(-1px);
      border-color: rgba(255, 189, 74, 0.24);
      background: rgba(255,255,255,0.036);
    }

    .btn.primary {
      border-color: rgba(255, 189, 74, 0.18);
      background: linear-gradient(180deg, rgba(255, 189, 74, 0.09), rgba(255, 189, 74, 0.035));
      color: #eadbbb;
    }
    .btn.dataset {
      border-color: rgba(7, 173, 248, 0.36);
      background: linear-gradient(180deg, rgba(7, 173, 248, 0.11), rgba(7, 173, 248, 0.04));
      color: #a8dff7;
    }
    .btn.dataset:hover { background: rgba(7, 173, 248, 0.18); border-color: var(--cyan); }
    .btn.emb {
      border-color: rgba(179, 136, 255, 0.36);
      background: linear-gradient(180deg, rgba(179, 136, 255, 0.11), rgba(179, 136, 255, 0.04));
      color: #d4bbff;
    }
    .btn.emb:hover { background: rgba(179, 136, 255, 0.2); border-color: #b388ff; }
    .btn.attn {
      border-color: rgba(77, 208, 225, 0.36);
      background: linear-gradient(180deg, rgba(77, 208, 225, 0.11), rgba(77, 208, 225, 0.04));
      color: #a5eaf2;
    }
    .btn.attn:hover { background: rgba(77, 208, 225, 0.2); border-color: #4dd0e1; }

    .btn.compare-on {
      border-color: rgba(7, 173, 248, 0.34);
      background: linear-gradient(180deg, rgba(7, 173, 248, 0.14), rgba(7, 173, 248, 0.05));
      color: #cfefff;
    }

    .compare-layout {
      display: grid;
      gap: 16px;
    }

    .compare-toolbar {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
    }

    .compare-toolbar .meta-block {
      display: grid;
      gap: 8px;
    }

    .compare-hints,
    .compare-actions,
    .compare-badges {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .compare-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }

    .compare-section {
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 18px;
      background: rgba(255,255,255,0.02);
      padding: 16px;
    }

    .compare-section h3 {
      margin: 0 0 10px 0;
      font-size: 1rem;
    }

    .compare-table-wrap {
      overflow-x: auto;
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 16px;
      background: rgba(12, 12, 12, 0.48);
    }

    .compare-table {
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }

    .compare-table th,
    .compare-table td {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      text-align: left;
      vertical-align: top;
      font-size: 0.82rem;
    }

    .compare-table thead th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: rgba(20, 20, 20, 0.96);
      color: #f5f5f5;
    }

    .compare-table tbody th {
      min-width: 210px;
      color: var(--muted);
      font-weight: 600;
      background: rgba(255,255,255,0.015);
    }

    .compare-group-row td {
      background: rgba(255, 180, 0, 0.06);
      color: #f8e8ba;
      font-size: 0.74rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      border-top: 1px solid rgba(255, 180, 0, 0.12);
    }

    .compare-mini {
      color: var(--dim);
      font-size: 0.74rem;
    }

    .run-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 20px;
    }

    body.compact-mode .run-grid {
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 14px;
    }

    body.table-mode .run-grid {
      display: block;
    }

    .table-wrap {
      overflow: auto;
      border-radius: 22px;
      border: 1px solid rgba(255,255,255,0.05);
      background: rgba(18, 18, 18, 0.82);
    }

    body.table-mode .table-wrap {
      border-radius: 18px;
      background: rgba(17, 17, 17, 0.96);
      box-shadow: none;
    }

    .run-table {
      width: 100%;
      border-collapse: collapse;
      min-width: 1360px;
    }

    .run-table th,
    .run-table td {
      padding: 11px 12px;
      text-align: left;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      vertical-align: top;
      font-size: 0.76rem;
    }

    body.compact-mode .run-table th,
    body.compact-mode .run-table td {
      padding: 9px 10px;
      font-size: 0.73rem;
    }

    .run-table th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: rgba(29, 29, 29, 0.98);
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.66rem;
    }

    .run-table td:first-child,
    .run-table th:first-child {
      position: sticky;
      left: 0;
    }

    .run-table th:first-child {
      z-index: 3;
      background: rgba(29, 29, 29, 1);
    }

    .run-table td:first-child {
      z-index: 2;
      background: rgba(17, 17, 17, 0.98);
    }

    .run-table tr:hover td {
      background: rgba(255,255,255,0.025);
    }

    .run-table tr:hover td:first-child {
      background: rgba(28, 28, 28, 0.98);
    }

    .run-table .col-run { width: 300px; }
    .run-table .col-model { width: 260px; }
    .run-table .col-status { width: 190px; }
    .run-table .col-metrics { width: 150px; }
    .run-table .col-weights { width: 220px; }
    .run-table .col-updated { width: 180px; }
    .run-table .col-actions { width: 180px; }

    .table-primary {
      color: var(--text);
      font-weight: 700;
      line-height: 1.2;
      word-break: break-word;
    }

    .table-secondary {
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.72rem;
      line-height: 1.35;
      word-break: break-word;
    }

    .table-secondary.tight {
      margin-top: 2px;
      color: var(--dim);
    }

    .table-badges {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .table-metric {
      display: grid;
      gap: 6px;
    }

    .table-metric .row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
    }

    .table-metric .label {
      color: var(--dim);
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      font-weight: 700;
    }

    .table-metric .value {
      color: var(--text);
      font-weight: 700;
      font-size: 0.76rem;
    }

    .table-reason {
      display: inline-block;
      margin-top: 6px;
      max-width: 100%;
      color: var(--dim);
      font-size: 0.68rem;
      line-height: 1.35;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      vertical-align: top;
    }

    .run-card {
      position: relative;
      overflow: hidden;
      border-radius: 24px;
      background:
        linear-gradient(180deg, rgba(17, 22, 29, 0.97), rgba(11, 15, 20, 0.99));
      transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
    }

    .run-card:hover {
      transform: translateY(-4px);
      border-color: rgba(255, 189, 74, 0.34);
      box-shadow: 0 20px 44px rgba(0, 0, 0, 0.32);
    }

    .run-card::before {
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 3px;
      background: linear-gradient(90deg, rgba(7, 173, 248, 0.5), rgba(255, 180, 0, 0.44), rgba(120, 120, 120, 0.24));
      opacity: 0.54;
    }

    .run-card-body {
      padding: 22px 20px;
      display: grid;
      gap: 16px;
    }

    body.compact-mode .run-card-body {
      padding: 16px 15px;
      gap: 12px;
    }

    .run-name {
      margin: 0;
      font-size: 1.04rem;
      line-height: 1.18;
      letter-spacing: -0.03em;
      word-break: break-word;
      max-height: 2.6em;
      overflow: hidden;
    }

    body.compact-mode .run-name {
      font-size: 0.98rem;
    }

    .badge.pass { color: #8fd1ab; background: rgba(87, 216, 156, 0.08); border-color: rgba(87, 216, 156, 0.12); }
    .badge.fail { color: #d798a3; background: rgba(255, 111, 127, 0.08); border-color: rgba(255, 111, 127, 0.12); }
    .badge.skip { color: #d6c191; background: rgba(255, 208, 113, 0.08); border-color: rgba(255, 208, 113, 0.12); }
    .badge.missing { color: #b8afd1; background: rgba(120, 120, 120, 0.08); border-color: rgba(120, 120, 120, 0.12); }
    .badge.report { color: #8ebed0; background: rgba(7, 173, 248, 0.08); border-color: rgba(7, 173, 248, 0.12); }
    .badge.train { color: #d4bb84; background: rgba(255, 180, 0, 0.08); border-color: rgba(255, 180, 0, 0.12); }
    .badge.inference { color: #adb9cb; background: rgba(149, 176, 227, 0.08); border-color: rgba(149, 176, 227, 0.12); }

    .run-health {
      display: grid;
      gap: 10px;
      padding-top: 2px;
    }

    .health-note {
      color: #7d8796;
      font-size: 0.72rem;
      line-height: 1.4;
    }

    .section-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 2px;
    }

    .action-stack {
      display: grid;
      gap: 8px;
      margin-top: 6px;
      padding-top: 10px;
      border-top: 1px solid rgba(255,255,255,0.04);
    }

    .action-row-inline {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 9px 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.02);
      border: 1px solid rgba(255,255,255,0.04);
    }

    .action-label {
      color: #aab4c1;
      font-size: 0.76rem;
      font-weight: 600;
      line-height: 1.35;
    }

    .health-bar,
    .coverage-bar {
      width: 100%;
      height: 8px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.04);
      overflow: hidden;
      box-shadow: inset 0 1px 2px rgba(0,0,0,0.22);
    }

    .health-bar span,
    .coverage-bar span {
      display: block;
      height: 100%;
      border-radius: inherit;
    }

    .health-bar span {
      background: linear-gradient(90deg, rgba(68, 128, 150, 0.74), rgba(138, 130, 84, 0.64), rgba(106, 142, 117, 0.64));
    }

    .coverage-bar span {
      background: linear-gradient(90deg, rgba(72, 136, 161, 0.76), rgba(155, 135, 82, 0.68));
    }

    .run-stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }

    body.compact-mode .run-stats {
      gap: 7px;
    }

    .run-stat {
      padding: 11px 13px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.04);
      background: rgba(255,255,255,0.018);
    }

    body.compact-mode .run-stat {
      padding: 10px 11px;
      border-radius: 12px;
    }

    .run-spec {
      padding: 12px 13px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.05);
      background:
        linear-gradient(135deg, rgba(70, 219, 255, 0.028), rgba(255, 189, 74, 0.03)),
        rgba(255,255,255,0.02);
      color: var(--muted);
      font-size: 0.86rem;
    }

    body.compact-mode .run-spec {
      padding: 10px 11px;
      font-size: 0.8rem;
    }

    .codebox {
      padding: 13px 14px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.05);
      background: rgba(3, 7, 14, 0.8);
      color: #dce6fb;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.74rem;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-x: auto;
    }

    .run-path {
      color: var(--dim);
      font-size: 0.72rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 100%;
    }

    .run-card .coverage-label {
      color: #a8afb7;
      font-size: 0.74rem;
      font-weight: 600;
    }

    .run-card .detail-toggle {
      border-top-color: rgba(255,255,255,0.035);
      padding-top: 12px;
    }

    .run-card .detail-toggle summary {
      color: #91a6b0;
      font-size: 0.72rem;
      font-weight: 600;
    }

    .rail {
      display: grid;
      gap: 22px;
      align-content: start;
    }

    body.table-mode .rail {
      display: none;
    }

    .rail-card {
      position: sticky;
      top: 22px;
    }

    .rail-card + .rail-card,
    .rail-card + .rail-card + .rail-card {
      position: static;
    }

    .kv-stack {
      display: grid;
      gap: 10px;
    }

    .kv-row {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      color: var(--muted);
      font-size: 0.85rem;
    }

    .kv-row .v,
    .coverage-label strong {
      color: var(--text);
      font-weight: 700;
      text-align: right;
    }

    .coverage-stack,
    .legend-list {
      display: grid;
      gap: 12px;
    }

    .coverage-item {
      display: grid;
      gap: 8px;
    }

    .coverage-label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 600;
    }

    .legend-item {
      display: grid;
      grid-template-columns: 12px minmax(0, 1fr);
      gap: 10px;
      align-items: start;
      color: var(--muted);
      font-size: 0.84rem;
    }

    .legend-dot {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      margin-top: 4px;
    }

    .empty {
      padding: 34px 24px;
      text-align: center;
      color: var(--muted);
    }

    .muted { color: var(--muted); }

    .mono {
      font-family: 'JetBrains Mono', monospace;
      word-break: break-word;
    }

    .table-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    body.compact-mode .btn {
      min-height: 36px;
      padding: 0 11px;
      border-radius: 11px;
      font-size: 0.78rem;
    }

    body.table-mode .table-actions .btn {
      min-height: 34px;
      padding: 0 10px;
      border-radius: 10px;
      font-size: 0.74rem;
    }

    @keyframes spin {
      from { transform: translate(-50%, -50%) rotate(0deg); }
      to { transform: translate(-50%, -50%) rotate(360deg); }
    }

    @keyframes spinReverse {
      from { transform: translate(-50%, -50%) rotate(360deg); }
      to { transform: translate(-50%, -50%) rotate(0deg); }
    }

    @keyframes bob {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-8px); }
    }

    @media (max-width: 1280px) {
      .hero-grid,
      .workspace,
      .spotlight-shell {
        grid-template-columns: 1fr;
      }

      .metric-ribbon {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }

      .control-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }

      .rail {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }

      .rail-card,
      .rail-card + .rail-card,
      .rail-card + .rail-card + .rail-card {
        position: static;
      }

      body.table-mode .hero-grid {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 900px) {
      .page-shell {
        width: min(100vw - 18px, 100%);
        margin: 12px auto 28px auto;
      }

      .hero {
        padding: 22px;
      }

      .metric-ribbon,
      .control-grid,
      .spotlight-summary,
      .run-stats,
      .rail {
        grid-template-columns: 1fr;
      }

      .hero-orbital {
        min-height: 320px;
      }
    }
  </style>
</head>
<body>
  <div class="page-shell">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <div class="brand-badge">
            <div class="brand-mark">C</div>
            <span class="brand-label">C-Kernel-Engine <span>v7</span></span>
          </div>
          <p class="eyebrow">Research Run Hub</p>
          <h1 class="headline">Run Hub<span class="accent">Operations Deck</span></h1>
          <p class="lede">Scan every v7 run, spot weak artifacts fast, and jump straight into the right report or terminal action.</p>
          <div class="hero-tags">
            <span class="tag"><strong>Ops</strong> fast triage</span>
            <span class="tag"><strong>Parity</strong> health scan</span>
            <span class="tag"><strong>Reports</strong> direct entry</span>
          </div>
          <div class="hero-meta" id="heroMeta"></div>
        </div>
        <aside class="hero-orbital">
          <div class="orbital-core">
            <div>
              <span class="value" id="orbitalCoreValue">0</span>
              <span class="label">Indexed Runs</span>
            </div>
          </div>
          <div class="orbit-node node-a">
            <div class="k">Freshest</div>
            <div class="v" id="orbitFreshest">n/a</div>
          </div>
          <div class="orbit-node node-b">
            <div class="k">Reports</div>
            <div class="v" id="orbitReports">n/a</div>
          </div>
          <div class="orbit-node node-c">
            <div class="k">Parity Pass</div>
            <div class="v" id="orbitParity">n/a</div>
          </div>
        </aside>
      </div>
      <div class="metric-ribbon" id="metricRibbon"></div>
    </section>

    <div class="workspace">
      <main class="stack">
        <section class="panel">
          <div class="panel-head">
            <div>
              <h2>Live Controls</h2>
              <p class="panel-sub">Slice the run atlas by type, parity, report presence, SVG telemetry, or freshness.</p>
            </div>
            <div class="muted" id="resultSummary"></div>
          </div>
          <div class="toolbar-body">
            <div class="control-grid">
              <div class="field">
                <label for="searchInput">Search</label>
                <input id="searchInput" type="text" placeholder="run name, path, shape signature, weights reason..." />
              </div>
              <div class="field">
                <label for="kindFilter">Kind</label>
                <select id="kindFilter">
                  <option value="all">All</option>
                  <option value="train">Train</option>
                  <option value="inference">Inference</option>
                </select>
              </div>
              <div class="field">
                <label for="parityFilter">Parity</label>
                <select id="parityFilter">
                  <option value="all">All</option>
                  <option value="pass">PASS / PASS_REUSED</option>
                  <option value="fail">FAIL</option>
                  <option value="missing">Missing / Skip</option>
                </select>
              </div>
              <div class="field">
                <label for="reportFilter">Report</label>
                <select id="reportFilter">
                  <option value="all">All</option>
                  <option value="present">Report ready</option>
                  <option value="missing">Missing</option>
                </select>
              </div>
              <div class="field">
                <label for="svgFilter">SVG Eval</label>
                <select id="svgFilter">
                  <option value="all">All</option>
                  <option value="present">Has SVG telemetry</option>
                  <option value="strong">SVG rate &gt;= 80%</option>
                  <option value="missing">Missing</option>
                </select>
              </div>
              <div class="field">
                <label for="sortSelect">Sort</label>
                <select id="sortSelect">
                  <option value="updated_desc">Newest first</option>
                  <option value="updated_asc">Oldest first</option>
                  <option value="loss_asc">Lowest loss first</option>
                  <option value="health_desc">Highest health first</option>
                  <option value="checkpoints_desc">Most checkpoints</option>
                  <option value="name_asc">Name A-Z</option>
                </select>
              </div>
              <div class="field">
                <label for="viewSelect">View</label>
                <select id="viewSelect">
                  <option value="cards">Cards</option>
                  <option value="table">Table</option>
                </select>
              </div>
              <div class="field">
                <label for="densitySelect">Density</label>
                <select id="densitySelect">
                  <option value="comfortable">Comfortable</option>
                  <option value="compact">Compact</option>
                </select>
              </div>
            </div>
            <div class="result-pills" id="resultPills"></div>
          </div>
        </section>

        <section class="panel" id="spotlightPanel"></section>
        <section class="panel" id="comparePanel"></section>

        <section id="runGrid" class="run-grid"></section>
        <section id="emptyState" class="panel empty" hidden>
          No runs match the current filters. Widen the query or relax parity/report constraints.
        </section>
      </main>

      <aside class="rail">
        <section class="panel rail-card" id="coveragePanel"></section>
        <section class="panel rail-card" id="legendPanel"></section>
        <section class="panel rail-card" id="actionPanel"></section>
      </aside>
    </div>
  </div>

  <script>
    const HUB = __HUB_DATA__;

    const els = {
      heroMeta: document.getElementById('heroMeta'),
      orbitalCoreValue: document.getElementById('orbitalCoreValue'),
      orbitFreshest: document.getElementById('orbitFreshest'),
      orbitReports: document.getElementById('orbitReports'),
      orbitParity: document.getElementById('orbitParity'),
      metricRibbon: document.getElementById('metricRibbon'),
      searchInput: document.getElementById('searchInput'),
      kindFilter: document.getElementById('kindFilter'),
      parityFilter: document.getElementById('parityFilter'),
      reportFilter: document.getElementById('reportFilter'),
      svgFilter: document.getElementById('svgFilter'),
      sortSelect: document.getElementById('sortSelect'),
      viewSelect: document.getElementById('viewSelect'),
      densitySelect: document.getElementById('densitySelect'),
      resultSummary: document.getElementById('resultSummary'),
      resultPills: document.getElementById('resultPills'),
      spotlightPanel: document.getElementById('spotlightPanel'),
      comparePanel: document.getElementById('comparePanel'),
      runGrid: document.getElementById('runGrid'),
      emptyState: document.getElementById('emptyState'),
      coveragePanel: document.getElementById('coveragePanel'),
      legendPanel: document.getElementById('legendPanel'),
      actionPanel: document.getElementById('actionPanel'),
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function embViewerLink(run, label) {
      if (!run.embeddings_uri) return '';
      const href = HUB.global_viewer_uri
        ? `${HUB.global_viewer_uri}?tab=embeddings&embUrl=${encodeURIComponent(run.embeddings_uri)}`
        : run.embeddings_uri;
      return `<a class="btn emb" target="_blank" rel="noopener" href="${escapeHtml(href)}">${label}</a>`;
    }

    function attnViewerLink(run, label) {
      if (!run.attention_uri) return '';
      const href = HUB.global_viewer_uri
        ? `${HUB.global_viewer_uri}?tab=attention&attnUrl=${encodeURIComponent(run.attention_uri)}`
        : run.attention_uri;
      return `<a class="btn attn" target="_blank" rel="noopener" href="${escapeHtml(href)}">${label}</a>`;
    }

    function cmdBlock(cmd, label, desc) {
      if (!cmd) return '';
      return `<div style="margin-top:8px;">
        <div style="font-size:11px;font-weight:700;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px;">${label}</div>
        ${desc ? `<div style="font-size:11px;color:var(--muted);margin-bottom:4px;">${desc}</div>` : ''}
        <div class="spotlight-command" style="display:flex;align-items:center;gap:8px;">
          <code style="flex:1;overflow:auto">${escapeHtml(cmd)}</code>
          <button class="btn" style="flex-shrink:0" data-copy="${encodeURIComponent(cmd)}">Copy</button>
        </div>
      </div>`;
    }

    function renderCommandsPanel(run) {
      return `
        <div style="margin-top:14px;border:1px solid rgba(255,180,0,0.14);border-radius:12px;padding:14px 16px;background:rgba(20,20,20,0.6);">
          <div style="font-size:12px;font-weight:700;color:var(--gold);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">⚡ Operator Commands</div>
          ${cmdBlock(run.generate_report_cmd, '📊 Generate IR Report', 'Runs the full IR visualizer pipeline and writes ir_report.html into the run directory.')}
          ${cmdBlock(run.datasetRebuildViewerCmd, '📦 Build Dataset Viewer', 'Reads workspace manifests (raw inventory, normalized, classified) and generates a standalone dataset_viewer.html for this run.')}
          ${run.datasetRefreshCmd ? cmdBlock(run.datasetRefreshCmd, '🔄 Materialize Dataset', 'Stages raw SVG assets, normalizes them, and writes split manifests into the workspace.') : ''}
          ${cmdBlock(run.export_embeddings_cmd, '🧬 Export Embeddings', 'Extracts the token embedding matrix from the latest checkpoint into embeddings.json — view in the Dataset Viewer Embeddings tab.')}
          ${cmdBlock(run.export_attention_cmd, '🔭 Export Attention', 'Runs a full forward pass on probe sequences and saves per-layer/per-head attention matrices to attention.json.')}
          ${cmdBlock(run.prepare_all_cmd, '🚀 Prepare All Viewer Artifacts', 'One-click: generates embeddings.json + attention.json + dataset_viewer.html. Use --force to regenerate existing.')}
        </div>
      `;
    }

    function fmtInt(value) {
      return typeof value === 'number' && Number.isFinite(value)
        ? value.toLocaleString()
        : 'n/a';
    }

    function fmtLoss(value) {
      return typeof value === 'number' && Number.isFinite(value)
        ? value.toFixed(value < 1 ? 4 : 3)
        : 'n/a';
    }

    function fmtPct(value) {
      return typeof value === 'number' && Number.isFinite(value)
        ? `${(value * 100).toFixed(1)}%`
        : 'n/a';
    }

    function fmtPercentWhole(value) {
      return typeof value === 'number' && Number.isFinite(value)
        ? `${Math.round(value)}%`
        : 'n/a';
    }

    function relativeTime(epoch) {
      if (typeof epoch !== 'number' || !Number.isFinite(epoch) || epoch <= 0) {
        return 'unknown';
      }
      const delta = Math.max(0, Date.now() / 1000 - epoch);
      if (delta < 60) return `${Math.max(1, Math.round(delta))}s ago`;
      if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
      if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
      if (delta < 86400 * 30) return `${Math.round(delta / 86400)}d ago`;
      return `${Math.round(delta / (86400 * 30))}mo ago`;
    }

    function makeModelSpec(dims) {
      const bits = [];
      if (typeof dims.num_layers === 'number') bits.push(`L${dims.num_layers}`);
      if (typeof dims.embed_dim === 'number') bits.push(`d${dims.embed_dim}`);
      if (typeof dims.hidden_size === 'number') bits.push(`h${dims.hidden_size}`);
      if (typeof dims.num_heads === 'number') bits.push(`${dims.num_heads} heads`);
      if (typeof dims.context_len === 'number') bits.push(`ctx${dims.context_len}`);
      return bits.length ? bits.join(' / ') : 'model dims unavailable';
    }

    function parityTone(status) {
      if (status === 'PASS' || status === 'PASS_REUSED') return 'pass';
      if (status === 'FAIL') return 'fail';
      if (status === 'SKIP') return 'skip';
      return 'missing';
    }

    function describeCoverageSummary(coverageSummary, sections, core) {
      const target = Array.isArray(sections) ? sections.filter((section) => Boolean(section.core) === core) : [];
      if (!target.length) return core ? 'core coverage unavailable' : 'no advanced checks tracked';
      return target.map((section) => `${section.title.toLowerCase()} ${section.present}/${section.total}`).join(' · ');
    }

    function showParityBadge(run) {
      if (run.kind === 'train') return true;
      return run.parityStatus === 'PASS' || run.parityStatus === 'PASS_REUSED' || run.parityStatus === 'FAIL';
    }

    function makeSearchBlob(run) {
      return [
        run.name,
        run.rel_path,
        run.kind,
        run.compare_family,
        run.datasetType,
        run.shape_signature,
        run.modelSpec,
        run.parityStatus,
        run.weights_reason,
        run.updated_iso,
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
    }

    const runs = Array.isArray(HUB.runs) ? HUB.runs.map((run) => {
      const dims = run.dims || {};
      const parity = run.parity_regimen || {};
      const parityStatus = String(parity.status || 'MISSING').toUpperCase();
      const normalized = {
        ...run,
        dims,
        kind: run.kind || 'inference',
        parityStatus,
        artifactSections: Array.isArray(run.artifact_sections) ? run.artifact_sections : [],
        coverageSummary: (run.coverage_summary && typeof run.coverage_summary === 'object') ? run.coverage_summary : {},
        nextActions: Array.isArray(run.next_actions) ? run.next_actions : [],
        reportReady: Boolean(run.report_path),
        datasetViewerReady: Boolean(run.dataset_viewer_path),
        galleryReady: Boolean(run.gallery_path),
        datasetSnapshotReady: Boolean(run.dataset_snapshot_path),
        datasetWorkspace: run.dataset_workspace || '',
        datasetType: run.dataset_type || '',
        datasetStageMode: run.dataset_stage_mode || '',
        datasetStagedEntries: Array.isArray(run.dataset_staged_entries) ? run.dataset_staged_entries : [],
        datasetMissingEntries: Array.isArray(run.dataset_missing_entries) ? run.dataset_missing_entries : [],
        datasetRefreshCmd: run.dataset_refresh_cmd || '',
        datasetRebuildViewerCmd: run.dataset_rebuild_viewer_cmd || '',
        datasetPrepChecklist: Array.isArray(run.dataset_prep_checklist) ? run.dataset_prep_checklist : [],
        compareFamily: run.compare_family || '',
        tokenizerSummary: (run.tokenizer_summary && typeof run.tokenizer_summary === 'object') ? run.tokenizer_summary : {},
        evalSummary: (run.eval_summary && typeof run.eval_summary === 'object') ? run.eval_summary : {},
        probeSummary: (run.probe_summary && typeof run.probe_summary === 'object') ? run.probe_summary : {},
        validSvgRate: typeof run.valid_svg_rate === 'number' ? run.valid_svg_rate : null,
        checkpointCount: typeof run.checkpoint_count === 'number' ? run.checkpoint_count : 0,
        latestCheckpointStep: typeof run.latest_checkpoint_step === 'number' ? run.latest_checkpoint_step : null,
        weightsStep: typeof run.weights_step === 'number' ? run.weights_step : null,
        modelSpec: makeModelSpec(dims),
        updatedLabel: relativeTime(run.updated_epoch),
      };
      normalized.healthScore = Number.isFinite(Number(normalized.coverageSummary.core_pct))
        ? Number(normalized.coverageSummary.core_pct)
        : 0;
      normalized.healthReason = describeCoverageSummary(normalized.coverageSummary, normalized.artifactSections, true);
      normalized.advancedReason = describeCoverageSummary(normalized.coverageSummary, normalized.artifactSections, false);
      normalized.searchBlob = makeSearchBlob(normalized);
      return normalized;
    }) : [];

    function loadPreference(key, fallback) {
      try {
        const value = window.localStorage.getItem(key);
        return value || fallback;
      } catch (error) {
        return fallback;
      }
    }

    function savePreference(key, value) {
      try {
        window.localStorage.setItem(key, value);
      } catch (error) {
        // ignore storage failures
      }
    }

    const state = {
      search: '',
      kind: 'all',
      parity: 'all',
      report: 'all',
      svg: 'all',
      sort: 'updated_desc',
      view: loadPreference('ck_v7_run_hub_view', 'cards'),
      density: loadPreference('ck_v7_run_hub_density', 'comfortable'),
      selected: new Set(),
    };

    function uniqueNonEmpty(values) {
      return [...new Set((Array.isArray(values) ? values : []).filter((value) => value !== null && value !== undefined && String(value) !== ''))];
    }

    function isSelected(run) {
      return Boolean(run && state.selected.has(run.rel_path));
    }

    function selectedRuns() {
      return runs.filter((run) => state.selected.has(run.rel_path));
    }

    function toggleSelection(relPath) {
      if (!relPath) return;
      if (state.selected.has(relPath)) {
        state.selected.delete(relPath);
      } else {
        state.selected.add(relPath);
      }
    }

    function clearSelection() {
      state.selected.clear();
    }

    function selectionButton(run, idleLabel = 'Select', activeLabel = 'Selected') {
      const active = isSelected(run);
      return `<button class="btn ${active ? 'compare-on' : ''}" data-toggle-select="${encodeURIComponent(run.rel_path)}">${active ? escapeHtml(activeLabel) : escapeHtml(idleLabel)}</button>`;
    }

    function metricLabel(key) {
      return String(key || '')
        .replace(/_/g, ' ')
        .replace(/\b[a-z]/g, (match) => match.toUpperCase());
    }

    function fmtCompareValue(value, key = '') {
      if (value === null || value === undefined || value === '') return 'n/a';
      if (typeof value === 'boolean') return value ? 'yes' : 'no';
      if (typeof value === 'number' && Number.isFinite(value)) {
        const lowerKey = String(key || '').toLowerCase();
        if (
          lowerKey.includes('rate')
          || lowerKey.includes('pct')
          || lowerKey.includes('adherence')
          || lowerKey.includes('robustness')
          || lowerKey.includes('match')
          || lowerKey.includes('coverage')
          || lowerKey.includes('integrity')
        ) {
          return fmtPct(value);
        }
        if (lowerKey.includes('loss')) return fmtLoss(value);
        if (
          lowerKey.includes('count')
          || lowerKey.includes('step')
          || lowerKey.includes('lines')
          || lowerKey.includes('bytes')
          || lowerKey.includes('tokens')
        ) {
          return fmtInt(value);
        }
        return Math.abs(value) >= 10 ? value.toFixed(2) : value.toFixed(4);
      }
      return String(value);
    }

    function collectRowsFromDefs(selected, defs) {
      return defs
        .map((def) => {
          const values = selected.map((run) => def.get(run));
          const hasSignal = values.some((value) => value !== null && value !== undefined && value !== '');
          return hasSignal ? { label: def.label, values } : null;
        })
        .filter(Boolean);
    }

    function orderedMetricKeys(metricObjects) {
      const preferred = [
        'valid_svg_rate',
        'closure_success_rate',
        'prefix_integrity',
        'ood_robustness',
        'adherence',
        'tag_adherence',
        'repetition_loop_score',
        'exact_rate',
        'renderable_rate',
        'holdout_exact_rate',
        'coverage_rate',
        'exact_match_rate',
        'byte_match_rate',
        'line_match_rate',
        'token_count',
        'input_lines',
        'input_bytes',
      ];
      const seen = new Set();
      const discovered = [];
      (metricObjects || []).forEach((obj) => {
        if (!obj || typeof obj !== 'object') return;
        Object.keys(obj).forEach((key) => {
          if (!seen.has(key)) {
            seen.add(key);
            discovered.push(key);
          }
        });
      });
      const front = preferred.filter((key) => seen.has(key));
      const rest = discovered.filter((key) => !front.includes(key)).sort();
      return [...front, ...rest];
    }

    function buildCompareSections(selected) {
      const sections = [];
      const identity = collectRowsFromDefs(selected, [
        { label: 'Run Kind', get: (run) => run.kind },
        { label: 'Compare Family', get: (run) => run.compareFamily || null },
        { label: 'Dataset Type', get: (run) => run.datasetType || null },
        { label: 'Model Spec', get: (run) => run.modelSpec || null },
        { label: 'Shape Signature', get: (run) => run.shape_signature || null },
      ]);
      if (identity.length) sections.push({ title: 'Identity', rows: identity });

      const tokenizerRows = collectRowsFromDefs(selected, [
        { label: 'Tokenizer Mode', get: (run) => run.tokenizerSummary.tokenizer_mode || null },
        { label: 'Tokenizer Status', get: (run) => run.tokenizerSummary.status || null },
        { label: 'Roundtrip Exact', get: (run) => run.tokenizerSummary.exact_match },
      ]);
      const tokenizerMetricKeys = orderedMetricKeys(selected.map((run) => run.tokenizerSummary || {}).map((obj) => {
        const copy = { ...obj };
        delete copy.path;
        delete copy.tokenizer_mode;
        delete copy.status;
        delete copy.exact_match;
        return copy;
      }));
      tokenizerMetricKeys.forEach((key) => {
        tokenizerRows.push({
          label: metricLabel(key),
          values: selected.map((run) => run.tokenizerSummary ? run.tokenizerSummary[key] : null),
        });
      });
      if (tokenizerRows.some((row) => row.values.some((value) => value !== null && value !== undefined && value !== ''))) {
        sections.push({ title: 'Tokenizer', rows: tokenizerRows });
      }

      const trainRows = collectRowsFromDefs(selected, [
        { label: 'Final Loss', get: (run) => run.final_loss },
        { label: 'Start Loss', get: (run) => run.loss_curve_summary && run.loss_curve_summary.available ? run.loss_curve_summary.start_loss : null },
        { label: 'Reduction', get: (run) => run.loss_curve_summary && run.loss_curve_summary.available ? run.loss_curve_summary.reduction + '\u00d7' : null },
        { label: 'Total Steps', get: (run) => run.loss_curve_summary && run.loss_curve_summary.available ? run.loss_curve_summary.total_steps : null },
        { label: 'Convergence Rate', get: (run) => run.loss_curve_summary && run.loss_curve_summary.available ? run.loss_curve_summary.convergence_rate : null },
        { label: 'Final LR', get: (run) => run.loss_curve_summary && run.loss_curve_summary.available ? run.loss_curve_summary.final_lr : null },
        { label: 'Valid SVG Rate', get: (run) => run.validSvgRate },
        { label: 'Parity', get: (run) => run.parityStatus },
        { label: 'Weights Step', get: (run) => run.weightsStep },
        { label: 'Checkpoint Count', get: (run) => run.checkpointCount },
        { label: 'Updated', get: (run) => run.updated_iso || run.updatedLabel },
      ]);
      if (trainRows.length) sections.push({ title: 'Training Surface', rows: trainRows });

      const latestEvalKeys = orderedMetricKeys(selected.map((run) => (run.evalSummary && run.evalSummary.latest_metrics) || {}));
      const latestEvalRows = collectRowsFromDefs(selected, [
        { label: 'Latest Eval Phase', get: (run) => run.evalSummary.latest_phase || null },
      ]);
      latestEvalKeys.forEach((key) => {
        latestEvalRows.push({
          label: metricLabel(key),
          values: selected.map((run) => run.evalSummary && run.evalSummary.latest_metrics ? run.evalSummary.latest_metrics[key] : null),
        });
      });
      if (latestEvalRows.some((row) => row.values.some((value) => value !== null && value !== undefined && value !== ''))) {
        sections.push({ title: 'Eval Latest', rows: latestEvalRows });
      }

      const bestEvalKeys = orderedMetricKeys(selected.map((run) => (run.evalSummary && run.evalSummary.best_metrics) || {}));
      const bestEvalRows = collectRowsFromDefs(selected, [
        { label: 'Best Eval Phase', get: (run) => run.evalSummary.best_phase || null },
      ]);
      bestEvalKeys.forEach((key) => {
        bestEvalRows.push({
          label: metricLabel(key),
          values: selected.map((run) => run.evalSummary && run.evalSummary.best_metrics ? run.evalSummary.best_metrics[key] : null),
        });
      });
      if (bestEvalRows.some((row) => row.values.some((value) => value !== null && value !== undefined && value !== ''))) {
        sections.push({ title: 'Eval Best', rows: bestEvalRows });
      }

      const probeKeys = orderedMetricKeys(selected.map((run) => (run.probeSummary && run.probeSummary.metrics) || {}));
      const probeRows = collectRowsFromDefs(selected, [
        { label: 'Probe Summary Kind', get: (run) => run.probeSummary.kind || null },
        { label: 'Probe Count', get: (run) => run.probeSummary.probe_count || null },
        { label: 'Holdout Count', get: (run) => run.probeSummary.holdout_count || null },
      ]);
      probeKeys.forEach((key) => {
        probeRows.push({
          label: metricLabel(key),
          values: selected.map((run) => run.probeSummary && run.probeSummary.metrics ? run.probeSummary.metrics[key] : null),
        });
      });
      if (probeRows.some((row) => row.values.some((value) => value !== null && value !== undefined && value !== ''))) {
        sections.push({ title: 'Probe Summary', rows: probeRows });
      }

      return sections;
    }

    function buildCompatibility(selected) {
      const families = uniqueNonEmpty(selected.map((run) => run.compareFamily));
      const kinds = uniqueNonEmpty(selected.map((run) => run.kind));
      const datasetTypes = uniqueNonEmpty(selected.map((run) => run.datasetType));
      const shapes = uniqueNonEmpty(selected.map((run) => run.shape_signature));
      const tokenModes = uniqueNonEmpty(selected.map((run) => run.tokenizerSummary && run.tokenizerSummary.tokenizer_mode));
      const signals = [
        { label: families.length === 1 && families[0] ? `family ${families[0]}` : 'mixed family', good: families.length === 1 && families[0] },
        { label: kinds.length === 1 ? `kind ${kinds[0]}` : 'mixed kind', good: kinds.length === 1 },
        { label: datasetTypes.length === 1 && datasetTypes[0] ? `dataset ${datasetTypes[0]}` : 'mixed dataset', good: datasetTypes.length === 1 && datasetTypes[0] },
        { label: shapes.length === 1 && shapes[0] ? 'same shape signature' : 'shape varies', good: shapes.length === 1 && shapes[0] },
        { label: tokenModes.length === 1 && tokenModes[0] ? `tokenizer ${tokenModes[0]}` : 'tokenizer varies', good: tokenModes.length === 1 && tokenModes[0] },
      ];
      const score = signals.reduce((sum, signal) => sum + (signal.good ? 1 : 0), 0);
      const tone = score >= 4 ? 'pass' : (score >= 2 ? 'skip' : 'missing');
      const note = score >= 4
        ? 'Strong compare set: same family/model shape or tokenizer contract.'
        : (score >= 2 ? 'Usable compare set: some shared structure, but not perfectly aligned.' : 'Loose compare set: table is still useful, but treat differences as cross-family.');
      return { score, tone, note, signals };
    }

    function autoSelectSimilar(seedRelPath, limit = 4) {
      const seed = runs.find((run) => run.rel_path === seedRelPath) || runs[0];
      if (!seed) return;
      const peers = runs
        .filter((run) => run.rel_path !== seed.rel_path)
        .map((run) => {
          let score = 0;
          if (run.kind === seed.kind) score += 2;
          if (run.compareFamily && run.compareFamily === seed.compareFamily) score += 3;
          if (run.datasetType && run.datasetType === seed.datasetType) score += 2;
          if (run.shape_signature && run.shape_signature === seed.shape_signature) score += 4;
          if ((run.tokenizerSummary && run.tokenizerSummary.tokenizer_mode) && run.tokenizerSummary.tokenizer_mode === (seed.tokenizerSummary && seed.tokenizerSummary.tokenizer_mode)) score += 1;
          return { run, score };
        })
        .filter((item) => item.score > 0)
        .sort((a, b) => (b.score - a.score) || ((b.run.updated_epoch || 0) - (a.run.updated_epoch || 0)))
        .slice(0, Math.max(1, limit - 1));
      state.selected = new Set([seed.rel_path, ...peers.map((item) => item.run.rel_path)]);
    }

    function renderStandaloneCompareHtml(selected) {
      const compatibility = buildCompatibility(selected);
      const sections = buildCompareSections(selected);
      const families = uniqueNonEmpty(selected.map((run) => run.compareFamily || run.kind));
      const tokenModes = uniqueNonEmpty(selected.map((run) => run.tokenizerSummary && run.tokenizerSummary.tokenizer_mode));
      const headerCells = selected.map((run) => `<th>${escapeHtml(run.name)}<div class="compare-mini">${escapeHtml(run.rel_path)}</div></th>`).join('');
      const bodyRows = sections.map((section) => {
        const rows = section.rows.map((row) => `
          <tr>
            <th>${escapeHtml(row.label)}</th>
            ${row.values.map((value, idx) => `<td>${escapeHtml(fmtCompareValue(value, row.label || String(idx)))}</td>`).join('')}
          </tr>
        `).join('');
        return `<tr class="compare-group-row"><td colspan="${selected.length + 1}">${escapeHtml(section.title)}</td></tr>${rows}`;
      }).join('');
      const signalBadges = compatibility.signals.map((signal) => `<span class="badge ${signal.good ? 'pass' : compatibility.tone}">${escapeHtml(signal.label)}</span>`).join('');
      const summaryCards = [
        { label: 'Selected Runs', value: String(selected.length), note: 'Runs in this compare set.' },
        { label: 'Families', value: families.length === 1 ? families[0] : `${families.length} mixed`, note: families.length === 1 ? 'Single inferred family.' : 'Multiple families are present.' },
        { label: 'Tokenizer', value: tokenModes.length === 1 ? tokenModes[0] : (tokenModes.length ? 'mixed' : 'n/a'), note: tokenModes.length === 1 ? 'Shared tokenizer mode.' : 'Tokenizer contract differs.' },
        { label: 'Compatibility', value: `${compatibility.score}/5`, note: compatibility.note },
      ].map((card) => `
        <article class="summary-card">
          <div class="k">${escapeHtml(card.label)}</div>
          <div class="v">${escapeHtml(card.value)}</div>
          <div class="n">${escapeHtml(card.note)}</div>
        </article>
      `).join('');
      const runCards = selected.map((run) => {
        const compareLabel = run.compareFamily || run.kind || 'run';
        const tokenizerLabel = run.tokenizerSummary && run.tokenizerSummary.tokenizer_mode ? run.tokenizerSummary.tokenizer_mode : 'n/a';
        const evalLabel = (run.evalSummary && (run.evalSummary.best_phase || run.evalSummary.latest_phase)) || (run.probeSummary && run.probeSummary.kind) || 'n/a';
        return `
          <article class="run-card">
            <div class="eyebrow">${escapeHtml(compareLabel)}</div>
            <h2>${escapeHtml(run.name)}</h2>
            <div class="path">${escapeHtml(run.rel_path)}</div>
            <div class="metric-grid">
              <div class="metric">
                <div class="k">Loss</div>
                <div class="v">${escapeHtml(fmtLoss(run.final_loss))}</div>
              </div>
              <div class="metric">
                <div class="k">SVG</div>
                <div class="v">${escapeHtml(fmtPct(run.validSvgRate))}</div>
              </div>
              <div class="metric">
                <div class="k">Tokenizer</div>
                <div class="v small">${escapeHtml(tokenizerLabel)}</div>
              </div>
              <div class="metric">
                <div class="k">Best Signal</div>
                <div class="v small">${escapeHtml(evalLabel)}</div>
              </div>
            </div>
            <div class="badges" style="margin-top:12px;">
              <span class="badge">${escapeHtml(run.kind)}</span>
              ${run.compareFamily ? `<span class="badge">${escapeHtml(run.compareFamily)}</span>` : ''}
              ${run.shape_signature ? `<span class="badge mono">${escapeHtml(run.shape_signature)}</span>` : ''}
            </div>
            <div class="actions" style="margin-top:14px;">
              ${run.report_uri ? `<a class="btn primary" target="_blank" rel="noopener" href="${escapeHtml(run.report_uri)}">Open report</a>` : ''}
              ${run.dataset_viewer_uri ? `<a class="btn dataset" target="_blank" rel="noopener" href="${escapeHtml(run.dataset_viewer_uri)}">Dataset viewer</a>` : ''}
              ${embViewerLink(run, '🧬 Embeddings')}
              ${attnViewerLink(run, '🔭 Attention')}
              <a class="btn" target="_blank" rel="noopener" href="${escapeHtml(run.run_uri)}">Run dir</a>
            </div>
          </article>
        `;
      }).join('');
      return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CK Run Compare</title>
  <style>
    :root {
      --bg: #0b0d12;
      --panel: rgba(255,255,255,0.05);
      --border: rgba(255,255,255,0.12);
      --text: #edf2f7;
      --muted: #98a2b3;
      --good: #39d98a;
      --mid: #ffb020;
      --bad: #ff7b72;
      --accent: #7aa2ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(122,162,255,0.15), transparent 30%),
        radial-gradient(circle at top right, rgba(57,217,138,0.10), transparent 24%),
        linear-gradient(180deg, #12151c 0%, #090b10 100%);
    }
    .page { width: min(1480px, calc(100vw - 32px)); margin: 20px auto 40px; }
    .hero,
    .panel,
    .run-card,
    .summary-card {
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 22px;
      box-shadow: 0 24px 60px rgba(0,0,0,0.28);
      backdrop-filter: blur(10px);
    }
    .hero { padding: 28px 30px; margin-bottom: 20px; }
    .eyebrow {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(122,162,255,0.16);
      color: #bfd1ff;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 12px;
      font-weight: 800;
    }
    h1 { margin: 12px 0 8px; font-size: 38px; line-height: 1.05; }
    h2 { margin: 10px 0 8px; font-size: 22px; }
    h3 { margin: 0 0 8px; font-size: 20px; }
    p { color: var(--muted); line-height: 1.6; }
    .summary-strip,
    .run-grid,
    .metric-grid {
      display: grid;
      gap: 14px;
    }
    .summary-strip { grid-template-columns: repeat(4, minmax(0, 1fr)); margin-top: 18px; }
    .summary-card { padding: 16px; }
    .summary-card .k,
    .metric .k {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 12px;
      font-weight: 700;
    }
    .summary-card .v,
    .metric .v {
      margin-top: 8px;
      font-size: 28px;
      font-weight: 800;
    }
    .summary-card .n { margin-top: 6px; color: var(--muted); font-size: 13px; line-height: 1.5; }
    .panel { padding: 24px 26px; margin-top: 20px; }
    .run-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .run-card { padding: 20px; }
    .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 12px; }
    .metric {
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      border-radius: 16px;
      padding: 14px 16px;
    }
    .metric .v.small { font-size: 16px; line-height: 1.3; }
    .path,
    .compare-mini { color: var(--muted); font-size: 13px; word-break: break-word; margin-top: 4px; }
    .badges,
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.10);
      color: #ddd;
      font-size: 12px;
      background: rgba(255,255,255,0.03);
    }
    .badge.pass { color: var(--good); border-color: rgba(57,217,138,0.3); }
    .badge.skip { color: var(--mid); border-color: rgba(255,176,32,0.3); }
    .badge.missing { color: #aaa; border-color: rgba(170,170,170,0.2); }
    .badge.mono { font-family: "JetBrains Mono", monospace; font-size: 11px; }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 38px;
      padding: 0 13px;
      border-radius: 12px;
      border: 1px solid rgba(255, 180, 0, 0.12);
      background: rgba(255,255,255,0.02);
      color: #e0e0e0;
      font: inherit;
      font-size: 0.82rem;
      font-weight: 600;
      text-decoration: none;
    }
    .btn.primary {
      border-color: rgba(255, 189, 74, 0.18);
      background: linear-gradient(180deg, rgba(255, 189, 74, 0.09), rgba(255, 189, 74, 0.035));
      color: #eadbbb;
    }
    .btn.dataset {
      border-color: rgba(7, 173, 248, 0.36);
      background: linear-gradient(180deg, rgba(7, 173, 248, 0.11), rgba(7, 173, 248, 0.04));
      color: #a8dff7;
    }
    .btn.dataset:hover { background: rgba(7, 173, 248, 0.18); border-color: var(--cyan); }
    .btn.emb {
      border-color: rgba(179, 136, 255, 0.36);
      background: linear-gradient(180deg, rgba(179, 136, 255, 0.11), rgba(179, 136, 255, 0.04));
      color: #d4bbff;
    }
    .btn.emb:hover { background: rgba(179, 136, 255, 0.2); border-color: #b388ff; }
    .btn.attn {
      border-color: rgba(77, 208, 225, 0.36);
      background: linear-gradient(180deg, rgba(77, 208, 225, 0.11), rgba(77, 208, 225, 0.04));
      color: #a5eaf2;
    }
    .btn.attn:hover { background: rgba(77, 208, 225, 0.2); border-color: #4dd0e1; }
    .compare-table-wrap {
      overflow-x: auto;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 18px;
      background: rgba(10,10,10,0.38);
    }
    .compare-table {
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }
    .compare-table th,
    .compare-table td {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      text-align: left;
      vertical-align: top;
    }
    .compare-table thead th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: rgba(16,16,16,0.94);
    }
    .compare-table tbody th {
      min-width: 210px;
      color: var(--muted);
      font-weight: 600;
      background: rgba(255,255,255,0.015);
    }
    .compare-group-row td {
      background: rgba(255,180,0,0.08);
      color: #f8e8ba;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    @media (max-width: 1100px) {
      .summary-strip,
      .run-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Dynamic Compare</div>
      <h1>Model Family Comparison</h1>
      <p>This page is generated from the IR hub selection. It compares shared metadata automatically, so the same layout can work for SVG today and SQL or other structured domains later.</p>
      <div class="badges">${signalBadges}</div>
      <div class="summary-strip">${summaryCards}</div>
    </section>

    <section class="panel">
      <h3>Selected Runs</h3>
      <div class="run-grid">
        ${runCards}
      </div>
    </section>

    <section class="panel">
      <h3>Comparison Matrix</h3>
      <p>${escapeHtml(compatibility.note)}</p>
      <div class="compare-table-wrap">
        <table class="compare-table">
          <thead>
            <tr>
              <th>Field</th>
              ${headerCells}
            </tr>
          </thead>
          <tbody>${bodyRows}</tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>`;
    }

    // ── Loss Curve Overlay for Compare Panel ───────────────────────
    const SPARK_COLORS = ['#f87171','#60a5fa','#4ade80','#fbbf24','#a78bfa','#2dd4bf','#fb923c','#e879f9'];

    function buildLossCurveOverlay(selected) {
      const withCurves = selected.filter(function(run) {
        return run.loss_curve_summary && run.loss_curve_summary.available && run.loss_curve_summary.sparkline && run.loss_curve_summary.sparkline.length > 1;
      });
      if (withCurves.length === 0) return '';
      var canvasId = 'hub-loss-overlay-' + Date.now();
      setTimeout(function() { drawHubLossOverlay(canvasId, withCurves); }, 50);
      return '<div class="compare-section"><h3>Loss Curve Overlay</h3>'
        + '<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:6px;overflow:hidden;">'
        + '<canvas id="' + canvasId + '" style="width:100%;height:250px;display:block;"></canvas>'
        + '</div>'
        + '<div class="compare-mini" style="margin-top:0.3rem;">' + withCurves.map(function(run, i) {
            return '<span style="color:' + SPARK_COLORS[i % SPARK_COLORS.length] + ';">\u25cf ' + escapeHtml(run.name) + '</span>';
          }).join(' &nbsp; ') + '</div>'
        + '</div>';
    }

    function drawHubLossOverlay(canvasId, withCurves) {
      var canvas = document.getElementById(canvasId);
      if (!canvas || !canvas.getContext) return;
      var ctx = canvas.getContext('2d');
      var dpr = window.devicePixelRatio || 1;
      var rect = canvas.getBoundingClientRect();
      var W = rect.width || 500;
      var H = rect.height || 250;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.scale(dpr, dpr);
      canvas.style.width = W + 'px';
      canvas.style.height = H + 'px';

      var margin = {top: 24, right: 14, bottom: 32, left: 56};
      var pw = W - margin.left - margin.right;
      var ph = H - margin.top - margin.bottom;
      if (pw <= 0 || ph <= 0) return;

      // Compute global axis range across all runs
      var xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
      withCurves.forEach(function(run) {
        run.loss_curve_summary.sparkline.forEach(function(pt) {
          if (pt.s < xMin) xMin = pt.s;
          if (pt.s > xMax) xMax = pt.s;
          if (pt.l < yMin) yMin = pt.l;
          if (pt.l > yMax) yMax = pt.l;
        });
      });
      if (!Number.isFinite(xMin) || xMin === xMax) return;
      if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }

      function toX(v) { return margin.left + ((v - xMin) / (xMax - xMin)) * pw; }
      function toY(v) { return margin.top + (1 - (v - yMin) / (yMax - yMin)) * ph; }

      // Background
      ctx.fillStyle = '#1a1a2e';
      ctx.fillRect(0, 0, W, H);

      // Grid
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.lineWidth = 0.5;
      for (var i = 0; i <= 5; i++) {
        var y = margin.top + (i / 5) * ph;
        ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + pw, y); ctx.stroke();
        var x = margin.left + (i / 5) * pw;
        ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + ph); ctx.stroke();
      }

      // Axis labels
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      for (var i = 0; i <= 5; i++) {
        var v = yMax - (i / 5) * (yMax - yMin);
        var label = v >= 1000 ? (v / 1000).toFixed(1) + 'K' : v < 0.01 && v !== 0 ? v.toExponential(1) : v.toPrecision(3);
        ctx.fillText(label, margin.left - 4, margin.top + (i / 5) * ph + 3);
      }
      ctx.textAlign = 'center';
      for (var i = 0; i <= 5; i++) {
        var v = xMin + (i / 5) * (xMax - xMin);
        ctx.fillText(Math.round(v).toLocaleString(), margin.left + (i / 5) * pw, H - margin.bottom + 16);
      }

      // Title
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.font = 'bold 11px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText('Loss Curve Overlay \u2014 Cross-Run Comparison', margin.left, margin.top - 8);

      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('step', margin.left + pw / 2, H - 2);

      // Draw each run's sparkline
      withCurves.forEach(function(run, idx) {
        var spark = run.loss_curve_summary.sparkline;
        var color = SPARK_COLORS[idx % SPARK_COLORS.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(toX(spark[0].s), toY(spark[0].l));
        for (var i = 1; i < spark.length; i++) {
          ctx.lineTo(toX(spark[i].s), toY(spark[i].l));
        }
        ctx.stroke();

        // End-point dot with final loss label
        var last = spark[spark.length - 1];
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(toX(last.s), toY(last.l), 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    function renderComparePanel(filteredRuns) {
      const selected = selectedRuns();
      const seed = filteredRuns[0] || runs[0] || null;
      const compatibility = selected.length >= 2 ? buildCompatibility(selected) : null;
      const selectedTags = selected.length
        ? selected.map((run) => `<span class="tag"><strong>${escapeHtml(run.name)}</strong>${run.compareFamily ? ` · ${escapeHtml(run.compareFamily)}` : ''}</span>`).join('')
        : '<span class="compare-mini">No runs selected yet.</span>';

      let tableHtml = '<div class="compare-mini">Select at least two runs. The hub will build a comparison table from shared identity, tokenizer, eval, and probe summaries.</div>';
      if (selected.length >= 2) {
        const sections = buildCompareSections(selected);
        const headerCells = selected.map((run) => `<th>${escapeHtml(run.name)}<div class="compare-mini">${escapeHtml(run.rel_path)}</div></th>`).join('');
        const bodyRows = sections.map((section) => {
          const rows = section.rows.map((row) => `
            <tr>
              <th>${escapeHtml(row.label)}</th>
              ${row.values.map((value, idx) => `<td>${escapeHtml(fmtCompareValue(value, row.label || String(idx)))}</td>`).join('')}
            </tr>
          `).join('');
          return `<tr class="compare-group-row"><td colspan="${selected.length + 1}">${escapeHtml(section.title)}</td></tr>${rows}`;
        }).join('');
        tableHtml = `
          <div class="compare-table-wrap">
            <table class="compare-table">
              <thead>
                <tr>
                  <th>Field</th>
                  ${headerCells}
                </tr>
              </thead>
              <tbody>${bodyRows}</tbody>
            </table>
          </div>
        `;
      }

      els.comparePanel.innerHTML = `
        <div class="panel-head">
          <div>
            <h2>Dynamic Compare</h2>
            <p class="panel-sub">Pick a few trained runs or model families. The hub will compare shared metadata automatically, so the same workflow can work for SVG today and SQL later.</p>
          </div>
          <div class="muted">${selected.length} selected</div>
        </div>
        <div class="compare-layout">
          <div class="compare-toolbar">
            <div class="meta-block">
              <div class="compare-hints">${selectedTags}</div>
              ${compatibility ? `<div class="compare-badges">${compatibility.signals.map((signal) => `<span class="badge ${signal.good ? 'pass' : compatibility.tone}">${escapeHtml(signal.label)}</span>`).join('')}</div><div class="compare-mini">${escapeHtml(compatibility.note)}</div>` : '<div class="compare-mini">Best signal comes from runs that share kind, family, tokenizer mode, or shape signature.</div>'}
            </div>
            <div class="compare-actions">
              ${seed ? `<button class="btn" data-compare-auto="${encodeURIComponent(seed.rel_path)}">Auto similar to spotlight</button>` : ''}
              <button class="btn" data-clear-compare="1">Clear</button>
              ${selected.length >= 2 ? '<button class="btn primary" data-open-compare="1">Open compare page</button>' : ''}
            </div>
          </div>
          <div class="compare-grid">
            <div class="compare-section">
              <h3>Comparison Table</h3>
              ${tableHtml}
            </div>
            <div class="compare-section">
              <h3>Quick Links</h3>
              <div class="compare-actions">
                ${selected.map((run) => run.report_uri ? `<a class="btn" target="_blank" rel="noopener" href="${escapeHtml(run.report_uri)}">${escapeHtml(run.name)} report</a>` : `<a class="btn" target="_blank" rel="noopener" href="${escapeHtml(run.run_uri)}">${escapeHtml(run.name)} run</a>`).join('')}
              </div>
            </div>
            ${buildLossCurveOverlay(selected)}
          </div>
        </div>
      `;
    }

    function renderHero() {
      const summary = HUB.summary || {};
      const newest = runs[0];
      const reportCount = summary.runs_with_report || runs.filter((run) => run.reportReady).length;
      const datasetViewerCount = summary.runs_with_dataset_viewer || runs.filter((run) => run.datasetViewerReady).length;
      const embeddingsCount = summary.runs_with_embeddings || runs.filter((run) => run.embeddings_uri).length;
      const attentionCount = summary.runs_with_attention || runs.filter((run) => run.attention_uri).length;
      const parityPass = summary.runs_parity_pass || runs.filter((run) => run.parityStatus === 'PASS' || run.parityStatus === 'PASS_REUSED').length;
      const trainCount = summary.runs_train || runs.filter((run) => run.kind === 'train').length;
      const inferCount = runs.filter((run) => run.kind === 'inference').length;
      const bestLoss = runs
        .map((run) => run.final_loss)
        .filter((value) => typeof value === 'number' && Number.isFinite(value))
        .sort((a, b) => a - b)[0];
      const totalCheckpoints = runs.reduce((sum, run) => sum + (run.checkpointCount || 0), 0);

      els.heroMeta.innerHTML = [
        `<span class="tag"><strong>${escapeHtml(HUB.schema || 'ck.ir.hub.v1')}</strong> schema</span>`,
        `<span class="tag"><strong>${escapeHtml(HUB.generated_at || 'n/a')}</strong> generated</span>`,
        `<span class="tag"><strong>${escapeHtml(HUB.models_root || 'n/a')}</strong> models root</span>`,
      ].join('');

      els.orbitalCoreValue.textContent = fmtInt(summary.runs_total || runs.length);
      els.orbitFreshest.textContent = newest ? newest.updatedLabel : 'n/a';
      els.orbitReports.textContent = `${fmtInt(reportCount)} ready`;
      els.orbitParity.textContent = `${fmtInt(parityPass)} pass`;

      const metrics = [
        { label: 'Indexed Runs', value: fmtInt(summary.runs_total || runs.length), note: 'All discovered run roots.' },
        { label: 'Training Roots', value: fmtInt(trainCount), note: 'Train pipeline directories.' },
        { label: 'Inference Roots', value: fmtInt(inferCount), note: 'Inference and runtime roots.' },
        { label: 'Reports Ready', value: fmtInt(reportCount), note: 'ir_report.html or .ck_build report.' },
        { label: 'Dataset Viewers', value: fmtInt(datasetViewerCount), note: 'run-local dataset_viewer.html snapshots.' },
        { label: '🧬 Embeddings', value: fmtInt(embeddingsCount), note: 'Runs with exported embeddings.json.' },
        { label: '🔭 Attention', value: fmtInt(attentionCount), note: 'Runs with exported attention.json.' },
        { label: 'Parity Pass', value: fmtInt(parityPass), note: 'PASS plus PASS_REUSED regimen states.' },
        { label: 'Best Loss / Checkpoints', value: `${fmtLoss(bestLoss)} / ${fmtInt(totalCheckpoints)}`, note: 'Lowest loss and total checkpoint volume.' },
      ];

      els.metricRibbon.innerHTML = metrics.map((metric) => `
        <article class="metric-card">
          <div class="metric-label">${escapeHtml(metric.label)}</div>
          <div class="metric-value">${escapeHtml(metric.value)}</div>
          <div class="metric-note">${escapeHtml(metric.note)}</div>
        </article>
      `).join('');
    }

    function renderSectionSummary(run, core) {
      const sections = Array.isArray(run.artifactSections)
        ? run.artifactSections.filter((section) => Boolean(section.core) === core)
        : [];
      if (!sections.length) return '';
      return `
        <div class="section-summary">
          ${sections.map((section) => {
            const pct = section.total ? Math.round((section.present / section.total) * 100) : 0;
            const state = section.present === section.total ? 'pass' : (section.present > 0 ? 'skip' : 'missing');
            return `<span class="badge ${state}">${escapeHtml(section.title)} ${escapeHtml(String(section.present))}/${escapeHtml(String(section.total))}</span>`;
          }).join('')}
        </div>
      `;
    }

    function renderNextActions(run) {
      if (!Array.isArray(run.nextActions) || !run.nextActions.length) return '';
      return `
        <div class="action-stack">
          ${run.nextActions.map((action) => `
            <div class="action-row-inline">
              <span class="action-label">${escapeHtml(action.label || 'Action')}</span>
              <button class="btn" data-copy="${encodeURIComponent(action.cmd || '')}">Copy cmd</button>
            </div>
          `).join('')}
        </div>
      `;
    }

    function renderDatasetPrepChecklist(run) {
      const items = Array.isArray(run.datasetPrepChecklist) ? run.datasetPrepChecklist : [];
      if (!items.length) return '';
      const ready = items.filter((item) => item.ready).length;
      const total = items.length;
      const rows = items.map((item) => `
        <tr>
          <td style="padding:8px 10px;color:#e5e7eb;">${escapeHtml(item.label || item.key || 'step')}</td>
          <td style="padding:8px 10px;color:${item.ready ? '#4ade80' : '#f59e0b'};font-weight:700;">${item.ready ? 'READY' : 'PENDING'}</td>
          <td style="padding:8px 10px;color:#9ca3af;">${escapeHtml(item.hint || '')}</td>
        </tr>
      `).join('');
      return `
        <div style="margin-top:14px;border:1px solid rgba(110,231,183,0.18);background:rgba(6,18,15,0.55);border-radius:14px;padding:14px;">
          <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;">
            <div>
              <div style="font-size:0.76rem;color:#6ee7b7;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;">Dataset Prep Checklist</div>
              <div style="font-size:1rem;font-weight:700;color:#f8fafc;margin-top:4px;">${escapeHtml(String(ready))}/${escapeHtml(String(total))} ready before pretraining</div>
              <div style="font-size:0.86rem;color:#94a3b8;margin-top:4px;">Workspace: <span class="mono">${escapeHtml(run.datasetWorkspace || 'n/a')}</span>${run.datasetType ? ` · type=${escapeHtml(run.datasetType)}` : ''}${run.datasetStageMode ? ` · snapshot=${escapeHtml(run.datasetStageMode)}` : ''}</div>
            </div>
            <div class="action-row" style="margin:0;">
              ${run.datasetRefreshCmd ? `<button class="btn" data-copy="${encodeURIComponent(run.datasetRefreshCmd)}">Copy materialize dataset cmd</button>` : ''}
              ${run.datasetRebuildViewerCmd ? `<button class="btn" data-copy="${encodeURIComponent(run.datasetRebuildViewerCmd)}">Copy rebuild viewer cmd</button>` : ''}
            </div>
          </div>
          <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;">
            ${items.map((item) => `<span class="badge ${item.ready ? 'pass' : 'skip'}">${escapeHtml(item.label || item.key || 'step')} ${item.ready ? 'ready' : 'pending'}</span>`).join('')}
          </div>
          <div style="margin-top:12px;overflow:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:0.86rem;">
              <thead>
                <tr style="text-align:left;border-bottom:1px solid rgba(148,163,184,0.18);">
                  <th style="padding:8px 10px;color:#fbbf24;">Step</th>
                  <th style="padding:8px 10px;color:#fbbf24;">Status</th>
                  <th style="padding:8px 10px;color:#fbbf24;">What remains</th>
                </tr>
              </thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </div>
      `;
    }

    function renderSpotlight(filteredRuns) {
      const run = filteredRuns[0] || runs[0];
      if (!run) {
        els.spotlightPanel.innerHTML = `
          <div class="panel-head"><h2>Spotlight</h2></div>
          <div class="spotlight-body"><p class="muted">No discovered runs under ${escapeHtml(HUB.models_root || '')}.</p></div>
        `;
        return;
      }

      const tone = parityTone(run.parityStatus);
      const ring = Math.max(0, Math.min(360, Math.round((run.healthScore / 100) * 360)));
      els.spotlightPanel.innerHTML = `
        <div class="panel-head">
          <div>
            <h2>Spotlight Run</h2>
            <p class="panel-sub">Top match after the current filters and sort order.</p>
          </div>
          <div class="spotlight-tags">
            <span class="badge ${run.kind}">${escapeHtml(run.kind)}</span>
            ${showParityBadge(run) ? `<span class="badge ${tone}">${escapeHtml(run.parityStatus)}</span>` : ''}
            ${run.reportReady ? '<span class="badge report">report ready</span>' : ''}
            ${run.datasetViewerReady ? '<span class="badge report">dataset viewer</span>' : ''}
          </div>
        </div>
        <div class="spotlight-body">
          <div class="spotlight-shell">
            <div class="spotlight-main">
              <div class="spotlight-title-row">
                <div>
                  <h3 class="spotlight-title">${escapeHtml(run.name)}</h3>
                  <div class="spotlight-path">${escapeHtml(run.rel_path)}</div>
                </div>
                <div class="tag"><strong>${escapeHtml(run.updatedLabel)}</strong> latest signal</div>
              </div>
              <div class="spotlight-summary">
                <div class="summary-tile"><div class="k">Model Spec</div><div class="v">${escapeHtml(run.modelSpec)}</div></div>
                <div class="summary-tile"><div class="k">Final Loss</div><div class="v">${escapeHtml(fmtLoss(run.final_loss))}</div></div>
                <div class="summary-tile"><div class="k">Valid SVG Rate</div><div class="v">${escapeHtml(fmtPct(run.validSvgRate))}</div></div>
                <div class="summary-tile"><div class="k">Checkpoints</div><div class="v">${escapeHtml(fmtInt(run.checkpointCount))}</div></div>
                <div class="summary-tile"><div class="k">Weights Step</div><div class="v">${escapeHtml(fmtInt(run.weightsStep))}</div></div>
              </div>
              <div class="action-row">
                ${run.report_uri ? `<a class="btn primary" target="_blank" rel="noopener" href="${escapeHtml(run.report_uri)}">Open report</a>` : ''}
                ${run.dataset_viewer_uri ? `<a class="btn dataset" target="_blank" rel="noopener" href="${escapeHtml(run.dataset_viewer_uri)}">Dataset viewer</a>` : ''}
                ${embViewerLink(run, '🧬 Embeddings')}
                ${attnViewerLink(run, '🔭 Attention')}
                ${run.gallery_uri ? `<a class="btn" target="_blank" rel="noopener" href="${escapeHtml(run.gallery_uri)}">SVG Gallery</a>` : ''}
                <a class="btn" target="_blank" rel="noopener" href="${escapeHtml(run.run_uri)}">Open run dir</a>
                ${selectionButton(run, 'Select for compare', 'Selected for compare')}
                <button class="btn" data-compare-auto="${encodeURIComponent(run.rel_path)}">Auto similar</button>
                <button class="btn" data-copy="${encodeURIComponent(run.generate_report_cmd || '')}">Copy cmd</button>
              </div>
              ${renderSectionSummary(run, true)}
              <div class="health-note">Core dashboard coverage. Advanced correctness and deep profiling are tracked separately below.</div>
              ${renderDatasetPrepChecklist(run)}
              ${renderCommandsPanel(run)}
              <details class="detail-toggle">
                <summary>Advanced details</summary>
                <div class="spotlight-summary" style="margin-top:12px;">
                  <div class="summary-tile"><div class="k">Shape Signature</div><div class="v">${escapeHtml(run.shape_signature || 'n/a')}</div></div>
                  <div class="summary-tile"><div class="k">Weights Reason</div><div class="v">${escapeHtml(run.weights_reason || 'n/a')}</div></div>
                  <div class="summary-tile"><div class="k">Updated ISO</div><div class="v mono">${escapeHtml(run.updated_iso || 'n/a')}</div></div>
                  <div class="summary-tile"><div class="k">Checkpoint Step</div><div class="v">${escapeHtml(fmtInt(run.latestCheckpointStep))}</div></div>
                  <div class="summary-tile"><div class="k">Dataset Snapshot</div><div class="v">${run.datasetSnapshotReady ? 'ready' : 'missing'}</div></div>
                  <div class="summary-tile"><div class="k">Dataset Entries</div><div class="v">${escapeHtml(fmtInt(run.datasetStagedEntries.length))}</div></div>
                </div>
                ${renderSectionSummary(run, false)}
                <div class="health-note" style="margin-top:10px;">Advanced checks: ${escapeHtml(run.advancedReason || 'none tracked')}</div>
                ${renderNextActions(run)}
              </details>
            </div>
            <aside class="spotlight-side">
              <div class="health-ring" style="--ring:${ring}deg;">
                <div class="health-core">
                  <div>
                    <strong>${escapeHtml(String(run.healthScore))}</strong>
                    <span>Coverage</span>
                  </div>
                </div>
              </div>
              <div class="mini-grid">
                <div class="mini-tile"><div class="k">Updated</div><div class="v">${escapeHtml(run.updatedLabel)}</div></div>
                <div class="mini-tile"><div class="k">Core Coverage</div><div class="v">${escapeHtml(fmtInt(run.coverageSummary.core_present))}/${escapeHtml(fmtInt(run.coverageSummary.core_total))}</div></div>
                <div class="mini-tile"><div class="k">Advanced Checks</div><div class="v">${escapeHtml(fmtInt(run.coverageSummary.advanced_present))}/${escapeHtml(fmtInt(run.coverageSummary.advanced_total))}</div></div>
                <div class="mini-tile"><div class="k">Report Surface</div><div class="v">${run.reportReady ? 'Ready for direct entry' : 'Generate report first'}</div></div>
                <div class="mini-tile"><div class="k">Dataset</div><div class="v">${run.datasetViewerReady ? 'Viewer ready' : 'No dataset snapshot'}</div></div>
              </div>
            </aside>
          </div>
        </div>
      `;
    }

    function makeCoveragePct(list, predicate) {
      if (!list.length) return 0;
      return Math.round((list.filter(predicate).length / list.length) * 100);
    }

    function sectionCompletePct(list, key) {
      const relevant = list.filter((run) => Array.isArray(run.artifactSections) && run.artifactSections.some((section) => section.key === key));
      if (!relevant.length) return null;
      return Math.round((relevant.filter((run) => {
        const section = run.artifactSections.find((item) => item.key === key);
        return section && section.total > 0 && section.present === section.total;
      }).length / relevant.length) * 100);
    }

    function renderCoverage(filteredRuns) {
      const target = filteredRuns.length ? filteredRuns : runs;
      const reportPct = makeCoveragePct(target, (run) => run.reportReady);
      const avgCorePct = target.length
        ? Math.round(target.reduce((sum, run) => sum + (run.coverageSummary.core_pct || 0), 0) / target.length)
        : 0;
      const avgAdvancedPct = target.length
        ? Math.round(target.reduce((sum, run) => {
          const pct = typeof run.coverageSummary.advanced_pct === 'number' ? run.coverageSummary.advanced_pct : 0;
          return sum + pct;
        }, 0) / target.length)
        : 0;
      const coreCompletePct = makeCoveragePct(target, (run) => (run.coverageSummary.core_pct || 0) === 100);
      const correctnessPct = sectionCompletePct(target, 'correctness');
      const profilingPct = sectionCompletePct(target, 'basic_profiling');
      const datasetPct = makeCoveragePct(target, (run) => run.datasetViewerReady);
      const trainParityPct = makeCoveragePct(
        target.filter((run) => run.kind === 'train'),
        (run) => run.parityStatus === 'PASS' || run.parityStatus === 'PASS_REUSED',
      );

      const items = [
        ['Report surface', reportPct],
        ['Avg core coverage', avgCorePct],
        ['Core complete', coreCompletePct],
        ['Avg advanced checks', avgAdvancedPct],
        ['Basic profiling complete', profilingPct ?? 0],
        ['Correctness complete', correctnessPct ?? 0],
        ['Dataset viewer', datasetPct],
      ];
      if (target.some((run) => run.kind === 'train')) {
        items.push(['Training parity pass', trainParityPct]);
      }

      els.coveragePanel.innerHTML = `
        <div class="panel-head"><h3>Coverage Matrix</h3></div>
        <div class="panel-sub">Coverage is scoped to the visible result set. Core coverage tracks dashboard-ready artifacts; advanced checks are shown separately.</div>
        <div class="rail-body">
          <div class="coverage-stack">
            ${items.map(([label, pct]) => `
              <div class="coverage-item">
                <div class="coverage-label"><span>${escapeHtml(label)}</span><strong>${escapeHtml(fmtPercentWhole(pct))}</strong></div>
                <div class="coverage-bar"><span style="width:${Math.max(0, Math.min(100, pct))}%"></span></div>
              </div>
            `).join('')}
          </div>
          <div class="codebox" style="margin-top:14px;">scope=${escapeHtml(filteredRuns.length ? 'filtered' : 'all-runs')} total=${escapeHtml(String(target.length))}</div>
        </div>
      `;
    }

    function renderLegend() {
      els.legendPanel.innerHTML = `
        <div class="panel-head"><h3>Status Legend</h3></div>
        <div class="rail-body">
          <div class="legend-list">
            <div class="legend-item"><span class="legend-dot" style="background:#57d89c"></span><div><strong>PASS</strong><br><span>Fresh parity evidence exists and passed.</span></div></div>
            <div class="legend-item"><span class="legend-dot" style="background:#ffd071"></span><div><strong>PASS_REUSED / SKIP</strong><br><span>Usable parity signal exists, but it was skipped or reused.</span></div></div>
            <div class="legend-item"><span class="legend-dot" style="background:#ff6f7f"></span><div><strong>FAIL</strong><br><span>Parity regimen recorded a failing state.</span></div></div>
            <div class="legend-item"><span class="legend-dot" style="background:#9a7dff"></span><div><strong>MISSING</strong><br><span>No parity JSON or readable summary was found.</span></div></div>
            <div class="legend-item"><span class="legend-dot" style="background:#46dbff"></span><div><strong>Report Ready</strong><br><span>Run root already has a generated HTML report to open.</span></div></div>
          </div>
        </div>
      `;
    }

    function renderActions() {
      els.actionPanel.innerHTML = `
        <div class="panel-head"><h3>Operator Actions</h3></div>
        <div class="rail-body">
          <div class="kv-stack">
            <div class="kv-row"><span class="k">Hub</span><span class="v mono">python3 version/v7/tools/open_ir_hub.py --open</span></div>
            <div class="kv-row"><span class="k">Refresh Report</span><span class="v mono">make v7-capture-artifacts-run RUN=/path/to/run</span></div>
            <div class="kv-row"><span class="k">Profile Dashboard</span><span class="v mono">make v7-profile-dashboard-run RUN=/path/to/run</span></div>
            <div class="kv-row"><span class="k">Memory Signoff</span><span class="v mono">make v7-memory-signoff V7_MODEL=/path/to/run</span></div>
            <div class="kv-row"><span class="k">Perf Gate</span><span class="v mono">make v7-perf-gate-evaluate V7_MODEL=/path/to/run</span></div>
            <div class="kv-row"><span class="k">Checkpoint History</span><span class="v mono">python3 version/v7/scripts/promote_latest_checkpoint_v7.py --run /path/to/run --list-runs</span></div>
          </div>
          <div class="codebox" style="margin-top:14px;">Use the hub to choose a run. Use the run-dir wrappers to fill missing core artifacts, profiling, and correctness checks without reconstructing the command chain by hand.</div>
        </div>
      `;
    }

    function sortRuns(list) {
      const copy = [...list];
      copy.sort((a, b) => {
        switch (state.sort) {
          case 'updated_asc':
            return (a.updated_epoch || 0) - (b.updated_epoch || 0);
          case 'loss_asc':
            return (a.final_loss ?? Number.POSITIVE_INFINITY) - (b.final_loss ?? Number.POSITIVE_INFINITY);
          case 'health_desc':
            return (b.healthScore || 0) - (a.healthScore || 0);
          case 'checkpoints_desc':
            return (b.checkpointCount || 0) - (a.checkpointCount || 0);
          case 'name_asc':
            return String(a.name || '').localeCompare(String(b.name || ''));
          case 'updated_desc':
          default:
            return (b.updated_epoch || 0) - (a.updated_epoch || 0);
        }
      });
      return copy;
    }

    function applyFilters() {
      const needle = state.search.trim().toLowerCase();
      return sortRuns(runs.filter((run) => {
        if (state.kind !== 'all' && run.kind !== state.kind) return false;
        if (state.report === 'present' && !run.reportReady) return false;
        if (state.report === 'missing' && run.reportReady) return false;
        if (state.parity === 'pass' && !(run.parityStatus === 'PASS' || run.parityStatus === 'PASS_REUSED')) return false;
        if (state.parity === 'fail' && run.parityStatus !== 'FAIL') return false;
        if (state.parity === 'missing' && !['MISSING', 'SKIP'].includes(run.parityStatus)) return false;
        if (state.svg === 'present' && run.validSvgRate === null) return false;
        if (state.svg === 'strong' && !(run.validSvgRate !== null && run.validSvgRate >= 0.8)) return false;
        if (state.svg === 'missing' && run.validSvgRate !== null) return false;
        if (needle && !run.searchBlob.includes(needle)) return false;
        return true;
      }));
    }

    function renderResultsSummary(filteredRuns) {
      els.resultSummary.textContent = `${filteredRuns.length} shown / ${runs.length} total`;
      const avgHealth = filteredRuns.length
        ? Math.round(filteredRuns.reduce((sum, run) => sum + run.healthScore, 0) / filteredRuns.length)
        : 0;
      els.resultPills.innerHTML = [
        `<span class="result-pill"><strong>${filteredRuns.length}</strong> visible</span>`,
        `<span class="result-pill"><strong>${filteredRuns.filter((run) => run.kind === 'train').length}</strong> train</span>`,
        `<span class="result-pill"><strong>${filteredRuns.filter((run) => run.kind === 'inference').length}</strong> inference</span>`,
        `<span class="result-pill"><strong>${filteredRuns.filter((run) => run.reportReady).length}</strong> reports ready</span>`,
        `<span class="result-pill"><strong>${filteredRuns.filter((run) => run.datasetViewerReady).length}</strong> dataset viewers</span>`,
        `<span class="result-pill"><strong>${selectedRuns().length}</strong> compare selected</span>`,
        `<span class="result-pill"><strong>${avgHealth}</strong> avg coverage</span>`,
        `<span class="result-pill"><strong>${escapeHtml(state.view)}</strong> view</span>`,
        `<span class="result-pill"><strong>${escapeHtml(state.density)}</strong> density</span>`,
      ].join('');
    }

    function syncPresentationMode() {
      document.body.classList.toggle('table-mode', state.view === 'table');
      document.body.classList.toggle('compact-mode', state.density === 'compact');
      els.runGrid.className = state.view === 'table' ? 'table-host' : 'run-grid';
      if (els.viewSelect.value !== state.view) {
        els.viewSelect.value = state.view;
      }
      if (els.densitySelect.value !== state.density) {
        els.densitySelect.value = state.density;
      }
    }

    function renderRunCards(filteredRuns) {
      if (!filteredRuns.length) {
        els.runGrid.innerHTML = '';
        els.emptyState.hidden = false;
        return;
      }

      els.emptyState.hidden = true;
      els.runGrid.innerHTML = filteredRuns.map((run) => {
        const tone = parityTone(run.parityStatus);
        return `
          <article class="run-card">
            <div class="run-card-body">
              <div class="run-header">
                <div>
                  <h3 class="run-name">${escapeHtml(run.name)}</h3>
                  <div class="run-path">${escapeHtml(run.rel_path)}</div>
                </div>
                <div class="run-badges">
                  <span class="badge ${run.kind}">${escapeHtml(run.kind)}</span>
                  ${showParityBadge(run) ? `<span class="badge ${tone}">${escapeHtml(run.parityStatus)}</span>` : ''}
                  ${run.reportReady ? '<span class="badge report">report</span>' : ''}
                  ${run.datasetViewerReady ? '<span class="badge report">dataset</span>' : ''}
                </div>
              </div>

              <div class="run-health">
                <div class="coverage-label"><span title="Core dashboard coverage for this run scope. Training parity/checkpoint metadata no longer penalize inference runs; advanced correctness checks are tracked separately.">Inspection coverage</span><strong>${escapeHtml(String(run.healthScore))}/100</strong></div>
                <div class="health-bar"><span style="width:${Math.max(0, Math.min(100, run.healthScore))}%"></span></div>
                ${renderSectionSummary(run, true)}
                <div class="health-note">Measures core dashboard completeness, not model quality.</div>
              </div>

              <div class="run-stats">
                <div class="run-stat"><div class="k">Updated</div><div class="v">${escapeHtml(run.updatedLabel)}</div></div>
                <div class="run-stat"><div class="k">Final Loss</div><div class="v">${escapeHtml(fmtLoss(run.final_loss))}</div></div>
              </div>

              <div class="run-actions">
                ${run.report_uri ? `<a class="btn primary" target="_blank" rel="noopener" href="${escapeHtml(run.report_uri)}">Report</a>` : ''}
                ${run.dataset_viewer_uri ? `<a class="btn dataset" target="_blank" rel="noopener" href="${escapeHtml(run.dataset_viewer_uri)}">Dataset</a>` : ''}
                ${embViewerLink(run, '🧬 Emb')}
                ${attnViewerLink(run, '🔭 Attn')}
                ${run.gallery_uri ? `<a class="btn" target="_blank" rel="noopener" href="${escapeHtml(run.gallery_uri)}">Gallery</a>` : ''}
                <a class="btn" target="_blank" rel="noopener" href="${escapeHtml(run.run_uri)}">Run dir</a>
                ${selectionButton(run)}
                <button class="btn" data-copy="${encodeURIComponent(run.generate_report_cmd || '')}">Copy</button>
              </div>

              <details class="detail-toggle">
                <summary>Run details</summary>
                <div class="run-spec">${escapeHtml(run.modelSpec)}${run.shape_signature ? ` | ${escapeHtml(run.shape_signature)}` : ''}</div>
                <div class="health-note" style="margin-bottom:10px;">Core coverage: ${escapeHtml(run.healthReason || 'n/a')}</div>
                ${renderSectionSummary(run, false)}
                <div class="health-note" style="margin-top:10px;">Advanced checks: ${escapeHtml(run.advancedReason || 'none tracked')}</div>
                <div class="run-stats">
                  <div class="run-stat"><div class="k">SVG Rate</div><div class="v">${escapeHtml(fmtPct(run.validSvgRate))}</div></div>
                  <div class="run-stat"><div class="k">Checkpoints</div><div class="v">${escapeHtml(fmtInt(run.checkpointCount))}</div></div>
                  <div class="run-stat"><div class="k">Weights Step</div><div class="v">${escapeHtml(fmtInt(run.weightsStep))}</div></div>
                  <div class="run-stat"><div class="k">Latest Checkpoint</div><div class="v">${escapeHtml(fmtInt(run.latestCheckpointStep))}</div></div>
                  <div class="run-stat"><div class="k">Weights Reason</div><div class="v">${escapeHtml(run.weights_reason || 'n/a')}</div></div>
                  <div class="run-stat"><div class="k">Updated ISO</div><div class="v mono">${escapeHtml(run.updated_iso || 'n/a')}</div></div>
                </div>
                ${renderNextActions(run)}
                ${renderCommandsPanel(run)}
              </details>
            </div>
          </article>
        `;
      }).join('');
    }

    function renderRunTable(filteredRuns) {
      if (!filteredRuns.length) {
        els.runGrid.innerHTML = '';
        els.emptyState.hidden = false;
        return;
      }

      els.emptyState.hidden = true;
      els.runGrid.innerHTML = `
        <div class="table-wrap">
          <table class="run-table">
            <thead>
              <tr>
                <th class="col-run">Run</th>
                <th class="col-model">Model / Shape</th>
                <th class="col-status">Status</th>
                <th class="col-metrics">Loss / SVG</th>
                <th class="col-weights">Weights / Ckpt</th>
                <th class="col-updated">Updated</th>
                <th class="col-actions">Actions</th>
              </tr>
            </thead>
            <tbody>
              ${filteredRuns.map((run) => {
                const tone = parityTone(run.parityStatus);
                return `
                  <tr>
                    <td>
                      <div class="table-primary">${escapeHtml(run.name)}</div>
                      <div class="table-secondary">${escapeHtml(run.rel_path)}</div>
                    </td>
                    <td>
                      <div class="table-primary">${escapeHtml(run.modelSpec)}</div>
                      <div class="table-secondary tight">${escapeHtml(run.shape_signature || 'shape n/a')}</div>
                    </td>
                    <td>
                      <div class="table-badges">
                        <span class="badge ${run.kind}">${escapeHtml(run.kind)}</span>
                        ${showParityBadge(run) ? `<span class="badge ${tone}">${escapeHtml(run.parityStatus)}</span>` : ''}
                        ${run.reportReady ? '<span class="badge report">report</span>' : ''}
                        ${run.datasetViewerReady ? '<span class="badge report">dataset</span>' : ''}
                      </div>
                      <div class="table-secondary tight">${escapeHtml(run.healthReason || 'coverage unavailable')}</div>
                    </td>
                    <td>
                      <div class="table-metric">
                        <div class="row"><span class="label">Loss</span><span class="value">${escapeHtml(fmtLoss(run.final_loss))}</span></div>
                        <div class="row"><span class="label">SVG</span><span class="value">${escapeHtml(fmtPct(run.validSvgRate))}</span></div>
                      </div>
                    </td>
                    <td>
                      <div class="table-metric">
                        <div class="row"><span class="label">Step</span><span class="value">${escapeHtml(fmtInt(run.weightsStep))}</span></div>
                        <div class="row"><span class="label">Ckpt</span><span class="value">${escapeHtml(fmtInt(run.checkpointCount))}</span></div>
                      </div>
                      <span class="table-reason" title="${escapeHtml(run.weights_reason || 'n/a')}">${escapeHtml(run.weights_reason || 'n/a')}</span>
                    </td>
                    <td>
                      <div class="table-primary">${escapeHtml(run.updatedLabel)}</div>
                      <div class="table-secondary tight mono">${escapeHtml(run.updated_iso || 'n/a')}</div>
                    </td>
                    <td>
                      <div class="table-actions">
                        ${run.report_uri ? `<a class="btn primary" target="_blank" rel="noopener" href="${escapeHtml(run.report_uri)}">Report</a>` : ''}
                        ${run.dataset_viewer_uri ? `<a class="btn dataset" target="_blank" rel="noopener" href="${escapeHtml(run.dataset_viewer_uri)}">Dataset</a>` : ''}
                        ${embViewerLink(run, '🧬')}
                        ${attnViewerLink(run, '🔭')}
                        ${run.gallery_uri ? `<a class="btn" target="_blank" rel="noopener" href="${escapeHtml(run.gallery_uri)}">Gallery</a>` : ''}
                        <a class="btn" target="_blank" rel="noopener" href="${escapeHtml(run.run_uri)}">Run</a>
                        ${selectionButton(run)}
                        <button class="btn" data-copy="${encodeURIComponent(run.generate_report_cmd || '')}">Cmd</button>
                      </div>
                    </td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>
      `;
    }

    function refresh() {
      const filteredRuns = applyFilters();
      syncPresentationMode();
      renderResultsSummary(filteredRuns);
      renderComparePanel(filteredRuns);
      renderCoverage(filteredRuns);
      if (state.view !== 'table') {
        renderSpotlight(filteredRuns);
      }
      if (state.view === 'table') {
        renderRunTable(filteredRuns);
      } else {
        renderRunCards(filteredRuns);
      }
    }

    function bind() {
      els.searchInput.addEventListener('input', (event) => {
        state.search = event.target.value || '';
        refresh();
      });
      els.kindFilter.addEventListener('change', (event) => {
        state.kind = event.target.value || 'all';
        refresh();
      });
      els.parityFilter.addEventListener('change', (event) => {
        state.parity = event.target.value || 'all';
        refresh();
      });
      els.reportFilter.addEventListener('change', (event) => {
        state.report = event.target.value || 'all';
        refresh();
      });
      els.svgFilter.addEventListener('change', (event) => {
        state.svg = event.target.value || 'all';
        refresh();
      });
      els.sortSelect.addEventListener('change', (event) => {
        state.sort = event.target.value || 'updated_desc';
        refresh();
      });
      els.viewSelect.addEventListener('change', (event) => {
        state.view = event.target.value || 'cards';
        savePreference('ck_v7_run_hub_view', state.view);
        refresh();
      });
      els.densitySelect.addEventListener('change', (event) => {
        state.density = event.target.value || 'comfortable';
        savePreference('ck_v7_run_hub_density', state.density);
        refresh();
      });

      document.addEventListener('click', async (event) => {
        const toggle = event.target.closest('[data-toggle-select]');
        if (toggle) {
          const relPath = decodeURIComponent(toggle.getAttribute('data-toggle-select') || '');
          toggleSelection(relPath);
          refresh();
          return;
        }

        const auto = event.target.closest('[data-compare-auto]');
        if (auto) {
          const relPath = decodeURIComponent(auto.getAttribute('data-compare-auto') || '');
          autoSelectSimilar(relPath);
          refresh();
          return;
        }

        const clear = event.target.closest('[data-clear-compare]');
        if (clear) {
          clearSelection();
          refresh();
          return;
        }

        const openCompare = event.target.closest('[data-open-compare]');
        if (openCompare) {
          const selected = selectedRuns();
          if (selected.length >= 2) {
            const popup = window.open('', '_blank', 'noopener');
            if (popup) {
              popup.document.open();
              popup.document.write(renderStandaloneCompareHtml(selected));
              popup.document.close();
            } else {
              window.alert('Popup blocked. Allow popups for the hub to open the compare page.');
            }
          }
          return;
        }

        const button = event.target.closest('[data-copy]');
        if (!button) return;
        const raw = button.getAttribute('data-copy') || '';
        const cmd = decodeURIComponent(raw);
        if (!cmd) return;
        const original = button.textContent;
        try {
          await navigator.clipboard.writeText(cmd);
          button.textContent = 'Copied';
          setTimeout(() => { button.textContent = original; }, 1200);
        } catch (error) {
          window.prompt('Copy command', cmd);
        }
      });
    }

    renderHero();
    renderLegend();
    renderActions();
    syncPresentationMode();
    bind();
    refresh();
  </script>
</body>
</html>
""".replace("__HUB_DATA__", data_json)
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a parent IR Run Hub page for CK v7 runs.")
    p.add_argument(
        "--models-root",
        type=Path,
        default=Path.home() / ".cache" / "ck-engine-v7" / "models",
        help="Root directory that contains run/model folders (default: ~/.cache/ck-engine-v7/models)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <models-root>/ir_hub.html)",
    )
    p.add_argument(
        "--index-out",
        type=Path,
        default=None,
        help="Optional JSON index output path (default: <models-root>/runs_hub_index.json)",
    )
    p.add_argument("--open", action="store_true", help="Open generated hub in default browser.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    models_root = args.models_root.expanduser().resolve()
    if not models_root.exists():
        print(f"Error: models root not found: {models_root}")
        return 1

    output = args.output.expanduser().resolve() if args.output else (models_root / "ir_hub.html")
    index_out = args.index_out.expanduser().resolve() if args.index_out else (models_root / "runs_hub_index.json")

    index_payload = build_index(models_root)
    html = render_html(index_payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    index_out.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    index_out.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    print(f"Generated hub:   {output}")
    print(f"Generated index: {index_out}")
    print(f"Runs: {index_payload.get('summary', {}).get('runs_total', 0)}")

    if args.open:
        try:
            webbrowser.open(output.resolve().as_uri())
        except Exception as e:
            print(f"Warning: failed to open browser: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
