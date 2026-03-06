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


MARKER_FILES = {
    "run_index.json",
    "ir_report.html",
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


def _find_report_path(run_dir: Path) -> Path | None:
    direct = run_dir / "ir_report.html"
    if direct.exists():
        return direct
    ck_build = run_dir / ".ck_build" / "ir_report.html"
    if ck_build.exists():
        return ck_build
    return None


def _extract_dims(weights_manifest: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
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


def _extract_manifest_info(weights_manifest: Path) -> tuple[int | None, str | None]:
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


def _extract_final_loss(path: Path) -> float | None:
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


def _extract_valid_svg_rate(path: Path) -> float | None:
    obj = _safe_read_json(path)
    if not isinstance(obj, dict):
        return None
    v = obj.get("valid_svg_rate")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _extract_parity_regimen(path: Path) -> dict[str, Any]:
    status = {"status": "MISSING", "passed": None, "generated_at": None}
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


@dataclass
class RunRecord:
    run_dir: Path
    rel_path: str
    name: str
    kind: str
    report_path: Path | None
    dims: dict[str, Any]
    parity_regimen: dict[str, Any]
    final_loss: float | None
    valid_svg_rate: float | None
    checkpoint_count: int
    latest_checkpoint_step: int | None
    latest_checkpoint_bump: Path | None
    latest_checkpoint_manifest: Path | None
    weights_step: int | None
    weights_reason: str | None
    shape_signature: str | None
    generate_report_cmd: str
    updated_epoch: float
    updated_iso: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "run_uri": self.run_dir.resolve().as_uri(),
            "rel_path": self.rel_path,
            "name": self.name,
            "kind": self.kind,
            "report_path": str(self.report_path) if self.report_path else None,
            "report_uri": self.report_path.resolve().as_uri() if self.report_path else None,
            "dims": self.dims,
            "parity_regimen": self.parity_regimen,
            "final_loss": self.final_loss,
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
            "updated_epoch": self.updated_epoch,
            "updated_iso": self.updated_iso,
        }


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
    wm = run_dir / "weights_manifest.json"
    parity = run_dir / "training_parity_regimen_latest.json"
    loss = run_dir / "training_loss_curve_latest.json"
    post_eval = run_dir / "post_train_eval.json"
    run_index = run_dir / "run_index.json"

    dims = _extract_dims(wm)
    weights_step, weights_reason = _extract_manifest_info(wm)
    parity_status = _extract_parity_regimen(parity)
    final_loss = _extract_final_loss(loss)
    valid_svg_rate = _extract_valid_svg_rate(post_eval)
    latest_ckpt_step, latest_ckpt_bump, latest_ckpt_manifest, ckpt_count = _latest_checkpoint(run_dir)

    mtimes = []
    for p in (report, wm, parity, loss, post_eval, run_index):
        m = _file_mtime(p)
        if m is not None:
            mtimes.append(m)
    updated_epoch = max(mtimes) if mtimes else 0.0
    updated_iso = _epoch_to_iso(updated_epoch)

    return RunRecord(
        run_dir=run_dir,
        rel_path=rel,
        name=run_dir.name,
        kind=_infer_kind(run_dir, models_root),
        report_path=report,
        dims=dims,
        parity_regimen=parity_status,
        final_loss=final_loss,
        valid_svg_rate=valid_svg_rate,
        checkpoint_count=ckpt_count,
        latest_checkpoint_step=latest_ckpt_step,
        latest_checkpoint_bump=latest_ckpt_bump,
        latest_checkpoint_manifest=latest_ckpt_manifest,
        weights_step=weights_step,
        weights_reason=weights_reason,
        shape_signature=_shape_signature(dims),
        generate_report_cmd=_build_generate_report_cmd(run_dir),
        updated_epoch=updated_epoch,
        updated_iso=updated_iso,
    )


def build_index(models_root: Path) -> dict[str, Any]:
    runs = [collect_run_record(r, models_root) for r in discover_run_dirs(models_root)]
    runs.sort(key=lambda r: r.updated_epoch, reverse=True)
    payload_runs = [r.to_json() for r in runs]
    train_count = sum(1 for r in payload_runs if r.get("kind") == "train")
    report_count = sum(1 for r in payload_runs if r.get("report_path"))
    pass_count = sum(1 for r in payload_runs if (r.get("parity_regimen") or {}).get("status") in ("PASS", "PASS_REUSED"))
    return {
        "schema": "ck.ir.hub.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_root": str(models_root),
        "summary": {
            "runs_total": len(payload_runs),
            "runs_train": train_count,
            "runs_with_report": report_count,
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
      line-height: 1.45;
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
      width: min(1540px, calc(100vw - 32px));
      margin: 22px auto 44px auto;
    }

    body.table-mode .page-shell {
      width: min(1760px, calc(100vw - 20px));
      margin-top: 14px;
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
      padding: 26px;
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
      gap: 10px;
    }

    .hero-tags { margin-top: 14px; }
    .hero-meta { margin-top: 14px; }

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
      gap: 8px;
      min-height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.022);
      color: var(--muted);
      font-size: 0.69rem;
      font-weight: 700;
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
      gap: 12px;
      margin-top: 22px;
    }

    body.compact-mode .metric-ribbon {
      gap: 9px;
      margin-top: 16px;
    }

    body.table-mode .metric-ribbon {
      gap: 10px;
      margin-top: 16px;
    }

    .metric-card {
      position: relative;
      overflow: hidden;
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.014));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }

    body.compact-mode .metric-card {
      padding: 12px 13px;
    }

    body.table-mode .metric-card {
      padding: 12px 14px;
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
      gap: 18px;
      margin-top: 22px;
    }

    body.compact-mode .workspace {
      gap: 14px;
      margin-top: 16px;
    }

    body.table-mode .workspace {
      grid-template-columns: 1fr;
      gap: 14px;
      margin-top: 16px;
    }

    .stack {
      display: grid;
      gap: 18px;
      align-content: start;
    }

    body.compact-mode .stack {
      gap: 14px;
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
      padding: 18px 20px 0 20px;
    }

    .panel-head h2,
    .panel-head h3 {
      margin: 0;
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
    }

    .panel-sub {
      margin: 10px 0 0 0;
      padding: 0 20px;
      color: var(--muted);
      font-size: 0.9rem;
    }

    body.table-mode .panel-head {
      padding: 14px 18px 0 18px;
    }

    body.table-mode .panel-sub {
      padding: 0 18px;
      font-size: 0.82rem;
    }

    .toolbar-body,
    .spotlight-body,
    .rail-body {
      padding: 18px 20px 20px 20px;
    }

    body.compact-mode .toolbar-body,
    body.compact-mode .spotlight-body,
    body.compact-mode .rail-body {
      padding: 14px 16px 16px 16px;
    }

    body.table-mode .toolbar-body,
    body.table-mode .spotlight-body,
    body.table-mode .rail-body {
      padding: 14px 18px 18px 18px;
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
      border-top: 1px solid rgba(255,255,255,0.05);
      padding-top: 12px;
    }

    .detail-toggle summary {
      list-style: none;
      cursor: pointer;
      color: var(--cyan);
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.02em;
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
      padding: 13px 14px;
      border-radius: 16px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      background: rgba(255, 255, 255, 0.03);
    }

    .summary-tile .k,
    .mini-tile .k,
    .run-stat .k,
    .kv-row .k {
      color: var(--dim);
      font-size: 0.66rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-weight: 700;
      margin-bottom: 6px;
    }

    .summary-tile .v,
    .mini-tile .v,
    .run-stat .v {
      font-size: 1rem;
      line-height: 1.15;
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

    .run-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
      gap: 16px;
    }

    body.compact-mode .run-grid {
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 12px;
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
      padding: 18px;
      display: grid;
      gap: 14px;
    }

    body.compact-mode .run-card-body {
      padding: 14px;
      gap: 10px;
    }

    .run-name {
      margin: 0;
      font-size: 1.08rem;
      line-height: 1.08;
      letter-spacing: -0.04em;
      word-break: break-word;
    }

    body.compact-mode .run-name {
      font-size: 1.02rem;
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
      gap: 8px;
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
      gap: 10px;
    }

    body.compact-mode .run-stats {
      gap: 8px;
    }

    .run-stat {
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.035);
      background: rgba(255,255,255,0.014);
    }

    body.compact-mode .run-stat {
      padding: 10px 11px;
      border-radius: 14px;
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
      font-size: 0.76rem;
    }

    .run-card .coverage-label {
      color: #a8afb7;
      font-size: 0.76rem;
      font-weight: 600;
    }

    .run-card .detail-toggle {
      border-top-color: rgba(255,255,255,0.035);
      padding-top: 10px;
    }

    .run-card .detail-toggle summary {
      color: #91a6b0;
      font-size: 0.74rem;
      font-weight: 600;
    }

    .rail {
      display: grid;
      gap: 18px;
      align-content: start;
    }

    body.table-mode .rail {
      display: none;
    }

    .rail-card {
      position: sticky;
      top: 18px;
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
          <p class="eyebrow">C-Kernel-Engine / Version 7</p>
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

    function scoreRun(run) {
      let score = 18;
      if (run.reportReady) score += 24;
      if (run.parityStatus === 'PASS') score += 28;
      else if (run.parityStatus === 'PASS_REUSED') score += 22;
      else if (run.parityStatus === 'FAIL') score += 4;
      if (run.validSvgRate !== null) {
        if (run.validSvgRate >= 0.9) score += 14;
        else if (run.validSvgRate >= 0.75) score += 10;
        else if (run.validSvgRate >= 0.5) score += 6;
        else score += 3;
      }
      if (run.checkpointCount > 0) score += 8;
      if (run.weightsStep !== null) score += 6;
      return Math.max(0, Math.min(100, score));
    }

    function makeSearchBlob(run) {
      return [
        run.name,
        run.rel_path,
        run.kind,
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
        reportReady: Boolean(run.report_path),
        validSvgRate: typeof run.valid_svg_rate === 'number' ? run.valid_svg_rate : null,
        checkpointCount: typeof run.checkpoint_count === 'number' ? run.checkpoint_count : 0,
        latestCheckpointStep: typeof run.latest_checkpoint_step === 'number' ? run.latest_checkpoint_step : null,
        weightsStep: typeof run.weights_step === 'number' ? run.weights_step : null,
        modelSpec: makeModelSpec(dims),
        updatedLabel: relativeTime(run.updated_epoch),
      };
      normalized.healthScore = scoreRun(normalized);
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
    };

    function renderHero() {
      const summary = HUB.summary || {};
      const newest = runs[0];
      const reportCount = summary.runs_with_report || runs.filter((run) => run.reportReady).length;
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
            <span class="badge ${tone}">${escapeHtml(run.parityStatus)}</span>
            ${run.reportReady ? '<span class="badge report">report ready</span>' : ''}
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
                <a class="btn" target="_blank" rel="noopener" href="${escapeHtml(run.run_uri)}">Open run dir</a>
                <button class="btn" data-copy="${encodeURIComponent(run.generate_report_cmd || '')}">Copy cmd</button>
              </div>
              <details class="detail-toggle">
                <summary>Advanced details</summary>
                <div class="spotlight-command">${escapeHtml(run.generate_report_cmd || 'No report generation command available')}</div>
                <div class="spotlight-summary" style="margin-top:12px;">
                  <div class="summary-tile"><div class="k">Shape Signature</div><div class="v">${escapeHtml(run.shape_signature || 'n/a')}</div></div>
                  <div class="summary-tile"><div class="k">Weights Reason</div><div class="v">${escapeHtml(run.weights_reason || 'n/a')}</div></div>
                  <div class="summary-tile"><div class="k">Updated ISO</div><div class="v mono">${escapeHtml(run.updated_iso || 'n/a')}</div></div>
                  <div class="summary-tile"><div class="k">Checkpoint Step</div><div class="v">${escapeHtml(fmtInt(run.latestCheckpointStep))}</div></div>
                </div>
              </details>
            </div>
            <aside class="spotlight-side">
              <div class="health-ring" style="--ring:${ring}deg;">
                <div class="health-core">
                  <div>
                    <strong>${escapeHtml(String(run.healthScore))}</strong>
                    <span>Health</span>
                  </div>
                </div>
              </div>
              <div class="mini-grid">
                <div class="mini-tile"><div class="k">Updated</div><div class="v">${escapeHtml(run.updatedLabel)}</div></div>
                <div class="mini-tile"><div class="k">Parity</div><div class="v">${escapeHtml(run.parityStatus)}</div></div>
                <div class="mini-tile"><div class="k">Report Surface</div><div class="v">${run.reportReady ? 'Ready for direct entry' : 'Generate report first'}</div></div>
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

    function renderCoverage(filteredRuns) {
      const target = filteredRuns.length ? filteredRuns : runs;
      const reportPct = makeCoveragePct(target, (run) => run.reportReady);
      const parityPct = makeCoveragePct(target, (run) => run.parityStatus === 'PASS' || run.parityStatus === 'PASS_REUSED');
      const svgPct = makeCoveragePct(target, (run) => run.validSvgRate !== null);
      const svgStrongPct = makeCoveragePct(target, (run) => run.validSvgRate !== null && run.validSvgRate >= 0.8);
      const ckptPct = makeCoveragePct(target, (run) => run.checkpointCount > 0);

      const items = [
        ['Report surface', reportPct],
        ['Parity pass', parityPct],
        ['SVG telemetry', svgPct],
        ['Strong SVG telemetry', svgStrongPct],
        ['Checkpointed roots', ckptPct],
      ];

      els.coveragePanel.innerHTML = `
        <div class="panel-head"><h3>Coverage Matrix</h3></div>
        <div class="panel-sub">Coverage is scoped to the visible result set, so the bars move with your filters.</div>
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
            <div class="kv-row"><span class="k">Generate Report</span><span class="v mono">python3 version/v7/tools/open_ir_visualizer.py --generate --run /path/to/run --html-only --strict-run-artifacts --output /path/to/run/ir_report.html</span></div>
            <div class="kv-row"><span class="k">Checkpoint History</span><span class="v mono">python3 version/v7/scripts/promote_latest_checkpoint_v7.py --run /path/to/run --list-runs</span></div>
          </div>
          <div class="codebox" style="margin-top:14px;">Use the hub to select the run. Use the terminal to regenerate or promote artifacts.</div>
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
        `<span class="result-pill"><strong>${avgHealth}</strong> avg health</span>`,
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
                  <span class="badge ${tone}">${escapeHtml(run.parityStatus)}</span>
                  ${run.reportReady ? '<span class="badge report">report</span>' : ''}
                </div>
              </div>

              <div class="run-health">
                <div class="coverage-label"><span>Run health</span><strong>${escapeHtml(String(run.healthScore))}/100</strong></div>
                <div class="health-bar"><span style="width:${Math.max(0, Math.min(100, run.healthScore))}%"></span></div>
              </div>

              <div class="run-stats">
                <div class="run-stat"><div class="k">Updated</div><div class="v">${escapeHtml(run.updatedLabel)}</div></div>
                <div class="run-stat"><div class="k">Final Loss</div><div class="v">${escapeHtml(fmtLoss(run.final_loss))}</div></div>
              </div>

              <div class="run-actions">
                ${run.report_uri ? `<a class="btn primary" target="_blank" rel="noopener" href="${escapeHtml(run.report_uri)}">Report</a>` : ''}
                <a class="btn" target="_blank" rel="noopener" href="${escapeHtml(run.run_uri)}">Run dir</a>
                <button class="btn" data-copy="${encodeURIComponent(run.generate_report_cmd || '')}">Copy</button>
              </div>

              <details class="detail-toggle">
                <summary>Run details</summary>
                <div class="run-spec">${escapeHtml(run.modelSpec)}${run.shape_signature ? ` | ${escapeHtml(run.shape_signature)}` : ''}</div>
                <div class="run-stats">
                  <div class="run-stat"><div class="k">SVG Rate</div><div class="v">${escapeHtml(fmtPct(run.validSvgRate))}</div></div>
                  <div class="run-stat"><div class="k">Checkpoints</div><div class="v">${escapeHtml(fmtInt(run.checkpointCount))}</div></div>
                  <div class="run-stat"><div class="k">Weights Step</div><div class="v">${escapeHtml(fmtInt(run.weightsStep))}</div></div>
                  <div class="run-stat"><div class="k">Latest Checkpoint</div><div class="v">${escapeHtml(fmtInt(run.latestCheckpointStep))}</div></div>
                  <div class="run-stat"><div class="k">Weights Reason</div><div class="v">${escapeHtml(run.weights_reason || 'n/a')}</div></div>
                  <div class="run-stat"><div class="k">Updated ISO</div><div class="v mono">${escapeHtml(run.updated_iso || 'n/a')}</div></div>
                </div>
                <div class="codebox" style="margin-top:12px;">${escapeHtml(run.generate_report_cmd || 'No report generation command available')}</div>
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
                        <span class="badge ${tone}">${escapeHtml(run.parityStatus)}</span>
                        ${run.reportReady ? '<span class="badge report">report</span>' : ''}
                      </div>
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
                        <a class="btn" target="_blank" rel="noopener" href="${escapeHtml(run.run_uri)}">Run</a>
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
