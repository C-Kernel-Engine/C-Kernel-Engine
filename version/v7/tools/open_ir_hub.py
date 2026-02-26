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
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CK v7 Run Hub</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg:      #232323;
      --surface: #2a2a2a;
      --card:    #323232;
      --line:    #454545;
      --sub:     #3a3a3a;
      --text:    #f5f5f5;
      --muted:   #b0b0b0;
      --dim:     #808080;
      --orange:  #ffb400;
      --orange-d:#e5a200;
      --orange-l:#ffc933;
      --blue:    #07adf8;
      --green:   #47b475;
      --red:     #ef4444;
      --warn:    #ffd166;
      --shadow:  0 4px 12px rgba(0,0,0,0.4);
      --radius:  8px;
      --tr:      all 0.2s ease;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Space Grotesk', 'Avenir Next', 'Segoe UI', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      line-height: 1.6;
    }}

    /* ── HEADER ── */
    .site-header {{
      background: linear-gradient(135deg, #2a2a2a 0%, #363636 50%, #2a2a2a 100%);
      border-bottom: 3px solid var(--orange);
      padding: 0.75rem 2rem;
      display: flex;
      align-items: center;
      gap: 1rem;
      flex-wrap: wrap;
      justify-content: space-between;
      box-shadow: var(--shadow);
    }}
    .header-brand {{ display: flex; align-items: center; gap: 0.75rem; }}
    .header-logo {{
      width: 36px; height: 36px;
      background: var(--orange);
      border-radius: 6px;
      display: flex; align-items: center; justify-content: center;
      font-weight: 700; color: #2a2a2a; font-size: 0.82rem;
      font-family: 'JetBrains Mono', monospace;
      flex-shrink: 0;
    }}
    .header-text {{ display: flex; flex-direction: column; line-height: 1.2; }}
    .header-title {{ color: #f5f5f5; font-size: 1.15rem; font-weight: 700; }}
    .header-sub   {{ color: var(--orange); font-size: 0.7rem; font-weight: 500; }}
    .header-meta  {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--dim); text-align: right; }}

    /* ── LAYOUT ── */
    .wrap {{ padding: 1.5rem 2rem; }}

    /* ── STATS ── */
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 12px;
      margin-bottom: 1.25rem;
    }}
    .stat {{
      background: var(--card); border: 1px solid var(--sub);
      border-top: 3px solid var(--orange);
      border-radius: var(--radius); padding: 0.9rem 1.1rem;
    }}
    .stat.blue  {{ border-top-color: var(--blue);  }}
    .stat.green {{ border-top-color: var(--green); }}
    .stat-k {{ font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 0.2rem; }}
    .stat-v {{ font-size: 1.9rem; font-weight: 700; line-height: 1; }}

    /* ── CONTROLS ── */
    .controls {{ display: flex; gap: 8px; margin-bottom: 1rem; flex-wrap: wrap; align-items: center; }}
    .controls input, .controls select {{
      background: var(--card); color: var(--text);
      border: 1px solid var(--line); border-radius: var(--radius);
      padding: 7px 12px; font-family: inherit; font-size: 0.82rem; outline: none;
      transition: var(--tr);
    }}
    .controls input {{ min-width: 260px; }}
    .controls input:focus, .controls select:focus {{
      border-color: var(--orange);
      box-shadow: 0 0 0 2px rgba(255,180,0,0.12);
    }}

    /* ── RULES ── */
    .rules {{
      background: var(--card); border: 1px solid var(--sub);
      border-radius: var(--radius); padding: 0.8rem 1.1rem;
      margin-bottom: 1.1rem; font-size: 0.78rem; line-height: 1.9;
    }}
    .rules-label {{ font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); display: block; margin-bottom: 0.3rem; }}

    /* ── TABLE ── */
    .table-wrap {{
      background: var(--card); border: 1px solid var(--sub);
      border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow);
    }}
    table {{ width: 100%; table-layout: fixed; border-collapse: collapse; }}
    th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--sub); font-size: 0.75rem; vertical-align: top; overflow-wrap: anywhere; word-break: break-word; }}
    th {{
      text-align: left; color: var(--muted); font-size: 0.68rem; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.06em;
      position: sticky; top: 0; background: #1e1e1e; z-index: 1;
      border-bottom: 2px solid var(--orange);
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #2e2e2e; }}

    /* ── ROW KIND HIGHLIGHTS ── */
    tr.row-train td {{ background: rgba(7,173,248,0.04); }}
    tr.row-train td:first-child {{ box-shadow: inset 3px 0 0 rgba(7,173,248,0.55); }}
    tr.row-train:hover td {{ background: rgba(7,173,248,0.09); }}
    tr.row-inference td:first-child {{ box-shadow: inset 3px 0 0 rgba(255,180,0,0.30); }}
    tr.row-inference:hover td {{ background: #2e2e2e; }}

    /* ── LEGEND ── */
    .legend {{ display: flex; gap: 16px; margin-bottom: 0.75rem; font-size: 0.72rem; color: var(--muted); align-items: center; }}
    .leg-item {{ display: flex; align-items: center; gap: 5px; }}
    .leg-dot {{ width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }}
    .leg-train {{ background: rgba(7,173,248,0.25); border: 1px solid var(--blue); }}
    .leg-infer {{ background: rgba(255,180,0,0.12); border: 1px solid rgba(255,180,0,0.40); }}

    /* ── MISC ── */
    .mono  {{ font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 0.7rem; }}
    .small {{ font-size: 0.72rem; line-height: 1.35; }}
    .muted {{ color: var(--dim); }}
    .tight {{ line-height: 1.2; }}
    .status {{ display: flex; flex-direction: column; gap: 2px; }}
    .kind   {{ margin-right: 5px; }}
    .badge  {{ display: inline-block; border: 1px solid; border-radius: 999px; padding: 1px 7px; font-size: 0.68rem; font-weight: 600; }}
    .b-good  {{ border-color: #1d6644; color: var(--green); }}
    .b-bad   {{ border-color: #7a2f2f; color: var(--red); }}
    .b-warn  {{ border-color: #7a6a2f; color: var(--warn); }}
    .b-muted {{ border-color: var(--line); color: var(--muted); }}
    a        {{ color: var(--blue); text-decoration: none; }}
    a:hover  {{ color: var(--orange-l); text-decoration: underline; }}
    .actions a {{ margin-right: 6px; white-space: nowrap; }}
    .copy-btn {{
      margin-top: 4px; font-size: 0.68rem; padding: 2px 7px;
      border-radius: 4px; border: 1px solid var(--line);
      background: var(--surface); color: var(--muted); cursor: pointer;
      transition: var(--tr); font-family: inherit;
    }}
    .copy-btn:hover {{ border-color: var(--orange); color: var(--orange); }}
    .cmdline {{
      margin-top: 4px; background: #1a1a1a; border: 1px solid var(--sub);
      border-radius: 5px; padding: 5px 8px;
      font-family: 'JetBrains Mono', monospace; font-size: 0.66rem;
      line-height: 1.45; word-break: break-all;
    }}
    details summary {{ cursor: pointer; color: var(--blue); font-size: 0.72rem; }}
  </style>
</head>
<body>
  <header class="site-header">
    <div class="header-brand">
      <div class="header-logo">CK</div>
      <div class="header-text">
        <span class="header-title">CK v7 Run Hub</span>
        <span class="header-sub">IR Visualizer · Training · Inference · Parity</span>
      </div>
    </div>
    <div class="header-meta" id="meta"></div>
  </header>

  <div class="wrap">
    <div class="stats">
      <div class="stat">
        <div class="stat-k">Total Runs</div>
        <div class="stat-v" id="s-runs">0</div>
      </div>
      <div class="stat blue">
        <div class="stat-k">Train Runs</div>
        <div class="stat-v" id="s-train" style="color:var(--blue)">0</div>
      </div>
      <div class="stat green">
        <div class="stat-k">With ir_report</div>
        <div class="stat-v" id="s-report" style="color:var(--green)">0</div>
      </div>
      <div class="stat">
        <div class="stat-k">Parity PASS</div>
        <div class="stat-v" id="s-pass" style="color:var(--orange)">0</div>
      </div>
    </div>
    <div class="controls">
      <input id="q" placeholder="Search run / path…" />
      <select id="kind">
        <option value="all">All kinds</option>
        <option value="train">Train</option>
        <option value="inference">Inference</option>
      </select>
    </div>
    <div class="legend">
      <span class="leg-item"><span class="leg-dot leg-train"></span>Train run</span>
      <span class="leg-item"><span class="leg-dot leg-infer"></span>Inference run</span>
    </div>
    <div class="rules">
      <span class="rules-label">Reuse Rules</span>
      <span class="badge b-good">same-shape: YES</span> Continue from checkpoints/weights in same run.&nbsp;&nbsp;
      <span class="badge b-warn">context_len change: MAYBE</span> Reusable with same weight shape; regenerate IR/runtime and re-run parity gates.&nbsp;&nbsp;
      <span class="badge b-bad">embed/vocab/layer/head change: NO</span> Weight shapes change; start a new initialization or migration.
    </div>
    <div class="table-wrap">
      <table>
        <colgroup>
          <col style="width:22%;" />
          <col style="width:13%;" />
          <col style="width:14%;" />
          <col style="width:20%;" />
          <col style="width:12%;" />
          <col style="width:11%;" />
          <col style="width:8%;" />
        </colgroup>
        <thead>
          <tr>
            <th>Run</th>
            <th>Dims</th>
            <th>Status</th>
            <th>Weights / Ckpt</th>
            <th>Reuse</th>
            <th>Updated (UTC)</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </div>
  <script>
    const HUB = {data_json};
    const rowsEl = document.getElementById('rows');
    const qEl = document.getElementById('q');
    const kindEl = document.getElementById('kind');

    const esc = (v) => String(v ?? '').replace(/[&<>\"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}}[m]));
    const fmt = (v, n=4) => (typeof v === 'number' && isFinite(v)) ? v.toFixed(n) : '-';
    const copyCmd = async (btn) => {{
      const cmd = btn?.dataset?.cmd || '';
      if (!cmd) return;
      try {{
        await navigator.clipboard.writeText(cmd);
        const prev = btn.textContent;
        btn.textContent = 'Copied';
        setTimeout(() => {{ btn.textContent = prev; }}, 1000);
      }} catch (_) {{
        // no-op
      }}
    }};
    const parityBadge = (status) => {{
      if (status === 'PASS') return '<span class=\"badge b-good\">PASS</span>';
      if (status === 'PASS_REUSED') return '<span class=\"badge b-good\">PASS(reused)</span>';
      if (status === 'FAIL') return '<span class=\"badge b-bad\">FAIL</span>';
      if (status === 'SKIP') return '<span class=\"badge b-warn\">SKIP</span>';
      return '<span class=\"badge b-muted\">MISSING</span>';
    }};

    function render() {{
      const q = (qEl.value || '').toLowerCase().trim();
      const kind = kindEl.value;
      const runs = HUB.runs.filter(r => {{
        if (kind !== 'all' && r.kind !== kind) return false;
        if (!q) return true;
        const hay = `${{r.name}} ${{r.rel_path}} ${{r.run_dir}}`.toLowerCase();
        return hay.includes(q);
      }});

      rowsEl.innerHTML = runs.map(r => {{
        const d = r.dims || {{}};
        const dim = `L${{d.num_layers ?? '-'}} d${{d.embed_dim ?? '-'}} h${{d.hidden_size ?? '-'}} v${{d.vocab_size ?? '-'}} ctx${{d.context_len ?? '-'}}`;
        const actions = [];
        if (r.report_uri) actions.push(`<a href=\"${{esc(r.report_uri)}}\" target=\"_blank\" rel=\"noopener\">Open report</a>`);
        if (r.run_uri) actions.push(`<a href=\"${{esc(r.run_uri)}}\" target=\"_blank\" rel=\"noopener\">Open dir</a>`);
        if (!r.report_uri && r.generate_report_cmd) {{
          actions.push(`
            <details class=\"small\" style=\"margin-top:6px;\">
              <summary>Generate report</summary>
              <div class=\"cmdline mono\">${{esc(r.generate_report_cmd)}}</div>
              <button class=\"copy-btn\" data-cmd=\"${{esc(r.generate_report_cmd)}}\" onclick=\"copyCmd(this)\">Copy</button>
            </details>
          `);
        }}
        const weightsCell = `
          <div class=\"mono small tight\">step=${{esc(r.weights_step ?? '-')}} · reason=${{esc(r.weights_reason ?? '-')}}</div>
          <div class=\"mono small tight\">ckpt=${{esc(r.checkpoint_count ?? 0)}} · latest=${{esc(r.latest_checkpoint_step ?? '-')}}</div>
          <div class=\"mono small muted tight\">${{esc(r.shape_signature ?? 'shape=unknown')}}</div>
        `;
        const statusCell = `
          <div class=\"status\">
            <div>${{parityBadge((r.parity_regimen || {{}}).status)}}</div>
            <div class=\"mono small tight\">loss=${{fmt(r.final_loss, 5)}}</div>
            <div class=\"mono small tight\">svg=${{fmt(r.valid_svg_rate, 4)}}</div>
          </div>
        `;
        const reuseCell = `
          <div class=\"small\"><span class=\"badge b-good\">shape YES</span></div>
          <div class=\"small\"><span class=\"badge b-warn\">ctx MAYBE</span></div>
          <div class=\"small\"><span class=\"badge b-bad\">dim NO</span></div>
        `;
        return `<tr class="row-${{r.kind}}">
          <td><div><span class=\"badge b-muted kind\">${{esc(r.kind)}}</span><strong>${{esc(r.name)}}</strong></div><div class=\"mono muted\">${{esc(r.rel_path)}}</div></td>
          <td class=\"mono\">${{esc(dim)}}</td>
          <td>${{statusCell}}</td>
          <td>${{weightsCell}}</td>
          <td>${{reuseCell}}</td>
          <td class=\"mono\">${{esc(r.updated_iso || '-')}}</td>
          <td class=\"actions\">${{actions.join('')}}</td>
        </tr>`;
      }}).join('');
    }}

    document.getElementById('meta').textContent =
      `generated_at=${{HUB.generated_at}}  models_root=${{HUB.models_root}}`;
    document.getElementById('s-runs').textContent = HUB.summary.runs_total ?? 0;
    document.getElementById('s-train').textContent = HUB.summary.runs_train ?? 0;
    document.getElementById('s-report').textContent = HUB.summary.runs_with_report ?? 0;
    document.getElementById('s-pass').textContent = HUB.summary.runs_parity_pass ?? 0;
    qEl.addEventListener('input', render);
    kindEl.addEventListener('change', render);
    render();
  </script>
</body>
</html>
"""


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
