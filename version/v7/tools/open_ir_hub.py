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
  <style>
    :root {{
      --bg: #111318;
      --panel: #1a1f28;
      --line: #2c3442;
      --text: #e6edf3;
      --muted: #9fb0c0;
      --accent: #ffb300;
      --good: #4ad295;
      --bad: #ff6b6b;
      --warn: #ffd166;
    }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
    .wrap {{ max-width: 100%; margin: 0 auto; padding: 12px; }}
    h1 {{ margin: 0 0 10px; font-size: 24px; }}
    .meta {{ color: var(--muted); font-size: 13px; margin-bottom: 14px; }}
    .stats {{ display:grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap:10px; margin-bottom: 14px; }}
    .card {{ background: var(--panel); border:1px solid var(--line); border-radius: 8px; padding: 10px; }}
    .k {{ color: var(--muted); font-size: 12px; }}
    .v {{ font-size: 20px; font-weight: 700; }}
    .controls {{ display:flex; gap:8px; margin-bottom: 12px; flex-wrap: wrap; }}
    input, select {{ background:#121722; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:7px 10px; }}
    input {{ min-width: 280px; }}
    table {{ width:100%; table-layout: fixed; border-collapse: collapse; background: var(--panel); border:1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 6px 8px; border-bottom: 1px solid var(--line); font-size: 12px; vertical-align: top; overflow-wrap:anywhere; word-break:break-word; }}
    th {{ text-align:left; color:#d4deea; position: sticky; top: 0; background:#1d2430; z-index:1; }}
    tr:hover {{ background:#202838; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
    .muted {{ color: var(--muted); }}
    .badge {{ display:inline-block; border:1px solid var(--line); border-radius: 999px; padding:2px 8px; font-size: 11px; }}
    .b-good {{ border-color:#2f7a5e; color:var(--good); }}
    .b-bad {{ border-color:#7f3b3b; color:var(--bad); }}
    .b-warn {{ border-color:#7a6a2f; color:var(--warn); }}
    .b-muted {{ border-color:var(--line); color:var(--muted); }}
    a {{ color: #8fc2ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .actions a {{ margin-right: 6px; white-space: nowrap; }}
    .copy-btn {{ margin-top: 6px; font-size: 11px; padding: 2px 8px; border-radius: 6px; border: 1px solid var(--line); background:#121722; color:var(--text); cursor:pointer; }}
    .copy-btn:hover {{ border-color:#3b4b62; }}
    .cmdline {{ margin-top: 6px; background:#121722; border:1px solid var(--line); border-radius:6px; padding:6px; }}
    .small {{ font-size: 11px; line-height: 1.35; }}
    .rules {{ margin-bottom: 12px; }}
    .status {{ display:flex; flex-direction:column; gap:3px; }}
    .tight {{ line-height: 1.2; }}
    .kind {{ margin-right: 6px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>CK v7 Run Hub</h1>
    <div class="meta mono" id="meta"></div>
    <div class="stats">
      <div class="card"><div class="k">Runs</div><div class="v" id="s-runs">0</div></div>
      <div class="card"><div class="k">Train Runs</div><div class="v" id="s-train">0</div></div>
      <div class="card"><div class="k">With ir_report</div><div class="v" id="s-report">0</div></div>
      <div class="card"><div class="k">Parity PASS</div><div class="v" id="s-pass">0</div></div>
    </div>
    <div class="controls">
      <input id="q" placeholder="Search run/path..." />
      <select id="kind">
        <option value="all">All kinds</option>
        <option value="train">Train</option>
        <option value="inference">Inference</option>
      </select>
    </div>
    <div class="card rules small">
      <strong>Reuse Rules</strong><br/>
      <span class="badge b-good">same-shape: YES</span> Continue from checkpoints/weights in same run.<br/>
      <span class="badge b-warn">context_len change: MAYBE</span> Usually reusable with same weight shape, but regenerate IR/runtime and re-run parity gates.<br/>
      <span class="badge b-bad">embed/vocab/layer/head change: NO</span> Weight shapes change; start a new compatible initialization or migration.
    </div>
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
        return `<tr>
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
