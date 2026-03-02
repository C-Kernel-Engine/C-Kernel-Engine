#!/usr/bin/env python3
"""
Eval runner for CK Engine v7 training stages.

For each training run captured in run_ledger.jsonl:
1. Promotes the checkpoint (promote_latest_checkpoint_v7.py --run-id)
2. Builds the inference runtime (ck_run_v7.py run --generate-only)
3. Runs SVG probe prompts via ck_chat.py
4. Scores metrics: valid_svg_rate, prefix_integrity, eos_clean_stop,
   closure_success_rate, repetition_loop_score, ood_robustness, adherence
5. Writes/updates stage_eval_matrix.json in run_dir

Usage:
    python3 eval_stage_v7.py --run RUN_DIR --all-stages
    python3 eval_stage_v7.py --run RUN_DIR --stage sft --stage-pass 4
    python3 eval_stage_v7.py --run RUN_DIR --run-id ascii_bpe_20260301_171735
    python3 eval_stage_v7.py --run RUN_DIR --all-stages --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
V7_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = V7_ROOT.parent.parent

PROMOTE_SCRIPT = SCRIPT_DIR / "promote_latest_checkpoint_v7.py"
CK_RUN_SCRIPT = SCRIPT_DIR / "ck_run_v7.py"
CK_CHAT_SCRIPT = PROJECT_ROOT / "scripts" / "ck_chat.py"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA = "ck.stage_eval_matrix.v1"

# ---------------------------------------------------------------------------
# Probe definitions
# A probe has: id, prompt, type (svg_gen | ood), optional expect_* fields
# ---------------------------------------------------------------------------
PROBES = [
    {
        "id": "circle_cool_minimal",
        "prompt": "[circle][palette:cool][style:minimal]<svg",
        "type": "svg_gen",
        "expect_shape": "circle",
        "expect_palette": "cool",
        "expect_style": "minimal",
    },
    {
        "id": "bar_chart_warm_bold",
        "prompt": "[bar_chart][palette:warm][style:bold]<svg",
        "type": "svg_gen",
        "expect_shape": "bar_chart",
        "expect_palette": "warm",
        "expect_style": "bold",
    },
    {
        "id": "scatter_earth_clean",
        "prompt": "[scatter][palette:earth][style:clean]<svg",
        "type": "svg_gen",
        "expect_shape": "scatter",
        "expect_palette": "earth",
        "expect_style": "clean",
    },
    {
        "id": "line_chart_mono_minimal",
        "prompt": "[line_chart][palette:mono][style:minimal]<svg",
        "type": "svg_gen",
        "expect_shape": "line_chart",
        "expect_palette": "mono",
        "expect_style": "minimal",
    },
    {
        "id": "ood_unlabeled",
        "prompt": "<svg",
        "type": "ood",
        "expect_shape": None,
        "expect_palette": None,
        "expect_style": None,
    },
    {
        "id": "ood_unknown_type",
        "prompt": "[unknown_shape][palette:cool]<svg",
        "type": "ood",
        "expect_shape": None,
        "expect_palette": "cool",
        "expect_style": None,
    },
]

# ---------------------------------------------------------------------------
# Stage normalizer
# ---------------------------------------------------------------------------
_STAGE_ALIASES: dict[str, str] = {
    "stage_a": "pretrain",
    "pretrain": "pretrain",
    "stage_b": "midtrain",
    "midtrain": "midtrain",
    "sft": "sft",
    "dpo": "dpo",
    "grpo": "grpo",
    "ppo": "ppo",
    "rlhf": "ppo",
}


def _normalize_stage(raw: str) -> str:
    s = str(raw or "").strip().lower()
    return _STAGE_ALIASES.get(s, s or "pretrain")


# ---------------------------------------------------------------------------
# Ledger reader
# ---------------------------------------------------------------------------
def _read_ledger(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "run_ledger.jsonl"
    if not path.exists():
        return []
    by_run_id: dict[str, dict[str, Any]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict) and rec.get("run_id"):
                by_run_id[str(rec["run_id"])] = rec
    except OSError:
        return []
    return sorted(by_run_id.values(), key=lambda r: int(r.get("run_order") or 0))


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
_SVG_OPEN_RE = re.compile(r"<svg[\s>]", re.IGNORECASE)
_SVG_CLOSE_RE = re.compile(r"</svg\s*>", re.IGNORECASE)

# Palette heuristic colour sets (rough membership checks)
_PALETTE_COLORS: dict[str, set[str]] = {
    "cool": {"blue", "#0", "#1", "#2", "#3", "#4", "#5", "steelblue", "skyblue", "lightblue",
             "cornflowerblue", "royalblue", "navy", "teal", "cyan", "turquoise", "aqua"},
    "warm": {"red", "orange", "yellow", "coral", "tomato", "gold", "darkorange", "#e", "#f", "#d",
             "salmon", "crimson", "firebrick", "maroon", "sienna"},
    "earth": {"sienna", "brown", "tan", "khaki", "olive", "saddlebrown", "peru", "chocolate",
              "#8b", "#6b", "#a0", "bisque", "wheat"},
    "mono": {"gray", "grey", "silver", "black", "white", "#3", "#6", "#9", "#a", "#b", "#c", "#d", "#e"},
}

_SHAPE_SVG_TAGS: dict[str, list[str]] = {
    "circle": ["<circle", "cx="],
    "bar_chart": ["<rect"],
    "scatter": ["<circle", "<ellipse"],
    "line_chart": ["<line", "<polyline", "<path"],
}


def _is_valid_svg(text: str) -> bool:
    return bool(_SVG_OPEN_RE.search(text) and _SVG_CLOSE_RE.search(text))


def _has_closure(text: str) -> bool:
    return bool(_SVG_CLOSE_RE.search(text.rstrip()))


def _prefix_integrity(text: str) -> bool:
    return text.strip().startswith("<svg")


def _repetition_score(text: str, n: int = 5) -> float:
    """Fraction of n-grams that are repeated (0=clean, 1=all repeated)."""
    tokens = text.split()
    if len(tokens) < n * 2:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    seen: set[tuple] = set()
    repeated = 0
    for g in ngrams:
        if g in seen:
            repeated += 1
        seen.add(g)
    return repeated / len(ngrams)


def _adherence_score(text: str, probe: dict[str, Any]) -> float:
    """Check adherence to probe's expect_shape / expect_palette."""
    score = 0.0
    checks = 0

    shape = probe.get("expect_shape")
    palette = probe.get("expect_palette")

    if palette and palette in _PALETTE_COLORS:
        checks += 1
        palette_tokens = _PALETTE_COLORS[palette]
        text_lower = text.lower()
        if any(tok in text_lower for tok in palette_tokens):
            score += 1.0

    if shape and shape in _SHAPE_SVG_TAGS:
        checks += 1
        text_lower = text.lower()
        if any(tag.lower() in text_lower for tag in _SHAPE_SVG_TAGS[shape]):
            score += 1.0

    if checks == 0:
        return 1.0  # OOD probe — no requirements
    return score / checks


def _score_output(text: str, probe: dict[str, Any]) -> dict[str, float]:
    return {
        "valid_svg": float(_is_valid_svg(text)),
        "closure": float(_has_closure(text)),
        "prefix_integrity": float(_prefix_integrity(text)),
        "repetition": _repetition_score(text),
        "adherence": _adherence_score(text, probe),
    }


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------
def _run_cmd(cmd: list[str], *, capture: bool = False, dry: bool = False, timeout: int = 600) -> str | None:
    if dry:
        print(f"  [DRY-RUN] {' '.join(str(c) for c in cmd)}")
        return None
    print(f"  [run] {' '.join(str(c) for c in cmd)}")
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"  [WARN] exit={result.returncode} stderr={result.stderr[:200]}", file=sys.stderr)
        return result.stdout
    else:
        subprocess.run(cmd, check=True, timeout=timeout)
        return None


# ---------------------------------------------------------------------------
# Per-run eval
# ---------------------------------------------------------------------------
def eval_run(
    run_dir: Path,
    ledger_rec: dict[str, Any],
    *,
    n_samples: int,
    max_tokens: int,
    temperature: float,
    dry: bool,
) -> dict[str, Any]:
    run_id = str(ledger_rec["run_id"])
    stage = _normalize_stage(ledger_rec.get("stage_id") or "pretrain")
    stage_pass = int(ledger_rec.get("stage_pass") or 0)
    phase_label = str(ledger_rec.get("phase_label") or f"{stage}_{stage_pass}")
    run_order = int(ledger_rec.get("run_order") or 0)
    final_loss = ledger_rec.get("loss_final")

    ck_build = run_dir / ".ck_build"

    print(f"\n{'=' * 60}")
    print(f"  run_id:     {run_id}")
    print(f"  stage:      {stage}  pass={stage_pass}  label={phase_label}")
    print(f"  final_loss: {final_loss}")
    print(f"{'=' * 60}")

    # ── Step 1: Promote checkpoint ──────────────────────────────────────────
    print(f"\n[Step 1] Promote checkpoint: {run_id}")
    _run_cmd(
        [sys.executable, str(PROMOTE_SCRIPT), "--run", str(run_dir), "--run-id", run_id],
        dry=dry,
        timeout=120,
    )

    # ── Step 2: Build inference runtime ────────────────────────────────────
    print(f"\n[Step 2] Build inference runtime (generate-only)")
    _run_cmd(
        [sys.executable, str(CK_RUN_SCRIPT), "run", str(run_dir), "--generate-only", "--context-len", "512"],
        dry=dry,
        timeout=600,
    )

    # ── Step 3: Run probes ──────────────────────────────────────────────────
    probe_results: list[dict[str, Any]] = []
    for probe in PROBES:
        probe_id = probe["id"]
        prompt = probe["prompt"]
        print(f"\n[Step 3] Probe: {probe_id} (n={n_samples})")
        samples: list[dict[str, Any]] = []
        for i in range(n_samples):
            output = _run_cmd(
                [
                    sys.executable, str(CK_CHAT_SCRIPT),
                    "--model-dir", str(ck_build),
                    "--python-tokenizer",
                    "--chat-template", "none",
                    "--prompt", prompt,
                    "--max-tokens", str(max_tokens),
                    "--temperature", str(temperature),
                    "--stop-at-eos",
                    "--no-stats",
                ],
                capture=True,
                dry=dry,
                timeout=120,
            )
            if dry:
                # Synthetic output for dry-run validation
                output = (
                    f"<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'>"
                    f"<circle cx='50' cy='50' r='40' fill='blue'/></svg>"
                )
            text = (output or "").strip()
            scores = _score_output(text, probe)
            samples.append({
                "idx": i,
                "text": text[:800],   # cap storage at 800 chars
                "scores": scores,
            })
            valid_tag = "OK" if scores["valid_svg"] else "FAIL"
            print(f"    sample {i}: valid={valid_tag} closure={scores['closure']:.0f} rep={scores['repetition']:.2f}")

        agg = {
            k: round(sum(s["scores"][k] for s in samples) / len(samples), 4) if samples else 0.0
            for k in ["valid_svg", "closure", "prefix_integrity", "repetition", "adherence"]
        }
        probe_results.append({
            "probe_id": probe_id,
            "prompt": prompt,
            "type": probe["type"],
            "samples": samples,
            "agg": agg,
        })
        print(f"    agg: valid={agg['valid_svg']:.2f} closure={agg['closure']:.2f} prefix={agg['prefix_integrity']:.2f} rep={agg['repetition']:.2f} adh={agg['adherence']:.2f}")

    # ── Step 4: Aggregate across probes ────────────────────────────────────
    svg_probes = [r for r in probe_results if r.get("type") == "svg_gen"]
    ood_probes = [r for r in probe_results if r.get("type") == "ood"]

    def _avg(probe_list: list[dict], key: str) -> float:
        vals = [r["agg"].get(key, 0.0) for r in probe_list]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    metrics: dict[str, Any] = {
        "valid_svg_rate": _avg(svg_probes, "valid_svg"),
        "closure_success_rate": _avg(svg_probes, "closure"),
        "eos_clean_stop": _avg(svg_probes + ood_probes, "closure"),
        "prefix_integrity": _avg(svg_probes + ood_probes, "prefix_integrity"),
        "repetition_loop_score": _avg(svg_probes + ood_probes, "repetition"),
        "ood_robustness": _avg(ood_probes, "valid_svg"),
        "adherence": _avg(svg_probes, "adherence"),
        "n_samples": n_samples,
        "n_probes": len(probe_results),
    }

    print(f"\n  → metrics: valid_svg={metrics['valid_svg_rate']} closure={metrics['closure_success_rate']}"
          f" prefix={metrics['prefix_integrity']} rep={metrics['repetition_loop_score']}"
          f" ood={metrics['ood_robustness']} adh={metrics['adherence']}")

    return {
        "run_id": run_id,
        "stage": stage,
        "stage_pass": stage_pass,
        "phase_label": phase_label,
        "run_order": run_order,
        "final_loss": final_loss,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "probe_results": probe_results,
    }


# ---------------------------------------------------------------------------
# Matrix file I/O
# ---------------------------------------------------------------------------
def _load_matrix(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "stage_eval_matrix.json"
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                return doc
        except Exception:
            pass
    return {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "probes": PROBES,
        "entries": [],
    }


def _save_matrix(run_dir: Path, matrix: dict[str, Any]) -> None:
    matrix["generated_at"] = datetime.now(timezone.utc).isoformat()
    path = run_dir / "stage_eval_matrix.json"
    path.write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  [saved] {path}")


def _upsert_entry(matrix: dict[str, Any], entry: dict[str, Any]) -> None:
    """Replace existing entry with same run_id, or append."""
    entries: list[dict[str, Any]] = matrix.setdefault("entries", [])
    for i, e in enumerate(entries):
        if e.get("run_id") == entry["run_id"]:
            entries[i] = entry
            return
    entries.append(entry)
    entries.sort(key=lambda e: int(e.get("run_order") or 0))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage eval matrix runner — evaluates model quality at each training run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all completed runs (generates full matrix)
  python3 eval_stage_v7.py --run ~/.cache/ck-engine-v7/models/train/MODEL --all-stages

  # Evaluate only the latest SFT pass
  python3 eval_stage_v7.py --run ... --stage sft --stage-pass 4

  # Evaluate a specific run
  python3 eval_stage_v7.py --run ... --run-id ascii_bpe_20260301_171735

  # Dry-run preview
  python3 eval_stage_v7.py --run ... --all-stages --dry-run
""",
    )
    ap.add_argument("--run", required=True, help="Run dir (contains run_ledger.jsonl)")
    ap.add_argument("--stage", default=None, help="Stage filter: pretrain / midtrain / sft / dpo / grpo")
    ap.add_argument("--stage-pass", type=int, default=None, help="Stage pass filter (requires --stage)")
    ap.add_argument("--run-id", default=None, help="Evaluate a specific run by run_id")
    ap.add_argument("--all-stages", action="store_true", help="Evaluate all completed runs in ledger")
    ap.add_argument("--n-samples", type=int, default=3, help="Samples per probe (default: 3)")
    ap.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate per sample (default: 256)")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0 = greedy)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would run, without executing")
    args = ap.parse_args()

    if args.stage_pass is not None and not args.stage:
        ap.error("--stage-pass requires --stage")

    run_dir = Path(args.run).expanduser().resolve()
    if not run_dir.exists():
        print(f"ERROR: run dir not found: {run_dir}", file=sys.stderr)
        return 1

    ledger_entries = _read_ledger(run_dir)
    if not ledger_entries:
        print(f"ERROR: no ledger entries in {run_dir}/run_ledger.jsonl", file=sys.stderr)
        print("  Hint: run train_data_pipeline_v7.py --backfill-ledger --run-dir PATH", file=sys.stderr)
        return 1

    # ── Select targets ──────────────────────────────────────────────────────
    if args.run_id:
        targets = [e for e in ledger_entries if e.get("run_id") == args.run_id]
        if not targets:
            print(f"ERROR: run_id '{args.run_id}' not in ledger.", file=sys.stderr)
            return 1

    elif args.stage:
        stage_norm = _normalize_stage(args.stage)
        targets = [e for e in ledger_entries if _normalize_stage(e.get("stage_id") or "") == stage_norm]
        if args.stage_pass is not None:
            targets = [e for e in targets if int(e.get("stage_pass") or 0) == args.stage_pass]
        if not targets:
            print(f"ERROR: no ledger entries for stage={args.stage}"
                  + (f" pass={args.stage_pass}" if args.stage_pass else ""), file=sys.stderr)
            return 1

    elif args.all_stages:
        targets = [e for e in ledger_entries if e.get("status") == "completed"]
        if not targets:
            targets = ledger_entries   # if status not set, include all

    else:
        ap.error("specify one of: --all-stages, --stage, --run-id")
        return 1

    print(f"[eval-matrix] run_dir:  {run_dir}")
    print(f"[eval-matrix] targets:  {len(targets)} run(s)")
    for t in targets:
        print(f"  [{t.get('run_order')}] {t.get('run_id')}  stage={t.get('stage_id')}  pass={t.get('stage_pass')}  loss={t.get('loss_final')}")

    matrix = _load_matrix(run_dir)
    matrix["schema"] = SCHEMA
    matrix["run_dir"] = str(run_dir)
    matrix.setdefault("model_name", run_dir.name)
    matrix["probes"] = PROBES

    for rec in targets:
        entry = eval_run(
            run_dir, rec,
            n_samples=args.n_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            dry=args.dry_run,
        )
        _upsert_entry(matrix, entry)
        if not args.dry_run:
            _save_matrix(run_dir, matrix)

    if args.dry_run:
        print("\n[DRY-RUN] would write stage_eval_matrix.json with:")
        for e in matrix.get("entries", []):
            m = e.get("metrics", {})
            print(f"  [{e.get('run_order')}] {e.get('phase_label'):15s}  "
                  f"valid={m.get('valid_svg_rate', '?'):.2f}  "
                  f"ood={m.get('ood_robustness', '?'):.2f}")
        return 0

    print(f"\n[done] stage_eval_matrix.json  —  {len(targets)} entry/entries updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
