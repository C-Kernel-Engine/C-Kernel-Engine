#!/usr/bin/env python3
"""
build_spec_training_story_v7.py — Collect training data from local cache and
reports, emit a structured JSON manifest, then render a static HTML story page
from that manifest.

Usage:
    python3 version/v7/scripts/build_spec_training_story_v7.py

Reads:
    ~/.cache/ck-engine-v7/models/train/spec*/run_ledger.jsonl
    ~/.cache/ck-engine-v7/models/train/spec*/*_probe_report*.json
    version/v7/runs/spec*/run_ledger.jsonl
    version/v7/runs/spec*/*_probe_report*.json
    version/v7/reports/TRAINING_LOGBOOK.md
    version/v7/reports/spec*.md / SPEC*.md

Writes:
    version/v7/reports/spec_training_manifest.json
    version/v7/reports/spec_training_story.html
"""

import json
import re
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path

# ── ANSI helpers ──────────────────────────────────────────────────────────────

_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text

def green(t: str)  -> str: return _c("32", t)
def yellow(t: str) -> str: return _c("33", t)
def red(t: str)    -> str: return _c("31", t)
def cyan(t: str)   -> str: return _c("36", t)
def bold(t: str)   -> str: return _c("1", t)

# ── Paths ─────────────────────────────────────────────────────────────────────

CACHE_ROOT = Path.home() / ".cache" / "ck-engine-v7" / "models" / "train"
REPO_ROOT  = Path(__file__).resolve().parents[3]       # …/C-Kernel-Engine
REPORTS    = REPO_ROOT / "version" / "v7" / "reports"
REPO_RUNS  = REPO_ROOT / "version" / "v7" / "runs"
OUT_FILE   = REPORTS / "spec_training_manifest.json"
HTML_FILE  = REPORTS / "spec_training_story.html"
SOURCE_ROOTS = [
    ("cache", CACHE_ROOT),
    ("repo", REPO_RUNS),
]

# ── Spec metadata (hand-curated context) ──────────────────────────────────────

SPEC_META = {
    "spec02": {
        "name":           "SVG Atoms",
        "goal":           "Legacy raw-SVG bootstrapping line",
        "representation": "raw SVG",
        "note":           "Legacy raw-SVG probe regime; not directly comparable to later structured probe lines.",
    },
    "spec03": {
        "name":           "Bootstrap Tokenizer Failure",
        "goal":           "Expose representation and tokenizer-contract failure modes",
        "representation": "raw SVG bootstrap",
        "note":           "Documented legacy failure line; kept as a design lesson rather than a cache-backed structured run family.",
    },
    "spec04": {
        "name":           "Structured Atoms",
        "goal":           "Structured atom tokenisation pipeline",
        "representation": "structured SVG atoms",
    },
    "spec05": {
        "name":           "Structured Scenes",
        "goal":           "Multi-element scene composition",
        "representation": "structured SVG scenes",
    },
    "spec06": {
        "name":           "Structured Infographics",
        "goal":           "Infographic-grade structured generation",
        "representation": "structured SVG infographics",
    },
    "spec07": {
        "name":           "Scene DSL v1",
        "goal":           "First scene DSL grammar training",
        "representation": "scene DSL text",
    },
    "spec08": {
        "name":           "Rich Scene DSL",
        "goal":           "Extended scene DSL with richer attributes",
        "representation": "scene DSL text",
    },
    "spec10": {
        "name":           "Asset Scene DSL",
        "goal":           "DSL with asset library integration",
        "representation": "scene DSL + asset keys",
    },
    "spec11": {
        "name":           "Keyed Scene DSL",
        "goal":           "DSL with keyed component vocabulary",
        "representation": "keyed scene DSL",
    },
    "spec12": {
        "name":           "Scene DSL (gold)",
        "goal":           "Gold-quality scene DSL with full layout families",
        "representation": "scene DSL v3 (ctx768)",
    },
    "spec13a": {
        "name":           "Intent-Prompt Bridge",
        "goal":           "Prompt-to-scene intent mapping",
        "representation": "scene DSL + intent prompts",
    },
    "spec13b": {
        "name":           "Decision-Tree Scene IR",
        "goal":           "Generalized scene planning via decision-tree IR and renderer bridge",
        "representation": "decision-tree IR + scene DSL bridge",
    },
    "spec14a": {
        "name":           "Comparison Board Family",
        "goal":           "Family-specific comparison-board scene DSL contract",
        "representation": "family scene DSL (comparison board)",
    },
    "spec14b": {
        "name":           "Timeline Family",
        "goal":           "Timeline-family scene DSL contract and decode validation",
        "representation": "family scene DSL (timeline)",
    },
    "spec15a": {
        "name":           "Memory Map Family",
        "goal":           "Memory-map family scene DSL contract and decode validation",
        "representation": "family scene DSL (memory map)",
    },
    "spec15b": {
        "name":           "System Diagram Family",
        "goal":           "System-diagram family scene DSL contract and decode validation",
        "representation": "family scene DSL (system diagram)",
    },
    "spec16": {
        "name":           "Generalized Visual Bundle",
        "goal":           "Shared [bundle] control language lowered through solved family compilers",
        "representation": "generalized visual bundle DSL",
    },
    "spec09": {
        "name":           "Scene DSL v2 Grammar",
        "goal":           "Design/compiler bridge toward structured asset-backed scenes",
        "representation": "design-only grammar and compiler planning",
        "note":           "Important design spec, but not a cache-backed rung family in this manifest.",
    },
}

LEGACY_SPECS = [
    {
        "id": "spec02",
        "name": "SVG Atoms",
        "status": "legacy",
        "note": "Legacy raw-SVG line with an older probe regime. Keep it in the story, but keep it out of unified structured charts.",
        "sources": [
            REPORTS / "spec02_spec06_evolution_executive_report_20260312.md",
            REPORTS / "spec02_svg_training_report_card.html",
        ],
    },
    {
        "id": "spec03",
        "name": "Bootstrap Tokenizer Failure",
        "status": "legacy",
        "note": "Representation failure case that taught the tokenizer/contract lesson before the structured scene DSL line stabilized.",
        "sources": [
            REPORTS / "SPEC03_REPRESENTATION_WORKSHEET.md",
            REPORTS / "SPEC03_TOKENIZER_PROMPT_ATOMS_ANALYSIS.md",
            REPORTS / "spec02_spec06_evolution_executive_report_20260312.md",
        ],
    },
    {
        "id": "spec09",
        "name": "Scene DSL v2 Grammar",
        "status": "design_only",
        "note": "Design/compiler bridge that informed later specs, but not a normal cache-backed training family.",
        "sources": [
            REPORTS / "SPEC09_BACKWARD_DESIGN_PLAN_2026-03-17.md",
            REPORTS / "SPEC09_SCENE_DSL_V2_GRAMMAR_2026-03-17.md",
        ],
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _warn(msg: str):
    print(f"  {yellow('WARN')} {msg}")

def _extract_spec_id(dirname: str) -> str:
    """Return the canonical spec id from a directory name like
    'spec12_scene_dsl_l3_d192_h384_ctx768_r14'."""
    m = re.match(r"(spec\d+[a-z]?)", dirname)
    return m.group(1) if m else dirname

def _extract_run_number(dirname: str) -> int | None:
    m = re.search(r"_r(\d+)(?:_|$)", dirname)
    return int(m.group(1)) if m else None

def _spec_sort_key(spec_id: str) -> tuple[int, str]:
    m = re.match(r"spec(\d+)([a-z]?)", spec_id)
    if not m:
        return (9999, spec_id)
    return (int(m.group(1)), m.group(2))

def _path_uri(path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        return Path(path).resolve().as_uri()
    except OSError:
        return ""

def _fmt_pct(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{decimals}f}%"

def _fmt_pct_compact(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value - 1.0) < 1e-9:
        return "100%"
    return f"{value * 100:.1f}%"

def _fmt_tokens(value: int | None) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(value)

def _fmt_num(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value:,}"

def _badge_class(status: str) -> str:
    mapping = {
        "gold": "badge-gold",
        "iterating": "badge-blue",
        "regressed": "badge-red",
        "in_progress": "badge-amber",
        "trained_no_probe": "badge-muted",
        "no_data": "badge-muted",
        "legacy": "badge-muted",
        "design_only": "badge-muted",
    }
    return mapping.get(status, "badge-muted")

def _source_links(items: list[tuple[str, str]]) -> str:
    links: list[str] = []
    for label, path in items:
        uri = _path_uri(path)
        if uri:
            links.append(f'<a href="{escape(uri)}">{escape(label)}</a>')
    if not links:
        return ""
    return f'<div class="source-links">Source: {" · ".join(links)}</div>'

def _read_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    try:
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    _warn(f"{path.name}:{lineno} bad JSON – skipped")
    except OSError as exc:
        _warn(f"cannot read {path}: {exc}")
    return entries

def _read_json(path: Path) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _warn(f"cannot read {path}: {exc}")
        return None

def _read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError as exc:
        _warn(f"cannot read {path}: {exc}")
        return None


def _source_priority(kind: str | None) -> int:
    return 1 if str(kind or "") == "repo" else 0


def _weighted_summary(parts: list[dict] | None) -> dict | None:
    rows = [part for part in (parts or []) if isinstance(part, dict)]
    if not rows:
        return None
    total_count = sum(int(part.get("count") or 0) for part in rows)
    if total_count <= 0:
        total_count = len(rows)

    def _weighted(field: str) -> float | None:
        values: list[tuple[float, int]] = []
        for part in rows:
            value = part.get(field)
            if value is None:
                continue
            weight = int(part.get("count") or 0) or 1
            values.append((float(value), weight))
        if not values:
            return None
        denom = sum(weight for _, weight in values) or len(values)
        return sum(value * weight for value, weight in values) / denom

    return {
        "count": total_count,
        "exact_rate": _weighted("exact_rate"),
        "renderable_rate": _weighted("renderable_rate"),
        "materialized_exact_rate": _weighted("materialized_exact_rate"),
        "svg_exact_rate": _weighted("svg_exact_rate"),
        "budget_truncation_rate": _weighted("budget_truncation_rate"),
        "split_summary": rows,
    }


def _summary_from_totals(totals: dict | None) -> dict | None:
    if not isinstance(totals, dict) or not totals:
        return None
    return {
        "count": totals.get("count"),
        "exact_rate": totals.get("exact_rate"),
        "renderable_rate": totals.get("renderable_rate"),
        "materialized_exact_rate": totals.get("materialized_exact_rate"),
        "svg_exact_rate": totals.get("svg_exact_rate"),
        "budget_truncation_rate": totals.get("budget_truncation_rate"),
        "split_summary": [],
    }


def _dedupe_records(records: list[dict], *, key_fields: list[str]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for record in records:
        key = tuple(record.get(field) for field in key_fields)
        prev = best.get(key)
        if prev is None:
            best[key] = record
            continue
        prev_rank = (_source_priority(prev.get("source_kind")), prev.get("_source_mtime") or 0.0)
        cur_rank = (_source_priority(record.get("source_kind")), record.get("_source_mtime") or 0.0)
        if cur_rank >= prev_rank:
            best[key] = record
    return list(best.values())

# ── Ledger collection ─────────────────────────────────────────────────────────

def collect_runs(roots: list[tuple[str, Path]]) -> list[dict]:
    """Walk spec* directories across cache and repo-local roots, parse
    run_ledger.jsonl files, and return normalised stage records."""
    runs: list[dict] = []
    for source_kind, root in roots:
        if not root.is_dir():
            _warn(f"run root not found: {root}")
            continue
        for spec_dir in sorted(root.iterdir()):
            if not spec_dir.is_dir() or not spec_dir.name.startswith("spec"):
                continue
            ledger = spec_dir / "run_ledger.jsonl"
            if not ledger.exists():
                continue
            entries = _read_jsonl(ledger)
            best: dict[tuple[str, str], dict] = {}
            for entry in entries:
                run_id = str(entry.get("run_id") or "")
                stage_id = str(entry.get("stage_id") or "")
                key = (run_id, stage_id)
                if run_id and stage_id and (key not in best or entry.get("status") == "completed"):
                    best[key] = entry
            for entry in best.values():
                runs.append(_normalise_run(entry, spec_dir.name, ledger, source_kind=source_kind))
    return _dedupe_records(
        runs,
        key_fields=["dir_name", "run_id", "stage_id", "started_at", "ended_at"],
    )


def _normalise_run(entry: dict, dirname: str, ledger_path: Path, *, source_kind: str) -> dict:
    spec_id = _extract_spec_id(dirname)
    run_num = _extract_run_number(dirname)
    stat = ledger_path.stat()
    return {
        "spec":       spec_id,
        "dir_name":   dirname,
        "run_dir_path": str(ledger_path.parent.resolve()),
        "ledger_path": str(ledger_path.resolve()),
        "source_kind": source_kind,
        "source_root": str(ledger_path.parent.parent.resolve()),
        "_source_mtime": stat.st_mtime,
        "run_number": run_num,
        "run_id":     entry.get("run_id"),
        "stage_id":   entry.get("stage_id"),
        "status":     entry.get("status"),
        "lr":         entry.get("lr"),
        "seq_len":    entry.get("seq_len"),
        "steps":      entry.get("steps"),
        "total_tokens": entry.get("total_tokens"),
        "loss_first": entry.get("loss_first"),
        "loss_final": entry.get("loss_final"),
        "loss_min":   entry.get("loss_min"),
        "loss_min_step": entry.get("loss_min_step"),
        "started_at": entry.get("started_at"),
        "ended_at":   entry.get("ended_at"),
    }

# ── Probe collection ─────────────────────────────────────────────────────────

def collect_probes(roots: list[tuple[str, Path]]) -> list[dict]:
    """Walk spec* directories across cache and repo-local roots and parse
    probe-report JSON files."""
    probes: list[dict] = []
    for source_kind, root in roots:
        if not root.is_dir():
            continue
        for spec_dir in sorted(root.iterdir()):
            if not spec_dir.is_dir() or not spec_dir.name.startswith("spec"):
                continue
            for probe_file in sorted(spec_dir.glob("*_probe_report*.json")):
                data = _read_json(probe_file)
                if data is None:
                    continue
                probes.append(_normalise_probe(data, spec_dir.name, probe_file, source_kind=source_kind))
    return _dedupe_records(probes, key_fields=["dir_name", "probe_file"])


def _normalise_probe(data: dict, dirname: str, probe_path: Path, *, source_kind: str) -> dict:
    spec_id = _extract_spec_id(dirname)
    totals = data.get("totals", {})
    filename = probe_path.name
    is_hidden = "hidden" in filename.lower()
    split_summary = [item for item in (data.get("split_summary") or []) if isinstance(item, dict)]
    hidden_split_summary = [
        item for item in split_summary
        if str(item.get("split") or "").lower().startswith("hidden")
    ]
    visible_split_summary = [
        item for item in split_summary
        if not str(item.get("split") or "").lower().startswith("hidden")
    ]
    visible_summary = _weighted_summary(visible_split_summary)
    hidden_summary = _weighted_summary(hidden_split_summary)
    totals_summary = _summary_from_totals(totals)
    if visible_summary is None and not is_hidden:
        visible_summary = totals_summary
    if hidden_summary is None and is_hidden:
        hidden_summary = totals_summary
    stat = probe_path.stat()
    return {
        "spec":                     spec_id,
        "dir_name":                 dirname,
        "probe_path":               str(probe_path.resolve()),
        "probe_file":               filename,
        "hidden":                   is_hidden,
        "source_kind":              source_kind,
        "source_root":              str(probe_path.parent.parent.resolve()),
        "_source_mtime":            stat.st_mtime,
        "generated_at":             data.get("generated_at"),
        "count":                    visible_summary.get("count") if visible_summary else None,
        "exact_rate":               visible_summary.get("exact_rate") if visible_summary else None,
        "renderable_rate":          visible_summary.get("renderable_rate") if visible_summary else None,
        "materialized_exact_rate":  visible_summary.get("materialized_exact_rate") if visible_summary else None,
        "svg_exact_rate":           visible_summary.get("svg_exact_rate") if visible_summary else None,
        "budget_truncation_rate":   visible_summary.get("budget_truncation_rate") if visible_summary else None,
        "overall_count":            totals.get("count"),
        "overall_exact_rate":       totals.get("exact_rate"),
        "overall_renderable_rate":  totals.get("renderable_rate"),
        "overall_materialized_exact_rate": totals.get("materialized_exact_rate"),
        "overall_svg_exact_rate":   totals.get("svg_exact_rate"),
        "overall_budget_truncation_rate": totals.get("budget_truncation_rate"),
        "split_summary":            split_summary,
        "visible_split_summary":    visible_split_summary,
        "hidden_split_summary":     hidden_split_summary,
        "hidden_summary":           hidden_summary,
    }

# ── Logbook parsing ──────────────────────────────────────────────────────────

_LOGBOOK_HEADING = re.compile(
    r"^##\s+(\d{4}-\d{2}-\d{2})\s*[-–—]\s*(.+)$"
)

def parse_logbook(path: Path) -> list[dict]:
    """Parse TRAINING_LOGBOOK.md into timeline/incident entries."""
    text = _read_text(path)
    if text is None:
        return []
    entries: list[dict] = []
    for line in text.splitlines():
        m = _LOGBOOK_HEADING.match(line.strip())
        if m:
            date_str, title = m.group(1), m.group(2).strip()
            spec_id = None
            sm = re.search(r"(spec\d+[a-z]?)", title, re.IGNORECASE)
            if sm:
                spec_id = sm.group(1).lower()
            entries.append({
                "date":    date_str,
                "title":   title,
                "spec":    spec_id,
            })
    return entries

# ── Reports inventory ─────────────────────────────────────────────────────────

_SPEC_REPORT_PAT = re.compile(r"(spec\d+[a-z]?)", re.IGNORECASE)

def collect_report_files(reports_dir: Path) -> list[dict]:
    """Inventory spec-related report files."""
    result: list[dict] = []
    if not reports_dir.is_dir():
        _warn(f"reports dir not found: {reports_dir}")
        return result
    for fp in sorted(reports_dir.iterdir()):
        if fp.is_dir():
            continue
        m = _SPEC_REPORT_PAT.search(fp.name)
        if m:
            result.append({
                "file": fp.name,
                "spec": m.group(1).lower(),
            })
    return result

# ── Per-spec summary builder ──────────────────────────────────────────────────

def _status_label(run: dict) -> str:
    s = run.get("status") or "unknown"
    return s

def build_per_spec(runs: list[dict], probes: list[dict]) -> list[dict]:
    """Aggregate runs and probes into per-spec summary entries."""
    spec_ids: set[str] = set()
    for r in runs:
        spec_ids.add(r["spec"])
    for p in probes:
        spec_ids.add(p["spec"])

    out: list[dict] = []
    for sid in sorted(spec_ids, key=_spec_sort_key):
        meta = SPEC_META.get(sid, {})
        s_runs   = [r for r in runs   if r["spec"] == sid]
        s_probs  = [p for p in probes  if p["spec"] == sid]
        best_probe = _pick_best_probe(s_probs)
        best_hidden = _pick_best_hidden_probe(s_probs)
        best_capability_run = _pick_capability_run(s_runs, best_probe["dir_name"] if best_probe else None)
        lowest_loss_run = _pick_lowest_loss_run(s_runs)
        run_dirs = sorted({r["dir_name"] for r in s_runs})
        out.append({
            "id":             sid,
            "name":           meta.get("name", sid),
            "goal":           meta.get("goal", ""),
            "representation": meta.get("representation", ""),
            "note":           meta.get("note", ""),
            "total_runs":     len(run_dirs),
            "total_stage_records": len(s_runs),
            "total_probes":   len(s_probs),
            "best_result":    best_probe,
            "best_hidden_result": best_hidden,
            "best_capability_run": _summarise_run(best_capability_run) if best_capability_run else None,
            "lowest_loss_run": _summarise_run(lowest_loss_run) if lowest_loss_run else None,
            "lesson":         _derive_lesson(sid, s_runs, s_probs),
            "status":         _derive_status(s_runs, s_probs),
        })
    return out

def _pick_best_probe(probes: list[dict]) -> dict | None:
    """Select the probe with highest visible exact_rate when available."""
    visible = [probe for probe in probes if probe.get("exact_rate") is not None]
    if not visible:
        return None
    def _key(probe: dict):
        return (
            probe.get("exact_rate") or 0.0,
            probe.get("materialized_exact_rate") or 0.0,
            probe.get("overall_exact_rate") or 0.0,
        )
    best = max(visible, key=_key)
    return {
        "dir_name":                best["dir_name"],
        "probe_path":              best["probe_path"],
        "exact_rate":              best.get("exact_rate"),
        "renderable_rate":         best.get("renderable_rate"),
        "materialized_exact_rate": best.get("materialized_exact_rate"),
        "count":                   best.get("count"),
        "overall_exact_rate":      best.get("overall_exact_rate"),
        "overall_renderable_rate": best.get("overall_renderable_rate"),
        "overall_materialized_exact_rate": best.get("overall_materialized_exact_rate"),
        "visible_split_summary":   best.get("visible_split_summary"),
        "hidden_split_summary":    best.get("hidden_split_summary"),
        "hidden_summary":          best.get("hidden_summary"),
        "split_summary":           best.get("split_summary"),
    }


def _pick_best_hidden_probe(probes: list[dict]) -> dict | None:
    hidden = [probe for probe in probes if isinstance(probe.get("hidden_summary"), dict)]
    if not hidden:
        return None
    def _key(probe: dict):
        summary = probe.get("hidden_summary") or {}
        return (
            summary.get("exact_rate") or 0.0,
            summary.get("materialized_exact_rate") or 0.0,
            summary.get("renderable_rate") or 0.0,
        )
    best = max(hidden, key=_key)
    summary = best.get("hidden_summary") or {}
    return {
        "dir_name":                best["dir_name"],
        "probe_path":              best["probe_path"],
        "exact_rate":              summary.get("exact_rate"),
        "renderable_rate":         summary.get("renderable_rate"),
        "materialized_exact_rate": summary.get("materialized_exact_rate"),
        "count":                   summary.get("count"),
        "split_summary":           summary.get("split_summary"),
    }

def _pick_capability_run(runs: list[dict], dir_name: str | None) -> dict | None:
    if not dir_name:
        return None
    completed = [r for r in runs if r.get("status") == "completed" and r["dir_name"] == dir_name]
    if not completed:
        return None
    return max(completed, key=lambda r: (r.get("ended_at") or "", r.get("steps") or 0))

def _pick_lowest_loss_run(runs: list[dict]) -> dict | None:
    completed = [r for r in runs if r.get("status") == "completed"]
    if not completed:
        return runs[-1] if runs else None
    def _key(r: dict):
        return r.get("loss_min") if r.get("loss_min") is not None else 1e9
    return min(completed, key=_key)

def _summarise_run(r: dict) -> dict:
    return {
        "dir_name":     r["dir_name"],
        "run_dir_path": r["run_dir_path"],
        "ledger_path":  r["ledger_path"],
        "run_id":       r["run_id"],
        "stage_id":     r["stage_id"],
        "steps":        r["steps"],
        "total_tokens": r["total_tokens"],
        "loss_first":   r["loss_first"],
        "loss_final":   r["loss_final"],
        "loss_min":     r["loss_min"],
        "started_at":   r["started_at"],
        "ended_at":     r["ended_at"],
    }

def _derive_lesson(sid: str, runs: list[dict], probes: list[dict]) -> str:
    if not probes:
        return "no probe data"
    visible = [probe for probe in probes if probe.get("exact_rate") is not None]
    if not visible:
        return "no visible probe data"
    best = max(visible, key=lambda probe: probe.get("exact_rate") or 0.0)
    er = best.get("exact_rate") or 0.0
    if er >= 1.0:
        return "converged – perfect exact match"
    elif er >= 0.8:
        return "strong – minor gaps remain"
    elif er >= 0.5:
        return "partial – further iteration needed"
    elif er > 0.0:
        return "weak – fundamental issues likely"
    else:
        return "failed – no correct outputs"

def _derive_status(runs: list[dict], probes: list[dict]) -> str:
    if not runs:
        return "no_data"
    completed = [r for r in runs if r.get("status") == "completed"]
    if not completed:
        return "in_progress"
    if not probes:
        return "trained_no_probe"
    visible = [probe for probe in probes if probe.get("exact_rate") is not None]
    if not visible:
        return "trained_no_probe"
    best_er = max((probe.get("exact_rate") or 0.0) for probe in visible)
    if best_er >= 1.0:
        return "gold"
    elif best_er >= 0.5:
        return "iterating"
    else:
        return "regressed"

# ── Per-run entries ───────────────────────────────────────────────────────────

def build_per_run(runs: list[dict], probes: list[dict]) -> list[dict]:
    """Merge run data with its matching probe report."""
    # Index probes by dir_name. Prefer repo-local copies when both roots contain the same run.
    probe_by_dir: dict[str, dict] = {}
    for p in probes:
        key = p["dir_name"]
        prev = probe_by_dir.get(key)
        if prev is None:
            probe_by_dir[key] = p
            continue
        prev_rank = (_source_priority(prev.get("source_kind")), prev.get("_source_mtime") or 0.0)
        cur_rank = (_source_priority(p.get("source_kind")), p.get("_source_mtime") or 0.0)
        if cur_rank >= prev_rank:
            probe_by_dir[key] = p

    out: list[dict] = []
    for r in runs:
        pr = probe_by_dir.get(r["dir_name"])
        hidden_summary = pr.get("hidden_summary") if pr else None
        out.append({
            "spec":                     r["spec"],
            "dir_name":                 r["dir_name"],
            "run_dir_path":             r["run_dir_path"],
            "ledger_path":              r["ledger_path"],
            "run_number":               r["run_number"],
            "run_id":                   r["run_id"],
            "stage_id":                 r["stage_id"],
            "steps":                    r["steps"],
            "total_tokens":             r["total_tokens"],
            "loss_first":               r["loss_first"],
            "loss_final":               r["loss_final"],
            "loss_min":                 r["loss_min"],
            "started_at":               r["started_at"],
            "ended_at":                 r["ended_at"],
            "exact_rate":               pr["exact_rate"]              if pr else None,
            "renderable_rate":          pr["renderable_rate"]         if pr else None,
            "materialized_exact_rate":  pr["materialized_exact_rate"] if pr else None,
            "overall_exact_rate":       pr["overall_exact_rate"]      if pr else None,
            "overall_renderable_rate":  pr["overall_renderable_rate"] if pr else None,
            "hidden_exact_rate":        hidden_summary.get("exact_rate") if isinstance(hidden_summary, dict) else None,
            "hidden_renderable_rate":   hidden_summary.get("renderable_rate") if isinstance(hidden_summary, dict) else None,
            "probe_path":               pr["probe_path"]              if pr else None,
            "status_label":             _status_label(r),
        })
    return out

# ── Manifest assembly ────────────────────────────────────────────────────────

def build_manifest(
    runs: list[dict],
    probes: list[dict],
    logbook_entries: list[dict],
    report_files: list[dict],
) -> dict:
    return {
        "schema":       "ck.spec_training_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator":    "build_spec_training_story_v7.py",
        "source_roots": [
            {"kind": kind, "path": str(path.resolve())}
            for kind, path in SOURCE_ROOTS
            if path.exists()
        ],
        "specs":        build_per_spec(runs, probes),
        "runs":         build_per_run(runs, probes),
        "probe_count":  len(probes),
        "stage_record_count": len(runs),
        "run_dir_count": len({r["dir_name"] for r in runs}),
        "timeline":     logbook_entries,
        "incidents":    [e for e in logbook_entries
                         if any(kw in e["title"].lower()
                                for kw in ("collapse", "regress", "fail",
                                           "broke", "corrupt", "drift"))],
        "legacy_specs": [
            {
                "id": item["id"],
                "name": item["name"],
                "status": item["status"],
                "note": item["note"],
                "sources": [str(src.resolve()) for src in item["sources"] if src.exists()],
            }
            for item in LEGACY_SPECS
        ],
        "report_files": report_files,
    }

# ── HTML rendering ────────────────────────────────────────────────────────────

def _group_runs_for_spec(manifest: dict, spec_id: str) -> list[dict]:
    groups: dict[str, dict] = {}
    for run in manifest["runs"]:
        if run["spec"] != spec_id:
            continue
        group = groups.setdefault(run["dir_name"], {
            "dir_name": run["dir_name"],
            "run_dir_path": run["run_dir_path"],
            "ledger_path": run["ledger_path"],
            "run_number": run["run_number"],
            "exact_rate": run["exact_rate"],
            "renderable_rate": run["renderable_rate"],
            "materialized_exact_rate": run["materialized_exact_rate"],
            "probe_path": run["probe_path"],
            "latest_stage": run["stage_id"],
            "latest_steps": run["steps"],
            "latest_tokens": run["total_tokens"],
            "loss_first": run["loss_first"],
            "loss_final": run["loss_final"],
            "loss_min": run["loss_min"],
            "ended_at": run["ended_at"] or "",
        })
        if run["exact_rate"] is not None:
            group["exact_rate"] = run["exact_rate"]
            group["renderable_rate"] = run["renderable_rate"]
            group["materialized_exact_rate"] = run["materialized_exact_rate"]
            group["probe_path"] = run["probe_path"]
        if (run["ended_at"] or "", run["steps"] or 0) >= (group["ended_at"], group["latest_steps"] or 0):
            group["latest_stage"] = run["stage_id"]
            group["latest_steps"] = run["steps"]
            group["latest_tokens"] = run["total_tokens"]
            group["loss_first"] = run["loss_first"]
            group["loss_final"] = run["loss_final"]
            group["loss_min"] = run["loss_min"]
            group["ended_at"] = run["ended_at"] or ""
    return sorted(groups.values(), key=lambda g: (g["run_number"] if g["run_number"] is not None else 9999, g["dir_name"]))

def _render_best_exact_chart(specs: list[dict]) -> str:
    chart_specs = [s for s in specs if s.get("best_result")]
    if not chart_specs:
        return ""
    width = max(720, 90 * len(chart_specs) + 120)
    height = 320
    left = 70
    bottom = 270
    top = 30
    usable_h = bottom - top
    bar_w = 42
    gap = 28
    parts = [
        f'<div class="chart-container"><svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="{width/2:.1f}" y="16" text-anchor="middle" class="chart-title">Best Exact Match by Probe-Backed Spec</text>',
    ]
    for pct in range(0, 101, 20):
        y = bottom - usable_h * (pct / 100.0)
        klass = "grid-line" if pct < 100 else "grid-line"
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-30}" y2="{y:.1f}" class="{klass}"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="axis-label">{pct}%</text>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{width-30}" y2="{bottom}" stroke="#30363d" stroke-width="1"/>')
    for idx, spec in enumerate(chart_specs):
        x = left + 20 + idx * (bar_w + gap)
        er = spec["best_result"]["exact_rate"] or 0.0
        bar_h = usable_h * er
        y = bottom - bar_h
        color = "#e3b341" if spec["id"] == "spec12" and abs(er - 1.0) < 1e-9 else "#58a6ff"
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="4" fill="{color}" opacity="0.88"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" class="axis-label" fill="{color}">{_fmt_pct_compact(er)}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{bottom+18}" text-anchor="middle" class="axis-label">{escape(spec["id"])}</text>')
    parts.append(f'<text x="18" y="{(top+bottom)/2:.1f}" text-anchor="middle" transform="rotate(-90,18,{(top+bottom)/2:.1f})" class="axis-title">Exact Match</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-10}" text-anchor="middle" class="axis-title">Spec</text>')
    parts.append('</svg></div>')
    return "".join(parts)

def render_html(manifest: dict) -> str:
    specs = sorted(manifest["specs"], key=lambda s: _spec_sort_key(s["id"]))
    cache_specs = [s for s in specs if s.get("best_result")]
    legacy_specs = manifest.get("legacy_specs", [])
    spec12 = next((s for s in specs if s["id"] == "spec12"), None)
    champion = next((s for s in specs if s["id"] == "spec12"), cache_specs[-1] if cache_specs else None)
    champion_visible = champion.get("best_result") if champion else None
    champion_hidden = champion.get("best_hidden_result") if champion else None
    spec13a = next((s for s in specs if s["id"] == "spec13a"), None)
    spec12_runs = _group_runs_for_spec(manifest, "spec12")
    r20_run = next((run for run in spec12_runs if run.get("run_number") == 20), None)

    spec_cards: list[str] = []
    for spec in specs:
        best = spec.get("best_result")
        best_hidden = spec.get("best_hidden_result")
        best_cap = spec.get("best_capability_run")
        low_loss = spec.get("lowest_loss_run")
        split_line = ""
        if best and best.get("visible_split_summary"):
            bits = [
                f'{escape(item["split"])} {_fmt_pct(item.get("exact_rate"))}'
                for item in best["visible_split_summary"]
            ]
            split_line = f'<p><strong>Visible split exact:</strong> {", ".join(bits)}</p>'
        if spec["id"] == "spec13a" and best and best.get("visible_split_summary"):
            split_line = (
                '<p><strong>Visible and hidden split exact:</strong> '
                + ", ".join(
                    f'{escape(item["split"])} {_fmt_pct(item.get("exact_rate"))}'
                    for item in best["visible_split_summary"]
                )
                + '</p>'
            )
        card = [
            '<details>',
            f'<summary>{escape(spec["id"])} — {escape(spec["name"])} '
            f'<span class="badge {_badge_class(spec["status"])}">{escape(spec["status"])}</span></summary>',
            '<div class="detail-body">',
            f'<p><strong>Goal:</strong> {escape(spec["goal"] or "—")}</p>',
            f'<p><strong>Representation:</strong> {escape(spec["representation"] or "—")}</p>',
        ]
        if spec.get("note"):
            card.append(f'<p><strong>Note:</strong> {escape(spec["note"])}</p>')
        card.append(
            '<div class="card-grid">'
            f'<div class="card"><div class="label">Run Dirs</div><div class="value">{spec["total_runs"]}</div></div>'
            f'<div class="card"><div class="label">Stage Records</div><div class="value">{spec["total_stage_records"]}</div></div>'
            f'<div class="card"><div class="label">Best Exact</div><div class="value">{_fmt_pct(best.get("exact_rate") if best else None)}</div></div>'
            f'<div class="card"><div class="label">Best Renderable</div><div class="value">{_fmt_pct(best.get("renderable_rate") if best else None)}</div></div>'
            f'<div class="card"><div class="label">Best Materialized</div><div class="value">{_fmt_pct(best.get("materialized_exact_rate") if best else None)}</div></div>'
            '</div>'
        )
        if best:
            card.append(
                f'<p><strong>Best capability run:</strong> {escape(best["dir_name"])}'
                + (f' ({escape(best_cap["stage_id"])}, {_fmt_num(best_cap["steps"])} steps, {_fmt_tokens(best_cap["total_tokens"])})' if best_cap else "")
                + '</p>'
            )
            card.append(_source_links([
                ("visible probe", best.get("probe_path", "")),
                ("run ledger", best_cap.get("ledger_path", "") if best_cap else ""),
            ]))
        if best_hidden:
            card.append(
                f'<p><strong>Best hidden probe:</strong> {_fmt_pct(best_hidden.get("exact_rate"))} exact, '
                f'{_fmt_pct(best_hidden.get("renderable_rate"))} renderable, '
                f'{_fmt_pct(best_hidden.get("materialized_exact_rate"))} materialized'
                f' ({best_hidden.get("count", "—")} rows)</p>'
            )
            card.append(_source_links([("hidden probe", best_hidden.get("probe_path", ""))]))
        if split_line:
            card.append(split_line)
        if low_loss:
            card.append(
                f'<p><strong>Lowest-loss run:</strong> {escape(low_loss["dir_name"])} '
                f'({escape(low_loss["stage_id"])}, min loss {low_loss["loss_min"]:.6f})</p>'
            )
            card.append(_source_links([("run ledger", low_loss.get("ledger_path", ""))]))
        card.append(f'<p><strong>Lesson:</strong> {escape(spec["lesson"])}</p>')
        card.append('</div></details>')
        spec_cards.append("".join(card))

    legacy_cards = []
    for item in legacy_specs:
        legacy_cards.append(
            "<details>"
            f"<summary>{escape(item['id'])} — {escape(item['name'])} "
            f"<span class=\"badge {_badge_class(item['status'])}\">{escape(item['status'])}</span></summary>"
            "<div class=\"detail-body\">"
            f"<p>{escape(item['note'])}</p>"
            f"{_source_links([('report', src) for src in item.get('sources', [])])}"
            "</div></details>"
        )

    spec12_rows = []
    for run in spec12_runs:
        label = f"r{run['run_number']}" if run["run_number"] is not None else run["dir_name"]
        spec12_rows.append(
            "<tr>"
            f"<td><strong>{escape(label)}</strong></td>"
            f"<td>{escape(run['latest_stage'] or '—')}</td>"
            f"<td>{_fmt_num(run['latest_steps'])}</td>"
            f"<td>{_fmt_tokens(run['latest_tokens'])}</td>"
            f"<td>{_fmt_pct(run.get('exact_rate'))}</td>"
            f"<td>{_fmt_pct(run.get('renderable_rate'))}</td>"
            f"<td>{_fmt_pct(run.get('materialized_exact_rate'))}</td>"
            f"<td>{_source_links([('probe', run.get('probe_path', '')), ('ledger', run.get('ledger_path', ''))])}</td>"
            "</tr>"
        )

    champion_sources = _source_links([
        ("visible probe", champion_visible.get("probe_path", "") if champion_visible else ""),
        ("hidden probe", champion_hidden.get("probe_path", "") if champion_hidden else ""),
        ("run ledger", champion.get("best_capability_run", {}).get("ledger_path", "") if champion else ""),
    ])
    spec13a_split_line = ""
    if spec13a and spec13a.get("best_result", {}).get("split_summary"):
        spec13a_split_line = ", ".join(
            f'{escape(item["split"])} {_fmt_pct(item.get("exact_rate"))}'
            for item in spec13a["best_result"]["split_summary"]
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>C-Kernel-Engine — Spec Training Story</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0f141b;--card:#171d26;--border:#2d3642;--text:#d6dde6;--heading:#f5f7fb;--muted:#94a3b8;--accent:#4bb2ff;--success:#4cc77a;--error:#ff6b6b;--warning:#e2a93b;--gold:#e3b341;--mono:'JetBrains Mono','SFMono-Regular',Consolas,monospace;--sans:'IBM Plex Sans','Segoe UI',sans-serif}}
body{{background:radial-gradient(circle at top,#162232 0,#0f141b 45%);color:var(--text);font-family:var(--sans);line-height:1.6;margin:0}}
.page{{max-width:1120px;margin:0 auto;padding:2rem 1.25rem 4rem}}
h1{{font-size:2.2rem;color:var(--heading);margin-bottom:.5rem}}
h2{{font-size:1.35rem;color:var(--heading);margin:2.2rem 0 1rem;padding-bottom:.45rem;border-bottom:1px solid var(--border)}}
h3{{font-size:1.05rem;color:var(--heading);margin:1.4rem 0 .75rem}}
p{{margin:.65rem 0}}
a{{color:var(--accent);text-decoration:none}}
a:hover{{text-decoration:underline}}
.subtitle{{color:var(--muted);max-width:80ch}}
.card{{background:linear-gradient(180deg,rgba(255,255,255,.02),rgba(255,255,255,.01));border:1px solid var(--border);border-radius:14px;padding:1rem 1.1rem;margin:1rem 0;box-shadow:0 10px 30px rgba(0,0,0,.18)}}
.card-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1rem;margin:1rem 0}}
.label{{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:.2rem}}
.value{{font-size:1.15rem;font-weight:700;color:var(--heading)}}
.badge{{display:inline-block;padding:.18rem .6rem;border-radius:999px;font-size:.75rem;font-weight:700;vertical-align:middle}}
.badge-gold{{background:rgba(227,179,65,.16);color:var(--gold);border:1px solid rgba(227,179,65,.35)}}
.badge-blue{{background:rgba(75,178,255,.14);color:var(--accent);border:1px solid rgba(75,178,255,.3)}}
.badge-red{{background:rgba(255,107,107,.14);color:var(--error);border:1px solid rgba(255,107,107,.3)}}
.badge-amber{{background:rgba(226,169,59,.14);color:var(--warning);border:1px solid rgba(226,169,59,.3)}}
.badge-muted{{background:rgba(148,163,184,.12);color:var(--muted);border:1px solid rgba(148,163,184,.24)}}
details{{background:var(--card);border:1px solid var(--border);border-radius:12px;margin:.9rem 0;overflow:hidden}}
summary{{cursor:pointer;padding:1rem 1.1rem;font-weight:700;color:var(--heading)}}
.detail-body{{padding:0 1.1rem 1rem}}
.source-links{{margin-top:.45rem;color:var(--muted);font-size:.84rem}}
.highlight{{border-left:3px solid var(--accent);padding:.8rem 1rem;background:rgba(75,178,255,.06);border-radius:0 10px 10px 0}}
.table-wrap{{overflow-x:auto;border:1px solid var(--border);border-radius:12px;margin:1rem 0}}
table{{width:100%;border-collapse:collapse;font-size:.9rem}}
th,td{{padding:.68rem .8rem;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}}
th{{background:#1b2330;color:var(--heading)}}
tr:nth-child(even) td{{background:rgba(255,255,255,.01)}}
code{{font-family:var(--mono);background:#131922;padding:.14rem .34rem;border-radius:5px}}
.muted{{color:var(--muted)}}
ul{{margin:.6rem 0 1rem 1.2rem}}
li{{margin:.3rem 0}}
@media (max-width: 720px) {{ .page{{padding:1.1rem .8rem 2.5rem}} h1{{font-size:1.8rem}} }}
</style>
</head>
<body>
<main class="page">
<header>
  <h1>Spec Training Story</h1>
  <p class="subtitle">A manifest-driven report built from cache-backed and repo-local run ledgers plus probe files. This page separates probe-backed structured metrics from legacy or design-only specs so the numbers stay tied to the source artifacts.</p>
</header>

<section>
  <h2>Executive Summary</h2>
  <div class="card-grid">
    <div class="card"><div class="label">Probe-Backed Specs</div><div class="value">{len(cache_specs)}</div></div>
    <div class="card"><div class="label">Legacy / Design Specs</div><div class="value">{len(legacy_specs)}</div></div>
    <div class="card"><div class="label">Run Dirs</div><div class="value">{manifest['run_dir_count']}</div></div>
    <div class="card"><div class="label">Stage Records</div><div class="value">{manifest['stage_record_count']}</div></div>
    <div class="card"><div class="label">Probe Reports</div><div class="value">{manifest['probe_count']}</div></div>
  </div>
  <div class="card">
    <p><strong>Strict champion:</strong> {escape(champion['id'])} {escape(champion_visible['dir_name']) if champion_visible else '—'} at {_fmt_pct(champion_visible.get('exact_rate') if champion_visible else None)} visible exact, {_fmt_pct(champion_hidden.get('exact_rate') if champion_hidden else None)} hidden exact.</p>
    <p><strong>Key lesson:</strong> the project's strongest repeated finding still holds: compiler-backed probe metrics are the decision surface, not loss.</p>
    {champion_sources}
  </div>
</section>

<section>
  <h2>Scope Note</h2>
  <div class="highlight">
    The bar chart and per-spec metrics below cover only the probe-backed structured training lines found in the combined cache-backed and repo-local run roots. Legacy raw-SVG lines and design-only specs are still listed, but not mixed into the unified metric ladder.
  </div>
  {_render_best_exact_chart(cache_specs)}
</section>

<section>
  <h2>Legacy and Design-Only Specs</h2>
  {''.join(legacy_cards)}
</section>

<section>
  <h2>Probe-Backed Spec Ladder</h2>
  {''.join(spec_cards)}
</section>

<section>
  <h2>Spec12 Deep Dive</h2>
  <div class="card">
    <p><strong>Why this still matters:</strong> `spec12` is the cleanest recovery story in the repo. It contains the collapse (`r7/r8`), the recovery (`r9/r12/r15`), the champion (`r17`), and the frozen-vocab breadth check (`r20`).</p>
    {_source_links([('champion visible probe', champion_visible.get('probe_path', '') if champion_visible else ''), ('champion hidden probe', champion_hidden.get('probe_path', '') if champion_hidden else '')])}
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr><th>Run</th><th>Latest Stage</th><th>Steps</th><th>Tokens</th><th>Exact</th><th>Renderable</th><th>Materialized</th><th>Sources</th></tr>
      </thead>
      <tbody>
        {''.join(spec12_rows)}
      </tbody>
    </table>
  </div>
</section>

<section>
  <h2>Current State</h2>
  <div class="card-grid">
    <div class="card">
      <div class="label">Champion</div>
      <div class="value">{escape(champion['id'])} {escape(champion_visible['dir_name'].split('_')[-1] if champion_visible else '—')}</div>
      <p class="muted">{_fmt_pct(champion_visible.get('exact_rate') if champion_visible else None)} visible exact, {_fmt_pct(champion_hidden.get('exact_rate') if champion_hidden else None)} hidden exact</p>
      {champion_sources}
    </div>
    <div class="card">
      <div class="label">Frozen-Vocab Breadth Proof</div>
      <div class="value">spec12 r20</div>
      <p class="muted">Strict champion remains spec12 r17; r20 visible exact is {_fmt_pct(r20_run.get('exact_rate') if r20_run else None)} on the canonical visible probe.</p>
      {_source_links([('r20 probe', r20_run.get('probe_path', '') if r20_run else ''), ('r20 ledger', r20_run.get('ledger_path', '') if r20_run else '')])}
    </div>
    <div class="card">
      <div class="label">Spec13a Best Overall</div>
      <div class="value">spec13a r2</div>
      <p class="muted">Overall exact {_fmt_pct(spec13a.get('best_result', {}).get('exact_rate') if spec13a else None)}; split-specific exact: {escape(spec13a_split_line or '—')}</p>
      {_source_links([('spec13a r2 probe', spec13a.get('best_result', {}).get('probe_path', '') if spec13a else '')])}
    </div>
  </div>
  <div class="card">
    <p><strong>Best next move:</strong> do not rung-chase `spec13a` blindly. The current failure is a family-choice attractor, not training instability. `spec13b` should begin as a generalized decision-tree IR plus deterministic layered renderer with backward-compatible adapters.</p>
    {_source_links([
      ('spec13b brief', REPORTS / 'SPEC13B_GENERALIZED_SCENE_IR_BRIEF_2026-03-24.md'),
      ('prediction framework', REPORTS / 'SPEC13_RUN_PREDICTION_FRAMEWORK_2026-03-24.md'),
    ])}
  </div>
</section>

<section>
  <h2>Primary Sources</h2>
  <div class="card">
    <ul>
      <li><a href="{escape(_path_uri(REPORTS / 'TRAINING_LOGBOOK.md'))}">TRAINING_LOGBOOK.md</a></li>
      <li><a href="{escape(_path_uri(REPORTS / 'SPEC12_R1_TO_R8_PROGRESS_REPORT_2026-03-20.md'))}">SPEC12_R1_TO_R8_PROGRESS_REPORT_2026-03-20.md</a></li>
      <li><a href="{escape(_path_uri(REPORTS / 'SPEC13A_INTENT_PROMPT_BRIDGE_2026-03-18.md'))}">SPEC13A_INTENT_PROMPT_BRIDGE_2026-03-18.md</a></li>
      <li><a href="{escape(_path_uri(REPORTS / 'SPEC13_ROADMAP_2026-03-23.md'))}">SPEC13_ROADMAP_2026-03-23.md</a></li>
      <li><a href="{escape(_path_uri(REPORTS / 'SPEC13B_GENERALIZED_SCENE_IR_BRIEF_2026-03-24.md'))}">SPEC13B_GENERALIZED_SCENE_IR_BRIEF_2026-03-24.md</a></li>
      <li><a href="{escape(_path_uri(REPORTS / 'SPEC13_RUN_PREDICTION_FRAMEWORK_2026-03-24.md'))}">SPEC13_RUN_PREDICTION_FRAMEWORK_2026-03-24.md</a></li>
      <li><a href="{escape(_path_uri(OUT_FILE))}">spec_training_manifest.json</a></li>
    </ul>
  </div>
</section>

<footer class="muted" style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid var(--border)">
  Generated from local artifacts at {escape(manifest['generated_at'])}. Visible source links are included next to each major metric block.
</footer>
</main>
</body>
</html>
"""
    return html

# ── Summary printer ──────────────────────────────────────────────────────────

def print_summary(
    runs: list[dict],
    probes: list[dict],
    logbook_entries: list[dict],
    report_files: list[dict],
    manifest: dict,
):
    sep = "─" * 60
    print(f"\n{bold('spec_training_manifest builder')}")
    print(sep)

    # Sources
    spec_dirs = {r["dir_name"] for r in runs}
    probe_dirs = {p["dir_name"] for p in probes}
    all_dirs = spec_dirs | probe_dirs
    print(f"  Run dirs scanned          : {green(str(len(all_dirs)))}")
    print(f"  Run ledger entries        : {green(str(len(runs)))}")
    print(f"  Probe reports loaded      : {green(str(len(probes)))}")
    print(f"  Logbook timeline entries  : {green(str(len(logbook_entries)))}")
    print(f"  Spec report files found   : {green(str(len(report_files)))}")

    # Missing checks
    dirs_no_ledger = probe_dirs - spec_dirs
    dirs_no_probe  = spec_dirs - probe_dirs
    if dirs_no_ledger:
        print(f"  Dirs with probe but no ledger : {yellow(str(len(dirs_no_ledger)))}")
    if dirs_no_probe:
        print(f"  Dirs with ledger but no probe : {yellow(str(len(dirs_no_probe)))}")
    if not logbook_entries:
        print(f"  {red('TRAINING_LOGBOOK.md not found or empty')}")

    # Per-spec summary table
    print(f"\n{bold('Per-spec summary')}:")
    print(f"  {'spec':<10} {'status':<18} {'runs':>5} {'best exact':>11}")
    print(f"  {'─'*10} {'─'*18} {'─'*5} {'─'*11}")
    for s in manifest["specs"]:
        best = s["best_result"]
        er_str = f"{best['exact_rate']:.3f}" if best and best.get("exact_rate") is not None else "—"
        status = s["status"]
        if status == "gold":
            status_str = green(status)
        elif status in ("iterating", "in_progress"):
            status_str = yellow(status)
        else:
            status_str = red(status)
        print(f"  {s['id']:<10} {status_str:<27} {s['total_runs']:>5} {er_str:>11}")

    # Incidents
    incidents = manifest["incidents"]
    if incidents:
        print(f"\n{bold('Incidents')}: {len(incidents)}")
        for inc in incidents[:5]:
            print(f"  {cyan(inc['date'])}  {inc['title'][:70]}")
        if len(incidents) > 5:
            print(f"  … and {len(incidents) - 5} more")

    print(sep)
    print(f"  Output: {green(str(OUT_FILE))}")
    print(f"  HTML  : {green(str(HTML_FILE))}")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{bold('Collecting training data …')}\n")

    runs            = collect_runs(SOURCE_ROOTS)
    probes          = collect_probes(SOURCE_ROOTS)
    logbook_entries = parse_logbook(REPORTS / "TRAINING_LOGBOOK.md")
    report_files    = collect_report_files(REPORTS)

    manifest = build_manifest(runs, probes, logbook_entries, report_files)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    HTML_FILE.write_text(render_html(manifest))

    print_summary(runs, probes, logbook_entries, report_files, manifest)

if __name__ == "__main__":
    main()
