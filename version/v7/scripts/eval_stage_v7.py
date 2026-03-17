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
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from probe_report_adapters_v7 import apply_output_adapter

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
SCHEMA = "ck.stage_eval_matrix.v2"
DEFAULT_BUDGET_MULTIPLIER = 1.15
DEFAULT_CONTEXT_LEN = 512
EVAL_CONTRACT_SCHEMA = "ck.eval_contract.v1"

# ---------------------------------------------------------------------------
# Probe definitions
# A probe has: id, prompt, type (svg_gen | ood), optional expect_* fields
# ---------------------------------------------------------------------------
DEFAULT_PROBES = [
    {
        "id": "circle_cool_minimal",
        "description": "In-distribution SVG generation probe for circle + cool palette + minimal style.",
        "prompt": "[circle][palette:cool][style:minimal]<svg",
        "type": "svg_gen",
        "expect_shape": "circle",
        "expect_palette": "cool",
        "expect_style": "minimal",
    },
    {
        "id": "bar_chart_warm_bold",
        "description": "In-distribution SVG generation probe for bar-chart + warm palette + filled style.",
        "prompt": "[bar-chart][palette:warm][style:filled]<svg",
        "type": "svg_gen",
        "expect_shape": "bar-chart",
        "expect_palette": "warm",
        "expect_style": "filled",
    },
    {
        "id": "scatter_cool_minimal",
        "description": "In-distribution SVG generation probe for scatter + cool palette + minimal style.",
        "prompt": "[scatter][palette:cool][style:minimal]<svg",
        "type": "svg_gen",
        "expect_shape": "scatter",
        "expect_palette": "cool",
        "expect_style": "minimal",
    },
    {
        "id": "line_chart_bold_minimal",
        "description": "In-distribution SVG generation probe for line-chart + bold palette + minimal style.",
        "prompt": "[line-chart][palette:bold][style:minimal]<svg",
        "type": "svg_gen",
        "expect_shape": "line-chart",
        "expect_palette": "bold",
        "expect_style": "minimal",
    },
    {
        "id": "ood_unlabeled",
        "description": "OOD robustness probe: unlabeled prompt (<svg only). Checks whether model can still produce valid SVG without instruction tags.",
        "prompt": "<svg",
        "type": "ood",
        "expect_shape": None,
        "expect_palette": None,
        "expect_style": None,
    },
    {
        "id": "ood_unknown_type",
        "description": "OOD robustness probe with unknown shape tag. Checks fallback behavior under unseen tag types.",
        "prompt": "[unknown-shape][palette:cool]<svg",
        "type": "ood",
        "expect_shape": None,
        "expect_palette": "cool",
        "expect_style": None,
    },
]

DEFAULT_STAGE_METRICS = [
    {"key": "valid_svg_rate", "label": "Valid SVG", "description": "Fraction of probe outputs that parse as valid SVG/XML.", "source": "valid_svg", "probe_type": "svg_gen", "good": 0.75, "warn": 0.35, "format": "pct", "higher_is_better": True, "headline": True},
    {"key": "closure_success_rate", "label": "Closure", "description": "Fraction of outputs containing a proper closing </svg> tag.", "source": "closure", "probe_type": "svg_gen", "good": 0.70, "warn": 0.30, "format": "pct", "higher_is_better": True},
    {"key": "prefix_integrity", "label": "Prefix", "description": "How often output starts cleanly at expected content boundary (no preamble drift).", "source": "prefix_integrity", "probe_type": "all", "good": 0.80, "warn": 0.40, "format": "pct", "higher_is_better": True},
    {"key": "ood_robustness", "label": "OOD", "description": "Validity under out-of-distribution probes (unlabeled/unknown-tag prompts).", "source": "valid_svg", "probe_type": "ood", "good": 0.50, "warn": 0.20, "format": "pct", "higher_is_better": True, "headline": True},
    {"key": "adherence", "label": "Adherence", "description": "Instruction adherence score based on requested shape/palette/style behavior.", "source": "adherence", "probe_type": "svg_gen", "good": 0.70, "warn": 0.30, "format": "pct", "higher_is_better": True, "regression_watch": True, "headline": True},
    {"key": "repetition_loop_score", "label": "Loop Score", "description": "Repetition risk (lower is better). High values indicate loop/collapse patterns.", "source": "repetition", "probe_type": "all", "good": 0.10, "warn": 0.30, "format": "pct", "higher_is_better": False},
    {"key": "tag_adherence", "label": "Tag Adh", "description": "Match score for explicit control tags (shape/palette tags echoed or respected).", "source": "tag_adherence", "probe_type": "svg_gen", "good": 0.70, "warn": 0.30, "format": "pct", "higher_is_better": True},
]

DEFAULT_PROBE_METRICS = [
    {"key": "valid_svg", "label": "Valid", "description": "This probe output parsed as valid SVG/XML.", "good": 0.75, "warn": 0.35, "format": "pct", "higher_is_better": True},
    {"key": "closure", "label": "Closure", "description": "This probe output contains </svg> closure.", "good": 0.70, "warn": 0.30, "format": "pct", "higher_is_better": True},
    {"key": "prefix_integrity", "label": "Prefix", "description": "Output starts where expected (no unwanted preamble).", "good": 0.80, "warn": 0.40, "format": "pct", "higher_is_better": True},
    {"key": "repetition", "label": "Loop", "description": "Repetition score for this probe output (lower is better).", "good": 0.10, "warn": 0.30, "format": "pct", "higher_is_better": False},
    {"key": "adherence", "label": "Adh", "description": "Instruction adherence for this specific probe.", "good": 0.70, "warn": 0.30, "format": "pct", "higher_is_better": True},
    {"key": "tag_adherence", "label": "Tag Adh", "description": "Tag-level adherence for this specific probe.", "good": 0.70, "warn": 0.30, "format": "pct", "higher_is_better": True},
]


def _default_eval_contract() -> dict[str, Any]:
    return {
        "schema": EVAL_CONTRACT_SCHEMA,
        "dataset_type": "svg",
        "scorer": "svg",
        "output_adapter": None,
        "probes": [dict(p) for p in DEFAULT_PROBES],
        "stage_metrics": [dict(m) for m in DEFAULT_STAGE_METRICS],
        "probe_metrics": [dict(m) for m in DEFAULT_PROBE_METRICS],
        "headline_metrics": ["valid_svg_rate", "ood_robustness", "adherence"],
    }

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


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_metric_def(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    key = str(raw.get("key") or "").strip()
    source = str(raw.get("source") or key).strip()
    if not key:
        return None
    probe_type = str(raw.get("probe_type") or "all").strip().lower() or "all"
    fmt = str(raw.get("format") or "pct").strip().lower()
    if fmt not in {"pct", "float", "int", "text"}:
        fmt = "pct"
    m: dict[str, Any] = {
        "key": key,
        "label": str(raw.get("label") or key),
        "description": str(raw.get("description") or raw.get("tooltip") or raw.get("help") or "").strip(),
        "source": source,
        "probe_type": probe_type,
        "format": fmt,
        "higher_is_better": bool(raw.get("higher_is_better", True)),
    }
    for src_key, out_key in (
        ("good", "good"),
        ("warn", "warn"),
        ("regression_watch", "regression_watch"),
        ("headline", "headline"),
    ):
        if src_key in raw:
            m[out_key] = raw[src_key]
    return m


def _normalize_probe_metric_def(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    key = str(raw.get("key") or "").strip()
    if not key:
        return None
    fmt = str(raw.get("format") or "pct").strip().lower()
    if fmt not in {"pct", "float", "int", "text"}:
        fmt = "pct"
    m: dict[str, Any] = {
        "key": key,
        "label": str(raw.get("label") or key),
        "description": str(raw.get("description") or raw.get("tooltip") or raw.get("help") or "").strip(),
        "format": fmt,
        "higher_is_better": bool(raw.get("higher_is_better", True)),
    }
    for src_key in ("good", "warn"):
        if src_key in raw:
            m[src_key] = raw[src_key]
    return m


def _load_eval_contract(run_dir: Path, probe_config_path: str | None) -> tuple[dict[str, Any], str]:
    default_contract = _default_eval_contract()
    candidates: list[Path] = []
    if probe_config_path:
        candidates.append(Path(probe_config_path).expanduser())
    candidates.extend([
        run_dir / "eval_probes.json",
        run_dir / "eval_contract.json",
    ])
    for path in candidates:
        if not path.exists():
            continue
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        probes = doc.get("probes")
        if not isinstance(probes, list) or len(probes) == 0:
            print(f"[WARN] eval contract missing non-empty probes list: {path}; using defaults.")
            continue
        contract = dict(default_contract)
        contract["dataset_type"] = str(doc.get("dataset_type") or default_contract["dataset_type"]).strip().lower() or "svg"
        contract["scorer"] = str(doc.get("scorer") or ("svg" if contract["dataset_type"] == "svg" else "text_rules")).strip().lower()
        if isinstance(doc.get("output_adapter"), dict):
            contract["output_adapter"] = dict(doc["output_adapter"])
        normalized_probes: list[dict[str, Any]] = []
        for p in probes:
            if not isinstance(p, dict):
                continue
            pid = str(p.get("id") or "").strip()
            prompt = str(p.get("prompt") or "").strip()
            if not pid or not prompt:
                continue
            rec = dict(p)
            rec["id"] = pid
            rec["prompt"] = prompt
            rec["type"] = str(p.get("type") or "all").strip().lower() or "all"
            normalized_probes.append(rec)
        contract["probes"] = normalized_probes
        if not contract["probes"]:
            print(f"[WARN] eval contract has no valid probes: {path}; using defaults.")
            continue

        stage_metrics_raw = doc.get("stage_metrics")
        if isinstance(stage_metrics_raw, list) and stage_metrics_raw:
            normalized_stage = [_normalize_metric_def(x) for x in stage_metrics_raw]
            contract["stage_metrics"] = [m for m in normalized_stage if isinstance(m, dict)]
        probe_metrics_raw = doc.get("probe_metrics")
        if isinstance(probe_metrics_raw, list) and probe_metrics_raw:
            normalized_probe = [_normalize_probe_metric_def(x) for x in probe_metrics_raw]
            contract["probe_metrics"] = [m for m in normalized_probe if isinstance(m, dict)]
        headline = doc.get("headline_metrics")
        if isinstance(headline, list) and headline:
            contract["headline_metrics"] = [str(x) for x in headline if str(x).strip()]

        if not contract.get("stage_metrics"):
            # Minimal generic defaults for non-svg contracts
            if contract["scorer"] == "text_rules":
                contract["stage_metrics"] = [
                    {"key": "non_empty_rate", "label": "Non-empty", "source": "non_empty", "probe_type": "all", "good": 0.95, "warn": 0.70, "format": "pct", "higher_is_better": True, "headline": True},
                    {"key": "contains_all_rate", "label": "Contains", "source": "contains_all", "probe_type": "all", "good": 0.80, "warn": 0.50, "format": "pct", "higher_is_better": True, "headline": True},
                    {"key": "repetition_loop_score", "label": "Loop Score", "source": "repetition", "probe_type": "all", "good": 0.10, "warn": 0.30, "format": "pct", "higher_is_better": False, "headline": True},
                ]
            else:
                contract["stage_metrics"] = [dict(m) for m in DEFAULT_STAGE_METRICS]
        if not contract.get("probe_metrics"):
            if contract["scorer"] == "text_rules":
                contract["probe_metrics"] = [
                    {"key": "non_empty", "label": "Non-empty", "good": 0.95, "warn": 0.70, "format": "pct", "higher_is_better": True},
                    {"key": "contains_all", "label": "Contains", "good": 0.80, "warn": 0.50, "format": "pct", "higher_is_better": True},
                    {"key": "repetition", "label": "Loop", "good": 0.10, "warn": 0.30, "format": "pct", "higher_is_better": False},
                ]
            else:
                contract["probe_metrics"] = [dict(m) for m in DEFAULT_PROBE_METRICS]
        if not contract.get("headline_metrics"):
            contract["headline_metrics"] = [m["key"] for m in contract["stage_metrics"][:3]]
        return contract, str(path)

    return default_contract, "builtin_default"


def _candidate_dataset_manifests(dataset_path: Path) -> list[Path]:
    p = dataset_path.expanduser().resolve()
    parent = p.parent
    stem = p.stem
    candidates = [
        parent / f"{stem}_manifest.json",
        parent / f"{stem}.manifest.json",
        parent / "dataset_manifest.json",
        parent / "manifest.json",
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.exists():
            out.append(c)
    return out


def _extract_budget_from_manifest(dataset_path: Path) -> tuple[int | None, int | None, str | None]:
    for manifest_path in _candidate_dataset_manifests(dataset_path):
        doc = _load_json(manifest_path)
        if not isinstance(doc, dict):
            continue

        sources: list[tuple[str, dict[str, Any]]] = []
        ctr = doc.get("completion_token_stats")
        if isinstance(ctr, dict):
            sources.append(("completion_token_stats", ctr))
        ctr = doc.get("eval_contract")
        if isinstance(ctr, dict):
            sources.append(("eval_contract", ctr))
        sources.append(("manifest_root", doc))

        p95: int | None = None
        p99: int | None = None
        src_name: str | None = None
        for src, payload in sources:
            v95 = payload.get("p95_completion_tokens")
            v99 = payload.get("p99_completion_tokens")
            if isinstance(v95, (int, float)) and int(v95) > 0:
                p95 = int(v95)
                src_name = src
            if isinstance(v99, (int, float)) and int(v99) > 0:
                p99 = int(v99)
                src_name = src
            if p95 is not None or p99 is not None:
                break
        if p95 is None and p99 is None:
            continue
        source = f"dataset_manifest:{manifest_path.name}:{src_name or 'root'}"
        return p95, p99, source
    return None, None, None


def _extract_budget_from_pack_report(run_dir: Path, run_id: str) -> tuple[int | None, str | None]:
    candidate_paths = [
        run_dir / ".ck_pipeline" / run_id / "train_token_pack.json",
        run_dir / "train_token_pack.json",
    ]
    for path in candidate_paths:
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        for key in ("row_tokens_p99", "row_tokens_p95", "row_tokens_max", "max_row_tokens"):
            v = doc.get(key)
            if isinstance(v, (int, float)) and int(v) > 0:
                return int(v), f"pack_report:{path.name}:{key}"
    return None, None


def _resolve_probe_max_tokens(cli_max_tokens: int, run_dir: Path, ledger_rec: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    hard_cap = max(64, int(DEFAULT_CONTEXT_LEN) - 16)
    fallback = min(512, hard_cap)

    if int(cli_max_tokens or 0) > 0:
        override = int(cli_max_tokens)
        return max(64, min(override, hard_cap)), {
            "source": "cli_override",
            "base_tokens": int(override),
            "multiplier": 1.0,
            "fallback": int(fallback),
            "cap": int(hard_cap),
        }

    dataset_raw = str(ledger_rec.get("dataset") or "").strip()
    if dataset_raw:
        ds_path = Path(dataset_raw).expanduser()
        if ds_path.exists():
            p95, p99, manifest_source = _extract_budget_from_manifest(ds_path)
            base = p99 if isinstance(p99, int) and p99 > 0 else p95
            if isinstance(base, int) and base > 0:
                budget = int(math.ceil(float(base) * DEFAULT_BUDGET_MULTIPLIER))
                budget = max(64, min(budget, hard_cap))
                return int(budget), {
                    "source": str(manifest_source or "dataset_manifest"),
                    "base_tokens": int(base),
                    "multiplier": float(DEFAULT_BUDGET_MULTIPLIER),
                    "fallback": int(fallback),
                    "cap": int(hard_cap),
                }

    run_id = str(ledger_rec.get("run_id") or "")
    if run_id:
        base, source = _extract_budget_from_pack_report(run_dir, run_id)
        if isinstance(base, int) and base > 0:
            budget = int(math.ceil(float(base) * DEFAULT_BUDGET_MULTIPLIER))
            budget = max(64, min(budget, hard_cap))
            return int(budget), {
                "source": str(source or "pack_report"),
                "base_tokens": int(base),
                "multiplier": float(DEFAULT_BUDGET_MULTIPLIER),
                "fallback": int(fallback),
                "cap": int(hard_cap),
            }

    return int(fallback), {
        "source": "fallback",
        "base_tokens": None,
        "multiplier": float(DEFAULT_BUDGET_MULTIPLIER),
        "fallback": int(fallback),
        "cap": int(hard_cap),
    }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
_SVG_OPEN_RE = re.compile(r"<svg[\s>]", re.IGNORECASE)
_SVG_CLOSE_RE = re.compile(r"</svg\s*>", re.IGNORECASE)

# Palette keyword sets used only as fallback when palette tag is missing.
_PALETTE_KEYWORDS: dict[str, set[str]] = {
    "cool": {"blue", "teal", "cyan", "turquoise", "aqua", "skyblue", "royalblue", "steelblue"},
    "warm": {"red", "orange", "yellow", "coral", "tomato", "gold", "crimson", "firebrick", "maroon"},
    "neutral": {"gray", "grey", "slate", "silver", "black", "white", "zinc", "stone"},
    "bold": {"red", "orange", "blue", "green", "pink", "magenta", "cyan", "violet"},
    "pastel": {"pastel", "lavender", "mint", "peach", "rose", "cream"},
    "dark": {"dark", "midnight", "navy", "charcoal", "black"},
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


def _canonical_shape(shape: str | None) -> str:
    s = str(shape or "").strip().lower()
    return s.replace("-", "_")


def _prefix_integrity(text: str) -> bool:
    t = text.strip()
    if t.startswith("<svg"):
        return True
    # Accept tag-conditioned responses like:
    # ][circle][palette:cool]<svg ... or [circle][palette:cool]<svg ...
    return bool(re.match(r"^\]?(?:\[[^\]]+\])+\s*<svg", t, re.IGNORECASE))


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
    """Check adherence to probe's expect_shape / expect_palette / expect_color."""
    score = 0.0
    checks = 0

    shape = probe.get("expect_shape")
    palette = probe.get("expect_palette")
    color = probe.get("expect_color")
    text_lower = text.lower()
    tag_block_raw = _extract_tag_block(text).lower()

    if palette:
        checks += 1
        p = str(palette).strip().lower()
        hit = f"palette:{p}" in tag_block_raw
        if not hit and p in _PALETTE_KEYWORDS:
            hit = any(tok in text_lower for tok in _PALETTE_KEYWORDS[p])
        if hit:
            score += 1.0

    if shape:
        checks += 1
        canonical_shape = _canonical_shape(shape)
        shape_variants = [str(shape).strip().lower(), canonical_shape, canonical_shape.replace("_", "-")]
        hit = any(v in tag_block_raw for v in shape_variants)
        if not hit and canonical_shape in _SHAPE_SVG_TAGS:
            hit = any(tag.lower() in text_lower for tag in _SHAPE_SVG_TAGS[canonical_shape])
        if hit:
            score += 1.0

    if color:
        checks += 1
        c = str(color).strip().lower()
        hit = f"color:{c}" in tag_block_raw or f"fill:{c}" in text_lower or f">{c}<" in text_lower
        if hit:
            score += 1.0

    if checks == 0:
        return 1.0  # OOD probe — no requirements
    return score / checks


def _extract_response_text(raw: str, prompt: str) -> str:
    """Strip ck_chat.py loading preamble and return just the model response.

    ck_chat.py emits 'Loading model from...' + tokenizer messages to stdout even
    with --no-stats.  The actual generation appears after 'Response: ' on its own
    line (or 'Response:').  Fall back to the raw text so scoring still works for
    unexpected output shapes.
    """
    # Try 'Response: ' marker first (most reliable)
    for marker in ("\nResponse: ", "\nResponse:", "Response: ", "Response:"):
        idx = raw.find(marker)
        if idx != -1:
            return raw[idx + len(marker):]
    # Fallback: find the first occurrence of '<svg' and use from there
    idx = raw.lower().find("<svg")
    if idx != -1:
        return raw[idx:]
    return raw


def _score_output_svg(raw: str, probe: dict[str, Any], output_adapter: dict[str, Any] | None = None) -> dict[str, Any]:
    response = _extract_response_text(raw, probe["prompt"])
    scored_text = response
    materialized_output = ""
    if isinstance(output_adapter, dict) and str(output_adapter.get("name") or "").strip():
        try:
            adapted = apply_output_adapter(str(output_adapter.get("name")), response, output_adapter)
        except Exception:
            adapted = {}
        parsed_output = adapted.get("parsed_output")
        if isinstance(parsed_output, str) and parsed_output.strip():
            scored_text = parsed_output.strip()
        materialized = adapted.get("materialized_output")
        if isinstance(materialized, str) and materialized.strip():
            materialized_output = materialized.strip()
    # Reconstruct full sequence for validity/closure checks.
    # Insert a guaranteed space between prompt and response so "<svg"+"xmlns=..."
    # doesn't collapse to "<svgxmlns=...>".
    validity_target = materialized_output or scored_text
    if "<svg" in probe["prompt"]:
        sep = " " if validity_target and validity_target[0] not in (" ", ">", "\n", "\t", "]", "/") else ""
        full_text = probe["prompt"] + sep + validity_target
    else:
        full_text = validity_target
    return {
        "valid_svg": float(_is_valid_svg(full_text)),
        "closure": float(_has_closure(full_text)),
        # prefix_integrity is measured on generated response only.
        "prefix_integrity": float(_prefix_integrity(scored_text)),
        "repetition": _repetition_score(scored_text),
        "adherence": _adherence_score(scored_text, probe),
        "tag_adherence": _tag_adherence_score(scored_text, probe),
        # Store the substituted tag block so the visualizer can show it
        "model_tag_block": _extract_tag_block(scored_text),
    }


def _extract_tag_block(response: str) -> str:
    """Extract leading [...][...] tag block from the response if present."""
    m = re.match(r'^(\]?(?:\[[^\]]*\])+)', response.strip())
    return m.group(1) if m else ""


def _tag_adherence_score(response: str, probe: dict[str, Any]) -> float:
    """Check whether the model echoed the correct palette/shape tags in its output.

    Many spec02 outputs start with a tag block like ][line-chart][palette:cool][labeled].
    This metric checks if the expected palette and chart-type tags appear verbatim
    in that block, giving a cleaner signal than SVG color-keyword matching.
    Returns 1.0 if all expected tags are present, 0.5 if palette OR shape only,
    0.0 if neither.  OOD probes (no expectations) return 1.0.
    """
    tag_block_raw = response.strip()[:120].lower()
    checks = 0
    score = 0.0

    palette = probe.get("expect_palette")
    shape = _canonical_shape(probe.get("expect_shape"))
    color = str(probe.get("expect_color") or "").strip().lower()

    if palette:
        checks += 1
        if f"palette:{palette}" in tag_block_raw:
            score += 1.0

    if shape:
        checks += 1
        # Normalise: bar_chart vs bar-chart etc.
        shape_variants = [shape, shape.replace("_", "-"), shape.replace("-", "_")]
        if any(v in tag_block_raw for v in shape_variants):
            score += 1.0

    if color:
        checks += 1
        if f"color:{color}" in tag_block_raw or f"fill:{color}" in tag_block_raw:
            score += 1.0

    if checks == 0:
        return 1.0
    return score / checks


def _score_output_text_rules(raw: str, probe: dict[str, Any]) -> dict[str, Any]:
    response = _extract_response_text(raw, str(probe.get("prompt") or "")).strip()
    out: dict[str, Any] = {
        "non_empty": float(bool(response)),
        "repetition": _repetition_score(response),
    }

    contains = probe.get("expect_contains")
    if contains is None:
        contains = probe.get("expect_contains_all")
    contains_items: list[str] = []
    if isinstance(contains, str) and contains.strip():
        contains_items = [contains]
    elif isinstance(contains, list):
        contains_items = [str(x) for x in contains if str(x).strip()]
    if contains_items:
        contains_val = float(all(item in response for item in contains_items))
        out["contains_all"] = contains_val
        out["contains"] = contains_val

    forbid = probe.get("expect_forbid")
    forbid_items: list[str] = []
    if isinstance(forbid, str) and forbid.strip():
        forbid_items = [forbid]
    elif isinstance(forbid, list):
        forbid_items = [str(x) for x in forbid if str(x).strip()]
    if forbid_items:
        out["forbid_clean"] = float(all(item not in response for item in forbid_items))

    prefix = probe.get("expect_prefix")
    if isinstance(prefix, str) and prefix:
        out["prefix_match"] = float(response.startswith(prefix))

    exact = probe.get("expect_exact")
    if isinstance(exact, str):
        out["exact_match"] = float(response == exact)

    regex = probe.get("expect_regex")
    if isinstance(regex, str) and regex:
        try:
            out["regex_match"] = float(bool(re.search(regex, response, re.MULTILINE)))
        except re.error:
            out["regex_match"] = 0.0
    return out


def _score_output(raw: str, probe: dict[str, Any], scorer: str, output_adapter: dict[str, Any] | None = None) -> dict[str, Any]:
    s = str(scorer or "").strip().lower()
    if s in {"text", "text_rules", "generic"}:
        return _score_output_text_rules(raw, probe)
    # default svg scorer
    return _score_output_svg(raw, probe, output_adapter)


def _probe_expected_summary(probe: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("expect_shape", "expect_palette", "expect_style"):
        value = probe.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key.replace('expect_', '')}={value.strip()}")
    color = probe.get("expect_color")
    if isinstance(color, str) and color.strip():
        parts.append(f"color={color.strip()}")
    contains = probe.get("expect_contains")
    if contains is None:
        contains = probe.get("expect_contains_all")
    contains_items: list[str] = []
    if isinstance(contains, str) and contains.strip():
        contains_items = [contains.strip()]
    elif isinstance(contains, list):
        contains_items = [str(x).strip() for x in contains if str(x).strip()]
    if contains_items:
        parts.append(f"contains={','.join(contains_items[:4])}")
    forbid = probe.get("expect_forbid")
    forbid_items: list[str] = []
    if isinstance(forbid, str) and forbid.strip():
        forbid_items = [forbid.strip()]
    elif isinstance(forbid, list):
        forbid_items = [str(x).strip() for x in forbid if str(x).strip()]
    if forbid_items:
        parts.append(f"forbid={','.join(forbid_items[:4])}")
    prefix = probe.get("expect_prefix")
    if isinstance(prefix, str) and prefix.strip():
        parts.append(f"prefix={prefix.strip()}")
    regex = probe.get("expect_regex")
    if isinstance(regex, str) and regex.strip():
        parts.append(f"regex={regex.strip()}")
    if parts:
        return "; ".join(parts)
    return "none"


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
# Aggregation helpers
# ---------------------------------------------------------------------------
def _aggregate_probe_scores(samples: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {}
    if not samples:
        return agg
    numeric_values: dict[str, list[float]] = {}
    model_tag_blocks: list[str] = []
    for sample in samples:
        scores = sample.get("scores")
        if not isinstance(scores, dict):
            continue
        for key, value in scores.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                numeric_values.setdefault(str(key), []).append(float(value))
        mtb = scores.get("model_tag_block")
        if isinstance(mtb, str) and mtb.strip():
            model_tag_blocks.append(mtb)
    for key, vals in numeric_values.items():
        agg[key] = round(sum(vals) / len(vals), 4) if vals else 0.0
    if model_tag_blocks:
        agg["dominant_tag_block"] = max(set(model_tag_blocks), key=model_tag_blocks.count)
    return agg


def _metric_avg_from_probe_results(probe_results: list[dict[str, Any]], *, probe_type: str, source_key: str) -> float:
    values: list[float] = []
    for row in probe_results:
        if probe_type != "all" and str(row.get("type") or "") != probe_type:
            continue
        agg = row.get("agg")
        if not isinstance(agg, dict):
            continue
        v = agg.get(source_key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            values.append(float(v))
    return round(sum(values) / len(values), 4) if values else 0.0


def _build_stage_metrics(
    probe_results: list[dict[str, Any]],
    stage_metrics: list[dict[str, Any]],
    *,
    n_samples: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for mdef in stage_metrics:
        key = str(mdef.get("key") or "").strip()
        source = str(mdef.get("source") or key).strip()
        probe_type = str(mdef.get("probe_type") or "all").strip().lower() or "all"
        if not key or not source:
            continue
        metrics[key] = _metric_avg_from_probe_results(probe_results, probe_type=probe_type, source_key=source)
    metrics["n_samples"] = int(max(1, n_samples))
    metrics["n_probes"] = int(len(probe_results))
    return metrics


# ---------------------------------------------------------------------------
# Per-run eval
# ---------------------------------------------------------------------------
def eval_run(
    run_dir: Path,
    ledger_rec: dict[str, Any],
    eval_contract: dict[str, Any],
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
    resolved_max_tokens, budget_meta = _resolve_probe_max_tokens(max_tokens, run_dir, ledger_rec)
    probes = eval_contract.get("probes") if isinstance(eval_contract.get("probes"), list) else []
    scorer_default = str(eval_contract.get("scorer") or "svg").strip().lower()
    dataset_type = str(eval_contract.get("dataset_type") or "svg").strip().lower()
    output_adapter_default = eval_contract.get("output_adapter") if isinstance(eval_contract.get("output_adapter"), dict) else None
    stage_metric_defs = eval_contract.get("stage_metrics") if isinstance(eval_contract.get("stage_metrics"), list) else []
    probe_metric_defs = eval_contract.get("probe_metrics") if isinstance(eval_contract.get("probe_metrics"), list) else []
    headline_metrics = eval_contract.get("headline_metrics") if isinstance(eval_contract.get("headline_metrics"), list) else []

    ck_build = run_dir / ".ck_build"

    print(f"\n{'=' * 60}")
    print(f"  run_id:     {run_id}")
    print(f"  stage:      {stage}  pass={stage_pass}  label={phase_label}")
    print(f"  final_loss: {final_loss}")
    print(f"  dataset:    {dataset_type}  scorer={scorer_default}")
    if output_adapter_default:
        print(f"  adapter:    {output_adapter_default.get('name')}:{output_adapter_default.get('renderer', '-')}")
    print(
        f"  eval budget:{resolved_max_tokens} "
        f"(source={budget_meta.get('source')}, base={budget_meta.get('base_tokens')}, x{budget_meta.get('multiplier')})"
    )
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
    for probe in probes:
        probe_id = probe["id"]
        prompt = probe["prompt"]
        scorer = str(probe.get("scorer") or scorer_default).strip().lower() or "svg"
        probe_output_adapter = probe.get("output_adapter") if isinstance(probe.get("output_adapter"), dict) else output_adapter_default
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
                    "--max-tokens", str(resolved_max_tokens),
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
                    f"Response: <svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'>"
                    f"<circle cx='50' cy='50' r='40' fill='blue'/></svg>"
                )
            raw = (output or "").strip()
            response = _extract_response_text(raw, prompt)
            scores = _score_output(raw, probe, scorer, probe_output_adapter)
            samples.append({
                "idx": i,
                "response": response[:600],   # store clean response text
                "scores": scores,
            })
            primary_metric = "valid_svg" if "valid_svg" in scores else ("non_empty" if "non_empty" in scores else "")
            primary_label = f"{primary_metric}={scores.get(primary_metric):.2f}" if primary_metric else "sample-scored"
            rep_txt = f" rep={scores.get('repetition', 0.0):.2f}" if isinstance(scores.get("repetition"), (int, float)) else ""
            tag_block = scores.get("model_tag_block", "")
            print(
                f"    sample {i}: {primary_label}{rep_txt}"
                + (f"  model_tags={str(tag_block)[:40]}" if tag_block else "")
            )

        agg = _aggregate_probe_scores(samples)
        probe_results.append({
            "probe_id": probe_id,
            "prompt": prompt,
            "type": str(probe.get("type") or "all"),
            "description": str(probe.get("description") or "").strip(),
            "expected": _probe_expected_summary(probe),
            "scorer": scorer,
            "samples": samples,
            "agg": agg,
        })
        dtag = agg.get("dominant_tag_block", "")
        numeric_preview = ", ".join(
            f"{k}={v:.2f}" for k, v in agg.items()
            if isinstance(v, (int, float)) and math.isfinite(float(v))
        )
        print(f"    agg: {numeric_preview if numeric_preview else '-'}"
              + (f"  model→{str(dtag)[:40]}" if dtag else ""))

    # ── Step 4: Aggregate across probes ────────────────────────────────────
    metrics = _build_stage_metrics(probe_results, stage_metric_defs, n_samples=n_samples)
    metric_preview = ", ".join(
        f"{k}={v}" for k, v in metrics.items()
        if k not in {"n_samples", "n_probes"} and isinstance(v, (int, float))
    )
    print(f"\n  → metrics: {metric_preview}")

    return {
        "run_id": run_id,
        "stage": stage,
        "stage_pass": stage_pass,
        "phase_label": phase_label,
        "run_order": run_order,
        "final_loss": final_loss,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_type": dataset_type,
        "eval_config": {
            "max_tokens": int(resolved_max_tokens),
            "token_budget_source": str(budget_meta.get("source") or "unknown"),
            "token_budget_base_tokens": budget_meta.get("base_tokens"),
            "token_budget_multiplier": float(budget_meta.get("multiplier") or 1.0),
            "scorer": scorer_default,
            "probe_config_source": str(eval_contract.get("_source") or "builtin_default"),
        },
        "metric_columns": stage_metric_defs,
        "probe_metric_columns": probe_metric_defs,
        "headline_metrics": headline_metrics,
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
        "probes": [dict(p) for p in DEFAULT_PROBES],
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
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Max tokens per sample (0 = auto from dataset/pack stats, fallback capped by context)",
    )
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0 = greedy)")
    ap.add_argument("--probe-config", default="", help="Optional eval contract/probe file (JSON). Default: RUN_DIR/eval_probes.json")
    ap.add_argument("--dry-run", action="store_true", help="Print what would run, without executing")
    args = ap.parse_args()

    if args.stage_pass is not None and not args.stage:
        ap.error("--stage-pass requires --stage")
    if args.max_tokens < 0:
        ap.error("--max-tokens must be >= 0 (0=auto)")

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

    eval_contract, contract_source = _load_eval_contract(run_dir, args.probe_config or None)
    eval_contract["_source"] = contract_source
    print(f"[eval-matrix] contract: {contract_source}  dataset_type={eval_contract.get('dataset_type')}  probes={len(eval_contract.get('probes', []))}")

    matrix = _load_matrix(run_dir)
    matrix["schema"] = SCHEMA
    matrix["run_dir"] = str(run_dir)
    matrix.setdefault("model_name", run_dir.name)
    matrix["dataset_type"] = eval_contract.get("dataset_type", "svg")
    matrix["probes"] = eval_contract.get("probes", [])
    matrix["metric_columns"] = eval_contract.get("stage_metrics", [])
    matrix["probe_metric_columns"] = eval_contract.get("probe_metrics", [])
    matrix["headline_metrics"] = eval_contract.get("headline_metrics", [])
    matrix["probe_config_source"] = contract_source

    for rec in targets:
        entry = eval_run(
            run_dir, rec, eval_contract,
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
            metric_cols = matrix.get("metric_columns") if isinstance(matrix.get("metric_columns"), list) else []
            preview_cols = [c for c in metric_cols if isinstance(c, dict)][:2]
            preview_bits = []
            for col in preview_cols:
                key = str(col.get("key") or "")
                val = m.get(key)
                if isinstance(val, (int, float)) and math.isfinite(float(val)):
                    preview_bits.append(f"{key}={float(val):.2f}")
            preview = "  ".join(preview_bits) if preview_bits else "-"
            print(f"  [{e.get('run_order')}] {e.get('phase_label'):15s}  {preview}")
        return 0

    print(f"\n[done] stage_eval_matrix.json  —  {len(targets)} entry/entries updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
