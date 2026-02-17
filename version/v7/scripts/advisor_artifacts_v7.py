#!/usr/bin/env python3
"""
Generate advisor_summary.json for v7 IR visualizer.

Primary goal: keep a stable, portable schema with artifact paths and lightweight
summary metrics extracted from Advisor roofline reports when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_model_dir_from_input(model_input: str) -> Optional[Path]:
    try:
        from ck_run_v7 import CACHE_DIR, detect_input_type  # type: ignore
    except Exception:
        return None

    input_type, info = detect_input_type(model_input)
    if input_type == "hf_gguf":
        return CACHE_DIR / info["repo_id"].replace("/", "--")
    if input_type == "hf_id":
        return CACHE_DIR / info["model_id"].replace("/", "--")
    if input_type == "gguf":
        return CACHE_DIR / info["path"].stem
    if input_type == "local_dir":
        local = Path(info["path"]).resolve()
        return local.parent if local.name == ".ck_build" else local
    if input_type == "local_config":
        cfg_parent = Path(info["path"]).resolve().parent
        return cfg_parent.parent if cfg_parent.name == ".ck_build" else cfg_parent
    return None


def truncate_text(text: str, limit: int = 120_000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _normalize_metric_key(name: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return key[:80]


def parse_report_metrics(raw_text: str) -> Dict[str, float]:
    if not raw_text:
        return {}

    metrics: Dict[str, float] = {}
    kv_re = re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9 _()./%:-]{1,80})\s*[:=]\s*([-+]?[0-9]*\.?[0-9]+)\s*([A-Za-z/%]+)?\s*$"
    )
    for line in raw_text.splitlines():
        m = kv_re.match(line)
        if not m:
            continue
        raw_key = m.group(1)
        raw_value = m.group(2)
        unit = (m.group(3) or "").strip()
        key = _normalize_metric_key(raw_key)
        if not key:
            continue
        if unit:
            key = _normalize_metric_key(f"{key}_{unit}")
        if key in metrics:
            continue
        try:
            metrics[key] = float(raw_value)
        except ValueError:
            continue
    return metrics


def parse_csv_preview(path: Path, max_rows: int = 32) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    if not text.strip():
        return []

    sample = "\n".join(text.splitlines()[:5])
    delimiter = ","
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except Exception:
        first = text.splitlines()[0] if text.splitlines() else ""
        if "\t" in first:
            delimiter = "\t"
        elif ";" in first:
            delimiter = ";"

    rows: List[Dict[str, str]] = []
    with open(path, "r", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            cleaned = {str(k): (v or "").strip() for k, v in row.items() if k}
            if cleaned:
                rows.append(cleaned)
            if len(rows) >= max_rows:
                break
    return rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate advisor_summary.json for v7")
    parser.add_argument("--model-dir", type=Path, help="Model output directory")
    parser.add_argument("--model-input", type=str, help="Model input string (same as ck_run_v7.py)")
    parser.add_argument("--out-dir", type=Path, help="Output directory (default: model dir)")
    parser.add_argument("--analysis", type=str, default="roofline", help="Advisor analysis label")
    parser.add_argument("--project-dir", type=Path, required=True, help="Advisor project directory")
    parser.add_argument("--report-text", type=Path, help="Advisor report text output")
    parser.add_argument("--report-csv", type=Path, help="Advisor report CSV output")
    parser.add_argument("--report-html", type=Path, help="Advisor report HTML output")
    args = parser.parse_args()

    model_dir = args.model_dir
    if model_dir is None and args.model_input:
        model_dir = detect_model_dir_from_input(args.model_input)
    out_dir = args.out_dir or model_dir
    if out_dir is None:
        parser.error("Provide --out-dir or one of --model-dir/--model-input")

    report_text = args.report_text if args.report_text and args.report_text.exists() else None
    report_csv = args.report_csv if args.report_csv and args.report_csv.exists() else None
    report_html = args.report_html if args.report_html and args.report_html.exists() else None
    raw_text = truncate_text(report_text.read_text(errors="ignore")) if report_text else ""

    artifacts = [{"label": "Advisor project", "path": str(args.project_dir)}]
    if report_text:
        artifacts.append({"label": "Advisor report (text)", "path": str(report_text)})
    if report_csv:
        artifacts.append({"label": "Advisor report (csv)", "path": str(report_csv)})
    if report_html:
        artifacts.append({"label": "Advisor report (html)", "path": str(report_html)})

    payload: Dict[str, object] = {
        "generated_at": utc_now_iso(),
        "analysis": str(args.analysis or "roofline"),
        "project_dir": str(args.project_dir),
        "project_path": str(args.project_dir),
        "report_path": str(report_text) if report_text else None,
        "csv_path": str(report_csv) if report_csv else None,
        "html_path": str(report_html) if report_html else None,
        "summary_metrics": parse_report_metrics(raw_text),
        "preview_rows": parse_csv_preview(report_csv) if report_csv else [],
        "raw_text": raw_text,
        "artifacts": artifacts,
    }

    out_path = Path(out_dir) / "advisor_summary.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
