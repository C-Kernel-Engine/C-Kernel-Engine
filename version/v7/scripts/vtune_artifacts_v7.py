#!/usr/bin/env python3
"""
Generate vtune_summary.json for IR visualizer.

Primary compatibility payload stays intact (hotspots fields), and richer
multi-analysis data is exposed under `analyses`.
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
        return Path(info["path"]).resolve()
    if input_type == "local_config":
        return Path(info["path"]).resolve().parent
    return None


def truncate_text(text: str, limit: int = 120_000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def parse_number(text: str) -> Optional[float]:
    raw = (text or "").strip().replace(",", "")
    if not raw:
        return None
    for suffix in ("ms", "s", "%"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)].strip()
    try:
        return float(raw)
    except ValueError:
        return None


def pick_value(row: Dict[str, str], keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in row:
            val = parse_number(row[key])
            if val is not None:
                return val
    return None


def pick_text(row: Dict[str, str], keys: List[str]) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def parse_hotspots_csv(path: Path, top_k: int = 25) -> List[Dict[str, object]]:
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
        first_line = text.splitlines()[0] if text.splitlines() else ""
        if "\t" in first_line:
            delimiter = "\t"
        elif ";" in first_line:
            delimiter = ";"

    rows: List[Dict[str, object]] = []

    def _collect(reader: csv.DictReader) -> None:
        for row in reader:
            symbol = pick_text(
                row,
                [
                    "Function",
                    "Function/Call Stack",
                    "Call Stack",
                    "Source Function",
                    "Module",
                ],
            )
            if not symbol:
                continue
            value = pick_value(
                row,
                [
                    "CPU Time",
                    "CPU Time:Self",
                    "CPU Time:Total",
                    "Effective Time",
                    "Elapsed Time",
                ],
            )
            percent = pick_value(
                row,
                [
                    "CPU Time:Self %",
                    "CPU Time %",
                    "Effective Time %",
                    "Elapsed Time %",
                ],
            )
            rows.append(
                {
                    "symbol": symbol,
                    "value": value if value is not None else 0.0,
                    "percent": percent if percent is not None else 0.0,
                }
            )

    with open(path, "r", errors="ignore", newline="") as f:
        _collect(csv.DictReader(f, delimiter=delimiter))

    # VTune commonly emits TSV; retry with tab if first parse yielded nothing.
    if not rows and delimiter != "\t":
        with open(path, "r", errors="ignore", newline="") as f:
            _collect(csv.DictReader(f, delimiter="\t"))

    rows.sort(key=lambda x: float(x.get("value", 0.0) or 0.0), reverse=True)
    return rows[:top_k]


def parse_summary_metrics(raw_text: str, analysis_name: str) -> Dict[str, float]:
    if not raw_text:
        return {}

    def find_pct(label: str) -> Optional[float]:
        pat = re.compile(rf"{re.escape(label)}[^\n\r%]*?([-+]?[0-9]*\.?[0-9]+)\s*%", re.IGNORECASE)
        m = pat.search(raw_text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    def find_gbs(label: str) -> Optional[float]:
        pat = re.compile(rf"{re.escape(label)}[^\n\r]*?([-+]?[0-9]*\.?[0-9]+)\s*GB/s", re.IGNORECASE)
        m = pat.search(raw_text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    metrics: Dict[str, float] = {}
    analysis_key = analysis_name.lower()

    common_pct = [
        "Memory Bound",
        "Core Bound",
        "Retiring",
        "Frontend Bound",
        "Backend Bound",
        "Bad Speculation",
    ]
    mem_pct = ["DRAM Bound", "L1 Bound", "L2 Bound", "L3 Bound", "Store Bound"]

    keys = list(common_pct)
    if "memory" in analysis_key:
        keys += mem_pct

    for key in keys:
        val = find_pct(key)
        if val is not None:
            metrics[key] = val

    bw = find_gbs("Bandwidth")
    if bw is not None:
        metrics["Bandwidth GB/s"] = bw

    return metrics


def build_analysis_entry(
    name: str,
    result_dir: Optional[Path],
    report_text: Optional[Path],
    report_csv: Optional[Path],
) -> Dict[str, object]:
    raw_text = ""
    if report_text and report_text.exists():
        raw_text = truncate_text(report_text.read_text(errors="ignore"))

    hotspots = parse_hotspots_csv(report_csv) if report_csv and report_csv.exists() else []

    return {
        "name": name,
        "result_dir": str(result_dir) if result_dir else None,
        "report_text": str(report_text) if report_text else None,
        "report_csv": str(report_csv) if report_csv else None,
        "hotspots": hotspots,
        "top_hotspots": hotspots,
        "raw_text": raw_text,
        "summary_metrics": parse_summary_metrics(raw_text, name),
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate vtune_summary.json for v7")
    parser.add_argument("--model-dir", type=Path, help="Model output directory")
    parser.add_argument("--model-input", type=str, help="Model input string (same as ck_run_v7.py)")
    parser.add_argument("--out-dir", type=Path, help="Output directory (default: model dir)")
    parser.add_argument("--result-dir", type=Path, required=True, help="VTune result directory")
    parser.add_argument("--report-text", type=Path, help="VTune hotspots text report")
    parser.add_argument("--report-csv", type=Path, help="VTune hotspots CSV report")
    parser.add_argument(
        "--analysis-name",
        action="append",
        default=[],
        help="Optional extra VTune analysis name (repeatable), e.g. memory-access",
    )
    parser.add_argument(
        "--analysis-result-dir",
        action="append",
        type=Path,
        default=[],
        help="Optional extra VTune analysis result dir (repeatable)",
    )
    parser.add_argument(
        "--analysis-report-text",
        action="append",
        type=Path,
        default=[],
        help="Optional extra VTune analysis text report (repeatable)",
    )
    parser.add_argument(
        "--analysis-report-csv",
        action="append",
        type=Path,
        default=[],
        help="Optional extra VTune analysis CSV report (repeatable)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    if model_dir is None and args.model_input:
        model_dir = detect_model_dir_from_input(args.model_input)
    out_dir = args.out_dir or model_dir
    if out_dir is None:
        parser.error("Provide --out-dir or one of --model-dir/--model-input")

    primary = build_analysis_entry("hotspots", args.result_dir, args.report_text, args.report_csv)

    analyses: List[Dict[str, object]] = [primary]
    extra_count = max(
        len(args.analysis_name),
        len(args.analysis_result_dir),
        len(args.analysis_report_text),
        len(args.analysis_report_csv),
    )
    for idx in range(extra_count):
        name = args.analysis_name[idx] if idx < len(args.analysis_name) else f"analysis_{idx + 1}"
        result_dir = args.analysis_result_dir[idx] if idx < len(args.analysis_result_dir) else None
        report_text = args.analysis_report_text[idx] if idx < len(args.analysis_report_text) else None
        report_csv = args.analysis_report_csv[idx] if idx < len(args.analysis_report_csv) else None
        analyses.append(build_analysis_entry(name, result_dir, report_text, report_csv))

    artifacts: List[Dict[str, object]] = []
    for entry in analyses:
        name = str(entry.get("name") or "analysis")
        result_dir = entry.get("result_dir")
        report_text = entry.get("report_text")
        report_csv = entry.get("report_csv")
        if result_dir:
            artifacts.append({"label": f"VTune {name} result", "path": str(result_dir)})
        if report_text:
            artifacts.append({"label": f"VTune {name} report (text)", "path": str(report_text)})
        if report_csv:
            artifacts.append({"label": f"VTune {name} report (csv)", "path": str(report_csv)})

    payload: Dict[str, object] = {
        "generated_at": utc_now_iso(),
        "analysis": "hotspots",
        "result_dir": primary.get("result_dir"),
        "report_path": primary.get("report_text"),
        "csv_path": primary.get("report_csv"),
        "top_hotspots": primary.get("top_hotspots", []),
        "hotspots": primary.get("hotspots", []),
        "raw_text": primary.get("raw_text", ""),
        "analyses": analyses,
        "analysis_metrics": {
            str(entry.get("name") or f"analysis_{i}"): entry.get("summary_metrics", {})
            for i, entry in enumerate(analyses)
        },
        "artifacts": artifacts,
    }

    out_path = Path(out_dir) / "vtune_summary.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
