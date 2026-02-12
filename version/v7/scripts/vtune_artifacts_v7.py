#!/usr/bin/env python3
"""
Generate vtune_summary.json for IR visualizer.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    args = parser.parse_args()

    model_dir = args.model_dir
    if model_dir is None and args.model_input:
        model_dir = detect_model_dir_from_input(args.model_input)
    out_dir = args.out_dir or model_dir
    if out_dir is None:
        parser.error("Provide --out-dir or one of --model-dir/--model-input")

    hotspots = parse_hotspots_csv(args.report_csv) if args.report_csv else []
    raw_text = ""
    if args.report_text and args.report_text.exists():
        raw_text = args.report_text.read_text(errors="ignore")
        # Keep payload reasonably small for embedding.
        if len(raw_text) > 120_000:
            raw_text = raw_text[:120_000] + "\n...[truncated]..."

    payload: Dict[str, object] = {
        "generated_at": utc_now_iso(),
        "analysis": "hotspots",
        "result_dir": str(args.result_dir),
        "report_path": str(args.report_text) if args.report_text else None,
        "csv_path": str(args.report_csv) if args.report_csv else None,
        "top_hotspots": hotspots,
        "hotspots": hotspots,
        "raw_text": raw_text,
        "artifacts": [
            {"label": "VTune Result Directory", "path": str(args.result_dir)},
        ],
    }
    if args.report_text:
        payload["artifacts"].append({"label": "VTune Hotspots (text)", "path": str(args.report_text)})
    if args.report_csv:
        payload["artifacts"].append({"label": "VTune Hotspots (csv)", "path": str(args.report_csv)})

    out_path = Path(out_dir) / "vtune_summary.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
