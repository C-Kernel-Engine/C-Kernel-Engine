#!/usr/bin/env python3
"""
Generate advisor_summary.json for v7 IR visualizer.

Primary goal: keep a stable, portable schema with artifact paths and lightweight
summary metrics extracted from Advisor roofline reports when available.

Enhanced: also parses advisor XML .advisum files (metrics, vectorization,
threading) which always contain structured data even when the text/csv
report exports are HTML-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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


# ── XML .advisum parsers ──────────────────────────────────────────────


def _find_advisum_files(project_dir: Path) -> Dict[str, Path]:
    """Walk the advisor project dir and find .advisum XML summary files."""
    found: Dict[str, Path] = {}
    if not project_dir.is_dir():
        return found
    for p in project_dir.rglob("*.advisum"):
        stem = p.stem.lower()  # e.g. "metrics", "vectorization", "threading"
        if stem not in found:
            found[stem] = p
    return found


def _parse_float_attr(elem: ET.Element, attr: str) -> Optional[float]:
    """Extract a float from an attribute like double:bandwidth='28.98'."""
    for key, val in elem.attrib.items():
        if key.split("}")[-1] == attr or key == attr:
            try:
                return float(val)
            except ValueError:
                pass
    # Also try namespace-prefixed keys like "double:bandwidth"
    for key, val in elem.attrib.items():
        if ":" in key and key.split(":", 1)[1] == attr:
            try:
                return float(val)
            except ValueError:
                pass
    return None


def parse_metrics_advisum(path: Path) -> Dict[str, Any]:
    """Parse metrics.advisum → roofline ceilings + global metrics."""
    result: Dict[str, Any] = {}
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
    except Exception:
        return result

    # Roof items (bandwidth ceilings)
    roof_items: List[Dict[str, Any]] = []
    for ri in root.iter("roofItem"):
        name = ri.get("name", "")
        bw = _parse_float_attr(ri, "bandwidth")
        types_el = ri.find("types")
        is_compute = False
        is_memory = False
        is_mt = False
        is_st = False
        if types_el is not None:
            for k, v in types_el.attrib.items():
                key = k.split("}")[-1] if "}" in k else k.split(":", 1)[-1] if ":" in k else k
                if key == "compute" and v == "true":
                    is_compute = True
                if key == "memory" and v == "true":
                    is_memory = True
                if key == "multiThreaded" and v == "true":
                    is_mt = True
                if key == "singleThreaded" and v == "true":
                    is_st = True
        if name and bw is not None:
            roof_items.append({
                "name": name,
                "bandwidth_gops": round(bw, 3),
                "kind": "compute" if is_compute else ("memory" if is_memory else "unknown"),
                "threading": "multi" if is_mt else ("single" if is_st else "unknown"),
            })
    if roof_items:
        result["roof_items"] = roof_items

    # Global metrics from <Metrics> element
    for metrics_el in root.iter("Metrics"):
        global_metrics: Dict[str, Any] = {}
        for k, v in metrics_el.attrib.items():
            # Strip namespace prefix: "double:ProgramTime" → "ProgramTime"
            clean_key = k
            if "}" in k:
                clean_key = k.split("}")[-1]
            elif ":" in k:
                clean_key = k.split(":", 1)[-1]
            try:
                global_metrics[clean_key] = float(v)
            except ValueError:
                global_metrics[clean_key] = v
        if global_metrics:
            result["global_metrics"] = global_metrics
        break  # only first

    # Memory traffic
    for level in root.iter("level"):
        for k, v in level.attrib.items():
            clean = k.split("}")[-1] if "}" in k else k.split(":", 1)[-1] if ":" in k else k
            try:
                result.setdefault("memory_traffic_gb", {})[clean] = round(float(v), 6)
            except ValueError:
                pass

    return result


def parse_vectorization_advisum(path: Path) -> List[Dict[str, Any]]:
    """Parse vectorization.advisum → per-function hotspot list."""
    hotspots: List[Dict[str, Any]] = []
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
    except Exception:
        return hotspots

    for hs in root.iter("hotspot"):
        routine = hs.get("routine", "")
        if not routine:
            continue
        is_vec = hs.get("is_vectorized", "0") == "1"
        self_time = 0.0
        total_time = 0.0
        try:
            self_time = float(hs.get("self_time", "0"))
        except ValueError:
            pass
        try:
            total_time = float(hs.get("total_time", "0"))
        except ValueError:
            pass
        hotspots.append({
            "routine": routine,
            "is_vectorized": is_vec,
            "self_time_s": round(self_time, 6),
            "total_time_s": round(total_time, 6),
        })
    return hotspots


def parse_threading_advisum(path: Path) -> List[Dict[str, Any]]:
    """Parse threading.advisum → per-function threading hotspot list."""
    return parse_vectorization_advisum(path)  # same XML schema


def enrich_from_advisum(project_dir: Path) -> Dict[str, Any]:
    """Walk advisor project, parse .advisum XML files, return enrichment dict."""
    enrichment: Dict[str, Any] = {}
    advisum_files = _find_advisum_files(project_dir)

    if "metrics" in advisum_files:
        enrichment["advisum_metrics"] = parse_metrics_advisum(advisum_files["metrics"])

    if "vectorization" in advisum_files:
        enrichment["vectorization_hotspots"] = parse_vectorization_advisum(
            advisum_files["vectorization"]
        )

    if "threading" in advisum_files:
        enrichment["threading_hotspots"] = parse_threading_advisum(
            advisum_files["threading"]
        )

    # Build a flat summary_metrics dict from the advisum global metrics
    # This is what the IR visualizer renders in the Advisor panel
    gm = enrichment.get("advisum_metrics", {}).get("global_metrics", {})
    flat: Dict[str, float] = {}
    key_map = {
        "ProgramTime": "elapsed_time_s",
        "ElapsedTime": "elapsed_time_s",
        "TotalGFLOPS": "total_gflops",
        "TotalGFLOPCount": "total_gflop_count",
        "TotalGINTOPS": "total_gintops",
        "TotalGINTOPCount": "total_gintop_count",
        "TotalGMixedOPS": "total_gmixed_ops",
        "TotalGMixedOPCount": "total_gmixed_op_count",
        "TotalFloatAI": "float_arithmetic_intensity",
        "TotalIntAI": "int_arithmetic_intensity",
        "TotalMixedAI": "mixed_arithmetic_intensity",
        "TotalCPUTime": "total_cpu_time_s",
        "TimeInVectorizedLoops": "time_in_vectorized_loops_s",
        "TimeInScalarLoops": "time_in_scalar_loops_s",
        "TimeOutsideOfAnyLoop": "time_outside_loops_s",
        "VectorizedLoopsCount": "vectorized_loops_count",
        "CPUThreads": "cpu_threads",
    }
    for gm_key, flat_key in key_map.items():
        if gm_key in gm:
            try:
                flat[flat_key] = round(float(gm[gm_key]), 6)
            except (ValueError, TypeError):
                pass

    # Add key roofline ceilings
    roof = enrichment.get("advisum_metrics", {}).get("roof_items", [])
    for item in roof:
        name = item.get("name", "")
        bw = item.get("bandwidth_gops")
        if not name or bw is None:
            continue
        if name == "DRAM Bandwidth":
            flat["dram_bw_gb_s"] = round(bw, 3)
        elif name == "SP Vector FMA Peak":
            flat["sp_fma_peak_gflops"] = round(bw, 3)
        elif name == "DP Vector FMA Peak":
            flat["dp_fma_peak_gflops"] = round(bw, 3)
        elif name == "L1 Bandwidth":
            flat["l1_bw_gb_s"] = round(bw, 3)
        elif name == "L2 Bandwidth":
            flat["l2_bw_gb_s"] = round(bw, 3)
        elif name == "L3 Bandwidth":
            flat["l3_bw_gb_s"] = round(bw, 3)

    # Add ISA used
    isa = gm.get("ISAUsed", "")
    if isa:
        flat["isa_used"] = str(isa)  # type: ignore[assignment]

    enrichment["flat_summary_metrics"] = flat
    return enrichment


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

    # Try text-based metrics first
    text_metrics = parse_report_metrics(raw_text)

    # Always try to enrich from .advisum XML in the project dir
    advisum_enrichment: Dict[str, Any] = {}
    if args.project_dir and args.project_dir.is_dir():
        advisum_enrichment = enrich_from_advisum(args.project_dir)
        print(f"  advisum enrichment: {len(advisum_enrichment)} sections")

    # Merge: prefer text metrics if available, fall back to advisum flat metrics
    summary_metrics = text_metrics
    flat_from_advisum = advisum_enrichment.get("flat_summary_metrics", {})
    if not summary_metrics and flat_from_advisum:
        summary_metrics = flat_from_advisum
        print("  Using advisum XML metrics (text report had 0 parseable metrics)")
    elif flat_from_advisum:
        # Supplement: add advisum keys that text parsing missed
        for k, v in flat_from_advisum.items():
            if k not in summary_metrics:
                summary_metrics[k] = v

    payload: Dict[str, object] = {
        "generated_at": utc_now_iso(),
        "analysis": str(args.analysis or "roofline"),
        "project_dir": str(args.project_dir),
        "project_path": str(args.project_dir),
        "report_path": str(report_text) if report_text else None,
        "csv_path": str(report_csv) if report_csv else None,
        "html_path": str(report_html) if report_html else None,
        "summary_metrics": summary_metrics,
        "preview_rows": parse_csv_preview(report_csv) if report_csv else [],
        "raw_text": raw_text,
        "artifacts": artifacts,
    }

    # Add advisum-specific sections when available
    if "advisum_metrics" in advisum_enrichment:
        payload["advisum_metrics"] = advisum_enrichment["advisum_metrics"]
    if "vectorization_hotspots" in advisum_enrichment:
        payload["vectorization_hotspots"] = advisum_enrichment["vectorization_hotspots"]
    if "threading_hotspots" in advisum_enrichment:
        payload["threading_hotspots"] = advisum_enrichment["threading_hotspots"]

    out_path = Path(out_dir) / "advisor_summary.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
