#!/usr/bin/env python3
"""
Generate normalized v7 profiling artifacts for IR visualizer.

Outputs (when source files are available):
  - perf_stat_summary.json
  - flamegraph_manifest.json
  - vtune_summary.json (passthrough if provided)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_perf_number(value: str) -> Optional[float]:
    raw = value.strip().replace(",", "")
    if not raw or raw == "<not":
        return None
    # Handle "<not supported>" and "<not counted>"
    if raw.startswith("<not"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_perf_stat_csv_line(line: str) -> Optional[Tuple[str, float, str]]:
    """
    Parse one line from `perf stat -x,` output.

    Expected format (kernel/version dependent):
      value,unit,event,run_time,pcnt_running[,metric_value,metric_unit]
    """
    try:
        fields = next(csv.reader([line], delimiter=","))
    except Exception:
        return None
    if len(fields) < 3:
        return None

    raw_value = (fields[0] or "").strip()
    metric = (fields[2] or "").strip()
    if not raw_value or not metric:
        return None

    value = parse_perf_number(raw_value)
    if value is None:
        return None

    note = ""
    if len(fields) >= 6:
        metric_value = (fields[5] or "").strip()
        metric_unit = (fields[6] or "").strip() if len(fields) >= 7 else ""
        if metric_value:
            note = f"metric={metric_value}{(' ' + metric_unit) if metric_unit else ''}"

    return metric, value, note


def parse_perf_stat_text(text: str) -> Dict[str, object]:
    counters: Dict[str, float] = {}
    notes: Dict[str, str] = {}
    elapsed_sec: Optional[float] = None
    rows = []

    # Typical line:
    #  1,234,567      cycles                    # 1.23 GHz
    line_re = re.compile(r"^\s*([<\w\.,-]+)\s+([A-Za-z0-9_\-\.\/:]+)(?:\s+(?:#\s*)?(.*))?$")
    elapsed_re = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+seconds\s+time\s+elapsed")

    for line in text.splitlines():
        m_elapsed = elapsed_re.search(line)
        if m_elapsed:
            try:
                elapsed_sec = float(m_elapsed.group(1))
            except ValueError:
                pass
            continue

        m = line_re.match(line)
        if not m:
            parsed_csv = parse_perf_stat_csv_line(line)
            if parsed_csv is None:
                continue
            metric, value, note = parsed_csv
        else:
            raw_value, metric, note = m.groups()
            value = parse_perf_number(raw_value)
            if value is None:
                parsed_csv = parse_perf_stat_csv_line(line)
                if parsed_csv is None:
                    continue
                metric, value, note = parsed_csv
            else:
                note = note.strip() if note else ""

        counters[metric] = float(value)
        if note:
            notes[metric] = note
        rows.append({"metric": metric, "value": float(value), "note": note})

    def _sum_matching(patterns: List[str]) -> Optional[float]:
        total = 0.0
        found = False
        for metric, value in counters.items():
            metric_l = metric.lower()
            for pat in patterns:
                if re.search(pat, metric_l):
                    total += float(value)
                    found = True
                    break
        return total if found else None

    inst = counters.get("instructions")
    cyc = counters.get("cycles")
    cache_ref = counters.get("cache-references")
    cache_miss = counters.get("cache-misses")
    branches = counters.get("branches")
    branch_miss = counters.get("branch-misses")

    if inst is None:
        inst = _sum_matching([r"(^|/)instructions/?$", r"\binstructions\b"])
    if cyc is None:
        cyc = _sum_matching([r"(^|/)cycles/?$", r"\bcycles\b"])
    if cache_ref is None:
        cache_ref = _sum_matching([r"cache-references"])
    if cache_miss is None:
        cache_miss = _sum_matching([r"cache-misses"])
    if branches is None:
        branches = _sum_matching([r"(^|/)branches/?$", r"\bbranches\b"])
    if branch_miss is None:
        branch_miss = _sum_matching([r"branch-misses"])

    if inst is not None:
        counters["instructions"] = float(inst)
    if cyc is not None:
        counters["cycles"] = float(cyc)
    if cache_ref is not None:
        counters["cache-references"] = float(cache_ref)
    if cache_miss is not None:
        counters["cache-misses"] = float(cache_miss)
    if branches is not None:
        counters["branches"] = float(branches)
    if branch_miss is not None:
        counters["branch-misses"] = float(branch_miss)

    derived: Dict[str, float] = {}

    if inst is not None and cyc and cyc > 0:
        derived["ipc"] = inst / cyc
    if cache_ref and cache_ref > 0 and cache_miss is not None:
        derived["cache_miss_rate"] = cache_miss / cache_ref
    if branches and branches > 0 and branch_miss is not None:
        derived["branch_miss_rate"] = branch_miss / branches

    return {
        "generated_at": utc_now_iso(),
        "counters": counters,
        "derived": derived,
        "elapsed_seconds": elapsed_sec,
        "notes": notes,
        "rows": rows,
    }


def parse_folded_top_symbols(folded_path: Path, top_k: int = 25) -> List[Dict[str, object]]:
    samples: Dict[str, int] = {}
    if not folded_path.exists():
        return []

    for line in folded_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            stack, count_s = line.rsplit(" ", 1)
            count = int(count_s)
        except ValueError:
            continue
        if ";" in stack:
            symbol = stack.split(";")[-1]
        else:
            symbol = stack
        symbol = symbol.strip()
        if not symbol:
            continue
        samples[symbol] = samples.get(symbol, 0) + count

    ranked = sorted(samples.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [{"symbol": sym, "samples": cnt} for sym, cnt in ranked]


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


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def read_json_dict(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(errors="ignore"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate v7 perf artifact JSON files")
    parser.add_argument("--model-dir", type=Path, help="Model output directory")
    parser.add_argument("--model-input", type=str, help="Model input string (same as ck_run_v7.py)")
    parser.add_argument("--out-dir", type=Path, help="Output directory for JSON artifacts")
    parser.add_argument("--perf-stat", type=Path, help="perf stat text output file")
    parser.add_argument("--perf-data", type=Path, help="perf.data file used for flamegraph")
    parser.add_argument("--folded", type=Path, help="Folded stack file from stackcollapse-perf.pl")
    parser.add_argument("--flamegraph-svg", type=Path, help="Generated flamegraph SVG")
    parser.add_argument("--mode", choices=["decode", "prefill"], help="Flamegraph capture mode")
    parser.add_argument("--vtune-summary", type=Path, help="Optional VTune summary JSON/text")
    args = parser.parse_args()

    model_dir = args.model_dir
    if model_dir is None and args.model_input:
        model_dir = detect_model_dir_from_input(args.model_input)

    out_dir = args.out_dir or model_dir
    if out_dir is None:
        parser.error("Provide --out-dir or one of --model-dir/--model-input")

    wrote_any = False

    if args.perf_stat and args.perf_stat.exists():
        perf_summary = parse_perf_stat_text(args.perf_stat.read_text(errors="ignore"))
        perf_summary["source"] = str(args.perf_stat)
        write_json(out_dir / "perf_stat_summary.json", perf_summary)
        print(f"Wrote {out_dir / 'perf_stat_summary.json'}")
        wrote_any = True

    if args.flamegraph_svg and args.flamegraph_svg.exists():
        top_symbols = parse_folded_top_symbols(args.folded) if args.folded else []
        mode_key = args.mode or "decode"
        entry = {
            "generated_at": utc_now_iso(),
            "mode": mode_key,
            "svg_path": str(args.flamegraph_svg),
            "perf_data_path": str(args.perf_data) if args.perf_data else None,
            "folded_path": str(args.folded) if args.folded else None,
            "top_symbols": top_symbols,
        }
        manifest_path = out_dir / "flamegraph_manifest.json"
        prev = read_json_dict(manifest_path)
        prev_by_mode = prev.get("by_mode")
        by_mode: Dict[str, Dict[str, object]] = {}
        if isinstance(prev_by_mode, dict):
            for k, v in prev_by_mode.items():
                if isinstance(k, str) and isinstance(v, dict):
                    by_mode[k] = dict(v)
        by_mode[mode_key] = entry

        preferred_mode = "decode" if "decode" in by_mode else ("prefill" if "prefill" in by_mode else mode_key)
        preferred = by_mode.get(preferred_mode, entry)

        manifest = {
            "generated_at": utc_now_iso(),
            "mode": preferred_mode,
            "available_modes": sorted(by_mode.keys()),
            "svg_path": preferred.get("svg_path"),
            "perf_data_path": preferred.get("perf_data_path"),
            "folded_path": preferred.get("folded_path"),
            "top_symbols": preferred.get("top_symbols", []),
            "by_mode": by_mode,
        }
        manifest_source = prev.get("source")
        if isinstance(manifest_source, str) and manifest_source:
            manifest["source"] = manifest_source
        write_json(manifest_path, manifest)
        print(f"Wrote {out_dir / 'flamegraph_manifest.json'}")
        wrote_any = True

    if args.vtune_summary and args.vtune_summary.exists():
        vtune_payload: Dict[str, object]
        try:
            vtune_payload = json.loads(args.vtune_summary.read_text())
        except Exception:
            vtune_payload = {
                "generated_at": utc_now_iso(),
                "source": str(args.vtune_summary),
                "raw_text": args.vtune_summary.read_text(errors="ignore"),
            }
        if "generated_at" not in vtune_payload:
            vtune_payload["generated_at"] = utc_now_iso()
        vtune_payload["source"] = str(args.vtune_summary)
        write_json(out_dir / "vtune_summary.json", vtune_payload)
        print(f"Wrote {out_dir / 'vtune_summary.json'}")
        wrote_any = True

    if not wrote_any:
        print("No artifacts generated (missing input files).")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
