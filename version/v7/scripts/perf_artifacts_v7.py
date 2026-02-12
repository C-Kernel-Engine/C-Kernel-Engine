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


def parse_perf_stat_text(text: str) -> Dict[str, object]:
    counters: Dict[str, float] = {}
    notes: Dict[str, str] = {}
    elapsed_sec: Optional[float] = None
    csv = []

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
            continue

        raw_value, metric, note = m.groups()
        value = parse_perf_number(raw_value)
        if value is None:
            continue

        counters[metric] = value
        if note:
            notes[metric] = note.strip()
        csv.append({"metric": metric, "value": value, "note": note.strip() if note else ""})

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
        "rows": csv,
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
        return Path(info["path"]) / ".ck_build"
    if input_type == "local_config":
        return Path(info["path"]).parent / ".ck_build"
    return None


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate v7 perf artifact JSON files")
    parser.add_argument("--model-dir", type=Path, help="Model output directory")
    parser.add_argument("--model-input", type=str, help="Model input string (same as ck_run_v7.py)")
    parser.add_argument("--out-dir", type=Path, help="Output directory for JSON artifacts")
    parser.add_argument("--perf-stat", type=Path, help="perf stat text output file")
    parser.add_argument("--perf-data", type=Path, help="perf.data file used for flamegraph")
    parser.add_argument("--folded", type=Path, help="Folded stack file from stackcollapse-perf.pl")
    parser.add_argument("--flamegraph-svg", type=Path, help="Generated flamegraph SVG")
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
        manifest = {
            "generated_at": utc_now_iso(),
            "svg_path": str(args.flamegraph_svg),
            "perf_data_path": str(args.perf_data) if args.perf_data else None,
            "folded_path": str(args.folded) if args.folded else None,
            "top_symbols": top_symbols,
        }
        write_json(out_dir / "flamegraph_manifest.json", manifest)
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
