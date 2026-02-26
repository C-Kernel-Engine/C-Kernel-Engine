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
    dtlb_loads = counters.get("dTLB-loads")
    dtlb_load_miss = counters.get("dTLB-load-misses")
    dtlb_stores = counters.get("dTLB-stores")
    dtlb_store_miss = counters.get("dTLB-store-misses")
    itlb_load_miss = counters.get("iTLB-load-misses")
    dtlb_load_walk_completed = counters.get("dtlb_load_misses.walk_completed")
    dtlb_store_walk_completed = counters.get("dtlb_store_misses.walk_completed")
    itlb_walk_completed = counters.get("itlb_misses.walk_completed")
    dtlb_load_stlb_hit = counters.get("dtlb_load_misses.stlb_hit")
    itlb_stlb_hit = counters.get("itlb_misses.stlb_hit")
    minor_faults = counters.get("minor-faults")
    major_faults = counters.get("major-faults")

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
    if dtlb_loads is None:
        dtlb_loads = _sum_matching([r"dtlb-loads"])
    if dtlb_load_miss is None:
        dtlb_load_miss = _sum_matching([r"dtlb-load-misses"])
    if dtlb_stores is None:
        dtlb_stores = _sum_matching([r"dtlb-stores"])
    if dtlb_store_miss is None:
        dtlb_store_miss = _sum_matching([r"dtlb-store-misses"])
    if itlb_load_miss is None:
        itlb_load_miss = _sum_matching([r"itlb-load-misses"])
    if dtlb_load_walk_completed is None:
        dtlb_load_walk_completed = _sum_matching([r"dtlb_load_misses\.walk_completed"])
    if dtlb_store_walk_completed is None:
        dtlb_store_walk_completed = _sum_matching([r"dtlb_store_misses\.walk_completed"])
    if itlb_walk_completed is None:
        itlb_walk_completed = _sum_matching([r"itlb_misses\.walk_completed"])
    if dtlb_load_stlb_hit is None:
        dtlb_load_stlb_hit = _sum_matching([r"dtlb_load_misses\.stlb_hit"])
    if itlb_stlb_hit is None:
        itlb_stlb_hit = _sum_matching([r"itlb_misses\.stlb_hit"])
    if minor_faults is None:
        minor_faults = _sum_matching([r"minor-faults"])
    if major_faults is None:
        major_faults = _sum_matching([r"major-faults"])

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
    if dtlb_loads is not None:
        counters["dTLB-loads"] = float(dtlb_loads)
    if dtlb_load_miss is not None:
        counters["dTLB-load-misses"] = float(dtlb_load_miss)
    if dtlb_stores is not None:
        counters["dTLB-stores"] = float(dtlb_stores)
    if dtlb_store_miss is not None:
        counters["dTLB-store-misses"] = float(dtlb_store_miss)
    if itlb_load_miss is not None:
        counters["iTLB-load-misses"] = float(itlb_load_miss)
    if dtlb_load_walk_completed is not None:
        counters["dtlb_load_misses.walk_completed"] = float(dtlb_load_walk_completed)
    if dtlb_store_walk_completed is not None:
        counters["dtlb_store_misses.walk_completed"] = float(dtlb_store_walk_completed)
    if itlb_walk_completed is not None:
        counters["itlb_misses.walk_completed"] = float(itlb_walk_completed)
    if dtlb_load_stlb_hit is not None:
        counters["dtlb_load_misses.stlb_hit"] = float(dtlb_load_stlb_hit)
    if itlb_stlb_hit is not None:
        counters["itlb_misses.stlb_hit"] = float(itlb_stlb_hit)
    if minor_faults is not None:
        counters["minor-faults"] = float(minor_faults)
    if major_faults is not None:
        counters["major-faults"] = float(major_faults)

    derived: Dict[str, float] = {}

    if inst is not None and cyc and cyc > 0:
        derived["ipc"] = inst / cyc
    if cache_ref and cache_ref > 0 and cache_miss is not None:
        derived["cache_miss_rate"] = cache_miss / cache_ref
    if branches and branches > 0 and branch_miss is not None:
        derived["branch_miss_rate"] = branch_miss / branches
    if dtlb_loads and dtlb_loads > 0 and dtlb_load_miss is not None:
        derived["dtlb_load_miss_rate"] = dtlb_load_miss / dtlb_loads
    if dtlb_stores and dtlb_stores > 0 and dtlb_store_miss is not None:
        derived["dtlb_store_miss_rate"] = dtlb_store_miss / dtlb_stores
    dtlb_accesses = 0.0
    dtlb_misses = 0.0
    if dtlb_loads is not None:
        dtlb_accesses += float(dtlb_loads)
    if dtlb_stores is not None:
        dtlb_accesses += float(dtlb_stores)
    if dtlb_load_miss is not None:
        dtlb_misses += float(dtlb_load_miss)
    if dtlb_store_miss is not None:
        dtlb_misses += float(dtlb_store_miss)
    if dtlb_accesses > 0:
        derived["dtlb_miss_rate"] = dtlb_misses / dtlb_accesses
    if itlb_load_miss is not None and inst and inst > 0:
        derived["itlb_misses_per_kinst"] = (float(itlb_load_miss) * 1000.0) / float(inst)
    page_walks = 0.0
    has_page_walks = False
    for val in (dtlb_load_walk_completed, dtlb_store_walk_completed, itlb_walk_completed):
        if val is not None:
            page_walks += float(val)
            has_page_walks = True
    if has_page_walks:
        derived["page_walks"] = page_walks
        if inst and inst > 0:
            derived["page_walks_per_kinst"] = (page_walks * 1000.0) / float(inst)
    stlb_hits = 0.0
    has_stlb_hits = False
    for val in (dtlb_load_stlb_hit, itlb_stlb_hit):
        if val is not None:
            stlb_hits += float(val)
            has_stlb_hits = True
    if has_stlb_hits:
        derived["stlb_hits"] = stlb_hits
        if inst and inst > 0:
            derived["stlb_hits_per_kinst"] = (stlb_hits * 1000.0) / float(inst)
    if minor_faults is not None and inst and inst > 0:
        derived["minor_faults_per_kinst"] = (float(minor_faults) * 1000.0) / float(inst)
    if major_faults is not None and inst and inst > 0:
        derived["major_faults_per_kinst"] = (float(major_faults) * 1000.0) / float(inst)

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


def collect_cpu_topology() -> Dict[str, object]:
    """Collect CPU topology, cache sizes, ISA flags, and compute peak estimates.

    Uses /proc/cpuinfo and /sys/devices/system/cpu/ as primary sources.
    Every step is wrapped in try/except so partial failures never break the caller.
    """
    result: Dict[str, object] = {"source": "sysfs+cpuinfo"}

    # ── 1. Parse /proc/cpuinfo ──────────────────────────────────────
    try:
        cpuinfo_text = Path("/proc/cpuinfo").read_text(errors="ignore")
        processors: List[Dict[str, str]] = []
        current: Dict[str, str] = {}
        for line in cpuinfo_text.splitlines():
            line = line.strip()
            if not line:
                if current:
                    processors.append(current)
                    current = {}
            elif ":" in line:
                k, _, v = line.partition(":")
                current[k.strip()] = v.strip()
        if current:
            processors.append(current)

        model_names: set = set()
        mhz_values: List[float] = []
        all_flags: set = set()
        physical_ids: set = set()
        core_ids_per_socket: Dict[str, set] = {}

        for proc in processors:
            pkeys = {k.lower(): v for k, v in proc.items()}
            if "model name" in pkeys:
                model_names.add(pkeys["model name"])
            if "cpu mhz" in pkeys:
                try:
                    mhz_values.append(float(pkeys["cpu mhz"]))
                except ValueError:
                    pass
            if "flags" in pkeys:
                all_flags.update(pkeys["flags"].split())
            phys_id = pkeys.get("physical id", "0")
            core_id = pkeys.get("core id", "0")
            physical_ids.add(phys_id)
            core_ids_per_socket.setdefault(phys_id, set()).add(core_id)

        if model_names:
            result["model_name"] = next(iter(model_names))
        result["num_sockets"]    = len(physical_ids) if physical_ids else 1
        result["physical_cores"] = sum(len(v) for v in core_ids_per_socket.values()) or len(processors)
        result["logical_cpus"]   = len(processors)
        result["has_avx"]        = "avx"          in all_flags
        result["has_avx2"]       = "avx2"         in all_flags
        result["has_avx512f"]    = "avx512f"      in all_flags
        result["has_fma"]        = "fma"          in all_flags
        result["has_avx512vnni"] = ("avx512_vnni" in all_flags) or ("avx512vnni" in all_flags)
        result["has_amx_bf16"]   = "amx_bf16"     in all_flags
        if mhz_values:
            result["current_mhz_avg"] = round(sum(mhz_values) / len(mhz_values), 1)
    except Exception:
        pass

    # ── 2. Max boost frequency from cpufreq ─────────────────────────
    try:
        max_freq_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        if max_freq_path.exists():
            result["max_freq_mhz"] = round(float(max_freq_path.read_text().strip()) / 1000.0, 1)
    except Exception:
        pass

    # ── 3. Cache sizes from sysfs ────────────────────────────────────
    try:
        cache_path = Path("/sys/devices/system/cpu/cpu0/cache")
        caches: Dict[str, object] = {}
        if cache_path.exists():
            for idx_dir in sorted(cache_path.iterdir()):
                level_f = idx_dir / "level"
                type_f  = idx_dir / "type"
                size_f  = idx_dir / "size"
                if not (level_f.exists() and type_f.exists() and size_f.exists()):
                    continue
                level    = level_f.read_text().strip()
                ctype    = type_f.read_text().strip().lower()   # "data", "instruction", "unified"
                size_raw = size_f.read_text().strip()
                m = re.match(r"(\d+)([KMG]?)", size_raw)
                if not m:
                    continue
                val  = int(m.group(1))
                unit = m.group(2)
                if unit == "K": val *= 1024
                elif unit == "M": val *= 1024 * 1024
                elif unit == "G": val *= 1024 * 1024 * 1024
                key = f"L{level}_{ctype}"
                if key not in caches:
                    caches[key] = {"size_bytes": val, "size_human": size_raw}
        if caches:
            result["caches"] = caches
    except Exception:
        pass

    # ── 4. P-core / E-core detection (Intel hybrid, Linux ≥5.15) ────
    try:
        cpu_base = Path("/sys/devices/system/cpu")
        p_core_count = e_core_count = 0
        seen_keys: set = set()
        for cpu_dir in sorted(cpu_base.glob("cpu[0-9]*")):
            ct_f = cpu_dir / "topology" / "core_type"
            ci_f = cpu_dir / "topology" / "core_id"
            if not ct_f.exists():
                continue
            ctype   = int(ct_f.read_text().strip())
            core_id = ci_f.read_text().strip() if ci_f.exists() else cpu_dir.name
            key = (ctype, core_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if ctype == 0:
                p_core_count += 1
            else:
                e_core_count += 1
        if seen_keys:
            result["hybrid_cpu"]   = e_core_count > 0
            result["p_core_count"] = p_core_count
            result["e_core_count"] = e_core_count
    except Exception:
        pass

    # ── 5. Compute theoretical FP32/FP64 peak GFLOP/s ───────────────
    try:
        phys     = int(result.get("physical_cores", 1))
        # E-cores don't run AVX2/FMA; use P-core count for peak
        if result.get("hybrid_cpu"):
            phys = int(result.get("p_core_count", phys))
        freq_mhz = float(result.get("max_freq_mhz") or result.get("current_mhz_avg", 3000.0))
        freq_ghz = freq_mhz / 1000.0
        has_fma  = bool(result.get("has_fma"))
        if result.get("has_avx512f") and has_fma:
            fp32_lanes, fp64_lanes = 16, 8
        elif (result.get("has_avx2") or result.get("has_avx")) and has_fma:
            fp32_lanes, fp64_lanes = 8, 4
        elif result.get("has_avx") or result.get("has_avx2"):
            fp32_lanes, fp64_lanes = 8, 4   # no FMA → 1 FLOP per lane per cycle
        else:
            fp32_lanes, fp64_lanes = 1, 1
        fma_mult = 2.0 if has_fma else 1.0
        result["peak_fp32_gflops"] = round(phys * freq_ghz * fp32_lanes * fma_mult, 1)
        result["peak_fp64_gflops"] = round(phys * freq_ghz * fp64_lanes * fma_mult, 1)
    except Exception:
        pass

    # ── 6. Transparent HugePages status ─────────────────────────────
    try:
        thp_path = Path("/sys/kernel/mm/transparent_hugepage/enabled")
        if thp_path.exists():
            content = thp_path.read_text().strip()
            m = re.search(r"\[(\w+)\]", content)
            result["thp_status"] = m.group(1) if m else "unknown"
    except Exception:
        pass

    # ── 7. Hugepage size from /proc/meminfo ──────────────────────────
    try:
        for line in Path("/proc/meminfo").read_text(errors="ignore").splitlines():
            if line.startswith("Hugepagesize:"):
                m = re.match(r"Hugepagesize:\s+(\d+)\s*kB", line)
                if m:
                    result["hugepage_size_kb"] = int(m.group(1))
                break
    except Exception:
        pass

    return result


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
        perf_summary["cpu_topology"] = collect_cpu_topology()
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
