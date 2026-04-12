#!/usr/bin/env python3
"""
Generate cachegrind_summary.json for the v8 IR visualizer.
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


def detect_model_dir_from_input(model_input: str) -> Optional[Path]:
    try:
        from ck_run_v8 import CACHE_DIR, detect_input_type  # type: ignore
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


def parse_int_metric(text: str) -> Optional[int]:
    raw = (text or "").strip().replace(",", "")
    if not raw:
        return None
    if not re.fullmatch(r"[-+]?\d+", raw):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def parse_annotate_totals(text: str) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    if not text:
        return totals

    keys = ("Ir", "Dr", "Dw", "D1mr", "D1mw", "LLmr", "LLmw")
    for line in text.splitlines():
        if "PROGRAM TOTALS" not in line:
            continue
        cleaned = re.sub(r"\([^)]*\)", "", line)
        nums = re.findall(r"[0-9][0-9,]*", cleaned)
        if len(nums) < 7:
            continue
        for key, raw in zip(keys, nums[:7]):
            val = parse_int_metric(raw)
            if val is not None:
                totals[key] = val
        if totals:
            break
    return totals


def parse_top_functions(text: str, top_k: int = 20) -> List[Dict[str, object]]:
    if not text:
        return []

    rows: List[Tuple[int, Dict[str, object]]] = []
    in_file_function_section = False
    for line in text.splitlines():
        if "-- File:function summary" in line:
            in_file_function_section = True
            continue
        if "-- Function:file summary" in line:
            break
        if not in_file_function_section:
            continue
        stripped = line.lstrip()
        if "PROGRAM TOTALS" in line or stripped.startswith(("#", "-", "I", "Events")):
            continue
        cleaned = re.sub(r"\([^)]*\)", "", line)
        cleaned = cleaned.replace("<", " ").replace(">", " ")
        tokens = cleaned.strip().split()
        if len(tokens) < 8:
            continue
        nums: List[int] = []
        num_end = -1
        for idx, tok in enumerate(tokens):
            if re.fullmatch(r"[0-9][0-9,]*", tok):
                val = parse_int_metric(tok)
                if val is not None:
                    nums.append(val)
                    if len(nums) == 7:
                        num_end = idx
                        break
        if len(nums) < 7 or num_end < 0:
            continue
        name = " ".join(tokens[num_end + 1 :]).strip()
        if not name:
            continue
        lower_name = name.lower()
        if lower_name.startswith("annotated:") or lower_name.startswith("unannotated:"):
            continue
        row = {
            "function": name,
            "Ir": int(nums[0]),
            "Dr": int(nums[1]),
            "Dw": int(nums[2]),
            "D1mr": int(nums[3]),
            "D1mw": int(nums[4]),
            "LLmr": int(nums[5]),
            "LLmw": int(nums[6]),
        }
        rows.append((int(nums[0]), row))

    rows.sort(key=lambda kv: kv[0], reverse=True)
    return [row for _, row in rows[:top_k]]


def derive_metrics(totals: Dict[str, int]) -> Dict[str, float]:
    ir = float(totals.get("Ir", 0))
    dr = float(totals.get("Dr", 0))
    dw = float(totals.get("Dw", 0))
    d1mr = float(totals.get("D1mr", 0))
    d1mw = float(totals.get("D1mw", 0))
    llmr = float(totals.get("LLmr", 0))
    llmw = float(totals.get("LLmw", 0))
    data_refs = dr + dw
    d1_misses = d1mr + d1mw
    ll_misses = llmr + llmw

    out: Dict[str, float] = {}
    if data_refs > 0:
        out["d1_miss_rate"] = d1_misses / data_refs
        out["ll_miss_rate"] = ll_misses / data_refs
        out["ll_miss_given_d1_miss"] = (ll_misses / d1_misses) if d1_misses > 0 else 0.0
    if ir > 0:
        out["data_refs_per_ir"] = data_refs / ir
    return out


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate cachegrind_summary.json for v8")
    parser.add_argument("--model-dir", type=Path, help="Model output directory")
    parser.add_argument("--model-input", type=str, help="Model input string (same as ck_run_v8.py)")
    parser.add_argument("--out-dir", type=Path, help="Output directory (default: model dir)")
    parser.add_argument("--cachegrind-out", type=Path, help="Cachegrind raw output file")
    parser.add_argument("--annotate", type=Path, help="cg_annotate text output")
    parser.add_argument("--top-k", type=int, default=20, help="Top functions to keep")
    args = parser.parse_args()

    model_dir = args.model_dir
    if model_dir is None and args.model_input:
        model_dir = detect_model_dir_from_input(args.model_input)
    out_dir = args.out_dir or model_dir
    if out_dir is None:
        parser.error("Provide --out-dir or one of --model-dir/--model-input")

    annotate_text = ""
    if args.annotate and args.annotate.exists():
        annotate_text = args.annotate.read_text(errors="ignore")

    totals = parse_annotate_totals(annotate_text)
    top_functions = parse_top_functions(annotate_text, top_k=max(1, int(args.top_k or 20)))
    derived = derive_metrics(totals)

    artifacts = []
    if args.cachegrind_out and args.cachegrind_out.exists():
        artifacts.append({"label": "cachegrind.out", "path": str(args.cachegrind_out)})
    if args.annotate and args.annotate.exists():
        artifacts.append({"label": "cg_annotate", "path": str(args.annotate)})

    payload: Dict[str, object] = {
        "generated_at": utc_now_iso(),
        "cachegrind_out": str(args.cachegrind_out) if args.cachegrind_out else None,
        "annotate_path": str(args.annotate) if args.annotate else None,
        "totals": totals,
        "derived": derived,
        "top_functions": top_functions,
        "artifacts": artifacts,
    }

    out_path = Path(out_dir) / "cachegrind_summary.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
