#!/usr/bin/env python3
"""Check whether Gemma q/k/v parity deltas are due to layout mismatch.

Usage:
  python version/v6.6/scripts/check_qkv_layout_match.py \
    --ck-dump ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps/dump.bin \
    --ref-dump ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/llama_parity_dumps/dump.bin
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


def load_parity_module(parity_test_path: Path):
    spec = importlib.util.spec_from_file_location("parity_test_mod", str(parity_test_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {parity_test_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metric(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    n = min(a.size, b.size)
    if n == 0:
        return float("inf"), float("inf")
    d = np.abs(a.reshape(-1)[:n] - b.reshape(-1)[:n])
    return float(d.max()), float(d.mean())


def iter_tensors(dumps: Iterable, layer: int, token: int, op_name: str):
    for d in dumps:
        if d.layer_id == layer and d.token_id == token and d.op_name == op_name:
            yield d


def choose_ref_tensor(ref_dumps: list, layer: int, token: int, base_name: str, target_size: int):
    op = f"{base_name}-{layer}"
    cands = list(iter_tensors(ref_dumps, layer, token, op))
    if not cands:
        return None

    # Prefer exact-size candidates, then prefer 2D tensors for mapping checks.
    exact = [d for d in cands if int(d.data.size) == target_size]
    pool = exact if exact else cands
    pool.sort(key=lambda d: (0 if d.data.ndim == 2 else 1, -int(d.data.size)))
    return pool[0]


def analyze_pair(ck_vec: np.ndarray, ref_arr: np.ndarray):
    results = []

    raw_max, raw_mean = metric(ref_arr.flatten(), ck_vec)
    results.append(("raw_flatten", raw_max, raw_mean))

    if ref_arr.ndim == 2:
        t_max, t_mean = metric(ref_arr.T.flatten(), ck_vec)
        results.append(("transpose_flatten", t_max, t_mean))

        rows, cols = ref_arr.shape
        if cols == 2 and ck_vec.size >= rows * 2:
            c_even = ck_vec[0::2][:rows]
            c_odd = ck_vec[1::2][:rows]
            e0_max, e0_mean = metric(c_even, ref_arr[:, 0])
            e1_max, e1_mean = metric(c_odd, ref_arr[:, 1])
            es0_max, es0_mean = metric(c_even, ref_arr[:, 1])
            es1_max, es1_mean = metric(c_odd, ref_arr[:, 0])
            results.append(("ck_even/odd->ref_col0/1", max(e0_max, e1_max), (e0_mean + e1_mean) / 2.0))
            results.append(("ck_even/odd->ref_col1/0", max(es0_max, es1_max), (es0_mean + es1_mean) / 2.0))

            c0 = ck_vec[:rows]
            c1 = ck_vec[rows: rows * 2]
            s0_max, s0_mean = metric(c0, ref_arr[:, 0])
            s1_max, s1_mean = metric(c1, ref_arr[:, 1])
            ss0_max, ss0_mean = metric(c0, ref_arr[:, 1])
            ss1_max, ss1_mean = metric(c1, ref_arr[:, 0])
            results.append(("ck_split->ref_col0/1", max(s0_max, s1_max), (s0_mean + s1_mean) / 2.0))
            results.append(("ck_split->ref_col1/0", max(ss0_max, ss1_max), (ss0_mean + ss1_mean) / 2.0))

    results.sort(key=lambda x: x[2])
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Check q/k/v layout alignment between CK and llama dumps")
    default_base = Path.home() / ".cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build"
    ap.add_argument("--ck-dump", type=Path, default=default_base / "ck_parity_dumps/dump.bin")
    ap.add_argument("--ref-dump", type=Path, default=default_base / "llama_parity_dumps/dump.bin")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--token", type=int, default=0)
    ap.add_argument("--parity-test", type=Path, default=Path("version/v6.6/scripts/parity_test.py"))
    args = ap.parse_args()

    if not args.ck_dump.exists():
        print(f"[ERROR] CK dump not found: {args.ck_dump}")
        return 1
    if not args.ref_dump.exists():
        print(f"[ERROR] Ref dump not found: {args.ref_dump}")
        return 1
    if not args.parity_test.exists():
        print(f"[ERROR] parity_test.py not found: {args.parity_test}")
        return 1

    pt = load_parity_module(args.parity_test)
    ck_dumps = pt.read_dump_file(args.ck_dump)
    ref_dumps = pt.read_dump_file(args.ref_dump)

    mapping = [
        ("q_proj", "Qcur"),
        ("k_proj", "Kcur"),
        ("v_proj", "Vcur"),
    ]

    print(f"Layer={args.layer}, Token={args.token}")
    print(f"CK dumps={len(ck_dumps)}  Ref dumps={len(ref_dumps)}")
    print("-" * 96)

    for ck_name, ref_base in mapping:
        ck_cands = list(iter_tensors(ck_dumps, args.layer, args.token, ck_name))
        if not ck_cands:
            print(f"{ck_name:8}  [MISSING] CK tensor not found")
            continue
        ck = ck_cands[0].data.flatten()
        ref_t = choose_ref_tensor(ref_dumps, args.layer, args.token, ref_base, ck.size)
        if ref_t is None:
            print(f"{ck_name:8}  [MISSING] Ref tensor {ref_base}-{args.layer} not found")
            continue

        ref = ref_t.data
        print(f"{ck_name:8}  CK size={ck.size:<6}  Ref op={ref_t.op_name:<10} shape={tuple(ref.shape)}")
        results = analyze_pair(ck, ref)
        raw_mean = next((x[2] for x in results if x[0] == "raw_flatten"), None)
        best = results[0]
        for name, mx, mn in results:
            print(f"    {name:<28} max={mx:9.3e}  mean={mn:9.3e}")
        if raw_mean is not None and best[0] != "raw_flatten" and best[2] < raw_mean * 0.6:
            print("    -> likely layout mismatch (non-raw mapping is much better)")
        elif best[0] == "raw_flatten":
            print("    -> raw flatten is best (less likely to be pure layout issue)")
        else:
            print("    -> ambiguous: alternate mapping helps but not decisive")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
