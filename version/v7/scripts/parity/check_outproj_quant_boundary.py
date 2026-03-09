#!/usr/bin/env python3
"""
Check the attention -> out_proj -> ffn_norm boundary against fresh llama.cpp dumps.

This script is meant to answer a narrow question:

1. Is the decode attention output itself close to llama.cpp?
2. Does CK's out_proj kernel match a scalar/reference replay exactly?
3. Does CK's RMSNorm kernel match a scalar replay exactly?

If (2) and (3) pass while (1) drifts slightly versus llama.cpp, then the likely
issue is not a broken kernel but sensitivity at the quantized out_proj boundary.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from compare_first_token_logits import discover_ck_model_dir, discover_gguf, parse_tokens_csv
from check_sequence_hidden_state_vs_llama import (
    compare_arrays,
    load_json,
    load_model_lib,
    make_probe_specs,
    parse_raw_index,
    read_probe,
    resolve_raw_dump,
    run_decode_tokens_until,
    run_llama_raw_dumps,
)


QK_K = 256
BLOCK_Q8_K_SIZE = 4 + QK_K + 32


def _load_manifest_entries(model_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json(model_dir / "weights_manifest.json")
    entries = manifest.get("entries", [])
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name = entry.get("name")
        if isinstance(name, str):
            out[name] = entry
    return out


def _load_model_config(model_dir: Path) -> dict[str, Any]:
    return load_json(model_dir / "config.json")


def _read_weight_slice_f32(weights_path: Path, offset: int, count: int) -> np.ndarray:
    with weights_path.open("rb") as f:
        f.seek(int(offset))
        return np.fromfile(f, dtype=np.float32, count=int(count))


def _compare_named_probe(
    *,
    probes: dict[str, Any],
    dump_map: dict[int, dict[str, list[Any]]],
    lib: ctypes.CDLL,
    base_ptr: int,
    lowered: dict[str, Any],
    tokens_prefix: list[int],
    token_step: int,
    probe_name: str,
    atol: float,
    rtol: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    probe = probes[probe_name]
    raw_entry = resolve_raw_dump(dump_map[token_step], probe)
    if raw_entry is None:
        raise RuntimeError(f"missing llama raw dump for token_step={token_step} probe={probe_name}")
    ref = np.fromfile(raw_entry.path, dtype=np.float32)
    run_decode_tokens_until(lib, base_ptr, lowered, tokens_prefix, probe.stop_idx)
    got = read_probe(base_ptr, probe)
    cmp = compare_arrays(ref, got, atol=atol, rtol=rtol)
    return ref, got, cmp


def _scalar_rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    mean_sq = float(np.mean(x32.astype(np.float64) * x32.astype(np.float64)))
    rstd = 1.0 / np.sqrt(mean_sq + float(eps))
    return (x32 * np.float32(rstd)) * gamma.astype(np.float32)


def _configure_quant_lib(lib: ctypes.CDLL) -> None:
    lib.quantize_row_q8_k.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.quantize_row_q8_k.restype = None
    lib.gemv_q4_k_q8_k_ref.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemv_q4_k_q8_k_ref.restype = None


def _scalar_out_proj_ref(
    *,
    lib: ctypes.CDLL,
    base_ptr: int,
    wo_abs: int,
    attn_token: np.ndarray,
    out_dim: int,
) -> np.ndarray:
    if attn_token.size % QK_K != 0:
        raise RuntimeError(f"out_proj input size {attn_token.size} is not a multiple of {QK_K}")
    q8_buf = ctypes.create_string_buffer((attn_token.size // QK_K) * BLOCK_Q8_K_SIZE)
    attn32 = np.asarray(attn_token, dtype=np.float32)
    lib.quantize_row_q8_k(
        attn32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(q8_buf, ctypes.c_void_p),
        ctypes.c_int(int(attn32.size)),
    )
    out = np.zeros(int(out_dim), dtype=np.float32)
    lib.gemv_q4_k_q8_k_ref(
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_void_p(int(base_ptr) + int(wo_abs)),
        ctypes.cast(q8_buf, ctypes.c_void_p),
        ctypes.c_int(int(out_dim)),
        ctypes.c_int(int(attn32.size)),
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Check the out_proj quantization boundary against llama.cpp")
    ap.add_argument("--model-dir", required=True, type=Path, help="model dir or parent containing .ck_build")
    ap.add_argument("--gguf", type=Path, default=None, help="GGUF path (auto-discovered if omitted)")
    ap.add_argument("--tokens", default="1,2,3,4,5", help="comma-separated explicit token IDs")
    ap.add_argument("--ctx-len", type=int, default=1024, help="context length for llama helper")
    ap.add_argument("--threads", type=int, default=0, help="llama helper threads (0 = auto)")
    ap.add_argument("--top-k", type=int, default=8, help="top-k emitted by llama helper")
    ap.add_argument("--atol", type=float, default=1e-3, help="llama vs CK absolute tolerance")
    ap.add_argument("--rtol", type=float, default=1e-3, help="llama vs CK relative tolerance")
    ap.add_argument("--ref-tol", type=float, default=1e-6, help="reference-vs-CK kernel tolerance")
    ap.add_argument(
        "--keep-workdir",
        type=Path,
        default=None,
        help="optional work dir for fresh llama raw dumps; regenerated on each run",
    )
    args = ap.parse_args()

    model_dir = discover_ck_model_dir(args.model_dir)
    gguf = discover_gguf(args.gguf, model_dir)
    tokens = parse_tokens_csv(args.tokens)

    lowered = load_json(model_dir / "lowered_decode.json")
    lowered_call = load_json(model_dir / "lowered_decode_call.json")
    probes = {probe.probe_name: probe for probe in make_probe_specs(lowered, lowered_call)}

    entries = _load_manifest_entries(model_dir)
    config = _load_model_config(model_dir)
    weights_path = model_dir / "weights.bump"

    ln2_entry = entries.get("layer.0.ln2_gamma")
    wo_entry = entries.get("layer.0.wo")
    if ln2_entry is None or wo_entry is None:
        raise RuntimeError("required layer.0 weights missing from weights_manifest.json")

    gamma = _read_weight_slice_f32(weights_path, int(ln2_entry["offset"]), int(ln2_entry["size"]) // 4)
    eps = float(config.get("rms_norm_eps", 1e-5))
    wo_abs = int(wo_entry["offset"])

    if args.keep_workdir is not None:
        work_dir = args.keep_workdir.expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        meta, dump_map, _ = run_llama_raw_dumps(gguf, tokens, args.ctx_len, args.threads, args.top_k, work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="ckv7_outproj_boundary_") as td:
            work_dir = Path(td)
            meta, dump_map, _ = run_llama_raw_dumps(gguf, tokens, args.ctx_len, args.threads, args.top_k, work_dir)

    lib = load_model_lib(model_dir)
    if lib.ck_model_init(str(weights_path).encode("utf-8")) != 0:
        raise RuntimeError("ck_model_init failed")
    _configure_quant_lib(lib)
    base_ptr = int(lib.ck_model_get_base_ptr())
    if not base_ptr:
        raise RuntimeError("ck_model_get_base_ptr returned null")

    print("=" * 96)
    print("OUT_PROJ QUANT BOUNDARY CHECK")
    print("=" * 96)
    print(f"model_dir       : {model_dir}")
    print(f"gguf            : {gguf}")
    print(f"tokens          : {tokens}")
    print(f"llama_work_dir  : {work_dir}")
    print(f"decode_mode     : {meta.get('decode_mode')}")
    print(f"atol/rtol       : {args.atol}/{args.rtol}")
    print(f"ref_tol         : {args.ref_tol}")
    print("")

    all_ref_ok = True
    for token_step in sorted(dump_map.keys()):
        prefix_tokens = tokens[: token_step + 1]
        _, attn_ck, attn_cmp = _compare_named_probe(
            probes=probes,
            dump_map=dump_map,
            lib=lib,
            base_ptr=base_ptr,
            lowered=lowered,
            tokens_prefix=prefix_tokens,
            token_step=token_step,
            probe_name="__fattn__-0",
            atol=args.atol,
            rtol=args.rtol,
        )
        _, out_ck, out_cmp = _compare_named_probe(
            probes=probes,
            dump_map=dump_map,
            lib=lib,
            base_ptr=base_ptr,
            lowered=lowered,
            tokens_prefix=prefix_tokens,
            token_step=token_step,
            probe_name="attn_out-0",
            atol=args.atol,
            rtol=args.rtol,
        )
        _, ffn_inp_ck, ffn_inp_cmp = _compare_named_probe(
            probes=probes,
            dump_map=dump_map,
            lib=lib,
            base_ptr=base_ptr,
            lowered=lowered,
            tokens_prefix=prefix_tokens,
            token_step=token_step,
            probe_name="ffn_inp-0",
            atol=args.atol,
            rtol=args.rtol,
        )
        _, ffn_norm_ck, ffn_norm_cmp = _compare_named_probe(
            probes=probes,
            dump_map=dump_map,
            lib=lib,
            base_ptr=base_ptr,
            lowered=lowered,
            tokens_prefix=prefix_tokens,
            token_step=token_step,
            probe_name="ffn_norm-0",
            atol=args.atol,
            rtol=args.rtol,
        )

        out_ref = _scalar_out_proj_ref(
            lib=lib,
            base_ptr=base_ptr,
            wo_abs=wo_abs,
            attn_token=attn_ck,
            out_dim=out_ck.size,
        )
        out_ref_cmp = compare_arrays(out_ref, out_ck, atol=args.ref_tol, rtol=0.0)

        norm_ref = _scalar_rmsnorm(ffn_inp_ck, gamma, eps)
        norm_ref_cmp = compare_arrays(norm_ref, ffn_norm_ck, atol=args.ref_tol, rtol=0.0)

        all_ref_ok = all_ref_ok and bool(out_ref_cmp.get("ok")) and bool(norm_ref_cmp.get("ok"))

        print(f"token_step      : {token_step}")
        print(f"prefix_tokens   : {prefix_tokens}")
        print(
            f"  llama_vs_ck __fattn__ : ok={attn_cmp.get('ok')} "
            f"max={attn_cmp.get('max_diff', 0.0):.6e} cos={attn_cmp.get('cosine', 0.0):.6f}"
        )
        print(
            f"  llama_vs_ck attn_out  : ok={out_cmp.get('ok')} "
            f"max={out_cmp.get('max_diff', 0.0):.6e} cos={out_cmp.get('cosine', 0.0):.6f}"
        )
        print(
            f"  llama_vs_ck ffn_inp   : ok={ffn_inp_cmp.get('ok')} "
            f"max={ffn_inp_cmp.get('max_diff', 0.0):.6e} cos={ffn_inp_cmp.get('cosine', 0.0):.6f}"
        )
        print(
            f"  llama_vs_ck ffn_norm  : ok={ffn_norm_cmp.get('ok')} "
            f"max={ffn_norm_cmp.get('max_diff', 0.0):.6e} cos={ffn_norm_cmp.get('cosine', 0.0):.6f}"
        )
        print(
            f"  ref_vs_ck out_proj    : ok={out_ref_cmp.get('ok')} "
            f"max={out_ref_cmp.get('max_diff', 0.0):.6e} cos={out_ref_cmp.get('cosine', 0.0):.6f}"
        )
        print(
            f"  ref_vs_ck ffn_norm    : ok={norm_ref_cmp.get('ok')} "
            f"max={norm_ref_cmp.get('max_diff', 0.0):.6e} cos={norm_ref_cmp.get('cosine', 0.0):.6f}"
        )
        print("")

    if hasattr(lib, "ck_model_free"):
        lib.ck_model_free()

    if all_ref_ok:
        print("PASS: out_proj and RMSNorm kernels match scalar/reference replay on the checked steps.")
        print("      Remaining drift, if any, is entering before or at the quantized out_proj boundary.")
        return 0

    print("FAILED: scalar/reference replay found a CK kernel mismatch in the checked boundary.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
