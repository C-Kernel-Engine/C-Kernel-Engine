#!/usr/bin/env python3
"""
Check final footer contracts for decode:
  1) final RMSNorm output vs local FP32 reference
  2) Q8_K quantized footer input vs dequantized local view
  3) sampled logits rows vs scalar Q6_K x Q8_K reference

This narrows "bad final logits" failures to either:
  - footer math/kernel mismatch, or
  - earlier hidden-state divergence before the footer.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_DIR = Path.home() / ".cache/ck-engine-v7/models/mradermacher--Nanbeige4.1-3B-GGUF/.ck_build"
QK_K = 256
Q6_K_BLOCK_SIZE = 210
Q8_K_BLOCK_SIZE = 292


def load_json(p: Path) -> dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_op(
    ops: list[dict[str, Any]],
    layer: int,
    op_name: str,
    occurrence: int = 0,
) -> tuple[int, dict[str, Any]] | tuple[None, None]:
    idxs = [i for i, op in enumerate(ops) if op.get("layer") == layer and op.get("op") == op_name]
    if len(idxs) <= occurrence:
        return None, None
    i = idxs[occurrence]
    return i, ops[i]


def find_last_op(
    ops: list[dict[str, Any]],
    layer: int,
    op_name: str,
) -> tuple[int, dict[str, Any]] | tuple[None, None]:
    idx = None
    row = None
    for i, op in enumerate(ops):
        if op.get("layer") == layer and op.get("op") == op_name:
            idx = i
            row = op
    return idx, row


def resolve_abs_offset(lowered: dict[str, Any], rel_off: int) -> int:
    arena = lowered.get("memory", {}).get("arena", {})
    if str(arena.get("mode", "")) == "region":
        return int(arena.get("activations_base", 0)) + int(rel_off)
    return int(rel_off)


def read_f32(base_ptr: int, abs_offset: int, count: int) -> np.ndarray:
    ptr = ctypes.cast(base_ptr + abs_offset, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(count,)).copy()


def read_bytes(base_ptr: int, abs_offset: int, size: int) -> bytes:
    return ctypes.string_at(base_ptr + abs_offset, size)


def parse_tokens_csv(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise RuntimeError("token list is empty")
    return out


def zero_activations_preserve_rope(base_ptr: int, lowered: dict[str, Any]) -> None:
    memory = lowered.get("memory", {})
    arena = memory.get("arena", {})
    act = memory.get("activations", {})
    act_size = int(act.get("size", 0))
    if act_size <= 0:
        return

    act_base = int(arena.get("activations_base", 0))

    def abs_offset(buf: dict[str, Any]) -> int:
        if "abs_offset" in buf:
            return int(buf["abs_offset"])
        return act_base + int(buf.get("offset", 0))

    protected: list[tuple[int, int]] = []
    for buf in act.get("buffers", []):
        if buf.get("name") not in {"rope_cache", "kv_cache"}:
            continue
        size = int(buf.get("size", 0))
        if size <= 0:
            continue
        protected.append((abs_offset(buf), size))

    if not protected:
        ctypes.memset(base_ptr + act_base, 0, act_size)
        return

    end = act_base + act_size
    cursor = act_base
    for off, size in sorted(protected):
        if off > cursor:
            ctypes.memset(base_ptr + cursor, 0, off - cursor)
        cursor = max(cursor, off + size)
    if cursor < end:
        ctypes.memset(base_ptr + cursor, 0, end - cursor)


def run_decode_tokens_until(
    model_lib: ctypes.CDLL,
    base_ptr: int,
    lowered: dict[str, Any],
    tokens: list[int],
    stop_idx: int,
) -> None:
    if not tokens:
        raise RuntimeError("token list is empty")
    os.environ.pop("CK_STOP_OP", None)
    if hasattr(model_lib, "ck_model_kv_cache_reset"):
        model_lib.ck_model_kv_cache_reset()
    zero_activations_preserve_rope(base_ptr, lowered)
    for tok in tokens[:-1]:
        rc = model_lib.ck_model_decode(ctypes.c_int32(int(tok)), None)
        if rc != 0:
            raise RuntimeError(f"history ck_model_decode failed rc={rc}")
    os.environ["CK_STOP_OP"] = str(stop_idx)
    zero_activations_preserve_rope(base_ptr, lowered)
    rc = model_lib.ck_model_decode(ctypes.c_int32(int(tokens[-1])), None)
    if rc != 0:
        raise RuntimeError(f"final ck_model_decode failed at CK_STOP_OP={stop_idx}, rc={rc}")


def rmsnorm_ref(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    mean_sq = np.mean(x * x, dtype=np.float32)
    rstd = np.float32(1.0) / np.sqrt(np.float32(mean_sq) + np.float32(eps))
    return x.astype(np.float32) * rstd * gamma.astype(np.float32)


def diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return float(np.max(d)), float(np.mean(d)), int(np.argmax(d))


def _load_weight_row(weight_path: Path, file_offset: int, row_bytes: int, row_idx: int) -> bytes:
    with weight_path.open("rb") as f:
        f.seek(int(file_offset) + int(row_idx) * int(row_bytes))
        buf = f.read(int(row_bytes))
    if len(buf) != int(row_bytes):
        raise RuntimeError(f"short read for weight row {row_idx}")
    return buf


def _parse_q8_blocks(raw: bytes, k: int) -> list[dict[str, Any]]:
    if k % QK_K != 0:
        raise RuntimeError(f"K must be multiple of {QK_K}, got {k}")
    nb = k // QK_K
    expected = nb * Q8_K_BLOCK_SIZE
    if len(raw) < expected:
        raise RuntimeError(f"Q8_K buffer too small: got={len(raw)} expected>={expected}")
    out: list[dict[str, Any]] = []
    for i in range(nb):
        off = i * Q8_K_BLOCK_SIZE
        block = raw[off : off + Q8_K_BLOCK_SIZE]
        d = float(np.frombuffer(block[0:4], dtype=np.float32)[0])
        qs = np.frombuffer(block[4 : 4 + QK_K], dtype=np.int8).astype(np.int16, copy=True)
        bsums = np.frombuffer(block[4 + QK_K : 4 + QK_K + 32], dtype=np.int16).copy()
        out.append({"d": d, "qs": qs, "bsums": bsums})
    return out


def _dequantize_q8_blocks(blocks: list[dict[str, Any]]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for block in blocks:
        rows.append(block["qs"].astype(np.float32) * np.float32(block["d"]))
    return np.concatenate(rows).astype(np.float32, copy=False)


def _dot_q6_k_q8_k_ref(row_bytes: bytes, q8_blocks: list[dict[str, Any]], k: int) -> float:
    if k % QK_K != 0:
        raise RuntimeError(f"K must be multiple of {QK_K}, got {k}")
    nb = k // QK_K
    expected = nb * Q6_K_BLOCK_SIZE
    if len(row_bytes) != expected:
        raise RuntimeError(f"Q6_K row size mismatch: got={len(row_bytes)} expected={expected}")

    sumf = 0.0
    for i in range(nb):
        off = i * Q6_K_BLOCK_SIZE
        block = row_bytes[off : off + Q6_K_BLOCK_SIZE]
        ql = np.frombuffer(block[0:128], dtype=np.uint8)
        qh = np.frombuffer(block[128:192], dtype=np.uint8)
        scales = np.frombuffer(block[192:208], dtype=np.int8).astype(np.int16, copy=False)
        d_q6 = float(np.frombuffer(block[208:210], dtype=np.float16)[0])
        q8 = q8_blocks[i]
        d = d_q6 * float(q8["d"])
        q8_vals = q8["qs"]

        ql_ptr = 0
        qh_ptr = 0
        sc_ptr = 0
        q8_ptr = 0
        for _ in range(2):
            ql_chunk = ql[ql_ptr : ql_ptr + 64]
            qh_chunk = qh[qh_ptr : qh_ptr + 32]
            sc_chunk = scales[sc_ptr : sc_ptr + 8]
            q8_chunk = q8_vals[q8_ptr : q8_ptr + 128]
            for l in range(32):
                isub = l // 16
                q1 = int((int(ql_chunk[l + 0]) & 0xF) | (((int(qh_chunk[l]) >> 0) & 3) << 4)) - 32
                q2 = int((int(ql_chunk[l + 32]) & 0xF) | (((int(qh_chunk[l]) >> 2) & 3) << 4)) - 32
                q3 = int((int(ql_chunk[l + 0]) >> 4) | (((int(qh_chunk[l]) >> 4) & 3) << 4)) - 32
                q4 = int((int(ql_chunk[l + 32]) >> 4) | (((int(qh_chunk[l]) >> 6) & 3) << 4)) - 32
                sumf += d * float(sc_chunk[isub + 0]) * float(q1) * float(q8_chunk[l + 0])
                sumf += d * float(sc_chunk[isub + 2]) * float(q2) * float(q8_chunk[l + 32])
                sumf += d * float(sc_chunk[isub + 4]) * float(q3) * float(q8_chunk[l + 64])
                sumf += d * float(sc_chunk[isub + 6]) * float(q4) * float(q8_chunk[l + 96])
            ql_ptr += 64
            qh_ptr += 32
            sc_ptr += 8
            q8_ptr += 128
    return float(sumf)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check footer RMSNorm/Q8/logits contracts")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument(
        "--tokens",
        default="1,2,3,4,5",
        help="comma-separated token ids; the final id is the decode step being checked",
    )
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--rows", type=str, default="", help="extra comma-separated logits row ids to sample")
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    tokens = parse_tokens_csv(args.tokens)
    lowered = load_json(model_dir / "lowered_decode.json")
    lowered_call = load_json(model_dir / "lowered_decode_call.json")
    manifest = load_json(model_dir / "weights_manifest.json")
    ops = lowered.get("operations", [])
    call_ops = lowered_call.get("operations", [])
    cfg = lowered.get("config", {})

    embed_dim = int(cfg.get("embed_dim", 0))
    num_layers = int(cfg.get("num_layers", 0))
    eps = float(cfg.get("rms_eps", 1e-6))
    if embed_dim <= 0 or num_layers <= 0:
        raise RuntimeError("invalid embed_dim/num_layers in lowered config")

    last_layer = num_layers - 1
    last_res_idx, last_res_op = find_last_op(ops, last_layer, "residual_add")
    last_res_stop_idx, _ = find_last_op(call_ops, last_layer, "residual_add")
    final_rms_idx, final_rms_op = find_op(ops, -1, "rmsnorm", 0)
    final_rms_stop_idx, _ = find_op(call_ops, -1, "rmsnorm", 0)
    q8_idx, q8_op = find_op(ops, -1, "quantize_final_output", 0)
    q8_stop_idx, _ = find_op(call_ops, -1, "quantize_final_output", 0)
    logits_idx, logits_op = find_op(ops, -1, "logits", 0)
    logits_stop_idx, _ = find_op(call_ops, -1, "logits", 0)
    if None in (last_res_op, final_rms_op, q8_op, logits_op):
        raise RuntimeError("required footer ops missing")
    if None in (last_res_stop_idx, final_rms_stop_idx, q8_stop_idx, logits_stop_idx):
        raise RuntimeError("required footer stop ops missing")

    final_in_rel = int(final_rms_op.get("activations", {}).get("input", {}).get("activation_offset", 0))
    final_out_rel = int(final_rms_op.get("outputs", {}).get("output", {}).get("activation_offset", 0))
    q8_out_rel = int(q8_op.get("outputs", {}).get("output", {}).get("activation_offset", 0))
    logits_rel = int(logits_op.get("outputs", {}).get("y", {}).get("activation_offset", 0))
    final_in_abs = resolve_abs_offset(lowered, final_in_rel)
    final_out_abs = resolve_abs_offset(lowered, final_out_rel)
    q8_out_abs = resolve_abs_offset(lowered, q8_out_rel)
    logits_abs = resolve_abs_offset(lowered, logits_rel)

    final_ln_name = str(final_rms_op.get("weights", {}).get("final_ln_weight", {}).get("name", ""))
    lm_head_name = str(logits_op.get("weights", {}).get("lm_head", {}).get("name", ""))
    entry_by_name = {e.get("name"): e for e in manifest.get("entries", []) if isinstance(e, dict)}
    final_ln_entry = entry_by_name.get(final_ln_name)
    lm_head_entry = entry_by_name.get(lm_head_name)
    if not final_ln_entry or not lm_head_entry:
        raise RuntimeError("missing final_ln_weight or lm_head manifest entry")

    vocab_size = int(logits_op.get("params", {}).get("_output_dim", 0))
    if vocab_size <= 0:
        raise RuntimeError("invalid vocab size in logits op")
    blocks_per_row = embed_dim // QK_K
    if blocks_per_row <= 0 or (embed_dim % QK_K) != 0:
        raise RuntimeError(f"embed_dim must be multiple of {QK_K}, got {embed_dim}")
    row_bytes = blocks_per_row * Q6_K_BLOCK_SIZE
    lm_head_size = int(lm_head_entry.get("size", 0))
    if lm_head_size != vocab_size * row_bytes:
        raise RuntimeError(
            f"lm_head size mismatch: size={lm_head_size} vocab={vocab_size} row_bytes={row_bytes}"
        )

    with (model_dir / "weights.bump").open("rb") as f:
        f.seek(int(final_ln_entry["file_offset"]))
        gamma = np.frombuffer(f.read(int(final_ln_entry["size"])), dtype=np.float32).copy()
    if gamma.size != embed_dim:
        raise RuntimeError(f"final gamma size mismatch: {gamma.size} vs {embed_dim}")

    lib = ctypes.CDLL(str(model_dir / "libmodel.so"))
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
    if hasattr(lib, "ck_model_kv_cache_reset"):
        lib.ck_model_kv_cache_reset.argtypes = []
        lib.ck_model_kv_cache_reset.restype = None
    if hasattr(lib, "ck_model_free"):
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None

    if lib.ck_model_init(str(model_dir / "weights.bump").encode()) != 0:
        raise RuntimeError("ck_model_init failed")
    base_ptr = int(lib.ck_model_get_base_ptr())
    if not base_ptr:
        raise RuntimeError("base_ptr null")

    try:
        run_decode_tokens_until(lib, base_ptr, lowered, tokens, int(last_res_stop_idx))
        x_final = read_f32(base_ptr, final_in_abs, embed_dim)

        run_decode_tokens_until(lib, base_ptr, lowered, tokens, int(final_rms_stop_idx))
        y_final = read_f32(base_ptr, final_out_abs, embed_dim)

        run_decode_tokens_until(lib, base_ptr, lowered, tokens, int(q8_stop_idx))
        q8_raw = read_bytes(base_ptr, q8_out_abs, blocks_per_row * Q8_K_BLOCK_SIZE)

        run_decode_tokens_until(lib, base_ptr, lowered, tokens, int(logits_stop_idx))
        logits_runtime = read_f32(base_ptr, logits_abs, vocab_size)
    finally:
        os.environ.pop("CK_STOP_OP", None)
        if hasattr(lib, "ck_model_free"):
            lib.ck_model_free()

    y_ref = rmsnorm_ref(x_final, gamma, eps)
    rms_max, rms_mean, rms_worst = diff_stats(y_final, y_ref)

    q8_blocks = _parse_q8_blocks(q8_raw, embed_dim)
    q8_dequant = _dequantize_q8_blocks(q8_blocks)
    q8_max, q8_mean, q8_worst = diff_stats(q8_dequant, y_final)

    topk = np.argsort(logits_runtime)[::-1][: max(1, int(args.top_k))]
    extra_rows = [int(x.strip()) for x in str(args.rows or "").split(",") if x.strip()]
    sample_rows = sorted({int(i) for i in topk.tolist() + extra_rows if 0 <= int(i) < vocab_size})

    sampled: list[dict[str, Any]] = []
    max_logit_diff = 0.0
    mean_logit_diff = 0.0
    if sample_rows:
        diffs: list[float] = []
        weight_path = model_dir / "weights.bump"
        lm_head_file_offset = int(lm_head_entry["file_offset"])
        for row in sample_rows:
            row_bytes_buf = _load_weight_row(weight_path, lm_head_file_offset, row_bytes, row)
            ref = _dot_q6_k_q8_k_ref(row_bytes_buf, q8_blocks, embed_dim)
            runtime = float(logits_runtime[row])
            diff = abs(runtime - ref)
            diffs.append(diff)
            sampled.append(
                {
                    "row": int(row),
                    "runtime": runtime,
                    "ref": float(ref),
                    "abs_diff": float(diff),
                }
            )
        max_logit_diff = float(max(diffs))
        mean_logit_diff = float(sum(diffs) / len(diffs))

    report = {
        "model_dir": str(model_dir),
        "tokens": [int(t) for t in tokens],
        "embed_dim": embed_dim,
        "vocab_size": vocab_size,
        "footer_indices": {
            "last_residual_add": int(last_res_idx),
            "final_rmsnorm": int(final_rms_idx),
            "quantize_final_output": int(q8_idx),
            "logits": int(logits_idx),
        },
        "rmsnorm_check": {
            "max_diff": rms_max,
            "mean_diff": rms_mean,
            "worst_index": rms_worst,
            "runtime": float(y_final[rms_worst]),
            "ref": float(y_ref[rms_worst]),
        },
        "q8_roundtrip_check": {
            "max_diff": q8_max,
            "mean_diff": q8_mean,
            "worst_index": q8_worst,
            "runtime": float(y_final[q8_worst]),
            "dequant_q8": float(q8_dequant[q8_worst]),
        },
        "sampled_logits_check": {
            "rows_checked": len(sample_rows),
            "max_abs_diff": max_logit_diff,
            "mean_abs_diff": mean_logit_diff,
            "rows": sampled,
        },
    }

    print("=" * 88)
    print("FOOTER LOGITS CONTRACT CHECK")
    print("=" * 88)
    print(f"model_dir         : {model_dir}")
    print(f"tokens            : {tokens}")
    print(
        f"indices            : last_res={last_res_idx}, final_rms={final_rms_idx}, "
        f"q8={q8_idx}, logits={logits_idx}"
    )
    print("")
    print("final RMSNorm vs local FP32 reference")
    print(f"  max_diff        : {rms_max:.6e}")
    print(f"  mean_diff       : {rms_mean:.6e}")
    print(f"  worst idx       : {rms_worst}")
    print("")
    print("Q8_K roundtrip vs runtime final output")
    print(f"  max_diff        : {q8_max:.6e}")
    print(f"  mean_diff       : {q8_mean:.6e}")
    print(f"  worst idx       : {q8_worst}")
    print("")
    print("sampled logits vs scalar Q6_K x Q8_K reference")
    print(f"  rows_checked    : {len(sample_rows)}")
    print(f"  max_abs_diff    : {max_logit_diff:.6e}")
    print(f"  mean_abs_diff   : {mean_logit_diff:.6e}")
    for row in sampled[: min(8, len(sampled))]:
        print(
            f"  row {row['row']:6d}: runtime={row['runtime']:.6f} "
            f"ref={row['ref']:.6f} abs_diff={row['abs_diff']:.6e}"
        )
    print("=" * 88)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
