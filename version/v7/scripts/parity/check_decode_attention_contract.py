#!/usr/bin/env python3
"""
Focused decode attention contract checker.

This validates a single layer on a realistic decode step using explicit token IDs:
1) capture pre-RoPE q/k/v for the final token,
2) compare runtime post-RoPE q/k against a local RoPE reference using the
   model-selected layout (Llama pairwise vs split-half/NEOX),
3) compare the KV cache slot written for the final token against expected k/v,
4) compare runtime attention output against a local GQA attention reference.

It is intentionally tokenizer-free and is meant to narrow first divergence after
QKV projection parity has already passed.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_DIR = Path.home() / ".cache/ck-engine-v7/models/mradermacher--Nanbeige4.1-3B-GGUF/.ck_build"


@dataclass
class DecodeAttentionBindings:
    layer: int
    q_idx: int
    k_idx: int
    v_idx: int
    rope_idx: int
    kv_idx: int
    attn_idx: int
    q_abs: int
    k_abs: int
    v_abs: int
    attn_abs: int
    q_count: int
    k_count: int
    v_count: int
    attn_count: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    aligned_head_dim: int
    rotary_dim: int
    cache_capacity: int
    rope_theta: float
    rope_layout: str
    kv_cache_abs: int
    attn_fn_name: str


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_tokens_csv(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise RuntimeError("token list is empty")
    return out


def _load_model_macro_offsets(model_dir: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    model_c = model_dir / "model_v7.c"
    if not model_c.exists():
        return out
    for line in model_c.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"^\s*#define\s+([A-Z0-9_]+)\s+([0-9]+)\s*$", line)
        if not m:
            continue
        out[m.group(1)] = int(m.group(2))
    return out


def _find_layer_op(
    ops: list[dict[str, Any]],
    layer: int,
    op_name: str,
) -> tuple[int | None, dict[str, Any] | None]:
    for i, op in enumerate(ops):
        if op.get("layer") == layer and op.get("op") == op_name:
            return i, op
    return None, None


def _find_layer_op_any(
    ops: list[dict[str, Any]],
    layer: int,
    op_names: tuple[str, ...],
) -> tuple[int | None, dict[str, Any] | None]:
    for i, op in enumerate(ops):
        if op.get("layer") == layer and op.get("op") in op_names:
            return i, op
    return None, None


def _resolve_abs_offset(lowered: dict[str, Any], rel_off: int) -> int:
    arena = lowered.get("memory", {}).get("arena", {})
    if str(arena.get("mode", "")) == "region":
        return int(arena.get("activations_base", 0)) + int(rel_off)
    return int(rel_off)


def _first_activation_offset(d: dict[str, Any] | None) -> int:
    if not isinstance(d, dict):
        raise RuntimeError("activation/output map missing")
    for v in d.values():
        if isinstance(v, dict) and "activation_offset" in v:
            return int(v["activation_offset"])
    raise RuntimeError("activation_offset not found in map")


def _find_arg_expr(call_op: dict[str, Any], name: str) -> str | None:
    for arg in call_op.get("args", []):
        if not isinstance(arg, dict):
            continue
        if str(arg.get("name", "")) == name:
            expr = arg.get("expr")
            if isinstance(expr, str):
                return expr
    return None


def _parse_int_expr(expr: str | None) -> int | None:
    if not expr:
        return None
    expr = str(expr).strip()
    try:
        return int(expr)
    except Exception:
        return None


def _assert_runtime_artifact_freshness(model_dir: Path) -> None:
    libmodel = model_dir / "libmodel.so"
    if not libmodel.exists():
        return
    deps = [
        model_dir / "model_v7.c",
        model_dir / "lowered_decode.json",
        model_dir / "lowered_decode_call.json",
    ]
    try:
        lib_mtime = libmodel.stat().st_mtime
    except OSError:
        return
    newer = [p.name for p in deps if p.exists() and p.stat().st_mtime > lib_mtime]
    if newer:
        joined = ", ".join(newer)
        raise RuntimeError(
            f"stale runtime artifacts detected: libmodel.so is older than [{joined}]. "
            "Rebuild with `ck_run_v7.py run <model-dir> --force-compile`."
        )


def _load_model_lib(model_dir: Path) -> ctypes.CDLL:
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
    return lib


def _read_f32(base_ptr: int, abs_off: int, count: int) -> np.ndarray:
    ptr = ctypes.cast(base_ptr + abs_off, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(count,)).copy()


def _zero_activations_preserve_rope(base_ptr: int, lowered: dict[str, Any]) -> None:
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


def _run_decode_tokens_until(
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
    _zero_activations_preserve_rope(base_ptr, lowered)
    for tok in tokens[:-1]:
        rc = model_lib.ck_model_decode(ctypes.c_int32(int(tok)), None)
        if rc != 0:
            raise RuntimeError(f"history ck_model_decode failed rc={rc}")
    os.environ["CK_STOP_OP"] = str(stop_idx)
    _zero_activations_preserve_rope(base_ptr, lowered)
    rc = model_lib.ck_model_decode(ctypes.c_int32(int(tokens[-1])), None)
    if rc != 0:
        raise RuntimeError(f"final ck_model_decode failed at CK_STOP_OP={stop_idx}, rc={rc}")


def _compare_pair(name: str, expected: np.ndarray, got: np.ndarray, tol: float) -> tuple[bool, str]:
    if expected.shape != got.shape:
        return False, f"{name}: shape mismatch expected={expected.shape} got={got.shape}"
    diff = np.abs(expected.astype(np.float32) - got.astype(np.float32))
    max_diff = float(np.max(diff)) if diff.size else 0.0
    mean_diff = float(np.mean(diff)) if diff.size else 0.0
    worst_idx = int(np.argmax(diff)) if diff.size else 0
    flat_exp = expected.reshape(-1)
    flat_got = got.reshape(-1)
    exp_v = float(flat_exp[worst_idx]) if flat_exp.size else 0.0
    got_v = float(flat_got[worst_idx]) if flat_got.size else 0.0
    ok = bool(np.all(np.isfinite(diff)) and max_diff <= float(tol))
    msg = (
        f"{name}\n"
        f"  max_diff      : {max_diff:.6e}\n"
        f"  mean_diff     : {mean_diff:.6e}\n"
        f"  worst idx     : {worst_idx}\n"
        f"  runtime[idx]  : {got_v:.8f}\n"
        f"  ref[idx]      : {exp_v:.8f}"
    )
    return ok, msg


def _apply_rope_split_half_single(
    x: np.ndarray,
    *,
    pos: int,
    theta: float,
    rotary_dim: int,
) -> np.ndarray:
    out = np.array(x, dtype=np.float32, copy=True)
    width = int(out.shape[-1])
    rotary = min(int(rotary_dim), width)
    rotary -= (rotary % 2)
    if rotary <= 0:
        return out
    half = rotary // 2
    freq_seq = np.arange(half, dtype=np.float32)
    inv_freq = 1.0 / (float(theta) ** (freq_seq / max(1.0, float(half))))
    freqs = float(pos) * inv_freq
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)

    x0 = np.array(out[:, :half], copy=True)
    x1 = np.array(out[:, half:rotary], copy=True)
    out[:, :half] = (x0 * cos) - (x1 * sin)
    out[:, half:rotary] = (x0 * sin) + (x1 * cos)
    return out


def _apply_rope_pairwise_single(
    x: np.ndarray,
    *,
    pos: int,
    theta: float,
    rotary_dim: int,
) -> np.ndarray:
    out = np.array(x, dtype=np.float32, copy=True)
    width = int(out.shape[-1])
    rotary = min(int(rotary_dim), width)
    rotary -= (rotary % 2)
    if rotary <= 0:
        return out
    half = rotary // 2
    freq_seq = np.arange(half, dtype=np.float32)
    inv_freq = 1.0 / (float(theta) ** (freq_seq / max(1.0, float(half))))
    freqs = float(pos) * inv_freq
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)

    even = np.array(out[:, 0:rotary:2], copy=True)
    odd = np.array(out[:, 1:rotary:2], copy=True)
    out[:, 0:rotary:2] = (even * cos) - (odd * sin)
    out[:, 1:rotary:2] = (even * sin) + (odd * cos)
    return out


def _detect_rope_layout(manifest: dict[str, Any]) -> str:
    template = manifest.get("template") if isinstance(manifest.get("template"), dict) else {}
    kernels = template.get("kernels") if isinstance(template.get("kernels"), dict) else {}
    rope_kernel = str(kernels.get("rope_qk", "") or "").strip().lower()
    if "pairwise" in rope_kernel:
        return "pairwise"

    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    arch = str(metadata.get("general.architecture", "") or "").strip().lower()
    family = str(template.get("family", "") or template.get("name", "") or config.get("model", "")).strip().lower()
    if arch == "llama" or family == "llama":
        return "pairwise"
    return "split_half"


def _apply_rope_single(
    x: np.ndarray,
    *,
    pos: int,
    theta: float,
    rotary_dim: int,
    layout: str,
) -> np.ndarray:
    if layout == "pairwise":
        return _apply_rope_pairwise_single(x, pos=pos, theta=theta, rotary_dim=rotary_dim)
    return _apply_rope_split_half_single(x, pos=pos, theta=theta, rotary_dim=rotary_dim)


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(shifted)
    denom = np.sum(ex, axis=-1, keepdims=True)
    return ex / np.maximum(denom, 1e-30)


def _round_fp16_scalar(x: float) -> float:
    return float(np.float16(np.float32(x)))


def _decode_attention_ref(
    q_post: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    if num_heads % num_kv_heads != 0:
        raise RuntimeError(f"Unsupported GQA ratio: q_heads={num_heads}, kv_heads={num_kv_heads}")
    rep = num_heads // num_kv_heads
    q = np.asarray(q_post, dtype=np.float32)
    k = np.asarray(k_cache, dtype=np.float32)
    v = np.asarray(v_cache, dtype=np.float32)
    k = np.repeat(k, rep, axis=0)
    v = np.repeat(v, rep, axis=0)
    scores = np.einsum("hd,htd->ht", q, k, optimize=True) * (1.0 / math.sqrt(float(head_dim)))
    probs = _softmax_rows(scores.astype(np.float64)).astype(np.float32)
    return np.einsum("ht,htd->hd", probs, v, optimize=True).astype(np.float32)


def _decode_attention_ref_f16kv(
    q_post: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    if num_heads % num_kv_heads != 0:
        raise RuntimeError(f"Unsupported GQA ratio: q_heads={num_heads}, kv_heads={num_kv_heads}")
    rep = num_heads // num_kv_heads
    q = np.asarray(q_post, dtype=np.float32)
    k = np.repeat(np.asarray(k_cache, dtype=np.float32), rep, axis=0)
    v = np.repeat(np.asarray(v_cache, dtype=np.float32), rep, axis=0)
    scale = 1.0 / math.sqrt(float(head_dim))
    out = np.zeros((num_heads, head_dim), dtype=np.float32)

    for h in range(num_heads):
        max_score = -math.inf
        sum_exp = 0.0
        acc = np.zeros((head_dim,), dtype=np.float32)
        for t in range(k.shape[1]):
            dot = 0.0
            for d in range(head_dim):
                dot += _round_fp16_scalar(float(q[h, d])) * _round_fp16_scalar(float(k[h, t, d]))
            score = dot * scale

            prev_max = max_score
            max_scale = 1.0
            value_scale = 1.0
            if score > max_score:
                max_score = score
                max_scale = math.exp(prev_max - max_score) if math.isfinite(prev_max) else 0.0
                for d in range(head_dim):
                    acc[d] = _round_fp16_scalar(float(acc[d] * max_scale))
            else:
                value_scale = math.exp(score - max_score)

            for d in range(head_dim):
                updated = float(acc[d]) + value_scale * _round_fp16_scalar(float(v[h, t, d]))
                acc[d] = _round_fp16_scalar(updated)

            sum_exp = sum_exp * max_scale + value_scale

        if sum_exp > 0.0:
            out[h, :] = acc / np.float32(sum_exp)

    return out


def _infer_bindings(model_dir: Path, layer: int) -> tuple[dict[str, Any], DecodeAttentionBindings]:
    lowered = _load_json(model_dir / "lowered_decode.json")
    lowered_call = _load_json(model_dir / "lowered_decode_call.json")
    manifest = _load_json(model_dir / "weights_manifest.json")
    ops = lowered.get("operations", [])
    call_ops = lowered_call.get("operations", [])
    macro_offsets = _load_model_macro_offsets(model_dir)

    needed = {}
    for name in ("q_proj", "k_proj", "v_proj", "rope_qk", "kv_cache_store"):
        idx, op = _find_layer_op(ops, layer, name)
        call_idx, call_op = _find_layer_op(call_ops, layer, name)
        if op is None or call_op is None:
            raise RuntimeError(f"missing decode op for layer={layer}: {name}")
        needed[name] = (idx, op, call_idx, call_op)

    # Dense families lower the decode attention core as `attn`; Gemma and other
    # sliding-window families lower it as `attn_sliding`. Treat this as an
    # operator alias at the parity layer rather than hard-coding by model name.
    attn_aliases = ("attn", "attn_sliding")
    attn_idx, attn_op = _find_layer_op_any(ops, layer, attn_aliases)
    attn_call_idx, attn_call = _find_layer_op_any(call_ops, layer, attn_aliases)
    if attn_op is None or attn_call is None:
        raise RuntimeError(
            f"missing decode op for layer={layer}: one of {', '.join(attn_aliases)}"
        )
    needed["attn"] = (attn_idx, attn_op, attn_call_idx, attn_call)

    q_idx, q_op, _, q_call = needed["q_proj"]
    k_idx, k_op, _, k_call = needed["k_proj"]
    v_idx, v_op, _, v_call = needed["v_proj"]
    rope_idx, rope_op, _, rope_call = needed["rope_qk"]
    kv_idx, kv_op, _, kv_call = needed["kv_cache_store"]
    attn_idx, attn_op, _, attn_call = needed["attn"]

    num_heads = _parse_int_expr(_find_arg_expr(rope_call, "num_heads"))
    num_kv_heads = _parse_int_expr(_find_arg_expr(rope_call, "num_kv_heads"))
    head_dim = _parse_int_expr(_find_arg_expr(rope_call, "head_dim"))
    aligned_head_dim = _parse_int_expr(_find_arg_expr(rope_call, "aligned_head_dim"))
    rotary_dim = _parse_int_expr(_find_arg_expr(rope_call, "rotary_dim"))
    cache_capacity = _parse_int_expr(_find_arg_expr(attn_call, "cache_capacity"))
    rope_theta = float((manifest.get("config") or {}).get("rope_theta", 10000.0) or 10000.0)
    rope_layout = _detect_rope_layout(manifest)
    kv_cache_abs = macro_offsets.get("A_KV_CACHE")
    if kv_cache_abs is None:
        raise RuntimeError("A_KV_CACHE macro not found in model_v7.c")

    q_count = _parse_int_expr(_find_arg_expr(q_call, "M"))
    k_count = _parse_int_expr(_find_arg_expr(k_call, "M"))
    v_count = _parse_int_expr(_find_arg_expr(v_call, "M"))
    attn_count = int(num_heads or 0) * int(aligned_head_dim or 0)
    if q_count is None or k_count is None or v_count is None:
        raise RuntimeError("projection dims missing in lowered_decode_call.json")
    if num_heads is None or num_kv_heads is None or head_dim is None or aligned_head_dim is None:
        raise RuntimeError("rope dims missing in lowered_decode_call.json")
    if rotary_dim is None:
        rotary_dim = head_dim
    if cache_capacity is None:
        cache_capacity = macro_offsets.get("MAX_SEQ_LEN")
    if cache_capacity is None:
        raise RuntimeError("cache_capacity/MAX_SEQ_LEN missing")

    q_abs = _resolve_abs_offset(lowered, _first_activation_offset(q_op.get("outputs")))
    k_abs = _resolve_abs_offset(lowered, _first_activation_offset(k_op.get("outputs")))
    v_abs = _resolve_abs_offset(lowered, _first_activation_offset(v_op.get("outputs")))
    attn_abs = _resolve_abs_offset(lowered, _first_activation_offset(attn_op.get("outputs")))

    return lowered, DecodeAttentionBindings(
        layer=layer,
        q_idx=int(q_op.get("idx", q_idx)),
        k_idx=int(k_op.get("idx", k_idx)),
        v_idx=int(v_op.get("idx", v_idx)),
        rope_idx=int(rope_op.get("idx", rope_idx)),
        kv_idx=int(kv_op.get("idx", kv_idx)),
        attn_idx=int(attn_op.get("idx", attn_idx)),
        q_abs=q_abs,
        k_abs=k_abs,
        v_abs=v_abs,
        attn_abs=attn_abs,
        q_count=int(q_count),
        k_count=int(k_count),
        v_count=int(v_count),
        attn_count=int(attn_count),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        aligned_head_dim=int(aligned_head_dim),
        rotary_dim=int(rotary_dim),
        cache_capacity=int(cache_capacity),
        rope_theta=float(rope_theta),
        rope_layout=str(rope_layout),
        kv_cache_abs=int(kv_cache_abs),
        attn_fn_name=str(attn_call.get("function", "")),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Check decode RoPE/KV/attention contract for one layer")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="ck_build model directory")
    ap.add_argument("--layer", type=int, default=0, help="layer index")
    ap.add_argument(
        "--tokens",
        default="1,2,3,4,5",
        help="comma-separated token ids; the final id is the decode step being checked",
    )
    ap.add_argument("--rope-tol", type=float, default=1e-5, help="absolute tolerance for post-RoPE q/k")
    ap.add_argument("--cache-tol", type=float, default=1e-6, help="absolute tolerance for KV cache slot writes")
    ap.add_argument("--attn-tol", type=float, default=1e-4, help="absolute tolerance for attention output")
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    tokens = _parse_tokens_csv(args.tokens)
    if len(tokens) > 1_000_000:
        raise RuntimeError("token list is unreasonably large")

    needed = [
        model_dir / "weights.bump",
        model_dir / "weights_manifest.json",
        model_dir / "libmodel.so",
        model_dir / "lowered_decode.json",
        model_dir / "lowered_decode_call.json",
        model_dir / "model_v7.c",
    ]
    for p in needed:
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")

    _assert_runtime_artifact_freshness(model_dir)
    lowered, b = _infer_bindings(model_dir, args.layer)
    model_lib = _load_model_lib(model_dir)
    model_rc = model_lib.ck_model_init(str(model_dir / "weights.bump").encode())
    if model_rc != 0:
        raise RuntimeError(f"ck_model_init failed with code {model_rc}")

    base_ptr = int(model_lib.ck_model_get_base_ptr())
    if not base_ptr:
        raise RuntimeError("ck_model_get_base_ptr returned null")

    q_dim = b.q_count // b.num_heads
    k_dim = b.k_count // b.num_kv_heads
    v_dim = b.v_count // b.num_kv_heads
    if q_dim <= 0 or k_dim <= 0 or v_dim <= 0:
        raise RuntimeError(f"invalid head dims q/k/v = {q_dim}/{k_dim}/{v_dim}")
    if b.aligned_head_dim != q_dim:
        print(
            f"warning: aligned_head_dim={b.aligned_head_dim} but q rows imply q_dim={q_dim}; "
            "using q_dim from runtime rows."
        )
    if b.head_dim != k_dim or b.head_dim != v_dim:
        print(
            f"warning: head_dim={b.head_dim}, but projection rows imply k_dim={k_dim}, v_dim={v_dim}; "
            "cache/reference math uses projection row dims where needed."
        )
    if q_dim != k_dim:
        raise RuntimeError(
            f"checker currently expects q_dim == k_dim for decode attention parity (got {q_dim} vs {k_dim})"
        )

    final_pos = len(tokens) - 1
    kv_tokens = len(tokens)
    k_layer_bytes = b.num_kv_heads * b.cache_capacity * b.head_dim * 4
    k_cache_layer_abs = b.kv_cache_abs + (b.layer * 2) * k_layer_bytes
    v_cache_layer_abs = b.kv_cache_abs + (b.layer * 2 + 1) * k_layer_bytes

    failures: list[str] = []
    try:
        print("=" * 88)
        print("DECODE ATTENTION CONTRACT CHECK")
        print("=" * 88)
        print(f"model_dir        : {model_dir}")
        print(f"layer            : {b.layer}")
        print(f"tokens           : {tokens}")
        print(f"final token pos  : {final_pos}")
        print(
            f"indices          : q/k/v/rope/kv/attn = "
            f"{b.q_idx}/{b.k_idx}/{b.v_idx}/{b.rope_idx}/{b.kv_idx}/{b.attn_idx}"
        )
        print(
            f"dims             : q_heads={b.num_heads}, kv_heads={b.num_kv_heads}, "
            f"head_dim={b.head_dim}, aligned={q_dim}, rotary={b.rotary_dim}, cache={b.cache_capacity}"
        )
        print(f"rope_theta       : {b.rope_theta}")
        print(f"rope_layout      : {b.rope_layout}")
        print("")

        _run_decode_tokens_until(model_lib, base_ptr, lowered, tokens, b.q_idx)
        q_pre = _read_f32(base_ptr, b.q_abs, b.q_count).reshape(b.num_heads, q_dim)

        _run_decode_tokens_until(model_lib, base_ptr, lowered, tokens, b.k_idx)
        k_pre = _read_f32(base_ptr, b.k_abs, b.k_count).reshape(b.num_kv_heads, k_dim)

        _run_decode_tokens_until(model_lib, base_ptr, lowered, tokens, b.v_idx)
        v_pre = _read_f32(base_ptr, b.v_abs, b.v_count).reshape(b.num_kv_heads, v_dim)

        _run_decode_tokens_until(model_lib, base_ptr, lowered, tokens, b.rope_idx)
        q_post_rt = _read_f32(base_ptr, b.q_abs, b.q_count).reshape(b.num_heads, q_dim)
        k_post_rt = _read_f32(base_ptr, b.k_abs, b.k_count).reshape(b.num_kv_heads, k_dim)

        q_post_ref = _apply_rope_single(
            q_pre,
            pos=final_pos,
            theta=b.rope_theta,
            rotary_dim=b.rotary_dim,
            layout=b.rope_layout,
        )
        k_post_ref = _apply_rope_single(
            k_pre,
            pos=final_pos,
            theta=b.rope_theta,
            rotary_dim=b.rotary_dim,
            layout=b.rope_layout,
        )

        ok, msg = _compare_pair("post-RoPE q vs local reference", q_post_ref, q_post_rt, args.rope_tol)
        print(msg)
        if not ok:
            failures.append("post-RoPE q mismatch")
        print("")

        ok, msg = _compare_pair("post-RoPE k vs local reference", k_post_ref, k_post_rt, args.rope_tol)
        print(msg)
        if not ok:
            failures.append("post-RoPE k mismatch")
        print("")

        _run_decode_tokens_until(model_lib, base_ptr, lowered, tokens, b.kv_idx)
        k_cache_rt = _read_f32(base_ptr, k_cache_layer_abs, b.num_kv_heads * b.cache_capacity * b.head_dim)
        v_cache_rt = _read_f32(base_ptr, v_cache_layer_abs, b.num_kv_heads * b.cache_capacity * b.head_dim)
        k_cache_rt = k_cache_rt.reshape(b.num_kv_heads, b.cache_capacity, b.head_dim)[:, :kv_tokens, :]
        v_cache_rt = v_cache_rt.reshape(b.num_kv_heads, b.cache_capacity, b.head_dim)[:, :kv_tokens, :]

        k_slot_exp = np.zeros((b.num_kv_heads, b.head_dim), dtype=np.float32)
        v_slot_exp = np.zeros((b.num_kv_heads, b.head_dim), dtype=np.float32)
        k_copy = min(b.head_dim, k_post_ref.shape[1])
        v_copy = min(b.head_dim, v_pre.shape[1])
        k_slot_exp[:, :k_copy] = k_post_ref[:, :k_copy]
        v_slot_exp[:, :v_copy] = v_pre[:, :v_copy]

        ok, msg = _compare_pair(
            "KV cache k[current_pos] vs expected",
            k_slot_exp,
            k_cache_rt[:, final_pos, :],
            args.cache_tol,
        )
        print(msg)
        if not ok:
            failures.append("kv-cache k slot mismatch")
        print("")

        ok, msg = _compare_pair(
            "KV cache v[current_pos] vs expected",
            v_slot_exp,
            v_cache_rt[:, final_pos, :],
            args.cache_tol,
        )
        print(msg)
        if not ok:
            failures.append("kv-cache v slot mismatch")
        print("")

        _run_decode_tokens_until(model_lib, base_ptr, lowered, tokens, b.attn_idx)
        attn_rt = _read_f32(base_ptr, b.attn_abs, b.attn_count).reshape(b.num_heads, q_dim)

        if q_dim != b.head_dim:
            raise RuntimeError(
                f"checker currently expects q_dim == head_dim for attention reference (got {q_dim} vs {b.head_dim})"
            )

        if "f16kv" in b.attn_fn_name:
            attn_ref = _decode_attention_ref_f16kv(
                q_post_ref,
                k_cache_rt,
                v_cache_rt,
                num_heads=b.num_heads,
                num_kv_heads=b.num_kv_heads,
                head_dim=b.head_dim,
            )
        else:
            attn_ref = _decode_attention_ref(
                q_post_ref,
                k_cache_rt,
                v_cache_rt,
                num_heads=b.num_heads,
                num_kv_heads=b.num_kv_heads,
                head_dim=b.head_dim,
            )
        ok, msg = _compare_pair("attention output vs local reference", attn_ref, attn_rt, args.attn_tol)
        print(msg)
        if not ok:
            failures.append("attention output mismatch")
        print("")

        if failures:
            print("FAILED")
            for item in failures:
                print(f"  - {item}")
            return 1

        print("PASS: decode RoPE, KV-store, and attention output match local references.")
        return 0
    finally:
        os.environ.pop("CK_STOP_OP", None)


if __name__ == "__main__":
    raise SystemExit(main())
