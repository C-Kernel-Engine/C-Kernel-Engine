#!/usr/bin/env python3
"""
Focused layer-0 QKV contract checker (before RoPE/attention).

For q_proj/k_proj/v_proj this script:
1) Captures runtime attn_norm output (x) and projection outputs (y) from CK.
2) Recomputes projection outputs through CK parity kernels directly.
3) Recomputes projection outputs through llama.cpp parity helpers directly.
4) Compares tensors with explicit shape/order checks and fail-fast diagnostics.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


QK8_0 = 32
BLOCK_Q8_0_SIZE = 34

DEFAULT_MODEL_DIR = Path.home() / ".cache/ck-engine-v7/models/unsloth--gemma-3-270m-it-GGUF/ck_build"
PROJ_OPS = ("q_proj", "k_proj", "v_proj")


@dataclass
class OpBinding:
    op: str
    idx: int
    call_idx: int
    fn_name: str
    dtype: str
    rows: int
    cols: int
    x_abs: int
    y_abs: int
    y_rel: int
    w_name: str
    w_file_off: int
    w_size: int
    b_name: str
    b_file_off: int
    b_size: int


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    # Prefer roots that contain both parity artifacts and llama.cpp checkout.
    candidates = [Path.cwd(), *here.parents]
    for c in candidates:
        if (c / "build" / "libck_parity.so").exists() and (c / "llama.cpp").exists():
            return c
    # Fallback: any ancestor with llama.cpp.
    for c in candidates:
        if (c / "llama.cpp").exists():
            return c
    # Last resort for repo layout: .../version/v7/scripts/parity/<file>
    return here.parents[4]


def _parse_tokens_csv(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise RuntimeError("prefill token list is empty")
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


def _first_activation_offset(d: dict[str, Any] | None) -> int:
    if not isinstance(d, dict):
        raise RuntimeError("activation/output map missing")
    for v in d.values():
        if isinstance(v, dict) and "activation_offset" in v:
            return int(v["activation_offset"])
    raise RuntimeError("activation_offset not found in map")


def _resolve_abs_offset(lowered: dict[str, Any], rel_off: int) -> int:
    arena = lowered.get("memory", {}).get("arena", {})
    if str(arena.get("mode", "")) == "region":
        return int(arena.get("activations_base", 0)) + int(rel_off)
    return int(rel_off)


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
    rope_buf = None
    for buf in act.get("buffers", []):
        if buf.get("name") == "rope_cache":
            rope_buf = buf
            break

    def abs_offset(buf: dict[str, Any]) -> int:
        if "abs_offset" in buf:
            return int(buf["abs_offset"])
        return act_base + int(buf.get("offset", 0))

    if rope_buf is None:
        ctypes.memset(base_ptr + act_base, 0, act_size)
        return

    rope_off = abs_offset(rope_buf)
    rope_size = int(rope_buf.get("size", 0))
    if rope_size <= 0:
        ctypes.memset(base_ptr + act_base, 0, act_size)
        return

    pre = max(0, rope_off - act_base)
    if pre:
        ctypes.memset(base_ptr + act_base, 0, pre)
    end = act_base + act_size
    post_start = rope_off + rope_size
    if post_start < end:
        ctypes.memset(base_ptr + post_start, 0, end - post_start)


def _file_offset_from_binding(
    weights_base: int,
    bump_off: int,
    name: str,
    entry_by_name: dict[str, dict[str, Any]],
) -> int:
    ent = entry_by_name.get(name)
    if ent and ent.get("file_offset") is not None:
        return int(ent["file_offset"])
    return int(weights_base + bump_off)


def _weights_and_bias(
    weights_path: Path,
    binding: OpBinding,
) -> tuple[bytes, np.ndarray]:
    with weights_path.open("rb") as f:
        f.seek(binding.w_file_off)
        w_bytes = f.read(binding.w_size)
        if len(w_bytes) != binding.w_size:
            raise RuntimeError(f"{binding.op}: short read for weights ({len(w_bytes)} vs {binding.w_size})")

        if binding.b_file_off >= 0 and binding.b_size > 0:
            f.seek(binding.b_file_off)
            b = np.frombuffer(f.read(binding.b_size), dtype=np.float32).copy()
        else:
            b = np.zeros((binding.rows,), dtype=np.float32)

    if b.size != binding.rows:
        raise RuntimeError(f"{binding.op}: bias size mismatch {b.size} vs rows={binding.rows}")
    return w_bytes, b


def _load_parity_libs(root: Path) -> tuple[ctypes.CDLL, ctypes.CDLL]:
    ck_candidates = [
        root / "build" / "libck_parity.so",
        root / "libck_parity.so",
    ]
    ggml_candidates = [
        root / "llama.cpp" / "libggml_kernel_test.so",
        root / "llama.cpp" / "build" / "libggml_kernel_test.so",
    ]

    ck_lib = None
    for p in ck_candidates:
        if p.exists():
            ck_lib = ctypes.CDLL(str(p))
            break
    if ck_lib is None:
        raise FileNotFoundError("CK parity library not found (build/libck_parity.so)")

    ggml_lib = None
    for p in ggml_candidates:
        if p.exists():
            ggml_lib = ctypes.CDLL(str(p))
            break
    if ggml_lib is None:
        raise FileNotFoundError("llama.cpp parity library not found (llama.cpp/libggml_kernel_test.so)")

    if hasattr(ggml_lib, "test_init"):
        ggml_lib.test_init.argtypes = []
        ggml_lib.test_init.restype = None
        ggml_lib.test_init()

    required_ck = [
        "ck_test_gemv_q5_1",
        "ck_test_gemm_q5_1",
        "ck_test_gemv_q8_0",
        "ck_test_gemm_q8_0",
    ]
    required_ggml = [
        "test_gemv_q5_1",
        "test_gemm_q5_1",
        "test_gemv_q8_0",
        "test_gemm_q8_0",
    ]
    for sym in required_ck:
        if not hasattr(ck_lib, sym):
            raise RuntimeError(f"missing CK parity symbol: {sym}")
    for sym in required_ggml:
        if not hasattr(ggml_lib, sym):
            raise RuntimeError(f"missing llama parity symbol: {sym}")

    ck_lib.ck_test_gemv_q5_1.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
    ]
    ck_lib.ck_test_gemv_q5_1.restype = None
    ck_lib.ck_test_gemm_q5_1.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    ck_lib.ck_test_gemm_q5_1.restype = None
    ck_lib.ck_test_gemv_q8_0.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
    ]
    ck_lib.ck_test_gemv_q8_0.restype = None
    ck_lib.ck_test_gemm_q8_0.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    ck_lib.ck_test_gemm_q8_0.restype = None

    ggml_lib.test_gemv_q5_1.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
    ]
    ggml_lib.test_gemv_q5_1.restype = None
    ggml_lib.test_gemm_q5_1.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    ggml_lib.test_gemm_q5_1.restype = None
    ggml_lib.test_gemv_q8_0.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    ggml_lib.test_gemv_q8_0.restype = None
    ggml_lib.test_gemm_q8_0.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    ggml_lib.test_gemm_q8_0.restype = None

    return ck_lib, ggml_lib


def _load_model_lib(model_dir: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(model_dir / "libmodel.so"))
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
    if hasattr(lib, "ck_model_kv_cache_reset"):
        lib.ck_model_kv_cache_reset.argtypes = []
        lib.ck_model_kv_cache_reset.restype = None
    if hasattr(lib, "ck_model_free"):
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None
    return lib


def _run_decode_until(
    model_lib: ctypes.CDLL,
    base_ptr: int,
    lowered: dict[str, Any],
    token_id: int,
    stop_idx: int,
    read_off: int,
    read_count: int,
) -> np.ndarray:
    os.environ["CK_STOP_OP"] = str(stop_idx)
    if hasattr(model_lib, "ck_model_kv_cache_reset"):
        model_lib.ck_model_kv_cache_reset()
    _zero_activations_preserve_rope(base_ptr, lowered)
    rc = model_lib.ck_model_decode(ctypes.c_int32(token_id), None)
    if rc != 0:
        raise RuntimeError(f"ck_model_decode failed at CK_STOP_OP={stop_idx}, rc={rc}")
    return _read_f32(base_ptr, read_off, read_count)


def _run_prefill_until(
    model_lib: ctypes.CDLL,
    base_ptr: int,
    lowered: dict[str, Any],
    tokens: list[int],
    stop_idx: int,
    read_off: int,
    read_count: int,
) -> np.ndarray:
    os.environ["CK_STOP_OP"] = str(stop_idx)
    if hasattr(model_lib, "ck_model_kv_cache_reset"):
        model_lib.ck_model_kv_cache_reset()
    _zero_activations_preserve_rope(base_ptr, lowered)
    tok_arr = (ctypes.c_int32 * len(tokens))(*tokens)
    rc = model_lib.ck_model_embed_tokens(tok_arr, len(tokens))
    if rc != 0:
        raise RuntimeError(f"ck_model_embed_tokens failed at CK_STOP_OP={stop_idx}, rc={rc}")
    return _read_f32(base_ptr, read_off, read_count)


def _run_ck_projection(
    ck_lib: ctypes.CDLL,
    dtype: str,
    w_bytes: bytes,
    x: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    x = np.ascontiguousarray(x.astype(np.float32, copy=False))
    w_buf = (ctypes.c_uint8 * len(w_bytes)).from_buffer_copy(w_bytes)
    w_ptr = ctypes.cast(w_buf, ctypes.c_void_p)

    if x.ndim == 1:
        out = np.zeros((rows,), dtype=np.float32)
        if dtype == "q5_1":
            ck_lib.ck_test_gemv_q5_1(
                w_ptr,
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                rows,
                cols,
            )
        elif dtype == "q8_0":
            ck_lib.ck_test_gemv_q8_0(
                w_ptr,
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                rows,
                cols,
            )
        else:
            raise RuntimeError(f"unsupported dtype for CK projection: {dtype}")
        return out

    if x.ndim != 2:
        raise RuntimeError(f"invalid x rank for CK projection: {x.ndim}")

    n_tokens = int(x.shape[0])
    out2 = np.zeros((n_tokens, rows), dtype=np.float32)
    if dtype == "q5_1":
        ck_lib.ck_test_gemm_q5_1(
            w_ptr,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            cols,
            n_tokens,
        )
    elif dtype == "q8_0":
        ck_lib.ck_test_gemm_q8_0(
            w_ptr,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            cols,
            n_tokens,
        )
    else:
        raise RuntimeError(f"unsupported dtype for CK projection: {dtype}")
    return out2


def _run_llama_projection(
    ggml_lib: ctypes.CDLL,
    dtype: str,
    w_bytes: bytes,
    x: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    x = np.ascontiguousarray(x.astype(np.float32, copy=False))
    w_buf = (ctypes.c_uint8 * len(w_bytes)).from_buffer_copy(w_bytes)
    w_ptr = ctypes.cast(w_buf, ctypes.c_void_p)

    if x.ndim == 1:
        out = np.zeros((rows,), dtype=np.float32)
        if dtype == "q5_1":
            ggml_lib.test_gemv_q5_1(
                w_ptr,
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                rows,
                cols,
            )
            return out
        if dtype == "q8_0":
            row_bytes = (cols // QK8_0) * BLOCK_Q8_0_SIZE
            for r in range(rows):
                off = r * row_bytes
                row_view = (ctypes.c_uint8 * row_bytes).from_buffer_copy(w_bytes[off:off + row_bytes])
                y_tmp = np.zeros((1,), dtype=np.float32)
                ggml_lib.test_gemv_q8_0(
                    ctypes.cast(row_view, ctypes.c_void_p),
                    x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    y_tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    cols,
                )
                out[r] = y_tmp[0]
            return out
        raise RuntimeError(f"unsupported dtype for llama projection: {dtype}")

    if x.ndim != 2:
        raise RuntimeError(f"invalid x rank for llama projection: {x.ndim}")
    n_tokens = int(x.shape[0])
    out2 = np.zeros((n_tokens, rows), dtype=np.float32)

    if dtype == "q5_1":
        ggml_lib.test_gemm_q5_1(
            w_ptr,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_tokens,
            rows,
            cols,
        )
        return out2
    if dtype == "q8_0":
        ggml_lib.test_gemm_q8_0(
            w_ptr,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            cols,
            n_tokens,
        )
        return out2
    raise RuntimeError(f"unsupported dtype for llama projection: {dtype}")


def _diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, tuple[int, ...]]:
    d = np.abs(a - b)
    worst = int(np.argmax(d))
    return float(np.max(d)), float(np.mean(d)), tuple(np.unravel_index(worst, d.shape))


def _format_idx(idx: tuple[int, ...]) -> str:
    if len(idx) == 1:
        return f"[{idx[0]}]"
    return "[" + ",".join(str(i) for i in idx) + "]"


def _validate_shapes(
    pass_name: str,
    binding: OpBinding,
    x: np.ndarray,
    y: np.ndarray,
    n_tokens: int,
) -> None:
    if pass_name == "decode":
        if x.shape != (binding.cols,):
            raise RuntimeError(f"{binding.op}/{pass_name}: x shape {x.shape} != ({binding.cols},)")
        if y.shape != (binding.rows,):
            raise RuntimeError(f"{binding.op}/{pass_name}: y shape {y.shape} != ({binding.rows},)")
        return
    if x.shape != (n_tokens, binding.cols):
        raise RuntimeError(f"{binding.op}/{pass_name}: x shape {x.shape} != ({n_tokens},{binding.cols})")
    if y.shape != (n_tokens, binding.rows):
        raise RuntimeError(f"{binding.op}/{pass_name}: y shape {y.shape} != ({n_tokens},{binding.rows})")


def _parse_bindings(
    lowered: dict[str, Any],
    lowered_call: dict[str, Any],
    manifest: dict[str, Any],
    layer: int,
) -> tuple[int, int, dict[str, OpBinding]]:
    ops = lowered.get("operations", [])
    call_ops = lowered_call.get("operations", [])
    arena = lowered.get("memory", {}).get("arena", {})
    weights_base = int(arena.get("weights_base", 0))

    entry_by_name = {
        e.get("name"): e
        for e in manifest.get("entries", [])
        if isinstance(e, dict) and e.get("name")
    }

    attn_idx, attn_op = _find_layer_op_any(ops, layer, ("attn_norm", "rmsnorm"))
    if attn_idx is None or attn_op is None:
        raise RuntimeError(f"attn_norm/rmsnorm not found for layer={layer}")
    attn_out_rel = _first_activation_offset(attn_op.get("outputs"))
    attn_out_abs = _resolve_abs_offset(lowered, attn_out_rel)

    weight_map = {
        "q_proj": ("wq", "bq"),
        "k_proj": ("wk", "bk"),
        "v_proj": ("wv", "bv"),
    }

    bindings: dict[str, OpBinding] = {}
    for op_name in PROJ_OPS:
        idx, op = _find_layer_op(ops, layer, op_name)
        cidx, cop = _find_layer_op(call_ops, layer, op_name)
        if idx is None or op is None:
            raise RuntimeError(f"{op_name} not found for layer={layer}")
        if cidx is None or cop is None:
            raise RuntimeError(f"{op_name} call op not found for layer={layer}")

        w_key, b_key = weight_map[op_name]
        w_ref = op.get("weights", {}).get(w_key)
        b_ref = op.get("weights", {}).get(b_key)
        if not isinstance(w_ref, dict):
            raise RuntimeError(f"{op_name}: missing {w_key} binding")
        if not isinstance(b_ref, dict):
            raise RuntimeError(f"{op_name}: missing {b_key} binding")

        dtype = str(w_ref.get("dtype"))
        if dtype not in ("q5_1", "q8_0"):
            raise RuntimeError(f"{op_name}: unsupported dtype {dtype}")

        args = {a.get("name"): a.get("expr") for a in cop.get("args", [])}
        fn_name = str(cop.get("function"))
        try:
            if fn_name.startswith("gemv"):
                rows = int(args.get("M", "0"))
                cols = int(args.get("K", "0"))
            elif fn_name.startswith("gemm"):
                rows = int(args.get("N", "0"))
                cols = int(args.get("K", "0"))
            else:
                raise RuntimeError(f"{op_name}: unsupported function {fn_name}")
        except ValueError as e:
            raise RuntimeError(f"{op_name}: failed parsing call dims") from e
        if rows <= 0 or cols <= 0:
            raise RuntimeError(f"{op_name}: invalid dims rows={rows} cols={cols}")

        x_rel = _first_activation_offset(op.get("activations"))
        y_rel = _first_activation_offset(op.get("outputs"))
        x_abs = _resolve_abs_offset(lowered, x_rel)
        y_abs = _resolve_abs_offset(lowered, y_rel)

        w_name = str(w_ref.get("name"))
        b_name = str(b_ref.get("name"))
        w_file_off = _file_offset_from_binding(weights_base, int(w_ref["bump_offset"]), w_name, entry_by_name)
        b_file_off = _file_offset_from_binding(weights_base, int(b_ref["bump_offset"]), b_name, entry_by_name)

        bindings[op_name] = OpBinding(
            op=op_name,
            idx=int(idx),
            call_idx=int(cidx),
            fn_name=fn_name,
            dtype=dtype,
            rows=rows,
            cols=cols,
            x_abs=x_abs,
            y_abs=y_abs,
            y_rel=y_rel,
            w_name=w_name,
            w_file_off=w_file_off,
            w_size=int(w_ref["size"]),
            b_name=b_name,
            b_file_off=b_file_off,
            b_size=int(b_ref["size"]),
        )

    return int(attn_idx), int(attn_out_abs), bindings


def _compare_pair(
    label: str,
    expected: np.ndarray,
    actual: np.ndarray,
    tol: float,
) -> tuple[bool, str]:
    max_diff, mean_diff, idx = _diff_stats(expected, actual)
    ok = bool(max_diff <= tol and np.isfinite(expected).all() and np.isfinite(actual).all())
    idx_text = _format_idx(idx)
    msg = (
        f"{label}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
        f"worst={idx_text}, exp={float(expected[idx]):.8f}, got={float(actual[idx]):.8f}"
    )
    return ok, msg


def main() -> int:
    ap = argparse.ArgumentParser(description="Layer-0 QKV contract checker against CK+llama parity helpers")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="ck_build model directory")
    ap.add_argument("--layer", type=int, default=0, help="layer index")
    ap.add_argument("--decode-token", type=int, default=9259, help="decode token id")
    ap.add_argument("--prefill-tokens", default="2,9259", help="comma-separated prefill tokens")
    ap.add_argument("--tol", type=float, default=1e-3, help="max abs diff tolerance")
    ap.add_argument("--skip-decode", action="store_true", help="skip decode-path checks")
    ap.add_argument("--skip-prefill", action="store_true", help="skip prefill-path checks")
    ap.add_argument("--no-fail-fast", action="store_true", help="report all mismatches before exiting")
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    root = _find_project_root()
    fail_fast = not args.no_fail_fast
    prefill_tokens = _parse_tokens_csv(args.prefill_tokens)
    if len(prefill_tokens) < 2 and not args.skip_prefill:
        raise RuntimeError("prefill path needs at least 2 tokens")

    needed = [
        model_dir / "weights.bump",
        model_dir / "weights_manifest.json",
        model_dir / "libmodel.so",
        model_dir / "lowered_decode.json",
        model_dir / "lowered_decode_call.json",
        model_dir / "lowered_prefill.json",
        model_dir / "lowered_prefill_call.json",
    ]
    for p in needed:
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")

    manifest = _load_json(model_dir / "weights_manifest.json")
    lower_decode = _load_json(model_dir / "lowered_decode.json")
    call_decode = _load_json(model_dir / "lowered_decode_call.json")
    lower_prefill = _load_json(model_dir / "lowered_prefill.json")
    call_prefill = _load_json(model_dir / "lowered_prefill_call.json")

    decode_attn_idx, decode_attn_abs, decode_bindings = _parse_bindings(lower_decode, call_decode, manifest, args.layer)
    prefill_attn_idx, prefill_attn_abs, prefill_bindings = _parse_bindings(lower_prefill, call_prefill, manifest, args.layer)

    ck_lib, ggml_lib = _load_parity_libs(root)
    model_lib = _load_model_lib(model_dir)

    weights_path = model_dir / "weights.bump"
    model_rc = model_lib.ck_model_init(str(weights_path).encode())
    if model_rc != 0:
        raise RuntimeError(f"ck_model_init failed with code {model_rc}")

    base_ptr = int(model_lib.ck_model_get_base_ptr())
    if not base_ptr:
        raise RuntimeError("ck_model_get_base_ptr returned null")

    failures: list[str] = []
    try:
        print("=" * 92)
        print("LAYER-0 QKV CONTRACT CHECK (PRE-ROPE / PRE-ATTENTION)")
        print("=" * 92)
        print(f"model_dir      : {model_dir}")
        print(f"layer          : {args.layer}")
        print(f"decode token   : {args.decode_token}")
        print(f"prefill tokens : {prefill_tokens}")
        print(f"tolerance      : {args.tol:.2e}")
        print("")

        if not args.skip_decode:
            print("[decode] capturing runtime activations")
            x_decode = _run_decode_until(
                model_lib,
                base_ptr,
                lower_decode,
                args.decode_token,
                decode_attn_idx,
                decode_attn_abs,
                decode_bindings["q_proj"].cols,
            )
            y_decode: dict[str, np.ndarray] = {}
            for op_name in PROJ_OPS:
                b = decode_bindings[op_name]
                y_decode[op_name] = _run_decode_until(
                    model_lib,
                    base_ptr,
                    lower_decode,
                    args.decode_token,
                    b.idx,
                    b.y_abs,
                    b.rows,
                )

            for op_name in PROJ_OPS:
                b = decode_bindings[op_name]
                _validate_shapes("decode", b, x_decode, y_decode[op_name], 1)
                w_bytes, bias = _weights_and_bias(weights_path, b)
                y_ck = _run_ck_projection(ck_lib, b.dtype, w_bytes, x_decode, b.rows, b.cols) + bias
                y_llama = _run_llama_projection(ggml_lib, b.dtype, w_bytes, x_decode, b.rows, b.cols) + bias
                y_rt = y_decode[op_name]

                print(
                    f"[decode/{op_name}] idx={b.idx}, fn={b.fn_name}, dtype={b.dtype}, "
                    f"rows={b.rows}, cols={b.cols}, y_rel={b.y_rel}"
                )
                for label, exp, got in (
                    ("runtime vs ck-kernel", y_ck, y_rt),
                    ("runtime vs llama-ref", y_llama, y_rt),
                    ("ck-kernel vs llama-ref", y_llama, y_ck),
                ):
                    ok, msg = _compare_pair(f"  {label}", exp, got, args.tol)
                    print(msg)
                    if not ok:
                        failures.append(f"decode/{op_name}: {label} mismatch")
                        if fail_fast:
                            return 1
                print("")

        if not args.skip_prefill:
            print("[prefill] capturing runtime activations")
            n_tokens = len(prefill_tokens)
            x_pref_raw = _run_prefill_until(
                model_lib,
                base_ptr,
                lower_prefill,
                prefill_tokens,
                prefill_attn_idx,
                prefill_attn_abs,
                n_tokens * prefill_bindings["q_proj"].cols,
            )
            x_pref = x_pref_raw.reshape(n_tokens, prefill_bindings["q_proj"].cols)

            y_pref_raw: dict[str, np.ndarray] = {}
            y_pref_tok_major: dict[str, np.ndarray] = {}
            y_pref_row_major: dict[str, np.ndarray] = {}
            for op_name in PROJ_OPS:
                b = prefill_bindings[op_name]
                raw = _run_prefill_until(
                    model_lib,
                    base_ptr,
                    lower_prefill,
                    prefill_tokens,
                    b.idx,
                    b.y_abs,
                    n_tokens * b.rows,
                )
                y_pref_raw[op_name] = raw
                y_pref_tok_major[op_name] = raw.reshape(n_tokens, b.rows)
                y_pref_row_major[op_name] = raw.reshape(b.rows, n_tokens).T

            for op_name in PROJ_OPS:
                b = prefill_bindings[op_name]
                y_rt = y_pref_tok_major[op_name]
                _validate_shapes("prefill", b, x_pref, y_rt, n_tokens)
                w_bytes, bias = _weights_and_bias(weights_path, b)
                y_ck = _run_ck_projection(ck_lib, b.dtype, w_bytes, x_pref, b.rows, b.cols) + bias[np.newaxis, :]
                y_llama = _run_llama_projection(ggml_lib, b.dtype, w_bytes, x_pref, b.rows, b.cols) + bias[np.newaxis, :]

                print(
                    f"[prefill/{op_name}] idx={b.idx}, fn={b.fn_name}, dtype={b.dtype}, "
                    f"rows={b.rows}, cols={b.cols}, tokens={n_tokens}, y_rel={b.y_rel}"
                )
                checks = (
                    ("runtime(tok-major) vs ck-kernel", y_ck, y_rt),
                    ("runtime(tok-major) vs llama-ref", y_llama, y_rt),
                    ("ck-kernel vs llama-ref", y_llama, y_ck),
                )
                for label, exp, got in checks:
                    ok, msg = _compare_pair(f"  {label}", exp, got, args.tol)
                    print(msg)
                    if not ok and label.startswith("runtime(tok-major)"):
                        alt = y_pref_row_major[op_name]
                        alt_ok, alt_msg = _compare_pair(
                            "  runtime(row-major reinterpret) vs expected",
                            exp,
                            alt,
                            args.tol,
                        )
                        print(alt_msg)
                        if alt_ok:
                            print("  NOTE: row/token ordering mismatch suspected in runtime layout.")
                    if not ok:
                        failures.append(f"prefill/{op_name}: {label} mismatch")
                        if fail_fast:
                            return 1
                print("")

        if failures:
            print("FAILED")
            for f in failures:
                print(f"  - {f}")
            return 1

        print("PASS: layer-0 qkv projections match CK kernel and llama helper references.")
        return 0
    finally:
        os.environ.pop("CK_STOP_OP", None)
        if hasattr(model_lib, "ck_model_free"):
            model_lib.ck_model_free()


if __name__ == "__main__":
    raise SystemExit(main())
