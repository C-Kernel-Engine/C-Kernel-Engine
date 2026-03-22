#!/usr/bin/env python3
"""
Sequence-aware hidden-state checker against raw llama.cpp dumps.

This runs explicit token IDs through:
  1) llama.cpp helper in sequential decode mode with token-aware raw dumps
  2) CK runtime via ck_model_decode() with CK_STOP_OP stop-points

It compares layer-0 and footer hidden-state probes per token step to locate
the first real divergence on the sequential decode path.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from compare_first_token_logits import (
    discover_ck_model_dir,
    discover_gguf,
    ensure_llama_helper,
    parse_tokens_csv,
)


DEFAULT_TOKENS = "1,2,3,4,5"


@dataclass(frozen=True)
class Probe:
    raw_base_name: str
    stop_idx: int
    abs_offset: int
    full_count: int
    slice_start: int = 0
    slice_end: int | None = None
    raw_occurrence: int = 0
    label: str | None = None

    @property
    def output_count(self) -> int:
        end = self.slice_end if self.slice_end is not None else self.full_count
        return max(0, int(end) - int(self.slice_start))

    @property
    def probe_name(self) -> str:
        if self.label:
            return str(self.label)
        if self.raw_occurrence > 0:
            return f"{self.raw_base_name}@{self.raw_occurrence}"
        return self.raw_base_name


@dataclass(frozen=True)
class RawDumpEntry:
    raw_name: str
    path: Path
    occurrence: int
    elem_count: int


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_op(
    ops: list[dict[str, Any]],
    layer: int,
    op_name: str,
    occurrence: int = 0,
) -> tuple[int | None, dict[str, Any] | None]:
    idxs = [i for i, op in enumerate(ops) if op.get("layer") == layer and op.get("op") == op_name]
    if len(idxs) <= occurrence:
        return None, None
    idx = idxs[occurrence]
    return idx, ops[idx]


def find_op_any(
    ops: list[dict[str, Any]],
    layer: int,
    op_names: tuple[str, ...],
    occurrence: int = 0,
) -> tuple[int | None, dict[str, Any] | None]:
    for op_name in op_names:
        idx, op = find_op(ops, layer, op_name, occurrence)
        if idx is not None and op is not None:
            return idx, op
    return None, None


def find_named_or_fallback(
    ops: list[dict[str, Any]],
    layer: int,
    preferred_names: tuple[str, ...],
    fallback_name: str,
    fallback_occurrence: int,
) -> tuple[int | None, dict[str, Any] | None]:
    idx, op = find_op_any(ops, layer, preferred_names, 0)
    if idx is not None and op is not None:
        return idx, op
    return find_op(ops, layer, fallback_name, fallback_occurrence)


def resolve_abs_offset(lowered: dict[str, Any], rel_off: int) -> int:
    arena = lowered.get("memory", {}).get("arena", {})
    if str(arena.get("mode", "")) == "region":
        return int(arena.get("activations_base", 0)) + int(rel_off)
    return int(rel_off)


def first_output_abs(lowered: dict[str, Any], op: dict[str, Any], keys: tuple[str, ...]) -> int:
    outputs = op.get("outputs", {})
    for key in keys:
        binding = outputs.get(key)
        if isinstance(binding, dict) and "activation_offset" in binding:
            return resolve_abs_offset(lowered, int(binding["activation_offset"]))
    raise RuntimeError(f"activation output binding not found for op={op.get('op')} keys={keys}")


def load_model_lib(model_dir: Path) -> ctypes.CDLL:
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


def read_f32(base_ptr: int, abs_off: int, count: int) -> np.ndarray:
    ptr = ctypes.cast(base_ptr + abs_off, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(count,)).copy()


def zero_activations_preserve_rope(base_ptr: int, lowered: dict[str, Any]) -> None:
    memory = lowered.get("memory", {})
    arena = memory.get("arena", {})
    activations = memory.get("activations", {})
    act_size = int(activations.get("size", 0))
    if act_size <= 0:
        return

    act_base = int(arena.get("activations_base", 0))

    def abs_offset(buf: dict[str, Any]) -> int:
        if "abs_offset" in buf:
            return int(buf["abs_offset"])
        return act_base + int(buf.get("offset", 0))

    protected: list[tuple[int, int]] = []
    for buf in activations.get("buffers", []):
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
    os.environ["CK_STOP_OP"] = str(int(stop_idx))
    zero_activations_preserve_rope(base_ptr, lowered)
    rc = model_lib.ck_model_decode(ctypes.c_int32(int(tokens[-1])), None)
    if rc != 0:
        raise RuntimeError(f"final ck_model_decode failed rc={rc} CK_STOP_OP={stop_idx}")


def parse_raw_index(
    index_path: Path,
) -> tuple[dict[int, dict[str, list[RawDumpEntry]]], dict[tuple[int, str], int]]:
    dumps: dict[int, dict[str, list[RawDumpEntry]]] = {}
    duplicates: dict[tuple[int, str], int] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        token_id = int(row.get("token_id", 0))
        base_name = str(row.get("base_name") or row.get("name") or "")
        raw_name = str(row.get("name") or "")
        if not base_name or not raw_name:
            continue
        file_path = index_path.parent / f"{raw_name}.bin"
        if not file_path.exists():
            continue
        by_token = dumps.setdefault(token_id, {})
        entries = by_token.setdefault(base_name, [])
        occurrence = row.get("occurrence", len(entries))
        try:
            occurrence = int(occurrence)
        except Exception:
            occurrence = len(entries)
        key = (token_id, base_name)
        duplicates[key] = duplicates.get(key, 0) + 1
        entries.append(
            RawDumpEntry(
                raw_name=raw_name,
                path=file_path,
                occurrence=occurrence,
                elem_count=int(row.get("elem_count", 0)),
            )
        )
    return dumps, duplicates


def resolve_raw_dump(raw_by_name: dict[str, list[RawDumpEntry]], probe: Probe) -> RawDumpEntry | None:
    entries = raw_by_name.get(probe.raw_base_name, [])
    if not entries:
        return None
    for entry in sorted(entries, key=lambda item: item.occurrence):
        if int(entry.occurrence) == int(probe.raw_occurrence):
            return entry
    return None


def run_llama_raw_dumps(
    gguf: Path,
    tokens: list[int],
    ctx_len: int,
    threads: int,
    top_k: int,
    work_dir: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, list[RawDumpEntry]]], dict[tuple[int, str], int]]:
    helper = ensure_llama_helper()
    dump_dir = work_dir / "llama_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    for stale in dump_dir.glob("*"):
        try:
            stale.unlink()
        except OSError:
            pass
    cmd = [
        str(helper),
        "--model",
        str(gguf),
        "--tokens",
        ",".join(str(t) for t in tokens),
        "--ctx",
        str(int(ctx_len)),
        "--top-k",
        str(int(top_k)),
        "--decode-mode",
        "sequential",
        "--dump-dir",
        str(dump_dir),
        "--logits-out",
        str(work_dir / "llama_logits.f32"),
    ]
    if threads > 0:
        cmd.extend(["--threads", str(int(threads))])
    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "llama_token_replay failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"rc: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )
    meta = json.loads(proc.stdout.strip())
    if not isinstance(meta, dict) or not meta.get("ok"):
        raise RuntimeError(f"llama_token_replay returned invalid payload: {proc.stdout.strip()}")
    index_path = dump_dir / "index.json"
    if not index_path.exists():
        raise RuntimeError(f"llama raw dump index missing: {index_path}")
    dump_map, duplicates = parse_raw_index(index_path)
    return meta, dump_map, duplicates


def make_probe_specs(lowered: dict[str, Any], lowered_call: dict[str, Any]) -> list[Probe]:
    ops = lowered.get("operations", [])
    call_ops = lowered_call.get("operations", [])

    embed_dim = int(lowered.get("config", {}).get("embed_dim", 0))
    if embed_dim <= 0:
        raise RuntimeError("invalid embed_dim in lowered decode config")

    res0_stop, res0_op = find_op(call_ops, 0, "residual_save", 0)
    attn_aliases = ("attn", "attn_sliding")
    mlp_act_aliases = ("silu_mul", "geglu")
    attn_norm_stop, attn_norm_op = find_named_or_fallback(call_ops, 0, ("attn_norm",), "rmsnorm", 0)
    q_stop, q_op = find_op(call_ops, 0, "q_proj", 0)
    k_stop, k_op = find_op(call_ops, 0, "k_proj", 0)
    v_stop, v_op = find_op(call_ops, 0, "v_proj", 0)
    rope_stop, rope_op = find_op(call_ops, 0, "rope_qk", 0)
    attn_stop, attn_op = find_op_any(call_ops, 0, attn_aliases, 0)
    out_stop, out_op = find_op(call_ops, 0, "out_proj", 0)
    ffn_inp_stop, ffn_inp_op = find_op(call_ops, 0, "residual_add", 0)
    ffn_norm_stop, ffn_norm_op = find_named_or_fallback(call_ops, 0, ("ffn_norm",), "rmsnorm", 1)
    gate_up_stop, gate_up_op = find_op(call_ops, 0, "mlp_gate_up", 0)
    swiglu_stop, swiglu_op = find_op_any(call_ops, 0, mlp_act_aliases, 0)
    down_stop, down_op = find_op(call_ops, 0, "mlp_down", 0)
    l_out_stop, l_out_op = find_op(call_ops, 0, "residual_add", 1)
    final_norm_stop, final_norm_op = find_named_or_fallback(call_ops, -1, ("final_rmsnorm",), "rmsnorm", 0)
    logits_stop, logits_op = find_op(call_ops, -1, "logits", 0)

    res0_idx, res0_meta = find_op(ops, 0, "residual_save", 0)
    attn_norm_idx, attn_norm_meta = find_named_or_fallback(ops, 0, ("attn_norm",), "rmsnorm", 0)
    q_idx, q_meta = find_op(ops, 0, "q_proj", 0)
    k_idx, k_meta = find_op(ops, 0, "k_proj", 0)
    v_idx, v_meta = find_op(ops, 0, "v_proj", 0)
    rope_idx, rope_meta = find_op(ops, 0, "rope_qk", 0)
    attn_idx, attn_meta = find_op_any(ops, 0, attn_aliases, 0)
    out_idx, out_meta = find_op(ops, 0, "out_proj", 0)
    ffn_inp_idx, ffn_inp_meta = find_op(ops, 0, "residual_add", 0)
    ffn_norm_idx, ffn_norm_meta = find_named_or_fallback(ops, 0, ("ffn_norm",), "rmsnorm", 1)
    gate_up_idx, gate_up_meta = find_op(ops, 0, "mlp_gate_up", 0)
    swiglu_idx, swiglu_meta = find_op_any(ops, 0, mlp_act_aliases, 0)
    down_idx, down_meta = find_op(ops, 0, "mlp_down", 0)
    l_out_idx, l_out_meta = find_op(ops, 0, "residual_add", 1)
    final_norm_idx, final_norm_meta = find_named_or_fallback(ops, -1, ("final_rmsnorm",), "rmsnorm", 0)
    logits_idx, logits_meta = find_op(ops, -1, "logits", 0)

    required = {
        "res0": (res0_stop, res0_idx, res0_meta),
        "attn_norm": (attn_norm_stop, attn_norm_idx, attn_norm_meta),
        "q_proj": (q_stop, q_idx, q_meta),
        "k_proj": (k_stop, k_idx, k_meta),
        "v_proj": (v_stop, v_idx, v_meta),
        "rope_qk": (rope_stop, rope_idx, rope_meta),
        "attn": (attn_stop, attn_idx, attn_meta),
        "out_proj": (out_stop, out_idx, out_meta),
        "ffn_inp": (ffn_inp_stop, ffn_inp_idx, ffn_inp_meta),
        "ffn_norm": (ffn_norm_stop, ffn_norm_idx, ffn_norm_meta),
        "mlp_gate_up": (gate_up_stop, gate_up_idx, gate_up_meta),
        "silu_mul": (swiglu_stop, swiglu_idx, swiglu_meta),
        "mlp_down": (down_stop, down_idx, down_meta),
        "l_out": (l_out_stop, l_out_idx, l_out_meta),
        "final_norm": (final_norm_stop, final_norm_idx, final_norm_meta),
        "logits": (logits_stop, logits_idx, logits_meta),
    }
    missing = [name for name, vals in required.items() if any(v is None for v in vals)]
    if missing:
        raise RuntimeError(f"missing probe ops in lowered decode: {', '.join(missing)}")

    gate_up_dim = int(gate_up_meta.get("params", {}).get("_output_dim", 0))
    inter_dim = gate_up_dim // 2
    if gate_up_dim <= 0 or inter_dim <= 0 or gate_up_dim % 2 != 0:
        raise RuntimeError(f"invalid mlp_gate_up output dim: {gate_up_dim}")

    swiglu_dim = int(swiglu_meta.get("params", {}).get("intermediate_size", 0))
    if swiglu_dim <= 0:
        swiglu_dim = inter_dim

    vocab_size = int(logits_meta.get("params", {}).get("_output_dim", 0))
    if vocab_size <= 0:
        raise RuntimeError(f"invalid logits output dim: {vocab_size}")

    q_dim = int(q_meta.get("params", {}).get("_output_dim", 0))
    k_dim = int(k_meta.get("params", {}).get("_output_dim", 0))
    v_dim = int(v_meta.get("params", {}).get("_output_dim", 0))
    if q_dim <= 0 or k_dim <= 0 or v_dim <= 0:
        raise RuntimeError(f"invalid qkv output dims: q={q_dim} k={k_dim} v={v_dim}")

    q_abs = first_output_abs(lowered, q_meta, ("y", "out", "q"))
    k_abs = first_output_abs(lowered, k_meta, ("y", "out", "k"))
    v_abs = first_output_abs(lowered, v_meta, ("y", "out", "v"))

    return [
        Probe("inp_embd", int(res0_stop), first_output_abs(lowered, res0_meta, ("dst",)), embed_dim),
        Probe("attn_norm-0", int(attn_norm_stop), first_output_abs(lowered, attn_norm_meta, ("output", "out", "y")), embed_dim),
        Probe("Qcur-0", int(q_stop), q_abs, q_dim, raw_occurrence=0, label="Qcur-0@0"),
        Probe("Qcur-0", int(rope_stop), q_abs, q_dim, raw_occurrence=1, label="Qcur-0@1"),
        Probe("Kcur-0", int(k_stop), k_abs, k_dim, raw_occurrence=0, label="Kcur-0@0"),
        Probe("Kcur-0", int(rope_stop), k_abs, k_dim, raw_occurrence=1, label="Kcur-0@1"),
        Probe("Vcur-0", int(v_stop), v_abs, v_dim),
        Probe("__fattn__-0", int(attn_stop), first_output_abs(lowered, attn_meta, ("out_token", "out", "y")), embed_dim),
        Probe("attn_out-0", int(out_stop), first_output_abs(lowered, out_meta, ("y", "out", "C")), embed_dim),
        Probe("ffn_inp-0", int(ffn_inp_stop), first_output_abs(lowered, ffn_inp_meta, ("out", "output")), embed_dim),
        Probe("ffn_norm-0", int(ffn_norm_stop), first_output_abs(lowered, ffn_norm_meta, ("output", "out", "y")), embed_dim),
        Probe("ffn_gate-0", int(gate_up_stop), first_output_abs(lowered, gate_up_meta, ("y", "out")), gate_up_dim, 0, inter_dim),
        Probe("ffn_up-0", int(gate_up_stop), first_output_abs(lowered, gate_up_meta, ("y", "out")), gate_up_dim, inter_dim, gate_up_dim),
        Probe("ffn_swiglu-0", int(swiglu_stop), first_output_abs(lowered, swiglu_meta, ("out", "output", "y")), swiglu_dim),
        Probe("ffn_out-0", int(down_stop), first_output_abs(lowered, down_meta, ("y", "out", "C")), embed_dim),
        Probe("l_out-0", int(l_out_stop), first_output_abs(lowered, l_out_meta, ("out", "output")), embed_dim),
        Probe("result_norm", int(final_norm_stop), first_output_abs(lowered, final_norm_meta, ("output", "out", "y")), embed_dim),
        Probe("result_output", int(logits_stop), first_output_abs(lowered, logits_meta, ("y", "out", "C")), vocab_size),
    ]


def read_probe(base_ptr: int, probe: Probe) -> np.ndarray:
    full = read_f32(base_ptr, probe.abs_offset, probe.full_count)
    end = probe.slice_end if probe.slice_end is not None else probe.full_count
    return full[int(probe.slice_start):int(end)].copy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a32 = a.astype(np.float32, copy=False).reshape(-1)
    b32 = b.astype(np.float32, copy=False).reshape(-1)
    denom = float(np.linalg.norm(a32) * np.linalg.norm(b32))
    if denom == 0.0:
        return 1.0 if np.allclose(a32, b32) else 0.0
    return float(np.dot(a32, b32) / denom)


def compare_arrays(ref: np.ndarray, got: np.ndarray, atol: float, rtol: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "shape_ref": list(ref.shape),
        "shape_got": list(got.shape),
        "ok": False,
    }
    if ref.shape != got.shape:
        out["error"] = f"shape mismatch ref={ref.shape} got={got.shape}"
        return out
    diff = np.abs(ref.astype(np.float32) - got.astype(np.float32))
    max_diff = float(np.max(diff)) if diff.size else 0.0
    mean_diff = float(np.mean(diff)) if diff.size else 0.0
    worst = int(np.argmax(diff)) if diff.size else 0
    ref_flat = ref.reshape(-1)
    got_flat = got.reshape(-1)
    out.update(
        {
            "ok": bool(np.allclose(ref, got, atol=atol, rtol=rtol)),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "worst_idx": worst,
            "ref_value": float(ref_flat[worst]) if ref_flat.size else 0.0,
            "got_value": float(got_flat[worst]) if got_flat.size else 0.0,
            "cosine": cosine_similarity(ref, got),
        }
    )
    if ref_flat.size:
        out["top1_ref"] = int(np.argmax(ref_flat))
        out["top1_got"] = int(np.argmax(got_flat))
        out["top1_match"] = bool(out["top1_ref"] == out["top1_got"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Sequence-aware hidden-state parity vs llama.cpp")
    ap.add_argument("--model-dir", required=True, type=Path, help="model dir or parent containing .ck_build")
    ap.add_argument("--gguf", type=Path, default=None, help="GGUF path (auto-discovered if omitted)")
    ap.add_argument("--tokens", default=DEFAULT_TOKENS, help="comma-separated explicit token IDs")
    ap.add_argument("--ctx-len", type=int, default=1024, help="context length for llama helper")
    ap.add_argument("--threads", type=int, default=0, help="llama helper threads (0 = auto)")
    ap.add_argument("--top-k", type=int, default=8, help="top-k emitted by llama helper")
    ap.add_argument("--atol", type=float, default=1e-3, help="absolute tolerance for pass/fail")
    ap.add_argument("--rtol", type=float, default=1e-3, help="relative tolerance for pass/fail")
    ap.add_argument("--report-json", type=Path, default=None, help="optional report JSON path")
    ap.add_argument("--keep-workdir", type=Path, default=None, help="keep llama raw dumps in this directory")
    args = ap.parse_args()

    model_dir = discover_ck_model_dir(args.model_dir)
    gguf = discover_gguf(args.gguf, model_dir)
    tokens = parse_tokens_csv(args.tokens)

    lowered = load_json(model_dir / "lowered_decode.json")
    lowered_call = load_json(model_dir / "lowered_decode_call.json")
    probes = make_probe_specs(lowered, lowered_call)

    if args.keep_workdir is not None:
        work_dir = args.keep_workdir.expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        meta, dump_map, duplicates = run_llama_raw_dumps(
            gguf, tokens, args.ctx_len, args.threads, args.top_k, work_dir
        )
    else:
        with tempfile.TemporaryDirectory(prefix="ckv7_seq_hidden_") as td:
            work_dir = Path(td)
            meta, dump_map, duplicates = run_llama_raw_dumps(
                gguf, tokens, args.ctx_len, args.threads, args.top_k, work_dir
            )
            report = _run_ck_compare(
                model_dir=model_dir,
                lowered=lowered,
                probes=probes,
                tokens=tokens,
                dump_map=dump_map,
                duplicates=duplicates,
                meta=meta,
                atol=args.atol,
                rtol=args.rtol,
                report_json=args.report_json,
                llama_work_dir=work_dir,
            )
            return 0 if report.get("all_ok", False) else 1

    report = _run_ck_compare(
        model_dir=model_dir,
        lowered=lowered,
        probes=probes,
        tokens=tokens,
        dump_map=dump_map,
        duplicates=duplicates,
        meta=meta,
        atol=args.atol,
        rtol=args.rtol,
        report_json=args.report_json,
        llama_work_dir=work_dir,
    )
    return 0 if report.get("all_ok", False) else 1


def _run_ck_compare(
    *,
    model_dir: Path,
    lowered: dict[str, Any],
    probes: list[Probe],
    tokens: list[int],
    dump_map: dict[int, dict[str, list[RawDumpEntry]]],
    duplicates: dict[tuple[int, str], int],
    meta: dict[str, Any],
    atol: float,
    rtol: float,
    report_json: Path | None,
    llama_work_dir: Path,
) -> dict[str, Any]:
    lib = load_model_lib(model_dir)
    if lib.ck_model_init(str(model_dir / "weights.bump").encode("utf-8")) != 0:
        raise RuntimeError("ck_model_init failed")
    base_ptr = int(lib.ck_model_get_base_ptr())
    if not base_ptr:
        raise RuntimeError("ck_model_get_base_ptr returned null")

    report: dict[str, Any] = {
        "model_dir": str(model_dir),
        "llama_work_dir": str(llama_work_dir),
        "tokens": [int(t) for t in tokens],
        "llama_meta": meta,
        "duplicate_raw_dumps": {
            f"tok{tok}:{name}": int(count)
            for (tok, name), count in sorted(duplicates.items())
            if count > 1
        },
        "results": [],
        "all_ok": True,
        "first_failure": None,
        "first_stable_failure": None,
    }

    try:
        for token_idx in sorted(dump_map.keys()):
            prefix_tokens = tokens[: token_idx + 1]
            raw_by_name = dump_map.get(token_idx, {})
            for probe in probes:
                raw_entry = resolve_raw_dump(raw_by_name, probe)
                dup_key = (int(token_idx), str(probe.raw_base_name))
                ambiguous_raw_name = int(duplicates.get(dup_key, 0)) > 1
                row: dict[str, Any] = {
                    "token_step": int(token_idx),
                    "probe": probe.probe_name,
                    "stop_idx": int(probe.stop_idx),
                    "raw_file": str(raw_entry.path) if raw_entry else None,
                    "raw_occurrence": int(probe.raw_occurrence),
                    "ambiguous_raw_name": bool(ambiguous_raw_name),
                }
                if raw_entry is None:
                    row["status"] = "MISSING"
                    report["results"].append(row)
                    report["all_ok"] = False
                    if report["first_failure"] is None:
                        report["first_failure"] = {
                            "token_step": int(token_idx),
                            "probe": probe.probe_name,
                            "reason": "missing raw dump",
                        }
                    if (not ambiguous_raw_name) and report["first_stable_failure"] is None:
                        report["first_stable_failure"] = {
                            "token_step": int(token_idx),
                            "probe": probe.probe_name,
                            "reason": "missing raw dump",
                        }
                    continue

                ref = np.fromfile(raw_entry.path, dtype=np.float32)
                run_decode_tokens_until(lib, base_ptr, lowered, prefix_tokens, probe.stop_idx)
                got = read_probe(base_ptr, probe)
                cmp = compare_arrays(ref, got, atol=atol, rtol=rtol)
                row.update(cmp)
                row["status"] = "PASS" if cmp.get("ok") else "FAIL"
                report["results"].append(row)
                if not cmp.get("ok"):
                    report["all_ok"] = False
                    if report["first_failure"] is None:
                        report["first_failure"] = {
                            "token_step": int(token_idx),
                            "probe": probe.probe_name,
                            "max_diff": cmp.get("max_diff"),
                            "mean_diff": cmp.get("mean_diff"),
                            "cosine": cmp.get("cosine"),
                        }
                    if (not ambiguous_raw_name) and report["first_stable_failure"] is None:
                        report["first_stable_failure"] = {
                            "token_step": int(token_idx),
                            "probe": probe.probe_name,
                            "max_diff": cmp.get("max_diff"),
                            "mean_diff": cmp.get("mean_diff"),
                            "cosine": cmp.get("cosine"),
                        }
    finally:
        os.environ.pop("CK_STOP_OP", None)
        if hasattr(lib, "ck_model_free"):
            lib.ck_model_free()

    print("=" * 96)
    print("SEQUENCE-AWARE HIDDEN-STATE PARITY")
    print("=" * 96)
    print(f"model_dir       : {model_dir}")
    print(f"llama_work_dir  : {llama_work_dir}")
    print(f"tokens          : {tokens}")
    print(f"decode_mode     : {meta.get('decode_mode')}")
    print(f"atol/rtol       : {atol}/{rtol}")
    if report["duplicate_raw_dumps"]:
        print("duplicate raw names:")
        for name, count in report["duplicate_raw_dumps"].items():
            print(f"  {name} -> {count}")
    print("")
    for row in report["results"]:
        status = row.get("status")
        token_step = row.get("token_step")
        probe = row.get("probe")
        if status == "PASS":
            print(
                f"[PASS] token={token_step:02d} probe={probe:<14} "
                f"max={row.get('max_diff', 0.0):.6e} "
                f"mean={row.get('mean_diff', 0.0):.6e} "
                f"cos={row.get('cosine', 0.0):.6f}"
            )
        elif status == "FAIL":
            print(
                f"[FAIL] token={token_step:02d} probe={probe:<14} "
                f"max={row.get('max_diff', 0.0):.6e} "
                f"mean={row.get('mean_diff', 0.0):.6e} "
                f"cos={row.get('cosine', 0.0):.6f}"
            )
        else:
            print(f"[{status}] token={token_step:02d} probe={probe:<14} raw dump unavailable")
    print("")
    print(f"all_ok          : {report['all_ok']}")
    print(f"first_failure   : {json.dumps(report['first_failure'])}")
    print(f"first_stable    : {json.dumps(report['first_stable_failure'])}")
    print("=" * 96)

    if report_json is not None:
        out = report_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report_json     : {out}")

    return report


if __name__ == "__main__":
    raise SystemExit(main())
