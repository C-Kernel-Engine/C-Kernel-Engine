#!/usr/bin/env python3
"""
Tokenizer-free first-token logits parity probe.

Runs the same explicit token IDs through:
1) CK runtime (`libmodel.so` via ck_model_embed_tokens + ck_model_forward)
2) llama.cpp runtime (small helper linked to libllama, no tokenizer path)

Writes a compact JSON report with top-k overlap and full-logits diff stats.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
LLAMA_CPP = ROOT / "llama.cpp"
HELPER_SRC = SCRIPT_DIR / "llama_token_replay.cpp"
HELPER_BIN = Path("/tmp/ckv7_llama_token_replay")


def parse_tokens_csv(text: str) -> list[int]:
    tokens: list[int] = []
    for part in str(text or "").split(","):
        p = part.strip()
        if not p:
            continue
        tokens.append(int(p))
    if not tokens:
        raise ValueError("token list is empty")
    return tokens


def discover_ck_model_dir(path: Path) -> Path:
    p = path.expanduser().resolve()
    # Prefer isolated build dirs first; parent run-dir copies can be stale.
    candidates = [p / ".ck_build", p / "ck_build", p]
    for c in candidates:
        if (c / "libmodel.so").exists() and (c / "weights.bump").exists():
            return c
    raise FileNotFoundError(f"Could not find CK model runtime dir under: {p}")


def discover_gguf(path: Path | None, model_dir: Path) -> Path:
    if path is not None:
        p = path.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"GGUF not found: {p}")
        return p

    local = sorted(model_dir.glob("*.gguf"))
    if local:
        return local[0].resolve()

    parent_local = sorted(model_dir.parent.glob("*.gguf"))
    if parent_local:
        return parent_local[0].resolve()

    raise FileNotFoundError("Unable to locate GGUF; pass --gguf explicitly")


def load_runtime_contract(model_dir: Path) -> dict[str, Any]:
    contract: dict[str, Any] = {"prefill_policy": "batched"}
    candidates = [
        model_dir / "config.json",
        model_dir / "weights_manifest.json",
        (model_dir.parent / "config.json") if model_dir.name in {".ck_build", "ck_build"} else None,
        (model_dir.parent / "weights_manifest.json") if model_dir.name in {".ck_build", "ck_build"} else None,
    ]
    for path in candidates:
        if path is None or not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        cfg = data.get("config") if isinstance(data.get("config"), dict) else data
        if isinstance(cfg, dict):
            explicit_prefill = cfg.get("prefill_policy")
            if isinstance(explicit_prefill, str) and explicit_prefill.strip():
                contract["prefill_policy"] = explicit_prefill.strip()
            layer_kinds = cfg.get("layer_kinds")
            if (
                contract["prefill_policy"] == "batched"
                and isinstance(layer_kinds, list)
                and any(str(kind).strip().lower() not in {"full_attention", "attention", "dense_attention"} for kind in layer_kinds)
            ):
                contract["prefill_policy"] = "sequential_decode"
        template = data.get("template")
        if isinstance(template, dict):
            flags = template.get("flags")
            if isinstance(flags, dict):
                template_prefill = flags.get("prefill_policy")
                if isinstance(template_prefill, str) and template_prefill.strip():
                    contract["prefill_policy"] = template_prefill.strip()
    return contract


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def ensure_llama_helper() -> Path:
    if not HELPER_SRC.exists():
        raise FileNotFoundError(f"Missing helper source: {HELPER_SRC}")

    rebuild = True
    if HELPER_BIN.exists():
        try:
            rebuild = HELPER_BIN.stat().st_mtime < HELPER_SRC.stat().st_mtime
        except OSError:
            rebuild = True
    if not rebuild:
        return HELPER_BIN

    include_dir = LLAMA_CPP / "include"
    ggml_include_dir = LLAMA_CPP / "ggml" / "include"
    lib_dir = LLAMA_CPP / "build" / "bin"
    if not include_dir.exists():
        raise FileNotFoundError(f"llama include dir missing: {include_dir}")
    if not ggml_include_dir.exists():
        raise FileNotFoundError(f"ggml include dir missing: {ggml_include_dir}")
    if not lib_dir.exists():
        raise FileNotFoundError(f"llama lib dir missing: {lib_dir}")

    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        str(HELPER_SRC),
        "-I",
        str(include_dir),
        "-I",
        str(ggml_include_dir),
        "-L",
        str(lib_dir),
        f"-Wl,-rpath,{lib_dir}",
        "-lllama",
        "-lggml",
        "-lggml-cpu",
        "-lggml-base",
        "-pthread",
        "-ldl",
        "-o",
        str(HELPER_BIN),
    ]
    proc = _run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to compile llama_token_replay helper\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )
    HELPER_BIN.chmod(0o755)
    return HELPER_BIN


def run_llama_logits(gguf_path: Path, tokens: list[int], ctx_len: int, top_k: int, threads: int) -> dict[str, Any]:
    helper = ensure_llama_helper()
    with tempfile.TemporaryDirectory(prefix="llama_token_replay_") as td:
        logits_path = Path(td) / "llama_logits.f32"
        cmd = [
            str(helper),
            "--model",
            str(gguf_path),
            "--tokens",
            ",".join(str(t) for t in tokens),
            "--ctx",
            str(int(ctx_len)),
            "--top-k",
            str(int(top_k)),
            "--logits-out",
            str(logits_path),
        ]
        if threads > 0:
            cmd.extend(["--threads", str(int(threads))])
        proc = _run(cmd, cwd=ROOT)
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

        n_vocab = int(meta.get("n_vocab", 0))
        logits = np.fromfile(logits_path, dtype=np.float32)
        if logits.size != n_vocab:
            raise RuntimeError(f"llama logits size mismatch: got={logits.size} expected={n_vocab}")
        return {
            "meta": meta,
            "logits": logits,
        }


def load_ck_logits(model_dir: Path, tokens: list[int]) -> dict[str, Any]:
    lib_path = model_dir / "libmodel.so"
    lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int
    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    has_stride = hasattr(lib, "ck_model_get_logits_stride")
    if has_stride:
        lib.ck_model_get_logits_stride.argtypes = []
        lib.ck_model_get_logits_stride.restype = ctypes.c_int
    has_active = hasattr(lib, "ck_model_get_active_tokens")
    if has_active:
        lib.ck_model_get_active_tokens.argtypes = []
        lib.ck_model_get_active_tokens.restype = ctypes.c_int
    has_decode = hasattr(lib, "ck_model_decode")
    if has_decode:
        lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        lib.ck_model_decode.restype = ctypes.c_int
    has_kv_reset = hasattr(lib, "ck_model_kv_cache_reset")
    if has_kv_reset:
        lib.ck_model_kv_cache_reset.argtypes = []
        lib.ck_model_kv_cache_reset.restype = None
    has_free = hasattr(lib, "ck_model_free")
    if has_free:
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None

    init_candidates = [model_dir / "weights.bump", model_dir]
    if model_dir.name in {".ck_build", "ck_build"}:
        init_candidates.extend([model_dir.parent / "weights.bump", model_dir.parent])
    # Preserve order while removing duplicates.
    deduped: list[Path] = []
    for c in init_candidates:
        c = c.resolve()
        if c not in deduped:
            deduped.append(c)
    init_candidates = deduped
    init_ok = False
    init_dir_used = None
    init_errors: list[str] = []
    for init_dir in init_candidates:
        rc = lib.ck_model_init(str(init_dir).encode("utf-8"))
        if rc == 0:
            init_ok = True
            init_dir_used = init_dir
            break
        init_errors.append(f"{init_dir}:rc={rc}")
    if not init_ok:
        raise RuntimeError(
            f"ck_model_init failed for all init dirs ({', '.join(init_errors)}) lib_dir={model_dir}"
        )
    try:
        runtime_contract = load_runtime_contract(model_dir)
        prefill_policy = str(runtime_contract.get("prefill_policy") or "batched").strip().lower()
        vocab = int(lib.ck_model_get_vocab_size())
        if vocab <= 0:
            raise RuntimeError(f"invalid CK vocab size: {vocab}")
        if prefill_policy == "sequential_decode":
            if not has_decode:
                raise RuntimeError("runtime contract requires sequential prefill but ck_model_decode is unavailable")
            if has_kv_reset:
                lib.ck_model_kv_cache_reset()
            for tok in tokens:
                rc = lib.ck_model_decode(ctypes.c_int32(int(tok)), None)
                if rc != 0:
                    raise RuntimeError(f"ck_model_decode failed rc={rc}")
        else:
            arr = (ctypes.c_int32 * len(tokens))(*[int(x) for x in tokens])
            rc = lib.ck_model_embed_tokens(arr, len(tokens))
            if rc != 0:
                raise RuntimeError(f"ck_model_embed_tokens failed rc={rc}")
            rc = lib.ck_model_forward(None)
            if rc != 0:
                raise RuntimeError(f"ck_model_forward failed rc={rc}")

        logits_ptr = lib.ck_model_get_logits()
        if not logits_ptr:
            raise RuntimeError("ck_model_get_logits returned null")

        stride = int(lib.ck_model_get_logits_stride()) if has_stride else 0
        active = int(lib.ck_model_get_active_tokens()) if has_active else len(tokens)
        if stride > 0 and active > 0:
            flat = np.ctypeslib.as_array(logits_ptr, shape=(active * stride,))
            start = (active - 1) * stride
            logits = flat[start : start + vocab].astype(np.float32, copy=True)
        else:
            logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab,)).astype(np.float32, copy=True)
        return {
            "vocab": vocab,
            "active_tokens": active,
            "stride": stride,
            "init_dir": str(init_dir_used) if init_dir_used is not None else str(model_dir),
            "prefill_policy": prefill_policy,
            "logits": logits,
        }
    finally:
        if has_free:
            try:
                lib.ck_model_free()
            except Exception:
                pass


def compare_logits(
    ck_logits: np.ndarray,
    llama_logits: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    n = min(int(ck_logits.size), int(llama_logits.size))
    a = ck_logits[:n].astype(np.float64, copy=False)
    b = llama_logits[:n].astype(np.float64, copy=False)
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom > 0.0 else 0.0

    k = max(1, min(int(top_k), n))
    ck_top = np.argpartition(-a, k - 1)[:k]
    ck_top = ck_top[np.argsort(-a[ck_top])]
    ll_top = np.argpartition(-b, k - 1)[:k]
    ll_top = ll_top[np.argsort(-b[ll_top])]

    ck_top_list = [int(x) for x in ck_top.tolist()]
    ll_top_list = [int(x) for x in ll_top.tolist()]
    overlap = set(ck_top_list) & set(ll_top_list)

    return {
        "n_compared": n,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rmse": rmse,
        "cosine": cosine,
        "top1_ck": ck_top_list[0],
        "top1_llama": ll_top_list[0],
        "top1_match": bool(ck_top_list[0] == ll_top_list[0]),
        "topk_overlap_count": len(overlap),
        "topk_overlap_ratio": float(len(overlap) / float(k)),
        "topk": k,
        "ck_topk_ids": ck_top_list,
        "llama_topk_ids": ll_top_list,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Tokenizer-free first-token logits parity (CK vs llama.cpp)")
    ap.add_argument("--model-dir", required=True, type=Path, help="run dir or .ck_build dir containing libmodel.so")
    ap.add_argument("--gguf", default=None, type=Path, help="GGUF path for llama.cpp runtime")
    ap.add_argument("--tokens", required=True, help="comma-separated token IDs")
    ap.add_argument("--ctx-len", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--require-top1-match", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-topk-overlap", type=float, default=0.50)
    ap.add_argument("--max-abs-threshold", type=float, default=1.0e9)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    model_dir = discover_ck_model_dir(args.model_dir)
    gguf_path = discover_gguf(args.gguf, model_dir)
    tokens = parse_tokens_csv(args.tokens)

    # Run llama helper first. CK runtime initializes OpenMP/threadpool state;
    # forking a subprocess after that can crash on some systems.
    ll = run_llama_logits(gguf_path, tokens, int(args.ctx_len), int(args.top_k), int(args.threads))
    ck = load_ck_logits(model_dir, tokens)
    cmp = compare_logits(ck["logits"], ll["logits"], int(args.top_k))

    overlap_ok = cmp["topk_overlap_ratio"] >= float(args.min_topk_overlap)
    top1_ok = (not bool(args.require_top1_match)) or cmp["top1_match"]
    max_abs_ok = cmp["max_abs_diff"] <= float(args.max_abs_threshold)
    passed = bool(top1_ok and overlap_ok and max_abs_ok)

    report = {
        "status": "pass" if passed else "fail",
        "pass": passed,
        "model_dir": str(model_dir),
        "gguf_path": str(gguf_path),
        "tokens": tokens,
        "ctx_len": int(args.ctx_len),
        "thresholds": {
            "require_top1_match": bool(args.require_top1_match),
            "min_topk_overlap": float(args.min_topk_overlap),
            "max_abs_threshold": float(args.max_abs_threshold),
        },
        "ck": {
            "vocab": int(ck["vocab"]),
            "active_tokens": int(ck["active_tokens"]),
            "logits_stride": int(ck["stride"]),
            "init_dir": str(ck.get("init_dir", "")),
            "prefill_policy": str(ck.get("prefill_policy", "batched")),
        },
        "llama": {
            "n_vocab": int(ll["meta"]["n_vocab"]),
            "token_count": int(ll["meta"]["token_count"]),
            "topk_count": int(len(ll["meta"].get("topk", []) or [])),
            "topk_sample": ll["meta"].get("topk", [])[: min(8, int(args.top_k))],
        },
        "compare": cmp,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
