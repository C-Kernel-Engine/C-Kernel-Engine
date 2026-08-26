#!/usr/bin/env python3
from __future__ import annotations

"""
Tokenizer-free multi-token greedy parity probe.

This script repeatedly compares CK and llama.cpp logits for the same explicit
token prefix, appends the shared greedy top-1 token, and stops at the first
top-1 divergence. It is deliberately deterministic and sampler-free so that
generation collapse can be separated from sampling/template issues.
"""

import argparse
import ctypes
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import tempfile
import traceback
from typing import Any

import numpy as np

from compare_first_token_logits_v8 import (  # type: ignore
    _llama_trajectory_temp_root,
    compare_logits,
    discover_ck_model_dir,
    discover_gguf,
    load_ck_logits,
    load_ck_logits_segmented,
    load_runtime_contract,
    parse_tokens_csv,
    run_llama_greedy_trajectory,
    run_llama_logits,
    run_llama_logits_segmented,
)


_AUTO_STREAM_LOGITS_BYTES = 256 * 1024 * 1024
_HIDDEN_CAPTURE_SUFFIXES = {".f32", ".i32"}


def _hidden_capture_paths(dump_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dump_dir.resolve().glob("tok_*_layer_*_*")
        if path.suffix in _HIDDEN_CAPTURE_SUFFIXES
        and path.is_file()
        and path.stat().st_size > 0
    )


def _trajectory_logits_bytes(model_dir: Path, max_new_tokens: int) -> int | None:
    """Estimate one full trajectory tensor without loading the model runtime."""
    for name in ("config.json", "weights_manifest.json", "layout_decode.json"):
        candidate = model_dir / name
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            vocab = int(payload.get("vocab_size", 0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if vocab > 0:
            return int(max_new_tokens) * vocab * np.dtype(np.float32).itemsize
    return None


def _select_trajectory_storage(
    requested: str,
    *,
    capture_requested: bool,
    estimated_logits_bytes: int | None,
) -> str:
    mode = str(requested)
    if mode not in {"auto", "memory", "stream"}:
        raise ValueError(f"unsupported trajectory storage mode: {mode}")
    if mode == "auto":
        return (
            "stream"
            if not capture_requested
            and estimated_logits_bytes is not None
            and estimated_logits_bytes > _AUTO_STREAM_LOGITS_BYTES
            else "memory"
        )
    if mode == "stream" and capture_requested:
        raise ValueError(
            "streaming trajectory storage does not support boundary capture; "
            "use auto/memory for capture or run a separate bounded trajectory certification"
        )
    return mode


def _configure_ck_threads(threads: int) -> dict[str, str]:
    value = str(max(1, int(threads)))
    configured = {
        "CK_NUM_THREADS": value,
        "CK_THREADPOOL_THREADS": value,
        "OMP_NUM_THREADS": value,
    }
    os.environ.update(configured)
    return configured


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ck_greedy_trajectory(
    *,
    model_dir: Path,
    prompt_tokens: list[int],
    max_new_tokens: int,
    stop_token_ids: set[int] | None = None,
    runtime_so: Path | None = None,
    dump_step: int | None = None,
    dump_dir: Path | None = None,
    dump_layer: int | None = None,
    dump_names: str = "",
    dump_format: str = "hidden",
    dump_kv_layer: int | None = None,
    forced_tokens: list[int] | None = None,
    reference_logits_path: Path | None = None,
    reference_steps: int | None = None,
    comparison_top_k: int = 20,
    stop_on_top1_divergence: bool = False,
) -> dict[str, Any]:
    capture_step = None if dump_step is None else int(dump_step)
    if capture_step is not None:
        if capture_step < 0 or capture_step >= int(max_new_tokens):
            raise ValueError(
                f"dump_step={capture_step} is outside trajectory [0, {int(max_new_tokens) - 1}]"
            )
        if dump_dir is None:
            raise ValueError("dump_dir is required when dump_step is set")
        if dump_format not in {"hidden", "parity"}:
            raise ValueError(f"unsupported dump_format: {dump_format}")

    runtime_path = (runtime_so or (model_dir / "libmodel.so")).resolve()
    if not runtime_path.is_file():
        raise FileNotFoundError(f"CK runtime does not exist: {runtime_path}")
    if capture_step is not None:
        resolved_dump_dir = dump_dir.resolve()
        resolved_dump_dir.mkdir(parents=True, exist_ok=True)
        if dump_format == "hidden":
            os.environ["CK_DEBUG_EXPORT_HIDDEN"] = ""
            if dump_layer is not None:
                os.environ["CK_DEBUG_EXPORT_HIDDEN_LAYER"] = str(int(dump_layer))
            if str(dump_names).strip():
                os.environ["CK_DEBUG_EXPORT_HIDDEN_NAMES"] = str(dump_names).strip()
        else:
            os.environ["CK_PARITY_DIR"] = str(resolved_dump_dir)
            os.environ["CK_PARITY_CAPTURE_ENABLED"] = "0"
            if dump_layer is not None:
                os.environ["CK_PARITY_LAYER_FILTER"] = str(int(dump_layer))
            if str(dump_names).strip():
                os.environ["CK_PARITY_OP_FILTER"] = str(dump_names).strip()

    lib = ctypes.CDLL(str(runtime_path), mode=ctypes.RTLD_GLOBAL)
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
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
    has_free = hasattr(lib, "ck_model_free")
    if has_free:
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None
    has_strict = hasattr(lib, "ck_set_strict_parity")
    if has_strict:
        lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
        lib.ck_set_strict_parity.restype = None
    kv_export_name = (
        "ck_model_debug_export_kv"
        if hasattr(lib, "ck_model_debug_export_kv")
        else "ck_model_debug_export_kv_f16"
    )
    has_kv_export = hasattr(lib, kv_export_name)
    if has_kv_export:
        kv_export = getattr(lib, kv_export_name)
        kv_export.argtypes = [ctypes.c_char_p, ctypes.c_int]
        kv_export.restype = ctypes.c_int

    init_candidates = [model_dir / "weights.bump", model_dir]
    if model_dir.name in {".ck_build", "ck_build"}:
        init_candidates.extend([model_dir.parent / "weights.bump", model_dir.parent])
    init_dir: Path | None = None
    for candidate in init_candidates:
        candidate = candidate.resolve()
        if lib.ck_model_init(str(candidate).encode("utf-8")) == 0:
            init_dir = candidate
            break
    if init_dir is None:
        raise RuntimeError(f"ck_model_init failed under {model_dir}")

    try:
        if has_strict:
            strict = os.environ.get("CK_STRICT_PARITY", "0")
            lib.ck_set_strict_parity(1 if int(strict or "0") != 0 else 0)
        def set_capture_enabled(enabled: bool) -> None:
            if capture_step is None:
                return
            if dump_format == "hidden":
                os.environ["CK_DEBUG_EXPORT_HIDDEN"] = (
                    str(dump_dir.resolve()) if enabled else ""
                )
            else:
                os.environ["CK_PARITY_CAPTURE_ENABLED"] = "1" if enabled else "0"

        prompt = [int(token) for token in prompt_tokens]
        if not prompt:
            raise ValueError("CK trajectory requires prompt tokens")
        token_array = (ctypes.c_int32 * len(prompt))(*prompt)
        if capture_step == 0:
            # Batched prefill executes inside embed_tokens. Enabling capture
            # only around ck_model_forward misses every prefill checkpoint
            # because forward merely returns the logits already computed here.
            set_capture_enabled(True)
        if lib.ck_model_embed_tokens(token_array, len(prompt)) != 0:
            raise RuntimeError("ck_model_embed_tokens failed")

        kv_dump_path = (
            dump_dir.resolve() / f"kv_layer_{int(dump_kv_layer):03d}.ckx"
            if capture_step is not None and dump_kv_layer is not None
            else None
        )

        def export_kv() -> None:
            if kv_dump_path is None:
                return
            if not has_kv_export:
                raise RuntimeError(
                    "requested KV capture but runtime lacks ck_model_debug_export_kv_f16"
                )
            rc = int(
                kv_export(
                    str(kv_dump_path).encode("utf-8"),
                    int(dump_kv_layer),
                )
            )
            if rc != 0:
                raise RuntimeError(f"CK FP16 KV export failed with rc={rc}")

        if lib.ck_model_forward(None) != 0:
            raise RuntimeError("ck_model_forward failed")
        if capture_step == 0:
            set_capture_enabled(False)
            export_kv()

        vocab = int(lib.ck_model_get_vocab_size())
        if vocab <= 0:
            raise RuntimeError(f"invalid CK vocabulary size: {vocab}")

        def read_logits() -> np.ndarray:
            pointer = lib.ck_model_get_logits()
            if not pointer:
                raise RuntimeError("ck_model_get_logits returned null")
            stride = int(lib.ck_model_get_logits_stride()) if has_stride else 0
            active = int(lib.ck_model_get_active_tokens()) if has_active else 1
            if stride > 0 and active > 0:
                flat = np.ctypeslib.as_array(pointer, shape=(active * stride,))
                start = (active - 1) * stride
                return flat[start : start + vocab].astype(np.float32, copy=True)
            return np.ctypeslib.as_array(pointer, shape=(vocab,)).astype(np.float32, copy=True)

        reference_handle: Any | None = None
        if reference_logits_path is not None:
            resolved_reference = reference_logits_path.expanduser().resolve()
            steps_expected = int(reference_steps or max_new_tokens)
            expected_bytes = steps_expected * vocab * np.dtype(np.float32).itemsize
            if not resolved_reference.is_file():
                raise FileNotFoundError(
                    f"reference logits stream does not exist: {resolved_reference}"
                )
            if resolved_reference.stat().st_size != expected_bytes:
                raise RuntimeError(
                    "reference logits stream size mismatch: "
                    f"got={resolved_reference.stat().st_size} expected={expected_bytes}"
                )
            reference_handle = resolved_reference.open("rb")

        rows: list[np.ndarray] = []
        stream_steps: list[dict[str, Any]] = []
        generated: list[int] = []
        stops = {int(token) for token in (stop_token_ids or set())}
        teacher = [int(token) for token in (forced_tokens or [])]
        if teacher and len(teacher) < max(0, int(max_new_tokens) - 1):
            raise ValueError(
                "forced_tokens must provide every decoded token before the final prediction"
            )
        for step in range(int(max_new_tokens)):
            logits = read_logits()
            token = int(np.argmax(logits))
            if reference_handle is None:
                rows.append(logits)
            else:
                oracle = np.fromfile(reference_handle, dtype=np.float32, count=vocab)
                if oracle.size != vocab:
                    raise RuntimeError(
                        "reference logits stream ended early: "
                        f"step={step} got={oracle.size} expected={vocab}"
                    )
                comparison = compare_logits(logits, oracle, int(comparison_top_k))
                oracle_token = int(comparison["top1_llama"])
                exact = bool(
                    np.array_equal(logits.view(np.uint32), oracle.view(np.uint32))
                )
                stream_steps.append({
                    "step": int(step),
                    "prefix_len": len(prompt) + int(step),
                    "ck_next": token,
                    "llama_next": oracle_token,
                    "top1_match": token == oracle_token,
                    "bit_exact": exact,
                    "cosine": float(comparison["cosine"]),
                    "rmse": float(comparison["rmse"]),
                    "mean_abs_diff": float(comparison["mean_abs_diff"]),
                    "max_abs_diff": float(comparison["max_abs_diff"]),
                    "ck_top1_margin": float(comparison["ck_top1_margin"]),
                    "llama_top1_margin": float(comparison["llama_top1_margin"]),
                    "topk_overlap_count": int(comparison["topk_overlap_count"]),
                    "topk_overlap_ratio": float(comparison["topk_overlap_ratio"]),
                    "ck_topk_ids": list(comparison["ck_topk_ids"]),
                    "llama_topk_ids": list(comparison["llama_topk_ids"]),
                    "topk_logits": list(comparison["topk_logits"]),
                })
            generated.append(token)
            forced_stop = bool(
                teacher and step < len(teacher) and teacher[step] in stops
            )
            stream_diverged = bool(
                reference_handle is not None
                and stop_on_top1_divergence
                and stream_steps
                and not stream_steps[-1]["top1_match"]
            )
            if (
                (not teacher and token in stops)
                or forced_stop
                or stream_diverged
                or step + 1 >= int(max_new_tokens)
            ):
                break
            decode_token = teacher[step] if teacher else token
            if capture_step == step + 1:
                set_capture_enabled(True)
            if lib.ck_model_decode(ctypes.c_int32(decode_token), None) != 0:
                raise RuntimeError(f"ck_model_decode failed at greedy step {step}")
            if capture_step == step + 1:
                set_capture_enabled(False)
                export_kv()
        dump_paths: list[Path] = []
        if capture_step is not None:
            if dump_format == "hidden":
                dump_paths = _hidden_capture_paths(dump_dir)
                if kv_dump_path is not None and kv_dump_path.is_file():
                    dump_paths.append(kv_dump_path)
            else:
                candidate = dump_dir.resolve() / "dump.bin"
                if candidate.is_file() and candidate.stat().st_size > 0:
                    dump_paths = [candidate]
            if not dump_paths:
                raise RuntimeError(
                    "requested persistent trajectory dump was not emitted; "
                    "verify the generated runtime exports the requested checkpoints"
                )
        return {
            "logits": np.stack(rows) if rows else None,
            "stream_steps": stream_steps,
            "logits_storage": (
                "bounded_stream_comparison"
                if reference_handle is not None
                else "retained_memory"
            ),
            "generated_tokens": generated,
            "forced_tokens": teacher,
            "vocab": vocab,
            "init_dir": str(init_dir),
            "runtime": {
                "path": str(runtime_path),
                "sha256": _sha256_file(runtime_path),
            },
            "capture": {
                "execution_mode": "persistent_greedy_trajectory",
                "step": capture_step,
                "layer": None if dump_layer is None else int(dump_layer),
                "op_filter": str(dump_names).strip(),
                "format": dump_format,
                "kv_layer": None if dump_kv_layer is None else int(dump_kv_layer),
                "artifacts": [
                    {
                        "path": str(path),
                        "sha256": _sha256_file(path),
                        "size": int(path.stat().st_size),
                    }
                    for path in dump_paths
                ],
            },
        }
    finally:
        if "reference_handle" in locals() and reference_handle is not None:
            reference_handle.close()
        if has_free:
            lib.ck_model_free()


def _load_ck_greedy_trajectory_worker(
    connection: Any,
    model_dir: Path,
    prompt_tokens: list[int],
    max_new_tokens: int,
    stop_token_ids: set[int],
    threads: int,
    runtime_so: Path | None,
    dump_step: int | None,
    dump_dir: Path | None,
    dump_layer: int | None,
    dump_names: str,
    dump_format: str,
    dump_kv_layer: int | None,
    forced_tokens: list[int] | None,
    reference_logits_path: Path | None,
    reference_steps: int | None,
    comparison_top_k: int,
    stop_on_top1_divergence: bool,
) -> None:
    try:
        thread_environment = _configure_ck_threads(threads)
        result = load_ck_greedy_trajectory(
            model_dir=model_dir,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            runtime_so=runtime_so,
            dump_step=dump_step,
            dump_dir=dump_dir,
            dump_layer=dump_layer,
            dump_names=dump_names,
            dump_format=dump_format,
            dump_kv_layer=dump_kv_layer,
            forced_tokens=forced_tokens,
            reference_logits_path=reference_logits_path,
            reference_steps=reference_steps,
            comparison_top_k=comparison_top_k,
            stop_on_top1_divergence=stop_on_top1_divergence,
        )
        result["thread_environment"] = thread_environment
        connection.send(("ok", result))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def load_ck_greedy_trajectory_isolated(
    *,
    model_dir: Path,
    prompt_tokens: list[int],
    max_new_tokens: int,
    stop_token_ids: set[int] | None = None,
    threads: int = 1,
    runtime_so: Path | None = None,
    dump_step: int | None = None,
    dump_dir: Path | None = None,
    dump_layer: int | None = None,
    dump_names: str = "",
    dump_format: str = "hidden",
    dump_kv_layer: int | None = None,
    forced_tokens: list[int] | None = None,
    reference_logits_path: Path | None = None,
    reference_steps: int | None = None,
    comparison_top_k: int = 20,
    stop_on_top1_divergence: bool = False,
) -> dict[str, Any]:
    """Capture CK logits in a short-lived process so model mappings are released."""
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_load_ck_greedy_trajectory_worker,
        args=(
            send,
            model_dir,
            prompt_tokens,
            int(max_new_tokens),
            {int(token) for token in (stop_token_ids or set())},
            max(1, int(threads)),
            runtime_so,
            dump_step,
            dump_dir,
            dump_layer,
            dump_names,
            dump_format,
            dump_kv_layer,
            forced_tokens,
            reference_logits_path,
            reference_steps,
            int(comparison_top_k),
            bool(stop_on_top1_divergence),
        ),
    )
    process.start()
    send.close()
    try:
        status, payload = receive.recv()
    except EOFError as exc:
        process.join()
        raise RuntimeError(
            f"isolated CK trajectory failed with exit code {process.exitcode}"
        ) from exc
    finally:
        receive.close()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"isolated CK trajectory failed with exit code {process.exitcode}"
        )
    if status != "ok":
        raise RuntimeError(f"isolated CK trajectory failed:\n{payload}")
    return payload


def _compare_ck_trajectory_identity(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    """Compare two CKE runs on an identical forced causal history."""
    a = np.asarray(reference.get("logits"), dtype=np.float32)
    b = np.asarray(candidate.get("logits"), dtype=np.float32)
    result: dict[str, Any] = {
        "contract": "bit_exact_full_logits_same_forced_prefix",
        "reference_shape": list(a.shape),
        "candidate_shape": list(b.shape),
        "exact": False,
        "first_different_step": None,
    }
    if a.shape != b.shape:
        result["status"] = "shape_mismatch"
        return result
    a_bits = np.ascontiguousarray(a).view(np.uint32)
    b_bits = np.ascontiguousarray(b).view(np.uint32)
    if np.array_equal(a_bits, b_bits):
        result.update({"status": "pass", "exact": True})
        return result

    result["status"] = "fail"
    differing = np.flatnonzero(np.any(a_bits != b_bits, axis=1))
    step = int(differing[0]) if differing.size else 0
    comparison = compare_logits(a[step], b[step], int(top_k))
    result.update({
        "first_different_step": step,
        "cosine": float(comparison["cosine"]),
        "rmse": float(comparison["rmse"]),
        "mean_abs_diff": float(comparison["mean_abs_diff"]),
        "max_abs_diff": float(comparison["max_abs_diff"]),
        "top1_reference": int(comparison["top1_ck"]),
        "top1_candidate": int(comparison["top1_llama"]),
        "top1_match": bool(comparison["top1_ck"] == comparison["top1_llama"]),
    })
    return result


def _capture_boundary_names(dump_names: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw in str(dump_names).split(","):
        name = raw.strip()
        if not name or name in seen:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"unsafe CK capture boundary name: {name!r}")
        seen.add(name)
        names.append(name)
    return names


def _require_empty_capture_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError(
            f"CK dump directory must be empty to prevent stale evidence: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run_ck_capture_with_neutrality(
    *,
    model_dir: Path,
    prompt_tokens: list[int],
    max_new_tokens: int,
    top_k: int,
    threads: int,
    runtime_so: Path | None,
    dump_step: int,
    dump_dir: Path,
    dump_layer: int | None,
    dump_names: str,
    dump_format: str,
    dump_kv_layer: int | None,
    stop_token_ids: set[int],
    forced_tokens: list[int] | None = None,
    diagnose_single_thread: bool = False,
) -> dict[str, Any]:
    """Capture CKE checkpoints only after proving observer neutrality.

    Two uncaptured runs first establish that the runtime is repeatable on the
    same causal history. The aggregate capture is then compared bit-for-bit to
    that control. A non-neutral multi-boundary capture falls back to one replay
    per boundary; rejected artifacts remain explicitly labelled in the report.
    """
    root = _require_empty_capture_dir(dump_dir)
    boundaries = _capture_boundary_names(dump_names)

    control_a = load_ck_greedy_trajectory_isolated(
        model_dir=model_dir,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        threads=threads,
        runtime_so=runtime_so,
        forced_tokens=forced_tokens,
    )
    replay_tokens = (
        [int(token) for token in forced_tokens]
        if forced_tokens
        else [int(token) for token in control_a["generated_tokens"]]
    )
    replay_steps = min(int(max_new_tokens), len(replay_tokens))
    control_b = load_ck_greedy_trajectory_isolated(
        model_dir=model_dir,
        prompt_tokens=prompt_tokens,
        max_new_tokens=replay_steps,
        stop_token_ids=stop_token_ids,
        threads=threads,
        runtime_so=runtime_so,
        forced_tokens=replay_tokens,
    )
    repeatability = _compare_ck_trajectory_identity(control_a, control_b, top_k)
    neutrality: dict[str, Any] = {
        "schema": "cke.xray.capture-neutrality.v1",
        "acceptance_contract": "bit_exact_full_logits_same_forced_prefix",
        "baseline_repeatability": repeatability,
        "single_thread_simd_reference": {
            "attempted": False,
            "reason": "parallel_baseline_is_repeatable",
        },
        "aggregate_capture": None,
        "fallback": {"attempted": False, "boundaries": []},
        "accepted_mode": None,
        "status": "rejected",
    }
    if not repeatability["exact"]:
        if int(threads) > 1 and diagnose_single_thread:
            single_a = load_ck_greedy_trajectory_isolated(
                model_dir=model_dir,
                prompt_tokens=prompt_tokens,
                max_new_tokens=replay_steps,
                stop_token_ids=stop_token_ids,
                threads=1,
                runtime_so=runtime_so,
                forced_tokens=replay_tokens,
            )
            single_b = load_ck_greedy_trajectory_isolated(
                model_dir=model_dir,
                prompt_tokens=prompt_tokens,
                max_new_tokens=replay_steps,
                stop_token_ids=stop_token_ids,
                threads=1,
                runtime_so=runtime_so,
                forced_tokens=replay_tokens,
            )
            neutrality["single_thread_simd_reference"] = {
                "attempted": True,
                "threads": 1,
                "simd": "enabled_by_runtime_build",
                "causal_history": "same_forced_prefix",
                "repeatability": _compare_ck_trajectory_identity(
                    single_a, single_b, top_k
                ),
                "parallel_vs_reference": _compare_ck_trajectory_identity(
                    single_a, control_b, top_k
                ),
            }
        elif int(threads) <= 1:
            neutrality["single_thread_simd_reference"] = {
                "attempted": False,
                "reason": "baseline_already_uses_one_thread",
                "threads": 1,
                "simd": "enabled_by_runtime_build",
            }
        else:
            neutrality["single_thread_simd_reference"] = {
                "attempted": False,
                "reason": "not_requested",
                "threads": 1,
                "simd": "enabled_by_runtime_build",
            }
        neutrality["reason"] = "uncaptured_runtime_is_not_repeatable"
        control_b["capture"] = {"neutrality": neutrality, "artifacts": []}
        return control_b

    aggregate = load_ck_greedy_trajectory_isolated(
        model_dir=model_dir,
        prompt_tokens=prompt_tokens,
        max_new_tokens=replay_steps,
        stop_token_ids=stop_token_ids,
        threads=threads,
        runtime_so=runtime_so,
        dump_step=dump_step,
        dump_dir=root,
        dump_layer=dump_layer,
        dump_names=dump_names,
        dump_format=dump_format,
        dump_kv_layer=dump_kv_layer,
        forced_tokens=replay_tokens,
    )
    aggregate_comparison = _compare_ck_trajectory_identity(control_b, aggregate, top_k)
    neutrality["aggregate_capture"] = aggregate_comparison
    aggregate_artifacts = list((aggregate.get("capture") or {}).get("artifacts") or [])
    if aggregate_comparison["exact"]:
        neutrality.update({"status": "accepted", "accepted_mode": "aggregate"})
        aggregate["capture"]["neutrality"] = neutrality
        return aggregate

    if len(boundaries) <= 1 or dump_kv_layer is not None or dump_format != "hidden":
        neutrality["reason"] = "single_capture_is_not_observationally_neutral"
        control_b["capture"] = {
            "neutrality": neutrality,
            "rejected_artifacts": aggregate_artifacts,
            "artifacts": [],
        }
        return control_b

    neutrality["fallback"]["attempted"] = True
    accepted_artifacts: list[dict[str, Any]] = []
    all_neutral = True
    isolated_root = root / "isolated"
    isolated_root.mkdir(parents=True, exist_ok=True)
    for index, boundary in enumerate(boundaries):
        boundary_dir = isolated_root / f"{index:03d}_{boundary}"
        isolated = load_ck_greedy_trajectory_isolated(
            model_dir=model_dir,
            prompt_tokens=prompt_tokens,
            max_new_tokens=replay_steps,
            stop_token_ids=stop_token_ids,
            threads=threads,
            runtime_so=runtime_so,
            dump_step=dump_step,
            dump_dir=boundary_dir,
            dump_layer=dump_layer,
            dump_names=boundary,
            dump_format="hidden",
            forced_tokens=replay_tokens,
        )
        comparison = _compare_ck_trajectory_identity(control_b, isolated, top_k)
        artifacts = list((isolated.get("capture") or {}).get("artifacts") or [])
        accepted = bool(comparison["exact"])
        all_neutral = all_neutral and accepted
        if accepted:
            accepted_artifacts.extend(artifacts)
        neutrality["fallback"]["boundaries"].append({
            "name": boundary,
            "status": "accepted" if accepted else "rejected",
            "comparison": comparison,
            "artifacts": artifacts,
        })

    if all_neutral:
        neutrality.update({"status": "accepted", "accepted_mode": "isolated_boundaries"})
    else:
        neutrality["reason"] = "one_or_more_isolated_boundaries_are_not_neutral"
    control_b["capture"] = {
        "execution_mode": "persistent_greedy_trajectory",
        "step": int(dump_step),
        "layer": dump_layer,
        "op_filter": dump_names,
        "format": dump_format,
        "neutrality": neutrality,
        "rejected_artifacts": aggregate_artifacts,
        "artifacts": accepted_artifacts if all_neutral else [],
    }
    return control_b


def run_multitoken_trajectory_parity(
    *,
    model_dir: Path,
    gguf_path: Path,
    prompt_tokens: list[int],
    max_new_tokens: int,
    ctx_len: int,
    top_k: int,
    threads: int,
    llama_no_repack: bool,
    stop_token_ids: set[int] | None = None,
    ck_runtime_so: Path | None = None,
    ck_dump_step: int | None = None,
    ck_dump_dir: Path | None = None,
    ck_dump_layer: int | None = None,
    ck_dump_names: str = "",
    ck_dump_format: str = "hidden",
    ck_dump_kv_layer: int | None = None,
    llama_dump_step: int | None = None,
    llama_dump_dir: Path | None = None,
    llama_dump_names: str = "",
    llama_dump_flash_inputs: bool = False,
    llama_profile_layers_out: Path | None = None,
    append_on_divergence: str = "stop",
    diagnose_single_thread: bool = False,
) -> dict[str, Any]:
    stops = {int(token) for token in (stop_token_ids or set())}
    if append_on_divergence not in {"stop", "llama"}:
        raise ValueError("persistent trajectory supports stop or llama teacher forcing")
    ck: dict[str, Any] | None = None
    if append_on_divergence == "stop":
        # Keep the lower-memory certification order: release CKE's mappings
        # before llama.cpp brings the GGUF into its page cache.
        if ck_dump_step is not None:
            assert ck_dump_dir is not None
            ck = run_ck_capture_with_neutrality(
                model_dir=model_dir,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                threads=threads,
                runtime_so=ck_runtime_so,
                dump_step=ck_dump_step,
                dump_dir=ck_dump_dir,
                dump_layer=ck_dump_layer,
                dump_names=ck_dump_names,
                dump_format=ck_dump_format,
                dump_kv_layer=ck_dump_kv_layer,
                stop_token_ids=stops,
                diagnose_single_thread=diagnose_single_thread,
            )
        else:
            ck = load_ck_greedy_trajectory_isolated(
                model_dir=model_dir,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stops,
                threads=threads,
                runtime_so=ck_runtime_so,
            )
    llama = run_llama_greedy_trajectory(
        gguf_path,
        prompt_tokens,
        max_new_tokens,
        ctx_len,
        top_k,
        threads,
        llama_no_repack,
        dump_step=llama_dump_step,
        dump_dir=llama_dump_dir,
        dump_names=llama_dump_names,
        dump_flash_inputs=llama_dump_flash_inputs,
        profile_layers_out=llama_profile_layers_out,
    )
    # Teacher forcing needs the oracle sequence before CKE starts. Both model
    # mappings are still isolated in separate processes; only the small token
    # sequence crosses the boundary.
    if append_on_divergence == "llama":
        teacher = [int(token) for token in llama["generated_tokens"]]
        if ck_dump_step is not None:
            assert ck_dump_dir is not None
            ck = run_ck_capture_with_neutrality(
                model_dir=model_dir,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                threads=threads,
                runtime_so=ck_runtime_so,
                dump_step=ck_dump_step,
                dump_dir=ck_dump_dir,
                dump_layer=ck_dump_layer,
                dump_names=ck_dump_names,
                dump_format=ck_dump_format,
                dump_kv_layer=ck_dump_kv_layer,
                stop_token_ids=stops,
                forced_tokens=teacher,
                diagnose_single_thread=diagnose_single_thread,
            )
        else:
            ck = load_ck_greedy_trajectory_isolated(
                model_dir=model_dir,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stops,
                threads=threads,
                runtime_so=ck_runtime_so,
                forced_tokens=teacher,
            )
    assert ck is not None
    steps: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    matched_stop_token: int | None = None
    compared = min(len(ck["generated_tokens"]), len(llama["generated_tokens"]))
    for step in range(compared):
        cmp = compare_logits(ck["logits"][step], llama["logits"][step], int(top_k))
        ck_next = int(ck["generated_tokens"][step])
        llama_next = int(llama["generated_tokens"][step])
        row = {
            "step": step,
            "prefix_len": len(prompt_tokens) + step,
            "ck_next": ck_next,
            "llama_next": llama_next,
            "top1_match": ck_next == llama_next,
            "cosine": float(cmp["cosine"]),
            "rmse": float(cmp["rmse"]),
            "mean_abs_diff": float(cmp["mean_abs_diff"]),
            "max_abs_diff": float(cmp["max_abs_diff"]),
            "ck_top1_margin": float(cmp["ck_top1_margin"]),
            "llama_top1_margin": float(cmp["llama_top1_margin"]),
            "topk_overlap_count": int(cmp["topk_overlap_count"]),
            "topk_overlap_ratio": float(cmp["topk_overlap_ratio"]),
            "ck_topk_ids": list(cmp["ck_topk_ids"]),
            "llama_topk_ids": list(cmp["llama_topk_ids"]),
            "topk_logits": list(cmp["topk_logits"]),
        }
        steps.append(row)
        if ck_next != llama_next:
            if first_divergence is None:
                first_divergence = row
            if append_on_divergence == "stop":
                break
        if ck_next in stops:
            matched_stop_token = ck_next
            break

    generated_prefix = (
        [int(token) for token in llama["generated_tokens"][: len(steps)]]
        if append_on_divergence == "llama"
        else [int(token) for token in ck["generated_tokens"][: len(steps)]]
    )
    # A shared token belongs to the causal prefix. A stop token or divergent
    # prediction was compared but was not decoded by both runtimes.
    if (matched_stop_token is not None or (first_divergence is not None and append_on_divergence == "stop")) and generated_prefix:
        generated_prefix.pop()
    return {
        "status": "pass" if first_divergence is None else "fail",
        "pass": first_divergence is None,
        "model_dir": str(model_dir),
        "gguf_path": str(gguf_path),
        "initial_tokens": [int(token) for token in prompt_tokens],
        "final_prefix": [int(token) for token in prompt_tokens] + generated_prefix,
        "max_new_tokens": int(max_new_tokens),
        "ctx_len": int(ctx_len),
        "top_k": int(top_k),
        "threads": int(threads),
        "ck_thread_environment": dict(ck.get("thread_environment", {})),
        "ck_runtime": dict(ck.get("runtime", {})),
        "ck_capture": dict(ck.get("capture", {})),
        "llama_capture": dict(llama.get("capture", {})),
        "execution_mode": "persistent_greedy_trajectory",
        "trajectory_policy": "llama_teacher_forced" if append_on_divergence == "llama" else "shared_until_divergence",
        "ck_prefill_mode": "hybrid",
        "llama_decode_mode": "hybrid",
        "llama_no_repack": bool(llama_no_repack),
        "stop_token_ids": sorted(stops),
        "matched_stop_token": matched_stop_token,
        "first_divergence": first_divergence,
        "steps": steps,
        "llama_layer_profile": llama.get("layer_profile"),
    }


def run_multitoken_trajectory_parity_streaming(
    *,
    model_dir: Path,
    gguf_path: Path,
    prompt_tokens: list[int],
    max_new_tokens: int,
    ctx_len: int,
    top_k: int,
    threads: int,
    llama_no_repack: bool,
    stop_token_ids: set[int] | None = None,
    ck_runtime_so: Path | None = None,
    llama_profile_layers_out: Path | None = None,
    append_on_divergence: str = "stop",
) -> dict[str, Any]:
    """Compare a persistent trajectory with bounded resident logits memory.

    llama.cpp writes the oracle rows once to a temporary file. CKE reads that
    file sequentially and compares one vocabulary row at a time inside its
    isolated model process. Only compact per-step metrics cross the process
    boundary; neither runtime retains the complete logits tensor in RAM.
    """
    if append_on_divergence not in {"stop", "llama"}:
        raise ValueError("streaming trajectory supports stop or llama teacher forcing")
    stops = {int(token) for token in (stop_token_ids or set())}
    with tempfile.TemporaryDirectory(
        prefix="cke_xray_stream_",
        dir=_llama_trajectory_temp_root(),
    ) as td:
        sequence_path = Path(td) / "llama_logits_sequence.f32"
        llama = run_llama_greedy_trajectory(
            gguf_path,
            prompt_tokens,
            max_new_tokens,
            ctx_len,
            top_k,
            threads,
            llama_no_repack,
            profile_layers_out=llama_profile_layers_out,
            logits_sequence_out=sequence_path,
            load_logits=False,
        )
        teacher = [int(token) for token in llama["generated_tokens"]]
        ck = load_ck_greedy_trajectory_isolated(
            model_dir=model_dir,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stops,
            threads=threads,
            runtime_so=ck_runtime_so,
            forced_tokens=teacher,
            reference_logits_path=sequence_path,
            reference_steps=max_new_tokens,
            comparison_top_k=top_k,
            stop_on_top1_divergence=append_on_divergence == "stop",
        )

    steps = list(ck.get("stream_steps") or [])
    first_divergence = next(
        (row for row in steps if not bool(row["top1_match"])),
        None,
    )
    matched_stop_token = next(
        (
            int(row["ck_next"])
            for row in steps
            if bool(row["top1_match"]) and int(row["ck_next"]) in stops
        ),
        None,
    )
    generated_prefix = teacher[: len(steps)]
    if (matched_stop_token is not None or first_divergence is not None) and generated_prefix:
        generated_prefix.pop()
    exact_steps = sum(bool(row.get("bit_exact")) for row in steps)
    return {
        "status": "pass" if first_divergence is None else "fail",
        "pass": first_divergence is None,
        "model_dir": str(model_dir),
        "gguf_path": str(gguf_path),
        "initial_tokens": [int(token) for token in prompt_tokens],
        "final_prefix": [int(token) for token in prompt_tokens] + generated_prefix,
        "max_new_tokens": int(max_new_tokens),
        "ctx_len": int(ctx_len),
        "top_k": int(top_k),
        "threads": int(threads),
        "ck_thread_environment": dict(ck.get("thread_environment", {})),
        "ck_runtime": dict(ck.get("runtime", {})),
        "ck_capture": dict(ck.get("capture", {})),
        "llama_capture": dict(llama.get("capture", {})),
        "execution_mode": "persistent_greedy_trajectory_streaming",
        "trajectory_policy": (
            "llama_teacher_forced" if append_on_divergence == "llama"
            else "shared_until_divergence"
        ),
        "logits_storage": {
            "mode": "bounded_stream_comparison",
            "oracle": "temporary_file_backed",
            "cke": "single_row",
            "temporary_artifact_retained": False,
            "exact_steps": int(exact_steps),
            "compared_steps": len(steps),
        },
        "ck_prefill_mode": "hybrid",
        "llama_decode_mode": "hybrid",
        "llama_no_repack": bool(llama_no_repack),
        "stop_token_ids": sorted(stops),
        "matched_stop_token": matched_stop_token,
        "first_divergence": first_divergence,
        "steps": steps,
        "llama_layer_profile": llama.get("layer_profile"),
    }


def run_multitoken_parity(
    *,
    model_dir: Path,
    gguf_path: Path,
    prompt_tokens: list[int],
    max_new_tokens: int,
    ctx_len: int,
    top_k: int,
    threads: int,
    append_on_divergence: str,
    ck_prefill_mode: str,
    llama_decode_mode: str,
    llama_no_repack: bool,
    stop_token_ids: set[int] | None = None,
) -> dict[str, Any]:
    thread_environment = _configure_ck_threads(threads)
    tokens = [int(t) for t in prompt_tokens]
    steps: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    matched_stop_token: int | None = None
    declared_stop_tokens = {int(token_id) for token_id in (stop_token_ids or set())}

    for step in range(max(1, int(max_new_tokens))):
        if llama_decode_mode == "hybrid":
            ll = run_llama_logits_segmented(
                gguf_path,
                [int(t) for t in prompt_tokens],
                [int(t) for t in tokens[len(prompt_tokens) :]],
                int(ctx_len),
                int(top_k),
                int(threads),
                prefix_decode_mode="batched",
                decode_mode="sequential",
                no_repack=llama_no_repack,
            )
        else:
            ll = run_llama_logits(
                gguf_path,
                tokens,
                int(ctx_len),
                int(top_k),
                int(threads),
                decode_mode=llama_decode_mode,
                no_repack=llama_no_repack,
            )
        generated_tokens = [int(t) for t in tokens[len(prompt_tokens) :]]
        if ck_prefill_mode == "hybrid":
            ck = load_ck_logits_segmented(
                model_dir=model_dir,
                prompt_tokens=[int(t) for t in prompt_tokens],
                decode_tokens=generated_tokens,
                ck_prefill_mode="hybrid",
            )
        else:
            ck = load_ck_logits(model_dir, tokens, ck_prefill_mode=ck_prefill_mode)
        cmp = compare_logits(ck["logits"], ll["logits"], int(top_k))
        ck_next = int(cmp["top1_ck"])
        llama_next = int(cmp["top1_llama"])
        top1_match = bool(ck_next == llama_next)

        row = {
            "step": int(step),
            "prefix_len": int(len(tokens)),
            "ck_next": ck_next,
            "llama_next": llama_next,
            "top1_match": top1_match,
            "cosine": float(cmp["cosine"]),
            "rmse": float(cmp["rmse"]),
            "mean_abs_diff": float(cmp["mean_abs_diff"]),
            "max_abs_diff": float(cmp["max_abs_diff"]),
            "ck_top1_margin": float(cmp.get("ck_top1_margin", 0.0)),
            "llama_top1_margin": float(cmp.get("llama_top1_margin", 0.0)),
            "ck_llama_winner_delta_in_ck": float(cmp.get("ck_llama_winner_delta_in_ck", 0.0)),
            "llama_winner_delta_in_llama": float(cmp.get("llama_winner_delta_in_llama", 0.0)),
            "topk_overlap_count": int(cmp["topk_overlap_count"]),
            "topk_overlap_ratio": float(cmp["topk_overlap_ratio"]),
            "ck_topk_ids": list(cmp["ck_topk_ids"]),
            "llama_topk_ids": list(cmp["llama_topk_ids"]),
            "topk_logits": list(cmp.get("topk_logits", [])),
        }
        steps.append(row)

        if not top1_match and first_divergence is None:
            first_divergence = row
            if append_on_divergence == "stop":
                break

        if top1_match and ck_next in declared_stop_tokens:
            matched_stop_token = ck_next
            break

        if top1_match or append_on_divergence == "llama":
            tokens.append(llama_next)
        elif append_on_divergence == "ck":
            tokens.append(ck_next)
        else:
            break

    return {
        "status": "pass" if first_divergence is None else "fail",
        "pass": first_divergence is None,
        "model_dir": str(model_dir),
        "gguf_path": str(gguf_path),
        "initial_tokens": [int(t) for t in prompt_tokens],
        "final_prefix": tokens,
        "max_new_tokens": int(max_new_tokens),
        "ctx_len": int(ctx_len),
        "top_k": int(top_k),
        "threads": int(threads),
        "ck_thread_environment": thread_environment,
        "append_on_divergence": str(append_on_divergence),
        "ck_prefill_mode": str(ck_prefill_mode),
        "llama_decode_mode": str(llama_decode_mode),
        "llama_no_repack": bool(llama_no_repack),
        "stop_token_ids": sorted(declared_stop_tokens),
        "matched_stop_token": matched_stop_token,
        "first_divergence": first_divergence,
        "steps": steps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Tokenizer-free multi-token greedy parity (CK vs llama.cpp)")
    ap.add_argument("--model-dir", required=True, type=Path, help="run dir or .ck_build dir containing libmodel.so")
    ap.add_argument("--gguf", default=None, type=Path, help="GGUF path for llama.cpp runtime")
    ap.add_argument("--tokens", required=True, help="comma-separated prompt token IDs")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--ctx-len", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument(
        "--llama-decode-mode",
        choices=["auto", "batched", "sequential", "hybrid"],
        default="auto",
        help="llama.cpp replay mode; hybrid batches the initial prompt then decodes generated tokens sequentially.",
    )
    ap.add_argument(
        "--ck-prefill-mode",
        choices=["auto", "sequential", "batched", "hybrid"],
        default="auto",
        help=(
            "CK replay mode. auto follows runtime_contract; sequential feeds every token through decode; "
            "batched runs the whole prefix through ck_model_forward; hybrid batches the initial prompt "
            "then decodes generated tokens one by one."
        ),
    )
    ap.add_argument(
        "--llama-no-repack",
        action="store_true",
        help="Disable llama.cpp CPU tensor repacking in the replay helper for accumulation-order attribution.",
    )
    ap.add_argument(
        "--append-on-divergence",
        choices=["stop", "llama", "ck"],
        default="stop",
        help="What to append after first top-1 mismatch.",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument(
        "--stop-tokens",
        default="",
        help="comma-separated token IDs; a matched CK/llama token ends parity successfully",
    )
    ap.add_argument("--summary", action="store_true", help="Print a compact one-line result instead of full JSON.")
    ap.add_argument(
        "--execution-mode",
        choices=["replay", "trajectory"],
        default="replay",
        help="trajectory keeps each runtime loaded and is intended for long deterministic certification.",
    )
    ap.add_argument(
        "--trajectory-storage",
        choices=["auto", "memory", "stream"],
        default="auto",
        help=(
            "Logits storage for persistent trajectories. auto uses bounded streaming "
            "when one full trajectory tensor exceeds 256 MiB; memory preserves the "
            "legacy in-memory path. Boundary capture currently uses memory mode."
        ),
    )
    ap.add_argument(
        "--ck-runtime-so",
        type=Path,
        default=None,
        help="Explicit generated CK runtime; required when capture uses a dedicated parity-dump build.",
    )
    ap.add_argument(
        "--ck-dump-step",
        type=int,
        default=None,
        help="Capture only this zero-based persistent greedy trajectory step.",
    )
    ap.add_argument(
        "--ck-dump-dir",
        type=Path,
        default=None,
        help="Output directory for --ck-dump-step.",
    )
    ap.add_argument(
        "--ck-dump-layer",
        type=int,
        default=None,
        help="Optional CK_PARITY_LAYER_FILTER for bounded trajectory capture.",
    )
    ap.add_argument(
        "--ck-dump-names",
        default="",
        help="Optional comma-separated CK parity operation filter.",
    )
    ap.add_argument(
        "--ck-dump-format",
        choices=("hidden", "parity"),
        default="hidden",
        help=(
            "hidden uses the gated exports already compiled into the production runtime; "
            "parity requires a dedicated CK_PARITY_DUMP build"
        ),
    )
    ap.add_argument(
        "--ck-dump-kv-layer",
        type=int,
        default=None,
        help="Also export currently valid FP16 K/V rows for this layer.",
    )
    ap.add_argument(
        "--llama-dump-step",
        type=int,
        default=None,
        help="Capture llama.cpp production-trajectory tensors at this zero-based step.",
    )
    ap.add_argument(
        "--llama-dump-dir",
        type=Path,
        default=None,
        help="Empty output directory for --llama-dump-step.",
    )
    ap.add_argument(
        "--llama-dump-names",
        default="",
        help="Comma-separated llama.cpp graph tensor names; empty captures the full graph.",
    )
    ap.add_argument(
        "--llama-dump-flash-inputs",
        action="store_true",
        help="Capture Q/K/V/mask inputs for a selected production flash-attention node.",
    )
    ap.add_argument(
        "--llama-profile-layers-out",
        type=Path,
        default=None,
        help=(
            "Write persistent llama.cpp decode layer-boundary wall times to "
            "a new CSV using the public cb_eval oracle hook."
        ),
    )
    ap.add_argument(
        "--diagnose-single-thread",
        action="store_true",
        help=(
            "When multi-thread controls are non-repeatable, run two additional "
            "one-thread SIMD trajectories and compare them with the parallel control."
        ),
    )
    args = ap.parse_args()

    model_dir = discover_ck_model_dir(args.model_dir)
    gguf_path = discover_gguf(args.gguf, model_dir)
    prompt_tokens = parse_tokens_csv(args.tokens)
    runtime_contract = load_runtime_contract(model_dir)
    llama_decode_mode = str(args.llama_decode_mode)
    if llama_decode_mode == "auto":
        prefill_policy = str(runtime_contract.get("prefill_policy") or "batched").strip().lower()
        llama_decode_mode = "hybrid" if prefill_policy == "sequential_decode" else "batched"
    stop_tokens = set(parse_tokens_csv(args.stop_tokens)) if str(args.stop_tokens).strip() else set()
    if args.execution_mode == "trajectory":
        if args.append_on_divergence not in {"stop", "llama"}:
            raise ValueError(
                "trajectory execution supports stopping at divergence or llama teacher forcing"
            )
        if args.ck_prefill_mode not in {"auto", "hybrid"} or llama_decode_mode != "hybrid":
            raise ValueError("trajectory execution requires hybrid CK and llama schedules")
        if args.ck_dump_step is not None and args.ck_dump_dir is None:
            raise ValueError("--ck-dump-step requires --ck-dump-dir")
        if args.llama_dump_step is not None and args.llama_dump_dir is None:
            raise ValueError("--llama-dump-step requires --llama-dump-dir")
        capture_requested = bool(
            args.ck_dump_step is not None or args.llama_dump_step is not None
        )
        estimated_logits_bytes = _trajectory_logits_bytes(
            model_dir, int(args.max_new_tokens)
        )
        storage_mode = _select_trajectory_storage(
            str(args.trajectory_storage),
            capture_requested=capture_requested,
            estimated_logits_bytes=estimated_logits_bytes,
        )
        trajectory_runner = (
            run_multitoken_trajectory_parity_streaming
            if storage_mode == "stream"
            else run_multitoken_trajectory_parity
        )
        trajectory_kwargs: dict[str, Any] = {
            "model_dir": model_dir,
            "gguf_path": gguf_path,
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": int(args.max_new_tokens),
            "ctx_len": int(args.ctx_len),
            "top_k": int(args.top_k),
            "threads": int(args.threads),
            "llama_no_repack": bool(args.llama_no_repack),
            "stop_token_ids": stop_tokens,
            "ck_runtime_so": args.ck_runtime_so,
            "llama_profile_layers_out": args.llama_profile_layers_out,
            "append_on_divergence": str(args.append_on_divergence),
        }
        if storage_mode == "memory":
            trajectory_kwargs.update({
                "ck_dump_step": args.ck_dump_step,
                "ck_dump_dir": args.ck_dump_dir,
                "ck_dump_layer": args.ck_dump_layer,
                "ck_dump_names": args.ck_dump_names,
                "ck_dump_format": args.ck_dump_format,
                "ck_dump_kv_layer": args.ck_dump_kv_layer,
                "llama_dump_step": args.llama_dump_step,
                "llama_dump_dir": args.llama_dump_dir,
                "llama_dump_names": args.llama_dump_names,
                "llama_dump_flash_inputs": bool(args.llama_dump_flash_inputs),
                "diagnose_single_thread": bool(args.diagnose_single_thread),
            })
        report = trajectory_runner(**trajectory_kwargs)
        report["trajectory_storage_selection"] = {
            "requested": str(args.trajectory_storage),
            "selected": storage_mode,
            "estimated_single_tensor_bytes": estimated_logits_bytes,
            "auto_threshold_bytes": _AUTO_STREAM_LOGITS_BYTES,
        }
    else:
        report = run_multitoken_parity(
            model_dir=model_dir,
            gguf_path=gguf_path,
            prompt_tokens=prompt_tokens,
            max_new_tokens=int(args.max_new_tokens),
            ctx_len=int(args.ctx_len),
            top_k=int(args.top_k),
            threads=int(args.threads),
            append_on_divergence=str(args.append_on_divergence),
            ck_prefill_mode=str(args.ck_prefill_mode),
            llama_decode_mode=llama_decode_mode,
            llama_no_repack=bool(args.llama_no_repack),
            stop_token_ids=stop_tokens,
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    neutrality = ((report.get("ck_capture") or {}).get("neutrality") or None)
    if args.summary:
        first = report.get("first_divergence")
        if first:
            print(
                "status=fail "
                f"step={first['step']} prefix_len={first['prefix_len']} "
                f"ck_next={first['ck_next']} llama_next={first['llama_next']} "
                f"llama_mode={llama_decode_mode} "
                f"ck_mode={args.ck_prefill_mode} "
                f"llama_no_repack={bool(args.llama_no_repack)} "
                f"cosine={first['cosine']:.6f} rmse={first['rmse']:.6f} "
                f"ck_margin={first['ck_top1_margin']:.6f} llama_margin={first['llama_top1_margin']:.6f} "
                f"topk_overlap={first['topk_overlap_count']}/{args.top_k}"
            )
        else:
            print(
                "status=pass "
                f"llama_mode={llama_decode_mode} "
                f"ck_mode={args.ck_prefill_mode} "
                f"llama_no_repack={bool(args.llama_no_repack)} "
                f"matched_stop_token={report.get('matched_stop_token')} "
                f"steps={len(report.get('steps', []))} "
                f"final_prefix_len={len(report.get('final_prefix', []))}"
            )
        if neutrality is not None:
            repeatability = neutrality.get("baseline_repeatability") or {}
            aggregate = neutrality.get("aggregate_capture") or {}
            print(
                f"xray_status={neutrality.get('status')} "
                f"accepted_mode={neutrality.get('accepted_mode')} "
                f"reason={neutrality.get('reason')} "
                f"baseline_exact={repeatability.get('exact')} "
                f"baseline_first_different_step={repeatability.get('first_different_step')} "
                f"capture_exact={aggregate.get('exact')} "
                f"capture_first_different_step={aggregate.get('first_different_step')}"
            )
    else:
        print(json.dumps(report))
    if neutrality is not None and neutrality.get("status") != "accepted":
        return 4
    return 0 if report.get("pass") else 3


if __name__ == "__main__":
    raise SystemExit(main())
