#!/usr/bin/env python3
from __future__ import annotations

"""
Tokenizer-free multimodal multi-token greedy parity probe.

This is the multimodal counterpart to compare_multitoken_logits_v8.py.  It
starts from a bridge_report.json + prefix.f32 produced by run_multimodal_bridge_v8,
does the circuit-selected CK mixed visual/text prefill schedule, then advances CK through real
ck_model_decode calls while llama.cpp replays the same generated suffix.
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import decoder_first_token_parity_v8 as first_token  # type: ignore  # noqa: E402
from gguf_tokenizer import GGUFTokenizer  # type: ignore  # noqa: E402


def _ck_logits_from_buffer(buf: Any, vocab_size: int) -> np.ndarray:
    return np.ctypeslib.as_array(buf, shape=(int(vocab_size),)).astype(np.float32, copy=True)


def _decode_topk(logits: np.ndarray, tokenizer: GGUFTokenizer, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(logits.size)
    k = max(1, min(int(top_k), n))
    top = np.argpartition(-logits, k - 1)[:k]
    top = top[np.argsort(-logits[top])]
    for idx in top.tolist():
        rows.append(
            {
                "token_id": int(idx),
                "logit": float(logits[int(idx)]),
                "token_text": tokenizer.decode([int(idx)], skip_special=False),
            }
        )
    return rows





def _set_process_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = str(value)
    try:
        libc = ctypes.CDLL(None)
        if value is None:
            libc.unsetenv.argtypes = [ctypes.c_char_p]
            libc.unsetenv(name.encode())
        else:
            libc.setenv.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            libc.setenv(name.encode(), str(value).encode(), 1)
    except Exception:
        pass


def _restore_process_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        _set_process_env(key, value)


def _single_hidden_file(root: Path) -> Path:
    files = sorted(root.glob("*.f32"))
    if len(files) != 1:
        raise RuntimeError(f"expected exactly one hidden dump in {root}, found {len(files)}")
    return files[0]


def _hidden_files_by_layer(root: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in sorted(root.glob("*.f32")):
        m = re.search(r"_layer_(\d+)_", path.name)
        if not m:
            continue
        layer = int(m.group(1))
        if layer in out:
            raise RuntimeError(f"duplicate hidden dump for layer {layer} in {root}")
        out[layer] = path
    if not out:
        raise RuntimeError(f"expected hidden dumps in {root}, found 0")
    return out


def _hidden_compare_many(persistent_dir: Path, full_dir: Path) -> list[dict[str, Any]]:
    persistent = _hidden_files_by_layer(persistent_dir)
    full = _hidden_files_by_layer(full_dir)
    rows: list[dict[str, Any]] = []
    for layer in sorted(set(persistent) | set(full)):
        if layer not in persistent or layer not in full:
            rows.append({
                "layer": int(layer),
                "status": "missing",
                "persistent_path": None if layer not in persistent else str(persistent[layer]),
                "full_replay_path": None if layer not in full else str(full[layer]),
            })
            continue
        rows.append({"layer": int(layer), **_hidden_compare(persistent[layer], full[layer])})
    return rows


def _hidden_compare(a_path: Path, b_path: Path) -> dict[str, Any]:
    a = np.fromfile(a_path, dtype=np.float32)
    b = np.fromfile(b_path, dtype=np.float32)
    if a.shape != b.shape:
        return {
            "status": "shape_mismatch",
            "persistent_path": str(a_path),
            "full_replay_path": str(b_path),
            "persistent_shape": list(a.shape),
            "full_replay_shape": list(b.shape),
        }
    diff = np.abs(a - b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom > 0.0 else 1.0
    return {
        "status": "ok",
        "persistent_path": str(a_path),
        "full_replay_path": str(b_path),
        "shape": list(a.shape),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))) if diff.size else 0.0,
        "cosine": cosine,
    }


def _hidden_full_replay_name(name: str) -> str:
    # Full replay runs through prefill, which exports last-row tensors with
    # the *_last suffix for token-major intermediates.
    if name.endswith("_last"):
        return name
    return f"{name}_last"

def _run_llama_greedy_sequence(inputs: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    helper = first_token.compare_first_token_logits_v7.ensure_llama_helper()
    with tempfile.TemporaryDirectory(prefix="llama_token_replay_v8_seq_") as td:
        tmp = Path(td)
        logits_out = tmp / "llama_final.f32"
        seq_out = tmp / "llama_seq.f32"
        cmd = [
            str(helper),
            "--model",
            str(inputs["gguf_path"]),
            "--ctx",
            str(int(inputs["ctx_len"])),
            "--top-k",
            str(int(args.top_k)),
            "--decode-mode",
            str(args.llama_decode_mode),
            "--logits-out",
            str(logits_out),
            "--logits-seq-out",
            str(seq_out),
            "--greedy-steps",
            str(int(args.max_new_tokens)),
        ]
        if inputs["tokens_before"]:
            cmd.extend(["--tokens-before", ",".join(str(t) for t in inputs["tokens_before"])])
            if inputs["tokens_after"]:
                cmd.extend(["--tokens-after", ",".join(str(t) for t in inputs["tokens_after"])])
        elif inputs["tokens_after"]:
            cmd.extend(["--tokens", ",".join(str(t) for t in inputs["tokens_after"])])
        if inputs["llama_prefix_path"] is not None:
            cmd.extend(["--prefix-f32", str(inputs["llama_prefix_path"])])
        if inputs["prefix_grid"] is not None:
            gx, gy = inputs["prefix_grid"]
            cmd.extend(["--prefix-grid-x", str(int(gx)), "--prefix-grid-y", str(int(gy))])
        if int(inputs["prefix_row_dim"]) > 0:
            cmd.extend(["--prefix-row-dim", str(int(inputs["prefix_row_dim"]))])
        cmd.extend(["--prefix-text-pos", str(int(inputs["prefix_text_pos"]))])
        if int(args.threads) > 0:
            cmd.extend(["--threads", str(int(args.threads))])
        if bool(args.llama_no_repack):
            cmd.append("--no-repack")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                "llama_token_replay greedy sequence failed\n"
                f"cmd: {' '.join(cmd)}\nrc: {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n"
            )
        payload = proc.stdout.strip().splitlines()[-1]
        meta = json.loads(payload)
        if not isinstance(meta, dict) or not meta.get("ok"):
            raise RuntimeError(f"llama_token_replay returned invalid payload: {payload}")
        n_vocab = int(meta.get("n_vocab", 0))
        steps = int(meta.get("greedy_steps", args.max_new_tokens) or 0)
        logits = np.fromfile(seq_out, dtype=np.float32)
        expected = int(steps) * int(n_vocab)
        if logits.size != expected:
            raise RuntimeError(f"llama sequence logits size mismatch: got={logits.size} expected={expected}")
        return {
            "meta": meta,
            "logits": logits.reshape((steps, n_vocab)),
        }

def _load_exact_decoder_runtime(
    gguf_path: Path,
    decoder_dir: Path,
    *,
    so_override: Path | None,
    manifest_map_override: Path | None,
) -> dict[str, Any]:
    layout_path = decoder_dir / "layout_decode.json"
    weights_bump = decoder_dir / "weights.bump"
    manifest_map = manifest_map_override or (decoder_dir / "weights_manifest.map")
    so_path = so_override or (decoder_dir / "libdecoder_v8.so")
    required = [layout_path, weights_bump, manifest_map, so_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "exact decoder runtime is incomplete; refusing to regenerate or guess: "
            + ", ".join(missing)
        )

    layout = first_token.bridge_runner_v8._load_layout(layout_path)
    cfg = dict(layout.get("config", {}) or {})
    embed_dim = int(cfg.get("embed_dim", 0) or 0)
    num_deepstack_layers = int(cfg.get("num_deepstack_layers", 0) or 0)
    input_embed_dim = int(cfg.get("input_embed_dim", 0) or 0)
    if input_embed_dim <= 0 and embed_dim > 0 and num_deepstack_layers > 0:
        input_embed_dim = embed_dim * (1 + num_deepstack_layers)
    if input_embed_dim <= 0:
        input_embed_dim = embed_dim
    return {
        "gguf": gguf_path,
        "workdir": decoder_dir,
        "so_path": so_path,
        "weights_bump": weights_bump,
        "manifest_map": manifest_map,
        "decode_layout_path": layout_path,
        "embed_dim": embed_dim,
        "input_embed_dim": input_embed_dim,
        "num_deepstack_layers": num_deepstack_layers,
        "context_length": int(cfg.get("context_length", cfg.get("context_len", 0)) or 0),
        "vocab_size": int(cfg.get("vocab_size", 0) or 0),
    }


def _runtime_evidence(runtime: dict[str, Any], *, exact_reuse: bool) -> dict[str, Any]:
    def describe(path_value: Any) -> dict[str, Any]:
        path = Path(path_value).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        return {"path": str(path), "sha256": digest}

    return {
        "exact_reuse": bool(exact_reuse),
        "shared_library": describe(runtime["so_path"]),
        "engine_library": describe(REPO_ROOT / "build" / "libckernel_engine.so"),
        "manifest_map": describe(runtime["manifest_map"]),
        "weights_bump_path": str(Path(runtime["weights_bump"]).resolve()),
        "context_length": int(runtime.get("context_length", 0) or 0),
    }


def _ck_threads(args: argparse.Namespace) -> int:
    explicit = getattr(args, "ck_threads", None)
    return int(args.threads if explicit is None else explicit)


def _prepare_inputs(args: argparse.Namespace, *, parity_dump: bool = False) -> dict[str, Any]:
    bridge_report = first_token._load_bridge_report(args.bridge_report.resolve())
    gguf_value = str(((bridge_report.get("decoder_runtime") or {}).get("gguf") or "")).strip()
    if not gguf_value:
        raise ValueError("bridge report does not include decoder_runtime.gguf; pass a fresh report")
    gguf_path = Path(gguf_value).resolve()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    exact_runtime = bool(getattr(args, "reuse_bridge_decoder_runtime_exact", False))
    if bool(getattr(args, "reuse_bridge_decoder_runtime", False)) or exact_runtime:
        decoder_workdir = str(((bridge_report.get("decoder_runtime") or {}).get("workdir") or "")).strip()
        if not decoder_workdir:
            raise ValueError("bridge report does not include decoder_runtime.workdir")
        decoder_dir = Path(decoder_workdir).resolve()
    else:
        decoder_dir = workdir / "decoder"

    requested_ctx_len = int(args.ctx_len or bridge_report.get("decoder_context_len") or 256)
    if exact_runtime:
        runtime = _load_exact_decoder_runtime(
            gguf_path,
            decoder_dir,
            so_override=getattr(args, "ck_runtime_so", None),
            manifest_map_override=getattr(args, "ck_runtime_manifest_map", None),
        )
    else:
        runtime = first_token.bridge_runner_v8._prepare_decoder_runtime(
            gguf_path,
            decoder_dir,
            parity_dump=bool(parity_dump),
            context_override=max(1, requested_ctx_len),
        )
    tokenizer = GGUFTokenizer.from_gguf(str(gguf_path))
    _, tokens_before, tokens_after, prompt_meta = first_token._resolve_prompt_token_segments(
        tokenizer,
        prompt=None,
        tokens_csv=None,
        tokens_before_csv=None,
        tokens_after_csv=None,
        bridge_report=bridge_report,
    )
    prefix_path = args.prefix_f32.resolve() if args.prefix_f32 is not None else None
    if prefix_path is None:
        report_prefix = str(bridge_report.get("prefix_dump_path") or "").strip()
        if report_prefix:
            prefix_path = Path(report_prefix).resolve()
    prefix_embeddings, prefix_tokens, prefix_row_dim, prefix_source = first_token._load_prefix_embeddings(
        prefix_path,
        0,
        int(runtime["embed_dim"]),
        int(runtime.get("input_embed_dim", 0) or 0),
        int(args.prefix_row_dim) if args.prefix_row_dim is not None else None,
    )
    total_prompt_tokens = len(tokens_before) + len(tokens_after)
    resolved_ctx_len = max(int(requested_ctx_len), int(prefix_tokens) + total_prompt_tokens, 1)
    if resolved_ctx_len != requested_ctx_len and exact_runtime:
        effective_context = int(runtime.get("context_length", 0) or 0)
        if effective_context < resolved_ctx_len:
            raise RuntimeError(
                "exact decoder runtime context is too small: "
                f"required={resolved_ctx_len} effective={effective_context}"
            )
    elif resolved_ctx_len != requested_ctx_len:
        runtime = first_token.bridge_runner_v8._prepare_decoder_runtime(
            gguf_path,
            decoder_dir,
            parity_dump=bool(parity_dump),
            context_override=resolved_ctx_len,
        )

    bridge_grid_x = None if bridge_report.get("prefix_grid_x") is None else int(bridge_report.get("prefix_grid_x"))
    bridge_grid_y = None if bridge_report.get("prefix_grid_y") is None else int(bridge_report.get("prefix_grid_y"))
    prefix_grid = first_token._resolve_prefix_grid(
        int(prefix_tokens),
        int(args.prefix_grid_x) if args.prefix_grid_x is not None else bridge_grid_x,
        int(args.prefix_grid_y) if args.prefix_grid_y is not None else bridge_grid_y,
    )
    prefix_text_pos = (
        int(args.prefix_text_pos)
        if args.prefix_text_pos is not None
        else (
            int(bridge_report.get("prefix_text_pos"))
            if bridge_report.get("prefix_text_pos") is not None
            else (
                len(tokens_before) + max(int(prefix_grid[0]), int(prefix_grid[1]))
                if prefix_grid is not None
                else len(tokens_before) + int(prefix_tokens)
            )
        )
    )
    bridge_contract = bridge_report.get("bridge_contract") or {}
    encoder_report = bridge_report.get("encoder_report") or {}
    prefix_decode_policy = (
        str(
            bridge_report.get("prefix_decode_policy")
            or bridge_contract.get("decode_policy")
            or encoder_report.get("prefix_decode_policy")
            or "causal_mixed_prefix"
        )
        .strip()
        .lower()
        or "causal_mixed_prefix"
    )
    llama_prefix_path = first_token._materialize_llama_prefix(
        prefix_embeddings,
        int(prefix_tokens),
        workdir,
        prefix_path=prefix_path,
    )
    return {
        "bridge_report": bridge_report,
        "gguf_path": gguf_path,
        "workdir": workdir,
        "runtime": runtime,
        "tokenizer": tokenizer,
        "tokens_before": [int(t) for t in tokens_before],
        "tokens_after": [int(t) for t in tokens_after],
        "prompt_meta": prompt_meta,
        "prefix_embeddings": prefix_embeddings,
        "prefix_tokens": int(prefix_tokens),
        "prefix_row_dim": int(prefix_row_dim),
        "prefix_source": prefix_source,
        "prefix_path": prefix_path,
        "llama_prefix_path": llama_prefix_path,
        "prefix_grid": prefix_grid,
        "prefix_text_pos": int(prefix_text_pos),
        "prefix_decode_policy": prefix_decode_policy,
        "ctx_len": int(resolved_ctx_len),
        "requested_ctx_len": int(requested_ctx_len),
    }


def _apply_prefix_decode_policy_env(inputs: dict[str, Any]) -> dict[str, str | None]:
    old = {
        "CK_BRIDGE_NONCAUSAL_VISUAL_CHUNK": os.environ.get("CK_BRIDGE_NONCAUSAL_VISUAL_CHUNK"),
        "CK_BRIDGE_VISUAL_START": os.environ.get("CK_BRIDGE_VISUAL_START"),
        "CK_BRIDGE_VISUAL_TOKENS": os.environ.get("CK_BRIDGE_VISUAL_TOKENS"),
    }
    policy = str(inputs.get("prefix_decode_policy") or "").strip().lower()
    if policy == "non_causal_visual_chunk":
        os.environ["CK_BRIDGE_NONCAUSAL_VISUAL_CHUNK"] = "1"
        os.environ["CK_BRIDGE_VISUAL_START"] = str(len(inputs.get("tokens_before") or []))
        os.environ["CK_BRIDGE_VISUAL_TOKENS"] = str(int(inputs.get("prefix_tokens") or 0))
    else:
        os.environ.pop("CK_BRIDGE_NONCAUSAL_VISUAL_CHUNK", None)
        os.environ.pop("CK_BRIDGE_VISUAL_START", None)
        os.environ.pop("CK_BRIDGE_VISUAL_TOKENS", None)
    return old


def _restore_prefix_decode_policy_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _capture_step_dump(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    step_index = int(args.dump_step)
    steps = list(report.get("steps") or [])
    if step_index < 0 or step_index >= len(steps):
        raise ValueError(f"--dump-step {step_index} is outside captured steps [0, {max(0, len(steps) - 1)}]")
    dump_dir = args.dump_dir
    if dump_dir is None:
        dump_dir = args.workdir / f"dump_step_{step_index:04d}"

    inputs = _prepare_inputs(args, parity_dump=True)
    row = dict(steps[step_index])
    generated_prefix = [int(t) for t in list(row.get("generated_prefix") or [])]
    replay_tokens_after = [int(t) for t in inputs["tokens_after"]] + generated_prefix
    old_env = _apply_prefix_decode_policy_env(inputs)
    try:
        ck, dump_report = first_token._capture_dump_compare(
            inputs["gguf_path"],
            inputs["runtime"],
            inputs["prefix_embeddings"],
            int(inputs["prefix_tokens"]),
            replay_tokens_after,
            tokens_before=inputs["tokens_before"],
            prefix_row_dim=int(inputs["prefix_row_dim"]),
            ctx_len=int(inputs["ctx_len"]),
            top_k=int(args.top_k),
            threads=_ck_threads(args),
            dump_root=dump_dir.resolve(),
            dump_names=str(args.dump_names),
            dump_pass=str(args.dump_pass),
            dump_atol=float(args.dump_atol),
            dump_rtol=float(args.dump_rtol),
            prefix_grid=inputs["prefix_grid"],
            prefix_text_pos=int(inputs["prefix_text_pos"]),
            ck_strict_parity=bool(args.ck_strict_parity),
        )
    finally:
        _restore_prefix_decode_policy_env(old_env)
    return {
        "step": int(step_index),
        "step_row": row,
        "dump_dir": str(dump_dir.resolve()),
        "replay_tokens_after_count": int(len(replay_tokens_after)),
        "generated_prefix": generated_prefix,
        "ck_top1": int((ck.get("comparison") or {}).get("top1_ck", ck.get("ck_top1", -1))),
        "dump": dump_report,
    }




def _capture_hidden_state_step(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    step_index = int(args.hidden_state_step)
    steps = list(report.get("steps") or [])
    if step_index < 0 or step_index >= len(steps):
        raise ValueError(f"--hidden-state-step {step_index} is outside captured steps [0, {max(0, len(steps) - 1)}]")
    row = dict(steps[step_index])
    generated_prefix = [int(t) for t in list(row.get("generated_prefix") or [])]
    if not generated_prefix:
        raise ValueError("hidden persistent decode capture needs a generated prefix; step 0 has no decode call to capture")

    inputs = _prepare_inputs(args)
    layer = int(args.hidden_state_layer)
    names = [item.strip() for item in str(args.hidden_state_names).split(",") if item.strip()]
    if not names:
        raise ValueError("--hidden-state-names did not contain any names")
    root = (args.hidden_state_dir or (args.workdir / f"hidden_state_step_{step_index:04d}")).resolve()
    root.mkdir(parents=True, exist_ok=True)

    env_keys = ["CK_DEBUG_EXPORT_HIDDEN", "CK_DEBUG_EXPORT_HIDDEN_NAME", "CK_DEBUG_EXPORT_HIDDEN_LAYER"]
    saved_env = {key: os.environ.get(key) for key in env_keys}
    results: list[dict[str, Any]] = []
    for name in names:
        persistent_dir = root / f"persistent_{name}"
        full_dir = root / f"full_replay_{name}"
        if persistent_dir.exists():
            import shutil
            shutil.rmtree(persistent_dir)
        if full_dir.exists():
            import shutil
            shutil.rmtree(full_dir)
        persistent_dir.mkdir(parents=True, exist_ok=True)
        full_dir.mkdir(parents=True, exist_ok=True)

        try:
            _restore_process_env({key: None for key in env_keys})
            lib, logits_buf, _vocab_size = _init_ck_state(inputs, bool(args.ck_strict_parity))
            try:
                last_i = len(generated_prefix) - 1
                for i, token in enumerate(generated_prefix):
                    if i == last_i:
                        _set_process_env("CK_DEBUG_EXPORT_HIDDEN", str(persistent_dir))
                        _set_process_env("CK_DEBUG_EXPORT_HIDDEN_NAME", name)
                        if layer >= 0:
                            _set_process_env("CK_DEBUG_EXPORT_HIDDEN_LAYER", str(layer))
                    rc = lib.ck_model_decode(ctypes.c_int32(int(token)), logits_buf)
                    if rc != 0:
                        raise RuntimeError(f"ck_model_decode failed rc={rc} while capturing persistent hidden state")
                    if i == last_i:
                        _restore_process_env({key: None for key in env_keys})
            finally:
                if hasattr(lib, "ck_model_free"):
                    try:
                        lib.ck_model_free()
                    except Exception:
                        pass

            replay_inputs = dict(inputs)
            replay_inputs["tokens_after"] = [int(t) for t in inputs["tokens_after"]] + generated_prefix
            _set_process_env("CK_DEBUG_EXPORT_HIDDEN", str(full_dir))
            _set_process_env("CK_DEBUG_EXPORT_HIDDEN_NAME", _hidden_full_replay_name(name))
            if layer >= 0:
                _set_process_env("CK_DEBUG_EXPORT_HIDDEN_LAYER", str(layer))
            lib2, _logits2, _vocab2 = _init_ck_state(replay_inputs, bool(args.ck_strict_parity))
            if hasattr(lib2, "ck_model_free"):
                try:
                    lib2.ck_model_free()
                except Exception:
                    pass
            _restore_process_env({key: None for key in env_keys})

            if layer < 0:
                per_layer = _hidden_compare_many(persistent_dir, full_dir)
                max_diff = max(float(r.get("max_abs_diff", 0.0)) for r in per_layer if r.get("status") == "ok") if per_layer else 0.0
                first_layer_issue = next((r for r in per_layer if r.get("status") != "ok" or float(r.get("max_abs_diff", 0.0)) > float(args.hidden_state_atol)), None)
                results.append({
                    "name": name,
                    "full_replay_name": _hidden_full_replay_name(name),
                    "status": "ok" if first_layer_issue is None else "fail",
                    "max_abs_diff": float(max_diff),
                    "first_layer_issue": first_layer_issue,
                    "layers": per_layer,
                })
            else:
                persistent_file = _single_hidden_file(persistent_dir)
                full_file = _single_hidden_file(full_dir)
                results.append({
                    "name": name,
                    "full_replay_name": _hidden_full_replay_name(name),
                    **_hidden_compare(persistent_file, full_file),
                })
        except Exception as exc:
            results.append({
                "name": name,
                "full_replay_name": _hidden_full_replay_name(name),
                "status": "error",
                "error": str(exc),
                "persistent_dir": str(persistent_dir),
                "full_replay_dir": str(full_dir),
            })
        finally:
            _restore_process_env(saved_env)

    first_issue = next((r for r in results if r.get("status") != "ok" or float(r.get("max_abs_diff", 0.0)) > float(args.hidden_state_atol)), None)
    return {
        "step": int(step_index),
        "layer": int(layer),
        "generated_prefix_count": int(len(generated_prefix)),
        "persistent_ck_next": int(row.get("ck_next", -1)),
        "persistent_ck_next_text": str(row.get("ck_next_text", "")),
        "llama_next": int(row.get("llama_next", -1)),
        "llama_next_text": str(row.get("llama_next_text", "")),
        "root": str(root),
        "atol": float(args.hidden_state_atol),
        "first_issue": first_issue,
        "results": results,
    }

def _capture_full_replay_step(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    step_index = int(args.full_replay_step)
    steps = list(report.get("steps") or [])
    if step_index < 0 or step_index >= len(steps):
        raise ValueError(f"--full-replay-step {step_index} is outside captured steps [0, {max(0, len(steps) - 1)}]")

    inputs = _prepare_inputs(args)
    tokenizer: GGUFTokenizer = inputs["tokenizer"]
    row = dict(steps[step_index])
    generated_prefix = [int(t) for t in list(row.get("generated_prefix") or [])]

    replay_inputs = dict(inputs)
    replay_inputs["tokens_after"] = [int(t) for t in inputs["tokens_after"]] + generated_prefix

    llama_args = argparse.Namespace(**vars(args))
    llama_args.max_new_tokens = max(int(args.max_new_tokens), step_index + 1)
    llama_seq = _run_llama_greedy_sequence(inputs, llama_args)
    llama_logits = llama_seq["logits"][step_index]

    lib, logits_buf, vocab_size = _init_ck_state(replay_inputs, bool(args.ck_strict_parity))
    try:
        replay_logits = _ck_logits_from_buffer(logits_buf, vocab_size)
    finally:
        if hasattr(lib, "ck_model_free"):
            try:
                lib.ck_model_free()
            except Exception:
                pass

    cmp = first_token.compare_first_token_logits_v7.compare_logits(replay_logits, llama_logits, int(args.top_k))
    replay_top1 = int(cmp["top1_ck"])
    llama_generated = list((llama_seq.get("meta") or {}).get("greedy_generated", []))
    llama_top1 = int(llama_generated[step_index]) if step_index < len(llama_generated) else int(cmp["top1_llama"])
    return {
        "step": int(step_index),
        "generated_prefix": generated_prefix,
        "replay_tokens_after_count": int(len(replay_inputs["tokens_after"])),
        "persistent_ck_next": int(row.get("ck_next", -1)),
        "persistent_ck_next_text": str(row.get("ck_next_text", "")),
        "llama_next": int(llama_top1),
        "llama_next_text": tokenizer.decode([int(llama_top1)], skip_special=False),
        "full_replay_ck_next": int(replay_top1),
        "full_replay_ck_next_text": tokenizer.decode([int(replay_top1)], skip_special=False),
        "full_replay_matches_persistent_ck": bool(replay_top1 == int(row.get("ck_next", -1))),
        "full_replay_matches_llama": bool(replay_top1 == int(llama_top1)),
        "comparison_vs_llama": {
            "cosine": float(cmp["cosine"]),
            "rmse": float(cmp["rmse"]),
            "mean_abs_diff": float(cmp["mean_abs_diff"]),
            "max_abs_diff": float(cmp["max_abs_diff"]),
            "topk_overlap_count": int(cmp["topk_overlap_count"]),
            "topk_overlap_ratio": float(cmp["topk_overlap_ratio"]),
            "ck_top1_margin": float(cmp.get("ck_top1_margin", 0.0)),
            "llama_top1_margin": float(cmp.get("llama_top1_margin", 0.0)),
        },
        "full_replay_ck_topk": _decode_topk(replay_logits, tokenizer, int(args.top_k)),
        "llama_topk": _decode_topk(llama_logits, tokenizer, int(args.top_k)),
    }


def _set_ck_strict_parity(lib: Any, strict_parity: bool) -> None:
    value = 1 if strict_parity else 0
    if hasattr(lib, "ck_set_strict_parity"):
        lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
        lib.ck_set_strict_parity.restype = None
        lib.ck_set_strict_parity(value)
        return

    engine_so = REPO_ROOT / "build" / "libckernel_engine.so"
    if not engine_so.exists():
        return
    engine = ctypes.CDLL(str(engine_so))
    if hasattr(engine, "ck_set_strict_parity"):
        engine.ck_set_strict_parity.argtypes = [ctypes.c_int]
        engine.ck_set_strict_parity.restype = None
        engine.ck_set_strict_parity(value)


def _init_ck_state(inputs: dict[str, Any], strict_parity: bool) -> tuple[Any, Any, int]:
    runtime = inputs["runtime"]
    model_so = Path(runtime["so_path"])
    lib = first_token.bridge_runner_v8._load_decoder_lib(model_so)
    rc = lib.ck_model_init_with_manifest(
        str(runtime["weights_bump"]).encode(),
        str(runtime["manifest_map"]).encode(),
    )
    if rc != 0:
        raise RuntimeError(f"decoder init failed with rc={rc}")
    _set_ck_strict_parity(lib, strict_parity)

    vocab_size = int(lib.ck_model_get_vocab_size())
    if vocab_size <= 0:
        vocab_size = int(runtime["vocab_size"])
    logits = (ctypes.c_float * vocab_size)()

    tokens_before = inputs["tokens_before"]
    tokens_after = inputs["tokens_after"]
    prefix_tokens = int(inputs["prefix_tokens"])
    prefix_row_dim = int(inputs["prefix_row_dim"])
    prefix_embeddings: array = inputs["prefix_embeddings"]

    prefix_ptr = None
    if prefix_tokens > 0:
        expected = prefix_tokens * prefix_row_dim
        if len(prefix_embeddings) != expected:
            raise RuntimeError(f"prefix float count mismatch: got={len(prefix_embeddings)} expected={expected}")
        prefix_ptr = (ctypes.c_float * len(prefix_embeddings))(*prefix_embeddings)
    before_arr = (ctypes.c_int32 * len(tokens_before))(*tokens_before) if tokens_before else None
    after_arr = (ctypes.c_int32 * len(tokens_after))(*tokens_after) if tokens_after else None
    grid = inputs["prefix_grid"]
    old_env = _apply_prefix_decode_policy_env(inputs)
    try:
        if tokens_before and hasattr(lib, "ck_model_forward_segments_grid_ex"):
            grid_x, grid_y = grid if grid is not None else (0, 0)
            rc = lib.ck_model_forward_segments_grid_ex(
                before_arr,
                len(tokens_before),
                prefix_ptr,
                prefix_tokens,
                prefix_row_dim,
                int(grid_x),
                int(grid_y),
                int(inputs["prefix_text_pos"]),
                after_arr,
                len(tokens_after),
                logits,
            )
        elif hasattr(lib, "ck_model_forward_mixed_grid_ex") and grid is not None:
            grid_x, grid_y = grid
            rc = lib.ck_model_forward_mixed_grid_ex(
                prefix_ptr,
                prefix_tokens,
                prefix_row_dim,
                int(grid_x),
                int(grid_y),
                int(inputs["prefix_text_pos"]),
                after_arr,
                len(tokens_after),
                logits,
            )
        elif hasattr(lib, "ck_model_forward_mixed_ex"):
            rc = lib.ck_model_forward_mixed_ex(prefix_ptr, prefix_tokens, prefix_row_dim, after_arr, len(tokens_after), logits)
        else:
            rc = lib.ck_model_forward_mixed(prefix_ptr, prefix_tokens, after_arr, len(tokens_after), logits)
    finally:
        _restore_prefix_decode_policy_env(old_env)
    if rc != 0:
        raise RuntimeError(f"decoder forward_mixed failed rc={rc}")
    return lib, logits, vocab_size


def run_multimodal_multitoken_parity(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _prepare_inputs(args)
    tokenizer: GGUFTokenizer = inputs["tokenizer"]
    llama_seq = _run_llama_greedy_sequence(inputs, args)
    lib, logits_buf, vocab_size = _init_ck_state(inputs, bool(args.ck_strict_parity))
    generated: list[int] = []
    llama_generated = [int(t) for t in list((llama_seq.get("meta") or {}).get("greedy_generated", []))]
    stop_token_ids = _resolve_stop_token_ids(inputs["bridge_report"])
    stop_reason: str | None = None
    steps: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    try:
        for step in range(max(1, int(args.max_new_tokens))):
            ck_logits = _ck_logits_from_buffer(logits_buf, vocab_size)
            if step >= int(llama_seq["logits"].shape[0]):
                raise RuntimeError(f"llama sequence ended before step={step}")
            llama_logits = llama_seq["logits"][step]
            cmp = first_token.compare_first_token_logits_v7.compare_logits(ck_logits, llama_logits, int(args.top_k))
            ck_next = int(cmp["top1_ck"])
            llama_next = int(llama_generated[step]) if step < len(llama_generated) else int(cmp["top1_llama"])
            top1_match = ck_next == llama_next
            row = {
                "step": int(step),
                "generated_prefix": [int(t) for t in generated],
                "prefix_len_after_image": int(len(inputs["tokens_after"]) + len(generated)),
                "ck_next": ck_next,
                "llama_next": llama_next,
                "ck_next_text": tokenizer.decode([ck_next], skip_special=False),
                "llama_next_text": tokenizer.decode([llama_next], skip_special=False),
                "top1_match": bool(top1_match),
                "llama_no_repack": bool(args.llama_no_repack),
                "cosine": float(cmp["cosine"]),
                "rmse": float(cmp["rmse"]),
                "mean_abs_diff": float(cmp["mean_abs_diff"]),
                "max_abs_diff": float(cmp["max_abs_diff"]),
                "ck_top1_margin": float(cmp.get("ck_top1_margin", 0.0)),
                "llama_top1_margin": float(cmp.get("llama_top1_margin", 0.0)),
                "topk_overlap_count": int(cmp["topk_overlap_count"]),
                "topk_overlap_ratio": float(cmp["topk_overlap_ratio"]),
                "ck_topk_ids": list(cmp["ck_topk_ids"]),
                "llama_topk_ids": list(cmp["llama_topk_ids"]),
                "ck_topk": _decode_topk(ck_logits, tokenizer, int(args.top_k)),
                "llama_topk": _decode_topk(llama_logits, tokenizer, int(args.top_k)),
                "topk_logits": list(cmp.get("topk_logits", [])),
            }
            steps.append(row)
            if not top1_match and first_divergence is None:
                first_divergence = row
                if args.append_on_divergence == "stop":
                    break
            if _is_matched_stop_token(ck_next, llama_next, stop_token_ids):
                stop_reason = "matched_stop_token"
                break
            if top1_match or args.append_on_divergence == "llama":
                next_token = llama_next
            elif args.append_on_divergence == "ck":
                next_token = ck_next
            else:
                break
            generated.append(int(next_token))
            if step + 1 >= int(args.max_new_tokens):
                break
            rc = lib.ck_model_decode(ctypes.c_int32(int(next_token)), logits_buf)
            if rc != 0:
                raise RuntimeError(f"ck_model_decode failed rc={rc} at step={step}")
    finally:
        if hasattr(lib, "ck_model_free"):
            try:
                lib.ck_model_free()
            except Exception:
                pass

    return {
        "status": "pass" if first_divergence is None else "fail",
        "pass": first_divergence is None,
        "bridge_report_path": str(args.bridge_report.resolve()),
        "gguf_path": str(inputs["gguf_path"]),
        "workdir": str(inputs["workdir"]),
        "ctx_len": int(inputs["ctx_len"]),
        "requested_ctx_len": int(inputs["requested_ctx_len"]),
        "threads": int(args.threads),
        "ck_threads": _ck_threads(args),
        "thread_config": {
            "llama_cpp": int(args.threads),
            "ck_runtime": _ck_threads(args),
        },
        "top_k": int(args.top_k),
        "llama_decode_mode": str(args.llama_decode_mode),
        "llama_greedy_generated": llama_generated,
        "append_on_divergence": str(args.append_on_divergence),
        "ck_runtime": _runtime_evidence(
            inputs["runtime"],
            exact_reuse=bool(getattr(args, "reuse_bridge_decoder_runtime_exact", False)),
        ),
        "stop_token_ids": sorted(stop_token_ids),
        "stop_reason": stop_reason,
        "prompt_tokens_before_image": inputs["tokens_before"],
        "prompt_tokens_after_image": inputs["tokens_after"],
        "prefix": {
            "source": str(inputs["prefix_source"]),
            "tokens": int(inputs["prefix_tokens"]),
            "row_dim": int(inputs["prefix_row_dim"]),
            "path": None if inputs["prefix_path"] is None else str(inputs["prefix_path"]),
            "grid": None if inputs["prefix_grid"] is None else [int(inputs["prefix_grid"][0]), int(inputs["prefix_grid"][1])],
            "text_pos": int(inputs["prefix_text_pos"]),
        },
        "generated_shared_tokens": [int(t) for t in generated],
        "generated_shared_text": tokenizer.decode(generated, skip_special=False) if generated else "",
        "first_divergence": first_divergence,
        "steps": steps,
    }


def _resolve_stop_token_ids(bridge_report: dict[str, Any]) -> set[int]:
    stop_ids = {int(token) for token in list(bridge_report.get("stop_token_ids") or [])}
    if not stop_ids and bridge_report.get("eos_token_id") is not None:
        stop_ids.add(int(bridge_report["eos_token_id"]))
    return stop_ids


def _is_matched_stop_token(ck_token: int, llama_token: int, stop_token_ids: set[int]) -> bool:
    return ck_token == llama_token and ck_token in stop_token_ids


def main() -> int:
    ap = argparse.ArgumentParser(description="Multimodal multi-token greedy parity (CK vs llama.cpp)")
    ap.add_argument("--bridge-report", required=True, type=Path)
    ap.add_argument("--prefix-f32", type=Path, default=None)
    ap.add_argument("--prefix-row-dim", type=int, default=None)
    ap.add_argument("--prefix-grid-x", type=int, default=None)
    ap.add_argument("--prefix-grid-y", type=int, default=None)
    ap.add_argument("--prefix-text-pos", type=int, default=None)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--reuse-bridge-decoder-runtime", action="store_true", help="Reuse decoder_runtime.workdir from bridge_report instead of creating a new decoder copy under --workdir")
    ap.add_argument(
        "--reuse-bridge-decoder-runtime-exact",
        action="store_true",
        help="Trust the existing bridge decoder artifacts without regeneration; fail if any required artifact is missing",
    )
    ap.add_argument("--ck-runtime-so", type=Path, default=None, help="Explicit CK decoder shared library for exact-runtime parity")
    ap.add_argument("--ck-runtime-manifest-map", type=Path, default=None, help="Explicit CK decoder manifest map for exact-runtime parity")
    ap.add_argument("--ctx-len", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1, help="llama.cpp oracle threads")
    ap.add_argument(
        "--ck-threads",
        type=int,
        default=None,
        help="CK runtime threads; defaults to --threads when omitted",
    )
    ap.add_argument("--llama-decode-mode", choices=["batched", "sequential"], default="sequential")
    ap.add_argument("--llama-no-repack", action="store_true", help="Disable llama.cpp CPU tensor repacking in the replay helper")
    ap.add_argument("--llama-repack-fallback", action=argparse.BooleanOptionalAction, default=True, help="Retry a failed llama replay step with --no-repack")
    ap.add_argument("--append-on-divergence", choices=["stop", "llama", "ck"], default="stop")
    ap.add_argument("--ck-strict-parity", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--dump-step", type=int, default=None, help="Capture CK-vs-llama tensor dumps at this generated step")
    ap.add_argument("--full-replay-step", type=int, default=None, help="Compare a generated step by full mixed-prefix replay instead of incremental CK decode, without tensor dumps")
    ap.add_argument("--dump-dir", type=Path, default=None, help="Directory for --dump-step tensor dumps")
    ap.add_argument(
        "--dump-names",
        type=str,
        default="Qcur-0,Kcur-0,Vcur-0,Qcur_normed-0,Kcur_normed-0,kqv_out-0",
        help="Comma-separated llama dump names for --dump-step",
    )
    ap.add_argument("--dump-pass", choices=("all", "prefill", "decode"), default="decode")
    ap.add_argument("--dump-atol", type=float, default=1.0e-4)
    ap.add_argument("--dump-rtol", type=float, default=1.0e-3)
    ap.add_argument("--hidden-state-step", type=int, default=None, help="Compare CK persistent decode hidden tensors against CK full replay at this generated step")
    ap.add_argument("--hidden-state-layer", type=int, default=0)
    ap.add_argument("--hidden-state-names", type=str, default="attn_out,out_proj,after_attn,layer_out")
    ap.add_argument("--hidden-state-dir", type=Path, default=None)
    ap.add_argument("--hidden-state-atol", type=float, default=1.0e-5)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    ck_threads = _ck_threads(args)
    if ck_threads > 0:
        os.environ["CK_NUM_THREADS"] = str(ck_threads)
        os.environ["OMP_NUM_THREADS"] = str(ck_threads)
    report = run_multimodal_multitoken_parity(args)
    if args.dump_step is not None:
        report["step_dump"] = _capture_step_dump(report, args)
    if args.full_replay_step is not None:
        report["full_replay_step"] = _capture_full_replay_step(report, args)
    if args.hidden_state_step is not None:
        report["hidden_state_step"] = _capture_hidden_state_step(report, args)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.summary:
        first = report.get("first_divergence")
        if first:
            print(
                "status=fail "
                f"step={first['step']} ck_next={first['ck_next']}({first['ck_next_text']!r}) "
                f"llama_next={first['llama_next']}({first['llama_next_text']!r}) "
                f"cosine={first['cosine']:.6f} rmse={first['rmse']:.6f} "
                f"topk_overlap={first['topk_overlap_count']}/{args.top_k}"
            )
        else:
            print(
                "status=pass "
                f"steps={len(report.get('steps', []))} "
                f"generated={report.get('generated_shared_text', '')!r}"
            )
        if report.get("step_dump"):
            dump = (report["step_dump"] or {}).get("dump") or {}
            first_issue = dump.get("first_issue")
            summary = dump.get("summary") or {}
            if first_issue:
                print(
                    "dump=fail "
                    f"step={report['step_dump']['step']} "
                    f"layer={first_issue.get('layer')} op={first_issue.get('op')} "
                    f"status={first_issue.get('status')} "
                    f"max_abs_diff={float(first_issue.get('max_abs_diff', 0.0)):.6g} "
                    f"summary={summary}"
                )
            else:
                print(f"dump={dump.get('status', 'ok')} step={report['step_dump']['step']} summary={summary}")
        if report.get("full_replay_step"):
            replay = report["full_replay_step"]
            cmp = replay.get("comparison_vs_llama") or {}
            print(
                "full_replay "
                f"step={replay['step']} "
                f"persistent_ck={replay['persistent_ck_next']}({replay['persistent_ck_next_text']!r}) "
                f"full_replay_ck={replay['full_replay_ck_next']}({replay['full_replay_ck_next_text']!r}) "
                f"llama={replay['llama_next']}({replay['llama_next_text']!r}) "
                f"matches_persistent={replay['full_replay_matches_persistent_ck']} "
                f"matches_llama={replay['full_replay_matches_llama']} "
                f"cosine={float(cmp.get('cosine', 0.0)):.6f} "
                f"rmse={float(cmp.get('rmse', 0.0)):.6f} "
                f"topk_overlap={int(cmp.get('topk_overlap_count', 0))}/{args.top_k}"
            )
    else:
        print(json.dumps(report, indent=2))
    return 0 if report.get("pass") else 3


if __name__ == "__main__":
    raise SystemExit(main())
