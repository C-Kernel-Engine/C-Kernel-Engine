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
import struct
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


def _hidden_token_position(path: Path) -> int | None:
    match = re.search(r"(?:^|_)tok_(\d+)(?:_|$)", path.name)
    return int(match.group(1)) if match else None


def _select_final_hidden_file(files: list[Path], *, context: str) -> Path:
    if not files:
        raise RuntimeError(f"expected hidden dump for {context}, found 0")
    if len(files) == 1:
        return files[0]

    positioned = [(path, _hidden_token_position(path)) for path in files]
    if any(position is None for _, position in positioned):
        raise RuntimeError(
            f"multiple hidden dumps for {context} lack unambiguous token positions: "
            + ", ".join(path.name for path, _ in positioned)
        )
    final_position = max(int(position) for _, position in positioned if position is not None)
    final = [path for path, position in positioned if position == final_position]
    if len(final) != 1:
        raise RuntimeError(
            f"multiple hidden dumps for {context} share final token position {final_position}: "
            + ", ".join(path.name for path in final)
        )
    return final[0]


def _hidden_named_files(root: Path, name: str | None) -> list[Path]:
    files = sorted(root.glob("*.f32"))
    if name is None:
        return files
    suffix = f"_{name}.f32"
    return [path for path in files if path.name.endswith(suffix)]


def _single_hidden_file(root: Path, name: str | None = None) -> Path:
    return _select_final_hidden_file(
        _hidden_named_files(root, name), context=f"{name or 'hidden tensor'} in {root}"
    )


def _hidden_files_by_layer(root: Path, name: str | None = None) -> dict[int, Path]:
    candidates: dict[int, list[Path]] = {}
    for path in _hidden_named_files(root, name):
        m = re.search(r"_layer_(\d+)_", path.name)
        if not m:
            continue
        layer = int(m.group(1))
        candidates.setdefault(layer, []).append(path)
    if not candidates:
        raise RuntimeError(f"expected hidden dumps in {root}, found 0")
    return {
        layer: _select_final_hidden_file(paths, context=f"layer {layer} in {root}")
        for layer, paths in candidates.items()
    }


def _hidden_compare_many(
    persistent_dir: Path,
    full_dir: Path,
    persistent_name: str | None = None,
    full_name: str | None = None,
) -> list[dict[str, Any]]:
    persistent = _hidden_files_by_layer(persistent_dir, persistent_name)
    full = _hidden_files_by_layer(full_dir, full_name)
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


_HIDDEN_EXPORT_CALL_RE = re.compile(
    r"ck_debug_export_hidden\s*\(\s*model\s*,\s*(\d+)\s*,\s*\"([^\"]+)\""
)


def _hidden_export_catalog(runtime: dict[str, Any]) -> dict[str, list[int]]:
    """Inventory the exact hidden exporters compiled into a decoder runtime."""
    c_path_value = runtime.get("c_path")
    if c_path_value is None:
        c_path_value = Path(runtime["workdir"]) / "decoder_v8.c"
    c_path = Path(c_path_value).resolve()
    if not c_path.is_file():
        raise FileNotFoundError(
            "hidden-state preflight requires the generated decoder source matching the runtime: "
            f"{c_path}"
        )

    catalog: dict[str, set[int]] = {}
    for layer_text, name in _HIDDEN_EXPORT_CALL_RE.findall(c_path.read_text(encoding="utf-8")):
        catalog.setdefault(name, set()).add(int(layer_text))
    if not catalog:
        raise RuntimeError(f"generated decoder has no hidden-state exporters: {c_path}")
    return {name: sorted(layers) for name, layers in sorted(catalog.items())}


def _validate_hidden_capture_request(
    runtime: dict[str, Any], names: list[str], layer: int
) -> dict[str, list[int]]:
    catalog = _hidden_export_catalog(runtime)
    issues: list[str] = []
    for name in names:
        replay_name = _hidden_full_replay_name(name)
        if name not in catalog:
            issues.append(f"decode exporter {name!r} does not exist")
            continue
        if replay_name not in catalog:
            issues.append(f"full-replay exporter {replay_name!r} does not exist")
            continue
        if layer >= 0 and layer not in catalog[name]:
            issues.append(f"decode exporter {name!r} is unavailable at layer {layer}")
        if layer >= 0 and layer not in catalog[replay_name]:
            issues.append(f"full-replay exporter {replay_name!r} is unavailable at layer {layer}")
    if issues:
        base_names = sorted(name for name in catalog if not name.endswith("_last"))
        raise ValueError(
            "hidden-state checkpoint preflight failed before model execution: "
            + "; ".join(issues)
            + ". Valid base names: "
            + ",".join(base_names)
        )
    return catalog

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
        "c_path": decoder_dir / "decoder_v8.c",
        "weights_bump": weights_bump,
        "manifest_map": manifest_map,
        "decode_layout_path": layout_path,
        "embed_dim": embed_dim,
        "input_embed_dim": input_embed_dim,
        "num_deepstack_layers": num_deepstack_layers,
        "context_length": int(cfg.get("context_length", cfg.get("context_len", 0)) or 0),
        "vocab_size": int(cfg.get("vocab_size", 0) or 0),
    }


def _runtime_evidence(
    runtime: dict[str, Any],
    *,
    exact_reuse: bool,
    engine_so: Path | None = None,
) -> dict[str, Any]:
    def describe(path_value: Any) -> dict[str, Any]:
        path = Path(path_value).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        result: dict[str, Any] = {"path": str(path), "sha256": digest}
        build_path = path.with_suffix(path.suffix + ".build.json")
        if build_path.is_file():
            try:
                result["build"] = json.loads(build_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                result["build_metadata_error"] = str(exc)
        if path.is_file():
            probe = subprocess.run(
                ["readelf", "--string-dump=.comment", str(path)],
                text=True,
                capture_output=True,
                check=False,
            )
            if probe.returncode == 0:
                comments = [
                    line.split("]", 1)[-1].strip()
                    for line in probe.stdout.splitlines()
                    if "]" in line and line.split("]", 1)[-1].strip()
                ]
                result["elf_compiler_comments"] = comments
            else:
                result["elf_comment_error"] = probe.stderr.strip() or f"readelf rc={probe.returncode}"
        return result

    requested_engine = Path(engine_so or REPO_ROOT / "build" / "libckernel_engine.so").resolve()
    runtime_engine = Path(runtime["so_path"]).resolve().parent / "libckernel_engine.so"
    if not runtime_engine.is_file():
        runtime_engine = requested_engine
    return {
        "exact_reuse": bool(exact_reuse),
        "shared_library": describe(runtime["so_path"]),
        "engine_library": describe(runtime_engine),
        "engine_library_requested": describe(requested_engine),
        "tokenizer_library": describe(
            Path(runtime["so_path"]).resolve().parent / "libckernel_tokenizer.so"
        ),
        "manifest_map": describe(runtime["manifest_map"]),
        "weights_bump_path": str(Path(runtime["weights_bump"]).resolve()),
        "context_length": int(runtime.get("context_length", 0) or 0),
    }


def _ck_environment_evidence() -> dict[str, str]:
    evidence: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if not key.startswith("CK_"):
            continue
        if any(marker in key for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")):
            evidence[key] = "<redacted>"
        else:
            evidence[key] = str(value)
    return evidence


def _ck_threads(args: argparse.Namespace) -> int:
    explicit = getattr(args, "ck_threads", None)
    return int(args.threads if explicit is None else explicit)


def _resolve_decoder_runtime_request(
    args: argparse.Namespace,
    bridge_report: dict[str, Any],
    workdir: Path,
) -> tuple[Path, bool]:
    explicit_so = getattr(args, "ck_runtime_so", None)
    explicit_manifest = getattr(args, "ck_runtime_manifest_map", None)
    if bool(explicit_so) != bool(explicit_manifest):
        raise ValueError(
            "exact runtime reuse requires both --ck-runtime-so and "
            "--ck-runtime-manifest-map"
        )
    if explicit_so and explicit_manifest:
        return Path(explicit_so).resolve().parent, True

    exact_runtime = bool(getattr(args, "reuse_bridge_decoder_runtime_exact", False))
    if bool(getattr(args, "reuse_bridge_decoder_runtime", False)) or exact_runtime:
        decoder_workdir = str(
            ((bridge_report.get("decoder_runtime") or {}).get("workdir") or "")
        ).strip()
        if not decoder_workdir:
            raise ValueError("bridge report does not include decoder_runtime.workdir")
        return Path(decoder_workdir).resolve(), exact_runtime
    return workdir / "decoder", False


def _prepare_inputs(args: argparse.Namespace, *, parity_dump: bool = False) -> dict[str, Any]:
    bridge_report = first_token._load_bridge_report(args.bridge_report.resolve())
    gguf_value = str(((bridge_report.get("decoder_runtime") or {}).get("gguf") or "")).strip()
    if not gguf_value:
        raise ValueError("bridge report does not include decoder_runtime.gguf; pass a fresh report")
    gguf_path = Path(gguf_value).resolve()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    decoder_dir, exact_runtime = _resolve_decoder_runtime_request(
        args, bridge_report, workdir
    )

    requested_ctx_len = int(args.ctx_len or bridge_report.get("decoder_context_len") or 256)
    if exact_runtime and not parity_dump:
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

    requested_engine = getattr(args, "ck_engine_so", None)
    if requested_engine is not None:
        requested_engine = Path(requested_engine).resolve()
        first_token.bridge_runner_v8._sync_runtime_engine(
            requested_engine, Path(runtime["so_path"])
        )
        # The runtime dictionary is passed through replay and granular dump
        # helpers. Preserve the requested engine here so those nested paths
        # cannot silently fall back to the repository build library.
        runtime["engine_so"] = str(requested_engine)

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
        "exact_runtime": bool(exact_runtime and not parity_dump),
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
    dump_source = Path(inputs["runtime"].get("c_path") or "")
    if not dump_source.is_file() or "#define CK_PARITY_DUMP 1" not in dump_source.read_text(encoding="utf-8"):
        raise RuntimeError(
            "granular capture requires a dedicated decoder runtime compiled with CK_PARITY_DUMP; "
            f"instrumented source is missing or invalid: {dump_source}"
        )
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
            llama_decode_mode=str(args.llama_decode_mode),
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
        "instrumented_runtime": {
            "c_path": str(dump_source.resolve()),
            "so_path": str(Path(inputs["runtime"]["so_path"]).resolve()),
            "parity_dump": True,
        },
        "dump": dump_report,
    }


def _resolve_dump_step(
    report: dict[str, Any], requested_step: int | None, dump_first_divergence: bool
) -> int | None:
    if requested_step is not None and dump_first_divergence:
        raise ValueError("--dump-step and --dump-first-divergence are mutually exclusive")
    if not dump_first_divergence:
        return requested_step
    first = report.get("first_divergence")
    if not isinstance(first, dict) or "step" not in first:
        raise RuntimeError("--dump-first-divergence requested, but the parity run did not diverge")
    return int(first["step"])


def _describe_kv_f16_difference(
    persistent_bytes: bytes,
    replay_bytes: bytes,
    first_byte: int | None,
) -> dict[str, Any] | None:
    header_size = 8 * 4
    if first_byte is None or first_byte < header_size:
        return None
    if len(persistent_bytes) < header_size or len(replay_bytes) < header_size:
        return None
    persistent_header = struct.unpack_from("<8I", persistent_bytes)
    replay_header = struct.unpack_from("<8I", replay_bytes)
    if persistent_header != replay_header or persistent_header[0] != 0x564B5843:
        return None

    _magic, version, layer, valid_tokens, num_heads, max_seq_len, head_dim, _reserved = persistent_header
    if version != 1 or valid_tokens == 0 or num_heads == 0 or head_dim == 0:
        return None
    element_index = (first_byte - header_size) // 2
    values_per_head = valid_tokens * head_dim
    values_per_kind = num_heads * values_per_head
    if element_index < 0 or element_index >= 2 * values_per_kind:
        return None

    kind_index, kind_offset = divmod(element_index, values_per_kind)
    head, head_offset = divmod(kind_offset, values_per_head)
    token, channel = divmod(head_offset, head_dim)
    value_offset = header_size + element_index * 2
    persistent_bits = struct.unpack_from("<H", persistent_bytes, value_offset)[0]
    replay_bits = struct.unpack_from("<H", replay_bytes, value_offset)[0]
    persistent_value = struct.unpack("<e", struct.pack("<H", persistent_bits))[0]
    replay_value = struct.unpack("<e", struct.pack("<H", replay_bits))[0]
    return {
        "kind": "K" if kind_index == 0 else "V",
        "layer": int(layer),
        "head": int(head),
        "token": int(token),
        "channel": int(channel),
        "element_index": int(element_index),
        "byte_in_element": int((first_byte - header_size) % 2),
        "persistent_fp16_bits": int(persistent_bits),
        "full_replay_fp16_bits": int(replay_bits),
        "fp16_bit_distance": int(abs(persistent_bits - replay_bits)),
        "persistent_value": float(persistent_value),
        "full_replay_value": float(replay_value),
        "valid_tokens": int(valid_tokens),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "max_seq_len": int(max_seq_len),
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
    catalog = _validate_hidden_capture_request(inputs["runtime"], names, layer)
    root = (args.hidden_state_dir or (args.workdir / f"hidden_state_step_{step_index:04d}")).resolve()
    root.mkdir(parents=True, exist_ok=True)

    env_keys = [
        "CK_DEBUG_EXPORT_HIDDEN",
        "CK_DEBUG_EXPORT_HIDDEN_NAME",
        "CK_DEBUG_EXPORT_HIDDEN_NAMES",
        "CK_DEBUG_EXPORT_HIDDEN_LAYER",
    ]
    saved_env = {key: os.environ.get(key) for key in env_keys}
    results: list[dict[str, Any]] = []
    persistent_dir = root / "persistent"
    full_dir = root / "full_replay"
    import shutil
    for directory in (persistent_dir, full_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    capture_error: Exception | None = None
    kv_comparison: dict[str, Any] | None = None

    def export_kv(lib: Any, path: Path) -> dict[str, Any]:
        if not hasattr(lib, "ck_model_debug_export_kv_f16"):
            return {
                "status": "unavailable",
                "reason": "runtime lacks bounded ck_model_debug_export_kv_f16 X-ray ABI",
                "path": str(path),
            }
        lib.ck_model_debug_export_kv_f16.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.ck_model_debug_export_kv_f16.restype = ctypes.c_int
        rc = int(lib.ck_model_debug_export_kv_f16(str(path).encode("utf-8"), int(layer)))
        if rc != 0:
            return {
                "status": "error",
                "reason": f"FP16 KV export failed rc={rc} layer={layer}",
                "path": str(path),
            }
        return {"status": "ok", "path": str(path)}

    persistent_kv_export: dict[str, Any] | None = None
    replay_kv_export: dict[str, Any] | None = None
    try:
        _restore_process_env({key: None for key in env_keys})
        lib, logits_buf, _vocab_size = _init_ck_state(
            inputs, bool(args.ck_strict_parity), getattr(args, "ck_engine_so", None)
        )
        try:
            last_i = len(generated_prefix) - 1
            for i, token in enumerate(generated_prefix):
                if i == last_i:
                    _set_process_env("CK_DEBUG_EXPORT_HIDDEN", str(persistent_dir))
                    _set_process_env("CK_DEBUG_EXPORT_HIDDEN_NAMES", ",".join(names))
                    if layer >= 0:
                        _set_process_env("CK_DEBUG_EXPORT_HIDDEN_LAYER", str(layer))
                rc = lib.ck_model_decode(ctypes.c_int32(int(token)), logits_buf)
                if rc != 0:
                    raise RuntimeError(f"ck_model_decode failed rc={rc} while capturing persistent hidden state")
                if i == last_i:
                    _restore_process_env({key: None for key in env_keys})
            persistent_kv_export = export_kv(lib, persistent_dir / "kv_cache_f16.bin")
        finally:
            if hasattr(lib, "ck_model_free"):
                try:
                    lib.ck_model_free()
                except Exception:
                    pass

        replay_inputs = dict(inputs)
        replay_inputs["tokens_after"] = [int(t) for t in inputs["tokens_after"]] + generated_prefix
        replay_names = [_hidden_full_replay_name(name) for name in names]
        _set_process_env("CK_DEBUG_EXPORT_HIDDEN", str(full_dir))
        _set_process_env("CK_DEBUG_EXPORT_HIDDEN_NAMES", ",".join(replay_names))
        if layer >= 0:
            _set_process_env("CK_DEBUG_EXPORT_HIDDEN_LAYER", str(layer))
        lib2, _logits2, _vocab2 = _init_ck_state(
            replay_inputs, bool(args.ck_strict_parity), getattr(args, "ck_engine_so", None)
        )
        replay_kv_export = export_kv(lib2, full_dir / "kv_cache_f16.bin")
        if hasattr(lib2, "ck_model_free"):
            try:
                lib2.ck_model_free()
            except Exception:
                pass
        _restore_process_env({key: None for key in env_keys})

    except Exception as exc:
        capture_error = exc
    finally:
        _restore_process_env(saved_env)

    if capture_error is not None:
        for name in names:
            results.append({
                "name": name,
                "full_replay_name": _hidden_full_replay_name(name),
                "status": "error",
                "error": str(capture_error),
                "persistent_dir": str(persistent_dir),
                "full_replay_dir": str(full_dir),
            })
    else:
        if (
            persistent_kv_export is not None
            and replay_kv_export is not None
            and persistent_kv_export.get("status") == "ok"
            and replay_kv_export.get("status") == "ok"
        ):
            persistent_kv = Path(str(persistent_kv_export["path"]))
            replay_kv = Path(str(replay_kv_export["path"]))
            persistent_bytes = persistent_kv.read_bytes()
            replay_bytes = replay_kv.read_bytes()
            first_byte = next(
                (i for i, (a, b) in enumerate(zip(persistent_bytes, replay_bytes)) if a != b),
                None,
            )
            kv_comparison = {
                "status": "ok" if persistent_bytes == replay_bytes else "fail",
                "persistent_path": str(persistent_kv),
                "full_replay_path": str(replay_kv),
                "persistent_bytes": len(persistent_bytes),
                "full_replay_bytes": len(replay_bytes),
                "first_differing_byte": first_byte,
                "first_difference": _describe_kv_f16_difference(
                    persistent_bytes, replay_bytes, first_byte
                ),
                "persistent_sha256": hashlib.sha256(persistent_bytes).hexdigest(),
                "full_replay_sha256": hashlib.sha256(replay_bytes).hexdigest(),
            }
        else:
            statuses = [item for item in (persistent_kv_export, replay_kv_export) if item is not None]
            kv_comparison = {
                "status": "error" if any(item.get("status") == "error" for item in statuses) else "unavailable",
                "persistent": persistent_kv_export,
                "full_replay": replay_kv_export,
            }
        for name in names:
            replay_name = _hidden_full_replay_name(name)
            try:
                if layer < 0:
                    per_layer = _hidden_compare_many(persistent_dir, full_dir, name, replay_name)
                    ok_rows = [r for r in per_layer if r.get("status") == "ok"]
                    max_diff = max(float(r.get("max_abs_diff", 0.0)) for r in ok_rows) if ok_rows else 0.0
                    first_layer_issue = next(
                        (
                            r
                            for r in per_layer
                            if r.get("status") != "ok"
                            or float(r.get("max_abs_diff", 0.0)) > float(args.hidden_state_atol)
                        ),
                        None,
                    )
                    results.append({
                        "name": name,
                        "full_replay_name": replay_name,
                        "status": "ok" if first_layer_issue is None else "fail",
                        "max_abs_diff": float(max_diff),
                        "first_layer_issue": first_layer_issue,
                        "layers": per_layer,
                    })
                else:
                    persistent_file = _single_hidden_file(persistent_dir, name)
                    full_file = _single_hidden_file(full_dir, replay_name)
                    results.append({
                        "name": name,
                        "full_replay_name": replay_name,
                        **_hidden_compare(persistent_file, full_file),
                    })
            except Exception as exc:
                results.append({
                    "name": name,
                    "full_replay_name": replay_name,
                    "status": "error",
                    "error": str(exc),
                    "persistent_dir": str(persistent_dir),
                    "full_replay_dir": str(full_dir),
                })

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
        "preflight": {
            "status": "pass",
            "requested_names": names,
            "available_exporter_count": len(catalog),
        },
        "ck_execution_count": 2,
        "atol": float(args.hidden_state_atol),
        "first_issue": first_issue,
        "kv_cache": kv_comparison,
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

    lib, logits_buf, vocab_size = _init_ck_state(
        replay_inputs, bool(args.ck_strict_parity), getattr(args, "ck_engine_so", None)
    )
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


def _set_ck_strict_parity(lib: Any, strict_parity: bool, engine_so: Path | None = None) -> None:
    value = 1 if strict_parity else 0
    if hasattr(lib, "ck_set_strict_parity"):
        lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
        lib.ck_set_strict_parity.restype = None
        lib.ck_set_strict_parity(value)
        return

    engine_path = engine_so or REPO_ROOT / "build" / "libckernel_engine.so"
    if not engine_path.exists():
        return
    engine = ctypes.CDLL(str(engine_path))
    if hasattr(engine, "ck_set_strict_parity"):
        engine.ck_set_strict_parity.argtypes = [ctypes.c_int]
        engine.ck_set_strict_parity.restype = None
        engine.ck_set_strict_parity(value)


def _init_ck_state(
    inputs: dict[str, Any],
    strict_parity: bool,
    engine_so: Path | None = None,
) -> tuple[Any, Any, int]:
    runtime = inputs["runtime"]
    model_so = Path(runtime["so_path"])
    lib = first_token.bridge_runner_v8._load_decoder_lib(model_so, engine_so=engine_so)
    rc = lib.ck_model_init_with_manifest(
        str(runtime["weights_bump"]).encode(),
        str(runtime["manifest_map"]).encode(),
    )
    if rc != 0:
        raise RuntimeError(f"decoder init failed with rc={rc}")
    _set_ck_strict_parity(lib, strict_parity, engine_so)

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
    mode_contract = getattr(args, "prefill_mode_contract", None)
    if not isinstance(mode_contract, dict):
        mode_contract = _resolve_oracle_prefill_mode(
            str(args.llama_decode_mode),
            inputs["bridge_report"],
            allow_diagnostic_mismatch=True,
        )
    tokenizer: GGUFTokenizer = inputs["tokenizer"]
    llama_seq = _run_llama_greedy_sequence(inputs, args)
    lib, logits_buf, vocab_size = _init_ck_state(
        inputs, bool(args.ck_strict_parity), getattr(args, "ck_engine_so", None)
    )
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

    production_pass = first_divergence is None and bool(mode_contract["compatible"])
    return {
        "status": (
            "pass" if production_pass else
            "diagnostic_only" if first_divergence is None else
            "fail"
        ),
        "pass": production_pass,
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
        "execution_modes": {
            "ck_strict_parity": bool(args.ck_strict_parity),
            "llama_decode_mode": str(args.llama_decode_mode),
            "llama_tensor_repack": not bool(args.llama_no_repack),
            "diagnostic_tensor_dump": bool(
                getattr(args, "dump_step", None) is not None
                or getattr(args, "dump_first_divergence", False)
            ),
            "ck_environment": _ck_environment_evidence(),
            "prefill_mode_contract": mode_contract,
        },
        "llama_greedy_generated": llama_generated,
        "append_on_divergence": str(args.append_on_divergence),
        "ck_runtime": _runtime_evidence(
            inputs["runtime"],
            exact_reuse=bool(inputs.get("exact_runtime", False)),
            engine_so=getattr(args, "ck_engine_so", None),
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


def _resolve_oracle_prefill_mode(
    requested: str,
    bridge_report: dict[str, Any],
    *,
    allow_diagnostic_mismatch: bool = False,
) -> dict[str, Any]:
    """Fail closed when the oracle and CK execute different prefill schedules."""
    bridge = bridge_report.get("bridge_contract")
    bridge = bridge if isinstance(bridge, dict) else {}
    schedule = bridge.get("prefill_schedule")
    schedule = schedule if isinstance(schedule, dict) else {}
    segmented_append = (
        schedule.get("segments") == ["text_before", "visual", "text_after"]
        and schedule.get("cache_transition") == "append_preserve"
    )
    required = "batched" if segmented_append else None
    resolved = "batched" if requested == "auto" and required == "batched" else requested
    compatible = required is None or resolved == required
    if not compatible and not allow_diagnostic_mismatch:
        raise RuntimeError(
            "HARD PARITY CONTRACT FAULT: CK segmented-append prefill requires a batched "
            f"llama.cpp oracle, but {resolved!r} was requested. Use --llama-decode-mode batched "
            "for production parity. Sequential replay is diagnostic-only and requires "
            "--allow-diagnostic-prefill-mode-mismatch."
        )
    return {
        "requested": requested,
        "resolved": resolved,
        "required": required,
        "compatible": compatible,
        "scope": "production" if compatible else "diagnostic_only",
    }


def _run_requested_diagnostics(report: dict[str, Any], args: argparse.Namespace) -> bool:
    """Run bounded diagnostics without discarding the completed coarse report."""
    errors: list[dict[str, str]] = []

    def capture(name: str, callback: Any) -> None:
        try:
            report[name] = callback()
        except Exception as exc:
            errors.append(
                {
                    "diagnostic": name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    try:
        args.dump_step = _resolve_dump_step(
            report,
            args.dump_step,
            bool(args.dump_first_divergence),
        )
    except Exception as exc:
        errors.append(
            {
                "diagnostic": "step_dump",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
    else:
        if args.dump_step is not None:
            capture("step_dump", lambda: _capture_step_dump(report, args))
    if args.full_replay_step is not None:
        capture("full_replay_step", lambda: _capture_full_replay_step(report, args))
    if args.hidden_state_step is not None:
        capture("hidden_state_step", lambda: _capture_hidden_state_step(report, args))

    if errors:
        report["diagnostic_errors"] = errors
    return not errors


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
    ap.add_argument(
        "--ck-engine-so",
        type=Path,
        default=None,
        help="Explicit CK engine shared library; the executed file is hashed into parity evidence",
    )
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
    ap.add_argument("--llama-decode-mode", choices=["auto", "batched", "sequential"], default="auto")
    ap.add_argument(
        "--allow-diagnostic-prefill-mode-mismatch",
        action="store_true",
        help="Allow sequential llama replay against batched CK prefill; the report is diagnostic-only.",
    )
    ap.add_argument("--llama-no-repack", action="store_true", help="Disable llama.cpp CPU tensor repacking in the replay helper")
    ap.add_argument("--llama-repack-fallback", action=argparse.BooleanOptionalAction, default=True, help="Retry a failed llama replay step with --no-repack")
    ap.add_argument("--append-on-divergence", choices=["stop", "llama", "ck"], default="stop")
    ap.add_argument(
        "--ck-strict-parity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable CK's global diagnostic/reference branches. Production parity defaults this off; "
            "strict-mode results must not be reported as production behavior."
        ),
    )
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--dump-step", type=int, default=None, help="Capture CK-vs-llama tensor dumps at this generated step")
    ap.add_argument(
        "--dump-first-divergence",
        action="store_true",
        help="Capture CK-vs-llama tensors at the first observed pre-EOS divergence",
    )
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
    if args.ck_engine_so is not None:
        args.ck_engine_so = args.ck_engine_so.resolve()
        if not args.ck_engine_so.is_file():
            raise FileNotFoundError(f"explicit CK engine shared library does not exist: {args.ck_engine_so}")

    bridge_report = json.loads(args.bridge_report.resolve().read_text(encoding="utf-8"))
    mode_contract = _resolve_oracle_prefill_mode(
        str(args.llama_decode_mode),
        bridge_report,
        allow_diagnostic_mismatch=bool(args.allow_diagnostic_prefill_mode_mismatch),
    )
    args.llama_decode_mode = mode_contract["resolved"]
    args.prefill_mode_contract = mode_contract

    ck_threads = _ck_threads(args)
    if ck_threads > 0:
        os.environ["CK_NUM_THREADS"] = str(ck_threads)
        os.environ["OMP_NUM_THREADS"] = str(ck_threads)
    report = run_multimodal_multitoken_parity(args)
    diagnostics_passed = _run_requested_diagnostics(report, args)
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
    return 0 if report.get("pass") and diagnostics_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
