#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
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
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gguf_tokenizer import GGUFTokenizer  # type: ignore  # noqa: E402


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bridge_runner_v8 = _load_module("run_multimodal_bridge_v8_decoder_parity", SCRIPT_DIR / "run_multimodal_bridge_v8.py")
compare_first_token_logits_v7 = _load_module(
    "compare_first_token_logits_v7_decoder_parity",
    SCRIPT_DIR / "compare_first_token_logits_v8.py",
)
parity_test_v7 = _load_module(
    "parity_test_v7_decoder_parity",
    SCRIPT_DIR / "parity_test_v8.py",
)


def parse_tokens_csv(text: str) -> list[int]:
    tokens: list[int] = []
    for part in str(text or "").split(","):
        item = part.strip()
        if not item:
            continue
        tokens.append(int(item))
    if not tokens:
        raise ValueError("token list is empty")
    return tokens


def _resolve_prompt_tokens(prompt: str | None, tokens_csv: str | None, tokenizer: GGUFTokenizer) -> tuple[str, list[int]]:
    if bool(prompt) == bool(tokens_csv):
        raise ValueError("pass exactly one of --prompt or --tokens")
    if tokens_csv:
        token_ids = parse_tokens_csv(tokens_csv)
        resolved_prompt = tokenizer.decode(token_ids, skip_special=False)
    else:
        resolved_prompt = str(prompt or "")
        token_ids = tokenizer.encode(resolved_prompt)
    return resolved_prompt, token_ids


def _parse_optional_tokens_csv(text: str | None) -> list[int]:
    value = str(text or "").strip()
    return parse_tokens_csv(value) if value else []


def _load_bridge_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"bridge report must be a JSON object: {path}")
    return payload


def _resolve_prompt_token_segments(
    tokenizer: GGUFTokenizer,
    *,
    prompt: str | None,
    tokens_csv: str | None,
    tokens_before_csv: str | None,
    tokens_after_csv: str | None,
    bridge_report: dict[str, Any] | None,
) -> tuple[str, list[int], list[int], dict[str, Any]]:
    if bridge_report is not None:
        before = [int(tok) for tok in list(bridge_report.get("prompt_tokens_before_image") or [])]
        if bridge_report.get("prompt_tokens_after_image") is not None:
            after = [int(tok) for tok in list(bridge_report.get("prompt_tokens_after_image") or [])]
        else:
            after = [int(tok) for tok in list(bridge_report.get("prompt_tokens") or [])]
            before = []
        formatted_prompt = str(bridge_report.get("formatted_prompt") or bridge_report.get("prompt") or "")
        prompt_text = str(bridge_report.get("prompt") or formatted_prompt or tokenizer.decode(before + after, skip_special=False))
        return prompt_text, before, after, {
            "formatted_prompt": formatted_prompt,
            "uses_image_chunks": bool(bridge_report.get("multimodal_prompt_segmented")) or bool(before),
        }

    before = _parse_optional_tokens_csv(tokens_before_csv)
    after = _parse_optional_tokens_csv(tokens_after_csv)
    has_segmented_tokens = bool(before or after)
    if has_segmented_tokens:
        if prompt is not None or tokens_csv is not None:
            raise ValueError("use either --prompt/--tokens or --tokens-before/--tokens-after")
        prompt_text = tokenizer.decode(before + after, skip_special=False)
        return prompt_text, before, after, {
            "formatted_prompt": prompt_text,
            "uses_image_chunks": bool(before),
        }

    prompt_text, token_ids = _resolve_prompt_tokens(prompt, tokens_csv, tokenizer)
    return prompt_text, [], token_ids, {
        "formatted_prompt": prompt_text,
        "uses_image_chunks": False,
    }


def _load_prefix_embeddings(
    prefix_path: Path | None,
    synthetic_prefix_tokens: int,
    embed_dim: int,
    input_embed_dim: int = 0,
    prefix_row_dim: int | None = None,
) -> tuple[array, int, int, str]:
    if prefix_path is not None and synthetic_prefix_tokens > 0:
        raise ValueError("use either --prefix-f32 or --synthetic-prefix-tokens, not both")

    if prefix_path is not None:
        blob = prefix_path.read_bytes()
        if len(blob) % 4 != 0:
            raise ValueError(f"prefix file size must be a multiple of 4 bytes: {prefix_path}")
        prefix = array("f")
        prefix.frombytes(blob)
        row_dim = int(prefix_row_dim or 0)
        if row_dim > 0:
            if len(prefix) % row_dim != 0:
                raise ValueError(
                    f"prefix row count does not match explicit prefix_row_dim: floats={len(prefix)} row_dim={row_dim}"
                )
            return prefix, len(prefix) // row_dim, row_dim, "file"

        if embed_dim > 0 and len(prefix) % embed_dim == 0 and (
            input_embed_dim <= 0 or input_embed_dim == embed_dim or len(prefix) % input_embed_dim != 0
        ):
            return prefix, len(prefix) // embed_dim, embed_dim, "file"
        if input_embed_dim > 0 and len(prefix) % input_embed_dim == 0:
            return prefix, len(prefix) // input_embed_dim, input_embed_dim, "file"
        raise ValueError(
            f"prefix row count does not match decoder dims: floats={len(prefix)} embed_dim={embed_dim} input_embed_dim={input_embed_dim}"
        )

    resolved_row_dim = int(input_embed_dim if input_embed_dim > 0 else embed_dim)
    if synthetic_prefix_tokens > 0:
        if resolved_row_dim <= 0:
            raise ValueError("decoder embed_dim must be positive when synthetic prefix rows are requested")
        return array("f", [0.0] * (synthetic_prefix_tokens * resolved_row_dim)), synthetic_prefix_tokens, resolved_row_dim, "synthetic_zero"

    return array("f"), 0, max(0, resolved_row_dim), "none"


def _decode_topk_tokens(logits: np.ndarray, tokenizer: GGUFTokenizer, top_k: int) -> list[dict[str, Any]]:
    n = int(logits.size)
    k = max(1, min(int(top_k), n))
    top = np.argpartition(-logits, k - 1)[:k]
    top = top[np.argsort(-logits[top])]
    rows: list[dict[str, Any]] = []
    for idx in top.tolist():
        token_id = int(idx)
        rows.append(
            {
                "token_id": token_id,
                "logit": float(logits[token_id]),
                "token_text": tokenizer.decode([token_id], skip_special=False),
            }
        )
    return rows


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def _infer_prefix_grid(prefix_tokens: int) -> tuple[int, int] | None:
    if prefix_tokens <= 0:
        return None
    side = int(math.isqrt(int(prefix_tokens)))
    if side > 0 and side * side == int(prefix_tokens):
        return side, side
    return None


def _resolve_prefix_grid(
    prefix_tokens: int,
    grid_x: int | None = None,
    grid_y: int | None = None,
) -> tuple[int, int] | None:
    resolved_x = int(grid_x or 0)
    resolved_y = int(grid_y or 0)
    if resolved_x <= 0 and resolved_y <= 0:
        return _infer_prefix_grid(prefix_tokens)
    if resolved_x <= 0 or resolved_y <= 0:
        raise ValueError("explicit prefix grid requires both --prefix-grid-x and --prefix-grid-y")
    if resolved_x * resolved_y != int(prefix_tokens):
        raise ValueError(
            f"explicit prefix grid does not match prefix token count: {resolved_x}x{resolved_y} != {prefix_tokens}"
        )
    return resolved_x, resolved_y


def _run_llama_capture(
    gguf_path: Path,
    tokens: list[int],
    ctx_len: int,
    top_k: int,
    threads: int,
    *,
    tokens_before: list[int] | None = None,
    prefix_path: Path | None = None,
    prefix_grid: tuple[int, int] | None = None,
    prefix_row_dim: int | None = None,
    prefix_text_pos: int | None = None,
    decode_mode: str = "batched",
    dump_dir: Path | None = None,
    dump_names: str | None = None,
) -> dict[str, Any]:
    helper = compare_first_token_logits_v7.ensure_llama_helper()
    if dump_dir is not None:
        if dump_dir.exists():
            shutil.rmtree(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="llama_token_replay_v8_") as td:
        logits_path = Path(td) / "llama_logits.f32"
        before_token_ids = [int(tok) for tok in list(tokens_before or [])]
        after_token_ids = [int(tok) for tok in list(tokens)]
        cmd = [
            str(helper),
            "--model",
            str(gguf_path),
            "--ctx",
            str(int(ctx_len)),
            "--top-k",
            str(int(top_k)),
            "--decode-mode",
            str(decode_mode),
            "--logits-out",
            str(logits_path),
        ]
        if before_token_ids:
            cmd.extend(["--tokens-before", ",".join(str(t) for t in before_token_ids)])
            if after_token_ids:
                cmd.extend(["--tokens-after", ",".join(str(t) for t in after_token_ids)])
        elif after_token_ids:
            cmd.extend(["--tokens", ",".join(str(t) for t in after_token_ids)])
        if prefix_path is not None:
            cmd.extend(["--prefix-f32", str(prefix_path)])
        if prefix_grid is not None:
            grid_x, grid_y = prefix_grid
            if int(grid_x) > 0 and int(grid_y) > 0:
                cmd.extend(["--prefix-grid-x", str(int(grid_x)), "--prefix-grid-y", str(int(grid_y))])
        if prefix_row_dim is not None and int(prefix_row_dim) > 0:
            cmd.extend(["--prefix-row-dim", str(int(prefix_row_dim))])
        if prefix_text_pos is not None:
            cmd.extend(["--prefix-text-pos", str(int(prefix_text_pos))])
        if threads > 0:
            cmd.extend(["--threads", str(int(threads))])
        if dump_dir is not None:
            cmd.extend(["--dump-dir", str(dump_dir)])
            if dump_names:
                cmd.extend(["--dump-names", str(dump_names)])

        proc = _run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(
                "llama_token_replay failed\n"
                f"cmd: {' '.join(cmd)}\n"
                f"rc: {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )

        payload = proc.stdout.strip()
        meta = json.loads(payload)
        if not isinstance(meta, dict) or not meta.get("ok"):
            raise RuntimeError(f"llama_token_replay returned invalid payload: {payload}")
        n_vocab = int(meta.get("n_vocab", 0))
        logits = np.fromfile(logits_path, dtype=np.float32)
        if logits.size != n_vocab:
            raise RuntimeError(f"llama logits size mismatch: got={logits.size} expected={n_vocab}")
    return {
        "meta": meta,
        "logits": logits,
    }


def _materialize_llama_prefix(
    prefix_embeddings: array,
    prefix_tokens: int,
    workdir: Path,
    *,
    prefix_path: Path | None,
) -> Path | None:
    if prefix_tokens <= 0:
        return None
    if prefix_path is not None:
        return prefix_path.resolve()
    resolved = (workdir / "resolved_prefix.f32").resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(prefix_embeddings.tobytes())
    return resolved


def _load_llama_dump_dir(dump_dir: Path) -> list[Any]:
    index_path = dump_dir / "index.json"
    if not index_path.exists():
        return []

    dumps: list[Any] = []
    with index_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            elem_count = int(row.get("elem_count", 0) or 0)
            nbytes = int(row.get("nbytes", 0) or 0)
            if elem_count <= 0 or nbytes <= 0:
                continue

            bin_path = dump_dir / f"{row['name']}.bin"
            if not bin_path.exists():
                continue

            elem_size = nbytes // elem_count if nbytes % elem_count == 0 else 4
            if elem_size == 4:
                raw = np.fromfile(bin_path, dtype=np.float32)
                dtype_name = "fp32"
            elif elem_size == 2:
                raw = np.fromfile(bin_path, dtype=np.float16).astype(np.float32)
                dtype_name = "fp16"
            else:
                raw = np.fromfile(bin_path, dtype=np.uint8).astype(np.float32)
                dtype_name = f"raw{elem_size}"

            rank = max(1, int(row.get("rank", 1) or 1))
            raw_shape = row.get("shape", [])
            shape = [int(x) for x in list(raw_shape)[:rank] if int(x) > 0]
            data = raw.astype(np.float32, copy=False)
            if shape:
                expected = int(np.prod(np.array(shape, dtype=np.int64)))
                if expected == int(data.size):
                    data = data.reshape(shape)

            norm_layer, norm_op = parity_test_v7._normalize_layer_and_op(
                -1,
                str(row.get("base_name", row.get("name", ""))),
            )
            dumps.append(
                parity_test_v7.ParityDump(
                    norm_layer,
                    norm_op,
                    data,
                    int(row.get("token_id", 0) or 0),
                    dtype_name,
                )
            )
    return dumps


def _summarize_statuses(results: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "total": len(results),
        "pass": 0,
        "fail": 0,
        "error": 0,
        "warn": 0,
        "missing": 0,
    }
    for row in results:
        status = str(row.get("status", "")).upper()
        if status == "PASS":
            summary["pass"] += 1
        elif status == "FAIL":
            summary["fail"] += 1
        elif status == "ERROR":
            summary["error"] += 1
        elif status == "WARN":
            summary["warn"] += 1
        elif status == "MISSING":
            summary["missing"] += 1
    return summary


def _canonical_dump_op_name(op_name: str) -> str:
    return str(op_name)


def _augment_legacy_kqv_aliases(dumps: list[Any]) -> list[Any]:
    """
    Preserve modern native ``kqv_out`` dumps, but backfill a compatibility
    alias for older captures that only exposed attention output under
    ``attn_output``.
    """
    out = list(dumps)
    seen = {
        (int(d.layer_id), str(d.op_name), int(getattr(d, "token_id", 0)))
        for d in out
    }
    for dump in dumps:
        if str(dump.op_name) != "attn_output":
            continue
        alias_key = (int(dump.layer_id), "kqv_out", int(getattr(dump, "token_id", 0)))
        if alias_key in seen:
            continue
        out.append(
            parity_test_v7.ParityDump(
                int(dump.layer_id),
                "kqv_out",
                np.array(dump.data, copy=True),
                int(getattr(dump, "token_id", 0)),
                str(dump.dtype),
            )
        )
        seen.add(alias_key)
    rope_aliases = {
        "Qcur_rope": "Qcur",
        "Kcur_rope": "Kcur",
    }
    for dump in dumps:
        alias_name = rope_aliases.get(str(dump.op_name))
        if not alias_name:
            continue
        alias_key = (int(dump.layer_id), alias_name, int(getattr(dump, "token_id", 0)))
        if alias_key in seen:
            continue
        out.append(
            parity_test_v7.ParityDump(
                int(dump.layer_id),
                alias_name,
                np.array(dump.data, copy=True),
                int(getattr(dump, "token_id", 0)),
                str(dump.dtype),
            )
        )
        seen.add(alias_key)
    return out


def _build_llama_row_specs(llama_dumps: list[Any]) -> dict[tuple[int, str], tuple[int, tuple[int, ...]]]:
    specs: dict[tuple[int, str], tuple[int, tuple[int, ...]]] = {}
    for dump in llama_dumps:
        op_name = _canonical_dump_op_name(str(dump.op_name))
        key = (int(dump.layer_id), op_name)
        row_elems = int(np.asarray(dump.data).size)
        row_shape = tuple(int(x) for x in np.asarray(dump.data).shape)
        prev = specs.get(key)
        choose = prev is None or row_elems < prev[0]
        if not choose and prev is not None and row_elems == prev[0]:
            if op_name in {"qcur_normed", "kcur_normed"} and len(row_shape) > len(prev[1]):
                choose = True
        if choose:
            specs[key] = (row_elems, row_shape)
    return specs


def _resolve_decode_prompt_start_tokens(
    *,
    tokens_before_count: int,
    prefix_tokens: int,
    prefix_text_pos: int | None,
    llama_meta: dict[str, Any] | None = None,
) -> tuple[int, int]:
    ck_prompt_start = max(0, int(tokens_before_count) + int(prefix_tokens))
    if llama_meta is not None and llama_meta.get("prefix_text_pos") is not None:
        llama_prompt_start = int(llama_meta.get("prefix_text_pos") or 0)
    elif prefix_tokens > 0 and prefix_text_pos is not None:
        llama_prompt_start = int(prefix_text_pos)
    else:
        llama_prompt_start = int(tokens_before_count)
    return ck_prompt_start, max(0, llama_prompt_start)


def _build_multimodal_position_contract(
    *,
    tokens_before_count: int,
    prefix_tokens: int,
    prefix_grid: tuple[int, int] | None,
    prefix_text_pos: int | None,
    llama_meta: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if prefix_tokens <= 0 or prefix_grid is None:
        return None

    grid_x, grid_y = int(prefix_grid[0]), int(prefix_grid[1])
    if grid_x <= 0 or grid_y <= 0:
        return None
    if grid_x * grid_y != int(prefix_tokens):
        return None

    ck_start = int(tokens_before_count)
    llama_start = int(llama_meta.get("prefix_start_pos", ck_start) if llama_meta is not None else ck_start)
    ck_text = int(prefix_text_pos if prefix_text_pos is not None else (ck_start + max(grid_x, grid_y)))
    llama_text = int(llama_meta.get("prefix_text_pos", ck_text) if llama_meta is not None else ck_text)

    def _rows(start_pos: int) -> list[list[int]]:
        rows: list[list[int]] = []
        for idx in range(int(prefix_tokens)):
            x = idx % grid_x
            y = idx // grid_x
            rows.append([int(start_pos), int(start_pos + y), int(start_pos + x), 0])
        return rows

    ck_rows = _rows(ck_start)
    llama_rows = _rows(llama_start)
    return {
        "grid": [grid_x, grid_y],
        "prefix_tokens": int(prefix_tokens),
        "ck": {
            "prefix_start_pos": ck_start,
            "prefix_text_pos": ck_text,
            "rows": ck_rows,
        },
        "llama": {
            "prefix_start_pos": llama_start,
            "prefix_text_pos": llama_text,
            "rows": llama_rows,
        },
        "rows_match": ck_rows == llama_rows,
        "text_pos_match": ck_text == llama_text,
    }


def _expand_ck_prefill_decode_dumps(
    ck_dumps: list[Any],
    llama_dumps: list[Any],
    *,
    prompt_start_token: int,
    prompt_token_count: int,
) -> list[Any]:
    if prompt_token_count <= 0:
        return list(ck_dumps)

    prompt_start = max(0, int(prompt_start_token))
    rebased: list[Any] = []
    for dump in ck_dumps:
        token_id = int(getattr(dump, "token_id", 0))
        if token_id < prompt_start:
            continue
        if token_id >= prompt_start + prompt_token_count:
            continue
        rebased.append(
            parity_test_v7.ParityDump(
                int(dump.layer_id),
                str(dump.op_name),
                np.array(dump.data, copy=True),
                token_id - prompt_start,
                str(dump.dtype),
            )
        )
    if len({int(getattr(dump, "token_id", 0)) for dump in rebased}) >= prompt_token_count:
        return rebased

    row_specs = _build_llama_row_specs(llama_dumps)
    if not row_specs:
        return list(ck_dumps)

    expanded: list[Any] = []
    for dump in ck_dumps:
        op_name = _canonical_dump_op_name(str(dump.op_name))
        key = (int(dump.layer_id), op_name)
        row_spec = row_specs.get(key)
        if row_spec is None:
            expanded.append(dump)
            continue

        row_elems, row_shape = row_spec
        flat = np.asarray(dump.data, dtype=np.float32).reshape(-1)
        if row_elems <= 0 or flat.size <= row_elems or flat.size % row_elems != 0:
            expanded.append(dump)
            continue

        batch_rows = flat.size // row_elems
        if batch_rows < prompt_token_count:
            expanded.append(dump)
            continue

        prompt_start = batch_rows - prompt_token_count
        head_major = (
            len(row_shape) == 2
            and row_shape[0] > 0
            and row_shape[1] > 0
            and flat.size % (int(row_shape[0]) * int(row_shape[1])) == 0
        )
        if head_major:
            dim = int(row_shape[0])
            heads = int(row_shape[1])
            token_major_stride = dim * heads
            batch_rows = flat.size // token_major_stride
            if batch_rows < prompt_token_count:
                expanded.append(dump)
                continue
            tensor = flat.reshape(heads, batch_rows, dim)
        for prompt_idx in range(prompt_token_count):
            if head_major:
                token_idx = (batch_rows - prompt_token_count) + prompt_idx
                # Preserve CK's native head-major flat order here.
                # compare_dumps() flattens both sides, and llama.cpp's dump
                # index shape metadata does not imply NumPy/C-order transpose.
                row = tensor[:, token_idx, :].copy()
            else:
                start = (prompt_start + prompt_idx) * row_elems
                end = start + row_elems
                row = flat[start:end].copy()
                if row_shape and int(np.prod(np.array(row_shape, dtype=np.int64))) == row.size:
                    row = row.reshape(row_shape)
            expanded.append(
                parity_test_v7.ParityDump(
                    int(dump.layer_id),
                    op_name,
                    row,
                    prompt_idx,
                    str(dump.dtype),
                )
            )
    return expanded


def _trim_llama_prefill_decode_dumps(
    llama_dumps: list[Any],
    *,
    prompt_start_token: int,
    prompt_token_count: int,
) -> list[Any]:
    if prompt_token_count <= 0:
        return list(llama_dumps)

    prompt_start = max(0, int(prompt_start_token))
    grouped: dict[tuple[int, str], list[Any]] = {}
    for dump in llama_dumps:
        grouped.setdefault((int(dump.layer_id), str(dump.op_name)), []).append(dump)

    trimmed: list[Any] = []
    for (_, _), group in grouped.items():
        abs_tokens = sorted(
            {
                int(getattr(dump, "token_id", 0))
                for dump in group
                if prompt_start <= int(getattr(dump, "token_id", 0)) < prompt_start + prompt_token_count
            }
        )
        if len(abs_tokens) == prompt_token_count:
            selected_tokens = set(abs_tokens)
            token_rebase = {token_id: token_id - prompt_start for token_id in abs_tokens}
        else:
            tail_tokens = sorted({int(getattr(dump, "token_id", 0)) for dump in group})[-prompt_token_count:]
            if not tail_tokens:
                continue
            selected_tokens = set(tail_tokens)
            token_rebase = {token_id: idx for idx, token_id in enumerate(tail_tokens)}

        for dump in group:
            token_id = int(getattr(dump, "token_id", 0))
            if token_id not in selected_tokens:
                continue
            trimmed.append(
                parity_test_v7.ParityDump(
                    int(dump.layer_id),
                    str(dump.op_name),
                    np.array(dump.data, copy=True),
                    int(token_rebase[token_id]),
                    str(dump.dtype),
                )
            )
    return trimmed

def _compare_dump_sets(
    ck_dumps: list[Any],
    llama_dumps: list[Any],
    *,
    atol: float,
    rtol: float,
    pass_filter: str,
) -> dict[str, Any]:
    ck_filtered = _augment_legacy_kqv_aliases(
        parity_test_v7._filter_by_pass(list(ck_dumps), pass_filter)
    )
    llama_filtered = _augment_legacy_kqv_aliases(
        parity_test_v7._filter_by_pass(list(llama_dumps), pass_filter)
    )

    ck_by_key: dict[tuple[int, str], list[Any]] = {}
    for dump in ck_filtered:
        key = (int(dump.layer_id), _canonical_dump_op_name(str(dump.op_name)))
        ck_by_key.setdefault(key, []).append(dump)
    llama_by_key: dict[tuple[int, str], list[Any]] = {}
    for dump in llama_filtered:
        key = (int(dump.layer_id), _canonical_dump_op_name(str(dump.op_name)))
        llama_by_key.setdefault(key, []).append(dump)

    results: list[dict[str, Any]] = []
    all_keys = sorted(set(ck_by_key.keys()) | set(llama_by_key.keys()))
    for layer_id, op_name in all_keys:
        ck_candidates = ck_by_key.get((layer_id, op_name), [])
        llama_candidates = llama_by_key.get((layer_id, op_name), [])
        ck_dump, llama_dump, precomputed, ambiguous = parity_test_v7._pick_best_alignment(
            ck_candidates,
            llama_candidates,
            float(atol),
            float(rtol),
        )

        if ck_dump is None:
            results.append(
                {
                    "layer": int(layer_id),
                    "op": str(op_name),
                    "status": "MISSING",
                    "max_abs_diff": float("inf"),
                    "ck_missing": True,
                    "ck_candidates": 0,
                    "llama_candidates": len(llama_candidates),
                    "alignment_ambiguous": False,
                }
            )
            continue

        if llama_dump is None:
            results.append(
                {
                    "layer": int(layer_id),
                    "op": str(op_name),
                    "status": "WARN",
                    "max_abs_diff": 0.0,
                    "llama_missing": True,
                    "token": int(ck_dump.token_id),
                    "ck_token": int(ck_dump.token_id),
                    "llama_token": None,
                    "ck_candidates": len(ck_candidates),
                    "llama_candidates": 0,
                    "alignment_ambiguous": bool(ambiguous),
                }
            )
            continue

        comp = precomputed if precomputed is not None else parity_test_v7.compare_dumps(
            llama_dump,
            ck_dump,
            float(atol),
            float(rtol),
        )
        results.append(
            {
                "layer": int(layer_id),
                "op": str(op_name),
                **comp,
                "token": int(ck_dump.token_id),
                "ck_token": int(ck_dump.token_id),
                "llama_token": int(llama_dump.token_id),
                "ck_candidates": len(ck_candidates),
                "llama_candidates": len(llama_candidates),
                "alignment_ambiguous": bool(ambiguous),
            }
        )

    summary = _summarize_statuses(results)
    first_issue = next((row for row in results if str(row.get("status", "")).upper() in {"ERROR", "FAIL"}), None)
    return {
        "summary": summary,
        "first_issue": first_issue,
        "results": results,
    }


def _capture_ck_dump(
    runtime: dict[str, Any],
    prefix_embeddings: array,
    prefix_tokens: int,
    token_ids: list[int],
    dump_dir: Path,
    prefix_row_dim: int,
    *,
    tokens_before: list[int] | None = None,
    prefix_grid: tuple[int, int] | None = None,
    prefix_text_pos: int | None = None,
) -> dict[str, Any]:
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    old_dump = os.environ.get("CK_PARITY_DIR")
    os.environ["CK_PARITY_DIR"] = str(dump_dir)
    try:
        return bridge_runner_v8._run_decoder(
            runtime,
            prefix_embeddings,
            prefix_tokens,
            token_ids,
            tokens_before=tokens_before,
            prefix_embed_dim=prefix_row_dim,
            prefix_grid=prefix_grid,
            prefix_text_pos=prefix_text_pos,
            strict_parity=True,
        )
    finally:
        if old_dump is None:
            os.environ.pop("CK_PARITY_DIR", None)
        else:
            os.environ["CK_PARITY_DIR"] = old_dump


def _capture_dump_compare(
    gguf_path: Path,
    runtime: dict[str, Any],
    prefix_embeddings: array,
    prefix_tokens: int,
    token_ids: list[int],
    *,
    tokens_before: list[int] | None = None,
    prefix_row_dim: int,
    ctx_len: int,
    top_k: int,
    threads: int,
    dump_root: Path,
    dump_names: str,
    dump_pass: str,
    dump_atol: float,
    dump_rtol: float,
    prefix_grid: tuple[int, int] | None = None,
    prefix_text_pos: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if prefix_tokens > 0:
        if str(dump_pass) != "decode":
            ck = bridge_runner_v8._run_decoder(
                runtime,
                prefix_embeddings,
                prefix_tokens,
                token_ids,
                tokens_before=tokens_before,
                prefix_embed_dim=prefix_row_dim,
                strict_parity=True,
            )
            return ck, {
                "status": "skipped",
                "reason": "multimodal dump alignment is only implemented for decode-mode comparisons; use --dump-pass decode",
            }
        prefix_path = dump_root / "prefix.f32"
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        prefix_path.write_bytes(prefix_embeddings.tobytes())
    else:
        prefix_path = None

    llama_dump_dir = dump_root / "llama"
    ck_dump_dir = dump_root / "ck"
    llama_capture = _run_llama_capture(
        gguf_path,
        token_ids,
        int(ctx_len),
        int(top_k),
        int(threads),
        tokens_before=tokens_before,
        prefix_path=prefix_path,
        prefix_grid=prefix_grid,
        prefix_row_dim=int(prefix_row_dim),
        prefix_text_pos=prefix_text_pos,
        decode_mode="sequential",
        dump_dir=llama_dump_dir,
        dump_names=dump_names,
    )
    ck = _capture_ck_dump(
        runtime,
        prefix_embeddings,
        prefix_tokens,
        token_ids,
        ck_dump_dir,
        int(prefix_row_dim),
        tokens_before=tokens_before,
        prefix_grid=prefix_grid,
        prefix_text_pos=prefix_text_pos,
    )

    ck_dump_path = ck_dump_dir / "dump.bin"
    ck_dumps = parity_test_v7.read_dump_file(ck_dump_path)
    llama_dumps = _load_llama_dump_dir(llama_dump_dir)
    tokens_before_count = len(list(tokens_before or []))
    ck_prompt_start_token = None
    llama_prompt_start_token = None
    if str(dump_pass) == "decode":
        ck_prompt_start_token, llama_prompt_start_token = _resolve_decode_prompt_start_tokens(
            tokens_before_count=tokens_before_count,
            prefix_tokens=int(prefix_tokens),
            prefix_text_pos=prefix_text_pos,
            llama_meta=dict(llama_capture.get("meta") or {}),
        )
        llama_dumps = _trim_llama_prefill_decode_dumps(
            llama_dumps,
            prompt_start_token=int(llama_prompt_start_token),
            prompt_token_count=len(token_ids),
        )
        ck_dumps = _expand_ck_prefill_decode_dumps(
            ck_dumps,
            llama_dumps,
            prompt_start_token=int(ck_prompt_start_token),
            prompt_token_count=len(token_ids),
        )
    compare = _compare_dump_sets(
        ck_dumps,
        llama_dumps,
        atol=float(dump_atol),
        rtol=float(dump_rtol),
        pass_filter=str(dump_pass),
    )

    status = "ok"
    if not ck_dumps:
        status = "error"
    elif compare["summary"]["error"] > 0 or compare["summary"]["fail"] > 0:
        status = "fail"

    return ck, {
        "status": status,
        "dump_names": [item.strip() for item in str(dump_names).split(",") if item.strip()],
        "pass_filter": str(dump_pass),
        "atol": float(dump_atol),
        "rtol": float(dump_rtol),
        "ck_dump_path": str(ck_dump_path),
        "llama_dump_dir": str(llama_dump_dir),
        "llama_decode_mode": str(llama_capture["meta"].get("decode_mode", "sequential")),
        "llama_dumped": int(llama_capture["meta"].get("dumped", 0)),
        "prefix_path": None if prefix_path is None else str(prefix_path),
        "tokens_before_count": int(tokens_before_count),
        "ck_prompt_start_token": None if ck_prompt_start_token is None else int(ck_prompt_start_token),
        "llama_prompt_start_token": None if llama_prompt_start_token is None else int(llama_prompt_start_token),
        **compare,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v8 decoder first-token parity against local llama.cpp")
    ap.add_argument("--gguf", required=False, type=Path, help="Decoder GGUF to lower/codegen and replay")
    ap.add_argument("--bridge-report", type=Path, default=None, help="Optional bridge_report.json from run_multimodal_bridge_v8; replays the exact segmented multimodal prompt")
    ap.add_argument("--workdir", required=True, type=Path, help="Artifact/output directory")
    ap.add_argument("--prompt", type=str, default=None, help="Prompt text to tokenize through the GGUF tokenizer")
    ap.add_argument("--tokens", type=str, default=None, help="Explicit comma-separated token IDs")
    ap.add_argument("--tokens-before", type=str, default=None, help="Optional comma-separated token IDs that must appear before the multimodal prefix")
    ap.add_argument("--tokens-after", type=str, default=None, help="Optional comma-separated token IDs that must appear after the multimodal prefix")
    ap.add_argument("--prefix-f32", type=Path, default=None, help="Optional float32 prefix embeddings for ck_model_forward_mixed")
    ap.add_argument("--prefix-row-dim", type=int, default=None, help="Optional explicit row width for --prefix-f32 (for example qwen3vl n_embd_inp)")
    ap.add_argument("--prefix-grid-x", type=int, default=None, help="Optional explicit multimodal prefix grid width")
    ap.add_argument("--prefix-grid-y", type=int, default=None, help="Optional explicit multimodal prefix grid height")
    ap.add_argument("--prefix-text-pos", type=int, default=None, help="Optional explicit text rope position after the prefix grid")
    ap.add_argument("--synthetic-prefix-tokens", type=int, default=0, help="Use N zero prefix rows instead of --prefix-f32")
    ap.add_argument("--ctx-len", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--require-top1-match", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-topk-overlap", type=float, default=0.50)
    ap.add_argument("--max-abs-threshold", type=float, default=1.0e9)
    ap.add_argument("--dump-dir", type=Path, default=None, help="Optional directory to capture CK and llama decoder dumps")
    ap.add_argument(
        "--dump-names",
        type=str,
        default="Qcur-0,Kcur-0,Vcur-0,Qcur_normed-0,Kcur_normed-0,kqv_out-0",
        help="Comma-separated llama dump names for sequential decoder dump capture",
    )
    ap.add_argument("--dump-pass", choices=("all", "prefill", "decode"), default="decode")
    ap.add_argument("--dump-atol", type=float, default=1.0e-4)
    ap.add_argument("--dump-rtol", type=float, default=1.0e-3)
    ap.add_argument("--json-out", type=Path, default=None, help="Optional explicit JSON report path")
    args = ap.parse_args(argv)

    bridge_report = _load_bridge_report(args.bridge_report.resolve()) if args.bridge_report is not None else None
    if args.gguf is not None:
        gguf_path = args.gguf.resolve()
    elif bridge_report is not None:
        gguf_value = str(((bridge_report.get("decoder_runtime") or {}).get("gguf") or "")).strip()
        if not gguf_value:
            raise ValueError("--gguf is required when bridge report does not include decoder_runtime.gguf")
        gguf_path = Path(gguf_value).resolve()
    else:
        raise ValueError("pass --gguf or --bridge-report")
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    decoder_dir = workdir / "decoder"
    json_out = args.json_out.resolve() if args.json_out is not None else workdir / "decoder_first_token_parity_report.json"

    requested_ctx_source = args.ctx_len
    if requested_ctx_source is None and bridge_report is not None and bridge_report.get("decoder_context_len") is not None:
        requested_ctx_source = int(bridge_report.get("decoder_context_len") or 0)
    requested_ctx_len = max(1, int(requested_ctx_source if requested_ctx_source is not None else 256))
    decoder_runtime = bridge_runner_v8._prepare_decoder_runtime(
        gguf_path,
        decoder_dir,
        parity_dump=args.dump_dir is not None,
        context_override=requested_ctx_len,
    )
    tokenizer = GGUFTokenizer.from_gguf(str(gguf_path))
    resolved_prompt, token_ids_before, token_ids_after, prompt_meta = _resolve_prompt_token_segments(
        tokenizer,
        prompt=args.prompt,
        tokens_csv=args.tokens,
        tokens_before_csv=args.tokens_before,
        tokens_after_csv=args.tokens_after,
        bridge_report=bridge_report,
    )
    resolved_prefix_path = args.prefix_f32.resolve() if args.prefix_f32 is not None else None
    if resolved_prefix_path is None and bridge_report is not None:
        bridge_prefix_dump = str(bridge_report.get("prefix_dump_path") or "").strip()
        if bridge_prefix_dump:
            resolved_prefix_path = Path(bridge_prefix_dump).resolve()
    prefix_embeddings, prefix_tokens, prefix_row_dim, prefix_source = _load_prefix_embeddings(
        resolved_prefix_path,
        int(args.synthetic_prefix_tokens),
        int(decoder_runtime["embed_dim"]),
        int(decoder_runtime.get("input_embed_dim", 0) or 0),
        int(args.prefix_row_dim) if args.prefix_row_dim is not None else None,
    )
    total_prompt_tokens = len(token_ids_before) + len(token_ids_after)
    resolved_ctx_len = max(requested_ctx_len, int(prefix_tokens) + total_prompt_tokens, 1)
    if resolved_ctx_len != requested_ctx_len:
        decoder_runtime = bridge_runner_v8._prepare_decoder_runtime(
            gguf_path,
            decoder_dir,
            parity_dump=args.dump_dir is not None,
            context_override=resolved_ctx_len,
        )
    llama_prefix_path = _materialize_llama_prefix(
        prefix_embeddings,
        prefix_tokens,
        workdir,
        prefix_path=resolved_prefix_path,
    )

    bridge_grid_x = None if bridge_report is None or bridge_report.get("prefix_grid_x") is None else int(bridge_report.get("prefix_grid_x"))
    bridge_grid_y = None if bridge_report is None or bridge_report.get("prefix_grid_y") is None else int(bridge_report.get("prefix_grid_y"))
    resolved_prefix_grid = _resolve_prefix_grid(
        prefix_tokens,
        int(args.prefix_grid_x) if args.prefix_grid_x is not None else bridge_grid_x,
        int(args.prefix_grid_y) if args.prefix_grid_y is not None else bridge_grid_y,
    )
    resolved_prefix_text_pos = (
        int(args.prefix_text_pos)
        if args.prefix_text_pos is not None
        else (
            int(bridge_report.get("prefix_text_pos"))
            if bridge_report is not None and bridge_report.get("prefix_text_pos") is not None
            else (
                len(token_ids_before) + max(int(resolved_prefix_grid[0]), int(resolved_prefix_grid[1]))
                if resolved_prefix_grid is not None
                else len(token_ids_before) + int(prefix_tokens)
            )
        )
    )
    ll = _run_llama_capture(
        gguf_path,
        token_ids_after,
        int(resolved_ctx_len),
        int(args.top_k),
        int(args.threads),
        tokens_before=token_ids_before,
        prefix_path=llama_prefix_path,
        prefix_grid=resolved_prefix_grid,
        prefix_row_dim=int(prefix_row_dim),
        prefix_text_pos=resolved_prefix_text_pos,
    )
    position_contract = _build_multimodal_position_contract(
        tokens_before_count=len(token_ids_before),
        prefix_tokens=int(prefix_tokens),
        prefix_grid=resolved_prefix_grid,
        prefix_text_pos=resolved_prefix_text_pos,
        llama_meta=dict(ll.get("meta") or {}),
    )

    dump_report: dict[str, Any] | None = None
    if args.dump_dir is not None:
        dump_root = args.dump_dir.resolve()
        ck, dump_report = _capture_dump_compare(
            gguf_path,
            decoder_runtime,
            prefix_embeddings,
            prefix_tokens,
            token_ids_after,
            tokens_before=token_ids_before,
            prefix_row_dim=int(prefix_row_dim),
            ctx_len=int(resolved_ctx_len),
            top_k=int(args.top_k),
            threads=int(args.threads),
            dump_root=dump_root,
            dump_names=str(args.dump_names),
            dump_pass=str(args.dump_pass),
            dump_atol=float(args.dump_atol),
            dump_rtol=float(args.dump_rtol),
            prefix_grid=resolved_prefix_grid,
            prefix_text_pos=resolved_prefix_text_pos,
        )
    else:
        ck = bridge_runner_v8._run_decoder(
            decoder_runtime,
            prefix_embeddings,
            prefix_tokens,
            token_ids_after,
            tokens_before=token_ids_before,
            prefix_embed_dim=prefix_row_dim,
            prefix_grid=resolved_prefix_grid,
            prefix_text_pos=resolved_prefix_text_pos,
            strict_parity=True,
        )

    ck_logits = np.array(ck["logits"], dtype=np.float32, copy=False)
    cmp = compare_first_token_logits_v7.compare_logits(ck_logits, ll["logits"], int(args.top_k))

    overlap_ok = cmp["topk_overlap_ratio"] >= float(args.min_topk_overlap)
    top1_ok = (not bool(args.require_top1_match)) or bool(cmp["top1_match"])
    max_abs_ok = cmp["max_abs_diff"] <= float(args.max_abs_threshold)
    passed = bool(top1_ok and overlap_ok and max_abs_ok)

    report = {
        "status": "pass" if passed else "fail",
        "pass": passed,
        "gguf_path": str(gguf_path),
        "workdir": str(workdir),
        "decoder_runtime": {
            "embed_dim": int(decoder_runtime["embed_dim"]),
            "vocab_size": int(decoder_runtime["vocab_size"]),
            "so_path": str(decoder_runtime["so_path"]),
            "c_path": str(decoder_runtime["c_path"]),
        },
        "bridge_report_path": None if args.bridge_report is None else str(args.bridge_report.resolve()),
        "prompt": resolved_prompt,
        "formatted_prompt": str(prompt_meta.get("formatted_prompt") or resolved_prompt),
        "tokens": [int(tok) for tok in (token_ids_before + token_ids_after)],
        "prompt_token_count": total_prompt_tokens,
        "prompt_tokens_before_image": [int(tok) for tok in token_ids_before],
        "prompt_tokens_after_image": [int(tok) for tok in token_ids_after],
        "multimodal_prompt_segmented": bool(prompt_meta.get("uses_image_chunks")),
        "prefix": {
            "source": prefix_source,
            "tokens": int(prefix_tokens),
            "row_dim": int(prefix_row_dim),
            "path": None if resolved_prefix_path is None else str(resolved_prefix_path),
            "grid": None if resolved_prefix_grid is None else [int(resolved_prefix_grid[0]), int(resolved_prefix_grid[1])],
            "text_pos": int(resolved_prefix_text_pos),
            "llama_position_count": int(ll["meta"].get("prefix_position_count", prefix_tokens)),
            "llama_start_pos": int(ll["meta"].get("prefix_start_pos", len(token_ids_before))),
        },
        "position_contract": position_contract,
        "ctx_len": int(resolved_ctx_len),
        "requested_ctx_len": int(requested_ctx_len),
        "thresholds": {
            "require_top1_match": bool(args.require_top1_match),
            "min_topk_overlap": float(args.min_topk_overlap),
            "max_abs_threshold": float(args.max_abs_threshold),
        },
        "ck": {
            "vocab": int(ck["vocab_size"]),
            "topk_sample": _decode_topk_tokens(ck_logits, tokenizer, int(args.top_k)),
        },
        "llama": {
            "n_vocab": int(ll["meta"]["n_vocab"]),
            "token_count": int(ll["meta"]["token_count"]),
            "token_count_before": int(ll["meta"].get("token_count_before", len(token_ids_before))),
            "token_count_after": int(ll["meta"].get("token_count_after", len(token_ids_after))),
            "topk_count": int(len(ll["meta"].get("topk", []) or [])),
            "topk_sample": ll["meta"].get("topk", [])[: max(1, min(8, int(args.top_k)))],
        },
        "compare": cmp,
        "notes": [
            "This is decoder-only parity: identical token IDs into llama.cpp and the generated v8 runtime.",
            "prefix_tokens=0 isolates text-decoder parity before the encoder->decoder bridge is introduced.",
            "When a bridge report is provided, the parity tool replays the exact segmented multimodal prompt: text-before + image prefix + text-after.",
            "That validates decoder first-token logits under the real multimodal bridge contract; encoder/preprocessing parity still must be checked separately.",
        ],
    }
    if dump_report is not None:
        report["dump_compare"] = dump_report

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
