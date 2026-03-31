#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
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
    REPO_ROOT / "version" / "v7" / "scripts" / "parity" / "compare_first_token_logits.py",
)
parity_test_v7 = _load_module(
    "parity_test_v7_decoder_parity",
    REPO_ROOT / "version" / "v7" / "scripts" / "parity_test.py",
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


def _load_prefix_embeddings(prefix_path: Path | None, synthetic_prefix_tokens: int, embed_dim: int) -> tuple[array, int, str]:
    if prefix_path is not None and synthetic_prefix_tokens > 0:
        raise ValueError("use either --prefix-f32 or --synthetic-prefix-tokens, not both")

    if prefix_path is not None:
        blob = prefix_path.read_bytes()
        if len(blob) % 4 != 0:
            raise ValueError(f"prefix file size must be a multiple of 4 bytes: {prefix_path}")
        prefix = array("f")
        prefix.frombytes(blob)
        if embed_dim <= 0 or len(prefix) % embed_dim != 0:
            raise ValueError(
                f"prefix row count does not match decoder embed_dim: floats={len(prefix)} embed_dim={embed_dim}"
            )
        return prefix, len(prefix) // embed_dim, "file"

    if synthetic_prefix_tokens > 0:
        return array("f", [0.0] * (synthetic_prefix_tokens * embed_dim)), synthetic_prefix_tokens, "synthetic_zero"

    return array("f"), 0, "none"


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


def _run_llama_capture(
    gguf_path: Path,
    tokens: list[int],
    ctx_len: int,
    top_k: int,
    threads: int,
    *,
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
            "--decode-mode",
            str(decode_mode),
            "--logits-out",
            str(logits_path),
        ]
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


def _compare_dump_sets(
    ck_dumps: list[Any],
    llama_dumps: list[Any],
    *,
    atol: float,
    rtol: float,
    pass_filter: str,
) -> dict[str, Any]:
    ck_filtered = parity_test_v7._filter_by_pass(list(ck_dumps), pass_filter)
    llama_filtered = parity_test_v7._filter_by_pass(list(llama_dumps), pass_filter)

    ck_by_key: dict[tuple[int, str], list[Any]] = {}
    for dump in ck_filtered:
        ck_by_key.setdefault((int(dump.layer_id), str(dump.op_name)), []).append(dump)
    llama_by_key: dict[tuple[int, str], list[Any]] = {}
    for dump in llama_filtered:
        llama_by_key.setdefault((int(dump.layer_id), str(dump.op_name)), []).append(dump)

    results: list[dict[str, Any]] = []
    all_keys = sorted(set(llama_by_key.keys()) if llama_by_key else set(ck_by_key.keys()))
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
) -> dict[str, Any]:
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    old_dump = os.environ.get("CK_PARITY_DIR")
    os.environ["CK_PARITY_DIR"] = str(dump_dir)
    try:
        return bridge_runner_v8._run_decoder(runtime, prefix_embeddings, prefix_tokens, token_ids)
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
    ctx_len: int,
    top_k: int,
    threads: int,
    dump_root: Path,
    dump_names: str,
    dump_pass: str,
    dump_atol: float,
    dump_rtol: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if prefix_tokens > 0:
        ck = bridge_runner_v8._run_decoder(runtime, prefix_embeddings, prefix_tokens, token_ids)
        return ck, {
            "status": "skipped",
            "reason": "prefix embeddings are not replayed on the llama.cpp reference side yet",
        }

    llama_dump_dir = dump_root / "llama"
    ck_dump_dir = dump_root / "ck"
    llama_capture = _run_llama_capture(
        gguf_path,
        token_ids,
        int(ctx_len),
        int(top_k),
        int(threads),
        decode_mode="sequential",
        dump_dir=llama_dump_dir,
        dump_names=dump_names,
    )
    ck = _capture_ck_dump(runtime, prefix_embeddings, prefix_tokens, token_ids, ck_dump_dir)

    ck_dump_path = ck_dump_dir / "dump.bin"
    ck_dumps = parity_test_v7.read_dump_file(ck_dump_path)
    llama_dumps = _load_llama_dump_dir(llama_dump_dir)
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
        **compare,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v8 decoder first-token parity against local llama.cpp")
    ap.add_argument("--gguf", required=True, type=Path, help="Decoder GGUF to lower/codegen and replay")
    ap.add_argument("--workdir", required=True, type=Path, help="Artifact/output directory")
    ap.add_argument("--prompt", type=str, default=None, help="Prompt text to tokenize through the GGUF tokenizer")
    ap.add_argument("--tokens", type=str, default=None, help="Explicit comma-separated token IDs")
    ap.add_argument("--prefix-f32", type=Path, default=None, help="Optional float32 prefix embeddings for ck_model_forward_mixed")
    ap.add_argument("--synthetic-prefix-tokens", type=int, default=0, help="Use N zero prefix rows instead of --prefix-f32")
    ap.add_argument("--ctx-len", type=int, default=256)
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

    gguf_path = args.gguf.resolve()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    decoder_dir = workdir / "decoder"
    json_out = args.json_out.resolve() if args.json_out is not None else workdir / "decoder_first_token_parity_report.json"

    decoder_runtime = bridge_runner_v8._prepare_decoder_runtime(
        gguf_path,
        decoder_dir,
        parity_dump=args.dump_dir is not None,
    )
    tokenizer = GGUFTokenizer.from_gguf(str(gguf_path))
    resolved_prompt, token_ids = _resolve_prompt_tokens(args.prompt, args.tokens, tokenizer)
    prefix_embeddings, prefix_tokens, prefix_source = _load_prefix_embeddings(
        args.prefix_f32.resolve() if args.prefix_f32 is not None else None,
        int(args.synthetic_prefix_tokens),
        int(decoder_runtime["embed_dim"]),
    )

    ll = compare_first_token_logits_v7.run_llama_logits(
        gguf_path,
        token_ids,
        int(args.ctx_len),
        int(args.top_k),
        int(args.threads),
    )

    dump_report: dict[str, Any] | None = None
    if args.dump_dir is not None:
        dump_root = args.dump_dir.resolve()
        ck, dump_report = _capture_dump_compare(
            gguf_path,
            decoder_runtime,
            prefix_embeddings,
            prefix_tokens,
            token_ids,
            ctx_len=int(args.ctx_len),
            top_k=int(args.top_k),
            threads=int(args.threads),
            dump_root=dump_root,
            dump_names=str(args.dump_names),
            dump_pass=str(args.dump_pass),
            dump_atol=float(args.dump_atol),
            dump_rtol=float(args.dump_rtol),
        )
    else:
        ck = bridge_runner_v8._run_decoder(decoder_runtime, prefix_embeddings, prefix_tokens, token_ids)

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
        "prompt": resolved_prompt,
        "tokens": token_ids,
        "prompt_token_count": len(token_ids),
        "prefix": {
            "source": prefix_source,
            "tokens": int(prefix_tokens),
            "path": str(args.prefix_f32.resolve()) if args.prefix_f32 is not None else None,
        },
        "ctx_len": int(args.ctx_len),
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
            "topk_count": int(len(ll["meta"].get("topk", []) or [])),
            "topk_sample": ll["meta"].get("topk", [])[: max(1, min(8, int(args.top_k)))],
        },
        "compare": cmp,
        "notes": [
            "This is decoder-only parity: identical token IDs into llama.cpp and the generated v8 runtime.",
            "prefix_tokens=0 isolates text-decoder parity before the encoder->decoder bridge is introduced.",
            "If a multimodal prefix file is supplied, the comparison still only validates the decoder seam, not encoder preprocessing parity.",
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
