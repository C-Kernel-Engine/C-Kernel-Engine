#!/usr/bin/env python3
"""Run one rank of the research Qwen3.5 MoE ZIP prefill experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_first_token_logits_v8 import load_ck_logits  # noqa: E402


DEFAULT_TOKEN_PATTERN = [
    1,
    248045,
    846,
    198,
    9419,
    248046,
    198,
    248045,
    74455,
    198,
]


def _tokens(args: argparse.Namespace) -> list[int]:
    if args.tokens_json is not None:
        payload = json.loads(args.tokens_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("tokens")
        if not isinstance(payload, list) or not payload:
            raise ValueError("--tokens-json must contain a non-empty list or {tokens: [...]}")
        tokens = [int(value) for value in payload]
        if args.token_count is not None and len(tokens) != args.token_count:
            raise ValueError(
                f"token file has {len(tokens)} entries, expected {args.token_count}"
            )
        return tokens

    count = int(args.token_count or 512)
    if count <= 1:
        raise ValueError("ZIP prefill requires at least two tokens")
    repeats = (count + len(DEFAULT_TOKEN_PATTERN) - 1) // len(DEFAULT_TOKEN_PATTERN)
    return (DEFAULT_TOKEN_PATTERN * repeats)[:count]


def _top_tokens(logits: np.ndarray, count: int = 16) -> list[dict[str, float | int]]:
    k = min(count, int(logits.size))
    indexes = np.argpartition(-logits, k - 1)[:k]
    indexes = indexes[np.argsort(-logits[indexes])]
    return [
        {"token": int(index), "logit": float(logits[index])}
        for index in indexes.tolist()
    ]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logits-out", type=Path)
    parser.add_argument("--tokens-json", type=Path)
    parser.add_argument("--token-count", type=int, default=512)
    parser.add_argument(
        "--ck-prefill-mode",
        choices=("auto", "batched", "sequential", "hybrid"),
        default="batched",
    )
    args = parser.parse_args()

    tokens = _tokens(args)
    token_bytes = np.asarray(tokens, dtype=np.int32).tobytes()
    started_ns = time.monotonic_ns()
    try:
        result = load_ck_logits(
            args.model_dir.expanduser().resolve(),
            tokens,
            ck_prefill_mode=args.ck_prefill_mode,
        )
        elapsed_ns = time.monotonic_ns() - started_ns
        logits = np.asarray(result["logits"], dtype=np.float32)
        if args.logits_out is not None:
            args.logits_out.parent.mkdir(parents=True, exist_ok=True)
            logits.tofile(args.logits_out)
        payload: dict[str, object] = {
            "schema_version": 1,
            "status": "pass",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "model_dir": str(args.model_dir.expanduser().resolve()),
            "token_count": len(tokens),
            "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "active_tokens": int(result["active_tokens"]),
            "prefill_policy": str(result["prefill_policy"]),
            "wall_ms": elapsed_ns / 1.0e6,
            "tokens_per_second": len(tokens) * 1.0e9 / elapsed_ns,
            "vocab": int(result["vocab"]),
            "finite_logits": bool(np.all(np.isfinite(logits))),
            "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
            "logits_path": str(args.logits_out.resolve()) if args.logits_out else None,
            "top_tokens": _top_tokens(logits),
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "threads": os.environ.get("CK_NUM_THREADS"),
            "zip": {
                "role": os.environ.get("CK_ZIP_RESEARCH_ROLE"),
                "host": os.environ.get("CK_ZIP_RESEARCH_HOST"),
                "port": os.environ.get("CK_ZIP_RESEARCH_PORT"),
                "local_percent": os.environ.get(
                    "CK_ZIP_RESEARCH_LOCAL_PERCENT"
                ),
                "preload": os.environ.get("LD_PRELOAD"),
            },
        }
    except Exception as exc:
        elapsed_ns = time.monotonic_ns() - started_ns
        payload = {
            "schema_version": 1,
            "status": "fail",
            "host": socket.gethostname(),
            "model_dir": str(args.model_dir.expanduser().resolve()),
            "token_count": len(tokens),
            "wall_ms": elapsed_ns / 1.0e6,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(args.output, payload)
        raise

    _write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
