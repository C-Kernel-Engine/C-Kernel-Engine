#!/usr/bin/env python3
"""Certify Cohere decoder token parity through a generated model library."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


def _load_provenance():
    path = Path(__file__).with_name("certify_parakeet_generated_frontend_v8.py")
    spec = importlib.util.spec_from_file_location(
        "cohere_generated_decoder_provenance", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


provenance = _load_provenance()


F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int32)
ENCODER_CHECKPOINT = "audio.encoder.projected.output"


def _encoder_fixture(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as fixture:
        if ENCODER_CHECKPOINT not in fixture.files:
            raise ValueError(f"encoder fixture is missing {ENCODER_CHECKPOINT}")
        value = np.ascontiguousarray(fixture[ENCODER_CHECKPOINT])
    if value.dtype != np.float32 or value.ndim != 2 or not value.size:
        raise ValueError("encoder fixture must be a non-empty rank-2 float32 tensor")
    if not np.isfinite(value).all():
        raise ValueError("encoder fixture contains non-finite values")
    return value


def _expected_tokens(path: Path, eos_token_id: int | None) -> list[int]:
    report = json.loads(path.read_text(encoding="utf-8"))
    decode = report.get("decode") if isinstance(report.get("decode"), dict) else {}
    values = decode.get("emitted_token_ids") or report.get("emitted_token_ids")
    if values is None:
        values = report.get("generated_token_ids")
        if values is not None and eos_token_id is not None:
            values = [*values, eos_token_id]
    if not isinstance(values, list) or not values:
        raise ValueError("reference report has no emitted token trajectory")
    tokens = [int(value) for value in values]
    if any(token < 0 for token in tokens):
        raise ValueError("reference token trajectory contains a negative token")
    return tokens


def _parse_prompt_ids(value: str) -> np.ndarray:
    try:
        tokens = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("prompt IDs must be comma-separated integers") from error
    if not tokens or any(token < 0 for token in tokens):
        raise ValueError("prompt IDs must contain nonnegative token IDs")
    return np.ascontiguousarray(tokens, dtype=np.int32)


def _configure(library: ctypes.CDLL) -> None:
    library.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.ck_model_init_with_manifest.restype = ctypes.c_int
    library.ck_model_free.argtypes = []
    library.ck_model_set_encoder_memory.argtypes = [F32P, ctypes.c_int, ctypes.c_int]
    library.ck_model_set_encoder_memory.restype = ctypes.c_int
    library.ck_model_get_encoder_memory_capacity.restype = ctypes.c_int
    library.ck_model_get_encoder_memory_dim.restype = ctypes.c_int
    library.ck_model_embed_tokens.argtypes = [I32P, ctypes.c_int]
    library.ck_model_embed_tokens.restype = ctypes.c_int
    library.ck_model_forward.argtypes = [F32P]
    library.ck_model_forward.restype = ctypes.c_int
    library.ck_model_decode.argtypes = [ctypes.c_int32, F32P]
    library.ck_model_decode.restype = ctypes.c_int
    library.ck_model_get_vocab_size.restype = ctypes.c_int


def certify(args: argparse.Namespace) -> dict[str, object]:
    runtime = args.runtime.resolve()
    library_path = runtime / "libmodel.so"
    library = ctypes.CDLL(str(library_path))
    loaded = provenance._verify_loaded_libraries(
        library,
        runtime,
        {
            "model_library": "ck_model_init_with_manifest",
            "engine_library": "ck_set_num_threads",
        },
    )
    _configure(library)
    encoder = _encoder_fixture(args.encoder_fixture)
    prompt = _parse_prompt_ids(args.prompt_ids)
    expected = _expected_tokens(args.reference_summary, args.eos_token_id)
    identities = {
        "generated_c": provenance._identity(runtime / "model_v8.c"),
        "model_library": provenance._identity(library_path),
        "engine_library": provenance._identity(loaded["engine_library"]),
        "weights": provenance._identity(args.weights),
        "manifest_map": provenance._identity(args.manifest_map),
        "encoder_fixture": provenance._identity(args.encoder_fixture),
        "reference_summary": provenance._identity(args.reference_summary),
    }
    started = time.perf_counter()
    statuses: dict[str, int] = {}
    actual: list[int] = []
    try:
        statuses["init"] = int(
            library.ck_model_init_with_manifest(
                str(args.weights.resolve()).encode(),
                str(args.manifest_map.resolve()).encode(),
            )
        )
        if statuses["init"] != 0:
            raise RuntimeError(f"generated decoder initialization failed: {statuses['init']}")
        capacity = int(library.ck_model_get_encoder_memory_capacity())
        dimension = int(library.ck_model_get_encoder_memory_dim())
        if encoder.shape[0] > capacity or encoder.shape[1] != dimension:
            raise ValueError(
                f"encoder fixture shape {encoder.shape} exceeds generated decoder "
                f"contract [{capacity}, {dimension}]"
            )
        statuses["set_encoder_memory"] = int(
            library.ck_model_set_encoder_memory(
                encoder.ctypes.data_as(F32P), encoder.shape[0], encoder.shape[1]
            )
        )
        vocab_size = int(library.ck_model_get_vocab_size())
        if vocab_size <= 0 or any(token >= vocab_size for token in prompt):
            raise ValueError("prompt IDs exceed the generated decoder vocabulary")
        statuses["prompt_prefill"] = int(
            library.ck_model_embed_tokens(prompt.ctypes.data_as(I32P), prompt.size)
        )
        logits = np.empty(vocab_size, dtype=np.float32)
        statuses["prompt_logits"] = int(
            library.ck_model_forward(logits.ctypes.data_as(F32P))
        )
        if any(statuses[name] != 0 for name in ("set_encoder_memory", "prompt_prefill", "prompt_logits")):
            raise RuntimeError(f"generated decoder setup failed: {statuses}")
        for index in range(len(expected)):
            if not np.isfinite(logits).all():
                raise RuntimeError(f"non-finite logits at generated step {index}")
            token = int(np.argmax(logits))
            actual.append(token)
            if index + 1 == len(expected):
                break
            status = int(library.ck_model_decode(token, logits.ctypes.data_as(F32P)))
            statuses[f"decode_{index}"] = status
            if status != 0:
                raise RuntimeError(f"generated decoder failed at step {index}: {status}")
    finally:
        library.ck_model_free()
    checks = {
        "trajectory_exact": actual == expected,
        "complete_trajectory": len(actual) == len(expected),
        "generated_runtime_only": True,
    }
    return {
        "schema": "cke.v8.cohere_generated_decoder_certification",
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "generated decoder token trajectory with supplied encoder memory",
        "checks": checks,
        "prompt_token_ids": prompt.tolist(),
        "expected_emitted_token_ids": expected,
        "actual_emitted_token_ids": actual,
        "encoder_shape": list(encoder.shape),
        "statuses": statuses,
        "wall_seconds": time.perf_counter() - started,
        "identity": identities,
        "not_certified": [
            "native tokenizer and prompt construction",
            "native timestamp extraction",
            "standalone end-to-end transcription",
            "long-audio scheduling",
            "performance",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest-map", type=Path, required=True)
    parser.add_argument("--encoder-fixture", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--prompt-ids", required=True)
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report: dict[str, object]
    try:
        report = certify(args)
    except Exception as error:
        report = {
            "schema": "cke.v8.cohere_generated_decoder_certification",
            "schema_version": 1,
            "status": "ERROR",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
