#!/usr/bin/env python3
"""Certify native WAV-to-text execution across generated Cohere components."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


TOKEN_IDS = re.compile(r"^token_ids=([0-9]+(?:,[0-9]+)*)$", re.MULTILINE)


def _identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _reference(path: Path) -> tuple[list[int], str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    tokens = report.get("generated_token_ids")
    transcript = report.get("transcript")
    if not isinstance(tokens, list) or not tokens or not all(
        isinstance(token, int) and token >= 0 for token in tokens
    ):
        raise ValueError("reference report has no valid generated token IDs")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("reference report has no transcript")
    return tokens, transcript.strip()


def _tokens(stderr: str) -> list[int]:
    match = TOKEN_IDS.search(stderr)
    if match is None:
        raise ValueError("native host did not report its token trajectory")
    return [int(value) for value in match.group(1).split(",")]


def _dependency_closure(paths: list[Path]) -> dict[str, str]:
    closure: dict[str, str] = {}
    for path in paths:
        completed = subprocess.run(
            ["ldd", str(path)], text=True, capture_output=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(f"cannot inspect native dependencies for {path}")
        closure[str(path.resolve())] = completed.stdout
    return closure


def certify(args: argparse.Namespace) -> dict[str, object]:
    encoder = args.encoder_runtime.resolve()
    decoder = args.decoder_runtime.resolve()
    paths = {
        "native_host": args.host.resolve(),
        "encoder_model_library": encoder / "libmodel.so",
        "encoder_engine_library": encoder / "libckernel_engine.so",
        "encoder_weights": encoder / "weights.bump",
        "encoder_manifest_map": encoder / "weights_manifest.map",
        "decoder_model_library": decoder / "libmodel.so",
        "decoder_engine_library": decoder / "libckernel_engine.so",
        "decoder_tokenizer_library": decoder / "libckernel_tokenizer.so",
        "decoder_weights": decoder / "weights.bump",
        "decoder_manifest_map": decoder / "weights_manifest.map",
        "input": args.input.resolve(),
        "reference": args.reference.resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing certification inputs: " + ", ".join(missing))
    expected_tokens, expected_text = _reference(paths["reference"])
    dependencies = _dependency_closure([
        paths["native_host"],
        paths["encoder_model_library"],
        paths["decoder_model_library"],
    ])
    command = [
        str(paths["native_host"]),
        str(paths["encoder_model_library"]),
        str(paths["encoder_weights"]),
        str(paths["encoder_manifest_map"]),
        str(paths["decoder_model_library"]),
        str(paths["decoder_weights"]),
        str(paths["decoder_manifest_map"]),
        str(paths["input"]),
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTHON") and key not in {"VIRTUAL_ENV", "CONDA_PREFIX"}
    }
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    actual_tokens = _tokens(completed.stderr) if completed.returncode == 0 else []
    actual_text = completed.stdout.strip()
    checks = {
        "native_exit_zero": completed.returncode == 0,
        "token_trajectory_exact": actual_tokens == expected_tokens,
        "transcript_exact": actual_text == expected_text,
        "python_environment_removed": all(
            not key.startswith("PYTHON") for key in environment
        ),
        "native_dependencies_resolved": all(
            "not found" not in value for value in dependencies.values()
        ),
        "no_python_dynamic_dependency": all(
            "libpython" not in value.lower() for value in dependencies.values()
        ),
    }
    return {
        "schema": "cke.v8.cohere_generated_standalone_certification",
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "native short WAV-to-text through generated frontend, encoder, decoder, and tokenizer",
        "checks": checks,
        "expected_token_ids": expected_tokens,
        "actual_token_ids": actual_tokens,
        "expected_transcript": expected_text,
        "actual_transcript": actual_text,
        "wall_seconds": elapsed,
        "stderr": completed.stderr,
        "native_dependencies": dependencies,
        "identity": {name: _identity(path) for name, path in paths.items()},
        "not_certified": [
            "timestamps",
            "long-audio window scheduling",
            "resampling",
            "multilingual quality",
            "performance",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--encoder-runtime", type=Path, required=True)
    parser.add_argument("--decoder-runtime", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = certify(args)
    except Exception as error:
        report = {
            "schema": "cke.v8.cohere_generated_standalone_certification",
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
