#!/usr/bin/env python3
"""Capture a reproducible Cohere Transcribe oracle run and X-Ray checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


SCHEMA = "cke.v8.cohere_transcribe_oracle_certification"
CHECKPOINT_SCHEMA = "cke.checkpoint_manifest"
CHECKPOINT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas/checkpoint_manifest.schema.json"
)
BENCH_PATTERN = re.compile(r"^cohere:\s+([^\n]+?)\s+([0-9]+(?:\.[0-9]+)?) ms(?:\s|$)")


class CertificationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _host_metadata() -> dict[str, Any]:
    processor = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == "model name":
                processor = value.strip()
                break
    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "processor": processor,
        "logical_cpus": os.cpu_count(),
    }


def _oracle_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def _read_headered_f32(path: Path, dimensions: int) -> tuple[list[int], bytes]:
    if not path.is_file():
        raise CertificationError(f"oracle did not export checkpoint: {path}")
    payload = path.read_bytes()
    header_bytes = dimensions * 4
    if len(payload) < header_bytes:
        raise CertificationError(f"checkpoint header is truncated: {path}")
    shape = list(struct.unpack(f"<{dimensions}i", payload[:header_bytes]))
    if any(extent <= 0 for extent in shape):
        raise CertificationError(f"checkpoint has invalid shape {shape}: {path}")
    values = payload[header_bytes:]
    expected = 4
    for extent in shape:
        expected *= extent
    if len(values) != expected:
        raise CertificationError(
            f"checkpoint payload has {len(values)} bytes, expected {expected}: {path}"
        )
    return shape, values


def _checkpoint_entry(
    *,
    checkpoint_id: str,
    producer: str,
    layer: int,
    source_path: Path,
    normalized_path: Path,
    axis_names: list[str],
    dimensions: int = 2,
    phase: str = "prefill",
) -> dict[str, Any]:
    source_shape, values = _read_headered_f32(source_path, dimensions)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.write_bytes(values)
    logical_shape = list(reversed(source_shape)) if dimensions == 2 else source_shape
    if len(axis_names) != dimensions:
        raise CertificationError(
            f"{checkpoint_id}: {len(axis_names)} axes do not match {dimensions} dimensions"
        )
    return {
        "checkpoint_id": checkpoint_id,
        "producer": producer,
        "phase": phase,
        "layer": layer,
        "tensor_path": str(normalized_path.resolve()),
        "storage_dtype": "fp32",
        "exported_dtype": "fp32",
        "logical_shape": logical_shape,
        "physical_shape": logical_shape,
        "logical_layout": "_".join(axis_names),
        "axis_names": axis_names,
        "physical_axis_names": axis_names,
        "resolved_contract_id": f"oracle.crispasr.{checkpoint_id}",
        "kernel_id": f"oracle_crispasr_{producer}",
        "function": f"crispasr::{producer}",
        "sha256": sha256_file(normalized_path),
    }


def build_checkpoint_manifest(dump_dir: Path, normalized_dir: Path) -> dict[str, Any]:
    checkpoints: list[dict[str, Any]] = []
    checkpoints.append(
        _checkpoint_entry(
            checkpoint_id="audio.frontend.log_mel.output",
            producer="cohere_log_mel",
            layer=-1,
            source_path=dump_dir / "crisp.mel.bin",
            normalized_path=normalized_dir / "log_mel.f32",
            axis_names=["frame", "feature"],
        )
    )
    block_paths: dict[int, Path] = {}
    for path in dump_dir.glob("crisp.block*.bin"):
        match = re.fullmatch(r"crisp\.block(\d+)\.bin", path.name)
        if match:
            block_paths[int(match.group(1))] = path
    expected_layers = set(range(48))
    if set(block_paths) != expected_layers:
        missing = sorted(expected_layers - set(block_paths))
        unexpected = sorted(set(block_paths) - expected_layers)
        raise CertificationError(
            "oracle must export all 48 Conformer blocks; "
            f"missing={missing} unexpected={unexpected}"
        )
    for layer in range(48):
        path = block_paths[layer]
        checkpoints.append(
            _checkpoint_entry(
                checkpoint_id=f"audio.encoder.layer.{layer}.output",
                producer="cohere_conformer_block",
                layer=layer,
                source_path=path,
                normalized_path=normalized_dir / f"encoder_layer_{layer}.f32",
                axis_names=["token", "channel"],
            )
        )
    checkpoints.append(
        _checkpoint_entry(
            checkpoint_id="audio.encoder.final_preprojection.output",
            producer="cohere_encoder_final",
            layer=-1,
            source_path=dump_dir / "crisp.enc_final.bin",
            normalized_path=normalized_dir / "encoder_final_preprojection.f32",
            axis_names=["token", "channel"],
        )
    )
    checkpoints.append(
        _checkpoint_entry(
            checkpoint_id="audio.decoder.cross_attention.context",
            producer="cohere_encoder_projection",
            layer=-1,
            source_path=dump_dir / "crisp.enc_out.bin",
            normalized_path=normalized_dir / "cross_attention_context.f32",
            axis_names=["token", "channel"],
        )
    )
    attention = dump_dir / "crisp.decoder_attention.bin"
    if attention.is_file():
        checkpoints.append(
            _checkpoint_entry(
                checkpoint_id="audio.decoder.cross_attention.weights",
                producer="cohere_decoder_cross_attention",
                layer=7,
                source_path=attention,
                normalized_path=normalized_dir / "decoder_cross_attention.f32",
                axis_names=["token", "head", "encoder_token"],
                dimensions=3,
                phase="decode",
            )
        )
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "backend": "crispasr",
        "run": {
            "model": "cohere-transcribe",
            "phase": "decode",
            "source": "CrispASR Cohere backend",
        },
        "checkpoints": checkpoints,
    }
    schema = json.loads(CHECKPOINT_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise CertificationError(
            f"checkpoint manifest violates schema at {location}: {errors[0].message}"
        )
    return manifest


def _parse_benchmarks(log: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    for line in log.splitlines():
        match = BENCH_PATTERN.match(line)
        if match:
            key = re.sub(r"[^a-z0-9]+", "_", match.group(1).lower()).strip("_")
            timings[key] = float(match.group(2))
    return timings


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-bin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--language", default="en")
    parser.add_argument("--dump-attention", action="store_true")
    args = parser.parse_args(argv)

    oracle = args.oracle_bin.expanduser().resolve()
    model = args.model.expanduser().resolve()
    audio = args.audio.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    for label, path in (("oracle", oracle), ("model", model), ("audio", audio)):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")

    dump_dir = output_dir / "oracle_raw"
    normalized_dir = output_dir / "checkpoints"
    shutil.rmtree(dump_dir, ignore_errors=True)
    shutil.rmtree(normalized_dir, ignore_errors=True)
    dump_dir.mkdir(parents=True, exist_ok=True)
    transcript_base = output_dir / "transcript"
    env = os.environ.copy()
    env.update(
        {
            "CRISPASR_COHERE_BENCH": "1",
            "CRISPASR_COHERE_PROF": "1",
            "CRISPASR_COHERE_THREADS": str(args.threads),
            "CRISPASR_COHERE_DUMP_STAGES": str(dump_dir),
            "CRISPASR_COHERE_DUMP_ENCOUT": str(dump_dir / "crisp.enc_out.bin"),
        }
    )
    if args.dump_attention:
        env["CRISPASR_COHERE_DUMP_ATTN"] = str(
            dump_dir / "crisp.decoder_attention.bin"
        )
    command = [
        str(oracle),
        "--backend",
        "cohere",
        "-m",
        str(model),
        "-t",
        str(args.threads),
        "-l",
        args.language,
        "-nt",
        "-oj",
        "-of",
        str(transcript_base),
        "-f",
        str(audio),
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started
    log_path = output_dir / "oracle.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    transcript_path = transcript_base.with_suffix(".json")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "FAIL",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": _host_metadata(),
        "oracle": {
            "path": str(oracle),
            "sha256": sha256_file(oracle),
            "version": _oracle_version(oracle),
            "returncode": completed.returncode,
        },
        "model": {"path": str(model), "sha256": sha256_file(model)},
        "input": {"path": str(audio), "sha256": sha256_file(audio)},
        "threads": args.threads,
        "language": args.language,
        "wall_seconds": elapsed,
        "timings_ms": _parse_benchmarks(completed.stdout),
        "log": str(log_path),
        "command": command,
    }
    try:
        if completed.returncode != 0:
            raise CertificationError(
                f"oracle exited with status {completed.returncode}; inspect {log_path}"
            )
        if not transcript_path.is_file():
            raise CertificationError(f"oracle did not write transcript JSON: {transcript_path}")
        json.loads(transcript_path.read_text(encoding="utf-8"))
        manifest = build_checkpoint_manifest(dump_dir, normalized_dir)
        manifest_path = output_dir / "oracle_checkpoint_manifest.json"
        _write_json(manifest_path, manifest)
        report.update(
            {
                "status": "PASS",
                "transcript": {
                    "path": str(transcript_path),
                    "sha256": sha256_file(transcript_path),
                },
                "checkpoint_manifest": str(manifest_path),
                "checkpoint_count": len(manifest["checkpoints"]),
            }
        )
    except (CertificationError, json.JSONDecodeError) as exc:
        report["reason"] = str(exc)
    report_path = output_dir / "summary.json"
    _write_json(report_path, report)
    print(f"status={report['status']} report={report_path}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
