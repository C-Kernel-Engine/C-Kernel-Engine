#!/usr/bin/env python3
"""Certify Cohere Transcribe's generated-C encoder components and schedule."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import certify_parakeet_generated_frontend_v8 as provenance


F32P = ctypes.POINTER(ctypes.c_float)


def _fixture(path: Path) -> dict[str, np.ndarray]:
    required = {
        "audio.frontend.log_mel.output": 2,
        "audio.encoder.subsampling.output": 2,
        "audio.encoder.relative_position.output": 2,
        "audio.encoder.layer.0.output": 2,
        "audio.encoder.projected.output": 2,
    }
    with np.load(path, allow_pickle=False) as fixture:
        missing = sorted(set(required) - set(fixture.files))
        if missing:
            raise ValueError(f"Cohere encoder fixture is incomplete: {missing}")
        arrays = {
            name: np.ascontiguousarray(fixture[name]) for name in required
        }
    for name, rank in required.items():
        value = arrays[name]
        if value.dtype != np.float32 or value.ndim != rank or not value.size:
            raise ValueError(f"{name} must be a non-empty rank-{rank} float32 tensor")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    features = arrays["audio.frontend.log_mel.output"]
    subsampling = arrays["audio.encoder.subsampling.output"]
    relative = arrays["audio.encoder.relative_position.output"]
    layer0 = arrays["audio.encoder.layer.0.output"]
    projected = arrays["audio.encoder.projected.output"]
    if features.shape[1] != 128:
        raise ValueError("Cohere fixture must contain 128-channel frontend features")
    if subsampling.shape != layer0.shape or subsampling.shape[1] != 1280:
        raise ValueError("Cohere encoder fixture must preserve [frames, 1280]")
    if relative.shape != (2 * subsampling.shape[0] - 1, 1280):
        raise ValueError("Cohere relative-position fixture has invalid geometry")
    if projected.shape != (subsampling.shape[0], 1024):
        raise ValueError("Cohere projected encoder fixture has invalid geometry")
    return arrays


def _compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "finite": bool(np.isfinite(actual).all()),
        "bit_exact": bool(np.array_equal(actual.view(np.uint32), expected.view(np.uint32))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def _load(runtime: Path, weights: Path, manifest_map: Path) -> tuple[ctypes.CDLL, dict]:
    library = ctypes.CDLL(str(runtime / "libmodel.so"))
    loaded = provenance._verify_loaded_libraries(
        library,
        runtime,
        {
            "model_library": "ck_model_init_with_manifest",
            "engine_library": "ck_set_num_threads",
        },
    )
    library.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.ck_model_init_with_manifest.restype = ctypes.c_int
    library.ck_model_free.argtypes = []
    status = int(
        library.ck_model_init_with_manifest(
            str(weights.resolve()).encode(), str(manifest_map.resolve()).encode()
        )
    )
    return library, {
        "init_status": status,
        "weights": provenance._identity(weights),
        "manifest_map": provenance._identity(manifest_map),
        "generated_c": provenance._identity(runtime / "model_v8.c"),
        "loaded_libraries": {
            name: provenance._identity(path) for name, path in loaded.items()
        },
    }


def _certify_subsampling(
    runtime: Path, weights: Path, manifest_map: Path, fixture_path: Path
) -> dict[str, object]:
    arrays = _fixture(fixture_path)
    features = arrays["audio.frontend.log_mel.output"]
    expected = arrays["audio.encoder.subsampling.output"]
    library, identity = _load(runtime, weights, manifest_map)
    library.ck_model_audio_subsampling_workspace_bytes.argtypes = [ctypes.c_int]
    library.ck_model_audio_subsampling_workspace_bytes.restype = ctypes.c_size_t
    library.ck_model_run_audio_subsampling.argtypes = [
        F32P, ctypes.c_int, ctypes.c_int, F32P, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_size_t,
    ]
    library.ck_model_run_audio_subsampling.restype = ctypes.c_int
    output = np.empty_like(expected)
    rejected = np.full_like(expected, np.float32(1234.5))
    produced = ctypes.c_int(-1)
    rejected_frames = ctypes.c_int(-1)
    workspace_bytes = int(
        library.ck_model_audio_subsampling_workspace_bytes(features.shape[0])
    )
    workspace = np.empty(workspace_bytes, dtype=np.uint8)
    started = time.perf_counter()
    try:
        rejected_status = int(library.ck_model_run_audio_subsampling(
            features.ctypes.data_as(F32P), features.shape[0], features.shape[0],
            rejected.ctypes.data_as(F32P), expected.shape[0] - 1,
            ctypes.byref(rejected_frames), workspace.ctypes.data, workspace_bytes))
        status = int(library.ck_model_run_audio_subsampling(
            features.ctypes.data_as(F32P), features.shape[0], features.shape[0],
            output.ctypes.data_as(F32P), output.shape[0], ctypes.byref(produced),
            workspace.ctypes.data, workspace_bytes))
    finally:
        library.ck_model_free()
    comparison = _compare(output, expected)
    checks = {
        "initialized": identity["init_status"] == 0,
        "call_succeeded": status == 0 and produced.value == expected.shape[0],
        "undersized_output_rejected": rejected_status != 0
        and bool(np.all(rejected == np.float32(1234.5))),
        "rmse_within_2e_4": comparison["rmse"] <= 2.0e-4,
        "max_abs_within_3e_3": comparison["max_abs"] <= 3.0e-3,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "statuses": {"run": status, "undersized_output": rejected_status},
        "numerical": comparison,
        "execution": {"seconds": time.perf_counter() - started,
                      "input_frames": features.shape[0],
                      "output_frames": produced.value,
                      "workspace_bytes": workspace_bytes},
        "identity": identity,
    }


def _certify_block(
    runtime: Path, weights: Path, manifest_map: Path, fixture_path: Path
) -> dict[str, object]:
    arrays = _fixture(fixture_path)
    input_value = arrays["audio.encoder.subsampling.output"]
    expected_relative = arrays["audio.encoder.relative_position.output"]
    expected = arrays["audio.encoder.layer.0.output"]
    library, identity = _load(runtime, weights, manifest_map)
    library.ck_model_prepare_audio_relative_positions.argtypes = [
        ctypes.c_int, F32P, ctypes.c_size_t]
    library.ck_model_prepare_audio_relative_positions.restype = ctypes.c_int
    library.ck_model_audio_fastconformer_block_workspace_bytes.argtypes = [ctypes.c_int]
    library.ck_model_audio_fastconformer_block_workspace_bytes.restype = ctypes.c_size_t
    library.ck_model_run_audio_fastconformer_block.argtypes = [
        ctypes.c_int, F32P, F32P, ctypes.c_int, F32P,
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    library.ck_model_run_audio_fastconformer_block.restype = ctypes.c_int
    relative = np.empty_like(expected_relative)
    output = np.empty_like(expected)
    rejected = np.full_like(expected, np.float32(1234.5))
    workspace_bytes = int(
        library.ck_model_audio_fastconformer_block_workspace_bytes(input_value.shape[0])
    )
    workspace = np.empty(workspace_bytes, dtype=np.uint8)
    started = time.perf_counter()
    try:
        relative_status = int(library.ck_model_prepare_audio_relative_positions(
            input_value.shape[0], relative.ctypes.data_as(F32P), relative.size))
        rejected_status = int(library.ck_model_run_audio_fastconformer_block(
            0, input_value.ctypes.data_as(F32P), relative.ctypes.data_as(F32P),
            input_value.shape[0], rejected.ctypes.data_as(F32P),
            workspace.ctypes.data, workspace_bytes - 1))
        status = int(library.ck_model_run_audio_fastconformer_block(
            0, input_value.ctypes.data_as(F32P), relative.ctypes.data_as(F32P),
            input_value.shape[0], output.ctypes.data_as(F32P),
            workspace.ctypes.data, workspace_bytes))
    finally:
        library.ck_model_free()
    relative_comparison = _compare(relative, expected_relative)
    comparison = _compare(output, expected)
    checks = {
        "initialized": identity["init_status"] == 0,
        "relative_position_succeeded": relative_status == 0,
        "call_succeeded": status == 0,
        "undersized_workspace_rejected": rejected_status != 0
        and bool(np.all(rejected == np.float32(1234.5))),
        "relative_position_bit_exact": relative_comparison["bit_exact"],
        "rmse_within_1e_5": comparison["rmse"] <= 1.0e-5,
        "max_abs_within_1e_4": comparison["max_abs"] <= 1.0e-4,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "statuses": {"relative_position": relative_status, "run": status,
                     "undersized_workspace": rejected_status},
        "numerical": {"relative_position": relative_comparison, "layer_0": comparison},
        "execution": {"seconds": time.perf_counter() - started,
                      "frames": input_value.shape[0],
                      "workspace_bytes": workspace_bytes},
        "identity": identity,
    }


def _certify_full_encoder(
    runtime: Path, weights: Path, manifest_map: Path, fixture_path: Path
) -> dict[str, object]:
    arrays = _fixture(fixture_path)
    features = arrays["audio.frontend.log_mel.output"]
    expected = arrays["audio.encoder.projected.output"]
    library, identity = _load(runtime, weights, manifest_map)
    library.ck_model_audio_encoder_workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    library.ck_model_audio_encoder_workspace_bytes.restype = ctypes.c_size_t
    library.ck_model_run_audio_encoder.argtypes = [
        F32P, ctypes.c_int, ctypes.c_int, F32P, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_size_t,
    ]
    library.ck_model_run_audio_encoder.restype = ctypes.c_int
    output = np.empty_like(expected)
    rejected = np.full_like(expected, np.float32(1234.5))
    frames = ctypes.c_int(-1)
    rejected_frames = ctypes.c_int(-1)
    workspace_bytes = int(library.ck_model_audio_encoder_workspace_bytes(
        features.shape[0], expected.shape[0]))
    workspace = np.empty(workspace_bytes, dtype=np.uint8)
    started = time.perf_counter()
    try:
        rejected_status = int(library.ck_model_run_audio_encoder(
            features.ctypes.data_as(F32P), features.shape[0], features.shape[0],
            rejected.ctypes.data_as(F32P), rejected.shape[0],
            ctypes.byref(rejected_frames), workspace.ctypes.data,
            workspace_bytes - 1))
        status = int(library.ck_model_run_audio_encoder(
            features.ctypes.data_as(F32P), features.shape[0], features.shape[0],
            output.ctypes.data_as(F32P), output.shape[0], ctypes.byref(frames),
            workspace.ctypes.data, workspace_bytes))
    finally:
        library.ck_model_free()
    comparison = _compare(output, expected)
    checks = {
        "initialized": identity["init_status"] == 0,
        "call_succeeded": status == 0 and frames.value == expected.shape[0],
        "undersized_workspace_rejected": rejected_status != 0
        and bool(np.all(rejected == np.float32(1234.5))),
        "bit_exact": comparison["bit_exact"],
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "statuses": {"run": status, "undersized_workspace": rejected_status},
        "numerical": comparison,
        "execution": {"seconds": time.perf_counter() - started,
                      "input_frames": features.shape[0],
                      "output_frames": frames.value,
                      "layers": 48,
                      "workspace_bytes": workspace_bytes},
        "identity": identity,
    }


def _run_child(component: str, args: argparse.Namespace, output: Path) -> None:
    if component == "subsampling":
        runtime, weights, manifest_map = (
            args.subsampling_runtime, args.subsampling_weights,
            args.subsampling_manifest_map,
        )
    elif component == "block":
        runtime, weights, manifest_map = (
            args.block_runtime, args.block_weights, args.block_manifest_map,
        )
    else:
        runtime, weights, manifest_map = (
            args.encoder_runtime, args.encoder_weights, args.encoder_manifest_map,
        )
    command = [
        sys.executable, str(Path(__file__).resolve()), "--component", component,
        "--runtime", str(runtime), "--weights", str(weights),
        "--manifest-map", str(manifest_map), "--fixture", str(args.fixture),
        "--output", str(output),
    ]
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(Path(runtime).resolve())
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode not in {0, 1} or not output.is_file():
        raise RuntimeError(f"{component} certification process failed: {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("subsampling", "block", "full_encoder"))
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--manifest-map", type=Path)
    parser.add_argument("--subsampling-runtime", type=Path)
    parser.add_argument("--subsampling-weights", type=Path)
    parser.add_argument("--subsampling-manifest-map", type=Path)
    parser.add_argument("--block-runtime", type=Path)
    parser.add_argument("--block-weights", type=Path)
    parser.add_argument("--block-manifest-map", type=Path)
    parser.add_argument("--encoder-runtime", type=Path)
    parser.add_argument("--encoder-weights", type=Path)
    parser.add_argument("--encoder-manifest-map", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.component:
        required = (args.runtime, args.weights, args.manifest_map)
        if any(value is None for value in required):
            parser.error("component certification requires runtime, weights, and manifest-map")
        if args.component == "subsampling":
            report = _certify_subsampling(
                args.runtime.resolve(), args.weights, args.manifest_map, args.fixture)
        elif args.component == "block":
            report = _certify_block(
                args.runtime.resolve(), args.weights, args.manifest_map, args.fixture)
        else:
            report = _certify_full_encoder(
                args.runtime.resolve(), args.weights, args.manifest_map, args.fixture)
    else:
        required = (
            args.subsampling_runtime, args.subsampling_weights,
            args.subsampling_manifest_map, args.block_runtime,
            args.block_weights, args.block_manifest_map, args.encoder_runtime,
            args.encoder_weights, args.encoder_manifest_map,
        )
        if any(value is None for value in required):
            parser.error("combined certification requires all three encoder runtime bundles")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            subsampling_path = directory / "subsampling.json"
            block_path = directory / "block.json"
            encoder_path = directory / "encoder.json"
            _run_child("subsampling", args, subsampling_path)
            _run_child("block", args, block_path)
            _run_child("full_encoder", args, encoder_path)
            components = {
                "subsampling": json.loads(subsampling_path.read_text(encoding="utf-8")),
                "fastconformer_block_0": json.loads(block_path.read_text(encoding="utf-8")),
                "full_encoder": json.loads(encoder_path.read_text(encoding="utf-8")),
            }
        report = {
            "schema": "cke.cohere_transcribe.generated_encoder_components.v1",
            "status": "pass" if all(
                row.get("status") == "pass" for row in components.values()
            ) else "fail",
            "scope": "generated_c_complete_48_block_encoder",
            "components": components,
            "identity": {"fixture": provenance._identity(args.fixture)},
            "claim_boundary": {
                "subsampling": "certified",
                "fastconformer_block_0": "certified",
                "complete_encoder": "certified_short_fixture",
                "decoder": "not_generated",
                "standalone_transcription": "not_certified",
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
