#!/usr/bin/env python3
"""Certify all generated-C Parakeet FastConformer encoder blocks."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import time
from pathlib import Path

import numpy as np

import certify_parakeet_generated_frontend_v8 as frontend
import certify_parakeet_generated_subsampling_v8 as subsampling


F32P = ctypes.POINTER(ctypes.c_float)


def _checked_fixture(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as fixture:
        first_input = fixture["encoder.subsampling"]
        first_expected = fixture["encoder.layer.0"]
        final_expected = fixture["encoder.layer.23"]
    arrays = (first_input, first_expected, final_expected)
    if any(array.dtype != np.float32 for array in arrays):
        raise ValueError("encoder block fixtures must use float32 storage")
    if any(array.ndim != 3 or array.shape[0] != 1 for array in arrays):
        raise ValueError("encoder block fixtures must have shape [1, frames, hidden]")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("encoder block fixture shapes must agree")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("encoder block fixtures must contain only finite values")
    return tuple(np.ascontiguousarray(array[0]) for array in arrays)


def _comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | bool]:
    actual_f64 = actual.astype(np.float64)
    expected_f64 = expected.astype(np.float64)
    difference = actual_f64 - expected_f64
    expected_rms = float(np.sqrt(np.mean(expected_f64 ** 2)))
    denominator = float(np.linalg.norm(actual_f64.ravel()) * np.linalg.norm(expected_f64.ravel()))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    return {
        "finite": bool(np.isfinite(actual).all()),
        "rmse": rmse,
        "normalized_rmse": rmse / expected_rms if expected_rms else math.inf,
        "max_abs": float(np.max(np.abs(difference))),
        "cosine": float(np.dot(actual_f64.ravel(), expected_f64.ravel()) / denominator)
        if denominator else math.nan,
    }


def certify(runtime_dir: Path, weights: Path, manifest_map: Path,
            fixture_path: Path) -> dict[str, object]:
    runtime_dir = runtime_dir.resolve()
    codegen_bundle = subsampling._validate_bundle_outputs(
        runtime_dir, ".ck_codegen_bundle.json")
    runtime_bundle = subsampling._validate_bundle_outputs(
        runtime_dir, ".ck_runtime_bundle.json")
    first_input, first_expected, final_expected = _checked_fixture(fixture_path)
    frames, hidden_size = first_input.shape

    ctypes.CDLL(str(runtime_dir / "libckernel_engine.so"), mode=ctypes.RTLD_GLOBAL)
    library = ctypes.CDLL(str(runtime_dir / "libmodel.so"))
    loaded = frontend._verify_loaded_libraries(library, runtime_dir)
    library.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.ck_model_init_with_manifest.restype = ctypes.c_int
    library.ck_model_free.argtypes = []
    library.ck_model_audio_relative_position_elements.argtypes = [ctypes.c_int]
    library.ck_model_audio_relative_position_elements.restype = ctypes.c_size_t
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

    init_status = int(library.ck_model_init_with_manifest(
        str(weights.resolve()).encode(), str(manifest_map.resolve()).encode()))
    relative = np.empty((2 * frames - 1, hidden_size), dtype=np.float32)
    current = first_input.copy()
    next_output = np.empty_like(current)
    statuses = {"init": init_status, "relative_position": -1, "layers": []}
    first_actual = None
    elapsed = 0.0
    try:
        if init_status == 0:
            relative_elements = int(
                library.ck_model_audio_relative_position_elements(frames))
            if relative_elements != relative.size:
                raise RuntimeError("generated runtime returned invalid relative-position size")
            statuses["relative_position"] = int(
                library.ck_model_prepare_audio_relative_positions(
                    frames, relative.ctypes.data_as(F32P), relative.size))
            workspace_bytes = int(
                library.ck_model_audio_fastconformer_block_workspace_bytes(frames))
            if workspace_bytes <= 0:
                raise RuntimeError("generated runtime returned invalid block workspace size")
            workspace = np.empty(workspace_bytes, dtype=np.uint8)
            started = time.perf_counter()
            if statuses["relative_position"] == 0:
                for layer in range(24):
                    status = int(library.ck_model_run_audio_fastconformer_block(
                        layer, current.ctypes.data_as(F32P),
                        relative.ctypes.data_as(F32P), frames,
                        next_output.ctypes.data_as(F32P), workspace.ctypes.data,
                        workspace_bytes))
                    statuses["layers"].append(status)
                    if status != 0:
                        break
                    if layer == 0:
                        first_actual = next_output.copy()
                    current, next_output = next_output, current
            elapsed = time.perf_counter() - started
    finally:
        library.ck_model_free()

    first_cmp = _comparison(
        first_actual if first_actual is not None else next_output, first_expected)
    final_cmp = _comparison(current, final_expected)
    checks = {
        "all_calls_succeeded": init_status == 0
        and statuses["relative_position"] == 0
        and len(statuses["layers"]) == 24
        and all(status == 0 for status in statuses["layers"]),
        "first_layer_rmse_within_1e_5": first_cmp["rmse"] <= 1.0e-5,
        "first_layer_max_abs_within_1e_4": first_cmp["max_abs"] <= 1.0e-4,
        "final_layer_rmse_within_1e_5": final_cmp["rmse"] <= 1.0e-5,
        "final_layer_max_abs_within_1e_4": final_cmp["max_abs"] <= 1.0e-4,
    }
    return {
        "schema": "cke.parakeet.generated_fastconformer_certification.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "generated_c_fastconformer_24_block_encoder_body",
        "checks": checks,
        "statuses": statuses,
        "numerical": {"layer_0": first_cmp, "layer_23": final_cmp},
        "execution": {"elapsed_seconds": elapsed, "frames": frames,
                      "hidden_size": hidden_size, "layers": 24},
        "identity": {
            "weights": frontend._identity(weights),
            "manifest_map": frontend._identity(manifest_map),
            "fixture": frontend._identity(fixture_path),
            "loaded_libraries": {
                name: frontend._identity(path) for name, path in loaded.items()
            },
            "codegen_bundle": frontend._identity(
                runtime_dir / ".ck_codegen_bundle.json"),
            "runtime_bundle": frontend._identity(
                runtime_dir / ".ck_runtime_bundle.json"),
            "codegen_bundle_schema": codegen_bundle.get("schema"),
            "runtime_bundle_schema": runtime_bundle.get("schema"),
        },
        "claim_boundary": {
            "fastconformer_blocks": "certified",
            "encoder_projection": "not_generated",
            "tdt_decoder": "not_generated",
            "standalone_transcription": "not_certified",
            "cohere_transcribe": "not_certified",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest-map", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = certify(
        args.runtime_dir, args.weights, args.manifest_map, args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
