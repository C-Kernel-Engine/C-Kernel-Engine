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


def _checked_fixture(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as fixture:
        frontend_features = fixture["frontend.input_features"]
        frontend_mask = fixture["frontend.attention_mask"]
        first_input = fixture["encoder.subsampling"]
        first_expected = fixture["encoder.layer.0"]
        final_expected = fixture["encoder.layer.23"]
        projected_expected = fixture["encoder.projected"]
    arrays = (
        frontend_features,
        first_input,
        first_expected,
        final_expected,
        projected_expected,
    )
    if any(array.dtype != np.float32 for array in arrays):
        raise ValueError("encoder block fixtures must use float32 storage")
    if any(array.ndim != 3 or array.shape[0] != 1 for array in arrays):
        raise ValueError("encoder block fixtures must have shape [1, frames, hidden]")
    if len({array.shape for array in arrays[1:4]}) != 1:
        raise ValueError("encoder block fixture shapes must agree")
    if (projected_expected.shape[1] != first_input.shape[1]
            or projected_expected.shape[2] <= 0):
        raise ValueError("encoder projection fixture must preserve frames")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("encoder block fixtures must contain only finite values")
    if frontend_mask.dtype != np.bool_ or frontend_mask.shape != frontend_features.shape[:2]:
        raise ValueError("frontend attention mask must be boolean and match feature frames")
    live_frames = int(np.count_nonzero(frontend_mask[0]))
    if live_frames <= 0 or not np.all(frontend_mask[0, :live_frames]) or np.any(
            frontend_mask[0, live_frames:]):
        raise ValueError("frontend attention mask must contain one contiguous live prefix")
    contiguous = tuple(np.ascontiguousarray(array[0]) for array in arrays)
    return (*contiguous, live_frames)


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
    (frontend_features, first_input, first_expected, final_expected,
     projected_expected, live_frames) = _checked_fixture(fixture_path)
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
    library.ck_model_run_audio_encoder_projection.argtypes = [F32P, ctypes.c_int, F32P]
    library.ck_model_run_audio_encoder_projection.restype = ctypes.c_int
    library.ck_model_audio_encoder_workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    library.ck_model_audio_encoder_workspace_bytes.restype = ctypes.c_size_t
    library.ck_model_run_audio_encoder.argtypes = [
        F32P, ctypes.c_int, ctypes.c_int, F32P, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_size_t,
    ]
    library.ck_model_run_audio_encoder.restype = ctypes.c_int

    init_status = int(library.ck_model_init_with_manifest(
        str(weights.resolve()).encode(), str(manifest_map.resolve()).encode()))
    relative = np.empty((2 * frames - 1, hidden_size), dtype=np.float32)
    current = first_input.copy()
    next_output = np.empty_like(current)
    projected_actual = np.empty_like(projected_expected)
    encoder_actual = np.empty_like(projected_expected)
    statuses = {
        "init": init_status,
        "relative_position": -1,
        "layers": [],
        "encoder_projection": -1,
        "native_encoder": -1,
        "native_encoder_undersized_workspace": -1,
    }
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
                if len(statuses["layers"]) == 24 and all(
                        status == 0 for status in statuses["layers"]):
                    statuses["encoder_projection"] = int(
                        library.ck_model_run_audio_encoder_projection(
                            final_expected.ctypes.data_as(F32P), frames,
                            projected_actual.ctypes.data_as(F32P)))
            encoder_workspace_bytes = int(
                library.ck_model_audio_encoder_workspace_bytes(
                    frontend_features.shape[0], projected_expected.shape[0]))
            if encoder_workspace_bytes <= 0:
                raise RuntimeError("generated runtime returned invalid encoder workspace size")
            encoder_workspace = np.empty(encoder_workspace_bytes, dtype=np.uint8)
            encoder_frames = ctypes.c_int(0)
            rejected_output = np.full_like(projected_expected, 1234.5)
            statuses["native_encoder_undersized_workspace"] = int(
                library.ck_model_run_audio_encoder(
                    frontend_features.ctypes.data_as(F32P),
                    frontend_features.shape[0], live_frames,
                    rejected_output.ctypes.data_as(F32P), projected_expected.shape[0],
                    ctypes.byref(encoder_frames), encoder_workspace.ctypes.data,
                    encoder_workspace_bytes - 1))
            statuses["undersized_workspace_preserved_output"] = bool(
                np.all(rejected_output == np.float32(1234.5)))
            statuses["native_encoder"] = int(library.ck_model_run_audio_encoder(
                frontend_features.ctypes.data_as(F32P),
                frontend_features.shape[0], live_frames,
                encoder_actual.ctypes.data_as(F32P), projected_expected.shape[0],
                ctypes.byref(encoder_frames), encoder_workspace.ctypes.data,
                encoder_workspace_bytes))
            statuses["native_encoder_frames"] = int(encoder_frames.value)
            elapsed = time.perf_counter() - started
    finally:
        library.ck_model_free()

    first_cmp = _comparison(
        first_actual if first_actual is not None else next_output, first_expected)
    final_cmp = _comparison(current, final_expected)
    projected_cmp = _comparison(projected_actual, projected_expected)
    native_encoder_cmp = _comparison(encoder_actual, projected_expected)
    checks = {
        "all_calls_succeeded": init_status == 0
        and statuses["relative_position"] == 0
        and len(statuses["layers"]) == 24
        and all(status == 0 for status in statuses["layers"])
        and statuses["encoder_projection"] == 0,
        "native_encoder_succeeded": statuses["native_encoder"] == 0
        and statuses.get("native_encoder_frames") == projected_expected.shape[0],
        "native_encoder_rejects_undersized_workspace":
        statuses["native_encoder_undersized_workspace"] != 0
        and statuses["undersized_workspace_preserved_output"],
        "first_layer_rmse_within_1e_5": first_cmp["rmse"] <= 1.0e-5,
        "first_layer_max_abs_within_1e_4": first_cmp["max_abs"] <= 1.0e-4,
        "final_layer_rmse_within_1e_5": final_cmp["rmse"] <= 1.0e-5,
        "final_layer_max_abs_within_1e_4": final_cmp["max_abs"] <= 1.0e-4,
        "encoder_projection_rmse_within_1e_5": projected_cmp["rmse"] <= 1.0e-5,
        "encoder_projection_max_abs_within_1e_4": projected_cmp["max_abs"] <= 1.0e-4,
        "native_encoder_rmse_within_1e_5": native_encoder_cmp["rmse"] <= 1.0e-5,
        "native_encoder_max_abs_within_1e_4": native_encoder_cmp["max_abs"] <= 1.0e-4,
    }
    return {
        "schema": "cke.parakeet.generated_encoder_certification.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "generated_c_fastconformer_24_block_encoder_and_projection",
        "checks": checks,
        "statuses": statuses,
        "numerical": {
            "layer_0": first_cmp,
            "layer_23": final_cmp,
            "encoder_projection": projected_cmp,
            "native_encoder": native_encoder_cmp,
        },
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
            "encoder_projection": "certified",
            "native_encoder_schedule": "certified",
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
