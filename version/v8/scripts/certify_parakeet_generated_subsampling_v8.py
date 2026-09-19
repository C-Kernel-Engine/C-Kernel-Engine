#!/usr/bin/env python3
"""Certify generated-C Parakeet frontend and subsampling execution."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import time
from pathlib import Path

import numpy as np

import certify_parakeet_generated_frontend_v8 as frontend


F32P = ctypes.POINTER(ctypes.c_float)
U8P = ctypes.POINTER(ctypes.c_uint8)


def _checked_fixture(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as fixture:
        features = fixture["frontend.input_features"]
        mask = fixture["frontend.attention_mask"]
        expected = fixture["encoder.subsampling"]
    if features.dtype != np.float32 or features.ndim != 3 or features.shape[0] != 1:
        raise ValueError("frontend.input_features must be float32 [1, frames, channels]")
    if mask.dtype != np.bool_ or mask.shape != features.shape[:2]:
        raise ValueError("frontend.attention_mask must be bool [1, frames]")
    if expected.dtype != np.float32 or expected.ndim != 3 or expected.shape[0] != 1:
        raise ValueError("encoder.subsampling must be float32 [1, frames, hidden]")
    live_frames = int(mask[0].sum())
    if live_frames <= 0 or not np.all(mask[0, :live_frames]) or np.any(mask[0, live_frames:]):
        raise ValueError("frontend.attention_mask must describe one contiguous live prefix")
    return (
        np.ascontiguousarray(features[0]),
        np.ascontiguousarray(expected[0]),
        live_frames,
    )


def _validate_bundle_outputs(runtime_dir: Path, bundle_name: str) -> dict[str, object]:
    path = runtime_dir / bundle_name
    bundle = json.loads(path.read_text(encoding="utf-8"))
    outputs = bundle.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError(f"{bundle_name}: missing outputs")
    for name, expected in outputs.items():
        if not isinstance(expected, dict) or not expected.get("path"):
            raise ValueError(f"{bundle_name}: invalid output {name}")
        actual_path = Path(str(expected["path"]))
        if actual_path.parent.resolve() != runtime_dir.resolve():
            actual_path = runtime_dir / actual_path.name
        actual = frontend._identity(actual_path)
        if actual["bytes"] != expected.get("size") or actual["sha256"] != expected.get("sha256"):
            raise ValueError(f"stale generated runtime output: {name}")
    return bundle


def _comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | bool]:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    expected_rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
    actual_flat = actual.astype(np.float64).ravel()
    expected_flat = expected.astype(np.float64).ravel()
    denominator = float(np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    return {
        "finite": bool(np.isfinite(actual).all()),
        "rmse": rmse,
        "normalized_rmse": rmse / expected_rms if expected_rms > 0 else math.inf,
        "max_abs": float(np.max(np.abs(difference))),
        "cosine": float(np.dot(actual_flat, expected_flat) / denominator) if denominator else math.nan,
    }


def certify(runtime_dir: Path, weights: Path, manifest_map: Path,
            wav_path: Path, fixture_path: Path) -> dict[str, object]:
    runtime_dir = runtime_dir.resolve()
    codegen_bundle = _validate_bundle_outputs(runtime_dir, ".ck_codegen_bundle.json")
    runtime_bundle = _validate_bundle_outputs(runtime_dir, ".ck_runtime_bundle.json")
    expected_features, expected_subsampling, live_frames = _checked_fixture(fixture_path)
    wav = np.frombuffer(wav_path.read_bytes(), dtype=np.uint8)

    library = ctypes.CDLL(str(runtime_dir / "libmodel.so"))
    loaded = frontend._verify_loaded_libraries(library, runtime_dir)
    library.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.ck_model_init_with_manifest.restype = ctypes.c_int
    library.ck_model_free.argtypes = []
    library.ck_model_prepare_audio_wav_features.argtypes = [
        U8P, ctypes.c_size_t, F32P, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(frontend.CKAudioWavInfo),
    ]
    library.ck_model_prepare_audio_wav_features.restype = ctypes.c_int
    library.ck_model_audio_subsampling_workspace_bytes.argtypes = [ctypes.c_int]
    library.ck_model_audio_subsampling_workspace_bytes.restype = ctypes.c_size_t
    library.ck_model_run_audio_subsampling.argtypes = [
        F32P, ctypes.c_int, ctypes.c_int, F32P, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_size_t,
    ]
    library.ck_model_run_audio_subsampling.restype = ctypes.c_int

    init_status = int(library.ck_model_init_with_manifest(
        str(weights.resolve()).encode(), str(manifest_map.resolve()).encode()))
    generated_features = np.empty_like(expected_features)
    isolated = np.empty_like(expected_subsampling)
    chained = np.empty_like(expected_subsampling)
    produced_features = ctypes.c_int(-1)
    produced_isolated = ctypes.c_int(-1)
    produced_chained = ctypes.c_int(-1)
    info = frontend.CKAudioWavInfo()
    statuses = {"init": init_status, "frontend": -1, "isolated": -1, "chained": -1}
    elapsed = 0.0
    try:
        if init_status == 0:
            started = time.perf_counter()
            statuses["frontend"] = int(library.ck_model_prepare_audio_wav_features(
                wav.ctypes.data_as(U8P), wav.size, generated_features.ctypes.data_as(F32P),
                generated_features.shape[0], ctypes.byref(produced_features), ctypes.byref(info)))
            workspace_bytes = int(library.ck_model_audio_subsampling_workspace_bytes(
                expected_features.shape[0]))
            if workspace_bytes <= 0:
                raise RuntimeError("generated runtime returned an invalid subsampling workspace size")
            workspace = np.empty(workspace_bytes, dtype=np.uint8)
            statuses["isolated"] = int(library.ck_model_run_audio_subsampling(
                expected_features.ctypes.data_as(F32P), expected_features.shape[0], live_frames,
                isolated.ctypes.data_as(F32P), isolated.shape[0], ctypes.byref(produced_isolated),
                workspace.ctypes.data, workspace_bytes))
            if statuses["frontend"] == 0:
                statuses["chained"] = int(library.ck_model_run_audio_subsampling(
                    generated_features.ctypes.data_as(F32P), produced_features.value, live_frames,
                    chained.ctypes.data_as(F32P), chained.shape[0], ctypes.byref(produced_chained),
                    workspace.ctypes.data, workspace_bytes))
            elapsed = time.perf_counter() - started
    finally:
        library.ck_model_free()

    isolated_cmp = _comparison(isolated, expected_subsampling)
    chained_cmp = _comparison(chained, expected_subsampling)
    frontend_cmp = _comparison(generated_features, expected_features)
    checks = {
        "all_calls_succeeded": all(value == 0 for value in statuses.values()),
        "feature_shape_exact": produced_features.value == expected_features.shape[0],
        "subsampling_shape_exact": produced_isolated.value == expected_subsampling.shape[0]
            and produced_chained.value == expected_subsampling.shape[0],
        "isolated_rmse_within_2e_4": isolated_cmp["rmse"] <= 2.0e-4,
        "isolated_max_abs_within_3e_3": isolated_cmp["max_abs"] <= 3.0e-3,
        "chained_normalized_rmse_within_5e_4": chained_cmp["normalized_rmse"] <= 5.0e-4,
        "chained_cosine_at_least_0_999999": chained_cmp["cosine"] >= 0.999999,
    }
    return {
        "schema": "cke.parakeet.generated_subsampling_certification.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "generated_c_frontend_and_subsampling",
        "checks": checks,
        "statuses": statuses,
        "numerical": {"frontend": frontend_cmp, "isolated_subsampling": isolated_cmp,
                      "chained_frontend_subsampling": chained_cmp},
        "execution": {"elapsed_seconds": elapsed, "live_feature_frames": live_frames,
                      "subsampling_frames": produced_chained.value},
        "identity": {
            "weights": frontend._identity(weights), "manifest_map": frontend._identity(manifest_map),
            "wav": frontend._identity(wav_path), "fixture": frontend._identity(fixture_path),
            "loaded_libraries": {name: frontend._identity(path) for name, path in loaded.items()},
            "codegen_bundle": frontend._identity(runtime_dir / ".ck_codegen_bundle.json"),
            "runtime_bundle": frontend._identity(runtime_dir / ".ck_runtime_bundle.json"),
            "codegen_bundle_schema": codegen_bundle.get("schema"),
            "runtime_bundle_schema": runtime_bundle.get("schema"),
        },
        "claim_boundary": {"frontend": "certified", "subsampling": "certified",
                           "encoder": "not_generated", "tdt_decoder": "not_generated",
                           "standalone_transcription": "not_certified"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest-map", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = certify(args.runtime_dir, args.weights, args.manifest_map, args.wav, args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
