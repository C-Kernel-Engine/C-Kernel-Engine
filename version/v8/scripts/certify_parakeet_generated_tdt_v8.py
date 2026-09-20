#!/usr/bin/env python3
"""Certify generated-C Parakeet TDT decode and complete WAV transcription."""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np

import certify_parakeet_generated_frontend_v8 as frontend


U8P = ctypes.POINTER(ctypes.c_uint8)
I32P = ctypes.POINTER(ctypes.c_int32)
F32P = ctypes.POINTER(ctypes.c_float)


def _fixture(path: Path) -> dict[str, np.ndarray]:
    required = ("encoder.projected", "joint.first_logits", "decode.sequences",
                "decode.durations")
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"TDT fixture is missing {missing}")
        values = {name: np.ascontiguousarray(data[name].reshape(-1)) for name in required}
    if values["encoder.projected"].dtype != np.float32:
        raise ValueError("encoder.projected must use float32 storage")
    if values["joint.first_logits"].dtype != np.float32:
        raise ValueError("joint.first_logits must use float32 storage")
    if not np.isfinite(values["encoder.projected"]).all() or not np.isfinite(
            values["joint.first_logits"]).all():
        raise ValueError("TDT floating-point fixtures must be finite")
    for name in ("decode.sequences", "decode.durations"):
        if values[name].dtype.kind not in "iu":
            raise ValueError(f"{name} must use integer storage")
        values[name] = values[name].astype(np.int32)
    if values["decode.sequences"].shape != values["decode.durations"].shape:
        raise ValueError("TDT token and duration trajectories must align")
    return values


def _bind(library: ctypes.CDLL) -> None:
    library.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.ck_model_init_with_manifest.restype = ctypes.c_int
    library.ck_model_free.argtypes = []
    library.ck_model_audio_tdt_workspace_bytes.restype = ctypes.c_size_t
    library.ck_model_run_audio_tdt_decode.argtypes = [
        F32P, ctypes.c_int, I32P, I32P, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), F32P, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    library.ck_model_run_audio_tdt_decode.restype = ctypes.c_int
    library.ck_model_audio_transcription_workspace_bytes.argtypes = [
        U8P, ctypes.c_size_t, ctypes.c_int]
    library.ck_model_audio_transcription_workspace_bytes.restype = ctypes.c_size_t
    library.ck_model_transcribe_audio_wav.argtypes = [
        U8P, ctypes.c_size_t, I32P, I32P, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_size_t,
    ]
    library.ck_model_transcribe_audio_wav.restype = ctypes.c_int
    library.ck_model_decode_tokens.argtypes = [I32P, ctypes.c_int,
                                                ctypes.c_char_p, ctypes.c_int]
    library.ck_model_decode_tokens.restype = ctypes.c_int


def _trajectory(library: ctypes.CDLL, encoder: np.ndarray, capacity: int,
                joint_size: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    workspace_size = int(library.ck_model_audio_tdt_workspace_bytes())
    workspace = np.empty(workspace_size, dtype=np.uint8)
    tokens = np.full(capacity, -1, dtype=np.int32)
    durations = np.full(capacity, -1, dtype=np.int32)
    logits = np.empty(joint_size, dtype=np.float32)
    count = ctypes.c_int(0)
    frames = encoder.size // 640
    status = int(library.ck_model_run_audio_tdt_decode(
        encoder.ctypes.data_as(F32P), frames,
        tokens.ctypes.data_as(I32P), durations.ctypes.data_as(I32P), capacity,
        ctypes.byref(count), logits.ctypes.data_as(F32P), joint_size,
        workspace.ctypes.data, workspace_size))
    return status, tokens[:count.value].copy(), durations[:count.value].copy(), logits


def _complete(library: ctypes.CDLL, wav: np.ndarray, capacity: int,
              *, workspace_delta: int = 0) -> tuple[int, np.ndarray, np.ndarray]:
    required = int(library.ck_model_audio_transcription_workspace_bytes(
        wav.ctypes.data_as(U8P), wav.size, capacity))
    if required <= 0:
        raise RuntimeError("generated runtime rejected the WAV/capacity geometry")
    actual_size = required + workspace_delta
    workspace = np.empty(max(1, actual_size), dtype=np.uint8)
    tokens = np.full(capacity, -123456, dtype=np.int32)
    durations = np.full(capacity, -123456, dtype=np.int32)
    count = ctypes.c_int(0)
    status = int(library.ck_model_transcribe_audio_wav(
        wav.ctypes.data_as(U8P), wav.size,
        tokens.ctypes.data_as(I32P), durations.ctypes.data_as(I32P), capacity,
        ctypes.byref(count), workspace.ctypes.data, actual_size))
    return status, tokens[:max(0, count.value)].copy(), durations[:max(0, count.value)].copy()


def certify(runtime_dir: Path, weights: Path, manifest_map: Path,
            fixture_path: Path, wav_path: Path) -> dict[str, object]:
    expected = _fixture(fixture_path)
    wav = np.frombuffer(wav_path.read_bytes(), dtype=np.uint8)
    runtime_dir = runtime_dir.resolve()
    ctypes.CDLL(str(runtime_dir / "libckernel_engine.so"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(runtime_dir / "libckernel_tokenizer.so"), mode=ctypes.RTLD_GLOBAL)
    library = ctypes.CDLL(str(runtime_dir / "libmodel.so"))
    loaded = frontend._verify_loaded_libraries(library, runtime_dir)
    _bind(library)
    init = int(library.ck_model_init_with_manifest(
        str(weights.resolve()).encode(), str(manifest_map.resolve()).encode()))
    statuses: dict[str, int] = {"init": init}
    started = time.perf_counter()
    try:
        if init != 0:
            raise RuntimeError(f"generated model initialization failed: {init}")
        expected_tokens = expected["decode.sequences"]
        expected_durations = expected["decode.durations"]
        capacity = max(1000, int(expected_tokens.size) + 1)
        isolated = _trajectory(
            library, expected["encoder.projected"], capacity,
            expected["joint.first_logits"].size)
        statuses["isolated_tdt"] = isolated[0]
        complete = _complete(library, wav, capacity)
        repeat = _complete(library, wav, capacity)
        undersized = _complete(library, wav, capacity, workspace_delta=-1)
        statuses["complete"] = complete[0]
        statuses["repeat"] = repeat[0]
        statuses["undersized_workspace"] = undersized[0]
        logits_difference = isolated[3].astype(np.float64) - expected[
            "joint.first_logits"].astype(np.float64)
        transcript_buffer = ctypes.create_string_buffer(8192)
        decoded = complete[1][
            (complete[1] != np.int32(8192)) & (complete[1] != np.int32(2))]
        decoded_length = int(library.ck_model_decode_tokens(
            decoded.ctypes.data_as(I32P), decoded.size, transcript_buffer,
            len(transcript_buffer)))
        checks = {
            "isolated_tdt_succeeded": isolated[0] == 0,
            "isolated_tokens_exact": bool(np.array_equal(isolated[1], expected_tokens)),
            "isolated_durations_exact": bool(np.array_equal(isolated[2], expected_durations)),
            "first_logits_rmse_within_1e_5": float(np.sqrt(np.mean(
                logits_difference * logits_difference))) <= 1.0e-5,
            "complete_transcription_succeeded": complete[0] == 0,
            "complete_tokens_exact": bool(np.array_equal(complete[1], expected_tokens)),
            "complete_durations_exact": bool(np.array_equal(
                complete[2], expected_durations)),
            "repeat_trajectory_exact": bool(
                repeat[0] == 0 and np.array_equal(repeat[1], complete[1])
                and np.array_equal(repeat[2], complete[2])),
            "undersized_workspace_rejected_before_output": bool(
                undersized[0] != 0 and undersized[1].size == 0 and undersized[2].size == 0),
            "native_token_decode_nonempty": decoded_length > 0,
        }
        elapsed = time.perf_counter() - started
    finally:
        library.ck_model_free()
    return {
        "schema": "cke.parakeet.generated_tdt_certification.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "generated_c_tdt_and_single_window_transcription",
        "checks": checks,
        "statuses": statuses,
        "execution": {"elapsed_seconds": elapsed, "tokens": int(complete[1].size)},
        "transcript": transcript_buffer.value.decode("utf-8", errors="strict"),
        "identity": {
            "weights": frontend._identity(weights),
            "manifest_map": frontend._identity(manifest_map),
            "fixture": frontend._identity(fixture_path),
            "wav": frontend._identity(wav_path),
            "loaded_libraries": {
                name: frontend._identity(path) for name, path in loaded.items()
            },
        },
        "claim_boundary": {
            "generated_encoder": "certified",
            "generated_tdt": "certified",
            "single_window_transcription": "certified",
            "bounded_long_audio": "requires_native_host_run",
            "cohere_transcribe": "not_certified",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest-map", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = certify(args.runtime_dir, args.weights, args.manifest_map,
                     args.fixture, args.wav)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
