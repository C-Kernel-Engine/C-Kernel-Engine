#!/usr/bin/env python3
"""Run generated CKE Whisper encoder and decoder artifacts on a PCM16 WAV."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import wave
from typing import Any

import numpy as np


_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_U8_P = ctypes.POINTER(ctypes.c_uint8)
_FRONTEND_REUSE_UNAVAILABLE = 78
_FRONTEND_REUSE_RESOURCE_ERROR = 79


class CKAudioWavInfo(ctypes.Structure):
    _fields_ = [
        ("format_tag", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("sample_rate", ctypes.c_int),
        ("bits_per_sample", ctypes.c_int),
        ("frames", ctypes.c_int),
        ("data_offset", ctypes.c_size_t),
        ("data_bytes", ctypes.c_size_t),
    ]


def _fptr(values: np.ndarray) -> _FLOAT_P:
    return values.ctypes.data_as(_FLOAT_P)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_geometry_for_cache(path: Path) -> tuple[int, int] | None:
    try:
        with wave.open(str(path), "rb") as wav_reader:
            return wav_reader.getnframes(), wav_reader.getframerate()
    except (EOFError, OSError, wave.Error):
        return None


def _probe_cache_capacity(directory: Path, required_bytes: int) -> str | None:
    if required_bytes <= 0:
        return "invalid cache-capacity request"
    probe = directory / ".cke-whisper-capacity-probe"
    try:
        with probe.open("w+b") as handle:
            os.posix_fallocate(handle.fileno(), 0, required_bytes)
    except OSError as error:
        if error.errno in (errno.EDQUOT, errno.ENOSPC, errno.EFBIG):
            return f"temporary storage cannot reserve {required_bytes} bytes: {error}"
        raise
    finally:
        probe.unlink(missing_ok=True)
    return None


def _frontend_reuse_worker_failure(returncode: int) -> str | None:
    if returncode == 0:
        return None
    if returncode == _FRONTEND_REUSE_UNAVAILABLE:
        return (
            "generated encoder runtime does not provide full-feature reuse; "
            "rebuild to enable it"
        )
    if returncode == _FRONTEND_REUSE_RESOURCE_ERROR:
        return "temporary feature cache could not be stored"
    raise ValueError(f"unexpected frontend worker return code: {returncode}")


def _frontend_cache_required_bytes(
    encoder_config: dict[str, Any], feature_frames: int
) -> int:
    feature_bytes = (
        int(encoder_config["audio_feature_channels"]) * feature_frames * 4
    )
    encoder_output_bytes = (
        int(encoder_config["context_length"])
        * int(encoder_config["embed_dim"])
        * 4
    )
    return feature_bytes + encoder_output_bytes + 8 * 1024 * 1024


def _parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.strip().split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first > last:
                raise ValueError(f"invalid CPU range: {item}")
            cpus.update(range(first, last + 1))
        else:
            cpus.add(int(item))
    return cpus


def _hybrid_performance_cpus(
    allowed: set[int],
    *,
    sysfs_root: Path = Path("/sys/devices/system/cpu"),
) -> list[int] | None:
    """Return SMT-capable cores only when SMT and singleton cores coexist."""
    groups: dict[frozenset[int], set[int]] = {}
    for cpu in sorted(allowed):
        siblings_path = sysfs_root / f"cpu{cpu}" / "topology" / "thread_siblings_list"
        try:
            siblings = _parse_cpu_list(siblings_path.read_text(encoding="ascii"))
        except (FileNotFoundError, OSError, UnicodeError, ValueError):
            return None
        visible = siblings & allowed
        if not visible:
            return None
        groups.setdefault(frozenset(visible), set()).update(visible)

    widths = {len(group) for group in groups}
    if 1 not in widths or not any(width > 1 for width in widths):
        return None
    selected = sorted(
        cpu for group in groups.values() if len(group) > 1 for cpu in group
    )
    return selected or None


def _worker_environment(
    encoder_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Keep NumPy's idle BLAS pool from competing with CKE worker threads."""
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    policy = str(
        (encoder_config or {}).get("audio_runtime_topology_policy") or ""
    )
    if (
        policy == "performance_core_smt_on_hybrid"
        and hasattr(os, "sched_getaffinity")
    ):
        selected = _hybrid_performance_cpus(set(os.sched_getaffinity(0)))
        if selected:
            env["CK_AUDIO_WORKER_CPUS"] = ",".join(str(cpu) for cpu in selected)
            if "CK_NUM_THREADS" not in os.environ:
                env["CK_NUM_THREADS"] = str(len(selected))
    return env


def _apply_worker_affinity() -> dict[str, Any]:
    requested = os.environ.get("CK_AUDIO_WORKER_CPUS", "").strip()
    if not requested:
        return {"policy": "inherited", "cpus": None}
    cpus = _parse_cpu_list(requested)
    if not cpus or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("requested audio CPU affinity is unsupported")
    os.sched_setaffinity(0, cpus)
    return {
        "policy": "performance_core_smt_on_hybrid",
        "cpus": sorted(os.sched_getaffinity(0)),
        "threads": int(os.environ.get("CK_NUM_THREADS", len(cpus))),
    }


def _require_artifact(run_dir: Path) -> None:
    for name in (
        "libckernel_engine.so",
        "libmodel.so",
        "weights.bump",
        "weights_manifest.map",
        "config.json",
    ):
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)


def _load_generated_model(run_dir: Path) -> ctypes.CDLL:
    ctypes.CDLL(str(run_dir / "libckernel_engine.so"), mode=ctypes.RTLD_GLOBAL)
    model = ctypes.CDLL(str(run_dir / "libmodel.so"))
    model.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    model.ck_model_init_with_manifest.restype = ctypes.c_int
    model.ck_model_free.argtypes = []
    model.ck_model_free.restype = None
    return model


def _initialize_model(model: ctypes.CDLL, run_dir: Path, role: str) -> float:
    started = time.perf_counter()
    status = int(
        model.ck_model_init_with_manifest(
            str(run_dir / "weights.bump").encode(),
            str(run_dir / "weights_manifest.map").encode(),
        )
    )
    elapsed = time.perf_counter() - started
    if status != 0:
        raise RuntimeError(f"{role} initialization failed with code {status}")
    return elapsed


def _open_encoder_model(run_dir: Path) -> tuple[ctypes.CDLL, float]:
    model = _load_generated_model(run_dir)
    model.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
    model.ck_model_get_named_activation_ptr.restype = ctypes.c_void_p
    model.ck_model_get_named_activation_nbytes.argtypes = [ctypes.c_char_p]
    model.ck_model_get_named_activation_nbytes.restype = ctypes.c_ssize_t
    model.ck_model_prepare_audio_wav_window.argtypes = [
        _U8_P,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(CKAudioWavInfo),
    ]
    model.ck_model_prepare_audio_wav_window.restype = ctypes.c_int
    model.ck_model_run_encoder.argtypes = []
    model.ck_model_run_encoder.restype = ctypes.c_int
    return model, _initialize_model(model, run_dir, "encoder")


def _open_decoder_model(run_dir: Path) -> tuple[ctypes.CDLL, float]:
    model = _load_generated_model(run_dir)
    model.ck_model_set_encoder_memory.argtypes = [
        _FLOAT_P,
        ctypes.c_int,
        ctypes.c_int,
    ]
    model.ck_model_set_encoder_memory.restype = ctypes.c_int
    model.ck_model_embed_tokens.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
    ]
    model.ck_model_embed_tokens.restype = ctypes.c_int
    model.ck_model_decode.argtypes = [ctypes.c_int32, _FLOAT_P]
    model.ck_model_decode.restype = ctypes.c_int
    model.ck_model_get_logits.argtypes = []
    model.ck_model_get_logits.restype = _FLOAT_P
    model.ck_model_get_vocab_size.argtypes = []
    model.ck_model_get_vocab_size.restype = ctypes.c_int
    try:
        reset = model.ck_model_kv_cache_reset
    except AttributeError as error:
        raise RuntimeError(
            "decoder runtime lacks ck_model_kv_cache_reset; rebuild before "
            "using persistent workers"
        ) from error
    reset.argtypes = []
    reset.restype = None
    return model, _initialize_model(model, run_dir, "decoder")


def _run_audio_encoder_window(
    model: ctypes.CDLL,
    wav: np.ndarray,
    window_start_frame: int,
    info: CKAudioWavInfo,
) -> tuple[float, float]:
    started = time.perf_counter()
    status = int(
        model.ck_model_prepare_audio_wav_window(
            wav.ctypes.data_as(_U8_P),
            wav.size,
            window_start_frame,
            ctypes.byref(info),
        )
    )
    frontend_seconds = time.perf_counter() - started
    if status != 0:
        raise RuntimeError(f"generated audio frontend failed with code {status}")

    started = time.perf_counter()
    status = int(model.ck_model_run_encoder())
    encoder_seconds = time.perf_counter() - started
    if status != 0:
        raise RuntimeError(f"generated audio encoder failed with code {status}")
    return frontend_seconds, encoder_seconds


def _copy_cached_feature_window(
    full_features: np.ndarray,
    output: np.ndarray,
    *,
    window_start_frame: int,
    hop_length: int,
) -> int:
    if full_features.ndim != 2 or output.ndim != 2:
        raise ValueError("cached audio features must be two-dimensional")
    if full_features.shape[0] != output.shape[0]:
        raise ValueError("cached audio feature channel count does not match output")
    if hop_length <= 0 or window_start_frame < 0:
        raise ValueError("cached audio window geometry is invalid")
    if window_start_frame % hop_length != 0:
        raise ValueError("cached audio window start is not hop-aligned")
    output.fill(0.0)
    start_feature = window_start_frame // hop_length
    valid_features = min(
        output.shape[1],
        max(0, full_features.shape[1] - start_feature),
    )
    if valid_features:
        output[:, :valid_features] = full_features[
            :, start_feature : start_feature + valid_features
        ]
    return valid_features


def _frontend_worker(args: argparse.Namespace) -> int:
    execution_topology = _apply_worker_affinity()
    run_dir = args.encoder_run_dir.resolve()
    _require_artifact(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    feature_channels = int(config["audio_feature_channels"])
    feature_frames = int(args.feature_frames)
    model = _load_generated_model(run_dir)
    try:
        prepare_features = model.ck_model_prepare_audio_wav_features
    except AttributeError:
        print(
            "frontend reuse unavailable: generated encoder runtime lacks "
            "ck_model_prepare_audio_wav_features; rebuild to enable reuse",
            file=sys.stderr,
        )
        return _FRONTEND_REUSE_UNAVAILABLE
    prepare_features.argtypes = [
        _U8_P,
        ctypes.c_size_t,
        _FLOAT_P,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(CKAudioWavInfo),
    ]
    prepare_features.restype = ctypes.c_int
    features = np.empty((feature_channels, feature_frames), dtype=np.float32)
    status = int(
        model.ck_model_init_with_manifest(
            str(run_dir / "weights.bump").encode(),
            str(run_dir / "weights_manifest.map").encode(),
        )
    )
    if status != 0:
        raise RuntimeError(f"frontend initialization failed with code {status}")
    try:
        wav = np.frombuffer(args.wav.resolve().read_bytes(), dtype=np.uint8)
        info = CKAudioWavInfo()
        produced = ctypes.c_int()
        started = time.perf_counter()
        status = int(
            prepare_features(
                wav.ctypes.data_as(_U8_P),
                wav.size,
                _fptr(features),
                feature_frames,
                ctypes.byref(produced),
                ctypes.byref(info),
            )
        )
        frontend_seconds = time.perf_counter() - started
        if status != 0:
            raise RuntimeError(f"full audio frontend failed with code {status}")
        if produced.value != feature_frames:
            raise RuntimeError(
                "full audio frontend frame mismatch: "
                f"expected={feature_frames} actual={produced.value}"
            )
    finally:
        model.ck_model_free()

    try:
        np.save(args.feature_output, features)
    except OSError as error:
        args.feature_output.unlink(missing_ok=True)
        print(
            f"frontend reuse unavailable: cannot store feature cache: {error}",
            file=sys.stderr,
        )
        return _FRONTEND_REUSE_RESOURCE_ERROR
    args.worker_report.write_text(
        json.dumps(
            {
                "audio": {
                    "source_sample_rate": info.sample_rate,
                    "source_channels": info.channels,
                    "source_frames": info.frames,
                },
                "features_shape": list(features.shape),
                "frontend_seconds": frontend_seconds,
                "feature_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                "execution_topology": execution_topology,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _encoder_worker(
    args: argparse.Namespace,
    *,
    model: ctypes.CDLL | None = None,
    execution_topology: dict[str, Any] | None = None,
    feature_cache: dict[Path, np.ndarray] | None = None,
) -> int:
    if (args.full_features is None) != (args.frontend_report is None):
        raise ValueError(
            "--full-features and --frontend-report must be provided together"
        )
    if execution_topology is None:
        execution_topology = _apply_worker_affinity()
    run_dir = args.encoder_run_dir.resolve()
    _require_artifact(run_dir)
    owns_model = model is None
    init_seconds = 0.0
    if model is None:
        model, init_seconds = _open_encoder_model(run_dir)
    try:
        info = CKAudioWavInfo()
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        target_rate = int(config["audio_sample_rate"])
        sample_extent = int(config["audio_sample_extent"])
        hop_length = int(config["audio_hop_length"])
        feature_channels = int(config["audio_feature_channels"])
        feature_frames = int(config["audio_feature_frames"])
        feature_ptr = int(
            model.ck_model_get_named_activation_ptr(b"audio_features") or 0
        )
        feature_bytes = int(
            model.ck_model_get_named_activation_nbytes(b"audio_features")
        )
        feature_required = feature_channels * feature_frames * 4
        if feature_ptr == 0 or feature_bytes < feature_required:
            raise RuntimeError("generated audio feature checkpoint is unavailable")
        if args.full_features is not None:
            started = time.perf_counter()
            feature_path = args.full_features.resolve()
            if feature_cache is not None and feature_path in feature_cache:
                full_features = feature_cache[feature_path]
            else:
                full_features = np.load(feature_path, mmap_mode="r")
                if feature_cache is not None:
                    feature_cache[feature_path] = full_features
            frontend_report = json.loads(
                args.frontend_report.read_text(encoding="utf-8")
            )
            audio = frontend_report["audio"]
            info.sample_rate = int(audio["source_sample_rate"])
            info.channels = int(audio["source_channels"])
            info.frames = int(audio["source_frames"])
            if info.sample_rate != target_rate:
                raise RuntimeError(
                    "cached long-audio features require source sample rate "
                    f"{target_rate}; source is {info.sample_rate}"
                )
            expected_shape = (feature_channels, info.frames // hop_length)
            if tuple(full_features.shape) != expected_shape:
                raise RuntimeError(
                    "cached audio feature shape mismatch: "
                    f"expected={expected_shape} actual={tuple(full_features.shape)}"
                )
            feature_view = np.ctypeslib.as_array(
                ctypes.cast(feature_ptr, _FLOAT_P),
                shape=(feature_channels * feature_frames,),
            ).reshape(feature_channels, feature_frames)
            _copy_cached_feature_window(
                full_features,
                feature_view,
                window_start_frame=args.window_start_frame,
                hop_length=hop_length,
            )
            frontend_seconds = time.perf_counter() - started
            started = time.perf_counter()
            status = int(model.ck_model_run_encoder())
            encoder_seconds = time.perf_counter() - started
            if status != 0:
                raise RuntimeError(f"generated audio encoder failed with code {status}")
        else:
            wav = np.frombuffer(args.wav.resolve().read_bytes(), dtype=np.uint8)
            frontend_seconds, encoder_seconds = _run_audio_encoder_window(
                model,
                wav,
                args.window_start_frame,
                info,
            )
        if args.window_start_frame and info.sample_rate != target_rate:
            raise RuntimeError(
                "long-audio windowing currently requires source sample rate "
                f"{target_rate}; source is {info.sample_rate}"
            )
        window_source_capacity = (
            sample_extent
            if info.sample_rate == target_rate
            else int(np.ceil(sample_extent * info.sample_rate / target_rate))
        )
        window_source_frames = min(
            window_source_capacity,
            info.frames - args.window_start_frame,
        )
        features = np.ctypeslib.as_array(
            ctypes.cast(feature_ptr, _FLOAT_P),
            shape=(feature_channels * feature_frames,),
        ).copy().reshape(feature_channels, feature_frames)
        tokens = int(config["context_length"])
        embed = int(config["embed_dim"])
        output_ptr = int(
            model.ck_model_get_named_activation_ptr(b"embedded_input") or 0
        )
        output_bytes = int(
            model.ck_model_get_named_activation_nbytes(b"embedded_input")
        )
        required = tokens * embed * np.dtype(np.float32).itemsize
        if output_ptr == 0 or output_bytes < required:
            raise RuntimeError(
                "encoder output ABI mismatch: "
                f"ptr={output_ptr} bytes={output_bytes} required={required}"
            )
        output = np.ctypeslib.as_array(
            ctypes.cast(output_ptr, _FLOAT_P), shape=(tokens * embed,)
        ).copy().reshape(tokens, embed)
    finally:
        if owns_model:
            model.ck_model_free()

    np.save(args.encoder_output, output)
    if args.feature_output is not None:
        np.save(args.feature_output, features)
    args.worker_report.write_text(
        json.dumps(
            {
                "audio": {
                    "source_sample_rate": info.sample_rate,
                    "source_channels": info.channels,
                    "source_frames": info.frames,
                    "window_start_frame": args.window_start_frame,
                    "window_source_frames": window_source_frames,
                },
                "features_shape": list(features.shape),
                "encoder_shape": list(output.shape),
                "frontend_seconds": frontend_seconds,
                "audio_encoder_seconds": frontend_seconds + encoder_seconds,
                "encoder_seconds": encoder_seconds,
                "model_init_seconds": init_seconds,
                "feature_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                "encoder_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                "execution_topology": execution_topology,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def forced_decoder_prefix(
    generation: dict[str, Any],
    language: str,
    task: str,
    *,
    timestamps: bool = False,
) -> list[int]:
    start = int(generation["decoder_start_token_id"])
    language_token = generation.get("lang_to_id", {}).get(f"<|{language}|>")
    task_token = generation.get("task_to_id", {}).get(task)
    no_timestamps = generation.get("no_timestamps_token_id")
    if language_token is None:
        raise ValueError(f"unsupported Whisper language: {language}")
    if task_token is None:
        raise ValueError(f"unsupported Whisper task: {task}")
    if no_timestamps is None:
        raise ValueError("generation_config.json has no no_timestamps_token_id")
    prefix = [start, int(language_token), int(task_token)]
    if not timestamps:
        prefix.append(int(no_timestamps))
    return prefix


def plan_audio_windows(
    source_frames: int,
    source_sample_rate: int,
    target_sample_rate: int,
    target_window_frames: int,
) -> list[tuple[int, int]]:
    if min(
        source_frames,
        source_sample_rate,
        target_sample_rate,
        target_window_frames,
    ) <= 0:
        raise ValueError("audio window geometry must be positive")
    source_window_frames = int(
        np.ceil(
            target_window_frames
            * source_sample_rate
            / target_sample_rate
        )
    )
    if (
        source_frames > source_window_frames
        and source_sample_rate != target_sample_rate
    ):
        raise ValueError(
            "long-audio windowing requires source and target sample rates "
            "to match until globally phased resampling is certified"
        )
    return [
        (start, min(source_frames, start + source_window_frames))
        for start in range(0, source_frames, source_window_frames)
    ]


def global_timestamp_events(
    tokens: list[int],
    generation: dict[str, Any],
    offset_seconds: float,
) -> list[dict[str, float | int]]:
    timestamp_begin = int(generation["no_timestamps_token_id"]) + 1
    return [
        {
            "token_id": token,
            "local_seconds": (token - timestamp_begin) * 0.02,
            "global_seconds": offset_seconds
            + (token - timestamp_begin) * 0.02,
        }
        for token in tokens
        if token >= timestamp_begin
    ]


def timestamp_seek_consumed_frames(
    tokens: list[int],
    generation: dict[str, Any],
    source_rate: int,
    window_frames: int,
) -> int:
    """Mirror Whisper long-form seek advancement from timestamp boundaries."""
    timestamp_begin = int(generation["no_timestamps_token_id"]) + 1
    is_timestamp = [token >= timestamp_begin for token in tokens]
    consecutive = [
        index
        for index in range(len(tokens) - 1)
        if is_timestamp[index] and is_timestamp[index + 1]
    ]
    single_timestamp_ending = (
        len(tokens) >= 2
        and not is_timestamp[-2]
        and is_timestamp[-1]
    )
    if not consecutive or single_timestamp_ending:
        return window_frames
    local_seconds = (tokens[consecutive[-1]] - timestamp_begin) * 0.02
    consumed = int(round(local_seconds * source_rate))
    return consumed if 0 < consumed <= window_frames else window_frames


def consume_timestamp_sized_tail(
    consumed_end_frame: int,
    source_frames: int,
    source_rate: int,
) -> int:
    """Avoid a padded 30-second decode for at most 100 ms of trailing audio."""
    timestamp_tail_frames = max(1, source_rate // 10)
    remaining_frames = source_frames - consumed_end_frame
    if 0 < remaining_frames <= timestamp_tail_frames:
        return source_frames
    return consumed_end_frame


def apply_timestamp_logits_contract(
    logits: np.ndarray,
    generated_tokens: list[int],
    generation: dict[str, Any],
) -> np.ndarray:
    """Apply Whisper's timestamp sequence and probability constraints."""
    scores = logits.copy()
    no_timestamps = int(generation["no_timestamps_token_id"])
    timestamp_begin = no_timestamps + 1
    eos = int(generation["eos_token_id"])
    scores[no_timestamps] = -np.inf

    last_was_timestamp = (
        bool(generated_tokens) and generated_tokens[-1] >= timestamp_begin
    )
    penultimate_was_timestamp = (
        len(generated_tokens) < 2
        or generated_tokens[-2] >= timestamp_begin
    )
    if last_was_timestamp:
        if penultimate_was_timestamp:
            scores[timestamp_begin:] = -np.inf
        else:
            scores[:eos] = -np.inf

    timestamps = [
        token for token in generated_tokens if token >= timestamp_begin
    ]
    if timestamps:
        timestamp_last = timestamps[-1]
        if not (last_was_timestamp and not penultimate_was_timestamp):
            timestamp_last += 1
        scores[timestamp_begin:timestamp_last] = -np.inf

    if not generated_tokens:
        scores[:timestamp_begin] = -np.inf
        max_initial = generation.get("max_initial_timestamp_index")
        if max_initial is not None:
            last_allowed = timestamp_begin + int(max_initial)
            scores[last_allowed + 1 :] = -np.inf

    timestamp_scores = scores[timestamp_begin:]
    finite_timestamps = timestamp_scores[np.isfinite(timestamp_scores)]
    text_scores = scores[:timestamp_begin]
    finite_text = text_scores[np.isfinite(text_scores)]
    if finite_timestamps.size and finite_text.size:
        maximum = float(np.max(finite_timestamps))
        timestamp_logsumexp = maximum + float(
            np.log(np.exp(finite_timestamps - maximum).sum())
        )
        if timestamp_logsumexp > float(np.max(finite_text)):
            scores[:timestamp_begin] = -np.inf
    return scores


def _decoder_worker(
    args: argparse.Namespace,
    *,
    model: ctypes.CDLL | None = None,
    execution_topology: dict[str, Any] | None = None,
    tokenizer: Any | None = None,
) -> int:
    if execution_topology is None:
        execution_topology = _apply_worker_affinity()
    run_dir = args.decoder_run_dir.resolve()
    _require_artifact(run_dir)
    generation_path = run_dir / "generation_config.json"
    tokenizer_path = run_dir / "tokenizer.json"
    for path in (generation_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    encoder_memory = np.load(args.encoder_output).astype(np.float32, copy=False)

    owns_model = model is None
    init_seconds = 0.0
    if model is None:
        model, init_seconds = _open_decoder_model(run_dir)
    try:
        # Each audio window is an independent decoder request. Retain immutable
        # weights and prepared constants, but never retain generated state.
        model.ck_model_kv_cache_reset()
        status = int(
            model.ck_model_set_encoder_memory(
                _fptr(encoder_memory),
                encoder_memory.shape[0],
                encoder_memory.shape[1],
            )
        )
        if status != 0:
            raise RuntimeError(f"encoder-memory binding failed with code {status}")
        prefix = forced_decoder_prefix(
            generation,
            args.language,
            args.task,
            timestamps=args.timestamps,
        )
        prefix_array = (ctypes.c_int32 * len(prefix))(*prefix)
        started = time.perf_counter()
        status = int(model.ck_model_embed_tokens(prefix_array, len(prefix)))
        prefill_seconds = time.perf_counter() - started
        if status != 0:
            raise RuntimeError(f"decoder prefill failed with code {status}")

        vocab_size = int(model.ck_model_get_vocab_size())
        suppress = np.asarray(generation.get("suppress_tokens", []), dtype=np.int64)
        begin_suppress = np.asarray(
            generation.get("begin_suppress_tokens", []), dtype=np.int64
        )
        no_timestamps = int(generation["no_timestamps_token_id"])
        eos = int(generation["eos_token_id"])
        tokens: list[int] = []
        decode_started = time.perf_counter()
        stop = "max_tokens"
        for step in range(args.max_tokens):
            logits = np.ctypeslib.as_array(
                model.ck_model_get_logits(), shape=(vocab_size,)
            ).copy()
            logits[suppress] = -np.inf
            if step == 0:
                logits[begin_suppress] = -np.inf
            if args.timestamps:
                logits = apply_timestamp_logits_contract(
                    logits, tokens, generation
                )
            else:
                logits[no_timestamps:] = -np.inf
            token = int(np.argmax(logits))
            if token == eos:
                stop = "eos"
                break
            tokens.append(token)
            status = int(model.ck_model_decode(token, None))
            if status != 0:
                raise RuntimeError(
                    f"decoder step {step} failed with code {status}"
                )
        decode_seconds = time.perf_counter() - decode_started
    finally:
        if owns_model:
            model.ck_model_free()

    if tokenizer is None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    timestamp_begin = int(generation["no_timestamps_token_id"]) + 1
    transcript_tokens = [
        token for token in tokens if token < timestamp_begin
    ]
    transcript_text = tokenizer.decode(
        transcript_tokens,
        skip_special_tokens=True,
    )
    args.worker_report.write_text(
        json.dumps(
            {
                "forced_prefix": prefix,
                "timestamps": bool(args.timestamps),
                "generated_tokens": tokens,
                "generated_count": len(tokens),
                "stop": stop,
                "text": text,
                "transcript_text": transcript_text,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "model_init_seconds": init_seconds,
                "execution_topology": execution_topology,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _persistent_worker_main(
    connection: Any,
    role: str,
    run_dir_text: str,
    environment: dict[str, str],
) -> None:
    """Own one generated model for a sequence of independent window requests."""
    model: ctypes.CDLL | None = None
    terminal_response: dict[str, Any] | None = None
    phase = "initialization"
    try:
        os.environ.update(environment)
        execution_topology = _apply_worker_affinity()
        run_dir = Path(run_dir_text).resolve()
        if role == "encoder":
            model, init_seconds = _open_encoder_model(run_dir)
            feature_cache: dict[Path, np.ndarray] = {}
            tokenizer = None
        elif role == "decoder":
            model, init_seconds = _open_decoder_model(run_dir)
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(run_dir / "tokenizer.json"))
            feature_cache = {}
        else:
            raise ValueError(f"unknown persistent worker role: {role}")
        connection.send(
            {
                "status": "ready",
                "role": role,
                "model_init_seconds": init_seconds,
                "execution_topology": execution_topology,
            }
        )
        phase = "request"
        while True:
            request = connection.recv()
            if request.get("command") == "close":
                model_to_free = model
                model = None
                phase = "teardown"
                if model_to_free is not None:
                    model_to_free.ck_model_free()
                terminal_response = {"status": "closed", "role": role}
                break
            if request.get("command") != "run":
                raise ValueError(f"invalid {role} worker command")
            arguments = argparse.Namespace(**request["args"])
            if role == "encoder":
                _encoder_worker(
                    arguments,
                    model=model,
                    execution_topology=execution_topology,
                    feature_cache=feature_cache,
                )
            else:
                _decoder_worker(
                    arguments,
                    model=model,
                    execution_topology=execution_topology,
                    tokenizer=tokenizer,
                )
            connection.send({"status": "ok", "role": role})
    except BaseException as error:
        terminal_response = {
            "status": "error",
            "role": role,
            "phase": phase,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        cleanup_error: BaseException | None = None
        if model is not None:
            model_to_free = model
            model = None
            try:
                model_to_free.ck_model_free()
            except BaseException as error:
                cleanup_error = error

        if cleanup_error is not None:
            cleanup_traceback = "".join(
                traceback.format_exception(
                    type(cleanup_error), cleanup_error, cleanup_error.__traceback__
                )
            )
            if terminal_response is not None and terminal_response["status"] == "error":
                terminal_response["cleanup_error"] = str(cleanup_error)
                terminal_response["cleanup_traceback"] = cleanup_traceback
            else:
                terminal_response = {
                    "status": "error",
                    "role": role,
                    "phase": "teardown",
                    "error": str(cleanup_error),
                    "traceback": cleanup_traceback,
                }

        try:
            if terminal_response is not None:
                connection.send(terminal_response)
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()


class _PersistentWorker:
    def __init__(
        self,
        role: str,
        run_dir: Path,
        environment: dict[str, str],
        timeout_seconds: float,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("persistent worker timeout must be positive")
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_persistent_worker_main,
            args=(child, role, str(run_dir), environment),
            name=f"cke-whisper-{role}",
        )
        self._role = role
        self._timeout_seconds = timeout_seconds
        self._closed = False
        self._terminal_error_received = False
        self._cleanup_failure: str | None = None
        self.shutdown_response: dict[str, Any] | None = None
        self._process.start()
        child.close()
        try:
            ready = self._receive()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(
                    f"persistent {self._role} worker startup cleanup also "
                    f"failed: {cleanup_error}"
                )
            raise
        if ready.get("status") != "ready":
            error = RuntimeError(f"{role} worker did not become ready")
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(
                    f"persistent {self._role} worker startup cleanup also "
                    f"failed: {cleanup_error}"
                )
            raise error
        self.model_init_seconds = float(ready["model_init_seconds"])
        self.execution_topology = ready["execution_topology"]

    def _receive(self, timeout_seconds: float | None = None) -> dict[str, Any]:
        deadline = self._timeout_seconds if timeout_seconds is None else timeout_seconds
        if not self._connection.poll(deadline):
            raise TimeoutError(
                f"persistent {self._role} worker exceeded "
                f"{deadline:.1f}s response deadline"
            )
        try:
            response = self._connection.recv()
        except EOFError as error:
            raise RuntimeError(
                f"persistent {self._role} worker exited without a response "
                f"(exitcode={self._process.exitcode})"
            ) from error
        if response.get("status") == "error":
            self._terminal_error_received = True
            detail = (
                f"persistent {self._role} worker failed: {response['error']}\n"
                f"{response['traceback']}"
            )
            if response.get("cleanup_error"):
                detail += (
                    f"\npersistent {self._role} worker cleanup also failed: "
                    f"{response['cleanup_error']}\n"
                    f"{response.get('cleanup_traceback', '')}"
                )
            raise RuntimeError(detail)
        return response

    def run(self, arguments: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError(f"persistent {self._role} worker is closed")
        self._connection.send({"command": "run", "args": arguments})
        response = self._receive()
        if response.get("status") != "ok":
            raise RuntimeError(f"invalid persistent {self._role} response")

    def __enter__(self) -> _PersistentWorker:
        return self

    def __exit__(self, _type: object, error: BaseException | None, _tb: object) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is None:
                raise
            error.add_note(
                f"persistent {self._role} worker cleanup also failed: "
                f"{cleanup_error}"
            )

    def _force_stop(self, timeout_seconds: float) -> str | None:
        """Terminate, then kill, and never return while the process is alive."""
        if not self._process.is_alive():
            return None
        action = "terminated"
        self._process.terminate()
        self._process.join(timeout=timeout_seconds)
        if self._process.is_alive():
            action = "killed"
            self._process.kill()
            self._process.join(timeout=timeout_seconds)
        if self._process.is_alive():
            raise RuntimeError(
                f"persistent {self._role} worker survived terminate and kill"
            )
        return action

    def _remember_cleanup_failure(self, error: BaseException) -> None:
        self._cleanup_failure = "".join(
            traceback.format_exception_only(type(error), error)
        ).strip()

    def close(self) -> dict[str, Any] | None:
        if self._closed:
            if self._cleanup_failure is not None:
                raise RuntimeError(self._cleanup_failure)
            return self.shutdown_response
        self._closed = True
        shutdown_timeout = min(self._timeout_seconds, 10.0)
        try:
            if self._terminal_error_received:
                self._process.join(timeout=shutdown_timeout)
                if self._process.is_alive():
                    raise RuntimeError(
                        f"persistent {self._role} worker did not exit after "
                        "reporting its failure"
                    )
                if self._process.exitcode != 0:
                    raise RuntimeError(
                        f"persistent {self._role} worker exited after reporting "
                        f"its failure with code {self._process.exitcode}"
                    )
                return None
            if self._process.is_alive():
                self._connection.send({"command": "close"})
                response = self._receive(shutdown_timeout)
                if (
                    response.get("status") != "closed"
                    or response.get("role") != self._role
                ):
                    raise RuntimeError(
                        f"invalid persistent {self._role} shutdown response: "
                        f"{response!r}"
                    )
                self.shutdown_response = response
            elif not self._terminal_error_received:
                raise RuntimeError(
                    f"persistent {self._role} worker exited before shutdown "
                    f"acknowledgment (exitcode={self._process.exitcode})"
                )
            self._process.join(timeout=shutdown_timeout)
            if self._process.is_alive():
                raise RuntimeError(
                    f"persistent {self._role} worker did not exit after "
                    "shutdown acknowledgment"
                )
            if self._process.exitcode != 0:
                raise RuntimeError(
                    f"persistent {self._role} worker exited after shutdown with "
                    f"code {self._process.exitcode}"
                )
            return self.shutdown_response
        except BaseException as error:
            if self._terminal_error_received and self._process.is_alive():
                self._process.join(timeout=shutdown_timeout)
            try:
                forced_action = self._force_stop(shutdown_timeout)
            except BaseException as force_error:
                error.add_note(
                    f"persistent {self._role} forced cleanup also failed: "
                    f"{force_error}"
                )
            else:
                if forced_action is not None:
                    error.add_note(
                        f"persistent {self._role} worker was {forced_action}"
                    )
            self._remember_cleanup_failure(error)
            raise
        finally:
            self._connection.close()


def _close_persistent_workers(
    workers: list[_PersistentWorker],
    primary_error: BaseException | None = None,
) -> tuple[dict[str, dict[str, Any] | None], list[str]]:
    """Close workers in reverse construction order without hiding failures."""
    responses: dict[str, dict[str, Any] | None] = {}
    failures: list[str] = []
    for worker in reversed(workers):
        try:
            responses[worker._role] = worker.close()
        except BaseException as cleanup_error:
            detail = (
                f"persistent {worker._role} worker cleanup failed: "
                f"{cleanup_error}"
            )
            failures.append(detail)
            if primary_error is not None:
                primary_error.add_note(detail)
    return responses, failures


def _run_segment(
    args: argparse.Namespace,
    *,
    common: list[str],
    encoder_dir: Path,
    decoder_dir: Path,
    wav_path: Path,
    temp: Path,
    index: int,
    window_start_frame: int,
    worker_env: dict[str, str],
    full_features: Path | None = None,
    frontend_report: Path | None = None,
    encoder_worker: _PersistentWorker | None = None,
    decoder_worker: _PersistentWorker | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    encoder_output = temp / f"encoder-{index:04d}.npy"
    encoder_report = temp / f"encoder-{index:04d}.json"
    decoder_report = temp / f"decoder-{index:04d}.json"
    encoder_arguments = {
        "encoder_run_dir": encoder_dir,
        "wav": wav_path,
        "window_start_frame": window_start_frame,
        "full_features": full_features,
        "frontend_report": frontend_report,
        "encoder_output": encoder_output,
        "feature_output": None,
        "worker_report": encoder_report,
    }
    decoder_arguments = {
        "decoder_run_dir": decoder_dir,
        "encoder_output": encoder_output,
        "language": args.language,
        "task": args.task,
        "max_tokens": args.max_tokens,
        "timestamps": args.timestamps,
        "worker_report": decoder_report,
    }
    encoder_started = time.perf_counter()
    if encoder_worker is not None and decoder_worker is not None:
        encoder_worker.run(encoder_arguments)
        encoder_wall_seconds = time.perf_counter() - encoder_started
        decoder_started = time.perf_counter()
        decoder_worker.run(decoder_arguments)
        decoder_wall_seconds = time.perf_counter() - decoder_started
    else:
        subprocess.run(
            [
                *common,
                "_encoder",
                "--encoder-run-dir",
                str(encoder_dir),
                "--wav",
                str(wav_path),
                "--window-start-frame",
                str(window_start_frame),
                *(
                    [
                        "--full-features",
                        str(full_features),
                        "--frontend-report",
                        str(frontend_report),
                    ]
                    if full_features is not None and frontend_report is not None
                    else []
                ),
                "--encoder-output",
                str(encoder_output),
                "--worker-report",
                str(encoder_report),
            ],
            check=True,
            env=worker_env,
        )
        encoder_wall_seconds = time.perf_counter() - encoder_started
        decoder_started = time.perf_counter()
        subprocess.run(
            [
                *common,
                "_decoder",
                "--decoder-run-dir",
                str(decoder_dir),
                "--encoder-output",
                str(encoder_output),
                "--language",
                args.language,
                "--task",
                args.task,
                "--max-tokens",
                str(args.max_tokens),
                *(["--timestamps"] if args.timestamps else []),
                "--worker-report",
                str(decoder_report),
            ],
            check=True,
            env=worker_env,
        )
        decoder_wall_seconds = time.perf_counter() - decoder_started
    encoder_result = json.loads(encoder_report.read_text(encoding="utf-8"))
    decoder_result = json.loads(decoder_report.read_text(encoding="utf-8"))
    encoder_result["worker_wall_seconds"] = encoder_wall_seconds
    decoder_result["worker_wall_seconds"] = decoder_wall_seconds
    return encoder_result, decoder_result


def _collect_segments(
    args: argparse.Namespace,
    *,
    common: list[str],
    encoder_dir: Path,
    decoder_dir: Path,
    wav_path: Path,
    temp: Path,
    worker_env: dict[str, str],
    encoder_config: dict[str, Any],
    generation: dict[str, Any],
    full_features: Path | None,
    frontend_report: Path | None,
    encoder_worker: _PersistentWorker | None,
    decoder_worker: _PersistentWorker | None,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    window_start_frame = 0
    target_sample_rate = int(encoder_config["audio_sample_rate"])
    while True:
        encoder, decoder = _run_segment(
            args,
            common=common,
            encoder_dir=encoder_dir,
            decoder_dir=decoder_dir,
            wav_path=wav_path,
            temp=temp,
            index=len(segments),
            window_start_frame=window_start_frame,
            worker_env=worker_env,
            full_features=full_features,
            frontend_report=frontend_report,
            encoder_worker=encoder_worker,
            decoder_worker=decoder_worker,
        )
        audio = encoder["audio"]
        source_rate = int(audio["source_sample_rate"])
        source_frames = int(audio["source_frames"])
        window_frames = int(audio["window_source_frames"])
        try:
            plan_audio_windows(
                source_frames,
                source_rate,
                target_sample_rate,
                int(encoder_config["audio_sample_extent"]),
            )
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        start_seconds = window_start_frame / source_rate
        window_end_frame = min(source_frames, window_start_frame + window_frames)
        timestamp_events = global_timestamp_events(
            decoder["generated_tokens"], generation, start_seconds
        )
        consumed_frames = (
            timestamp_seek_consumed_frames(
                decoder["generated_tokens"],
                generation,
                source_rate,
                window_frames,
            )
            if args.timestamps
            else window_frames
        )
        consumed_end_frame = min(
            source_frames, window_start_frame + consumed_frames
        )
        if args.timestamps:
            consumed_end_frame = consume_timestamp_sized_tail(
                consumed_end_frame, source_frames, source_rate
            )
        segments.append(
            {
                "index": len(segments),
                "source_frame_start": window_start_frame,
                "source_frame_window_end": window_end_frame,
                "source_frame_consumed_end": consumed_end_frame,
                "start_seconds": start_seconds,
                "end_seconds": consumed_end_frame / source_rate,
                "timestamp_offset_seconds": start_seconds,
                "timestamp_events": timestamp_events,
                "encoder": encoder,
                "decoder": decoder,
            }
        )
        if consumed_end_frame >= source_frames:
            return segments
        if consumed_end_frame <= window_start_frame:
            raise RuntimeError("audio window scheduler made no progress")
        window_start_frame = consumed_end_frame


def _run_parent(args: argparse.Namespace) -> int:
    parent_started = time.perf_counter()
    encoder_dir = args.encoder_run_dir.resolve()
    decoder_dir = args.decoder_run_dir.resolve()
    wav_path = args.wav.resolve()
    _require_artifact(encoder_dir)
    _require_artifact(decoder_dir)
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)
    encoder_config = json.loads(
        (encoder_dir / "config.json").read_text(encoding="utf-8")
    )
    worker_env = _worker_environment(encoder_config)
    target_sample_rate = int(encoder_config["audio_sample_rate"])
    generation = json.loads(
        (decoder_dir / "generation_config.json").read_text(encoding="utf-8")
    )

    temp_root = args.temp_dir.resolve() if args.temp_dir is not None else None
    if temp_root is not None:
        temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cke-whisper-", dir=temp_root) as temp_text:
        temp = Path(temp_text)
        common = [sys.executable, str(Path(__file__).resolve())]
        full_features: Path | None = None
        frontend_report_path: Path | None = None
        frontend_report: dict[str, Any] | None = None
        frontend_reuse_reason = "input is not eligible for native-rate long-audio reuse"
        wav_geometry = _wav_geometry_for_cache(wav_path)
        source_frames, source_rate = wav_geometry or (0, 0)
        if (
            source_frames > int(encoder_config["audio_sample_extent"])
            and source_rate == target_sample_rate
        ):
            feature_frames = source_frames // int(encoder_config["audio_hop_length"])
            required_bytes = _frontend_cache_required_bytes(
                encoder_config, feature_frames
            )
            frontend_reuse_reason = _probe_cache_capacity(temp, required_bytes)
            candidate_features = temp / "audio-features-full.npy"
            candidate_report = temp / "audio-features-full.json"
            completed = None
            if frontend_reuse_reason is None:
                completed = subprocess.run(
                [
                    *common,
                    "_frontend",
                    "--encoder-run-dir",
                    str(encoder_dir),
                    "--wav",
                    str(wav_path),
                    "--feature-frames",
                    str(feature_frames),
                    "--feature-output",
                    str(candidate_features),
                    "--worker-report",
                    str(candidate_report),
                ],
                check=False,
                env=worker_env,
            )
            if completed is not None and completed.returncode == 0:
                full_features = candidate_features
                frontend_report_path = candidate_report
                frontend_report = json.loads(
                    frontend_report_path.read_text(encoding="utf-8")
                )
                frontend_reuse_reason = "enabled"
            elif completed is not None:
                try:
                    frontend_reuse_reason = _frontend_reuse_worker_failure(
                        completed.returncode
                    )
                except ValueError:
                    raise subprocess.CalledProcessError(
                        completed.returncode, completed.args
                    ) from None
            if frontend_report is None:
                print(
                    f"warning: frontend reuse disabled: {frontend_reuse_reason}",
                    file=sys.stderr,
                )
        persistent = args.worker_lifecycle == "persistent"
        encoder_worker: _PersistentWorker | None = None
        decoder_worker: _PersistentWorker | None = None
        workers: list[_PersistentWorker] = []
        cleanup_failures: list[str] = []
        shutdown_responses: dict[str, dict[str, Any] | None] = {}
        try:
            if persistent:
                encoder_worker = _PersistentWorker(
                    "encoder", encoder_dir, worker_env, args.worker_timeout_seconds
                )
                workers.append(encoder_worker)
                decoder_worker = _PersistentWorker(
                    "decoder", decoder_dir, worker_env, args.worker_timeout_seconds
                )
                workers.append(decoder_worker)
            worker_lifecycle = {
                "mode": args.worker_lifecycle,
                "encoder_processes": 1 if persistent else None,
                "decoder_processes": 1 if persistent else None,
                "encoder_model_initializations": 1 if persistent else None,
                "decoder_model_initializations": 1 if persistent else None,
                "encoder_model_init_seconds": (
                    encoder_worker.model_init_seconds if encoder_worker else None
                ),
                "decoder_model_init_seconds": (
                    decoder_worker.model_init_seconds if decoder_worker else None
                ),
            }
            segments = _collect_segments(
                args,
                common=common,
                encoder_dir=encoder_dir,
                decoder_dir=decoder_dir,
                wav_path=wav_path,
                temp=temp,
                worker_env=worker_env,
                encoder_config=encoder_config,
                generation=generation,
                full_features=full_features,
                frontend_report=frontend_report_path,
                encoder_worker=encoder_worker,
                decoder_worker=decoder_worker,
            )
        except BaseException as error:
            _close_persistent_workers(workers, error)
            raise
        else:
            shutdown_responses, cleanup_failures = _close_persistent_workers(workers)
            worker_lifecycle["shutdown"] = {
                "status": "failed" if cleanup_failures else "ok",
                "responses": shutdown_responses,
                "failures": cleanup_failures,
            }
        if not persistent:
            worker_lifecycle.update(
                {
                    "encoder_processes": len(segments),
                    "decoder_processes": len(segments),
                    "encoder_model_initializations": len(segments),
                    "decoder_model_initializations": len(segments),
                    "encoder_model_init_seconds": sum(
                        float(segment["encoder"].get("model_init_seconds", 0.0))
                        for segment in segments
                    ),
                    "decoder_model_init_seconds": sum(
                        float(segment["decoder"].get("model_init_seconds", 0.0))
                        for segment in segments
                    ),
                }
            )

    generated_tokens = [
        token
        for segment in segments
        for token in segment["decoder"]["generated_tokens"]
    ]
    stitched_text = "".join(
        str(segment["decoder"]["text"]) for segment in segments
    )
    stitched_transcript = "".join(
        str(segment["decoder"]["transcript_text"])
        for segment in segments
    )
    encoder = {
        **segments[0]["encoder"],
        "frontend_seconds": sum(
            float(segment["encoder"]["frontend_seconds"])
            for segment in segments
        ) + float((frontend_report or {}).get("frontend_seconds", 0.0)),
        "audio_encoder_seconds": sum(
            float(segment["encoder"]["audio_encoder_seconds"])
            for segment in segments
        ) + float((frontend_report or {}).get("frontend_seconds", 0.0)),
        "encoder_seconds": sum(
            float(segment["encoder"]["encoder_seconds"])
            for segment in segments
        ),
        "window_count": len(segments),
        "frontend_reuse": {
            "enabled": frontend_report is not None,
            "reason": frontend_reuse_reason,
            "full_feature_seconds": float(
                (frontend_report or {}).get("frontend_seconds", 0.0)
            ),
            "full_feature_sha256": str(
                (frontend_report or {}).get("feature_sha256", "")
            ),
        },
    }
    decoder = {
        **segments[0]["decoder"],
        "generated_tokens": generated_tokens,
        "generated_count": len(generated_tokens),
        "stop": (
            segments[0]["decoder"]["stop"]
            if len(segments) == 1
            else "segment_complete"
        ),
        "text": stitched_text,
        "transcript_text": stitched_transcript,
        "prefill_seconds": sum(
            float(segment["decoder"]["prefill_seconds"])
            for segment in segments
        ),
        "decode_seconds": sum(
            float(segment["decoder"]["decode_seconds"])
            for segment in segments
        ),
        "window_count": len(segments),
    }

    decoder_config = json.loads(
        (decoder_dir / "config.json").read_text(encoding="utf-8")
    )
    phase_seconds = (
        float(encoder["frontend_seconds"])
        + float(encoder["encoder_seconds"])
        + float(decoder["prefill_seconds"])
        + float(decoder["decode_seconds"])
    )
    parent_wall_seconds = time.perf_counter() - parent_started
    report = {
        "schema": "cke.whisper_e2e",
        "schema_version": 5,
        "status": "error" if cleanup_failures else "ok",
        "wav": str(wav_path),
        "wav_sha256": _sha256(wav_path),
        "encoder_run_dir": str(encoder_dir),
        "decoder_run_dir": str(decoder_dir),
        "encoder_runtime_sha256": _sha256(encoder_dir / "libmodel.so"),
        "decoder_runtime_sha256": _sha256(decoder_dir / "libmodel.so"),
        "encoder_engine_sha256": _sha256(encoder_dir / "libckernel_engine.so"),
        "decoder_engine_sha256": _sha256(decoder_dir / "libckernel_engine.so"),
        "request": {
            "max_tokens_per_window": int(args.max_tokens),
        },
        "provenance": {
            "encoder": {
                "config_sha256": _sha256(encoder_dir / "config.json"),
                "weights_sha256": _sha256(encoder_dir / "weights.bump"),
                "manifest_sha256": _sha256(
                    encoder_dir / "weights_manifest.map"
                ),
                "layers": int(encoder_config["num_layers"]),
                "embed_dim": int(encoder_config["embed_dim"]),
                "heads": int(encoder_config["num_heads"]),
                "context_length": int(encoder_config["context_length"]),
            },
            "decoder": {
                "config_sha256": _sha256(decoder_dir / "config.json"),
                "weights_sha256": _sha256(decoder_dir / "weights.bump"),
                "manifest_sha256": _sha256(
                    decoder_dir / "weights_manifest.map"
                ),
                "generation_config_sha256": _sha256(
                    decoder_dir / "generation_config.json"
                ),
                "tokenizer_sha256": _sha256(decoder_dir / "tokenizer.json"),
                "layers": int(decoder_config["num_layers"]),
                "embed_dim": int(decoder_config["embed_dim"]),
                "heads": int(decoder_config["num_heads"]),
                "context_length": int(decoder_config["context_length"]),
                "encoder_memory_length": int(
                    decoder_config["encoder_memory_length"]
                ),
                "vocab_size": int(decoder_config["vocab_size"]),
            },
        },
        "language": args.language,
        "task": args.task,
        "timestamps": bool(args.timestamps),
        "windowing": {
            "policy": (
                "timestamp_seek"
                if args.timestamps
                else "fixed_non_overlapping_source_windows"
            ),
            "window_count": len(segments),
            "long_audio_sample_rate": target_sample_rate,
        },
        "execution_topology": {
            "encoder": encoder.get("execution_topology"),
            "decoder": decoder.get("execution_topology"),
        },
        "worker_lifecycle": worker_lifecycle,
        "timing": {
            "parent_wall_seconds": parent_wall_seconds,
            "reported_phase_seconds": phase_seconds,
            "outside_phase_seconds": max(0.0, parent_wall_seconds - phase_seconds),
            "encoder_worker_wall_seconds": sum(
                float(segment["encoder"]["worker_wall_seconds"])
                for segment in segments
            ),
            "decoder_worker_wall_seconds": sum(
                float(segment["decoder"]["worker_wall_seconds"])
                for segment in segments
            ),
        },
        "segments": segments,
        "encoder": encoder,
        "decoder": decoder,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(decoder["text"])
    print(
        "frontend={:.3f}s encoder={:.3f}s prefill={:.3f}s "
        "decode={:.3f}s tokens={} stop={}".format(
            encoder["frontend_seconds"],
            encoder["encoder_seconds"],
            decoder["prefill_seconds"],
            decoder["decode_seconds"],
            decoder["generated_count"],
            decoder["stop"],
        ),
        file=sys.stderr,
    )
    if args.output:
        print(f"report={args.output}", file=sys.stderr)
    if cleanup_failures:
        raise RuntimeError("; ".join(cleanup_failures))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--encoder-run-dir", type=Path, required=True)
    run.add_argument("--decoder-run-dir", type=Path, required=True)
    run.add_argument("--wav", type=Path, required=True)
    run.add_argument("--language", default="en")
    run.add_argument("--task", choices=("transcribe", "translate"), default="transcribe")
    run.add_argument("--max-tokens", type=int, default=128)
    run.add_argument(
        "--timestamps",
        action="store_true",
        help="Generate Whisper timestamp tokens using the model contract",
    )
    run.add_argument("--output", type=Path)
    run.add_argument(
        "--temp-dir",
        type=Path,
        help="Temporary storage root for request-scoped audio artifacts",
    )
    run.add_argument(
        "--worker-lifecycle",
        choices=("per-window", "persistent"),
        default="persistent",
        help=(
            "Worker/model lifetime; persistent retains immutable model state "
            "while resetting each audio window (default: persistent)"
        ),
    )
    run.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=600.0,
        help="Maximum time for one persistent worker response (default: 600)",
    )

    encoder = subparsers.add_parser("_encoder", help=argparse.SUPPRESS)
    encoder.add_argument("--encoder-run-dir", type=Path, required=True)
    encoder.add_argument("--wav", type=Path, required=True)
    encoder.add_argument("--window-start-frame", type=int, default=0)
    encoder.add_argument("--full-features", type=Path)
    encoder.add_argument("--frontend-report", type=Path)
    encoder.add_argument("--encoder-output", type=Path, required=True)
    encoder.add_argument("--feature-output", type=Path)
    encoder.add_argument("--worker-report", type=Path, required=True)

    frontend = subparsers.add_parser("_frontend", help=argparse.SUPPRESS)
    frontend.add_argument("--encoder-run-dir", type=Path, required=True)
    frontend.add_argument("--wav", type=Path, required=True)
    frontend.add_argument("--feature-frames", type=int, required=True)
    frontend.add_argument("--feature-output", type=Path, required=True)
    frontend.add_argument("--worker-report", type=Path, required=True)

    decoder = subparsers.add_parser("_decoder", help=argparse.SUPPRESS)
    decoder.add_argument("--decoder-run-dir", type=Path, required=True)
    decoder.add_argument("--encoder-output", type=Path, required=True)
    decoder.add_argument("--language", required=True)
    decoder.add_argument("--task", required=True)
    decoder.add_argument("--max-tokens", type=int, required=True)
    decoder.add_argument("--timestamps", action="store_true")
    decoder.add_argument("--worker-report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "_frontend":
        return _frontend_worker(args)
    if args.command == "_encoder":
        return _encoder_worker(args)
    if args.command == "_decoder":
        return _decoder_worker(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
