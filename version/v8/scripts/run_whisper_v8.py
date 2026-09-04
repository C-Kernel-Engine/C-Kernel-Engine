#!/usr/bin/env python3
"""Run generated CKE Whisper encoder and decoder artifacts on a PCM16 WAV."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_U8_P = ctypes.POINTER(ctypes.c_uint8)


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


def _encoder_worker(args: argparse.Namespace) -> int:
    execution_topology = _apply_worker_affinity()
    run_dir = args.encoder_run_dir.resolve()
    _require_artifact(run_dir)
    model = _load_generated_model(run_dir)
    model.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
    model.ck_model_get_named_activation_ptr.restype = ctypes.c_void_p
    model.ck_model_get_named_activation_nbytes.argtypes = [ctypes.c_char_p]
    model.ck_model_get_named_activation_nbytes.restype = ctypes.c_ssize_t
    model.ck_model_run_audio_wav.argtypes = [
        _U8_P,
        ctypes.c_size_t,
        ctypes.POINTER(CKAudioWavInfo),
    ]
    model.ck_model_run_audio_wav.restype = ctypes.c_int
    model.ck_model_run_audio_wav_window.argtypes = [
        _U8_P,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(CKAudioWavInfo),
    ]
    model.ck_model_run_audio_wav_window.restype = ctypes.c_int
    status = int(
        model.ck_model_init_with_manifest(
            str(run_dir / "weights.bump").encode(),
            str(run_dir / "weights_manifest.map").encode(),
        )
    )
    if status != 0:
        raise RuntimeError(f"encoder initialization failed with code {status}")
    try:
        wav = np.frombuffer(args.wav.resolve().read_bytes(), dtype=np.uint8)
        info = CKAudioWavInfo()
        started = time.perf_counter()
        status = int(
            model.ck_model_run_audio_wav_window(
                wav.ctypes.data_as(_U8_P),
                wav.size,
                args.window_start_frame,
                ctypes.byref(info),
            )
        )
        encoder_seconds = time.perf_counter() - started
        if status != 0:
            raise RuntimeError(f"generated audio runtime failed with code {status}")
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        target_rate = int(config["audio_sample_rate"])
        sample_extent = int(config["audio_sample_extent"])
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
                "frontend_seconds": None,
                "audio_encoder_seconds": encoder_seconds,
                "encoder_seconds": encoder_seconds,
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


def _decoder_worker(args: argparse.Namespace) -> int:
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

    status = int(
        model.ck_model_init_with_manifest(
            str(run_dir / "weights.bump").encode(),
            str(run_dir / "weights_manifest.map").encode(),
        )
    )
    if status != 0:
        raise RuntimeError(f"decoder initialization failed with code {status}")
    try:
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
        model.ck_model_free()

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
                "execution_topology": execution_topology,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


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
) -> tuple[dict[str, Any], dict[str, Any]]:
    encoder_output = temp / f"encoder-{index:04d}.npy"
    encoder_report = temp / f"encoder-{index:04d}.json"
    decoder_report = temp / f"decoder-{index:04d}.json"
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
            "--encoder-output",
            str(encoder_output),
            "--worker-report",
            str(encoder_report),
        ],
        check=True,
        env=worker_env,
    )
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
    return (
        json.loads(encoder_report.read_text(encoding="utf-8")),
        json.loads(decoder_report.read_text(encoding="utf-8")),
    )


def _run_parent(args: argparse.Namespace) -> int:
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

    with tempfile.TemporaryDirectory(prefix="cke-whisper-") as temp_text:
        temp = Path(temp_text)
        common = [sys.executable, str(Path(__file__).resolve())]
        segments: list[dict[str, Any]] = []
        window_start_frame = 0
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
            window_end_frame = min(
                source_frames,
                window_start_frame + window_frames,
            )
            timestamp_events = global_timestamp_events(
                decoder["generated_tokens"],
                generation,
                start_seconds,
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
                source_frames,
                window_start_frame + consumed_frames,
            )
            if args.timestamps:
                consumed_end_frame = consume_timestamp_sized_tail(
                    consumed_end_frame,
                    source_frames,
                    source_rate,
                )
            end_seconds = consumed_end_frame / source_rate
            segment = {
                "index": len(segments),
                "source_frame_start": window_start_frame,
                "source_frame_window_end": window_end_frame,
                "source_frame_consumed_end": consumed_end_frame,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "timestamp_offset_seconds": start_seconds,
                "timestamp_events": timestamp_events,
                "encoder": encoder,
                "decoder": decoder,
            }
            segments.append(segment)
            if consumed_end_frame >= source_frames:
                break
            if consumed_end_frame <= window_start_frame:
                raise RuntimeError("audio window scheduler made no progress")
            window_start_frame = consumed_end_frame

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
        "audio_encoder_seconds": sum(
            float(segment["encoder"]["audio_encoder_seconds"])
            for segment in segments
        ),
        "encoder_seconds": sum(
            float(segment["encoder"]["encoder_seconds"])
            for segment in segments
        ),
        "window_count": len(segments),
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
    report = {
        "schema": "cke.whisper_e2e",
        "schema_version": 3,
        "status": "ok",
        "wav": str(wav_path),
        "wav_sha256": _sha256(wav_path),
        "encoder_run_dir": str(encoder_dir),
        "decoder_run_dir": str(decoder_dir),
        "encoder_runtime_sha256": _sha256(encoder_dir / "libmodel.so"),
        "decoder_runtime_sha256": _sha256(decoder_dir / "libmodel.so"),
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
        "audio+encoder={:.3f}s prefill={:.3f}s decode={:.3f}s "
        "tokens={} stop={}".format(
            encoder["audio_encoder_seconds"],
            decoder["prefill_seconds"],
            decoder["decode_seconds"],
            decoder["generated_count"],
            decoder["stop"],
        ),
        file=sys.stderr,
    )
    if args.output:
        print(f"report={args.output}", file=sys.stderr)
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

    encoder = subparsers.add_parser("_encoder", help=argparse.SUPPRESS)
    encoder.add_argument("--encoder-run-dir", type=Path, required=True)
    encoder.add_argument("--wav", type=Path, required=True)
    encoder.add_argument("--window-start-frame", type=int, default=0)
    encoder.add_argument("--encoder-output", type=Path, required=True)
    encoder.add_argument("--feature-output", type=Path)
    encoder.add_argument("--worker-report", type=Path, required=True)

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
    if args.command == "_encoder":
        return _encoder_worker(args)
    if args.command == "_decoder":
        return _decoder_worker(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
