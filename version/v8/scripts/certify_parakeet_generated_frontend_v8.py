#!/usr/bin/env python3
from __future__ import annotations

"""Certify the compiler-generated Parakeet frontend against the pinned fixture."""

import argparse
import ctypes
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
V8_ROOT = SCRIPT_DIR.parent
COMPILER_SOURCES = {
    "circuit": V8_ROOT / "circuits" / "parakeet_tdt.json",
    "build_ir": SCRIPT_DIR / "build_ir_v8.py",
    "codegen": SCRIPT_DIR / "codegen_v8.py",
    "codegen_core": SCRIPT_DIR / "codegen_core_v8.py",
    "converter": SCRIPT_DIR / "convert_safetensors_to_bump_v8.py",
}
BUILD_MANIFEST_SCHEMA = "cke.generated_runtime.build_manifest.v1"
MAX_ABS_TOLERANCE = 5.0e-2


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


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def _runtime_artifacts(runtime_dir: Path, manifest_map: Path) -> dict[str, Path]:
    return {
        "generated_c": runtime_dir / "model_v8.c",
        "model_library": runtime_dir / "libmodel.so",
        "engine_library": runtime_dir / "libckernel_engine.so",
        "tokenizer_library": runtime_dir / "libckernel_tokenizer.so",
        "call_ir": runtime_dir / "call.json",
        "manifest_map": manifest_map,
    }


def create_build_manifest(
    runtime_dir: Path,
    manifest_map: Path,
    compiler_sources: dict[str, Path] | None = None,
) -> dict[str, object]:
    runtime_dir = runtime_dir.resolve()
    sources = compiler_sources or COMPILER_SOURCES
    artifacts = _runtime_artifacts(runtime_dir, manifest_map.resolve())
    missing = [
        str(path)
        for path in (*artifacts.values(), *sources.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing generated build input: " + ", ".join(missing))
    return {
        "schema": BUILD_MANIFEST_SCHEMA,
        "artifacts": {name: _identity(path) for name, path in artifacts.items()},
        "compiler_sources": {
            name: _identity(path) for name, path in sources.items()
        },
    }


def _validate_build_manifest(
    manifest_path: Path,
    runtime_dir: Path,
    manifest_map: Path,
    compiler_sources: dict[str, Path] | None = None,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BUILD_MANIFEST_SCHEMA:
        raise ValueError("generated runtime build manifest has an unsupported schema")
    current = create_build_manifest(runtime_dir, manifest_map, compiler_sources)
    for section in ("artifacts", "compiler_sources"):
        expected_rows = manifest.get(section)
        if not isinstance(expected_rows, dict):
            raise ValueError(f"build manifest missing {section}")
        for name, actual in current[section].items():
            expected = expected_rows.get(name)
            if not isinstance(expected, dict):
                raise ValueError(f"build manifest missing {section}.{name}")
            for field in ("bytes", "sha256"):
                if expected.get(field) != actual.get(field):
                    raise ValueError(
                        f"stale generated runtime identity for {section}.{name}.{field}"
                    )
    return manifest


def _resolved_symbol_library(library: ctypes.CDLL, symbol: str) -> Path:
    function = getattr(library, symbol)
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    dladdr.restype = ctypes.c_int
    info = _DlInfo()
    if (
        dladdr(ctypes.cast(function, ctypes.c_void_p), ctypes.byref(info)) == 0
        or not info.dli_fname
    ):
        raise RuntimeError(f"dladdr could not resolve runtime symbol {symbol}")
    return Path(os.fsdecode(info.dli_fname)).resolve()


def _verify_loaded_libraries(
    library: ctypes.CDLL,
    runtime_dir: Path,
    symbols: dict[str, str] | None = None,
) -> dict[str, Path]:
    requested = {
        "model_library": runtime_dir / "libmodel.so",
        "engine_library": runtime_dir / "libckernel_engine.so",
        "tokenizer_library": runtime_dir / "libckernel_tokenizer.so",
    }
    resolved_symbols = symbols or {
        "model_library": "ck_model_prepare_audio_wav_features",
        "engine_library": "ck_set_num_threads",
        "tokenizer_library": "ck_tokenizer_create",
    }
    loaded = {
        name: _resolved_symbol_library(library, symbol)
        for name, symbol in resolved_symbols.items()
    }
    for name, path in loaded.items():
        expected = requested[name].resolve()
        if path != expected:
            raise RuntimeError(
                f"generated frontend resolved the wrong {name}: "
                f"requested={expected} loaded={path}"
            )
    return loaded


def _load_expected_fixture(
    path: Path,
    expected_channels: int,
    fixture_key: str = "frontend.input_features",
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as fixture:
        if fixture_key not in fixture:
            raise ValueError(f"fixture missing {fixture_key}")
        raw = fixture[fixture_key]
    if raw.dtype != np.float32:
        raise ValueError(f"frontend fixture must be float32, got {raw.dtype}")
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(
            "frontend fixture must have shape [1, frames, channels], "
            f"got {list(raw.shape)}"
        )
    if raw.shape[1] <= 1 or raw.shape[2] != expected_channels:
        raise ValueError(
            "frontend fixture has invalid frame/channel geometry: "
            f"expected channels={expected_channels}, got {list(raw.shape)}"
        )
    return np.ascontiguousarray(raw[0])


def certify(
    runtime_dir: Path,
    weights: Path,
    manifest_map: Path,
    wav_path: Path,
    fixture_path: Path,
    build_manifest_path: Path,
    *,
    compiler_sources: dict[str, Path] | None = None,
    fixture_key: str = "frontend.input_features",
    schema: str = "cke.parakeet.generated_frontend_certification.v1",
    require_terminal_padding_zero: bool = True,
    claim_boundary: dict[str, str] | None = None,
    loaded_library_symbols: dict[str, str] | None = None,
) -> dict[str, object]:
    runtime_dir = runtime_dir.resolve()
    artifacts = _runtime_artifacts(runtime_dir, manifest_map.resolve())
    library_path = artifacts["model_library"]
    generated_path = artifacts["generated_c"]
    engine_path = artifacts["engine_library"]
    tokenizer_path = artifacts["tokenizer_library"]
    call_path = artifacts["call_ir"]
    required = (
        library_path,
        generated_path,
        engine_path,
        call_path,
        weights,
        manifest_map,
        wav_path,
        fixture_path,
        build_manifest_path,
        *(compiler_sources or COMPILER_SOURCES).values(),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing generated frontend evidence: " + ", ".join(missing))

    build_manifest = _validate_build_manifest(
        build_manifest_path.resolve(),
        runtime_dir,
        manifest_map.resolve(),
        compiler_sources,
    )
    call_ir = json.loads(call_path.read_text(encoding="utf-8"))
    config = call_ir.get("config")
    if not isinstance(config, dict):
        raise ValueError("generated call IR is missing config")
    expected_channels = int(config.get("audio_feature_channels", 0) or 0)
    if expected_channels <= 0:
        raise ValueError("generated call IR has invalid audio_feature_channels")
    expected = _load_expected_fixture(fixture_path, expected_channels, fixture_key)

    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    library = ctypes.CDLL(str(library_path), mode=mode)
    loaded_libraries = _verify_loaded_libraries(
        library, runtime_dir, loaded_library_symbols
    )
    u8p = ctypes.POINTER(ctypes.c_uint8)
    f32p = ctypes.POINTER(ctypes.c_float)
    library.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.ck_model_init_with_manifest.restype = ctypes.c_int
    library.ck_model_prepare_audio_wav_features.argtypes = [
        u8p,
        ctypes.c_size_t,
        f32p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(CKAudioWavInfo),
    ]
    library.ck_model_prepare_audio_wav_features.restype = ctypes.c_int
    library.ck_model_free.argtypes = []

    actual = np.empty_like(expected)
    wav = np.frombuffer(wav_path.read_bytes(), dtype=np.uint8)
    produced = ctypes.c_int()
    info = CKAudioWavInfo()
    init_status = int(
        library.ck_model_init_with_manifest(
            str(weights.resolve()).encode(), str(manifest_map.resolve()).encode()
        )
    )
    run_status = -1
    elapsed = 0.0
    try:
        if init_status == 0:
            started = time.perf_counter()
            run_status = int(
                library.ck_model_prepare_audio_wav_features(
                    wav.ctypes.data_as(u8p),
                    wav.size,
                    actual.ctypes.data_as(f32p),
                    actual.shape[0],
                    ctypes.byref(produced),
                    ctypes.byref(info),
                )
            )
            elapsed = time.perf_counter() - started
    finally:
        library.ck_model_free()

    finite = bool(np.isfinite(actual).all()) if run_status == 0 else False
    diff = actual - expected if run_status == 0 else np.full_like(expected, math.nan)
    rmse = float(np.sqrt(np.mean(diff * diff))) if finite else math.inf
    max_abs = float(np.max(np.abs(diff))) if finite else math.inf
    checks = {
        "init_success": init_status == 0,
        "run_success": run_status == 0,
        "shape_exact": int(produced.value) == expected.shape[0],
        "finite": finite,
        "rmse_within_5e_4": rmse <= 5.0e-4,
        "max_abs_within_5e_2": max_abs <= MAX_ABS_TOLERANCE,
    }
    if require_terminal_padding_zero:
        checks["padding_row_exact_zero"] = bool(
            run_status == 0 and np.array_equal(actual[-1], np.zeros_like(actual[-1]))
        )
    return {
        "schema": schema,
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "generated_c_frontend_only",
        "checks": checks,
        "numerical": {
            "rmse": rmse,
            "rmse_tolerance": 5.0e-4,
            "max_abs": max_abs,
            "max_abs_tolerance": MAX_ABS_TOLERANCE,
        },
        "execution": {
            "elapsed_seconds": elapsed,
            "produced_frames": int(produced.value),
            "shape": list(actual.shape),
            "wav": {
                "sample_rate": int(info.sample_rate),
                "channels": int(info.channels),
                "bits_per_sample": int(info.bits_per_sample),
                "frames": int(info.frames),
            },
        },
        "identity": {
            "generated_c": _identity(generated_path),
            "model_library": _identity(library_path),
            "engine_library": _identity(engine_path),
            "tokenizer_library": _identity(tokenizer_path),
            "call_ir": _identity(call_path),
            "weights": _identity(weights),
            "manifest_map": _identity(manifest_map),
            "wav": _identity(wav_path),
            "fixture": _identity(fixture_path),
            "compiler_sources": {
                name: _identity(path)
                for name, path in (compiler_sources or COMPILER_SOURCES).items()
            },
            "loaded_libraries": {
                name: _identity(path) for name, path in loaded_libraries.items()
            },
            "build_manifest": {
                **_identity(build_manifest_path),
                "schema": build_manifest["schema"],
            },
        },
        "claim_boundary": claim_boundary or {
            "frontend": "certified",
            "subsampling": "not_tested",
            "encoder": "not_generated",
            "tdt_decoder": "not_generated",
            "standalone_transcription": "not_certified",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--manifest-map", type=Path, required=True)
    parser.add_argument("--snapshot-build-manifest", type=Path)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.snapshot_build_manifest:
        manifest = create_build_manifest(args.runtime_dir, args.manifest_map)
        args.snapshot_build_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot_build_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))
        return 0
    missing_args = [
        name
        for name in ("build_manifest", "weights", "wav", "fixture", "output")
        if getattr(args, name) is None
    ]
    if missing_args:
        parser.error("certification requires: " + ", ".join(missing_args))
    report = certify(
        args.runtime_dir,
        args.weights,
        args.manifest_map,
        args.wav,
        args.fixture,
        args.build_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
