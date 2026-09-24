#!/usr/bin/env python3
"""Compare CKE's inverse-STFT primitive with a pinned Kokoro oracle capture.

This is a primitive check, not generated-model or end-to-end TTS evidence.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src" / "kernels" / "audio_istft_mag_phase.c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tensor(capture_dir: Path, manifest: dict, name: str) -> np.ndarray:
    record = manifest["tensors"][name]
    path = capture_dir / record["file"]
    actual = sha256(path)
    if actual != record["sha256"]:
        raise ValueError(f"{name} SHA256 mismatch: {actual} != {record['sha256']}")
    return np.load(path, allow_pickle=False)


def compare(capture_dir: Path, atol: float) -> dict:
    manifest_path = capture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "full_oracle_captured":
        raise ValueError("capture is not a completed full oracle")
    mag = load_tensor(capture_dir, manifest, "istft_magnitude")
    phase = load_tensor(capture_dir, manifest, "istft_phase")
    expected = load_tensor(capture_dir, manifest, "waveform_f32").reshape(-1)
    if mag.shape != phase.shape or mag.ndim != 3 or mag.shape[0] != 1:
        raise ValueError("unexpected oracle magnitude/phase shape")
    mag = np.ascontiguousarray(mag[0].T, dtype=np.float32)
    phase = np.ascontiguousarray(phase[0].T, dtype=np.float32)
    if mag.shape[1] != 11:
        raise ValueError("first Kokoro fixture requires FFT20/11 bins")

    with tempfile.TemporaryDirectory() as temp_dir:
        library = Path(temp_dir) / "libcke_istft_oracle.so"
        subprocess.run([
            "cc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
            "-shared", "-fPIC", "-I", str(ROOT / "include"),
            str(SOURCE), "-lm", "-o", str(library),
        ], check=True)
        native = ctypes.CDLL(str(library))
        size = ctypes.c_size_t
        fptr = ctypes.POINTER(ctypes.c_float)
        native.audio_istft_mag_phase_plan_f32.argtypes = [
            size, size, size, ctypes.POINTER(size), ctypes.POINTER(size), ctypes.POINTER(size),
        ]
        native.audio_istft_mag_phase_plan_f32.restype = ctypes.c_int
        native.audio_istft_mag_phase_f32.argtypes = [
            fptr, fptr, size, size, size, size, fptr, size, fptr, size,
        ]
        native.audio_istft_mag_phase_f32.restype = ctypes.c_int
        frames = mag.shape[0]
        spectrum_n, output_n, scratch_n = size(), size(), size()
        status = native.audio_istft_mag_phase_plan_f32(
            frames, 20, 5, ctypes.byref(spectrum_n), ctypes.byref(output_n),
            ctypes.byref(scratch_n),
        )
        if status != 0 or output_n.value != expected.size:
            raise ValueError(f"native plan failed or mismatched waveform extent: {status}")
        output = np.empty(output_n.value, dtype=np.float32)
        scratch = np.empty(scratch_n.value, dtype=np.float32)
        ptr = lambda array: array.ctypes.data_as(fptr)
        status = native.audio_istft_mag_phase_f32(
            ptr(mag), ptr(phase), spectrum_n.value, frames, 20, 5,
            ptr(output), output.size, ptr(scratch), scratch.size,
        )
        if status != 0:
            raise ValueError(f"native inverse STFT failed: {status}")
        difference = np.abs(output - expected)
        max_abs = float(difference.max())
        mean_abs = float(difference.mean())
        passed = bool(np.isfinite(output).all() and max_abs <= atol)
        return {
            "schema": "cke.v8.tts_primitive_oracle.v1",
            "primitive": "audio_istft_mag_phase_f32",
            "status": "passed" if passed else "failed",
            "generated_model": False,
            "oracle_manifest_sha256": sha256(manifest_path),
            "source_sha256": sha256(SOURCE),
            "native_library_sha256": sha256(library),
            "frames": frames,
            "output_samples": output.size,
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "absolute_tolerance": atol,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=2e-5)
    args = parser.parse_args()
    report = compare(args.capture_dir, args.atol)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
