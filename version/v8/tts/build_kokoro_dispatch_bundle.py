#!/usr/bin/env python3
"""Build exact-length or bounded runtime-length Kokoro request bundles.

This runs at export time. The deployed selector and generated model are C.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from build_kokoro_request_runtime import CONV_PROVIDERS, ROOT, build_runtime
from export_kokoro_request_package import DEFAULT_BINDINGS, sha256_file


SUPPORTED_COUNTS = (8, 36)
RUNTIME_EXECUTION_COUNTS = (8, 13, 36)
RUNTIME_MIN_COUNT = 3
RUNTIME_MAX_COUNT = 40  # Native frontend, CSV parser, and voice-row contract.
DISPATCH_SOURCE = ROOT / "version/v8/tts/native_kokoro_dispatch.c"


def selected_counts(experimental_counts: tuple[int, ...]) -> tuple[int, ...]:
    if len(experimental_counts) > 4:
        raise ValueError("at most four experimental phoneme counts per bundle")
    if any(isinstance(count, bool) or not isinstance(count, int) or
           count < 3 or count > 40 or count in SUPPORTED_COUNTS
           for count in experimental_counts):
        raise ValueError("experimental phoneme counts must be distinct integers "
                         "from 3 to 40 outside the 8/36 baseline")
    if len(set(experimental_counts)) != len(experimental_counts):
        raise ValueError("duplicate experimental phoneme count")
    return tuple(sorted((*SUPPORTED_COUNTS, *experimental_counts)))


def selected_runtime_counts(experimental_counts: tuple[int, ...],
                            experimental_range: bool) -> tuple[int, ...]:
    if experimental_range and experimental_counts:
        raise ValueError("select an experimental runtime range or explicit counts")
    if experimental_range:
        return tuple(range(RUNTIME_MIN_COUNT, RUNTIME_MAX_COUNT + 1))
    if any(isinstance(count, bool) or not isinstance(count, int) or
           count < RUNTIME_MIN_COUNT or count > RUNTIME_MAX_COUNT or
           count in RUNTIME_EXECUTION_COUNTS for count in experimental_counts):
        raise ValueError("experimental runtime counts must be distinct integers "
                         "from 3 to 40 outside the captured 8/13/36 counts")
    if len(experimental_counts) > 4 or len(set(experimental_counts)) != len(experimental_counts):
        raise ValueError("at most four distinct experimental runtime counts")
    return tuple(sorted((*RUNTIME_EXECUTION_COUNTS, *experimental_counts)))


def build_bundle(bump_dir: Path, voice_bank: Path, output: Path,
                 *, conv_provider: str = "scalar",
                 conv_workers: int = 4,
                 experimental_counts: tuple[int, ...] = (),
                 runtime_lengths: bool = False,
                 runtime_experimental_range: bool = False) -> dict:
    if conv_provider not in CONV_PROVIDERS:
        raise ValueError(f"unsupported Kokoro Conv1D provider: {conv_provider}")
    if isinstance(conv_workers, bool) or not isinstance(conv_workers, int) or \
            not 1 <= conv_workers <= 16:
        raise ValueError("Conv1D worker count must be an integer in 1..16")
    if runtime_experimental_range and not runtime_lengths:
        raise ValueError("experimental runtime range requires --runtime-lengths")
    counts = (selected_runtime_counts(experimental_counts, runtime_experimental_range)
              if runtime_lengths else selected_counts(experimental_counts))
    if runtime_lengths:
        execution_only = tuple(count for count in counts
                               if count not in RUNTIME_EXECUTION_COUNTS)
    else:
        execution_only = experimental_counts
    if output.is_symlink() or (output.exists() and
            (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("output directory must be empty")
    bindings = json.loads(DEFAULT_BINDINGS.read_text())
    expected_voice = bindings["native_voice_sha256"]
    if sha256_file(voice_bank) != expected_voice:
        raise ValueError("native voice bank differs from pinned af_heart artifact")
    with voice_bank.open("rb") as stream:
        header = stream.read(16)
    if header[:8] != b"CKVOICE1" or \
            int.from_bytes(header[8:12], "little") < max(counts) - 2 or \
            int.from_bytes(header[12:16], "little") != 256:
        raise ValueError("unsupported native af_heart voice bank geometry")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-",
                                     dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        report = _build_into(bump_dir, voice_bank, staging, bindings,
                             expected_voice, conv_provider, conv_workers, counts,
                             execution_only, runtime_lengths)
        # The complete bundle becomes visible in one same-filesystem rename.
        # Replacing an existing empty directory preserves the prior API.
        if output.is_symlink() or (output.exists() and
                (not output.is_dir() or any(output.iterdir()))):
            raise ValueError("output directory must be empty")
        staging.replace(output)
    return report


def _build_into(bump_dir: Path, voice_bank: Path, output: Path,
                bindings: dict, expected_voice: str,
                conv_provider: str, conv_workers: int, counts: tuple[int, ...],
                experimental_counts: tuple[int, ...],
                runtime_lengths: bool) -> dict:
    output.mkdir()
    shutil.copy2(voice_bank, output / "af_heart.ckvoice")
    if sha256_file(output / "af_heart.ckvoice") != expected_voice:
        raise ValueError("copied native voice bank differs from pinned artifact")
    buckets = {}
    for count in ((64,) if runtime_lengths else counts):
        runtime_options = {"runtime_lengths": True} if runtime_lengths else {}
        if conv_provider == "avx2_parallel":
            runtime_options["conv_workers"] = conv_workers
        report = build_runtime(bump_dir, output / str(count), count,
                               conv_provider=conv_provider,
                               **runtime_options)
        for accepted_count in (counts if runtime_lengths else (count,)):
            buckets[accepted_count] = {
                "library_sha256": report["library_sha256"],
                "host_sha256": report["host_sha256"],
                "package_sha256": report["package_sha256"],
                "compiler_manifest_sha256": report["compiler_manifest_sha256"],
                "generated_c_sha256": report["generated_c_sha256"],
                "generated_operations": report["generated_operations"],
                "selected_conv_calls": report["selected_conv_calls"],
            }
    lines = [
        "/* Generated by build_kokoro_dispatch_bundle.py; do not edit. */",
        "#ifndef CKE_KOKORO_BUNDLE_IDENTITY_H",
        "#define CKE_KOKORO_BUNDLE_IDENTITY_H",
        "typedef struct { unsigned count; const char *host_sha256; "
        "const char *library_sha256; const char *package_sha256; } ck_tts_bucket;",
        f'static const char ck_tts_voice_sha256[] = "{expected_voice}";',
        "static const ck_tts_bucket ck_tts_buckets[] = {",
    ]
    for count in counts:
        item = buckets[count]
        lines.append(f'    {{{count}, "{item["host_sha256"]}", '
                     f'"{item["library_sha256"]}", '
                     f'"{item["package_sha256"]}"}},')
    lines.extend(["};", "#endif"])
    (output / "bundle_identity.h").write_text("\n".join(lines) + "\n")
    command = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
               "-pedantic", "-I", str(output), str(DISPATCH_SOURCE),
               "-lcrypto", "-o", str(output / "native-kokoro-dispatch")]
    if runtime_lengths:
        command.insert(6, "-DCK_TTS_RUNTIME_BUNDLE")
    result = subprocess.run(command, capture_output=True, text=True)
    (output / "dispatch-compile.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"native dispatcher compilation failed: {result.stderr}")
    report = {
        "schema": "cke.kokoro.native_dispatch_bundle.v1",
        "runtime_lengths": runtime_lengths,
        "conv_provider": conv_provider,
        "conv_workers": conv_workers if conv_provider == "avx2_parallel" else 0,
        "supported_phoneme_counts": list(counts),
        "baseline_phoneme_counts": list(SUPPORTED_COUNTS),
        "captured_phoneme_counts": (list(RUNTIME_EXECUTION_COUNTS)
                                    if runtime_lengths else list(SUPPORTED_COUNTS)),
        "experimental_phoneme_counts": list(sorted(experimental_counts)),
        "execution_only_phoneme_counts": (list(sorted(experimental_counts))
                                           if runtime_lengths else []),
        "voice_sha256": expected_voice,
        "source_bump_sha256": bindings["source_bump_sha256"],
        "binding_catalog_sha256": sha256_file(DEFAULT_BINDINGS),
        "dispatcher_sha256": sha256_file(output / "native-kokoro-dispatch"),
        "bundle_identity_sha256": sha256_file(output / "bundle_identity.h"),
        "buckets": buckets,
        "dispatcher_compile_command": command,
        "scope": ("one generated 64-token-capacity graph; 8/13/36 have "
                  "captured request evidence; optional other counts have "
                  "bounded execution only; connected waveform parity remains FAIL"
                  if runtime_lengths else
                  "exact compiled phoneme counts only; additional counts require "
                  "independent model-level evidence before numerical certification"),
    }
    (output / "bundle_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bump-dir", required=True, type=Path)
    parser.add_argument("--voice-bank", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conv-provider", choices=CONV_PROVIDERS,
                        default="scalar")
    parser.add_argument("--conv-workers", type=int, default=4,
                        help="caller-owned worker pool size for avx2_parallel")
    parser.add_argument("--experimental-count", type=int, action="append",
                        default=[], help="add an experimental count (3..40); "
                        "runtime mode reuses one generated graph")
    parser.add_argument("--runtime-lengths", action="store_true",
                        help="one 64-token-capacity generated graph for checked requests")
    parser.add_argument("--runtime-experimental-range", action="store_true",
                        help="opt in to execution-only counts 3..40 with one runtime graph; "
                        "independent model parity is not established")
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.bump_dir, args.voice_bank,
                                 args.output_dir, conv_provider=args.conv_provider,
                                 conv_workers=args.conv_workers,
                                 experimental_counts=tuple(args.experimental_count),
                                 runtime_lengths=args.runtime_lengths,
                                 runtime_experimental_range=args.runtime_experimental_range),
                     indent=2))


if __name__ == "__main__":
    main()
