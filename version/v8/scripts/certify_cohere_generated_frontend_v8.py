#!/usr/bin/env python3
from __future__ import annotations

"""Certify Cohere Transcribe's generated-C frontend against a reference fixture."""

import argparse
import json
from pathlib import Path

import certify_parakeet_generated_frontend_v8 as common


SCRIPT_DIR = Path(__file__).resolve().parent
V8_ROOT = SCRIPT_DIR.parent
COMPILER_SOURCES = {
    "circuit": V8_ROOT / "circuits" / "cohere_transcribe.json",
    "build_ir": SCRIPT_DIR / "build_ir_v8.py",
    "codegen": SCRIPT_DIR / "codegen_v8.py",
    "codegen_core": SCRIPT_DIR / "codegen_core_v8.py",
    "converter": SCRIPT_DIR / "convert_cohere_transcribe_gguf_to_bump_v8.py",
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
        manifest = common.create_build_manifest(
            args.runtime_dir, args.manifest_map, COMPILER_SOURCES
        )
        args.snapshot_build_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot_build_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))
        return 0
    missing = [
        name
        for name in ("build_manifest", "weights", "wav", "fixture", "output")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("certification requires: " + ", ".join(missing))
    report = common.certify(
        args.runtime_dir,
        args.weights,
        args.manifest_map,
        args.wav,
        args.fixture,
        args.build_manifest,
        compiler_sources=COMPILER_SOURCES,
        fixture_key="audio.frontend.log_mel.output",
        schema="cke.cohere_transcribe.generated_frontend_certification.v1",
        require_terminal_padding_zero=False,
        loaded_library_symbols={
            "model_library": "ck_model_prepare_audio_wav_features",
            "engine_library": "ck_set_num_threads",
        },
        claim_boundary={
            "frontend": "certified_16khz_pcm16_mono_bounded_clip",
            "subsampling": "not_generated",
            "encoder": "not_generated",
            "decoder": "not_generated",
            "timestamps": "not_generated",
            "standalone_transcription": "not_certified",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
