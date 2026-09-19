from __future__ import annotations

import ctypes
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codegen = _load("parakeet_generated_frontend_codegen", SCRIPTS / "codegen_v8.py")
certifier = _load(
    "parakeet_generated_frontend_certifier",
    SCRIPTS / "certify_parakeet_generated_frontend_v8.py",
)


def _call(operation: str, function: str, args: list[tuple[str, str]]) -> dict:
    return {
        "op": operation,
        "function": function,
        "args": [
            {"name": f"arg_{index}", "source": source, "expr": expression}
            for index, (source, expression) in enumerate(args)
        ],
    }


def _compiled_entrypoint() -> tuple[tempfile.TemporaryDirectory, ctypes.CDLL]:
    operations = [
        _call(
            "audio_wav_decode",
            "stub_decode",
            [
                ("runtime:audio_wav_bytes", "audio_wav_bytes"),
                ("runtime:audio_wav_byte_count", "audio_wav_byte_count"),
                ("output:mono", "audio_mono"),
                ("dim:mono_capacity", "audio_mono_capacity"),
                ("runtime:audio_window_start_frame", "audio_window_start_frame"),
                ("runtime:audio_wav_info", "audio_wav_info"),
            ],
        ),
        _call("audio_preemphasis", "stub_preemphasis", [("dim:frames", "configured_frames")]),
        _call("audio_hann_window", "stub_zero", []),
        _call("audio_stft_tables", "stub_zero", []),
        _call(
            "audio_stft",
            "stub_stft",
            [("dim:n_samples", "configured_samples"), ("dim:n_frames", "configured_frames")],
        ),
        _call("audio_mel_filters", "stub_zero", []),
        _call("audio_log_mel", "stub_log_mel", [("dim:frames", "configured_frames")]),
        _call(
            "audio_feature_normalize",
            "stub_normalize",
            [("dim:frames", "configured_frames"), ("output:output", "configured_output")],
        ),
    ]
    entrypoint = codegen._emit_audio_wav_entrypoint(
        operations,
        {
            "audio_sample_rate": 16000,
            "audio_max_source_frames": 320,
            "audio_hop_length": 160,
            "audio_feature_channels": 4,
        },
    )
    source = f"""
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#define CK_EXPORT __attribute__((visibility("default")))
typedef struct {{
    int format_tag;
    int channels;
    int sample_rate;
    int bits_per_sample;
    int frames;
    size_t data_offset;
    size_t data_bytes;
}} CKAudioWavInfo;
typedef struct {{ uint8_t *bump; }} CKModel;
static uint8_t arena[4096];
static CKModel model_instance = {{ arena }};
static CKModel *g_model = &model_instance;
#define A_AUDIO_SAMPLES 0
static int audio_wav_parse_memory(const uint8_t *bytes, size_t count, CKAudioWavInfo *info) {{
    if (!bytes || !info || count != sizeof(*info)) return -1;
    memcpy(info, bytes, sizeof(*info));
    return 0;
}}
static int stub_decode(const uint8_t *bytes, size_t count, float *output, int capacity,
                       int start, CKAudioWavInfo *info) {{
    (void)bytes; (void)count; (void)output; (void)capacity; (void)start;
    return info->frames;
}}
static int stub_zero(void) {{ return 0; }}
static int stub_preemphasis(int frames) {{ return frames > 0 ? 0 : -1; }}
static int stub_stft(int samples, int frames) {{ return samples > 0 && frames > 0 ? 0 : -1; }}
static int stub_log_mel(int frames) {{ return frames > 0 ? 0 : -1; }}
static int stub_normalize(int frames, float *output) {{
    for (int frame = 0; frame < frames; ++frame) {{
        for (int channel = 0; channel < 4; ++channel) output[frame * 4 + channel] = 7.0f;
    }}
    return 0;
}}
{entrypoint}
"""
    temporary = tempfile.TemporaryDirectory()
    directory = Path(temporary.name)
    source_path = directory / "frontend.c"
    library_path = directory / "frontend.so"
    source_path.write_text(source, encoding="utf-8")
    subprocess.run(
        ["gcc", "-std=c11", "-shared", "-fPIC", str(source_path), "-o", str(library_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(library_path))
    library.ck_model_prepare_audio_wav_features.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(certifier.CKAudioWavInfo),
    ]
    library.ck_model_prepare_audio_wav_features.restype = ctypes.c_int
    return temporary, library


class GeneratedParakeetFrontendTests(unittest.TestCase):
    def test_fixture_validation_rejects_dtype_rank_and_channel_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.npz"
            for value, message in (
                (np.zeros((1, 2, 4), dtype=np.float64), "must be float32"),
                (np.zeros((2, 4), dtype=np.float32), "must have shape"),
                (np.zeros((1, 2, 3), dtype=np.float32), "invalid frame/channel"),
            ):
                with self.subTest(message=message):
                    np.savez(path, **{"frontend.input_features": value})
                    with self.assertRaisesRegex(ValueError, message):
                        certifier._load_expected_fixture(path, 4)

    def test_build_manifest_rejects_stale_runtime_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runtime = root / "runtime"
            runtime.mkdir()
            manifest_map = runtime / "weights_manifest.map"
            artifacts = certifier._runtime_artifacts(runtime, manifest_map)
            for index, path in enumerate(artifacts.values()):
                path.write_bytes(f"artifact-{index}".encode())
            compiler_sources = {"compiler": root / "compiler.py"}
            compiler_sources["compiler"].write_text("version = 1\n", encoding="utf-8")
            with mock.patch.object(certifier, "COMPILER_SOURCES", compiler_sources):
                manifest = certifier.create_build_manifest(runtime, manifest_map)
                manifest_path = root / "build.json"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                certifier._validate_build_manifest(manifest_path, runtime, manifest_map)
                artifacts["engine_library"].write_bytes(b"stale-engine")
                with self.assertRaisesRegex(ValueError, "stale generated runtime identity"):
                    certifier._validate_build_manifest(manifest_path, runtime, manifest_map)

    def test_loaded_engine_must_be_the_requested_library(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = Path(td)
            requested = [
                runtime / "libmodel.so",
                runtime / "libckernel_engine.so",
                runtime / "libckernel_tokenizer.so",
            ]
            wrong_engine = runtime / "elsewhere" / "libckernel_engine.so"
            with mock.patch.object(
                certifier,
                "_resolved_symbol_library",
                side_effect=[requested[0], wrong_engine, requested[2]],
            ):
                with self.assertRaisesRegex(RuntimeError, "wrong engine_library"):
                    certifier._verify_loaded_libraries(mock.Mock(), runtime)

    def test_compiled_entrypoint_rejects_boundaries_and_preserves_canaries(self) -> None:
        temporary, library = _compiled_entrypoint()
        self.addCleanup(temporary.cleanup)

        def invoke(frames: int, capacity: int, *, rate=16000, channels=1, bits=16):
            info = certifier.CKAudioWavInfo(1, channels, rate, bits, frames, 0, frames * 2)
            payload = np.frombuffer(bytes(info), dtype=np.uint8)
            guard = 8
            storage = np.full(guard + capacity * 4 + guard, -1234.5, dtype=np.float32)
            output = storage[guard : guard + capacity * 4]
            produced = ctypes.c_int(-1)
            actual_info = certifier.CKAudioWavInfo()
            status = library.ck_model_prepare_audio_wav_features(
                payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                payload.size,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                capacity,
                ctypes.byref(produced),
                ctypes.byref(actual_info),
            )
            self.assertTrue(np.all(storage[:guard] == -1234.5))
            self.assertTrue(np.all(storage[-guard:] == -1234.5))
            return status, produced.value, output

        status, produced, output = invoke(160, 3)
        self.assertEqual((status, produced), (0, 2))
        np.testing.assert_array_equal(output[:4], np.full(4, 7.0, dtype=np.float32))
        np.testing.assert_array_equal(output[4:8], np.zeros(4, dtype=np.float32))

        status, produced, _ = invoke(320, 3)
        self.assertEqual((status, produced), (0, 3))
        status, produced, output = invoke(160, 3)
        self.assertEqual((status, produced), (0, 2))
        np.testing.assert_array_equal(output[4:8], np.zeros(4, dtype=np.float32))

        self.assertEqual(invoke(159, 3)[0], -4)
        self.assertEqual(invoke(321, 4)[0], -3)
        self.assertEqual(invoke(160, 1)[0], -4)
        self.assertEqual(invoke(160, 3, rate=8000)[0], -10)
        self.assertEqual(invoke(160, 3, channels=2)[0], -10)
        self.assertEqual(invoke(160, 3, bits=24)[0], -10)


if __name__ == "__main__":
    unittest.main()
