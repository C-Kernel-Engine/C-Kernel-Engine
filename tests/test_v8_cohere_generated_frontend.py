from __future__ import annotations

import ctypes
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
CIRCUIT = ROOT / "version" / "v8" / "circuits" / "cohere_transcribe.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codegen = _load("cohere_generated_frontend_codegen", SCRIPTS / "codegen_v8.py")
certifier = _load(
    "cohere_generated_frontend_common_certifier",
    SCRIPTS / "certify_parakeet_generated_frontend_v8.py",
)


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


def _call(operation: str, function: str, args: list[tuple[str, str]]) -> dict:
    return {
        "op": operation,
        "function": function,
        "args": [
            {"name": f"arg_{index}", "source": source, "expr": expression}
            for index, (source, expression) in enumerate(args)
        ],
    }


def _operations() -> list[dict]:
    return [
        _call(
            "audio_wav_decode",
            "stub_decode",
            [
                ("runtime:audio_wav_bytes", "audio_wav_bytes"),
                ("runtime:audio_wav_byte_count", "audio_wav_byte_count"),
                ("runtime:audio_window_start_frame", "audio_window_start_frame"),
                ("runtime:audio_mono", "audio_mono"),
                ("runtime:audio_mono_capacity", "audio_mono_capacity"),
                ("runtime:audio_wav_info", "audio_wav_info"),
            ],
        ),
        _call("audio_preemphasis", "stub_preemphasis", [("dim:frames", "configured_frames")]),
        _call("audio_stft_tables", "stub_zero", []),
        _call(
            "audio_stft",
            "stub_stft",
            [("dim:n_samples", "configured_samples"), ("dim:n_frames", "configured_frames")],
        ),
        _call("audio_log_mel", "stub_log_mel", [("dim:frames", "configured_frames")]),
        _call(
            "audio_feature_normalize",
            "stub_normalize",
            [("dim:frames", "configured_frames"), ("output:output", "configured_output")],
        ),
    ]


def _config(frame_policy: str = "all_stft_frames") -> dict[str, object]:
    return {
        "audio_sample_rate": 16000,
        "audio_max_source_frames": 320,
        "audio_hop_length": 160,
        "audio_feature_channels": 4,
        "audio_frontend_asset_policy": "model_assets",
        "audio_normalization_frame_policy": frame_policy,
    }


def _compiled_entrypoint() -> tuple[tempfile.TemporaryDirectory, ctypes.CDLL]:
    entrypoint = codegen._emit_audio_wav_entrypoint(_operations(), _config())
    source = f"""
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#define CK_EXPORT __attribute__((visibility("default")))
typedef struct {{
    int format_tag; int channels; int sample_rate; int bits_per_sample;
    int frames; size_t data_offset; size_t data_bytes;
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
static int stub_decode(const uint8_t *bytes, size_t count, int start, float *output,
                       int capacity, CKAudioWavInfo *info) {{
    (void)bytes; (void)count; (void)start; (void)output; (void)capacity;
    return info->frames;
}}
static int stub_zero(void) {{ return 0; }}
static int stub_preemphasis(int frames) {{ return frames > 0 ? 0 : -1; }}
static int stub_stft(int samples, int frames) {{ return samples > 0 && frames > 0 ? 0 : -1; }}
static int stub_log_mel(int frames) {{ return frames > 0 ? 0 : -1; }}
static int stub_normalize(int frames, float *output) {{
    for (int frame = 0; frame < frames; ++frame)
        for (int channel = 0; channel < 4; ++channel)
            output[frame * 4 + channel] = 7.0f;
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
        ctypes.POINTER(CKAudioWavInfo),
    ]
    library.ck_model_prepare_audio_wav_features.restype = ctypes.c_int
    return temporary, library


class GeneratedCohereFrontendTests(unittest.TestCase):
    def test_cohere_fixture_key_requires_fp32_frame_channel_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.npz"
            expected = np.ones((1, 2, 4), dtype=np.float32)
            np.savez(path, **{"audio.frontend.log_mel.output": expected})
            actual = certifier._load_expected_fixture(
                path, 4, "audio.frontend.log_mel.output"
            )
            np.testing.assert_array_equal(actual, expected[0])

    def test_circuit_declares_model_assets_and_exact_frame_policy(self) -> None:
        circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
        frontend = circuit["block_types"]["frontend"]["header"]
        by_op = {row["op"]: row for row in frontend}
        self.assertEqual(by_op["audio_stft"]["weight_refs"]["window"], "fe.window")
        self.assertEqual(
            by_op["audio_log_mel"]["weight_refs"]["mel_filters"], "fe.mel_fb"
        )
        self.assertEqual(
            circuit["contract"]["audio_frontend"]["normalization"],
            "per_feature_sample_variance_all_stft_frames",
        )
        ignored = circuit["contract"]["weight_policy"]["ignore"]
        frontend_ignored = {
            row["pattern"]
            for row in ignored
            if row.get("when", {}).get("equals") == "audio_frontend"
        }
        self.assertEqual(frontend_ignored, {"enc.*", "dec.*"})

    def test_model_asset_recipe_does_not_synthesize_window_or_filter(self) -> None:
        emitted = codegen._emit_audio_wav_entrypoint(_operations(), _config())
        self.assertNotIn("audio_hann_window", emitted)
        self.assertNotIn("audio_mel_filters", emitted)
        self.assertIn("stub_normalize(required_frames, audio_features)", emitted)

    def test_unknown_normalization_frame_policy_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsupported.*frame_policy"):
            codegen._emit_audio_wav_entrypoint(_operations(), _config("guess"))

    def test_compiled_entrypoint_normalizes_every_stft_frame(self) -> None:
        temporary, library = _compiled_entrypoint()
        self.addCleanup(temporary.cleanup)
        info = CKAudioWavInfo(1, 1, 16000, 16, 160, 0, 320)
        payload = np.frombuffer(bytes(info), dtype=np.uint8)
        guard = 8
        storage = np.full(guard + 3 * 4 + guard, -1234.5, dtype=np.float32)
        output = storage[guard:-guard]
        produced = ctypes.c_int(-1)
        actual_info = CKAudioWavInfo()
        status = library.ck_model_prepare_audio_wav_features(
            payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            payload.size,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            3,
            ctypes.byref(produced),
            ctypes.byref(actual_info),
        )
        self.assertEqual((status, produced.value), (0, 2))
        np.testing.assert_array_equal(output[:8], np.full(8, 7.0, dtype=np.float32))
        self.assertTrue(np.all(storage[:guard] == -1234.5))
        self.assertTrue(np.all(storage[-guard:] == -1234.5))


if __name__ == "__main__":
    unittest.main()
