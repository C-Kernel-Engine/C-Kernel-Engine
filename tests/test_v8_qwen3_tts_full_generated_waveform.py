"""Pinned Qwen3-TTS natural-EOS codes through the generated 46-frame codec."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from tests.test_v8_checked_frame_loop import Loop
from tests.test_v8_qwen3_tts_codec_waveform import CHECKPOINT
from tests.test_v8_qwen3_tts_first_frame_waveform import (
    CODEC_SOURCES, _codec_value, _weight_shapes,
)
from tests.test_v8_qwen3_tts_loop_codec import FrameCodec
from tests.test_v8_qwen3_tts_prompt_first_block import (
    FIXTURE, compile_entry, pinned_prompt_ids, view,
)
from tests.test_v8_qwen3_tts_second_generated_frame import prepare_inputs
from tests.test_v8_qwen3_tts_two_phase_speech import SOURCES, segment_function
from version.v8.tts.build_qwen3_tts_loop_codec_circuit import build_circuit

FRAMES = 46
CODEBOOKS = 16
SAMPLES = 88320
MAX_FRAMES = 64
SOURCES_ALL = tuple(dict.fromkeys((*SOURCES, *CODEC_SOURCES,
                                   "src/runtime/checked_frame_loop.c",
                                   "src/runtime/checked_frame_codec.c")))


class Qwen3TTSFullGeneratedWaveformTest(unittest.TestCase):
    def test_declared_full_geometry_and_fixture_identity(self):
        graph = build_circuit(frames=FRAMES, max_generated_frames=MAX_FRAMES)
        self.assertEqual(graph["activation_buffers"]["loop_frame_major_codes"]["shape"],
                         [FRAMES * CODEBOOKS])
        self.assertEqual(graph["activation_buffers"]["codec_waveform_output"]["shape"],
                         [1, SAMPLES])
        self.assertEqual(len(graph["native_entry"]["segments"]), 3)
        self.assertEqual(graph["contract"]["runtime_invariants"]["loop_codec_frames"],
                         FRAMES)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        request_manifest = json.loads((FIXTURE / "qwen3_tts_customvoice_reference.json").read_text())
        self.assertEqual(manifest["model_revision"], request_manifest["model"]["revision"])
        self.assertEqual(manifest["reference_revision"],
                         request_manifest["reference"]["revision"])
        self.assertEqual(manifest["codec_source_sha256"],
                         request_manifest["reference"]["files_sha256"][
                             "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"])
        self.assertEqual(hashlib.sha256(
            (FIXTURE / "qwen3_tts_customvoice_reference.npz").read_bytes()).hexdigest(),
            request_manifest["result"]["npz_sha256"])
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        with np.load(FIXTURE / "qwen3_tts_customvoice_reference.npz") as request, \
             np.load(CHECKPOINT) as oracle:
            codes = np.ascontiguousarray(request["codec_tokens"].astype(np.int32))
            self.assertEqual(codes.shape, (FRAMES, CODEBOOKS))
            codec_oracle_layout = np.ascontiguousarray(
                request["codec_tokens"].T[None], dtype=np.int64)
            self.assertEqual(hashlib.sha256(codec_oracle_layout.tobytes()).hexdigest(),
                             manifest["full_codec_ids_sha256"])
            self.assertEqual(oracle["full_waveform_output"].shape, (1, SAMPLES))

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; full waveform NOT_TESTED")
    def test_pinned_natural_eos_to_waveform(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        provider = os.environ.get("CKE_QWEN3_TTS_CODEC_CONV_PROVIDER", "scalar")
        graph = build_circuit(frames=FRAMES, max_generated_frames=MAX_FRAMES,
                              codec_conv_provider=provider)
        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as main, \
             safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as codec, \
             np.load(FIXTURE / "qwen3_tts_customvoice_reference.npz") as request_oracle, \
             np.load(CHECKPOINT) as waveform_oracle, \
             tempfile.TemporaryDirectory() as temp:
            shapes = _weight_shapes(graph, main, codec)
            _, acts, weights, layout, calls, library = compile_entry(
                Path(temp), graph, shapes, extra_sources=SOURCES_ALL)
            self.assertEqual(len(calls["entry"]["segments"]), 3)
            if provider == "avx2_fma":
                available = ctypes.CDLL(str(library)).audio_conv1d_checked_avx2_fma_available
                available.argtypes = []
                available.restype = ctypes.c_int
                if not available():
                    self.skipTest("AVX2/FMA codec provider unavailable on this CPU")
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for name, item in weights.items():
                value = (_codec_value(codec, name) if name.startswith("codec.")
                         else np.ascontiguousarray(main.get_tensor(name).float().numpy()))
                self.assertEqual(value.shape, shapes[name], name)
                arena[item["abs_offset"]:item["abs_offset"] + value.nbytes] = value.tobytes()
            text_ids, codec_ids = pinned_prompt_ids(model_dir, request_oracle)
            prepare_inputs(arena, acts)
            view(arena, acts, "text_ids", np.int32, text_ids.shape)[:] = text_ids
            view(arena, acts, "codec_ids", np.int32, codec_ids.shape)[:] = codec_ids
            first = ["talker_first_code_id", "pred_predictor_codebook0_selected_id"] + [
                f"pred_step{i}_selected_id" for i in range(1, 15)]
            later = ["second_talker_first_code_id",
                     "second_predictor_codebook0_selected_id"] + [
                f"second_step{i}_selected_id" for i in range(1, 15)]
            first_offsets = (ctypes.c_size_t * CODEBOOKS)(
                *(acts[name]["abs_offset"] for name in first))
            later_offsets = (ctypes.c_size_t * CODEBOOKS)(
                *(acts[name]["abs_offset"] for name in later))
            staging = np.full(MAX_FRAMES * CODEBOOKS, -901, np.int32)
            codes = np.full(MAX_FRAMES * CODEBOOKS, -901, np.int32)
            samples = np.full(SAMPLES + 64, -901, np.float32)
            valid_frames = ctypes.c_size_t(99)
            valid_samples = ctypes.c_size_t(99)
            loop = Loop(
                ctypes.cast(segment_function(library, graph, 0), ctypes.c_void_p),
                ctypes.cast(segment_function(library, graph, 1), ctypes.c_void_p),
                arena, size, acts["talker_step_index_input"]["abs_offset"],
                acts["pred_predictor_valid_code_count"]["abs_offset"],
                acts["second_predictor_valid_code_count"]["abs_offset"],
                first_offsets, later_offsets, CODEBOOKS, MAX_FRAMES,
                staging.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(staging),
                codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(codes),
                ctypes.pointer(valid_frames))
            codec_request = FrameCodec(
                ctypes.pointer(loop),
                ctypes.cast(segment_function(library, graph, 2), ctypes.c_void_p),
                acts["loop_frame_major_codes"]["abs_offset"], CODEBOOKS, FRAMES,
                acts["codec_waveform_output"]["abs_offset"],
                acts["waveform_valid_samples"]["abs_offset"], SAMPLES,
                samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(samples),
                ctypes.pointer(valid_samples))
            run = ctypes.CDLL(str(library)).ck_run_checked_frame_codec
            run.argtypes = [ctypes.POINTER(FrameCodec)]
            run.restype = ctypes.c_int
            loop.arena_bytes = size - 1
            self.assertEqual(run(ctypes.byref(codec_request)), -2)
            self.assertEqual((valid_frames.value, valid_samples.value), (0, 0))
            np.testing.assert_array_equal(samples, -901)
            loop.arena_bytes = size
            start = time.monotonic()
            self.assertEqual(run(ctypes.byref(codec_request)), 0)
            seconds = time.monotonic() - start
            self.assertEqual((valid_frames.value, valid_samples.value), (FRAMES, SAMPLES))
            reference_codes = request_oracle["codec_tokens"].astype(np.int32)
            np.testing.assert_array_equal(codes[:FRAMES * CODEBOOKS].reshape(
                FRAMES, CODEBOOKS), reference_codes)
            np.testing.assert_array_equal(codes[FRAMES * CODEBOOKS:], -901)
            np.testing.assert_array_equal(samples[SAMPLES:], -901)
            self.assertTrue(np.isfinite(samples[:SAMPLES]).all())
            reference_samples = waveform_oracle["full_waveform_output"].reshape(-1)
            delta = samples[:SAMPLES].astype(np.float64) - reference_samples.astype(np.float64)
            worst = int(np.argmax(np.abs(delta)))
            maximum = float(np.max(np.abs(delta)))
            metric = {
                "code_frames": FRAMES, "codebooks": CODEBOOKS,
                "codec_conv_provider": provider,
                "code_sha256": hashlib.sha256(
                    codes[:FRAMES * CODEBOOKS].tobytes()).hexdigest(),
                "samples": SAMPLES, "sample_rate_hz": 24000,
                "audio_seconds": SAMPLES / 24000,
                "peak": float(np.max(np.abs(samples[:SAMPLES]))),
                "waveform_f32_sha256": hashlib.sha256(
                    samples[:SAMPLES].tobytes()).hexdigest(),
                "max_abs": maximum,
                "rmse": float(np.sqrt(np.mean(np.square(delta)))),
                "worst_index": worst,
                "actual": float(samples[worst]),
                "reference": float(reference_samples[worst]),
                "run_seconds": round(seconds, 2),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            }
            print("QWEN3_TTS_FULL_GENERATED_WAVEFORM " + json.dumps(metric, sort_keys=True))
            self.assertLessEqual(maximum, 3e-6)


if __name__ == "__main__":
    unittest.main()
