"""Bounded native Qwen3-TTS session input and PCM packaging contracts."""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from version.v8.tts.qwen3_tts_pinned_session import (
    CancelCallback, DEFAULT_TEXT, PinnedQwenSession, SAMPLE_RATE, SAMPLES,
    SynthesisCancelled,
    _prompt_ids, write_wav,
)

SECOND_TEXT = "Yes, this is CKE."
SECOND_FIXTURE = (Path(__file__).resolve().parents[1] / "tests/fixtures/tts"
                  / "qwen3_tts_0b6_customvoice/qwen3_tts_yes_40frame_reference.npz")


class Qwen3TTSPinnedSessionTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; repeated native session NOT_TESTED")
    def test_repeated_pinned_native_synthesis(self):
        with tempfile.TemporaryDirectory() as temp:
            session = PinnedQwenSession(
                Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"]), Path(temp),
                provider=os.environ.get("CKE_QWEN3_TTS_CODEC_CONV_PROVIDER", "scalar"))
            checks = 0
            original_callback = session._cancel_callback

            def stop_after_first_frame(_context):
                nonlocal checks
                checks += 1
                if checks == 4:
                    session.stop()
                return int(session._cancel_requested.is_set())

            session._cancel_callback = CancelCallback(stop_after_first_frame)
            try:
                with self.assertRaises(SynthesisCancelled):
                    session.synthesize(DEFAULT_TEXT)
            finally:
                session._cancel_callback = original_callback
            self.assertGreaterEqual(checks, 4)
            with self.assertRaisesRegex(ValueError, "exactly 7 tokens"):
                session.synthesize("Hi.")
            first, first_metric = session.synthesize(DEFAULT_TEXT)
            try:
                second, second_metric = session.synthesize(SECOND_TEXT)
            except RuntimeError:
                recovered, recovered_metric = session.synthesize(DEFAULT_TEXT)
                np.testing.assert_array_equal(recovered, first)
                self.assertEqual(recovered_metric["load_count"], 1)
                raise
            with np.load(SECOND_FIXTURE) as reference:
                expected_codes = reference["codec_tokens"].copy()
                expected_waveform = reference["waveform"].copy()
            if second.shape == expected_waveform.shape:
                waveform_delta = second.astype(np.float64) - expected_waveform.astype(np.float64)
                waveform_max_abs = float(np.max(np.abs(waveform_delta)))
                waveform_rmse = float(np.sqrt(np.mean(np.square(waveform_delta))))
            else:
                waveform_max_abs = float("inf")
                waveform_rmse = float("inf")
            print("QWEN3_TTS_SECOND_TEXT_ORACLE " + json.dumps({
                "code_frames": second_metric["frames"],
                "code_sha256": second_metric["code_sha256"],
                "reference_code_sha256": hashlib.sha256(expected_codes.tobytes()).hexdigest(),
                "waveform_max_abs": waveform_max_abs,
                "waveform_rmse": waveform_rmse,
            }, sort_keys=True), flush=True)
            third, third_metric = session.synthesize(DEFAULT_TEXT)
            np.testing.assert_array_equal(third, first)
            self.assertEqual(first_metric["frames"], 46)
            self.assertEqual(first_metric["samples"], SAMPLES)
            self.assertEqual(first_metric["load_count"], 1)
            self.assertEqual(second_metric["load_count"], 1)
            self.assertEqual(third_metric["load_count"], 1)
            self.assertGreater(len(second), 0)
            self.assertLessEqual(len(second), SAMPLES)
            self.assertEqual(len(second), second_metric["samples"])
            self.assertNotEqual(first_metric["code_sha256"], second_metric["code_sha256"])
            self.assertEqual(second_metric["frames"], 40)
            self.assertEqual(second.shape, expected_waveform.shape)
            self.assertEqual(second_metric["code_sha256"],
                             hashlib.sha256(expected_codes.tobytes()).hexdigest())
            self.assertLessEqual(waveform_max_abs, 3e-6)
            self.assertEqual(first_metric["code_sha256"],
                             "9431ba62c93b23f2327626314e6d338efdbe18ec7b733826f4fa2142f11386bf")
            self.assertEqual(first_metric["waveform_f32_sha256"],
                             "f9a8c5fae95a19888a3cadf8c909375f82ee9a0de28bd8aa93328ffcabed663f")
            path = Path(temp) / "pinned.wav"
            write_wav(path, third)
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getnframes(), SAMPLES)

    def test_unsupported_text_rejected_before_model_access(self):
        for text in ("", " ", "Hello\nCKE"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "nonempty"):
                _prompt_ids(Path("/unavailable-model"), text)
        with self.assertRaisesRegex(ValueError, "special-token syntax"):
            _prompt_ids(Path("/unavailable-model"), "<|im_start|> hello")

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3] if text.startswith("<|im_start|>") else [1]

        with self.assertRaisesRegex(ValueError, "exactly 7 tokens"):
            _prompt_ids(Path("/unavailable-model"), "Hi.", FakeTokenizer())
        self.assertEqual(DEFAULT_TEXT, "Hello, this is CKE.")

    def test_request_text_selects_distinct_prompt_ids(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text.startswith("<|im_start|>"):
                    return [1, 2, 3]
                return [10, 11, 12, 13, 14, 15, 16 if text == DEFAULT_TEXT else 17]

        config = {
            "tts_pad_token_id": 20, "tts_bos_token_id": 21,
            "tts_eos_token_id": 22,
            "talker_config": {
                "codec_think_id": 30, "codec_think_bos_id": 31,
                "codec_language_id": {"english": 32},
                "codec_think_eos_id": 33, "spk_id": {"ryan": 34},
                "codec_pad_id": 35, "codec_bos_id": 36,
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            (model_dir / "config.json").write_text(json.dumps(config))
            first, first_codec = _prompt_ids(model_dir, DEFAULT_TEXT, FakeTokenizer())
            second, second_codec = _prompt_ids(model_dir, SECOND_TEXT, FakeTokenizer())
        self.assertEqual(first.shape, (13,))
        np.testing.assert_array_equal(first[:-1], second[:-1])
        self.assertEqual((first[-1], second[-1]), (16, 17))
        np.testing.assert_array_equal(first_codec, second_codec)

    def test_second_reference_fixture_integrity(self):
        manifest = json.loads(SECOND_FIXTURE.with_suffix(".json").read_text())
        baseline = json.loads((SECOND_FIXTURE.parent /
                               "qwen3_tts_customvoice_reference.json").read_text())
        self.assertEqual(hashlib.sha256(SECOND_FIXTURE.read_bytes()).hexdigest(),
                         manifest["fixture_sha256"])
        self.assertEqual(manifest["text"], SECOND_TEXT)
        self.assertEqual(manifest["seed"], 20261022)
        self.assertEqual(manifest["max_new_tokens"], 64)
        self.assertFalse(manifest["do_sample"])
        self.assertFalse(manifest["subtalker_dosample"])
        self.assertEqual(manifest["model_revision"],
                         "85e237c12c027371202489a0ec509ded67b5e4b5")
        self.assertEqual(manifest["reference_revision"],
                         "022e286b98fbec7e1e916cb940cdf532cd9f488e")
        self.assertEqual(manifest["model_files_sha256"],
                         baseline["model"]["files_sha256"])
        self.assertEqual(manifest["reference_files_sha256"],
                         baseline["reference"]["files_sha256"])
        self.assertEqual(manifest["runtime"]["transformers"], "4.57.3")
        self.assertEqual(manifest["runtime"]["torch"], "2.12.1+cpu")
        with np.load(SECOND_FIXTURE) as fixture:
            self.assertEqual(fixture["codec_tokens"].shape, (40, 16))
            self.assertEqual(fixture["waveform"].shape, (40 * 1920,))
            self.assertTrue(np.isfinite(fixture["waveform"]).all())

    def test_pcm_wav_header_and_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "speech.wav"
            values = np.zeros(SAMPLES, dtype=np.float32)
            values[0], values[1], values[2] = -1, 0.5, 1
            write_wav(path, values)
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getnchannels(), 1)
                self.assertEqual(audio.getsampwidth(), 2)
                self.assertEqual(audio.getframerate(), SAMPLE_RATE)
                self.assertEqual(audio.getnframes(), SAMPLES)
                first = np.frombuffer(audio.readframes(3), dtype="<i2")
                np.testing.assert_array_equal(first, [-32767, 16384, 32767])
            original = path.read_bytes()
            shorter = values[:SAMPLES // 46]
            write_wav(path, shorter)
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getnframes(), len(shorter))
            path.write_bytes(original)
            for invalid in (values[:-1], values.reshape(2, -1),
                            np.full(SAMPLES, np.nan, np.float32)):
                with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                    write_wav(path, invalid)
                self.assertEqual(path.read_bytes(), original)

if __name__ == "__main__":
    unittest.main()
