"""Bounded native Qwen3-TTS session input and PCM packaging contracts."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from version.v8.tts.qwen3_tts_pinned_session import (
    DEFAULT_TEXT, PinnedQwenSession, SAMPLE_RATE, SAMPLES,
    _prompt_ids, write_wav,
)


class Qwen3TTSPinnedSessionTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; repeated native session NOT_TESTED")
    def test_repeated_pinned_native_synthesis(self):
        with tempfile.TemporaryDirectory() as temp:
            session = PinnedQwenSession(
                Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"]), Path(temp),
                provider=os.environ.get("CKE_QWEN3_TTS_CODEC_CONV_PROVIDER", "scalar"))
            with self.assertRaises(ValueError):
                session.synthesize("Read issue 833.")
            first, first_metric = session.synthesize(DEFAULT_TEXT)
            second, second_metric = session.synthesize(DEFAULT_TEXT)
            np.testing.assert_array_equal(second, first)
            self.assertEqual(first_metric["frames"], 46)
            self.assertEqual(first_metric["samples"], SAMPLES)
            self.assertEqual(first_metric["load_count"], 1)
            self.assertEqual(second_metric["load_count"], 1)
            self.assertEqual(first_metric["code_sha256"],
                             "9431ba62c93b23f2327626314e6d338efdbe18ec7b733826f4fa2142f11386bf")
            self.assertEqual(first_metric["waveform_f32_sha256"],
                             "f9a8c5fae95a19888a3cadf8c909375f82ee9a0de28bd8aa93328ffcabed663f")
            path = Path(temp) / "pinned.wav"
            write_wav(path, second)
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getnframes(), SAMPLES)

    def test_unsupported_text_rejected_before_model_access(self):
        for text in ("", "Hello, this is CKE", "Read issue 833."):
            with self.subTest(text=text), self.assertRaisesRegex(
                    ValueError, "only the pinned request is certified"):
                _prompt_ids(Path("/unavailable-model"), text)
        session = PinnedQwenSession.__new__(PinnedQwenSession)
        with self.assertRaisesRegex(ValueError, "only the pinned request"):
            session.synthesize("Read issue 833.")
        self.assertEqual(DEFAULT_TEXT, "Hello, this is CKE.")

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
            for invalid in (values[:-1], values.reshape(2, -1),
                            np.full(SAMPLES, np.nan, np.float32)):
                with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                    write_wav(path, invalid)
                self.assertEqual(path.read_bytes(), original)

if __name__ == "__main__":
    unittest.main()
