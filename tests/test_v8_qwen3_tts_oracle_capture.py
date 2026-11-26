"""Cheap validation of the pinned Qwen3-TTS oracle capture contract."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave


CAPTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "version/v8/tts/capture_qwen3_tts_customvoice_oracle.py"
)
spec = importlib.util.spec_from_file_location("qwen3_tts_oracle_capture", CAPTURE_PATH)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures/tts/qwen3_tts_0b6_customvoice"


class Qwen3TTSOracleCaptureContractTest(unittest.TestCase):
    def test_pins_model_and_reference_code(self):
        self.assertEqual(len(capture.MODEL_REVISION), 40)
        self.assertEqual(len(capture.REFERENCE_REVISION), 40)
        self.assertEqual(capture.MODEL_REPO, "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

    def test_rejects_empty_text_nul_and_invalid_generation_bounds(self):
        for text, limit in (("", 64), ("  ", 64), ("a\x00b", 64), ("hello", 1), ("hello", 4097)):
            with self.subTest(text=text, limit=limit), self.assertRaises(ValueError):
                capture.validate_request(text, limit)
        capture.validate_request("Hello, this is CKE.", 2)
        capture.validate_request("Hello, this is CKE.", 4096)

    def test_rejects_incomplete_or_wrong_model_before_loading_torch(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root)
            with self.assertRaisesRegex(FileNotFoundError, "missing pinned model files"):
                capture.verify_model_dir(model_dir)
            for name in capture.MODEL_FILE_SHA256:
                if name == "config.json":
                    continue
                path = model_dir / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            config_path = model_dir / "config.json"
            config_path.write_text(json.dumps({"tts_model_size": "1b7", "tts_model_type": "custom_voice"}))
            with self.assertRaisesRegex(ValueError, r"expected Qwen3-TTS 0\.6B custom_voice"):
                capture.verify_model_dir(model_dir)
            config_path.write_text(json.dumps({"tts_model_size": "0b6", "tts_model_type": "custom_voice"}))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                capture.verify_model_dir(model_dir)

    def test_rejects_changed_reference_code(self):
        with tempfile.TemporaryDirectory() as root:
            package_dir = Path(root)
            for name in capture.REFERENCE_FILE_SHA256:
                path = package_dir / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "reference code checksum mismatch"):
                capture.verify_reference_code(package_dir)

    def test_rejects_other_snapshot_revision(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root) / "snapshots" / ("0" * 40)
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text(
                json.dumps({"tts_model_size": "0b6", "tts_model_type": "custom_voice"})
            )
            for name in capture.MODEL_FILE_SHA256:
                if name == "config.json":
                    continue
                path = model_dir / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "snapshot revision"):
                capture.verify_model_dir(model_dir)

    def test_committed_reference_has_checked_codec_and_pcm(self):
        import numpy as np

        report = json.loads((FIXTURE_DIR / "qwen3_tts_customvoice_reference.json").read_text())
        wav_path = FIXTURE_DIR / "qwen3_tts_customvoice_reference.wav"
        npz_path = FIXTURE_DIR / "qwen3_tts_customvoice_reference.npz"
        self.assertEqual(report["scope"], "official PyTorch reference only; CKE generated execution NOT_TESTED")
        self.assertEqual(report["model"]["revision"], capture.MODEL_REVISION)
        self.assertEqual(report["model"]["files_sha256"], capture.MODEL_FILE_SHA256)
        self.assertEqual(report["reference"]["revision"], capture.REFERENCE_REVISION)
        self.assertEqual(report["reference"]["files_sha256"], capture.REFERENCE_FILE_SHA256)
        self.assertEqual(capture.sha256_file(wav_path), report["result"]["wav_sha256"])
        self.assertEqual(capture.sha256_file(npz_path), report["result"]["npz_sha256"])
        with np.load(npz_path) as arrays:
            waveform = arrays["waveform"]
            text_ids = arrays["text_token_ids"]
            context_ids = arrays["context_token_ids"]
            projection_input = arrays["context_projection_input"]
            fc1_output = arrays["context_fc1_output"]
            projection_output = arrays["context_projection_output"]
            codes = arrays["codec_tokens"]
        self.assertEqual(text_ids.size, report["result"]["text_tokens"])
        self.assertTrue(np.array_equal(context_ids, text_ids[:, 3:-5]))
        self.assertEqual(projection_input.shape, (1, report["result"]["context_tokens"], 2048))
        self.assertEqual(fc1_output.shape, projection_input.shape)
        self.assertEqual(projection_output.shape, (1, report["result"]["context_tokens"], 1024))
        self.assertTrue(np.isfinite(projection_input).all())
        self.assertTrue(np.isfinite(fc1_output).all())
        self.assertTrue(np.isfinite(projection_output).all())
        self.assertEqual(codes.shape, (report["result"]["codec_frames"], 16))
        self.assertEqual(waveform.size, report["result"]["samples"])
        self.assertTrue(np.isfinite(waveform).all())
        self.assertAlmostEqual(float(np.max(np.abs(waveform))), report["result"]["waveform_peak"])
        with wave.open(str(wav_path), "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 24000))
            self.assertEqual(wav.getnframes(), waveform.size)
            pcm = np.asarray(struct.unpack(f"<{waveform.size}h", wav.readframes(wav.getnframes())))
        self.assertLessEqual(float(np.max(np.abs(pcm / 32768.0 - waveform))), 1 / 32768)


if __name__ == "__main__":
    unittest.main()
