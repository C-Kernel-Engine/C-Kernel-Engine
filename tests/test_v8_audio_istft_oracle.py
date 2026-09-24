"""Independent PyTorch oracle and capacity checks for the native inverse STFT."""

import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AudioIstftOracleTest(unittest.TestCase):
    def test_committed_torch_fixture(self):
        fixture = json.loads((ROOT / "tests" / "fixtures" / "tts" /
                              "istft_torch20_hop5.json").read_text())
        frames, n_fft, hop = fixture["frames"], fixture["n_fft"], fixture["hop"]
        bins = n_fft // 2 + 1
        mag = (ctypes.c_float * (frames * bins))(
            *(x for row in fixture["magnitude"] for x in row))
        phase = (ctypes.c_float * (frames * bins))(
            *(x for row in fixture["phase"] for x in row))
        expected = fixture["waveform"]
        with tempfile.TemporaryDirectory() as temp_dir:
            library = Path(temp_dir) / "libaudio_istft_test.so"
            subprocess.run([
                "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
                "-shared", "-fPIC", "-I", str(ROOT / "include"),
                str(ROOT / "src" / "kernels" / "audio_istft_mag_phase.c"),
                "-lm", "-o", str(library),
            ], check=True)
            native = ctypes.CDLL(str(library))
            native.audio_istft_mag_phase_f32.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ]
            native.audio_istft_mag_phase_f32.restype = ctypes.c_int
            output = (ctypes.c_float * len(expected))()
            scratch = (ctypes.c_float * (2 * (len(expected) + n_fft) + n_fft))()
            status = native.audio_istft_mag_phase_f32(
                mag, phase, frames * bins, frames, n_fft, hop,
                output, len(expected), scratch, len(scratch))
            self.assertEqual(status, 0)
            self.assertLessEqual(max(abs(a - b) for a, b in zip(output, expected)),
                                 fixture["atol"])

    def test_centered_hann_against_torch(self):
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest(f"live PyTorch oracle dependency unavailable: {exc}")

        torch.manual_seed(17)
        frames, n_fft, hop = 31, 20, 5
        bins = n_fft // 2 + 1
        magnitude = torch.rand((frames, bins), dtype=torch.float32) + 0.1
        phase = (torch.rand((frames, bins), dtype=torch.float32) * 2 - 1) * torch.pi
        spectrum = torch.polar(magnitude, phase).transpose(0, 1).contiguous()
        expected = torch.istft(
            spectrum, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=torch.hann_window(n_fft, periodic=True), center=True,
        ).numpy()

        with tempfile.TemporaryDirectory() as temp_dir:
            library = Path(temp_dir) / "libaudio_istft_test.so"
            subprocess.run([
                "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
                "-shared", "-fPIC", "-I", str(ROOT / "include"),
                str(ROOT / "src" / "kernels" / "audio_istft_mag_phase.c"),
                "-lm", "-o", str(library),
            ], check=True)
            native = ctypes.CDLL(str(library))
            size = ctypes.c_size_t
            native.audio_istft_mag_phase_plan_f32.argtypes = [
                size, size, size, ctypes.POINTER(size), ctypes.POINTER(size), ctypes.POINTER(size),
            ]
            native.audio_istft_mag_phase_plan_f32.restype = ctypes.c_int
            native.audio_istft_mag_phase_f32.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                size, size, size, size, ctypes.POINTER(ctypes.c_float), size,
                ctypes.POINTER(ctypes.c_float), size,
            ]
            native.audio_istft_mag_phase_f32.restype = ctypes.c_int

            spectral_elements, output_samples, scratch_elements = size(), size(), size()
            self.assertEqual(native.audio_istft_mag_phase_plan_f32(
                frames, n_fft, hop, ctypes.byref(spectral_elements),
                ctypes.byref(output_samples), ctypes.byref(scratch_elements),
            ), 0)
            self.assertEqual(spectral_elements.value, frames * bins)
            self.assertEqual(output_samples.value, len(expected))
            self.assertEqual(scratch_elements.value, 2 * (len(expected) + n_fft) + n_fft)

            mag = np.ascontiguousarray(magnitude.numpy())
            ph = np.ascontiguousarray(phase.numpy())
            output = np.full(output_samples.value, -999.0, dtype=np.float32)
            scratch = np.zeros(scratch_elements.value, dtype=np.float32)
            ptr = lambda array: array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            args = (ptr(mag), ptr(ph), spectral_elements.value,
                    frames, n_fft, hop, ptr(output), output.size,
                    ptr(scratch), scratch.size)
            self.assertEqual(native.audio_istft_mag_phase_f32(*args), 0)
            np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)

            output.fill(-999.0)
            too_small = args[:-1] + (scratch.size - 1,)
            self.assertEqual(native.audio_istft_mag_phase_f32(*too_small), -2)
            self.assertTrue(np.all(output == -999.0))
            self.assertEqual(native.audio_istft_mag_phase_f32(
                *args[:7], output.size - 1, *args[8:]), -2)
            self.assertTrue(np.all(output == -999.0))


if __name__ == "__main__":
    unittest.main()
