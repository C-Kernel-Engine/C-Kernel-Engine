"""Independent PyTorch oracle and capacity checks for the native inverse STFT."""

import ctypes
import json
import math
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
            self.assertTrue(all(math.isfinite(value) for value in output))
            self.assertTrue(all(math.isfinite(value) for value in expected))
            errors = [abs(a - b) for a, b in zip(output, expected)]
            worst = max(range(len(errors)), key=errors.__getitem__)
            self.assertLessEqual(errors[worst], fixture["atol"],
                                 f"worst sample {worst}: native={output[worst]}, "
                                 f"oracle={expected[worst]}, error={errors[worst]}")

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
            self.assertTrue(np.isfinite(output).all())
            self.assertTrue(np.isfinite(expected).all())
            errors = np.abs(output - expected)
            worst = int(np.argmax(errors))
            self.assertLessEqual(float(errors[worst]), 2e-5,
                                 f"worst sample {worst}: native={output[worst]}, "
                                 f"oracle={expected[worst]}, error={errors[worst]}")

            output.fill(-999.0)
            too_small = args[:-1] + (scratch.size - 1,)
            self.assertEqual(native.audio_istft_mag_phase_f32(*too_small), -2)
            self.assertTrue(np.all(output == -999.0))
            self.assertEqual(native.audio_istft_mag_phase_f32(
                *args[:7], output.size - 1, *args[8:]), -2)
            self.assertTrue(np.all(output == -999.0))


class AudioIstftSafetyTest(unittest.TestCase):
    """Portable geometry and rejection checks; PyTorch is not required at test time."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        library = Path(cls.temp_dir.name) / "libaudio_istft_safety.so"
        subprocess.run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-shared", "-fPIC", "-I", str(ROOT / "include"),
            str(ROOT / "src" / "kernels" / "audio_istft_mag_phase.c"),
            "-lm", "-o", str(library),
        ], check=True)
        native = ctypes.CDLL(str(library))
        size = ctypes.c_size_t
        fptr = ctypes.POINTER(ctypes.c_float)
        native.audio_istft_mag_phase_plan_f32.argtypes = [
            size, size, size, ctypes.POINTER(size), ctypes.POINTER(size), ctypes.POINTER(size)]
        native.audio_istft_mag_phase_plan_f32.restype = ctypes.c_int
        native.audio_istft_mag_phase_f32.argtypes = [
            fptr, fptr, size, size, size, size, fptr, size, fptr, size]
        native.audio_istft_mag_phase_f32.restype = ctypes.c_int
        cls.native = native

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def _buffers(self, frames=4, n_fft=8, hop=2):
        bins = n_fft // 2 + 1
        spectrum = frames * bins
        samples = (frames - 1) * hop
        scratch_count = 2 * (samples + n_fft) + n_fft
        magnitude = (ctypes.c_float * spectrum)(*([0.5] * spectrum))
        phase = (ctypes.c_float * spectrum)(*([0.0] * spectrum))
        output = (ctypes.c_float * samples)(*([-777.0] * samples))
        scratch = (ctypes.c_float * scratch_count)()
        args = [magnitude, phase, spectrum, frames, n_fft, hop,
                output, samples, scratch, scratch_count]
        return args

    def test_checked_rejection_preserves_output(self):
        baseline = self._buffers()
        spectrum, samples, scratch_count = baseline[2], baseline[7], baseline[9]
        size_max = ctypes.c_size_t(-1).value
        cases = [
            ("short_spectrum", {2: spectrum - 1}, -2),
            ("short_output", {7: samples - 1}, -2),
            ("short_scratch", {9: scratch_count - 1}, -2),
            ("zero_frames", {3: 0}, -1),
            ("one_frame", {3: 1}, -1),
            ("zero_fft", {4: 0}, -1),
            ("odd_fft", {4: 7}, -1),
            ("zero_hop", {5: 0}, -1),
            ("excess_hop", {5: 5}, -1),
            ("spectrum_overflow", {3: size_max}, -3),
            ("output_overflow", {3: size_max // 5 + 1, 5: 4}, -3),
        ]
        for name, replacements, expected_status in cases:
            with self.subTest(name=name):
                args = self._buffers()
                for index, value in replacements.items():
                    args[index] = value
                self.assertEqual(self.native.audio_istft_mag_phase_f32(*args),
                                 expected_status)
                self.assertEqual(list(args[6]), [-777.0] * samples)

        for name, array_index, value in [
            ("negative_magnitude", 0, -0.1),
            ("nan_magnitude", 0, float("nan")),
            ("infinite_magnitude", 0, float("inf")),
            ("nan_phase", 1, float("nan")),
            ("infinite_phase", 1, float("inf")),
        ]:
            with self.subTest(name=name):
                args = self._buffers()
                args[array_index][-1] = value
                self.assertEqual(self.native.audio_istft_mag_phase_f32(*args), -1)
                self.assertEqual(list(args[6]), [-777.0] * samples)

    def test_planner_rejects_invalid_and_overflow(self):
        size = ctypes.c_size_t
        for frames, n_fft, hop, status in [
            (1, 8, 2, -1), (2, 7, 2, -1), (2, 8, 0, -1),
            (2, 8, 5, -1), (size(-1).value, 8, 2, -3),
        ]:
            with self.subTest(frames=frames, n_fft=n_fft, hop=hop):
                a, b, c = size(91), size(92), size(93)
                self.assertEqual(self.native.audio_istft_mag_phase_plan_f32(
                    frames, n_fft, hop, ctypes.byref(a), ctypes.byref(b),
                    ctypes.byref(c)), status)
                self.assertEqual((a.value, b.value, c.value), (91, 92, 93))

    def test_committed_torch_geometry_fixtures(self):
        fixture = json.loads((ROOT / "tests" / "fixtures" / "tts" /
                              "istft_geometry_torch.json").read_text())
        for case in fixture["cases"]:
            with self.subTest(case=case["name"]):
                frames, n_fft, hop = case["frames"], case["n_fft"], case["hop"]
                bins = n_fft // 2 + 1
                args = self._buffers(frames, n_fft, hop)
                args[0][:] = [v for row in case["magnitude"] for v in row]
                args[1][:] = [v for row in case["phase"] for v in row]
                self.assertEqual(len(args[0]), frames * bins)
                self.assertEqual(self.native.audio_istft_mag_phase_f32(*args), 0)
                actual = list(args[6])
                expected = case["waveform"]
                self.assertEqual(len(actual), len(expected))
                self.assertTrue(all(math.isfinite(v) for v in actual))
                self.assertTrue(all(math.isfinite(v) for v in expected))
                errors = [abs(a - b) for a, b in zip(actual, expected)]
                worst = max(range(len(errors)), key=errors.__getitem__)
                self.assertLessEqual(errors[worst], case["atol"],
                                     f"{case['name']} worst sample {worst}: "
                                     f"native={actual[worst]}, oracle={expected[worst]}, "
                                     f"error={errors[worst]}")


if __name__ == "__main__":
    unittest.main()
