"""Checked channelwise Snake against committed and live independent PyTorch."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/tts/audio_snake_pytorch28_reference.npz'
FLOAT = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t


class CheckedAudioSnakeOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.functions = []
        for flag in ('-O0', '-O3'):
            library = Path(cls.temporary.name) / f'snake_{flag}.so'
            subprocess.run(['cc', '-std=c11', flag, '-Wall', '-Wextra',
                '-Werror', '-pedantic', '-shared', '-fPIC', '-I',
                str(ROOT / 'include'),
                str(ROOT / 'src/kernels/audio_snake_checked.c'),
                '-lm', '-o', str(library)], check=True, capture_output=True)
            function = ctypes.CDLL(str(library)).audio_snake_strided_f32_checked
            function.argtypes = [FLOAT, SIZE, SIZE, FLOAT, SIZE,
                                 FLOAT, SIZE, SIZE, SIZE, SIZE]
            function.restype = ctypes.c_int
            cls.functions.append(function)
        cls.metadata = json.loads(FIXTURE.with_suffix('.json').read_text())
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != cls.metadata['fixture_sha256']:
            raise RuntimeError('Snake reference fixture checksum mismatch')
        with np.load(FIXTURE) as archive:
            cls.arrays = {name: archive[name].copy() for name in archive.files}
        for name, array in cls.arrays.items():
            if hashlib.sha256(array.tobytes()).hexdigest() != cls.metadata['array_sha256'][name]:
                raise RuntimeError(f'Snake reference array checksum mismatch: {name}')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def pointer(array):
        return array.ctypes.data_as(FLOAT)

    def call(self, function, source, alpha, output, frames,
             input_capacity=None, alpha_capacity=None, output_capacity=None):
        return function(self.pointer(source),
            source.size if input_capacity is None else input_capacity,
            source.shape[1], self.pointer(alpha),
            alpha.size if alpha_capacity is None else alpha_capacity,
            self.pointer(output),
            output.size if output_capacity is None else output_capacity,
            output.shape[1], source.shape[0], frames)

    def test_bounded_preflight_sine_count_extremes_and_atomic_rejection(self):
        # Link wrapping counts real libm calls: skipping validation or replacing
        # sinf with an approximation cannot satisfy the arithmetic and rejection
        # assertions. The former two-pass implementation fails the ordinary count.
        wrapper = Path(self.temporary.name) / "count_sine.c"
        wrapper.write_text("""
#include <stddef.h>
#include <math.h>
static size_t calls;
float __real_sinf(float);
float __wrap_sinf(float x) { ++calls; return __real_sinf(x); }
size_t sine_calls(void) { return calls; }
void clear_calls(void) { calls = 0; }
""")
        for flag in ("-O0", "-O3"):
            libpath = Path(self.temporary.name) / f"count_sine_{flag}.so"
            subprocess.run(["cc", "-std=c11", flag, "-ffp-contract=off",
                "-fno-builtin-sinf", "-shared", "-fPIC", "-I", str(ROOT / "include"),
                str(ROOT / "src/kernels/audio_snake_checked.c"), str(wrapper),
                "-Wl,--wrap=sinf", "-lm", "-o", str(libpath)], check=True)
            lib = ctypes.CDLL(str(libpath))
            lib.sine_calls.restype = SIZE
            snake = lib.audio_snake_strided_f32_checked
            snake.argtypes = self.functions[0].argtypes
            beta = lib.audio_snake_beta_strided_f32_checked
            beta.argtypes = [FLOAT, SIZE, SIZE, FLOAT, SIZE, FLOAT, SIZE,
                             FLOAT, SIZE, SIZE, SIZE, SIZE]
            sine = ctypes.CDLL("libm.so.6").sinf
            sine.argtypes = [ctypes.c_float]; sine.restype = ctypes.c_float
            # The last two values require the exact fallback; their correction
            # rounds away. Signed zero and padded cells are checked bitwise.
            source = np.array([[0., -0., .25, -2., 4., 1e38, -1e38, np.nan]], np.float32)
            alpha = np.array([.5], np.float32)
            log_alpha = np.array([0.], np.float32)
            log_beta = np.array([0.], np.float32)
            for use_beta in (False, True):
                def invoke(data, output):
                    if use_beta:
                        return beta(self.pointer(data), data.size, 8,
                            self.pointer(log_alpha), 1, self.pointer(log_beta), 1,
                            self.pointer(output), output.size, 9, 1, 7)
                    return self.call(snake, data, alpha, output, 7)
                expected = []
                for x in source[0, :7]:
                    argument = np.float32(x * (1. if use_beta else .5))
                    v = np.float32(sine(float(argument)))
                    periodic = np.float32(np.float32(1. if use_beta else 2.) * np.float32(v*v))
                    expected.append(np.float32(x + periodic))
                output = np.full((1, 9), -91., np.float32)
                lib.clear_calls()
                self.assertEqual(invoke(source, output), 0)
                # Seven writes, plus exact preflight for two extreme values.
                self.assertEqual(lib.sine_calls(), 9)
                np.testing.assert_array_equal(output[0, :7].view(np.uint32),
                    np.array(expected, np.float32).view(np.uint32))
                np.testing.assert_array_equal(output[0, 7:], -91.)
                invalid = source.copy(); invalid[0, 6] = np.nan
                output.fill(-91.)
                self.assertNotEqual(invoke(invalid, output), 0)
                np.testing.assert_array_equal(output, -91.)
                self.assertEqual(invoke(source, output), 0)
                np.testing.assert_array_equal(output[0, :7].view(np.uint32),
                    np.array(expected, np.float32).view(np.uint32))

    def test_committed_pytorch_oracle_padding_and_production_geometry(self):
        self.assertEqual(self.metadata['torch'], '2.8.0+cpu')
        for case in range(3):
            expected = self.arrays[f'case{case}_output']
            alpha = self.arrays[f'case{case}_alpha']
            channels, frames = expected.shape
            source = np.full((channels, frames + 5), np.nan, np.float32)
            source[:, :frames] = self.arrays[f'case{case}_input']
            outputs = []
            for variant, function in enumerate(self.functions):
                result = np.full((channels, frames + 7), -91., np.float32)
                self.assertEqual(self.call(function, source, alpha, result,
                                           frames), 0)
                self.assertTrue(np.isfinite(result[:, :frames]).all())
                self.assertTrue(np.all(result[:, frames:] == -91.))
                error = np.abs(result[:, :frames] - expected)
                worst = np.unravel_index(np.argmax(error), error.shape)
                maximum = float(error[worst])
                self.assertLessEqual(maximum, 2e-6,
                    (case, variant, worst, float(result[worst]), float(expected[worst])))
                print('CKE_NUMERICAL_CASE ' + json.dumps({
                    'case_id': f'audio.snake.pytorch28.case{case}.{("o0", "o3")[variant]}',
                    'name': 'checked channelwise Snake versus committed PyTorch 2.8',
                    'provider': 'audio_snake_strided_f32_checked',
                    'oracle': 'committed-pytorch-' + self.metadata['torch'],
                    'status': 'pass', 'max_diff': maximum, 'tolerance': 2e-6,
                    'rmse': float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
                    'worst_index': list(map(int, worst)),
                    'actual': float(result[worst]),
                    'reference': float(expected[worst]),
                    'configuration': f'channels{channels} frames{frames} padded-strides',
                    'reproduction_command': 'python3 -m unittest ' + self.id()}))
                outputs.append(result)
            np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_rejection_preserves_output_and_recovery(self):
        source = np.full((3, 9), np.nan, np.float32)
        source[:, :7] = self.arrays['case0_input']
        alpha = self.arrays['case0_alpha'].copy()
        for function in self.functions:
            result = np.full((3, 10), -91., np.float32)
            for kind in ('input_short', 'alpha_short', 'output_short',
                         'zero_alpha', 'nan_alpha', 'nan_input',
                         'product_overflow', 'inverse_overflow'):
                data = source.copy()
                coefficients = alpha.copy()
                input_capacity = alpha_capacity = output_capacity = None
                if kind == 'input_short': input_capacity = 24
                if kind == 'alpha_short': alpha_capacity = 2
                if kind == 'output_short': output_capacity = 26
                if kind == 'zero_alpha': coefficients[-1] = 0
                if kind == 'nan_alpha': coefficients[-1] = np.nan
                if kind == 'nan_input': data[-1, 6] = np.nan
                if kind == 'product_overflow':
                    data[-1, 6] = np.float32(3e38)
                    coefficients[-1] = 2
                if kind == 'inverse_overflow': coefficients[-1] = np.nextafter(
                    np.float32(0), np.float32(1))
                self.assertNotEqual(self.call(function, data, coefficients,
                    result, 7, input_capacity, alpha_capacity, output_capacity),
                    0, kind)
                self.assertTrue(np.all(result == -91.), kind)
            self.assertNotEqual(self.call(function, source, alpha, source,
                7), 0)
            self.assertTrue(np.isnan(source[:, 7:]).all())
            self.assertNotEqual(function(FLOAT(), source.size, 9,
                self.pointer(alpha), alpha.size, self.pointer(result),
                result.size, 10, 3, 7), 0)
            self.assertNotEqual(function(self.pointer(source), source.size, 9,
                FLOAT(), alpha.size, self.pointer(result),
                result.size, 10, 3, 7), 0)
            self.assertNotEqual(function(self.pointer(source), source.size, 9,
                self.pointer(alpha), alpha.size, self.pointer(result),
                result.size, 10, SIZE(-2).value, 7), 0)
            self.assertTrue(np.all(result == -91.))
            backing = np.arange(40, dtype=np.float32)
            overlapping_input = backing[:27].reshape(3, 9)
            overlapping_output = backing[1:31].reshape(3, 10)
            before = overlapping_output.copy()
            self.assertNotEqual(self.call(function, overlapping_input, alpha,
                                          overlapping_output, 7), 0)
            np.testing.assert_array_equal(overlapping_output, before)
            self.assertEqual(self.call(function, source, alpha, result, 7,
                                       input_capacity=25, alpha_capacity=3,
                                       output_capacity=27), 0)
            np.testing.assert_allclose(result[:, :7], self.arrays['case0_output'],
                                       rtol=0, atol=2e-6)

    def test_full_generator_geometry_repeated_lengths_and_independent_calls(self):
        rng = np.random.default_rng(3319)
        channels, stride = 256, 2064
        alpha = rng.uniform(0.3, 2.5, channels).astype(np.float32)
        source = np.full((channels, stride), np.nan, np.float32)
        source[:, :2060] = rng.normal(0, .8, (channels, 2060)).astype(np.float32)
        function = self.functions[1]

        def execute(frames):
            result = np.full((channels, stride), -91., np.float32)
            status = self.call(function, source, alpha, result, frames)
            return status, result

        long_status, first = execute(2060)
        short_status, middle = execute(1960)
        last_status, last = execute(2060)
        self.assertEqual((long_status, short_status, last_status), (0, 0, 0))
        np.testing.assert_array_equal(first, last)
        self.assertTrue(np.all(middle[:, 1960:] == -91.))
        self.assertTrue(np.all(first[:, 2060:] == -91.))
        x = source[:, :2060]
        a = alpha[:, None]
        expected = x + (1 / a) * (np.sin(a * x) ** 2)
        error = np.abs(first[:, :2060] - expected)
        self.assertTrue(np.isfinite(first[:, :2060]).all())
        self.assertLessEqual(float(error.max()), 2e-6)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(execute, frames) for frames in (1960, 2060)]
            concurrent_results = [future.result() for future in futures]
        self.assertEqual([status for status, _ in concurrent_results], [0, 0])
        np.testing.assert_array_equal(concurrent_results[0][1], middle)
        np.testing.assert_array_equal(concurrent_results[1][1], first)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'audio.snake.generator-shape.numpy',
            'name': 'full 256-channel generator geometry and repeated calls',
            'provider': 'audio_snake_strided_f32_checked',
            'oracle': 'independent-numpy-fp32', 'status': 'pass',
            'max_diff': float(error.max()), 'tolerance': 2e-6,
            'configuration': 'channels256 frames2060/1960 stride2064; A-B-A and two concurrent independent calls',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_live_pytorch_oracle_when_available(self):
        try:
            import torch
        except ImportError:
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': 'audio.snake.live-pytorch', 'status': 'not_tested',
                'name': 'live PyTorch channelwise Snake comparison',
                'reason': 'PyTorch unavailable', 'provider':
                'audio_snake_strided_f32_checked', 'oracle': 'live-pytorch'}))
            self.skipTest('live PyTorch unavailable; committed PyTorch oracle runs')
        data = self.arrays['case1_input']
        alpha = self.arrays['case1_alpha']
        x = torch.from_numpy(data.copy())
        a = torch.from_numpy(alpha.copy())[:, None]
        live = (x + (1 / a) * (torch.sin(a * x) ** 2)).numpy()
        stored = self.arrays['case1_output']
        fixture_error = float(np.max(np.abs(live - stored)))
        source = np.ascontiguousarray(data)
        result = np.empty_like(source)
        self.assertEqual(self.call(self.functions[1], source, alpha, result,
                                   source.shape[1]), 0)
        native_error = float(np.max(np.abs(result - live)))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'audio.snake.live-pytorch',
            'name': 'live PyTorch channelwise Snake comparison',
            'provider': 'audio_snake_strided_f32_checked',
            'oracle': 'live-pytorch-' + torch.__version__,
            'status': 'pass' if native_error <= 2e-6 else 'fail',
            'max_diff': native_error, 'tolerance': 2e-6,
            'fixture_reproduction_max_diff': fixture_error,
            'configuration': 'channels17 frames31',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        self.assertLessEqual(native_error, 2e-6)


if __name__ == '__main__':
    unittest.main()
