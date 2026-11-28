"""Independent counter/Box-Muller oracle and bounds for native excitation."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MASK = (1 << 64) - 1
FLOAT = ctypes.POINTER(ctypes.c_float)


def mix64(value):
    value = (value + 0x9e3779b97f4a7c15) & MASK
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & MASK
    return value ^ (value >> 31)


def oracle(count, seed, stream, *, dtype=np.float32):
    domain = mix64(seed ^ mix64(stream))
    result = np.empty(count, dtype=dtype)
    for index in range(count):
        first = mix64((domain + 2 * index) & MASK)
        second = mix64((domain + 2 * index + 1) & MASK)
        positive = ((first >> 11) + 0.5) * 2.0 ** -53
        angle = (second >> 11) * 2.0 ** -53
        result[index] = math.sqrt(-2.0 * math.log(positive)) * math.cos(
            2.0 * math.pi * angle)
    return result


class NativeNoiseCounterOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.functions = []
        for optimization in ('-O0', '-O3'):
            binary = Path(cls.temp.name) / f'noise_{optimization}.so'
            subprocess.run(['cc', '-std=c11', optimization, '-Wall', '-Wextra',
                '-Werror', '-pedantic', '-shared', '-fPIC', '-I', str(ROOT / 'include'),
                str(ROOT / 'src/kernels/audio_noise_counter.c'), '-lm', '-o',
                str(binary)], check=True, capture_output=True)
            function = ctypes.CDLL(str(binary)).audio_noise_counter_gaussian_f32
            function.argtypes = [FLOAT, ctypes.c_size_t, ctypes.c_size_t,
                                 ctypes.c_uint64, ctypes.c_uint64]
            function.restype = ctypes.c_int
            cls.functions.append(function)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def pointer(array):
        return array.ctypes.data_as(FLOAT)

    def test_oracle_seed_stream_prefix_and_repeated_requests(self):
        worst = 0.
        for count in (1, 2, 9, 37, 257):
            expected = oracle(count, 0x123456789ABCDEF0, 9)
            for function in self.functions:
                output = np.full(count + 5, -91., dtype=np.float32)
                args = (self.pointer(output), len(output), count,
                        0x123456789ABCDEF0, 9)
                self.assertEqual(function(*args), 0)
                np.testing.assert_allclose(output[:count], expected,
                                           rtol=0, atol=2e-6)
                worst = max(worst, float(np.max(np.abs(
                    output[:count] - expected))))
                self.assertTrue(np.all(output[count:] == -91.))
                first = output[:count].copy()
                output.fill(-91.)
                self.assertEqual(function(*args), 0)
                np.testing.assert_array_equal(output[:count], first)
                other = np.empty(count, dtype=np.float32)
                self.assertEqual(function(self.pointer(other), count, count,
                                          0x123456789ABCDEF0, 10), 0)
                self.assertFalse(np.array_equal(other, first))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'audio.noise.counter-gaussian.oracle-v1',
            'name': 'request-local Gaussian counter kernel versus independent Python oracle',
            'provider': 'audio_noise_counter_gaussian_f32',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'independent-python-splitmix64-box-muller',
            'status': 'pass', 'gate': 'numerical', 'blocking': True,
            'max_diff': worst, 'tolerance': 2e-6,
            'configuration': 'O0/O3; counts 1,2,9,37,257; fixed seed/stream; padding and replay',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_audio_noise_counter_oracle'}))

    def test_rejection_preserves_output_and_recovery(self):
        for function in self.functions:
            output = np.full(32, -91., dtype=np.float32)
            pointer = self.pointer(output)
            for capacity, count in ((31, 32), (32, 0), (32, 1 << 24 | 1)):
                self.assertNotEqual(function(pointer, capacity, count, 7, 1), 0)
                self.assertTrue(np.all(output == -91.))
            self.assertNotEqual(function(FLOAT(), 32, 32, 7, 1), 0)
            self.assertTrue(np.all(output == -91.))
            raw = (ctypes.c_uint8 * 132)()
            misaligned = ctypes.cast(ctypes.addressof(raw) + 1, FLOAT)
            self.assertNotEqual(function(misaligned, 32, 32, 7, 1), 0)
            self.assertEqual(function(pointer, 32, 32, 7, 1), 0)
            np.testing.assert_allclose(output, oracle(32, 7, 1),
                                       rtol=0, atol=2e-6)

    def test_distribution_and_independent_calls(self):
        for function in self.functions:
            first = np.empty(8192, dtype=np.float32)
            second = np.empty(8192, dtype=np.float32)
            self.assertEqual(function(self.pointer(first), first.size,
                                      first.size, 11, 0), 0)
            self.assertEqual(function(self.pointer(second), second.size,
                                      second.size, 12, 0), 0)
            self.assertLess(abs(statistics.mean(first)), .06)
            self.assertTrue(.9 < statistics.pvariance(first) < 1.1)
            self.assertFalse(np.array_equal(first, second))
            self.assertTrue(np.isfinite(first).all())

    def test_independent_concurrent_calls_match_serial(self):
        function = self.functions[-1]

        def generate(seed):
            output = np.empty(1025, dtype=np.float32)
            self.assertEqual(function(self.pointer(output), output.size,
                                      output.size, seed, 4), 0)
            return output

        seeds = (1, 2, 3, 4)
        expected = [generate(seed) for seed in seeds]
        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent = list(pool.map(generate, seeds))
        for actual, reference in zip(concurrent, expected):
            np.testing.assert_array_equal(actual, reference)


if __name__ == '__main__':
    unittest.main()
