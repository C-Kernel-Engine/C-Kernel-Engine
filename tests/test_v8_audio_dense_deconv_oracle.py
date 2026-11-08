"""Checked dense transposed Conv1D versus independent PyTorch fixtures."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/tts/audio_dense_deconv_torch_reference.npz'
POINTER = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t


class AudioDenseDeconvOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.meta = json.loads(FIXTURE.with_suffix('.json').read_text())
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != cls.meta['fixture_sha256']:
            raise RuntimeError('dense transposed-convolution fixture hash mismatch')
        with np.load(FIXTURE) as archive:
            cls.arrays = {name: archive[name].copy() for name in archive.files}
        for name, value in cls.arrays.items():
            if hashlib.sha256(value.tobytes()).hexdigest() != cls.meta['array_sha256'][name]:
                raise RuntimeError(f'dense transposed-convolution tensor mismatch: {name}')
        cls.functions = []
        for optimization in ('-O0', '-O3'):
            library = Path(cls.temp.name) / f'dense_deconv_{optimization}.so'
            subprocess.run(['cc', '-std=c11', optimization, '-Wall', '-Wextra',
                '-Werror', '-pedantic', '-shared', '-fPIC',
                '-I', str(ROOT / 'include'),
                str(ROOT / 'src/kernels/audio_conv_transpose_dense_checked.c'),
                '-lm', '-o', str(library)], check=True, capture_output=True)
            loaded = ctypes.CDLL(str(library))
            run = loaded.audio_conv_transpose1d_dense_channel_major_f32_checked
            run.argtypes = [POINTER, SIZE, SIZE, POINTER, SIZE,
                            POINTER, SIZE, POINTER, SIZE, SIZE,
                            POINTER] + [SIZE] * 9
            run.restype = ctypes.c_int
            workspace = loaded.audio_conv_transpose1d_dense_f32_workspace
            workspace.argtypes = [SIZE, SIZE, ctypes.POINTER(SIZE)]
            workspace.restype = ctypes.c_int
            cls.functions.append((run, workspace))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def ptr(value):
        return value.ctypes.data_as(POINTER)

    def call(self, function, index):
        inputs, outputs, frames, kernel, stride, padding, output_padding = \
            self.meta['cases'][index]
        expected = self.arrays[f'case{index}_output']
        out_frames = expected.shape[1]
        source = np.full((inputs, frames + 3), np.nan, np.float32)
        source[:, :frames] = self.arrays[f'case{index}_input']
        output = np.full((outputs, out_frames + 4), -91., np.float32)
        weight = self.arrays[f'case{index}_weight'].copy()
        bias = self.arrays[f'case{index}_bias'].copy()
        scratch = np.full(outputs * out_frames, -37., np.float32)
        args = [self.ptr(source), source.size, source.shape[1],
                self.ptr(weight), weight.size, self.ptr(bias), bias.size,
                self.ptr(output), output.size, output.shape[1],
                self.ptr(scratch), scratch.size,
                inputs, outputs, frames, kernel, stride, padding,
                output_padding, out_frames]
        return function(*args), output, expected, args, (source, weight, bias, scratch)

    def test_committed_pytorch_oracle_strides_tails_and_threads(self):
        worst = (0., None)
        for index, geometry in enumerate(self.meta['cases']):
            outputs = []
            for function, workspace in self.functions:
                required = SIZE(0)
                self.assertEqual(workspace(geometry[1],
                    self.arrays[f'case{index}_output'].shape[1],
                    ctypes.byref(required)), 0)
                status, actual, expected, _, _ = self.call(function, index)
                self.assertEqual(status, 0)
                self.assertEqual(required.value, expected.size)
                self.assertTrue(np.isfinite(actual[:, :expected.shape[1]]).all())
                delta = np.abs(actual[:, :expected.shape[1]] - expected)
                point = np.unravel_index(np.argmax(delta), delta.shape)
                if float(delta[point]) > worst[0]:
                    worst = (float(delta[point]), [index, *map(int, point)])
                self.assertLessEqual(float(delta[point]), 3e-6,
                                     (index, point, float(delta[point])))
                self.assertTrue(np.all(actual[:, expected.shape[1]:] == -91.))
                outputs.append(actual.copy())
            np.testing.assert_array_equal(outputs[0], outputs[1])
        function = self.functions[1][0]
        with ThreadPoolExecutor(max_workers=3) as pool:
            parallel = list(pool.map(lambda _: self.call(function, 2)[1], range(3)))
        for actual in parallel[1:]:
            np.testing.assert_array_equal(actual, parallel[0])
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'tts.dense-deconv.native-vs-pytorch-committed',
            'name': 'checked dense ConvTranspose1D versus committed PyTorch',
            'provider': 'audio_conv_transpose1d_dense_channel_major_f32_checked',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'committed-pytorch',
            'backend_version': self.meta['oracle_version'],
            'status': 'pass', 'max_diff': worst[0],
            'worst_index': worst[1], 'tolerance': 3e-6,
            'configuration': str(self.meta['cases']),
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_00_rejection_preserves_output_and_recovers_oracle(self):
        for function, workspace in self.functions:
            status, output, expected, args, buffers = self.call(function, 1)
            self.assertEqual(status, 0)
            for index, value in ((1, 0), (4, 0), (6, 0), (8, 0), (11, 0),
                                 (12, 0), (13, 0), (14, 0), (15, 0),
                                 (16, 0), (18, 2), (19, 10),
                                 (2, 1), (9, 1),
                                 (14, SIZE(-1).value)):
                with self.subTest(index=index, value=value):
                    output.fill(-91.)
                    bad = args.copy(); bad[index] = value
                    self.assertNotEqual(function(*bad), 0)
                    self.assertTrue(np.all(output == -91.))
            source, weight, bias, _ = buffers
            for candidate in (source, weight, bias):
                original = float(candidate.flat[0])
                candidate.flat[0] = math.inf
                output.fill(-91.)
                self.assertEqual(function(*args), -3)
                self.assertTrue(np.all(output == -91.))
                candidate.flat[0] = original
            output.fill(-91.)
            alias = args.copy(); alias[10] = self.ptr(output)
            self.assertNotEqual(function(*alias), 0)
            self.assertTrue(np.all(output == -91.))
            # Reject overlapping subregions, not only identical base pointers.
            partial = args.copy()
            partial[10] = self.ptr(output.ravel()[1:])
            self.assertNotEqual(function(*partial), 0)
            self.assertTrue(np.all(output == -91.))
            for pointer_index in (0, 3, 5, 7, 10):
                with self.subTest(null_pointer=pointer_index):
                    invalid = args.copy()
                    invalid[pointer_index] = POINTER()
                    self.assertNotEqual(function(*invalid), 0)
                    self.assertTrue(np.all(output == -91.))
            for capacity_index, needed in ((1, 18), (4, 75),
                                           (6, 5), (8, 61), (11, 45)):
                # Use the actual checked span or exact weight/scratch geometry.
                exact = args.copy()
                exact[capacity_index] = needed
                self.assertEqual(function(*exact), 0)
                short = exact.copy()
                short[capacity_index] = needed - 1
                output.fill(-91.)
                self.assertNotEqual(function(*short), 0)
                self.assertTrue(np.all(output == -91.))
            original_source = source.copy()
            original_weight = weight.copy()
            source[:, :self.meta['cases'][1][2]] = np.float32(1e30)
            weight[:] = np.float32(1e30)
            output.fill(-91.)
            self.assertEqual(function(*args), -3)
            self.assertTrue(np.all(output == -91.))
            np.copyto(source, original_source)
            np.copyto(weight, original_weight)
            output.fill(-91.)
            self.assertEqual(function(*args), 0)
            actual = output[:, :expected.shape[1]]
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-6)
            self.assertTrue(np.all(output[:, expected.shape[1]:] == -91.))
            for name, value in self.arrays.items():
                self.assertEqual(hashlib.sha256(value.tobytes()).hexdigest(),
                    self.meta['array_sha256'][name], name)
            required = SIZE(0)
            self.assertEqual(workspace(0, 3, ctypes.byref(required)), -4)
            self.assertEqual(workspace(SIZE(-1).value, 3,
                                       ctypes.byref(required)), -4)

    def test_ordered_fma_bits_across_vector_and_crop_boundaries(self):
        # Output-coordinate oracle is independent of the kernel's scatter loop.
        # libm fmaf rounds once to FP32, including signed-zero behavior.
        import ctypes.util
        fma = ctypes.CDLL(ctypes.util.find_library('m')).fmaf
        fma.argtypes = [ctypes.c_float] * 3
        fma.restype = ctypes.c_float
        rng = np.random.default_rng(7129)
        for kernel, stride, padding, extra in ((7, 3, 2, 1), (8, 4, 3, 0),
                (9, 3, 4, 2), (15, 5, 7, 1), (16, 8, 4, 0),
                (17, 6, 8, 3), (20, 10, 5, 0), (12, 6, 3, 0)):
            inputs, channels, frames = 3, 5, 7
            width = (frames - 1) * stride + kernel - 2 * padding + extra
            x = rng.uniform(-1, 1, (inputs, frames + 3)).astype(np.float32)
            w = rng.uniform(-1, 1, (inputs, channels, kernel)).astype(np.float32)
            w[0] *= np.float32(1e-4)
            w[1] *= np.float32(1e4)
            b = rng.uniform(-1, 1, channels).astype(np.float32)
            b[0] = np.float32(-0.0)
            expected = np.full((channels, width + 4), -91, np.float32)
            for oc in range(channels):
                for pos in range(width):
                    value = float(b[oc])
                    for ic in range(inputs):
                        for frame in range(frames):
                            tap = pos + padding - frame * stride
                            if 0 <= tap < kernel:
                                value = fma(float(x[ic, frame]), float(w[ic, oc, tap]), value)
                    expected[oc, pos] = value
            for run, _ in self.functions:
                with self.subTest(kernel=kernel, stride=stride, padding=padding):
                    actual = np.full_like(expected, -91)
                    scratch = np.empty(channels * width, np.float32)
                    self.assertEqual(run(self.ptr(x), x.size, x.shape[1],
                        self.ptr(w), w.size, self.ptr(b), b.size,
                        self.ptr(actual), actual.size, actual.shape[1],
                        self.ptr(scratch), scratch.size, inputs, channels, frames,
                        kernel, stride, padding, extra, width), 0)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_fused_rounding_is_not_replaced_by_multiply_add(self):
        x = np.array([1.0 + 2.0**-23], np.float32)
        w = np.full(12, 1.0 - 2.0**-23, np.float32)
        b = np.array([-1.0], np.float32)
        expected = np.full(12, -2.0**-46, np.float32)
        unfused = np.float32(np.float32(x[0] * w[0]) + b[0])
        self.assertNotEqual(unfused.view(np.uint32), expected[0].view(np.uint32))
        for run, _ in self.functions:
            output = np.full(12, -91, np.float32)
            scratch = np.empty_like(output)
            self.assertEqual(run(self.ptr(x), 1, 1, self.ptr(w), 12,
                self.ptr(b), 1, self.ptr(output), 12, 12,
                self.ptr(scratch), 12, 1, 1, 1, 12, 1, 0, 0, 12), 0)
            np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))

    def test_live_pytorch_oracle_if_available(self):
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            self.skipTest('live PyTorch unavailable; committed fixture still runs')
        rng = np.random.default_rng(333)
        x = (rng.standard_normal((16, 17)) * .1).astype(np.float32)
        w = (rng.standard_normal((16, 8, 20)) * .1).astype(np.float32)
        b = (rng.standard_normal(8) * .1).astype(np.float32)
        expected = functional.conv_transpose1d(
            torch.from_numpy(x)[None], torch.from_numpy(w),
            torch.from_numpy(b), stride=10, padding=5)[0].detach().numpy()
        actual = np.full((8, expected.shape[1]), -91., np.float32)
        scratch = np.empty(expected.size, np.float32)
        function = self.functions[1][0]
        self.assertEqual(function(self.ptr(x), x.size, x.shape[1],
            self.ptr(w), w.size, self.ptr(b), b.size,
            self.ptr(actual), actual.size, actual.shape[1],
            self.ptr(scratch), scratch.size, 16, 8, 17, 20, 10, 5, 0,
            expected.shape[1]), 0)
        self.assertTrue(np.isfinite(actual).all())
        error = np.abs(actual - expected)
        point = np.unravel_index(np.argmax(error), error.shape)
        self.assertLessEqual(float(error[point]), 1e-5,
                             (point, float(error[point])))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'tts.dense-deconv.native-vs-pytorch-live',
            'name': 'checked dense ConvTranspose1D versus live PyTorch',
            'provider': 'audio_conv_transpose1d_dense_channel_major_f32_checked',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'live-pytorch', 'backend_version': torch.__version__,
            'status': 'pass', 'max_diff': float(error[point]),
            'worst_index': list(map(int, point)), 'tolerance': 1e-5,
            'configuration': 'I=16 O=8 T=17 K=20 stride=10 padding=5',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_full_generator_shape_if_requested(self):
        if os.environ.get('CKE_TTS_GENERATOR_FULL_SHAPE') != '1':
            self.skipTest('full 512-to-256 generator shape reserved for asset-backed host')
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            self.skipTest('live PyTorch unavailable for full generator geometry')
        torch.set_num_threads(1)
        rng = np.random.default_rng(719)
        x = (rng.standard_normal((512, 206)) * .01).astype(np.float32)
        w = (rng.standard_normal((512, 256, 20)) * .01).astype(np.float32)
        b = (rng.standard_normal(256) * .01).astype(np.float32)
        expected = functional.conv_transpose1d(
            torch.from_numpy(x)[None], torch.from_numpy(w),
            torch.from_numpy(b), stride=10, padding=5)[0].detach().numpy()
        actual = np.full_like(expected, -91.)
        scratch = np.empty(expected.size, np.float32)
        function = self.functions[1][0]
        self.assertEqual(function(self.ptr(x), x.size, x.shape[1],
            self.ptr(w), w.size, self.ptr(b), b.size,
            self.ptr(actual), actual.size, actual.shape[1],
            self.ptr(scratch), scratch.size, 512, 256, 206, 20, 10, 5, 0,
            expected.shape[1]), 0)
        self.assertTrue(np.isfinite(actual).all())
        error = np.abs(actual - expected)
        point = np.unravel_index(np.argmax(error), error.shape)
        self.assertLessEqual(float(error[point]), 1e-5,
                             (point, float(error[point])))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'tts.dense-deconv.generator-shape-live',
            'name': 'dense ConvTranspose1D at first Kokoro generator geometry',
            'provider': 'audio_conv_transpose1d_dense_channel_major_f32_checked',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'live-pytorch', 'backend_version': torch.__version__,
            'status': 'pass', 'max_diff': float(error[point]),
            'worst_index': list(map(int, point)), 'tolerance': 1e-5,
            'configuration': 'I=512 O=256 T=206 K=20 stride=10 padding=5',
            'reproduction_command': 'CKE_TTS_GENERATOR_FULL_SHAPE=1 '
                'python3 -m unittest ' + self.id()}))


if __name__ == '__main__':
    unittest.main()
