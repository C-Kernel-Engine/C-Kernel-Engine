"""Connected generated source STFT/convolution with explicit raw-phase diagnostic."""

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_stage_circuit_support import materialize_stage_circuit
from tests.v8_kokoro_duration_fixture_support import (
    populated_duration_arena, prepare_duration_fixture)
from tests.test_v8_kokoro_generated_prosody_complete import (
    complete_prosody_weights, verified_fixture)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_generated_source_stft_conv_circuit as author


class KokoroGeneratedSourceStftConvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        source_path = ROOT / 'tests/fixtures/tts/kokoro_source_pinned.npz'
        cls.source_meta = json.loads(source_path.with_suffix('.json').read_text())
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != cls.source_meta['fixture_sha256']:
            raise RuntimeError('pinned source fixture checksum mismatch')
        with np.load(source_path) as archive:
            cls.oracle = {key: archive[key].copy() for key in archive.files}
        weights, cls.prosody_oracle, cls.prosody_meta = complete_prosody_weights()
        if (cls.source_meta['model_pin'] != cls.prosody_meta['model_pin'] or
            cls.source_meta['code_pin'] != cls.prosody_meta['code_pin'] or
            cls.source_meta['asset_sha256'] != cls.prosody_meta['asset_sha256']):
            raise RuntimeError('source and prosody fixture identities differ')
        for key, name in (
            ('linear_weight', 'waveform_decoder.generator.m_source.l_linear.weight'),
            ('linear_bias', 'waveform_decoder.generator.m_source.l_linear.bias'),
            ('source_conv_weight', 'waveform_decoder.generator.noise_convs.0.weight'),
            ('source_conv_bias', 'waveform_decoder.generator.noise_convs.0.bias')):
            weights[name] = cls.oracle[key]
        circuit = materialize_stage_circuit(Path(cls.temp.name), author)
        fixture = prepare_duration_fixture(Path(cls.temp.name), circuit, weights)
        for name, value in vars(fixture).items():
            setattr(cls, name, value)
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = \
            compile_native_graph(Path(cls.temp.name), cls.source, circuit)
        conv_library = Path(cls.temp.name) / 'source_conv_checked.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-shared', '-fPIC',
            '-I', str(ROOT / 'include'),
            str(ROOT / 'src/kernels/audio_conv1d_checked.c'),
            '-lm', '-o', str(conv_library)], check=True, capture_output=True)
        cls.conv_loaded = ctypes.CDLL(str(conv_library))
        stft_library = Path(cls.temp.name) / 'source_stft_checked.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-shared', '-fPIC',
            '-I', str(ROOT / 'include'),
            str(ROOT / 'src/kernels/audio_stft_mag_phase_checked.c'),
            '-lm', '-o', str(stft_library)], check=True, capture_output=True)
        cls.stft_loaded = ctypes.CDLL(str(stft_library))
        cls.stft_trace = cls.stft_loaded.audio_stft_mag_phase_trace_checked_f32
        cls.stft_trace.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        cls.stft_trace.restype = ctypes.c_int
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + \
            [ctypes.POINTER(ctypes.c_int32)] * 5
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def view(self, arena, name, dtype=np.float32):
        item = self.buffers[name]
        return np.ndarray((item['size'] // np.dtype(dtype).itemsize,), dtype,
            buffer=arena, offset=item['abs_offset'])

    def arena(self):
        arena = populated_duration_arena(self.layout, self.entries, self.bump,
                                         self.encoder, self.duration)
        self.view(arena, 'source_gaussian').reshape(76800, 9)[:61800] = \
            self.oracle['gaussian'].reshape(61800, 9)
        self.view(arena, 'source_stft_window')[:] = self.oracle['window']
        taps = np.arange(20, dtype=np.float64)
        angles = -2 * np.pi * np.arange(11)[:, None] * taps / 20
        self.view(arena, 'source_stft_cos').reshape(11, 20)[:] = np.cos(angles)
        self.view(arena, 'source_stft_sin').reshape(11, 20)[:] = np.sin(angles)
        self.view(arena, 'source_conv0')[:] = -91.
        return arena

    def execute(self, arena):
        outputs = [ctypes.c_int32(-999) for _ in range(5)]
        status = self.fn(arena, len(arena), *(ctypes.byref(x) for x in outputs))
        return status, tuple(x.value for x in outputs)

    def native_source_convolution(self, channels):
        """Call the selected production kernel with an explicitly owned workspace."""
        channels = np.ascontiguousarray(channels, dtype=np.float32)
        weight = np.ascontiguousarray(self.oracle['source_conv_weight'])
        bias = np.ascontiguousarray(self.oracle['source_conv_bias'])
        frames = channels.shape[1]
        self.assertEqual(channels.shape[0], 22)
        self.assertEqual((frames + 6 - 12) // 6 + 1, 2060)
        output = np.full((256, 2060), -91., dtype=np.float32)
        scratch = np.empty_like(output)
        call = self.conv_loaded.audio_conv1d_checked_channel_major_f32
        call.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t] + [ctypes.c_size_t] * 7
        call.restype = ctypes.c_int
        pointer = lambda value: value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        status = call(pointer(channels), channels.size, frames,
            pointer(weight), weight.size, pointer(bias), bias.size,
            pointer(output), output.size, 2060, pointer(scratch), scratch.size,
            22, 256, frames, 12, 6, 3, 2060)
        self.assertEqual(status, 0)
        return output

    def native_stft(self, samples):
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        frames = len(samples) // 5 + 1
        taps = np.arange(20, dtype=np.float64)
        angles = -2 * np.pi * np.arange(11)[:, None] * taps / 20
        cosine = np.ascontiguousarray(np.cos(angles), dtype=np.float32)
        sine = np.ascontiguousarray(np.sin(angles), dtype=np.float32)
        window = np.ascontiguousarray(self.oracle['window'])
        output = np.empty((22, frames), dtype=np.float32)
        scratch = np.empty_like(output)
        call = self.stft_loaded.audio_stft_mag_phase_checked_f32
        call.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        call.restype = ctypes.c_int
        pointer = lambda value: value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        status = call(pointer(samples), samples.size, pointer(window), window.size,
            pointer(cosine), pointer(sine), cosine.size, pointer(output),
            output.size, frames, pointer(scratch), scratch.size,
            samples.size, 20, 5, frames)
        self.assertEqual(status, 0)
        return output

    def native_stft_trace(self, samples):
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        frames = len(samples) // 5 + 1
        taps = np.arange(20, dtype=np.float64)
        angles = -2 * np.pi * np.arange(11)[:, None] * taps / 20
        cosine = np.ascontiguousarray(np.cos(angles), dtype=np.float32)
        sine = np.ascontiguousarray(np.sin(angles), dtype=np.float32)
        window = np.ascontiguousarray(self.oracle['window'])
        output = np.empty((22, frames), dtype=np.float32)
        scratch = np.empty_like(output)
        trace = np.empty((11, frames, 2), dtype=np.float64)
        pointer = lambda value: value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        status = self.stft_trace(pointer(samples), samples.size,
            pointer(window), window.size, pointer(cosine), pointer(sine),
            cosine.size, pointer(output), output.size, frames,
            pointer(scratch), scratch.size, samples.size, 20, 5, frames,
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), trace.size)
        self.assertEqual(status, 0)
        return output, trace

    def test_reference_stft_to_native_source_convolution(self):
        """Isolate convolution from source generation and STFT arithmetic."""
        reference_channels = np.ascontiguousarray(np.concatenate((
            self.oracle['magnitude'][0], self.oracle['phase'][0]), axis=0))
        output = self.native_source_convolution(reference_channels)
        error = np.abs(output - self.oracle['first_source_conv'][0])
        worst = np.unravel_index(np.argmax(error), error.shape)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.source-conv.reference-stft-input',
            'name': 'native first source convolution with pinned reference STFT channels',
            'provider': 'audio_conv1d_checked_channel_major_f32',
            'oracle': 'direct-pinned-full-model-pytorch28',
            'status': 'pass' if float(error[worst]) <= 5e-5 else 'fail',
            'gate': 'numerical', 'blocking': True,
            'max_diff': float(error[worst]), 'tolerance': 5e-5,
            'worst_index': list(map(int, worst)),
            'configuration': 'pinned source STFT; 22 channels; kernel12/stride6/pad3',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        self.assertTrue(np.isfinite(output).all())
        self.assertLessEqual(float(error[worst]), 5e-5)

    def test_identical_waveform_stft_and_generated_source_isolation(self):
        """Keep reference-input diagnostics separate from generated execution."""
        try:
            import torch
        except ImportError:
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': 'kokoro.source-path.identical-input.live-pytorch28',
                'name': 'identical-input source STFT and convolution isolation',
                'status': 'not_tested', 'gate': 'diagnostic', 'blocking': False,
                'reason': 'live PyTorch unavailable',
                'reproduction_command': 'python3 -m unittest ' + self.id()}))
            self.skipTest('live PyTorch unavailable; committed reference-input convolution still runs')
        if not torch.__version__.startswith('2.8.'):
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': 'kokoro.source-path.identical-input.live-pytorch28',
                'name': 'identical-input source STFT and convolution isolation',
                'status': 'not_tested', 'gate': 'diagnostic', 'blocking': False,
                'reason': 'requires pinned PyTorch 2.8; found ' + torch.__version__,
                'reproduction_command': 'python3 -m unittest ' + self.id()}))
            self.skipTest('raw-phase compatibility diagnostic requires pinned PyTorch 2.8')
        torch.set_num_threads(1)
        window = torch.from_numpy(self.oracle['window'].copy())

        def torch_channels(samples):
            complex_spectrum = torch.stft(torch.from_numpy(samples.copy()),
                20, 5, 20, window=window, return_complex=True).numpy()
            channels = np.ascontiguousarray(np.concatenate((
                np.abs(complex_spectrum), np.angle(complex_spectrum)), axis=0))
            return complex_spectrum, channels

        arena = self.arena()
        self.assertEqual(self.execute(arena), (0, (103, 206, 61800, 12361, 2060)))
        generated_source = self.view(arena, 'source_waveform').reshape(-1)[:61800].copy()
        generated_channels = self.view(arena, 'source_stft_channels').reshape(
            22, 15361)[:, :12361].copy()
        generated_conv = self.view(arena, 'source_conv0').reshape(
            256, 2560)[:, :2060].copy()
        for label, samples, native_channels in (
            ('pinned-source', self.oracle['source'][0].copy(),
             self.native_stft(self.oracle['source'][0])),
            ('generated-source', generated_source, generated_channels)):
            traced_channels, complex_trace = self.native_stft_trace(samples)
            np.testing.assert_array_equal(traced_channels, native_channels)
            complex_reference, reference_channels = torch_channels(samples)
            if label == 'pinned-source':
                fixture_channels = np.concatenate((self.oracle['magnitude'][0],
                    self.oracle['phase'][0]), axis=0)
                fixture_mag_error = np.abs(reference_channels[:11] - fixture_channels[:11])
                fixture_phase_error = np.abs(reference_channels[11:] - fixture_channels[11:])
                fixture_pass = (float(fixture_mag_error.max()) <= 1e-6 and
                                float(fixture_phase_error.max()) <= 1e-5)
                print('CKE_NUMERICAL_CASE ' + json.dumps({
                    'case_id': 'kokoro.source-path.fixture-vs-live-pytorch28',
                    'name': 'pinned source STFT fixture reproducibility',
                    'oracle': 'live-pinned-pytorch-' + torch.__version__,
                    'status': 'pass' if fixture_pass else 'fail',
                    'gate': 'diagnostic', 'blocking': False,
                    'max_magnitude_error': float(fixture_mag_error.max()),
                    'max_raw_phase_error': float(fixture_phase_error.max()),
                    'magnitude_tolerance': 1e-6,
                    'raw_phase_tolerance': 1e-5,
                    'torch_git_version': torch.version.git_version,
                    'torch_cpu_capability': torch.backends.cpu.get_cpu_capability(),
                    'torch_num_threads': torch.get_num_threads(),
                    'torch_config_sha256': hashlib.sha256(
                        torch.__config__.show().encode()).hexdigest(),
                    'reproduction_command': 'python3 -m unittest ' + self.id()}))
            magnitude_error = np.abs(native_channels[:11] - reference_channels[:11])
            phase_error = np.abs(native_channels[11:] - reference_channels[11:])
            # This is the production kernel's actual FP64 pre-atan2 trace,
            # not a reconstruction from rounded magnitude/phase channels.
            complex_native = (complex_trace[:, :, 0] +
                              1j * complex_trace[:, :, 1])
            complex_error = np.abs(complex_native - complex_reference)
            branch = np.argwhere(phase_error > 1.)
            branch_details = []
            for bin_index, frame_index in branch[:8]:
                source_frame = np.pad(samples, 10, mode='reflect')[
                    frame_index * 5:frame_index * 5 + 20]
                windowed = np.ascontiguousarray(source_frame *
                    self.oracle['window'])
                minimal_fft = torch.fft.rfft(torch.from_numpy(windowed),
                    n=20).numpy()[bin_index]
                # A 20-sample excerpt reproduces the exact CKE frame: frame
                # zero stays at zero; an interior frame becomes local frame 2.
                excerpt_start = max(0, int(frame_index) * 5 - 10)
                excerpt = samples[excerpt_start:excerpt_start + 20].copy()
                local_frame = 0 if frame_index == 0 else 2
                _, excerpt_trace = self.native_stft_trace(excerpt)
                np.testing.assert_array_equal(
                    excerpt_trace[bin_index, local_frame],
                    complex_trace[bin_index, frame_index])
                branch_details.append({
                    'bin': int(bin_index), 'frame': int(frame_index),
                    'native_magnitude': float(native_channels[bin_index, frame_index]),
                    'native_raw_phase': float(native_channels[11 + bin_index, frame_index]),
                    'native_pre_atan2_real_f64': float(complex_trace[
                        bin_index, frame_index, 0]),
                    'native_pre_atan2_imag_f64': float(complex_trace[
                        bin_index, frame_index, 1]),
                    'native_pre_atan2_imag_signbit': bool(np.signbit(
                        complex_trace[bin_index, frame_index, 1])),
                    'oracle_real': float(complex_reference[bin_index, frame_index].real),
                    'oracle_imag': float(complex_reference[bin_index, frame_index].imag),
                    'oracle_imag_signbit': bool(np.signbit(
                        complex_reference[bin_index, frame_index].imag)),
                    'oracle_raw_phase': float(reference_channels[11 + bin_index, frame_index]),
                    'minimal_fft_real': float(minimal_fft.real),
                    'minimal_fft_imag': float(minimal_fft.imag),
                    'minimal_excerpt_start': excerpt_start,
                    'minimal_excerpt_local_frame': local_frame,
                    'minimal_excerpt_sha256': hashlib.sha256(
                        excerpt.tobytes()).hexdigest(),
                    'windowed_frame_sha256': hashlib.sha256(windowed.tobytes()).hexdigest()})
            native_conv = (generated_conv if label == 'generated-source' else
                           self.native_source_convolution(native_channels))
            reference_input_conv = self.native_source_convolution(reference_channels)
            conv_input_error = np.abs(native_conv - reference_input_conv)
            worst = np.unravel_index(np.argmax(conv_input_error),
                                     conv_input_error.shape)
            model_error = np.abs(reference_input_conv -
                                 self.oracle['first_source_conv'][0])
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': 'kokoro.source-path.identical-input.' + label,
                'name': 'same waveform through native and pinned PyTorch STFT; native convolution isolation',
                'provider': 'audio_stft_mag_phase_checked_f32',
                'oracle': 'live-pinned-pytorch-' + torch.__version__,
                'status': 'fail' if len(branch) or float(conv_input_error[worst]) > 5e-5 else 'pass',
                'gate': 'diagnostic', 'blocking': False,
                'reason': 'raw phase remains a convolution input; complex values are actual native pre-atan2 accumulators',
                'max_complex_real_error': float(np.max(np.abs(
                    complex_native.real - complex_reference.real))),
                'max_complex_imag_error': float(np.max(np.abs(
                    complex_native.imag - complex_reference.imag))),
                'max_complex_error': float(complex_error.max()),
                'max_magnitude_error': float(magnitude_error.max()),
                'max_raw_phase_error': float(phase_error.max()),
                'raw_phase_branch_cut_mismatch_count': len(branch),
                'first_raw_phase_branch_cut_mismatches': branch[:8].tolist(),
                'first_raw_phase_branch_cut_details': branch_details,
                'max_generated_to_pinned_source_error': float(np.max(np.abs(
                    samples - self.oracle['source'][0]))),
                'max_same_source_conv_propagation_error':
                    float(conv_input_error[worst]),
                'same_source_conv_tolerance': 5e-5,
                'worst_source_conv_index': list(map(int, worst)),
                'max_reference_input_conv_to_model_error': float(model_error.max()),
                'configuration': label + '; 61,800 samples; nfft20/hop5',
                'reproduction_command': 'python3 -m unittest ' + self.id()}))
            self.assertTrue(np.isfinite(native_channels).all())
            self.assertTrue(np.isfinite(native_conv).all())

    def test_generated_source_stft_conv_and_known_raw_phase_failure(self):
        self.assertEqual(self.calls['errors'], [])
        self.assertEqual([op['function'] for op in self.calls['operations'][-4:]],
            ['ck_runtime_affine_i32_checked', 'ck_runtime_scale_i32_checked',
             'audio_stft_mag_phase_checked_f32',
             'audio_conv1d_checked_channel_major_f32'])
        arena = self.arena()
        status, lengths = self.execute(arena)
        self.assertEqual(status, 0)
        self.assertEqual(lengths, (103, 206, 61800, 12361, 2060))
        spectrum = self.view(arena, 'source_stft_channels').reshape(22, 15361)[:, :12361]
        expected = np.concatenate((self.oracle['magnitude'][0],
                                   self.oracle['phase'][0]), axis=0)
        self.assertTrue(np.isfinite(spectrum).all())
        mag_error = np.abs(spectrum[:11] - expected[:11])
        phase_error = np.abs(spectrum[11:] - expected[11:])
        conv = self.view(arena, 'source_conv0').reshape(256, 2560)[:, :2060]
        self.assertTrue(np.isfinite(conv).all())
        conv_error = np.abs(conv - self.oracle['first_source_conv'][0])
        worst = np.unravel_index(np.argmax(conv_error), conv_error.shape)
        mismatches = np.argwhere(phase_error > 1)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.generated-source-stft-conv.raw-phase-v1',
            'name': 'connected generated source through scalar STFT and first source convolution',
            'provider': 'audio_stft_mag_phase_checked_f32',
            'oracle': 'direct-pinned-full-model-pytorch28',
            'status': 'fail', 'gate': 'diagnostic', 'blocking': False,
            'reason': 'known raw-phase branch-cut incompatibility; source STFT convolution parity unresolved',
            'max_diff': float(conv_error[worst]), 'tolerance': 5e-5,
            'worst_index': list(map(int, worst)),
            'max_magnitude_error': float(mag_error.max()),
            'max_raw_phase_error': float(phase_error.max()),
            'raw_phase_branch_cut_mismatch_count': len(mismatches),
            'first_raw_phase_branch_cut_mismatches': mismatches[:8].tolist(),
            'configuration': '36 phonemes; 61,800 generated source samples; captured Gaussian',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        self.assertTrue(np.all(self.view(arena, 'source_conv0').reshape(256, 2560)[:, 2060:] == -91.))

    def test_rejection_and_recovery(self):
        arena = self.arena()
        lengths = [ctypes.c_int32(-999) for _ in range(5)]
        self.assertNotEqual(self.fn(arena, len(arena) - 1,
            *(ctypes.byref(x) for x in lengths)), 0)
        self.assertEqual([x.value for x in lengths], [-999] * 5)
        self.assertTrue(np.all(self.view(arena, 'source_conv0') == -91.))
        self.assertEqual(self.execute(arena), (0, (103, 206, 61800, 12361, 2060)))
        original = self.view(arena, 'source_conv0').reshape(256, 2560)[:, :2060].copy()
        self.view(arena, 'source_stft_window')[0] = np.nan
        self.view(arena, 'source_conv0')[:] = -91.
        status, lengths = self.execute(arena)
        self.assertNotEqual(status, 0)
        self.assertEqual(lengths[-1], -999)
        self.assertTrue(np.all(self.view(arena, 'source_conv0') == -91.))
        self.view(arena, 'source_stft_window')[0] = self.oracle['window'][0]
        self.assertEqual(self.execute(arena), (0, (103, 206, 61800, 12361, 2060)))
        alternate, meta = verified_fixture('generator_stage0_your')
        self.assertEqual(meta['model_pin'], self.source_meta['model_pin'])
        self.assertEqual(meta['asset_sha256'], self.source_meta['asset_sha256'])
        self.view(arena, 'word_ids', np.int32)[:36] = alternate['word_ids']
        self.view(arena, 'predictor_style')[:128] = alternate['predictor_style']
        self.view(arena, 'source_conv0')[:] = -91.
        self.assertEqual(self.execute(arena), (0, (98, 196, 58800, 11761, 1960)))
        other = self.view(arena, 'source_conv0').reshape(256, 2560)
        self.assertTrue(np.isfinite(other[:, :1960]).all())
        self.assertTrue(np.all(other[:, 1960:] == -91.))
        self.view(arena, 'word_ids', np.int32)[:36] = self.encoder['word_ids']
        self.view(arena, 'predictor_style')[:128] = self.duration['predictor_style']
        self.view(arena, 'source_conv0')[:] = -91.
        self.assertEqual(self.execute(arena), (0, (103, 206, 61800, 12361, 2060)))
        np.testing.assert_array_equal(
            self.view(arena, 'source_conv0').reshape(256, 2560)[:, :2060], original)


if __name__ == '__main__':
    unittest.main()
