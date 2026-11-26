"""Runtime Kokoro source STFT/convolution: execution and numerical diagnosis."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_duration_fixture_support import prepare_duration_fixture
from tests.v8_kokoro_decoder_weight_support import decoder_weights_and_references
from tests.test_v8_kokoro_generated_decoder_complete import (
    require_matching_reference_identity)
from tests.test_v8_kokoro_generated_prosody_complete import verified_fixture
from tests import test_v8_kokoro_generated_albert_layer as xray_support
from version.v8.scripts.replay_audio_stft_bundle_v8 import replay
from version.v8.scripts.xray_operation_bundle_v8 import write_bundle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from build_kokoro_runtime_source_stft_conv_circuit import build_circuit


class KokoroRuntimeSourceStftConvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        graph = build_circuit()
        circuit = cls.root / 'circuit.json'
        circuit.write_text(json.dumps(graph))
        weights, _, direct_meta, encode, encode_meta, _, decode_meta = (
            decoder_weights_and_references())
        cls.hello8, cls.hello_meta = verified_fixture('runtime_decoder_hello_8')
        cls.generator36, generator_meta = verified_fixture('generator_stage0')
        cls.source36, source_meta = verified_fixture('source')
        cls.source_meta = source_meta
        for meta in (encode_meta, *decode_meta.values(), cls.hello_meta,
                     generator_meta, source_meta):
            require_matching_reference_identity(direct_meta, meta)
        for name, value in cls.generator36.items():
            if name.startswith('waveform_decoder.'):
                weights[name] = value
        for key, name in (
            ('linear_weight', 'waveform_decoder.generator.m_source.l_linear.weight'),
            ('linear_bias', 'waveform_decoder.generator.m_source.l_linear.bias'),
            ('source_conv_weight', 'waveform_decoder.generator.noise_convs.0.weight'),
            ('source_conv_bias', 'waveform_decoder.generator.noise_convs.0.bias')):
            weights[name] = cls.source36[key]
        cls.fixture = prepare_duration_fixture(cls.root, circuit, weights)
        cls.fixture.source['config']['activation_buffer_dtypes'].update({
            'request_length': 'i32', 'valid_length': 'i32',
            'valid_residual_elements': 'i32'})
        cls.fixture.source['config']['context_length'] = 64
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(cls.root, cls.fixture.source, circuit,
                extra_exports=('audio_stft_mag_phase_checked_f32',)))
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + [
            ctypes.POINTER(ctypes.c_int32)] * 8
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}
        cls.weights = {item['name']: item for item in
                       cls.layout['memory']['weights']['entries']}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def view(self, arena, name, shape, dtype=np.float32):
        return np.ndarray(shape, dtype, buffer=arena,
                          offset=self.buffers[name]['abs_offset'])

    def arena(self):
        size = self.layout['memory']['arena']['total_size']
        raw = (ctypes.c_uint8 * (size + 63))()
        arena = (ctypes.c_uint8 * size).from_buffer(
            raw, (-ctypes.addressof(raw)) & 63)
        for planned in self.weights.values():
            entry = self.fixture.entries[planned['name']]
            start = entry['file_offset']
            arena[planned['abs_offset']:planned['abs_offset'] + entry['size']] = (
                self.fixture.bump[start:start + entry['size']])
        self.view(arena, 'type_ids', (64,), np.int32)[:] = 0
        np.testing.assert_array_equal(self.hello8['generator_stft_window'],
                                      self.source36['window'])
        self.view(arena, 'source_stft_window', (20,))[:] = self.source36['window']
        taps = np.arange(20, dtype=np.float64)
        angles = -2 * np.pi * np.arange(11)[:, None] * taps / 20
        self.view(arena, 'source_stft_cos', (11, 20))[:] = np.cos(angles)
        self.view(arena, 'source_stft_sin', (11, 20))[:] = np.sin(angles)
        return arena

    def execute(self, arena, count, ids, predictor_style, decoder_style,
                gaussian):
        words = self.view(arena, 'word_ids', (64,), np.int32)
        words[:] = 178
        words[:count] = ids[:count]
        self.view(arena, 'request_length', (1,), np.int32)[0] = count
        self.view(arena, 'predictor_style', (128,))[:] = predictor_style
        self.view(arena, 'decoder_style', (128,))[:] = decoder_style
        noise = self.view(arena, 'source_gaussian', (76800, 9))
        noise[:] = 0
        noise[:gaussian.size // 9] = gaussian.reshape(-1, 9)
        self.view(arena, 'source_stft_channels', (22, 15361))[:] = -91.
        self.view(arena, 'source_conv0', (256, 2560))[:] = -91.
        extents = [ctypes.c_int32(-1) for _ in range(8)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(value) for value in extents))
        return status, [value.value for value in extents]

    def test_two_lengths_and_recovery(self):
        self.assertEqual(self.calls['errors'], [])
        graph = build_circuit()
        self.assertEqual(graph['block_types']['source_stft_conv']['body']
                         ['ops'][0]['graph_slots']['inputs']['samples'],
                         'source_waveform')
        runtime = xray_support.xray_builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_runtime_source_stft_conv_bounded')
        self.assertEqual(runtime['sha256'],
                         hashlib.sha256(self.library.read_bytes()).hexdigest())
        cases = ((8, self.hello8, 108, 32400, 6481, 1080, 5,
                  'generator_harmonic_gaussian',
                  'generator_source_magnitude', 'generator_source_phase',
                  'decoder_generator_noise_convs_0'),
                 (36, self.source36, 206, 61800, 12361, 2060, 33,
                  'gaussian', 'magnitude', 'phase', 'first_source_conv'))
        arena = self.arena()
        first8 = None
        for count, reference, f0, samples, stft_frames, conv_frames, row, \
                gaussian_name, magnitude_name, phase_name, conv_name in (
                    cases[0], cases[1], cases[0]):
            ids = (self.hello8['word_ids'] if count == 8 else
                   self.fixture.encoder['word_ids'])
            predictor = (self.hello8['predictor_style'].ravel()
                         if count == 8 else
                         self.fixture.duration['predictor_style'])
            decoder = (self.hello8['decoder_style'].ravel()
                       if count == 8 else
                       self.generator36['decoder_style'].ravel())
            status, lengths = self.execute(arena, count, ids, predictor,
                                           decoder, reference[gaussian_name])
            self.assertEqual(status, 0)
            self.assertEqual(lengths, [count, count * 768, f0 // 2, f0,
                                       conv_frames, samples, stft_frames,
                                       conv_frames])
            channels = self.view(arena, 'source_stft_channels', (22, 15361))
            conv = self.view(arena, 'source_conv0', (256, 2560))
            self.assertTrue(np.isfinite(channels[:, :stft_frames]).all())
            self.assertTrue(np.isfinite(conv[:, :conv_frames]).all())
            self.assertTrue(np.all(channels[:, stft_frames:] == -91.))
            self.assertTrue(np.all(conv[:, conv_frames:] == -91.))
            if count == 8 and first8 is not None:
                np.testing.assert_array_equal(conv, first8)
                continue
            if count == 8:
                first8 = conv.copy()
            magnitude = reference[magnitude_name].reshape(11, stft_frames)
            phase = reference[phase_name].reshape(11, stft_frames)
            expected_conv = reference[conv_name].reshape(256, conv_frames)
            if count == 36:
                self.capture_and_replay_stft(arena, reference, samples,
                                             stft_frames, channels)
            mag_error = np.abs(channels[:11, :stft_frames] - magnitude)
            phase_error = np.abs(channels[11:, :stft_frames] - phase)
            conv_error = np.abs(conv[:, :conv_frames] - expected_conv)
            worst = np.unravel_index(np.argmax(conv_error), conv_error.shape)
            # The known raw-phase branch-cut mismatch remains numerical FAIL.
            # This lane proves connected execution/safety and reports that
            # numerical comparison separately until its contract is resolved.
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'kokoro.runtime-source-stft-conv.{count}',
                'name': 'runtime source/STFT/convolution versus direct pinned model',
                'provider': 'generated_kokoro_runtime_source_stft_conv',
                'dtype': 'fp32', 'direction': 'inference',
                'oracle': 'pinned PyTorch 2.8 direct full-model capture',
                'status': 'fail' if float(conv_error[worst]) > 5e-5 else 'pass',
                'gate': 'diagnostic', 'blocking': False,
                'reason': 'Connected source-convolution parity remains unresolved: generated F0 drift and raw STFT phase differences propagate into this output',
                'configuration': f'{count} IDs; af_heart row {row}; '
                    f'samples={samples}; captured Gaussian',
                'max_magnitude_error': float(mag_error.max()),
                'max_raw_phase_error': float(phase_error.max()),
                'max_conv_error': float(conv_error[worst]),
                'max_diff': float(conv_error[worst]),
                'rmse': float(np.sqrt(np.mean(conv_error.astype(np.float64) ** 2))),
                'actual': float(conv[worst]),
                'reference': float(expected_conv[worst]),
                'worst_index': list(map(int, worst)),
                'conv_tolerance': 5e-5,
                'tolerance': 5e-5,
                'reproduction_command': 'python3 -m unittest '
                    'tests.test_v8_kokoro_runtime_source_stft_conv',
            }, sort_keys=True))
        args = (8, self.hello8['word_ids'],
                self.hello8['predictor_style'].ravel(),
                self.hello8['decoder_style'].ravel(),
                self.hello8['generator_harmonic_gaussian'])
        bad = args[-1].copy()
        bad.reshape(-1)[0] = np.nan
        status, lengths = self.execute(arena, *args[:-1], bad)
        self.assertNotEqual(status, 0)
        self.assertEqual(lengths, [-1] * 8)
        self.assertTrue(np.all(self.view(arena, 'source_stft_channels',
                                          (22, 15361)) == -91.))
        self.assertTrue(np.all(self.view(arena, 'source_conv0',
                                          (256, 2560)) == -91.))
        status, lengths = self.execute(arena, *args)
        self.assertEqual((status, lengths[-1]), (0, 1080))
        np.testing.assert_array_equal(
            self.view(arena, 'source_conv0', (256, 2560)), first8)

    def capture_and_replay_stft(self, arena, reference, samples,
                                frames, channels):
        """Keep full STFT context at the first unresolved source boundary."""
        selected = [op for op in self.calls['operations']
                    if op.get('op') == 'audio_stft_mag_phase_checked']
        self.assertEqual(len(selected), 1)
        map_path = ROOT / 'version/v8/kernel_maps' / selected[0]['call_abi']['source_file']
        provider = json.loads(map_path.read_text())
        self.assertEqual(selected[0]['call_abi']['kernel_id'], provider['id'])
        self.assertEqual(selected[0]['function'], provider['impl']['function'])
        self.assertEqual(len(provider['numerical_capabilities']), 1)
        contract = provider['numerical_capabilities'][0]['contract_id']
        args = {item['name']: item for item in selected[0]['args']}
        static = {key: int(args[key]['expr']) for key in (
            'samples_capacity', 'window_capacity', 'table_capacity',
            'output_capacity', 'output_stride', 'scratch_capacity',
            'n_fft', 'hop')}
        self.assertEqual(args['samples_count']['source'], 'runtime:stft_samples_count')
        self.assertEqual(args['frames']['source'], 'runtime:stft_frames')
        samples_actual = self.view(arena, 'source_waveform',
                                   (static['samples_capacity'],))
        reference_samples = np.zeros_like(samples_actual)
        reference_samples[:samples] = reference['source'].reshape(-1)
        expected = np.ascontiguousarray(np.concatenate((
            reference['magnitude'].reshape(11, frames),
            reference['phase'].reshape(11, frames)), axis=0))
        retained = os.environ.get('CK_PRIVATE_STFT_PACKET_DIR')
        packet_dir = (Path(retained).expanduser() if retained else
                      self.root / 'source-stft-private-replay')
        packet = write_bundle(
            packet_dir,
            operation=provider['op'], provider=selected[0]['function'],
            contract=contract, runtime=self.library, embed_runtime=True,
            context={
                **static, 'samples_count': samples, 'frames': frames,
                'state': 'stateless',
                'reference_identity': {key: self.source_meta[key]
                    for key in ('model_pin', 'code_pin', 'asset_sha256')},
            },
            tensors={
                'samples': (samples_actual.copy(), 'fp32', ('sample',)),
                'reference_samples': (reference_samples,
                                      'fp32', ('sample',)),
                'window': (self.view(arena, 'source_stft_window', (20,)).copy(),
                           'fp32', ('tap',)),
                'cos_table': (self.view(arena, 'source_stft_cos', (11, 20)).copy(),
                              'fp32', ('bin', 'tap')),
                'sin_table': (self.view(arena, 'source_stft_sin', (11, 20)).copy(),
                              'fp32', ('bin', 'tap')),
                'cke_output': (channels.copy(), 'fp32',
                               ('channel', 'physical_frame')),
                'reference_output': (expected, 'fp32',
                                     ('channel', 'valid_frame')),
            })
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and not torch.__version__.startswith('2.8.'):
            torch = None
        result = replay(packet, runtime=self.library, torch_module=torch)
        self.assertEqual(result['capture_reproduced']['status'], 'PASS')
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.runtime-source-stft.same-input-packet.36',
            'name': 'private STFT packet: capture, same-input oracle, connected model',
            'provider': result['provider'], 'dtype': 'fp32',
            'numerical_contract': result['contract'],
            'runtime_sha256': result['runtime_sha256'],
            'direction': 'inference', 'oracle': 'pinned PyTorch 2.8',
            'status': 'fail' if result['same_input_oracle']['status'] == 'FAIL'
                else 'not_tested' if result['same_input_oracle']['status'] == 'NOT_TESTED'
                else 'pass',
            'max_diff': result['same_input_oracle'].get('max_abs'),
            'tolerance': result['same_input_oracle'].get('ceiling'),
            'gate': 'diagnostic', 'blocking': False,
            'configuration': '36 IDs; captured Gaussian; 61,800 samples; 12,361 frames',
            'capture_reproduced': result['capture_reproduced'],
            'upstream_source': result['upstream_source'],
            'same_input_oracle': result['same_input_oracle'],
            'same_input_magnitude': result.get('same_input_magnitude'),
            'same_input_raw_phase': result.get('same_input_raw_phase'),
            'cke_on_reference_inputs': result['cke_on_reference_inputs'],
            'reference_input_raw_phase': result.get('reference_input_raw_phase'),
            'connected_model': result['connected_model'],
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_kokoro_runtime_source_stft_conv',
        }, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
