"""Two Kokoro phoneme lengths through generated second-stage ingress."""

import ctypes
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_duration_fixture_support import prepare_duration_fixture
from tests.test_v8_kokoro_generator_stage0_join import (
    join_weights_and_references)
from tests.test_v8_kokoro_generator_stage0_pool import stage0_pool_weights
from tests.test_v8_kokoro_generated_decoder_complete import (
    checkpoint_metrics, require_matching_reference_identity)
from tests.test_v8_kokoro_generated_prosody_complete import verified_fixture
from tests import test_v8_kokoro_runtime_stage0_pool as pool_support
from tests import test_v8_kokoro_source_residual_first_pair as fixture_support
from tests import test_v8_kokoro_generated_albert_layer as xray_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from build_kokoro_runtime_stage1_ingress_circuit import build_circuit


class KokoroRuntimeStage1IngressTest(unittest.TestCase):
    view = pool_support.KokoroRuntimeStage0PoolTest.view
    arena = pool_support.KokoroRuntimeStage0PoolTest.arena

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        graph = build_circuit()
        cls.circuit = cls.root / 'circuit.json'
        cls.circuit.write_text(json.dumps(graph))
        cls.refs = join_weights_and_references()
        cls.hello8, hello_meta = verified_fixture('runtime_decoder_hello_8')
        first, first_meta = fixture_support.load_fixture(
            'kokoro_generator_main_resblock0_pinned')
        tail, tail_meta = fixture_support.load_fixture(
            'kokoro_generator_stage0_pool_pinned')
        cls.stage1, stage1_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_ingress_pinned')
        require_matching_reference_identity(cls.refs.join_meta, hello_meta)
        for meta in (first_meta, tail_meta, stage1_meta):
            for field in ('model_pin', 'code_pin', 'asset_sha256'):
                if meta[field] != cls.refs.join_meta[field]:
                    raise RuntimeError(f'generator ingress {field} identity differs')
        weights = stage0_pool_weights(cls.refs, first, tail)
        weights['waveform_decoder.generator.ups.1.weight'] = \
            cls.stage1['weight']
        weights['waveform_decoder.generator.ups.1.bias'] = \
            cls.stage1['bias']
        cls.fixture = prepare_duration_fixture(
            cls.root, cls.circuit, weights)
        cls.fixture.source['config']['activation_buffer_dtypes'].update({
            'request_length': 'i32', 'valid_length': 'i32',
            'valid_residual_elements': 'i32',
            'generator_stage1_deconv_extent_value': 'i32',
            'generator_stage1_pad_extent_value': 'i32'})
        cls.fixture.source['config']['context_length'] = 64
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(cls.root, cls.fixture.source, cls.circuit))
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + [
            ctypes.POINTER(ctypes.c_int32)] * 9
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}
        cls.weights = {item['name']: item for item in
                       cls.layout['memory']['weights']['entries']}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

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
        for name, shape in (
                ('generator_stage1_activation', (256, 2560)),
                ('generator_stage1_upsample', (128, 15360)),
                ('generator_stage1_reflection', (128, 15361))):
            self.view(arena, name, shape)[:] = -91.
        lengths = [ctypes.c_int32(-1) for _ in range(9)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(value) for value in lengths))
        return status, [value.value for value in lengths]

    def test_two_lengths_stage1_ingress_and_recovery(self):
        graph = build_circuit()
        self.assertEqual(graph['sequence'][-1], 'generator_stage1_ingress')
        self.assertEqual(self.calls['errors'], [])
        self.assertEqual(self.calls['operations'][-1]['function'],
                         'audio_reflect_pad1d_left_channel_major_f32_checked')
        runtime = xray_support.xray_builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_runtime_stage1_ingress_bounded')
        self.assertEqual(runtime['sha256'],
                         hashlib.sha256(self.library.read_bytes()).hexdigest())

        arena = self.arena()
        first8 = None
        for count, reference, frames, stage0, samples, stft, row, noise_name in (
                (8, self.hello8, 54, 1080, 32400, 6481, 5,
                 'generator_harmonic_gaussian'),
                (36, self.refs.source, 103, 2060, 61800, 12361, 33,
                 'gaussian'),
                (8, self.hello8, 54, 1080, 32400, 6481, 5,
                 'generator_harmonic_gaussian')):
            ids = (self.hello8['word_ids'] if count == 8 else
                   self.fixture.encoder['word_ids'])
            predictor = (self.hello8['predictor_style'].ravel()
                         if count == 8 else
                         self.fixture.duration['predictor_style'])
            decoder = (self.hello8['decoder_style'].ravel()
                       if count == 8 else
                       self.refs.main['decoder_style'].ravel())
            status, lengths = self.execute(arena, count, ids, predictor,
                                           decoder, reference[noise_name])
            self.assertEqual(status, 0)
            self.assertEqual(lengths, [count, count * 768, frames, frames * 2,
                stage0, samples, stft, stage0 * 6, stage0 * 6 + 1])
            activation = self.view(arena, 'generator_stage1_activation',
                                   (256, 2560))
            deconv = self.view(arena, 'generator_stage1_upsample',
                               (128, 15360))
            output = self.view(arena, 'generator_stage1_reflection',
                               (128, 15361))
            self.assertTrue(np.isfinite(output[:, :stage0 * 6 + 1]).all())
            self.assertTrue(np.all(activation[:, stage0:] == -91.))
            self.assertTrue(np.all(deconv[:, stage0 * 6:] == -91.))
            self.assertTrue(np.all(output[:, stage0 * 6 + 1:] == -91.))
            np.testing.assert_array_equal(output[:, 0], deconv[:, 1])
            np.testing.assert_array_equal(output[:, 1:stage0 * 6 + 1],
                                          deconv[:, :stage0 * 6])
            if count == 8 and first8 is not None:
                np.testing.assert_array_equal(output, first8)
                continue
            if count == 8:
                first8 = output.copy()
                expected = (reference['decoder_generator_stage1_activation'][0],
                    reference['decoder_generator_ups_1'][0],
                    reference['decoder_generator_reflection_pad'][0])
            else:
                expected = (self.stage1['activation'][0],
                    self.stage1['deconv'][0], self.stage1['reflection'][0])
            metrics = {name: checkpoint_metrics(actual, target) for
                name, actual, target in zip(('activation', 'deconv', 'reflection'),
                    (activation[:, :stage0], deconv[:, :stage0 * 6],
                     output[:, :stage0 * 6 + 1]), expected)}
            final = metrics['reflection']
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'kokoro.runtime-stage1-ingress.{count}',
                'name': 'variable-length second generator ingress versus full model',
                'provider': 'generated_kokoro_runtime_stage1_ingress',
                'dtype': 'fp32', 'direction': 'inference',
                'oracle': 'pinned PyTorch 2.8 direct full-model captures',
                'status': 'fail' if final['max_abs'] > 2e-4 else 'pass',
                'gate': 'diagnostic', 'blocking': False,
                'reason': 'Connected first-stage source and residual-pool '
                          'numerical differences propagate into stage two',
                'configuration': f'{count} IDs; af_heart row {row}; '
                    f'frames={stage0 * 6 + 1}; captured Gaussian',
                'max_diff': final['max_abs'], 'tolerance': 2e-4,
                'rmse': final['rmse'], 'worst_index': final['worst_index'],
                'actual': final['actual'], 'reference': final['reference'],
                'checkpoint_metrics': metrics,
                'reproduction_command': 'python3 -m unittest '
                    'tests.test_v8_kokoro_runtime_stage1_ingress',
            }, sort_keys=True))

        bad = self.hello8['generator_harmonic_gaussian'].copy()
        bad.reshape(-1)[0] = np.nan
        args = (8, self.hello8['word_ids'],
                self.hello8['predictor_style'].ravel(),
                self.hello8['decoder_style'].ravel())
        status, lengths = self.execute(arena, *args, bad)
        self.assertNotEqual(status, 0)
        self.assertEqual(lengths, [-1] * 9)
        output = self.view(arena, 'generator_stage1_reflection', (128, 15361))
        self.assertTrue(np.all(output == -91.))
        self.view(arena, 'source_gaussian', (76800, 9))[0, 0] = \
            self.hello8['generator_harmonic_gaussian'].reshape(-1, 9)[0, 0]
        self.view(arena, 'request_length', (1,), np.int32)[0] = 65
        published = [ctypes.c_int32(-1) for _ in range(9)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(value) for value in published))
        self.assertNotEqual(status, 0)
        self.assertEqual([value.value for value in published], [-1] * 9)
        self.assertTrue(np.all(output == -91.))
        status, lengths = self.execute(arena, *args,
            self.hello8['generator_harmonic_gaussian'])
        self.assertEqual((status, lengths[-1]), (0, 6481))
        np.testing.assert_array_equal(output, first8)


if __name__ == '__main__':
    unittest.main()
