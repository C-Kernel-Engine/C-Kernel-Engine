"""Two phoneme lengths through Kokoro's generated first generator pool."""

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
from tests import test_v8_kokoro_runtime_stage0_join as join_support
from tests import test_v8_kokoro_source_residual_first_pair as fixture_support
from tests import test_v8_kokoro_generated_albert_layer as xray_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from build_kokoro_runtime_stage0_pool_circuit import build_circuit


class KokoroRuntimeStage0PoolTest(unittest.TestCase):
    view = join_support.KokoroRuntimeStage0JoinTest.view
    arena = join_support.KokoroRuntimeStage0JoinTest.arena
    execute = join_support.KokoroRuntimeStage0JoinTest.execute

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        graph = build_circuit()
        cls.circuit = cls.root / 'circuit.json'
        cls.circuit.write_text(json.dumps(graph))
        cls.refs = join_weights_and_references()
        cls.hello8, hello_meta = verified_fixture('runtime_decoder_hello_8')
        cls.first, first_meta = fixture_support.load_fixture(
            'kokoro_generator_main_resblock0_pinned')
        cls.tail, tail_meta = fixture_support.load_fixture(
            'kokoro_generator_stage0_pool_pinned')
        require_matching_reference_identity(cls.refs.join_meta, hello_meta)
        for meta in (first_meta, tail_meta):
            for field in ('model_pin', 'code_pin', 'asset_sha256'):
                if meta[field] != cls.refs.join_meta[field]:
                    raise RuntimeError(f'generator pool {field} identity differs')
        weights = stage0_pool_weights(cls.refs, cls.first, cls.tail)
        cls.fixture = prepare_duration_fixture(
            cls.root, cls.circuit, weights)
        cls.fixture.source['config']['activation_buffer_dtypes'].update({
            'request_length': 'i32', 'valid_length': 'i32',
            'valid_residual_elements': 'i32'})
        cls.fixture.source['config']['context_length'] = 64
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(cls.root, cls.fixture.source, cls.circuit))
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + [
            ctypes.POINTER(ctypes.c_int32)] * 7
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}
        cls.weights = {item['name']: item for item in
                       cls.layout['memory']['weights']['entries']}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_two_lengths_pool_parity_and_recovery(self):
        graph = build_circuit()
        self.assertEqual(graph['sequence'][-1], 'generator_stage0_pool')
        self.assertEqual(self.calls['errors'], [])
        self.assertEqual(self.calls['operations'][-1]['function'],
                         'audio_scaled_sum_strided_f32_checked')
        runtime = xray_support.xray_builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_runtime_stage0_pool_bounded')
        self.assertEqual(runtime['sha256'],
                         hashlib.sha256(self.library.read_bytes()).hexdigest())

        arena = self.arena()
        first8 = None
        for count, reference, frames, stage_frames, samples, stft_frames, \
                row, gaussian_name in (
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
            mean = self.view(arena, 'generator_stage0_residual_mean',
                             (256, 2560))
            mean[:] = -91.
            status, lengths = self.execute(arena, count, ids, predictor,
                                           decoder, reference[gaussian_name])
            self.assertEqual(status, 0)
            self.assertEqual(lengths, [count, count * 768, frames, frames * 2,
                                       stage_frames, samples, stft_frames])
            self.assertTrue(np.isfinite(mean[:, :stage_frames]).all())
            self.assertTrue(np.all(mean[:, stage_frames:] == -91.))
            block0 = self.view(arena, 'main_res_pair2_output', (256, 2560))
            block1 = self.view(arena, 'main_resblock1_pair2_output',
                               (256, 2560))
            block2 = self.view(arena, 'main_resblock2_pair2_output',
                               (256, 2560))
            sum01 = self.view(arena, 'generator_stage0_residual_sum01',
                              (256, 2560))
            np.testing.assert_array_equal(sum01[:, :stage_frames],
                (block0[:, :stage_frames] + block1[:, :stage_frames])
                .astype(np.float32))
            np.testing.assert_allclose(mean[:, :stage_frames],
                ((sum01[:, :stage_frames] + block2[:, :stage_frames]) *
                 np.float32(1. / 3.)).astype(np.float32), rtol=0, atol=1e-6)
            if count == 8 and first8 is not None:
                np.testing.assert_array_equal(mean, first8)
                continue
            if count == 8:
                first8 = mean.copy()
                expected_blocks = [reference[
                    f'decoder_generator_resblocks_{block}'][0]
                    for block in range(3)]
                expected_mean = ((expected_blocks[0] + expected_blocks[1] +
                    expected_blocks[2]) / np.float32(3.)).astype(np.float32)
            else:
                expected_blocks = [self.first['pair2_output'][0],
                    self.tail['block1_pair2_output'][0],
                    self.tail['block2_pair2_output'][0]]
                expected_mean = self.tail['stage0_mean'][0]
            block_metrics = [checkpoint_metrics(actual[:, :stage_frames],
                expected) for actual, expected in zip((block0, block1, block2),
                                                       expected_blocks)]
            metrics = checkpoint_metrics(mean[:, :stage_frames], expected_mean)
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'kokoro.runtime-stage0-pool.{count}',
                'name': 'variable-length first generator pool versus full model',
                'provider': 'generated_kokoro_runtime_stage0_pool',
                'dtype': 'fp32', 'direction': 'inference',
                'oracle': 'pinned PyTorch 2.8 direct block captures and '
                          'explicit three-way mean',
                'status': 'fail' if metrics['max_abs'] > 2e-4 else 'pass',
                'gate': 'diagnostic', 'blocking': False,
                'reason': 'The connected source branch inherits unresolved '
                          'generated F0 and raw-phase differences',
                'configuration': f'{count} IDs; af_heart row {row}; '
                    f'frames={stage_frames}; captured Gaussian',
                'max_diff': metrics['max_abs'], 'tolerance': 2e-4,
                'rmse': metrics['rmse'],
                'worst_index': metrics['worst_index'],
                'actual': metrics['actual'], 'reference': metrics['reference'],
                'block_metrics': {f'resblock{block}': item for block, item in
                    enumerate(block_metrics)},
                'reproduction_command': 'python3 -m unittest '
                    'tests.test_v8_kokoro_runtime_stage0_pool',
            }, sort_keys=True))

        bad = self.hello8['generator_harmonic_gaussian'].copy()
        bad.reshape(-1)[0] = np.nan
        mean[:] = -91.
        args = (8, self.hello8['word_ids'],
                self.hello8['predictor_style'].ravel(),
                self.hello8['decoder_style'].ravel())
        status, lengths = self.execute(arena, *args, bad)
        self.assertNotEqual(status, 0)
        self.assertEqual(lengths, [-1] * 7)
        self.assertTrue(np.all(mean == -91.))
        self.view(arena, 'source_gaussian', (76800, 9))[0, 0] = \
            self.hello8['generator_harmonic_gaussian'].reshape(-1, 9)[0, 0]
        self.view(arena, 'request_length', (1,), np.int32)[0] = 65
        published = [ctypes.c_int32(-1) for _ in range(7)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(value) for value in published))
        self.assertNotEqual(status, 0)
        self.assertEqual([value.value for value in published], [-1] * 7)
        self.assertTrue(np.all(mean == -91.))
        status, lengths = self.execute(arena, *args,
            self.hello8['generator_harmonic_gaussian'])
        self.assertEqual((status, lengths[4]), (0, 1080))
        np.testing.assert_array_equal(mean, first8)


if __name__ == '__main__':
    unittest.main()
