"""Connected phonemes to F0 and source projection using captured Gaussian input."""

import ctypes
import hashlib
import json
from pathlib import Path
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
import build_kokoro_generated_f0_source_circuit as author


class KokoroGeneratedF0SourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        source_path = ROOT / 'tests/fixtures/tts/kokoro_source_pinned.npz'
        cls.source_meta = json.loads(source_path.with_suffix('.json').read_text())
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != \
                cls.source_meta['fixture_sha256']:
            raise RuntimeError('pinned source fixture mismatch')
        with np.load(source_path) as archive:
            cls.oracle = {key: archive[key].copy() for key in archive.files}
        weights, cls.prosody_oracle, cls.prosody_meta = \
            complete_prosody_weights()
        if (cls.source_meta['model_pin'] != cls.prosody_meta['model_pin'] or
            cls.source_meta['code_pin'] != cls.prosody_meta['code_pin'] or
            cls.source_meta['asset_sha256'] != cls.prosody_meta['asset_sha256']):
            raise RuntimeError('source and prosody oracle identities differ')
        weights['waveform_decoder.generator.m_source.l_linear.weight'] = \
            cls.oracle['linear_weight']
        weights['waveform_decoder.generator.m_source.l_linear.bias'] = \
            cls.oracle['linear_bias']
        circuit = materialize_stage_circuit(cls.root, author)
        fixture = prepare_duration_fixture(cls.root, circuit, weights)
        for name, value in vars(fixture).items():
            setattr(cls, name, value)
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = \
            compile_native_graph(cls.root, cls.source, circuit)
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32)]
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
        gaussian = self.view(arena, 'source_gaussian').reshape(76800, 9)
        gaussian[:61800] = self.oracle['gaussian'].reshape(61800, 9)
        self.view(arena, 'source_waveform')[:] = -91.
        return arena

    def execute(self, arena):
        lengths = [ctypes.c_int32(-999) for _ in range(3)]
        status = self.fn(arena, len(arena), *(ctypes.byref(x) for x in lengths))
        return status, tuple(x.value for x in lengths)

    def test_connected_generated_f0_source(self):
        self.assertEqual(self.calls['errors'], [])
        self.assertEqual([op['function'] for op in self.calls['operations'][-4:]],
            ['ck_runtime_scale_i32_checked', 'audio_harmonic_source_weighted_fma_checked_f32',
             'linear_rows_checked_f32', 'tanh_strided_f32_checked'])
        arena = self.arena()
        status, lengths = self.execute(arena)
        self.assertEqual(status, 0)
        self.assertEqual(lengths, (103, 206, 61800))
        f0 = self.view(arena, 'f0_output').reshape(1, 256)[0, :206]
        expected_f0 = self.oracle['f0'].reshape(-1)
        f0_error = np.abs(f0 - expected_f0)
        self.assertTrue(np.isfinite(f0).all())
        self.assertLessEqual(float(np.max(f0_error)), 5e-4)
        source = self.view(arena, 'source_waveform')[:61800]
        expected_source = self.oracle['source'].reshape(-1)
        self.assertTrue(np.isfinite(source).all())
        error = np.abs(source - expected_source)
        worst = int(np.argmax(error))
        self.assertLessEqual(float(error[worst]), 1e-3,
            (worst, float(source[worst]), float(expected_source[worst])))
        f0_worst = int(np.argmax(f0_error))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.generated-f0-source.connected-v1',
            'name': 'connected generated F0 through harmonic linear tanh source',
            'provider': 'generated_kokoro_f0_source',
            'oracle': 'direct-pinned-full-model-pytorch28',
            'backend_version': self.source_meta['environment']['torch'],
            'status': 'pass',
            'max_diff': float(error[worst]), 'tolerance': 1e-3,
            'worst_index': worst, 'actual': float(source[worst]),
            'reference': float(expected_source[worst]),
            'rmse': float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
            'f0_max_diff': float(f0_error[f0_worst]),
            'f0_worst_index': f0_worst,
            'configuration': '36 phonemes; 206 F0 frames; captured Gaussian',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        self.assertTrue(np.all(self.view(arena, 'source_waveform')[61800:] == -91.))

    def test_repeated_a_b_a_and_failed_source_recovery(self):
        alternate, alternate_meta = verified_fixture('generator_stage0_your')
        self.assertEqual(alternate_meta['model_pin'], self.source_meta['model_pin'])
        self.assertEqual(alternate_meta['asset_sha256'],
                         self.source_meta['asset_sha256'])
        arena = self.arena()
        self.assertEqual(self.execute(arena), (0, (103, 206, 61800)))
        first = self.view(arena, 'source_waveform')[:61800].copy()
        gaussian = self.view(arena, 'source_gaussian').reshape(76800, 9)
        saved = float(gaussian[0, 0])
        gaussian[0, 0] = np.nan
        self.view(arena, 'source_waveform')[:] = -91.
        status, lengths = self.execute(arena)
        self.assertNotEqual(status, 0)
        self.assertTrue(np.all(self.view(arena, 'source_waveform') == -91.))
        self.assertEqual(lengths[2], -999)
        gaussian[0, 0] = saved
        self.view(arena, 'word_ids', np.int32)[:36] = alternate['word_ids']
        self.view(arena, 'predictor_style')[:128] = alternate['predictor_style']
        self.view(arena, 'source_waveform')[:] = -91.
        self.assertEqual(self.execute(arena), (0, (98, 196, 58800)))
        second = self.view(arena, 'source_waveform')
        self.assertTrue(np.isfinite(second[:58800]).all())
        self.assertTrue(np.all(second[58800:] == -91.))
        self.view(arena, 'word_ids', np.int32)[:36] = self.encoder['word_ids']
        self.view(arena, 'predictor_style')[:128] = self.duration['predictor_style']
        self.view(arena, 'source_waveform')[:] = -91.
        self.assertEqual(self.execute(arena), (0, (103, 206, 61800)))
        np.testing.assert_array_equal(self.view(arena, 'source_waveform')[:61800],
                                      first)


if __name__ == '__main__':
    unittest.main()
