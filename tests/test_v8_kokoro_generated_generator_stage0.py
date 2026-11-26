"""Connected Kokoro decoder -> first generator upsample versus pinned hook."""

import ctypes
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
from tests.test_v8_kokoro_generated_prosody_complete import verified_fixture
from tests.test_v8_kokoro_generated_decoder_complete import (
    checkpoint_metrics, require_matching_reference_identity)
from tests.v8_kokoro_decoder_weight_support import decoder_weights_and_references


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_generator_stage0_circuit as author

GENERATOR_STAGE0_TOLERANCE = 5e-5


class KokoroGeneratorStage0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (weights, direct, direct_meta, encode, encode_meta,
         decode, decode_meta) = decoder_weights_and_references()
        cls.reference, cls.reference_meta = verified_fixture('generator_stage0')
        for meta in (encode_meta, *decode_meta.values(), cls.reference_meta):
            require_matching_reference_identity(direct_meta, meta)
        for name, value in cls.reference.items():
            if name.startswith('waveform_decoder.'):
                weights[name] = value
        circuit = materialize_stage_circuit(cls.root, author)
        fixture = prepare_duration_fixture(cls.root, circuit, weights)
        cls.encoder = fixture.encoder
        cls.duration = fixture.duration
        cls.entries = fixture.entries
        cls.bump = fixture.bump
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = \
            compile_native_graph(cls.root, fixture.source, circuit)
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32)]
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}
        cls.weight_layout = {item['name']: item for item in
                             cls.layout['memory']['weights']['entries']}
        cls.decoder_style = encode['decoder_style'].ravel()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def view(self, arena, name):
        item = self.buffers[name]
        return np.ndarray((item['size'] // 4,), np.float32, buffer=arena,
                          offset=item['abs_offset'])

    def arena(self):
        arena = populated_duration_arena(self.layout, self.entries, self.bump,
                                         self.encoder, self.duration)
        self.view(arena, 'decoder_style')[:] = self.decoder_style
        return arena

    def run_entry(self, arena):
        lengths = [ctypes.c_int32(-999) for _ in range(3)]
        status = self.fn(arena, len(arena), *(ctypes.byref(x) for x in lengths))
        return status, tuple(x.value for x in lengths)

    def test_connected_stage_against_direct_full_model_hook(self):
        self.assertEqual(self.calls['errors'], [])
        self.assertEqual(self.calls['operations'][-1]['function'],
                         'audio_conv_transpose1d_dense_channel_major_f32_checked')
        arena = self.arena()
        self.view(arena, 'generator_stage0_upsample')[:] = -91.
        status, lengths = self.run_entry(arena)
        self.assertEqual(status, 0)
        self.assertEqual(lengths, (103, 206, 2060))
        actual = self.view(arena, 'generator_stage0_upsample').reshape(256, 2560)
        expected = self.reference['decoder_generator_ups_0'][0]
        metrics = checkpoint_metrics(actual[:, :2060], expected)
        self.assertLessEqual(metrics['max_abs'], GENERATOR_STAGE0_TOLERANCE,
                             metrics)
        self.assertTrue(np.all(actual[:, 2060:] == -91.))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.generator-stage0.connected-v1',
            'name': 'connected generated first generator upsample versus direct pinned hook',
            'provider': 'generated_kokoro_generator_stage0',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pinned-full-kmodel-pytorch28',
            'backend_version': self.reference_meta['environment']['torch'],
            'status': 'pass', 'max_diff': metrics['max_abs'],
            'worst_index': metrics['worst_index'],
            'rmse': metrics['rmse'], 'tolerance': GENERATOR_STAGE0_TOLERANCE,
            'configuration': '36 phonemes; 206/256 input frames; 2060/2560 output frames; af_heart',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_distinct_utterance_and_repeated_request(self):
        reference, meta = verified_fixture('generator_stage0_your')
        require_matching_reference_identity(self.reference_meta, meta)
        self.assertEqual(meta['effective_weight_sha256'],
                         self.reference_meta['effective_weight_sha256'])
        self.assertFalse(meta['weights_included'])
        arena = self.arena()
        self.view(arena, 'word_ids').view(np.int32)[:] = reference['word_ids']
        self.view(arena, 'predictor_style')[:] = reference['predictor_style'].ravel()
        self.view(arena, 'decoder_style')[:] = reference['decoder_style'].ravel()
        output = self.view(arena, 'generator_stage0_upsample').reshape(256, 2560)
        output[:] = -91.
        status, lengths = self.run_entry(arena)
        self.assertEqual(status, 0)
        self.assertEqual(lengths, (98, 196, 1960))
        metrics = checkpoint_metrics(output[:, :1960],
            reference['decoder_generator_ups_0'][0])
        self.assertLessEqual(metrics['max_abs'], GENERATOR_STAGE0_TOLERANCE,
                             metrics)
        self.assertTrue(np.all(output[:, 1960:] == -91.))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.generator-stage0.your-utterance-v1',
            'name': 'connected generated first generator upsample for distinct utterance',
            'provider': 'generated_kokoro_generator_stage0',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pinned-full-kmodel-pytorch28',
            'backend_version': meta['environment']['torch'],
            'status': 'pass', 'max_diff': metrics['max_abs'],
            'worst_index': metrics['worst_index'], 'rmse': metrics['rmse'],
            'tolerance': GENERATOR_STAGE0_TOLERANCE,
            'configuration': '36 phonemes; 196/256 input frames; 1960/2560 output frames; af_heart',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        self.view(arena, 'word_ids').view(np.int32)[:] = self.encoder['word_ids']
        self.view(arena, 'predictor_style')[:] = self.duration['predictor_style'].ravel()
        self.view(arena, 'decoder_style')[:] = self.decoder_style
        output[:] = -91.
        status, lengths = self.run_entry(arena)
        self.assertEqual((status, lengths), (0, (103, 206, 2060)))
        self.assertLessEqual(checkpoint_metrics(output[:, :2060],
            self.reference['decoder_generator_ups_0'][0])['max_abs'],
            GENERATOR_STAGE0_TOLERANCE)

    def test_failed_upstream_does_not_publish_generator_extent(self):
        arena = self.arena()
        item = self.weight_layout['duration_prosody.duration_head.weight']
        np.ndarray((item['size'] // 4,), np.float32, buffer=arena,
                   offset=item['abs_offset'])[:] = 0.
        item = self.weight_layout['duration_prosody.duration_head.bias']
        np.ndarray((item['size'] // 4,), np.float32, buffer=arena,
                   offset=item['abs_offset'])[:] = 1.
        self.view(arena, 'generator_stage0_upsample')[:] = -91.
        status, lengths = self.run_entry(arena)
        self.assertNotEqual(status, 0)
        self.assertEqual(lengths, (-999, -999, -999))
        self.assertTrue(np.all(self.view(arena,
            'generator_stage0_upsample') == -91.))

    def test_failed_generator_weight_then_recovery(self):
        arena = self.arena()
        item = self.weight_layout['waveform_decoder.generator.ups.0.weight']
        weight = np.ndarray((item['size'] // 4,), np.float32, buffer=arena,
                            offset=item['abs_offset'])
        original = weight[0].copy()
        weight[0] = np.nan
        output = self.view(arena, 'generator_stage0_upsample')
        output[:] = -91.
        status, lengths = self.run_entry(arena)
        self.assertNotEqual(status, 0)
        self.assertEqual(lengths, (-999, -999, -999))
        self.assertTrue(np.all(output == -91.))
        weight[0] = original
        status, lengths = self.run_entry(arena)
        self.assertEqual((status, lengths), (0, (103, 206, 2060)))
        self.assertLessEqual(checkpoint_metrics(output.reshape(256, 2560)[:, :2060],
            self.reference['decoder_generator_ups_0'][0])['max_abs'],
            GENERATOR_STAGE0_TOLERANCE)


if __name__ == '__main__':
    unittest.main()
