"""Runtime Kokoro generator and F0-driven source projection in one entry."""

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
from tests.v8_kokoro_decoder_weight_support import decoder_weights_and_references
from tests.test_v8_kokoro_generated_decoder_complete import (
    checkpoint_metrics, require_matching_reference_identity)
from tests.test_v8_kokoro_generated_prosody_complete import verified_fixture
from tests import test_v8_kokoro_generated_albert_layer as xray_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from build_kokoro_runtime_source_projection_circuit import build_circuit


class KokoroRuntimeSourceProjectionTest(unittest.TestCase):

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
        for meta in (encode_meta, *decode_meta.values(), cls.hello_meta,
                     generator_meta, source_meta):
            require_matching_reference_identity(direct_meta, meta)
        if cls.hello_meta['voice_row_index'] != 5:
            raise RuntimeError('eight-ID source capture selected wrong voice row')
        for name, value in cls.generator36.items():
            if name.startswith('waveform_decoder.'):
                weights[name] = value
        weights['waveform_decoder.generator.m_source.l_linear.weight'] = (
            cls.source36['linear_weight'])
        weights['waveform_decoder.generator.m_source.l_linear.bias'] = (
            cls.source36['linear_bias'])
        cls.fixture = prepare_duration_fixture(cls.root, circuit, weights)
        cls.fixture.source['config']['activation_buffer_dtypes'].update({
            'request_length': 'i32', 'valid_length': 'i32',
            'valid_residual_elements': 'i32'})
        cls.fixture.source['config']['context_length'] = 64
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(cls.root, cls.fixture.source, circuit))
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + [
            ctypes.POINTER(ctypes.c_int32)] * 6
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
        output = self.view(arena, 'source_waveform', (76800, 1))
        output[:] = -91.0
        extents = [ctypes.c_int32(-1) for _ in range(6)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(value) for value in extents))
        return status, [value.value for value in extents], output[:, 0]

    def test_two_lengths_and_repeated_source_against_full_model(self):
        self.assertEqual(self.calls['errors'], [])
        graph = build_circuit()
        self.assertEqual(graph['block_types']['source_projection']['body']
                         ['ops'][0]['graph_slots']['inputs']['f0'], 'f0_output')
        runtime = xray_support.xray_builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_runtime_source_projection_bounded')
        self.assertEqual(runtime['sha256'],
                         hashlib.sha256(self.library.read_bytes()).hexdigest())
        cases = ((8, self.hello8, 54, 108, 1080, 32400, 5,
                  'generator_harmonic_gaussian', 'generator_source_samples',
                  'decoder_input_1'),
                 (36, self.source36, 103, 206, 2060, 61800, 33,
                  'gaussian', 'source', 'f0'))
        arena = self.arena()
        first8 = None
        for count, reference, frames, doubled, stage0, samples, row, \
                gaussian_name, source_name, f0_name in (
                    cases[0], cases[1], cases[0]):
            with self.subTest(ids=count):
                ids = (self.hello8['word_ids'] if count == 8 else
                       self.fixture.encoder['word_ids'])
                predictor = (self.hello8['predictor_style'].ravel()
                             if count == 8 else
                             self.fixture.duration['predictor_style'])
                decoder = (self.hello8['decoder_style'].ravel()
                           if count == 8 else
                           self.generator36['decoder_style'].ravel())
                status, extents, output = self.execute(arena, count, ids,
                    predictor, decoder, reference[gaussian_name])
                self.assertEqual(status, 0)
                self.assertEqual(extents,
                    [count, count * 768, frames, doubled, stage0, samples])
                main_reference = (self.hello8 if count == 8
                                  else self.generator36)
                main = self.view(arena, 'generator_stage0_upsample',
                                 (256, 2560))
                main_metrics = checkpoint_metrics(main[:, :stage0],
                    main_reference['decoder_generator_ups_0'][0])
                self.assertLessEqual(main_metrics['max_abs'], 5e-5,
                                     main_metrics)
                f0 = self.view(arena, 'f0_output', (1, 6400))[0, :doubled]
                f0_metrics = checkpoint_metrics(f0,
                    reference[f0_name].reshape(-1))
                source_metrics = checkpoint_metrics(output[:samples],
                    reference[source_name].reshape(-1))
                self.assertLessEqual(f0_metrics['max_abs'], 5e-4, f0_metrics)
                self.assertLessEqual(source_metrics['max_abs'], 1e-3,
                                     source_metrics)
                self.assertTrue(np.all(output[samples:] == -91.0))
                if count == 8:
                    if first8 is None:
                        first8 = output.copy()
                    else:
                        np.testing.assert_array_equal(output, first8)
                        continue
                print('CKE_NUMERICAL_CASE ' + json.dumps({
                    'case_id': f'kokoro.runtime-source-projection.{count}',
                    'name': 'runtime generated F0 source versus direct full-model capture',
                    'provider': 'generated_kokoro_runtime_source_projection',
                    'dtype': 'fp32', 'direction': 'inference',
                    'oracle': 'pinned PyTorch 2.8 direct full-model capture',
                    'status': 'pass', 'gate': 'composition', 'blocking': True,
                    'configuration': f'{count} IDs; af_heart row {row}; '
                        f'F0={doubled}; samples={samples}; captured Gaussian',
                    'max_diff': source_metrics['max_abs'],
                    'rmse': source_metrics['rmse'],
                    'worst_index': source_metrics['worst_index'],
                    'f0_max_diff': f0_metrics['max_abs'], 'tolerance': 1e-3,
                    'main_stage0_max_diff': main_metrics['max_abs'],
                    'reproduction_command': 'python3 -m unittest '
                        'tests.test_v8_kokoro_runtime_source_projection',
                }, sort_keys=True))

    def test_rejected_gaussian_then_recovery(self):
        arena = self.arena()
        reference = self.hello8
        args = (8, reference['word_ids'],
                reference['predictor_style'].ravel(),
                reference['decoder_style'].ravel(),
                reference['generator_harmonic_gaussian'])
        status, extents, output = self.execute(arena, *args)
        self.assertEqual((status, extents[-1]), (0, 32400))
        baseline = output.copy()
        bad = args[-1].copy()
        bad.reshape(-1)[0] = np.nan
        status, extents, output = self.execute(arena, *args[:-1], bad)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1] * 6)
        self.assertTrue(np.all(output == -91.0))
        status, extents, output = self.execute(arena, *args)
        self.assertEqual((status, extents[-1]), (0, 32400))
        np.testing.assert_array_equal(output, baseline)


if __name__ == '__main__':
    unittest.main()
