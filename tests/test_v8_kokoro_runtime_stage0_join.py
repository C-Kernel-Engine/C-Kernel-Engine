"""Variable-length generated first Kokoro source/main generator join."""

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
from tests.test_v8_kokoro_generated_decoder_complete import (
    checkpoint_metrics, require_matching_reference_identity)
from tests.test_v8_kokoro_generated_prosody_complete import verified_fixture
from tests import test_v8_kokoro_generated_albert_layer as xray_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from build_kokoro_runtime_stage0_join_circuit import OUTPUT, build_circuit


class KokoroRuntimeStage0JoinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        graph = build_circuit()
        if OUTPUT.read_text() != json.dumps(graph, indent=2) + '\n':
            raise RuntimeError('committed runtime stage0 join differs')
        cls.circuit = cls.root / 'circuit.json'
        cls.circuit.write_text(json.dumps(graph))
        cls.refs = join_weights_and_references()
        cls.hello8, hello_meta = verified_fixture('runtime_decoder_hello_8')
        require_matching_reference_identity(cls.refs.join_meta, hello_meta)
        cls.fixture = prepare_duration_fixture(
            cls.root, cls.circuit, cls.refs.weights)
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
        self.view(arena, 'source_stft_window', (20,))[:] = \
            self.refs.source['window']
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
        self.view(arena, 'generator_stage0_join', (256, 2560))[:] = -91.
        lengths = [ctypes.c_int32(-1) for _ in range(7)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(value) for value in lengths))
        return status, [value.value for value in lengths]

    def test_two_lengths_shared_extent_and_recovery(self):
        graph = build_circuit()
        self.assertNotIn('source_conv_frames', graph['runtime_lengths'])
        source_conv = graph['block_types']['source_stft_conv']['footer'][0]
        self.assertEqual(source_conv['runtime_scalar_bindings']
                         ['conv_output_frames'], 'generator_stage0_frames')
        self.assertEqual(graph['block_types']['generator_stage0_join']
                         ['body']['ops'][0]['runtime_scalar_bindings']
                         ['sum_columns'], 'generator_stage0_frames')
        self.assertEqual(self.calls['errors'], [])
        self.assertEqual(self.calls['operations'][-1]['function'],
                         'audio_scaled_sum_strided_f32_checked')
        runtime = xray_support.xray_builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_runtime_stage0_join_bounded')
        self.assertEqual(runtime['sha256'],
                         hashlib.sha256(self.library.read_bytes()).hexdigest())

        arena = self.arena()
        first8 = None
        cases = ((8, self.hello8, 54, 1080, 32400, 6481, 5,
                  'generator_harmonic_gaussian',
                  'decoder_generator_noise_res_0',
                  'decoder_generator_stage0_join'),
                 (36, self.refs.source, 103, 2060, 61800, 12361, 33,
                  'gaussian', None, None))
        for count, reference, frames, stage_frames, samples, stft_frames, \
                row, gaussian_name, source_name, join_name in (
                    cases[0], cases[1], cases[0]):
            ids = (self.hello8['word_ids'] if count == 8 else
                   self.fixture.encoder['word_ids'])
            predictor = (self.hello8['predictor_style'].ravel()
                         if count == 8 else
                         self.fixture.duration['predictor_style'])
            decoder = (self.hello8['decoder_style'].ravel()
                       if count == 8 else
                       self.refs.main['decoder_style'].ravel())
            status, lengths = self.execute(arena, count, ids, predictor,
                                           decoder, reference[gaussian_name])
            self.assertEqual(status, 0)
            self.assertEqual(lengths, [count, count * 768, frames, frames * 2,
                                       stage_frames, samples, stft_frames])
            main = self.view(arena, 'generator_stage0_upsample', (256, 2560))
            source = self.view(arena, 'source_res_pair2_output', (256, 2560))
            join = self.view(arena, 'generator_stage0_join', (256, 2560))
            self.assertTrue(np.isfinite(join[:, :stage_frames]).all())
            self.assertTrue(np.all(join[:, stage_frames:] == -91.))
            np.testing.assert_array_equal(join[:, :stage_frames],
                (main[:, :stage_frames] + source[:, :stage_frames])
                .astype(np.float32))
            if count == 8 and first8 is not None:
                np.testing.assert_array_equal(join, first8)
                continue
            if count == 8:
                first8 = join.copy()
            expected_source = (reference[source_name][0] if count == 8
                               else self.refs.block['pair2_output'][0])
            expected_join = (reference[join_name][0] if count == 8
                             else self.refs.join['join'][0])
            main_ref = (self.hello8 if count == 8 else self.refs.main)
            main_metrics = checkpoint_metrics(main[:, :stage_frames],
                main_ref['decoder_generator_ups_0'][0])
            source_metrics = checkpoint_metrics(source[:, :stage_frames],
                                                 expected_source)
            metrics = checkpoint_metrics(join[:, :stage_frames], expected_join)
            self.assertLessEqual(main_metrics['max_abs'], 5e-5, main_metrics)
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'kokoro.runtime-stage0-join.{count}',
                'name': 'variable-length first generator join versus full model',
                'provider': 'generated_kokoro_runtime_stage0_join',
                'dtype': 'fp32', 'direction': 'inference',
                'oracle': 'pinned PyTorch 2.8 direct full-model capture',
                'status': 'fail' if metrics['max_abs'] > 2e-4 else 'pass',
                'gate': 'diagnostic', 'blocking': False,
                'configuration': f'{count} IDs; af_heart row {row}; '
                    f'frames={stage_frames}; captured Gaussian',
                'max_diff': metrics['max_abs'], 'tolerance': 2e-4,
                'rmse': metrics['rmse'],
                'worst_index': metrics['worst_index'],
                'source_max_diff': source_metrics['max_abs'],
                'main_max_diff': main_metrics['max_abs'],
                'reproduction_command': 'python3 -m unittest '
                    'tests.test_v8_kokoro_runtime_stage0_join',
            }, sort_keys=True))

        args = (8, self.hello8['word_ids'],
                self.hello8['predictor_style'].ravel(),
                self.hello8['decoder_style'].ravel(),
                self.hello8['generator_harmonic_gaussian'])
        bad = args[-1].copy()
        bad.reshape(-1)[0] = np.nan
        status, lengths = self.execute(arena, *args[:-1], bad)
        self.assertNotEqual(status, 0)
        self.assertEqual(lengths, [-1] * 7)
        self.assertTrue(np.all(self.view(arena, 'generator_stage0_join',
                                          (256, 2560)) == -91.))
        status, lengths = self.execute(arena, *args)
        self.assertEqual((status, lengths[4]), (0, 1080))
        np.testing.assert_array_equal(
            self.view(arena, 'generator_stage0_join', (256, 2560)), first8)


if __name__ == '__main__':
    unittest.main()
