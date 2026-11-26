"""Checked variable-length decoder to first generator upsample through generated C."""

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
from build_kokoro_runtime_generator_stage0_circuit import (
    OUTPUT_CAPACITY, UPSAMPLED_FRAMES, build_circuit)


class KokoroRuntimeGeneratorStage0Test(unittest.TestCase):
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
        cls.reference36, cls.reference_meta = verified_fixture('generator_stage0')
        if cls.hello_meta['voice_row_index'] != 5:
            raise RuntimeError('eight-ID capture selected the wrong voice row')
        for meta in (encode_meta, *decode_meta.values(), cls.hello_meta,
                     cls.reference_meta):
            require_matching_reference_identity(direct_meta, meta)
        for name, value in cls.reference36.items():
            if name.startswith('waveform_decoder.'):
                weights[name] = value
        cls.fixture = prepare_duration_fixture(cls.root, circuit, weights)
        cls.fixture.source['config']['activation_buffer_dtypes'].update({
            'request_length': 'i32', 'valid_length': 'i32',
            'valid_residual_elements': 'i32'})
        cls.fixture.source['config']['context_length'] = 64
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(cls.root, cls.fixture.source, circuit))
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + [
            ctypes.POINTER(ctypes.c_int32)] * 5
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

    def execute(self, arena, count, ids, predictor_style, decoder_style):
        words = self.view(arena, 'word_ids', (64,), np.int32)
        words[:] = 178
        words[:min(count, 64, len(ids))] = ids[:min(count, 64, len(ids))]
        self.view(arena, 'request_length', (1,), np.int32)[0] = count
        self.view(arena, 'predictor_style', (128,))[:] = predictor_style
        self.view(arena, 'decoder_style', (128,))[:] = decoder_style
        output = self.view(arena, 'generator_stage0_upsample',
                           (256, OUTPUT_CAPACITY))
        output[:] = -91.0
        extents = [ctypes.c_int32(-1) for _ in range(5)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(value) for value in extents))
        return status, [value.value for value in extents], output

    def test_two_lengths_and_repeated_request_against_full_model(self):
        self.assertEqual(self.calls['errors'], [])
        runtime = xray_support.xray_builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_runtime_generator_stage0_bounded')
        self.assertEqual(runtime['sha256'],
                         hashlib.sha256(self.library.read_bytes()).hexdigest())
        cases = ((8, self.hello8, 54, 108, 1080, 5),
                 (36, self.reference36, 103, 206, 2060, 33))
        arena = self.arena()
        first8 = None
        for count, reference, frames, doubled, generated, row in (
                cases[0], cases[1], cases[0]):
            with self.subTest(ids=count):
                status, extents, output = self.execute(
                    arena, count, reference['word_ids'],
                    reference['predictor_style'].ravel(),
                    reference['decoder_style'].ravel())
                self.assertEqual(status, 0)
                self.assertEqual(extents,
                                 [count, count * 768, frames, doubled, generated])
                expected = reference['decoder_generator_ups_0'][0]
                observed = checkpoint_metrics(output[:, :generated], expected)
                self.assertLessEqual(observed['max_abs'], 5e-5, observed)
                self.assertTrue(np.all(output[:, generated:] == -91.0))
                if count == 8:
                    if first8 is None:
                        first8 = output.copy()
                    else:
                        np.testing.assert_array_equal(output, first8)
                        continue
                print('CKE_NUMERICAL_CASE ' + json.dumps({
                    'case_id': f'kokoro.runtime-generator-stage0.{count}',
                    'name': 'runtime generated first upsample versus direct full-model capture',
                    'provider': 'generated_kokoro_runtime_generator_stage0',
                    'dtype': 'fp32', 'direction': 'inference',
                    'oracle': 'pinned PyTorch 2.8 direct full-model capture',
                    'status': 'pass', 'gate': 'composition', 'blocking': True,
                    'configuration': f'{count} IDs; af_heart row {row}; '
                        f'A={frames}; 2A={doubled}; 20A={generated}',
                    'max_diff': observed['max_abs'], 'rmse': observed['rmse'],
                    'worst_index': observed['worst_index'], 'tolerance': 5e-5,
                    'reproduction_command': 'python3 -m unittest '
                        'tests.test_v8_kokoro_runtime_generator_stage0',
                }, sort_keys=True))

    def test_failed_input_does_not_publish_generator_length_then_recovers(self):
        arena = self.arena()
        reference = self.hello8
        args = (8, reference['word_ids'], reference['predictor_style'].ravel(),
                reference['decoder_style'].ravel())
        status, extents, output = self.execute(arena, *args)
        self.assertEqual(status, 0)
        self.assertEqual(extents[-1], 1080)
        baseline = output.copy()
        bad = args[-1].copy()
        bad[0] = np.nan
        status, extents, output = self.execute(arena, 8, args[1], args[2], bad)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1] * 5)
        self.assertTrue(np.all(output == -91.0))
        status, extents, output = self.execute(arena, *args)
        self.assertEqual((status, extents[-1]), (0, 1080))
        np.testing.assert_array_equal(output, baseline)

    def test_generator_capacity_rejects_before_output_write(self):
        arena = self.arena()
        reference = self.hello8
        args = (8, reference['word_ids'], reference['predictor_style'].ravel(),
                reference['decoder_style'].ravel())
        weight_entry = self.weights['duration_prosody.duration_head.weight']
        bias_entry = self.weights['duration_prosody.duration_head.bias']
        weight = np.ndarray((weight_entry['size'] // 4,), np.float32,
            buffer=arena, offset=weight_entry['abs_offset'])
        bias = np.ndarray((bias_entry['size'] // 4,), np.float32,
            buffer=arena, offset=bias_entry['abs_offset'])
        original_weight, original_bias = weight.copy(), bias.copy()
        # Fifty zero logits predict 25 frames per token: A=200 fits the
        # upstream 3,200-frame arena, but 20A exceeds this stage's 2,560.
        weight[:] = 0
        bias[:] = 0
        status, extents, output = self.execute(arena, *args)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1] * 5)
        self.assertTrue(np.all(output == -91.0))
        weight[:], bias[:] = original_weight, original_bias
        status, extents, output = self.execute(arena, *args)
        self.assertEqual((status, extents[-1]), (0, 1080))
        self.assertLessEqual(checkpoint_metrics(output[:, :1080],
            reference['decoder_generator_ups_0'][0])['max_abs'], 5e-5)


if __name__ == '__main__':
    unittest.main()
