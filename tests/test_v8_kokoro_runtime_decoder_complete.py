"""Complete variable-length Kokoro decoder through one generated native entry."""

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
    checkpoint_metrics, decoder_checkpoint_tolerance,
    require_matching_reference_identity)
from tests.test_v8_kokoro_generated_prosody_complete import verified_fixture
from tests import test_v8_kokoro_generated_albert_layer as xray_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from build_kokoro_runtime_decoder_complete_circuit import (
    build_circuit, FRAMES, UPSAMPLED_FRAMES)

TOKEN_CAPACITY = 64
PAD = -91.0


class KokoroRuntimeDecoderCompleteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        graph = build_circuit()
        circuit = cls.root / 'circuit.json'
        circuit.write_text(json.dumps(graph))
        weights, _, direct_meta, encode, encode_meta, decode, decode_meta = (
            decoder_weights_and_references())
        cls.direct36 = decode
        cls.encode36 = encode
        cls.hello8, cls.hello_meta = verified_fixture('runtime_decoder_hello_8')
        if (cls.hello_meta['voice_row_index'] !=
                len(cls.hello8['word_ids']) - 3):
            raise RuntimeError('eight-ID voice row does not match pinned selection')
        if np.array_equal(cls.hello8['decoder_style'].ravel(),
                          cls.encode36['decoder_style'].ravel()):
            raise RuntimeError('eight- and 36-ID voice styles must differ')
        for meta in (encode_meta, *decode_meta.values(), cls.hello_meta):
            require_matching_reference_identity(direct_meta, meta)
        cls.fixture = prepare_duration_fixture(cls.root, circuit, weights)
        if cls.fixture.duration_meta['voice_row_index'] != 33:
            raise RuntimeError('36-ID voice row does not match pinned selection')
        cls.fixture.source['config']['activation_buffer_dtypes'].update({
            'request_length': 'i32', 'valid_length': 'i32',
            'valid_residual_elements': 'i32'})
        cls.fixture.source['config']['context_length'] = TOKEN_CAPACITY
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(cls.root, cls.fixture.source, circuit))
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32)]
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
        self.view(arena, 'type_ids', (TOKEN_CAPACITY,), np.int32)[:] = 0
        return arena

    def execute(self, arena, count, ids, predictor_style, decoder_style):
        words = self.view(arena, 'word_ids', (TOKEN_CAPACITY,), np.int32)
        words[:] = 178
        words[:min(count, TOKEN_CAPACITY, len(ids))] = (
            ids[:min(count, TOKEN_CAPACITY, len(ids))])
        self.view(arena, 'request_length', (1,), np.int32)[0] = count
        self.view(arena, 'predictor_style', (128,))[:] = predictor_style
        self.view(arena, 'decoder_style', (128,))[:] = decoder_style
        for index in range(4):
            capacity = UPSAMPLED_FRAMES if index == 3 else FRAMES
            self.view(arena, f'decoder_decode{index}_output',
                      (512 if index == 3 else 1024, capacity))[:] = PAD
        extents = [ctypes.c_int32(-1) for _ in range(4)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(value) for value in extents))
        return status, [value.value for value in extents]

    def test_direct_oracle_parity_at_two_phoneme_lengths(self):
        self.assertEqual(self.calls['errors'], [])
        for index in range(4):
            name = f'decoder_decode{index}_output'
            self.assertTrue(any(op.get('template_op_id') == name
                                for op in self.calls['operations']), name)
        point = next(point for operation in self.calls['operations']
            for point in operation.get('semantic_checkpoints', [])
            if point['tensor'] == 'decoder_decode3_output')
        self.assertEqual(point['resolved_contract_id'],
                         'audio_scaled_sum_strided_sum_then_scale_fp32')
        runtime = xray_support.xray_builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_runtime_decoder_complete_bounded')
        self.assertEqual(runtime['sha256'],
            hashlib.sha256(self.library.read_bytes()).hexdigest())
        arena = self.arena()
        cases = (
            (8, self.hello8['word_ids'],
             self.hello8['predictor_style'].ravel(),
             self.hello8['decoder_style'].ravel(),
             {index: self.hello8[f'decoder_decode_{index}']
              for index in range(4)}, 54, 108, 5),
            (36, self.fixture.encoder['word_ids'],
             self.fixture.duration['predictor_style'],
             self.encode36['decoder_style'].ravel(),
             {index: self.direct36[index][f'decoder_decode_{index}']
              for index in range(4)}, 103, 206, 33),
        )
        first8 = None
        for request_index, (count, ids, predictor, decoder, reference,
                            frames, doubled, row) in enumerate(
                                (cases[0], cases[1], cases[0])):
            with self.subTest(tokens=count, voice_row=row):
                status, extents = self.execute(
                    arena, count, ids, predictor, decoder)
                self.assertEqual(status, 0)
                self.assertEqual(extents, [count, count * 768, frames, doubled])
                metrics = {}
                for index in range(4):
                    name = f'decoder_decode{index}_output'
                    channels = 512 if index == 3 else 1024
                    capacity = UPSAMPLED_FRAMES if index == 3 else FRAMES
                    valid = doubled if index == 3 else frames
                    actual = self.view(arena, name, (channels, capacity))
                    expected = reference[index][0]
                    self.assertEqual(expected.shape, (channels, valid))
                    observed = checkpoint_metrics(actual[:, :valid], expected)
                    metrics[name] = observed
                    self.assertTrue(np.all(actual[:, valid:] == PAD), name)
                    self.assertLessEqual(observed['max_abs'],
                        decoder_checkpoint_tolerance(name),
                        (count, name, observed))
                if count == 8:
                    output = self.view(arena, 'decoder_decode3_output',
                                       (512, UPSAMPLED_FRAMES))[:, :doubled]
                    if first8 is None:
                        first8 = output.copy()
                    else:
                        np.testing.assert_array_equal(output, first8)
                if request_index == 2:
                    continue
                print('CKE_NUMERICAL_CASE ' + json.dumps({
                    'case_id': f'kokoro.runtime-decoder-complete.{count}',
                    'name': 'runtime decoder versus direct pinned full-model capture',
                    'provider': 'generated_kokoro_runtime_decoder_complete',
                    'dtype': 'fp32', 'direction': 'inference',
                    'oracle': 'pinned PyTorch 2.8 direct full-model capture',
                    'status': 'pass', 'gate': 'composition', 'blocking': True,
                    'configuration': f'{count} IDs; af_heart row {row}; '
                        f'A={frames}/{FRAMES}; 2A={doubled}/{UPSAMPLED_FRAMES}',
                    'max_diff': max(m['max_abs'] for m in metrics.values()),
                    'tolerance': max(decoder_checkpoint_tolerance(name)
                                     for name in metrics),
                    'checkpoint_metrics': metrics,
                    'checkpoint_tolerances': {
                        name: decoder_checkpoint_tolerance(name)
                        for name in metrics},
                    'reproduction_command': 'python3 -m unittest '
                        'tests.test_v8_kokoro_runtime_decoder_complete',
                }, sort_keys=True))

    def test_rejection_does_not_publish_output_and_recovers(self):
        arena = self.arena()
        hello = self.hello8
        args = (8, hello['word_ids'], hello['predictor_style'].ravel(),
                hello['decoder_style'].ravel())
        self.assertEqual(self.execute(arena, *args)[0], 0)
        expected = self.view(arena, 'decoder_decode3_output',
                             (512, UPSAMPLED_FRAMES)).copy()
        bad = hello['decoder_style'].ravel().copy()
        bad[0] = np.nan
        status, extents = self.execute(arena, 8, args[1], args[2], bad)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1] * 4)
        self.assertTrue(np.all(self.view(arena, 'decoder_decode3_output',
                            (512, UPSAMPLED_FRAMES)) == PAD))
        self.assertEqual(self.execute(arena, *args)[0], 0)
        np.testing.assert_array_equal(
            self.view(arena, 'decoder_decode3_output',
                      (512, UPSAMPLED_FRAMES)), expected)
        for invalid in (0, TOKEN_CAPACITY + 1):
            with self.subTest(invalid_tokens=invalid):
                status, extents = self.execute(
                    arena, invalid, args[1], args[2], args[3])
                self.assertNotEqual(status, 0)
                self.assertEqual(extents, [-1] * 4)
                self.assertTrue(np.all(self.view(arena, 'decoder_decode3_output',
                                    (512, UPSAMPLED_FRAMES)) == PAD))
                self.assertEqual(self.execute(arena, *args)[0], 0)
                np.testing.assert_array_equal(
                    self.view(arena, 'decoder_decode3_output',
                              (512, UPSAMPLED_FRAMES)), expected)

    def test_wrong_decoder_weight_changes_output(self):
        arena = self.arena()
        item = self.weights['waveform_decoder.decode.3.conv2.weight']
        np.ndarray((item['size'] // 4,), np.float32, buffer=arena,
                   offset=item['abs_offset'])[:] = 0
        hello = self.hello8
        status, extents = self.execute(arena, 8, hello['word_ids'],
            hello['predictor_style'].ravel(), hello['decoder_style'].ravel())
        self.assertEqual(status, 0)
        self.assertEqual(extents[2:], [54, 108])
        actual = self.view(arena, 'decoder_decode3_output',
                           (512, UPSAMPLED_FRAMES))[:, :108]
        self.assertGreater(float(np.max(np.abs(
            actual - hello['decoder_decode_3'][0]))), 1e-3)


if __name__ == '__main__':
    unittest.main()
