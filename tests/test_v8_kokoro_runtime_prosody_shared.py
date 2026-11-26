"""One bounded generated entry joins runtime text/duration to prosody."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from build_kokoro_runtime_prosody_shared_circuit import build_circuit

CAPACITY = 64
FRAMES = CAPACITY * 50


class KokoroRuntimeProsodySharedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        graph = build_circuit()
        cls.circuit = root / 'circuit.json'
        cls.circuit.write_text(json.dumps(graph))
        base = ROOT / 'tests/fixtures/tts'
        text_path = base / 'kokoro_text_encoder_pinned.npz'
        text_meta = json.loads(text_path.with_suffix('.json').read_text())
        if hashlib.sha256(text_path.read_bytes()).hexdigest() != text_meta['fixture_sha256']:
            raise RuntimeError('text fixture hash mismatch')
        text = dict(np.load(text_path))
        embedding = dict(np.load(base / 'kokoro_text_embedding_pinned.npz'))
        prosody_path = base / 'kokoro_prosody_shared_pinned.npz'
        prosody_meta = json.loads(prosody_path.with_suffix('.json').read_text())
        if hashlib.sha256(prosody_path.read_bytes()).hexdigest() != prosody_meta['fixture_sha256']:
            raise RuntimeError('prosody fixture hash mismatch')
        if (prosody_meta['model_pin'] != text_meta['model_pin']['revision'] or
            text_meta['asset_sha256'] != cls.fixture_asset_hashes()):
            raise RuntimeError('text/prosody fixtures refer to different model assets')
        cls.prosody = dict(np.load(prosody_path))
        weights = {'acoustic_text_encoder.embedding.weight': embedding['table']}
        for index in range(3):
            for kind in ('weight', 'bias'):
                weights[f'acoustic_text_encoder.conv{index}.{kind}'] = text[f'conv{index}_{kind}']
            weights[f'acoustic_text_encoder.norm{index}.weight'] = text[f'norm{index}_gamma']
            weights[f'acoustic_text_encoder.norm{index}.bias'] = text[f'norm{index}_beta']
        for kind in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
            weights[f'acoustic_text_encoder.lstm.{kind}'] = text[f'lstm_{kind}']
            weights[f'duration_prosody.shared_scan.{kind}'] = cls.prosody[kind]
        cls.fixture = prepare_duration_fixture(root, cls.circuit, weights)
        cls.fixture.source['config']['activation_buffer_dtypes'].update({
            'request_length': 'i32', 'valid_length': 'i32',
            'valid_residual_elements': 'i32'})
        cls.fixture.source['config']['context_length'] = CAPACITY
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(root, cls.fixture.source, cls.circuit))
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32)]
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}
        cls.weights = {item['name']: item for item in
                       cls.layout['memory']['weights']['entries']}
        cls.ids = {36: cls.fixture.encoder['word_ids']}
        for count in (2, 8, 64):
            cls.ids[count] = dict(np.load(
                base / f'kokoro_encoder_{count}_pinned.npz'))['word_ids']

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def fixture_asset_hashes():
        path = ROOT / 'tests/fixtures/tts/kokoro_duration_predictor_pinned.json'
        return json.loads(path.read_text())['asset_hashes']

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
        self.view(arena, 'type_ids', (CAPACITY,), np.int32)[:] = 0
        self.view(arena, 'predictor_style', (128,))[:] = (
            self.fixture.duration['predictor_style'])
        return arena

    def execute(self, arena, count):
        ids = self.view(arena, 'word_ids', (CAPACITY,), np.int32)
        ids[:] = 178
        ids[:count] = self.ids[count]
        self.view(arena, 'request_length', (1,), np.int32)[0] = count
        output = self.view(arena, 'prosody_shared_features', (FRAMES, 512))
        transposed = self.view(arena, 'prosody_input_token', (FRAMES, 640))
        output[:] = -91
        transposed[:] = -92
        results = [ctypes.c_int32(-1) for _ in range(3)]
        status = self.fn(arena, len(arena), *(ctypes.byref(x) for x in results))
        return status, [x.value for x in results], output, transposed

    def test_dynamic_shared_scan_and_pinned_36_token_oracle(self):
        self.assertEqual(self.calls['errors'], [])
        for name in ('prosody_input_transpose', 'prosody_shared_scan'):
            op = next(item for item in self.calls['operations']
                      if item.get('template_op_id') == name)
            self.assertIn('expanded_frames', op['consumes_runtime_lengths'])
        arena = self.arena()
        short = None
        for count in (2, 8, 36, 64, 2):
            with self.subTest(count=count):
                status, extents, output, transposed = self.execute(arena, count)
                self.assertEqual(status, 0)
                frames = extents[2]
                self.assertEqual(extents[:2], [count, count * 768])
                self.assertGreater(frames, 0)
                self.assertLessEqual(frames, FRAMES)
                self.assertTrue(np.isfinite(output[:frames]).all())
                self.assertTrue(np.all(output[frames:] == -91))
                self.assertTrue(np.all(transposed[frames:] == -92))
                if count == 36:
                    self.assertEqual(frames, 103)
                    error = np.abs(output[:frames] - self.prosody['output'])
                    worst = np.unravel_index(np.argmax(error), error.shape)
                    self.assertLessEqual(float(error[worst]), 1e-5,
                                         (worst, float(error[worst])))
                    print('CKE_NUMERICAL_CASE ' + json.dumps({
                        'case_id': 'kokoro.runtime-prosody-shared.36.pytorch28',
                        'name': 'runtime phonemes through shared prosody scan',
                        'provider': 'generated_kokoro_runtime_prosody_shared',
                        'dtype': 'fp32', 'direction': 'inference',
                        'oracle': 'pinned independent PyTorch 2.8 full-model capture',
                        'status': 'pass', 'gate': 'composition', 'blocking': True,
                        'max_diff': float(error[worst]), 'tolerance': 1e-5,
                        'configuration': 'capacity=64 tokens, valid=36, frames=103',
                        'reproduction_command': 'python3 -m unittest '
                            'tests.test_v8_kokoro_runtime_prosody_shared',
                    }, sort_keys=True))
                if count == 2 and short is None:
                    short = (frames, output[:frames].copy())
                elif count == 2:
                    self.assertEqual(frames, short[0])
                    np.testing.assert_array_equal(output[:frames], short[1])

    def test_failed_producer_stops_scan_then_recovers(self):
        arena = self.arena()
        status, baseline_extent, baseline, _ = self.execute(arena, 8)
        self.assertEqual(status, 0)
        baseline = baseline[:baseline_extent[2]].copy()
        style = self.view(arena, 'predictor_style', (128,))
        original = float(style[0])
        style[0] = np.nan
        status, extents, output, transposed = self.execute(arena, 8)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1, -1, -1])
        self.assertTrue(np.all(output == -91))
        self.assertTrue(np.all(transposed == -92))
        style[0] = original
        status, extents, output, _ = self.execute(arena, 8)
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(output[:extents[2]], baseline)

    def test_bad_request_length_and_short_arena_do_not_touch_scan_output(self):
        arena = self.arena()
        self.assertEqual(self.execute(arena, 8)[0], 0)
        output = self.view(arena, 'prosody_shared_features', (FRAMES, 512))
        transposed = self.view(arena, 'prosody_input_token', (FRAMES, 640))
        for bad in (0, -1, CAPACITY + 1, 2147483647):
            with self.subTest(request_length=bad):
                self.view(arena, 'request_length', (1,), np.int32)[0] = bad
                output[:] = -91
                transposed[:] = -92
                extents = [ctypes.c_int32(-1) for _ in range(3)]
                self.assertNotEqual(self.fn(arena, len(arena),
                    *(ctypes.byref(x) for x in extents)), 0)
                self.assertEqual([x.value for x in extents], [-1, -1, -1])
                self.assertTrue(np.all(output == -91))
                self.assertTrue(np.all(transposed == -92))
        self.view(arena, 'request_length', (1,), np.int32)[0] = 8
        output[:] = -91
        transposed[:] = -92
        extents = [ctypes.c_int32(-1) for _ in range(3)]
        self.assertNotEqual(self.fn(arena, len(arena) - 1,
            *(ctypes.byref(x) for x in extents)), 0)
        self.assertEqual([x.value for x in extents], [-1, -1, -1])
        self.assertTrue(np.all(output == -91))
        self.assertTrue(np.all(transposed == -92))
        self.assertEqual(self.execute(arena, 8)[0], 0)

    def test_invalid_duration_logits_stop_before_transpose_and_recover(self):
        arena = self.arena()
        status, baseline_extent, baseline, _ = self.execute(arena, 36)
        self.assertEqual(status, 0)
        baseline = baseline[:baseline_extent[2]].copy()
        for name, value in (('duration_prosody.duration_head.weight', 0.),
                            ('duration_prosody.duration_head.bias', np.inf)):
            item = self.weights[name]
            np.ndarray((item['size'] // 4,), np.float32, buffer=arena,
                       offset=item['abs_offset'])[:] = value
        status, extents, output, transposed = self.execute(arena, 36)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1, -1, -1])
        self.assertTrue(np.all(output == -91))
        self.assertTrue(np.all(transposed == -92))
        for name in ('duration_prosody.duration_head.weight',
                     'duration_prosody.duration_head.bias'):
            item = self.weights[name]
            entry = self.fixture.entries[name]
            start = entry['file_offset']
            arena[item['abs_offset']:item['abs_offset'] + entry['size']] = (
                self.fixture.bump[start:start + entry['size']])
        status, extents, output, _ = self.execute(arena, 36)
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(output[:extents[2]], baseline)


if __name__ == '__main__':
    unittest.main()
