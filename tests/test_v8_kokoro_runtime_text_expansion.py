"""Connected generated Kokoro text encoder and two checked frame streams."""
import ctypes
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph, build_ir_v8
from tests.v8_kokoro_duration_fixture_support import prepare_duration_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from build_kokoro_runtime_text_expansion_circuit import build_circuit

CAPACITY = 64
FRAME_CAPACITY = CAPACITY * 50


class KokoroRuntimeTextExpansionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.circuit = root / 'circuit.json'
        graph = build_circuit()
        cls.circuit.write_text(json.dumps(graph))
        base = ROOT / 'tests/fixtures/tts'
        path = base / 'kokoro_text_encoder_pinned.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['fixture_sha256']:
            raise RuntimeError('text encoder fixture SHA-256 mismatch')
        cls.text = dict(np.load(path))
        embedding = dict(np.load(base / 'kokoro_text_embedding_pinned.npz'))
        weights = {'acoustic_text_encoder.embedding.weight': embedding['table']}
        for index in range(3):
            for kind in ('weight', 'bias'):
                weights[f'acoustic_text_encoder.conv{index}.{kind}'] = (
                    cls.text[f'conv{index}_{kind}'])
            weights[f'acoustic_text_encoder.norm{index}.weight'] = (
                cls.text[f'norm{index}_gamma'])
            weights[f'acoustic_text_encoder.norm{index}.bias'] = (
                cls.text[f'norm{index}_beta'])
        for kind in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
            weights[f'acoustic_text_encoder.lstm.{kind}'] = cls.text[f'lstm_{kind}']
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
        cls.buffers = {entry['name']: entry for entry in
                       cls.layout['memory']['activations']['buffers']}
        cls.weights = {entry['name']: entry for entry in
                       cls.layout['memory']['weights']['entries']}
        cls.ids = {36: cls.fixture.encoder['word_ids']}
        for count in (2, 8, 64):
            cls.ids[count] = dict(np.load(
                base / f'kokoro_encoder_{count}_pinned.npz'))['word_ids']

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
        self.view(arena, 'type_ids', (CAPACITY,), np.int32)[:] = 0
        self.view(arena, 'predictor_style', (128,))[:] = (
            self.fixture.duration['predictor_style'])
        return arena

    def execute(self, arena, count):
        ids = self.view(arena, 'word_ids', (CAPACITY,), np.int32)
        ids[:] = 178  # Invalid reserved IDs detect accidental padding reads.
        ids[:count] = self.ids[count]
        self.view(arena, 'request_length', (1,), np.int32)[0] = count
        duration = self.view(arena, 'duration_expanded', (640, FRAME_CAPACITY))
        text = self.view(arena, 'text_expanded', (512, FRAME_CAPACITY))
        features = self.view(arena, 'text_encoder_features', (CAPACITY, 512))
        durations = self.view(arena, 'runtime_values', (CAPACITY,), np.int32)
        duration[:] = -91
        text[:] = -92
        features[:] = -93
        durations[:] = -999
        results = [ctypes.c_int32(-1) for _ in range(3)]
        status = self.fn(arena, len(arena), *(ctypes.byref(x) for x in results))
        self.assertEqual(status, 0)
        frames = results[2].value
        self.assertEqual((results[0].value, results[1].value),
                         (count, count * 768))
        self.assertGreater(frames, 0)
        self.assertLessEqual(frames, FRAME_CAPACITY)
        self.assertEqual(int(durations[:count].sum(dtype=np.int64)), frames)
        self.assertTrue(np.isfinite(duration[:, :frames]).all())
        self.assertTrue(np.isfinite(text[:, :frames]).all())
        self.assertTrue(np.all(duration[:, frames:] == -91))
        self.assertTrue(np.all(text[:, frames:] == -92))
        self.assertTrue(np.all(features[count:] == -93))
        self.assertTrue(np.all(durations[count:] == -999))
        return (frames, duration[:, :frames].copy(), text[:, :frames].copy(),
                features[:count].copy(), durations[:count].copy())

    def test_joined_streams_match_pinned_reference_and_change_length(self):
        self.assertEqual(self.calls['errors'], [])
        self.assertEqual(len(self.calls['operations']), 180)
        text_ops = {'text_embedding', 'text_lstm',
                    'expand_generated_duration', 'expand_generated_text_encoder'}
        for op in self.calls['operations']:
            if op.get('template_op_id') in text_ops:
                self.assertIn('valid_tokens', op['consumes_runtime_lengths'])
                self.assertNotIn('unresolved', [
                    point.get('resolved_contract_id') for point in
                    op.get('semantic_checkpoints', [])])
            if op.get('template_op_id', '').startswith('expand_'):
                self.assertIn('expanded_frames', op['consumes_runtime_lengths'])
        arena = self.arena()
        first_short = None
        for count in (2, 8, 36, 64, 2):
            with self.subTest(count=count):
                result = self.execute(arena, count)
                if count == 36:
                    frames, duration, text, features, durations = result
                    self.assertEqual(frames, 103)
                    for index in range(3):
                        conv = self.view(arena, f'text_conv{index}',
                                         (512, CAPACITY))[:, :count]
                        norm = self.view(arena, f'text_norm{index}',
                                         (CAPACITY, 512))[:count]
                        activation = self.view(arena, f'text_act{index}',
                                               (CAPACITY, 512))[:count]
                        for actual, key, ceiling in (
                            (conv, f'conv{index}_output', (1.3e-5, 5e-6, 9e-6)[index]),
                            (norm.T, f'norm{index}_output', (8e-6, 7e-6, 6e-6)[index]),
                            (activation.T, f'act{index}_output', (8e-6, 7e-6, 6e-6)[index]),
                        ):
                            self.assertTrue(np.isfinite(actual).all(), key)
                            np.testing.assert_allclose(actual, self.text[key],
                                                       rtol=0, atol=ceiling)
                    np.testing.assert_array_equal(
                        durations, self.fixture.duration['durations'])
                    np.testing.assert_allclose(
                        features, self.text['lstm_output_token_major'],
                        rtol=0, atol=3e-6)
                    np.testing.assert_allclose(
                        text, self.text['text_expanded'], rtol=0, atol=3e-6)
                    expected_duration = np.repeat(
                        self.fixture.duration['predictor_output'].T,
                        durations, axis=1)
                    np.testing.assert_allclose(
                        duration, expected_duration, rtol=0, atol=3e-5)
                    maximum = float(np.max(np.abs(text - self.text['text_expanded'])))
                    print('CKE_NUMERICAL_CASE ' + json.dumps({
                        'case_id': 'kokoro.runtime-text-expansion.36.pytorch28',
                        'name': 'Kokoro generated acoustic text and two frame streams',
                        'provider': 'generated_kokoro_runtime_text_expansion',
                        'dtype': 'fp32', 'direction': 'inference',
                        'oracle': 'pinned independent PyTorch 2.8 text capture',
                        'status': 'pass', 'gate': 'composition', 'blocking': True,
                        'max_diff': maximum, 'tolerance': 3e-6,
                        'configuration': 'capacity=64 tokens, valid=36, frames=103',
                        'reproduction_command': 'python3 -m unittest '
                            'tests.test_v8_kokoro_runtime_text_expansion',
                    }, sort_keys=True))
                    duration_error = float(np.max(np.abs(duration - expected_duration)))
                    print('CKE_NUMERICAL_CASE ' + json.dumps({
                        'case_id': 'kokoro.runtime-duration-expansion.36.pytorch28',
                        'name': 'Kokoro generated duration feature expansion',
                        'provider': 'audio_duration_expand_token_major_f32',
                        'dtype': 'fp32', 'direction': 'inference',
                        'oracle': 'pinned PyTorch predictor features with independent NumPy repeat',
                        'status': 'pass', 'gate': 'composition', 'blocking': True,
                        'max_diff': duration_error, 'tolerance': 3e-5,
                        'configuration': 'capacity=64 tokens, valid=36, frames=103',
                        'reproduction_command': 'python3 -m unittest '
                            'tests.test_v8_kokoro_runtime_text_expansion',
                    }, sort_keys=True))
                elif count == 2 and first_short is not None:
                    for actual, previous in zip(result, first_short):
                        np.testing.assert_array_equal(actual, previous)
                elif count == 2:
                    first_short = result

    def test_failure_stops_both_expansions_and_recovers(self):
        arena = self.arena()
        baseline = self.execute(arena, 8)
        style = self.view(arena, 'predictor_style', (128,))
        old = float(style[0])
        style[0] = np.nan
        self.view(arena, 'request_length', (1,), np.int32)[0] = 8
        duration = self.view(arena, 'duration_expanded', (640, FRAME_CAPACITY))
        text = self.view(arena, 'text_expanded', (512, FRAME_CAPACITY))
        duration[:] = -91
        text[:] = -92
        results = [ctypes.c_int32(-1) for _ in range(3)]
        self.assertNotEqual(self.fn(arena, len(arena),
            *(ctypes.byref(x) for x in results)), 0)
        self.assertEqual([x.value for x in results], [-1, -1, -1])
        self.assertTrue(np.all(duration == -91))
        self.assertTrue(np.all(text == -92))
        style[0] = old
        recovered = self.execute(arena, 8)
        for actual, previous in zip(recovered, baseline):
            np.testing.assert_array_equal(actual, previous)

    def test_invalid_lengths_and_arena_capacity_reject_before_expansion(self):
        arena = self.arena()
        self.execute(arena, 8)
        duration = self.view(arena, 'duration_expanded', (640, FRAME_CAPACITY))
        text = self.view(arena, 'text_expanded', (512, FRAME_CAPACITY))
        for bad in (0, -1, 65, 2147483647):
            with self.subTest(length=bad):
                self.view(arena, 'request_length', (1,), np.int32)[0] = bad
                before_duration = duration.copy()
                before_text = text.copy()
                results = [ctypes.c_int32(-1) for _ in range(3)]
                self.assertNotEqual(self.fn(arena, len(arena),
                    *(ctypes.byref(x) for x in results)), 0)
                self.assertEqual([x.value for x in results], [-1, -1, -1])
                np.testing.assert_array_equal(duration, before_duration)
                np.testing.assert_array_equal(text, before_text)
        self.view(arena, 'request_length', (1,), np.int32)[0] = 8
        results = [ctypes.c_int32(-1) for _ in range(3)]
        self.assertNotEqual(self.fn(arena, len(arena) - 1,
            *(ctypes.byref(x) for x in results)), 0)
        self.assertEqual([x.value for x in results], [-1, -1, -1])
        self.execute(arena, 8)

    def test_wrong_text_feature_edge_fails_planned_stride_contract(self):
        graph = build_circuit()
        graph['block_types']['duration_expansion']['body']['ops'][1][
            'graph_slots']['inputs']['features'] = 'text_features'
        source = copy.deepcopy(self.fixture.source)
        source['template'] = graph
        registry = build_ir_v8.load_kernel_registry()
        with contextlib.redirect_stdout(io.StringIO()):
            ir1 = build_ir_v8.build_ir1_direct(source, None, mode='prefill')
            lower1 = build_ir_v8.generate_ir_lower_1(
                ir1, registry, source, 'prefill')
            layout = build_ir_v8.generate_memory_layout(
                lower1, source, registry, mode='prefill', context_len=CAPACITY)
            lower2 = build_ir_v8.generate_ir_lower_2(
                lower1, layout, source, registry, mode='prefill')
            with self.assertRaisesRegex(RuntimeError,
                                        'stride disagrees with planned layout'):
                build_ir_v8.generate_ir_lower_3(lower2, mode='prefill')


if __name__ == '__main__':
    unittest.main()
